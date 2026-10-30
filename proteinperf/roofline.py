"""Roofline analysis: how close each ESM-2 operation gets to the GPU's compute and memory limits"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List
import torch

from proteinperf.models import ESM2_MODELS, ModelSpec

@dataclass
class LayerOp:
    """One operation inside a transformer layer, with the work it has to do"""
    name: str
    kind: str          # Matmul, Attention, LayerNorm or Elementwise
    flops: float
    bytes: float       # minimum bytes moved to and from GPU memory

    @property
    def intensity(self) -> float:
        """Arithmetic intensity: FLOPs per byte of memory traffic"""
        return self.flops / self.bytes

    def to_dict(self) -> dict:
        return {**asdict(self), "intensity": self.intensity}

def layer_ops(spec: ModelSpec, batch_size: int, seq_len: int, element_bytes: int = 2) -> List[LayerOp]:
    """FLOPs and minimum memory traffic of each operation in one ESM-2 layer.

    Bytes count each input read once and each output written once, the least any
    implementation can move. Attention is counted as fused (as the efficient and
    flash kernels are), so the sequence x sequence score matrix never reaches memory.
    """
    tokens = seq_len + 2                  # <cls> and <eos>
    n = batch_size * tokens               # rows of every activation matrix
    h, i = spec.hidden_size, spec.intermediate_size
    s = element_bytes

    def linear(name, k, m):
        # (n x k) @ (k x m) + bias
        return LayerOp(name, "Matmul", 2 * n * k * m, s * (n * k + k * m + m + n * m))

    return [
        LayerOp("layernorm (attention)", "LayerNorm", 8 * n * h, s * (2 * n * h + 2 * h)),
        linear("q, k, v projection", h, 3 * h),
        # Rotate q and k: x * cos + rotate_half(x) * sin, 3 FLOPs per element
        LayerOp("rotary embedding", "Elementwise", 3 * 2 * n * h, s * 2 * (2 * n * h)),
        # Q @ K^T and scores @ V: 2 FLOPs per multiply-add, over every pair of tokens
        LayerOp("attention", "Attention", 4 * batch_size * tokens * tokens * h, s * 4 * n * h),
        linear("output projection", h, h),
        LayerOp("residual add (attention)", "Elementwise", n * h, s * 3 * n * h),
        LayerOp("layernorm (feed-forward)", "LayerNorm", 8 * n * h, s * (2 * n * h + 2 * h)),
        linear("feed-forward up", h, i),
        # Exact (erf) GELU: about 8 FLOPs per element
        LayerOp("gelu", "Elementwise", 8 * n * i, s * 2 * n * i),
        linear("feed-forward down", i, h),
        LayerOp("residual add (feed-forward)", "Elementwise", n * h, s * 3 * n * h),
    ]

def model_flops(spec: ModelSpec, batch_size: int, seq_len: int) -> float:
    """FLOPs of one forward pass through all layers (embeddings are lookups, not FLOPs)"""
    return spec.layers * sum(op.flops for op in layer_ops(spec, batch_size, seq_len))

def attainable_tflops(intensity: float, peak_tflops: float, bandwidth_gbs: float) -> float:
    """The roofline: performance is capped by compute, or by bandwidth x intensity"""
    return min(peak_tflops, intensity * bandwidth_gbs / 1000)

def ridge_point(peak_tflops: float, bandwidth_gbs: float) -> float:
    """Intensity (FLOPs/byte) above which an operation can be compute-bound"""
    return peak_tflops * 1000 / bandwidth_gbs

def roofline_time_ms(op: LayerOp, peak_tflops: float, bandwidth_gbs: float) -> float:
    """Shortest possible time for an operation on this GPU"""
    return max(op.flops / (peak_tflops * 1e12), op.bytes / (bandwidth_gbs * 1e9)) * 1000

def measured_peak_tflops(dtype: torch.dtype = torch.float16, size: int = 4096) -> float:
    """Practical compute peak: a large square matrix multiply with cuBLAS"""
    from triton.testing import do_bench

    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    ms = do_bench(lambda: a @ b, warmup=25, rep=200, return_mode="median")
    return 2 * size ** 3 / (ms / 1000) / 1e12

def _op_functions(spec: ModelSpec, batch_size: int, seq_len: int, dtype: torch.dtype) -> Dict[str, Callable]:
    """Each layer operation as PyTorch runs it in eager mode, on tensors of the model's shapes"""
    import torch.nn.functional as F
    from proteinperf.kernels.gelu import gelu_reference
    from proteinperf.kernels.rotary import rotary_embedding_reference

    tokens = seq_len + 2
    h, i, heads = spec.hidden_size, spec.intermediate_size, spec.heads
    device = "cuda"

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    x, residual = randn(batch_size, tokens, h), randn(batch_size, tokens, h)
    up = randn(batch_size, tokens, i)
    norm_weight, norm_bias = randn(h), randn(h)
    w_qkv, b_qkv = randn(3 * h, h) / h ** 0.5, randn(3 * h)
    w_out, b_out = randn(h, h) / h ** 0.5, randn(h)
    w_up, b_up = randn(i, h) / h ** 0.5, randn(i)
    w_down, b_down = randn(h, i) / i ** 0.5, randn(h)
    q, k, v = (randn(batch_size, tokens, h).view(batch_size, tokens, heads, -1).transpose(1, 2) for _ in range(3))
    angles = torch.randn(batch_size, tokens, spec.head_dim // 2, device=device)
    cos = torch.cat([angles.cos()] * 2, dim=-1).to(dtype)
    sin = torch.cat([angles.sin()] * 2, dim=-1).to(dtype)

    layernorm = lambda: F.layer_norm(x, (h,), norm_weight, norm_bias)
    return {
        "layernorm (attention)": layernorm,
        "q, k, v projection": lambda: F.linear(x, w_qkv, b_qkv),
        "rotary embedding": lambda: rotary_embedding_reference(q, k, cos, sin),
        "attention": lambda: F.scaled_dot_product_attention(q, k, v),
        "output projection": lambda: F.linear(x, w_out, b_out),
        "residual add (attention)": lambda: x + residual,
        "layernorm (feed-forward)": layernorm,
        "feed-forward up": lambda: F.linear(x, w_up, b_up),
        "gelu": lambda: gelu_reference(up),
        "feed-forward down": lambda: F.linear(up, w_down, b_down),
        "residual add (feed-forward)": lambda: x + residual,
    }

def measure_layer(size: str, batch_size: int, seq_len: int, dtype: torch.dtype,
                  peak_tflops: float, bandwidth_gbs: float) -> List[dict]:
    """Time each layer operation and place it on the roofline"""
    from triton.testing import do_bench

    spec = ESM2_MODELS[size]
    functions = _op_functions(spec, batch_size, seq_len, dtype)
    rows = []
    for op in layer_ops(spec, batch_size, seq_len, torch.finfo(dtype).bits // 8):
        ms = do_bench(functions[op.name], warmup=25, rep=100, return_mode="median")
        achieved = op.flops / (ms / 1000) / 1e12
        attainable = attainable_tflops(op.intensity, peak_tflops, bandwidth_gbs)
        rows.append({
            "model": size, "batch_size": batch_size, "seq_len": seq_len, **op.to_dict(),
            "bound": "compute" if op.intensity >= ridge_point(peak_tflops, bandwidth_gbs) else "memory",
            "measured_ms": ms,
            "roofline_ms": roofline_time_ms(op, peak_tflops, bandwidth_gbs),
            "achieved_tflops": achieved,
            "achieved_gbs": op.bytes / (ms / 1000) / 1e9,
            "attainable_tflops": attainable,
            "efficiency": achieved / attainable,
        })
    return rows
