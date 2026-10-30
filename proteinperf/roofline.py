"""Roofline analysis: how close each ESM-2 operation gets to the GPU's compute and memory limits"""
from dataclasses import dataclass, asdict
from typing import List

from proteinperf.models import ModelSpec

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
