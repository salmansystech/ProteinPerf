"""Microbenchmarks: each Triton kernel against PyTorch eager, torch.compile and cuBLAS"""
from typing import Callable, Dict, List
import math
import torch
import torch.nn.functional as F
from triton.testing import do_bench

from proteinperf.kernels.gelu import gelu, gelu_reference
from proteinperf.kernels.matmul import linear
from proteinperf.kernels.rotary import rotary_embedding, rotary_embedding_reference
from proteinperf.models import ESM2_MODELS

def _time_ms(fn: Callable) -> float:
    """Median time with Triton's benchmarking helper, which flushes the L2 cache between runs"""
    return do_bench(fn, warmup=25, rep=100, return_mode="median")

def measured_bandwidth_gbs(size_mb: int = 512) -> float:
    """Practical peak memory bandwidth: a large device-to-device copy (reads + writes)"""
    x = torch.empty(size_mb * 2**20, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    ms = _time_ms(lambda: y.copy_(x))
    return 2 * x.numel() / (ms / 1000) / 1e9

def _max_diff(a, b) -> float:
    if isinstance(a, tuple):
        return max(_max_diff(x, y) for x, y in zip(a, b))
    return (a.float() - b.float()).abs().max().item()

def _bytes(*tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)

def benchmark_kernels(size: str, batch_size: int = 8, seq_len: int = 512, dtype=torch.float16) -> List[Dict]:
    """Time each kernel at the shapes it sees inside the given ESM-2 model"""
    spec = ESM2_MODELS[size]
    tokens = seq_len + 2
    M = batch_size * tokens
    torch.manual_seed(0)
    rows = []

    def record(kernel, reference, compiled, triton_fn, check_against, bytes_moved=None, flops=None):
        reference_ms = _time_ms(reference)
        compiled_ms = _time_ms(compiled)
        triton_ms = _time_ms(triton_fn)
        row = {
            "model": size, "kernel": kernel, "batch_size": batch_size, "seq_len": seq_len,
            "eager_ms": reference_ms, "compiled_ms": compiled_ms, "triton_ms": triton_ms,
            "speedup_vs_eager": reference_ms / triton_ms, "speedup_vs_compile": compiled_ms / triton_ms,
            "max_abs_diff": _max_diff(check_against(), triton_fn()),
        }
        if bytes_moved:
            row["triton_gbs"] = bytes_moved / (triton_ms / 1000) / 1e9
        if flops:
            row["triton_tflops"] = flops / (triton_ms / 1000) / 1e12
        rows.append(row)

    # Rotary embedding: q and k are views of a linear layer's output, as in the model
    heads, head_dim = spec.heads, spec.head_dim
    q = torch.randn(batch_size, tokens, spec.hidden_size, device="cuda", dtype=dtype).view(batch_size, tokens, heads, head_dim).transpose(1, 2)
    k = torch.randn_like(q)
    angles = torch.randn(batch_size, tokens, head_dim // 2, device="cuda")
    cos = torch.cat([angles.cos()] * 2, dim=-1).to(dtype)
    sin = torch.cat([angles.sin()] * 2, dim=-1).to(dtype)
    compiled_rotary = torch.compile(rotary_embedding_reference)
    record("rotary", lambda: rotary_embedding_reference(q, k, cos, sin), lambda: compiled_rotary(q, k, cos, sin),
           lambda: rotary_embedding(q, k, cos, sin), lambda: rotary_embedding_reference(q, k, cos, sin),
           bytes_moved=2 * _bytes(q, k) + _bytes(cos, sin))

    # GELU on the feed-forward activation
    x = torch.randn(M, spec.intermediate_size, device="cuda", dtype=dtype)
    compiled_gelu = torch.compile(gelu_reference)
    record("gelu", lambda: gelu_reference(x), lambda: compiled_gelu(x), lambda: gelu(x), lambda: gelu_reference(x),
           bytes_moved=2 * _bytes(x))

    # Feed-forward up-projection: plain matmul (eager F.linear runs cuBLAS), then with bias and GELU fused
    a = torch.randn(M, spec.hidden_size, device="cuda", dtype=dtype)
    w = torch.randn(spec.intermediate_size, spec.hidden_size, device="cuda", dtype=dtype) / math.sqrt(spec.hidden_size)
    bias = torch.randn(spec.intermediate_size, device="cuda", dtype=dtype) * 0.1
    flops = 2 * M * spec.intermediate_size * spec.hidden_size
    compiled_linear = torch.compile(F.linear)
    record("matmul", lambda: F.linear(a, w, bias), lambda: compiled_linear(a, w, bias), lambda: linear(a, w, bias),
           lambda: F.linear(a, w, bias), flops=flops)

    def linear_gelu_reference():
        return gelu_reference(F.linear(a, w, bias))
    compiled_linear_gelu = torch.compile(linear_gelu_reference)
    record("matmul+gelu", linear_gelu_reference, compiled_linear_gelu, lambda: linear(a, w, bias, gelu=True),
           linear_gelu_reference, flops=flops)
    return rows
