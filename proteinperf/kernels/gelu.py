"""Fused exact GELU in one Triton kernel.

ESM-2 computes GELU as x * 0.5 * (1 + erf(x / sqrt(2))): five elementwise kernels that each
read and write the whole feed-forward activation (batch × length × 4 × hidden). This kernel
reads it once and writes it once, computing in FP32.
"""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)

def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf-based) GELU, the variant ESM-2 was trained with"""
    if not x.is_cuda:
        raise RuntimeError("The Triton GELU kernel needs a CUDA tensor")
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    block = 1024
    _gelu_kernel[(triton.cdiv(n, block),)](x, out, n, BLOCK=block)
    return out

def gelu_reference(x: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation, identical to Hugging Face's ESM gelu()"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
