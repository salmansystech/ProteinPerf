"""Tiled matrix multiplication in Triton, with an optional fused bias + GELU epilogue.

ESM-2's feed-forward block computes gelu(x @ W1.T + b1). PyTorch runs a cuBLAS matmul that
writes the (tokens × 4·hidden) activation to GPU memory, then GELU kernels that read it back.
Fusing the bias and GELU into the matmul's epilogue applies them while the result is still
in registers, so the activation is written only once.

The kernel follows the standard Triton design:
- each program computes one BLOCK_M × BLOCK_N tile of the output, accumulating in FP32
  over BLOCK_K slices of the shared dimension with tl.dot (tensor cores for FP16/BF16);
- programs are ordered in groups of GROUP_M rows so neighbouring programs reuse the same
  input tiles from the L2 cache;
- tile sizes, pipeline stages and warps are chosen per shape by the autotuner.
"""
from typing import Optional
import torch
import triton
import triton.language as tl

def _configs():
    shapes = [
        (128, 256, 64, 3, 8),
        (128, 128, 64, 3, 4),
        (128, 128, 32, 4, 4),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (64, 256, 32, 4, 4),
        (64, 64, 32, 5, 2),
        (256, 128, 32, 3, 8),
    ]
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": 8}, num_stages=s, num_warps=w)
            for m, n, k, s, w in shapes]

@triton.autotune(configs=_configs(), key=["M", "N", "K", "HAS_BIAS", "APPLY_GELU"])
@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    HAS_BIAS: tl.constexpr, APPLY_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Grouped ordering: walk GROUP_M row-tiles down a column before moving right,
    # so programs running together share input tiles in L2
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # Wrap out-of-range rows/columns to valid addresses; they're masked out when storing
    rows = offs_m % M
    cols = offs_n % N
    x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is (N, K) as in nn.Linear; read it transposed as a (K, N) tile
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + cols[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Epilogue: applied in FP32 registers before the single write to memory
    if HAS_BIAS:
        acc += tl.load(bias_ptr + cols).to(tl.float32)[None, :]
    if APPLY_GELU:
        acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, gelu: bool = False) -> torch.Tensor:
    """x @ weight.T (+ bias), optionally followed by exact GELU, like nn.Linear then gelu"""
    if not x.is_cuda:
        raise RuntimeError("The Triton matmul kernel needs CUDA tensors")
    leading = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    M, K = x2d.shape
    N, k_weight = weight.shape
    if K != k_weight:
        raise ValueError(f"Shapes don't match: x has {K} features, weight expects {k_weight}")
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _linear_kernel[grid](
        x2d, weight, bias if bias is not None else weight, out,
        M, N, K,
        x2d.stride(0), x2d.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None, APPLY_GELU=gelu,
    )
    return out.reshape(*leading, N)
