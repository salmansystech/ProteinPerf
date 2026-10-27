"""Fused rotary position embedding for queries and keys in one Triton kernel.

Hugging Face's ESM-2 applies rotary embeddings as
    q.float() * cos + rotate_half(q.float()) * sin, then casts back to the input dtype,
and the same for k. In FP16 that's about 16 kernels per layer (casts, negate, concatenate,
multiplies, adds) that each read and write whole tensors. This kernel reads q, k, cos and sin
once, does the math in FP32 registers, and writes q and k once.
"""
from typing import Tuple
import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    n_rows, heads, seq_len, half,
    sq_b, sq_h, sq_l, sq_d,
    sk_b, sk_h, sk_l, sk_d,
    sc_b, sc_l, sc_d,
    so_b, so_h, so_l, so_d,
    BLOCK_ROWS: tl.constexpr, BLOCK_HALF: tl.constexpr,
):
    # Each program rotates BLOCK_ROWS (batch, head, position) rows of q and k
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_HALF)
    row_mask = rows < n_rows
    mask = row_mask[:, None] & (cols[None, :] < half)

    b = rows // (heads * seq_len)
    h = (rows // seq_len) % heads
    l = rows % seq_len

    # cos and sin are shared by all heads: (batch, position, head_dim)
    c_base = (b * sc_b + l * sc_l)[:, None]
    cos1 = tl.load(cos_ptr + c_base + cols[None, :] * sc_d, mask=mask).to(tl.float32)
    cos2 = tl.load(cos_ptr + c_base + (cols[None, :] + half) * sc_d, mask=mask).to(tl.float32)
    sin1 = tl.load(sin_ptr + c_base + cols[None, :] * sc_d, mask=mask).to(tl.float32)
    sin2 = tl.load(sin_ptr + c_base + (cols[None, :] + half) * sc_d, mask=mask).to(tl.float32)

    o_base = (b * so_b + h * so_h + l * so_l)[:, None]

    # rotate_half(x) = cat(-x2, x1), so the first half gets x1*cos - x2*sin
    # and the second half gets x2*cos + x1*sin
    q_base = (b * sq_b + h * sq_h + l * sq_l)[:, None]
    q1 = tl.load(q_ptr + q_base + cols[None, :] * sq_d, mask=mask).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + (cols[None, :] + half) * sq_d, mask=mask).to(tl.float32)
    tl.store(q_out_ptr + o_base + cols[None, :] * so_d, (q1 * cos1 - q2 * sin1).to(q_out_ptr.dtype.element_ty), mask=mask)
    tl.store(q_out_ptr + o_base + (cols[None, :] + half) * so_d, (q2 * cos2 + q1 * sin2).to(q_out_ptr.dtype.element_ty), mask=mask)

    k_base = (b * sk_b + h * sk_h + l * sk_l)[:, None]
    k1 = tl.load(k_ptr + k_base + cols[None, :] * sk_d, mask=mask).to(tl.float32)
    k2 = tl.load(k_ptr + k_base + (cols[None, :] + half) * sk_d, mask=mask).to(tl.float32)
    tl.store(k_out_ptr + o_base + cols[None, :] * so_d, (k1 * cos1 - k2 * sin1).to(k_out_ptr.dtype.element_ty), mask=mask)
    tl.store(k_out_ptr + o_base + (cols[None, :] + half) * so_d, (k2 * cos2 + k1 * sin2).to(k_out_ptr.dtype.element_ty), mask=mask)

def rotary_embedding(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary embeddings for q and k of shape (batch, heads, length, head_dim).

    cos and sin have shape (batch, length, head_dim). Same signature and results as
    Hugging Face's apply_rotary_pos_emb, which computes in FP32 and returns the input dtype.
    """
    if unsqueeze_dim != 1:
        raise NotImplementedError("Only the (batch, heads, length, head_dim) layout is supported")
    if not q.is_cuda:
        raise RuntimeError("The Triton rotary kernel needs a CUDA tensor")
    batch, heads, seq_len, head_dim = q.shape
    if head_dim % 2:
        raise ValueError("head_dim must be even")

    cos = cos.expand(batch, seq_len, head_dim)
    sin = sin.expand(batch, seq_len, head_dim)
    if cos.stride() != sin.stride():
        sin = sin.contiguous()
        cos = cos.contiguous()
    q_out = torch.empty((batch, heads, seq_len, head_dim), device=q.device, dtype=q.dtype)
    k_out = torch.empty_like(q_out)

    half = head_dim // 2
    n_rows = batch * heads * seq_len
    block_rows = 32
    grid = (triton.cdiv(n_rows, block_rows),)
    _rotary_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        n_rows, heads, seq_len, half,
        *q.stride(), *k.stride(), *cos.stride(), *q_out.stride(),
        BLOCK_ROWS=block_rows, BLOCK_HALF=triton.next_power_of_2(half),
    )
    return q_out, k_out

def rotary_embedding_reference(q, k, cos, sin, unsqueeze_dim: int = 1):
    """PyTorch implementation, identical to Hugging Face's apply_rotary_pos_emb"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q.float() * cos + rotate_half(q.float()) * sin
    k_embed = k.float() * cos + rotate_half(k.float()) * sin
    return q_embed.to(dtype), k_embed.to(dtype)
