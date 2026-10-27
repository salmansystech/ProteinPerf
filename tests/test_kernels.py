"""Correctness of the Triton kernels against PyTorch (needs a CUDA GPU; skipped otherwise)"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU")

def _qk_cos_sin(batch, heads, tokens, head_dim, dtype):
    hidden = heads * head_dim
    # Views of a linear layer's output, like in the model: non-contiguous (batch, heads, length, head_dim)
    q = torch.randn(batch, tokens, hidden, device="cuda", dtype=dtype).view(batch, tokens, heads, head_dim).transpose(1, 2)
    k = torch.randn(batch, tokens, hidden, device="cuda", dtype=dtype).view(batch, tokens, heads, head_dim).transpose(1, 2)
    angles = torch.randn(batch, tokens, head_dim // 2, device="cuda")
    cos = torch.cat([angles.cos()] * 2, dim=-1).to(dtype)
    sin = torch.cat([angles.sin()] * 2, dim=-1).to(dtype)
    return q, k, cos, sin

# Head sizes of ESM-2 8M (16), 35M (24, so half is not a power of two), 150M (32) and 650M (64)
@pytest.mark.parametrize("head_dim", [16, 24, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rotary_matches_reference(head_dim, dtype):
    from proteinperf.kernels.rotary import rotary_embedding, rotary_embedding_reference
    q, k, cos, sin = _qk_cos_sin(2, 3, 37, head_dim, dtype)

    expected_q, expected_k = rotary_embedding_reference(q, k, cos, sin)
    actual_q, actual_k = rotary_embedding(q, k, cos, sin)

    assert actual_q.dtype == dtype and actual_q.shape == q.shape
    # Both compute in FP32 and round once, so results match to the last bit up to FMA rounding
    assert torch.allclose(actual_q.float(), expected_q.float(), atol=1e-6, rtol=1e-3)
    assert torch.allclose(actual_k.float(), expected_k.float(), atol=1e-6, rtol=1e-3)

def test_rotary_rejects_odd_head_dim():
    from proteinperf.kernels.rotary import rotary_embedding
    q = torch.randn(1, 1, 4, 7, device="cuda")

    with pytest.raises(ValueError, match="even"):
        rotary_embedding(q, q, torch.ones(1, 4, 7, device="cuda"), torch.ones(1, 4, 7, device="cuda"))

@pytest.mark.parametrize("shape", [(1000,), (3, 517, 2560), (5, 7)])
def test_gelu_matches_reference(shape):
    from proteinperf.kernels.gelu import gelu, gelu_reference
    x = torch.randn(*shape, device="cuda", dtype=torch.float16) * 3

    # The kernel computes in FP32, the reference in FP16, so they differ by FP16 rounding
    assert torch.allclose(gelu(x).float(), gelu_reference(x.float()), atol=2e-3, rtol=2e-3)

@pytest.mark.parametrize("m,n,k", [(4112, 2560, 640), (100, 72, 40), (1, 64, 16), (257, 129, 33)])
@pytest.mark.parametrize("with_bias", [True, False])
def test_linear_matches_torch(m, n, k, with_bias):
    from proteinperf.kernels.matmul import linear
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) / k ** 0.5
    bias = torch.randn(n, device="cuda", dtype=torch.float16) if with_bias else None

    expected = F.linear(x.float(), w.float(), bias.float() if bias is not None else None)
    actual = linear(x, w, bias)

    assert actual.shape == (m, n)
    assert torch.allclose(actual.float(), expected, atol=2e-2, rtol=1e-2)

def test_linear_with_fused_gelu_and_3d_input():
    from proteinperf.kernels.gelu import gelu_reference
    from proteinperf.kernels.matmul import linear
    torch.manual_seed(0)
    x = torch.randn(2, 33, 64, device="cuda", dtype=torch.float16)
    w = torch.randn(256, 64, device="cuda", dtype=torch.float16) / 8
    bias = torch.randn(256, device="cuda", dtype=torch.float16)

    expected = gelu_reference(F.linear(x.float(), w.float(), bias.float()))
    actual = linear(x, w, bias, gelu=True)

    assert actual.shape == (2, 33, 256)
    assert torch.allclose(actual.float(), expected, atol=2e-2, rtol=1e-2)

def test_linear_rejects_mismatched_shapes():
    from proteinperf.kernels.matmul import linear

    with pytest.raises(ValueError, match="don't match"):
        linear(torch.randn(4, 8, device="cuda"), torch.randn(16, 9, device="cuda"))
