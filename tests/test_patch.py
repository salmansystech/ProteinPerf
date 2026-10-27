"""Swapping Triton kernels into ESM-2 (runs on CPU: checks the patching, not the kernels)"""
import pytest
from transformers.models.esm import modeling_esm
from proteinperf.kernels.patch import KERNELS, esm_triton_kernels
from proteinperf.variants import TRITON_VARIANTS, TritonKernelModel, build_variant

def _originals():
    return modeling_esm.apply_rotary_pos_emb, modeling_esm.gelu, modeling_esm.EsmIntermediate.forward

def test_ffn_is_swapped_inside_and_restored_after():
    before = _originals()

    with esm_triton_kernels(["ffn"]):
        assert modeling_esm.EsmIntermediate.forward is not before[2]
        assert modeling_esm.gelu is before[1]

    assert _originals() == before

def test_originals_restored_after_an_error():
    before = _originals()

    with pytest.raises(RuntimeError):
        with esm_triton_kernels(["ffn"]):
            raise RuntimeError("model failed")

    assert _originals() == before

def test_unknown_kernel():
    with pytest.raises(ValueError, match="Unknown kernels"):
        with esm_triton_kernels(["rotary", "softmax"]):
            pass

def test_triton_variants_use_known_kernels():
    assert all(set(kernels) <= set(KERNELS) for kernels in TRITON_VARIANTS.values())

def test_build_triton_variant(tiny_model):
    variant = build_variant(tiny_model, "triton")

    assert isinstance(variant, TritonKernelModel)
    assert variant.kernels == ("rotary", "ffn")
