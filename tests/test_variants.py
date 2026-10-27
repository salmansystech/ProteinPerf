import pytest
import torch
from proteinperf.sequences import synthetic_sequences
from proteinperf.variants import AttentionBackendModel, VARIANTS, build_variant, output_difference

def test_eager_variant_is_the_model_itself(tiny_model):
    assert build_variant(tiny_model, "eager") is tiny_model

def test_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_variant(torch.nn.Identity(), "turbo")

def test_every_variant_is_described():
    assert {"eager", "compile", "compile-cudagraphs", "sdpa-efficient", "sdpa-cudnn", "sdpa-flash", "sdpa-math",
            "triton-rotary", "triton-gelu", "triton-ffn", "triton"} == set(VARIANTS)

def test_math_attention_matches_default(tiny_model, tokenizer):
    inputs = tokenizer(synthetic_sequences(2, 20), return_tensors="pt", padding=True)
    variant = build_variant(tiny_model, "sdpa-math")

    with torch.inference_mode():
        expected = tiny_model(**inputs).last_hidden_state
        actual = variant(**inputs).last_hidden_state

    assert isinstance(variant, AttentionBackendModel)
    assert torch.allclose(expected, actual, atol=1e-5)

def test_output_difference_identical():
    x = torch.randn(2, 10, 16)

    diff = output_difference(x, x.clone())

    assert diff["max_abs_diff"] == 0
    assert diff["mean_abs_diff"] == 0
    assert diff["min_cosine_similarity"] == pytest.approx(1.0)

def test_output_difference_detects_changes():
    reference = torch.ones(1, 3, 4)
    changed = reference.clone()
    changed[0, 1] = -1  # one token's embedding points the opposite way

    diff = output_difference(reference, changed)

    assert diff["max_abs_diff"] == pytest.approx(2.0)
    assert diff["mean_abs_diff"] == pytest.approx(2.0 * 4 / 12)
    assert diff["min_cosine_similarity"] == pytest.approx(-1.0)

def test_output_difference_works_in_half_precision():
    x = torch.randn(1, 4, 8, dtype=torch.float16)

    assert output_difference(x, x)["min_cosine_similarity"] == pytest.approx(1.0, abs=1e-3)
