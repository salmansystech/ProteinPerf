import pytest

from proteinperf.models import ESM2_MODELS
from proteinperf.roofline import (LayerOp, attainable_tflops, layer_ops, model_flops, ridge_point,
                                  roofline_time_ms)

def _ops(size="650M", batch=8, length=510):
    return {op.name: op for op in layer_ops(ESM2_MODELS[size], batch, length)}

def test_projection_flops():
    ops = _ops(batch=1, length=510)          # 512 tokens with <cls> and <eos>
    h = 1280
    assert ops["q, k, v projection"].flops == 2 * 512 * h * 3 * h
    assert ops["feed-forward up"].flops == ops["feed-forward down"].flops == 2 * 512 * h * 4 * h

def test_attention_flops_grow_with_the_square_of_length():
    short, long = _ops(length=254)["attention"], _ops(length=510)["attention"]
    assert long.flops / short.flops == pytest.approx((512 / 256) ** 2)

def test_model_flops_close_to_two_flops_per_parameter_per_token():
    # Rule of thumb for transformers: about 2 FLOPs per weight per token, plus attention
    spec = ESM2_MODELS["650M"]
    per_token = model_flops(spec, 1, 126) / 128
    weights = spec.layers * 12 * spec.hidden_size ** 2
    assert 2 * weights < per_token < 2.1 * weights

def test_matmuls_compute_bound_and_elementwise_memory_bound():
    ridge = ridge_point(30, 360)
    for op in _ops().values():
        if op.kind == "Matmul":
            assert op.intensity > ridge
        if op.kind in ("Elementwise", "LayerNorm"):
            assert op.intensity < 10

def test_fp32_halves_intensity():
    spec = ESM2_MODELS["150M"]
    half = {op.name: op for op in layer_ops(spec, 8, 512, element_bytes=2)}
    full = {op.name: op for op in layer_ops(spec, 8, 512, element_bytes=4)}
    assert half["gelu"].intensity == pytest.approx(2 * full["gelu"].intensity)

def test_roofline_bounds():
    assert ridge_point(30, 300) == pytest.approx(100)
    assert attainable_tflops(10, 30, 300) == pytest.approx(3)      # below the ridge: bandwidth x intensity
    assert attainable_tflops(1000, 30, 300) == 30                  # above it: the compute peak
    op = LayerOp("x", "Elementwise", flops=1e9, bytes=3e9)
    assert roofline_time_ms(op, 30, 300) == pytest.approx(10)      # 3 GB at 300 GB/s
