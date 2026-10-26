from types import SimpleNamespace
import pytest
from proteinperf.profiling import categorize, profile_inference, summarize
from proteinperf.sequences import synthetic_sequences

@pytest.mark.parametrize("op,category", [
    ("aten::addmm", "Matmul"),
    ("aten::mm", "Matmul"),
    ("aten::linear", "Matmul"),
    ("aten::bmm", "Matmul"),
    ("aten::_scaled_dot_product_efficient_attention", "Attention"),
    ("aten::_softmax", "Attention"),
    ("aten::native_layer_norm", "LayerNorm"),
    ("aten::gelu", "Activation"),
    ("aten::erf", "Activation"),
    ("aten::add", "Elementwise"),
    ("aten::add_", "Elementwise"),
    ("aten::mul", "Elementwise"),
    ("aten::copy_", "Memory"),
    ("aten::_to_copy", "Memory"),
    ("aten::embedding", "Memory"),
    ("aten::something_new", "Other"),
])
def test_categorize(op, category):
    assert categorize(op) == category

def _event(key, device_us, count=1):
    return SimpleNamespace(key=key, self_device_time_total=device_us, count=count)

def test_summarize_groups_and_ranks_operators():
    events = [
        _event("aten::addmm", 6000, count=10),
        _event("aten::mm", 2000),
        _event("aten::native_layer_norm", 1000),
        _event("aten::gelu", 500),
        _event("aten::view", 0),  # no GPU time: ignored
    ]

    report = summarize(events)

    assert report.total_device_ms == pytest.approx(9.5)
    assert report.categories == {"Matmul": pytest.approx(8.0), "LayerNorm": pytest.approx(1.0), "Activation": pytest.approx(0.5)}
    assert [op["op"] for op in report.top_ops] == ["aten::addmm", "aten::mm", "aten::native_layer_norm", "aten::gelu"]
    assert list(report.category_shares()) == ["Matmul", "LayerNorm", "Activation"]
    assert report.category_shares()["Matmul"] == pytest.approx(8.0 / 9.5)

def test_summarize_skips_kernels_and_profiler_events():
    """GPU kernels repeat their operator's time, so counting them would double count"""
    events = [
        SimpleNamespace(key="aten::mm", self_device_time_total=2000, count=1, device_type="DeviceType.CPU"),
        SimpleNamespace(key="ampere_sgemm_64x32_nn", self_device_time_total=2000, count=1, device_type="DeviceType.CUDA"),
        SimpleNamespace(key="Activity Buffer Request", self_device_time_total=2000, count=1, device_type="DeviceType.CPU"),
    ]
    
    report = summarize(events)
    
    assert report.total_device_ms == pytest.approx(2.0)
    assert [op["op"] for op in report.top_ops] == ["aten::mm"]

def test_summarize_handles_older_pytorch_attribute():
    events = [SimpleNamespace(key="aten::addmm", self_cuda_time_total=3000, count=1)]

    assert summarize(events).total_device_ms == pytest.approx(3.0)

def test_profile_inference_on_cpu(tiny_model, tokenizer, tmp_path):
    sequences = synthetic_sequences(2, 40)
    trace = tmp_path / "trace.json"

    report = profile_inference(tiny_model, tokenizer, sequences, device="cpu", warmup=1, iters=2, trace_path=str(trace))

    assert report.total_device_ms > 0
    assert "Matmul" in report.categories
    assert report.top_ops
    assert trace.exists()
