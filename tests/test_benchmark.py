import pytest
from proteinperf.benchmark import benchmark_inference, percentile, time_function
from proteinperf.sequences import synthetic_sequences

def test_percentile():
    values = [10, 20, 30, 40, 50]

    assert percentile(values, 0.5) == 30
    assert percentile(values, 0.9) == pytest.approx(46)
    assert percentile(values, 0.0) == 10
    assert percentile(values, 1.0) == 50

def test_time_function_runs_warmup_and_timed_iterations():
    calls = []

    times = time_function(lambda: calls.append(1), device="cpu", warmup=2, iters=5)

    assert len(calls) == 7
    assert len(times) == 5
    assert all(t >= 0 for t in times)

def test_benchmark_inference(tiny_model, tokenizer):
    sequences = synthetic_sequences(4, 30)

    result = benchmark_inference(tiny_model, tokenizer, sequences, "tiny", "fp32", device="cpu", warmup=1, iters=3)

    assert result.model == "tiny"
    assert result.batch_size == 4
    assert result.seq_len == 30
    assert result.latency_ms > 0
    assert result.latency_p90_ms >= result.latency_ms * 0.5
    assert result.sequences_per_s == pytest.approx(4 / (result.latency_ms / 1000))
    # Every sequence also gets a <cls> and an <eos> token
    assert result.tokens_per_s == pytest.approx(4 * 32 / (result.latency_ms / 1000))
    assert result.peak_memory_mb == 0
    assert set(result.to_dict()) >= {"model", "latency_ms", "tokens_per_s", "peak_memory_mb"}
