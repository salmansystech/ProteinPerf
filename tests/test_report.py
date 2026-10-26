import pytest
from proteinperf.report import markdown_table, plot_results, speedups

def _row(model, dtype, batch, length, latency, memory):
    return {
        "model": model, "dtype": dtype, "batch_size": batch, "seq_len": length,
        "latency_ms": latency, "sequences_per_s": batch / (latency / 1000),
        "tokens_per_s": batch * (length + 2) / (latency / 1000), "peak_memory_mb": memory,
    }

ROWS = [
    _row("8M", "fp32", 8, 128, 10.0, 100),
    _row("8M", "fp16", 8, 128, 4.0, 60),
    _row("8M", "bf16", 8, 128, 5.0, 60),
    _row("650M", "fp32", 8, 128, 80.0, 3000),
    _row("650M", "fp16", 8, 128, 20.0, 1600),
    _row("8M", "fp32", 8, 256, 18.0, 150),
    _row("8M", "fp16", 8, 256, 6.0, 90),
]

def test_speedups_compare_against_fp32():
    result = {(s["model"], s["dtype"], s["seq_len"]): s for s in speedups(ROWS)}

    assert result[("8M", "fp16", 128)]["speedup"] == pytest.approx(2.5)
    assert result[("8M", "bf16", 128)]["speedup"] == pytest.approx(2.0)
    assert result[("650M", "fp16", 128)]["speedup"] == pytest.approx(4.0)
    assert result[("650M", "fp16", 128)]["memory_ratio"] == pytest.approx(1600 / 3000)
    assert not any(s["dtype"] == "fp32" for s in speedups(ROWS))

def test_speedups_skip_configurations_without_baseline():
    rows = [_row("8M", "fp16", 8, 128, 4.0, 60)]

    assert speedups(rows) == []

def test_markdown_table_shows_missing_results_as_oom():
    table = markdown_table(ROWS, batch_size=8, seq_len=128)
    lines = table.splitlines()

    assert lines[0] == "| Model | FP32 seq/s | FP16 seq/s | BF16 seq/s | FP32 memory | FP16 memory | BF16 memory |"
    assert lines[2] == "| ESM-2 8M | 800.0 | 2000.0 | 1600.0 | 0.10 GB | 0.06 GB | 0.06 GB |"
    # No BF16 result for 650M (e.g. it ran out of memory)
    assert lines[3].startswith("| ESM-2 650M | 100.0 | 400.0 | OOM |")

def test_plot_results_saves_charts(tmp_path):
    paths = plot_results(ROWS, str(tmp_path), batch_size=8, title_suffix="Test GPU")

    assert [p.split("\\")[-1].split("/")[-1] for p in paths] == ["throughput.png", "speedup.png", "memory.png"]
    assert all((tmp_path / name).stat().st_size > 0 for name in ("throughput.png", "speedup.png", "memory.png"))
