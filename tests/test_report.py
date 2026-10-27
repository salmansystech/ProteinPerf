import pytest
from proteinperf.report import (batching_table, markdown_table, noisy_shapes, optimization_table, plot_optimizations,
                                plot_results, speedups, valid_rows)

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

def test_valid_rows_excludes_spilled_measurements():
    spilled = dict(_row("650M", "fp32", 32, 1000, 135800.0, 5685), exceeds_gpu_memory=True)
    older = _row("8M", "fp32", 8, 128, 10.0, 100)  # results saved before the flag existed
    
    assert valid_rows([spilled, older]) == [older]

def _opt(model, batch, length, variant, latency, speedup):
    return {"model": model, "batch_size": batch, "seq_len": length, "variant": variant,
            "latency_ms": latency, "speedup_vs_eager": speedup}

OPT_ROWS = [
    _opt("150M", 1, 256, "eager", 27.5, 1.0),
    _opt("150M", 1, 256, "compile", 14.7, 1.88),
    _opt("150M", 1, 256, "compile-cudagraphs", 10.8, 2.56),
    _opt("150M", 8, 256, "eager", 65.3, 1.0),
    _opt("150M", 8, 256, "compile", 40.5, 1.61),
    _opt("650M", 1, 256, "eager", 60.0, 1.0),
]

def test_optimization_table():
    lines = optimization_table(OPT_ROWS, "150M").splitlines()
    
    assert lines[0] == "| Batch × length | Eager | `compile` | `compile-cudagraphs` |"
    assert lines[2] == "| 1 × 256 | 27.5 ms | 1.88× | 2.56× |"
    # Missing measurements are shown as n/a
    assert lines[3] == "| 8 × 256 | 65.3 ms | 1.61× | n/a |"

def test_batching_table():
    rows = [
        {"strategy": "unsorted", "batches": 125, "total_s": 40.0, "sequences_per_s": 25.0, "padding_efficiency": 0.55, "speedup_vs_unsorted": 1.0},
        {"strategy": "sorted", "batches": 125, "total_s": 25.0, "sequences_per_s": 40.0, "padding_efficiency": 0.97, "speedup_vs_unsorted": 1.6},
    ]
    
    lines = batching_table(rows).splitlines()
    
    assert lines[2] == "| `unsorted` | 125 | 40.0 s | 25.0 | 55% | 1.00× |"
    assert lines[3] == "| `sorted` | 125 | 25.0 s | 40.0 | 97% | 1.60× |"

def test_plot_optimizations(tmp_path):
    path = plot_optimizations(OPT_ROWS, str(tmp_path), title_suffix="Test GPU")
    
    assert (tmp_path / "optimizations.png").stat().st_size > 0
    assert path.endswith("optimizations.png")

def test_noisy_shapes_flagged_by_control_variant():
    rows = OPT_ROWS + [
        _opt("150M", 1, 256, "sdpa-efficient", 37.6, 1.47),  # same kernel as eager, yet 1.47x: noise
        _opt("150M", 8, 256, "sdpa-efficient", 65.9, 0.99),
    ]
    
    assert noisy_shapes(rows) == {("150M", 1, 256)}
    lines = optimization_table(rows, "150M").splitlines()
    assert lines[2].startswith("| 1 × 256 ⚠ |")
    assert lines[3].startswith("| 8 × 256 |")
