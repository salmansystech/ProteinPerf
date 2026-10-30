import json

import pytest

from proteinperf.findings import collect_findings, parse_profile

PROFILE = """ESM-2 150M (fp16), batch 8 x length 512
GPU time per forward pass: 125.59 ms

Time by category:
  Matmul          53.67 ms   42.7%  #################
  Elementwise     36.39 ms   29.0%  ############
  Memory          14.08 ms   11.2%  ####
  LayerNorm        2.52 ms    2.0%  #

Slowest operators:
  aten::addmm                                   Matmul          53.67 ms    180 calls
"""

def _write(directory, name, rows, settings=None):
    (directory / name).write_text(json.dumps({"metadata": {"settings": settings or {}}, "results": rows}))

def _bench(dtype, latency, spilled=False):
    return {"model": "150M", "dtype": dtype, "batch_size": 8, "seq_len": 512, "latency_ms": latency,
            "peak_memory_mb": 100.0, "exceeds_gpu_memory": spilled}

def _variant(variant, speedup, batch=8):
    return {"model": "150M", "batch_size": batch, "seq_len": 512, "variant": variant, "speedup_vs_eager": speedup}

def test_parse_profile():
    profile = parse_profile(PROFILE)
    assert profile["title"] == "ESM-2 150M (fp16), batch 8 x length 512"
    assert profile["total_ms"] == 125.59
    assert profile["shares"] == pytest.approx({"Matmul": 0.427, "Elementwise": 0.29, "Memory": 0.112, "LayerNorm": 0.02})

def test_no_results_no_findings(tmp_path):
    assert collect_findings(str(tmp_path)) == []

def test_profile_suggests_fusion(tmp_path):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "p.txt").write_text(PROFILE)
    findings = collect_findings(str(tmp_path))
    assert any("Matmul is the largest category at 43%" in f for f in findings)
    assert any("elementwise and memory operations take 40%" in f for f in findings)

def test_baseline_speedup_and_spill(tmp_path):
    _write(tmp_path, "baseline.json", [_bench("fp32", 30.0), _bench("fp16", 10.0), _bench("bf16", 99.0, spilled=True)])
    findings = collect_findings(str(tmp_path))
    assert "1 baseline result(s) exceeded GPU memory and spilled into system RAM; they are excluded." in findings
    assert any("FP16 is 3.0-3.0x faster than FP32" in f for f in findings)
    assert not any("BF16" in f for f in findings)

def test_roofline_flags_slow_operations_only(tmp_path):
    def op(name, ms, efficiency):
        return {"model": "150M", "name": name, "bound": "memory", "measured_ms": ms,
                "roofline_ms": ms * efficiency, "efficiency": efficiency}
    rows = [op("gelu", 3.0, 0.2), op("matmul", 6.0, 0.9), op("layernorm", 0.1, 0.1)]
    _write(tmp_path, "roofline.json", rows, {"measured_peak_tflops": 30.0, "measured_bandwidth_gbs": 350.0,
                                             "models": [{"model": "150M", "whole_model_ms": 100.0, "roofline_ms": 50.0}]})
    findings = collect_findings(str(tmp_path))
    assert any(f.startswith("ESM-2 150M: gelu takes 33% of each layer") for f in findings)
    assert not any("matmul takes" in f or "layernorm takes" in f for f in findings)
    assert any("2.0x the roofline lower bound" in f for f in findings)

def test_optimizations_skip_noisy_shapes(tmp_path):
    rows = [_variant("eager", 1.0), _variant("sdpa-efficient", 1.0), _variant("compile", 1.5),
            _variant("eager", 1.0, batch=1), _variant("sdpa-efficient", 1.3, batch=1), _variant("compile", 2.0, batch=1)]
    _write(tmp_path, "optimizations.json", rows)
    findings = collect_findings(str(tmp_path))
    assert "ESM-2 150M batch 8 x length 512: best framework optimization is compile at 1.50x over eager." in findings
    assert "ESM-2 150M batch 1 x length 512: framework optimization results are too noisy to compare." in findings
