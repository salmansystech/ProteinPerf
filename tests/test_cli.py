import pytest
from proteinperf.cli import main
from proteinperf.models import load_model

def test_rejects_sequences_longer_than_esm2_supports():
    with pytest.raises(SystemExit):
        main(["benchmark", "--seq-lens", "128,2000"])

def test_rejects_unknown_model():
    with pytest.raises(SystemExit):
        main(["profile", "--model", "999B"])

def test_load_model_validates_arguments():
    with pytest.raises(ValueError, match="Unknown model size"):
        load_model("999B")
    with pytest.raises(ValueError, match="Unknown dtype"):
        load_model("8M", dtype="int3")

def test_save_results_handles_rows_with_different_fields(tmp_path):
    import csv, json
    from proteinperf.cli import _save_results
    rows = [{"kernel": "rotary", "triton_gbs": 330.0}, {"kernel": "matmul", "triton_tflops": 27.5}]
    
    _save_results(tmp_path / "out.csv", rows, {"note": "test"})
    
    with open(tmp_path / "out.csv", newline="") as f:
        saved = list(csv.DictReader(f))
    assert list(saved[0]) == ["kernel", "triton_gbs", "triton_tflops"]
    assert saved[1]["triton_tflops"] == "27.5" and saved[1]["triton_gbs"] == ""
    assert json.loads((tmp_path / "out.json").read_text())["results"] == rows
