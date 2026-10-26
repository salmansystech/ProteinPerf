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
