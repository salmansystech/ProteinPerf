import pytest
from proteinperf.sequences import AMINO_ACID_FREQUENCIES, read_fasta, synthetic_sequences

def test_count_and_length():
    sequences = synthetic_sequences(5, 120)

    assert len(sequences) == 5
    assert all(len(s) == 120 for s in sequences)

def test_only_standard_amino_acids():
    sequence = synthetic_sequences(1, 5000)[0]

    assert set(sequence) <= set(AMINO_ACID_FREQUENCIES)

def test_reproducible_with_seed():
    assert synthetic_sequences(3, 50, seed=7) == synthetic_sequences(3, 50, seed=7)
    assert synthetic_sequences(3, 50, seed=7) != synthetic_sequences(3, 50, seed=8)

def test_composition_matches_natural_proteins():
    sequence = synthetic_sequences(1, 200_000, seed=1)[0]
    total = sum(AMINO_ACID_FREQUENCIES.values())

    # Leucine is the most common amino acid and tryptophan one of the rarest
    assert sequence.count("L") / len(sequence) == pytest.approx(AMINO_ACID_FREQUENCIES["L"] / total, abs=0.003)
    assert sequence.count("W") / len(sequence) == pytest.approx(AMINO_ACID_FREQUENCIES["W"] / total, abs=0.002)

@pytest.mark.parametrize("count,length", [(0, 10), (3, 0)])
def test_invalid_arguments(count, length):
    with pytest.raises(ValueError):
        synthetic_sequences(count, length)

def test_read_fasta(tmp_path):
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">sp|P1|first protein\nMKTA\nyiak\n\n>sp|P2|second\nGGSW\n")

    assert read_fasta(str(fasta)) == ["MKTAYIAK", "GGSW"]

def test_read_empty_fasta(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text(">header only\n")

    with pytest.raises(ValueError, match="No sequences"):
        read_fasta(str(fasta))
