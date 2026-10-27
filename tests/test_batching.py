import pytest
from proteinperf.batching import STRATEGIES, make_batches, padding_stats, run_dataset

LENGTHS = [300, 20, 150, 25, 310, 30, 140, 22]

@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_every_sequence_in_exactly_one_batch(strategy):
    batches = make_batches(LENGTHS, strategy, batch_size=3, max_tokens=700)

    assert sorted(i for batch in batches for i in batch) == list(range(len(LENGTHS)))

def test_unsorted_keeps_dataset_order():
    assert make_batches(LENGTHS, "unsorted", batch_size=3) == [[0, 1, 2], [3, 4, 5], [6, 7]]

def test_sorted_groups_similar_lengths():
    batches = make_batches(LENGTHS, "sorted", batch_size=4)

    assert [[LENGTHS[i] for i in batch] for batch in batches] == [[20, 22, 25, 30], [140, 150, 300, 310]]

def test_token_budget_is_respected():
    batches = make_batches(LENGTHS, "token-budget", max_tokens=400)

    for batch in batches:
        longest = max(LENGTHS[i] for i in batch)
        # A single sequence longer than the budget still gets its own batch
        assert len(batch) * (longest + 2) <= 400 or len(batch) == 1
    # Short sequences share a batch; long ones don't
    assert [[LENGTHS[i] for i in batch] for batch in batches][0] == [20, 22, 25, 30]

def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        make_batches(LENGTHS, "random")

def test_padding_stats():
    lengths = [10, 20, 98]
    batches = [[0, 1], [2]]

    stats = padding_stats(batches, lengths)

    # Real: 12 + 22 + 100. Computed: two sequences padded to 22, plus 100.
    assert stats.real_tokens == 134
    assert stats.padded_tokens == 144
    assert stats.efficiency == pytest.approx(134 / 144)

def test_sorting_improves_padding_efficiency():
    unsorted = padding_stats(make_batches(LENGTHS, "unsorted", batch_size=2), LENGTHS)
    sorted_ = padding_stats(make_batches(LENGTHS, "sorted", batch_size=2), LENGTHS)

    assert sorted_.efficiency > unsorted.efficiency

def test_run_dataset_on_cpu(tiny_model, tokenizer):
    sequences = ["M" * n for n in (12, 40, 8, 33, 25)]

    result = run_dataset(tiny_model, tokenizer, sequences, "sorted", batch_size=2, device="cpu", warmup_batches=1)

    assert result.sequences == 5
    assert result.batches == 3
    assert result.total_s > 0
    assert result.sequences_per_s == pytest.approx(5 / result.total_s)
    assert 0 < result.padding_efficiency <= 1
