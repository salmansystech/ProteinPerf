"""Batching strategies for real proteins of different lengths"""
from dataclasses import dataclass, asdict
from typing import List
import time
import torch

STRATEGIES = {
    "unsorted": "Fixed batch size in dataset order: each batch pads to its longest protein",
    "sorted": "Fixed batch size after sorting by length: similar lengths share a batch",
    "token-budget": "Sorted, with batches filled up to a token budget: many short proteins or few long ones",
}

def make_batches(lengths: List[int], strategy: str, batch_size: int = 8, max_tokens: int = 4096) -> List[List[int]]:
    """Group sequence indices into batches; every index appears in exactly one batch"""
    if strategy == "unsorted":
        order = list(range(len(lengths)))
    elif strategy in ("sorted", "token-budget"):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")

    if strategy != "token-budget":
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    # Each sequence is padded to the longest in its batch, plus CLS and EOS tokens
    batches, current = [], []
    for index in order:
        padded_tokens = (len(current) + 1) * (lengths[index] + 2)
        if current and padded_tokens > max_tokens:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches

@dataclass
class PaddingStats:
    real_tokens: int
    padded_tokens: int

    @property
    def efficiency(self) -> float:
        """Fraction of computed tokens that are real protein, not padding"""
        return self.real_tokens / self.padded_tokens if self.padded_tokens else 0.0

def padding_stats(batches: List[List[int]], lengths: List[int]) -> PaddingStats:
    """Real tokens versus tokens actually computed once batches are padded"""
    real = sum(lengths[i] + 2 for batch in batches for i in batch)
    padded = sum(len(batch) * (max(lengths[i] for i in batch) + 2) for batch in batches)
    return PaddingStats(real, padded)

@dataclass
class DatasetResult:
    strategy: str
    sequences: int
    batches: int
    total_s: float
    sequences_per_s: float
    real_tokens_per_s: float
    padding_efficiency: float
    peak_memory_mb: float

    def to_dict(self) -> dict:
        return asdict(self)

def run_dataset(model: torch.nn.Module, tokenizer, sequences: List[str], strategy: str,
                batch_size: int = 8, max_tokens: int = 4096, device: str = "cuda", warmup_batches: int = 3) -> DatasetResult:
    """Embed every sequence with one batching strategy and time the whole pass"""
    lengths = [len(s) for s in sequences]
    batches = make_batches(lengths, strategy, batch_size, max_tokens)
    # Tokenize up front so the timing covers the model, not string processing
    encoded = [tokenizer([sequences[i] for i in batch], return_tensors="pt", padding=True) for batch in batches]

    def forward(inputs):
        with torch.inference_mode():
            model(**inputs.to(device))

    for inputs in encoded[:warmup_batches]:
        forward(inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for inputs in encoded:
        forward(inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total = time.perf_counter() - start

    stats = padding_stats(batches, lengths)
    return DatasetResult(
        strategy=strategy,
        sequences=len(sequences),
        batches=len(batches),
        total_s=total,
        sequences_per_s=len(sequences) / total,
        real_tokens_per_s=stats.real_tokens / total,
        padding_efficiency=stats.efficiency,
        peak_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0,
    )
