"""Accurate inference timing for protein language models"""
from dataclasses import dataclass, asdict
from typing import Callable, List
import statistics
import time
import torch

@dataclass
class BenchmarkResult:
    model: str
    dtype: str
    batch_size: int
    seq_len: int
    latency_ms: float          # median time for one forward pass of the whole batch
    latency_p90_ms: float
    sequences_per_s: float
    tokens_per_s: float
    peak_memory_mb: float
    device: str

    def to_dict(self) -> dict:
        return asdict(self)

def time_function(fn: Callable[[], None], device: str, warmup: int = 3, iters: int = 10) -> List[float]:
    """Run fn repeatedly and return each run's time in milliseconds"""
    for _ in range(warmup):
        fn()

    times = []
    if device.startswith("cuda"):
        # CUDA events measure GPU time precisely, without CPU launch noise
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
    return times

def percentile(values: List[float], fraction: float) -> float:
    """Value below which the given fraction of values fall (linear interpolation)"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

def benchmark_inference(model: torch.nn.Module, tokenizer, sequences: List[str], model_name: str,
                        dtype: str, device: str = "cuda", warmup: int = 3, iters: int = 10) -> BenchmarkResult:
    """Time a forward pass over one batch of equal-length sequences"""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)
    batch_size, tokens_per_sequence = inputs["input_ids"].shape

    def forward():
        with torch.inference_mode():
            model(**inputs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = time_function(forward, device, warmup, iters)
    latency = statistics.median(times)
    peak_memory = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0

    return BenchmarkResult(
        model=model_name,
        dtype=dtype,
        batch_size=batch_size,
        seq_len=len(sequences[0]),
        latency_ms=latency,
        latency_p90_ms=percentile(times, 0.9),
        sequences_per_s=batch_size / (latency / 1000),
        tokens_per_s=batch_size * tokens_per_sequence / (latency / 1000),
        peak_memory_mb=peak_memory,
        device=device,
    )
