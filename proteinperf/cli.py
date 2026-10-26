"""Command-line interface: python -m proteinperf <command>"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import torch

from proteinperf.benchmark import benchmark_inference
from proteinperf.environment import environment_info
from proteinperf.models import DTYPES, ESM2_MODELS, MAX_SEQUENCE_LENGTH, load_model
from proteinperf.profiling import profile_inference
from proteinperf.sequences import synthetic_sequences

RESULTS_DIR = Path("results")

def _int_list(value: str):
    return [int(v) for v in value.split(",")]

def _str_list(value: str):
    return [v.strip() for v in value.split(",")]

def cmd_info(args):
    for key, value in environment_info().items():
        print(f"{key:20} {value}")

def cmd_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for size in args.models:
        for dtype in args.dtypes:
            print(f"\nLoading ESM-2 {size} ({dtype}) on {device}...")
            model, tokenizer = load_model(size, dtype, device)
            for seq_len in args.seq_lens:
                for batch_size in args.batch_sizes:
                    sequences = synthetic_sequences(batch_size, seq_len, seed=args.seed)
                    try:
                        result = benchmark_inference(model, tokenizer, sequences, size, dtype, device,
                                                     warmup=args.warmup, iters=args.iters)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  batch {batch_size:>3} x len {seq_len:>4}: out of GPU memory")
                        continue
                    rows.append(result.to_dict())
                    print(f"  batch {batch_size:>3} x len {seq_len:>4}: {result.latency_ms:8.2f} ms  "
                          f"{result.sequences_per_s:8.1f} seq/s  {result.tokens_per_s:10.0f} tok/s  "
                          f"{result.peak_memory_mb:7.0f} MB")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        print("No results to save.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "settings": {"warmup": args.warmup, "iters": args.iters, "seed": args.seed},
    }
    output.with_suffix(".json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))
    print(f"\nSaved {len(rows)} results to {output} and {output.with_suffix('.json')}")

def cmd_profile(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.model, args.dtype, device)
    sequences = synthetic_sequences(args.batch_size, args.seq_len, seed=args.seed)

    trace_dir = RESULTS_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"esm2_{args.model}_{args.dtype}_b{args.batch_size}_l{args.seq_len}.json"

    report = profile_inference(model, tokenizer, sequences, device, iters=args.iters, trace_path=str(trace_path))

    print(f"\nESM-2 {args.model} ({args.dtype}), batch {args.batch_size} x length {args.seq_len}")
    print(f"GPU time per forward pass: {report.total_device_ms:.2f} ms\n")
    print("Time by category:")
    for category, share in report.category_shares().items():
        print(f"  {category:12} {report.categories[category]:8.2f} ms  {share:6.1%}  {'#' * round(share * 40)}")
    print("\nSlowest operators:")
    for op in report.top_ops:
        print(f"  {op['op']:45} {op['category']:12} {op['device_ms']:8.2f} ms  {op['calls']:>5} calls")
    print(f"\nTrace saved to {trace_path} (open it at https://ui.perfetto.dev)")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="proteinperf", description="Benchmark and profile ESM-2 protein language models")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="Show GPU and library versions").set_defaults(func=cmd_info)

    bench = commands.add_parser("benchmark", help="Measure latency, throughput and memory")
    bench.add_argument("--models", type=_str_list, default=["8M", "35M", "150M"], help=f"Comma-separated: {','.join(ESM2_MODELS)}")
    bench.add_argument("--dtypes", type=_str_list, default=["fp32"], help=f"Comma-separated: {','.join(DTYPES)}")
    bench.add_argument("--batch-sizes", type=_int_list, default=[1, 8, 32])
    bench.add_argument("--seq-lens", type=_int_list, default=[128, 256, 512])
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--iters", type=int, default=10)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--output", default=str(RESULTS_DIR / "benchmark.csv"))
    bench.set_defaults(func=cmd_benchmark)

    prof = commands.add_parser("profile", help="Break down GPU time by operator")
    prof.add_argument("--model", default="150M", choices=list(ESM2_MODELS))
    prof.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    prof.add_argument("--batch-size", type=int, default=8)
    prof.add_argument("--seq-len", type=int, default=512)
    prof.add_argument("--iters", type=int, default=5)
    prof.add_argument("--seed", type=int, default=0)
    prof.set_defaults(func=cmd_profile)

    args = parser.parse_args(argv)
    if getattr(args, "seq_len", 0) > MAX_SEQUENCE_LENGTH or any(l > MAX_SEQUENCE_LENGTH for l in getattr(args, "seq_lens", [])):
        parser.error(f"ESM-2 supports sequences up to {MAX_SEQUENCE_LENGTH} residues")
    args.func(args)

if __name__ == "__main__":
    main()
