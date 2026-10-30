"""Command-line interface: python -m proteinperf <command>"""
import argparse
import csv
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

from proteinperf.batching import STRATEGIES, run_dataset
from proteinperf.benchmark import benchmark_inference
from proteinperf.environment import environment_info
from proteinperf.models import DTYPES, ESM2_MODELS, MAX_SEQUENCE_LENGTH, load_model
from proteinperf.profiling import profile_inference
from proteinperf.report import (batching_table, kernel_table, load_results, markdown_table, optimization_table,
                                plot_optimizations, plot_results, plot_roofline, roofline_table, utilization_table,
                                valid_rows)
from proteinperf.sequences import read_fasta, synthetic_sequences
from proteinperf.variants import VARIANTS, build_variant, output_difference

RESULTS_DIR = Path("results")

def _int_list(value: str):
    return [int(v) for v in value.split(",")]

def _str_list(value: str):
    return [v.strip() for v in value.split(",")]

def cmd_info(args):
    for key, value in environment_info().items():
        print(f"{key:20} {value}")

def _save_results(output: Path, rows: list, metadata: dict):
    """Write results as CSV and as JSON with metadata"""
    # Rows can have different fields (e.g. GB/s for some kernels, TFLOPS for others)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))

def _release_gpu_memory(device: str):
    """Free models that are no longer referenced, and report what remains allocated"""
    # Hugging Face models contain reference cycles, so a deleted model is only
    # freed after garbage collection. Without this, each model stays in GPU
    # memory, inflating later memory numbers and eventually spilling to RAM.
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"  GPU memory still allocated after unloading: {torch.cuda.memory_allocated() / 2**20:.0f} MB", flush=True)

def _metadata(**settings) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "settings": settings,
    }

def cmd_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(warmup=args.warmup, iters=args.iters, seed=args.seed)

    for size in args.models:
        for dtype in args.dtypes:
            print(f"\nLoading ESM-2 {size} ({dtype}) on {device}...", flush=True)
            model, tokenizer = load_model(size, dtype, device)
            for seq_len in args.seq_lens:
                for batch_size in args.batch_sizes:
                    sequences = synthetic_sequences(batch_size, seq_len, seed=args.seed)
                    try:
                        result = benchmark_inference(model, tokenizer, sequences, size, dtype, device,
                                                     warmup=args.warmup, iters=args.iters)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  batch {batch_size:>3} x len {seq_len:>4}: out of GPU memory", flush=True)
                        continue
                    rows.append(result.to_dict())
                    if result.exceeds_gpu_memory:
                        print(f"  batch {batch_size:>3} x len {seq_len:>4}: exceeded GPU memory, timing not valid", flush=True)
                    # Save after every result so a long run can be stopped without losing data
                    _save_results(output, rows, metadata)
                    print(f"  batch {batch_size:>3} x len {seq_len:>4}: {result.latency_ms:8.2f} ms  "
                          f"{result.sequences_per_s:8.1f} seq/s  {result.tokens_per_s:10.0f} tok/s  "
                          f"{result.peak_memory_mb:7.0f} MB", flush=True)
            del model, tokenizer
            _release_gpu_memory(device)

    if rows:
        print(f"\nSaved {len(rows)} results to {output} and {output.with_suffix('.json')}")
    else:
        print("No results to save.")

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

def cmd_optimize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Eager mode always runs first: every variant is compared with it in the same session,
    # because GPU clocks on laptops vary from run to run
    variants = ["eager"] + [v for v in args.variants if v != "eager"]
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(warmup=args.warmup, iters=args.iters, seed=args.seed, variants=variants)

    for size in args.models:
        print(f"\nLoading ESM-2 {size} ({args.dtype}) on {device}...", flush=True)
        model, tokenizer = load_model(size, args.dtype, device)
        for seq_len in args.seq_lens:
            for batch_size in args.batch_sizes:
                sequences = synthetic_sequences(batch_size, seq_len, seed=args.seed)
                inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    reference = model(**inputs).last_hidden_state.clone()
                print(f"  batch {batch_size} x len {seq_len}:", flush=True)
                eager_latency = None

                for name in variants:
                    # Start every compiled variant from a clean slate, so each shape gets its own compiled graph
                    torch._dynamo.reset()
                    try:
                        variant = build_variant(model, name)
                        start = time.perf_counter()
                        with torch.inference_mode():
                            output_tensor = variant(**inputs).last_hidden_state.clone()
                        if device == "cuda":
                            torch.cuda.synchronize()
                        first_call_s = time.perf_counter() - start
                        result = benchmark_inference(variant, tokenizer, sequences, size, args.dtype, device,
                                                     warmup=args.warmup, iters=args.iters)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"    {name:20} out of GPU memory", flush=True)
                        continue
                    except RuntimeError as e:
                        if "No available kernel" not in str(e):
                            raise
                        print(f"    {name:20} not available on this GPU/build", flush=True)
                        continue

                    if name == "eager":
                        eager_latency = result.latency_ms
                    row = {**result.to_dict(), "variant": name, "first_call_s": first_call_s,
                           "speedup_vs_eager": eager_latency / result.latency_ms if eager_latency else None,
                           **output_difference(reference, output_tensor)}
                    rows.append(row)
                    _save_results(output, rows, metadata)
                    speedup = f"{row['speedup_vs_eager']:.2f}x" if row["speedup_vs_eager"] else "-"
                    print(f"    {name:20} {result.latency_ms:9.2f} ms  {speedup:>6}  first call {first_call_s:6.1f} s  "
                          f"max diff {row['max_abs_diff']:.4f}", flush=True)
                    del variant
        torch._dynamo.reset()
        del model, tokenizer
        _release_gpu_memory(device)

    print(f"\nSaved {len(rows)} results to {output} and {output.with_suffix('.json')}")

def cmd_batching(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequences = read_fasta(args.fasta)[:args.limit]
    lengths = [len(s) for s in sequences]
    print(f"{len(sequences)} proteins from {args.fasta}: lengths {min(lengths)}-{max(lengths)}, "
          f"median {sorted(lengths)[len(lengths) // 2]}")
    model, tokenizer = load_model(args.model, args.dtype, device)

    rows = []
    for strategy in STRATEGIES:
        result = run_dataset(model, tokenizer, sequences, strategy, args.batch_size, args.max_tokens, device)
        rows.append(result.to_dict())
        print(f"  {strategy:13} {result.total_s:7.2f} s  {result.sequences_per_s:7.1f} seq/s  "
              f"{result.batches:4} batches  padding efficiency {result.padding_efficiency:5.1%}  "
              f"{result.peak_memory_mb:6.0f} MB", flush=True)

    baseline = rows[0]["total_s"]
    for row in rows:
        row["speedup_vs_unsorted"] = baseline / row["total_s"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(model=args.model, dtype=args.dtype, fasta=args.fasta, sequences=len(sequences),
                         batch_size=args.batch_size, max_tokens=args.max_tokens)
    _save_results(output, rows, metadata)
    print(f"\nSaved results to {output} and {output.with_suffix('.json')}")

def cmd_report(args):
    data = load_results(args.input)
    rows = valid_rows(data["results"])
    excluded = len(data["results"]) - len(rows)
    if excluded:
        print(f"Excluded {excluded} result(s) that exceeded GPU memory")
    gpu = data.get("metadata", {}).get("environment", {}).get("gpu")
    
    for path in plot_results(rows, args.output_dir, batch_size=args.batch_size, title_suffix=gpu):
        print(f"Saved {path}")
    print(f"\nBatch {args.batch_size}, length {args.seq_len}:\n")
    print(markdown_table(rows, args.batch_size, args.seq_len))

    if Path(args.optimizations).exists():
        opt_rows = valid_rows(load_results(args.optimizations)["results"])
        print(f"\nSaved {plot_optimizations(opt_rows, args.output_dir, title_suffix=gpu)}")
        for model in dict.fromkeys(r["model"] for r in opt_rows):
            print(f"\nESM-2 {model}, speed-up over eager:\n")
            print(optimization_table(opt_rows, model))

    if Path(args.batching).exists():
        print("\nBatching strategies:\n")
        print(batching_table(load_results(args.batching)["results"]))

    if Path(args.kernels).exists():
        print("\nTriton kernel microbenchmarks:\n")
        print(kernel_table(load_results(args.kernels)["results"]))

    if Path(args.kernel_variants).exists():
        kv_rows = valid_rows(load_results(args.kernel_variants)["results"])
        title = "Custom Triton kernels vs torch.compile, FP16 (above the dashed line is faster)"
        print(f"\nSaved {plot_optimizations(kv_rows, args.output_dir, title_suffix=gpu, filename='kernels.png', title=title)}")
        for model in dict.fromkeys(r["model"] for r in kv_rows):
            print(f"\nESM-2 {model}, whole-model speed-up over eager:\n")
            print(optimization_table(kv_rows, model))

    if Path(args.roofline).exists():
        roof = load_results(args.roofline)
        settings = roof["metadata"]["settings"]
        peak, bandwidth = settings["measured_peak_tflops"], settings["measured_bandwidth_gbs"]
        print(f"\nSaved {plot_roofline(roof['results'], peak, bandwidth, args.output_dir, title_suffix=gpu)}")
        for model in dict.fromkeys(r["model"] for r in roof["results"]):
            print(f"\nESM-2 {model}, one layer, batch {settings['batch_size']} x length {settings['seq_len']}:\n")
            print(roofline_table(roof["results"], model))
        print("\nWhole model:\n")
        print("| Model | Measured | Sum of operations | Roofline | Measured / roofline |\n|---|---|---|---|---|")
        for m in settings["models"]:
            print(f"| ESM-2 {m['model']} | {m['whole_model_ms']:.1f} ms | {m['sum_of_operations_ms']:.1f} ms | "
                  f"{m['roofline_ms']:.1f} ms | {m['whole_model_ms'] / m['roofline_ms']:.1f}× |")
        print(f"\nFP16 model FLOPs utilization, TFLOPS (share of the {peak:.1f} TFLOPS matmul peak):\n")
        print(utilization_table(rows, peak))

def cmd_kernels(args):
    from proteinperf.kernels.bench import benchmark_kernels, measured_bandwidth_gbs

    bandwidth = measured_bandwidth_gbs()
    print(f"Measured copy bandwidth: {bandwidth:.0f} GB/s", flush=True)
    rows = []
    for size in args.models:
        for row in benchmark_kernels(size, args.batch_size, args.seq_len, DTYPES[args.dtype]):
            if "triton_gbs" in row:
                row["bandwidth_utilization"] = row["triton_gbs"] / bandwidth
            rows.append(row)
            extra = (f"{row['triton_gbs']:6.0f} GB/s ({row['bandwidth_utilization']:.0%})" if "triton_gbs" in row
                     else f"{row['triton_tflops']:6.1f} TFLOPS")
            print(f"  {size:5} {row['kernel']:12} eager {row['eager_ms']:7.3f} ms  compile {row['compiled_ms']:7.3f} ms  "
                  f"triton {row['triton_ms']:7.3f} ms  {row['speedup_vs_eager']:5.2f}x vs eager  "
                  f"{row['speedup_vs_compile']:5.2f}x vs compile  {extra}  max diff {row['max_abs_diff']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_results(output, rows, _metadata(dtype=args.dtype, batch_size=args.batch_size, seq_len=args.seq_len,
                                          measured_bandwidth_gbs=bandwidth))
    print(f"\nSaved results to {output} and {output.with_suffix('.json')}")

def cmd_roofline(args):
    from proteinperf.kernels.bench import measured_bandwidth_gbs
    from proteinperf.roofline import measure_layer, measured_peak_tflops, ridge_point

    dtype = DTYPES[args.dtype]
    bandwidth = measured_bandwidth_gbs()
    peak = measured_peak_tflops(dtype)
    print(f"Measured copy bandwidth {bandwidth:.0f} GB/s, {args.dtype.upper()} matmul peak {peak:.1f} TFLOPS, "
          f"ridge point {ridge_point(peak, bandwidth):.0f} FLOPs/byte", flush=True)

    rows, models = [], []
    for size in args.models:
        layer = measure_layer(size, args.batch_size, args.seq_len, dtype, peak, bandwidth)
        rows.extend(layer)
        print(f"\nESM-2 {size}, one layer, batch {args.batch_size} x length {args.seq_len}:")
        for r in layer:
            print(f"  {r['name']:28} {r['intensity']:7.1f} FLOPs/byte  {r['bound']:7}  {r['measured_ms']:7.3f} ms  "
                  f"roofline {r['roofline_ms']:7.3f} ms  efficiency {r['efficiency']:4.0%}", flush=True)

        # The whole model, to see how much time the per-operation model doesn't account for
        model, tokenizer = load_model(size, args.dtype, "cuda")
        sequences = synthetic_sequences(args.batch_size, args.seq_len, seed=0)
        whole = benchmark_inference(model, tokenizer, sequences, size, args.dtype, "cuda", warmup=5, iters=10)
        layers = ESM2_MODELS[size].layers
        summary = {
            "model": size, "layers": layers,
            "whole_model_ms": whole.latency_ms,
            "sum_of_operations_ms": layers * sum(r["measured_ms"] for r in layer),
            "roofline_ms": layers * sum(r["roofline_ms"] for r in layer),
        }
        models.append(summary)
        print(f"  whole model {summary['whole_model_ms']:.1f} ms, sum of operations {summary['sum_of_operations_ms']:.1f} ms, "
              f"roofline {summary['roofline_ms']:.1f} ms", flush=True)
        del model, tokenizer
        _release_gpu_memory("cuda")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_results(output, rows, _metadata(dtype=args.dtype, batch_size=args.batch_size, seq_len=args.seq_len,
                                          measured_bandwidth_gbs=bandwidth, measured_peak_tflops=peak, models=models))
    print(f"\nSaved results to {output} and {output.with_suffix('.json')}")

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

    opt = commands.add_parser("optimize", help="Compare framework-level optimizations with eager mode")
    opt.add_argument("--models", type=_str_list, default=["150M"], help=f"Comma-separated: {','.join(ESM2_MODELS)}")
    opt.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    opt.add_argument("--variants", type=_str_list, default=list(VARIANTS), help=f"Comma-separated: {','.join(VARIANTS)}")
    opt.add_argument("--batch-sizes", type=_int_list, default=[1, 8, 32])
    opt.add_argument("--seq-lens", type=_int_list, default=[256, 512])
    opt.add_argument("--warmup", type=int, default=5)
    opt.add_argument("--iters", type=int, default=10)
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--output", default=str(RESULTS_DIR / "optimizations.csv"))
    opt.set_defaults(func=cmd_optimize)

    bat = commands.add_parser("batching", help="Compare batching strategies on real proteins of different lengths")
    bat.add_argument("--fasta", default="data/uniprot_human_sample.fasta")
    bat.add_argument("--limit", type=int, default=1000, help="Use only the first N proteins")
    bat.add_argument("--model", default="150M", choices=list(ESM2_MODELS))
    bat.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    bat.add_argument("--batch-size", type=int, default=8)
    bat.add_argument("--max-tokens", type=int, default=4096, help="Token budget per batch for the token-budget strategy")
    bat.add_argument("--output", default=str(RESULTS_DIR / "batching.csv"))
    bat.set_defaults(func=cmd_batching)

    ker = commands.add_parser("kernels", help="Microbenchmark the custom Triton kernels against PyTorch and torch.compile")
    ker.add_argument("--models", type=_str_list, default=["150M", "650M"], help="Model sizes whose layer shapes are used")
    ker.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ker.add_argument("--batch-size", type=int, default=8)
    ker.add_argument("--seq-len", type=int, default=512)
    ker.add_argument("--output", default=str(RESULTS_DIR / "kernels.csv"))
    ker.set_defaults(func=cmd_kernels)

    roof = commands.add_parser("roofline", help="Place each layer operation on the GPU's roofline")
    roof.add_argument("--models", type=_str_list, default=["150M", "650M"], help=f"Comma-separated: {','.join(ESM2_MODELS)}")
    roof.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    roof.add_argument("--batch-size", type=int, default=8)
    roof.add_argument("--seq-len", type=int, default=512)
    roof.add_argument("--output", default=str(RESULTS_DIR / "roofline.csv"))
    roof.set_defaults(func=cmd_roofline)

    rep = commands.add_parser("report", help="Charts and a summary table from benchmark results")
    rep.add_argument("--input", default=str(RESULTS_DIR / "baseline.json"))
    rep.add_argument("--output-dir", default=str(RESULTS_DIR / "charts"))
    rep.add_argument("--batch-size", type=int, default=8)
    rep.add_argument("--seq-len", type=int, default=512)
    rep.add_argument("--optimizations", default=str(RESULTS_DIR / "optimizations.json"), help="Included when the file exists")
    rep.add_argument("--batching", default=str(RESULTS_DIR / "batching.json"), help="Included when the file exists")
    rep.add_argument("--kernels", default=str(RESULTS_DIR / "kernels.json"), help="Included when the file exists")
    rep.add_argument("--kernel-variants", default=str(RESULTS_DIR / "kernel_variants.json"), help="Included when the file exists")
    rep.add_argument("--roofline", default=str(RESULTS_DIR / "roofline.json"), help="Included when the file exists")
    rep.set_defaults(func=cmd_report)
    
    args = parser.parse_args(argv)
    if getattr(args, "seq_len", 0) > MAX_SEQUENCE_LENGTH or any(l > MAX_SEQUENCE_LENGTH for l in getattr(args, "seq_lens", [])):
        parser.error(f"ESM-2 supports sequences up to {MAX_SEQUENCE_LENGTH} residues")
    args.func(args)

if __name__ == "__main__":
    main()
