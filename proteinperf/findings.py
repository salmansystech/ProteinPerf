"""Rule-based findings: the main conclusions from each results file, without a language model"""
from collections import defaultdict
from pathlib import Path
from typing import List
import re

from proteinperf.report import load_results, noisy_shapes, speedups, valid_rows

# An operation is worth optimizing when it takes a real share of the layer and runs far below its roofline
ROOFLINE_SHARE = 0.10
ROOFLINE_EFFICIENCY = 0.5
# Elementwise and memory operations above this share of GPU time suggest kernel fusion
FUSION_SHARE = 0.25

def baseline_findings(path: Path) -> List[str]:
    data = load_results(path)
    rows = valid_rows(data["results"])
    out = []
    spilled = len(data["results"]) - len(rows)
    if spilled:
        out.append(f"{spilled} baseline result(s) exceeded GPU memory and spilled into system RAM; they are excluded.")
    grouped = defaultdict(list)
    for s in speedups(rows):
        grouped[(s["model"], s["dtype"])].append(s["speedup"])
    for (model, dtype), values in grouped.items():
        out.append(f"ESM-2 {model}: {dtype.upper()} is {min(values):.1f}-{max(values):.1f}x faster than FP32 "
                   f"(average {sum(values) / len(values):.1f}x over {len(values)} shapes).")
    return out

def parse_profile(text: str) -> dict:
    """Header and category shares from a profile report written by the `profile` command"""
    title = text.strip().splitlines()[0]
    total = re.search(r"GPU time per forward pass: ([\d.]+) ms", text)
    shares = {m.group(1): float(m.group(2)) / 100
              for m in re.finditer(r"^\s+(\w+)\s+[\d.]+ ms\s+([\d.]+)%", text, re.MULTILINE)}
    return {"title": title, "total_ms": float(total.group(1)) if total else None, "shares": shares}

def profile_findings(path: Path) -> List[str]:
    profile = parse_profile(path.read_text(encoding="utf-8"))
    shares = profile["shares"]
    if not shares:
        return []
    top = max(shares, key=shares.get)
    out = [f"{profile['title']}: {profile['total_ms']} ms per forward pass; {top} is the largest category "
           f"at {shares[top]:.0%}."]
    small_ops = shares.get("Elementwise", 0) + shares.get("Memory", 0)
    if small_ops > FUSION_SHARE:
        out.append(f"{profile['title']}: elementwise and memory operations take {small_ops:.0%} of GPU time. "
                   "These are memory-bound, so fusing them (torch.compile or custom kernels) should help.")
    return out

def roofline_findings(path: Path) -> List[str]:
    data = load_results(path)
    settings = data["metadata"]["settings"]
    out = [f"Measured limits: {settings['measured_peak_tflops']:.1f} TFLOPS FP16 matmul, "
           f"{settings['measured_bandwidth_gbs']:.0f} GB/s memory bandwidth."]
    rows = data["results"]
    for model in dict.fromkeys(r["model"] for r in rows):
        layer = [r for r in rows if r["model"] == model]
        layer_ms = sum(r["measured_ms"] for r in layer)
        for r in sorted(layer, key=lambda r: r["measured_ms"], reverse=True):
            share = r["measured_ms"] / layer_ms
            if share >= ROOFLINE_SHARE and r["efficiency"] < ROOFLINE_EFFICIENCY:
                out.append(f"ESM-2 {model}: {r['name']} takes {share:.0%} of each layer but reaches only "
                           f"{r['efficiency']:.0%} of its {r['bound']}-bound roofline "
                           f"({r['measured_ms']:.3f} ms vs {r['roofline_ms']:.3f} ms possible).")
    for m in settings.get("models", []):
        out.append(f"ESM-2 {m['model']}: the whole model takes {m['whole_model_ms']:.1f} ms, "
                   f"{m['whole_model_ms'] / m['roofline_ms']:.1f}x the roofline lower bound of {m['roofline_ms']:.1f} ms.")
    return out

def _best_variants(rows: List[dict], label: str) -> List[str]:
    noisy = noisy_shapes(rows)
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        for shape in sorted({(r["batch_size"], r["seq_len"]) for r in rows if r["model"] == model}):
            if (model, *shape) in noisy:
                out.append(f"ESM-2 {model} batch {shape[0]} x length {shape[1]}: {label} results are too noisy to compare.")
                continue
            candidates = [r for r in rows if r["model"] == model and (r["batch_size"], r["seq_len"]) == shape
                          and r["variant"] != "eager" and r.get("speedup_vs_eager")]
            if candidates:
                best = max(candidates, key=lambda r: r["speedup_vs_eager"])
                out.append(f"ESM-2 {model} batch {shape[0]} x length {shape[1]}: best {label} is "
                           f"{best['variant']} at {best['speedup_vs_eager']:.2f}x over eager.")
    return out

def optimization_findings(path: Path) -> List[str]:
    return _best_variants(valid_rows(load_results(path)["results"]), "framework optimization")

def kernel_variant_findings(path: Path) -> List[str]:
    return _best_variants(valid_rows(load_results(path)["results"]), "variant in the Triton kernel comparison")

def batching_findings(path: Path) -> List[str]:
    return [f"Batching strategy {r['strategy']}: {r['speedup_vs_unsorted']:.2f}x over unsorted batches, "
            f"padding efficiency {r['padding_efficiency']:.0%}, {r['batches']} batches, "
            f"peak memory {r['peak_memory_mb']:.0f} MB." for r in load_results(path)["results"]]

def kernel_findings(path: Path) -> List[str]:
    out = []
    for r in load_results(path)["results"]:
        detail = (f", {r['bandwidth_utilization']:.0%} of measured bandwidth" if "bandwidth_utilization" in r
                  else f", {r['triton_tflops']:.1f} TFLOPS" if "triton_tflops" in r else "")
        out.append(f"Triton {r['kernel']} kernel (ESM-2 {r['model']} shapes): {r['speedup_vs_eager']:.2f}x vs eager, "
                   f"{r['speedup_vs_compile']:.2f}x vs torch.compile{detail}.")
    return out

# Results file -> rules that read it
RULES = {
    "baseline.json": baseline_findings,
    "roofline.json": roofline_findings,
    "optimizations.json": optimization_findings,
    "batching.json": batching_findings,
    "kernels.json": kernel_findings,
    "kernel_variants.json": kernel_variant_findings,
}

def collect_findings(results_dir: str = "results") -> List[str]:
    """Findings from every results file and profile report that exists"""
    directory = Path(results_dir)
    out = []
    for name, rule in RULES.items():
        if (directory / name).exists():
            out.extend(rule(directory / name))
    for profile in sorted((directory / "profiles").glob("*.txt")):
        out.extend(profile_findings(profile))
    return out
