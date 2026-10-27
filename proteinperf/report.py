"""Charts and summary tables from benchmark results"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import json

MODEL_ORDER = ["8M", "35M", "150M", "650M", "3B"]
DTYPE_COLORS = {"fp32": "#64748b", "fp16": "#2563eb", "bf16": "#16a34a"}
# BF16 is dashed so it stays visible where it overlaps FP16 (e.g. identical memory use)
DTYPE_LINESTYLES = {"fp32": "-", "fp16": "-", "bf16": "--"}

def load_results(path: str) -> dict:
    """Benchmark results and metadata from the JSON file written by `benchmark`"""
    return json.loads(Path(path).read_text())

def valid_rows(rows: List[dict]) -> List[dict]:
    """Results whose timings are meaningful (not spilled out of GPU memory)"""
    return [r for r in rows if not r.get("exceeds_gpu_memory", False)]

def _index(rows: List[dict]) -> Dict[tuple, dict]:
    return {(r["model"], r["dtype"], r["batch_size"], r["seq_len"]): r for r in rows}

def speedups(rows: List[dict], baseline: str = "fp32") -> List[dict]:
    """Speed-up of every non-baseline precision over the baseline, for each matching configuration"""
    index = _index(rows)
    out = []
    for (model, dtype, batch, length), row in index.items():
        reference = index.get((model, baseline, batch, length))
        if dtype == baseline or reference is None:
            continue
        out.append({
            "model": model, "dtype": dtype, "batch_size": batch, "seq_len": length,
            "speedup": reference["latency_ms"] / row["latency_ms"],
            "memory_ratio": row["peak_memory_mb"] / reference["peak_memory_mb"] if reference["peak_memory_mb"] else None,
        })
    return out

def markdown_table(rows: List[dict], batch_size: int, seq_len: int) -> str:
    """Throughput and memory per model and precision, for one batch size and length"""
    selected = [r for r in rows if r["batch_size"] == batch_size and r["seq_len"] == seq_len]
    dtypes = [d for d in DTYPE_COLORS if any(r["dtype"] == d for r in selected)]
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in selected)]
    index = _index(selected)

    header = "| Model | " + " | ".join(f"{d.upper()} seq/s" for d in dtypes) + " | " + \
             " | ".join(f"{d.upper()} memory" for d in dtypes) + " |"
    lines = [header, "|" + "---|" * (1 + 2 * len(dtypes))]
    for model in models:
        cells = []
        for dtype in dtypes:
            row = index.get((model, dtype, batch_size, seq_len))
            cells.append(f"{row['sequences_per_s']:.1f}" if row else "OOM")
        for dtype in dtypes:
            row = index.get((model, dtype, batch_size, seq_len))
            cells.append(f"{row['peak_memory_mb'] / 1024:.2f} GB" if row else "OOM")
        lines.append(f"| ESM-2 {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def plot_results(rows: List[dict], output_dir: str, batch_size: int = 8, title_suffix: Optional[str] = None) -> List[str]:
    """Throughput, speed-up and memory charts; returns the saved file paths"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"\n{title_suffix}" if title_suffix else ""
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    saved = []

    # 1. Throughput vs sequence length, one panel per model
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.9), squeeze=False)
    for ax, model in zip(axes[0], models):
        for dtype, color in DTYPE_COLORS.items():
            points = sorted((r["seq_len"], r["tokens_per_s"]) for r in rows
                            if r["model"] == model and r["dtype"] == dtype and r["batch_size"] == batch_size)
            if points:
                ax.plot(*zip(*points), marker="o", color=color, linestyle=DTYPE_LINESTYLES[dtype], label=dtype.upper())
        ax.set_title(f"ESM-2 {model}")
        ax.set_xlabel("Sequence length")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Tokens / second")
    axes[0][0].legend()
    fig.suptitle(f"Inference throughput, batch {batch_size}{suffix}", fontsize=11)
    fig.tight_layout()
    path = out / "throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    # 2. Average speed-up of half precision over FP32, per model
    grouped = defaultdict(list)
    for s in speedups(rows):
        grouped[(s["model"], s["dtype"])].append(s["speedup"])
    half = [d for d in ("fp16", "bf16") if any(k[1] == d for k in grouped)]
    if half:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        width = 0.8 / len(half)
        for i, dtype in enumerate(half):
            values = [sum(grouped[(m, dtype)]) / len(grouped[(m, dtype)]) if grouped[(m, dtype)] else 0 for m in models]
            positions = [x + i * width for x in range(len(models))]
            bars = ax.bar(positions, values, width, color=DTYPE_COLORS[dtype], label=dtype.upper())
            ax.bar_label(bars, fmt="%.1fx", fontsize=8)
        ax.axhline(1, color="#64748b", linestyle="--", linewidth=1)
        ax.set_xticks([x + width * (len(half) - 1) / 2 for x in range(len(models))], [f"ESM-2 {m}" for m in models])
        ax.set_ylabel("Speed-up over FP32")
        ax.set_title(f"Half-precision speed-up over FP32, averaged over all batch sizes and lengths{suffix}", fontsize=10)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out / "speedup.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # 3. Peak memory vs sequence length for the largest model
    largest = models[-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for dtype, color in DTYPE_COLORS.items():
        points = sorted((r["seq_len"], r["peak_memory_mb"] / 1024) for r in rows
                        if r["model"] == largest and r["dtype"] == dtype and r["batch_size"] == batch_size)
        if points:
            ax.plot(*zip(*points), marker="o", color=color, linestyle=DTYPE_LINESTYLES[dtype], label=dtype.upper())
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_xscale("log", base=2)
    ax.set_title(f"ESM-2 {largest} peak GPU memory, batch {batch_size}{suffix}", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "memory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))
    return saved

VARIANT_COLORS = {
    "compile": "#2563eb",
    "compile-cudagraphs": "#7c3aed",
    "sdpa-efficient": "#16a34a",
    "sdpa-cudnn": "#ea580c",
    "sdpa-flash": "#0891b2",
    "sdpa-math": "#64748b",
}

# sdpa-efficient runs exactly the same kernels as eager mode, so its speed-up should be 1.0.
# Any deviation is measurement noise (for example GPU clock changes on a laptop).
NOISE_CONTROL = "sdpa-efficient"
NOISE_TOLERANCE = 0.1

def noisy_shapes(rows: List[dict]) -> set:
    """(model, batch, length) whose control variant deviates from 1.0 by more than the tolerance"""
    return {(r["model"], r["batch_size"], r["seq_len"]) for r in rows
            if r["variant"] == NOISE_CONTROL and r.get("speedup_vs_eager")
            and abs(r["speedup_vs_eager"] - 1) > NOISE_TOLERANCE}

def optimization_table(rows: List[dict], model: str) -> str:
    """Speed-up of each variant over eager mode for one model, one row per batch size × length.

    Shapes where the noise control deviates from 1.0 are marked with a warning sign.
    """
    noisy = noisy_shapes(rows)
    rows = [r for r in rows if r["model"] == model]
    variants = [v for v in VARIANT_COLORS if any(r["variant"] == v for r in rows)]
    shapes = sorted({(r["batch_size"], r["seq_len"]) for r in rows})
    index = {(r["batch_size"], r["seq_len"], r["variant"]): r for r in rows}

    lines = ["| Batch × length | Eager | " + " | ".join(f"`{v}`" for v in variants) + " |",
             "|" + "---|" * (2 + len(variants))]
    for batch, length in shapes:
        eager = index.get((batch, length, "eager"))
        cells = [f"{eager['latency_ms']:.1f} ms" if eager else "-"]
        for variant in variants:
            row = index.get((batch, length, variant))
            cells.append(f"{row['speedup_vs_eager']:.2f}×" if row and row.get("speedup_vs_eager") else "n/a")
        marker = " ⚠" if (model, batch, length) in noisy else ""
        lines.append(f"| {batch} × {length}{marker} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def batching_table(rows: List[dict]) -> str:
    """Time, throughput and padding efficiency per batching strategy"""
    lines = ["| Strategy | Batches | Total time | Proteins/s | Padding efficiency | Speed-up |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['strategy']}` | {r['batches']} | {r['total_s']:.1f} s | {r['sequences_per_s']:.1f} | "
                     f"{r['padding_efficiency']:.0%} | {r['speedup_vs_unsorted']:.2f}× |")
    return "\n".join(lines)

def plot_optimizations(rows: List[dict], output_dir: str, title_suffix: Optional[str] = None) -> str:
    """Speed-up of each variant over eager, one panel per model; returns the saved path"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    variants = [v for v in VARIANT_COLORS if any(r["variant"] == v for r in rows)]
    noisy = noisy_shapes(rows)
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.4), squeeze=False)

    for ax, model in zip(axes[0], models):
        model_rows = [r for r in rows if r["model"] == model]
        shapes = sorted({(r["batch_size"], r["seq_len"]) for r in model_rows})
        index = {(r["batch_size"], r["seq_len"], r["variant"]): r for r in model_rows}
        width = 0.8 / len(variants)
        for i, variant in enumerate(variants):
            values = [(index.get((b, l, variant)) or {}).get("speedup_vs_eager") or 0 for b, l in shapes]
            ax.bar([x + i * width for x in range(len(shapes))], values, width, color=VARIANT_COLORS[variant], label=variant)
        ax.axhline(1, color="#0f172a", linestyle="--", linewidth=1)
        labels = [f"{b} × {l}" + ("\n(noisy)" if (model, b, l) in noisy else "") for b, l in shapes]
        ax.set_xticks([x + width * (len(variants) - 1) / 2 for x in range(len(shapes))], labels)
        ax.set_xlabel("Batch size × sequence length")
        ax.set_title(f"ESM-2 {model}")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Speed-up over eager")
    axes[0][-1].legend(fontsize=8)
    suffix = f"\n{title_suffix}" if title_suffix else ""
    fig.suptitle(f"Framework-level optimizations, FP16 (above the dashed line is faster){suffix}", fontsize=11)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "optimizations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
