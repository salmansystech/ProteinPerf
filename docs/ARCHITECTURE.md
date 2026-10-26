# Architecture

This document explains how ProteinPerf is built and, just as importantly, how it measures. Performance numbers are only useful if the method behind them is sound, so the measurement and profiling methodology is described in detail.

## Contents

1. [System overview](#1-system-overview)
2. [Modules](#2-modules)
3. [The model: ESM-2](#3-the-model-esm-2)
4. [Inputs](#4-inputs)
5. [Benchmark methodology](#5-benchmark-methodology)
6. [Profiling methodology](#6-profiling-methodology)
7. [Results and reporting](#7-results-and-reporting)
8. [Testing strategy](#8-testing-strategy)
9. [Design decisions](#9-design-decisions)
10. [Planned architecture](#10-planned-architecture)

---

## 1. System overview

```mermaid
flowchart LR
    CLI["CLI<br/>python -m proteinperf"]

    subgraph Inputs
        Seq["sequences.py<br/>synthetic proteins / FASTA"]
        Models["models.py<br/>ESM-2 from Hugging Face"]
    end

    subgraph Measurement
        Bench["benchmark.py<br/>latency · throughput · memory"]
        Prof["profiling.py<br/>time by operator category"]
    end

    Env["environment.py<br/>GPU · CUDA · versions"]
    Report["report.py<br/>charts · tables"]

    subgraph Outputs["results/"]
        CSV[("benchmark .csv / .json")]
        Trace[("traces/*.json")]
        Charts[("charts/*.png")]
    end

    CLI --> Seq
    CLI --> Models
    Seq --> Bench
    Models --> Bench
    Seq --> Prof
    Models --> Prof
    Bench --> CSV
    Env --> CSV
    Prof --> Trace
    CSV --> Report
    Report --> Charts
```

There are four commands, each a thin layer over the modules:

| Command | Flow |
|---|---|
| `info` | `environment.py` → console |
| `benchmark` | for each model and precision: `models.py` → for each length and batch size: `sequences.py` → `benchmark.py` → CSV + JSON with environment metadata |
| `profile` | `models.py` + `sequences.py` → `profiling.py` → category breakdown + Perfetto trace |
| `report` | results JSON → `report.py` → PNG charts + Markdown table |

Models are loaded once per model and precision, then reused for every sequence length and batch size, because loading weights is far slower than any measurement.

---

## 2. Modules

| Module | Responsibility | Key functions |
|---|---|---|
| `models.py` | ESM-2 registry, precisions and loading | `ESM2_MODELS`, `DTYPES`, `load_model()` |
| `sequences.py` | Benchmark inputs | `synthetic_sequences()`, `read_fasta()` |
| `benchmark.py` | Timing, throughput and memory | `time_function()`, `benchmark_inference()`, `percentile()` |
| `profiling.py` | Operator-level breakdown | `profile_inference()`, `summarize()`, `categorize()` |
| `report.py` | Charts and tables | `speedups()`, `markdown_table()`, `plot_results()` |
| `environment.py` | Reproducibility metadata | `environment_info()` |
| `cli.py` | Argument parsing and orchestration | `main()` and one `cmd_*` function per command |

```mermaid
flowchart TD
    cli[cli.py] --> models[models.py]
    cli --> sequences[sequences.py]
    cli --> benchmark[benchmark.py]
    cli --> profiling[profiling.py]
    cli --> report[report.py]
    cli --> environment[environment.py]
```

The measurement modules (`benchmark.py`, `profiling.py`) take any PyTorch model and any callable tokenizer. They don't depend on `models.py`, which is what lets the tests run them with a tiny model, and will let later phases measure compiled models and custom kernels with the same code.

---

## 3. The model: ESM-2

ESM-2 is a BERT-style transformer encoder trained on hundreds of millions of protein sequences. Each amino acid is one token.

```mermaid
flowchart LR
    S["Protein sequence<br/>MKTAYIAK..."] --> T["Tokenizer<br/>CLS + residues + EOS"]
    T --> E["Embedding"]
    E --> L["Transformer layer × N"]
    L --> N["Final LayerNorm"]
    N --> O["Per-residue embeddings<br/>(batch, length + 2, hidden)"]

    subgraph Layer["Each layer"]
        direction TB
        LN1["LayerNorm"] --> QKV["Q, K, V projections<br/>(matmul)"]
        QKV --> R["Rotary position<br/>embeddings"]
        R --> A["Attention<br/>(SDPA kernel)"]
        A --> P["Output projection + residual"]
        P --> LN2["LayerNorm"]
        LN2 --> F1["Feed-forward up<br/>(matmul)"]
        F1 --> G["GELU"]
        G --> F2["Feed-forward down<br/>(matmul) + residual"]
    end
```

- ProteinPerf loads it with Hugging Face Transformers (`EsmModel`) without the pooling head, which isn't used for embeddings.
- The output is the per-residue embedding, which is what downstream tasks use.
- Rotary embeddings are computed with several small elementwise operations (multiply, add, negate, concatenate), which shows up clearly in profiles.
- Attention runs through PyTorch's `scaled_dot_product_attention`, which picks a fused kernel (memory-efficient or Flash attention) depending on precision and hardware.

---

## 4. Inputs

**Synthetic sequences** (`synthetic_sequences(count, length, seed)`) draw residues from the 20 standard amino acids, weighted by their frequency in UniProtKB/Swiss-Prot. This gives realistic token statistics, exact control over length, and reproducibility through the seed.

**Equal lengths:** all sequences in a benchmark batch have the same length, so there's no padding. Each configuration measures exactly the work it names. Padding effects with real, mixed-length data will be studied in phase 3 (length-bucketed batching).

**FASTA files** (`read_fasta(path)`) load real proteins, joining multi-line sequences and skipping headers.

**Token count:** the tokenizer adds `<cls>` and `<eos>`, so a sequence of length *L* becomes *L + 2* tokens. Throughput in tokens per second counts these, because the model processes them.

---

## 5. Benchmark methodology

`benchmark_inference()` measures one forward pass over one batch.

```mermaid
sequenceDiagram
    participant B as Benchmark
    participant G as GPU

    B->>G: Tokenize and move inputs to the GPU (once, not timed)
    B->>G: Reset peak memory statistics
    loop Warm-up (default 3)
        B->>G: Forward pass
    end
    loop Timed iterations (default 10)
        B->>G: Record start event
        B->>G: Forward pass
        B->>G: Record end event
        B->>G: Synchronize
        G-->>B: Elapsed GPU time
    end
    B->>B: Median, p90, throughput, peak memory
```

| Practice | Why |
|---|---|
| **Warm-up runs** | The first passes include one-off costs: CUDA context setup, cuBLAS heuristics, memory allocator growth and kernel selection. These aren't part of steady-state inference. |
| **CUDA events** | GPU work runs asynchronously. CPU timers like `time.time()` measure when work was queued, not when it finished. Events are recorded on the GPU stream itself and measure actual GPU execution. |
| **Synchronize after each iteration** | Ensures the end event has completed before reading it, so each measurement is one complete forward pass. |
| **Median, not mean** | One slow run (a background process, a clock change) can skew a mean; the median is robust to outliers. **p90** is reported alongside to show variability. |
| **`torch.inference_mode()`** | Disables autograd tracking, as in real inference. |
| **Inputs prepared once** | Tokenization and host-to-device copies are excluded, so the measurement is model compute only. |
| **Peak memory** | `torch.cuda.max_memory_allocated()` after resetting statistics: the most memory the forward passes needed, including activations. |
| **Out-of-memory handling** | A configuration that doesn't fit is reported and skipped, the cache is cleared, and the run continues. |

**Reported metrics**

| Metric | Definition |
|---|---|
| `latency_ms` | Median time for one forward pass of the whole batch |
| `latency_p90_ms` | 90th percentile of the same timings |
| `sequences_per_s` | batch size ÷ median latency |
| `tokens_per_s` | batch size × (length + 2) ÷ median latency |
| `peak_memory_mb` | Peak allocated GPU memory during the measured passes |

On a CPU (used in tests), `time.perf_counter()` replaces CUDA events and memory is reported as 0.

---

## 6. Profiling methodology

`profile_inference()` runs warm-up passes, then several forward passes under the PyTorch Profiler with CPU and CUDA activity recording.

### Operators, not kernels

The profiler records two kinds of GPU-time events:

| Event | Example | Represents |
|---|---|---|
| PyTorch operator | `aten::addmm` | One operation in the model |
| GPU kernel | `ampere_sgemm_128x64_tn` | The CUDA kernel that operator launched |

A kernel's time also appears as its operator's device time, so counting both double counts the work. ProteinPerf counts only **operators**: CPU-side events with a namespace such as `aten::`. It uses each operator's **self** device time, so a parent operator (`aten::linear`) and its child (`aten::addmm`) aren't counted twice either. Profiler bookkeeping events (for example `Activity Buffer Request`) are excluded.

**Validation:** the profiler's total GPU time per pass closely matches the benchmark's measured latency for the same configuration. Any gap is CPU launch overhead, which the profiler doesn't count as GPU time.

### Categories

Operators are grouped so the breakdown is readable. Names are normalized first (`aten::_to_copy` → `to_copy`, `aten::add_` → `add`), then matched in order:

| Category | Matches |
|---|---|
| Attention | `scaled_dot_product*`, `*flash_attention*`, `*efficient_attention*`, `softmax` |
| Matmul | `mm`, `bmm`, `addmm`, `baddbmm`, `matmul`, `linear`, `*gemm*` |
| LayerNorm | `*layer_norm*` |
| Activation | `gelu`, `erf` (ESM-2's exact GELU uses `erf`), `relu`, `silu`, `tanh` |
| Elementwise | `add`, `mul`, `div`, `sub`, `neg`, `where`, `masked_fill`, `cos`, `sin`, `pow`, `sqrt`, `exp`, ... |
| Memory | `*copy*`, `to`, `contiguous`, `clone`, `cat`, `stack`, `index`, `embedding`, `fill`, `zero`, ... |
| Other | Everything else |

The categories map to optimization strategies: Matmul time responds to precision and tensor cores, Attention to kernel choice, and Elementwise, Activation, LayerNorm and Memory time to **kernel fusion**, the main target for phase 4.

### Trace files

The full trace is saved to `results/traces/` and can be opened at [ui.perfetto.dev](https://ui.perfetto.dev) to see every operator and kernel on a timeline, including gaps where the GPU waits for the CPU.

---

## 7. Results and reporting

`benchmark` writes two files:

- **CSV:** one row per configuration, easy to open in a spreadsheet or pandas.
- **JSON:** the same rows plus metadata (timestamp, GPU, compute capability, CUDA, PyTorch and Transformers versions, warm-up and iteration counts, and seed).

`report` reads the JSON and produces:

| Chart | Shows |
|---|---|
| `throughput.png` | Tokens per second vs sequence length, one panel per model, one line per precision |
| `speedup.png` | Average speed-up of FP16 and BF16 over FP32 per model |
| `memory.png` | Peak memory vs sequence length for the largest model |

It also prints a Markdown table of sequences per second and peak memory, ready for the README.

---

## 8. Testing strategy

| Test file | Covers |
|---|---|
| `test_sequences.py` | Lengths, alphabet, reproducibility, natural composition, FASTA parsing |
| `test_benchmark.py` | Percentiles, warm-up and iteration counts, throughput and token arithmetic |
| `test_profiling.py` | Operator categorization, kernel and bookkeeping events excluded, older PyTorch attribute names, a real profile run |
| `test_report.py` | Speed-up calculation, Markdown table (including OOM cells), chart files |
| `test_cli.py` | Argument validation |

- **No GPU and no downloads:** `conftest.py` builds a 2-layer ESM model with random weights and a minimal tokenizer, so the real benchmark and profiler code runs on CPU in seconds.
- **CI:** GitHub Actions installs the CPU build of PyTorch and runs the suite on Python 3.11 and 3.13 for every push and pull request.
- GPU results themselves are validated by cross-checking: the profiler total against benchmark latency, and token throughput against sequences per second.

---

## 9. Design decisions

| Decision | Reason |
|---|---|
| **Hugging Face Transformers for the baseline** | It's how most people run ESM-2, so improvements are measured against a realistic starting point. |
| **Measurement code independent of the model** | The same benchmark and profiler measure the baseline, compiled models and custom kernels, so comparisons are fair. |
| **Equal-length synthetic batches** | Isolates the effect of length and batch size; padding is a separate question studied separately. |
| **Median of CUDA-event timings** | The standard approach for GPU benchmarking: precise and robust to outliers. |
| **Operator self time for categories** | Avoids double counting between operators, sub-operators and kernels. |
| **Environment saved with every result** | Performance depends on GPU, driver and library versions; results without them can't be compared or reproduced. |
| **A tiny model for tests** | Tests exercise the real code paths without needing a GPU or network access. |
| **Triton for custom kernels** | Python-level GPU kernels that compile for both NVIDIA and AMD GPUs. |

---

## 10. Planned architecture

Later phases plug into the same measurement pipeline:

```mermaid
flowchart LR
    Base["Baseline<br/>Hugging Face ESM-2"] --> M["Same benchmark<br/>and profiler"]
    Compiled["torch.compile<br/>(Inductor)"] --> M
    Kernels["Model with<br/>Triton kernels"] --> M
    M --> R["Side-by-side results"]
    R --> Agent["Profiling assistant<br/>(LLM)"]
```

| Phase | New modules (planned) |
|---|---|
| 2. Compute and memory analysis | `roofline.py`: FLOPs and bytes per layer from the model configuration, arithmetic intensity, and roofline charts against the GPU's peak compute and bandwidth |
| 3. Framework-level optimization | Model variants: `torch.compile` modes, attention backends, and length-bucketed batching for mixed-length data |
| 4. Custom kernels | `kernels/` with Triton kernels (fused LayerNorm and GELU, rotary embeddings, tiled matmul), tested for numerical correctness against PyTorch, then swapped into the model |
| 5. Profiling assistant | An agent that reads profile reports and explains bottlenecks in plain language |
