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
10. [Framework-level optimizations (phase 3)](#10-framework-level-optimizations-phase-3)
11. [Custom Triton kernels (phase 4)](#11-custom-triton-kernels-phase-4)
12. [Roofline analysis (phase 2)](#12-roofline-analysis-phase-2)
13. [Profiling assistant (phase 5)](#13-profiling-assistant-phase-5)
14. [Future work](#14-future-work)

---

## 1. System overview

ProteinPerf is a pipeline in four stages: **inputs** (models and proteins), **model variants** (the same ESM-2 model run in different ways), **measurement** (one set of timing, profiling and roofline code for every variant) and **analysis** (results files, then charts, tables, findings and the assistant). Every stage communicates through files in `results/`, so analysis never depends on in-memory state and can be re-run without a GPU.

```mermaid
flowchart LR
    CLI["CLI<br/>python -m proteinperf"]

    subgraph Inputs
        Seq["sequences.py<br/>synthetic proteins / FASTA"]
        Models["models.py<br/>ESM-2 from Hugging Face"]
    end

    subgraph Variants["Model variants"]
        Eager["Eager PyTorch<br/>(baseline)"]
        Var["variants.py<br/>torch.compile · CUDA graphs<br/>attention backends"]
        Kern["kernels/<br/>Triton rotary · GELU · matmul"]
    end

    subgraph Measurement
        Bench["benchmark.py<br/>latency · throughput · memory"]
        Prof["profiling.py<br/>time by operator category"]
        Batch["batching.py<br/>real mixed-length proteins"]
        Roof["roofline.py<br/>FLOPs · bytes · efficiency"]
        KBench["kernels/bench.py<br/>kernel microbenchmarks"]
    end

    Env["environment.py<br/>GPU · CUDA · versions"]

    subgraph Results["results/"]
        Data[("*.csv / *.json<br/>with metadata")]
        Profiles[("profiles/*.txt<br/>traces/*.json")]
    end

    subgraph Analysis
        Report["report.py<br/>charts · tables"]
        Find["findings.py<br/>rule-based conclusions"]
        Agent["assistant.py<br/>LLM agent (Ollama)"]
    end

    Charts[("charts/*.png")]
    User(["Answers in<br/>plain English"])

    CLI --> Seq & Models
    Models --> Eager & Var & Kern
    Seq --> Bench & Prof & Batch
    Eager & Var & Kern --> Bench
    Eager --> Prof & Batch
    Kern --> KBench
    Kern --> Roof
    Env --> Data
    Bench & Batch & Roof & KBench --> Data
    Prof --> Profiles
    Data --> Report --> Charts
    Data & Profiles --> Find
    Find --> Agent
    Data & Profiles --> Agent
    Agent --> User
```

There are nine commands, each a thin layer over the modules:

| Command | Phase | Flow |
|---|---|---|
| `info` | | `environment.py` → console |
| `benchmark` | 1 | for each model and precision: `models.py` → for each length and batch size: `sequences.py` → `benchmark.py` → CSV + JSON with environment metadata |
| `profile` | 1 | `models.py` + `sequences.py` → `profiling.py` → category breakdown + Perfetto trace |
| `roofline` | 2 | measure peak TFLOPS and bandwidth → `roofline.py` times each layer operation → whole model with `benchmark.py` → `roofline.csv/.json` |
| `optimize` | 3, 4 | `models.py` → eager reference → each variant from `variants.py` (including the Triton variants) → `benchmark.py` + accuracy check → CSV + JSON |
| `batching` | 3 | FASTA → `batching.py` strategies → total time, padding efficiency, memory → CSV + JSON |
| `kernels` | 4 | measured bandwidth → `kernels/bench.py` times each kernel against eager and `torch.compile` → CSV + JSON |
| `report` | all | every results JSON that exists → `report.py` → PNG charts + Markdown tables |
| `ask` | 5 | question → `assistant.py` tool-calling loop over `results/` (or `findings.py` alone with `--no-llm`) → answer |

Models are loaded once per model and precision, then reused for every sequence length, batch size and variant, because loading weights is far slower than any measurement.

### How the phases build on each other

```mermaid
flowchart LR
    P1["Phase 1<br/>Baseline + profile<br/>where does time go?"] --> P2["Phase 2<br/>Roofline<br/>how much is necessary?"]
    P1 --> P3["Phase 3<br/>Framework optimizations<br/>compile · graphs · batching"]
    P2 --> P4["Phase 4<br/>Triton kernels<br/>rotary · GELU · matmul"]
    P3 --> P4
    P1 & P2 & P3 & P4 --> P5["Phase 5<br/>Profiling assistant<br/>explains all results"]
```

Phase 1's profile showed elementwise operations growing to 29% of FP16 time; phase 2's roofline showed rotary embedding and GELU running at 6–18% of their limit; phase 4 replaced exactly those operations. Phase 3's `optimize` command and noise control are reused unchanged to measure the phase 4 kernels inside the model.

---

## 2. Modules

| Module | Responsibility | Key functions |
|---|---|---|
| `models.py` | ESM-2 registry, precisions and loading | `ESM2_MODELS`, `DTYPES`, `load_model()` |
| `sequences.py` | Benchmark inputs | `synthetic_sequences()`, `read_fasta()` |
| `benchmark.py` | Timing, throughput and memory | `time_function()`, `benchmark_inference()`, `percentile()` |
| `profiling.py` | Operator-level breakdown | `profile_inference()`, `summarize()`, `categorize()` |
| `report.py` | Charts and tables | `speedups()`, `markdown_table()`, `plot_results()`, `plot_roofline()`, `utilization_table()` |
| `environment.py` | Reproducibility metadata | `environment_info()` |
| `variants.py` | Framework-level optimizations (phase 3) | `build_variant()`, `output_difference()`, `VARIANTS` |
| `batching.py` | Batching strategies for mixed-length proteins (phase 3) | `make_batches()`, `padding_stats()`, `run_dataset()` |
| `kernels/` | Custom Triton kernels, model patching and microbenchmarks (phase 4) | `rotary_embedding()`, `gelu()`, `linear()`, `esm_triton_kernels()`, `benchmark_kernels()` |
| `roofline.py` | FLOPs, bytes and roofline efficiency of each layer operation (phase 2) | `layer_ops()`, `model_flops()`, `measure_layer()`, `measured_peak_tflops()` |
| `findings.py` | Rule-based conclusions from all results (phase 5) | `collect_findings()`, `parse_profile()` |
| `assistant.py` | Profiling assistant: LLM agent with read-only tools (phase 5) | `ask()`, `ResultsTools`, `ollama_chat()` |
| `cli.py` | Argument parsing and orchestration | `main()` and one `cmd_*` function per command |

```mermaid
flowchart TD
    cli[cli.py] --> models[models.py]
    cli --> sequences[sequences.py]
    cli --> benchmark[benchmark.py]
    cli --> profiling[profiling.py]
    cli --> report[report.py]
    cli --> environment[environment.py]
    cli --> variants[variants.py]
    cli --> batching[batching.py]
    cli --> kernels[kernels/]
    variants --> kernels
    cli --> roofline[roofline.py]
    roofline --> kernels
    cli --> assistant[assistant.py]
    assistant --> findings[findings.py]
    findings --> report
```

The measurement modules (`benchmark.py`, `profiling.py`) take any PyTorch model and any callable tokenizer. They don't depend on `models.py`, which is what lets the tests run them with a tiny model, and lets phases 3 and 4 measure compiled models and models with custom kernels with the same code.

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

**Equal lengths:** all sequences in a benchmark batch have the same length, so there's no padding. Each configuration measures exactly the work it names. Padding with real, mixed-length data is studied separately in phase 3 (see [section 10](#10-framework-level-optimizations-phase-3)).

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
| **Spill detection** | Windows can silently spill GPU memory into system RAM instead of raising an out-of-memory error. Reserved memory above 90% of the GPU's capacity flags the result as invalid. |
| **Releasing models between runs** | Hugging Face models contain reference cycles, so a deleted model stays in GPU memory until garbage collection runs. The benchmark calls `gc.collect()` before `torch.cuda.empty_cache()` and prints the memory still allocated, so a leak shows up immediately. |
| **Results saved continuously** | Results are written after every configuration, so a long run can be stopped without losing data. |

**Reported metrics**

| Metric | Definition |
|---|---|
| `latency_ms` | Median time for one forward pass of the whole batch |
| `latency_p90_ms` | 90th percentile of the same timings |
| `sequences_per_s` | batch size ÷ median latency |
| `tokens_per_s` | batch size × (length + 2) ÷ median latency |
| `peak_memory_mb` | Peak allocated GPU memory during the measured passes |
| `peak_reserved_mb` | Peak memory reserved by PyTorch's caching allocator (allocated plus cached blocks) |
| `exceeds_gpu_memory` | `True` when reserved memory passed 90% of the GPU's total memory. On Windows, the NVIDIA driver can then spill into system RAM instead of failing, which makes the timing meaningless. Such results are excluded from charts and tables. |

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

Every measuring command writes two files:

- **CSV:** one row per configuration, easy to open in a spreadsheet or pandas. Columns are the union of all rows' fields, since some rows have extra fields (for example GB/s for memory-bound kernels, TFLOPS for matmuls).
- **JSON:** the same rows plus metadata (timestamp, GPU, compute capability, CUDA, PyTorch and Transformers versions, and the command's settings, such as warm-up and iteration counts, seed, or the measured peaks for the roofline).

Results are saved after every configuration, so a long run can be stopped without losing data.

| File | Written by | One row per |
|---|---|---|
| `baseline.csv/.json` | `benchmark` | model × precision × batch size × length |
| `profiles/*.txt` | `profile` (saved from its output) | configuration, as a text report |
| `roofline.csv/.json` | `roofline` | model × layer operation |
| `optimizations.csv/.json` | `optimize` | model × shape × variant |
| `batching.csv/.json` | `batching` | batching strategy |
| `kernels.csv/.json` | `kernels` | model × kernel |
| `kernel_variants.csv/.json` | `optimize` with the Triton variants | model × shape × variant |

`report` reads every JSON file that exists and produces:

| Chart | Shows |
|---|---|
| `throughput.png` | Tokens per second vs sequence length, one panel per model, one line per precision |
| `speedup.png` | Average speed-up of FP16 and BF16 over FP32 per model |
| `memory.png` | Peak memory vs sequence length for the largest model |
| `roofline.png` | Every layer operation on a log-log roofline against the measured peak and bandwidth |
| `optimizations.png` | Speed-up of each framework-level variant over eager, per shape, with noisy shapes labeled |
| `kernels.png` | Whole-model speed-up of the Triton kernel variants and `torch.compile` |

It also prints Markdown tables for each, ready for the README: throughput and memory, roofline efficiency per operation, whole model vs roofline, model FLOPs utilization, speed-ups per variant (⚠ on noisy shapes), batching strategies and kernel microbenchmarks.

Results that spilled out of GPU memory (`exceeds_gpu_memory`) are excluded from every chart, table and finding by `valid_rows()`.

---

## 8. Testing strategy

| Test file | Covers |
|---|---|
| `test_sequences.py` | Lengths, alphabet, reproducibility, natural composition, FASTA parsing |
| `test_benchmark.py` | Percentiles, warm-up and iteration counts, throughput and token arithmetic |
| `test_profiling.py` | Operator categorization, kernel and bookkeeping events excluded, older PyTorch attribute names, a real profile run |
| `test_report.py` | Speed-up calculation, Markdown tables (including OOM cells and noise markers), chart files |
| `test_roofline.py` | FLOPs formulas, attention growing with length², total FLOPs vs the 2-FLOPs-per-weight rule, compute- vs memory-bound classification, roofline bounds, roofline and utilization tables |
| `test_variants.py` | Variant construction, attention backend wrapper, output comparison |
| `test_batching.py` | Every strategy covers each sequence once, sorting, token budgets, padding statistics, a dataset run |
| `test_kernels.py` | Each Triton kernel against PyTorch on the GPU (skipped without CUDA) |
| `test_patch.py` | Kernels are swapped in and always restored, variant construction |
| `test_findings.py` | Profile parsing, each finding rule, spilled results and noisy shapes excluded |
| `test_assistant.py` | Tools (listing, filtering, rounding, staying inside `results/`, bad arguments), the tool-calling loop with a scripted model, the round limit, `.env` loading |
| `test_cli.py` | Argument validation, saving mixed-field rows, `ask --no-llm`, unreachable Ollama |

- **No GPU, no downloads and no network:** `conftest.py` builds a 2-layer ESM model with random weights and a minimal tokenizer, so the real benchmark and profiler code runs on CPU in seconds. The assistant's tests replace the language model with a scripted fake.
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
| **Eager mode measured first, in the same session** | Laptop GPU clocks vary between runs, so every variant is compared with an eager run made minutes earlier, not with an old baseline. |
| **A noise control variant** | `sdpa-efficient` runs the same kernels as eager, so its deviation from 1.0× measures noise. Shapes off by more than 10% are flagged and left out of conclusions. |
| **Measured hardware limits for the roofline** | A laptop GPU never reaches its datasheet clocks under sustained load. Peaks measured on the same machine give an honest ceiling. |
| **Rules first, language model second** | Conclusions that must be right come from tested code (`findings.py`). The LLM only reads and explains, through read-only tools, and every tool call is shown. |
| **Ollama through the standard library** | Free local models or Ollama Cloud, no paid subscription and no extra dependency. The API key stays in a git-ignored `.env`. |

---

## 10. Framework-level optimizations (phase 3)

Phase 3 adds two modules that reuse the same benchmark code, so every result is directly comparable with eager mode.

### Model variants (`variants.py`)

`build_variant(model, name)` returns the model optimized in one way, without changing its weights:

| Variant | How | Targets |
|---|---|---|
| `eager` | The model as loaded | Baseline |
| `compile` | `torch.compile(model)`: TorchDynamo captures the graph, Inductor generates fused Triton kernels | Elementwise and memory operations that were separate kernels |
| `compile-cudagraphs` | `torch.compile(model, mode="reduce-overhead")`: also records the kernels into a CUDA graph and replays it with a single launch | CPU kernel-launch overhead |
| `sdpa-efficient`, `sdpa-cudnn`, `sdpa-flash`, `sdpa-math` | `AttentionBackendModel` runs the model inside `torch.nn.attention.sdpa_kernel(backend)` | Choice of attention kernel |

The `optimize` command runs them like this for every model and shape:

```mermaid
flowchart TD
    A["Eager reference output<br/>(computed once)"] --> L
    L{"For each variant<br/>(eager first)"} --> R["torch._dynamo.reset()"]
    R --> B["build_variant()"]
    B --> F["First call: timed separately<br/>(includes compilation)"]
    F --> D["Compare output with the<br/>eager reference"]
    D --> M["benchmark_inference()<br/>warm-up + timed runs"]
    M --> S["Speed-up vs eager<br/>measured in the same session"]
```

| Practice | Why |
|---|---|
| **Eager measured in the same run** | Laptop GPU clocks vary between sessions (phase 1 measured 195 ms for a configuration that later took 128 ms), so speed-ups are only valid against a baseline measured minutes apart under the same conditions. |
| **`torch._dynamo.reset()` before each variant** | Each variant and shape gets its own compiled graph, instead of reusing or recompiling one built for another shape. |
| **First call timed separately** | Compilation takes seconds to tens of seconds. It's a one-off cost reported as `first_call_s`, not part of steady-state latency. |
| **Accuracy checked for every variant** | A faster result only counts if the embeddings stay the same. Each variant's output is compared with eager: maximum and mean absolute difference, and the lowest per-token cosine similarity. |
| **Unavailable backends skipped** | FlashAttention isn't included in PyTorch's Windows build. Such variants are reported as not available instead of failing the run. |

### Batching strategies (`batching.py`)

Real proteins vary in length, and every sequence in a batch is padded to the longest one, so padding wastes compute. `make_batches()` implements three strategies:

| Strategy | Batches |
|---|---|
| `unsorted` | A fixed number of sequences, in dataset order |
| `sorted` | A fixed number of sequences, after sorting by length |
| `token-budget` | Sorted, then filled while `sequences × (longest length + 2)` stays within a token budget: many short proteins or a few long ones |

`padding_stats()` reports **padding efficiency**: real tokens divided by tokens actually computed. `run_dataset()` tokenizes all batches first, runs warm-up batches, then times embedding the entire dataset.

The input is `data/uniprot_human_sample.fasta`: 2,000 reviewed human proteins from UniProtKB/Swiss-Prot, randomly sampled (seed 0) from the 89% that fit ESM-2's 1022-residue limit. Lengths range from 8 to 1,021 residues (median 365).

---

## 11. Custom Triton kernels (phase 4)

Phase 1 profiling showed that after switching to FP16, elementwise and memory operations take about 40% of GPU time. They're memory-bound: each one reads a whole tensor from GPU memory and writes a whole tensor back, doing very little math. Phase 4 replaces the biggest of them with **fused** Triton kernels that read their inputs once, do all the math in registers, and write once.

```
proteinperf/kernels/
├── rotary.py   # fused rotary position embedding for q and k
├── gelu.py     # fused exact GELU
├── matmul.py   # tiled matmul with optional fused bias + GELU epilogue
├── patch.py    # swaps the kernels into Hugging Face's ESM-2 at run time
└── bench.py    # microbenchmarks against PyTorch eager and torch.compile
```

### What each kernel replaces

```mermaid
flowchart LR
    subgraph Before["Hugging Face ESM-2 (per layer, FP16)"]
        direction TB
        A1["q.float(), k.float()"] --> A2["rotate_half: slice, neg, cat (× 2)"]
        A2 --> A3["x·cos + rot(x)·sin (× 2)"] --> A4[".to(fp16) (× 2)"]
        B1["addmm (cuBLAS)"] --> B2["÷√2 → erf → +1 → ×x → ×0.5"]
    end
    subgraph After["With Triton kernels"]
        direction TB
        C1["rotary kernel:<br/>1 launch for q and k"]
        D1["matmul kernel with<br/>bias + GELU epilogue"]
    end
    Before --> After
```

| Kernel | Replaces | Design |
|---|---|---|
| **Rotary** (`rotary.py`) | Per layer: casts of q and k to FP32, `rotate_half` (slice, negate, concatenate), 4 multiplies, 2 adds and 2 casts back | Each program handles 32 (batch, head, position) rows. It loads both halves of q and k plus cos and sin, computes `x1·cos − x2·sin` and `x2·cos + x1·sin` in FP32, and stores the result in the input dtype. It reads strided inputs directly, so the transposed q and k views need no copy. |
| **GELU** (`gelu.py`) | Divide, `erf`, add, multiply, multiply | One flat elementwise kernel: load, compute `x·0.5·(1 + erf(x/√2))` in FP32, store. |
| **Matmul** (`matmul.py`) | cuBLAS `addmm` + the 5 GELU kernels in the feed-forward up-projection | Each program computes a `BLOCK_M × BLOCK_N` output tile, looping over `BLOCK_K` slices with `tl.dot` (tensor cores) and accumulating in FP32. Programs are ordered in groups of 8 row-tiles so concurrent programs share input tiles in the L2 cache. The bias and GELU are applied in the **epilogue**, while the result is still in registers, so the large activation is written to memory once instead of being written and read back by separate GELU kernels. `@triton.autotune` picks tile sizes, pipeline stages and warps per shape. |

### Swapping kernels into the model

Hugging Face's `EsmSelfAttention.forward` calls the module-level function `apply_rotary_pos_emb`, and `EsmIntermediate.forward` calls `gelu`. Both are looked up at call time, so `esm_triton_kernels()` (`patch.py`) replaces them inside a context manager and restores the originals afterwards, even if the model raises an error. The model's code and weights are unchanged.

`TritonKernelModel` (`variants.py`) runs a model inside that context, so each combination of kernels is just another variant for the `optimize` command:

| Variant | Kernels |
|---|---|
| `triton-rotary` | Rotary |
| `triton-gelu` | GELU |
| `triton-ffn` | Matmul with fused bias + GELU (replaces the whole up-projection, including GELU) |
| `triton` | Rotary + matmul with fused bias + GELU |

### Correctness and measurement

- **Correctness tests** (`tests/test_kernels.py`) compare each kernel with PyTorch on the GPU: the head sizes of every ESM-2 model (16, 24, 32 and 64, including a half size of 12 that isn't a power of two), FP16, BF16 and FP32, non-contiguous inputs, and matrix sizes that aren't multiples of the tile size. They're skipped on machines without a CUDA GPU, such as CI.
- **The rotary kernel is exact.** Like Hugging Face, it computes in FP32 and rounds once, so its output matches to the last bit up to fused multiply-add rounding.
- **GELU and the fused matmul differ by FP16 rounding.** They compute GELU in FP32, where Hugging Face computes it step by step in FP16, so the kernels are slightly more accurate.
- **Microbenchmarks** (`kernels` command, `bench.py`) use `triton.testing.do_bench`, which flushes the L2 cache between runs, and compare each kernel with the eager PyTorch code it replaces and with `torch.compile` of that same code.
- **Bandwidth** of the memory-bound kernels is reported as a percentage of the bandwidth measured by a large device-to-device copy: the practical ceiling on this GPU.
- **Whole-model results** use the same `optimize` command as phase 3, with the `sdpa-efficient` noise control.

---

## 12. Roofline analysis (phase 2)

The profiler (section 6) says where time goes. The roofline says how much of that time is necessary. `roofline.py` combines an analytical model of the work with measured hardware limits and measured operation times.

```mermaid
flowchart LR
    Spec["ModelSpec<br/>layers, hidden size, heads"] --> Ops["layer_ops()<br/>FLOPs and bytes per operation"]
    Ops --> AI["Arithmetic intensity<br/>FLOPs / byte"]
    Peak["measured_peak_tflops()<br/>4096³ FP16 matmul"] --> Roof["Roofline<br/>min(peak, intensity × bandwidth)"]
    BW["measured_bandwidth_gbs()<br/>512 MB copy"] --> Roof
    AI --> Roof
    Time["measure_layer()<br/>each operation timed with do_bench"] --> Eff["Efficiency<br/>achieved / attainable"]
    Roof --> Eff
```

### The work model (`layer_ops`)

For batch `b`, `n = b × (length + 2)` token rows, hidden size `h` and feed-forward size `4h`, one layer is 11 operations:

| Operation | FLOPs | Minimum bytes moved |
|---|---|---|
| LayerNorm (×2) | ≈ 8nh | read and write the activations |
| q, k, v projection | 2n·h·3h | input, weights, bias, output |
| Rotary embedding | 3 per element of q and k | read and write q and k |
| Attention | 4·b·tokens²·h | read q, k, v, write the output (fused, as the efficient and flash kernels are, so the tokens × tokens scores never reach memory) |
| Output projection | 2n·h·h | input, weights, bias, output |
| Residual add (×2) | nh | two inputs, one output |
| Feed-forward up and down | 2n·h·4h each | input, weights, bias, output |
| GELU | ≈ 8 per element | read and write the activation |

Bytes are the minimum: each input read once and each output written once. A real implementation that makes several passes over memory, such as Hugging Face's rotary code, can't reach that bound, which is exactly what the efficiency number exposes. The model's total FLOPs are within a few percent of the transformer rule of thumb of 2 FLOPs per weight per token plus attention, and a test checks this.

### Hardware limits

Peaks are measured, not taken from a datasheet, because a laptop GPU's clocks depend on its power and cooling limits:

- **Compute peak:** a 4096 × 4096 × 4096 matmul in the chosen precision with cuBLAS.
- **Bandwidth:** a 512 MB device-to-device copy (the same measurement as phase 4).
- **Ridge point:** peak / bandwidth, about 85 FLOPs per byte on the RTX 3070 Laptop GPU. Operations below it are memory-bound, operations above it compute-bound.

### Measurement

`measure_layer()` builds tensors with the model's real shapes and times each operation as eager PyTorch runs it (`F.linear`, `F.layer_norm`, `F.scaled_dot_product_attention`, and Hugging Face's own rotary and GELU code), using `triton.testing.do_bench`, which flushes the L2 cache between runs. For each operation it records the measured time, the roofline time `max(FLOPs / peak, bytes / bandwidth)`, the achieved TFLOPS and GB/s, and the efficiency.

The `roofline` command also times the whole model at the same shape. Comparing it with the layer count × (sum of measured operations) shows how much time falls outside the modeled operations, for example copies, masking and launch gaps.

### Model FLOPs utilization

`utilization_table()` in `report.py` divides `model_flops()` by each FP16 latency from the phase 1 baseline, giving whole-model TFLOPS and its share of the measured peak for every model, batch size and length, without running anything new.

---

## 13. Profiling assistant (phase 5)

The assistant answers questions about the results. It is split so that the part that must be right is deterministic and tested, and the language model only reads and explains.

```mermaid
flowchart LR
    Q["Question"] --> Agent["assistant.ask()<br/>tool-calling loop"]
    Agent <--> LLM["Ollama chat API<br/>local or cloud"]
    Agent <--> Tools["ResultsTools<br/>read-only"]
    Tools --> Files["results/*.json<br/>results/profiles/*.txt"]
    Tools --> Findings["findings.py<br/>rule-based conclusions"]
    Findings --> Files
    Agent --> A["Answer with<br/>measured numbers"]
```

### Rule-based findings (`findings.py`)

One rule function per results file turns rows into sentences, reusing `report.py` (`speedups`, `valid_rows`, `noisy_shapes`) so the conclusions match the tables:

| Rule | Finding |
|---|---|
| Baseline | Half-precision speed-up range per model; results that spilled out of GPU memory |
| Profiles | Largest category; elementwise + memory share above 25% suggests fusion |
| Roofline | Operations taking ≥10% of a layer at <50% of their roofline; whole model vs roofline |
| Optimizations, kernel variants | Best variant per shape, or a warning when the noise control deviates by more than 10% |
| Batching | Speed-up, padding efficiency and memory of each strategy |
| Kernels | Speed-up over eager and `torch.compile`, and bandwidth or TFLOPS reached |

`ask --no-llm` prints these directly.

### The agent (`assistant.py`)

- **Tools:** `list_results`, `get_findings`, `read_results(name, model)` and `read_profile(name)`. They are read-only. File names are checked against a fixed list, and profile names are reduced to a bare file name, so the model can't read outside `results/`. Results are rounded and capped at 60 rows per call to keep the context small.
- **Loop:** the question goes to the model with the tool definitions. While it returns tool calls, `ask()` runs them and sends the results back as `tool` messages, for at most 8 rounds. Each call is printed, so the user sees what the answer is based on.
- **Grounding:** the system prompt tells the model to read before answering, quote measured numbers with their model, precision and shape, mention noisy or missing results, and never invent measurements.
- **Provider:** Ollama's `/api/chat` with Python's standard library (`urllib`), so no extra dependency. `OLLAMA_BASE_URL`, `OLLAMA_MODEL` and `OLLAMA_API_KEY` come from the environment or a git-ignored `.env` file. A local Ollama model costs nothing; Ollama Cloud needs only a key.
- **Testing:** the chat function is a parameter of `ask()`, so tests replay scripted model replies (tool calls, then an answer) and check the messages sent, without a network.

---

## 14. Future work

| Area | Idea |
|---|---|
| Combined optimizations | Register the Triton kernels as custom ops so `torch.compile` can keep them inside its graphs, combining both speed-ups |
| Attention | A Triton attention kernel tuned for ESM-2's small heads (32 dimensions in 150M), where the efficient kernel reaches 38% of its roofline |
| AMD GPUs | Run the same benchmarks and Triton kernels on ROCm |
| Scale | Multi-GPU throughput and the 3B and 15B models |
