# ProteinPerf

[![CI](https://github.com/salmansystech/ProteinPerf/actions/workflows/ci.yml/badge.svg)](https://github.com/salmansystech/ProteinPerf/actions/workflows/ci.yml)

**Profiling and optimizing protein language model inference on GPUs.**

ProteinPerf measures where the time goes when running [ESM-2](https://github.com/facebookresearch/esm), Meta's family of protein language models, and then makes it faster, step by step: from careful baselines and operator-level profiling, through framework-level optimization (`torch.compile`, mixed precision, attention kernels), to custom GPU kernels written in Triton.

Every optimization is measured against the same baseline on the same hardware, with the results, charts and methodology in this repository.

## Why

Protein language models like ESM-2 are used across drug discovery and biology to turn amino acid sequences into embeddings that predict structure, function and mutation effects. Screening millions of proteins makes inference cost a real bottleneck, so understanding and reducing it matters:

- **How fast is it, really?** Latency, throughput and memory across model sizes (8M to 650M parameters), precisions, batch sizes and sequence lengths.
- **Where does the time go?** GPU time broken down into matrix multiplies, attention, LayerNorm, activations, elementwise and memory operations.
- **What actually helps?** Each optimization isolated and measured, so its real effect is known.

## Features

| | |
|---|---|
| **Benchmarking** | GPU timing with CUDA events, warm-up runs, median and p90 latency, throughput in sequences and tokens per second, peak GPU memory, and out-of-memory handling |
| **Profiling** | PyTorch Profiler breakdown by operator category, the slowest operators, and a full trace you can open in Perfetto |
| **Reporting** | Throughput, speed-up and memory charts, plus Markdown summary tables |
| **Reproducibility** | Every result records the GPU, driver, CUDA, PyTorch and Transformers versions and the benchmark settings |
| **Realistic inputs** | Synthetic proteins with natural amino acid composition (UniProtKB/Swiss-Prot), or your own FASTA file |
| **Tested** | Unit tests with a tiny in-memory ESM model, so CI runs without a GPU or downloads |

## Quick start

Requires Python 3.10+ and, for GPU results, an NVIDIA GPU with a recent driver.

```bash
git clone https://github.com/salmansystech/ProteinPerf.git
cd ProteinPerf

python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS / Linux

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that PyTorch can see your GPU:

```bash
python -m proteinperf info
```

Then measure, profile and chart:

```bash
# Latency, throughput and memory for several models and settings
python -m proteinperf benchmark --models 8M,35M,150M --dtypes fp32,fp16,bf16 --batch-sizes 1,8,32 --seq-lens 128,256,512

# Where GPU time goes in one configuration
python -m proteinperf profile --model 150M --dtype fp16 --batch-size 8 --seq-len 512

# Charts and a summary table from the benchmark results
python -m proteinperf report --input results/benchmark.json
```

Model weights download automatically from Hugging Face the first time each model is used (31 MB for 8M, up to 2.6 GB for 650M).

Full setup instructions, including WSL2 for the Triton work, are in the [Developer Guide](docs/DEVELOPER_GUIDE.md).

## Commands

| Command | What it does | Key options |
|---|---|---|
| `info` | GPU, compute capability, CUDA, BF16 support and library versions | |
| `benchmark` | Times every combination of the given settings and saves CSV and JSON results | `--models`, `--dtypes`, `--batch-sizes`, `--seq-lens`, `--warmup`, `--iters`, `--output` |
| `profile` | GPU time by operator category and the slowest operators, plus a trace file | `--model`, `--dtype`, `--batch-size`, `--seq-len`, `--iters` |
| `report` | Throughput, speed-up and memory charts, and a Markdown table | `--input`, `--output-dir`, `--batch-size`, `--seq-len` |

Run `python -m proteinperf <command> --help` for every option.

| Model | Hugging Face name | Layers | Hidden size |
|---|---|---|---|
| `8M` | `facebook/esm2_t6_8M_UR50D` | 6 | 320 |
| `35M` | `facebook/esm2_t12_35M_UR50D` | 12 | 480 |
| `150M` | `facebook/esm2_t30_150M_UR50D` | 30 | 640 |
| `650M` | `facebook/esm2_t33_650M_UR50D` | 33 | 1280 |
| `3B` | `facebook/esm2_t36_3B_UR50D` | 36 | 2560 |

Precisions: `fp32`, `fp16`, `bf16`. Sequences can be up to 1022 residues long.

## Results

<!-- RESULTS:START -->
### Baseline (phase 1)

Measured on an **NVIDIA GeForce RTX 3070 Laptop GPU** (8 GB, Ampere, compute capability 8.6) with PyTorch 2.11, CUDA 12.8 and Transformers 5.21 on Windows 11. 144 configurations: 4 model sizes × 3 precisions × batch sizes 1, 8 and 32 × sequence lengths 128, 256, 512 and 1000. Each result is the median of 10 timed runs after 3 warm-up runs. Raw data: [`results/baseline.csv`](results/baseline.csv) and [`results/baseline.json`](results/baseline.json).

**Throughput and memory at batch 8, length 512**

| Model | FP32 seq/s | FP16 seq/s | BF16 seq/s | FP32 memory | FP16 memory | BF16 memory |
|---|---|---|---|---|---|---|
| ESM-2 8M | 232.5 | 610.8 | 609.6 | 0.13 GB | 0.07 GB | 0.07 GB |
| ESM-2 35M | 55.4 | 150.0 | 153.6 | 0.28 GB | 0.15 GB | 0.15 GB |
| ESM-2 150M | 16.7 | 41.1 | 38.4 | 0.76 GB | 0.38 GB | 0.38 GB |
| ESM-2 650M | 5.4 | 14.2 | 13.7 | 2.89 GB | 1.41 GB | 1.41 GB |

![Throughput by model, precision and sequence length](results/charts/throughput.png)

| | |
|---|---|
| ![Half-precision speed-up over FP32](results/charts/speedup.png) | ![ESM-2 650M peak GPU memory](results/charts/memory.png) |

**Where GPU time goes** (batch 8, length 512; full profiles in [`results/profiles/`](results/profiles/))

| Category | 150M FP32 | 150M FP16 | 650M FP16 |
|---|---|---|---|
| Matmul | 163.7 ms (49%) | 53.7 ms (43%) | 214.1 ms (59%) |
| Attention | 94.1 ms (28%) | 14.9 ms (12%) | 20.1 ms (6%) |
| Elementwise | 59.3 ms (18%) | 36.4 ms (29%) | 82.5 ms (23%) |
| Memory | 5.5 ms (2%) | 14.1 ms (11%) | 31.0 ms (9%) |
| Activation (GELU) | 7.8 ms (2%) | 4.0 ms (3%) | 8.8 ms (2%) |
| LayerNorm | 4.6 ms (1%) | 2.5 ms (2%) | 4.8 ms (1%) |
| **Total GPU time** | **335.0 ms** | **125.6 ms** | **361.3 ms** |

### Key findings

1. **Half precision is 2.2–2.7× faster and halves memory.** Averaged over all batch sizes and lengths, FP16 is 2.2× faster on the smaller models and 2.7× faster on 650M, where the matrix multiplies dominate and run on tensor cores. FP16 and BF16 perform almost identically on this Ampere GPU, so BF16's wider numeric range comes at almost no cost.
2. **FP32 doesn't fit the largest workloads.** ESM-2 650M in FP32 at batch 32 × 1000 residues exceeded the 8 GB GPU. On Windows the NVIDIA driver spilled into system RAM instead of failing, making it about 11× slower than expected, so that result is flagged and excluded. The same workload in FP16 needs 2.8 GB.
3. **Half precision moves the bottleneck to elementwise operations.** Going from FP32 to FP16 on 150M, matmuls get 3.0× faster and attention 6.3× faster (tensor cores and the fused memory-efficient attention kernel), but elementwise operations only 1.6× faster. They become 29% of GPU time. Most come from the rotary position embeddings (multiply, negate and concatenate on queries and keys in every layer) and the unfused exact GELU (`erf` plus several multiplies and adds). These are memory-bound and are the main target for **kernel fusion** in phase 4.
4. **FP16 adds extra copies.** The FP16 model runs 192 copy operations per forward pass versus 9 in FP32 (about 6 more per layer), costing 9 ms, most likely data type conversions. Removing them is another phase 4 target.
5. **The GPU is often waiting for the CPU.** For 150M FP16, kernels are busy for 126 ms of the 195 ms forward pass: about a third of the time the GPU sits idle while Python launches hundreds of small kernels. At batch size 1, the 8M model takes 5–7 ms whatever the sequence length, because launching kernels costs more than computing them. This launch overhead is what `torch.compile` and CUDA graphs reduce, the focus of phase 3.
6. **Longer sequences cost more per token.** ESM-2 650M in FP16 drops from 10,800 tokens/s at length 128 to about 7,500 at lengths 512–1000, as attention cost grows with the square of the length.

**About the hardware:** under sustained load, this laptop's GPU ran at about 800–1,700 MHz, below its 2,100 MHz maximum, because of the laptop's power and cooling limits. The numbers reflect this real-world laptop, not a desktop RTX 3070.
<!-- RESULTS:END -->

## Project structure

```
ProteinPerf/
├── proteinperf/
│   ├── cli.py            # Command-line interface (python -m proteinperf)
│   ├── models.py         # ESM-2 model registry and loading
│   ├── sequences.py      # Synthetic proteins and FASTA reading
│   ├── benchmark.py      # Latency, throughput and memory measurement
│   ├── profiling.py      # PyTorch Profiler breakdown by operator category
│   ├── report.py         # Charts and summary tables
│   └── environment.py    # Hardware and software details for each result
├── tests/                # Unit tests (run on CPU with a tiny ESM model)
├── results/
│   ├── baseline.csv/.json # Raw benchmark data with environment metadata
│   ├── charts/           # Throughput, speed-up and memory charts
│   └── profiles/         # GPU time breakdowns per configuration
├── docs/
│   ├── ARCHITECTURE.md   # How the system works and how it measures
│   └── DEVELOPER_GUIDE.md
└── .github/workflows/    # CI
```

## Roadmap

| Phase | Status | Focus |
|---|---|---|
| 1. Baseline and profiling | ✅ Done | Benchmarks across models, precisions, batch sizes and lengths; operator-level profiling |
| 2. Compute and memory analysis | Planned | FLOPs and bytes per layer, arithmetic intensity, roofline charts |
| 3. Framework-level optimization | Planned | `torch.compile` (Inductor), attention backends, length-bucketed batching |
| 4. Custom Triton kernels | Planned | Fused LayerNorm and GELU, rotary embeddings, a tiled matmul, compared with cuBLAS |
| 5. Profiling assistant | Planned | An LLM agent that explains profiler results and suggests optimizations |
| Future | | AMD GPUs (ROCm), multi-GPU scaling, protein structure models |

Kernels will be written in [Triton](https://github.com/triton-lang/triton), which compiles for both NVIDIA (CUDA) and AMD (ROCm) GPUs.

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, measurement and profiling methodology, design decisions |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Setup on Windows, Linux and WSL2, testing, code conventions, extending the project, getting reliable benchmark numbers, troubleshooting |

## License

MIT License. See [LICENSE](LICENSE).

ESM-2 models are released by Meta under the MIT License.

## Author

**Salman Khan** - [GitHub](https://github.com/salmansystech)
