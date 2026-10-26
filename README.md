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
Baseline results are being collected and will be added here.
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
├── results/              # Benchmark results and charts
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
