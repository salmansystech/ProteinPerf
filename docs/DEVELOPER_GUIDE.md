# Developer Guide

Everything you need to set up ProteinPerf, run it, get reliable numbers, and extend it.

## Contents

1. [Requirements](#1-requirements)
2. [Setup on Windows](#2-setup-on-windows)
3. [Setup on Linux](#3-setup-on-linux)
4. [Setup on WSL2 (for Triton)](#4-setup-on-wsl2-for-triton)
5. [Running the tests](#5-running-the-tests)
6. [Using the CLI](#6-using-the-cli)
7. [Getting reliable benchmark numbers](#7-getting-reliable-benchmark-numbers)
8. [Code conventions](#8-code-conventions)
9. [Extending ProteinPerf](#9-extending-proteinperf)
10. [Continuous integration](#10-continuous-integration)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Requirements

| | Minimum | Used for development |
|---|---|---|
| Python | 3.10 | 3.13 |
| GPU | NVIDIA, compute capability 7.0+ (BF16 needs 8.0+) | GeForce RTX 3070 Laptop GPU, 8 GB, compute capability 8.6 |
| NVIDIA driver | Supporting CUDA 12.8 | 596.x |
| Disk | About 10 GB (PyTorch with CUDA is about 5 GB, models up to 2.6 GB each) | |
| OS | Windows 10/11, Linux, or WSL2 | Windows 11 |

The tests run on any machine with no GPU.

---

## 2. Setup on Windows

Native Windows supports everything in phase 1: benchmarking, profiling and reporting.

```powershell
git clone https://github.com/salmansystech/ProteinPerf.git
cd ProteinPerf

python -m venv venv
venv\Scripts\activate

# PyTorch with CUDA 12.8 (about 3 GB download)
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt -r requirements-dev.txt

python -m proteinperf info
```

`info` should show `cuda_available  True` and your GPU's name. If it shows `False`, see [Troubleshooting](#11-troubleshooting).

---

## 3. Setup on Linux

```bash
git clone https://github.com/salmansystech/ProteinPerf.git
cd ProteinPerf

python3 -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-dev.txt

python -m proteinperf info
```

---

## 4. Setup on WSL2 (for Triton)

Triton, and `torch.compile` which uses it, officially support Linux. On Windows, use WSL2: a Linux environment that can use your NVIDIA GPU directly. This is needed from phase 3 onward.

1. **Install a Linux distribution** in PowerShell (administrator rights may be needed the first time):
   ```powershell
   wsl --install -d Ubuntu
   ```
   Restart if asked, then open **Ubuntu** from the Start menu and create a user.
2. **Don't install a Linux NVIDIA driver.** The Windows driver provides GPU access to WSL2 automatically. Check it with:
   ```bash
   nvidia-smi
   ```
3. **Set up Python** inside Ubuntu:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip git
   ```
4. **Clone into the Linux file system** (for example `~/ProteinPerf`), not under `/mnt/c/`. File access across the Windows boundary is much slower.
5. Follow the [Linux setup](#3-setup-on-linux).

---

## 5. Running the tests

```bash
python -m pytest            # all tests
python -m pytest -v         # with test names
python -m pytest tests/test_profiling.py -k categorize   # a subset
```

The suite runs in a few seconds on CPU. `tests/conftest.py` provides:

| Fixture | What it is |
|---|---|
| `tiny_model` | A 2-layer ESM model (hidden size 32) with random weights, built in memory |
| `tokenizer` | `FakeEsmTokenizer`: adds CLS and EOS tokens and pads like the real ESM tokenizer, with no download |

Use these for any new test that needs a model, so tests stay fast and offline.

---

## 6. Using the CLI

### Benchmark

```bash
python -m proteinperf benchmark \
    --models 8M,35M,150M,650M \
    --dtypes fp32,fp16,bf16 \
    --batch-sizes 1,8,32 \
    --seq-lens 128,256,512,1000 \
    --output results/baseline.csv
```

Every combination is measured. The command prints one line per configuration and writes `baseline.csv` and `baseline.json` (with environment metadata). Configurations that don't fit in GPU memory are reported as out of memory and skipped.

| Option | Default | Notes |
|---|---|---|
| `--models` | `8M,35M,150M` | Any of `8M`, `35M`, `150M`, `650M`, `3B` |
| `--dtypes` | `fp32` | `fp32`, `fp16`, `bf16` |
| `--batch-sizes` | `1,8,32` | |
| `--seq-lens` | `128,256,512` | Maximum 1022 |
| `--warmup` | `3` | Untimed passes before measuring |
| `--iters` | `10` | Timed passes; the median is reported |
| `--seed` | `0` | Seed for synthetic sequences |
| `--output` | `results/benchmark.csv` | The JSON is written next to it |

### Profile

```bash
python -m proteinperf profile --model 150M --dtype fp16 --batch-size 8 --seq-len 512
```

Prints GPU time per forward pass by category and the slowest operators, and saves a trace to `results/traces/`. Open the trace at [ui.perfetto.dev](https://ui.perfetto.dev) (drag the file in) to see a timeline of every operator and kernel.

### Report

```bash
python -m proteinperf report --input results/baseline.json --batch-size 8 --seq-len 512
```

Saves `throughput.png`, `speedup.png` and `memory.png` to `results/charts/` and prints a Markdown table.

---

## 7. Getting reliable benchmark numbers

GPU timings vary with power, temperature and background activity, especially on laptops.

| Do | Why |
|---|---|
| **Plug in the charger** | On battery, laptop GPUs run at much lower power limits. |
| **Use a high-performance power mode** | Windows: Settings → System → Power → **Best performance**. Vendor tools (for example Armoury Crate or Lenovo Vantage) often have their own performance mode too. |
| **Close GPU-heavy apps** | Browsers with video, games and other ML jobs compete for the GPU. |
| **Don't run anything else during a benchmark** | Even CPU-only work adds launch delays, which matters most for small models. |
| **Compare within one run** | Configurations measured back to back under the same conditions are comparable. Across days, re-run the baseline. |
| **Look at p90 as well as the median** | A large gap means noisy measurements; re-run with more `--iters`. |
| **Watch the GPU** | `nvidia-smi -l 1` shows clock speed, temperature and power while a benchmark runs. Falling clocks mean thermal throttling. |

Small models (8M, 35M) at small batch sizes are dominated by kernel launch overhead rather than compute. That's a real effect, worth reporting, but those numbers are also the most sensitive to CPU activity.

---

## 8. Code conventions

- **Style:** PEP 8, type hints on public functions, a one-line docstring on every module and public function, and comments only where the reason isn't obvious.
- **Measurement code stays model-agnostic:** `benchmark.py` and `profiling.py` take a model and a tokenizer and must not import `models.py`.
- **Results are data:** commands write CSV/JSON first, and charts and tables are generated from those files, never from in-memory state.
- **Every measurement records its environment** (see `environment.py`).
- **No GPU in tests:** use the `tiny_model` and `tokenizer` fixtures.
- **Commits:** one logical change per commit, with a message explaining why.

---

## 9. Extending ProteinPerf

### Add a model

Add an entry to `ESM2_MODELS` in `proteinperf/models.py`:

```python
"15B": ModelSpec("facebook/esm2_t48_15B_UR50D", 48, 5120, "15B"),
```

It becomes available to every command, and `report.py` orders it by its position in `MODEL_ORDER`.

### Add a precision

Add it to `DTYPES` in `models.py` (and a color to `DTYPE_COLORS` in `report.py` so it appears in charts).

### Add a profiler category

Add a row to `OP_CATEGORIES` in `proteinperf/profiling.py`: a name, a tuple of exact operator names and a tuple of substrings. Order matters, because the first match wins. Add test cases to `test_categorize` in `tests/test_profiling.py`.

### Add a command

1. Write a `cmd_<name>(args)` function in `proteinperf/cli.py`.
2. Register it in `main()` with `commands.add_parser(...)` and `set_defaults(func=cmd_<name>)`.
3. Put the actual logic in a module, not in `cli.py`, and test that module.

### Benchmark a new model variant

`benchmark_inference()` and `profile_inference()` accept any `torch.nn.Module` whose forward takes the tokenizer's output, for example a compiled model:

```python
import torch
from proteinperf.models import load_model
from proteinperf.benchmark import benchmark_inference
from proteinperf.sequences import synthetic_sequences

model, tokenizer = load_model("150M", "fp16")
compiled = torch.compile(model)
sequences = synthetic_sequences(8, 512)
result = benchmark_inference(compiled, tokenizer, sequences, "150M-compiled", "fp16", warmup=10)
```

Use more warm-up runs for compiled models, because the first calls trigger compilation.

---

## 10. Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. Python 3.11 and 3.13 on Ubuntu.
2. Installs the CPU build of PyTorch (no GPU on GitHub's runners), then the requirements.
3. Runs `python -m pytest -v`.

GPU benchmarks are run manually on real hardware, and their results are committed to `results/`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `cuda_available False` | You installed the CPU build of PyTorch. Reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu128`. Also check `nvidia-smi` works. |
| `torch.cuda.OutOfMemoryError` | Use a smaller batch size, a shorter sequence length, a smaller model or FP16/BF16. The `benchmark` command skips such configurations automatically. |
| `ESM-2 supports sequences up to 1022 residues` | ESM-2 has 1026 positions, including the CLS and EOS tokens. Use lengths up to 1022. |
| Model download is slow or fails | Weights come from Hugging Face. Retry, or set `HF_HOME` to a drive with more space. Downloaded models are cached and reused. |
| Very different numbers between runs | See [Getting reliable benchmark numbers](#7-getting-reliable-benchmark-numbers): check power mode, charger and background apps. |
| `UserWarning: Profiler clears events at the end of each cycle` | Harmless. The profiler reports on the passes inside the profiling block, which is what ProteinPerf uses. |
| BF16 results missing or very slow | BF16 needs compute capability 8.0+ (Ampere or newer). Check `bf16_supported` in `python -m proteinperf info`. |
| `nvidia-smi` not found in WSL2 | Update the Windows NVIDIA driver and run `wsl --update` in PowerShell. Don't install a Linux driver inside WSL. |
