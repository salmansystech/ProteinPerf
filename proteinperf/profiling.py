"""Find where inference time goes, using the PyTorch Profiler"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
from torch.profiler import ProfilerActivity, profile

# Groups of PyTorch operators, checked in order (first match wins).
# Each group has exact operator names and substrings that identify it.
OP_CATEGORIES = [
    ("Attention", (), ("scaled_dot_product", "flash_attention", "efficient_attention", "softmax")),
    ("Matmul", ("mm", "bmm", "addmm", "baddbmm", "matmul", "linear"), ("gemm",)),
    ("LayerNorm", (), ("layer_norm",)),
    ("Activation", ("gelu", "erf", "relu", "silu", "tanh"), ()),
    ("Elementwise", ("add", "mul", "div", "sub", "rsub", "neg", "where", "masked_fill",
                     "cos", "sin", "pow", "sqrt", "rsqrt", "exp"), ()),
    ("Memory", ("to", "contiguous", "clone", "cat", "stack", "index", "index_select", "embedding",
                "fill", "zero", "empty", "ne", "eq", "cumsum"), ("copy",)),
]

@dataclass
class ProfileReport:
    total_device_ms: float
    categories: Dict[str, float]                       # category -> device ms
    top_ops: List[dict] = field(default_factory=list)  # slowest operators
    trace_path: Optional[str] = None

    def category_shares(self) -> Dict[str, float]:
        """Fraction of device time per category, largest first"""
        if self.total_device_ms == 0:
            return {}
        shares = {name: ms / self.total_device_ms for name, ms in self.categories.items()}
        return dict(sorted(shares.items(), key=lambda item: item[1], reverse=True))

def categorize(op_name: str) -> str:
    """Category of a PyTorch operator name such as 'aten::addmm' or 'aten::add_'"""
    # Strip the namespace, private prefix and in-place suffix: 'aten::_to_copy' -> 'to_copy'
    name = op_name.split("::")[-1].lower().strip("_")
    for category, exact_names, substrings in OP_CATEGORIES:
        if name in exact_names or any(substring in name for substring in substrings):
            return category
    return "Other"

def _self_time_us(event, gpu: bool) -> float:
    """Self time of a profiler event in microseconds, on the GPU or the CPU"""
    # The GPU attribute was renamed in newer PyTorch versions
    attributes = ("self_device_time_total", "self_cuda_time_total") if gpu else ("self_cpu_time_total",)
    for attribute in attributes:
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0

def _is_operator(event) -> bool:
    """Whether a profiler event is a PyTorch operator such as 'aten::addmm'.
    
    The profiler also lists the GPU kernels each operator launches and its own
    bookkeeping events; counting those as well would double count GPU time.
    """
    device_type = getattr(event, "device_type", None)
    if device_type is not None and str(device_type) != "DeviceType.CPU":
        return False
    return "::" in event.key

def summarize(events, gpu: bool = True, top_n: int = 15) -> ProfileReport:
    """Group operator time into categories (GPU time, or CPU time when running without a GPU)"""
    categories = defaultdict(float)
    ops = []
    for event in events:
        if not _is_operator(event):
            continue
        device_ms = _self_time_us(event, gpu) / 1000
        if device_ms <= 0:
            continue
        category = categorize(event.key)
        categories[category] += device_ms
        ops.append({"op": event.key, "category": category, "device_ms": device_ms, "calls": event.count})

    ops.sort(key=lambda op: op["device_ms"], reverse=True)
    return ProfileReport(
        total_device_ms=sum(categories.values()),
        categories=dict(categories),
        top_ops=ops[:top_n],
    )

def profile_inference(model: torch.nn.Module, tokenizer, sequences: List[str], device: str = "cuda",
                      warmup: int = 3, iters: int = 5, trace_path: Optional[str] = None) -> ProfileReport:
    """Profile several forward passes and report GPU time per operator category"""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)

    def forward():
        with torch.inference_mode():
            model(**inputs)

    for _ in range(warmup):
        forward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(iters):
            forward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    report = summarize(prof.key_averages(), gpu=device.startswith("cuda"))
    # Report time per forward pass
    report.total_device_ms /= iters
    report.categories = {name: ms / iters for name, ms in report.categories.items()}
    for op in report.top_ops:
        op["device_ms"] /= iters
        op["calls"] //= iters

    if trace_path:
        prof.export_chrome_trace(trace_path)
        report.trace_path = trace_path
    return report
