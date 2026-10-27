"""Framework-level optimizations of a model: compilation, CUDA graphs and attention backends"""
from typing import Dict
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ATTENTION_BACKENDS = {
    "sdpa-efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa-cudnn": SDPBackend.CUDNN_ATTENTION,
    "sdpa-flash": SDPBackend.FLASH_ATTENTION,
    "sdpa-math": SDPBackend.MATH,
}

VARIANTS = {
    "eager": "PyTorch eager mode (the baseline)",
    "compile": "torch.compile with Inductor: fuses operators into Triton kernels",
    "compile-cudagraphs": "torch.compile mode='reduce-overhead': adds CUDA graphs to cut kernel launch overhead",
    **{name: f"Attention forced to the {backend.name} kernel" for name, backend in ATTENTION_BACKENDS.items()},
    "triton-rotary": "Custom Triton kernel: fused rotary embedding",
    "triton-gelu": "Custom Triton kernel: fused GELU",
    "triton-ffn": "Custom Triton kernel: feed-forward matmul with fused bias and GELU",
    "triton": "All custom Triton kernels: fused rotary embedding and feed-forward matmul with fused GELU",
}

TRITON_VARIANTS = {
    "triton-rotary": ("rotary",),
    "triton-gelu": ("gelu",),
    "triton-ffn": ("ffn",),
    "triton": ("rotary", "ffn"),
}

class AttentionBackendModel(torch.nn.Module):
    """Runs a model with scaled_dot_product_attention restricted to one backend"""

    def __init__(self, model: torch.nn.Module, backend: SDPBackend):
        super().__init__()
        self.model = model
        self.backend = backend

    def forward(self, **inputs):
        with sdpa_kernel(self.backend):
            return self.model(**inputs)

class TritonKernelModel(torch.nn.Module):
    """Runs an ESM-2 model with some of its operations replaced by custom Triton kernels"""
    
    def __init__(self, model: torch.nn.Module, kernels):
        super().__init__()
        self.model = model
        self.kernels = tuple(kernels)
    
    def forward(self, **inputs):
        from proteinperf.kernels.patch import esm_triton_kernels
        with esm_triton_kernels(self.kernels):
            return self.model(**inputs)

def build_variant(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """The model wrapped or compiled for one optimization variant"""
    if name == "eager":
        return model
    if name == "compile":
        return torch.compile(model)
    if name == "compile-cudagraphs":
        return torch.compile(model, mode="reduce-overhead")
    if name in ATTENTION_BACKENDS:
        return AttentionBackendModel(model, ATTENTION_BACKENDS[name])
    if name in TRITON_VARIANTS:
        return TritonKernelModel(model, TRITON_VARIANTS[name])
    raise ValueError(f"Unknown variant '{name}'. Choose from: {', '.join(VARIANTS)}")

def output_difference(reference: torch.Tensor, output: torch.Tensor) -> Dict[str, float]:
    """How far an optimized model's embeddings are from the baseline's"""
    reference = reference.float()
    output = output.float()
    cosine = torch.nn.functional.cosine_similarity(reference, output, dim=-1)
    return {
        "max_abs_diff": (reference - output).abs().max().item(),
        "mean_abs_diff": (reference - output).abs().mean().item(),
        "min_cosine_similarity": cosine.min().item(),
    }
