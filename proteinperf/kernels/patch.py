"""Swap ESM-2's rotary embedding, GELU and feed-forward layer for the Triton kernels.

Hugging Face's EsmSelfAttention and EsmIntermediate call module-level functions
(apply_rotary_pos_emb, gelu) and their own forward methods at run time, so replacing
those attributes inside a context manager changes what the model runs without editing
its code, and restores the original afterwards.
"""
from contextlib import contextmanager
from typing import Iterable

KERNELS = {
    "rotary": "Fused rotary position embedding for queries and keys",
    "gelu": "Fused exact GELU",
    "ffn": "Feed-forward up-projection with bias and GELU fused into the matmul",
}

def _ffn_forward(self, hidden_states):
    from proteinperf.kernels.matmul import linear
    return linear(hidden_states, self.dense.weight, self.dense.bias, gelu=True)

@contextmanager
def esm_triton_kernels(kernels: Iterable[str]):
    """Run ESM-2 with the given Triton kernels: any of 'rotary', 'gelu', 'ffn'"""
    from transformers.models.esm import modeling_esm

    kernels = set(kernels)
    unknown = kernels - set(KERNELS)
    if unknown:
        raise ValueError(f"Unknown kernels {sorted(unknown)}. Choose from: {', '.join(KERNELS)}")

    originals = {
        "apply_rotary_pos_emb": modeling_esm.apply_rotary_pos_emb,
        "gelu": modeling_esm.gelu,
        "ffn": modeling_esm.EsmIntermediate.forward,
    }
    try:
        if "rotary" in kernels:
            from proteinperf.kernels.rotary import rotary_embedding
            modeling_esm.apply_rotary_pos_emb = rotary_embedding
        if "gelu" in kernels:
            from proteinperf.kernels.gelu import gelu
            modeling_esm.gelu = gelu
        if "ffn" in kernels:
            modeling_esm.EsmIntermediate.forward = _ffn_forward
        yield
    finally:
        modeling_esm.apply_rotary_pos_emb = originals["apply_rotary_pos_emb"]
        modeling_esm.gelu = originals["gelu"]
        modeling_esm.EsmIntermediate.forward = originals["ffn"]
