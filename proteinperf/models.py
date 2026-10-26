"""ESM-2 protein language models from Hugging Face"""
from dataclasses import dataclass
from typing import Dict, Tuple
import torch

@dataclass(frozen=True)
class ModelSpec:
    hf_name: str
    layers: int
    hidden_size: int
    parameters: str

# ESM-2 family (Lin et al., 2023). All are free to download from Hugging Face.
ESM2_MODELS: Dict[str, ModelSpec] = {
    "8M": ModelSpec("facebook/esm2_t6_8M_UR50D", 6, 320, "8M"),
    "35M": ModelSpec("facebook/esm2_t12_35M_UR50D", 12, 480, "35M"),
    "150M": ModelSpec("facebook/esm2_t30_150M_UR50D", 30, 640, "150M"),
    "650M": ModelSpec("facebook/esm2_t33_650M_UR50D", 33, 1280, "650M"),
    "3B": ModelSpec("facebook/esm2_t36_3B_UR50D", 36, 2560, "3B"),
}

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# ESM-2 positions include the <cls> and <eos> tokens added to every sequence
MAX_SEQUENCE_LENGTH = 1022

def load_model(size: str, dtype: str = "fp32", device: str = "cuda") -> Tuple[torch.nn.Module, object]:
    """Load an ESM-2 model and its tokenizer in inference mode"""
    from transformers import AutoTokenizer, EsmModel

    if size not in ESM2_MODELS:
        raise ValueError(f"Unknown model size '{size}'. Choose from: {', '.join(ESM2_MODELS)}")
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype '{dtype}'. Choose from: {', '.join(DTYPES)}")

    spec = ESM2_MODELS[size]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    # Skip the pooling head: it isn't used for embeddings and is not pretrained
    model = EsmModel.from_pretrained(spec.hf_name, torch_dtype=DTYPES[dtype], add_pooling_layer=False)
    model.to(device).eval()
    return model, tokenizer
