import pytest
import torch
from transformers import BatchEncoding, EsmConfig, EsmModel

class FakeEsmTokenizer:
    """Minimal ESM-style tokenizer: <cls> + residues + <eos>, no download needed"""
    CLS, PAD, EOS = 0, 1, 2

    def __call__(self, sequences, return_tensors="pt", padding=True):
        longest = max(len(s) for s in sequences)
        ids, mask = [], []
        for sequence in sequences:
            tokens = [self.CLS] + [4 + (ord(c) % 20) for c in sequence] + [self.EOS]
            padding_needed = longest - len(sequence)
            ids.append(tokens + [self.PAD] * padding_needed)
            mask.append([1] * len(tokens) + [0] * padding_needed)
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})

@pytest.fixture(scope="session")
def tiny_model():
    """A 2-layer ESM model with random weights that runs quickly on CPU"""
    torch.manual_seed(0)
    config = EsmConfig(
        vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=1026, position_embedding_type="rotary",
        pad_token_id=1, mask_token_id=32, token_dropout=False,
    )
    return EsmModel(config, add_pooling_layer=False).eval()

@pytest.fixture
def tokenizer():
    return FakeEsmTokenizer()
