"""Protein sequences for benchmarking: synthetic sequences with natural composition, or FASTA files"""
from pathlib import Path
from typing import List
import random

# Amino acid frequencies (%) in UniProtKB/Swiss-Prot, so synthetic sequences
# have the same composition as real proteins
AMINO_ACID_FREQUENCIES = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.37,
    "Q": 3.93, "E": 6.75, "G": 7.07, "H": 2.27, "I": 5.96,
    "L": 9.66, "K": 5.84, "M": 2.42, "F": 3.86, "P": 4.70,
    "S": 6.56, "T": 5.34, "W": 1.08, "Y": 2.92, "V": 6.87,
}

def synthetic_sequences(count: int, length: int, seed: int = 0) -> List[str]:
    """Random protein sequences of a fixed length, reproducible for a given seed"""
    if count < 1 or length < 1:
        raise ValueError("count and length must be at least 1")
    rng = random.Random(seed)
    residues = list(AMINO_ACID_FREQUENCIES)
    weights = list(AMINO_ACID_FREQUENCIES.values())
    return ["".join(rng.choices(residues, weights=weights, k=length)) for _ in range(count)]

def read_fasta(path: str) -> List[str]:
    """Sequences from a FASTA file (headers are skipped, multi-line sequences are joined)"""
    sequences = []
    current = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
            current = []
        elif line:
            current.append(line.upper())
    if current:
        sequences.append("".join(current))
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    return sequences
