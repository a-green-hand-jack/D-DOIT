"""DRAKES DNA tokenization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


DNA_ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_TO_DNA = {value: key for key, value in DNA_ALPHABET.items()}


def dna_detokenize(seq: Iterable[int]) -> str:
    """Convert integer DNA tokens to an A/C/G/T sequence."""

    return "".join(INDEX_TO_DNA[int(token)] for token in seq)


def batch_dna_detokenize(batch_seq: Iterable[Iterable[int]]) -> list[str]:
    """Convert a batch of integer DNA tokens to A/C/G/T sequences."""

    return [dna_detokenize(seq) for seq in batch_seq]


def dna_tokenize(seq: str) -> list[int]:
    """Convert an A/C/G/T sequence to integer DNA tokens."""

    return [DNA_ALPHABET[base] for base in seq]


def batch_dna_tokenize(batch_seq: Sequence[str]) -> list[list[int]]:
    """Convert a batch of A/C/G/T sequences to integer DNA tokens."""

    return [dna_tokenize(seq) for seq in batch_seq]
