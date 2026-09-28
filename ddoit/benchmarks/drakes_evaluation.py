"""DRAKES evaluation helpers shared by package-owned runners."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def sample_from_model(
    model: Any,
    num_batches: int,
    batch_size: int,
    *,
    detokenize_batch: Callable[[Any], list[str]],
    w: float | None = None,
) -> tuple[list[str], Any]:
    """Draw samples from an unconditional model and detokenize them."""

    import torch
    from tqdm import tqdm

    all_detokenized: list[str] = []
    all_raw = []
    for _ in tqdm(range(num_batches)):
        if w is not None:
            samples = model._sample(eval_sp_size=batch_size, w=w)
        else:
            samples = model._sample(eval_sp_size=batch_size)
        all_raw.append(samples)
        detokenized = detokenize_batch(samples.detach().cpu().numpy())
        all_detokenized.extend(detokenized)
    return all_detokenized, torch.concat(all_raw)


def sample_controlled(
    model: Any,
    method: str,
    reward_model: Any,
    num_batches: int,
    batch_size: int,
    *,
    detokenize_batch: Callable[[Any], list[str]],
    **kwargs: Any,
) -> tuple[list[str], Any, float]:
    """Draw controlled samples and return detokenized sequences plus timing."""

    import torch
    from tqdm import tqdm

    all_detokenized: list[str] = []
    all_raw = []
    sample_fn = getattr(model, f"controlled_sample_{method}")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in tqdm(range(num_batches)):
        samples = sample_fn(
            reward_model=reward_model,
            eval_sp_size=batch_size,
            **kwargs,
        )
        all_raw.append(samples)
        detokenized = detokenize_batch(samples.detach().cpu().numpy())
        all_detokenized.extend(detokenized)
    torch.cuda.synchronize()
    return all_detokenized, torch.concat(all_raw), time.perf_counter() - t_start


def count_kmers(seqs: Sequence[str], k: int = 3) -> dict[str, int]:
    """Count k-mers in a sequence collection."""

    counts: dict[str, int] = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            subseq = seq[i : i + k]
            counts[subseq] = counts.get(subseq, 0) + 1
    return counts


def kmer_count_matrix(
    reference_kmers: Mapping[str, int],
    generated_kmers: Mapping[str, int],
    n_reference: int,
    n_generated: int,
) -> list[list[float]]:
    """Build the normalized two-column k-mer count matrix used by DRAKES."""

    kmer_set = sorted(set(reference_kmers.keys()) | set(generated_kmers.keys()))
    counts = [[0.0, 0.0] for _ in kmer_set]
    for i, kmer in enumerate(kmer_set):
        if kmer in reference_kmers:
            counts[i][1] = reference_kmers[kmer] * n_generated / n_reference
        if kmer in generated_kmers:
            counts[i][0] = generated_kmers[kmer]
    return counts


def compare_kmer(
    reference_kmers: Mapping[str, int],
    generated_kmers: Mapping[str, int],
    n_reference: int,
    n_generated: int,
    *,
    title: str,
    save_path: str | Path,
) -> float:
    """Plot and save a k-mer correlation scatter. Returns Pearson r."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    counts = np.asarray(
        kmer_count_matrix(
            reference_kmers,
            generated_kmers,
            n_reference,
            n_generated,
        )
    )
    r, p = pearsonr(counts[:, 0], counts[:, 1])

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.scatter(counts[:, 0], counts[:, 1], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("k-mer train top 0.1% count")
    ax.set_ylabel("k-mer generated count")
    ax.set_ylim((-5, np.max(counts) + 5))
    ax.set_xlim((-5, np.max(counts) + 5))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(0.5, 0.5, f"Pearson Corr: {r:.3f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  {title}: Pearson r = {r:.4f}  (p = {p:.2e})  -> {save_path}")
    return float(r)
