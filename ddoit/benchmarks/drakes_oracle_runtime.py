"""DRAKES oracle runtime implementation under the unified ddoit namespace."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from typing import Any

import grelu
import grelu.data.dataset
import grelu.data.preprocess
import numpy as np
import pandas as pd
from grelu.lightning import LightningModel
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F

from ddoit.benchmarks import drakes_dataloader as dataloader_gosai
from ddoit.benchmarks.drakes_base_config import BASE_PATH


base_path = BASE_PATH

# Resolve device once at import time; fall back to CPU if no GPU is available.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_gosai_oracle(mode: str = "train") -> Any:
    if mode == "train":
        model_load = LightningModel.load_from_checkpoint(
            os.path.join(
                base_path,
                "mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt",
            ),
            map_location=_DEVICE,
        )
    elif mode == "eval":
        model_load = LightningModel.load_from_checkpoint(
            os.path.join(
                base_path,
                "mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt",
            ),
            map_location=_DEVICE,
        )
    else:
        raise ValueError
    model_load.train_params["logger"] = None
    return model_load


def cal_gosai_pred(seqs: Sequence[str], model: Any = None, mode: str = "eval") -> Any:
    """Predict sequence activity with the Lightning dataset API."""

    if model is None:
        model = get_gosai_oracle(mode=mode)
    df_seqs = pd.DataFrame(seqs, columns=["seq"])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    preds = model.predict_on_dataset(pred_dataset, devices=[0])
    return preds.squeeze()


def cal_gosai_pred_new(seqs: Sequence[str], model: Any = None, mode: str = "eval") -> Any:
    """Predict sequence activity with direct tokenization."""

    if model is None:
        model = get_gosai_oracle(mode=mode)
    model.eval()
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens = torch.tensor(tokens).long().to(_DEVICE)
    onehot_tokens = F.one_hot(tokens, num_classes=4).float()
    preds = model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
    return preds.squeeze()


def cal_atac_pred(seqs: Sequence[str], model: Any = None) -> Any:
    """Predict ATAC accessibility with the Lightning dataset API."""

    if model is None:
        model = LightningModel.load_from_checkpoint(
            os.path.join(base_path, "mdlm/gosai_data/binary_atac_cell_lines.ckpt"),
            map_location=_DEVICE,
        )
    df_seqs = pd.DataFrame(seqs, columns=["seq"])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    preds = model.predict_on_dataset(pred_dataset, devices=[0])
    return preds.squeeze()


def cal_atac_pred_new(seqs: Sequence[str], model: Any = None) -> Any:
    """Predict ATAC accessibility with direct tokenization."""

    if model is None:
        model = LightningModel.load_from_checkpoint(
            os.path.join(base_path, "mdlm/gosai_data/binary_atac_cell_lines.ckpt"),
            map_location=_DEVICE,
        )
    model.eval()
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens = torch.tensor(tokens).long().to(_DEVICE)
    onehot_tokens = F.one_hot(tokens, num_classes=4).float()
    preds = model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
    return preds.squeeze()


def count_kmers(seqs: Sequence[str], k: int = 3) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            subseq = seq[i : i + k]
            try:
                counts[subseq] += 1
            except KeyError:
                counts[subseq] = 1
    return counts


def subset_for_eval(n: int = 5000, seed: int = 0) -> Any:
    train_set = dataloader_gosai.get_datasets_gosai()
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set_sp = torch.utils.data.Subset(
        train_set,
        np.random.choice(len(train_set), n, replace=False),
    )
    return train_set_sp


def subset_eval_groundtruth(sets_sp: Any) -> Any:
    train_set_sp = sets_sp
    train_set_sp_clss = train_set_sp.dataset.clss[train_set_sp.indices]
    return train_set_sp_clss


def subset_eval_preds(sets_sp: Any, oracle_model: Any = None) -> Any:
    train_set_sp = sets_sp
    train_preds = cal_gosai_pred(
        dataloader_gosai.batch_dna_detokenize(
            train_set_sp.dataset.seqs[train_set_sp.indices].numpy()
        ),
        oracle_model,
    )
    return train_preds


def subset_eval_kmers(sets_sp: Any, k: int = 3) -> dict[str, int]:
    train_set_sp = sets_sp
    train_seqs = dataloader_gosai.batch_dna_detokenize(
        train_set_sp.dataset.seqs[train_set_sp.indices].numpy()
    )
    train_kmers = count_kmers(train_seqs, k)
    return train_kmers


def subset_eval_embs(sets_sp: Any, oracle_model: Any = None) -> Any:
    train_set_sp = sets_sp
    train_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(
            train_set_sp.dataset.seqs[train_set_sp.indices].numpy()
        ),
        oracle_model,
    )
    return train_sp_emb


def cal_emb_pca(sets_sp: Any, n_components: int = 50, oracle_model: Any = None) -> Any:
    train_set_sp = sets_sp
    train_sp_emb = cal_gosai_emb(
        dataloader_gosai.batch_dna_detokenize(
            train_set_sp.dataset.seqs[train_set_sp.indices].numpy()
        ),
        oracle_model,
    )
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    pca.fit(train_sp_emb.reshape(train_sp_emb.shape[0], -1))
    return pca


def subset_eval_embs_pca(sets_sp: Any, pca: Any, oracle_model: Any = None) -> Any:
    train_sp_emb = subset_eval_embs(sets_sp, oracle_model)
    train_sp_emb_pca = pca.transform(train_sp_emb.reshape(train_sp_emb.shape[0], -1))
    return train_sp_emb_pca


# https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py
def get_wasserstein_dist(embeds1: Any, embeds2: Any) -> float:
    if (
        np.isnan(embeds2).any()
        or np.isnan(embeds1).any()
        or len(embeds1) == 0
        or len(embeds2) == 0
    ):
        return float("nan")
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist


def embed_on_dataset(
    model: Any,
    dataset: Callable[..., Any],
    devices: str | int | list[int] = "cpu",
    num_workers: int = 1,
    batch_size: int = 256,
) -> Any:
    """Return embeddings for a dataset of sequences."""

    torch.set_float32_matmul_precision("medium")

    dataloader = model.make_predict_loader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    orig_device = model.device
    device = model.parse_devices(devices)[1]
    if isinstance(device, list):
        device = device[0]
    model.to(device)

    preds = []
    model.model = model.model.eval()
    for batch in iter(dataloader):
        batch = batch.to(device)
        preds.append(model.model.embedding(batch).detach().cpu())

    model.to(orig_device)
    return torch.vstack(preds).numpy()


def cal_gosai_emb(seqs: Sequence[str], model: Any = None) -> Any:
    """Return GOSAI oracle embeddings for sequences."""

    if model is None:
        model = get_gosai_oracle()
    df_seqs = pd.DataFrame(seqs, columns=["seq"])
    pred_dataset = grelu.data.dataset.DFSeqDataset(df_seqs)
    embs = embed_on_dataset(model, pred_dataset, devices=[0])
    return embs


def cal_highexp_kmers(k: int = 3, return_clss: bool = False) -> Any:
    train_set = dataloader_gosai.get_datasets_gosai()
    exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.99)
    highexp_indices = [
        i for i, data in enumerate(train_set) if data["clss"][0] > exp_threshold
    ]
    highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(
        highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy()
    )
    highexp_kmers_99 = count_kmers(highexp_seqs, k=k)
    n_highexp_kmers_99 = len(highexp_indices)

    exp_threshold = np.quantile(train_set.clss[:, 0].numpy(), 0.999)
    highexp_indices = [
        i for i, data in enumerate(train_set) if data["clss"][0] > exp_threshold
    ]
    highexp_set_sp = torch.utils.data.Subset(train_set, highexp_indices)
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(
        highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy()
    )
    highexp_kmers_999 = count_kmers(highexp_seqs, k=k)
    n_highexp_kmers_999 = len(highexp_indices)

    if return_clss:
        highexp_set_sp_clss_999 = highexp_set_sp.dataset.clss[highexp_set_sp.indices]
        highexp_preds_999 = cal_gosai_pred_new(
            dataloader_gosai.batch_dna_detokenize(
                highexp_set_sp.dataset.seqs[highexp_set_sp.indices].numpy()
            )
        )
        return (
            highexp_kmers_99,
            n_highexp_kmers_99,
            highexp_kmers_999,
            n_highexp_kmers_999,
            highexp_set_sp_clss_999,
            highexp_preds_999,
            highexp_seqs,
        )

    return highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999


def cal_kmer_corr(model: Any, highexp_kmers: dict[str, int], n_highexp_kmers: int, n_sample: int = 128) -> float:
    model.eval()
    all_detokenized_samples = []
    for _ in range(10):
        samples = model._sample(eval_sp_size=n_sample).detach().cpu().numpy()
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples)
        all_detokenized_samples.extend(detokenized_samples)
    generated_kmer = count_kmers(all_detokenized_samples)

    kmer_set = set(highexp_kmers.keys()) | set(generated_kmer.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in highexp_kmers:
            counts[i][1] = highexp_kmers[kmer] * len(generated_kmer) / n_highexp_kmers
        if kmer in generated_kmer:
            counts[i][0] = generated_kmer[kmer]

    corr = pearsonr(counts[:, 0], counts[:, 1])[0]
    return corr


def cal_avg_likelihood(model: Any, old_model: Any, n_sample: int = 128) -> float:
    model.eval()
    old_model.eval()
    all_raw_samples = []
    for _ in range(10):
        samples = model._sample(eval_sp_size=n_sample)
        all_raw_samples.append(samples)
    all_raw_samples = torch.concat(all_raw_samples)
    avg_likelihood = old_model._forward_pass_diffusion(all_raw_samples).sum(-1).mean().item()
    return avg_likelihood
