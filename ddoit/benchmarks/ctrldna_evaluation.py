"""Ctrl-DNA sequence scoring and summary evaluation.

Heavy dependencies are imported lazily so `ddoit` registry and CLI dry-runs
remain dependency-free.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ddoit.benchmarks.ctrldna import (
    default_oracle_checkpoint_path,
    default_starting_sequences_path,
    ensure_legacy_repo_on_path,
    get_fitness_info,
    get_task_cells,
    normalize_score,
    resolve_legacy_repo,
)


@dataclass
class EvaluationArtifacts:
    detail_df: object
    summary: dict[str, float]

    def save(self, output: str | Path, summary_output: str | Path) -> None:
        import pandas as pd

        output = Path(output)
        summary_output = Path(summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        self.detail_df.to_csv(output, index=False)
        pd.DataFrame([self.summary]).to_csv(summary_output, index=False)


def distance(seq_a: str, seq_b: str) -> int:
    return sum(1 for base_a, base_b in zip(seq_a, seq_b) if base_a != base_b)


def median_pairwise_distance(seqs: list[str]) -> float:
    if len(seqs) < 2:
        return 0.0
    import numpy as np

    distances = [distance(s1, s2) for s1, s2 in itertools.combinations(seqs, 2)]
    return float(np.median(distances))


def sequence_shannon_entropy(seqs: list[str]) -> float:
    if not seqs:
        return 0.0
    seq_len = len(seqs[0])
    alphabet = "ACGT"
    total_entropy = 0.0
    for index in range(seq_len):
        counts = Counter(seq[index] for seq in seqs if index < len(seq))
        total = sum(counts.get(base, 0) for base in alphabet)
        if total == 0:
            continue
        position_entropy = 0.0
        for base in alphabet:
            probability = counts.get(base, 0) / total
            if probability > 0:
                import math

                position_entropy -= probability * math.log2(probability)
        total_entropy += position_entropy
    return float(total_entropy / seq_len)


def cell_type_specificity(
    df,
    on_target_col: str,
    off_target_cols: list[str],
    log: bool = False,
) -> list[float]:
    import numpy as np

    off_target = df[off_target_cols].values.T
    on_target = np.tile(df[on_target_col].values, (off_target.shape[0], 1))
    if log:
        values = np.log2(on_target / off_target)
    else:
        values = on_target - off_target
    return values.min(0).tolist()


def novelty_and_diversity_summary(
    round_df,
    starting_sequences,
    top_k: int = 128,
) -> dict[str, float]:
    import numpy as np

    if "true_score" not in round_df.columns:
        raise KeyError("round_df must contain a 'true_score' column")
    if "sequence" not in round_df.columns:
        raise KeyError("round_df must contain a 'sequence' column")
    if "sequence" not in starting_sequences.columns:
        raise KeyError("starting_sequences must contain a 'sequence' column")

    data = round_df.sort_values(by="true_score", ascending=False).iloc[:top_k]
    seqs = data["sequence"].tolist()
    inits = starting_sequences["sequence"].tolist()
    novelty_distances = [min(distance(seq, init_seq) for init_seq in inits) for seq in seqs]
    return {
        "top": float(data.iloc[:16]["true_score"].mean()),
        "fitness": float(data["true_score"].median()),
        "diversity": median_pairwise_distance(seqs),
        "novelty": float(np.median(novelty_distances)) if novelty_distances else 0.0,
    }


class SequenceOracle:
    """Load and run Ctrl-DNA reward models without the RL layer."""

    def __init__(
        self,
        task: str,
        oracle_type: str = "paired",
        device: str | object = "cuda",
        checkpoint_root: str | Path = "/",
        checkpoint_paths: dict[str, str | Path] | None = None,
        legacy_root: str | Path | None = None,
    ) -> None:
        import torch

        ensure_legacy_repo_on_path(legacy_root or resolve_legacy_repo())
        from ctrl_dna.src.reglm.regression import EnformerModel

        self._enformer_model = EnformerModel
        self.task = task
        self.oracle_type = oracle_type
        self.device = torch.device(device)
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_paths = {k: Path(v) for k, v in (checkpoint_paths or {}).items()}
        self.panel_tasks = get_task_cells(task)
        self.targets = {cell: self._load_target_model(cell) for cell in self.panel_tasks}

    def _checkpoint_path(self, cell: str) -> Path:
        if cell in self.checkpoint_paths:
            return self.checkpoint_paths[cell]
        return default_oracle_checkpoint_path(
            task=cell,
            oracle_type=self.oracle_type,
            checkpoint_root=self.checkpoint_root,
        )

    def _load_target_model(self, cell: str):
        map_location = "cpu" if self.device.type == "cpu" else self.device
        model = self._enformer_model.load_from_checkpoint(
            str(self._checkpoint_path(cell)),
            map_location=map_location,
        ).to(self.device)
        model.eval()
        return model

    def normalize_target(self, score: float, cell: str) -> float:
        return normalize_score(score, cell)

    def _predict_cell(
        self,
        cell: str,
        sequences: list[str],
        batch_size: int,
    ):
        import numpy as np
        import torch

        outputs = []
        model = self.targets[cell]
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                preds = model(batch).squeeze(-1).detach().float().cpu().numpy()
                outputs.append(preds)
        raw_scores = np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)
        return np.array([self.normalize_target(score, cell) for score in raw_scores], dtype=np.float32)

    def score_sequences(
        self,
        sequences: Iterable[str],
        batch_size: int = 128,
    ):
        import pandas as pd

        seqs = list(sequences)
        data: dict[str, list[float] | list[str]] = {"sequence": seqs}
        for cell in self.panel_tasks:
            data[f"{cell}_mean"] = self._predict_cell(cell, seqs, batch_size=batch_size).tolist()
        return pd.DataFrame(data)

    def constraint_target(
        self,
        scores,
        constraint: tuple[float, float] = (0.5, 0.5),
    ):
        import numpy as np
        import pandas as pd

        if isinstance(scores, pd.DataFrame):
            target_col = f"{self.task}_mean"
            off_target_cols = [f"{cell}_mean" for cell in self.panel_tasks if cell != self.task]
            return (
                scores[target_col]
                - (scores[off_target_cols[0]] - constraint[0])
                + scores[target_col]
                - (scores[off_target_cols[1]] - constraint[1])
            ).to_numpy(dtype=np.float32)

        score_matrix = np.asarray(scores, dtype=np.float32)
        target_idx = list(self.panel_tasks).index(self.task)
        off_target_indices = [i for i in range(score_matrix.shape[1]) if i != target_idx]
        return (
            score_matrix[:, target_idx]
            - (score_matrix[:, off_target_indices[0]] - constraint[0])
            + score_matrix[:, target_idx]
            - (score_matrix[:, off_target_indices[1]] - constraint[1])
        )


def evaluate_sequence_table(
    sequence_df,
    *,
    task: str,
    sequence_col: str = "sequence",
    oracle_type: str = "paired",
    device: str = "cuda",
    batch_size: int = 128,
    checkpoint_root: str | Path = "/",
    constraint: tuple[float, float] = (0.5, 0.5),
    starting_sequences=None,
    legacy_root: str | Path | None = None,
) -> EvaluationArtifacts:
    import numpy as np
    import pandas as pd

    if sequence_col not in sequence_df.columns:
        raise KeyError(f"Missing sequence column: {sequence_col}")

    oracle = SequenceOracle(
        task=task,
        oracle_type=oracle_type,
        device=device,
        checkpoint_root=checkpoint_root,
        legacy_root=legacy_root,
    )
    scored_df = oracle.score_sequences(
        sequence_df[sequence_col].tolist(),
        batch_size=batch_size,
    )
    scored_df["true_score"] = oracle.constraint_target(scored_df, constraint=constraint)
    score_columns = [column for column in scored_df.columns if column != "sequence"]
    base_df = sequence_df.reset_index(drop=True).drop(
        columns=[column for column in score_columns if column in sequence_df.columns],
        errors="ignore",
    )
    if sequence_col != "sequence":
        base_df = base_df.rename(columns={sequence_col: "sequence"})
    detail_df = pd.concat(
        [base_df, scored_df.drop(columns=["sequence"])],
        axis=1,
    )

    target_col = f"{task}_mean"
    off_target_cols = [f"{cell}_mean" for cell in oracle.panel_tasks if cell != task]
    mean_target = float(detail_df[target_col].mean())
    mean_off_targets = float(detail_df[off_target_cols].mean(axis=1).mean())
    summary: dict[str, float] = {
        "n_sequences": float(len(detail_df)),
        "median_true_score": float(detail_df["true_score"].median()),
        "max_true_score": float(detail_df["true_score"].max()),
        "median_target_score": float(detail_df[target_col].median()),
        "median_specificity": float(
            np.median(
                cell_type_specificity(
                    detail_df,
                    on_target_col=target_col,
                    off_target_cols=off_target_cols,
                )
            )
        ),
        "delta_r": mean_target - mean_off_targets,
        "diversity_entropy": sequence_shannon_entropy(detail_df["sequence"].tolist()),
    }
    for col in off_target_cols:
        summary[f"median_{col}"] = float(detail_df[col].median())

    if starting_sequences is not None:
        summary.update(novelty_and_diversity_summary(detail_df, starting_sequences))

    return EvaluationArtifacts(detail_df=detail_df, summary=summary)


def resolve_starting_sequences(
    task: str,
    *,
    starting_sequences_path: str | Path | None = None,
    level: str = "hard",
    data_root: str | Path,
):
    import pandas as pd

    if starting_sequences_path is not None:
        return pd.read_csv(starting_sequences_path)

    candidate = default_starting_sequences_path(task, level=level, data_root=data_root)
    if candidate.exists():
        return pd.read_csv(candidate)
    return None
