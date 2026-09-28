"""Package-owned Ctrl-DNA guided generation entrypoint.

This module is the native ``ddoit`` runtime surface for Ctrl-DNA generation.
The heavy MDLM model class is still imported from the retained legacy
Ctrl-DNA repository, but reward loading, scoring, and CLI dispatch now live in
the unified code repo.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Sequence

from ddoit.benchmarks.ctrldna import (
    ensure_legacy_repo_on_path,
    normalize_generation_method,
    resolve_legacy_repo,
)
from ddoit.benchmarks.ctrldna_evaluation import evaluate_sequence_table
from ddoit.benchmarks.ctrldna_reward import load_paired_reward


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free reward-guided Ctrl-DNA MDLM generation"
    )
    parser.add_argument("--legacy-ctrldna-root", type=Path, default=resolve_legacy_repo())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--task", required=True, choices=["hepg2", "k562", "sknsh"])
    parser.add_argument(
        "--method",
        default="SMC",
        help="Guidance method or paper-facing alias.",
    )
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--resample-every", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--bok-split-step", type=int, default=None)
    parser.add_argument("--doit-M", type=int, default=8)
    parser.add_argument("--doit-omega", type=float, default=4.0)
    parser.add_argument("--doit-beta", type=float, default=1.0)
    parser.add_argument("--doit-alpha", type=float, default=3.0)
    parser.add_argument("--doit-l-star", type=int, default=128)
    parser.add_argument("--doit-gamma", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reward-mode", choices=["weighted", "composite"], default="weighted")
    parser.add_argument("--reward-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--reference-col", default="seq")
    parser.add_argument("--reference-n", type=int, default=256)
    args = parser.parse_args(argv)
    args.method = normalize_generation_method(args.method)
    if args.num_sequences <= 0:
        parser.error("--num-sequences must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_reference_sequences(args: argparse.Namespace) -> list[str] | None:
    if args.reference_csv is None:
        return None

    import pandas as pd

    print(f"      Loading reference sequences from {args.reference_csv} ...")
    ref_df = pd.read_csv(args.reference_csv)
    n_sample = min(args.reference_n, len(ref_df))
    sequences = ref_df[args.reference_col].sample(
        n_sample,
        random_state=args.seed if args.seed is not None else 42,
    ).tolist()
    print(f"      Sampled {n_sample} reference sequences for motif correlation.")
    return sequences


def _guided_sampling_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_steps": args.num_steps,
        "num_sequences": args.num_sequences,
    }
    if args.method == "SMC":
        kwargs.update(beta=args.beta, resample_every=args.resample_every)
    elif args.method == "CG":
        kwargs.update(guidance_scale=args.guidance_scale)
    elif args.method == "TDS":
        kwargs.update(
            guidance_scale=args.guidance_scale,
            beta=args.beta,
            resample_every=args.resample_every,
        )
    elif args.method == "DOIT_LBOK":
        kwargs.update(
            K=args.K,
            bok_split_step=args.bok_split_step,
            M=args.doit_M,
            omega=args.doit_omega,
            beta=args.doit_beta,
            alpha=args.doit_alpha,
            l_star=args.doit_l_star,
            gamma=args.doit_gamma,
        )
    return kwargs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    legacy_root = ensure_legacy_repo_on_path(args.legacy_ctrldna_root)

    import pandas as pd
    import torch
    from ctrl_dna.mdlm.guided_diffusion import GuidedDiffusion

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _set_seed(args.seed)

    print(f"=== Guided Generation ({args.method}) ===")
    print(f"  Legacy root    : {legacy_root}")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Task           : {args.task}")
    print(f"  Sequences      : {args.num_sequences}")
    print(f"  Seed           : {args.seed}")
    print(f"  Device         : {device}")

    print("\n[1/4] Loading MDLM...")
    model = GuidedDiffusion.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    ).to(device)
    model.eval()
    print(f"      Config sampling steps : {model.config.sampling.steps}")
    print(f"      Sequence length       : {model.config.model.length}")

    if args.method == "unconditional":
        print("\n[2/4] Skipping reward model (unconditional generation).")
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        sequences = model.sample_sequences(
            num_sequences=args.num_sequences,
            num_steps=args.num_steps,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_s = time.perf_counter() - start_time
    else:
        print(f"\n[2/4] Loading reward model (mode={args.reward_mode}, lambda={args.reward_lambda})...")
        reward_fn = load_paired_reward(
            task=args.task,
            checkpoint_root=args.checkpoint_root,
            device=device,
            reward_mode=args.reward_mode,
            lambda_=args.reward_lambda,
            legacy_root=legacy_root,
        )

        print(f"\n[3/4] Sampling {args.num_sequences} sequences with {args.method}...")
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        sequences = model.sample_guided(
            reward_fn,
            method=args.method,
            **_guided_sampling_kwargs(args),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_s = time.perf_counter() - start_time

    print(f"      Generated {len(sequences)} sequences in {inference_time_s:.1f}s.")

    print("\n[4/4] Scoring with oracle...")
    reference_sequences = _load_reference_sequences(args)
    if reference_sequences is not None:
        print("      Reference motif correlation is not yet migrated; scoring proceeds without it.")

    sequence_df = pd.DataFrame({"sequence": sequences})
    artifacts = evaluate_sequence_table(
        sequence_df,
        task=args.task,
        oracle_type="paired",
        device=str(device),
        batch_size=args.batch_size,
        checkpoint_root=args.checkpoint_root,
        legacy_root=legacy_root,
    )
    artifacts.summary["inference_time_s"] = float(inference_time_s)
    artifacts.summary["seconds_per_seq"] = float(inference_time_s / max(len(sequences), 1))
    artifacts.save(args.output, args.summary_output)

    print("\n=== Summary ===")
    for key, value in artifacts.summary.items():
        if isinstance(value, float):
            print(f"  {key:30s} {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
