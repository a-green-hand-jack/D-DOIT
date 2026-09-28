"""Native DRAKES evaluator behind the unified ``ddoit`` entrypoint.

Compares multiple sampling methods (finetuned, pretrained, zero-alpha, CFG,
CG, SMC, TDS) across four metrics:
  1. Log-likelihood under the pretrained model
  2. Predicted expression activity (eval oracle)
  3. ATAC accessibility accuracy
  4. 3-mer Pearson correlation with top-expressing sequences
  5. JASPAR motif Spearman correlation

Results (metrics + plots) are written to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ddoit.benchmarks.drakes_evaluation import (
    compare_kmer,
    count_kmers,
    sample_controlled,
    sample_from_model,
)
from ddoit.benchmarks.drakes_data import batch_dna_detokenize
from ddoit.benchmarks.drakes_models import load_model_modules
from ddoit.benchmarks.drakes_oracle import (
    high_expression_kmers,
    load_reward_model,
    predict_activity,
    predict_atac,
)
from ddoit.benchmarks.drakes_paths import base_path, config_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None):
    base = str(base_path())
    p = argparse.ArgumentParser(
        description="Evaluate DRAKES DNA models across multiple metrics."
    )
    p.add_argument("--ckpt-path", type=str,
                   default=os.path.join(base, "mdlm/reward_bp_results_final/finetuned.ckpt"),
                   help="Path to finetuned checkpoint")
    p.add_argument("--pretrained-path", type=str,
                   default=os.path.join(base, "mdlm/outputs_gosai/pretrained.ckpt"),
                   help="Path to pretrained checkpoint")
    p.add_argument("--zero-alpha-path", type=str,
                   default=os.path.join(base, "mdlm/reward_bp_results_final/zero_alpha.ckpt"),
                   help="Path to zero-alpha checkpoint")
    p.add_argument("--cfg-path", type=str,
                   default=os.path.join(base, "mdlm/outputs_gosai/cfg.ckpt"),
                   help="Path to CFG checkpoint")
    p.add_argument("--num-sample-batches", type=int, default=10)
    p.add_argument("--num-samples-per-batch", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="./outputs/eval_results",
                   help="Directory for plots and metrics JSON")
    p.add_argument("--seed", type=int, default=0)

    # --- Method selection ---
    p.add_argument("--methods", type=str, default="all",
                   help="Comma-separated list of methods to run, e.g. "
                        "'TDS' or 'CG,SMC,DOIT'. "
                        "Choices: pretrained,finetuned,zero_alpha,CFG,CG,SMC,TDS,DOIT,DOIT_LBOK. "
                        "Default: 'all' (run everything).")

    # --- DOIT hyperparameters ---
    p.add_argument("--doit-l-star", type=int, default=64,
                   help="Time threshold: guidance activates for the final l* steps (l ≤ l*)")
    p.add_argument("--doit-l-end", type=int, default=1,
                   help="Early-stop threshold: guidance deactivates below this step (l ≥ l_end). "
                        "Guidance window is l_end ≤ l ≤ l*. Default 1 = guide until the very last step.")
    p.add_argument("--doit-M", type=int, default=32,
                   help="Number of MC candidate next-states per guided step")
    p.add_argument("--doit-omega", type=float, default=1.0,
                   help="Guidance strength (exponent on h-hat in resampling)")
    p.add_argument("--doit-beta", type=float, default=0.5,
                   help="Reward temperature for h-hat = exp(R/beta)")
    p.add_argument("--doit-alpha", type=float, default=1.5,
                   help="Proposal temperature for candidate diversity")
    p.add_argument("--doit-gamma", type=float, default=1.2,
                   help="Completion temperature for x̂_0 proxy (default 1.2)")
    p.add_argument("--doit-verbose", action="store_true", default=False,
                   help="Print per-step diversity/reward/ESS diagnostics for DOIT")
    p.add_argument("--doit-M-schedule", type=str, default="fixed",
                   choices=["fixed", "dynamic", "dynamic_reverse"],
                   help="M schedule: 'fixed', 'dynamic' (ramp up late), "
                        "or 'dynamic_reverse' (ramp down late)")
    p.add_argument("--doit-M-min", type=int, default=4,
                   help="Minimum M when using dynamic schedule (default 4)")
    p.add_argument("--doit-M-thresh-high", type=int, default=80,
                   help="n_masked above this -> M=M_min (default 80)")
    p.add_argument("--doit-M-thresh-low", type=int, default=20,
                   help="n_masked below this -> M=M_max (default 20)")

    # --- DOIT omega schedule ---
    p.add_argument("--doit-omega-schedule", type=str, default="fixed",
                   choices=["fixed", "dynamic", "dynamic_reverse"],
                   help="Omega schedule: 'fixed', 'dynamic' (weak early, strong late), "
                        "or 'dynamic_reverse' (strong early, weak late)")
    p.add_argument("--doit-omega-min", type=float, default=0.5,
                   help="Minimum omega when using dynamic schedule (default 0.5)")
    p.add_argument("--doit-omega-thresh-high", type=int, default=80,
                   help="n_masked above this -> omega=omega_min (default 80)")
    p.add_argument("--doit-omega-thresh-low", type=int, default=20,
                   help="n_masked below this -> omega=omega_max (default 20)")

    # --- DOIT phase strategy ---
    p.add_argument("--doit-phase-strategy", type=str, default="none",
                   choices=["none", "adaptive"],
                   help="Phase strategy: 'none' (disabled) or 'adaptive' "
                        "(skip early, full middle, refine late based on n_masked)")
    p.add_argument("--doit-phase-skip-thresh", type=int, default=100,
                   help="n_masked above this -> skip guidance entirely (default 100)")
    p.add_argument("--doit-phase-active-thresh", type=int, default=20,
                   help="n_masked below this -> refinement phase with smaller M (default 20)")
    p.add_argument("--doit-phase-M-refine", type=int, default=8,
                   help="M to use in refinement phase (default 8)")
    p.add_argument("--doit-phase-omega-refine-scale", type=float, default=1.5,
                   help="Multiply omega by this factor in refinement phase (default 1.5)")
    # --- CG hyperparameters ---
    p.add_argument("--cg-guidance-scale", type=float, default=300000,
                   help="Gradient multiplier for Classifier Guidance (CG)")

    # --- SMC hyperparameters ---
    p.add_argument("--smc-alpha", type=float, default=0.5,
                   help="SMC temperature (controls resampling sharpness)")

    # --- TDS hyperparameters ---
    p.add_argument("--tds-alpha", type=float, default=0.5,
                   help="TDS SMC temperature")
    p.add_argument("--tds-guidance-scale", type=float, default=1000,
                   help="TDS CG gradient multiplier for the proposal")

    # --- Best-of-K selection ---
    p.add_argument("--best-of-k", type=int, default=0,
                   help="If >0, generate K× more sequences and keep top-N by "
                        "reward score (N = num_batches × num_samples_per_batch). "
                        "E.g., --best-of-k 4 with 5×64=320 generates 1280 seqs, "
                        "keeps top 320.")
    p.add_argument("--doit-resample-strategy", type=str, default="per_sequence",
                   choices=["per_sequence", "global"],
                   help="'per_sequence': each seq picks from M candidates (original). "
                        "'global': pool all B×M candidates, resample B (DOIT-resample).")
    p.add_argument("--doit-bok-split-step", type=int, default=110,
                   help="DOIT_LBOK: step at which to split into K copies. "
                        "0=full bok, 110=last 18 steps, 128=no bok.")

    # --- Oracle best-of-K (paper supplement A.1.3) ---
    p.add_argument("--oracle-best-of-k", type=int, default=1,
                   help="If >1, for methods listed in --oracle-bok-methods, "
                        "oversample K× sequences then select top-N by FT-oracle "
                        "(reward) score. Used to compare LBOK-S against TDS/SMC "
                        "under a paired BoK control protocol.")
    p.add_argument("--oracle-bok-methods", type=str, default="TDS,SMC",
                   help="Comma-separated methods to which oracle-BoK is applied.")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    runtime = load_model_modules()
    runtime.set_seed(args.seed, use_cuda=True)
    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["figure.dpi"] = 200

    NB = args.num_sample_batches
    BS = args.num_samples_per_batch
    TOTAL = NB * BS
    # When oracle-best-of-K is active, oversample K× for the targeted methods
    OBOK_K = max(1, args.oracle_best_of_k)
    OBOK_METHODS = {m.strip() for m in args.oracle_bok_methods.split(",") if m.strip()}
    NB_OBOK = NB * OBOK_K  # number of batches to draw for oracle-BoK methods
    metrics = {}  # will be dumped to JSON at the end

    # --- Method selection helper ---
    if args.methods.strip().lower() == "all":
        _run_set = None  # means run everything
    else:
        _run_set = {m.strip() for m in args.methods.split(",")}

    def should_run(name: str) -> bool:
        """Return True if *name* is in the user-requested method set."""
        return _run_set is None or name in _run_set

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(config_dir()),
        job_name="eval",
    )
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = args.ckpt_path

    # Always load the pretrained model — it is used for log-likelihood scoring
    # in section 3 for every evaluated method, not just pretrained/DOIT.
    print("Loading pretrained model …")
    old_model = runtime.diffusion_gosai_update.Diffusion.load_from_checkpoint(
        args.pretrained_path, config=cfg
    )
    old_model.eval()

    model = None
    if should_run("finetuned"):
        print("Loading finetuned model …")
        model = runtime.diffusion_gosai_update.Diffusion(cfg, eval=False).cuda()
        model.load_state_dict(torch.load(cfg.eval.checkpoint_path, weights_only=False))
        model.eval()

    zero_alpha_model = None
    if should_run("zero_alpha"):
        print("Loading zero-alpha model …")
        zero_alpha_model = runtime.diffusion_gosai_update.Diffusion(cfg).cuda()
        zero_alpha_model.load_state_dict(
            torch.load(args.zero_alpha_path, weights_only=False)
        )
        zero_alpha_model.eval()

    cfg_model = None
    if should_run("CFG"):
        print("Loading CFG model …")
        cfg_cfg = compose(config_name="config_gosai.yaml")
        cfg_cfg.model.cls_free_guidance = True
        cfg_cfg.model.cls_free_weight = 10
        cfg_cfg.model.cls_free_prob = 0.1
        cfg_cfg.eval.checkpoint_path = args.cfg_path
        cfg_model = runtime.diffusion_gosai_cfg.Diffusion(cfg_cfg, eval=False).cuda()
        cfg_model.load_state_dict(
            torch.load(cfg_cfg.eval.checkpoint_path, weights_only=False)["state_dict"]
        )
        cfg_model.eval()

    # DOIT: load pretrained weights into the DiffusionDOIT subclass
    doit_model = None
    if should_run("DOIT"):
        print("Loading DOIT model (pretrained weights) …")
        doit_model = runtime.diffusion_doit.DiffusionDOIT.load_from_checkpoint(
            args.pretrained_path, config=cfg
        )
        doit_model.eval()

    # DOIT_LBOK: DOIT with late-stage best-of-K
    doit_lbok_model = None
    if should_run("DOIT_LBOK"):
        print("Loading DOIT_LBOK model (pretrained weights) …")
        doit_lbok_model = runtime.doit_late_bok.DiffusionDOITLateBok.load_from_checkpoint(
            args.pretrained_path, config=cfg
        )
        doit_lbok_model.eval()

    # CG: load pretrained weights into the DiffusionCG subclass
    cg_model = None
    if should_run("CG"):
        print("Loading CG model (pretrained weights) …")
        cg_model = runtime.diffusion_cg.DiffusionCG.load_from_checkpoint(
            args.pretrained_path, config=cfg
        )
        cg_model.eval()

    # SMC: load pretrained weights into the DiffusionSMC subclass
    smc_model = None
    if should_run("SMC"):
        print("Loading SMC model (pretrained weights) …")
        smc_model = runtime.diffusion_smc.DiffusionSMC.load_from_checkpoint(
            args.pretrained_path, config=cfg
        )
        smc_model.eval()

    # TDS: load pretrained weights into the DiffusionTDS subclass
    tds_model = None
    if should_run("TDS"):
        print("Loading TDS model (pretrained weights) …")
        tds_model = runtime.diffusion_tds.DiffusionTDS.load_from_checkpoint(
            args.pretrained_path, config=cfg
        )
        tds_model.eval()

    # ------------------------------------------------------------------
    # 2. Sample from selected methods
    # ------------------------------------------------------------------
    reward_model_bs = None
    if should_run("CG") or should_run("SMC") or should_run("TDS") \
            or should_run("DOIT") or should_run("DOIT_LBOK"):
        reward_model_bs = load_reward_model(mode="train")
        reward_model_bs.eval()
        reward_model_bs.cuda()

    print("\n=== Sampling ===")
    methods = {}  # name -> (detokenized, raw)

    if should_run("pretrained"):
        print("  Pretrained …")
        det_pretrained, raw_pretrained = sample_from_model(
            old_model,
            NB,
            BS,
            detokenize_batch=batch_dna_detokenize,
        )
        methods["pretrained"] = (det_pretrained, raw_pretrained)

    if should_run("zero_alpha"):
        print("  Zero-alpha …")
        det_zero_alpha, raw_zero_alpha = sample_from_model(
            zero_alpha_model,
            NB,
            BS,
            detokenize_batch=batch_dna_detokenize,
        )
        methods["zero_alpha"] = (det_zero_alpha, raw_zero_alpha)

    if should_run("finetuned"):
        print("  Finetuned …")
        det_finetuned, raw_finetuned = sample_from_model(
            model,
            NB,
            BS,
            detokenize_batch=batch_dna_detokenize,
        )
        methods["finetuned"] = (det_finetuned, raw_finetuned)

    if should_run("CFG"):
        print("  CFG …")
        det_cfg, raw_cfg = sample_from_model(
            cfg_model,
            NB,
            BS,
            detokenize_batch=batch_dna_detokenize,
            w=10,
        )
        methods["CFG"] = (det_cfg, raw_cfg)

    inference_times = {}  # method → seconds (pure sampling time)

    def _nb_for(name: str) -> int:
        """Oversample batches for oracle-BoK methods."""
        return NB_OBOK if (name in OBOK_METHODS and OBOK_K > 1) else NB

    if should_run("TDS"):
        nb = _nb_for("TDS")
        if nb > NB:
            print(f"  TDS (oracle-BoK K={OBOK_K}: oversample {nb}×{BS}) …")
        else:
            print("  TDS …")
        det_tds, raw_tds, t_tds = sample_controlled(
            tds_model, "TDS", reward_model_bs, nb, BS,
            detokenize_batch=batch_dna_detokenize,
            alpha=args.tds_alpha,
            guidance_scale=args.tds_guidance_scale,
        )
        methods["TDS"] = (det_tds, raw_tds)
        inference_times["TDS"] = t_tds

    if should_run("CG"):
        nb = _nb_for("CG")
        print(f"  CG ({nb}×{BS}) …")
        det_cg, raw_cg, t_cg = sample_controlled(
            cg_model, "CG", reward_model_bs, nb, BS,
            detokenize_batch=batch_dna_detokenize,
            guidance_scale=args.cg_guidance_scale,
        )
        methods["CG"] = (det_cg, raw_cg)
        inference_times["CG"] = t_cg

    if should_run("SMC"):
        nb = _nb_for("SMC")
        if nb > NB:
            print(f"  SMC (oracle-BoK K={OBOK_K}: oversample {nb}×{BS}) …")
        else:
            print("  SMC …")
        det_smc, raw_smc, t_smc = sample_controlled(
            smc_model, "SMC", reward_model_bs, nb, BS,
            detokenize_batch=batch_dna_detokenize,
            alpha=args.smc_alpha,
        )
        methods["SMC"] = (det_smc, raw_smc)
        inference_times["SMC"] = t_smc

    if should_run("DOIT"):
        bok = max(1, args.best_of_k)
        if bok > 1:
            print(f"  DOIT (best-of-{bok} inline, {NB}×{BS}={NB*BS} final seqs) …")
        else:
            print("  DOIT …")
        det_doit, raw_doit, t_doit = sample_controlled(
            doit_model, "DOIT", reward_model_bs, NB, BS,
            detokenize_batch=batch_dna_detokenize,
            l_star=args.doit_l_star,
            l_end=args.doit_l_end,
            M=args.doit_M,
            omega=args.doit_omega,
            beta=args.doit_beta,
            alpha=args.doit_alpha,
            gamma=args.doit_gamma,
            verbose=args.doit_verbose,
            M_schedule=args.doit_M_schedule,
            M_min=args.doit_M_min,
            M_thresh_high=args.doit_M_thresh_high,
            M_thresh_low=args.doit_M_thresh_low,
            omega_schedule=args.doit_omega_schedule,
            omega_min=args.doit_omega_min,
            omega_thresh_high=args.doit_omega_thresh_high,
            omega_thresh_low=args.doit_omega_thresh_low,
            phase_strategy=args.doit_phase_strategy,
            phase_skip_thresh=args.doit_phase_skip_thresh,
            phase_active_thresh=args.doit_phase_active_thresh,
            phase_M_refine=args.doit_phase_M_refine,
            phase_omega_refine_scale=args.doit_phase_omega_refine_scale,
            best_of_k=bok,
            resample_strategy=args.doit_resample_strategy,
        )
        methods["DOIT"] = (det_doit, raw_doit)
        inference_times["DOIT"] = t_doit

    if should_run("DOIT_LBOK"):
        split = args.doit_bok_split_step
        bok = max(1, args.best_of_k)
        print(f"  DOIT_LBOK (split={split}, K={bok}) …")
        det_lbok, raw_lbok, t_lbok = sample_controlled(
            doit_lbok_model, "DOIT", reward_model_bs, NB, BS,
            detokenize_batch=batch_dna_detokenize,
            l_star=args.doit_l_star,
            l_end=args.doit_l_end,
            M=args.doit_M,
            omega=args.doit_omega,
            beta=args.doit_beta,
            alpha=args.doit_alpha,
            gamma=args.doit_gamma,
            verbose=args.doit_verbose,
            best_of_k=bok,
            bok_split_step=split,
            M_schedule=args.doit_M_schedule,
            M_min=args.doit_M_min,
            M_thresh_high=args.doit_M_thresh_high,
            M_thresh_low=args.doit_M_thresh_low,
            omega_schedule=args.doit_omega_schedule,
            omega_min=args.doit_omega_min,
            omega_thresh_high=args.doit_omega_thresh_high,
            omega_thresh_low=args.doit_omega_thresh_low,
        )
        methods["DOIT_LBOK"] = (det_lbok, raw_lbok)
        inference_times["DOIT_LBOK"] = t_lbok

    # Print inference times
    if inference_times:
        print("\n=== Inference Times ===")
        for m, t in inference_times.items():
            print(f"  {m}: {t:.2f}s ({t/NB:.2f}s/batch)")
        print()

    if not methods:
        print("No methods selected — nothing to do.")
        return

    # ------------------------------------------------------------------
    # 2b. Oracle best-of-K selection (paper supplement A.1.3)
    # ------------------------------------------------------------------
    if OBOK_K > 1 and OBOK_METHODS:
        print(f"\n=== Oracle best-of-{OBOK_K} selection (FT oracle picks top-N) ===")
        for name in list(methods.keys()):
            if name not in OBOK_METHODS:
                continue
            det, raw = methods[name]
            if len(det) <= TOTAL:
                print(f"  [{name}] not oversampled (have {len(det)}, target {TOTAL}); skip selection")
                continue
            ft_scores = predict_activity(det, mode="train")
            target_scores = ft_scores[:, 0] if ft_scores.ndim == 2 else ft_scores
            target_scores = np.asarray(target_scores).reshape(-1)
            top_idx = np.argsort(-target_scores)[:TOTAL]
            det_sel = [det[i] for i in top_idx]
            raw_sel = raw[top_idx]
            methods[name] = (det_sel, raw_sel)
            print(f"  [{name}] selected top-{TOTAL} of {len(det)} by FT-oracle "
                  f"target activity (median pre-sel={float(np.median(target_scores)):.3f}, "
                  f"selected median={float(np.median(target_scores[top_idx])):.3f})")
        metrics["oracle_best_of_k"] = OBOK_K
        metrics["oracle_bok_methods"] = sorted(OBOK_METHODS)

    # ------------------------------------------------------------------
    # 3. Log-likelihood
    # ------------------------------------------------------------------
    if old_model is not None:
        print("\n=== Log-likelihood ===")
        logl = {}
        for name, (_, raw) in methods.items():
            logl[name] = old_model.get_likelihood(raw, num_steps=128, n_samples=1)

        logl_all = np.concatenate(
            [logl[n].detach().cpu().numpy() for n in methods], axis=0
        )
        medians = np.median(logl_all.reshape(len(methods), -1), axis=-1)
        for name, med in zip(methods, medians):
            metrics[f"logl_median/{name}"] = float(med)
            print(f"  {name:12s}: median logl = {med:.4f}")
    else:
        print("\n=== Log-likelihood (skipped — pretrained model not loaded) ===")

    # ------------------------------------------------------------------
    # 4. Pred-Activity (eval oracle)
    # ------------------------------------------------------------------
    print("\n=== Pred-Activity (eval oracle) ===")
    for name, (det, _) in methods.items():
        preds = predict_activity(det, mode="eval")
        med = float(np.median(preds[:, 0]))
        metrics[f"pred_activity_median/{name}"] = med
        print(f"  {name:12s}: median = {med:.4f}")

    # ------------------------------------------------------------------
    # 5. ATAC Accessibility
    # ------------------------------------------------------------------
    print("\n=== ATAC Accessibility ===")
    for name, (det, _) in methods.items():
        preds_atac = predict_atac(det)
        acc = float((preds_atac[:, 1] > 0.5).sum() / len(det))
        metrics[f"atac_acc/{name}"] = acc
        print(f"  {name:12s}: accuracy = {acc:.4f}")

    # ------------------------------------------------------------------
    # 6. 3-mer Pearson Correlation
    # ------------------------------------------------------------------
    print("\n=== 3-mer Pearson Correlation ===")
    (
        highexp_kmers_99, n_highexp_kmers_99,
        highexp_kmers_999, n_highexp_kmers_999,
        highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs_999,
    ) = high_expression_kmers(return_clss=True)

    for name, (det, _) in methods.items():
        gen_kmer = count_kmers(det)
        r = compare_kmer(
            highexp_kmers_999, gen_kmer,
            n_highexp_kmers_999, len(det),
            title=name,
            save_path=os.path.join(args.output_dir, f"kmer_{name}.png"),
        )
        metrics[f"kmer_pearson/{name}"] = float(r)

    # ------------------------------------------------------------------
    # 7. JASPAR Motif Analysis
    # ------------------------------------------------------------------
    print("\n=== JASPAR Motif Analysis ===")
    from grelu.interpret.motifs import scan_sequences

    motif_sums = {}
    for name, (det, _) in methods.items():
        mc = scan_sequences(det, "jaspar")
        motif_sums[name] = mc["motif"].value_counts()

    mc_top = scan_sequences(highexp_seqs_999, "jaspar")
    motif_sums["top_data"] = mc_top["motif"].value_counts()

    motifs_summary = pd.concat(
        [motif_sums["top_data"]] + [motif_sums[n] for n in methods],
        axis=1,
    )
    motifs_summary.columns = ["top_data"] + list(methods.keys())
    spearman = motifs_summary.corr(method="spearman")
    print(spearman.to_string())

    # Store Spearman correlations with top_data
    for name in methods:
        val = float(spearman.loc["top_data", name])
        metrics[f"motif_spearman_vs_top/{name}"] = val

    # ------------------------------------------------------------------
    # 8. Save all metrics (including inference times)
    # ------------------------------------------------------------------
    for m, t in inference_times.items():
        metrics[f"inference_time_s/{m}"] = round(t, 2)
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll metrics saved to {out_path}")


if __name__ == "__main__":
    main()
