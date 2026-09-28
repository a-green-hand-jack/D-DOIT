"""
DiffusionDOIT – Discrete DOIT for MDLM, two-stage proposal-resample.

Implements Algorithm 2 (Discrete DOIT) with the following structure:

  Stage 1 – Proposal & h-function estimation
    1a.  Sample M candidate next-states y^(m) from a tempered proposal
         distribution (temperature α) at positions to be unmasked.
    1b.  For each candidate, run a second forward pass to predict x_0
         for still-masked positions, evaluate reward  →  ĥ = exp(R/β).

  Stage 2 – Tilted resampling
    Compute weights  w_m ∝ ĥ_m^ω, softmax-normalise, and categorically
    resample one candidate  →  x_{l-1}.

Hyperparameters (see Table in hyper_discrete_doit.tex):
  L       Total diffusion steps             (from config)
  M       MC candidate samples per step     (default 32)
  ω       Guidance strength (resampling)    (default 1.0)
  β       Reward temperature  ĥ=exp(R/β)   (default 0.5)
  α       Proposal temperature              (default 1.5)
  γ       Completion temperature  x̂_0      (default 1.2)
  l*      Start threshold: guidance begins  (default 64)
  l_end   End threshold: guidance stops at  (default 1)
          Guidance is active for l_end ≤ l ≤ l*.

References
──────────
  Algorithm 2 in the DOIT paper (Discrete DOIT).
  diffusion_gosai_update.py  (base Diffusion class).
"""

from collections import Counter

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ddoit.benchmarks.drakes_diffusion_gosai_update import (
    Diffusion,
    _sample_categorical,
)


class DiffusionDOIT(Diffusion):
    """Diffusion model with Discrete DOIT guidance (two-stage proposal-resample).

    Inherits the full MDLM diffusion model and overrides only the controlled-
    sampling methods.  Call via ``controlled_sample_DOIT(...)``
    """

    # ------------------------------------------------------------------
    # Stage 1 + Stage 2: Proposal, h-estimation, and tilted resampling
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dynamic M schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dynamic_M(
        n_masked: int,
        M_max: int,
        M_min: int = 4,
        thresh_high: int = 80,
        thresh_low: int = 20,
        reverse: bool = False,
    ) -> int:
        """Piecewise-linear M schedule based on n_masked.

        Default (reverse=False):
            n_masked > thresh_high  →  M_min     (early: candidates redundant)
            n_masked < thresh_low   →  M_max     (late:  candidates diverse)

        Reverse (reverse=True):
            n_masked > thresh_high  →  M_max     (early: keep full exploration)
            n_masked < thresh_low   →  M_min     (late:  sequence mostly decided)

        Returns:
            M_eff: int — effective M for this step.
        """
        if reverse:
            # Flip: high n_masked → M_max, low n_masked → M_min
            if n_masked >= thresh_high:
                return M_max
            if n_masked <= thresh_low:
                return M_min
            frac = (n_masked - thresh_low) / (thresh_high - thresh_low)
            M_raw = M_min + (M_max - M_min) * frac
        else:
            if n_masked >= thresh_high:
                return M_min
            if n_masked <= thresh_low:
                return M_max
            frac = (thresh_high - n_masked) / (thresh_high - thresh_low)
            M_raw = M_min + (M_max - M_min) * frac
        # Round to nearest even for cleaner batch computation
        M_eff = max(M_min, int(round(M_raw / 2) * 2))
        return min(M_eff, M_max)

    # ------------------------------------------------------------------
    # Adaptive phase strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phase_config(
        n_masked: int,
        M_max: int,
        omega_max: float,
        skip_thresh: int = 100,
        active_thresh: int = 20,
        M_refine: int = 8,
        omega_refine_scale: float = 1.5,
    ) -> tuple:
        """Three-phase adaptive strategy based on n_masked.

        Phase 1 — Skip    (n_masked > skip_thresh):
            Candidates are nearly identical due to copy_flag dominance.
            Skip guidance entirely, saving the h-estimation forward pass.

        Phase 2 — Guide   (active_thresh < n_masked ≤ skip_thresh):
            Meaningful candidate diversity. Full M and omega.

        Phase 3 — Refine  (n_masked ≤ active_thresh):
            Most positions decided. Use fewer candidates with stronger
            guidance for the remaining critical positions.

        Returns:
            (M_eff, omega_eff, skip_guidance: bool)
        """
        if n_masked > skip_thresh:
            return M_max, omega_max, True
        elif n_masked > active_thresh:
            return M_max, omega_max, False
        else:
            return M_refine, omega_max * omega_refine_scale, False

    # ------------------------------------------------------------------
    # Dynamic omega schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dynamic_omega(
        n_masked: int,
        omega_max: float,
        omega_min: float = 0.5,
        thresh_high: int = 80,
        thresh_low: int = 20,
        reverse: bool = False,
    ) -> float:
        """Piecewise-linear omega schedule based on n_masked.

        Default (reverse=False):
            n_masked > thresh_high  →  omega_min   (early: weak guidance)
            n_masked < thresh_low   →  omega_max   (late:  strong guidance)

        Reverse (reverse=True):
            n_masked > thresh_high  →  omega_max   (early: strong guidance)
            n_masked < thresh_low   →  omega_min   (late:  weak guidance)

        Returns:
            omega_eff: float — effective omega for this step.
        """
        if reverse:
            if n_masked >= thresh_high:
                return omega_max
            if n_masked <= thresh_low:
                return omega_min
            frac = (n_masked - thresh_low) / (thresh_high - thresh_low)
            return omega_min + (omega_max - omega_min) * frac
        else:
            if n_masked >= thresh_high:
                return omega_min
            if n_masked <= thresh_low:
                return omega_max
            frac = (thresh_high - n_masked) / (thresh_high - thresh_low)
            return omega_min + (omega_max - omega_min) * frac

    @torch.no_grad()
    def _doit_proposal_resample(
        self,
        x: torch.Tensor,
        log_p_x0: torch.Tensor,
        sigma_s: torch.Tensor,
        move_chance_t: torch.Tensor,
        move_chance_s: torch.Tensor,
        reward_model,
        M: int,
        omega: float,
        beta: float,
        alpha: float,
        gamma: float,
        verbose: bool = False,
        step_idx: int = 0,
        diagnostics_sink: list = None,
        candidate_sink: list = None,
        l_star: int = None,
        l_end: int = None,
        completion_noise_K: int = 0,
        completion_noise_every: int = 5,
        resample_strategy: str = "per_sequence",
    ) -> torch.Tensor:
        """Two-stage proposal-resample for one denoising step (guided).

        Stage 1a – Construct M candidate next-states y^(m):
          Apply proposal temperature α to p_θ(x_0|x_l), convert to MDLM
          transition distribution q(x_{l-1}|x_l), and sample M candidates.

        Stage 1b – Estimate h-function for each candidate:
          Forward pass on each candidate at σ_s, fill remaining masks by
          sampling from γ-tempered p_θ(x_0|y^(m)), evaluate reward → ĥ = exp(R/β).

        Stage 2 – Tilted resampling:
          Weights w_m ∝ ĥ_m^ω, softmax, categorically select one candidate.

        Args:
            x:              LongTensor [B, L]       current noisy sequence x_l.
            log_p_x0:       Tensor [B, L, V]        log p_θ(x_0 | x_l) from
                                                    the already-computed forward pass.
            sigma_s:        Tensor [B]              noise level for step l-1.
            move_chance_t:  Tensor [B, 1, 1]        1 - exp(-σ_t).
            move_chance_s:  Tensor [B, 1, 1]        1 - exp(-σ_s).
            reward_model:   callable                reward oracle.
            M:              int                     number of MC candidates.
            omega:          float                   guidance strength (resampling exponent).
            beta:           float                   reward temperature for ĥ=exp(R/β).
            alpha:          float                   proposal temperature.
            gamma:          float                   completion temperature for x̂_0 proxy.
            verbose:        bool                    if True, print per-step diagnostics.
            step_idx:       int                     current step index (for verbose label).
            diagnostics_sink: list | None            if not None, append a diagnostics dict
                                                    per guided step (for offline analysis).
            l_star:         int | None               passed through to the sink for labelling.
            l_end:          int | None               passed through to the sink for labelling.

        Returns:
            x_next: LongTensor [B, L]   selected next state x_{l-1}.
        """
        B, L, V = log_p_x0.shape
        acgt_size = self.mask_index  # = 4 (A/C/G/T); mask token is index 4

        # ── Stage 1a: Proposal sampling ─────────────────────────────────────
        # Tempered token probabilities over ACGT tokens only
        p_x0 = log_p_x0[..., :acgt_size].exp()                     # [B, L, 4]
        if alpha != 1.0:
            p_x0_tempered = p_x0.pow(1.0 / alpha)                  # [B, L, 4]
            p_x0_tempered = p_x0_tempered / (
                p_x0_tempered.sum(dim=-1, keepdim=True) + 1e-8
            )
        else:
            p_x0_tempered = p_x0 / (p_x0.sum(dim=-1, keepdim=True) + 1e-8)

        # Build tempered transition distribution q_xs for MDLM
        # q_xs[v] = p_tempered(v) * (move_chance_t - move_chance_s)  for v in ACGT
        # q_xs[mask] = move_chance_s
        q_xs_tempered = torch.zeros(B, L, V, device=x.device)
        q_xs_tempered[..., :acgt_size] = p_x0_tempered * (
            move_chance_t - move_chance_s
        )
        q_xs_tempered[..., self.mask_index] = move_chance_s[..., 0]

        # Sample M candidate next-states in parallel: [B, M, L]
        # Expand q_xs to [B, M, L, V] and sample
        
        q_xs_exp = q_xs_tempered.unsqueeze(1).expand(B, M, L, V)   # [B, M, L, V]
        flat_q = q_xs_exp.reshape(B * M, L, V)
        candidates_flat = _sample_categorical(flat_q)               # [B*M, L]

        # Copy flag: already-unmasked positions must keep their current token
        copy_flag = (x != self.mask_index).to(x.dtype)              # [B, L]
        copy_flag_exp = copy_flag.unsqueeze(1).expand(B, M, L)     # [B, M, L]
        x_exp = x.unsqueeze(1).expand(B, M, L)                     # [B, M, L]
        candidates = candidates_flat.view(B, M, L)                 # [B, M, L]
        candidates = (copy_flag_exp * x_exp
                       + (1 - copy_flag_exp) * candidates).long()

        # ── Candidate sink: record all M candidates for offline visualisation ─
        if candidate_sink is not None:
            n_masked_now = int((x[0] == self.mask_index).sum().item())
            # p_x0_max: max token probability per position from p_θ(x0|x_t)
            # shape [L] — shared across all M candidates (same x_t input)
            # Must normalise over ACGT (log_p_x0 logits are not pre-normalised)
            _p_acgt = log_p_x0[0, :, :self.mask_index].exp()           # [L, 4]
            _p_norm = _p_acgt / (_p_acgt.sum(dim=-1, keepdim=True) + 1e-8)
            p_x0_max  = _p_norm.max(dim=-1).values.cpu().numpy().copy() # [L]
            p_x0_full = _p_norm.cpu().numpy().copy()                    # [L, 4]
            candidate_sink.append({
                'step_idx':   step_idx,
                'n_masked':   n_masked_now,
                'x':          x[0].cpu().numpy().copy(),            # [L] state before step
                'candidates': candidates[0].cpu().numpy().copy(),   # [M, L] all M candidates
                'p_x0_max':   p_x0_max,                            # [L] max prob at each pos
                'p_x0_full':  p_x0_full,                           # [L, 4] full distribution
            })

        # ── Pre-forward diversity diagnostics ────────────────────────────────
        # Compute BEFORE the expensive forward pass so we know the potential
        # saving from deduplication.
        _pre_fwd_unique = None
        _pre_fwd_dedup_saving = None
        _masked_entropy_mean = None
        _n_masked = 0
        _hamming_mean = None
        _hamming_norm = None
        _pos_agreement_mean = None
        _largest_cluster_frac = None
        _cluster_sizes_str = None
        # New token-coverage metrics
        _n_newly_unmasked = None
        _effective_unmask_rate = None
        _token_entropy_at_new_pos = None
        _token_coverage_at_new_pos = None
        _min_token_group_size = None
        _majority_unmasked = None   # saved for per-token reward analysis below
        if verbose or diagnostics_sink is not None:
            mask_flag = (x[0] == self.mask_index)                     # [L]
            n_masked = mask_flag.sum().item()
            _n_masked = n_masked
            _effective_unmask_rate = round(
                (move_chance_t - move_chance_s)[0, 0, 0].item(), 6)
            if n_masked > 0:
                # unique candidates on masked positions only (batch elem 0)
                masked_tokens = candidates[0, :, mask_flag]           # [M, n_masked]
                cand_tuples = [tuple(r.tolist()) for r in masked_tokens]
                cluster_counter = Counter(cand_tuples)
                _pre_fwd_unique = len(cluster_counter)
                _pre_fwd_dedup_saving = round(1.0 - _pre_fwd_unique / M, 6)
                cluster_sizes_sorted = sorted(cluster_counter.values(), reverse=True)
                _largest_cluster_frac = round(cluster_sizes_sorted[0] / M, 4)
                _cluster_sizes_str = ",".join(str(s) for s in cluster_sizes_sorted)

                # entropy of p_θ(x_0|x_l) at masked positions (batch elem 0)
                p0_masked = p_x0[0, mask_flag]                        # [n_masked, 4]
                ent = -(p0_masked * p0_masked.clamp(min=1e-8).log()).sum(-1)
                _masked_entropy_mean = round(ent.mean().item(), 6)

                # ── D3: Pairwise Hamming distance on masked positions ────
                diffs = (masked_tokens.unsqueeze(0)
                         != masked_tokens.unsqueeze(1))               # [M, M, n_masked]
                hamming_all = diffs.float().sum(-1)                   # [M, M]
                _hamming_mean = round(
                    (hamming_all.sum() / (M * (M - 1))).item(), 4)    # exclude diag
                _hamming_norm = round(_hamming_mean / n_masked, 6)    # normalized

                # ── D4: Position-wise token agreement ────────────────────
                # For each masked position, max fraction of candidates on same token
                agree_sum = 0.0
                for pi in range(n_masked):
                    counts = torch.bincount(masked_tokens[:, pi], minlength=4)
                    agree_sum += counts.max().item() / M
                _pos_agreement_mean = round(agree_sum / n_masked, 6)

                # ── NEW D5: Token coverage at newly-unmasked positions ───
                # For each candidate, which masked positions became non-MASK?
                newly_unmasked_per_cand = (
                    (candidates[0] != self.mask_index) & mask_flag)   # [M, L]
                _n_newly_unmasked = round(
                    newly_unmasked_per_cand.float().sum(dim=1).mean().item(), 4)
                # "Shared-unmasked": positions that ≥ 2 candidates fill
                # (M/2 was too strict: with ~1 unmask/step, most positions are
                # only unmasked by 1-3 candidates, never by M/2=16)
                majority_unmasked = (
                    newly_unmasked_per_cand.sum(dim=0) >= 2)          # [L]
                _majority_unmasked = majority_unmasked
                n_new_majority = majority_unmasked.sum().item()
                if n_new_majority > 0:
                    tokens_at_new = candidates[0][:, majority_unmasked]  # [M, n_new]
                    tok_ent_list, tok_cov_list, min_grp_list = [], [], []
                    for pi in range(tokens_at_new.shape[1]):
                        col = tokens_at_new[:, pi]
                        col_acgt = col[col != self.mask_index]
                        if col_acgt.numel() == 0:
                            continue
                        cnts = torch.bincount(col_acgt, minlength=4)[:4].float()
                        probs_nz = (cnts / cnts.sum())
                        probs_nz = probs_nz[probs_nz > 0]
                        tok_ent_list.append(
                            -(probs_nz * probs_nz.log2()).sum().item())
                        tok_cov_list.append((cnts > 0).sum().item() / 4.0)
                        min_grp_list.append(int(cnts[cnts > 0].min().item()))
                    if tok_ent_list:
                        _token_entropy_at_new_pos = round(
                            sum(tok_ent_list) / len(tok_ent_list), 6)
                        _token_coverage_at_new_pos = round(
                            sum(tok_cov_list) / len(tok_cov_list), 6)
                        _min_token_group_size = min(min_grp_list)
            else:
                _pre_fwd_unique = M
                _pre_fwd_dedup_saving = 0.0
                _masked_entropy_mean = 0.0
                _hamming_mean = 0.0
                _hamming_norm = 0.0
                _pos_agreement_mean = 1.0
                _n_newly_unmasked = 0.0
                _effective_unmask_rate = 0.0

        # ── Stage 1b: h-function estimation ─────────────────────────────────
        # Deduplicated forward pass: early in diffusion most candidates are
        # identical (copy_flag dominates), so we forward-pass only unique
        # sequences and scatter back.  Saves up to (M-1)/M of compute.
        cand_flat = candidates.reshape(B * M, L)                    # [B*M, L]
        unique_cand, inv_idx = torch.unique(
            cand_flat, dim=0, return_inverse=True)                  # [U, L], [B*M]
        sigma_s_uniq = sigma_s[0].expand(unique_cand.shape[0])     # [U]
        uniq_log_p_x0 = self.forward(unique_cand, sigma_s_uniq)    # [U, L, V]
        cand_log_p_x0 = uniq_log_p_x0[inv_idx]                    # [B*M, L, V]

        # Build proxy x̂_0: keep already-decoded positions, sample rest from
        # γ-tempered p_θ(x_0 | y^(m))  (Alg 2, Line 18: completion temp γ)
        cand_p_x0 = cand_log_p_x0[..., :acgt_size].exp()           # [B*M, L, 4]
        if gamma != 1.0:
            cand_p_x0 = cand_p_x0.pow(1.0 / gamma)                 # [B*M, L, 4]
        cand_p_x0 = cand_p_x0 / (cand_p_x0.sum(dim=-1, keepdim=True) + 1e-8)
        sampled_x0 = Categorical(probs=cand_p_x0).sample()         # [B*M, L]

        # Positions that are still masked in the candidate get filled
        is_mask_cand = (cand_flat == self.mask_index)                # [B*M, L]
        x_hat = torch.where(is_mask_cand, sampled_x0, cand_flat)   # [B*M, L]

        # Evaluate reward oracle: expects one-hot [B*M, 4, L]
        one_hot_x_hat = F.one_hot(x_hat, num_classes=acgt_size).float()
        scores = reward_model(one_hot_x_hat.transpose(1, 2))       # [B*M, T]
        if scores.ndim > 1:
            scores = scores[:, 0]                                   # [B*M]
        rewards = scores.view(B, M)                                 # [B, M]

        # ĥ = exp(R / β)  →  log_h = R / β
        log_h = rewards / beta                                      # [B, M]

        # ── NEW: Per-token reward analysis & Hamming-reward correlation ──────
        _per_token_reward_gap = None
        _per_token_reward_cv = None
        _between_within_snr = None
        _hamming_reward_corr = None
        if verbose or diagnostics_sink is not None:
            r0_diag = rewards[0].float()                              # [M]
            # Per-token analysis: use shared-unmasked positions (≥2 candidates)
            if _majority_unmasked is not None and _majority_unmasked.sum() > 0:
                new_pos_indices = _majority_unmasked.nonzero(as_tuple=True)[0]
                first_new_pos = new_pos_indices[0]
                tok_at_first = candidates[0, :, first_new_pos]       # [M]
                group_means, group_cvs, group_vars = [], [], []
                for tok in range(4):
                    tok_mask = (tok_at_first == tok)
                    if tok_mask.sum() > 0:
                        gr = r0_diag[tok_mask]
                        group_means.append(gr.mean().item())
                        if gr.numel() > 1:
                            group_cvs.append(
                                gr.std().item() / (abs(gr.mean().item()) + 1e-8))
                            group_vars.append(gr.var().item())
                if len(group_means) > 1:
                    _per_token_reward_gap = round(
                        max(group_means) - min(group_means), 6)
                    if group_cvs:
                        _per_token_reward_cv = round(
                            sum(group_cvs) / len(group_cvs), 6)
                    if group_vars:
                        within_var = sum(group_vars) / len(group_vars)
                        between_var = (torch.tensor(group_means).var().item()
                                       if len(group_means) > 1 else 0.0)
                        _between_within_snr = round(
                            between_var / (within_var + 1e-8), 6)
            # Spearman correlation: pairwise Hamming vs |reward diff|
            # Computed independently of majority_unmasked (always when masked)
            if _n_masked and _n_masked > 0 and M > 3:
                mask_flag_diag = (x[0] == self.mask_index)
                masked_cands_diag = candidates[0][:, mask_flag_diag]  # [M, n_mask]
                pairs_ham, pairs_rdiff = [], []
                for ci in range(M):
                    for cj in range(ci + 1, M):
                        ham = ((masked_cands_diag[ci] != masked_cands_diag[cj])
                               .float().mean().item())
                        pairs_rdiff.append(
                            abs(r0_diag[ci].item() - r0_diag[cj].item()))
                        pairs_ham.append(ham)
                if len(pairs_ham) > 2:
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(pairs_ham, pairs_rdiff)
                    _hamming_reward_corr = round(float(corr), 6)

        # ── Stage 2: Tilted resampling ──────────────────────────────────────
        # Weights w_m ∝ ĥ_m^ω  →  log_w = ω · log_h
        log_w = omega * log_h                                       # [B, M]

        if resample_strategy == "global":
            # Global resampling: pool all B×M candidates, sample B
            log_w_flat = log_w.reshape(-1)                          # [B*M]
            log_w_flat = log_w_flat - log_w_flat.max()
            weights_flat = F.softmax(log_w_flat, dim=0)             # [B*M]
            indices = Categorical(probs=weights_flat).sample(
                [B]).to(x.device)                                   # [B]
            cand_flat = candidates.reshape(B * M, L)                # [B*M, L]
            x_next = cand_flat[indices]                             # [B, L]
            # For diagnostics compatibility
            m_star = indices % M                                    # [B]
            weights = weights_flat.reshape(B, M)
        else:
            # Per-sequence resampling (original)
            log_w = log_w - log_w.max(dim=1, keepdim=True).values
            weights = F.softmax(log_w, dim=1)                       # [B, M]
            m_star = Categorical(probs=weights).sample()             # [B]
            x_next = candidates[torch.arange(B, device=x.device), m_star]

        # ── NEW: Selection quality metrics ────────────────────────────────
        _regret = None
        _rank_of_selected = None
        if verbose or diagnostics_sink is not None:
            r0_sel = rewards[0].float()                               # [M]
            _regret = round((r0_sel.max() - r0_sel[m_star[0]]).item(), 6)
            # Rank: how many rewards are strictly better than selected?
            sel_reward = r0_sel[m_star[0]].item()
            _rank_of_selected = int(
                (r0_sel > sel_reward).sum().item() + 1)

        # ── NEW: Token-coverage selection quality ─────────────────────────
        # Tests whether M matters because it covers rare high-reward tokens.
        # Key fix: analyse positions that the SELECTED candidate newly unmasked
        # (not majority_unmasked positions), so these metrics are always
        # defined when x_next unmasked ≥1 position.
        #
        #   sel_token_rank_in_model : avg rank of selected token in p_θ (1=argmax)
        #   sel_token_count_in_M    : avg # of M candidates placing same token
        #   n_unique_tokens_at_new_pos : avg distinct ACGT tokens across M at those pos
        _sel_token_rank_in_model = None
        _sel_token_count_in_M = None
        _n_unique_tokens_at_new_pos = None
        if verbose or diagnostics_sink is not None:
            mask_flag_sel = (x[0] == self.mask_index)                 # [L]
            sel_newly = (x_next[0] != self.mask_index) & mask_flag_sel  # [L]
            if sel_newly.sum() > 0:
                sel_new_positions = sel_newly.nonzero(as_tuple=True)[0]
                rank_list, cnt_list, utok_list = [], [], []
                for pos in sel_new_positions:
                    sel_tok_p = x_next[0, pos].item()
                    tok_at_p = candidates[0, :, pos]                  # [M]
                    # ACGT tokens placed by M candidates at this position
                    valid_toks_p = tok_at_p[tok_at_p != self.mask_index]
                    if valid_toks_p.numel() > 0:
                        utok_list.append(int(valid_toks_p.unique().numel()))
                    # How many of M placed the same token as selected?
                    cnt_list.append(int((tok_at_p == sel_tok_p).sum().item()))
                    # Rank of selected token in model p_θ(x₀|xₗ) (1=argmax)
                    if sel_tok_p < self.mask_index:
                        model_probs_p = p_x0[0, pos]                  # [4]
                        rank_p = int(
                            (model_probs_p > model_probs_p[sel_tok_p])
                            .sum().item() + 1)
                        rank_list.append(rank_p)
                if rank_list:
                    _sel_token_rank_in_model = round(
                        sum(rank_list) / len(rank_list), 3)
                if cnt_list:
                    _sel_token_count_in_M = round(
                        sum(cnt_list) / len(cnt_list), 2)
                if utok_list:
                    _n_unique_tokens_at_new_pos = round(
                        sum(utok_list) / len(utok_list), 3)

        # ── D7: Top-1 reward dominance (batch elem 0) ────────────────────
        _r_2nd = None
        _r_gap = None
        _w_max = None
        _weighted_hamming = None
        if verbose or diagnostics_sink is not None:
            r_sorted = rewards[0].sort(descending=True).values
            _r_2nd = round(r_sorted[1].item(), 6)
            _r_gap = round((r_sorted[0] - r_sorted[1]).item(), 6)
            _w_max = round(weights[0].max().item(), 6)
            # Weighted Hamming: content diversity after resampling
            if _n_masked > 0:
                mask_flag_w = (x[0] == self.mask_index)
                masked_cands_w = candidates[0, :, mask_flag_w]
                diffs_w = (masked_cands_w.unsqueeze(0)
                           != masked_cands_w.unsqueeze(1)).float()
                hamming_mat = diffs_w.sum(-1) / _n_masked
                w0 = weights[0]
                w_outer = w0.unsqueeze(1) * w0.unsqueeze(0)
                _weighted_hamming = round((w_outer * hamming_mat).sum().item(), 6)

        # ── D8: Completion noise probe (optional) ────────────────────────
        _completion_noise_std = None
        _completion_noise_snr = None
        if (completion_noise_K > 0
                and (verbose or diagnostics_sink is not None)
                and step_idx % completion_noise_every == 0
                and _n_masked > 0):
            # Re-sample completion K times for the top-1 candidate
            top_m = rewards[0].argmax().item()
            cand_top = candidates[0, top_m].unsqueeze(0)              # [1, L]
            is_mask_top = (cand_top == self.mask_index)                # [1, L]

            # Get p_θ(x_0|y^(top)) from the already-computed cand_log_p_x0
            top_idx = top_m  # within batch elem 0
            cand_p_top = cand_log_p_x0[top_idx, :, :acgt_size].exp() # [L, 4]
            if gamma != 1.0:
                cand_p_top = cand_p_top.pow(1.0 / gamma)
            cand_p_top = cand_p_top / (
                cand_p_top.sum(dim=-1, keepdim=True) + 1e-8)

            noise_rewards = []
            for _ in range(completion_noise_K):
                resample = Categorical(probs=cand_p_top).sample()     # [L]
                x_hat_k = torch.where(
                    is_mask_top[0], resample, cand_top[0]).unsqueeze(0)  # [1, L]
                oh = F.one_hot(x_hat_k, num_classes=acgt_size).float()
                r_k = reward_model(oh.transpose(1, 2))
                if r_k.ndim > 1:
                    r_k = r_k[:, 0]
                noise_rewards.append(r_k.item())

            import statistics
            _completion_noise_std = round(statistics.stdev(noise_rewards), 6) \
                if len(noise_rewards) > 1 else 0.0
            inter_std = rewards[0].std().item()
            _completion_noise_snr = round(
                (inter_std ** 2) / (_completion_noise_std ** 2 + 1e-12), 4)

        # ── Debug diagnostics ────────────────────────────────────────────────
        if verbose or diagnostics_sink is not None:
            diag = self._doit_collect_diagnostics(
                step_idx=step_idx,
                x=x, candidates=candidates,
                rewards=rewards, weights=weights, m_star=m_star,
                alpha=alpha, gamma=gamma, beta=beta, omega=omega,
                l_star=l_star, l_end=l_end,
                pre_fwd_unique=_pre_fwd_unique,
                pre_fwd_dedup_saving=_pre_fwd_dedup_saving,
                masked_entropy_mean=_masked_entropy_mean,
                n_masked=_n_masked,
                hamming_mean=_hamming_mean,
                hamming_norm=_hamming_norm,
                pos_agreement_mean=_pos_agreement_mean,
                r_2nd=_r_2nd,
                r_gap=_r_gap,
                w_max=_w_max,
                largest_cluster_frac=_largest_cluster_frac,
                cluster_sizes_str=_cluster_sizes_str,
                weighted_hamming=_weighted_hamming,
                completion_noise_std=_completion_noise_std,
                completion_noise_snr=_completion_noise_snr,
                # new fields
                n_newly_unmasked=_n_newly_unmasked,
                effective_unmask_rate=_effective_unmask_rate,
                token_entropy_at_new_pos=_token_entropy_at_new_pos,
                token_coverage_at_new_pos=_token_coverage_at_new_pos,
                min_token_group_size=_min_token_group_size,
                per_token_reward_gap=_per_token_reward_gap,
                per_token_reward_cv=_per_token_reward_cv,
                between_within_snr=_between_within_snr,
                hamming_reward_corr=_hamming_reward_corr,
                regret=_regret,
                rank_of_selected=_rank_of_selected,
                sel_token_rank_in_model=_sel_token_rank_in_model,
                sel_token_count_in_M=_sel_token_count_in_M,
                n_unique_tokens_at_new_pos=_n_unique_tokens_at_new_pos,
            )
            if verbose:
                self._doit_print_diagnostics(diag)
            if diagnostics_sink is not None:
                diagnostics_sink.append(diag)

        return x_next

    # ------------------------------------------------------------------
    # Debug diagnostics helper
    # ------------------------------------------------------------------

    @staticmethod
    def _doit_collect_diagnostics(
        step_idx: int,
        x: torch.Tensor,
        candidates: torch.Tensor,
        rewards: torch.Tensor,
        weights: torch.Tensor,
        m_star: torch.Tensor,
        alpha: float, gamma: float, beta: float, omega: float,
        l_star: int = None,
        l_end: int = None,
        pre_fwd_unique: int = None,
        pre_fwd_dedup_saving: float = None,
        masked_entropy_mean: float = None,
        n_masked: int = None,
        hamming_mean: float = None,
        hamming_norm: float = None,
        pos_agreement_mean: float = None,
        r_2nd: float = None,
        r_gap: float = None,
        w_max: float = None,
        largest_cluster_frac: float = None,
        cluster_sizes_str: str = None,
        weighted_hamming: float = None,
        completion_noise_std: float = None,
        completion_noise_snr: float = None,
        # new fields
        n_newly_unmasked: float = None,
        effective_unmask_rate: float = None,
        token_entropy_at_new_pos: float = None,
        token_coverage_at_new_pos: float = None,
        min_token_group_size: int = None,
        per_token_reward_gap: float = None,
        per_token_reward_cv: float = None,
        between_within_snr: float = None,
        hamming_reward_corr: float = None,
        regret: float = None,
        rank_of_selected: int = None,
        sel_token_rank_in_model: int = None,
        sel_token_count_in_M: int = None,
        n_unique_tokens_at_new_pos: int = None,
    ) -> dict:
        """Collect per-step DOIT diagnostics into a dict (batch element 0).

        Returns
        -------
        dict with keys: step_idx, new_pos, n_masked, cand_div, unique_cands, M,
                        pre_fwd_unique, pre_fwd_dedup_saving,
                        masked_entropy_mean,
                        hamming_mean, hamming_norm, pos_agreement_mean,
                        r_mean, r_std, r_min, r_max, r_2nd, r_gap, w_max,
                        ess, entropy_bits,
                        completion_noise_std, completion_noise_snr,
                        m_star, alpha, gamma, beta, omega, l_star, l_end.
        """
        B, M, L = candidates.shape

        # ── Candidate diversity (newly-unmasked positions only) ──────────────
        mask_id = candidates[0, 0].max().item()
        newly_unmasked = (x[0] == mask_id)        # [L]  bool
        n_new = newly_unmasked.sum().item()
        if n_new > 0:
            new_tokens = candidates[0, :, newly_unmasked]  # [M, n_new]
            unique_cands = len(set(tuple(r.tolist()) for r in new_tokens))
            diversity = unique_cands / M
        else:
            diversity = float('nan')
            unique_cands = 0

        # ── Reward statistics (batch elem 0) ────────────────────────────────
        r0 = rewards[0].float()
        r_mean = r0.mean().item()
        r_std  = r0.std().item()
        r_min  = r0.min().item()
        r_max  = r0.max().item()

        # ── Effective sample size ────────────────────────────────────────────
        w0  = weights[0].float()
        ess = (1.0 / (w0 ** 2).sum()).item()

        # ── Weight entropy (bits) ────────────────────────────────────────────
        eps = 1e-8
        entropy_bits = -(w0 * (w0 + eps).log2()).sum().item()

        return dict(
            step_idx=step_idx,
            new_pos=n_new,
            n_masked=n_masked,
            cand_div=round(diversity, 6) if diversity == diversity else None,
            unique_cands=unique_cands,
            M=M,
            pre_fwd_unique=pre_fwd_unique,
            pre_fwd_dedup_saving=pre_fwd_dedup_saving,
            masked_entropy_mean=masked_entropy_mean,
            hamming_mean=hamming_mean,
            hamming_norm=hamming_norm,
            pos_agreement_mean=pos_agreement_mean,
            r_mean=round(r_mean, 6),
            r_std=round(r_std, 6),
            r_min=round(r_min, 6),
            r_max=round(r_max, 6),
            r_2nd=r_2nd,
            r_gap=r_gap,
            w_max=w_max,
            largest_cluster_frac=largest_cluster_frac,
            cluster_sizes_str=cluster_sizes_str,
            weighted_hamming=weighted_hamming,
            ess=round(ess, 4),
            entropy_bits=round(entropy_bits, 4),
            completion_noise_std=completion_noise_std,
            completion_noise_snr=completion_noise_snr,
            m_star=int(m_star[0].item()),
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            omega=omega,
            l_star=l_star,
            l_end=l_end,
            # new fields
            n_newly_unmasked=n_newly_unmasked,
            effective_unmask_rate=effective_unmask_rate,
            token_entropy_at_new_pos=token_entropy_at_new_pos,
            token_coverage_at_new_pos=token_coverage_at_new_pos,
            min_token_group_size=min_token_group_size,
            per_token_reward_gap=per_token_reward_gap,
            per_token_reward_cv=per_token_reward_cv,
            between_within_snr=between_within_snr,
            hamming_reward_corr=hamming_reward_corr,
            regret=regret,
            rank_of_selected=rank_of_selected,
            sel_token_rank_in_model=sel_token_rank_in_model,
            sel_token_count_in_M=sel_token_count_in_M,
            n_unique_tokens_at_new_pos=n_unique_tokens_at_new_pos,
        )

    @staticmethod
    def _doit_print_diagnostics(diag: dict) -> None:
        """Print a diagnostics dict produced by _doit_collect_diagnostics."""
        # Pre-forward dedup info
        pfu = diag.get('pre_fwd_unique')
        pfd = diag.get('pre_fwd_dedup_saving')
        ment = diag.get('masked_entropy_mean')
        dedup_str = (
            f"pre_uniq={pfu}/{diag['M']} save={pfd:.0%}  "
            f"H_mask={ment:.3f}  "
            if pfu is not None else ""
        )
        # Extended metrics
        hmn = diag.get('hamming_norm')
        ext_str = ""
        if hmn is not None:
            ext_str += f"ham_n={hmn:.4f} "
        lcf = diag.get('largest_cluster_frac')
        if lcf is not None:
            ext_str += f"top_clust={lcf:.0%} "
        wh = diag.get('weighted_hamming')
        if wh is not None:
            ext_str += f"wham={wh:.4f} "
        pag = diag.get('pos_agreement_mean')
        if pag is not None:
            ext_str += f"agree={pag:.3f} "
        rg = diag.get('r_gap')
        if rg is not None:
            ext_str += f"r_gap={rg:.4f} "
        wm = diag.get('w_max')
        if wm is not None:
            ext_str += f"w_max={wm:.4f} "
        snr = diag.get('completion_noise_snr')
        if snr is not None:
            ext_str += f"SNR={snr:.2f} "
        # new metrics
        tok_ent = diag.get('token_entropy_at_new_pos')
        if tok_ent is not None:
            ext_str += f"tok_ent={tok_ent:.2f}b "
        tok_cov = diag.get('token_coverage_at_new_pos')
        if tok_cov is not None:
            ext_str += f"tok_cov={tok_cov:.2f} "
        bwsnr = diag.get('between_within_snr')
        if bwsnr is not None:
            ext_str += f"bw_snr={bwsnr:.3f} "
        rgt = diag.get('regret')
        if rgt is not None:
            ext_str += f"regret={rgt:.4f} "
        rnk = diag.get('rank_of_selected')
        if rnk is not None:
            ext_str += f"rank={rnk} "
        stk_rank = diag.get('sel_token_rank_in_model')
        stk_cnt  = diag.get('sel_token_count_in_M')
        n_utok   = diag.get('n_unique_tokens_at_new_pos')
        if stk_rank is not None:
            ext_str += f"tok_rank={stk_rank}/4 cnt={stk_cnt}/{diag['M']} utok={n_utok} "
        print(
            f"[DOIT step {diag['step_idx']:4d}] "
            f"n_mask={diag.get('n_masked', '?'):>3} "
            f"α={diag['alpha']:.2f} γ={diag['gamma']:.2f} "
            f"β={diag['beta']:.2f} ω={diag['omega']:.1f} | "
            f"new_pos={diag['new_pos']:3d}  "
            f"{dedup_str}"
            f"cand_div={diag['cand_div']:.3f} "
            f"({diag['unique_cands']}/{diag['M']} uniq) | "
            f"{ext_str}| "
            f"R: {diag['r_mean']:.4f}±{diag['r_std']:.4f} "
            f"[{diag['r_min']:.4f},{diag['r_max']:.4f}] | "
            f"ESS={diag['ess']:.1f}/{diag['M']}  "
            f"H={diag['entropy_bits']:.2f}b  "
            f"m*={diag['m_star']}"
        )

    # ------------------------------------------------------------------
    # Per-step denoising update (Algorithm 2, full loop body)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _ddpm_update_controlled_DOIT(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        reward_model,
        step_idx: int,
        num_steps: int,
        l_star: int,
        l_end: int,
        M: int,
        omega: float,
        beta: float,
        alpha: float,
        gamma: float,
        verbose: bool = False,
        diagnostics_sink: list = None,
        candidate_sink: list = None,
        completion_noise_K: int = 0,
        completion_noise_every: int = 5,
        M_schedule: str = "fixed",
        M_min: int = 4,
        M_thresh_high: int = 80,
        M_thresh_low: int = 20,
        omega_schedule: str = "fixed",
        omega_min: float = 0.5,
        omega_thresh_high: int = 80,
        omega_thresh_low: int = 20,
        phase_strategy: str = "none",
        phase_skip_thresh: int = 100,
        phase_active_thresh: int = 20,
        phase_M_refine: int = 8,
        phase_omega_refine_scale: float = 1.5,
        resample_strategy: str = "per_sequence",
    ) -> torch.Tensor:
        """Single MDLM denoising step with optional DOIT guidance.

        Follows Algorithm 2 (extended for l_end):
          if l_end ≤ l ≤ l*:  two-stage proposal-resample
          else:               standard MDLM transition

        Args:
            x:          LongTensor [B, L]   current state x_l.
            t:          Tensor [B, 1]       current timestep t_l.
            dt:         float               timestep interval.
            reward_model: callable          reward oracle.
            step_idx:   int                 0-based step index (0 = noisiest).
            num_steps:  int                 total denoising steps.
            l_star:     int                 guidance start — apply for l ≤ l*.
            l_end:      int                 guidance stop — apply for l ≥ l_end
                                            (i.e. stop guiding below this step).
            M:          int                 MC candidates per batch element.
            omega:      float               guidance strength (resampling).
            beta:       float               reward temperature.
            alpha:      float               proposal temperature.
            gamma:      float               completion temperature for x̂_0 proxy.
            verbose:    bool                print per-step diagnostics.
            diagnostics_sink: list | None   if not None, append a dict per guided step.

        Returns:
            x_next: LongTensor [B, L]   next state x_{l-1}.
        """
        # Compute σ_t and σ_s from the noise schedule
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)   # [B]
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)   # [B]
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape

        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]  # [B, 1, 1]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]  # [B, 1, 1]

        # ── Single forward pass (reused) ────────────────────────────────────
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)             # [B, L, V]
        assert move_chance_t.ndim == log_p_x0.ndim

        # ── Guidance window: l_end ≤ l ≤ l*  (and l > 1) ──────────────────
        # step_idx=0 → l=L (noisiest); step_idx=num_steps-1 → l=1 (last).
        # l ≤ l*    ⟺  step_idx ≥ num_steps − l*
        # l ≥ l_end ⟺  step_idx ≤ num_steps − l_end
        # l  > 1    ⟺  step_idx < num_steps − 1
        in_guidance_window = (
            step_idx >= num_steps - l_star          # l ≤ l*
            and step_idx <= num_steps - l_end       # l ≥ l_end
            and step_idx < num_steps - 1            # l > 1
        )

        if in_guidance_window:
            # Always compute n_masked (cheap) — needed by phase / dynamic schedules
            M_eff = M
            omega_eff = omega
            n_masked_val = int(
                (x == self.mask_index).sum(dim=-1).float().mean().item()
            )

            # ── Adaptive phase override ──────────────────────────────────
            if phase_strategy == "adaptive":
                M_eff, omega_eff, skip = self._compute_phase_config(
                    n_masked_val, M_max=M, omega_max=omega,
                    skip_thresh=phase_skip_thresh,
                    active_thresh=phase_active_thresh,
                    M_refine=phase_M_refine,
                    omega_refine_scale=phase_omega_refine_scale,
                )
                if skip:
                    # Phase 1: skip guidance, fall through to standard MDLM
                    in_guidance_window = False

        if in_guidance_window:
            # ── Dynamic M: compute from n_masked if requested ────────────
            if M_schedule in ("dynamic", "dynamic_reverse"):
                M_eff = self._compute_dynamic_M(
                    n_masked_val, M_max=M, M_min=M_min,
                    thresh_high=M_thresh_high, thresh_low=M_thresh_low,
                    reverse=(M_schedule == "dynamic_reverse"),
                )

            # ── Dynamic omega: compute from n_masked if requested ────────
            if omega_schedule in ("dynamic", "dynamic_reverse"):
                omega_eff = self._compute_dynamic_omega(
                    n_masked_val, omega_max=omega, omega_min=omega_min,
                    thresh_high=omega_thresh_high, thresh_low=omega_thresh_low,
                    reverse=(omega_schedule == "dynamic_reverse"),
                )

            # Two-stage proposal-resample (1 extra NFE)
            return self._doit_proposal_resample(
                x, log_p_x0, sigma_s,
                move_chance_t, move_chance_s,
                reward_model,
                M=M_eff, omega=omega_eff, beta=beta, alpha=alpha, gamma=gamma,
                verbose=verbose, step_idx=step_idx,
                diagnostics_sink=diagnostics_sink,
                candidate_sink=candidate_sink,
                l_star=l_star, l_end=l_end,
                completion_noise_K=completion_noise_K,
                completion_noise_every=completion_noise_every,
                resample_strategy=resample_strategy,
            )
        else:
            # Standard MDLM transition (no guidance)
            q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
            q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
            _x = _sample_categorical(q_xs)
            copy_flag = (x != self.mask_index).to(x.dtype)
            return copy_flag * x + (1 - copy_flag) * _x

    # ------------------------------------------------------------------
    # Sampling loop (Algorithm 2, outer for-loop)
    # ------------------------------------------------------------------

    def controlled_sample_DOIT(
        self,
        reward_model,
        *,
        l_star: int = 64,
        l_end: int = 1,
        M: int = 32,
        omega: float = 1.0,
        beta: float = 0.5,
        alpha: float = 1.5,
        gamma: float = 1.2,
        verbose: bool = False,
        diagnostics_sink: list = None,
        candidate_sink: list = None,
        num_steps: int = None,
        eps: float = 1e-5,
        eval_sp_size: int = None,
        completion_noise_K: int = 0,
        completion_noise_every: int = 5,
        M_schedule: str = "fixed",
        M_min: int = 4,
        M_thresh_high: int = 80,
        M_thresh_low: int = 20,
        omega_schedule: str = "fixed",
        omega_min: float = 0.5,
        omega_thresh_high: int = 80,
        omega_thresh_low: int = 20,
        phase_strategy: str = "none",
        phase_skip_thresh: int = 100,
        phase_active_thresh: int = 20,
        phase_M_refine: int = 8,
        phase_omega_refine_scale: float = 1.5,
        best_of_k: int = 1,
        resample_strategy: str = "per_sequence",
    ) -> torch.Tensor:
        """Generate sequences using Discrete DOIT guidance (Algorithm 2).

        Compatible with ``eval.py``'s
        ``sample_controlled(model, "DOIT", reward_model, ...)``.

        Args:
            reward_model:      callable  [B, 4, L] float one-hot → [B, num_targets].
            l_star:            guidance start — apply in the final l* steps (l ≤ l*).
            l_end:             guidance stop  — stop below this step (l ≥ l_end).
                               Guidance is active for l_end ≤ l ≤ l*.
            M:                 MC candidates per guided step.
            omega:             guidance strength — exponent on ĥ in resampling.
            beta:              reward temperature  ĥ=exp(R/β).
            alpha:             proposal temperature for candidate diversity.
            gamma:             completion temperature for x̂_0 proxy.
            verbose:           if True, print per-step diagnostics.
            diagnostics_sink:  list | None  append a dict per guided step.
            num_steps:         total denoising steps (default from config).
            eps:               small constant to avoid t=0 singularity.
            eval_sp_size:      batch size override (default from config).

        Returns:
            x: LongTensor [B, L] — generated discrete sequences.
        """
        if self.parameterization == 'ar':
            bsz = eval_sp_size or self.config.loader.eval_batch_size
            return self._ar_sampler(bsz)

        batch_size = eval_sp_size or self.config.loader.eval_batch_size
        num_steps  = num_steps   or self.config.sampling.steps
        timesteps  = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt         = (1 - eps) / num_steps

        # ── Helper: guidance-window predicate ───────────────────────────────
        def _in_window(step_idx: int) -> bool:
            return (
                step_idx >= num_steps - l_star   # l ≤ l*
                and step_idx <= num_steps - l_end # l ≥ l_end
                and step_idx < num_steps - 1      # l > 1
            )

        # ══════════════════════════════════════════════════════════════════
        # Original single-trajectory + M-candidates-per-step (Algorithm 2).
        # When best_of_k > 1 the batch is inflated to B×K, ONE diffusion
        # loop runs over B×K sequences, then top-B are kept by reward
        # (same compute as K separate runs but in one pass).
        # ══════════════════════════════════════════════════════════════════
        actual_bs = batch_size * max(1, best_of_k)
        x = self._sample_prior(actual_bs, self.config.model.length).to(self.device)

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update_controlled_DOIT(
                    x, t, dt, reward_model,
                    step_idx=i,
                    num_steps=num_steps,
                    l_star=l_star,
                    l_end=l_end,
                    M=M,
                    omega=omega,
                    beta=beta,
                    alpha=alpha,
                    gamma=gamma,
                    verbose=verbose,
                    diagnostics_sink=diagnostics_sink,
                    candidate_sink=candidate_sink,
                    completion_noise_K=completion_noise_K,
                    completion_noise_every=completion_noise_every,
                    M_schedule=M_schedule,
                    M_min=M_min,
                    M_thresh_high=M_thresh_high,
                    M_thresh_low=M_thresh_low,
                    omega_schedule=omega_schedule,
                    omega_min=omega_min,
                    omega_thresh_high=omega_thresh_high,
                    omega_thresh_low=omega_thresh_low,
                    phase_strategy=phase_strategy,
                    phase_skip_thresh=phase_skip_thresh,
                    phase_active_thresh=phase_active_thresh,
                    phase_M_refine=phase_M_refine,
                    phase_omega_refine_scale=phase_omega_refine_scale,
                    resample_strategy=resample_strategy,
                )
            else:
                # Analytic sampler — fall back to unguided
                x = self._analytic_update(x, t, dt)

        # Optional final denoising step (noise_removal)
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                logits = self.forward(x, unet_conditioning)
                x = logits[:, :, :-1].argmax(dim=-1)

        # ── Best-of-K: each sequence picks its best particle ─────────────
        if best_of_k > 1:
            K = best_of_k
            acgt = self.mask_index  # 4
            oh = F.one_hot(x, num_classes=acgt).float()
            scores = reward_model(oh.transpose(1, 2))
            if scores.ndim > 1:
                scores = scores[:, 0]
            # x: [B*K, L] → [B, K, L], scores: [B*K] → [B, K]
            x = x.view(batch_size, K, -1)                            # [B, K, L]
            scores = scores.view(batch_size, K)                      # [B, K]
            best_k = scores.argmax(dim=1)                            # [B]
            x = x[torch.arange(batch_size, device=x.device), best_k] # [B, L]

        return x
