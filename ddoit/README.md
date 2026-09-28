# ddoit

Unified code namespace for the D-DOIT project.

The target architecture is one code repo with:

- shared D-DOIT guidance logic in `ddoit.core`
- benchmark-specific adapters in `ddoit.benchmarks`
- one future CLI that selects `--benchmark drakes` or `--benchmark ctrldna`

Current status:

- DRAKES generation/evaluation is now orchestrated by
  `ddoit.benchmarks.drakes_evaluator`.
- Ctrl-DNA generation now enters through
  `ddoit.benchmarks.ctrldna_generation`; the retained external repo still
  supplies the heavy MDLM model package while migration continues.
- `ddoit/` records the stable interface contract and benchmark registry for the migration.
- `ddoit.benchmarks.drakes` owns dependency-free DRAKES method aliases,
  supported task metadata, and generation command planning.
- `ddoit.benchmarks.drakes_evaluation` owns shared DRAKES sampling and k-mer
  evaluation helpers.
- `ddoit.benchmarks.drakes_oracle` owns DRAKES oracle/reward accessors and
  lazily loads the package-owned oracle runtime.
- `ddoit.benchmarks.drakes_oracle_runtime` owns the migrated DRAKES
  oracle/evaluation helper implementation.
- `ddoit.benchmarks.drakes_data` owns DRAKES DNA tokenization helpers used by
  the native evaluator.
- `ddoit.benchmarks.drakes_paths` owns DRAKES data/model path resolution and
  the `DRAKES_BASE_PATH` override.
- `ddoit.benchmarks.drakes_base_config` and
  `ddoit.benchmarks.drakes_dataloader` own DRAKES base path compatibility,
  datasets, tokenization compatibility, and samplers.
- `ddoit.benchmarks.drakes_models` owns DRAKES diffusion model module imports
  and seed setup behind a lazy adapter.
- `ddoit.benchmarks.drakes_diffusion_gosai_update` owns the migrated base
  DRAKES diffusion runtime used for pretrained/finetuned sampling.
- `ddoit.benchmarks.drakes_diffusion_gosai_cfg` owns the migrated CFG
  diffusion runtime.
- `ddoit.benchmarks.drakes_diffusion_doit` and
  `ddoit.benchmarks.drakes_doit_late_bok` own the migrated D-DOIT and
  D-DOIT-LBOK-S runtimes.
- `ddoit.benchmarks.drakes_diffusion_cg`,
  `ddoit.benchmarks.drakes_diffusion_smc`, and
  `ddoit.benchmarks.drakes_diffusion_tds` own the migrated CG, SMC, and TDS
  controlled sampling runtimes.
- `ddoit.benchmarks.drakes_noise` owns DRAKES noise schedules used by diffusion
  runtimes.
- `ddoit.benchmarks.drakes_models_pkg` owns DRAKES CNN backbone and EMA model
  helpers used by diffusion runtimes.
- `ddoit.benchmarks.drakes_utils` owns DRAKES diffusion utility helpers used
  by diffusion runtimes.
- `ddoit.benchmarks.drakes_configs_gosai` owns the DRAKES Hydra configs used
  by the native evaluator.
- `ddoit.benchmarks.ctrldna` owns dependency-free Ctrl-DNA task metadata,
  method aliases, oracle checkpoint path rules, and generation command
  planning.
- `ddoit.benchmarks.ctrldna_generation` owns the Ctrl-DNA guided generation
  runtime entrypoint used by `ddoit.cli.generate`.
- `ddoit.benchmarks.ctrldna_reward` and `ddoit.benchmarks.ctrldna_evaluation`
  migrate the Ctrl-DNA reward and evaluation surfaces while keeping heavy
  imports lazy.
- `ddoit.cli.generate` provides the stable benchmark-selection interface.
- `ddoit.cli.evaluate` currently exposes native Ctrl-DNA sequence evaluation.
