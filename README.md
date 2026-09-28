# D-DOIT Code Artifact

This repository contains code for training-free inference-time guidance of
masked discrete diffusion models for regulatory DNA sequence design. This is
the public code release accompanying the D-DOIT paper.

The artifact includes:

- DRAKES/Gosai enhancer-design generation and evaluation code under `ddoit/`.
- Ctrl-DNA benchmark adapters under `ddoit/`.

Tests, raw experiment logs, paper result tables, cluster-specific launch
scripts, local machine paths, and private operational notes are intentionally
excluded from this release.

## Repository Layout

```text
.
├── ddoit/              # benchmark adapters, runtimes, and CLI entry points
└── requirements.txt    # Python runtime dependencies
```

## Environment

The code was developed for Python 3.9, CUDA 12.1, PyTorch 2.3.1, Lightning,
Hydra, and gReLU 1.0.2.

```bash
conda create -n ddoit python=3.9 -y
conda activate ddoit

# CUDA 12.1 build used by the DRAKES runtime.
pip install \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Remaining runtime dependencies.
pip install -r requirements.txt
```

The dependency list pins `numpy<2.0` for gReLU compatibility. Install the
PyTorch build matching your CUDA runtime if the default wheel is not suitable
for your system.

## DRAKES Data And Checkpoints

The DRAKES/Gosai data and model weights are public. The
[original DRAKES repository](https://github.com/ChenyuWang-Monica/DRAKES)
provides them as a single archive:

```bash
mkdir -p /path/to/drakes_assets
cd /path/to/drakes_assets
curl -L -o DRAKES_data.zip \
  "https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=1"
unzip DRAKES_data.zip
```

After extraction, set `DRAKES_BASE_PATH` to the extracted `data_and_model`
directory:

```bash
export DRAKES_BASE_PATH=/path/to/drakes_assets/data_and_model
```

The expected layout is:

```text
data_and_model/
└── mdlm/
    ├── gosai_data/
    │   ├── gosai_dataset.h5
    │   ├── processed_data/gosai_all.csv
    │   └── binary_atac_cell_lines.ckpt
    ├── outputs_gosai/
    │   ├── pretrained.ckpt
    │   ├── cfg.ckpt
    │   └── lightning_logs/
    │       ├── reward_oracle_ft.ckpt
    │       └── reward_oracle_eval.ckpt
    └── reward_bp_results_final/
        ├── finetuned.ckpt
        └── zero_alpha.ckpt
```

For the quick D-DOIT run below, the required files are the Gosai data,
`pretrained.ckpt`, `reward_oracle_ft.ckpt`, `reward_oracle_eval.ckpt`, and
`binary_atac_cell_lines.ckpt`. The other checkpoints are needed only when
running the corresponding DRAKES baselines.

## Ctrl-DNA Assets

The Ctrl-DNA adapter is included in the code, but this release does not provide
Ctrl-DNA data or checkpoints. Those assets were prepared separately and should
be supplied explicitly by anyone running the Ctrl-DNA benchmark.

## Weights & Biases

gReLU may use Weights & Biases internally when loading oracle assets. If your
runtime requires W&B authentication, export a reviewer-local key:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

Do not commit credentials. If you need a specific W&B entity for training runs,
set:

```bash
export WANDB_ENTITY=<your_entity>
```

Evaluation does not require this repository to contain any author-specific W&B
account name.

## Quick Evaluation

From the repository root:

```bash
conda activate ddoit
export DRAKES_BASE_PATH=/path/to/drakes_assets/data_and_model

python -m ddoit.benchmarks.drakes_generation --methods DOIT_LBOK \
  --doit-l-star 128 \
  --doit-M 10 \
  --doit-omega 3.0 \
  --doit-beta 0.2 \
  --doit-alpha 3.0 \
  --doit-gamma 1.2 \
  --best-of-k 4 \
  --doit-bok-split-step 105 \
  --num-sample-batches 10 \
  --num-samples-per-batch 64 \
  --seed 42 \
  --output-dir outputs/reproduce_lbok_seed42
```

The output metrics are written to
`outputs/reproduce_lbok_seed42/metrics.json`.

## Unified CLI

The `ddoit` namespace exposes benchmark-level generation and evaluation entry
points:

```bash
python -m ddoit.cli.list_benchmarks
python -m ddoit.cli.generate --benchmark drakes --method DOIT_LBOK --num-sequences 64
python -m ddoit.cli.evaluate --benchmark ctrldna --input <sequences.csv>
```

Ctrl-DNA support is transitional. If using the legacy Ctrl-DNA adapter, provide
the legacy checkout and private assets explicitly:

```bash
export DDOIT_LEGACY_CTRLDNA_ROOT=/path/to/Ctrl-DNA
```

## Attribution

This code builds on DRAKES and MDLM. The public DRAKES data/checkpoint archive
is provided by the original DRAKES authors.
