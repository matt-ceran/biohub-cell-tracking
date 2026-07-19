# Biohub Cell Tracking

This repository contains our work for the Kaggle competition `Biohub - Cell Tracking During Development`.
The goal is to detect zebrafish cell centers in 3D microscopy movies, link the same cells through time, detect divisions, and write a valid `submission.csv`.

## Current Status

Phase 10 completed a locked 71-movie validation of a learned 3D appearance filter.
The retained pipeline uses DoG k=3 proposals, a CNN score threshold of 0.20, whole-movie min-cost-flow linking, and pruning at minimum track length 6.
It improved mean local edge Jaccard from 0.668489 to 0.674186 while reducing aggregate predicted-to-estimated node ratio from 1.446x to 1.303x and raising sparse-label recall from 0.876371 to 0.894906.
The learned policy won, tied, and lost on 36, 4, and 31 movies respectively, and its median edge Jaccard was slightly lower, so the improvement is real but modest and heterogeneous.
Geometric division repair remains off because it measured as a net loss.
The next modeling target is learned mother-and-daughter appearance, alongside submission packaging and runtime hardening.

## Layout

- `src/biohub/` contains reusable project code.
- `tests/` contains synthetic tests that do not require the competition dataset.
- `scripts/` contains runnable project utilities.
- `data/raw/` is for downloaded Kaggle data and is ignored by Git.
- `data/working/` is for generated intermediate files and is ignored by Git.
- `submissions/` is for generated submission CSV files and is ignored by Git.

## Setup

Create and activate a local environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,notebooks]"
```

The Kaggle dataset should be downloaded or attached under `data/raw/`.
Do not commit raw competition data.

## Learned Appearance Pipeline

The first learned model is a small 3D CNN that scores image patches centered on DoG k=3 candidates.
GEFF-matched candidates are trusted positives, while all other candidates remain explicitly unlabeled because sparse annotations do not make them background.
Training uses a non-negative positive-unlabeled objective and rejects embryo `44b6` at the dataset boundary.

Install the optional ML dependencies.

```bash
python -m pip install -e ".[dev,ml]"
```

Build patch shards from embryo `6bba` only.
The first command creates a one-movie smoke shard, while `--all` creates the complete 128-movie training set.

```bash
python scripts/build_appearance_dataset.py --dataset 6bba_0e7c0d07
python scripts/build_appearance_dataset.py --all
```

Train with a deterministic movie-level internal development split inside embryo `6bba`.
This is the accepted full-training invocation.

```bash
.venv/bin/python -u scripts/train_appearance.py \
  --shards data/working/appearance_shards \
  --checkpoint data/working/appearance_checkpoints/cellness_cnn_phase10_full.pt \
  --epochs 40 \
  --batch-size 64 \
  --dev-batch-size 256 \
  --base-channels 16 \
  --dropout 0.2 \
  --device mps \
  --require-movies 128 \
  --patience 6 \
  --selection-metric pu-auc \
  --steps-per-epoch 250
```

Run the learned inference boundary after DoG proposal generation.
The generated NPZ keeps every candidate score so later linking experiments can sweep the decision threshold without rescoring the image patches.

```bash
.venv/bin/python scripts/score_appearance.py \
  6bba_0e7c0d07 \
  --split train \
  --checkpoint data/working/appearance_checkpoints/cellness_cnn_phase10_full.pt \
  --threshold 0.20
```

Sweep the complete internal-development policy grid in two stages, then freeze the winner.
The second validation command resumes the same CSV and adds the refined frontier.

```bash
.venv/bin/python -u scripts/validate_appearance.py \
  --cohort internal-dev \
  --score-thresholds 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --min-lengths 1,2,3,8 \
  --batch-size 128

.venv/bin/python -u scripts/validate_appearance.py \
  --cohort internal-dev \
  --score-thresholds 0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2 \
  --min-lengths 4,5,6,7,8 \
  --batch-size 128 \
  --resume

.venv/bin/python scripts/freeze_appearance_policy.py
```

The locked holdout command accepts only the frozen policy and exact 71-movie `44b6` cohort.
It refuses manual threshold changes, partial cohorts, or mismatched source and checkpoint fingerprints.

```bash
.venv/bin/python -u scripts/validate_appearance.py \
  --cohort locked-holdout \
  --policy data/working/phase10_frozen_candidate_policy.json \
  --confirm-locked-holdout \
  --batch-size 128

.venv/bin/python scripts/summarize_appearance_holdout.py
```

The final comparison is stored in `data/working/phase10_locked_comparison.json`.

## First Checks

Run the synthetic tests.

```bash
python -m pytest
```

Inspect the local dataset after download.

```bash
python scripts/inspect_dataset.py data/raw
```

Download the competition data after Kaggle authentication and rules acceptance.

```bash
python scripts/download_data.py
```

If Kaggle authentication is not set up yet, run:

```bash
source .venv/bin/activate
kaggle auth login
```
