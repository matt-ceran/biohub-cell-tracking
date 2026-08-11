# Biohub Cell Tracking

This repository contains our work for the Kaggle competition `Biohub - Cell Tracking During Development`.
The goal is to detect zebrafish cell centers in 3D microscopy movies, link the same cells through time, detect divisions, and write a valid `submission.csv`.

## Current Status

Phase 10 completed a locked 71-movie validation of a learned 3D appearance filter.
The retained pipeline uses DoG k=3 proposals, a CNN score threshold of 0.20, whole-movie min-cost-flow linking, and pruning at minimum track length 6.
It improved mean local edge Jaccard from 0.668489 to 0.674186 while reducing aggregate predicted-to-estimated node ratio from 1.446x to 1.303x and raising sparse-label recall from 0.876371 to 0.894906.
The learned policy won, tied, and lost on 36, 4, and 31 movies respectively, and its median edge Jaccard was slightly lower, so the improvement is real but modest and heterogeneous.
Geometric division repair remains off because it measured as a net loss.
Phase 11 reproduced the frozen Phase 10 path, froze a bounded high-recall mother-and-daughter proposal rule, and built 108 strictly validated node caches for 88 fit movies and 20 development movies.
The sampled fit dataset contains 6,620 records, including 3,804 supervised records and 2,816 zero-weight diagnostic unknowns.
The frozen proposal rule represents 76 of 89 known fit divisions and 10 of 18 known development divisions, so the division heads cannot learn or recover events whose correct triplets are missing.
The fixed `scalar`, `embedding_linear`, and `embedding_mlp32` heads each completed exactly 100 production epochs on the fit cohort.
Their final fit losses were 0.299763, 0.179191, and 0.000540 respectively, but fit loss does not select a model or prove movie-level improvement.
The complete [Phase 11 fixed-head training record](experiments/2026-08-02-phase11-fixed-head-training.md) preserves the commands, data identities, artifact hashes, runtime, validation evidence, and interpretation limits.
The complete 660-row development grid has now evaluated all 20 development movies across all three heads and eleven thresholds.
None of the 33 policies passed every frozen guard, and none recovered one exact mother-plus-two-daughters triplet.
At the strictest threshold of 0.9999, scalar, embedding-linear, and embedding-MLP32 still created 7, 63, and 98 strict false forks respectively.
The complete [Phase 11 development-grid record](experiments/2026-08-11-phase11-development-grid.md) preserves the command, full failure pattern, guard results, hashes, validation evidence, and stop-before-policy-freezing boundary.
The development-only truth-rank atlas has now traced all 18 labelled division events through proposal generation, head ranking, thresholds, and movie assignment.
The frozen proposal rule exposed 10 exact truths and hid 8 before scoring: 4 lacked a raw true daughter, 2 lacked the linked mother, and 2 placed a daughter beyond the nearest-12 cap.
Among the 10 reachable truths, scalar, embedding-linear, and embedding-MLP32 placed the exact pair first for 1, 0, and 3 events respectively.
None survived assignment because most lost to a wrong same-mother pair or fell below threshold, while all four head-event rank-1 results were crowded out by higher-scoring false mothers at the five-fork movie cap.
The complete [Phase 11 truth-rank atlas record](experiments/2026-08-11-phase11-truth-rank-atlas.md) preserves the event ranks, stopping stages, contact-sheet evidence, artifact identities, and next modeling workstreams.
No division policy has been frozen and no development winner exists.
All 20 evaluation caches remain absent and evaluation remains fail-closed.

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

## Mother-To-Daughter Proposal Audit

The first Phase 11 slice preserves the accepted Phase 10 pipeline and measures where labelled division evidence disappears before any new model is trained.
It reuses the exact Phase 10 candidate-score caches, verifies all frozen SHA-256 identities, reruns filtering and linking, and refuses a movie whose divisions-off graph does not reproduce its saved Phase 10 result.

Run the complete 13-movie internal audit.

```bash
.venv/bin/python -u scripts/audit_division_proposals.py \
  --output-dir data/working/phase11_division_audit/full13_orphan
```

The generated event, proposal, movie, and summary artifacts stay under ignored `data/working/` paths.
The measured result rejects a post-prune orphan-only scorer because its labelled proposal recall is 0 of 18.
Division evidence must be generated before pruning, and current parent ownership, orphan status, distance, angle, and track context must be treated as features rather than final geometric acceptance rules.

The second Phase 11 audit removes the current-child and orphan prerequisites and evaluates a frozen grid of unordered daughter-pair rules.
It keeps mothers in the CNN-filtered pre-prune graph while comparing filtered daughters with tightly local raw-scored daughters.

```bash
.venv/bin/python -u scripts/audit_division_pair_proposals.py \
  --output-dir data/working/phase11_division_audit/full13_pairs
```

The smallest rule retaining all 18 labelled events uses a 16 micrometer maximum distance, no minimum distance or angle gate, and at most 12 nearest candidates per mother.
It produces 22,308,810 raw-scored proposals from 362,366 unique proposed mothers.
The same geometry over filtered candidates covers 16 of 18 events with 18,210,993 proposals.
All three no-labelled-division controls contain proposals, totaling 3,331,415 under the selected raw-scored rule.
This is a bounded high-recall proposal contract, not a fork-acceptance rule or a default un-sampled training table.

## Learned Division Pipeline

The Phase 11 implementation keeps the accepted Phase 10 appearance encoder frozen.
Each cache stores candidate positions, cellness scores, 64-value appearance embeddings, and intensity summaries so later work can build division records without running the 3D CNN again.
The exact split and cache provenance are checked before any movie is read.

Build the frozen fit and development cache cohorts.

```bash
.venv/bin/python -u scripts/build_division_cache.py \
  --role fit \
  --batch-size 128 \
  --device mps

.venv/bin/python -u scripts/build_division_cache.py \
  --role development \
  --batch-size 128 \
  --device mps
```

Build the bounded sampled fit dataset and learn normalization statistics only from fit records.

```bash
.venv/bin/python -u scripts/build_division_dataset.py --fit
```

Train the fixed three-head ladder for exactly 100 production epochs per head.
The scalar head uses only the 58 context values, the embedding-linear head uses all 314 values with one linear decision layer, and the embedding-MLP32 head uses all 314 values with one 32-unit hidden layer.

```bash
.venv/bin/python -u scripts/train_division.py --all --device mps
```

The complete 660-row development grid has run and found no policy that passes every declared guard.
The exact reproduction command uses the frozen default CPU execution contract.

```bash
.venv/bin/python -u scripts/validate_division_model.py \
  --cohort development
```

Replay the exact known truth proposals through the completed grid and generate the development-only contact-sheet atlas.

```bash
.venv/bin/python -u scripts/audit_division_truth_ranks.py \
  --cohort development
```

The atlas separates proposal recall, within-mother ranking, thresholding, and movie-wide assignment failures.
It writes canonical JSON plus 10 mother-and-daughter contact sheets under ignored `data/working/phase11_division_truth_rank_atlas/`.
It does not alter model weights, select a policy, or open evaluation.

After explicit review, freeze the development rejection.
The freezer will record `evaluation_unlocked=false` and leave the accepted divisions-off fallback unchanged.

```bash
.venv/bin/python scripts/freeze_division_policy.py
```

The locked evaluation cohort cannot be cached or inspected until strict replay of the exact frozen development result authorizes it.

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
