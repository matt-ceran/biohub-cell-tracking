# Biohub Cell Tracking

This repository contains our work for the Kaggle competition `Biohub - Cell Tracking During Development`.
The goal is to detect zebrafish cell centers in 3D microscopy movies, link the same cells through time, detect divisions, and write a valid `submission.csv`.

## Current Status

A reproducible classical baseline is in place: an anisotropic 3D blob / local-maxima detector, distance-gated frame-to-frame linking, a local approximation of the competition metric, an embryo-disjoint validation split, and a submission writer with tests.
Current work focuses on stronger, adaptive detection and min-cost-flow linking over the whole movie.

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
