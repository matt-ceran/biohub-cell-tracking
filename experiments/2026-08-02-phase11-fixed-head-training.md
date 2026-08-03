# Phase 11 Fixed-Head Training

Date: 2026-08-02.

Status: Fixed-head fit training is complete, but development model selection has not started.

## Conclusion

The fixed Phase 11 training procedure successfully produced all three planned division heads.
This is an engineering and fit-training milestone, not proof that any head improves cell tracking on unseen movies.
The `embedding_mlp32` head reached the lowest fit loss, but its result may reflect memorization because training repeatedly presented only 76 unique positive division examples.
No head, threshold, or division policy has been selected.
The next experiment is the complete 660-row development grid, and evaluation remains sealed until a development policy passes every frozen guard.

## Reproduction Command

Training used the Phase 11 implementation at source commit `e1447c9ba9700f2dfc731e5d539b3d843dd1f070`.

```bash
/usr/bin/caffeinate -i \
  .venv/bin/python -u scripts/train_division.py --all
```

The strict replay command returned all three completed epoch-100 checkpoints without running another epoch.

```bash
/usr/bin/caffeinate -i \
  .venv/bin/python -u scripts/train_division.py --all --resume
```

## Data Boundary

Training used only the frozen 88-movie `6bba` fit cohort.
The sampled fit dataset contains 6,620 records, of which 3,804 contribute to the loss.
Those supervised records consist of 76 positive division triplets, 912 same-mother negatives, and 2,816 matched non-division controls.
The remaining 2,816 diagnostic unknowns have zero weight and never enter the loss or gradients.
The frozen proposal rule represents 76 of 89 known fit divisions and 10 of 18 known development divisions.
Events missing from the proposed triplets cannot be learned or recovered by these classifier heads.

The canonical dataset identities are:

| Identity | SHA-256 |
| --- | --- |
| Dataset manifest | `a253ba29a38fb0dc76c79d1a19af71b0047ab0140c7dce923af43bdeb7f391c5` |
| Realized sample tuple | `9f86009b60a050fc12fe286d74677bf2fc03d4794af002442c93e3c29bd76aa1` |
| Model inputs | `7778341922920a8cd6fd1f43cb14a3f34c7ece13dcb8efcc9bfe555f97d3bd0f` |
| Feature schema | `2626e5177f33411635148ac1e11529b14d6f1f9e044af3544626f74f53caa17e` |
| Frozen Phase 10 encoder | `bd35d082c985641728bc340817d08d4750a3c83c34d25f99d1507cbc8b7bb8d4` |
| Trainer source | `8f100e19fdb2a9417559ed03e5106c6d334e07a4bf5364b270aa0e05e72b66ec` |

## Training Contract

Each head trained for exactly 100 epochs on MPS with seed 0, AdamW, learning rate 0.001, weight decay 0.001, batch size 64, gradient clipping at 1.0, and binary cross-entropy with logits.
The accepted Phase 10 3D appearance encoder remained frozen and byte-identical.
Only the separate division head was trainable.

Each epoch repeated the smaller supervised strata until it presented 2,816 records from each of the three supervised groups.
That produced 8,448 presentations in 132 batches per epoch.
Across all three heads and 100 epochs, training processed 2,534,400 record presentations.
This balancing changes teaching attention but does not create new biological examples.

## Fit Results

| Head | Parameters | Final fit loss | Minimum fit loss | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `scalar` | 59 | 0.299762996 | 0.298932299 at epoch 99 | 81.292 s |
| `embedding_linear` | 315 | 0.179190679 | 0.179034624 at epoch 99 | 83.341 s |
| `embedding_mlp32` | 10,741 | 0.000540420 | 0.000249927 at epoch 98 | 177.018 s |

The recorded runtimes total 341.652 seconds, or 5.694 minutes.
Fit loss measures how closely each head learned the fit records and is not a development score or final tracking score.

## Artifact Identities

| Artifact | Local generated path | SHA-256 |
| --- | --- | --- |
| Frozen encoder | `data/working/appearance_checkpoints/cellness_cnn_phase10_full.pt` | `bd35d082c985641728bc340817d08d4750a3c83c34d25f99d1507cbc8b7bb8d4` |
| Scalar head | `data/working/phase11_division_checkpoints/scalar.pt` | `cc05f80f48ccbb3a28df3d3c3b43d24d9a8b878c61290ec298e5e4c7b464469d` |
| Embedding-linear head | `data/working/phase11_division_checkpoints/embedding_linear.pt` | `78b90f27cde1cc8dd4b32dc488e48c013c6bb5d3f8b83dc4c18977ec794aab04` |
| Embedding-MLP32 head | `data/working/phase11_division_checkpoints/embedding_mlp32.pt` | `d760c03809e9fd0a234e28c26667d91d3908cc78695b15e92ee590358bf6158a` |

The checkpoints, histories, and optimizer resume states remain under ignored `data/working/` paths as generated research artifacts.
Their exact paths and hashes are recorded here so the completed training run remains durable and auditable without adding generated model files to Git.
No model checkpoint has been released or submitted to Kaggle.

## Verification

All three checkpoints, histories, and resume states passed strict identity, epoch, finite-tensor, head-only, balance, and replay validation.
The accepted Phase 10 encoder remained byte-identical after training.
The repository passed 400 tests, Ruff, Python bytecode compilation, and whitespace validation immediately after the run.
The four final checkpoint files were independently hash-checked and reloaded through the production model loaders.

## Next Boundary

Do not infer a winning head from these fit losses.
The next approved technical command is shown below, but it must not run until the user explicitly chooses to begin development validation.

```bash
.venv/bin/python -u scripts/validate_division_model.py --cohort development
```

That command must produce exactly 660 complete rows from 20 development movies, three frozen heads, and eleven thresholds.
No evaluation cache may be built or inspected unless a later frozen development policy explicitly unlocks evaluation.
