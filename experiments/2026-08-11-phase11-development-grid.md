# Phase 11 Development Grid

Date: 2026-08-11.

Status: The complete 660-row development grid is finished and strictly validated, but no policy has been frozen.

## Conclusion

None of the 33 frozen model-threshold policies passed every development guard.
No head recovered one exact mother-plus-two-daughters triplet from the 18 labelled development divisions.
Every accepted fork was therefore a strict false fork.
The embedding heads remained severely overconfident on unseen proposals, while the conservative scalar head preserved the fallback score without adding a true recovery.
Evaluation remains sealed and the accepted Phase 10 divisions-off pipeline remains the safe fallback.

## Reproduction Command

The grid ran from repository checkpoint `6457b26e2d15bb3895ab72dec071541f11e802e4`.
It used the exact default CPU execution contract and stayed awake under `caffeinate`.

```bash
/usr/bin/caffeinate -i \
  .venv/bin/python -u scripts/validate_division_model.py \
  --cohort development
```

The observed wall time was approximately 2 hours 15 minutes.
Movie runtimes varied substantially because proposal volume and graph mutation cost vary by candidate count.
The command wrote one atomic 33-row checkpoint after each complete movie and exited with code 0.

The strict replay command returned the complete grid immediately without recomputing a movie.

```bash
.venv/bin/python -u scripts/validate_division_model.py \
  --cohort development
```

## Frozen Grid Contract

The experiment contains exactly 20 development movies, three trained heads, and eleven thresholds.

```text
20 movies x 3 heads x 11 thresholds = 660 rows
```

The heads are `scalar`, `embedding_linear`, and `embedding_mlp32`.
The thresholds are `0.5`, `0.7`, `0.8`, `0.9`, `0.95`, `0.975`, `0.99`, `0.995`, `0.999`, `0.9995`, and `0.9999`.
The accepted Phase 10 divisions-off graph is the exact per-movie fallback.
Development changed no model weights, fit-only normalization, proposal rule, assignment rule, threshold, or acceptance guard.

## Main Result

Zero of the 33 policies passed all nine frozen guards.
All 33 policies produced zero strict true recoveries.
The baseline mean full score across the 20 development movies is `0.5265460965725556`.

The most conservative threshold gives the clearest head-level comparison.

| Head | Accepted forks | Strict true recoveries | Strict false forks | Mean full-score delta | Wins / ties / losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scalar` | 7 | 0 | 7 | `+0.000000000` | 0 / 20 / 0 |
| `embedding_linear` | 63 | 0 | 63 | `-0.004931818` | 0 / 19 / 1 |
| `embedding_mlp32` | 98 | 0 | 98 | `-0.010096154` | 0 / 17 / 3 |

At threshold `0.5`, every head reached the frozen safety cap of five accepted forks per movie, or 100 forks across the cohort.
Scalar remained at 100 accepted forks through threshold `0.995`, then declined to 72, 36, and 7 at thresholds `0.999`, `0.9995`, and `0.9999`.
Embedding-linear retained 63 false forks at `0.9999`.
Embedding-MLP32 retained 98 false forks at `0.9999` despite its epoch-100 fit loss of `0.000540420`.

## Proposal Ceiling And Generalization Failure

The frozen development preflight had found 10 of 18 labelled divisions representable under the proposal rule.
The remaining eight events were unreachable before classification because a required predicted cell was missing or a daughter fell outside the nearest-12 cap.
The grid now shows that none of the ten representable truth triplets survived model scoring and assignment as an accepted exact fork.
This is a classifier-ranking generalization failure downstream of the already-known proposal ceiling.

The MLP's near-zero fit loss therefore measured capacity to reproduce the fit lessons, not reusable division recognition.
The high-confidence false forks also show that the embedding-head probabilities are poorly calibrated on unseen proposals.

## Frozen Guard Outcome

No eligible development policy exists.
The pure read-only selector preview returns `scalar` at threshold `0.9999` as the least harmful rejection candidate.
That preview passes six guards and fails three.

The passing guards are complete cohort coverage, node-density tolerance, edge-Jaccard tolerance, no fully matched wrong fork, no false division on a no-labelled-division control movie, and exact no-fork fallback identity.
The failed guards are full-score wins greater than losses, mean full-score delta of at least `+0.001`, and at least one strict true recovery.

The preview has these exact metrics:

| Metric | Value |
| --- | ---: |
| Accepted forks | 7 |
| Strict true recoveries | 0 |
| Fully matched wrong forks | 0 |
| Full-score wins / ties / losses | 0 / 20 / 0 |
| Mean full-score delta | `0.0` |
| Mean edge-Jaccard delta | `0.0` |
| Aggregate node-ratio delta | `0.000039865438811181875` |

No development-result or division-policy artifact was written.
Running the policy freezer later would record a rejection with `evaluation_unlocked=false`, not accept the seven preview forks for evaluation or deployment.

## Artifact Identities

The canonical generated grid remains ignored under `data/working/`.

| Artifact | SHA-256 |
| --- | --- |
| Grid rows | `5d9312ac50e39e7da5f3a81aefd350fe581d22b74743fdd00e8d4725b1ef9144` |
| Grid file | `a3ff21f16cd32fea6702064372ed9c633e0fcc3e3f67316eb2e6294193b46447` |
| Validation source | `fa37ba3be1f25469509bd84606335b7bdf4e9fb5a9eea745249b06f862680596` |
| Assignment contract | `22bbd5cef2864769ac2b17f7fe9f4275a7ea53d8d4715103cfffce03b4459a35` |

The beginner-first report is `output/pdf/biohub_phase11_development_grid_findings.pdf`.
It contains 10 letter-size pages and passed metadata inspection, required-fact text extraction, complete Poppler rendering, and page-by-page visual inspection.
Its SHA-256 is `7cc303f2785045971f0aab3a7e3f8953c54150f885ff64b8cbe1ca7e5decbc8a`.

## Verification

The completed grid replayed with the same row and file hashes.
Every one of the 33 policies contained all 20 movies and recomputed against the frozen guard contract.
The cache cohort remained exactly 88 fit, 20 development, and zero evaluation files.
No development-result file or division-policy file existed after the audit.

`.venv/bin/python -m pytest -q` passed all 400 tests in 32.82 seconds.
`.venv/bin/python -m ruff check src scripts tests` passed.
`.venv/bin/python -m compileall -q src scripts tests` passed.
`git diff --check` passed.

## Next Boundary

Stop before policy freezing until the development findings have been reviewed.
When explicitly approved, run `scripts/freeze_division_policy.py` to record the frozen rejection and keep evaluation locked.
Then complete the Phase 11 negative-result audit and final report while retaining the accepted Phase 10 divisions-off pipeline.
Do not build or inspect evaluation caches, submit to Kaggle, or release generated model artifacts from this milestone.
