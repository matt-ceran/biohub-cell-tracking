# Phase 11 Truth-Rank Atlas

Date: 2026-08-11.

Status: The complete development-only truth-rank diagnostic is finished and strictly validated, but no division policy has been frozen.

## Conclusion

The atlas traced every one of the 18 labelled development divisions through the frozen proposal, scoring, threshold, and assignment path.
Ten exact truth proposals were available to the heads and eight were impossible before scoring.
No reachable truth became an accepted fork at any of the eleven tested thresholds.
The failure is not one-dimensional: proposal recall, within-mother ranking, score calibration, and movie-wide false-mother crowding all require separate repairs.
Evaluation remained sealed and the accepted Phase 10 divisions-off graph remained unchanged.

## Reproduction Command

The complete audit used the exact default CPU contract.

```bash
.venv/bin/python -u scripts/audit_division_truth_ranks.py \
  --cohort development
```

The observed wall time was approximately five minutes.
The command exited with code 0 only after validating the completed grid, all three epoch-100 checkpoints, the frozen contracts, the 18-event funnel, every assignment trace, and every contact-sheet hash.

The canonical ignored outputs are:

```text
data/working/phase11_division_truth_rank_atlas/atlas.json
data/working/phase11_division_truth_rank_atlas/manifest.json
data/working/phase11_division_truth_rank_atlas/images/*.png
```

## Exact Scope And Boundary

The audit read the complete 660-row development grid and replayed only the 10 development movies containing labelled divisions.
It opened development labels and images only for those declared movies.
It did not build, open, or inspect an evaluation cache.
It did not retrain a head, change a threshold, alter the proposal or assignment rule, mutate a saved fallback, write a development-result file, or freeze a policy.

The working cache directory remained exactly 108 files: 88 fit, 20 development, and zero evaluation.
The Phase 11 development-result and policy paths remained absent.

## Method

For each labelled truth division, the audit matched the mother into the frozen linked mother source and matched both daughters into the raw-scored candidate source.
It then rebuilt the exact 16 micrometer, nearest-12 pair neighborhood used by inference.

For each reachable exact truth pair and each of the three heads, it recorded:

- The true proposal logit, probability, and exact rank among that mother's proposals.
- The strongest false proposal and its score gap over the truth.
- The stopping stage at all eleven frozen thresholds.
- The accepted-fork records already stored in the completed development grid.
- A contact sheet containing the predicted mother, both truth-matched daughters, and each head's strongest false daughter pair.

The exact per-mother order remained higher logit, fewer edge removals, fewer restored nodes, and then smaller candidate identifiers.
The movie assignment trace remained thresholding, one best pair per mother, daughter-conflict exclusion, and a maximum of five accepted forks.

## Proposal Reachability

The frozen proposal funnel reproduced exactly.

| Outcome | Events | Meaning |
| --- | ---: | --- |
| Exact truth proposal reachable | 10 | The three heads could score the correct mother-plus-two-daughters choice. |
| True daughter missing from raw candidates | 4 | No fork head can recover a daughter that the cell proposal stage did not produce. |
| Mother missing from the linked source | 2 | The declared mother population removed the event before pair scoring. |
| True daughter beyond nearest-12 cap | 2 | The matched daughter ranked 14th or 18th by distance and the exact pair was never emitted. |

The two candidate-cap misses were truth mother `53000762` in `6bba_5c039895`, with daughter ranks 18 and 3, and truth mother `67000304` in `6bba_c328f2fd`, with daughter ranks 14 and 1.

## Exact Truth Ranks

| Head | Rank 1 | Top 3 | Top 5 | Top 10 | Median rank | Worst rank | Strongest false beat truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scalar` | 1 | 5 | 7 | 8 | 4.0 | 53 | 9 of 10 |
| `embedding_linear` | 0 | 3 | 4 | 7 | 7.5 | 34 | 10 of 10 |
| `embedding_mlp32` | 3 | 6 | 6 | 7 | 2.5 | 44 | 7 of 10 |

The MLP produced the strongest local ranking result, with three rank-1 truths and six top-3 truths.
Scalar produced one rank-1 truth and a median rank of 4.
Embedding-linear never placed an exact truth first.

Three unique truth events produced four head-event rank-1 results because scalar and MLP both ranked one event first.
Every one of those locally correct results was blocked by five higher-scoring false mothers before it could enter the movie assignment.

## Event-Level Rank Atlas

| Movie / truth mother | Scalar | Embedding linear | Embedding MLP-32 | Proposals |
| --- | ---: | ---: | ---: | ---: |
| `062c8d37 / 90001276` | 3 | 2 | 1 | 66 |
| `09961292 / 45001493` | 5 | 9 | 1 | 66 |
| `09961292 / 49001609` | 1 | 2 | 1 | 66 |
| `55b7eebe / 49000351` | 2 | 2 | 3 | 15 |
| `5c039895 / 11000170` | 9 | 9 | 7 | 66 |
| `5c039895 / 59000855` | 5 | 17 | 38 | 66 |
| `705ec2c9 / 43000460` | 31 | 22 | 16 | 66 |
| `b329af44 / 83001755` | 3 | 4 | 2 | 66 |
| `c328f2fd / 29000074` | 53 | 34 | 44 | 66 |
| `fe670320 / 67000712` | 2 | 6 | 2 | 66 |

## Stopping Stages

At threshold `0.5`, scalar lost seven truths to a wrong same-mother pair, two below threshold, and one at the five-fork cap.
Embedding-linear lost seven to a wrong same-mother pair and three below threshold.
Embedding-MLP32 lost one to a wrong same-mother pair, six below threshold, and three at the five-fork cap.

At threshold `0.9999`, all 10 scalar and all 10 embedding-linear truths were below threshold.
Embedding-MLP32 left seven truths below threshold, one behind a wrong same-mother pair, and two behind the five-fork cap.

No reachable truth was stopped by the daughter-conflict branch.
No reachable truth was ever accepted.

The completed grid still accepted 7 scalar, 63 embedding-linear, and 98 embedding-MLP32 false forks at threshold `0.9999`.
The MLP therefore combined the best local top-1 count with the worst movie-wide calibration.

## Modeling Consequence

More threshold sweeps cannot solve this result.
The next build needs three coordinated workstreams.

1. Restore proposal recall for the four missing-daughter events, two missing-mother events, and two nearest-cap misses while retaining bounded volume and control checks.
2. Improve within-mother ordering using event-balanced positives, atlas-style hard negatives, movie-grouped fit cross-validation, and a ranking-aware objective.
3. Add a fit-calibrated mother-level division gate or equivalent two-stage score so high-confidence false mothers do not occupy all five movie slots.

The immediate success signal is not a larger number of accepted forks.
It is more exact rank-1 truths together with fewer high-confidence false mothers and unchanged no-division controls.

## Artifact Identities

| Artifact | SHA-256 |
| --- | --- |
| Truth-rank atlas JSON | `6d177858354dc4f94d88571d5ec320819fe74883cf1b60355a755f33bc0f5d0a` |
| Truth-rank manifest | `4172fd13a857db4b6f6a35255363d3ad24b5d018c968b23e1531d03beef2af85` |
| Truth-rank audit source | `ef9de26c44da86b79611103453dc0cd3af225500a0666011bea3b8d6b9818fa4` |
| Development grid file | `a3ff21f16cd32fea6702064372ed9c633e0fcc3e3f67316eb2e6294193b46447` |
| Development grid rows | `5d9312ac50e39e7da5f3a81aefd350fe581d22b74743fdd00e8d4725b1ef9144` |
| Validation source | `fa37ba3be1f25469509bd84606335b7bdf4e9fb5a9eea745249b06f862680596` |
| Assignment contract | `22bbd5cef2864769ac2b17f7fe9f4275a7ea53d8d4715103cfffce03b4459a35` |

The manifest records and binds all 10 contact-sheet paths and SHA-256 values.
Every sheet is a `2790 x 1475` PNG with three head rows and five image columns.

## PDF Report

The beginner-first report is `output/pdf/biohub_phase11_truth_rank_atlas.pdf`.
It contains 18 letter-size pages, including 10 event contact-sheet pages and an appendix for all eight unreachable truths.
Its SHA-256 is `433c02b5b1a65b30aba2090cbb1f6551f6d9ab4034304660e086799e813a6cfe`.

The PDF passed metadata inspection, required-fact text extraction, full Poppler rendering, and page-by-page visual inspection.
The ignored generator is `scripts/create_phase11_truth_rank_atlas_pdf.py`.

## Verification

The focused truth-rank suite passed all 8 tests.
The complete repository suite passed all 408 tests in 40.42 seconds.
Ruff passed across `src`, `scripts`, and `tests`.
Python compilation passed across `src`, `scripts`, and `tests`.
The atlas manifest and every contact-sheet hash were independently recomputed and matched.

## Next Boundary

This milestone completes the truth-rank diagnostic slice only.
Do not freeze a policy, build evaluation caches, inspect evaluation labels, submit to Kaggle, or promote a division head without a separate explicit decision.
The next modeling iteration should start from the proposal-recall, local-ranking, and global-calibration workstreams above while retaining the accepted divisions-off fallback.
