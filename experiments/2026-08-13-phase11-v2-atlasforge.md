# Phase 11 v2 ATLASFORGE

Date: 2026-08-13.

Status: The separately versioned fit comparison and its one post-selection development diagnostic are complete.
The bounded proposal rule improved independent reachability, but the learned ranking and calibration failed development.
The v2 model is rejected for promotion.
No policy was frozen, no evaluation cache or label was opened, and `evaluation_unlocked` remains `false`.

## Conclusion

The nearest-12 v1 rule hid recoverable daughter pairs, so v2 preserved its complete base and added a strictly bounded outer ring.
This raised fit reachability from 76 to 81 of 89 and development reachability from 10 to 12 of 18.
Every v1-reachable event remained reachable.

Fit-only five-fold cross-validation selected a full-MLP16 ranker plus full-MLP16 mother gate.
It ranked 52 of 81 reachable out-of-fold truths first and recovered 29 under the two-stage rule while accepting zero of 2,816 heldout matched controls.
That result did not transfer to the frozen development cohort.
Only 2 of 12 reachable development truths ranked first, zero passed the two-stage rule, and 43 of 640 independently selected matched controls passed all five fold gates.

The proposal change generalized, but the representation, negative construction, ranking, and gate calibration did not.
The accepted Phase 10 divisions-off pipeline remains the safe fallback.

## Reproduction Commands

The fit-only proposal audit and model comparison used:

```bash
.venv/bin/python -u scripts/run_division_v2_fit.py
```

The one authorized development diagnostic used:

```bash
.venv/bin/python -u scripts/audit_division_v2_development.py
```

The strict artifact replays are:

```bash
.venv/bin/python scripts/run_division_v2_fit.py --verify-existing
.venv/bin/python scripts/audit_division_v2_development.py --verify-existing
.venv/bin/python scripts/audit_division_truth_ranks.py \
  --cohort development \
  --verify-existing
```

The fit transaction took approximately five minutes on MPS after the focused preflight work.
The exact 20-movie development diagnostic took approximately fifteen minutes on MPS.
Development runtime was dominated by exact min-cost-flow reconstruction for every movie.

## Data Boundary

All architecture, proposal, loss, fold, model, and gate choices used only the frozen 88-movie 6bba fit role.
The fit cohort contains 89 labelled division events.
The five cross-validation folds held out complete movies rather than individual proposals.
Each fold fitted its context normalization only on its own training movies.

The model ladder contained only `context_linear`, `full_linear`, and `full_mlp16`.
The development labels were opened once after fit selection became immutable.
Development could diagnose the fixed ensemble but could not change the selected model, thresholds, or weights.
Evaluation caches remained absent and evaluation labels were not opened.

## Bounded Proposal Rule

For each filtered pre-prune mother, v2 scans at most the nearest 24 raw-scored candidates within 18 micrometers.
It emits every unordered pair among the nearest 12 candidates, preserving the exact v1 pair set.
It then emits only pairs between the nearest 6 anchors and candidates ranked 13 through 24.

The exact per-mother upper bound is:

```text
C(12, 2) + 6 x 12 = 66 + 72 = 138 pairs
```

The rule never pairs two outer candidates and never pairs ranks 7 through 12 with an outer candidate.
That structure recovers the observed cap and radius misses without turning a 24-candidate neighborhood into all `C(24, 2) = 276` pairs.

## Fit Proposal Audit

| Measure | v1 | v2 | Result |
| --- | ---: | ---: | --- |
| Reachable truth events | 76 of 89 | 81 of 89 | Five additional fit truths became scoreable. |
| All fit proposals | 155,616,652 | 303,946,119 | Ratio `1.9531722029336551`. |
| No-division fit proposals | 63,273,218 | 123,565,721 | Ratio `1.9528913639258874`. |
| Maximum pairs per mother | 66 | 138 | The exact hard bound held. |

Both measured volume ratios passed the predeclared maximum of `2.0`.
The raw radius query saw as many as 159 candidates around one mother, but the scan cap prevented any additional emitted pairs beyond rank 24.

The eight remaining unreachable fit events are upstream missing-member failures.
Broadly lowering detector thresholds was rejected in preflight because it added hundreds or thousands of candidates per frame and often still missed the required cell.

## Fit-Only Ranking And Gate

The ranking loss was listwise softmax cross entropy with one equally weighted loss per truth mother.
This prevents mothers with more emitted pairs from dominating the objective.
The mother gate used balanced binary cross entropy on true pairs and ranker-selected matched-control pairs.
Each fold threshold was the next representable floating-point value above that fold's maximum heldout matched-control logit.

The selected full-MLP16 pair had 5,685 trainable parameters in each stage.
Its exact out-of-fold result was:

| Measure | Result |
| --- | ---: |
| Reachable fit truth groups | 81 |
| Rank 1 | 52 |
| Top 3 | 68 |
| Shared v1 truths ranked first | 51 of 76 |
| Two-stage recoveries | 29 |
| Heldout matched-control accepts | 0 of 2,816 |

The fold truth-event counts were 18, 17, 15, 17, and 14.
The folds contained 18, 18, 17, 17, and 18 complete movies.

## Independent Development Result

The frozen v2 proposal rule increased development reachability from 10 to 12 of 18.
The two new scoreable events were truth mother `53000762` in `6bba_5c039895` and truth mother `67000304` in `6bba_c328f2fd`.
The latter ranked first, while the former ranked seventh.

The fit-selected ranker and gate then failed the independent diagnostic:

| Measure | Result |
| --- | ---: |
| Reachable development truths | 12 of 18 |
| Rank 1 | 2 of 12 |
| Top 3 | 5 of 12 |
| Two-stage recoveries | 0 of 12 |
| Independent matched controls | 640 |
| All-fold gate accepts | 43 of 640 |

The model therefore fails both sides of the biological requirement.
It misses every exact truth under the complete two-stage rule and admits false division mothers on independent controls.
Threshold adjustment is not authorized and would not repair the large ranking collapse.

## Artifact Identities

The generated model artifacts remain ignored under `data/working/`, and the beginner report remains ignored under `output/pdf/`.

| Artifact | SHA-256 |
| --- | --- |
| Fit result JSON | `e2eb7c56ccd6038b2822505394af7ad6cf04ed1e5c19705f578afa5fb0b965dc` |
| Fit group logical content | `ed008b388f2b0adf49ac31c4e2104cb7cf0e17e319d2d53a5b1f2f028e2d251e` |
| Development result JSON | `6d0a23ef696b3316e5aeb4bc71055a4f01288d0491b862956a0bb7b1dcd5034e` |
| Replayed v1 truth-rank atlas | `6d177858354dc4f94d88571d5ec320819fe74883cf1b60355a755f33bc0f5d0a` |
| Beginner findings PDF | `a871234403aef552ebf1fae50032e43c76a56e8ae3dc4c9e821e4f6fe8489e91` |

The fit result binds five selected checkpoints, their whole-file hashes, each ranker and gate state hash, the fit-group logical hash, the v1 dataset identities, all 88 cache identities, and every fit event record.
The development result binds all 20 development cache identities, all 18 event records, all 640 control decisions, the selected fit-result hash, and the sealed boundary fields.
The 11-page report is `output/pdf/biohub_phase11_v2_atlasforge_findings.pdf`.
It was rendered at 144 DPI, inspected page by page, checked for required text and metadata, and opened in Preview.

## Promotion Decision

Do not promote the v2 model.
Do not freeze a division policy.
Do not build evaluation caches, inspect evaluation labels, submit to Kaggle, or alter the accepted Phase 10 fallback.

The bounded proposal rule is the only independently supported part of this milestone.
A future separately authorized iteration should first explain the ranking collapse and false-mother shift, then test materially different negative construction or representations with the same movie-grouped and evaluation-sealed discipline.
