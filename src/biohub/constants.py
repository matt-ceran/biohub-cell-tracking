"""Shared competition constants."""

SUBMISSION_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)

VOXEL_SCALE_UM = {
    "z": 1.625,
    "y": 0.40625,
    "x": 0.40625,
}

MATCH_RADIUS_UM = 7.0

# The official score weights divisions far below edges; this is a local approximation.
DIVISION_WEIGHT = 0.1

# Cells move a median of ~1.7 um/frame (p99 ~7.2 um) in the training tracks, so an
# 8 um gate captures nearly all true frame-to-frame motion for the baseline linker.
LINK_MAX_UM = 8.0

# Division (mitosis) geometry, mined from the 199 training graphs (151 divisions).
# A mother sits a median 5.8 um (p90 9.1, max 13.5) from each daughter -- farther than a
# normal 1.8 um step, and often BEYOND the 8 um link gate, which is exactly why the second
# daughter is left orphaned by the linker. The two daughters push APART: the angle between
# their displacement vectors from the mother is a median 138 deg, with 89% over 90 deg and
# ~70% over 120 deg. The fork classifier gates on all three signals -- both children at a
# division-scale distance (not a short normal drift), well within reach, and strongly
# opposite. Even so these gates are only conservative enough for an off-by-default lever:
# over an over-predicted linkage the true forks are not separable from coincidental ones
# (see the Phase-7 experiment note), so add_divisions is not in the shipped pipeline.
DIVISION_MAX_UM = 10.0          # mother->daughter gate (covers ~p90 of true divisions)
DIVISION_MIN_CHILD_UM = 4.0     # BOTH children must jump this far -- rejects normal drift
DIVISION_MIN_ANGLE_DEG = 120.0  # daughters must move to strongly opposite sides
