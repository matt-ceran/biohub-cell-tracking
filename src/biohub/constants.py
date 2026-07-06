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
