"""Tests for the local scoring approximation.

The main test rebuilds the worked example used to explain the metric: two correct
edges, one wrong link between labeled cells, and two edges on unlabeled cells that
must be ignored rather than penalised.
"""

from __future__ import annotations

import numpy as np

from biohub.metric import TrackingGraph, evaluate, evaluate_datasets, match_nodes

# A unit scale keeps distances equal to raw coordinate differences in these tests.
UNIT_SCALE = {"z": 1.0, "y": 1.0, "x": 1.0}


def _graph(nodes, edges):
    """Build a TrackingGraph from (node_id, t, z, y, x) tuples and (source, target) edges."""
    nodes = np.array(nodes, dtype=float)
    return TrackingGraph(
        node_ids=nodes[:, 0].astype(np.int64),
        t=nodes[:, 1],
        z=nodes[:, 2],
        y=nodes[:, 3],
        x=nodes[:, 4],
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_perfect_prediction_scores_one():
    truth = _graph(
        [(1, 0, 0, 0, 0), (2, 1, 0, 0, 1), (3, 0, 0, 5, 0), (4, 1, 0, 5, 1)],
        [(1, 2), (3, 4)],
    )
    result = evaluate(truth, truth, scale=UNIT_SCALE, max_distance=2.0)
    assert result.edge_jaccard == 1.0
    assert result.score == 1.0
    assert result.edge_fp == 0 and result.edge_fn == 0


def test_worked_three_bucket_example():
    # Ground truth: three tracks A, B, C across frames 0 and 1.
    truth = _graph(
        [
            (10, 0, 0, 0, 0),  # A0
            (11, 1, 0, 0, 0),  # A1
            (20, 0, 0, 10, 0),  # B0
            (21, 1, 0, 10, 0),  # B1
            (30, 0, 0, 20, 0),  # C0
            (31, 1, 0, 20, 0),  # C1
        ],
        [(10, 11), (20, 21), (30, 31)],
    )
    # Prediction:
    #  - correct A and B links (true positives)
    #  - a wrong link C0 -> B1 between labeled cells (false positive)
    #  - two edges on unlabeled cells far from any truth node (ignored)
    pred = _graph(
        [
            (100, 0, 0, 0, 0),  # matches A0
            (101, 1, 0, 0, 0),  # matches A1
            (200, 0, 0, 10, 0),  # matches B0
            (201, 1, 0, 10, 0),  # matches B1
            (300, 0, 0, 20, 0),  # matches C0
            (400, 0, 0, 90, 0),  # unlabeled region
            (401, 1, 0, 90, 0),  # unlabeled region
            (500, 0, 0, 95, 0),  # unlabeled region
            (501, 1, 0, 95, 0),  # unlabeled region
        ],
        [(100, 101), (200, 201), (300, 201), (400, 401), (500, 501)],
    )
    result = evaluate(pred, truth, scale=UNIT_SCALE, max_distance=2.0)

    assert result.edge_tp == 2
    assert result.edge_fp == 1  # C0 -> B1 links two labeled cells that are not connected
    assert result.edge_fn == 1  # C0 -> C1 was never predicted
    assert result.edge_ignored == 2  # the two unlabeled-cell edges
    assert result.edge_jaccard == 2 / (2 + 1 + 1)  # == 0.5


def test_matching_respects_the_distance_gate():
    truth = _graph([(1, 0, 0, 0, 0)], [])
    # A predicted node just outside the radius must not match.
    pred = _graph([(9, 0, 0, 0, 3)], [])
    assert match_nodes(pred, truth, scale=UNIT_SCALE, max_distance=2.0) == {}
    # Inside the radius it matches.
    pred_close = _graph([(9, 0, 0, 0, 1)], [])
    assert match_nodes(pred_close, truth, scale=UNIT_SCALE, max_distance=2.0) == {9: 1}


def test_division_is_scored_separately():
    # Truth: node 1 divides into 2 and 3 at the next frame.
    truth = _graph(
        [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0), (3, 1, 0, 1, 0)],
        [(1, 2), (1, 3)],
    )
    # Prediction reproduces the division exactly.
    pred = _graph(
        [(10, 0, 0, 0, 0), (20, 1, 0, 0, 0), (30, 1, 0, 1, 0)],
        [(10, 20), (10, 30)],
    )
    result = evaluate(pred, truth, scale=UNIT_SCALE, max_distance=2.0)
    assert result.division_tp == 1
    assert result.division_fp == 0 and result.division_fn == 0
    assert result.division_jaccard == 1.0


def test_evaluate_datasets_averages_scores():
    truth = _graph([(1, 0, 0, 0, 0), (2, 1, 0, 0, 0)], [(1, 2)])
    good = evaluate(truth, truth, scale=UNIT_SCALE, max_distance=2.0)
    empty_pred = _graph([(9, 0, 0, 9, 9)], [])
    bad = evaluate(empty_pred, truth, scale=UNIT_SCALE, max_distance=2.0)
    summary = evaluate_datasets({"a": good, "b": bad})
    assert 0.0 <= summary["mean_score"] <= 1.0
    assert summary["mean_edge_jaccard"] == (good.edge_jaccard + bad.edge_jaccard) / 2
