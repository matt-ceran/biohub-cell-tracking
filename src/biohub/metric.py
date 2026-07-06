"""Local approximation of the competition score.

This module scores a predicted tracking graph against sparse GEFF ground truth.
It mirrors the official recipe closely enough to rank experiments locally, but it
is deliberately a *local approximation*: the hidden Kaggle metric may differ in the
exact division-timing tolerance and node over-prediction penalty. Use these scores
to compare pipeline versions, not to predict the leaderboard value.

The scoring pipeline has three stages, matching the project brief:

1. Match predicted nodes to ground-truth nodes per timepoint by optimal bipartite
   assignment on physical (micrometer) centroid distance, gated at ``MATCH_RADIUS_UM``.
2. Score edges with an *adjusted* Jaccard: a predicted edge whose endpoints do not
   both match ground-truth nodes is ignored rather than penalised, because labels
   are sparse. Only edges between two matched-and-truly-connected nodes score.
3. Score divisions with a separate Jaccard, then combine with a small weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub.constants import DIVISION_WEIGHT, MATCH_RADIUS_UM, VOXEL_SCALE_UM
from biohub.io import GeffGraph


@dataclass(frozen=True)
class TrackingGraph:
    """A tracking graph in the form the metric needs: nodes plus directed edges."""

    node_ids: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray
    edges: np.ndarray  # shape (M, 2): source node id, target node id

    @classmethod
    def from_geff(cls, graph: GeffGraph) -> TrackingGraph:
        """Adapt a parsed GEFF graph into a TrackingGraph."""
        return cls(
            node_ids=np.asarray(graph.node_ids),
            t=np.asarray(graph.t),
            z=np.asarray(graph.z),
            y=np.asarray(graph.y),
            x=np.asarray(graph.x),
            edges=np.asarray(graph.edges).reshape(-1, 2),
        )

    @classmethod
    def from_submission(cls, df, dataset: str) -> TrackingGraph:
        """Extract one dataset's nodes and edges from a submission dataframe."""
        group = df[df["dataset"] == dataset]
        nodes = group[group["row_type"] == "node"]
        edges = group[group["row_type"] == "edge"]
        edge_array = (
            edges[["source_id", "target_id"]].to_numpy(dtype=np.int64)
            if len(edges)
            else np.empty((0, 2), dtype=np.int64)
        )
        return cls(
            node_ids=nodes["node_id"].to_numpy(dtype=np.int64),
            t=nodes["t"].to_numpy(),
            z=nodes["z"].to_numpy(dtype=float),
            y=nodes["y"].to_numpy(dtype=float),
            x=nodes["x"].to_numpy(dtype=float),
            edges=edge_array,
        )


@dataclass(frozen=True)
class MetricResult:
    """The full scored breakdown for one dataset."""

    score: float
    edge_jaccard: float
    division_jaccard: float
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edge_ignored: int
    division_tp: int
    division_fp: int
    division_fn: int
    matched_nodes: int
    predicted_nodes: int
    truth_nodes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "edge_jaccard": self.edge_jaccard,
            "division_jaccard": self.division_jaccard,
            "edge_tp": self.edge_tp,
            "edge_fp": self.edge_fp,
            "edge_fn": self.edge_fn,
            "edge_ignored": self.edge_ignored,
            "division_tp": self.division_tp,
            "division_fp": self.division_fp,
            "division_fn": self.division_fn,
            "matched_nodes": self.matched_nodes,
            "predicted_nodes": self.predicted_nodes,
            "truth_nodes": self.truth_nodes,
        }


def _scaled_coords(graph: TrackingGraph, scale: dict[str, float]) -> np.ndarray:
    """Stack node coordinates scaled into micrometers."""
    return np.column_stack(
        [
            np.asarray(graph.z, dtype=float) * scale["z"],
            np.asarray(graph.y, dtype=float) * scale["y"],
            np.asarray(graph.x, dtype=float) * scale["x"],
        ]
    )


def match_nodes(
    pred: TrackingGraph,
    truth: TrackingGraph,
    scale: dict[str, float] = VOXEL_SCALE_UM,
    max_distance: float = MATCH_RADIUS_UM,
) -> dict[int, int]:
    """Match predicted to truth node ids per timepoint by gated bipartite assignment.

    Returns a mapping ``predicted_node_id -> truth_node_id`` for accepted matches only.
    A pairing is accepted only if the two centroids are within ``max_distance`` micrometers.
    """
    pred_coords = _scaled_coords(pred, scale)
    truth_coords = _scaled_coords(truth, scale)
    pred_t = np.asarray(pred.t)
    truth_t = np.asarray(truth.t)
    pred_ids = np.asarray(pred.node_ids)
    truth_ids = np.asarray(truth.node_ids)

    matches: dict[int, int] = {}
    for timepoint in np.unique(truth_t):
        p_idx = np.flatnonzero(pred_t == timepoint)
        g_idx = np.flatnonzero(truth_t == timepoint)
        if p_idx.size == 0 or g_idx.size == 0:
            continue

        diff = pred_coords[p_idx][:, None, :] - truth_coords[g_idx][None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))

        # Forbid pairings beyond the radius by inflating their cost, then filter.
        forbid = max_distance * 10.0 + 1.0
        cost = np.where(dist <= max_distance, dist, forbid)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            if dist[r, c] <= max_distance:
                matches[int(pred_ids[p_idx[r]])] = int(truth_ids[g_idx[c]])
    return matches


def _directed_edges(graph: TrackingGraph) -> set[tuple[int, int]]:
    """Return edges oriented from the earlier timepoint to the later one."""
    if graph.edges.size == 0:
        return set()
    t_by_id = {int(nid): float(tv) for nid, tv in zip(graph.node_ids, graph.t, strict=True)}
    oriented: set[tuple[int, int]] = set()
    for source, target in graph.edges:
        source, target = int(source), int(target)
        ts = t_by_id.get(source, 0.0)
        tt = t_by_id.get(target, 0.0)
        if tt < ts:
            source, target = target, source
        oriented.add((source, target))
    return oriented


def _out_degree(edges: set[tuple[int, int]]) -> dict[int, int]:
    """Count outgoing (forward-in-time) edges per source node."""
    degree: dict[int, int] = {}
    for source, _ in edges:
        degree[source] = degree.get(source, 0) + 1
    return degree


def _jaccard(tp: int, fp: int, fn: int) -> float:
    """Jaccard overlap; an empty-vs-empty comparison scores a perfect 1.0."""
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def evaluate(
    pred: TrackingGraph,
    truth: TrackingGraph,
    division_weight: float = DIVISION_WEIGHT,
    scale: dict[str, float] = VOXEL_SCALE_UM,
    max_distance: float = MATCH_RADIUS_UM,
) -> MetricResult:
    """Score one predicted graph against one ground-truth graph."""
    matches = match_nodes(pred, truth, scale=scale, max_distance=max_distance)

    pred_edges = _directed_edges(pred)
    truth_edges = _directed_edges(truth)

    # Edge scoring with the sparse-label adjustment.
    edge_tp = edge_fp = edge_ignored = 0
    for source, target in pred_edges:
        if source in matches and target in matches:
            mapped = (matches[source], matches[target])
            if mapped in truth_edges:
                edge_tp += 1
            else:
                edge_fp += 1
        else:
            edge_ignored += 1
    edge_fn = len(truth_edges) - edge_tp
    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)

    # Division scoring: a division node has two or more forward edges.
    truth_div = {node for node, deg in _out_degree(truth_edges).items() if deg >= 2}
    pred_div = {node for node, deg in _out_degree(pred_edges).items() if deg >= 2}
    matched_truth_div: set[int] = set()
    div_fp = 0
    for node in pred_div:
        mapped = matches.get(node)
        if mapped is None:
            continue  # sparse-label adjustment: unmatched division is ignored
        if mapped in truth_div:
            matched_truth_div.add(mapped)
        else:
            div_fp += 1
    div_tp = len(matched_truth_div)
    div_fn = len(truth_div) - div_tp
    division_jaccard = _jaccard(div_tp, div_fp, div_fn)

    score = (1.0 - division_weight) * edge_jaccard + division_weight * division_jaccard
    return MetricResult(
        score=score,
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        edge_ignored=edge_ignored,
        division_tp=div_tp,
        division_fp=div_fp,
        division_fn=div_fn,
        matched_nodes=len(matches),
        predicted_nodes=len(pred.node_ids),
        truth_nodes=len(truth.node_ids),
    )


def evaluate_datasets(results: dict[str, MetricResult]) -> dict[str, float]:
    """Average per-dataset scores into a headline mean, as a validation loop would."""
    if not results:
        return {"mean_score": 0.0, "mean_edge_jaccard": 0.0, "mean_division_jaccard": 0.0}
    n = len(results)
    return {
        "mean_score": sum(r.score for r in results.values()) / n,
        "mean_edge_jaccard": sum(r.edge_jaccard for r in results.values()) / n,
        "mean_division_jaccard": sum(r.division_jaccard for r in results.values()) / n,
    }
