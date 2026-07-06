"""Conservative frame-to-frame linking.

Given detected cell centers per timepoint, this links each cell to its most likely
continuation in the next labeled timepoint by gated optimal bipartite assignment on
physical distance. It is the "connect the dots" step: no motion model, just the
physical prior that a cell barely moves between frames, so the nearest plausible
detection is almost always the same cell.

The linker is greedy across frame pairs (each adjacent pair is solved independently),
which is the baseline. A later phase can replace this with min-cost flow over the
whole movie to handle gaps and divisions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub.constants import LINK_MAX_UM, VOXEL_SCALE_UM
from biohub.metric import TrackingGraph


def _scaled(graph: TrackingGraph, scale: dict[str, float]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(graph.z, dtype=float) * scale["z"],
            np.asarray(graph.y, dtype=float) * scale["y"],
            np.asarray(graph.x, dtype=float) * scale["x"],
        ]
    )


def link_graph(
    nodes: TrackingGraph,
    max_distance_um: float = LINK_MAX_UM,
    scale: dict[str, float] = VOXEL_SCALE_UM,
) -> TrackingGraph:
    """Add edges to a node-only graph by linking each consecutive timepoint pair.

    Returns a new TrackingGraph with the same nodes and the linked edges filled in.
    Only pairings within ``max_distance_um`` micrometers are accepted, so cells that
    appear or disappear are left unlinked instead of forced into an implausible edge.
    """
    coords = _scaled(nodes, scale)
    node_t = np.asarray(nodes.t)
    node_ids = np.asarray(nodes.node_ids)
    timepoints = np.unique(node_t)

    edges: list[tuple[int, int]] = []
    for earlier, later in zip(timepoints[:-1], timepoints[1:], strict=False):
        a_idx = np.flatnonzero(node_t == earlier)
        b_idx = np.flatnonzero(node_t == later)
        if a_idx.size == 0 or b_idx.size == 0:
            continue

        diff = coords[a_idx][:, None, :] - coords[b_idx][None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))

        forbid = max_distance_um * 10.0 + 1.0
        cost = np.where(dist <= max_distance_um, dist, forbid)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            if dist[r, c] <= max_distance_um:
                edges.append((int(node_ids[a_idx[r]]), int(node_ids[b_idx[c]])))

    edge_array = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
    return TrackingGraph(
        node_ids=nodes.node_ids,
        t=nodes.t,
        z=nodes.z,
        y=nodes.y,
        x=nodes.x,
        edges=edge_array,
    )
