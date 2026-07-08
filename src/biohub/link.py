"""Frame-to-frame and whole-movie linking.

Given detected cell centers per timepoint, a linker connects each cell to its most
likely continuation in later timepoints. This is the "connect the dots" step: cells
barely move between frames, so the nearest plausible detection is almost always the
same cell. Two linkers live here.

* ``link_graph`` (Phase-2 baseline) is *greedy across frame pairs*: it solves each
  adjacent pair ``(t, t+1)`` as an independent gated bipartite assignment (Hungarian /
  ``linear_sum_assignment``) on physical distance, then throws that pair away. The
  matching *within* a pair is optimal, but it commits with no view of the rest of the
  movie, so a nearby junk detection can steal a link from the true continuation.

* ``link_graph_flow`` (Phase-4) solves the *whole movie at once* as a min-cost flow.
  Every detection is a node that may carry one unit of "cell identity"; each plausible
  hop between frames is a priced route (price = physical distance); starting or ending a
  track costs a fixed fee. The solver finds the single cheapest set of routes that
  threads identity through the whole movie. Because it weighs whole journeys rather than
  one hop, it resists distraction: routing a real track through a junk detection forces
  that track to dead-end (a death fee) and the real continuation to restart (a birth
  fee), so the globally cheapest solution prefers the true, continuous track.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub.constants import LINK_MAX_UM, VOXEL_SCALE_UM
from biohub.metric import TrackingGraph

# Phase-4 flow-linker defaults, in micrometer-equivalent units.
# APPEAR_REWARD == the gate means "any hop within the gate is worth making"; END_COST
# (birth == death fee) is the precision knob: higher -> fewer, longer, more-committed
# tracks. GAP_PENALTY prices each skipped frame when gap-closing is enabled.
FLOW_APPEAR_REWARD_UM = LINK_MAX_UM        # 8.0
FLOW_END_COST_UM = LINK_MAX_UM / 2.0       # 4.0
FLOW_GAP_PENALTY_UM = 3.0
FLOW_KNN = 8                               # candidate targets kept per source (nearest)
_FLOW_SCALE = 1000                         # micrometers -> integer nanometers for the solver


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


def _candidate_links(
    coords: np.ndarray,
    node_t: np.ndarray,
    max_distance_um: float,
    max_gap: int,
    knn: int,
) -> list[tuple[int, int, float]]:
    """Enumerate plausible ``(source_idx, target_idx, distance_um)`` hops.

    For each detection, its ``knn`` nearest detections in each of the next ``max_gap``
    timepoints are kept, provided they lie within ``max_distance_um``. Restricting to
    nearest neighbours keeps the flow network small without losing the true continuation,
    which is essentially always the closest plausible detection.
    """
    from scipy.spatial import cKDTree

    tps = np.unique(node_t)
    by_t = {int(tp): np.flatnonzero(node_t == tp) for tp in tps}
    tp_set = set(by_t)

    links: list[tuple[int, int, float]] = []
    for tp in tps:
        src = by_t[int(tp)]
        if src.size == 0:
            continue
        for gap in range(1, max_gap + 1):
            later = int(tp) + gap
            if later not in tp_set:
                continue
            dst = by_t[later]
            if dst.size == 0:
                continue
            tree = cKDTree(coords[dst])
            k = min(knn, dst.size)
            dists, idxs = tree.query(coords[src], k=k)
            dists = np.atleast_2d(dists.T).T  # normalise shape to (len(src), k)
            idxs = np.atleast_2d(idxs.T).T
            for si in range(src.shape[0]):
                for d, j in zip(dists[si], idxs[si], strict=True):
                    if d <= max_distance_um:
                        links.append((int(src[si]), int(dst[int(j)]), float(d)))
    return links


def link_graph_flow(
    nodes: TrackingGraph,
    max_distance_um: float = LINK_MAX_UM,
    max_gap: int = 1,
    appear_reward_um: float = FLOW_APPEAR_REWARD_UM,
    end_cost_um: float = FLOW_END_COST_UM,
    gap_penalty_um: float = FLOW_GAP_PENALTY_UM,
    knn: int = FLOW_KNN,
    scale: dict[str, float] = VOXEL_SCALE_UM,
) -> TrackingGraph:
    """Link a node-only graph by whole-movie min-cost flow.

    The movie is turned into one flow network and solved for the single cheapest set of
    tracks (see the module docstring). Compared with :func:`link_graph`, the frame pairs
    are coupled through each detection's shared in/out capacity, so links are chosen for
    global track consistency rather than one frame pair at a time.

    Parameters mirror the physical priors, all in micrometers:

    * ``max_distance_um`` gates hops, as in the greedy linker.
    * ``appear_reward_um`` is how much cost is saved by using a detection in a track;
      set to the gate, so any within-gate hop is worth making.
    * ``end_cost_um`` is the fee to start or end a track. Raising it favours fewer,
      longer, more-committed tracks (higher precision, the main knob).
    * ``gap_penalty_um`` prices each skipped frame when ``max_gap > 1``. With dense
      ground truth (labels at every frame) gap-closing cannot create true edges, so the
      default ``max_gap=1`` keeps links strictly frame-to-frame.

    Returns a new :class:`TrackingGraph` with the same nodes and the flow-selected edges.
    """
    import networkx as nx

    coords = _scaled(nodes, scale)
    node_t = np.asarray(nodes.t)
    node_ids = np.asarray(nodes.node_ids)
    n = node_ids.shape[0]
    empty = np.empty((0, 2), dtype=np.int64)
    if n == 0:
        return TrackingGraph(nodes.node_ids, nodes.t, nodes.z, nodes.y, nodes.x, empty)

    def w(um: float) -> int:
        return int(round(um * _FLOW_SCALE))

    reward = w(appear_reward_um)
    end = w(end_cost_um)

    graph = nx.DiGraph()
    for i in range(n):
        # Using a detection saves ``reward``; each detection carries at most one unit.
        graph.add_edge(("u", i), ("v", i), capacity=1, weight=-reward)
        graph.add_edge("S", ("u", i), capacity=1, weight=end)  # birth fee
        graph.add_edge(("v", i), "T", capacity=1, weight=end)  # death fee

    for src, dst, dist_um in _candidate_links(coords, node_t, max_distance_um, max_gap, knn):
        gap = int(node_t[dst]) - int(node_t[src])
        cost = w(dist_um + (gap - 1) * gap_penalty_um)
        graph.add_edge(("v", src), ("u", dst), capacity=1, weight=cost)

    # Feedback edge closes the network into a circulation: min_cost_flow then routes flow
    # around exactly those source->...->sink loops (real tracks) whose total cost is
    # negative, i.e. whose reward outweighs their birth + hop + death fees.
    graph.add_edge("T", "S", capacity=n, weight=0)
    for node in graph.nodes:
        graph.nodes[node]["demand"] = 0

    flow = nx.min_cost_flow(graph)

    edges: list[tuple[int, int]] = []
    for i in range(n):
        for dst_node, sent in flow.get(("v", i), {}).items():
            if sent > 0 and isinstance(dst_node, tuple) and dst_node[0] == "u":
                edges.append((int(node_ids[i]), int(node_ids[dst_node[1]])))

    edge_array = np.array(edges, dtype=np.int64) if edges else empty
    return TrackingGraph(
        node_ids=nodes.node_ids,
        t=nodes.t,
        z=nodes.z,
        y=nodes.y,
        x=nodes.x,
        edges=edge_array,
    )


def prune_to_tracks(linked: TrackingGraph, min_track_length: int = 2) -> TrackingGraph:
    """Keep only detections that link into a track of at least ``min_track_length`` nodes.

    A linked graph's edges form tracks: because each detection carries at most one unit of
    identity, every node has at most one predecessor and one successor, so the connected
    components of the (undirected) edge graph are simple paths -- the tracks. Their node
    count is the track length.

    This is linker-aware pruning: a real cell persists across frames and links into a
    multi-node track, whereas a junk detection tends to stay isolated or form only a short
    stub. Dropping the short components therefore ranks detections by *track participation*
    rather than raw response strength -- the one signal that separates faint-but-real cells
    (which a strength cap would wrongly discard) from junk.

    With the default ``min_track_length=2`` only edgeless nodes are removed, so no edge is
    dropped and the edge score is unchanged -- the node count simply falls. Larger
    thresholds trim short tracks too, cutting more nodes at the cost of any true short
    tracks' edges.
    """
    import networkx as nx

    node_ids = np.asarray(linked.node_ids)
    if node_ids.size == 0:
        return linked
    edges = np.asarray(linked.edges).reshape(-1, 2)

    graph = nx.Graph()
    graph.add_nodes_from(int(i) for i in node_ids)
    graph.add_edges_from((int(a), int(b)) for a, b in edges)
    keep: set[int] = set()
    for component in nx.connected_components(graph):
        if len(component) >= min_track_length:
            keep.update(component)

    node_mask = np.array([int(i) in keep for i in node_ids], dtype=bool)
    if edges.size:
        edge_mask = np.array([int(a) in keep and int(b) in keep for a, b in edges], dtype=bool)
        kept_edges = edges[edge_mask].astype(np.int64)
    else:
        kept_edges = np.empty((0, 2), dtype=np.int64)

    return TrackingGraph(
        node_ids=node_ids[node_mask],
        t=np.asarray(linked.t)[node_mask],
        z=np.asarray(linked.z)[node_mask],
        y=np.asarray(linked.y)[node_mask],
        x=np.asarray(linked.x)[node_mask],
        edges=kept_edges,
    )
