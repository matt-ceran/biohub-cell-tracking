"""Tests for the classical detector and the frame-to-frame linker."""

from __future__ import annotations

import numpy as np

from biohub.detect import (
    DOG_DETECTOR,
    DetectorConfig,
    detect_centers,
    detect_movie,
)
from biohub.link import link_graph, link_graph_flow, prune_to_tracks
from biohub.metric import TrackingGraph

# Identity scale so voxel coordinates equal micrometers, making distances easy to reason
# about in the linker tests below.
UNIT_SCALE = {"z": 1.0, "y": 1.0, "x": 1.0}


def _volume_with_blobs(centers, shape=(16, 64, 64), peak=2000.0, sigma=1.5):
    """Build a dark volume with bright Gaussian blobs at the given z,y,x centers."""
    from scipy.ndimage import gaussian_filter

    vol = np.zeros(shape, dtype=np.float32)
    for z, y, x in centers:
        vol[z, y, x] = peak
    vol = gaussian_filter(vol, sigma=sigma)
    vol += 20.0  # flat background
    return vol


def _found_within(found, center, tol=2):
    """True if some detection lands within ``tol`` voxels (L1) of ``center``."""
    if len(found) == 0:
        return False
    return np.abs(found - np.array(center)).sum(axis=1).min() <= tol


def test_detect_centers_finds_known_blobs():
    centers = [(8, 20, 20), (8, 20, 45), (4, 45, 30)]
    vol = _volume_with_blobs(centers)
    found = detect_centers(vol, DetectorConfig(background_radius=0, threshold_percentile=99.0))
    # Every planted blob should have a detection within a couple of voxels.
    for cz, cy, cx in centers:
        dists = np.abs(found - np.array([cz, cy, cx])).sum(axis=1)
        assert dists.min() <= 2, f"missed blob at {(cz, cy, cx)}"


def test_dog_detector_finds_known_blobs():
    centers = [(8, 20, 20), (8, 20, 45), (4, 45, 30)]
    vol = _volume_with_blobs(centers)
    rng = np.random.default_rng(0)
    vol = vol + rng.normal(0.0, 3.0, size=vol.shape).astype(np.float32)  # noise floor for k-sigma
    found = detect_centers(vol, DOG_DETECTOR)
    for center in centers:
        assert _found_within(found, center), f"dog missed blob at {center}"


def test_dog_finds_dim_blob_that_global_threshold_misses():
    """A crowded bright region pushes a global intensity percentile above a dim cell
    sitting in the dark region, so the percentile detector misses it. The DoG
    band-pass removes the smooth bright background, so local contrast still reveals
    the dim cell. This is the Phase-2 failure mode the adaptive detector fixes."""
    from scipy.ndimage import gaussian_filter

    shape = (16, 64, 64)
    dim = (8, 32, 16)  # in the dark half (x < 40)
    bright = (8, 32, 52)  # in the bright half (x >= 40)
    vol = np.zeros(shape, dtype=np.float32)
    vol[dim] = 4000.0
    vol[bright] = 4000.0
    vol = gaussian_filter(vol, sigma=1.5)  # two equal-contrast blobs (~75 above local bg)
    vol[:, :, 40:] += 500.0  # a crowded/bright half raises the global percentile
    vol += 20.0
    rng = np.random.default_rng(1)
    vol = vol + rng.normal(0.0, 3.0, size=vol.shape).astype(np.float32)

    peak = detect_centers(vol, DetectorConfig(background_radius=0, threshold_percentile=99.0))
    dog = detect_centers(vol, DOG_DETECTOR)

    # The global-threshold detector catches the bright-region blob but misses the dim one.
    assert _found_within(peak, bright), "peak detector should still find the bright blob"
    assert not _found_within(peak, dim), "global percentile is expected to miss the dim blob"
    # The adaptive DoG detector finds both.
    assert _found_within(dog, bright) and _found_within(dog, dim), "dog should find both blobs"


def test_max_peaks_caps_per_volume_keeping_strongest():
    """The ``max_peaks`` cap keeps the N strongest-response detections per volume — the
    Phase-5 knob for reining in over-prediction after lowering the threshold."""
    from scipy.ndimage import gaussian_filter

    shape = (16, 64, 64)
    blobs = {(8, 20, 20): 4000.0, (8, 20, 45): 3000.0, (4, 45, 30): 800.0}
    vol = np.zeros(shape, dtype=np.float32)
    for (z, y, x), amp in blobs.items():
        vol[z, y, x] = amp
    vol = gaussian_filter(vol, sigma=1.5) + 20.0
    rng = np.random.default_rng(2)
    vol = vol + rng.normal(0.0, 3.0, size=vol.shape).astype(np.float32)

    cfg = DetectorConfig(method="dog", threshold_k=3.0, max_peaks=2)
    capped = detect_centers(vol, cfg)
    assert len(capped) == 2
    # The dropped one should be the weakest blob, not one of the two bright ones.
    assert not _found_within(capped, (4, 45, 30))
    assert _found_within(capped, (8, 20, 20)) and _found_within(capped, (8, 20, 45))


def test_unknown_detector_method_raises():
    vol = _volume_with_blobs([(8, 20, 20)])
    try:
        detect_centers(vol, DetectorConfig(method="bogus"))
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown detector method")


def test_detect_movie_tags_timepoints():
    class FakeArray:
        shape = (3, 16, 64, 64)

        def __getitem__(self, t):
            return _volume_with_blobs([(8, 20, 20 + 3 * t)])

    cfg = DetectorConfig(background_radius=0, threshold_percentile=99.0)
    nodes = detect_movie(FakeArray(), cfg)
    assert set(np.unique(nodes.t)) == {0, 1, 2}
    assert nodes.edges.shape == (0, 2)
    assert len(nodes.node_ids) >= 3


def test_linker_connects_a_moving_cell():
    # One cell drifting a small distance across three frames should link into a chain.
    nodes = TrackingGraph(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        t=np.array([0, 1, 2]),
        z=np.array([0.0, 0.0, 0.0]),
        y=np.array([10.0, 12.0, 14.0]),
        x=np.array([10.0, 10.0, 10.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    linked = link_graph(nodes, max_distance_um=5.0)
    pairs = {tuple(e) for e in linked.edges}
    assert pairs == {(1, 2), (2, 3)}


def test_linker_respects_the_gate():
    # Two cells one frame apart but far beyond the gate must not be linked.
    nodes = TrackingGraph(
        node_ids=np.array([1, 2], dtype=np.int64),
        t=np.array([0, 1]),
        z=np.array([0.0, 0.0]),
        y=np.array([10.0, 200.0]),
        x=np.array([10.0, 200.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    linked = link_graph(nodes, max_distance_um=8.0)
    assert linked.edges.shape == (0, 2)


def _nodes(rows):
    """Build a node-only TrackingGraph from (node_id, t, z, y, x) rows."""
    rows = list(rows)
    return TrackingGraph(
        node_ids=np.array([r[0] for r in rows], dtype=np.int64),
        t=np.array([r[1] for r in rows]),
        z=np.array([r[2] for r in rows], dtype=float),
        y=np.array([r[3] for r in rows], dtype=float),
        x=np.array([r[4] for r in rows], dtype=float),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_flow_linker_connects_a_moving_cell():
    # The whole-movie flow linker should also chain a single drifting cell.
    nodes = _nodes([(1, 0, 0.0, 10.0, 10.0), (2, 1, 0.0, 12.0, 10.0), (3, 2, 0.0, 14.0, 10.0)])
    linked = link_graph_flow(nodes, scale=UNIT_SCALE)
    assert {tuple(e) for e in linked.edges} == {(1, 2), (2, 3)}


def test_flow_linker_respects_the_gate():
    # Two cells one frame apart but beyond the gate have no plausible hop, so no edge.
    nodes = _nodes([(1, 0, 0.0, 10.0, 10.0), (2, 1, 0.0, 200.0, 200.0)])
    linked = link_graph_flow(nodes, scale=UNIT_SCALE)
    assert linked.edges.shape == (0, 2)


def test_flow_linker_resists_a_distractor_that_greedy_falls_for():
    """The key Phase-4 property: a junk detection sitting closer to the cell than its
    true continuation steals the greedy per-frame link, but the whole-movie flow keeps
    the true, continuous track because dead-ending it through the junk costs more overall.

    Layout (micrometers): a cell A0->A1->A2 drifts straight along x; a junk detection J1
    sits nearer to A0 than A1 is, but off the track's line so it leads nowhere.
    """
    nodes = _nodes(
        [
            (10, 0, 0.0, 0.0, 0.0),  # A0
            (11, 1, 0.0, 0.0, 3.0),  # A1  true continuation (3 um from A0)
            (99, 1, 0.0, 2.0, 0.0),  # J1  junk, only 2 um from A0 but off the line
            (12, 2, 0.0, 0.0, 6.0),  # A2  continues the straight track
        ]
    )

    greedy = {tuple(e) for e in link_graph(nodes, scale=UNIT_SCALE).edges}
    # Greedy links A0 to the nearer junk and never recovers the true A0->A1 edge.
    assert (10, 99) in greedy
    assert (10, 11) not in greedy

    # With a firm end-cost (the precision knob), the flow prefers one clean A0->A1->A2
    # track and leaves the junk unused: dead-ending the track through J1 would pay an
    # extra birth + death that outweighs the shorter first hop.
    flow = {tuple(e) for e in link_graph_flow(nodes, end_cost_um=8.0, scale=UNIT_SCALE).edges}
    assert flow == {(10, 11), (11, 12)}


def test_prune_to_tracks_drops_isolated_junk_but_keeps_the_track():
    """A drifting cell (a real 3-node track) plus an isolated junk detection: default
    pruning (min_track_length=2) removes the junk and keeps the track, and because the
    junk had no edge, every edge survives."""
    nodes = _nodes(
        [
            (1, 0, 0.0, 10.0, 10.0),  # A0 ┐
            (2, 1, 0.0, 12.0, 10.0),  # A1 ├ one real track
            (3, 2, 0.0, 14.0, 10.0),  # A2 ┘
            (99, 1, 0.0, 200.0, 200.0),  # junk, far from everything -> no link
        ]
    )
    linked = link_graph_flow(nodes, scale=UNIT_SCALE)
    assert 99 in set(linked.node_ids)  # junk survives linking as an isolated node

    pruned = prune_to_tracks(linked, min_track_length=2)
    assert set(pruned.node_ids) == {1, 2, 3}  # junk dropped, track kept
    assert {tuple(e) for e in pruned.edges} == {(1, 2), (2, 3)}  # no edge lost


def test_prune_to_tracks_min_length_three_drops_a_two_node_track():
    """A stricter threshold trims short tracks too: a 2-node track is removed (with its
    edge) when min_track_length=3, while a 3-node track is kept."""
    nodes = _nodes(
        [
            (1, 0, 0.0, 10.0, 10.0),  # 3-node track
            (2, 1, 0.0, 12.0, 10.0),
            (3, 2, 0.0, 14.0, 10.0),
            (10, 0, 0.0, 50.0, 50.0),  # 2-node track, far from the first
            (11, 1, 0.0, 52.0, 50.0),
        ]
    )
    linked = link_graph_flow(nodes, scale=UNIT_SCALE)
    assert {tuple(e) for e in linked.edges} == {(1, 2), (2, 3), (10, 11)}

    pruned = prune_to_tracks(linked, min_track_length=3)
    assert set(pruned.node_ids) == {1, 2, 3}
    assert {tuple(e) for e in pruned.edges} == {(1, 2), (2, 3)}


def test_flow_linker_can_close_a_one_frame_gap_when_enabled():
    """Gap-closing is off by default (dense labels need no skips) but works when asked:
    a cell detected at t=0 and t=2 with nothing at t=1 links only when max_gap >= 2."""
    nodes = _nodes([(1, 0, 0.0, 0.0, 0.0), (2, 2, 0.0, 0.0, 3.0)])

    no_gap = link_graph_flow(nodes, max_gap=1, scale=UNIT_SCALE)
    assert no_gap.edges.shape == (0, 2)

    with_gap = link_graph_flow(nodes, max_gap=2, scale=UNIT_SCALE)
    assert {tuple(e) for e in with_gap.edges} == {(1, 2)}
