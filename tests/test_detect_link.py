"""Tests for the classical detector and the frame-to-frame linker."""

from __future__ import annotations

import numpy as np

from biohub.detect import DetectorConfig, detect_centers, detect_movie
from biohub.link import link_graph
from biohub.metric import TrackingGraph


def _volume_with_blobs(centers, shape=(16, 64, 64), peak=2000.0, sigma=1.5):
    """Build a dark volume with bright Gaussian blobs at the given z,y,x centers."""
    from scipy.ndimage import gaussian_filter

    vol = np.zeros(shape, dtype=np.float32)
    for z, y, x in centers:
        vol[z, y, x] = peak
    vol = gaussian_filter(vol, sigma=sigma)
    vol += 20.0  # flat background
    return vol


def test_detect_centers_finds_known_blobs():
    centers = [(8, 20, 20), (8, 20, 45), (4, 45, 30)]
    vol = _volume_with_blobs(centers)
    found = detect_centers(vol, DetectorConfig(background_radius=0, threshold_percentile=99.0))
    # Every planted blob should have a detection within a couple of voxels.
    for cz, cy, cx in centers:
        dists = np.abs(found - np.array([cz, cy, cx])).sum(axis=1)
        assert dists.min() <= 2, f"missed blob at {(cz, cy, cx)}"


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
