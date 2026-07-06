"""Tests for the classical detector and the frame-to-frame linker."""

from __future__ import annotations

import numpy as np

from biohub.detect import (
    DOG_DETECTOR,
    DetectorConfig,
    detect_centers,
    detect_movie,
)
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
