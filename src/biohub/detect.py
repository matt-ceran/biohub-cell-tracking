"""Classical 3D cell-center detection.

This is the baseline detector: no training, just image processing. A cell center
shows up as a small bright blob against a darker background, so we smooth each 3D
volume, subtract the local background, and keep bright local maxima as detections.

The detector is deliberately simple and fully parameterised so the threshold can be
tuned to produce plausible per-movie cell counts (see the project notes on the node
over-prediction penalty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.metric import TrackingGraph


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable knobs for the classical detector."""

    smoothing_sigma: float = 1.0  # gaussian blur in voxels before peak finding
    background_radius: int = 6  # white-tophat radius in voxels; 0 disables background subtraction
    min_distance: int = 3  # minimum voxel separation between two detected centers
    threshold_percentile: float = 99.5  # intensity percentile used as the peak threshold
    threshold_abs: float | None = None  # absolute threshold; overrides the percentile when set
    max_peaks: int | None = None  # optional cap on detections per volume (brightest kept)


DEFAULT_DETECTOR = DetectorConfig()


def detect_centers(volume: np.ndarray, config: DetectorConfig = DEFAULT_DETECTOR) -> np.ndarray:
    """Detect bright cell centers in one 3D volume, returning (N, 3) z,y,x voxel coords."""
    from scipy.ndimage import gaussian_filter, white_tophat
    from skimage.feature import peak_local_max

    vol = np.asarray(volume, dtype=np.float32)
    if config.smoothing_sigma > 0:
        vol = gaussian_filter(vol, sigma=config.smoothing_sigma)
    if config.background_radius > 0:
        vol = white_tophat(vol, size=config.background_radius)

    threshold = (
        config.threshold_abs
        if config.threshold_abs is not None
        else float(np.percentile(vol, config.threshold_percentile))
    )
    coords = peak_local_max(
        vol,
        min_distance=config.min_distance,
        threshold_abs=threshold,
        num_peaks=config.max_peaks if config.max_peaks is not None else np.inf,
    )
    return coords


def detect_movie(
    image_array,
    config: DetectorConfig = DEFAULT_DETECTOR,
    t_range: range | None = None,
) -> TrackingGraph:
    """Run the detector on every timepoint and return nodes (no edges yet).

    ``image_array`` is any object indexable as ``[t]`` yielding a (Z, Y, X) volume,
    such as the array returned by ``biohub.io.open_image_array``.
    """
    n_timepoints = image_array.shape[0]
    frames = t_range if t_range is not None else range(n_timepoints)

    node_ids: list[int] = []
    ts: list[int] = []
    zs: list[float] = []
    ys: list[float] = []
    xs: list[float] = []
    next_id = 0
    for t in frames:
        volume = np.asarray(image_array[t])
        for z, y, x in detect_centers(volume, config):
            node_ids.append(next_id)
            ts.append(int(t))
            zs.append(float(z))
            ys.append(float(y))
            xs.append(float(x))
            next_id += 1

    return TrackingGraph(
        node_ids=np.array(node_ids, dtype=np.int64),
        t=np.array(ts),
        z=np.array(zs, dtype=float),
        y=np.array(ys, dtype=float),
        x=np.array(xs, dtype=float),
        edges=np.empty((0, 2), dtype=np.int64),
    )
