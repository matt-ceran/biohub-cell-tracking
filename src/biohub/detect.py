"""Classical 3D cell-center detection.

Two detector modes share one code path:

* ``"peak"`` (the Phase-2 baseline): smooth each volume, subtract the local
  background with a white top-hat, and keep bright local maxima. Its threshold is
  a single intensity value per frame (a percentile, or an absolute number). This is
  a *global* rule: a dim cell is only found if it out-shines the brightest ~0.5%
  of the whole frame, so a few bright structures can hide the dim cells entirely.

* ``"dog"`` (Phase-3 adaptive blob detection): compute a Difference-of-Gaussians
  response, which is a band-pass filter tuned to the cell scale. It reacts to
  *local contrast* - a blob standing out from its immediate surroundings - not to
  absolute brightness, so a dim-but-clear cell next to bright junk still produces a
  clear peak. Its threshold is *adaptive*: ``median + k * robust_sigma`` of the
  response, i.e. "k noise levels above this frame's own noise floor", which tracks
  each frame instead of assuming a fixed brightness. Lowering ``k`` recovers fainter
  cells but admits more false detections (the Phase-5 recall/over-prediction trade-off);
  ``max_peaks`` caps the count by keeping the strongest responses per volume.

The detector is deliberately simple and fully parameterised so either mode can be
tuned to produce plausible per-movie cell counts (see the notes on the node
over-prediction penalty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub.metric import TrackingGraph


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable knobs for the classical detector.

    ``method`` selects the response image: ``"peak"`` (smoothed + top-hat intensity)
    or ``"dog"`` (Difference-of-Gaussians blob response). The threshold is chosen in
    priority order: ``threshold_k`` (adaptive k-sigma) if set, else ``threshold_abs``
    if set, else ``threshold_percentile``. ``max_peaks`` optionally caps detections per
    volume, keeping the strongest responses (the over-prediction knob).
    """

    method: str = "peak"  # "peak" | "dog"
    min_distance: int = 3  # minimum voxel separation between two detected centers
    max_peaks: int | None = None  # optional cap on detections per volume (brightest kept)

    # Shared / peak-mode knobs.
    smoothing_sigma: float = 1.0  # gaussian blur in voxels before peak finding (peak mode)
    background_radius: int = 6  # white-tophat radius in voxels; 0 disables background subtraction

    # dog-mode knobs (blobs are bright at the small scale and flat at the large scale).
    dog_sigma_small: float = 1.0  # inner gaussian, ~cell core in voxels
    dog_sigma_large: float = 2.0  # outer gaussian, ~local background in voxels

    # Threshold selection.
    threshold_k: float | None = None  # adaptive: median + k * (1.4826 * MAD) of the response
    threshold_percentile: float = 99.5  # intensity percentile used as the peak threshold
    threshold_abs: float | None = None  # absolute threshold; overrides the percentile when set


DEFAULT_DETECTOR = DetectorConfig()
# Phase-3 adaptive blob preset: local-contrast response with a noise-relative threshold.
DOG_DETECTOR = DetectorConfig(
    method="dog",
    dog_sigma_small=1.0,
    dog_sigma_large=2.0,
    threshold_k=4.0,
)
# Phase-5 high-recall preset: a lower bar (k=3) recovers the faint, contrast-limited
# cells the k=4 detector misses -- local flow edge Jaccard 0.573 -> 0.629 across the 6
# validation movies (144b256d 0.32 -> 0.59). It is NOT the default: the gain rides on
# ~25-55% more detections, which the real competition node score penalises and which
# strength-capping cannot trim without discarding the same faint cells (see the
# calibration experiments note). Bank it via linker-aware pruning before shipping.
HIGH_RECALL_DETECTOR = DetectorConfig(
    method="dog",
    dog_sigma_small=1.0,
    dog_sigma_large=2.0,
    threshold_k=3.0,
)


def _peak_response(vol: np.ndarray, smoothing_sigma: float, background_radius: int) -> np.ndarray:
    """Smoothed, background-subtracted intensity image (the Phase-2 response)."""
    from scipy.ndimage import gaussian_filter, white_tophat

    if smoothing_sigma > 0:
        vol = gaussian_filter(vol, sigma=smoothing_sigma)
    if background_radius > 0:
        vol = white_tophat(vol, size=background_radius)
    return vol


def _dog_response(vol: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    """Difference-of-Gaussians band-pass response, bright on cell-scale blobs.

    The small blur keeps the cell core; the large blur estimates the local background.
    Their difference is near zero on flat or slowly-varying regions and peaks on blobs
    of roughly the cell scale, so it responds to local contrast rather than absolute
    brightness and needs no separate background subtraction.
    """
    from scipy.ndimage import gaussian_filter

    small = gaussian_filter(vol, sigma=sigma_small)
    large = gaussian_filter(vol, sigma=sigma_large)
    return small - large


def _robust_threshold(response: np.ndarray, k: float) -> float:
    """Adaptive threshold ``median + k * sigma`` with sigma from the MAD.

    The median absolute deviation (scaled by 1.4826 to match a Gaussian standard
    deviation) is a robust noise estimate that a handful of bright blobs cannot skew,
    so ``k`` reads as "how many noise levels above this frame's floor" - the same
    stringency in a dim frame as in a bright one.
    """
    med = float(np.median(response))
    mad = float(np.median(np.abs(response - med)))
    sigma = 1.4826 * mad
    return med + k * sigma


def detect_centers(volume: np.ndarray, config: DetectorConfig = DEFAULT_DETECTOR) -> np.ndarray:
    """Detect bright cell centers in one 3D volume, returning (N, 3) z,y,x voxel coords."""
    from skimage.feature import peak_local_max

    vol = np.asarray(volume, dtype=np.float32)
    if config.method == "dog":
        response = _dog_response(vol, config.dog_sigma_small, config.dog_sigma_large)
    elif config.method == "peak":
        response = _peak_response(vol, config.smoothing_sigma, config.background_radius)
    else:
        raise ValueError(f"Unknown detector method {config.method!r}; use 'peak' or 'dog'.")

    if config.threshold_k is not None:
        threshold = _robust_threshold(response, config.threshold_k)
    elif config.threshold_abs is not None:
        threshold = config.threshold_abs
    else:
        threshold = float(np.percentile(response, config.threshold_percentile))

    coords = peak_local_max(
        response,
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
