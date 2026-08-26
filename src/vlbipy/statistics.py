"""Robust statistics shared by the calibration steps (pure numpy, no CASA).

Calibration diagnostics cannot use mean/standard deviation: a single bad solution
(a Tsys spike, a dead channel) moves both, so real outliers hide behind the
statistic meant to reveal them. Everything here is median/MAD based, which stays
stable until more than half the data is bad.

Used by the Tsys de-spiking in the a-priori step and by the subband edge-channel
detection after bandpass calibration.
"""
from __future__ import annotations

import numpy as np

#: Scale factor converting a median absolute deviation into a Gaussian-equivalent sigma.
MAD_TO_SIGMA = 1.4826


def mad_sigma(values: np.ndarray) -> float:
    """Return the MAD-based robust sigma of ``values`` (``nan`` if fewer than 2 finite points).

    Parameters
    ----------
    values : numpy.ndarray
        Sample; non-finite entries are ignored.

    Returns
    -------
    float
        ``1.4826 * median(|x - median(x)|)``, or 0.0 when every point is identical.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(finite - np.median(finite))))


def outlier_mask(values: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """Return a boolean mask of points deviating from the median by more than ``threshold`` sigma.

    Parameters
    ----------
    values : numpy.ndarray
        Sample to test (1-D).
    threshold : float
        Deviation in robust sigmas beyond which a point counts as an outlier.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the point is an outlier. Non-finite points are
        never marked (they are already unusable); an all-identical series yields
        an all-False mask.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.zeros(values.shape, dtype=bool)
    sigma = mad_sigma(values)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.zeros(values.shape, dtype=bool)
    deviation = np.abs(values - np.median(values[finite]))
    return finite & (deviation > threshold * sigma)


def running_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Return the running median of a 1-D series, with the window shrinking at the edges.

    Parameters
    ----------
    values : numpy.ndarray
        Input series (may contain ``nan``, which is ignored inside each window).
    window : int
        Window length in samples; forced odd and to at least 3.

    Returns
    -------
    numpy.ndarray
        Smoothed series, ``nan`` where a window held no finite point.
    """
    values = np.asarray(values, dtype=float)
    window = max(3, int(window) | 1)
    half = window // 2
    smoothed = np.full(values.shape, np.nan, dtype=float)
    for i in range(values.size):
        chunk = values[max(0, i - half):i + half + 1]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            smoothed[i] = np.median(chunk)
    return smoothed


def despike(values: np.ndarray, *, threshold: float = 6.0, max_passes: int = 3,
            window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Replace outliers in a series with the local running median, iterating until stable.

    Each pass re-measures the median and MAD of the *current* series, so spikes
    removed in one pass stop inflating the sigma for the next — which is what
    lets a heavily-spiked series converge instead of hiding its own outliers.

    Parameters
    ----------
    values : numpy.ndarray
        Series to clean (1-D). ``nan`` marks already-unusable points and is left alone.
    threshold : float
        Outlier cut in robust sigmas.
    max_passes : int
        Maximum number of cleaning passes; stops early once a pass finds nothing.
    window : int
        Running-median window used for the replacement value.

    Returns
    -------
    cleaned : numpy.ndarray
        The series with outliers replaced.
    replaced : numpy.ndarray
        Boolean mask of the points that were replaced (across all passes).
    """
    cleaned = np.array(values, dtype=float, copy=True)
    replaced = np.zeros(cleaned.shape, dtype=bool)
    for _ in range(max(1, int(max_passes))):
        mask = outlier_mask(cleaned, threshold)
        if not mask.any():
            break
        smooth = running_median(np.where(mask, np.nan, cleaned), window)
        cleaned[mask] = smooth[mask]
        replaced |= mask & np.isfinite(cleaned)
    return cleaned, replaced


def find_flat_range(profile: np.ndarray, *, threshold: float = 6.0,
                    max_edge_fraction: float = 0.25) -> tuple[int, int]:
    """Return the ``(first, last)`` indices of the flat interior of a bandpass-like profile.

    Subband edges roll off, so the first and last channels deviate from the flat
    interior. The interior is characterised by the median/MAD of the central
    half of the profile — a region that is flat by construction — and channels
    are then trimmed inward from each edge while they deviate by more than
    ``threshold`` sigma or are unusable.

    Parameters
    ----------
    profile : numpy.ndarray
        Per-channel statistic (amplitude or phase scatter), 1-D.
    threshold : float
        Deviation in robust sigmas beyond which an edge channel is rejected.
    max_edge_fraction : float
        Never trim more than this fraction of the band from either edge; a
        profile that is bad throughout is a calibration problem, not an edge one.

    Returns
    -------
    tuple of int
        Inclusive ``(first, last)`` channel indices of the flat range. Returns the
        full range when the profile is too short or uniformly flat.
    """
    profile = np.asarray(profile, dtype=float)
    n_channels = profile.size
    if n_channels < 8:
        return (0, max(0, n_channels - 1))
    core = profile[n_channels // 4: 3 * n_channels // 4]
    core = core[np.isfinite(core)]
    if core.size < 3:
        return (0, n_channels - 1)
    centre, sigma = float(np.median(core)), mad_sigma(core)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return (0, n_channels - 1)
    max_trim = int(n_channels * max_edge_fraction)

    def deviates(index: int) -> bool:
        value = profile[index]
        return not np.isfinite(value) or abs(value - centre) > threshold * sigma

    first = 0
    while first < max_trim and deviates(first):
        first += 1
    last = n_channels - 1
    while (n_channels - 1 - last) < max_trim and deviates(last):
        last -= 1
    return (first, last)
