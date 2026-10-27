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


def smooth_in_time(values: np.ndarray, times: np.ndarray, *, window: float = 900.0,
                   kind: str = "median", min_points: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Replace every point by the median/mean of the points within a time window around it.

    A centred boxcar in *time* rather than in samples, because calibration
    solutions are not evenly spaced: scans have gaps, and a sample-counted
    window would reach minutes further on a sparse stretch than on a dense one.
    A centred window leaves a linear drift unbiased, which matters because the
    atmospheric delay and rate it is used on do drift linearly over a scan or two.

    Points that are ``nan`` take no part in any window and come back ``nan``: a
    flagged solution is a non-detection, and turning it into an average of its
    neighbours would apply a number the data never supported.

    Parameters
    ----------
    values : numpy.ndarray
        Series to smooth (1-D); ``nan`` marks unusable points.
    times : numpy.ndarray
        Time of each point, same length as ``values``, any consistent unit.
    window : float
        Full width of the window in the units of ``times``; <= 0 returns the
        input unchanged.
    kind : str
        ``"median"`` (robust, the default) or ``"mean"``.
    min_points : int
        Leave a point untouched when its window holds fewer usable points than
        this, so a single isolated solution is never "smoothed" with itself.

    Returns
    -------
    smoothed : numpy.ndarray
        The smoothed series (``nan`` where the input was ``nan``).
    changed : numpy.ndarray
        Boolean mask of the points whose value actually moved.
    """
    series = np.asarray(values, dtype=float)
    stamps = np.asarray(times, dtype=float)
    smoothed = np.array(series, dtype=float, copy=True)
    if window <= 0 or series.size == 0:
        return smoothed, np.zeros(series.shape, dtype=bool)
    if kind not in ("median", "mean"):
        raise ValueError(f"unknown smoothing kind {kind!r} (use 'median' or 'mean')")
    reduce = np.median if kind == "median" else np.mean
    usable = np.isfinite(series)
    half = 0.5 * float(window)
    for index in np.flatnonzero(usable):
        inside = usable & (np.abs(stamps - stamps[index]) <= half)
        if int(inside.sum()) < max(1, int(min_points)):
            continue
        smoothed[index] = float(reduce(series[inside]))
    changed = usable & (smoothed != series)
    return smoothed, changed


def outliers_in_time(values: np.ndarray, times: np.ndarray, *, window: float = 7200.0,
                     threshold: float = 5.0, min_points: int = 4, floor: float = 0.0) -> np.ndarray:
    """Return a mask of the points that depart from their neighbours in time by > ``threshold`` sigma.

    Each point is compared with the median of the *other* usable points within
    a centred time window, in units of their MAD-based scatter. Leaving the point
    itself out keeps a wild value from inflating the very scatter it is judged
    against, which is what lets a single false fringe-fit detection stand out.

    Parameters
    ----------
    values : numpy.ndarray
        Series to test (1-D); ``nan`` marks unusable points, never marked.
    times : numpy.ndarray
        Time of each point, same length as ``values``, any consistent unit.
    window : float
        Full width of the window in the units of ``times``; <= 0 marks nothing.
    threshold : float
        Departure in robust sigmas beyond which a point is an outlier; <= 0 marks nothing.
    min_points : int
        Leave a point unjudged when fewer than this many *other* usable points
        share its window: too few neighbours give no trustworthy reference.
    floor : float
        Smallest departure ever marked, in the units of ``values``. Precise
        solutions have a tiny scatter, so a genuine slow drift can read as many
        sigmas; the floor keeps only departures that are physically implausible.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where the point is an outlier.
    """
    series = np.asarray(values, dtype=float)
    stamps = np.asarray(times, dtype=float)
    mask = np.zeros(series.shape, dtype=bool)
    if window <= 0 or threshold <= 0 or series.size == 0:
        return mask
    usable = np.isfinite(series)
    half = 0.5 * float(window)
    for index in np.flatnonzero(usable):
        inside = usable & (np.abs(stamps - stamps[index]) <= half)
        inside[index] = False
        if int(inside.sum()) < max(1, int(min_points)):
            continue
        sigma = mad_sigma(series[inside])
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        departure = abs(series[index] - np.median(series[inside]))
        mask[index] = departure > max(threshold * sigma, float(floor))
    return mask


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


def normalised_baseline_amplitude(amplitude: np.ndarray, scan_index: np.ndarray, *,
                                  window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Divide every baseline's amplitude by its own level, scan by scan.

    The level of a baseline in a scan is the median over ``window`` scans (centred
    on it) of the scan medians: one bad scan, or a scan that is bad for most of
    its length, does not set its own reference.

    Parameters
    ----------
    amplitude : numpy.ndarray
        ``(n_antenna, n_antenna, n_time)`` amplitudes, symmetric, ``nan`` where there is no data.
    scan_index : numpy.ndarray
        Scan number of every time sample.
    window : int
        Number of scans the level is taken over.

    Returns
    -------
    (ratio, scatter)
        ``ratio`` has the shape of ``amplitude`` (1 = the baseline's level); ``scatter`` is the
        robust relative scatter of each baseline over the whole track, ``(n_antenna, n_antenna)``.
    """
    import warnings
    scans = np.unique(scan_index)
    ratio = np.full(amplitude.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-nan slices: baselines without data
        medians = np.stack([np.nanmedian(amplitude[:, :, scan_index == s], axis=2) for s in scans], axis=2)
        half = max(1, int(window)) // 2
        for k, scan in enumerate(scans):
            level = np.nanmedian(medians[:, :, max(0, k - half):k + half + 1], axis=2)
            level = np.where(level > 0, level, np.nan)
            ratio[:, :, scan_index == scan] = amplitude[:, :, scan_index == scan] / level[:, :, np.newaxis]
        centre = np.nanmedian(ratio, axis=2, keepdims=True)
        scatter = 1.4826 * np.nanmedian(np.abs(ratio - centre), axis=2)
    return ratio, scatter


def antenna_on_source_fraction(ratio: np.ndarray, usable: np.ndarray, *, iterations: int = 30,
                               prior: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Split normalised baseline amplitudes into one factor per antenna and time sample.

    Solves ``ratio[i, j, t] ~ g[i, t] * g[j, t]`` for every time sample at once:
    an antenna that is off source (still slewing, a dropout) pulls *all* its
    baselines down together and comes out with a small ``g``, while the antennas
    it is correlated with keep theirs near 1. This is what tells "this antenna
    is late" from "its partner is late", which no per-baseline cut can.

    ``prior`` is a weak pull towards 1 that keeps the solution defined when an
    antenna has no partner on source. A group of antennas that are all off
    together still comes out low, and so does an antenna whose every partner is
    off - none of its baselines is usable at that moment anyway.

    Parameters
    ----------
    ratio : numpy.ndarray
        ``(n_antenna, n_antenna, n_time)``, symmetric, 1 = the baseline's own level, ``nan`` = no data.
    usable : numpy.ndarray
        ``(n_antenna, n_antenna)`` boolean: baselines steady enough to be trusted.

    Returns
    -------
    (gains, n_baselines)
        ``gains`` ``(n_antenna, n_time)`` and the number of usable baselines with
        data each one rests on.
    """
    weight = (np.isfinite(ratio) & usable[:, :, np.newaxis]).astype(float)
    values = np.where(weight > 0, np.clip(np.nan_to_num(ratio), 0.0, 3.0), 0.0)
    gains = np.ones((ratio.shape[0], ratio.shape[2]))
    for _ in range(int(iterations)):
        numerator = np.einsum("ijt,jt->it", weight * values, gains) + prior
        denominator = np.einsum("ijt,jt->it", weight, gains ** 2) + prior
        gains = np.clip(0.5 * (gains + numerator / denominator), 0.0, 3.0)
    return gains, weight.sum(axis=1).astype(int)


def weak_scans_per_antenna(amplitude: np.ndarray, scan_index: np.ndarray, *, level: float = 0.3,
                            min_baselines: int = 2) -> dict:
    """Find the scans of one source in which an antenna is far below its own best scan.

    An antenna that is not properly on source (in a sidelobe, mispointed) can
    still be *detected* on a bright fringe finder: the fringe SNR CASA reports
    hardly drops (EM163: CM at 6% of its amplitude in one scan kept two thirds of
    its SNR), so a search on SNR alone takes such a scan for a good one. The
    amplitude does show it. Every baseline's level in a scan (median over the
    scan) is divided by its level in its best scan, and those ratios are split
    into one factor per antenna and scan (:func:`antenna_on_source_fraction`), so
    that a weak partner or a baseline that resolves the source differently at
    another hour angle is not held against the antenna.

    Works on uncalibrated amplitudes: only the ratio between scans of the same
    baseline is used. With a single scan there is nothing to compare with and
    nothing is reported.

    Parameters
    ----------
    amplitude : numpy.ndarray
        ``(n_antenna, n_antenna, n_time)`` amplitudes of one source, symmetric, ``nan`` = no data.
    scan_index : numpy.ndarray
        Scan number of every time sample.
    level : float
        An antenna is weak in a scan when its factor is below this (1 = as in its best scan).
    min_baselines : int
        Factors resting on fewer baselines with data are not judged.

    Returns
    -------
    dict
        ``scans`` (scan numbers, the last axis of the rest), ``gains``
        ``(n_antenna, n_scan)`` and ``weak`` (boolean, same shape).
    """
    import warnings
    scans = np.unique(scan_index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-nan slices: baselines without data
        levels = np.stack([np.nanmedian(amplitude[:, :, scan_index == s], axis=2) for s in scans], axis=2)
        best = np.nanmax(levels, axis=2)
    best = np.where(best > 0, best, np.nan)
    ratio = levels / best[:, :, np.newaxis]
    gains, n_baselines = antenna_on_source_fraction(ratio, np.isfinite(best))
    weak = (n_baselines >= int(min_baselines)) & (gains < level) if scans.size > 1 else np.zeros(gains.shape, bool)
    return {"scans": scans, "gains": gains, "weak": weak}


def detect_off_source(amplitude: np.ndarray, scan_index: np.ndarray, *, level: float = 0.8,
                      max_scatter: float = 0.3, min_baselines: int = 2) -> dict:
    """Find the time samples at which an antenna was not on source, from calibrated amplitudes.

    Parameters
    ----------
    amplitude : numpy.ndarray
        ``(n_antenna, n_antenna, n_time)`` calibrated amplitudes of one bright
        source (channel-averaged), symmetric, ``nan`` where flagged.
    scan_index : numpy.ndarray
        Scan number of every time sample.
    level : float
        An antenna counts as off source when its factor (see
        :func:`antenna_on_source_fraction`) is below this.
    max_scatter : float
        Baselines whose relative scatter exceeds this are too noisy to vote.
    min_baselines : int
        Samples resting on fewer usable baselines are left undecided.

    Returns
    -------
    dict
        ``off`` and ``judged`` ``(n_antenna, n_time)`` booleans, ``gains``, and
        ``usable`` ``(n_antenna, n_antenna)``.
    """
    ratio, scatter = normalised_baseline_amplitude(amplitude, scan_index)
    usable = np.isfinite(scatter) & (scatter < max_scatter)
    gains, n_baselines = antenna_on_source_fraction(ratio, usable)
    judged = n_baselines >= int(min_baselines)
    return {"off": judged & (gains < level), "judged": judged, "gains": gains, "usable": usable}
