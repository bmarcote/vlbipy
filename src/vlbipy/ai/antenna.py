"""Antenna-based decomposition of baseline amplitudes.

Every amplitude problem of a VLBI calibration that matters - an antenna losing sensitivity, a noisy receptor,
a subband sitting low, two hands at different levels - is *antenna*-based, while the source structure is
*baseline*-based. Taking the logarithm turns the product of two station gains into a sum,
``log|V_ij| = x_i + x_j + log|S_ij|``, and the source term cancels when a baseline is compared with itself:

* over **time**, against its own median over the observation (:func:`gains_vs_time`): what changes is the
  stations, unless the structure changes with the uv track (the longest baselines of a resolved source - the
  check is that two different sources show the same trend);
* over **subbands**, against its own median over the band (:func:`subband_offsets`): VLBI bandwidths are far too
  narrow for a spectral index, so any subband-to-subband step is instrumental;
* between the **hands**, RR against LL (:func:`hand_ratios`): the calibrators are unpolarized in circular
  hands, so ``|RR| = |LL|`` and any ratio belongs to the stations.

Each set of baseline log-ratios is then solved for one value per antenna by iteratively reweighted least
squares (:func:`solve_antenna_terms`), which also returns how well the antenna model explains the baselines:
a baseline-based residual (a correlator problem, or real structure) is reported, not forced onto the stations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import visstats


def solve_antenna_terms(values: np.ndarray, weights: np.ndarray, ant1: np.ndarray, ant2: np.ndarray, nant: int, *,
                        clip: float = 4.0, iterations: int = 5, prior: float = 0.0) -> dict:
    """Least-squares ``values[k] = x[ant1[k]] + x[ant2[k]]``, with outlier rejection.

    Parameters
    ----------
    values, weights : array (nbl,)
        Baseline log-ratios and their weights (1 / variance); non-finite values or zero weights are ignored.
    clip : float
        Baselines whose normalised residual exceeds ``clip`` robust sigmas are left out of the next iteration.
    prior : float
        Tikhonov term pulling every ``x`` towards 0 (1 / prior variance); keeps a weakly connected antenna
        finite instead of letting it absorb noise.

    Returns
    -------
    dict
        ``x`` (nant,), ``err`` (nant,), ``nbl`` (baselines per antenna), ``resid`` (nbl,) normalised residuals,
        ``chi2r`` reduced chi-squared. Antennas without baselines get ``nan``.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x = np.full(nant, np.nan)
    err = np.full(nant, np.nan)
    nbl = np.bincount(ant1[ok], minlength=nant) + np.bincount(ant2[ok], minlength=nant)
    resid = np.full(values.shape, np.nan)
    if ok.sum() < 3:
        return {"x": x, "err": err, "nbl": nbl, "resid": resid, "chi2r": np.nan}
    active = np.flatnonzero(nbl > 0)
    index = -np.ones(nant, dtype=np.int64)
    index[active] = np.arange(active.size)
    rows = np.flatnonzero(ok)
    design = np.zeros((rows.size, active.size))
    design[np.arange(rows.size), index[ant1[rows]]] += 1.0
    design[np.arange(rows.size), index[ant2[rows]]] += 1.0
    y = values[rows]
    w = weights[rows].copy()
    robust = np.ones_like(w)
    covariance = np.eye(active.size)
    r = np.zeros_like(y)

    def fit(ww):
        normal = design.T @ (design * ww[:, None]) + prior * np.eye(active.size)
        cov = np.linalg.pinv(normal)
        return cov, cov @ (design.T @ (ww * y))

    # An L1 start (iteratively reweighted): least squares would spread a gross outlier over every residual and
    # hide it in an inflated scale. Then hard clipping at ``clip`` robust sigmas, iterated to convergence.
    for _ in range(5):
        covariance, solution = fit(w * robust)
        r = (y - design @ solution) * np.sqrt(w)
        robust = 1.0 / np.maximum(np.abs(r), 1e-3)
    for _ in range(iterations):
        scale = max(1.4826 * np.median(np.abs(r)), 1.0) if r.size > active.size else 1.0
        new = (np.abs(r) <= clip * scale).astype(float)    # a gross outlier is rejected, not merely down-weighted
        if new.sum() <= active.size:                      # never reject so much that the system loses its rank
            new = np.ones_like(new)
        covariance, solution = fit(w * new)
        r = (y - design @ solution) * np.sqrt(w)
        if np.array_equal(new, robust):
            break
        robust = new
    dof = max(rows.size - active.size, 1)
    chi2r = float(np.sum(robust * r ** 2) / dof)
    x[active] = solution
    err[active] = np.sqrt(np.clip(np.diag(covariance), 0, None)) * np.sqrt(max(chi2r, 1.0))
    resid[rows] = r
    return {"x": x, "err": err, "nbl": nbl, "resid": resid, "chi2r": chi2r}


def _log_amplitude(cube: dict, coherent: Optional[bool]) -> tuple[np.ndarray, np.ndarray]:
    """``log`` amplitude per cell and hand, and its variance (from the noise of the bin average)."""
    amp = visstats.amplitude(cube, coherent=coherent)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = visstats.sigma(cube)
        snr = amp / sigma
        logamp = np.where(snr > 1.0, np.log(amp), np.nan)
        variance = 1.0 / np.maximum(snr, 1e-6) ** 2 + 1e-4           # 1% floor: no cell is perfect
    return logamp, variance


def gains_vs_time(cube: dict, *, fields: Optional[list] = None, coherent: Optional[bool] = None,
                  per_scan: bool = True, combine_spw: bool = True) -> dict:
    """Relative amplitude gain of every antenna over time, per field and hand (and subband if not combined).

    Each baseline/subband/hand series is divided by its own median over the field's scans, and every time slot
    (a scan with ``per_scan``, else a time bin) is then solved for one log-gain per antenna.

    Returns
    -------
    dict
        ``{field name: {"time": (nslot,), "scan": (nslot,), "x": (nslot, nant, nhand[, nspw]), "err": ...,
        "chi2r": (nslot, nhand[, nspw])}}``, ``x`` in natural-log units (0 = the antenna's median level).
    """
    logamp, variance = _log_amplitude(cube, coherent)
    nant = len(cube["antennas"])
    names = list(cube["field_names"])
    wanted = [names.index(f) for f in fields] if fields else sorted(set(cube["field"].tolist()))
    spws = np.unique(cube["spw"])
    nhand = logamp.shape[1]
    out = {}
    for fid in wanted:
        sel = cube["field"] == fid
        if not sel.any():
            continue
        slot_key = cube["scan"][sel] if per_scan else cube["scan"][sel] * visstats._MAX_BINS + cube["tbin"][sel]
        # Normalise each (baseline, spw, hand) by its median over the field.
        la = logamp[sel].copy()
        bl = cube["ant1"][sel] * nant + cube["ant2"][sel]
        series = bl * len(spws) + np.searchsorted(spws, cube["spw"][sel])
        for key in np.unique(series):
            rows = series == key
            for h in range(nhand):
                col = la[rows, h]
                if np.isfinite(col).sum() >= 3:
                    la[rows, h] = col - np.nanmedian(col)
                else:
                    la[rows, h] = np.nan
        var = variance[sel]
        slots, slot_index = np.unique(slot_key, return_inverse=True)
        nsp = 1 if combine_spw else len(spws)
        x = np.full((slots.size, nant, nhand, nsp), np.nan)
        err = np.full_like(x, np.nan)
        chi2r = np.full((slots.size, nhand, nsp), np.nan)
        times = np.zeros(slots.size)
        scans = np.zeros(slots.size, dtype=np.int64)
        a1, a2 = cube["ant1"][sel], cube["ant2"][sel]
        spw_index = np.searchsorted(spws, cube["spw"][sel])
        for s in range(slots.size):
            rows = slot_index == s
            times[s] = float(np.nanmean(cube["time"][sel][rows]))
            scans[s] = int(cube["scan"][sel][rows][0])
            for h in range(nhand):
                for k in range(nsp):
                    r = rows if combine_spw else rows & (spw_index == k)
                    if r.sum() < 3:
                        continue
                    if combine_spw:
                        # One value per baseline: the inverse-variance mean over the subbands and bins.
                        b = bl[r]
                        vals, wts = la[r, h], 1.0 / var[r, h]
                        good = np.isfinite(vals)
                        ub, inv = np.unique(b[good], return_inverse=True)
                        if ub.size < 3:
                            continue
                        wsum = np.bincount(inv, weights=wts[good])
                        vmean = np.bincount(inv, weights=wts[good] * vals[good]) / wsum
                        res = solve_antenna_terms(vmean, wsum, ub // nant, ub % nant, nant)
                    else:
                        res = solve_antenna_terms(la[r, h], 1.0 / var[r, h], a1[r], a2[r], nant)
                    x[s, :, h, k] = res["x"]
                    err[s, :, h, k] = res["err"]
                    chi2r[s, h, k] = res["chi2r"]
        if combine_spw:
            x, err, chi2r = x[..., 0], err[..., 0], chi2r[..., 0]
        out[names[fid]] = {"time": times, "scan": scans, "x": x, "err": err, "chi2r": chi2r}
    return out


def subband_offsets(cube: dict, *, fields: Optional[list] = None, coherent: Optional[bool] = None) -> dict:
    """Antenna-based amplitude offset of every subband from the antenna's own band median, per hand.

    Every baseline/hand/time-bin is divided by its median over the subbands it has, and the log-ratios of each
    subband are solved for one term per antenna over all the bins of the selected fields.

    Returns
    -------
    dict
        ``x`` (nspw, nant, nhand) natural-log offsets (0 = level with the rest of the band), ``err``, ``chi2r``
        (nspw, nhand), ``spws``.
    """
    logamp, variance = _log_amplitude(cube, coherent)
    nant = len(cube["antennas"])
    names = list(cube["field_names"])
    sel = np.isin(cube["field"], [names.index(f) for f in fields]) if fields else np.ones(cube["field"].size, bool)
    spws = np.unique(cube["spw"])
    nhand = logamp.shape[1]
    # Cells of one baseline and time bin, across subbands.
    group = ((cube["scan"] * nant * nant + cube["ant1"] * nant + cube["ant2"]) * visstats._MAX_BINS
             + cube["tbin"])[sel]
    la = logamp[sel]
    var = variance[sel]
    rel = np.full_like(la, np.nan)
    order = np.argsort(group, kind="stable")
    uniq, starts = np.unique(group[order], return_index=True)
    bounds = np.append(starts, order.size)
    for g in range(uniq.size):
        rows = order[bounds[g]:bounds[g + 1]]
        if rows.size < 2:
            continue
        for h in range(nhand):
            col = la[rows, h]
            if np.isfinite(col).sum() >= 2:
                rel[rows, h] = col - np.nanmedian(col)
    x = np.full((spws.size, nant, nhand), np.nan)
    err = np.full_like(x, np.nan)
    chi2r = np.full((spws.size, nhand), np.nan)
    a1, a2, sp = cube["ant1"][sel], cube["ant2"][sel], cube["spw"][sel]
    for k, spw in enumerate(spws):
        rows = sp == spw
        for h in range(nhand):
            vals, wts = rel[rows, h], 1.0 / var[rows, h]
            good = np.isfinite(vals)
            b = (a1[rows] * nant + a2[rows])[good]
            ub, inv = np.unique(b, return_inverse=True)
            if ub.size < 3:
                continue
            wsum = np.bincount(inv, weights=wts[good])
            vmean = np.bincount(inv, weights=wts[good] * vals[good]) / wsum
            res = solve_antenna_terms(vmean, wsum, ub // nant, ub % nant, nant)
            x[k, :, h], err[k, :, h], chi2r[k, h] = res["x"], res["err"], res["chi2r"]
    return {"x": x, "err": err, "chi2r": chi2r, "spws": spws}


def hand_ratios(cube: dict, *, fields: Optional[list] = None, coherent: Optional[bool] = None,
                per_spw: bool = True) -> dict:
    """Antenna-based ``log(|g_R| / |g_L|)``: the amplitude ratio of the two hands of every station.

    The calibrators are unpolarized in the circular basis, so on every baseline ``log|RR| - log|LL|`` is the sum
    of the two stations' hand ratios.

    Returns
    -------
    dict
        ``x`` (nspw or 1, nant) natural-log ratios R/L, ``err``, ``chi2r``, ``spws``.
    """
    logamp, variance = _log_amplitude(cube, coherent)
    if logamp.shape[1] < 2:
        raise ValueError("hand_ratios needs both parallel hands")
    nant = len(cube["antennas"])
    names = list(cube["field_names"])
    sel = np.isin(cube["field"], [names.index(f) for f in fields]) if fields else np.ones(cube["field"].size, bool)
    diff = (logamp[:, 0] - logamp[:, 1])[sel]
    var = (variance[:, 0] + variance[:, 1])[sel]
    spws = np.unique(cube["spw"]) if per_spw else np.array([-1])
    x = np.full((spws.size, nant), np.nan)
    err = np.full_like(x, np.nan)
    chi2r = np.full(spws.size, np.nan)
    a1, a2, sp = cube["ant1"][sel], cube["ant2"][sel], cube["spw"][sel]
    for k, spw in enumerate(spws):
        rows = (sp == spw) if per_spw else np.ones(sp.size, bool)
        good = rows & np.isfinite(diff)
        b = a1[good] * nant + a2[good]
        ub, inv = np.unique(b, return_inverse=True)
        if ub.size < 3:
            continue
        wsum = np.bincount(inv, weights=1.0 / var[good])
        vmean = np.bincount(inv, weights=diff[good] / var[good]) / wsum
        res = solve_antenna_terms(vmean, wsum, ub // nant, ub % nant, nant)
        x[k], err[k], chi2r[k] = res["x"], res["err"], res["chi2r"]
    return {"x": x, "err": err, "chi2r": chi2r, "spws": spws}
