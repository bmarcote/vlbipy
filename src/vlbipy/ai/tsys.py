"""Quality of the system temperatures, hand by hand, and whether the raw data agree with them.

The a-priori amplitude calibration multiplies every visibility by ``sqrt(SEFD_i SEFD_j)``, with
``SEFD = Tsys / (DPFU * gain curve)``. A Tsys measurement that is wrong therefore goes straight into the
calibrated amplitudes, and the usual failures are recognisable in the table itself:

* **nominal** - a constant value for the whole experiment (JB 80 K, T6 26 K in EB104G with DPFU = 1: an SEFD
  written as a Tsys). Gives the absolute scale but follows neither elevation nor weather; the amplitude
  self-calibration has to carry the time dependence;
* **noisy hand** - one receptor's measurement jumps from sample to sample far more than its partner's
  (SR and UR RCP in EB104G). Both hands see the same sky and the same atmosphere, so the time dependence of
  the steady hand, scaled to the noisy hand's median level, is the better estimate of both;
* **unphysical** - values far below the receiver temperature (EF LCP 3.3 K at the start of EB104G) or
  non-positive placeholders;
* **high subband** - one subband well above the antenna's other subbands (WB spw 6 RCP, spw 7 LCP). Real (RFI,
  a bad IF) or not, the raw amplitudes decide: a real increase lowers the correlation coefficient of every
  baseline of that antenna and subband by ``sqrt`` of the ratio (:func:`check_against_raw`);
* **trend** - a slow increase (low elevation, weather: YS in the last 1.5 h of EB104G), which is real when the
  raw amplitudes drop by the matching ``sqrt``.

:func:`read_syscal` reads the unsmoothed measurements from the measurement set (the pipeline smooths its
table in place), :func:`assess` classifies every series, and :func:`substitute_hand` writes the repair into a
Tsys calibration table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..statistics import mad_sigma

logger = get_logger()

#: A hand is "noisy" when its sample-to-sample scatter exceeds its partner's by this factor (and the floor).
NOISY_HAND_RATIO = 2.5
#: ... and its relative point-to-point scatter is above this fraction of its level (EB104G: UR RCP 1.6-2.5% in
#: spw 0-3 against 0.3-0.7% in LCP is noise the raw data do not show).
NOISY_FLOOR = 0.01
#: Relative scatter below this over the whole experiment: a nominal (constant) value.
CONSTANT_SCATTER = 1e-4
#: A subband whose median Tsys exceeds the antenna's median over subbands by this factor is "high".
HIGH_SUBBAND = 1.3
#: Values below this fraction of the series median are unphysical (a receiver is never that cold).
UNPHYSICAL_FRACTION = 0.3


def read_syscal(ms) -> dict:
    """The raw Tsys measurements of ``ms`` (its SYSCAL subtable), with the scan and field of every sample.

    Returns
    -------
    dict
        ``time`` (n,), ``antenna`` (n,), ``spw`` (n,), ``tsys`` (n, npol), ``scan`` (n,), ``field`` (n,) and
        ``antennas`` (names).
    """
    from ..solvers.apriori import scan_of_times
    from ..solvers.msio import open_table, read_layout, read_setup, table_engine
    engine = table_engine()
    handle = open_table(Path(ms) / "SYSCAL", engine=engine)
    try:
        out = {"time": np.asarray(handle.getcol("TIME"), dtype=np.float64),
               "antenna": np.asarray(handle.getcol("ANTENNA_ID"), dtype=np.int64),
               "spw": np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64),
               "tsys": np.asarray(handle.getcol("TSYS"), dtype=np.float64)}
    finally:
        handle.close()
    if out["tsys"].ndim == 1:
        out["tsys"] = out["tsys"][:, None]
    scan, field = scan_of_times(read_layout(ms, engine=engine), out["time"])
    out.update(scan=scan, field=field, antennas=np.array(read_setup(ms, engine=engine)["antenna_names"]))
    return out


def _point_scatter(series: np.ndarray) -> float:
    """Relative sample-to-sample scatter: robust sigma of first differences / sqrt(2), over the median level."""
    finite = series[np.isfinite(series) & (series > 0)]
    if finite.size < 5:
        return np.nan
    return float(mad_sigma(np.diff(finite)) / np.sqrt(2.0) / np.median(finite))


def assess(syscal: dict, *, hands: tuple = ("R", "L")) -> list[dict]:
    """Classify every (antenna, subband, hand) Tsys series.

    Returns
    -------
    list of dict
        One finding per series with ``antenna``, ``spw``, ``hand``, ``median`` [K], ``scatter`` (relative
        point-to-point), ``partner_scatter``, ``n``, ``unphysical`` (count of samples below
        ``UNPHYSICAL_FRACTION`` x median or non-positive), ``trend`` (log ratio of the last to the first 15%
        of the samples), ``level_vs_antenna`` (median over the antenna's subband medians, same hand), and
        ``labels``: a subset of ``{"nominal", "noisy_hand", "unphysical", "high_subband", "trend_up",
        "trend_down"}``.
    """
    findings = []
    names = list(syscal["antennas"])
    npol = syscal["tsys"].shape[1]
    for antenna in np.unique(syscal["antenna"]):
        rows_a = syscal["antenna"] == antenna
        spws = np.unique(syscal["spw"][rows_a])
        medians = np.full((spws.size, npol), np.nan)
        stats = {}
        for k, spw in enumerate(spws):
            rows = rows_a & (syscal["spw"] == spw)
            order = np.argsort(syscal["time"][rows])
            for p in range(npol):
                series = syscal["tsys"][rows, p][order]
                good = np.isfinite(series) & (series > 0)
                med = float(np.median(series[good])) if good.any() else np.nan
                medians[k, p] = med
                bad = (~good) | (series < UNPHYSICAL_FRACTION * med)
                clean = np.where(bad, np.nan, series)
                n15 = max(3, int(0.15 * clean.size))
                with np.errstate(invalid="ignore", divide="ignore"):
                    first, last = np.nanmedian(clean[:n15]), np.nanmedian(clean[-n15:])
                rel = np.nanstd(clean) / med if np.isfinite(med) and med > 0 else np.nan
                stats[(int(spw), p)] = {"median": med, "scatter": _point_scatter(clean), "n": int(good.sum()),
                                        "unphysical": int(bad.sum()), "rel_std": float(rel),
                                        "trend": float(np.log(last / first)) if first > 0 and last > 0 else np.nan}
        for k, spw in enumerate(spws):
            for p in range(npol):
                s = stats[(int(spw), p)]
                partner = stats.get((int(spw), 1 - p)) if npol == 2 else None
                level = float(np.nanmedian(medians[:, p]))
                labels = []
                if s["rel_std"] < CONSTANT_SCATTER:
                    labels.append("nominal")
                if partner is not None and np.isfinite(s["scatter"]) and np.isfinite(partner["scatter"]) \
                        and s["scatter"] > NOISY_FLOOR and s["scatter"] > NOISY_HAND_RATIO * max(partner["scatter"], 1e-3):
                    labels.append("noisy_hand")
                if s["unphysical"] > 0:
                    labels.append("unphysical")
                if spws.size > 1 and level > 0 and s["median"] > HIGH_SUBBAND * level:
                    labels.append("high_subband")
                if np.isfinite(s["trend"]) and abs(s["trend"]) > 0.15 and "nominal" not in labels:
                    labels.append("trend_up" if s["trend"] > 0 else "trend_down")
                findings.append({"antenna": names[int(antenna)], "spw": int(spw), "hand": hands[p] if p < len(hands)
                                 else str(p), **s, "partner_scatter": partner["scatter"] if partner else np.nan,
                                 "level_vs_antenna": s["median"] / level if level > 0 else np.nan, "labels": labels})
    return findings


def check_against_raw(findings: list[dict], raw_subband: dict, antennas: list[str], *, tolerance: float = 0.5) -> list[dict]:
    """Decide whether each "high_subband" Tsys is real, from the antenna-based subband offsets of the raw data.

    A real Tsys excess ``T / T_ref`` lowers the raw amplitude of that antenna and subband by
    ``sqrt(T / T_ref)``: the expected log offset is ``-0.5 log(T / T_ref)``. ``raw_subband`` is
    :func:`vlbipy.ai.antenna.subband_offsets` of the raw cube (hands ordered R, L).

    The verdict is written into each finding as ``raw_offset`` (measured), ``expected_offset`` and ``tsys_real``
    (True when the measured offset is at least ``tolerance`` of the expected one, False when it is near zero,
    None when unknown).
    """
    spws = list(raw_subband["spws"])
    for finding in findings:
        if "high_subband" not in finding["labels"]:
            continue
        try:
            a = antennas.index(finding["antenna"])
            k = spws.index(finding["spw"])
        except ValueError:
            continue
        h = 0 if finding["hand"] == "R" else 1
        measured = float(raw_subband["x"][k, a, h])
        expected = -0.5 * float(np.log(finding["level_vs_antenna"]))
        finding["raw_offset"], finding["expected_offset"] = measured, expected
        finding["tsys_real"] = None if not np.isfinite(measured) else bool(measured / expected >= tolerance)
    return findings


def check_trend_against_raw(syscal: dict, raw_gains: dict, antenna: str, *, hand: int = 1, spw: Optional[int] = None) -> dict:
    """Compare an antenna's Tsys evolution with its raw amplitude gain over time (one field's series).

    ``raw_gains`` is one field's entry of :func:`vlbipy.ai.antenna.gains_vs_time` on the raw cube. A real Tsys
    change ``T(t)`` moves the raw log-amplitude by ``-0.5 log T(t)``; the returned ``slope`` of measured
    against expected is ~1 when the Tsys explains the amplitude, ~0 when it does not.

    Returns
    -------
    dict
        ``slope``, ``corr`` (correlation coefficient), ``n`` (scans compared), ``expected`` and ``measured``
        arrays (per scan, median-subtracted).
    """
    a = list(syscal["antennas"]).index(antenna)
    rows = syscal["antenna"] == a
    if spw is not None:
        rows &= syscal["spw"] == spw
    measured, expected = [], []
    for s, scan in enumerate(raw_gains["scan"]):
        x = raw_gains["x"][s, a, hand]
        sel = rows & (syscal["scan"] == scan)
        values = syscal["tsys"][sel, hand]
        values = values[np.isfinite(values) & (values > 0)]
        if not np.isfinite(x) or values.size == 0:
            continue
        measured.append(x)
        expected.append(-0.5 * np.log(np.median(values)))
    measured, expected = np.array(measured), np.array(expected)
    if measured.size < 5:
        return {"slope": np.nan, "corr": np.nan, "fast_slope": np.nan, "fast_corr": np.nan, "fast_rms": np.nan,
                "n": int(measured.size), "expected": expected, "measured": measured}
    measured -= np.median(measured)
    expected -= np.median(expected)
    slope, corr = _regress(expected, measured)
    # The scan-to-scan part only (a 5-scan running median removed from both): what a noisy measurement adds on
    # top of the slow, elevation-driven trend that the gain curve may also be following.
    from scipy.ndimage import median_filter
    fast_e = expected - median_filter(expected, 5, mode="nearest")
    fast_m = measured - median_filter(measured, 5, mode="nearest")
    fast_slope, fast_corr = _regress(fast_e, fast_m)
    return {"slope": slope, "corr": corr, "fast_slope": fast_slope, "fast_corr": fast_corr,
            "fast_rms": float(np.std(fast_e)), "n": int(measured.size), "expected": expected, "measured": measured}


def _regress(expected: np.ndarray, measured: np.ndarray) -> tuple[float, float]:
    """Slope of ``measured`` on ``expected`` through the origin, and their correlation coefficient."""
    denom = float(np.sum(expected ** 2))
    slope = float(np.sum(expected * measured) / denom) if denom > 0 else np.nan
    corr = float(np.corrcoef(expected, measured)[0, 1]) if np.std(expected) > 0 and np.std(measured) > 0 else np.nan
    return slope, corr


def substitute_hand(caltable, substitutions: dict[str, dict[int, int]], *, antennas: Optional[list[str]] = None) -> int:
    """Replace the Tsys of a noisy hand by the steady hand's, scaled to the noisy hand's median level.

    Parameters
    ----------
    caltable : path
        A ``B TSYS`` calibration table (FPARAM ``(nrow, 1, npol)``), modified in place.
    substitutions : dict
        ``{antenna name: {spw: source hand index}}``: in each listed subband, the *other* hand takes the time
        dependence of ``source`` (``0`` = first hand, ``1`` = second). ``spw = -1`` stands for every subband.

    Returns
    -------
    int
        Number of rows changed.
    """
    from ..solvers.msio import open_table
    path = Path(caltable)
    names = antennas or [str(n) for n in _table_column(path / "ANTENNA", "NAME")]
    handle = open_table(path, readonly=False)
    changed = 0
    try:
        fparam = np.asarray(handle.getcol("FPARAM"), dtype=np.float64)
        flags = np.asarray(handle.getcol("FLAG"), dtype=bool)
        ant = np.asarray(handle.getcol("ANTENNA1"))
        spw = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
        for name, per_spw in substitutions.items():
            if name not in names:
                continue
            a = names.index(name)
            for target_spw, source in per_spw.items():
                spw_rows = np.unique(spw[ant == a]) if int(target_spw) < 0 else [int(target_spw)]
                for s in spw_rows:
                    rows = (ant == a) & (spw == s)
                    other = 1 - int(source)
                    src = np.where(flags[rows, 0, source], np.nan, fparam[rows, 0, source])
                    dst = np.where(flags[rows, 0, other], np.nan, fparam[rows, 0, other])
                    ok = np.isfinite(src) & np.isfinite(dst) & (src > 0) & (dst > 0)
                    if ok.sum() < 3:
                        continue
                    scale = float(np.median(dst[ok] / src[ok]))
                    new = src * scale
                    valid = np.isfinite(new) & (new > 0)
                    idx = np.flatnonzero(rows)[valid]
                    fparam[idx, 0, other] = new[valid]
                    flags[idx, 0, other] = False
                    changed += int(valid.sum())
                    logger.info("tsys: {} spw {}: hand {} now follows hand {} (x{:.3f})", name, s, other, source, scale)
        if changed:
            handle.putcol("FPARAM", fparam.astype(np.float32) if handle.getcol("FPARAM").dtype == np.float32 else fparam)
            handle.putcol("FLAG", flags)
    finally:
        handle.close()
    return changed


def _table_column(path: Path, column: str) -> np.ndarray:
    from ..solvers.msio import open_table
    handle = open_table(path)
    try:
        return np.asarray(handle.getcol(column))
    finally:
        handle.close()


#: Smoothing windows (s): measurement noise the raw data do not show, the default, and a real fast variation.
SMOOTH_LONG, SMOOTH_DEFAULT, SMOOTH_SHORT = 1800.0, 600.0, 300.0
#: Largest exponent the opacity-like correction may give a Tsys series (Tsys_eff = Tsys (Tsys / median)^(e - 1)).
EXPONENT_MAX = 3.0
#: ... and the largest factor that correction may apply to any Tsys value, either way.
OPACITY_FACTOR_MAX = 3.0
#: The source must raise the Tsys by more than this (log) on average for fields to be smoothed apart.
FIELD_STEP = 0.02
#: ... and the step must be this many standard errors from zero.
FIELD_SIGMA = 4.0


def field_steps(syscal: dict, antenna: int, spw: int, hand: int) -> float:
    """Typical Tsys step at a change of field (median |log ratio| of the per-scan medians across field changes).

    A bright source adds to the system temperature, so its scans sit above the neighbouring target scans; a faint
    one does not, and then the fields are one series.
    """
    rows = (syscal["antenna"] == antenna) & (syscal["spw"] == spw)
    scans = syscal["scan"][rows]
    values = syscal["tsys"][rows, hand]
    good = np.isfinite(values) & (values > 0) & (scans >= 0)
    if good.sum() < 10:
        return 0.0
    numbers = np.unique(scans[good])
    level = np.array([np.median(values[good & (scans == n)]) for n in numbers])
    fields = np.array([syscal["field"][rows][good & (scans == n)][0] for n in numbers])
    change = fields[1:] != fields[:-1]
    if not change.any():
        return 0.0
    steps = np.log(level[1:][change] / level[:-1][change])
    # A trend is common to both directions of the change; the source term flips sign with it.
    pairs = {}
    for f0, f1, step in zip(fields[:-1][change], fields[1:][change], steps):
        key = (min(f0, f1), max(f0, f1))
        pairs.setdefault(key, []).append(step if f0 < f1 else -step)
    significant = []
    for v in pairs.values():
        if len(v) < 3:
            continue
        v = np.asarray(v)
        median = float(np.median(v))
        error = float(mad_sigma(v)) / np.sqrt(v.size)
        if abs(median) > FIELD_SIGMA * error:          # a noisy series has noisy steps, not a source term
            significant.append(abs(median))
    return float(max(significant, default=0.0))


def plan(syscal: dict, findings: list[dict], trend_checks: Optional[dict] = None) -> dict:
    """Decide the repair of every (antenna, spw, hand) Tsys series from its findings and the raw-data checks.

    Parameters
    ----------
    syscal : dict
        :func:`read_syscal`.
    findings : list of dict
        :func:`assess` (after :func:`check_against_raw`, so ``high_subband`` findings carry ``tsys_real``).
    trend_checks : dict, optional
        ``{(antenna, spw, hand): check_trend_against_raw(...)}``.

    Returns
    -------
    dict
        ``{(antenna name, spw, hand index): {"action": ..., "window": s, "combine_fields": bool, "source_hand":
        int (substitute), "scale": float (rescale), "reason": str}}``.
    """
    from . import knowledge
    names = list(syscal["antennas"])
    trend_checks = trend_checks or {}
    decisions = {}
    for f in findings:
        hand = 0 if f["hand"] == "R" else 1
        key = (f["antenna"], f["spw"], hand)
        check = trend_checks.get(key, {})
        feature = dict(f, **{k: check.get(k) for k in ("slope", "corr", "fast_slope", "fast_corr", "fast_rms")})
        fired = knowledge.label(feature, "tsys")
        labels = {r["label"] for r in fired}
        antenna = names.index(f["antenna"])
        step = field_steps(syscal, antenna, f["spw"], hand)
        decision = {"action": "keep", "window": SMOOTH_DEFAULT, "combine_fields": step < FIELD_STEP,
                    "field_step": step, "rules": [r["rule"] for r in fired], "reason": ""}
        if "nominal_tsys" in labels:
            decision.update(action="keep", window=0.0, reason="nominal value: nothing to smooth")
        elif "noisy_tsys_hand" in labels:
            decision.update(action="substitute", source_hand=1 - hand, window=SMOOTH_DEFAULT,
                            reason=f"noisy hand (scatter {f['scatter']:.3f} vs {f['partner_scatter']:.3f})")
        elif "tsys_variation_real" in labels or "tsys_undercorrects" in labels:
            decision.update(window=SMOOTH_SHORT if (check.get("fast_slope") or 0) > 0.3 else SMOOTH_DEFAULT,
                            reason=f"real variation (slope {check.get('slope', np.nan):.2f})")
            if "tsys_undercorrects" in labels:
                exponent = float(np.clip(check["slope"], 1.0, EXPONENT_MAX))
                decision.update(exponent=exponent, reason=decision["reason"] + f"; the raw amplitude moves as "
                                f"Tsys^-{0.5 * exponent:.2f}: absorption the ground-level Tsys misses -> Tsys^{exponent:.2f}")
        elif "tsys_fluctuation_not_in_data" in labels:
            decision.update(window=SMOOTH_LONG, reason=f"fluctuations not in the raw data "
                                                       f"(fast slope {check.get('fast_slope', np.nan):.2f})")
        if "spurious_high_tsys" in labels and f.get("raw_offset") is not None:
            scale = float(np.exp(-2.0 * f["raw_offset"]) / f["level_vs_antenna"])
            decision.update(scale=scale, reason=(decision["reason"] + "; " if decision["reason"] else "")
                            + f"spurious subband excess x{f['level_vs_antenna']:.2f} (raw {100 * f['raw_offset']:+.0f}%)")
            if decision["action"] == "keep":
                decision["action"] = "rescale"
        decisions[key] = decision
    for key, decision in decisions.items():
        if decision["action"] == "substitute":         # the series is the partner's now, and so is its field test
            partner = decisions.get((key[0], key[1], decision["source_hand"]), {})
            decision["combine_fields"] = partner.get("combine_fields", True)
            decision["window"] = partner.get("window", decision["window"]) or decision["window"]
    return decisions


def repair_table(caltable, decisions: dict, *, despike_sigma: float = 6.0) -> dict:
    """Apply a :func:`plan` to a ``B TSYS`` table in place: substitute, rescale, de-spike and smooth.

    Order per antenna/subband: unphysical values are flagged; noisy hands take the partner's time dependence
    (scaled to their own median); spurious subband levels are rescaled; every series is de-spiked against its
    running median and smoothed with its own window, over all fields together unless the source measurably
    raises the Tsys (then field by field).

    Returns
    -------
    dict
        Counts: ``substituted``, ``rescaled``, ``flagged``, ``smoothed`` (values).
    """
    from ..solvers.msio import open_table
    from ..statistics import despike, smooth_in_time
    path = Path(caltable)
    names = [str(n) for n in _table_column(path / "ANTENNA", "NAME")]
    handle = open_table(path, readonly=False)
    counts = {"substituted": 0, "rescaled": 0, "flagged": 0, "smoothed": 0}
    try:
        original = handle.getcol("FPARAM")
        fparam = np.asarray(original, dtype=np.float64).copy()
        flags = np.asarray(handle.getcol("FLAG"), dtype=bool).copy()
        ant = np.asarray(handle.getcol("ANTENNA1"))
        spw = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
        fld = np.asarray(handle.getcol("FIELD_ID"))
        times = np.asarray(handle.getcol("TIME"), dtype=np.float64)
        npol = fparam.shape[2]
        for a in np.unique(ant):
            name = names[int(a)]
            for s in np.unique(spw[ant == a]):
                rows = np.flatnonzero((ant == a) & (spw == s))
                rows = rows[np.argsort(times[rows], kind="stable")]
                # 1. unphysical
                for p in range(npol):
                    v = fparam[rows, 0, p]
                    good = ~flags[rows, 0, p] & (v > 0)
                    if good.sum():
                        bad = good & (v < UNPHYSICAL_FRACTION * np.median(v[good]))
                        flags[rows[bad], 0, p] = True
                        counts["flagged"] += int(bad.sum())
                # 2. substitution
                for p in range(npol):
                    d = decisions.get((name, int(s), p), {})
                    if d.get("action") != "substitute" or npol < 2:
                        continue
                    src = d["source_hand"]
                    sv = np.where(flags[rows, 0, src], np.nan, fparam[rows, 0, src])
                    dv = np.where(flags[rows, 0, p], np.nan, fparam[rows, 0, p])
                    ok = np.isfinite(sv) & np.isfinite(dv) & (sv > 0) & (dv > 0)
                    if ok.sum() < 3:
                        continue
                    scale = float(np.median(dv[ok]) / np.median(sv[ok]))
                    new = sv * scale
                    valid = np.isfinite(new)
                    fparam[rows[valid], 0, p] = new[valid]
                    flags[rows[valid], 0, p] = False
                    counts["substituted"] += int(valid.sum())
                # 3. rescale
                for p in range(npol):
                    d = decisions.get((name, int(s), p), {})
                    if "scale" in d and np.isfinite(d["scale"]) and d["scale"] > 0:
                        fparam[rows, 0, p] *= d["scale"]
                        counts["rescaled"] += rows.size
                # 4. de-spike and smooth
                for p in range(npol):
                    d = decisions.get((name, int(s), p), {"window": SMOOTH_DEFAULT, "combine_fields": True})
                    window = float(d.get("window", SMOOTH_DEFAULT))
                    groups = [rows] if d.get("combine_fields", True) else \
                        [rows[fld[rows] == f] for f in np.unique(fld[rows])]
                    for g in groups:
                        if g.size < 3:
                            continue
                        series = np.where(flags[g, 0, p], np.nan, fparam[g, 0, p])
                        if np.isfinite(series).sum() < 3:
                            continue
                        cleaned, _ = despike(series, threshold=despike_sigma, max_passes=3, window=5)
                        if window > 0:
                            smoothed, changed = smooth_in_time(cleaned, times[g], window=window, min_points=3)
                        else:
                            smoothed, changed = cleaned, np.zeros(cleaned.shape, bool)
                        ok = np.isfinite(smoothed)
                        fparam[g[ok], 0, p] = smoothed[ok]
                        counts["smoothed"] += int(changed.sum())
                    exponent = float(d.get("exponent", 1.0))
                    if exponent != 1.0:
                        # The raw amplitudes follow Tsys^(-exponent/2): an absorbing atmosphere attenuates the
                        # source as it raises the Tsys, which a Tsys measured at the ground does not include. The
                        # extra factor comes from a 10-minute running median (a power law magnifies any spike the
                        # de-spiking missed) and is capped at OPACITY_FACTOR_MAX either way.
                        v = np.where(flags[rows, 0, p], np.nan, fparam[rows, 0, p])
                        level = np.nanmedian(v)
                        if np.isfinite(level) and level > 0:
                            slow, _ = smooth_in_time(v, times[rows], window=SMOOTH_DEFAULT, min_points=3)
                            ok = np.isfinite(v) & (v > 0) & np.isfinite(slow) & (slow > 0)
                            factor = np.clip((slow[ok] / level) ** (exponent - 1.0), 1.0 / OPACITY_FACTOR_MAX,
                                             OPACITY_FACTOR_MAX)
                            fparam[rows[ok], 0, p] = v[ok] * factor
                            counts["opacity"] = counts.get("opacity", 0) + int(ok.sum())
        handle.putcol("FPARAM", fparam.astype(np.asarray(original).dtype))
        handle.putcol("FLAG", flags)
    finally:
        handle.close()
    counts.setdefault("opacity", 0)
    logger.info("tsys repair: {substituted} substituted, {rescaled} rescaled, {flagged} flagged, {smoothed} smoothed, "
                "{opacity} opacity-corrected value(s)", **counts)
    return counts
