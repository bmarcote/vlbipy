"""Do the calibrated visibilities follow the distribution they must follow? Find, catalogue and flag what does not.

After calibration, the visibility of a baseline is the source's coherent signal plus thermal noise. The signal
``M`` changes only slowly - the source structure seen along the uv track, an elevation-dependent attenuation -
while the noise is complex Gaussian with a variance known from the data themselves (the scatter between consecutive
integrations, with the correlation of adjacent channels, :mod:`vlbipy.ai.visstats`). Hence, for every channel-averaged time bin,

    ``z = (V - M) / sigma``  is a standard complex normal variable:

* ``|V|`` is Rice-distributed around ``|M|`` (Rayleigh when ``|M| << sigma``, Gaussian of width sigma when
  ``|M| >> sigma``), and ``arg(V / M)`` is von Mises-like, concentrated for high SNR and uniform for low SNR;
* ``|z|^2`` is exponential with mean 1, so a single bin with ``|z|^2 > -ln(p)`` happens with probability ``p``;
* the mean of ``|z|^2`` over a scan (its reduced chi-squared) is 1 within ``1/sqrt(n)``;
* the amplitude is the same in every subband: VLBI bandwidths are far too narrow for a spectral index.

What departs from that is bad data (RFI, a phase jump, a receiver dropping out, a wrong Tsys) - or, for the
slow components, real source structure, which is why ``M`` is estimated per baseline along time and never
across baselines. :func:`assess` returns the per-bin, per-scan and per-subband findings;
:func:`flag_commands` turns the bad ones into ``flagdata`` list commands.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from . import visstats

#: Probability below which a single bin is an outlier (|z|^2 > -ln P_BIN ~ 16.1 for 1e-7).
P_BIN = 1e-7
#: Scans whose reduced chi-squared exceeds 1 by this many standard errors are flagged as "excess scatter".
SCAN_SIGMA = 6.0
#: ... and by at least this factor (keeps the test meaningful at very high SNR, where systematics dominate).
SCAN_FACTOR = 2.0
#: A subband whose amplitude departs from the band median by more than this many sigma and fraction.
SPW_SIGMA, SPW_FRACTION = 6.0, 0.15
#: Half-width (bins) of the running complex median that estimates the signal within a scan.
MODEL_HALF = 3
#: A bin is an outlier only if it also departs from the signal by this fraction of it (or the signal is weak,
#: ``|M| < 5 sigma``): a 3-degree wobble is many sigma at SNR 100 and harmless.
MIN_DEVIATION = 0.1
#: A scan of a high-SNR group is bad when its own fractional jitter exceeds this (~12 deg rms of phase or 20% of
#: amplitude): phase jumps, a receiver dropping in and out. Low-SNR groups use the reduced chi-squared instead.
SCAN_JITTER_MAX = 0.2


def _running_complex_median(values: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Running median of the real and imaginary parts (nan-aware), window ``2 * half + 1`` without the point itself.

    Returns the medians and how many points each one used.
    """
    n = values.size
    out = np.full(n, np.nan + 0j)
    used = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = np.r_[values[lo:i], values[i + 1:hi]]          # leave the point itself out
        window = window[np.isfinite(window)]
        if window.size >= 2:
            out[i] = np.median(window.real) + 1j * np.median(window.imag)
            used[i] = window.size
    return out, used


def _fit_jitter(r2: np.ndarray, base: np.ndarray, m2: np.ndarray) -> float:
    """The fractional jitter ``f`` for which ``r2 / (base + f^2 m2)`` has the median of an exponential law (ln 2).

    ``r2`` are squared residuals, ``base`` their thermal variance, ``m2`` the squared signal. 0 when the thermal
    noise alone already explains them.
    """
    ok = np.isfinite(r2) & np.isfinite(base) & np.isfinite(m2) & (base > 0)
    if ok.sum() < 5:
        return 0.0
    r2, base, m2 = r2[ok], base[ok], m2[ok]
    target = np.log(2.0)
    if np.median(r2 / base) <= target:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.median(r2 / (base + mid ** 2 * m2)) > target:
            lo = mid
        else:
            hi = mid
    return hi


def residuals(cube: dict, *, fields: Optional[list] = None, half: int = MODEL_HALF) -> dict:
    """Signal model and normalised residuals of every cell of a calibrated cube.

    The expected scatter of a bin about the model is ``sigma^2 (1 + pi / 2k) + f^2 |M|^2``: the thermal noise of
    the bin, the noise of a median of ``k`` neighbours, and a fractional jitter ``f`` - the atmospheric and
    instrumental fluctuation that dominates once the SNR of a bin is high (1-2 degrees of phase are 3-5 thermal
    sigma at SNR 100) - fitted per baseline, hand and field so that the median of ``|z|^2`` is that of an
    exponential law. Bad data are then what departs from the baseline's own typical behaviour.

    Returns
    -------
    dict
        ``index`` (cells used), ``model`` (n, nhand) complex, ``sigma`` (n, nhand) expected scatter, ``z`` (n, nhand)
        complex, and ``jitter``: ``{(field, ant1, ant2, hand): f}``.
    """
    names = list(cube["field_names"])
    sel = np.isin(cube["field"], [names.index(f) for f in fields]) if fields else np.ones(cube["field"].size, bool)
    index = np.flatnonzero(sel)
    vec = cube["vec"][index]
    thermal = visstats.sigma(cube)[index]
    model = np.full(vec.shape, np.nan + 0j)
    used = np.zeros(vec.shape, dtype=np.int64)
    nant = len(cube["antennas"])
    key = ((cube["scan"][index] * nant * nant + cube["ant1"][index] * nant + cube["ant2"][index]) * 64
           + cube["spw"][index])
    order = np.lexsort((cube["tbin"][index], key))
    uniq, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    # Every group padded into one row of a 2-D array, so the leave-one-out running median of all of them is one
    # nanmedian over the stacked neighbours (the per-group loop took minutes on a full experiment).
    width = int(counts.max())
    position = np.arange(order.size) - np.repeat(starts, counts)
    group_of = np.repeat(np.arange(uniq.size), counts)
    for h in range(vec.shape[1]):
        grid = np.full((uniq.size, width + 2 * half), complex(np.nan, np.nan))   # nan in both parts
        values = vec[order, h]
        grid[group_of, position + half] = np.where(np.isfinite(values), values, complex(np.nan, np.nan))
        shifts = [grid[:, half + k: half + k + width] for k in range(-half, half + 1) if k != 0]
        stack = np.stack(shifts, axis=2)                          # (ngroup, width, 2 * half)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack.real, axis=2) + 1j * np.nanmedian(stack.imag, axis=2)
        n_used = np.isfinite(stack).sum(axis=2)
        med[n_used < 2] = np.nan + 0j
        model[order, h] = med[group_of, position]
        used[order, h] = n_used[group_of, position]
    with np.errstate(invalid="ignore", divide="ignore"):
        base = thermal ** 2 * (1.0 + np.pi / (2.0 * np.maximum(used, 1)))
        r2 = np.abs(vec - model) ** 2
        m2 = np.abs(model) ** 2
    expected = np.full(vec.shape, np.nan)
    jitter = {}
    group = (cube["field"][index] * nant + cube["ant1"][index]) * nant + cube["ant2"][index]
    for g in np.unique(group):
        rows = group == g
        fid, rest = divmod(int(g), nant * nant)
        a1, a2 = divmod(rest, nant)
        for h in range(vec.shape[1]):
            f = _fit_jitter(r2[rows, h], base[rows, h], m2[rows, h])
            jitter[(fid, a1, a2, h)] = f
            expected[rows, h] = np.sqrt(base[rows, h] + f ** 2 * m2[rows, h])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (vec - model) / expected
    return {"index": index, "model": model, "sigma": expected, "thermal": thermal, "z": z, "jitter": jitter}


def assess(cube: dict, *, fields: Optional[list] = None, p_bin: float = P_BIN) -> dict:
    """Test every baseline of a calibrated cube against the noise distribution.

    Returns
    -------
    dict
        ``bins``: list of outlier cells (``field``, ``scan``, ``ant1``, ``ant2``, ``spw``, ``hand``, ``time``,
        ``z2``); ``scans``: per (scan, baseline, spw, hand) groups with excess scatter (``chi2r``, ``n``);
        ``spws``: per (scan, baseline, hand) subbands off the band median (``ratio``, ``nsigma``);
        ``baselines``: per-baseline summary (``chi2r``, ``frac_out``, ``snr``, ``ks``: KS distance of ``|z|^2``
        from the exponential law) - the features the model learns from.
    """
    res = residuals(cube, fields=fields)
    idx, z = res["index"], res["z"]
    z2 = np.abs(z) ** 2
    threshold = -np.log(p_bin)
    nant = len(cube["antennas"])
    out = {"bins": [], "scans": [], "spws": [], "baselines": []}
    a1, a2 = cube["ant1"][idx], cube["ant2"][idx]
    scan, spw, field, time = cube["scan"][idx], cube["spw"][idx], cube["field"][idx], cube["time"][idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = np.abs(cube["vec"][idx] - res["model"]) / np.abs(res["model"])
        weak = np.abs(res["model"]) < 5.0 * res["thermal"]
    for h in range(z.shape[1]):
        bad = np.isfinite(z2[:, h]) & (z2[:, h] > threshold) & ((deviation[:, h] > MIN_DEVIATION) | weak[:, h])
        for k in np.flatnonzero(bad):
            out["bins"].append({"field": int(field[k]), "scan": int(scan[k]), "ant1": int(a1[k]), "ant2": int(a2[k]),
                                "spw": int(spw[k]), "hand": h, "time": float(time[k]), "z2": float(z2[k, h])})
    # Per scan/baseline/spw/hand: is the scan worse than the noise (low SNR) or than any usable data (high SNR)?
    group = ((scan * nant * nant + a1 * nant + a2) * 64 + spw)
    vec = cube["vec"][idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_all = np.abs(vec - res["model"]) ** 2
        thermal2 = res["thermal"] ** 2 * (1.0 + np.pi / 6.0)
        m2_all = np.abs(res["model"]) ** 2
    for h in range(z.shape[1]):
        good = np.isfinite(z2[:, h])
        g, inv = np.unique(group[good], return_inverse=True)
        n = np.bincount(inv)
        chi2r = np.bincount(inv, weights=z2[good, h]) / np.maximum(n, 1)
        rows_of = np.flatnonzero(good)
        order = np.argsort(inv, kind="stable")
        bounds = np.append(np.searchsorted(inv[order], np.arange(g.size)), order.size)
        for k in range(g.size):
            if n[k] < 4:
                continue
            members = rows_of[order[bounds[k]:bounds[k + 1]]]
            snr_g = np.nanmedian(np.sqrt(m2_all[members, h] / thermal2[members, h]))
            if snr_g >= 5.0:
                jitter = _fit_jitter(r2_all[members, h], thermal2[members, h], m2_all[members, h])
                bad = jitter > SCAN_JITTER_MAX
            else:
                jitter = np.nan
                bad = chi2r[k] > 1 + SCAN_SIGMA / np.sqrt(n[k]) and chi2r[k] > SCAN_FACTOR
            if bad:
                s, rest = divmod(int(g[k]), nant * nant * 64)
                bl, sp = divmod(rest, 64)
                out["scans"].append({"scan": s, "ant1": bl // nant, "ant2": bl % nant, "spw": sp, "hand": h,
                                     "chi2r": float(chi2r[k]), "n": int(n[k]), "jitter": float(jitter),
                                     "snr": float(snr_g)})
    # Subbands of one scan/baseline/hand against the band median.
    amp = np.abs(cube["vec"][idx])
    sig = res["sigma"]                       # thermal + jitter: a subband test at the level the data scatter
    sgroup = scan * nant * nant + a1 * nant + a2
    for h in range(z.shape[1]):
        good = np.isfinite(amp[:, h]) & np.isfinite(sig[:, h])
        keys = sgroup[good] * 64 + spw[good]
        u, inv = np.unique(keys, return_inverse=True)
        wsum = np.bincount(inv, weights=1 / sig[good, h] ** 2)
        mean = np.bincount(inv, weights=amp[good, h] / sig[good, h] ** 2) / wsum
        err = 1 / np.sqrt(wsum)
        sg, sp = np.divmod(u, 64)
        for s in np.unique(sg):
            rows = sg == s
            if rows.sum() < 3:
                continue
            med = np.median(mean[rows])
            if med <= 0:
                continue
            for k in np.flatnonzero(rows):
                ratio = mean[k] / med
                nsig = (mean[k] - med) / err[k]
                if abs(nsig) > SPW_SIGMA and abs(ratio - 1) > SPW_FRACTION:
                    scan_no, bl = divmod(int(s), nant * nant)
                    out["spws"].append({"scan": scan_no, "ant1": bl // nant, "ant2": bl % nant, "spw": int(sp[k]),
                                        "hand": h, "ratio": float(ratio), "nsigma": float(nsig)})
    # Baseline summaries.
    bkey = a1 * nant + a2
    for b in np.unique(bkey):
        rows = bkey == b
        for h in range(z.shape[1]):
            v = z2[rows, h]
            v = v[np.isfinite(v)]
            if v.size < 10:
                continue
            srt = np.sort(v)
            ks = float(np.max(np.abs(np.arange(1, v.size + 1) / v.size - (1 - np.exp(-srt)))))
            with np.errstate(invalid="ignore", divide="ignore"):
                snr_b = np.nanmedian(np.abs(res["model"][rows, h]) / res["thermal"][rows, h])
            jit = [f for (fid, x1, x2, hh), f in res["jitter"].items() if x1 == b // nant and x2 == b % nant and hh == h]
            out["baselines"].append({"ant1": int(b // nant), "ant2": int(b % nant), "hand": h,
                                     "chi2r": float(np.mean(v)), "frac_out": float(np.mean(v > threshold)),
                                     "snr": float(snr_b), "ks": ks, "n": int(v.size),
                                     "jitter": float(np.max(jit)) if jit else 0.0})
    return out


def amplitude_departures(cube: dict, *, fields: Optional[list] = None, window: int = 7, nsigma: float = 5.0,
                         fraction: float = 0.2) -> list[dict]:
    """Scans where a baseline's amplitude departs from its own slow trend (a drop no source model explains).

    The per-scan amplitude of every baseline/hand (all subbands) is compared with a running median over
    ``window`` scans of the same field; source structure changes slowly along the uv track, a failing
    receiver or a wrong calibration changes from one scan to the next.
    """
    amp = visstats.amplitude(cube, coherent=True)
    sigma = visstats.sigma(cube)
    names = list(cube["field_names"])
    sel = np.isin(cube["field"], [names.index(f) for f in fields]) if fields else np.ones(cube["field"].size, bool)
    nant = len(cube["antennas"])
    out = []
    bkey = cube["ant1"] * nant + cube["ant2"]
    for fid in np.unique(cube["field"][sel]):
        fsel = sel & (cube["field"] == fid)
        for b in np.unique(bkey[fsel]):
            rows = fsel & (bkey == b)
            scans = np.unique(cube["scan"][rows])
            if scans.size < window:
                continue
            for h in range(amp.shape[1]):
                per = np.full(scans.size, np.nan)
                err = np.full(scans.size, np.nan)
                for i, s in enumerate(scans):
                    r = rows & (cube["scan"] == s)
                    a, e = amp[r, h], sigma[r, h]
                    ok = np.isfinite(a) & np.isfinite(e) & (e > 0)
                    if ok.sum():
                        w = 1 / e[ok] ** 2
                        per[i] = np.sum(w * a[ok]) / np.sum(w)
                        err[i] = 1 / np.sqrt(np.sum(w))
                half = window // 2
                for i in range(scans.size):
                    lo, hi = max(0, i - half), min(scans.size, i + half + 1)
                    ref = np.r_[per[lo:i], per[i + 1:hi]]
                    ref = ref[np.isfinite(ref)]
                    if ref.size < 3 or not np.isfinite(per[i]):
                        continue
                    med = np.median(ref)
                    spread = max(1.4826 * np.median(np.abs(ref - med)), err[i])
                    dev = (per[i] - med) / spread
                    if abs(dev) > nsigma and abs(per[i] / med - 1) > fraction:
                        out.append({"field": int(fid), "scan": int(scans[i]), "ant1": int(b // nant),
                                    "ant2": int(b % nant), "hand": h, "ratio": float(per[i] / med), "nsigma": float(dev)})
    return out


def flag_commands(findings: dict, cube: dict, *, departures: Optional[list] = None, binsec: Optional[float] = None,
                  min_bins_per_scan: int = 3) -> list[str]:
    """``flagdata`` list commands for the bad data found by :func:`assess` (and :func:`amplitude_departures`).

    Isolated outlier bins are flagged over their own time range; a scan/baseline/subband/hand with excess scatter
    (or with ``min_bins_per_scan`` outlier bins) as a whole scan; an off-level subband for its scan; an amplitude
    departure for the whole baseline and scan. Correlations are named (``RR``/``LL``), so one bad hand does not
    take the other with it.
    """
    names = [str(n) for n in cube["antennas"]]
    hands = [str(h) for h in cube["hands"]]
    binsec = float(binsec or cube["binsec"])
    commands: list[str] = []
    per_scan: dict = {}
    for b in findings["bins"]:
        key = (b["scan"], b["ant1"], b["ant2"], b["spw"], b["hand"])
        per_scan.setdefault(key, []).append(b)
    whole = {(s["scan"], s["ant1"], s["ant2"], s["spw"], s["hand"]) for s in findings["scans"]}
    whole |= {k for k, v in per_scan.items() if len(v) >= min_bins_per_scan}
    whole |= {(s["scan"], s["ant1"], s["ant2"], s["spw"], s["hand"]) for s in findings["spws"]}
    for scan, a1, a2, spw, hand in sorted(whole):
        commands.append(f"mode='manual' scan='{scan}' antenna='{names[a1]}&{names[a2]}' spw='{spw}' "
                        f"correlation='{hands[hand]}' reason='AI_DISTRIBUTION'")
    for key, bins in per_scan.items():
        if key in whole:
            continue
        for b in bins:
            t0, t1 = b["time"] - binsec / 2, b["time"] + binsec / 2
            commands.append(f"mode='manual' scan='{b['scan']}' antenna='{names[b['ant1']]}&{names[b['ant2']]}' "
                            f"spw='{b['spw']}' correlation='{hands[b['hand']]}' timerange='{_mjds(t0)}~{_mjds(t1)}' "
                            "reason='AI_OUTLIER'")
    for d in departures or []:
        commands.append(f"mode='manual' scan='{d['scan']}' antenna='{names[d['ant1']]}&{names[d['ant2']]}' "
                        f"correlation='{hands[d['hand']]}' reason='AI_AMPLITUDE_DEPARTURE'")
    return commands


def _mjds(seconds: float) -> str:
    """MJD seconds as a CASA time string (``YYYY/MM/DD/hh:mm:ss.s``)."""
    from datetime import datetime, timedelta
    moment = datetime(1858, 11, 17) + timedelta(seconds=float(seconds))
    return f"{moment:%Y/%m/%d/%H:%M}:{moment.second + moment.microsecond / 1e6:04.1f}"


#: A channel is bad when its amplitude, normalised per baseline and subband, departs from 1 by this fraction ...
CHANNEL_FRACTION = 0.1
#: ... and by this many robust sigmas of its subband's channels, consistently over the baselines.
CHANNEL_SIGMA = 6.0


def channel_outliers(ms, *, fields: list[str], refant: int, column: str = "CORRECTED_DATA") -> list[dict]:
    """Channels whose time-averaged amplitude is off on every baseline (a correlator or RFI channel).

    The vector-averaged spectrum of every baseline to ``refant`` (each subband and hand normalised to its own
    median) is combined over baselines and hands with a median, so one baseline's structure or noise cannot make a
    channel bad; a channel departing by more than ``CHANNEL_FRACTION`` and ``CHANNEL_SIGMA`` robust sigmas of its
    subband is returned as ``{"spw", "channel", "ratio"}``.
    """
    from ..solvers.plot_read import read_spectrum
    from ..statistics import mad_sigma
    spectra = read_spectrum(ms, fields=fields, scans=None, column=column, corrs=None, refant=int(refant))["spectra"]
    if not spectra:
        return []
    stack = []
    for values in spectra.values():                              # (nspw, nchan, ncorr)
        amp = np.abs(values[..., [0, values.shape[-1] - 1]])
        with np.errstate(invalid="ignore"):
            stack.append(amp / np.nanmedian(amp, axis=1, keepdims=True))
    profile = np.nanmedian(np.concatenate(stack, axis=-1), axis=-1)  # (nspw, nchan)
    out = []
    for spw in range(profile.shape[0]):
        row = profile[spw]
        good = np.isfinite(row)
        if good.sum() < 5:
            continue
        scale = max(float(mad_sigma(row[good])), 1e-3)
        for chan in np.flatnonzero(good & (np.abs(row - 1.0) > CHANNEL_FRACTION) &
                                   (np.abs(row - 1.0) > CHANNEL_SIGMA * scale)):
            out.append({"spw": int(spw), "channel": int(chan), "ratio": float(row[chan])})
    return out
