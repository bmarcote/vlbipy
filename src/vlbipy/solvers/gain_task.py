"""Bandpass and gain solvers for the dask-ms backend, writing CASA B/G Jones tables.

:func:`run_bandpass` and :func:`run_gaincal` take the keyword sets of
``casatasks.bandpass`` / ``casatasks.gaincal``. Like CASA they solve on
*averages*: within a solution interval the calibrated visibilities of each
baseline are vector-averaged in time (and, for a gain, over the channels), and
the antenna gains are then fitted to those baseline averages.

That split is what makes it fast. Worker processes read the measurement set
chunk by chunk, apply the prior tables on the fly and return only the baseline
sums — a few kilobytes per chunk (:func:`accumulate_job`). The parent adds them
up and solves all intervals, channels and polarizations in one vectorised pass
(:func:`solve_antenna_gains`, an alternating least-squares fit, which reaches the
same weighted least-squares solution as CASA's solver).

Supported: ``bandtype='B'``; ``gaintype='G'`` with ``calmode`` ``'a'``, ``'p'``
or ``'ap'``; ``solint`` ``'inf'``, ``'int'`` or a duration; ``combine`` with
``scan`` and/or ``field``. Anything else raises :class:`NotImplementedError`, on
which the backend falls back to the CASA task.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .caltable import write_jones_table
from .fringefit_task import (_prior_entries, parse_antennas, parse_names, parse_scans, parse_solint, parse_spw,
                            parse_timerange)
from .msio import read_layout, read_setup, run_specs, select_runs, split_rows, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows read per worker job (64 channels x 2 hands: ~15 MB of visibilities, ~150 MB peak in the worker).
SOLVE_ROWS = 30_000


# ---------------------------------------------------------------------------
# Worker side: read a chunk and reduce it to baseline sums
# ---------------------------------------------------------------------------
def _interval_keys(block: dict, solint_sec: float, combine_scan: bool, origin: float) -> np.ndarray:
    """Per-row solution-interval key: (scan or -1) and the index of the ``solint_sec`` bin since ``origin``.

    ``inf`` gives one bin per scan (or one overall with ``combine_scan``); 0 (``'int'``) gives one per timestamp.
    """
    scan = np.full(block["time"].shape, -1, dtype=np.int64) if combine_scan else block["scan"].astype(np.int64)
    if not np.isfinite(solint_sec):
        bins = np.zeros(scan.shape, dtype=np.int64)
    elif solint_sec <= 0:
        bins = np.rint((block["time"] - origin) * 1000.0).astype(np.int64)
    else:
        bins = np.floor((block["time"] - origin) / solint_sec).astype(np.int64)
    return np.stack([scan, bins], axis=1)


def accumulate_job(*, specs: list[dict], setup: dict, engine: str, columns: list[str], data_column: str,
                   chans: dict, per_channel: bool, antennas: Optional[set], among: bool,
                   timerange: Optional[tuple], parang: bool, priors: list, solint_sec: float, combine_scan: bool,
                   origin: float) -> list[dict]:
    """Worker job: read some runs of one field, calibrate them and return the baseline sums per solution interval.

    Returns
    -------
    list of dict
        One per (spw, interval) present: ``key`` = (field, spw, scan or -1, time bin); ``cross`` complex
        (nant, nant, nchan or 1, npol) = sum of w V conj(M); ``wsum`` float, same shape = sum of w |M|^2;
        ``time_sum``/``weight_total`` for the weighted time centroid; ``t_min``/``t_max``; ``scan`` (lowest).
    """
    from .calblock import load_block
    block = load_block(specs, setup=setup, engine=engine, columns=columns, data_column=data_column, chans=chans,
                       parallel_hands=True, timerange=timerange, antennas=antennas, among=among, parang=parang,
                       priors=priors, model=True, sigma_weights=True)
    if block is None:
        return []
    nant = len(setup["antenna_names"])
    vis, flag, weight = block["vis"], block["flag"], block["weight"]
    usable = ~flag & np.isfinite(vis) & (weight > 0)
    if "model" in block and np.any(block["model"]):
        model = block["model"]
        usable &= np.abs(model) > 0
        w = np.where(usable, weight, 0.0).astype(np.float64)
        y = np.where(usable, vis * np.conj(model), 0.0) * w
        w = w * np.abs(model) ** 2
    else:
        w = np.where(usable, weight, 0.0).astype(np.float64)
        y = np.where(usable, vis, 0.0) * w
    if not per_channel:
        y, w = y.sum(axis=1, keepdims=True), w.sum(axis=1, keepdims=True)
    row_weight = w.sum(axis=(1, 2))
    keys = _interval_keys(block, solint_sec, combine_scan, origin)
    group = np.column_stack([block["spw"], keys, block["antenna1"] * nant + block["antenna2"]])
    order = np.lexsort(group.T[::-1])
    group, y, w = group[order], y[order], w[order]
    time, row_weight, scan = block["time"][order], row_weight[order], block["scan"][order]
    # Rows are now sorted by (spw, scan, bin, baseline): one reduceat sums every baseline of every interval.
    starts = np.concatenate([[0], np.flatnonzero(np.any(np.diff(group, axis=0) != 0, axis=1)) + 1])
    y_bl, w_bl = np.add.reduceat(y, starts, axis=0), np.add.reduceat(w, starts, axis=0)
    head = group[starts]
    out = []
    interval_starts = np.concatenate([[0], np.flatnonzero(np.any(np.diff(head[:, :3], axis=0) != 0, axis=1)) + 1,
                                      [len(head)]])
    row_edges = np.concatenate([starts, [len(group)]])
    for lo, hi in zip(interval_starts[:-1], interval_starts[1:]):
        cross = np.zeros((nant, nant) + y_bl.shape[1:], dtype=np.complex128)
        wsum = np.zeros((nant, nant) + w_bl.shape[1:], dtype=np.float64)
        baseline = head[lo:hi, 3]
        cross[baseline // nant, baseline % nant] = y_bl[lo:hi]
        wsum[baseline // nant, baseline % nant] = w_bl[lo:hi]
        rows = slice(int(row_edges[lo]), int(row_edges[hi]))
        out.append({"key": (block["field"], int(head[lo, 0]), int(head[lo, 1]), int(head[lo, 2])),
                    "cross": cross, "wsum": wsum, "time_sum": float(np.dot(time[rows], row_weight[rows])),
                    "weight_total": float(row_weight[rows].sum()), "t_min": float(time[rows].min()),
                    "t_max": float(time[rows].max()), "scan": int(scan[rows].min())})
    return out


# ---------------------------------------------------------------------------
# Parent side: solve the antenna gains from the baseline sums
# ---------------------------------------------------------------------------
def solve_antenna_gains(cross: np.ndarray, wsum: np.ndarray, *, mode: str = "ap", minsnr: float = 0.0,
                        minblperant: int = 4, iterations: int = 200,
                        tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit antenna gains to baseline averages, for every channel and polarization at once.

    Model: the averaged visibility of baseline (i, j) is ``g_i conj(g_j)`` (unit point-source model, or the
    data already divided by the model), weighted by the summed weights. The "channel" axis is any set of
    independent solves: the channels of a bandpass, or the solution intervals of a gain table stacked together.

    Parameters
    ----------
    cross, wsum : arrays (nant, nant, nchan, npol)
        Sum of w V (complex) and sum of w (float) per baseline as stored in the MS (each baseline once, either order).
    mode : {"ap", "p", "a"}
        ``"a"`` fits amplitudes to the amplitudes of the averages (phases discarded, as CASA ``calmode='a'``);
        ``"p"`` fits phases to the unit-amplitude averages and returns unit-modulus gains.
    minsnr : float
        Solutions below this signal-to-noise are flagged. The signal-to-noise is |g| over its formal error,
        divided by the square root of the reduced chi-square of the fit (CASA's convention).
    minblperant : int
        Antennas with fewer baselines carrying data are not solved (flagged).
    iterations, tolerance : int, float
        Alternating-least-squares limits.

    Returns
    -------
    (gains, flags, snr)
        ``gains`` complex (nant, nchan, npol) with an arbitrary phase origin per channel/polarization (see
        :func:`reference_gains`), 1 where flagged; ``flags`` bool and ``snr`` float of that shape.
    """
    nant = cross.shape[0]
    # Hermitian baseline matrix: element (i, j) is the average visibility with i as the first antenna.
    weight = wsum + wsum.transpose(1, 0, 2, 3)
    total = cross + np.conj(cross.transpose(1, 0, 2, 3))
    target = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
    if mode == "a":
        target = np.abs(target).astype(np.complex128)
    elif mode == "p":
        amplitude = np.abs(target)
        target = np.divide(target, amplitude, out=np.zeros_like(target), where=amplitude > 0)
        weight = weight * amplitude ** 2
    # Drop antennas with too few baselines, repeatedly: removing one can leave another short.
    active = np.ones((nant,) + cross.shape[2:], dtype=bool)
    for _ in range(nant):
        linked = weight * active[None, :] * active[:, None]
        short = active & ((linked > 0).sum(axis=1) < max(1, int(minblperant)))
        if not short.any():
            break
        active &= ~short
    weight = weight * active[None, :] * active[:, None]
    # Start from the principal eigenvector of the weighted baseline matrix: a unit-gain start can settle in a
    # wrong phase minimum on the sparse arrays VLBI scans often are.
    matrix = np.moveaxis(target * (weight > 0), (0, 1), (-2, -1))
    values, vectors = np.linalg.eigh(matrix)
    gains = np.moveaxis(vectors[..., -1] * np.sqrt(np.maximum(values[..., -1:], 0.0)), -1, 0)
    connections = np.maximum((weight > 0).sum(axis=1), 1)
    gains = gains * np.sqrt(nant / connections) if mode != "p" else gains
    gains = np.where(np.abs(gains) > 0, gains, 1.0).astype(np.complex128)
    if mode == "a":
        gains = np.abs(gains).astype(np.complex128)
    for iteration in range(iterations):
        # g_i = sum_j w_ij V_ij g_j / sum_j w_ij |g_j|^2, all antennas together; averaging successive iterates
        # (StEFCal) keeps the alternating update from oscillating.
        numerator = np.einsum("ijcp,ijcp,jcp->icp", weight, target, gains)
        denominator = np.einsum("ijcp,jcp->icp", weight, np.abs(gains) ** 2)
        update = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
        if mode == "a":
            update = np.abs(update).astype(np.complex128)
        elif mode == "p":
            modulus = np.abs(update)
            update = np.divide(update, modulus, out=np.ones_like(update), where=modulus > 0)
        new = update if iteration % 2 == 0 else 0.5 * (gains + update)
        change = float(np.max(np.abs(new - gains) * active)) if active.any() else 0.0
        gains = new
        if iteration % 2 == 1 and change < tolerance:
            break
    information = np.einsum("ijcp,jcp->icp", weight, np.abs(gains) ** 2)
    snr = np.abs(gains) * np.sqrt(np.maximum(information, 0.0))
    # CASA reports the formal signal-to-noise scaled by the misfit of the solve: sqrt(chi^2 / (baselines -
    # antennas)). On real data that factor is large (a calibrator is not the unit point source of the model), and
    # minsnr is applied to the scaled value, so the same convention is needed to flag the same solutions.
    upper = np.triu_indices(nant, 1)
    residual = np.abs(target - gains[:, None] * np.conj(gains[None, :])) ** 2
    chi2 = (weight * residual)[upper].sum(axis=0)
    freedom = (weight[upper] > 0).sum(axis=0) - active.sum(axis=0)
    misfit = np.sqrt(np.divide(chi2, freedom, out=np.ones_like(chi2), where=(freedom > 0) & (chi2 > 0)))
    snr = snr / misfit[None]
    flags = ~active | (information <= 0) | (snr < float(minsnr))
    return np.where(flags, 1.0, gains), flags, np.where(flags, 0.0, snr)


def reference_gains(gains: np.ndarray, flags: np.ndarray, refant_chain: list[int]) -> tuple[np.ndarray, int]:
    """Rotate one solution (nant, nchan, npol) so its reference antenna has phase zero in every channel and pol.

    The reference is the first antenna of ``refant_chain`` with any unflagged value; returns the rotated gains
    and that antenna (-1, gains unchanged, when the chain has no solution).
    """
    refant = next((ant for ant in refant_chain if 0 <= ant < gains.shape[0] and not flags[ant].all()), -1)
    if refant < 0:
        return gains, refant
    reference = gains[refant]
    modulus = np.abs(reference)
    rotation = np.conj(np.divide(reference, modulus, out=np.ones_like(reference), where=modulus > 0))
    return np.where(flags, 1.0, gains * rotation[None]), refant


def fill_channel_gaps(gains: np.ndarray, flags: np.ndarray, max_gap: int) -> None:
    """Interpolate (amplitude and phase, linearly) across flagged channel runs of at most ``max_gap`` channels.

    Works in place on ``gains``/``flags`` (nant, nchan, npol); gaps touching either band edge are left flagged.
    """
    if max_gap <= 0:
        return
    nchan = gains.shape[1]
    for ant in range(gains.shape[0]):
        for pol in range(gains.shape[2]):
            bad = flags[ant, :, pol]
            if not bad.any() or bad.all():
                continue
            edges = np.diff(np.r_[False, bad, False].astype(np.int8))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if end - start > max_gap or start == 0 or end == nchan:
                    continue
                left, right = start - 1, end
                inside = np.arange(start, end)
                amplitude = np.interp(inside, [left, right], np.abs(gains[ant, [left, right], pol]))
                phase = np.interp(inside, [left, right], np.unwrap(np.angle(gains[ant, [left, right], pol])))
                gains[ant, start:end, pol] = amplitude * np.exp(1j * phase)
                flags[ant, start:end, pol] = False


def normalise_bandpass(gains: np.ndarray, flags: np.ndarray) -> None:
    """Scale each antenna/polarization bandpass to unit RMS amplitude over its unflagged channels (CASA ``solnorm``)."""
    good = ~flags
    count = good.sum(axis=1, keepdims=True)
    power = np.sum(np.where(good, np.abs(gains) ** 2, 0.0), axis=1, keepdims=True)
    norm = np.sqrt(np.divide(power, count, out=np.ones_like(power), where=count > 0))
    gains /= np.where(norm > 0, norm, 1.0)


def _solve(vis, caltable, *, bandpass: bool, field="", spw="", scan="", timerange="", antenna="", solint="inf",
           combine="", refant="", minsnr=3.0, minblperant=4, solnorm=False, fillgaps=0, calmode="ap", parang=False,
           gaintable=None, gainfield=None, interp=None, spwmap=None, docallib=False, workers: Optional[int] = None,
           data_column: str = "DATA", **ignored) -> Path:
    """Shared driver of :func:`run_bandpass` and :func:`run_gaincal` (see those for the parameters)."""
    if docallib:
        raise NotImplementedError("cal-library files are not supported by the dask-ms solvers; pass gaintable lists")
    combine_parts = {part.strip() for part in str(combine or "").split(",") if part.strip()}
    if combine_parts - {"scan", "field"}:
        raise NotImplementedError(f"combine={combine!r} is not supported by the dask-ms gain solvers")
    mode = str(calmode or "ap").lower()
    if mode not in ("a", "p", "ap"):
        raise NotImplementedError(f"calmode={calmode!r} is not supported")
    started = _time.perf_counter()
    vis, caltable = str(vis), Path(caltable)
    engine = table_engine()
    setup = read_setup(vis, engine=engine)
    layout = read_layout(vis, engine=engine)
    nant = len(setup["antenna_names"])
    nspw, nchan = setup["chan_freq"].shape
    field_ids = set(parse_names(field, setup["field_names"]))
    spw_sel = parse_spw(spw, nspw, nchan)
    scans = parse_scans(scan)
    trange = parse_timerange(timerange)
    antenna_sel, among = parse_antennas(antenna, setup["antenna_names"])
    refant_chain = parse_names(refant, setup["antenna_names"]) if str(refant).strip() else []
    solint_sec = parse_solint(solint)
    from .apply import normalise_entries
    # CASA calibrates the weights with every prior table when it solves (setapply calwt=True).
    prior_entries = _prior_entries(gaintable, gainfield, interp, spwmap, setup["field_names"])
    table_cache: dict = {}
    priors_by_field = {field_id: [entry[:4] + (True,) for entry in normalise_entries(
        prior_entries, cache=table_cache, field_dirs=setup["field_dirs"], target_field=field_id)]
        for field_id in sorted(field_ids)}

    common = {"setup": setup, "engine": engine, "columns": layout["columns"], "data_column": data_column,
              "per_channel": bandpass, "antennas": antenna_sel, "among": among, "timerange": trange,
              "parang": bool(parang), "solint_sec": solint_sec,
              "combine_scan": "scan" in combine_parts, "origin": float(layout["t_min"].min()) if layout["nrows"] else 0.0}
    jobs = []
    for field_id in sorted(field_ids):
        for spw_id, channels in sorted(spw_sel.items()):
            ddids = {ddid for ddid, s in enumerate(setup["ddid_to_spw"]) if int(s) == spw_id}
            selected = select_runs(layout, fields={field_id}, scans=scans, ddids=ddids, timerange=trange)
            for chunk in split_rows(run_specs(layout, selected), SOLVE_ROWS):
                jobs.append(dict(common, specs=chunk, chans={spw_id: channels}, priors=priors_by_field[field_id]))
    if not jobs:
        raise RuntimeError(f"no data selected for the {'bandpass' if bandpass else 'gain'} solve on {Path(vis).name}")
    order = np.argsort([-sum(spec["nrow"] for spec in job["specs"]) for job in jobs], kind="stable")
    results = run_jobs("vlbipy.solvers.gain_task:accumulate_job", [jobs[i] for i in order], workers)

    sums: dict[tuple, dict] = {}
    for part in (item for group in results for item in group):
        key = part["key"] if "field" not in combine_parts else (-1,) + part["key"][1:]
        total = sums.get(key)
        if total is None:
            sums[key] = dict(part, field=part["key"][0])
            continue
        for name in ("cross", "wsum", "time_sum", "weight_total"):
            total[name] = total[name] + part[name]
        total["t_min"], total["t_max"] = min(total["t_min"], part["t_min"]), max(total["t_max"], part["t_max"])
        total["scan"], total["field"] = min(total["scan"], part["scan"]), min(total["field"], part["key"][0])
    if not sums:
        raise RuntimeError(f"no unflagged data for the {'bandpass' if bandpass else 'gain'} solve on {Path(vis).name}")

    table = {name: [] for name in ("time", "field", "spw", "antenna", "refant", "scan", "interval", "gain", "flag",
                                   "snr")}
    keys = sorted(sums, key=lambda k: (k[2], k[3], k[1], k[0]))
    solve_mode = "ap" if bandpass else mode
    # Every solution interval is an independent solve of the same shape: stack them along the channel axis and
    # solve them in one vectorised pass.
    nsol = sums[keys[0]]["cross"].shape[2]
    gains_all, flags_all, snr_all = solve_antenna_gains(
        np.concatenate([sums[key]["cross"] for key in keys], axis=2),
        np.concatenate([sums[key]["wsum"] for key in keys], axis=2), mode=solve_mode, minsnr=float(minsnr),
        minblperant=int(minblperant))
    for index, key in enumerate(keys):
        total = sums[key]
        spw_id = key[1]
        chunk = slice(index * nsol, (index + 1) * nsol)
        flags, snr = flags_all[:, chunk], snr_all[:, chunk]
        gains, used_refant = reference_gains(gains_all[:, chunk], flags, refant_chain)
        if solve_mode == "a":
            used_refant = next((ant for ant in refant_chain if not flags[ant].all()), -1)
        if used_refant < 0 and refant_chain:
            logger.warning("calibration: no reference antenna of the chain has a solution (field {} spw {})",
                           total["field"], spw_id)
        if bandpass:
            channels = spw_sel[spw_id]
            full_gain = np.ones((nant, nchan, gains.shape[2]), dtype=np.complex128)
            full_flag = np.ones((nant, nchan, gains.shape[2]), dtype=bool)
            full_snr = np.zeros((nant, nchan, gains.shape[2]), dtype=np.float64)
            full_gain[:, channels], full_flag[:, channels], full_snr[:, channels] = gains, flags, snr
            gains, flags, snr = full_gain, full_flag, full_snr
            fill_channel_gaps(gains, flags, int(fillgaps))
            if solnorm:
                normalise_bandpass(gains, flags)
        elif solnorm:
            logger.warning("gaincal: solnorm is not applied by the dask-ms solver (the pipeline normalises itself)")
        centroid = total["time_sum"] / total["weight_total"] if total["weight_total"] > 0 else 0.5 * (
            total["t_min"] + total["t_max"])
        table["time"].extend([centroid] * nant)
        table["field"].extend([total["field"]] * nant)
        table["spw"].extend([spw_id] * nant)
        table["antenna"].extend(range(nant))
        table["refant"].extend([used_refant] * nant)
        table["scan"].extend([total["scan"]] * nant)
        table["interval"].extend([0.0 if not np.isfinite(solint_sec) else float(solint_sec)] * nant)
        table["gain"].append(gains)
        table["flag"].append(flags)
        table["snr"].append(snr)
    flags = np.concatenate(table["flag"])
    # A gain table describes each subband as the single channel its solutions were averaged over.
    averaged = {} if bandpass else {
        "spw_chan_freq": [float(setup["chan_freq"][s, spw_sel[s]].mean()) if s in spw_sel
                          else float(setup["chan_freq"][s].mean()) for s in range(nspw)],
        "spw_chan_width": [float(np.ptp(setup["chan_freq"][s]) * nchan / max(nchan - 1, 1)) for s in range(nspw)]}
    write_jones_table(caltable, vis, viscal="B Jones" if bandpass else "G Jones", times=table["time"],
                      field_ids=table["field"], spw_ids=table["spw"], antenna_ids=table["antenna"],
                      refant_ids=table["refant"], scan_numbers=table["scan"], intervals=table["interval"],
                      cparam=np.concatenate(table["gain"]).astype(np.complex64), flag=flags,
                      snr=np.concatenate(table["snr"]).astype(np.float32), **averaged)
    logger.info("{}[dask-ms]: {} solution(s) x {} antennas from {} job(s), {:.0%} flagged; {} written in {:.2f}s",
                "bandpass" if bandpass else "gaincal", len(sums), nant, len(jobs), float(flags.mean()),
                caltable.name, _time.perf_counter() - started)
    return caltable


def run_bandpass(vis: str, caltable: str, *, bandtype="B", **params) -> Path:
    """Solve a per-channel bandpass into a CASA "B Jones" table (``casatasks.bandpass`` keyword set).

    Honoured: ``field``, ``spw``, ``scan``, ``timerange``, ``antenna``, ``solint``, ``combine``
    (``scan``/``field``), ``refant`` (fallback chain), ``minsnr``, ``minblperant``, ``solnorm`` (unit RMS
    amplitude per antenna and polarization), ``fillgaps``, ``parang`` and the prior lists
    ``gaintable``/``gainfield``/``interp``/``spwmap``. ``workers`` sets the number of worker processes.
    ``bandtype='BPOLY'`` and cal libraries raise :class:`NotImplementedError`.
    """
    if str(bandtype).upper() != "B":
        raise NotImplementedError("the dask-ms bandpass solver supports only bandtype='B'")
    params.pop("calmode", None)
    return _solve(vis, caltable, bandpass=True, **params)


def run_gaincal(vis: str, caltable: str, *, gaintype="G", calmode="ap", **params) -> Path:
    """Solve channel-averaged antenna gains into a CASA "G Jones" table (``casatasks.gaincal`` keyword set).

    Honoured: the selections, ``solint`` (``'inf'``, ``'int'``, seconds), ``combine`` (``scan``/``field``),
    ``refant``, ``minsnr``, ``minblperant``, ``calmode`` (``'a'``, ``'p'``, ``'ap'``), ``parang`` and the prior
    lists. ``gaintype`` other than ``'G'`` raises :class:`NotImplementedError`.
    """
    if str(gaintype).upper() != "G":
        raise NotImplementedError("the dask-ms gain solver supports only gaintype='G'")
    params.pop("fillgaps", None)
    return _solve(vis, caltable, bandpass=False, calmode=calmode, **params)
