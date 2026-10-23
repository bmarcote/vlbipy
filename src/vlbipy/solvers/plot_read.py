"""Parallel readers for the diagnostic plots of the dask-ms backend.

The CASA readers behind the radplot and the per-baseline corner plots read one
subband and a few scans at a time through the ``ms`` tool and then loop in
Python over every baseline and time bin: 22 minutes for a phase calibrator of a
global VLBI track. These readers produce the same products from row blocks of
the table parts, read in parallel worker processes (only the rows of the wanted
fields), and reduce every block by integer keys with numpy (sort once,
``add.reduceat``); the main process only adds the per-block partial sums.

The averages are the same as the CASA readers' (a vector average over channels,
then over the samples of a bin, ignoring flagged samples); only the order of the
floating-point sums differs.
"""
from __future__ import annotations

import time as _time
import warnings
from typing import Optional

import numpy as np

from .apply_task import read_view_runs
from .corrected import view as corrected_view
from .msio import read_layout, read_setup, run_specs, select_runs, split_rows, table_engine
from .workers import run_jobs

#: Rows per worker job: 100k rows x 64 channels x 2 hands of complex64 is ~100 MB.
PLOT_ROWS = 100_000


def _jobs(ms, fields: Optional[list[str]], engine: str, **extra) -> tuple[dict, list[dict]]:
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    field_ids = None
    if fields:
        field_ids = {setup["field_names"].index(name) for name in fields if name in setup["field_names"]}
    selected = select_runs(layout, fields=field_ids)
    view = _view(ms, extra.get("column", ""), field_ids, setup, layout)
    jobs = [dict(extra, specs=chunk, engine=engine, ncorr=int(setup["ncorr"]), view=view)
            for chunk in split_rows(run_specs(layout, selected), PLOT_ROWS)]
    return setup, jobs


def _view(ms, column: str, field_ids, setup: dict, layout: dict) -> Optional[dict]:
    """How the jobs read ``column``: None, or the deferred calibration to apply to DATA (see
    :mod:`vlbipy.solvers.corrected`)."""
    return corrected_view(ms, column, field_ids, setup=setup, columns=layout["columns"])


def _reduce(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per distinct key: the sum of the finite ``values`` (rows) and how many there were (per column)."""
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]
    unique, starts = np.unique(keys, return_index=True)
    finite = np.isfinite(values)
    sums = np.add.reduceat(np.where(finite, values, 0), starts, axis=0)
    counts = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
    return unique, sums, counts


def _masked(block: dict, column: str, corrs: Optional[list[int]]) -> np.ndarray:
    """``(nrow, nchan, npol)`` visibilities of the selected hands, ``nan`` where flagged."""
    values = block.pop(column)
    flags = np.asarray(block.pop("FLAG"), dtype=bool)
    if corrs is not None:
        values, flags = values[:, :, corrs], flags[:, :, corrs]
    return np.where(flags, np.nan, values)


# ---------------------------------------------------------------------------
# Dynamic spectra (corner plots)
# ---------------------------------------------------------------------------
def dynamic_spectra_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int, engine: str,
                        edges: list[float], nant: int, nddid: int, stokes_i: bool, view: Optional[dict] = None) -> dict:
    """Worker job: per (baseline, subband, time bin), the sum and count of the unflagged samples per channel."""
    block = read_view_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine, view=view)
    if not block:
        return {}
    values = _masked(block, column, corrs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.nanmean(values, axis=2) if stokes_i else values[:, :, 0]     # I = (RR + LL) / 2
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    low, high = np.minimum(ant1, ant2), np.maximum(ant1, ant2)
    keep = (low != high) & (high < nant)
    n_time = len(edges) - 1
    bins = np.clip(np.digitize(np.asarray(block["TIME"], dtype=np.float64), edges) - 1, 0, n_time - 1)
    keys = ((low * nant + high) * nddid + np.asarray(block["DATA_DESC_ID"], dtype=np.int64)) * n_time + bins
    unique, sums, counts = _reduce(keys[keep], values[keep].astype(np.complex128))
    return {"keys": unique, "sums": sums, "counts": counts}


def read_dynamic_spectra(ms, *, fields: Optional[list[str]], column: str, corrs: Optional[list[int]],
                         edges: np.ndarray, stokes_i: bool = True, workers: Optional[int] = None) -> dict:
    """Time x frequency spectra of every baseline of ``fields``, binned on ``edges`` (MJD seconds).

    Returns
    -------
    dict
        ``spectra`` (``{(a, b): complex (n_time, nddid * nchan)}``, antenna indices, ``nan`` where no sample),
        ``ddid_to_spw`` and ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    nant, nddid = len(setup["antenna_names"]), len(setup["ddid_to_spw"])
    nchan = int(setup["chan_freq"].shape[1])
    n_time = len(edges) - 1
    _, jobs = _jobs(ms, fields, engine, column=column, corrs=corrs, edges=list(map(float, edges)), nant=nant,
                    nddid=nddid, stokes_i=stokes_i)
    sums: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, np.ndarray] = {}
    for part in run_jobs("vlbipy.solvers.plot_read:dynamic_spectra_job", jobs, workers):
        if not part:
            continue
        rest, bins = np.divmod(part["keys"], n_time)
        pairs, ddids = np.divmod(rest, nddid)
        for key, pair, ddid, bin_index, total, count in zip(part["keys"], pairs, ddids, bins, part["sums"],
                                                             part["counts"]):
            baseline = tuple(int(v) for v in divmod(int(pair), nant))
            if baseline not in sums:
                sums[baseline] = np.zeros((n_time, nddid * nchan), dtype=np.complex128)
                counts[baseline] = np.zeros((n_time, nddid * nchan), dtype=np.int64)
            slot = slice(int(ddid) * nchan, (int(ddid) + 1) * nchan)
            sums[baseline][bin_index, slot] += total
            counts[baseline][bin_index, slot] += count
    spectra = {}
    for baseline, total in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            spectra[baseline] = np.where(counts[baseline] > 0, total / np.maximum(counts[baseline], 1), np.nan)
    return {"spectra": spectra, "ddid_to_spw": setup["ddid_to_spw"], "seconds": _time.perf_counter() - started}


# ---------------------------------------------------------------------------
# Amplitude/phase against uv distance (radplot)
# ---------------------------------------------------------------------------
def uvdistance_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int, engine: str,
                   time_bin: float, scan_starts: dict, nant: int, nddid: int, with_model: bool,
                   view: Optional[dict] = None) -> dict:
    """Worker job: per (scan, baseline, subband, time bin), mean u, v and the summed channel-averaged values."""
    columns = [column, "FLAG", "UVW", "TIME", "ANTENNA1", "ANTENNA2"] + (["MODEL_DATA"] if with_model else [])
    block = read_view_runs(specs, columns, engine=engine, view=view)
    if not block:
        return {}
    model = block.pop("MODEL_DATA", None)
    flags = np.asarray(block["FLAG"], dtype=bool)
    values = _masked(block, column, corrs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_row = np.nanmean(values, axis=1)                         # vector average over channels
        if model is not None:
            if corrs is not None:
                model, flags = model[:, :, corrs], flags[:, :, corrs]
            model_row = np.nanmean(np.where(flags, np.nan, model), axis=1)
    times = np.asarray(block["TIME"], dtype=np.float64)
    scans = np.asarray(block["SCAN_NUMBER"], dtype=np.int64)
    # Bins count from each scan's start, so a bin never depends on how the rows were split into jobs.
    present, which = np.unique(scans, return_inverse=True)
    origin = np.array([scan_starts.get(int(s), times.min()) for s in present])[which]
    bins = np.floor((times - origin) / max(time_bin, 1e-6)).astype(np.int64)
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    ddid = np.asarray(block["DATA_DESC_ID"], dtype=np.int64)
    keys = (((scans * nant + ant1) * nant + ant2) * nddid + ddid) * 1_000_000 + bins
    uvw = np.asarray(block["UVW"], dtype=np.float64)
    unique, value_sums, value_counts = _reduce(keys, per_row.astype(np.complex128))
    _, uv_sums, uv_counts = _reduce(keys, uvw[:, :2])
    out = {"keys": unique, "sums": value_sums, "counts": value_counts, "uv": uv_sums, "rows": uv_counts[:, 0]}
    if model is not None:
        _, out["model_sums"], out["model_counts"] = _reduce(keys, model_row.astype(np.complex128))
    return out


def read_uvdistance(ms, *, fields: Optional[list[str]], column: str, corrs: Optional[list[int]],
                    time_bin: float, scan_starts: dict, with_model: bool = False,
                    workers: Optional[int] = None) -> dict:
    """Channel- and time-averaged visibilities per (scan, baseline, subband, ``time_bin``) with their uv distance.

    Returns
    -------
    dict
        ``uvdist_mlambda``, ``values`` (``(n, npol)`` complex), ``model`` (or None) and ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    nant, nddid = len(setup["antenna_names"]), len(setup["ddid_to_spw"])
    _, jobs = _jobs(ms, fields, engine, column=column, corrs=corrs, time_bin=float(time_bin),
                    scan_starts={int(k): float(v) for k, v in scan_starts.items()}, nant=nant, nddid=nddid,
                    with_model=with_model)
    parts = [p for p in run_jobs("vlbipy.solvers.plot_read:uvdistance_job", jobs, workers) if p]
    if not parts:
        npol = len(corrs) if corrs is not None else int(setup["ncorr"])
        return {"uvdist_mlambda": np.zeros(0), "values": np.zeros((0, npol), complex), "model": None,
                "seconds": _time.perf_counter() - started}
    keys = np.concatenate([p["keys"] for p in parts])
    # A bin split between two jobs is merged here: add the partial sums (uv positions weighted by rows).
    unique, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros((unique.size, parts[0]["sums"].shape[1]), dtype=np.complex128)
    counts = np.zeros(sums.shape, dtype=np.int64)
    uv = np.zeros((unique.size, 2))
    weight = np.zeros(unique.size)
    np.add.at(sums, inverse, np.concatenate([p["sums"] for p in parts]))
    np.add.at(counts, inverse, np.concatenate([p["counts"] for p in parts]))
    np.add.at(uv, inverse, np.concatenate([p["uv"] for p in parts]))
    np.add.at(weight, inverse, np.concatenate([p["rows"] for p in parts]))
    uv /= weight[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    model = None
    if with_model and "model_sums" in parts[0]:
        model_sums = np.zeros(sums.shape, dtype=np.complex128)
        model_counts = np.zeros(sums.shape, dtype=np.int64)
        np.add.at(model_sums, inverse, np.concatenate([p["model_sums"] for p in parts]))
        np.add.at(model_counts, inverse, np.concatenate([p["model_counts"] for p in parts]))
        with np.errstate(invalid="ignore", divide="ignore"):
            model = np.where(model_counts > 0, model_sums / np.maximum(model_counts, 1), np.nan)
    keep = np.isfinite(values).any(axis=1)
    ddid = (unique // 1_000_000) % nddid
    spw = setup["ddid_to_spw"][ddid]
    frequency = setup["chan_freq"].mean(axis=1)[spw]
    uvdist = np.hypot(uv[:, 0], uv[:, 1]) * frequency / 299792458.0 / 1e6
    return {"uvdist_mlambda": uvdist[keep], "values": values[keep],
            "model": model[keep] if model is not None else None, "seconds": _time.perf_counter() - started}


# ---------------------------------------------------------------------------
# Baselines to one antenna in one subband (time-evolution figures)
# ---------------------------------------------------------------------------
def time_evolution_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int, engine: str,
                       channels: list[int], nchan: int, refant: int, view: Optional[dict] = None) -> dict:
    """Worker job: rows of ``refant``'s baselines, averaged over the recorded unflagged ``channels``."""
    chans = np.asarray(channels)
    block = read_view_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine, view=view,
                           chans_by_ddid={spec["ddid"]: chans for spec in specs},
                           corrs=None if corrs is None else np.asarray(corrs), nchan=nchan, ncorr=ncorr)
    if not block:
        return {}
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    rows = np.flatnonzero(((ant1 == refant) | (ant2 == refant)) & (ant1 != ant2))
    values = block[column][rows]
    good = ~np.asarray(block["FLAG"], dtype=bool)[rows] & (values != 0)     # recorded and unflagged channels
    count = good.sum(axis=1)
    total = np.where(good, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.complex64)
    averaged = np.where((ant2[rows] == refant)[:, None], np.conj(averaged), averaged)   # reference antenna first
    keep = np.isfinite(averaged).any(axis=1)
    rows, averaged = rows[keep], averaged[keep]
    return {"other": np.where(ant1[rows] == refant, ant2[rows], ant1[rows]),
            "time": np.asarray(block["TIME"], dtype=np.float64)[rows],
            "scan": np.asarray(block["SCAN_NUMBER"], dtype=np.int64)[rows], "vis": averaged}


def read_time_evolution_rows(ms, *, column: str, corrs: Optional[list[int]], refant: int, ddids: list[int],
                             scans: Optional[list[int]], channels: list[int],
                             workers: Optional[int] = None) -> tuple:
    """``(other antenna, time, scan, vis)`` of every baseline to ``refant`` in the data descriptions ``ddids``."""
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    selected = select_runs(layout, ddids=set(ddids), scans=None if scans is None else set(scans))
    view = _view(ms, column, None, setup, layout)
    jobs = [{"specs": chunk, "column": column, "corrs": corrs, "ncorr": int(setup["ncorr"]), "engine": engine,
             "channels": list(channels), "nchan": int(setup["chan_freq"].shape[1]), "refant": int(refant),
             "view": view}
            for chunk in split_rows(run_specs(layout, selected), PLOT_ROWS * 4)]
    parts = [p for p in run_jobs("vlbipy.solvers.plot_read:time_evolution_job", jobs, workers) if p]
    npol = len(corrs) if corrs is not None else int(setup["ncorr"])
    if not parts:
        return (np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=int), np.zeros((0, npol), np.complex64))
    return tuple(np.concatenate([p[name] for p in parts]) for name in ("other", "time", "scan", "vis"))


# ---------------------------------------------------------------------------
# Coherent sum of every baseline per integration (light curve)
# ---------------------------------------------------------------------------
def total_visibility_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int,
                         engine: str, view: Optional[dict] = None) -> dict:
    """Worker job: per (field, scan, time), the sum and count of the unflagged cross-correlation samples."""
    block = read_view_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine, view=view,
                           corrs=None if corrs is None else np.asarray(corrs), ncorr=ncorr)
    if not block:
        return {}
    cross = np.asarray(block["ANTENNA1"]) != np.asarray(block["ANTENNA2"])
    ok = ~np.asarray(block["FLAG"], dtype=bool)[cross]
    row_sum = np.where(ok, block[column][cross], 0).sum(axis=(1, 2), dtype=np.complex128)
    row_num = ok.sum(axis=(1, 2))
    field = np.asarray(block["FIELD_ID"], dtype=np.int64)[cross]
    scan = np.asarray(block["SCAN_NUMBER"], dtype=np.int64)[cross]
    time = np.asarray(block["TIME"], dtype=np.float64)[cross]
    return _reduce_by_stamp(field, scan, time, row_sum, row_num)


def _reduce_by_stamp(field, scan, time, sums, counts) -> dict:
    """Add ``sums``/``counts`` per distinct (field, scan, time)."""
    if not time.size:
        return {}
    order = np.lexsort((time, scan, field))
    field, scan, time, sums, counts = field[order], scan[order], time[order], sums[order], counts[order]
    starts = np.flatnonzero(np.concatenate([[True], (np.diff(field) != 0) | (np.diff(scan) != 0)
                                            | (np.diff(time) != 0)]))
    return {"field": field[starts], "scan": scan[starts], "time": time[starts],
            "sum": np.add.reduceat(sums, starts), "count": np.add.reduceat(counts, starts)}


def read_total_visibility(ms, *, fields: Optional[list[str]], column: str, corrs: Optional[list[int]],
                          workers: Optional[int] = None) -> dict:
    """The coherent sum of every unflagged cross-correlation sample per field, scan and integration.

    Returns
    -------
    dict
        ``field_names`` and parallel arrays ``field``, ``scan``, ``time``, ``sum`` (complex) and ``count``,
        sorted by time; ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup, jobs = _jobs(ms, fields, engine, column=column, corrs=corrs)
    parts = [p for p in run_jobs("vlbipy.solvers.plot_read:total_visibility_job", jobs, workers) if p]
    out = {"field_names": setup["field_names"], "seconds": 0.0}
    if parts:
        merged = _reduce_by_stamp(*(np.concatenate([p[name] for p in parts])
                                    for name in ("field", "scan", "time", "sum", "count")))
        order = np.argsort(merged["time"], kind="stable")
        out.update({name: values[order] for name, values in merged.items()})
    else:
        out.update(field=np.zeros(0, np.int64), scan=np.zeros(0, np.int64), time=np.zeros(0),
                   sum=np.zeros(0, complex), count=np.zeros(0, np.int64))
    out["seconds"] = _time.perf_counter() - started
    return out


# ---------------------------------------------------------------------------
# One band-averaged value per baseline and integration (off-source detection)
# ---------------------------------------------------------------------------
def baseline_timeline_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int, engine: str,
                          nant: int, delay_search: bool, view: Optional[dict] = None) -> dict:
    """Worker job: per (baseline, time), the sum over subbands of the band-averaged value and how many there were.

    The value of a row is its vector average over the channels and hands (or, with ``delay_search``, its
    delay-searched fringe amplitude, see :func:`vlbipy.backends.casa._delay_peak_amplitude`), exactly as the
    CASA reader computes it.
    """
    block = read_view_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine, view=view,
                           corrs=None if corrs is None else np.asarray(corrs), ncorr=ncorr)
    if not block:
        return {}
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    good = ~np.asarray(block["FLAG"], dtype=bool)
    count = good.sum(axis=(1, 2))
    keep = (count > 0) & (ant1 != ant2) & (ant1 < nant) & (ant2 < nant)
    values, good, count = block[column][keep], good[keep], count[keep]
    if delay_search:
        from ..backends.casa import _delay_peak_amplitude
        row_values = _delay_peak_amplitude(np.transpose(values, (2, 1, 0)), np.transpose(good, (2, 1, 0)))
    else:
        row_values = np.where(good, values, 0).sum(axis=(1, 2), dtype=np.complex128) / count
    low, high = np.minimum(ant1, ant2)[keep], np.maximum(ant1, ant2)[keep]
    reduced = _reduce_by_stamp(low * nant + high, np.asarray(block["SCAN_NUMBER"], dtype=np.int64)[keep],
                               np.asarray(block["TIME"], dtype=np.float64)[keep],
                               row_values.astype(np.complex128), np.ones(row_values.shape, dtype=np.int64))
    if reduced:
        reduced["baseline"] = reduced.pop("field")
    return reduced


def read_baseline_timeline(ms, *, field: str, column: str, corrs: Optional[list[int]], scans: Optional[list[int]],
                           nant: int, delay_search: bool = False, workers: Optional[int] = None) -> dict:
    """Per (baseline, scan, time): the sum over subbands of the band-averaged value and the number of subbands.

    Returns
    -------
    dict
        Parallel arrays ``baseline`` (``low * nant + high``), ``scan``, ``time``, ``sum``, ``count``; ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    field_ids = {setup["field_names"].index(field)} if field in setup["field_names"] else set()
    selected = select_runs(layout, fields=field_ids, scans=None if scans is None else set(scans))
    view = _view(ms, column, field_ids, setup, layout)
    jobs = [{"specs": chunk, "column": column, "corrs": corrs, "ncorr": int(setup["ncorr"]), "engine": engine,
             "nant": int(nant), "delay_search": bool(delay_search), "view": view}
            for chunk in split_rows(run_specs(layout, selected), PLOT_ROWS)]
    parts = [p for p in run_jobs("vlbipy.solvers.plot_read:baseline_timeline_job", jobs, workers) if p]
    if not parts:
        return {"baseline": np.zeros(0, np.int64), "scan": np.zeros(0, np.int64), "time": np.zeros(0),
                "sum": np.zeros(0, complex), "count": np.zeros(0, np.int64), "seconds": _time.perf_counter() - started}
    merged = _reduce_by_stamp(*(np.concatenate([p[name] for p in parts])
                                for name in ("baseline", "scan", "time", "sum", "count")))
    merged["baseline"] = merged.pop("field")
    merged["seconds"] = _time.perf_counter() - started
    return merged


# ---------------------------------------------------------------------------
# Baselines to one antenna, averaged over frequency (time series)
# ---------------------------------------------------------------------------
def timeseries_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int, engine: str,
                   edges: list[float], refant: int, nant: int, view: Optional[dict] = None) -> dict:
    """Worker job: per (other antenna, time bin), the sum and count of the unflagged samples of ``refant``'s
    baselines over channels and rows, per hand."""
    block = read_view_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine, view=view,
                           corrs=None if corrs is None else np.asarray(corrs), ncorr=ncorr)
    if not block:
        return {}
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    other = np.where(ant1 == refant, ant2, ant1)
    keep = ((ant1 == refant) | (ant2 == refant)) & (ant1 != ant2) & (other < nant)
    if not keep.any():
        return {}
    values = np.where(np.asarray(block["FLAG"], dtype=bool)[keep], np.nan, block[column][keep])
    finite = np.isfinite(values)
    row_sums = np.where(finite, values, 0).sum(axis=1, dtype=np.complex128)      # (nrow, npol)
    row_counts = finite.sum(axis=1)
    n_time = len(edges) - 1
    bins = np.clip(np.digitize(np.asarray(block["TIME"], dtype=np.float64)[keep], edges) - 1, 0, n_time - 1)
    keys = other[keep] * n_time + bins
    order = np.argsort(keys, kind="stable")
    unique, starts = np.unique(keys[order], return_index=True)
    return {"keys": unique, "sums": np.add.reduceat(row_sums[order], starts, axis=0),
            "counts": np.add.reduceat(row_counts[order], starts, axis=0)}


def read_timeseries(ms, *, fields: Optional[list[str]], scans: Optional[list[int]], column: str,
                    corrs: Optional[list[int]], refant: int, edges: np.ndarray,
                    workers: Optional[int] = None) -> dict:
    """Vector averages over channels and the rows of each time bin (``edges``, MJD seconds) of every baseline
    to ``refant``, per hand, as the CASA reader computes them.

    Returns
    -------
    dict
        ``{other antenna id: complex (n_time, npol)}`` (``nan`` where a bin has no unflagged sample) under
        ``baselines``, and ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    field_ids = None
    if fields:
        field_ids = {setup["field_names"].index(name) for name in fields if name in setup["field_names"]}
    selected = select_runs(layout, fields=field_ids, scans=None if not scans else set(int(s) for s in scans))
    view = _view(ms, column, field_ids, setup, layout)
    nant = len(setup["antenna_names"])
    n_time = len(edges) - 1
    jobs = [{"specs": chunk, "column": column, "corrs": corrs, "ncorr": int(setup["ncorr"]), "engine": engine,
             "edges": list(map(float, edges)), "refant": int(refant), "nant": nant, "view": view}
            for chunk in split_rows(run_specs(layout, selected), PLOT_ROWS)]
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for part in run_jobs("vlbipy.solvers.plot_read:timeseries_job", jobs, workers):
        if not part:
            continue
        others, bins = np.divmod(part["keys"], n_time)
        for other, bin_index, total, count in zip(others, bins, part["sums"], part["counts"]):
            if int(other) not in sums:
                sums[int(other)] = np.zeros((n_time, total.size), dtype=np.complex128)
                counts[int(other)] = np.zeros((n_time, total.size), dtype=np.float64)
            sums[int(other)][bin_index] += total
            counts[int(other)][bin_index] += count
    baselines = {}
    for other, total in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            baselines[other] = np.where(counts[other] > 0, total / np.maximum(counts[other], 1), np.nan)
    return {"baselines": baselines, "seconds": _time.perf_counter() - started}
