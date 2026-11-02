"""The visibility-statistics cube: the reduced view of a measurement set that every AI-mode diagnostic reads.

A VLBI data set is too large to look at sample by sample, but almost every question asked of it during
calibration ("did this antenna lose sensitivity?", "is that subband low?", "does this baseline follow a Rice
distribution?") can be answered from a handful of moments per *cell*: one field, scan, baseline, subband,
hand and time bin. :func:`visibility_stats` computes them in parallel worker jobs (the same row-block readers
as the plots, :mod:`vlbipy.solvers.plot_read`, so a deferred calibration is applied on the fly) and returns a
flat table (a dict of equal-length arrays, saved as ``.npz``) with, per cell and hand:

* ``vec`` - the vector average over the central channels and the integrations of the bin. Meaningful once
  the delays and rates are calibrated: on raw data it decorrelates;
* ``power`` - the mean of ``|V|^2``, and ``noise2`` - the variance of one complex sample, estimated from the
  differences of consecutive integrations of the cell, channel by channel: the second difference
  (``|V[t+1] - 2 V[t] + V[t-1]|^2 / 6``) on raw DATA, where it cancels the phase a residual fringe rate turns
  between integrations, the first difference on calibrated data. Differences *across channels* are not used:
  SFXC's channels are correlated (adjacent-channel noise correlation ``rho1 = 0.50`` measured on EB104G, from
  its windowed FFT), which makes them underestimate the noise by a factor ~3 in variance.
  ``sqrt(power - noise2)`` is an unbiased estimate of the amplitude *whatever the phases do*: the quantity that
  shows a sensitivity drop in data no fringe fit has touched yet (noisy below a per-sample SNR of ~1);
* ``noise2_avg`` - the variance of ``vec`` itself: ``noise2 (1 + 2 rho1) / count``, with ``rho1`` (stored per
  hand) measured from the same time differences. Ignoring the channel correlation would make every channel
  average look sqrt(2) less noisy than it is;
* ``scalar`` - the mean of ``|V|`` (the scalar average, biased by the noise: Rice);
* ``count`` - unflagged samples, ``weight`` - mean visibility weight, ``uvw`` - mean baseline [m],
  ``time`` - mean time [MJD s].

Only the parallel hands are kept (RR and LL, or XX and YY), and only the central channels of every subband
(``channel_fraction``), where the bandpass is flat.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from ..solvers.applycal import read_view_runs
from ..solvers.corrected import view as corrected_view
from ..solvers.msio import read_layout, read_setup, run_specs, select_runs, split_rows, table_engine
from ..solvers.workers import run_jobs

logger = get_logger()

#: Rows per worker job (100k rows x 64 channels x 2 hands of complex64 is ~100 MB).
STATS_ROWS = 100_000
#: Time bins are numbered within this many per scan (keys are packed into one int64).
_MAX_BINS = 1 << 20
#: The per-hand moments accumulated per cell (all summed, then divided by ``count`` at the end).
_SUMMED = ("vsum", "asum", "psum", "d2sum", "d2n", "count", "wsum", "wn",
           "t1sum", "t1n", "t1c", "t1cn", "t2sum", "t2n")


def _central_channels(nchan: int, fraction: float) -> np.ndarray:
    """Indices of the central ``fraction`` of ``nchan`` channels (at least three)."""
    drop = int(round(nchan * (1.0 - fraction) / 2.0))
    drop = min(drop, max(0, (nchan - 3) // 2))
    return np.arange(drop, nchan - drop)


def _version_flags(specs: list[dict], version: dict, chans_by_ddid: dict, corrs: list[int]) -> np.ndarray:
    """FLAG of ``specs`` (in read order) from a saved flag version instead of the main table."""
    from ..casacore_format.table import Table
    pieces = []
    with Table(version["path"]) as table:
        for spec in specs:
            row = int(version["offsets"][str(Path(spec["path"]).resolve())]) + int(spec["start"])
            flags = np.asarray(table.getcol("FLAG", row, int(spec["nrow"])), dtype=bool)
            chans = chans_by_ddid.get(spec["ddid"])
            if chans is not None:
                flags = flags[:, chans]
            pieces.append(flags[:, :, corrs])
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def stats_job(*, specs: list[dict], column: str, engine: str, ncorr: int, nant: int, nddid: int, corrs: list[int],
              chans_by_ddid: dict, binsec: float, t0: float, view: Optional[dict] = None,
              flag_version: Optional[dict] = None) -> dict:
    """Worker job: the summed moments of every (scan, baseline, DDID, time bin) cell of a block of rows."""
    # A deferred chain cannot calibrate the weights of a channel selection, and the cube does not need them then.
    columns = [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2", "UVW"] + ([] if view is not None else ["WEIGHT"])
    block = read_view_runs(specs, columns, engine=engine, view=view, chans_by_ddid=chans_by_ddid,
                           corrs=np.asarray(corrs), ncorr=ncorr,
                           nchan=0 if view is None else int(view["setup"]["chan_freq"].shape[1]))
    if not block:
        return {}
    if flag_version is not None:
        block["FLAG"] = _version_flags(specs, flag_version, chans_by_ddid, corrs)
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    keep = (ant1 != ant2) & (ant1 < nant) & (ant2 < nant)
    if not keep.any():
        return {}
    values = np.asarray(block[column])[keep]
    flags = np.asarray(block["FLAG"], dtype=bool)[keep]
    if values.shape[2] != len(corrs):          # a view returns every correlation: keep the parallel hands
        values, flags = values[:, :, corrs], flags[:, :, corrs]
    valid = ~flags & np.isfinite(values)
    zero = np.zeros((), dtype=values.dtype)
    v = np.where(valid, values, zero)
    vsum = v.sum(axis=1, dtype=np.complex128)                                  # (nrow, npol)
    amp = np.abs(v).astype(np.float64)
    asum = amp.sum(axis=1)
    psum = (amp ** 2).sum(axis=1)
    count = valid.sum(axis=1).astype(np.float64)
    # Second differences across channels: a linear phase slope (residual delay) cancels.
    d2valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    d2 = np.where(d2valid, v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2], zero)
    d2sum = (np.abs(d2).astype(np.float64) ** 2).sum(axis=1)
    d2n = d2valid.sum(axis=1).astype(np.float64)
    weight = np.asarray(block["WEIGHT"], dtype=np.float64)[keep] if "WEIGHT" in block else \
        np.full((int(keep.sum()), len(corrs)), np.nan)
    if weight.ndim == 2 and weight.shape[1] != len(corrs):
        weight = weight[:, corrs]
    weight = np.broadcast_to(weight.reshape(weight.shape[0], -1), count.shape)
    used = count > 0
    wsum = np.where(used, weight, 0.0)
    wn = used.astype(np.float64)

    time = np.asarray(block["TIME"], dtype=np.float64)[keep]
    uvw = np.asarray(block["UVW"], dtype=np.float64)[keep]
    scan = np.asarray(block["SCAN_NUMBER"], dtype=np.int64)[keep]
    field = np.asarray(block["FIELD_ID"], dtype=np.int64)[keep]
    ddid = np.asarray(block["DATA_DESC_ID"], dtype=np.int64)[keep]
    low, high = np.minimum(ant1[keep], ant2[keep]), np.maximum(ant1[keep], ant2[keep])
    tbin = np.clip(np.floor((time - t0) / binsec).astype(np.int64), 0, _MAX_BINS - 1)
    keys = ((scan * (nant * nant) + low * nant + high) * nddid + ddid) * _MAX_BINS + tbin
    order = np.lexsort((time, keys))             # by cell, then in time within the cell
    unique, starts = np.unique(keys[order], return_index=True)
    rows_used = np.ones(order.size)          # time and uvw: every row of the cell, flagged or not

    def reduce(values):
        return np.add.reduceat(values[order], starts, axis=0)

    out = {"keys": unique, "field": field[order][starts],
           "vsum": reduce(vsum), "asum": reduce(asum), "psum": reduce(psum), "d2sum": reduce(d2sum),
           "d2n": reduce(d2n), "count": reduce(count), "wsum": reduce(wsum), "wn": reduce(wn),
           "tsum": np.add.reduceat(time[order], starts),
           "uvwsum": np.add.reduceat(uvw[order], starts, axis=0),
           "rows": np.add.reduceat(rows_used, starts)}
    out.update(_time_differences(v[order], valid[order], keys[order], unique))
    return out


def _time_differences(v: np.ndarray, valid: np.ndarray, keys: np.ndarray, unique: np.ndarray) -> dict:
    """Noise moments from consecutive integrations of each cell (rows sorted by cell, then time).

    Per cell and hand: ``t1sum``/``t1n`` (``|V[t+1] - V[t]|^2`` summed over channels, and how many), ``t1c``/
    ``t1cn`` (``Re(d[c] conj d[c+1])`` of those differences: the adjacent-channel noise covariance), and
    ``t2sum``/``t2n`` for the second differences ``V[t+1] - 2 V[t] + V[t-1]``.
    """
    npol = v.shape[2]
    out = {name: np.zeros((unique.size, npol)) for name in ("t1sum", "t1n", "t1c", "t1cn", "t2sum", "t2n")}
    if v.shape[0] < 2:
        return out
    same1 = keys[1:] == keys[:-1]
    d1 = v[1:] - v[:-1]
    ok1 = valid[1:] & valid[:-1] & same1[:, None, None]
    d1 = np.where(ok1, d1, 0)
    cell1 = np.searchsorted(unique, keys[1:])
    p1 = (np.abs(d1).astype(np.float64) ** 2).sum(axis=1)
    n1 = ok1.sum(axis=1).astype(np.float64)
    okc = ok1[:, 1:] & ok1[:, :-1]
    c1 = np.where(okc, (d1[:, 1:] * np.conj(d1[:, :-1])).real, 0).astype(np.float64).sum(axis=1)
    nc = okc.sum(axis=1).astype(np.float64)
    for name, values in (("t1sum", p1), ("t1n", n1), ("t1c", c1), ("t1cn", nc)):
        np.add.at(out[name], cell1, values)
    if v.shape[0] >= 3:
        same2 = same1[1:] & same1[:-1]
        ok2 = valid[2:] & valid[1:-1] & valid[:-2] & same2[:, None, None]
        d2 = np.where(ok2, v[2:] - 2 * v[1:-1] + v[:-2], 0)
        cell2 = np.searchsorted(unique, keys[2:])
        np.add.at(out["t2sum"], cell2, (np.abs(d2).astype(np.float64) ** 2).sum(axis=1))
        np.add.at(out["t2n"], cell2, ok2.sum(axis=1).astype(np.float64))
    return out


def visibility_stats(ms, *, column: str = "DATA", fields: Optional[list[str]] = None, binsec: float = 30.0,
                     channel_fraction: float = 0.8, flag_version: str = "", chain: Optional[list] = None,
                     workers: Optional[int] = None) -> dict:
    """The statistics cube of ``column`` (see the module docstring), as a dict of flat arrays.

    Parameters
    ----------
    ms : path
        Measurement set (or Multi-MS).
    column : str
        ``"DATA"`` or ``"CORRECTED_DATA"`` (a deferred calibration is applied while reading).
    fields : list of str, optional
        Field names to read (default: all).
    binsec : float
        Time-bin width in seconds; bins never straddle scans.
    channel_fraction : float
        Central fraction of the channels of every subband that is used.
    flag_version : str, optional
        Read the flags from this saved flag version (``<ms>.flagversions/flags.<name>``) instead of FLAG:
        the pre-diagnostic looks at the data with the a-priori flags only, whatever the pipeline flagged since.
    chain : list of dict, optional
        Calibrate DATA on the fly with these tables instead of reading ``column`` (``{"path", "interp",
        "gainfield"}`` per table, in application order): the cube of any partial chain, which is how a
        problem is traced to the table that introduced it (:func:`chain_bisection`).

    Returns
    -------
    dict
        Per cell: ``field``, ``scan``, ``ant1``, ``ant2``, ``spw``, ``time``, ``uvw`` (n, 3), and per hand
        ``(n, 2)``: ``vec`` (complex), ``scalar``, ``power``, ``noise2``, ``count``, ``weight``. Plus the
        metadata ``antennas``, ``field_names``, ``spw_freq`` (Hz, centre of each subband), ``hands``,
        ``column``, ``binsec``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    column = {"data": "DATA", "corrected": "CORRECTED_DATA"}.get(str(column).lower(), str(column).upper())
    field_ids = None
    if fields:
        field_ids = {setup["field_names"].index(name) for name in fields if name in setup["field_names"]}
    selected = select_runs(layout, fields=field_ids)
    if not selected.size:
        raise ValueError(f"{ms}: no data for fields {fields}")
    if chain is not None:
        column = "CORRECTED_DATA"
        view = _chain_view(chain, field_ids if field_ids is not None else set(range(len(setup["field_names"]))),
                           setup, layout["columns"])
    else:
        view = corrected_view(ms, column, field_ids, setup=setup, columns=layout["columns"])
    ncorr = int(setup["ncorr"])
    corrs = [0, ncorr - 1] if ncorr > 1 else [0]
    nchan = int(setup["chan_freq"].shape[1])
    chans = _central_channels(nchan, channel_fraction)
    nddid = len(setup["ddid_to_spw"])
    chans_by_ddid = {d: chans for d in range(nddid)}
    nant = len(setup["antenna_names"])
    t0 = float(layout["t_min"][selected].min())
    version = None
    if flag_version:
        from ..solvers.flagmanager import _parts, versions_dir
        path = versions_dir(ms) / f"flags.{flag_version}"
        if not path.is_dir():
            raise ValueError(f"{ms}: no flag version {flag_version!r}")
        offsets, row = {}, 0
        for part in _parts(ms):
            offsets[str(Path(part).resolve())] = row
            row += _nrows(part)
        version = {"path": str(path), "offsets": offsets}
    jobs = [{"specs": chunk, "column": column, "engine": engine, "ncorr": ncorr, "nant": nant, "nddid": nddid,
             "corrs": corrs, "chans_by_ddid": chans_by_ddid, "binsec": float(binsec), "t0": t0, "view": view,
             "flag_version": version}
            for chunk in split_rows(run_specs(layout, selected), STATS_ROWS)]
    parts = [p for p in run_jobs("vlbipy.ai.visstats:stats_job", jobs, workers, desc="visibility statistics") if p]
    if not parts:
        raise ValueError(f"{ms}: no unflagged cross-correlations in {column}")
    keys = np.concatenate([p["keys"] for p in parts])
    unique, inverse = np.unique(keys, return_inverse=True)

    def merge(name):
        stacked = np.concatenate([p[name] for p in parts])
        out = np.zeros((unique.size,) + stacked.shape[1:], dtype=stacked.dtype)
        np.add.at(out, inverse, stacked)
        return out

    sums = {name: merge(name) for name in _SUMMED + ("tsum", "uvwsum", "rows")}
    field = np.zeros(unique.size, dtype=np.int64)
    field[inverse] = np.concatenate([p["field"] for p in parts])
    rest, tbin = np.divmod(unique, _MAX_BINS)
    rest, ddid = np.divmod(rest, nddid)
    scan, baseline = np.divmod(rest, nant * nant)
    ant1, ant2 = np.divmod(baseline, nant)
    with np.errstate(invalid="ignore", divide="ignore"):
        count = sums["count"]
        rows = np.maximum(sums["rows"], 1)
        cube = {"field": field, "scan": scan, "ant1": ant1, "ant2": ant2,
                "spw": np.asarray(setup["ddid_to_spw"])[ddid], "tbin": tbin,
                "time": sums["tsum"] / rows, "uvw": sums["uvwsum"] / rows[:, None],
                "vec": np.where(count > 0, sums["vsum"] / np.maximum(count, 1), np.nan + 0j),
                "scalar": np.where(count > 0, sums["asum"] / np.maximum(count, 1), np.nan),
                "power": np.where(count > 0, sums["psum"] / np.maximum(count, 1), np.nan),
                "noise2": _noise2(sums, column),
                "count": count,
                "weight": np.where(sums["wn"] > 0, sums["wsum"] / np.maximum(sums["wn"], 1), np.nan)}
    with np.errstate(invalid="ignore", divide="ignore"):
        # Adjacent-channel noise correlation: covariance over variance of the time differences, per cell,
        # and its median per hand over the cube (a property of the correlator, noisy in any one cell).
        per_cell = (sums["t1c"] / sums["t1cn"]) / (sums["t1sum"] / sums["t1n"])
        enough = sums["t1cn"] >= 50
        rho1 = np.array([np.nanmedian(per_cell[enough[:, h], h]) if enough[:, h].any() else 0.0
                         for h in range(count.shape[1])])
        rho1 = np.clip(np.nan_to_num(rho1), 0.0, 0.95)
        cube["noise2_avg"] = cube["noise2"] * (1.0 + 2.0 * rho1[None, :]) / np.maximum(count, 1)
        cube["noise2_avg"][count == 0] = np.nan
    cube["rho1"] = rho1
    spw_freq = np.asarray(setup["chan_freq"], dtype=np.float64).mean(axis=1)
    cube.update({"antennas": np.array(setup["antenna_names"]), "field_names": np.array(setup["field_names"]),
                 "spw_freq": spw_freq, "hands": np.array(["RR", "LL"] if len(corrs) == 2 else ["RR"]),
                 "column": np.array(column), "binsec": np.array(float(binsec)),
                 "chan_width": np.array(float(np.median(np.abs(np.diff(setup["chan_freq"], axis=1)))))})
    logger.info("visibility_stats: {} cells of {} ({} field(s)) in {:.1f} s", unique.size, column,
                len(np.unique(field)), _time.perf_counter() - started)
    return cube


def _noise2(sums: dict, column: str) -> np.ndarray:
    """Per-sample noise variance of every cell from its time differences (see the module docstring).

    Raw DATA uses the second differences (a residual fringe rate cancels), calibrated data the first ones
    (twice the samples). A cell with too few integrations for either takes its channel-difference estimate,
    rescaled by the cube-wide ratio of the two estimators (the channel correlation is a property of the
    correlator, the same everywhere).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        first = np.where(sums["t1n"] > 0, sums["t1sum"] / (2.0 * sums["t1n"]), np.nan)
        second = np.where(sums["t2n"] > 0, sums["t2sum"] / (6.0 * sums["t2n"]), np.nan)
        chan = np.where(sums["d2n"] > 0, sums["d2sum"] / (6.0 * sums["d2n"]), np.nan)
        primary = second if column == "DATA" else first
        primary = np.where(np.isfinite(primary), primary, first)
        ratio = np.nanmedian(primary / chan, axis=0) if np.isfinite(primary / chan).any() else np.ones(chan.shape[1])
        return np.where(np.isfinite(primary), primary, chan * np.nan_to_num(ratio, nan=1.0)[None, :])


def _chain_view(chain: list, field_ids, setup: dict, columns: list) -> dict:
    """A :func:`vlbipy.solvers.corrected.view`-like dict applying ``chain`` to DATA for ``field_ids``."""
    from ..solvers.apply import normalise_entries
    entries = [{"path": str(e["path"]), "interp": e.get("interp", "linear"), "spwmap": list(e.get("spwmap", [])),
                "gainfield": [setup["field_names"].index(n) for n in e["gainfield"].split(",")]
                if e.get("gainfield") else [], "calwt": False} for e in chain]
    cache: dict = {}
    return {"entries_by_field": {int(f): normalise_entries(entries, calwt=False, cache=cache,
                                                           field_dirs=setup["field_dirs"], target_field=int(f))
                                 for f in field_ids},
            "parang": {int(f): True for f in field_ids}, "write_flags": {int(f): True for f in field_ids},
            "weights": {int(f): False for f in field_ids}, "setup": setup, "columns": list(columns)}


def _nrows(table_path) -> int:
    from ..casacore_format.table import Table
    with Table(table_path) as table:
        return int(table.nrows())


def save(cube: dict, path) -> Path:
    """Write a cube to ``path`` (``.npz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cube)
    return path


def load(path) -> dict:
    """Read a cube written by :func:`save`."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def amplitude(cube: dict, *, coherent: Optional[bool] = None) -> np.ndarray:
    """Per-cell, per-hand amplitude estimate.

    ``coherent=True``: ``|vec|`` debiased for the noise of the average (calibrated data);
    ``coherent=False``: ``sqrt(power - noise2)``, independent of the phases (raw data). Default: coherent
    for CORRECTED_DATA, incoherent for DATA.
    """
    if coherent is None:
        coherent = str(cube.get("column", "")) == "CORRECTED_DATA"
    if "noise2_avg" not in cube:          # a cube written before the channel correlation was accounted for
        cube["noise2_avg"] = cube["noise2"] / np.maximum(cube["count"], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if coherent:
            biased = np.abs(cube["vec"]) ** 2 - cube["noise2_avg"]
        else:
            biased = cube["power"] - cube["noise2"]
        return np.sqrt(np.clip(biased, 0.0, None))


def snr(cube: dict) -> np.ndarray:
    """Per-cell, per-hand signal-to-noise of the bin average (``amplitude / sqrt(noise2_avg)``)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return amplitude(cube) / np.sqrt(cube["noise2_avg"])


def sigma(cube: dict) -> np.ndarray:
    """Per-cell, per-hand standard deviation of the complex bin average ``vec`` (sqrt of ``noise2_avg``)."""
    return np.sqrt(cube["noise2_avg"])
