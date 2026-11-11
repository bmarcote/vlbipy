"""Smart (antenna-based) outlier flagging of a measurement set: ``vlbipy flag smart-outliers``.

The statistics live in :mod:`difmapy.outliers` (shared with difmapy's Shift+S selection): per source, every
baseline, subband, channel block and parallel hand is expected to follow a smooth trend; the cells that depart
from it by more than ``sigma`` are traced to the antenna common to most of them, and that antenna is flagged
over the time and frequency range the departure spans, its rising and decaying edges included. This module is
the measurement-set side:

* **Grids.** The rows of the wanted fields are read in parallel worker processes (row blocks of the table
  parts, :func:`vlbipy.solvers.msio.split_rows`), the parallel hands only, CORRECTED_DATA computed on the fly
  where the calibration was deferred (:func:`vlbipy.solvers.applycal.read_view_runs`). Every block is reduced
  at once to sums and counts per (baseline, subband, time bin, channel block, hand); the main process adds the
  partial sums into one :class:`difmapy.outliers.VisGrid` per source. Time bins are ``timebin`` seconds,
  counted from the start of every scan, so a bin never straddles two scans.
* **Commands.** The masks become ``flagdata`` list commands: one per antenna (or baseline), run of time bins
  inside a scan and set of (subband, channel range) flagged, ``correlation`` limited to the hand when only one
  parallel hand is bad (with both cross hands, which carry it too).
* **Phase-referencing.** An antenna flagged over (nearly) a whole scan of a phase calibrator leaves the target
  scans next to it without a solution of their own: those target scans are flagged for that antenna too
  (``full_scan_fraction``).

:func:`smart_outliers` returns the commands and a report; the caller applies them (``flagdata(mode='list')``).
"""
from __future__ import annotations

import datetime as _dt
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .applycal import read_view_runs
from .corrected import view as corrected_view
from .msio import open_table, read_layout, read_setup, run_specs, select_runs, split_rows, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows per worker job: 100k rows x 64 channels x 2 hands of complex64 is ~100 MB.
GRID_ROWS = 100_000
#: CASA's Stokes enumeration of the correlation products this module knows.
CORR_LABELS = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
#: Parallel hands, and the cross hands that share a receptor with each.
PARALLEL = {"RR": ("RL", "LR"), "LL": ("RL", "LR"), "XX": ("XY", "YX"), "YY": ("XY", "YX")}
#: Data columns: ``column`` argument -> main-table column.
DATA_COLUMNS = {"data": "DATA", "corrected": "CORRECTED_DATA"}
_EPOCH = _dt.datetime(1858, 11, 17)
#: Margin (s) around a time range, below the sampling of any VLBI correlator.
_EDGE = 0.001


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def correlation_labels(ms, engine: str) -> list[str]:
    """Labels (``RR``, ``RL``, ...) of the correlation products of the measurement set, in column order."""
    handle = open_table(Path(ms) / "POLARIZATION", engine=engine)
    try:
        types = np.asarray(handle.getcol("CORR_TYPE")).reshape(handle.nrows(), -1)[0]
    finally:
        handle.close()
    return [CORR_LABELS.get(int(t), str(int(t))) for t in types]


def scan_table(layout: dict, field_ids: Optional[set] = None) -> list[dict]:
    """Every (field, scan) of the layout with its first and last sample time, in time order.

    Returns
    -------
    list of dict
        ``field``, ``scan``, ``t_min``, ``t_max`` [MJD s].
    """
    keys: dict[tuple[int, int], list[float]] = {}
    for field, scan, t_min, t_max in zip(layout["field"], layout["scan"], layout["t_min"], layout["t_max"]):
        if field_ids is not None and int(field) not in field_ids:
            continue
        span = keys.setdefault((int(field), int(scan)), [float(t_min), float(t_max)])
        span[0], span[1] = min(span[0], float(t_min)), max(span[1], float(t_max))
    rows = [{"field": f, "scan": s, "t_min": span[0], "t_max": span[1]} for (f, s), span in keys.items()]
    return sorted(rows, key=lambda row: (row["t_min"], row["field"], row["scan"]))


def time_bins(scans: list[dict], timebin: float) -> dict:
    """The time axis of one source: ``timebin``-second bins counted from the start of each of its ``scans``.

    Returns
    -------
    dict
        ``scans`` (``{scan: (t0, first bin, number of bins)}``), ``start``/``stop`` (per bin [MJD s]: the time
        range a flag of the bin covers), ``times`` (bin centres), ``segment`` (scan number per bin), ``ntime``.
    """
    table, starts, stops, segment = {}, [], [], []
    offset = 0
    for row in sorted(scans, key=lambda r: r["t_min"]):
        t0, t1 = float(row["t_min"]), float(row["t_max"])
        nbins = int(np.floor((t1 - t0) / timebin)) + 1
        table[int(row["scan"])] = (t0, offset, nbins)
        edges = t0 + timebin * np.arange(nbins + 1)
        starts.append(np.concatenate([[t0 - _EDGE], edges[1:-1]]))
        stops.append(np.concatenate([edges[1:-1] - _EDGE, [t1 + _EDGE]]))
        segment.append(np.full(nbins, int(row["scan"]), dtype=np.int64))
        offset += nbins
    start = np.concatenate(starts) if starts else np.zeros(0)
    stop = np.concatenate(stops) if stops else np.zeros(0)
    return {"scans": table, "start": start, "stop": stop, "times": 0.5 * (start + stop),
            "segment": np.concatenate(segment) if segment else np.zeros(0, dtype=np.int64), "ntime": offset}


def channel_blocks(nchan: int, blocks: int) -> np.ndarray:
    """First channel of each of ``blocks`` near-equal channel blocks of a subband (at most one per channel)."""
    blocks = max(1, min(int(blocks), nchan))
    return np.unique((np.arange(blocks) * nchan) // blocks).astype(np.int64)


# ---------------------------------------------------------------------------
# Worker job: one row block -> partial sums per cell
# ---------------------------------------------------------------------------
def grid_job(*, specs: list[dict], column: str, corrs: list[int], ncorr: int, nchan: int, engine: str,
             view: Optional[dict], field: int, scans: dict, timebin: float, ntime: int, block_starts: list[int],
             pair_index: list[list[int]], nddid: int) -> dict:
    """Worker job: sums and counts of the unflagged samples of the rows ``specs`` per grid cell.

    Parameters
    ----------
    specs : list of dict
        Row runs (:func:`vlbipy.solvers.msio.run_specs`), all of field ``field``.
    column : str
        DATA or CORRECTED_DATA.
    corrs : list of int
        Correlation indices to read (the parallel hands).
    scans : dict
        ``{scan: (t0, first bin, number of bins)}`` (:func:`time_bins`).
    block_starts : list of int
        First channel of every channel block.
    pair_index : list of list of int
        ``[a][b]`` grid baseline index of antennas ``a < b`` (-1: none).

    Returns
    -------
    dict
        ``keys`` (cell index ``(baseline * nddid + ddid) * ntime + bin``), ``sums`` complex128
        ``(n, nblock, ncorr)``, ``counts`` float64 (same shape); empty when no row has data.
    """
    block = read_view_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine, view=view,
                           corrs=np.asarray(corrs, dtype=np.int64), nchan=nchan, ncorr=ncorr)
    if not block:
        return {}
    values = np.asarray(block.pop(column))
    flags = np.asarray(block.pop("FLAG"), dtype=bool)
    if values.shape[-1] != len(corrs):                   # a reader that ignored the correlation selection
        values, flags = values[:, :, corrs], flags[:, :, corrs]
    usable = ~flags & np.isfinite(values) & (values != 0)          # a never-recorded subband is zero-filled
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    pairs = np.asarray(pair_index, dtype=np.int64)
    low, high = np.minimum(ant1, ant2), np.maximum(ant1, ant2)
    baseline = np.where(low != high, pairs[low, high], -1)
    times = np.asarray(block["TIME"], dtype=np.float64)
    scan_numbers = np.asarray(block["SCAN_NUMBER"], dtype=np.int64)
    bins = np.full(times.size, -1, dtype=np.int64)
    for scan in np.unique(scan_numbers):
        entry = scans.get(int(scan))
        if entry is None:
            continue
        t0, first, nbins = entry
        rows = scan_numbers == scan
        bins[rows] = first + np.clip(np.floor((times[rows] - t0) / timebin).astype(np.int64), 0, nbins - 1)
    keep = (baseline >= 0) & (bins >= 0) & usable.any(axis=(1, 2))
    if not keep.any():
        return {}
    starts = np.asarray(block_starts, dtype=np.int64)
    sums = np.add.reduceat(np.where(usable[keep], values[keep], 0).astype(np.complex128), starts, axis=1)
    counts = np.add.reduceat(usable[keep].astype(np.float64), starts, axis=1)
    keys = (baseline[keep] * nddid + np.asarray(block["DATA_DESC_ID"], dtype=np.int64)[keep]) * ntime + bins[keep]
    order = np.argsort(keys, kind="stable")
    keys, sums, counts = keys[order], sums[order], counts[order]
    unique, first_rows = np.unique(keys, return_index=True)
    return {"keys": unique, "sums": np.add.reduceat(sums, first_rows, axis=0),
            "counts": np.add.reduceat(counts, first_rows, axis=0)}


def build_grid(ms, *, layout: dict, setup: dict, field: int, scans: list[dict], column: str, corrs: list[int],
               timebin: float, blocks: int, engine: str, workers: Optional[int] = None) -> dict:
    """Bin one source's visibilities into a :class:`difmapy.outliers.VisGrid`.

    Returns
    -------
    dict
        ``grid`` (VisGrid), ``bins`` (:func:`time_bins`), ``block_starts`` and ``rows`` read.
    """
    from difmapy.outliers import VisGrid
    nant, nddid = len(setup["antenna_names"]), len(setup["ddid_to_spw"])
    nchan = int(setup["chan_freq"].shape[1])
    pairs = [(a, b) for a in range(nant) for b in range(a + 1, nant)]
    pair_index = np.full((nant, nant), -1, dtype=np.int64)
    for index, (a, b) in enumerate(pairs):
        pair_index[a, b] = index
    bins = time_bins(scans, timebin)
    starts = channel_blocks(nchan, blocks)
    ntime, nblock = bins["ntime"], starts.size
    sums = np.zeros((len(pairs) * nddid * ntime, nblock, len(corrs)), dtype=np.complex64)
    counts = np.zeros(sums.shape, dtype=np.float32)
    selected = select_runs(layout, fields={field})
    specs = run_specs(layout, selected)
    view = corrected_view(ms, column, [field], setup=setup, columns=layout["columns"])
    jobs = [{"specs": chunk, "column": column, "corrs": list(corrs), "ncorr": int(setup["ncorr"]), "nchan": nchan,
             "engine": engine, "view": view, "field": field, "scans": bins["scans"], "timebin": float(timebin),
             "ntime": ntime, "block_starts": starts.tolist(), "pair_index": pair_index.tolist(), "nddid": nddid}
            for chunk in split_rows(specs, GRID_ROWS)]
    for part in run_jobs("vlbipy.solvers.smart_outliers:grid_job", jobs, workers):
        if part:
            sums[part["keys"]] += part["sums"].astype(np.complex64)
            counts[part["keys"]] += part["counts"].astype(np.float32)
    shape = (len(pairs), nddid, ntime, nblock, len(corrs))
    sums, counts = sums.reshape(shape), counts.reshape(shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = np.where(counts > 0, sums / np.maximum(counts, 1), 0).astype(np.complex64)
    grid = VisGrid(np.moveaxis(vis, 2, -1), np.moveaxis(counts, 2, -1),
                   np.array([a for a, _ in pairs], dtype=np.int64), np.array([b for _, b in pairs], dtype=np.int64),
                   bins["times"], bins["segment"], nant)
    return {"grid": grid, "bins": bins, "block_starts": starts, "rows": int(sum(s["nrow"] for s in specs))}


# ---------------------------------------------------------------------------
# Masks -> flagdata commands
# ---------------------------------------------------------------------------
def stamp(mjd_seconds: float) -> str:
    """CASA time string (``YYYY/MM/DD/hh:mm:ss.sss``) of an MJD in seconds, truncated to the millisecond."""
    moment = _EPOCH + _dt.timedelta(seconds=float(mjd_seconds))
    return moment.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3]


def spw_selection(mask: np.ndarray, block_starts: np.ndarray, nchan: int, ddid_to_spw) -> str:
    """flagdata ``spw`` of a ``(nddid, nblock)`` mask: whole subbands by number, others with channel ranges."""
    pieces = []
    ends = np.concatenate([block_starts[1:], [nchan]]) - 1
    for ddid in range(mask.shape[0]):
        flagged = np.flatnonzero(mask[ddid])
        if not flagged.size:
            continue
        spw = int(ddid_to_spw[ddid])
        if flagged.size == mask.shape[1]:
            pieces.append(str(spw))
            continue
        runs = np.split(flagged, np.flatnonzero(np.diff(flagged) != 1) + 1)
        ranges = ";".join(f"{block_starts[run[0]]}~{ends[run[-1]]}" for run in runs)
        pieces.append(f"{spw}:{ranges}")
    return ",".join(pieces)


def _hand_correlations(hand: str, labels: list[str]) -> str:
    """The correlations to flag for one bad receptor: its parallel hand and the cross hands present."""
    return ",".join([hand] + [c for c in PARALLEL.get(hand, ()) if c in labels])


def mask_commands(mask: np.ndarray, *, selector: str, field_name: str, bins: dict, block_starts: np.ndarray,
                  nchan: int, ddid_to_spw, hands: list[str], labels: list[str]) -> list[str]:
    """flagdata commands of one antenna's (or baseline's) flags ``(nddid, nblock, nhand, ntime)``.

    Consecutive time bins of one scan with the same flagged subbands, channel blocks and hands make one command.
    Both hands alike: no correlation selection (every product); one hand only: that hand and the cross hands.
    """
    commands: list[str] = []
    ntime = mask.shape[-1]
    if not mask.any():
        return commands
    signatures = [mask[..., t].tobytes() if mask[..., t].any() else b"" for t in range(ntime)]
    segment = bins["segment"]
    t = 0
    while t < ntime:
        signature = signatures[t]
        if not signature:
            t += 1
            continue
        last = t
        while last + 1 < ntime and signatures[last + 1] == signature and segment[last + 1] == segment[t]:
            last += 1
        cells = mask[..., t]                                          # (nddid, nblock, nhand)
        timerange = f"{stamp(bins['start'][t])}~{stamp(bins['stop'][last])}"
        base = f"{selector} field='{field_name}' timerange='{timerange}'"
        if len(hands) == 1 or all(np.array_equal(cells[..., 0], cells[..., h]) for h in range(1, len(hands))):
            spw = spw_selection(cells.any(axis=-1), block_starts, nchan, ddid_to_spw)
            commands.append(f"{base} spw='{spw}'")
        else:
            for h, hand in enumerate(hands):
                if cells[..., h].any():
                    spw = spw_selection(cells[..., h], block_starts, nchan, ddid_to_spw)
                    commands.append(f"{base} spw='{spw}' correlation='{_hand_correlations(hand, labels)}'")
        t = last + 1
    return commands


def full_scan_hands(result, grid, antenna: int, scan_bins: np.ndarray, fraction: float) -> list[int]:
    """Hands (indices) in which ``antenna`` is flagged over at least ``fraction`` of its data in the bins ``scan_bins``."""
    mine = (grid.ant1 == antenna) | (grid.ant2 == antenna)
    has = (grid.count[mine][..., scan_bins] > 0).any(axis=0)          # (nddid, nblock, nhand, nbin)
    flagged = result.antenna_flags[antenna][..., scan_bins] & has
    out = []
    for hand in range(has.shape[2]):
        total = int(has[:, :, hand].sum())
        if total and flagged[:, :, hand].sum() >= fraction * total:
            out.append(hand)
    return out


def receiver_commands(*, result, grid, bins: dict, field: int, all_scans: list[dict], receivers: set,
                      antenna_names: list[str], field_names: list[str], hands: list[str], labels: list[str],
                      fraction: float) -> tuple[list[str], list[dict]]:
    """Flags for the target scans next to a phase-calibrator scan an antenna lost (nearly) entirely.

    The phase calibrator's solutions reach the target scans between its neighbouring scans by interpolation:
    without the scan in the middle they come from twice as far, so those target scans are flagged for that
    antenna (in the hands it lost) rather than calibrated by extrapolation.

    Returns
    -------
    (list of str, list of dict)
        The commands, and one record per (antenna, calibrator scan) extended.
    """
    own = [row for row in all_scans if row["field"] == field]
    commands, records = [], []
    for position, row in enumerate(own):
        entry = bins["scans"].get(row["scan"])
        if entry is None:
            continue
        _, first, nbins = entry
        scan_bins = np.arange(first, first + nbins)
        previous = own[position - 1]["t_max"] if position > 0 else -np.inf
        following = own[position + 1]["t_min"] if position + 1 < len(own) else np.inf
        neighbours = [r for r in all_scans if r["field"] in receivers and r["t_min"] > previous
                      and r["t_max"] < following]
        if not neighbours:
            continue
        for antenna in range(grid.nant):
            lost = full_scan_hands(result, grid, antenna, scan_bins, fraction)
            if not lost:
                continue
            correlation = "" if len(lost) == len(hands) else \
                ",".join(dict.fromkeys(c for h in lost for c in _hand_correlations(hands[h], labels).split(",")))
            for target in neighbours:
                command = (f"antenna='{antenna_names[antenna]}' field='{field_names[target['field']]}' "
                           f"timerange='{stamp(target['t_min'] - _EDGE)}~{stamp(target['t_max'] + _EDGE)}'")
                commands.append(command + (f" correlation='{correlation}'" if correlation else ""))
            records.append({"antenna": antenna_names[antenna], "scan": row["scan"],
                            "hands": [hands[h] for h in lost],
                            "targets": [f"{field_names[t['field']]}:{t['scan']}" for t in neighbours]})
    return commands, records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def smart_outliers(ms, *, fields: list[str], column: str = "data", settings=None, timebin: float = 15.0,
                   blocks: int = 8, receivers: Optional[dict] = None, full_scan_fraction: float = 0.9,
                   workers: Optional[int] = None) -> dict:
    """Find the outliers of every field of ``fields`` and the flag commands that remove them (nothing is written).

    Parameters
    ----------
    ms : str or pathlib.Path
        Measurement set or Multi-MS.
    fields : list of str
        Field names, each modelled on its own.
    column : str
        ``'data'`` (raw) or ``'corrected'`` (a deferred calibration is applied while reading).
    settings : difmapy.outliers.OutlierSettings or dict, optional
        Model settings (defaults otherwise). A dict without ``align_phases`` / ``use_phases`` aligns the phases of
        raw data (``column='data'``: not fringe-fitted, they rotate within every scan) and tests their amplitude
        only (their antenna phases wander until calibrated); corrected data are tested as they are, in full.
    timebin : float
        Time bin [s].
    blocks : int
        Channel blocks per subband.
    receivers : dict, optional
        ``{phase calibrator: [fields it calibrates]}``: an antenna flagged over (nearly) a whole scan of the
        calibrator is flagged in the scans of those fields next to it.
    full_scan_fraction : float
        Share of an antenna's data in a calibrator scan that counts as the whole scan.
    workers : int, optional
        Worker processes for the reads.

    Returns
    -------
    dict
        ``commands`` (flagdata list commands), ``fields`` (per field: ``rows``, ``cells``, ``flagged`` cells,
        ``fraction``, ``iterations``, ``antennas`` ``{name: fraction of its cells}``, ``commands``,
        ``extended`` records), ``settings``, ``timebin``, ``blocks``, ``column``, ``seconds``.
    """
    from difmapy.outliers import OutlierSettings, find_outliers
    started = _time.perf_counter()
    if column not in DATA_COLUMNS:
        raise ValueError(f"smart_outliers: column must be one of {sorted(DATA_COLUMNS)}, not {column!r}")
    if not timebin > 0:
        raise ValueError(f"smart_outliers: timebin must be positive, not {timebin}")
    if not isinstance(settings, OutlierSettings):
        raw = column == "data"
        settings = OutlierSettings.from_mapping({"align_phases": raw, "use_phases": not raw, **dict(settings or {})})
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    labels = correlation_labels(ms, engine)
    corrs = [i for i, label in enumerate(labels) if label in PARALLEL]
    if not corrs:
        raise ValueError(f"smart_outliers: {Path(ms).name} has no parallel-hand correlation ({labels})")
    hands = [labels[i] for i in corrs]
    names, antenna_names = setup["field_names"], setup["antenna_names"]
    nchan = int(setup["chan_freq"].shape[1])
    all_scans = scan_table(layout)
    report: dict = {"commands": [], "fields": {}, "settings": settings.to_dict(), "timebin": float(timebin),
                    "blocks": int(blocks), "column": column}
    for name in fields:
        if name not in names:
            logger.warning("smart outliers: no field {} in {}; skipped", name, Path(ms).name)
            continue
        field = names.index(name)
        scans = [row for row in all_scans if row["field"] == field]
        if not scans:
            logger.info("smart outliers: {} has no data; skipped", name)
            continue
        read_started = _time.perf_counter()
        built = build_grid(ms, layout=layout, setup=setup, field=field, scans=scans, column=DATA_COLUMNS[column],
                           corrs=corrs, timebin=timebin, blocks=blocks, engine=engine, workers=workers)
        grid, bins = built["grid"], built["bins"]
        read_seconds = _time.perf_counter() - read_started
        model_started = _time.perf_counter()
        result = find_outliers(grid, settings)
        model_seconds = _time.perf_counter() - model_started
        commands = []
        for antenna in range(grid.nant):
            commands += mask_commands(result.antenna_flags[antenna], selector=f"antenna='{antenna_names[antenna]}'",
                                      field_name=name, bins=bins, block_starts=built["block_starts"], nchan=nchan,
                                      ddid_to_spw=setup["ddid_to_spw"], hands=hands, labels=labels)
        for baseline in np.flatnonzero(result.baseline_flags.reshape(result.baseline_flags.shape[0], -1).any(axis=1)):
            pair = f"{antenna_names[grid.ant1[baseline]]}&{antenna_names[grid.ant2[baseline]]}"
            commands += mask_commands(result.baseline_flags[baseline], selector=f"antenna='{pair}'", field_name=name,
                                      bins=bins, block_starts=built["block_starts"], nchan=nchan,
                                      ddid_to_spw=setup["ddid_to_spw"], hands=hands, labels=labels)
        extended: list[dict] = []
        targets = {names.index(t) for t in (receivers or {}).get(name, []) if t in names}
        if targets:
            more, extended = receiver_commands(result=result, grid=grid, bins=bins, field=field, all_scans=all_scans,
                                               receivers=targets, antenna_names=antenna_names, field_names=names,
                                               hands=hands, labels=labels, fraction=full_scan_fraction)
            commands += more
        summary = result.summary(grid)
        report["fields"][name] = {
            "rows": built["rows"], "cells": summary["cells"], "flagged": summary["flagged"],
            "fraction": summary["fraction"], "iterations": summary["iterations"], "commands": len(commands),
            "antennas": {antenna_names[a]: round(v["fraction"], 5) for a, v in summary["antennas"].items()
                         if v["flagged"]},
            "extended": extended, "read_seconds": round(read_seconds, 2), "model_seconds": round(model_seconds, 2)}
        report["commands"] += commands
        worst = sorted(report["fields"][name]["antennas"].items(), key=lambda kv: -kv[1])[:5]
        logger.info("smart outliers: {} ({}) - {} row(s) read in {:.1f}s, model {:.1f}s ({} fit(s)): {} of {} cells "
                    "({:.2%}) to flag in {} command(s){}{}", name, column, built["rows"], read_seconds, model_seconds,
                    summary["iterations"], summary["flagged"], summary["cells"], summary["fraction"], len(commands),
                    "; worst " + ", ".join(f"{a} {f:.1%}" for a, f in worst) if worst else "",
                    f"; {len(extended)} calibrator scan(s) extended to the target" if extended else "")
    report["seconds"] = _time.perf_counter() - started
    return report
