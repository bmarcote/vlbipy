"""Apply calibration tables to a measurement set (the dask-ms backend's ``applycal``).

:func:`apply_to_ms` writes what ``casatasks.applycal`` writes — CORRECTED_DATA,
the calibration flags and the calibrated weights — into the measurement set
itself, so CASA flagging, imaging and export downstream are unaffected.

:func:`defer_to_ms` records the same application without writing
CORRECTED_DATA, only the calibration flags (see :mod:`vlbipy.solvers.corrected`).
The readers of the corrected column then correct DATA while they read it
(:func:`read_view_runs`, :func:`corrected_chunks`). Both paths go through the
same steps as :func:`apply_part_job` (:func:`row_grids`, :func:`correct_rows`),
so the readers see exactly the numbers applycal would have written.

The work is I/O bound (every visibility is read and written once), so it is
organised around the tables rather than the maths:

* one worker process per table **part** (the sub-MSs of a Multi-MS): different
  processes then read and write different tables with no lock between them. A
  plain MS has a single part, processed by a single worker whose arithmetic
  uses every core (:func:`vlbipy.solvers.workers.job_threads`);
* all requested fields are corrected in **one pass** over each part, each row
  block with the table chain of its own field;
* the whole chain is multiplied into one gain per antenna, time and channel
  before it touches the visibilities, at the points the rows use
  (:func:`row_grids`), and the rows are then corrected in one pass
  (:func:`correct_rows`).

Conventions follow CASA, checked table by table against ``applycal``: the
weights are rebuilt from SIGMA (``1 / sigma^2``, SIGMA_SPECTRUM when present)
and scaled by ``|g_i g_j|^2`` of the Tsys, gain-curve and gain tables with
``calwt`` (never by bandpasses), so applying twice does not calibrate them
twice; data whose solution is flagged or missing are flagged and get weight 0
(``applymode='calflag'``/``'calflagstrict'``).

``source='CORRECTED_DATA'`` applies the tables on top of the corrected column
instead of DATA (an incremental application, what ``vlbipy applycal`` does on
a set that already has one): the weights are then scaled from the WEIGHT
columns instead of being rebuilt from SIGMA.
"""
from __future__ import annotations

import tempfile
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .apply import apply_point_flags, calibrates_weights
from .msio import CUBE_COLUMNS, open_table, read_layout, read_runs, read_setup, select_runs, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows corrected per read/write block inside a worker (64 channels x 4 correlations: 40 MB of visibilities).
APPLY_ROWS = 20_000
#: Rows whose gains are evaluated together (see :func:`_pieces`): the table lookups are done once per batch and
#: (field, spw) instead of once per (scan, spw) run, which is only a few hundred rows of a VLBI scan.
BATCH_ROWS = 400_000
#: Rows per FLAG block of a deferred application (FLAG alone is small).
FLAG_ROWS = 200_000
#: ``applymode`` values this implementation covers.
APPLY_MODES = ("", "calflag", "calflagstrict", "calonly")


def _parang_for(parang, field: int) -> bool:
    """``parang`` of a field: one value for all, or ``{field id: bool}``."""
    return bool(parang.get(int(field), False)) if isinstance(parang, dict) else bool(parang)


def _pieces(segments: list[dict], max_rows: int) -> list[dict]:
    """Contiguous row ranges covering ``segments``, of at most ``max_rows`` rows, with the field and DDID of every row.

    Adjacent runs are read together whatever the order of fields and subbands in the table: a Multi-MS imported
    from FITS-IDI holds one subband after another (runs of hundreds of rows), but a split or an mstransform output
    interleaves them row by row.
    """
    pieces: list[dict] = []
    current: list[tuple] = []

    def flush() -> None:
        if current:
            counts = np.array([n for _, n, _, _ in current])
            pieces.append({"start": current[0][0], "nrow": int(counts.sum()),
                           "field": np.repeat([f for _, _, f, _ in current], counts).astype(np.int64),
                           "ddid": np.repeat([d for _, _, _, d in current], counts).astype(np.int64)})
            current.clear()

    filled = 0
    for segment in sorted(segments, key=lambda s: s["start"]):
        offset = 0
        while offset < segment["nrow"]:
            start = segment["start"] + offset
            if current and (current[-1][0] + current[-1][1] != start or filled >= max_rows):
                flush()
                filled = 0
            take = min(segment["nrow"] - offset, max_rows - filled)
            current.append((start, take, int(segment["field"]), int(segment["ddid"])))
            filled += take
            offset += take
    flush()
    return pieces


def row_grids(entries_by_field: dict, setup: dict, parang, time, antenna1, antenna2, field, ddid) -> tuple:
    """The gains a set of rows of any fields and subbands needs, and the feed angles of their fields.

    The gains are evaluated at the points the rows use, one antenna of one (field, DDID) at one time, each with
    the chain of its field (:func:`vlbipy.solvers.apply.point_grid`). Neither depends on the other rows: a point
    is evaluated on its own, and the feed angle of a time does not depend on the other times
    (:mod:`vlbipy.solvers.parang`). Rows corrected in any batch of rows that contains them come out the same.

    Parameters
    ----------
    parang : bool or dict
        Feed rotation for every field, or ``{field id: bool}``.
    time, antenna1, antenna2, field, ddid : arrays (nrow,)

    Returns
    -------
    (grid, angles)
        The point grid of all the rows, with ``point1``/``point2`` (nrow,), the points of every row's two
        antennas (:func:`rows_of` cuts it to a subset of the rows), and ``{field: (times, angles)}``.
    """
    from .apply import empty_grid, fill_grid, row_points, weight_channels
    from .parang import feed_angles_for_ms
    points, point1, point2 = row_points(antenna1, antenna2, time, field, ddid)
    point_field, point_ddid = points["groups"]
    spw = np.asarray(setup["ddid_to_spw"], dtype=np.int64)[point_ddid]
    chan_freq = np.asarray(setup["chan_freq"], dtype=np.float64)
    fields = np.unique(point_field).tolist()
    # One set of arrays for all fields; a field whose chain leaves the weights alone scales them by 1.
    weights = {weight_channels(entries_by_field[f], chan_freq.shape[1]) for f in fields} - {None}
    grid = empty_grid(point_field.size, chan_freq.shape[1], np.complex64, np.float32, max(weights, default=None))
    bounds = np.searchsorted(point_field, fields + [max(fields, default=0) + 1])
    angles = {}
    for index, field_id in enumerate(fields):
        block = slice(int(bounds[index]), int(bounds[index + 1]))
        spws, spw_index = np.unique(spw[block], return_inverse=True)
        fill_grid(grid, block, entries_by_field[field_id], points["antenna"][block], points["time"][block],
                  spw_index.reshape(-1), spws, chan_freq[spws], antenna_xyz=setup["antenna_xyz"],
                  field_dir=setup["field_dirs"][field_id])
        if _parang_for(parang, field_id):
            utimes = np.unique(points["time"][block])
            ra, dec = setup["field_dirs"][field_id]
            angles[field_id] = (utimes, feed_angles_for_ms(setup["antenna_xyz"], setup["mounts"], float(ra),
                                                           float(dec), utimes))
    grid.update(point1=point1, point2=point2)
    return grid, angles


def rows_of(grid: dict, rows) -> dict:
    """The grid of :func:`row_grids` for the subset ``rows`` (a slice or an index array) of its rows."""
    return dict(grid, point1=grid["point1"][rows], point2=grid["point2"][rows])


def correct_rows(vis, flag, weight, time, antenna1, antenna2, field, ddid, grid, angles, *, chans_by_ddid=None,
                 calflags=True):
    """Correct rows of any fields and subbands in place as applycal does, and return the visibilities.

    ``grid`` and ``angles`` come from :func:`row_grids` for these rows. importfitsidi leaves NaN where the
    correlator gave zero weight (DiFX) and flags it. Should such a flag ever be lost, the value must not come
    out as "calibrated data": it is flagged again here, and zeroed so that no later average can pick it up.
    Then the feed rotation and the gains (one pass, :func:`vlbipy.solvers.apply.apply_point_grid`). ``weight``
    (or None) is calibrated where it is given; ``chans_by_ddid`` gives the channels of each DDID the rows were
    read with; ``calflags`` False (or False for a field id) keeps the flags as they were
    (``applymode='calonly'``). ``flag`` and ``weight`` change in place.
    """
    from .apply import apply_point_grid
    from .parang import parang_rotation
    kept = None
    if calflags is not True:
        keep = [f for f in np.unique(field).tolist()
                if not (calflags.get(f, True) if isinstance(calflags, dict) else calflags)]
        if keep:
            rows = np.isin(field, keep)
            kept = (rows, flag[rows].copy())
    rotation = rotate = None
    for field_id, (utimes, angle) in angles.items():
        rows = field == field_id
        if not rows.any():
            continue
        if rotation is None:
            rotation = np.ones((vis.shape[0], vis.shape[2]), dtype=vis.dtype)
            rotate = np.zeros(vis.shape[0], dtype=bool)
        rotation[rows] = parang_rotation(antenna1[rows], antenna2[rows], time[rows], angle, vis.shape[2], vis.dtype,
                                         time_index=np.searchsorted(utimes, time[rows]))
        rotate |= rows
    if chans_by_ddid is None:
        apply_point_grid(vis, flag, weight, grid, grid["point1"], grid["point2"], rotation=rotation, rotate=rotate,
                         invalid=True)
    else:
        for ddid_id in np.unique(ddid).tolist():
            rows = np.flatnonzero(ddid == ddid_id)
            part, part_flag = vis[rows], flag[rows]
            part_weight = None if weight is None else weight[rows]
            apply_point_grid(part, part_flag, part_weight, grid, grid["point1"][rows], grid["point2"][rows],
                             chans=chans_by_ddid[ddid_id], rotation=None if rotation is None else rotation[rows],
                             rotate=None if rotate is None else rotate[rows], invalid=True)
            vis[rows], flag[rows] = part, part_flag
            if part_weight is not None:
                weight[rows] = part_weight
    if kept is not None:
        flag[kept[0]] = kept[1]
    return vis


def corrected_chunks(handle, segments: list[dict], *, entries_by_field: dict, setup: dict, columns: list[str],
                     parang, chunk_rows: int, source: str = "DATA", weights: Optional[str] = "sigma",
                     calflags=True):
    """Read the row ranges ``segments`` of an open table part and correct them chunk by chunk, as applycal.

    Parameters
    ----------
    source : str
        Column corrected: DATA (applycal) or CORRECTED_DATA (an incremental application).
    weights : str or None
        ``"sigma"``: the weights applycal writes (rebuilt from SIGMA_SPECTRUM/SIGMA); ``"weight"``: the WEIGHT
        columns scaled; None: no weights.
    calflags : bool or dict
        Whether the calibration flags of a field are kept (False, or False for its id, when applycal would not
        write them: ``applymode='calonly'``); the visibilities are corrected either way.

    Yields
    ------
    (start, nrow, vis, flag, weight, weighted, flagged_before)
        ``weight`` is the (nrow, nchan, ncorr) cube, or None when no chain of the chunk calibrates the weights;
        ``weighted`` the rows whose weights it calibrated (None: all of them); ``flagged_before`` the flagged
        samples as read.
    """
    from .calblock import sigma_to_weight
    sigma_column = "SIGMA_SPECTRUM" if "SIGMA_SPECTRUM" in columns else "SIGMA"
    weight_column = "WEIGHT_SPECTRUM" if "WEIGHT_SPECTRUM" in columns else "WEIGHT"
    calibrated = np.zeros(max(entries_by_field) + 1, dtype=bool)
    for field, entries in entries_by_field.items():
        calibrated[field] = weights is not None and calibrates_weights(entries)
    for piece in _pieces(segments, BATCH_ROWS):
        start, nrow = piece["start"], piece["nrow"]
        time = np.asarray(handle.getcol("TIME", start, nrow), dtype=np.float64)
        antenna1 = np.asarray(handle.getcol("ANTENNA1", start, nrow), dtype=np.int64)
        antenna2 = np.asarray(handle.getcol("ANTENNA2", start, nrow), dtype=np.int64)
        grid, angles = row_grids(entries_by_field, setup, parang, time, antenna1, antenna2, piece["field"],
                                 piece["ddid"])
        weighted_rows = calibrated[piece["field"]]
        for offset in range(0, nrow, chunk_rows):
            rows = slice(offset, min(offset + chunk_rows, nrow))
            count = rows.stop - rows.start
            vis = handle.getcol(source, start + offset, count)
            flag = np.asarray(handle.getcol("FLAG", start + offset, count), dtype=bool)
            weight, weighted = None, weighted_rows[rows]
            if weighted.any():
                weight = (sigma_to_weight(handle.getcol(sigma_column, start + offset, count)) if weights == "sigma"
                          else np.array(handle.getcol(weight_column, start + offset, count), dtype=np.float32))
                if weight.ndim == 2:
                    weight = np.array(np.broadcast_to(weight[:, None, :], vis.shape), dtype=np.float32)
            flagged_before = int(np.count_nonzero(flag))
            vis = correct_rows(vis, flag, weight, time[rows], antenna1[rows], antenna2[rows], piece["field"][rows],
                               piece["ddid"][rows], rows_of(grid, rows), angles, calflags=calflags)
            yield start + offset, count, vis, flag, weight, None if weighted.all() else weighted, flagged_before


def _row_runs(mask: np.ndarray):
    """``(first, stop)`` of every run of True in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return zip(edges[::2].tolist(), edges[1::2].tolist())


def apply_part_job(*, path: str, segments: list[dict], entries_by_field: dict, setup: dict, engine: str,
                   columns: list[str], parang: bool, write_flags: bool, chunk_rows: int,
                   source: str = "DATA", write_weights: bool = True) -> dict:
    """Worker job: correct the row ranges ``segments`` of one table part in place.

    Parameters
    ----------
    path : str
        The table part (a sub-MS or the MS).
    segments : list of dict
        Row runs ``{"start", "nrow", "field", "ddid"}`` to correct.
    entries_by_field : dict
        ``{field id: entries}`` with entries from :func:`vlbipy.solvers.apply.normalise_entries`.
    setup : dict
        Output of :func:`vlbipy.solvers.msio.read_setup`.
    columns : list of str
        Main-table column names (decides SIGMA_SPECTRUM / WEIGHT_SPECTRUM handling).
    parang : bool
        Apply the parallactic-angle (feed) rotation.
    write_flags : bool
        Write the calibration flags to FLAG (False for ``applymode='calonly'``).
    chunk_rows : int
        Rows per read/correct/write block.
    source : str
        DATA, or CORRECTED_DATA to apply on top of the corrected column (weights then scaled from WEIGHT).
    write_weights : bool
        Write the calibrated weights (False leaves the WEIGHT columns as they are).

    Returns
    -------
    dict
        ``rows`` corrected, ``samples`` and ``flagged_before``/``flagged_after`` sample counts.
    """
    stats = {"rows": 0, "samples": 0, "flagged_before": 0, "flagged_after": 0}
    has_weight_spectrum = "WEIGHT_SPECTRUM" in columns
    handle = open_table(path, readonly=False, engine=engine)
    try:
        for start, nrow, vis, flag, weight, weighted, flagged_before in corrected_chunks(
                handle, segments, entries_by_field=entries_by_field, setup=setup, columns=columns, parang=parang,
                chunk_rows=chunk_rows, source=source,
                weights=None if not write_weights else "sigma" if source == "DATA" else "weight"):
            flagged_after = int(np.count_nonzero(flag))
            handle.putcol("CORRECTED_DATA", vis, start)
            # Calibration only ever adds flags, so an unchanged count means an unchanged column: skip the
            # write (every pass after the first one, typically).
            if write_flags and flagged_after != flagged_before:
                handle.putcol("FLAG", flag, start)
            if weight is not None:
                # Rows of a chain that does not calibrate the weights keep theirs (only where fields mix in a chunk).
                for first, stop in ([(0, nrow)] if weighted is None else _row_runs(weighted)):
                    if has_weight_spectrum:
                        handle.putcol("WEIGHT_SPECTRUM", weight[first:stop], start + first)
                    handle.putcol("WEIGHT", weight[first:stop].mean(axis=1), start + first)
            stats["rows"] += nrow
            stats["samples"] += int(flag.size)
            stats["flagged_before"] += flagged_before
            stats["flagged_after"] += flagged_after
    finally:
        handle.close()
    return stats


def flag_part_job(*, path: str, segments: list[dict], entries_by_field: dict, setup: dict, engine: str,
                  chunk_rows: int) -> dict:
    """Worker job: add to FLAG the calibration flags of ``segments`` (data without a valid solution), as
    :func:`apply_part_job` does, without reading or writing a visibility (a deferred application)."""
    stats = {"rows": 0, "samples": 0, "flagged_before": 0, "flagged_after": 0}
    handle = open_table(path, readonly=False, engine=engine)
    try:
        for piece in _pieces(segments, BATCH_ROWS):
            start, nrow = piece["start"], piece["nrow"]
            time = np.asarray(handle.getcol("TIME", start, nrow), dtype=np.float64)
            antenna1 = np.asarray(handle.getcol("ANTENNA1", start, nrow), dtype=np.int64)
            antenna2 = np.asarray(handle.getcol("ANTENNA2", start, nrow), dtype=np.int64)
            grid, _ = row_grids(entries_by_field, setup, False, time, antenna1, antenna2, piece["field"],
                                piece["ddid"])
            for offset in range(0, nrow, chunk_rows):
                rows = slice(offset, min(offset + chunk_rows, nrow))
                count = rows.stop - rows.start
                flag = np.asarray(handle.getcol("FLAG", start + offset, count), dtype=bool)
                flagged_before = int(np.count_nonzero(flag))
                apply_point_flags(flag, grid, grid["point1"][rows], grid["point2"][rows])
                flagged_after = int(np.count_nonzero(flag))
                if flagged_after != flagged_before:
                    handle.putcol("FLAG", flag, start + offset)
                stats["rows"] += count
                stats["samples"] += int(flag.size)
                stats["flagged_before"] += flagged_before
                stats["flagged_after"] += flagged_after
    finally:
        handle.close()
    return stats


def ensure_column(ms, name: str, template: str, *, engine: Optional[str] = None) -> bool:
    """Add column ``name`` (laid out like ``template``) to every part of ``ms`` that lacks it; True if any was added."""
    from .msio import ms_parts
    added = False
    for part in ms_parts(ms):
        handle = open_table(part, readonly=False, engine=engine)
        try:
            if name not in handle.colnames():
                handle.add_column_like(name, template)
                added = True
        finally:
            handle.close()
    return added


def _resolve(ms, entries_by_field: dict, setup: dict) -> dict:
    """``{field id: normalised entries}`` (entries given as dicts are read; tuples are kept)."""
    from .apply import normalise_entries
    table_cache: dict = {}
    return {int(field): (list(entries) if entries and isinstance(entries[0], tuple)
                         else normalise_entries(entries, calwt=True, cache=table_cache,
                                                field_dirs=setup["field_dirs"], target_field=int(field)))
            for field, entries in entries_by_field.items()}


def _part_jobs(layout: dict, field_ids, *, ddids=None, scans=None, **job) -> list[dict]:
    """One job per table part holding rows of ``field_ids`` (and ``ddids``/``scans``), its row runs as ``segments``."""
    selected = select_runs(layout, fields=set(field_ids), ddids=None if ddids is None else set(ddids),
                           scans=None if scans is None else set(scans))
    jobs = []
    for part_index, path in enumerate(layout["parts"]):
        runs = [i for i in selected if int(layout["part"][i]) == part_index]
        if not runs:
            continue
        segments = [{"start": int(layout["start"][i]), "nrow": int(layout["nrow"][i]),
                     "field": int(layout["field"][i]), "ddid": int(layout["ddid"][i])} for i in runs]
        fields = {s["field"] for s in segments}
        jobs.append(dict(job, path=path, segments=segments,
                         entries_by_field={f: e for f, e in job["entries_by_field"].items() if f in fields}))
    return jobs


def apply_to_ms(ms, entries_by_field: dict, *, parang: bool = True, applymode: str = "calflagstrict",
                workers: Optional[int] = None, chunk_rows: int = APPLY_ROWS, source: str = "DATA",
                record: bool = True, ddids=None, scans=None, write_weights: bool = True) -> dict:
    """Correct the given fields of ``ms`` with their calibration chains, writing CORRECTED_DATA, FLAG and weights.

    Parameters
    ----------
    ms : str or pathlib.Path
        Measurement set or Multi-MS.
    entries_by_field : dict
        ``{field id: entries}``: the table chain of each field to correct, entries as accepted by
        :func:`vlbipy.solvers.apply.apply_tables` (dicts, or tuples from ``normalise_entries``). Fields not
        listed are left untouched.
    parang : bool
        Apply the parallactic-angle correction.
    applymode : str
        ``'calflag'``/``'calflagstrict'`` (flag data without a valid solution) or ``'calonly'`` (leave FLAG alone).
    workers : int, optional
        Worker processes (default: :func:`vlbipy.solvers.workers.default_workers`).
    chunk_rows : int
        Rows per read/correct/write block.
    source : str
        ``"DATA"`` (applycal), or ``"CORRECTED_DATA"`` to apply the tables on top of what that column holds.
    record : bool
        Keep the calibration record of ``ms`` (:mod:`vlbipy.solvers.corrected`) up to date.
    ddids, scans : iterable of int, optional
        Correct only these data descriptions / scans of the fields (the rest is left as it is).
    write_weights : bool
        Write the calibrated weights (False keeps the WEIGHT columns, see
        :func:`vlbipy.solvers.corrected.weights_set`).

    Returns
    -------
    dict
        ``rows``, ``samples``, ``flagged_before``, ``flagged_after``, ``parts`` and ``seconds``.
    """
    from . import corrected
    mode = str(applymode or "calflagstrict").lower()
    if mode not in APPLY_MODES:
        raise NotImplementedError(f"applymode={applymode!r} is not supported by the dask-ms apply")
    if int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be positive")
    if source not in ("DATA", "CORRECTED_DATA"):
        raise ValueError(f"cannot apply calibration to the {source} column")
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    resolved = _resolve(ms, entries_by_field, setup)
    if source == "DATA":
        ensure_column(ms, "CORRECTED_DATA", "DATA", engine=engine)
    layout = read_layout(ms, engine=engine)
    if source not in layout["columns"]:
        raise RuntimeError(f"{Path(ms).name} has no {source} column")
    partial = ddids is not None or scans is not None
    if record:
        corrected.begin_write(ms, resolved)
    jobs = _part_jobs(layout, resolved, ddids=ddids, scans=scans, entries_by_field=resolved, setup=setup,
                      engine=engine, columns=layout["columns"], parang=bool(parang), write_flags=mode != "calonly",
                      chunk_rows=int(chunk_rows), source=source, write_weights=bool(write_weights))
    results = run_jobs("vlbipy.solvers.apply_task:apply_part_job", jobs, workers)
    if record:
        if source == "DATA" and not partial:
            corrected.record_written(ms, entries_by_field, parang=bool(parang), write_flags=mode != "calonly")
        else:
            corrected.record_external(ms, resolved, "incremental" if source != "DATA" else "partial")
    total = {name: sum(result[name] for result in results)
             for name in ("rows", "samples", "flagged_before", "flagged_after")}
    total.update(parts=len(jobs), seconds=_time.perf_counter() - started)
    logger.info("applycal[dask-ms]: {} — {} row(s) of {} field(s) in {} part(s){}, flagged {:.1%} -> {:.1%}, {:.2f}s",
                Path(ms).name, total["rows"], len(resolved), total["parts"],
                " on top of CORRECTED_DATA" if source != "DATA" else "",
                total["flagged_before"] / max(total["samples"], 1),
                total["flagged_after"] / max(total["samples"], 1), total["seconds"])
    return total


def defer_to_ms(ms, entries_by_field: dict, *, parang: bool = True, applymode: str = "calflagstrict",
                workers: Optional[int] = None, chunk_rows: int = FLAG_ROWS) -> dict:
    """Record the application of ``entries_by_field`` to ``ms`` without writing CORRECTED_DATA.

    The calibration flags are written (unless ``applymode='calonly'``), so every later step sees the flags
    an applycal would have left; the corrected visibilities and weights are computed by whoever reads them
    (see :mod:`vlbipy.solvers.corrected`). Same arguments and statistics as :func:`apply_to_ms`, plus
    ``pending``, the fields whose CORRECTED_DATA is now out of date.
    """
    from . import corrected
    mode = str(applymode or "calflagstrict").lower()
    if mode not in APPLY_MODES:
        raise NotImplementedError(f"applymode={applymode!r} is not supported by the dask-ms apply")
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    resolved = _resolve(ms, entries_by_field, setup)
    layout = read_layout(ms, engine=engine)
    results, jobs = [], []
    if mode != "calonly":
        jobs = _part_jobs(layout, resolved, entries_by_field=resolved, setup=setup, engine=engine,
                          chunk_rows=int(chunk_rows))
        results = run_jobs("vlbipy.solvers.apply_task:flag_part_job", jobs, workers)
    stale = corrected.record_deferred(ms, entries_by_field, parang=bool(parang), write_flags=mode != "calonly")
    total = {name: sum(result[name] for result in results)
             for name in ("rows", "samples", "flagged_before", "flagged_after")}
    total.update(parts=len(jobs), pending=stale, seconds=_time.perf_counter() - started)
    logger.info("applycal[dask-ms]: {} — deferred for {} field(s) (CORRECTED_DATA is computed when it is read); "
                "calibration flags {:.1%} -> {:.1%} in {} part(s), {:.2f}s", Path(ms).name, len(resolved),
                total["flagged_before"] / max(total["samples"], 1),
                total["flagged_after"] / max(total["samples"], 1), total["parts"], total["seconds"])
    return total


def _field_runs(specs: list[dict]) -> list[list[dict]]:
    """``specs`` cut into consecutive groups of one field each."""
    groups: list[list[dict]] = []
    for spec in specs:
        if groups and int(groups[-1][-1]["field"]) == int(spec["field"]):
            groups[-1].append(spec)
        else:
            groups.append([spec])
    return groups


def _select_corrs(block: dict, corrs) -> None:
    """Keep the correlations ``corrs`` of every column of ``block`` that has them."""
    corrs = np.asarray(corrs, dtype=np.int64)
    for name, values in block.items():
        if name in CUBE_COLUMNS:
            block[name] = values[:, :, corrs]
        elif name in ("WEIGHT", "SIGMA"):
            block[name] = values[:, corrs]


def _read_corrected_field(specs: list[dict], columns: list[str], view: dict, *, engine: str, chans_by_ddid,
                          corrs, nchan: int, ncorr: int, timerange) -> dict:
    """The runs ``specs`` (one field of ``view``) with CORRECTED_DATA computed from DATA (:func:`read_view_runs`)."""
    from .calblock import sigma_to_weight
    field = int(specs[0]["field"])
    entries = view["entries_by_field"][field]
    want_weights = [c for c in ("WEIGHT", "WEIGHT_SPECTRUM") if c in columns]
    calibrate_weights = (bool(want_weights) and view.get("weights", {}).get(field, True)
                         and calibrates_weights(entries))
    if calibrate_weights and chans_by_ddid is not None:
        raise NotImplementedError("calibrated weights of a channel selection")
    sigma_column = "SIGMA_SPECTRUM" if "SIGMA_SPECTRUM" in view["columns"] else "SIGMA"
    names = ["DATA" if c == "CORRECTED_DATA" else c for c in columns if not (calibrate_weights and c in want_weights)]
    names += [c for c in ("FLAG", "TIME", "ANTENNA1", "ANTENNA2") if c not in names]
    names += [sigma_column] if calibrate_weights and sigma_column not in names else []
    block = read_runs(specs, names, engine=engine, chans_by_ddid=chans_by_ddid, nchan=nchan, ncorr=ncorr,
                      timerange=timerange)
    if not block:
        return {}
    vis = block.pop("DATA")
    flag = np.array(block["FLAG"], dtype=bool)
    as_read = None if view["write_flags"].get(field, True) else flag.copy()
    weight = None
    if calibrate_weights:
        weight = sigma_to_weight(block[sigma_column] if sigma_column in columns else block.pop(sigma_column))
        if weight.ndim == 2:
            weight = np.array(np.broadcast_to(weight[:, None, :], vis.shape), dtype=np.float32)
    time = np.asarray(block["TIME"], dtype=np.float64)
    antenna1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    antenna2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    ddid = np.asarray(block["DATA_DESC_ID"], dtype=np.int64)
    fields = np.full(ddid.size, field, dtype=np.int64)
    grid, angles = row_grids({field: entries}, view["setup"], view["parang"], time, antenna1, antenna2, fields, ddid)
    vis = correct_rows(vis, flag, weight, time, antenna1, antenna2, fields, ddid, grid, angles,
                       chans_by_ddid=chans_by_ddid)
    block["CORRECTED_DATA"] = vis
    block["FLAG"] = flag if as_read is None else as_read
    if calibrate_weights:
        if "WEIGHT_SPECTRUM" in columns:
            block["WEIGHT_SPECTRUM"] = weight
        if "WEIGHT" in columns:
            block["WEIGHT"] = weight.mean(axis=1)
    for name in ("TIME", "ANTENNA1", "ANTENNA2"):
        if name not in columns:
            block.pop(name)
    if corrs is not None:
        _select_corrs(block, corrs)
    return block


def read_view_runs(specs: list[dict], columns: list[str], *, engine: str, view: Optional[dict] = None,
                   chans_by_ddid: Optional[dict] = None, corrs=None, nchan: int = 0, ncorr: int = 0,
                   timerange: Optional[tuple] = None) -> dict:
    """:func:`vlbipy.solvers.msio.read_runs`, with CORRECTED_DATA of the fields of ``view`` computed from DATA.

    ``view`` comes from :func:`vlbipy.solvers.corrected.view` (None: read every run as it is). The runs of
    its fields read DATA of every correlation (the feed rotation mixes them), FLAG and, when WEIGHT or
    WEIGHT_SPECTRUM is asked for and the chain calibrates weights, SIGMA[_SPECTRUM]; they come back corrected
    under the names asked for, with the flags and weights applycal would have written.
    """
    if view is None or "CORRECTED_DATA" not in columns:
        return read_runs(specs, columns, engine=engine, chans_by_ddid=chans_by_ddid, corrs=corrs, nchan=nchan,
                         ncorr=ncorr, timerange=timerange)
    live = set(view["entries_by_field"])
    pieces = []
    for group in _field_runs(specs):
        if int(group[0]["field"]) in live:
            block = _read_corrected_field(group, columns, view, engine=engine, chans_by_ddid=chans_by_ddid,
                                          corrs=corrs, nchan=nchan or int(view["setup"]["chan_freq"].shape[1]),
                                          ncorr=ncorr or int(view["setup"]["ncorr"]), timerange=timerange)
        else:
            block = read_runs(group, columns, engine=engine, chans_by_ddid=chans_by_ddid, corrs=corrs, nchan=nchan,
                              ncorr=ncorr, timerange=timerange)
        if block:
            pieces.append(block)
    if not pieces:
        return {}
    if len(pieces) == 1:
        return pieces[0]
    return {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}


def _table_fields(table, gainfield: str, ms_field_names: list[str]):
    """``gainfield`` of the command line as table field ids (names are looked up in the table's FIELD subtable)."""
    text = str(gainfield or "").strip()
    if not text:
        return []
    if text.lower() == "nearest":
        own = [str(name) for name in table.field_names]
        if own != list(ms_field_names)[:len(own)]:
            raise ValueError(f"{table.path}: gainfield='nearest' needs a table solved on this measurement set "
                             "(its fields differ); name the field instead")
        return "nearest"
    ids = []
    names = [str(name) for name in table.field_names]
    for item in (part.strip() for part in text.split(",")):
        if item.isdigit():
            ids.append(int(item))
        elif item in names:
            ids.append(names.index(item))
        else:
            raise ValueError(f"{table.path} has no field {item!r} (it has {', '.join(names)})")
    return ids


def _interpolated_entries(ms: Path, entries: list[dict], field_ids: list[int], setup: dict, layout: dict,
                          max_extrapolation: float, workdir: Path, max_gap: Optional[float] = None) -> dict:
    """``{field: entries}`` where a table whose gainfield lists two or more fields reaches the other fields
    interpolated between the two of them nearest on the sky (:func:`vlbipy.phaseref.interpolated_table`)."""
    from types import SimpleNamespace
    from .. import phaseref
    from .apply import adapt_table, load_caltable
    names = setup["field_names"]
    coords = {name: tuple(np.degrees(setup["field_dirs"][index])) for index, name in enumerate(names)}
    spans: dict = {}
    for f, s, lo, hi in zip(layout["field"], layout["scan"], layout["t_min"], layout["t_max"]):
        old = spans.get((int(f), int(s)), (lo, hi))
        spans[(int(f), int(s))] = (min(old[0], lo), max(old[1], hi))
    scans = [SimpleNamespace(source=names[f], time_start=lo, time_end=hi) for (f, _), (lo, hi) in spans.items()]
    out = {}
    for f in field_ids:
        chosen = []
        for index, entry in enumerate(entries):
            ids = entry["gainfield"]
            table = entry["table"]
            if not isinstance(ids, list) or len(ids) < 2 or names[f] in {str(table.field_names[i]) for i in ids}:
                chosen.append(entry)
                continue
            references = [str(table.field_names[i]) for i in ids]
            missing = [name for name in references if name not in coords]
            if missing:
                raise ValueError(f"--phaseref interpolate: {', '.join(missing)} not a field of {ms.name}")
            a, b, x, offset = phaseref.interpolation_pair(references, coords, names[f], max_extrapolation)
            logger.info("applycal: {} at x = {:.3f} between {} and {} ({:.2f} deg off their line)", names[f], x, a,
                        b, offset)
            derived = workdir / f"{index}.{Path(table.path).name}.{names[f]}"
            phaseref.interpolated_table((table.path, a), (table.path, b), x, names[f],
                                        phaseref.receiver_times(scans, names[f]), derived, max_gap=max_gap)
            new = adapt_table(load_caltable(derived), setup["antenna_names"])
            chosen.append(dict(entry, table=new, gainfield=[[str(n) for n in new.field_names].index(names[f])]))
        out[f] = chosen
    return out


def applycal(ms, caltables: list[str], *, field: str = "", spw: str = "", scan: str = "", gainfield=None,
             interp=None, spwmap=None, calwt=None, parang: Optional[bool] = None, applymode: str = "calflagstrict",
             on_corrected: Optional[bool] = None, workers: Optional[int] = None, phaseref: str = "time",
             max_extrapolation: float = 1.0, max_gap: Optional[float] = 600.0) -> dict:
    """Apply calibration tables to any measurement set (``vlbipy applycal``).

    On a set that has CORRECTED_DATA the tables are applied on top of it: their corrections multiply what
    the column holds, their flags are added and the weights scaled (incremental). Without one, they are
    applied to DATA into a new CORRECTED_DATA, as ``casatasks.applycal`` does; ``on_corrected=False``
    forces that on a set that has one. A table solved on another set (a split, another epoch) is matched to
    this one by antenna name, and the names of ``gainfield`` are those of the table's own fields.

    Parameters
    ----------
    field, spw, scan : str
        Comma-separated field names (or ids), subband ids and scan numbers to correct (empty = all).
    gainfield, interp, spwmap, calwt : list, optional
        Per table, as in applycal: the table fields to use (names, ids or ``nearest``), the interpolation,
        the subband map (``"0,0,0,0"``) and whether the weights are calibrated (default True).
    parang : bool, optional
        The feed rotation. Off by default, as in ``casatasks.applycal``: the tables one applies this way (a
        self-calibration, a table from another set) go on data that are calibrated already, the feed rotation
        included (CORRECTED_DATA, or the DATA of a split).
    phaseref : {"time", "interpolate"}
        For a table whose ``gainfield`` lists two or more fields: ``"time"`` takes the solutions nearest in
        time, whichever field they belong to (as CASA does); ``"interpolate"`` gives every other field the
        solutions of the two of them nearest on the sky combined along the line joining them
        (:mod:`vlbipy.phaseref`), at most ``max_extrapolation`` times their separation beyond either. A time
        either of them was not observed within ``max_gap`` seconds of (before and after) takes the time
        interpolation instead (None: never).

    Returns
    -------
    dict
        The statistics of :func:`apply_to_ms` plus ``source``, ``parang`` and ``fields``.
    """
    from . import corrected
    from .apply import adapt_table, load_caltable
    ms = Path(ms)
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    has_corrected = "CORRECTED_DATA" in layout["columns"]
    on_corrected = has_corrected if on_corrected is None else bool(on_corrected)
    if on_corrected and not has_corrected:
        raise ValueError(f"{ms.name} has no CORRECTED_DATA to apply the tables on top of")
    if on_corrected:
        corrected.write_pending(ms, workers=workers)      # a deferred calibration is part of the corrected data
    parang = bool(parang)
    count = len(caltables)
    if not count:
        raise ValueError("no calibration table to apply")

    def per_table(values, default):
        values = list(values or [])
        if len(values) == 1:
            values = values * count
        if values and len(values) != count:
            raise ValueError(f"{len(values)} values for {count} calibration tables")
        return values or [default] * count

    gainfields, interps = per_table(gainfield, ""), per_table(interp, "linear")
    spwmaps, calwts = per_table(spwmap, ""), per_table(calwt, True)
    entries = []
    for index, path in enumerate(caltables):
        table = adapt_table(load_caltable(path), setup["antenna_names"])
        spw_map = spwmaps[index]
        if isinstance(spw_map, str):
            spw_map = [int(value) for value in spw_map.split(",") if value.strip()]
        entries.append({"table": table, "interp": interps[index] or "linear", "spwmap": list(spw_map),
                        "gainfield": _table_fields(table, gainfields[index], setup["field_names"]),
                        "calwt": str(calwts[index]).strip().lower() not in ("false", "f", "0", "no")})
    present = sorted({int(f) for f in layout["field"]})
    if field:
        field_ids = []
        for item in (part.strip() for part in field.split(",") if part.strip()):
            if item.isdigit():
                field_ids.append(int(item))
            elif item in setup["field_names"]:
                field_ids.append(setup["field_names"].index(item))
            else:
                raise ValueError(f"{ms.name} has no field {item!r}")
    else:
        field_ids = present
    ddids = None
    if spw:
        spw_ids = {int(value) for value in spw.split(",") if value.strip()}
        ddids = [ddid for ddid, spw_id in enumerate(setup["ddid_to_spw"]) if int(spw_id) in spw_ids]
    scans = [int(value) for value in scan.split(",") if value.strip()] if scan else None
    source = "CORRECTED_DATA" if on_corrected else "DATA"
    logger.info("applycal: {} {} {} ({}), parang {}", ", ".join(Path(t).name for t in caltables),
                "on top of" if on_corrected else "applied to", source if on_corrected else "DATA -> CORRECTED_DATA",
                ms.name, "on" if parang else "off (--parang to de-rotate the feeds of uncalibrated data)")
    if phaseref not in ("time", "interpolate"):
        raise ValueError(f"phaseref must be 'time' or 'interpolate', got {phaseref!r}")
    with tempfile.TemporaryDirectory(prefix=".phaseref.", dir=ms.parent) as workdir:
        if phaseref == "interpolate":
            entries_by_field = _interpolated_entries(ms, entries, field_ids, setup, layout, max_extrapolation,
                                                     Path(workdir), max_gap=max_gap)
        else:
            entries_by_field = {f: entries for f in field_ids}
        total = apply_to_ms(ms, entries_by_field, parang=parang, applymode=applymode,
                            workers=workers, source=source, ddids=ddids, scans=scans)
    total.update(source=source, parang=parang, fields=[setup["field_names"][f] for f in field_ids])
    return total
