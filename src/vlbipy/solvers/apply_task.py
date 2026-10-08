"""Apply calibration tables to a measurement set (the dask-ms backend's ``applycal``).

:func:`apply_to_ms` writes what ``casatasks.applycal`` writes — CORRECTED_DATA,
the calibration flags and the calibrated weights — into the measurement set
itself, so CASA flagging, imaging and export downstream are unaffected.

The work is I/O bound (every visibility is read and written once), so it is
organised around the tables rather than the maths:

* one worker process per table **part** (the sub-MSs of a Multi-MS): different
  processes then read and write different tables with no lock between them. A
  plain MS has a single part and is processed by a single worker;
* all requested fields are corrected in **one pass** over each part, each row
  block with the table chain of its own field;
* the whole chain is multiplied into one gain per antenna, time and channel
  before it touches the visibilities (:func:`vlbipy.solvers.apply.combined_gains`).

Conventions follow CASA, checked table by table against ``applycal``: the
weights are rebuilt from SIGMA (``1 / sigma^2``, SIGMA_SPECTRUM when present)
and scaled by ``|g_i g_j|^2`` of the Tsys, gain-curve and gain tables with
``calwt`` (never by bandpasses), so applying twice does not calibrate them
twice; data whose solution is flagged or missing are flagged and get weight 0
(``applymode='calflag'``/``'calflagstrict'``).
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .apply import apply_tables, calibrates_weights
from .msio import open_table, read_layout, read_setup, select_runs, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows corrected per read/write block inside a worker (64 channels x 4 correlations: 40 MB of visibilities).
APPLY_ROWS = 20_000
#: ``applymode`` values this implementation covers.
APPLY_MODES = ("", "calflag", "calflagstrict", "calonly")


def apply_part_job(*, path: str, segments: list[dict], entries_by_field: dict, setup: dict, engine: str,
                   columns: list[str], parang: bool, write_flags: bool, chunk_rows: int) -> dict:
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

    Returns
    -------
    dict
        ``rows`` corrected, ``samples`` and ``flagged_before``/``flagged_after`` sample counts.
    """
    from .calblock import sigma_to_weight
    stats = {"rows": 0, "samples": 0, "flagged_before": 0, "flagged_after": 0}
    sigma_column = "SIGMA_SPECTRUM" if "SIGMA_SPECTRUM" in columns else "SIGMA"
    has_weight_spectrum = "WEIGHT_SPECTRUM" in columns
    chan_freq, ddid_to_spw = setup["chan_freq"], setup["ddid_to_spw"]
    handle = open_table(path, readonly=False, engine=engine)
    try:
        for segment in segments:
            field = int(segment["field"])
            entries = entries_by_field[field]
            calibrate_weights = calibrates_weights(entries)
            spw_id = int(ddid_to_spw[segment["ddid"]])
            for offset in range(0, segment["nrow"], chunk_rows):
                start, nrow = segment["start"] + offset, min(chunk_rows, segment["nrow"] - offset)
                vis = handle.getcol("DATA", start, nrow)
                flag = np.asarray(handle.getcol("FLAG", start, nrow), dtype=bool)
                time = np.asarray(handle.getcol("TIME", start, nrow), dtype=np.float64)
                antenna1 = np.asarray(handle.getcol("ANTENNA1", start, nrow), dtype=np.int64)
                antenna2 = np.asarray(handle.getcol("ANTENNA2", start, nrow), dtype=np.int64)
                weight = sigma_to_weight(handle.getcol(sigma_column, start, nrow)) if calibrate_weights else None
                flagged_before = int(np.count_nonzero(flag))
                # importfitsidi leaves NaN where the correlator gave zero weight (DiFX) and flags it.
                # Should such a flag ever be lost, the value must not come out as "calibrated data":
                # it is flagged again here, and zeroed so that no later average can pick it up.
                invalid = ~np.isfinite(vis)
                if invalid.any():
                    vis = np.where(invalid, 0, vis).astype(vis.dtype, copy=False)
                    flag |= invalid
                if parang:
                    from .parang import apply_parang, feed_angles_for_ms
                    times, time_index = np.unique(time, return_inverse=True)
                    ra, dec = setup["field_dirs"][field]
                    angles = feed_angles_for_ms(setup["antenna_xyz"], setup["mounts"], float(ra), float(dec), times)
                    vis = apply_parang(vis, antenna1, antenna2, time, angles, time_index=time_index)
                vis, flag, weight = apply_tables(vis, flag, weight, antenna1, antenna2, time,
                                                 np.full(nrow, spw_id, dtype=np.int64), chan_freq, entries,
                                                 antenna_xyz=setup["antenna_xyz"], field_dir=setup["field_dirs"][field],
                                                 copy=False)
                flagged_after = int(np.count_nonzero(flag))
                handle.putcol("CORRECTED_DATA", vis, start)
                # Calibration only ever adds flags, so an unchanged count means an unchanged column: skip the write
                # (every pass after the first one, typically).
                if write_flags and flagged_after != flagged_before:
                    handle.putcol("FLAG", flag, start)
                if weight is not None:
                    if has_weight_spectrum:
                        handle.putcol("WEIGHT_SPECTRUM", weight, start)
                    handle.putcol("WEIGHT", weight.mean(axis=1), start)
                stats["rows"] += nrow
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


def apply_to_ms(ms, entries_by_field: dict, *, parang: bool = True, applymode: str = "calflagstrict",
                workers: Optional[int] = None, chunk_rows: int = APPLY_ROWS) -> dict:
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

    Returns
    -------
    dict
        ``rows``, ``samples``, ``flagged_before``, ``flagged_after``, ``parts`` and ``seconds``.
    """
    from .apply import normalise_entries
    mode = str(applymode or "calflagstrict").lower()
    if mode not in APPLY_MODES:
        raise NotImplementedError(f"applymode={applymode!r} is not supported by the dask-ms apply")
    if int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be positive")
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    table_cache: dict = {}
    resolved = {int(field): (list(entries) if entries and isinstance(entries[0], tuple)
                             else normalise_entries(entries, calwt=True, cache=table_cache,
                                                    field_dirs=setup["field_dirs"], target_field=int(field)))
                for field, entries in entries_by_field.items()}
    ensure_column(ms, "CORRECTED_DATA", "DATA", engine=engine)
    layout = read_layout(ms, engine=engine)
    selected = select_runs(layout, fields=set(resolved))
    jobs = []
    for part_index, path in enumerate(layout["parts"]):
        runs = [i for i in selected if int(layout["part"][i]) == part_index]
        if not runs:
            continue
        segments = [{"start": int(layout["start"][i]), "nrow": int(layout["nrow"][i]),
                     "field": int(layout["field"][i]), "ddid": int(layout["ddid"][i])} for i in runs]
        jobs.append({"path": path, "segments": segments, "setup": setup, "engine": engine,
                     "entries_by_field": {field: resolved[field] for field in {s["field"] for s in segments}},
                     "columns": layout["columns"], "parang": bool(parang), "write_flags": mode != "calonly",
                     "chunk_rows": int(chunk_rows)})
    results = run_jobs("vlbipy.solvers.apply_task:apply_part_job", jobs, workers)
    total = {name: sum(result[name] for result in results)
             for name in ("rows", "samples", "flagged_before", "flagged_after")}
    total.update(parts=len(jobs), seconds=_time.perf_counter() - started)
    logger.info("applycal[dask-ms]: {} — {} row(s) of {} field(s) in {} part(s), flagged {:.1%} -> {:.1%}, {:.2f}s",
                Path(ms).name, total["rows"], len(resolved), total["parts"],
                total["flagged_before"] / max(total["samples"], 1),
                total["flagged_after"] / max(total["samples"], 1), total["seconds"])
    return total
