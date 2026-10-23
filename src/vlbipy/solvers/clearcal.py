"""Reset the corrected (and model) data of a measurement set: what ``casatasks.clearcal`` writes.

:func:`clearcal` repeats ``clearcal(vis, addmodel=...)`` without a selection: CORRECTED_DATA is created
(laid out like DATA) when it is missing and then set to DATA on every row; MODEL_DATA, when the measurement
set has one, is set to 1 + 0j; ``addmodel=True`` creates it first. Weights and flags are left alone, as
CASA's ``initcalset`` does. The parts of a Multi-MS are copied by parallel worker processes, each on its own
table. The record of a deferred calibration (:mod:`vlbipy.solvers.corrected`) is dropped: the column now holds
the uncalibrated data. Unlike CASA no HISTORY row is written.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .msio import ms_parts, open_table, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows per read/write block (64 channels x 4 correlations: 40 MB of visibilities).
CLEARCAL_ROWS = 50_000


def clearcal_part_job(*, path: str, engine: str, chunk_rows: int) -> dict:
    """Worker job: CORRECTED_DATA = DATA and MODEL_DATA = 1 (when present) on every row of one table part."""
    handle = open_table(path, readonly=False, engine=engine)
    try:
        nrow = handle.nrows()
        has_model = "MODEL_DATA" in handle.colnames()
        for start in range(0, nrow, chunk_rows):
            count = min(chunk_rows, nrow - start)
            data = handle.getcol("DATA", start, count)
            handle.putcol("CORRECTED_DATA", data, start)
            if has_model:
                handle.putcol("MODEL_DATA", np.ones_like(data), start)
    finally:
        handle.close()
    return {"rows": nrow, "model": has_model}


def clearcal(ms, *, addmodel: bool = False, workers: Optional[int] = None,
                chunk_rows: int = CLEARCAL_ROWS) -> dict:
    """Set CORRECTED_DATA to DATA (and MODEL_DATA to 1) on every row of ``ms``.

    Parameters
    ----------
    ms : path
        The measurement set or Multi-MS (modified in place).
    addmodel : bool
        Create MODEL_DATA when it is missing (CASA's ``addmodel``).
    workers : int
        Worker processes (default: the pool's size).
    chunk_rows : int
        Rows per read/write block.

    Returns
    -------
    dict
        ``rows``, ``parts``, ``model`` (whether MODEL_DATA was reset) and ``seconds``.
    """
    from . import corrected
    from .applycal import ensure_column
    started = _time.perf_counter()
    engine = table_engine()
    ensure_column(ms, "CORRECTED_DATA", "DATA", engine=engine)
    if addmodel:
        ensure_column(ms, "MODEL_DATA", "DATA", engine=engine)
    jobs = [{"path": part, "engine": engine, "chunk_rows": int(chunk_rows)} for part in ms_parts(ms)]
    results = run_jobs("vlbipy.solvers.clearcal:clearcal_part_job", jobs, workers)
    corrected.forget(ms)
    report = {"rows": int(sum(r["rows"] for r in results)), "parts": len(jobs),
              "model": bool(results) and all(r["model"] for r in results), "seconds": _time.perf_counter() - started}
    logger.info("clearcal[native]: {} — CORRECTED_DATA = DATA{} on {} row(s) in {} part(s), {:.1f}s", Path(ms).name,
                ", MODEL_DATA = 1" if report["model"] else "", report["rows"], report["parts"], report["seconds"])
    return report
