"""The system-temperature table of the a-priori amplitude calibration (the dask-ms backend's ``gencal tsys``).

:func:`write_tsys_table` writes what ``casatasks.gencal(caltype='tsys')`` writes for a SYSCAL subtable with
one Tsys per receptor (no ``TSYS_SPECTRUM``): a ``B TSYS`` table with one row per SYSCAL row, FPARAM the
system temperature, flagged where it is not positive, PARAMERR and SNR 0.1. Each row belongs to the scan
whose integrations span its time, or else to the next scan (a measurement in the gap before a scan, as
stations log it while they slew), and takes that scan's field. One channel per subband, at the mean frequency
of the subband and its total width.

gencal took 22 minutes on GL048 (576 000 SYSCAL rows) walking the main table scan by scan; the scan
boundaries come from the measurement-set layout here, read once.
"""
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger
from .caltable import write_jones_table
from .msio import open_table, read_layout, read_setup, table_engine

logger = get_logger()


def scan_of_times(layout: dict, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scan number and field of each time: the scan containing it, or else the next one (-1 after the last)."""
    numbers = np.unique(layout["scan"])
    starts = np.array([layout["t_min"][layout["scan"] == n].min() for n in numbers])
    ends = np.array([layout["t_max"][layout["scan"] == n].max() for n in numbers])
    fields = np.array([layout["field"][layout["scan"] == n][0] for n in numbers])
    order = np.argsort(starts)
    numbers, starts, ends, fields = numbers[order], starts[order], ends[order], fields[order]
    index = np.searchsorted(starts, times, side="right") - 1
    inside = (index >= 0) & (times <= ends[np.clip(index, 0, None)])
    index = np.where(inside, index, index + 1)
    known = index < numbers.size
    safe = np.clip(index, 0, numbers.size - 1)
    return np.where(known, numbers[safe], -1), np.where(known, fields[safe], -1)


def write_tsys_table(ms, caltable) -> dict:
    """Write the ``B TSYS`` table of ``ms`` from its SYSCAL subtable (see the module docstring).

    Raises
    ------
    NotImplementedError
        When SYSCAL holds a Tsys spectrum (``TSYS_SPECTRUM``) or no TSYS column: CASA handles those.
    """
    started = _time.perf_counter()
    ms, caltable = Path(ms), Path(caltable)
    engine = table_engine()
    handle = open_table(ms / "SYSCAL", engine=engine)
    try:
        columns = handle.colnames()
        if "TSYS_SPECTRUM" in columns or "TSYS" not in columns:
            raise NotImplementedError("a SYSCAL table with a Tsys spectrum (or without TSYS)")
        if handle.nrows() == 0:
            raise NotImplementedError("an empty SYSCAL table")
        time = np.asarray(handle.getcol("TIME"), dtype=np.float64)
        antenna = np.asarray(handle.getcol("ANTENNA_ID"), dtype=np.int32)
        spw = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
        interval = np.asarray(handle.getcol("INTERVAL"), dtype=np.float64)
        tsys = np.asarray(handle.getcol("TSYS"), dtype=np.float64)
    finally:
        handle.close()
    if tsys.ndim == 1:
        tsys = tsys[:, None]
    layout = read_layout(ms, engine=engine)
    scan, field = scan_of_times(layout, time)
    keep = scan >= 0                                   # after the last scan: nothing to calibrate
    order = np.lexsort((antenna[keep], spw[keep], time[keep]))
    rows = np.flatnonzero(keep)[order]
    values = tsys[rows][:, None, :]                    # (nrow, 1 channel, npol)
    flag = ~(values > 0)
    setup = read_setup(ms, engine=engine)
    widths = _channel_widths(ms, engine)
    write_jones_table(caltable, ms, viscal="B TSYS", times=time[rows], field_ids=field[rows], spw_ids=spw[rows],
                      antenna_ids=antenna[rows], scan_numbers=scan[rows], intervals=interval[rows],
                      fparam=values, flag=flag, snr=np.full(values.shape, 0.1),
                      paramerr=np.full(values.shape, 0.1), spw_chan_freq=setup["chan_freq"].mean(axis=1),
                      spw_chan_width=widths)
    report = {"rows": int(rows.size), "flagged": int(flag.sum()), "dropped": int((~keep).sum()),
              "seconds": _time.perf_counter() - started}
    logger.info("gencal[dask-ms] tsys: {} SYSCAL row(s) -> {} ({} value(s) flagged, {} after the last scan "
                "dropped) in {:.1f}s", rows.size, caltable.name, report["flagged"], report["dropped"],
                report["seconds"])
    return report


def _channel_widths(ms: Path, engine: str) -> np.ndarray:
    """Total width of every subband [Hz]."""
    handle = open_table(ms / "SPECTRAL_WINDOW", engine=engine)
    try:
        width = np.asarray(handle.getcol("CHAN_WIDTH"), dtype=np.float64)
    finally:
        handle.close()
    return width.reshape(width.shape[0], -1).sum(axis=1)
