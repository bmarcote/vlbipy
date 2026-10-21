"""Correlator amplitude correction from the autocorrelations: the dask-ms backend's ``accor`` and ``smoothcal``.

:func:`run_accor` writes the table ``casatasks.accor(vis, caltable, solint=..., corrdepflags=...)`` writes with
every other keyword at its default, and :func:`smooth_caltable` the table ``casatasks.smoothcal`` makes of it
(``smoothtype`` ``'median'`` or ``'mean'``, any ``smoothtime``). Checked against CASA 6.7.6 on RSM07: every
cell of both tables is identical (only the ``CASA_Version`` keyword reads ``vlbipy``). Anything else raises
NotImplementedError so that the caller runs CASA instead.

What ``accor`` computes (``AccorJones::selfSolveOne`` fed by ``Calibrater::genericGatherAndSolve``), repeated here
operation by operation in float32 where CASA works in float:

* the data are the autocorrelation rows (``ANTENNA1 == ANTENNA2``) of DATA, every field, subband and scan;
  a row with FLAG_ROW set counts as flagged in every sample. Weights are re-derived from the noise:
  ``1 / sigma**2`` from SIGMA_SPECTRUM when the MS has both WEIGHT_SPECTRUM and SIGMA_SPECTRUM, else from SIGMA
  on every channel (``VisBufferImpl2::resetWeightsUsingSigma``), and set to 0 on flagged samples. Without ``corrdepflags`` a
  sample flagged in any correlation is flagged in all of them (its weight is kept);
* solution intervals are VI2 chunks (``MSIter2``): one per array, observation, scan, field, data description and
  time bin ``floor((TIME - bound) / solint)``, where ``bound`` is the first TIME of the (scan, field) group, in
  row order, minus 0.01 s (``MSSmartInterval``); ``solint`` is rounded to float32 as CASA passes it;
* channel averaging (``ChannelAverageTVI``, all channels into one): per row and correlation the weighted sum
  ``sum(V w)`` and ``sum(w)`` over the unflagged channels (all channels when none is), added one channel after
  the other, then divided as libc++ divides ``std::complex<float>`` by a complex number (with the logb/scalbn
  scaling, which a plain float division does not reproduce bit for bit); flagged when every channel is;
* time averaging over the interval (``AveragingTvi2``) per antenna and correlation: ``sum(V w) / sum(w)`` over the
  unflagged rows (all rows when none is), in row order, divided component by component;
* the solution of each antenna and polarization (correlations 0 and 3 of four, 0 and 1 of two) is
  ``sqrt(|V|)`` with the flagged data set to zero first, flagged where ``|V| == 0``; antennas absent from the
  interval are flagged with ``CPARAM = 1``. ``PARAMERR`` and ``SNR`` are 0 (CASA writes the parameter errors
  into SNR), ``INTERVAL`` is 0, ``ANTENNA2`` is -1, ``WEIGHT`` is undefined; an interval with no unflagged
  weight at all gives no rows;
* ``TIME`` is the weighted mean of the averaged rows' time centroids (``SDBList::aggregateTimeCentroid``): each
  row's TIME_CENTROID mean weighted by the summed weights of its correlations, then the rows weighted by
  their parallel-hand weights;
* rows sorted (stably) by subband, time and antenna (``NewCalTable::writeToDisk``). Subtables as for every CASA
  calibration table (see :mod:`vlbipy.solvers.gaincurve_task`); SPECTRAL_WINDOW has one channel per subband: for
  a solved subband the sequential mean of the channel frequencies, the summed widths as CHAN_WIDTH and
  EFFECTIVE_BW, and the frequency itself as RESOLUTION and TOTAL_BANDWIDTH (as CASA's
  ``SolvableVisCal::setCTFrequencies`` writes them); an unsolved subband keeps the collapsed MS values and is
  flagged.

What ``smoothcal`` does (``casa::smoothCT``): per subband, field and antenna with more than one solution, every
solution is replaced by the median or mean of the unflagged solutions within ``(t - smoothtime/2,
t + smoothtime/2]``, smoothed in amplitude and phase. A solution with at least one unflagged neighbour in its
window becomes unflagged (CASA unflags it); one without stays flagged and unchanged. The mean is a float32
running sum divided by the count; the median of an even number of values is the mean of the two middle ones
when the series holds at most 100 solutions and the lower middle one otherwise (casacore's ``median`` of a masked
array decides on the size of the whole array, not on the number of values in the window). The table is written
from CASA's in-memory copy, so its storage manager is named ``MSMTAB_1``. Only real, non-negative CPARAMs (zero
phase, which every accor table holds) are smoothed here, where the amplitude/phase round trip is exact.

The visibilities are read by worker processes, one job per Multi-MS part, each returning only its per-interval,
per-antenna sums; the solutions are formed in the calling process.
"""
from __future__ import annotations

import re
import shutil
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..casacore_format.aipsio import TableRef
from ..casacore_format.table import Table
from ..casacore_format.writer import create_table
from ..logging_utils import get_logger
from .caltable import _array_column, _create, _link_subtables, history_table_desc, main_table_desc
from .gaincurve_task import HISTORY_ORDER, SUBTABLE_DESCS, write_calibration_subtable
from .msio import ms_parts, open_table, table_engine
from .workers import run_jobs

logger = get_logger()

VISCAL = "Accor Jones"
CASA_VERSION = "vlbipy"
SUBTABLES = ("OBSERVATION", "ANTENNA", "FIELD", "SPECTRAL_WINDOW", "HISTORY")
#: accor keywords this module reproduces, with the only value accepted for each (CASA's defaults).
ACCOR_DEFAULTS = {"field": "", "spw": "", "intent": "", "selectdata": None, "timerange": "", "antenna": "",
                  "scan": "", "observation": "", "msselect": "", "combine": "", "append": False,
                  "docallib": False, "callib": "", "gaintable": [], "gainfield": [], "interp": [], "spwmap": []}
#: smoothcal keywords this module reproduces, with the only value accepted for each.
SMOOTHCAL_DEFAULTS = {"field": "", "ratesmooth": False}
#: ``MSIter2::discernEnforcedTimeBounds``: the time bins of a (scan, field) start this long before its first row.
BOUND_OFFSET = 0.01
#: ``VbAvg``: an average is closed when a row is later than this fraction of the averaging time after its first.
TIME_DISTANCE_FRACTION = 0.999
#: casacore ``median`` of a masked array: when the whole array (the series, not the window) has at most this many
#: values, an even count of unmasked ones gives the mean of the two middle values, else the lower middle one.
MEDIAN_EVEN_MEAN_MAX = 100
#: Storage manager name of a table written by CASA's smoothcal (an in-memory copy renames ``MSMTAB``).
SMOOTHED_MANAGER = "MSMTAB_1"
#: Rows spanned by one read of a worker (autocorrelations are read out of this many consecutive rows).
READ_ROWS = 25_000
#: Index columns of the main table read for every autocorrelation row.
INDEX_COLUMNS = ("ANTENNA1", "ANTENNA2", "TIME", "TIME_CENTROID", "SCAN_NUMBER", "FIELD_ID", "DATA_DESC_ID",
                 "OBSERVATION_ID", "ARRAY_ID")
#: Columns of a solution interval's key, in the order VI2 sorts the chunks.
CHUNK_KEYS = ("array", "obs", "scan", "field", "ddid", "bound", "bin")
_UNITS = {"s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


# ---------------------------------------------------------------------------
# accor
# ---------------------------------------------------------------------------
def run_accor(vis, caltable, *, solint="30s", corrdepflags: bool = True, workers: Optional[int] = None,
              **accor_kwargs) -> dict:
    """Write the ``Accor Jones`` table ``casatasks.accor(vis, caltable, solint, corrdepflags)`` writes.

    Parameters
    ----------
    vis : str or pathlib.Path
        The measurement set or Multi-MS (only read).
    caltable : str or pathlib.Path
        The table to write; an existing table there is replaced. Nothing is written when no interval has an
        unflagged autocorrelation (CASA then writes no table either).
    solint : str or float
        Solution interval: seconds (a number, or a string with unit ``s``, ``min``, ``h`` or ``d``), or
        ``'inf'``/``''`` (one solution per scan, field and subband).
    corrdepflags : bool
        CASA's ``corrdepflags``: False flags a sample in all correlations when it is flagged in any.
    workers : int, optional
        Worker processes (default: the pool's size).
    **accor_kwargs
        The other keywords of ``casatasks.accor``; each must hold its default (:data:`ACCOR_DEFAULTS`).

    Returns
    -------
    dict
        ``rows`` (table rows), ``intervals`` (solution intervals with data), ``solved`` (intervals with
        solutions), ``flagged`` (flagged solutions among the rows), ``spws`` (subbands solved), ``parts``,
        ``seconds``.

    Raises
    ------
    NotImplementedError
        For a request this module does not reproduce: a non-default keyword, ``solint='int'`` or a frequency
        solution interval, FLOAT_DATA, subbands of different shapes, scans not in time order, an interval that
        CASA's time averaging would split, or an antenna's interval spread over several Multi-MS parts.
    """
    started = _time.perf_counter()
    vis, caltable = Path(vis), Path(caltable)
    _check_accor_keywords(accor_kwargs)
    interval = parse_solint(solint)
    engine = table_engine()
    setup = read_accor_setup(vis, engine=engine)
    parts = ms_parts(vis)
    firsts = run_jobs("vlbipy.solvers.accor_task:accor_index_job",
                      [{"path": part, "engine": engine} for part in parts], workers)
    bounds = time_bounds(firsts)
    jobs = [{"path": part, "engine": engine, "bounds": bounds, "interval": interval, "corrdepflags": bool(corrdepflags),
             "weight_column": setup["weight_column"], "read_rows": READ_ROWS} for part in parts]
    results = run_jobs("vlbipy.solvers.accor_task:accor_part_job", jobs, workers)
    segments = merge_segments(results)
    solution = solve_intervals(segments, nant=setup["nant"], ncorr=setup["ncorr"],
                               ddid_to_spw=setup["ddid_to_spw"])
    report = {"rows": int(solution["antenna"].size), "intervals": int(solution["intervals"]),
              "solved": int(solution["solved"]), "flagged": int(np.count_nonzero(solution["flag"])),
              "spws": int(np.unique(solution["spw"]).size), "parts": len(parts)}
    if solution["antenna"].size:
        write_accor_table(caltable, vis, solution, setup)
    else:
        logger.warning("accor[native]: {} has no unflagged autocorrelation; no table written (as CASA)", vis.name)
    report["seconds"] = _time.perf_counter() - started
    logger.info("accor[native]: {} solint={!r} corrdepflags={} -> {} ({} row(s), {} of {} interval(s) solved, {} "
                "flagged solution(s), {} part(s)) in {:.2f}s", vis.name, solint, bool(corrdepflags), caltable.name,
                report["rows"], report["solved"], report["intervals"], report["flagged"], report["parts"],
                report["seconds"])
    return report


def _check_accor_keywords(accor_kwargs: dict) -> None:
    """Raise NotImplementedError unless every keyword of ``accor_kwargs`` holds its CASA default."""
    for name, value in accor_kwargs.items():
        if name not in ACCOR_DEFAULTS:
            raise NotImplementedError(f"accor keyword {name!r} is not reproduced natively")
        if name == "selectdata":
            continue                      # with every selection empty, selectdata True or False is the same
        default = ACCOR_DEFAULTS[name]
        if isinstance(default, list):
            empty = value is None or (isinstance(value, (list, tuple)) and all(item in ("", None) for item in value))
            if not empty:
                raise NotImplementedError(f"accor {name}={value!r} (only the empty default is reproduced)")
        elif value not in (default, None) and not (default == "" and value in ([], ())):
            raise NotImplementedError(f"accor {name}={value!r} (only the default {default!r} is reproduced)")


def parse_solint(solint) -> float:
    """CASA's time solution interval in seconds, as the float32 value VI2 iterates with (inf for ``'inf'``).

    Raises
    ------
    NotImplementedError
        For ``'int'``, a frequency interval (``'30s,4ch'``) or a unit other than s, sec, min, h, d.
    """
    if isinstance(solint, (int, float)) and not isinstance(solint, bool):
        seconds = float(solint)
    else:
        text = str(solint).strip()
        if "," in text:
            raise NotImplementedError(f"accor solint={solint!r} (a frequency solution interval)")
        if text == "" or "INF" in text.upper():
            return float("inf")
        if "INT" in text.upper():
            raise NotImplementedError("accor solint='int' (no time averaging) is not reproduced natively")
        match = re.fullmatch(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-z]*)", text)
        if match is None or match.group(2) not in ("",) + tuple(_UNITS):
            raise NotImplementedError(f"accor solint={solint!r} (unit not reproduced natively)")
        seconds = float(match.group(1)) * _UNITS.get(match.group(2) or "s")
    if not seconds > 0:
        raise NotImplementedError(f"accor solint={solint!r} (not a positive interval)")
    return float(np.float32(seconds))


def read_accor_setup(vis, *, engine: str) -> dict:
    """The metadata a solve needs: antennas, subbands, correlations and which weights CASA would use.

    Returns
    -------
    dict
        ``nant``, ``ncorr``, ``nchan``, ``ddid_to_spw`` (array), ``weight_column`` (``"SIGMA_SPECTRUM"`` or
        ``"SIGMA"``), ``ms_name``.

    Raises
    ------
    NotImplementedError
        FLOAT_DATA (or no DATA), subbands with different channel counts, polarizations with different
        correlation counts, or a correlation count other than 1, 2 or 4.
    """
    vis = Path(vis)
    part = ms_parts(vis)[0]
    handle = open_table(part, engine=engine)
    try:
        columns = set(handle.colnames())
    finally:
        handle.close()
    if "FLOAT_DATA" in columns or "DATA" not in columns:
        raise NotImplementedError(f"{vis.name}: accor of FLOAT_DATA (or without DATA) is not reproduced natively")
    sub = {}
    for name, cols in (("ANTENNA", ("NAME",)), ("SPECTRAL_WINDOW", ("NUM_CHAN",)), ("POLARIZATION", ("NUM_CORR",)),
                       ("DATA_DESCRIPTION", ("SPECTRAL_WINDOW_ID", "POLARIZATION_ID"))):
        handle = open_table(vis / name, engine=engine)
        try:
            sub[name] = {col: np.asarray(handle.getcol(col)) for col in cols}
        finally:
            handle.close()
    ddid_to_spw = sub["DATA_DESCRIPTION"]["SPECTRAL_WINDOW_ID"].astype(np.int64)
    ncorr = sub["POLARIZATION"]["NUM_CORR"].astype(np.int64)[sub["DATA_DESCRIPTION"]["POLARIZATION_ID"]]
    nchan = sub["SPECTRAL_WINDOW"]["NUM_CHAN"].astype(np.int64)[ddid_to_spw]
    if np.unique(ncorr).size != 1 or np.unique(nchan).size != 1:
        raise NotImplementedError(f"{vis.name}: data descriptions of different shapes (channels {np.unique(nchan)}, "
                                  f"correlations {np.unique(ncorr)}) are not read natively")
    if int(ncorr[0]) not in (1, 2, 4):
        raise NotImplementedError(f"{vis.name}: {int(ncorr[0])} correlations (accor uses 1, 2 or 4)")
    weight_column = "SIGMA_SPECTRUM" if {"WEIGHT_SPECTRUM", "SIGMA_SPECTRUM"} <= columns else "SIGMA"
    return {"nant": int(sub["ANTENNA"]["NAME"].size), "ncorr": int(ncorr[0]), "nchan": int(nchan[0]),
            "ddid_to_spw": ddid_to_spw, "weight_column": weight_column, "ms_name": vis.absolute().name}


def accor_index_job(*, path: str, engine: str) -> dict:
    """Worker job: the first autocorrelation TIME of every (scan, field) of one table part, in row order.

    Returns
    -------
    dict
        Parallel arrays ``scan``, ``field``, ``time`` (one entry per (scan, field) present).
    """
    handle = open_table(path, engine=engine)
    try:
        if handle.nrows() == 0:
            return {"scan": np.zeros(0, np.int64), "field": np.zeros(0, np.int64), "time": np.zeros(0)}
        auto = np.flatnonzero(np.asarray(handle.getcol("ANTENNA1")) == np.asarray(handle.getcol("ANTENNA2")))
        scan = np.asarray(handle.getcol("SCAN_NUMBER"), dtype=np.int64)[auto]
        field = np.asarray(handle.getcol("FIELD_ID"), dtype=np.int64)[auto]
        time = np.asarray(handle.getcol("TIME"), dtype=np.float64)[auto]
    finally:
        handle.close()
    keys = np.stack([scan, field], axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    return {"scan": scan[first], "field": field[first], "time": time[first]}


def time_bounds(firsts: list[dict]) -> np.ndarray:
    """The time bounds of ``MSSmartInterval``: the first TIME of each (scan, field), minus 0.01 s.

    ``firsts`` are :func:`accor_index_job` results in part order; a (scan, field) found in several parts takes
    its time from the first of them (the Multi-MS reads its parts in order). The bounds come in (scan, field)
    order.

    Raises
    ------
    NotImplementedError
        When the bounds are not strictly increasing (scans out of time order): CASA's binary search over them
        is then not well defined.
    """
    found = {}
    for result in firsts:
        for scan, field, time in zip(result["scan"].tolist(), result["field"].tolist(), result["time"].tolist()):
            found.setdefault((scan, field), time)
    bounds = np.array([found[key] - BOUND_OFFSET for key in sorted(found)], dtype=np.float64)
    if bounds.size > 1 and np.any(np.diff(bounds) <= 0):
        raise NotImplementedError("accor: scans are not in time order (CASA's solution-interval bounds would not be "
                                  "sorted)")
    return bounds


def sigma_weights(sigma: np.ndarray) -> np.ndarray:
    """``1 / sigma**2`` in float32 (``1.0f / square(s)``), 0 where ``sigma <= 0``."""
    sigma = np.asarray(sigma, dtype=np.float32)
    with np.errstate(divide="ignore"):
        weight = np.float32(1.0) / (sigma * sigma)
    return np.where(sigma > 0, weight, np.float32(0.0)).astype(np.float32)


def divide_like_libcxx(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / std::complex<float>(denominator, 0)`` as libc++ computes it, in float32.

    libc++ scales the divisor by ``2**-logb(|c|)`` (``c'`` in [1, 2)) and returns
    ``scalbn((a c') / (c' c'), -logb)`` for the real part (``(b c') / (c' c')`` for the imaginary one), which
    differs in the last bit from ``a / c``. ``denominator`` must be positive and finite.
    """
    real = np.asarray(numerator.real, dtype=np.float32)
    imag = np.asarray(numerator.imag, dtype=np.float32)
    divisor = np.asarray(denominator, dtype=np.float32)
    _, exponent = np.frexp(divisor)
    logb = (exponent - 1).astype(np.int32)
    scaled = np.ldexp(divisor, -logb).astype(np.float32)
    square = (scaled * scaled).astype(np.float32)
    out_real = np.ldexp(((real * scaled).astype(np.float32) / square).astype(np.float32), -logb)
    out_imag = np.ldexp(((imag * scaled).astype(np.float32) / square).astype(np.float32), -logb)
    return (out_real.astype(np.float32) + 1j * out_imag.astype(np.float32)).astype(np.complex64)


def channel_average(data: np.ndarray, flag: np.ndarray, weight: np.ndarray, *,
                    corrdepflags: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What VI2's calibration and channel-averaging layers give a solve, per row and correlation.

    Parameters
    ----------
    data : complex64 (nrow, nchan, ncorr)
    flag : bool (nrow, nchan, ncorr)
    weight : float32 (nrow, nchan, ncorr)
        ``1 / sigma**2`` (:func:`sigma_weights`); zeroed here where ``flag`` is set.
    corrdepflags : bool
        False: a sample flagged in one correlation is flagged in all (its weight is kept, as CASA's
        ``CalSolvingVi2::corrIndepFlags`` runs after the weights of flagged samples were zeroed).

    Returns
    -------
    (value, flagged, weight)
        complex64 (nrow, ncorr) weighted channel average, bool (nrow, ncorr) all channels flagged, float32
        (nrow, ncorr) summed weight of the channels averaged.
    """
    weight = np.where(flag, np.float32(0.0), np.asarray(weight, dtype=np.float32)).astype(np.float32)
    if not corrdepflags:
        flag = np.broadcast_to(flag.any(axis=2, keepdims=True), flag.shape)
    good = ~flag
    any_good = good.any(axis=1)
    use = np.where(any_good[:, None, :], good, True)
    nrow, nchan, ncorr = data.shape
    acc_real = np.zeros((nrow, ncorr), dtype=np.float32)
    acc_imag = np.zeros((nrow, ncorr), dtype=np.float32)
    norm = np.zeros((nrow, ncorr), dtype=np.float32)
    for chan in range(nchan):
        used, w = use[:, chan], weight[:, chan]
        acc_real = np.where(used, acc_real + (data[:, chan].real * w).astype(np.float32), acc_real)
        acc_imag = np.where(used, acc_imag + (data[:, chan].imag * w).astype(np.float32), acc_imag)
        norm = np.where(used, norm + w, norm).astype(np.float32)
    positive = norm > 0
    value = np.zeros((nrow, ncorr), dtype=np.complex64)
    if positive.any():
        value[positive] = divide_like_libcxx(acc_real[positive] + 1j * acc_imag[positive], norm[positive])
    return value, ~any_good, norm


def accor_part_job(*, path: str, engine: str, bounds: np.ndarray, interval: float, corrdepflags: bool,
                   weight_column: str, read_rows: int = READ_ROWS) -> dict:
    """Worker job: the time-averaged autocorrelations of every (interval, antenna) of one table part.

    The autocorrelation rows are channel averaged (:func:`channel_average`) and then averaged over their
    solution interval (:func:`average_segments`), each (interval, antenna) in row order.

    Returns
    -------
    dict
        One entry per (interval, antenna) segment: the :data:`CHUNK_KEYS` and ``antenna`` int arrays,
        ``first_row`` (row of the part where the segment starts), ``value`` complex64 (nseg, ncorr), ``weight``
        float32 (nseg, ncorr), ``flag`` bool (nseg, ncorr), ``centroid`` and ``center`` float64 (the averaged
        TIME_CENTROID and the middle of the averaged TIMEs), ``rows`` (rows averaged).
    """
    handle = open_table(path, engine=engine)
    try:
        if handle.nrows() == 0:
            return empty_segments()
        index = {name: np.asarray(handle.getcol(name)) for name in INDEX_COLUMNS}
        auto = np.flatnonzero(index["ANTENNA1"] == index["ANTENNA2"])
        if not auto.size:
            return empty_segments()
        row_flag = (np.asarray(handle.getcol("FLAG_ROW"), dtype=bool) if "FLAG_ROW" in handle.colnames()
                    else np.zeros(index["TIME"].size, dtype=bool))
        columns = ("DATA", "FLAG", weight_column)
        value, flagged, weight = [], [], []
        for block in _row_blocks(auto, read_rows):
            cells = {name: _read_rows(handle, name, block) for name in columns}
            sigma = cells[weight_column]
            if weight_column == "SIGMA":
                sigma = np.broadcast_to(sigma[:, None, :], cells["FLAG"].shape)
            flag = np.asarray(cells["FLAG"], dtype=bool) | row_flag[block][:, None, None]
            averaged = channel_average(np.asarray(cells["DATA"], dtype=np.complex64), flag, sigma_weights(sigma),
                                       corrdepflags=corrdepflags)
            value.append(averaged[0])
            flagged.append(averaged[1])
            weight.append(averaged[2])
    finally:
        handle.close()
    rows = {name: index[name][auto] for name in INDEX_COLUMNS}
    keys = chunk_keys(rows, bounds, interval)
    return average_segments(keys, auto, rows["TIME"].astype(np.float64), rows["TIME_CENTROID"].astype(np.float64),
                            np.concatenate(value), np.concatenate(flagged), np.concatenate(weight), interval)


def _row_blocks(rows: np.ndarray, span: int) -> list[np.ndarray]:
    """``rows`` (ascending) cut into blocks that each lie within ``span`` consecutive rows."""
    blocks, start = [], 0
    while start < rows.size:
        stop = int(np.searchsorted(rows, rows[start] + span, side="left"))
        blocks.append(rows[start:max(stop, start + 1)])
        start = max(stop, start + 1)
    return blocks


def _read_rows(handle, name: str, rows: np.ndarray) -> np.ndarray:
    """The cells of ``rows`` (ascending, within one read span) of column ``name``, row-major."""
    first, count = int(rows[0]), int(rows[-1] - rows[0] + 1)
    if hasattr(handle, "getcolview"):
        cells = handle.getcolview(name, first, count)
    else:
        cells = handle.getcol(name, first, count)
    return np.array(cells[rows - first])


def chunk_keys(rows: dict, bounds: np.ndarray, interval: float) -> dict:
    """The solution interval (VI2 chunk) of each row: :data:`CHUNK_KEYS` as int64 arrays.

    ``bound`` is the index of the last time bound at or before the row's TIME, ``bin`` the time bin
    ``floor((TIME - bound) / interval)`` (0 for an infinite interval).
    """
    time = np.asarray(rows["TIME"], dtype=np.float64)
    bound = np.searchsorted(bounds, time, side="right") - 1
    if np.any(bound < 0):
        raise NotImplementedError("accor: a row lies before the first scan bound")
    if np.isfinite(interval):
        time_bin = np.floor((time - bounds[bound]) / interval).astype(np.int64)
    else:
        time_bin = np.zeros(time.size, dtype=np.int64)
    return {"array": rows["ARRAY_ID"].astype(np.int64), "obs": rows["OBSERVATION_ID"].astype(np.int64),
            "scan": rows["SCAN_NUMBER"].astype(np.int64), "field": rows["FIELD_ID"].astype(np.int64),
            "ddid": rows["DATA_DESC_ID"].astype(np.int64), "bound": bound.astype(np.int64), "bin": time_bin,
            "antenna": rows["ANTENNA1"].astype(np.int64)}


def empty_segments(ncorr: int = 0) -> dict:
    """An :func:`accor_part_job` result without segments."""
    out = {name: np.zeros(0, dtype=np.int64) for name in CHUNK_KEYS + ("antenna", "first_row", "rows")}
    out.update(value=np.zeros((0, ncorr), np.complex64), weight=np.zeros((0, ncorr), np.float32),
               flag=np.zeros((0, ncorr), bool), centroid=np.zeros(0), center=np.zeros(0))
    return out


def _sequential_sums(start: np.ndarray, length: np.ndarray, values: list[np.ndarray], used: np.ndarray,
                     dtype) -> list[np.ndarray]:
    """Per segment, the sums of each array of ``values`` over its ``used`` rows, added one row after the other.

    Rows of segment ``s`` are ``start[s] .. start[s] + length[s] - 1``; ``values`` arrays are (nrow, ...) and
    ``used`` broadcasts against them. Each sum starts at 0 and takes the rows in order, as CASA's accumulators.
    """
    sums = [np.zeros((start.size,) + v.shape[1:], dtype=dtype) for v in values]
    for k in range(int(length.max(initial=0))):
        live = np.flatnonzero(length > k)
        rows = start[live] + k
        for total, v in zip(sums, values):
            take = used[rows] if used.ndim == v.ndim else used[rows].reshape((-1,) + (1,) * (v.ndim - 1))
            total[live] = np.where(take, total[live] + v[rows], total[live]).astype(dtype)
    return sums


def average_segments(keys: dict, row_index: np.ndarray, time: np.ndarray, centroid: np.ndarray, value: np.ndarray,
                     flag: np.ndarray, weight: np.ndarray, interval: float) -> dict:
    """``AveragingTvi2`` over each (solution interval, antenna) segment of channel-averaged rows.

    Per correlation the average is ``sum(V w) / sum(w)`` over the unflagged rows (all rows when none is; the
    sum itself when the weights sum to 0), flagged when every row is. The time centroid is the mean of
    ``TIME_CENTROID`` weighted by each row's summed correlation weights over the rows with some unflagged
    correlation (unit weights over all rows when there is none), as offsets from the segment's first TIME.

    Raises
    ------
    NotImplementedError
        When TIME decreases within a segment, or spans more than 0.999 of the interval (CASA would then close
        the average early and solve from the first part only).
    """
    order = np.lexsort((row_index,) + tuple(keys[name] for name in reversed(CHUNK_KEYS + ("antenna",))))
    keys = {name: values[order] for name, values in keys.items()}
    row_index, time, centroid = row_index[order], time[order], centroid[order]
    value, flag, weight = value[order], flag[order], weight[order]
    nrow = row_index.size
    stacked = np.stack([keys[name] for name in CHUNK_KEYS + ("antenna",)], axis=1)
    change = np.flatnonzero(np.any(np.diff(stacked, axis=0) != 0, axis=1)) + 1
    start = np.concatenate([[0], change]).astype(np.int64)
    length = np.diff(np.concatenate([start, [nrow]])).astype(np.int64)
    segment = np.repeat(np.arange(start.size), length)
    first_time = time[start]
    if np.any(np.diff(time)[np.diff(segment) == 0] < 0):
        raise NotImplementedError("accor: TIME decreases within a solution interval")
    if np.isfinite(interval) and np.any(time - first_time[segment] > interval * TIME_DISTANCE_FRACTION):
        raise NotImplementedError("accor: an interval spans more than CASA's averaging time (it would be split)")
    any_good = np.logical_or.reduceat(~flag, start, axis=0)
    used = np.where(any_good[segment], ~flag, True)
    products = [(value.real * weight).astype(np.float32), (value.imag * weight).astype(np.float32), weight]
    acc_real, acc_imag, total = _sequential_sums(start, length, products, used, np.float32)
    positive = total > 0
    safe = np.where(positive, total, np.float32(1.0))
    out_real = np.where(positive, (acc_real / safe).astype(np.float32), acc_real)
    out_imag = np.where(positive, (acc_imag / safe).astype(np.float32), acc_imag)
    row_flagged = flag.all(axis=1)
    row_good = np.logical_or.reduceat(~row_flagged, start)
    adjusted = np.zeros(nrow, dtype=np.float64)
    for corr in range(weight.shape[1]):
        adjusted = adjusted + weight[:, corr].astype(np.float64)
    use_row = np.where(row_good[segment], ~row_flagged, True)
    row_weight = np.where(row_good[segment], adjusted, 1.0)
    offsets = (centroid - first_time[segment]) * row_weight
    norm, weighted, count = _sequential_sums(start, length, [row_weight, offsets, np.ones(nrow)], use_row, np.float64)
    last_used = np.maximum.reduceat(np.where(use_row, np.arange(nrow), -1), start)
    divisor = np.where(norm != 0, norm, count)
    out = {name: keys[name][start] for name in CHUNK_KEYS + ("antenna",)}
    out.update(first_row=row_index[start].astype(np.int64), rows=length,
               value=(out_real + 1j * out_imag).astype(np.complex64), weight=total.astype(np.float32),
               flag=~any_good, centroid=weighted / divisor + first_time,
               center=first_time + (time[last_used] - first_time) / 2)
    return out


def merge_segments(results: list[dict]) -> dict:
    """The segments of every part in solution order: by interval (VI2's chunk order), then antenna by first row.

    Raises
    ------
    NotImplementedError
        When one (interval, antenna) has rows in two parts (CASA would average them across the parts).
    """
    ncorr = max((r["value"].shape[1] for r in results if r["value"].size), default=0)
    pieces = [r for r in results if r["antenna"].size]
    if not pieces:
        return empty_segments(ncorr)
    merged = {name: np.concatenate([r[name] for r in pieces]) for name in pieces[0]}
    merged["part"] = np.concatenate([np.full(r["antenna"].size, i, np.int64) for i, r in enumerate(results)
                                     if r["antenna"].size])
    order = np.lexsort((merged["first_row"], merged["part"]) + tuple(merged[name] for name in reversed(CHUNK_KEYS)))
    merged = {name: values[order] for name, values in merged.items()}
    stacked = np.stack([merged[name] for name in CHUNK_KEYS + ("antenna",)], axis=1)
    if np.unique(stacked, axis=0).shape[0] != stacked.shape[0]:
        raise NotImplementedError("accor: an antenna's autocorrelations of one interval are split over several parts")
    return merged


def solve_intervals(segments: dict, *, nant: int, ncorr: int, ddid_to_spw: np.ndarray) -> dict:
    """The accor solutions of every interval (``AccorJones::selfSolveOne`` and ``keepNCT``), sorted for the table.

    Returns
    -------
    dict
        Row arrays ``time``, ``field``, ``spw``, ``scan``, ``obs``, ``antenna`` and ``cparam`` complex64
        (nrow, 1, 2), ``flag`` bool (nrow, 1, 2), sorted by (spw, time, antenna); ``intervals`` (intervals with
        rows) and ``solved`` (intervals that gave solutions).
    """
    pol_corr = {1: (0, 0), 2: (0, 1), 4: (0, 3)}[ncorr] if ncorr else (0, 0)
    stacked = np.stack([segments[name] for name in CHUNK_KEYS], axis=1) if segments["antenna"].size else \
        np.zeros((0, len(CHUNK_KEYS)), np.int64)
    change = np.flatnonzero(np.any(np.diff(stacked, axis=0) != 0, axis=1)) + 1
    starts = np.concatenate([[0], change]).astype(np.int64) if stacked.shape[0] else np.zeros(0, np.int64)
    stops = np.concatenate([starts[1:], [stacked.shape[0]]]).astype(np.int64)
    rows = {name: [] for name in ("time", "field", "spw", "scan", "obs", "antenna", "cparam", "flag")}
    solved = 0
    for lo, hi in zip(starts.tolist(), stops.tolist()):
        solution = solve_interval(segments["value"][lo:hi], segments["weight"][lo:hi], segments["flag"][lo:hi],
                                  segments["centroid"][lo:hi], segments["center"][lo:hi], pol_corr)
        if solution is None:
            continue
        solved += 1
        cparam = np.ones((nant, 2), dtype=np.complex64)
        good = np.zeros((nant, 2), dtype=bool)
        antennas = segments["antenna"][lo:hi]
        cparam[antennas] = solution["cparam"]
        good[antennas] = solution["good"]
        if ncorr == 1:
            cparam[:, 1] = 0.0                       # "For scalar data, set 'other' pol soln to zero"
        rows["time"].append(np.full(nant, solution["time"]))
        for name, key in (("field", "field"), ("scan", "scan"), ("obs", "obs")):
            rows[name].append(np.full(nant, segments[key][lo], dtype=np.int64))
        rows["spw"].append(np.full(nant, ddid_to_spw[segments["ddid"][lo]], dtype=np.int64))
        rows["antenna"].append(np.arange(nant, dtype=np.int64))
        rows["cparam"].append(cparam)
        rows["flag"].append(~good)
    if not solved:
        out = {name: np.zeros(0, dtype=np.int64) for name in ("field", "spw", "scan", "obs", "antenna")}
        out.update(time=np.zeros(0), cparam=np.zeros((0, 1, 2), np.complex64), flag=np.zeros((0, 1, 2), bool),
                   intervals=int(starts.size), solved=0)
        return out
    out = {name: np.concatenate(values) for name, values in rows.items()}
    order = np.lexsort((out["antenna"], out["time"], out["spw"]))
    out = {name: values[order] for name, values in out.items()}
    out["cparam"] = out["cparam"].reshape(-1, 1, 2)
    out["flag"] = out["flag"].reshape(-1, 1, 2)
    out.update(intervals=int(starts.size), solved=solved)
    return out


def solve_interval(value: np.ndarray, weight: np.ndarray, flag: np.ndarray, centroid: np.ndarray, center: np.ndarray,
                   pol_corr: tuple) -> Optional[dict]:
    """One interval's solutions from its averaged rows (in VI2's baseline order), or None without usable data.

    The interval is solved when some unflagged correlation has a positive weight (``SDBList::Ok``). Cross hands
    then lose their weight (``enforceSolveWeights``), flagged data and weights are zeroed, and the time is
    ``sum(w t) / sum(w)`` over the rows with ``w`` the float32 sum of the row's weights (their mean TIME when
    every weight is 0). Each polarization's solution is ``sqrt(|V|)``, good where ``|V| > 0``.

    Returns
    -------
    dict or None
        ``time`` (float), ``cparam`` complex64 (nrow, 2) (1 where not good), ``good`` bool (nrow, 2).
    """
    weight = np.where(flag, np.float32(0.0), weight).astype(np.float32)
    if not np.any(~flag & (weight > 0)):
        return None
    if weight.shape[1] > 2:
        weight[:, 1:3] = 0.0
    value = np.where(flag, np.complex64(0.0), value)
    row_weight = np.zeros(weight.shape[0], dtype=np.float32)
    for corr in range(weight.shape[1]):
        row_weight = (row_weight + weight[:, corr]).astype(np.float32)
    total = np.add.accumulate(row_weight.astype(np.float64))[-1]
    if total > 0:
        time = np.add.accumulate(row_weight.astype(np.float64) * centroid)[-1] / total
    else:
        time = np.add.accumulate(center)[-1] / center.size
    cparam = np.ones((value.shape[0], 2), dtype=np.complex64)
    good = np.zeros((value.shape[0], 2), dtype=bool)
    for pol, corr in enumerate(pol_corr):
        amplitude = np.hypot(value[:, corr].real, value[:, corr].imag).astype(np.float32)
        good[:, pol] = amplitude > 0
        cparam[:, pol] = np.where(good[:, pol], np.sqrt(amplitude), np.float32(1.0))
    return {"time": float(time), "cparam": cparam, "good": good}


def accor_main_desc() -> dict:
    """Column descriptions of an ``Accor Jones`` main table (CASA's: the fringe-table layout with CPARAM)."""
    return {("CPARAM" if name == "FPARAM" else name): (_array_column("complex") if name == "FPARAM" else desc)
            for name, desc in main_table_desc().items()}


def write_accor_table(caltable, vis, solution: dict, setup: dict) -> Path:
    """Write the main table (one StandardStMan, like CASA) and its five subtables at ``caltable``."""
    caltable, vis = Path(caltable).absolute(), Path(vis)
    nrow = int(solution["antenna"].size)
    values = {"TIME": solution["time"].astype(np.float64), "FIELD_ID": solution["field"].astype(np.int32),
              "SPECTRAL_WINDOW_ID": solution["spw"].astype(np.int32), "ANTENNA1": solution["antenna"].astype(np.int32),
              "ANTENNA2": np.full(nrow, -1, dtype=np.int32), "INTERVAL": np.zeros(nrow, dtype=np.float64),
              "SCAN_NUMBER": solution["scan"].astype(np.int32), "OBSERVATION_ID": solution["obs"].astype(np.int32),
              "CPARAM": solution["cparam"].astype(np.complex64), "PARAMERR": np.zeros((nrow, 1, 2), np.float32),
              "FLAG": solution["flag"].astype(bool), "SNR": np.zeros((nrow, 1, 2), np.float32)}
    keywords = {"ParType": "Complex", "MSName": setup["ms_name"], "VisCal": VISCAL, "PolBasis": "unknown",
                "CASA_Version": CASA_VERSION}
    keywords.update(_link_subtables(caltable, SUBTABLES))
    _create(caltable, accor_main_desc(), nrow, values, keywords=keywords, info=("Calibration", VISCAL))
    for name in ("OBSERVATION", "ANTENNA", "FIELD"):
        write_calibration_subtable(vis / name, caltable / name, SUBTABLE_DESCS[name])
    solved = np.unique(solution["spw"])
    write_calibration_subtable(vis / "SPECTRAL_WINDOW", caltable / "SPECTRAL_WINDOW", SUBTABLE_DESCS["SPECTRAL_WINDOW"],
                               transform=lambda columns, count: solution_spectral_windows(columns, count, solved))
    history = history_table_desc()
    _create(caltable / "HISTORY", {name: history[name] for name in HISTORY_ORDER}, 0, {})
    return caltable


def solution_spectral_windows(values: dict, nrow: int, solved: np.ndarray) -> dict:
    """SPECTRAL_WINDOW values of an accor table: one channel per subband, as CASA writes them.

    A solved subband (``SolvableVisCal::setCTFrequencies`` from VI2's channel-averaged frequencies) gets the mean
    channel frequency (summed channel by channel, divided by the count) as CHAN_FREQ, RESOLUTION and
    TOTAL_BANDWIDTH and the summed widths as CHAN_WIDTH and EFFECTIVE_BW; the others keep what
    ``createMemCalTable2`` makes of the MS values (mean frequency, summed widths, resolutions and effective
    bandwidths, the MS TOTAL_BANDWIDTH) and are flagged.
    """
    out = dict(values)
    reduced = {name: np.zeros((nrow, 1), dtype=np.float64) for name in ("CHAN_FREQ", "CHAN_WIDTH", "RESOLUTION",
                                                                          "EFFECTIVE_BW")}
    total = np.asarray(values["TOTAL_BANDWIDTH"], dtype=np.float64).copy()
    for row in range(nrow):
        cells = {name: np.asarray(values[name][row], dtype=np.float64).reshape(-1) for name in reduced}
        mean_freq = np.add.accumulate(cells["CHAN_FREQ"])[-1] / cells["CHAN_FREQ"].size
        width = np.add.accumulate(cells["CHAN_WIDTH"])[-1]
        if row in solved:
            reduced["CHAN_FREQ"][row, 0] = reduced["RESOLUTION"][row, 0] = total[row] = mean_freq
            reduced["CHAN_WIDTH"][row, 0] = reduced["EFFECTIVE_BW"][row, 0] = width
        else:
            reduced["CHAN_FREQ"][row, 0] = mean_freq
            reduced["CHAN_WIDTH"][row, 0] = width
            reduced["RESOLUTION"][row, 0] = np.add.accumulate(cells["RESOLUTION"])[-1]
            reduced["EFFECTIVE_BW"][row, 0] = np.add.accumulate(cells["EFFECTIVE_BW"])[-1]
    out.update(reduced)
    out["TOTAL_BANDWIDTH"] = total
    out["NUM_CHAN"] = np.ones(nrow, dtype=np.int32)
    out["FLAG_ROW"] = ~np.isin(np.arange(nrow), solved)
    return out


# ---------------------------------------------------------------------------
# smoothcal
# ---------------------------------------------------------------------------
def smooth_caltable(vis, tablein, caltable, *, smoothtype: str = "median", smoothtime: float = 1800.0,
                    **smoothcal_kwargs) -> dict:
    """Write what ``casatasks.smoothcal(vis, tablein, caltable, smoothtype=..., smoothtime=...)`` writes.

    Parameters
    ----------
    vis : str or pathlib.Path
        The measurement set (must exist, as for CASA; not read).
    tablein : str or pathlib.Path
        An ``Accor Jones`` table (written by CASA's ``accor`` or :func:`run_accor`).
    caltable : str or pathlib.Path
        The smoothed table to write (replaced if it exists); ``''`` overwrites ``tablein`` as CASA does.
    smoothtype : str
        ``'median'`` or ``'mean'``.
    smoothtime : float
        Filter width in seconds (> 0).
    **smoothcal_kwargs
        ``field`` (empty) and ``ratesmooth`` (False) only.

    Returns
    -------
    dict
        ``rows``, ``series`` (subband/field/antenna series smoothed), ``unflagged`` (flagged solutions that the
        smoothing unflagged, as CASA does), ``flagged`` (flagged solutions left), ``seconds``.

    Raises
    ------
    NotImplementedError
        Another table type, a CPARAM with a nonzero phase, rows out of time order, or a non-default keyword.
    RuntimeError
        An unknown ``smoothtype`` or ``smoothtime <= 0`` (CASA fails on them too).
    """
    started = _time.perf_counter()
    for name, value in smoothcal_kwargs.items():
        if name not in SMOOTHCAL_DEFAULTS:
            raise NotImplementedError(f"smoothcal keyword {name!r} is not reproduced natively")
        if value not in (SMOOTHCAL_DEFAULTS[name], None) and not (name == "field" and value in ([], ())):
            raise NotImplementedError(f"smoothcal {name}={value!r} (only the default is reproduced)")
    if smoothtype not in ("mean", "median"):
        raise RuntimeError(f"smoothcal: unrecognized smooth type {smoothtype!r}")
    if not float(smoothtime) > 0:
        raise RuntimeError("smoothcal: please specify a strictly positive smoothtime")
    if not Path(vis).exists():
        raise ValueError(f"Visibility data set not found - please verify the name ({vis})")
    tablein = Path(tablein).absolute()
    caltable = Path(caltable).absolute() if str(caltable) else tablein
    table = read_caltable(tablein)
    if table["keywords"].get("VisCal") != VISCAL or "CPARAM" not in table["values"]:
        raise NotImplementedError(f"smoothcal of {table['keywords'].get('VisCal')!r} tables is not reproduced natively")
    cparam = np.asarray(table["values"]["CPARAM"], dtype=np.complex64)
    if np.any(cparam.imag != 0) or np.any(cparam.real < 0):
        raise NotImplementedError("smoothcal: CPARAM with a nonzero phase (the amplitude/phase round trip is not "
                                  "reproduced natively)")
    values = table["values"]
    amplitude = cparam.real.astype(np.float32)
    flag = np.asarray(values["FLAG"], dtype=bool).copy()
    before = int(np.count_nonzero(flag))
    times = np.asarray(values["TIME"], dtype=np.float64)
    series = 0
    for rows in smoothing_groups(values):
        if rows.size < 2:
            continue
        series += 1
        for chan in range(amplitude.shape[1]):
            for pol in range(amplitude.shape[2]):
                new, ok = smooth_series(times[rows], amplitude[rows, chan, pol], ~flag[rows, chan, pol],
                                        smoothtype=smoothtype, smoothtime=float(smoothtime))
                amplitude[rows, chan, pol] = new
                flag[rows, chan, pol] = ~ok
    values = dict(values, CPARAM=amplitude.astype(np.complex64), FLAG=flag)
    order = np.lexsort((values["ANTENNA2"], values["ANTENNA1"], values["TIME"], values["SPECTRAL_WINDOW_ID"]))
    values = {name: _take(column, order) for name, column in values.items()}
    write_smoothed_table(caltable, tablein, table, values)
    report = {"rows": int(order.size), "series": series, "unflagged": int(before - np.count_nonzero(flag)),
              "flagged": int(np.count_nonzero(flag)), "seconds": _time.perf_counter() - started}
    logger.info("smoothcal[native]: {} smoothtype={} smoothtime={:g}s -> {} ({} row(s), {} series, {} solution(s) "
                "unflagged by the smoothing, {} still flagged) in {:.2f}s", tablein.name, smoothtype,
                float(smoothtime), caltable.name, report["rows"], report["series"], report["unflagged"],
                report["flagged"], report["seconds"])
    return report


def smoothing_groups(values: dict) -> list[np.ndarray]:
    """The rows of each (subband, field, ANTENNA1, ANTENNA2) series, each in table order (``CTIter``'s stable
    iteration), the series in ascending key order."""
    keys = np.stack([np.asarray(values[name], dtype=np.int64) for name in ("SPECTRAL_WINDOW_ID", "FIELD_ID",
                                                                           "ANTENNA1", "ANTENNA2")], axis=1)
    if not keys.shape[0]:
        return []
    order = np.lexsort(keys.T[::-1])
    change = np.flatnonzero(np.any(np.diff(keys[order], axis=0) != 0, axis=1)) + 1
    return np.split(order, change)


def smooth_series(times: np.ndarray, values: np.ndarray, ok: np.ndarray, *, smoothtype: str,
                  smoothtime: float) -> tuple[np.ndarray, np.ndarray]:
    """``casa::smoothCT`` on one series of float32 values (one subband, field, antenna and parameter).

    Each value becomes the median or mean of the ``ok`` values with ``t_i - smoothtime/2 < t <= t_i +
    smoothtime/2`` and is then ok; one without such values keeps its value and is not ok. ``times`` must be
    non-decreasing (a calibration table is sorted by time).

    Returns
    -------
    (values, ok)
        float32 smoothed values and the new ok mask.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    ok = np.asarray(ok, dtype=bool)
    if np.any(np.diff(times) < 0):
        raise NotImplementedError("smoothcal: solutions out of time order")
    half = float(smoothtime) / 2.0
    low = np.searchsorted(times, times - half, side="right")
    high = np.searchsorted(times, times + half, side="right")
    width = int((high - low).max(initial=0))
    window = np.full((times.size, width), np.nan, dtype=np.float32)
    taken = np.zeros((times.size, width), dtype=bool)
    for k in range(width):
        index = low + k
        inside = index < high
        safe = np.minimum(index, times.size - 1)
        taken[:, k] = inside & ok[safe]
        window[:, k] = np.where(taken[:, k], values[safe], np.nan)
    count = taken.sum(axis=1)
    if smoothtype == "mean":
        total = np.zeros(times.size, dtype=np.float32)
        for k in range(width):
            total = np.where(taken[:, k], total + window[:, k], total).astype(np.float32)
        smoothed = (total / np.maximum(count, 1).astype(np.float32)).astype(np.float32)
    else:
        smoothed = casacore_median(window, count, take_even_mean=times.size <= MEDIAN_EVEN_MEAN_MAX)
    return np.where(count > 0, smoothed, values).astype(np.float32), count > 0


def casacore_median(window: np.ndarray, count: np.ndarray, *, take_even_mean: bool) -> np.ndarray:
    """casacore's ``median(MaskedArray, sorted=False)`` of each row of ``window`` (``count`` values, NaN elsewhere).

    An odd count gives the middle value; an even one ``T(0.5 * (a + b))`` of the two middle values when
    ``take_even_mean`` (casacore: the masked array has at most :data:`MEDIAN_EVEN_MEAN_MAX` elements, masked
    ones included), else the lower middle one. Rows with ``count == 0`` give 0.
    """
    ordered = np.sort(window, axis=1)
    rows = np.arange(window.shape[0])
    lower = np.maximum((count - 1) // 2, 0)
    if ordered.shape[1] == 0:
        return np.zeros(window.shape[0], dtype=np.float32)
    low_value = ordered[rows, np.minimum(lower, ordered.shape[1] - 1)]
    high_value = ordered[rows, np.minimum(lower + 1, ordered.shape[1] - 1)]
    pair_mean = ((low_value + high_value).astype(np.float32) * np.float32(0.5)).astype(np.float32)
    even_mean = (count % 2 == 0) & bool(take_even_mean)
    return np.where(count > 0, np.where(even_mean, pair_mean, low_value), np.float32(0.0)).astype(np.float32)


def _take(column, order: np.ndarray):
    """Rows ``order`` of a column read by :func:`read_caltable` (an array, or a list of cells)."""
    if isinstance(column, np.ndarray):
        return column[order]
    return [column[i] for i in order.tolist()]


def read_caltable(path) -> dict:
    """Every main column (``values``), the column descriptions (``columns``), ``keywords`` and ``info`` of a table."""
    with Table(path) as table:
        nrow = table.nrows()
        columns = list(table.dat.columns)
        values = {column.name: (table.getcol(column.name) if nrow else None) for column in columns}
        keywords = dict(table.dat.keywords)
        info = table.info()
    return {"nrow": nrow, "columns": columns, "values": values, "keywords": keywords, "info": info}


def write_smoothed_table(caltable: Path, tablein: Path, table: dict, values: dict) -> Path:
    """Write the smoothed table: the input's columns, keywords and ``table.info`` holding ``values``, and its
    subtables copied, SPECTRAL_WINDOW with FLAG_ROW set on the subbands without rows (``flagAbsentSpws``).

    CASA writes the smoothed table from an in-memory copy, whose storage manager (main table and subtables) is
    renamed ``MSMTAB_1``; that name is kept. The table is written beside ``caltable`` and then moved in place,
    so ``caltable`` may be ``tablein``.
    """
    staging = caltable.with_name(caltable.name + ".vlbipy_smoothing")
    shutil.rmtree(staging, ignore_errors=True)
    names = [name for name, value in table["keywords"].items() if isinstance(value, TableRef)]
    keywords = {name: value for name, value in table["keywords"].items() if not isinstance(value, TableRef)}
    keywords.update(_link_subtables(staging, names))
    _write_like(staging, table, values, keywords, len(values["TIME"]))
    for name in names:
        sub = read_caltable(tablein / name)
        sub_values = dict(sub["values"])
        if name == "SPECTRAL_WINDOW":
            sub_values["FLAG_ROW"] = ~np.isin(np.arange(sub["nrow"]), np.unique(values["SPECTRAL_WINDOW_ID"]))
        sub_keywords = {key: value for key, value in sub["keywords"].items() if not isinstance(value, TableRef)}
        _write_like(staging / name, sub, sub_values, sub_keywords, sub["nrow"])
    if caltable.exists():
        shutil.rmtree(caltable)
    staging.rename(caltable)
    return caltable


def _write_like(path: Path, table: dict, values: dict, keywords: dict, nrow: int) -> Path:
    """Write ``nrow`` rows of ``values`` as a table with the columns and ``table.info`` of ``table`` (a
    :func:`read_caltable` result), all in one StandardStMan named ``MSMTAB_1``."""
    info = table["info"]
    return create_table(path, table["columns"], nrow, values, keywords=keywords, info=(info["type"], info["subType"]),
                        readme=info["readme"], manager_name=SMOOTHED_MANAGER)
