"""``flagdata(mode='tfcrop')`` natively: CASA's TFCrop auto-flagger on every part of a Multi-MS (dask-ms backend).

:func:`tfcrop_ms` writes the FLAG column ``casatasks.flagdata(mode='tfcrop', datacolumn=..., field=...,
timecutoff=..., freqcutoff=..., extendflags=False)`` writes with every other keyword at its default
(``ntime='scan'``, ``combinescans=False``, ``timefit='line'``, ``freqfit='poly'``, ``maxnpieces=7``,
``flagdimension='freqtime'``, ``usewindowstats='none'``, ``correlation=''``). It is a port of
``FlagAgentTimeFreqCrop.cc`` (casa6) and of the casacore least-squares code it calls, operation for
operation and in the same precision, so that every decision comes out the same:

* **Chunks.** ``ntime='scan'`` iterates the rows of one (observation, array, scan, field, data description)
  at a time, sorted by time (``FlagDataHandler::COMPLETE_SCAN_MAP_ANTENNA_PAIRS_ONLY``). Every baseline
  of a chunk — autocorrelations included — and every correlation is judged on its own
  (``correlation=''`` becomes one agent ``ABS_<product>`` per correlation product): a *plane* of
  ``|V|`` in float32 (``std::abs`` of the complex64 visibility, i.e. ``hypotf``), time (the baseline's
  rows in time order, not a regular grid) by channel, with the flags already in FLAG. FLAG_ROW is not read.
* **Two passes** (``flagdimension='freqtime'``): first in frequency (the plane averaged over time, a
  piecewise polynomial fitted to that spectrum — 1, 3, 5, 7, 7 pieces of degree 1, 1, 2, 3, 3 with 3-, 3-,
  2-, 2-, 2-sigma clipping of positive outliers and a 5-point running mean after each fit — every time step
  divided by it and iteratively clipped at ``freqcutoff`` sigma about 1), then the same in time with one
  straight line and ``timecutoff``, on the flags the first pass left.
* **Arithmetic.** Sums run in sequence in float32, as the C++ loops do (never numpy's pairwise sums); the
  polynomial pieces are solved in double precision with casacore's ``LSQFit`` normal equations, Cholesky-like
  decomposition and back substitution, the fitted values rounded to float32 term by term; the library is
  built without fused multiply-adds (checked in its machine code), so none are used here either. Empty or
  all-flagged spectra, zero averages, a single sample, a zero spread: the same IEEE infinities and NaNs as
  CASA, and therefore the same (non-)decisions.

Only FLAG is written (FLAG_ROW stays as it was, as CASA leaves it), and no HISTORY row is added. The parts of
a Multi-MS are flagged by parallel worker processes, one part each, reading and cropping one chunk at a time.
The compiled loops (numba, when installed; ``$VLBIPY_NUMBA=0`` turns them off) and the numpy implementation
(vectorised over the baselines and correlations of a chunk) give the same flags.

``datacolumn='corrected'`` reads CORRECTED_DATA; a field whose calibration was deferred
(:mod:`vlbipy.solvers.corrected`) is corrected from DATA while it is read, exactly as applycal would have
written it. Any other keyword or value raises :class:`NotImplementedError` before anything is written, so the
caller can hand the call to CASA unchanged.
"""
from __future__ import annotations

import math
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from . import kernels
from .msio import ms_parts, open_table, read_layout, read_setup, table_engine
from .workers import run_jobs

logger = get_logger()

F32 = np.float32
#: flagdata keywords reproduced only at these values (their ``casatasks.flagdata`` defaults).
REQUIRED_VALUES = {"mode": "tfcrop", "ntime": "scan", "combinescans": False, "timefit": "line", "freqfit": "poly",
                   "maxnpieces": 7, "flagdimension": "freqtime", "usewindowstats": "none", "halfwin": 1,
                   "extendflags": False, "channelavg": False, "timeavg": False, "correlation": "", "spw": "",
                   "antenna": "", "scan": "", "timerange": "", "uvrange": "", "intent": "", "array": "",
                   "observation": "", "feed": "", "action": "apply", "display": "", "savepars": False,
                   "flagbackup": False}
#: flagdata keywords that mode='tfcrop' does not use (``task_flagdata.py`` never hands them to the agent, or
#: only with an option above at a non-default value): any value gives CASA's result.
IGNORED_KEYWORDS = {"autocorr", "inpfile", "reason", "tbuff", "clipminmax", "clipoutside", "clipzeros",
                    "quackinterval", "quackmode", "quackincrement", "tolerance", "addantenna", "lowerlimit",
                    "upperlimit", "winsize", "timedev", "freqdev", "timedevscale", "freqdevscale", "spectralmax",
                    "spectralmin", "antint_ref_antenna", "minchanfrac", "verbose", "extendpols", "growtime",
                    "growfreq", "growaround", "flagneartime", "flagnearfreq", "minrel", "maxrel", "minabs",
                    "maxabs", "spwchan", "spwcorr", "basecnt", "fieldcnt", "name", "cmdreason", "outfile",
                    "overwrite", "chanbin", "timebin"}
#: Data columns: flagdata's ``datacolumn`` -> main-table column.
DATA_COLUMNS = {"data": "DATA", "corrected": "CORRECTED_DATA"}
#: Correlation products (CASA Stokes enumeration) a TFCrop agent handles: RR, RL, LR, LL, XX, XY, YX, YY.
CORRELATION_TYPES = set(range(5, 13))
#: Index columns that define a flagdata chunk (with ntime='scan') and the baselines in it.
META_COLUMNS = ("OBSERVATION_ID", "ARRAY_ID", "SCAN_NUMBER", "FIELD_ID", "DATA_DESC_ID", "ANTENNA1", "ANTENNA2",
                "TIME")
#: Frequency pass: CASA's freqfit='poly' fit (maxnpieces, maximum degree).
FREQ_FIT = (7, 4)
#: Time pass: CASA's timefit='line' fit.
TIME_FIT = (1, 1)
#: Iterations of the clipping about 1 in each pass, and the change of sigma that ends them early (a double).
CLIP_LOOPS = 5
CLIP_CONVERGED = 0.1

#: Set when the compiled loop could not be compiled on this machine: numpy is used from then on.
_FAILED: list = []

if kernels.numba is not None:
    _jit = kernels.numba.njit(cache=True, nogil=True, error_model="numpy")
    _parallel_jit = kernels.numba.njit(cache=True, nogil=True, parallel=True, error_model="numpy")
    _prange = kernels.numba.prange
else:                                                    # plain Python: the same code, slowly (tests only)
    def _jit(function):
        return function

    _parallel_jit = _jit
    _prange = range


# ---------------------------------------------------------------------------
# The port, one plane at a time (compiled with numba when it is installed)
# ---------------------------------------------------------------------------
@_jit
def _mean_unflagged(values, flags, n):
    """``calcMean``: float32 mean of the unflagged ``values[:n]`` (0 when none)."""
    total = F32(0.0)
    count = 0
    for i in range(n):
        if not flags[i]:
            total = F32(total + values[i])
            count += 1
    if count == 0:
        count = 1
    return F32(total / F32(count))


@_jit
def _std_about(values, flags, n, mean):
    """``calcStd(vect, flag, mean)``: float32 rms of the unflagged values about ``mean`` (NaN when none)."""
    total = F32(0.0)
    count = 0
    for i in range(n):
        if not flags[i]:
            deviation = F32(values[i] - mean)
            total = F32(total + F32(deviation * deviation))
            count += 1
    return F32(math.sqrt(F32(total / F32(count))))


@_jit
def _std_fit(values, flags, fit, n):
    """``calcStd(vect, flag, fit)``: float32 rms of the unflagged values about ``fit`` (0 when none)."""
    total = F32(0.0)
    count = 0
    for i in range(n):
        if not flags[i]:
            deviation = F32(values[i] - fit[i])
            total = F32(total + F32(deviation * deviation))
            count += 1
    if count == 0:
        count = 1
    return F32(math.sqrt(F32(total / F32(count))))


@_jit
def _line_fit(data, flags, fit, n, lim1, lim2):
    """``lineFit``: straight line through the unflagged ``data[lim1:lim2+1]`` against the index, in float32.

    The weight is ``1 / sd^2`` with ``sd`` the rms of *all* unflagged data about their mean, as in CASA.
    """
    mean = _mean_unflagged(data, flags, n)
    sd = _std_about(data, flags, n, mean)
    s = F32(0.0)
    sx = F32(0.0)
    sy = F32(0.0)
    sxx = F32(0.0)
    sxy = F32(0.0)
    for i in range(lim1, lim2 + 1):
        if not flags[i]:
            var = F32(sd * sd)
            index = F32(i)
            s = F32(s + F32(F32(1.0) / var))
            sx = F32(sx + F32(index / var))
            sy = F32(sy + F32(data[i] / var))
            sxx = F32(sxx + F32(F32(i * i) / var))
            sxy = F32(sxy + F32(F32(index * data[i]) / var))
    denominator = F32(F32(s * sxx) - F32(sx * sx))
    a = F32(F32(F32(sxx * sy) - F32(sx * sxy)) / denominator)
    b = F32(F32(F32(s * sxy) - F32(sx * sy)) / denominator)
    for i in range(lim1, lim2 + 1):
        fit[i] = F32(a + F32(b * F32(i)))


@_jit
def _poly_fit(data, flags, fit, n, lim1, lim2, deg):
    """``polyFit``: polynomial of degree ``deg`` in ``x = index + 1`` through the unflagged ``data[lim1:lim2+1]``.

    casacore ``LinearFit<Double>`` with unit weights: normal equations accumulated row by row (``makeNorm``),
    ``LSQFit::invert`` (no SVD: a dependent pivot leaves the coefficients at 0) and ``LSQFit::solveIt``; the
    values are summed term by term into a float32. Too few points for the degree: :func:`_line_fit`.
    """
    count = 0
    for i in range(lim1, lim2 + 1):
        if not flags[i]:
            count += 1
    if count <= deg:
        _line_fit(data, flags, fit, n, lim1, lim2)
        return
    nun = deg + 1
    norm = np.zeros((nun, nun))
    known = np.zeros(nun)
    equation = np.zeros(nun)
    for i in range(lim1, lim2 + 1):
        if flags[i]:
            continue
        x = float(i + 1)
        power = 1.0
        for j in range(nun):
            equation[j] = power
            power = power * x
        observed = float(data[i]) - 0.0                  # minus the value of the start polynomial (all zero)
        for j in range(nun):
            if equation[j] != 0.0:
                for k in range(j, nun):
                    norm[j, k] += equation[j] * 1.0 * equation[k]
        for j in range(nun):
            known[j] += equation[j] * observed
    for j in range(nun):
        if norm[j, j] == 0.0:
            norm[j, j] = 1.0
    solved = True
    for i in range(nun):
        pivot = norm[i, i]
        for i2 in range(i):
            pivot -= norm[i2, i] * norm[i2, i] / norm[i2, i2]
        if pivot * pivot / norm[i, i] <= 0.0:
            solved = False
            break
        norm[i, i] = pivot
        for i1 in range(i + 1, nun):
            for i2 in range(i):
                norm[i, i1] -= norm[i2, i] * norm[i2, i1] / norm[i2, i2]
    coefficients = np.zeros(nun)
    if solved:
        solution = np.zeros(nun)
        for i1 in range(nun):
            solution[i1] = known[i1]
            for i2 in range(i1):
                solution[i1] -= norm[i2, i1] * solution[i2] / norm[i2, i2]
        for i1 in range(nun - 1, -1, -1):
            for i2 in range(i1 + 1, nun):
                solution[i1] -= norm[i1, i2] * solution[i2]
            solution[i1] /= norm[i1, i1]
        for j in range(nun):
            coefficients[j] = 0.0 + solution[j]
    for i in range(lim1, lim2 + 1):
        x = float(i + 1)
        value = F32(0.0)
        power = 1.0
        for j in range(nun):
            value = F32(float(value) + coefficients[j] * power)
            power = power * x
        fit[i] = value


@_jit
def _fit_piecewise_poly(data, flags, fit, n, maxnpieces, maxdeg, tdata):
    """``fitPiecewisePoly``: the clipped piecewise fit of ``data[:n]``; ``flags`` gains the clipped points.

    ``tdata`` is a work vector of length ``n``. Zeros (flagged averages) are first replaced by their
    neighbours, as CASA does in place; a vector of zeros is flagged entirely.
    """
    for i in range(n):
        tdata[i] = data[i]
    for i in range(n):
        if tdata[i] == 0:
            if i == 0:
                ind = 0
                for j in range(1, n):
                    if tdata[j] != 0:
                        ind = j
                        break
                if ind == 0:
                    tdata[i] = F32(0.0)
                else:
                    tdata[i] = tdata[ind]
            else:
                indr = 0
                for j in range(i + 1, n):
                    if tdata[j] != 0:
                        indr = j
                        break
                indl = -1
                for j in range(i, -1, -1):
                    if tdata[j] != 0:
                        indl = j
                        break
                if indl == -1 and indr == 0:
                    tdata[i] = F32(0.0)
                if indl > -1 and indr == 0:
                    tdata[i] = tdata[indl]
                if indl == -1 and indr > 0:
                    tdata[i] = tdata[indr]
                if indl > -1 and indr > 0:
                    tdata[i] = F32(F32(tdata[indl] + tdata[indr]) / F32(2.0))
    for i in range(n):
        if tdata[i] == 0:
            flags[i] = True
    for i in range(n):
        fit[i] = tdata[i]
    deg = 1
    for j in range(5):
        npieces = min(2 * j + 1, maxnpieces)
        if j > 1:
            deg = 2
        if j > 2:
            deg = 3
        deg = min(deg, maxdeg)
        psize = n // npieces
        leftover_front = (n % npieces) // 2
        left = 0
        right = n - 1
        for p in range(npieces):
            if npieces > 1:
                left = leftover_front + p * psize
                right = left + psize
                if p == 0:
                    left = 0
                    right = leftover_front + psize
                if p == npieces - 1:
                    right = n - 1
            if deg == 1:
                _line_fit(tdata, flags, fit, n, left, right)
            else:
                _poly_fit(tdata, flags, fit, n, left, right, deg)
        for i in range(2, n - 2):                        # 5-point running mean, in place
            total = F32(0.0)
            for w in range(i - 2, i + 3):
                total = F32(total + fit[w])
            fit[i] = F32(total / F32(5.0))
        sd = _std_fit(tdata, flags, fit, n)
        limit = F32(F32(2.0) * sd) if j >= 2 else F32(F32(3.0) * sd)
        for i in range(n):
            if F32(tdata[i] - fit[i]) > limit:
                flags[i] = True


@_jit
def _fit_base_and_flag(values, flags, tol, maxnpieces, maxdeg):
    """``fitBaseAndFlag`` along axis 0 of a ``(n0, n1)`` plane: average over axis 1, fit, divide, clip.

    ``flags`` (same shape) is updated in place. ``tol`` is the float32 cutoff in sigma.
    """
    n0 = values.shape[0]
    n1 = values.shape[1]
    average = np.zeros(n0, dtype=np.float32)
    average_flag = np.zeros(n0, dtype=np.bool_)
    fit = np.zeros(n0, dtype=np.float32)
    work = np.zeros(n0, dtype=np.float32)
    all_flagged = True
    for i0 in range(n0):
        total = F32(0.0)
        count = 0
        for i1 in range(n1):
            if not flags[i0, i1]:
                total = F32(total + values[i0, i1])
                count += 1
        if count > 0:
            average[i0] = F32(total / F32(count))
            all_flagged = False
        else:
            average[i0] = F32(0.0)
            average_flag[i0] = True
    if all_flagged:
        return
    _fit_piecewise_poly(average, average_flag, fit, n0, maxnpieces, maxdeg, work)
    one = F32(1.0)
    for i1 in range(n1):
        for i0 in range(n0):
            average[i0] = F32(0.0)
            average_flag[i0] = flags[i0, i1]
            if not average_flag[i0]:
                average[i0] = F32(values[i0, i1] / fit[i0])
        previous = F32(0.0)
        for _ in range(CLIP_LOOPS):
            sd = _std_about(average, average_flag, n0, one)
            limit = F32(tol * sd)
            for i0 in range(n0):
                if not average_flag[i0] and abs(F32(average[i0] - one)) > limit:
                    average_flag[i0] = True
            if abs(float(F32(previous - sd))) < CLIP_CONVERGED:
                break
            previous = sd
        for i0 in range(n0):
            if average_flag[i0]:
                flags[i0, i1] = True


@_parallel_jit
def _crop_kernel(amplitude, flags, starts, freq_tol, time_tol):
    """Both passes on every (series, correlation) plane of ``amplitude`` (nrow, nchan, ncorr); flags in place."""
    nchan = amplitude.shape[1]
    ncorr = amplitude.shape[2]
    nseries = starts.size - 1
    for task in _prange(nseries * ncorr):
        series = task // ncorr
        corr = task % ncorr
        first = starts[series]
        ntime = starts[series + 1] - first
        values = np.empty((nchan, ntime), dtype=np.float32)
        plane_flags = np.empty((nchan, ntime), dtype=np.bool_)
        for t in range(ntime):
            for c in range(nchan):
                values[c, t] = amplitude[first + t, c, corr]
                plane_flags[c, t] = flags[first + t, c, corr]
        _fit_base_and_flag(values, plane_flags, freq_tol, FREQ_FIT[0], FREQ_FIT[1])
        values_t = np.empty((ntime, nchan), dtype=np.float32)
        flags_t = np.empty((ntime, nchan), dtype=np.bool_)
        for t in range(ntime):
            for c in range(nchan):
                values_t[t, c] = amplitude[first + t, c, corr]
                flags_t[t, c] = plane_flags[c, t]
        _fit_base_and_flag(values_t, flags_t, time_tol, TIME_FIT[0], TIME_FIT[1])
        for t in range(ntime):
            for c in range(nchan):
                flags[first + t, c, corr] = flags_t[t, c]


# ---------------------------------------------------------------------------
# The same, vectorised with numpy over the planes of equal length
# ---------------------------------------------------------------------------
def _seq_sum(values: np.ndarray, axis: int) -> np.ndarray:
    """Sum along ``axis`` one element after the other in the array's dtype (a C loop's sum, not a pairwise one)."""
    if values.shape[axis] == 0:
        return np.zeros(np.delete(np.array(values.shape), axis), dtype=values.dtype)
    return np.add.accumulate(values, axis=axis).take(-1, axis=axis)


def _np_mean_sd(tdata: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``calcMean`` and ``calcStd(vect, flag, mean)`` of every row of ``(Q, n)`` float32 data (``lineFit``'s)."""
    good = ~flags
    count = good.sum(axis=1)
    mean = _seq_sum(np.where(good, tdata, F32(0.0)), 1) / np.maximum(count, 1).astype(F32)
    deviation = tdata - mean[:, None]
    sd = np.sqrt(_seq_sum(np.where(good, deviation * deviation, F32(0.0)), 1) / count.astype(F32))
    return mean, sd


def _np_line_fit(tdata: np.ndarray, flags: np.ndarray, sd: np.ndarray, lim1: int, lim2: int) -> np.ndarray:
    """:func:`_line_fit` of ``tdata[:, lim1:lim2+1]`` for every row; returns the fitted values."""
    piece = slice(lim1, lim2 + 1)
    good = ~flags[:, piece]
    index = np.arange(lim1, lim2 + 1)
    index_f = index.astype(F32)
    square_f = (index * index).astype(F32)
    var = (sd * sd)[:, None]
    zero = F32(0.0)
    data = tdata[:, piece]
    s = _seq_sum(np.where(good, F32(1.0) / var, zero), 1)
    sx = _seq_sum(np.where(good, index_f / var, zero), 1)
    sy = _seq_sum(np.where(good, data / var, zero), 1)
    sxx = _seq_sum(np.where(good, square_f / var, zero), 1)
    sxy = _seq_sum(np.where(good, (index_f * data) / var, zero), 1)
    denominator = s * sxx - sx * sx
    a = (sxx * sy - sx * sxy) / denominator
    b = (s * sxy - sx * sy) / denominator
    return a[:, None] + b[:, None] * index_f


def _np_poly_fit(tdata: np.ndarray, flags: np.ndarray, lim1: int, lim2: int, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_poly_fit` of ``tdata[:, lim1:lim2+1]`` for every row.

    Returns ``(values, usable)``: the fitted float32 values and whether a row has more than ``deg`` unflagged
    points (the others take :func:`_np_line_fit` instead).
    """
    piece = slice(lim1, lim2 + 1)
    good = ~flags[:, piece]
    nun = deg + 1
    x = np.arange(lim1, lim2 + 1, dtype=np.float64) + 1.0
    powers = [np.ones_like(x)]
    for _ in range(2 * deg):
        powers.append(powers[-1] * x)                    # exact integers: any summation order gives the same
    norm = np.zeros((tdata.shape[0], nun, nun))
    for j in range(nun):
        for k in range(j, nun):
            norm[:, j, k] = np.where(good, powers[j] * powers[k], 0.0).sum(axis=1)
    observed = tdata[:, piece].astype(np.float64)
    known = np.stack([_seq_sum(np.where(good, powers[j] * observed, 0.0), 1) for j in range(nun)], axis=1)
    for j in range(nun):
        norm[:, j, j] = np.where(norm[:, j, j] == 0.0, 1.0, norm[:, j, j])
    solved = np.ones(tdata.shape[0], dtype=bool)
    for i in range(nun):
        pivot = norm[:, i, i].copy()
        for i2 in range(i):
            pivot = pivot - norm[:, i2, i] * norm[:, i2, i] / norm[:, i2, i2]
        solved &= ~(pivot * pivot / norm[:, i, i] <= 0.0)
        norm[:, i, i] = pivot
        for i1 in range(i + 1, nun):
            for i2 in range(i):
                norm[:, i, i1] = norm[:, i, i1] - norm[:, i2, i] * norm[:, i2, i1] / norm[:, i2, i2]
    solution = np.zeros((tdata.shape[0], nun))
    for i1 in range(nun):
        solution[:, i1] = known[:, i1]
        for i2 in range(i1):
            solution[:, i1] = solution[:, i1] - norm[:, i2, i1] * solution[:, i2] / norm[:, i2, i2]
    for i1 in range(nun - 1, -1, -1):
        for i2 in range(i1 + 1, nun):
            solution[:, i1] = solution[:, i1] - norm[:, i1, i2] * solution[:, i2]
        solution[:, i1] = solution[:, i1] / norm[:, i1, i1]
    coefficients = np.where(solved[:, None], 0.0 + solution, 0.0)
    values = np.zeros(observed.shape, dtype=F32)
    for j in range(nun):
        values = (values.astype(np.float64) + coefficients[:, j:j + 1] * powers[j]).astype(F32)
    return values, good.sum(axis=1) > deg


def _np_fill_zeros(tdata: np.ndarray) -> None:
    """``fitPiecewisePoly``'s in-place replacement of zeros by their neighbours, for every row of ``tdata``."""
    nrow, n = tdata.shape
    nonzero = tdata != 0
    has_next = np.zeros((nrow, n), dtype=bool)           # first non-zero value after each position
    next_value = np.zeros((nrow, n), dtype=F32)
    has, value = np.zeros(nrow, dtype=bool), np.zeros(nrow, dtype=F32)
    for i in range(n - 1, -1, -1):
        has_next[:, i], next_value[:, i] = has, value
        value = np.where(nonzero[:, i], tdata[:, i], value)
        has = has | nonzero[:, i]
    has_left, left = np.zeros(nrow, dtype=bool), np.zeros(nrow, dtype=F32)
    for i in range(n):
        empty = tdata[:, i] == 0
        if empty.any():
            if i == 0:
                filled = np.where(has_next[:, 0], next_value[:, 0], F32(0.0))
            else:
                after = has_next[:, i]
                filled = np.where(has_left & after, (left + next_value[:, i]) / F32(2.0),
                                  np.where(has_left, left, np.where(after, next_value[:, i], F32(0.0))))
            tdata[:, i] = np.where(empty, filled, tdata[:, i])
        now = tdata[:, i] != 0
        left = np.where(now, tdata[:, i], left)
        has_left |= now


def _np_fit_piecewise_poly(data: np.ndarray, flags: np.ndarray, maxnpieces: int, maxdeg: int) -> np.ndarray:
    """:func:`_fit_piecewise_poly` of every row of ``(Q, n)`` float32 ``data``; ``flags`` is updated in place."""
    n = data.shape[1]
    tdata = data.copy()
    _np_fill_zeros(tdata)
    flags |= tdata == 0
    fit = tdata.copy()
    deg = 1
    for j in range(5):
        npieces = min(2 * j + 1, maxnpieces)
        if j > 1:
            deg = 2
        if j > 2:
            deg = 3
        deg = min(deg, maxdeg)
        psize = n // npieces
        leftover_front = (n % npieces) // 2
        _, sd = _np_mean_sd(tdata, flags)
        for p in range(npieces):
            left, right = 0, n - 1
            if npieces > 1:
                left = leftover_front + p * psize
                right = left + psize
                if p == 0:
                    left, right = 0, leftover_front + psize
                if p == npieces - 1:
                    right = n - 1
            line = _np_line_fit(tdata, flags, sd, left, right)
            if deg == 1:
                fit[:, left:right + 1] = line
            else:
                poly, usable = _np_poly_fit(tdata, flags, left, right, deg)
                fit[:, left:right + 1] = np.where(usable[:, None], poly, line)
        for i in range(2, n - 2):                        # 5-point running mean, in place
            total = F32(0.0) + fit[:, i - 2]
            for w in range(i - 1, i + 3):
                total = total + fit[:, w]
            fit[:, i] = total / F32(5.0)
        good = ~flags
        deviation = tdata - fit
        sd_fit = np.sqrt(_seq_sum(np.where(good, deviation * deviation, F32(0.0)), 1)
                         / np.maximum(good.sum(axis=1), 1).astype(F32))
        flags |= deviation > (F32(2.0) if j >= 2 else F32(3.0)) * sd_fit[:, None]
    return fit


def _np_fit_base_and_flag(values: np.ndarray, flags: np.ndarray, tol, maxnpieces: int, maxdeg: int) -> np.ndarray:
    """:func:`_fit_base_and_flag` along axis 1 of ``(P, n0, n1)`` planes; returns the new flags (a copy)."""
    good = ~flags
    count = good.sum(axis=2)
    average = np.where(count > 0, _seq_sum(np.where(good, values, F32(0.0)), 2) / np.maximum(count, 1).astype(F32),
                       F32(0.0))
    average_flag = count == 0
    out = flags.copy()
    todo = np.flatnonzero(~average_flag.all(axis=1))
    if not todo.size:
        return out
    fit = _np_fit_piecewise_poly(average[todo], average_flag[todo], maxnpieces, maxdeg)
    plane_flags = flags[todo]
    ratio = np.where(plane_flags, F32(0.0), values[todo] / fit[:, :, None])
    deviation = ratio - F32(1.0)
    square = deviation * deviation
    distance = np.abs(deviation)
    previous = np.zeros((todo.size, values.shape[2]), dtype=F32)
    active = np.ones((todo.size, values.shape[2]), dtype=bool)
    tol = F32(tol)
    for _ in range(CLIP_LOOPS):
        keep = ~plane_flags
        sd = np.sqrt(_seq_sum(np.where(keep, square, F32(0.0)), 1) / keep.sum(axis=1).astype(F32))
        plane_flags |= keep & (distance > (tol * sd)[:, None, :]) & active[:, None, :]
        converged = np.abs(previous - sd).astype(np.float64) < CLIP_CONVERGED
        previous = np.where(active, sd, previous)
        active &= ~converged
        if not active.any():
            break
    out[todo] = plane_flags
    return out


def _np_crop_planes(amplitude: np.ndarray, flags: np.ndarray, freq_tol, time_tol) -> np.ndarray:
    """Both passes on ``(P, ntime, nchan)`` planes; returns the new flags."""
    after_freq = _np_fit_base_and_flag(amplitude.transpose(0, 2, 1), flags.transpose(0, 2, 1), freq_tol, *FREQ_FIT)
    return _np_fit_base_and_flag(amplitude, after_freq.transpose(0, 2, 1), time_tol, *TIME_FIT)


# ---------------------------------------------------------------------------
# Chunks and series
# ---------------------------------------------------------------------------
def numba_available() -> bool:
    """True when the compiled loop is used (numba installed, not disabled, and it compiles here)."""
    return kernels.numba is not None and not _FAILED


def crop_series(amplitude: np.ndarray, flags: np.ndarray, starts: np.ndarray, *, timecutoff: float = 4.0,
                freqcutoff: float = 3.0, use_numba: Optional[bool] = None) -> None:
    """Run TFCrop on every baseline series and correlation of one chunk; ``flags`` is updated in place.

    Parameters
    ----------
    amplitude : numpy.ndarray
        float32 ``(nrow, nchan, ncorr)``: ``|V|`` of the chunk's rows, grouped by baseline and in time order
        within each baseline.
    flags : numpy.ndarray
        bool, same shape: the flags before (in) and after (out).
    starts : numpy.ndarray
        ``(nseries + 1,)`` first row of every baseline series, and the row count at the end.
    timecutoff, freqcutoff : float
        flagdata's cutoffs (in sigma, used as float32).
    use_numba : bool, optional
        Force the compiled loop (True) or numpy (False); default: numba when available.
    """
    amplitude = np.ascontiguousarray(amplitude, dtype=F32)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    if use_numba is None:
        use_numba = numba_available()
    if use_numba and kernels.numba is not None:
        from .workers import in_thread_map, job_threads
        kernels.numba.set_num_threads(1 if in_thread_map() else
                                      max(1, min(job_threads(), kernels.numba.config.NUMBA_NUM_THREADS)))
        work = np.ascontiguousarray(flags, dtype=bool)
        try:
            _crop_kernel(amplitude, work, starts, F32(freqcutoff), F32(timecutoff))
        except kernels.numba.core.errors.NumbaError as exc:
            if not _FAILED:
                logger.warning("numba could not compile the tfcrop loop ({}); using numpy", str(exc).splitlines()[0])
            _FAILED.append(exc)
        else:
            if work is not flags:
                flags[...] = work
            return
    lengths = np.diff(starts)
    nchan, ncorr = amplitude.shape[1], amplitude.shape[2]
    with np.errstate(all="ignore"):                      # CASA's arithmetic, infinities and NaNs included
        for ntime in np.unique(lengths):
            series = np.flatnonzero(lengths == ntime)
            rows = starts[series][:, None] + np.arange(ntime)[None, :]
            planes = amplitude[rows].transpose(0, 3, 1, 2).reshape(-1, ntime, nchan)
            plane_flags = flags[rows].transpose(0, 3, 1, 2).reshape(-1, ntime, nchan)
            cropped = _np_crop_planes(planes, plane_flags, F32(freqcutoff), F32(timecutoff))
            flags[rows] = cropped.reshape(series.size, ncorr, ntime, nchan).transpose(0, 2, 3, 1)


def chunk_series(meta: dict, rows: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group rows as flagdata does with ``ntime='scan'``: chunks, then baselines in time order.

    Parameters
    ----------
    meta : dict
        The :data:`META_COLUMNS` of a table part.
    rows : numpy.ndarray
        Row numbers to group (the data selection).

    Returns
    -------
    list of (order, starts)
        Per chunk (one observation, array, scan, field and data description): its row numbers sorted by
        (antenna1, antenna2, time, row) and the first position of every baseline in them, plus their count.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if not rows.size:
        return []
    keys = [np.asarray(meta[name])[rows] for name in META_COLUMNS]
    order = np.lexsort([rows] + keys[::-1])              # last key first: ..., ANTENNA2, ANTENNA1 within the chunk
    chunk_keys = np.stack([k[order] for k in keys[:5]], axis=1)
    baseline_keys = np.stack([k[order] for k in keys[:7]], axis=1)
    chunk_edges = np.flatnonzero(np.any(chunk_keys[1:] != chunk_keys[:-1], axis=1)) + 1
    series_edges = np.flatnonzero(np.any(baseline_keys[1:] != baseline_keys[:-1], axis=1)) + 1
    bounds = np.concatenate([[0], chunk_edges, [rows.size]])
    out = []
    for first, stop in zip(bounds[:-1], bounds[1:]):
        inner = series_edges[(series_edges > first) & (series_edges < stop)] - first
        out.append((rows[order[first:stop]], np.concatenate([[0], inner, [stop - first]]).astype(np.int64)))
    return out


def _row_runs(sorted_rows: np.ndarray) -> list[tuple[int, int]]:
    """``(start, nrow)`` of every run of consecutive row numbers in ``sorted_rows``."""
    breaks = np.flatnonzero(np.diff(sorted_rows) != 1) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [sorted_rows.size]])
    return [(int(sorted_rows[a]), int(b - a)) for a, b in zip(starts, stops)]


def _read_chunk(handle, runs: list[tuple[int, int]], column: str, field: int, ddid: int,
                view: Optional[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Visibilities (complex64) and flags of the rows ``runs``, and the flagged count of each run on disk.

    A field of ``view`` (a deferred calibration) is corrected from DATA as applycal would have written it,
    with the calibration flags applycal would have added.
    """
    if view is not None and field in view["entries_by_field"]:
        from .apply_task import corrected_chunks
        segments = [{"start": start, "nrow": nrow, "field": field, "ddid": ddid} for start, nrow in runs]
        vis, flag, on_disk = [], [], [0] * len(runs)
        run = 0
        # The pieces come in row order and never straddle two runs (runs are not adjacent).
        for start, _, values, flags, _, _, flagged_before in corrected_chunks(
                handle, segments, entries_by_field=view["entries_by_field"], setup=view["setup"],
                columns=view["columns"], parang=view["parang"], chunk_rows=1 << 30, weights=None,
                calflags=view["write_flags"]):
            while start >= runs[run][0] + runs[run][1]:
                run += 1
            vis.append(np.asarray(values, dtype=np.complex64))
            flag.append(np.asarray(flags, dtype=bool))
            on_disk[run] += int(flagged_before)
        return np.concatenate(vis), np.concatenate(flag), on_disk
    vis = [np.asarray(handle.getcol(column, start, nrow), dtype=np.complex64) for start, nrow in runs]
    flag = [np.array(handle.getcol("FLAG", start, nrow), dtype=bool) for start, nrow in runs]
    return np.concatenate(vis), np.concatenate(flag), [int(np.count_nonzero(f)) for f in flag]


def tfcrop_part_job(*, path: str, engine: str, column: str, field_ids: Optional[list[int]], timecutoff: float,
                    freqcutoff: float, view: Optional[dict] = None) -> dict:
    """Worker job: TFCrop every chunk of one table part in place (FLAG only), one chunk in memory at a time.

    Parameters
    ----------
    path : str
        The table part (a sub-MS of a Multi-MS, or the MS).
    engine : str
        Table engine (:func:`vlbipy.solvers.msio.table_engine`).
    column : str
        DATA or CORRECTED_DATA.
    field_ids : list of int, optional
        Fields to flag (None: all).
    timecutoff, freqcutoff : float
        The cutoffs.
    view : dict, optional
        :func:`vlbipy.solvers.corrected.view` of the deferred fields (CORRECTED_DATA only).

    Returns
    -------
    dict
        ``rows``, ``chunks``, ``series`` and ``cells`` processed, ``flagged_before``/``flagged_after`` among
        those cells, ``seconds``.
    """
    started = _time.perf_counter()
    stats = {"rows": 0, "chunks": 0, "series": 0, "cells": 0, "flagged_before": 0, "flagged_after": 0}
    handle = open_table(path, readonly=False, engine=engine)
    try:
        nrows = handle.nrows()
        if nrows:
            meta = {name: np.asarray(handle.getcol(name)) for name in META_COLUMNS}
            rows = (np.arange(nrows) if field_ids is None
                    else np.flatnonzero(np.isin(meta["FIELD_ID"], np.asarray(field_ids, dtype=np.int64))))
            for order, starts in chunk_series(meta, rows):
                block_rows = np.sort(order)
                runs = _row_runs(block_rows)
                vis, flag, on_disk = _read_chunk(handle, runs, column, int(meta["FIELD_ID"][order[0]]),
                                                 int(meta["DATA_DESC_ID"][order[0]]), view)
                position = np.searchsorted(block_rows, order)
                amplitude = np.hypot(vis.real, vis.imag)[position]       # std::abs(complex<float>) = hypotf
                del vis
                cropped = flag[position]
                crop_series(amplitude, cropped, starts, timecutoff=timecutoff, freqcutoff=freqcutoff)
                flag[position] = cropped
                offset = 0
                for (start, nrow), before in zip(runs, on_disk):
                    block = flag[offset:offset + nrow]
                    after = int(np.count_nonzero(block))
                    if after != before:                  # flags are only ever added: a same count is no change
                        handle.putcol("FLAG", block, start)
                    stats["flagged_before"] += before
                    stats["flagged_after"] += after
                    offset += nrow
                stats["rows"] += int(order.size)
                stats["chunks"] += 1
                stats["series"] += int(starts.size - 1)
                stats["cells"] += int(flag.size)
    finally:
        handle.close()
    stats["seconds"] = _time.perf_counter() - started
    logger.debug("tfcrop[dask-ms]: {} - {} chunk(s), {} row(s), {} -> {} flagged cell(s), {:.2f}s",
                 Path(path).name, stats["chunks"], stats["rows"], stats["flagged_before"], stats["flagged_after"],
                 stats["seconds"])
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _same(value, required) -> bool:
    """Whether a flagdata keyword value means the same as the required one (strings case-insensitively)."""
    if isinstance(required, bool):
        return isinstance(value, (bool, np.bool_)) and bool(value) == required
    if isinstance(required, int):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer)) and int(value) == required
    return isinstance(value, str) and value.strip().lower() == required


def check_tfcrop_kwargs(kwargs: dict) -> None:
    """Raise NotImplementedError unless every flagdata keyword in ``kwargs`` is reproduced by :func:`tfcrop_ms`.

    ``kwargs`` excludes ``vis``, ``datacolumn``, ``field``, ``timecutoff`` and ``freqcutoff`` (arguments of
    :func:`tfcrop_ms`).
    """
    for key, value in kwargs.items():
        if key in IGNORED_KEYWORDS:
            continue
        if key not in REQUIRED_VALUES:
            raise NotImplementedError(f"tfcrop: keyword {key!r} is not reproduced natively")
        if not _same(value, REQUIRED_VALUES[key]):
            raise NotImplementedError(f"tfcrop: {key}={value!r} is not reproduced natively "
                                      f"(only {REQUIRED_VALUES[key]!r})")


def resolve_fields(field, field_names: list[str]) -> Optional[list[int]]:
    """Field ids of a flagdata ``field`` selection made of field names (comma-separated or a list).

    Returns None for no selection. Raises NotImplementedError for anything MSSelection would read otherwise
    (ids, ranges, patterns) or a name the FIELD table does not have.
    """
    if field is None:
        return None
    names = [str(n).strip() for n in field] if isinstance(field, (list, tuple)) else str(field).split(",")
    names = [n.strip() for n in names if n.strip()]
    if not names:
        return None
    ids: set[int] = set()
    for name in names:
        if name.isdigit() or any(c in name for c in "*?~<>^![]{}/\"'") or name not in field_names:
            raise NotImplementedError(f"tfcrop: field selection {name!r} is not a field name")
        ids.update(i for i, known in enumerate(field_names) if known == name)
    return sorted(ids)


def _check_correlations(ms, engine: str) -> None:
    """Raise NotImplementedError unless every polarization setup has distinct standard correlation products."""
    handle = open_table(Path(ms) / "POLARIZATION", engine=engine)
    try:
        corr_types = [np.asarray(handle.getcol("CORR_TYPE", row, 1)).reshape(-1) for row in range(handle.nrows())]
    finally:
        handle.close()
    for types in corr_types:
        values = [int(t) for t in types]
        if set(values) - CORRELATION_TYPES or len(set(values)) != len(values):
            raise NotImplementedError(f"tfcrop: correlation types {values} are not reproduced natively")


def _cutoff(name: str, value) -> float:
    """A flagdata cutoff as a finite float (CASA takes int, float or double)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise NotImplementedError(f"tfcrop: {name}={value!r} is not a number")
    if not math.isfinite(float(value)):
        raise NotImplementedError(f"tfcrop: {name}={value!r} is not finite")
    return float(value)


def tfcrop_ms(ms, *, datacolumn: str = "data", field=None, timecutoff: float = 4.0, freqcutoff: float = 3.0,
              workers: Optional[int] = None, **flagdata_kwargs) -> dict:
    """Flag ``ms`` in place as ``casatasks.flagdata(vis=ms, mode='tfcrop', ...)`` would (FLAG column only).

    Parameters
    ----------
    ms : str or pathlib.Path
        Measurement set or Multi-MS.
    datacolumn : str
        ``'data'`` or ``'corrected'`` (a deferred calibration is applied while reading).
    field : str or list of str, optional
        Field names (comma-separated) to flag; None or '' flags every field.
    timecutoff, freqcutoff : float
        flagdata's cutoffs in sigma.
    workers : int, optional
        Worker processes (one job per table part).
    **flagdata_kwargs
        Any other flagdata keyword (not ``vis``): accepted only at the values :func:`tfcrop_ms` reproduces
        (:data:`REQUIRED_VALUES`, or any value of :data:`IGNORED_KEYWORDS`).

    Returns
    -------
    dict
        ``rows`` and ``cells`` processed, ``flagged_before``, ``flagged_after``, ``newly_flagged`` cells,
        ``chunks``, ``parts`` and ``seconds``.

    Raises
    ------
    NotImplementedError
        A keyword, value or data layout this port does not reproduce (nothing has been written).
    RuntimeError
        The field selection holds no data (flagdata fails the same way).
    """
    started = _time.perf_counter()
    check_tfcrop_kwargs(flagdata_kwargs)
    column = DATA_COLUMNS.get(str(datacolumn).strip().lower())
    if column is None:
        raise NotImplementedError(f"tfcrop: datacolumn={datacolumn!r} is not reproduced natively")
    time_tol, freq_tol = _cutoff("timecutoff", timecutoff), _cutoff("freqcutoff", freqcutoff)
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    field_ids = resolve_fields(field, setup["field_names"])
    _check_correlations(ms, engine)
    parts = ms_parts(ms)
    handle = open_table(parts[0], engine=engine)
    try:
        columns = list(handle.colnames())
    finally:
        handle.close()
    view = None
    if column == "CORRECTED_DATA":
        from .corrected import view as corrected_view
        view = corrected_view(ms, column, field_ids, setup=setup, columns=columns)
        if column not in columns:
            present = set(np.unique(read_layout(ms, engine=engine)["field"]).tolist())
            wanted = present if field_ids is None else present & set(field_ids)
            if view is None or wanted - set(view["entries_by_field"]):
                raise NotImplementedError(f"tfcrop: {Path(ms).name} has no {column} column")
    jobs = [{"path": part, "engine": engine, "column": column, "field_ids": field_ids, "timecutoff": time_tol,
             "freqcutoff": freq_tol, "view": view} for part in parts]
    results = run_jobs("vlbipy.solvers.tfcrop_task:tfcrop_part_job", jobs, workers)
    total = {name: sum(r[name] for r in results)
             for name in ("rows", "chunks", "series", "cells", "flagged_before", "flagged_after")}
    if field_ids is not None and total["rows"] == 0:
        raise RuntimeError(f"flagdata selected no data: field={field!r} has no rows in {Path(ms).name}")
    total.update(newly_flagged=total["flagged_after"] - total["flagged_before"], parts=len(parts),
                 seconds=_time.perf_counter() - started)
    logger.info("flagdata[native tfcrop]: {} {} — {} chunk(s), {} row(s) in {} part(s): {} cell(s) newly flagged "
                "({:.2%} -> {:.2%} of the processed data), {:.1f}s", Path(ms).name, column, total["chunks"],
                total["rows"], total["parts"], total["newly_flagged"],
                total["flagged_before"] / max(total["cells"], 1), total["flagged_after"] / max(total["cells"], 1),
                total["seconds"])
    return total
