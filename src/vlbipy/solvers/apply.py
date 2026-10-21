"""Apply CASA calibration tables to a block of visibilities with numpy (on-the-fly applycal).

Row layout (as read from an MS by dask-ms / casacore): vis (nrow, nchan, ncorr) complex, flag (nrow, nchan, ncorr)
bool, weight (nrow, nchan, ncorr) float, antenna1/antenna2/time/spw (nrow,).

Conventions (see /tmp/vlbipy_spec/CASA_FRINGE_CONVENTIONS.md):
- V_ij = g_i conj(g_j) V_true for ANTENNA1 = i, ANTENNA2 = j; correction V_corr = V_ij / (g_i conj(g_j)).
- Per correlation: RR uses (g_i^R, g_j^R), LL (g_i^L, g_j^L), RL (g_i^R, g_j^L), LR (g_i^L, g_j^R).
- A visibility whose antenna gain is flagged or missing is flagged (CASA applymode='calflag').
- With calwt the weights are scaled by |g_i g_j|^2 (noise scales like the data: sigma_corr = sigma / |g_i g_j|) for
  the table kinds in WEIGHT_CAL_KINDS, and set to 0 where the solution of such a table is flagged or missing.

Supported VisCal kinds: "G Jones", "T Jones" (CPARAM (1, npol)), "B Jones" (CPARAM (nchan_t, npol)), "Fringe Jones"
(FPARAM (1, 8) = (phi0, delay_ns, rate, disp) x 2 pols, evaluated with ``vlbipy.solvers.fringe.predict_phase``),
"B TSYS" (FPARAM (1, npol) Tsys [K]; gain = 1/sqrt(Tsys) so that V_corr = V sqrt(Tsys_i Tsys_j) like CASA).
Gain curves ("EPowerCurve"/"EGainCurve") are evaluated from their stored power polynomials and per-row elevation.

Time interpolation: "nearest" takes the nearest solution; "linear" interpolates amplitude and phase separately
between the two bracketing solutions (CASA interpolates amp/phase, not re/im) and uses the nearest solution beyond
the ends.  Flagged solutions are NOT skipped: as in CASA, a time whose nearest solution (or either bracketing
solution) is flagged is flagged.  For "Fringe Jones" the rate extrapolates the phase from the solution time, and
"linear" interpolates the parameters between the bracketing rows with a rate-aware phase unwrap (see
`_fringe_grid`, CASA CTRateAwareTimeInterp1).

Frequency interpolation (second token of ``interp``, default "linear"): flagged channels of a solution are
interpolated across from its unflagged ones; the ``...flag`` modes keep them flagged (see `_interp_freqs`).

Evaluation: a table is evaluated on the whole (subband, antenna, time, hand) grid of a block in one pass, with
no loop over antennas or subbands. Its rows are indexed once (`_lookup`), and a single search then finds the
solutions around every grid point (`_brackets`). Every grid value depends only on its own subband, antenna,
time and channel, so a row is corrected the same way whichever block it is evaluated in.

casatools is imported lazily; getcol returns Fortran order so array columns are transposed (see caltable.py).
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field, replace
from typing import ClassVar
from pathlib import Path

import numpy as np

from vlbipy.solvers.fringe import TWO_PI, k_disp

log = logging.getLogger(__name__)

AMPLITUDE_ONLY_KINDS = ("B TSYS", "EPowerCurve", "EGainCurve")
#: Table kinds whose gains scale the weights under ``calwt`` in CASA. Bandpasses never do (CASA does not
#: calibrate weights with channel-dependent Jones terms) and fringe solutions have unit modulus.
WEIGHT_CAL_KINDS = ("B TSYS", "EPowerCurve", "EGainCurve", "G Jones", "T Jones")
#: Marker for ``gainfield='nearest'`` until the target field is known (see :func:`normalise_entries`).
NEAREST_FIELD = "nearest"
GAIN_CURVE_KINDS = ("EPowerCurve", "EGainCurve")
FRINGE_NPARAM_PER_POL = 4
#: Grid values (subband x antenna x time x channel x hand) evaluated together: a few complex arrays of this size
#: are the memory an evaluation needs, whatever the size of the block.
GRID_VALUES = 1 << 21


# ----------------------------------------------------------------------------------------------------------------
# Table reading
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class CalTableData:
    """In-memory copy of a CASA calibration table.

    Attributes
    ----------
    path : str
    kind : str
        VisCal table keyword ("Fringe Jones", "B Jones", "G Jones", "T Jones", "B TSYS", "EPowerCurve", ...).
    partype : str
        "Complex" (CPARAM) or "Float" (FPARAM).
    time, interval : float64 arrays (nrow,)
    field_id, spw_id, antenna1, antenna2, scan_number : int arrays (nrow,)
    param : array (nrow, nchan_t, npar)
        CPARAM (complex) or FPARAM (float64).
    flag : bool array (nrow, nchan_t, npar)
    snr : float array (nrow, nchan_t, npar)
    spw_chan_freq : float64 array (nspw, nchan_t)
        CHAN_FREQ of the table SPECTRAL_WINDOW subtable [Hz].
    antenna_names, field_names : str arrays
    nant : int
        Number of rows of the ANTENNA subtable.

    The rows are indexed on first use (:func:`_lookup`), so the arrays must not be modified after the table
    has been applied once.
    """

    path: str
    kind: str
    partype: str
    time: np.ndarray
    interval: np.ndarray
    field_id: np.ndarray
    spw_id: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    scan_number: np.ndarray
    param: np.ndarray
    flag: np.ndarray
    snr: np.ndarray
    spw_chan_freq: np.ndarray
    antenna_names: np.ndarray
    field_names: np.ndarray
    nant: int
    _row_index: dict = field(default_factory=dict, repr=False, compare=False)
    #: Sent once to each worker process and cached there (see :mod:`vlbipy.solvers.workers`).
    _worker_shared: ClassVar[bool] = True

    @property
    def nrow(self):
        """Number of solution rows."""
        return int(self.time.shape[0])

    @property
    def nchan_t(self):
        """Number of channels per solution row (1 for G/T/Fringe/Tsys)."""
        return int(self.param.shape[1])

    @property
    def npar(self):
        """Number of parameters per channel (npol, or 8 for Fringe Jones)."""
        return int(self.param.shape[2])

    def __getstate__(self):
        """Pickled without the row indices (rebuilt where the table is applied), which can outweigh the table."""
        return dict(vars(self), _row_index={})


def _row_major_column(values, nrow, name, path) -> np.ndarray:
    """Return a caltable array column as (nrow, nchan_t, npar), whatever shape ``getcol`` returned.

    FPARAM/CPARAM, FLAG and SNR are declared with a variable number of dimensions (see
    :func:`vlbipy.solvers.caltable._array_column`), so one table can hold (npar, nchan_t) cells
    for one of them and (npar,) cells for another — CASA's gain curves do. Each column is
    therefore normalised on its own ndim: adding the missing channel axis to all three because
    the *parameter* column lacked it turns an already-3-D column into a 4-D array whose first
    index is no longer the row, and every later ``column[rows, chan, par]`` then breaks.

    Parameters
    ----------
    values : array
        Output of ``tb.getcol(name)``, in Fortran order (npar[, nchan_t], nrow).
    nrow : int
        Number of rows of the table, checked against the transposed first axis.
    name : str
        Column name, for the error message.
    path : Path or str
        Table path, for the error message.

    Returns
    -------
    numpy.ndarray
        Shape (nrow, nchan_t, npar).
    """
    column = np.ascontiguousarray(np.asarray(values).T)
    if column.ndim == 1:                                     # one value per row: (nrow,)
        column = column[:, None, None]
    elif column.ndim == 2:                                   # no channel axis: (nrow, npar)
        column = column[:, None, :]
    if column.ndim != 3 or column.shape[0] != nrow:
        raise ValueError(f"{path}: column {name} has shape {np.asarray(values).shape}, which is not "
                         f"(npar, nchan_t, {nrow}), (npar, {nrow}) or ({nrow},)")
    return column


def _match_param_grid(column, param_shape, name, path) -> np.ndarray:
    """Expand a (nrow, nchan_t, npar) FLAG/SNR column onto the parameter column's grid.

    CASA flags one parameter at a time, but a table may store one value per channel-less row, or
    one per polarization rather than per parameter. A degenerate channel axis is stretched, and a
    shorter parameter axis is repeated over the parameters of each polarization — which is what
    "this polarization is flagged" means for a gain curve (npar = 2 x ncoef) or a fringe solution
    (npar = 2 x 4).

    Parameters
    ----------
    column : numpy.ndarray
        Normalised FLAG or SNR column, shape (nrow, nchan_t, npar).
    param_shape : tuple
        Shape of the parameter column, which defines the grid.
    name : str
        Column name, for the error message.
    path : Path or str
        Table path, for the error message.

    Returns
    -------
    numpy.ndarray
        Shape ``param_shape``.
    """
    if column.shape == tuple(param_shape):
        return column
    nrow, nchan, npar = param_shape
    if column.shape[1] != nchan:
        if column.shape[1] != 1:
            raise ValueError(f"{path}: column {name} has {column.shape[1]} channels, the parameter column "
                             f"{nchan}; they must match or the column must hold one value per row")
        column = np.broadcast_to(column, (nrow, nchan, column.shape[2]))
    if column.shape[2] != npar:
        if column.shape[2] < 1 or npar % column.shape[2]:
            raise ValueError(f"{path}: column {name} has {column.shape[2]} parameters, which does not divide "
                             f"the {npar} of the parameter column")
        column = np.repeat(column, npar // column.shape[2], axis=2)
    return np.ascontiguousarray(column)


def load_caltable(path) -> CalTableData:
    """Read a CASA calibration table (main columns + SPECTRAL_WINDOW/ANTENNA/FIELD subtables) into a CalTableData.

    CPARAM tables give complex `param`, FPARAM tables float64.  All spws must have the same number of channels.
    """
    from .msio import table_tool

    t_start = _time.perf_counter()
    path = Path(path).absolute()
    tb = table_tool()
    tb.open(str(path))
    keywords = tb.getkeywords()
    kind = str(keywords.get("VisCal", tb.info().get("subType", "")))
    if kind == "Accor Jones":
        kind = "G Jones"       # the accor table is a plain per-antenna gain (CPARAM (1, npol)), applied like one
    partype = str(keywords.get("ParType", "Complex" if "CPARAM" in tb.colnames() else "Float"))
    param_col = "CPARAM" if "CPARAM" in tb.colnames() else "FPARAM"
    nrow = tb.nrows()
    scalars = {name: np.asarray(tb.getcol(name)) for name in ("TIME", "INTERVAL", "FIELD_ID", "SPECTRAL_WINDOW_ID",
                                                               "ANTENNA1", "ANTENNA2", "SCAN_NUMBER")}
    if nrow == 0:
        raise ValueError(f"calibration table {path} has no rows")
    # The casatools-style table returns Fortran order (npar, nchan_t, nrow); transpose to (nrow, nchan_t, npar).
    param = _row_major_column(tb.getcol(param_col), nrow, param_col, path)
    flag = _row_major_column(tb.getcol("FLAG"), nrow, "FLAG", path)
    snr = _row_major_column(tb.getcol("SNR"), nrow, "SNR", path)
    tb.close()
    param = param.astype(np.complex128 if np.iscomplexobj(param) else np.float64)
    flag = _match_param_grid(flag, param.shape, "FLAG", path).astype(bool)
    snr = _match_param_grid(snr, param.shape, "SNR", path).astype(np.float64)

    tb.open(str(path / "SPECTRAL_WINDOW"))
    nspw = tb.nrows()
    num_chan = np.asarray(tb.getcol("NUM_CHAN"))
    if np.any(num_chan != num_chan[0]):
        tb.close()
        raise NotImplementedError(f"{path}: spws with different NUM_CHAN {num_chan.tolist()} are not supported")
    spw_chan_freq = np.array([np.asarray(tb.getcell("CHAN_FREQ", i), dtype=np.float64).reshape(-1) for i in range(nspw)])
    tb.close()
    tb.open(str(path / "ANTENNA"))
    antenna_names = np.asarray(tb.getcol("NAME"), dtype=str)
    nant = tb.nrows()
    tb.close()
    tb.open(str(path / "FIELD"))
    field_names = np.asarray(tb.getcol("NAME"), dtype=str)
    tb.close()
    out = CalTableData(path=str(path), kind=kind, partype=partype, time=scalars["TIME"].astype(np.float64),
                       interval=scalars["INTERVAL"].astype(np.float64), field_id=scalars["FIELD_ID"].astype(np.int64),
                       spw_id=scalars["SPECTRAL_WINDOW_ID"].astype(np.int64),
                       antenna1=scalars["ANTENNA1"].astype(np.int64), antenna2=scalars["ANTENNA2"].astype(np.int64),
                       scan_number=scalars["SCAN_NUMBER"].astype(np.int64), param=param, flag=flag, snr=snr,
                       spw_chan_freq=spw_chan_freq, antenna_names=antenna_names, field_names=field_names, nant=int(nant))
    log.debug("load_caltable: %s kind=%r partype=%s rows=%d nchan_t=%d npar=%d nspw=%d in %.3f s", path, kind, partype,
             nrow, out.nchan_t, out.npar, nspw, _time.perf_counter() - t_start)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Interpolation helpers
# ----------------------------------------------------------------------------------------------------------------
def _mapped_spw(spw, spwmap):
    """Return the caltable spw to use for data spw `spw` (identity when `spwmap` is None/empty)."""
    if spwmap is None or len(spwmap) == 0:
        return int(spw)
    if spw >= len(spwmap):
        raise ValueError(f"spwmap {list(spwmap)} has no entry for spw {spw}")
    return int(spwmap[spw])


def _lookup(table, field_id=None, flag_column=None):
    """The rows of ``table`` in a field selection sorted by (antenna, spw, time), and the keys that find them.

    A table is applied to every block of every antenna and subband, and masking its rows each time costs more
    than the whole correction (a phase self-calibration table has millions of rows). The rows are therefore
    sorted once; each (antenna, spw) group is a run of them, and the key of a row is its group times (number
    of distinct table times + 1) plus the rank of its time. One ``searchsorted`` of integer keys then finds
    the solutions of every antenna, subband and time at once, in exactly the order a search of each group's
    own times gives (:func:`_brackets`). Cached on the table per selection.

    Parameters
    ----------
    field_id : int or sequence of int, optional
        FIELD_ID selection on the rows (None = all fields).
    flag_column : int, optional
        Keep only the rows whose flag of that parameter is not set (gain curves skip flagged solutions).
    """
    fields = None if field_id is None else tuple(sorted({int(f) for f in np.asarray(field_id).reshape(-1)}))
    key = ("lookup", fields, flag_column)
    found = table._row_index.get(key)
    if found is not None:
        return found
    utimes = table._row_index.get("times")
    if utimes is None:
        utimes = table._row_index["times"] = np.unique(table.time)
    keep = (table.antenna1 >= 0) & (table.spw_id >= 0)
    if fields is not None:
        keep &= np.isin(table.field_id, fields)
    if flag_column is not None:
        keep &= ~table.flag[:, 0, flag_column]
    rows = np.flatnonzero(keep)
    rows = rows[np.lexsort((table.time[rows], table.spw_id[rows], table.antenna1[rows]))]
    nspw = int(max(table.spw_chan_freq.shape[0], table.spw_id.max(initial=-1) + 1))
    nant = int(max(table.nant, table.antenna1.max(initial=-1) + 1))
    group = table.antenna1[rows] * nspw + table.spw_id[rows]
    found = {"rows": rows, "time": table.time[rows], "times": utimes, "stride": utimes.size + 1,
             "keys": group * (utimes.size + 1) + np.searchsorted(utimes, table.time[rows]),
             "bounds": np.searchsorted(group, np.arange(nant * nspw + 1)), "nspw": nspw, "nant": nant}
    table._row_index[key] = found
    return found


def _brackets(look, antenna, spw_t, times, interp):
    """The solutions every point takes: :func:`_bracket` in the rows of its antenna and spw, for all points at once.

    Parameters
    ----------
    look : dict
        From :func:`_lookup`.
    antenna, spw_t, times : arrays (npoint,)
        Antenna, caltable spw and time [s] of every point.
    interp : {"nearest", "linear"}

    Returns
    -------
    (i0, i1, w, found)
        Positions in ``look["rows"]`` and weights (npoint,), such that the value is (1 - w) v[i0] + w v[i1];
        ``found`` is False where the table has no row for the antenna and spw (i0 and i1 are then some valid
        position).
    """
    antenna = np.asarray(antenna, dtype=np.int64).reshape(-1)
    spw = np.asarray(spw_t, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    valid = (antenna >= 0) & (antenna < look["nant"]) & (spw >= 0) & (spw < look["nspw"])
    group = np.where(valid, antenna * look["nspw"] + spw, 0)
    bounds = look["bounds"]
    count = np.where(valid, bounds[group + 1] - bounds[group], 0)
    nrow = look["rows"].size
    start = np.minimum(bounds[group], max(nrow - 1, 0))
    weight = np.zeros(times.shape)
    if not nrow:
        nowhere = np.zeros(times.shape, dtype=np.int64)
        return nowhere, nowhere, weight, count > 0
    index = np.searchsorted(look["keys"], group * look["stride"] + np.searchsorted(look["times"], times))
    several = count >= 2
    hi = np.where(several, np.minimum(np.maximum(index, start + 1), start + count - 1), start)
    lo = np.where(several, hi - 1, start)
    row_time = look["time"]
    if interp == "nearest":
        near = np.where(np.abs(times - row_time[lo]) <= np.abs(row_time[hi] - times), lo, hi)
        return near, near, weight, count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.clip((times - row_time[lo]) / (row_time[hi] - row_time[lo]), 0.0, 1.0)
    # A time at (or beyond) the upper row uses that row alone, so the lower row's flag cannot reach it.
    at_hi = (w >= 1.0) | ~several
    return np.where(at_hi, hi, lo), hi, np.where(at_hi, 0.0, w), count > 0


def _bracket(row_times, times, interp):
    """Return (i0, i1, w): indices into sorted `row_times` and weights such that value = (1-w) v[i0] + w v[i1].

    "nearest" gives i0 == i1 (nearest row) and w = 0; "linear" gives the bracketing rows, clamped to the nearest
    row (w = 0 or 1) beyond the ends.
    """
    n = row_times.shape[0]
    times = np.asarray(times, dtype=np.float64)
    if n == 1:
        zero = np.zeros(times.shape, dtype=np.int64)
        return zero, zero, np.zeros(times.shape, dtype=np.float64)
    hi = np.clip(np.searchsorted(row_times, times), 1, n - 1)
    lo = hi - 1
    if interp == "nearest":
        near = np.where(np.abs(times - row_times[lo]) <= np.abs(row_times[hi] - times), lo, hi)
        return near, near, np.zeros(times.shape, dtype=np.float64)
    w = np.clip((times - row_times[lo]) / (row_times[hi] - row_times[lo]), 0.0, 1.0)
    # A time at (or beyond) the upper row uses that row alone, so the lower row's flag cannot reach it.
    at_hi = w >= 1.0
    return np.where(at_hi, hi, lo), hi, np.where(at_hi, 0.0, w)


def _interp_amp_phase(v0, v1, w):
    """Interpolate complex gains between v0 and v1 with weight w (broadcast): amplitude and phase separately."""
    amp = (1.0 - w) * np.abs(v0) + w * np.abs(v1)
    dphi = np.angle(v1 * np.conj(v0))
    phase = np.angle(v0) + w * dphi
    return amp * np.exp(1j * phase)


def parse_interp(interp):
    """Split a CASA ``interp`` string into (time mode, frequency mode, keep channel flags).

    ``"nearest,linearflag"`` -> ``("nearest", "linear", True)``. Time modes reduce to "nearest" or "linear"
    (the ``perobs``/``PD`` variants are treated as their base mode); frequency modes to "nearest" or "linear"
    ("cubic" and "spline" fall back to "linear"). CASA's default is ``"linear,linear"``.
    """
    tokens = [token.strip().lower() for token in str(interp or "").split(",")]
    time_mode = "nearest" if tokens[0].startswith("nearest") else "linear"
    freq_token = tokens[1] if len(tokens) > 1 and tokens[1] else "linear"
    keep_flags = freq_token.endswith("flag")
    freq_mode = "nearest" if freq_token.startswith("nearest") else "linear"
    return time_mode, freq_mode, keep_flags


def _interp_freqs(values, flags, pair, table_freq, chan_freq, mode="linear", keep_flags=False):
    """Bring solutions from their table channels to the data channels, every solution on its own.

    Row k of ``values`` (complex) and ``flags`` (bool), both (K, nchan_t), goes from the table channels
    ``table_freq[pair[k]]`` (npair, nchan_t) to the data channels ``chan_freq[pair[k]]`` (npair, nchan) [Hz].
    Like CASA, flagged table channels are *interpolated across*: each data channel takes the nearest
    unflagged table channel ("nearest") or the amp/phase interpolation of the two unflagged channels around
    it ("linear", nearest beyond the ends), and is flagged only when the solution has no unflagged channel at
    all. With ``keep_flags`` (CASA's ``...flag`` modes) a data channel whose nearest table channel is flagged
    stays flagged. A solution without flags on the data's own channel grid is kept as it is.

    Returns
    -------
    (values, flags)
        Each (K, nchan).
    """
    table_freq = np.asarray(table_freq, dtype=np.float64)
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    nrow, nchan_t = values.shape
    nchan = chan_freq.shape[1]
    order = np.argsort(table_freq, axis=1)
    table_sorted = np.take_along_axis(table_freq, order, axis=1)
    # Per (table, data) channel pair: the table channels below every data channel, the nearest table channel.
    below_all = np.stack([np.searchsorted(t, c) for t, c in zip(table_sorted, chan_freq)])
    nearest_all = np.stack([_bracket(t, c, "nearest")[0] for t, c in zip(table_sorted, chan_freq)])
    same_grid = np.array([nchan_t == nchan and np.allclose(t, c, rtol=0, atol=1.0)
                          for t, c in zip(table_freq, chan_freq)])
    order = order[pair]
    vals = np.take_along_axis(values, order, axis=1)
    fl = np.take_along_axis(flags, order, axis=1)
    count = np.zeros((nrow, nchan_t + 1), dtype=np.int64)
    np.cumsum(~fl, axis=1, out=count[:, 1:])
    ngood = count[:, -1:]
    below = np.take_along_axis(count, below_all[pair], axis=1)       # unflagged table channels below each channel
    good = np.argsort(fl, axis=1, kind="stable")                      # unflagged channels first, in frequency order
    several = ngood >= 2
    rank = np.where(several, np.minimum(np.maximum(below, 1), ngood - 1), 0)
    lo = np.take_along_axis(good, np.where(several, rank - 1, 0), axis=1)
    hi = np.take_along_axis(good, rank, axis=1)
    freq, row_freq = chan_freq[pair], table_sorted[pair]
    f_lo, f_hi = np.take_along_axis(row_freq, lo, axis=1), np.take_along_axis(row_freq, hi, axis=1)
    w = None
    if mode == "nearest":
        i0 = i1 = np.where(np.abs(freq - f_lo) <= np.abs(f_hi - freq), lo, hi)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.clip((freq - f_lo) / (f_hi - f_lo), 0.0, 1.0)
        at_hi = (w >= 1.0) | ~several
        i0, i1, w = np.where(at_hi, hi, lo), hi, np.where(at_hi, 0.0, w)
    out = np.take_along_axis(vals, i0, axis=1)
    if w is not None:
        blend = (i1 != i0) & (w > 0)
        if blend.any():
            row, chan = np.nonzero(blend)
            out[row, chan] = _interp_amp_phase(out[row, chan], vals[row, i1[row, chan]], w[row, chan])
    out_flag = (np.take_along_axis(fl, nearest_all[pair], axis=1) if keep_flags
                else np.zeros((nrow, nchan), dtype=bool))
    none = ngood[:, 0] == 0
    out[none], out_flag[none] = 1.0, True
    as_is = same_grid[pair] & ~flags.any(axis=1)              # only possible on the same number of channels
    if as_is.any():
        out[as_is], out_flag[as_is] = values[as_is], False
    return out, out_flag


def _interpolated_solutions(table, row0, row1, w, pols):
    """Solutions of a G/T/B Jones or Tsys table interpolated in time: (values, flags), each (npoint, nchan_t, 2).

    Point k lies between the table rows ``row0[k]`` and ``row1[k]`` with weight ``w[k]`` (``row0 == row1`` or
    ``w == 0``: that row alone); ``pols`` are the parameters of the two hands. Gains: amplitude and phase
    interpolated separately, flagged where a solution used is flagged or zero. Tsys: interpolated itself (as
    CASA does), gain 1/sqrt(Tsys), flagged where a Tsys used is flagged or not positive.
    """
    blend = (row1 != row0) & (w > 0)
    p0 = table.param[row0][:, :, pols]
    f0 = table.flag[row0][:, :, pols]
    if table.kind == "B TSYS":
        tsys0 = np.real(p0).astype(np.float64)
        tsys1 = np.real(table.param[row1][:, :, pols]).astype(np.float64)
        f1 = table.flag[row1][:, :, pols]
        frac = w[:, None, None]
        value = (1.0 - frac) * tsys0 + frac * tsys1
        flags = f0 | ~(tsys0 > 0) | (blend[:, None, None] & (f1 | ~(tsys1 > 0))) | ~(value > 0)
        return (1.0 / np.sqrt(np.where(flags, 1.0, value))).astype(np.complex128), flags
    values = p0.astype(np.complex128)
    flags = f0 | (values == 0)
    if blend.any():
        b = np.flatnonzero(blend)
        other = table.param[row1[b]][:, :, pols].astype(np.complex128)
        values[b] = _interp_amp_phase(values[b], other, w[b][:, None, None])
        flags[b] |= table.flag[row1[b]][:, :, pols] | (other == 0)
    return values, flags


def _table_gains(table, antenna, times, spw_index, spw_t, chan_freq, *, field_id=None, interp="linear"):
    """Gains (1 where flagged) and flags of a G/T/B Jones or Tsys table at points.

    Point k is ``antenna[k]`` at ``times[k]`` in the data spw ``spw_index[k]``, whose caltable spw is
    ``spw_t[spw_index[k]]`` and whose channels are ``chan_freq[spw_index[k]]`` ((nspw, nchan) [Hz]).

    Returns
    -------
    (values, flags, index)
        A table without a frequency axis: values and flags (npoint, 2) and ``index`` None. A bandpass: values
        and flags (nsolution, 2, nchan) of every distinct solution the points take (a solution is the same at
        every time between the same rows: each is brought to the data channels once) and ``index`` (npoint,),
        the solution of every point.
    """
    time_mode, freq_mode, keep_chan_flags = parse_interp(interp)
    pols = [0, min(1, table.npar - 1)]
    look = _lookup(table, field_id)
    i0, i1, w, found = _brackets(look, antenna, spw_t[spw_index], times, time_mode)
    rows = look["rows"]
    if table.nchan_t == 1:
        if not rows.size:
            return np.ones((i0.size, 2), dtype=np.complex128), np.ones((i0.size, 2), dtype=bool), None
        values, flags = _interpolated_solutions(table, rows[i0], rows[i1], w, pols)
        flags = flags[:, 0] | ~found[:, None]
        return np.where(flags, 1.0, values[:, 0]), flags, None
    nchan = chan_freq.shape[1]
    if not rows.size:
        return np.ones((1, 2, nchan), dtype=np.complex128), np.ones((1, 2, nchan), dtype=bool), \
            np.zeros(i0.size, dtype=np.int64)
    key = ((spw_index * (rows.size + 1) + i0) * (rows.size + 1) + i1) * 2 + found
    if (w != 0).any():
        _, first, index = np.unique(np.stack([key, w.view(np.int64)], axis=1), axis=0, return_index=True,
                                    return_inverse=True)
    else:
        _, first, index = np.unique(key, return_index=True, return_inverse=True)
    values, flags = _interpolated_solutions(table, rows[i0[first]], rows[i1[first]], w[first], pols)
    count = first.size
    values, flags = _interp_freqs(values.transpose(0, 2, 1).reshape(2 * count, -1),
                                  flags.transpose(0, 2, 1).reshape(2 * count, -1),
                                  np.repeat(spw_index[first], 2), table.spw_chan_freq[spw_t], chan_freq,
                                  freq_mode, keep_chan_flags)
    flags = flags.reshape(count, 2, nchan) | ~found[first, None, None]
    return np.where(flags, 1.0, values.reshape(count, 2, nchan)), flags, index.reshape(-1)


def _fringe_grid(table, antenna, times, spw_index, spw_t, *, field_id=None, interp="linear"):
    """Fringe Jones parameters of both hands at points (as for :func:`_table_gains`), zero where flagged.

    Follows CASA's rate-aware interpolation (``CTRateAwareTimeInterp1``), which works on the parameters:

    * a time that uses a single solution ("nearest", or beyond the first/last solution) takes that solution's
      delay and dispersive term, and its phase advanced by the rate from the solution time;
    * a time between two solutions ("linear") takes delay and dispersive term interpolated linearly, and for the
      phase the argument of the weighted mean of the two unit phasors obtained by advancing each solution's
      phase to that time with its own rate (so the result does not depend on how many turns apart they are).

    The rate always multiplies the table's reference frequency.

    Returns
    -------
    (phase, delay, disp, flag, f_ref)
        Arrays (npoint, 2) such that the phase at frequency ``f`` is ``phase + 2 pi 1e-9 delay (f - f_ref) +
        disp k_disp(f)``, and the reference frequency of each data spw (nspw,) [Hz].
    """
    if table.npar != 2 * FRINGE_NPARAM_PER_POL:
        raise ValueError(f"{table.path}: Fringe Jones tables must have 8 parameters, got {table.npar}")
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    f_ref = table.spw_chan_freq[spw_t, table.spw_chan_freq.shape[1] // 2].astype(np.float64)
    look = _lookup(table, field_id)
    i0, i1, w, found = _brackets(look, antenna, spw_t[spw_index], times, interp)
    shape = i0.shape + (2,)
    phase, delay, disp = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    flag = np.ones(shape, dtype=bool)
    if not look["rows"].size:
        return phase, delay, disp, flag, f_ref
    row0, row1 = look["rows"][i0], look["rows"][i1]
    turn = (TWO_PI * f_ref)[spw_index]                   # rate -> phase change per second at the reference frequency
    elapsed = times - look["time"][i0]
    blend = np.flatnonzero((i1 != i0) & (w > 0))
    frac = w[blend]
    later_elapsed = times[blend] - look["time"][i1[blend]]
    for hand in range(2):
        block = slice(FRINGE_NPARAM_PER_POL * hand, FRINGE_NPARAM_PER_POL * (hand + 1))
        lo = np.real(table.param[row0, 0, block]).astype(np.float64)               # phase, delay, rate, disp
        p = lo[:, 0] + turn * lo[:, 2] * elapsed
        d, x = lo[:, 1].copy(), lo[:, 3].copy()
        rowflag = table.flag[:, 0, FRINGE_NPARAM_PER_POL * hand]
        f = rowflag[row0]
        if blend.size:
            first = lo[blend]
            second = np.real(table.param[row1[blend], 0, block]).astype(np.float64)
            later = second[:, 0] + turn[blend] * second[:, 2] * later_elapsed
            p[blend] = np.angle((1.0 - frac) * np.exp(1j * p[blend]) + frac * np.exp(1j * later))
            d[blend] = (1.0 - frac) * first[:, 1] + frac * second[:, 1]
            x[blend] = (1.0 - frac) * first[:, 3] + frac * second[:, 3]
            f[blend] |= rowflag[row1[blend]]
        f |= ~found
        phase[:, hand] = np.where(f, 0.0, p)
        delay[:, hand] = np.where(f, 0.0, d)
        disp[:, hand] = np.where(f, 0.0, x)
        flag[:, hand] = f
    return phase, delay, disp, flag, f_ref


def _gain_curve_gains(table, antenna, times, spw_index, spw_t, elevation, *, field_id=None, interp="linear"):
    """Gains (1 where flagged) and flags (npoint, 2) of a gain-curve table at points (as for :func:`_table_gains`).

    The power polynomial of each hand is evaluated in zenith angle [deg] (EPowerCurve/EGainCurve::calcPar of
    CASA) at the ``elevation`` of every point [rad], its coefficients interpolated in time between the
    unflagged solutions; EPowerCurve gives the square root of the power.
    """
    shape = (np.size(times), 2)
    gains, flags = np.ones(shape, dtype=np.complex128), np.ones(shape, dtype=bool)
    ncoef = table.npar // 2                                  # npar = 2 pols x ncoef polynomial coefficients
    if ncoef < 1:
        return gains, flags
    x = 90.0 - np.degrees(np.asarray(elevation, dtype=np.float64))
    for hand in range(2):
        pol = min(hand, table.npar - 1)
        # The flag of a polarization sits with the first coefficient of its block of npar.
        look = _lookup(table, field_id, flag_column=pol * ncoef)
        if not look["rows"].size:
            continue
        i0, i1, w, found = _brackets(look, antenna, spw_t[spw_index], times, interp)
        coefficients = slice(pol * ncoef, (pol + 1) * ncoef)
        frac = w[:, None]
        coeff = ((1.0 - frac) * table.param[look["rows"][i0], 0, coefficients]
                 + frac * table.param[look["rows"][i1], 0, coefficients])
        power = np.zeros(i0.shape, dtype=np.float64)
        for degree in range(ncoef - 1, -1, -1):
            power = power * x + np.real(coeff[:, degree])
        gain = np.sqrt(np.maximum(power, 0.0)) if table.kind == "EPowerCurve" else power
        bad = ~np.isfinite(gain) | (gain <= 0) | ~found
        gains[:, hand] = np.where(bad, 1.0, gain)
        flags[:, hand] = bad
    return gains, flags


def _grid_points(antenna_ids, times):
    """The (antenna, time) points of a dense grid, antenna-major: (antenna (npoint,), time (npoint,))."""
    antenna_ids = np.asarray(antenna_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    return np.repeat(antenna_ids, times.size), np.tile(times, antenna_ids.size)


def fringe_phase_terms(table, antenna_ids, times, spw, *, field_id=None, interp="linear", spwmap=None):
    """The Fringe Jones phase of every (antenna, time, hand) as parameters, on no channel grid.

    Returns ``(phase, delay, disp, flag, f_ref_hz)``: arrays (nant_sel, ntime, 2) such that the phase at
    frequency ``f`` is ``phase + 2 pi 1e-9 delay (f - f_ref_hz) + disp k_disp(f)`` (zero where flagged), as
    :func:`antenna_gains` evaluates it. Several fringe tables therefore add up before a single exponential.
    """
    spw_t = _mapped_spw(int(spw), spwmap)
    if spw_t >= table.spw_chan_freq.shape[0]:
        raise ValueError(f"{table.path}: caltable spw {spw_t} (data spw {spw}) does not exist")
    antenna, time = _grid_points(antenna_ids, times)
    phase, delay, disp, flag, f_ref = _fringe_grid(table, antenna, time, np.zeros(antenna.size, dtype=np.int64),
                                                   np.array([spw_t]), field_id=field_id,
                                                   interp=parse_interp(interp)[0])
    shape = (np.size(antenna_ids), np.size(times), 2)
    return (phase.reshape(shape), delay.reshape(shape), disp.reshape(shape), flag.reshape(shape),
            float(f_ref[0]))


def antenna_gains(table, antenna_ids, times, spw, chan_freq, *, field_id=None, interp="linear", spwmap=None,
                  elevation_grid=None):
    """Evaluate the antenna gains of a calibration table on a (antenna, time, channel) grid of one data spw.

    Parameters
    ----------
    table : CalTableData or path
    antenna_ids : int array (nant_sel,)
    times : float array (ntime,)  [s]
    spw : int
        Data spectral window id being corrected.
    chan_freq : float array (nchan,)
        Channel frequencies of that spw [Hz].
    field_id : int or sequence of int, optional
        FIELD_ID selection on the table rows (None = all fields).
    interp : {"linear", "nearest"}
    spwmap : sequence of int, optional
        Data spw -> caltable spw (empty/None = identity).
    elevation_grid : float array (nant_sel, ntime), optional
        Elevations [rad]; required by gain-curve tables.

    Returns
    -------
    gains : complex128 array (nant_sel, ntime, nchan, 2)
        Gains for the two polarisation hands (unit gains where flagged).
    gflag : bool array (nant_sel, ntime, nchan, 2)
    """
    if not isinstance(table, CalTableData):
        table = load_caltable(table)
    if table.kind in GAIN_CURVE_KINDS and elevation_grid is None:
        raise ValueError("gain-curve application requires elevations for every selected antenna and time")
    chan_freq = np.asarray(chan_freq, dtype=np.float64).reshape(-1)
    spw_t = _mapped_spw(int(spw), spwmap)
    if spw_t >= table.spw_chan_freq.shape[0]:
        raise ValueError(f"{table.path}: caltable spw {spw_t} (data spw {spw}) does not exist")
    _check_kind(table)
    shape = (np.size(antenna_ids), np.size(times), chan_freq.size, 2)
    if table.kind == "Fringe Jones":
        phase, delay, disp, flag, f_ref = fringe_phase_terms(table, antenna_ids, times, spw_t, field_id=field_id,
                                                             interp=interp)
        theta = phase[:, :, None, :] + delay[:, :, None, :] * (TWO_PI * (chan_freq - f_ref) * 1e-9)[:, None]
        if np.any(disp):
            theta = theta + disp[:, :, None, :] * k_disp(chan_freq, float(np.min(chan_freq)),
                                                         float(np.max(chan_freq)))[:, None]
        flags = np.broadcast_to(flag[:, :, None, :], shape)
        return np.where(flags, 1.0, np.exp(1j * theta)), flags.copy()
    antenna, time = _grid_points(antenna_ids, times)
    spw_index, spw_ts = np.zeros(antenna.size, dtype=np.int64), np.array([spw_t])
    if table.kind in GAIN_CURVE_KINDS:
        gains, flags = _gain_curve_gains(table, antenna, time, spw_index, spw_ts,
                                         np.asarray(elevation_grid, dtype=np.float64).reshape(-1),
                                         field_id=field_id, interp=parse_interp(interp)[0])
        index = None
    else:
        gains, flags, index = _table_gains(table, antenna, time, spw_index, spw_ts, chan_freq[None],
                                           field_id=field_id, interp=interp)
    if index is None:
        gains, flags = gains[:, None, :], flags[:, None, :]
    else:
        gains, flags = gains[index].transpose(0, 2, 1), flags[index].transpose(0, 2, 1)
    dense = shape[:2] + gains.shape[1:]
    return (np.broadcast_to(gains.reshape(dense), shape).copy(), np.broadcast_to(flags.reshape(dense), shape).copy())


def _check_kind(table):
    """Warn about a table kind that is applied as a plain gain without being one of those known to be."""
    if table.kind not in ("G Jones", "T Jones", "B Jones", "B TSYS", "Fringe Jones", *GAIN_CURVE_KINDS):
        log.warning("table kind %r of %s not explicitly supported, applying as a G-like gain", table.kind,
                    table.path)


# ----------------------------------------------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------------------------------------------
def _corr_hands(ncorr):
    """Return (hand of antenna1, hand of antenna2) per correlation for ncorr in (1, 2, 4): RR, RL, LR, LL order."""
    if ncorr == 4:
        return np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    if ncorr == 2:
        return np.array([0, 1]), np.array([0, 1])
    if ncorr == 1:
        return np.array([0]), np.array([0])
    raise ValueError(f"unsupported number of correlations {ncorr}")


def _normalise_entry(entry, cache):
    """Return (CalTableData, interp, spwmap, field_id, calwt) for one entry of the `tables` list of apply_tables."""
    source = entry.get("table", entry.get("path"))
    if isinstance(source, CalTableData):
        table = source
    else:
        key = str(Path(source).absolute())
        if key not in cache:
            cache[key] = load_caltable(key)
        table = cache[key]
    interp = str(entry.get("interp", "linear") or "linear")
    spwmap = list(entry.get("spwmap", []) or [])
    gainfield = entry.get("gainfield", None)
    field_id = None
    if isinstance(gainfield, str):
        if gainfield.strip().lower() == "nearest":
            field_id = NEAREST_FIELD
        elif gainfield.strip():
            field_id = [int(x) for x in gainfield.split(",")]
    elif gainfield is not None and len(gainfield) > 0:
        field_id = [int(x) for x in gainfield]
    return table, interp, spwmap, field_id, bool(entry.get("calwt", False))


def adapt_table(table: CalTableData, antenna_names) -> CalTableData:
    """``table`` with its antennas numbered as in the antenna list ``antenna_names`` of a measurement set.

    A table numbers the antennas of its own ANTENNA subtable, copied from the set it was solved on: one
    solved on another set (another epoch, a set with a different array) is matched by antenna name, and its
    rows for antennas the set does not have are dropped. Returned unchanged when the numbering agrees.
    """
    wanted = [str(name).strip().upper() for name in antenna_names]
    own = [str(name).strip().upper() for name in table.antenna_names]
    if own == wanted[:len(own)]:
        return table
    lookup = {name: index for index, name in enumerate(wanted)}
    mapping = np.array([lookup.get(name, -1) for name in own] + [-1], dtype=np.int64)   # [-1]: out of range

    def renumber(ids):
        ids = np.asarray(ids, dtype=np.int64)
        return np.where(ids >= 0, mapping[np.clip(ids, 0, len(own))], ids)

    antenna1, antenna2 = renumber(table.antenna1), renumber(table.antenna2)
    keep = antenna1 >= 0
    missing = sorted({own[i] for i in np.unique(np.asarray(table.antenna1)[~keep]) if 0 <= i < len(own)})
    if missing:
        log.info("%s: no antenna %s in the measurement set; their solutions are not applied", table.path,
                 ", ".join(missing))
    rows = {name: getattr(table, name)[keep] for name in ("time", "interval", "field_id", "spw_id", "scan_number",
                                                           "param", "flag", "snr")}
    return replace(table, antenna1=antenna1[keep], antenna2=antenna2[keep], antenna_names=np.asarray(antenna_names),
                   nant=len(antenna_names), _row_index={}, **rows)


def nearest_table_field(table, field_dirs, target_field):
    """Field id, among those ``table`` has solutions for, closest on the sky to ``target_field`` (None if none).

    ``field_dirs`` is the MS FIELD direction array (nfield, 2) [rad]. This is CASA's ``gainfield='nearest'``.
    """
    candidates = np.unique(table.field_id[(table.field_id >= 0) & (table.field_id < len(field_dirs))])
    if not candidates.size:
        return None
    ra0, dec0 = field_dirs[int(target_field)]
    ra, dec = field_dirs[candidates, 0], field_dirs[candidates, 1]
    cos_sep = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    return int(candidates[int(np.argmax(cos_sep))])


def normalise_entries(tables, *, phase_only=False, calwt=False, cache=None, field_dirs=None, target_field=None):
    """Resolve the ``tables`` list of :func:`apply_tables` into ``(CalTableData, interp, spwmap, field_id, calwt)``.

    Tables given by path are read once (``cache``, a dict, shares them between calls); with ``phase_only`` the
    amplitude-only kinds (Tsys, gain curves) are dropped. ``calwt`` is the default for entries that carry no
    "calwt" key of their own. A ``gainfield`` of ``"nearest"`` becomes the table field closest to
    ``target_field`` when ``field_dirs`` (MS FIELD directions, (nfield, 2) [rad]) is given, and all fields
    otherwise.
    """
    cache = {} if cache is None else cache
    entries = []
    for entry in tables:
        table, interp, spwmap, field_id, entry_calwt = _normalise_entry(entry, cache)
        if phase_only and table.kind in AMPLITUDE_ONLY_KINDS:
            log.debug("apply_tables: phase_only, skipping %s (%s)", table.path, table.kind)
            continue
        if field_id is NEAREST_FIELD:
            field_id = None
            if field_dirs is not None and target_field is not None:
                nearest = nearest_table_field(table, np.asarray(field_dirs, dtype=np.float64), target_field)
                field_id = None if nearest is None else [nearest]
        entries.append((table, interp, spwmap, field_id, entry_calwt or ("calwt" not in entry and calwt)))
    return entries


def _known_antennas(table, antenna, spw_t):
    """Whether ``table`` has any solution, flagged or not, for every point's antenna in its caltable spw."""
    known = table._row_index.get("known")
    if known is None:
        known = np.zeros((int(max(table.nant, table.antenna1.max(initial=-1) + 1)),
                          int(max(table.spw_chan_freq.shape[0], table.spw_id.max(initial=-1) + 1))), dtype=bool)
        listed = (table.antenna1 >= 0) & (table.spw_id >= 0)
        known[table.antenna1[listed], table.spw_id[listed]] = True
        table._row_index["known"] = known
    antenna = np.asarray(antenna, dtype=np.int64)
    spw = np.asarray(spw_t, dtype=np.int64)
    inside = (antenna >= 0) & (antenna < known.shape[0]) & (spw >= 0) & (spw < known.shape[1])
    return inside & known[np.where(inside, antenna, 0), np.where(inside, spw, 0)]


def _point_elevations(antenna, times, antenna_xyz, field_dir):
    """Elevation [rad] of the source ``field_dir`` (ra, dec) [rad] at every point (``antenna[k]``, ``times[k]``)."""
    if antenna_xyz is None or field_dir is None:
        raise ValueError("gain-curve application requires antenna positions and field direction")
    from .parang import elevation
    utimes, at_time = np.unique(times, return_inverse=True)
    return elevation(antenna_xyz, float(field_dir[0]), float(field_dir[1]), utimes)[antenna, at_time.reshape(-1)]


def _chain_factors(entries, antenna, times, spw_index, spws, chan_freq, *, antenna_xyz=None, field_dir=None,
                   elevation=None):
    """The factors of a table chain at points: point k is ``antenna[k]`` at ``times[k]`` in the data spw
    ``spws[spw_index[k]]``, whose channels are ``chan_freq[spw_index[k]]`` ((nspw, nchan) [Hz]). Gain curves
    need the source ``elevation`` at every point [rad] (computed from ``antenna_xyz`` and ``field_dir`` when
    None).

    The chain is evaluated in factors, each on the smallest grid it needs: tables without a frequency axis
    (Tsys, gain curves, G/T Jones) multiplied per point and hand, the fringe tables as summed phase parameters
    (to be exponentiated once), and channel-dependent tables (bandpasses) per distinct solution. :func:`_assemble`
    (or :func:`vlbipy.solvers.kernels.assemble`) multiplies them out on every channel.

    Returns
    -------
    dict
        ``flat``/``flat_flag`` (npoint, 2); ``fringe`` (phase, delay, disp, flags) each (npoint, 2), or None;
        ``dispersive`` (whether a fringe table has a dispersive term); ``spectral`` (values, flags, index): the
        bandpass product (nvalue, 2, nchan) and the row of every point in it, or None; ``offset`` (nspw, nchan):
        2 pi 1e-9 (f - f0) of every channel from the first one; ``kdisp`` (nspw, nchan) or None; ``wscale`` the
        product of |g|^2 of the entries with ``calwt`` whose kind is in :data:`WEIGHT_CAL_KINDS` and ``wflag``
        their flags where the table knows the antenna ((npoint, 2), or (npoint, 2, nchan) when one of them has
        channels), or None.
    """
    antenna = np.asarray(antenna, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    spw_index = np.asarray(spw_index, dtype=np.int64).reshape(-1)
    spws = np.asarray(spws, dtype=np.int64).reshape(-1)
    chan_freq = np.asarray(chan_freq, dtype=np.float64).reshape(spws.size, -1)
    npoint = times.size
    flat = np.ones((npoint, 2), dtype=np.complex128)
    flat_flag = np.zeros((npoint, 2), dtype=bool)
    spectral = []
    phase = delay = disp = fringe_flag = None
    dispersive = False
    f0 = chan_freq[:, 0]
    wscale, wflag = None, None
    for table, interp, spwmap, field_id, entry_calwt in entries:
        spw_t = np.array([_mapped_spw(int(spw), spwmap) for spw in spws], dtype=np.int64)
        missing = spw_t >= table.spw_chan_freq.shape[0]
        if missing.any():
            raise ValueError(f"{table.path}: caltable spw {int(spw_t[missing][0])} (data spw "
                             f"{int(spws[missing][0])}) does not exist")
        if table.kind in GAIN_CURVE_KINDS and elevation is None:
            elevation = _point_elevations(antenna, times, antenna_xyz, field_dir)
        if table.kind == "Fringe Jones":
            p, d, x, f, f_ref = _fringe_grid(table, antenna, times, spw_index, spw_t, field_id=field_id,
                                             interp=parse_interp(interp)[0])
            # Referenced to the first channel, so that tables with different reference frequencies add up.
            p = p + np.where(f, 0.0, TWO_PI * 1e-9 * d * (f0 - f_ref)[spw_index][:, None])
            if phase is None:
                phase, delay, disp, fringe_flag = p, d, x, f
            else:
                phase, delay, disp, fringe_flag = phase + p, delay + d, disp + x, fringe_flag | f
            # A property of the tables, not of the points: a point never depends on the others.
            dispersive = dispersive or bool(np.any(table.param[:, 0, 3::FRINGE_NPARAM_PER_POL]))
            continue
        _check_kind(table)
        if table.kind in GAIN_CURVE_KINDS:
            values, flags = _gain_curve_gains(table, antenna, times, spw_index, spw_t, elevation,
                                              field_id=field_id, interp=parse_interp(interp)[0])
            index = None
        else:
            values, flags, index = _table_gains(table, antenna, times, spw_index, spw_t, chan_freq,
                                                field_id=field_id, interp=interp)
        if index is None:
            flat *= values
            flat_flag |= flags
        else:
            spectral.append((values, flags, index))
        if entry_calwt and table.kind in WEIGHT_CAL_KINDS:
            power = np.abs(values) ** 2 if index is None else np.abs(values[index]) ** 2
            # An antenna the table does not know at all leaves the weight alone (its data are flagged, but CASA
            # scales the weight by 1); only a solution that exists and is flagged zeroes it.
            known = _known_antennas(table, antenna, spw_t[spw_index])
            flagged = flags & known[:, None] if index is None else flags[index] & known[:, None, None]
            if wscale is not None and wscale.ndim < power.ndim:           # the first such table with channels
                wscale, wflag = wscale[:, :, None], wflag[:, :, None]
            elif wscale is not None and power.ndim < wscale.ndim:
                power, flagged = power[:, :, None], flagged[:, :, None]
            wscale = power if wscale is None else wscale * power
            wflag = flagged if wflag is None else wflag | flagged
    if len(spectral) > 1:
        # Several bandpass-like tables: their product at every point, in the order of the chain.
        values = spectral[0][0][spectral[0][2]]
        flags = spectral[0][1][spectral[0][2]]
        for more, more_flags, index in spectral[1:]:
            values *= more[index]
            flags = flags | more_flags[index]
        spectral = [(values, flags, np.arange(npoint))]
    kdisp = None
    if dispersive:
        kdisp = np.stack([k_disp(freq, float(freq.min()), float(freq.max())) for freq in chan_freq])
    return {"flat": flat, "flat_flag": flat_flag, "spw_index": spw_index,
            "fringe": None if phase is None else (phase, delay, disp, fringe_flag), "dispersive": dispersive,
            "offset": TWO_PI * 1e-9 * (chan_freq - f0[:, None]), "kdisp": kdisp,
            "spectral": spectral[0] if spectral else None, "wscale": wscale, "wflag": wflag}


def _assemble(factors, nchan):
    """The gains and flags (npoint, 2, nchan) of every point, channel and hand from :func:`_chain_factors`."""
    flat, spw_index = factors["flat"], factors["spw_index"]
    shape = flat.shape + (nchan,)
    gflag = np.broadcast_to(factors["flat_flag"][:, :, None], shape).copy()
    if factors["fringe"] is not None:
        phase, delay, disp, fringe_flag = factors["fringe"]
        theta = delay[:, :, None] * factors["offset"][spw_index][:, None, :]
        theta += phase[:, :, None]
        if factors["kdisp"] is not None:
            theta += disp[:, :, None] * factors["kdisp"][spw_index][:, None, :]
        gain = np.exp(1j * theta)
        gain *= flat[:, :, None]
        gflag |= fringe_flag[:, :, None]
    else:
        gain = np.broadcast_to(flat[:, :, None], shape).copy()
    if factors["spectral"] is not None:
        values, flags, index = factors["spectral"]
        gain *= values[index]
        gflag |= flags[index]
    return gain, gflag


def _chain_gains(entries, antenna, times, spw_index, spws, chan_freq, *, antenna_xyz=None, field_dir=None):
    """:func:`combined_gains` at points (see :func:`_chain_factors`): (gain, gflag, wscale, wflag), the first two
    (npoint, nchan, 2), the last two (npoint, 1 or nchan, 2) or None."""
    factors = _chain_factors(entries, antenna, times, spw_index, spws, chan_freq, antenna_xyz=antenna_xyz,
                             field_dir=field_dir)
    gain, gflag = _assemble(factors, np.shape(chan_freq)[-1])
    wscale, wflag = factors["wscale"], factors["wflag"]
    if wscale is not None:
        wscale = wscale.transpose(0, 2, 1) if wscale.ndim == 3 else wscale[:, None, :]
        wflag = wflag.transpose(0, 2, 1) if wflag.ndim == 3 else wflag[:, None, :]
    return gain.transpose(0, 2, 1), gflag.transpose(0, 2, 1), wscale, wflag


def combined_gains(entries, antenna_ids, times, spw, chan_freq, *, antenna_xyz=None, field_dir=None):
    """Multiply the antenna gains of a whole chain of tables on one (antenna, time, channel, hand) grid.

    Applying N tables to a block is then one gather and one division per visibility instead of N: the grid
    (a few antennas x the integrations of a chunk x the channels) is tiny next to the visibilities.

    Parameters
    ----------
    entries : list of tuple
        Output of :func:`normalise_entries`.
    antenna_ids, times : arrays
        Antenna ids and (sorted, unique) times [s] spanning the block.
    spw : int
        Data spectral window.
    chan_freq : float array (nchan,)
        Frequencies of the block's channels [Hz].
    antenna_xyz, field_dir : optional
        Antenna positions (nant_ms, 3) [m] and source (ra, dec) [rad]; required by gain-curve tables.

    Returns
    -------
    (gain, gflag, wscale, wflag)
        ``gain`` complex128 and ``gflag`` bool, shape (nant_sel, ntime, nchan, 2); ``wscale`` float64 holding the
        product of |g|^2 over the entries with ``calwt`` whose kind is in :data:`WEIGHT_CAL_KINDS`, and ``wflag``
        the flags of those same entries (both None when there is none), broadcastable to that shape: their
        channel axis has length 1 when no channel-dependent table calibrates the weights.
    """
    antenna, time = _grid_points(antenna_ids, times)
    dense = (np.size(antenna_ids), np.size(times))
    values = _chain_gains(entries, antenna, time, np.zeros(antenna.size, dtype=np.int64), [spw],
                          np.asarray(chan_freq, dtype=np.float64).reshape(1, -1), antenna_xyz=antenna_xyz,
                          field_dir=field_dir)
    return tuple(None if value is None else value.reshape(dense + value.shape[1:]) for value in values)


def calibrates_weights(entries):
    """True when applying ``entries`` rebuilds the weights from SIGMA, as CASA does.

    That is the case as soon as one table other than a fringe table has ``calwt``; a chain of fringe tables
    alone (or with ``calwt`` off everywhere) leaves the weight columns untouched.
    """
    return any(entry_calwt and table.kind != "Fringe Jones" for table, _, _, _, entry_calwt in entries)


def apply_tables(vis, flag, weight, antenna1, antenna2, time, spw, chan_freq, tables, *, phase_only=False,
                 calwt=False, antenna_xyz=None, field_dir=None, copy=True):
    """Apply a chain of calibration tables to a block of MS rows.

    Parameters
    ----------
    vis : complex array (nrow, nchan, ncorr)
    flag : bool array (nrow, nchan, ncorr)
    weight : float array (nrow, nchan, ncorr) or (nrow, ncorr) or None
    antenna1, antenna2, time, spw : arrays (nrow,)
    chan_freq : float array (nspw, nchan)
        Frequencies of the block's channels for ALL spws of the MS [Hz]; rows are grouped per unique `spw`.
    tables : list of dict
        Keys: "path" (str or CalTableData; alias "table"), "interp" ("linear"/"nearest", first token before a
        comma), "spwmap" (list, empty = identity), "gainfield" (empty/None = all fields; list of field ids or a
        comma-separated string; "nearest" = all fields here, the nearest one via :func:`normalise_entries`),
        "calwt" (bool). A list already resolved with :func:`normalise_entries` is accepted as is.
    phase_only : bool
        Skip amplitude-only tables (B TSYS, gain curves): used before fringe fitting where amplitudes are irrelevant.
    calwt : bool
        Default weight calibration for entries without their own "calwt" key... entries override it.
    antenna_xyz, field_dir : optional
        Antenna positions and source direction, needed only by gain-curve tables.
    copy : bool
        False corrects ``vis``/``flag``/``weight`` in place where their dtypes allow (the caller owns them).

    Returns
    -------
    (vis_corr, flag_corr, weight_corr)
        Corrected arrays (new ones unless ``copy=False``); weight_corr has the shape of `weight` broadcast to
        vis (None stays None).
    """
    t_start = _time.perf_counter()
    vis = np.array(vis, copy=copy)
    flag = np.array(flag, dtype=bool, copy=copy)
    nrow, nchan, ncorr = vis.shape
    weight = _weight_cube(weight, vis.shape, copy)
    antenna1 = np.asarray(antenna1, dtype=np.int64)
    antenna2 = np.asarray(antenna2, dtype=np.int64)
    time = np.asarray(time, dtype=np.float64)
    spw = np.asarray(spw, dtype=np.int64)
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    entries = tables if tables and isinstance(tables[0], tuple) else normalise_entries(tables, phase_only=phase_only,
                                                                                         calwt=calwt)
    if entries:
        # The gains at every (spw, antenna, time) the rows use, then one division for all rows.
        points, point1, point2 = row_points(antenna1, antenna2, time, spw)
        grid = point_grid(entries, points["groups"][0], points["antenna"], points["time"], chan_freq,
                          antenna_xyz=antenna_xyz, field_dir=field_dir, vis_dtype=vis.dtype,
                          weight_dtype=None if weight is None else weight.dtype)
        apply_point_grid(vis, flag, weight, grid, point1, point2)
    log.debug("apply_tables: %d tables applied to %d rows x %d chan x %d corr (%d spws) in %.3f s", len(entries), nrow,
              nchan, ncorr, np.unique(spw).size, _time.perf_counter() - t_start)
    return vis, flag, weight


def _weight_cube(weight, shape, copy):
    """Weights as a writable (nrow, nchan, ncorr) float array (a per-row (nrow, ncorr) array is broadcast)."""
    if weight is None:
        return None
    weight = np.asarray(weight)
    wtype = np.float32 if weight.dtype.kind == "f" and weight.itemsize <= 4 else np.float64
    if weight.ndim == 2:
        return np.array(np.broadcast_to(weight[:, None, :], shape), dtype=wtype)
    return np.array(weight, dtype=wtype, copy=copy)


def row_points(antenna1, antenna2, time, *groups):
    """The points a set of rows needs gains at, and the two points of every row.

    A point is one antenna at one time, apart for every value of each array of ``groups`` (int, (nrow,): the
    field, the spw...). Returns ``(points, point1, point2)``: ``points`` a dict of the arrays (npoint,)
    ``"groups"`` (a list, one array per group), ``"antenna"`` and ``"time"``, sorted by (groups..., antenna,
    time), and the index of the point of every row's ANTENNA1 and ANTENNA2 (nrow,).
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    nrow = time.size
    antenna = np.concatenate([np.asarray(antenna1, dtype=np.int64).reshape(-1),
                              np.asarray(antenna2, dtype=np.int64).reshape(-1)])
    groups = [np.tile(np.asarray(group, dtype=np.int64).reshape(-1), 2) for group in groups]
    utimes, at_time = np.unique(time, return_inverse=True)
    # One integer per (groups..., antenna, time): every value shifted to start at 0, times as their rank.
    key = np.zeros(2 * nrow, dtype=np.int64)
    for values in groups + [antenna]:
        low = int(values.min()) if values.size else 0
        key = key * ((int(values.max()) - low + 1) if values.size else 1) + (values - low)
    key = key * max(utimes.size, 1) + np.tile(at_time.reshape(-1), 2)
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    points = {"groups": [values[first] for values in groups], "antenna": antenna[first],
              "time": utimes[np.tile(at_time.reshape(-1), 2)[first]]}
    return points, inverse[:nrow], inverse[nrow:]


def point_grid(entries, spw, antenna, time, chan_freq, *, antenna_xyz=None, field_dir=None,
               vis_dtype=np.complex64, weight_dtype=np.float32):
    """The combined gains of a table chain at points, ready to apply (:func:`apply_point_grid`).

    Parameters
    ----------
    entries : list of tuple
        Output of :func:`normalise_entries`.
    spw, antenna, time : arrays (npoint,)
        Data spw, antenna and time [s] of every point (:func:`row_points`).
    chan_freq : float array (nspw_ms, nchan)
        Channel frequencies of every spw of the set [Hz].
    vis_dtype, weight_dtype : numpy dtypes
        Types of the visibilities and weights the gains will be applied to.

    Returns
    -------
    dict
        ``gain`` (npoint, nchan, 2) with 1 where flagged, ``gflag``, and ``wscale`` (npoint, 1 or nchan, 2), 0
        where a weight-calibrating solution is flagged, or None when no entry calibrates the weights.
    """
    spws, spw_index = np.unique(np.asarray(spw, dtype=np.int64).reshape(-1), return_inverse=True)
    freqs = np.asarray(chan_freq, dtype=np.float64)[spws]
    nchan = freqs.shape[1]
    grid = empty_grid(spw_index.size, nchan, vis_dtype, weight_dtype, weight_channels(entries, nchan))
    fill_grid(grid, slice(None), entries, antenna, time, spw_index.reshape(-1), spws, freqs, antenna_xyz=antenna_xyz,
              field_dir=field_dir)
    return grid


def weight_channels(entries, nchan):
    """Channels of the weight scale of a chain: None when no entry calibrates the weights, else 1, or ``nchan``
    when one that does depends on frequency."""
    weighting = [table for table, _, _, _, calwt in entries if calwt and table.kind in WEIGHT_CAL_KINDS]
    if not weighting:
        return None
    return nchan if any(t.kind not in GAIN_CURVE_KINDS and t.nchan_t > 1 for t in weighting) else 1


def empty_grid(npoint, nchan, vis_dtype, weight_dtype, weights):
    """Arrays of a point grid of ``npoint`` points, with a ``wscale`` of ``weights`` channels (None: without)."""
    gain_type = np.complex64 if np.dtype(vis_dtype) == np.complex64 else np.complex128
    return {"gain": np.empty((npoint, nchan, 2), dtype=gain_type), "gflag": np.empty((npoint, nchan, 2), dtype=bool),
            "wscale": None if weights is None else np.empty((npoint, weights, 2), dtype=weight_dtype or np.float64)}


def fill_grid(grid, points, entries, antenna, time, spw_index, spws, chan_freq, *, antenna_xyz, field_dir):
    """Evaluate a chain at points into the slice ``points`` of the arrays of ``grid`` (see :func:`point_grid`).

    The points are evaluated in blocks of at most :data:`GRID_VALUES` values, which bounds the memory of the
    intermediate arrays without changing any value (every point is evaluated on its own). The factors of the
    blocks (:func:`_chain_factors`, numpy) are evaluated in parallel threads
    (:func:`vlbipy.solvers.workers.thread_map`); with numba, each block is then multiplied out by
    :func:`vlbipy.solvers.kernels.assemble` on all the cores of the job, otherwise by numpy in those threads.
    A grid with a ``wscale`` gets 1 at points whose chain does not calibrate the weights.
    """
    from . import kernels
    from .workers import thread_map
    antenna = np.asarray(antenna, dtype=np.int64).reshape(-1)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    gain, gflag = grid["gain"][points], grid["gflag"][points]
    wscale = None if grid["wscale"] is None else grid["wscale"][points]
    nchan = chan_freq.shape[1]
    step = max(GRID_VALUES // max(2 * nchan, 1), 1)
    blocks = [slice(k, min(k + step, time.size)) for k in range(0, time.size, step)]
    # The source elevations of gain curves once for all points, before the threads (astropy underneath).
    elevation = (_point_elevations(antenna, time, antenna_xyz, field_dir)
                 if any(table.kind in GAIN_CURVE_KINDS for table, *_ in entries) else None)

    def factors_of(block):
        return _chain_factors(entries, antenna[block], time[block], spw_index[block], spws, chan_freq,
                              antenna_xyz=antenna_xyz, field_dir=field_dir,
                              elevation=None if elevation is None else elevation[block])

    def scale_weights(block, factors):
        if wscale is None:
            return
        scale = factors["wscale"]
        if scale is None:
            wscale[block] = 1.0
        else:
            # A flagged solution of a weight-calibrating table scales the weight to 0 (as CASA writes it).
            scale = np.where(factors["wflag"], 0.0, scale)
            wscale[block] = scale[:, None, :] if scale.ndim == 2 else scale.transpose(0, 2, 1)

    def assemble(block, factors):
        g, f = _assemble(factors, nchan)
        # A solution that is zero or not finite without being flagged (a Tsys of 0, an interpolation
        # between a valid and an invalid value) cannot calibrate anything: dividing by it would leave
        # NaN or infinite visibilities that are not flagged. Treat it as the flagged solution it is.
        f |= ~np.isfinite(g) | (g == 0)
        np.copyto(g, 1.0, where=f)
        gain[block] = g.transpose(0, 2, 1)
        gflag[block] = f.transpose(0, 2, 1)

    def evaluate(block):
        factors = factors_of(block)
        assemble(block, factors)
        scale_weights(block, factors)

    if not kernels.available():
        thread_map(evaluate, blocks)
        return
    for block, factors in zip(blocks, thread_map(factors_of, blocks)):
        if not kernels.assemble(gain[block], gflag[block], factors["flat"], factors["flat_flag"],
                                factors["spw_index"], fringe=factors["fringe"], offset=factors["offset"],
                                kdisp=factors["kdisp"], spectral=factors["spectral"]):
            assemble(block, factors)
        scale_weights(block, factors)


def gain_grid(entries, antenna_ids, times, spw, chan_freq, *, antenna_xyz=None, field_dir=None,
              vis_dtype=np.complex64, weight_dtype=np.float32):
    """The combined gains of a table chain on the (antenna, time, channel, hand) grid of a block, ready to apply.

    The grid can serve any set of rows of one spw whose antennas and times are on it: computing it once for
    the union of many (scan, spw) runs and applying it run by run (:func:`apply_gain_grid`) gives exactly the
    result of calibrating each run on its own, with the table lookups done once instead of once per run.

    Parameters
    ----------
    entries : list of tuple
        Output of :func:`normalise_entries`.
    antenna_ids, times : arrays
        Antennas and times to cover (duplicates allowed).
    spw : int
    chan_freq : float array (nchan,)
    vis_dtype, weight_dtype : numpy dtypes
        Types of the visibilities and weights the grid will be applied to.

    Returns
    -------
    dict
        ``antennas`` and ``times`` (sorted unique), ``gain`` (nant, ntime, nchan, 2) with 1 where flagged,
        ``gflag``, and ``wscale`` (0 where a weight-calibrating solution is flagged; None when no entry
        calibrates the weights; broadcastable to the shape of ``gain``, see :func:`combined_gains`).
    """
    uants = np.unique(np.asarray(antenna_ids, dtype=np.int64))
    utimes = np.unique(np.asarray(times, dtype=np.float64))
    antenna, time = _grid_points(uants, utimes)
    chan_freq = np.asarray(chan_freq, dtype=np.float64).reshape(1, -1)
    grid = empty_grid(antenna.size, chan_freq.shape[1], vis_dtype, weight_dtype,
                      weight_channels(entries, chan_freq.shape[1]))
    fill_grid(grid, slice(None), entries, antenna, time, np.zeros(antenna.size, dtype=np.int64), np.array([int(spw)]),
              chan_freq, antenna_xyz=antenna_xyz, field_dir=field_dir)
    dense = (uants.size, utimes.size)
    return {"antennas": uants, "times": utimes, "gain": grid["gain"].reshape(dense + grid["gain"].shape[1:]),
            "gflag": grid["gflag"].reshape(dense + grid["gflag"].shape[1:]),
            "wscale": None if grid["wscale"] is None else grid["wscale"].reshape(dense + grid["wscale"].shape[1:])}


def _grid_index(grid, antenna1, antenna2, time):
    """Points of the rows' antennas and time on a grid of :func:`gain_grid` (ValueError when a row is not on it)."""
    lookup = np.full(int(grid["antennas"].max()) + 1, -1, dtype=np.int64)
    lookup[grid["antennas"]] = np.arange(grid["antennas"].size)
    antenna1, antenna2 = np.asarray(antenna1, dtype=np.int64), np.asarray(antenna2, dtype=np.int64)
    if (antenna1 >= lookup.size).any() or (antenna2 >= lookup.size).any():
        raise ValueError("apply_gain_grid: rows outside the antennas/times of the gain grid")
    a1_idx, a2_idx = lookup[antenna1], lookup[antenna2]
    t_idx = np.searchsorted(grid["times"], np.asarray(time, dtype=np.float64))
    if (a1_idx < 0).any() or (a2_idx < 0).any() or (t_idx >= grid["times"].size).any() or \
            (grid["times"][np.minimum(t_idx, grid["times"].size - 1)] != time).any():
        raise ValueError("apply_gain_grid: rows outside the antennas/times of the gain grid")
    ntime = grid["times"].size
    return a1_idx * ntime + t_idx, a2_idx * ntime + t_idx


def _as_points(grid):
    """The arrays of a :func:`gain_grid` grid with one point per (antenna, time), as :func:`point_grid` has them."""
    return {name: None if grid[name] is None else grid[name].reshape((-1,) + grid[name].shape[2:])
            for name in ("gain", "gflag", "wscale")}


def _per_corr(values, hands):
    """``values`` (nrow, nchan, 2 hands) for the correlations of ``hands`` (as is when they are the two hands)."""
    return values if hands.size == 2 and hands[0] == 0 and hands[1] == 1 else values[:, :, hands]


def apply_point_flags(flag, grid, point1, point2, chans=None):
    """Flag in place, as :func:`apply_point_grid` does, the data whose solution is flagged or missing (FLAG only)."""
    hand1, hand2 = _corr_hands(flag.shape[2])
    gflag = grid["gflag"]
    f1, f2 = gflag[point1], gflag[point2]
    if chans is not None:
        f1, f2 = f1[:, chans], f2[:, chans]
    flag |= _per_corr(f1, hand1) | _per_corr(f2, hand2)


def apply_point_grid(vis, flag, weight, grid, point1, point2, chans=None, *, rotation=None, rotate=None,
                     invalid=False):
    """Correct rows in place with the gains of a point grid (:func:`point_grid`).

    Row k is corrected with the gains of the points ``point1[k]`` (its ANTENNA1) and ``point2[k]`` (its
    ANTENNA2), on the channels ``chans`` of the grid when the rows were read with a channel selection.
    ``vis``/``flag`` (nrow, nchan, ncorr) and ``weight`` (same shape, or None) are modified in place.

    With ``invalid``, values that are not finite are first zeroed and flagged; ``rotation`` (nrow, ncorr)
    then multiplies the rows ``rotate`` (the feed rotation) before the division. The work is done by
    :func:`vlbipy.solvers.kernels.correct` in one parallel pass when numba is installed.
    """
    from . import kernels
    hand1, hand2 = _corr_hands(vis.shape[2])
    if kernels.available() and kernels.correct(vis, flag, weight, grid["gain"], grid["gflag"], grid["wscale"],
                                               point1, point2, hand1, hand2, chans=chans, rotation=rotation,
                                               rotate=rotate, invalid=invalid):
        return
    if invalid:
        bad = ~np.isfinite(vis)
        if bad.any():
            vis[bad] = 0
            flag |= bad
    if rotation is not None and rotate is not None and rotate.any():
        if rotate.all():
            vis *= rotation[:, None, :]
        else:
            vis[rotate] *= rotation[rotate][:, None, :]
    gain = grid["gain"]
    g1, g2 = gain[point1], np.conj(gain[point2])                                  # (nrow, nchan, 2)
    if chans is not None:
        g1, g2 = g1[:, chans], g2[:, chans]
    vis /= _per_corr(g1, hand1) * _per_corr(g2, hand2)
    apply_point_flags(flag, grid, point1, point2, chans)
    if weight is not None and grid["wscale"] is not None:
        wscale = grid["wscale"]
        w1, w2 = wscale[point1], wscale[point2]
        if chans is not None and wscale.shape[1] > 1:
            w1, w2 = w1[:, chans], w2[:, chans]
        weight *= _per_corr(w1, hand1) * _per_corr(w2, hand2)


def apply_gain_grid(vis, flag, weight, antenna1, antenna2, time, grid):
    """Correct rows in place with a grid from :func:`gain_grid` (every row's antennas and time must be on it).

    ``vis``/``flag`` (nrow, nchan, ncorr) and ``weight`` (same shape, or None) are modified in place.
    """
    apply_point_grid(vis, flag, weight, _as_points(grid), *_grid_index(grid, antenna1, antenna2, time))


def caltable_entries_from_vlbipy(tables, field_ids_by_name=None):
    """Convert vlbipy ``CalTable`` objects (models.py: path, interp, spwmap, gainfield, calwt, cal_type) to dicts.

    `gainfield` names are mapped to field ids with `field_ids_by_name` (name -> id); unknown names (or no mapping)
    select all fields with a log message.  Returns the list of dicts accepted by `apply_tables`.
    """
    entries = []
    for ct in tables:
        gainfield = str(getattr(ct, "gainfield", "") or "").strip()
        field_ids = []
        if gainfield and gainfield.lower() != "nearest":
            for name in gainfield.split(","):
                name = name.strip()
                if name.isdigit():
                    field_ids.append(int(name))
                elif field_ids_by_name is not None and name in field_ids_by_name:
                    field_ids.append(int(field_ids_by_name[name]))
                else:
                    log.info("caltable_entries_from_vlbipy: unknown gainfield %r for %s (%s); using all fields", name,
                             ct.path, getattr(ct, "cal_type", ""))
                    field_ids = []
                    break
        entries.append({"path": ct.path, "interp": str(getattr(ct, "interp", "linear") or "linear"),
                        "spwmap": list(getattr(ct, "spwmap", []) or []), "gainfield": field_ids,
                        "calwt": bool(getattr(ct, "calwt", False)), "cal_type": getattr(ct, "cal_type", "")})
    return entries
