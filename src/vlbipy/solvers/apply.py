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
`_gains_fringe`, CASA CTRateAwareTimeInterp1).

Frequency interpolation (second token of ``interp``, default "linear"): flagged channels of a solution are
interpolated across from its unflagged ones; the ``...flag`` modes keep them flagged (see `_interp_freq`).

casatools is imported lazily; getcol returns Fortran order so array columns are transposed (see caltable.py).
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
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


def load_caltable(path) -> CalTableData:
    """Read a CASA calibration table (main columns + SPECTRAL_WINDOW/ANTENNA/FIELD subtables) into a CalTableData.

    CPARAM tables give complex `param`, FPARAM tables float64.  All spws must have the same number of channels.
    """
    import casatools

    t_start = _time.perf_counter()
    path = Path(path).absolute()
    tb = casatools.table()
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
    # casatools returns Fortran order (npar, nchan_t, nrow); transpose to (nrow, nchan_t, npar).
    param = np.ascontiguousarray(np.asarray(tb.getcol(param_col)).T)
    flag = np.ascontiguousarray(np.asarray(tb.getcol("FLAG")).T, dtype=bool)
    snr = np.ascontiguousarray(np.asarray(tb.getcol("SNR")).T, dtype=np.float64)
    tb.close()
    if param.ndim == 2:
        param, flag, snr = param[:, None, :], flag[:, None, :], snr[:, None, :]
    param = param.astype(np.complex128 if np.iscomplexobj(param) else np.float64)

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


def _select_rows(table, ant, spw_t, field_id):
    """Return the row indices (sorted by time) for antenna `ant`, caltable spw `spw_t` and the field selection."""
    mask = (table.antenna1 == ant) & (table.spw_id == spw_t)
    if field_id is not None:
        mask &= np.isin(table.field_id, np.asarray(field_id, dtype=np.int64).reshape(-1))
    rows = np.nonzero(mask)[0]
    return rows[np.argsort(table.time[rows], kind="stable")]


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


def _interp_freq(values, flags, table_freq, chan_freq, mode="linear", keep_flags=False):
    """Bring gains from the table channels `table_freq` (nchan_t,) to the data channels `chan_freq` (nchan,).

    `values` complex and `flags` bool, both (ntime, nchan_t). Like CASA, flagged table channels are *interpolated
    across*: each data channel takes the nearest unflagged table channel ("nearest") or the amp/phase
    interpolation of the two unflagged channels around it ("linear", nearest beyond the ends), and is flagged only
    when the solution has no unflagged channel at all. With `keep_flags` (CASA's ``...flag`` modes) a data channel
    whose nearest table channel is flagged stays flagged. Returns (values (ntime, nchan), flags (ntime, nchan)).
    """
    table_freq = np.asarray(table_freq, dtype=np.float64)
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    ntime, nchan_t = values.shape
    nchan = chan_freq.size
    same_grid = table_freq.shape == chan_freq.shape and np.allclose(table_freq, chan_freq, rtol=0, atol=1.0)
    if same_grid and not flags.any():
        return values, flags
    if nchan_t == 1:
        return np.broadcast_to(values, (ntime, nchan)), np.broadcast_to(flags, (ntime, nchan))
    order = np.argsort(table_freq)
    tf, vals, fl = table_freq[order], values[:, order], flags[:, order]
    out = np.ones((ntime, nchan), dtype=np.complex128)
    out_flag = np.ones((ntime, nchan), dtype=bool)
    # Solutions rarely differ in which channels they flag (a bandpass is one row per antenna), so the channel
    # mapping is built once per distinct flag pattern.
    if (fl == fl[0]).all():
        patterns, member = fl[:1], np.zeros(ntime, dtype=np.int64)
    else:
        patterns, member = np.unique(fl, axis=0, return_inverse=True)
    nearest_all = _bracket(tf, chan_freq, "nearest")[0]
    for index, pattern in enumerate(patterns):
        rows = np.flatnonzero(member == index)
        good = np.flatnonzero(~pattern)
        if not good.size:
            continue
        lo, hi, w = _bracket(tf[good], chan_freq, mode)
        picked = vals[np.ix_(rows, good[lo])]
        blend = np.flatnonzero((hi != lo) & (w > 0))
        if blend.size:
            picked[:, blend] = _interp_amp_phase(picked[:, blend], vals[np.ix_(rows, good[hi[blend]])], w[blend])
        out[rows] = picked
        out_flag[rows] = pattern[nearest_all][None, :] if keep_flags else False
    return out, out_flag


def _gains_generic(table, rows, times, interp, pol):
    """Time-interpolated gains for one antenna/pol: returns (values (ntime, nchan_t), flags (ntime, nchan_t)).

    G/T/B Jones: amp/phase interpolation of CPARAM.  B TSYS: Tsys interpolated, gain = 1/sqrt(Tsys).
    """
    ntime = len(times)
    nchan_t = table.nchan_t
    if rows.size == 0:
        return np.ones((ntime, nchan_t), dtype=np.complex128), np.ones((ntime, nchan_t), dtype=bool)
    i0, i1, w = _bracket(table.time[rows], times, interp)
    p = table.param[rows][:, :, pol]
    f = table.flag[rows][:, :, pol]
    if table.kind == "B TSYS":
        # CASA interpolates the system temperature itself and then takes 1 / sqrt(Tsys).
        tsys = np.real(p).astype(np.float64)
        bad = ~(tsys > 0)
        used_hi = ((i1 != i0) & (w > 0))[:, None]
        value = (1.0 - w)[:, None] * tsys[i0] + w[:, None] * tsys[i1]
        flags = f[i0] | bad[i0] | (used_hi & (f[i1] | bad[i1])) | ~(value > 0)
        return (1.0 / np.sqrt(np.where(flags, 1.0, value))).astype(np.complex128), flags
    cp = p.astype(np.complex128)
    zero = cp == 0
    values = cp[i0]
    flags = f[i0] | zero[i0]
    blend = np.flatnonzero((i1 != i0) & (w > 0))
    if blend.size:
        values[blend] = _interp_amp_phase(values[blend], cp[i1[blend]], w[blend][:, None])
        flags[blend] |= f[i1[blend]] | zero[i1[blend]]
    return values, flags


def _gains_fringe(table, rows, times, chan_freq, interp, pol, f_ref_hz):
    """Time-interpolated Fringe Jones unit gains for one antenna/pol: (values (ntime, nchan), flags (ntime, nchan)).

    Follows CASA's rate-aware interpolation (``CTRateAwareTimeInterp1``), which works on the parameters:

    * a time that uses a single solution ("nearest", or beyond the first/last solution) takes that solution's
      delay and dispersive term, and its phase advanced by the rate from the solution time;
    * a time between two solutions ("linear") takes delay and dispersive term interpolated linearly, and for the
      phase the argument of the weighted mean of the two unit phasors obtained by advancing each solution's
      phase to that time with its own rate (so the result does not depend on how many turns apart they are).

    The rate always multiplies the table's reference frequency ``f_ref_hz``.
    """
    ntime, nchan = len(times), len(chan_freq)
    if rows.size == 0:
        return np.ones((ntime, nchan), dtype=np.complex128), np.ones((ntime, nchan), dtype=bool)
    i0, i1, w = _bracket(table.time[rows], times, interp)
    block = slice(FRINGE_NPARAM_PER_POL * pol, FRINGE_NPARAM_PER_POL * (pol + 1))
    params = np.real(table.param[rows, 0, block]).astype(np.float64)              # (nrow, 4): phase, delay, rate, disp
    row_time = table.time[rows]
    lo = params[i0]
    phase = lo[:, 0] + TWO_PI * f_ref_hz * lo[:, 2] * (times - row_time[i0])
    delay, disp = lo[:, 1].copy(), lo[:, 3].copy()
    blend = np.flatnonzero((i1 != i0) & (w > 0))
    if blend.size:
        first, second, frac = params[i0[blend]], params[i1[blend]], w[blend]
        later = second[:, 0] + TWO_PI * f_ref_hz * second[:, 2] * (times[blend] - row_time[i1[blend]])
        phase[blend] = np.angle((1.0 - frac) * np.exp(1j * phase[blend]) + frac * np.exp(1j * later))
        delay[blend] = (1.0 - frac) * first[:, 1] + frac * second[:, 1]
        disp[blend] = (1.0 - frac) * first[:, 3] + frac * second[:, 3]
    theta = phase[:, None] + delay[:, None] * (TWO_PI * (chan_freq - f_ref_hz) * 1e-9)[None, :]
    if np.any(disp):
        theta = theta + disp[:, None] * k_disp(chan_freq, float(np.min(chan_freq)), float(np.max(chan_freq)))[None, :]
    rowflag = table.flag[rows, 0, FRINGE_NPARAM_PER_POL * pol]
    flagged = rowflag[i0].copy()
    flagged[blend] |= rowflag[i1[blend]]
    return np.exp(1j * theta), np.broadcast_to(flagged[:, None], (ntime, nchan))


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
    interp, freq_interp, keep_chan_flags = parse_interp(interp)
    antenna_ids = np.asarray(antenna_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    chan_freq = np.asarray(chan_freq, dtype=np.float64).reshape(-1)
    spw_t = _mapped_spw(int(spw), spwmap)
    if spw_t >= table.spw_chan_freq.shape[0]:
        raise ValueError(f"{table.path}: caltable spw {spw_t} (data spw {spw}) does not exist")
    nant_sel, ntime, nchan = antenna_ids.size, times.size, chan_freq.size
    gains = np.ones((nant_sel, ntime, nchan, 2), dtype=np.complex128)
    gflag = np.ones((nant_sel, ntime, nchan, 2), dtype=bool)
    is_fringe = table.kind == "Fringe Jones"
    npol_t = table.npar // FRINGE_NPARAM_PER_POL if is_fringe else table.npar
    if is_fringe and table.npar != 2 * FRINGE_NPARAM_PER_POL:
        raise ValueError(f"{table.path}: Fringe Jones tables must have 8 parameters, got {table.npar}")
    if table.kind not in ("G Jones", "T Jones", "B Jones", "B TSYS", "Fringe Jones", *GAIN_CURVE_KINDS):
        log.warning("antenna_gains: table kind %r not explicitly supported, applying as a G-like gain", table.kind)
    f_ref_hz = float(table.spw_chan_freq[spw_t, table.spw_chan_freq.shape[1] // 2])
    for k, ant in enumerate(antenna_ids):
        rows = _select_rows(table, int(ant), spw_t, field_id)
        if table.kind in GAIN_CURVE_KINDS:
            for hand in range(2):
                pol = min(hand, npol_t - 1)
                good_rows = rows[~table.flag[rows, 0, pol]] if rows.size else rows
                if not good_rows.size:
                    continue
                ncoef = table.npar // 2
                if ncoef < 1:
                    continue
                i0, i1, w = _bracket(table.time[good_rows], times, interp)
                coefficients = table.param[good_rows, 0, pol * ncoef:(pol + 1) * ncoef]
                coeff = (1.0 - w[:, None]) * coefficients[i0] + w[:, None] * coefficients[i1]
                # CASA evaluates the curve in zenith angle [deg] (EPowerCurve/EGainCurve::calcPar).
                x = 90.0 - np.degrees(np.asarray(elevation_grid[k], dtype=np.float64))
                power = np.zeros(times.size, dtype=np.float64)
                for degree in range(ncoef - 1, -1, -1):
                    power = power * x + np.real(coeff[:, degree])
                if table.kind == "EPowerCurve":
                    gain = np.sqrt(np.maximum(power, 0.0))
                else:
                    gain = power
                bad = ~np.isfinite(gain) | (gain <= 0)
                gains[k, :, :, hand] = np.where(bad[:, None], 1.0, gain[:, None])
                gflag[k, :, :, hand] = np.broadcast_to(bad[:, None], (ntime, nchan))
            continue
        for hand in range(2):
            pol = min(hand, npol_t - 1)
            if is_fringe:
                values, flags = _gains_fringe(table, rows, times, chan_freq, interp, pol, f_ref_hz)
            else:
                values, flags = _gains_generic(table, rows, times, interp, pol)
                values, flags = _interp_freq(values, flags, table.spw_chan_freq[spw_t], chan_freq, freq_interp,
                                             keep_chan_flags)
            gains[k, :, :, hand] = np.where(flags, 1.0, values)
            gflag[k, :, :, hand] = flags
    return gains, gflag


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
        ``gain`` complex128 and ``gflag`` bool, shape (nant_sel, ntime, nchan, 2); ``wscale`` float64 of that
        shape holding the product of |g|^2 over the entries with ``calwt`` whose kind is in
        :data:`WEIGHT_CAL_KINDS`, and ``wflag`` the flags of those same entries (both None when there is none).
    """
    antenna_ids = np.asarray(antenna_ids, dtype=np.int64)
    shape = (antenna_ids.size, len(times), len(chan_freq), 2)
    gain = np.ones(shape, dtype=np.complex128)
    gflag = np.zeros(shape, dtype=bool)
    wscale, wflag = None, None
    elevation_grid = None
    for table, interp, spwmap, field_id, entry_calwt in entries:
        if table.kind in GAIN_CURVE_KINDS and elevation_grid is None:
            if antenna_xyz is None or field_dir is None:
                raise ValueError(f"{table.kind} application requires antenna positions and field direction")
            from .parang import elevation
            elevation_grid = elevation(antenna_xyz, float(field_dir[0]), float(field_dir[1]), times)[antenna_ids]
        gains, flags = antenna_gains(table, antenna_ids, times, int(spw), chan_freq, field_id=field_id,
                                     interp=interp, spwmap=spwmap,
                                     elevation_grid=elevation_grid if table.kind in GAIN_CURVE_KINDS else None)
        gain *= gains
        gflag |= flags
        if entry_calwt and table.kind in WEIGHT_CAL_KINDS:
            power = np.abs(gains) ** 2
            wscale = power if wscale is None else wscale * power
            # An antenna the table does not know at all leaves the weight alone (its data are flagged, but CASA
            # scales the weight by 1); only a solution that exists and is flagged zeroes it.
            spw_t = _mapped_spw(int(spw), spwmap)
            known = np.isin(antenna_ids, table.antenna1[table.spw_id == spw_t])
            flagged = flags & known[:, None, None, None]
            wflag = flagged if wflag is None else wflag | flagged
    return gain, gflag, wscale, wflag


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
    if weight is not None:
        weight = np.asarray(weight)
        wtype = np.float32 if weight.dtype.kind == "f" and weight.itemsize <= 4 else np.float64
        if weight.ndim == 2:
            weight = np.array(np.broadcast_to(weight[:, None, :], vis.shape), dtype=wtype)
        else:
            weight = np.array(weight, dtype=wtype, copy=copy)
    antenna1 = np.asarray(antenna1, dtype=np.int64)
    antenna2 = np.asarray(antenna2, dtype=np.int64)
    time = np.asarray(time, dtype=np.float64)
    spw = np.asarray(spw, dtype=np.int64)
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    hand1, hand2 = _corr_hands(ncorr)
    entries = tables if tables and isinstance(tables[0], tuple) else normalise_entries(tables, phase_only=phase_only,
                                                                                         calwt=calwt)
    gain_type = np.complex64 if vis.dtype == np.complex64 else np.complex128
    spws = np.unique(spw)
    for s in spws if entries else []:
        # A block of a single spw (the usual chunk) is corrected through a plain slice: no row copies.
        rows = slice(None) if spws.size == 1 else np.nonzero(spw == s)[0]
        utimes, t_idx = np.unique(time[rows], return_inverse=True)
        nsel = t_idx.size
        uants, a_idx = np.unique(np.concatenate([antenna1[rows], antenna2[rows]]), return_inverse=True)
        a1_idx, a2_idx = a_idx[:nsel], a_idx[nsel:]
        gain, gflag, wscale, wflag = combined_gains(entries, uants, utimes, int(s), chan_freq[s],
                                                    antenna_xyz=antenna_xyz, field_dir=field_dir)
        # A solution that is zero or not finite without being flagged (a Tsys of 0, an interpolation
        # between a valid and an invalid value) cannot calibrate anything: dividing by it would leave
        # NaN or infinite visibilities that are not flagged. Treat it as the flagged solution it is.
        gflag = gflag | ~np.isfinite(gain) | (gain == 0)
        gain = np.where(gflag, 1.0, gain).astype(gain_type)
        g1, g2 = gain[a1_idx, t_idx], np.conj(gain[a2_idx, t_idx])            # (nrow_spw, nchan, 2)
        f1, f2 = gflag[a1_idx, t_idx], gflag[a2_idx, t_idx]
        vis_s, flag_s = vis[rows], flag[rows]
        for c in range(ncorr):
            vis_s[:, :, c] /= g1[:, :, hand1[c]] * g2[:, :, hand2[c]]
            flag_s[:, :, c] |= f1[:, :, hand1[c]] | f2[:, :, hand2[c]]
        if spws.size > 1:
            vis[rows], flag[rows] = vis_s, flag_s
        if weight is not None and wscale is not None:
            # A flagged solution of a weight-calibrating table scales the weight to 0 (as CASA writes it).
            wscale = np.where(wflag, 0.0, wscale).astype(weight.dtype)
            w1, w2 = wscale[a1_idx, t_idx], wscale[a2_idx, t_idx]
            weight_s = weight[rows]
            for c in range(ncorr):
                weight_s[:, :, c] *= w1[:, :, hand1[c]] * w2[:, :, hand2[c]]
            if spws.size > 1:
                weight[rows] = weight_s
    log.debug("apply_tables: %d tables applied to %d rows x %d chan x %d corr (%d spws) in %.3f s", len(entries), nrow,
              nchan, ncorr, spws.size, _time.perf_counter() - t_start)
    return vis, flag, weight


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
