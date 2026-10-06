"""Apply CASA calibration tables to a block of visibilities with numpy (on-the-fly applycal).

Row layout (as read from an MS by dask-ms / casacore): vis (nrow, nchan, ncorr) complex, flag (nrow, nchan, ncorr)
bool, weight (nrow, nchan, ncorr) float, antenna1/antenna2/time/spw (nrow,).

Conventions (see /tmp/vlbipy_spec/CASA_FRINGE_CONVENTIONS.md):
- V_ij = g_i conj(g_j) V_true for ANTENNA1 = i, ANTENNA2 = j; correction V_corr = V_ij / (g_i conj(g_j)).
- Per correlation: RR uses (g_i^R, g_j^R), LL (g_i^L, g_j^L), RL (g_i^R, g_j^L), LR (g_i^L, g_j^R).
- A visibility whose antenna gain is flagged or missing is flagged (CASA applymode='calflag').
- With calwt the weights are scaled by |g_i g_j|^2 (noise scales like the data: sigma_corr = sigma / |g_i g_j|).

Supported VisCal kinds: "G Jones", "T Jones" (CPARAM (1, npol)), "B Jones" (CPARAM (nchan_t, npol)), "Fringe Jones"
(FPARAM (1, 8) = (phi0, delay_ns, rate, disp) x 2 pols, evaluated with ``vlbipy.solvers.fringe.predict_phase``),
"B TSYS" (FPARAM (1, npol) Tsys [K]; gain = 1/sqrt(Tsys) so that V_corr = V sqrt(Tsys_i Tsys_j) like CASA).
Gain curves ("EPowerCurve"/"EGainCurve") need the elevation and raise NotImplementedError unless skipped with
``phase_only=True``.

Time interpolation: "nearest" takes the nearest unflagged solution; "linear" interpolates amplitude and phase
separately between the two bracketing unflagged solutions (CASA interpolates amp/phase, not re/im) and uses the
nearest solution beyond the ends.  For "Fringe Jones" the rate extrapolates the phase from the solution time, and
"linear" interpolates the complex unit vectors obtained from both bracketing rows (CASA CTRateAwareTimeInterp1).

casatools is imported lazily; getcol returns Fortran order so array columns are transposed (see caltable.py).
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vlbipy.solvers.fringe import predict_phase

log = logging.getLogger(__name__)

AMPLITUDE_ONLY_KINDS = ("B TSYS", "EPowerCurve", "EGainCurve")
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
    log.info("load_caltable: %s kind=%r partype=%s rows=%d nchan_t=%d npar=%d nspw=%d in %.3f s", path, kind, partype,
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
    return lo, hi, w


def _interp_amp_phase(v0, v1, w):
    """Interpolate complex gains between v0 and v1 with weight w (broadcast): amplitude and phase separately."""
    amp = (1.0 - w) * np.abs(v0) + w * np.abs(v1)
    dphi = np.angle(v1 * np.conj(v0))
    phase = np.angle(v0) + w * dphi
    return amp * np.exp(1j * phase)


def _interp_freq(values, flags, table_freq, chan_freq):
    """Interpolate gains along the last-but-one axis from `table_freq` (nchan_t,) to `chan_freq` (nchan,).

    `values` complex (..., nchan_t, npol), `flags` same shape.  Identical grids are returned as is; otherwise linear
    amp/phase interpolation between the bracketing table channels (nearest at the edges); a data channel is flagged
    when either bracketing table channel is flagged.
    """
    table_freq = np.asarray(table_freq, dtype=np.float64)
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    if table_freq.shape == chan_freq.shape and np.allclose(table_freq, chan_freq, rtol=0, atol=1.0):
        return values, flags
    order = np.argsort(table_freq)
    tf, vals, fl = table_freq[order], values[..., order, :], flags[..., order, :]
    lo, hi, w = _bracket(tf, chan_freq, "linear")
    w = w[:, None]
    out = _interp_amp_phase(vals[..., lo, :], vals[..., hi, :], w)
    return out, fl[..., lo, :] | fl[..., hi, :]


def _unflagged_rows_per_pol(table, rows, pol):
    """Rows (subset of `rows`) with at least one unflagged element for polarisation `pol`."""
    if table.kind == "Fringe Jones":
        ok = ~table.flag[rows, 0, FRINGE_NPARAM_PER_POL * pol]
    else:
        ok = ~table.flag[rows][:, :, pol].all(axis=1)
    return rows[ok]


def _gains_generic(table, rows, times, interp, pol):
    """Time-interpolated gains for one antenna/pol: returns (values (ntime, nchan_t), flags (ntime, nchan_t)).

    G/T/B Jones: amp/phase interpolation of CPARAM.  B TSYS: 1/sqrt(Tsys) interpolated in amplitude.
    """
    ntime = len(times)
    nchan_t = table.nchan_t
    if rows.size == 0:
        return np.ones((ntime, nchan_t), dtype=np.complex128), np.ones((ntime, nchan_t), dtype=bool)
    i0, i1, w = _bracket(table.time[rows], times, interp)
    p = table.param[rows][:, :, pol]
    f = table.flag[rows][:, :, pol]
    if table.kind == "B TSYS":
        tsys = np.real(p).astype(np.float64)
        bad = ~(tsys > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            amp = np.where(bad, 1.0, 1.0 / np.sqrt(np.where(bad, 1.0, tsys)))
        values = ((1.0 - w)[:, None] * amp[i0] + w[:, None] * amp[i1]).astype(np.complex128)
        flags = f[i0] | f[i1] | bad[i0] | bad[i1]
        return values, flags
    cp = p.astype(np.complex128)
    values = _interp_amp_phase(cp[i0], cp[i1], w[:, None])
    flags = f[i0] | f[i1] | (np.abs(cp[i0]) == 0) | (np.abs(cp[i1]) == 0)
    return values, flags


def _gains_fringe(table, rows, times, chan_freq, interp, pol, f_ref_hz):
    """Time-interpolated Fringe Jones unit gains for one antenna/pol: (values (ntime, nchan), flags (ntime, nchan)).

    Each candidate row is evaluated at the data times with predict_phase (the rate extrapolates from the row TIME);
    "linear" interpolates the complex unit vectors of the two bracketing rows and renormalises them.
    """
    ntime, nchan = len(times), len(chan_freq)
    if rows.size == 0:
        return np.ones((ntime, nchan), dtype=np.complex128), np.ones((ntime, nchan), dtype=bool)
    i0, i1, w = _bracket(table.time[rows], times, interp)
    used, inv = np.unique(np.concatenate([i0, i1]), return_inverse=True)
    j0, j1 = inv[:ntime], inv[ntime:]
    fmin, fmax = float(np.min(chan_freq)), float(np.max(chan_freq))
    block = slice(FRINGE_NPARAM_PER_POL * pol, FRINGE_NPARAM_PER_POL * (pol + 1))
    units = np.empty((used.size, ntime, nchan), dtype=np.complex128)
    for k, r in enumerate(used):
        params = np.real(table.param[rows[r], 0, block])
        phase = predict_phase(params, chan_freq, times, f_ref_hz, float(table.time[rows[r]]), fmin, fmax)
        units[k] = np.exp(1j * phase)
    t_idx = np.arange(ntime)
    v0, v1 = units[j0, t_idx], units[j1, t_idx]
    values = (1.0 - w)[:, None] * v0 + w[:, None] * v1
    mod = np.abs(values)
    values = np.where(mod > 0, values / np.where(mod > 0, mod, 1.0), 1.0)
    rowflag = table.flag[rows, 0, FRINGE_NPARAM_PER_POL * pol]
    flags = np.broadcast_to((rowflag[i0] | rowflag[i1])[:, None], (ntime, nchan)).copy()
    return values, flags


def antenna_gains(table, antenna_ids, times, spw, chan_freq, *, field_id=None, interp="linear", spwmap=None):
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
    if table.kind in GAIN_CURVE_KINDS:
        raise NotImplementedError("gain curve needs elevation; apply amplitude tables with CASA")
    interp = str(interp).split(",")[0].strip().lower() or "linear"
    if interp not in ("linear", "nearest"):
        raise ValueError(f"interp must be 'linear' or 'nearest', got {interp!r}")
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
    if table.kind not in ("G Jones", "T Jones", "B Jones", "B TSYS", "Fringe Jones"):
        log.warning("antenna_gains: table kind %r not explicitly supported, applying as a G-like gain", table.kind)
    f_ref_hz = float(table.spw_chan_freq[spw_t, table.spw_chan_freq.shape[1] // 2])
    for k, ant in enumerate(antenna_ids):
        rows = _select_rows(table, int(ant), spw_t, field_id)
        for hand in range(2):
            pol = min(hand, npol_t - 1)
            rows_p = _unflagged_rows_per_pol(table, rows, pol)
            if is_fringe:
                values, flags = _gains_fringe(table, rows_p, times, chan_freq, interp, pol, f_ref_hz)
            else:
                values, flags = _gains_generic(table, rows_p, times, interp, pol)
                if table.nchan_t > 1 or values.shape[1] != nchan:
                    values, flags = _interp_freq(values[..., None], flags[..., None], table.spw_chan_freq[spw_t],
                                                 chan_freq)
                    values, flags = values[..., 0], flags[..., 0]
                else:
                    values = np.broadcast_to(values, (ntime, nchan))
                    flags = np.broadcast_to(flags, (ntime, nchan))
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
    interp = str(entry.get("interp", "linear") or "linear").split(",")[0].strip().lower()
    spwmap = list(entry.get("spwmap", []) or [])
    gainfield = entry.get("gainfield", None)
    field_id = None
    if isinstance(gainfield, str):
        if gainfield.strip().lower() == "nearest":
            log.info("apply_tables: gainfield='nearest' for %s treated as all fields", table.path)
        elif gainfield.strip():
            field_id = [int(x) for x in gainfield.split(",")]
    elif gainfield is not None and len(gainfield) > 0:
        field_id = [int(x) for x in gainfield]
    return table, interp, spwmap, field_id, bool(entry.get("calwt", False))


def apply_tables(vis, flag, weight, antenna1, antenna2, time, spw, chan_freq, tables, *, phase_only=False,
                 calwt=False):
    """Apply a chain of calibration tables to a block of MS rows.

    Parameters
    ----------
    vis : complex array (nrow, nchan, ncorr)
    flag : bool array (nrow, nchan, ncorr)
    weight : float array (nrow, nchan, ncorr) or (nrow, ncorr) or None
    antenna1, antenna2, time, spw : arrays (nrow,)
    chan_freq : float array (nspw, nchan)
        Channel frequencies of ALL spws of the MS [Hz]; rows are grouped per unique `spw`.
    tables : list of dict
        Keys: "path" (str or CalTableData; alias "table"), "interp" ("linear"/"nearest", first token before a
        comma), "spwmap" (list, empty = identity), "gainfield" (empty/None = all fields; list of field ids or a
        comma-separated string; "nearest" = all fields for now), "calwt" (bool).
    phase_only : bool
        Skip amplitude-only tables (B TSYS, gain curves): used before fringe fitting where amplitudes are irrelevant.
    calwt : bool
        Default weight calibration for entries without their own "calwt" key... entries override it.

    Returns
    -------
    (vis_corr, flag_corr, weight_corr)
        New arrays (inputs untouched); weight_corr has the shape of `weight` broadcast to vis (None stays None).
    """
    t_start = _time.perf_counter()
    vis = np.array(vis, copy=True)
    flag = np.array(flag, dtype=bool, copy=True)
    nrow, nchan, ncorr = vis.shape
    if weight is not None:
        weight = np.asarray(weight)
        weight = np.array(np.broadcast_to(weight[:, None, :], vis.shape) if weight.ndim == 2 else weight, copy=True,
                          dtype=np.float32 if weight.dtype.kind == "f" and weight.itemsize <= 4 else np.float64)
    antenna1 = np.asarray(antenna1, dtype=np.int64)
    antenna2 = np.asarray(antenna2, dtype=np.int64)
    time = np.asarray(time, dtype=np.float64)
    spw = np.asarray(spw, dtype=np.int64)
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    hand1, hand2 = _corr_hands(ncorr)
    cache = {}
    entries = []
    for entry in tables:
        table, interp, spwmap, field_id, entry_calwt = _normalise_entry(entry, cache)
        if phase_only and table.kind in AMPLITUDE_ONLY_KINDS:
            log.info("apply_tables: phase_only, skipping %s (%s)", table.path, table.kind)
            continue
        entries.append((table, interp, spwmap, field_id, entry_calwt or ("calwt" not in entry and calwt)))
    for s in np.unique(spw):
        rows = np.nonzero(spw == s)[0]
        utimes, t_idx = np.unique(time[rows], return_inverse=True)
        uants, a_idx = np.unique(np.concatenate([antenna1[rows], antenna2[rows]]), return_inverse=True)
        a1_idx, a2_idx = a_idx[:rows.size], a_idx[rows.size:]
        for table, interp, spwmap, field_id, entry_calwt in entries:
            gains, gflag = antenna_gains(table, uants, utimes, int(s), chan_freq[s], field_id=field_id, interp=interp,
                                         spwmap=spwmap)
            g1 = gains[a1_idx, t_idx][:, :, hand1]          # (nrow_spw, nchan, ncorr)
            g2 = gains[a2_idx, t_idx][:, :, hand2]
            f12 = gflag[a1_idx, t_idx][:, :, hand1] | gflag[a2_idx, t_idx][:, :, hand2]
            factor = g1 * np.conj(g2)
            factor = np.where(f12, 1.0, factor)
            vis[rows] = vis[rows] / factor.astype(vis.dtype)
            flag[rows] |= f12
            if entry_calwt and weight is not None:
                weight[rows] = weight[rows] * (np.abs(factor) ** 2).astype(weight.dtype)
    log.info("apply_tables: %d tables applied to %d rows x %d chan x %d corr (%d spws) in %.3f s", len(entries), nrow,
             nchan, ncorr, np.unique(spw).size, _time.perf_counter() - t_start)
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
