"""Pure numpy/scipy VLBI fringe fitting reproducing the CASA ``fringefit`` conventions.

Conventions (see CASA_FRINGE_CONVENTIONS.md):
- V_ij (ANTENNA1=i < ANTENNA2=j) = g_i conj(g_j) V_true, so arg(V_ij) = theta_i - theta_j.
- Antenna phase model: theta(f, t) = phi0 + 2pi tau (f - f_ref) 1e-9 + 2pi r f_ref (t - t_sol) + K k_disp(f).
- FFT stage: weighted unit vectors on the baseline to the reference antenna, 2D FFT padded by ``pad``,
  peak search within windows, sub-bin refinement; phases referenced to (first channel, first integration).
- Least squares: weighted unit-vector residuals on every baseline (w = sqrt(WEIGHT_SPECTRUM)), solved with a
  Levenberg-Marquardt iteration on the exact normal equations (J^T J is constant for unit-vector residuals).
- Afterwards phi0 is moved from the grid origin to (f_ref, t_sol); the reference antenna row has zero
  parameters, is unflagged and gets SNR = 999.

Units: delays in ns, rates in s/s, phases in rad, dispersive term in CASA units, frequencies in Hz, times in s.
"""
from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass, field

import numpy as np
import scipy.fft

log = logging.getLogger(__name__)

REFANT_SNR_SENTINEL = 999.0
#: Threads scipy.fft may use per transform (-1 = all cores); worker processes run with 1.
FFT_WORKERS = int(os.environ.get("VLBIPY_FFT_WORKERS", "-1"))
TWO_PI = 2.0 * np.pi
# Parameter order inside FPARAM blocks: phi0 [rad], tau [ns], rate [s/s], disp [CASA units].
PARAM_NAMES = ("phase", "delay_ns", "rate", "disp")


# ----------------------------------------------------------------------------------------------------------------
# Phase model
# ----------------------------------------------------------------------------------------------------------------
def k_disp(freq, fmin, fmax):
    """Dispersive basis function of CASA: 1e6 * 2pi * (1/f + (f - fmin - fmax) / (fmin fmax)).

    Parameters
    ----------
    freq : array_like
        Channel frequencies [Hz].
    fmin, fmax : float or array_like
        Minimum/maximum channel frequency of the spectral window of each channel [Hz] (broadcastable to freq).

    Returns
    -------
    numpy.ndarray
        k_disp(f) [rad per CASA dispersive unit], same shape as freq.
    """
    f = np.asarray(freq, dtype=np.float64)
    fmin = np.asarray(fmin, dtype=np.float64)
    fmax = np.asarray(fmax, dtype=np.float64)
    return 1e6 * TWO_PI * (1.0 / f + (f - fmin - fmax) / (fmin * fmax))


def predict_phase(params, freq, time, f_ref_hz, t_sol, spw_fmin, spw_fmax):
    """Evaluate the CASA antenna phase model theta(f, t) for one set of parameters.

    Parameters
    ----------
    params : array_like, shape (..., 4)
        (phi0 [rad], tau [ns], rate [s/s], disp [CASA units]).
    freq : array_like, shape (nchan,)
        Channel frequencies [Hz].
    time : array_like, shape (ntime,)
        Sample times [s].
    f_ref_hz : float
        Reference frequency of the solution [Hz].
    t_sol : float
        Reference time of the solution [s].
    spw_fmin, spw_fmax : float or array_like, shape (nchan,)
        Min/max frequency of the spectral window each channel belongs to [Hz].

    Returns
    -------
    numpy.ndarray, shape (..., ntime, nchan)
        Model phase [rad] (not wrapped).
    """
    p = np.asarray(params, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    phi0, tau, rate, disp = (p[..., i][..., None, None] for i in range(4))
    df_term = TWO_PI * (freq - f_ref_hz) * 1e-9
    dt_term = TWO_PI * f_ref_hz * (time - t_sol)
    kd = k_disp(freq, spw_fmin, spw_fmax)
    return phi0 + tau * df_term[None, :] + rate * dt_term[:, None] + disp * kd[None, :]


def wrap_phase(phase):
    """Wrap phases to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(phase, dtype=np.float64)))


# ----------------------------------------------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class FringeData:
    """One solution interval laid out as (baseline, time, channel, pol) arrays.

    Attributes
    ----------
    vis : complex array (nbl, ntime, nchan, npol)
        Parallel-hand visibilities (RR[, LL]); zero where absent.
    weight : float array (nbl, ntime, nchan, npol)
        WEIGHT_SPECTRUM; 0 where flagged or absent.
    flag : bool array (nbl, ntime, nchan, npol)
    antenna1, antenna2 : int arrays (nbl,)
        antenna1 < antenna2 (MS convention).
    time : float64 array (ntime,)
        Sample times [s] (MJD seconds), increasing.
    freq : float64 array (nchan,)
        Frequency of each channel of the concatenated (all spws) grid [Hz], increasing.
    spw_of_chan : int array (nchan,)
        Spectral window id of each channel.
    chan_offset : int array (nchan,)
        Position of each channel on the uniform FFT grid, round((f - f[0]) / df).
    nant : int
        Number of antennas in the ANTENNA table (solution arrays are indexed by antenna id).
    f_ref_fixed : float, optional
        Reference frequency to fit against [Hz]; ``None`` uses the centre of ``freq``. A
        ``combine='spw'`` solve passes the centre of its whole channel selection so that every
        solution interval shares one reference frequency even when an interval is missing the
        antenna that holds an edge subband.
    """

    vis: np.ndarray
    weight: np.ndarray
    flag: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    time: np.ndarray
    freq: np.ndarray
    spw_of_chan: np.ndarray
    chan_offset: np.ndarray
    nant: int
    f_ref_fixed: float | None = None

    @property
    def nbl(self):
        """Number of baselines."""
        return self.vis.shape[0]

    @property
    def ntime(self):
        """Number of integrations."""
        return self.vis.shape[1]

    @property
    def nchan(self):
        """Number of channels of the concatenated grid."""
        return self.vis.shape[2]

    @property
    def npol(self):
        """Number of parallel-hand polarisations (1 or 2)."""
        return self.vis.shape[3]

    @property
    def f_ref_hz(self):
        """Reference frequency of the solve [Hz]: ``f_ref_fixed``, else the centre of the grid."""
        if self.f_ref_fixed is not None:
            return float(self.f_ref_fixed)
        return 0.5 * (float(self.freq.min()) + float(self.freq.max()))

    @property
    def t_sol(self):
        """Centre of the solution interval [s]."""
        return 0.5 * (float(self.time.min()) + float(self.time.max()))

    def spw_freq_limits(self):
        """Per-channel (fmin, fmax) of the spectral window each channel belongs to, shapes (nchan,) [Hz]."""
        fmin = np.empty(self.nchan, dtype=np.float64)
        fmax = np.empty(self.nchan, dtype=np.float64)
        for spw in np.unique(self.spw_of_chan):
            sel = self.spw_of_chan == spw
            fmin[sel] = self.freq[sel].min()
            fmax[sel] = self.freq[sel].max()
        return fmin, fmax

    @classmethod
    def from_baselines(cls, vis, flag, weight, antenna1, antenna2, time, spw, chan_freq, *, nant=None,
                       f_ref_hz=None):
        """Build a FringeData from MS row-ordered arrays (the only place that knows the MS row layout).

        Parameters
        ----------
        vis : complex array (nrow, nchan_spw, ncorr)
        flag : bool array (nrow, nchan_spw, ncorr)
        weight : float array (nrow, nchan_spw, ncorr) or (nrow, ncorr) or None
            WEIGHT_SPECTRUM (preferred), WEIGHT (broadcast over channels) or None (unit weights).
        antenna1, antenna2 : int arrays (nrow,)
        time : float array (nrow,)  [s]
        spw : int array (nrow,)
            Spectral window id of each row (DATA_DESC_ID already mapped to spw).
        chan_freq : float array (nspw_total, nchan_spw)
            Channel frequencies per spw id [Hz], already restricted to the selected channels.
        nant : int, optional
            Number of antennas in the ANTENNA table; default max(antenna id) + 1.
        f_ref_hz : float, optional
            Reference frequency of the solve [Hz]; default the centre of the channels present.

        Returns
        -------
        FringeData
            Correlations 0 and ncorr-1 are taken as the parallel hands; autocorrelations are dropped; rows with
            antenna1 > antenna2 are swapped and conjugated; channels are ordered by increasing frequency.
        """
        vis = np.asarray(vis)
        flag = np.asarray(flag, dtype=bool)
        a1 = np.asarray(antenna1, dtype=np.int64).copy()
        a2 = np.asarray(antenna2, dtype=np.int64).copy()
        time = np.asarray(time, dtype=np.float64)
        spw = np.asarray(spw, dtype=np.int64)
        chan_freq = np.asarray(chan_freq, dtype=np.float64)
        nrow, nchan_spw, ncorr = vis.shape
        corr = [0] if ncorr == 1 else [0, ncorr - 1]
        if weight is None:
            weight = np.ones(vis.shape, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim == 2:
            weight = np.broadcast_to(weight[:, None, :], vis.shape)
        swap = a1 > a2
        if swap.any():
            a1[swap], a2[swap] = a2[swap], a1[swap]
            vis = vis.copy()
            vis[swap] = np.conj(vis[swap])
        keep = a1 != a2
        if not keep.all():
            vis, flag, weight = vis[keep], flag[keep], weight[keep]
            a1, a2, time, spw = a1[keep], a2[keep], time[keep], spw[keep]
        if len(corr) != ncorr:
            vis, flag, weight = vis[..., corr], flag[..., corr], weight[..., corr]
        if nant is None:
            nant = int(max(a1.max(), a2.max())) + 1 if a1.size else 0
        spw_ids, spw_idx = np.unique(spw, return_inverse=True)
        times, t_idx = np.unique(time, return_inverse=True)
        pair_key = a1 * (nant + 1) + a2
        pair_ids, bl_idx = np.unique(pair_key, return_inverse=True)
        bl_a1 = pair_ids // (nant + 1)
        bl_a2 = pair_ids % (nant + 1)
        nbl, nt, nspw, npol = pair_ids.size, times.size, spw_ids.size, len(corr)
        nchan = nspw * nchan_spw
        out_vis = np.zeros((nbl, nt, nchan, npol), dtype=np.complex64)
        out_flag = np.ones((nbl, nt, nchan, npol), dtype=bool)
        out_w = np.zeros((nbl, nt, nchan, npol), dtype=np.float32)
        # Rows arrive run by run (one spw each), so the scatter goes segment by segment: two index arrays and a
        # channel slice per segment copy whole (channel, pol) cells, far faster than indexing every sample.
        edges = np.concatenate([[0], np.flatnonzero(np.diff(spw_idx)) + 1, [spw_idx.size]])
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            chans = slice(int(spw_idx[lo]) * nchan_spw, (int(spw_idx[lo]) + 1) * nchan_spw)
            rows_bl, rows_t = bl_idx[lo:hi], t_idx[lo:hi]
            out_vis[rows_bl, rows_t, chans] = vis[lo:hi]
            out_flag[rows_bl, rows_t, chans] = flag[lo:hi]
            out_w[rows_bl, rows_t, chans] = np.where(flag[lo:hi], np.float32(0.0), weight[lo:hi])
        freq = chan_freq[spw_ids].reshape(-1)
        spw_of_chan = np.repeat(spw_ids, nchan_spw)
        order = np.argsort(freq, kind="stable")
        if not np.array_equal(order, np.arange(nchan)):
            freq, spw_of_chan = freq[order], spw_of_chan[order]
            out_vis, out_flag, out_w = out_vis[:, :, order], out_flag[:, :, order], out_w[:, :, order]
        df = float(np.median(np.abs(np.diff(freq)))) if nchan > 1 else 1.0
        chan_offset = np.rint((freq - freq[0]) / df).astype(np.int64)
        log.debug("FringeData.from_baselines: nbl=%d ntime=%d nchan=%d npol=%d nspw=%d", nbl, nt, nchan, npol, nspw)
        return cls(vis=out_vis, weight=out_w, flag=out_flag, antenna1=bl_a1, antenna2=bl_a2, time=times, freq=freq,
                   spw_of_chan=spw_of_chan, chan_offset=chan_offset, nant=int(nant), f_ref_fixed=f_ref_hz)


def antenna_time_centroid(data):
    """Weighted time centroid per antenna over all its baselines, shape (nant,) [s]; t_sol where no data."""
    w_bl = data.weight.sum(axis=(2, 3), dtype=np.float64)  # (nbl, ntime)
    wt_bl = w_bl * data.time[None, :]
    sw = np.zeros(data.nant, dtype=np.float64)
    swt = np.zeros(data.nant, dtype=np.float64)
    for ants in (data.antenna1, data.antenna2):
        np.add.at(sw, ants, w_bl.sum(axis=1))
        np.add.at(swt, ants, wt_bl.sum(axis=1))
    centroid = np.full(data.nant, data.t_sol, dtype=np.float64)
    has = sw > 0
    centroid[has] = swt[has] / sw[has]
    return centroid


# ----------------------------------------------------------------------------------------------------------------
# FFT stage
# ----------------------------------------------------------------------------------------------------------------
def _baselines_to_refant(data, refant):
    """Return (antenna ids, baseline index, conjugate flag) of every baseline that includes the reference antenna.

    conjugate is True when the refant is ANTENNA1 so that conj(V) has arg = theta_k - theta_ref = theta_k.
    """
    is_a1 = data.antenna1 == refant
    is_a2 = data.antenna2 == refant
    bl = np.nonzero(is_a1 | is_a2)[0]
    ants = np.where(is_a1[bl], data.antenna2[bl], data.antenna1[bl])
    return ants, bl, is_a1[bl]


def _fft_snr(peak, sumw, sumww, xcount):
    """AIPS FRING signal-to-noise estimate (CASA DelayRateFFTCombo::snr), vectorised; 0 where no data."""
    snr = np.zeros_like(peak, dtype=np.float64)
    ok = (sumw > 0) & (xcount > 0)
    pk = np.minimum(peak[ok], 0.999 * sumw[ok])
    x = 0.5 * np.pi * pk / sumw[ok]
    snr[ok] = np.tan(x) ** 1.163 * np.sqrt(sumw[ok] / np.sqrt(sumww[ok] / xcount[ok]))
    return snr


def _refine_peak(z, x, y, tau0, rate0, bin_tau, bin_rate, nstep=21):
    """Refine an FFT peak by zooming a local DFT around (tau0, rate0).

    Parameters
    ----------
    z : complex array (ntime, nchan)
        Weighted unit vectors (zero where flagged).
    x : float array (nchan,)
        (f - f0) * 1e-9  [GHz], so that 2pi tau x is the delay phase.
    y : float array (ntime,)
        (t - t0) * f0  [cycles per s/s], so that 2pi rate y is the rate phase.
    tau0, rate0 : float
        Integer-bin FFT estimates [ns], [s/s].
    bin_tau, bin_rate : float
        FFT bin sizes [ns], [s/s].

    Returns
    -------
    tuple (tau, rate, phase, amplitude)
    """
    grid = np.linspace(-1.0, 1.0, nstep)
    for span in (1.0, 0.1):
        taus = tau0 + span * grid * bin_tau
        rates = rate0 + span * grid * bin_rate
        ef = np.exp(-1j * TWO_PI * np.outer(x, taus))
        et = np.exp(-1j * TWO_PI * np.outer(rates, y))
        amp = np.abs(et @ z @ ef)
        it, jf = np.unravel_index(np.argmax(amp), amp.shape)
        tau0, rate0 = taus[jf], rates[it]
    step_tau, step_rate = 0.1 * bin_tau * (grid[1] - grid[0]), 0.1 * bin_rate * (grid[1] - grid[0])
    if 0 < jf < nstep - 1:
        den = amp[it, jf - 1] - 2 * amp[it, jf] + amp[it, jf + 1]
        if den < 0:
            tau0 += 0.5 * (amp[it, jf - 1] - amp[it, jf + 1]) / den * step_tau
    if 0 < it < nstep - 1:
        den = amp[it - 1, jf] - 2 * amp[it, jf] + amp[it + 1, jf]
        if den < 0:
            rate0 += 0.5 * (amp[it - 1, jf] - amp[it + 1, jf]) / den * step_rate
    s = np.sum(z * np.exp(-1j * TWO_PI * (tau0 * x[None, :] + rate0 * y[:, None])))
    return float(tau0), float(rate0), float(np.angle(s)), float(np.abs(s))


def _empty_fft_result(nant, npol):
    """All-flagged FFT result dictionary."""
    z = np.zeros((nant, npol), dtype=np.float64)
    return dict(phase=z.copy(), delay_ns=z.copy(), rate=z.copy(), snr=z.copy(), peak=z.copy(), sumw=z.copy(),
                sumww=z.copy(), xcount=np.zeros((nant, npol), dtype=np.int64), ok=np.zeros((nant, npol), dtype=bool))


def _contiguous_segments(index):
    """Split a strictly increasing integer index into (source slice, target slice) pairs of contiguous runs."""
    index = np.asarray(index, dtype=np.int64)
    edges = np.concatenate([[0], np.flatnonzero(np.diff(index) != 1) + 1, [index.size]])
    return [(slice(int(lo), int(hi)), slice(int(index[lo]), int(index[lo]) + int(hi - lo)))
            for lo, hi in zip(edges[:-1], edges[1:])]


def _place_on_grid(grid, z, t_off, chan_offset):
    """Copy ``z`` (..., ntime, nchan) onto the uniform FFT ``grid`` at integration offsets ``t_off`` and channel
    offsets ``chan_offset`` (both increasing), block by contiguous block."""
    for t_src, t_dst in _contiguous_segments(t_off):
        for c_src, c_dst in _contiguous_segments(chan_offset):
            grid[..., t_dst, c_dst] = z[..., t_src, c_src]


def fringe_fft_search(data, refant, *, pad=4, delay_window_ns=None, rate_window=None):
    """FFT delay/rate search on the baselines to the reference antenna (CASA DelayRateFFT).

    Parameters
    ----------
    data : FringeData
    refant : int
        Reference antenna id.
    pad : int
        Zero-padding factor on both FFT axes (CASA nPadFactor = 4).
    delay_window_ns : (float, float), optional
        Delay search window [ns]; default the full unambiguous range.
    rate_window : (float, float), optional
        Rate search window [s/s]; default the full range.

    Returns
    -------
    dict
        Arrays of shape (nant, npol): phase [rad], delay_ns, rate [s/s] (phase referenced to the first channel
        and the first integration of the grid), snr, peak, sumw, sumww, xcount, ok (bool), plus scalars
        f0_hz (centroid frequency used for the rate), f_grid0_hz, t_grid0, bin_delay_ns, bin_rate.
    """
    t_start = _time.perf_counter()
    nant, npol = data.nant, data.npol
    ants, bl_index, conj = _baselines_to_refant(data, refant)
    result = _empty_fft_result(nant, npol)
    f_grid0, t_grid0, f0 = float(data.freq[0]), float(data.time[0]), data.f_ref_hz
    ngrid = int(data.chan_offset.max()) + 1
    df = (float(data.freq[-1]) - f_grid0) / data.chan_offset[-1] if ngrid > 1 else 1.0
    dt = float(np.median(np.diff(data.time))) if data.ntime > 1 else 1.0
    t_off = np.rint((data.time - t_grid0) / dt).astype(np.int64)
    nt_grid = int(t_off.max()) + 1
    nt_pad, nf_pad = nt_grid * pad, ngrid * pad
    bin_delay = 1e9 / (nf_pad * df)
    bin_rate = 1.0 / (nt_pad * dt * f0)
    result.update(f0_hz=f0, f_grid0_hz=f_grid0, t_grid0=t_grid0, bin_delay_ns=bin_delay, bin_rate=bin_rate)
    if ants.size == 0:
        log.warning("fringe_fft_search: no baselines to refant %d", refant)
        return result
    # Weighted unit vectors, zero where flagged; conjugated when the refant is ANTENNA1 (sgn rule of CASA).
    flagged = data.flag[bl_index]
    w = np.where(flagged, np.float32(0.0), data.weight[bl_index])
    z = data.vis[bl_index].astype(np.complex64)
    amp = np.abs(z)
    z *= np.divide(w, amp, out=np.zeros_like(amp), where=amp > 0)
    z[conj] = np.conj(z[conj])
    z = np.ascontiguousarray(z.transpose(0, 3, 1, 2))  # (nk, npol, ntime, nchan)
    grid = np.zeros((ants.size, npol, nt_grid, ngrid), dtype=np.complex64)
    _place_on_grid(grid, z, t_off, data.chan_offset)
    # Frequency axis first, on the unpadded time rows only (a quarter of the padded grid), then the time axis.
    spec = scipy.fft.fft(grid, n=nf_pad, axis=-1, workers=FFT_WORKERS)
    spec = scipy.fft.fft(spec, n=nt_pad, axis=-2, workers=FFT_WORKERS)
    power = spec.real ** 2 + spec.imag ** 2
    delay_axis = scipy.fft.fftfreq(nf_pad, d=df) * 1e9
    rate_axis = scipy.fft.fftfreq(nt_pad, d=dt) / f0
    if delay_window_ns is not None or rate_window is not None:
        valid = np.ones((nt_pad, nf_pad), dtype=bool)
        if delay_window_ns is not None:
            valid &= ((delay_axis >= min(delay_window_ns)) & (delay_axis <= max(delay_window_ns)))[None, :]
        if rate_window is not None:
            valid &= ((rate_axis >= min(rate_window)) & (rate_axis <= max(rate_window)))[:, None]
        power[:, :, ~valid] = -1.0
    flat = np.argmax(power.reshape(ants.size, npol, -1), axis=-1)
    it, jf = np.unravel_index(flat, (nt_pad, nf_pad))
    peak = np.sqrt(np.maximum(np.take_along_axis(power.reshape(ants.size, npol, -1), flat[..., None],
                                                 axis=-1)[..., 0], 0.0)).astype(np.float64)
    sumw = w.sum(axis=(1, 2), dtype=np.float64)
    sumww = np.einsum("ktfp,ktfp->kp", w, w, dtype=np.float64)
    xcount = np.count_nonzero(~flagged, axis=(1, 2))
    ok = (sumw > 0) & (peak > 0)
    x = (data.freq - f_grid0) * 1e-9
    y = (data.time - t_grid0) * f0
    for k in range(ants.size):
        for p in range(npol):
            if not ok[k, p]:
                continue
            tau, rate, phase, _ = _refine_peak(z[k, p], x, y, delay_axis[jf[k, p]], rate_axis[it[k, p]], bin_delay,
                                               bin_rate)
            result["delay_ns"][ants[k], p], result["rate"][ants[k], p], result["phase"][ants[k], p] = tau, rate, phase
    result["peak"][ants], result["sumw"][ants], result["sumww"][ants] = peak, sumw, sumww
    result["xcount"][ants], result["ok"][ants] = xcount, ok
    result["snr"][ants] = _fft_snr(peak, sumw, sumww, xcount)
    log.info("fringe_fft_search: refant=%d nant=%d grid=(%d x %d) pad=%d in %.3f s", refant, ants.size, nt_grid, ngrid,
             pad, _time.perf_counter() - t_start)
    return result


# ----------------------------------------------------------------------------------------------------------------
# Global least squares
# ----------------------------------------------------------------------------------------------------------------
def _normal_matrix(w2, basis_f, basis_t, slot1, slot2, nslot):
    """Constant J^T J of the unit-vector residuals, shape (nslot*nparam, nslot*nparam).

    With g_p(t, f) = basis_f[p](f) + basis_t[p](t) the per-baseline block sum_tf w2 g_p g_q splits into four small
    matmuls; the blocks are then scattered onto (ant1, ant1), (ant2, ant2) and -(ant1, ant2), -(ant2, ant1).
    """
    nparam = basis_f.shape[0]
    wf = w2.sum(axis=1)  # (nbl, nchan)
    wt = w2.sum(axis=2)  # (nbl, ntime)
    cross = np.einsum("pt,btf,qf->bpq", basis_t, w2, basis_f, optimize=True)
    blocks = (np.einsum("bf,pf,qf->bpq", wf, basis_f, basis_f) + np.einsum("bt,pt,qt->bpq", wt, basis_t, basis_t)
              + cross + cross.transpose(0, 2, 1))
    h4 = np.zeros((nslot + 1, nparam, nslot + 1, nparam), dtype=np.float64)
    np.add.at(h4, (slot1, slice(None), slot1), blocks)
    np.add.at(h4, (slot2, slice(None), slot2), blocks)
    np.add.at(h4, (slot1, slice(None), slot2), -blocks)
    np.add.at(h4, (slot2, slice(None), slot1), -blocks)
    n = nslot * nparam
    return h4[:nslot, :, :nslot, :].reshape(n, n)


def _cost_and_gradient(d, wz, basis_f, basis_t, sumw2):
    """Cost and per-baseline J^T r of the separable phase model, evaluated without per-sample trigonometry.

    exp(i m) with m = -(sum_p d_p g_p) factorises as ef(bl, f) et(bl, t), so sum w2 exp(i (m - a)) and its
    derivatives reduce to one batched matmul over the sample cube ``wz = w2 exp(-i a)``.

    Returns
    -------
    tuple (cost, g_bl)
        cost = sum w2 (1 - cos(m - a)); g_bl (nbl, nparam) = sum w2 sin(m - a) g_p.
    """
    ef = np.exp(-1j * (d @ basis_f))  # (nbl, nchan)
    et = np.exp(-1j * (d @ basis_t))  # (nbl, ntime)
    ef_cols = np.concatenate([ef[:, :, None], ef[:, :, None] * basis_f.T[None]], axis=2)  # (nbl, nchan, 1+nparam)
    proj = wz @ ef_cols  # (nbl, ntime, 1+nparam)
    q = np.einsum("bt,btk->bk", et, proj)  # (nbl, 1+nparam)
    g_bl = q[:, 1:].imag + np.einsum("bt,pt,bt->bp", et, basis_t, proj[:, :, 0]).imag
    cost = sumw2 - float(q[:, 0].real.sum())
    return cost, g_bl


def _weighted_unit_conj(vis, w2):
    """w2 * exp(-i arg V) as complex128 (nbl, ntime, nchan); samples with V = 0 get arg 0 (as np.angle does)."""
    v = np.asarray(vis, dtype=np.complex128)
    amp = np.abs(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(amp > 0, np.conj(v) / amp, 1.0 + 0j)
    return unit * w2


def _lm_fit(x0, wz, w2, basis_f, basis_t, slot1, slot2, nslot, max_iter):
    """Levenberg-Marquardt fit of weighted unit-vector residuals using the exact normal equations.

    Residuals per sample: w (cos m - cos a), w (sin m - sin a) with m = -(sum_p d_p g_p), d = x[slot2] - x[slot1]
    and g_p(t, f) = basis_f[p](f) + basis_t[p](t). For such residuals J^T J = sum w^2 g g^T is constant and
    J^T r = sum w^2 g sin(m - a), g = dm/dx.

    Parameters
    ----------
    x0 : float array (nslot, nparam)
        Start values in scaled units (slot = non-reference antenna index).
    wz : complex array (nbl, ntime, nchan)
        w^2 exp(-i arg V) of each sample (zero where flagged).
    w2 : float array (nbl, ntime, nchan)
        WEIGHT_SPECTRUM (w^2), zero where flagged.
    basis_f : float array (nparam, nchan)
        Frequency-only part of the scaled basis functions (1, wDf, 0, k_disp)[active] / scale.
    basis_t : float array (nparam, ntime)
        Time-only part of the scaled basis functions (0, 0, wDt, 0)[active] / scale.
    slot1, slot2 : int arrays (nbl,)
        Slot of antenna1 / antenna2 of each baseline; nslot denotes the reference antenna (fixed at zero).
    nslot : int
    max_iter : int

    Returns
    -------
    tuple (x, cost, n_iter, covariance)
        x (nslot, nparam) scaled parameters; cost = 0.5 sum r^2; covariance (nslot*nparam, nslot*nparam) scaled by
        the residual variance 2 cost / (2 nsamp - nparam_total).
    """
    nparam = basis_f.shape[0]
    hess = _normal_matrix(w2, basis_f, basis_t, slot1, slot2, nslot)
    n = nslot * nparam
    diag = np.diag(hess).copy()
    diag[diag <= 0] = 1.0
    sumw2 = float(w2.sum())

    def evaluate(x):
        xa = np.vstack([x, np.zeros((1, nparam))])
        cost, g_bl = _cost_and_gradient(xa[slot2] - xa[slot1], wz, basis_f, basis_t, sumw2)
        g = np.zeros((nslot + 1, nparam), dtype=np.float64)
        np.add.at(g, slot1, g_bl)
        np.add.at(g, slot2, -g_bl)
        return cost, g[:nslot].reshape(-1)

    x = np.array(x0, dtype=np.float64)
    cost, g = evaluate(x)
    lam, n_iter = 1e-3, 0
    for n_iter in range(1, max_iter + 1):
        accepted = False
        for _ in range(12):
            a = hess + lam * np.diag(diag)
            try:
                delta = np.linalg.solve(a, -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(a, -g, rcond=None)[0]
            x_new = x + delta.reshape(nslot, nparam)
            cost_new, g_new = evaluate(x_new)
            if cost_new <= cost:
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        improvement = cost - cost_new
        x, cost, g = x_new, cost_new, g_new
        lam = max(lam / 10.0, 1e-12)
        if np.linalg.norm(delta) <= 1e-10 * (np.linalg.norm(x) + 1e-10) or improvement <= 1e-12 * max(cost, 1e-300):
            break
    nsamp = int(np.count_nonzero(w2))
    dof = max(2 * nsamp - n, 1)
    cov = np.linalg.pinv(hess) * (2.0 * cost / dof)
    return x, cost, n_iter, cov


def fringe_global_solve(data, refant, start, *, active=(True, True, False), minsnr=0.0, max_iter=100):
    """Global least-squares fringe fit on all baselines (CASA expb_f / Levenberg-Marquardt stage).

    Parameters
    ----------
    data : FringeData
    refant : int
    start : dict
        Output of fringe_fft_search (phase, delay_ns, rate, snr, ok used as start values / selection).
    active : (bool, bool, bool)
        Solve for (delay, rate, dispersive) in addition to phase.
    minsnr : float
        Antennas with FFT snr below this are excluded from the solve and flagged.
    max_iter : int
        Maximum LM iterations per polarisation.

    Returns
    -------
    dict
        Arrays (nant, npol): phi0 [rad] (referenced to first channel / first integration), delay_ns, rate, disp, snr
        (copied from the FFT stage), flag; paramerr (nant, npol, 4); n_iter, cost (npol,); f_grid0_hz, t_grid0,
        f_rate_hz (frequency multiplying the rate term).
    """
    t_start = _time.perf_counter()
    nant, npol = data.nant, data.npol
    kinds = [0] + [i + 1 for i, flag in enumerate(active) if flag]
    nparam = len(kinds)
    f_grid0, t_grid0, f_rate = float(data.freq[0]), float(data.time[0]), data.f_ref_hz
    fmin_c, fmax_c = data.spw_freq_limits()
    # Every basis function depends on frequency only or on time only: g_p(t, f) = basis_f[p](f) + basis_t[p](t).
    zf, zt = np.zeros(data.nchan, dtype=np.float64), np.zeros(data.ntime, dtype=np.float64)
    full_basis_f = np.stack([np.ones(data.nchan), TWO_PI * (data.freq - f_grid0) * 1e-9, zf,
                             k_disp(data.freq, fmin_c, fmax_c)])
    full_basis_t = np.stack([zt, zt, TWO_PI * (data.time - t_grid0) * f_rate, zt])
    basis_f, basis_t = full_basis_f[kinds], full_basis_t[kinds]
    scale = np.maximum(np.abs(basis_f).max(axis=1), np.abs(basis_t).max(axis=1))
    scale[scale <= 0] = 1.0
    basis_f, basis_t = basis_f / scale[:, None], basis_t / scale[:, None]
    params = np.zeros((nant, npol, 4), dtype=np.float64)
    paramerr = np.zeros((nant, npol, 4), dtype=np.float64)
    flag = np.ones((nant, npol), dtype=bool)
    n_iter = np.zeros(npol, dtype=np.int64)
    cost = np.zeros(npol, dtype=np.float64)
    start_params = np.stack([start["phase"], start["delay_ns"], start["rate"], np.zeros((nant, npol))], axis=-1)
    for p in range(npol):
        good = start["ok"][:, p] & (start["snr"][:, p] >= minsnr)
        good[refant] = True
        bl_sel = np.nonzero(good[data.antenna1] & good[data.antenna2])[0]
        if bl_sel.size == 0:
            log.warning("fringe_global_solve: pol %d has no usable baselines", p)
            continue
        a1, a2 = data.antenna1[bl_sel], data.antenna2[bl_sel]
        solved = np.setdiff1d(np.union1d(a1, a2), [refant])
        nslot = solved.size
        slot_of = np.full(nant, nslot, dtype=np.int64)
        slot_of[solved] = np.arange(nslot)
        w2 = np.where(data.flag[bl_sel, :, :, p], 0.0, data.weight[bl_sel, :, :, p]).astype(np.float64)
        wz = _weighted_unit_conj(data.vis[bl_sel, :, :, p], w2)
        x0 = start_params[solved, p][:, kinds] * scale[None, :]
        x, cost[p], n_iter[p], cov = _lm_fit(x0, wz, w2, basis_f, basis_t, slot_of[a1], slot_of[a2], nslot, max_iter)
        err = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(nslot, nparam)
        for i, kind in enumerate(kinds):
            params[solved, p, kind] = x[:, i] / scale[i]
            paramerr[solved, p, kind] = err[:, i] / scale[i]
        flag[solved, p] = False
        log.info("fringe_global_solve: pol %d nant=%d nbl=%d nparam=%d iter=%d cost=%.6g", p, nslot, bl_sel.size, nparam,
                 n_iter[p], cost[p])
    flag[refant] = False
    log.info("fringe_global_solve: done in %.3f s", _time.perf_counter() - t_start)
    return dict(phi0=params[..., 0], delay_ns=params[..., 1], rate=params[..., 2], disp=params[..., 3],
                paramerr=paramerr, flag=flag, snr=np.array(start["snr"], dtype=np.float64), n_iter=n_iter, cost=cost,
                f_grid0_hz=f_grid0, t_grid0=t_grid0, f_rate_hz=f_rate)


# ----------------------------------------------------------------------------------------------------------------
# Re-referencing and driver
# ----------------------------------------------------------------------------------------------------------------
def rereference(phases, delays_ns, rates, data, f_ref_hz, t_sol, *, zerorates=False, time_centroid=None):
    """Move phi0 from the grid origin (first channel, first integration) to (f_ref, t_sol) as CASA does.

    Parameters
    ----------
    phases, delays_ns, rates : arrays (nant, npol)
    data : FringeData
        Provides the grid origin (freq[0], time[0]) and the per-antenna time centroids for zerorates.
    f_ref_hz, t_sol : float
    zerorates : bool
        Reference each antenna phase to its weighted time centroid instead of t_sol and zero the rates.
    time_centroid : array (nant,), optional
        Per-antenna time centroid [s]; computed from data when None and zerorates is True.

    Returns
    -------
    tuple (phases, rates, time_centroid)
        Wrapped phases (nant, npol), rates (zeroed if requested), time centroid (nant,) (t_sol if not zerorates).
    """
    phases, delays_ns, rates = (np.asarray(a, dtype=np.float64) for a in (phases, delays_ns, rates))
    df0 = f_ref_hz - float(data.freq[0])
    if zerorates:
        centroid = antenna_time_centroid(data) if time_centroid is None else np.asarray(time_centroid, np.float64)
    else:
        centroid = np.full(phases.shape[0], float(t_sol), dtype=np.float64)
    dt0 = centroid - float(data.time[0])
    new_phase = phases + TWO_PI * (df0 * delays_ns * 1e-9 + f_ref_hz * dt0[:, None] * rates)
    new_rates = np.zeros_like(rates) if zerorates else rates.copy()
    return wrap_phase(new_phase), new_rates, centroid


@dataclass
class FringeSolution:
    """Fringe-fit solution of one interval, indexed by antenna id (shape (nant, npol) unless noted)."""

    antenna: np.ndarray
    refant: int
    phase: np.ndarray
    delay_ns: np.ndarray
    rate: np.ndarray
    disp: np.ndarray
    snr: np.ndarray
    flag: np.ndarray
    paramerr: np.ndarray  # (nant, npol, 4)
    f_ref_hz: float
    t_sol: float
    time_centroid: np.ndarray  # (nant,)
    n_iter: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    cost: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timings: dict = field(default_factory=dict)

    @property
    def npol(self):
        """Number of solved polarisations."""
        return self.phase.shape[1]

    def to_fparam(self):
        """Return the CASA FPARAM layout: dict(fparam, flag, snr, paramerr) with arrays (nant, 8).

        Slots 4p + (0, 1, 2, 3) hold (phi0, tau_ns, rate, disp) of pol p; a single polarisation is copied into both
        halves. The reference antenna row is unflagged with zero parameters and SNR = 999.
        """
        nant = self.antenna.size
        per_pol = np.stack([self.phase, self.delay_ns, self.rate, self.disp], axis=-1)  # (nant, npol, 4)
        if self.npol == 1:
            per_pol = np.repeat(per_pol, 2, axis=1)
            flag = np.repeat(self.flag, 2, axis=1)
            snr = np.repeat(self.snr, 2, axis=1)
            err = np.repeat(self.paramerr, 2, axis=1)
        else:
            flag, snr, err = self.flag, self.snr, self.paramerr
        fparam = per_pol.reshape(nant, 8).astype(np.float32)
        flag8 = np.repeat(flag, 4, axis=1).reshape(nant, 8)
        snr8 = np.repeat(snr, 4, axis=1).reshape(nant, 8).astype(np.float32)
        err8 = err.reshape(nant, 8).astype(np.float32)
        fparam[flag8] = 0.0
        err8[flag8] = 0.0
        fparam[self.refant] = 0.0
        flag8[self.refant] = False
        snr8[self.refant] = REFANT_SNR_SENTINEL
        err8[self.refant] = 0.0
        return dict(fparam=fparam, flag=flag8, snr=snr8, paramerr=err8)


def fringefit_interval(data, refant, *, active=(True, True, False), minsnr=5.0, pad=4, zerorates=False,
                       global_solve=True, max_iter=100, t_sol=None, delay_window_ns=None, rate_window=None):
    """Fringe fit one solution interval: FFT search -> SNR threshold -> global least squares -> re-reference.

    Parameters
    ----------
    data : FringeData
    refant : int
    active : (bool, bool, bool)
        Solve for (delay, rate, dispersive) in the least-squares stage.
    minsnr : float
        Antennas with FFT snr below minsnr are flagged before the least squares.
    pad : int
        FFT zero-padding factor.
    zerorates : bool
        Reference phases to each antenna's time centroid and zero the rates.
    global_solve : bool
        Run the least-squares stage (False keeps the FFT estimates).
    max_iter : int
    t_sol : float, optional
        Solution time [s]; default the centre of the interval.
    delay_window_ns, rate_window : (float, float), optional
        FFT search windows.

    Returns
    -------
    FringeSolution
    """
    t0 = _time.perf_counter()
    fft = fringe_fft_search(data, refant, pad=pad, delay_window_ns=delay_window_ns, rate_window=rate_window)
    t1 = _time.perf_counter()
    nant, npol = data.nant, data.npol
    if global_solve:
        gs = fringe_global_solve(data, refant, fft, active=active, minsnr=minsnr, max_iter=max_iter)
        phase, delay, rate, disp = gs["phi0"], gs["delay_ns"], gs["rate"], gs["disp"]
        flag, paramerr, n_iter, cost = gs["flag"], gs["paramerr"], gs["n_iter"], gs["cost"]
    else:
        flag = ~(fft["ok"] & (fft["snr"] >= minsnr))
        flag[refant] = False
        phase, delay = np.where(flag, 0.0, fft["phase"]), np.where(flag, 0.0, fft["delay_ns"])
        rate, disp = np.where(flag, 0.0, fft["rate"]), np.zeros((nant, npol))
        paramerr, n_iter, cost = np.zeros((nant, npol, 4)), np.zeros(npol, dtype=np.int64), np.zeros(npol)
    if not active[0]:
        delay = np.zeros_like(delay)
    if not active[1]:
        rate = np.zeros_like(rate)
    t2 = _time.perf_counter()
    f_ref = data.f_ref_hz
    t_sol = data.t_sol if t_sol is None else float(t_sol)
    phase, rate, centroid = rereference(phase, delay, rate, data, f_ref, t_sol, zerorates=zerorates)
    phase[flag], delay[flag], rate[flag], disp[flag] = 0.0, 0.0, 0.0, 0.0
    phase[refant], delay[refant], rate[refant], disp[refant] = 0.0, 0.0, 0.0, 0.0
    snr = np.array(fft["snr"], dtype=np.float64)
    snr[refant] = REFANT_SNR_SENTINEL
    timings = dict(fft=t1 - t0, lsq=t2 - t1, total=_time.perf_counter() - t0)
    log.info("fringefit_interval: refant=%d flagged=%d/%d fft=%.3fs lsq=%.3fs", refant, int(flag.sum()), flag.size,
             timings["fft"], timings["lsq"])
    return FringeSolution(antenna=np.arange(nant), refant=int(refant), phase=phase, delay_ns=delay, rate=rate, disp=disp,
                          snr=snr, flag=flag, paramerr=paramerr, f_ref_hz=f_ref, t_sol=t_sol, time_centroid=centroid,
                          n_iter=np.asarray(n_iter), cost=np.asarray(cost), timings=timings)
