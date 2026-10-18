"""Compiled loops for the per-visibility work of applying calibration (used when numba is installed).

Two steps of on-the-fly calibration touch every value, and numpy does each of them in a dozen passes over
temporary arrays:

* :func:`assemble` multiplies the factors of a table chain into the gain of every point, channel and hand
  (the fringe phase exponentiated, the flat and bandpass factors, the flags) and writes the grid;
* :func:`correct` corrects rows with the gains of their two antennas (feed rotation, division, flags,
  weights).

Here each is one loop over the values, run on several cores (``prange``) with no intermediate array. The loops
do the arithmetic of the numpy code they replace operation for operation, with numpy's formulas for the
complex product and quotient and no fast-math, so they give the same numbers (bit for bit wherever numpy does
not itself use fused multiply-adds for complex products). ``$VLBIPY_NUMBA=0`` turns them off (numpy is used,
as without numba).
"""
from __future__ import annotations

import math
import os

import numpy as np

#: Environment variable: ``0`` keeps the numpy implementation even when numba is installed.
NUMBA_ENV = "VLBIPY_NUMBA"

try:                                                     # numba is optional
    if os.environ.get(NUMBA_ENV, "").strip().lower() in ("0", "false", "no", "off"):
        raise ImportError("disabled by $VLBIPY_NUMBA")
    import numba
    from numba import njit, prange
except ImportError:
    numba = None


#: Set when a loop could not be compiled on this machine: numpy is used from then on.
_FAILED: list = []


def available() -> bool:
    """True when the compiled loops are used (numba is installed, not disabled, and compiles here)."""
    return numba is not None and not _FAILED


def _run(kernel, *args) -> bool:
    """Run a compiled loop; False (and numpy from then on) when numba cannot compile it on this machine."""
    try:
        kernel(*args)
    except numba.core.errors.NumbaError as exc:
        if not _FAILED:
            from ..logging_utils import get_logger
            get_logger().warning("numba could not compile the calibration loops ({}); using numpy",
                                 str(exc).splitlines()[0])
        _FAILED.append(exc)
        return False
    return True


def _threads() -> int:
    """Threads for a compiled loop: the job's share of the cores (:func:`vlbipy.solvers.workers.job_threads`), or
    one inside a thread of :func:`vlbipy.solvers.workers.thread_map`, which runs several loops at once."""
    from .workers import in_thread_map, job_threads
    return 1 if in_thread_map() else max(1, min(job_threads(), numba.config.NUMBA_NUM_THREADS))


if numba is not None:
    @njit(inline="always")
    def _cdiv(ar, ai, br, bi, one):
        """Complex quotient (ar + i ai) / (br + i bi) as numpy computes it (Smith's method), in the precision of
        its arguments (``one`` is 1 in that precision)."""
        abs_r, abs_i = abs(br), abs(bi)
        if abs_r >= abs_i:
            if abs_r == 0 and abs_i == 0:
                return ar / abs_r, ai / abs_r
            rat = bi / br
            scl = one / (br + bi * rat)
            return (ar + ai * rat) * scl, (ai - ar * rat) * scl
        rat = br / bi
        scl = one / (bi + br * rat)
        return (ar * rat + ai) * scl, (ai * rat - ar) * scl

    @njit(parallel=True, nogil=True, cache=True)
    def _correct(vis, flag, weight, gain, gflag, wscale, point1, point2, hand1, hand2, chans, rotation, rotate,
                 invalid, zero, one):
        # ``weight``/``wscale``, ``chans`` and ``rotation``/``rotate`` may be None: numba then compiles the
        # loop without them (the ``is None`` branches are resolved at compile time).
        nrow, nchan, ncorr = vis.shape
        for r in prange(nrow):
            p, q = point1[r], point2[r]
            turn = False
            if rotation is not None:
                turn = rotate[r]
            for k in range(ncorr):
                h1, h2 = hand1[k], hand2[k]
                rr, ri = zero, zero
                if turn:
                    rr, ri = rotation[r, k].real, rotation[r, k].imag
                spectral_weights = False
                scale = zero
                if weight is not None:
                    spectral_weights = wscale.shape[1] > 1
                    if not spectral_weights:
                        scale = wscale[p, 0, h1] * wscale[q, 0, h2]      # the weight factor without channels
                for c in range(nchan):
                    g = c
                    if chans is not None:
                        g = chans[c]
                    vr, vi = vis[r, c, k].real, vis[r, c, k].imag
                    bad = flag[r, c, k]
                    # importfitsidi leaves NaN where the correlator gave zero weight: flagged, and zeroed.
                    if invalid and not (math.isfinite(vr) and math.isfinite(vi)):
                        vr, vi = zero, zero
                        bad = True
                    if turn:
                        vr, vi = vr * rr - vi * ri, vr * ri + vi * rr
                    # gain of antenna 1 times the conjugate gain of antenna 2, then the quotient
                    ar, ai = gain[p, g, h1].real, gain[p, g, h1].imag
                    cr, ci = gain[q, g, h2].real, -gain[q, g, h2].imag
                    xr, xi = _cdiv(vr, vi, ar * cr - ai * ci, ar * ci + ai * cr, one)
                    vis[r, c, k] = complex(xr, xi)
                    flag[r, c, k] = bad | gflag[p, g, h1] | gflag[q, g, h2]
                    if weight is not None:
                        if spectral_weights:
                            weight[r, c, k] *= wscale[p, g, h1] * wscale[q, g, h2]
                        else:
                            weight[r, c, k] *= scale

    @njit(parallel=True, nogil=True, cache=True)
    def _assemble(gain, gflag, flat, flat_flag, spw_index, phase, delay, disp, fringe_flag, offset, kdisp, spec,
                  spec_flag, spec_index):
        # The fringe arrays (``phase`` ... ``offset``), ``kdisp`` and the spectral ones may be None: numba then
        # compiles the loop without them.
        npoint, nchan, nhand = gain.shape
        for k in prange(npoint):
            s = spw_index[k]
            m = 0
            if spec is not None:
                m = spec_index[k]
            for h in range(nhand):
                base_r, base_i = flat[k, h].real, flat[k, h].imag
                base_flag = flat_flag[k, h]
                if phase is not None:
                    base_flag = base_flag | fringe_flag[k, h]
                for c in range(nchan):
                    vr, vi = base_r, base_i
                    bad = base_flag
                    if phase is not None:
                        # exp(i theta) times the flat factors (numpy: np.exp(1j * theta) * flat)
                        theta = delay[k, h] * offset[s, c] + phase[k, h]
                        if kdisp is not None:
                            theta = theta + disp[k, h] * kdisp[s, c]
                        er, ei = math.cos(theta), math.sin(theta)
                        vr, vi = er * base_r - ei * base_i, er * base_i + ei * base_r
                    if spec is not None:
                        sr, si = spec[m, h, c].real, spec[m, h, c].imag
                        vr, vi = vr * sr - vi * si, vr * si + vi * sr
                        bad = bad | spec_flag[m, h, c]
                    # A gain that is zero or not finite cannot calibrate anything: flagged.
                    if not (math.isfinite(vr) and math.isfinite(vi)) or (vr == 0 and vi == 0):
                        bad = True
                    if bad:
                        gain[k, c, h] = 1.0
                    else:
                        gain[k, c, h] = complex(vr, vi)
                    gflag[k, c, h] = bad


def correct(vis, flag, weight, gain, gflag, wscale, point1, point2, hand1, hand2, *, chans=None, rotation=None,
            rotate=None, invalid=False) -> bool:
    """Correct rows in place: zero and flag values that are not finite (``invalid``), rotate the feeds, divide by
    the gains, flag, scale the weights.

    Row ``r`` is corrected with the gains of the points ``point1[r]`` and ``point2[r]`` (its two antennas) of the
    grid arrays ``gain``/``gflag`` (npoint, nchan_grid, 2) and ``wscale`` (npoint, 1 or nchan_grid, 2, or None),
    on the grid channels ``chans`` (all when None); ``hand1``/``hand2`` are the hands of every correlation.
    ``rotation`` (nrow, ncorr) multiplies the rows ``rotate`` (bool, nrow) before the division (the feed
    rotation). ``weight`` (or None) is scaled by the two weight factors. False when the loop cannot be compiled
    here (nothing was changed: the caller uses numpy).
    """
    weighted = weight is not None and wscale is not None
    real = np.float32 if vis.dtype == np.complex64 else np.float64
    rotated = rotation is not None and rotate is not None
    numba.set_num_threads(_threads())
    return _run(_correct, vis, flag, weight if weighted else None, gain, gflag, wscale if weighted else None,
                np.ascontiguousarray(point1, dtype=np.int64), np.ascontiguousarray(point2, dtype=np.int64),
                np.ascontiguousarray(hand1, dtype=np.int64), np.ascontiguousarray(hand2, dtype=np.int64),
                None if chans is None else np.ascontiguousarray(chans, dtype=np.int64),
                np.ascontiguousarray(rotation, dtype=vis.dtype) if rotated else None,
                np.ascontiguousarray(rotate, dtype=bool) if rotated else None, bool(invalid), real(0), real(1))


def assemble(gain, gflag, flat, flat_flag, spw_index, *, fringe=None, offset=None, kdisp=None,
             spectral=None) -> bool:
    """Write the gains of points from their factors into ``gain`` and ``gflag`` (npoint, nchan, 2).

    ``flat`` (npoint, 2) is the product of the factors without a frequency axis and ``flat_flag`` their flags.
    ``fringe`` = (phase, delay, disp, flags), each (npoint, 2), with ``offset`` (nspw, nchan): 2 pi 1e-9 (f - f0)
    of the channels of each spw (``spw_index``, (npoint,)) and ``kdisp`` (nspw, nchan) or None (no dispersive
    term). ``spectral`` = (values, flags, index): the product of the bandpass factors, (nvalue, 2, nchan), and the
    row of every point in them. A gain that is flagged, zero or not finite becomes 1 and is flagged. False when
    the loop cannot be compiled here (the caller uses numpy).
    """
    numba.set_num_threads(_threads())
    phase, delay, disp, fringe_flag = (None,) * 4 if fringe is None else (
        np.ascontiguousarray(value) for value in fringe)
    spec, spec_flag, spec_index = (None,) * 3 if spectral is None else (
        np.ascontiguousarray(spectral[0], dtype=np.complex128), np.ascontiguousarray(spectral[1], dtype=bool),
        np.ascontiguousarray(spectral[2], dtype=np.int64))
    return _run(_assemble, gain, gflag, np.ascontiguousarray(flat, dtype=np.complex128),
                np.ascontiguousarray(flat_flag), np.ascontiguousarray(spw_index, dtype=np.int64), phase, delay, disp,
                fringe_flag, None if fringe is None else np.ascontiguousarray(offset, dtype=np.float64),
                None if fringe is None or kdisp is None else np.ascontiguousarray(kdisp, dtype=np.float64),
                spec, spec_flag, spec_index)
