"""Synthetic-data tests for vlbipy.solvers.fringe (CASA fringefit conventions)."""
import time

import numpy as np
import pytest

from vlbipy.solvers.fringe import (
    REFANT_SNR_SENTINEL,
    FringeData,
    fringe_fft_search,
    fringe_global_solve,
    fringefit_interval,
    predict_phase,
    wrap_phase,
)

T_START = 5.0e9  # MJD seconds


def simulate(nant=8, refant=2, npol=2, nspw=2, nchan=32, df=0.5e6, f_start=1.6e9, nint=60, dt=2.0, noise=0.7,
             with_disp=False, flag_frac=0.1, dead_ant=6, seed=0, ncorr=4):
    """Build row-ordered MS-like arrays with known antenna parameters.

    Returns (FringeData, truth dict). Truth holds params (nant, npol, 4), f_ref, t_sol, freq, time, fmin_c, fmax_c.
    """
    rng = np.random.default_rng(seed)
    chan_freq = f_start + (np.arange(nspw)[:, None] * nchan + np.arange(nchan)[None, :]) * df
    freq = chan_freq.reshape(-1)
    times = T_START + np.arange(nint) * dt
    f_ref = 0.5 * (freq.min() + freq.max())
    t_sol = 0.5 * (times.min() + times.max())
    fmin_c = np.repeat(chan_freq.min(axis=1), nchan)
    fmax_c = np.repeat(chan_freq.max(axis=1), nchan)
    params = np.zeros((nant, npol, 4))
    params[..., 0] = rng.uniform(-np.pi, np.pi, (nant, npol))
    params[..., 1] = rng.uniform(-50, 50, (nant, npol))
    params[..., 2] = rng.uniform(-1e-12, 1e-12, (nant, npol))
    if with_disp:
        params[..., 3] = rng.uniform(2e6, 4e6, (nant, npol)) * rng.choice([-1, 1], (nant, npol))
    params[refant] = 0.0
    theta = predict_phase(params, freq, times, f_ref, t_sol, fmin_c, fmax_c)  # (nant, npol, nint, nchan_all)
    a1, a2 = np.triu_indices(nant, 1)
    keep = (a1 != dead_ant) & (a2 != dead_ant)
    a1, a2 = a1[keep], a2[keep]
    nbl = a1.size
    bl_phase = theta[a1] - theta[a2]  # (nbl, npol, nint, nchan_all)
    vis_all = np.exp(1j * bl_phase) + noise * (rng.standard_normal(bl_phase.shape) + 1j * rng.standard_normal(bl_phase.shape))
    # Row layout: (time, baseline, spw) -> (nrow, nchan, ncorr).
    nrow = nint * nbl * nspw
    row_t = np.repeat(np.arange(nint), nbl * nspw)
    row_bl = np.tile(np.repeat(np.arange(nbl), nspw), nint)
    row_spw = np.tile(np.arange(nspw), nint * nbl)
    vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex64)
    pol_slots = [0] if ncorr == 1 else [0, ncorr - 1]
    for p, slot in enumerate(pol_slots):
        block = vis_all[row_bl, p, row_t]  # (nrow, nchan_all)
        vis[:, :, slot] = block.reshape(nrow, nspw, nchan)[np.arange(nrow), row_spw]
    if ncorr == 4:
        vis[:, :, 1:3] = noise * (rng.standard_normal((nrow, nchan, 2)) + 1j * rng.standard_normal((nrow, nchan, 2)))
    flag = rng.uniform(size=(nrow, nchan, ncorr)) < flag_frac
    weight = rng.uniform(0.5, 1.5, (nrow, nchan, ncorr)).astype(np.float32)
    data = FringeData.from_baselines(vis, flag, weight, a1[row_bl], a2[row_bl], times[row_t], row_spw, chan_freq,
                                     nant=nant)
    truth = dict(params=params, f_ref=f_ref, t_sol=t_sol, freq=freq, time=times, fmin_c=fmin_c, fmax_c=fmax_c,
                 refant=refant, dead_ant=dead_ant)
    return data, truth


def baseline_phase_error(sol, truth):
    """Max |wrapped difference| between predicted and true baseline phases over all unflagged baselines."""
    fit = np.stack([sol.phase, sol.delay_ns, sol.rate, sol.disp], axis=-1)
    fmin_c, fmax_c = truth["fmin_c"], truth["fmax_c"]
    theta_fit = predict_phase(fit, truth["freq"], truth["time"], sol.f_ref_hz, sol.t_sol, fmin_c, fmax_c)
    theta_true = predict_phase(truth["params"], truth["freq"], truth["time"], truth["f_ref"], truth["t_sol"], fmin_c, fmax_c)
    good = ~sol.flag.any(axis=1)
    ants = np.nonzero(good)[0]
    worst = 0.0
    for i in ants:
        for j in ants:
            if j <= i:
                continue
            diff = wrap_phase((theta_fit[i] - theta_fit[j]) - (theta_true[i] - theta_true[j]))
            worst = max(worst, float(np.abs(diff).max()))
    return worst


def test_from_baselines_layout():
    data, truth = simulate()
    assert data.vis.shape == (21, 60, 64, 2)
    assert np.all(data.antenna1 < data.antenna2)
    assert data.nant == 8
    assert np.array_equal(data.chan_offset, np.arange(64))
    assert np.all(data.weight[data.flag] == 0)
    assert data.f_ref_hz == pytest.approx(truth["f_ref"])


def test_fft_stage_recovers_within_one_bin():
    data, truth = simulate()
    fft = fringe_fft_search(data, truth["refant"])
    ok = fft["ok"]
    assert not ok[truth["dead_ant"]].any()
    assert ok.sum() == (8 - 2) * 2
    d_delay = np.abs(fft["delay_ns"] - truth["params"][..., 1])[ok]
    d_rate = np.abs(fft["rate"] - truth["params"][..., 2])[ok]
    assert d_delay.max() < fft["bin_delay_ns"]
    assert d_rate.max() < fft["bin_rate"]
    assert fft["snr"][ok].min() > 20


def test_global_solve_recovers_parameters():
    data, truth = simulate()
    sol = fringefit_interval(data, truth["refant"], minsnr=5.0)
    good = ~sol.flag
    good[truth["refant"]] = False
    assert good.sum() == (8 - 2) * 2
    d_delay = np.abs(sol.delay_ns - truth["params"][..., 1])[good]
    d_rate = np.abs(sol.rate - truth["params"][..., 2])[good]
    # Within 3 sigma (from the Jacobian covariance) or the absolute floor, whichever is larger.
    assert np.all(d_delay < np.maximum(3 * sol.paramerr[..., 1][good], 0.1))
    assert np.all(d_rate < np.maximum(3 * sol.paramerr[..., 2][good], 1e-14))
    assert d_delay.max() < 0.3
    assert d_rate.max() < 5e-14
    assert baseline_phase_error(sol, truth) < 0.05
    assert sol.flag[truth["dead_ant"]].all()
    assert sol.n_iter.max() <= 100


def test_dispersive_term_recovered():
    data, truth = simulate(with_disp=True, seed=3)
    sol = fringefit_interval(data, truth["refant"], active=(True, True, True), minsnr=5.0)
    good = ~sol.flag
    good[truth["refant"]] = False
    k_true = truth["params"][..., 3][good]
    d_disp = np.abs(sol.disp[good] - k_true)
    assert np.all(d_disp < np.maximum(5 * sol.paramerr[..., 3][good], 0.1 * np.abs(k_true)))
    d_delay = np.abs(sol.delay_ns - truth["params"][..., 1])[good]
    assert np.all(d_delay < np.maximum(3 * sol.paramerr[..., 1][good], 0.1))
    # One more free parameter per antenna: the noise-driven phase error is larger than in the 3-parameter fit.
    assert baseline_phase_error(sol, truth) < 0.1


def test_to_fparam_layout_and_refant_sentinel():
    data, truth = simulate()
    sol = fringefit_interval(data, truth["refant"])
    out = sol.to_fparam()
    for key in ("fparam", "flag", "snr", "paramerr"):
        assert out[key].shape == (8, 8)
    assert not out["flag"][truth["refant"]].any()
    assert np.all(out["snr"][truth["refant"]] == REFANT_SNR_SENTINEL)
    assert np.all(out["fparam"][truth["refant"]] == 0)
    assert out["flag"][truth["dead_ant"]].all()
    assert np.all(out["fparam"][truth["dead_ant"]] == 0)
    good = np.nonzero(~out["flag"][:, 0])[0]
    assert np.allclose(out["fparam"][good, 1], sol.delay_ns[good, 0])
    assert np.allclose(out["fparam"][good, 6], sol.rate[good, 1], rtol=1e-6)


def test_single_pol_fparam_copies_pol():
    data, truth = simulate(ncorr=1)
    assert data.npol == 1
    sol = fringefit_interval(data, truth["refant"])
    out = sol.to_fparam()
    assert np.array_equal(out["fparam"][:, :4], out["fparam"][:, 4:])
    assert np.array_equal(out["flag"][:, :4], out["flag"][:, 4:])


def test_zerorates_references_to_centroid():
    data, truth = simulate()
    sol = fringefit_interval(data, truth["refant"], zerorates=True)
    assert np.all(sol.rate == 0)
    good = np.nonzero(~sol.flag.any(axis=1))[0]
    assert np.all(np.abs(sol.time_centroid[good] - truth["t_sol"]) < 5.0)


def test_snr_closed_form_single_baseline():
    nint, nchan = 20, 16
    freq = 1.6e9 + np.arange(nchan) * 0.5e6
    times = T_START + np.arange(nint) * 2.0
    nrow = nint
    vis = np.ones((nrow, nchan, 2), dtype=np.complex64)
    flag = np.zeros((nrow, nchan, 2), dtype=bool)
    weight = np.ones((nrow, nchan, 2), dtype=np.float32)
    data = FringeData.from_baselines(vis, flag, weight, np.zeros(nrow, int), np.ones(nrow, int), times, np.zeros(nrow, int),
                                     freq[None, :])
    fft = fringe_fft_search(data, 0)
    n = nint * nchan
    expected = np.tan(0.5 * np.pi * 0.999) ** 1.163 * np.sqrt(n / np.sqrt(n / n))
    assert fft["xcount"][1, 0] == n
    assert fft["sumw"][1, 0] == pytest.approx(n)
    assert fft["snr"][1, 0] == pytest.approx(expected, rel=1e-6)
    assert fft["snr"][1, 1] == pytest.approx(expected, rel=1e-6)
    assert fft["delay_ns"][1, 0] == pytest.approx(0.0, abs=1e-6)
    assert fft["rate"][1, 0] == pytest.approx(0.0, abs=1e-20)


def test_sign_convention_both_sides_of_refant():
    # Antenna 0 (< refant) and antenna 7 (> refant) must both come out with the model sign.
    data, truth = simulate(noise=0.0, flag_frac=0.0)
    fft = fringe_fft_search(data, truth["refant"])
    gs = fringe_global_solve(data, truth["refant"], fft)
    for ant in (0, 7):
        assert np.allclose(gs["delay_ns"][ant], truth["params"][ant, :, 1], atol=1e-3)
        assert np.allclose(gs["rate"][ant], truth["params"][ant, :, 2], atol=1e-16)


def test_speed_12_antennas():
    data, truth = simulate(nant=12, refant=3, nspw=4, nchan=52, nint=150, dead_ant=-1, seed=1)
    t0 = time.perf_counter()
    sol = fringefit_interval(data, truth["refant"])
    elapsed = time.perf_counter() - t0
    print(f"\n12-antenna fringefit_interval: total {elapsed:.2f} s (fft {sol.timings['fft']:.2f} s, "
          f"lsq {sol.timings['lsq']:.2f} s, n_iter {sol.n_iter.tolist()})")
    assert elapsed < 5.0
    assert not sol.flag.any()
    good = np.ones_like(sol.flag)
    good[truth["refant"]] = False
    d_delay = np.abs(sol.delay_ns - truth["params"][..., 1])[good]
    assert np.all(d_delay < np.maximum(3 * sol.paramerr[..., 1][good], 0.1))
