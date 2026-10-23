"""End-to-end tests of the dask-ms backend's solvers on a small synthetic measurement set.

The measurement set is simulated from known antenna gains (a bandpass and a delay per antenna and
subband), so every task has a ground truth: the worker pool and the MS reader, the bandpass and gain
solvers, the fringe fitter and the calibration application.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

from vlbipy.solvers import caltable, msio, workers  # noqa: E402
from vlbipy.solvers.apply_task import apply_to_ms  # noqa: E402
from vlbipy.solvers.fringefit_task import run_fringefit  # noqa: E402
from vlbipy.solvers.gain_task import (_solve_shape_and_continuum, fill_channel_gaps, normalise_bandpass,  # noqa: E402
                                      reference_gains, run_bandpass, run_gaincal, solve_antenna_gains)

NANT, NSPW, NCHAN, NCORR, NTIME = 5, 2, 16, 4, 24
CHAN_FREQ = np.array([[1.60e9 + 1e6 * c for c in range(NCHAN)], [1.70e9 + 1e6 * c for c in range(NCHAN)]])
T0, DT = 5.0e9, 2.0
SIGMA = 0.05
#: Receptor of antenna 1 / antenna 2 for the correlations RR, RL, LR, LL.
HAND1, HAND2 = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])


def _scalar(value_type):
    """Scalar column description."""
    return {"valueType": value_type, "dataManagerType": "StandardStMan", "dataManagerGroup": "SSM", "option": 0,
            "maxlen": 0, "comment": "", "keywords": {}}


def _array(value_type, shape=None, ndim=1):
    """Array column description (fixed shape when ``shape`` is given)."""
    desc = _scalar(value_type)
    desc["ndim"] = len(shape) if shape is not None else ndim
    if shape is not None:
        desc["option"], desc["shape"] = 5, np.array(shape)
    return desc


def _create(path, desc, nrow, columns):
    """Create a table and fill ``columns`` (row-major arrays; transposed to casatools' Fortran order)."""
    tb = casatools.table()
    tb.create(str(path), desc)
    tb.addrows(nrow)
    for name, values in columns.items():
        values = np.asarray(values)
        tb.putcol(name, np.asfortranarray(values.T) if values.ndim > 1 else values)
    tb.close()


def true_gains(rng) -> np.ndarray:
    """Antenna gains (nant, nspw, nchan, 2): a smooth bandpass times a delay and phase per antenna/subband/hand."""
    channel = np.arange(NCHAN)
    amplitude = 1.0 + 0.2 * rng.uniform(-1, 1, (NANT, NSPW, 1, 2)) + 0.1 * np.cos(
        channel[None, None, :, None] / 3.0 + rng.uniform(0, 6, (NANT, NSPW, 1, 2)))
    delay_ns = rng.uniform(-15, 15, (NANT, NSPW, 1, 2))
    phase = rng.uniform(-3, 3, (NANT, NSPW, 1, 2)) + 2 * np.pi * delay_ns * 1e-9 * (
        CHAN_FREQ - CHAN_FREQ[:, :1])[None, :, :, None]
    gains = amplitude * np.exp(1j * phase)
    gains[0] = np.abs(gains[0])                                   # antenna 0 is the reference: zero phase
    return gains


@pytest.fixture
def sim(tmp_path):
    """A synthetic MS (2 scans x 2 subbands, 5 antennas, unit point source) and the gains it was corrupted with."""
    rng = np.random.default_rng(11)
    gains = true_gains(rng)
    ms = tmp_path / "sim.ms"
    pairs = [(i, j) for i in range(NANT) for j in range(i + 1, NANT)]
    rows = [(scan, spw, t, i, j) for scan in (1, 2) for spw in range(NSPW) for t in range(NTIME) for i, j in pairs]
    scan, spw, step, ant1, ant2 = (np.array(column) for column in zip(*rows))
    time = T0 + DT * step + 600.0 * (scan - 1)
    truth = gains[ant1, spw][:, :, HAND1] * np.conj(gains[ant2, spw][:, :, HAND2])            # (nrow, nchan, ncorr)
    noise = SIGMA * (rng.standard_normal(truth.shape) + 1j * rng.standard_normal(truth.shape))
    data = (truth + noise).astype(np.complex64)
    nrow = len(rows)
    main = {"TIME": _scalar("double"), "ANTENNA1": _scalar("int"), "ANTENNA2": _scalar("int"),
            "FIELD_ID": _scalar("int"), "SCAN_NUMBER": _scalar("int"), "DATA_DESC_ID": _scalar("int"),
            "FLAG_ROW": _scalar("boolean"), "DATA": _array("complex", [NCORR, NCHAN]),
            "FLAG": _array("boolean", [NCORR, NCHAN]), "WEIGHT": _array("float", [NCORR]),
            "SIGMA": _array("float", [NCORR])}
    _create(ms, main, nrow, {
        "TIME": time, "ANTENNA1": ant1.astype(np.int32), "ANTENNA2": ant2.astype(np.int32),
        "FIELD_ID": np.zeros(nrow, dtype=np.int32), "SCAN_NUMBER": scan.astype(np.int32),
        "DATA_DESC_ID": spw.astype(np.int32), "FLAG_ROW": np.zeros(nrow, dtype=bool), "DATA": data,
        "FLAG": np.zeros(data.shape, dtype=bool), "WEIGHT": np.full((nrow, NCORR), 7.0, dtype=np.float32),
        "SIGMA": np.full((nrow, NCORR), SIGMA, dtype=np.float32)})
    # Subtables go inside the main table directory, so they are written after it exists.
    names = np.array([f"A{i}" for i in range(NANT)])
    _create(ms / "ANTENNA", {"NAME": _scalar("string"), "STATION": _scalar("string"), "MOUNT": _scalar("string"),
                             "POSITION": _array("double", [3]), "DISH_DIAMETER": _scalar("double"),
                             "FLAG_ROW": _scalar("boolean")}, NANT,
            {"NAME": names, "STATION": names, "MOUNT": np.array(["EQUATORIAL"] * NANT),
             "POSITION": 6.371e6 * np.array([[np.cos(0.1 * i), np.sin(0.1 * i), 0.5] for i in range(NANT)]),
             "DISH_DIAMETER": np.full(NANT, 25.0), "FLAG_ROW": np.zeros(NANT, dtype=bool)})
    _create(ms / "FIELD", {"NAME": _scalar("string"), "PHASE_DIR": _array("double", [2, 1])}, 1,
            {"NAME": np.array(["CAL"]), "PHASE_DIR": np.array([[[0.5, 0.6]]])})
    _create(ms / "OBSERVATION", {"TELESCOPE_NAME": _scalar("string"), "TIME_RANGE": _array("double", [2])}, 1,
            {"TELESCOPE_NAME": np.array(["SIM"]), "TIME_RANGE": np.array([[T0, T0 + 3600.0]])})
    spw_desc = caltable.spectral_window_table_desc()
    width = np.full((NSPW, NCHAN), 1e6)
    _create(ms / "SPECTRAL_WINDOW", spw_desc, NSPW, {
        "CHAN_FREQ": CHAN_FREQ, "CHAN_WIDTH": width, "EFFECTIVE_BW": width, "RESOLUTION": width,
        "FLAG_ROW": np.zeros(NSPW, dtype=bool), "FREQ_GROUP": np.zeros(NSPW, dtype=np.int32),
        "FREQ_GROUP_NAME": np.array([""] * NSPW), "IF_CONV_CHAIN": np.zeros(NSPW, dtype=np.int32),
        "MEAS_FREQ_REF": np.full(NSPW, 5, dtype=np.int32), "NAME": np.array(["IF0", "IF1"]),
        "NET_SIDEBAND": np.ones(NSPW, dtype=np.int32), "NUM_CHAN": np.full(NSPW, NCHAN, dtype=np.int32),
        "REF_FREQUENCY": CHAN_FREQ[:, 0], "TOTAL_BANDWIDTH": np.full(NSPW, NCHAN * 1e6)})
    _create(ms / "DATA_DESCRIPTION", {"SPECTRAL_WINDOW_ID": _scalar("int"), "POLARIZATION_ID": _scalar("int")},
            NSPW, {"SPECTRAL_WINDOW_ID": np.arange(NSPW, dtype=np.int32),
                   "POLARIZATION_ID": np.zeros(NSPW, dtype=np.int32)})
    _create(ms / "POLARIZATION", {"NUM_CORR": _scalar("int")}, 1, {"NUM_CORR": np.array([NCORR], dtype=np.int32)})

    return {"ms": ms, "gains": gains, "ant1": ant1, "ant2": ant2, "spw": spw, "truth": truth, "data": data,
            "scan": scan, "time": time}


# ----------------------------------------------------------------------------------------------------------------
# Worker pool and measurement-set reader
# ----------------------------------------------------------------------------------------------------------------
def test_worker_pool_runs_jobs_in_order_and_reports_failures():
    jobs = [{"index": i, "label": f"job{i}"} for i in range(7)]
    assert workers.run_jobs("builtins:dict", jobs, workers=2) == jobs          # results come back in job order
    assert workers.run_jobs("builtins:dict", jobs[:1], workers=2) == jobs[:1]    # a single job runs in-process
    with pytest.raises(workers.WorkerError, match="JSONDecodeError"):
        workers.run_jobs("json:loads", [{"s": "{"}, {"s": "[1]"}], workers=2)
    with pytest.raises(ValueError):
        workers.resolve("no_colon_here")
    workers.close_pool()


def test_layout_and_sliced_reads_agree_between_engines(sim):
    layout = msio.read_layout(sim["ms"])
    assert layout["nrows"] == sim["data"].shape[0] and len(layout["parts"]) == 1
    assert sorted(zip(layout["scan"].tolist(), layout["ddid"].tolist())) == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert layout["nrow"].sum() == layout["nrows"] and "DATA" in layout["columns"]
    runs = msio.select_runs(layout, scans={2}, ddids={1})
    assert runs.size == 1 and layout["scan"][runs[0]] == 2
    specs = msio.run_specs(layout, runs)
    chans = np.arange(3, 11)
    rows = np.flatnonzero((sim["scan"] == 2) & (sim["spw"] == 1))
    for engine in ("casacore", "casatools") if msio.table_engine() == "casacore" else ("casatools",):
        block = msio.read_runs(specs, ["DATA", "FLAG", "TIME", "SIGMA"], engine=engine, chans_by_ddid={1: chans},
                               corrs=np.array([0, 3]), nchan=NCHAN, ncorr=NCORR)
        assert block["DATA"].shape == (rows.size, chans.size, 2) and block["DATA"].dtype == np.complex64
        np.testing.assert_array_equal(block["DATA"], sim["data"][rows][:, chans][:, :, [0, 3]])
        np.testing.assert_array_equal(block["TIME"], sim["time"][rows])
        assert block["SIGMA"].shape == (rows.size, 2) and set(block["DATA_DESC_ID"]) == {1}
    # a time range narrower than the run reads only the rows inside it
    t_lo, t_hi = sim["time"][rows].min() + 3 * DT, sim["time"][rows].min() + 6 * DT
    inside = msio.read_runs(specs, ["TIME"], engine=msio.table_engine(), timerange=(t_lo, t_hi))
    assert inside["TIME"].min() == t_lo and inside["TIME"].max() == t_hi
    pieces = msio.split_rows(msio.run_specs(layout, range(len(layout["start"]))), 100)
    assert sum(spec["nrow"] for chunk in pieces for spec in chunk) == layout["nrows"]
    assert max(sum(spec["nrow"] for spec in chunk) for chunk in pieces) <= 100


# ----------------------------------------------------------------------------------------------------------------
# Gain solver
# ----------------------------------------------------------------------------------------------------------------
def _baseline_sums(gains, sparse=()):
    """Noise-free baseline sums (nant, nant, nchan, npol) for antenna gains (nant, nchan, npol), unit weights."""
    nant = gains.shape[0]
    cross = np.zeros((nant, nant) + gains.shape[1:], dtype=np.complex128)
    wsum = np.zeros(cross.shape, dtype=np.float64)
    for i in range(nant):
        for j in range(i + 1, nant):
            if (i, j) in sparse:
                continue
            cross[i, j], wsum[i, j] = gains[i] * np.conj(gains[j]), 1.0
    return cross, wsum


def test_gain_solver_recovers_gains_in_every_mode():
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.6, 1.5, (6, 4, 2)) * np.exp(1j * rng.uniform(-3, 3, (6, 4, 2)))
    cross, wsum = _baseline_sums(truth, sparse={(1, 4), (2, 5)})
    gains, flags, snr = solve_antenna_gains(cross, wsum, mode="ap", minblperant=3)
    gains, refant = reference_gains(gains, flags, [9, 2, 0])                     # antenna 9 does not exist
    expected = truth * np.exp(-1j * np.angle(truth[2]))[None]
    assert refant == 2 and not flags.any() and np.all(snr > 0)
    np.testing.assert_allclose(gains, expected, atol=1e-6)
    amplitude, flags, _ = solve_antenna_gains(cross, wsum, mode="a", minblperant=3)
    np.testing.assert_allclose(amplitude, np.abs(truth), atol=1e-6)
    phase, flags, _ = solve_antenna_gains(cross, wsum, mode="p", minblperant=3)
    phase, _ = reference_gains(phase, flags, [2])
    np.testing.assert_allclose(phase, expected / np.abs(expected), atol=1e-6)


def test_gain_solver_solves_disconnected_subarrays_independently():
    """Two sub-arrays without a common baseline, one 100x weaker (staggered global VLBI, scans combined): both
    are solved, and the weak one's signal-to-noise does not depend on how well the strong one fits."""
    rng = np.random.default_rng(5)
    truth = rng.uniform(0.8, 1.2, (8, 3, 1)) * np.exp(1j * rng.uniform(-3, 3, (8, 3, 1)))
    truth[4:] *= 0.1
    cross, wsum = _baseline_sums(truth, sparse={(i, j) for i in range(4) for j in range(4, 8)})
    noise = 1e-3 * (rng.normal(size=cross.shape) + 1j * rng.normal(size=cross.shape))
    gains, flags, snr = solve_antenna_gains(cross + noise * (wsum > 0), wsum, minsnr=3.0, minblperant=3)
    assert not flags.any()
    for group in ([0, 1, 2, 3], [4, 5, 6, 7]):
        model = gains[group][:, None] * np.conj(gains[group][None, :])
        expected = truth[group][:, None] * np.conj(truth[group][None, :])
        off = ~np.eye(4, dtype=bool)
        np.testing.assert_allclose(model[off], expected[off], atol=5e-3)
    # A badly fitting strong sub-array (one corrupted baseline) leaves the weak one's signal-to-noise alone.
    corrupted = cross.copy()
    corrupted[0, 1] *= 3.0
    _, _, snr_bad = solve_antenna_gains(corrupted + noise * (wsum > 0), wsum, minsnr=3.0, minblperant=3)
    np.testing.assert_allclose(snr_bad[4:], snr[4:], rtol=1e-6)


def test_divide_by_continuum_removes_source_structure():
    """A resolved source (baseline-dependent visibility) times an antenna bandpass: after dividing every baseline
    by its channel average the antenna solve recovers the band shape and fits it without misfit."""
    rng = np.random.default_rng(7)
    nant, nchan = 5, 16
    band = (1 + 0.2 * rng.normal(size=(nant, nchan, 1))) * np.exp(0.3j * rng.normal(size=(nant, nchan, 1)))
    cross, wsum = _baseline_sums(band)
    structure = rng.uniform(0.02, 1.0, (nant, nant)) * np.exp(1j * rng.uniform(-3, 3, (nant, nant)))
    cross = cross * structure[:, :, None, None]
    raw_gains, raw_flags, raw_snr = solve_antenna_gains(cross, wsum, minsnr=0.0)
    gains, flags, snr = _solve_shape_and_continuum([cross], [wsum], minsnr=0.0, minblperant=4)
    assert not flags.any()
    # Each channel has its own phase origin: reference both to antenna 0, then compare the shapes up to a
    # per-antenna complex scale.
    gains, _ = reference_gains(gains, flags, [0])
    expected = band * np.exp(-1j * np.angle(band[0]))[None]
    ratio = gains / expected
    np.testing.assert_allclose(ratio / ratio[:, :1], 1.0, atol=5e-3)
    assert np.median(snr) > 10 * np.median(raw_snr)


def test_gain_solver_flags_poorly_connected_antennas_and_low_snr():
    truth = np.ones((5, 1, 1), dtype=np.complex128)
    cross, wsum = _baseline_sums(truth, sparse={(0, 4), (1, 4), (2, 4)})          # antenna 4 keeps one baseline
    _, flags, _ = solve_antenna_gains(cross, wsum, minblperant=3)
    assert flags[4].all() and not flags[:4].any()
    noisy = cross + 0.3 * (np.random.default_rng(1).standard_normal(cross.shape) + 0j) * (wsum > 0)
    _, _, snr = solve_antenna_gains(noisy, wsum * 100.0, minblperant=3)
    _, flags, _ = solve_antenna_gains(noisy, wsum * 100.0, minblperant=3, minsnr=float(snr[:4].max()) + 1.0)
    assert flags.all()                                                            # nothing reaches the threshold


def test_bandpass_gap_filling_and_normalisation():
    gains = np.ones((1, 10, 1), dtype=np.complex128) * np.linspace(1.0, 1.9, 10)[None, :, None]
    flags = np.zeros(gains.shape, dtype=bool)
    flags[0, [0, 4, 5]] = True
    gains[0, [4, 5]] = 99.0
    fill_channel_gaps(gains, flags, max_gap=2)
    assert not flags[0, 4:6].any() and flags[0, 0, 0]                              # the band-edge gap stays flagged
    np.testing.assert_allclose(np.abs(gains[0, 4:6, 0]), [1.4, 1.5])
    normalise_bandpass(gains, flags)
    assert np.sqrt(np.mean(np.abs(gains[0, 1:, 0]) ** 2)) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------------------------
# Tasks on the synthetic measurement set
# ----------------------------------------------------------------------------------------------------------------
def _read_table(path):
    """Return (CPARAM/FPARAM, FLAG, SNR, ANTENNA1, SPECTRAL_WINDOW_ID, ANTENNA2) of a caltable, row-major."""
    tb = casatools.table()
    tb.open(str(path))
    name = "CPARAM" if "CPARAM" in tb.colnames() else "FPARAM"
    out = tuple(np.asarray(tb.getcol(c)).T for c in (name, "FLAG", "SNR", "ANTENNA1", "SPECTRAL_WINDOW_ID", "ANTENNA2"))
    tb.close()
    return out


def test_run_bandpass_recovers_the_bandpass(sim, tmp_path):
    # CASA's estimator (a point-source fit per channel) and the default (band shape x continuum gain).
    for divide, tolerance in ((False, 0.02), (True, 0.03)):
        table = run_bandpass(str(sim["ms"]), str(tmp_path / "sim.B"), field="CAL", solint="inf", combine="scan",
                             refant="A0", minsnr=3.0, solnorm=False, workers=2, divide_continuum=divide)
        param, flag, snr, ant, spw, ref = _read_table(table)
        assert param.shape == (NANT * NSPW, NCHAN, 2) and not flag.any() and np.all(ref == 0) and np.all(snr > 3)
        np.testing.assert_allclose(param, sim["gains"][ant, spw], atol=tolerance)
    normalised = run_bandpass(str(sim["ms"]), str(tmp_path / "norm.B"), field="CAL", solint="inf", combine="scan",
                              refant="A0", solnorm=True, spw="0:2~13", workers=1)
    param, flag, _, ant, spw, _ = _read_table(normalised)
    assert set(spw) == {0} and flag[:, [0, 1, 14, 15]].all() and not flag[:, 2:14].any()
    np.testing.assert_allclose(np.sqrt(np.mean(np.abs(param[:, 2:14]) ** 2, axis=1)), 1.0, atol=1e-6)


def test_run_gaincal_solves_amplitudes_per_interval(sim, tmp_path):
    bandpass = run_bandpass(str(sim["ms"]), str(tmp_path / "sim.B"), field="CAL", solint="inf", combine="scan",
                            refant="A0", solnorm=True, workers=1)
    table = run_gaincal(str(sim["ms"]), str(tmp_path / "sim.G"), field="CAL", solint="inf", combine="", calmode="a",
                        refant="A0", gaintable=[str(bandpass)], interp=["nearest,nearest"], workers=2)
    param, flag, _, ant, spw, _ = _read_table(table)
    assert param.shape == (2 * NANT * NSPW, 1, 2) and not flag.any()               # one solution per scan and subband
    level = np.sqrt(np.mean(np.abs(sim["gains"]) ** 2, axis=2))                    # what solnorm took out of the bandpass
    np.testing.assert_allclose(np.abs(param[:, 0]), level[ant, spw], rtol=0.02)
    assert np.all(param.imag == 0)
    tb = casatools.table()
    tb.open(str(table / "SPECTRAL_WINDOW"))
    assert list(tb.getcol("NUM_CHAN")) == [1, 1]                                    # CASA cannot read it otherwise
    tb.close()
    with pytest.raises(NotImplementedError):
        run_gaincal(str(sim["ms"]), str(tmp_path / "bad.G"), field="CAL", gaintype="K")


def test_run_fringefit_recovers_delays(sim, tmp_path):
    table = run_fringefit(str(sim["ms"]), str(tmp_path / "sim.sbd"), field="CAL", scan="1", solint="inf",
                          refant="A0", minsnr=5.0, zerorates=True, workers=2)
    param, flag, snr, ant, spw, ref = _read_table(table)
    assert param.shape == (NANT * NSPW, 1, 8) and not flag.any() and np.all(ref == 0)
    centre = CHAN_FREQ.mean(axis=1)
    for hand in range(2):
        truth = sim["gains"][ant, spw][:, :, hand]                                 # (nrow, nchan)
        frequency = CHAN_FREQ[spw]
        model = np.exp(1j * (param[:, 0, 4 * hand, None] + 2 * np.pi * param[:, 0, 4 * hand + 1, None] * 1e-9 *
                             (frequency - centre[spw][:, None])))
        residual = np.angle(np.mean(truth / np.abs(truth) * np.conj(model), axis=1))
        assert np.max(np.abs(residual)) < 0.05                                      # rad, after removing phase + delay


def test_apply_to_ms_writes_corrected_data_flags_and_weights(sim, tmp_path):
    bandpass = run_bandpass(str(sim["ms"]), str(tmp_path / "sim.B"), field="CAL", solint="inf", combine="scan",
                            refant="A0", solnorm=False, workers=1)
    gain = run_gaincal(str(sim["ms"]), str(tmp_path / "sim.G"), field="CAL", solint="inf", combine="scan",
                       calmode="a", refant="A0", gaintable=[str(bandpass)], interp=["nearest,nearest"], workers=1)
    tb = casatools.table()
    tb.open(str(bandpass), nomodify=False)                                         # flag antenna 4, subband 1
    rows = [r for r in range(tb.nrows()) if tb.getcell("ANTENNA1", r) == 4 and tb.getcell("SPECTRAL_WINDOW_ID", r) == 1]
    for row in rows:
        tb.putcell("FLAG", row, np.ones((2, NCHAN), dtype=bool))
    tb.close()
    entries = [{"path": str(bandpass), "interp": "nearest,nearest", "calwt": True},
               {"path": str(gain), "interp": "nearest", "calwt": True}]
    stats = apply_to_ms(sim["ms"], {0: entries}, parang=False, workers=2, chunk_rows=97)
    assert stats["rows"] == sim["data"].shape[0] and stats["flagged_after"] > stats["flagged_before"] == 0
    tb.open(str(sim["ms"]))
    corrected, flag, weight = (np.asarray(tb.getcol(c)).T for c in ("CORRECTED_DATA", "FLAG", "WEIGHT"))
    tb.close()
    lost = ((sim["ant1"] == 4) | (sim["ant2"] == 4)) & (sim["spw"] == 1)
    assert flag[lost].all() and not flag[~lost].any()
    # The unit point source comes back on the parallel hands (noise 0.05 per sample).
    parallel = corrected[~lost][:, :, [0, 3]]
    assert abs(np.mean(parallel) - 1.0) < 0.01 and np.std(parallel) < 0.15
    # Weights are rebuilt from SIGMA (not from the WEIGHT column, which held 7) and scaled by the gain table only.
    gparam, _, _, gant, gspw, _ = _read_table(gain)
    level = {(a, s): np.abs(gparam[i, 0]) for i, (a, s) in enumerate(zip(gant, gspw))}
    row = int(np.flatnonzero(~lost)[0])
    a1, a2, s = int(sim["ant1"][row]), int(sim["ant2"][row]), int(sim["spw"][row])
    expected = (level[(a1, s)][HAND1] * level[(a2, s)][HAND2]) ** 2 / SIGMA ** 2
    np.testing.assert_allclose(weight[row], expected, rtol=1e-4)
    # A second application gives the same weights: they are not calibrated twice.
    apply_to_ms(sim["ms"], {0: entries}, parang=False, workers=1)
    tb.open(str(sim["ms"]))
    np.testing.assert_allclose(np.asarray(tb.getcol("WEIGHT")).T[row], expected, rtol=1e-4)
    tb.close()
    with pytest.raises(NotImplementedError):
        apply_to_ms(sim["ms"], {0: entries}, applymode="trial")
    workers.close_pool()


def test_reference_gains_references_each_sub_array_on_its_own_antenna():
    # Antennas 0-2 and 3-5 never share a baseline (two sub-arrays combined in one solve interval).
    rng = np.random.default_rng(7)
    truth = rng.uniform(0.6, 1.5, (6, 4, 2)) * np.exp(1j * rng.uniform(-3, 3, (6, 4, 2)))
    cross, wsum = _baseline_sums(truth, sparse={(i, j) for i in range(3) for j in range(3, 6)})
    gains, flags, _ = solve_antenna_gains(cross, wsum, mode="ap", minblperant=2)
    gains, refant = reference_gains(gains, flags, [1, 4], wsum=wsum)
    assert refant == 1 and not flags.any()
    np.testing.assert_allclose(gains[:3], truth[:3] * np.exp(-1j * np.angle(truth[1]))[None], atol=1e-6)
    np.testing.assert_allclose(gains[3:], truth[3:] * np.exp(-1j * np.angle(truth[4]))[None], atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------
# Throughput paths: batched gain grids, shared tables in the workers, statwt, split, light curve
# ----------------------------------------------------------------------------------------------------------------
def _columns(ms, *names):
    tb = casatools.table()
    tb.open(str(ms))
    try:
        return [np.asarray(tb.getcol(name)).T for name in names]
    finally:
        tb.close()


def test_apply_in_batches_matches_one_batch(sim, tmp_path, monkeypatch):
    from vlbipy.solvers import apply_task
    fringe = run_fringefit(str(sim["ms"]), str(tmp_path / "sim.K"), field="CAL", solint="20s", refant="A0",
                           minsnr=3.0, workers=1)
    entries = [{"path": str(fringe), "interp": "linear", "calwt": True}]
    apply_to_ms(sim["ms"], {0: entries}, parang=False, workers=1)
    whole = _columns(sim["ms"], "CORRECTED_DATA", "FLAG", "WEIGHT")
    monkeypatch.setattr(apply_task, "BATCH_ROWS", 37)          # many batches, runs split across them
    apply_to_ms(sim["ms"], {0: entries}, parang=False, workers=1, chunk_rows=11)
    for before, after in zip(whole, _columns(sim["ms"], "CORRECTED_DATA", "FLAG", "WEIGHT")):
        np.testing.assert_array_equal(before, after)


def test_gain_grid_on_a_union_of_blocks_equals_each_block_alone(sim, tmp_path):
    from vlbipy.solvers.apply import apply_gain_grid, apply_tables, gain_grid, normalise_entries
    fringe = run_fringefit(str(sim["ms"]), str(tmp_path / "sim.K"), field="CAL", solint="20s", refant="A0",
                           minsnr=3.0, workers=1)
    entries = normalise_entries([{"path": str(fringe), "interp": "linear"}])
    rows = np.flatnonzero(sim["spw"] == 1)
    data = sim["data"][rows]
    a1, a2, t = sim["ant1"][rows], sim["ant2"][rows], sim["time"][rows]
    grid = gain_grid(entries, np.concatenate([a1, a2]), t, 1, CHAN_FREQ[1])
    half = rows.size // 2
    for block in (slice(0, half), slice(half, None)):
        alone, flag_alone, _ = apply_tables(data[block], np.zeros(data[block].shape, bool), None, a1[block],
                                            a2[block], t[block], np.ones(t[block].size, int), CHAN_FREQ, entries)
        vis, flag = data[block].copy(), np.zeros(data[block].shape, bool)
        apply_gain_grid(vis, flag, None, a1[block], a2[block], t[block], grid)
        np.testing.assert_array_equal(vis, alone)
        np.testing.assert_array_equal(flag, flag_alone)
    with pytest.raises(ValueError, match="outside"):
        apply_gain_grid(data[:1].copy(), np.zeros(data[:1].shape, bool), None, a1[:1], a2[:1], t[:1] + 0.5, grid)


def test_caltable_rows_are_indexed_once_and_found_like_a_search_of_each_antenna(sim, tmp_path):
    from vlbipy.solvers.apply import _bracket, _brackets, _lookup, load_caltable
    table = load_caltable(run_fringefit(str(sim["ms"]), str(tmp_path / "sim.K"), field="CAL", solint="20s",
                                        refant="A0", minsnr=3.0, workers=1))
    table.field_id[::3] = 1                                       # two fields, interleaved
    rng = np.random.default_rng(3)
    times = np.concatenate([table.time[rng.integers(0, table.nrow, 20)],
                            rng.uniform(table.time.min() - 60, table.time.max() + 60, 40)])
    for fields in (None, [0], [1], [0, 1]):
        look = _lookup(table, fields)
        assert _lookup(table, fields) is look
        for ant in range(NANT):
            for spw in range(NSPW):
                mask = (table.antenna1 == ant) & (table.spw_id == spw)
                if fields is not None:
                    mask &= np.isin(table.field_id, fields)
                expected = np.flatnonzero(mask)
                expected = expected[np.argsort(table.time[expected], kind="stable")]
                group = ant * look["nspw"] + spw
                np.testing.assert_array_equal(look["rows"][look["bounds"][group]:look["bounds"][group + 1]],
                                              expected)
                for interp in ("nearest", "linear"):
                    i0, i1, w, found = _brackets(look, np.full(times.size, ant), np.full(times.size, spw), times,
                                                 interp)
                    assert (found == bool(expected.size)).all()
                    if expected.size:
                        lo, hi, weight = _bracket(table.time[expected], times, interp)
                        np.testing.assert_array_equal(look["rows"][i0], expected[lo])
                        np.testing.assert_array_equal(look["rows"][i1], expected[hi])
                        np.testing.assert_array_equal(w, weight)


def test_workers_receive_each_shared_table_once(sim, tmp_path):
    from vlbipy.solvers.apply import antenna_gains, load_caltable
    table = load_caltable(run_fringefit(str(sim["ms"]), str(tmp_path / "sim.K"), field="CAL", solint="20s",
                                        refant="A0", minsnr=3.0, workers=1))
    other = load_caltable(table.path)
    other.param = other.param.copy()
    other.param[:, :, 1] += 1.0                                   # same path, different content
    times = sim["time"][:5]
    jobs = [{"table": tab, "antenna_ids": [1, 2], "times": times, "spw": 0, "chan_freq": CHAN_FREQ[0]}
            for tab in (table, other) * 4]
    sent = []
    original = workers._share

    def spy(value, digests, held, outgoing, depth=0):
        result = original(value, digests, held, outgoing, depth)
        if depth == 0:
            sent.extend(outgoing)
        return result
    workers._share = spy
    try:
        first = workers.run_jobs("vlbipy.solvers.apply:antenna_gains", jobs, workers=2)
        second = workers.run_jobs("vlbipy.solvers.apply:antenna_gains", jobs[:2] * 3, workers=2)
    finally:
        workers._share = original
    for job, (gains, flags) in zip(jobs + jobs[:2] * 3, first + second):
        expected, expected_flags = antenna_gains(**job)
        np.testing.assert_array_equal(gains, expected)
        np.testing.assert_array_equal(flags, expected_flags)
    digests = {workers._digest(table), workers._digest(other)}
    assert len(digests) == 2 and set(sent) == digests
    # 14 jobs refer to the two tables, but each travels at most once to each of the two workers.
    assert all(sent.count(digest) <= 2 for digest in digests)
    workers.close_pool()


def test_statwt_weights_rows_like_casa(sim):
    from vlbipy.solvers.statwt_task import row_weights, statwt_ms
    tb = casatools.table()
    tb.open(str(sim["ms"]), nomodify=False)
    tb.addcols({"CORRECTED_DATA": _array("complex", [NCORR, NCHAN])})
    tb.putcol("CORRECTED_DATA", np.asfortranarray(sim["data"].T))
    flag = np.zeros(sim["data"].shape, dtype=bool)
    flag[0, 1:, :] = True                                         # row 0: one channel left -> starved
    flag[1, :5, 0] = True
    tb.putcol("FLAG", np.asfortranarray(flag.T))
    tb.close()
    report = statwt_ms(sim["ms"], column="CORRECTED_DATA", minsamp=2, workers=2)
    weight, new_flag, flag_row = _columns(sim["ms"], "WEIGHT", "FLAG", "FLAG_ROW")
    values = sim["data"][1, 5:, 0]
    expected = 2.0 / (np.var(values.real) + np.var(values.imag))
    np.testing.assert_allclose(weight[1, 0], expected, rtol=1e-5)
    assert (weight[0] == 0).all() and new_flag[0].all() and flag_row[0] and not flag_row[1:].any()
    assert report["flagged_after"] > report["flagged_before"]
    w, starved = row_weights(sim["data"][:3], flag[:3], 2)
    assert starved[0].all() and not starved[1:].any()
    np.testing.assert_allclose(w[1, 0], expected, rtol=1e-5)
    np.testing.assert_allclose(weight, row_weights(sim["data"], new_flag, 2)[0], rtol=1e-6)
    workers.close_pool()


def _split_reference(ms, column, chanbin, timebin):
    """The mstransform averaging rules, written out loop by loop."""
    data, flag, weight, time, interval, exposure, centroid, uvw, scan, ddid, a1, a2 = _columns(
        ms, column, "FLAG", "WEIGHT", "TIME", "INTERVAL", "EXPOSURE", "TIME_CENTROID", "UVW", "SCAN_NUMBER",
        "DATA_DESC_ID", "ANTENNA1", "ANTENNA2")
    ws = np.broadcast_to(weight[:, None, :], data.shape).astype(np.float64)
    starts = {s: time[scan == s].min() for s in np.unique(scan)}
    bins = {}
    for r in range(time.size):
        key = (scan[r], ddid[r], a1[r], a2[r], int(np.floor((time[r] - starts[scan[r]] + 1e-6) / timebin)))
        bins.setdefault(key, []).append(r)
    out = {}
    for key, rows in bins.items():
        rows = np.array(rows)
        ok = ~flag[rows]
        row_ok = ok.any(axis=(1, 2))
        last = time[rows][row_ok].max() if row_ok.any() else np.inf
        upto = time[rows] <= last
        nout = int(np.ceil(data.shape[1] / chanbin))
        vis = np.zeros((nout, data.shape[2]), complex)
        fl = np.zeros((nout, data.shape[2]), bool)
        for c in range(nout):
            sel = slice(c * chanbin, (c + 1) * chanbin)
            w_ok = np.where(ok[:, sel], ws[rows, sel], 0)
            if ok[:, sel].any(axis=(0, 1)).all():
                vis[c] = (w_ok * data[rows, sel]).sum(axis=(0, 1)) / w_ok.sum(axis=(0, 1))
            fl[c] = ~ok[:, sel].any(axis=(0, 1))
        rw = np.where(row_ok, weight[rows].sum(1), 0.0)
        out[key] = {"TIME": time[rows][upto].mean(), "INTERVAL": interval[rows][upto].sum(),
                    "EXPOSURE": exposure[rows][row_ok].sum() if row_ok.any() else exposure[rows].sum(),
                    "TIME_CENTROID": (rw * centroid[rows]).sum() / rw.sum() if rw.sum() else centroid[rows].mean(),
                    "UVW": (rw[:, None] * uvw[rows]).sum(0) / rw.sum() if rw.sum() else uvw[rows].mean(0),
                    "DATA": vis, "FLAG": fl}
    return out


def test_split_field_averages_like_mstransform(sim, tmp_path):
    from vlbipy.solvers.split_task import split_field
    pytest.importorskip("casacore.tables")
    tb = casatools.table()
    tb.open(str(sim["ms"]), nomodify=False)
    nrow = tb.nrows()
    tb.addcols({"CORRECTED_DATA": _array("complex", [NCORR, NCHAN]), "UVW": _array("double", [3]),
                "INTERVAL": _scalar("double"), "EXPOSURE": _scalar("double"), "TIME_CENTROID": _scalar("double"),
                "ARRAY_ID": _scalar("int"), "FEED1": _scalar("int"), "FEED2": _scalar("int"),
                "OBSERVATION_ID": _scalar("int"), "PROCESSOR_ID": _scalar("int"), "STATE_ID": _scalar("int")})
    rng = np.random.default_rng(3)
    tb.putcol("CORRECTED_DATA", np.asfortranarray(sim["data"].T))
    flag = rng.uniform(size=sim["data"].shape) < 0.3
    flag[::7] = True                                              # whole integrations flagged
    tb.putcol("FLAG", np.asfortranarray(flag.T))
    tb.putcol("WEIGHT", np.asfortranarray(rng.uniform(1, 5, (nrow, NCORR)).astype(np.float32).T))
    tb.putcol("UVW", np.asfortranarray(rng.normal(0, 1e6, (nrow, 3)).T))
    tb.putcol("INTERVAL", np.full(nrow, DT))
    tb.putcol("EXPOSURE", np.full(nrow, DT))
    tb.putcol("TIME_CENTROID", sim["time"] + 0.1)
    tb.close()
    target = tmp_path / "split.ms"
    report = split_field(sim["ms"], target, "CAL", chanbin=4, timebin=3 * DT, workers=2)
    expected = _split_reference(sim["ms"], "CORRECTED_DATA", 4, 3 * DT)
    assert report["rows_out"] == len(expected)
    got = _columns(target, "SCAN_NUMBER", "DATA_DESC_ID", "ANTENNA1", "ANTENNA2", "TIME", "INTERVAL", "EXPOSURE",
                   "TIME_CENTROID", "UVW", "DATA", "FLAG", "FIELD_ID")
    scan0 = {s: sim["time"][sim["scan"] == s].min() for s in (1, 2)}
    for s, d, i, j, t, iv, ex, tc, uvw, vis, fl, field in zip(*got):
        ref = expected[(s, d, i, j, int(np.floor((t - scan0[s] + 1e-6) / (3 * DT))))]
        assert field == 0
        np.testing.assert_allclose([t, iv, ex, tc], [ref["TIME"], ref["INTERVAL"], ref["EXPOSURE"],
                                                     ref["TIME_CENTROID"]], rtol=0, atol=1e-6)
        np.testing.assert_allclose(uvw, ref["UVW"], rtol=1e-9)
        np.testing.assert_array_equal(fl, ref["FLAG"])
        np.testing.assert_allclose(vis[~fl], ref["DATA"][~fl], rtol=1e-5)
    freq, width, nchan = (np.asarray(v) for v in _columns(target / "SPECTRAL_WINDOW", "CHAN_FREQ", "CHAN_WIDTH",
                                                          "NUM_CHAN"))
    np.testing.assert_allclose(freq, CHAN_FREQ.reshape(NSPW, -1, 4).mean(axis=2))
    np.testing.assert_allclose(width, np.full((NSPW, NCHAN // 4), 4e6))
    assert (nchan == NCHAN // 4).all()
    workers.close_pool()


def test_total_visibility_sums_every_unflagged_cross_correlation(sim):
    from vlbipy.solvers.plot_read import read_total_visibility
    result = read_total_visibility(sim["ms"], fields=["CAL"], column="DATA", corrs=[0, 3], workers=2)
    values = sim["data"][:, :, [0, 3]]
    stamps = np.unique(sim["time"])
    np.testing.assert_array_equal(result["time"], stamps)
    for stamp, total, count in zip(stamps, result["sum"], result["count"]):
        rows = sim["time"] == stamp
        np.testing.assert_allclose(total, values[rows].sum(dtype=np.complex128), rtol=1e-6)
        assert count == values[rows].size
    workers.close_pool()


def test_tsys_table_assigns_each_measurement_to_its_scan_or_the_next(sim, tmp_path):
    from vlbipy.solvers.apriori_task import write_tsys_table
    scan_start, scan_end = T0, T0 + DT * (NTIME - 1)               # scan 1; scan 2 starts 600 s later
    times = np.array([scan_start, scan_end, scan_end + 30.0, scan_start + 600.0, scan_start + 640.0])
    rows = [(t, a, s) for t in times for a in range(NANT) for s in range(NSPW)]
    t, a, s = (np.array(c) for c in zip(*rows))
    tsys = np.full((len(rows), 2), 80.0)
    tsys[0] = [-1.0, 90.0]                                        # a bad value is flagged, the other hand kept
    _create(sim["ms"] / "SYSCAL", {"TIME": _scalar("double"), "ANTENNA_ID": _scalar("int"),
                                    "SPECTRAL_WINDOW_ID": _scalar("int"), "INTERVAL": _scalar("double"),
                                    "FEED_ID": _scalar("int"), "TSYS": _array("float", [2])}, len(rows),
            {"TIME": t, "ANTENNA_ID": a.astype(np.int32), "SPECTRAL_WINDOW_ID": s.astype(np.int32),
             "INTERVAL": np.zeros(len(rows)), "FEED_ID": np.zeros(len(rows), dtype=np.int32), "TSYS": tsys})
    report = write_tsys_table(sim["ms"], tmp_path / "sim.tsys")
    assert report["rows"] == len(rows) and report["flagged"] == 1
    tb = casatools.table()
    tb.open(str(tmp_path / "sim.tsys"))
    try:
        got_t, got_scan, fparam, flag = (np.asarray(tb.getcol(c)) for c in ("TIME", "SCAN_NUMBER", "FPARAM", "FLAG"))
        assert tb.getkeyword("VisCal") == "B TSYS" and tb.getkeyword("ParType") == "Float"
    finally:
        tb.close()
    expected = {scan_start: 1, scan_end: 1, scan_end + 30.0: 2, scan_start + 600.0: 2, scan_start + 640.0: 2}
    assert all(expected[float(x)] == n for x, n in zip(got_t, got_scan))
    assert flag.sum() == 1 and fparam[0, 0, flag[0, 0]].item() == -1.0


# ----------------------------------------------------------------------------------------------------------------
# Deferred application (CORRECTED_DATA computed while it is read) and incremental applycal
# ----------------------------------------------------------------------------------------------------------------
def _add_split_columns(ms, rng):
    """The columns the split and the uv readers need, some flags, and alt-az mounts (a real feed rotation)."""
    tb = casatools.table()
    tb.open(str(ms), nomodify=False)
    nrow = tb.nrows()
    tb.addcols({"UVW": _array("double", [3]), "INTERVAL": _scalar("double"), "EXPOSURE": _scalar("double"),
                "TIME_CENTROID": _scalar("double"), "ARRAY_ID": _scalar("int"), "FEED1": _scalar("int"),
                "FEED2": _scalar("int"), "OBSERVATION_ID": _scalar("int"), "PROCESSOR_ID": _scalar("int"),
                "STATE_ID": _scalar("int")})
    time = np.asarray(tb.getcol("TIME"))
    flag = rng.uniform(size=(nrow, NCHAN, NCORR)) < 0.1
    tb.putcol("FLAG", np.asfortranarray(flag.T))
    tb.putcol("UVW", np.asfortranarray(rng.normal(0, 1e6, (nrow, 3)).T))
    tb.putcol("INTERVAL", np.full(nrow, DT))
    tb.putcol("EXPOSURE", np.full(nrow, DT))
    tb.putcol("TIME_CENTROID", time + 0.1)
    tb.close()
    tb.open(str(ms / "ANTENNA"), nomodify=False)
    tb.putcol("MOUNT", np.array(["ALT-AZ"] * NANT))
    tb.close()


def _chain(ms, tmp_path):
    """A bandpass, an amplitude gain and a fringe table (with calwt), antenna 4 of subband 1 flagged."""
    bandpass = run_bandpass(str(ms), str(tmp_path / "sim.B"), field="CAL", solint="inf", combine="scan",
                            refant="A0", solnorm=False, workers=1)
    gain = run_gaincal(str(ms), str(tmp_path / "sim.G"), field="CAL", solint="inf", combine="scan",
                       calmode="a", refant="A0", gaintable=[str(bandpass)], interp=["nearest,nearest"], workers=1)
    fringe = run_fringefit(str(ms), str(tmp_path / "sim.K"), field="CAL", solint="20s", refant="A0",
                           minsnr=3.0, gaintable=[str(bandpass)], workers=1)
    tb = casatools.table()
    tb.open(str(bandpass), nomodify=False)
    for row in range(tb.nrows()):
        if tb.getcell("ANTENNA1", row) == 4 and tb.getcell("SPECTRAL_WINDOW_ID", row) == 1:
            tb.putcell("FLAG", row, np.ones((2, NCHAN), dtype=bool))
    tb.close()
    return [{"path": str(bandpass), "interp": "nearest,linear", "calwt": True},
            {"path": str(gain), "interp": "nearest", "calwt": True},
            {"path": str(fringe), "interp": "linear", "calwt": True}]


def _assert_same(left, right):
    """Two reader results (dicts of arrays, possibly nested) are identical.

    They can be: the gains and feed angles of an integration do not depend on which other rows they are
    computed with (the feed angles are interpolated between nodes fixed in time).
    """
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            if key != "seconds":
                _assert_same(left[key], right[key])
    elif left is None:
        assert right is None
    else:
        np.testing.assert_array_equal(np.asarray(right), np.asarray(left))


def test_deferred_application_reads_exactly_like_the_written_one(sim, tmp_path):
    import shutil
    from vlbipy.solvers import corrected
    from vlbipy.solvers.apply_task import apply_to_ms, defer_to_ms
    from vlbipy.solvers.plot_read import (read_baseline_timeline, read_dynamic_spectra, read_timeseries,
                                          read_total_visibility, read_uvdistance)
    from vlbipy.solvers.split_task import split_field
    from vlbipy.solvers.statwt_task import statwt_ms
    pytest.importorskip("casacore.tables")
    _add_split_columns(sim["ms"], np.random.default_rng(5))
    entries = _chain(sim["ms"], tmp_path)
    written, deferred = tmp_path / "written.ms", tmp_path / "deferred.ms"
    shutil.copytree(sim["ms"], written)
    shutil.copytree(sim["ms"], deferred)
    apply_to_ms(written, {0: entries}, parang=True, workers=2)
    stats = defer_to_ms(deferred, {0: entries}, parang=True, workers=2)
    assert stats["pending"] == [0] and stats["flagged_after"] > stats["flagged_before"]
    assert "CORRECTED_DATA" not in _main_columns(deferred)
    # The calibration flags are the ones applycal writes, without touching a visibility.
    np.testing.assert_array_equal(*(_columns(ms, "FLAG")[0] for ms in (written, deferred)))
    # A later solve rewrites a table in place: the deferred chain keeps the version it was given.
    run_gaincal(str(sim["ms"]), entries[1]["path"], field="CAL", solint="20s", calmode="ap", refant="A0",
                workers=1)
    edges = np.linspace(sim["time"].min(), sim["time"].max() + 1, 7)
    reads = [
        lambda ms: read_dynamic_spectra(ms, fields=["CAL"], column="CORRECTED_DATA", corrs=[0, 3], edges=edges,
                                        stokes_i=False, workers=2),
        lambda ms: read_uvdistance(ms, fields=["CAL"], column="CORRECTED_DATA", corrs=[0, 3], time_bin=6.0,
                                   scan_starts={1: T0, 2: T0 + 600.0}, workers=2),
        lambda ms: read_baseline_timeline(ms, field="CAL", column="CORRECTED_DATA", corrs=[0, 3], scans=None,
                                          nant=NANT, workers=2),
        lambda ms: read_total_visibility(ms, fields=["CAL"], column="CORRECTED_DATA", corrs=[0, 3], workers=2),
        lambda ms: read_timeseries(ms, fields=["CAL"], scans=[2], column="CORRECTED_DATA", corrs=[0, 3],
                                   refant=1, edges=edges, workers=2),
    ]
    for read in reads:
        _assert_same(read(written), read(deferred))
    # statwt measures the scatter of the corrected data; afterwards its weights are the ones the split uses.
    reports = [statwt_ms(ms, column="CORRECTED_DATA", workers=2) for ms in (written, deferred)]
    assert reports[0]["flagged_after"] == reports[1]["flagged_after"] and reports[0]["mean"] == reports[1]["mean"]
    for name in ("WEIGHT", "FLAG", "FLAG_ROW"):
        _assert_same(*(_columns(ms, name)[0] for ms in (written, deferred)))
    for ms in (written, deferred):
        split_field(ms, tmp_path / f"{ms.stem}.split.ms", "CAL", chanbin=4, timebin=3 * DT, workers=2)
    names = ("DATA", "FLAG", "WEIGHT", "SIGMA", "TIME", "UVW", "TIME_CENTROID", "EXPOSURE")
    for left, right in zip(_columns(tmp_path / "written.split.ms", *names),
                           _columns(tmp_path / "deferred.split.ms", *names)):
        _assert_same(left, right)
    # Written at last, the column holds what applycal wrote (and statwt's weights stay).
    assert corrected.write_pending(deferred) == [0] and not corrected.pending(deferred)
    for name in ("CORRECTED_DATA", "FLAG", "WEIGHT"):
        _assert_same(*(_columns(ms, name)[0] for ms in (written, deferred)))
    workers.close_pool()


def _main_columns(ms):
    tb = casatools.table()
    tb.open(str(ms))
    try:
        return list(tb.colnames())
    finally:
        tb.close()


def test_a_new_deferred_chain_rebuilds_the_weights_and_an_applycal_clears_the_record(sim, tmp_path):
    import shutil
    from vlbipy.solvers import corrected
    from vlbipy.solvers.apply_task import apply_to_ms, defer_to_ms
    from vlbipy.solvers.split_task import split_field
    from vlbipy.solvers.statwt_task import statwt_ms
    pytest.importorskip("casacore.tables")
    _add_split_columns(sim["ms"], np.random.default_rng(6))
    entries = _chain(sim["ms"], tmp_path)
    written, deferred = tmp_path / "written.ms", tmp_path / "deferred.ms"
    shutil.copytree(sim["ms"], written)
    shutil.copytree(sim["ms"], deferred)
    # applycal -> statwt -> applycal again (other tables) -> split, against the same with deferred applications.
    for ms, apply in ((written, apply_to_ms), (deferred, defer_to_ms)):
        apply(ms, {0: entries}, parang=False, workers=2)
        statwt_ms(ms, column="CORRECTED_DATA", workers=2)
        apply(ms, {0: entries[:2]}, parang=False, workers=2)
        split_field(ms, tmp_path / f"{ms.stem}.split.ms", "CAL", chanbin=4, timebin=3 * DT, workers=2)
    for left, right in zip(_columns(tmp_path / "written.split.ms", "DATA", "WEIGHT", "FLAG"),
                           _columns(tmp_path / "deferred.split.ms", "DATA", "WEIGHT", "FLAG")):
        _assert_same(left, right)
    # A written application of the same chain makes the column current, and the record follows.
    apply_to_ms(deferred, {0: entries[:2]}, parang=False, workers=2)
    assert not corrected.pending(deferred)
    _assert_same(*(_columns(ms, "CORRECTED_DATA")[0] for ms in (written, deferred)))
    workers.close_pool()


def test_applycal_on_top_of_corrected_data_equals_the_whole_chain(sim, tmp_path):
    import shutil
    from vlbipy.solvers.apply_task import apply_to_ms, applycal
    entries = _chain(sim["ms"], tmp_path)
    whole, stepwise = tmp_path / "whole.ms", tmp_path / "stepwise.ms"
    shutil.copytree(sim["ms"], whole)
    shutil.copytree(sim["ms"], stepwise)
    apply_to_ms(whole, {0: entries}, parang=False, workers=2)
    first = applycal(stepwise, [e["path"] for e in entries[:2]], interp=["nearest,linear", "nearest"], workers=2)
    assert first["source"] == "DATA" and not first["parang"]      # no CORRECTED_DATA yet: from DATA
    second = applycal(stepwise, [entries[2]["path"]], interp=["linear"], workers=2)
    assert second["source"] == "CORRECTED_DATA" and not second["parang"]
    vis_whole, flag_whole, weight_whole = _columns(whole, "CORRECTED_DATA", "FLAG", "WEIGHT")
    vis_step, flag_step, weight_step = _columns(stepwise, "CORRECTED_DATA", "FLAG", "WEIGHT")
    np.testing.assert_array_equal(flag_whole, flag_step)
    ok = ~flag_whole
    np.testing.assert_allclose(vis_step[ok], vis_whole[ok], rtol=1e-5)
    np.testing.assert_allclose(weight_step, weight_whole, rtol=1e-5)
    workers.close_pool()


def test_a_table_of_another_set_is_matched_by_antenna_name(sim, tmp_path):
    from vlbipy.solvers.apply import adapt_table, load_caltable
    gain = load_caltable(run_gaincal(str(sim["ms"]), str(tmp_path / "sim.G"), field="CAL", solint="inf",
                                     combine="scan", calmode="ap", refant="A0", workers=1))
    # The same table numbered for an array listed backwards, with an extra antenna and without A0.
    order = ["A4", "XX", "A3", "A2", "A1"]
    names = list(gain.antenna_names)
    foreign = adapt_table(gain, order)
    back = adapt_table(foreign, names)
    assert list(foreign.antenna_names) == order and foreign.nant == 5
    kept = gain.antenna1 != names.index("A0")
    np.testing.assert_array_equal(back.antenna1, gain.antenna1[kept])
    np.testing.assert_array_equal(back.param, gain.param[kept])
    for row in range(foreign.nrow):
        assert order[foreign.antenna1[row]] == names[back.antenna1[row]]
    assert adapt_table(gain, names) is gain


def test_interleaved_subbands_are_corrected_like_blocked_ones(sim, tmp_path):
    """A split or an mstransform output interleaves the subbands row by row: every run of the layout is one row.
    The correction must not depend on that order (nor take a pass per row)."""
    import shutil
    from vlbipy.solvers.apply_task import apply_to_ms, defer_to_ms
    from vlbipy.solvers.split_task import split_field
    tables = pytest.importorskip("casacore.tables")
    _add_split_columns(sim["ms"], np.random.default_rng(7))
    entries = _chain(sim["ms"], tmp_path)
    interleaved = tmp_path / "interleaved.ms"
    source = tables.table(str(sim["ms"]), ack=False)
    ordered = source.sort("TIME, ANTENNA1, ANTENNA2, DATA_DESC_ID")
    ordered.copy(str(interleaved), deep=True).close()
    ordered.close()
    source.close()
    for subtable in ("ANTENNA", "FIELD", "OBSERVATION", "SPECTRAL_WINDOW", "DATA_DESCRIPTION", "POLARIZATION"):
        shutil.copytree(sim["ms"] / subtable, interleaved / subtable)
    deferred = tmp_path / "interleaved_deferred.ms"
    shutil.copytree(interleaved, deferred)
    stats = apply_to_ms(sim["ms"], {0: entries}, parang=True, workers=2)
    assert apply_to_ms(interleaved, {0: entries}, parang=True, workers=2)["rows"] == stats["rows"]
    defer_to_ms(deferred, {0: entries}, parang=True, workers=2)

    def by_row(ms, *names):
        time, a1, a2, ddid, *values = _columns(ms, "TIME", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID", *names)
        order = np.lexsort((ddid, a2, a1, time))
        return [v[order] for v in values]
    for left, right in zip(by_row(sim["ms"], "CORRECTED_DATA", "FLAG", "WEIGHT"),
                           by_row(interleaved, "CORRECTED_DATA", "FLAG", "WEIGHT")):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(*(by_row(ms, "FLAG")[0] for ms in (interleaved, deferred)))
    for ms in (interleaved, deferred):
        split_field(ms, tmp_path / f"{ms.stem}.split.ms", "CAL", chanbin=4, timebin=3 * DT, workers=2)
    for left, right in zip(_columns(tmp_path / "interleaved.split.ms", "DATA", "FLAG", "WEIGHT"),
                           _columns(tmp_path / "interleaved_deferred.split.ms", "DATA", "FLAG", "WEIGHT")):
        np.testing.assert_array_equal(left, right)
    info = tables.table(str(tmp_path / "interleaved.split.ms"), ack=False)
    assert info.info()["type"] == "Measurement Set"
    info.close()
    workers.close_pool()


def test_thread_map_keeps_the_order_raises_and_runs_nested_calls_in_line(monkeypatch):
    monkeypatch.setenv("VLBIPY_THREADS", "4")
    assert workers.thread_map(lambda x: x * x, range(10)) == [x * x for x in range(10)]

    def boom(x):
        if x == 3:
            raise ValueError("boom")
        return x
    with pytest.raises(ValueError, match="boom"):
        workers.thread_map(boom, range(6))
    nested = workers.thread_map(lambda x: (workers.in_thread_map(), workers.thread_map(lambda y: y + x, [0, 1])),
                                range(3))
    assert nested == [(True, [0, 1]), (True, [1, 2]), (True, [2, 3])]
    assert not workers.in_thread_map()
    monkeypatch.setenv("VLBIPY_THREADS", "1")
    assert workers.thread_map(lambda x: workers.in_thread_map(), range(3)) == [False] * 3


def test_pool_jobs_share_the_cores(monkeypatch):
    monkeypatch.delenv("VLBIPY_THREADS", raising=False)
    try:
        shares = workers.run_jobs("vlbipy.solvers.workers:job_threads", [{}, {}], workers=2)
        if "VLBIPY_THREADS" not in os.environ:
            assert shares == [max(1, (os.cpu_count() or 1) // 2)] * 2
        assert workers.job_threads() == max(1, os.cpu_count() or 1)          # this process is no worker
    finally:
        workers.close_pool()


def _rows_only_in_the_lock_file(path):
    """Leave the table ``path`` as CASA's import leaves the SPECTRAL_WINDOW of a set: ``table.dat`` records no
    row, and only the lock file holds the current number of rows."""
    ct = pytest.importorskip("casacore.tables")
    empty = path.with_name(path.name + ".empty")
    table = ct.table(str(path), ack=False)
    nrow = table.nrows()
    table.copy(str(empty), deep=True, valuecopy=True, copynorows=True).close()
    table.close()
    (path / "table.dat").write_bytes((empty / "table.dat").read_bytes())
    table = ct.table(str(path), ack=False)
    assert table.nrows() == nrow > 0
    table.close()
    shutil.rmtree(empty)
    return nrow


def test_frozen_tables_keep_the_rows_only_their_lock_files_hold(sim, tmp_path):
    from vlbipy.solvers import corrected
    from vlbipy.solvers.apply import load_caltable
    from vlbipy.solvers.apply_task import defer_to_ms
    table = run_fringefit(str(sim["ms"]), str(tmp_path / "sim.K"), field="CAL", solint="20s", refant="A0",
                          minsnr=3.0, workers=1)
    nspw = _rows_only_in_the_lock_file(Path(table) / "SPECTRAL_WINDOW")
    ms = tmp_path / "deferred.ms"
    shutil.copytree(sim["ms"], ms)
    defer_to_ms(ms, {0: [{"path": str(table), "interp": "linear"}]}, parang=False, workers=1)
    frozen = corrected.frozen_entries(ms, corrected.load_record(ms)["fields"][0])
    assert load_caltable(frozen[0]["path"]).spw_chan_freq.shape[0] == nspw
    assert not list((ms / corrected.RECORD_DIR).rglob("*.partial"))


def test_caltable_subtables_copied_from_a_set_record_their_rows(sim, tmp_path):
    """A caltable copies the SPECTRAL_WINDOW of the set it is solved on: written value by value, its table.dat
    holds the rows even when the set's own one does not."""
    pytest.importorskip("casacore.tables")
    from vlbipy.solvers.caltable import _copy_subtable
    nspw = _rows_only_in_the_lock_file(sim["ms"] / "SPECTRAL_WINDOW")
    copied = _copy_subtable(sim["ms"], "SPECTRAL_WINDOW", tmp_path)
    (copied / "table.lock").unlink(missing_ok=True)
    import casacore.tables as ct
    table = ct.table(str(copied), ack=False)
    assert table.nrows() == nspw
    table.close()
