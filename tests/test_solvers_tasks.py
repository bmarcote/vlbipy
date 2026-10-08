"""End-to-end tests of the dask-ms backend's solvers on a small synthetic measurement set.

The measurement set is simulated from known antenna gains (a bandpass and a delay per antenna and
subband), so every task has a ground truth: the worker pool and the MS reader, the bandpass and gain
solvers, the fringe fitter and the calibration application.
"""
from __future__ import annotations

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

from vlbipy.solvers import caltable, msio, workers  # noqa: E402
from vlbipy.solvers.apply_task import apply_to_ms  # noqa: E402
from vlbipy.solvers.fringefit_task import run_fringefit  # noqa: E402
from vlbipy.solvers.gain_task import (fill_channel_gaps, normalise_bandpass, reference_gains, run_bandpass,  # noqa: E402
                                      run_gaincal, solve_antenna_gains)

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
    table = run_bandpass(str(sim["ms"]), str(tmp_path / "sim.B"), field="CAL", solint="inf", combine="scan",
                         refant="A0", minsnr=3.0, solnorm=False, workers=2)
    param, flag, snr, ant, spw, ref = _read_table(table)
    assert param.shape == (NANT * NSPW, NCHAN, 2) and not flag.any() and np.all(ref == 0) and np.all(snr > 3)
    np.testing.assert_allclose(param, sim["gains"][ant, spw], atol=0.02)
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
