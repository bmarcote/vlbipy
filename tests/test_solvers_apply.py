"""Tests for vlbipy.solvers.apply: on-the-fly application of CASA calibration tables to visibility blocks."""

import time as _time
from pathlib import Path

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

from vlbipy.solvers import apply as ap  # noqa: E402
from vlbipy.solvers import caltable  # noqa: E402
from vlbipy.solvers.fringe import predict_phase  # noqa: E402

NANT, NFIELD, NSPW, NCHAN = 4, 2, 2, 8
MS_CHAN_FREQ = np.array([[1.5e9 + 1e6 * c for c in range(NCHAN)], [1.6e9 + 1e6 * c for c in range(NCHAN)]])
T0 = 5.0e9
REAL_MS = Path("/home/marcote/Programing/vlbipy/rsm07_manual/rsm07.ms")
REAL_SBD = Path("/tmp/vlbipy_spec/casa_ab/sbd_parang0")


# ----------------------------------------------------------------------------------------------------------------
# Fake MS / caltable builders
# ----------------------------------------------------------------------------------------------------------------
def _scalar(value_type, option=0):
    """Minimal scalar column description."""
    return {"valueType": value_type, "dataManagerType": "StandardStMan", "dataManagerGroup": "MSMTAB",
            "option": option, "maxlen": 0, "comment": "", "keywords": {}}


def _array(value_type, ndim=1, shape=None):
    """Minimal array column description."""
    desc = _scalar(value_type, option=5 if shape is not None else 0)
    desc["ndim"] = ndim
    if shape is not None:
        desc["shape"] = np.array(shape)
    return desc


def _create(path, desc, nrow, columns):
    """Create a table at `path`, add `nrow` rows and fill `columns` (casatools Fortran-ordered arrays)."""
    tb = casatools.table()
    tb.create(str(path), desc)
    tb.addrows(nrow)
    for name, values in columns.items():
        tb.putcol(name, values)
    tb.close()


@pytest.fixture
def fake_ms(tmp_path):
    """Directory with ANTENNA (4 rows), FIELD (2), OBSERVATION (1), SPECTRAL_WINDOW (2 spws x 8 chans)."""
    ms = tmp_path / "fake.ms"
    ms.mkdir()
    names = np.array(["EF", "WB", "O8", "MC"])
    ant_desc = {"NAME": _scalar("string"), "STATION": _scalar("string"), "TYPE": _scalar("string"),
                "MOUNT": _scalar("string"), "POSITION": _array("double", 1, [3]), "OFFSET": _array("double", 1, [3]),
                "DISH_DIAMETER": _scalar("double"), "FLAG_ROW": _scalar("boolean")}
    _create(ms / "ANTENNA", ant_desc, NANT, {
        "NAME": names, "STATION": names, "TYPE": np.array(["GROUND-BASED"] * NANT),
        "MOUNT": np.array(["ALT-AZ"] * NANT), "POSITION": np.arange(3 * NANT, dtype=float).reshape(NANT, 3).T,
        "OFFSET": np.zeros((3, NANT)), "DISH_DIAMETER": np.full(NANT, 25.0), "FLAG_ROW": np.zeros(NANT, dtype=bool)})
    field_desc = {"NAME": _scalar("string"), "CODE": _scalar("string"), "SOURCE_ID": _scalar("int"),
                  "NUM_POLY": _scalar("int"), "TIME": _scalar("double"), "FLAG_ROW": _scalar("boolean"),
                  "PHASE_DIR": _array("double", 2), "DELAY_DIR": _array("double", 2),
                  "REFERENCE_DIR": _array("double", 2)}
    dirs = np.array([[[0.1, 0.2]], [[0.3, 0.4]]]).T
    _create(ms / "FIELD", field_desc, NFIELD, {
        "NAME": np.array(["SRC_A", "SRC_B"]), "CODE": np.array(["", ""]), "SOURCE_ID": np.arange(NFIELD, dtype=np.int32),
        "NUM_POLY": np.zeros(NFIELD, dtype=np.int32), "TIME": np.zeros(NFIELD), "FLAG_ROW": np.zeros(NFIELD, dtype=bool),
        "PHASE_DIR": dirs, "DELAY_DIR": dirs, "REFERENCE_DIR": dirs})
    obs_desc = {"TELESCOPE_NAME": _scalar("string"), "OBSERVER": _scalar("string"), "PROJECT": _scalar("string"),
                "RELEASE_DATE": _scalar("double"), "FLAG_ROW": _scalar("boolean"),
                "TIME_RANGE": _array("double", 1, [2]), "SCHEDULE_TYPE": _scalar("string")}
    _create(ms / "OBSERVATION", obs_desc, 1, {
        "TELESCOPE_NAME": np.array(["EVN"]), "OBSERVER": np.array(["me"]), "PROJECT": np.array(["TEST"]),
        "RELEASE_DATE": np.zeros(1), "FLAG_ROW": np.zeros(1, dtype=bool), "TIME_RANGE": np.array([[T0], [T0 + 3600]]),
        "SCHEDULE_TYPE": np.array([""])})
    _create(ms / "SPECTRAL_WINDOW", caltable.spectral_window_table_desc(), NSPW, _spw_columns(MS_CHAN_FREQ))
    return ms


def _spw_columns(chan_freq):
    """Columns of a SPECTRAL_WINDOW table with channel frequencies `chan_freq` (nspw, nchan_t)."""
    nspw, nchan_t = chan_freq.shape
    width = 1e6 if nchan_t > 1 else NCHAN * 1e6
    return {"CHAN_FREQ": chan_freq.T, "CHAN_WIDTH": np.full((nchan_t, nspw), width),
            "EFFECTIVE_BW": np.full((nchan_t, nspw), width), "RESOLUTION": np.full((nchan_t, nspw), width),
            "FLAG_ROW": np.zeros(nspw, dtype=bool), "FREQ_GROUP": np.zeros(nspw, dtype=np.int32),
            "FREQ_GROUP_NAME": np.array([""] * nspw), "IF_CONV_CHAIN": np.zeros(nspw, dtype=np.int32),
            "MEAS_FREQ_REF": np.full(nspw, 5, dtype=np.int32), "NAME": np.array([f"IF{i}" for i in range(nspw)]),
            "NET_SIDEBAND": np.ones(nspw, dtype=np.int32), "NUM_CHAN": np.full(nspw, nchan_t, dtype=np.int32),
            "REF_FREQUENCY": chan_freq[:, 0], "TOTAL_BANDWIDTH": np.full(nspw, NCHAN * 1e6)}


def _write_generic_table(path, fake_ms, kind, param, flag, times, spw_ids, ant_ids, spw_chan_freq, field_ids=None,
                         flat_param=False):
    """Write a minimal CASA caltable of VisCal `kind` with CPARAM (complex param) or FPARAM (float param).

    `param`/`flag` are C-ordered (nrow, nchan_t, npar); `spw_chan_freq` (nspw, nchan_t) fills SPECTRAL_WINDOW.
    `flat_param` writes the parameter column with no channel axis (cells of shape (npar,), as CASA's gain
    curves have) while FLAG and SNR keep theirs: both columns allow a variable number of dimensions.
    """
    path = Path(path).absolute()
    nrow, nchan_t, npar = param.shape
    is_complex = np.iscomplexobj(param)
    desc = caltable.main_table_desc()
    if is_complex:
        del desc["FPARAM"]
        desc["CPARAM"] = caltable._array_column("complex")
    tb = casatools.table()
    tb.create(str(path), desc, dminfo=caltable._dminfo(list(desc)))
    tb.addrows(nrow)
    tb.putcol("TIME", np.asarray(times, dtype=np.float64))
    tb.putcol("FIELD_ID", np.zeros(nrow, dtype=np.int32) if field_ids is None else np.asarray(field_ids, np.int32))
    tb.putcol("SPECTRAL_WINDOW_ID", np.asarray(spw_ids, dtype=np.int32))
    tb.putcol("ANTENNA1", np.asarray(ant_ids, dtype=np.int32))
    tb.putcol("ANTENNA2", np.full(nrow, -1, dtype=np.int32))
    tb.putcol("INTERVAL", np.zeros(nrow))
    tb.putcol("SCAN_NUMBER", np.ones(nrow, dtype=np.int32))
    tb.putcol("OBSERVATION_ID", np.zeros(nrow, dtype=np.int32))
    values = np.ascontiguousarray(param[:, 0, :] if flat_param else param)
    tb.putcol("CPARAM" if is_complex else "FPARAM", values.T)
    tb.putcol("PARAMERR", np.zeros((nrow, nchan_t, npar), dtype=np.float32).T)
    tb.putcol("FLAG", np.ascontiguousarray(flag, dtype=bool).T)
    tb.putcol("SNR", np.full((nrow, nchan_t, npar), 10.0, dtype=np.float32).T)
    tb.putinfo({"type": "Calibration", "subType": kind, "readme": ""})
    tb.putkeyword("ParType", "Complex" if is_complex else "Float")
    tb.putkeyword("MSName", "fake.ms")
    tb.putkeyword("VisCal", kind)
    tb.putkeyword("PolBasis", "unknown")
    tb.close()
    subtables = {name: caltable._copy_subtable(fake_ms, name, path) for name in ("OBSERVATION", "ANTENNA", "FIELD")}
    spw_path = path / "SPECTRAL_WINDOW"
    _create(spw_path, caltable.spectral_window_table_desc(), spw_chan_freq.shape[0], _spw_columns(spw_chan_freq))
    subtables["SPECTRAL_WINDOW"] = spw_path
    tb.open(str(path), nomodify=False)
    for name, sub_path in subtables.items():
        tb.putkeyword(name, f"Table: {sub_path}")
    tb.close()
    return path


def _baseline_block(times, spws=(0, 1), ncorr=4):
    """All cross baselines of NANT antennas at every time and spw: returns (a1, a2, time, spw) row arrays."""
    pairs = [(i, j) for i in range(NANT) for j in range(i + 1, NANT)]
    a1, a2, tt, ss = [], [], [], []
    for s in spws:
        for t in times:
            for i, j in pairs:
                a1.append(i), a2.append(j), tt.append(t), ss.append(s)
    return np.array(a1), np.array(a2), np.array(tt, dtype=np.float64), np.array(ss)


def _vis_from_gains(gains, a1, a2, t_idx, s_idx):
    """V_ij = g_i conj(g_j) per correlation from per-antenna gains (nant, ntime, nspw, nchan, 2) complex."""
    hand1, hand2 = ap._corr_hands(4)
    g1 = gains[a1, t_idx, s_idx][:, :, hand1]
    g2 = gains[a2, t_idx, s_idx][:, :, hand2]
    return (g1 * np.conj(g2)).astype(np.complex64)


# ----------------------------------------------------------------------------------------------------------------
# Fringe Jones
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_fringe_jones_round_trip(fake_ms, tmp_path, interp):
    """Visibilities simulated from predict_phase are corrected to zero phase; a flagged table row flags its baselines."""
    rng = np.random.default_rng(3)
    t_sol = T0 + 30.0
    f_ref = MS_CHAN_FREQ.mean(axis=1)
    fparam = np.zeros((NSPW * NANT, 8), dtype=np.float32)
    flag = np.zeros((NSPW * NANT, 8), dtype=bool)
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    for r, (s, a) in enumerate(rows):
        if a == 0:
            continue
        for p in range(2):
            fparam[r, 4 * p:4 * p + 4] = [rng.uniform(-3, 3), rng.uniform(-20, 20), rng.uniform(-1, 1) * 1e-12, 0.0]
    flag[rows.index((1, 3))] = True
    path = caltable.write_fringe_table(tmp_path / "test.sbd", fake_ms, times=np.full(len(rows), t_sol),
                                       field_ids=np.zeros(len(rows), int), spw_ids=[r[0] for r in rows],
                                       antenna_ids=[r[1] for r in rows], refant_id=0,
                                       scan_numbers=np.ones(len(rows), int), fparam=fparam,
                                       paramerr=np.zeros_like(fparam), flag=flag, snr=np.full(fparam.shape, 20.0),
                                       spw_chan_freq=f_ref, spw_chan_width=np.full(NSPW, NCHAN * 1e6))
    times = t_sol + np.arange(-10, 11, 2.0)
    a1, a2, tt, ss = _baseline_block(times)
    t_idx = np.searchsorted(times, tt)
    gains = np.empty((NANT, times.size, NSPW, NCHAN, 2), dtype=np.complex128)
    for r, (s, a) in enumerate(rows):
        cf = MS_CHAN_FREQ[s]
        for p in range(2):
            ph = predict_phase(fparam[r, 4 * p:4 * p + 4], cf, times, f_ref[s], t_sol, cf.min(), cf.max())
            gains[a, :, s, :, p] = np.exp(1j * ph)
    vis = _vis_from_gains(gains, a1, a2, t_idx, ss)
    flag_in = np.zeros(vis.shape, dtype=bool)
    weight = np.ones(vis.shape, dtype=np.float32)
    before = np.abs(np.angle(vis)).max()
    assert before > 1.0
    vis_c, flag_c, w_c = ap.apply_tables(vis, flag_in, weight, a1, a2, tt, ss, MS_CHAN_FREQ,
                                         [{"path": str(path), "interp": interp, "spwmap": []}])
    has3_spw1 = ((a1 == 3) | (a2 == 3)) & (ss == 1)
    assert flag_c[has3_spw1].all()
    assert not flag_c[~has3_spw1].any()
    np.testing.assert_allclose(np.angle(vis_c[~has3_spw1]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.abs(vis_c[~has3_spw1]), 1.0, atol=1e-5)
    np.testing.assert_array_equal(w_c, weight)


def test_fringe_jones_spwmap_and_linear_time(fake_ms, tmp_path):
    """combine='spw'-style table (all spws -> spw 0) with two solution times; linear interpolation is exact between
    rows carrying the same parameters and the rate extrapolates correctly with nearest."""
    f_ref = MS_CHAN_FREQ.mean(axis=1)
    t_rows = [T0, T0 + 100.0]
    rows = [(t, a) for t in t_rows for a in range(NANT)]
    fparam = np.zeros((len(rows), 8), dtype=np.float32)
    # Both rows describe the same phase trajectory: phi0 of the second row is advanced by the rate over 100 s.
    for r, (t, a) in enumerate(rows):
        dphi = 2 * np.pi * f_ref[0] * (t - T0)
        fparam[r, 0], fparam[r, 1], fparam[r, 2] = 0.3 * a + 1e-13 * a * dphi, 2.0 * a, 1e-13 * a
        fparam[r, 4], fparam[r, 5], fparam[r, 6] = -0.2 * a - 1e-13 * a * dphi, -1.5 * a, -1e-13 * a
    path = caltable.write_fringe_table(tmp_path / "test.mbd", fake_ms, times=[r[0] for r in rows],
                                       field_ids=np.zeros(len(rows), int), spw_ids=np.zeros(len(rows), int),
                                       antenna_ids=[r[1] for r in rows], refant_id=0,
                                       scan_numbers=np.ones(len(rows), int), fparam=fparam,
                                       paramerr=np.zeros_like(fparam), flag=np.zeros(fparam.shape, bool),
                                       snr=np.full(fparam.shape, 20.0), spw_chan_freq=f_ref,
                                       spw_chan_width=np.full(NSPW, NCHAN * 1e6))
    times = T0 + np.array([10.0, 50.0, 90.0])
    a1, a2, tt, ss = _baseline_block(times)
    t_idx = np.searchsorted(times, tt)
    gains = np.empty((NANT, times.size, NSPW, NCHAN, 2), dtype=np.complex128)
    for a in range(NANT):
        for s in range(NSPW):
            cf = MS_CHAN_FREQ[s]
            for p in range(2):
                ph = predict_phase(fparam[a, 4 * p:4 * p + 4], cf, times, f_ref[0], T0, cf.min(), cf.max())
                gains[a, :, s, :, p] = np.exp(1j * ph)
    vis = _vis_from_gains(gains, a1, a2, t_idx, ss)
    for interp in ("nearest", "linear"):
        vis_c, flag_c, _ = ap.apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, tt, ss, MS_CHAN_FREQ,
                                           [{"path": str(path), "interp": interp, "spwmap": [0, 0]}])
        assert not flag_c.any()
        np.testing.assert_allclose(np.angle(vis_c), 0.0, atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------
# G Jones
# ----------------------------------------------------------------------------------------------------------------
def _g_jones_setup(fake_ms, tmp_path):
    """Write a G Jones table with 2 time stamps and distinct gains; returns (path, g0, g1) with g (nant, nspw, 2)."""
    rng = np.random.default_rng(5)
    g0 = rng.uniform(0.5, 2.0, (NANT, NSPW, 2)) * np.exp(1j * rng.uniform(-2.5, 2.5, (NANT, NSPW, 2)))
    g1 = rng.uniform(0.5, 2.0, (NANT, NSPW, 2)) * np.exp(1j * rng.uniform(-2.5, 2.5, (NANT, NSPW, 2)))
    rows = [(k, s, a) for k in range(2) for s in range(NSPW) for a in range(NANT)]
    param = np.array([(g0, g1)[k][a, s] for k, s, a in rows], dtype=np.complex64)[:, None, :]
    path = _write_generic_table(tmp_path / "test.G", fake_ms, "G Jones", param, np.zeros(param.shape, bool),
                                [T0 + 100.0 * k for k, _, _ in rows], [s for _, s, _ in rows],
                                [a for _, _, a in rows], MS_CHAN_FREQ[:, [NCHAN // 2]])
    return path, g0, g1


def test_g_jones_nearest_vs_linear_and_calwt(fake_ms, tmp_path):
    """At t0 + 25 s nearest uses the first solution; linear interpolates amplitude and phase; calwt scales weights."""
    path, g0, g1 = _g_jones_setup(fake_ms, tmp_path)
    table = ap.load_caltable(path)
    assert table.kind == "G Jones" and table.partype == "Complex" and table.param.shape == (2 * NSPW * NANT, 1, 2)
    times = np.array([T0 + 25.0])
    a1, a2, tt, ss = _baseline_block(times)
    vis = np.ones((a1.size, NCHAN, 4), dtype=np.complex64)
    weight = np.full(vis.shape, 2.0, dtype=np.float32)
    w = 0.25
    amp = (1 - w) * np.abs(g0) + w * np.abs(g1)
    phase = np.angle(g0) + w * np.angle(g1 * np.conj(g0))
    g_lin = amp * np.exp(1j * phase)
    hand1, hand2 = ap._corr_hands(4)
    for interp, g in (("nearest", g0), ("linear", g_lin)):
        vis_c, flag_c, w_c = ap.apply_tables(vis, np.zeros(vis.shape, bool), weight, a1, a2, tt, ss, MS_CHAN_FREQ,
                                             [{"path": str(path), "interp": interp, "calwt": True}])
        expected = 1.0 / (g[a1, ss][:, hand1] * np.conj(g[a2, ss][:, hand2]))
        np.testing.assert_allclose(vis_c, np.broadcast_to(expected[:, None, :], vis.shape), rtol=1e-5)
        assert not flag_c.any()
        # weights scale like 1/sigma^2: V is divided by g_i conj(g_j), so weight is multiplied by |g_i g_j|^2.
        np.testing.assert_allclose(w_c, 2.0 / np.broadcast_to(np.abs(expected[:, None, :]) ** 2, vis.shape), rtol=1e-5)
    _, _, w_same = ap.apply_tables(vis, np.zeros(vis.shape, bool), weight, a1, a2, tt, ss, MS_CHAN_FREQ,
                                   [{"path": str(path), "interp": "nearest", "calwt": False}])
    np.testing.assert_array_equal(w_same, weight)


def test_g_jones_flagged_solution_flags_the_data(fake_ms, tmp_path):
    """A flagged solution is not skipped (CASA): the times that use it are flagged, the others are not."""
    path, g0, g1 = _g_jones_setup(fake_ms, tmp_path)
    table = ap.load_caltable(path)
    first = (table.time == T0) & (table.antenna1 == 1)
    table.flag[first] = True
    # nearest: the time next to the flagged solution is flagged, the one next to the good solution is not.
    gains, gflag = ap.antenna_gains(table, [1], [T0 + 10.0, T0 + 110.0], 0, MS_CHAN_FREQ[0], interp="nearest")
    assert gflag[0, 0].all() and not gflag[0, 1].any()
    np.testing.assert_allclose(gains[0, 1, 0], g1[1, 0], rtol=1e-6)
    # linear: any time strictly between a flagged and a good solution is flagged; at the good solution it is not.
    gains, gflag = ap.antenna_gains(table, [1, 2], [T0 + 60.0, T0 + 120.0], 0, MS_CHAN_FREQ[0], interp="linear")
    assert gflag[0, 0].all() and not gflag[0, 1].any() and not gflag[1].any()


# ----------------------------------------------------------------------------------------------------------------
# B Jones / B TSYS / phase_only
# ----------------------------------------------------------------------------------------------------------------
def test_b_jones_identical_grid(fake_ms, tmp_path):
    """A bandpass on the data channel grid flattens the simulated bandpass; a flagged channel is interpolated across
    from its neighbours (CASA) unless a ``...flag`` frequency mode keeps it flagged."""
    rng = np.random.default_rng(7)
    bp = rng.uniform(0.5, 1.5, (NANT, NSPW, NCHAN, 2)) * np.exp(1j * rng.uniform(-3, 3, (NANT, NSPW, NCHAN, 2)))
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    param = np.array([bp[a, s] for s, a in rows], dtype=np.complex64)
    flag = np.zeros(param.shape, dtype=bool)
    flag[rows.index((0, 2)), 3, :] = True
    path = _write_generic_table(tmp_path / "test.B", fake_ms, "B Jones", param, flag, np.full(len(rows), T0),
                                [s for s, _ in rows], [a for _, a in rows], MS_CHAN_FREQ)
    times = T0 + np.array([0.0, 60.0])
    a1, a2, tt, ss = _baseline_block(times)
    t_idx = np.searchsorted(times, tt)
    gains = np.transpose(np.broadcast_to(bp[:, None], (NANT, times.size, NSPW, NCHAN, 2)), (0, 1, 2, 3, 4))
    vis = _vis_from_gains(gains, a1, a2, t_idx, ss)
    bad = ((a1 == 2) | (a2 == 2)) & (ss == 0)
    vis_c, flag_c, _ = ap.apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, tt, ss, MS_CHAN_FREQ,
                                       [{"path": str(path), "interp": "linear,linearflag"}])
    assert flag_c[bad, 3, :].all()
    assert not flag_c[~bad].any() and not flag_c[bad][:, [0, 1, 2, 4, 5, 6, 7]].any()
    np.testing.assert_allclose(vis_c[~flag_c], 1.0, atol=1e-5)
    vis_c, flag_c, _ = ap.apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, tt, ss, MS_CHAN_FREQ,
                                       [{"path": str(path), "interp": "linear"}])
    assert not flag_c.any()
    patched = np.ones(vis.shape, dtype=bool)
    patched[bad, 3, :] = False
    np.testing.assert_allclose(vis_c[patched], 1.0, atol=1e-5)


def test_b_jones_frequency_interpolation(fake_ms, tmp_path):
    """A bandpass sampled on a coarser grid is linearly interpolated (amp/phase) onto the data channels."""
    coarse = MS_CHAN_FREQ[:, ::2] + 0.5e6  # 4 channels centred between data channels
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    param = np.ones((len(rows), coarse.shape[1], 2), dtype=np.complex64)
    for r, (s, a) in enumerate(rows):
        param[r, :, 0] = (1.0 + 0.1 * a * np.arange(4)) * np.exp(1j * 0.1 * a * np.arange(4))
        param[r, :, 1] = param[r, :, 0]
    path = _write_generic_table(tmp_path / "test.Bc", fake_ms, "B Jones", param, np.zeros(param.shape, bool),
                                np.full(len(rows), T0), [s for s, _ in rows], [a for _, a in rows], coarse)
    gains, gflag = ap.antenna_gains(path, [2], [T0], 0, MS_CHAN_FREQ[0], interp="nearest")
    assert not gflag.any()
    pos = np.clip((MS_CHAN_FREQ[0] - coarse[0, 0]) / 2e6, 0, 3)  # fractional position on the coarse grid
    np.testing.assert_allclose(np.abs(gains[0, 0, :, 0]), 1.0 + 0.2 * pos, rtol=1e-6)
    np.testing.assert_allclose(np.angle(gains[0, 0, :, 0]), 0.2 * pos, atol=1e-6)


def _tsys_table(fake_ms, tmp_path):
    """Write a B TSYS table (constant in time) and return (path, tsys (nant, nspw, 2))."""
    rng = np.random.default_rng(11)
    tsys = rng.uniform(20.0, 200.0, (NANT, NSPW, 2))
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    param = np.array([tsys[a, s] for s, a in rows], dtype=np.float32)[:, None, :]
    return _write_generic_table(tmp_path / "test.tsys", fake_ms, "B TSYS", param, np.zeros(param.shape, bool),
                                np.full(len(rows), T0), [s for s, _ in rows], [a for _, a in rows],
                                MS_CHAN_FREQ[:, [0]]), tsys


def test_b_tsys_scaling(fake_ms, tmp_path):
    """Tsys scales unit visibilities by sqrt(Tsys_i Tsys_j) per hand and the weights by 1/(Tsys_i Tsys_j)."""
    path, tsys = _tsys_table(fake_ms, tmp_path)
    a1, a2, tt, ss = _baseline_block([T0 + 5.0])
    vis = np.ones((a1.size, NCHAN, 4), dtype=np.complex64)
    weight = np.ones(vis.shape, dtype=np.float32)
    vis_c, flag_c, w_c = ap.apply_tables(vis, np.zeros(vis.shape, bool), weight, a1, a2, tt, ss, MS_CHAN_FREQ,
                                         [{"path": str(path), "interp": "linear", "calwt": True}])
    hand1, hand2 = ap._corr_hands(4)
    expected = np.sqrt(tsys[a1, ss][:, hand1] * tsys[a2, ss][:, hand2])
    assert not flag_c.any()
    np.testing.assert_allclose(np.abs(vis_c), np.broadcast_to(expected[:, None, :], vis.shape), rtol=1e-5)
    np.testing.assert_allclose(np.angle(vis_c), 0.0, atol=1e-6)
    np.testing.assert_allclose(w_c, np.broadcast_to(1.0 / expected[:, None, :] ** 2, vis.shape), rtol=1e-5)


def test_phase_only_skips_amplitude_tables(fake_ms, tmp_path):
    """phase_only=True leaves a B TSYS table out of the chain but still applies a Fringe Jones one."""
    tsys_path, _ = _tsys_table(fake_ms, tmp_path)
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    fparam = np.zeros((len(rows), 8), dtype=np.float32)
    fparam[:, 0] = fparam[:, 4] = 0.5
    fparam[[r for r, (_, a) in enumerate(rows) if a == 0], :] = 0.0
    sbd = caltable.write_fringe_table(tmp_path / "po.sbd", fake_ms, times=np.full(len(rows), T0),
                                      field_ids=np.zeros(len(rows), int), spw_ids=[s for s, _ in rows],
                                      antenna_ids=[a for _, a in rows], refant_id=0,
                                      scan_numbers=np.ones(len(rows), int), fparam=fparam,
                                      paramerr=np.zeros_like(fparam), flag=np.zeros(fparam.shape, bool),
                                      snr=np.full(fparam.shape, 20.0), spw_chan_freq=MS_CHAN_FREQ.mean(axis=1),
                                      spw_chan_width=np.full(NSPW, NCHAN * 1e6))
    a1, a2, tt, ss = _baseline_block([T0])
    vis = np.ones((a1.size, NCHAN, 4), dtype=np.complex64)
    chain = [{"path": str(tsys_path), "interp": "nearest"}, {"path": str(sbd), "interp": "nearest"}]
    vis_c, flag_c, _ = ap.apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, tt, ss, MS_CHAN_FREQ, chain,
                                       phase_only=True)
    np.testing.assert_allclose(np.abs(vis_c), 1.0, rtol=1e-6)
    expected_phase = np.where(a1 == 0, 0.5, 0.0)[:, None, None] * np.ones((1, NCHAN, 4))
    np.testing.assert_allclose(np.angle(vis_c), expected_phase, atol=1e-6)
    vis_full, _, _ = ap.apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, tt, ss, MS_CHAN_FREQ, chain)
    assert np.abs(vis_full).min() > 20.0


def test_gain_curve_needs_elevation_unless_phase_only(fake_ms, tmp_path):
    """An EPowerCurve table needs elevations in antenna_gains (ValueError without) and is skipped with phase_only=True."""
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    param = np.ones((len(rows), 1, 2), dtype=np.float32)
    path = _write_generic_table(tmp_path / "test.gc", fake_ms, "EPowerCurve", param, np.zeros(param.shape, bool),
                                np.full(len(rows), T0), [s for s, _ in rows], [a for _, a in rows],
                                MS_CHAN_FREQ[:, [0]])
    with pytest.raises(ValueError):
        ap.antenna_gains(path, [0], [T0], 0, MS_CHAN_FREQ[0])
    a1, a2, tt, ss = _baseline_block([T0])
    vis = np.ones((a1.size, NCHAN, 4), dtype=np.complex64)
    vis_c, _, _ = ap.apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, tt, ss, MS_CHAN_FREQ,
                                  [{"path": str(path)}], phase_only=True)
    np.testing.assert_array_equal(vis_c, vis)


def test_gain_curve_with_a_channel_less_parameter_column(fake_ms, tmp_path):
    """A gain curve whose FPARAM has no channel axis while FLAG has one still reads and applies.

    Both columns allow a variable number of dimensions, so a table can mix them. Normalising all
    three columns from the parameter column's ndim made the flags 4-D and every later
    ``flag[rows, chan, par]`` raised "too many indices".
    """
    rows = [(s, a) for s in range(NSPW) for a in range(NANT)]
    ncoef = 4
    param = np.zeros((len(rows), 1, 2 * ncoef), dtype=np.float32)
    param[:, 0, 0] = param[:, 0, ncoef] = 2.0                     # constant power of 2 in both polarizations
    path = _write_generic_table(tmp_path / "flat.gc", fake_ms, "EPowerCurve", param, np.zeros(param.shape, bool),
                                np.full(len(rows), T0), [s for s, _ in rows], [a for _, a in rows],
                                MS_CHAN_FREQ[:, [0]], flat_param=True)
    table = ap.load_caltable(path)
    assert table.param.shape == table.flag.shape == table.snr.shape == (len(rows), 1, 2 * ncoef)
    gains, gflag = ap.antenna_gains(path, [0], [T0], 0, MS_CHAN_FREQ[0],
                                    elevation_grid=np.full((1, 1), np.pi / 4))
    assert not gflag.any()
    np.testing.assert_allclose(gains.real, np.sqrt(2.0), rtol=1e-6)   # EPowerCurve: gain = sqrt(power)


def test_flag_column_stored_per_polarization_covers_that_polarization():
    """One flag per polarization expands over the parameters of that polarization, not just the first."""
    flag = np.zeros((3, 1, 2), dtype=bool)
    flag[:, 0, 1] = True
    out = ap._match_param_grid(flag, (3, 1, 8), "FLAG", "t")
    assert out.shape == (3, 1, 8)
    assert not out[:, 0, :4].any() and out[:, 0, 4:].all()


def test_caltable_entries_from_vlbipy():
    """vlbipy CalTable objects are converted to apply_tables entries with gainfield names resolved to ids."""
    from vlbipy.models import CalTable

    tables = [CalTable(cal_type="sbd", path="/x/a.sbd", interp="nearest", spwmap=[0, 0], gainfield="3C345", calwt=False),
              CalTable(cal_type="tsys", path="/x/t.tsys", interp="linear", gainfield="UNKNOWN", calwt=True),
              CalTable(cal_type="mbd", path="/x/m.mbd", gainfield="3C345,1")]
    entries = ap.caltable_entries_from_vlbipy(tables, {"3C345": 16})
    assert entries[0] == {"path": "/x/a.sbd", "interp": "nearest", "spwmap": [0, 0], "gainfield": [16],
                          "calwt": False, "cal_type": "sbd"}
    assert entries[1]["gainfield"] == [] and entries[1]["calwt"] is True
    assert entries[2]["gainfield"] == [16, 1]


# ----------------------------------------------------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (REAL_MS.exists() and REAL_SBD.exists()), reason="real MS / CASA sbd table not present")
def test_real_sbd_flattens_phase_vs_channel():
    """Applying the CASA sbd table to scan 63 baselines to EF leaves a flat corrected phase across channels 6..57."""
    tables = pytest.importorskip("casacore.tables")
    ms = tables.table(str(REAL_MS), ack=False, readonly=True)
    q = ms.query("SCAN_NUMBER==63 && (ANTENNA1==2 || ANTENNA2==2)")
    vis = q.getcol("DATA")
    flag = q.getcol("FLAG")
    weight = q.getcol("WEIGHT_SPECTRUM")
    a1, a2, tt, ddid = q.getcol("ANTENNA1"), q.getcol("ANTENNA2"), q.getcol("TIME"), q.getcol("DATA_DESC_ID")
    q.close()
    dd = tables.table(str(REAL_MS / "DATA_DESCRIPTION"), ack=False)
    spw = dd.getcol("SPECTRAL_WINDOW_ID")[ddid]
    dd.close()
    spwt = tables.table(str(REAL_MS / "SPECTRAL_WINDOW"), ack=False)
    chan_freq = spwt.getcol("CHAN_FREQ")
    spwt.close()
    names = tables.table(str(REAL_MS / "ANTENNA"), ack=False).getcol("NAME")
    ms.close()
    table = ap.load_caltable(REAL_SBD)
    assert table.kind == "Fringe Jones"
    entry = [{"path": table, "interp": "nearest", "spwmap": [], "gainfield": [16]}]
    t_start = _time.perf_counter()
    vis_c, flag_c, _ = ap.apply_tables(vis, flag, weight, a1, a2, tt, spw, chan_freq, entry)
    elapsed = _time.perf_counter() - t_start
    print(f"\napply_tables on {vis.shape[0]} rows x {vis.shape[1]} ch x {vis.shape[2]} corr: {elapsed:.3f} s")
    chans = slice(6, 58)
    checked = 0
    total_before, total_after = 0.0, 0.0
    for ant in np.unique(np.concatenate([a1, a2])):
        if ant == 2:
            continue
        for s in range(chan_freq.shape[0]):
            sel = ((a1 == ant) | (a2 == ant)) & (spw == s)
            for pol, corr in (("RR", 0), ("LL", 3)):
                good = ~flag_c[sel][:, chans, corr]
                if good.mean() < 0.5:
                    continue
                raw = np.where(~flag[sel][:, chans, corr], vis[sel][:, chans, corr], 0).mean(axis=0)
                cor = np.where(good, vis_c[sel][:, chans, corr], 0).mean(axis=0)
                std_before = np.std(np.unwrap(np.angle(raw)))
                std_after = np.std(np.unwrap(np.angle(cor)))
                print(f"{names[ant]:>3s} spw{s} {pol}: phase-vs-channel std before {std_before:.3f} rad, "
                      f"after {std_after:.3f} rad")
                assert std_after < 0.15, (names[ant], s, pol, std_after)
                if std_before > 0.3:  # clearly above the noise floor of a time-averaged baseline
                    assert std_after < 0.5 * std_before, (names[ant], s, pol, std_before, std_after)
                total_before, total_after, checked = total_before + std_before, total_after + std_after, checked + 1
    print(f"checked {checked} baseline/spw/pol combinations: mean std before {total_before / checked:.3f} rad, "
          f"after {total_after / checked:.3f} rad")
    assert checked > 10
    assert total_after < 0.3 * total_before
