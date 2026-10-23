"""The native accor + smoothcal (``vlbipy.solvers.accor``): a synthetic measurement set with known
autocorrelation scales (no CASA needed), the smoothing filter on known series, and CASA's tables for RSM07."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vlbipy.casacore_format.aipsio import TableRef
from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.writer import array_column, create_table, scalar_column
from vlbipy.solvers.accor import (MEDIAN_EVEN_MEAN_MAX, parse_solint, accor, smoothcal,
                                       smooth_series)
from vlbipy.solvers.caltable import _columns
from vlbipy.solvers.gaincurve import SUBTABLE_DESCS

RSM07_MMS = Path(__file__).resolve().parents[1] / "test" / "RSM07_import_ref" / "RSM07.mms"
NANT, NSPW, NCHAN, NCORR, NTIME, DT = 4, 2, 8, 4, 10, 2.0          # antenna 3 has no data at all
T0, SCAN2_START = 5.0e9, 100.0
CHAN_FREQ = np.array([[1.6e9 + 1e6 * c for c in range(NCHAN)], [1.7e9 + 1e6 * c for c in range(NCHAN)]])


def true_scales() -> np.ndarray:
    """Amplitude scale (scan, antenna, spw, pol) that accor must find: the autocorrelations are its square."""
    antenna = np.arange(NANT)[None, :, None, None]
    spw = np.arange(NSPW)[None, None, :, None]
    pol = np.arange(2)[None, None, None, :]
    scan = np.array([1.0, 1.1])[:, None, None, None]
    return scan * (1.0 + 0.1 * antenna + 0.01 * spw + 0.05 * pol)


def make_ms(path: Path, *, flag_ll_antenna1_scan1: bool = True, flag_everything: bool = False) -> Path:
    """A small MS written with :mod:`vlbipy.casacore_format`: 2 scans x 2 subbands x 10 integrations, 3 antennas
    with autocorrelations ``scale**2`` in the parallel hands (cross-correlation rows hold junk accor must ignore).

    Antenna 1's LL is flagged in scan 1 (``flag_ll_antenna1_scan1``), antenna 2's RR on its lower half band, and
    antenna 0's rows of scan 2 have FLAG_ROW set (with FLAG clear: CASA still treats them as flagged).
    """
    scales = true_scales()
    rows = []
    for scan in (1, 2):
        for spw in range(NSPW):
            for step in range(NTIME):
                time = T0 + (scan - 1) * SCAN2_START + DT * step
                rows += [(scan, spw, time, i, j) for i in range(3) for j in range(i, 3)]
    scan, spw, time, ant1, ant2 = (np.array(column) for column in zip(*rows))
    nrow = scan.size
    data = np.full((nrow, NCHAN, NCORR), 100.0 + 50.0j, dtype=np.complex64)
    flag = np.zeros((nrow, NCHAN, NCORR), dtype=bool)
    auto = ant1 == ant2
    for row in np.flatnonzero(auto):
        scale = scales[scan[row] - 1, ant1[row], spw[row]]
        data[row, :, 0], data[row, :, 3] = scale[0] ** 2, scale[1] ** 2
        data[row, :, 1:3] = 0.1 + 0.05j
    if flag_ll_antenna1_scan1:
        flag[auto & (ant1 == 1) & (scan == 1), :, 3] = True
    flag[auto & (ant1 == 2), :NCHAN // 2, 0] = True
    if flag_everything:
        flag[:] = True
    main = [scalar_column(name, kind) for name, kind in (
        ("TIME", "double"), ("TIME_CENTROID", "double"), ("ANTENNA1", "int"), ("ANTENNA2", "int"),
        ("FIELD_ID", "int"), ("SCAN_NUMBER", "int"), ("DATA_DESC_ID", "int"), ("OBSERVATION_ID", "int"),
        ("ARRAY_ID", "int"), ("FLAG_ROW", "boolean"))]
    main += [array_column("SIGMA", "float", shape=(NCORR,)), array_column("WEIGHT", "float", shape=(NCORR,))]
    cube = (NCHAN, NCORR)
    main += [array_column(name, kind, shape=cube) for name, kind in (
        ("DATA", "complex"), ("FLAG", "boolean"), ("SIGMA_SPECTRUM", "float"), ("WEIGHT_SPECTRUM", "float"))]
    values = {"TIME": time, "TIME_CENTROID": time, "ANTENNA1": ant1.astype(np.int32),
              "ANTENNA2": ant2.astype(np.int32), "FIELD_ID": np.zeros(nrow, np.int32),
              "SCAN_NUMBER": scan.astype(np.int32), "DATA_DESC_ID": spw.astype(np.int32),
              "OBSERVATION_ID": np.zeros(nrow, np.int32), "ARRAY_ID": np.zeros(nrow, np.int32),
              "FLAG_ROW": auto & (ant1 == 0) & (scan == 2), "SIGMA": np.full((nrow, NCORR), 0.01, np.float32),
              "WEIGHT": np.full((nrow, NCORR), 1e4, np.float32), "DATA": data, "FLAG": flag,
              "SIGMA_SPECTRUM": np.full((nrow, NCHAN, NCORR), 0.01, np.float32),
              "WEIGHT_SPECTRUM": np.full((nrow, NCHAN, NCORR), 1e4, np.float32)}
    names = ("ANTENNA", "FIELD", "OBSERVATION", "SPECTRAL_WINDOW", "DATA_DESCRIPTION", "POLARIZATION")
    create_table(path, main, nrow, values, keywords={name: TableRef(f"././{name}") for name in names},
                 tiled={name: cube for name in ("DATA", "FLAG", "SIGMA_SPECTRUM", "WEIGHT_SPECTRUM")})
    antennas = [f"A{i}" for i in range(NANT)]
    create_table(path / "ANTENNA", _columns(SUBTABLE_DESCS["ANTENNA"]), NANT, {
        "OFFSET": np.zeros((NANT, 3)), "POSITION": np.ones((NANT, 3)), "TYPE": ["GROUND-BASED"] * NANT,
        "DISH_DIAMETER": np.full(NANT, 25.0), "FLAG_ROW": np.zeros(NANT, bool), "MOUNT": ["ALT-AZ"] * NANT,
        "NAME": antennas, "STATION": antennas})
    create_table(path / "FIELD", _columns(SUBTABLE_DESCS["FIELD"]), 1, {
        "DELAY_DIR": [np.zeros((1, 2))], "PHASE_DIR": [np.zeros((1, 2))], "REFERENCE_DIR": [np.zeros((1, 2))],
        "CODE": [""], "FLAG_ROW": np.zeros(1, bool), "NAME": ["CAL"], "NUM_POLY": np.zeros(1, np.int32),
        "SOURCE_ID": np.zeros(1, np.int32), "TIME": np.array([T0])})
    create_table(path / "OBSERVATION", _columns(SUBTABLE_DESCS["OBSERVATION"]), 1, {
        "TIME_RANGE": np.array([[T0, T0 + 200.0]]), "LOG": [np.array([""], dtype=object)],
        "SCHEDULE": [np.array([""], dtype=object)], "FLAG_ROW": np.zeros(1, bool), "OBSERVER": ["me"],
        "PROJECT": ["SIM"], "RELEASE_DATE": np.zeros(1), "SCHEDULE_TYPE": [""], "TELESCOPE_NAME": ["VLBA"]})
    width = np.full((NSPW, NCHAN), 1e6)
    create_table(path / "SPECTRAL_WINDOW", _columns(SUBTABLE_DESCS["SPECTRAL_WINDOW"]), NSPW, {
        "MEAS_FREQ_REF": np.full(NSPW, 5, np.int32), "CHAN_FREQ": CHAN_FREQ, "REF_FREQUENCY": CHAN_FREQ[:, 0],
        "CHAN_WIDTH": width, "EFFECTIVE_BW": width, "RESOLUTION": width, "FLAG_ROW": np.zeros(NSPW, bool),
        "FREQ_GROUP": np.zeros(NSPW, np.int32), "FREQ_GROUP_NAME": [""] * NSPW,
        "IF_CONV_CHAIN": np.arange(NSPW, dtype=np.int32), "NAME": ["IF0", "IF1"],
        "NET_SIDEBAND": np.ones(NSPW, np.int32), "NUM_CHAN": np.full(NSPW, NCHAN, np.int32),
        "TOTAL_BANDWIDTH": np.full(NSPW, NCHAN * 1e6)})
    create_table(path / "DATA_DESCRIPTION", [scalar_column("SPECTRAL_WINDOW_ID", "int"),
                                             scalar_column("POLARIZATION_ID", "int")], NSPW,
                 {"SPECTRAL_WINDOW_ID": np.arange(NSPW, dtype=np.int32), "POLARIZATION_ID": np.zeros(NSPW, np.int32)})
    create_table(path / "POLARIZATION", [scalar_column("NUM_CORR", "int")], 1,
                 {"NUM_CORR": np.array([NCORR], np.int32)})
    return path


def read_table(path: Path) -> dict:
    """The main columns of a calibration table (row-major) and its SPECTRAL_WINDOW CHAN_FREQ and FLAG_ROW."""
    with Table(path) as table:
        out = {name: np.asarray(table.getcol(name)) for name in ("TIME", "SPECTRAL_WINDOW_ID", "ANTENNA1",
                                                                  "ANTENNA2", "SCAN_NUMBER", "CPARAM", "FLAG",
                                                                  "SNR", "PARAMERR", "INTERVAL")}
        out["keywords"] = table.getkeywords()
    with Table(path / "SPECTRAL_WINDOW") as table:
        out["spw_freq"] = np.asarray(table.getcol("CHAN_FREQ"))
        out["spw_flag"] = np.asarray(table.getcol("FLAG_ROW"))
    return out


@pytest.fixture
def native_engine(monkeypatch):
    """Read the synthetic tables with the native engine (they have no casacore-library counterpart to compare)."""
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")


@pytest.mark.parametrize("corrdepflags", [True, False])
def test_accor_recovers_the_autocorrelation_scale_of_every_antenna_subband_and_polarization(tmp_path, native_engine,
                                                                                            corrdepflags):
    ms = make_ms(tmp_path / "sim.ms")
    report = accor(ms, tmp_path / "sim.accor", solint="10s", corrdepflags=corrdepflags, workers=1)
    table = read_table(tmp_path / "sim.accor")
    # 2 scans x 2 bins of 10 s x 2 subbands, one row per antenna of the ANTENNA table each
    assert report["rows"] == 2 * 2 * NSPW * NANT == table["TIME"].size and report["solved"] == 8
    assert table["keywords"]["VisCal"] == "Accor Jones" and table["keywords"]["ParType"] == "Complex"
    order = np.lexsort((table["ANTENNA1"], table["TIME"], table["SPECTRAL_WINDOW_ID"]))
    assert np.array_equal(order, np.arange(order.size))
    assert np.all(table["ANTENNA2"] == -1) and np.all(table["SNR"] == 0) and np.all(table["PARAMERR"] == 0)
    assert np.all(table["INTERVAL"] == 0)
    # the first 10-s bin of scan 1 holds the integrations at 0, 2, ..., 8 s: equal weights put it at 4 s
    assert np.isclose(table["TIME"].min(), T0 + 4.0, atol=1e-5)
    scales = true_scales()
    cparam, flag = table["CPARAM"][:, 0, :], table["FLAG"][:, 0, :]
    for row in range(table["TIME"].size):
        antenna, spw, scan = table["ANTENNA1"][row], table["SPECTRAL_WINDOW_ID"][row], table["SCAN_NUMBER"][row]
        if antenna == 3:                                              # no data: flagged, CPARAM = 1
            assert flag[row].all() and np.all(cparam[row] == 1.0)
            continue
        dead_ll = antenna == 1 and scan == 1
        row_flagged = antenna == 0 and scan == 2
        expected_flag = [row_flagged or (dead_ll and not corrdepflags), row_flagged or dead_ll]
        assert list(flag[row]) == expected_flag
        for pol in range(2):
            if not expected_flag[pol]:
                assert cparam[row, pol].imag == 0
                assert np.isclose(cparam[row, pol].real, scales[scan - 1, antenna, spw, pol], rtol=1e-6)
    assert np.allclose(table["spw_freq"][:, 0], CHAN_FREQ.mean(axis=1)) and not table["spw_flag"].any()


def test_accor_with_an_infinite_interval_solves_each_scan_and_subband_once(tmp_path, native_engine):
    ms = make_ms(tmp_path / "sim.ms", flag_ll_antenna1_scan1=False)
    report = accor(ms, tmp_path / "sim.accor", solint="inf", workers=1)
    table = read_table(tmp_path / "sim.accor")
    assert report["solved"] == 2 * NSPW and table["TIME"].size == 2 * NSPW * NANT
    assert np.isclose(np.unique(table["TIME"])[0], T0 + DT * (NTIME - 1) / 2, atol=1e-5)


def test_accor_without_usable_autocorrelations_writes_no_table(tmp_path, native_engine):
    ms = make_ms(tmp_path / "sim.ms", flag_everything=True)
    report = accor(ms, tmp_path / "sim.accor", solint="30s", workers=1)
    assert report["rows"] == 0 and not (tmp_path / "sim.accor").exists()


def test_requests_that_casa_answers_differently_are_refused(tmp_path, native_engine):
    ms = make_ms(tmp_path / "sim.ms")
    with pytest.raises(NotImplementedError):
        accor(ms, tmp_path / "x.accor", solint="30s", combine="scan", workers=1)
    with pytest.raises(NotImplementedError):
        accor(ms, tmp_path / "x.accor", solint="30s", field="CAL", workers=1)
    for solint in ("int", "30s,4ch", "3 parsecs"):
        with pytest.raises(NotImplementedError):
            parse_solint(solint)
    assert parse_solint("1min") == 60.0 and parse_solint(2.5) == 2.5 and parse_solint("") == float("inf")
    assert parse_solint("0.1") == float(np.float32(0.1))                 # CASA iterates with a float32 interval
    accor(ms, tmp_path / "sim.accor", solint="30s", workers=1)
    with pytest.raises(NotImplementedError):
        smoothcal(ms, tmp_path / "sim.accor", tmp_path / "s", smoothtype="median", smoothtime=60, field="CAL")
    with pytest.raises(RuntimeError):
        smoothcal(ms, tmp_path / "sim.accor", tmp_path / "s", smoothtype="boxcar", smoothtime=60)


def test_smooth_series_takes_the_median_or_mean_of_the_window_and_unflags_like_casa():
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    values = np.array([1.0, 2.0, 10.0, 3.0, 4.0], dtype=np.float32)
    ok = np.ones(5, dtype=bool)
    median, new_ok = smooth_series(times, values, ok, smoothtype="median", smoothtime=25.0)
    assert np.array_equal(median, np.float32([1.5, 2.0, 3.0, 4.0, 3.5])) and new_ok.all()
    mean, _ = smooth_series(times, values, ok, smoothtype="mean", smoothtime=25.0)
    expected = [np.float32(np.float32(1 + 2) / 2), np.float32(np.float32(np.float32(1 + 2) + 10) / 3)]
    assert np.array_equal(mean[:2], np.float32(expected))
    # the window is (t - w/2, t + w/2]: the later neighbour is in, the earlier one at the same distance is not
    both, _ = smooth_series(np.array([0.0, 10.0]), np.float32([1.0, 3.0]), np.ones(2, bool), smoothtype="mean",
                            smoothtime=20.0)
    assert np.array_equal(both, np.float32([2.0, 3.0]))
    # a flagged value with unflagged neighbours is replaced and unflagged; one without any stays flagged and unchanged
    values = np.float32([1.0, 9.0, 3.0, 7.0])
    smoothed, new_ok = smooth_series(np.array([0.0, 1.0, 2.0, 100.0]), values, np.array([True, False, True, False]),
                                     smoothtype="median", smoothtime=5.0)
    assert list(new_ok) == [True, True, True, False] and smoothed[1] == 2.0 and smoothed[3] == 7.0


@pytest.mark.parametrize("size", [MEDIAN_EVEN_MEAN_MAX, MEDIAN_EVEN_MEAN_MAX + 2])
def test_median_of_an_even_window_depends_on_the_length_of_the_whole_series(size):
    times = np.arange(size, dtype=np.float64)
    values = np.arange(size, dtype=np.float32)
    smoothed, _ = smooth_series(times, values, np.ones(size, bool), smoothtype="median", smoothtime=2.0)
    # windows (t - 1, t + 1] hold t and t + 1: their mean up to 100 solutions, the lower one (t) beyond
    expected = values[:-1] + (np.float32(0.5) if size <= MEDIAN_EVEN_MEAN_MAX else np.float32(0.0))
    assert np.array_equal(smoothed[:-1], expected) and smoothed[-1] == values[-1]


def test_smoothcal_of_a_synthetic_table_unflags_the_dead_polarization_from_its_neighbours(tmp_path, native_engine):
    ms = make_ms(tmp_path / "sim.ms")
    accor(ms, tmp_path / "sim.accor", solint="10s", workers=1)
    report = smoothcal(ms, tmp_path / "sim.accor", tmp_path / "sim.smooth", smoothtype="median",
                             smoothtime=1000.0)
    raw, smooth = read_table(tmp_path / "sim.accor"), read_table(tmp_path / "sim.smooth")
    # antenna 1's LL in the 2 bins of scan 1 and antenna 0's two hands in the 2 bins of scan 2, per subband
    assert report["rows"] == raw["TIME"].size and report["unflagged"] == (2 + 2 * 2) * NSPW
    for spw in range(NSPW):
        rows = np.flatnonzero((smooth["ANTENNA1"] == 1) & (smooth["SPECTRAL_WINDOW_ID"] == spw))
        good = ~raw["FLAG"][rows, 0, 1]
        ll = raw["CPARAM"][rows, 0, 1].real[good]                      # the two scan-2 solutions
        expected = np.float32(np.float32(np.sort(ll)[0] + np.sort(ll)[1]) * np.float32(0.5))
        assert not smooth["FLAG"][rows, 0, 1].any() and np.all(smooth["CPARAM"][rows, 0, 1].real == expected)
    assert np.all(smooth["FLAG"][smooth["ANTENNA1"] == 3])               # nothing to smooth from
    assert np.array_equal(smooth["TIME"], raw["TIME"])


def _clone(source: Path, destination: Path) -> Path:
    """A copy-on-write clone (APFS ``cp -c``); the test is skipped where the file system cannot make one."""
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["cp", "-cR", str(source), str(destination)], capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"no copy-on-write clone of {source.name} possible here")
    return destination / source.name


def _assert_tables_identical(native: Path, casa: Path) -> None:
    """Every column of the main table and of each subtable is equal; keywords too, except CASA_Version."""
    for sub in ("", "OBSERVATION", "ANTENNA", "FIELD", "SPECTRAL_WINDOW", "HISTORY"):
        with Table(native / sub if sub else native) as ours, Table(casa / sub if sub else casa) as theirs:
            assert ours.colnames() == theirs.colnames() and ours.nrows() == theirs.nrows(), sub
            for name in theirs.colnames():
                left, right = ours.getcol(name), theirs.getcol(name)
                if isinstance(right, list):
                    assert left == right, (sub, name)
                else:
                    assert np.array_equal(np.asarray(left), np.asarray(right)), (sub, name)
            if not sub:
                assert {k: v for k, v in ours.getkeywords().items() if k != "CASA_Version"} == \
                    {k.replace(str(casa), str(native)): (v.replace(str(casa), str(native)) if isinstance(v, str)
                                                         else v)
                     for k, v in theirs.getkeywords().items() if k != "CASA_Version"}
                assert ours.info() == theirs.info()


@pytest.mark.skipif(not RSM07_MMS.is_dir(), reason="RSM07 imported Multi-MS not available")
def test_rsm07_accor_and_smoothcal_tables_are_identical_to_casa(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                       # CASA writes its log in the working directory
    casatasks = pytest.importorskip("casatasks")
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")
    vis = _clone(RSM07_MMS, tmp_path / "clone")
    casatasks.accor(vis=str(vis), caltable=str(tmp_path / "casa.accor"), solint="30s", corrdepflags=True)
    casatasks.smoothcal(vis=str(vis), tablein=str(tmp_path / "casa.accor"), caltable=str(tmp_path / "casa.smooth"),
                        smoothtype="median", smoothtime=1800.0)
    report = accor(vis, tmp_path / "native.accor", solint="30s", corrdepflags=True)
    smoothcal(vis, tmp_path / "native.accor", tmp_path / "native.smooth", smoothtype="median",
                    smoothtime=1800.0)
    assert report["rows"] == 17304 and report["solved"] == 1236
    _assert_tables_identical(tmp_path / "native.accor", tmp_path / "casa.accor")
    _assert_tables_identical(tmp_path / "native.smooth", tmp_path / "casa.smooth")
    shutil.rmtree(tmp_path / "clone")
