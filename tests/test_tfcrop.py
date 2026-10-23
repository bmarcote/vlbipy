"""Tests of the native ``flagdata(mode='tfcrop')`` (:mod:`vlbipy.solvers.tfcrop`).

A small synthetic measurement set written without casacore (:mod:`vlbipy.casacore_format.writer`) carries a
smooth bandpass with bounded noise and injected RFI (a broadband burst, a narrowband line, a single spike):
the RFI must be flagged, the clean data must not, and the compiled and numpy implementations must agree.
The comparison with CASA runs ``casatasks.flagdata`` on two parts of a copy-on-write clone of the RSM07
reference import and requires identical FLAG and FLAG_ROW columns.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.writer import array_column, create_table, scalar_column
from vlbipy.solvers import tfcrop as tfcrop_module
from vlbipy.solvers.tfcrop import crop_series, numba_available, tfcrop

NANT, NSPW, NCHAN, NCORR, NTIME = 4, 2, 32, 4, 40
RSM07_REF = Path(__file__).resolve().parent.parent / "test" / "RSM07_import_ref" / "RSM07.mms"
TFCROP = dict(mode="tfcrop", datacolumn="data", timecutoff=4.0, freqcutoff=3.0, extendflags=False, flagbackup=False)


def _write_ms(path: Path) -> dict:
    """Write the synthetic MS; return its index columns and the masks of the RFI and pre-flagged cells."""
    rng = np.random.default_rng(3)
    pairs = [(i, j) for i in range(NANT) for j in range(i, NANT)]          # autocorrelations included
    rows = [(scan, spw, t, i, j) for scan in (1, 2) for spw in range(NSPW) for t in range(NTIME) for i, j in pairs]
    scan, spw, step, ant1, ant2 = (np.array(column) for column in zip(*rows))
    nrow = len(rows)
    level = rng.uniform(1.0, 3.0, (NANT, NANT, NCORR))[ant1, ant2]
    band = 1.0 + 0.3 * np.arange(NCHAN) / NCHAN
    # Bounded noise (+-5%): no clean sample is ever 3 standard deviations away.
    amplitude = level[:, None, :] * band[None, :, None] * (1.0 + rng.uniform(-0.05, 0.05, (nrow, NCHAN, NCORR)))
    rfi = np.zeros((nrow, NCHAN, NCORR), dtype=bool)
    rfi[(scan == 1) & (spw == 0) & (ant1 == 0) & (ant2 == 1) & (step == 17)] = True               # burst
    line = (scan == 2) & (spw == 1) & (ant1 == 1) & (ant2 == 2)
    rfi[np.ix_(np.flatnonzero(line), [9], [0, 3])] = True                                          # narrowband
    rfi[np.flatnonzero((scan == 1) & (spw == 1) & (ant1 == 0) & (ant2 == 3) & (step == 5)), 20, 1] = True
    amplitude = np.where(rfi, amplitude * 15.0, amplitude)
    data = (amplitude * np.exp(1j * rng.uniform(-np.pi, np.pi, amplitude.shape))).astype(np.complex64)
    data[ant1 == ant2] = np.abs(data[ant1 == ant2])
    preflag = np.zeros(rfi.shape, dtype=bool)
    dead = (scan == 2) & (spw == 0) & ((ant1 == 3) | (ant2 == 3))
    preflag[dead] = True
    # Scattered flagged samples. (Not whole channels: next to a channel flagged at every time, the piecewise fit
    # extrapolates and its running mean drags the neighbouring channel, which CASA then flags too.)
    preflag |= rng.random(preflag.shape) < 0.02
    columns = [scalar_column("TIME", "double"), scalar_column("ANTENNA1", "int"), scalar_column("ANTENNA2", "int"),
               scalar_column("FIELD_ID", "int"), scalar_column("SCAN_NUMBER", "int"),
               scalar_column("DATA_DESC_ID", "int"), scalar_column("OBSERVATION_ID", "int"),
               scalar_column("ARRAY_ID", "int"), scalar_column("FLAG_ROW", "boolean"),
               array_column("DATA", "complex", shape=(NCHAN, NCORR)),
               array_column("FLAG", "boolean", shape=(NCHAN, NCORR))]
    values = {"TIME": 4.9e9 + 2.0 * step + 1000.0 * (scan - 1), "ANTENNA1": ant1.astype(np.int32),
              "ANTENNA2": ant2.astype(np.int32), "FIELD_ID": (scan - 1).astype(np.int32),
              "SCAN_NUMBER": scan.astype(np.int32), "DATA_DESC_ID": spw.astype(np.int32),
              "OBSERVATION_ID": np.zeros(nrow, np.int32), "ARRAY_ID": np.zeros(nrow, np.int32),
              "FLAG_ROW": dead, "DATA": data, "FLAG": preflag}
    create_table(path, columns, nrow, values, tiled={"DATA": (NCHAN, NCORR), "FLAG": (NCHAN, NCORR)})
    create_table(path / "ANTENNA", [scalar_column("NAME", "string"), array_column("POSITION", "double", shape=(3,)),
                                    scalar_column("MOUNT", "string")], NANT,
                 {"NAME": [f"A{i}" for i in range(NANT)], "POSITION": rng.normal(0, 6e6, (NANT, 3)),
                  "MOUNT": ["ALT-AZ"] * NANT})
    create_table(path / "FIELD", [scalar_column("NAME", "string"), array_column("PHASE_DIR", "double", shape=(1, 2))],
                 2, {"NAME": ["CAL", "TGT"], "PHASE_DIR": np.array([[[0.1, 0.5]], [[0.2, 0.6]]])})
    create_table(path / "SPECTRAL_WINDOW", [array_column("CHAN_FREQ", "double", shape=(NCHAN,))], NSPW,
                 {"CHAN_FREQ": np.array([1.6e9 + 1e6 * np.arange(NCHAN), 1.7e9 + 1e6 * np.arange(NCHAN)])})
    create_table(path / "DATA_DESCRIPTION", [scalar_column("SPECTRAL_WINDOW_ID", "int"),
                                             scalar_column("POLARIZATION_ID", "int")], NSPW,
                 {"SPECTRAL_WINDOW_ID": np.arange(NSPW, dtype=np.int32), "POLARIZATION_ID": np.zeros(NSPW, np.int32)})
    create_table(path / "POLARIZATION", [scalar_column("NUM_CORR", "int"), array_column("CORR_TYPE", "int",
                                                                                       shape=(NCORR,))], 1,
                 {"NUM_CORR": np.array([NCORR], np.int32), "CORR_TYPE": np.array([[5, 6, 7, 8]], np.int32)})
    return {"ms": path, "rfi": rfi, "preflag": preflag, "dead": dead, "scan": scan}


def _flags(ms) -> tuple[np.ndarray, np.ndarray]:
    """FLAG and FLAG_ROW of a table."""
    with Table(ms) as table:
        return np.asarray(table.getcol("FLAG")), np.asarray(table.getcol("FLAG_ROW"))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """The synthetic MS, read and written with the native table engine."""
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")
    return _write_ms(tmp_path / "sim.ms")


# ----------------------------------------------------------------------------------------------------------------
# Without CASA
# ----------------------------------------------------------------------------------------------------------------
def test_tfcrop_flags_the_rfi_and_nothing_else(synthetic):
    report = tfcrop(synthetic["ms"], workers=1)
    flag, flag_row = _flags(synthetic["ms"])
    clean = ~synthetic["rfi"] & ~synthetic["preflag"]
    assert flag[synthetic["rfi"]].all()
    assert not flag[clean].any()
    assert flag[synthetic["preflag"]].all()
    np.testing.assert_array_equal(flag_row, synthetic["dead"])             # FLAG_ROW is left as it was
    new = synthetic["rfi"] & ~synthetic["preflag"]
    assert report["newly_flagged"] == int(new.sum()) == int(flag.sum() - synthetic["preflag"].sum())
    assert report["chunks"] == 4 and report["rows"] == flag.shape[0] and report["parts"] == 1


def test_field_selection_flags_only_the_selected_field(synthetic):
    tfcrop(synthetic["ms"], field="TGT", workers=1)
    flag, _ = _flags(synthetic["ms"])
    in_target = (synthetic["scan"] == 2)[:, None, None]
    assert flag[synthetic["rfi"] & in_target].all()
    np.testing.assert_array_equal(flag[~in_target[:, 0, 0]], synthetic["preflag"][~in_target[:, 0, 0]])


@pytest.mark.skipif(not numba_available(), reason="numba is not installed")
def test_compiled_and_numpy_implementations_agree(synthetic, monkeypatch, tmp_path):
    other = _write_ms(tmp_path / "copy.ms")
    tfcrop(synthetic["ms"], workers=1)
    monkeypatch.setattr(tfcrop_module, "numba_available", lambda: False)
    tfcrop(other["ms"], workers=1)
    np.testing.assert_array_equal(_flags(synthetic["ms"])[0], _flags(other["ms"])[0])
    # Awkward planes: one-row and short series, few channels, zeros, constant data, NaNs, flagged throughout.
    rng = np.random.default_rng(5)
    for nchan in (1, 3, 5, 9, 64):
        lengths = [1, 2, 3, 4, 5, 8, 13, 40, 40, 40, 40]
        starts = np.concatenate([[0], np.cumsum(lengths)])
        amplitude = np.abs(rng.normal(1.0, 0.3, (starts[-1], nchan, NCORR))).astype(np.float32)
        flags = rng.random(amplitude.shape) < 0.1
        amplitude[starts[7]:starts[8]] = 0.0
        amplitude[starts[8]:starts[9]] = 1.0
        flags[starts[9]:starts[10]] = True
        amplitude[starts[10]:starts[11], :, 0] = np.nan
        amplitude[starts[10] + 3, :, 1] = 100.0
        compiled, vectorised = flags.copy(), flags.copy()
        crop_series(amplitude, compiled, starts, use_numba=True)
        with np.errstate(all="ignore"):
            crop_series(amplitude, vectorised, starts, use_numba=False)
        np.testing.assert_array_equal(compiled, vectorised)


@pytest.mark.parametrize("kwargs", [{"ntime": 60.0}, {"extendflags": True}, {"correlation": "RR"},
                                    {"flagdimension": "time"}, {"combinescans": True}, {"datacolumn": "model"},
                                    {"field": "0"}, {"field": "C*"}, {"field": "NOPE"}, {"timecutoff": "4"},
                                    {"flagbackup": True}, {"action": "calculate"}, {"vis": "x.ms"},
                                    {"datacolumn": "corrected"}])
def test_unsupported_requests_raise_before_writing(synthetic, kwargs):
    before, _ = _flags(synthetic["ms"])
    with pytest.raises(NotImplementedError):
        tfcrop(synthetic["ms"], workers=1, **kwargs)
    np.testing.assert_array_equal(_flags(synthetic["ms"])[0], before)


def test_keywords_tfcrop_ignores_are_accepted(synthetic):
    report = tfcrop(synthetic["ms"], workers=1, mode="tfcrop", autocorr=True, growtime=10.0, chanbin=4,
                       ntime="scan", maxnpieces=7, field="CAL,TGT")
    assert report["newly_flagged"] == int((synthetic["rfi"] & ~synthetic["preflag"]).sum())


# ----------------------------------------------------------------------------------------------------------------
# Against CASA, on RSM07
# ----------------------------------------------------------------------------------------------------------------
def _clone(source: Path, dest: Path) -> None:
    """Copy-on-write clone of a directory (APFS), or skip the test."""
    if subprocess.run(["cp", "-cR", str(source), str(dest)], capture_output=True).returncode != 0:
        pytest.skip("copy-on-write clones (APFS) are not available here")


@pytest.mark.skipif(not RSM07_REF.is_dir(), reason="RSM07 reference import not present")
def test_tfcrop_is_identical_to_casa_on_rsm07(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # casatasks writes its log in the working directory
    casatasks = pytest.importorskip("casatasks")
    from vlbipy.solvers.flagdata import apply_flag_selections, compile_flagdata, read_flag_setup
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")
    base = tmp_path / "base.mms"
    _clone(RSM07_REF, base)
    # Part 1 gets flags first (an antenna for a while, a quack, band edges, one correlation of one antenna).
    preflagged = base / "SUBMSS" / "RSM07.mms.0001.ms"
    setup = read_flag_setup(preflagged)
    for kwargs in (dict(mode="manual", antenna="JB", timerange="2025/09/16/13:09:59~2025/09/16/13:40:00"),
                   dict(mode="quack", quackinterval=5.0, quackmode="beg"), dict(mode="manual", spw="*:0~1;62~63"),
                   dict(mode="manual", antenna="MC", correlation="RL")):
        apply_flag_selections(preflagged, compile_flagdata(dict(kwargs, vis=str(preflagged)), setup), setup=setup,
                              workers=1)
    for name in ("casa", "native", "numpy"):
        _clone(base, tmp_path / f"{name}.mms")
    cases = [("RSM07.mms.0000.ms", {"field": "J1848+3219,3C345"}), ("RSM07.mms.0001.ms", {})]
    for part, extra in cases:
        casatasks.flagdata(vis=str(tmp_path / "casa.mms" / "SUBMSS" / part), **TFCROP, **extra)
        tfcrop(tmp_path / "native.mms" / "SUBMSS" / part, workers=1, **extra)
    monkeypatch.setattr(tfcrop_module, "numba_available", lambda: False)
    tfcrop(tmp_path / "numpy.mms" / "SUBMSS" / cases[1][0], workers=1)
    for part, _ in cases:
        before, before_row = _flags(base / "SUBMSS" / part)
        casa, casa_row = _flags(tmp_path / "casa.mms" / "SUBMSS" / part)
        native, native_row = _flags(tmp_path / "native.mms" / "SUBMSS" / part)
        assert casa.sum() > before.sum()                                   # CASA did flag something
        np.testing.assert_array_equal(native, casa)
        np.testing.assert_array_equal(native_row, casa_row)
        np.testing.assert_array_equal(casa_row, before_row)
    np.testing.assert_array_equal(_flags(tmp_path / "numpy.mms" / "SUBMSS" / cases[1][0])[0],
                                  _flags(tmp_path / "casa.mms" / "SUBMSS" / cases[1][0])[0])
