"""Tests for vlbipy.solvers.caltable: round-trip a fringe caltable and compare its layout with a real CASA one."""

from pathlib import Path

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

from vlbipy.solvers import caltable  # noqa: E402

REFERENCE_TABLE = Path("/home/marcote/Programing/vlbipy/rsm07/caltables/rsm07.mbd")
NANT, NFIELD, NSPW, NCHAN = 3, 2, 2, 8
MS_CHAN_FREQ = np.array([[1.5e9 + 1e6 * c for c in range(NCHAN)], [1.6e9 + 1e6 * c for c in range(NCHAN)]])


def _scalar(value_type, option=0):
    """Minimal scalar column description for the fake MS subtables."""
    return {"valueType": value_type, "dataManagerType": "StandardStMan", "dataManagerGroup": "MSMTAB",
            "option": option, "maxlen": 0, "comment": "", "keywords": {}}


def _array(value_type, ndim=1, shape=None):
    """Minimal array column description for the fake MS subtables."""
    desc = _scalar(value_type, option=5 if shape is not None else 0)
    desc["ndim"] = ndim
    if shape is not None:
        desc["shape"] = np.array(shape)
    return desc


def _create(path, desc, nrow, columns):
    """Create a table at `path` from `desc`, add `nrow` rows and fill `columns` (name -> casatools-ordered array)."""
    tb = casatools.table()
    tb.create(str(path), desc)
    tb.addrows(nrow)
    for name, values in columns.items():
        tb.putcol(name, values)
    tb.close()


@pytest.fixture
def fake_ms(tmp_path):
    """Directory with ANTENNA (3 rows), FIELD (2), OBSERVATION (1), SPECTRAL_WINDOW (2 spws x 8 chans)."""
    ms = tmp_path / "fake.ms"
    ms.mkdir()
    ant_desc = {"NAME": _scalar("string"), "STATION": _scalar("string"), "TYPE": _scalar("string"),
                "MOUNT": _scalar("string"), "POSITION": _array("double", 1, [3]), "OFFSET": _array("double", 1, [3]),
                "DISH_DIAMETER": _scalar("double"), "FLAG_ROW": _scalar("boolean")}
    _create(ms / "ANTENNA", ant_desc, NANT, {
        "NAME": np.array(["EF", "WB", "O8"]), "STATION": np.array(["EF", "WB", "O8"]),
        "TYPE": np.array(["GROUND-BASED"] * NANT), "MOUNT": np.array(["ALT-AZ"] * NANT),
        "POSITION": np.arange(9, dtype=float).reshape(NANT, 3).T, "OFFSET": np.zeros((3, NANT)),
        "DISH_DIAMETER": np.array([100.0, 25.0, 20.0]), "FLAG_ROW": np.zeros(NANT, dtype=bool)})
    field_desc = {"NAME": _scalar("string"), "CODE": _scalar("string"), "SOURCE_ID": _scalar("int"),
                  "NUM_POLY": _scalar("int"), "TIME": _scalar("double"), "FLAG_ROW": _scalar("boolean"),
                  "PHASE_DIR": _array("double", 2), "DELAY_DIR": _array("double", 2),
                  "REFERENCE_DIR": _array("double", 2)}
    dirs = np.array([[[0.1, 0.2]], [[0.3, 0.4]]]).T  # (2, 1, NFIELD) Fortran order of per-row (1, 2)
    _create(ms / "FIELD", field_desc, NFIELD, {
        "NAME": np.array(["SRC_A", "SRC_B"]), "CODE": np.array(["", ""]), "SOURCE_ID": np.arange(NFIELD, dtype=np.int32),
        "NUM_POLY": np.zeros(NFIELD, dtype=np.int32), "TIME": np.zeros(NFIELD), "FLAG_ROW": np.zeros(NFIELD, dtype=bool),
        "PHASE_DIR": dirs, "DELAY_DIR": dirs, "REFERENCE_DIR": dirs})
    obs_desc = {"TELESCOPE_NAME": _scalar("string"), "OBSERVER": _scalar("string"), "PROJECT": _scalar("string"),
                "RELEASE_DATE": _scalar("double"), "FLAG_ROW": _scalar("boolean"),
                "TIME_RANGE": _array("double", 1, [2]), "SCHEDULE_TYPE": _scalar("string")}
    _create(ms / "OBSERVATION", obs_desc, 1, {
        "TELESCOPE_NAME": np.array(["EVN"]), "OBSERVER": np.array(["me"]), "PROJECT": np.array(["TEST"]),
        "RELEASE_DATE": np.zeros(1), "FLAG_ROW": np.zeros(1, dtype=bool), "TIME_RANGE": np.array([[5.0e9], [5.0e9 + 3600]]),
        "SCHEDULE_TYPE": np.array([""])})
    spw_desc = caltable.spectral_window_table_desc()
    _create(ms / "SPECTRAL_WINDOW", spw_desc, NSPW, {
        "CHAN_FREQ": MS_CHAN_FREQ.T, "CHAN_WIDTH": np.full((NCHAN, NSPW), 1e6),
        "EFFECTIVE_BW": np.full((NCHAN, NSPW), 1e6), "RESOLUTION": np.full((NCHAN, NSPW), 1e6),
        "FLAG_ROW": np.zeros(NSPW, dtype=bool), "FREQ_GROUP": np.zeros(NSPW, dtype=np.int32),
        "FREQ_GROUP_NAME": np.array(["", ""]), "IF_CONV_CHAIN": np.zeros(NSPW, dtype=np.int32),
        "MEAS_FREQ_REF": np.full(NSPW, 5, dtype=np.int32), "NAME": np.array(["IF0", "IF1"]),
        "NET_SIDEBAND": np.ones(NSPW, dtype=np.int32), "NUM_CHAN": np.full(NSPW, NCHAN, dtype=np.int32),
        "REF_FREQUENCY": MS_CHAN_FREQ[:, 0], "TOTAL_BANDWIDTH": np.full(NSPW, NCHAN * 1e6)})
    return ms


def _solution_inputs():
    """Return the keyword inputs for a 2 times x 2 spws x 3 antennas (12 rows) fringe table."""
    times = np.array([5.0e9, 5.0e9 + 60.0])
    rows = [(t, s, a) for t in times for s in range(NSPW) for a in range(NANT)]
    nrow = len(rows)
    rng = np.random.default_rng(1)
    fparam = rng.normal(size=(nrow, 8)).astype(np.float32)
    flag = np.zeros((nrow, 8), dtype=bool)
    flag[5] = True
    return {"times": np.array([r[0] for r in rows]), "field_ids": np.array([0] * 6 + [1] * 6),
            "spw_ids": np.array([r[1] for r in rows]), "antenna_ids": np.array([r[2] for r in rows]),
            "refant_id": 0, "scan_numbers": np.array([1] * 6 + [2] * 6), "fparam": fparam,
            "paramerr": np.abs(fparam) * 0.1, "flag": flag, "snr": rng.uniform(5, 50, size=(nrow, 8)),
            "intervals": np.full(nrow, 60.0), "spw_chan_freq": MS_CHAN_FREQ.mean(axis=1),
            "spw_chan_width": np.array([NCHAN * 1e6, NCHAN * 1e6])}


@pytest.fixture
def written(fake_ms, tmp_path):
    """Write the fringe table and return (path, inputs)."""
    inputs = _solution_inputs()
    path = caltable.write_fringe_table(tmp_path / "test.fringe", fake_ms, casa_version="test-version", **inputs)
    return path, inputs


def test_round_trip(written):
    """Every main column, the keywords, info and SPECTRAL_WINDOW survive a write/read cycle."""
    path, inputs = written
    out = caltable.read_fringe_table(path)
    nrow = len(inputs["times"])
    assert out["nrow"] == nrow
    np.testing.assert_array_equal(out["TIME"], inputs["times"])
    np.testing.assert_array_equal(out["FIELD_ID"], inputs["field_ids"])
    np.testing.assert_array_equal(out["SPECTRAL_WINDOW_ID"], inputs["spw_ids"])
    np.testing.assert_array_equal(out["ANTENNA1"], inputs["antenna_ids"])
    np.testing.assert_array_equal(out["ANTENNA2"], np.zeros(nrow, dtype=np.int32))
    np.testing.assert_array_equal(out["INTERVAL"], inputs["intervals"])
    np.testing.assert_array_equal(out["SCAN_NUMBER"], inputs["scan_numbers"])
    np.testing.assert_array_equal(out["OBSERVATION_ID"], np.zeros(nrow, dtype=np.int32))
    for name, key in (("FPARAM", "fparam"), ("PARAMERR", "paramerr"), ("SNR", "snr")):
        assert out[name].shape == (nrow, 8) and out[name].dtype == np.float32
        np.testing.assert_allclose(out[name], np.asarray(inputs[key], dtype=np.float32), rtol=0, atol=0)
    assert out["FLAG"].dtype == bool and out["FLAG"].shape == (nrow, 8)
    np.testing.assert_array_equal(out["FLAG"], inputs["flag"])
    assert "WEIGHT" not in out
    assert out["TIME"].dtype == np.float64 and out["FIELD_ID"].dtype == np.int32
    assert out["info"] == {"type": "Calibration", "subType": "Fringe Jones", "readme": ""}
    kw = out["keywords"]
    assert kw["ParType"] == "Float" and kw["VisCal"] == "Fringe Jones" and kw["PolBasis"] == "unknown"
    assert kw["MSName"] == "fake.ms" and kw["CASA_Version"] == "test-version"
    for sub in ("ANTENNA", "FIELD", "OBSERVATION", "SPECTRAL_WINDOW", "HISTORY"):
        assert kw[sub] == f"Table: {path / sub}"
        assert (path / sub / "table.dat").exists()
    np.testing.assert_array_equal(out["antenna_names"], ["EF", "WB", "O8"])
    np.testing.assert_allclose(out["spw_chan_freq"], inputs["spw_chan_freq"])
    np.testing.assert_allclose(out["spw_chan_width"], inputs["spw_chan_width"])


def test_cell_shapes_and_subtables(written):
    """Cells are (1, 8) like CASA, WEIGHT is undefined, SPECTRAL_WINDOW has one channel and MS scalars copied."""
    path, inputs = written
    tb = casatools.table()
    tb.open(str(path))
    assert tb.getcell("FPARAM", 0).shape == (8, 1)  # casatools Fortran view of a (1, 8) cell
    assert tb.getcell("FLAG", 0).shape == (8, 1)
    assert not tb.iscelldefined("WEIGHT", 0)
    np.testing.assert_allclose(tb.getcell("FPARAM", 3).ravel(), inputs["fparam"][3])
    tb.close()
    tb.open(str(path / "SPECTRAL_WINDOW"))
    assert tb.nrows() == NSPW
    np.testing.assert_array_equal(tb.getcol("NUM_CHAN"), [1, 1])
    assert tb.getcol("CHAN_FREQ").shape == (1, NSPW)
    np.testing.assert_allclose(tb.getcol("CHAN_FREQ")[0], inputs["spw_chan_freq"])
    np.testing.assert_allclose(tb.getcol("CHAN_WIDTH")[0], inputs["spw_chan_width"])
    np.testing.assert_allclose(tb.getcol("REF_FREQUENCY"), MS_CHAN_FREQ[:, 0])
    np.testing.assert_array_equal(tb.getcol("NAME"), ["IF0", "IF1"])
    tb.close()
    tb.open(str(path / "FIELD"))
    np.testing.assert_array_equal(tb.getcol("NAME"), ["SRC_A", "SRC_B"])
    tb.close()
    tb.open(str(path / "OBSERVATION"))
    assert tb.nrows() == 1
    tb.close()
    tb.open(str(path / "HISTORY"))
    assert tb.nrows() == 0
    tb.close()


def test_write_history(written):
    """write_history appends one well-formed row to the HISTORY subtable."""
    path, _ = written
    caltable.write_history(path, "solved fringes")
    tb = casatools.table()
    tb.open(str(path / "HISTORY"))
    assert tb.nrows() == 1
    assert tb.getcell("MESSAGE", 0) == "solved fringes"
    assert tb.getcell("APPLICATION", 0) == "vlbipy" and tb.getcell("PRIORITY", 0) == "INFO"
    assert tb.getcell("TIME", 0) > 5.0e9
    assert tb.iscelldefined("APP_PARAMS", 0) and len(tb.getcell("APP_PARAMS", 0)) == 0
    tb.close()


def test_overwrite_existing(written, fake_ms):
    """Writing to an existing path replaces the table."""
    path, inputs = written
    inputs["fparam"] = np.zeros_like(inputs["fparam"])
    caltable.write_fringe_table(path, fake_ms, **inputs)
    out = caltable.read_fringe_table(path)
    assert np.all(out["FPARAM"] == 0)


def test_import_without_casatools_side_effects():
    """The module keeps casatools imports inside functions."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module("vlbipy.solvers.caltable"))
    top_level = [line for line in src.splitlines() if line.startswith("import ") or line.startswith("from ")]
    assert not any("casatools" in line for line in top_level)


def test_casacore_opens_and_taql(written):
    """python-casacore opens the table and TaQL can query the array columns."""
    tables = pytest.importorskip("casacore.tables")
    path, inputs = written
    t = tables.table(str(path), ack=False)
    assert set(t.colnames()) == set(caltable.MAIN_COLUMNS)
    assert t.nrows() == len(inputs["times"])
    t.close()
    sel = tables.taql(f"select from '{path}' where SNR[0,0] > 0")
    assert sel.nrows() == len(inputs["times"])
    sel.close()
    sel = tables.taql(f"select from '{path}' where ANTENNA1 == 1 and SPECTRAL_WINDOW_ID == 0")
    assert sel.nrows() == 2
    sel.close()


@pytest.mark.skipif(not REFERENCE_TABLE.exists(), reason=f"reference CASA caltable {REFERENCE_TABLE} not present")
def test_matches_reference_casa_table(written):
    """Column names/types/ndim, table info and keyword names equal those of a real CASA 6.7 fringefit table."""
    path, _ = written
    tb = casatools.table()
    tb.open(str(REFERENCE_TABLE))
    ref_desc, ref_info, ref_kw = tb.getdesc(), tb.info(), tb.getkeywords()
    ref_subs = {}
    for sub in ("ANTENNA", "FIELD", "OBSERVATION", "SPECTRAL_WINDOW", "HISTORY"):
        tb.close()
        tb.open(str(REFERENCE_TABLE / sub))
        ref_subs[sub] = tb.getdesc()
    tb.close()
    tb.open(str(path))
    mine_desc, mine_info, mine_kw = tb.getdesc(), tb.info(), tb.getkeywords()
    tb.close()

    ref_cols = {k for k in ref_desc if not k.startswith("_")}
    mine_cols = {k for k in mine_desc if not k.startswith("_")}
    assert mine_cols == ref_cols
    for col in ref_cols:
        for key in ("valueType", "ndim", "option", "dataManagerType", "dataManagerGroup"):
            assert mine_desc[col].get(key) == ref_desc[col].get(key), (col, key)
        assert set(mine_desc[col]["keywords"]) == set(ref_desc[col]["keywords"]), col
    assert mine_info == ref_info
    assert set(mine_kw) == set(ref_kw)
    for sub in ("SPECTRAL_WINDOW", "HISTORY"):
        tb.open(str(path / sub))
        mine_sub = tb.getdesc()
        tb.close()
        ref_cols = {k for k in ref_subs[sub] if not k.startswith("_")}
        assert {k for k in mine_sub if not k.startswith("_")} == ref_cols, sub
        for col in ref_cols:
            assert mine_sub[col]["valueType"] == ref_subs[sub][col]["valueType"], (sub, col)
            assert mine_sub[col].get("ndim") == ref_subs[sub][col].get("ndim"), (sub, col)
