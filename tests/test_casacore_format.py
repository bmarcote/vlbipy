"""The casacore table format without casacore (:mod:`vlbipy.casacore_format`) and the native flag versions.

Pure-native round trips run everywhere; where python-casacore, casatools or casatasks are installed the same
tables are cross-checked against them (they are the reference implementation of the format).
"""
from __future__ import annotations

import numpy as np
import pytest

try:                       # casatools must load before python-casacore (both register the same TaQL functions)
    import casatools  # noqa: F401
except ImportError:
    pass

from vlbipy.casacore_format.aipsio import AipsReader, AipsWriter, TableRef
from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.tabledat import read_table_dat, serialize_table_dat
from vlbipy.casacore_format.writer import array_column, copy_table, create_table, rewrite_table, scalar_column

NROW = 300


def _values(rng):
    """Values for every kind of column the writer supports (C-order cells)."""
    return {"I": rng.integers(-100, 100, NROW).astype(np.int32), "D": rng.random(NROW),
            "B": rng.random(NROW) < 0.5, "S": ["x" * int(k) for k in rng.integers(0, 20, NROW)],
            "C": (rng.random(NROW) + 1j * rng.random(NROW)).astype(np.complex64),
            "FA": rng.random((NROW, 1, 8)).astype(np.float32), "BA": rng.random((NROW, 3, 5)) < 0.5,
            "V": [None if k % 7 == 0 else rng.random(k % 5 + 1) for k in range(NROW)],
            "SA": np.array([[f"a{k}", "b"] for k in range(NROW)], dtype=object),
            "SV": [np.array(["p", "qq", "rrr"][:k % 3 + 1], dtype=object) for k in range(NROW)],
            "DATA": (rng.standard_normal((NROW, 16, 4)) + 1j * rng.standard_normal((NROW, 16, 4))).astype(np.complex64),
            "FLAG": rng.random((NROW, 16, 4)) < 0.2}


def _columns():
    return [scalar_column("I", "int"), scalar_column("D", "double", keywords={"QuantumUnits": ["s"]}),
            scalar_column("B", "boolean"), scalar_column("S", "string"), scalar_column("C", "complex"),
            array_column("FA", "float", shape=(1, 8)), array_column("BA", "boolean", shape=(3, 5), direct=True),
            array_column("V", "double"), array_column("SA", "string", shape=(2,)), array_column("SV", "string"),
            array_column("DATA", "complex", shape=(16, 4)), array_column("FLAG", "boolean", shape=(16, 4))]


def _same(a, b) -> bool:
    if isinstance(a, dict) and "array" in a:                         # python-casacore's string arrays
        a = np.array(a["array"], dtype=object).reshape(a["shape"])
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return a.shape == b.shape and all(str(x) == str(y) for x, y in zip(a.reshape(-1), b.reshape(-1)))
    return a.shape == b.shape and np.array_equal(a, b)


@pytest.fixture
def written(tmp_path):
    rng = np.random.default_rng(5)
    values = _values(rng)
    path = create_table(tmp_path / "t.tab", _columns(), NROW, values,
                        keywords={"VisCal": "G Jones", "ANTENNA": TableRef("././ANTENNA")},
                        info=("Calibration", "G Jones"), tiled={"DATA": (16, 4), "FLAG": (16, 4)})
    return path, values


def test_aipsio_round_trips_nested_records():
    record = {"a": 1, "b": 2.5, "s": "text", "arr": np.arange(6, dtype=np.float64).reshape(2, 3),
              "names": ["x", "y"], "empty": [], "sub": {"flag": True, "c": complex(1, 2)},
              "table": TableRef("././SUB")}
    writer = AipsWriter()
    writer.table_record(record)
    reader = AipsReader(writer.getvalue())
    back = reader.table_record()
    assert back["a"] == 1 and back["b"] == 2.5 and back["s"] == "text" and back["names"] == ["x", "y"]
    np.testing.assert_array_equal(back["arr"], record["arr"])
    assert back["empty"] == [] and back["sub"] == {"flag": True, "c": complex(1, 2)}
    assert back["table"] == TableRef("././SUB")
    again = AipsWriter()
    again.table_record(back)
    assert again.getvalue() == writer.getvalue()                      # the remembered description is reused


def test_native_tables_read_back_natively(written):
    path, values = written
    with Table(path) as table:
        assert table.nrows() == NROW
        assert table.info() == {"type": "Calibration", "subType": "G Jones", "readme": ""}
        assert table.getkeywords()["ANTENNA"] == f"Table: {(path / 'ANTENNA').absolute()}"
        assert table.getcolkeywords("D") == {"QuantumUnits": ["s"]}
        for name, expected in values.items():
            got = table.getcol(name)
            if isinstance(expected, list):
                assert all(_same(g, e) for g, e in zip(got, expected)), name
            else:
                assert _same(got, expected), name
        np.testing.assert_array_equal(table.getcolslice("DATA", [2, 1], [9, 3], [3, 2], 10, 20),
                                      values["DATA"][10:30, 2:10:3, 1:4:2])
        np.testing.assert_array_equal(table.getcolslice("DATA", [0, 0], [-1, -1], None, 0, 5), values["DATA"][:5])


def test_table_dat_round_trips_byte_for_byte(written):
    path, _ = written
    raw = (path / "table.dat").read_bytes()
    assert serialize_table_dat(read_table_dat(path)) == raw


def test_in_place_writes_and_added_columns(written):
    path, values = written
    rng = np.random.default_rng(9)
    flag = rng.random((NROW, 16, 4)) < 0.5
    data = values["DATA"][50:80] * 2
    with Table(path, readonly=False) as table:
        table.putcol("FLAG", flag[:100], 0)
        table.putcol("FLAG", flag[100:], 100)
        table.putcol("DATA", data, 50)
        table.putcol("B", np.ones(10, dtype=bool), 7)
        table.add_column_like("CORRECTED_DATA", "DATA")         # tiled template: tiling copied
        table.add_column_like("FA_COPY", "FA")                  # StandardStMan template: new TiledShapeStMan
        table.putcol("CORRECTED_DATA", values["DATA"][:3], 297)
    with Table(path) as table:
        np.testing.assert_array_equal(table.getcol("FLAG"), flag)
        np.testing.assert_array_equal(table.getcol("DATA", 50, 30), data)
        np.testing.assert_array_equal(table.getcol("DATA", 0, 50), values["DATA"][:50])
        assert table.getcol("B", 7, 10).all()
        corrected = table.getcol("CORRECTED_DATA")
        assert corrected.shape == values["DATA"].shape and not corrected[:297].any()
        np.testing.assert_array_equal(corrected[297:], values["DATA"][:3])
        assert table.getcol("FA_COPY").shape == (NROW, 1, 8)


def test_copy_and_rewrite_subtables(written, tmp_path):
    path, values = written
    copy = copy_table(path, tmp_path / "copy.tab")
    rewrite_table(copy, lambda v, n: ({k: (c[::2] if isinstance(c, np.ndarray) else c[::2]) for k, c in v.items()},
                                      (n + 1) // 2))
    with Table(copy) as table:
        assert table.nrows() == NROW // 2
        np.testing.assert_array_equal(table.getcol("I"), values["I"][::2])
        assert table.getcol("S") == values["S"][::2]


def test_casacore_reads_native_tables(written):
    ct = pytest.importorskip("casacore.tables")
    path, values = written
    table = ct.table(str(path), ack=False)
    try:
        assert table.nrows() == NROW
        for name, expected in values.items():
            if isinstance(expected, list):
                got = [table.getcell(name, r) if table.iscelldefined(name, r) else None for r in range(NROW)]
                assert all(_same(g, e) for g, e in zip(got, expected)), name
            else:
                assert _same(table.getcol(name), expected), name
        assert table.info()["subType"] == "G Jones"
    finally:
        table.close()


def test_native_reads_casacore_tables(tmp_path):
    ct = pytest.importorskip("casacore.tables")
    rng = np.random.default_rng(2)
    desc = ct.maketabdesc([ct.makescacoldesc("TIME", 0.0), ct.makescacoldesc("ANT", 0),
                           ct.makescacoldesc("NAME", "x"),
                           ct.makearrcoldesc("DATA", 0j, shape=[8, 2], valuetype="complex"),
                           ct.makearrcoldesc("FLAG", False, shape=[8, 2]),
                           ct.makearrcoldesc("UVW", 0.0, shape=[3])])
    dminfo = {"*1": {"TYPE": "IncrementalStMan", "NAME": "ism", "SPEC": {}, "COLUMNS": ["TIME", "ANT"]},
              "*2": {"TYPE": "TiledShapeStMan", "NAME": "tsm", "SPEC": {"DEFAULTTILESHAPE": np.array([2, 8, 37])},
                     "COLUMNS": ["DATA"]},
              "*3": {"TYPE": "TiledShapeStMan", "NAME": "tflag", "SPEC": {"DEFAULTTILESHAPE": np.array([2, 8, 50])},
                     "COLUMNS": ["FLAG"]}}
    path = tmp_path / "cc.tab"
    table = ct.table(str(path), desc, dminfo=dminfo, nrow=NROW, ack=False)
    time = np.repeat(np.arange(NROW // 10, dtype=np.float64), 10)
    data = (rng.standard_normal((NROW, 8, 2)) + 1j * rng.standard_normal((NROW, 8, 2))).astype(np.complex64)
    flag = rng.random((NROW, 8, 2)) < 0.3
    table.putcol("TIME", time)
    table.putcol("ANT", np.arange(NROW, dtype=np.int32) % 4)
    table.putcol("NAME", [f"n{k}" for k in range(NROW)])
    table.putcol("DATA", data)
    table.putcol("FLAG", flag)
    table.putcol("UVW", rng.random((NROW, 3)))
    uvw = table.getcol("UVW")
    table.close()
    with Table(path) as native:
        np.testing.assert_array_equal(native.getcol("TIME"), time)
        np.testing.assert_array_equal(native.getcol("ANT", 5, 20), np.arange(5, 25) % 4)
        assert native.getcol("NAME", 3, 2) == ["n3", "n4"]
        np.testing.assert_array_equal(native.getcol("DATA", 30, 100), data[30:130])
        np.testing.assert_array_equal(native.getcol("FLAG"), flag)
        np.testing.assert_array_equal(native.getcol("UVW"), uvw)
    with Table(path, readonly=False) as native:
        native.putcol("FLAG", ~flag[60:70], 60)
        native.putcol("DATA", data[:5] * 3, 200)
    table = ct.table(str(path), ack=False)
    np.testing.assert_array_equal(table.getcol("FLAG", 60, 10), ~flag[60:70])
    np.testing.assert_array_equal(table.getcol("FLAG", 0, 60), flag[:60])
    np.testing.assert_array_equal(table.getcol("DATA", 200, 5), data[:5] * 3)
    table.close()


def test_rows_recorded_only_in_the_lock_file(tmp_path):
    pytest.importorskip("casacore.tables")
    from test_solvers_caltable import _lock_only_table
    _lock_only_table(tmp_path / "SPW", nrow=8)
    with Table(tmp_path / "SPW") as table:
        assert table.nrows() == 8
        np.testing.assert_array_equal(table.getcol("NUM_CHAN"), np.full(8, 32))


def test_casatools_compatible_tool(written):
    casatools = pytest.importorskip("casatools")
    from vlbipy.casacore_format.compat import table as native_tool
    path, _ = written
    reference, native = casatools.table(), native_tool()
    reference.open(str(path))
    native.open(str(path))
    try:
        for name in ("I", "D", "B", "S", "C", "FA", "BA", "SA", "DATA", "FLAG"):
            a, b = reference.getcol(name), native.getcol(name)
            assert a.shape == b.shape and (a.dtype == b.dtype or a.dtype.kind == b.dtype.kind == "U"), name
            assert np.array_equal(a, b), name
            a, b = reference.getcell(name, 3), native.getcell(name, 3)
            assert np.array_equal(np.asarray(a), np.asarray(b)), name
        assert reference.iscelldefined("V", 0) == native.iscelldefined("V", 0) is False
        assert reference.info() == native.info()
    finally:
        reference.close()
        native.close()


def test_flag_versions_save_restore_delete(written, tmp_path):
    from vlbipy.solvers import flagversions
    vis = tmp_path / "vis.ms"
    path, values = written
    create_table(vis, [array_column("FLAG", "boolean", shape=(16, 4)), scalar_column("FLAG_ROW", "boolean")], NROW,
                 {"FLAG_ROW": np.zeros(NROW, dtype=bool)}, tiled={"FLAG": (16, 4)})
    with Table(vis, readonly=False) as table:
        table.putcol("FLAG", values["FLAG"])
    flagversions.flagmanager(vis=str(vis), mode="save", versionname="v1", comment="first")
    with Table(vis, readonly=False) as table:
        table.putcol("FLAG", np.ones((NROW, 16, 4), dtype=bool))
    assert flagversions.flagmanager(vis=str(vis), mode="list")[0] == {"name": "v1", "comment": "first"}
    flagversions.flagmanager(vis=str(vis), mode="restore", versionname="v1")
    with Table(vis) as table:
        np.testing.assert_array_equal(table.getcol("FLAG"), values["FLAG"])
    flagversions.flagmanager(vis=str(vis), mode="save", versionname="v1", comment="again")   # old one renamed
    names = [name for name, _ in flagversions.list_versions(vis)]
    assert names[-1] == "v1" and names[0].startswith("v1.old.")
    flagversions.flagmanager(vis=str(vis), mode="delete", versionname="v1")
    assert [name for name, _ in flagversions.list_versions(vis)] == names[:1]


def test_flag_versions_interoperate_with_casa(tmp_path):
    """CASA restores a version saved natively and the other way round (on a real split Multi-MS, when the
    repository's RSM07 test data are present)."""
    import shutil
    from pathlib import Path

    casatasks = pytest.importorskip("casatasks")
    from vlbipy.solvers import flagversions
    source = Path(__file__).resolve().parent.parent / "test" / "e2e_daskms" / "calibrated_data" / "RSM07_3C345.ms"
    if not source.is_dir():
        pytest.skip("RSM07 test data not present")
    vis = tmp_path / source.name
    shutil.copytree(source, vis)
    parts = flagversions._parts(vis)

    def set_flags(value=None):
        for part in parts:
            with Table(part, readonly=False) as table:
                shape = (table.nrows(),) + table.getcol("FLAG", 0, 1).shape[1:]
                table.putcol("FLAG", np.full(shape, value, dtype=bool))

    with Table(vis) as table:
        original = table.getcol("FLAG")
    flagversions.save_version(vis, "native")
    casatasks.flagmanager(vis=str(vis), mode="save", versionname="casa")
    set_flags(True)
    casatasks.flagmanager(vis=str(vis), mode="restore", versionname="native")      # CASA reads ours
    with Table(vis) as table:
        np.testing.assert_array_equal(table.getcol("FLAG"), original)
    set_flags(False)
    flagversions.restore_version(vis, "casa")                                       # we read CASA's
    with Table(vis) as table:
        np.testing.assert_array_equal(table.getcol("FLAG"), original)
    assert [n for n, _ in flagversions.list_versions(vis)] == ["native", "casa"]


def test_tiled_column_without_defined_cells_opens_in_casacore(tmp_path):
    casatools = pytest.importorskip("casatools")
    columns = [scalar_column("TIME", "double"), array_column("DATA", "complex", ndim=2),
               array_column("FLAG_CATEGORY", "boolean", ndim=3)]
    create_table(tmp_path / "t.ms", columns, 3, {"TIME": np.arange(3.0)},
                 tiled={"DATA": (8, 2), "FLAG_CATEGORY": [None] * 3})
    tb = casatools.table()
    tb.open(str(tmp_path / "t.ms"))
    try:
        assert tb.nrows() == 3 and not tb.iscelldefined("FLAG_CATEGORY", 0)
        assert np.asarray(tb.getcol("DATA")).shape == (2, 8, 3)
    finally:
        tb.close()
