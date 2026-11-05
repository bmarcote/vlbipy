"""The native FITS-IDI import (:func:`vlbipy.solvers.idi_import.import_fitsidi`) end to end against CASA's
``importfitsidi`` + ``partition(createmms=True)`` of the same two RSM07 files.

The Multi-MS must have the same layout (parts, (scan, subband) pairs per part, symlinked and per-part
subtables), the same main-table rows in the same order with bit-identical values in every column, identical
subtables (HISTORY apart from time stamps and the version line), the same :func:`ms_summary`, and must open in
casatools with the same msmetadata. Known difference: the main-table UVW column is stored in a TiledShapeStMan
without the Direct option (CASA: TiledColumnStMan, Direct). Skipped without the data or the reference import.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.tabledat import read_table_dat, serialize_table_dat
from vlbipy.solvers import idi_import
from vlbipy.solvers.idi_import import import_fitsidi
from vlbipy.solvers.ms_metadata import ms_summary

REPO = Path(__file__).resolve().parents[1]
IDI_DIR = Path(os.environ.get("VLBIPY_RSM07_IDI", REPO / "test" / "RSM07"))
IDI_FILES = [IDI_DIR / "rsm07_1_1.IDI1", IDI_DIR / "rsm07_1_1.IDI2"]
REFERENCE = Path(os.environ.get("VLBIPY_RSM07_IMPORT_REF", REPO / "test" / "RSM07_import_ref")) / "RSM07.mms"
HAVE_RSM07 = all(f.is_file() for f in IDI_FILES)
needs_rsm07 = pytest.mark.skipif(not HAVE_RSM07, reason="RSM07 FITS-IDI files not available")
needs_reference = pytest.mark.skipif(not (HAVE_RSM07 and (REFERENCE / "table.dat").is_file()),
                                     reason="RSM07 FITS-IDI files or the CASA reference import not available")
#: HISTORY messages that depend on the CASA version or on where the import ran.
PATH_DEPENDENT = ("version:", "vis ", "outputvis ", "fitsidifile ")
#: msmetadata of a Multi-MS, printed as one repr (run in a subprocess started in a scratch directory).
MSMD_SCRIPT = """
import sys, casatools
md = casatools.msmetadata()
md.open(sys.argv[1])
scans = [int(s) for s in md.scannumbers()]
print(repr((md.nrows(), md.nfields(), md.nspw(), md.nantennas(), list(md.fieldnames()), scans,
            [md.spwsforscan(s).tolist() for s in scans], [md.fieldsforscan(s).tolist() for s in scans],
            [md.chanfreqs(s).tolist() for s in range(md.nspw())], md.timerangeforobs(0)["begin"]["m0"]["value"],
            md.timerangeforobs(0)["end"]["m0"]["value"])))
md.close()
"""


def _entries(directory: Path) -> list:
    """The tables of a directory: (name, symlink target or "DIR")."""
    return sorted((p.name, os.readlink(p) if p.is_symlink() else "DIR") for p in directory.iterdir()
                  if p.is_symlink() or (p / "table.dat").is_file())


def _description(path: Path) -> bytes:
    """table.dat without its data managers."""
    dat = read_table_dat(path)
    dat.managers, dat.seq_count, dat.nrow = [], 0, 0
    for column in dat.columns:
        column.seqnr, column.shape_col_def, column.shape_col = 0, False, ()
    return serialize_table_dat(dat)


def _same(a, b) -> bool:
    """Exact equality of values, lists and dicts (NaN equal to NaN, undefined cells as None)."""
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(
            _same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def _subtable_differences(ours: Path, casa: Path) -> list:
    """Differences between two subtables (HISTORY: the task rows' TIME and :data:`PATH_DEPENDENT` messages are
    not compared)."""
    problems = [] if _description(ours) == _description(casa) else ["description"]
    with Table(ours) as a, Table(casa) as b:
        if a.nrows() != b.nrows():
            return problems + ["nrow"]
        values = {name: (a.getcol(name), b.getcol(name)) for name in b.colnames()} if b.nrows() else {}
    keep = list(range(len(values["MESSAGE"][1]))) if "MESSAGE" in values else []
    if ours.name == "HISTORY":
        # the version line and the parameters holding paths (CASA's reference ran elsewhere) are not compared
        keep = [k for k in keep if not values["MESSAGE"][1][k].startswith(PATH_DEPENDENT)]
        task = [origin in ("importfitsidi", "partition") for origin in values["ORIGIN"][1]]
        for name, (x, y) in values.items():
            rows = [k for k in keep if not (name == "TIME" and task[k])]
            if not _same([x[k] for k in rows], [y[k] for k in rows]):
                problems.append(name)
        return problems
    return problems + [name for name, (x, y) in values.items() if not _same(x, y)]


def _main_differences(ours: Path, casa: Path) -> list:
    """Main-table columns of a part whose values (or units and measures) differ, read in blocks."""
    problems = []
    with Table(ours) as a, Table(casa) as b:
        if a.nrows() != b.nrows() or a.colnames() != b.colnames():
            return ["rows or columns"]
        for name in b.colnames():
            if a.coldesc(name).keywords != b.coldesc(name).keywords:
                problems.append(f"{name} keywords")
            if name == "FLAG_CATEGORY":                      # no cell defined, in either
                continue
            for start in range(0, b.nrows(), 50_000):
                count = min(50_000, b.nrows() - start)
                if not _same(a.getcol(name, start, count), b.getcol(name, start, count)):
                    problems.append(name)
                    break
    return problems


@needs_reference
def test_import_fitsidi_matches_casa(tmp_path):
    out = tmp_path / "RSM07.mms"
    try:
        report = import_fitsidi([str(f) for f in IDI_FILES], out, scan_gap=15.0)
        assert (report["rows"], report["parts"], report["scans"]) == (1399920, 8, 63)
        for name in ("table.dat", "table.info"):
            assert (out / name).read_bytes() == (REFERENCE / name).read_bytes()
        assert _entries(out) == _entries(REFERENCE)
        parts = sorted(p.name for p in (REFERENCE / "SUBMSS").iterdir())
        assert sorted(p.name for p in (out / "SUBMSS").iterdir()) == parts
        problems = {}
        for name in parts:
            ours, casa = out / "SUBMSS" / name, REFERENCE / "SUBMSS" / name
            assert _entries(ours) == _entries(casa)
            assert repr(read_table_dat(ours).keywords) == repr(read_table_dat(casa).keywords)
            assert (ours / "table.info").read_bytes() == (casa / "table.info").read_bytes()
            problems[name] = _main_differences(ours, casa)
            for sub, target in _entries(casa):
                if target == "DIR":
                    problems[f"{name}/{sub}"] = _subtable_differences(ours / sub, casa / sub)
        assert {key: value for key, value in problems.items() if value} == {}
        assert _same(ms_summary(out), ms_summary(REFERENCE))
        if subprocess.run([sys.executable, "-c", "import casatools"], cwd=tmp_path, capture_output=True,
                          check=False).returncode:
            return
        runs = [subprocess.run([sys.executable, "-c", MSMD_SCRIPT, str(path)], cwd=tmp_path, capture_output=True,
                               text=True, timeout=300, check=False) for path in (out, REFERENCE)]
        assert all(run.returncode == 0 for run in runs), runs[0].stderr[-2000:] + runs[1].stderr[-2000:]
        assert runs[0].stdout.strip().splitlines()[-1] == runs[1].stdout.strip().splitlines()[-1]
    finally:
        shutil.rmtree(out, ignore_errors=True)                # ~6 GB


def test_time_index_summary_uses_the_time_range_not_the_first_and_last_rows():
    index = {"time": np.array([200.0, 100.0, 300.0, 150.0]), "interval": np.array([2.0, 2.0, 4.0, 1.0], np.float32),
             "field": np.array([0, 1, 0, 1])}
    summary = idi_import.time_index_summary(index, {"nspw": 2}, np.array([1, 2, 1, 2]))
    assert (summary["time_start"], summary["time_end"]) == (100.0, 304.0)
    assert (summary["nfield"], summary["nspw"], summary["nscan"]) == (2, 2, 2)


def test_part_index_carries_only_the_rows_of_its_chunks():
    index = {"time": np.arange(10.0) * 3.0, "ant1": np.arange(10) % 4}
    chunks = [{"rows": np.array([7, 2, 5]), "band": 0, "spw": 0, "scan": 1},
              {"rows": np.array([7, 2, 5]), "band": 1, "spw": 1, "scan": 1}, {"rows": np.array([9]), "band": 0,
                                                                               "spw": 0, "scan": 3}]
    cut = idi_import.part_index(index, chunks)
    assert cut["index"]["time"].size == 4
    for original, local in zip(chunks, cut["chunks"]):
        assert {k: v for k, v in local.items() if k != "rows"} == {k: v for k, v in original.items() if k != "rows"}
        for key in index:
            assert np.array_equal(cut["index"][key][local["rows"]], index[key][original["rows"]])


def _rsm07_plan():
    """Setup, UV tables, row index and part plan of the RSM07 files (8 parts, as the import)."""
    from vlbipy.solvers.idi_subtables import read_idi_setup
    setup = read_idi_setup([str(f) for f in IDI_FILES])
    tables = idi_import.uv_tables([str(f) for f in IDI_FILES])
    index = idi_import.read_index(tables, setup)
    scans = idi_import.scan_numbers(index["time"], index["field"], index["array"], 15.0)
    parts = idi_import.plan_parts(index, scans, int(setup["nif"]), 8, int(setup["nchan"]) * int(setup["ncorr"]))
    return setup, tables, index, parts


@needs_rsm07
def test_part_written_from_the_cut_index_is_bit_identical(tmp_path):
    setup, tables, index, parts = _rsm07_plan()
    chunks = [dict(c, rows=c["rows"][:300]) for c in parts[0][:2]]      # small, two subbands of one scan
    idi_import.write_part_job(path=str(tmp_path / "full.ms"), tables=tables, chunks=chunks, index=index,
                              setup=setup)
    idi_import.write_part_job(path=str(tmp_path / "cut.ms"), tables=tables, setup=setup,
                              **idi_import.part_index(index, chunks))
    with Table(tmp_path / "full.ms") as full, Table(tmp_path / "cut.ms") as cut:
        assert full.nrows() == cut.nrows() == 600
        for name in full.colnames():
            if name != "FLAG_CATEGORY":
                assert np.array_equal(full.getcol(name), cut.getcol(name)), name


@needs_rsm07
def test_failed_import_removes_the_partial_multi_ms(tmp_path, monkeypatch):
    from vlbipy.solvers.workers import WorkerError

    def failing_run_jobs(*args, **kwargs):
        raise WorkerError("worker job write_part_job failed")
    monkeypatch.setattr(idi_import, "run_jobs", failing_run_jobs)
    out = tmp_path / "RSM07.mms"
    with pytest.raises(WorkerError):
        import_fitsidi([str(f) for f in IDI_FILES], out, scan_gap=15.0)
    assert not out.exists()


def test_uv_dtype_reads_one_element_tdim_columns_as_scalars():
    """A UV_DATA column with TDIM='(1)' (DiFX/VLBA) is a scalar field, so row reads give one value per row."""
    from astropy.io import fits
    header = fits.Header([("NAXIS1", 4 + 8 + 4 * 2 * 2), ("TFIELDS", 3),
                          ("TTYPE1", "INTTIM"), ("TFORM1", "1E"), ("TDIM1", "(1)"),
                          ("TTYPE2", "DATE"), ("TFORM2", "1D"),
                          ("TTYPE3", "FLUX"), ("TFORM3", "4E"), ("TDIM3", "(2,2)")])
    dtype, shapes = idi_import._uv_dtype(header)
    assert shapes["INTTIM"] == () and dtype["INTTIM"].shape == ()
    assert shapes["FLUX"] == (2, 2)
    rows = np.zeros(2, dtype=dtype)
    assert float(rows["INTTIM"][0]) == 0.0
