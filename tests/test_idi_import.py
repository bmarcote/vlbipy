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
from vlbipy.solvers.idi_import import import_fitsidi
from vlbipy.solvers.ms_metadata import ms_summary

REPO = Path(__file__).resolve().parents[1]
IDI_DIR = Path(os.environ.get("VLBIPY_RSM07_IDI", REPO / "test" / "RSM07"))
IDI_FILES = [IDI_DIR / "rsm07_1_1.IDI1", IDI_DIR / "rsm07_1_1.IDI2"]
REFERENCE = Path(os.environ.get("VLBIPY_RSM07_IMPORT_REF", REPO / "test" / "RSM07_import_ref")) / "RSM07.mms"
pytestmark = pytest.mark.skipif(not (all(f.is_file() for f in IDI_FILES) and (REFERENCE / "table.dat").is_file()),
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
