"""The subtables of the native FITS-IDI import (:mod:`vlbipy.solvers.idi_subtables`) against CASA's.

* RSM07 (two FITS-IDI files): the subtables written for the first part of the Multi-MS are compared with those
  of ``importfitsidi`` + ``partition(createmms=True)`` (reference made beforehand, see ``REFERENCE``).
* Synthetic FITS-IDI files built from RSM07 (FLAG, WEATHER and PHASE-CAL tables, two files with their own
  SYSCAL/GAIN_CURVE/EOP rows and a later FEED validity, a SOURCE table with a gap, a proper motion and EPOCH,
  single-polarization tables): CASA is run on them in a subprocess and the outputs compared.

Tables must be identical (values bit for bit, column descriptions, keywords, table.info), HISTORY apart from
its time stamps and version line. Skipped when the data (or casatasks, for the synthetic case) are absent.
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.tabledat import read_table_dat, serialize_table_dat
from vlbipy.solvers.idi_subtables import read_idi_setup, write_part_history, write_subtables

REPO = Path(__file__).resolve().parents[1]
IDI_DIR = Path(os.environ.get("VLBIPY_RSM07_IDI", REPO / "test" / "RSM07"))
IDI_FILES = [IDI_DIR / "rsm07_1_1.IDI1", IDI_DIR / "rsm07_1_1.IDI2"]
#: importfitsidi + partition of IDI_FILES (constobsid=True, scanreindexgap_s=15, specframe GEO).
REFERENCE = Path(os.environ.get("VLBIPY_RSM07_IMPORT_REF", REPO / "test" / "RSM07_import_ref"))
REFERENCE_PART = REFERENCE / "RSM07.mms" / "SUBMSS" / "RSM07.mms.0000.ms"
HAVE_RSM07 = all(f.is_file() for f in IDI_FILES)
needs_rsm07 = pytest.mark.skipif(not HAVE_RSM07, reason="RSM07 FITS-IDI files not available")
needs_reference = pytest.mark.skipif(not (HAVE_RSM07 and (REFERENCE_PART / "table.dat").is_file()),
                                     reason="RSM07 FITS-IDI files or the CASA reference import not available")
#: Reads every defined cell of the tables named on the command line with python-casacore (run in a subprocess:
#: casacore and casatools must not be loaded into one process in the wrong order, and CASA logs land in cwd).
CASACORE_SCRIPT = """
import sys
import casacore.tables as ct
for path in sys.argv[1:]:
    table = ct.table(path, ack=False)
    for column in table.colnames():
        for row in range(table.nrows()):
            if table.iscelldefined(column, row):
                table.getcell(column, row)
    table.close()
"""


def _python_ok(code: str, cwd: Path) -> bool:
    """True when ``python -c code`` runs (a module is importable) in a subprocess started in ``cwd``."""
    try:
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=120, cwd=cwd)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _description_bytes(path: Path) -> bytes:
    """table.dat with the data managers left out: the table and column descriptions only."""
    dat = read_table_dat(path)
    dat.managers, dat.seq_count, dat.nrow = [], 0, 0
    for column in dat.columns:
        column.seqnr, column.shape_col_def, column.shape_col = 0, False, ()
    return serialize_table_dat(dat)


def _same(a, b) -> bool:
    """Exact equality of two column values (NaN equal to NaN; undefined cells as None)."""
    if isinstance(a, list) or isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def _differences(ours: Path, casa: Path) -> list[str]:
    """What differs between two tables (HISTORY: the rows of the task calls only by MESSAGE, version excepted)."""
    problems = []
    if _description_bytes(ours) != _description_bytes(casa):
        problems.append("column/table description")
    if (ours / "table.info").read_bytes() != (casa / "table.info").read_bytes():
        problems.append("table.info")
    with Table(ours) as a, Table(casa) as b:
        if repr(a.dat.keywords) != repr(b.dat.keywords):
            problems.append("keywords")
        if a.nrows() != b.nrows():
            return problems + [f"nrow {a.nrows()} != {b.nrows()}"]
        if not a.nrows():
            return problems
        values = {name: (a.getcol(name), b.getcol(name)) for name in b.colnames()}
    if ours.name == "HISTORY":
        task = [origin in ("importfitsidi", "partition") for origin in values["ORIGIN"][1]]
        keep = [k for k, message in enumerate(values["MESSAGE"][1]) if not message.startswith("version:")]
        for name, (x, y) in values.items():
            rows = [k for k in keep if not (name == "TIME" and task[k])]
            x, y = [x[k] for k in rows], [y[k] for k in rows]
            if not _same(x, y):
                problems.append(f"column {name}")
        return problems
    return problems + [f"column {name}" for name, (x, y) in values.items() if not _same(x, y)]


def _compare_part(ours: Path, casa: Path, names: list[str]) -> None:
    """Every subtable of the CASA part must be written, in CASA's keyword order, and be identical."""
    expected = [key for key in read_table_dat(casa).keywords if key != "MS_VERSION"]
    assert names == expected
    problems = {name: _differences(ours / name, casa / name) for name in names}
    assert {name: p for name, p in problems.items() if p} == {}


@needs_rsm07
def test_read_idi_setup_rsm07():
    setup = read_idi_setup([str(f) for f in IDI_FILES])
    assert setup["antenna_ids"] == {k: k - 1 for k in range(1, 15)}
    assert (setup["ncorr"], setup["nchan"], setup["nif"], setup["nspw"]) == (4, 64, 4, 4)
    assert setup["corr_types"] == [5, 6, 7, 8]                       # RR RL LR LL
    assert setup["corr_index"] == [0, 3, 1, 2]                       # FITS RR LL RL LR -> MS positions
    assert setup["corr_swap_index"] == [0, 3, 2, 1]
    assert np.array_equal(setup["eff_chan_bw"], np.full(4, 500000.0))
    assert (setup["vis_scale"], setup["weightyp"], setup["correlator"]) == (1.0, "", "SFXC")
    assert setup["uv_data_has_weights"] is True
    assert setup["chan_freq"].shape == (4, 64) and setup["chan_freq"][0, 0] == setup["ref_freq"] == 1594990000.0
    assert setup["nfield_from_source_table"] == 20 and setup["source_ids"] == list(range(1, 21))


@needs_reference
def test_rsm07_subtables_match_casa(tmp_path):
    part = tmp_path / "RSM07.mms" / "SUBMSS" / "RSM07.mms.0000.ms"
    part.mkdir(parents=True)
    files = [str(f) for f in IDI_FILES]
    with Table(REFERENCE / "RSM07.mms") as main:
        time, interval = main.getcol("TIME"), main.getcol("INTERVAL")
    summary = {"time_start": float(time.min()), "time_end": float((time + interval).max()), "nfield": 20,
               "nspw": 4, "nscan": 63}
    names = write_subtables(files, part, read_idi_setup(files), summary, specframe="GEO",
                            history_params={"vis": "RSM07.ms", "outputvis": "RSM07.mms"})
    _compare_part(part, REFERENCE_PART, names)
    other = tmp_path / "RSM07.mms" / "SUBMSS" / "RSM07.mms.0001.ms"
    other.mkdir()
    write_part_history(part, other)
    casa_other = REFERENCE / "RSM07.mms" / "SUBMSS" / "RSM07.mms.0001.ms"
    assert _differences(other / "HISTORY", casa_other / "HISTORY") == []
    if _python_ok("import casacore.tables", tmp_path):         # casacore reads every cell of what we wrote
        run = subprocess.run([sys.executable, "-c", CASACORE_SCRIPT] + [str(part / name) for name in names],
                             cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False)
        assert run.returncode == 0, run.stderr[-2000:]


# ---------------------------------------------------------------------------
# Synthetic FITS-IDI files (built from RSM07) for the tables RSM07 lacks
# ---------------------------------------------------------------------------
BLOCK = 2880
CASA_SCRIPT = """
import sys, casatasks
for name, files in ((sys.argv[1], sys.argv[2].split(",")), (sys.argv[3], sys.argv[4].split(","))):
    casatasks.importfitsidi(vis=name + ".ms", fitsidifile=files, constobsid=True, scanreindexgap_s=15.0,
                            specframe="GEO")
    casatasks.partition(vis=name + ".ms", outputvis=name + ".mms", createmms=True, separationaxis="auto",
                        numsubms="auto", flagbackup=False, datacolumn="all")
"""


def _raw_hdu(path, name: str) -> bytes:
    """The bytes (header and data) of one HDU of a FITS file."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdus:
        info = hdus[name].fileinfo()
    with open(path, "rb") as handle:
        handle.seek(info["hdrLoc"])
        return handle.read(info["datLoc"] - info["hdrLoc"] + info["datSpan"])


def _uv_rows(path, first: int, last: int) -> bytes:
    """The UV_DATA HDU of ``path`` cut to rows [first, last)."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdus:
        info, naxis1 = hdus["UV_DATA"].fileinfo(), int(hdus["UV_DATA"].header["NAXIS1"])
    with open(path, "rb") as handle:
        handle.seek(info["hdrLoc"])
        header = bytearray(handle.read(info["datLoc"] - info["hdrLoc"]))
        handle.seek(info["datLoc"] + first * naxis1)
        data = handle.read((last - first) * naxis1)
    for k in range(0, len(header), 80):
        if header[k:k + 8] == b"NAXIS2  ":
            header[k:k + 80] = f"NAXIS2  = {last - first:20d}".ljust(80).encode()
    return bytes(header) + data + b"\0" * ((-len(data)) % BLOCK)


def _table_bytes(columns: list, extname: str, extra: dict, template=None) -> bytes:
    """A binary-table HDU as bytes, with FITS-IDI's common keywords (or those of ``template``)."""
    from astropy.io import fits
    hdu = fits.BinTableHDU.from_columns(columns)
    cards = template.cards if template is not None else []
    skip = ("XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT", "TFIELDS")
    for card in cards:
        if card.keyword not in skip and not card.keyword.startswith(("TTYPE", "TFORM", "TUNIT", "TDIM")):
            hdu.header[card.keyword] = (card.value, card.comment)
    if template is None:
        for key, value in {"EXTNAME": extname, "TABREV": 1, "OBSCODE": "RSM07", "NO_STKD": 4, "STK_1": -1,
                           "NO_BAND": 4, "NO_CHAN": 64, "REF_FREQ": 1594990000.0, "CHAN_BW": 500000.0,
                           "REF_PIXL": 1.0, "RDATE": "2025-09-16", **extra}.items():
            hdu.header[key] = value
    buffer = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(buffer)
    return buffer.getvalue()[BLOCK:]


def _modified(path, name: str, change) -> bytes:
    """HDU ``name`` of ``path`` with its columns changed by ``change({column: array}) -> {column: array}``."""
    from astropy.io import fits
    with fits.open(path) as hdus:
        hdu = hdus[name]
        arrays = change({c.name: np.array(hdu.data[c.name]) for c in hdu.columns})
        columns = []
        for c in hdu.columns:
            if c.name in arrays:
                columns.append(fits.Column(name=c.name, format=c.format, unit=c.unit, array=arrays[c.name]))
        if "EPOCH" in arrays:
            columns.append(fits.Column(name="EPOCH", format="1D", unit="YEARS", array=arrays["EPOCH"]))
        return _table_bytes(columns, name, {}, template=hdu.header)


def _flag(rows: list) -> bytes:
    """A FLAG table: rows (source, array, ants, freqid, timerang, bands, chans, pflags, reason, severity)."""
    from astropy.io import fits
    formats = ("1J", "1J", "2J", "1J", "2E", "4J", "2J", "4J", "24A", "1J")
    names = ("SOURCE_ID", "ARRAY", "ANTS", "FREQID", "TIMERANG", "BANDS", "CHANS", "PFLAGS", "REASON", "SEVERITY")
    dtypes = {"TIMERANG": np.float32, "REASON": None}
    columns = [fits.Column(name=n, format=f, array=np.array([r[k] for r in rows], dtype=dtypes.get(n, np.int32)))
               for k, (n, f) in enumerate(zip(names, formats))]
    return _table_bytes(columns, "FLAG", {})


def _weather(t0: float, n: int, water: bool) -> bytes:
    """A WEATHER table of ``n`` rows from day ``t0`` (with WVR_H2O and IONOS_ELECTRON when ``water``)."""
    from astropy.io import fits
    rng = np.random.default_rng(int(t0 * 1000))
    columns = [fits.Column(name="TIME", format="1D", array=t0 + np.arange(n) * 0.001),
               fits.Column(name="TIME_INTERVAL", format="1E", array=np.full(n, 0.001, dtype=np.float32)),
               fits.Column(name="ANTENNA_NO", format="1J", array=(1 + np.arange(n) % 14).astype(np.int32))]
    for name, lo, hi in (("TEMPERATURE", 0, 20), ("PRESSURE", 990, 1010), ("DEWPOINT", -5, 5),
                         ("WIND_VELOCITY", 0, 9), ("WIND_DIRECTION", 0, 360)) + (
            (("WVR_H2O", 0, 1), ("IONOS_ELECTRON", 5, 6)) if water else ()):
        columns.append(fits.Column(name=name, format="1E", array=rng.uniform(lo, hi, n).astype(np.float32)))
    return _table_bytes(columns, "WEATHER", {})


def _phase_cal(t0: float, n: int, dual: bool, ntones: int) -> bytes:
    """A PHASE-CAL table of ``n`` rows (``ntones`` tones per band, one or two polarizations)."""
    from astropy.io import fits
    rng = np.random.default_rng(int(t0 * 7000))
    width = 4 * ntones
    columns = [fits.Column(name="TIME", format="1D", array=t0 + np.arange(n) * 0.0005),
               fits.Column(name="TIME_INTERVAL", format="1E", array=np.full(n, 0.0005, dtype=np.float32)),
               fits.Column(name="SOURCE_ID", format="1J", array=np.full(n, 17, dtype=np.int32)),
               fits.Column(name="ANTENNA_NO", format="1J", array=(1 + np.arange(n) % 14).astype(np.int32)),
               fits.Column(name="ARRAY", format="1J", array=np.ones(n, dtype=np.int32)),
               fits.Column(name="FREQID", format="1J", array=np.ones(n, dtype=np.int32)),
               fits.Column(name="CABLE_CAL", format="1D", array=rng.normal(0, 1e-9, n))]
    for pol in ("1", "2") if dual else ("1",):
        columns += [fits.Column(name=f"PC_FREQ_{pol}", format=f"{width}D",
                                array=1.6e9 + rng.uniform(0, 1e8, (n, width))),
                    fits.Column(name=f"PC_REAL_{pol}", format=f"{width}E",
                                array=rng.normal(0, 1, (n, width)).astype(np.float32)),
                    fits.Column(name=f"PC_IMAG_{pol}", format=f"{width}E",
                                array=rng.normal(0, 1, (n, width)).astype(np.float32))]
    return _table_bytes(columns, "PHASE-CAL", {"NO_POL": 2 if dual else 1, "NO_TONES": ntones})


def _write_fits(path: Path, pieces: list) -> str:
    """Concatenate HDU byte strings into a FITS file."""
    with open(path, "wb") as handle:
        for piece in pieces:
            assert len(piece) % BLOCK == 0
            handle.write(piece)
    return str(path)


def _synthetic_files(work: Path) -> dict:
    """Three small FITS-IDI files: a + b (imported together) and c (alone)."""
    idi1, idi2 = IDI_FILES
    with open(idi2, "rb") as handle:
        primary = handle.read(BLOCK)
    base = [primary, _raw_hdu(idi2, "ARRAY_GEOMETRY"), _raw_hdu(idi2, "FREQUENCY")]

    def skip_row(arrays):                       # a gain curve the filler skips (distinct types per polarization)
        arrays["TYPE_1"][3, 2] = 1
        return arrays

    def later_antennas(arrays):                 # FEED rows valid at another time: concatenation widens them
        arrays["TIME"][:], arrays["TIME_INTERVAL"][:] = 0.65, 0.05
        return arrays

    def sources_c(arrays):                      # a missing source id, a proper motion, EPOCH instead of EQUINOX
        keep = arrays["SOURCE_ID"] != 10
        arrays = {k: v[keep] for k, v in arrays.items() if k != "EQUINOX"}
        arrays["PMRA"][4], arrays["PMDEC"][4] = 1e-6, -2e-7
        arrays["EPOCH"] = np.full(int(keep.sum()), 2000.0)
        return arrays

    flags_a = _flag([(0, 0, [0, 0], -1, [0.60, 0.61], [1, 1, 1, 1], [0, 0], [1, 1, 1, 1], "ALL", -1),
                     (17, 0, [3, 0], 1, [0.601, 0.602], [1, 0, 1, 0], [5, 10], [1, 1, 0, 0], "SOME", 2),
                     (0, 0, [2, 5], 0, [0.61, 0.62], [0, 0, 0, 1], [0, 0], [0, 0, 1, 1], "CROSS", 0),
                     (0, 1, [2, 5], 0, [0.61, 0.62], [0, 0, 0, 1], [0, 0], [0, 0, 1, 1], "ARRAY1", 0),
                     (16, 0, [1, 2], 1, [0.63, 0.64], [1, 1, 1, 1], [7, 7], [1, 1, 1, 1], "CHANNEL", 1)])
    a = _write_fits(work / "syn_a.IDI", base + [
        _raw_hdu(idi2, "SOURCE"), _raw_hdu(idi2, "ANTENNA"), _raw_hdu(idi2, "CALC"), _uv_rows(idi2, 0, 600),
        _modified(idi1, "SYSTEM_TEMPERATURE", lambda x: {k: v[:60] for k, v in x.items()}),
        _modified(idi1, "GAIN_CURVE", skip_row), flags_a, _weather(0.60, 40, True), _phase_cal(0.60, 30, True, 2)])
    b = _write_fits(work / "syn_b.IDI", base + [
        _raw_hdu(idi2, "SOURCE"), _modified(idi2, "ANTENNA", later_antennas), _raw_hdu(idi2, "CALC"),
        _uv_rows(idi2, 600, 1200),
        _modified(idi1, "SYSTEM_TEMPERATURE", lambda x: {k: v[100:140] for k, v in x.items()}),
        _modified(idi1, "GAIN_CURVE", lambda x: {k: v[:5] for k, v in x.items()}),
        _flag([(0, 0, [0, 0], -1, [0.70, 0.71], [1, 1, 1, 1], [0, 0], [1, 1, 1, 1], "B", -1)]),
        _weather(0.70, 25, True), _phase_cal(0.70, 20, True, 2)])
    c = _write_fits(work / "syn_c.IDI", base + [
        _modified(idi2, "SOURCE", sources_c), _raw_hdu(idi2, "ANTENNA"), _uv_rows(idi2, 0, 300),
        _modified(idi1, "SYSTEM_TEMPERATURE", lambda x: {k: v[:30] for k, v in x.items() if not k.endswith("_2")}),
        _weather(0.60, 10, False), _phase_cal(0.60, 10, False, 3)])
    return {"syn_ab": [a, b], "syn_c": [c]}


@needs_rsm07
def test_synthetic_subtables_match_casa(tmp_path):
    pytest.importorskip("astropy")
    if not _python_ok("import casatasks", tmp_path):
        pytest.skip("casatasks not available")
    cases = _synthetic_files(tmp_path)
    arguments = ["syn_ab", ",".join(cases["syn_ab"]), "syn_c", ",".join(cases["syn_c"])]
    run = subprocess.run([sys.executable, "-c", CASA_SCRIPT] + arguments, cwd=tmp_path, capture_output=True,
                         text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stderr[-2000:]
    written = {}
    for name, files in cases.items():
        casa = tmp_path / f"{name}.mms" / "SUBMSS" / f"{name}.mms.0000.ms"
        ours = tmp_path / "native" / f"{name}.mms" / "SUBMSS" / f"{name}.mms.0000.ms"
        ours.mkdir(parents=True)
        written[name] = write_subtables(files, ours, read_idi_setup(files), {}, specframe="GEO",
                                        history_params={"vis": f"{name}.ms", "outputvis": f"{name}.mms"})
        _compare_part(ours, casa, written[name])
    assert {"WEATHER", "PHASE_CAL", "SYSCAL", "GAIN_CURVE", "EARTH_ORIENTATION"} <= set(written["syn_ab"])
    with Table(tmp_path / "native" / "syn_ab.mms" / "SUBMSS" / "syn_ab.mms.0000.ms" / "FLAG_CMD") as flags:
        assert flags.nrows() == 4                     # the ARRAY 1 row is skipped, the second file's is dropped
