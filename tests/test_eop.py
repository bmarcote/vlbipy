"""The native gencal eop (``vlbipy.solvers.eop``): sampling, EOP parsing and interpolation (no CASA, no data),
the table written for an RSM07 clone, and every cell against ``casatasks.gencal(caltype='eop')``."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import erfa
import numpy as np
import pytest

from vlbipy.casacore_format.table import Table
from vlbipy.solvers import eop as eop_module
from vlbipy.solvers.eop import eops_from_file, eops_from_ms, sample_times, eop

RSM07_MMS = Path(__file__).resolve().parents[1] / "test" / "RSM07_import_ref" / "RSM07.mms"
MAIN_COLUMNS = ("TIME", "FIELD_ID", "SPECTRAL_WINDOW_ID", "ANTENNA1", "ANTENNA2", "INTERVAL", "SCAN_NUMBER",
                "OBSERVATION_ID", "FPARAM", "PARAMERR", "FLAG", "SNR", "WEIGHT")


def usno_lines(mjds, pm_x, pm_y, ut1_utc, version="2.1") -> list[str]:
    """USNO finals lines as eop.py parses them (TAI Julian dates, PM x 10 [arcsec], UT1-TAI [us])."""
    lines = [f"EOP-MOD Ver {version}  test file\n", "# comment\n"]
    for mjd, x, y, ut in zip(mjds, pm_x, pm_y, ut1_utc):
        if version == "2.1":
            lines.append(f"{mjd + 2400000.5:9.1f} {x * 10:8.5f} {y * 10:8.5f} {(ut - 37.0) * 1e6:11.1f}  rest\n")
        else:
            lines.append(f"{mjd + 2400000.5:9.1f} {x * 10:7.4f} {y * 10:7.4f} {(ut - 37.0) * 1e6:9.0f}  rest\n")
    return lines


def write_usno_from_ms(ms: Path, path: Path) -> Path:
    """A USNO 2.1 file holding the MS EARTH_ORIENTATION values of observation 0 with small offsets."""
    eops = eops_from_ms(ms, 0)
    offsets = 0.0004 * (np.arange(eops["MJD"].size) % 3 - 1)
    path.write_text("".join(usno_lines(eops["MJD"], eops["PM_x"] + 0.0012, eops["PM_y"] - 0.0007,
                                       eops["UT1_UTC"] + offsets)))
    return path


def test_sample_times_follow_eop_py():
    times = np.array([100.0, 130.0, 161.0, 170.0, 300.0, 310.0])
    assert sample_times(times) == [100.0, 161.0, 300.0, 310.0]
    assert sample_times(np.array([100.0])) == [100.0]
    assert sample_times(np.array([30.0, 50.0])) == [50.0]            # first time <= 60 s after t0 = 0


def test_parse_usno_and_c04(tmp_path):
    mjds = [60000.0, 60001.0, 60002.0, 60003.0, 60004.0]
    for version in ("2.0", "2.1"):
        path = tmp_path / f"usno{version}.erp"
        path.write_text("".join(usno_lines(mjds, [0.1] * 5, [0.2] * 5, [0.05] * 5, version)))
        eops = eops_from_file(path, 60001.0, 60003.0)
        assert eops["PM_x"].tolist() == pytest.approx([0.1] * 3) and eops["PM_y"].tolist() == pytest.approx([0.2] * 3)
        assert eops["MJD"] == pytest.approx(np.array(mjds[1:4]) - 37.0 / 86400)        # TAI -> UTC
        assert eops["UT1_UTC"] == pytest.approx([0.05] * 3, abs=1e-6)
    c04 = tmp_path / "c04.txt"
    c04.write_text("  header line\n" + "".join(f"2023  1  1  {m:5d}  {0.1:9.6f}  {0.2:9.6f}  {0.05:10.7f}\n"
                                                 for m in range(60000, 60005)))
    eops = eops_from_file(c04, 60001, 60002)
    assert eops["MJD"].tolist() == [60001, 60002] and eops["UT1_UTC"].tolist() == [0.05, 0.05]
    assert eops_from_file(c04, 70000, 70001) is None


def test_interpolation_uses_the_entries_around_each_time_unlike_eop_py():
    eops = {"MJD": np.arange(6.0), "UT1_UTC": np.zeros(6), "PM_x": np.array([0.0, 1, 2, 3, 10, 20]),
            "PM_y": np.zeros(6)}
    xp, _ = eop_module.pm_xy(eops, 2400000.5, 4.5)
    assert xp == pytest.approx(15.0 * eop_module.ARCSEC)                 # between entries 4 and 5
    xp, _ = eop_module.pm_xy(eops, 2400000.5, 4.5, casa_compatible=True)
    assert xp == pytest.approx(4.5 * eop_module.ARCSEC)                  # CASA: extrapolated from entries 2 and 3
    eops["UT1_UTC"] = np.array([0.4, -0.5, -0.4, -0.3, -0.2, -0.1])      # leap second between entries 0 and 1
    assert eop_module.ut1_utc(eops, 2400000.5, 0.5) == pytest.approx(0.45)
    short = {name: values[:2] for name, values in eops.items()}
    assert eop_module.pm_xy(short, 2400000.5, 0.5)[0] == pytest.approx(0.5 * eop_module.ARCSEC)
    with pytest.raises(NotImplementedError):                           # CASA fails with an IndexError there
        eop_module.pm_xy(short, 2400000.5, 1.5, casa_compatible=True)
    with pytest.raises(NotImplementedError):                           # one entry cannot be interpolated
        eop_module.pm_xy({name: values[:1] for name, values in eops.items()}, 2400000.5, 0.5)


def test_keywords_and_infile(tmp_path):
    with pytest.raises(NotImplementedError):
        eop(tmp_path / "x.ms", tmp_path / "x.eop", infile=None)
    with pytest.raises(NotImplementedError):
        eop(tmp_path / "x.ms", tmp_path / "x.eop", infile=tmp_path / "missing.erp")
    path = tmp_path / "f.erp"
    path.write_text("")
    with pytest.raises(NotImplementedError):
        eop(tmp_path / "x.ms", tmp_path / "x.eop", infile=path, caltype="gc")
    with pytest.raises(NotImplementedError):
        eop(tmp_path / "x.ms", tmp_path / "x.eop", infile=path, caltype="eop", bogus=1)


@pytest.fixture(scope="module")
def rsm07(tmp_path_factory):
    """An APFS clone of the RSM07 Multi-MS, a USNO file built from it and the native table."""
    if not RSM07_MMS.is_dir():
        pytest.skip("RSM07 reference Multi-MS not available")
    work = tmp_path_factory.mktemp("eop")
    ms = work / "RSM07.mms"
    subprocess.run(["cp", "-cR", str(RSM07_MMS), str(ms)], check=True)
    infile = write_usno_from_ms(ms, work / "usno_finals.erp")
    report = eop(ms, work / "native.eop", infile=infile, caltype="eop", spw="", antenna="", pol="",
                             parameter=[], engine="native")
    eop(ms, work / "casa_compatible.eop", infile=infile, caltype="eop", engine="native",
                    casa_compatible=True)
    return {"work": work, "ms": ms, "infile": infile, "table": work / "native.eop", "report": report,
            "casa_compatible": work / "casa_compatible.eop"}


def test_rsm07_table_without_casa(rsm07):
    report = rsm07["report"]
    assert report["rows"] == 4 * report["samples"] and report["observations"] == 1 and not report["removed"]
    with Table(rsm07["table"]) as table:
        assert table.colnames() == list(MAIN_COLUMNS)
        assert table.getkeywords()["VisCal"] == "Fringe Jones"
        fparam = np.asarray(table.getcol("FPARAM"))
        spw, time = table.getcol("SPECTRAL_WINDOW_ID"), table.getcol("TIME")
        antenna, field = table.getcol("ANTENNA1"), table.getcol("FIELD_ID")
    with Table(rsm07["table"] / "SPECTRAL_WINDOW") as table:
        freq = np.asarray(table.getcol("CHAN_FREQ")).reshape(-1)
    assert spw[:4].tolist() == [0, 1, 2, 3]
    assert np.array_equal(fparam[:, 0, :4], fparam[:, 0, 4:]) and not fparam[:, 0, [3, 7]].any()
    # independent recomputation of the first row's delay (scalar erfa, eop.py's formula)
    old = eops_from_ms(rsm07["ms"], 0)
    new = eops_from_file(rsm07["infile"], old["MJD"].min(), old["MJD"].max())
    with Table(rsm07["ms"] / "ANTENNA") as table:
        pos = np.asarray(table.getcol("POSITION")).reshape(-1, 3)[antenna[0]]
    with Table(rsm07["ms"] / "FIELD") as table:
        lon, lat = np.asarray(table.getcol("PHASE_DIR")[field[0]]).reshape(-1, 2)[0]
    pc = erfa.s2c(lon, lat)
    delays = [eop_module.calc_delay(e, pos, pc, time[0] / 86400) for e in (new, old)]
    delay = delays[0] - delays[1]
    assert fparam[0, 0, 1] == pytest.approx(delay * 1e9, rel=1e-6)
    assert fparam[0, 0, 0] == pytest.approx(2 * math.pi * delay * freq[0], rel=1e-6)


def test_rsm07_matches_casa(rsm07, monkeypatch):
    casatasks = pytest.importorskip("casatasks")
    import casatools
    monkeypatch.chdir(rsm07["work"])
    reference = rsm07["work"] / "casa.eop"
    casatasks.gencal(vis=str(rsm07["ms"]), caltable=str(reference), caltype="eop", infile=str(rsm07["infile"]))
    tb = casatools.table()
    for sub in ("", "OBSERVATION", "ANTENNA", "FIELD", "SPECTRAL_WINDOW", "HISTORY"):
        tb.open(str(reference / sub))
        expected = {name: (tb.getcol(name) if tb.nrows() and tb.iscelldefined(name, 0) else None)
                    for name in tb.colnames()}
        names, keywords = tb.colnames(), tb.getkeywords()
        tb.close()
        tb.open(str(rsm07["casa_compatible"] / sub))
        assert tb.colnames() == names, sub
        for name, values in expected.items():
            if values is not None:
                got = tb.getcol(name)
                assert np.asarray(got).dtype == np.asarray(values).dtype and np.array_equal(got, values), (sub, name)
        if not sub:
            got_keywords = tb.getkeywords()
            assert {k: v for k, v in got_keywords.items() if k not in ("CASA_Version",) and "Table:" not in str(v)} \
                == {k: v for k, v in keywords.items() if k not in ("CASA_Version",) and "Table:" not in str(v)}
        tb.close()
    shutil.rmtree(reference)
