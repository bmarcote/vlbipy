"""Tests for the VLBA import: <CODE>.idifits discovery, FITS-IDI checks, and the FLAG_CMD a-priori flags."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from vlbipy import VLBIObs, fitsidi
from vlbipy.flag_cmd import command_from_row, commands_from_flag_cmd, format_time
from vlbipy.logging_utils import warnings
from vlbipy.observatories.vlba import VLBAObservatory
from vlbipy.solvers.flagdata import parse_command, parse_timerange

ANTENNAS = ["BR", "FD", "HN", "KP"]
#: 2023-10-02 16:00:00 UTC in MJD seconds.
T0 = 60219.0 * 86400.0 + 16 * 3600.0


def _table(name: str, columns: list) -> fits.BinTableHDU:
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header["EXTNAME"] = name
    return hdu


def write_idi(path: Path, *, tsys_antennas=(1, 2, 3, 4), gc_antennas=(1, 2, 3, 4), flags: bool = True) -> Path:
    """A minimal VLBA-like FITS-IDI file: ARRAY_GEOMETRY, a UV_DATA stand-in, SYSTEM_TEMPERATURE, GAIN_CURVE, FLAG."""
    hdus = [fits.PrimaryHDU(),
            _table("ARRAY_GEOMETRY", [fits.Column("ANNAME", "8A", array=ANTENNAS),
                                      fits.Column("NOSTA", "J", array=np.arange(1, 5))]),
            _table("UV_DATA", [fits.Column("FLUX", "64E", array=np.zeros((200, 64)))])]
    if tsys_antennas:
        hdus.append(_table("SYSTEM_TEMPERATURE", [fits.Column("ANTENNA_NO", "J", array=np.array(tsys_antennas))]))
    if gc_antennas:
        hdus.append(_table("GAIN_CURVE", [fits.Column("ANTENNA_NO", "J", array=np.array(gc_antennas))]))
    if flags:
        hdus.append(_table("FLAG", [fits.Column("ANTS", "2J", array=np.array([[1, 0], [2, 3]]))]))
    fits.HDUList(hdus).writeto(path)
    return path


def anomalies() -> list[str]:
    return [item for item in warnings.summary() if item.startswith("ANOMALY")]


# -- discovery and checks of the FITS-IDI file --

def test_vlba_finds_code_idifits_beside_the_working_directory(tmp_path):
    work = tmp_path / "BM538"
    work.mkdir()
    write_idi(tmp_path / "VLBA_BM538B_bm538b_BIN0_SRC0_0_231002T155741.idifits")
    (tmp_path / "BM538B.idifits").write_text("")          # another epoch's code must not match
    handler = VLBAObservatory()
    assert handler.find_data_files("BM538", str(work)) == []
    (tmp_path / "BM538.idifits").symlink_to(tmp_path / "VLBA_BM538B_bm538b_BIN0_SRC0_0_231002T155741.idifits")
    assert handler.find_data_files("BM538", str(work)) == [str(tmp_path / "BM538.idifits")]
    assert handler.find_data_files("bm538", str(work)) == [str(tmp_path / "BM538.idifits")]
    write_idi(work / "bm538.IDIFITS")                     # nearest directory wins, any case
    assert handler.find_data_files("BM538", str(work)) == [str(work / "bm538.IDIFITS")]
    assert handler.get_antab_file("BM538", str(work)) is None and handler.get_flag_file("BM538", str(work)) is None
    assert handler.flags_in_fitsidi


def test_prepare_accepts_a_complete_file_with_tsys_gc_and_flags(tmp_path):
    path = write_idi(tmp_path / "BM538.idifits")
    warnings.reset()
    assert VLBAObservatory().prepare_for_import([str(path)], str(tmp_path), project_code="BM538") == [str(path)]
    assert anomalies() == []
    assert fitsidi.table_rows(path)["FLAG"] == 2


def test_prepare_reports_missing_tables_and_uncovered_antennas(tmp_path):
    path = write_idi(tmp_path / "BM538.idifits", gc_antennas=(1, 2, 3), flags=False)
    warnings.reset()
    VLBAObservatory().prepare_for_import([str(path)], str(tmp_path), project_code="BM538")
    found = anomalies()
    assert any("no FLAG table rows" in a for a in found)
    assert any("no gain curve for KP" in a for a in found)
    assert not any("no Tsys" in a for a in found)
    assert fitsidi.apriori_coverage(path) == {"Tsys": [], "gain curve": ["KP"]}


def test_prepare_refuses_a_file_still_downloading(tmp_path):
    path = write_idi(tmp_path / "BM538.idifits")
    complete = path.stat().st_size
    assert fitsidi.expected_size(path) == complete
    with open(path, "r+b") as handle:
        handle.truncate(complete - 2880 * 3)
    with pytest.raises(ValueError, match="incomplete"):
        VLBAObservatory().prepare_for_import([str(path)], str(tmp_path), project_code="BM538")


# -- FLAG_CMD rows -> flagdata commands --

def test_flag_cmd_row_gets_time_range_and_names():
    names, fields = ["BR", "FD", "HN"], ["3C345", "J1848+3219"]
    line = command_from_row(T0 + 30.0, 60.0, "antenna='1' field='1' spw='0,1:4~9'' correlation='RR'", names, fields)
    assert line == ("antenna='FD' field='J1848+3219' spw='0,1:4~9' correlation='RR' "
                    "timerange='2023/10/02/16:00:00.000~2023/10/02/16:01:00.000'")
    parsed = parse_command(line)                      # readable by the native flagger as well as CASA
    assert parsed["antenna"] == "FD" and parse_timerange(parsed["timerange"]) == [(T0, T0 + 60.0)]
    assert command_from_row(T0, 2.0, "antenna='0&2'", names, fields).startswith("antenna='BR&HN' ")
    # A flag without any selection still covers only its own time range.
    assert command_from_row(T0, 2.0, "", names, fields) == \
        "timerange='2023/10/02/15:59:59.000~2023/10/02/16:00:01.000'"
    assert format_time(T0 + 0.25) == "2023/10/02/16:00:00.250"


def test_flag_cmd_rows_never_widen_to_the_whole_observation():
    result = commands_from_flag_cmd([T0, T0, T0], [10.0, 0.0, 10.0], ["antenna='0'", "antenna='1'", "antenna='9'"],
                                    ["BR", "FD"], ["3C345"])
    assert len(result["commands"]) == 1 and all("timerange=" in c for c in result["commands"])
    assert len(result["skipped"]) == 2 and result["per_antenna"] == {"BR": 1}


class _Table:
    """Stand-in for casatools.table over a dict of subtables {name: {column: array}}."""

    def __init__(self, tables: dict) -> None:
        self.tables, self.name = tables, ""

    def open(self, path, nomodify=True):
        self.name = Path(path).name
        return self.name in self.tables

    def nrows(self):
        return len(next(iter(self.tables[self.name].values())))

    def getcol(self, column):
        return self.tables[self.name][column]

    def putcol(self, column, values):
        self.tables[self.name][column] = values

    def close(self):
        pass


def test_casa_flag_ops_apply_flag_cmd_and_mark_it_applied(tmp_path):
    from types import SimpleNamespace

    from vlbipy.backends.casa import CasaFlagOps
    ms = tmp_path / "bm538.ms"
    (ms / "FLAG_CMD").mkdir(parents=True)
    tables = {"FLAG_CMD": {"TIME": np.array([T0, T0 + 100.0]), "INTERVAL": np.array([4.0, 20.0]),
                           "COMMAND": np.array(["antenna='0'", "antenna='1' spw='2'"]),
                           "APPLIED": np.zeros(2, dtype=bool)},
              "ANTENNA": {"NAME": np.array(["BR", "FD"])}, "FIELD": {"NAME": np.array(["3C345"])}}
    calls = []
    backend = SimpleNamespace(ms_path=lambda code: ms, tools=SimpleNamespace(table=lambda: _Table(tables)),
                              tasks=SimpleNamespace(flagdata=lambda **kw: calls.append(kw)))
    ops = CasaFlagOps(backend)
    assert ops._apply_fitsidi_flags("BM538") == 2
    assert calls[0]["mode"] == "list" and calls[0]["action"] == "apply"
    assert calls[0]["inpfile"][1] == ("antenna='FD' spw='2' "
                                      "timerange='2023/10/02/16:01:30.000~2023/10/02/16:01:50.000'")
    assert tables["FLAG_CMD"]["APPLIED"].all()
    tables["FLAG_CMD"] = {key: value[:0] for key, value in tables["FLAG_CMD"].items()}
    warnings.reset()
    assert ops._apply_fitsidi_flags("BM538") == 0 and any("no FLAG_CMD rows" in a for a in anomalies())


def test_apriori_on_vlba_applies_the_fitsidi_flags_not_a_flag_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obs = VLBIObs("vb01", network="VLBA", backend="dummy", target="3C395", phasecal="J1848+3219")
    obs.import_data()
    warnings.reset()
    obs["vb01"].flag.apriori()
    assert obs["vb01"]._state.status("flag_fitsidi") == "done"
    assert not any("no a-priori flag file" in a for a in anomalies())


def test_one_element_tdim_columns_read_as_scalars(tmp_path):
    """DiFX/VLBA files give one-element columns TDIM='(1)', which astropy reads as (nrow, 1): NumPy >= 2.5 refuses
    float()/int() on those cells ("only 0-dimensional arrays can be converted to Python scalars")."""
    one = {"dim": "(1)"}
    hdus = [fits.PrimaryHDU(),
            _table("ARRAY_GEOMETRY", [fits.Column("ANNAME", "8A", array=ANTENNAS),
                                      fits.Column("NOSTA", "1J", array=np.arange(1, 5).reshape(4, 1), **one),
                                      fits.Column("DIAMETER", "1E", array=np.full((4, 1), 25.0), **one)]),
            _table("SOURCE", [fits.Column("SOURCE", "16A", array=["J0000+0000"]),
                              fits.Column("RAEPO", "1D", array=np.array([[-10.0]]), **one),
                              fits.Column("DECEPO", "1D", array=np.array([[45.0]]), **one)]),
            _table("SYSTEM_TEMPERATURE", [fits.Column("ANTENNA_NO", "1J", array=np.array([[1], [2]]), **one)])]
    path = tmp_path / "vb02.idifits"
    fits.HDUList(hdus).writeto(path)
    with fits.open(path) as hdulist:
        assert hdulist["ARRAY_GEOMETRY"].data["NOSTA"].shape == (4, 1)        # the layout this test is about
        antennas = fitsidi._read_antennas(hdulist)
        names, coords = fitsidi._read_sources(hdulist)
    assert [a.diameter for a in antennas.values()] == [25.0] * 4
    assert names == ["J0000+0000"] and coords["J0000+0000"] == (350.0, 45.0)
    assert fitsidi.apriori_coverage(path) == {"Tsys": ["HN", "KP"], "gain curve": ANTENNAS}
