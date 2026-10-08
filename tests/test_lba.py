"""Tests for the LBA-specific pieces: tolerant UVFLG reading, file discovery, ANTAB quirks, campaign naming."""
from __future__ import annotations

import numpy as np

from vlbipy.campaign import campaign_name
from vlbipy.casavlbitools.fitsidi import fill_missing_bands, get_timetuple
from vlbipy.observatories.lba import LBAObservatory
from vlbipy.uvflg import parse_uvflg, uvflg_to_casa

UVFLG = """ant_name='AT' timerang=82,10,00,00, 82,10,01,46 reason='SLEWING' /
ant_name='AT' timerang=82,10,01,46, 82,10,01,46 reason='Mixed' /
!  Apriori edit data for v589a
opcode = 'FLAG'
dtimrang = 1  timeoff = 0
ant_name='Ho' timerang= 82,11,05,59,  82,11,06,02  reason='Slewing expected.' /
ant_name ='PA', timerang=082,23,03,06, 082,23,03,08 reason='DISH IS STATIONARY' /
ant_name='XX' timerang=82,10,00,00, 82,10,01,46 bif=2 eif=3 /
"""


def test_parse_uvflg_reads_every_dialect(tmp_path):
    path = tmp_path / "v589a.uvflg"
    path.write_text(UVFLG)
    records = parse_uvflg(path)
    assert [r["ANT_NAME"] for r in records] == ["AT", "AT", "Ho", "PA", "XX"]
    assert records[3]["TIMERANG"] == [82.0, 23.0, 3.0, 6.0, 82.0, 23.0, 3.0, 8.0]
    assert records[2]["REASON"] == "Slewing expected."


def test_uvflg_to_casa_drops_unknown_stations_and_empty_ranges(tmp_path):
    path = tmp_path / "v589a.uvflg"
    path.write_text(UVFLG)
    out = tmp_path / "v589a.flag"
    result = uvflg_to_casa(path, out, 2020, antennas=["AT", "HO", "PA"])
    lines = out.read_text().splitlines()
    assert result["written"] == len(lines) == 3
    assert result["unknown"] == ["XX"] and result["empty"] == 1
    assert lines[0] == "antenna='AT' timerange='2020/03/22/10:00:00~2020/03/22/10:01:46' reason='SLEWING'"
    assert lines[1].startswith("antenna='HO' ") and "reason='Slewing_expected'" in lines[1]
    assert result["seconds"]["AT"] == 106.0


def test_lba_finds_raw_files_beside_the_working_directory(tmp_path):
    for name in ("V589A.FITS", "v589a.antab", "v589a.uvflg", "V589AB.FITS", "V589B.FITS"):
        (tmp_path / name).write_text("")
    work = tmp_path / "V589A"
    work.mkdir()
    handler = LBAObservatory()
    assert handler.find_data_files("V589A", str(work)) == [str(tmp_path / "V589A.FITS")]
    assert handler.get_antab_file("V589A", str(work)) == str(tmp_path / "v589a.antab")
    assert handler.get_flag_file("V589A", str(work)) == str(tmp_path / "v589a.uvflg")
    (work / "v589a.flag").write_text("")
    assert handler.get_flag_file("V589A", str(work)) == str(work / "v589a.flag")


def test_antab_time_stamps_with_one_digit_hours_and_rounded_minutes():
    assert get_timetuple("7:04.8") == (7, 4, 48)
    assert get_timetuple("07:60.00") == (7, 60, 0)
    assert get_timetuple("10:00:13.5") == (10, 0, 13)


def test_fill_missing_bands_uses_the_level_of_the_bands_an_antenna_has():
    tsys = [[10.0, 20.0, -999.9, -999.9], [12.0, 22.0, -999.9, -999.9], [5.0, 5.0, 5.0, 5.0]]
    filled, report = fill_missing_bands(tsys, [7, 7, 8])
    assert report == {7: [2, 3]}
    assert np.allclose(filled[0], [10.0, 20.0, 15.0, 15.0]) and np.allclose(filled[2], 5.0)


def test_campaign_name_is_the_common_prefix():
    assert campaign_name(["V589A", "V589B", "V589C"]) == "V589"
    assert campaign_name(["AB1", "XY2"]) == "AB1+XY2"


def test_uvflg_records_of_hours_are_reported_not_applied(tmp_path):
    path = tmp_path / "x.uvflg"
    path.write_text("ant_name='AT' timerang=82,10,46,10, 82,23,03,40 reason='Mixed' /\n"
                    "ant_name='AT' timerang=82,10,00,00, 82,10,01,46 reason='SLEWING' /\n")
    result = uvflg_to_casa(path, tmp_path / "x.flag", 2020, antennas=["AT"], max_seconds=7200.0)
    assert result["written"] == 1 and len(result["too_long"]) == 1 and "12.3 h" in result["too_long"][0]


def test_campaign_report_is_outdated_by_a_recalibrated_epoch(tmp_path):
    import os
    from types import SimpleNamespace
    from vlbipy.campaign import _epochs_changed_since
    report = tmp_path / "V589.campaign.json"
    report.write_text("{}")
    chains = []
    for code, offset in (("V589A", -100), ("V589B", +100)):
        chain = tmp_path / f"{code}.caltables.json"
        chain.write_text("[]")
        stamp = report.stat().st_mtime + offset
        os.utime(chain, (stamp, stamp))
        chains.append(SimpleNamespace(project_code=code, _caltables_path=chain))
    assert _epochs_changed_since(chains, report) == ["V589B"]
