"""Tests for the observation summary report (diagnostics module)."""
from vlbipy.diagnostics import full_summary, scan_summary, write_summary
from vlbipy.models import Antenna, FreqSetup, ObsMetadata, Scan, Stokes
from vlbipy.sources import SourceSet


def _metadata():
    """Three scans over four stations: HH misses scan 2, DA never observes."""
    antennas = {name: Antenna(name=name, observed=name != "DA", subbands=(0, 1))
                for name in ("EF", "HH", "WB", "DA")}
    antennas["DA"].subbands = ()
    scans = [Scan(scan_number=1, source="3C345", time_start=0.0, time_end=100.0,
                  antennas=["EF", "HH", "WB"]),
             Scan(scan_number=2, source="J1848+3219", time_start=200.0, time_end=290.0,
                  antennas=["EF", "WB"]),
             Scan(scan_number=3, source="3C395", time_start=400.0, time_end=560.0,
                  antennas=["EF", "HH", "WB"])]
    freq = FreqSetup(ref_freq=1.6e9, total_bandwidth=1.28e8, n_subbands=2, n_channels=64,
                     channel_width=5e5, polarizations=[Stokes.RR, Stokes.LL])
    return ObsMetadata(project_code="TEST", time_range=(0.0, 560.0), antennas=antennas, scans=scans,
                       freq_setup=freq, source_names=["3C345", "J1848+3219", "3C395"],
                       source_coords={"3C345": (255.7, 39.8)}, source_ids={"3C345": 16})


def test_scan_table_keeps_one_column_per_antenna():
    rows = [line for line in scan_summary(_metadata()).splitlines() if line.startswith("    ")]
    assert rows[0].endswith("EF HH WB")
    assert rows[1].endswith("EF -- WB")   # HH absent, but still holds its column
    assert rows[2].endswith("EF HH WB")
    # The antenna lists all start at the same offset, so a station reads down one column.
    starts = [row.index("EF") for row in rows]
    assert len(set(starts)) == 1


def test_scan_table_counts_only_present_antennas():
    rows = [line for line in scan_summary(_metadata()).splitlines() if line.startswith("    ")]
    assert rows[0].split()[-4:-3] == ["3"]   # #Ant column, before the antenna list
    assert "  2  " in rows[1]


def test_scan_table_omits_antennas_without_data():
    """DA recorded nothing, so it gets no column (the antenna table still lists it)."""
    summary = scan_summary(_metadata())
    assert "DA" not in summary


def test_full_summary_has_every_section():
    sources = SourceSet.from_config({"targets": ["3C395"], "phase_calibrators": ["J1848+3219"],
                                     "fringe_finders": ["3C345"]})
    summary = full_summary(_metadata(), sources)
    for heading in ("VLBI Observation Summary: TEST", "Source Summary", "Frequency Setup",
                    "Antenna Summary", "Scan Summary"):
        assert heading in summary
    assert " 16  3C345" in summary          # field ID and role from the source set
    assert "fringefinder" in summary
    assert "DA" in summary                  # non-observing station is still reported


def test_write_summary_creates_the_file(tmp_path):
    out = write_summary(_metadata(), tmp_path / "sub" / "summary.md")
    assert out.exists()
    assert out.read_text().startswith("VLBI Observation Summary: TEST")


def test_summary_written_at_import_leaves_a_placeholder_for_the_general_comments():
    """The import step has nothing to comment on yet: the first section says when it gets filled in."""
    lines = full_summary(_metadata()).splitlines()
    assert lines[0] == "VLBI Observation Summary: TEST" and lines[3] == ""
    assert lines[4] == "General Comments for TEST" and lines[5] == "=" * 100
    assert "filled in once the pipeline has run" in lines[6]
    assert lines.index("General Comments for TEST") < lines.index("Source Summary for TEST")


def test_general_comments_are_the_first_section_of_the_summary(tmp_path):
    comments = [{"severity": "note", "category": "calibration", "antenna": "",
                 "text": "Scan 1 (3C345) was selected for the instrumental delay (reference antenna EF)."},
                {"severity": "warning", "category": "antenna", "antenna": "HH", "text": "no data in scan 2"},
                {"severity": "note", "category": "antenna", "antenna": "", "text": "Nothing to report for: EF, WB."}]
    out = write_summary(_metadata(), tmp_path / "summary.md", comments=comments)
    lines = out.read_text().splitlines()
    block = lines[lines.index("General Comments for TEST"):lines.index("Source Summary for TEST")]
    assert block[2] == "Calibration and results"
    assert block[3] == "  note     Scan 1 (3C345) was selected for the instrumental delay (reference antenna EF)."
    assert block[4:7] == ["Antennas", "  HH  WARNING  no data in scan 2", "  Nothing to report for: EF, WB."]
    assert "filled in once the pipeline has run" not in "\n".join(block)
    # The tables of the import step are all still there, after the comments.
    for heading in ("Source Summary", "Frequency Setup", "Antenna Summary", "Scan Summary"):
        assert any(line.startswith(heading) for line in lines[lines.index("Source Summary for TEST"):])
