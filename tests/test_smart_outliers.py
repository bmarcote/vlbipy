"""Tests of ``vlbipy flag smart-outliers`` (:mod:`vlbipy.solvers.smart_outliers`, statistics in difmapy.outliers).

A small synthetic measurement set, written without casacore, alternates a phase calibrator (CAL) and a target
(TGT) scan by scan. Antenna A3 loses a whole calibrator scan, antenna A1 has a short burst in another: the
commands must flag A3 over that scan and over the target scans on either side of it, A1 over the burst only,
and applying them through the native flagdata must flag exactly those rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("difmapy.outliers")

from vlbipy.casacore_format.table import Table  # noqa: E402
from vlbipy.casacore_format.writer import array_column, create_table, scalar_column  # noqa: E402
from vlbipy.solvers import smart_outliers as so  # noqa: E402
from vlbipy.solvers.flagdata import apply_flag_selections, compile_flagdata, read_flag_setup  # noqa: E402

NANT, NCHAN, NCORR, NBIN = 5, 16, 2, 8
CAL_SCANS = 12                                  # CAL, TGT, CAL, TGT, ... CAL: 12 calibrator and 11 target scans
T0 = 5.0e9
DROPPED_SCAN = 6                                # calibrator scan A3 loses entirely
BURST_SCAN, BURST_BINS = 3, (2, 3, 4)           # calibrator scan and bins of A1's burst


def _layout() -> list[tuple[int, int, float]]:
    """(scan number, field, start time) in time order: calibrator scans 1, 3, 5, ..., target scans in between."""
    scans, start = [], T0
    for k in range(2 * CAL_SCANS - 1):
        scans.append((k + 1, 0 if k % 2 == 0 else 1, start))
        start += NBIN * 15.0 + 60.0
    return scans


def _write_ms(path: Path) -> dict:
    rng = np.random.default_rng(7)
    pairs = [(i, j) for i in range(NANT) for j in range(i + 1, NANT)]
    rows = []
    for scan, field, start in _layout():
        for b in range(NBIN):
            for i, j in pairs:
                rows.append((scan, field, start + 15.0 * b + 7.5, b, i, j))
    scan, field, time, step, ant1, ant2 = (np.array(column) for column in zip(*rows))
    nrow = len(rows)
    level = np.where(field == 0, 1.0, 0.5)[:, None, None] * np.exp(0.3j)
    noise = (rng.normal(size=(nrow, NCHAN, NCORR)) + 1j * rng.normal(size=(nrow, NCHAN, NCORR))) * 0.02
    data = level * np.ones((nrow, NCHAN, NCORR)) + noise
    cal_number = (scan + 1) // 2                                     # calibrator scans 1, 3, 5 -> 1, 2, 3
    dropped = (field == 0) & (cal_number == DROPPED_SCAN) & ((ant1 == 3) | (ant2 == 3))
    burst = (field == 0) & (cal_number == BURST_SCAN) & np.isin(step, BURST_BINS) & ((ant1 == 1) | (ant2 == 1))
    data[dropped] *= 0.05
    data[burst] *= 3.0
    columns = [scalar_column("TIME", "double"), scalar_column("ANTENNA1", "int"), scalar_column("ANTENNA2", "int"),
               scalar_column("FIELD_ID", "int"), scalar_column("SCAN_NUMBER", "int"),
               scalar_column("DATA_DESC_ID", "int"), array_column("DATA", "complex", shape=(NCHAN, NCORR)),
               array_column("FLAG", "boolean", shape=(NCHAN, NCORR))]
    values = {"TIME": time, "ANTENNA1": ant1.astype(np.int32), "ANTENNA2": ant2.astype(np.int32),
              "FIELD_ID": field.astype(np.int32), "SCAN_NUMBER": scan.astype(np.int32),
              "DATA_DESC_ID": np.zeros(nrow, np.int32), "DATA": data.astype(np.complex64),
              "FLAG": np.zeros((nrow, NCHAN, NCORR), dtype=bool)}
    create_table(path, columns, nrow, values, tiled={"DATA": (NCHAN, NCORR), "FLAG": (NCHAN, NCORR)})
    create_table(path / "ANTENNA", [scalar_column("NAME", "string"), array_column("POSITION", "double", shape=(3,)),
                                    scalar_column("MOUNT", "string")], NANT,
                 {"NAME": [f"A{i}" for i in range(NANT)], "POSITION": rng.normal(0, 6e6, (NANT, 3)),
                  "MOUNT": ["ALT-AZ"] * NANT})
    create_table(path / "FIELD", [scalar_column("NAME", "string"), array_column("PHASE_DIR", "double", shape=(1, 2))],
                 2, {"NAME": ["CAL", "TGT"], "PHASE_DIR": np.array([[[0.1, 0.5]], [[0.11, 0.51]]])})
    create_table(path / "SPECTRAL_WINDOW", [array_column("CHAN_FREQ", "double", shape=(NCHAN,))], 1,
                 {"CHAN_FREQ": np.array([5.0e9 + 1e6 * np.arange(NCHAN)])})
    create_table(path / "DATA_DESCRIPTION", [scalar_column("SPECTRAL_WINDOW_ID", "int"),
                                             scalar_column("POLARIZATION_ID", "int")], 1,
                 {"SPECTRAL_WINDOW_ID": np.zeros(1, np.int32), "POLARIZATION_ID": np.zeros(1, np.int32)})
    create_table(path / "POLARIZATION", [scalar_column("NUM_CORR", "int"),
                                         array_column("CORR_TYPE", "int", shape=(NCORR,))], 1,
                 {"NUM_CORR": np.array([NCORR], np.int32), "CORR_TYPE": np.array([[5, 8]], np.int32)})
    return {"ms": path, "dropped": dropped, "burst": burst, "scan": scan, "field": field, "ant1": ant1,
            "ant2": ant2, "cal_number": cal_number}


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")
    return _write_ms(tmp_path / "sim.ms")


def _run(synthetic) -> dict:
    return so.smart_outliers(synthetic["ms"], fields=["CAL", "TGT"], column="data", receivers={"CAL": ["TGT"]},
                             timebin=15.0, blocks=4, workers=1)


def test_commands_trace_the_faults_to_their_antennas_and_the_target_scans_next_to_them(synthetic):
    report = _run(synthetic)
    commands = report["commands"]
    assert all("antenna='A3'" in c or "antenna='A1'" in c for c in commands), commands
    cal = [c for c in commands if "field='CAL'" in c]
    assert any("antenna='A3'" in c for c in cal) and any("antenna='A1'" in c for c in cal)
    # The target scans between the calibrator scans around the lost one (TGT scans 10 and 12 of the layout).
    extended = report["fields"]["CAL"]["extended"]
    assert [(e["antenna"], e["targets"]) for e in extended] == [("A3", ["TGT:10", "TGT:12"])]
    assert len([c for c in commands if "field='TGT'" in c]) == 2
    assert report["fields"]["TGT"]["flagged"] == 0


def test_applying_the_commands_flags_the_faulty_rows_and_little_else(synthetic):
    report = _run(synthetic)
    setup = read_flag_setup(synthetic["ms"])
    apply_flag_selections(synthetic["ms"], compile_flagdata({"mode": "list", "inpfile": report["commands"]}, setup),
                          setup=setup, workers=1)
    with Table(synthetic["ms"]) as table:
        flagged = np.asarray(table.getcol("FLAG")).all(axis=(1, 2))
    assert flagged[synthetic["dropped"]].all()
    assert flagged[synthetic["burst"]].all()
    a3 = (synthetic["ant1"] == 3) | (synthetic["ant2"] == 3)
    target_next = (synthetic["field"] == 1) & np.isin(synthetic["scan"], [10, 12]) & a3
    assert flagged[target_next].all()
    expected = synthetic["dropped"] | synthetic["burst"] | target_next
    # Beyond the faults only the edges the burst's growth takes (one bin either side) may go.
    assert flagged[~expected].sum() <= 2 * 4 * (NANT - 1)


def test_time_bins_never_straddle_two_scans():
    scans = [{"scan": 1, "t_min": 100.0, "t_max": 160.0}, {"scan": 2, "t_min": 300.0, "t_max": 310.0}]
    bins = so.time_bins(scans, 15.0)
    assert bins["ntime"] == 5 + 1 and bins["scans"] == {1: (100.0, 0, 5), 2: (300.0, 5, 1)}
    np.testing.assert_array_equal(bins["segment"], [1, 1, 1, 1, 1, 2])
    assert bins["start"][0] < 100.0 and bins["stop"][4] > 160.0 and bins["stop"][0] < bins["start"][1]


def test_spw_selection_names_whole_subbands_and_channel_ranges():
    starts = so.channel_blocks(16, 4)
    np.testing.assert_array_equal(starts, [0, 4, 8, 12])
    mask = np.array([[True, True, True, True], [False, True, True, False], [True, False, False, True]])
    assert so.spw_selection(mask, starts, 16, [0, 1, 2]) == "0,1:4~11,2:0~3;12~15"


def test_mask_commands_limit_the_correlations_to_a_lost_hand():
    bins = so.time_bins([{"scan": 1, "t_min": 0.0, "t_max": 45.0}], 15.0)
    mask = np.zeros((1, 2, 2, bins["ntime"]), dtype=bool)
    mask[0, :, 0, 1:3] = True                                      # RR only, bins 1-2
    commands = so.mask_commands(mask, selector="antenna='EF'", field_name="X", bins=bins,
                                block_starts=np.array([0, 8]), nchan=16, ddid_to_spw=[0], hands=["RR", "LL"],
                                labels=["RR", "RL", "LR", "LL"])
    assert commands == [f"antenna='EF' field='X' timerange='{so.stamp(bins['start'][1])}~{so.stamp(bins['stop'][2])}' "
                        "spw='0' correlation='RR,RL,LR'"]
