"""Tests for the CASA-compatible fringefit task driver (selection parsing, interval splitting, backend wiring).

The real-data comparison against CASA (RSM07) runs only when the measurement set is present.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vlbipy.solvers import fringefit as task

RSM07_MS = Path("/home/marcote/Programing/vlbipy/rsm07_manual/rsm07.ms")


def test_parse_names_accepts_names_ids_and_ranges():
    names = ["3C345", "J1848+3219", "3C395"]
    assert task.parse_names("", names) == [0, 1, 2]
    assert task.parse_names("3C395,3C345", names) == [2, 0]
    assert task.parse_names("0~1", names) == [0, 1]
    with pytest.raises(ValueError):
        task.parse_names("NOPE", names)


def test_parse_spw_channel_ranges():
    sel = task.parse_spw("*:6~57", 4, 64)
    assert sorted(sel) == [0, 1, 2, 3] and sel[0][0] == 6 and sel[0][-1] == 57 and sel[0].size == 52
    sel = task.parse_spw("1,2:4~10;20~22", 4, 64)
    assert sorted(sel) == [1, 2] and sel[1].size == 64 and list(sel[2]) == [4, 5, 6, 7, 8, 9, 10, 20, 21, 22]
    assert task.parse_spw("", 2, 8)[1].size == 8


def test_parse_scans_timerange_antennas_solint():
    assert task.parse_scans("1,2,4~6") == {1, 2, 4, 5, 6} and task.parse_scans("") is None
    t0, t1 = task.parse_timerange("2024/03/05/12:00:00~2024/03/05/12:01:30.5")
    assert t1 - t0 == pytest.approx(90.5) and t0 == pytest.approx(60374 * 86400 + 12 * 3600)
    ids, among = task.parse_antennas("EF,JB&", ["JB", "WB", "EF"])
    assert ids == {0, 2} and among is True
    assert task.parse_antennas("", ["JB"]) == (None, False)
    assert task.parse_solint("inf") == float("inf") and task.parse_solint("2min") == 120.0
    assert task.parse_solint("30s") == 30.0 and task.parse_solint("int") == 0.0


def test_interval_edges_split_a_scan():
    times = 1000.0 + 2.0 * np.arange(10)
    assert task._interval_edges(times, float("inf")) == [(1000.0, 1018.0)]
    edges = task._interval_edges(times, 6.0)
    assert edges == [(1000.0, 1004.0), (1006.0, 1010.0), (1012.0, 1016.0), (1018.0, 1018.0)]
    assert len(task._interval_edges(times, 0.0)) == 10


def test_first_refant_with_data_follows_the_chain():
    a1 = np.array([0, 0, 1]); a2 = np.array([1, 2, 2])
    flag = np.zeros((3, 2, 2), dtype=bool)
    flag[1:] = True  # only baseline 0-1 has data
    assert task._first_refant_with_data([2, 1, 0], a1, a2, flag) == 1
    assert task._first_refant_with_data([2], a1, a2, flag) == -1


def test_prior_entries_from_casa_lists():
    entries = task._prior_entries(["a.tsys", "b.sbd"], ["", "3C345"], ["nearest", "linear"], [[], [0, 0, 0, 0]],
                                  ["3C345", "J1848"])
    assert entries[0]["gainfield"] == [] and entries[0]["interp"] == "nearest" and entries[0]["spwmap"] == []
    assert entries[1]["gainfield"] == [0] and entries[1]["spwmap"] == [0, 0, 0, 0]


def test_fast_engine_lives_in_the_dask_ms_backend_only():
    """The numpy calibration engine is the dask-ms backend; the old casa-fast/fast names are gone."""
    from vlbipy.registry import list_backends
    names = set(list_backends())
    assert "dask-ms" in names and not {"casa-fast", "fast"} & names


@pytest.mark.skipif(not RSM07_MS.is_dir() or os.environ.get("VLBIPY_SKIP_REALDATA"), reason="RSM07 MS not available")
def test_run_fringefit_matches_casa_on_rsm07(tmp_path):
    """Single-band delay on scan 63 (refant EF, channels 6~57, zerorates) against the CASA table of the same solve."""
    pytest.importorskip("daskms")
    casa_table = Path("/tmp/vlbipy_spec/casa_ab/sbd_parang1")
    out = task.fringefit(vis=str(RSM07_MS), caltable=str(tmp_path / "sbd"), field="3C345", spw="*:6~57",
                             scan="63", solint="inf", refant="EF", minsnr=10.0, zerorates=True, parang=True, workers=1)
    from vlbipy.solvers.caltable import read_fringe_table
    ours = read_fringe_table(out)
    assert ours["nrow"] == 4 * 14 and set(ours["SPECTRAL_WINDOW_ID"]) == {0, 1, 2, 3}
    assert np.all(ours["ANTENNA2"] == 2) and np.all(ours["SNR"][ours["ANTENNA1"] == 2] == 999.0)
    assert np.all(ours["FPARAM"][:, [2, 6]] == 0.0)  # zerorates
    if not casa_table.is_dir():
        pytest.skip("CASA reference table not available")
    ref = read_fringe_table(casa_table)
    key = lambda d: {(int(s), int(a)): i for i, (s, a) in enumerate(zip(d["SPECTRAL_WINDOW_ID"], d["ANTENNA1"]))}
    ko, kr = key(ours), key(ref)
    diffs = []
    for k, i in ko.items():
        j = kr[k]
        if ours["FLAG"][i, 0] or ref["FLAG"][j, 0]:
            assert bool(ours["FLAG"][i, 0]) == bool(ref["FLAG"][j, 0]), f"flag mismatch for spw/antenna {k}"
            continue
        diffs.append([ours["FPARAM"][i, 1] - ref["FPARAM"][j, 1], ours["FPARAM"][i, 5] - ref["FPARAM"][j, 5],
                      np.angle(np.exp(1j * (ours["FPARAM"][i, 0] - ref["FPARAM"][j, 0])))])
    diffs = np.abs(np.array(diffs))
    assert np.median(diffs[:, :2]) < 0.05, "delays differ from CASA by more than 50 ps (median)"
    assert np.max(diffs[:, :2]) < 0.5, "a delay differs from CASA by more than 0.5 ns"
    assert np.degrees(np.median(diffs[:, 2])) < 5.0, "phases differ from CASA by more than 5 degrees (median)"


def test_the_logged_median_snr_leaves_the_reference_antenna_out():
    """An interval where only the reference antenna is "solved" has no SNR to report (EM178, inverse phase
    referencing: the log said median SNR 999 for a target nobody detected); a bright source's SNRs above the
    sentinel stay in (the instrumental-delay scan on J2136+0041: median ~4700)."""
    from types import SimpleNamespace
    from vlbipy.solvers.fringe import REFANT_SNR_SENTINEL
    from vlbipy.solvers.fringefit import _median_snr
    only_ref = SimpleNamespace(snr=np.array([[REFANT_SNR_SENTINEL] * 2, [2.0, 2.5], [3.0, 3.5]]),
                               flag=np.array([[False] * 2, [True] * 2, [True] * 2]), refant=0)
    assert _median_snr(only_ref) == 0.0
    solved = SimpleNamespace(snr=np.array([8.0, REFANT_SNR_SENTINEL, 12.0, 2.0]),
                             flag=np.array([False, False, False, True]), refant=1)
    assert _median_snr(solved) == 10.0
    bright = SimpleNamespace(snr=np.array([REFANT_SNR_SENTINEL, 4000.0, 6000.0]), flag=np.zeros(3, bool), refant=0)
    assert _median_snr(bright) == 5000.0
