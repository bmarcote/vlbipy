"""Tests for the staged single-band-delay plan (selection.plan_sbd_stages)."""
from __future__ import annotations

from pathlib import Path

from vlbipy.models import ScanSNRSurvey
from vlbipy.selection import plan_sbd_stages, select_calibration_scans

NAN = float("nan")


def survey(rows: dict[int, list[float]], antennas=("EF", "WB", "JB", "MC", "T6"), refant=""):
    """Build a one-polarization survey (no auto-detected refant); ``rows`` maps scan -> SNR per antenna."""
    scans = sorted(rows)
    return ScanSNRSurvey(project_code="p", scan_numbers=scans, scan_sources=["FF"] * len(scans),
                         antennas=list(antennas), snr={"RR": [rows[s] for s in scans]}, refant=refant)


def test_single_scan_covering_all_antennas_gives_one_stage():
    s = survey({1: [90, 50, 40, 30, 20], 2: [90, 60, NAN, 30, 20]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert len(stages) == 1
    assert stages[0]["scan"] == 1 and stages[0]["refant"] == "EF"
    assert set(stages[0]["antennas"]) == {"EF", "WB", "JB", "MC", "T6"}
    assert select_calibration_scans(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7) == [1]


def test_chained_stages_share_an_antenna_and_use_it_as_reference():
    # Scan 1 detects EF, WB, JB; scan 2 detects WB, MC, T6 (EF absent). Two stages, chained by WB.
    s = survey({1: [90, 50, 40, NAN, NAN], 2: [NAN, 60, NAN, 30, 20]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [st["scan"] for st in stages] == [1, 2]
    assert set(stages[0]["antennas"]) == {"EF", "WB", "JB"} and stages[0]["refant"] == "EF"
    assert set(stages[1]["antennas"]) == {"MC", "T6"} and stages[1]["refant"] == "WB"


def test_stage_reference_is_the_best_already_solved_antenna_present():
    # Scan 2 detects WB (60) and JB (10) from stage 1, plus MC: WB should be the reference.
    s = survey({1: [90, 50, 40, NAN, NAN], 2: [NAN, 60, 10, 30, NAN]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC"], min_snr=7)
    assert stages[1]["refant"] == "WB" and stages[1]["antennas"] == ["MC"]


def test_unlinkable_scan_is_not_used_and_uses_fewest_scans():
    # Scan 3 detects only T6 with nobody shared: it cannot be tied in.
    s = survey({1: [90, 50, 40, 30, NAN], 3: [NAN, NAN, NAN, NAN, 20]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [st["scan"] for st in stages] == [1]


def test_no_detection_gives_no_plan():
    s = survey({1: [2, 2, 3, 1, NAN]})
    assert plan_sbd_stages(s, ["EF", "WB", "JB"], min_snr=7) == []


def test_partial_band_antennas_are_kept_in_the_selection():
    """An antenna recording only some subbands is still calibrated (heterogeneous arrays are common)."""
    from vlbipy.models import Antenna, FreqSetup, ObsMetadata
    from vlbipy.selection import select_antennas
    antennas = {n: Antenna(name=n, observed=True, subbands=(0, 1, 2, 3)) for n in ("EF", "WB", "JB")}
    antennas["DE"] = Antenna(name="DE", observed=True, subbands=(1, 2))
    meta = ObsMetadata(project_code="p", antennas=antennas, freq_setup=FreqSetup(n_subbands=4, n_channels=64))
    s = survey({1: [90, 50, 40, 30]}, antennas=("EF", "WB", "JB", "DE"))
    assert select_antennas(s, meta, min_snr=7) == ["EF", "WB", "JB", "DE"]


def test_subband_phase_jump_plot(tmp_path):
    import numpy as np
    from vlbipy.plotting import plot_subband_phase_jumps
    phases = {"WB": np.zeros((3, 4, 2)), "T6": np.zeros((3, 4, 2))}
    phases["T6"][:, 2, :] = 60.0                       # a constant 60 deg jump in subband 2
    phases["WB"][1, 3, :] = np.nan                     # subband 3 missing in one scan
    data = {"antennas": ["WB", "T6"], "phases": phases, "refant": "EF", "polarizations": ["RR", "LL"],
            "scans": [{"scan": k, "source": "J1", "time": 1e9 + 600.0 * k} for k in range(3)],
            "n_spw": 4, "column": "corrected", "fields": ["J1"]}
    out = plot_subband_phase_jumps(data, tmp_path, "p", label="final")
    assert Path(out).is_file() and Path(out).name == "p.final.subband_phases.png"


def test_namespace_subband_phases_on_dummy_backend():
    from vlbipy import VLBIObs
    obs = VLBIObs("sp01", network="EVN", backend="dummy", target="3C395", phasecal="J1848+3219")
    obs.import_data()
    path = obs["sp01"].plot.subband_phases()
    assert path.endswith(".subband_phases.png")
    assert not any(p.endswith(".subband_phases.png") for p in obs["sp01"].plot.final_data())
    assert any(p.endswith(".subband_phases.png") for p in obs["sp01"].plot.diagnostics())


def test_initial_calibration_plots_every_sbd_stage_with_its_refant(monkeypatch):
    """Each staged solve gets an exact-scan data plot using that stage's reference antenna."""
    from vlbipy import VLBIObs

    collection = VLBIObs("stage01", network="EVN", backend="dummy", target="3C395",
                         phasecal="J1848+3219", fringe_finder="3C345")
    collection.import_data()
    obs = collection["stage01"]
    selected = obs.metadata.scans_for_source("3C345")[:2]
    assert len(selected) == 2
    stages = [{"scan": selected[0].scan_number, "refant": selected[0].antennas[0], "antennas": []},
              {"scan": selected[1].scan_number, "refant": selected[1].antennas[1], "antennas": []}]
    obs.set_cal_selection(list(obs.metadata.antennas), [s["scan"] for s in stages], stages)
    calls = []
    monkeypatch.setattr(obs.plot, "timeseries", lambda **kwargs: calls.append(kwargs) or [])

    obs.calibrate.initial_calibration(force=True)

    assert [call["scans"] for call in calls] == [[stage["scan"]] for stage in stages]
    assert [call["refant"] for call in calls] == [stage["refant"] for stage in stages]
    assert all(f"scan{stage['scan']}" in call["label"] for call, stage in zip(calls, stages))


def test_a_hand_missing_in_the_best_scan_comes_from_a_later_stage():
    """An antenna with one polarization in the best scan gets the other from a scan that has it."""
    nan = float("nan")
    s = ScanSNRSurvey(project_code="T", scan_numbers=[1, 2], scan_sources=["FF", "FF"],
                      antennas=["EF", "WB", "JB", "MC"], refant="EF",
                      snr={"RR": [[nan, 50.0, 50.0, nan], [nan, 40.0, nan, 30.0]],
                           "LL": [[nan, 50.0, 50.0, 50.0], [nan, 40.0, nan, 30.0]]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC"], min_snr=7)
    assert [st["scan"] for st in stages] == [1, 2]
    assert stages[1]["antennas"] == ["MC"] and stages[1]["refant"] == "EF"


def test_a_dead_polarization_does_not_cost_the_antenna():
    nan = float("nan")
    s = ScanSNRSurvey(project_code="T", scan_numbers=[1], scan_sources=["FF"], antennas=["EF", "WB", "TI"],
                      refant="EF", snr={"RR": [[nan, 50.0, 50.0]], "LL": [[nan, 50.0, nan]]})
    stages = plan_sbd_stages(s, ["EF", "WB", "TI"], min_snr=7)
    assert len(stages) == 1 and set(stages[0]["antennas"]) == {"EF", "WB", "TI"}


def test_antenna_with_one_dead_polarization_is_selected_on_the_live_one():
    from vlbipy.models import Antenna, ObsMetadata
    from vlbipy.selection import select_antennas
    nan = float("nan")
    s = ScanSNRSurvey(project_code="T", scan_numbers=[1, 2], scan_sources=["FF", "FF"], antennas=["AT", "MP", "KE"],
                      refant="AT", snr={"RR": [[nan, 80.0, 40.0], [nan, 90.0, 45.0]],
                                        "LL": [[nan, 80.0, 2.0], [nan, 90.0, 3.0]]})
    meta = ObsMetadata(project_code="T", antennas={n: Antenna(name=n) for n in ("AT", "MP", "KE")})
    assert select_antennas(s, meta, min_snr=7) == ["AT", "MP", "KE"]
    stages = plan_sbd_stages(s, ["AT", "MP", "KE"], min_snr=7)
    assert len(stages) == 1 and "KE" in stages[0]["antennas"]
