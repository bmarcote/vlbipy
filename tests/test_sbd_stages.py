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


def test_amend_gives_a_lost_antenna_another_scan_and_keeps_the_rest():
    from vlbipy.selection import amend_sbd_stages
    # EM163: scan 103 detects everyone, but CM (here T6) was off source in it and got flagged there.
    s = survey({1: [92, 58, 48, 33, 24], 103: [95, 60, 50, 35, 25], 195: [90, 55, NAN, 30, 22]})
    plan = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [stage["scan"] for stage in plan] == [103]
    amended = amend_sbd_stages(plan, s, {103: ["T6"]}, min_snr=7)
    assert [stage["scan"] for stage in amended] == [103, 1]          # scan 1 has the higher median SNR
    assert amended[0]["antennas"] == ["EF", "WB", "JB", "MC"] and amended[0]["refant"] == "EF"
    assert amended[1] == {"scan": 1, "antennas": ["T6"], "refant": "EF"}
    assert plan[0]["antennas"] == ["EF", "WB", "JB", "MC", "T6"]        # the plan passed in is not modified


def test_amend_reuses_a_stage_already_on_the_helping_scan():
    from vlbipy.selection import amend_sbd_stages
    s = survey({1: [90, 50, 40, NAN, NAN], 2: [NAN, 60, 45, 30, 20]})
    plan = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [stage["scan"] for stage in plan] == [2, 1]
    # JB lost scan 2; scan 1 (already the EF stage) also detects it.
    amended = amend_sbd_stages(plan, s, {2: ["JB"]}, min_snr=7)
    assert [stage["scan"] for stage in amended] == [2, 1]
    assert "JB" not in amended[0]["antennas"] and amended[1]["antennas"] == ["EF", "JB"]


def test_amend_leaves_out_an_antenna_no_other_scan_detects():
    from vlbipy.selection import amend_sbd_stages
    s = survey({1: [90, 50, 40, 30, 20], 2: [90, 60, 45, 30, NAN]})
    plan = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    amended = amend_sbd_stages(plan, s, {1: ["T6"]}, min_snr=7)
    assert [stage["scan"] for stage in amended] == [1]
    assert "T6" not in amended[0]["antennas"]


def test_amend_gives_up_when_a_stage_lost_its_reference():
    from vlbipy.selection import amend_sbd_stages
    s = survey({1: [90, 50, 40, 30, 20], 2: [90, 60, 45, 30, 25]})
    plan = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert amend_sbd_stages(plan, s, {plan[0]["scan"]: ["EF"]}, min_snr=7) == []


def test_selection_is_replanned_when_flagging_empties_an_antennas_stage_scan(monkeypatch):
    """An antenna flagged away in its stage scan gets another scan instead of losing its calibration."""
    from vlbipy import VLBIObs

    collection = VLBIObs("stage02", network="EVN", backend="dummy", target="3C395",
                         phasecal="J1848+3219", fringe_finder="3C345")
    collection.import_data()
    obs = collection["stage02"]
    first, second = [scan.scan_number for scan in obs.metadata.scans_for_source("3C345")[:2]]
    antennas = ["EF", "WB", "JB"]
    plan = [{"scan": first, "antennas": antennas, "refant": "EF"}]
    obs.set_cal_selection(antennas, [first], plan)
    assert obs.calibrate.select_calibration_data() == (antennas, [first])       # nothing lost: reused as it is

    def summary(code, where=""):
        """JB has no unflagged data left in the first scan."""
        lost = where.endswith(f"== {first}")
        return {"antenna": {name: {"flagged": 100 if lost and name == "JB" else 10, "observable": 100,
                                   "fraction": 1.0 if lost and name == "JB" else 0.1} for name in antennas}}

    found = ScanSNRSurvey(project_code="stage02", scan_numbers=[first, second], scan_sources=["3C345", "3C345"],
                          antennas=antennas, refant="EF", snr={"RR": [[NAN, 50.0, 40.0], [NAN, 45.0, 35.0]]})
    surveyed = []
    monkeypatch.setattr(obs.calibrate._backend.flag, "summary", summary)
    monkeypatch.setattr(obs.calibrate, "scan_snr", lambda **kwargs: surveyed.append(kwargs) or found)

    assert obs.calibrate.select_calibration_data() == (antennas, [first, second])
    assert surveyed == [{"field": "3C345", "force": True}]
    assert obs.cal_stages == [{"scan": first, "antennas": ["EF", "WB"], "refant": "EF"},
                              {"scan": second, "antennas": ["JB"], "refant": "EF"}]
    # JB is no longer solved on the first scan, so the amended plan is stable.
    assert obs.calibrate.select_calibration_data() == (antennas, [first, second])
    assert len(surveyed) == 1


def test_weak_detections_are_blanked_so_the_plan_solves_the_antenna_elsewhere():
    from vlbipy.selection import without_weak_detections
    # T6 is detected in every scan, but in scan 103 it is at a few per cent of its amplitude.
    s = survey({1: [92, 58, 48, NAN, 24], 103: [95, 60, 50, 35, 25], 195: [90, 55, NAN, 30, 22]})
    assert [stage["scan"] for stage in plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)] == [103]
    planning, removed = without_weak_detections(s, {103: ["T6"], 1: ["MC"]}, min_snr=7)
    assert removed == {103: ["T6"]}                                  # MC was not detected in scan 1 anyway
    assert s.snr["RR"][1][4] == 25                                   # the survey itself is untouched
    stages = plan_sbd_stages(planning, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [(stage["scan"], stage["antennas"]) for stage in stages] == [(103, ["EF", "WB", "JB", "MC"]), (1, ["T6"])]


def test_a_weak_antenna_no_other_scan_detects_keeps_its_detection():
    from vlbipy.selection import without_weak_detections
    s = survey({1: [90, 50, 40, 30, NAN], 2: [90, 60, 45, 30, 25]})
    planning, removed = without_weak_detections(s, {2: ["T6"]}, min_snr=7)
    assert removed == {} and planning.snr == s.snr


def test_a_weak_reference_antenna_takes_its_scan_out():
    from vlbipy.selection import without_weak_detections
    s = survey({1: [NAN, 50, 40, 30, 20], 2: [NAN, 60, 45, 30, 25]}, refant="EF")
    planning, removed = without_weak_detections(s, {2: ["EF"]}, min_snr=7)
    assert removed == {2: ["EF"]}
    assert all(value != value for value in planning.snr["RR"][1])
    assert [stage["scan"] for stage in plan_sbd_stages(planning, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)] == [1]


def test_first_pass_selection_does_not_solve_an_antenna_on_a_scan_where_it_is_weak(monkeypatch):
    """The scan search compares raw amplitudes between scans, not only the fringe SNR."""
    import numpy as np
    from vlbipy import VLBIObs

    collection = VLBIObs("stage03", network="EVN", backend="dummy", target="3C395",
                         phasecal="J1848+3219", fringe_finder="3C345")
    collection.import_data()
    obs = collection["stage03"]
    first, second = [scan.scan_number for scan in obs.metadata.scans_for_source("3C345")[:2]]
    names = list(obs.metadata.antennas)
    index = {name: k for k, name in enumerate(names)}
    snr = [[NAN if name == "EF" else 60.0 for name in names], [NAN if name == "EF" else 50.0 for name in names]]
    found = ScanSNRSurvey(project_code="stage03", scan_numbers=[first, second], scan_sources=["3C345", "3C345"],
                          antennas=names, refant="EF", snr={"RR": snr, "LL": snr})
    scan_of_time = np.repeat([first, second], 20)
    amplitude = np.full((len(names), len(names), 40), 1.0)
    amplitude[np.arange(len(names)), np.arange(len(names))] = np.nan
    amplitude[index["JB"], :, :20] *= 0.05                           # JB: detected in the first scan, but off source
    amplitude[:, index["JB"], :20] *= 0.05
    reads = []

    def read_baseline_timeline(self, code, **kwargs):
        reads.append(kwargs)
        return {"antennas": names, "amplitude": amplitude, "scans": scan_of_time}

    monkeypatch.setattr(type(obs.calibrate._backend.data), "read_baseline_timeline", read_baseline_timeline,
                        raising=False)
    monkeypatch.setattr(obs.calibrate, "scan_snr", lambda **kwargs: found)

    # The second scan detects everyone and JB is fine in it: that is the scan to solve on.
    antennas, scans = obs.calibrate.select_calibration_data()
    assert scans == [second] and "JB" in obs.cal_stages[0]["antennas"]
    assert reads[0]["column"] == "data" and reads[0]["delay_search"] and reads[0]["scans"] == [first, second]

    # When the second scan misses an antenna (WB), the first is still needed, but not for JB.
    obs.clear_cal_selection()
    snr[1][index["WB"]] = NAN
    antennas, scans = obs.calibrate.select_calibration_data()
    assert scans == [first, second]
    assert "JB" not in obs.cal_stages[0]["antennas"] and "WB" in obs.cal_stages[0]["antennas"]
    assert obs.cal_stages[1]["antennas"] == ["JB"]


# -- source ranking: the instrumental solve must land on the brightest source ------------------

def multi_source_survey(rows: dict[int, tuple[str, list[float]]],
                        antennas=("EF", "WB", "JB", "MC", "T6"), refant="EF"):
    """Build a one-polarization survey where ``rows`` maps scan -> (source, SNR per antenna)."""
    scans = sorted(rows)
    return ScanSNRSurvey(project_code="p", scan_numbers=scans, scan_sources=[rows[s][0] for s in scans],
                         antennas=list(antennas), snr={"RR": [rows[s][1] for s in scans]}, refant=refant)


def test_source_ranking_counts_antennas_a_source_never_detects_as_zero():
    # A source most of the array cannot see is no basis for an array-wide solution, however
    # bright the baselines that do see it: counting the others as zero sinks it below the
    # median. BRIGHT is seen by one of the five non-reference antennas, WEAK by all five.
    s = multi_source_survey({1: ("BRIGHT", [NAN, 9000, NAN, NAN, NAN, NAN]),
                             2: ("WEAK", [NAN, 20, 18, 16, 14, 12])},
                            antennas=("EF", "WB", "JB", "MC", "T6", "HH"))
    ranked = {name: snr for name, snr, _ in s.rank_sources()}
    assert ranked["BRIGHT"] == 0.0 and ranked["WEAK"] == 16.0
    assert [found for _, _, found in s.rank_sources()] == [5, 1]


def test_source_ranking_prefers_brightness_once_a_source_covers_the_array():
    s = multi_source_survey({1: ("BRIGHT", [NAN, 900, 800, 700, 600]),
                             2: ("WEAK", [NAN, 20, 18, 16, 14])})
    assert [name for name, _, _ in s.rank_sources()] == ["BRIGHT", "WEAK"]


def test_instrumental_scan_is_the_bright_source_not_the_weak_one_with_an_extra_antenna():
    # The reported failure: the weak source's scan covers one antenna more, so coverage-first
    # selection solved the instrumental delay at SNR ~8 and the fringe fit returned nothing.
    s = multi_source_survey({1: ("FF", [NAN, 900, 800, 700, NAN]),
                             2: ("WEAK", [NAN, 9, 8, 8, 7])})
    antennas = ["EF", "WB", "JB", "MC", "T6"]
    stages = plan_sbd_stages(s, antennas, min_snr=7)
    assert stages[0]["scan"] == 1                          # the bright scan leads
    assert set(stages[0]["antennas"]) == {"EF", "WB", "JB", "MC"}
    assert [st["scan"] for st in stages] == [1, 2]         # T6 comes from the weak source after
    assert "T6" in stages[1]["antennas"]


def test_a_single_scan_on_the_best_source_still_wins_over_more_scans_elsewhere():
    s = multi_source_survey({1: ("FF", [NAN, 900, 800, 700, 600]),
                             2: ("WEAK", [NAN, 20, 20, 20, 20]),
                             3: ("WEAK", [NAN, 21, 21, 21, 21])})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert len(stages) == 1 and stages[0]["scan"] == 1


def test_restricting_a_survey_keeps_only_the_wanted_sources():
    s = multi_source_survey({1: ("FF", [NAN, 900, 800, 700, 600]),
                             2: ("WEAK", [NAN, 20, 20, 20, 20])})
    only_ff = s.restricted_to(["FF"])
    assert only_ff.scan_numbers == [1] and only_ff.scan_sources == ["FF"]
    assert only_ff.antennas == s.antennas and only_ff.refant == s.refant
    assert only_ff.snr["RR"] == [s.snr["RR"][0]]
    assert s.scan_numbers == [1, 2]                        # the original is untouched
    assert s.restricted_to(["NOPE"]).scan_numbers == []


def test_a_sub_array_that_never_meets_the_rest_gets_its_own_chain():
    # Staggered global VLBI (GL048): EF/WB/JB observe first, MC/T6 later, and no scan has both groups.
    s = survey({1: [90, 50, 40, NAN, NAN], 2: [NAN, NAN, NAN, 30, 20]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [st["scan"] for st in stages] == [1, 2]
    assert not stages[0].get("independent")
    assert stages[1]["independent"] and set(stages[1]["antennas"]) == {"MC", "T6"}
    assert stages[1]["refant"] in {"MC", "T6"}


def test_a_scan_that_can_be_chained_is_preferred_over_a_separate_sub_array():
    # Scan 2 ties MC to WB; scan 3 holds MC and T6 alone. Chain first, then T6 comes through MC.
    s = survey({1: [90, 50, 40, NAN, NAN], 2: [NAN, 60, NAN, 30, NAN], 3: [NAN, NAN, NAN, 30, 20]})
    stages = plan_sbd_stages(s, ["EF", "WB", "JB", "MC", "T6"], min_snr=7)
    assert [st["scan"] for st in stages] == [1, 2, 3]
    assert not any(st.get("independent") for st in stages)
    assert stages[2]["refant"] == "MC" and stages[2]["antennas"] == ["T6"]


def test_solved_scans_leaves_out_the_rows_a_sampled_survey_never_fitted():
    s = survey({1: [90, 50, 40, NAN, NAN], 2: [NAN] * 5, 3: [NAN, NAN, NAN, 30, 20]})
    assert s.solved_scans() == [1, 3]


def test_a_sampled_wider_survey_is_not_reused_for_scans_it_never_fitted():
    from types import SimpleNamespace
    from vlbipy.models import Scan
    from vlbipy.namespaces import CalibrateNamespace
    # The phase-calibrator survey sampled scan 1 of the fringe finder but never fitted scan 3.
    wide = ScanSNRSurvey(project_code="p", scan_numbers=[1, 2, 3], scan_sources=["FF", "PC", "FF"],
                         antennas=["EF", "WB"], snr={"RR": [[NAN, 50], [NAN, 40], [NAN, NAN]]})
    scans = [Scan(scan_number=n, source=src) for n, src in ((1, "FF"), (2, "PC"), (3, "FF"))]
    obs = SimpleNamespace(metadata=SimpleNamespace(scans=scans), _snr_surveys={"PC,FF": wide})
    namespace = CalibrateNamespace.__new__(CalibrateNamespace)
    namespace._obs = obs
    assert namespace._wider_survey("FF") is None
    wide.snr["RR"][2] = [NAN, 30]                          # once scan 3 is fitted the slice is reused
    assert namespace._wider_survey("FF").scan_numbers == [1, 3]
