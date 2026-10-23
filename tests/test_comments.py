"""Tests for the general comments of a reduction (vlbipy.comments): one rule at a time, on small contexts."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from vlbipy import VLBIObs
from vlbipy import comments as cm
from vlbipy.dashboard import gain_class
from vlbipy.logging_utils import warnings as collector


# -- context builders ---------------------------------------------------------------------------

def antenna(name: str, *, observed: bool = True, subbands=(0, 1, 2, 3), fraction=0.30) -> dict:
    """One entry of the context ``antennas`` list."""
    return {"name": name, "observed": observed, "subbands": list(subbands), "flagged_fraction": fraction}


def scan(number: int, antennas: list[str], *, source: str = "CAL", role: str = "phasecal", snrs=None,
         refant: str = "EF", start: datetime = datetime(2024, 1, 1, 10, 0, 0), duration: int = 60) -> dict:
    """One context scan row, one minute per scan number from ``start``.

    ``snrs`` maps antenna -> fringe SNR; None (the default) leaves the scan unsurveyed, and an antenna
    missing from a given map has no fringe solution. The reference antenna never has an SNR of its own.
    """
    begin = start + timedelta(minutes=number)
    surveyed = snrs is not None
    states, values = {}, {}
    for name in antennas:
        value = None if name == refant else (snrs or {}).get(name)
        values[name] = value
        states[name] = "refant" if name == refant else (
            "snr" if value is not None else ("nofringe" if surveyed else "unsurveyed"))
    return {"number": number, "source": source, "role": role, "start_utc": begin.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration, "antennas": list(antennas), "surveyed": surveyed, "antenna_snrs": values,
            "antenna_states": states}


def context(antennas: list[dict], scans: list[dict] | None = None, **extra) -> dict:
    """A dashboard context with just what the rules read."""
    return {"project": "t01", "refant": "EF", "observation": {"n_subbands": 4}, "antennas": antennas,
            "scans": scans or [], **extra}


def texts(comments: list[dict], name: str = "") -> list[str]:
    """Sentences of the comments about one antenna (or about none)."""
    return [c["text"] for c in comments if c["antenna"] == name]


ARRAY = ["EF", "JB", "WB", "MC", "T6"]


# -- helpers ------------------------------------------------------------------------------------

def test_ranges_names_and_fluxes_read_naturally():
    assert cm.format_ranges([3, 0, 1, 2, 8]) == "0-3, 8"
    assert cm.format_ranges([9, 10]) == "9-10"
    assert cm.join_names(["A"]) == "A" and cm.join_names(["A", "B", "C"]) == "A, B and C" and cm.join_names([]) == ""
    assert cm.format_flux(1.16) == "1.16 Jy" and cm.format_flux(0.136) == "136 mJy"
    assert cm.format_flux(6.23e-5) == "62.3 uJy"


def test_gain_levels_are_the_colours_of_the_gain_table():
    """The comments and the table must call the same corrections moderate or large."""
    assert cm.gain_level(1.10) == "good" and cm.gain_level(1.104) == "good"     # reads 1.10
    assert cm.gain_level(1.106) == "warn" and cm.gain_level(0.70) == "warn" and cm.gain_level(1.31) == "bad"
    assert cm.gain_level(None) is None and cm.gain_level(float("nan")) is None
    for hundredths in range(40, 201):
        gain = hundredths / 100.0
        assert gain_class(gain) == f"gain-{cm.gain_level(gain)}", gain


# -- antennas: data, band, gaps -----------------------------------------------------------------

def test_antennas_without_data_or_signal_and_heavily_flagged_ones_are_reported():
    ctx = context([antenna("EF", fraction=0.30), antenna("JB", fraction=0.32), antenna("WB", fraction=0.34),
                   antenna("TR", observed=False, subbands=(), fraction=0.0), antenna("O8", fraction=1.0),
                   antenna("DE", fraction=0.53)])
    comments = cm.data_comments(ctx)
    assert [(c["antenna"], c["severity"]) for c in comments] == [("TR", "warning"), ("O8", "problem"), ("DE", "warning")]
    assert "recorded nothing" in texts(comments, "TR")[0]
    assert "all of its data is flagged" in texts(comments, "O8")[0]
    # The median is taken over the antennas with data: 0.30, 0.32, 0.34, 0.53 -> 33%.
    assert texts(comments, "DE") == ["53% of its data is flagged, against 33% for the median antenna"]


def test_partial_band_is_said_once_per_antenna_naming_the_others_like_it():
    ctx = context([antenna("EF"), antenna("CM", subbands=(1, 2)), antenna("DE", subbands=(1, 2)),
                   antenna("WB", subbands=(3,)), antenna("KN", subbands=(1, 2), fraction=1.0)])
    comments = cm.band_comments(ctx)
    assert texts(comments, "CM") == ["recorded only subbands 1-2 (2 of 4), like DE"]
    assert texts(comments, "WB") == ["recorded only subband 3 (1 of 4)"]
    assert not texts(comments, "EF") and not texts(comments, "KN")      # full band; no usable data at all


def test_stretches_without_data_carry_utc_times_and_scan_numbers():
    scans = [scan(n, [a for a in ARRAY if not (a == "JB" and 4 <= n <= 8) and not (a == "WB" and n <= 3)
                      and not (a == "MC" and n >= 9) and not (a == "T6" and n in (2, 6))])
             for n in range(1, 11)]
    comments = cm.gap_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "JB") == ["no data from 10:04 to 10:09 UTC (scans 4-8)"]
    assert texts(comments, "WB") == ["no data until 10:04 UTC (scans 1-3)"]             # joins at scan 4
    # Two scans are below the shortest stretch reported, and T6 only skips isolated scans.
    assert not texts(comments, "MC") and not texts(comments, "T6") and not texts(comments, "EF")
    assert all(c["severity"] == "warning" for c in comments)


def test_a_stretch_to_the_end_and_short_gaps_are_worded_and_counted():
    scans = [scan(n, [a for a in ARRAY if not (a == "JB" and (n >= 7 or n == 2))]) for n in range(1, 11)]
    comments = cm.gap_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "JB") == ["no data from 10:07 UTC to the end of the observation (scans 7-10); "
                                     "absent from 1 other scan"]


def test_only_the_longest_stretches_are_listed_and_scattered_absences_are_a_single_note():
    absent = set(range(2, 5)) | set(range(8, 14)) | set(range(20, 24)) | set(range(30, 39)) | {50}
    alternate = set(range(1, 61, 2))
    scans = [scan(n, [a for a in ARRAY if not (a == "JB" and n in absent) and not (a == "WB" and n in alternate)])
             for n in range(1, 61)]
    comments = cm.gap_comments(context([antenna(a) for a in ARRAY], scans))
    line = texts(comments, "JB")[0]
    assert line.index("scans 30-38") < line.index("scans 8-13") < line.index("scans 20-23")      # longest first
    assert "scans 2-4" not in line and line.endswith("absent from 4 other scans")
    assert texts(comments, "WB") == ["absent from 30 of the 60 scans, never more than 2 in a row"]
    assert [c["severity"] for c in comments if c["antenna"] == "WB"] == ["note"]


def test_times_after_midnight_are_marked_and_missing_times_fall_back_to_scan_numbers():
    late = datetime(2024, 1, 1, 23, 50, 0)
    scans = [scan(n, [a for a in ARRAY if not (a == "JB" and 5 <= n <= 20)], start=late) for n in range(1, 31)]
    comments = cm.gap_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "JB") == ["no data from 23:55 to 00:11 (+1d) UTC (scans 5-20)"]
    for row in scans:
        row["start_utc"] = ""
    comments = cm.gap_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "JB") == ["no data in scans 5-20"]


# -- antennas: fringes and polarization ---------------------------------------------------------

def fringing(*missing: str) -> dict:
    """SNR map in which every antenna but ``missing`` (and the reference) fringes well."""
    return {name: 50.0 for name in ARRAY if name not in missing and name != "EF"}


def test_a_stretch_without_fringes_is_reported_when_the_rest_of_the_array_fringes():
    scans = [scan(n, ARRAY, snrs=fringing("T6") if 4 <= n <= 8 else fringing()) for n in range(1, 13)]
    comments = cm.fringe_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "T6") == ["no fringes from 10:04 to 10:09 UTC (scans 4-8)"]
    assert [c["antenna"] for c in comments] == ["T6"]                 # the reference antenna is never judged


def test_a_weak_fringe_counts_as_none_and_scans_where_nobody_fringes_say_nothing():
    weak = dict(fringing(), T6=2.0)                                   # below the detection threshold
    nobody = {"JB": 30.0}                                             # one detection among four
    scans = ([scan(n, ARRAY, snrs=fringing()) for n in (1, 2)] + [scan(n, ARRAY, snrs=weak) for n in (3, 4, 5)]
             + [scan(n, ARRAY, snrs=nobody) for n in (6, 7, 8)] + [scan(9, ARRAY, snrs=fringing())])
    comments = cm.fringe_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "T6") == ["no fringes from 10:03 to 10:06 UTC (scans 3-5)"]
    assert not texts(comments, "WB") and not texts(comments, "MC")    # scans 6-8 blame no single antenna


def test_target_scans_count_only_for_an_antenna_that_normally_fringes_on_that_target():
    """A target too weak for an antenna says nothing about it, however long the stretch."""
    scans = []
    for n in range(1, 31):
        if n % 2:
            scans.append(scan(n, ARRAY, snrs=fringing()))
        else:
            # WB never fringes on the target; T6 misses it in 4 of its 15 scans (73%, below the 80% asked).
            missing = ("WB", "T6") if n in (10, 12, 14, 16) else ("WB",)
            scans.append(scan(n, ARRAY, source="TARGET", role="target", snrs=fringing(*missing)))
    assert cm.fringe_comments(context([antenna(a) for a in ARRAY], scans)) == []


def test_a_bright_target_reveals_a_stretch_without_fringes():
    scans = []
    for n in range(1, 31):
        role, source = ("phasecal", "CAL") if n % 2 else ("target", "TARGET")
        scans.append(scan(n, ARRAY, source=source, role=role, snrs=fringing("T6") if 10 <= n <= 14 else fringing()))
    comments = cm.fringe_comments(context([antenna(a) for a in ARRAY], scans))
    assert texts(comments, "T6") == ["no fringes from 10:10 to 10:15 UTC (scans 10-14)"]


def test_an_antenna_that_never_fringes_is_a_problem_and_dead_antennas_are_left_out():
    scans = [scan(n, ARRAY + ["O8"], snrs=fringing("T6")) for n in range(1, 6)]
    antennas = [antenna(a) for a in ARRAY] + [antenna("O8", fraction=1.0)]
    comments = cm.fringe_comments(context(antennas, scans))
    assert [(c["antenna"], c["severity"]) for c in comments] == [("T6", "problem")]
    assert texts(comments, "T6")[0].startswith("no fringes in any of the 5 scans it should have been detected in")
    assert "scans 1-5" in texts(comments, "T6")[0]


def test_unsurveyed_scans_and_scattered_misses_stay_quiet():
    scans = [scan(n, ARRAY) for n in range(1, 6)] + [scan(6, ARRAY, snrs=fringing("T6"))] \
        + [scan(n, ARRAY, snrs=fringing()) for n in range(7, 30)]
    assert cm.fringe_comments(context([antenna(a) for a in ARRAY], scans)) == []


def test_a_dead_polarization_is_reported_but_a_small_asymmetry_is_not():
    survey = {"hands": ["RR", "LL"], "antennas": {
        "T6": {"scans": 12, "detected": {"RR": 11, "LL": 0}, "median": {"RR": 80.0, "LL": None}},
        "JB": {"scans": 12, "detected": {"RR": 9, "LL": 12}, "median": {"RR": 60.0, "LL": 70.0}},
        "WB": {"scans": 2, "detected": {"RR": 2, "LL": 0}, "median": {"RR": 9.0, "LL": None}},
        "EF": {"scans": 12, "detected": {"RR": 0, "LL": 0}, "median": {"RR": None, "LL": None}}}}
    comments = cm.polarization_comments(context([antenna(a) for a in ARRAY], polarization_snr=survey))
    assert [(c["antenna"], c["severity"]) for c in comments] == [("T6", "problem")]
    assert texts(comments, "T6") == ["no signal in LCP (LL): no fringes in any of the 12 surveyed scans it took part "
                                     "in, while RCP (RR) was detected in 11"]


# -- calibration --------------------------------------------------------------------------------

def test_instrumental_delay_scan_observed_and_solved_by_everyone():
    antennas = [antenna(a) for a in ARRAY] + [antenna("O8", fraction=1.0)]
    stages = [{"scan": 7, "source": "3C345", "refant": "EF", "antennas": ARRAY + ["O8"]}]
    ctx = context(antennas, [scan(7, ARRAY + ["O8"], source="3C345", role="fringefinder")], calibration_selection=stages)
    assert texts(cm.instrumental_delay_comments(ctx)) == [
        "Scan 7 (3C345) was selected for the instrumental delay (reference antenna EF): it was observed by all 5 "
        "antennas with usable data, and all were solved there."]


def test_instrumental_delay_says_who_needed_another_stage_and_who_got_none():
    stages = [{"scan": 7, "source": "3C345", "refant": "EF", "antennas": ["EF", "JB", "WB"]},
              {"scan": 2, "source": "3C84", "refant": "EF", "antennas": ["MC"]}]
    ctx = context([antenna(a) for a in ARRAY], [scan(7, ["EF", "JB", "WB", "MC"]), scan(2, ARRAY)],
                  calibration_selection=stages)
    comments = cm.instrumental_delay_comments(ctx)
    assert texts(comments) == [
        "Scan 7 (3C345) was selected for the instrumental delay (reference antenna EF): 3 of the 5 antennas with "
        "usable data were solved there; not in that scan: T6; not solved there: MC.",
        "Scan 2 (3C84) was used for the instrumental delay of MC (reference antenna EF), which the first stage does "
        "not cover."]
    assert [(c["antenna"], c["severity"]) for c in comments if c["antenna"]] == [("T6", "warning")]
    assert cm.instrumental_delay_comments(context([antenna(a) for a in ARRAY])) == []     # nothing decided yet


def gains(source: str, medians: dict, *, naive=None, per_if=None, sigma=None, findings=(), applied=True) -> dict:
    """One entry of ``gain_corrections`` over two subbands; ``per_if`` defaults to the median in both."""
    names = list(medians)
    return {"source": source, "role": "phasecal", "applied": applied, "antennas": names,
            "subband_labels": ["SB 0", "SB 1"],
            "gain_per_if": {n: (per_if or {}).get(n, [medians[n], medians[n]]) for n in names},
            "gain_median": dict(medians), "naive_gscale": {n: (naive or {}).get(n) for n in names},
            "sigma": {n: (sigma or {}).get(n) for n in names}, "p_correction": {n: None for n in names},
            "has_data": {n: medians[n] is not None for n in names}, "findings": list(findings)}


def test_large_and_moderate_amplitude_corrections_are_merged_per_antenna():
    entries = [gains("CAL1", {"EF": 1.0, "IR": 1.45, "TR": 0.85, "JB": 1.10, "O8": None}),
               gains("CAL2", {"EF": 1.02, "IR": 1.20, "TR": 0.98, "JB": 1.0, "O8": None})]
    comments = cm.gain_comments(context([antenna(a) for a in ("EF", "IR", "TR", "JB", "O8")], gain_corrections=entries))
    assert texts(comments, "IR") == ["required a large amplitude correction: x1.45 on CAL1, x1.20 on CAL2 (median "
                                     "over its subbands)"]
    assert texts(comments, "TR") == ["amplitude correction beyond 10%: x0.85 on CAL1 (median over its subbands)"]
    assert {c["antenna"]: c["severity"] for c in comments} == {"IR": "problem", "TR": "warning"}


def test_a_single_subband_beyond_thirty_percent_is_reported_when_the_median_is_not():
    entry = gains("CAL", {"EF": 1.0, "DE": 1.02}, per_if={"DE": [1.49, 1.02]})
    comments = cm.gain_comments(context([antenna("EF"), antenna("DE")], gain_corrections=[entry]))
    assert texts(comments, "DE") == ["single subbands needed a correction beyond 30% although the median did not: "
                                     "SB 0 x1.49 on CAL"]


def test_an_antenna_that_looks_off_but_was_not_corrected_is_flagged_for_a_careful_look():
    entry = gains("CAL", {"EF": 1.0, "JB": 0.99, "WB": 1.0, "MC": 1.01}, naive={"EF": 1.0, "JB": 0.78, "WB": 1.02, "MC": 1.0},
                  sigma={"JB": 0.08})
    comments = cm.gain_comments(context([antenna(a) for a in ARRAY], gain_corrections=[entry]))
    assert [c["antenna"] for c in comments] == ["JB"] and comments[0]["severity"] == "warning"
    assert texts(comments, "JB")[0] == (
        "seems to be off in amplitude but could not be fixed, as neither the other antennas nor another source "
        "constrain its gain: on CAL an unconstrained gscale asks for x0.78 but x0.99 was applied (uncertainty 8%); "
        "take a careful look at its amplitudes")


def test_an_antenna_whose_gain_another_source_pins_down_is_not_called_off():
    loose = gains("FF", {"EF": 1.0, "JB": 0.99, "WB": 1.0, "MC": 1.0}, naive={"EF": 1.0, "JB": 0.78, "WB": 1.0, "MC": 1.0})
    tight = gains("PC", {"EF": 1.0, "JB": 1.01, "WB": 1.0, "MC": 1.0}, naive={"EF": 1.0, "JB": 1.02, "WB": 1.0, "MC": 1.0})
    assert cm.gain_comments(context([antenna(a) for a in ARRAY], gain_corrections=[loose, tight])) == []


def test_a_source_where_half_the_antennas_look_off_is_reported_as_poorly_constrained():
    entry = gains("3C345", {"EF": 1.0, "JB": 1.0, "WB": 0.99, "MC": 0.99, "IR": 1.0, "DE": 0.99},
                  naive={"EF": 0.75, "JB": 0.90, "WB": 0.88, "MC": 0.98, "IR": 1.13, "DE": 1.0})
    comments = cm.gain_comments(context([antenna(a) for a in ("EF", "JB", "WB", "MC", "IR", "DE")],
                                        gain_corrections=[entry]))
    assert [c["category"] for c in comments] == ["calibration"] and comments[0]["severity"] == "warning"
    assert "3 of its 6 antennas by more than 10% (EF x0.75, WB x0.88, IR x1.13)" in comments[0]["text"]
    assert comments[0]["text"].startswith("The amplitude self-calibration of 3C345 is poorly constrained")


def test_one_station_with_a_wrong_scale_is_blamed_instead_of_the_source():
    """EB104D: T6 ~20x low drags the model down; the others all look "x0.5", T6 is the one that is off."""
    names = ("WB", "EF", "O6", "T6", "UR", "TR", "YS")
    pc = gains("J1031", {"WB": 0.99, "EF": 0.99, "O6": 0.98, "T6": 1.29, "UR": 0.99, "TR": 0.99, "YS": 0.99},
               naive={"WB": 0.54, "EF": 0.54, "O6": 0.48, "T6": 4.70, "UR": 0.50, "TR": 0.45, "YS": 0.40})
    ff = gains("J1048", {"WB": 0.99, "EF": 1.0, "O6": 1.0, "T6": 1.03, "UR": 1.0, "TR": 0.99, "YS": 0.99},
               naive={"WB": 0.65, "EF": 0.91, "O6": 0.75, "T6": 2.83, "UR": 0.76, "TR": 0.68, "YS": 0.60})
    comments = cm.gain_comments(context([antenna(a) for a in names], gain_corrections=[pc, ff]))
    problems = [c for c in comments if c["severity"] == "problem"]
    assert [c["antenna"] for c in problems] == ["T6"]
    assert problems[0]["text"].startswith("amplitude scale looks wrong against the rest of the array: relative to "
                                          "the array median, an unconstrained gscale asks for x9.40 on J1031, x3.77 "
                                          "on J1048, but at most x1.29 was applied")
    assert "(ANTAB Tsys, DPFU, FT)" in problems[0]["text"]
    source_level = [c for c in comments if c["category"] == "calibration"]
    assert source_level and all("most likely the model following T6" in c["text"] for c in source_level)
    assert not any(c["antenna"] == "T6" and "could not be fixed" in c["text"] for c in comments)


def test_report_findings_about_the_source_are_passed_on_condensed_and_unapplied_gains_are_flagged():
    findings = ["Best source model: clean (posterior probability 1.000)",
                "JB: leaving it out of the model moves its gain by -77.6% (median over IFs)",
                "Net flux-scale change: IF1 -1.00%, IF2 -1.29%, IF3 -1.48%, IF4 +0.20%",
                "3 of 36 runs failed and were left out (see `runs`)"]
    entry = gains("CAL", {"EF": 1.0, "JB": 1.0}, findings=findings, applied=False)
    comments = cm.gain_comments(context([antenna("EF"), antenna("JB")], gain_corrections=[entry]))
    assert texts(comments) == [
        "The amplitude corrections of CAL were computed but not applied to the data.",
        "Amplitude self-calibration of CAL: net flux-scale change: -1.48% to +0.20% over the 4 IFs; 3 of 36 runs "
        "failed and were left out."]


# -- results ------------------------------------------------------------------------------------

def test_flagged_fraction_is_given_with_and_without_the_dead_antennas():
    flagging = {"fraction": 0.427, "excluding_dead": {"fraction": 0.240, "excluded_antennas": ["O8", "CM"]}}
    comments = cm.flagging_comments(context([], flagging=flagging))
    assert texts(comments) == ["42.7% of the cross-correlation data is flagged (24.0% without the antennas that have "
                               "no usable signal: O8 and CM)."]
    assert comments[0]["severity"] == "note" and comments[0]["category"] == "result"
    heavy = cm.flagging_comments(context([], flagging={"fraction": 0.62, "excluding_dead": {"fraction": None}}))
    assert heavy[0]["severity"] == "warning" and heavy[0]["text"] == "62.0% of the cross-correlation data is flagged."
    assert cm.flagging_comments(context([], flagging={"fraction": None})) == []


def test_source_results_combine_the_search_and_the_clean_image():
    results = [
        {"source": "TGT", "role": "target", "images": [],
         "search": {"detected": True, "recentred": True, "snr": 26.6, "peak": 1.36e-4, "fov_mas": 1000.0,
                    "offset_mas": 120.0, "offset_beams": 41.0}},
        {"source": "PC", "role": "phasecal",
         "search": {"detected": True, "recentred": False, "snr": 425.0, "peak": 0.18, "offset_mas": 0.0, "offset_beams": 0.0},
         "images": [{"robust": -2.0, "peak": 0.12, "rms": 7e-5, "dynamic_range": 1800.0},
                    {"robust": 0.0, "peak": 0.136, "rms": 6.23e-5, "dynamic_range": 2175.0}]},
        {"source": "FF", "role": "fringefinder", "search": None,
         "images": [{"robust": 0.0, "peak": 3.24e-3, "rms": 1.47e-4, "dynamic_range": 22.0}]},
        {"source": "CHK", "role": "checksource", "images": [],
         "search": {"detected": False, "recentred": False, "snr": 4.2, "peak": 1e-4, "fov_mas": 1000.0}}]
    comments = cm.source_comments(context([], source_results=results))
    assert texts(comments) == [
        "TGT (target): detected at 27 sigma in the search map (dirty-map peak 136 uJy/beam), 120.0 mas (41.0 beams) "
        "from the phase centre and re-centred on the peak.",
        "PC (phasecal): detected at 425 sigma in the search map, at the phase centre; CLEAN image (robust 0): peak "
        "136 mJy/beam, rms 62.3 uJy/beam, dynamic range 2175.",
        "FF (fringefinder): CLEAN image (robust 0): peak 3.24 mJy/beam, rms 147 uJy/beam, dynamic range 22 - low for "
        "a calibrator, check its calibration and model.",
        "CHK (checksource): not detected in the 1000 mas search map (brightest peak at 4.2 sigma)."]
    assert [c["severity"] for c in comments] == ["note", "note", "warning", "warning"]


def test_poor_tables_and_failed_steps_are_reported():
    tables = [{"cal_type": "tsys", "snr": 0.0, "step": "a_priori"}, {"cal_type": "mbd", "snr": 4.2, "step": "fringefit"},
              {"cal_type": "sbd", "snr": 3951.0, "step": "initial_calibration"}]
    steps = {"import_data": {"status": "done"}, "bandpass": {"status": "failed", "error": "no solutions"}}
    ctx = context([], caltables=tables, steps=steps)
    assert texts(cm.table_comments(ctx)) == ["The mbd table (step fringefit) has a low median solution SNR: 4.2."]
    failed = cm.step_comments(ctx)
    assert texts(failed) == ["Step bandpass failed: no solutions"]
    assert failed[0]["severity"] == "problem" and failed[0]["category"] == "pipeline"


# -- warnings of the pipeline -------------------------------------------------------------------

def run_warning(message: str, *, anomaly: bool = False, seen: str = "2026-10-10 12:07:00") -> dict:
    """One entry of the context ``run_warnings`` list."""
    return {"message": message, "anomaly": anomaly, "first_seen_utc": seen}


def test_pipeline_warnings_are_filed_under_their_antenna_and_not_repeated():
    recorded = [run_warning("antenna T6 shows no usable fringes on 3C345"),
                run_warning("JB has no mbd solution in subband(s) [3] although it recorded them", anomaly=True),
                run_warning("antenna O8 has 100% of its recorded data flagged", anomaly=True),
                run_warning("flag[quack] removed 61.0% of data", anomaly=True),
                run_warning("no phase self-cal step improved the fit on 3C345")]
    ctx = context([antenna(a) for a in ARRAY] + [antenna("O8", fraction=1.0)], run_warnings=recorded)
    comments = cm.pipeline_comments(ctx, reported={"O8"})
    assert texts(comments, "T6") == ["shows no usable fringes on 3C345"]
    assert texts(comments, "JB") == ["has no mbd solution in subband(s) [3] although it recorded them"]
    assert not texts(comments, "O8")                                  # the data rule already says it
    assert texts(comments) == ["flag[quack] removed 61.0% of data", "no phase self-cal step improved the fit on 3C345"]
    by_text = {c["text"]: c for c in comments}
    assert by_text["flag[quack] removed 61.0% of data"]["severity"] == "problem"
    assert by_text["no phase self-cal step improved the fit on 3C345"]["severity"] == "warning"
    assert by_text["shows no usable fringes on 3C345"]["category"] == "antenna"
    assert all(c["when"] == "2026-10-10 12:07:00" for c in comments)


def test_pipeline_warnings_are_capped_with_a_count_of_the_rest():
    recorded = [run_warning(f"plot {index} failed") for index in range(cm.MAX_PIPELINE_WARNINGS + 4)]
    recorded.append(run_warning("flagging left no data", anomaly=True))
    comments = cm.pipeline_comments(context([], run_warnings=recorded), reported=set())
    assert len(comments) == cm.MAX_PIPELINE_WARNINGS + 1
    assert comments[0]["text"] == "flagging left no data"             # anomalies survive the cap
    assert comments[-1]["text"].startswith("5 more warnings of the pipeline are not listed here")
    assert cm.RUN_WARNINGS_FILENAME in comments[-1]["text"]


# -- assembly and text --------------------------------------------------------------------------

def full_context() -> dict:
    """A context that triggers one comment of every group."""
    scans = [scan(n, [a for a in ARRAY if not (a == "JB" and n >= 6)], snrs=fringing()) for n in range(1, 11)]
    return context(
        [antenna(a) for a in ARRAY] + [antenna("TR", observed=False, subbands=(), fraction=0.0)], scans,
        calibration_selection=[{"scan": 1, "source": "CAL", "refant": "EF", "antennas": ARRAY}],
        flagging={"fraction": 0.3, "excluding_dead": {"fraction": None, "excluded_antennas": []}},
        steps={"bandpass": {"status": "failed", "error": "no solutions"}},
        run_warnings=[run_warning("antenna WB shows no usable fringes on CAL"), run_warning("uv plot failed")])


def test_build_comments_groups_and_orders_everything():
    comments = cm.build_comments(full_context())
    assert [c["category"] for c in comments] == ["calibration", "result", "antenna", "antenna", "antenna", "antenna",
                                                 "pipeline", "pipeline"]
    stations = [c["antenna"] for c in comments if c["category"] == "antenna"]
    assert stations == ["JB", "WB", "TR", ""]                         # context order, then the closing line
    assert comments[5]["text"] == "Nothing to report for: EF, MC, T6."
    assert [c["severity"] for c in comments if c["category"] == "pipeline"] == ["problem", "warning"]
    assert all(set(c) >= {"severity", "category", "antenna", "text"} for c in comments)
    json.dumps(comments)                                              # the list goes into dashboard.json


def test_build_comments_tolerates_an_empty_or_old_context():
    assert cm.build_comments({}) == []
    assert cm.build_comments({"antennas": [antenna("EF")], "scans": []}) == [
        {"severity": "note", "category": "antenna", "antenna": "", "text": "Nothing to report for: EF."}]


def test_comments_text_is_a_fixed_width_block_grouped_like_the_page():
    block = cm.comments_text(cm.build_comments(full_context()), "t01")
    lines = block.splitlines()
    assert lines[0] == "General Comments for t01" and lines[1] == "=" * 100
    assert lines.index("Calibration and results") < lines.index("Antennas") < lines.index("Warnings raised by the pipeline")
    assert "  JB  WARNING  no data from 10:06 UTC to the end of the observation (scans 6-10)" in lines
    assert "  WB  WARNING  shows no usable fringes on CAL [first seen 2026-10-10 12:07:00 UTC]" in lines
    assert "  Nothing to report for: EF, MC, T6." in lines
    assert "  PROBLEM  Step bandpass failed: no solutions" in lines
    assert "  WARNING  uv plot failed [first seen 2026-10-10 12:07:00 UTC]" in lines
    assert all(len(line) <= 100 for line in lines)
    wrapped = [line for line in lines if line.startswith("           ")]
    assert wrapped, "long sentences continue under the text, not under the label"


def test_comments_text_before_the_pipeline_has_run_and_with_nothing_to_say():
    pending = cm.comments_text(None, "t01")
    assert pending.splitlines()[0] == "General Comments for t01"
    assert "filled in once the pipeline has run" in pending
    assert cm.comments_text([], "t01").splitlines()[-1] == "  Nothing to report."


# -- keeping the warnings between processes -----------------------------------------------------

def test_run_warnings_accumulate_keep_their_first_time_and_skip_other_projects(tmp_path):
    first = cm.merge_run_warnings(tmp_path, ["t01: antenna T6 shows no usable fringes on 3C345",
                                             "ANOMALY: t02: flag[quack] removed 70.0% of data",
                                             "imaging 3C345 failed: boom"], other_projects=["t02"])
    assert set(first) == {"t01: antenna T6 shows no usable fringes on 3C345", "imaging 3C345 failed: boom"}
    path = tmp_path / cm.RUN_WARNINGS_FILENAME
    stored = json.loads(path.read_text())
    stored["t01: antenna T6 shows no usable fringes on 3C345"] = "2026-01-01 00:00:00"
    path.write_text(json.dumps(stored))
    second = cm.merge_run_warnings(tmp_path, ["t01: antenna T6 shows no usable fringes on 3C345",
                                              "ANOMALY: t01: JB has no mbd solution in subband(s) [3]"])
    assert second["t01: antenna T6 shows no usable fringes on 3C345"] == "2026-01-01 00:00:00"
    loaded = cm.load_run_warnings(tmp_path, "t01")
    assert loaded[0] == {"message": "antenna T6 shows no usable fringes on 3C345", "anomaly": False,
                         "first_seen_utc": "2026-01-01 00:00:00"}
    assert {"message": "JB has no mbd solution in subband(s) [3]", "anomaly": True,
            "first_seen_utc": second["ANOMALY: t01: JB has no mbd solution in subband(s) [3]"]} in loaded
    assert len(loaded) == 3


def test_no_warning_file_is_written_without_warnings_and_a_missing_or_corrupt_one_reads_empty(tmp_path):
    assert cm.merge_run_warnings(tmp_path, []) == {}
    assert not (tmp_path / cm.RUN_WARNINGS_FILENAME).exists()
    assert cm.load_run_warnings(tmp_path, "t01") == []
    (tmp_path / cm.RUN_WARNINGS_FILENAME).write_text("{not json")
    assert cm.load_run_warnings(tmp_path, "t01") == []


@pytest.fixture
def quiet_collector():
    """Leave the process-wide warning collector as it was found."""
    before = len(collector._items)
    yield collector
    del collector._items[before:]


def test_observation_keeps_only_its_own_new_warnings_and_scratch_forgets_them(tmp_path, quiet_collector):
    quiet_collector.warn("db01: raised before the observation existed")
    obs = VLBIObs("db01", network="EVN", backend="dummy", work_dir=tmp_path / "db01", target="3C395")
    observation = obs["db01"]
    path = tmp_path / "db01" / cm.RUN_WARNINGS_FILENAME
    quiet_collector.warn("db01: antenna T6 shows no usable fringes on 3C345")
    observation.persist_run_warnings()
    assert not path.exists()                                          # the dummy backend keeps nothing on disk

    observation._backend.requires_data_files = True
    (tmp_path / "db01").mkdir(parents=True, exist_ok=True)
    (tmp_path / "db01" / ".project.json").write_text(json.dumps({"projects": ["db01", "db02"]}))
    quiet_collector.anomaly("db02: belongs to the other epoch")
    observation.persist_run_warnings()
    assert [w["message"] for w in cm.load_run_warnings(tmp_path / "db01", "db01")] == [
        "antenna T6 shows no usable fringes on 3C345"]
    observation.prepare_run(scratch=True)
    assert not path.exists()
