<<<<<<< HEAD
"""A receptor much weaker than the other one of its antenna is flagged before any solve."""
from __future__ import annotations

import math

=======
"""A receptor much weaker than the other one of its antenna is flagged."""
from __future__ import annotations

>>>>>>> 5d3198be0e79117f058939871de32e9e7c0bf006
from vlbipy import VLBIObs
from vlbipy.models import ScanSNRSurvey

NAN = float("nan")


def _survey(rr, ll):
    """Three antennas (AA is the reference, nan), one row per scan."""
    return ScanSNRSurvey(project_code="TS01", scan_numbers=list(range(1, len(rr) + 1)),
                         scan_sources=["CAL"] * len(rr), antennas=["AA", "BB", "CC"],
                         snr={"RR": rr, "LL": ll}, refant="AA")


<<<<<<< HEAD
def test_weak_and_dead_hands():
    # BB: LL eleven times weaker (V589A's CD); CC: RR never solved while LL is detected (dead).
    rr = [[NAN, 110.0, NAN], [NAN, 90.0, NAN], [NAN, 100.0, NAN], [NAN, 4.0, NAN]]
    ll = [[NAN, 10.0, 60.0], [NAN, 8.0, 50.0], [NAN, 9.0, 70.0], [NAN, 3.0, 40.0]]
    weak = _survey(rr, ll).weak_hands(5.0)
    assert weak["BB"][0] == "LL" and 10.0 < weak["BB"][1] < 12.0
    assert weak["CC"][0] == "RR" and math.isinf(weak["CC"][1])
    assert "AA" not in weak


def test_comparable_hands_or_too_few_detections_are_left_alone():
    rr = [[NAN, 40.0, 8.0], [NAN, 35.0, 2.0], [NAN, 30.0, 2.0]]
    ll = [[NAN, 12.0, NAN], [NAN, 10.0, NAN], [NAN, 9.0, NAN]]      # BB: x3.3; CC: one detection only
    assert _survey(rr, ll).weak_hands(5.0) == {}
    assert _survey(rr, ll).weak_hands(3.0)["BB"][0] == "LL"


def test_the_step_flags_the_weak_receptor_and_its_cross_hands(monkeypatch):
    obs = VLBIObs("TS01", network="EVN", backend="dummy", target="SRC", phasecal="CAL")["TS01"]
    survey = _survey([[NAN, 110.0, 50.0]] * 3, [[NAN, 10.0, 45.0]] * 3)
    monkeypatch.setattr(type(obs.calibrate), "scan_snr", lambda self, **kwargs: survey)
    calls = []
    monkeypatch.setattr(obs._backend.flag, "run", lambda code, kind, **sel: calls.append((kind, sel)) or 0.0)
    weak = obs.flag.weak_polarizations(force=True)
    assert list(weak) == ["BB"]
=======
def test_a_hand_with_only_noise_fringes_is_dead():
    # BB: LL at the noise floor while RR is detected (V589A's KE); CC: LL never solved (V589D's TI).
    rr = [[NAN, 17.0, 110.0], [NAN, 12.0, 75.0], [NAN, 15.0, 100.0], [NAN, 4.0, 90.0]]
    ll = [[NAN, 3.3, NAN], [NAN, 4.8, NAN], [NAN, 5.4, NAN], [NAN, 3.5, NAN]]
    dead = _survey(rr, ll).dead_hands(detection=7.0)
    assert dead["BB"][0] == "LL" and dead["BB"][1] < 7.0
    assert dead["CC"] == ("LL", 0.0, 95.0)            # median of 110, 75, 100, 90
    assert "AA" not in dead


def test_detected_hands_are_not_judged_on_their_snr():
    """V589A's CD (LL fringe SNR ~18 against ~75, an amplitude 11 times lower) and V589D's MP (LL SNR 3-7
    times that of an RR of the same amplitude): both hands detected, so nothing is decided here."""
    rr = [[NAN, 75.0, 70.0], [NAN, 80.0, 105.0], [NAN, 63.0, 77.0]]
    ll = [[NAN, 18.0, 300.0], [NAN, 22.0, 480.0], [NAN, 15.0, 415.0]]
    assert _survey(rr, ll).dead_hands(detection=7.0) == {}


def _observation(monkeypatch):
    obs = VLBIObs("TS01", network="EVN", backend="dummy", target="SRC", phasecal="CAL")["TS01"]
    calls = []
    monkeypatch.setattr(obs._backend.flag, "run", lambda code, kind, **sel: calls.append((kind, sel)) or 0.0)
    return obs, calls


def test_the_survey_step_flags_a_dead_receptor_and_its_cross_hands(monkeypatch):
    obs, calls = _observation(monkeypatch)
    survey = _survey([[NAN, 17.0, 70.0]] * 3, [[NAN, 4.0, 300.0]] * 3)
    monkeypatch.setattr(type(obs.calibrate), "scan_snr", lambda self, **kwargs: survey)
    assert obs.flag.weak_polarizations(force=True) == {"BB": "LL"}
>>>>>>> 5d3198be0e79117f058939871de32e9e7c0bf006
    assert calls == [("manual", {"antenna": "BB", "correlation": "LL,RL,LR"})]
    assert obs._state.status("flag_weak_polarization") == "done"
    obs.config["flagging"]["weak_polarization_ratio"] = 0       # off
    calls.clear()
    assert obs.flag.weak_polarizations(force=True) == {} and calls == []
<<<<<<< HEAD
=======


def test_amplitudes_of_the_scalar_bandpass_decide_a_weak_receptor(monkeypatch):
    obs, calls = _observation(monkeypatch)
    levels = {"CD": {"ratio": None, "pairs": 0, "solved": {"RR": 16, "LL": 0}},      # LCP never solved
              "WK": {"ratio": 0.15, "pairs": 16, "solved": {"RR": 16, "LL": 16}},    # RCP 6.7x weaker
              "MP": {"ratio": 1.02, "pairs": 16, "solved": {"RR": 16, "LL": 16}},    # fine
              "KE": {"ratio": None, "pairs": 0, "solved": {"RR": 0, "LL": 0}}}       # nothing: not this rule
    assert obs.flag.weak_polarizations_from_gains(levels) == {"CD": "LL", "WK": "RR"}
    assert ("manual", {"antenna": "WK", "correlation": "RR,RL,LR"}) in calls
    obs.config["flagging"]["weak_polarization_ratio"] = 8.0
    assert obs.flag.weak_polarizations_from_gains(levels) == {"CD": "LL"}
>>>>>>> 5d3198be0e79117f058939871de32e9e7c0bf006
