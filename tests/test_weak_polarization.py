"""A receptor much weaker than the other one of its antenna is flagged before any solve."""
from __future__ import annotations

import math

from vlbipy import VLBIObs
from vlbipy.models import ScanSNRSurvey

NAN = float("nan")


def _survey(rr, ll):
    """Three antennas (AA is the reference, nan), one row per scan."""
    return ScanSNRSurvey(project_code="TS01", scan_numbers=list(range(1, len(rr) + 1)),
                         scan_sources=["CAL"] * len(rr), antennas=["AA", "BB", "CC"],
                         snr={"RR": rr, "LL": ll}, refant="AA")


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
    assert calls == [("manual", {"antenna": "BB", "correlation": "LL,RL,LR"})]
    assert obs._state.status("flag_weak_polarization") == "done"
    obs.config["flagging"]["weak_polarization_ratio"] = 0       # off
    calls.clear()
    assert obs.flag.weak_polarizations(force=True) == {} and calls == []
