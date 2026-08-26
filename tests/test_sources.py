"""Tests for Source / SourceSet role handling and phase-referencing."""
import pytest

from vlbipy.errors import SourceNotFoundError
from vlbipy.models import SourceType
from vlbipy.sources import SourceSet


def _cfg(**roles):
    return {k: v for k, v in roles.items()}


def test_roles_and_accessors():
    ss = SourceSet.from_config(_cfg(targets="3C286", phase_calibrators=["J1048+7143"],
                                    fringe_finders=["3C345"], check_sources=["JCHK"]))
    assert ss.target.name == "3C286"
    assert [s.name for s in ss.phase_calibrators] == ["J1048+7143"]
    assert [s.name for s in ss.fringe_finders] == ["3C345"]
    assert [s.name for s in ss.check_sources] == ["JCHK"]
    assert {s.name for s in ss.calibrators} == {"J1048+7143", "3C345"}
    assert ss["3C345"].source_type is SourceType.FRINGE_FINDER
    assert "3C286" in ss and len(ss) == 4


def test_ambiguous_target_raises():
    ss = SourceSet.from_config(_cfg(targets=["A", "B"]))
    with pytest.raises(SourceNotFoundError):
        _ = ss.target
    assert [s.name for s in ss.targets] == ["A", "B"]


def test_missing_target_raises():
    ss = SourceSet.from_config(_cfg(fringe_finders=["3C345"]))
    with pytest.raises(SourceNotFoundError):
        _ = ss.target


def test_unknown_name_raises():
    ss = SourceSet.from_config(_cfg(targets="3C286"))
    with pytest.raises(SourceNotFoundError):
        _ = ss["nope"]


def test_phase_referencing_explicit_mapping():
    ss = SourceSet.from_config(
        _cfg(targets=["T1"], phase_calibrators=["C1", "C2"], fringe_finders=["FF"]),
        phaseref_cfg={"T1": ["C2", "C1"]})
    cals = ss.calibrators_for("T1")
    assert [c.name for c in cals] == ["C2", "C1"]


def test_phase_referencing_fallback_order():
    # No mapping, but fringe finder present -> fall back to fringe finders.
    ss = SourceSet.from_config(_cfg(targets=["T1"], phase_calibrators=["C1"], fringe_finders=["FF"]))
    assert [c.name for c in ss.calibrators_for("T1")] == ["FF"]
    # No fringe finder -> fall back to phase calibrators.
    ss2 = SourceSet.from_config(_cfg(targets=["T1"], phase_calibrators=["C1"]))
    assert [c.name for c in ss2.calibrators_for("T1")] == ["C1"]
    # Nothing -> fall back to the target itself.
    ss3 = SourceSet.from_config(_cfg(targets=["T1"]))
    assert [c.name for c in ss3.calibrators_for("T1")] == ["T1"]
