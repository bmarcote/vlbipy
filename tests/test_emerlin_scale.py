from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vlbipy.namespaces import CalibrateNamespace


@pytest.mark.parametrize("antennas", [[], ["AT", "PA", "MP"], ["Jb1"]])
@pytest.mark.parametrize("force", [False, True])
def test_no_outstations_returns_before_step_announcement(antennas, force):
    obs = SimpleNamespace(metadata=SimpleNamespace(antennas=antennas), _state=MagicMock())
    namespace = object.__new__(CalibrateNamespace)
    namespace._obs = obs

    assert namespace.emerlin_scale(force=force) is None
    obs._state.should_run.assert_not_called()
    obs._state.mark_complete.assert_not_called()


@pytest.mark.parametrize("config, expected", [({}, ["Cm", "Kn", "Jb1"]),
                                              ({"calibration": {"emerlin": {"hold_reference": True}}}, ["Cm", "Kn"])])
def test_jb_is_solved_with_the_outstations_by_default(config, expected):
    """Jb never triggers the e-MERLIN flux scale but is solved (and later held) with the outstations."""
    from vlbipy.namespaces import emerlin_antennas
    obs = SimpleNamespace(gaintables=[SimpleNamespace(cal_type="emgain")], config=config,
                          metadata=SimpleNamespace(antennas=["EF", "Cm", "Kn", "Jb1"]))
    assert emerlin_antennas(obs) == expected
