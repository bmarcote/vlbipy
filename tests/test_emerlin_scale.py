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
