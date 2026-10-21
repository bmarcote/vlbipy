"""Pytest configuration: make the src-layout package importable, keep CASA's log out of the repo, quiet logs."""
import os
import pathlib
import sys
import tempfile

import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# casatools opens its log (casa-<date>.log) in the working directory when it is first imported, and it must load
# before python-casacore: import it here, from a scratch directory, so that later imports reuse that log.
_CASA_LOG_DIR = pathlib.Path(tempfile.gettempdir()) / "vlbipy-tests-casa-logs"
_CASA_LOG_DIR.mkdir(exist_ok=True)
_cwd = os.getcwd()
try:
    os.chdir(_CASA_LOG_DIR)
    import casatools  # noqa: F401,E402
except ImportError:
    pass
finally:
    os.chdir(_cwd)

from vlbipy.logging_utils import configure_logging  # noqa: E402
from vlbipy.solvers.msio import DEFAULT_ENGINE_ENV  # noqa: E402

# Keep test output readable: only show warnings and above.
configure_logging("WARNING")


@pytest.fixture(autouse=True)
def _default_table_engine(monkeypatch):
    """Every test starts with no backend-chosen table engine (constructing a backend sets one for the process)."""
    monkeypatch.delenv(DEFAULT_ENGINE_ENV, raising=False)
