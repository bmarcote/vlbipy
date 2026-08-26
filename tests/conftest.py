"""Pytest configuration: make the src-layout package importable and quiet logs."""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vlbipy.logging_utils import configure_logging  # noqa: E402

# Keep test output readable: only show warnings and above.
configure_logging("WARNING")
