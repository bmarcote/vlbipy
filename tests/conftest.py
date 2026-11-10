"""Pytest configuration: make the src-layout package importable and quiet logs."""
import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vlbipy.logging_utils import configure_logging  # noqa: E402

# Keep test output readable: only show warnings and above.
configure_logging("WARNING")


@pytest.fixture(autouse=True)
def _no_archive_network(monkeypatch):
    """Tests never reach the archives: an http(s) fetch the code under test tries fails as if offline (local
    ``file://`` URLs still work; tests that exercise a download monkeypatch these themselves)."""
    from vlbipy import tools

    def offline(function):
        def guarded(url, *args, **kwargs):
            if str(url).startswith(("http://", "https://")):
                raise OSError("network disabled in the tests")
            return function(url, *args, **kwargs)
        return guarded

    monkeypatch.setattr(tools, "download_file", offline(tools.download_file))
    monkeypatch.setattr(tools, "fetch_url_text", offline(tools.fetch_url_text))
