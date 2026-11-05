"""The CASA measures directory (``~/.casa/data``) that casatools and python-casacore need before they import."""
from __future__ import annotations

from vlbipy.tools import CASA_DATA_DIR, ensure_casa_data_dir


def test_missing_casa_data_dir_is_created(tmp_path):
    target = tmp_path / "home" / ".casa" / "data"
    assert ensure_casa_data_dir(target) == target
    assert target.is_dir()


def test_existing_casa_data_dir_is_left_untouched(tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    (target / "geodetic").mkdir()
    assert ensure_casa_data_dir(target) == target
    assert (target / "geodetic").is_dir()


def test_default_is_the_casa_home_data_dir():
    assert CASA_DATA_DIR.parts[-2:] == (".casa", "data")
