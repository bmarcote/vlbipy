"""Tests for the configuration cascade and kwarg mapping."""
import pytest

from vlbipy.config import kwargs_to_overrides, load_config
from vlbipy.errors import ConfigError


def test_defaults_loaded():
    cfg = load_config()
    assert cfg["global"]["backend"] == "casa"
    assert cfg["global"]["observatory"] == "EVN"
    assert cfg["imaging"]["robust"] == [-2, -1, 0, 1, 2]
    assert "phase_referencing" in cfg


def test_dict_override_merges_deeply():
    cfg = load_config({"global": {"observatory": "VLBA"}})
    assert cfg["global"]["observatory"] == "VLBA"
    # untouched keys survive the deep merge
    assert cfg["global"]["backend"] == "casa"
    assert cfg["imaging"]["robust"] == [-2, -1, 0, 1, 2]


def test_overrides_win_over_user_config():
    cfg = load_config({"global": {"observatory": "VLBA"}}, overrides={"global": {"observatory": "LBA"}})
    assert cfg["global"]["observatory"] == "LBA"


def test_kwargs_to_overrides():
    ov = kwargs_to_overrides(network="EVN", target="3C286", phasecal=["J1", "J2"],
                             fringe_finder="3C345", refant="EF")
    assert ov["global"]["observatory"] == "EVN"
    assert ov["sources"]["targets"] == ["3C286"]
    assert ov["sources"]["phase_calibrators"] == ["J1", "J2"]
    assert ov["sources"]["fringe_finders"] == ["3C345"]
    assert ov["global"]["reference_antenna"] == ["EF"]


def test_kwargs_to_overrides_ignores_none():
    assert kwargs_to_overrides() == {}
    assert kwargs_to_overrides(network=None, target=None) == {}


def test_bad_config_type_raises():
    with pytest.raises(ConfigError):
        load_config(12345)


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/no/such/file.toml")
