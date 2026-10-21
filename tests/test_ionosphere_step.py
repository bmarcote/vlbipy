"""The ionospheric (TEC) correction step: when it runs, how its table joins the chain and how it is applied."""
from __future__ import annotations

import pytest

from vlbipy import VLBIObs
from vlbipy.errors import StepError
from vlbipy.models import CalTable


def _observation(network: str, monkeypatch, *, ionosphere=None, enabled="auto", threshold=None):
    obs = VLBIObs("tec01", network=network, backend="dummy", target="SRC")
    obs.import_data()
    observation = obs.observations[0]
    observation.config.setdefault("calibration", {})["tec"] = {"enabled": enabled, "model": "jplg"}
    if threshold is not None:
        monkeypatch.setattr(type(observation._observatory_handler), "tec_below_ghz", threshold)
    calls = []
    if ionosphere is not None:
        def method(self, project_code, *, ionex_dir, **kwargs):
            calls.append({"ionex_dir": ionex_dir, **kwargs})
            return ionosphere()
        monkeypatch.setattr(type(observation._backend.calibrate), "ionosphere", method, raising=False)
    return obs, observation, calls


def _table():
    return CalTable(cal_type="tec", path="x.tec", field="SRC", interp="linear", calwt=False, step="tec")


def test_runs_below_the_threshold_and_joins_the_chain(monkeypatch):
    obs, observation, calls = _observation("EVN", monkeypatch, ionosphere=_table, threshold=1000.0)
    table = observation.calibrate.ionosphere()
    assert table.cal_type == "tec" and [t.cal_type for t in observation.gaintables] == ["tec"]
    assert calls and calls[0]["model"] == "jplg" and calls[0]["ionex_dir"].endswith("ionex")
    assert observation.state.status("tec") == "done"


def test_skipped_above_the_threshold_or_when_disabled(monkeypatch):
    _, observation, calls = _observation("VLBA", monkeypatch, ionosphere=_table, threshold=0.001)
    assert observation.calibrate.ionosphere() is None and not calls
    _, observation, calls = _observation("VLBA", monkeypatch, ionosphere=_table, enabled=False, threshold=1000.0)
    assert observation.calibrate.ionosphere() is None and not calls
    _, observation, calls = _observation("LBA", monkeypatch, ionosphere=_table, enabled=True, threshold=0.001)
    assert observation.calibrate.ionosphere() is not None and calls                 # forced


def test_a_backend_without_it_skips_the_step(monkeypatch):
    _, observation, _ = _observation("EVN", monkeypatch, threshold=1000.0)
    assert observation.calibrate.ionosphere() is None and not observation.gaintables


def test_missing_maps_warn_under_auto_and_fail_when_asked_for(monkeypatch):
    def missing():
        raise FileNotFoundError("no IONEX file for 2025-259 (tried ...)")
    _, observation, _ = _observation("EVN", monkeypatch, ionosphere=missing, threshold=1000.0)
    assert observation.calibrate.ionosphere() is None and observation.state.status("tec") == "done"
    _, observation, _ = _observation("EVN", monkeypatch, ionosphere=missing, enabled=True)
    with pytest.raises(StepError):
        observation.calibrate.ionosphere()


def test_each_field_is_corrected_with_its_own_rows():
    from vlbipy.backends.casa import CasaCalibrationOps
    table = _table()
    assert CasaCalibrationOps._resolve_table_gainfield(None, table, "J1848+3219") == "J1848+3219"
    assert CasaCalibrationOps._resolve_table_gainfield(None, table, "3C345,J1848+3219") == ""
    assert CasaCalibrationOps._resolve_table_interp(table, "J1848+3219") == "linear"
