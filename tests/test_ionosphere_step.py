"""The ionospheric (TEC) correction: when it runs and how its table joins the chain."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vlbipy import VLBIObs
from vlbipy.models import CalTable


def _observation(network: str, monkeypatch, *, freq_ghz: float, cfg: dict | None = None):
    """An observation whose a_priori backend call only records the keyword it was given."""
    obs = VLBIObs("tec01", network=network, backend="dummy", target="SRC")
    obs.import_data()
    observation = obs.observations[0]
    observation._metadata = SimpleNamespace(freq_setup=SimpleNamespace(freq_ghz=freq_ghz),
                                            antennas={}, source_names=[])
    for key, value in (cfg or {}).items():
        observation.config.setdefault("calibration", {})[key] = value
    calls = []
    backend = MagicMock(return_value=[])
    observation._backend.calibrate.a_priori = backend
    calls.append(backend)
    return observation, backend


def _needs_tec(observation):
    observation.calibrate.a_priori()
    return observation._backend.calibrate.a_priori.call_args.kwargs["needs_tec"]


def test_tec_is_requested_below_8_ghz_for_every_network(monkeypatch):
    for network in ("EVN", "VLBA", "LBA"):
        obs, _ = _observation(network, monkeypatch, freq_ghz=2.3)
        assert _needs_tec(obs) is True


def test_tec_is_skipped_above_8_ghz_or_when_disabled(monkeypatch):
    obs, _ = _observation("EVN", monkeypatch, freq_ghz=10.0)
    assert _needs_tec(obs) is False
    obs, _ = _observation("EVN", monkeypatch, freq_ghz=2.3, cfg={"ionos": False})
    assert _needs_tec(obs) is False
    obs, _ = _observation("EVN", monkeypatch, freq_ghz=6.0, cfg={"ionos_max_ghz": 5.0})
    assert _needs_tec(obs) is False


def test_tec_table_spec_is_in_the_chain():
    from vlbipy.backends.casa import APRIORI_TABLE_SPECS
    assert APRIORI_TABLE_SPECS["tec"] == (".tec", "linear")


def test_daskms_tec_table_uses_the_native_solver(tmp_path, monkeypatch):
    """The dask-ms backend produces the Fringe Jones TEC table natively (CASA's TECOR is buggy)."""
    from vlbipy.backends.dask_ms import DaskMsCalibrationOps
    from vlbipy.solvers import tec as tec_module
    written = {}

    def fake(vis, caltable, **kwargs):
        written["call"] = (vis, caltable)
        return {"product": "jplg", "stec_range": (1.0, 2.0)}

    monkeypatch.setattr(tec_module, "tec", fake)
    ops = object.__new__(DaskMsCalibrationOps)
    ops._backend = MagicMock(work_dir=tmp_path)
    table = ops._tec_table("TEC01", "SRC", tmp_path / "x.tec")
    assert written["call"] and table.cal_type == "tec" and table.interp == "linear"
    assert table.field == "SRC"
