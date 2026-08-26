"""Tests for the a-priori amplitude calibration step (Tsys + GC gencal, EOP for VLBA/LBA)."""
from __future__ import annotations

import pytest

from vlbipy import VLBIObs, load_config, tools
from vlbipy.observation import Observation
from vlbipy.observatories import get_observatory_handler


def _apriori_types(network: str) -> list[str]:
    obs = VLBIObs("ts01", network=network, backend="dummy", target="SRC")
    obs.import_data()
    tables = obs.calibrate.a_priori()
    return [t.cal_type for t in tables]


def test_handlers_eop_flags():
    assert get_observatory_handler("EVN").needs_eop is False
    assert get_observatory_handler("VLBA").needs_eop is True
    assert get_observatory_handler("LBA").needs_eop is True


def test_apriori_evn_has_no_eop():
    assert _apriori_types("EVN") == ["tsys", "gc"]


def test_apriori_vlba_and_lba_include_eop():
    assert _apriori_types("VLBA") == ["tsys", "gc", "eop"]
    assert _apriori_types("LBA") == ["tsys", "gc", "eop"]


def test_apriori_tables_registered_in_gaintables():
    obs = VLBIObs("ts02", network="VLBA", backend="dummy", target="SRC")
    obs.import_data()
    obs.calibrate.a_priori()
    observation = obs.observations[0]
    assert [t.cal_type for t in observation.gaintables] == ["tsys", "gc", "eop"]
    assert observation.state.status("a_priori") == "done"


def test_fetch_eop_file_reuses_existing(tmp_path):
    existing = tmp_path / "usno_finals.erp"
    existing.write_text("EOP data")
    assert tools.fetch_eop_file(tmp_path) == existing


def test_fetch_eop_file_downloads(tmp_path, monkeypatch):
    def fake_download(url, dest, *args, **kwargs):
        from pathlib import Path
        Path(dest).write_text("downloaded EOPs")
        return Path(dest)

    monkeypatch.setattr(tools, "download_file", fake_download)
    path = tools.fetch_eop_file(tmp_path)
    assert path.read_text() == "downloaded EOPs"


def test_casa_apriori_requires_ms(tmp_path):
    pytest.importorskip("casatools")
    from vlbipy.backends.casa import CasaBackend
    from vlbipy.errors import BackendError
    backend = CasaBackend(work_dir=str(tmp_path))
    with pytest.raises(BackendError, match="import_data"):
        backend.calibrate.a_priori("nodata", "SRC")


def test_daskms_apriori_missing_tables_and_ms(tmp_path):
    pytest.importorskip("daskms")
    pytest.importorskip("casatools")
    from vlbipy.backends.dask_ms import DaskMsBackend
    from vlbipy.errors import BackendError
    backend = DaskMsBackend(work_dir=str(tmp_path))
    with pytest.raises(BackendError, match="re-run import_data"):
        backend.calibrate.a_priori("nodata", "SRC")


def _obs_at(freq_ghz, **overrides):
    """A dummy-backend observation whose metadata reports a given observing frequency."""
    vobs = VLBIObs("ts_ion", network="EVN", backend="dummy", target="SRC", **overrides)
    vobs.import_data()
    obs = vobs.observations[0]
    obs.metadata.freq_setup.ref_freq = freq_ghz * 1e9
    return obs


def test_dispersive_delay_default_depends_on_frequency():
    """Below 6 GHz the ionosphere matters, so the fringe fit solves the dispersive delay."""
    cfg = lambda o: o.config.get("calibration", {})
    low = _obs_at(1.658)
    assert low.calibrate._solve_dispersive(cfg(low)) is True
    high = _obs_at(22.0)
    assert high.calibrate._solve_dispersive(cfg(high)) is False


def test_dispersive_delay_disabled_by_config():
    """[calibration].ionos = false wins over the frequency test."""
    obs = _obs_at(1.658, config={"calibration": {"ionos": False}})
    assert obs.calibrate._solve_dispersive(obs.config.get("calibration", {})) is False


def test_dispersive_delay_threshold_is_configurable():
    obs = _obs_at(8.4, config={"calibration": {"ionos_max_ghz": 10.0}})
    assert obs.calibrate._solve_dispersive(obs.config.get("calibration", {})) is True
