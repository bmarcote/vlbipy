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


def test_fetch_eop_file_downloads_from_cddis(tmp_path, monkeypatch):
    """The standard source comes first: curl -u anonymous:<e-mail> --ftp-ssl <CDDIS url>."""
    import subprocess
    from pathlib import Path
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_text("EOPs from CDDIS")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    monkeypatch.setattr(tools, "download_file", lambda *a, **k: pytest.fail("the mirror must not be needed"))
    assert tools.fetch_eop_file(tmp_path).read_text() == "EOPs from CDDIS"
    command = calls[0]
    assert command[0] == "curl" and "--ftp-ssl" in command and tools.EOP_CDDIS_URL in command
    assert command[command.index("-u") + 1] == f"anonymous:{tools.EOP_CDDIS_EMAIL}"


def test_fetch_eop_file_falls_back_to_the_mirror(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    def fake_download(url, dest, *args, **kwargs):
        Path(dest).write_text("downloaded EOPs")
        return Path(dest)

    monkeypatch.setattr(tools.subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 7, "", "no route"))
    monkeypatch.setattr(tools, "download_file", fake_download)
    assert tools.fetch_eop_file(tmp_path).read_text() == "downloaded EOPs"


def test_accor_and_eop_are_applied_with_nearest_interpolation():
    pytest.importorskip("casatools")
    from vlbipy.backends.casa import APRIORI_TABLE_SPECS
    assert APRIORI_TABLE_SPECS["accor"] == (".accor", "nearest")
    assert APRIORI_TABLE_SPECS["eop"] == (".eop", "nearest")


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


# -- gain curves longer than CASA can hold --

JB_GAIN_CURVE = [-0.396588307618, 0.319842329619, -0.0305994179281, 0.00155527421339, -4.6141588862e-05,
                 8.26673512293e-07, -8.82266489915e-09, 5.17279119815e-11, -1.28519741343e-13]


def test_refit_polynomial_reproduces_a_nine_term_gain_curve_with_at_most_eight():
    import numpy as np
    from vlbipy.backends.casa import refit_polynomial
    coefficients, deviation = refit_polynomial(JB_GAIN_CURVE, 15.0, 63.0)
    assert len(coefficients) <= 8 and deviation <= 1e-3
    elevation = np.linspace(15.0, 63.0, 50)
    original = np.polynomial.polynomial.polyval(elevation, JB_GAIN_CURVE)
    assert np.allclose(np.polynomial.polynomial.polyval(elevation, coefficients), original, rtol=1e-3)
    # what CASA did instead: the first eight terms only - wrong by a factor ~20 in power at 60 deg
    truncated = np.polynomial.polynomial.polyval(60.0, JB_GAIN_CURVE[:8])
    assert truncated / np.polynomial.polynomial.polyval(60.0, JB_GAIN_CURVE) > 20.0


def test_refit_polynomial_keeps_a_short_polynomial_exactly():
    import numpy as np
    from vlbipy.backends.casa import refit_polynomial
    coefficients, deviation = refit_polynomial([1.0434, -1.9066e-3, 2.7559e-5, -2.1536e-7], 10.0, 80.0, tolerance=1e-9)
    assert len(coefficients) == 4 and deviation < 1e-9
    assert np.allclose(coefficients, [1.0434, -1.9066e-3, 2.7559e-5, -2.1536e-7], rtol=1e-6)


def test_antenna_elevation_range_follows_the_scans():
    from vlbipy.backends.casa import antenna_elevation_range
    from vlbipy.models import Antenna, ObsMetadata, Scan
    # Jodrell Bank, a source at +24 deg crossing the meridian during the scans
    jb = Antenna(name="JB", position=(3822626.04, -154105.65, 5086486.04))
    start = 59739.0 * 86400.0 + 14 * 3600.0
    scans = [Scan(scan_number=i + 1, source="CAL", time_start=start + i * 3600.0, time_end=start + i * 3600.0 + 120.0,
                  antennas=["JB"]) for i in range(12)]
    meta = ObsMetadata(antennas={"JB": jb}, scans=scans, source_coords={"CAL": (216.75, 23.80)})
    low, high = antenna_elevation_range(meta, "JB", pad=0.0)
    assert 1.0 <= low < 30.0 and 58.0 < high < 62.0            # culminates at 90 - 53.2 + 23.8 = 60.6 deg
    assert antenna_elevation_range(meta, "XX") == (8.0, 88.0)   # unknown antenna: the default range


def test_second_pass_keeps_every_apriori_table(monkeypatch):
    """ACCOR is made by the a-priori step like Tsys and EOP: a re-solve must not drop it from the chain."""
    from vlbipy import VLBIObs
    from vlbipy.models import CalTable
    from vlbipy.namespaces import CalibrateNamespace
    obs = VLBIObs("RSM07", network="VLBA", backend="dummy", target="3C286", phasecal="J1048+7143",
                  fringe_finder="3C345")
    obs.import_data()
    obs.calibrate.a_priori()
    observation = obs.observations[0]
    observation.gaintables.insert(0, CalTable(cal_type="accor", path="x.accor_smooth", step="a_priori"))
    observation.add_gaintable(CalTable(cal_type="sbd", path="x.sbd"), "initial_calibration")
    # Only the bookkeeping is under test: what the re-solve starts from.
    monkeypatch.setattr(CalibrateNamespace, "instrumental", lambda self, **kwargs: [])
    obs.calibrate.second_pass(force=True, plot=False)
    assert [t.cal_type for t in observation.gaintables] == ["accor", "tsys", "gc", "eop"]
