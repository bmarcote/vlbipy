"""Tests for the a-priori amplitude calibration step (Tsys + GC gencal, EOP for VLBA/LBA)."""
from __future__ import annotations

import numpy as np
import pytest
from test_phaseref import CHAN_FREQ, T0, three_field_ms  # noqa: F401 - three_field_ms is a fixture

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
    assert antenna_elevation_range(meta, "XX", default=None) is None


def test_observed_elevation_ranges_leave_out_antennas_without_a_range():
    """Only antennas whose elevations can be worked out get a range: none is ever made up."""
    from vlbipy.backends.casa import antenna_elevation_range, observed_elevation_ranges
    from vlbipy.models import Antenna, ObsMetadata, Scan
    start = 59739.0 * 86400.0 + 14 * 3600.0
    scans = [Scan(scan_number=i + 1, source="CAL", time_start=start + i * 3600.0, time_end=start + i * 3600.0 + 120.0,
                  antennas=["JB", "NOPOS"]) for i in range(6)]
    # NOPOS is in the scans but its position is unknown; IDLE has a position but is in no scan
    antennas = {"JB": Antenna(name="JB", position=(3822626.04, -154105.65, 5086486.04)), "NOPOS": Antenna(name="NOPOS"),
                "IDLE": Antenna(name="IDLE", position=(3822626.04, -154105.65, 5086486.04))}
    meta = ObsMetadata(antennas=antennas, scans=scans, source_coords={"CAL": (216.75, 23.80)})
    ranges = observed_elevation_ranges(meta)
    assert list(ranges) == ["JB"] and ranges["JB"] == antenna_elevation_range(meta, "JB")
    assert observed_elevation_ranges(None) == {}


# -- gain curves: the DPFU record kept beside the table, and the plot made without a measurement set --

def _write_gain_curve_subtables(ms, rows, names):
    """Write the ``GAIN_CURVE`` and ``ANTENNA`` subtables of a pretend measurement set.

    ``rows`` are ``(antenna id, subband, (dpfu pol 0, dpfu pol 1))``; ``names`` the antenna names in ID order.
    """
    import casatools
    import numpy as np

    from vlbipy.solvers import caltable as ct
    tb = casatools.table()
    tb.create(str(ms / "ANTENNA"), {"NAME": ct._scalar_column("string")})
    tb.addrows(len(names))
    tb.putcol("NAME", np.array(names))
    tb.close()
    tb.create(str(ms / "GAIN_CURVE"), {"ANTENNA_ID": ct._scalar_column("int"), "TYPE": ct._scalar_column("string"),
                                       "SPECTRAL_WINDOW_ID": ct._scalar_column("int"),
                                       "SENSITIVITY": ct._array_column("float", ndim=1)})
    tb.addrows(len(rows))
    for row, (antenna, spw, sensitivity) in enumerate(rows):
        tb.putcell("ANTENNA_ID", row, antenna)
        tb.putcell("SPECTRAL_WINDOW_ID", row, spw)
        tb.putcell("TYPE", row, "POWER(EL)")
        tb.putcell("SENSITIVITY", row, np.asarray(sensitivity, dtype=np.float32))
    tb.close()


def _write_gain_curve_caltable(path, names, coefficients):
    """Write a minimal EPowerCurve caltable: one row per antenna with ``coefficients[name]`` (16 values each)."""
    import casatools
    import numpy as np

    from vlbipy.solvers import caltable as ct
    desc = ct.main_table_desc()
    tb = casatools.table()
    tb.create(str(path), desc, dminfo=ct._dminfo(list(desc)))
    tb.addrows(len(names))
    tb.putcol("TIME", np.full(len(names), 5.0e9))
    tb.putcol("SPECTRAL_WINDOW_ID", np.zeros(len(names), dtype=np.int32))
    tb.putcol("ANTENNA1", np.arange(len(names), dtype=np.int32))
    param = np.array([coefficients[name] for name in names], dtype=np.float32)          # (nrow, npar)
    tb.putcol("FPARAM", param.T[:, None, :])                                              # casatools: (npar, 1, nrow)
    tb.putcol("FLAG", np.zeros((param.shape[1], 1, len(names)), dtype=bool))
    tb.putkeyword("VisCal", "EPowerCurve")
    tb.close()
    tb.create(str(path / "ANTENNA"), {"NAME": ct._scalar_column("string")})
    tb.addrows(len(names))
    tb.putcol("NAME", np.array(names))
    tb.close()


def test_gain_curve_dpfu_record_is_written_beside_the_table_and_read_back(tmp_path):
    """gencal folds the DPFU into the gain-curve table; the record keeps it, per antenna and subband."""
    pytest.importorskip("casatools")
    import json

    from vlbipy.backends import casa
    backend = casa.CasaBackend(work_dir=str(tmp_path))
    ms = tmp_path / "x.ms"
    ms.mkdir()
    _write_gain_curve_subtables(ms, [(1, 0, (0.084134, 0.0854)), (1, 1, (0.084134, 0.0854)),
                                     (0, 0, (0.16, 0.16)), (0, 1, (0.16, 0.16))], ["MC", "WB"])
    table = backend.caldir() / "x.gcal"
    assert casa.read_gain_curve_dpfu(table) == {}                                  # no record yet: nothing known
    record = backend.calibrate._record_gain_curve_dpfu(ms, table)
    assert record == casa.gain_curve_dpfu_path(table) and record.name == "x.gcal.dpfu.json"
    # single-precision values come back as the numbers the station wrote, not as 0.08413399755954742
    assert json.loads(record.read_text())["antennas"]["WB"] == {
        "type": "POWER(EL)", "dpfu": {"0": [0.084134, 0.0854], "1": [0.084134, 0.0854]}}
    assert casa.read_gain_curve_dpfu(table) == {"WB": {0: (0.084134, 0.0854), 1: (0.084134, 0.0854)},
                                                "MC": {0: (0.16, 0.16), 1: (0.16, 0.16)}}
    # a record that cannot be made is not fatal, and the one of an earlier import does not outlive it
    assert backend.calibrate._record_gain_curve_dpfu(tmp_path / "missing.ms", table) is None
    assert not record.exists() and casa.read_gain_curve_dpfu(table) == {}
    record.write_text("not json")
    assert casa.read_gain_curve_dpfu(table) == {}                                  # unreadable: ignored


def test_gain_curve_plot_works_from_the_table_alone_and_uses_the_record(tmp_path, monkeypatch):
    """No measurement set, no metadata: still a plot. With the DPFU record the curves are divided by the DPFU."""
    pytest.importorskip("casatools")
    import json

    import matplotlib.pyplot as plt

    from vlbipy.backends import casa
    backend = casa.CasaBackend(work_dir=str(tmp_path))
    table = backend.caldir() / "x.gcal"
    # Medicina in EM163 (DPFU 0.16 x POLY, in zenith angle) and a constant curve with one DPFU per polarization
    mc = [1.559041440e-01, 3.503856715e-04, -7.493592420e-06] + [0.0] * 5
    wb = [0.084134] + [0.0] * 7 + [0.0854] + [0.0] * 7
    _write_gain_curve_caltable(table, ["MC", "WB"], {"MC": mc + mc, "WB": wb})
    titles = {}
    monkeypatch.setattr(plt, "close", lambda figure: titles.update(
        {axis.get_title(loc="left"): axis.get_title(loc="right") for axis in figure.axes}))
    paths = backend.plot.caltable("x", str(table), cal_type="gc")
    assert paths == [str(tmp_path / "plots" / "caltables" / "x.gcal.png")]
    assert (tmp_path / "plots" / "caltables" / "x.gcal.png").stat().st_size > 1000
    assert titles == {"MC": "peak R 0.16  L 0.16 K/Jy", "WB": "peak R 0.08413  L 0.0854 K/Jy"}
    casa.gain_curve_dpfu_path(table).write_text(json.dumps({"antennas": {"MC": {"dpfu": {"0": [0.16, 0.16]}}}}))
    backend.plot.caltable("x", str(table), cal_type="gc")
    assert titles == {"MC": "DPFU R 0.16  L 0.16 K/Jy", "WB": "peak R 0.08413  L 0.0854 K/Jy"}


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


def test_tsys_is_smoothed_per_source(three_field_ms, tmp_path):
    """Each source keeps its own Tsys level: the running median never blends a calibrator into the target, a
    spike is removed and the noise drops."""
    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable
    from vlbipy.solvers import caltable

    ms = three_field_ms
    rng = np.random.default_rng(1)
    times = T0 + np.arange(0.0, 1800.0, 10.0)
    fields = np.where((times - T0 >= 600) & (times - T0 < 1200), 1, 0)       # 0, then 1, then 0 again
    level = np.where(fields == 1, 150.0, 50.0)
    values = level[:, None] + rng.normal(0.0, 2.0, (times.size, 2))
    values[20] = 400.0                                                        # a spike on source 0
    path = tmp_path / "test.tsys"
    caltable.write_jones_table(path, ms, viscal="B TSYS", times=times, field_ids=fields,
                               spw_ids=np.zeros(times.size, int), antenna_ids=np.ones(times.size, int),
                               scan_numbers=np.ones(times.size, int), fparam=values[:, None, :],
                               flag=np.zeros((times.size, 1, 2), bool), snr=np.ones((times.size, 1, 2)),
                               spw_chan_freq=CHAN_FREQ[:1], spw_chan_width=np.array([16e6]))
    backend = CasaBackend(work_dir=str(tmp_path))
    backend.calibrate.smooth("TEST", CalTable(cal_type="tsys", path=str(path)), time_window=600.0)
    handle = backend.tools.table()
    handle.open(str(path))
    smoothed = np.asarray(handle.getcol("FPARAM"))[:, 0, :].T                 # (nrow, npol)
    handle.close()
    for field, expected in ((0, 50.0), (1, 150.0)):
        rows = fields == field
        assert np.all(np.abs(smoothed[rows] - expected) < 4.0)               # no blending, no spike left
        assert smoothed[rows].std() < values[rows & (np.arange(times.size) != 20)].std()
