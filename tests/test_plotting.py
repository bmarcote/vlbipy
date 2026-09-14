"""Tests for calibration-table plotting (pure helpers + namespace wiring on dummy)."""
from __future__ import annotations

import numpy as np
import pytest

from vlbipy import VLBIObs
from vlbipy.plotting import SUBBAND_COLORS, evaluate_gain_curve, subplot_grid


def test_subplot_grid_shapes():
    assert subplot_grid(1) == (1, 1)
    assert subplot_grid(4) == (1, 4)
    assert subplot_grid(5) == (2, 4)
    assert subplot_grid(12) == (3, 4)
    assert subplot_grid(14) == (4, 4)


def test_evaluate_gain_curve_polynomial():
    # g(el) = 0.5 + 0.01*el - 0.0001*el^2
    coeffs = [0.5, 0.01, -1e-4]
    el = np.array([0.0, 10.0, 50.0])
    expected = 0.5 + 0.01 * el - 1e-4 * el**2
    assert np.allclose(evaluate_gain_curve(coeffs, el), expected)
    # trailing zero-padded coefficients (from the zero-padded GAIN storage) are harmless
    assert np.allclose(evaluate_gain_curve([0.5, 0.01, -1e-4, 0.0, 0.0], el), expected)


def test_subband_colors_fixed_order():
    """Fixed-order categorical palette: first four are the validated Okabe-Ito set."""
    assert SUBBAND_COLORS[:4] == ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def test_plot_namespace_caltables_dummy():
    obs = VLBIObs("pl01", network="VLBA", backend="dummy", target="SRC")
    obs.import_data()
    obs.calibrate.a_priori()
    paths = obs.plot.caltables()
    assert len(paths) == 3  # tsys, gc, eop
    assert all(p.endswith(".png") for p in paths)


def test_plot_namespace_caltable_single_dummy():
    obs = VLBIObs("pl02", network="EVN", backend="dummy", target="SRC")
    obs.import_data()
    tables = obs.calibrate.a_priori()
    paths = obs.plot.caltable(tables[0])
    assert isinstance(paths, list) and len(paths) == 1


# -- scan x antenna SNR matrix --

def _demo_survey(n_scans=6, n_antennas=4):
    """Build a small survey with a masked refant column and one absent antenna."""
    from vlbipy.models import ScanSNRSurvey
    nan = float("nan")
    antennas = ["EF", "WB", "O8", "TR"][:n_antennas]
    matrices = {}
    for offset, pol in enumerate(("RR", "LL")):
        matrices[pol] = [[nan if a == 0 else float(10 ** (1 + a % 3) + 5 * s + offset)
                          for a in range(n_antennas)] for s in range(n_scans)]
    matrices["RR"][2][1] = nan  # a failed solution in the middle of the matrix
    return ScanSNRSurvey(project_code="rsm07", scan_numbers=list(range(1, n_scans + 1)),
                         scan_sources=["3C345"] * (n_scans // 2) + ["J1848+3219"] * (n_scans - n_scans // 2),
                         antennas=antennas, snr=matrices, channel_fraction=0.8, refant="EF")


def test_scan_snr_plotter_writes_one_png_per_polarization(tmp_path):
    from vlbipy.plotting import ScanSNRPlotter
    paths = ScanSNRPlotter(tmp_path).plot(_demo_survey())
    assert [p.name for p in paths] == ["rsm07.snr_matrix.RR.png", "rsm07.snr_matrix.LL.png"]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)


def test_scan_snr_plotter_handles_all_nan_and_empty(tmp_path):
    """An all-failed polarization still renders; an empty survey writes nothing."""
    from vlbipy.models import ScanSNRSurvey
    from vlbipy.plotting import ScanSNRPlotter
    nan = float("nan")
    blank = ScanSNRSurvey(project_code="p", scan_numbers=[1], scan_sources=["A"],
                          antennas=["EF", "WB"], snr={"RR": [[nan, nan]]})
    assert len(ScanSNRPlotter(tmp_path).plot(blank)) == 1
    assert ScanSNRPlotter(tmp_path).plot(ScanSNRSurvey(project_code="p")) == []


def test_plot_namespace_scan_snr_dummy():
    """obs.plot.scan_snr() computes the survey on demand and stores it on the metadata."""
    obs = VLBIObs("pl02", network="EVN", backend="dummy", target="SRC", phasecal="CAL")
    obs.import_data()
    paths = obs.plot.scan_snr()
    assert len(paths) == 2 and all(p.endswith(".png") for p in paths)
    assert obs["pl02"].metadata.snr_survey is not None


# -- plot directory layout --

def test_plot_categories_and_column_mapping():
    """Plots are filed by what they were made from; the data column decides raw vs calibrated."""
    from vlbipy.backends import get_backend
    plots = get_backend("dummy", work_dir="/tmp/x").plot
    assert plots.CATEGORIES == ("raw", "caltables", "calibrated")
    assert plots.plot_dir("caltables").name == "caltables"
    assert plots.plot_dir("raw").parent.name == "plots"
    assert plots.category_for_column("corrected") == "calibrated"
    assert plots.category_for_column("data") == "raw"
    with pytest.raises(ValueError, match="unknown plot category"):
        plots.plot_dir("nonsense")


def test_dummy_plots_are_filed_by_category():
    """The dummy backend reports the same layout a real backend writes."""
    obs = VLBIObs("pl03", network="EVN", backend="dummy", target="SRC", phasecal="CAL")
    obs.import_data()
    obs.calibrate.a_priori()
    assert all("/plots/caltables/" in p for p in obs.plot.caltables())
    assert all("/plots/raw/" in p for p in obs.plot.scan_snr())
    assert all("/plots/calibrated/" in p for p in obs.plot.spectrum(column="corrected"))
    assert all("/plots/raw/" in p for p in obs.plot.spectrum(column="data"))


def test_plot_uv_coverage_writes_png(tmp_path):
    """Two fields -> one PNG with a panel each (pure plotting, no CASA)."""
    from vlbipy.plotting import plot_uv_coverage
    rng = np.random.default_rng(1)
    hour_angle = np.linspace(-2.0, 2.0, 200)
    fields = {}
    for name, scale in (("3C345", 120.0), ("J1848+3219", 60.0)):
        u = scale * np.cos(hour_angle) + rng.normal(0, 2, hour_angle.size)
        v = scale * 0.4 * np.sin(hour_angle) + rng.normal(0, 2, hour_angle.size)
        fields[name] = {"u": u.tolist(), "v": v.tolist()}
    data = {"fields": fields, "unit": "Mlambda", "freq_ghz": 4.99}
    out = plot_uv_coverage(data, str(tmp_path / "uvc.uv_coverage.png"), title="uvc — uv coverage")
    assert out.endswith("uvc.uv_coverage.png")
    assert (tmp_path / "uvc.uv_coverage.png").stat().st_size > 1000
    # nothing to plot -> no file, but the path is still returned
    empty = plot_uv_coverage({"fields": {}}, str(tmp_path / "none.png"))
    assert empty.endswith("none.png") and not (tmp_path / "none.png").exists()
