"""Tests for calibration-table plotting (pure helpers + namespace wiring on dummy)."""
from __future__ import annotations

from pathlib import Path

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


# -- stacked amplitude/phase pairs --

def _demo_uvdata(with_model=False):
    rng = np.random.default_rng(3)
    uvdist = rng.uniform(5.0, 200.0, 300)
    values = np.exp(-uvdist[:, None] / 300.0) * (1 + 0.05 * rng.normal(size=(300, 2))) \
        * np.exp(1j * rng.normal(0, 0.2, (300, 2)))
    data = {"uvdist_mlambda": uvdist, "values": values, "polarizations": ["RR", "LL"],
            "field": "3C345", "column": "corrected", "time_bin": 30.0}
    if with_model:
        grid = np.linspace(0.0, 200.0, 50)
        data["model"] = {"uvdist_mlambda": grid[::-1],
                         "values": np.exp(-grid[::-1] / 300.0)[:, None] * np.ones((1, 2), dtype=complex)}
    return data


def test_plot_radplot_with_model_line(tmp_path):
    from vlbipy.plotting import plot_radplot
    out = plot_radplot(_demo_uvdata(with_model=True), tmp_path, "rp01", label="cal")
    assert out.name == "rp01.cal.radplot.3C345.png" and out.stat().st_size > 1000
    assert plot_radplot(_demo_uvdata(), tmp_path, "rp01").name == "rp01.radplot.3C345.png"


def test_stacked_pair_amplitude_on_top_without_gap():
    import matplotlib.pyplot as plt
    from vlbipy.plotting import _stacked_pair, _style_stacked_pair
    figure = plt.figure()
    amp_axis, phase_axis = _stacked_pair(figure, figure.add_gridspec(1, 1)[0])
    _style_stacked_pair(amp_axis, phase_axis)
    amp_box, phase_box = amp_axis.get_position(), phase_axis.get_position()
    assert amp_box.y0 == pytest.approx(phase_box.y1)   # no vertical gap
    assert amp_box.height == pytest.approx(2 * phase_box.height, rel=1e-6)
    assert phase_axis.get_shared_x_axes().joined(amp_axis, phase_axis)
    assert not any(t.get_visible() for t in amp_axis.get_xticklabels()) or \
        not amp_axis.xaxis.get_tick_params()["labelbottom"]
    plt.close(figure)


def test_plot_corrected_spectrum_and_timeseries(tmp_path):
    from vlbipy.plotting import plot_baseline_timeseries, plot_corrected_spectrum
    rng = np.random.default_rng(0)
    spectrum = {"antennas": ["WB", "O8"], "n_spw": 2, "n_channels": 16, "polarizations": ["RR", "LL"],
                "refant": "EF", "frequencies_ghz": [np.linspace(4.9, 4.92, 16), np.linspace(4.93, 4.95, 16)],
                "spectra": {a: rng.normal(1, 0.1, (2, 16, 2)) + 0.1j for a in ("WB", "O8")},
                "scans": [3], "field": "CAL", "column": "corrected"}
    (out,) = plot_corrected_spectrum(spectrum, tmp_path, "sp01", label="x")
    assert out.name == "sp01.x.spectrum.png" and out.stat().st_size > 1000
    series = {"baselines": {("EF", "WB"): rng.normal(1, 0.1, (20, 2)) + 0.1j},
              "times": (np.arange(20) * 60.0 + 5e9).tolist(), "polarizations": ["RR", "LL"], "field": "CAL",
              "column": "data"}
    (out,) = plot_baseline_timeseries(series, tmp_path, "sp01")
    assert out.name == "sp01.timeseries.png" and out.stat().st_size > 1000


# -- new diagnostic plots --

def test_plot_autocorr_spectrum_writes_png(tmp_path):
    from vlbipy.plotting import plot_autocorr_spectrum
    rng = np.random.default_rng(5)
    spectra = {a: rng.uniform(0.8, 1.2, (2, 32, 2)) for a in ("EF", "WB", "O8", "TR", "MC")}
    spectra["MC"][:] = 0.0                       # fully flagged antenna
    spectra["TR"][1] = np.nan                    # a dead subband
    spectrum = {"antennas": list(spectra), "spectra": spectra, "n_spw": 2, "n_channels": 32,
                "polarizations": ["RR", "LL"], "scans": [2, 3], "field": "3C345", "column": "data",
                "frequencies_ghz": [np.linspace(4.9, 4.93, 32), np.linspace(4.93, 4.96, 32)]}
    out = plot_autocorr_spectrum(spectrum, tmp_path, "ac01", label="raw")
    assert out.endswith("ac01.raw.autocorr.png") and (tmp_path / "ac01.raw.autocorr.png").stat().st_size > 1000
    spectrum["frequencies_ghz"] = None
    out = plot_autocorr_spectrum(spectrum, tmp_path, "ac01")
    assert (tmp_path / "ac01.autocorr.png").stat().st_size > 1000


def test_plot_autocorr_one_linestyle_polarization_colors(tmp_path, monkeypatch):
    """All subbands share a solid line; only polarization sets the colour."""
    from matplotlib.axes import Axes
    from vlbipy.plotting import plot_autocorr_spectrum, polarization_color
    calls = []
    original = Axes.plot
    def spy(axis, *args, **kwargs):
        calls.append(kwargs)
        return original(axis, *args, **kwargs)
    monkeypatch.setattr(Axes, "plot", spy)
    rng = np.random.default_rng(7)
    spectrum = {"antennas": ["EF", "WB"], "n_spw": 3, "n_channels": 16,
                "polarizations": ["RR", "LL"], "scans": [2], "field": "3C345", "column": "data",
                "spectra": {a: rng.uniform(0.8, 1.2, (3, 16, 2)) for a in ("EF", "WB")},
                "frequencies_ghz": [np.linspace(1.63 + 0.01 * s, 1.64 + 0.01 * s, 16) for s in range(3)]}
    plot_autocorr_spectrum(spectrum, tmp_path, "ac02")
    assert calls and all(c.get("ls", "-") == "-" for c in calls)
    assert {c["color"] for c in calls} == {polarization_color("RR", 0), polarization_color("LL", 1)}


def test_bin_lightcurve_scales():
    from vlbipy.plotting import _bin_lightcurve
    times = np.arange(60) * 2.0                          # 2 s integrations, 2 scans of 60 s
    scans = np.repeat([1, 2], 30)
    vis_sum = np.full(60, 10.0 + 0j)
    n_vis = np.full(60, 5.0)
    for width, expected in ((0, 60), (30, 4), (120, 2), (-1, 2)):
        bin_times, amp = _bin_lightcurve(times, vis_sum, n_vis, scans, width)
        assert bin_times.size == expected and np.allclose(amp, 2.0)


def test_plot_total_lightcurve_writes_png(tmp_path):
    from vlbipy.plotting import plot_total_lightcurve
    rng = np.random.default_rng(7)
    sources = {}
    for name, flux in (("TARGET", 0.02), ("CAL", 1.5)):
        times = np.concatenate([5e9 + 300 * s + np.arange(0, 120, 2.0) for s in range(3)])
        scans = np.repeat([1, 2, 3], 60)
        n_vis = np.full(times.size, 400.0)
        vis_sum = n_vis * flux * np.exp(1j * rng.normal(0, 0.1, times.size)) + rng.normal(0, 3, times.size)
        sources[name] = {"times": times, "vis_sum": vis_sum, "n_vis": n_vis, "scans": scans}
    data = {"sources": sources, "column": "corrected", "time_start": 5e9}
    out = plot_total_lightcurve(data, tmp_path, "lc01", label="cal")
    assert out.endswith("lc01.cal.lightcurve.png") and (tmp_path / "lc01.cal.lightcurve.png").stat().st_size > 1000
    assert plot_total_lightcurve({"sources": {}}, tmp_path, "lc02").endswith("lc02.lightcurve.png")


def _write_fits(path, shape=(1, 1, 64, 64), peak=0.05):
    from astropy.io import fits
    rng = np.random.default_rng(11)
    data = rng.normal(0, 1e-4, shape)
    data[..., shape[-2] // 2, shape[-1] // 2] = peak
    header = fits.Header({"BUNIT": "Jy/beam", "CDELT1": -1.0 / 3.6e6, "CDELT2": 1.0 / 3.6e6})
    fits.PrimaryHDU(data=data, header=header).writeto(str(path), overwrite=True)
    return str(path)


def test_plot_image_grid_writes_one_png_per_source(tmp_path):
    from vlbipy.plotting import plot_image_grid
    images = {"TARGET": {0.0: _write_fits(tmp_path / "t_r0.fits"), -1.0: _write_fits(tmp_path / "t_rm1.fits"),
                         2.0: _write_fits(tmp_path / "t_r2.fits", shape=(64, 64))},
              "CAL": {0.5: _write_fits(tmp_path / "c.fits", peak=2.0), 1.0: str(tmp_path / "missing.fits")}}
    paths = plot_image_grid(images, tmp_path / "plots", "im01")
    assert [Path(p).name for p in paths] == ["im01.images.TARGET.png", "im01.images.CAL.png"]
    assert all(Path(p).stat().st_size > 1000 for p in paths)
    assert plot_image_grid({"X": {0.0: str(tmp_path / "missing.fits")}}, tmp_path, "im01") == []


def test_plot_bandpass_profile_per_antenna(tmp_path):
    from vlbipy.plotting import plot_bandpass_profile
    profile = np.r_[np.linspace(0.2, 1, 4), np.ones(24), np.linspace(1, 0.2, 4)]
    measurement = {"n_channels": 32, "n_edge": 4, "amplitude_profile": profile,
                   "phase_profile": np.zeros(32), "flagged_fraction": np.zeros(32)}
    out = plot_bandpass_profile(measurement, tmp_path, "bp01")
    assert out.name == "bp01.bandpass_profile.png" and out.stat().st_size > 1000
    measurement["antennas"] = {a: {"amplitude_profile": profile * s, "phase_profile": np.zeros(32),
                                   "n_edge": (3, 5) if a == "WB" else 4}
                               for a, s in (("EF", 1.0), ("WB", 0.9), ("O8", 1.1))}
    out = plot_bandpass_profile(measurement, tmp_path / "per_antenna", "bp01")
    assert out.name == "bp01.bandpass_profile.png" and out.stat().st_size > 1000


# -- caltable gains without casatools --

def _gains_data(n_chan, n_ant=3, n_spw=2, n_pol=2):
    rng = np.random.default_rng(2)
    n_rows = n_ant * n_spw * 4
    param = rng.normal(1, 0.05, (n_rows, n_chan, n_pol)) * np.exp(1j * rng.normal(0, 0.3, (n_rows, n_chan, n_pol)))
    return {"viscal": "B Jones", "time": 5e9 + np.repeat(np.arange(4) * 600.0, n_ant * n_spw),
            "antenna": np.tile(np.repeat(np.arange(n_ant), n_spw), 4),
            "spw": np.tile(np.arange(n_spw), n_ant * 4), "flag": np.zeros(param.shape, dtype=bool),
            "param": param, "antenna_names": ["EF", "WB", "O8"]}


def test_caltable_bandpass_is_one_combined_png(tmp_path):
    from vlbipy.plotting import CalTablePlotter
    plotter = CalTablePlotter(tmp_path, frequencies={0: np.linspace(4.9, 4.93, 32), 1: np.linspace(4.93, 4.96, 32)})
    paths = plotter._plot_gains(_gains_data(32), tmp_path / "x.bpass.png", "bpass")
    assert [p.name for p in paths] == ["x.bpass.png"] and paths[0].stat().st_size > 1000
    assert not (tmp_path / "x.bpass.amp.png").exists() and not (tmp_path / "x.bpass.phase.png").exists()


def test_caltable_scalar_bandpass_is_amplitude_only(tmp_path):
    from vlbipy.plotting import CalTablePlotter
    paths = CalTablePlotter(tmp_path)._plot_gains(_gains_data(1), tmp_path / "x.scalar_bp.png", "scalar_bp")
    assert [p.name for p in paths] == ["x.scalar_bp.amp.png"] and paths[0].stat().st_size > 1000
