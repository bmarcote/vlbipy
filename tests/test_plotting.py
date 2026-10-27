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


#: Medicina in EM163. The station's ANTAB entry is ``GAIN MC ELEV DPFU = 0.16, 0.16 POLY = ...`` (a
#: polynomial in elevation); MC_CALTABLE is what ``gencal(caltype='gc')`` wrote for it in the EPowerCurve
#: table: DPFU x POLY re-expanded in zenith angle, zero-padded to eight coefficients.
MC_DPFU = 0.16
MC_ANTAB_POLY = [0.79212981, 0.0062403816, -4.6834953e-05]
MC_CALTABLE = [1.559041440e-01, 3.503856715e-04, -7.493592420e-06, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_evaluate_gain_curve_says_what_the_polynomial_is_a_function_of():
    """The second argument is always the elevation; ``variable`` names the polynomial's own variable."""
    el = np.array([30.0, 50.0, 70.0])
    antab = MC_DPFU * evaluate_gain_curve(MC_ANTAB_POLY, el, variable="el")
    assert np.allclose(antab, MC_DPFU * (0.79212981 + 0.0062403816 * el - 4.6834953e-05 * el**2))
    # the caltable holds the same curve as a polynomial in zenith angle = 90 - elevation
    assert np.allclose(evaluate_gain_curve(MC_CALTABLE, el, variable="za"), antab, rtol=1e-6)
    # read as a polynomial in elevation it is mirrored: 6% high at 30 deg and 10% low at 70 deg
    mirrored = evaluate_gain_curve(MC_CALTABLE, el, variable="el") / antab
    assert mirrored[0] == pytest.approx(1.065, abs=2e-3) and mirrored[2] == pytest.approx(0.899, abs=2e-3)
    # trailing zero-padded coefficients (the table always stores eight per polarization) are harmless
    assert np.allclose(evaluate_gain_curve(MC_CALTABLE[:3], el, variable="za"), antab, rtol=1e-6)
    with pytest.raises(ValueError, match="'za' or 'el'"):
        evaluate_gain_curve(MC_CALTABLE, el, variable="elevation")
    with pytest.raises(TypeError):
        evaluate_gain_curve(MC_CALTABLE, el)        # the variable has to be named: there is no safe default


def test_normalise_gain_curve_prefers_the_dpfu_and_falls_back_to_the_observed_peak():
    from vlbipy.plotting import normalise_gain_curve
    el = np.linspace(0.0, 90.0, 181)
    power = evaluate_gain_curve(MC_CALTABLE, el, variable="za")
    curve, reference, kind = normalise_gain_curve(power, np.ones(el.size, dtype=bool), MC_DPFU)
    assert (reference, kind) == (MC_DPFU, "DPFU")
    # divided by the DPFU the table gives back the station's own POLY(el): 1.0 at 66.6 deg
    assert np.allclose(curve, evaluate_gain_curve(MC_ANTAB_POLY, el, variable="el"), rtol=1e-6)
    assert curve.max() == pytest.approx(1.0, abs=1e-4) and el[np.argmax(curve)] == pytest.approx(66.5, abs=0.5)
    # no DPFU on record: the peak over the observed elevations (here the antenna never got above 40 deg)
    low = el <= 40.0
    curve, reference, kind = normalise_gain_curve(power, low)
    assert kind == "peak" and reference == pytest.approx(power[low].max())
    assert curve[low].max() == pytest.approx(1.0) and curve.max() > 1.0
    # nothing positive to scale by: the curve is left as stored
    assert normalise_gain_curve(np.zeros(5), np.ones(5, dtype=bool))[1:] == (1.0, "unit")


def test_gain_curve_ylim_keeps_unity_in_view_and_never_collapses():
    from vlbipy.plotting import gain_curve_ylim
    assert gain_curve_ylim(np.ones(10)) == pytest.approx((0.975, 1.025))
    assert gain_curve_ylim(np.array([])) == pytest.approx((0.975, 1.025))
    low, high = gain_curve_ylim(np.array([0.8, 0.9, np.nan]))
    assert low == pytest.approx(0.775) and high == pytest.approx(1.025)


def _za_coefficients(elevation_poly, dpfu, n_coeff=8):
    """What gencal stores for an elevation polynomial: DPFU x POLY re-expanded in zenith angle, zero-padded."""
    from numpy.polynomial import Polynomial
    za = Polynomial(elevation_poly)(Polynomial([90.0, -1.0])).coef * dpfu
    return np.r_[za, np.zeros(n_coeff - za.size)]


def _gain_curve_data(curves, n_spw=2):
    """In-memory EPowerCurve table. ``curves``: {antenna: (first pol, second pol)} coefficients in zenith angle."""
    names = list(curves)
    param = np.array([np.r_[curves[name][0], curves[name][1]] for name in names for _ in range(n_spw)])[:, None, :]
    return {"viscal": "EPowerCurve", "time": np.full(param.shape[0], 5e9),
            "antenna": np.repeat(np.arange(len(names)), n_spw), "spw": np.tile(np.arange(n_spw), len(names)),
            "flag": np.zeros(param.shape, dtype=bool), "param": param, "antenna_names": names}


def _capture_panels(monkeypatch):
    """Record, per antenna panel, what the next gain-curve figure shows (read when the figure is closed)."""
    import matplotlib.pyplot as plt
    panels = {}
    monkeypatch.setattr(plt, "close", lambda figure: panels.update({
        axis.get_title(loc="left"): {"right": axis.get_title(loc="right"), "ylabel": axis.get_ylabel(),
                                     "ylim": axis.get_ylim(), "xlabel": figure.get_supxlabel(),
                                     "curve": axis.get_lines()[0].get_xydata()}
        for axis in figure.axes if axis.get_visible()}))
    return panels


def test_gain_curve_plot_is_normalised_and_drawn_against_elevation(tmp_path, monkeypatch):
    """Curves sit near 1 with their peak at the right elevation; a refit running away does not set the limits."""
    from vlbipy.plotting import CalTablePlotter
    mc = _za_coefficients(MC_ANTAB_POLY, MC_DPFU)
    assert np.allclose(mc, MC_CALTABLE, rtol=1e-6)                      # the helper reproduces the real table
    # like a refitted polynomial: fine where the antenna observed (below 65 deg), 50% too high at the zenith
    wild = _za_coefficients([1.0, 0, 0, 0, 0, 0, 0, 1.02e-14], 0.12)
    data = _gain_curve_data({"MC": (mc, mc), "WB": (_za_coefficients([1.0], 0.084134), _za_coefficients([1.0], 0.0854)),
                             "JB": (wild, wild)})
    panels = _capture_panels(monkeypatch)
    dpfu = {"MC": {0: (0.16, 0.16), 1: (0.16, 0.16)}, "WB": {0: (0.084134, 0.0854), 1: (0.084134, 0.0854)},
            "JB": {0: (0.12, 0.12), 1: (0.12, 0.12)}}
    out = CalTablePlotter(tmp_path)._plot_gain_curve(data, tmp_path / "x.gcal.png", "gc", dpfu=dpfu,
                                                     elevation_ranges={"MC": (9.4, 76.9), "JB": (4.8, 64.3)})
    assert out.stat().st_size > 1000
    assert panels["MC"]["xlabel"] == "elevation (deg)" and panels["MC"]["ylabel"] == "gain / DPFU"
    assert panels["MC"]["right"] == "DPFU R 0.16  L 0.16 K/Jy"
    elevation, gain = panels["MC"]["curve"].T
    assert gain.max() == pytest.approx(1.0, abs=1e-4) and elevation[np.argmax(gain)] == pytest.approx(66.5, abs=0.5)
    assert gain[elevation == 30.0][0] == pytest.approx(0.79212981 + 0.0062403816 * 30 - 4.6834953e-05 * 900, rel=1e-6)
    # each polarization is divided by its own DPFU: a constant curve is exactly 1 in both
    assert panels["WB"]["right"] == "DPFU R 0.08413  L 0.0854 K/Jy"
    assert np.allclose(panels["WB"]["curve"][:, 1], 1.0) and panels["WB"]["ylim"] == pytest.approx((0.975, 1.025))
    # the runaway above the observed range is drawn but does not flatten the panel
    assert panels["JB"]["curve"][:, 1].max() > 1.4 and panels["JB"]["ylim"][1] < 1.1


def test_gain_curve_plot_without_dpfu_or_ranges_divides_by_the_peak(tmp_path, monkeypatch):
    """A table with no DPFU record is still plotted, labelled as divided by its peak (over the observed range)."""
    from vlbipy.plotting import CalTablePlotter
    mc = _za_coefficients(MC_ANTAB_POLY, MC_DPFU)
    wb = (_za_coefficients([1.0], 0.084134), _za_coefficients([1.0], 0.0854))
    data = _gain_curve_data({"MC": (mc, mc), "WB": wb})
    panels = _capture_panels(monkeypatch)
    plotter = CalTablePlotter(tmp_path)
    plotter._plot_gain_curve(data, tmp_path / "a.gcal.png", "gc", elevation_ranges={"MC": (9.4, 40.0)})
    # MC never got above 40 deg: its highest gain there is 0.16 x POLY(40 deg) = 0.1547 K/Jy
    assert panels["MC"]["right"] == "peak R 0.1547  L 0.1547 K/Jy" and panels["MC"]["ylabel"] == "gain / peak"
    elevation, gain = panels["MC"]["curve"].T
    assert gain[elevation <= 40.0].max() == pytest.approx(1.0) and gain.max() == pytest.approx(1.0 / 0.96681, rel=1e-4)
    panels.clear()
    out = plotter._plot_gain_curve(data, tmp_path / "b.gcal.png", "gc")
    assert out.stat().st_size > 1000
    assert panels["MC"]["right"] == "peak R 0.16  L 0.16 K/Jy"
    assert panels["WB"]["right"] == "peak R 0.08413  L 0.0854 K/Jy"


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


def test_plot_caltables_from_disk_skips_what_is_not_a_calibration_table(tmp_path):
    """With no chain in memory the tables on disk are plotted - not the other things kept in ``caltables/``."""
    obs = VLBIObs("pl03", network="EVN", backend="dummy", target="SRC", work_dir=str(tmp_path))
    caldir = tmp_path / "caltables"
    for name, kind in (("pl03.gcal", "Calibration"), ("pl03.tsys", "Calibration"), ("pl03.gain_curve.original", "")):
        (caldir / name).mkdir(parents=True)
        (caldir / name / "table.info").write_text(f"Type = {kind}\nSubType = \n")
    (caldir / "pl03.gcal.dpfu.json").write_text("{}")        # the DPFU record kept beside the gain-curve table
    (caldir / "empty_directory").mkdir()
    assert sorted(Path(p).name for p in obs.plot.caltables()) == ["pl03.gcal.png", "pl03.tsys.png"]


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


def test_scan_snr_plotter_writes_one_png_per_source_and_polarization(tmp_path):
    from vlbipy.dashboard import snr_matrix_key
    from vlbipy.plotting import ScanSNRPlotter
    (tmp_path / "rsm07.snr_matrix.RR.png").write_bytes(b"old")        # the all-source figure of an older run
    paths = ScanSNRPlotter(tmp_path).plot(_demo_survey())
    assert [p.name for p in paths] == ["rsm07.snr_matrix.3C345.RR.png", "rsm07.snr_matrix.J1848+3219.RR.png",
                                       "rsm07.snr_matrix.3C345.LL.png", "rsm07.snr_matrix.J1848+3219.LL.png"]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)
    assert not (tmp_path / "rsm07.snr_matrix.RR.png").exists()
    assert [snr_matrix_key(p) for p in paths[:2]] == [("3C345", "RR"), ("J1848+3219", "RR")]
    assert snr_matrix_key(Path("rsm07.snr_matrix.LL.png")) == ("", "LL")


def test_scan_snr_rows_of_one_source_are_contiguous(tmp_path, monkeypatch):
    """A source observed in alternating blocks gets only its own scans, one after the other."""
    from matplotlib.figure import Figure

    from vlbipy.plotting import ScanSNRPlotter
    survey = _demo_survey(8)
    survey.scan_sources = ["A", "B", "A", "B", "B", "A", "C", "A"]
    shown = {}

    def inspect(figure, outfile, *args, **kwargs):
        image = figure.axes[0].get_images()[0]
        labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        shown[Path(outfile).name] = (image.get_array().shape, labels)

    monkeypatch.setattr(Figure, "savefig", inspect)
    ScanSNRPlotter(tmp_path).plot(survey)
    assert shown["rsm07.snr_matrix.A.RR.png"] == ((4, 4), ["1", "3", "6", "8"])
    assert shown["rsm07.snr_matrix.C.LL.png"][0] == (1, 4)


@pytest.mark.parametrize("n_scans", [259, 2000])
def test_scan_snr_ticks_remain_readable(tmp_path, monkeypatch, caplog, n_scans):
    """Long schedules have bounded, non-overlapping ticks and no categorical minor ticks."""
    from matplotlib.figure import Figure

    from vlbipy.plotting import ScanSNRPlotter

    survey = _demo_survey(n_scans)
    survey.scan_sources = ["3C345" if i % 2 else "J1848+3219" for i in range(n_scans)]
    captured = []

    def inspect(figure, *args, **kwargs):
        figure.canvas.draw()
        axis = figure.axes[0]
        assert len(axis.get_yticks()) <= 30
        assert len(axis.yaxis.get_minorticklocs()) == 0
        boxes = sorted((label.get_window_extent() for label in axis.get_yticklabels()), key=lambda box: box.y0)
        assert all(a.y1 < b.y0 for a, b in zip(boxes, boxes[1:]))
        captured.append(figure)

    monkeypatch.setattr(Figure, "savefig", inspect)
    ScanSNRPlotter(tmp_path).plot(survey)
    assert captured
    assert "Locator attempting" not in caplog.text


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
    # A single-panel copy per source as well, so the per-source report has one each.
    assert {p.name for p in tmp_path.glob("uvc.uv_coverage.*.png")} == {
        "uvc.uv_coverage.3C345.png", "uvc.uv_coverage.J1848+3219.png"}
    # nothing to plot -> no file, but the path is still returned
    empty = plot_uv_coverage({"fields": {}}, str(tmp_path / "none.png"))
    assert empty.endswith("none.png") and not (tmp_path / "none.png").exists()


def test_plot_uv_coverage_can_skip_the_per_source_copies(tmp_path):
    from vlbipy.plotting import plot_uv_coverage

    data = {"fields": {"A": {"u": [1.0, 2.0], "v": [1.0, -2.0]}}, "unit": "Mlambda"}
    plot_uv_coverage(data, str(tmp_path / "p.uv_coverage.png"), per_source=False)
    assert [p.name for p in tmp_path.glob("*.png")] == ["p.uv_coverage.png"]


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


def test_lightcurve_panels_have_their_own_amplitude_axis(tmp_path, monkeypatch):
    """A faint target and a bright calibrator must not share y limits: each panel scales to its own source."""
    import matplotlib.pyplot as plt

    from vlbipy import plotting
    assert plotting.lightcurve_amplitude_limit([]) == 1.0
    assert plotting.lightcurve_amplitude_limit([np.array([0.5, np.nan]), np.array([2.0])]) == pytest.approx(2.1, rel=1e-2)
    times = 5e9 + np.arange(0, 120, 2.0)
    n_vis = np.full(times.size, 400.0)
    sources = {name: {"times": times, "vis_sum": n_vis * flux + 0j, "n_vis": n_vis, "scans": np.ones(times.size)}
               for name, flux in (("TARGET", 0.02), ("CAL", 1.5))}
    limits = {}
    monkeypatch.setattr(plt, "close", lambda figure: limits.update(
        {axis.get_title(loc="left"): axis.get_ylim() for axis in figure.axes}))
    plotting.plot_total_lightcurve({"sources": sources, "column": "corrected", "time_start": 5e9}, tmp_path, "lc03")
    assert limits["TARGET"] == pytest.approx((0.0, 0.021)) and limits["CAL"] == pytest.approx((0.0, 1.575))


def _scan_read(scan, antennas, refant=None, field="3C345"):
    """A per-scan read as the spectrum/autocorr readers return it: 2 subbands x 8 channels x 2 hands."""
    rng = np.random.default_rng(scan)
    spectra = {name: (1.0 + rng.random((2, 8, 2))) * np.exp(1j * rng.random((2, 8, 2))) for name in antennas}
    read = {"antennas": antennas, "spectra": spectra, "n_spw": 2, "n_channels": 8, "polarizations": ["RR", "LL"],
            "frequencies_ghz": [list(np.linspace(1.6, 1.61, 8)), list(np.linspace(1.62, 1.63, 8))],
            "scans": [scan], "field": field, "column": "data"}
    if refant:
        read["refant"] = refant
    return read


def test_antenna_scan_spectra_has_one_panel_per_scan_of_that_antenna(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from vlbipy.plotting import antenna_scan_filename, plot_antenna_scan_spectra
    titles = {}

    def inspect(figure, outfile, *args, **kwargs):
        titles[Path(outfile).name] = [axis.get_title(loc="left") for axis in figure.axes if axis.get_title(loc="left")]

    monkeypatch.setattr(Figure, "savefig", inspect)
    autos = [_scan_read(7, ["EF", "WB"]), _scan_read(3, ["EF", "WB", "O8"])]
    crosses = [_scan_read(7, ["WB"], refant="EF"), _scan_read(3, ["EF", "O8"], refant="WB")]
    assert plot_antenna_scan_spectra(autos, "O8", "autocorr", tmp_path, "rsm07", label="raw").name == \
        antenna_scan_filename("rsm07", "raw", "O8", "autocorr") == "rsm07.raw_antO8.autocorr.png"
    plot_antenna_scan_spectra(autos, "EF", "autocorr", tmp_path, "rsm07", label="raw")
    assert titles["rsm07.raw_antEF.autocorr.png"] == ["scan 3 · 3C345", "scan 7 · 3C345"]      # in time order
    plot_antenna_scan_spectra(crosses, "EF", "spectrum", tmp_path, "rsm07", label="raw")
    assert titles["rsm07.raw_antEF.spectrum.png"] == ["scan 3 · 3C345 · WB-EF"]                # not as refant
    assert plot_antenna_scan_spectra(crosses, "TR", "spectrum", tmp_path, "rsm07", label="raw") is None


def test_model_is_drawn_as_a_grey_dashed_line_broken_at_gaps_and_wraps(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from vlbipy.plotting import MODEL_LINE, _model_line, plot_baseline_timeseries
    hours = np.array([0.0, 0.1, 0.2, 2.0, 2.1])
    values = np.exp(1j * np.radians([170.0, 179.0, -179.0, 0.0, 10.0]))[:, None].repeat(2, axis=1)
    (amp_x, amplitude), (_, phase) = _model_line(hours, values, ["RR", "LL"], gap=0.3)
    assert np.isnan(amplitude).sum() == 1 and np.isnan(amp_x[3])           # one break: the gap after 0.2 h
    assert np.isnan(phase).sum() == 2                                        # and the wrap at +-180 deg
    lines = []

    def inspect(figure, *args, **kwargs):
        lines.extend(line for axis in figure.axes for line in axis.get_lines()
                     if line.get_color() == MODEL_LINE["color"] and line.get_linestyle() == "--")

    monkeypatch.setattr(Figure, "savefig", inspect)
    times = (5.0e9 + hours * 3600.0).tolist()
    series = {"baselines": {"EF-WB": values * 0.9, "EF-O8": values}, "times": times, "polarizations": ["RR", "LL"],
              "model_baselines": {"EF-WB": values}, "field": "TGT", "column": "corrected"}
    plot_baseline_timeseries(series, tmp_path, "p1", label="final_TGT")
    assert len(lines) == 2                                                   # amplitude and phase of EF-WB only
