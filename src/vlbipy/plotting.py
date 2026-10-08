"""Diagnostic plots of calibration tables (matplotlib -> PNG).

:class:`CalTablePlotter` reads CASA calibration tables (via casatools) and
renders per-antenna diagnostic figures. It dispatches on the table's ``VisCal``
keyword, so the same class serves every present and future caltable:

* ``B TSYS``       -> Tsys vs time, one subplot per antenna, color per polarization
* ``EPowerCurve``  -> gain-curve polynomial vs elevation, one subplot per antenna
* ``Fringe Jones`` -> phase / delay / rate vs time (one PNG each), per antenna
* ``G/B Jones``    -> amplitude and phase of the (complex) gains vs time/channel

Style: fixed colors always encode polarization; subbands use line style or marker
shape as a secondary encoding, with recessive grids and shared axes per figure.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # headless PNG rendering; must precede pyplot import
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from .logging_utils import get_logger
from .models import ScanSNRSurvey
from .tools import mjdsec2datetime

logger = get_logger()

#: Fixed-order subband colors (Okabe-Ito, CVD-validated); polarization uses linestyle.
SUBBAND_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
                  "#56B4E9", "#D55E00", "#F0E442", "#000000")
#: Fixed color per correlation product, used in *every* plot so a polarization always
#: looks the same: parallel hands red/green, cross hands dark blue/dark green. The
#: linear-basis products map onto the same scheme.
POLARIZATION_COLORS = {
    "RR": "#D62728", "LL": "#2CA02C", "RL": "#00008B", "LR": "#006400",
    "XX": "#D62728", "YY": "#2CA02C", "XY": "#00008B", "YX": "#006400",
}
#: Fallback cycle for labels outside the known correlation products.
POL_FALLBACK_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
#: Kept for the caltable plots, which index by solution order rather than by name.
POL_COLORS = ("#D62728", "#2CA02C", "#00008B", "#006400")


def polarization_color(label: str, index: int = 0) -> str:
    """Return the fixed color for a correlation product.

    Colors are keyed by name rather than by position so RR is the same red
    whether it is the first product in one dataset and the fourth in another.
    """
    return POLARIZATION_COLORS.get(str(label).upper(),
                                   POL_FALLBACK_COLORS[index % len(POL_FALLBACK_COLORS)])
#: Linestyle per polarization (used where color encodes the subband).
POL_LINESTYLES = ("-", "--", ":", "-.")
#: Polarization display names (circular basis, the VLBI standard).
POL_LABELS = ("R", "L", "P3", "P4")

#: Fringe-table parameter layout: FPARAM holds (phase, delay, rate, disp) per polarization.
_FRINGE_PARAMS = (("phase", "phase (deg)"), ("delay", "delay (ns)"),
                  ("rate", "rate (ps/s)"), ("disp", "dispersive delay"))

#: Colormap for the SNR matrix (perceptually uniform, CVD-safe).
SNR_COLORMAP = "viridis"
#: Fill color for scan/antenna cells with no solution (absent antenna or failed solve).
NO_DATA_COLOR = "#d9d9d9"


def style_axis(axis, *, grid: bool = True) -> None:
    """Give an axis a closed black frame with ticks on all four sides.

    A full box makes values readable against either edge and keeps panels in a
    grid visually separated; ticks on all four sides let a point be read off
    without tracing across the whole panel.
    """
    for side in ("top", "bottom", "left", "right"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color("black")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True,
                     color="black")
    axis.minorticks_on()
    if grid:
        axis.grid(True, alpha=0.25, linewidth=0.5)


def subplot_grid(n_panels: int, max_cols: int = 4) -> tuple[int, int]:
    """Return (nrows, ncols) for a grid of ``n_panels`` antenna subplots."""
    ncols = min(max_cols, max(1, n_panels))
    return math.ceil(n_panels / ncols), ncols


def evaluate_gain_curve(coefficients, elevation_deg) -> np.ndarray:
    """Evaluate a gain-curve polynomial sum(c_k * el^k) over elevations in degrees."""
    coefficients = np.asarray(coefficients, dtype=float)
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    return sum(c * elevation_deg ** k for k, c in enumerate(coefficients))


def _parallel_hand_mean(values: np.ndarray, pol_labels: list) -> np.ndarray:
    """Average the parallel-hand columns of ``(n, n_pol)`` values into a Stokes-I-like series."""
    values = np.asarray(values)
    if values.ndim < 2:
        return values
    parallel = [p for p in range(values.shape[1])
                if p >= len(pol_labels) or str(pol_labels[p]).upper() in ("RR", "LL", "XX", "YY")]
    return np.nanmean(values[:, parallel or list(range(values.shape[1]))], axis=1)


def _stacked_pair(figure, cell, height_ratios=(2, 1)):
    """Split a gridspec cell into a gapless amplitude (top) / phase (bottom) pair sharing x.

    Returns ``(amp_axis, phase_axis)``.
    """
    inner = cell.subgridspec(2, 1, height_ratios=list(height_ratios), hspace=0)
    amp_axis = figure.add_subplot(inner[0])
    phase_axis = figure.add_subplot(inner[1], sharex=amp_axis)
    return amp_axis, phase_axis


def _style_stacked_pair(amp_axis, phase_axis, *, phase_ylim: tuple = (-180.0, 180.0),
                        fontsize: int = 8, amp_label: str = "amplitude",
                        phase_label: str = "phase (deg)") -> None:
    """Label and frame an amplitude/phase pair; the top panel loses its x tick labels."""
    phase_axis.set_ylim(*phase_ylim)
    phase_axis.set_yticks([-180, 0, 180])
    phase_axis.set_ylabel(phase_label, fontsize=fontsize)
    phase_axis.tick_params(labelsize=fontsize)
    amp_axis.set_ylabel(amp_label, fontsize=fontsize)
    amp_axis.tick_params(labelbottom=False, labelsize=fontsize)
    for axis in (amp_axis, phase_axis):
        style_axis(axis)


def plot_baseline_panels(panels: list, plot_dir: Union[str, Path], project_code: str, *,
                         pol_labels: list, x_label: str, title: str, outfile_stem: str,
                         phase_ylim: tuple = (-180.0, 180.0), dpi: int = 150) -> Path:
    """Render one figure of per-baseline amplitude+phase panels.

    Each baseline gets a stacked pair of axes sharing an x axis with no gap:
    amplitude on top across the upper two thirds, phase below in the lower
    third drawn as dots. Amplitude and phase belong together — a feature is only
    interpretable when you can see whether it appears in both — and the split
    heights reflect that phase is bounded while amplitude carries the dynamic
    range. Phase as dots avoids the vertical streaks that lines draw across
    every +-180 degree wrap.

    Parameters
    ----------
    panels : list
        One entry per baseline: ``(title, [(x, values), ...])`` where ``values``
        is complex with shape ``(len(x), n_pol)``. Several segments per baseline
        keep subbands from being joined by a line across the gap between them.
    pol_labels : list of str
        Correlation product names, used for the fixed colors and the legend.
    x_label : str
        Axis label for the shared x axis.
    title, outfile_stem : str
        Figure title and output file stem.
    phase_ylim : tuple
        Phase limits in degrees.

    Returns
    -------
    pathlib.Path
        The written PNG.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not panels:
        logger.warning("no baselines to plot for {}", project_code)
        return plot_dir / f"{outfile_stem}.png"

    nrows, ncols = subplot_grid(len(panels))
    figure = plt.figure(figsize=(3.7 * ncols, 3.0 * nrows), layout="constrained")
    outer = figure.add_gridspec(nrows, ncols)
    x_values = [np.asarray(x, dtype=float) for _, segments in panels for x, _ in segments]
    finite_x = np.concatenate([x[np.isfinite(x)] for x in x_values if np.isfinite(x).any()]) if any(
        np.isfinite(x).any() for x in x_values) else np.array([])
    shared_xlim = (float(finite_x.min()), float(finite_x.max())) if finite_x.size else None
    for index, (panel_title, segments) in enumerate(panels):
        amp_axis, phase_axis = _stacked_pair(figure, outer[index // ncols, index % ncols])
        for segment, (x, values) in enumerate(segments):
            x = np.asarray(x, dtype=float)
            values = np.asarray(values)
            for pol in range(values.shape[1] if values.ndim > 1 else 1):
                column = values[:, pol] if values.ndim > 1 else values
                name = pol_labels[pol] if pol < len(pol_labels) else f"P{pol + 1}"
                color = polarization_color(name, pol)
                phase_axis.plot(x[:len(column)], np.degrees(np.angle(column)), ls="none",
                                marker=".", ms=2.0, color=color)
                amp_axis.plot(x[:len(column)], np.abs(column), ls="none", marker=".",
                              ms=2.0, color=color,
                              label=name if segment == 0 else None)
        _style_stacked_pair(amp_axis, phase_axis, phase_ylim=phase_ylim, fontsize=7)
        if shared_xlim and shared_xlim[0] < shared_xlim[1]:
            phase_axis.set_xlim(*shared_xlim)
        amp_axis.set_title(panel_title, fontsize=10, fontweight="bold", loc="left")
        if index // ncols == nrows - 1:
            phase_axis.set_xlabel(x_label, fontsize=8)

    handles = [Line2D([], [], color=polarization_color(name, i), lw=2, label=name)
               for i, name in enumerate(pol_labels)]
    figure.legend(handles=handles, loc="outside upper right", ncols=min(len(handles), 6),
                  frameon=False, fontsize=8)
    figure.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    outfile = plot_dir / f"{outfile_stem}.png"
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("baseline plot written: {}", outfile)
    return outfile


def plot_corrected_spectrum(spectrum: dict, plot_dir: Union[str, Path], project_code: str = "",
                            label: str = "", dpi: int = 150) -> list[Path]:
    """Plot amplitude and phase vs sky frequency, per baseline to the reference antenna.

    One figure: each baseline is a stacked phase/amplitude pair (see
    :func:`plot_baseline_panels`). Subbands are drawn as separate segments so
    nothing is joined across the gap between them.

    This is the visual check that the instrumental calibration worked: after
    SBD + bandpass the amplitudes should be flat across each subband and the
    phases flat *and* aligned between subbands. A slope within a subband is
    residual single-band delay; a step between subbands is residual multi-band
    delay, which the global fringe fit removes.

    Returns
    -------
    list of pathlib.Path
        The written PNG (a single-item list, kept for caller compatibility).
    """
    antennas = spectrum.get("antennas") or []
    if not antennas:
        logger.warning("no baselines to plot for {}", project_code)
        return []
    n_spw, n_chan = int(spectrum["n_spw"]), int(spectrum["n_channels"])
    pol_labels = spectrum.get("polarizations") or ["P1", "P2"]
    refant = spectrum.get("refant", "")
    frequencies = spectrum.get("frequencies_ghz") or [list(range(n_chan)) for _ in range(n_spw)]

    panels = []
    for antenna in antennas:
        values = np.asarray(spectrum["spectra"][antenna])          # (n_spw, n_chan, n_pol)
        segments = [(np.asarray(frequencies[spw], dtype=float), values[spw])
                    for spw in range(min(n_spw, values.shape[0]))]
        panels.append((f"{refant}-{antenna}", segments))

    scans = spectrum.get("scans") or []
    details = [f"{spectrum.get('column', '')} data"]
    if spectrum.get("field"):
        details.append(spectrum["field"])
    if scans:
        details.append(f"scan {scans[0]}" if len(scans) == 1 else f"{len(scans)} scans")
    suffix = f".{label}" if label else ""
    return [plot_baseline_panels(
        panels, plot_dir, project_code, pol_labels=pol_labels, x_label="frequency (GHz)",
        title=f"{project_code} — amplitude and phase vs frequency ({', '.join(details)})",
        outfile_stem=f"{project_code}{suffix}.spectrum", dpi=dpi)]


def plot_baseline_timeseries(series: dict, plot_dir: Union[str, Path], project_code: str = "",
                             label: str = "", dpi: int = 150) -> list[Path]:
    """Plot amplitude and phase vs time, per baseline, in the same stacked layout.

    The frequency companion to :func:`plot_corrected_spectrum`: same panels, same
    fixed polarization colors, x axis in hours from the start of the observation.

    Returns
    -------
    list of pathlib.Path
        The written PNG (a single-item list).
    """
    baselines = series.get("baselines") or {}
    if not baselines:
        logger.warning("no baselines to plot for {}", project_code)
        return []
    pol_labels = series.get("polarizations") or ["P1", "P2"]
    times = np.asarray(series.get("times") or [], dtype=float)
    hours = (times - times.min()) / 3600.0 if times.size else np.array([0.0])
    panels = [(name if isinstance(name, str) else "-".join(name), [(hours, np.asarray(values))])
              for name, values in baselines.items()]
    details = [f"{series.get('column', '')} data"]
    if series.get("field"):
        details.append(series["field"])
    suffix = f".{label}" if label else ""
    return [plot_baseline_panels(
        panels, plot_dir, project_code, pol_labels=pol_labels,
        x_label="time (hours from start)",
        title=f"{project_code} — amplitude and phase vs time ({', '.join(details)})",
        outfile_stem=f"{project_code}{suffix}.timeseries", dpi=dpi)]


def plot_radplot(uvdata: dict, plot_dir: Union[str, Path], project_code: str = "",
                 label: str = "", dpi: int = 150) -> Path:
    """Plot amplitude and phase against uv distance for one source (a "radplot").

    Same stacked layout as the other baseline plots — amplitude across the upper
    two thirds, phase in the lower third, fixed color per polarization — but as
    a scatter, since uv distance orders points by baseline length rather than
    forming a series to join.

    This is the view that shows source structure: a flat amplitude profile means
    an unresolved point source, a profile falling with uv distance means resolved
    structure, and phases scattered about zero confirm the calibration on a
    calibrator (structure would show as an organised phase trend).

    Parameters
    ----------
    uvdata : dict
        Output of the backend's ``read_uvdistance``. An optional ``"model"`` entry
        ``{"uvdist_mlambda", "values"}`` is drawn as a black Stokes-I-like line
        (mean of the parallel hands) over both panels.
    label : str
        Extra tag for the file name.

    Returns
    -------
    pathlib.Path
        The written PNG.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    field = str(uvdata.get("field", "") or "source").replace(",", "_").replace("/", "_")
    suffix = f".{label}" if label else ""
    outfile = plot_dir / f"{project_code}{suffix}.radplot.{field}.png"

    uvdist = np.asarray(uvdata.get("uvdist_mlambda", []), dtype=float)
    values = np.asarray(uvdata.get("values", []))
    if uvdist.size == 0 or values.size == 0:
        logger.warning("no uv data to plot for {} on {}", project_code, field)
        return outfile
    pol_labels = uvdata.get("polarizations") or ["P1", "P2"]

    figure = plt.figure(figsize=(7.5, 5.5), layout="constrained")
    amp_axis, phase_axis = _stacked_pair(figure, figure.add_gridspec(1, 1)[0])
    for pol in range(values.shape[1] if values.ndim > 1 else 1):
        column = values[:, pol] if values.ndim > 1 else values
        name = pol_labels[pol] if pol < len(pol_labels) else f"P{pol + 1}"
        color = polarization_color(name, pol)
        finite = np.isfinite(column)
        phase_axis.plot(uvdist[finite], np.degrees(np.angle(column[finite])), ls="none",
                        marker=".", ms=1.8, alpha=0.5, color=color)
        amp_axis.plot(uvdist[finite], np.abs(column[finite]), ls="none", marker=".", ms=1.8,
                      alpha=0.5, color=color, label=name)
    handles = [Line2D([], [], color=polarization_color(name, i), lw=0, marker="o", ms=5,
                      label=name) for i, name in enumerate(pol_labels)]
    model = uvdata.get("model") or {}
    model_uv = np.asarray(model.get("uvdist_mlambda", []), dtype=float)
    model_values = np.asarray(model.get("values", []))
    if model_uv.size and model_values.size:
        stokes = _parallel_hand_mean(model_values, pol_labels)
        order = np.argsort(model_uv)
        amp_axis.plot(model_uv[order], np.abs(stokes)[order], color="black", lw=1.2, label="model")
        phase_axis.plot(model_uv[order], np.degrees(np.angle(stokes))[order], color="black", lw=0.6)
        handles.append(Line2D([], [], color="black", lw=1.2, label="model"))
    _style_stacked_pair(amp_axis, phase_axis, fontsize=8)
    phase_axis.set_xlabel(r"uv distance (M$\lambda$)", fontsize=9)
    amp_axis.set_ylim(bottom=0.0)
    figure.legend(handles=handles, loc="outside upper right", ncols=len(handles),
                  frameon=False, fontsize=8)
    figure.suptitle(f"{project_code} — {field}: amplitude and phase vs uv distance\n"
                    f"{uvdata.get('column', '')} data, {uvdata.get('time_bin', 0):.0f} s bins, "
                    "averaged per subband", x=0.01, ha="left", fontsize=11, fontweight="bold")
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("radplot written: {}", outfile)
    return outfile


def plot_uv_coverage(data: dict, output: str, *, title: str = "", dpi: int = 150) -> str:
    """Plot the sampled uv plane (u vs v) for each source, one panel per field.

    Each panel shows the sampled points and their conjugates ``(-u, -v)`` with
    equal aspect; all panels share the same symmetric limits so the coverage of
    a weak target can be compared directly with that of its calibrators.

    Parameters
    ----------
    data : dict
        Output of the backend's ``read_uv_coverage``.
    output : str
        Path of the PNG to write.
    title : str
        Figure title (the unit and frequency are appended).

    Returns
    -------
    str
        The written PNG path (``output``).
    """
    outfile = Path(output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fields = data.get("fields") or {}
    if not fields:
        logger.warning("no uv points to plot for {}", outfile.name)
        return str(outfile)
    unit = data.get("unit", "Mlambda")
    unit_label = r"M$\lambda$" if unit == "Mlambda" else unit
    limit = max((np.abs(np.r_[np.asarray(f.get("u", []), dtype=float),
                             np.asarray(f.get("v", []), dtype=float)]).max(initial=0.0)
                 for f in fields.values()), default=0.0) * 1.05 or 1.0

    nrows, ncols = subplot_grid(len(fields), max_cols=3)
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False,
                                layout="constrained")
    for axis, (name, points) in zip(axes.flat, fields.items()):
        u, v = np.asarray(points.get("u", []), dtype=float), np.asarray(points.get("v", []), dtype=float)
        axis.plot(np.r_[u, -u], np.r_[v, -v], ls="none", marker=".", ms=1.0, alpha=0.6,
                  color=SUBBAND_COLORS[0], rasterized=True)
        axis.set_xlim(limit, -limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
        axis.set_title(f"{name} ({u.size} pts)", fontsize=9)
        style_axis(axis)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(fields):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel(f"u ({unit_label})", fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel(f"v ({unit_label})", fontsize=9)
    freq = data.get("freq_ghz")
    detail = f" at {freq:.2f} GHz" if freq else ""
    figure.suptitle(f"{title or 'uv coverage'}{detail}", x=0.01, ha="left", fontsize=11,
                    fontweight="bold")
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("uv coverage plot written: {}", outfile)
    return str(outfile)


def plot_baseline_corner(spectra: dict, plot_dir: Union[str, Path], project_code: str = "",
                         quantity: str = "phase", label: str = "", dpi: int = 150) -> Path:
    """Plot a time x frequency dynamic spectrum for every antenna pair, corner-style.

    Antennas run along both axes and each cell holds one baseline's time
    (vertical) against frequency (horizontal), coloured by phase or amplitude.
    Only the lower triangle is drawn — a baseline and its conjugate carry the
    same information — so the grid stays one panel per physical baseline.

    Laying every baseline out together is what makes array-wide structure
    visible: a bad antenna shows as a whole row and column going wrong, while a
    single bad baseline stays confined to its own cell.

    Parameters
    ----------
    spectra : dict
        Output of the backend's ``read_dynamic_spectra``.
    quantity : str
        ``"phase"`` (cyclic colour map, fixed -180..180) or ``"amp"``.
    label : str
        Extra tag for the file name.

    Returns
    -------
    pathlib.Path
        The written PNG.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    antennas = spectra.get("antennas") or []
    baselines = spectra.get("baselines") or {}
    if len(antennas) < 2 or not baselines:
        logger.warning("not enough baselines for a corner plot of {}", project_code)
        return plot_dir / f"{project_code}.corner.{quantity}.png"

    n = len(antennas)
    index = {name: i for i, name in enumerate(antennas)}
    freqs = np.asarray(spectra.get("frequencies_ghz") or [], dtype=float)
    times = np.asarray(spectra.get("times") or [], dtype=float)
    hours = (times - times.min()) / 3600.0 if times.size else np.array([0.0])
    extent = ((float(freqs.min()), float(freqs.max()), float(hours.max()), float(hours.min()))
              if freqs.size and hours.size else None)

    if quantity == "phase":
        transform, cmap, norm_kw, cbar_label = (
            lambda v: np.degrees(np.angle(v)), "twilight_shifted",
            {"vmin": -180.0, "vmax": 180.0}, "phase (deg)")
    else:
        finite = np.concatenate([np.abs(v[np.isfinite(v)]).ravel() for v in baselines.values()
                                 if np.isfinite(v).any()] or [np.array([0.0, 1.0])])
        transform, cmap, cbar_label = np.abs, "viridis", "amplitude"
        norm_kw = {"vmin": float(np.nanpercentile(finite, 2)),
                   "vmax": float(np.nanpercentile(finite, 98))}

    # n-1 rows/columns: the diagonal (an antenna with itself) is never a baseline.
    figure, axes = plt.subplots(n - 1, n - 1, figsize=(1.7 * (n - 1) + 2.0, 1.7 * (n - 1) + 1.5),
                                squeeze=False, layout="constrained")
    image = None
    for row in range(n - 1):
        for column in range(n - 1):
            axis = axes[row][column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column > row:                       # upper triangle = conjugate duplicate
                axis.set_visible(False)
                continue
            pair = (antennas[column], antennas[row + 1])
            values = baselines.get(pair)
            if values is None:
                values = baselines.get((pair[1], pair[0]))
                if values is not None:
                    values = np.conjugate(values)
            if values is None or not np.isfinite(values).any():
                axis.set_facecolor(NO_DATA_COLOR)
            else:
                image = axis.imshow(transform(values), aspect="auto", origin="upper",
                                    interpolation="nearest", cmap=cmap, extent=extent, **norm_kw)
            for side in ("top", "bottom", "left", "right"):
                axis.spines[side].set_visible(True)
                axis.spines[side].set_color("black")
                axis.spines[side].set_linewidth(0.8)
            if column == 0:
                axis.set_ylabel(antennas[row + 1], fontsize=8, fontweight="bold")
            if row == n - 2:
                axis.set_xlabel(antennas[column], fontsize=8, fontweight="bold")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.01)
        colorbar.set_label(cbar_label, fontsize=9)
    stokes = "Stokes I" if spectra.get("stokes_i") else "parallel hands"
    figure.suptitle(f"{project_code} — {cbar_label} per baseline, time (vertical) vs frequency "
                    f"(horizontal)\n{spectra.get('field', '')}, {spectra.get('column', '')} data, "
                    f"{stokes}; grey = no data", x=0.01, ha="left", fontsize=11, fontweight="bold")
    suffix = f".{label}" if label else ""
    outfile = plot_dir / f"{project_code}{suffix}.corner.{quantity}.png"
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("baseline corner plot written: {}", outfile)
    return outfile


def plot_bandpass_profile(measurement: dict, plot_dir: Union[str, Path], project_code: str = "",
                          dpi: int = 150) -> Path:
    """Plot the per-channel band profile behind the edge-channel decision.

    Three stacked panels sharing the channel axis — median bandpass amplitude,
    phase scatter, and the fraction of solutions the solver flagged — with the
    channels selected for flagging shaded. This is the evidence for how many
    edge channels were trimmed, so a user can see whether the cut was right.

    When the measurement carries a per-antenna breakdown (``"antennas"``:
    ``{ant: {"amplitude_profile", "phase_profile", "n_edge"}}``) one amplitude
    panel per antenna is drawn instead, each with its own edge cut shaded, since
    a single dish with a narrower usable band is what decides the trim.

    Parameters
    ----------
    measurement : dict
        Output of the backend's ``measure_edge_channels``.
    plot_dir : str or pathlib.Path
        Output directory (created if needed).
    project_code : str
        Used in the title and file name.
    dpi : int
        Output resolution.

    Returns
    -------
    pathlib.Path
        The written PNG.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    outfile = plot_dir / f"{project_code}.bandpass_profile.png"
    channels = np.arange(int(measurement["n_channels"]))
    n_edge = measurement.get("n_edge", 0)
    edge_text = (f"{n_edge[0]}/{n_edge[1]}" if isinstance(n_edge, (tuple, list)) else str(int(n_edge)))
    antennas = measurement.get("antennas") or {}
    if antennas:
        nrows, ncols = subplot_grid(len(antennas))
        figure, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows), sharex=True,
                                    squeeze=False, layout="constrained")
        for axis, (name, entry) in zip(axes.flat, antennas.items()):
            values = np.asarray(entry.get("amplitude_profile", []), dtype=float)
            if values.size == channels.size:
                axis.plot(channels, values, ls="none", marker=".", ms=3, color=POL_COLORS[0])
            _shade_edges(axis, channels.size, entry.get("n_edge", n_edge))
            axis.set_title(name, fontsize=10, fontweight="bold", loc="left")
            axis.set_ylabel("median |B|", fontsize=8)
            axis.tick_params(labelsize=7)
            style_axis(axis)
        for axis in axes.flat[len(antennas):]:
            axis.set_visible(False)
        for axis in axes[-1, :]:
            axis.set_xlabel("channel", fontsize=9)
        figure.suptitle(f"{project_code} — subband response per antenna; shaded = flagged "
                        f"({edge_text} channel(s) each edge)", x=0.01, ha="left", fontsize=11,
                        fontweight="bold")
    else:
        panels = (("amplitude_profile", "median |B|", POL_COLORS[0]),
                  ("phase_profile", "phase scatter (rad)", POL_COLORS[1]),
                  ("flagged_fraction", "flagged fraction", "0.35"))
        figure, axes = plt.subplots(3, 1, figsize=(7.5, 6.0), sharex=True, layout="constrained")
        for axis, (key, label, color) in zip(axes, panels):
            values = np.asarray(measurement.get(key, []), dtype=float)
            if values.size == channels.size:
                axis.plot(channels, values, ls="none", marker=".", ms=3, color=color)
            axis.set_ylabel(label, fontsize=9)
            style_axis(axis)
            _shade_edges(axis, channels.size, n_edge)
        axes[-1].set_xlabel("channel", fontsize=9)
        axes[0].set_title(f"{project_code} — subband response; shaded = flagged "
                          f"({edge_text} channel(s) each edge)", fontsize=10, loc="left")
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("bandpass profile written: {}", outfile)
    return outfile


def _shade_edges(axis, n_channels: int, n_edge) -> None:
    """Shade the ``n_edge`` (int or ``(low, high)``) channels flagged at each subband edge."""
    low, high = (int(n_edge[0]), int(n_edge[1])) if isinstance(n_edge, (tuple, list)) \
        else (int(n_edge or 0), int(n_edge or 0))
    if low > 0:
        axis.axvspan(-0.5, low - 0.5, color="#D55E00", alpha=0.15, lw=0)
    if high > 0:
        axis.axvspan(n_channels - high - 0.5, n_channels - 0.5, color="#D55E00", alpha=0.15, lw=0)


def plot_autocorr_spectrum(spectrum: dict, plot_dir: Union[str, Path], project_code: str = "",
                           label: str = "", dpi: int = 150) -> str:
    """Plot autocorrelation amplitude vs frequency, one subplot per antenna.

    Autocorrelations show each antenna's own bandpass shape and any RFI or
    dead subband without the extra uncertainty of a fringe. Colour always encodes
    polarization; all subbands share one solid line and tile the frequency axis.

    Parameters
    ----------
    spectrum : dict
        Keys ``antennas``, ``spectra`` ({ant: (n_spw, n_chan, n_pol)}), ``n_spw``,
        ``n_channels``, ``polarizations``, ``frequencies_ghz`` (per-subband arrays or
        ``None`` for channel index), ``scans``, ``field``, ``column``.
    label : str
        Extra tag for the file name.

    Returns
    -------
    str
        The written PNG path ``{code}[.{label}].autocorr.png``.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{label}" if label else ""
    outfile = plot_dir / f"{project_code}{suffix}.autocorr.png"
    antennas = spectrum.get("antennas") or []
    if not antennas:
        logger.warning("no autocorrelations to plot for {}", project_code)
        return str(outfile)
    n_spw, n_chan = int(spectrum["n_spw"]), int(spectrum["n_channels"])
    pol_labels = spectrum.get("polarizations") or ["P1", "P2"]
    frequencies = spectrum.get("frequencies_ghz") or [np.arange(n_chan) for _ in range(n_spw)]
    x_label = "frequency (GHz)" if spectrum.get("frequencies_ghz") else "channel"

    nrows, ncols = subplot_grid(len(antennas))
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), sharex=True,
                                squeeze=False, layout="constrained")
    for axis, antenna in zip(axes.flat, antennas):
        values = np.asarray(spectrum["spectra"][antenna])
        for spw in range(min(n_spw, values.shape[0])):
            x = np.asarray(frequencies[spw], dtype=float)
            for pol in range(values.shape[2] if values.ndim > 2 else 1):
                amp = np.abs(values[spw, :, pol] if values.ndim > 2 else values[spw]).astype(float)
                amp[~np.isfinite(amp) | (amp == 0)] = np.nan
                name = pol_labels[pol] if pol < len(pol_labels) else f"P{pol + 1}"
                axis.plot(x[:amp.size], amp[:x.size], color=polarization_color(name, pol),
                          ls="-", lw=0.9)
        axis.set_title(antenna, fontsize=10, fontweight="bold", loc="left")
        axis.set_ylabel("amplitude", fontsize=8)
        axis.tick_params(labelsize=7)
        style_axis(axis)
    for axis in axes.flat[len(antennas):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel(x_label, fontsize=9)
    handles = [Line2D([], [], color=polarization_color(name, p), lw=2,
                       label=str(name)) for p, name in enumerate(pol_labels)]
    figure.legend(handles=handles, loc="outside upper right", ncols=min(len(handles), 8),
                  frameon=False, fontsize=8)
    scans = spectrum.get("scans") or []
    scan_text = "scan" if len(scans) == 1 else "scans"
    figure.suptitle(f"{spectrum.get('field', '')} autocorrelations {scan_text} "
                    f"{', '.join(str(s) for s in scans)} ({spectrum.get('column', '')})",
                    x=0.01, ha="left", fontsize=11, fontweight="bold")
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("autocorrelation plot written: {}", outfile)
    return str(outfile)


def _bin_lightcurve(times: np.ndarray, vis_sum: np.ndarray, n_vis: np.ndarray, scans: np.ndarray,
                    width_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Coherently average integrations into bins of ``width_sec`` within each scan.

    ``0`` keeps the native integrations and ``-1`` averages each whole scan.
    Returns ``(bin_times, amplitudes)`` with amplitude ``|sum(vis_sum)| / sum(n_vis)``.
    """
    if width_sec == 0:
        keys = np.arange(times.size)
    elif width_sec < 0:
        keys = scans
    else:
        keys = np.empty(times.size, dtype=np.int64)
        for scan in np.unique(scans):
            rows = scans == scan
            keys[rows] = scan * 10 ** 6 + ((times[rows] - times[rows].min()) // width_sec).astype(np.int64)
    _, index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    summed = np.zeros(index.size, dtype=complex)
    np.add.at(summed, inverse, vis_sum)
    counts = np.zeros(index.size)
    np.add.at(counts, inverse, n_vis)
    bin_times = np.zeros(index.size)
    np.add.at(bin_times, inverse, times * n_vis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return bin_times / counts, np.abs(summed) / counts


def lightcurve_amplitude_limit(amplitudes: list) -> float:
    """Upper y limit for one source's light curve: 5% above the 99.9th percentile of everything drawn for it.

    The percentile (rather than the maximum) keeps a single noise spike from setting the scale. Returns
    1.0 when there is nothing finite and positive to show.
    """
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in amplitudes]) if amplitudes else np.zeros(0)
    values = values[np.isfinite(values)]
    top = float(np.percentile(values, 99.9)) * 1.05 if values.size else 0.0
    return top if top > 0.0 else 1.0


def plot_total_lightcurve(data: dict, plot_dir: Union[str, Path], project_code: str = "",
                          label: str = "", averaging_sec: tuple = (0, 30, 120, -1),
                          dpi: int = 150) -> str:
    """Plot the coherently averaged total visibility amplitude vs time, per source.

    One row per source sharing the time axis. Every source has its own amplitude
    axis, from zero to just above its own values (:func:`lightcurve_amplitude_limit`):
    the sources of an observation differ in flux density by orders of magnitude, and
    on a common scale the faint ones are a flat line at the bottom. Each averaging
    scale is drawn as its own series so variability shows up as structure that
    survives the longer averages while noise averages down.

    Parameters
    ----------
    data : dict
        ``{"sources": {name: {"times", "vis_sum", "n_vis", "scans"}}, "column", "time_start"}``
        with times in MJD seconds, ``vis_sum`` the coherent sum over all baselines,
        channels and parallel hands per integration and ``n_vis`` the matching count.
    averaging_sec : tuple
        Scales to draw: ``0`` native integrations, positive seconds within-scan bins,
        ``-1`` one point per scan.

    Returns
    -------
    str
        The written PNG path ``{code}[.{label}].lightcurve.png``.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{label}" if label else ""
    outfile = plot_dir / f"{project_code}{suffix}.lightcurve.png"
    sources = data.get("sources") or {}
    if not sources:
        logger.warning("no visibilities for a lightcurve of {}", project_code)
        return str(outfile)
    time_start = float(data.get("time_start") or min(np.min(s["times"]) for s in sources.values()))
    scale_names = {0: "native", -1: "per scan"}

    figure, axes = plt.subplots(len(sources), 1, figsize=(8.0, 2.4 * len(sources) + 0.8), sharex=True,
                                squeeze=False, layout="constrained")
    for axis, (name, entry) in zip(axes[:, 0], sources.items()):
        times = np.asarray(entry["times"], dtype=float)
        vis_sum = np.asarray(entry["vis_sum"], dtype=complex)
        n_vis = np.asarray(entry["n_vis"], dtype=float)
        scans = np.asarray(entry.get("scans", np.zeros(times.size)), dtype=np.int64)
        plotted: list[np.ndarray] = []
        for k, width in enumerate(averaging_sec):
            if times.size == 0:
                break
            bin_times, amp = _bin_lightcurve(times, vis_sum, n_vis, scans, float(width))
            color = SUBBAND_COLORS[k % len(SUBBAND_COLORS)]
            style = ({"ls": "none", "marker": ".", "ms": 2.5, "alpha": 0.5} if width == 0
                     else {"ls": "-", "marker": "o", "ms": 3.5, "lw": 0.8})
            axis.plot((bin_times - time_start) / 3600.0, amp, color=color,
                      label=scale_names.get(width, f"{width:g} s"), **style)
            plotted.append(np.asarray(amp, dtype=float))
        axis.set_ylim(0.0, lightcurve_amplitude_limit(plotted))
        axis.set_title(name, fontsize=10, fontweight="bold", loc="left")
        axis.set_ylabel("amplitude", fontsize=8)
        axis.tick_params(labelsize=7)
        style_axis(axis)
    axes[-1, 0].set_xlabel("time (hours from start)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc="outside upper right", ncols=len(handles) or 1,
                  frameon=False, fontsize=8)
    figure.suptitle(f"{project_code} — total coherent amplitude vs time ({data.get('column', '')} data)",
                    x=0.01, ha="left", fontsize=11, fontweight="bold")
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("lightcurve written: {}", outfile)
    return str(outfile)


def plot_subband_phase_jumps(data: dict, plot_dir: Union[str, Path], project_code: str = "",
                             label: str = "", dpi: int = 150) -> str:
    """Plot the residual phase offset of every subband vs the first one, per calibrator scan.

    One panel per baseline to the reference antenna; x is time, y the phase of
    each subband relative to the lowest subband with data in that scan
    (wrapped to +-180 deg). After a correct single-band delay these offsets
    are flat around zero; a step or a drift is an instrumental phase jump the
    calibration did not remove. Polarizations are drawn with different markers.

    Parameters
    ----------
    data : dict
        ``read_subband_phases`` output: ``antennas``, ``phases``
        (``{antenna: (n_scan, n_spw, n_pol)}`` degrees), ``scans``, ``polarizations``.

    Returns
    -------
    str
        The written PNG path ``{code}[.{label}].subband_phases.png``.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{label}" if label else ""
    outfile = plot_dir / f"{project_code}{suffix}.subband_phases.png"
    antennas = list(data.get("antennas") or [])
    scans = data.get("scans") or []
    if not antennas or not scans:
        logger.warning("no subband phases to plot for {}", project_code)
        return str(outfile)
    times = np.asarray([s["time"] for s in scans], dtype=float)
    hours = (times - times.min()) / 3600.0
    sources = sorted({s["source"] for s in scans})
    pol_markers = ("o", "s", "^", "v")
    n_rows, n_cols = subplot_grid(len(antennas), max_cols=4)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.6 * n_rows + 0.8), squeeze=False,
                                sharex=True, sharey=True, layout="constrained")
    for axis, antenna in zip(axes.ravel(), antennas):
        phases = np.asarray(data["phases"][antenna], dtype=float)         # (n_scan, n_spw, n_pol)
        for scan_idx in range(phases.shape[0]):
            finite = np.where(np.isfinite(phases[scan_idx]).any(axis=1))[0]
            if finite.size < 2:
                continue
            reference = phases[scan_idx, finite[0], :]
            for spw in finite[1:]:
                offset = (phases[scan_idx, spw, :] - reference + 180.0) % 360.0 - 180.0
                for pol in range(phases.shape[2]):
                    if np.isfinite(offset[pol]):
                        axis.plot(hours[scan_idx], offset[pol], marker=pol_markers[pol % len(pol_markers)],
                                  ms=3.5, ls="none", color=SUBBAND_COLORS[spw % len(SUBBAND_COLORS)],
                                  mfc="none" if pol else None, mew=0.8)
        axis.axhline(0.0, color="0.4", lw=0.6)
        axis.set_ylim(-180, 180)
        axis.set_yticks([-180, -90, 0, 90, 180])
        axis.set_title(f"{data.get('refant', '')}-{antenna}", fontsize=10, fontweight="bold", loc="left")
        axis.tick_params(labelsize=7)
        style_axis(axis)
    for axis in axes.ravel()[len(antennas):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("time (hours from first calibrator scan)", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("phase - first subband (deg)", fontsize=8)
    handles = [plt.Line2D([], [], color=SUBBAND_COLORS[s % len(SUBBAND_COLORS)], lw=2, label=f"subband {s}")
               for s in range(1, int(data.get("n_spw") or 1))]
    handles += [plt.Line2D([], [], color="0.3", marker=pol_markers[p % len(pol_markers)], ls="none",
                           mfc="none" if p else None, label=pol) for p, pol in enumerate(data.get("polarizations") or [])]
    figure.legend(handles=handles, loc="outside upper right", ncols=len(handles) or 1, frameon=False, fontsize=8)
    figure.suptitle(f"{project_code} — subband phase offsets per scan on {', '.join(sources)} "
                    f"({data.get('column', '')} data)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    figure.savefig(outfile, dpi=dpi)
    plt.close(figure)
    logger.info("subband phase-jump plot written: {}", outfile)
    return str(outfile)


#: Half-width of the image preview in synthesised beams (when the FITS header carries BMAJ).
IMAGE_ZOOM_BEAMS = 40.0


def _read_image_plane(fits_path: Union[str, Path]) -> tuple[np.ndarray, float]:
    """Return the first 2D plane of a FITS image (in mJy/beam) and its pixel size in mas.

    When the header has a restoring beam the plane is cut to +-IMAGE_ZOOM_BEAMS
    beams around the peak, so a VLBI source is visible in a wide field.
    """
    from astropy.io import fits
    with fits.open(str(fits_path)) as hdul:
        header, plane = hdul[0].header, np.squeeze(np.asarray(hdul[0].data, dtype=float))
    while plane.ndim > 2:
        plane = plane[0]
    unit = str(header.get("BUNIT", "Jy/beam")).lower()
    scale = 1.0 if unit.startswith("mjy") else 1e3
    pixel_mas = abs(float(header.get("CDELT1", header.get("CDELT2", 1.0)))) * 3.6e6
    bmaj_mas = float(header.get("BMAJ", 0.0)) * 3.6e6
    if bmaj_mas > 0 and pixel_mas > 0:
        half = int(round(IMAGE_ZOOM_BEAMS * bmaj_mas / pixel_mas))
        ny, nx = plane.shape
        if 2 * half < min(nx, ny):
            finite = np.where(np.isfinite(plane), plane, -np.inf)
            iy, ix = np.unravel_index(int(np.argmax(finite)), plane.shape)
            y0, x0 = min(max(iy - half, 0), ny - 2 * half), min(max(ix - half, 0), nx - 2 * half)
            plane = plane[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return plane * scale, pixel_mas


def plot_image_grid(images: dict, plot_dir: Union[str, Path], project_code: str = "",
                    dpi: int = 150) -> list[str]:
    """Plot the FITS images of each source side by side, one panel per robust weighting.

    A linear stretch clipped to ``[-3 rms, peak]`` keeps the noise floor visible
    while the colour scale stays anchored on the source; rms is the MAD-based
    scatter of the outer 20 % border, which is source-free for a centred target.

    Parameters
    ----------
    images : dict
        ``{source_name: {robust: fits_path}}``.

    Returns
    -------
    list of str
        One PNG per source, ``{code}.images.{source}.png``.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source, by_robust in (images or {}).items():
        entries = sorted(((float(r), p) for r, p in by_robust.items() if Path(p).exists()),
                         key=lambda item: item[0])
        if not entries:
            logger.warning("no images to plot for {}", source)
            continue
        figure, axes = plt.subplots(1, len(entries), figsize=(4.2 * len(entries), 4.2), squeeze=False,
                                    layout="constrained")
        for axis, (robust, path) in zip(axes[0], entries):
            plane, pixel_mas = _read_image_plane(path)
            ny, nx = plane.shape
            border = np.ones(plane.shape, dtype=bool)
            border[int(0.2 * ny):ny - int(0.2 * ny), int(0.2 * nx):nx - int(0.2 * nx)] = False
            noise = plane[border & np.isfinite(plane)]
            rms = float(1.4826 * np.median(np.abs(noise - np.median(noise)))) if noise.size else 0.0
            peak = float(np.nanmax(plane)) if np.isfinite(plane).any() else 0.0
            extent = (nx / 2 * pixel_mas, -nx / 2 * pixel_mas, -ny / 2 * pixel_mas, ny / 2 * pixel_mas)
            image = axis.imshow(plane, origin="lower", cmap="inferno", extent=extent,
                                vmin=-3 * rms, vmax=peak if peak > -3 * rms else -3 * rms + 1e-9,
                                interpolation="nearest")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02).set_label("mJy/beam", fontsize=8)
            axis.set_title(f"robust {robust:g}  peak {peak:.2f} mJy/beam  rms {rms:.3f} mJy/beam",
                           fontsize=8)
            axis.set_xlabel("relative RA (mas)", fontsize=8)
            axis.set_ylabel("relative Dec (mas)", fontsize=8)
            axis.tick_params(labelsize=7)
            style_axis(axis, grid=False)
        figure.suptitle(f"{project_code} — {source}", x=0.01, ha="left", fontsize=11, fontweight="bold")
        safe = str(source).replace(",", "_").replace("/", "_")
        outfile = plot_dir / f"{project_code}.images.{safe}.png"
        figure.savefig(outfile, dpi=dpi)
        plt.close(figure)
        logger.info("image grid written: {}", outfile)
        written.append(str(outfile))
    return written


class ScanSNRPlotter:
    """Render a :class:`~vlbipy.models.ScanSNRSurvey` as a scan x antenna SNR matrix.

    One figure per polarization: antennas along x, scans along y, one filled cell
    per (scan, antenna) with no gaps, colored by fringe SNR on a logarithmic
    scale (SNR spans orders of magnitude between a big dish and a weak one).
    Cells with no solution — antenna absent from the scan, failed solve, or the
    reference antenna itself — are drawn flat grey, so gaps in the array are
    immediately visible.

    Parameters
    ----------
    plot_dir : str or pathlib.Path
        Directory where the PNG files are written (created if needed).
    dpi : int
        Output resolution.
    """

    def __init__(self, plot_dir: Union[str, Path], dpi: int = 150) -> None:
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot(self, survey: ScanSNRSurvey, outfile: Optional[str] = None) -> list[Path]:
        """Plot every polarization of a survey; return the written PNG paths.

        Parameters
        ----------
        survey : ScanSNRSurvey
            The survey to render.
        outfile : str, optional
            Base output path; the polarization label is inserted before ``.png``.
            Defaults to ``<plot_dir>/<project>.snr_matrix.<pol>.png``.

        Returns
        -------
        list of pathlib.Path
        """
        if not survey.scan_numbers or not survey.antennas:
            logger.warning("scan SNR survey is empty; nothing to plot")
            return []
        written = []
        for polarization in survey.polarizations:
            base = (Path(outfile) if outfile
                    else self.plot_dir / f"{survey.project_code}.snr_matrix.png")
            target = base.with_suffix(f".{polarization}.png")
            written.append(self._plot_one(survey, polarization, target))
        return written

    def _plot_one(self, survey: ScanSNRSurvey, polarization: str, outfile: Path) -> Path:
        """Render one polarization's SNR matrix to ``outfile``."""
        matrix = np.array(survey.matrix(polarization), dtype=float)
        n_scans, n_antennas = matrix.shape
        # constrained layout (not tight_layout): it reserves room for the colorbar and the
        # right-hand source axis instead of letting them overlap the matrix.
        figure, axis = plt.subplots(figsize=(max(5.0, 0.5 * n_antennas + 3.5),
                                             max(3.5, 0.16 * n_scans + 2.0)),
                                    layout="constrained")
        colormap = matplotlib.colormaps[SNR_COLORMAP].with_extremes(bad=NO_DATA_COLOR)
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", origin="upper",
                            cmap=colormap, norm=self._norm(matrix),
                            extent=(-0.5, n_antennas - 0.5, n_scans - 0.5, -0.5))

        style_axis(axis, grid=False)
        axis.set_xticks(range(n_antennas), survey.antennas, rotation=90, fontsize=8)
        self._label_scans(axis, survey, n_scans)
        axis.set_xlabel("antenna", fontsize=9)
        axis.set_ylabel("scan", fontsize=9)

        colorbar = figure.colorbar(image, ax=axis, pad=0.01, fraction=0.05, aspect=40)
        colorbar.set_label("fringe SNR", fontsize=9)
        # Added after the colorbar so constrained layout accounts for its labels too.
        self._mark_source_blocks(axis, survey)
        refant = f", refant {survey.refant}" if survey.refant else ""
        axis.set_title(f"{survey.project_code} — fringe SNR, {polarization}\n"
                       f"central {survey.channel_fraction:.0%} of channels{refant}; "
                       "grey = no solution", fontsize=10, loc="left")
        figure.savefig(outfile, dpi=self.dpi)
        plt.close(figure)
        logger.info("scan SNR matrix written: {}", outfile)
        return outfile

    def _norm(self, matrix: np.ndarray) -> LogNorm:
        """Return a log color normalization spanning the finite, positive SNRs."""
        finite = matrix[np.isfinite(matrix) & (matrix > 0)]
        if not finite.size:
            return LogNorm(vmin=1.0, vmax=10.0)
        vmin = max(1.0, float(np.nanpercentile(finite, 1)))
        vmax = max(vmin * 10.0, float(np.nanpercentile(finite, 99)))
        return LogNorm(vmin=vmin, vmax=vmax)

    def _label_scans(self, axis, survey: ScanSNRSurvey, n_scans: int) -> None:
        """Label the y axis with scan numbers, thinned so the ticks stay readable."""
        step = max(1, n_scans // 30)
        positions = list(range(0, n_scans, step))
        axis.set_yticks(positions, [str(survey.scan_numbers[i]) for i in positions], fontsize=7)

    def _mark_source_blocks(self, axis, survey: ScanSNRSurvey) -> None:
        """Separate consecutive runs of the same source and name them on the right edge."""
        sources = survey.scan_sources
        if not sources or len(sources) != len(survey.scan_numbers):
            return
        right = axis.secondary_yaxis("right")
        right.spines["right"].set_visible(False)
        right.tick_params(length=0)
        start = 0
        ticks, labels = [], []
        for index in range(1, len(sources) + 1):
            if index == len(sources) or sources[index] != sources[start]:
                ticks.append((start + index - 1) / 2.0)
                labels.append(sources[start])
                if index < len(sources):
                    axis.axhline(index - 0.5, color="white", lw=1.2)
                start = index
        right.set_yticks(ticks, labels, fontsize=7)


class CalTablePlotter:
    """Render CASA calibration tables as per-antenna PNG diagnostic plots.

    Parameters
    ----------
    plot_dir : str or pathlib.Path
        Directory where the PNG files are written (created if needed).
    dpi : int
        Output resolution.
    """

    def __init__(self, plot_dir: Union[str, Path], dpi: int = 150,
                 frequencies: Optional[dict] = None) -> None:
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        #: {subband: [channel frequencies in GHz]}, so bandpass plots get a physical axis.
        self.frequencies = frequencies or {}

    # -- table access --
    def _read(self, caltable: Union[str, Path]) -> dict:
        """Read a caltable into a dict of row-major arrays plus antenna names/VisCal."""
        import casatools
        table = casatools.table()
        if not table.open(str(caltable)):
            raise ValueError(f"could not open calibration table {caltable}")
        try:
            data = {
                "viscal": str(table.getkeyword("VisCal")),
                "time": np.asarray(table.getcol("TIME")),
                "antenna": np.asarray(table.getcol("ANTENNA1")),
                "spw": np.asarray(table.getcol("SPECTRAL_WINDOW_ID")),
                "flag": np.asarray(table.getcol("FLAG")).T,  # -> (nrows, nchan, npar)
            }
            param_col = "FPARAM" if "FPARAM" in table.colnames() else "CPARAM"
            data["param"] = np.asarray(table.getcol(param_col)).T  # -> (nrows, nchan, npar)
        finally:
            table.close()
        # Some caltable kinds (e.g. EPowerCurve gain curves) store their parameters
        # with no channel axis, so getcol returns 2D (npar, nrows) -> (nrows, npar)
        # after .T. Normalise every param/flag array to 3D (nrows, nchan, npar) so
        # the plotters can index a channel axis uniformly (they all use [..., 0, ...]).
        if data["param"].ndim == 2:
            data["param"] = data["param"][:, np.newaxis, :]
        if data["flag"].ndim == 2:
            data["flag"] = data["flag"][:, np.newaxis, :]
        table.open(str(Path(caltable) / "ANTENNA"))
        try:
            data["antenna_names"] = [str(n) for n in table.getcol("NAME")]
        finally:
            table.close()
        return data

    # -- public API --
    def plot(self, caltable: Union[str, Path], cal_type: str = "", outfile: Optional[str] = None) -> list[Path]:
        """Plot a calibration table, dispatching on its VisCal kind.

        Parameters
        ----------
        caltable : str or pathlib.Path
            Path to the CASA calibration table.
        cal_type : str
            Short type tag used in output names (falls back to the table stem).
        outfile : str, optional
            Explicit output PNG path (single-figure kinds only).

        Returns
        -------
        list of pathlib.Path
            The written PNG file(s).
        """
        data = self._read(caltable)
        stem = cal_type or Path(caltable).stem
        base = Path(outfile) if outfile else self.plot_dir / f"{Path(caltable).name}.png"
        viscal = data["viscal"].upper()
        if "TSYS" in viscal:
            return [self._plot_tsys(data, base, stem)]
        if "POWERCURVE" in viscal:
            return [self._plot_gain_curve(data, base, stem)]
        if "FRINGE" in viscal:
            return self._plot_fringe(data, base, stem)
        return self._plot_gains(data, base, stem)

    # -- figure builders --
    def _antenna_figure(self, antennas: np.ndarray, names: list[str], sharey: bool = False):
        """Create the per-antenna subplot grid; returns (fig, {antenna_id: axis})."""
        present = sorted(int(a) for a in np.unique(antennas))
        nrows, ncols = subplot_grid(len(present))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows),
                                 sharex=True, sharey=sharey, squeeze=False)
        axis_of = {}
        for i, antenna_id in enumerate(present):
            axis = axes[i // ncols][i % ncols]
            name = names[antenna_id] if antenna_id < len(names) else f"#{antenna_id}"
            axis.set_title(name, fontsize=10, fontweight="bold", loc="left")
            style_axis(axis)
            axis_of[antenna_id] = axis
        for j in range(len(present), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        return fig, axis_of

    def _legend_handles(self, n_spw: int, n_pol: int) -> list[Line2D]:
        """Figure-level legend for subband-colored plots (polarization = linestyle)."""
        handles = [Line2D([], [], color=SUBBAND_COLORS[s % len(SUBBAND_COLORS)], lw=2,
                          label=f"subband {s}") for s in range(min(n_spw, len(SUBBAND_COLORS)))]
        if n_pol > 1:
            handles += [Line2D([], [], color="0.3", lw=1.5, ls=POL_LINESTYLES[p % len(POL_LINESTYLES)],
                               label=POL_LABELS[p % len(POL_LABELS)]) for p in range(n_pol)]
        return handles

    def _pol_legend_handles(self, n_pol: int) -> list[Line2D]:
        """Figure-level legend for polarization-colored plots (gain curve, fringe)."""
        return [Line2D([], [], color=POL_COLORS[p % len(POL_COLORS)], lw=2,
                       label=POL_LABELS[p % len(POL_LABELS)]) for p in range(n_pol)]

    def _finish(self, fig, handles: list[Line2D], title: str, outfile: Path) -> Path:
        """Attach legend/title, save the figure, and close it."""
        fig.legend(handles=handles, loc="upper right", ncols=min(len(handles), 6),
                   frameon=False, fontsize=8)
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(outfile, dpi=self.dpi)
        plt.close(fig)
        logger.info("caltable plot written: {}", outfile)
        return outfile

    def _plot_tsys(self, data: dict, outfile: Path, stem: str) -> Path:
        """Tsys vs time: one subplot per antenna, color per subband, linestyle per pol."""
        n_pol = data["param"].shape[2]
        fig, axis_of = self._antenna_figure(data["antenna"], data["antenna_names"])
        times = np.array([mjdsec2datetime(t) for t in data["time"]])
        for antenna_id, axis in axis_of.items():
            for spw in np.unique(data["spw"]):
                rows = (data["antenna"] == antenna_id) & (data["spw"] == spw)
                order = np.argsort(data["time"][rows])
                for pol in range(n_pol):
                    values = data["param"][rows, 0, pol][order]
                    flags = data["flag"][rows, 0, pol][order]
                    values = np.where(flags, np.nan, values)
                    axis.plot(times[rows][order], values, ls="none", ms=2, marker=".",
                              color=POL_COLORS[pol % len(POL_COLORS)])
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axis.set_ylabel("Tsys (K)", fontsize=8)
        handles = self._pol_legend_handles(n_pol)
        return self._finish(fig, handles, f"{stem} — system temperature", outfile)

    def _plot_gain_curve(self, data: dict, outfile: Path, stem: str) -> Path:
        """Gain-curve polynomials vs elevation: one subplot per antenna, color per pol (R/L)."""
        n_par = data["param"].shape[2]
        n_pol = 2 if n_par % 2 == 0 else 1
        n_coeff = n_par // n_pol
        elevation = np.linspace(1.0, 90.0, 90)
        fig, axis_of = self._antenna_figure(data["antenna"], data["antenna_names"])
        for antenna_id, axis in axis_of.items():
            for spw in np.unique(data["spw"]):
                rows = np.where((data["antenna"] == antenna_id) & (data["spw"] == spw))[0]
                if not len(rows):
                    continue
                for pol in range(n_pol):
                    coeffs = data["param"][rows[0], 0, pol * n_coeff:(pol + 1) * n_coeff]
                    # First pol drawn wider underneath: both stay visible when curves coincide.
                    axis.plot(elevation, evaluate_gain_curve(coeffs, elevation), ls="none",
                              marker=".", ms=3.0 if pol == 0 else 1.5,
                              color=POL_COLORS[pol % len(POL_COLORS)])
            # No label_outer(): panels do not share the y scale, ticks must stay visible.
            axis.set_ylabel("gain", fontsize=8)
        fig.supxlabel("elevation (deg)", fontsize=9)
        return self._finish(fig, self._pol_legend_handles(n_pol), f"{stem} — gain curve", outfile)

    def _plot_fringe(self, data: dict, outfile: Path, stem: str) -> list[Path]:
        """Fringe solutions in phase, delay, rate order.

        SBD is a frequency-dependent instrumental solution, so its x axis is the
        subband centre frequency (or subband index when frequencies are unavailable).
        MBD remains a time-series solution.
        """
        n_pol = data["param"].shape[2] // len(_FRINGE_PARAMS)
        is_sbd = str(stem).lower().startswith("sbd")
        times = np.array([mjdsec2datetime(t) for t in data["time"]])
        written = []
        for param_index, (tag, ylabel) in enumerate(_FRINGE_PARAMS[:3]):  # skip disp by default
            fig, axis_of = self._antenna_figure(data["antenna"], data["antenna_names"])
            for antenna_id, axis in axis_of.items():
                for spw in np.unique(data["spw"]):
                    rows = (data["antenna"] == antenna_id) & (data["spw"] == spw)
                    order = np.argsort(data["time"][rows])
                    for pol in range(n_pol):
                        column = pol * len(_FRINGE_PARAMS) + param_index
                        values = data["param"][rows, 0, column][order]
                        flags = data["flag"][rows, 0, column][order]
                        values = np.where(flags, np.nan, values)
                        if tag == "phase":
                            values = np.degrees(values)
                        elif tag == "rate":
                            values = values * 1e12  # stored as s/s
                        # First pol drawn bigger underneath: both stay visible when values coincide.
                        if is_sbd:
                            frequencies = np.asarray(self.frequencies.get(int(spw), []), dtype=float)
                            x_value = float(np.nanmean(frequencies)) if frequencies.size else float(spw)
                            x = np.full(values.shape, x_value)
                        else:
                            x = times[rows][order]
                        axis.plot(x, values, lw=0.0, marker=".",
                                  ms=4.0 if pol == 0 else 2.0,
                                  color=POL_COLORS[pol % len(POL_COLORS)])
                if is_sbd:
                    axis.set_xlabel("frequency (GHz)" if self.frequencies else "subband", fontsize=8)
                else:
                    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                axis.set_ylabel(ylabel, fontsize=8)
            out = outfile.with_suffix(f".{tag}.png")
            written.append(self._finish(fig, self._pol_legend_handles(n_pol), f"{stem} — {tag}", out))
        return written

    def _freq_axis(self, spw: int, count: int) -> np.ndarray:
        """Sky frequency of a subband's channels, falling back to channel index."""
        freqs = (self.frequencies or {}).get(spw)
        if freqs is not None and len(freqs) >= count:
            return np.asarray(freqs[:count], dtype=float)
        return np.arange(count, dtype=float)

    def _plot_gains(self, data: dict, outfile: Path, stem: str) -> list[Path]:
        """Complex gains (G/B Jones): bandpass tables (nchan > 1) get one combined PNG with an
        amplitude/phase pair per antenna vs frequency; single-channel (scalar) gains get the
        amplitude vs time only."""
        if data["param"].shape[1] > 1:
            return [self._plot_bandpass(data, outfile, stem)]
        n_pol = data["param"].shape[2]
        times = np.array([mjdsec2datetime(t) for t in data["time"]])
        fig, axis_of = self._antenna_figure(data["antenna"], data["antenna_names"])
        for antenna_id, axis in axis_of.items():
            for spw in np.unique(data["spw"]):
                rows = np.where((data["antenna"] == antenna_id) & (data["spw"] == spw))[0]
                order = np.argsort(data["time"][rows])
                for pol in range(n_pol):
                    values = np.abs(data["param"][rows, 0, pol]).astype(float)
                    values[data["flag"][rows, 0, pol]] = np.nan
                    axis.plot(times[rows][order], values[order], ls="none", ms=2, marker=".",
                              color=POL_COLORS[pol % len(POL_COLORS)])
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            axis.set_ylabel("amplitude", fontsize=8)
        out = outfile.with_suffix(".amp.png")
        return [self._finish(fig, self._pol_legend_handles(n_pol), f"{stem} — amplitude", out)]

    def _plot_bandpass(self, data: dict, outfile: Path, stem: str) -> Path:
        """Bandpass amplitude (top) and phase (bottom) vs frequency, one gapless pair per antenna."""
        n_pol, n_chan = data["param"].shape[2], data["param"].shape[1]
        present = sorted(int(a) for a in np.unique(data["antenna"]))
        names = data["antenna_names"]
        nrows, ncols = subplot_grid(len(present))
        fig = plt.figure(figsize=(3.6 * ncols, 3.0 * nrows), layout="constrained")
        outer = fig.add_gridspec(nrows, ncols)
        for index, antenna_id in enumerate(present):
            amp_axis, phase_axis = _stacked_pair(fig, outer[index // ncols, index % ncols])
            for spw in np.unique(data["spw"]):
                rows = np.where((data["antenna"] == antenna_id) & (data["spw"] == spw))[0]
                x = self._freq_axis(int(spw), n_chan)
                for pol in range(n_pol):
                    for row in rows:
                        values = data["param"][row, :, pol].astype(complex)
                        values[data["flag"][row, :, pol]] = np.nan
                        amp_axis.plot(x, np.abs(values), ls="none", marker=".", ms=2,
                                      color=POL_COLORS[pol % len(POL_COLORS)])
                        phase_axis.plot(x, np.degrees(np.angle(values)), ls="none", marker=".", ms=2,
                                        color=POL_COLORS[pol % len(POL_COLORS)])
            _style_stacked_pair(amp_axis, phase_axis, fontsize=7)
            name = names[antenna_id] if antenna_id < len(names) else f"#{antenna_id}"
            amp_axis.set_title(name, fontsize=10, fontweight="bold", loc="left")
            if index // ncols == nrows - 1:
                phase_axis.set_xlabel("frequency (GHz)" if self.frequencies else "channel", fontsize=8)
        fig.legend(handles=self._pol_legend_handles(n_pol), loc="outside upper right",
                   ncols=min(n_pol, 6), frameon=False, fontsize=8)
        fig.suptitle(f"{stem} — bandpass amplitude and phase", x=0.01, ha="left", fontsize=12,
                     fontweight="bold")
        fig.savefig(outfile, dpi=self.dpi)
        plt.close(fig)
        logger.info("caltable plot written: {}", outfile)
        return outfile
