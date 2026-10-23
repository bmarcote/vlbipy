"""Generate the Jupyter notebook of a reduction (``<work_dir>/<code>.ipynb``).

The notebook is a readable, re-runnable record of the pipeline that can be
inspected **without executing any cell**: one section per step that has run
(what it did, when, what it produced, the plots it left on disk and the vlbipy
call that re-runs it), then the calibrated data (calibrator checks, uv
coverage, radplots, spectra, light curves, images), then the calibration
tables and the data as **interactive plots with flagging**, and a final
section to resume the pipeline from any step on the edited data.

Static plots
    Every plot is shown by a markdown cell, so Jupyter draws it on open with no
    kernel. By default the PNG files are **embedded** as attachments of those
    cells (``embed=True``): the notebook is self-contained and can be moved or
    sent on its own, at the cost of its size (the plots, base64-encoded) and of
    showing the files as they were when it was written. With ``embed=False``
    (``vlbipy notebook --link``) the plots are linked by their path relative to
    the notebook (``plots/...``, ``images/...``, ``selfcal/...``) instead: a
    small notebook that always shows the files on disk but has to stay next to
    those directories. Plots are found by globbing (:data:`STEP_PLOTS`,
    :data:`SOURCE_STEP_PLOTS` and the tables of ``obs.gaintables``); steps that
    are repeated overwrite their plots, so a section shows the latest version.

Interactive plots
    Visibilities open in difmapy's own pyqtgraph windows (``set_mode("window")``
    hooks the Qt loop into the kernel), whose flagging keys and undo are already
    there; ``obs.save_flags()`` writes the flags into the per-source split and
    ``obs.flag.from_split(source)`` carries them to the parent measurement set.
    Without a display (a remote kernel) the cells fall back to
    :mod:`vlbipy.interactive`'s plotly widgets. Calibration tables always use
    the plotly flagger, which flags solutions in the table itself.

The notebook is regenerated after the import step and at the end of every
run (``Observation.report``), so it always describes the products on disk.
Code cells only *read* state when executed; nothing runs on generation.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .dashboard import FINAL_GROUPS, find_plots, human_title
from .logging_utils import get_logger
from .observation import STEP_ORDER

logger = get_logger()

#: Human description and the API call that re-runs each pipeline step.
STEP_INFO: dict[str, tuple[str, str, str]] = {
    "import_data": ("Import", "FITS-IDI (with the ANTAB Tsys/gain curves appended) imported into a measurement "
                    "set and partitioned into a Multi-MS; metadata (antennas, scans, subband participation) read "
                    "and `summary.md` written.", "obs.import_data(force=FORCE)"),
    "flag_from_file": ("A-priori flags", "The observer/correlator flag file (`.uvflg` -> `.flag`) applied with "
                       "`flagdata(mode='list')`.", "obs.flag.apriori(force=FORCE)"),
    "flag_autocorr": ("Autocorrelations", "Autocorrelations flagged (`flagdata(autocorr=True)`).",
                      "obs.flag.apriori(force=FORCE)"),
    "a_priori": ("A-priori amplitude calibration", "System temperatures (`gencal caltype='tsys'`, de-spiked) and "
                 "gain curves (`caltype='gc'`) from the SYSCAL / GAIN_CURVE subtables.",
                 "obs.calibrate.a_priori(force=FORCE)"),
    "plot_raw": ("Raw-data diagnostics", "The standard diagnostic set on the uncalibrated data: scan x antenna "
                 "fringe SNR, per-scan autocorrelation and cross-correlation spectra on the fringe finders, "
                 "amplitude/phase vs frequency and vs time on baselines to the reference antenna, amplitude/phase "
                 "vs uv distance and per-baseline corner plots of every calibrator.",
                 "obs.plot.diagnostics(column='data', label='raw', force=FORCE)"),
    "flag_quack": ("Quack", "Settling ramp at scan starts, measured per antenna on the phase calibrator and "
                   "flagged where present.", "obs.flag.quack(force=FORCE)"),
    "flag_tfcrop": ("RFI (tfcrop)", "`flagdata(mode='tfcrop')` on the calibrators' raw data (time cutoff 4, "
                    "frequency cutoff 3, flags not extended).", "obs.flag.initial(force=FORCE)"),
    "scan_snr": ("Fringe-SNR survey", "One fringe fit per calibrator scan over the central channels; ranks antennas "
                 "and scans and drives the instrumental-calibration selection.", "obs.calibrate.scan_snr(force=FORCE)"),
    "initial_calibration": ("Single-band delay (SBD)", "Instrumental delay from **one** scan (the central <= 2 min), "
                            "on the antennas detected there, rates zero: one time-constant solution per antenna and "
                            "subband. Chained through a shared antenna when no scan detects every antenna.",
                            "obs.calibrate.initial_calibration(force=FORCE)"),
    "fringefit": ("Multi-band delay (MBD)", "Global fringe fit on the calibrators: subbands combined, rate fitted, "
                  "plus the dispersive (ionospheric) delay below 8 GHz.", "obs.calibrate.fringefit(force=FORCE)"),
    "bandpass": ("Bandpass", "Complex bandpass on the SBD scan, normalised, through the delays.",
                 "obs.calibrate.bandpass(force=FORCE)"),
    "initial_calibration_sbd2": ("SBD, second round", "The single-band delay re-solved through the bandpass.",
                                 "obs.calibrate.initial_calibration(force=FORCE, suffix='sbd2')"),
    "fringefit_mbd2": ("MBD, second round", "The multi-band delay re-solved through the bandpass.",
                       "obs.calibrate.fringefit(force=FORCE, suffix='mbd2')"),
    "flag_edges": ("Subband edges", "Edge channels measured per antenna from the bandpass roll-off and flagged.",
                   "obs.flag.edges(force=FORCE)"),
    "apply": ("Apply", "Every table of the chain applied to every field (`applycal`, cal library, "
              "`calflagstrict`: data without a solution are flagged).", "obs.calibrate.apply(force=True)"),
    "flag_aoflagger": ("RFI (AOFlagger)", "AOFlagger on the calibrators only (targets are protected).",
                       "obs.flag.aoflagger(force=FORCE)"),
    "flag_outliers": ("Outliers", "Per-baseline amplitude spikes (> 5 sigma from a running median) flagged on the "
                      "calibrated calibrators.", "obs.flag.outliers(force=FORCE)"),
    "second_pass": ("Second pass", "SBD/MBD/bandpass/SBD2/MBD2 re-derived from the a-priori tables on the flagged "
                    "data. Their plots, in the sections of those steps above, are overwritten by this pass.",
                    "obs.calibrate.second_pass(force=FORCE)"),
    "scalar_bandpass": ("Scalar bandpass", "One amplitude per antenna and subband (`gaincal calmode='a'`) on the "
                        "phase calibrator, levelling the subbands.", "obs.calibrate.scalar_bandpass(force=FORCE)"),
    "reweight": ("Re-weighting", "`statwt` on the calibrated data: weights from the measured scatter.",
                 "obs.calibrate.reweight(force=FORCE)"),
    "flag_outliers_reweighted": ("Outliers, after re-weighting", "Outlier flagging repeated with the new weights.",
                                 "obs.flag.outliers(force=FORCE, step='flag_outliers_reweighted')"),
    "third_pass": ("Third pass", "The whole chain re-derived once more on the re-weighted, re-flagged data. Its "
                   "plots, in the sections of those steps above, are overwritten by this pass.",
                   "obs.calibrate.second_pass(force=FORCE, step='third_pass')"),
    "flag_edges_final": ("Subband edges, final bandpass", "Edge channels and bandpass gaps measured again on the "
                         "bandpass of the last pass (the one applied) and flagged on top of the first trim.",
                         "obs.flag.edges(force=FORCE, step='flag_edges_final')"),
    "split": ("Split & export", "Per-source calibrated measurement sets (one channel per subband) and UVFITS in "
              "`calibrated_data/`.", "obs.export.per_source(force=FORCE)"),
    "final_plots": ("Final-data plots", "uv coverage, radplots, spectra and light curves of every source on the "
                    "calibrated data; they are shown in the *Calibrated data* section below.",
                    "obs.plot.final_data()"),
}

#: Plots each step leaves on disk, besides those of the calibration tables it made (see :func:`table_plots`).
#: Globs are relative to ``<work_dir>``; ``{code}`` is the project code. Every entry is a tuple of alternatives
#: and only the first one that matches anything is used: the SBD-stage plots exist under the same name in
#: ``plots/raw`` (first pass, before any apply) and ``plots/calibrated`` (later passes), and the later one wins.
#: The fringe tables are listed by name because the solved table leaves the chain when its time-smoothed copy
#: (``<code>.mbd_smooth``) is the one applied: both are plotted, and only the applied one is in ``obs.gaintables``.
STEP_PLOTS: dict[str, list[tuple[str, ...]]] = {
    "plot_raw": [("plots/raw/{code}.snr_matrix.*.png",), ("plots/raw/{code}.raw_scan*.autocorr.png",),
                 ("plots/raw/{code}.raw_scan*.spectrum.png",), ("plots/raw/{code}.raw_*.spectrum.png",),
                 ("plots/raw/{code}.raw_*.timeseries.png",), ("plots/raw/{code}.raw.radplot.*.png",),
                 ("plots/raw/{code}.raw_*.corner.*.png",)],
    "scan_snr": [("plots/raw/{code}.snr_matrix.*.png",)],
    "initial_calibration": [("plots/calibrated/{code}.sbd_stage*_scan*.timeseries.png",
                             "plots/raw/{code}.sbd_stage*_scan*.timeseries.png")],
    "initial_calibration_sbd2": [("plots/calibrated/{code}.sbd2_stage*_scan*.timeseries.png",
                                  "plots/raw/{code}.sbd2_stage*_scan*.timeseries.png")],
    "fringefit": [("plots/caltables/{code}.mbd.*.png",)],
    "fringefit_mbd2": [("plots/caltables/{code}.mbd2.*.png",)],
    "flag_edges": [("plots/caltables/{code}.bandpass_profile.png",)],
    "flag_edges_final": [("plots/caltables/{code}.bandpass_profile.png",)],
}

#: Same as :data:`STEP_PLOTS` for the per-source steps, keyed by step-name prefix. ``{src}`` is the source name
#: and ``{safe}`` the same name as the image grid spells it (``plotting.plot_image_grid``: ``,`` and ``/`` -> ``_``).
#: Self-cal also previews the image grid, but imaging overwrites that file: it is shown by ``clean_`` and in the
#: *Calibrated data* section only.
SOURCE_STEP_PLOTS: dict[str, list[tuple[str, ...]]] = {
    "selfcal_": [("selfcal/{code}.{src}.amp.png",)],
    "clean_": [("images/{code}.images.{safe}.png",)],
}

#: Calibrated diagnostics of the calibrators (``plot.diagnostics(column="corrected")``), which no step records.
CALIBRATOR_PLOTS: list[tuple[str, ...]] = [
    ("plots/calibrated/{code}.calibrated_*.spectrum.png",), ("plots/calibrated/{code}.calibrated_*.timeseries.png",),
    ("plots/calibrated/{code}.calibrated.radplot.*.png",), ("plots/calibrated/{code}.calibrated.subband_phases.png",),
    ("plots/calibrated/{code}.calibrated_*.corner.*.png",),
]

#: CLEAN image grids per source, relative to ``<work_dir>``.
IMAGE_PLOTS: list[tuple[str, ...]] = [("images/{code}.images.*.png",)]

#: A markdown image, as :func:`image_markdown` and :func:`embedded_markdown` write it; group 1 is the URL
#: (a percent-encoded relative path, or ``attachment:<key>`` for an embedded plot).
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")

#: Characters an attachment key must not hold: renderers percent-encode the URL before looking the key up.
UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]")

#: Size of an embedded notebook above which :func:`write_notebook` suggests linking the plots instead.
EMBED_WARN_MB = 50


def _step_info(step: str) -> tuple[str, str, str]:
    """Title, description and re-run call for any step name, including ``selfcal_<src>``/``clean_<src>``."""
    if step in STEP_INFO:
        return STEP_INFO[step]
    if step.startswith("selfcal_"):
        src = step[len("selfcal_"):]
        return (f"Self-calibration of {src} (difmapy)",
                "One circular Gaussian modelfit, phase-only self-cal from long to short solution intervals "
                "(each step kept only if the fit improves), Bayesian station amplitude gains; the gain tables are "
                "added to the apply chain.", f"obs.selfcal.calibrator('{src}', force=FORCE)")
    if step.startswith("clean_"):
        src = step[len("clean_"):]
        return (f"Imaging of {src}", "difmapy CLEAN at Briggs robust -2, 0 and +2; FITS in `images/` and a PNG grid.",
                f"obs.clean(target='{src}')")
    return (step.replace("_", " "), "", f"# no direct call recorded for {step}")


def _ordered_steps(state: dict) -> list[str]:
    """Steps in pipeline order: STEP_ORDER first, then self-cal, imaging and the rest in recorded order."""
    known = [s for s in STEP_ORDER if s in state]
    rest = [s for s in state if s not in STEP_ORDER]
    selfcal = [s for s in rest if s.startswith("selfcal_")]
    clean = [s for s in rest if s.startswith("clean_")]
    other = [s for s in rest if s not in selfcal and s not in clean]
    return known + selfcal + clean + other


def _md(text: str, attachments: Optional[dict] = None) -> dict:
    """A markdown cell; ``attachments`` (``{key: {"image/png": base64}}``) are stored in it when given."""
    import nbformat.v4 as nbf
    cell = nbf.new_markdown_cell(text)
    if attachments:
        cell["attachments"] = attachments
    return cell


def _code(text: str, stdout: str = "") -> dict:
    """A code cell; ``stdout`` is stored as its output so it reads without running the cell."""
    import nbformat.v4 as nbf
    cell = nbf.new_code_cell(text)
    if stdout:
        cell.outputs = [nbf.new_output("stream", name="stdout", text=stdout)]
    return cell


def _step_source(step: str) -> str:
    """Source name of a ``selfcal_<src>``/``clean_<src>`` step ("" for every other step)."""
    return next((step[len(prefix):] for prefix in SOURCE_STEP_PLOTS if step.startswith(prefix)), "")


def match_plots(work: Path, rules: list[tuple[str, ...]], code: str, source: str = "") -> list[Path]:
    """Existing files for ``rules`` (see :data:`STEP_PLOTS`), in rule order, without repeats.

    Parameters
    ----------
    work : Path
        The work directory the globs are relative to.
    rules : list of tuple of str
        Each tuple holds alternative globs; the first that matches anything is used.
    code, source : str
        Substituted for ``{code}``, ``{src}`` and ``{safe}`` (glob-escaped).
    """
    names = {"code": glob.escape(code), "src": glob.escape(source),
             "safe": glob.escape(source.replace(",", "_").replace("/", "_"))}
    found: list[Path] = []
    for alternatives in rules:
        for pattern in alternatives:
            hits = find_plots(work, [pattern.format(**names)])
            if hits:
                found += hits
                break
    return list(dict.fromkeys(found))


def table_plots(work: Path, table) -> list[Path]:
    """Plots of one calibration table, found from the table's name on disk.

    ``CalTablePlotter`` writes ``plots/caltables/<table name>.png`` or ``<table name>.<parameter>.png``; the
    match is on the exact name so ``<code>.sbd`` never picks up the plots of ``<code>.sbd2``. difmapy's
    amplitude self-cal table has its figure beside it (``selfcal/<code>.<src>.amp.G`` -> ``.amp.png``); that
    one is looked up inside ``work`` so a work directory that was moved still resolves.
    """
    path = Path(table.path)
    if not path.name:
        return []
    name = glob.escape(path.name)
    found = find_plots(work, [f"plots/caltables/{name}.png", f"plots/caltables/{name}.*.png"])
    beside = work / path.parent.name / path.with_suffix(".png").name
    return found + ([beside] if beside.is_file() else [])


def step_plots(obs, step: str) -> list[Path]:
    """Every plot on disk that ``step`` produced: its declared globs, then the plots of the tables it made."""
    work = Path(obs.work_dir)
    rules = STEP_PLOTS.get(step) or next((r for p, r in SOURCE_STEP_PLOTS.items() if step.startswith(p)), [])
    found = match_plots(work, rules, obs.project_code, _step_source(step))
    for table in obs.gaintables:
        if table.step == step:
            found += table_plots(work, table)
    return list(dict.fromkeys(found))


def relative_link(path: Path, notebook_dir: Path) -> str:
    """Path of ``path`` relative to the notebook's directory, with forward slashes (not URL-encoded)."""
    return os.path.relpath(path, notebook_dir).replace("\\", "/")


def image_markdown(path: Path, code: str, notebook_dir: Path) -> str:
    """Markdown for one linked plot: a caption naming the file, then the image.

    The URL is percent-encoded (source names contain ``+``); the caption shows the path as it is on disk.
    """
    rel = relative_link(path, notebook_dir)
    title = human_title(path, code)
    return f"**{title}** — `{rel}`\n\n![{title}]({quote(rel, safe='/')})"


def attachment_key(index: int, path: Path) -> str:
    """Attachment name of the ``index``-th plot of a cell.

    The index keeps keys unique when two directories hold the same file name; every character outside
    ``[A-Za-z0-9._-]`` becomes ``_`` because renderers percent-encode the URL before looking the key up.
    """
    return f"{index}_{UNSAFE_KEY_CHARS.sub('_', path.name)}"


def embedded_markdown(path: Path, code: str, work: Path, key: str) -> str:
    """Markdown for one embedded plot: a caption naming the file in the work directory, then the attachment."""
    title = human_title(path, code)
    return f"**{title}** — `{relative_link(path, work)}`\n\n![{title}](attachment:{key})"


def png_base64(path: Path) -> str:
    """The bytes of ``path`` base64-encoded, unmodified ("" with a warning when it is unreadable or empty)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("notebook: {} cannot be read ({}); linked instead of embedded", path, exc)
        return ""
    if not data:
        logger.warning("notebook: {} is empty; linked instead of embedded", path)
        return ""
    return base64.b64encode(data).decode("ascii")


def plots_cell(paths: list[Path], code: str, work: Path, notebook_dir: Path, embed: bool, heading: str = "") -> dict:
    """One markdown cell showing every plot of ``paths``, one under the other.

    Parameters
    ----------
    paths : list of Path
        The plot files, in display order.
    code : str
        Project code (stripped from the captions).
    work, notebook_dir : Path
        Work directory (captions of embedded plots are relative to it) and the notebook's directory (links
        are relative to it).
    embed : bool
        True: each PNG is stored in the cell as an attachment. False: each is linked by relative path. A file
        that cannot be embedded (unreadable or empty) is linked.
    heading : str
        Markdown placed before the plots (a section title and its description).
    """
    parts = [heading] if heading else []
    attachments = {}
    for index, path in enumerate(paths):
        data = png_base64(path) if embed else ""
        if not data:
            parts.append(image_markdown(path, code, notebook_dir))
            continue
        key = attachment_key(index, path)
        attachments[key] = {"image/png": data}
        parts.append(embedded_markdown(path, code, work, key))
    return _md("\n\n".join(parts), attachments)


def count_plots(cells: list) -> int:
    """Number of images (embedded or linked) shown by the markdown cells."""
    return sum(len(IMAGE_LINK.findall(c["source"])) for c in cells if c["cell_type"] == "markdown")


def _number(value, spec: str) -> str:
    """``value`` formatted with ``spec``, or "n/a" when it is not a number."""
    return format(value, spec) if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def selfcal_report_markdown(work: Path, code: str, source: str) -> str:
    """The difmapy self-cal report of ``source`` as markdown ("" when there is none).

    Reads ``selfcal/<code>.<source>.json``: the phase ladder, the reduced chi-squared of the model and one
    row per self-cal round (solution interval, chi-squared before/after, change, decision).
    """
    path = work / "selfcal" / f"{code}.{source}.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        rounds = [dict(r) for r in report.get("rounds") or []]
        ladder = ", ".join(str(s) for s in report.get("ladder") or [])
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("notebook: no self-cal report for {} ({})", source, exc)
        return ""
    lines = [f"Phase ladder: {ladder or 'none'}; model reduced chi² {_number(report.get('model_rchisq'), '.2f')}."]
    if rounds:
        lines += ["", "| solint | chi² before | chi² after | change | decision |", "|---|---|---|---|---|"]
    for r in rounds:
        improvement = r.get("improvement")
        change = _number(-improvement, "+.1%") if isinstance(improvement, (int, float)) else "n/a"
        lines.append(f"| {r.get('solint', '')} | {_number(r.get('chisq_before'), ',.1f')} | "
                     f"{_number(r.get('chisq_after'), ',.1f')} | {change} | "
                     f"{'accepted' if r.get('accepted') else 'rejected'} |")
    return "\n".join(lines)


def _summary_text(obs) -> str:
    """``obs.summary()`` for the stored output of the setup cell ("" when it cannot be built).

    ``summary()`` only describes metadata already in memory, so the ``metadata`` property is read first: it
    loads the cached ``.metadata.json`` when the data were not imported in this process (``vlbipy notebook``
    on a work directory whose measurement set is gone).
    """
    try:
        if obs.metadata is None:
            logger.debug("notebook: no metadata for {}; the stored summary lists the configuration only",
                         obs.project_code)
        return obs.summary() + "\n"
    except Exception as exc:  # noqa: BLE001 - the summary is a convenience; the notebook is written without it
        logger.debug("notebook: no summary for {} ({})", obs.project_code, exc)
        return ""


def _plots_note(obs, notebook_dir: Path, embed: bool) -> str:
    """The sentences of the intro that say where the plots shown in the notebook come from."""
    repeated = "steps that are repeated (second and third pass) overwrite their plots, so a section shows the latest"
    if embed:
        return ("the plots are embedded copies of the files in `plots/`, `images/` and `selfcal/` as they were "
                "when the notebook was written, so it can be moved or sent on its own; it is regenerated at the "
                f"end of every run and by `vlbipy notebook`. Mind that {repeated} version.")
    summary_file = Path(obs.work_dir) / "summary.md"
    summary_url = quote(relative_link(summary_file, notebook_dir), safe="/")
    summary_link = f" The observation is listed in [summary.md]({summary_url})." if summary_file.is_file() else ""
    return ("the plots are the files on disk, linked by relative path, so keep the notebook next to `plots/`, "
            f"`images/` and `selfcal/`. Each plot is the file as it is now; {repeated} version.{summary_link}")


def _intro_cells(obs, notebook_dir: Path, embed: bool) -> list:
    """Title, how the notebook works, and the setup cell with the observation summary as stored output."""
    code = obs.project_code
    work = Path(obs.work_dir)
    intro = _md(f"# {code} — vlbipy reduction notebook\n\n"
                f"Generated by vlbipy from the products in `{work}`. Every section below is a step the "
                "pipeline ran, with the plots it produced and the call that re-runs it, followed by the calibrated "
                f"data. **Nothing has to be executed to read it**: {_plots_note(obs, notebook_dir, embed)}\n\n"
                "The code cells are only needed to redo work or to flag interactively. `FORCE = False` (next cell) "
                "makes completed steps no-ops, so **Run All is safe**; set it to `True` in a cell, or use the "
                "*Resume* section at the end, to redo work after flagging.\n\n"
                "Products: `caltables/` (calibration tables), `plots/`, `calibrated_data/` (per-source "
                "measurement sets + UVFITS), `selfcal/` (difmapy reports and gain tables), `images/`, "
                "`html/index.html` (dashboard).")
    setup = _code(f"""import os
from pathlib import Path
from vlbipy import VLBIObs

FORCE = False                                   # True: re-run every step cell below
campaign = VLBIObs.load({str(work)!r})          # rebuilt from .project.json
obs = campaign[{code!r}]
obs.import_data(force=False)                   # metadata (cheap: reads the cache)
print(obs.summary())""", stdout=_summary_text(obs))
    return [intro, setup]


def _step_cells(obs, state: dict, steps: list[str], notebook_dir: Path, embed: bool) -> list:
    """The table of steps, then per step: description, its plots (when it has any) and the re-run cell."""
    code = obs.project_code
    work = Path(obs.work_dir)
    cells = [_md("## Pipeline steps\n\n"
                 "| step | status | when | outputs |\n|---|---|---|---|\n" + "\n".join(
                     f"| `{s}` | {state[s].get('status', '')} | {state[s].get('timestamp', '')} | "
                     f"{', '.join(map(str, state[s].get('outputs', [])))[:80]} |" for s in steps))]
    for step in steps:
        title, blurb, call = _step_info(step)
        entry = state[step]
        detail = f"**{entry.get('status', '')}** at {entry.get('timestamp', '')}"
        if entry.get("error"):
            detail += f" — error: `{entry['error']}`"
        outputs = entry.get("outputs") or []
        out_txt = ("\n\nOutputs: " + ", ".join(f"`{o}`" for o in outputs[:12])) if outputs else ""
        report = selfcal_report_markdown(work, code, _step_source(step)) if step.startswith("selfcal_") else ""
        cells.append(_md(f"### {title}\n\n{blurb}\n\n{detail}{out_txt}" + (f"\n\n{report}" if report else "")))
        plots = step_plots(obs, step)
        if plots:
            cells.append(plots_cell(plots, code, work, notebook_dir, embed))
        cells.append(_code(call))
    return cells


def _calibrated_data_cells(obs, notebook_dir: Path, embed: bool) -> list:
    """The calibrated data as static plots: calibrator checks, the final-data groups and the images."""
    code = obs.project_code
    work = Path(obs.work_dir)
    groups = [("Calibrators after calibration", "Amplitude and phase vs frequency, time and uv distance, the "
               "subband phase-jump check and the per-baseline corner plots of every calibrator, with the whole "
               "chain applied: phases flat near zero and stable amplitudes mean the calibration worked.",
               match_plots(work, CALIBRATOR_PLOTS, code))]
    for title, blurb, patterns, fallback in FINAL_GROUPS:
        groups.append((title, blurb, find_plots(work / "plots", patterns) or find_plots(work / "plots", fallback)))
    groups.append(("Images", "CLEAN images per source at Briggs robust -2, 0, +2 (difmapy).",
                   match_plots(work, IMAGE_PLOTS, code)))
    cells = [_md("## Calibrated data\n\nThe data after the whole calibration chain, as written by the pipeline: "
                 "first the calibrators, then every source (uv coverage, amplitude and phase vs uv distance, "
                 "frequency and time), then the images.")]
    shown = [g for g in groups if g[2]]
    for title, blurb, paths in shown:
        cells.append(plots_cell(paths, code, work, notebook_dir, embed, heading=f"### {title}\n\n{blurb}"))
    if not shown:
        cells.append(_md("No calibrated-data plots exist yet: they are written once the calibration is applied."))
    return cells


def _caltable_cells(obs) -> list:
    """The calibration chain as a table, then one interactive flagger cell per table."""
    cells = [_md("## Calibration tables — interactive flagging\n\n"
                 "Each plot shows one parameter of a table vs time, one trace per antenna (hollow = flagged). "
                 "Use the **box/lasso select** tool, then **Flag selected**: the solutions are flagged in the "
                 "table (`FLAG` column) and `applycal` (`calflagstrict`) will flag the data they covered. "
                 "**Undo** restores the previous flags. Re-run `obs.calibrate.apply(force=True)` (and the steps "
                 "after it) once you are done. The static plots of each table are in the section of the step "
                 "that made it.")]
    if obs.gaintables:
        cells.append(_md("The chain, in apply order:\n\n| cal_type | table | step | field | interp | apply to |\n"
                         "|---|---|---|---|---|---|\n" + "\n".join(
                             f"| `{t.cal_type}` | `{Path(t.path).name}` | `{t.step}` | {t.field} | {t.interp} | "
                             f"{t.apply_to or 'all fields'} |" for t in obs.gaintables)))
    cells.append(_code("from vlbipy.interactive import caltable_flagger\n"
                       "tables = {t.cal_type: t.path for t in obs.gaintables}\n"
                       "print('\\n'.join(f'{k:14s} {v}' for k, v in tables.items()))"))
    for table in obs.gaintables:
        cells.append(_md(f"### `{table.cal_type}` — {Path(table.path).name}"))
        cells.append(_code(f"caltable_flagger(tables[{table.cal_type!r}])"))
    return cells


def _data_cells(obs) -> list:
    """Interactive difmapy cells per source (with the plotly fallback for a kernel without a display)."""
    sources = list(obs.metadata.source_names) if obs.metadata else list(obs.sources.names)
    refant = str(obs.refant).split(",")[0] if obs.metadata else ""
    cells = [_md("## Data — interactive plots\n\n"
                 "The per-source calibrated splits (`calibrated_data/<code>_<source>.ms`, one channel per "
                 "subband) open in **difmapy**. With a display its pyqtgraph windows appear from the kernel "
                 "(`set_mode('window')`), with difmap's flagging keys (drag-select, `Ctrl+Z` undo, `F` "
                 "unflag nearest) — `radplot`, `vplot(nplot, refant)`, `tplot`, `uvplot`, `specplot`, "
                 "`mapplot`. When done, `d.save_flags()` writes the flags into the split and "
                 "`obs.flag.from_split(source)` carries them to the parent measurement set. Without a display "
                 "the plotly fallback below flags with `flagdata` directly."),
             _code(f"""import difmapy
from difmapy.plots.base import set_mode, has_display
set_mode("window" if has_display() else "inline")
REFANT = {refant!r} or obs.refant.split(",")[0]
def split(source):
    return str(Path(obs.work_dir) / "calibrated_data" / f"{{obs.project_code}}_{{source}}.ms")
print("display:", has_display(), "| plots in", ("windows" if has_display() else "inline images"))""")]
    for source in sources:
        cells.append(_md(f"### {source}"))
        cells.append(_code(f"""d = difmapy.load(split({source!r}))
d.radplot()                       # amplitude & phase vs uv radius
d.vplot(3, REFANT)                # amplitude/phase vs time per baseline to REFANT
# d.tplot(); d.uvplot(); d.specplot()"""))
        cells.append(_code(f"""# after flagging in the windows:
# d.save_flags()                              # into the split MS
# obs.flag.from_split({source!r})             # -> flagdata on the parent MS"""))
        cells.append(_code(f"""# plotly fallback (no display): box-select and press "Flag selected"
# from vlbipy.interactive import visibility_flagger
# visibility_flagger(str(obs._backend.ms_path(obs.project_code)), field={source!r}, refant=REFANT, quantity="amp")"""))
    return cells


def _resume_cells() -> list:
    """How to re-run the pipeline from a step after flagging."""
    return [_md("## Resume the pipeline\n\nAfter flagging, re-run from the first step whose inputs changed. "
                "`from_step` invalidates that step and everything after it. Steps, in order:\n\n" +
                ", ".join(f"`{s}`" for s in STEP_ORDER) + "\n\nTypical choices: `apply` after flagging "
                "calibration-table solutions; `second_pass` after flagging data on the calibrators; "
                "`split` to only redo the exports, self-calibration and imaging."),
            _code("# campaign.run(from_step='apply')"),
            _code("# from vlbipy.dashboard import serve_dashboard; serve_dashboard(campaign)   # rebuild + open html/")]


def build_notebook(obs, notebook_dir: Optional[Path] = None,
                   embed: bool = True) -> "nbformat.NotebookNode":  # noqa: F821 - nbformat imported lazily
    """Assemble the notebook for one :class:`~vlbipy.observation.Observation`.

    Parameters
    ----------
    obs : Observation
        The observation; its state, calibration tables and the plots in its work directory are read.
    notebook_dir : Path, optional
        Directory the notebook will live in; plot links are relative to it. Default: ``obs.work_dir``.
        Only matters for linked plots.
    embed : bool
        True (default): the plots are embedded as markdown-cell attachments (a self-contained notebook).
        False: they are linked by relative path (a small notebook that has to stay where it is).

    Returns
    -------
    nbformat.NotebookNode
        The notebook. ``metadata["vlbipy"]`` holds the project, work directory, steps, the number of plots
        shown and whether they are embedded.
    """
    import nbformat.v4 as nbf
    work = Path(obs.work_dir)
    target_dir = Path(notebook_dir) if notebook_dir is not None else work
    state = obs._state.as_dict()
    steps = _ordered_steps(state)
    cells = (_intro_cells(obs, target_dir, embed) + _step_cells(obs, state, steps, target_dir, embed) +
             _calibrated_data_cells(obs, target_dir, embed) + _caltable_cells(obs) + _data_cells(obs) +
             _resume_cells())
    nb = nbf.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["vlbipy"] = {"project": obs.project_code, "work_dir": str(work), "steps": steps,
                             "plots": count_plots(cells), "embedded": bool(embed)}
    return nb


def write_notebook(obs, path: Optional[str] = None, embed: bool = True) -> Path:
    """Write the notebook for ``obs`` (default ``<work_dir>/<code>.ipynb``) and return its path.

    With ``embed=True`` (default) the plots are stored in the notebook, which can then live anywhere. With
    ``embed=False`` they are linked relative to the directory of ``path``, so a notebook written outside the
    work directory still finds the plots as long as both stay where they are.
    """
    import nbformat
    target = Path(path) if path else Path(obs.work_dir) / f"{obs.project_code}.ipynb"
    nb = build_notebook(obs, target.parent, embed)
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(target))
    size_mb = target.stat().st_size / 1e6
    logger.info("notebook written: {} ({} cells, {} plots {}, {:.1f} MB)", target, len(nb.cells),
                nb.metadata["vlbipy"]["plots"], "embedded" if embed else "linked", size_mb)
    if embed and size_mb > EMBED_WARN_MB:
        logger.warning("notebook: {} is {:.0f} MB with the plots embedded; `vlbipy notebook --link` writes a "
                       "small one that links them instead", target, size_mb)
    return target


def write_campaign_notebooks(campaign, embed: bool = True) -> list[Path]:
    """One notebook per observation of a :class:`~vlbipy.vlbiobs.VLBIObs` (``embed``: see :func:`write_notebook`)."""
    return [write_notebook(o, embed=embed) for o in campaign.observations if o._backend.requires_data_files]


def notebook_summary(path) -> dict:
    """Small JSON summary of a written notebook (for tests and the dashboard): cells, steps, plots, embedded."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    info = data.get("metadata", {}).get("vlbipy", {})
    return {"cells": len(data["cells"]), "steps": info.get("steps", []), "plots": info.get("plots", 0),
            "embedded": bool(info.get("embedded", False))}
