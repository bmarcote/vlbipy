"""Generate the interactive Jupyter notebook of a reduction (``<work_dir>/<code>.ipynb``).

The notebook is a readable, re-runnable record of the pipeline: one section per
step that has run (what it did, when, what it produced, and the vlbipy call
that re-runs it), then the calibration tables and the data as **interactive
plots with flagging**, and a final section to resume the pipeline from any
step on the edited data.

Plots
    Visibilities open in difmapy's own pyqtgraph windows (``set_mode("window")``
    hooks the Qt loop into the kernel), whose flagging keys and undo are already
    there; ``obs.save_flags()`` writes the flags into the per-source split and
    ``obs.flag.from_split(source)`` carries them to the parent measurement set.
    Without a display (a remote kernel) the cells fall back to
    :mod:`vlbipy.interactive`'s plotly widgets. Calibration tables always use
    the plotly flagger, which flags solutions in the table itself.

The notebook is regenerated after the import step and at the end of every
run (``Observation.report``), so it always describes the products on disk.
Cells only *read* state when executed; nothing runs on generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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
    "flag_outliers": ("Outliers", "Per-baseline amplitude spikes (> 5 sigma from a running median) flagged on the "
                      "calibrated calibrators.", "obs.flag.outliers(force=FORCE)"),
    "second_pass": ("Second pass", "SBD/MBD/bandpass/SBD2/MBD2 re-derived from the a-priori tables on the flagged "
                    "data.", "obs.calibrate.second_pass(force=FORCE)"),
    "scalar_bandpass": ("Scalar bandpass", "One amplitude per antenna and subband (`gaincal calmode='a'`) on the "
                        "phase calibrator, levelling the subbands.", "obs.calibrate.scalar_bandpass(force=FORCE)"),
    "reweight": ("Re-weighting", "`statwt` on the calibrated data: weights from the measured scatter.",
                 "obs.calibrate.reweight(force=FORCE)"),
    "flag_outliers_reweighted": ("Outliers, after re-weighting", "Outlier flagging repeated with the new weights.",
                                 "obs.flag.outliers(force=FORCE, step='flag_outliers_reweighted')"),
    "third_pass": ("Third pass", "The whole chain re-derived once more on the re-weighted, re-flagged data.",
                   "obs.calibrate.second_pass(force=FORCE, step='third_pass')"),
    "split": ("Split & export", "Per-source calibrated measurement sets (one channel per subband) and UVFITS in "
              "`calibrated_data/`.", "obs.export.per_source(force=FORCE)"),
    "final_plots": ("Final-data plots", "uv coverage, radplots, spectra, light curves, subband phase-jump check.",
                    "obs.plot.final_data()"),
}


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


def _md(text: str) -> dict:
    import nbformat.v4 as nbf
    return nbf.new_markdown_cell(text)


def _code(text: str) -> dict:
    import nbformat.v4 as nbf
    return nbf.new_code_cell(text)


def build_notebook(obs) -> "nbformat.NotebookNode":  # noqa: F821 - nbformat imported lazily
    """Assemble the notebook for one :class:`~vlbipy.observation.Observation`."""
    import nbformat.v4 as nbf
    code = obs.project_code
    work = Path(obs.work_dir)
    state = obs._state.as_dict()
    steps = _ordered_steps(state)
    cells = []

    cells.append(_md(f"# {code} — vlbipy reduction notebook\n\n"
                     f"Generated by vlbipy from the products in `{work}`. Every section below is a step the "
                     "pipeline ran, with the call that re-runs it. `FORCE = False` (next cell) makes completed "
                     "steps no-ops, so **Run All is safe**; set it to `True` in a cell, or use the *Resume* section "
                     "at the end, to redo work after flagging.\n\n"
                     "Products: `caltables/` (calibration tables), `plots/`, `calibrated_data/` (per-source "
                     "measurement sets + UVFITS), `selfcal/` (difmapy reports and gain tables), `images/`, "
                     "`html/index.html` (dashboard)."))
    cells.append(_code(f"""import os
from pathlib import Path
from vlbipy import VLBIObs

FORCE = False                                   # True: re-run every step cell below
campaign = VLBIObs.load({str(work)!r})          # rebuilt from .project.json
obs = campaign[{code!r}]
obs.import_data(force=False)                   # metadata (cheap: reads the cache)
print(obs.summary())"""))

    cells.append(_md("## Pipeline steps\n\n"
                     "| step | status | when | outputs |\n|---|---|---|---|\n" + "\n".join(
                         f"| `{s}` | {state[s].get('status', '')} | {state[s].get('timestamp', '')} | "
                         f"{', '.join(map(str, state[s].get('outputs', [])))[:80]} |" for s in steps)))

    for step in steps:
        title, blurb, call = _step_info(step)
        entry = state[step]
        status = entry.get("status", "")
        detail = f"**{status}** at {entry.get('timestamp', '')}"
        if entry.get("error"):
            detail += f" — error: `{entry['error']}`"
        outputs = entry.get("outputs") or []
        out_txt = ("\n\nOutputs: " + ", ".join(f"`{o}`" for o in outputs[:12])) if outputs else ""
        cells.append(_md(f"### {title}\n\n{blurb}\n\n{detail}{out_txt}"))
        cells.append(_code(call))

    # ---- calibration tables
    cells.append(_md("## Calibration tables — interactive flagging\n\n"
                     "Each plot shows one parameter of a table vs time, one trace per antenna (hollow = flagged). "
                     "Use the **box/lasso select** tool, then **Flag selected**: the solutions are flagged in the "
                     "table (`FLAG` column) and `applycal` (`calflagstrict`) will flag the data they covered. "
                     "**Undo** restores the previous flags. Re-run `obs.calibrate.apply(force=True)` (and the steps "
                     "after it) once you are done."))
    cells.append(_code("from vlbipy.interactive import caltable_flagger\n"
                       "tables = {t.cal_type: t.path for t in obs.gaintables}\n"
                       "print('\\n'.join(f'{k:14s} {v}' for k, v in tables.items()))"))
    for table in obs.gaintables:
        cells.append(_md(f"### `{table.cal_type}` — {Path(table.path).name}"))
        cells.append(_code(f"caltable_flagger(tables[{table.cal_type!r}])"))

    # ---- data
    sources = list(obs.metadata.source_names) if obs.metadata else list(obs.sources.names)
    refant = str(obs.refant).split(",")[0] if obs.metadata else ""
    cells.append(_md("## Data — interactive plots\n\n"
                     "The per-source calibrated splits (`calibrated_data/<code>_<source>.ms`, one channel per "
                     "subband) open in **difmapy**. With a display its pyqtgraph windows appear from the kernel "
                     "(`set_mode('window')`), with difmap's flagging keys (drag-select, `Ctrl+Z` undo, `F` "
                     "unflag nearest) — `radplot`, `vplot(nplot, refant)`, `tplot`, `uvplot`, `specplot`, "
                     "`mapplot`. When done, `d.save_flags()` writes the flags into the split and "
                     "`obs.flag.from_split(source)` carries them to the parent measurement set. Without a display "
                     "the plotly fallback below flags with `flagdata` directly."))
    cells.append(_code(f"""import difmapy
from difmapy.plots.base import set_mode, has_display
set_mode("window" if has_display() else "inline")
REFANT = {refant!r} or obs.refant.split(",")[0]
def split(source):
    return str(Path(obs.work_dir) / "calibrated_data" / f"{{obs.project_code}}_{{source}}.ms")
print("display:", has_display(), "| plots in", ("windows" if has_display() else "inline images"))"""))
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

    # ---- images and self-cal reports
    images = sorted((work / "images").glob(f"{code}.images.*.png")) if (work / "images").is_dir() else []
    if images:
        cells.append(_md("## Images\n\nCLEAN images per source at Briggs robust -2, 0, +2 (difmapy)."))
        cells.append(_code("from IPython.display import Image as _Img, display\n" + "\n".join(
            f"display(_Img(filename={str(p)!r}))" for p in images)))
    reports = sorted((work / "selfcal").glob(f"{code}.*.json")) if (work / "selfcal").is_dir() else []
    reports = [r for r in reports if not r.name.endswith(".amp.json")]
    if reports:
        cells.append(_md("## Self-calibration reports (difmapy)\n\nPhase ladder decisions and the Bayesian "
                         "amplitude gains per calibrator; `selfcal/<code>.<source>.amp.png` is the gscale figure."))
        cells.append(_code("import json\nfrom IPython.display import Image as _Img, display\n" + "\n".join(
            f"rep = json.load(open({str(r)!r})); print(rep['source'], 'ladder', rep['ladder'])\n"
            f"for r_ in rep['rounds']: print('  ', r_['solint'], f\"{{-r_['improvement']:+.1%}}\", "
            f"'accepted' if r_['accepted'] else 'rejected')\n"
            f"display(_Img(filename={str(r.with_name(r.name.replace('.json', '.amp.png')))!r}))"
            for r in reports)))

    # ---- resume
    cells.append(_md("## Resume the pipeline\n\nAfter flagging, re-run from the first step whose inputs changed. "
                     "`from_step` invalidates that step and everything after it. Steps, in order:\n\n" +
                     ", ".join(f"`{s}`" for s in STEP_ORDER) + "\n\nTypical choices: `apply` after flagging "
                     "calibration-table solutions; `second_pass` after flagging data on the calibrators; "
                     "`split` to only redo the exports, self-calibration and imaging."))
    cells.append(_code("# campaign.run(from_step='apply')"))
    cells.append(_code("# from vlbipy.dashboard import serve_dashboard; serve_dashboard(campaign)   # rebuild + open html/"))

    nb = nbf.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["vlbipy"] = {"project": code, "work_dir": str(work), "steps": steps}
    return nb


def write_notebook(obs, path: Optional[str] = None) -> Path:
    """Write the notebook for ``obs`` (default ``<work_dir>/<code>.ipynb``) and return its path."""
    import nbformat
    target = Path(path) if path else Path(obs.work_dir) / f"{obs.project_code}.ipynb"
    nb = build_notebook(obs)
    nbformat.write(nb, str(target))
    logger.info("notebook written: {} ({} cells)", target, len(nb.cells))
    return target


def write_campaign_notebooks(campaign) -> list[Path]:
    """One notebook per observation of a :class:`~vlbipy.vlbiobs.VLBIObs`."""
    return [write_notebook(o) for o in campaign.observations if o._backend.requires_data_files]


def notebook_summary(path) -> dict:
    """Small JSON summary of a written notebook (for tests and the dashboard)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"cells": len(data["cells"]), "steps": data.get("metadata", {}).get("vlbipy", {}).get("steps", [])}
