"""Command-line interface for vlbipy (thin wrapper over :class:`VLBIObs`).

The CLI exposes one subcommand per pipeline stage, each a thin translation of
flags into :class:`~vlbipy.vlbiobs.VLBIObs` namespace calls::

    vlbipy pipeline -p RSM07 --network EVN --target 3C286  # full default pipeline
    vlbipy import -p RSM07 --obsdate 250916                # download (if needed) + import to MS
    vlbipy export -p RSM07 --format dask-ms                # export MS to dask-ms (lazy/distributed)
    vlbipy calibrate bandpass fringefit -p RSM07           # selected cal steps (positional)
    vlbipy flag aoflagger -p RSM07                         # selected flag steps (positional)
    vlbipy image -p RSM07 --target 3C286 --robust 0 2      # imaging
    vlbipy plot caltables uv_coverage -p RSM07             # diagnostic plots (positional kinds)
    vlbipy summary -p RSM07                                # import + summary

``vlbipy -p RSM07 ...`` (no subcommand) behaves as ``vlbipy pipeline ...``;
``run`` is kept as a hidden alias of ``pipeline``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from rich_argparse import RawDescriptionRichHelpFormatter, RichHelpFormatter

from .logging_utils import configure_logging, get_logger
from .models import CalTable
from .registry import get_backend
from .vlbiobs import VLBIObs

logger = get_logger()

#: Calibration sub-steps selectable via ``vlbipy calibrate --steps`` (pipeline order).
CALIBRATE_STEPS: list[str] = ["a_priori", "initial_calibration", "bandpass", "fringefit", "apply"]

#: Flagging sub-steps selectable via ``vlbipy flag --steps`` (pipeline order).
FLAG_STEPS: list[str] = ["autocorr", "edges", "quack", "tfcrop", "aoflagger"]

#: Plot kinds selectable via ``vlbipy plot --kinds``.
PLOT_KINDS: list[str] = ["caltables", "caltable", "tplot", "uv_coverage", "diagnostic", "elevation",
                         "amp_vs_time", "phase_vs_time", "autocorr", "crosscorr", "scan_snr",
                         "scan_diagnostics", "spectrum", "radplot", "timeseries", "corners",
                         "baseline_corner", "subband_phases", "lightcurve", "final_data"]

#: Known subcommand names (used to route the no-subcommand form to ``pipeline``).
COMMANDS: list[str] = ["pipeline", "import", "calibrate", "flag", "image", "plot", "applycal", "summary", "export", "report"]

#: Positional selector per subcommand: attribute name and its valid values.
_SELECTORS: dict[str, tuple[str, list[str]]] = {"calibrate": ("steps", CALIBRATE_STEPS),
                                               "flag": ("steps", FLAG_STEPS),
                                               "plot": ("kinds", PLOT_KINDS)}


def _relocate_selectors(args: argparse.Namespace) -> argparse.Namespace:
    """Move step/kind names that greedy multi-value flags swallowed into the selector.

    ``vlbipy calibrate -p RSM07 bandpass`` parses 'bandpass' into ``-p`` (nargs='+'
    is greedy); any trailing run of valid selector names is moved back where it
    belongs, so selectors work both before and after the common options.
    """
    selector = _SELECTORS.get(args.command)
    if selector is None:
        return args
    attr, valid = selector
    moved: list[str] = []
    for source_attr in ("project", "target"):
        values = getattr(args, source_attr, None) or []
        while len(values) > 1 and values[-1] in valid:
            moved.insert(0, values.pop())
    if moved:
        setattr(args, attr, moved + list(getattr(args, attr, []) or []))
    return args


def _common_parser(*, project_required: bool = True) -> argparse.ArgumentParser:
    """Return a parent parser with the options shared by pipeline subcommands."""
    p = argparse.ArgumentParser(add_help=False, formatter_class=RichHelpFormatter)
    p.add_argument("-p", "--project", nargs="+", required=project_required, help="Project code(s)")
    p.add_argument("-n", "--network", help="VLBI network (e.g. EVN, VLBA, LBA, or a plugin name)")
    p.add_argument("--backend", help="Backend (dummy|casa|aips); default from config (casa)")
    p.add_argument("-t", "--target", nargs="+", help="Target source name(s)")
    p.add_argument("--phasecal", nargs="+", help="Phase calibrator name(s)")
    p.add_argument("--fringe-finder", nargs="+", dest="fringe_finder", help="Fringe finder name(s)")
    p.add_argument("--check-source", nargs="+", dest="check_source", help="Check source name(s)")
    p.add_argument("--refant", help="Preferred reference antenna(s), comma-separated")
    p.add_argument("--config", help="Path to a TOML config file")
    p.add_argument("--force", action="store_true", help="Re-run steps even if already completed")
    p.add_argument("--no-ionos", dest="ionos", action="store_false", default=None,
                   help="Do not solve the dispersive (ionospheric) delay in the fringe fit. "
                        "By default it is solved below 6 GHz, where the ionosphere matters.")
    p.add_argument("--callib", action="store_true", default=None,
                   help="Use CASA cal-library files instead of explicit parallel "
                        "gaintable/gainfield/interp/spwmap/calwt parameters.")
    p.add_argument("--scratch", action="store_true",
                   help="Start over: forget all recorded progress and reset the data "
                        "(clearcal + unflag). Without it the run resumes where it stopped.")
    p.add_argument("--from-step", dest="from_step", metavar="STEP",
                   help="Re-run from this step onward, invalidating it and everything after")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging")
    return p


def _describe_option(action: argparse.Action) -> str:
    """Return a compact usage-like description of one option (with its choices)."""
    name = action.option_strings[-1]
    if action.nargs == 0:
        return name
    if action.choices is not None:
        value = "{" + ",".join(str(c) for c in action.choices) + "}"
    else:
        value = action.metavar or action.dest.upper()
    suffix = " ..." if action.nargs in ("+", "*") else ""
    return f"{name} {value}{suffix}"


def _options_epilog(sub: argparse._SubParsersAction, common: argparse.ArgumentParser) -> str:
    """Build an epilog listing, per subcommand, its specific options (introspected).

    Also lists the common options once, so the main --help shows everything each
    subcommand accepts without having to run '<subcommand> --help'.
    """
    common_flags = {s for action in common._actions for s in action.option_strings}
    lines = ["Options per subcommand (each also accepts the common options below):"]
    described: set[int] = set()
    for name, subparser in sub.choices.items():
        if name not in COMMANDS or id(subparser) in described:
            continue  # skip hidden aliases (e.g. 'run') and duplicates
        described.add(id(subparser))
        parts = []
        for action in subparser._actions:
            if not action.option_strings and action.dest not in ("help", "command"):
                choices = ",".join(str(c) for c in action.choices if c) if action.choices else action.dest
                parts.append(f"[{{{choices}}} ...]")  # positional selector
            elif action.option_strings and "--help" not in action.option_strings \
                    and action.option_strings[-1] not in common_flags:
                parts.append(_describe_option(action))
        detail = ", ".join(parts) or "(common options only)"
        lines.append(f"  {name:<10} {detail}")
    lines.append("")
    lines.append("Common options (all subcommands):")
    common_detail = ", ".join(_describe_option(action) for action in common._actions
                              if action.option_strings)
    lines.append(f"  {common_detail}")
    lines.append("")
    lines.append("For more information, visit: https://vlbipy.readthedocs.io/")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with one subparser per pipeline stage."""
    common = _common_parser()
    direct_plot_common = _common_parser(project_required=False)
    parser = argparse.ArgumentParser(
        prog="vlbipy",
        description="Backend-agnostic VLBI data reduction suite (API-first).",
        formatter_class=RawDescriptionRichHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(COMMANDS) + "}")

    p_pipe = sub.add_parser(
        "pipeline", parents=[common], formatter_class=RichHelpFormatter, aliases=["run"],
        help="Run the full default pipeline end to end (the default subcommand)"
    )
    p_pipe.add_argument("--summary-only", action="store_true", help="Import and print the summary, then stop")
    p_pipe.add_argument("--dashboard", action="store_true", help="Build the HTML dashboard and open it in the browser")

    p_imp = sub.add_parser(
        "import", parents=[common], formatter_class=RichHelpFormatter,
        help="Download the raw data (if not on disk) and import it into a measurement set"
    )
    p_imp.add_argument("--files", nargs="+", metavar="FILE",
                       help="Explicit FITS-IDI file(s) or a glob pattern; skips search/download")
    p_imp.add_argument("--obsdate", help="Observing date (YYMMDD) for the archive download; "
                                         "auto-resolved from the data or the EVN archive if omitted")
    p_imp.add_argument("--username", help="Archive username for proprietary data")
    p_imp.add_argument("--password", help="Archive password for proprietary data")
    p_imp.add_argument("--replace-tsys", action="store_true", dest="replace_tsys",
                       help="Overwrite Tsys/GC tables in the FITS-IDI files even if already present")

    p_cal = sub.add_parser(
        "calibrate", parents=[common], formatter_class=RichHelpFormatter,
        help="Run calibration (all steps or a selection)"
    )
    p_cal.add_argument("steps", nargs="*", default=[], choices=CALIBRATE_STEPS + [[]], metavar="STEP",
                       help=f"Calibration step(s) to run, in order given. Choices: {', '.join(CALIBRATE_STEPS)}. "
                            "Default: the full calibration chain")

    p_flag = sub.add_parser(
        "flag", parents=[common], formatter_class=RichHelpFormatter,
        help="Run flagging (all steps or a selection)"
    )
    p_flag.add_argument("steps", nargs="*", default=[], choices=FLAG_STEPS + [[]], metavar="STEP",
                        help=f"Flag step(s) to run, in order given. Choices: {', '.join(FLAG_STEPS)}. "
                             "Default: the default flag chain (autocorr, edges, quack, aoflagger)")
    p_flag.add_argument("--flagfile", help="Apply flags from an external flag command file")

    p_img = sub.add_parser(
        "image", parents=[common], formatter_class=RichHelpFormatter,
        help="Image target source(s)"
    )
    p_img.add_argument("--robust", nargs="+", type=float, help="Briggs robust value(s); default from config")
    p_img.add_argument("--imager", choices=["wsclean", "tclean"], help="Imager to use; default wsclean")
    p_img.add_argument("--niter", type=int, help="Clean iterations; default from config")

    p_plot = sub.add_parser(
        "plot", parents=[direct_plot_common], formatter_class=RichHelpFormatter,
        help="Produce diagnostic plots"
    )
    p_plot.add_argument("kinds", nargs="*", default=[], choices=PLOT_KINDS + [[]], metavar="KIND",
                        help=f"Plot kind(s) to produce. Choices: {', '.join(PLOT_KINDS)}. "
                             "Default: the standard diagnostic set")
    p_plot.add_argument("--ms", metavar="PATH", help="Plot directly from an existing CASA measurement set")
    p_plot.add_argument("--field", default="", help="CASA field selection for direct-MS plots")
    p_plot.add_argument("--scans", nargs="+", help="Scan numbers (comma- or space-separated)")
    p_plot.add_argument("--column", default="corrected", help="Measurement-set data column")
    p_plot.add_argument("--label", default="", help="Output filename label")
    p_plot.add_argument("--outdir", help="Output directory passed to direct plot operations")
    p_plot.add_argument("--caltable", help="Calibration table path for the direct caltable plot kind")

    p_apply = sub.add_parser("applycal", formatter_class=RichHelpFormatter,
                             help="Apply CASA calibration directly to an existing measurement set")
    p_apply.add_argument("ms", metavar="MS", help="Existing CASA measurement set")
    p_apply.add_argument("caltables", nargs="*", metavar="CALTABLE", help="Calibration table(s) to apply")
    p_apply.add_argument("--callib", metavar="FILE", help="Existing CASA calibration-library file")
    for selection in ("field", "spw", "intent", "timerange", "antenna", "scan", "observation"):
        p_apply.add_argument(f"--{selection}", default="", help=f"CASA {selection} selection")
    p_apply.add_argument("--parang", action=argparse.BooleanOptionalAction, default=True,
                         help="Apply parallactic-angle correction (default: true)")
    p_apply.add_argument("--applymode", help="CASA apply mode")
    p_apply.add_argument("--flagbackup", action=argparse.BooleanOptionalAction, default=None,
                         help="Back up flags before applying calibration")
    p_apply.add_argument("--gainfield", nargs="+", help="Per-table gain field(s); one broadcasts")
    p_apply.add_argument("--interp", nargs="+", help="Per-table interpolation mode(s); one broadcasts")
    p_apply.add_argument("--spwmap", nargs="+", help="Per-table comma-separated SPW mappings; one broadcasts")
    p_apply.add_argument("--calwt", nargs="+", help="Per-table true/false weight calibration; one broadcasts")
    p_apply.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging")

    sub.add_parser(
        "summary", parents=[common], formatter_class=RichHelpFormatter,
        help="Import the data and print an observation summary"
    )

    p_exp = sub.add_parser(
        "export", parents=[common], formatter_class=RichHelpFormatter,
        help="Export the imported data to another format (dask-ms zarr store, UVFITS)"
    )
    p_exp.add_argument("--format", choices=["daskms", "dask-ms", "uvfits"], default="daskms",
                       help="Output format; default daskms (lazy zarr store, distributed-ready)")
    p_exp.add_argument("--output", help="Output path; default <work_dir>/<code>.zarr")

    p_rep = sub.add_parser(
        "report", parents=[common], formatter_class=RichHelpFormatter,
        help="Build the HTML dashboard (<work_dir>/html/index.html) from the plots and state on disk"
    )
    p_rep.add_argument("--open", action="store_true", help="Open the dashboard in the default browser")
    p_nb = sub.add_parser(
        "notebook", parents=[common], formatter_class=RichHelpFormatter,
        help="Write the interactive Jupyter notebook (<work_dir>/<code>.ipynb) describing the reduction so far"
    )
    p_nb.add_argument("--open", action="store_true", help="Launch JupyterLab on the notebook")

    parser.epilog = _options_epilog(sub, common)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate direct CASA command combinations and filesystem inputs."""
    if args.command == "plot":
        if bool(args.project) == bool(args.ms):
            parser.error("plot requires exactly one of -p/--project or --ms")
        if args.ms and not Path(args.ms).is_dir():
            parser.error(f"measurement set not found: {args.ms}")
        if args.ms and not args.kinds:
            parser.error("plot --ms requires at least one KIND")
        return
    if args.command != "applycal":
        return
    if not args.caltables and not args.callib:
        parser.error("applycal requires at least one CALTABLE or --callib FILE")
    if args.caltables and args.callib:
        parser.error("CALTABLE arguments and --callib are mutually exclusive")
    if not Path(args.ms).is_dir():
        parser.error(f"measurement set not found: {args.ms}")
    paths = [args.callib] if args.callib else args.caltables
    for path in paths:
        if not Path(path).exists():
            parser.error(f"calibration path not found: {path}")
    for name in ("gainfield", "interp", "spwmap", "calwt"):
        values = getattr(args, name)
        if values and len(values) not in (1, len(args.caltables)):
            parser.error(f"--{name} accepts one value or exactly {len(args.caltables)} values")
    for value in args.calwt or []:
        if value.strip().lower() not in ("true", "false"):
            parser.error(f"--calwt values must be true or false, got {value!r}")
    for value in args.spwmap or []:
        try:
            [int(part) for part in value.split(",") if part.strip()]
        except ValueError:
            parser.error(f"--spwmap values must be comma-separated integers, got {value!r}")


def _broadcast(values: Optional[list[str]], count: int, default):
    """Return one parsed CLI value per calibration table."""
    if not values:
        return [default for _ in range(count)]
    return values * count if len(values) == 1 else values


def _parse_bool(value: str) -> bool:
    """Parse a strict command-line boolean value."""
    normalized = value.strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"expected true or false, got {value!r}")
    return normalized == "true"


def _direct_backend(ms: str):
    """Create a CASA backend and register an arbitrary existing measurement set."""
    path = Path(ms).resolve()
    code = path.stem
    backend = get_backend("casa", work_dir=str(path.parent))
    backend.data.adopt_ms(code, str(path))
    return backend, code


def _build_obs(args: argparse.Namespace) -> VLBIObs:
    """Construct a VLBIObs from the parsed common arguments (plus [import] overrides if given)."""
    refant = [a.strip() for a in args.refant.split(",")] if args.refant else None
    import_cfg = {key: getattr(args, key) for key in ("obsdate", "username", "password", "replace_tsys")
                  if getattr(args, key, None)}
    overrides = {"import": import_cfg} if import_cfg else {}
    if args.ionos is False:
        # --no-ionos: never solve the dispersive delay, whatever the frequency.
        overrides.setdefault("calibration", {})["ionos"] = False
    if args.callib:
        # --callib opts into the CASA cal-library path; absent leaves config/default.
        overrides.setdefault("calibration", {})["callib"] = True
    return VLBIObs(args.project, network=args.network, backend=args.backend, target=args.target,
                   phasecal=args.phasecal, fringe_finder=args.fringe_finder,
                   check_source=getattr(args, "check_source", None), refant=refant,
                   config=args.config, **overrides)


def _cmd_pipeline(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Full pipeline (or --summary-only / --dashboard)."""
    if args.dashboard:
        from .dashboard import serve_dashboard
        obs.import_data(force=False)
        for observation in obs.observations:
            observation.flag.statistics()
        serve_dashboard(obs)
        return 0
    if args.summary_only:
        obs.import_data(force=args.force)
        print(obs.summary())
        return 0
    obs.run(force=args.force, scratch=args.scratch,
            from_step=getattr(args, 'from_step', '') or '')
    return 0


def _cmd_import(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Import: download the raw data if missing, import to MS if missing, print the summary."""
    files = args.files
    if files and len(files) == 1:
        files = files[0]  # single argument may be a glob pattern; let the namespace expand it
    obs.import_data(force=args.force, files=files)
    print(obs.summary())
    return 0


def _cmd_calibrate(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Calibration: the full chain, or only the steps given via --steps."""
    obs.import_data(force=False)
    if args.steps:
        for step in args.steps:
            logger.info("calibrate: running step {}", step)
            getattr(obs.calibrate, step)(force=args.force)
    else:
        obs.calibrate(force=args.force)
    return 0


def _cmd_flag(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Flagging: the default chain, only the steps given via --steps, and/or a flag file."""
    obs.import_data(force=False)
    if args.flagfile:
        logger.info("flag: applying flag file {}", args.flagfile)
        obs.flag.from_file(args.flagfile, force=args.force)
    if args.steps:
        for step in args.steps:
            logger.info("flag: running step {}", step)
            getattr(obs.flag, step)(force=args.force)
    elif not args.flagfile:
        obs.flag(force=args.force)
    return 0


def _cmd_image(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Imaging: clean each requested target (or all configured targets)."""
    obs.import_data(force=False)
    robust = args.robust if args.robust else obs.config.get("imaging", {}).get("robust", [0])
    targets = args.target or [s.name for s in obs.sources.targets]
    kwargs = {}
    if args.imager:
        kwargs["imager"] = args.imager
    if args.niter is not None:
        kwargs["niter"] = args.niter
    for tgt in targets:
        logger.info("image: cleaning {} (robust={})", tgt, robust)
        obs.clean(target=tgt, robust=robust, **kwargs)
    return 0


def _cmd_applycal(obs: Optional[VLBIObs], args: argparse.Namespace) -> int:
    """Apply explicit calibration tables or an existing CASA cal-library to an arbitrary MS."""
    backend, code = _direct_backend(args.ms)
    task_kwargs = {name: getattr(args, name) for name in
                   ("spw", "intent", "timerange", "antenna", "scan", "observation", "applymode", "flagbackup")
                   if getattr(args, name) not in (None, "")}
    if args.callib:
        backend.calibrate.apply_callib(code, str(Path(args.callib).resolve()), field=args.field,
                                       parang=args.parang, **task_kwargs)
        return 0
    count = len(args.caltables)
    gainfields = _broadcast(args.gainfield, count, "")
    interps = _broadcast(args.interp, count, "linear")
    spwmaps = _broadcast(args.spwmap, count, "")
    calwts = _broadcast(args.calwt, count, "true")
    tables = [CalTable(Path(path).suffix.lstrip(".") or Path(path).name, path=str(Path(path).resolve()),
                       gainfield=gainfields[index], interp=interps[index],
                       spwmap=[int(value) for value in spwmaps[index].split(",") if value.strip()],
                       calwt=_parse_bool(calwts[index]))
              for index, path in enumerate(args.caltables)]
    backend.calibrate.apply(code, args.field, tables, parang=args.parang, **task_kwargs)
    return 0


def _parse_scans(values: Optional[list[str]]) -> Optional[list[int]]:
    """Parse comma- and whitespace-separated scan numbers."""
    return ([int(scan) for value in values for scan in value.split(",") if scan] if values else None)


def _cmd_direct_plot(args: argparse.Namespace) -> int:
    """Dispatch plot kinds directly to CASA plot operations for an arbitrary MS."""
    backend, code = _direct_backend(args.ms)
    common = {"field": args.field, "scans": _parse_scans(args.scans), "refant": args.refant or "",
              "column": args.column, "label": args.label}
    if args.outdir:
        common["outdir"] = str(Path(args.outdir).resolve())
    methods = {"spectrum": "spectrum", "autocorr": "autocorr", "radplot": "radplot",
               "lightcurve": "lightcurve", "timeseries": "timeseries", "subband_phases": "subband_phases",
               "baseline_corner": "baseline_corner", "corners": "baseline_corner", "scan_snr": "scan_snr"}
    results = []
    for kind in args.kinds:
        if kind in ("uv_coverage", "diagnostic"):
            results.append(backend.plot.diagnostic(code, "uv_coverage", **common))
        elif kind in ("caltable", "caltables"):
            if not args.caltable:
                raise ValueError(f"direct {kind} plotting requires --caltable PATH")
            results.append(backend.plot.caltable(code, args.caltable))
        elif kind in methods:
            kwargs = dict(common)
            if kind == "lightcurve":
                kwargs["fields"] = [args.field] if args.field else None
            elif kind == "subband_phases":
                kwargs["fields"] = [args.field] if args.field else []
            results.append(getattr(backend.plot, methods[kind])(code, **kwargs))
        else:
            raise ValueError(f"plot kind {kind!r} has no direct-MS implementation")
    for result in results:
        for path in (result if isinstance(result, list) else [result]):
            print(path)
    return 0


def _cmd_plot(obs: Optional[VLBIObs], args: argparse.Namespace) -> int:
    """Plot through VLBIObs, or dispatch directly when --ms is supplied."""
    if args.ms:
        return _cmd_direct_plot(args)
    obs.import_data(force=False)
    kinds_results = ([getattr(obs.plot, kind)() for kind in args.kinds] if args.kinds
                     else [obs.plot()])
    for result in kinds_results:
        for path in (result if isinstance(result, list) else [result]):
            print(path)
    return 0


def _cmd_summary(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Import the data and print the observation summary."""
    obs.import_data(force=args.force)
    print(obs.summary())
    return 0


def _cmd_export(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Export the imported data: dask-ms zarr store (default) or per-source UVFITS."""
    from pathlib import Path

    obs.import_data(force=False)  # ensure the data is imported first
    if args.format == "uvfits":
        for observation in obs.observations:
            print(observation.export.uvfits())
        return 0

    # dask-ms zarr store. The dask-ms backend produced it at import time already;
    # for the CASA backend, convert its MS here.
    from .backends.dask_ms import ms_to_daskms
    for observation in obs.observations:
        backend = observation._backend
        if hasattr(backend, "store_path"):
            print(backend.store_path(observation.project_code))
            continue
        if not hasattr(backend, "ms_path"):
            logger.error("the {} backend has no measurement set to convert; "
                         "use --backend casa or dask-ms", backend.kind)
            return 1
        store = Path(args.output) if args.output else Path(observation.work_dir) / \
            f"{observation.project_code}.zarr"
        if store.is_dir():
            logger.info("dask-ms store {} already exists", store)
        else:
            ms_to_daskms(backend.ms_path(observation.project_code), store)
        print(store)
    return 0


def _cmd_notebook(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Write the reduction notebook(s); optionally open JupyterLab on the first."""
    try:
        obs.import_data(force=False)
    except Exception as exc:  # noqa: BLE001 - cached metadata is enough for the notebook
        logger.warning("notebook: could not reload the data ({}); using the cached metadata", exc)
    paths = [p for p in obs.notebook() if p]
    for path in paths:
        print(path)
    if args.open and paths:
        import subprocess
        subprocess.Popen([sys.executable, "-m", "jupyterlab", paths[0]])
    return 0 if paths else 1


def _cmd_report(obs: VLBIObs, args: argparse.Namespace) -> int:
    """Build the static HTML dashboard from the products already on disk."""
    from .dashboard import build_dashboard, serve_dashboard
    try:
        obs.import_data(force=False)
    except Exception as exc:  # noqa: BLE001 - a reportable project may have lost its MS
        logger.warning("report: could not reload the data ({}); using the cached metadata", exc)
    for observation in obs.observations:
        try:
            observation.flag.statistics()
        except Exception:  # noqa: BLE001 - without the MS there are no flags to count
            logger.debug("report: flag statistics unavailable for {}", observation.project_code)
    index = serve_dashboard(obs) if args.open else build_dashboard(obs)
    print(index)
    return 0


#: Dispatch table mapping subcommand name -> handler(obs, args) -> exit code.
#: "run" stays as a hidden alias of "pipeline".
_HANDLERS = {"pipeline": _cmd_pipeline, "run": _cmd_pipeline, "import": _cmd_import,
             "calibrate": _cmd_calibrate, "flag": _cmd_flag,
             "image": _cmd_image, "plot": _cmd_plot, "applycal": _cmd_applycal,
             "summary": _cmd_summary, "export": _cmd_export, "report": _cmd_report,
             "notebook": _cmd_notebook}


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 = success).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default form: `vlbipy -p CODE ...` (no subcommand) is treated as `vlbipy pipeline ...`.
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "pipeline")

    parser = build_parser()
    args = _relocate_selectors(parser.parse_args(argv))
    _validate_args(parser, args)
    configure_logging("DEBUG" if args.verbose else "INFO")

    obs = None if args.command == "applycal" or (args.command == "plot" and args.ms) else _build_obs(args)
    try:
        return _HANDLERS[args.command](obs, args)
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("{} failed: {}", args.command, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
