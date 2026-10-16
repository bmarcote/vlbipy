"""The standard diagnostic plots drawn with jiveplot's ``jplotter`` (JIVE's ``standardplots``).

jplotter reads the measurement set with python-casacore in large chunks and
draws with PGPLOT: on GL048 (22 M rows, 65 GB) a scan's auto- or
cross-correlation spectra take 3-4 s and the amplitude/phase of every baseline
to one antenna over the whole 22-h track 8 s, against 20-130 s for the
matplotlib figures they replace. Opening the set and indexing its scans
(``indexr``) costs about 30 s per session, so figures are batched into a few
sessions that run side by side in worker processes.

A figure is a dict:

``name``
    Output path without extension. Multi-page figures get ``.page2.png``,
    ``.page3.png``, ... after the first page (jplotter draws into a dot-free
    scratch name, see :func:`_scratch_stem`).
``commands``
    jplotter commands that set up the plot (selection, plot type, layout);
    ``refile``/``pl`` are added here.
``plot_type``
    The jplotter plot type: figures of one session share it (switching the
    plot type inside a session does not always refresh the data).
``scans``
    Whether the figure selects scans; figures that do and figures that do not
    are drawn in different sessions (see :func:`_scan_select`).
``readflags``
    False to plot flagged data as well (the autocorrelations are flagged once
    the a-priori amplitude calibration has used them).

Pages are PNG (``giza``), ``PAGE_INCHES`` wide.
"""
from __future__ import annotations

import os
import sys
import time as _time
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

logger = get_logger()

#: Page width (inches) and height/width ratio of the PNG pages.
PAGE_INCHES, PAGE_ASPECT = 16.0, 0.6
#: Most figures drawn in one jplotter session (each session opens and indexes the MS once).
FIGURES_PER_SESSION = 12
#: Colour per polarization product, as in JIVE's standardplots.
POL_COLOURS = "ckey p[rr]=2 p[ll]=3 p[rl]=4 p[lr]=5 p[none]=1"


def available() -> bool:
    """Whether jiveplot (and its PGPLOT binding) can be imported."""
    try:
        import ppgplot  # noqa: F401
        from jiveplot import command, jplotter  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means "use the matplotlib figures"
        return False
    return True


# ---------------------------------------------------------------------------
# Figure recipes (after jiveplot's bin/standardplots)
# ---------------------------------------------------------------------------
def _scan_select(scans: Optional[list[int]]) -> list[str]:
    """The scan selection of a figure: none at all for the whole track.

    jplotter's way of selecting every scan (``scan start to end``) becomes one time range per scan in the
    query, which takes minutes on a long track (505 s for GL048's 1386 scans) against seconds without a scan
    selection. Figures with and without one never share a session (see :func:`draw`), so no selection is ever
    left over from a previous figure.
    """
    return ["scan " + " ".join(str(int(s)) for s in scans)] if scans else []


def cross_spectrum(name: str, *, refant: str, scans: Optional[list[int]] = None, source: str = "",
                   all_pols: bool = False) -> dict:
    """Amplitude and phase against channel of every baseline to ``refant``, vector-averaged in time."""
    return {"name": name, "plot_type": "anpchan", "readflags": True, "commands": [
        f"bl {refant}* -auto", "fq *;ch none" if all_pols else "fq */p;ch none", "avt vector;avc none",
        "time none", f"src {source}" if source else "src none", *_scan_select(scans), "pt anpchan", "y local",
        "new all false bl true sb false", "multi true", "sort bl", POL_COLOURS, "nxy 2 4"], "scans": bool(scans)}


def auto_spectrum(name: str, *, scans: Optional[list[int]] = None, source: str = "") -> dict:
    """Autocorrelation spectrum of every antenna (flagged data included), scalar-averaged in time."""
    return {"name": name, "plot_type": "ampchan", "readflags": False, "commands": [
        "bl auto", "fq */p;ch none", "avt scalar;avc none", "time none",
        f"src {source}" if source else "src none", *_scan_select(scans), "pt ampchan", "y local",
        "new all false bl true sb false", "multi true", "sort bl", POL_COLOURS, "nxy 2 4"], "scans": bool(scans)}


def amp_phase_time(name: str, *, refant: str, subband: int, source: str = "", by_source: bool = True) -> dict:
    """Amplitude and phase against time of every baseline to ``refant`` in one subband (central 80% of it)."""
    return {"name": name, "plot_type": "anptime", "readflags": True, "commands": [
        f"bl {refant}* -auto", f"fq {int(subband)}/p;ch 0.1*last:0.9*last", "avt none;avc vector", "time none",
        f"src {source}" if source else "src none", "pt anptime", "y local",
        "new all f bl t", "sort bl", "ckey src" if by_source else POL_COLOURS, "ptsz 2", "nxy 1 4"], "scans": False}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def _install_page_size(width: float, aspect: float) -> None:
    """Make every PNG device jplotter opens ``width`` inches wide (giza's default is 800 x 600 pixels)."""
    import ppgplot
    if getattr(ppgplot.pgopen, "_vlbipy", False):
        return
    original = ppgplot.pgopen

    def pgopen(device):
        ident = original(device)
        if str(device).lower().endswith("/png"):
            ppgplot.pgpap(width, aspect)
        return ident

    pgopen._vlbipy = True
    ppgplot.pgopen = pgopen


def _scratch_stem(stem: Path, index: int) -> Path:
    """A dot-free name to draw into: giza cuts a multi-page file name at its first dot (``a.b.png`` gives
    ``a_0000.png``, ``a_0001.png``), so two figures named ``<code>.raw...`` would overwrite each other."""
    return stem.parent / f"jplotter{os.getpid()}x{index}"


def _pages(scratch: Path, stem: Path) -> list[Path]:
    """Move giza's pages (``scratch.png`` or ``scratch_0000.png``, ...) to ``stem.png``, ``stem.page2.png``, ..."""
    single = scratch.with_name(scratch.name + ".png")
    pages = sorted(scratch.parent.glob(f"{scratch.name}_[0-9][0-9][0-9][0-9].png")) or (
        [single] if single.is_file() else [])
    written = []
    for index, page in enumerate(pages):
        target = stem.with_name(stem.name + ".png") if index == 0 else stem.with_name(f"{stem.name}.page{index + 1}.png")
        page.replace(target)
        written.append(target)
    return written


def session_job(*, ms: str, column: str, readflags: bool, figures: list[dict], log: str,
                width: float = PAGE_INCHES, aspect: float = PAGE_ASPECT) -> list[dict]:
    """Worker job: one jplotter session drawing ``figures``; returns per figure its files, seconds and error."""
    log_handle = open(log, "a", buffering=1)
    saved = (os.dup(1), os.dup(2))
    os.dup2(log_handle.fileno(), 1)
    os.dup2(log_handle.fileno(), 2)
    sys.stdout, sys.stderr = log_handle, log_handle
    results: list[dict] = []
    try:
        _install_page_size(width, aspect)
        from jiveplot import command, jplotter
        marks: list[float] = []

        def script():
            options = f" column={column}" if column and column.upper() != "DATA" else ""
            yield f"ms {ms} spw_order=by_id readflags={'t' if readflags else 'f'}{options}"
            yield "indexr"
            marks.append(_time.time())
            for index, figure in enumerate(figures):
                stem = Path(figure["name"])
                for stale in [stem.with_name(stem.name + ".png"), *stem.parent.glob(f"{stem.name}.page*.png")]:
                    stale.unlink(missing_ok=True)
                yield from figure["commands"]
                yield f"refile {_scratch_stem(stem, index)}.png/png"
                yield "pl"
                # jplotter asks for the next command only once this figure is drawn: put its pages in place
                # now, so an interrupted session keeps what it has done. giza writes a page when the next one
                # starts or the device closes (here: when the next figure opens its file), so a figure whose
                # last page is not on disk yet is collected later.
                marks.append(_time.time())
                seconds[index] = marks[-1] - marks[-2]
                collect()

        def collect() -> None:
            for index in [i for i in range(len(figures)) if i in seconds and i not in done]:
                figure = figures[index]
                files = _pages(_scratch_stem(Path(figure["name"]), index), Path(figure["name"]))
                if files:
                    done[index] = {"name": figure["name"], "files": [str(f) for f in files],
                                   "seconds": seconds[index], "error": ""}

        done: dict[int, dict] = {}
        seconds: dict[int, float] = {}
        try:
            jplotter.run_plotter(command.scripted(script()), debug=True)
            error = ""
        except BaseException as exc:  # noqa: BLE001 - jplotter may sys.exit; report instead of dying
            error = f"{type(exc).__name__}: {exc}"
        for index, figure in enumerate(figures):
            if index not in done:
                files = _pages(_scratch_stem(Path(figure["name"]), index), Path(figure["name"]))
                reason = ("jplotter produced no plot (no data selected?)" if index in seconds
                          else error or "jplotter stopped before this figure")
                done[index] = {"name": figure["name"], "files": [str(f) for f in files],
                               "seconds": seconds.get(index), "error": "" if files else reason}
            results.append(done[index])
    finally:
        sys.stdout.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_handle.close()
    return results


def draw(ms, figures: list[dict], *, column: str = "DATA", log_dir=None, workers: Optional[int] = None) -> dict:
    """Draw ``figures`` from ``ms`` in parallel jplotter sessions.

    Returns
    -------
    dict
        ``files`` (all PNG pages written), ``failed`` (``{figure name: reason}``) and ``seconds``.
    """
    from .solvers.workers import run_jobs
    started = _time.perf_counter()
    groups: dict[tuple, list[dict]] = {}
    for figure in figures:
        groups.setdefault((figure["readflags"], figure["plot_type"], figure.get("scans", False)), []).append(figure)
    log_dir = Path(log_dir or Path(ms).parent / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    jobs = []
    for (readflags, plot_type, _), members in groups.items():
        for start in range(0, len(members), FIGURES_PER_SESSION):
            jobs.append({"ms": str(ms), "column": column, "readflags": readflags,
                         "figures": members[start:start + FIGURES_PER_SESSION],
                         "log": str(log_dir / f"jplotter-{stamp}-{plot_type}-{len(jobs)}.log")})
    results = [r for session in run_jobs("vlbipy.jplot:session_job", jobs, workers) for r in session]
    files = [f for r in results for f in r["files"]]
    failed = {r["name"]: r["error"] for r in results if r["error"]}
    seconds = _time.perf_counter() - started
    logger.info("jplotter: {} figure(s), {} page(s) in {} session(s), {:.1f}s{}", len(figures), len(files),
                len(jobs), seconds, f"; {len(failed)} without a plot" if failed else "")
    for name, reason in failed.items():
        logger.info("jplotter: no plot for {} ({})", Path(name).name, reason)
    return {"files": files, "failed": failed, "seconds": seconds}
