"""Draw the plots of a diagnostic set several at a time, in worker processes.

Almost all the time of a diagnostic plot goes into reading the measurement set (minutes for a source
with many scans), and every plot reads its own data, so the plots of one set do not depend on each
other. They run as jobs in the worker processes of :mod:`vlbipy.solvers.workers`: processes, not threads,
because casatools is not thread-safe and holds the GIL while it reads, and matplotlib's pyplot is not
thread-safe either. ``fork`` is not used: a process with CASA loaded already runs some twenty threads.

A :class:`PlotJob` is either a backend plot method (``method`` + ``kwargs``), which may run in a worker
on a backend rebuilt there from :meth:`~vlbipy.backends.base.Backend.worker_spec`, or a callable
(``call``) that needs the observation and runs in this process, while the workers are busy. A worker
job's arguments travel pickled (the metadata included), so everything a plot needs is resolved before it
is submitted, and it returns only the paths it wrote.

How many plots run at once is decided before the first one starts (:func:`plan_workers`): at most the
configured number and the CPU count, and only as many as the largest plots fit together in a fraction of
the free memory, each plot's peak being estimated from the metadata by the backend
(:meth:`~vlbipy.backends.base.PlotOps.estimate_memory`). Each worker measures the real peak of every plot
it draws (Linux), and the log puts it beside the estimate.

A plot that fails is reported as a warning and the set carries on, as when the set is drawn serially. A
job whose worker could not run it (the worker died, or the backend could not be rebuilt) is drawn again
in this process.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .logging_utils import attach_logging, get_logger, logging_setup, warnings
from .tools import available_memory, usable_cpus

logger = get_logger()

#: The worker job, as :mod:`vlbipy.solvers.workers` addresses it.
WORKER_FUNCTION = "vlbipy.plot_jobs:plot_in_worker"

#: Backends rebuilt in this (worker) process, keyed by their spec, so later jobs skip importing CASA again.
_BACKENDS: dict = {}


@dataclass
class PlotJob:
    """One plot of a set.

    Parameters
    ----------
    name : str
        Label used in the warning when the plot fails (e.g. ``"spectrum[3C345]"``).
    method : str
        Backend plot method, called as ``backend.plot.<method>(project_code, **kwargs)``; may run in a
        worker process. Empty when ``call`` is given.
    kwargs : dict
        Its keyword arguments, picklable.
    call : callable, optional
        Run in this process instead (for plots that need the observation, not only the backend).
    """

    name: str
    method: str = ""
    kwargs: dict = field(default_factory=dict)
    call: Optional[Callable] = None


def as_paths(result) -> list:
    """Return a plot method's result as a list of paths (methods return one path or a list)."""
    return list(result) if isinstance(result, list) else [result]


def draw(backend, project_code: str, job: PlotJob):
    """Draw one job here and now; returns what the plot method returns and raises what it raises."""
    if job.call is not None:
        return job.call()
    return getattr(backend.plot, job.method)(project_code, **job.kwargs)


def job_cost(job: PlotJob) -> float:
    """Seconds of observation a job reads: total length of the scans of its field(s) and scans.

    Only an ordering key (longest first). Jobs without metadata, or that read every source (no field),
    count every scan.
    """
    metadata = job.kwargs.get("metadata")
    if metadata is None:
        return 0.0
    fields = job.kwargs.get("fields") or [name for name in str(job.kwargs.get("field") or "").split(",") if name]
    scans = set(job.kwargs.get("scans") or [])
    return float(sum(scan.time_end - scan.time_start for scan in metadata.scans
                     if (not fields or scan.source in fields) and (not scans or scan.scan_number in scans)))


def plot_workers(setting) -> int:
    """Most plots drawn at once for a ``[plotting].workers`` setting: ``"auto"`` (or 0) = every CPU this
    process may use (:func:`vlbipy.tools.usable_cpus`), a number = at most that many. The memory
    estimate of each plot set then lowers it where they would not fit (:func:`plan_workers`)."""
    if setting is None or str(setting).strip().lower() in ("auto", "", "0"):
        return usable_cpus()
    return max(1, int(setting))


def plan_workers(peaks: list, *, max_workers: int, available: Optional[int]) -> int:
    """Number of plots to draw at once.

    The most that ``max_workers`` allows (and no more than there are plots), as long as the largest plots,
    running together, fit in ``available`` bytes: the jobs start largest first, so that is the worst case.

    Parameters
    ----------
    peaks : list of int or None
        Estimated peak bytes of each plot; None (unknown) counts as nothing.
    max_workers : int
        Upper bound (configuration and CPU count).
    available : int or None
        Memory budget in bytes; None = unknown, only ``max_workers`` limits.

    Returns
    -------
    int
        At least 1.
    """
    sizes = sorted((int(peak or 0) for peak in peaks), reverse=True)
    limit = max(1, min(int(max_workers), len(sizes)))
    if available is None:
        return limit
    count = 1
    while count < limit and sum(sizes[:count + 1]) <= available:
        count += 1
    return count


def _gb(size: Optional[float]) -> str:
    """Bytes as ``"1.2 GB"`` (``"?"`` when unknown)."""
    return "?" if size is None else f"{size / 1e9:.1f} GB"


def _reset_peak_memory() -> None:
    """Reset this process's peak resident size (Linux), so the next reading is the peak of one job."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def _peak_memory() -> Optional[int]:
    """Peak resident size of this process in bytes since the last reset (Linux ``VmHWM``), else None."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _backend_for(spec: dict):
    """Return the backend of ``spec`` in this process, rebuilding it on first use."""
    from .solvers.workers import resolve
    key = json.dumps(spec, sort_keys=True)
    if key not in _BACKENDS:
        _BACKENDS[key] = resolve(spec["backend"]).from_worker_spec(spec)
    return _BACKENDS[key]


def plot_in_worker(*, backend: dict, project_code: str, method: str, kwargs: dict, log: dict) -> dict:
    """Worker job: draw one backend plot.

    Parameters
    ----------
    backend : dict
        :meth:`~vlbipy.backends.base.Backend.worker_spec` of the parent's backend.
    project_code : str
        Project the plot is of.
    method, kwargs : str, dict
        The backend plot method and its arguments.
    log : dict
        :func:`~vlbipy.logging_utils.logging_setup` of the parent, so this process logs to the same run log.

    Returns
    -------
    dict
        ``paths`` (list of str), ``error`` (empty, or the message of the exception the plot raised),
        ``seconds`` and ``peak_bytes`` (this process's peak resident size while drawing it, None where it
        cannot be measured). A backend that cannot be rebuilt raises instead, so the caller draws the job.
    """
    attach_logging(log)
    started = time.monotonic()
    plot_ops = _backend_for(backend).plot
    _reset_peak_memory()
    try:
        paths = [str(path) for path in as_paths(getattr(plot_ops, method)(project_code, **kwargs))]
        error = ""
    except Exception as exc:  # noqa: BLE001 - a failed plot is reported, never fatal
        logger.debug("plot job {} failed in a worker:\n{}", method, traceback.format_exc())
        paths, error = [], str(exc)
    return {"paths": paths, "error": error, "seconds": time.monotonic() - started, "peak_bytes": _peak_memory()}


def _draw_here(backend, project_code: str, job: PlotJob, label: str) -> list:
    """Draw ``job`` in this process; a failure becomes a warning and no paths."""
    try:
        return as_paths(draw(backend, project_code, job))
    except Exception as exc:  # noqa: BLE001 - a plot must never abort the reduction
        warnings.warn(f"{project_code}: {label} {job.name} plot failed ({exc})")
        return []


def _workers_for(backend, project_code: str, jobs: list[PlotJob], max_workers: int, memory_fraction: float,
                 label: str) -> tuple[int, list]:
    """Decide how many of ``jobs`` to draw at once (:func:`plan_workers`); returns it and the peak estimates."""
    peaks = [backend.plot.estimate_memory(project_code, job.method, job.kwargs) for job in jobs]
    cpus = usable_cpus()
    free = available_memory()
    budget = int(free * memory_fraction) if free is not None else None
    workers = plan_workers(peaks, max_workers=min(max_workers, cpus), available=budget)
    together = sum(sorted((peak or 0 for peak in peaks), reverse=True)[:workers])
    known = [peak for peak in peaks if peak is not None]
    logger.info("plot set {}: {} plot(s) up to {} each (estimated); {} at a time, needing {} of the {} usable "
                "({:.0%} of {} free; at most {} by configuration, {} CPUs)", label, len(jobs),
                _gb(max(known) if known else None), workers, _gb(together if known else None), _gb(budget),
                memory_fraction, _gb(free), max_workers, cpus)
    return workers, peaks


def _submit(backend, project_code: str, jobs: list[PlotJob], max_workers: int, memory_fraction: float,
            label: str) -> tuple[dict, dict, int]:
    """Start the backend-method jobs of ``jobs`` in worker processes, longest first.

    Returns ``({job index: future}, {job index: estimated peak bytes}, plots at a time)``; no futures when
    the jobs are to be drawn here (one at a time allowed, a single job, a backend that cannot be rebuilt
    elsewhere, or a pool that does not start).
    """
    from .solvers.workers import submit_jobs
    spec = backend.worker_spec(project_code) if max_workers > 1 and len(jobs) > 1 else None
    remote = [index for index, job in enumerate(jobs) if job.call is None] if spec else []
    if not remote:
        return {}, {}, 1
    workers, peaks = _workers_for(backend, project_code, [jobs[index] for index in remote], max_workers,
                                  memory_fraction, label)
    estimates = dict(zip(remote, peaks))
    if workers <= 1:
        return {}, estimates, 1
    # Longest first: the set ends no earlier than its longest plot, so that one must not start last.
    remote.sort(key=lambda index: -job_cost(jobs[index]))
    log = logging_setup()
    payloads = [{"backend": spec, "project_code": project_code, "method": jobs[index].method,
                 "kwargs": jobs[index].kwargs, "log": log} for index in remote]
    try:
        futures = submit_jobs(WORKER_FUNCTION, payloads, workers)
    except Exception as exc:  # noqa: BLE001 - the plots can still be drawn here, one at a time
        warnings.warn(f"{project_code}: the plot worker processes could not start ({exc}); drawing the "
                      f"{label} plots one at a time")
        return {}, estimates, 1
    return dict(zip(remote, futures)), estimates, workers


def _collect(backend, project_code: str, job: PlotJob, future, label: str, estimate: Optional[int]) -> tuple:
    """Wait for a worker job; returns its paths and measured peak bytes (None when drawn here instead)."""
    try:
        outcome = future.result()
    except Exception as exc:  # noqa: BLE001 - WorkerError: the worker died or could not rebuild the backend
        logger.warning("{}: the worker process could not draw {} ({}); drawing it here",
                       project_code, job.name, str(exc).splitlines()[0])
        return _draw_here(backend, project_code, job, label), None
    if outcome["error"]:
        warnings.warn(f"{project_code}: {label} {job.name} plot failed ({outcome['error']})")
    logger.debug("plot job {} ({}): {} file(s) in {:.0f} s in a worker process, peak memory {} (estimated {})",
                 job.name, label, len(outcome["paths"]), outcome["seconds"], _gb(outcome["peak_bytes"]),
                 _gb(estimate))
    return outcome["paths"], outcome["peak_bytes"]


def run_plot_jobs(backend, project_code: str, jobs: list[PlotJob], *, max_workers: int, memory_fraction: float,
                  label: str) -> list:
    """Draw every job of a plot set, several at a time; failures warn and the set carries on.

    Backend-method jobs go to worker processes when the backend can be rebuilt there
    (:meth:`~vlbipy.backends.base.Backend.worker_spec`) and more than one plot can run at once
    (:func:`plan_workers`); ``call`` jobs run in this process meanwhile, in order. Otherwise everything is
    drawn here, one job after another. The memory of the ``call`` jobs is not budgeted: they read a few
    scans at a time.

    Parameters
    ----------
    backend : Backend
        The observation's backend.
    project_code : str
        Project code.
    jobs : list of PlotJob
        The plots, in the order their paths are returned.
    max_workers : int
        Most plots drawn at the same time (``[plotting].workers``); the CPU count also caps it.
    memory_fraction : float
        Share of the free memory the plots drawn together may take (``[plotting].memory_fraction``).
    label : str
        Set label for the warnings and the log (``"raw"``, ``"calibrated"``, ``"final"``).

    Returns
    -------
    list of str
        Paths written, in job order.
    """
    started = time.monotonic()
    futures, estimates, workers = _submit(backend, project_code, jobs, max_workers, memory_fraction, label)
    measured: dict[int, Optional[int]] = {}
    try:
        results = {index: _draw_here(backend, project_code, job, label)
                   for index, job in enumerate(jobs) if index not in futures}
        for index, future in futures.items():
            results[index], measured[index] = _collect(backend, project_code, jobs[index], future, label,
                                                       estimates.get(index))
    except BaseException:
        # Interrupted (Ctrl-C): the jobs not started yet must not keep the process alive drawing them.
        for future in futures.values():
            future.cancel()
        raise
    if futures:
        peaks = [peak for peak in measured.values() if peak is not None]
        known = [peak for peak in estimates.values() if peak is not None]
        logger.info("plot set {}: {} plot job(s), {} in worker processes ({} at a time), {:.0f} s; peak memory "
                    "per plot up to {} measured, {} estimated", label, len(jobs), len(futures), workers,
                    time.monotonic() - started, _gb(max(peaks) if peaks else None), _gb(max(known) if known else None))
    return [path for index in range(len(jobs)) for path in results[index]]
