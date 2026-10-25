"""A shared rich progress display for the multi-job steps of the solver engine.

One :class:`rich.progress.Progress` serves the whole process: every
:func:`vlbipy.solvers.workers.run_jobs` call adds a task to it, so a step shows
one bar per running stage — and a multi-epoch run, whose projects work in
parallel, one bar per project-stage — instead of interleaved renderings
fighting for the same line. Bars disappear when their step finishes; the step's
own log lines (in the run log) keep the record.

The display is disabled on a non-terminal stderr and inside solver worker
processes (``VLBIPY_WORKER_AUTHKEY`` is set there), so batch logs stay clean and
a job that runs :func:`run_jobs` nested does not render.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

_PROGRESS: Progress | None = None
_LOCK = threading.Lock()


def _disabled() -> bool:
    """True when bars must not render (no terminal, or a worker process)."""
    return "VLBIPY_WORKER_AUTHKEY" in os.environ or not sys.stderr.isatty()


def shared_progress() -> Progress:
    """The process-wide progress display, started on first use."""
    global _PROGRESS
    with _LOCK:
        if _PROGRESS is None:
            _PROGRESS = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                                 MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
                                 console=Console(stderr=True), disable=_disabled())
            _PROGRESS.start()
        return _PROGRESS


@contextmanager
def track_jobs(description: str, total: int) -> Iterator[Callable[[int], None]]:
    """Add a progress task for ``total`` jobs; yield a callable that advances it one job per call.

    Safe to enter from several threads at once (the multi-epoch fan-out does): each call gets its own
    bar on the shared display.
    """
    progress = shared_progress()
    task = progress.add_task(description, total=max(1, int(total)))
    try:
        yield lambda n=1: progress.advance(task, n)
    finally:
        progress.remove_task(task)
