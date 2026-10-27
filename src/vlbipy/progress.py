"""A shared rich progress display for the multi-job steps of the solver engine.

One :class:`rich.progress.Progress` serves the whole process: every
:func:`vlbipy.solvers.workers.run_jobs` call adds a task to it, so a step shows
one bar per running stage — and a multi-epoch run, whose projects work in
parallel, one bar per project-stage — instead of interleaved renderings
fighting for the same line. Bars disappear when their step finishes; the step's
own log lines (in the run log) keep the record. Downloads (:func:`track_downloads`) share the
same display, counted in bytes instead of jobs.

The display is disabled on a non-terminal stderr and inside solver worker
processes (``VLBIPY_WORKER_AUTHKEY`` is set there), so batch logs stay clean and
a job that runs :func:`run_jobs` nested does not render.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from rich.console import Console
from rich.progress import (BarColumn, DownloadColumn, MofNCompleteColumn, Progress, ProgressColumn, SpinnerColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn, TransferSpeedColumn)
from rich.text import Text

_PROGRESS: Progress | None = None
_LOCK = threading.Lock()


def _disabled() -> bool:
    """True when bars must not render (no terminal, or a worker process)."""
    return "VLBIPY_WORKER_AUTHKEY" in os.environ or not sys.stderr.isatty()


class _CountColumn(ProgressColumn):
    """Jobs done out of the total, or, for a task with ``unit="bytes"``, the bytes moved and the transfer speed."""

    def __init__(self) -> None:
        super().__init__()
        self._jobs, self._bytes, self._speed = MofNCompleteColumn(), DownloadColumn(), TransferSpeedColumn()

    def render(self, task) -> Text:
        if task.fields.get("unit") == "bytes":
            return Text.assemble(self._bytes.render(task), " ", self._speed.render(task))
        return self._jobs.render(task)


def shared_progress() -> Progress:
    """The process-wide progress display, started on first use."""
    global _PROGRESS
    with _LOCK:
        if _PROGRESS is None:
            # One display for jobs and downloads alike: rich allows a single live display per console.
            _PROGRESS = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                                 _CountColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
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


class DownloadTracker:
    """Byte progress of a set of files downloaded together: one overall bar and one bar per file in flight.

    The overall total is the sum of the sizes the server announced so far, the files not started yet
    counted at the mean of those: archives list file names, not sizes, and asking for every size up front
    would cost a request per file.
    """

    def __init__(self, progress: Progress, description: str, names: list[str]) -> None:
        self._progress, self._description, self._names = progress, description, list(names)
        self._sizes: dict[str, Optional[int]] = {}
        self._done: set[str] = set()
        self._tasks: dict[str, int] = {}
        self._lock = threading.Lock()
        self.overall = progress.add_task(self._label(), total=None, unit="bytes")

    def _label(self) -> str:
        return f"{self._description} ({len(self._done)}/{len(self._names)} files)"

    def _update_total(self) -> None:
        known = [size for size in self._sizes.values() if size]
        if known:
            total = sum(known) + (len(self._names) - len(self._sizes)) * sum(known) / len(known)
            self._progress.update(self.overall, total=total)

    def callback(self, name: str) -> Callable[[int, Optional[int]], None]:
        """The ``progress`` argument of :func:`vlbipy.tools.download_file` for the file ``name``."""
        def update(n_bytes: int, size: Optional[int]) -> None:
            with self._lock:
                if name not in self._sizes:
                    self._sizes[name] = size
                    self._tasks[name] = self._progress.add_task(f"  {name}", total=size, unit="bytes")
                    self._update_total()
                if n_bytes:
                    self._progress.advance(self._tasks[name], n_bytes)
                    self._progress.advance(self.overall, n_bytes)
        return update

    def finished(self, name: str) -> None:
        """Mark ``name`` complete: its own bar goes, the overall count moves on."""
        with self._lock:
            self._done.add(name)
            task = self._tasks.pop(name, None)
            if task is not None:
                self._progress.remove_task(task)
            self._progress.update(self.overall, description=self._label())


@contextmanager
def track_downloads(description: str, names: list[str]) -> Iterator[DownloadTracker]:
    """A :class:`DownloadTracker` on the shared display for downloading ``names``; its bars go at the end."""
    progress = shared_progress()
    tracker = DownloadTracker(progress, description, names)
    try:
        yield tracker
    finally:
        for task in list(tracker._tasks.values()):
            progress.remove_task(task)
        progress.remove_task(tracker.overall)
