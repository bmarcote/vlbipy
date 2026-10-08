"""A small persistent pool of worker processes for the measurement-set solvers.

python-casacore holds the GIL while it reads or writes a column, so threads give
no speed-up on the part that dominates every calibration task: the table I/O.
Processes do (the whole 3 GB DATA column of an 8-part Multi-MS is read in under a
second by 16 of them, against 6.5 s single-threaded), so each solver cuts its work
into jobs that *read, reduce and return a small result* inside a worker.

The pool is started on first use and kept for the life of the interpreter: a
pipeline run calls the solvers dozens of times and must not pay the start-up
(importing numpy/scipy/casacore) each time.

Workers are plain ``python -m vlbipy.solvers.workers`` subprocesses that connect
back over a local socket. ``multiprocessing``'s own spawn/forkserver start
methods are deliberately not used: they re-import the caller's ``__main__`` in
every child, which re-runs an unguarded user script, and ``fork`` would duplicate
a process that already has CASA loaded.

A job is ``(function path, kwargs)`` with the function given as
``"package.module:name"``; arguments and results travel pickled, so they must be
small (selections, solutions) — never visibility cubes.
"""
from __future__ import annotations

import atexit
import importlib
import os
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client, Listener
from typing import Optional

#: Environment variable overriding the number of worker processes.
WORKERS_ENV = "VLBIPY_WORKERS"
#: Upper bound on the default pool size (the Multi-MS parts and scans rarely feed more).
MAX_DEFAULT_WORKERS = 16
#: Seconds to wait for the worker processes to start and connect back.
START_TIMEOUT = 120.0

_POOL: Optional["WorkerPool"] = None
_POOL_LOCK = threading.Lock()


def default_workers() -> int:
    """Number of worker processes to use: ``$VLBIPY_WORKERS`` or the CPU count capped at 16."""
    configured = os.environ.get(WORKERS_ENV, "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, min(os.cpu_count() or 1, MAX_DEFAULT_WORKERS))


def resolve(function_path: str):
    """Import and return the callable named by ``"package.module:name"``."""
    module_name, _, name = function_path.partition(":")
    if not name:
        raise ValueError(f"worker function must be 'module:name', got {function_path!r}")
    return getattr(importlib.import_module(module_name), name)


class WorkerError(RuntimeError):
    """A job raised inside a worker process; the message carries the remote traceback."""


class WorkerPool:
    """``size`` worker subprocesses, each serving one job at a time over its own connection."""

    def __init__(self, size: int) -> None:
        """Start ``size`` workers and wait until every one has connected back."""
        self.size = int(size)
        authkey = os.urandom(32)
        listener = Listener(authkey=authkey)
        environment = dict(os.environ)
        # One job per core: numerical libraries inside a worker must not start their own thread teams.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VLBIPY_FFT_WORKERS"):
            environment[name] = "1"
        environment["VLBIPY_WORKER_AUTHKEY"] = authkey.hex()
        command = [sys.executable, "-m", "vlbipy.solvers.workers", str(listener.address)]
        self._processes = [subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL)
                           for _ in range(self.size)]
        # A worker that cannot start (broken environment) would leave accept() waiting for ever.
        raw_socket = getattr(getattr(listener, "_listener", None), "_socket", None)
        if raw_socket is not None:
            raw_socket.settimeout(START_TIMEOUT)
        try:
            self._connections = [listener.accept() for _ in range(self.size)]
        except OSError as exc:
            for process in self._processes:
                process.kill()
            raise RuntimeError(f"the solver worker processes did not start within {START_TIMEOUT:.0f} s "
                               f"({sys.executable} -m vlbipy.solvers.workers): {exc!r}") from exc
        finally:
            listener.close()
        self._idle = list(range(self.size))
        self._idle_lock = threading.Lock()
        self._threads = ThreadPoolExecutor(max_workers=self.size)
        self.closed = False

    def _run(self, job: tuple) -> object:
        """Send one job to an idle worker and return its result (raises :class:`WorkerError` on failure)."""
        with self._idle_lock:
            index = self._idle.pop()
        try:
            connection = self._connections[index]
            connection.send(job)
            ok, payload = connection.recv()
        except (EOFError, OSError) as exc:
            # The worker died (out of memory, a crash in a C library): the pool cannot be trusted any more.
            self.closed = True
            raise WorkerError(f"worker process running {job[0]} died: {exc!r}") from exc
        finally:
            with self._idle_lock:
                self._idle.append(index)
        if not ok:
            raise WorkerError(f"worker job {job[0]} failed:\n{payload}")
        return payload

    def map(self, function_path: str, jobs: list[dict]) -> list:
        """Run ``function(**kwargs)`` for every kwargs dict of ``jobs``; results come back in job order."""
        if self.closed:
            raise RuntimeError("the worker pool is closed")
        return list(self._threads.map(self._run, [(function_path, kwargs) for kwargs in jobs]))

    def close(self) -> None:
        """Stop the workers (closing a connection ends that worker's loop)."""
        if self._threads is None:
            return
        self.closed = True
        self._threads.shutdown(wait=True)
        self._threads = None
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def get_pool(workers: Optional[int] = None) -> WorkerPool:
    """Return the shared pool, starting it (or restarting it with another size) when needed."""
    global _POOL
    size = int(workers) if workers else default_workers()
    with _POOL_LOCK:
        if _POOL is None or _POOL.closed or (workers and _POOL.size != size):
            if _POOL is not None:
                _POOL.close()
            _POOL = WorkerPool(size)
        return _POOL


def close_pool() -> None:
    """Stop the shared pool, if one is running (registered with ``atexit``)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.close()
            _POOL = None


atexit.register(close_pool)


def run_jobs(function_path: str, jobs: list[dict], workers: Optional[int] = None) -> list:
    """Run ``jobs`` and return their results in order: in this process for one job or ``workers=1``, else in the pool."""
    if not jobs:
        return []
    size = int(workers) if workers else default_workers()
    if size <= 1 or len(jobs) == 1:
        function = resolve(function_path)
        return [function(**kwargs) for kwargs in jobs]
    return get_pool(workers).map(function_path, jobs)


def _serve(address: str) -> None:
    """Worker main loop: receive ``(function path, kwargs)``, reply ``(True, result)`` or ``(False, traceback)``."""
    connection = Client(address, authkey=bytes.fromhex(os.environ["VLBIPY_WORKER_AUTHKEY"]))
    while True:
        try:
            function_path, kwargs = connection.recv()
        except (EOFError, OSError):
            return
        try:
            reply = (True, resolve(function_path)(**kwargs))
        except BaseException:  # noqa: BLE001 - every failure must travel back to the caller
            reply = (False, traceback.format_exc())
        connection.send(reply)


if __name__ == "__main__":
    _serve(sys.argv[1])
