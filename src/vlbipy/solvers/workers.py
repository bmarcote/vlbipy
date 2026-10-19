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
small (selections, solutions) — never visibility cubes. The one large kind of
argument, a calibration table loaded in memory (a Tsys table is 57 MB pickled),
is marked ``_worker_shared``: it is sent to each worker once, keyed by a digest
of its content, and every other job refers to the worker's cached copy (see
:func:`_share`). Without that, a fringe fit of 700 scans moved 40 GB through
the sockets.

Inside a job, array work that is cut into independent blocks runs in threads
(:func:`thread_map`): numpy releases the GIL in its array loops, so those scale
with the cores even though the table I/O does not. Every job gets its share of
the cores (all of them when fewer jobs than cores run), so a few long jobs, as
one per Multi-MS part, still use the whole machine.

A job that itself calls :func:`run_jobs` or :func:`submit_jobs` (a plot drawn in
a worker that reads the measurement set) runs those jobs in its own process: the
caller already spreads its work over the pool, and a pool started in every
worker multiplied the processes (16 x 16 on a 20-core machine) and their memory
beyond what the caller planned for.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib
import os
import queue
import subprocess
import sys
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Client, Listener
from typing import Optional

import numpy as np

#: Environment variable overriding the number of worker processes.
WORKERS_ENV = "VLBIPY_WORKERS"
#: Upper bound on the default pool size (the Multi-MS parts and scans rarely feed more).
MAX_DEFAULT_WORKERS = 16
#: Seconds to wait for the worker processes to start and connect back.
START_TIMEOUT = 120.0
#: Shared objects (calibration tables) a worker keeps; the least recently used one is dropped beyond that.
WORKER_CACHE_SIZE = 32
#: Environment variable overriding the number of threads a job uses for its array work (:func:`thread_map`).
THREADS_ENV = "VLBIPY_THREADS"
#: Environment variable the pool sets in every worker process (it carries the key of the connection back).
AUTHKEY_ENV = "VLBIPY_WORKER_AUTHKEY"

_POOL: Optional["WorkerPool"] = None
_POOL_LOCK = threading.Lock()
#: Threads the job running in this worker process may use (set by the pool with every job).
_JOB_THREADS: Optional[int] = None
_EXECUTOR: Optional[tuple] = None                # (executor, number of threads)
_EXECUTOR_LOCK = threading.Lock()
_IN_THREAD = threading.local()


def default_workers() -> int:
    """Number of worker processes to use: ``$VLBIPY_WORKERS`` or the CPU count capped at 16."""
    configured = os.environ.get(WORKERS_ENV, "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, min(os.cpu_count() or 1, MAX_DEFAULT_WORKERS))


def in_worker() -> bool:
    """True in a worker process of a :class:`WorkerPool`, where :func:`run_jobs` and :func:`submit_jobs` run
    their jobs in the process instead of starting a pool of their own."""
    return bool(os.environ.get(AUTHKEY_ENV))


def job_threads() -> int:
    """Threads the array work of the running job may use (:func:`thread_map`).

    ``$VLBIPY_THREADS`` when set; in a worker process, its share of the cores for the jobs running alongside
    it (set by the pool); otherwise (the main process) every core.
    """
    configured = os.environ.get(THREADS_ENV, "").strip()
    if configured:
        return max(1, int(configured))
    if _JOB_THREADS is not None:
        return _JOB_THREADS
    return max(1, os.cpu_count() or 1)


def in_thread_map() -> bool:
    """True inside one of the threads of :func:`thread_map` (which then share the cores among them)."""
    return bool(getattr(_IN_THREAD, "active", False))


def thread_map(function, items) -> list:
    """``[function(item) for item in items]``, in up to :func:`job_threads` threads.

    For blocks of numpy work that do not depend on each other (numpy releases the GIL in its array loops,
    so they run on several cores at once). Each thread takes every n-th item; the results come back in
    order and an exception in any of them is raised here. A call from inside one of these threads runs
    its items one after the other.
    """
    items = list(items)
    threads = min(job_threads(), len(items))
    if threads <= 1 or in_thread_map():
        return [function(item) for item in items]

    def share(first: int) -> list:
        _IN_THREAD.active = True
        try:
            return [function(item) for item in items[first::threads]]
        finally:
            _IN_THREAD.active = False

    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None or _EXECUTOR[1] < threads:
            size = max(threads, os.cpu_count() or 1)
            _EXECUTOR = (ThreadPoolExecutor(max_workers=size, thread_name_prefix="vlbipy-array"), size)
        executor = _EXECUTOR[0]
    shares = [future.result() for future in [executor.submit(share, first) for first in range(threads)]]
    return [shares[index % threads][index // threads] for index in range(len(items))]


def resolve(function_path: str):
    """Import and return the callable named by ``"package.module:name"``."""
    module_name, _, name = function_path.partition(":")
    if not name:
        raise ValueError(f"worker function must be 'module:name', got {function_path!r}")
    return getattr(importlib.import_module(module_name), name)


class WorkerError(RuntimeError):
    """A job raised inside a worker process; the message carries the remote traceback."""


class SharedRef:
    """Placeholder for a shared object in a job: the worker substitutes its cached copy."""

    __slots__ = ("digest",)

    def __init__(self, digest: str) -> None:
        self.digest = digest

    def __reduce__(self):
        return (SharedRef, (self.digest,))


def _digest(value) -> str:
    """Content digest of a shared object (its arrays and plain attributes)."""
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(type(value).__qualname__.encode())
    for name, item in sorted(vars(value).items()):
        if name.startswith("_"):
            continue      # caches built on the object (row indices) are not content
        hasher.update(name.encode())
        if isinstance(item, np.ndarray):
            hasher.update(str((item.dtype, item.shape)).encode())
            hasher.update(np.ascontiguousarray(item).view(np.uint8).reshape(-1) if item.dtype != object
                          else repr(item.tolist()).encode())
        else:
            hasher.update(repr(item).encode())
    return hasher.hexdigest()


def _share(value, digests: dict, held: set, outgoing: dict, depth: int = 0):
    """Replace every ``_worker_shared`` object in ``value`` (dicts, lists, tuples) by a :class:`SharedRef`.

    ``digests`` caches the digest per object id for one map call, ``held`` holds the digests the target worker
    already has, and objects it lacks are put in ``outgoing`` to travel with the job.
    """
    if getattr(type(value), "_worker_shared", False):
        key = id(value)
        if key not in digests:
            digests[key] = _digest(value)
        digest = digests[key]
        if digest not in held:
            outgoing[digest] = value
        return SharedRef(digest)
    if depth > 6:
        return value
    if isinstance(value, dict):
        return {k: _share(v, digests, held, outgoing, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_share(v, digests, held, outgoing, depth + 1) for v in value]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_share(v, digests, held, outgoing, depth + 1) for v in value)
    return value


def _unshare(value, cache: "OrderedDict", depth: int = 0):
    """Inverse of :func:`_share` inside a worker: substitute the cached objects."""
    if isinstance(value, SharedRef):
        cache.move_to_end(value.digest)
        return cache[value.digest]
    if depth > 6:
        return value
    if isinstance(value, dict):
        return {k: _unshare(v, cache, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_unshare(v, cache, depth + 1) for v in value]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_unshare(v, cache, depth + 1) for v in value)
    return value


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
        environment[AUTHKEY_ENV] = authkey.hex()
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
        # A queue, so a caller with more jobs in flight than workers waits for one to come free; last in,
        # first out, so the worker freed last (its libraries already loaded) is the one reused.
        self._idle: queue.LifoQueue = queue.LifoQueue()
        for index in reversed(range(self.size)):
            self._idle.put(index)
        # Digests of the shared objects each worker holds (mirrors the worker's LRU cache).
        self._held: list[OrderedDict] = [OrderedDict() for _ in range(self.size)]
        self._digests: dict = {}
        self._threads = ThreadPoolExecutor(max_workers=self.size)
        self.closed = False

    @property
    def busy(self) -> bool:
        """True while any worker is running a job (the pool must not be closed under it)."""
        return self._idle.qsize() < self.size

    def threads_for(self, together: int) -> int:
        """Threads each of ``together`` jobs running at once may use for its array work (its share of the cores)."""
        return max(1, (os.cpu_count() or 1) // max(1, min(int(together), self.size)))

    def _run(self, job: tuple, digests: Optional[dict] = None, threads: int = 1) -> object:
        """Send one job to an idle worker and return its result (raises :class:`WorkerError` on failure).

        ``digests`` caches shared-object digests by object id; it defaults to the one of the running map call.
        ``threads`` is the job's share of the cores (:func:`job_threads` in the worker).
        """
        if self.closed:
            raise WorkerError(f"the worker pool is closed; {job[0]} was not run")
        index = self._idle.get()
        try:
            connection = self._connections[index]
            function_path, kwargs = job
            held, outgoing = self._held[index], {}
            kwargs = _share(kwargs, self._digests if digests is None else digests, held, outgoing)
            for digest in outgoing:
                held[digest] = True
            for digest in kwargs_digests(kwargs):
                held.move_to_end(digest)
            while len(held) > WORKER_CACHE_SIZE:
                held.popitem(last=False)
            connection.send((function_path, kwargs, outgoing, threads))
            ok, payload = connection.recv()
        except (EOFError, OSError) as exc:
            # The worker died (out of memory, a crash in a C library): the pool cannot be trusted any more.
            self.closed = True
            raise WorkerError(f"worker process running {job[0]} died: {exc!r}") from exc
        finally:
            self._idle.put(index)
        if not ok:
            raise WorkerError(f"worker job {job[0]} failed:\n{payload}")
        return payload

    def run(self, function_path: str, kwargs: dict, threads: Optional[int] = None) -> object:
        """Run ``function(**kwargs)`` in an idle worker (waiting for one) and return its result.

        ``threads`` is the job's share of the cores; by default that of a job running beside a full pool.
        """
        return self._run((function_path, kwargs), digests={},
                         threads=self.threads_for(self.size) if threads is None else int(threads))

    def map(self, function_path: str, jobs: list[dict], workers: Optional[int] = None) -> list:
        """Run ``function(**kwargs)`` for every kwargs dict of ``jobs``, at most ``workers`` (default: the pool
        size) at a time; results come back in job order."""
        if self.closed:
            raise RuntimeError("the worker pool is closed")
        together = max(1, min(int(workers or self.size), self.size, len(jobs)))
        # The cores are shared among the jobs that run at the same time (see :func:`thread_map`).
        threads = self.threads_for(together)
        # Object ids are only stable while the caller holds the jobs; a map of its own keeps two maps running
        # at once (two threads of the main process) from sharing it.
        digests: dict = {}
        if together >= self.size:
            return list(self._threads.map(lambda job: self._run(job, digests, threads),
                                          [(function_path, kwargs) for kwargs in jobs]))
        with ThreadPoolExecutor(max_workers=together) as executor:
            return list(executor.map(lambda job: self._run(job, digests, threads),
                                     [(function_path, kwargs) for kwargs in jobs]))

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
    """Return the shared pool, starting it (or restarting it with another size) when needed.

    A pool running jobs is never restarted for another size (that would cut off the jobs of another thread,
    as the plot workers of :mod:`vlbipy.plot_jobs`): the caller gets the running one, and :meth:`WorkerPool.map`
    then runs at most the ``workers`` it asked for at a time.
    """
    global _POOL
    size = int(workers) if workers else default_workers()
    with _POOL_LOCK:
        if (_POOL is None or _POOL.closed
                or (workers and _POOL.size != size and not _POOL.busy)):
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
    """Run ``jobs`` and return their results in order: in this process for one job, ``workers=1`` or inside a
    worker process (:func:`in_worker`), else in the pool."""
    if not jobs:
        return []
    size = int(workers) if workers else default_workers()
    if size <= 1 or len(jobs) == 1 or in_worker():
        function = resolve(function_path)
        return [function(**kwargs) for kwargs in jobs]
    return get_pool(workers).map(function_path, jobs, workers=size)


def submit_jobs(function_path: str, jobs: list[dict], workers: int) -> list[Future]:
    """Start ``jobs`` in the shared pool, at most ``workers`` at a time, and return one future per job, in order.

    Returns at once, so the caller can do other work meanwhile. Each future gives the job's result or raises
    :class:`WorkerError` (the job raised, or its worker died); a failed job does not stop the others. Raises
    ``RuntimeError`` when the pool cannot be started. Inside a worker process (:func:`in_worker`) the jobs run
    here, one after the other, before this returns.
    """
    if not jobs:
        return []
    if in_worker():
        return [_run_here(function_path, kwargs) for kwargs in jobs]
    pool = get_pool()
    together = max(1, min(int(workers), pool.size))
    threads = pool.threads_for(together)
    executor = ThreadPoolExecutor(max_workers=together)
    futures = [executor.submit(pool.run, function_path, kwargs, threads) for kwargs in jobs]
    executor.shutdown(wait=False)       # the submitted jobs still run; the threads end with the last one
    return futures


def _run_here(function_path: str, kwargs: dict) -> Future:
    """A finished future of ``function(**kwargs)`` run in this process (a failure becomes a :class:`WorkerError`)."""
    future: Future = Future()
    try:
        future.set_result(resolve(function_path)(**kwargs))
    except Exception:  # noqa: BLE001 - reported through the future, as a worker job would be
        future.set_exception(WorkerError(f"job {function_path} failed:\n{traceback.format_exc()}"))
    return future


def kwargs_digests(value, depth: int = 0) -> list[str]:
    """Digests of the :class:`SharedRef` placeholders in a job (to refresh their place in the LRU order)."""
    if isinstance(value, SharedRef):
        return [value.digest]
    if depth > 6:
        return []
    if isinstance(value, dict):
        return [d for v in value.values() for d in kwargs_digests(v, depth + 1)]
    if isinstance(value, (list, tuple)):
        return [d for v in value for d in kwargs_digests(v, depth + 1)]
    return []


def _serve(address: str) -> None:
    """Worker main loop: receive ``(function path, kwargs, shared objects, threads)``, reply ``(True, result)``
    or ``(False, traceback)``; shared objects are kept for later jobs (see :func:`_share`)."""
    global _JOB_THREADS
    connection = Client(address, authkey=bytes.fromhex(os.environ[AUTHKEY_ENV]))
    cache: OrderedDict = OrderedDict()
    while True:
        try:
            function_path, kwargs, shared, _JOB_THREADS = connection.recv()
        except (EOFError, OSError):
            return
        try:
            cache.update(shared)
            kwargs = _unshare(kwargs, cache)
            while len(cache) > WORKER_CACHE_SIZE:
                cache.popitem(last=False)
            reply = (True, resolve(function_path)(**kwargs))
        except BaseException:  # noqa: BLE001 - every failure must travel back to the caller
            reply = (False, traceback.format_exc())
        connection.send(reply)


if __name__ == "__main__":
    # Serve from the imported module, not from __main__: the jobs unpickle vlbipy.solvers.workers.SharedRef,
    # which must be the class the loop checks against.
    from vlbipy.solvers.workers import _serve as serve
    serve(sys.argv[1])
