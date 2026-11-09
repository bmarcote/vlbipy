"""Logging utilities for vlbipy, built on loguru.

Provides a configured colored logger (orange/yellow warnings, red-bold errors)
and a :class:`WarningCollector` that aggregates warnings and anomalies during a
run so a summary can be printed at the end and embedded in reports.

loguru is used per project convention for this (complex) program. Configuration
is centralised here so the logging backend stays swappable.
"""
import contextlib
import datetime as dt
import logging as _stdlib_logging
import os
import sys
import tempfile
import warnings as _pywarnings
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

_CONFIGURED = False
#: Console level last configured, handed to worker processes (:func:`logging_setup`).
_CONSOLE_LEVEL = "INFO"
#: The :func:`logging_setup` this (worker) process was last attached to, so re-attaching is free.
_ATTACHED: dict = {}
_FORMAT = "<level>{message}</level>\n{exception}"
#: Step headlines (:func:`headline`) render in bold.
_HEADLINE_FORMAT = "<bold>{message}</bold>\n{exception}"
#: File sink format: timestamped (UTC) and level-tagged, so a run can be audited afterwards. The zone is fixed
#: because the local one does not stay put: the vendored casavlbitools sets ``TZ=UTC`` for the whole process
#: when it is imported, which switched every later line of a run log from local time to UTC.
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss!UTC} | {level: <7} | {message}"
#: Log files already attached, keyed by directory, so re-entry does not duplicate sinks.
_FILE_SINKS: dict = {}


def configure_logging(level: str = "INFO") -> None:
    """Configure the loguru logger for vlbipy.

    Removes any existing sinks and installs a single stderr sink with a concise
    colored format. Warnings render orange/yellow and errors red-bold. Idempotent.

    Parameters
    ----------
    level : str
        Minimum level to display (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED, _CONSOLE_LEVEL
    _logger.remove()
    _logger.level("WARNING", color="<yellow>")
    _logger.level("ERROR", color="<red><bold>")
    _logger.level("INFO", color="<normal>")
    _logger.add(sys.stderr, level=level, format=_console_format, colorize=True,
                filter=_steps_only if str(level).upper() == "INFO" else None)
    _CONFIGURED = True
    _CONSOLE_LEVEL = level


#: Environment variable that hands the run logs to the solver worker processes for their Python warnings.
WARNINGS_LOG_ENV = "VLBIPY_WARNINGS_LOG"


def capture_python_warnings(files: Optional[list] = None) -> list[int]:
    """Send Python warnings and stdlib ``logging`` records to the run log instead of the terminal.

    They come from the libraries underneath (numpy's "Mean of empty slice" for a time bin without data, astropy,
    dask, the stdlib loggers of the solvers) and describe a computation, not the data: on the terminal they read
    as problems with the run. Python warnings become DEBUG lines; a stdlib record keeps its level, and the console
    shows it only when it is an error (or with ``--verbose``). What the user should act on goes through
    :data:`warnings` and the console sink. Only the command line and its worker processes install this; a notebook
    keeps Python's own behaviour.

    ``files``: run logs to append these lines to, for a process that has no file sink of its own (a solver
    worker, which logs nothing else there). Returns the ids of the sinks added for them.
    """
    _pywarnings.showwarning = _log_python_warning
    root = _stdlib_logging.getLogger()
    root.handlers = [_InterceptHandler()]
    return [_logger.add(str(path), level="DEBUG", format=_FILE_FORMAT, colorize=False, enqueue=False,
                        filter=lambda record: record["extra"].get("library", False))
            for path in files or []]


@contextlib.contextmanager
def native_output_to_log(label: str):
    """Send what compiled libraries print on the terminal during the block to the run log (debug level).

    CASA's logger writes its WARN and SEVERE lines straight to the process's file descriptors, whatever
    ``casalog`` is told: a task expected to complain (concat of two epochs whose subtables differ) would
    otherwise put them in front of the user. Each captured line is logged as ``<label>: <line>``.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    sink = tempfile.TemporaryFile()
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        sink.seek(0)
        for line in sink.read().decode(errors="replace").splitlines():
            if line.strip():
                _logger.bind(library=True).debug("{}: {}", label, line.rstrip())
        sink.close()


def capturing_python_warnings() -> bool:
    """True when this process sends library output to the run log (:func:`capture_python_warnings`)."""
    return _pywarnings.showwarning is _log_python_warning


def _log_python_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """``warnings.showwarning`` replacement: one DEBUG line in the run log."""
    _logger.bind(library=True).debug("python warning: {}:{}: {}: {}", filename, lineno, category.__name__, message)


class _InterceptHandler(_stdlib_logging.Handler):
    """Hands stdlib ``logging`` records to loguru, so they go where vlbipy's own lines go."""

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        _logger.bind(library=True).opt(exception=record.exc_info).log(level, "{}: {}", record.name,
                                                                      record.getMessage())


def _console_format(record) -> str:
    """Bold for a step headline, the level's colour otherwise."""
    return _HEADLINE_FORMAT if record["extra"].get("headline") else _FORMAT


def _steps_only(record) -> bool:
    """The console filter of the default level: headlines, anomalies and errors (the rest is in the run log)."""
    extra = record["extra"]
    return bool(extra.get("headline") or extra.get("anomaly")) or record["level"].no >= _logger.level("ERROR").no


def headline(message: str, *args) -> None:
    """Log the main line of a step: shown in bold on the console at the default level (and in the run log).

    Parameters
    ----------
    message : str
        loguru format string.
    *args
        Its arguments.
    """
    get_logger().opt(depth=1).bind(headline=True).info(message, *args)


def run_logs() -> list[Path]:
    """The run log files this process writes (see :func:`add_file_log`)."""
    return sorted(sink["path"] for sink in _FILE_SINKS.values())


def add_file_log(log_dir, project_code: str = "", level: str = "DEBUG"):
    """Attach a file sink so the whole run is recorded, not just shown.

    Everything the pipeline reports — each step it starts or skips, every
    external task with its resolved parameters, and the numbers that matter
    (flagged percentages, solution SNRs, selected antennas and scans) — already
    goes through this logger. Persisting it is what makes a finished run
    auditable and a failed batch run debuggable without reproducing it.

    The sink is attached once per directory: constructing several observations
    against the same working directory must not multiply every line.

    Parameters
    ----------
    log_dir : str or pathlib.Path
        Directory for the log file (created if needed).
    project_code : str
        Included in the file name.
    level : str
        Minimum level written to the file (``DEBUG`` keeps the task parameters).

    Returns
    -------
    pathlib.Path
        The log file being written.
    """
    get_logger()                      # ensure the console sink exists first
    directory = Path(log_dir)
    existing = _FILE_SINKS.get(str(directory.resolve()) if directory.exists() else str(directory))
    if existing is not None:
        return existing["path"]
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / (f"vlbipy_{project_code}_{stamp}.log" if project_code
                        else f"vlbipy_{stamp}.log")
    sink_id = _logger.add(str(path), level=level, format=_FILE_FORMAT, colorize=False,
                          enqueue=False, backtrace=True, diagnose=False)
    _FILE_SINKS[str(directory.resolve())] = {"path": path, "id": sink_id}
    _logger.bind(headline=True).info("logging this run to {}", path)
    return path


def logging_setup() -> dict:
    """Describe this process's logging so a worker process can log the same way (:func:`attach_logging`).

    Returns
    -------
    dict
        ``{"level": console level, "files": [log file paths]}``; JSON- and pickle-safe.
    """
    return {"level": _CONSOLE_LEVEL, "files": sorted(str(sink["path"]) for sink in _FILE_SINKS.values())}


def attach_logging(setup: dict) -> None:
    """Log like the process that produced ``setup``: same console level, appending to the same run log(s).

    Meant for worker processes, which start with loguru's default sink. Several processes appending to one
    file is safe line by line (append mode, one write per line). Idempotent for the same ``setup``.

    Parameters
    ----------
    setup : dict
        The output of :func:`logging_setup` in the parent process.
    """
    global _ATTACHED
    if setup == _ATTACHED:
        return
    configure_logging(setup.get("level") or "INFO")
    capture_python_warnings()
    _FILE_SINKS.clear()               # configure_logging removed every sink, the files' included
    for path in setup.get("files") or []:
        sink_id = _logger.add(path, level="DEBUG", format=_FILE_FORMAT, colorize=False, enqueue=False,
                              backtrace=True, diagnose=False)
        _FILE_SINKS[str(Path(path).parent.resolve())] = {"path": Path(path), "id": sink_id}
    _ATTACHED = dict(setup)


def get_logger():
    """Return the shared vlbipy loguru logger, configuring it on first use.

    Returns
    -------
    loguru.Logger
        The process-wide loguru logger.
    """
    if not _CONFIGURED:
        configure_logging()

    return _logger


class WarningCollector:
    """Collect warnings and anomalies during a run for an end-of-run summary."""

    def __init__(self) -> None:
        self._items: list[str] = []
        self._keyed: dict = {}

    def warn(self, message: str, *, key=None) -> None:
        """Log a warning and record it for the summary.

        ``key``: a warning that a later pass repeats with updated numbers (the fringe fit of every calibration
        pass judges the weak antennas again). The run log keeps every one; the summary keeps the latest, in the
        place of the first, so the list neither grows with the passes nor shifts.
        """
        get_logger().warning(message)
        if key is not None and key in self._keyed:
            self._items[self._keyed[key]] = message
            return
        if key is not None:
            self._keyed[key] = len(self._items)
        self._items.append(message)

    def anomaly(self, message: str, *, console: bool = True, record: bool = True) -> None:
        """Log a prominent anomaly (shown on the console at every level) and record it for the summary.

        ``console=False`` leaves it to the run log (one console line sums up a group of them); ``record=False``
        keeps that summing-up line out of the summary, which lists the group's members.
        """
        (get_logger().bind(anomaly=True) if console else get_logger()).warning("ANOMALY: {}", message)
        if record:
            self._items.append("ANOMALY: " + message)

    def summary(self) -> list[str]:
        """Return the list of recorded warnings and anomalies."""
        return list(self._items)

    def reset(self) -> None:
        """Clear all recorded items."""
        self._items.clear()
        self._keyed.clear()


# Module-level default collector for convenience.
warnings = WarningCollector()
