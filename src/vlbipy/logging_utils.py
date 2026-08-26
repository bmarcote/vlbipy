"""Logging utilities for vlbipy, built on loguru.

Provides a configured colored logger (orange/yellow warnings, red-bold errors)
and a :class:`WarningCollector` that aggregates warnings and anomalies during a
run so a summary can be printed at the end and embedded in reports.

loguru is used per project convention for this (complex) program. Configuration
is centralised here so the logging backend stays swappable.
"""
import datetime as dt
import sys
from pathlib import Path
from loguru import logger as _logger

_CONFIGURED = False
_FORMAT = "<level>{message}</level>"
#: File sink format: timestamped and level-tagged, so a run can be audited afterwards.
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}"
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
    global _CONFIGURED
    _logger.remove()
    _logger.level("WARNING", color="<yellow>")
    _logger.level("ERROR", color="<red><bold>")
    _logger.add(sys.stderr, level=level, format=_FORMAT, colorize=True)
    _CONFIGURED = True


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
    _logger.info("logging this run to {}", path)
    return path


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

    def warn(self, message: str) -> None:
        """Log a warning and record it for the summary."""
        get_logger().warning(message)
        self._items.append(message)

    def anomaly(self, message: str) -> None:
        """Log a prominent anomaly and record it for the summary."""
        get_logger().warning("ANOMALY: {}", message)
        self._items.append("ANOMALY: " + message)

    def summary(self) -> list[str]:
        """Return the list of recorded warnings and anomalies."""
        return list(self._items)

    def reset(self) -> None:
        """Clear all recorded items."""
        self._items.clear()


# Module-level default collector for convenience.
warnings = WarningCollector()
