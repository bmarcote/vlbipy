"""Exception hierarchy for vlbipy.

All vlbipy-specific errors derive from :class:`Error`, so callers can catch
the whole family with a single ``except``. Errors are explicit and carry enough
context (step name, parameters, data selection) to debug a failed run after the
fact.
"""

class BackendError(Exception):
    """Raised when a backend cannot be instantiated or is misconfigured."""


class EmptySelectionError(BackendError):
    """Raised when a solve finds no usable (unflagged) data in its selection.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    scan : int, optional
        The single scan the solve was restricted to (None when it covered several or none).
    """

    def __init__(self, message: str, scan: int | None = None) -> None:
        self.scan: int | None = scan
        super().__init__(message)


class ConfigError(Exception):
    """Raised when configuration is missing, unreadable, or invalid."""


class PluginError(Exception):
    """Raised when a plugin cannot be loaded, is invalid, or conflicts with an existing registration."""


class SourceNotFoundError(Exception):
    """Raised when a requested source name or role cannot be resolved."""


class StepError(Exception):
    """Raised when a pipeline step fails.

    Parameters
    ----------
    step : str
        Name of the pipeline step that failed.
    message : str
        Human-readable description of the failure.
    params : dict, optional
        Resolved parameters passed to the step (for debugging).
    selection : str, optional
        Data selection in effect (e.g. field/spw/timerange).
    """

    def __init__(self, step: str, message: str, params: dict | None = None, selection: str = "") -> None:
        self.step: str = step
        self.params: dict = params or {}
        self.selection: str = selection
        detail = f"[step={step}]"
        if selection:
            detail += f" [selection={selection}]"

        if self.params:
            detail += f" [params={self.params}]"

        super().__init__(f"{message} {detail}")


class CampaignError(Exception):
    """Raised at the end of a multi-project run in which some projects failed.

    A project that fails is left out of the rest of the run while the others carry on to the end; this
    error then names every project that failed and why. ``failures`` maps each project code to its exception.
    """

    def __init__(self, failures: dict) -> None:
        self.failures: dict = dict(failures)
        super().__init__("; ".join(str(exc) if str(exc).startswith(f"{code}:") else f"{code}: {exc}"
                                   for code, exc in self.failures.items()))


class StaleCalibrationError(StepError):
    """Raised when legacy calibration provenance makes continuing the pipeline unsafe.

    Uses the :class:`StepError` constructor and context fields, but must propagate
    past best-effort step handlers so stale gains cannot enter final products.
    """
