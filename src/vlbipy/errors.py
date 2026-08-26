"""Exception hierarchy for vlbipy.

All vlbipy-specific errors derive from :class:`Error`, so callers can catch
the whole family with a single ``except``. Errors are explicit and carry enough
context (step name, parameters, data selection) to debug a failed run after the
fact.
"""

class BackendError(Exception):
    """Raised when configuration is missing, unreadable, or invalid."""

class ConfigError(Exception):
    """Raised when configuration is missing, unreadable, or invalid."""


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
