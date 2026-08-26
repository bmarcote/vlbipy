"""Backend registry for vlbipy.

Backends are selected by name via :func:`get_backend`; the API layer never
imports a concrete backend directly. Only the dummy backend is implemented in
the API-first milestone; ``casa`` and ``aips`` are stubs that raise on use.
"""
from __future__ import annotations

from ..errors import BackendError
from .base import Backend
from .dummy import DummyBackend


def get_backend(kind: str, **kwargs) -> Backend:
    """Return a backend instance for the given kind.

    Parameters
    ----------
    kind : str
        Backend identifier: ``"dummy"`` (always available), ``"casa"``, ``"aips"``,
        or ``"dask-ms"`` (require their respective packages to be installed).
    **kwargs
        Passed to the backend constructor (e.g. ``work_dir``).

    Returns
    -------
    Backend

    Raises
    ------
    BackendError
        If the kind is unknown, not implemented, or its package is missing.
    """
    key = str(kind).lower()
    if key == "dummy":
        return DummyBackend(**kwargs)
    if key == "casa":
        from .casa import CasaBackend
        return CasaBackend(**kwargs)
    if key == "aips":
        from .aips import AipsBackend
        return AipsBackend(**kwargs)
    if key in ("dask-ms", "daskms"):
        from .dask_ms import DaskMsBackend
        return DaskMsBackend(**kwargs)
    raise BackendError(f"unknown backend {kind!r} (available: dummy, casa, aips, dask-ms)")


__all__ = ["Backend", "DummyBackend", "get_backend"]
