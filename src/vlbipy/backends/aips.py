"""AIPS backend — STUB (deferred; PRD issue E8).

The AIPS backend (via ParselTongue) would implement the
:class:`~vlbipy.backends.base.Backend` interface for legacy workflows. It is not
implemented in the API-first milestone; instantiating it raises a clear error.
"""
from __future__ import annotations

from ..errors import BackendError
from .base import Backend


class AipsBackend(Backend):
    """Placeholder for the AIPS backend (not implemented yet)."""

    kind = "aips"

    def __init__(self, *args, **kwargs) -> None:
        raise BackendError(
            "The AIPS backend is not implemented yet (PRD issue E8). "
            "Use backend='dummy' to exercise the API in-memory.")
