"""Backend registry for vlbipy.

Backends are selected by name via :func:`get_backend`; the API layer never
imports a concrete backend directly. Built-in backends (``dummy``, ``casa``,
``aips``, ``dask-ms``) are registered lazily; third-party backends are
discovered via entry points (``vlbipy.backends``), :func:`register_backend`,
or ``module:qualname`` references. See :mod:`vlbipy.registry` for details.
"""
from __future__ import annotations

from ..errors import BackendError
from ..registry import get_backend, list_backends, register_backend  # noqa: F401
from .base import Backend
from .dummy import DummyBackend


__all__ = ["Backend", "DummyBackend", "get_backend", "register_backend", "list_backends"]
