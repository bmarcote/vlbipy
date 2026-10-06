"""Observatory-handler registry for vlbipy.

Observatories are selected by name via :func:`get_observatory_handler`.
Built-in networks (EVN, VLBA, LBA) are registered lazily; third-party
observatories are discovered via entry points (``vlbipy.observatories``),
:func:`register_observatory`, or ``module:qualname`` references.
See :mod:`vlbipy.registry` for details.
"""
from __future__ import annotations

from ..errors import ConfigError
from ..registry import (  # noqa: F401
    get_observatory_handler,
    list_observatories,
    register_observatory,
)
from .base import ObservatoryHandler
from .evn import EVNObservatory
from .lba import LBAObservatory
from .vlba import VLBAObservatory


__all__ = ["ObservatoryHandler", "get_observatory_handler",
           "register_observatory", "list_observatories",
           "EVNObservatory", "VLBAObservatory", "LBAObservatory"]
