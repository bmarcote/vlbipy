"""Observatory-handler registry for vlbipy."""
from __future__ import annotations

from ..errors import ConfigError
from .base import ObservatoryHandler
from .evn import EVNObservatory
from .lba import LBAObservatory
from .vlba import VLBAObservatory

_HANDLERS = {
    "EVN": EVNObservatory,
    "VLBA": VLBAObservatory,
    "LBA": LBAObservatory,
}


def get_observatory_handler(name: str) -> ObservatoryHandler:
    """Return the handler for a network name.

    Parameters
    ----------
    name : str
        ``"EVN"``, ``"VLBA"`` or ``"LBA"`` (case-insensitive).

    Returns
    -------
    ObservatoryHandler

    Raises
    ------
    ConfigError
        If the network is unknown.
    """
    cls = _HANDLERS.get(str(name).upper())
    if cls is None:
        raise ConfigError(f"unknown observatory {name!r} (available: {', '.join(_HANDLERS)})")
    return cls()


__all__ = ["ObservatoryHandler", "get_observatory_handler",
           "EVNObservatory", "VLBAObservatory", "LBAObservatory"]
