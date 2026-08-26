"""Configuration loading for vlbipy.

Implements the three-layer TOML cascade: built-in ``defaults.toml`` -> user file
or dict -> explicit overrides (later layers win). Also maps convenience
constructor keyword arguments onto the nested config structure.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional, Union

from .errors import ConfigError
from .logging_utils import get_logger

logger = get_logger()

ConfigLike = Union[str, Path, dict, None]

DEFAULTS_FILE = Path(__file__).resolve().parent / "templates" / "defaults.toml"


def _read_toml(path: Path) -> dict:
    """Read a TOML file into a dict, raising :class:`ConfigError` on failure."""
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config: ConfigLike = None, overrides: Optional[dict] = None) -> dict:
    """Load the fully-merged vlbipy configuration.

    Parameters
    ----------
    config : str or pathlib.Path or dict or None
        User configuration: a path to a TOML file, an already-parsed dict, or
        ``None`` to use only the built-in defaults.
    overrides : dict, optional
        Highest-priority overrides (e.g. from constructor kwargs / CLI).

    Returns
    -------
    dict
        The merged configuration.

    Raises
    ------
    ConfigError
        If a file is missing or contains invalid TOML, or ``config`` has an
        unsupported type.
    """
    merged = _read_toml(DEFAULTS_FILE)
    if isinstance(config, (str, Path)):
        merged = _deep_merge(merged, _read_toml(Path(config)))
        logger.debug("merged user config from {}", config)
    elif isinstance(config, dict):
        merged = _deep_merge(merged, config)
    elif config is not None:
        raise ConfigError(f"unsupported config type: {type(config)!r}")
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


_SOURCE_KWARGS = {
    "target": "targets",
    "phasecal": "phase_calibrators",
    "fringe_finder": "fringe_finders",
    "check_source": "check_sources",
}


def kwargs_to_overrides(**kwargs) -> dict:
    """Map convenience constructor kwargs onto the nested config structure.

    Recognised keys: ``network``, ``backend``, ``mode``, ``refant`` (str or
    list), and the source roles ``target`` / ``phasecal`` / ``fringe_finder`` /
    ``check_source`` (each str or list). ``None`` values are ignored.

    Returns
    -------
    dict
        A partial config suitable as the ``overrides`` argument of
        :func:`load_config`.
    """
    glob: dict = {}
    if kwargs.get("network") is not None:
        glob["observatory"] = kwargs["network"]
    if kwargs.get("backend") is not None:
        glob["backend"] = kwargs["backend"]
    if kwargs.get("mode") is not None:
        glob["mode"] = kwargs["mode"]
    if kwargs.get("refant") is not None:
        refant = kwargs["refant"]
        glob["reference_antenna"] = [refant] if isinstance(refant, str) else list(refant)

    sources: dict = {}
    for kw, cfg_key in _SOURCE_KWARGS.items():
        value = kwargs.get(kw)
        if value is not None:
            sources[cfg_key] = [value] if isinstance(value, str) else list(value)

    out: dict = {}
    if glob:
        out["global"] = glob
    if sources:
        out["sources"] = sources
    return out
