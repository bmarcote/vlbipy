"""The package version, taken from ``pyproject.toml`` — the only place it is written.

Two situations, one source of truth:

* **source checkout / editable install**: the ``pyproject.toml`` two directories
  above this file is read directly, so a version bump shows up at once, without
  reinstalling;
* **installed wheel**: no ``pyproject.toml`` is shipped, so the version comes from
  the distribution metadata, which the build wrote from that same
  ``[project].version``.

Never hard-code the version anywhere else: import ``__version__`` from here (or
from ``vlbipy``).
"""
from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path
from typing import Optional

#: Distribution name: the ``[project].name`` a ``pyproject.toml`` must carry to be ours.
DISTRIBUTION = "vlbipy"
#: Reported when neither a matching ``pyproject.toml`` nor the installed metadata is found.
UNKNOWN_VERSION = "unknown"


def pyproject_version(pyproject: Path) -> str:
    """Return ``[project].version`` of ``pyproject`` when that file describes this package, else ``""``.

    A missing or unreadable file, a file of another project (``[project].name`` differs) and a
    project without a static version all give ``""``: the caller then falls back to the
    installed metadata.
    """
    if not pyproject.is_file():
        return ""
    try:
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    if project.get("name") != DISTRIBUTION:
        return ""
    return str(project.get("version") or "")


def read_version(pyproject: Optional[Path] = None) -> str:
    """Return the package version: from ``pyproject.toml`` when it is there, else from the installed metadata.

    Parameters
    ----------
    pyproject : pathlib.Path, optional
        The ``pyproject.toml`` to read; defaults to the one of the source tree this module lives in
        (``<root>/src/vlbipy/_version.py`` -> ``<root>/pyproject.toml``).

    Returns
    -------
    str
        The version string, or :data:`UNKNOWN_VERSION` when the package is neither run from its
        source tree nor installed.
    """
    path = Path(__file__).resolve().parents[2] / "pyproject.toml" if pyproject is None else Path(pyproject)
    version = pyproject_version(path)
    if version:
        return version
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = read_version()
