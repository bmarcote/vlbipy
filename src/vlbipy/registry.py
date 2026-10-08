"""Lazy plugin registries for backends and observatories.

Third-party packages provide vlbipy plugins in three ways (any combination):

1. **Entry points** declared in ``pyproject.toml``::

       [project.entry-points."vlbipy.backends"]
       mybackend = "mypkg.backend:MyBackend"

       [project.entry-points."vlbipy.observatories"]
       mynet = "mypkg.observatory:MyObservatory"

2. **Programmatic registration** (e.g. in a package's ``__init__``)::

       from vlbipy.registry import register_backend, register_observatory
       register_backend("mybackend", MyBackend)
       register_observatory("MYNET", MyObservatoryHandler)

3. **Configuration import bridge** — the pipeline config lists modules that are
   imported (triggering their registrations) before name resolution::

       [global]
       plugins = ["mypkg.vlbipy_plugin"]
       backend = "mybackend"
       observatory = "MYNET"

   A ``module:qualname`` string (e.g. ``"mypkg.backend:MyBackend"``) can also be
   used directly as a backend or observatory value, bypassing the registry.

Registries are module-level singletons. Builtins are registered as lazy
callables so their heavyweight dependencies (casatools, parseltongue, dask-ms)
are never imported until the backend is actually requested.
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Union

from .errors import BackendError, ConfigError, PluginError

logger = logging.getLogger("vlbipy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_qualname(reference: str):
    """Import ``module:qualname`` and return the object.

    Parameters
    ----------
    reference : str
        A string of the form ``"some.module:SomeClass"`` or
        ``"some.module:SomeClass.attr"``.

    Returns
    -------
    object
        The resolved Python object.

    Raises
    ------
    PluginError
        If the format is invalid or the import/attribute lookup fails.
    """
    if ":" not in reference:
        raise PluginError(f"invalid module:qualname reference {reference!r} (missing ':')")
    module_path, qualname = reference.split(":", 1)
    if not module_path or not qualname:
        raise PluginError(f"invalid module:qualname reference {reference!r}")
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise PluginError(f"cannot import module {module_path!r} from "
                          f"reference {reference!r}: {exc}") from exc
    obj = mod
    for attr in qualname.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            raise PluginError(f"cannot resolve {qualname!r} in module "
                              f"{module_path!r}: {exc}") from exc
    return obj


def _load_entry_points(group: str) -> dict[str, Callable]:
    """Discover installed entry points for *group* and return ``{name: loader}``.

    Each loader is a zero-argument callable that returns the entry point's object
    (a Backend subclass or ObservatoryHandler subclass). The entry point's
    module is imported only when the loader is called, keeping optional
    dependencies lazy.

    Works with both the modern ``importlib.metadata`` API (Python >= 3.9 /
    importlib_metadata >= 3.6, where ``entry_points`` accepts the ``group``
    keyword) and older versions that return a dict or a flat list.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:                             # pragma: no cover – Python < 3.8
        return {}

    discovered: dict[str, Callable] = {}
    try:
        # Python >= 3.12 / importlib_metadata >= 3.6: keyword form.
        eps = entry_points(group=group)
    except TypeError:
        # Older API: entry_points() returns a dict keyed by group.
        all_eps = entry_points()
        eps = all_eps.get(group, []) if isinstance(all_eps, dict) else []

    for ep in eps:
        # Wrap in a factory so each ep.load() runs in its own closure.
        def _make_loader(_ep=ep):
            return _ep.load()
        discovered[ep.name] = _make_loader
    return discovered


def load_plugins(plugin_paths: list[str]) -> None:
    """Import each dotted module path in *plugin_paths*.

    This is the configured import bridge: a module's top-level code can call
    :func:`register_backend` / :func:`register_observatory` to make its
    plugins available by name before the pipeline resolves backend/observatory
    strings.

    Parameters
    ----------
    plugin_paths : list of str
        Dotted Python module paths (e.g. ``["mypkg.vlbipy_plugin"]``).

    Raises
    ------
    PluginError
        If any module cannot be imported.
    """
    for path in plugin_paths:
        path = str(path).strip()
        if not path:
            continue
        try:
            importlib.import_module(path)
            logger.debug("loaded plugin module %s", path)
        except ImportError as exc:
            raise PluginError(f"cannot import plugin module {path!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

#: ``{name: class_or_callable}`` — values are either a Backend subclass or a
#: zero-argument callable that returns one (for lazy imports).
_backend_registry: dict[str, Union[type, Callable]] = {}

#: Whether entry points for backends have been loaded yet.
_backend_eps_loaded: bool = False


def _ensure_backend_builtins() -> None:
    """Register the four built-in backends lazily (run once)."""
    if _backend_registry:
        return  # already populated (or partially by register_backend)

    def _dummy():
        from .backends.dummy import DummyBackend
        return DummyBackend

    def _casa():
        from .backends.casa import CasaBackend
        return CasaBackend

    def _aips():
        from .backends.aips import AipsBackend
        return AipsBackend

    def _daskms():
        from .backends.dask_ms import DaskMsBackend
        return DaskMsBackend

    for name, loader in [("dummy", _dummy), ("casa", _casa), ("aips", _aips), ("dask-ms", _daskms),
                         ("daskms", _daskms)]:
        _backend_registry.setdefault(name, loader)


def _load_backend_eps() -> None:
    """Load entry points for ``vlbipy.backends`` once."""
    global _backend_eps_loaded
    if _backend_eps_loaded:
        return
    _backend_eps_loaded = True
    for name, loader in _load_entry_points("vlbipy.backends").items():
        key = name.lower()
        if key in _backend_registry:
            logger.warning("entry-point backend %r shadowed by an existing registration", name)
            continue
        _backend_registry[key] = loader


def register_backend(name: str, backend_class) -> None:
    """Register a backend class under *name*.

    Parameters
    ----------
    name : str
        Short identifier (case-insensitive). Must not collide with an already-
        registered name.
    backend_class : type
        A :class:`~vlbipy.backends.base.Backend` subclass.

    Raises
    ------
    PluginError
        If *name* is already registered or *backend_class* is not a valid
        Backend subclass.
    """
    from .backends.base import Backend
    if not (isinstance(backend_class, type) and issubclass(backend_class, Backend)):
        raise PluginError(f"register_backend({name!r}): expected a Backend subclass, "
                          f"got {backend_class!r}")
    key = str(name).lower()
    _ensure_backend_builtins()
    if key in _backend_registry:
        raise PluginError(f"backend {name!r} is already registered")
    _backend_registry[key] = backend_class


def get_backend(kind: str, **kwargs):
    """Return a backend instance for the given *kind*.

    Resolution order:

    1. Exact match in the registry (builtins + ``register_backend`` +
       entry points).
    2. ``module:qualname`` direct reference (e.g.
       ``"mypkg.backend:MyBackend"``).

    Parameters
    ----------
    kind : str
        Backend identifier.
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
    _ensure_backend_builtins()
    key = str(kind).lower()
    entry = _backend_registry.get(key)
    if entry is None:
        # Try entry points before giving up.
        _load_backend_eps()
        entry = _backend_registry.get(key)
    if entry is None and ":" in kind:
        # Direct module:qualname reference.
        try:
            cls = _resolve_qualname(kind)
        except PluginError as exc:
            raise BackendError(str(exc)) from exc
        from .backends.base import Backend
        if not (isinstance(cls, type) and issubclass(cls, Backend)):
            raise BackendError(f"{kind!r} resolved to {cls!r}, which is not a Backend subclass")
        return cls(**kwargs)
    if entry is None:
        available = sorted(_backend_registry)
        raise BackendError(f"unknown backend {kind!r} (available: {', '.join(available)})")
    # entry is either a class or a lazy loader (callable returning a class).
    cls = entry() if callable(entry) and not isinstance(entry, type) else entry
    return cls(**kwargs)


def list_backends() -> list[str]:
    """Return the names of all registered backends (including entry-point ones).

    Entry points are loaded if they haven't been yet, so this is the
    authoritative list.
    """
    _ensure_backend_builtins()
    _load_backend_eps()
    return sorted(_backend_registry)


# ---------------------------------------------------------------------------
# Observatory registry
# ---------------------------------------------------------------------------

#: ``{NAME: class_or_callable}`` — upper-cased keys; values are either an
#: ObservatoryHandler subclass or a zero-argument callable that returns one.
_observatory_registry: dict[str, Union[type, Callable]] = {}

#: Whether entry points for observatories have been loaded yet.
_observatory_eps_loaded: bool = False


def _ensure_observatory_builtins() -> None:
    """Register the three built-in observatories lazily (run once)."""
    if _observatory_registry:
        return

    def _evn():
        from .observatories.evn import EVNObservatory
        return EVNObservatory

    def _vlba():
        from .observatories.vlba import VLBAObservatory
        return VLBAObservatory

    def _lba():
        from .observatories.lba import LBAObservatory
        return LBAObservatory

    for name, loader in [("EVN", _evn), ("VLBA", _vlba), ("LBA", _lba)]:
        _observatory_registry.setdefault(name, loader)


def _load_observatory_eps() -> None:
    """Load entry points for ``vlbipy.observatories`` once."""
    global _observatory_eps_loaded
    if _observatory_eps_loaded:
        return
    _observatory_eps_loaded = True
    for name, loader in _load_entry_points("vlbipy.observatories").items():
        key = name.upper()
        if key in _observatory_registry:
            logger.warning("entry-point observatory %r shadowed by an existing registration", name)
            continue
        _observatory_registry[key] = loader


def register_observatory(name: str, handler_class) -> None:
    """Register an observatory handler class under *name*.

    Parameters
    ----------
    name : str
        Network name (case-insensitive, stored upper-cased). Must not collide
        with an already-registered name.
    handler_class : type
        An :class:`~vlbipy.observatories.base.ObservatoryHandler` subclass.

    Raises
    ------
    PluginError
        If *name* is already registered or *handler_class* is not a valid
        ObservatoryHandler subclass.
    """
    from .observatories.base import ObservatoryHandler
    if not (isinstance(handler_class, type) and issubclass(handler_class, ObservatoryHandler)):
        raise PluginError(f"register_observatory({name!r}): expected an ObservatoryHandler "
                          f"subclass, got {handler_class!r}")
    key = str(name).upper()
    _ensure_observatory_builtins()
    if key in _observatory_registry:
        raise PluginError(f"observatory {name!r} is already registered")
    _observatory_registry[key] = handler_class


def get_observatory_handler(name: str):
    """Return an observatory handler instance for the given *name*.

    Resolution order:

    1. Exact match in the registry (builtins + ``register_observatory`` +
       entry points).
    2. ``module:qualname`` direct reference (e.g.
       ``"mypkg.obs:MyObservatory"``).

    Parameters
    ----------
    name : str
        Network name (case-insensitive).

    Returns
    -------
    ObservatoryHandler

    Raises
    ------
    ConfigError
        If the network is unknown.
    """
    _ensure_observatory_builtins()
    key = str(name).upper()
    entry = _observatory_registry.get(key)
    if entry is None:
        _load_observatory_eps()
        entry = _observatory_registry.get(key)
    if entry is None and ":" in name:
        try:
            cls = _resolve_qualname(name)
        except PluginError as exc:
            raise ConfigError(str(exc)) from exc
        from .observatories.base import ObservatoryHandler
        if not (isinstance(cls, type) and issubclass(cls, ObservatoryHandler)):
            raise ConfigError(f"{name!r} resolved to {cls!r}, which is not an "
                              f"ObservatoryHandler subclass")
        return cls()
    if entry is None:
        available = sorted(_observatory_registry)
        raise ConfigError(f"unknown observatory {name!r} (available: {', '.join(available)})")
    cls = entry() if callable(entry) and not isinstance(entry, type) else entry
    return cls()


def list_observatories() -> list[str]:
    """Return the names of all registered observatories (including entry-point ones).

    Entry points are loaded if they haven't been yet, so this is the
    authoritative list.
    """
    _ensure_observatory_builtins()
    _load_observatory_eps()
    return sorted(_observatory_registry)


# ---------------------------------------------------------------------------
# Testing helpers
# ---------------------------------------------------------------------------

def _reset_registries() -> None:
    """Clear all registries and entry-point caches (for testing only)."""
    global _backend_eps_loaded, _observatory_eps_loaded
    _backend_registry.clear()
    _observatory_registry.clear()
    _backend_eps_loaded = False
    _observatory_eps_loaded = False


__all__ = [
    "register_backend", "register_observatory",
    "get_backend", "get_observatory_handler",
    "list_backends", "list_observatories",
    "load_plugins",
]
