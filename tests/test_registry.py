"""Tests for the plugin registry: backends, observatories, entry points, config bridge, CLI.

All tests use in-memory fake modules and monkeypatched entry points, so no
external packages or CASA are needed.
"""
from __future__ import annotations

import sys
import types

import pytest

from vlbipy.backends.base import Backend, DataOps
from vlbipy.errors import BackendError, ConfigError, PluginError
from vlbipy.observatories.base import ObservatoryHandler
from vlbipy.registry import (
    _reset_registries,
    get_backend,
    get_observatory_handler,
    list_backends,
    list_observatories,
    load_plugins,
    register_backend,
    register_observatory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset the registries before and after every test so they don't leak."""
    _reset_registries()
    yield
    _reset_registries()


# ---------------------------------------------------------------------------
# Fake backend / observatory for testing
# ---------------------------------------------------------------------------

class FakeDataOps(DataOps):
    """Minimal data ops for testing."""

    def get_metadata(self, project_code, source_names, observatory):
        """Return a stub metadata (enough for construction tests)."""
        from vlbipy.models import ObsMetadata
        return ObsMetadata(project_code=project_code)


class FakeBackend(Backend):
    """Minimal backend for plugin registration tests."""

    kind = "fake"
    requires_data_files = False
    data_ops = FakeDataOps


class AnotherFakeBackend(Backend):
    """Second fake backend for duplicate-detection tests."""

    kind = "fake2"
    requires_data_files = False


class FakeObservatory(ObservatoryHandler):
    """Minimal observatory handler for plugin tests."""

    name = "FAKENET"
    auto_download = False


class AnotherFakeObservatory(ObservatoryHandler):
    """Second fake observatory for duplicate tests."""

    name = "FAKENET2"


# ---------------------------------------------------------------------------
# Builtin resolution
# ---------------------------------------------------------------------------

class TestBuiltinBackends:
    """The four built-in backends resolve by name, case-insensitively."""

    def test_dummy(self):
        """The dummy backend resolves and constructs without external deps."""
        be = get_backend("dummy")
        assert be.kind == "dummy"

    def test_dummy_case_insensitive(self):
        """Backend names are case-insensitive."""
        assert get_backend("Dummy").kind == "dummy"
        assert get_backend("DUMMY").kind == "dummy"

    def test_daskms_aliases(self):
        """Both 'dask-ms' and 'daskms' resolve to the same backend."""
        try:
            be1 = get_backend("dask-ms")
            be2 = get_backend("daskms")
            assert type(be1) is type(be2)
        except BackendError:
            pytest.skip("dask-ms not installed")

    def test_unknown_raises(self):
        """An unregistered name raises BackendError with available list."""
        with pytest.raises(BackendError, match="unknown backend.*nope"):
            get_backend("nope")

    def test_list_backends_includes_builtins(self):
        """list_backends returns at least the four built-in names."""
        names = list_backends()
        for builtin in ("dummy", "casa", "aips", "dask-ms", "daskms"):
            assert builtin in names


class TestBuiltinObservatories:
    """The three built-in observatories resolve by name."""

    def test_evn(self):
        """EVN resolves to the EVNObservatory handler."""
        handler = get_observatory_handler("EVN")
        assert handler.name == "EVN"

    def test_vlba(self):
        """VLBA resolves to the VLBAObservatory handler."""
        assert get_observatory_handler("VLBA").name == "VLBA"

    def test_lba(self):
        """LBA resolves to the LBAObservatory handler."""
        assert get_observatory_handler("LBA").name == "LBA"

    def test_case_insensitive(self):
        """Observatory names are case-insensitive."""
        assert get_observatory_handler("evn").name == "EVN"
        assert get_observatory_handler("Evn").name == "EVN"

    def test_unknown_raises(self):
        """An unregistered name raises ConfigError with available list."""
        with pytest.raises(ConfigError, match="unknown observatory.*NOPE"):
            get_observatory_handler("NOPE")

    def test_list_observatories_includes_builtins(self):
        """list_observatories returns at least the three built-in names."""
        names = list_observatories()
        for builtin in ("EVN", "LBA", "VLBA"):
            assert builtin in names


# ---------------------------------------------------------------------------
# Programmatic registration
# ---------------------------------------------------------------------------

class TestRegisterBackend:
    """register_backend adds a backend to the registry."""

    def test_register_and_resolve(self):
        """A registered backend resolves by name and constructs."""
        register_backend("fake", FakeBackend)
        be = get_backend("fake")
        assert isinstance(be, FakeBackend) and be.kind == "fake"

    def test_duplicate_raises(self):
        """Re-registering the same name raises PluginError."""
        register_backend("fake", FakeBackend)
        with pytest.raises(PluginError, match="already registered"):
            register_backend("fake", AnotherFakeBackend)

    def test_non_backend_raises(self):
        """Registering a non-Backend class raises PluginError."""
        with pytest.raises(PluginError, match="Backend subclass"):
            register_backend("bad", str)

    def test_non_class_raises(self):
        """Registering an instance (not a class) raises PluginError."""
        with pytest.raises(PluginError, match="Backend subclass"):
            register_backend("bad", FakeBackend())

    def test_builtin_name_collides(self):
        """Registering over a built-in name raises PluginError."""
        with pytest.raises(PluginError, match="already registered"):
            register_backend("dummy", FakeBackend)


class TestRegisterObservatory:
    """register_observatory adds an observatory to the registry."""

    def test_register_and_resolve(self):
        """A registered observatory resolves by name and constructs."""
        register_observatory("FAKENET", FakeObservatory)
        handler = get_observatory_handler("FAKENET")
        assert isinstance(handler, FakeObservatory) and handler.name == "FAKENET"

    def test_duplicate_raises(self):
        """Re-registering the same name raises PluginError."""
        register_observatory("FAKENET", FakeObservatory)
        with pytest.raises(PluginError, match="already registered"):
            register_observatory("FAKENET", AnotherFakeObservatory)

    def test_non_handler_raises(self):
        """Registering a non-ObservatoryHandler class raises PluginError."""
        with pytest.raises(PluginError, match="ObservatoryHandler subclass"):
            register_observatory("BAD", int)

    def test_builtin_name_collides(self):
        """Registering over a built-in name raises PluginError."""
        with pytest.raises(PluginError, match="already registered"):
            register_observatory("EVN", FakeObservatory)


# ---------------------------------------------------------------------------
# Entry-point discovery (monkeypatched)
# ---------------------------------------------------------------------------

def _make_fake_entry_point(name, obj):
    """Return an object that behaves like importlib.metadata.EntryPoint.

    ``ep.name`` is the registered name and ``ep.load()`` returns *obj*.
    """
    class FakeEP:
        pass

    ep = FakeEP()
    ep.name = name
    ep.load = lambda: obj
    return ep


class TestEntryPoints:
    """Entry-point discovery loads plugins lazily on first cache miss."""

    def test_backend_entry_point(self, monkeypatch):
        """A backend entry point is discovered when its name is requested."""
        eps = {
            "vlbipy.backends": [_make_fake_entry_point("epbackend", FakeBackend)],
            "vlbipy.observatories": [],
        }
        monkeypatch.setattr(
            "vlbipy.registry._load_entry_points",
            lambda group: {ep.name: ep.load for ep in eps.get(group, [])},
        )
        be = get_backend("epbackend")
        assert isinstance(be, FakeBackend)

    def test_observatory_entry_point(self, monkeypatch):
        """An observatory entry point is discovered when its name is requested."""
        eps = {
            "vlbipy.backends": [],
            "vlbipy.observatories": [_make_fake_entry_point("EPNET", FakeObservatory)],
        }
        monkeypatch.setattr(
            "vlbipy.registry._load_entry_points",
            lambda group: {ep.name: ep.load for ep in eps.get(group, [])},
        )
        handler = get_observatory_handler("EPNET")
        assert isinstance(handler, FakeObservatory)

    def test_entry_point_does_not_shadow_builtin(self, monkeypatch):
        """An entry point with a builtin name is ignored (logged, not fatal)."""
        eps = {"vlbipy.backends": [_make_fake_entry_point("dummy", FakeBackend)]}
        monkeypatch.setattr(
            "vlbipy.registry._load_entry_points",
            lambda group: {ep.name: ep.load for ep in eps.get(group, [])},
        )
        be = get_backend("dummy")
        # Still the real DummyBackend, not FakeBackend.
        assert be.kind == "dummy"
        assert not isinstance(be, FakeBackend)


# ---------------------------------------------------------------------------
# Config plugin import bridge
# ---------------------------------------------------------------------------

class TestConfigPlugins:
    """The [global].plugins config key imports modules before name resolution."""

    def test_plugin_module_registers_backend(self, monkeypatch):
        """A plugin module imported via config registers a backend that then resolves."""
        # Create a fake module whose import registers a backend.
        fake_mod = types.ModuleType("_vlbipy_test_plugin")
        fake_mod.__file__ = "<test>"

        def _on_import():
            register_backend("pluginbe", FakeBackend)

        _on_import()  # register directly to simulate what import would do
        monkeypatch.setitem(sys.modules, "_vlbipy_test_plugin", fake_mod)

        be = get_backend("pluginbe")
        assert isinstance(be, FakeBackend)

    def test_load_plugins_imports_modules(self, monkeypatch):
        """load_plugins imports the listed dotted paths."""
        imported = []
        monkeypatch.setattr(
            "vlbipy.registry.importlib.import_module",
            lambda path: imported.append(path) or types.ModuleType(path),
        )
        load_plugins(["pkg_a.plugin", "pkg_b.plugin"])
        assert imported == ["pkg_a.plugin", "pkg_b.plugin"]

    def test_load_plugins_bad_module_raises(self):
        """A module that does not exist raises PluginError."""
        with pytest.raises(PluginError, match="cannot import plugin module"):
            load_plugins(["no_such_vlbipy_plugin_module_xyz"])

    def test_load_plugins_skips_empty_strings(self, monkeypatch):
        """Empty or whitespace-only entries in the plugins list are skipped."""
        imported = []
        monkeypatch.setattr(
            "vlbipy.registry.importlib.import_module",
            lambda path: imported.append(path) or types.ModuleType(path),
        )
        load_plugins(["", "  ", "pkg.real"])
        assert imported == ["pkg.real"]

    def test_config_loads_plugins_before_resolution(self, monkeypatch):
        """load_config triggers load_plugins so backends registered there resolve."""
        from vlbipy.config import load_config

        # Simulate a plugin module that registers on import.
        fake_mod = types.ModuleType("_vlbipy_cfg_test_plugin")

        def fake_import(path):
            if path == "_vlbipy_cfg_test_plugin":
                register_backend("cfgplugin", FakeBackend)
                return fake_mod
            return types.ModuleType(path)

        monkeypatch.setattr("vlbipy.registry.importlib.import_module", fake_import)
        monkeypatch.setitem(sys.modules, "_vlbipy_cfg_test_plugin", fake_mod)
        cfg = load_config({"global": {"plugins": ["_vlbipy_cfg_test_plugin"]}})
        assert "plugins" in cfg["global"]
        # The plugin should have registered the backend.
        be = get_backend("cfgplugin")
        assert isinstance(be, FakeBackend)


# ---------------------------------------------------------------------------
# module:qualname direct references
# ---------------------------------------------------------------------------

class TestModuleQualname:
    """A 'module:ClassName' string resolves directly, bypassing the registry."""

    def test_backend_qualname(self, monkeypatch):
        """A backend specified as module:Class is imported and instantiated."""
        fake_mod = types.ModuleType("_fake_be_mod")
        fake_mod.MyBE = FakeBackend
        monkeypatch.setitem(sys.modules, "_fake_be_mod", fake_mod)
        be = get_backend("_fake_be_mod:MyBE")
        assert isinstance(be, FakeBackend)

    def test_observatory_qualname(self, monkeypatch):
        """An observatory specified as module:Class is imported and instantiated."""
        fake_mod = types.ModuleType("_fake_obs_mod")
        fake_mod.MyObs = FakeObservatory
        monkeypatch.setitem(sys.modules, "_fake_obs_mod", fake_mod)
        handler = get_observatory_handler("_fake_obs_mod:MyObs")
        assert isinstance(handler, FakeObservatory)

    def test_backend_qualname_bad_module(self):
        """A module:qualname with a non-importable module raises BackendError."""
        with pytest.raises(BackendError, match="cannot import"):
            get_backend("no_such_mod_xyz:Cls")

    def test_observatory_qualname_bad_module(self):
        """A module:qualname with a non-importable module raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot import"):
            get_observatory_handler("no_such_mod_xyz:Cls")

    def test_backend_qualname_not_backend(self, monkeypatch):
        """A module:qualname resolving to a non-Backend class raises BackendError."""
        fake_mod = types.ModuleType("_fake_notbe")
        fake_mod.NotBE = str
        monkeypatch.setitem(sys.modules, "_fake_notbe", fake_mod)
        with pytest.raises(BackendError, match="not a Backend subclass"):
            get_backend("_fake_notbe:NotBE")

    def test_observatory_qualname_not_handler(self, monkeypatch):
        """A module:qualname resolving to a non-ObservatoryHandler raises ConfigError."""
        fake_mod = types.ModuleType("_fake_notobs")
        fake_mod.NotObs = int
        monkeypatch.setitem(sys.modules, "_fake_notobs", fake_mod)
        with pytest.raises(ConfigError, match="not an.*ObservatoryHandler subclass"):
            get_observatory_handler("_fake_notobs:NotObs")

    def test_qualname_bad_attr(self, monkeypatch):
        """A module:qualname with a missing attribute raises BackendError."""
        fake_mod = types.ModuleType("_fake_noattr")
        monkeypatch.setitem(sys.modules, "_fake_noattr", fake_mod)
        with pytest.raises(BackendError, match="cannot resolve"):
            get_backend("_fake_noattr:NoSuchClass")

    def test_nested_qualname(self, monkeypatch):
        """A module:Outer.Inner qualname resolves nested attributes."""
        fake_mod = types.ModuleType("_fake_nested")

        class Outer:
            Inner = FakeBackend

        fake_mod.Outer = Outer
        monkeypatch.setitem(sys.modules, "_fake_nested", fake_mod)
        be = get_backend("_fake_nested:Outer.Inner")
        assert isinstance(be, FakeBackend)


# ---------------------------------------------------------------------------
# CLI: arbitrary network names
# ---------------------------------------------------------------------------

class TestCLIArbitraryNetwork:
    """The CLI --network flag accepts any string (not a hardcoded choices list)."""

    def test_cli_accepts_builtin_network(self):
        """Built-in network names still parse without error."""
        from vlbipy.cli import build_parser
        args = build_parser().parse_args(["pipeline", "-p", "X", "--network", "EVN"])
        assert args.network == "EVN"

    def test_cli_accepts_plugin_network(self):
        """An arbitrary (plugin) network name parses without error."""
        from vlbipy.cli import build_parser
        args = build_parser().parse_args(["pipeline", "-p", "X", "--network", "FAKENET"])
        assert args.network == "FAKENET"

    def test_cli_accepts_qualname_network(self):
        """A module:qualname observatory reference parses as --network."""
        from vlbipy.cli import build_parser
        args = build_parser().parse_args(
            ["pipeline", "-p", "X", "--network", "mypkg.obs:MyObs"]
        )
        assert args.network == "mypkg.obs:MyObs"


# ---------------------------------------------------------------------------
# Combined backend + observatory from one module
# ---------------------------------------------------------------------------

class TestCombinedPlugin:
    """A single plugin module can register both a backend and an observatory."""

    def test_combined_registration(self):
        """Registering both from one call site works and both resolve."""
        register_backend("combo", FakeBackend)
        register_observatory("COMBONET", FakeObservatory)

        be = get_backend("combo")
        obs = get_observatory_handler("COMBONET")
        assert isinstance(be, FakeBackend)
        assert isinstance(obs, FakeObservatory)


# ---------------------------------------------------------------------------
# Existing API compatibility
# ---------------------------------------------------------------------------

class TestExistingAPICompat:
    """The existing public API surface is preserved."""

    def test_get_backend_from_backends_init(self):
        """``from vlbipy.backends import get_backend`` still works."""
        from vlbipy.backends import get_backend as gb
        assert gb("dummy").kind == "dummy"

    def test_get_observatory_handler_from_observatories_init(self):
        """``from vlbipy.observatories import get_observatory_handler`` still works."""
        from vlbipy.observatories import get_observatory_handler as goh
        assert goh("EVN").name == "EVN"

    def test_backend_and_dummybackend_importable(self):
        """The existing ``from vlbipy.backends import Backend, DummyBackend`` import works."""
        from vlbipy.backends import Backend, DummyBackend
        assert issubclass(DummyBackend, Backend)

    def test_observatory_classes_importable(self):
        """The existing observatory class imports work."""
        from vlbipy.observatories import EVNObservatory, LBAObservatory, VLBAObservatory
        assert EVNObservatory().name == "EVN"
        assert VLBAObservatory().name == "VLBA"
        assert LBAObservatory().name == "LBA"

    def test_register_functions_on_vlbipy_top_level(self):
        """register_backend and register_observatory are importable from vlbipy."""
        from vlbipy import register_backend, register_observatory
        assert callable(register_backend) and callable(register_observatory)

    def test_plugin_error_on_vlbipy_top_level(self):
        """PluginError is importable from vlbipy."""
        from vlbipy import PluginError
        assert issubclass(PluginError, Exception)

    def test_backend_error_unchanged(self):
        """get_backend still raises BackendError for unknown backends."""
        with pytest.raises(BackendError):
            get_backend("this_does_not_exist")

    def test_config_error_unchanged(self):
        """get_observatory_handler still raises ConfigError for unknown networks."""
        with pytest.raises(ConfigError):
            get_observatory_handler("DOES_NOT_EXIST")

    def test_vlbiobs_with_dummy_backend_unchanged(self):
        """VLBIObs with dummy backend constructs and imports as before."""
        from vlbipy import VLBIObs
        obs = VLBIObs("TEST", network="EVN", backend="dummy", target="SRC")
        obs.import_data()
        assert obs.metadata is not None

    def test_defaults_contain_plugins_key(self):
        """The defaults.toml now contains the plugins key (empty list)."""
        from vlbipy.config import load_config
        cfg = load_config()
        assert cfg["global"]["plugins"] == []
