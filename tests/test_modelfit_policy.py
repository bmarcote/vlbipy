"""Namespace integration of the parametric imaging policy, without real data."""
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from vlbipy.backends import difmap
from vlbipy.errors import StaleCalibrationError, StepError
from vlbipy.namespaces import CleanNamespace, SelfcalNamespace, _require_current_selfcal


def test_default_policy():
    """The shipped policy starts with one component and no CLEAN candidates."""
    path = Path(__file__).parents[1] / "src/vlbipy/templates/defaults.toml"
    config = tomllib.loads(path.read_text())
    settings = difmap.imaging_settings(config["imaging"])
    assert settings["method"] == "modelfit"
    assert settings["max_components"] == 1
    assert config["selfcal"]["bayes_models"] == ["current", "gauss1", "gauss2", "gauss3"]


@pytest.mark.parametrize("method", ["modelfit", "clean"])
def test_image_settings_and_cache(tmp_path, monkeypatch, method):
    """Explicit overrides reach imaging and the policy-aware cache report."""
    captured = {}

    def image_source(*args, **kwargs):
        """Capture image dispatch without loading visibility data."""
        captured.update(kwargs)
        return {"images": {}}

    monkeypatch.setattr(difmap, "image_source", image_source)
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test")
    namespace = CleanNamespace(obs)
    monkeypatch.setattr(namespace, "split_ms", lambda source: "unused.ms")
    assert namespace._image_with_difmap("source", [0], {"max_components": 1}, "briggs", 17,
                                      method=method, component_sigma=8) == []
    assert captured["method"] == method
    assert captured["niter"] == 17
    assert captured["component_sigma"] == 8
    report = json.loads((tmp_path / "images/test_source.search.json").read_text())
    assert report["imaging"] == difmap.method_keywords({key: captured[key] for key in difmap.imaging_settings({})})


@pytest.mark.parametrize("method, expected", [("modelfit", ("current", "gauss2")),
                                               ("clean", ("clean", "clean10", "current", "gauss2"))])
def test_calibrator_filters_clean_candidates(tmp_path, monkeypatch, method, expected):
    """Only the opt-in CLEAN policy permits CLEAN Bayesian candidates."""
    captured = {}

    def calibrate_source(*args, **kwargs):
        """Stop at dispatch after capturing the policy keywords."""
        captured.update(kwargs)
        raise RuntimeError("captured dispatch")

    monkeypatch.setattr(difmap, "calibrate_source", calibrate_source)
    obs = SimpleNamespace(
        work_dir=tmp_path, project_code="test", observatory="EVN", gaintables=[],
        config={"imaging": {"method": method, "max_components": 1},
                "selfcal": {"bayes_models": ["clean", "clean10", "current", "gauss2"]}},
        sources=SimpleNamespace(fringe_finders=[]),
        _state=SimpleNamespace(should_run=lambda *args, **kwargs: True),
        _backend=SimpleNamespace(ms_path=lambda code: "parent.ms"),
        clean=SimpleNamespace(split_ms=lambda source: "unused.ms"),
        drop_gaintables=lambda steps: None,
    )
    with pytest.raises(RuntimeError, match="captured dispatch"):
        SelfcalNamespace(obs).calibrator("source", niter=19, component_sigma=7)
    assert captured["bayes_models"] == expected
    assert captured["method"] == method
    assert captured["niter"] == 19
    assert captured["component_sigma"] == 7


@pytest.mark.parametrize("imaging", [None, {}, {"method": "clean"},
                                      {"method": "modelfit", "revision": -1}])
@pytest.mark.parametrize("entrypoint", ["calibrator", "run_all", "image"])
def test_legacy_selfcal_is_rejected_before_writes(tmp_path, imaging, entrypoint):
    """Old calibrator policies must not reach an apply, split, or report overwrite."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    path = image_dir / "test_source.search.json"
    if imaging is not None:
        path.write_text(json.dumps({"imaging": imaging}))
    before = path.read_bytes() if path.exists() else None
    obs = SimpleNamespace(
        work_dir=tmp_path, project_code="test", config={}, gaintables=[],
        _state=SimpleNamespace(completed=lambda: ["selfcal_source"], should_run=lambda *a, **kw: False),
        _backend=SimpleNamespace(requires_data_files=True),
    )
    with pytest.raises(StepError, match="--scratch"):
        if entrypoint == "calibrator":
            SelfcalNamespace(obs).calibrator("source")
        elif entrypoint == "run_all":
            SelfcalNamespace(obs).run_all(force=True)
        else:
            CleanNamespace(obs)._image_with_difmap("target", [0], {}, "briggs", 10)
    assert (path.read_bytes() if path.exists() else None) == before


def test_current_selfcal_policy_is_reusable(tmp_path):
    """The current modelfit revision permits resume, including calibration-chain-only records."""
    settings = difmap.imaging_settings({})
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "test_source.search.json").write_text(json.dumps({"imaging": difmap.method_keywords(settings)}))
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test",
                          gaintables=[SimpleNamespace(step="selfcal_source")])
    _require_current_selfcal(obs, settings)


def test_stale_calibration_error_preserves_step_context():
    """The fatal stale-policy subtype retains the public StepError constructor."""
    error = StaleCalibrationError("selfcal_source", "legacy policy", params={"revision": 2}, selection="source")
    assert isinstance(error, StepError)
    assert error.step == "selfcal_source"
    assert error.params == {"revision": 2}
    assert error.selection == "source"
    assert "legacy policy" in str(error)
    assert "[step=selfcal_source]" in str(error)
    assert "[selection=source]" in str(error)
    assert "[params={'revision': 2}]" in str(error)


def test_stale_policy_guard_raises_fatal_subtype(tmp_path):
    """Missing provenance emits the subtype that pipeline handlers must re-raise."""
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test",
                          gaintables=[SimpleNamespace(step="selfcal_source")])
    with pytest.raises(StaleCalibrationError) as caught:
        _require_current_selfcal(obs, difmap.imaging_settings({}))
    assert caught.value.step == "selfcal_source"
    assert "test_source.search.json" in str(caught.value)
    assert "--scratch" in str(caught.value)
    assert "--force alone cannot" in str(caught.value)


@pytest.mark.parametrize("policy", [None, {}, {"method": "unverified"}, {"method": "modelfit", "revision": -1}])
def test_registered_joint_table_requires_current_sidecar(tmp_path, policy):
    """A chain entry alone cannot prove a joint refinement used verified parametric input."""
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test",
                          gaintables=[SimpleNamespace(step="joint_source", field="source")])
    path = tmp_path / "selfcal/test.source.joint.json"
    path.parent.mkdir()
    if policy is not None:
        path.write_text(json.dumps({"model_policy": policy}))
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(StaleCalibrationError, match="test.source.joint.json") as caught:
        _require_current_selfcal(obs, difmap.imaging_settings({}))
    assert caught.value.step == "joint_source"
    assert "--scratch" in str(caught.value)
    assert (path.read_bytes() if path.exists() else None) == before


def test_completed_joint_state_requires_sidecar(tmp_path):
    """Completed joint steps are validated even when their table is absent."""
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test", gaintables=[],
                          _state=SimpleNamespace(completed=lambda: ["joint_source"]))
    with pytest.raises(StaleCalibrationError) as caught:
        _require_current_selfcal(obs, difmap.imaging_settings({}))
    assert caught.value.step == "joint_source"
    assert "joint-model provenance" in str(caught.value)


def test_current_joint_sidecar_is_reusable(tmp_path):
    """A current verified-parametric policy allows a joint-only chain to resume."""
    settings = difmap.imaging_settings({})
    path = tmp_path / "selfcal/test.source.joint.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"model_policy": {"method": "modelfit",
                                                "revision": difmap.method_keywords(settings)["revision"]}}))
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test",
                          gaintables=[SimpleNamespace(step="joint_source", field="source")])
    _require_current_selfcal(obs, settings)


def test_joint_records_do_not_block_explicit_clean(tmp_path):
    """The provenance guard remains inactive under the explicit opt-in CLEAN policy."""
    obs = SimpleNamespace(work_dir=tmp_path, project_code="test",
                          gaintables=[SimpleNamespace(step="joint_source", field="source")])
    settings = difmap.imaging_settings({})
    settings["method"] = "clean"
    _require_current_selfcal(obs, settings)


@pytest.mark.parametrize("entrypoint", ["run", "finish"])
def test_legacy_policy_aborts_before_preparation_or_export(tmp_path, monkeypatch, entrypoint):
    """Neither forced resume nor epoch finishing may change products before policy validation."""
    from vlbipy import VLBIObs
    run = VLBIObs("test", backend="dummy", work_dir=tmp_path)
    obs = run.observations[0]
    obs._state.mark_complete("selfcal_source")
    monkeypatch.setattr(obs._backend, "requires_data_files", True)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy products were changed before the policy guard")

    monkeypatch.setattr(obs, "prepare_run", forbidden)
    monkeypatch.setattr(obs.export, "per_source", forbidden)
    with pytest.raises(StaleCalibrationError):
        run.run(force=True) if entrypoint == "run" else run._finish_epoch(obs, force=True)


def test_stale_policy_is_not_swallowed_by_selfcal_handler(tmp_path, monkeypatch):
    """Fatal provenance errors must escape the best-effort self-calibration boundary."""
    from vlbipy import VLBIObs
    run = VLBIObs("test", backend="dummy", work_dir=tmp_path)
    obs = run.observations[0]

    def stale(**kwargs):
        raise StaleCalibrationError("selfcal_source", "legacy policy")

    monkeypatch.setattr(obs.selfcal, "run_all", stale)
    with pytest.raises(StaleCalibrationError):
        run._selfcal_all(obs)
