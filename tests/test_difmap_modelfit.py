"""Tests for the parametric modelfit imaging path of vlbipy.backends.difmap (stub observation, no difmapy)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from vlbipy.backends import difmap
from vlbipy.errors import BackendError


def good_fit(flux: float = 1.0, rchisq: float = 1.0) -> dict:
    """A converged, finite, positive-flux modelfit result."""
    return {"converged": True, "rchisq": rchisq,
            "components": [{"flux": flux, "x": 0.1, "y": -0.2, "major": 0.5, "minor": 0.5, "phi": 0.0}]}


class ModelfitObservation:
    """Just enough of a difmapy Observation for the modelfit imaging path; ``clean`` is forbidden."""

    source = "TEST"
    antennas = ["EF", "WB"]
    nif = 4
    flagged_fraction = 0.0
    estimated_beam = (2.0, 1.5, 30.0)
    model_flux = 1.0

    def __init__(self, fit_results=None, residual_peak: float = 0.0, rms: float = 0.01, npix: int = 64):
        self.fit_results = list(fit_results) if fit_results is not None else [good_fit()]
        self.residual_peak, self.rms, self.npix = residual_peak, rms, npix
        self.model = list(self.fit_results[0].get("components", [])) if self.fit_results else []
        self.calls = {"modelfit": 0, "seed_model": 0, "clrmod": 0, "invert": 0, "uvweight": [],
                      "restore": [], "wmap": [], "auto_mapsize": []}

    def copy(self):
        clone = ModelfitObservation.__new__(ModelfitObservation)
        clone.__dict__.update(self.__dict__)
        clone.fit_results = list(self.fit_results)
        clone.model = list(self.model)
        clone.calls = self.calls
        return clone

    def modelfit(self, quiet=True):
        self.calls["modelfit"] += 1
        result = self.fit_results.pop(0)
        self.model = list(result.get("components", self.model))
        return result

    def seed_model(self, **kwargs):
        self.calls["seed_model"] += 1

    def clrmod(self):
        self.calls["clrmod"] += 1

    def invert(self):
        self.calls["invert"] += 1

    def uvweight(self, robust=None):
        self.calls["uvweight"].append(robust)

    def mapsize(self, npix, cell):
        self.npix = int(npix)

    def auto_mapsize(self, npix=None):
        self.calls["auto_mapsize"].append(npix)
        if npix:
            self.npix = int(npix)

    def imstat(self):
        return {"rms": self.rms, "max": self.residual_peak}

    def noise_stats(self, image=None):
        return {"rms": self.rms if image is None else float(np.std(image))}

    def peak_offset(self):
        return (0.0, 0.0), self.residual_peak

    def restore(self, no_residual=False):
        self.calls["restore"].append(no_residual)
        self.restored_map = np.ones((self.npix, self.npix))

    def wmap(self, path, overwrite=True):
        self.calls["wmap"].append(path)
        data = np.ones((self.npix, self.npix))
        hdu = fits.PrimaryHDU(data)
        hdu.header["CRPIX1"], hdu.header["CRPIX2"] = self.npix / 2, self.npix / 2
        hdu.writeto(str(path), overwrite=True)

    def wmodel(self, path):
        with open(path, "w") as handle:
            handle.write("1.0v 0.0v 0.0v 0.5v 0.5 0.0\n")

    @property
    def valid_slice(self):
        n = self.npix
        return slice(n // 4, n - n // 4), slice(n // 4, n - n // 4)

    def clean(self, *args, **kwargs):
        raise AssertionError("obs.clean must never run on the modelfit imaging path")

    def gain_snapshot(self):
        return "snapshot"

    def scans(self):
        return [{"tmin": 0.0, "tmax": 120.0}]

    def selfcal(self, **kwargs):
        return {"fit_before": {"chisq": 2.0}, "fit_after": {"chisq": 1.0}, "nbins": 10, "nbadsol": 0}

    def bayes_gscale(self, models=None, **kwargs):
        self.calls["bayes_models"] = tuple(models)
        return SimpleNamespace(files={"caltable": "out.G"}, best_model="gauss1", summary=lambda: "summary",
                               to_dict=lambda: {"stations": []})

    def station_gains(self, per_if=False):
        return {"EF": [1.0] * self.nif} if per_if else {"EF": 1.0}

    def savecaltable(self, path, **kwargs):
        return {"path": path, "nrows": 8}


def test_checked_modelfit_accepts_a_good_fit():
    assert difmap._checked_modelfit(good_fit())["converged"]


@pytest.mark.parametrize("result", [
    {"converged": False, "rchisq": 1.0, "components": good_fit()["components"]},
    {"converged": True, "rchisq": math.nan, "components": good_fit()["components"]},
    {"converged": True, "rchisq": math.inf, "components": good_fit()["components"]},
    {"converged": True, "rchisq": 1.0, "components": []},
    {"converged": True, "rchisq": 1.0, "components": [{"flux": -1.0, "x": 0.0, "y": 0.0, "major": 0.5}]},
    {"converged": True, "rchisq": 1.0, "components": [{"flux": 0.0, "x": 0.0, "y": 0.0, "major": 0.5}]},
    {"converged": True, "rchisq": 1.0, "components": [{"flux": 1.0, "x": 0.0, "y": 0.0, "major": -0.5}]},
    {"converged": True, "rchisq": 1.0, "components": [{"flux": math.nan, "x": 0.0, "y": 0.0, "major": 0.5}]},
])
def test_checked_modelfit_rejects_bad_fits(result):
    with pytest.raises(BackendError):
        difmap._checked_modelfit(result)


def test_fit_image_model_defaults_to_a_single_fit_and_no_clean():
    obs = ModelfitObservation(residual_peak=10.0)
    out, report = difmap.fit_image_model(obs)
    assert out is obs and report["method"] == "modelfit"
    assert len(report["components"]) == 1
    assert obs.calls["modelfit"] == 1 and obs.calls["seed_model"] == 0


def test_fit_image_model_rejects_a_failed_first_fit_without_clean():
    obs = ModelfitObservation(fit_results=[{"converged": False, "rchisq": math.nan, "components": []}])
    with pytest.raises(BackendError):
        difmap.fit_image_model(obs, max_components=3)


def test_fit_image_model_rejected_addition_preserves_the_original():
    obs = ModelfitObservation(fit_results=[good_fit(rchisq=1.0), good_fit(rchisq=0.9995)], residual_peak=10.0)
    out, report = difmap.fit_image_model(obs, max_components=2, component_improvement=0.01)
    assert out is obs and len(report["components"]) == 1
    assert report["stopped"] == "additional component did not improve the fit"
    assert obs.calls["modelfit"] == 2 and obs.calls["seed_model"] == 1


def test_fit_image_model_rejected_bad_candidate_preserves_the_original():
    obs = ModelfitObservation(fit_results=[good_fit(rchisq=1.0),
                                           {"converged": True, "rchisq": 0.5,
                                            "components": [{"flux": -1.0, "x": 0, "y": 0, "major": 0.5}]}],
                              residual_peak=10.0)
    out, report = difmap.fit_image_model(obs, max_components=2)
    assert out is obs and len(report["components"]) == 1
    assert report["stopped"] == "additional component fit rejected"


def test_fit_image_model_accepts_an_improving_addition():
    second = good_fit(rchisq=0.5)
    second["components"].append({"flux": 0.2, "x": 1.0, "y": 1.0, "major": 0.3})
    obs = ModelfitObservation(fit_results=[good_fit(rchisq=1.0), second], residual_peak=10.0)
    out, report = difmap.fit_image_model(obs, max_components=2)
    assert out is not obs and len(report["components"]) == 2
    assert report["stopped"] == "component limit reached"


def test_fit_image_model_stops_when_the_residual_is_noise():
    obs = ModelfitObservation(residual_peak=0.001, rms=0.01)
    _, report = difmap.fit_image_model(obs, max_components=3, component_sigma=5.0)
    assert report["stopped"] == "no significant positive residual component"
    assert obs.calls["seed_model"] == 0


def test_modelfit_image_restores_model_plus_residuals(tmp_path):
    obs = ModelfitObservation(npix=64)
    report = {**good_fit(), "stopped": "component limit reached"}
    _, info = difmap.modelfit_image(obs, str(tmp_path / "img.fits"), robust=0.0, model_report=report)
    assert obs.calls["modelfit"] == 0
    assert obs.calls["restore"] == [False]
    assert obs.calls["uvweight"][-1] == 0.0
    assert info["method"] == "modelfit" and info["model"] is report
    with fits.open(tmp_path / "img.fits") as hdul:
        assert hdul[0].data.shape == (32, 32)


def test_modelfit_image_fits_once_when_no_report(tmp_path):
    obs = ModelfitObservation(npix=64)
    difmap.modelfit_image(obs, str(tmp_path / "img.fits"), robust=2.0)
    assert obs.calls["modelfit"] == 1


def test_image_all_robust_fits_once_for_every_robust(tmp_path):
    obs = ModelfitObservation(npix=64)
    images = difmap.image_all_robust(obs, str(tmp_path / "img"), [-2.0, 0.0, 2.0])
    assert obs.calls["modelfit"] == 1
    assert sorted(images) == [-2.0, 0.0, 2.0]
    assert [info["robust"] for info in images.values()] == [-2.0, 0.0, 2.0]
    assert [obs.calls["uvweight"].count(r) for r in (-2.0, 0.0, 2.0)] == [1, 1, 1]


def test_image_all_robust_explicit_clean_opt_in(tmp_path, monkeypatch):
    used = []
    monkeypatch.setattr(difmap, "clean_image",
                        lambda obs, path, **kw: (used.append(kw["robust"]), (obs, {"robust": kw["robust"]}))[1])
    obs = ModelfitObservation()
    images = difmap.image_all_robust(obs, str(tmp_path / "img"), [-1.0, 1.0], method="clean")
    assert used == [-1.0, 1.0] and obs.calls["modelfit"] == 0
    assert sorted(images) == [-1.0, 1.0]


def test_image_all_robust_rejects_unknown_method_and_empty_list(tmp_path):
    obs = ModelfitObservation()
    with pytest.raises(ValueError):
        difmap.image_all_robust(obs, str(tmp_path / "img"), [0.0], method="dmap")
    assert difmap.image_all_robust(obs, str(tmp_path / "img"), []) == {}


def test_amplitude_calibration_filters_clean_candidates():
    obs = ModelfitObservation()
    models = ("current", "clean", SimpleNamespace(kind="clean"), "gauss1")
    difmap.amplitude_calibration(obs, str(obs.source), "parent.ms", models=models)
    assert obs.calls["bayes_models"] == ("current", "gauss1")


def test_amplitude_calibration_passes_clean_candidates_when_allowed():
    obs = ModelfitObservation()
    difmap.amplitude_calibration(obs, "p", "parent.ms", models=("current", "clean"), allow_clean=True)
    assert obs.calls["bayes_models"] == ("current", "clean")


def test_image_source_fits_a_starting_model_then_images(tmp_path, monkeypatch):
    obs = ModelfitObservation(npix=64)
    calls = {"fit_starting_model": 0, "modelfit_image": []}
    monkeypatch.setattr(difmap, "load", lambda *a, **k: obs)
    monkeypatch.setattr(difmap, "locate_source", lambda *a, **k: ({}, (0.0, 0.0)))
    monkeypatch.setattr(difmap, "fit_starting_model",
                        lambda o: (calls.__setitem__("fit_starting_model", calls["fit_starting_model"] + 1),
                                   good_fit())[1])

    def render(o, path, *, robust, model_report=None, **kw):
        calls["modelfit_image"].append((robust, model_report))
        return o, {"fits": path, "robust": robust, "model": model_report}

    monkeypatch.setattr(difmap, "modelfit_image", render)
    report = difmap.image_source("split.ms", str(tmp_path / "img"), robust_values=[0.0, 2.0], mapsize=128)
    assert calls["fit_starting_model"] == 1
    assert [r for r, _ in calls["modelfit_image"]] == [0.0, 2.0]
    reports = [m for _, m in calls["modelfit_image"]]
    assert len({id(m) for m in reports}) == 1 and reports[0] is not None
    assert sorted(report["images"]) == [0.0, 2.0]
    assert 128 in obs.calls["auto_mapsize"]


@pytest.mark.parametrize("robust_values", [[-2.0, 0.0, 2.0], [-2.0, 2.0]])
def test_joint_model_fits_once_renders_and_writes_the_model(tmp_path, monkeypatch, robust_values):
    obs = ModelfitObservation(npix=64)
    rendered = []
    monkeypatch.setattr(difmap, "load", lambda *a, **k: obs)
    monkeypatch.setattr(difmap, "fit_starting_model", lambda o: good_fit())
    monkeypatch.setattr(difmap, "phase_selfcal",
                        lambda o, ladder, **kw: (o, [{"solint": "scan", "accepted": False}]))
    monkeypatch.setattr(difmap, "modelfit_image",
                        lambda o, path, *, robust, model_report=None, **kw:
                        (rendered.append(robust), (o, {"robust": robust, "model": model_report}))[1])
    prefix = str(tmp_path / "joint")
    report = difmap.joint_model("combined.ms", prefix, robust_values=robust_values, model_robust=0.0)
    assert obs.calls["modelfit"] == 1
    assert sorted(rendered) == robust_values
    if 0.0 in robust_values:
        assert rendered[-1] == 0.0
    assert report["method"] == "modelfit" and report["model"] == f"{prefix}.mod"
    assert sorted(report["images"]) == robust_values
    with open(f"{prefix}.mod") as handle:
        assert "v" in handle.read().split()[0]


def test_deep_fit_does_not_model_float32_roundoff_as_faint_sources():
    """Near-perfect fits must not acquire extra components from numerical residuals with spuriously high SNR."""
    floor = float(difmap.MODEL_RESIDUAL_PRECISION)
    obs = ModelfitObservation(residual_peak=floor / 2, rms=floor / 100)
    fitted, report = difmap.fit_image_model(obs, max_components=10)
    assert fitted is obs and len(report["components"]) == 1
    assert obs.calls["seed_model"] == 0


def test_joint_model_refines_phases_again_after_learning_structure(tmp_path, monkeypatch):
    """A deeper combined model must enter the phase ladder before joint modelling is declared finished."""
    obs = ModelfitObservation(npix=64)
    events = []
    monkeypatch.setattr(difmap, "load", lambda *a, **kw: obs)
    monkeypatch.setattr(difmap, "fit_starting_model", lambda o: good_fit())

    def phases(o, ladder, **kw):
        events.append(("phase", len(o.model)))
        return o, [{"solint": "scan", "accepted": False}]

    def model(o, **kw):
        events.append(("fit", kw["max_components"]))
        if len(o.model) == 1:
            o.model.append({**o.model[0], "flux": 0.2, "x": 3.0})
        return o, {**good_fit(), "components": list(o.model), "method": "modelfit", "stopped": "converged"}

    monkeypatch.setattr(difmap, "phase_selfcal", phases)
    monkeypatch.setattr(difmap, "fit_image_model", model)
    monkeypatch.setattr(difmap, "modelfit_image", lambda o, *a, **kw: (o, {"robust": kw["robust"]}))
    report = difmap.joint_model("combined.ms", str(tmp_path / "deep"), robust_values=[0], model_rounds=3)
    assert events == [("phase", 1), ("fit", 10), ("phase", 2), ("fit", 10)]
    assert report["model_rounds"] == 2
    assert [round_["model_round"] for round_ in report["rounds"]] == [1, 2]


def test_joint_model_rejects_invalid_round_limit_before_loading(monkeypatch):
    """Invalid deep-fitting limits must fail before touching input data."""
    monkeypatch.setattr(difmap, "load", lambda *a, **kw: pytest.fail("loaded before validating joint rounds"))
    with pytest.raises(ValueError, match="model_rounds"):
        difmap.joint_model("combined.ms", "unused", model_rounds=0)


class FixedModelObservation(ModelfitObservation):
    """Track fixed-model Bayesian trials separately from the original observation."""

    def __init__(self, after=1.0):
        """Start at chi-squared two and apply the supplied trial chi-squared."""
        super().__init__()
        self.chisq, self.after = 2.0, after
        self.calls["exports"] = []

    def copy(self):
        """Clone trial state while retaining a shared audit trail."""
        trial = FixedModelObservation(self.after)
        trial.__dict__.update(self.__dict__)
        trial.model = [dict(component) for component in self.model]
        self.calls["trial"] = trial
        return trial

    def rmodel(self, path):
        """Record the input model without invoking difmapy."""
        self.calls["rmodel"] = path

    def moddif(self):
        """Return the chi-squared of this observation, not its sibling trial."""
        return {"chisq": self.chisq}

    def _replace_model(self, components):
        """Install the frozen trial components and record them for assertions."""
        self.model = components
        self.calls["frozen"] = components

    def bayes_gscale(self, **kwargs):
        """Apply a synthetic Bayesian correction and expose station diagnostics."""
        self.calls["bayes_kwargs"] = kwargs
        self.chisq = self.after
        rows = [{"station": "EF", "correction": 1.1, "has_data": True},
                {"station": "WB", "correction": 2.0, "has_data": False},
                {"station": "BAD", "correction": math.nan, "has_data": True}]
        return SimpleNamespace(files={"json": "trial.amp.json"}, station_table=lambda: rows)

    def gscale(self, **kwargs):
        """Forbid the old unconstrained amplitude calibration route."""
        raise AssertionError("plain gscale is not permitted")

    def savecaltable(self, path, **kwargs):
        """Capture which observation exported the accumulated corrections."""
        self.calls["exports"].append((self, path, kwargs))
        return {"path": path + ".G"}


@pytest.mark.parametrize("after, accepted", [(1.0, True), (1.999, False), (math.nan, False), (math.inf, False)])
def test_refine_fixed_model_bayesian_acceptance(tmp_path, monkeypatch, after, accepted):
    """Only significant finite Bayesian improvements replace the original gains."""
    obs = FixedModelObservation(after)
    monkeypatch.setattr(difmap, "load", lambda *args, **kwargs: obs)
    phase_options = {}

    def phase(o, ladder, **kwargs):
        """Keep phase unchanged while checking that the input model is never refitted."""
        phase_options.update(kwargs)
        return o, []

    monkeypatch.setattr(difmap, "phase_selfcal", phase)
    model_path = tmp_path / "input.mod"
    model_path.write_text("1.0v 0.0v 0.0v 0.5v 1.0 0.0 1\n")
    prefix = str(tmp_path / "refined")
    uncertainty = {"flux": 0.02}
    report = difmap.refine_against_model("split.ms", "parent.ms", prefix, str(model_path), solints=["scan"],
                                        prior_sigma=0.15, outlier_sigma=2.0, model_sigma=uncertainty, workers=3)
    options = obs.calls["bayes_kwargs"]
    assert options["models"] == ("current",) and options["nloop"] == 0 and options["apply"] is True
    assert options["prior_sigma"] == 0.15 and options["outlier_sigma"] == 2.0
    assert options["model_sigma"] == uncertainty and options["workers"] == 3
    assert options["ms"] == "parent.ms" and options["prefix"] == prefix + ".amp"
    assert phase_options["refit"] is False
    assert all(component["freepar"] == 0 for component in obs.calls["frozen"])
    assert "freepar" not in obs.model[0]
    assert obs.calls["modelfit"] == 0 and obs.chisq == 2.0
    assert report["amplitude_files"] == {"json": "trial.amp.json"}
    assert report["amplitude"] == ({"EF": 1.1} if accepted else {})
    assert report["chisq_after"] == (after if accepted else 2.0)
    if accepted:
        exported, path, keywords = obs.calls["exports"][0]
        assert exported is obs.calls["trial"] and path == prefix
        assert keywords["since"] == "snapshot" and keywords["ms"] == "parent.ms"
        assert report["caltable"] == prefix + ".G"
    else:
        assert obs.calls["exports"] == [] and report["caltable"] == ""


def test_refine_without_amplitudes_exports_accepted_phase_gains(tmp_path, monkeypatch):
    """Accepted phase corrections still export when Bayesian amplitudes are disabled."""
    obs = FixedModelObservation()
    monkeypatch.setattr(difmap, "load", lambda *args, **kwargs: obs)
    monkeypatch.setattr(difmap, "phase_selfcal", lambda o, ladder, **kw: (o, [{"accepted": True}]))
    path = tmp_path / "model.mod"
    path.write_text("1.0 0 0 0.5 1 0 1\n")
    report = difmap.refine_against_model("split.ms", "parent.ms", str(tmp_path / "phase"), str(path),
                                        amplitude=False, solints=["scan"])
    assert "bayes_kwargs" not in obs.calls
    assert report["amplitude_files"] == {} and report["amplitude"] == {}
    assert report["caltable"] and obs.calls["exports"][0][0] is obs


def test_phase_selfcal_failed_refit_discards_accepted_gain_trial():
    """A good phase gain solve cannot retain a nonphysical subsequent model refit."""
    obs = ModelfitObservation(fit_results=[{"converged": False, "rchisq": math.nan, "components": []}])
    obs.model = good_fit()["components"]
    original_model = [dict(component) for component in obs.model]
    returned, rounds = difmap.phase_selfcal(obs, ["30min", "scan"])
    assert returned is obs and obs.model == original_model
    assert len(rounds) == 1 and rounds[0]["accepted"] is False
    assert rounds[0]["reason"] == "modelfit refit failed"
    assert obs.calls["modelfit"] == 1
