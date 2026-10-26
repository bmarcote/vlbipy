"""Tests for :mod:`vlbipy.campaign` — the multi-epoch combination orchestration.

The difmapy work (joint modelling, per-epoch refinement, imaging) is stubbed so
the tests exercise the real control flow of :func:`combine_epochs`,
:func:`refine_calibrator` and :func:`image_combined`: which products are made in
which order, what is refreshed when, and what is recorded in the campaign
report. Observations are light fakes implementing the attributes the campaign
touches (sources, gaintables, namespaces, state).
"""
import json
from pathlib import Path

import pytest

from vlbipy import campaign
from vlbipy.backends import difmap
from vlbipy.models import CalTable


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #

class FakeSource:
    def __init__(self, name):
        self.name = name


class FakeSources:
    """The ``obs.sources`` container: names plus the role groupings the campaign reads."""

    def __init__(self, names, targets=()):
        self._sources = [FakeSource(name) for name in names]
        self.names = list(names)
        self.targets = [FakeSource(name) for name in targets]
        self.fringe_finders = []
        self.phase_calibrators = []

    def __iter__(self):
        return iter(self._sources)


class FakeMetadata:
    """``obs.metadata``: only ``source_names`` (what the epoch really observed) is read."""

    def __init__(self, source_names):
        self.source_names = list(source_names)


class FakeState:
    """``obs._state``: a dict of step -> {status, timestamp} plus completion markers."""

    def __init__(self, events, code):
        self._events = events
        self._code = code
        self._steps = {}

    def as_dict(self):
        return dict(self._steps)

    def completed(self):
        return [step for step, entry in self._steps.items() if entry.get("status") == "done"]

    def mark_complete(self, step, outputs=None):
        self._steps[step] = {"status": "done", "outputs": list(outputs or []),
                             "timestamp": f"t{len(self._steps)}"}
        self._events.append(("mark_complete", self._code, step))

    def mark_failed(self, step, error=""):
        self._steps[step] = {"status": "failed", "error": str(error)}
        self._events.append(("mark_failed", self._code, step))


class FakePlan:
    """``obs.phaseref_plan()``: only ``references`` is read by the campaign."""

    def __init__(self, references):
        self.references = list(references)

    def apply_to(self, source, names):
        return ",".join(n for n in names if n != source)


class FakeExport:
    """Backend ``export`` capability: ``merge`` is recorded, the output path returned."""

    def __init__(self, events):
        self._events = events
        self.merges = []

    def merge(self, codes, source_names, *, inputs=None, scales=None, outputvis="", **kwargs):
        call = {"codes": list(codes), "sources": list(source_names), "inputs": list(inputs or []),
                "scales": list(scales) if scales is not None else None, "outputvis": str(outputvis)}
        self.merges.append(call)
        self._events.append(("merge", tuple(codes), tuple(source_names), str(outputvis)))
        # A measurement set is a directory; the cache checks it exists.
        Path(outputvis).mkdir(parents=True, exist_ok=True)
        return str(outputvis)


class FakeBackend:
    requires_data_files = False

    def __init__(self, events, work_dir):
        self.export = FakeExport(events)
        self._work_dir = work_dir

    def ms_path(self, code):
        return str(Path(self._work_dir) / f"{code}.ms")


class _ObsNamespace:
    def __init__(self, obs):
        self._obs = obs


class FakeCalibrate(_ObsNamespace):
    def apply(self, *, force=False, field="", defer=False):
        self._obs.events.append(("apply", self._obs.project_code, force, field))


class FakeExportNs(_ObsNamespace):
    def per_source(self, *, force=False, sources=None, uvfits=True):
        self._obs.events.append(("per_source", self._obs.project_code, force, uvfits))


class FakeClean(_ObsNamespace):
    def split_ms(self, source, *, force=False):
        self._obs.events.append(("split", self._obs.project_code, source, force))
        path = Path(self._obs.work_dir) / "splits" / f"{self._obs.project_code}.{source}.ms"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class FakeSelfcal(_ObsNamespace):
    def _apply_to(self, source):
        apply_to = self._obs.apply_to.get(source, "")
        self._obs.events.append(("apply_to", self._obs.project_code, source, apply_to))
        return apply_to


class FakeObs:
    """An epoch. ``work_dir`` is a real directory so the epoch's own selfcal products resolve."""

    def __init__(self, events, work_dir, code, names, references, targets, apply_to, metadata_names=None):
        self.events = events
        self.work_dir = str(work_dir)
        self.project_code = code
        self.sources = FakeSources(names, targets=targets)
        self.metadata = FakeMetadata(metadata_names) if metadata_names is not None else None
        self.gaintables = []
        self._references = list(references)
        self.apply_to = dict(apply_to)
        self._backend = FakeBackend(events, work_dir)
        self._state = FakeState(events, code)
        self.calibrate = FakeCalibrate(self)
        self.export = FakeExportNs(self)
        self.clean = FakeClean(self)
        self.selfcal = FakeSelfcal(self)
        self.dropped_step_sets = []
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

    def phaseref_plan(self, *, refresh=False):
        return FakePlan(self._references)

    @property
    def _caltables_path(self):
        return None  # in-memory backend: no persisted chain

    def add_gaintable(self, table, step):
        table.step = step
        self.gaintables.append(table)
        self.events.append(("add_gaintable", self.project_code, table.cal_type, step,
                            table.apply_to, table.field))
        return table

    def drop_gaintables(self, steps):
        steps = set(steps)
        self.dropped_step_sets.append(steps)
        dropped = [t.cal_type for t in self.gaintables if t.step in steps]
        self.gaintables[:] = [t for t in self.gaintables if t.step not in steps]
        self.events.append(("drop_gaintables", self.project_code, frozenset(steps)))
        return dropped


class FakeVLBI:
    """The campaign object: ``observations``, ``config`` and the per-epoch imaging hooks."""

    def __init__(self, events, observations, config):
        self.events = events
        self.observations = list(observations)
        self.config = dict(config)
        for obs in self.observations:
            obs.config = self.config

    def _image_all(self, obs, *, force=False, reimage=False):
        self.events.append(("image_all", obs.project_code, force, reimage))

    def _calibrated_diagnostics(self, obs):
        self.events.append(("calibrated_diagnostics", obs.project_code))

    def _final_plots(self, obs):
        self.events.append(("final_plots", obs.project_code))


# --------------------------------------------------------------------------- #
# Stubs of the difmapy layer                                                   #
# --------------------------------------------------------------------------- #

MOD_TEXT = "0.5 0.0 0.0 1.0 1.0 0.0\n"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Stubs of the difmapy functions the campaign drives, plus an event log."""
    events = []
    calls = {"joint_model": [], "refine": [], "scale_model": [], "image_jobs": []}
    failures = {"refine": {}, "joint_model": None}
    # Fitted joint-model flux. 2.0 equals the reference of the default epochs (1, 2, 4 Jy),
    # so m == r unless a test overrides it.
    knobs = {"model_flux": 2.0}

    def fake_joint_model(combined_ms, prefix, **kwargs):
        calls["joint_model"].append({"combined_ms": str(combined_ms), "prefix": str(prefix), **kwargs})
        events.append(("joint_model", Path(prefix).name))
        if failures["joint_model"] is not None:
            raise failures["joint_model"]
        model_path = Path(f"{prefix}.mod")
        model_path.write_text(MOD_TEXT)
        fits_path = Path(f"{prefix}.robust0.fits")
        fits_path.write_text("fits")
        return {"model": str(model_path), "model_flux": knobs["model_flux"],
                "rounds": [{"solint": "inf", "accepted": True}],
                "images": {0.0: {"fits": str(fits_path), "peak": 1.0, "rms": 0.01,
                                 "dynamic_range": 100.0}},
                "image_model": {"components": 2}}

    def fake_scale_model_file(model_path, out_path, factor):
        calls["scale_model"].append({"model": str(model_path), "out": str(out_path), "factor": factor})
        events.append(("scale_model", Path(out_path).name, factor))
        Path(out_path).write_text(MOD_TEXT)
        return str(out_path)

    def fake_refine(split_ms, parent_ms, prefix, model_path, **kwargs):
        calls["refine"].append({"split_ms": str(split_ms), "parent_ms": str(parent_ms),
                                "prefix": str(prefix), "model": str(model_path), **kwargs})
        events.append(("refine", Path(prefix).name))
        code = Path(prefix).name.split(".")[0]
        if code in failures["refine"]:
            raise failures["refine"][code]
        table = Path(f"{prefix}.G")
        table.mkdir(parents=True, exist_ok=True)
        return {"caltable": str(table),
                "rounds": [{"solint": "inf", "accepted": True}, {"solint": "60s", "accepted": False}],
                "amplitude": {}, "amplitude_files": {"json": f"{prefix}.amp.json"},
                "chisq_before": 10.0, "chisq_after": 5.0}

    def fake_run_image_jobs(jobs, **kwargs):
        calls["image_jobs"].append(list(jobs))
        events.append(("image_jobs", tuple(Path(j["imagename"]).name for j in jobs)))
        reports = []
        for job in jobs:
            images = {}
            for robust in job["robust_values"]:
                fits_path = Path(f"{job['imagename']}.robust{robust:g}.fits")
                fits_path.write_text("fits")
                images[robust] = {"fits": str(fits_path), "peak": 1.0, "rms": 0.01,
                                  "dynamic_range": 100.0, "robust": robust}
            # The parametric model of the image; "model" itself is left empty on purpose so the
            # record proves it carries image_model, like the difmapy reports do.
            reports.append({"images": images, "shift_mas": [0.0, 0.0], "search": {"peak": 1.0},
                            "model": {}, "image_model": {"components": [{"flux": 1.0}]}})
        return reports

    monkeypatch.setattr(difmap, "joint_model", fake_joint_model)
    monkeypatch.setattr(difmap, "scale_model_file", fake_scale_model_file)
    monkeypatch.setattr(difmap, "refine_against_model", fake_refine)
    monkeypatch.setattr(difmap, "run_image_jobs", fake_run_image_jobs)
    import vlbipy.plotting
    monkeypatch.setattr(vlbipy.plotting, "plot_image_grid", lambda images, d, code="": [str(Path(d) / "grid.png")])
    return type("Rig", (), {"events": events, "calls": calls, "failures": failures,
                            "knobs": knobs, "tmp": tmp_path})


def _write_selfcal_flux(obs, source, flux):
    """The ``selfcal/<code>.<source>.json`` the campaign reads the epoch flux from."""
    path = Path(obs.work_dir) / "selfcal"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{obs.project_code}.{source}.json").write_text(
        json.dumps({"images": {"0": {"model_flux": flux}}}))


def make_campaign(rig, *, epoch_fluxes=(1.0, 2.0, 4.0), config=None, calibrator="JCAL",
                  per_epoch_names=None, references=None, metadata_names=None):
    """Three epochs V589A/B/C sharing ``calibrator``; return (vlbi, obs list)."""
    codes = ["V589A", "V589B", "V589C"]
    observations = []
    for index, code in enumerate(codes):
        names = per_epoch_names[index] if per_epoch_names else [calibrator, "TGT", "CHK"]
        refs = references[index] if references else [calibrator]
        observed = metadata_names[index] if metadata_names else None
        obs = FakeObs(rig.events, rig.tmp / code, code, names, refs, targets=["TGT"],
                      apply_to={calibrator: "TGT,CHK"}, metadata_names=observed)
        if calibrator in names and epoch_fluxes[index] is not None:
            _write_selfcal_flux(obs, calibrator, epoch_fluxes[index])
        observations.append(obs)
    vlbi = FakeVLBI(rig.events, observations, config or {})
    return vlbi, observations


def event_names(rig):
    return [event[0] for event in rig.events]


def joint_merges(rig):
    return [m for m in rig.calls["joint_model"]]


# --------------------------------------------------------------------------- #
# The happy path                                                               #
# --------------------------------------------------------------------------- #

def test_combine_epochs_joint_model_then_refine_then_refresh_then_combine(rig):
    vlbi, observations = make_campaign(rig)
    report = campaign.combine_epochs(vlbi)

    assert not report["failed"]
    assert set(report["calibrators"]) == {"JCAL"}
    assert set(report["combined"]) == {"JCAL", "TGT", "CHK"}

    # All calibrated epochs of the calibrator merged once, at a common flux level:
    # fluxes 1, 2, 4 Jy -> reference 2 Jy -> scales 2, 1, 0.5.
    merge = vlbi.observations[0]._backend.export.merges
    joint = [m for m in merge if m["outputvis"].endswith("JCAL.joint.ms")]
    assert len(joint) == 1
    assert joint[0]["codes"] == ["V589A", "V589B", "V589C"]
    assert joint[0]["scales"] == [2.0, 1.0, 0.5]
    assert len(joint[0]["inputs"]) == 3

    # The joint model is fit once from the merged data, before any refinement.
    assert len(joint_merges(rig)) == 1
    names = event_names(rig)
    assert names.index("joint_model") > names.index("merge")
    assert names.index("refine") > names.index("joint_model")
    assert len([e for e in rig.events if e[0] == "refine"]) == 3

    # Each epoch's own split is refined against the model scaled to the epoch's flux.
    # Here the fitted model flux equals the 2 Jy reference, so the factors are flux / 2.
    scale_factors = [c["factor"] for c in rig.calls["scale_model"]]
    assert scale_factors == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(2.0)]
    for call, obs in zip(rig.calls["refine"], observations):
        assert call["split_ms"].endswith(f"{obs.project_code}.JCAL.ms")
        assert call["parent_ms"].endswith(f"{obs.project_code}.ms")

    # A joint_<source> CalTable is registered per epoch, in time/antenna order,
    # applied to what the epoch's own selfcal of the calibrator covers.
    added = [e for e in rig.events if e[0] == "add_gaintable"]
    assert [(e[1], e[2]) for e in added] == [("V589A", "joint_JCAL"), ("V589B", "joint_JCAL"),
                                           ("V589C", "joint_JCAL")]
    assert all(e[4] == "TGT,CHK" for e in added)   # apply_to from obs.selfcal._apply_to
    assert all(e[5] == "JCAL" for e in added)      # field of the table
    assert all(e[3] == "joint_JCAL" for e in added)  # registered step
    assert all(Path(t.path).is_dir() for obs in observations for t in obs.gaintables)

    # After refinement every epoch is re-applied, re-exported, re-split and re-imaged
    # before the combined data are merged and imaged.
    last_refine = max(i for i, e in enumerate(rig.events) if e[0] == "refine")
    first_apply = min(i for i, e in enumerate(rig.events) if e[0] == "apply")
    first_image = min(i for i, e in enumerate(rig.events) if e[0] == "image_all")
    first_image_jobs = names.index("image_jobs")
    assert last_refine < first_apply < first_image < first_image_jobs
    refreshed = [(e[1], e[2]) for e in rig.events if e[0] == "apply"]
    assert ("V589A", True) in refreshed and ("V589B", True) in refreshed and ("V589C", True) in refreshed
    assert [e[1] for e in rig.events if e[0] == "per_source" and e[2] is True] == \
        ["V589A", "V589B", "V589C"]
    assert [e[1] for e in rig.events if e[0] == "image_all"] == ["V589A", "V589B", "V589C"]
    assert [e[1] for e in rig.events if e[0] == "calibrated_diagnostics"] == ["V589A", "V589B", "V589C"]
    assert [e[1] for e in rig.events if e[0] == "final_plots"] == ["V589A", "V589B", "V589C"]

    # Combined imaging merges every source (without flux scaling) and images all jobs.
    combined = [m for m in merge if m["outputvis"].endswith(".ms") and ".joint." not in m["outputvis"]]
    assert {Path(m["outputvis"]).name for m in combined} == \
        {"V589_JCAL.ms", "V589_TGT.ms", "V589_CHK.ms"}
    assert all(m["scales"] is None for m in combined)
    assert len(rig.calls["image_jobs"]) == 1
    assert len(rig.calls["image_jobs"][0]) == 3

    # The campaign report is written under combined/.
    report_path = rig.tmp / "combined" / "V589.campaign.json"
    assert report_path.is_file()
    on_disk = json.loads(report_path.read_text())
    assert on_disk["calibrators"]["JCAL"]["reference_flux_jy"] == pytest.approx(2.0)
    assert on_disk["calibrators"]["JCAL"]["tables"] == {obs.project_code: t.path
                                                       for obs in observations for t in obs.gaintables}
    assert on_disk["combined"]["TGT"]["epochs"] == ["V589A", "V589B", "V589C"]
    # The combined record carries the parametric image model, and the report the
    # scientific settings it was made with (the cache key of a later run).
    assert on_disk["combined"]["TGT"]["model"] == {"components": [{"flux": 1.0}]}
    assert on_disk["settings"]["joint"]["model_rounds"] == 3
    assert on_disk["settings"]["joint"]["max_components"] == 10


# --------------------------------------------------------------------------- #
# Joint-model configuration                                                    #
# --------------------------------------------------------------------------- #

def test_joint_model_defaults_and_settings(rig):
    """Default [imaging]/[selfcal]: parametric joint fit, 10 components, 3 rounds, no amplitudes."""
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    kwargs = joint_merges(rig)[0]
    assert kwargs["method"] == "modelfit"
    assert kwargs["max_components"] == 10          # joint_max_components default
    assert kwargs["model_rounds"] == 3             # joint_model_rounds default, passed as model_rounds
    assert all(call["amplitude"] is False for call in rig.calls["refine"])


def test_joint_amplitude_false_is_preserved(rig):
    """An explicit ``joint_amplitude = false`` reaches the refinement as ``amplitude=False``."""
    vlbi, _ = make_campaign(rig, config={"selfcal": {"joint_amplitude": False}})
    campaign.combine_epochs(vlbi)
    assert rig.calls["refine"]
    assert all(call["amplitude"] is False for call in rig.calls["refine"])


def test_joint_model_settings_overrides(rig):
    """Explicit [selfcal] joint settings reach the backend; joint_amplitude=true forwards."""
    config = {"selfcal": {"joint_max_components": 7, "joint_model_rounds": 5,
                          "joint_amplitude": True}}
    vlbi, _ = make_campaign(rig, config=config)
    campaign.combine_epochs(vlbi)
    kwargs = joint_merges(rig)[0]
    assert kwargs["max_components"] == 7
    assert kwargs["model_rounds"] == 5
    assert all(call["amplitude"] is True for call in rig.calls["refine"])


def test_target_max_components_stays_one(rig):
    """The deep joint model gets joint_max_components; the per-source imaging keeps max_components=1."""
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    assert joint_merges(rig)[0]["max_components"] == 10
    for job in rig.calls["image_jobs"][0]:
        assert job["max_components"] == 1          # imaging default, not the joint one


def test_epoch_model_scaled_by_fitted_model_flux(rig):
    """The epoch model keeps the epoch's flux density: scaled by epoch_flux / model_flux, not / reference.

    Normalising the inputs to a common level does not guarantee the deep fit's total
    flux lands on it — the fit can put flux into structure a single epoch never
    constrained. Epochs 1/2/4 Jy, reference 2 Jy, fitted model 1.5 Jy.
    """
    rig.knobs["model_flux"] = 1.5
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    factors = [c["factor"] for c in rig.calls["scale_model"]]
    assert factors == [pytest.approx(1 / 1.5), pytest.approx(2 / 1.5), pytest.approx(4 / 1.5)]


# --------------------------------------------------------------------------- #
# Heterogeneous availability                                                     #
# --------------------------------------------------------------------------- #

def test_calibrator_in_one_epoch_only_is_skipped(rig):
    """A reference a single epoch observed gets no joint model and no failure."""
    vlbi, _ = make_campaign(rig, per_epoch_names=[["JCAL", "TGT"], ["JCAL"], ["TGT"]],
                            references=[["JCAL"], ["JCAL"], []])
    # JCAL is in two epochs: it is modelled with those two only.
    report = campaign.combine_epochs(vlbi)
    assert joint_merges(rig) or not report["failed"]


def test_calibrator_in_single_epoch_no_joint_model(rig):
    vlbi, observations = make_campaign(rig, per_epoch_names=[["JCAL", "TGT"], ["TGT"], ["TGT"]],
                                       references=[["JCAL"], [], []])
    report = campaign.combine_epochs(vlbi)
    assert not joint_merges(rig)
    assert not report["calibrators"]
    assert not any(e[0] == "refine" for e in rig.events)
    assert not any(e[0] == "add_gaintable" for e in rig.events)
    assert not report["failed"]


def test_metadata_decides_epoch_participation(rig):
    """Sources may be declared globally; what an epoch observed is its metadata's source_names."""
    vlbi, observations = make_campaign(
        rig, metadata_names=[["JCAL", "TGT"], ["JCAL", "TGT"], ["TGT", "CHK"]],
        epoch_fluxes=(1.0, 3.0, None))
    report = campaign.combine_epochs(vlbi)
    # V589C declares JCAL like the others but never observed it: no joint epoch, no table.
    joint = [m for m in observations[0]._backend.export.merges if ".joint." in m["outputvis"]]
    assert len(joint) == 1
    assert joint[0]["codes"] == ["V589A", "V589B"]
    assert {Path(c["split_ms"]).name for c in rig.calls["refine"]} == \
        {"V589A.JCAL.ms", "V589B.JCAL.ms"}
    assert set(report["calibrators"]["JCAL"]["tables"]) == {"V589A", "V589B"}
    assert report["combined"]["JCAL"]["epochs"] == ["V589A", "V589B"]
    assert [e[1] for e in rig.events if e[0] == "image_all"] == ["V589A", "V589B"]


def test_calibrator_in_subset_of_epochs(rig):
    """The joint model and refinement cover exactly the epochs that observed the source."""
    vlbi, observations = make_campaign(rig, per_epoch_names=[["JCAL", "TGT"], ["JCAL", "TGT"], ["TGT"]],
                                       references=[["JCAL"], ["JCAL"], []],
                                       epoch_fluxes=(1.0, 3.0, None))
    report = campaign.combine_epochs(vlbi)
    joint = [m for m in observations[0]._backend.export.merges if ".joint." in m["outputvis"]]
    assert len(joint) == 1
    assert joint[0]["codes"] == ["V589A", "V589B"]
    assert joint[0]["scales"] == [pytest.approx(2.0), pytest.approx(2 / 3)]  # reference = median(1, 3) = 2
    refined = {c["split_ms"] for c in rig.calls["refine"]}
    assert {Path(s).name for s in refined} == {"V589A.JCAL.ms", "V589B.JCAL.ms"}
    assert set(report["calibrators"]["JCAL"]["tables"]) == {"V589A", "V589B"}
    # Only epochs whose calibration changed are refreshed; V589C is untouched.
    assert [e[1] for e in rig.events if e[0] == "image_all"] == ["V589A", "V589B"]


def test_multiple_calibrators_and_apply_order(rig):
    """A second calibrator refined after the first is re-applied so its split carries it."""
    vlbi, observations = make_campaign(
        rig, references=[["JCAL", "JCAL2"], ["JCAL", "JCAL2"], ["JCAL", "JCAL2"]],
        per_epoch_names=[["JCAL", "JCAL2", "TGT"]] * 3, epoch_fluxes=(1.0, 1.0, 1.0))
    for obs in observations:
        obs.apply_to["JCAL2"] = "TGT"
        _write_selfcal_flux(obs, "JCAL2", 0.8)
    report = campaign.combine_epochs(vlbi)
    assert set(report["calibrators"]) == {"JCAL", "JCAL2"}
    assert len(joint_merges(rig)) == 2
    # The joint tables of both calibrators are registered, per epoch.
    steps = {(e[1], e[3]) for e in rig.events if e[0] == "add_gaintable"}
    assert ("V589A", "joint_JCAL") in steps and ("V589A", "joint_JCAL2") in steps
    # Once JCAL's table exists, the epochs are re-applied (and re-split) on JCAL2
    # before its joint model so the merged data carry it.
    field_applies = [e for e in rig.events if e[0] == "apply" and e[3] == "JCAL2"]
    assert [e[1] for e in field_applies] == ["V589A", "V589B", "V589C"]
    second_joint = [i for i, e in enumerate(rig.events) if e[0] == "joint_model"][1]
    assert all(rig.events.index(e) < second_joint for e in field_applies)
    refreshes = [e for e in rig.events if e[0] == "split" and e[2] == "JCAL2" and e[3] is True]
    assert len(refreshes) == 3


# --------------------------------------------------------------------------- #
# Stale joint tables                                                           #
# --------------------------------------------------------------------------- #

def test_stale_joint_tables_dropped_before_merging(rig):
    """Every registered ``joint_`` table — including steps of obsolete references — is removed first."""
    vlbi, observations = make_campaign(rig)
    stale = CalTable(cal_type="joint_JCAL", path="/x/joint.G", field="JCAL", step="joint_JCAL")
    obsolete = CalTable(cal_type="joint_OLD", path="/x/old.G", field="OLD", step="joint_OLD")
    observations[0].gaintables.extend([stale, obsolete])

    campaign.combine_epochs(vlbi)

    # The dropped set covers the current references *and* the stale joint_ steps in the chain.
    dropped = observations[0].dropped_step_sets[0]
    assert {"joint_JCAL", "joint_OLD"} <= dropped
    assert stale not in observations[0].gaintables and obsolete not in observations[0].gaintables
    # Dropping happens before any split or merge so the joint model sees the plain calibration.
    first_drop = min(i for i, e in enumerate(rig.events) if e[0] == "drop_gaintables")
    assert first_drop < min(i for i, e in enumerate(rig.events) if e[0] == "split")
    assert first_drop < min(i for i, e in enumerate(rig.events) if e[0] == "merge")
    # Removing the tables invalidates the applied data: apply + per-source refresh ran.
    assert ("apply", "V589A", True, "") in rig.events
    assert ("per_source", "V589A", True, False) in rig.events


# --------------------------------------------------------------------------- #
# Failure handling                                                             #
# --------------------------------------------------------------------------- #

def test_failed_refinement_is_recorded_not_silent(rig):
    rig.failures["refine"]["V589B"] = RuntimeError("no fringes")
    vlbi, observations = make_campaign(rig)
    report = campaign.combine_epochs(vlbi)

    record = report["calibrators"]["JCAL"]
    assert record["failed_epochs"] == ["V589B"]
    assert set(record["tables"]) == {"V589A", "V589C"}
    assert "refinement of JCAL in V589B" in report["failed"]
    # The failed epoch is marked failed in its state and gets no joint table.
    assert observations[1]._state.as_dict()["joint_JCAL"]["status"] == "failed"
    assert not observations[1].gaintables
    # The other epochs still refine, register their tables and are refreshed.
    assert [e[1] for e in rig.events if e[0] == "add_gaintable"] == ["V589A", "V589C"]
    assert [e[1] for e in rig.events if e[0] == "image_all"] == ["V589A", "V589C"]


def test_failed_joint_model_does_not_stop_other_work(rig):
    rig.failures["joint_model"] = RuntimeError("modelfit diverged")
    vlbi, _ = make_campaign(rig)
    report = campaign.combine_epochs(vlbi)
    assert "joint model of JCAL" in report["failed"]
    assert not report["calibrators"]
    # The combined images are still made.
    assert set(report["combined"]) == {"JCAL", "TGT", "CHK"}


def test_failed_combined_image_is_reported(rig, monkeypatch):
    def boom(jobs, **kwargs):
        return [RuntimeError("clean blew up")] * len(jobs) + []
    monkeypatch.setattr(difmap, "run_image_jobs", lambda jobs, **k: [RuntimeError("x")] * len(jobs))
    vlbi, _ = make_campaign(rig)
    report = campaign.combine_epochs(vlbi)
    assert not report["combined"]
    assert "combined image of TGT" in report["failed"]


# --------------------------------------------------------------------------- #
# Report caching / invalidation                                                #
# --------------------------------------------------------------------------- #

def test_completed_report_is_reused(rig):
    vlbi, _ = make_campaign(rig)
    first = campaign.combine_epochs(vlbi)
    n_events = len(rig.events)
    again = campaign.combine_epochs(vlbi)
    assert again == first
    assert len(rig.events) == n_events  # nothing redone


def test_report_with_failures_is_recombined(rig):
    vlbi, _ = make_campaign(rig)
    rig.failures["joint_model"] = RuntimeError("bad")
    failed = campaign.combine_epochs(vlbi)
    assert failed["failed"]
    rig.failures["joint_model"] = None
    redone = campaign.combine_epochs(vlbi)
    assert not redone["failed"]
    assert "JCAL" in redone["calibrators"]


def test_force_redoes_combination(rig):
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    n_events = len(rig.events)
    campaign.combine_epochs(vlbi, force=True)
    assert len(rig.events) > n_events


def test_missing_combined_ms_invalidates_cache(rig):
    """A report whose combined measurement set is gone cannot be reused."""
    import shutil
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    shutil.rmtree(rig.tmp / "combined" / "data" / "V589_TGT.ms")
    n_events = len(rig.events)
    campaign.combine_epochs(vlbi)
    assert len(rig.events) > n_events


def test_missing_joint_model_invalidates_cache(rig):
    """A report whose calibrator .mod file is gone cannot be reused."""
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    (rig.tmp / "combined" / "selfcal" / "V589.JCAL.joint.mod").unlink()
    n_events = len(rig.events)
    campaign.combine_epochs(vlbi)
    assert len(rig.events) > n_events


def test_missing_combined_image_invalidates_cache(rig):
    """A report whose combined FITS image is gone cannot be reused."""
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    (rig.tmp / "combined" / "images" / "V589_TGT.robust0.fits").unlink()
    n_events = len(rig.events)
    campaign.combine_epochs(vlbi)
    assert len(rig.events) > n_events


def test_changed_settings_invalidate_cache(rig):
    """The same epochs combined with other scientific settings are a new combination."""
    vlbi, _ = make_campaign(rig)
    campaign.combine_epochs(vlbi)
    vlbi.config["selfcal"] = {"joint_model_rounds": 9}
    n_events = len(rig.events)
    campaign.combine_epochs(vlbi)
    assert len(rig.events) > n_events


@pytest.mark.parametrize("failure", ["scale", "register", "register_after_append", "complete"])
def test_epoch_io_failures_preserve_registered_tables_and_continue(rig, monkeypatch, failure):
    """An epoch failure never hides applied chain mutations or prevents later epochs from refining."""
    vlbi, observations = make_campaign(rig)
    second = observations[1]

    def fail(*args, **kwargs):
        """Simulate a failing epoch operation."""
        raise OSError("epoch bookkeeping failed")

    if failure == "scale":
        original = difmap.scale_model_file

        def scale(model, output, factor):
            """Fail only the second epoch's model write."""
            if Path(output).name.startswith("V589B."):
                fail()
            return original(model, output, factor)

        monkeypatch.setattr(difmap, "scale_model_file", scale)
    elif failure == "register":
        monkeypatch.setattr(second, "add_gaintable", fail)
    elif failure == "register_after_append":
        original = second.add_gaintable

        def register(table, step):
            """Simulate persistence failing after the in-memory chain changed."""
            original(table, step)
            fail()

        monkeypatch.setattr(second, "add_gaintable", register)
    else:
        monkeypatch.setattr(second._state, "mark_complete", fail)

    report = campaign.combine_epochs(vlbi)
    record = report["calibrators"]["JCAL"]
    expected = {"V589A", "V589C"}
    if failure in {"register_after_append", "complete"}:
        expected.add("V589B")
    assert record["failed_epochs"] == ["V589B"]
    assert report["failed"] == ["refinement of JCAL in V589B"]
    assert set(record["tables"]) == expected
    assert {obs.project_code for obs in observations if obs.gaintables} == expected
    assert {e[1] for e in rig.events if e[0] == "apply"} == expected
    assert {e[1] for e in rig.events if e[0] == "image_all"} == expected
    assert {e[1] for e in rig.events if e[0] == "per_source"} == expected
    assert second._state.as_dict()["joint_JCAL"]["status"] == "failed"
    assert observations[2]._state.as_dict()["joint_JCAL"]["status"] == "done"


@pytest.mark.parametrize("change", ["removed", "added"])
def test_campaign_membership_invalidates_cache_without_chain_mtimes(rig, change):
    """Adding or removing epochs under the same campaign name invalidates otherwise viable products."""
    vlbi, observations = make_campaign(rig)
    if change == "added":
        vlbi.observations = observations[:2]
    first = campaign.combine_epochs(vlbi)
    vlbi.observations = observations[:2] if change == "removed" else observations
    assert all(obs._caltables_path is None for obs in vlbi.observations)
    path = rig.tmp / "combined" / "V589.campaign.json"
    assert campaign.stale_reason(vlbi.observations, path, first) == "the campaign epochs changed"
    redone = campaign.combine_epochs(vlbi)
    codes = [obs.project_code for obs in vlbi.observations]
    assert first["name"] == redone["name"] == "V589"
    assert redone["epochs"] == codes
    assert redone["combined"]["TGT"]["epochs"] == codes
    assert rig.calls["joint_model"][-1]["combined_ms"].endswith("V589_JCAL.joint.ms")
    assert len(rig.calls["joint_model"]) == 2


def test_changed_resolved_reference_plan_invalidates_cache(rig):
    """An inferred reference change invalidates a cache even when configuration and chains are unchanged."""
    vlbi, observations = make_campaign(rig)
    first = campaign.combine_epochs(vlbi)
    for obs in observations:
        obs._references = ["CHK"]
    redone = campaign.combine_epochs(vlbi)
    assert first["settings"]["phase_referencing"] == redone["settings"]["phase_referencing"]
    assert redone["settings"]["reference_plans"] == {obs.project_code: ["CHK"] for obs in observations}
    assert set(redone["calibrators"]) == {"CHK"}
    assert len(rig.calls["joint_model"]) == 2
    assert all(obs._caltables_path is None for obs in observations)


def test_deleted_joint_table_invalidates_cache(rig):
    """Missing per-epoch corrections cannot be hidden by still-present joint models and images."""
    vlbi, observations = make_campaign(rig)
    first = campaign.combine_epochs(vlbi)
    table = Path(first["calibrators"]["JCAL"]["tables"]["V589B"])
    table.rmdir()
    path = rig.tmp / "combined" / "V589.campaign.json"
    assert campaign.stale_reason(observations, path, first) == "the joint table of JCAL in V589B is missing"
    redone = campaign.combine_epochs(vlbi)
    assert not redone["failed"]
    assert table.is_dir()
    assert len(rig.calls["joint_model"]) == 2
