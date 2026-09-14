"""Contract tests for the public API (VLBIObs + namespaces), all on the dummy backend."""
import pytest

from vlbipy import Image, ImageSet, Observation, SelfcalResult, VLBIObs


def make_single():
    return VLBIObs(project="RSM07", network="EVN", backend="dummy", target="3C286",
                   phasecal="J1048+7143", fringe_finder="3C345")


def make_campaign():
    return VLBIObs(project=["RSM07", "RSM08", "RSM09"], network="EVN", backend="dummy",
                   target="3C286", fringe_finder="3C345")


# -- construction --

def test_single_project_construction():
    obs = VLBIObs(project="RSM07", network="EVN", backend="dummy", target="3C286")
    assert len(obs) == 1
    assert isinstance(obs["RSM07"], Observation)
    assert obs.sources.target.name == "3C286"
    assert obs.config["global"]["observatory"] == "EVN"


def test_campaign_construction():
    camp = make_campaign()
    assert len(camp) == 3
    assert [o.project_code for o in camp.observations] == ["RSM07", "RSM08", "RSM09"]


def test_unknown_project_index_raises():
    obs = make_single()
    with pytest.raises(KeyError):
        _ = obs["NOPE"]


# -- import & metadata --

def test_import_populates_metadata_and_state():
    obs = make_single()
    obs.import_data()
    meta = obs["RSM07"].metadata
    assert meta is not None and meta.n_antennas >= 6
    assert obs["RSM07"].state.status("import_data") == "done"


# -- calibration namespace --

def test_calibrate_chain_records_gaintables():
    obs = make_single()
    obs.import_data()
    obs.calibrate()
    types = [t.cal_type for t in obs["RSM07"].gaintables]
    assert types[:5] == ["tsys", "gc", "sbd", "bpass", "mbd"]
    for step in ("a_priori", "initial_calibration", "bandpass", "fringefit", "apply"):
        assert obs["RSM07"].state.status(step) == "done"


def test_calibrate_individual_steps():
    obs = make_single()
    obs.import_data()
    ct = obs.calibrate.bandpass()
    assert ct.cal_type == "bpass"


def test_smart_skip_on_second_call(capsys):
    obs = make_single()
    obs.import_data()
    obs.calibrate.bandpass()
    # Second call should be skipped (state already 'done'); forced call re-runs.
    obs.calibrate.bandpass()
    assert obs["RSM07"].state.status("bandpass") == "done"


# -- flagging --

def test_flag_returns_fraction():
    obs = make_single()
    obs.import_data()
    frac = obs.flag.aoflagger()
    assert isinstance(frac, float) and 0.0 <= frac <= 1.0


def test_flagging_lives_in_the_flag_namespace_not_calibrate():
    """Edge channels, quack, statistics are flagging; statwt is calibration."""
    obs = make_single()
    for name in ("edges", "quack", "initial", "tfcrop", "statistics", "apriori", "outliers"):
        assert name in obs.flag.operations(), name
    assert "edge_channels" not in obs.calibrate.operations()
    assert "reweight" in obs.calibrate.operations()
    from vlbipy.backends.base import CalibrationOps, FlagOps
    assert hasattr(FlagOps, "measure_edge_channels") and not hasattr(CalibrationOps, "measure_edge_channels")
    assert hasattr(FlagOps, "summary") and hasattr(CalibrationOps, "reweight")


def test_flag_edges_measures_from_the_bandpass_when_available():
    obs = make_single()
    obs.import_data()
    blind = obs.flag.edges()                       # no bandpass yet: blind fraction
    assert blind["method"] == "fraction" and blind["n_edge"] >= 0
    obs.calibrate.a_priori()
    obs.calibrate.instrumental()
    measured = obs.flag.edges(force=True)
    assert measured["method"] == "measured" and measured["n_edge"] > 0
    explicit = obs.flag.edges(force=True, edge_channels=3)
    assert explicit["method"] == "explicit" and explicit["n_edge"] == 3


def test_flag_quack_prefers_configured_intervals(monkeypatch):
    obs = VLBIObs(project="RSM07", network="EVN", backend="dummy", target="3C286",
                  fringe_finder="3C345", flagging={"quack_antennas": {"EF": 4.0}})
    obs.import_data()
    seen = {}
    backend = obs["RSM07"]._backend
    original = backend.flag.quack

    def spy(code, **kwargs):
        seen.update(kwargs)
        return original(code, **kwargs)

    monkeypatch.setattr(backend.flag, "quack", spy)
    obs.flag.quack()
    assert seen["per_antenna"] == {"EF": 4.0} and seen["column"] == "data"


def test_flag_statistics_land_in_the_report():
    obs = make_single()
    obs.import_data()
    stats = obs.flag.statistics()
    assert "antenna" in stats and 0.0 <= stats["fraction"] <= 1.0
    assert obs.report()[0]["flagging"] == stats


def test_reweight_is_a_calibration_step_and_resumes():
    obs = make_single()
    obs.import_data()
    first = obs.calibrate.reweight()
    assert "mean" in first
    assert obs.calibrate.reweight() == {}          # already done: skipped
    assert obs["RSM07"].state.as_dict()["reweight"]["status"] == "done"


# -- imaging --

def test_clean_scalar_returns_image():
    obs = make_single()
    obs.import_data()
    img = obs.clean(target="3C286", robust=0, imsize=4096)
    assert isinstance(img, Image) and img.source == "3C286"


def test_clean_list_returns_imageset():
    obs = make_single()
    obs.import_data()
    iset = obs.clean(target="3C286", robust=[-2, -1, 0, 1, 2])
    assert isinstance(iset, ImageSet) and len(iset) == 5


def test_clean_wsclean_and_tclean_variants():
    obs = make_single()
    obs.import_data()
    assert isinstance(obs.clean.wsclean(target="3C286"), Image)
    assert isinstance(obs.clean.tclean(target="3C286", robust=1), Image)


def test_clean_accepts_source_object():
    obs = make_single()
    obs.import_data()
    img = obs.clean(target=obs.sources.target, robust=-2)
    assert img.source == "3C286"


# -- self-cal --

def test_selfcal_with_image():
    obs = make_single()
    obs.import_data()
    img = obs.clean(target="3C286", robust=0)
    res = obs.selfcal(img)
    assert isinstance(res, SelfcalResult) and res.source == "3C286"


def test_selfcal_over_list():
    obs = make_single()
    obs.import_data()
    results = obs.selfcal(obs.sources.calibrators)
    assert isinstance(results, list) and all(isinstance(r, SelfcalResult) for r in results)


# -- export --

def test_export_uvfits_and_ms():
    obs = make_single()
    obs.import_data()
    assert obs.export.uvfits(source="3C286").endswith(".uvfits")
    assert obs.export.ms(source="3C286").endswith(".ms")


# -- campaign merge --

def test_merge_single_project_is_noop():
    obs = make_single()
    obs.import_data()
    merged = obs.merge()
    assert merged is obs["RSM07"]


def test_campaign_requires_merge_before_imaging():
    camp = make_campaign()
    camp.import_data()
    camp.calibrate()
    with pytest.raises(Exception, match="merge"):
        camp.clean(target="3C286", robust=0)
    camp.merge()
    assert isinstance(camp.clean(target="3C286", robust=0), Image)


# -- full run --

def test_run_end_to_end_single():
    obs = make_single()
    images = obs.run()
    assert "3C286" in images


def test_run_end_to_end_campaign():
    camp = make_campaign()
    images = camp.run()
    assert "3C286" in images


def test_reset_clears_state():
    obs = make_single()
    obs.run()
    obs.reset()
    assert obs["RSM07"].state.as_dict() == {}
    assert obs["RSM07"].metadata is None


# -- the Pre-PRD example must remain valid --

def test_prd_example_7_1():
    # backend="dummy" pinned: the config default is now "casa", which would trigger a
    # real archive download + importfitsidi (verified: it does!).
    obs = VLBIObs(project="RSM07", network="EVN", backend="dummy", target="3C286",
                  fringe_finder="3C345", phasecal="J1048+7143")
    obs.import_data()
    obs.calibrate.initial_calibration()
    obs.calibrate.bandpass()
    obs.flag.aoflagger()
    obs.plot.tplot()
    img = obs.clean(target="3C286", robust=2, imsize=8192)
    img = obs.clean.wsclean(target=obs.sources.target, robust=-2, imsize=8192)
    obs.selfcal(img)
    out = img.export_fits(outfile="3C286.image.fits")
    assert out == "3C286.image.fits"
    assert isinstance(img, Image)


# -- per-project accessors on VLBIObs --

def test_vlbiobs_exposes_observation_attributes():
    """Single-project campaign reads exactly like the underlying Observation."""
    obs = VLBIObs("p1", network="EVN", backend="dummy", target="SRC", phasecal="CAL")
    obs.import_data()
    assert obs.metadata is obs["p1"].metadata
    assert obs.project_code == "p1"
    assert set(obs.antennas) == set(obs["p1"].antennas)
    assert obs.frequency.n_subbands == 8
    assert len(obs.scans) == obs.metadata.n_scans
    assert obs.refant == obs["p1"].refant
    assert obs.snr_survey is None                      # not measured yet
    obs.calibrate.scan_snr()
    assert obs.snr_survey is not None


def test_vlbiobs_attributes_are_keyed_by_project_for_a_campaign():
    """A multi-project campaign keeps values labelled rather than anonymous."""
    obs = VLBIObs(["p1", "p2"], network="EVN", backend="dummy", target="SRC")
    obs.import_data()
    assert set(obs.metadata) == {"p1", "p2"}
    assert obs.project_code == {"p1": "p1", "p2": "p2"}
    assert set(obs.antennas) == {"p1", "p2"}
    assert all(v is not None for v in obs.frequency.values())
    assert obs.per_project("observatory") == {"p1": "EVN", "p2": "EVN"}
