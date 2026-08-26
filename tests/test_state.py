"""Tests for the in-memory step-state / smart re-run logic."""
from vlbipy import load_config
from vlbipy.state import StepState


def test_should_run_and_skip():
    st = StepState("P")
    assert st.should_run("bandpass") is True          # not done yet -> run
    st.mark_complete("bandpass")
    assert st.should_run("bandpass") is False          # up-to-date -> skip
    assert st.should_run("bandpass", force=True) is True  # forced -> run
    assert st.status("bandpass") == "done"


def test_invalidate_downstream():
    order = ["a", "b", "c", "d"]
    st = StepState("P")
    for s in order:
        st.mark_complete(s)
    st.invalidate_downstream("b", order)
    assert st.status("a") == "done"
    assert st.status("b") is None
    assert st.status("c") is None
    assert st.status("d") is None


def test_mark_failed_and_reset():
    st = StepState("P")
    st.mark_failed("fringefit", "no solutions")
    assert st.status("fringefit") == "failed"
    st.reset()
    assert st.as_dict() == {}


def test_caltable_json_round_trip():
    from vlbipy.models import CalTable
    table = CalTable("mbd", path="/x/rsm07.mbd", field="3C345", gainfield="J1848+3219",
                     interp="linear", spwmap=[0, 0, 0, 0], snr=12.5, step="fringefit")
    assert CalTable.from_dict(table.to_dict()) == table
    # Unknown keys (an older or newer file) are ignored rather than raising.
    assert CalTable.from_dict({"cal_type": "tsys", "unexpected": 1}).cal_type == "tsys"


def test_gaintables_survive_a_new_process(tmp_path, monkeypatch):
    """A resumed run must solve on top of the chain earlier steps produced."""
    from vlbipy.models import CalTable
    from vlbipy.observation import Observation

    def make():
        obs = Observation("ts_chain", load_config(overrides={"global": {"backend": "casa"}}),
                          work_dir=str(tmp_path))
        return obs

    obs = make()
    obs.add_gaintable(CalTable("tsys", path="a"), "a_priori")
    obs.add_gaintable(CalTable("sbd", path="b"), "initial_calibration")
    obs.add_gaintable(CalTable("mbd", path="c"), "fringefit")

    resumed = make()
    assert [t.cal_type for t in resumed.gaintables] == ["tsys", "sbd", "mbd"]
    assert resumed.gaintables[2].step == "fringefit"


def test_resuming_from_a_step_drops_the_tables_it_will_redo(tmp_path):
    from vlbipy.models import CalTable
    from vlbipy.observation import Observation

    cfg = load_config(overrides={"global": {"backend": "casa"}})
    obs = Observation("ts_drop", cfg, work_dir=str(tmp_path))
    obs.add_gaintable(CalTable("tsys"), "a_priori")
    obs.add_gaintable(CalTable("sbd"), "initial_calibration")
    obs.add_gaintable(CalTable("mbd"), "fringefit")
    obs.add_gaintable(CalTable("bpass"), "bandpass")

    obs.prepare_run(from_step="fringefit")
    assert [t.cal_type for t in obs.gaintables] == ["tsys", "sbd"]
    # ...and the drop is persisted, not just in memory.
    assert [t.cal_type for t in Observation("ts_drop", cfg, work_dir=str(tmp_path)).gaintables] \
        == ["tsys", "sbd"]


def test_scratch_clears_the_persisted_chain(tmp_path):
    from vlbipy.models import CalTable
    from vlbipy.observation import Observation

    cfg = load_config(overrides={"global": {"backend": "casa"}})
    obs = Observation("ts_scratch", cfg, work_dir=str(tmp_path))
    obs.add_gaintable(CalTable("tsys"), "a_priori")
    obs.prepare_run(scratch=True)
    assert obs.gaintables == []
    assert Observation("ts_scratch", cfg, work_dir=str(tmp_path)).gaintables == []


def test_step_order_matches_the_names_steps_record():
    """Every step name the pipeline records must be in STEP_ORDER, exactly once."""
    from vlbipy.observation import STEP_ORDER
    assert len(STEP_ORDER) == len(set(STEP_ORDER))
    for name in ("import_data", "a_priori", "flag_apriori", "flag_from_file", "flag_autocorr",
                 "scan_snr", "initial_calibration", "fringefit", "bandpass",
                 "initial_calibration_sbd2", "fringefit_mbd2", "edge_channels", "flag_edges",
                 "apply", "flag_quack", "flag_outliers", "second_pass", "scalar_bandpass",
                 "split"):
        assert name in STEP_ORDER, name
    # SBD is solved before the bandpass, and the bandpass before the refined SBD.
    assert STEP_ORDER.index("initial_calibration") < STEP_ORDER.index("fringefit") \
        < STEP_ORDER.index("bandpass") < STEP_ORDER.index("initial_calibration_sbd2")
