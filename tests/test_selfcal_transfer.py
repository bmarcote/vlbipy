"""Which fields a calibrator's self-calibration tables go to, without real data."""
from types import SimpleNamespace

import pytest

from vlbipy.backends import difmap
from vlbipy.namespaces import SelfcalNamespace
from vlbipy.phaseref import PhaseRefPlan

FIELDS = ["FF", "PC", "TGT", "CHK"]


def _obs(tmp_path, *, references=("PC",), receivers=("TGT", "CHK"), inverse=False):
    """An observation with one fringe finder FF, a phase calibrator PC, a target TGT and a check source CHK."""
    tables = []
    completed = []
    obs = SimpleNamespace(
        work_dir=tmp_path, project_code="test", observatory="EVN", gaintables=tables, config={},
        sources=SimpleNamespace(fringe_finders=[SimpleNamespace(name="FF")],
                                phase_calibrators=[SimpleNamespace(name="PC")],
                                targets=[SimpleNamespace(name="TGT")], names=FIELDS),
        metadata=SimpleNamespace(source_names=FIELDS),
        _state=SimpleNamespace(should_run=lambda *a, **kw: True,
                               mark_complete=lambda step, **kw: completed.append(step)),
        _backend=SimpleNamespace(ms_path=lambda code: "parent.ms", requires_data_files=True,
                                 supports=lambda *a: False),
        _selfcal_results={},
        clean=SimpleNamespace(split_ms=lambda source, **kw: f"{source}.ms"),
        drop_gaintables=lambda steps: None,
        add_gaintable=lambda table, step: tables.append(table) or table,
    )
    plan = PhaseRefPlan("single", references=list(references), receivers=list(receivers), inverse=inverse)
    obs.phaseref_plan = lambda **kw: plan
    return obs


@pytest.fixture
def difmapy(tmp_path, monkeypatch):
    """A difmapy session that writes the separate phase and amplitude tables and the accumulated one."""
    def calibrate_source(split, parent, prefix, **kwargs):
        for suffix in (".phase", ".amp.G", ".all"):
            (tmp_path / f"{prefix}{suffix}").mkdir(parents=True)
        return {"images": {}, "rounds": [{"accepted": True, "solint": "60s"}],
                "phase_table": f"{prefix}.phase", "amplitude": {"caltable": f"{prefix}.amp.G"},
                "accumulated_table": f"{prefix}.all", "time_amplitude": {"table": ""}}

    monkeypatch.setattr(difmap, "calibrate_source", calibrate_source)


def _chain(obs):
    return {table.cal_type: (table.path.rsplit(".", 1)[-1] if not table.path.endswith(".amp.G") else "amp.G",
                             table.apply_to) for table in obs.gaintables}


def test_fringe_finder_phases_stay_on_the_fringe_finder(tmp_path, difmapy):
    """Only its amplitudes go to every field; the accumulated phase+amplitude table is not transferred."""
    obs = _obs(tmp_path)
    SelfcalNamespace(obs).calibrator("FF")
    assert _chain(obs) == {"selfamp_FF": ("amp.G", ""), "selfphase_FF": ("phase", "FF")}


def test_phase_calibrator_transfers_phases_and_amplitudes(tmp_path, difmapy):
    """The phase calibrator's accumulated table goes to itself, the target and the check source."""
    obs = _obs(tmp_path)
    SelfcalNamespace(obs).calibrator("PC")
    assert _chain(obs) == {"selfcal_PC": ("all", "PC,TGT,CHK")}


def test_transfer_phases_false_keeps_a_phase_calibrator_local(tmp_path, difmapy):
    """The explicit opt-out splits the tables like a fringe finder's."""
    obs = _obs(tmp_path)
    SelfcalNamespace(obs).calibrator("PC", transfer_phases=False)
    assert _chain(obs) == {"selfamp_PC": ("amp.G", "PC,TGT,CHK"), "selfphase_PC": ("phase", "PC")}


def test_fringe_finder_transfer_phases_true_sends_the_accumulated_table_everywhere(tmp_path, difmapy):
    obs = _obs(tmp_path)
    SelfcalNamespace(obs).calibrator("FF", transfer_phases=True)
    assert _chain(obs) == {"selfcal_FF": ("all", "")}


@pytest.mark.parametrize("inverse, expected", [(False, ["FF", "PC"]), (True, ["FF", "TGT"])])
def test_run_all_self_calibrates_the_phase_reference_sources(tmp_path, monkeypatch, inverse, expected):
    """With inverse phase referencing the stage self-calibrates the target, not the phase calibrator."""
    obs = _obs(tmp_path, references=("TGT",) if inverse else ("PC",),
               receivers=("PC", "CHK") if inverse else ("TGT", "CHK"), inverse=inverse)
    obs.calibrate = SimpleNamespace(apply=lambda **kw: None)
    obs.export = SimpleNamespace(per_source=lambda **kw: {})
    sessions = []
    monkeypatch.setattr(SelfcalNamespace, "calibrator", lambda self, name, **kw: sessions.append(name) or name)
    SelfcalNamespace(obs).run_all()
    assert sessions == expected


def test_inverse_target_tables_go_to_the_phase_calibrator(tmp_path, difmapy):
    """The target's self-calibration is transferred to the phase calibrator and the check source."""
    obs = _obs(tmp_path, references=("TGT",), receivers=("PC", "CHK"), inverse=True)
    SelfcalNamespace(obs).calibrator("TGT")
    assert _chain(obs) == {"selfcal_TGT": ("all", "PC,TGT,CHK")}


def test_run_all_carries_on_past_a_calibrator_that_cannot_be_self_calibrated(tmp_path, monkeypatch):
    """A fringe finder without unflagged data must not cost the phase calibrator its session."""
    obs = _obs(tmp_path)
    obs.calibrate = SimpleNamespace(apply=lambda **kw: None)
    obs.export = SimpleNamespace(per_source=lambda **kw: {})
    failed = {}
    obs._state.mark_failed = lambda step, error: failed.update({step: error})

    def calibrator(self, name, **kw):
        if name == "FF":
            raise RuntimeError("no unflagged data to estimate a resolution from")
        return name

    monkeypatch.setattr(SelfcalNamespace, "calibrator", calibrator)
    assert SelfcalNamespace(obs).run_all() == {"PC": "PC"}
    assert failed == {"selfcal_FF": "no unflagged data to estimate a resolution from"}


def test_run_all_fails_when_no_calibrator_can_be_self_calibrated(tmp_path, monkeypatch):
    obs = _obs(tmp_path)
    obs.calibrate = SimpleNamespace(apply=lambda **kw: None)
    obs._state.mark_failed = lambda step, error: None

    def calibrator(self, name, **kw):
        raise RuntimeError(f"nothing on {name}")

    monkeypatch.setattr(SelfcalNamespace, "calibrator", calibrator)
    with pytest.raises(RuntimeError, match="nothing on FF"):
        SelfcalNamespace(obs).run_all()


def test_the_total_amplitude_correction_multiplies_every_table_that_reaches_the_target():
    """Scalar bandpass x the fringe finders' amplitudes x the phase calibrator's; its local time curve is left out."""
    from vlbipy import dashboard
    tables = [{"cal_type": "scalar_bp", "path": "x", "apply_to": ""},
              {"cal_type": "selfamp_FF", "apply_to": ""}, {"cal_type": "selfphase_FF", "apply_to": "FF"},
              {"cal_type": "selfampt_FF", "apply_to": "FF"},
              {"cal_type": "selfcal_PC", "apply_to": "PC,TGT,CHK"}, {"cal_type": "selfampt_PC", "apply_to": "PC"}]
    entries = [{"source": "FF", "antennas": ["AA", "BB"], "subbands": [0, 1],
                "gain_per_if": {"AA": [2.0, 2.0], "BB": [None, None]}},
               {"source": "PC", "antennas": ["AA", "BB"], "subbands": [0, 1],
                "gain_per_if": {"AA": [0.5, 0.25], "BB": [1.5, 1.5]}}]
    antennas = [{"name": "AA", "observed": True, "subbands": [0, 1]}, {"name": "BB", "observed": True, "subbands": [1]}]
    scalar = {"AA": {(0, 0): 1.0, (0, 1): 1.1, (1, 0): 1.0, (1, 1): 1.0}}
    entry = dashboard.gain_total_entry(tables, entries, ["TGT", "CHK"], antennas, [[1.6e9], [1.7e9]], ["R", "L"],
                                       scalar)
    assert entry["factors"] == ["scalar_bp", "selfamp_FF", "selfcal_PC"]
    assert entry["subband_labels"] == ["SB 0 R", "SB 0 L", "SB 1 R", "SB 1 L"]
    assert entry["gain_per_if"]["AA"] == pytest.approx([1.0, 1.1, 0.5, 0.5])
    assert entry["gain_per_if"]["BB"] == [None, None, 1.5, 1.5]          # BB did not record subband 0
    obs = SimpleNamespace(phaseref_plan=lambda: PhaseRefPlan("single", references=["PC"], receivers=["TGT", "CHK"]),
                          sources=SimpleNamespace(fringe_finders=[SimpleNamespace(name="FF")], targets=[]))
    totals = dashboard._gain_totals(obs, [dict(t, path="") for t in tables], entries, antennas, [[1.6e9], [1.7e9]],
                                    ["RR", "RL", "LR", "LL"])
    assert [t["fields"] for t in totals] == [["PC", "TGT", "CHK"]]        # one chain of amplitudes for all three
    html = dashboard._gain_section({"gain_total": totals, "gain_corrections": entries})
    assert html.index("Total") < html.index(">FF ") and 'class="active" data-panel="gains-total' in html


def test_selfcal_amplitudes_are_per_station_and_subband(tmp_path, monkeypatch):
    """[selfcal]: the Bayesian gscale corrects every subband, each around its station's level (bayes_if_sigma 0.1);
    bayes_if_sigma = 0 judges every subband on its own."""
    seen = []

    def calibrate_source(split, parent, prefix, **kwargs):
        seen.append(kwargs)
        (tmp_path / f"{prefix}.amp.G").mkdir(parents=True, exist_ok=True)
        return {"images": {}, "rounds": [], "phase_table": "", "amplitude": {"caltable": f"{prefix}.amp.G"},
                "accumulated_table": "", "time_amplitude": {"table": ""}}

    monkeypatch.setattr(difmap, "calibrate_source", calibrate_source)
    SelfcalNamespace(_obs(tmp_path)).calibrator("FF")
    assert seen[-1]["bayes_per_if"] is True and seen[-1]["bayes_if_sigma"] == pytest.approx(0.1)
    obs = _obs(tmp_path)
    obs.config = {"selfcal": {"bayes_if_sigma": 0}}
    SelfcalNamespace(obs).calibrator("FF")
    assert seen[-1]["bayes_if_sigma"] is None
