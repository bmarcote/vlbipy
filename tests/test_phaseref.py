"""Phase-referencing strategies: settings, plan, geometry and the tables interpolated between two references."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlbipy import phaseref
from vlbipy.errors import ConfigError
from vlbipy.models import CalTable
from vlbipy.phaseref import PhaseRefPlan, PhaseRefSettings, resolve_plan
from vlbipy.sources import SourceSet

# -- settings and plan --------------------------------------------------------------------------------------------


def test_settings_defaults_and_validation():
    assert PhaseRefSettings.from_config({}) == PhaseRefSettings()
    settings = PhaseRefSettings.from_config({"strategy": "Interpolate", "order": "CAL2", "inverse": True})
    assert (settings.strategy, settings.order, settings.inverse) == ("interpolate", ("CAL2",), True)
    for bad in ({"strategy": "nearest"}, {"brightness_ratio": 0.5}, {"inverse": "yes"}, {"max_extrapolation": -1}):
        with pytest.raises(ConfigError):
            PhaseRefSettings.from_config(bad)


def test_the_settings_are_not_targets_of_the_phase_referencing_map():
    sources = SourceSet.from_config({"targets": ["T1"], "phase_calibrators": ["C1"]},
                                    {"strategy": "auto", "inverse": False, "order": [], "T1": "C1"})
    assert sources.phase_referencing == {"T1": ["C1"]}


def test_auto_is_sequential_only_for_a_much_brighter_reference():
    settings = PhaseRefSettings()
    plan = resolve_plan(settings, ["FAINT", "BRIGHT"], ["T"], {"FAINT": 10.0, "BRIGHT": 40.0})
    assert (plan.strategy, plan.references) == ("sequential", ["BRIGHT", "FAINT"])
    plan = resolve_plan(settings, ["FAINT", "BRIGHT"], ["T"], {"FAINT": 20.0, "BRIGHT": 40.0})
    assert (plan.strategy, plan.references) == ("interpolate", ["BRIGHT", "FAINT"])
    # Without the brightness of every reference nothing can be ranked: the symmetric strategy.
    plan = resolve_plan(settings, ["FAINT", "BRIGHT"], ["T"], {"BRIGHT": 40.0})
    assert (plan.strategy, plan.references) == ("interpolate", ["FAINT", "BRIGHT"])
    assert resolve_plan(settings, ["ONLY"], ["T"], {}).strategy == "single"


def test_explicit_strategy_and_order():
    settings = PhaseRefSettings(strategy="sequential", order=("C2",))
    plan = resolve_plan(settings, ["C1", "C2", "C3"], ["T"], {"C1": 50.0, "C2": 5.0, "C3": 30.0})
    assert plan.references == ["C2", "C1", "C3"]                   # listed first, the rest by brightness
    assert resolve_plan(PhaseRefSettings(strategy="time"), ["C1", "C2"], ["T"], {}).strategy == "time"


def test_where_the_tables_of_each_reference_go():
    everyone = ["C1", "T", "C2", "CHECK"]
    sequential = PhaseRefPlan("sequential", references=["C1", "C2"], receivers=["T", "CHECK"])
    assert sequential.apply_to("C1", everyone) == "C1,T,C2,CHECK"
    assert sequential.apply_to("C2", everyone) == "T,C2,CHECK"     # never back onto the brighter C1
    interpolate = PhaseRefPlan("interpolate", references=["C1", "C2"], receivers=["T", "CHECK"])
    assert interpolate.apply_to("C1", everyone) == "C1,T,CHECK"
    assert PhaseRefPlan("time", references=["C1", "C2"]).apply_to("C1", everyone) is None
    assert sequential.apply_to("T", everyone) is None


def test_plan_round_trip():
    plan = PhaseRefPlan("sequential", ["A", "B"], ["T"], inverse=True, brightness={"A": 3.0}, reason="r")
    assert PhaseRefPlan.from_dict(plan.to_dict()) == plan


# -- geometry ------------------------------------------------------------------------------------------------------


def test_projection_on_the_line_and_beyond():
    a, b = (10.0, 0.0), (12.0, 0.0)
    x, offset = phaseref.projection(a, b, (11.0, 0.5))
    assert x == pytest.approx(0.5, abs=1e-3) and offset == pytest.approx(0.5, abs=1e-2)
    assert phaseref.projection(a, b, (13.0, 0.0))[0] == pytest.approx(1.5, abs=1e-3)
    assert phaseref.projection(a, b, (9.0, 0.0))[0] == pytest.approx(-0.5, abs=1e-3)


def test_v589_target_lies_halfway_between_its_calibrators():
    coords = {"J1346-6024": (206.70434677, -60.40815421), "J1355-6326": (208.94421759, -63.44515908),
              "J1405-6118": (211.31008333, -61.30786944)}
    a, b, x, offset = phaseref.interpolation_pair(["J1346-6024", "J1355-6326"], coords, "J1405-6118")
    assert {a, b} == {"J1346-6024", "J1355-6326"}
    assert x == pytest.approx(0.5, abs=0.01) and offset == pytest.approx(1.8, abs=0.05)


def test_interpolation_pair_takes_the_two_nearest_and_limits_the_extrapolation():
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "FAR": (40.0, 0.0), "T": (4.0, 0.0)}
    a, b, x, _ = phaseref.interpolation_pair(["FAR", "A", "B"], coords, "T", max_extrapolation=1.0)
    assert (a, b) == ("B", "A") and x == pytest.approx(-1.0)     # x = -3 clipped to -1


def test_the_phase_difference_is_unwrapped_inside_a_scan_only():
    times = np.concatenate([np.arange(0, 120, 30.0), 1000 + np.arange(0, 120, 30.0)])
    delta = np.array([3.0, 3.1, -3.1, -3.0, 2.0, 2.2, 2.4, 2.6])     # wraps inside the first scan
    out = phaseref._unwrap_in_time(delta, np.zeros(8, bool), np.zeros(8, int), times)
    assert np.allclose(np.diff(out[:4]), [0.1, 2 * np.pi - 6.2, 0.1])  # continuous through +-pi
    assert np.allclose(out[4:], delta[4:])                         # the next scan starts on its own branch


# -- derived tables --------------------------------------------------------------------------------------------------

casatools = pytest.importorskip("casatools")
from vlbipy.solvers import caltable  # noqa: E402
from vlbipy.solvers.apply import load_caltable  # noqa: E402

T0 = 5.0e9
FIELDS = ["CAL_A", "CAL_B", "TARGET"]
NANT = 3
CHAN_FREQ = np.array([8.40e9, 8.42e9])


def _column(value_type, ndim=0, shape=None):
    desc = {"valueType": value_type, "dataManagerType": "StandardStMan", "dataManagerGroup": "StandardStMan",
            "option": 0, "maxlen": 0, "comment": ""}
    if ndim:
        desc["ndim"] = ndim
        if shape is not None:
            desc["shape"] = np.array(shape)
            desc["option"] = 5
    return desc


def _create(path, desc, nrow, columns):
    handle = casatools.table()
    handle.create(str(path), desc)
    handle.addrows(nrow)
    for name, values in columns.items():
        handle.putcol(name, values)
    handle.close()


@pytest.fixture
def three_field_ms(tmp_path):
    """The subtables a caltable copies: ANTENNA (3), FIELD (CAL_A, CAL_B, TARGET), OBSERVATION."""
    ms = tmp_path / "three.ms"
    ms.mkdir()
    names = np.array(["AA", "BB", "CC"])
    _create(ms / "ANTENNA", {"NAME": _column("string"), "STATION": _column("string"), "TYPE": _column("string"),
                             "MOUNT": _column("string"), "POSITION": _column("double", 1, [3]),
                             "OFFSET": _column("double", 1, [3]), "DISH_DIAMETER": _column("double"),
                             "FLAG_ROW": _column("boolean")}, NANT,
            {"NAME": names, "STATION": names, "TYPE": np.array(["GROUND-BASED"] * NANT),
             "MOUNT": np.array(["ALT-AZ"] * NANT), "POSITION": np.arange(9.0).reshape(3, 3).T,
             "OFFSET": np.zeros((3, NANT)), "DISH_DIAMETER": np.full(NANT, 25.0), "FLAG_ROW": np.zeros(NANT, bool)})
    dirs = np.radians(np.array([[[10.0, 0.0]], [[12.0, 0.0]], [[10.5, 0.2]]])).T
    _create(ms / "FIELD", {"NAME": _column("string"), "CODE": _column("string"), "SOURCE_ID": _column("int"),
                           "NUM_POLY": _column("int"), "TIME": _column("double"), "FLAG_ROW": _column("boolean"),
                           "PHASE_DIR": _column("double", 2), "DELAY_DIR": _column("double", 2),
                           "REFERENCE_DIR": _column("double", 2)}, 3,
            {"NAME": np.array(FIELDS), "CODE": np.array([""] * 3), "SOURCE_ID": np.arange(3, dtype=np.int32),
             "NUM_POLY": np.zeros(3, np.int32), "TIME": np.zeros(3), "FLAG_ROW": np.zeros(3, bool),
             "PHASE_DIR": dirs, "DELAY_DIR": dirs, "REFERENCE_DIR": dirs})
    _create(ms / "OBSERVATION", {"TELESCOPE_NAME": _column("string"), "OBSERVER": _column("string"),
                                 "PROJECT": _column("string"), "RELEASE_DATE": _column("double"),
                                 "FLAG_ROW": _column("boolean"), "TIME_RANGE": _column("double", 1, [2]),
                                 "SCHEDULE_TYPE": _column("string")}, 1,
            {"TELESCOPE_NAME": np.array(["LBA"]), "OBSERVER": np.array(["me"]), "PROJECT": np.array(["TEST"]),
             "RELEASE_DATE": np.zeros(1), "FLAG_ROW": np.zeros(1, bool), "TIME_RANGE": np.array([[T0], [T0 + 3600]]),
             "SCHEDULE_TYPE": np.array([""])})
    chan = (CHAN_FREQ[0] + 2e6 * np.arange(8))[:, None]
    _create(ms / "SPECTRAL_WINDOW", caltable.spectral_window_table_desc(), 1, {
        "CHAN_FREQ": chan, "CHAN_WIDTH": np.full((8, 1), 2e6), "EFFECTIVE_BW": np.full((8, 1), 2e6),
        "RESOLUTION": np.full((8, 1), 2e6), "FLAG_ROW": np.zeros(1, bool), "FREQ_GROUP": np.zeros(1, np.int32),
        "FREQ_GROUP_NAME": np.array([""]), "IF_CONV_CHAIN": np.zeros(1, np.int32),
        "MEAS_FREQ_REF": np.full(1, 5, np.int32), "NAME": np.array(["IF0"]), "NET_SIDEBAND": np.ones(1, np.int32),
        "NUM_CHAN": np.full(1, 8, np.int32), "REF_FREQUENCY": chan[0], "TOTAL_BANDWIDTH": np.full(1, 16e6)})
    return ms


def _fringe_table(path, ms, *, phase, delay, flagged=()):
    """Solutions of CAL_A (field 0) and CAL_B (field 1) at T0 and T0 + 600 s, one spw, constant in time.

    ``phase`` / ``delay`` map a field id to its value; ``flagged`` lists (field, antenna) rows to flag.
    """
    rows = [(field, antenna, time) for field in (0, 1) for antenna in range(NANT) for time in (T0, T0 + 600)]
    nrow = len(rows)
    fparam = np.zeros((nrow, 8))
    flag = np.zeros((nrow, 8), bool)
    for index, (field, antenna, _) in enumerate(rows):
        fparam[index, [0, 4]] = phase[field] if antenna else 0.0          # antenna 0 is the reference
        fparam[index, [1, 5]] = delay[field] if antenna else 0.0
        flag[index] = (field, antenna) in flagged
    caltable.write_fringe_table(path, ms, times=[r[2] for r in rows], field_ids=[r[0] for r in rows],
                                spw_ids=np.zeros(nrow, int), antenna_ids=[r[1] for r in rows], refant_id=0,
                                scan_numbers=np.ones(nrow, int), fparam=fparam, paramerr=np.zeros((nrow, 8)),
                                flag=flag, snr=np.full((nrow, 8), 50.0), spw_chan_freq=CHAN_FREQ[:1],
                                spw_chan_width=np.array([16e6]))
    return str(path)


def _gain_table(path, ms, field, gain, times=None):
    times = np.array([T0, T0 + 600]) if times is None else np.asarray(times, dtype=np.float64)
    rows = [(antenna, time) for antenna in range(NANT) for time in times]
    cparam = np.full((len(rows), 1, 2), gain, dtype=complex)
    caltable.write_jones_table(path, ms, viscal="G Jones", times=[r[1] for r in rows],
                               field_ids=np.full(len(rows), field), spw_ids=np.zeros(len(rows), int),
                               antenna_ids=[r[0] for r in rows], scan_numbers=np.ones(len(rows), int),
                               cparam=cparam, flag=np.zeros(cparam.shape, bool), snr=np.full(cparam.shape, 50.0),
                               spw_chan_freq=CHAN_FREQ[:1], spw_chan_width=np.array([16e6]))
    return str(path)


def _target_value(path, column, antenna=1, hand=0):
    table = load_caltable(path)
    rows = (table.antenna1 == antenna) & (table.field_id == 2)
    assert rows.any() and (table.field_id == 2).all()
    return np.asarray(table.param[rows, 0, column]), table.flag[rows, 0, column]


def test_fringe_solutions_combine_linearly(three_field_ms, tmp_path):
    source = _fringe_table(tmp_path / "mbd", three_field_ms, phase={0: 0.2, 1: 1.2}, delay={0: 1.0, 1: 3.0})
    times = T0 + np.arange(0, 600.1, 60.0)
    summary = phaseref.interpolated_table((source, "CAL_A"), (source, "CAL_B"), 0.25, "TARGET", times,
                                          tmp_path / "derived")
    assert summary["kind"] == "Fringe Jones" and summary["flagged"] == 0.0
    phase, flags = _target_value(tmp_path / "derived", 0)
    delay, _ = _target_value(tmp_path / "derived", 1)
    assert not flags.any()
    assert np.allclose(np.real(phase), 0.2 + 0.25 * 1.0) and np.allclose(np.real(delay), 1.5)
    # Beyond CAL_B the phases and delays are extrapolated.
    phaseref.interpolated_table((source, "CAL_A"), (source, "CAL_B"), 1.5, "TARGET", times, tmp_path / "beyond")
    assert np.allclose(np.real(_target_value(tmp_path / "beyond", 1)[0]), 4.0)


def test_a_reference_without_a_solution_leaves_the_other(three_field_ms, tmp_path):
    source = _fringe_table(tmp_path / "mbd", three_field_ms, phase={0: 0.2, 1: 1.2}, delay={0: 1.0, 1: 3.0},
                           flagged={(1, 2)})
    phaseref.interpolated_table((source, "CAL_A"), (source, "CAL_B"), 0.25, "TARGET", T0 + np.arange(0, 601, 300.0),
                                tmp_path / "derived")
    delay, flags = _target_value(tmp_path / "derived", 1, antenna=2)
    assert not flags.any() and np.allclose(np.real(delay), 1.0)      # CAL_A's alone


def test_self_calibration_gains_combine_with_amplitudes_not_extrapolated(three_field_ms, tmp_path):
    first = _gain_table(tmp_path / "a.phase", three_field_ms, 0, np.exp(0.1j))
    second = _gain_table(tmp_path / "b.phase", three_field_ms, 1, 2.0 * np.exp(0.5j))
    times = T0 + np.arange(0, 601, 120.0)
    phaseref.interpolated_table((first, "CAL_A"), (second, "CAL_B"), 0.25, "TARGET", times, tmp_path / "derived")
    gain, _ = _target_value(tmp_path / "derived", 0)
    assert np.allclose(np.angle(gain), 0.2) and np.allclose(np.abs(gain), 2 ** 0.25, rtol=1e-5)
    phaseref.interpolated_table((first, "CAL_A"), (second, "CAL_B"), 1.5, "TARGET", times, tmp_path / "beyond")
    gain, _ = _target_value(tmp_path / "beyond", 0)
    assert np.allclose(np.angle(gain), 0.7) and np.allclose(np.abs(gain), 2.0, rtol=1e-5)


def test_transfer_chain_gives_the_receiver_interpolated_tables(three_field_ms, tmp_path):
    mbd = _fringe_table(tmp_path / "mbd", three_field_ms, phase={0: 0.2, 1: 1.2}, delay={0: 1.0, 1: 3.0})
    phase_a = _gain_table(tmp_path / "a.phase", three_field_ms, 0, np.exp(0.1j))
    phase_b = _gain_table(tmp_path / "b.phase", three_field_ms, 1, np.exp(0.5j))
    chain = [CalTable("mbd", path=mbd, field="CAL_A,CAL_B", gainfield="CAL_A,CAL_B", interp="linear"),
             CalTable("selfphase_CAL_A", path=phase_a, field="CAL_A", gainfield="CAL_A", apply_to="CAL_A,TARGET"),
             CalTable("selfphase_CAL_B", path=phase_b, field="CAL_B", gainfield="CAL_B", apply_to="CAL_B,TARGET")]
    metadata = SimpleNamespace(source_names=FIELDS,
                               source_coords={"CAL_A": (10.0, 0.0), "CAL_B": (12.0, 0.0), "TARGET": (10.5, 0.2)},
                               scans=[SimpleNamespace(source="TARGET", time_start=T0 + 100, time_end=T0 + 220)])
    plan = PhaseRefPlan("interpolate", references=["CAL_A", "CAL_B"], receivers=["TARGET"])
    out = phaseref.transfer_chain(chain, plan, metadata, tmp_path / "phaseref")
    by_type = {t.cal_type: t for t in out}
    assert by_type["mbd"].apply_to == "CAL_A,CAL_B"
    assert by_type["selfphase_CAL_A"].apply_to == "CAL_A" and by_type["selfphase_CAL_B"].apply_to == "CAL_B"
    assert by_type["mbd>TARGET"].apply_to == "TARGET" and by_type["mbd>TARGET"].gainfield == "TARGET"
    assert by_type["selfphase>TARGET"].apply_to == "TARGET"
    assert chain[0].apply_to == ""                                  # the registered chain is left alone
    gain, _ = _target_value(by_type["selfphase>TARGET"].path, 0)
    assert np.allclose(np.angle(gain), 0.1 + 0.25 * 0.4, atol=1e-3)
    # A second call with unchanged inputs reuses the tables on disk.
    stamp = Path(by_type["mbd>TARGET"].path + ".json").stat().st_mtime_ns
    phaseref.transfer_chain(chain, plan, metadata, tmp_path / "phaseref")
    assert Path(by_type["mbd>TARGET"].path + ".json").stat().st_mtime_ns == stamp
    # Any other strategy keeps the chain as solved.
    assert phaseref.transfer_chain(chain, PhaseRefPlan("time", ["CAL_A", "CAL_B"], ["TARGET"]), metadata,
                                   tmp_path / "phaseref") == chain


def test_transfer_chain_interpolates_the_accumulated_selfcal_tables(three_field_ms, tmp_path):
    """The single complex ``selfcal_<source>`` table the difmapy stage registers (phase + amplitude
    accumulated over the session) is paired by kind like the selfamp/selfphase pair and the campaign's
    ``joint_<source>``: without it the receiver takes both calibrators' tables in series whatever the
    strategy says."""
    cal_a = _gain_table(tmp_path / "a.gains", three_field_ms, 0, np.exp(0.1j))
    cal_b = _gain_table(tmp_path / "b.gains", three_field_ms, 1, 2.0 * np.exp(0.5j))
    joint_a = _gain_table(tmp_path / "a.joint", three_field_ms, 0, np.exp(0.3j))
    joint_b = _gain_table(tmp_path / "b.joint", three_field_ms, 1, np.exp(0.7j))
    chain = [CalTable("selfcal_CAL_A", path=cal_a, field="CAL_A", gainfield="CAL_A", interp="linear",
                      step="selfcal_CAL_A", apply_to="CAL_A,TARGET"),
             CalTable("selfcal_CAL_B", path=cal_b, field="CAL_B", gainfield="CAL_B", interp="linear",
                      step="selfcal_CAL_B", apply_to="CAL_B,TARGET"),
             CalTable("joint_CAL_A", path=joint_a, field="CAL_A", gainfield="CAL_A", interp="linear",
                      step="joint_CAL_A", apply_to="CAL_A,TARGET"),
             CalTable("joint_CAL_B", path=joint_b, field="CAL_B", gainfield="CAL_B", interp="linear",
                      step="joint_CAL_B", apply_to="CAL_B,TARGET")]
    metadata = SimpleNamespace(source_names=FIELDS,
                               source_coords={"CAL_A": (10.0, 0.0), "CAL_B": (12.0, 0.0),
                                              "TARGET": (10.5, 0.2)},
                               scans=[SimpleNamespace(source="TARGET", time_start=T0 + 100,
                                                      time_end=T0 + 220)])
    plan = PhaseRefPlan("interpolate", references=["CAL_A", "CAL_B"], receivers=["TARGET"])
    _, _, x, _ = phaseref.interpolation_pair(plan.references, metadata.source_coords, "TARGET")
    out = phaseref.transfer_chain(chain, plan, metadata, tmp_path / "phaseref")
    by_type = {t.cal_type: t for t in out}
    assert by_type["selfcal>TARGET"].apply_to == "TARGET" and by_type["selfcal>TARGET"].gainfield == "TARGET"
    assert by_type["joint>TARGET"].apply_to == "TARGET"
    assert by_type["selfcal_CAL_A"].apply_to == "CAL_A" and by_type["selfcal_CAL_B"].apply_to == "CAL_B"
    assert by_type["joint_CAL_A"].apply_to == "CAL_A" and by_type["joint_CAL_B"].apply_to == "CAL_B"
    gain, flags = _target_value(by_type["selfcal>TARGET"].path, 0)
    assert not flags.any()
    assert np.allclose(np.angle(gain), 0.1 + x * 0.4, atol=1e-3)
    assert np.allclose(np.abs(gain), 2 ** x, rtol=1e-5)


def test_selfcal_tables_fall_back_to_the_nearer_reference_in_time(three_field_ms, tmp_path):
    """CAL_A's accumulated table is solved at T0 only and CAL_B's brackets T0 + 575 s; with a small max_gap
    the derived rows far from either take the solutions of the reference observed closer in time."""
    first = _gain_table(tmp_path / "a.gains", three_field_ms, 0, np.exp(0.1j), times=[T0])
    second = _gain_table(tmp_path / "b.gains", three_field_ms, 1, np.exp(0.5j), times=[T0 + 550, T0 + 600])
    times = np.array([T0 + 50.0, T0 + 575.0])
    summary = phaseref.interpolated_table((first, "CAL_A"), (second, "CAL_B"), 0.25, "TARGET", times,
                                          tmp_path / "derived", max_gap=100.0)
    assert summary["time_interpolated"] == pytest.approx(1.0)
    table = load_caltable(tmp_path / "derived")
    rows = table.antenna1 == 1
    gain = table.param[rows, 0, 0][np.argsort(table.time[rows])]
    assert np.allclose(np.angle(gain), [0.1, 0.5], atol=1e-3)


# -- pipeline input ----------------------------------------------------------------------------------------------------


def test_cli_flags_reach_the_phase_referencing_section(tmp_path):
    from vlbipy.cli import _build_obs, _relocate_selectors, build_parser
    config = tmp_path / "c.toml"
    config.write_text('[global]\nbackend = "dummy"\nobservatory = "EVN"\nwork_dir = "%s"\n' % tmp_path)
    args = _relocate_selectors(build_parser().parse_args(
        ["pipeline", "-p", "TS01", "--backend", "dummy", "-t", "SRC", "--phasecal", "C1", "C2",
         "--inverse-phaseref", "--phaseref-strategy", "sequential", "--config", str(config)]))
    obs = _build_obs(args)
    section = obs.config["phase_referencing"]
    assert (section["inverse"], section["strategy"]) == (True, "sequential")
    single = obs["TS01"]
    assert single.phaseref_settings.inverse
    plan = single.phaseref_plan()
    assert plan.references == ["SRC"] and set(plan.receivers) >= {"C1", "C2"}


def test_python_keywords_reach_the_phase_referencing_section():
    from vlbipy import VLBIObs
    obs = VLBIObs("TS02", network="EVN", backend="dummy", target="SRC", phasecal=["C1", "C2"],
                  phaseref_strategy="interpolate")
    assert obs.config["phase_referencing"]["strategy"] == "interpolate"
    assert obs.config["phase_referencing"]["inverse"] is False      # the default, untouched
    plan = obs["TS02"].phaseref_plan()
    assert (plan.strategy, plan.references, plan.inverse) == ("interpolate", ["C1", "C2"], False)


# -- time sampling of the references ------------------------------------------------------------------------------


def test_bracket_gaps():
    gaps = phaseref.bracket_gaps([100.0, 400.0], [50.0, 150.0, 400.0, 500.0])
    assert np.isinf(gaps[0]) and gaps[1] == 250.0 and gaps[2] == 0.0 and np.isinf(gaps[3])


def _scan(source, start, length=120.0):
    return SimpleNamespace(source=source, time_start=start, time_end=start + length)


def test_auto_interpolates_only_when_both_references_bracket_the_targets():
    coords = {"A": (10.0, 0.0), "B": (12.0, 0.0), "T": (11.0, 0.3)}
    # MultiView-like cycle: A T B every 5 minutes.
    cycle = [_scan(name, 300.0 * k + 100.0 * i) for k in range(10) for i, name in enumerate("ATB")]
    dense = phaseref.median_sampling(cycle, ["A", "B"], ["T"], coords)
    # V589-like: A T A, then a check source, then B T B ..., each target scan paired with one calibrator.
    paired = []
    for k in range(10):
        a = "A" if k % 2 == 0 else "B"
        paired += [_scan(a, 900.0 * k), _scan("T", 900.0 * k + 150), _scan(a, 900.0 * k + 300)]
    sparse = phaseref.median_sampling(paired, ["A", "B"], ["T"], coords)
    assert dense < 600.0 < sparse
    settings, bright = PhaseRefSettings(), {"A": 30.0, "B": 20.0}
    assert resolve_plan(settings, ["A", "B"], ["T"], bright, sampling=dense).strategy == "interpolate"
    plan = resolve_plan(settings, ["A", "B"], ["T"], bright, sampling=sparse)
    assert plan.strategy == "time" and "too sparse" in plan.reason
    # A brighter-by-far reference is sequential whatever the sampling; an explicit strategy is kept.
    assert resolve_plan(settings, ["A", "B"], ["T"], {"A": 90.0, "B": 20.0}, sampling=sparse).strategy == "sequential"
    assert resolve_plan(PhaseRefSettings(strategy="interpolate"), ["A", "B"], ["T"], bright,
                        sampling=sparse).strategy == "interpolate"


def test_times_far_from_a_reference_take_the_time_interpolation(three_field_ms, tmp_path):
    """CAL_A is solved at T0 and T0 + 600 s and CAL_B at the same times; with max_gap = 100 s the derived rows
    between them take the time interpolation of the table across both fields, not the 1-D combination."""
    source = _fringe_table(tmp_path / "mbd", three_field_ms, phase={0: 0.2, 1: 1.2}, delay={0: 1.0, 1: 3.0})
    times = np.array([T0, T0 + 300.0, T0 + 600.0])
    summary = phaseref.interpolated_table((source, "CAL_A"), (source, "CAL_B"), 0.25, "TARGET", times,
                                          tmp_path / "derived", max_gap=100.0)
    assert summary["time_interpolated"] == pytest.approx(1 / 3)
    table = load_caltable(tmp_path / "derived")
    rows = table.antenna1 == 1
    delay = np.real(table.param[rows, 0, 1])[np.argsort(table.time[rows])]
    assert delay[0] == pytest.approx(1.5) and delay[2] == pytest.approx(1.5)   # at a solution: interpolated
    assert delay[1] not in (pytest.approx(1.5),)                               # midway: the time interpolation
    assert 1.0 <= delay[1] <= 3.0
