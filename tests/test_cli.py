"""Tests for CLI argument parsing (positional step/kind selectors)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vlbipy.cli import _cmd_plot, _relocate_selectors, build_parser, main


def parse(argv: list[str]):
    return _relocate_selectors(build_parser().parse_args(argv))


def test_calibrate_steps_positional_before_flags():
    args = parse(["calibrate", "bandpass", "fringefit", "-p", "RSM07"])
    assert args.steps == ["bandpass", "fringefit"]
    assert args.project == ["RSM07"]


def test_calibrate_steps_after_project_are_relocated():
    # -p is greedy (nargs='+'): trailing valid step names are moved back to steps.
    args = parse(["calibrate", "-p", "RSM07", "bandpass", "fringefit"])
    assert args.steps == ["bandpass", "fringefit"]
    assert args.project == ["RSM07"]


def test_calibrate_no_steps_runs_full_chain():
    args = parse(["calibrate", "-p", "RSM07"])
    assert args.steps == []


def test_calibrate_invalid_step_rejected():
    with pytest.raises(SystemExit):
        parse(["calibrate", "bandpss", "-p", "RSM07"])


def test_flag_steps_positional():
    args = parse(["flag", "autocorr", "edges", "-p", "RSM07"])
    assert args.steps == ["autocorr", "edges"]


def test_plot_kinds_positional_and_relocated():
    assert parse(["plot", "caltables", "-p", "RSM07"]).kinds == ["caltables"]
    assert parse(["plot", "-p", "RSM07", "caltables", "uv_coverage"]).kinds == ["caltables", "uv_coverage"]


def test_multi_project_codes_survive_relocation():
    args = parse(["calibrate", "-p", "RSM07", "RSM08", "bandpass"])
    assert args.project == ["RSM07", "RSM08"]
    assert args.steps == ["bandpass"]


def test_steps_after_target_are_relocated():
    args = parse(["calibrate", "-p", "RSM07", "-t", "3C286", "bandpass"])
    assert args.target == ["3C286"]
    assert args.steps == ["bandpass"]


def test_callib_flag_is_store_true_default_none():
    """--callib parses as True; absent it is None so defaults take effect."""
    assert parse(["calibrate", "-p", "RSM07"]).callib is None
    assert parse(["calibrate", "-p", "RSM07", "--callib"]).callib is True


def test_applycal_parser_accepts_parallel_gainfields():
    args = parse(["applycal", "data.ms", "one.G", "two.K", "--gainfield", "A", "B"])
    assert args.caltables == ["one.G", "two.K"]
    assert args.gainfield == ["A", "B"]


def test_applycal_broadcasts_values_and_builds_tables(tmp_path, monkeypatch):
    ms = tmp_path / "data.ms"
    first = tmp_path / "one.G"
    second = tmp_path / "two.K"
    ms.mkdir()
    first.mkdir()
    second.mkdir()
    backend = SimpleNamespace(data=MagicMock(), calibrate=MagicMock())
    monkeypatch.setattr("vlbipy.cli.get_backend", MagicMock(return_value=backend))
    assert main(["applycal", str(ms), str(first), str(second), "--gainfield", "A",
                 "--interp", "nearest", "--spwmap", "0,0", "--calwt", "false"]) == 0
    tables = backend.calibrate.apply.call_args.args[2]
    assert [table.gainfield for table in tables] == ["A", "A"]
    assert [table.interp for table in tables] == ["nearest", "nearest"]
    assert [table.spwmap for table in tables] == [[0, 0], [0, 0]]
    assert [table.calwt for table in tables] == [False, False]


def test_applycal_rejects_parallel_length_mismatch(tmp_path):
    ms = tmp_path / "data.ms"
    tables = [tmp_path / name for name in ("one.G", "two.G")]
    ms.mkdir()
    for table in tables:
        table.mkdir()
    with pytest.raises(SystemExit):
        main(["applycal", str(ms), *(str(table) for table in tables), "--interp", "a", "b", "c"])


def test_applycal_rejects_callib_with_tables(tmp_path):
    ms = tmp_path / "data.ms"
    table = tmp_path / "one.G"
    callib = tmp_path / "apply.txt"
    ms.mkdir()
    table.mkdir()
    callib.write_text("")
    with pytest.raises(SystemExit):
        main(["applycal", str(ms), str(table), "--callib", str(callib)])


def test_applycal_rejects_missing_paths(tmp_path):
    with pytest.raises(SystemExit):
        main(["applycal", str(tmp_path / "missing.ms"), str(tmp_path / "missing.G")])


def test_plot_ms_dispatches_to_casa_plot_ops(tmp_path, monkeypatch):
    ms = tmp_path / "data.ms"
    ms.mkdir()
    backend = SimpleNamespace(data=MagicMock(), plot=MagicMock())
    backend.plot.spectrum.return_value = ["spectrum.png"]
    backend.plot.diagnostic.return_value = "uv.png"
    monkeypatch.setattr("vlbipy.cli.get_backend", MagicMock(return_value=backend))
    assert main(["plot", "--ms", str(ms), "spectrum", "uv_coverage", "--field", "target",
                 "--scans", "1,2", "3", "--refant", "EF"]) == 0
    backend.plot.spectrum.assert_called_once()
    assert backend.plot.spectrum.call_args.kwargs["scans"] == [1, 2, 3]
    backend.plot.diagnostic.assert_called_once()


def test_existing_project_plot_path_is_unchanged():
    obs = MagicMock()
    obs.plot.caltables.return_value = "plot.png"
    args = SimpleNamespace(ms=None, kinds=["caltables"])
    assert _cmd_plot(obs, args) == 0
    obs.import_data.assert_called_once_with(force=False)
    obs.plot.caltables.assert_called_once_with()


def test_check_source_option_is_parsed():
    args = parse(["pipeline", "-p", "RSM07", "-t", "3C286", "--check-source", "J1048+7143"])
    assert args.check_source == ["J1048+7143"]
