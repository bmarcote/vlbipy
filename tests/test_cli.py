"""Tests for CLI argument parsing (positional step/kind selectors)."""
from __future__ import annotations

import pytest

from vlbipy.cli import _relocate_selectors, build_parser


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
