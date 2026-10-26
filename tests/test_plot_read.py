"""Regression tests for plot readers without measurement sets or worker processes."""
import builtins
import dis
from types import CodeType, FunctionType

import numpy as np
import pytest

from vlbipy.solvers import msio, plot_read


def test_plot_reader_globals_are_defined():
    """Check global references in every reader, worker and nested comprehension."""
    available = set(vars(plot_read)) | set(vars(builtins))
    pending = [value.__code__ for value in vars(plot_read).values()
               if isinstance(value, FunctionType) and value.__module__ == plot_read.__name__]
    while pending:
        code = pending.pop()
        missing = {op.argval for op in dis.get_instructions(code)
                   if op.opname == "LOAD_GLOBAL" and op.argval not in available}
        assert not missing, (code.co_name, missing)
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))


def test_uv_coverage_job_filters_rows(monkeypatch):
    """Run the actual worker, retaining partial flags but rejecting autos and fully flagged rows."""
    assert plot_read.read_runs is msio.read_runs
    specs = [{"ddid": 0}]
    flags = np.zeros((6, 2, 2), dtype=bool)
    flags[1, 0, 0] = True
    flags[3] = True
    block = {"UVW": np.arange(18).reshape(6, 3), "FLAG": flags,
             "FLAG_ROW": np.array([False, False, False, False, True, False]),
             "ANTENNA1": np.array([0, 0, 1, 0, 0, 0]), "ANTENNA2": np.array([1, 1, 1, 1, 1, 2]),
             "FIELD_ID": np.array([2, 2, 2, 2, 2, 3])}

    def read_runs(actual_specs, columns, *, engine):
        """Supply synthetic rows at the raw table boundary."""
        assert actual_specs is specs
        assert columns == ["UVW", "FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2"]
        assert engine == "stub"
        return block

    monkeypatch.setattr(plot_read, "read_runs", read_runs)
    result = plot_read.uv_coverage_job(specs=specs, engine="stub", ncorr=2)
    np.testing.assert_array_equal(result["field"], [2, 2, 3])
    np.testing.assert_array_equal(result["u"], [0, 3, 15])
    np.testing.assert_array_equal(result["v"], [1, 4, 16])
    assert result["u"].dtype == np.float64


@pytest.mark.parametrize("empty", [True, False])
def test_uv_coverage_job_without_usable_rows(monkeypatch, empty):
    """Handle both no table rows and a block containing only autocorrelations."""
    block = {} if empty else {"UVW": np.zeros((1, 3)), "FLAG": np.zeros((1, 2, 2), bool),
                              "FLAG_ROW": [False], "ANTENNA1": [0], "ANTENNA2": [0], "FIELD_ID": [0]}
    monkeypatch.setattr(plot_read, "read_runs", lambda *args, **kwargs: block)
    result = plot_read.uv_coverage_job(specs=[], engine="stub", ncorr=2)
    if empty:
        assert result == {}
    else:
        assert all(values.size == 0 for values in result.values())


def test_read_uv_coverage_dispatches_real_job_and_samples_per_field(monkeypatch):
    """Exercise selection, worker dispatch and per-field sampling entirely in memory."""
    specs = [{"ddid": 0}]
    setup = {"field_names": ["target", "calibrator"], "ncorr": 2}
    layout = {"stub": True}
    monkeypatch.setattr(plot_read, "table_engine", lambda: "stub")
    monkeypatch.setattr(plot_read, "read_setup", lambda *args, **kwargs: setup)
    monkeypatch.setattr(plot_read, "read_layout", lambda *args, **kwargs: layout)

    def select_runs(actual_layout, *, fields, ddids):
        """Verify source and first-data-description selection before dispatch."""
        assert actual_layout is layout
        assert fields == {0, 1}
        assert ddids == {0}
        return [0]

    def run_jobs(function, jobs, workers):
        """Execute the production worker rather than replacing its output."""
        assert function == "vlbipy.solvers.plot_read:uv_coverage_job"
        assert workers == 1
        return [plot_read.uv_coverage_job(**job) for job in jobs]

    monkeypatch.setattr(plot_read, "select_runs", select_runs)
    monkeypatch.setattr(plot_read, "run_specs", lambda *args: specs)
    monkeypatch.setattr(plot_read, "split_rows", lambda *args: [specs])
    monkeypatch.setattr(plot_read, "run_jobs", run_jobs)
    monkeypatch.setattr(plot_read, "read_runs", lambda *args, **kwargs: {
        "UVW": np.arange(21).reshape(7, 3), "FLAG": np.zeros((7, 2, 2), bool),
        "FLAG_ROW": np.zeros(7, bool), "ANTENNA1": np.zeros(7, int), "ANTENNA2": np.ones(7, int),
        "FIELD_ID": np.array([0, 0, 0, 0, 0, 1, 1])})
    result = plot_read.read_uv_coverage("unused.ms", fields=["target", "calibrator"], max_points=2, workers=1)
    assert set(result["fields"]) == {0, 1}
    np.testing.assert_array_equal(result["fields"][0], [[0, 9], [1, 10]])
    np.testing.assert_array_equal(result["fields"][1], [[15, 18], [16, 19]])
    assert result["seconds"] >= 0
