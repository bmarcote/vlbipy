"""Tests for drawing plot sets in worker processes (vlbipy.plot_jobs) and the pool they run in."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlbipy import VLBIObs, plot_jobs
from vlbipy.backends.casa import CasaPlotOps, largest_read_rows
from vlbipy.logging_utils import _FILE_SINKS, _logger, add_file_log, logging_setup, warnings
from vlbipy.plot_jobs import PlotJob, job_cost, plan_workers, run_plot_jobs
from vlbipy.solvers import workers


def make_observation():
    """A dummy observation with a target, a phase calibrator and a fringe finder, already imported."""
    obs = VLBIObs("pj01", network="EVN", backend="dummy", target="3C395", phasecal="J1848+3219",
                  fringe_finder="3C345")
    obs.import_data()
    return obs["pj01"]


def rebuildable(backend, monkeypatch, backend_class: str = "vlbipy.backends.dummy:DummyBackend") -> None:
    """Let the dummy backend be rebuilt in worker processes (its plot methods keep no state)."""
    monkeypatch.setattr(backend, "worker_spec", lambda code: {"backend": backend_class,
                                                              "work_dir": str(backend.work_dir)})


def record_local_draws(monkeypatch) -> list:
    """Record the name of every job drawn in this process."""
    drawn, original = [], plot_jobs._draw_here

    def draw_here(backend, project_code, job, label):
        drawn.append(job.name)
        return original(backend, project_code, job, label)

    monkeypatch.setattr(plot_jobs, "_draw_here", draw_here)
    return drawn


class FakePlots:
    """Plot methods returning a path, or raising."""

    def good(self, project_code, *, name):
        return f"{project_code}.{name}.png"

    def several(self, project_code, *, name):
        return [f"{project_code}.{name}.amp.png", f"{project_code}.{name}.phase.png"]

    def bad(self, project_code, **kwargs):
        raise RuntimeError("no data on this source")


class FakeBackend:
    """A backend that cannot be rebuilt in another process."""

    plot = FakePlots()

    def worker_spec(self, project_code):
        return None


@pytest.fixture(autouse=True)
def stop_pool():
    """Each test that starts the shared pool stops it again."""
    yield
    workers.close_pool()


# -- the plot sets --

def test_final_data_is_drawn_in_worker_processes_with_the_serial_result(monkeypatch):
    observation = make_observation()
    serial = observation.plot.final_data()          # the dummy backend cannot be rebuilt elsewhere: drawn here
    rebuildable(observation._backend, monkeypatch)
    drawn_here = record_local_draws(monkeypatch)
    assert observation.plot.final_data() == serial  # same paths, same order
    assert drawn_here == []                         # every final-data plot went to a worker


def test_raw_diagnostics_keep_the_observation_jobs_here(monkeypatch):
    observation = make_observation()
    serial = observation.plot.diagnostics(column="data", force=True)
    rebuildable(observation._backend, monkeypatch)
    drawn_here = record_local_draws(monkeypatch)
    assert observation.plot.diagnostics(column="data", force=True) == serial
    # The survey and the time evolution (with its figures) need the observation: they run here, the rest in workers.
    assert drawn_here[:2] == ["scan_snr", "time_evolution"]
    assert all(name.startswith("time_evolution[") for name in drawn_here[2:])


def test_one_worker_draws_everything_here(monkeypatch):
    observation = make_observation()
    observation.config["plotting"]["workers"] = 1
    rebuildable(observation._backend, monkeypatch)
    drawn_here = record_local_draws(monkeypatch)
    assert observation.plot.final_data()
    names = observation.plot._all_source_names()
    # uv coverage, then a radplot, spectrum and timeseries per source, then the light curve: all drawn here.
    assert len(drawn_here) == 2 + 3 * len(names) and drawn_here[0] == "uv_coverage" and drawn_here[-1] == "lightcurve"


# -- failures --

def test_a_failing_plot_warns_and_the_set_carries_on():
    warnings.reset()
    jobs = [PlotJob("first", method="good", kwargs={"name": "a"}), PlotJob("broken", method="bad"),
            PlotJob("last", method="several", kwargs={"name": "b"}), PlotJob("local", call=lambda: "here.png")]
    paths = run_plot_jobs(FakeBackend(), "tp01", jobs, max_workers=4, memory_fraction=0.8, label="raw")
    assert paths == ["tp01.a.png", "tp01.b.amp.png", "tp01.b.phase.png", "here.png"]
    assert warnings.summary() == ["tp01: raw broken plot failed (no data on this source)"]


def test_a_plot_failing_in_a_worker_warns_and_the_set_carries_on(monkeypatch):
    observation = make_observation()
    backend = observation._backend
    rebuildable(backend, monkeypatch)
    warnings.reset()
    jobs = [PlotJob("missing", method="no_such_plot"),
            PlotJob("radplot", method="radplot", kwargs={"field": "3C345", "label": "final"})]
    paths = run_plot_jobs(backend, "pj01", jobs, max_workers=2, memory_fraction=0.8, label="final")
    assert paths == [backend.plot.radplot("pj01", field="3C345", label="final")]
    assert len(warnings.summary()) == 1 and "final missing plot failed" in warnings.summary()[0]


def test_jobs_a_worker_cannot_run_are_drawn_here(monkeypatch):
    observation = make_observation()
    backend = observation._backend
    rebuildable(backend, monkeypatch, backend_class="vlbipy.backends.dummy:NoSuchBackend")
    drawn_here = record_local_draws(monkeypatch)
    jobs = [PlotJob(f"radplot[{name}]", method="radplot", kwargs={"field": name}) for name in ("3C345", "3C395")]
    paths = run_plot_jobs(backend, "pj01", jobs, max_workers=2, memory_fraction=0.8, label="final")
    assert paths == [backend.plot.radplot("pj01", field=name) for name in ("3C345", "3C395")]
    assert drawn_here == ["radplot[3C345]", "radplot[3C395]"]


# -- how many at a time --

def test_plan_workers_takes_as_many_as_fit_largest_first():
    gb = 1_000_000_000
    peaks = [4 * gb, 1 * gb, 3 * gb, 1 * gb, 1 * gb]
    assert plan_workers(peaks, max_workers=8, available=None) == 5          # memory unknown: CPU/config only
    assert plan_workers(peaks, max_workers=3, available=None) == 3
    assert plan_workers(peaks, max_workers=8, available=7 * gb) == 2        # 4 + 3 fit, 4 + 3 + 1 do not
    assert plan_workers(peaks, max_workers=8, available=10 * gb) == 5
    assert plan_workers(peaks, max_workers=8, available=2 * gb) == 1        # never fewer than one
    assert plan_workers([None, None], max_workers=8, available=gb) == 2      # unknown peaks count as nothing
    assert plan_workers([], max_workers=8, available=gb) == 1


def test_one_plot_at_a_time_when_memory_allows_no_more(monkeypatch):
    observation = make_observation()
    backend = observation._backend
    rebuildable(backend, monkeypatch)
    monkeypatch.setattr(backend.plot, "estimate_memory", lambda code, method, kwargs: 10**12, raising=False)
    drawn_here = record_local_draws(monkeypatch)
    assert observation.plot.final_data()
    assert len(drawn_here) == 2 + 3 * len(observation.plot._all_source_names())   # all drawn here, in turn


def test_largest_read_rows_follows_the_read_pattern():
    observation = make_observation()
    metadata = observation.metadata
    scans = sorted(metadata.scans_for_source("J1848+3219"), key=lambda s: s.scan_number)

    def rows(scan, per_integration):
        return max(1, round((scan.time_end - scan.time_start) / scan.integration_time)) * per_integration

    n = [len(scan.antennas) for scan in scans]
    refant = sum(rows(scan, k - 1) for scan, k in zip(scans, n))
    assert largest_read_rows(metadata, fields=["J1848+3219"], baselines="refant") == refant
    assert largest_read_rows(metadata, fields=["J1848+3219"], baselines="all", scans_per_read=1) == \
        max(rows(scan, k * (k + 1) // 2) for scan, k in zip(scans, n))
    assert largest_read_rows(metadata, fields=["J1848+3219"], scans=[scans[0].scan_number], baselines="auto") == \
        rows(scans[0], n[0])
    assert largest_read_rows(metadata, fields=["nothing"]) == 0


def test_casa_memory_estimate_is_base_plus_largest_read():
    observation = make_observation()
    metadata = observation.metadata
    plot_ops = CasaPlotOps(backend=None)
    per_row = metadata.freq_setup.n_channels * len(metadata.freq_setup.polarizations)
    spectrum = plot_ops.estimate_memory("pj01", "spectrum", {"field": "3C395", "metadata": metadata})
    rows = largest_read_rows(metadata, fields=["3C395"], baselines="refant")
    fixed = CasaPlotOps.WORKER_BASE_BYTES + CasaPlotOps.PLOT_OVERHEAD_BYTES
    assert spectrum == fixed + rows * per_row * CasaPlotOps.READ_BYTES_PER_VISIBILITY
    plain = plot_ops.estimate_memory("pj01", "radplot", {"field": "3C395", "metadata": metadata})
    with_model = plot_ops.estimate_memory("pj01", "radplot", {"field": "3C395", "metadata": metadata,
                                                                 "with_model": True})
    assert with_model - fixed == 2 * (plain - fixed)
    assert plot_ops.estimate_memory("pj01", "time_evolution", {"metadata": metadata}) is None
    assert plot_ops.estimate_memory("pj01", "spectrum", {"field": "3C395"}) is None


# -- corner plots --

def test_casa_corners_read_the_data_once(tmp_path):
    reads = []
    spectra = {"antennas": ["EF", "JB", "WB"], "times": [0.0, 60.0], "frequencies_ghz": [4.9, 5.0],
               "baselines": {("EF", "JB"): np.full((2, 2), 1 + 1j), ("EF", "WB"): np.full((2, 2), 2 - 1j),
                             ("JB", "WB"): np.full((2, 2), 1j)}, "field": "3C345", "column": "corrected",
               "stokes_i": True}

    def read_dynamic_spectra(project_code, **kwargs):
        reads.append(kwargs["field"])
        return spectra

    backend = SimpleNamespace(work_dir=tmp_path, kind="casa",
                              data=SimpleNamespace(read_dynamic_spectra=read_dynamic_spectra))
    plot_ops = CasaPlotOps(backend)
    paths = plot_ops.baseline_corners("cp01", field="3C345", label="calibrated_3C345")
    assert reads == ["3C345"]                                             # one read for both grids
    assert [Path(p).name for p in paths] == ["cp01.calibrated_3C345.corner.phase.png",
                                             "cp01.calibrated_3C345.corner.amp.png"]
    assert all(Path(p).is_file() for p in paths)
    assert plot_ops.baseline_corner("cp01", field="3C345", quantity="amp") == str(tmp_path / "plots" / "calibrated"
                                                                                 / "cp01.corner.amp.png")


def test_diagnostics_make_one_corners_job_per_calibrator(monkeypatch):
    observation = make_observation()
    rebuildable(observation._backend, monkeypatch)
    submitted = []
    original = plot_jobs._submit

    def submit(backend, project_code, jobs, *args):
        submitted.extend(job.name for job in jobs)
        return original(backend, project_code, jobs, *args)

    monkeypatch.setattr(plot_jobs, "_submit", submit)
    paths = observation.plot.diagnostics(column="corrected")
    calibrators = [s.name for s in observation.sources.calibrators]
    assert [name for name in submitted if name.startswith("corner")] == [f"corners[{name}]" for name in calibrators]
    for name in calibrators:
        assert sum(f".calibrated_{name}.corner." in p for p in paths) == 2   # phase and amplitude


# -- ordering and logging --

def test_job_cost_is_the_observing_time_read():
    observation = make_observation()
    metadata = observation.metadata

    def seconds(source):
        return sum(s.time_end - s.time_start for s in metadata.scans if s.source == source)

    every_scan = sum(s.time_end - s.time_start for s in metadata.scans)
    first = metadata.scans_for_source("3C345")[0]
    assert job_cost(PlotJob("t", method="x", kwargs={"field": "3C395", "metadata": metadata})) == seconds("3C395")
    assert job_cost(PlotJob("f", method="x", kwargs={"field": "3C395,3C345", "metadata": metadata})) == \
        seconds("3C395") + seconds("3C345")
    assert job_cost(PlotJob("s", method="x", kwargs={"field": "3C345", "scans": [first.scan_number],
                                                     "metadata": metadata})) == first.time_end - first.time_start
    assert job_cost(PlotJob("a", method="x", kwargs={"metadata": metadata})) == every_scan
    assert job_cost(PlotJob("n", method="x", kwargs={"field": "3C395"})) == 0.0


def test_workers_log_to_the_run_log(tmp_path, monkeypatch):
    observation = make_observation()
    rebuildable(observation._backend, monkeypatch)
    log_file = add_file_log(tmp_path, "pj01")
    try:
        assert str(log_file) in logging_setup()["files"]
        observation.plot.final_data()
        text = log_file.read_text()
        assert "[dummy] plot[lightcurve]" in text                       # written by a worker process
        assert "plot set final:" in text and "in worker processes" in text
    finally:
        _logger.remove(_FILE_SINKS.pop(str(tmp_path.resolve()))["id"])


# -- the pool --

def test_pool_callers_beyond_its_size_wait_for_a_worker():
    pool = workers.WorkerPool(1)
    try:
        results = [None] * 3

        def call(index):
            results[index] = pool.run("builtins:dict", {"index": index})

        threads = [threading.Thread(target=call, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert results == [{"index": index} for index in range(3)]
    finally:
        pool.close()


def test_submit_jobs_returns_one_future_per_job_and_isolates_failures():
    futures = workers.submit_jobs("json:loads", [{"s": "{"}, {"s": "[1]"}, {"s": "2"}], workers=2)
    with pytest.raises(workers.WorkerError, match="JSONDecodeError"):
        futures[0].result(timeout=60)
    assert [future.result(timeout=60) for future in futures[1:]] == [[1], 2]
    assert workers.submit_jobs("json:loads", [], workers=2) == []
