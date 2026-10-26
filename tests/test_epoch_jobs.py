import pickle
import pytest

from vlbipy import VLBIObs
from vlbipy import epoch_jobs


def test_full_cpu_environment_overrides_inherited_limits(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setattr(epoch_jobs, "usable_cpus", lambda: 10)
    env = epoch_jobs.full_cpu_environment()
    for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VLBIPY_FFT_WORKERS"):
        assert env[key] == "10"


def test_batches_are_limited_by_memory_not_thread_sharing(monkeypatch):
    monkeypatch.setattr(epoch_jobs, "available_memory", lambda: 1000)
    assert epoch_jobs.batch_size([300, 300, 300], 8, 0.6) == 2
    assert epoch_jobs.batch_size([700, 300], 8, 0.6) == 1
    assert epoch_jobs.batch_size([100, 100], 1, 0.6) == 1
    monkeypatch.setattr(epoch_jobs, "available_memory", lambda: None)
    assert epoch_jobs.batch_size([100, 100], 8, 0.6) == 1
    with pytest.raises(ValueError):
        epoch_jobs.batch_size([100], 8, 0)


def test_epoch_spec_preserves_backend_reconstruction(tmp_path, monkeypatch):
    """Finishing workers retain backend-specific configuration, not only its kind."""
    run = VLBIObs("A", backend="dummy", work_dir=tmp_path)
    obs = run.observations[0]
    backend = {"backend": "custom:Backend", "imager": "difmap", "setting": 42}
    monkeypatch.setattr(obs._backend, "worker_spec", lambda code: backend)
    assert epoch_jobs.epoch_spec(obs, True)["backend"] == backend


@pytest.mark.parametrize("fail", [False, True])
def test_epoch_queue_preserves_state_and_finishes_other_epochs(tmp_path, monkeypatch, fail):
    run = VLBIObs(["A", "B", "C"], backend="dummy", work_dir=tmp_path)
    events, environments = [], []
    monkeypatch.setattr(epoch_jobs, "usable_cpus", lambda: 10)
    monkeypatch.setattr(epoch_jobs, "available_memory", lambda: 1000)
    monkeypatch.setattr(epoch_jobs, "epoch_memory", lambda obs: 300)

    class Process:
        def __init__(self, command, *, env):
            self.spec = pickle.loads(open(command[-2], "rb").read())
            self.path = command[-1]
            self.code = self.spec["project_code"]
            self.done = False
            environments.append(env)
            events.append(("start", self.code))

        def wait(self, timeout=None):
            payload = {"state": self.spec["state"], "steps": {"new_step": {"status": "done"}},
                       "images": {"target": self.code}, "warnings": []}
            payload["state"]["flag_statistics"] = {"epoch": self.code}
            if fail and self.code == "A":
                payload["error"] = "test failure"
            with open(self.path, "wb") as handle:
                pickle.dump(payload, handle)
            events.append(("finish", self.code))
            self.done = True
            return 0

        def poll(self):
            return 0 if self.done else None

    monkeypatch.setattr(epoch_jobs.subprocess, "Popen", Process)
    if fail:
        with pytest.raises(RuntimeError, match="A"):
            epoch_jobs.run_epochs(run.observations)
    else:
        assert epoch_jobs.run_epochs(run.observations) == {code: {"target": code} for code in "ABC"}
    assert events == [("start", "A"), ("start", "B"), ("finish", "A"), ("finish", "B"),
                      ("start", "C"), ("finish", "C")]
    assert all(env["RAYON_NUM_THREADS"] == "10" for env in environments)
    assert all(env["VLBIPY_EPOCH_MEMORY_BYTES"] == "300" for env in environments)
    for obs in run.observations:
        assert obs.flag_statistics == {"epoch": obs.project_code}
        assert obs._state.status("new_step") == "done"
    assert run.observations[0]._state.status("epoch_products") == ("failed" if fail else "done")


def test_epoch_queue_in_fresh_processes(tmp_path):
    run = VLBIObs(["A", "B"], backend="dummy", target="TARGET", work_dir=tmp_path,
                  imaging={"epoch_workers": 1})
    run.import_data()
    result = epoch_jobs.run_epochs(run.observations)
    assert list(result) == ["A", "B"]
    for obs in run.observations:
        assert obs._state.status("epoch_products") == "done"
        assert obs.flag_statistics


def test_epoch_steps_keep_calibration_dependencies(tmp_path, monkeypatch):
    run = VLBIObs("A", backend="dummy", work_dir=tmp_path)
    obs, events = run.observations[0], []
    monkeypatch.setattr(obs.export, "per_source", lambda **kw: events.append("split"))
    monkeypatch.setattr(obs.flag, "statistics", lambda: events.append("flags"))
    monkeypatch.setattr(run, "_selfcal_all", lambda *a, **kw: events.append("selfcal"))
    monkeypatch.setattr(run, "_calibrated_diagnostics", lambda *a: events.append("diagnostics"))
    monkeypatch.setattr(run, "_image_all", lambda *a, **kw: events.append("images") or {"target": "image"})
    monkeypatch.setattr(run, "_final_plots", lambda *a: events.append("plots"))
    assert run._finish_epoch(obs) == {"target": "image"}
    assert events == ["split", "flags", "selfcal", "diagnostics", "images", "plots"]


@pytest.mark.parametrize("failure", ["spec", "launch", "corrupt", "exit", "missing_images"])
def test_queue_failures_preserve_siblings_and_reload_disk(tmp_path, monkeypatch, failure):
    """An epoch boundary failure cannot discard siblings or the remaining queue."""
    run = VLBIObs(["A", "B", "C"], backend="dummy", work_dir=tmp_path)
    first = run.observations[0]
    events, recovered = [], []
    monkeypatch.setattr(epoch_jobs, "available_memory", lambda: 1000)
    monkeypatch.setattr(epoch_jobs, "epoch_memory", lambda obs: 300)
    monkeypatch.setattr(epoch_jobs, "usable_cpus", lambda: 8)
    original_spec = epoch_jobs.epoch_spec

    def specification(obs, force):
        if obs is first and failure == "spec":
            raise ValueError("bad spec")
        return original_spec(obs, force)

    monkeypatch.setattr(epoch_jobs, "epoch_spec", specification)
    monkeypatch.setattr(first._state, "load", lambda: recovered.append("state"))
    monkeypatch.setattr(first, "_load_gaintables", lambda: recovered.append("tables") or ["disk-table"])
    monkeypatch.setattr(first, "_load_cal_selection", lambda: recovered.append("selection") or {"disk": True})

    class Process:
        def __init__(self, command, *, env):
            self.spec = pickle.loads(open(command[-2], "rb").read())
            self.code = self.spec["project_code"]
            self.path, self.done = command[-1], False
            if self.code == "A" and failure == "launch":
                raise OSError("cannot launch")
            events.append(self.code)

        def wait(self, timeout=None):
            self.done = True
            if self.code == "A" and failure == "exit":
                return 7
            payload = {"state": self.spec["state"], "steps": {}, "images": {}, "warnings": []}
            if self.code == "A" and failure == "missing_images":
                del payload["images"]
            with open(self.path, "wb") as handle:
                if self.code == "A" and failure == "corrupt":
                    handle.write(b"not a pickle")
                else:
                    pickle.dump(payload, handle)
            return 0

        def poll(self):
            return 0 if self.done else None

    monkeypatch.setattr(epoch_jobs.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="failed for A"):
        epoch_jobs.run_epochs(run.observations)
    assert events[-2:] == ["B", "C"]
    assert first._state.status("epoch_products") == "failed"
    assert all(obs._state.status("epoch_products") == "done" for obs in run.observations[1:])
    if failure in ("corrupt", "exit", "missing_images"):
        assert recovered == ["state", "tables", "selection"]
        assert first.gaintables == ["disk-table"]
        assert first._cal_selection == {"disk": True}


@pytest.mark.parametrize("explicit,configured,real,queued", [(None, None, True, True),
    (None, "difmap", True, True), ("difmap", "casa", True, True),
    (None, "casa", True, False), (None, "difmap", False, False)])
def test_campaign_queue_gate_uses_config_for_casa(tmp_path, monkeypatch, explicit, configured, real, queued):
    """Real CASA backends use their configured imager when no explicit imager exists."""
    from types import SimpleNamespace
    run = VLBIObs(["A", "B"], backend="dummy", work_dir=tmp_path)
    for obs in run.observations:
        obs._backend = SimpleNamespace(requires_data_files=real)
        if explicit is not None:
            obs._backend.imager = explicit
        if configured is None:
            obs.config["imaging"].pop("imager", None)
        else:
            obs.config["imaging"]["imager"] = configured
    events = []
    monkeypatch.setattr(epoch_jobs, "run_epochs", lambda *a, **kw: events.append("queue") or {})
    monkeypatch.setattr(run, "_finish_epoch", lambda obs, **kw: events.append(obs.project_code) or {})
    monkeypatch.setattr(run, "combine", lambda **kw: {})
    run._finish_campaign()
    assert events == (["queue"] if queued else ["A", "B"])

