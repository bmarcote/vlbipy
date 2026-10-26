"""Memory-bounded epoch batches in fresh processes, each with the full CPU allowance."""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from .logging_utils import attach_logging, get_logger, headline, logging_setup, warnings
from .resources import available_memory, usable_cpus

logger = get_logger()
EPOCH_BYTES_PER_PIXEL = 96
EPOCH_RUNTIME_BYTES = 2 * 1024 ** 3
SEARCH_BYTES_PER_PIXEL = 24
_STATE_FIELDS = ("_metadata", "gaintables", "sources", "_cal_selection", "_cal_stages", "_snr_surveys",
                 "flag_statistics", "_selfcal_results", "_scratch", "_band_of")


def full_cpu_environment() -> dict:
    """Give a fresh epoch all usable CPUs, overriding inherited single-thread worker settings."""
    env = dict(os.environ)
    threads = str(usable_cpus())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
                 "VLBIPY_FFT_WORKERS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = threads
    return env


def epoch_memory(obs) -> int:
    """Conservative peak for maps, checkpoints, visibility data and the isolated CASA runtime."""
    from .backends.difmap import IMAGE_PIXELS, SEARCH_MAX_PIXELS
    cfg = obs.config.get("imaging", {})
    pixels = int(cfg.get("mapsize") or IMAGE_PIXELS)
    maps = EPOCH_BYTES_PER_PIXEL * pixels ** 2
    search = SEARCH_BYTES_PER_PIXEL * SEARCH_MAX_PIXELS ** 2 if cfg.get("search_fov", 1000.0) > 0 else 0
    return max(maps, search) + EPOCH_RUNTIME_BYTES


def epoch_spec(obs, force: bool) -> dict:
    """Capture live calibration/selection state as well as disk paths for one independent epoch."""
    return {"project_code": obs.project_code, "config": obs.config, "work_dir": str(Path(obs.work_dir).resolve()),
            "state": {name: getattr(obs, name) for name in _STATE_FIELDS}, "steps": obs._state._steps,
            "backend": obs._backend.worker_spec(obs.project_code), "force": force, "log": logging_setup()}


def finish_epoch(spec: dict) -> dict:
    """Rebuild an isolated observation and return products and all state changed by its finishing steps."""
    from .config import load_config
    from .observation import Observation
    from .vlbiobs import VLBIObs

    from .solvers.workers import resolve

    attach_logging(spec["log"])
    config = load_config(spec["config"])
    backend = spec.get("backend")
    backend = resolve(backend["backend"]).from_worker_spec(backend) if backend else None
    obs = Observation(spec["project_code"], config, backend=backend, work_dir=spec["work_dir"])
    for name, value in spec["state"].items():
        setattr(obs, name, value)
    obs._state._steps = spec["steps"]
    runner = object.__new__(VLBIObs)
    runner.config, runner._observations, runner._merged = obs.config, [obs], None
    payload = {}
    try:
        headline("[{}] epoch products: {} CPU threads", obs.project_code, usable_cpus())
        payload["images"] = runner._finish_epoch(obs, force=spec["force"])
    except Exception:
        payload["error"] = traceback.format_exc()
    finally:
        obs.persist_run_warnings()
    payload.update(state={name: getattr(obs, name) for name in _STATE_FIELDS}, steps=obs._state._steps,
                   warnings=warnings.summary())
    return payload


def batch_size(peaks: list[int], max_workers: int, memory_fraction: float) -> int:
    """Admit only epochs whose combined estimated peaks fit the current available-memory budget."""
    from .plot_jobs import plan_workers
    if not 0 < memory_fraction <= 1:
        raise ValueError("imaging.epoch_memory_fraction must be in (0, 1]")
    free = available_memory()
    if free is None:
        return 1
    return plan_workers(peaks, max_workers=max_workers, available=int(free * memory_fraction))


def run_epochs(observations: list, *, force: bool = False) -> dict:
    """Finish queued epochs in memory-bounded batches; do not divide their numerical thread budgets.

    Every process owns a different MS and retains the serial source/calibration dependencies within
    that epoch. Failed epochs retain their partial state; other queued epochs still finish before the
    failure is raised. The parent regains the updated chain, metadata and products before combination.
    """
    if not observations:
        return {}
    cfg = observations[0].config.get("imaging", {})
    setting = cfg.get("epoch_workers", "auto")
    limit = usable_cpus() if str(setting).lower() in ("auto", "0", "") else max(1, int(setting))
    fraction = float(cfg.get("epoch_memory_fraction", 0.6))
    pending = sorted(observations, key=epoch_memory, reverse=True)
    images, failures = {}, []
    with tempfile.TemporaryDirectory(prefix="vlbipy_epochs_") as scratch:
        while pending:
            count = batch_size([epoch_memory(obs) for obs in pending], min(limit, len(pending)), fraction)
            batch, pending = pending[:count], pending[count:]
            headline("epoch queue: {} running, {} queued; {} CPU threads per epoch, {:.1f} GB estimated each",
                     len(batch), len(pending), usable_cpus(), max(epoch_memory(o) for o in batch) / 1e9)
            running = []
            try:
                for index, obs in enumerate(batch):
                    try:
                        spec_path = Path(scratch) / f"{index}.pickle"
                        result_path = Path(scratch) / f"{index}.result"
                        result_path.unlink(missing_ok=True)
                        with spec_path.open("wb") as handle:
                            pickle.dump(epoch_spec(obs, force), handle)
                        env = full_cpu_environment()
                        env["VLBIPY_EPOCH_MEMORY_BYTES"] = str(epoch_memory(obs))
                        process = subprocess.Popen([sys.executable, "-m", "vlbipy.epoch_jobs", str(spec_path),
                                                    str(result_path)], env=env)
                        running.append((obs, result_path, process))
                    except Exception:
                        error = traceback.format_exc()
                        obs._state.mark_failed("epoch_products", error)
                        failures.append(obs.project_code)
                        logger.error("[{}] epoch could not start: {}", obs.project_code, error)
                for obs, result_path, process in running:
                    try:
                        code = process.wait()
                        if code != 0:
                            raise RuntimeError(f"epoch subprocess exited with status {code}")
                        with result_path.open("rb") as handle:
                            payload = pickle.load(handle)
                        if "error" not in payload:
                            images[obs.project_code] = payload["images"]
                        for name, value in payload["state"].items():
                            setattr(obs, name, value)
                        obs._state._steps = payload["steps"]
                        for message in payload["warnings"]:
                            warnings.warn(message)
                    except Exception:
                        error = traceback.format_exc()
                        try:
                            obs._state.load()
                            obs.gaintables = obs._load_gaintables()
                            obs._cal_selection = obs._load_cal_selection()
                        except Exception:
                            error += "\nDisk state recovery failed:\n" + traceback.format_exc()
                        payload = {"error": error}
                    if "error" in payload:
                        obs._state.mark_failed("epoch_products", payload["error"])
                        failures.append(obs.project_code)
                        logger.error("[{}] epoch failed: {}", obs.project_code, payload["error"])
                    else:
                        images[obs.project_code] = payload["images"]
                        obs._state.mark_complete("epoch_products")
            finally:
                for _, _, process in running:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    if failures:
        raise RuntimeError("epoch products failed for " + ", ".join(failures) + "; successful epochs are preserved")
    return {obs.project_code: images[obs.project_code] for obs in observations}


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as handle:
        specification = pickle.load(handle)
    result = finish_epoch(specification)
    with open(sys.argv[2], "wb") as handle:
        pickle.dump(result, handle)
