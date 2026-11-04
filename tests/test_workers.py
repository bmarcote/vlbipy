"""The solver worker pool (:mod:`vlbipy.solvers.workers`): error types across the process boundary, stopping
the other jobs when one fails, and the CPU count used for the per-job thread share."""
from __future__ import annotations

import time

import numpy as np
import pytest

from vlbipy.solvers import workers

#: A pool job that raises NotImplementedError (a row before the first scan bound).
NOT_IMPLEMENTED_JOB = {"rows": {"TIME": np.array([0.0])}, "bounds": np.array([10.0]), "interval": 1.0}


@pytest.fixture
def pool_of_two():
    """A shared pool of two workers, stopped afterwards."""
    yield workers.get_pool(2)
    workers.close_pool()


def test_not_implemented_in_a_pool_job_stays_a_not_implemented_error(pool_of_two):
    with pytest.raises(NotImplementedError, match="before the first scan bound") as caught:
        workers.run_jobs("vlbipy.solvers.accor:chunk_keys", [NOT_IMPLEMENTED_JOB] * 2, workers=2)
    assert isinstance(caught.value, workers.WorkerError)
    assert "Traceback" in caught.value.remote_traceback
    with pytest.raises(workers.WorkerError) as other:                    # any other failure is not
        workers.run_jobs("json:loads", [{"s": "{"}, {"s": "[1]"}], workers=2)
    assert not isinstance(other.value, NotImplementedError)


def test_not_implemented_in_an_in_process_future_is_typed_too():
    future = workers._run_here("vlbipy.solvers.accor:chunk_keys", NOT_IMPLEMENTED_JOB)
    with pytest.raises(workers.WorkerNotImplementedError):
        future.result()


def test_a_failed_job_cancels_the_pending_ones_and_waits_for_the_running_ones(pool_of_two, tmp_path):
    jobs = [{"args": ["false"], "check": True}]
    jobs += [{"args": ["sh", "-c", f"sleep 0.3; touch {tmp_path / str(k)}"]} for k in range(1, 8)]
    with pytest.raises(workers.WorkerError, match="CalledProcessError"):
        workers.run_jobs("subprocess:run", jobs, workers=2)
    assert pool_of_two._idle.qsize() == pool_of_two.size                # no job is still running
    written = sorted(p.name for p in tmp_path.iterdir())
    time.sleep(1.0)
    assert sorted(p.name for p in tmp_path.iterdir()) == written        # and none starts afterwards
    assert len(written) < 7


def test_job_threads_use_the_usable_cpu_count(monkeypatch):
    monkeypatch.delenv(workers.THREADS_ENV, raising=False)
    monkeypatch.setattr(workers, "_JOB_THREADS", None)
    monkeypatch.setattr(workers, "_CPUS", 3)
    assert workers.job_threads() == 3
