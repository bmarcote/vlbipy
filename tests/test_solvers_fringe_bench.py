"""Benchmark of vlbipy.solvers.fringe on the pipeline-sized problem (12 antennas, 4 spw x 52 channels, 2 pols).

Skipped unless the environment variable VLBIPY_BENCH=1 is set; run with ``-s`` to see the timing table::

    VLBIPY_BENCH=1 .venv/bin/pytest -s tests/test_solvers_fringe_bench.py
"""
import os
import time

import numpy as np
import pytest

from vlbipy.solvers.fringe import fringefit_interval

from test_solvers_fringe import simulate

pytestmark = pytest.mark.skipif(os.environ.get("VLBIPY_BENCH") != "1", reason="set VLBIPY_BENCH=1 to run benchmarks")


def _bench(data, refant, nrep=5):
    """Warm run followed by nrep timed runs; returns (median total, median fft, median lsq, solution) in seconds."""
    sol = fringefit_interval(data, refant)
    tot, fft, lsq = [], [], []
    for _ in range(nrep):
        t0 = time.perf_counter()
        sol = fringefit_interval(data, refant)
        tot.append(time.perf_counter() - t0)
        fft.append(sol.timings["fft"])
        lsq.append(sol.timings["lsq"])
    return float(np.median(tot)), float(np.median(fft)), float(np.median(lsq)), sol


@pytest.mark.parametrize("nint", [150, 600])
def test_bench_fringefit_interval(nint):
    data, truth = simulate(nant=12, refant=3, nspw=4, nchan=52, nint=nint, dead_ant=-1, seed=1)
    total, fft, lsq, sol = _bench(data, truth["refant"])
    nsamp = data.vis.size
    print(f"\nfringefit_interval nint={nint} ({nsamp / 1e6:.1f} M samples): total {total:.3f} s "
          f"(fft {fft:.3f} s, lsq {lsq:.3f} s, {1e9 * lsq / nsamp:.1f} ns/sample), n_iter {sol.n_iter.tolist()}")
    assert not sol.flag.any()
    good = np.ones_like(sol.flag)
    good[truth["refant"]] = False
    d_delay = np.abs(sol.delay_ns - truth["params"][..., 1])[good]
    assert np.all(d_delay < np.maximum(3 * sol.paramerr[..., 1][good], 0.1))
