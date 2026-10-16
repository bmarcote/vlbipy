"""Tests for the fast diagnostic plots of the dask-ms backend: the jplotter batching (vlbipy.jplot) and the
parallel readers behind the corner plots, the radplot and the time-evolution figures (vlbipy.solvers.plot_read).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vlbipy import jplot


# ----------------------------------------------------------------------------------------------------------------
# jplotter batching
# ----------------------------------------------------------------------------------------------------------------
def test_pages_are_renamed_from_the_dot_free_scratch_names(tmp_path):
    stem = tmp_path / "GL048.raw_scan1.spectrum"
    scratch = jplot._scratch_stem(stem, 3)
    assert "." not in scratch.name
    (scratch.parent / f"{scratch.name}.png").write_bytes(b"one")
    assert jplot._pages(scratch, stem) == [tmp_path / "GL048.raw_scan1.spectrum.png"]
    multi = jplot._scratch_stem(tmp_path / "GL048.raw.time_evolution.refYY.spw7", 4)
    for page in range(3):
        (multi.parent / f"{multi.name}_{page:04d}.png").write_bytes(b"p")
    names = [p.name for p in jplot._pages(multi, tmp_path / "GL048.raw.time_evolution.refYY.spw7")]
    assert names == ["GL048.raw.time_evolution.refYY.spw7.png", "GL048.raw.time_evolution.refYY.spw7.page2.png",
                     "GL048.raw.time_evolution.refYY.spw7.page3.png"]
    assert jplot._pages(jplot._scratch_stem(stem, 9), stem) == []


def test_recipes_never_select_every_scan_explicitly():
    figures = [jplot.cross_spectrum("a", refant="YY", source="J1"), jplot.auto_spectrum("b", scans=[3, 4]),
               jplot.amp_phase_time("c", refant="YY", subband=7), jplot.cross_spectrum("d", refant="EF", scans=[1])]
    for figure in figures:
        assert not any(c.startswith("scan start") for c in figure["commands"])
    assert "scan 3 4" in figures[1]["commands"] and figures[1]["scans"] and not figures[0]["scans"]
    assert figures[1]["readflags"] is False and figures[0]["readflags"] is True
    assert "fq 7/p;ch 0.1*last:0.9*last" in figures[2]["commands"]


def test_draw_groups_figures_into_sessions_by_flags_plot_type_and_scan_selection(tmp_path, monkeypatch):
    sessions = []

    def fake_run_jobs(function, jobs, workers=None):
        sessions.extend(jobs)
        return [[{"name": f["name"], "files": [f["name"] + ".png"], "seconds": 1.0, "error": ""}
                 for f in job["figures"]] for job in jobs]

    monkeypatch.setattr("vlbipy.solvers.workers.run_jobs", fake_run_jobs)
    figures = ([jplot.auto_spectrum(str(tmp_path / f"auto{n}"), scans=[n]) for n in range(3)]
               + [jplot.cross_spectrum(str(tmp_path / f"cross{n}"), refant="YY", scans=[n]) for n in range(3)]
               + [jplot.cross_spectrum(str(tmp_path / "field"), refant="YY", source="J1")]
               + [jplot.amp_phase_time(str(tmp_path / "time"), refant="YY", subband=7)])
    result = jplot.draw(tmp_path / "x.ms", figures, log_dir=tmp_path)
    kinds = sorted((job["readflags"], job["figures"][0]["plot_type"], job["figures"][0]["scans"],
                    len(job["figures"])) for job in sessions)
    assert kinds == [(False, "ampchan", True, 3), (True, "anpchan", False, 1), (True, "anpchan", True, 3),
                     (True, "anptime", False, 1)]
    assert len(result["files"]) == len(figures) and not result["failed"]


# ----------------------------------------------------------------------------------------------------------------
# Parallel readers
# ----------------------------------------------------------------------------------------------------------------
casatools = pytest.importorskip("casatools")

from vlbipy.solvers import plot_read  # noqa: E402

NANT, NSPW, NCHAN, NTIME = 4, 2, 8, 12


def _scalar(value_type):
    return {"valueType": value_type, "dataManagerType": "StandardStMan", "dataManagerGroup": "SSM", "option": 0,
            "maxlen": 0, "comment": "", "keywords": {}}


def _array(value_type, shape):
    desc = _scalar(value_type)
    desc["ndim"], desc["option"], desc["shape"] = len(shape), 5, np.array(shape)
    return desc


def _create(path, desc, nrow, columns):
    tb = casatools.table()
    tb.create(str(path), desc)
    tb.addrows(nrow)
    for name, values in columns.items():
        values = np.asarray(values)
        tb.putcol(name, np.asfortranarray(values.T) if values.ndim > 1 else values)
    tb.close()


@pytest.fixture
def ms(tmp_path):
    """4 antennas with autocorrelations, 2 subbands x 8 channels, RR/LL, 2 scans (fields 0 and 1), with UVW."""
    rng = np.random.default_rng(3)
    path = tmp_path / "plots.ms"
    pairs = [(i, j) for i in range(NANT) for j in range(i, NANT)]
    rows = [(scan, spw, t, i, j) for scan in (1, 2) for t in range(NTIME) for spw in range(NSPW) for i, j in pairs]
    scan, spw, step, ant1, ant2 = (np.array(c) for c in zip(*rows))
    nrow = len(rows)
    time = 5.0e9 + 2.0 * step + 100.0 * (scan - 1)
    data = (rng.uniform(0.5, 1.5, (nrow, NCHAN, 2)) * np.exp(1j * rng.uniform(-3, 3, (nrow, NCHAN, 2))))
    data = data.astype(np.complex64)
    data[(ant1 == 1) & (ant2 == 3), :3] = 0                    # unrecorded channels
    flag = rng.uniform(size=data.shape) < 0.1
    uvw = np.stack([1e5 * (ant2 - ant1) + step, 2e5 * (ant2 - ant1) - step, np.zeros(nrow)], axis=1)
    main = {"TIME": _scalar("double"), "ANTENNA1": _scalar("int"), "ANTENNA2": _scalar("int"),
            "FIELD_ID": _scalar("int"), "SCAN_NUMBER": _scalar("int"), "DATA_DESC_ID": _scalar("int"),
            "FLAG_ROW": _scalar("boolean"), "UVW": _array("double", [3]), "DATA": _array("complex", [2, NCHAN]),
            "FLAG": _array("boolean", [2, NCHAN])}
    _create(path, main, nrow, {"TIME": time, "ANTENNA1": ant1.astype(np.int32), "ANTENNA2": ant2.astype(np.int32),
                               "FIELD_ID": (scan - 1).astype(np.int32), "SCAN_NUMBER": scan.astype(np.int32),
                               "DATA_DESC_ID": spw.astype(np.int32), "FLAG_ROW": np.zeros(nrow, bool), "UVW": uvw,
                               "DATA": data, "FLAG": flag})
    names = np.array(["EF", "JB", "MC", "O6"])
    _create(path / "ANTENNA", {"NAME": _scalar("string"), "POSITION": _array("double", [3]),
                               "MOUNT": _scalar("string")}, NANT,
            {"NAME": names, "POSITION": np.ones((NANT, 3)), "MOUNT": np.array(["ALT-AZ"] * NANT)})
    _create(path / "FIELD", {"NAME": _scalar("string"), "PHASE_DIR": _array("double", [2, 1])}, 2,
            {"NAME": np.array(["CAL", "TARGET"]), "PHASE_DIR": np.zeros((2, 1, 2))})
    _create(path / "SPECTRAL_WINDOW", {"CHAN_FREQ": _array("double", [NCHAN])}, NSPW,
            {"CHAN_FREQ": np.array([[5e9 + 1e6 * c + 1e8 * s for c in range(NCHAN)] for s in range(NSPW)])})
    _create(path / "DATA_DESCRIPTION", {"SPECTRAL_WINDOW_ID": _scalar("int"), "POLARIZATION_ID": _scalar("int")},
            NSPW, {"SPECTRAL_WINDOW_ID": np.arange(NSPW, dtype=np.int32),
                   "POLARIZATION_ID": np.zeros(NSPW, dtype=np.int32)})
    _create(path / "POLARIZATION", {"NUM_CORR": _scalar("int")}, 1, {"NUM_CORR": np.array([2], dtype=np.int32)})
    return {"path": path, "time": time, "ant1": ant1, "ant2": ant2, "spw": spw, "scan": scan, "data": data,
            "flag": flag, "uvw": uvw}


def test_dynamic_spectra_average_the_unflagged_samples_of_each_bin(ms):
    edges = np.linspace(ms["time"][ms["scan"] == 1].min(), ms["time"][ms["scan"] == 1].max(), 4)
    result = plot_read.read_dynamic_spectra(ms["path"], fields=["CAL"], column="DATA", corrs=None, edges=edges,
                                            workers=1)
    values = np.where(ms["flag"], np.nan, ms["data"].astype(np.complex128))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stokes = np.nanmean(values, axis=2)
    bins = np.clip(np.digitize(ms["time"], edges) - 1, 0, 2)
    assert set(result["spectra"]) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    for (a, b), grid in result["spectra"].items():
        for spw in range(NSPW):
            for bin_index in range(3):
                rows = ((ms["ant1"] == a) & (ms["ant2"] == b) & (ms["spw"] == spw) & (ms["scan"] == 1)
                        & (bins == bin_index))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected = np.nanmean(stokes[rows], axis=0)
                np.testing.assert_allclose(grid[bin_index, spw * NCHAN:(spw + 1) * NCHAN], expected, rtol=1e-5)


def test_uvdistance_points_are_per_scan_baseline_subband_and_time_bin(ms):
    starts = {1: float(ms["time"][ms["scan"] == 1].min()), 2: float(ms["time"][ms["scan"] == 2].min())}
    result = plot_read.read_uvdistance(ms["path"], fields=["CAL"], column="DATA", corrs=None, time_bin=6.0,
                                       scan_starts=starts, workers=1)
    # scan 1 (field CAL): 10 baselines incl. autocorrelations x 2 subbands x 4 bins of 6 s (12 samples of 2 s)
    assert result["uvdist_mlambda"].size == 10 * NSPW * 4
    rows = (ms["ant1"] == 0) & (ms["ant2"] == 2) & (ms["spw"] == 1) & (ms["scan"] == 1) & (
        ms["time"] - starts[1] < 6.0)
    uv = ms["uvw"][rows, :2].mean(axis=0)
    expected = np.hypot(*uv) * np.mean(5e9 + 1e8 + 1e6 * np.arange(NCHAN)) / 299792458.0 / 1e6
    assert np.isclose(result["uvdist_mlambda"], expected).any()


def test_time_evolution_rows_lead_with_the_reference_antenna_and_skip_unrecorded_channels(ms):
    other, times, scans, vis = plot_read.read_time_evolution_rows(
        ms["path"], column="DATA", corrs=None, refant=1, ddids=[0], scans=None, channels=list(range(NCHAN)),
        workers=1)
    assert set(other.tolist()) == {0, 2, 3}
    row = np.flatnonzero((ms["ant1"] == 0) & (ms["ant2"] == 1) & (ms["spw"] == 0))[0]
    good = ~ms["flag"][row] & (ms["data"][row] != 0)
    expected = np.conj(np.where(good, ms["data"][row], 0).sum(axis=0) / good.sum(axis=0))   # JB leads: conjugated
    match = np.flatnonzero((other == 0) & (times == ms["time"][row]))
    np.testing.assert_allclose(vis[match[0]], expected, rtol=1e-5)
    row = np.flatnonzero((ms["ant1"] == 1) & (ms["ant2"] == 3) & (ms["spw"] == 0))[0]
    good = ~ms["flag"][row] & (ms["data"][row] != 0)
    assert not good[:3].any()                                     # the zero channels are left out of the mean
    match = np.flatnonzero((other == 3) & (times == ms["time"][row]))
    np.testing.assert_allclose(vis[match[0]], np.where(good, ms["data"][row], 0).sum(axis=0) / good.sum(axis=0),
                               rtol=1e-5)
