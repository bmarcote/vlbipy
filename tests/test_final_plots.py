"""Tests for the per-scan diagnostics, final-data plots, imaging sweep and SNR helpers (dummy backend)."""
from __future__ import annotations

import numpy as np

from vlbipy import ImageSet, VLBIObs
from vlbipy.models import ScanSNRSurvey


def make_obs(**roles):
    """A single-project dummy observation with the given source roles, already imported."""
    roles.setdefault("target", "3C395")
    obs = VLBIObs("fp01", network="EVN", backend="dummy", **roles)
    obs.import_data()
    return obs


# -- scan selection --

def test_scan_diagnostics_uses_every_fringe_finder_scan():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345")
    scans = obs["fp01"].plot._diagnostic_scans()
    assert scans and all(s.source == "3C345" for s in scans)
    assert len(scans) == len(obs["fp01"].metadata.scans_for_source("3C345"))


def test_scan_diagnostics_falls_back_to_spread_phasecal_scans():
    obs = make_obs(phasecal="J1848+3219")
    observation = obs["fp01"]
    # Give the phase calibrator twelve scans, evenly spaced; every antenna in all of them.
    template = observation.metadata.scans_for_source("J1848+3219")[0]
    observation.metadata.scans = [type(template)(scan_number=i + 1, source="J1848+3219", time_start=600.0 * i,
                                                 time_end=600.0 * i + 300.0, antennas=list(template.antennas),
                                                 integration_time=2.0, subbands=template.subbands)
                                  for i in range(12)]
    chosen = observation.plot._diagnostic_scans()
    assert len(chosen) == 5
    assert [s.scan_number for s in chosen] == [1, 4, 7, 9, 12]     # spread evenly over the track


def test_scan_diagnostics_covers_antennas_missing_from_the_spread_scans():
    obs = make_obs(phasecal="J1848+3219")
    observation = obs["fp01"]
    antennas = list(observation.metadata.antennas)
    template = observation.metadata.scans_for_source("J1848+3219")[0]
    scans = []
    for i in range(12):
        # The last antenna only shows up in scan 7, which the even spread never picks.
        present = antennas if i == 6 else antennas[:-1]
        scans.append(type(template)(scan_number=i + 1, source="J1848+3219", time_start=600.0 * i,
                                    time_end=600.0 * i + 300.0, antennas=list(present), integration_time=2.0,
                                    subbands=template.subbands))
    observation.metadata.scans = scans
    chosen = observation.plot._diagnostic_scans()
    assert len(chosen) <= 5
    assert 7 in [s.scan_number for s in chosen]
    assert set(antennas) <= {a for s in chosen for a in s.antennas}


def test_scan_diagnostics_returns_autocorr_and_spectrum_paths():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345")
    paths = obs.plot.scan_diagnostics(column="data", label="raw")
    finder_scans = obs["fp01"].metadata.scans_for_source("3C345")
    assert len(paths) == len(finder_scans) * 3             # 1 autocorr + 2 spectrum PNGs per scan
    first = finder_scans[0].scan_number
    assert any(f".raw_scan{first}.autocorr.png" in p for p in paths)
    assert any(f".raw_scan{first}.spectrum" in p for p in paths)
    assert all("/plots/raw/" in p for p in paths)
    assert obs.plot.raw_stokes() == paths                  # backwards-compatible alias


# -- reference antenna per scan --

def test_refant_for_scan_prefers_refant_then_ranking_then_first():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345", refant="EF")
    observation = obs["fp01"]
    scan = observation.metadata.scans[0]
    assert observation.plot._refant_for_scan(scan) == "EF"
    scan.antennas = [a for a in scan.antennas if a != "EF"]
    observation.calibrate.scan_snr()                        # survey ranking is now available
    ranked = [a for a, _ in observation.metadata.snr_survey.rank_antennas() if a in scan.antennas]
    assert observation.plot._refant_for_scan(scan) == ranked[0]
    observation.metadata.snr_survey = None
    assert observation.plot._refant_for_scan(scan) == scan.antennas[0]


def test_refant_chain_takes_first_member_present():
    obs = make_obs(phasecal="J1848+3219", refant=["ZZ", "O8", "EF"])
    scan = obs["fp01"].metadata.scans[0]
    assert obs["fp01"].plot._refant_for_scan(scan) == "O8"


# -- final-data set, lightcurve, autocorr --

def test_final_data_returns_plot_paths_for_every_source():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345")
    paths = obs.plot.final_data()
    assert isinstance(paths, list) and paths
    assert any(".final.lightcurve.png" in p for p in paths)
    for name in ("3C395", "J1848+3219", "3C345"):
        assert any(f".final.radplot.{name}.png" in p for p in paths)
        assert any(f".final_{name}.spectrum" in p for p in paths)
        assert any(f".final_{name}.timeseries.png" in p for p in paths)
    assert any(p.endswith(".uv_coverage.png") for p in paths)


def test_lightcurve_path_and_default_averaging():
    obs = make_obs(phasecal="J1848+3219")
    assert obs.config["export"]["lightcurve_averaging"] == [0, 30, 120, -1]
    path = obs.plot.lightcurve()
    assert path.endswith("fp01.calibrated.lightcurve.png") and "/plots/calibrated/" in path
    assert obs.plot.lightcurve(column="data", label="raw").endswith("/plots/raw/fp01.raw.lightcurve.png")


def test_dummy_read_autocorr_spectrum_shape():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345")
    backend = obs["fp01"]._backend
    meta = obs["fp01"].metadata
    scan = meta.scans_for_source("3C345")[0].scan_number
    spectrum = backend.data.read_autocorr_spectrum("fp01", field="3C345", scans=[scan], metadata=meta)
    assert spectrum["antennas"] == list(meta.antennas)
    assert spectrum["polarizations"] == ["RR", "LL"]
    first = np.asarray(spectrum["spectra"][spectrum["antennas"][0]])
    assert first.shape == (meta.freq_setup.n_subbands, meta.freq_setup.n_channels, 2)
    assert spectrum["scans"] == [scan] and spectrum["column"] == "data"
    # A scan of another source selects nothing.
    other = meta.scans_for_source("3C395")[0].scan_number
    assert backend.data.read_autocorr_spectrum("fp01", field="3C345", scans=[other], metadata=meta)["antennas"] == []


def test_dummy_read_total_visibility_matches_lightcurve_input():
    obs = make_obs(phasecal="J1848+3219")
    backend = obs["fp01"]._backend
    data = backend.data.read_total_visibility("fp01", fields=["3C395"], metadata=obs["fp01"].metadata)
    assert set(data["sources"]) == {"3C395"}
    entry = data["sources"]["3C395"]
    assert len(entry["times"]) == len(entry["vis_sum"]) == len(entry["n_vis"]) == len(entry["scans"]) > 0


def test_diagnostics_raw_includes_scan_diagnostics_and_uv_coverage():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345")
    paths = obs.plot.diagnostics(column="data")
    assert any(".autocorr.png" in p for p in paths)
    assert any(p.endswith("fp01.uv_coverage.png") for p in paths)
    calibrated = obs.plot.diagnostics(column="corrected")
    assert not any(".autocorr.png" in p for p in calibrated)


# -- imaging sweep and pipeline --

def test_radplot_all_sources():
    obs = make_obs(phasecal="J1848+3219", fringe_finder="3C345")
    paths = obs.plot.radplot(all_sources=True, with_model=True)
    assert len(paths) == 3
    assert len(obs.plot.radplot()) == 2                     # default: calibrators only


def test_clean_default_sweep_sets_png_from_image_grid():
    obs = make_obs(phasecal="J1848+3219")
    images = obs.clean("3C395")
    assert isinstance(images, ImageSet) and [im.robust for im in images] == [-2.0, 0.0, 2.0]
    assert all(im.paths["png"].endswith("fp01.images.3C395.png") for im in images)
    assert obs["fp01"].state.as_dict()["clean_3C395"]["status"] == "done"


def test_run_images_all_sources_and_records_final_plots():
    obs = VLBIObs("fp02", network="EVN", backend="dummy", target="3C395", phasecal="J1848+3219",
                  fringe_finder="3C345")
    images = obs.run()
    assert set(images) == {"3C395", "J1848+3219", "3C345"}
    state = obs["fp02"].state.as_dict()
    assert state["final_plots"]["status"] == "done"
    assert all(state[f"clean_{n}"]["status"] == "done" for n in images)


def test_run_finishes_with_all_source_all_scan_snr_without_changing_selection(monkeypatch):
    obs = VLBIObs("fp04", network="EVN", backend="dummy", target="3C395",
                  phasecal="J1848+3219", fringe_finder="3C345")
    observation = obs["fp04"]
    calls = []
    original = observation._backend.calibrate.scan_snr

    def record(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(observation._backend.calibrate, "scan_snr", record)
    obs.run()
    metadata = observation.metadata
    final_args, final_kwargs = calls[-1]
    assert set(final_args[1].split(",")) == set(metadata.source_names)
    assert final_kwargs["max_scans"] == 0
    assert set(metadata.snr_survey.scan_numbers) == {scan.scan_number for scan in metadata.scans}
    assert observation.cal_selection is not None


def test_run_images_only_targets_when_configured():
    obs = VLBIObs("fp03", network="EVN", backend="dummy", target="3C395", phasecal="J1848+3219",
                  imaging={"sources": "targets"})
    assert set(obs.run()) == {"3C395"}


# -- SNR per scan --

def test_scan_snr_survey_per_scan_antenna_and_namespace_helper():
    nan = float("nan")
    survey = ScanSNRSurvey(project_code="p", scan_numbers=[3, 7], scan_sources=["A", "A"],
                           antennas=["EF", "WB", "O8"],
                           snr={"RR": [[nan, 10.0, 30.0], [nan, nan, 4.0]],
                                "LL": [[nan, 20.0, nan], [nan, nan, 6.0]]})
    per_scan = survey.per_scan_antenna()
    assert per_scan == {3: {"EF": None, "WB": 15.0, "O8": 30.0}, 7: {"EF": None, "WB": None, "O8": 5.0}}

    obs = make_obs(phasecal="J1848+3219")
    assert obs.plot.snr_for_scans() == {}
    obs.calibrate.scan_snr()
    table = obs.plot.snr_for_scans()
    assert set(table) == set(obs["fp01"].metadata.snr_survey.scan_numbers)
    assert all(set(row) == set(obs["fp01"].metadata.antennas) for row in table.values())
