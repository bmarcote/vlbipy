"""Model visibilities of difmap components and their overlay in the plot readers."""
import json

import numpy as np
import pytest

from vlbipy.solvers import plot_read, skymodel

FREQ = np.array([1.6e9, 1.7e9])


def test_point_at_the_phase_centre_is_its_flux_everywhere():
    uvw = np.array([[1e6, -3e6, 0.0], [5e5, 2e5, 0.0]])
    model = skymodel.predict([{"type": "delta", "flux": 0.5, "x": 0.0, "y": 0.0}], uvw, FREQ)
    np.testing.assert_allclose(model, 0.5)


def test_offset_component_turns_the_phase_as_difmap_does():
    """V = S exp(+2 pi i (u x + v y)), u in wavelengths, x east: the convention difmapy fits with."""
    x_mas = 10.0
    u_lambda = 1.0 / (4.0 * x_mas * skymodel.MAS)                  # a quarter turn
    uvw = np.array([[u_lambda * skymodel.C_M_S / FREQ[0], 0.0, 0.0]])
    model = skymodel.predict([{"type": "delta", "flux": 1.0, "x": x_mas, "y": 0.0}], uvw, FREQ[:1])
    np.testing.assert_allclose(model[0, 0], 1j, atol=1e-9)


def test_gaussian_amplitude_falls_to_half_at_the_expected_baseline():
    """A circular Gaussian of FWHM a has |V| = 1/2 at a uv distance of 2 ln2 / (pi a)."""
    fwhm = 2.0 * skymodel.MAS
    q = 2.0 * np.log(2.0) / (np.pi * fwhm)
    for angle in (0.0, 0.7):
        uvw = np.array([[q * np.cos(angle), q * np.sin(angle), 0.0]]) * skymodel.C_M_S / FREQ[0]
        model = skymodel.predict([{"type": "gauss", "flux": 1.0, "x": 0, "y": 0, "major": 2.0, "ratio": 1.0,
                                   "phi": 30.0}], uvw, FREQ[:1])
        assert abs(model[0, 0]) == pytest.approx(0.5, rel=1e-6)


def test_recorded_model_moves_components_back_to_the_original_phase_centre(tmp_path):
    (tmp_path / "images").mkdir()
    record = {"model": {"components": [{"type": "gauss", "flux": 1e-3, "x": 0.5, "y": -0.2, "major": 1.0},
                                       {"type": "gauss", "flux": float("nan"), "x": 0.0, "y": 0.0}]},
              "shift_mas": [120.0, -40.0]}
    (tmp_path / "images" / "P1_TGT.search.json").write_text(json.dumps(record))
    components = skymodel.recorded_model(tmp_path, "P1", "TGT")
    assert [(c["x"], c["y"]) for c in components] == [(120.5, -40.2)]
    assert skymodel.recorded_model(tmp_path, "P1", "OTHER") == []


def test_spectrum_job_averages_the_model_under_the_data_flags(monkeypatch):
    """The model is predicted per row and channel, flagged where the data are, and averaged like them."""
    uvw = np.array([[1e5, 0.0, 0.0], [2e5, 0.0, 0.0], [3e5, 0.0, 0.0]])
    flags = np.zeros((3, 2, 2), dtype=bool)
    flags[1, 0, :] = True
    block = {"DATA": np.ones((3, 2, 2), complex), "FLAG": flags, "ANTENNA1": np.array([0, 0, 0]),
             "ANTENNA2": np.array([1, 1, 1]), "DATA_DESC_ID": np.array([0, 0, 0]), "UVW": uvw}
    monkeypatch.setattr(plot_read, "read_view_runs", lambda *a, **k: {key: value.copy() for key, value in block.items()})
    component = {"type": "delta", "flux": 2.0, "x": 50.0, "y": 0.0}
    chan_freq = [[1.6e9, 1.7e9]]
    out = plot_read.spectrum_job(specs=[], column="DATA", corrs=None, ncorr=2, engine="stub", refant=0, nant=2,
                                 nddid=1, model=[component], chan_freq=chan_freq, ddid_to_spw=[0])
    expected = skymodel.predict([component], uvw, np.array(chan_freq[0]))
    np.testing.assert_allclose(out["model_sums"][0, 0, 0], expected[[0, 2], 0].sum())       # row 1 flagged
    np.testing.assert_allclose(out["model_sums"][0, 1, 1], expected[:, 1].sum())
    np.testing.assert_array_equal(out["model_counts"][0, :, 0], [2, 3])
    assert "model_sums" not in plot_read.spectrum_job(specs=[], column="DATA", corrs=None, ncorr=2, engine="stub",
                                                       refant=0, nant=2, nddid=1)
