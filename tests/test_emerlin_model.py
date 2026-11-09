"""The e-MERLIN flux scale: the 3C286 model, its prediction, the table it solves and the check against it."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlbipy.models import CalTable
from vlbipy.solvers import setjy

MODELS = Path(__file__).resolve().parents[1] / "src" / "vlbipy" / "data" / "emerlin_calibrator_models"


@pytest.mark.parametrize("band", ["L", "C"])
def test_the_casa_model_is_3c286_at_its_catalogue_flux(band):
    """Every non-zero pixel is a component (negative CLEAN ones too), the total matches Perley-Butler 2017 at the
    image frequency, and the extended emission lies south-west of the core, where 3C286's jet points."""
    pytest.importorskip("casacore.images")
    flux, l, m, freq = setjy.casa_image_components(MODELS / f"3C286_{band}.clean.model.tt0")
    assert (flux < 0).any() and freq > 1e9
    assert flux.sum() == pytest.approx(setjy.pb17_flux_3c286(freq), rel=0.02)
    extended = np.where(np.arange(flux.size) == np.argmax(flux), 0.0, flux)
    assert np.sum(extended * l) < 0 and np.sum(extended * m) < 0


def test_prediction_is_the_direct_sum():
    """The phasor recurrence of an evenly spaced band, and the direct sum of an uneven one, give the same."""
    rng = np.random.default_rng(1)
    flux, l, m = rng.uniform(-0.2, 1.0, 40), rng.normal(0, 3e-7, 40), rng.normal(0, 3e-7, 40)
    uvw = rng.normal(0, 1e5, (25, 3))
    for freqs in (1.6e9 + 0.5e6 * np.arange(16), np.array([1.6e9, 1.61e9, 1.65e9])):
        direct = np.array([[np.sum(flux * np.exp(-2j * np.pi * (u * l + v * m) * f / setjy.SPEED_OF_LIGHT))
                            for f in freqs] for u, v, _ in uvw])
        np.testing.assert_allclose(setjy.predict(uvw, freqs, flux, l, m, max_cells=100), direct, atol=1e-9)


def test_only_a_standard_follows_its_catalogue_spectrum():
    flux = np.array([10.0, 4.0])
    freqs = np.array([1.6e9, 1.7e9])
    scale = setjy.spectral_scale(freqs, flux, "3C286")
    np.testing.assert_allclose(scale * flux.sum(), [setjy.pb17_flux_3c286(f) for f in freqs])
    assert setjy.spectral_scale(freqs, flux, "OQ208") is None
    np.testing.assert_allclose(setjy.spectral_scale(freqs, flux, "OQ208", flux_total=7.0), 0.5)
    assert setjy.spectral_scale(freqs, flux, "1331+305") is not None


def test_a_fits_model_keeps_its_ra_direction(tmp_path):
    """A component one pixel east of the reference (lower x, CDELT1 < 0) has l > 0: the absolute increments used
    before mirrored the source in RA, which changes the amplitudes of anything asymmetric."""
    from astropy.io import fits
    data = np.zeros((1, 1, 5, 5), np.float32)
    data[0, 0, 2, 2], data[0, 0, 3, 1] = 1.0, 0.5               # the centre, and one pixel east and north of it
    header = fits.Header({"CRPIX1": 3, "CRPIX2": 3, "CDELT1": -1e-5, "CDELT2": 1e-5, "CTYPE1": "RA---SIN",
                          "CTYPE2": "DEC--SIN", "CTYPE3": "FREQ", "CRVAL3": 1.5e9, "CDELT3": 1e6, "CRPIX3": 1})
    path = tmp_path / "model.fits"
    fits.PrimaryHDU(data, header).writeto(path)
    flux, l, m, freq = setjy.model_components(path, threshold=0.1)
    east = int(np.argmin(flux))
    assert l[east] > 0 and m[east] > 0 and freq == 1.5e9


def test_antenna_ratios_from_baselines():
    from vlbipy.solvers.fluxcheck import antenna_ratios
    gains = {0: 1.10, 1: 0.90, 2: 1.00, 3: 1.05}
    baselines = {(i, j): (gains[i] * gains[j], 10.0) for i in gains for j in gains if i < j}
    assert antenna_ratios(baselines) == pytest.approx(gains)
    assert antenna_ratios({(0, 1): (1.2, 1.0)}) == {}                 # one baseline cannot split its antennas


class _Table:
    """A writable in-memory calibration table for the e-MERLIN table helpers."""
    stores: dict = {}

    def open(self, path, nomodify=True):
        self.path = str(path)
        return self.path in self.stores

    def getcol(self, name):
        return np.array(self.stores[self.path][name])

    def putcol(self, name, value):
        self.stores[self.path][name] = np.array(value)

    def flush(self):
        pass

    def close(self):
        pass


def _ops(monkeypatch, names):
    from vlbipy.backends.casa import CasaCalibrationOps
    ops = CasaCalibrationOps.__new__(CasaCalibrationOps)
    ops._backend = SimpleNamespace(tools=SimpleNamespace(table=_Table))
    monkeypatch.setattr(ops, "_table_antenna_names", lambda path: list(names))
    return ops


def test_the_table_corrects_only_the_emerlin_antennas_and_keeps_one_level(tmp_path, monkeypatch):
    """EF gets unit gain; DA's flagged hand takes DA's median gain, so its data are not flagged at apply time."""
    path = tmp_path / "X.emgain"
    gains = np.array([[[0.8, 1.25, 2.0, 3.0]], [[0.8, 1.0, 2.0, 3.0]]], dtype=complex)   # (pol, chan, row)
    flags = np.zeros(gains.shape, bool)
    flags[1, 0, 1] = True                                                                 # DA, hand 2
    _Table.stores = {str(path): {"ANTENNA1": np.array([0, 1, 2, 3]), "SPECTRAL_WINDOW_ID": np.array([1, 1, 1, 1]),
                                 "CPARAM": gains, "FLAG": flags}}
    ops = _ops(monkeypatch, ["JB", "DA", "KN", "EF"])
    summary = ops._complete_emerlin_table(path, ["JB", "DA", "KN"])
    store = _Table.stores[str(path)]
    assert np.allclose(store["CPARAM"][:, 0, 3], 1.0) and not store["FLAG"].any()
    assert store["CPARAM"][1, 0, 1] == pytest.approx(1.25)                 # DA's only solved value
    assert summary["spws"] == {1} and summary["filled"] == {"DA": 1}
    assert summary["factors"]["JB"]["1"] == pytest.approx([1.25, 1.25])
    assert summary["factors"]["DA"]["1"] == pytest.approx([0.8, None])


def test_the_solve_is_phase_then_amplitude_on_the_emerlin_baselines(tmp_path, monkeypatch):
    """Model, phases, then one amplitude per subband over all scans, on the baselines among the outstations and
    Jb; a subband without e-MERLIN data maps to the nearest solved one, and the phase table is deleted."""
    import json
    ops = _ops(monkeypatch, [])
    caldir = tmp_path
    ops._backend = SimpleNamespace(ms_path=lambda code: tmp_path / "X.ms", caldir=lambda: caldir)
    calls = []
    monkeypatch.setattr(ops, "_set_model_column", lambda ms, field, model: calls.append(("model", field, model)))
    monkeypatch.setattr(ops, "_prior_callib", lambda code, tables, path, field="", callib=False:
                        {"gaintable": [t.path if isinstance(t, CalTable) else t for t in tables or []]})

    def gaincal(params):
        calls.append(params)
        Path(params["caltable"]).mkdir()

    monkeypatch.setattr(ops, "_run_gaincal_task", gaincal)
    monkeypatch.setattr(ops, "_complete_emerlin_table",
                        lambda path, solved: calls.append(("complete", solved)) or
                        {"spws": {1, 2}, "factors": {"DA": {"1": [1.1, 1.1]}}, "filled": {}})
    table = ops.emerlin_scale("X", "3C286", ["DA", "KN"], "JB", model="m.tt0", gaintable=["a.tsys"], nspw=4)
    model, phase, amplitude, complete = calls
    assert model == ("model", "3C286", "m.tt0")
    assert phase["calmode"] == "p" and phase["antenna"] == "DA,KN,JB&" and phase["refant"] == "JB"
    assert amplitude["calmode"] == "a" and amplitude["solint"] == "inf" and amplitude["combine"] == "scan"
    assert amplitude["gaintable"] == ["a.tsys", phase["caltable"]]
    assert complete == ("complete", ["DA", "KN", "JB"])                    # Jb solved too
    assert table.cal_type == "emgain" and table.spwmap == [1, 1, 2, 2]
    assert not Path(phase["caltable"]).exists()
    assert json.loads(Path(f"{table.path}.json").read_text())["spwmap"] == [1, 1, 2, 2]


@pytest.mark.parametrize("kind, name", [("daskms", "3C286_L.clean.model.tt0"), ("casa", "3C286_L.clean.model.tt0"),
                                        ("aips", "3C286_L.ICL001")])
def test_each_backend_takes_the_model_in_its_own_format(kind, name):
    from vlbipy.namespaces import CalibrateNamespace
    namespace = object.__new__(CalibrateNamespace)
    namespace._obs = SimpleNamespace(_backend=SimpleNamespace(kind=kind))
    assert Path(namespace._emerlin_model(1.66, {})).name == name
    assert namespace._emerlin_model(22.0, {}) is None


@pytest.mark.parametrize("hold, expected", [(False, ["DA", "KN", "JB"]), (True, ["DA", "KN"])])
def test_the_antennas_self_calibration_holds(hold, expected):
    from vlbipy.namespaces import emerlin_antennas
    obs = SimpleNamespace(gaintables=[CalTable(cal_type="emgain", path="x")],
                          metadata=SimpleNamespace(antennas={"EF": 0, "DA": 1, "KN": 2, "JB": 3}),
                          config={"calibration": {"emerlin": {"hold_reference": hold}}})
    assert emerlin_antennas(obs) == expected
    obs.gaintables = []
    assert emerlin_antennas(obs) == []


def test_a_rerun_of_the_ai_flagging_starts_from_the_flags_before_it(monkeypatch):
    """A resumed run (from emerlin_scale, say) redoes the AI distribution flagging on the new calibration: the
    previous run's flags go first, or every run would add its own on top."""
    from vlbipy.ai.namespace import AINamespace
    from vlbipy.solvers import flagmanager
    calls = []
    monkeypatch.setattr(flagmanager, "list_versions", lambda ms: [("as_imported", ""),
                                                                  ("before_ai_flag_distribution", "")])
    monkeypatch.setattr(flagmanager, "restore_version", lambda ms, name: calls.append(("restore", name)))
    monkeypatch.setattr(flagmanager, "delete_version", lambda ms, name: calls.append(("delete", name)))

    class Stop(Exception):
        pass

    namespace = object.__new__(AINamespace)
    namespace._obs = SimpleNamespace(config={"ai": {}}, _state=SimpleNamespace(should_run=lambda *a, **k: True),
                                     _backend=SimpleNamespace(requires_data_files=True, ms_path=lambda code: "X.ms"),
                                     project_code="X")
    monkeypatch.setattr(AINamespace, "_calibrators", lambda self: (_ for _ in ()).throw(Stop()))
    monkeypatch.setattr(AINamespace, "_code", property(lambda self: "X"))
    with pytest.raises(Stop):
        namespace.flag_distribution(force=True)
    assert calls == [("restore", "before_ai_flag_distribution"), ("delete", "before_ai_flag_distribution")]


def test_a_resumed_run_takes_the_roles_an_earlier_run_inferred():
    """EM178 resumed from emerlin_scale: the step that adopts the saved roles was skipped, so there was no fringe
    finder and self-calibration went straight to the phase calibrator."""
    from vlbipy.namespaces import CalibrateNamespace
    from vlbipy.sources import SourceType
    adopted = []
    obs = SimpleNamespace(metadata=SimpleNamespace(source_names=["FF", "PC", "TGT"]), sources=["PC", "TGT"],
                          load_source_roles=lambda: ({"FF": SourceType.FRINGE_FINDER, "PC": SourceType.PHASE_CALIBRATOR}, {}),
                          adopt_source_roles=lambda roles, phaseref, persist=True: adopted.append((roles, persist)))
    namespace = object.__new__(CalibrateNamespace)
    namespace._obs = obs
    assert namespace.reuse_source_roles() == {"FF": SourceType.FRINGE_FINDER}
    assert adopted == [({"FF": SourceType.FRINGE_FINDER}, False)]
