"""Tests for the backend registry, component interface, dummy and stub backends."""
import math

import pytest

from vlbipy.backends import DummyBackend, get_backend
from vlbipy.backends.base import Backend, CalibrationOps, DataOps
from vlbipy.errors import BackendError
from vlbipy.models import CalTable, ObsMetadata, ScanSNRSurvey
from vlbipy.results import Image, SelfcalResult


def test_get_backend_dummy():
    be = get_backend("dummy")
    assert isinstance(be, DummyBackend)
    assert be.kind == "dummy"


def test_get_backend_unknown_raises():
    with pytest.raises(BackendError):
        get_backend("nope")


def test_aips_is_stub():
    with pytest.raises(BackendError):
        get_backend("aips")


def test_casa_backend_needs_casatools():
    """CasaBackend constructs when casatools is present, raises BackendError otherwise."""
    try:
        import casatools  # noqa: F401
        backend = get_backend("casa")
        assert backend.kind == "casa" and backend.requires_data_files
    except ImportError:
        with pytest.raises(BackendError):
            get_backend("casa")


def test_dummy_metadata_deterministic():
    be = get_backend("dummy")
    m1 = be.data.get_metadata("RSM07", ["3C286"], "EVN")
    m2 = be.data.get_metadata("RSM07", ["3C286"], "EVN")
    assert isinstance(m1, ObsMetadata)
    assert m1.n_antennas == m2.n_antennas and m1.n_antennas >= 6
    assert m1.freq_setup.freq_ghz == pytest.approx(1.6)
    assert "3C286" in m1.source_names


def test_dummy_calibration_and_imaging_types():
    be = get_backend("dummy")
    assert isinstance(be.calibrate.initial_calibration("P", "FF", "EF"), CalTable)
    img = be.image.clean("P", "3C286", robust=0.0)
    assert isinstance(img, Image)
    assert img.stats.dynamic_range > 0
    # Lower (more negative) robust -> higher rms than natural weighting.
    hi_res = be.image.clean("P", "3C286", robust=-2.0)
    lo_res = be.image.clean("P", "3C286", robust=2.0)
    assert hi_res.stats.rms > lo_res.stats.rms


def test_dummy_selfcal_returns_result():
    be = get_backend("dummy")
    img = be.image.clean("P", "3C286", robust=0.0)
    res = be.image.selfcal("P", "3C286", image=img)
    assert isinstance(res, SelfcalResult)
    assert len(res.rounds) == 4 + 5


# -- component interface --

def test_backend_exposes_six_components():
    """Every backend carries the same six operation components."""
    be = get_backend("dummy")
    assert set(be.components()) == {"data", "calibrate", "flag", "image", "plot", "export"}
    assert be.data.group == "data" and be.calibrate.group == "calibrate"
    assert be.data.kind == "dummy" and be.data.backend is be


def test_capabilities_report_implemented_operations():
    """capabilities() lists what a backend really implements, without calling anything."""
    caps = get_backend("dummy").capabilities()
    assert "get_metadata" in caps["data"] and "import_data" in caps["data"]
    assert "scan_snr" in caps["calibrate"] and "a_priori" in caps["calibrate"]
    assert "clean" in caps["image"] and "selfcal" in caps["image"]


def test_supports_distinguishes_implemented_from_stub():
    """A partial backend reports only its own overrides as supported."""

    class PartialDataOps(DataOps):
        def get_metadata(self, project_code, source_names, observatory):
            return ObsMetadata(project_code=project_code)

    class PartialBackend(Backend):
        kind = "partial"
        data_ops = PartialDataOps

    be = PartialBackend()
    assert be.supports("data", "get_metadata")
    assert not be.supports("data", "import_data")
    assert not be.supports("calibrate", "bandpass")
    assert be.capabilities()["calibrate"] == []


def test_unimplemented_operation_raises_with_component_and_name():
    """The stub error names the backend, the component and the operation."""

    class BareBackend(Backend):
        kind = "bare"

    with pytest.raises(NotImplementedError, match=r"'bare'.*calibrate\.bandpass"):
        BareBackend().calibrate.bandpass("P", "FF", "EF")


def test_flag_modes_delegate_to_run():
    """Implementing FlagOps.run alone provides every named flagging mode."""
    be = get_backend("dummy")
    assert be.flag.autocorr("P") == be.flag.run("P", "autocorr")
    assert 0.0 <= be.flag.edges("P", edge_fraction=0.1) <= 1.0
    assert 0.0 <= be.flag.aoflagger("P", field="3C345") <= 1.0


# -- per-scan SNR survey --

def test_dummy_scan_snr_shape_and_refant_masking():
    be = get_backend("dummy")
    survey = be.calibrate.scan_snr("RSM07", "3C345", refant="EF")
    assert isinstance(survey, ScanSNRSurvey)
    assert survey.polarizations == ["RR", "LL"]
    assert survey.refant == "EF"
    for pol in survey.polarizations:
        matrix = survey.matrix(pol)
        assert len(matrix) == len(survey.scan_numbers)
        assert all(len(row) == len(survey.antennas) for row in matrix)
    # The reference antenna's own solutions are a sentinel, not a measurement.
    refant_column = survey.antennas.index("EF")
    assert all(math.isnan(row[refant_column]) for row in survey.matrix("RR"))
    assert survey.values_for(antenna="EF") == []


def test_scan_snr_survey_ranking_helpers():
    survey = ScanSNRSurvey(
        project_code="P", scan_numbers=[1, 2], scan_sources=["A", "A"], antennas=["EF", "WB", "XX"],
        snr={"RR": [[100.0, 10.0, float("nan")], [200.0, 20.0, float("nan")]]}, refant="")
    assert survey.median_snr(antenna="EF") == 150.0
    assert survey.median_snr(scan_number=2) == 110.0
    assert survey.rank_scans()[0][0] == 2                 # scan 2 is the better one
    assert [a for a, _ in survey.rank_antennas()] == ["EF", "WB"]
    assert survey.dead_antennas() == ["XX"]               # no solutions at all
    assert math.isnan(survey.median_snr(antenna="XX"))


def test_scan_snr_survey_rejects_unknown_polarization():
    survey = ScanSNRSurvey(project_code="P", antennas=["EF"], snr={"RR": [[5.0]]},
                           scan_numbers=[1], scan_sources=["A"])
    with pytest.raises(KeyError, match="LL"):
        survey.matrix("LL")


# -- metadata enrichment --

def test_metadata_derives_array_geometry():
    """Baseline lengths / resolution come from the ITRF positions in the metadata."""
    meta = get_backend("dummy").data.get_metadata("RSM07", ["3C345"], "EVN")
    assert meta.max_baseline > meta.min_baseline > 0.0
    assert meta.max_baseline > 1e6                        # intercontinental EVN baselines
    assert 0.0 < meta.resolution_mas < meta.largest_angular_scale_mas
    assert len(meta.baseline_lengths()) == meta.n_antennas * (meta.n_antennas - 1) // 2
    assert all(a.mount == "ALT-AZ" and a.n_scans > 0 for a in meta.observed_antennas)


def test_metadata_source_helpers():
    meta = get_backend("dummy").data.get_metadata("RSM07", ["3C345", "J1848+3219"], "EVN")
    assert len(meta.scans_for_source("3C345")) == 3
    assert meta.time_on_source("3C345") == pytest.approx(900.0)
    matrix = meta.antenna_scan_matrix()
    assert set(matrix) == set(meta.antennas)
    assert all(len(row) == meta.n_scans for row in matrix.values())


def test_central_channel_selection():
    casa = pytest.importorskip("vlbipy.backends.casa")
    assert casa.central_channel_selection(32, 0.8) == "*:3~28"
    assert casa.central_channel_selection(64, 0.8) == "*:6~57"
    assert casa.central_channel_selection(32, 1.0) == "*"    # nothing trimmed
    assert casa.central_channel_selection(2, 0.8) == "*"     # too few channels to trim


def test_central_channel_selection_single_channel_uses_everything():
    """A one-channel subband has nothing to trim: every solve must still use it."""
    casa = pytest.importorskip("vlbipy.backends.casa")
    for fraction in (0.5, 0.7, 0.8, 1.0):
        assert casa.central_channel_selection(1, fraction) == "*"
        assert casa.central_channel_selection(3, fraction) == "*"


def _flag_ops():
    """A CasaFlagOps instance for the pure selection logic (no MS or CASA session needed)."""
    casa = pytest.importorskip("vlbipy.backends.casa")
    return casa.CasaFlagOps.__new__(casa.CasaFlagOps)


def test_edge_spw_selection_symmetric_and_asymmetric():
    ops = _flag_ops()
    assert ops._edge_spw_selection("p", n_channels=64, edge_channels=6) == "*:0~5;58~63"
    # Per-antenna trims are asymmetric: only what that station actually needs.
    assert ops._edge_spw_selection("p", n_channels=64, left=4, right=2) == "*:0~3;62~63"
    assert ops._edge_spw_selection("p", n_channels=64, left=0, right=3) == "*:61~63"
    assert ops._edge_spw_selection("p", n_channels=64, left=2, right=0) == "*:0~1"


def test_edge_spw_selection_never_flags_everything():
    """No trim must yield an empty selection, never one that would match all the data."""
    ops = _flag_ops()
    assert ops._edge_spw_selection("p", n_channels=64, left=0, right=0) == ""
    assert ops._edge_spw_selection("p", n_channels=64, edge_channels=0, edge_fraction=0.0) == ""
    # Single- and two-channel subbands have no edge that can be trimmed.
    assert ops._edge_spw_selection("p", n_channels=1, edge_channels=6) == ""
    assert ops._edge_spw_selection("p", n_channels=1, left=1, right=1) == ""
    assert ops._edge_spw_selection("p", n_channels=2, edge_fraction=0.1) == ""


def test_edge_trim_takes_the_consensus_not_the_widest():
    """Two of the three indicators must agree before bandwidth is given up."""
    import numpy as np
    ops = _flag_ops()
    rng = np.random.default_rng(7)
    n = 64
    # Amplitude rolls off over 4 channels, phase scatter claims 8, the solver claims none.
    amp = 1.0 + rng.normal(0.0, 0.01, n); amp[:4] = 0.01; amp[-4:] = 0.01
    phase = 0.05 + rng.normal(0.0, 0.005, n); phase[:8] = 5.0; phase[-8:] = 5.0
    solved = np.zeros(n)
    left, right = ops._edge_trim(amp, phase, solved, threshold=6.0, max_trim=16, n_channels=n)
    assert (left, right) == (4, 4)      # the median vote, not the widest (8)


def test_single_channel_subbands_survive_every_channel_decision():
    """A one-channel spectral window: use the channel everywhere, flag nothing."""
    from vlbipy.models import FreqSetup
    casa = pytest.importorskip("vlbipy.backends.casa")
    from vlbipy.statistics import find_flat_range
    freq = FreqSetup(ref_freq=1.6e9, n_subbands=4, n_channels=1, channel_width=1.6e7,
                     polarizations=["RR", "LL"])
    assert len(freq.frequencies_ghz(0)) == 1          # a usable frequency axis
    # Every solve uses the single channel...
    assert casa.central_channel_selection(freq.n_channels, 0.8) == "*"
    # ...and nothing is ever trimmed away from it.
    ops = _flag_ops()
    assert ops._edge_spw_selection("p", n_channels=freq.n_channels, edge_channels=2) == ""
    assert find_flat_range([1.0], threshold=6.0) == (0, 0)


def test_edge_trim_respects_max_trim():
    import numpy as np
    ops = _flag_ops()
    rng = np.random.default_rng(3)
    n = 32
    amp = 1.0 + rng.normal(0.0, 0.01, n); amp[:12] = 0.01
    phase = 0.05 + rng.normal(0.0, 0.005, n); phase[:12] = 5.0
    solved = np.zeros(n); solved[:12] = 1.0
    left, _ = ops._edge_trim(amp, phase, solved, threshold=6.0, max_trim=4, n_channels=n)
    assert left == 4


# -- CASA log collection --

def test_collect_casa_logs_moves_and_never_overwrites(tmp_path):
    """Stray casa*.log files are swept up; a name clash is kept, not clobbered."""
    from vlbipy.tools import collect_casa_logs
    source = tmp_path / "cwd"
    source.mkdir()
    (source / "casa-20260101-120000.log").write_text("first")
    (source / "casa-20260101-120001.log").write_text("second")
    (source / "keep-me.txt").write_text("untouched")
    logs = tmp_path / "proj" / "logs"
    logs.mkdir(parents=True)
    (logs / "casa-20260101-120000.log").write_text("already here")

    moved = collect_casa_logs(logs, search_dirs=[source])
    assert len(moved) == 2
    assert not list(source.glob("casa*.log"))          # cwd is left clean
    assert (source / "keep-me.txt").is_file()          # unrelated files untouched
    assert (logs / "casa-20260101-120000.log").read_text() == "already here"
    assert (logs / "casa-20260101-120000.1.log").read_text() == "first"


def test_collect_casa_logs_skips_its_own_directory(tmp_path):
    """Sweeping the destination itself must not move files onto themselves."""
    from vlbipy.tools import collect_casa_logs
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "casa-20260101-120000.log").write_text("x")
    assert collect_casa_logs(logs, search_dirs=[logs]) == []
    assert (logs / "casa-20260101-120000.log").is_file()


def test_write_callib_declares_each_table(tmp_path):
    """The cal library is the record of what was applied; it must be complete."""
    pytest.importorskip("casatools")
    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable

    backend = CasaBackend(work_dir=str(tmp_path))
    tables = [
        CalTable("tsys", path="/c/rsm07.tsys", interp="nearest"),
        CalTable("bpass", path="/c/rsm07.bpass", interp="nearest,nearest"),
        CalTable("mbd", path="/c/rsm07.mbd", interp="linear",
                 gainfield="J1848+3219", spwmap=[0, 0, 0, 0]),
    ]
    path = backend.calibrate.write_callib("rsm07", tables)
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    assert len(lines) == 3
    # calwt defaults to True, matching CASA: the weights are calibrated with the data.
    assert lines[0] == "caltable='/c/rsm07.tsys' calwt=True tinterp='nearest'"
    # A two-part interp splits into time and frequency interpolation.
    assert "tinterp='nearest' finterp='nearest'" in lines[1]
    # fldmap is how the phase calibrator's solutions reach the target.
    assert "fldmap='J1848+3219'" in lines[2] and "spwmap=[0, 0, 0, 0]" in lines[2]
    # Field-independent tables must carry no fldmap: selecting a field on them
    # matches zero rows and applycal fails.
    assert "fldmap" not in lines[0]


def test_prior_callib_writes_one_file_per_solve(tmp_path):
    """Each solve declares its priors in a file named after the table it produces."""
    pytest.importorskip("casatools")
    from pathlib import Path as P

    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable

    backend = CasaBackend(work_dir=str(tmp_path))
    priors = [CalTable("tsys", path="/c/rsm07.tsys", interp="nearest")]
    params = backend.calibrate._prior_callib("rsm07", priors, P("/c/rsm07.mbd"))
    assert params["docallib"] is True
    written = P(params["callib"])
    assert written.is_file() and written.name == "rsm07.mbd.txt"
    assert "rsm07.tsys" in written.read_text()
    # No priors means no cal library and no docallib: the solve runs on raw data.
    assert backend.calibrate._prior_callib("rsm07", [], P("/c/rsm07.sbd")) == {}


def test_write_callib_honours_calwt_per_table(tmp_path):
    """calwt is per table and defaults to True; False must be written through."""
    pytest.importorskip("casatools")
    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable

    backend = CasaBackend(work_dir=str(tmp_path))
    path = backend.calibrate.write_callib("rsm07", [
        CalTable("tsys", path="/c/rsm07.tsys"),
        CalTable("mbd", path="/c/rsm07.mbd", calwt=False),
    ])
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    assert "calwt=True" in lines[0]
    assert "calwt=False" in lines[1]
    # It survives the round-trip through the persisted chain.
    assert CalTable.from_dict(CalTable("mbd", calwt=False).to_dict()).calwt is False
    assert CalTable.from_dict({"cal_type": "tsys"}).calwt is True
