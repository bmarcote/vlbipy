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


def test_prior_callib_returns_explicit_params_by_default(tmp_path):
    """By default prior tables are passed as aligned parallel parameter lists."""
    pytest.importorskip("casatools")
    from pathlib import Path as P

    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable

    backend = CasaBackend(work_dir=str(tmp_path))
    priors = [
        CalTable("tsys", path="/c/rsm07.tsys", interp="nearest"),
        CalTable("bpass", path="/c/rsm07.bpass", interp="nearest,nearest", calwt=False),
    ]
    params = backend.calibrate._prior_callib("rsm07", priors, P("/c/rsm07.mbd"), field="3C286")
    assert "docallib" not in params
    assert params["gaintable"] == ["/c/rsm07.tsys", "/c/rsm07.bpass"]
    assert params["gainfield"] == ["", ""]  # field-independent/no fldmap for solves
    assert params["interp"] == ["nearest", "nearest,nearest"]
    assert params["spwmap"] == [[], []]
    # Solving tasks do not accept calwt.
    assert "calwt" not in params
    # No priors means no apply parameters: the solve runs on raw data.
    assert backend.calibrate._prior_callib("rsm07", [], P("/c/rsm07.sbd")) == {}


def test_prior_callib_callib_path_writes_one_file_per_solve(tmp_path):
    """Optional callib=True writes one cal-library file per solve, named after the table."""
    pytest.importorskip("casatools")
    from pathlib import Path as P

    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable

    backend = CasaBackend(work_dir=str(tmp_path))
    priors = [CalTable("tsys", path="/c/rsm07.tsys", interp="nearest")]
    params = backend.calibrate._prior_callib("rsm07", priors, P("/c/rsm07.mbd"),
                                                field="3C286", callib=True)
    assert params["docallib"] is True
    written = P(params["callib"])
    assert written.is_file() and written.name == "rsm07.mbd.txt"
    assert "rsm07.tsys" in written.read_text()


def test_write_callib_resolves_per_field_for_phase_referencing(tmp_path):
    """Callib mode resolves fldmap per field: nearest for self, phasecal for target."""
    pytest.importorskip("casatools")
    from vlbipy.backends.casa import CasaBackend
    from vlbipy.models import CalTable

    backend = CasaBackend(work_dir=str(tmp_path))
    tables = [
        CalTable("tsys", path="/c/x.tsys", interp="nearest", field=""),
        CalTable("mbd", path="/c/x.mbd", interp="linear",
                 field="3C286,J1048+7143", gainfield="J1048+7143"),
    ]
    target_lines = [ln for ln in backend.calibrate.write_callib("p", tables, field="R20181030")
                    .read_text().splitlines() if not ln.startswith("#")]
    # Target field uses the stored phase-calibrator mapping on the MBD table.
    assert "fldmap='J1048+7143'" in target_lines[1]
    # Field-independent table carries no fldmap.
    assert "fldmap" not in target_lines[0]

    self_lines = [ln for ln in backend.calibrate.write_callib("p", tables, field="3C286")
                  .read_text().splitlines() if not ln.startswith("#")]
    # A field contained in the table uses its own (nearest) solutions.
    assert "fldmap='nearest'" in self_lines[1]


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


def _cal_ops():
    """A CasaCalibrationOps instance for pure helper logic tests."""
    casa = pytest.importorskip("vlbipy.backends.casa")
    return casa.CasaCalibrationOps.__new__(casa.CasaCalibrationOps)


def test_resolve_table_gainfield_respects_field_independence_and_self_solve():
    """Field-independent tables stay empty; self-solved fields use nearest."""
    ops = _cal_ops()
    tsys = CalTable("tsys", path="/c/x.tsys", field="3C286,J1048+7143")
    mbd = CalTable("mbd", path="/c/x.mbd", field="3C286,J1048+7143",
                   gainfield="J1048+7143")

    assert ops._resolve_table_gainfield(tsys, "3C286") == ""
    assert ops._resolve_table_gainfield(tsys, "") == ""
    assert ops._resolve_table_gainfield(mbd, "3C286") == "nearest"
    assert ops._resolve_table_gainfield(mbd, "J1048+7143") == "nearest"
    # A target/non-solved field uses the stored phase-calibrator mapping.
    assert ops._resolve_table_gainfield(mbd, "R20181030") == "J1048+7143"
    # Empty apply field falls back to the table's stored mapping.
    assert ops._resolve_table_gainfield(mbd, "") == "J1048+7143"


def test_compile_apply_params_explicit_lists_aligned():
    """Explicit params preserve list alignment and per-table values."""
    ops = _cal_ops()
    tables = [
        CalTable("tsys", path="/c/x.tsys", interp="nearest", field="3C286"),
        CalTable("mbd", path="/c/x.mbd", interp="linear", field="3C286",
                 gainfield="J1048+7143", spwmap=[0, 0, 0, 0], calwt=False),
    ]
    params = ops._compile_apply_params("p", tables, "R20181030", include_calwt=True)
    assert params["gaintable"] == ["/c/x.tsys", "/c/x.mbd"]
    assert params["gainfield"] == ["", "J1048+7143"]  # mbd target uses phase cal
    assert params["interp"] == ["nearest", "linear"]
    assert params["spwmap"] == [[], [0, 0, 0, 0]]
    assert params["calwt"] == [True, False]
    assert "docallib" not in params


def test_compile_apply_params_omits_calwt_for_solve():
    """Solving tasks do not accept calwt; the helper leaves it out."""
    ops = _cal_ops()
    tables = [CalTable("tsys", path="/c/x.tsys", interp="nearest", field="")]
    params = ops._compile_apply_params("p", tables, "3C286", include_calwt=False)
    assert "calwt" not in params
    assert params["gaintable"] == ["/c/x.tsys"]


def test_compile_apply_params_callib_returns_docallib():
    """Optional callib mode writes a cal-library file and returns docallib/callib."""
    ops = _cal_ops()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ops._backend = type("B", (), {"work_dir": __import__("pathlib").Path(tmp)})()
        tables = [CalTable("tsys", path="/c/x.tsys", interp="nearest", field="")]
        params = ops._compile_apply_params("p", tables, "3C286", callib=True,
                                           filename="callibs/solve.txt")
        assert params == {"docallib": True, "callib": __import__("os").path.join(tmp, "callibs", "solve.txt")}


def test_apply_loops_over_observed_sources_when_field_is_empty(tmp_path):
    """apply(field='') calls applycal once per observed source field."""
    casa = pytest.importorskip("vlbipy.backends.casa")
    from vlbipy.models import CalTable, ObsMetadata, Scan

    class FakeBackend:
        def __init__(self, scans):
            self.work_dir = tmp_path
            self.data = self
            self.tasks = self
            self.scans = scans
            self.calls = []

        def ms_path(self, code):
            return self.work_dir / f"{code}.ms"

        def caldir(self):
            return self.work_dir / "caltables"

        def get_metadata(self, *args, **kwargs):
            return ObsMetadata(project_code="p", scans=self.scans)

        def applycal(self, **kwargs):
            self.calls.append(kwargs)

    ops = casa.CasaCalibrationOps.__new__(casa.CasaCalibrationOps)
    ops._backend = FakeBackend([
        Scan(scan_number=1, source="3C286", time_start=0.0, time_end=100.0),
        Scan(scan_number=2, source="R20181030", time_start=100.0, time_end=200.0),
    ])
    (tmp_path / "p.ms").mkdir()
    caldir = tmp_path / "caltables"
    caldir.mkdir()
    tsys = CalTable("tsys", path=str(caldir / "x.tsys"), interp="nearest", field="")
    mbd = CalTable("mbd", path=str(caldir / "x.mbd"), interp="linear",
                   field="3C286", gainfield="3C286")
    for table in (tsys, mbd):
        (caldir / table.path).mkdir()

    ops.apply("p", "", [tsys, mbd], callib=False)
    assert len(ops.backend.calls) == 2
    fields = [c["field"] for c in ops.backend.calls]
    assert sorted(fields) == ["3C286", "R20181030"]
    # calibrators use nearest, targets use the stored mapping.
    calibrator_call = next(c for c in ops.backend.calls if c["field"] == "3C286")
    assert calibrator_call["gainfield"] == ["", "nearest"]
    target_call = next(c for c in ops.backend.calls if c["field"] == "R20181030")
    assert target_call["gainfield"] == ["", "3C286"]
    # applycal always receives calwt; explicit mode has no docallib.
    for call in ops.backend.calls:
        assert call["calwt"] == [True, True]
        assert "docallib" not in call


def test_self_solved_field_takes_its_own_scan_solution():
    """A field corrected with its own solutions uses them 'nearest' in time, not interpolated.

    Linear interpolation between scans of the same field flags an antenna with a single
    good solution there (EM163: HH and IB on the one fringe-finder scan with every antenna).
    """
    ops = _cal_ops()
    mbd = CalTable("mbd", path="/c/x.mbd", interp="linear,linear", field="3C286,J1048+7143",
                   gainfield="J1048+7143", spwmap=[0, 0, 0, 0], calwt=False)
    own = ops._compile_apply_params("p", [mbd], "3C286")
    assert (own["gainfield"], own["interp"]) == (["nearest"], ["nearest,linear"])
    target = ops._compile_apply_params("p", [mbd], "R20181030")
    assert (target["gainfield"], target["interp"]) == (["J1048+7143"], ["linear,linear"])


def test_tables_for_field_respects_apply_to():
    """A table restricted with apply_to must not be applied to other fields."""
    from vlbipy.backends.casa import CasaCalibrationOps
    from vlbipy.models import CalTable
    everywhere = CalTable(cal_type="selfamp_FF", path="ff.G", field="FF")
    restricted = CalTable(cal_type="selfphase_PC", path="pc.G", field="PC", apply_to="PC,TARGET,CHECK")
    pick = CasaCalibrationOps._tables_for_field
    assert [t.cal_type for t in pick([everywhere, restricted], "FF")] == ["selfamp_FF"]
    assert [t.cal_type for t in pick([everywhere, restricted], "TARGET")] == ["selfamp_FF", "selfphase_PC"]
    assert len(pick([everywhere, restricted], "")) == 2


def _casa_ops_without_backend(cls):
    """An ops object with no backend, for the pure-numpy helpers."""
    return cls.__new__(cls)


def test_spike_departures_reports_bins_and_their_size():
    from vlbipy.backends.casa import CasaFlagOps
    import numpy as np
    rng = np.random.default_rng(3)
    amplitude = 1.0 + 0.01 * rng.standard_normal((80, 16))
    amplitude[[10, 40]] *= 0.5                       # two dropouts of 50%
    ops = _casa_ops_without_backend(CasaFlagOps)
    bins, relative = ops._spike_departures(amplitude, np.isfinite(amplitude), 5.0, 9)
    assert set(bins) == {10, 40}
    assert relative.shape == (80,) and abs(relative[10] + 0.5) < 0.05
    assert list(ops._spike_bins(amplitude, np.isfinite(amplitude), 5.0, 9)) == list(bins)


def test_scalar_bandpass_is_normalised_per_antenna(tmp_path, monkeypatch):
    """Each antenna's subbands are levelled around 1; only a subband a factor 2 off is flagged."""
    import numpy as np
    from vlbipy.backends.casa import CasaCalibrationOps

    class FakeTable:
        store = {"CPARAM": None, "FLAG": None, "ANTENNA1": None, "SPECTRAL_WINDOW_ID": None}

        def open(self, path, nomodify=True):
            return True

        def getcol(self, name):
            return FakeTable.store[name].copy()

        def putcol(self, name, value):
            FakeTable.store[name] = np.array(value)

        def close(self):
            pass

    antennas = np.repeat([0, 1], 4)                                   # two antennas, four subbands each
    amplitude = np.array([0.40, 0.44, 0.36, 0.40, 1.2, 1.3, 1.25, 0.3])
    FakeTable.store.update(CPARAM=np.tile(amplitude.astype(complex), (2, 1, 1)), FLAG=np.zeros((2, 1, 8), bool),
                           ANTENNA1=antennas, SPECTRAL_WINDOW_ID=np.tile(np.arange(4), 2))
    ops = CasaCalibrationOps.__new__(CasaCalibrationOps)
    ops._backend = type("B", (), {"tools": type("T", (), {"table": staticmethod(FakeTable)})()})()
    monkeypatch.setattr(ops, "_table_antenna_names", lambda path: ["AA", "BB"])
    dropped = ops._normalise_per_antenna(tmp_path, max_factor=2.0)
    gains, flags = np.abs(FakeTable.store["CPARAM"][0, 0]), FakeTable.store["FLAG"][0, 0]
    assert np.allclose(gains[:4], [1.0, 1.1, 0.9, 1.0])               # antenna level 0.40 removed, steps kept
    assert not flags[:7].any() and flags[7]                            # 0.3 against ~1.22 is a factor 4: flagged
    assert dropped == ["BB spw 3 pol 0", "BB spw 3 pol 1"]
