"""Tests for the download / import / inspection phase.

Covers: FITS-IDI file discovery and inspection (synthetic files built with
astropy), the EVN archive download flow (network mocked at the vlbipy.tools
boundary), and the import_data namespace orchestration (fake file-requiring
backend). No CASA and no network are needed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from vlbipy import fitsidi, load_config, tools
from vlbipy.models import Antenna, FreqSetup, ObsMetadata
from vlbipy.observation import Observation
from vlbipy.observatories.evn import EVNObservatory
from vlbipy.backends.base import Backend, DataOps

PROJECT = "ee018a"


def make_fitsidi(path: Path, sources=("3C345", "J1848+3219"), antennas=("EF", "O8", "TR")) -> Path:
    """Write a minimal synthetic FITS-IDI file with the tables vlbipy inspects."""
    common = {"OBSCODE": PROJECT.upper(), "RDATE": "2024-03-01", "REF_FREQ": 4.926e9,
              "CH_WIDTH": 500e3, "NO_CHAN": 32, "NO_BAND": 8, "NO_STKD": 2, "STK_1": -1}
    ag_cols = fits.ColDefs([
        fits.Column(name="ANNAME", format="8A", array=np.array(antennas)),
        fits.Column(name="STABXYZ", format="3D", unit="METERS",
                    array=np.arange(len(antennas) * 3, dtype=float).reshape(len(antennas), 3)),
        fits.Column(name="DIAMETER", format="1E", unit="METERS",
                    array=np.full(len(antennas), 25.0)),
    ])
    array_geometry = fits.BinTableHDU.from_columns(ag_cols, name="ARRAY_GEOMETRY")
    src_cols = fits.ColDefs([
        fits.Column(name="SOURCE", format="16A", array=np.array(sources)),
        fits.Column(name="RAEPO", format="1D", unit="DEGREES",
                    array=np.linspace(250.0, 280.0, len(sources))),
        fits.Column(name="DECEPO", format="1D", unit="DEGREES",
                    array=np.linspace(30.0, 40.0, len(sources))),
        fits.Column(name="CALCODE", format="4A", array=np.array(["V"] * len(sources))),
    ])
    source_table = fits.BinTableHDU.from_columns(src_cols, name="SOURCE")
    freq_cols = fits.ColDefs([
        fits.Column(name="BANDFREQ", format="8D", array=np.zeros((1, 8))),
        fits.Column(name="TOTAL_BANDWIDTH", format="8E", array=np.full((1, 8), 16e6)),
    ])
    frequency = fits.BinTableHDU.from_columns(freq_cols, name="FREQUENCY")
    for hdu in (array_geometry, source_table, frequency):
        hdu.header.update(common)
    fits.HDUList([fits.PrimaryHDU(), array_geometry, source_table, frequency]).writeto(path)
    return path


@pytest.fixture
def idi_dir(tmp_path):
    """Directory with two synthetic FITS-IDI files following the EVN naming scheme."""
    for n in (1, 2):
        make_fitsidi(tmp_path / f"{PROJECT}_1_1.IDI{n}")
    return tmp_path


# -- FITS-IDI discovery and inspection --

def test_find_fitsidi_files_natural_order(tmp_path):
    for n in (10, 2, 1):
        make_fitsidi(tmp_path / f"{PROJECT}_1_1.IDI{n}")
    (tmp_path / f"{PROJECT}.checksum").write_text("dummy")
    files = fitsidi.find_fitsidi_files(tmp_path, PROJECT)
    assert [Path(f).name for f in files] == [f"{PROJECT}_1_1.IDI{n}" for n in (1, 2, 10)]


def test_find_fitsidi_files_empty(tmp_path):
    assert fitsidi.find_fitsidi_files(tmp_path, PROJECT) == []


def test_inspect_fitsidi(idi_dir):
    files = fitsidi.find_fitsidi_files(idi_dir, PROJECT)
    meta = fitsidi.inspect_fitsidi(files, project_code=PROJECT)
    assert meta.project_code == PROJECT
    assert meta.obs_date == dt.date(2024, 3, 1)
    assert set(meta.antennas) == {"EF", "O8", "TR"}
    assert meta.antennas["EF"].diameter == 25.0
    assert meta.source_names == ["3C345", "J1848+3219"]
    assert meta.source_coords["3C345"] == (250.0, 30.0)
    assert meta.freq_setup.ref_freq == 4.926e9
    assert meta.freq_setup.n_subbands == 8
    assert meta.freq_setup.n_channels == 32
    assert meta.freq_setup.total_bandwidth == pytest.approx(128e6)
    assert meta.scans == []


def test_obs_date_and_tsys_helpers(idi_dir):
    files = fitsidi.find_fitsidi_files(idi_dir, PROJECT)
    assert fitsidi.get_obs_date(files[0]) == dt.date(2024, 3, 1)
    assert fitsidi.get_ref_freq(files[0]) == 4.926e9
    assert fitsidi.has_tsys(files[0]) is False
    assert fitsidi.has_gain_curve(files[0]) is False


# -- EVN archive download (network mocked) --

@pytest.fixture
def fake_archive(monkeypatch, tmp_path):
    """Mock vlbipy.tools network functions with an in-memory EVN archive."""
    contents = {f"{PROJECT}_1_1.IDI1": b"idi-one", f"{PROJECT}_1_1.IDI2": b"idi-two"}
    checksum = "".join(f"{hashlib.md5(data).hexdigest()}  {name}\n" for name, data in contents.items())
    archive_files = dict(contents)
    archive_files[f"{PROJECT}.checksum"] = checksum.encode()
    base = f"https://archive.jive.eu/exp/{PROJECT.upper()}_240301"
    listing = "".join(f'<a href="{name}">{name}</a>\n' for name in archive_files)
    pages = {"https://archive.jive.eu/scripts/listarch.php":
             f'<a href="arch.php?exp={PROJECT.upper()}_240301">x</a>',
             f"{base}/fits/": listing}
    downloads = []

    def fake_fetch(url, username=None, password=None, timeout=120.0):
        if url in pages:
            return pages[url]
        raise ConnectionError(f"HTTP 404 fetching {url}")

    def fake_download(url, dest, username=None, password=None, timeout=600.0):
        name = url.rsplit("/", 1)[-1]
        if url.startswith(f"{base}/fits/") and name in archive_files:
            Path(dest).write_bytes(archive_files[name])
            downloads.append(name)
            return Path(dest)
        raise ConnectionError(f"HTTP 404 downloading {url}")

    monkeypatch.setattr(tools, "fetch_url_text", fake_fetch)
    monkeypatch.setattr(tools, "download_file", fake_download)
    return {"downloads": downloads, "dir": tmp_path}


def test_evn_download(fake_archive):
    evn = EVNObservatory()
    files = evn.download_data(PROJECT, str(fake_archive["dir"]))
    names = [Path(f).name for f in files]
    assert names == [f"{PROJECT}_1_1.IDI1", f"{PROJECT}_1_1.IDI2"]
    assert (fake_archive["dir"] / f"{PROJECT}_1_1.IDI1").read_bytes() == b"idi-one"
    # obsdate was auto-resolved from the archive index (no explicit obsdate given).
    assert f"{PROJECT}_1_1.IDI1" in fake_archive["downloads"]


def test_evn_download_detects_corruption(fake_archive, monkeypatch):
    evn = EVNObservatory()
    # Pre-place a corrupted file: the checksum pass must re-download it.
    (fake_archive["dir"] / f"{PROJECT}_1_1.IDI1").write_bytes(b"corrupted")
    files = evn.download_data(PROJECT, str(fake_archive["dir"]), obsdate="240301")
    assert (fake_archive["dir"] / f"{PROJECT}_1_1.IDI1").read_bytes() == b"idi-one"
    assert len(files) == 2


def test_evn_resolve_obsdate_from_idi(idi_dir):
    evn = EVNObservatory()
    assert evn.resolve_obsdate(PROJECT, str(idi_dir)) == "240301"


def test_evn_antab_and_flag_lookup(tmp_path):
    evn = EVNObservatory()
    assert evn.get_antab_file(PROJECT, str(tmp_path)) is None
    (tmp_path / f"{PROJECT}.antab").write_text("x")
    (tmp_path / f"{PROJECT}.uvflg").write_text("x")
    assert Path(evn.get_antab_file(PROJECT, str(tmp_path))).name == f"{PROJECT}.antab"
    assert Path(evn.get_flag_file(PROJECT, str(tmp_path))).name == f"{PROJECT}.uvflg"
    (tmp_path / f"{PROJECT}.flag").write_text("x")
    assert Path(evn.get_flag_file(PROJECT, str(tmp_path))).name == f"{PROJECT}.flag"


# -- import_data namespace orchestration --

class RecordingDataOps(DataOps):
    """Records import calls on the owning backend instead of touching any data."""

    def import_data(self, project_code, source_names, *, scan_gap=15, files=None, delete=False, **kw):
        self.backend.import_calls.append({"files": list(files or []), "scan_gap": scan_gap,
                                          "delete": delete})

    def get_metadata(self, project_code, source_names, observatory):
        return ObsMetadata(project_code=project_code, antennas={"EF": Antenna(name="EF", observed=True)},
                           freq_setup=FreqSetup(ref_freq=5e9, total_bandwidth=128e6, n_subbands=8,
                                                n_channels=32),
                           source_names=["3C345"], source_coords={"3C345": (250.0, 40.0)})


class RecordingBackend(Backend):
    """File-requiring fake backend that records import calls."""

    kind = "recording"
    requires_data_files = True
    data_ops = RecordingDataOps

    def __init__(self):
        self.import_calls = []
        super().__init__(".")


def _make_obs(tmp_path, backend):
    cfg = load_config({"global": {"work_dir": str(tmp_path)}, "sources": {"targets": ["3C345"]}})
    return Observation(PROJECT, cfg, backend=backend)


def test_import_orchestration_with_local_files(tmp_path):
    make_fitsidi(tmp_path / f"{PROJECT}_1_1.IDI1")
    backend = RecordingBackend()
    obs = _make_obs(tmp_path, backend)
    meta = obs.import_data()
    assert len(backend.import_calls) == 1
    assert [Path(f).name for f in backend.import_calls[0]["files"]] == [f"{PROJECT}_1_1.IDI1"]
    assert meta.source_names == ["3C345"]
    # Coordinates from the data were propagated onto the declared target source.
    assert obs.sources["3C345"].coordinates is not None
    assert obs.state.status("import_data") == "done"


def test_import_orchestration_downloads_when_no_files(tmp_path, monkeypatch):
    backend = RecordingBackend()
    obs = _make_obs(tmp_path, backend)
    called = {}

    def fake_download(project_code, directory, **kwargs):
        called.update(kwargs, project=project_code)
        idi = Path(directory) / f"{PROJECT}_1_1.IDI1"
        make_fitsidi(idi)
        return [str(idi)]

    monkeypatch.setattr(obs._observatory_handler, "download_data", fake_download)
    obs.import_data()
    assert called["project"] == PROJECT
    assert backend.import_calls[0]["files"] == [str(tmp_path / f"{PROJECT}_1_1.IDI1")]


def test_import_orchestration_explicit_glob(tmp_path):
    for n in (2, 1):
        make_fitsidi(tmp_path / f"{PROJECT}_1_1.IDI{n}")
    backend = RecordingBackend()
    obs = _make_obs(tmp_path, backend)
    obs.import_data.from_fitsidi(str(tmp_path / f"{PROJECT}_1_1.IDI*"))
    names = [Path(f).name for f in backend.import_calls[0]["files"]]
    assert names == [f"{PROJECT}_1_1.IDI1", f"{PROJECT}_1_1.IDI2"]


def test_import_orchestration_missing_glob_raises(tmp_path):
    obs = _make_obs(tmp_path, RecordingBackend())
    with pytest.raises(FileNotFoundError, match="no files match"):
        obs.import_data(files=str(tmp_path / "nothing.IDI*"))


# -- tools --

def test_tools_mjd_roundtrip():
    date = dt.datetime(2024, 3, 1, 12, 0, 0)
    assert tools.mjd2datetime(tools.datetime2mjd(date)) == date


def test_tools_gunzip(tmp_path):
    import gzip
    gz = tmp_path / "file.txt.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(b"hello")
    out = tools.gunzip(gz)
    assert out.read_bytes() == b"hello"
    assert not gz.exists()


def test_tools_md5sum(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"vlbipy")
    assert tools.md5sum(f) == hashlib.md5(b"vlbipy").hexdigest()


# -- resetting an already-imported dataset --

class ResettingDataOps(RecordingDataOps):
    """Reports the data as already imported and records reset calls."""

    def is_imported(self, project_code):
        return True

    def reset_calibration(self, project_code, *, unflag=True, backup_flags=True):
        self.backend.reset_calls.append({"unflag": unflag, "backup_flags": backup_flags})
        return {"flagged_before": 0.5, "flagged_after": 0.0, "backup": "v1"}


class ResettingBackend(RecordingBackend):
    """Backend whose data already exists on disk."""

    kind = "resetting"
    data_ops = ResettingDataOps

    def __init__(self):
        self.reset_calls = []
        super().__init__()


def test_scratch_run_resets_the_existing_dataset(tmp_path):
    """--scratch must clear a previous attempt's flags and corrected data."""
    backend = ResettingBackend()
    obs = _make_obs(tmp_path, backend)
    obs.prepare_run(scratch=True)
    obs.import_data()
    assert backend.reset_calls == [{"unflag": True, "backup_flags": True}]
    assert backend.import_calls == []  # already imported: nothing re-imported


def test_resuming_keeps_the_existing_calibration(tmp_path):
    """Without --scratch the run resumes, so the corrected data must survive.

    Resetting here would silently undo every calibration step already completed.
    """
    backend = ResettingBackend()
    _make_obs(tmp_path, backend).import_data()
    assert backend.reset_calls == []


def test_reset_can_be_disabled(tmp_path):
    """[import].reset_existing = false keeps hand-made flags in place."""
    backend = ResettingBackend()
    cfg = load_config({"global": {"work_dir": str(tmp_path)}, "sources": {"targets": ["3C345"]},
                       "import": {"reset_existing": False}})
    obs = Observation(PROJECT, cfg, backend=backend)
    obs.prepare_run(scratch=True)
    obs.import_data()
    assert backend.reset_calls == []


def test_reset_failure_does_not_block_the_run(tmp_path):
    """A dataset that cannot be reset is reported, not fatal."""
    backend = ResettingBackend()

    def boom(*a, **kw):
        raise RuntimeError("table locked")

    backend.data.reset_calibration = boom
    obs = _make_obs(tmp_path, backend)
    obs.prepare_run(scratch=True)
    assert obs.import_data() is not None      # metadata still loaded
