"""Tests for the dask-ms backend (zarr store metadata/data access).

Builds a small synthetic dask-ms zarr store directly (no measurement set and no
casacore needed) and exercises DaskMsBackend reads plus the import_data
namespace early-skip for already-imported projects.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

daskms = pytest.importorskip("daskms")
import dask  # noqa: E402
import dask.array as da  # noqa: E402
from daskms import Dataset  # noqa: E402
from daskms.experimental.zarr import xds_to_zarr  # noqa: E402

from vlbipy import load_config  # noqa: E402
from vlbipy.backends.dask_ms import DaskMsBackend, _zero_pad_cells  # noqa: E402
from vlbipy.errors import BackendError  # noqa: E402
from vlbipy.models import Stokes  # noqa: E402
from vlbipy.observation import Observation  # noqa: E402

PROJECT = "ee01"
T0 = 5.0e9  # MJD seconds


def _var(array, dims=("row",)):
    array = np.asarray(array)
    return (dims[:array.ndim], da.from_array(array, chunks=array.shape))


@pytest.fixture
def store(tmp_path):
    """Write a synthetic dask-ms zarr store: 2 main partitions + subtables."""
    path = tmp_path / f"{PROJECT}.zarr"
    writes = []
    subtables = {
        "ANTENNA": Dataset({"NAME": _var(["EF", "WB", "JB"]),
                            "STATION": _var(["EFLSBERG", "WSTRBORK", "JODRELL"]),
                            "DISH_DIAMETER": _var([100.0, 25.0, 76.0]),
                            "POSITION": _var(np.arange(9.0).reshape(3, 3), ("row", "xyz"))}),
        # UNOBSERVED is scheduled in the FIELD table but has no data (no main partition):
        # metadata must exclude it.
        "FIELD": Dataset({"NAME": _var(["3C345", "J1848+3219", "UNOBSERVED"]),
                          "PHASE_DIR": _var(np.deg2rad([[[250.745, 39.81]], [[282.0, 32.3]],
                                                        [[10.0, 5.0]]]),
                                            ("row", "poly", "radec"))}),
        "SPECTRAL_WINDOW": Dataset({"CHAN_FREQ": _var(np.array([[4.9e9, 4.902e9], [4.916e9, 4.918e9]]),
                                                      ("row", "chan")),
                                    "CHAN_WIDTH": _var(np.full((2, 2), 2e6), ("row", "chan")),
                                    "TOTAL_BANDWIDTH": _var([16e6, 16e6])}),
        "POLARIZATION": Dataset({"CORR_TYPE": _var(np.array([[int(Stokes.RR), int(Stokes.LL)]]),
                                                   ("row", "corr"))}),
        "OBSERVATION": Dataset({"TIME_RANGE": _var(np.array([[T0, T0 + 3600.0]]), ("row", "range"))}),
    }
    for name, dataset in subtables.items():
        writes.append(xds_to_zarr([dataset], f"{path}::{name}"))
    # Main table: field 0 observed by all three antennas (scan 1), field 1 by EF-WB only (scan 2).
    main = [
        Dataset({"SCAN_NUMBER": _var([1, 1, 1]), "TIME": _var([T0, T0 + 10, T0 + 20]),
                 "ANTENNA1": _var([0, 0, 1]), "ANTENNA2": _var([1, 2, 2]),
                 "DATA": _var(np.ones((3, 2, 2), dtype=np.complex64), ("row", "chan", "corr"))},
                attrs={"FIELD_ID": 0, "DATA_DESC_ID": 0}),
        Dataset({"SCAN_NUMBER": _var([2, 2]), "TIME": _var([T0 + 100, T0 + 130]),
                 "ANTENNA1": _var([0, 0]), "ANTENNA2": _var([1, 1]),
                 "DATA": _var(np.ones((2, 2, 2), dtype=np.complex64), ("row", "chan", "corr"))},
                attrs={"FIELD_ID": 1, "DATA_DESC_ID": 0}),
    ]
    writes.append(xds_to_zarr(main, str(path)))
    dask.compute(*writes)
    return tmp_path


def test_is_imported(store):
    backend = DaskMsBackend(work_dir=str(store))
    assert backend.data.is_imported(PROJECT)
    assert not backend.data.is_imported("other")


def test_get_data_lazy(store):
    backend = DaskMsBackend(work_dir=str(store))
    datasets = backend.data.datasets(PROJECT)
    assert len(datasets) == 2
    assert isinstance(datasets[0].DATA.data, da.Array)  # lazy, not materialized
    assert datasets[0].DATA.shape == (3, 2, 2)
    with pytest.raises(BackendError, match="not found"):
        backend.data.datasets("missing")


def test_get_metadata_from_store(store):
    backend = DaskMsBackend(work_dir=str(store))
    meta = backend.data.get_metadata(PROJECT, [], "EVN")
    assert list(meta.antennas) == ["EF", "WB", "JB"]
    assert meta.antennas["EF"].diameter == 100.0
    assert meta.antennas["EF"].observed and meta.antennas["WB"].observed
    assert meta.antennas["JB"].observed  # participates in scan 1
    # Only sources with data: UNOBSERVED sits in the FIELD table but has no rows.
    assert meta.source_names == ["3C345", "J1848+3219"]
    assert "UNOBSERVED" not in meta.source_coords
    assert meta.source_coords["3C345"][0] == pytest.approx(250.745)
    assert meta.source_coords["3C345"][1] == pytest.approx(39.81)
    assert meta.freq_setup.n_subbands == 2 and meta.freq_setup.n_channels == 2
    assert meta.freq_setup.total_bandwidth == pytest.approx(32e6)
    assert meta.freq_setup.polarizations == [Stokes.RR, Stokes.LL]
    assert meta.obs_date == (dt.datetime(1858, 11, 17) + dt.timedelta(seconds=T0)).date()
    assert len(meta.scans) == 2
    scan1, scan2 = meta.scans
    assert scan1.source == "3C345" and scan1.antennas == ["EF", "JB", "WB"]
    assert scan2.source == "J1848+3219" and scan2.antennas == ["EF", "WB"]
    assert scan2.duration_sec == pytest.approx(30.0)


def test_listobs(store, tmp_path):
    backend = DaskMsBackend(work_dir=str(store))
    listing = backend.data.listobs(PROJECT)
    assert listing["scan_1"]["source"] == "3C345"
    outfile = tmp_path / "listing.txt"
    backend.data.listobs(PROJECT, listfile=str(outfile))
    assert "3C345" in outfile.read_text()


def test_zero_pad_cells():
    """Variable-shape rows (GAIN_CURVE.GAIN style) are stacked zero-padded to the max shape."""
    cells = [np.ones((1, 2)), np.full((3, 2), 2.0), None]
    out = _zero_pad_cells(cells, np.float32)
    assert out.shape == (3, 3, 2) and out.dtype == np.float32
    assert out[0, 0].tolist() == [1.0, 1.0] and out[0, 1:].sum() == 0  # padded with zeros
    assert out[1].tolist() == [[2.0, 2.0]] * 3                          # full row kept as-is
    assert out[2].sum() == 0                                            # undefined row -> zeros
    with pytest.raises(ValueError, match="no readable cells"):
        _zero_pad_cells([None, None], np.float32)


def test_import_namespace_skips_when_store_exists(store):
    """A fresh Observation over an existing store must not search/download raw files."""
    cfg = load_config({"global": {"work_dir": str(store), "backend": "dask-ms"}})
    obs = Observation(PROJECT, cfg)
    meta = obs.import_data()  # no FITS-IDI files anywhere: must skip straight to metadata
    assert meta.source_names == ["3C345", "J1848+3219"]
    assert obs.data[0].DATA.shape == (3, 2, 2)
    assert obs.state.status("import_data") == "done"
