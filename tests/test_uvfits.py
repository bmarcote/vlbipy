"""The native UVFITS writer (``exportuvfits``) against CASA's file for an RSM07 split measurement set."""
from pathlib import Path

import numpy as np
import pytest

SPLIT = Path(__file__).resolve().parents[1] / "test" / "e2e_daskms" / "calibrated_data" / "RSM07_3C345.ms"
#: AIPS AN keywords casacore takes from its own IERS tables (the native writer uses astropy's).
EARTH_ORIENTATION = ("GSTIA0", "DEGPDY", "POLARX", "POLARY", "UT1UTC", "IATUTC")


def test_icrs_header_preserves_frame_and_legacy_epoch():
    """ICRS is not FK5: keep its name alongside the legacy UVFITS J2000 epoch."""
    from types import SimpleNamespace
    from astropy.io import fits
    from vlbipy.solvers.uvfits import _primary_cards

    main = {"TIME": np.array([5e9])}
    sub = {"phase_dir": np.deg2rad([211.31, -61.307]), "direction_ref": "ICRS", "field_name": "TARGET",
           "observation": {"TELESCOPE_NAME": ["LBA"], "OBSERVER": ["TEST"]}, "rest_freq": 0.0}
    setup = {"nif": 1, "numchan": 1, "numcorr": 2, "crval3": -1.0, "cdelt3": -1.0,
             "ref_freq": 8.4e9, "bw0": 32e6, "ref_pix": 1, "meas_freq": 5}
    table = SimpleNamespace(getcolkeywords=lambda column: {})
    header = fits.Header.fromstring("".join(_primary_cards(main, sub, setup, [], [], 1, "DATA", table)))
    assert header["RADESYS"] == "ICRS"
    assert header["EPOCH"] == 2000.0
    assert "EQUINOX" not in header
    assert header["CRVAL6"] == pytest.approx(211.31)
    assert header["CRVAL7"] == pytest.approx(-61.307)


@pytest.mark.parametrize("frame", ["ICRS", "J2000", "B1950"])
def test_imported_field_metadata_survives_split(tmp_path, frame):
    """The import schema and split row reduction retain directions, frames and UTC time metadata."""
    from vlbipy.casacore_format.table import Table
    from vlbipy.solvers.idi_subtables import _write
    from vlbipy.solvers.split import _reduce_field

    directions = np.array([[[1.0, -0.5]], [[2.0, -1.0]]])
    values = {name: directions.copy() for name in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR")}
    values.update(NAME=["CAL", "TARGET"], CODE=["", ""], FLAG_ROW=np.zeros(2, dtype=bool),
                  NUM_POLY=np.zeros(2, dtype=np.int32), SOURCE_ID=np.arange(2, dtype=np.int32),
                  TIME=np.array([5e9, 5e9 + 10]))
    _write(tmp_path, "FIELD", 2, values, field_ref=frame)
    _reduce_field(tmp_path / "FIELD", 1)
    with Table(tmp_path / "FIELD") as field:
        assert field.nrows() == 1
        for name in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            assert field.getcolkeywords(name)["MEASINFO"]["Ref"] == frame
            assert np.array_equal(field.getcell(name, 0), directions[1])
        assert field.getcolkeywords("TIME")["MEASINFO"]["Ref"] == "UTC"
        assert field.getcell("TIME", 0) == 5e9 + 10


@pytest.mark.parametrize("native_supported", [True, False])
def test_export_ops_prefer_native_writer(tmp_path, monkeypatch, native_supported):
    """Both real backends inherit the ICRS-aware writer, retaining CASA for unsupported layouts."""
    from unittest.mock import MagicMock
    from vlbipy.backends.casa import CasaExportOps
    from vlbipy.solvers import uvfits

    native = MagicMock(side_effect=None if native_supported else NotImplementedError("unsupported layout"))
    monkeypatch.setattr(uvfits, "exportuvfits", native)
    backend = MagicMock()
    ops = CasaExportOps(backend)
    target = ops.uvfits("TEST", "SRC", split_ms=str(tmp_path / "source.ms"))
    assert target == str(tmp_path / "source.uvfits")
    native.assert_called_once()
    assert backend.tasks.exportuvfits.call_count == (0 if native_supported else 1)
    if not native_supported:
        assert backend.tasks.exportuvfits.call_args == native.call_args


def test_multisource_export_is_left_to_casa(tmp_path):
    from vlbipy.solvers.uvfits import exportuvfits
    with pytest.raises(NotImplementedError):
        exportuvfits(tmp_path / "missing.ms", tmp_path / "out.uvfits", multisource=True)


@pytest.mark.skipif(not SPLIT.is_dir(), reason="RSM07 split measurement set not available")
def test_rsm07_split_exports_like_casa(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                   # CASA drops its log in the working directory
    pytest.importorskip("casatools")
    casatasks = pytest.importorskip("casatasks")
    from astropy.io import fits
    from vlbipy.solvers.uvfits import exportuvfits
    options = {"datacolumn": "data", "multisource": False, "combinespw": True, "padwithflags": True,
               "overwrite": True}
    casatasks.exportuvfits(vis=str(SPLIT), fitsfile=str(tmp_path / "casa.uvfits"), **options)
    report = exportuvfits(SPLIT, tmp_path / "native.uvfits", **options)
    assert report["groups"] > 0
    with fits.open(tmp_path / "casa.uvfits") as casa, fits.open(tmp_path / "native.uvfits") as ours:
        assert [h.name for h in ours] == [h.name for h in casa]
        header = ours[0].header.copy()
        if "RADESYS" in header:
            assert header.pop("RADESYS") == "ICRS"
            assert header["EPOCH"] == 2000.0
        assert list(header.items()) == list(casa[0].header.items())
        assert np.array_equal(ours[0].data.view(np.uint8), casa[0].data.view(np.uint8))   # data and parameters
        for index in range(1, len(casa)):
            assert list(ours[index].header) == list(casa[index].header)
            for card, our_card in zip(casa[index].header.cards, ours[index].header.cards):
                if card.keyword in EARTH_ORIENTATION:
                    assert our_card.value == pytest.approx(card.value, abs=2e-3), card.keyword
                else:
                    assert our_card.image == card.image, card.keyword
            for name in casa[index].data.columns.names:
                assert np.array_equal(np.asarray(ours[index].data[name]), np.asarray(casa[index].data[name])), name
