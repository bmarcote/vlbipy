"""The native UVFITS writer (``exportuvfits``) against CASA's file for an RSM07 split measurement set."""
import os
from pathlib import Path

import numpy as np
import pytest

SPLIT = Path(__file__).resolve().parents[1] / "test" / "e2e_daskms" / "calibrated_data" / "RSM07_3C345.ms"
#: AIPS AN keywords casacore takes from its own IERS tables (the native writer uses astropy's).
EARTH_ORIENTATION = ("GSTIA0", "DEGPDY", "POLARX", "POLARY", "UT1UTC", "IATUTC")


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
    assert os.path.getsize(tmp_path / "casa.uvfits") == os.path.getsize(tmp_path / "native.uvfits")
    with fits.open(tmp_path / "casa.uvfits") as casa, fits.open(tmp_path / "native.uvfits") as ours:
        assert [h.name for h in ours] == [h.name for h in casa]
        assert list(ours[0].header.items()) == list(casa[0].header.items())
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
