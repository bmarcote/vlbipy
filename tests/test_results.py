"""Tests for the returned product objects (Image, ImageSet, SelfcalResult)."""
import pytest

from vlbipy.models import QualityMetrics
from vlbipy.results import Image, ImageSet, SelfcalResult


def _img(source="S", robust=0.0, dr=1000.0):
    return Image(source, robust, "briggs", {"png": f"{source}.png", "fits": f"{source}.fits"},
                 QualityMetrics(peak=1.0, rms=1.0 / dr, dynamic_range=dr))


def test_image_export_and_preview():
    img = _img()
    assert img.export_fits("out.fits") == "out.fits"
    assert img.preview() == "S.png"


def test_imageset_indexing_and_best():
    imgs = [_img(robust=r, dr=1000 * (r + 3)) for r in (-2.0, 0.0, 2.0)]
    iset = ImageSet(imgs)
    assert len(iset) == 3
    assert iset[0].robust == -2.0            # positional
    assert iset[2.0].robust == 2.0            # by robust value
    assert iset.best().robust == 2.0          # highest DR
    assert set(iset.by_robust) == {-2.0, 0.0, 2.0}
    with pytest.raises(KeyError):
        _ = iset[9.0]


def test_imageset_empty_best_raises():
    with pytest.raises(ValueError):
        ImageSet([]).best()


def test_selfcal_summary():
    res = SelfcalResult("S", [{"mode": "p", "solint": "60s", "dr_before": 1000, "dr_after": 1200,
                               "accepted": True}], converged=True)
    assert "Self-cal S" in res.summary()
    assert res.converged is True
