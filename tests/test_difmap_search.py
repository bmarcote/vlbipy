"""Tests for the wide-field target search of vlbipy.backends.difmap (no difmapy needed: a stub observation)."""
from __future__ import annotations

import numpy as np

from vlbipy.backends import difmap


class StubObservation:
    """Just enough of a difmapy Observation for ``find_peak``: a point source in noise."""

    def __init__(self, x_mas: float, y_mas: float, flux: float, noise: float = 1.0, resolution: float = 3.0):
        self.x, self.y, self.flux, self.noise, self.resolution = x_mas, y_mas, flux, noise, resolution

    def estimated_resolution(self) -> float:
        return self.resolution

    def clrmod(self):
        pass

    def uvweight(self, robust=None):
        pass

    def mapsize(self, npix, cell):
        self.npix, self.cell = int(npix), float(cell)

    def invert(self):
        n = self.npix
        yy, xx = np.mgrid[:n, :n]
        self.dbeam = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 8.0)
        iy, ix = int(round(self.y / self.cell)) + n // 2, int(round(self.x / self.cell)) + n // 2
        rng = np.random.default_rng(1)
        self.dmap = rng.normal(0.0, self.noise, (n, n)) + self.flux * np.roll(self.dbeam, (iy - n // 2, ix - n // 2), (0, 1))

    @property
    def valid_slice(self):
        n = self.npix
        return slice(n // 4, n - n // 4), slice(n // 4, n - n // 4)

    def peak_offset(self):
        sub = self.dmap[self.valid_slice]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        iy, ix = iy + self.npix // 4, ix + self.npix // 4
        return ((ix - self.npix / 2) * self.cell, (iy - self.npix / 2) * self.cell), float(self.dmap[iy, ix])

    def noise_stats(self, image=None):
        return {"rms": float(np.std(image))}


def test_find_peak_locates_an_offset_source():
    result = difmap.find_peak(StubObservation(60.0, -40.0, flux=50.0), fov_mas=400.0, threshold_sigma=10.0)
    assert result["detected"]
    assert abs(result["x"] - 60.0) <= result["cell_mas"] and abs(result["y"] + 40.0) <= result["cell_mas"]
    assert result["npix"] % 4 == 0 and result["npix"] * result["cell_mas"] >= 800.0


def test_find_peak_does_not_detect_noise():
    result = difmap.find_peak(StubObservation(0.0, 0.0, flux=0.0), fov_mas=400.0, threshold_sigma=10.0)
    assert not result["detected"] and result["snr"] < 10.0


def test_find_peak_noise_excludes_the_sidelobes_of_the_source():
    # The rms must stay at the noise level (1.0) however bright the source is.
    result = difmap.find_peak(StubObservation(20.0, 20.0, flux=1000.0), fov_mas=400.0)
    assert 0.8 < result["rms"] < 1.2


class LocatableObservation(StubObservation):
    """Adds what ``locate_source`` needs on top of the search stub: a beam, a source name and a shift log."""

    source = "TEST"
    estimated_beam = (4.0, 2.0, 0.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shifts = []

    def shift(self, east, north):
        self.shifts.append((east, north))

    def auto_mapsize(self):
        pass


def test_locate_source_keeps_the_centre_for_a_nearby_source():
    obs = LocatableObservation(12.0, -9.0, flux=50.0)          # 15 mas = 3.75 beams of 4 mas
    search, shift = difmap.locate_source(obs, "", search_fov_mas=400.0, recentre_min_beams=10.0)
    assert search["detected"] and not search["recentred"]
    assert shift == (0.0, 0.0) and obs.shifts == []


def test_locate_source_recentres_a_distant_source():
    obs = LocatableObservation(60.0, -45.0, flux=50.0)         # 75 mas = 18.75 beams
    search, shift = difmap.locate_source(obs, "", search_fov_mas=400.0, recentre_min_beams=10.0)
    assert search["recentred"] and search["offset_beams"] > 10.0
    assert obs.shifts == [(-shift[0], -shift[1])]


def test_locate_source_never_recentres_on_noise():
    obs = LocatableObservation(0.0, 0.0, flux=0.0)
    search, shift = difmap.locate_source(obs, "", search_fov_mas=400.0)
    assert not search["detected"] and shift == (0.0, 0.0) and obs.shifts == []
