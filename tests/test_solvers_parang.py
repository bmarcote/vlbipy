"""Tests for vlbipy.solvers.parang: feed-angle geometry, the apply_parang sign convention and a CASA cross-check."""

import time
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from vlbipy.solvers import parang

MS_PATH = Path("/home/marcote/Programing/vlbipy/rsm07_manual/rsm07.ms")
CASA_PARANG0 = Path("/tmp/vlbipy_spec/casa_ab/sbd_parang0")
CASA_PARANG1 = Path("/tmp/vlbipy_spec/casa_ab/sbd_parang1")
FIELD_3C345 = 16
SCAN_TIME_MJD_S = 5264754457.7
REF_ANTENNA = 2  # EF
# chi_EF - chi_ant in degrees, verified by hand for rsm07 scan 63 (3C345).
EXPECTED_EF_MINUS_ANT_DEG = {0: 6.7, 1: -32.3, 3: 10.5, 5: -90.3, 7: -32.3, 8: -39.1}
T0_MJD_S = 58000.0 * 86400.0


def _site_xyz(lat_deg, lon_deg=0.0):
    """ITRF position (1, 3) in metres of a site at the given geodetic latitude/longitude."""
    loc = EarthLocation.from_geodetic(lon_deg * u.deg, lat_deg * u.deg, 0.0 * u.m)
    return np.array([[loc.x.to_value(u.m), loc.y.to_value(u.m), loc.z.to_value(u.m)]])


def _transit_ra(lon_deg, time_mjd_s):
    """Right ascension [rad] that transits (H = 0) at longitude ``lon_deg`` at ``time_mjd_s``."""
    t = Time(time_mjd_s / 86400.0, format="mjd", scale="utc")
    return t.sidereal_time("apparent", longitude=lon_deg * u.deg).to_value(u.rad)


def _wrap_deg(x):
    """Wrap angles in degrees to (-180, 180]."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def test_parallactic_angle_zero_at_transit_south_of_zenith():
    """At transit a source south of the zenith (lat 50, dec 0) has chi = 0 and el = 40 deg."""
    xyz = _site_xyz(50.0)
    ra = _transit_ra(0.0, T0_MJD_S)
    chi = parang.parallactic_angle(xyz, ra, 0.0, [T0_MJD_S])
    el = parang.elevation(xyz, ra, 0.0, [T0_MJD_S])
    assert chi.shape == (1, 1)
    assert abs(np.degrees(chi[0, 0])) < 0.05
    assert abs(np.degrees(el[0, 0]) - 40.0) < 0.05


def test_parallactic_angle_sign_before_and_after_transit():
    """Scanning a day around transit: chi is negative before (H < 0) and positive after (H > 0) transit."""
    xyz = _site_xyz(50.0)
    ra = _transit_ra(0.0, T0_MJD_S)
    times = T0_MJD_S + np.array([-3600.0, 3600.0])
    chi = parang.parallactic_angle(xyz, ra, 0.0, times)[0]
    assert chi[0] < 0 < chi[1]
    assert abs(chi[0] + chi[1]) < 1e-3


def test_elevation_at_zenith_transit():
    """A source with dec = site latitude transits at the zenith: elevation ~ 90 deg."""
    xyz = _site_xyz(50.0)
    ra = _transit_ra(0.0, T0_MJD_S)
    el = parang.elevation(xyz, ra, np.radians(50.0), [T0_MJD_S])
    assert np.degrees(el[0, 0]) > 89.9


def test_feed_angle_equatorial_is_zero():
    """EQUATORIAL mounts have no feed rotation whatever chi and el are."""
    chi = np.array([[0.3, -1.2, 2.0]])
    el = np.array([[0.5, 0.6, 0.7]])
    out = parang.feed_angle(chi, el, ["EQUATORIAL"])
    np.testing.assert_allclose(out, 0.0)


def test_feed_angle_mount_variants():
    """ALT-AZ -> chi; NASMYTH-R -> chi + el; NASMYTH-L -> chi - el; unknown mount -> chi (warns)."""
    chi = np.array([[0.3, -1.2], [0.3, -1.2], [0.3, -1.2], [0.3, -1.2]])
    el = np.array([[0.5, 0.6], [0.5, 0.6], [0.5, 0.6], [0.5, 0.6]])
    mounts = ["alt-az", "ALT-AZ+NASMYTH-R", "NASMYTH-L", "WEIRD-MOUNT"]
    out = parang.feed_angle(chi, el, mounts)
    np.testing.assert_allclose(out[0], chi[0])
    np.testing.assert_allclose(out[1], chi[1] + el[1])
    np.testing.assert_allclose(out[2], chi[2] - el[2])
    np.testing.assert_allclose(out[3], chi[3])


def test_feed_angle_bwg_and_xy_need_extra_inputs():
    """BWG mounts need az, X-Y mounts need xy_angle; the values follow the documented formulas."""
    chi, el, az, xy = np.array([[0.3]]), np.array([[0.5]]), np.array([[1.1]]), np.array([[2.2]])
    with pytest.raises(ValueError):
        parang.feed_angle(chi, el, ["ALT-AZ+BWG-R"])
    with pytest.raises(ValueError):
        parang.feed_angle(chi, el, ["X-Y"])
    np.testing.assert_allclose(parang.feed_angle(chi, el, ["BWG-R"], az=az), chi + (el - az))
    np.testing.assert_allclose(parang.feed_angle(chi, el, ["BWG-L"], az=az), chi - (el - az))
    np.testing.assert_allclose(parang.feed_angle(chi, el, ["X-Y"], xy_angle=xy), xy)


def _synthetic_baselines(nant):
    """All baselines i < j of ``nant`` antennas as (antenna1, antenna2) arrays."""
    a1, a2 = np.triu_indices(nant, k=1)
    return a1, a2


def _synthetic_vis(theta, a1, a2, nchan=4, dtype=np.complex128):
    """Visibilities (nrow, nchan, 4) with RR = LL = exp(i(theta_i - theta_j)) and RL = LR = 0."""
    phase = np.exp(1j * (theta[a1] - theta[a2]))
    vis = np.zeros((a1.size, nchan, 4), dtype=dtype)
    vis[:, :, 0] = phase[:, np.newaxis]
    vis[:, :, 3] = phase[:, np.newaxis]
    return vis


def test_apply_parang_phase_convention():
    """RR phase becomes (theta_i + chi_i) - (theta_j + chi_j); LL becomes (theta_i - chi_i) - (theta_j - chi_j)."""
    nant = 5
    rng = np.random.default_rng(1)
    theta = rng.uniform(-np.pi, np.pi, nant)
    chi = rng.uniform(-np.pi, np.pi, nant)
    a1, a2 = _synthetic_baselines(nant)
    times = np.array([1.0, 2.0, 3.0])
    row_time = np.repeat(times, a1.size)
    a1_rows, a2_rows = np.tile(a1, times.size), np.tile(a2, times.size)
    vis = _synthetic_vis(theta, a1_rows, a2_rows)
    angle = np.repeat(chi[:, np.newaxis], times.size, axis=1)
    out = parang.apply_parang(vis, a1_rows, a2_rows, row_time, angle)
    expected_rr = (theta[a1_rows] + chi[a1_rows]) - (theta[a2_rows] + chi[a2_rows])
    expected_ll = (theta[a1_rows] - chi[a1_rows]) - (theta[a2_rows] - chi[a2_rows])
    np.testing.assert_allclose(_wrap_deg(np.degrees(np.angle(out[:, 0, 0]) - expected_rr)), 0.0, atol=1e-9)
    np.testing.assert_allclose(_wrap_deg(np.degrees(np.angle(out[:, -1, 3]) - expected_ll)), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.abs(out[:, :, 0]), 1.0)
    np.testing.assert_allclose(out[:, :, 1], 0.0)
    np.testing.assert_allclose(out[:, :, 2], 0.0)


def test_apply_parang_time_index_variants_and_dtype():
    """Unique-time and per-row-index ``time_index`` give the same result as the default; complex64 is preserved."""
    nant = 5
    rng = np.random.default_rng(2)
    theta = rng.uniform(-np.pi, np.pi, nant)
    a1, a2 = _synthetic_baselines(nant)
    times = np.array([10.0, 20.0, 30.0])
    row_time = np.repeat(times, a1.size)
    a1_rows, a2_rows = np.tile(a1, times.size), np.tile(a2, times.size)
    angle = rng.uniform(-np.pi, np.pi, (nant, times.size))
    vis = _synthetic_vis(theta, a1_rows, a2_rows, dtype=np.complex64)
    out_default = parang.apply_parang(vis, a1_rows, a2_rows, row_time, angle)
    out_unique = parang.apply_parang(vis, a1_rows, a2_rows, row_time, angle, time_index=times)
    col = np.repeat(np.arange(times.size), a1.size)
    out_rows = parang.apply_parang(vis, a1_rows, a2_rows, row_time, angle, time_index=col)
    assert out_default.dtype == np.complex64
    assert out_unique.dtype == np.complex64
    assert out_rows.dtype == np.complex64
    np.testing.assert_allclose(out_unique, out_default, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_rows, out_default, rtol=1e-6, atol=1e-6)
    expected_rr = (theta[a1_rows] - theta[a2_rows]) + (angle[a1_rows, col] - angle[a2_rows, col])
    np.testing.assert_allclose(_wrap_deg(np.degrees(np.angle(out_rows[:, 0, 0]) - expected_rr)), 0.0, atol=1e-3)


def test_apply_parang_antenna_index_and_errors():
    """antenna_index maps MS ids to angle rows; wrong ncorr or mismatched time counts raise ValueError."""
    angle = np.array([[0.1], [0.2]])
    vis = np.ones((1, 2, 2), dtype=complex)
    out = parang.apply_parang(vis, [7], [9], [5.0], angle, antenna_index=[7, 9])
    np.testing.assert_allclose(np.angle(out[0, 0, 0]), 0.1 - 0.2)
    np.testing.assert_allclose(np.angle(out[0, 0, 1]), 0.2 - 0.1)
    with pytest.raises(ValueError):
        parang.apply_parang(vis, [7], [8], [5.0], angle, antenna_index=[7, 9])
    with pytest.raises(ValueError):
        parang.apply_parang(np.ones((1, 2, 3), dtype=complex), [0], [1], [5.0], angle)
    with pytest.raises(ValueError):
        parang.apply_parang(vis, [0, 1], [1, 0], [5.0, 6.0], angle)


@pytest.fixture(scope="module")
def rsm07_feed_angles():
    """Feed angles (nant,) [rad] for 3C345 in rsm07 at the scan-63 time, or skip when the MS is absent."""
    if not MS_PATH.exists():
        pytest.skip(f"measurement set not available: {MS_PATH}")
    inputs = parang.ms_parang_inputs(MS_PATH, FIELD_3C345)
    angles = parang.feed_angles_for_ms(inputs["antenna_xyz"], inputs["mounts"], inputs["ra_rad"], inputs["dec_rad"],
                                       [SCAN_TIME_MJD_S])
    assert angles.shape == (len(inputs["mounts"]), 1)
    return angles[:, 0]


def test_rsm07_feed_angles_match_hand_values(rsm07_feed_angles):
    """chi_EF - chi_ant for JB, WB, MC, T6, HH, IR agree with the hand-verified values within 2 degrees."""
    diff_deg = np.degrees(rsm07_feed_angles[REF_ANTENNA] - rsm07_feed_angles)
    for ant, expected in EXPECTED_EF_MINUS_ANT_DEG.items():
        assert abs(_wrap_deg(diff_deg[ant] - expected)) < 2.0, f"antenna {ant}: {diff_deg[ant]:.1f} vs {expected}"


def test_rsm07_feed_angles_match_casa_fringefit_tables(rsm07_feed_angles):
    """CASA parang=True minus parang=False phases (spw 0) equal chi_ant - chi_EF on RR and the negative on LL."""
    if not (CASA_PARANG0.exists() and CASA_PARANG1.exists()):
        pytest.skip(f"CASA reference tables not available: {CASA_PARANG0}, {CASA_PARANG1}")
    tables = pytest.importorskip("casacore.tables")
    with tables.table(str(CASA_PARANG0), ack=False) as t0, tables.table(str(CASA_PARANG1), ack=False) as t1:
        ant0, spw0 = t0.getcol("ANTENNA1"), t0.getcol("SPECTRAL_WINDOW_ID")
        assert np.array_equal(ant0, t1.getcol("ANTENNA1"))
        assert np.array_equal(spw0, t1.getcol("SPECTRAL_WINDOW_ID"))
        checked = 0
        for row in np.flatnonzero(spw0 == 0):
            if t0.getcell("FLAG", row)[0, 0] or t1.getcell("FLAG", row)[0, 0]:
                continue
            f0, f1 = t0.getcell("FPARAM", row), t1.getcell("FPARAM", row)
            expected = np.degrees(rsm07_feed_angles[ant0[row]] - rsm07_feed_angles[REF_ANTENNA])
            rr = _wrap_deg(np.degrees(f1[0, 0] - f0[0, 0]) - expected)
            ll = _wrap_deg(np.degrees(f1[0, 4] - f0[0, 4]) + expected)
            assert abs(rr) < 2.0, f"antenna {ant0[row]} RR: off by {rr:.1f} deg"
            assert abs(ll) < 2.0, f"antenna {ant0[row]} LL: off by {ll:.1f} deg"
            checked += 1
    assert checked >= 6


def test_feed_angles_for_ms_is_fast():
    """14 antennas x 150 times (mixed mounts, including BWG and X-Y) run in well under 2 s."""
    rng = np.random.default_rng(3)
    lats = rng.uniform(-60.0, 70.0, 14)
    lons = rng.uniform(-180.0, 180.0, 14)
    xyz = np.vstack([_site_xyz(lat, lon) for lat, lon in zip(lats, lons)])
    mounts = ["ALT-AZ"] * 10 + ["EQUATORIAL", "ALT-AZ+NASMYTH-R", "BWG-L", "X-Y"]
    times = T0_MJD_S + np.arange(150) * 2.0
    start = time.perf_counter()
    out = parang.feed_angles_for_ms(xyz, mounts, 1.0, 0.5, times)
    elapsed = time.perf_counter() - start
    assert out.shape == (14, 150)
    assert np.all(np.isfinite(out))
    assert elapsed < 2.0, f"feed_angles_for_ms took {elapsed:.2f} s"
