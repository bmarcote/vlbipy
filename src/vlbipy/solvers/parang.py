"""Parallactic-angle (CASA "P Jones") correction for circularly polarised VLBI visibilities.

Reproduces what CASA does with ``parang=True``: each antenna's feed rotates on the sky by a mount-dependent
angle (``feed_angle``) and the visibilities are de-rotated before solving.  The geometry follows casacore
``MSDerivedValues::parAngle`` (ms/MSOper/MSDerivedValues.cc):

- ALT-AZ (and ALT-AZ+ROTATOR, empty string): chi, the parallactic angle.
- EQUATORIAL: 0.
- X-Y: atan2(-cos(H), -sin(H) sin(dec)) with H the hour angle.
- ALT-AZ+NASMYTH-R: chi + el;  ALT-AZ+NASMYTH-L: chi - el.
- ALT-AZ+BWG-R: chi + (el - az);  ALT-AZ+BWG-L: chi - (el - az).
- anything else: casacore warns "unhandled mount type" and uses 0.  Here we warn and fall back to ALT-AZ
  (chi), which is the right choice for every VLBI antenna we know of with a mislabelled mount.

Sign convention of ``apply_parang`` (verified against CASA fringefit tables of rsm07 scan 63; do not change):
for baseline (i, j) with i = ANTENNA1, j = ANTENNA2 and dchi = angle_i - angle_j,
RR -> V exp(+1j dchi), LL -> V exp(-1j dchi), RL -> V exp(+1j (angle_i + angle_j)), LR -> V exp(-1j (angle_i + angle_j)).

Astropy's apparent sidereal time is used (accuracy of a few arcsec, far better than needed).  No Python loop
over times; one small loop over antennas.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ..logging_utils import get_logger

logger = get_logger()

ALT_AZ_MOUNTS = {"ALT-AZ", "ALT-AZ+ROTATOR", ""}
EQUATORIAL_MOUNTS = {"EQUATORIAL"}
XY_MOUNTS = {"X-Y"}
NASMYTH_R_MOUNTS = {"NASMYTH-R", "ALT-AZ+NASMYTH-R"}
NASMYTH_L_MOUNTS = {"NASMYTH-L", "ALT-AZ+NASMYTH-L"}
BWG_R_MOUNTS = {"BWG-R", "ALT-AZ+BWG-R"}
BWG_L_MOUNTS = {"BWG-L", "ALT-AZ+BWG-L"}


def _hour_angle_and_latitude(antenna_xyz, ra_rad, time_mjd_s):
    """Return (H, lat) with H the local hour angle (nant, ntime) [rad] and lat the geodetic latitude (nant,) [rad].

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres (MS ANTENNA.POSITION).
    ra_rad : float
        Source right ascension (ICRS/J2000) in radians.
    time_mjd_s : array (ntime,)
        Times in MJD seconds, UTC (MS TIME).

    Returns
    -------
    tuple of np.ndarray
        Hour angle (nant, ntime) in radians wrapped to [-pi, pi] and latitude (nant,) in radians.
    """
    xyz = np.atleast_2d(np.asarray(antenna_xyz, dtype=float))
    t_s = np.atleast_1d(np.asarray(time_mjd_s, dtype=float))
    location = EarthLocation.from_geocentric(xyz[:, 0] * u.m, xyz[:, 1] * u.m, xyz[:, 2] * u.m)
    lon = location.lon.to_value(u.rad)
    lat = location.lat.to_value(u.rad)
    times = Time(t_s / 86400.0, format="mjd", scale="utc")
    # Greenwich apparent sidereal time once; the per-antenna longitude offset is a cheap addition.
    gast = times.sidereal_time("apparent", longitude=0.0 * u.deg).to_value(u.rad)
    hour_angle = gast[np.newaxis, :] + lon[:, np.newaxis] - float(ra_rad)
    hour_angle = (hour_angle + np.pi) % (2.0 * np.pi) - np.pi
    return hour_angle, lat


def parallactic_angle(antenna_xyz, ra_rad, dec_rad, time_mjd_s):
    """Compute the parallactic angle chi [rad], shape (nant, ntime), of a source seen from each antenna.

    chi = atan2(sin(H), tan(lat) cos(dec) - sin(dec) cos(H)); chi is 0 at transit for a source south of the zenith
    and pi for a source north of it (northern-hemisphere convention), matching casacore's ``parAngle``.

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres.
    ra_rad, dec_rad : float
        Source direction (ICRS/J2000) in radians.
    time_mjd_s : array (ntime,)
        Times in MJD seconds (UTC).

    Returns
    -------
    np.ndarray
        Parallactic angle in radians, shape (nant, ntime), in (-pi, pi].
    """
    hour_angle, lat = _hour_angle_and_latitude(antenna_xyz, ra_rad, time_mjd_s)
    dec = float(dec_rad)
    denominator = np.tan(lat)[:, np.newaxis] * np.cos(dec) - np.sin(dec) * np.cos(hour_angle)
    return np.arctan2(np.sin(hour_angle), denominator)


def elevation(antenna_xyz, ra_rad, dec_rad, time_mjd_s):
    """Compute the source elevation [rad], shape (nant, ntime), ignoring refraction.

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres.
    ra_rad, dec_rad : float
        Source direction (ICRS/J2000) in radians.
    time_mjd_s : array (ntime,)
        Times in MJD seconds (UTC).

    Returns
    -------
    np.ndarray
        Elevation in radians, shape (nant, ntime), in [-pi/2, pi/2].
    """
    hour_angle, lat = _hour_angle_and_latitude(antenna_xyz, ra_rad, time_mjd_s)
    dec = float(dec_rad)
    sin_el = np.sin(lat)[:, np.newaxis] * np.sin(dec) + np.cos(lat)[:, np.newaxis] * np.cos(dec) * np.cos(hour_angle)
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))


def azimuth(antenna_xyz, ra_rad, dec_rad, time_mjd_s):
    """Compute the source azimuth [rad] (north through east), shape (nant, ntime), ignoring refraction.

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres.
    ra_rad, dec_rad : float
        Source direction (ICRS/J2000) in radians.
    time_mjd_s : array (ntime,)
        Times in MJD seconds (UTC).

    Returns
    -------
    np.ndarray
        Azimuth in radians, shape (nant, ntime), in [0, 2 pi).
    """
    hour_angle, lat = _hour_angle_and_latitude(antenna_xyz, ra_rad, time_mjd_s)
    dec = float(dec_rad)
    lat2 = lat[:, np.newaxis]
    x = np.cos(lat2) * np.sin(dec) - np.sin(lat2) * np.cos(dec) * np.cos(hour_angle)
    y = -np.cos(dec) * np.sin(hour_angle)
    return np.arctan2(y, x) % (2.0 * np.pi)


def xy_mount_angle(antenna_xyz, ra_rad, dec_rad, time_mjd_s):
    """Compute casacore's feed rotation for X-Y mounts: atan2(-cos(H), -sin(H) sin(dec)), shape (nant, ntime) [rad].

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres.
    ra_rad, dec_rad : float
        Source direction (ICRS/J2000) in radians.
    time_mjd_s : array (ntime,)
        Times in MJD seconds (UTC).

    Returns
    -------
    np.ndarray
        X-Y mount angle in radians, shape (nant, ntime).
    """
    hour_angle, _ = _hour_angle_and_latitude(antenna_xyz, ra_rad, time_mjd_s)
    return np.arctan2(-np.cos(hour_angle), -np.sin(hour_angle) * np.sin(float(dec_rad)))


def feed_angle(chi, el, mounts, az=None, xy_angle=None):
    """Return the per-antenna feed rotation angle CASA applies, by mount type (shape (nant, ntime) [rad]).

    Parameters
    ----------
    chi : array (nant, ntime)
        Parallactic angle from ``parallactic_angle``.
    el : array (nant, ntime)
        Elevation from ``elevation`` (only used for Nasmyth/BWG mounts).
    mounts : sequence of str (nant,)
        MS ANTENNA.MOUNT strings, matched case-insensitively.
    az : array (nant, ntime), optional
        Azimuth from ``azimuth``; required only for BWG mounts.
    xy_angle : array (nant, ntime), optional
        Angle from ``xy_mount_angle``; required only for X-Y mounts.

    Returns
    -------
    np.ndarray
        Feed angle in radians, same shape as ``chi``.  Unknown mounts are treated as ALT-AZ with a warning.
    """
    chi = np.asarray(chi, dtype=float)
    el = np.asarray(el, dtype=float)
    out = np.array(chi, copy=True)
    for i, mount in enumerate(mounts):
        key = str(mount).strip().upper()
        if key in ALT_AZ_MOUNTS:
            out[i] = chi[i]
        elif key in EQUATORIAL_MOUNTS:
            out[i] = 0.0
        elif key in NASMYTH_R_MOUNTS:
            out[i] = chi[i] + el[i]
        elif key in NASMYTH_L_MOUNTS:
            out[i] = chi[i] - el[i]
        elif key in BWG_R_MOUNTS or key in BWG_L_MOUNTS:
            if az is None:
                raise ValueError(f"antenna {i}: mount {mount!r} needs the azimuth (pass az=azimuth(...))")
            sign = 1.0 if key in BWG_R_MOUNTS else -1.0
            out[i] = chi[i] + sign * (el[i] - np.asarray(az, dtype=float)[i])
        elif key in XY_MOUNTS:
            if xy_angle is None:
                raise ValueError(f"antenna {i}: mount {mount!r} needs xy_angle (pass xy_angle=xy_mount_angle(...))")
            out[i] = np.asarray(xy_angle, dtype=float)[i]
        else:
            logger.warning(f"antenna {i}: unknown mount {mount!r}; treating as ALT-AZ for the parallactic angle")
            out[i] = chi[i]
    return out


def feed_angles_for_ms(antenna_xyz, mounts, ra_rad, dec_rad, time_mjd_s):
    """Convenience: compute the full mount-dependent feed angle (nant, ntime) [rad] for a source and set of times.

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres.
    mounts : sequence of str (nant,)
        MS ANTENNA.MOUNT strings.
    ra_rad, dec_rad : float
        Source direction (ICRS/J2000) in radians.
    time_mjd_s : array (ntime,)
        Times in MJD seconds (UTC).

    Returns
    -------
    np.ndarray
        Feed angle in radians, shape (nant, ntime).
    """
    chi = parallactic_angle(antenna_xyz, ra_rad, dec_rad, time_mjd_s)
    el = elevation(antenna_xyz, ra_rad, dec_rad, time_mjd_s)
    keys = {str(m).strip().upper() for m in mounts}
    az = azimuth(antenna_xyz, ra_rad, dec_rad, time_mjd_s) if keys & (BWG_R_MOUNTS | BWG_L_MOUNTS) else None
    xy = xy_mount_angle(antenna_xyz, ra_rad, dec_rad, time_mjd_s) if keys & XY_MOUNTS else None
    return feed_angle(chi, el, mounts, az=az, xy_angle=xy)


def apply_parang(vis, antenna1, antenna2, row_time, angle, antenna_index=None, *, time_index=None):
    """De-rotate circular-feed visibilities by the per-antenna feed angles (CASA ``parang=True`` convention).

    For baseline (i, j) with i = ANTENNA1, j = ANTENNA2 and dchi = angle_i - angle_j:
    RR (corr 0) -> V exp(+1j dchi); LL (last corr when ncorr >= 2) -> V exp(-1j dchi);
    RL (corr 1 of 4) -> V exp(+1j (angle_i + angle_j)); LR (corr 2 of 4) -> V exp(-1j (angle_i + angle_j)).
    After the rotation an antenna phase solved on RR becomes theta_RR + (chi_ant - chi_ref) and on LL
    theta_LL - (chi_ant - chi_ref), with chi as returned by :func:`parallactic_angle` here. The sign was
    verified against CASA fringefit tables of RSM07 scan 63 solved with parang=False/True; do not change.

    Parameters
    ----------
    vis : array (nrow, nchan, ncorr), complex
        Row-based visibilities with ncorr in (1, 2, 4); corr order RR, (RL, LR,) LL.
    antenna1, antenna2 : array (nrow,) int
        MS ANTENNA1 / ANTENNA2 ids per row.
    row_time : array (nrow,)
        MS TIME per row (MJD seconds).  Ignored when ``time_index`` is given.
    angle : array (nant, ntime)
        Feed angle [rad] per antenna sampled at the sorted unique times ``time_index`` refers to.
    antenna_index : array (nant,) int, optional
        MS antenna id of each row of ``angle``; default ``arange(nant)``.
    time_index : array (ntime,) or (nrow,), optional
        Either the sorted unique time array that ``angle`` is sampled on (length ntime, matched to ``row_time`` by
        nearest neighbour) or a precomputed per-row column index into ``angle`` (length nrow, integer dtype).
        When omitted, ``np.unique(row_time)`` is used and must have exactly ntime entries.

    Returns
    -------
    np.ndarray
        Rotated visibilities, new array with the dtype of ``vis`` (complex64 stays complex64).
    """
    vis = np.asarray(vis)
    if vis.ndim != 3 or vis.shape[2] not in (1, 2, 4):
        raise ValueError(f"vis must be (nrow, nchan, ncorr) with ncorr in (1, 2, 4); got shape {vis.shape}")
    angle = np.asarray(angle, dtype=float)
    antenna1 = np.asarray(antenna1, dtype=int)
    antenna2 = np.asarray(antenna2, dtype=int)
    nant, ntime = angle.shape
    col = _row_time_column(np.asarray(row_time, dtype=float), ntime, time_index)
    if antenna_index is None:
        row_a1, row_a2 = antenna1, antenna2
    else:
        lookup = np.full(int(np.max(antenna_index)) + 1, -1, dtype=int)
        lookup[np.asarray(antenna_index, dtype=int)] = np.arange(nant)
        row_a1, row_a2 = lookup[antenna1], lookup[antenna2]
        if np.any(row_a1 < 0) or np.any(row_a2 < 0):
            raise ValueError("apply_parang: some rows reference antennas missing from antenna_index")
    a_i = angle[row_a1, col]
    a_j = angle[row_a2, col]
    diff = a_i - a_j
    total = a_i + a_j
    ncorr = vis.shape[2]
    phase = np.zeros((vis.shape[0], ncorr), dtype=float)
    phase[:, 0] = diff
    if ncorr >= 2:
        phase[:, ncorr - 1] = -diff
    if ncorr == 4:
        phase[:, 1] = total
        phase[:, 2] = -total
    rotation = np.exp(1j * phase).astype(vis.dtype if np.iscomplexobj(vis) else np.complex128)
    out = vis * rotation[:, np.newaxis, :]
    logger.debug(f"apply_parang: rotated {vis.shape[0]} rows, ncorr={ncorr}, nant={nant}, ntime={ntime}")
    return out


def _row_time_column(row_time, ntime, time_index):
    """Map each row to a column of the (nant, ntime) angle array.

    Parameters
    ----------
    row_time : array (nrow,)
        MS TIME per row.
    ntime : int
        Number of time samples in the angle array.
    time_index : None, array (ntime,) float or array (nrow,) int
        See ``apply_parang``.

    Returns
    -------
    np.ndarray
        Integer column index per row, shape (nrow,).
    """
    if time_index is not None:
        time_index = np.asarray(time_index)
        if np.issubdtype(time_index.dtype, np.integer) and time_index.shape == row_time.shape:
            return time_index
        times = np.asarray(time_index, dtype=float)
    else:
        times = np.unique(row_time)
    if times.shape[0] != ntime:
        raise ValueError(f"angle has {ntime} time samples but {times.shape[0]} unique times were given")
    pos = np.searchsorted(times, row_time)
    pos = np.clip(pos, 1, ntime - 1) if ntime > 1 else np.zeros_like(pos)
    left, right = times[pos - 1], times[pos]
    col = np.where(np.abs(row_time - left) <= np.abs(right - row_time), pos - 1, pos)
    return col


def ms_parang_inputs(ms_path, field_id):
    """Read antenna positions, mounts and the phase centre of ``field_id`` from a measurement set.

    Tries casatools first and falls back to python-casacore; both are imported lazily.

    Parameters
    ----------
    ms_path : str or Path
        Measurement set path.
    field_id : int
        FIELD_ID whose PHASE_DIR is returned (first polynomial term).

    Returns
    -------
    dict
        {"antenna_xyz": (nant, 3) float array [m], "mounts": list of str, "ra_rad": float, "dec_rad": float}.
    """
    ms_path = str(ms_path)
    try:
        from casatools import table as _casa_table
    except ImportError:
        _casa_table = None
    if _casa_table is not None:
        tb = _casa_table()
        tb.open(f"{ms_path}/ANTENNA")
        antenna_xyz = np.asarray(tb.getcol("POSITION"), dtype=float).T
        mounts = [str(m) for m in tb.getcol("MOUNT")]
        tb.close()
        tb.open(f"{ms_path}/FIELD")
        phase_dir = np.asarray(tb.getcol("PHASE_DIR"))  # casatools: (2, npoly, nfield)
        tb.close()
        ra_rad, dec_rad = float(phase_dir[0, 0, field_id]), float(phase_dir[1, 0, field_id])
    else:
        from casacore.tables import table as _cc_table
        with _cc_table(f"{ms_path}/ANTENNA", ack=False) as t:
            antenna_xyz = np.asarray(t.getcol("POSITION"), dtype=float)
            mounts = [str(m) for m in t.getcol("MOUNT")]
        with _cc_table(f"{ms_path}/FIELD", ack=False) as t:
            phase_dir = np.asarray(t.getcell("PHASE_DIR", int(field_id)))  # (npoly, 2)
        ra_rad, dec_rad = float(phase_dir[0, 0]), float(phase_dir[0, 1])
    logger.debug(f"ms_parang_inputs: {len(mounts)} antennas, field {field_id} at ra={ra_rad:.6f} dec={dec_rad:.6f} rad")
    return {"antenna_xyz": antenna_xyz, "mounts": mounts, "ra_rad": ra_rad, "dec_rad": dec_rad}
