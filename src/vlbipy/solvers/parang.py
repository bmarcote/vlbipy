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

The source is taken to the true equator and equinox of date before the hour angle is formed from the apparent
sidereal time (``_directions_of_date``), as casacore's measures do.  No Python loop over times; both are
interpolated between nodes fixed in absolute time, so the angle of an integration does not depend on which
other integrations it is computed with.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.time import Time

from ..logging_utils import get_logger

logger = get_logger()

ALT_AZ_MOUNTS = {"ALT-AZ", "ALT-AZ+ROTATOR", ""}
# casacore antenna columns can truncate EQUATORIAL to EQUATO on older MMSes.
EQUATORIAL_MOUNTS = {"EQUATORIAL", "EQUATO"}
XY_MOUNTS = {"X-Y"}
NASMYTH_R_MOUNTS = {"NASMYTH-R", "ALT-AZ+NASMYTH-R"}
NASMYTH_L_MOUNTS = {"NASMYTH-L", "ALT-AZ+NASMYTH-L"}
BWG_R_MOUNTS = {"BWG-R", "ALT-AZ+BWG-R"}
BWG_L_MOUNTS = {"BWG-L", "ALT-AZ+BWG-L"}


#: Spacing of the nodes the apparent sidereal time is evaluated on before interpolating [s].
_GAST_NODE_SPACING = 600.0
#: Spacing of the nodes the direction of date is evaluated on before interpolating [s].
_DATE_NODE_SPACING = 3600.0


def _absolute_nodes(t_s, spacing):
    """Nodes at the multiples of ``spacing`` seconds that bracket ``t_s``.

    They are fixed in time, so a value interpolated between them does not depend on which other times are
    asked for in the same call: an integration gets the same feed angle whichever block it is read in.
    """
    first = np.floor(t_s.min() / spacing) * spacing
    last = max(np.ceil(t_s.max() / spacing) * spacing, first + spacing)
    return np.arange(first, last + 0.5 * spacing, spacing)


def _between_nodes(t_s, nodes, values):
    """Angles ``values`` at ``nodes`` interpolated linearly to ``t_s``, each from the two nodes around it."""
    index = np.clip(np.searchsorted(nodes, t_s, side="right") - 1, 0, nodes.size - 2)
    step = np.mod(values[index + 1] - values[index] + np.pi, 2.0 * np.pi) - np.pi
    return values[index] + step * (t_s - nodes[index]) / (nodes[index + 1] - nodes[index])


def _apparent_sidereal_time(time_mjd_s):
    """Greenwich apparent sidereal time [rad] at ``time_mjd_s`` (MJD seconds, UTC), shape (ntime,).

    The rigorous evaluation (precession-nutation series) costs ~25 us per epoch, which adds up over the
    thousands of integrations of an observation. Sidereal time advances at a constant rate apart from the
    equation of the equinoxes, which drifts by well under a microarcsecond per minute, so it is evaluated on
    nodes every :data:`_GAST_NODE_SPACING` seconds of absolute time and interpolated linearly (error < 1e-10 rad).
    """
    t_s = np.atleast_1d(np.asarray(time_mjd_s, dtype=float))
    nodes = _absolute_nodes(t_s, _GAST_NODE_SPACING)
    times = Time(nodes / 86400.0, format="mjd", scale="utc")
    gast = times.sidereal_time("apparent", longitude=0.0 * u.deg).to_value(u.rad)
    return _between_nodes(t_s, nodes, gast)


def _directions_of_date(ra_rad, dec_rad, t_s):
    """The direction of date (true equator and equinox) [rad] of an ICRS/J2000 direction at every time of ``t_s``.

    Evaluated on nodes every :data:`_DATE_NODE_SPACING` seconds of absolute time and interpolated: the direction
    moves by under 0.1 arcsec per hour, almost linearly.

    Returns
    -------
    (ra, dec)
        Arrays of shape (ntime,).
    """
    from astropy.coordinates import TETE, SkyCoord
    nodes = _absolute_nodes(t_s, _DATE_NODE_SPACING)
    epochs = Time(nodes / 86400.0, format="mjd", scale="utc")
    of_date = SkyCoord(np.full(nodes.size, float(ra_rad)) * u.rad, np.full(nodes.size, float(dec_rad)) * u.rad,
                       frame="icrs").transform_to(TETE(obstime=epochs))
    return (_between_nodes(t_s, nodes, of_date.ra.to_value(u.rad)),
            _between_nodes(t_s, nodes, of_date.dec.to_value(u.rad)))


def apparent_direction(ra_rad, dec_rad, time_mjd_s):
    """Return the direction of date (true equator and equinox) [rad] of an ICRS/J2000 direction at the mean of the times.

    Hour angles are measured from the *apparent* sidereal time, so the source must be in the frame of date too:
    precession alone moves a J2000 position by 0.37 degrees by 2026, which shows up one-to-one in the elevation
    and, near transit of a high-elevation source, several times over in the parallactic angle. One epoch per call
    is enough: the direction of date moves by under 0.1 arcsec per hour.
    """
    from astropy.coordinates import TETE, SkyCoord
    epoch = Time(float(np.mean(np.atleast_1d(np.asarray(time_mjd_s, dtype=float)))) / 86400.0, format="mjd", scale="utc")
    of_date = SkyCoord(float(ra_rad) * u.rad, float(dec_rad) * u.rad, frame="icrs").transform_to(TETE(obstime=epoch))
    return float(of_date.ra.to_value(u.rad)), float(of_date.dec.to_value(u.rad))


def _hour_angle_and_latitude(antenna_xyz, ra_rad, time_mjd_s):
    """Return (H, lat) with H the local hour angle (nant, ntime) [rad] and lat the geocentric latitude (nant,) [rad].

    Parameters
    ----------
    antenna_xyz : array (nant, 3)
        ITRF antenna positions in metres (MS ANTENNA.POSITION).
    ra_rad : float or array (ntime,)
        Source right ascension of date in radians (see :func:`_directions_of_date`).
    time_mjd_s : array (ntime,)
        Times in MJD seconds, UTC (MS TIME).

    Returns
    -------
    tuple of np.ndarray
        Hour angle (nant, ntime) in radians wrapped to [-pi, pi] and latitude (nant,) in radians.
    """
    xyz = np.atleast_2d(np.asarray(antenna_xyz, dtype=float))
    t_s = np.atleast_1d(np.asarray(time_mjd_s, dtype=float))
    # Geocentric latitude, not geodetic: casacore's AZEL frame (which MSDerivedValues uses for the parallactic
    # angle and the elevation) takes the vertical along the ITRF position vector. The two differ by up to 0.19
    # degrees, which near the zenith becomes more than a degree of parallactic angle.
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    gast = _apparent_sidereal_time(t_s)
    hour_angle = gast[np.newaxis, :] + lon[:, np.newaxis] - np.asarray(ra_rad, dtype=float)
    hour_angle = (hour_angle + np.pi) % (2.0 * np.pi) - np.pi
    return hour_angle, lat


#: Last result of :func:`_local_geometry`, keyed by its inputs: the four angle functions below are called in a
#: row for the same antennas, source and times, and the geometry is the expensive part.
_GEOMETRY_CACHE: dict = {}


def _local_geometry(antenna_xyz, ra_rad, dec_rad, time_mjd_s):
    """Return (H, lat, dec) for an ICRS/J2000 source: hour angle (nant, ntime), geocentric latitude (nant,) and the
    declination of date (ntime,), all in radians."""
    xyz = np.atleast_2d(np.asarray(antenna_xyz, dtype=float))
    t_s = np.atleast_1d(np.asarray(time_mjd_s, dtype=float))
    key = (xyz.tobytes(), float(ra_rad), float(dec_rad), t_s.tobytes())
    # One entry, read and replaced as a whole: threads asking for other times never see each other's values.
    cached = _GEOMETRY_CACHE.get("entry")
    if cached is not None and cached[0] == key:
        return cached[1]
    ra_date, dec_date = _directions_of_date(ra_rad, dec_rad, t_s)
    hour_angle, lat = _hour_angle_and_latitude(xyz, ra_date, t_s)
    _GEOMETRY_CACHE["entry"] = (key, (hour_angle, lat, dec_date))
    return hour_angle, lat, dec_date


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
    hour_angle, lat, dec = _local_geometry(antenna_xyz, ra_rad, dec_rad, time_mjd_s)
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
    hour_angle, lat, dec = _local_geometry(antenna_xyz, ra_rad, dec_rad, time_mjd_s)
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
    hour_angle, lat, dec = _local_geometry(antenna_xyz, ra_rad, dec_rad, time_mjd_s)
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
    hour_angle, _, dec = _local_geometry(antenna_xyz, ra_rad, dec_rad, time_mjd_s)
    return np.arctan2(-np.cos(hour_angle), -np.sin(hour_angle) * np.sin(dec))


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
    rotation = parang_rotation(antenna1, antenna2, row_time, angle, vis.shape[2],
                               vis.dtype if np.iscomplexobj(vis) else np.complex128, antenna_index=antenna_index,
                               time_index=time_index)
    # Flagged samples can hold NaN/inf (empty correlator cells); rotating them is harmless.
    with np.errstate(invalid="ignore"):
        out = vis * rotation[:, np.newaxis, :]
    logger.debug(f"apply_parang: rotated {vis.shape[0]} rows, ncorr={vis.shape[2]}, nant={np.shape(angle)[0]}, "
                 f"ntime={np.shape(angle)[1]}")
    return out


def parang_rotation(antenna1, antenna2, row_time, angle, ncorr, dtype=np.complex128, antenna_index=None, *,
                    time_index=None):
    """The factor :func:`apply_parang` multiplies every row and correlation by: (nrow, ncorr) of ``dtype``.

    Same arguments as :func:`apply_parang`, with the number of correlations ``ncorr`` in place of the data.
    """
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
    phase = np.zeros((antenna1.shape[0], ncorr), dtype=float)
    phase[:, 0] = diff
    if ncorr >= 2:
        phase[:, ncorr - 1] = -diff
    if ncorr == 4:
        phase[:, 1] = total
        phase[:, 2] = -total
    return np.exp(1j * phase).astype(dtype)


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

    Read with the selected table engine (:func:`vlbipy.solvers.msio.table_engine`).

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
    from .msio import open_table
    table = open_table(f"{ms_path}/ANTENNA")
    try:
        antenna_xyz = np.asarray(table.getcol("POSITION"), dtype=float)
        mounts = [str(m) for m in table.getcol("MOUNT")]
    finally:
        table.close()
    table = open_table(f"{ms_path}/FIELD")
    try:
        phase_dir = np.asarray(table.getcol("PHASE_DIR"))[int(field_id)]  # (npoly, 2)
    finally:
        table.close()
    ra_rad, dec_rad = float(phase_dir[0, 0]), float(phase_dir[0, 1])
    logger.debug(f"ms_parang_inputs: {len(mounts)} antennas, field {field_id} at ra={ra_rad:.6f} dec={dec_rad:.6f} rad")
    return {"antenna_xyz": antenna_xyz, "mounts": mounts, "ra_rad": ra_rad, "dec_rad": dec_rad}
