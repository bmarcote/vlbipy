"""Ionospheric (TEC) correction table from global ionosphere maps, as AIPS TECOR computes it (dask-ms backend).

:func:`write_tec_table` reproduces the dispersive part of AIPS ``TECOR`` (run by the ``VLBATECR`` procedure,
which also fetches the maps; see :mod:`vlbipy.ionex` for the maps, their names and their interpolation) and
writes it as an ordinary CASA ``Fringe Jones`` table, which the native apply (:mod:`vlbipy.solvers.apply`) and
CASA's ``applycal``/``fringefit`` both apply without any special code.

What TECOR does, and this module repeats (``TECCOR``, ``GETSIP``, ``COOELV``, ``IONTEC``):

* the source direction of date and the antenna's geodetic latitude/longitude (WGS84) give its azimuth and
  elevation; the line of sight meets a thin shell at the IONEX height (``HGT1``, 450 km) above a sphere of radius
  6378 km at the pierce point (spherical sine and cosine rules of ``GETSIP``);
* the vertical TEC at the pierce point comes from the maps (bilinear in space, linear in time between maps
  rotated with the Sun, ``rotation`` = AIPS ``APARM(2)``);
* the slant TEC is ``TEC x scale / sqrt(1 - (R / (R + H + dH))^2 cos^2(a EL))``, the thin-shell mapping
  ``1 / cos(ZA at the shell)`` generalised by Petrov (2023, AJ 165, 183): ``scale``, ``height_offset_km`` and
  ``elevation_factor`` are AIPS ``APARM(4)``, ``APARM(5)``, ``APARM(6)`` (defaults 1, 0, 1 = plain thin shell;
  :data:`PETROV_MAPPING` holds Petrov's 0.85, 56.7 km, 0.9782);
* TECOR writes the dispersive delay ``40.28 sTEC / c^3`` [s at 1 m wavelength] into the CL table (DISP column)
  and the ionospheric Faraday rotation ``2.6e-17 B_los sTEC`` [rad/m^2] (IGRF-14 or a dipole field). Only the
  dispersive term is written here: the Faraday rotation (R and L hands rotated oppositely) is not.

The correction (one antenna, slant TEC ``sTEC`` in el/m^2, ``K = e^2 / (8 pi^2 eps0 m_e) = 40.308 m^3/s^2``):

    theta(f) = +2 pi K sTEC / (c f)        [rad, gain phase, V_ij = g_i conj(g_j) V, g = exp(i theta)]

evaluated at every channel's own frequency, for each antenna with its own pierce point and each field with its own
direction. The sign is the one the data need (:data:`IONOSPHERE_SIGN`): on RSM07 (EVN, 1.6 GHz) this table removes
the ionospheric delays the fringe fit finds (HH: their 18 ns drift over the run drops to 2.3 ns), whereas CASA's
``F Jones`` (``gencal(caltype='tecim')``, ``del = -8.4483e-7 tec / f``) has the opposite sign and doubles them. In
the fringe-fit convention the group delay of the ionosphere therefore appears as a negative delay. A Fringe Jones
row holds the phase exactly: with ``C / f`` the phase, ``a, b`` the lowest and highest channel frequency of the
subband and ``f_ref`` the frequency of the table's subband,

    disp = C / (2 pi 1e6)                   (CASA dispersive unit: k_disp(f) = 2 pi 1e6 (1/f + (f - a - b) / (a b)))
    delay = -1e9 C / (2 pi a b)   [ns]
    phase = -C (f_ref - a - b) / (a b)      (wrapped to (-pi, pi])

so that ``phase + 2 pi 1e-9 delay (f - f_ref) + disp k_disp(f) = C / f`` for every ``f``. Both CASA
(``FringeJones::calcAllJones``) and the native apply take ``a`` and ``b`` from the channels being calibrated, so
the identity is exact on the whole subband; applied to a channel selection (the priors of a fringe fit on the
inner channels) it is off by a nearly constant phase (:func:`selection_error`: 1e-3 rad per 10 TECU for 56 of 64
channels). CASA's ``applycal`` and the native apply agree on these tables to 3e-5 rad (single precision).

Rows: one per antenna, subband and time, for every (field, scan) of the data and every antenna of the ANTENNA
table, at the first and last integration of the scan and every ``interval`` seconds between, with the field's
own FIELD_ID, so a field's integrations always lie between rows of its own scan. The rate of a row carries the
time derivative of its phase: CASA (``CTRateAwareTimeInterp1``) and the native apply advance each row's phase by
its rate and average the two around a time as unit phasors (delay and dispersive term linearly), which follows a
phase changing by radians per minute (low elevations) without ambiguity. Rows are never flagged. Deliberate
differences from TECOR: the times are a regular grid instead of the CL table's, the sidereal time and direction
of date come from astropy, and maps missing a value at a pierce point make the task fail instead of blanking the
row.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..ionex import TECU, IonexMaps, fetch_ionex, merge_maps, read_ionex, vertical_tec
from ..logging_utils import get_logger
from .caltable import write_fringe_table
from .msio import read_layout, read_setup, table_engine

logger = get_logger()

#: e^2 / (8 pi^2 eps0 m_e) [m^3 s^-2]: the ionospheric refractive index is 1 - K n_e / f^2 (CODATA 2018).
K_IONO = 40.30819293981814
SPEED_OF_LIGHT = 299792458.0
#: Earth radius of AIPS GETSIP and of the TECOR mapping function [m].
AIPS_EARTH_RADIUS_M = 6378000.0
#: Sign of the ionospheric gain phase ``+-2 pi K sTEC / (c f)`` in the convention V_ij = g_i conj(g_j) V of the
#: native apply and of CASA's FringeJones (both remove ``theta_i - theta_j``). +1: measured on RSM07, where it
#: removes the ionospheric delays found by the fringe fit; CASA's F Jones (gencal 'tecim') uses -1 and doubles them.
IONOSPHERE_SIGN = 1.0
#: CASA's dispersive parameter unit: k_disp(f) = DISP_SCALE 2 pi (1/f + ...) (FringeJones KDISPSCALE).
DISP_SCALE = 1.0e6
#: Petrov (2023) mapping (AIPS APARM(4)-(6)): TEC scale, shell height increment [km], elevation factor.
PETROV_MAPPING = {"scale": 0.85, "height_offset_km": 56.7, "elevation_factor": 0.9782}
#: SNR written into every row (CASA gencal writes 100 for its F Jones table).
ROW_SNR = 100.0
#: Interpolation error [rad] (halfway between rows, at the lowest frequency) above which a warning suggests a
#: shorter interval.
MAX_INTERP_ERROR = 0.1
#: Half-step [s] of the central difference that gives the slant-TEC rate of every row.
RATE_STEP = 5.0


# ----------------------------------------------------------------------------------------------------------------
# Geometry (AIPS COOELV, GETSIP and the TECCOR mapping function)
# ----------------------------------------------------------------------------------------------------------------
def antenna_geodetic(antenna_xyz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 geodetic (latitude [rad], east longitude [rad], height [m]) of ITRF positions (nant, 3) [m]."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    xyz = np.atleast_2d(np.asarray(antenna_xyz, dtype=np.float64))
    location = EarthLocation.from_geocentric(xyz[:, 0] * u.m, xyz[:, 1] * u.m, xyz[:, 2] * u.m)
    geodetic = location.to_geodetic("WGS84")
    return (np.asarray(geodetic.lat.to_value(u.rad)), np.asarray(geodetic.lon.to_value(u.rad)),
            np.asarray(geodetic.height.to_value(u.m)))


def apparent_sidereal_time(time_mjd_s) -> np.ndarray:
    """Greenwich apparent sidereal time [rad] at times [MJD seconds, UTC]."""
    from astropy import units as u
    from astropy.time import Time
    times = Time(np.atleast_1d(np.asarray(time_mjd_s, dtype=np.float64)) / 86400.0, format="mjd", scale="utc")
    return np.asarray(times.sidereal_time("apparent", longitude=0.0 * u.deg).to_value(u.rad))


def source_azel(lat, lon, ra_date, dec_date, gast) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth (north through east) and elevation [rad], (nant, ntime), of a source of date (AIPS COOELV).

    Parameters
    ----------
    lat, lon : arrays (nant,)
        Geodetic latitude and east longitude of the antennas [rad].
    ra_date, dec_date : float
        Apparent (true equator and equinox of date) right ascension and declination [rad].
    gast : array (ntime,)
        Greenwich apparent sidereal time [rad].
    """
    lat = np.asarray(lat, dtype=np.float64)[:, None]
    hour_angle = np.asarray(gast, dtype=np.float64)[None, :] + np.asarray(lon, dtype=np.float64)[:, None] - ra_date
    sin_el = np.sin(lat) * np.sin(dec_date) + np.cos(lat) * np.cos(dec_date) * np.cos(hour_angle)
    elevation = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    north = np.sin(dec_date) * np.cos(lat) - np.cos(dec_date) * np.sin(lat) * np.cos(hour_angle)
    east = -np.cos(dec_date) * np.sin(hour_angle)
    return np.mod(np.arctan2(east, north), 2.0 * np.pi), elevation


def pierce_points(lat, azimuth, elevation, height_m, radius_m: float = AIPS_EARTH_RADIUS_M):
    """Where the line of sight crosses the thin shell (AIPS GETSIP, spherical Earth).

    Parameters
    ----------
    lat : array broadcastable to ``azimuth``
        Latitude of the antenna [rad].
    azimuth, elevation : arrays
        Direction of the source at the antenna [rad].
    height_m : float
        Shell height [m].

    Returns
    -------
    (lat_ipp, dlon, za_ipp)
        Latitude of the pierce point [rad], its longitude offset from the antenna [rad, east positive] and the
        zenith angle of the line of sight there [rad].
    """
    zenith = 0.5 * np.pi - np.asarray(elevation, dtype=np.float64)
    za_ipp = np.arcsin(radius_m * np.sin(zenith) / (radius_m + height_m))
    theta = zenith - za_ipp
    lat = np.asarray(lat, dtype=np.float64)
    lat_ipp = np.arcsin(np.sin(lat) * np.cos(theta) + np.cos(lat) * np.sin(theta) * np.cos(azimuth))
    dlon = np.arcsin(np.clip(np.sin(azimuth) * np.sin(theta) / np.cos(lat_ipp), -1.0, 1.0))
    return lat_ipp, dlon, za_ipp


def slant_factor(elevation, height_m, *, scale: float = 1.0, height_offset_km: float = 0.0,
                 elevation_factor: float = 1.0, radius_m: float = AIPS_EARTH_RADIUS_M) -> np.ndarray:
    """TECOR's mapping function: slant / vertical TEC at ``elevation`` [rad] (AIPS APARM(4)-(6))."""
    ratio = radius_m / (radius_m + height_m + 1000.0 * height_offset_km)
    return scale / np.sqrt(1.0 - (ratio * np.cos(elevation_factor * np.asarray(elevation, dtype=np.float64))) ** 2)


def slant_tec(maps: IonexMaps, antenna_xyz, ra_rad: float, dec_rad: float, time_mjd_s, *,
              height_km: Optional[float] = None, rotation: float = 1.0, scale: float = 1.0,
              height_offset_km: float = 0.0, elevation_factor: float = 1.0) -> dict:
    """Slant TEC towards one source for every antenna and time, as TECOR computes it.

    Parameters
    ----------
    maps : IonexMaps
    antenna_xyz : array (nant, 3)
        ITRF antenna positions [m].
    ra_rad, dec_rad : float
        Source direction (ICRS/J2000, the MS FIELD PHASE_DIR) [rad].
    time_mjd_s : array (ntime,)
        Times [MJD seconds, UTC].
    height_km : float, optional
        Shell height; the maps' ``HGT1`` when None.
    rotation, scale, height_offset_km, elevation_factor : float
        AIPS APARM(2), APARM(4), APARM(5), APARM(6).

    Returns
    -------
    dict
        Arrays (nant, ntime): ``stec`` and ``vtec`` [TECU], ``elevation`` and ``azimuth`` [rad], ``lat_ipp`` and
        ``lon_ipp`` [deg].
    """
    from .parang import apparent_direction
    times = np.atleast_1d(np.asarray(time_mjd_s, dtype=np.float64))
    height_m = 1000.0 * float(maps.height_km if height_km is None else height_km)
    lat, lon, _ = antenna_geodetic(antenna_xyz)
    ra_date, dec_date = apparent_direction(ra_rad, dec_rad, times)
    azimuth, elevation = source_azel(lat, lon, ra_date, dec_date, apparent_sidereal_time(times))
    lat_ipp, dlon, _ = pierce_points(lat[:, None], azimuth, elevation, height_m)
    lat_deg = np.degrees(lat_ipp)
    lon_deg = np.mod(np.degrees(lon[:, None] + dlon) + 180.0, 360.0) - 180.0
    vtec = vertical_tec(maps, lat_deg, lon_deg, np.broadcast_to(times, lat_deg.shape), rotation=rotation)
    factor = slant_factor(elevation, height_m, scale=scale, height_offset_km=height_offset_km,
                          elevation_factor=elevation_factor)
    return {"stec": vtec * factor, "vtec": vtec, "elevation": elevation, "azimuth": azimuth, "lat_ipp": lat_deg,
            "lon_ipp": lon_deg}


# ----------------------------------------------------------------------------------------------------------------
# The phase model and its Fringe Jones representation
# ----------------------------------------------------------------------------------------------------------------
def ionospheric_phase(stec_tecu, freq_hz) -> np.ndarray:
    """Gain phase [rad] of the ionosphere, ``IONOSPHERE_SIGN 2 pi K sTEC / (c f)`` (broadcast over both inputs)."""
    return _coefficient(stec_tecu) / np.asarray(freq_hz, dtype=np.float64)


def _coefficient(stec_tecu) -> np.ndarray:
    """``C`` [rad Hz] of the ionospheric phase ``C / f`` for a slant TEC [TECU] (or a TEC rate [TECU/s])."""
    return IONOSPHERE_SIGN * 2.0 * np.pi * K_IONO * TECU * np.asarray(stec_tecu, dtype=np.float64) / SPEED_OF_LIGHT


def fringe_parameters(stec_tecu, fmin_hz: float, fmax_hz: float, f_ref_hz: float):
    """Fringe Jones (phase [rad], delay [ns], dispersive [CASA units]) whose phase model equals
    :func:`ionospheric_phase` at every frequency of a subband spanning ``[fmin_hz, fmax_hz]``.

    With ``C / f`` the phase: ``disp = C / (2 pi 1e6)``, ``delay = -1e9 C / (2 pi a b)`` and
    ``phase = -C (f_ref - a - b) / (a b)``, wrapped to (-pi, pi]. Arrays have the shape of ``stec_tecu``.
    """
    a, b = float(fmin_hz), float(fmax_hz)
    coefficient = _coefficient(stec_tecu)
    phase = -coefficient * (float(f_ref_hz) - a - b) / (a * b)
    delay_ns = -1e9 * coefficient / (2.0 * np.pi * a * b)
    disp = coefficient / (2.0 * np.pi * DISP_SCALE)
    return np.angle(np.exp(1j * phase)), delay_ns, disp


def phase_rate(stec_rate, fmin_hz: float, fmax_hz: float, f_ref_hz: float) -> np.ndarray:
    """Fringe Jones rate [s/s] carrying the time derivative of the phase of :func:`fringe_parameters`:
    ``d(phase)/dt / (2 pi f_ref)`` for a slant TEC changing by ``stec_rate`` [TECU/s]."""
    a, b = float(fmin_hz), float(fmax_hz)
    return -_coefficient(stec_rate) * (float(f_ref_hz) - a - b) / (a * b) / (2.0 * np.pi * float(f_ref_hz))


def selection_error(stec_tecu, fmin_hz, fmax_hz, sel_fmin_hz, sel_fmax_hz, freq_hz) -> np.ndarray:
    """Phase error [rad] at ``freq_hz`` when rows made for ``[fmin, fmax]`` are applied to the channels
    ``[sel_fmin, sel_fmax]``: the k_disp of CASA and of the native apply then uses the selection's edges."""
    from .fringe import k_disp
    disp = fringe_parameters(stec_tecu, fmin_hz, fmax_hz, 0.5 * (fmin_hz + fmax_hz))[2]
    return disp * (k_disp(freq_hz, sel_fmin_hz, sel_fmax_hz) - k_disp(freq_hz, fmin_hz, fmax_hz))


# ----------------------------------------------------------------------------------------------------------------
# The task
# ----------------------------------------------------------------------------------------------------------------
def row_times(t_min: float, t_max: float, interval: float) -> np.ndarray:
    """Row times of one scan: ``t_min``, every ``interval`` seconds after it, and ``t_max``."""
    if interval <= 0 or t_max <= t_min:
        return np.unique([t_min, t_max])
    inner = t_min + interval * np.arange(1, int(np.ceil((t_max - t_min) / interval)))
    return np.concatenate([[t_min], inner[inner < t_max], [t_max]])


def _field_ids(fields, field_names: list[str], observed) -> list[int]:
    """FIELD_IDs to correct: ``fields`` (names or ids, a list or comma string; None = every observed field)."""
    if fields is None or (isinstance(fields, str) and not fields.strip()):
        return sorted(int(f) for f in observed)
    tokens = [t.strip() for t in fields.split(",")] if isinstance(fields, str) else list(fields)
    ids = []
    for token in tokens:
        text = str(token).strip()
        if text.isdigit():
            ids.append(int(text))
        elif text in field_names:
            ids.append(field_names.index(text))
        else:
            raise ValueError(f"field {token!r} is not in the measurement set ({', '.join(field_names)})")
    return sorted(set(ids) & {int(f) for f in observed})


def _scan_ranges(layout: dict, field_ids: list[int]) -> list[tuple[int, int, float, float]]:
    """(field, scan, first time, last time) of every observed (field, scan) of ``field_ids``, in time order."""
    ranges = {}
    for field, scan, t_min, t_max in zip(layout["field"], layout["scan"], layout["t_min"], layout["t_max"]):
        if int(field) not in field_ids:
            continue
        key = (int(field), int(scan))
        low, high = ranges.get(key, (np.inf, -np.inf))
        ranges[key] = (min(low, float(t_min)), max(high, float(t_max)))
    return sorted(((f, s, lo, hi) for (f, s), (lo, hi) in ranges.items()), key=lambda item: item[2])


def _simultaneous_fields(ranges: list[tuple[int, int, float, float]]) -> list[tuple[int, int]]:
    """Pairs of different fields whose scans overlap in time (one MS with several phase centres)."""
    pairs = set()
    for i, (field_a, _, lo_a, hi_a) in enumerate(ranges):
        for field_b, _, lo_b, hi_b in ranges[i + 1:]:
            if lo_b > hi_a:
                break
            if field_b != field_a:
                pairs.add(tuple(sorted((field_a, field_b))))
    return sorted(pairs)


def load_maps(t_min: float, t_max: float, *, ionex_dir, model: str = "emrg", ionex_files=None,
              fallback: bool = True) -> tuple[str, IonexMaps]:
    """The merged maps covering ``[t_min, t_max]``: ``ionex_files`` when given, else downloaded into
    ``ionex_dir`` (:func:`vlbipy.ionex.fetch_ionex`). Returns (product, maps)."""
    if ionex_files:
        files, product = [Path(f) for f in ionex_files], "user"
    else:
        product, files = fetch_ionex(t_min, t_max, ionex_dir, model=model, fallback=fallback)
    maps = merge_maps([read_ionex(f) for f in files])
    if t_min < maps.epochs[0] or t_max > maps.epochs[-1]:
        raise ValueError(f"IONEX maps ({', '.join(Path(f).name for f in files)}) cover MJD s {maps.epochs[0]:.0f}"
                         f"-{maps.epochs[-1]:.0f}, not the data ({t_min:.0f}-{t_max:.0f})")
    return product, maps


def _midpoint_error(stec_mid, early: dict, late: dict, step, fmin_hz: float, fmax_hz: float, f_ref_hz: float) -> float:
    """Largest phase error [rad] at ``fmin_hz`` halfway between pairs of consecutive rows, interpolated as the
    native apply and CASA do (phases advanced by their own rates and averaged as unit phasors, delay and dispersive
    term averaged), against the exact phase from the slant TEC ``stec_mid`` at the midpoints [TECU].

    ``early`` and ``late`` hold the ``stec`` [TECU] and ``stec_rate`` [TECU/s] of the two rows of every pair (all
    arrays (nant, npair)); ``step`` (npair,) is the time between them [s].
    """
    from .fringe import k_disp
    half = 0.5 * np.asarray(step, dtype=np.float64)
    turn = 2.0 * np.pi * f_ref_hz
    phasors, delays, disps = [], [], []
    for row, sign in ((early, 1.0), (late, -1.0)):
        phase, delay, disp = fringe_parameters(row["stec"], fmin_hz, fmax_hz, f_ref_hz)
        rate = phase_rate(row["stec_rate"], fmin_hz, fmax_hz, f_ref_hz)
        phasors.append(np.exp(1j * (phase + sign * turn * rate * half)))
        delays.append(delay)
        disps.append(disp)
    blended = (np.angle(phasors[0] + phasors[1]) + 2e-9 * np.pi * 0.5 * (delays[0] + delays[1]) * (fmin_hz - f_ref_hz)
               + 0.5 * (disps[0] + disps[1]) * k_disp(fmin_hz, fmin_hz, fmax_hz))
    return float(np.abs(np.angle(np.exp(1j * (blended - ionospheric_phase(stec_mid, fmin_hz))))).max())


def _field_rows(maps: IonexMaps, setup: dict, field_id: int, scans: list, interval: float, rates: bool,
                options: dict, spw_low: tuple) -> dict:
    """Rows of one field: times (ntime,), scans (ntime,), slant TEC and its rate (ntime, nant) [TECU, TECU/s],
    the vertical TEC range and the largest midpoint interpolation error at the lowest frequency [rad].

    Raises
    ------
    ValueError
        When the maps have no value at some pierce point.
    """
    xyz, (ra, dec) = setup["antenna_xyz"], setup["field_dirs"][field_id]
    grids = [row_times(lo, hi, interval) for _, _, lo, hi in scans]
    times = np.concatenate(grids)
    geometry = slant_tec(maps, xyz, ra, dec, times, **options)
    stec = geometry["stec"]                                                          # (nant, ntime)
    if not np.isfinite(stec).all():
        ant, when = np.nonzero(~np.isfinite(stec))
        raise ValueError(f"IONEX maps give no TEC for field {setup['field_names'][field_id]}, antennas "
                         f"{sorted({setup['antenna_names'][a] for a in ant})} at {when.size} time(s)")
    stec_rate = np.zeros_like(stec)
    if rates:
        around = slant_tec(maps, xyz, ra, dec, np.concatenate([times - RATE_STEP, times + RATE_STEP]), **options)
        stec_rate = (around["stec"][:, times.size:] - around["stec"][:, :times.size]) / (2.0 * RATE_STEP)
    edges = np.cumsum([0] + [grid.size for grid in grids])
    first = np.asarray([k for k in range(times.size - 1) if k + 1 not in edges], dtype=np.int64)   # pairs in a scan
    error = 0.0
    if first.size:
        stec_mid = slant_tec(maps, xyz, ra, dec, 0.5 * (times[first] + times[first + 1]), **options)["stec"]
        early = {"stec": stec[:, first], "stec_rate": stec_rate[:, first]}
        late = {"stec": stec[:, first + 1], "stec_rate": stec_rate[:, first + 1]}
        error = _midpoint_error(stec_mid, early, late, times[first + 1] - times[first], *spw_low)
    scan_col = np.concatenate([np.full(grid.size, scan) for (_, scan, _, _), grid in zip(scans, grids)])
    return {"time": times, "scan": scan_col, "stec": stec.T, "stec_rate": stec_rate.T,
            "vtec": (float(np.nanmin(geometry["vtec"])), float(np.nanmax(geometry["vtec"]))), "error": error}


def write_tec_table(vis, caltable, *, ionex_dir, model: str = "emrg", shell_height: Optional[float] = None,
                    interval: float = 30.0, fields=None, rotation: float = 1.0, scale: float = 1.0,
                    height_offset_km: float = 0.0, elevation_factor: float = 1.0, ionex_files=None,
                    fallback: bool = True, rates: bool = True, engine: Optional[str] = None) -> dict:
    """Write the ionospheric correction of ``vis`` as a CASA Fringe Jones table at ``caltable``.

    Parameters
    ----------
    vis : str or Path
        Measurement set (or Multi-MS).
    caltable : str or Path
        Output table (replaced if it exists).
    ionex_dir : str or Path
        Where the IONEX files are cached / downloaded.
    model : str
        IONEX product (AIPS TECRTYPE, default ``emrg`` like VLBATECR); see :data:`vlbipy.ionex.PRODUCTS`.
    shell_height : float, optional
        Shell height [km]; the maps' ``HGT1`` when None.
    interval : float
        Row spacing inside a scan [s] (plus the first and last integration of every scan).
    fields : list or str, optional
        Fields (names or ids) to correct; None = every field with data.
    rotation, scale, height_offset_km, elevation_factor : float
        AIPS TECOR APARM(2), APARM(4), APARM(5), APARM(6) (see :data:`PETROV_MAPPING`).
    ionex_files : list of paths, optional
        Use these IONEX files instead of downloading.
    fallback : bool
        Try the other products when ``model`` is not available.
    rates : bool
        Write the phase rate of every row (robust interpolation of fast phase changes); False writes rate 0.
    engine : str, optional
        Table engine for reading ``vis`` (:func:`vlbipy.solvers.msio.table_engine` when None).

    Returns
    -------
    dict
        ``caltable``, ``product``, ``ionex_files``, ``shell_height_km``, ``nrow``, ``fields`` (names),
        ``simultaneous_fields`` (pairs of field names observed at the same time: they need the table applied
        per field with its own ``gainfield``), ``stec_range`` and ``vtec_range`` [TECU], ``interp_error``
        [rad, largest error of the interpolated phase halfway between rows at the lowest frequency],
        ``elapsed_s``.
    """
    started = _time.time()
    engine = engine or table_engine()
    setup = read_setup(vis, engine=engine)
    layout = read_layout(vis, engine=engine)
    field_ids = _field_ids(fields, setup["field_names"], np.unique(layout["field"]))
    if not field_ids:
        raise ValueError(f"{vis}: no data for fields {fields!r}")
    ranges = _scan_ranges(layout, field_ids)
    t_min, t_max = min(r[2] for r in ranges), max(r[3] for r in ranges)
    product, maps = load_maps(t_min, t_max, ionex_dir=ionex_dir, model=model, ionex_files=ionex_files,
                              fallback=fallback)
    height_km = float(maps.height_km if shell_height is None else shell_height)
    chan_freq = setup["chan_freq"]
    nspw, nant = chan_freq.shape[0], len(setup["antenna_names"])
    fmin, fmax = chan_freq.min(axis=1), chan_freq.max(axis=1)
    f_ref = chan_freq[:, chan_freq.shape[1] // 2]
    widths = np.abs(fmax - fmin) * chan_freq.shape[1] / max(chan_freq.shape[1] - 1, 1)
    low = int(np.argmin(fmin))
    options = dict(height_km=height_km, rotation=rotation, scale=scale, height_offset_km=height_offset_km,
                   elevation_factor=elevation_factor)
    observed = sorted({r[0] for r in ranges})
    per_field = [_field_rows(maps, setup, field_id, [r for r in ranges if r[0] == field_id], interval, rates,
                             options, (fmin[low], fmax[low], f_ref[low])) for field_id in observed]
    field_col = np.concatenate([np.full(rows["time"].size, f) for f, rows in zip(observed, per_field)])
    times = np.concatenate([rows["time"] for rows in per_field])
    order = np.argsort(times, kind="stable")
    times, field_col = times[order], field_col[order]
    scan_col = np.concatenate([rows["scan"] for rows in per_field])[order]
    stec = np.concatenate([rows["stec"] for rows in per_field])[order]                  # (ntime, nant)
    stec_rate = np.concatenate([rows["stec_rate"] for rows in per_field])[order]

    # Rows ordered by time, then subband, then antenna (as the fringe fit writes them); both hands alike.
    ntime = times.size
    fparam = np.zeros((ntime, nspw, nant, 8), dtype=np.float64)
    for spw in range(nspw):
        phase, delay, disp = fringe_parameters(stec, fmin[spw], fmax[spw], f_ref[spw])
        rate = phase_rate(stec_rate, fmin[spw], fmax[spw], f_ref[spw])
        for hand in range(2):
            fparam[:, spw, :, 4 * hand:4 * hand + 4] = np.stack([phase, delay, rate, disp], axis=-1)
    nrow = ntime * nspw * nant
    path = write_fringe_table(
        caltable, vis, times=np.repeat(times, nspw * nant), field_ids=np.repeat(field_col, nspw * nant),
        spw_ids=np.tile(np.repeat(np.arange(nspw), nant), ntime), antenna_ids=np.tile(np.arange(nant), ntime * nspw),
        refant_id=-1, scan_numbers=np.repeat(scan_col, nspw * nant), fparam=fparam.reshape(nrow, 8),
        paramerr=np.zeros((nrow, 8)), flag=np.zeros((nrow, 8), dtype=bool), snr=np.full((nrow, 8), ROW_SNR),
        intervals=np.full(nrow, float(interval)), spw_chan_freq=f_ref, spw_chan_width=widths)
    names = setup["field_names"]
    simultaneous = [(names[a], names[b]) for a, b in _simultaneous_fields(ranges)]
    if simultaneous:
        logger.info("tec: fields observed at the same time {}: {} must be applied per field with its own "
                    "gainfield (the pipeline does)", simultaneous, path.name)
    error = max(rows["error"] for rows in per_field)
    if error > MAX_INTERP_ERROR:
        logger.warning("tec: the phase interpolated between rows is off by up to {:.2f} rad at {:.0f} MHz; use a "
                       "shorter interval than {} s", error, fmin[low] / 1e6, interval)
    report = {"caltable": str(path), "product": product, "ionex_files": list(maps.files),
              "shell_height_km": height_km, "nrow": int(nrow), "fields": [names[f] for f in sorted(set(field_col))],
              "simultaneous_fields": simultaneous, "stec_range": [float(stec.min()), float(stec.max())],
              "vtec_range": [min(r["vtec"][0] for r in per_field), max(r["vtec"][1] for r in per_field)],
              "interp_error": error, "elapsed_s": _time.time() - started}
    logger.info("tec[native]: {} -> {} ({} rows, {} field(s), {} time(s); IONEX {} shell {:.0f} km; slant TEC "
                "{:.1f}-{:.1f} TECU; interpolation error <= {:.3f} rad; {:.1f} s)", Path(vis).name, path.name, nrow,
                len(report["fields"]), ntime, product, height_km, report["stec_range"][0], report["stec_range"][1],
                error, report["elapsed_s"])
    return report
