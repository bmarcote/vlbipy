"""Global ionosphere maps in IONEX format: product names, download, parsing and interpolation (AIPS TECOR).

What AIPS does (TECOR, the VLBATECR procedure of VLBAUTIL, 31DEC26) and this module repeats:

* **Products** (``TECRTYPE``): a four-letter code per analysis centre and solution type, ``emrg`` (Space
  Weather Canada, AIPS's default), ``jplg``, ``codg``, ``esag``, ``igsg``, ``upcg``, ``casg`` (final
  solutions) and the rapid ``igrg``, ``jprg``, ``corg``, ``esrg``, ``uqrg``, ``uhrg``, ``uprg``. Files live in
  CDDIS ``gnss/products/ionex/YYYY/DDD/`` (anonymous FTP over TLS). Until a per-centre date (``TECRVERS``) they
  are called ``<code>DDD0.YYi.Z`` (unix-compressed); since then ``<CEN>0OPSFIN_YYYYDDD0000_01D_<SMP>_GIM.INX.gz``
  (``OPSRAP`` for rapid) with the map interval ``SMP`` of the centre (01H for COD/EMR/ESA, 02H for IGS/JPL/UPC,
  30M for CAS). VLBATECR tries the other interval when the first is missing; here the old name is tried too.
* **Days**: one file per UTC day covering 00:00-24:00; an observation crossing midnight needs every day it
  touches. Consecutive files repeat the midnight map: AIPS keeps the first one read (the earlier day's).
* **Interpolation** (``IONTEC``/``TECVAL``): bilinear in latitude and longitude on the map grid, and linear in
  time between the two maps around the time, each map first rotated in longitude by the time to it (IONEX 1.0
  eq. 3, Schaer et al. 1998: the ionosphere stays roughly fixed with respect to the Sun). ``rotation`` is
  AIPS's ``APARM(2)``: the fraction of that rotation (1 = full, 0 = the ionosphere turns with the Earth).
* **Shell height**: ``HGT1`` of the header (AIPS refuses ``HGT1 != HGT2``); TEC values are ``value x
  10**EXPONENT`` TECU, 9999 marks a missing value.

Deviations: a latitude beyond the grid (the polar caps of a global map) takes the edge row instead of failing,
and a time equal to the last map is accepted. Nothing here reads a measurement set; :mod:`vlbipy.solvers.tec_task`
does the geometry.
"""
from __future__ import annotations

import datetime as dt
import gzip
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .logging_utils import get_logger
from .tools import EOP_CDDIS_EMAIL as CDDIS_EMAIL

logger = get_logger()

#: Electrons per square metre in one TEC unit.
TECU = 1.0e16
#: CDDIS anonymous FTP-over-TLS archive of IONEX files (``YYYY/DDD/<file>`` below it).
CDDIS_IONEX_URL = "ftp://gdc.cddis.eosdis.nasa.gov/gnss/products/ionex"
#: curl exit codes meaning CDDIS cannot be reached at all (resolve, connect, timeout, TLS, receive): no other file
#: is asked from it in the same :func:`fetch_ionex` call.
CURL_UNREACHABLE = (6, 7, 28, 35, 56)
#: JPL's own HTTPS copy of its final maps (``y<YYYY>/<file>``), tried for the JPL products when CDDIS fails.
JPL_FINAL_URL = "https://sideshow.jpl.nasa.gov/pub/iono_daily/IONEX_final"
#: AIPS TECRTYPE code -> (centre of the long name, solution type, map interval, first (year, day) of long names).
#: ``None`` as date: the product kept the short name (AIPS VLBATECR/TECRVERS, CASA tec_maps for the rapid ones).
PRODUCTS = {"emrg": ("EMR", "FIN", "01H", (2023, 64)), "jplg": ("JPL", "FIN", "02H", (2023, 220)),
            "codg": ("COD", "FIN", "01H", (2022, 331)), "esag": ("ESA", "FIN", "01H", (2022, 331)),
            "igsg": ("IGS", "FIN", "02H", (2022, 331)), "upcg": ("UPC", "FIN", "02H", (2023, 272)),
            "casg": ("CAS", "FIN", "30M", (2023, 1)), "igrg": ("IGS", "RAP", "02H", (2022, 331)),
            "jprg": ("JPL", "RAP", "02H", (2023, 221)), "corg": ("COD", "RAP", "01H", (2022, 331)),
            "esrg": ("ESA", "RAP", "01H", (2022, 331)), "uqrg": ("UPC", "RAP", "15M", None),
            "uhrg": ("UPC", "RAP", "01H", None), "uprg": ("UPC", "RAP", "02H", None)}
#: Products tried, in order, after the requested one: the final solutions, then the rapid ones.
FALLBACK_ORDER = ("emrg", "igsg", "jplg", "codg", "esag", "upcg", "casg", "igrg", "jprg", "corg", "esrg", "uqrg")
#: Seconds from the MJD epoch to the Unix epoch.
_MJD_UNIX_SECONDS = 40587.0 * 86400.0


@dataclass
class IonexMaps:
    """Vertical TEC maps read from one or more IONEX files.

    Attributes
    ----------
    epochs : float array (nmap,)
        Map epochs [MJD seconds, UTC], strictly increasing.
    lat, lon : float arrays (nlat,), (nlon,)
        Grid latitudes and east longitudes [deg], both increasing.
    tec : float array (nmap, nlat, nlon)
        Vertical TEC [TECU]; NaN where the file has no value (9999).
    rms : float array (nmap, nlat, nlon) or None
        RMS maps [TECU] when the files carry them.
    height_km : float
        Height of the single-layer shell (header ``HGT1``) [km].
    base_radius_km : float
        Mean Earth radius of the maps (header ``BASE RADIUS``) [km].
    mapping_function : str
        Header ``MAPPING FUNCTION`` (informative: TECOR uses its own thin-shell mapping).
    files : list of str
        Files the maps were read from.
    """

    epochs: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    tec: np.ndarray
    rms: Optional[np.ndarray]
    height_km: float
    base_radius_km: float
    mapping_function: str = ""
    files: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------------------------
# File names and download
# ----------------------------------------------------------------------------------------------------------------
def ionex_file_names(model: str, year: int, doy: int) -> list[str]:
    """Candidate archive names of product ``model`` for one UTC day, the expected naming convention first.

    Parameters
    ----------
    model : str
        AIPS TECRTYPE code (a key of :data:`PRODUCTS`, case-insensitive; a trailing letter other than ``g`` is
        replaced by ``g`` as AIPS does for the final codes it knows).
    year, doy : int
        Year and day of year.

    Returns
    -------
    list of str
        Long names (with the centre's map interval and, as VLBATECR does, the other usual interval) and the
        short ``<code>DDD0.YYi.Z`` name, ordered by the naming the date calls for.
    """
    code = str(model).strip().lower()
    if code not in PRODUCTS:
        raise ValueError(f"unknown IONEX product {model!r}; known: {', '.join(sorted(PRODUCTS))}")
    centre, kind, sampling, since = PRODUCTS[code]
    short = f"{code}{doy:03d}0.{year % 100:02d}i.Z"
    samplings = [sampling] + [other for other in ("01H", "02H") if other != sampling]
    long_names = [f"{centre}0OPS{kind}_{year:04d}{doy:03d}0000_01D_{smp}_GIM.INX.gz" for smp in samplings]
    if since is not None and (year, doy) >= tuple(since):
        return long_names + [short]
    return [short] + long_names


def observation_days(t_min: float, t_max: float) -> list[tuple[int, int]]:
    """UTC days (year, day of year) touched by the time range ``[t_min, t_max]`` [MJD seconds]."""
    first = int(np.floor(t_min / 86400.0))
    last = int(np.floor(t_max / 86400.0))
    days = []
    for mjd in range(first, last + 1):
        date = dt.date(1858, 11, 17) + dt.timedelta(days=mjd)
        days.append((date.year, date.timetuple().tm_yday))
    return days


class CddisUnreachable(FileNotFoundError):
    """CDDIS did not answer (network, firewall): the file may exist there."""


def _curl(url: str, dest: Path, timeout: float) -> tuple[Optional[str], bool]:
    """Download ``url`` to ``dest`` with curl (``ftp://``: anonymous FTP over TLS, as AIPS VLBATECR does).

    Returns
    -------
    (error, unreachable)
        ``error`` is None on success, else curl's message; ``unreachable`` tells a network failure (see
        :data:`CURL_UNREACHABLE`) from a missing file.
    """
    command = ["curl", "-sS", "-f", "-L", "--connect-timeout", "20", "-m", str(int(timeout))]
    if url.startswith("ftp://"):
        command += ["-u", f"anonymous:{CDDIS_EMAIL}", "--ftp-ssl"]
    try:
        result = subprocess.run(command + [url, "-o", str(dest)], capture_output=True, text=True,
                                timeout=timeout + 30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc), True
    if result.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
        return None, False
    dest.unlink(missing_ok=True)
    return (result.stderr.strip() or f"curl exit code {result.returncode}"), result.returncode in CURL_UNREACHABLE


def _decompress(path: Path) -> Path:
    """Decompress a ``.gz`` or unix-compressed ``.Z`` file next to itself (the original is removed)."""
    out = path.with_suffix("")
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    elif path.suffix == ".Z":
        # gzip reads the LZW format of compress(1); Python's standard library cannot.
        with open(out, "wb") as fout:
            result = subprocess.run(["gzip", "-dc", str(path)], stdout=fout, stderr=subprocess.PIPE)
        if result.returncode != 0:
            out.unlink(missing_ok=True)
            raise OSError(f"cannot uncompress {path}: {result.stderr.decode(errors='replace').strip()}")
    else:
        return path
    path.unlink()
    return out


def _local_file(directory: Path, names: list[str]) -> Optional[Path]:
    """An already-downloaded (decompressed) copy of any of ``names`` in ``directory``, else None."""
    for name in names:
        plain = directory / name.removesuffix(".gz").removesuffix(".Z")
        if plain.is_file() and plain.stat().st_size > 0:
            return plain
    return None


def fetch_ionex_day(model: str, year: int, doy: int, directory, *, timeout: float = 60.0,
                    use_cddis: bool = True) -> Path:
    """Path of the decompressed IONEX file of product ``model`` for one day, downloading it when not cached.

    The CDDIS archive is tried under every candidate name (:func:`ionex_file_names`), with the AIPS command
    ``curl -u anonymous:<e-mail> --ftp-ssl ftp://gdc.cddis.eosdis.nasa.gov/gnss/products/ionex/YYYY/DDD/<file>``
    (skipped with ``use_cddis=False``); the JPL final products also come from JPL's HTTPS server.

    Raises
    ------
    CddisUnreachable
        When CDDIS could not be reached and no other source had the file.
    FileNotFoundError
        When no source has the file (the message lists every URL tried and why it failed).
    """
    from .tools import download_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    code = str(model).strip().lower()
    names = ionex_file_names(code, year, doy)
    cached = _local_file(directory, names)
    if cached is not None:
        logger.debug("IONEX {} {}/{:03d}: using {}", code, year, doy, cached)
        return cached
    failures, unreachable = [], False
    for name in names if use_cddis else []:
        url = f"{CDDIS_IONEX_URL}/{year:04d}/{doy:03d}/{name}"
        error, unreachable = _curl(url, directory / name, timeout)
        if error is None:
            logger.info("IONEX {} {}/{:03d}: downloaded {}", code, year, doy, url)
            return _decompress(directory / name)
        (directory / name).unlink(missing_ok=True)
        failures.append(f"{url}: {error}")
        if unreachable:
            break
    if PRODUCTS[code][:2] == ("JPL", "FIN"):
        for name in (f"JPL0OPSFIN_{year:04d}{doy:03d}0000_01D_02H_GIM.INX.gz", f"JPLG{doy:03d}0.{year % 100:02d}I.gz"):
            url = f"{JPL_FINAL_URL}/y{year:04d}/{name}"
            error, _ = _curl(url, directory / name, timeout)
            if error is not None:
                try:                                  # no curl, or curl refused: Python's own HTTPS client
                    download_file(url, directory / name, timeout=timeout)
                    error = None
                except ConnectionError as exc:
                    error = f"{error}; {exc}"
            if error is not None:
                failures.append(f"{url}: {error}")
                continue
            logger.info("IONEX {} {}/{:03d}: downloaded {}", code, year, doy, url)
            return _decompress(directory / name)
    if not use_cddis:
        failures.append("CDDIS not tried (unreachable earlier)")
    message = f"IONEX product {code} for {year}/{doy:03d} is not available:\n  " + "\n  ".join(failures)
    raise CddisUnreachable(message) if unreachable else FileNotFoundError(message)


def fetch_ionex(t_min: float, t_max: float, directory, *, model: str = "emrg", fallback: bool = True,
                timeout: float = 60.0) -> tuple[str, list[Path]]:
    """IONEX files covering ``[t_min, t_max]`` [MJD seconds], all of one product.

    The requested product is tried first, then (``fallback``) the other final products and the rapid ones in
    :data:`FALLBACK_ORDER`; the first product available for every day is used, so maps of different centres
    are never mixed. Once CDDIS has failed to answer, only the cache and the other sources are tried.

    Returns
    -------
    (model, files)
        The product used and its decompressed files, one per day, in time order.

    Raises
    ------
    FileNotFoundError
        When no product covers every day (the message gives the reasons per product).
    """
    days = observation_days(t_min, t_max)
    order = [model.lower()] + ([code for code in FALLBACK_ORDER if code != model.lower()] if fallback else [])
    reasons, use_cddis = [], True
    for code in order:
        files = []
        try:
            for year, doy in days:
                files.append(fetch_ionex_day(code, year, doy, directory, timeout=timeout, use_cddis=use_cddis))
        except FileNotFoundError as exc:
            if isinstance(exc, CddisUnreachable) and use_cddis:
                use_cddis = False
                logger.warning("CDDIS cannot be reached; trying the cached IONEX files and other sources only")
            reasons.append(str(exc))
            logger.warning("IONEX product {} unavailable for days {}; trying the next one", code, days)
            continue
        if code != model.lower():
            logger.warning("IONEX product {} unavailable; using {} instead", model, code)
        return code, files
    raise FileNotFoundError("no IONEX product covers days " + ", ".join(f"{y}/{d:03d}" for y, d in days)
                            + " (download them by hand into the IONEX directory):\n" + "\n".join(reasons))


# ----------------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------------
def _label(line: str) -> str:
    """The IONEX record label (columns 61-80), stripped."""
    return line[60:80].strip()


def _epoch_mjd_seconds(line: str) -> float:
    """MJD seconds of an ``EPOCH OF ...`` record (6I6: year, month, day, hour, minute, second)."""
    year, month, day, hour, minute, second = (int(line[6 * i:6 * i + 6]) for i in range(6))
    moment = dt.datetime(year, month, day, tzinfo=dt.timezone.utc) + dt.timedelta(hours=hour, minutes=minute,
                                                                                  seconds=second)
    return moment.timestamp() + _MJD_UNIX_SECONDS


def _grid_axis(line: str) -> np.ndarray:
    """Grid values of a ``LAT1 / LAT2 / DLAT`` or ``LON1 / LON2 / DLON`` record (2X, 3F6.1)."""
    first, last, step = (float(line[2 + 6 * i:8 + 6 * i]) for i in range(3))
    count = int(round((last - first) / step)) + 1
    return first + step * np.arange(count)


def _read_block(lines: list[str], index: int, nvalue: int) -> tuple[np.ndarray, int]:
    """``nvalue`` I5 values starting at ``lines[index]`` (16 per line); returns (values, next line index)."""
    values = []
    while len(values) < nvalue:
        line = lines[index]
        count = min(16, nvalue - len(values))
        values.extend(int(line[5 * i:5 * i + 5]) for i in range(count))
        index += 1
    return np.asarray(values, dtype=np.float64), index


def read_ionex(path) -> IonexMaps:
    """Read the TEC (and RMS) maps of one IONEX 1.0/1.1 file of 2-D maps.

    Raises
    ------
    ValueError
        For a file that is not a 2-D IONEX file of constant height, or whose maps are not in time order.
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if not lines or _label(lines[0]) != "IONEX VERSION / TYPE" or lines[0][20:21] != "I":
        raise ValueError(f"{path}: not an IONEX file (first record {lines[0][60:80] if lines else ''!r})")
    header = {"exponent": -1, "base_radius": 6371.0, "mapping": "", "dimension": 2}
    index = 0
    while index < len(lines) and _label(lines[index]) != "END OF HEADER":
        line, label = lines[index], _label(lines[index])
        if label == "HGT1 / HGT2 / DHGT":
            header["hgt"] = [float(line[2 + 6 * i:8 + 6 * i]) for i in range(3)]
        elif label == "LAT1 / LAT2 / DLAT":
            header["lat"] = _grid_axis(line)
        elif label == "LON1 / LON2 / DLON":
            header["lon"] = _grid_axis(line)
        elif label == "EXPONENT":
            header["exponent"] = int(line[:6])
        elif label == "BASE RADIUS":
            header["base_radius"] = float(line[:8])
        elif label == "MAPPING FUNCTION":
            header["mapping"] = line[:60].strip()
        elif label == "MAP DIMENSION":
            header["dimension"] = int(line[:6])
        index += 1
    for key in ("hgt", "lat", "lon"):
        if key not in header:
            raise ValueError(f"{path}: IONEX header has no {key.upper()} record")
    if header["dimension"] != 2 or header["hgt"][0] != header["hgt"][1]:
        raise ValueError(f"{path}: only 2-D maps on a single shell are supported (HGT1/HGT2/DHGT = {header['hgt']})")
    lat, lon = header["lat"], header["lon"]
    maps = {"TEC": {}, "RMS": {}}
    exponent = header["exponent"]
    kind, number, epoch, grid = None, None, None, None
    while index < len(lines):
        line, label = lines[index], _label(lines[index])
        index += 1
        if label in ("START OF TEC MAP", "START OF RMS MAP"):
            kind, number, epoch = label.split()[2], int(line[:6]), None
            grid = np.full((lat.size, lon.size), np.nan)
        elif label == "EXPONENT":
            exponent = int(line[:6])
        elif label == "EPOCH OF CURRENT MAP" and kind is not None:
            epoch = _epoch_mjd_seconds(line)
        elif label == "LAT/LON1/LON2/DLON/H" and kind is not None:
            row_lat, lon1, lon2, dlon = (float(line[2 + 6 * i:8 + 6 * i]) for i in range(4))
            nvalue = int(round((lon2 - lon1) / dlon)) + 1
            values, index = _read_block(lines, index, nvalue)
            ilat = int(round((row_lat - lat[0]) / (lat[1] - lat[0])))
            ilon = np.rint((lon1 + dlon * np.arange(nvalue) - lon[0]) / (lon[1] - lon[0])).astype(int)
            grid[ilat, ilon] = np.where(values == 9999, np.nan, values * 10.0 ** exponent)
        elif label in ("END OF TEC MAP", "END OF RMS MAP") and kind is not None:
            if epoch is None:
                raise ValueError(f"{path}: {kind} map {number} has no epoch")
            maps[kind][number] = (epoch, grid)
            kind = None
    if not maps["TEC"]:
        raise ValueError(f"{path}: no TEC map")
    numbers = sorted(maps["TEC"])
    epochs = np.array([maps["TEC"][n][0] for n in numbers])
    if np.any(np.diff(epochs) <= 0):
        raise ValueError(f"{path}: TEC maps are not in increasing time order")
    tec = np.stack([maps["TEC"][n][1] for n in numbers])
    rms = np.stack([maps["RMS"][n][1] for n in numbers]) if sorted(maps["RMS"]) == numbers else None
    lat_order, lon_order = np.argsort(lat), np.argsort(lon)
    tec = tec[:, lat_order][:, :, lon_order]
    rms = None if rms is None else rms[:, lat_order][:, :, lon_order]
    logger.debug("IONEX {}: {} maps, {} x {} grid, shell {} km", path.name, epochs.size, lat.size, lon.size,
                 header["hgt"][0])
    return IonexMaps(epochs=epochs, lat=lat[lat_order], lon=lon[lon_order], tec=tec, rms=rms,
                     height_km=float(header["hgt"][0]), base_radius_km=float(header["base_radius"]),
                     mapping_function=header["mapping"], files=[str(path)])


def merge_maps(parts: list[IonexMaps]) -> IonexMaps:
    """Join the maps of consecutive days into one series.

    A map at the epoch of one already read (midnight, which ends one day file and starts the next) is dropped:
    AIPS keeps the first one read. The files must share the grid and the shell height.
    """
    if not parts:
        raise ValueError("no IONEX maps to merge")
    parts = sorted(parts, key=lambda maps: maps.epochs[0])
    first = parts[0]
    epochs, tec, rms = [first.epochs], [first.tec], [first.rms]
    for maps in parts[1:]:
        if (not np.array_equal(maps.lat, first.lat) or not np.array_equal(maps.lon, first.lon)
                or maps.height_km != first.height_km):
            raise ValueError(f"IONEX files {first.files[0]} and {maps.files[0]} have different grids or shells")
        keep = maps.epochs > epochs[-1][-1]
        epochs.append(maps.epochs[keep])
        tec.append(maps.tec[keep])
        rms.append(None if maps.rms is None else maps.rms[keep])
    merged_rms = None if any(r is None for r in rms) else np.concatenate(rms)
    return IonexMaps(epochs=np.concatenate(epochs), lat=first.lat, lon=first.lon, tec=np.concatenate(tec),
                     rms=merged_rms, height_km=first.height_km, base_radius_km=first.base_radius_km,
                     mapping_function=first.mapping_function, files=[f for maps in parts for f in maps.files])


# ----------------------------------------------------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------------------------------------------------
def _bilinear(grid: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray, lat: np.ndarray,
              lon: np.ndarray) -> np.ndarray:
    """AIPS ``TECVAL``: bilinear interpolation of ``grid[map, lat, lon]`` at points (lat [deg], lon [deg]).

    ``grid`` is (npoint, nlat, nlon), one map per point. Latitudes beyond the grid take its edge row; a global
    grid (360 degrees of longitude) wraps in longitude, a regional one gives NaN outside.
    """
    dlat, dlon = lat_axis[1] - lat_axis[0], lon_axis[1] - lon_axis[0]
    y = np.clip((lat - lat_axis[0]) / dlat, 0.0, lat_axis.size - 1)
    span = lon_axis[-1] - lon_axis[0]
    x = (lon - lon_axis[0]) / dlon
    if abs(span - 360.0) < 1e-6:
        x = np.mod(x, round(360.0 / dlon))
    outside = (x < 0) | (x > lon_axis.size - 1)
    x = np.clip(x, 0.0, lon_axis.size - 1)
    j = np.minimum(np.floor(y).astype(int), lat_axis.size - 2)
    i = np.minimum(np.floor(x).astype(int), lon_axis.size - 2)
    wy, wx = y - j, x - i
    point = np.arange(grid.shape[0])
    value = ((1.0 - wx) * (1.0 - wy) * grid[point, j, i] + (1.0 - wx) * wy * grid[point, j + 1, i]
             + wx * (1.0 - wy) * grid[point, j, i + 1] + wx * wy * grid[point, j + 1, i + 1])
    return np.where(outside, np.nan, value)


def vertical_tec(maps: IonexMaps, lat_deg, lon_deg, time_mjd_s, *, rotation: float = 1.0,
                 use_rms: bool = False) -> np.ndarray:
    """Vertical TEC [TECU] at the shell points (``lat_deg``, ``lon_deg``) [deg] and times [MJD seconds].

    AIPS ``IONTEC``: the two maps around the time, each evaluated at the longitude the ionosphere above the
    point had at that map's epoch (``lon + 360 rotation dt1`` for the earlier map, ``lon - 360 rotation dt2``
    for the later one, ``dt`` in days), weighted linearly in time. Times outside the maps give NaN.

    Parameters
    ----------
    rotation : float
        Fraction of the Sun-following rotation (AIPS ``APARM(2)``; 0 = maps fixed to the Earth).
    use_rms : bool
        Interpolate the RMS maps instead of the TEC maps.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    times = np.asarray(time_mjd_s, dtype=np.float64)
    lat, lon, times = np.broadcast_arrays(lat, lon, times)
    shape = lat.shape
    lat, lon, times = lat.reshape(-1), lon.reshape(-1), times.reshape(-1)
    grid = maps.rms if use_rms else maps.tec
    if grid is None:
        raise ValueError("the IONEX files carry no RMS maps")
    epochs = maps.epochs
    inside = (times >= epochs[0]) & (times <= epochs[-1])
    k = np.clip(np.searchsorted(epochs, times, side="right") - 1, 0, epochs.size - 2)
    dt1 = (times - epochs[k]) / 86400.0
    dt2 = (epochs[k + 1] - times) / 86400.0
    lon1 = lon + 360.0 * dt1 * rotation
    lon2 = lon - 360.0 * dt2 * rotation
    tec1 = _bilinear(grid[k], maps.lat, maps.lon, lat, lon1)
    tec2 = _bilinear(grid[k + 1], maps.lat, maps.lon, lat, lon2)
    value = (dt2 * tec1 + dt1 * tec2) / (dt1 + dt2)
    return np.where(inside, value, np.nan).reshape(shape)
