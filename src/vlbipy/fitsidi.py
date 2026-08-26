"""Direct FITS-IDI file inspection (astropy-only, no CASA required).

Reads observation metadata straight from FITS-IDI headers and tables so vlbipy
can inspect raw correlator output before any import: observing date, frequency
setup, sources (with coordinates), and the antenna array. Scan information is
not available at this level — it is derived from the measurement set after
import (see :meth:`vlbipy.backends.casa.CasaBackend.get_metadata`).

Recycled from ``casa_pipeline.obsdata.Importing.get_obsdate_from_fitsidi`` /
``get_freq_from_fitsidi`` and generalized to the full metadata set.
"""
from __future__ import annotations

import datetime as dt
import glob
from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits

from .logging_utils import get_logger
from .models import Antenna, FreqSetup, ObsMetadata, Stokes
from .tools import natsort_key

logger = get_logger()

#: Glob patterns (relative to a directory) used to locate FITS-IDI files for a project.
_FITSIDI_PATTERNS = ("{code}_*_1.IDI*", "{code}_*.IDI*", "{code}*.IDI*")


def find_fitsidi_files(directory: Union[str, Path], project_code: str) -> list[str]:
    """Return the naturally-sorted FITS-IDI files for a project in a directory.

    Tries both lower- and upper-case project-code spellings of the standard EVN
    naming scheme (``<code>_<pass>_1.IDI<n>``). Returns an empty list if none.
    """
    directory = Path(directory)
    found: set[str] = set()
    # The raw files may sit in the working directory or in its input_data/ subdirectory
    # (where the pipeline files them at import time); look in both.
    search_dirs = [directory, directory / "input_data"]
    for search_dir in search_dirs:
        for code in (project_code.lower(), project_code.upper()):
            for pattern in _FITSIDI_PATTERNS:
                found.update(glob.glob(str(search_dir / pattern.format(code=code))))
    files = [f for f in found if not f.endswith((".checksum", ".gz"))]
    return sorted(files, key=natsort_key)


def _common_header(hdulist: fits.HDUList) -> fits.Header:
    """Return the header of the first FITS-IDI table HDU (holds the common keywords)."""
    for hdu in hdulist[1:]:
        if hdu.header.get("REF_FREQ") is not None or hdu.header.get("RDATE") is not None:
            return hdu.header
    raise ValueError(f"No FITS-IDI table headers found in {hdulist.filename()}")


def get_obs_date(fitsidi_file: Union[str, Path]) -> dt.date:
    """Return the observing date (RDATE) of a FITS-IDI file as a date object."""
    with fits.open(fitsidi_file) as hdulist:
        rdate = _common_header(hdulist)["RDATE"]
    return dt.datetime.strptime(str(rdate), "%Y-%m-%d").date()


def get_ref_freq(fitsidi_file: Union[str, Path]) -> float:
    """Return the reference frequency (Hz) of a FITS-IDI file."""
    with fits.open(fitsidi_file) as hdulist:
        return float(_common_header(hdulist)["REF_FREQ"])


def has_tsys(fitsidi_file: Union[str, Path]) -> bool:
    """Return True if the FITS-IDI file contains a SYSTEM_TEMPERATURE table."""
    with fits.open(fitsidi_file) as hdulist:
        return any(hdu.name == "SYSTEM_TEMPERATURE" for hdu in hdulist)


def has_gain_curve(fitsidi_file: Union[str, Path]) -> bool:
    """Return True if the FITS-IDI file contains a GAIN_CURVE table."""
    with fits.open(fitsidi_file) as hdulist:
        return any(hdu.name == "GAIN_CURVE" for hdu in hdulist)


def _read_antennas(hdulist: fits.HDUList) -> dict[str, Antenna]:
    """Build the antenna dict from the ARRAY_GEOMETRY table (empty if absent)."""
    antennas: dict[str, Antenna] = {}
    if "ARRAY_GEOMETRY" not in hdulist:
        return antennas
    table = hdulist["ARRAY_GEOMETRY"].data
    columns = hdulist["ARRAY_GEOMETRY"].columns.names
    for i in range(len(table)):
        name = str(table["ANNAME"][i]).strip()
        position = tuple(float(v) for v in table["STABXYZ"][i]) if "STABXYZ" in columns else (0.0, 0.0, 0.0)
        diameter = float(table["DIAMETER"][i]) if "DIAMETER" in columns else 0.0
        antennas[name] = Antenna(name=name, diameter=diameter, position=position, observed=True)
    return antennas


def _read_sources(hdulist: fits.HDUList) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Return (source names, name -> (ra_deg, dec_deg)) from the SOURCE table (empty if absent)."""
    names: list[str] = []
    coords: dict[str, tuple[float, float]] = {}
    if "SOURCE" not in hdulist:
        return names, coords
    table = hdulist["SOURCE"].data
    columns = hdulist["SOURCE"].columns.names
    for i in range(len(table)):
        name = str(table["SOURCE"][i]).strip()
        if name in names:
            continue
        names.append(name)
        if "RAEPO" in columns and "DECEPO" in columns:
            # Some correlators write RA wrapped to (-180, 180]; normalize to [0, 360).
            coords[name] = (float(np.atleast_1d(table["RAEPO"][i])[0]) % 360.0,
                            float(np.atleast_1d(table["DECEPO"][i])[0]))
    return names, coords


def _read_freq_setup(hdulist: fits.HDUList, header: fits.Header) -> FreqSetup:
    """Build the frequency setup from the common header keywords (+FREQUENCY table if present)."""
    n_subbands = int(header.get("NO_BAND", 1))
    n_channels = int(header.get("NO_CHAN", 1))
    channel_width = float(header.get("CH_WIDTH", 0.0))
    total_bandwidth = channel_width * n_channels * n_subbands
    if "FREQUENCY" in hdulist:
        columns = hdulist["FREQUENCY"].columns.names
        if "TOTAL_BANDWIDTH" in columns:
            per_band = np.atleast_1d(hdulist["FREQUENCY"].data["TOTAL_BANDWIDTH"][0])
            total_bandwidth = float(np.sum(per_band))
    n_stokes = int(header.get("NO_STKD", 1))
    first_stokes = int(header.get("STK_1", 0))
    # FITS-IDI encodes RR,LL,RL,LR as -1..-4 and XX,YY,XY,YX as -5..-8.
    _idi_stokes = {-1: Stokes.RR, -2: Stokes.LL, -3: Stokes.RL, -4: Stokes.LR,
                   -5: Stokes.XX, -6: Stokes.YY, -7: Stokes.XY, -8: Stokes.YX}
    polarizations = []
    for i in range(n_stokes):
        code = first_stokes - i if first_stokes < 0 else first_stokes + i
        if code < 0:
            polarizations.append(_idi_stokes.get(code, Stokes.Undefined))
        else:
            polarizations.append(Stokes(code) if 0 <= code <= 12 else Stokes.Undefined)
    return FreqSetup(ref_freq=float(header.get("REF_FREQ", 0.0)), total_bandwidth=total_bandwidth,
                     n_subbands=n_subbands, n_channels=n_channels, channel_width=channel_width,
                     polarizations=polarizations)


def inspect_fitsidi(files: Union[str, Path, list], project_code: str = "") -> ObsMetadata:
    """Read observation metadata from FITS-IDI file(s) without importing them.

    Parameters
    ----------
    files : str or pathlib.Path or list
        One FITS-IDI file or a list of them; only the first is read (the
        common tables are replicated across the set).
    project_code : str
        Project code stored in the returned metadata (falls back to OBSCODE).

    Returns
    -------
    ObsMetadata
        Metadata with antennas, sources (names + coordinates), frequency setup
        and observing date. ``scans`` is empty (needs an imported MS).
    """
    first = Path(files[0] if isinstance(files, (list, tuple)) else files)
    if not first.is_file():
        raise FileNotFoundError(f"FITS-IDI file not found: {first}")
    with fits.open(first) as hdulist:
        header = _common_header(hdulist)
        antennas = _read_antennas(hdulist)
        source_names, source_coords = _read_sources(hdulist)
        freq_setup = _read_freq_setup(hdulist, header)
        obs_date = None
        if header.get("RDATE"):
            obs_date = dt.datetime.strptime(str(header["RDATE"]), "%Y-%m-%d").date()
        code = project_code or str(header.get("OBSCODE", "")).strip()
    meta = ObsMetadata(project_code=code, obs_date=obs_date, antennas=antennas,
                       freq_setup=freq_setup, source_names=source_names, source_coords=source_coords)
    logger.info("inspected {}: {} antennas, {} sources, {:.3f} GHz",
                first.name, len(antennas), len(source_names), freq_setup.freq_ghz)
    return meta
