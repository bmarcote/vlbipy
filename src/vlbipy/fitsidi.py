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


def search_dirs(directory: Union[str, Path]) -> list[Path]:
    """Directories searched for the raw files of a project, nearest first.

    The working directory, its ``input_data/`` and its parent: ``vlbipy -p CODE`` run where the files were
    downloaded by hand creates ``./CODE`` as its working directory, so the files sit one level up.
    """
    root = Path(directory)
    return [d for d in (root, root / "input_data", root.resolve().parent) if d.is_dir()]


def find_fitsidi_files(directory: Union[str, Path], project_code: str) -> list[str]:
    """Return the naturally-sorted FITS-IDI files for a project in a directory.

    Tries both lower- and upper-case project-code spellings of the standard EVN
    naming scheme (``<code>_<pass>_1.IDI<n>``) in each of :func:`search_dirs`;
    the nearest directory holding any wins. Returns an empty list if none.
    """
    for search_dir in search_dirs(directory):
        found: set[str] = set()
        for code in (project_code.lower(), project_code.upper()):
            for pattern in _FITSIDI_PATTERNS:
                found.update(glob.glob(str(search_dir / pattern.format(code=code))))
        files = [f for f in found if not f.endswith((".checksum", ".gz"))]
        if files:
            return sorted(files, key=natsort_key)
    return []


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
    """Return True if the FITS-IDI file contains a GAIN_CURVE table with at least one row."""
    with fits.open(fitsidi_file) as hdulist:
        return any(hdu.name == "GAIN_CURVE" and hdu.header.get("NAXIS2", 0) > 0 for hdu in hdulist)


def expected_size(fitsidi_file: Union[str, Path]) -> int:
    """Bytes the file must have to hold every HDU its headers announce (FITS 2880-byte blocks).

    Only headers are read, so this is cheap on a file of many GB. A file still being downloaded, or cut short,
    is smaller than this; its last HDUs are then missing or incomplete (in VLBA FITS-IDI, the FLAG,
    SYSTEM_TEMPERATURE and GAIN_CURVE tables come after UV_DATA).
    """
    import warnings as _warnings
    end = 0
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")                     # astropy warns about the truncation we measure
        with fits.open(fitsidi_file, memmap=True, lazy_load_hdus=False) as hdulist:
            for index, hdu in enumerate(hdulist):
                header = hdu.header
                naxes = [int(header.get(f"NAXIS{i}", 0)) for i in range(1, int(header.get("NAXIS", 0)) + 1)]
                elements = int(np.prod(naxes)) if naxes else 0
                size = abs(int(header.get("BITPIX", 8))) // 8 * int(header.get("GCOUNT", 1)) * (
                    int(header.get("PCOUNT", 0)) + elements)
                start = hdulist.fileinfo(index)["datLoc"]
                end = max(end, start + -(-size // 2880) * 2880)
    return end


def check_complete(fitsidi_file: Union[str, Path]) -> None:
    """Raise ``ValueError`` when the file is shorter than its headers say (:func:`expected_size`)."""
    actual, expected = Path(fitsidi_file).stat().st_size, expected_size(fitsidi_file)
    if actual < expected:
        raise ValueError(f"{Path(fitsidi_file).name} is incomplete: {actual / 1e9:.2f} of {expected / 1e9:.2f} GB "
                         "(still downloading, or cut short); its tables after the visibilities are missing")


def table_rows(fitsidi_file: Union[str, Path]) -> dict[str, int]:
    """Rows of every table HDU of the file, by EXTNAME (the first of each name; headers only)."""
    rows: dict[str, int] = {}
    with fits.open(fitsidi_file, memmap=True) as hdulist:
        for hdu in hdulist[1:]:
            rows.setdefault(hdu.name, int(hdu.header.get("NAXIS2", 0)))
    return rows


def apriori_coverage(fitsidi_file: Union[str, Path]) -> dict[str, list[str]]:
    """Antennas of ARRAY_GEOMETRY that the SYSTEM_TEMPERATURE and GAIN_CURVE tables say nothing about.

    Returns ``{"Tsys": [names], "gain curve": [names]}``; a missing table lists every antenna.
    """
    with fits.open(fitsidi_file, memmap=True) as hdulist:
        geometry = hdulist["ARRAY_GEOMETRY"].data
        numbers = {int(n): str(name).strip() for name, n in zip(geometry["ANNAME"], geometry["NOSTA"])}
        present = {}
        for label, extname in (("Tsys", "SYSTEM_TEMPERATURE"), ("gain curve", "GAIN_CURVE")):
            present[label] = set(np.unique(hdulist[extname].data["ANTENNA_NO"]).tolist()) \
                if extname in hdulist and hdulist[extname].header.get("NAXIS2", 0) > 0 else set()
    return {label: [name for number, name in sorted(numbers.items()) if number not in antennas]
            for label, antennas in present.items()}


def remove_empty_table(fitsidi_file: Union[str, Path], extname: str) -> bool:
    """Remove a zero-row table from a FITS-IDI file; return True if one was removed.

    DiFX writes a ``GAIN_CURVE`` table with no rows when it has no gain curves to
    give (LBA). It carries no information, but its presence stops the real table
    from being appended. When it is the last HDU — where DiFX puts it — the file
    is simply truncated at the start of that HDU, which costs nothing; anywhere
    else the file has to be rewritten without it.
    """
    with fits.open(fitsidi_file) as hdulist:
        empty = [i for i, hdu in enumerate(hdulist)
                 if hdu.name == extname and hdu.header.get("NAXIS2", 0) == 0]
        if not empty:
            return False
        index = empty[0]
        is_last = index == len(hdulist) - 1
        offset = hdulist.fileinfo(index)["hdrLoc"]
    if is_last:
        with open(fitsidi_file, "r+b") as handle:
            handle.truncate(offset)
    else:
        with fits.open(fitsidi_file, mode="update") as hdulist:
            del hdulist[index]
    logger.info("removed the empty {} table from {}", extname, Path(fitsidi_file).name)
    return True


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
