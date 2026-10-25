"""The subtables of a measurement set imported from FITS-IDI, written without CASA.

This is the subtable half of casacore's FITS-IDI filler (``FITSIDItoMS1``: ``fillAntennaTable``,
``fillFeedTable``, ``fillSpectralWindowTable``, ``fillFieldTable``, ``fillObsTables``, ``fillSysCalTable``,
``fillFlagCmdTable``, ``fillWeatherTable``, ``handleGainCurve``, ``handlePhaseCal``, ``handleCalc``,
``updateTables``), followed by what CASA does afterwards to reach the Multi-MS the pipeline uses:

* ``importfitsidi`` with several files imports each one on its own and ``ms.concatenate``s it (``MSConcat``):
  antennas, fields, subbands and the polarization setup of later files must match the first file's, which only
  changes the validity time of the FEED rows; their SYSCAL, WEATHER, GAIN_CURVE, PHASE_CAL and
  EARTH_ORIENTATION rows are appended (only when the first file has non-empty tables of that kind), their
  FLAG_CMD and HISTORY are dropped, and each file adds an OBSERVATION row (EARTH_ORIENTATION rows of file k
  get OBSERVATION_ID k);
* ``constobsid=True`` keeps a single OBSERVATION row spanning every file, the spectral frame of every subband
  becomes ``specframe``, the FIELD directions are declared ICRS instead of J2000, and HISTORY gets the task's
  parameters;
* ``partition(createmms=True)`` rebuilds the standard subtables row by row (the ANTENNA keywords are lost, the
  FIELD direction keywords are rewritten), adds an empty SOURCE table, keeps SYSCAL and WEATHER only when they
  have rows, copies the other subtables as they are, and appends its own parameters to HISTORY.

:func:`read_idi_setup` gives the main-table converter (:mod:`vlbipy.solvers.idi_import`) what the filler
derives from the headers (antenna numbering, correlation order, channel widths, weights);
:func:`write_subtables` writes the subtables of the first part of the Multi-MS. Values are computed as casacore
computes them (float/double promotions, directions through direction cosines), so the tables are identical to
CASA's apart from the HISTORY time stamps and version line. Input the native import cannot reproduce exactly
(later files with other antennas, fields or subbands; tables in unusual order; unsupported polarizations)
raises NotImplementedError, and the caller runs CASA instead.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger

logger = get_logger()

#: JD of MJD 0.
JD_MJD0 = 2400000.5
#: casacore constants (casa/BasicSL/Constants.h), as literals so the products round like casacore's.
DAY = 86400.0
DEGREE = 0.0174532925199432957692369
ARCSEC = 0.00000484813681109535993589914098765
#: MJD 0 as a proleptic Gregorian ordinal.
MJD0_ORDINAL = _dt.date(1858, 11, 17).toordinal()
#: Unix epoch in MJD seconds.
UNIX_EPOCH_MJD_SECONDS = 40587 * DAY
#: AIPS Stokes codes -> casacore Stokes enum (FITSIDItoMS1::getAxisInfo; 1 -> XX is casacore's SMA "quick fix").
AIPS_TO_STOKES = {-8: 11, -7: 10, -6: 12, -5: 9, -4: 7, -3: 6, -2: 8, -1: 5, 1: 9, 2: 2, 3: 3, 4: 4}
#: casacore Stokes enum -> (receptor 1, receptor 2); Stokes I/Q/U/V have none (casacore throws).
STOKES_RECEPTORS = {5: (0, 0), 6: (0, 1), 7: (1, 0), 8: (1, 1), 9: (0, 0), 10: (0, 1), 11: (1, 0), 12: (1, 1)}
#: Cross hands exchanged when a baseline is given the other way round.
STOKES_SWAPPED = {10: 11, 11: 10, 6: 7, 7: 6}
#: importfitsidi ``specframe`` -> MEAS_FREQ_REF code.
SPECTRAL_FRAMES = {"REST": 0, "LSRK": 1, "LSRD": 2, "BARY": 3, "GEO": 4, "TOPO": 5}
#: ARRAY_GEOMETRY MNTSTA -> ANTENNA MOUNT (fillAntennaTable).
MOUNTS = {0: "ALT-AZ", 1: "EQUATORIAL", 2: "ORBITING", 3: "X-Y", 4: "ALT-AZ+NASMYTH-R", 5: "ALT-AZ+NASMYTH-L",
          6: "ALT-AZ+BWG-R", 7: "ALT-AZ+BWG-L"}
#: FITS-IDI tables the filler turns into subtables (any other extension is ignored, as by casacore).
FILLED_TABLES = ("ARRAY_GEOMETRY", "SOURCE", "FREQUENCY", "ANTENNA", "SYSTEM_TEMPERATURE", "FLAG", "GAIN_CURVE",
                 "PHASE-CAL", "WEATHER", "CALC")
#: Tables that must precede the first UV_DATA of a file (the filler completes them when it reads UV_DATA).
HEADER_TABLES = ("ARRAY_GEOMETRY", "FREQUENCY", "SOURCE", "ANTENNA")
#: FITS-IDI table -> subtable copied as-is by partition (after SYSCAL and WEATHER, in file order).
GENERIC_SUBTABLES = {"CALC": "EARTH_ORIENTATION", "GAIN_CURVE": "GAIN_CURVE", "PHASE-CAL": "PHASE_CAL"}
#: Table comments (TableDesc) of the subtables the filler describes itself.
TABLE_COMMENTS = {"EARTH_ORIENTATION": "Earth Orientation Parameters table", "GAIN_CURVE": "Gain curve table",
                  "PHASE_CAL": "Phase calibration table"}
#: Version line of the HISTORY rows (CASA writes its own version there).
HISTORY_VERSION = "version: vlbipy native FITS-IDI import (no CASA)"
#: Parameters partition records in HISTORY, as the pipeline calls it (vlbipy.backends.casa.import_data).
PARTITION_PARAMS = (("createmms", True), ("separationaxis", "auto"), ("numsubms", "auto"), ("flagbackup", False),
                    ("datacolumn", "all"), ("field", ""), ("spw", ""), ("scan", ""), ("antenna", ""),
                    ("correlation", ""), ("timerange", ""), ("intent", ""), ("array", ""), ("uvrange", ""),
                    ("observation", ""), ("feed", ""), ("disableparallel", False), ("ddistart", -1), ("taql", ""))

# ---------------------------------------------------------------------------
# Column descriptions (as partition writes them; casacore MS definitions and the FITS-IDI filler)
# ---------------------------------------------------------------------------
_SECONDS = {"QuantumUnits": ["s"]}
_EPOCH = {"QuantumUnits": ["s"], "MEASINFO": {"type": "epoch", "Ref": "UTC"}}
_POSITION = {"QuantumUnits": ["m", "m", "m"], "MEASINFO": {"type": "position", "Ref": "ITRF"}}
_DIRECTION_J2000 = {"QuantumUnits": ["rad", "rad"], "MEASINFO": {"type": "direction", "Ref": "J2000"}}
_HZ = {"QuantumUnits": ["Hz"]}
_RAD = {"QuantumUnits": ["rad"]}
_KELVIN = {"QuantumUnits": ["K"]}
_METRE = {"QuantumUnits": ["m"]}
_FREQUENCY = {"QuantumUnits": ["Hz"], "MEASINFO": {
    "type": "frequency", "VarRefCol": "MEAS_FREQ_REF",
    "TabRefTypes": ["REST", "LSRK", "LSRD", "BARY", "GEO", "TOPO", "GALACTO", "LGROUP", "CMB", "Undefined"],
    "TabRefCodes": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 64], dtype=np.uint32)}}
#: Placeholder for the FIELD direction keywords (their Ref depends on the SOURCE table's equinox).
_FIELD_DIRECTION = "FIELD_DIRECTION"

#: Per subtable: (column, value type, cell, comment, keywords); cell None = scalar, int = array of that ndim
#: (-1 any), tuple = fixed C-order shape stored directly.
SCHEMAS = {
    "ANTENNA": [
        ("OFFSET", "double", (3,), "Axes offset of mount to FEED REFERENCE point", _POSITION),
        ("POSITION", "double", (3,), "Antenna X,Y,Z phase reference position", _POSITION),
        ("TYPE", "string", None, "Antenna type (e.g. SPACE-BASED)", {}),
        ("DISH_DIAMETER", "double", None, "Physical diameter of dish", _METRE),
        ("FLAG_ROW", "boolean", None, "Flag for this row", {}),
        ("MOUNT", "string", None, "Mount type e.g. alt-az, equatorial, etc.", {}),
        ("NAME", "string", None, "Antenna name, e.g. VLA22, CA03", {}),
        ("STATION", "string", None, "Station (antenna pad) name", {}),
    ],
    "DATA_DESCRIPTION": [
        ("FLAG_ROW", "boolean", None, "Flag this row", {}),
        ("POLARIZATION_ID", "int", None, "Pointer to polarization table", {}),
        ("SPECTRAL_WINDOW_ID", "int", None, "Pointer to spectralwindow table", {}),
    ],
    "FEED": [
        ("POSITION", "double", (3,), "Position of feed relative to feed reference position", _POSITION),
        ("BEAM_OFFSET", "double", 2, "Beam position offset (on sky but in antennareference frame)",
         _DIRECTION_J2000),
        ("POLARIZATION_TYPE", "string", 1, "Type of polarization to which a given RECEPTOR responds", {}),
        ("POL_RESPONSE", "complex", 2, "D-matrix i.e. leakage between two receptors", {}),
        ("RECEPTOR_ANGLE", "double", 1, "The reference angle for polarization", _RAD),
        ("ANTENNA_ID", "int", None, "ID of antenna in this array", {}),
        ("BEAM_ID", "int", None, "Id for BEAM model", {}),
        ("FEED_ID", "int", None, "Feed id", {}),
        ("INTERVAL", "double", None, "Interval for which this set of parameters is accurate", _SECONDS),
        ("NUM_RECEPTORS", "int", None, "Number of receptors on this feed (probably 1 or 2)", {}),
        ("SPECTRAL_WINDOW_ID", "int", None, "ID for this spectral window setup", {}),
        ("TIME", "double", None, "Midpoint of time for which this set of parameters is accurate", _EPOCH),
    ],
    "FLAG_CMD": [
        ("APPLIED", "boolean", None, "True if flag has been applied to main table", {}),
        ("COMMAND", "string", None, "Flagging command", {}),
        ("INTERVAL", "double", None, "Time interval for which this flag is valid", _SECONDS),
        ("LEVEL", "int", None, "Flag level - revision level ", {}),
        ("REASON", "string", None, "Flag reason", {}),
        ("SEVERITY", "int", None, "Severity code (0-10) ", {}),
        ("TIME", "double", None, "Midpoint of interval for which this flag is valid", _EPOCH),
        ("TYPE", "string", None, "Type of flag (FLAG or UNFLAG)", {}),
    ],
    "FIELD": [
        ("DELAY_DIR", "double", 2, "Direction of delay center (e.g. RA, DEC)as polynomial in time.", _FIELD_DIRECTION),
        ("PHASE_DIR", "double", 2, "Direction of phase center (e.g. RA, DEC).", _FIELD_DIRECTION),
        ("REFERENCE_DIR", "double", 2, "Direction of REFERENCE center (e.g. RA, DEC).as polynomial in time.",
         _FIELD_DIRECTION),
        ("CODE", "string", None, "Special characteristics of field, e.g. Bandpass calibrator", {}),
        ("FLAG_ROW", "boolean", None, "Row Flag", {}),
        ("NAME", "string", None, "Name of this field", {}),
        ("NUM_POLY", "int", None, "Polynomial order of _DIR columns", {}),
        ("SOURCE_ID", "int", None, "Source id", {}),
        ("TIME", "double", None, "Time origin for direction and rate", _EPOCH),
    ],
    "HISTORY": [
        ("APP_PARAMS", "string", 1, "Application parameters", {}),
        ("CLI_COMMAND", "string", 1, "CLI command sequence", {}),
        ("APPLICATION", "string", None, "Application name", {}),
        ("MESSAGE", "string", None, "Log message", {}),
        ("OBJECT_ID", "int", None, "Originating ObjectID", {}),
        ("OBSERVATION_ID", "int", None, "Observation id (index in OBSERVATION table)", {}),
        ("ORIGIN", "string", None, "(Source code) origin from which message originated", {}),
        ("PRIORITY", "string", None, "Message priority", {}),
        ("TIME", "double", None, "Timestamp of message", _EPOCH),
    ],
    "OBSERVATION": [
        ("TIME_RANGE", "double", (2,), "Start and end of observation", _EPOCH),
        ("LOG", "string", 1, "Observing log", {}),
        ("SCHEDULE", "string", 1, "Observing schedule", {}),
        ("FLAG_ROW", "boolean", None, "Row flag", {}),
        ("OBSERVER", "string", None, "Name of observer(s)", {}),
        ("PROJECT", "string", None, "Project identification string", {}),
        ("RELEASE_DATE", "double", None, "Release date when data becomes public", _EPOCH),
        ("SCHEDULE_TYPE", "string", None, "Observing schedule type", {}),
        ("TELESCOPE_NAME", "string", None, "Telescope Name (e.g. WSRT, VLBA)", {}),
    ],
    "POLARIZATION": [
        ("CORR_TYPE", "int", 1, "The polarization type for each correlation product, as a Stokes enum.", {}),
        ("CORR_PRODUCT", "int", 2, "Indices describing receptors of feed going into correlation", {}),
        ("FLAG_ROW", "boolean", None, "Row flag", {}),
        ("NUM_CORR", "int", None, "Number of correlation products", {}),
    ],
    "PROCESSOR": [
        ("FLAG_ROW", "boolean", None, "Row flag", {}),
        ("MODE_ID", "int", None, "Processor mode id", {}),
        ("TYPE", "string", None, "Processor type", {}),
        ("TYPE_ID", "int", None, "Processor type id", {}),
        ("SUB_TYPE", "string", None, "Processor sub type", {}),
    ],
    "SPECTRAL_WINDOW": [
        ("MEAS_FREQ_REF", "int", None, "Frequency Measure reference", {}),
        ("CHAN_FREQ", "double", 1, "Center frequencies for each channel in the data matrix", _FREQUENCY),
        ("REF_FREQUENCY", "double", None, "The reference frequency", _FREQUENCY),
        ("CHAN_WIDTH", "double", 1, "Channel width for each channel", _HZ),
        ("EFFECTIVE_BW", "double", 1, "Effective noise bandwidth of each channel", _HZ),
        ("RESOLUTION", "double", 1, "The effective noise bandwidth for each channel", _HZ),
        ("FLAG_ROW", "boolean", None, "Row flag", {}),
        ("FREQ_GROUP", "int", None, "Frequency group", {}),
        ("FREQ_GROUP_NAME", "string", None, "Frequency group name", {}),
        ("IF_CONV_CHAIN", "int", None, "The IF conversion chain number", {}),
        ("NAME", "string", None, "Spectral window name", {}),
        ("NET_SIDEBAND", "int", None, "Net sideband", {}),
        ("NUM_CHAN", "int", None, "Number of spectral channels", {}),
        ("TOTAL_BANDWIDTH", "double", None, "The total bandwidth for this window", _HZ),
    ],
    "STATE": [
        ("CAL", "double", None, "Noise calibration temperature", _KELVIN),
        ("FLAG_ROW", "boolean", None, "Row flag", {}),
        ("LOAD", "double", None, "Load temperature", _KELVIN),
        ("OBS_MODE", "string", None, "Observing mode, e.g., OFF_SPECTRUM", {}),
        ("REF", "boolean", None, "True for a reference observation", {}),
        ("SIG", "boolean", None, "True for a source observation", {}),
        ("SUB_SCAN", "int", None, "Sub scan number, relative to scan number", {}),
    ],
    "SOURCE": [
        ("DIRECTION", "double", (2,), "Direction (e.g. RA, DEC).", _DIRECTION_J2000),
        ("PROPER_MOTION", "double", (2,), "Proper motion", {"QuantumUnits": ["rad/s"]}),
        ("CALIBRATION_GROUP", "int", None, "Number of grouping for calibration purpose.", {}),
        ("CODE", "string", None, "Special characteristics of source, e.g. Bandpass calibrator", {}),
        ("INTERVAL", "double", None, "Interval of time for which this set of parameters is accurate", _SECONDS),
        ("NAME", "string", None, "Name of source as given during observations", {}),
        ("NUM_LINES", "int", None, "Number of spectral lines", {}),
        ("SOURCE_ID", "int", None, "Source id", {}),
        ("SPECTRAL_WINDOW_ID", "int", None, "ID for this spectral window setup", {}),
        ("TIME", "double", None, "Midpoint of time for which this set of parameters is accurate.", _EPOCH),
    ],
    "POINTING": [
        ("DIRECTION", "double", 2, "Antenna pointing direction as polynomial in time", _DIRECTION_J2000),
        ("ANTENNA_ID", "int", None, "Antenna Id", {}),
        ("INTERVAL", "double", None, "Time interval", _SECONDS),
        ("NAME", "string", None, "Pointing position name", {}),
        ("NUM_POLY", "int", None, "Series order", {}),
        ("TARGET", "double", -1, "target direction as polynomial in time", _DIRECTION_J2000),
        ("TIME", "double", None, "Time interval midpoint", _EPOCH),
        ("TIME_ORIGIN", "double", None, "Time origin for direction", _EPOCH),
        ("TRACKING", "boolean", None, "Tracking flag - True if on position", {}),
    ],
    "SYSCAL": [
        ("ANTENNA_ID", "int", None, "ID of antenna in this array", {}),
        ("FEED_ID", "int", None, "Feed id", {}),
        ("INTERVAL", "double", None, "Interval for which this set of parameters is accurate", _SECONDS),
        ("SPECTRAL_WINDOW_ID", "int", None, "ID for this spectral window setup", {}),
        ("TIME", "double", None, "Midpoint of time for which this set of parameters is accurate", _EPOCH),
        ("TSYS", "float", -1, "System temp. for each of the two receptors", _KELVIN),
    ],
    "WEATHER": [
        ("ANTENNA_ID", "int", None, "Antenna number", {}),
        ("INTERVAL", "double", None, "Interval over which data is relevant", _SECONDS),
        ("TIME", "double", None, "An MEpoch specifying the midpoint of the time forwhich data is relevant",
         _EPOCH),
        ("DEW_POINT", "float", None, "Dew point", _KELVIN),
        ("H2O", "float", None, "Average column density of water-vapor", {"QuantumUnits": ["m-2"]}),
        ("IONOS_ELECTRON", "float", None, "Average column density of electrons", {"QuantumUnits": ["m-2"]}),
        ("PRESSURE", "float", None, "Ambient atmospheric pressure", {"QuantumUnits": ["hPa"]}),
        ("TEMPERATURE", "float", None, "Ambient Air Temperature for an antenna", _KELVIN),
        ("WIND_DIRECTION", "float", None, "Average wind direction", _RAD),
        ("WIND_SPEED", "float", None, "Average wind speed", {"QuantumUnits": ["m/s"]}),
    ],
    "EARTH_ORIENTATION": [
        ("TIME", "double", None, "Time for which this set of parameters is accurate", _EPOCH),
        ("OBSERVATION_ID", "int", None, "Observation identifier", {}),
        ("UT1_UTC", "double", None, "UT1-UTC", _SECONDS),
        ("PM", "double", -1, "Position of celestial pole", _RAD),
        ("TYPE", "string", None, "EOP type", {}),
    ],
    "GAIN_CURVE": [
        ("ANTENNA_ID", "int", None, "Antenna identifier", {}),
        ("FEED_ID", "int", None, "Feed identifier", {}),
        ("SPECTRAL_WINDOW_ID", "int", None, "Spectral window identifier", {}),
        ("TIME", "double", None, "Midpoint of time for which this set of parameters is accurate", _EPOCH),
        ("INTERVAL", "double", None, "Interval for which this set of parameters is accurate", _SECONDS),
        ("TYPE", "string", None, "Gain curve type", {}),
        ("NUM_POLY", "int", None, "Number of terms in polynomial", {}),
        ("GAIN", "float", -1, "Gain polynomial", {}),
        ("SENSITIVITY", "float", -1, "Antenna sensitivity", {"QuantumUnits": ["K/Jy"]}),
    ],
    "PHASE_CAL": [
        ("ANTENNA_ID", "int", None, "Antenna identifier", {}),
        ("FEED_ID", "int", None, "Feed identifier", {}),
        ("SPECTRAL_WINDOW_ID", "int", None, "Spectral window identifier", {}),
        ("TIME", "double", None, "Midpoint of time for which this set of parameters is accurate", _EPOCH),
        ("INTERVAL", "double", None, "Interval for which this set of parameters is accurate", _SECONDS),
        ("NUM_TONES", "int", None, "Number of phase-cal tones", {}),
        ("TONE_FREQUENCY", "double", -1, "Phase-cal tone frequency", _HZ),
        ("PHASE_CAL", "complex", -1, "Phase-cal measurement", {}),
        ("CABLE_CAL", "double", None, "Cable calibration measurement", _SECONDS),
    ],
}


def _columns(subtable: str, *, field_ref: str = "ICRS", weather_columns: tuple = ()) -> list:
    """The column descriptions of ``subtable`` (fresh copies), as partition writes them.

    ``field_ref`` is the direction reference of the FIELD columns; ``weather_columns`` the optional WEATHER
    columns present (DEW_POINT, PRESSURE, ... as the filler adds them).
    """
    from ..casacore_format.writer import array_column, scalar_column
    columns = []
    for name, value_type, cell, comment, keywords in SCHEMAS[subtable]:
        if subtable == "WEATHER" and name not in ("ANTENNA_ID", "INTERVAL", "TIME") and name not in weather_columns:
            continue
        if keywords == _FIELD_DIRECTION:
            keywords = {"QuantumUnits": ["rad", "rad"], "MEASINFO": {"type": "direction", "Ref": field_ref}}
        keywords = copy.deepcopy(keywords)
        if cell is None:
            columns.append(scalar_column(name, value_type, comment=comment, keywords=keywords))
        elif isinstance(cell, tuple):
            columns.append(array_column(name, value_type, shape=cell, direct=True, comment=comment,
                                        keywords=keywords))
        else:
            columns.append(array_column(name, value_type, ndim=cell, comment=comment, keywords=keywords))
    return columns


# ---------------------------------------------------------------------------
# Reading FITS-IDI headers and small tables
# ---------------------------------------------------------------------------
def _repeat(header, name: str) -> tuple[int, str]:
    """Repeat count and type letter of binary-table column ``name`` (TFORM), or (0, "") when absent."""
    for k in range(1, int(header.get("TFIELDS", 0)) + 1):
        if str(header.get(f"TTYPE{k}", "")).strip() == name:
            form = str(header[f"TFORM{k}"]).strip()
            digits = "".join(ch for ch in form if ch.isdigit())
            return (int(digits) if digits else 1), form.lstrip("0123456789")[:1]
    return 0, ""


def _uv_times(path: str, header, data_offset: int) -> tuple:
    """(first row time, last row time, last row INTTIM) of a UV_DATA table, time in MJD seconds as the filler
    computes it: ``(DATE - JD(MJD 0) + TIME) * 86400``; INTTIM as float32 (None without an INTTIM column)."""
    widths = {"D": 8, "E": 4, "J": 4, "I": 2, "K": 8, "B": 1, "L": 1, "A": 1, "X": 1, "C": 8, "M": 16}
    types = {"D": ">f8", "E": ">f4", "J": ">i4", "I": ">i2", "K": ">i8"}
    offsets, position = {}, 0
    for k in range(1, int(header["TFIELDS"]) + 1):
        name = str(header[f"TTYPE{k}"]).strip()
        form = str(header[f"TFORM{k}"]).strip()
        digits = "".join(ch for ch in form if ch.isdigit())
        count = int(digits) if digits else 1
        letter = form.lstrip("0123456789")[:1]
        if letter not in widths:
            raise NotImplementedError(f"UV_DATA column {name} has unsupported TFORM {form}")
        offsets[name] = (position, types.get(letter))
        position += (count if letter != "X" else (count + 7) // 8) * widths[letter]
    row_bytes, nrow = int(header["NAXIS1"]), int(header["NAXIS2"])
    if position != row_bytes or nrow < 1:
        raise NotImplementedError(f"UV_DATA of {path}: row size {position} != NAXIS1 {row_bytes} or no rows")

    with open(path, "rb") as handle:
        rows = []
        for row in (0, nrow - 1):
            handle.seek(data_offset + row * row_bytes)
            rows.append(handle.read(row_bytes))

    def value(row: bytes, name: str):
        """Column ``name`` of a raw row."""
        start, kind = offsets[name]
        return np.frombuffer(row[start:start + np.dtype(kind).itemsize], dtype=kind)[0]

    def row_time(row: bytes) -> float:
        """MJD seconds of a raw row."""
        day = float(value(row, "DATE")) - JD_MJD0
        if "TIME" in offsets:
            day = day + float(value(row, "TIME"))
        return day * DAY

    inttim = np.float32(value(rows[1], "INTTIM")) if "INTTIM" in offsets else None
    return row_time(rows[0]), row_time(rows[1]), inttim


def scan_idi_file(path) -> dict:
    """The headers and small tables of one FITS-IDI file (UV_DATA only as its header and first/last times).

    Returns
    -------
    dict
        ``path``, ``correlat`` (primary CORRELAT, trimmed), ``corver`` (primary FXCORVER, float),
        ``vanvleck`` (primary VANVLECK, float), ``hdus``: one dict per binary table in file order with
        ``name``, ``header`` (astropy Header copy), ``data`` (``{column: numpy array / list of str}``, None for
        UV_DATA), ``repeat`` (``{column: (repeat count, TFORM letter)}``) and, for UV_DATA, ``times`` (see
        :func:`_uv_times`).
    """
    from astropy.io import fits
    path = str(path)
    hdus = []
    with fits.open(path, memmap=True) as hdulist:
        primary = hdulist[0].header
        correlat = str(primary.get("CORRELAT", "")).strip()
        try:
            corver = float(primary.get("FXCORVER") or 0.0)
        except (TypeError, ValueError):
            corver = 0.0
        try:
            vanvleck = float(primary.get("VANVLECK") or 0.0)
        except (TypeError, ValueError):
            vanvleck = 0.0
        for hdu in hdulist[1:]:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            name = str(hdu.header.get("EXTNAME", "")).strip()
            header = hdu.header.copy()
            repeat = {str(header[f"TTYPE{k}"]).strip(): _repeat(header, str(header[f"TTYPE{k}"]).strip())
                      for k in range(1, int(header.get("TFIELDS", 0)) + 1)}
            entry = {"name": name, "header": header, "data": None, "repeat": repeat}
            if name == "UV_DATA":
                entry["times"] = _uv_times(path, header, int(hdu.fileinfo()["datLoc"]))
            elif name in FILLED_TABLES:
                data = {}
                for column in repeat:
                    values = hdu.data[column] if len(hdu.data) else hdu.data[column][:0]
                    if repeat[column][1] == "A":
                        data[column] = [str(v).rstrip(" \x00") for v in values]
                    else:
                        data[column] = np.array(values)
                entry["data"] = data
            hdus.append(entry)
    return {"path": path, "correlat": correlat, "corver": corver, "vanvleck": vanvleck, "hdus": hdus}


def _nrows(entry: dict) -> int:
    """Rows of a scanned small table."""
    data = entry["data"] or {}
    return len(next(iter(data.values()))) if data else 0


def _is_scalar(entry: dict, column: str) -> bool:
    """True when the filler sees ``column`` as a scalar column (one element; strings are always scalars)."""
    count, letter = entry["repeat"].get(column, (0, ""))
    return letter == "A" or count <= 1


def _cell(entry: dict, column: str, row: int) -> np.ndarray:
    """One cell of a numeric column as a 1-D array (a scalar column gives one element)."""
    return np.atleast_1d(np.asarray(entry["data"][column][row])).reshape(-1)


def _first(scanned: dict, name: str) -> Optional[dict]:
    """The first table named ``name`` of a scanned file, or None."""
    for entry in scanned["hdus"]:
        if entry["name"] == name:
            return entry
    return None


def _fits_date_seconds(text: str) -> float:
    """MJD seconds of a FITS date (``YYYY-MM-DD[Thh:mm:ss[.s]]`` or ``DD/MM/YY``), as FITSDateUtil::fromFITS."""
    text = str(text).strip()
    if "/" in text:
        day, month, year = (int(p) for p in text.split("/")[:3])
        date, rest = _dt.date(1900 + year if year < 100 else year, month, day), ""
    else:
        date_part, _, rest = text.partition("T")
        year, month, day = (int(p) for p in date_part.split("-")[:3])
        date = _dt.date(year, month, day)
    mjd = float(date.toordinal() - MJD0_ORDINAL)
    if rest:
        hours, minutes, seconds = (rest.split(":") + ["0", "0"])[:3]
        mjd += (int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)) / DAY
    return mjd * DAY


def _direction(lon: float, lat: float) -> tuple[float, float]:
    """(lon, lat) [rad] as casacore stores an MVDirection built from them: through its direction cosines."""
    loc = math.cos(lat)
    x, y, z = math.cos(lon) * loc, math.sin(lon) * loc, math.sin(lat)
    return (math.atan2(y, x) if (x != 0.0 or y != 0.0) else 0.0), math.asin(z)


# ---------------------------------------------------------------------------
# What the filler derives from one file
# ---------------------------------------------------------------------------
def _antenna_numbering(scanned: dict) -> dict:
    """ANTENNA_NO -> MS antenna id, in the order the filler meets the numbers (ARRAY_GEOMETRY NOSTA and
    ANTENNA ANTENNA_NO, in file order)."""
    numbering: dict = {}
    for entry in scanned["hdus"]:
        column = {"ARRAY_GEOMETRY": "NOSTA", "ANTENNA": "ANTENNA_NO"}.get(entry["name"])
        if column is None:
            continue
        for number in entry["data"][column]:
            numbering.setdefault(int(number), len(numbering))
    return numbering


def _axes(header) -> dict:
    """The UV_DATA matrix axes (getAxisInfo): ``{CTYPE: (length, CRVAL, CRPIX, CDELT)}``."""
    lengths = [int(header[f"MAXIS{k}"]) for k in range(1, int(header.get("MAXIS", 0)) + 1)]
    axes = {}
    for k, length in enumerate(lengths, start=1):
        ctype = str(header.get(f"CTYPE{k}", "")).strip()
        axes[ctype] = (length, float(header.get(f"CRVAL{k}", 0.0)), float(header.get(f"CRPIX{k}", 0.0)),
                       float(header.get(f"CDELT{k}", 0.0)))
    return axes


def _correlations(header) -> dict:
    """Correlation order of a UV_DATA table (getAxisInfo): ``corr_types`` (sorted casacore Stokes codes),
    ``corr_index``/``corr_swap_index`` (FITS position -> MS position), ``corr_product`` (ncorr x 2)."""
    axes = _axes(header)
    for required in ("COMPLEX", "STOKES", "FREQ"):
        if required not in axes:
            raise NotImplementedError(f"UV_DATA has no {required} axis")
    length, crval, crpix, cdelt = axes["STOKES"]
    fits_types = []
    for i in range(length):
        code = math.floor(crval + (i + 1 - crpix) * cdelt + 0.5)
        if code not in AIPS_TO_STOKES:
            raise NotImplementedError(f"unknown correlation type {code}")
        fits_types.append(AIPS_TO_STOKES[code])
    corr_types = sorted(fits_types)
    if any(t not in STOKES_RECEPTORS for t in corr_types):
        raise NotImplementedError(f"correlations {corr_types} have no receptors (casacore refuses them)")
    corr_index = [max(j for j, t in enumerate(corr_types) if t == fits_type) for fits_type in fits_types]
    swap_index = [max(j for j, t in enumerate(corr_types) if t == STOKES_SWAPPED.get(fits_type, fits_type))
                  for fits_type in fits_types]
    return {"corr_types": corr_types, "corr_index": corr_index, "corr_swap_index": swap_index,
            "corr_product": [list(STOKES_RECEPTORS[t]) for t in corr_types]}


def _uv_setup(header, correlat: str, array_name: str) -> dict:
    """Weights and scaling of one UV_DATA table (describeColumns / getAxisInfo)."""
    if not correlat and array_name == "VLBA":
        correlat = "VLBA"                                   # FITSIDItoMS1 constructor
    weightyp = "CORRELAT" if correlat in ("DIFX", "VLBA") else ""
    if "WEIGHTYP" in header:
        weightyp = str(header["WEIGHTYP"]).strip().upper()
    lengths = [int(header[f"MAXIS{k}"]) for k in range(1, int(header.get("MAXIS", 0)) + 1)]
    axes = _axes(header)
    return {"correlator": correlat, "weightyp": weightyp, "vis_scale": float(header.get("VIS_SCAL", 1.0)),
            "uv_data_has_weights": bool(lengths and lengths[0] == 3), "ncorr": axes["STOKES"][0],
            "nchan": axes["FREQ"][0], "nif": axes["BAND"][0] if "BAND" in axes else 1}


def _array_name(scanned: dict) -> str:
    """The array name the filler knows when it reaches the first UV_DATA (ARRAY_GEOMETRY ARRNAM/TELESCOP)."""
    for entry in scanned["hdus"]:
        if entry["name"] == "UV_DATA":
            break
        if entry["name"] == "ARRAY_GEOMETRY":
            for key in ("ARRNAM", "TELESCOP"):
                if str(entry["header"].get(key, "")).strip() not in ("", "unknown"):
                    return str(entry["header"][key]).strip()
    return ""


def _spectral_windows(entry: dict) -> dict:
    """The subbands of a FREQUENCY table (fillSpectralWindowTable): ``nif``, ``nchan``, ``ref_freq``,
    ``ref_frequency``, ``chan_freq`` (nspw, nchan), ``width`` (|CH_WIDTH|), ``total_bandwidth``, ``sideband``."""
    header = entry["header"]
    nif, nchan = int(header.get("NO_BAND", 0)), int(header.get("NO_CHAN", 0))
    zero_ref, ref_chan = float(header.get("REF_FREQ", 0.0)), float(header.get("REF_PIXL", 0.0))
    nrow = _nrows(entry)
    out = {"nif": nif, "nchan": nchan, "ref_freq": zero_ref, "ref_frequency": [], "chan_freq": [], "width": [],
           "total_bandwidth": [], "sideband": [], "nfreqid": nrow}
    for spw in range(nif * nrow):
        band, row = spw % max(nif, 1), spw // max(1, nif)
        if int(_cell(entry, "FREQID", row)[0]) - 1 != row:
            logger.warning("FITS-IDI FREQUENCY table: FREQID of row {} is not {} (casacore warns too)", row, row + 1)
        ref_freq = zero_ref + float(_cell(entry, "BANDFREQ", row)[band])
        width = float(np.float32(_cell(entry, "CH_WIDTH", row)[band]))
        out["ref_frequency"].append(ref_freq)
        out["chan_freq"].append(np.array([ref_freq + (i + 1 - ref_chan) * width for i in range(nchan)]))
        out["width"].append(abs(width))
        out["total_bandwidth"].append(float(np.float32(_cell(entry, "TOTAL_BANDWIDTH", row)[band])))
        out["sideband"].append(int(_cell(entry, "SIDEBAND", row)[band]))
    return out


def _file_layout(scanned: dict) -> dict:
    """Check the table order of one file and give its UV_DATA times: ``uv`` (list of (first time, end)),
    ``start``/``end`` (the filler's startTime_p and final lastTime_p), ``rdate`` (ARRAY_GEOMETRY RDATE in MJD
    seconds) and ``gain_curve_end`` (lastTime_p when GAIN_CURVE is read).

    The order checks (ARRAY_GEOMETRY, FREQUENCY, SOURCE and ANTENNA before the first UV_DATA, GAIN_CURVE after
    it, tables using RDATE after ARRAY_GEOMETRY) are the cases where casacore's static state is what this module
    assumes: e.g. updateTables adds RDATE to the FEED times and sets the POLARIZATION correlations only for
    tables already read when the first UV_DATA arrives.
    """
    names = [entry["name"] for entry in scanned["hdus"]]
    if "UV_DATA" not in names:
        raise NotImplementedError(f"{scanned['path']}: no UV_DATA table")
    first_uv = names.index("UV_DATA")
    for name in set(names):
        if name in FILLED_TABLES and names.count(name) > 1:
            raise NotImplementedError(f"{scanned['path']}: {names.count(name)} {name} tables")
    for name in HEADER_TABLES:
        if name not in names[:first_uv]:
            raise NotImplementedError(f"{scanned['path']}: no {name} table before UV_DATA")
    for name in ("SYSTEM_TEMPERATURE", "FLAG", "WEATHER", "PHASE-CAL", "CALC"):
        if name in names and names.index(name) < names.index("ARRAY_GEOMETRY"):
            raise NotImplementedError(f"{scanned['path']}: {name} precedes ARRAY_GEOMETRY")
    if "GAIN_CURVE" in names and names.index("GAIN_CURVE") < first_uv:
        raise NotImplementedError(f"{scanned['path']}: GAIN_CURVE precedes UV_DATA")
    geometry = _first(scanned, "ARRAY_GEOMETRY")["header"]
    if not str(geometry.get("RDATE", "")).strip():
        raise NotImplementedError(f"{scanned['path']}: ARRAY_GEOMETRY without RDATE")
    uv, gain_curve_end, end = [], None, None
    for entry in scanned["hdus"]:
        if entry["name"] == "UV_DATA":
            first, last, inttim = entry["times"]
            if inttim is None:
                raise NotImplementedError("UV_DATA without INTTIM")
            end = last + float(inttim)
            uv.append((first, end))
        elif entry["name"] == "GAIN_CURVE":
            gain_curve_end = end
    return {"uv": uv, "start": uv[0][0], "end": end, "rdate": _fits_date_seconds(geometry.get("RDATE", "")),
            "gain_curve_end": gain_curve_end}


# ---------------------------------------------------------------------------
# Setup for the main table
# ---------------------------------------------------------------------------
def _check_later_file(first: dict, later: dict, k: int) -> None:
    """Raise NotImplementedError unless file ``k`` maps onto the first file's antennas, fields, subbands and
    polarization one to one (the only case where MSConcat leaves those subtables and the main-table ids alone)."""
    path = later["path"]
    if _antenna_numbering(later) != _antenna_numbering(first):
        raise NotImplementedError(f"{path}: antenna numbering differs from the first file")
    rows0, rows = _antenna_rows(first), _antenna_rows(later)
    for a in range(len(rows["NAME"])):
        match = -1
        for r in range(len(rows0["NAME"]) - 1, -1, -1):           # matchAntennaAndStation, tolerance 1 m
            if (not rows0["FLAG_ROW"][r] and rows0["NAME"][r] == rows["NAME"][a]
                    and rows0["STATION"][r] == rows["STATION"][a]
                    and np.all(np.abs(rows0["POSITION"][r] - rows["POSITION"][a]) <= 1.0)):
                match = r
                break
        if match != a:
            raise NotImplementedError(f"{path}: antenna {rows['NAME'][a]} does not match antenna {a} of the first "
                                      "file")
    fields0, fields = _field_rows(first, 0.0), _field_rows(later, 0.0)
    for f in range(len(fields["NAME"])):
        same_name = [r for r in range(len(fields0["NAME"])) if fields0["NAME"][r] == fields["NAME"][f]]
        if (not same_name or same_name[-1] != f or fields0["FLAG_ROW"][f] or fields0["NUM_POLY"][f] != 0
                or fields["FLAG_ROW"][f]
                or np.max(np.abs(fields0["PHASE_DIR"][f] - fields["PHASE_DIR"][f])) > 1e-9
                or fields0["_ref"] != fields["_ref"]):
            raise NotImplementedError(f"{path}: field {fields['NAME'][f]} does not match field {f} of the first file")
    spw0, spw = _spectral_windows(_first(first, "FREQUENCY")), _spectral_windows(_first(later, "FREQUENCY"))
    if (spw0["nchan"] != spw["nchan"] or len(spw["ref_frequency"]) > len(spw0["ref_frequency"])
            or any(spw0["ref_frequency"][s] != spw["ref_frequency"][s]
                   or spw0["total_bandwidth"][s] != spw["total_bandwidth"][s]
                   or not np.array_equal(spw0["chan_freq"][s], spw["chan_freq"][s])
                   for s in range(len(spw["ref_frequency"])))):
        raise NotImplementedError(f"{path}: subbands differ from the first file")
    if _correlations(_first(later, "UV_DATA")["header"]) != _correlations(_first(first, "UV_DATA")["header"]):
        raise NotImplementedError(f"{path}: correlations differ from the first file")
    obs0, obs = _observation_row(first, (0.0, 0.0)), _observation_row(later, (0.0, 0.0))
    if obs0["TELESCOPE_NAME"] != obs["TELESCOPE_NAME"]:
        raise NotImplementedError(f"{path}: another telescope ({obs['TELESCOPE_NAME']}): importfitsidi keeps "
                                  "several observation ids")
    logger.debug("FITS-IDI file {} ({}) matches the first file's antennas, fields and subbands", k, path)


def read_idi_setup(idi_files: list) -> dict:
    """What the main-table converter needs from the FITS-IDI files, as casacore's filler derives it.

    Reads headers and the small tables only. Raises NotImplementedError for input the native import cannot
    reproduce exactly (see :func:`_check_later_file`, :func:`_file_layout`).

    Parameters
    ----------
    idi_files : list of str
        The FITS-IDI files, in import order.

    Returns
    -------
    dict
        ``antenna_ids`` ({ANTENNA_NO: MS antenna id}), ``ncorr``, ``nchan``, ``nif`` (bands per FREQID),
        ``nspw``, ``corr_types`` (MS order, casacore Stokes codes), ``corr_index`` / ``corr_swap_index`` (FITS
        Stokes position -> MS correlation position, normal / baseline-swapped), ``corr_product``,
        ``eff_chan_bw`` (per spw, Hz), ``vis_scale``, ``weightyp`` (WEIGHTYP or the correlator default, else
        ""), ``correlator`` (primary CORRELAT, "VLBA" for VLBA data without one), ``uv_data_has_weights``,
        ``ref_freq`` (REF_FREQ, Hz), ``chan_freq`` ((nspw, nchan) Hz), ``nfield_from_source_table``,
        ``source_ids`` (SOURCE_ID values of the first file).
    """
    if not idi_files:
        raise ValueError("no FITS-IDI files")
    started = _time.perf_counter()
    scanned = [scan_idi_file(path) for path in idi_files]
    for one in scanned:
        _file_layout(one)
    first = scanned[0]
    uv_setups = []
    for one in scanned:
        array_name = _array_name(one)
        for entry in one["hdus"]:
            if entry["name"] == "UV_DATA":
                uv_setups.append({**_uv_setup(entry["header"], one["correlat"], array_name),
                                  **_correlations(entry["header"])})
    for other in uv_setups[1:]:
        if other != uv_setups[0]:
            raise NotImplementedError("UV_DATA tables differ in axes, correlations, weights or scaling")
    for k, later in enumerate(scanned[1:], start=1):
        _check_later_file(first, later, k)
    spw = _spectral_windows(_first(first, "FREQUENCY"))
    setup = dict(uv_setups[0])
    if spw["nif"] != setup["nif"] or spw["nchan"] != setup["nchan"]:
        raise NotImplementedError(f"FREQUENCY ({spw['nif']} bands x {spw['nchan']} channels) and UV_DATA "
                                  f"({setup['nif']} x {setup['nchan']}) disagree")
    source = _first(first, "SOURCE")
    ids_column = "SOURCE_ID" if "SOURCE_ID" in source["data"] else "ID_NO."
    source_ids = [int(v) for v in source["data"][ids_column]]
    setup.update({"antenna_ids": _antenna_numbering(first), "nspw": len(spw["ref_frequency"]),
                  "eff_chan_bw": np.asarray(spw["width"], dtype=np.float64), "ref_freq": spw["ref_freq"],
                  "chan_freq": np.asarray(spw["chan_freq"], dtype=np.float64).reshape(-1, spw["nchan"]),
                  "nfield_from_source_table": max([i if i != -1 else 1 for i in source_ids] + [0]),
                  "source_ids": source_ids})
    # NO_LEVELS of every ANTENNA table -> the MS antenna id, as the filler's digiLevels (missing -> 0,
    # an "unsupported" number of digitizer levels: no digital correction, no amplitude scaling).
    digi_levels: dict = {}
    for one in scanned:
        antenna = _first(one, "ANTENNA")
        if antenna is None or not antenna["data"] or "NO_LEVELS" not in antenna["data"]:
            continue
        numbering = _antenna_numbering(one)
        for number, levels in zip(antenna["data"]["ANTENNA_NO"], antenna["data"]["NO_LEVELS"]):
            digi_levels[numbering[int(number)]] = int(levels)
    setup.update({"digi_levels": digi_levels, "corver": float(first["corver"]),
                  "vanvleck": float(first["vanvleck"])})
    logger.debug("read_idi_setup: {} file(s), {} antennas, {} spw x {} chan x {} corr, correlator {!r} in "
                 "{:.2f}s", len(idi_files), len(setup["antenna_ids"]), setup["nspw"], setup["nchan"],
                 setup["ncorr"], setup["correlator"], _time.perf_counter() - started)
    return setup


# ---------------------------------------------------------------------------
# Rows of each subtable, for one file
# ---------------------------------------------------------------------------
def _antenna_rows(scanned: dict) -> dict:
    """ANTENNA rows of one file (fillAntennaTable)."""
    entry = _first(scanned, "ARRAY_GEOMETRY")
    header, data = entry["header"], entry["data"]
    numbering = _antenna_numbering(scanned)
    nrow = len(numbering)
    if nrow > _nrows(entry):
        raise NotImplementedError(f"{scanned['path']}: ANTENNA lists antennas missing from ARRAY_GEOMETRY")
    array_xyz = np.array([float(header.get(key, np.nan)) for key in ("ARRAYX", "ARRAYY", "ARRAYZ")])
    if np.any(np.isnan(array_xyz)):
        raise ValueError(f"{scanned['path']}: illegal ARRAY_GEOMETRY table, no array position")
    arrnam = str(header.get("ARRNAM", "unknown")).strip() if "ARRNAM" in header else "unknown"
    default_diameter = {"ATCA": 22.0, "SMA": 6.0}.get(arrnam, 0.0)
    rows = {"OFFSET": np.zeros((nrow, 3)), "POSITION": np.zeros((nrow, 3)), "TYPE": [""] * nrow,
            "DISH_DIAMETER": np.zeros(nrow), "FLAG_ROW": np.zeros(nrow, dtype=bool), "MOUNT": [""] * nrow,
            "NAME": [""] * nrow, "STATION": [""] * nrow}
    for i in range(nrow):
        number = int(data["NOSTA"][i])
        row = numbering[number]
        diameter = float(np.float32(data["DIAMETER"][i])) if "DIAMETER" in data else default_diameter
        rows["DISH_DIAMETER"][row] = diameter
        rows["MOUNT"][row] = MOUNTS.get(int(data["MNTSTA"][i]), "UNKNOWN")
        name = data["ANNAME"][i]
        label = f"{number:02d}" if number < 10 else str(number)
        rows["NAME"][row] = name if name else f"ANT{label}"
        rows["STATION"][row] = name if name else f"{arrnam}:{label}"
        rows["OFFSET"][row] = np.asarray(data["STAXOF"][i], dtype=np.float32).astype(np.float64)
        rows["TYPE"][row] = "GROUND-BASED"
        rows["POSITION"][row] = array_xyz + np.asarray(data["STABXYZ"][i], dtype=np.float64)
    return rows


def _feed_rows(scanned: dict, layout: dict) -> dict:
    """FEED rows of one file (fillFeedTable + the RDATE updateTables adds)."""
    entry = _first(scanned, "ANTENNA")
    header, data = entry["header"], entry["data"]
    numbering = _antenna_numbering(scanned)
    nif, nant = int(header.get("NO_BAND", 1)), _nrows(entry)
    rows = {key: [] for key in ("POSITION", "BEAM_OFFSET", "POLARIZATION_TYPE", "POL_RESPONSE", "RECEPTOR_ANGLE",
                                "ANTENNA_ID", "BEAM_ID", "FEED_ID", "INTERVAL", "NUM_RECEPTORS", "SPECTRAL_WINDOW_ID",
                                "TIME")}
    rdate = layout["rdate"]                                  # updateTables adds it (ANTENNA precedes UV_DATA)
    single_interval = entry["repeat"]["TIME_INTERVAL"][1] == "E"
    for i in range(nant):
        if _is_scalar(entry, "POLAA"):
            angles_a = np.full(nif, np.float32(data["POLAA"][i]))
            angles_b = np.full(nif, np.float32(data["POLAB"][i]))
        else:
            angles_a, angles_b = _cell(entry, "POLAA", i), _cell(entry, "POLAB", i)
        interval = (float(np.float32(data["TIME_INTERVAL"][i])) if single_interval
                    else float(data["TIME_INTERVAL"][i]))
        for band in range(nif):
            rows["ANTENNA_ID"].append(numbering[int(data["ANTENNA_NO"][i])])
            rows["BEAM_ID"].append(-1)
            rows["FEED_ID"].append(0)
            rows["INTERVAL"].append(interval * DAY)
            rows["SPECTRAL_WINDOW_ID"].append(band)
            rows["TIME"].append(float(data["TIME"][i]) * DAY + rdate)
            rows["NUM_RECEPTORS"].append(2)
            rows["BEAM_OFFSET"].append(np.zeros((2, 2)))
            rows["POLARIZATION_TYPE"].append(np.array([data["POLTYA"][i], data["POLTYB"][i]], dtype=object))
            rows["POL_RESPONSE"].append(np.array([[1, 0], [0, 1]], dtype=np.complex64))
            rows["POSITION"].append(np.zeros(3))
            rows["RECEPTOR_ANGLE"].append(np.array([float(np.float32(angles_a[band])) * DEGREE,
                                                    float(np.float32(angles_b[band])) * DEGREE]))
    return rows


def _field_rows(scanned: dict, time_origin: float) -> dict:
    """FIELD rows of one file (fillFieldTable); ``_ref`` is the direction reference ("J2000"/"B1950").

    SOURCE_ID is -1 in every row, also in the empty rows of missing source ids: partition relabels the source
    ids with -1 when there is no SOURCE table (the filler writes none).
    """
    entry = _first(scanned, "SOURCE")
    data = entry["data"]
    ids = data["SOURCE_ID"] if "SOURCE_ID" in data else data.get("ID_NO.")
    if ids is None:
        raise ValueError(f"{scanned['path']}: no SOURCE_ID column in the SOURCE table")
    nsrc = _nrows(entry)
    if nsrc == 0:
        raise NotImplementedError(f"{scanned['path']}: empty SOURCE table")
    has_equinox, has_epoch = "EQUINOX" in data, "EPOCH" in data
    if not has_equinox and not has_epoch:
        raise NotImplementedError(f"{scanned['path']}: SOURCE table without EQUINOX or EPOCH")

    def reference(row: int) -> str:
        """Direction reference of SOURCE row ``row`` (casacore: EQUINOX contains J2000 / 1950.0B, or EPOCH)."""
        if has_equinox:
            if "J2000" in data["EQUINOX"][row]:
                return "J2000"
            if "1950.0B" in data["EQUINOX"][row]:
                return "B1950"
        else:
            epoch = float(data["EPOCH"][row])
            if abs(epoch - 2000.0) <= 0.01 * max(abs(epoch), 2000.0):
                return "J2000"
            if abs(epoch - 1950.0) <= 0.01:
                return "B1950"
        raise NotImplementedError(f"{scanned['path']}: cannot handle the equinox of SOURCE row {row}")

    if has_equinox:
        column_ref = "B1950" if "1950.0B" in data["EQUINOX"][0] else "J2000"
    else:
        column_ref = "B1950" if abs(float(data["EPOCH"][0]) - 1950.0) <= 0.01 else "J2000"
    nfield = max(max(int(i) - 1 if int(i) - 1 != -2 else 0 for i in ids) + 1, 0)
    rows = {"DIR": [np.zeros((1, 2)) for _ in range(nfield)], "CODE": [""] * nfield,
            "FLAG_ROW": np.ones(nfield, dtype=bool), "NAME": [""] * nfield, "NUM_POLY": np.zeros(nfield, dtype=int),
            "SOURCE_ID": np.full(nfield, -1, dtype=int), "TIME": np.zeros(nfield), "_ref": column_ref}
    for row in range(nsrc):
        field = int(ids[row]) - 1
        if field == -2:
            field = 0                                     # casacore's fix for SMA source ids
        if field < 0:
            raise NotImplementedError(f"{scanned['path']}: invalid source id {int(ids[row])}")
        rows["CODE"][field] = data["CALCODE"][row]
        rows["NAME"][field] = data["SOURCE"][row]
        pmra, pmdec = float(data["PMRA"][row]), float(data["PMDEC"][row])
        num_poly = 0
        if (abs(pmra) > 1e-13 or abs(pmdec) > 1e-13) and abs(pmra) <= 1000.0 and abs(pmdec) <= 1000.0:
            num_poly = 1
        if num_poly == 0:
            ref = reference(row)
            lon, lat = float(data["RAEPO"][row]) * DEGREE, float(data["DECEPO"][row]) * DEGREE
        else:
            ref = "J2000"
            lon, lat = float(data["RAAPP"][row]) * DEGREE, float(data["DECAPP"][row]) * DEGREE
        if ref != column_ref:
            raise NotImplementedError(f"{scanned['path']}: SOURCE rows with different equinoxes")
        directions = [_direction(lon, lat)]
        if num_poly == 1:
            directions.append(_direction(pmra * DEGREE / DAY, pmdec * DEGREE / DAY))
        rows["DIR"][field] = np.array(directions, dtype=np.float64)
        rows["TIME"][field] = time_origin
        rows["NUM_POLY"][field] = num_poly
        rows["FLAG_ROW"][field] = False
    for field in range(nfield):
        if rows["FLAG_ROW"][field]:
            rows["TIME"][field] = time_origin             # updateTables sets every row's time
    rows["PHASE_DIR"] = [cell[0] for cell in rows["DIR"]]
    return rows


def _observation_row(scanned: dict, time_range: tuple) -> dict:
    """The OBSERVATION row of one file (fillObsTables, keywords of its first UV_DATA)."""
    header = _first(scanned, "UV_DATA")["header"]
    observer = str(header.get("OBSERVER", "unknown")).strip() if "OBSERVER" in header else "unknown"
    project = str(header.get("OBSCODE", "")).strip()
    telescope = str(header["TELESCOP"]).strip() if "TELESCOP" in header else _array_name(scanned) or "unknown"
    if telescope in ("", "unknown"):
        telescope = str(header.get("ARRNAM", "unknown")).strip() if "ARRNAM" in header else "unknown"
    return {"TIME_RANGE": [float(time_range[0]), float(time_range[1])], "OBSERVER": observer, "PROJECT": project,
            "TELESCOPE_NAME": telescope, "RELEASE_DATE": float(time_range[0])}


def _history_cards(header) -> list[str]:
    """HISTORY messages fillObsTables takes from a UV_DATA header: the text of every card from the first
    HISTORY card on, until one with no text."""
    cards = list(header.cards)
    start = next((k for k, card in enumerate(cards) if card.keyword == "HISTORY"), None)
    messages = []
    if start is None:
        return messages
    for card in cards[start:]:
        if card.keyword in ("HISTORY", "COMMENT", ""):
            text = card.image[8:].rstrip()
        else:
            text = str(card.comment or "").rstrip()
        if not text:
            break
        messages.append(text)
    return messages


def _syscal_rows(scanned: dict, layout: dict, numbering: dict) -> dict:
    """SYSCAL rows of one file (fillSysCalTable)."""
    entry = _first(scanned, "SYSTEM_TEMPERATURE")
    data, nif = entry["data"], int(entry["header"].get("NO_BAND", 1))
    dual = "TSYS_2" in data
    scalar = _is_scalar(entry, "TSYS_1")
    rows = {key: [] for key in ("ANTENNA_ID", "FEED_ID", "INTERVAL", "SPECTRAL_WINDOW_ID", "TIME", "TSYS")}
    for i in range(_nrows(entry)):
        tsys1 = _cell(entry, "TSYS_1", i).astype(np.float32)
        tsys2 = _cell(entry, "TSYS_2", i).astype(np.float32) if dual else None
        for band in range(nif):
            rows["ANTENNA_ID"].append(_antenna_id(numbering, data["ANTENNA_NO"][i]))
            rows["FEED_ID"].append(0)
            rows["TIME"].append(float(data["TIME"][i]) * DAY + layout["rdate"])
            rows["INTERVAL"].append(float(np.float32(data["TIME_INTERVAL"][i])) * DAY)
            rows["SPECTRAL_WINDOW_ID"].append(band)
            index = 0 if scalar else band
            cell = [tsys1[index]] + ([tsys2[index]] if dual else [])
            rows["TSYS"].append(np.array(cell, dtype=np.float32))
    return rows


def _antenna_id(numbering: dict, number) -> int:
    """The MS antenna id of an ANTENNA_NO (casacore throws for an unknown one)."""
    number = int(number)
    if number not in numbering:
        raise ValueError(f"no mapping for ANTENNA_NO {number}")
    return numbering[number]


def _flag_cmd_rows(scanned: dict, layout: dict, numbering: dict, nfreqid: int) -> dict:
    """FLAG_CMD rows of one file (fillFlagCmdTable): one flag command per FLAG row with ARRAY 0."""
    entry = _first(scanned, "FLAG")
    header, data = entry["header"], entry["data"]
    nstokes, first_stokes, nif = int(header.get("NO_STKD", 0)), int(header.get("STK_1", -1)), int(
        header.get("NO_BAND", 1))
    stokes = ("RR", "LL", "RL", "LR", "XX", "YY", "XY", "YX")
    if first_stokes >= 0 or nstokes < 1 or nstokes > 4 or first_stokes - nstokes < -9:
        raise NotImplementedError(f"FLAG table: unsupported STK_1 {first_stokes} / NO_STKD {nstokes}")
    if entry["repeat"]["TIMERANG"][1] != "E":
        raise NotImplementedError("FLAG table: TIMERANG is not single precision")
    rows = {key: [] for key in ("APPLIED", "COMMAND", "INTERVAL", "LEVEL", "REASON", "SEVERITY", "TIME", "TYPE")}
    for i in range(_nrows(entry)):
        if int(data["ARRAY"][i]) != 0:
            logger.warning("FLAG row {}: flagging by array number not supported (skipped, as casacore)", i)
            continue
        ants, chans = _cell(entry, "ANTS", i), _cell(entry, "CHANS", i)
        ant1, ant2, chan1, chan2 = int(ants[0]), int(ants[1]), int(chans[0]), int(chans[1])
        for ant in (ant1, ant2):
            if ant != 0 and ant not in numbering:
                raise ValueError(f"FLAG row {i}: no mapping for antenna {ant}")
        if chan1 > chan2 or chan1 < -1 or chan2 < -1:
            raise ValueError(f"FLAG row {i}: incorrect channel range {chan1}-{chan2}")
        span = _cell(entry, "TIMERANG", i).astype(np.float32)
        middle = np.float32(np.float32(span[0] + span[1]) / np.float32(2))
        rows["TIME"].append(float(middle) * DAY + layout["rdate"])
        rows["INTERVAL"].append(float(np.float32(span[1] - span[0])) * DAY)
        rows["TYPE"].append("FLAG")
        rows["REASON"].append(data["REASON"][i])
        rows["LEVEL"].append(0)
        rows["SEVERITY"].append(int(data["SEVERITY"][i]))
        rows["APPLIED"].append(False)
        command = ""
        id1 = -1 if ant1 == 0 else numbering[ant1]
        id2 = -1 if ant2 == 0 else numbering[ant2]
        if id1 != -1:
            command += f"antenna='{id1}" + (f"&{id2}" if id2 != -1 else "") + "'"
        source = int(data["SOURCE_ID"][i])
        if source != 0:
            command += (" " if command else "") + f"field='{source - 1}'"
        freqid = int(data["FREQID"][i])
        if freqid > 0:
            freqids = [freqid - 1]
        else:
            freqids = list(range(max(nfreqid, 1)))
        bands = _cell(entry, "BANDS", i)
        spws, need_spw = [], False
        for group in freqids:
            for band in range(len(bands)):
                if bands[band]:
                    spws.append(str(group * nif + band))
                else:
                    need_spw = True
        spw_text = ",".join(spws) if need_spw else "*"
        if need_spw or chan1 > 0:
            command += (" " if command else "") + f"spw='{spw_text}"
            if chan1 > 0:
                command += f":{chan1 - 1}~{chan2 - 1}'"
            command += "'"                                # casacore closes the quote twice with a channel range
        pflags = _cell(entry, "PFLAGS", i)
        hands, need_corr = [], False
        for k in range(nstokes):
            if pflags[k]:
                hands.append(stokes[k - first_stokes - 1])
            else:
                need_corr = True
        if need_corr:
            command += (" " if command else "") + f"correlation='{','.join(hands)}'"
        rows["COMMAND"].append(command)
    return rows


def _weather_rows(scanned: dict, layout: dict, numbering: dict) -> tuple[dict, tuple]:
    """WEATHER rows of one file (fillWeatherTable) and the optional columns the filler adds."""
    entry = _first(scanned, "WEATHER")
    data = entry["data"]
    water, electron = "WVR_H2O" in data, "IONOS_ELECTRON" in data
    # casacore attaches its WVR_H2O reader to IONOS_ELECTRON when that column exists (both outputs take it).
    h2o_source = "IONOS_ELECTRON" if electron else "WVR_H2O"
    present = ("DEW_POINT",) + (("H2O",) if water else ()) + (("IONOS_ELECTRON",) if electron else ()) + (
        "PRESSURE", "TEMPERATURE", "WIND_DIRECTION", "WIND_SPEED")
    rows = {key: [] for key in ("ANTENNA_ID", "INTERVAL", "TIME") + present}
    for i in range(_nrows(entry)):
        rows["ANTENNA_ID"].append(_antenna_id(numbering, data["ANTENNA_NO"][i]))
        rows["TIME"].append(float(data["TIME"][i]) * DAY + layout["rdate"])
        rows["INTERVAL"].append(float(np.float32(data["TIME_INTERVAL"][i])) * DAY)
        rows["DEW_POINT"].append(float(np.float32(data["DEWPOINT"][i])) + 273.15)
        rows["PRESSURE"].append(float(np.float32(data["PRESSURE"][i])))
        rows["TEMPERATURE"].append(float(np.float32(data["TEMPERATURE"][i])) + 273.15)
        rows["WIND_DIRECTION"].append(float(np.float32(data["WIND_DIRECTION"][i])) * math.pi / 180.0)
        rows["WIND_SPEED"].append(float(np.float32(data["WIND_VELOCITY"][i])))
        if water:
            rows["H2O"].append(float(np.float32(data[h2o_source][i])))
        if electron:
            rows["IONOS_ELECTRON"].append(float(np.float32(data[h2o_source][i])))
    for key in present:
        rows[key] = np.asarray(rows[key], dtype=np.float32)
    return rows, present


def _gain_curve_rows(scanned: dict, layout: dict, numbering: dict) -> dict:
    """GAIN_CURVE rows of one file (handleGainCurve). A row the filler skips (unsupported type) stays in the
    table with default values, as casacore adds it before checking."""
    entry = _first(scanned, "GAIN_CURVE")
    header, data = entry["header"], entry["data"]
    nif, ntabs = int(header.get("NO_BAND", 1)), int(header.get("NO_TABS", 1))
    dual = "TYPE_2" in data
    scalar = _is_scalar(entry, "TYPE_1")
    start, end = layout["start"], layout["gain_curve_end"]
    rows = {key: [] for key in ("ANTENNA_ID", "FEED_ID", "SPECTRAL_WINDOW_ID", "TIME", "INTERVAL", "TYPE",
                                "NUM_POLY", "GAIN", "SENSITIVITY")}
    ytyp, nterm = 0, 0
    for i in range(_nrows(entry)):
        for band in range(nif):
            defaults = {"ANTENNA_ID": 0, "FEED_ID": 0, "SPECTRAL_WINDOW_ID": 0, "TIME": 0.0, "INTERVAL": 0.0,
                        "TYPE": "", "NUM_POLY": 0, "GAIN": None, "SENSITIVITY": None}
            row = dict(defaults)
            skip = False
            pick = 0 if scalar else band
            if not scalar or band == 0:
                kinds = [int(_cell(entry, f"{key}_1", i)[pick]) for key in ("TYPE", "X_TYP", "Y_TYP", "NTERM")]
                if dual and kinds != [int(_cell(entry, f"{key}_2", i)[pick]) for key in ("TYPE", "X_TYP", "Y_TYP",
                                                                                         "NTERM")]:
                    logger.warning("GAIN_CURVE row {}: distinct gain curve types per polarisation (skipped)", i)
                    skip = True
                elif kinds[0] != 2:
                    logger.warning("GAIN_CURVE row {}: unsupported gain curve type {} (skipped)", i, kinds[0])
                    skip = True
                else:
                    ytyp, nterm = kinds[2], kinds[3]
            if not skip and ytyp not in (1, 2):
                logger.warning("GAIN_CURVE row {}: unsupported gain curve coordinate type {} (skipped)", i, ytyp)
                skip = True
            if not skip:
                row["ANTENNA_ID"] = _antenna_id(numbering, data["ANTENNA_NO"][i])
                row["FEED_ID"] = -1
                row["SPECTRAL_WINDOW_ID"] = band
                row["TYPE"] = "POWER(EL)" if ytyp == 1 else "POWER(ZA)"
                row["TIME"] = (start + end) / 2
                row["INTERVAL"] = end - start
                row["NUM_POLY"] = nterm
                gains = []
                for pol in ("1", "2") if dual else ("1",):
                    values = _cell(entry, f"GAIN_{pol}", i).astype(np.float32)
                    if scalar and ntabs == 1:
                        gains.append(np.full(nterm, values[0], dtype=np.float32))
                    else:
                        gains.append(values[band * ntabs:band * ntabs + nterm])
                row["GAIN"] = np.array(gains, dtype=np.float32).T.copy()       # (nterm, npol) C order
                row["SENSITIVITY"] = np.array([_cell(entry, f"SENS_{pol}", i).astype(np.float32)[pick]
                                               for pol in (("1", "2") if dual else ("1",))], dtype=np.float32)
            for key in rows:
                rows[key].append(row[key])
    return rows


def _phase_cal_rows(scanned: dict, layout: dict, numbering: dict) -> dict:
    """PHASE_CAL rows of one file (handlePhaseCal)."""
    entry = _first(scanned, "PHASE-CAL")
    header, data = entry["header"], entry["data"]
    nif, ntones = int(header.get("NO_BAND", 1)), int(header.get("NO_TONES", 0))
    dual = "PC_FREQ_2" in data
    scalar = _is_scalar(entry, "PC_FREQ_1")
    rows = {key: [] for key in ("ANTENNA_ID", "FEED_ID", "SPECTRAL_WINDOW_ID", "TIME", "INTERVAL", "NUM_TONES",
                                "TONE_FREQUENCY", "PHASE_CAL", "CABLE_CAL")}
    pols = ("1", "2") if dual else ("1",)
    for i in range(_nrows(entry)):
        for band in range(nif):
            rows["ANTENNA_ID"].append(_antenna_id(numbering, data["ANTENNA_NO"][i]))
            rows["FEED_ID"].append(-1)
            rows["SPECTRAL_WINDOW_ID"].append(band)
            rows["TIME"].append(float(data["TIME"][i]) * DAY + layout["rdate"])
            rows["INTERVAL"].append(float(np.float32(data["TIME_INTERVAL"][i])) * DAY)
            rows["NUM_TONES"].append(ntones)
            freqs, phases = [], []
            for pol in pols:
                freq = _cell(entry, f"PC_FREQ_{pol}", i).astype(np.float64)
                real = _cell(entry, f"PC_REAL_{pol}", i).astype(np.float32)
                imag = _cell(entry, f"PC_IMAG_{pol}", i).astype(np.float32)
                if scalar:
                    freqs.append(np.full(ntones, freq[0]))
                    phases.append(np.full(ntones, np.complex64(complex(real[0], imag[0]))))
                else:
                    tones = slice(band * ntones, band * ntones + ntones)
                    freqs.append(freq[tones])
                    phases.append((real[tones] + 1j * imag[tones].astype(np.float64)).astype(np.complex64))
            rows["TONE_FREQUENCY"].append(np.array(freqs, dtype=np.float64).T.copy())
            rows["PHASE_CAL"].append(np.array(phases, dtype=np.complex64).T.copy())
            rows["CABLE_CAL"].append(float(data["CABLE_CAL"][i]))
    return rows


def _eop_rows(scanned: dict, layout: dict, observation_id: int) -> dict:
    """EARTH_ORIENTATION rows of one file (handleCalc), with the observation id concatenation gives them."""
    entry = _first(scanned, "CALC")
    data = entry["data"]
    rows = {key: [] for key in ("TIME", "OBSERVATION_ID", "UT1_UTC", "PM", "TYPE")}
    for i in range(_nrows(entry)):
        rows["TIME"].append(float(data["TIME"][i]) * DAY + layout["rdate"])
        rows["OBSERVATION_ID"].append(observation_id)
        rows["UT1_UTC"].append(float(data["UT1-UTC"][i]))
        rows["PM"].append(_cell(entry, "WOBXY", i).astype(np.float64) * ARCSEC)
        kinds = (data["UT1 TYPE"][i], data["WOB TYPE"][i])
        kind = ""
        for code, label in (("E", "PREDICTED"), ("P", "PRELIMINARY"), ("F", "FINAL")):
            if code in kinds:
                kind = label
                break
        rows["TYPE"].append(kind)
    return rows


def _merge_feed(rows: dict, later: dict) -> None:
    """FEED rows of a later file merged into ``rows`` as MSConcat::copyAntennaAndFeed does with matching
    antennas: every feed must match an existing row (else NotImplementedError); a matched row's TIME and INTERVAL
    are widened to cover both validity ranges (MSFeedColumns::matchFeed)."""
    later_antennas = np.asarray(later["ANTENNA_ID"])
    for antenna in sorted(set(later_antennas.tolist())):
        own = [k for k in range(len(rows["ANTENNA_ID"])) if rows["ANTENNA_ID"][k] == antenna]
        others = [k for k in range(len(later_antennas)) if later_antennas[k] == antenna]
        if len(own) != len(others):
            raise NotImplementedError(f"FEED rows of antenna {antenna} differ between files")
        matched: list = []
        for k in others:
            time, half = later["TIME"][k], later["INTERVAL"][k] / 2.0
            match = -1
            for r in range(len(rows["ANTENNA_ID"]) - 1, -1, -1):
                if r in matched:
                    continue
                if (rows["ANTENNA_ID"][r] == antenna and rows["FEED_ID"][r] == later["FEED_ID"][k]
                        and rows["SPECTRAL_WINDOW_ID"][r] == later["SPECTRAL_WINDOW_ID"][k]
                        and rows["NUM_RECEPTORS"][r] == later["NUM_RECEPTORS"][k]
                        and np.array_equal(rows["POSITION"][r], later["POSITION"][k])
                        and np.array_equal(rows["BEAM_OFFSET"][r], later["BEAM_OFFSET"][k])
                        and list(rows["POLARIZATION_TYPE"][r]) == list(later["POLARIZATION_TYPE"][k])
                        and np.array_equal(rows["RECEPTOR_ANGLE"][r], later["RECEPTOR_ANGLE"][k])
                        and np.array_equal(rows["POL_RESPONSE"][r], later["POL_RESPONSE"][k])):
                    match = r
                    break
            if match < 0:
                raise NotImplementedError(f"FEED row of antenna {antenna} has no counterpart in the first file")
            own_half = rows["INTERVAL"][match] / 2.0 if rows["INTERVAL"][match] != 0.0 else 5e17
            own_time = rows["TIME"][match]
            if not (own_time - own_half <= time - half and own_time + own_half >= time + half):
                new_time = (own_time + time) / 2.0
                latest = max(own_time + own_half, time + half)
                rows["TIME"][match] = new_time
                rows["INTERVAL"][match] = 2 * (latest - new_time)
            matched.append(match)


# ---------------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------------
def _task_messages(task: str, params: list, version: str) -> list[str]:
    """The messages CASA's write_history records for a task call (``"%-11s = "`` + value, strings quoted)."""
    messages = [f"taskname={task}", version]
    for name, value in params:
        text = f"{name:<11s} = "
        messages.append(text + (f'"{value}"' if isinstance(value, str) else str(value)))
    return messages


def _history_rows(first: dict, idi_files: list, ms_dir: Path, specframe: str, params: dict) -> dict:
    """HISTORY rows: the FITS HISTORY cards of the first file's UV_DATA tables (fillObsTables, with its
    row-index quirk for several UV_DATA tables), then the importfitsidi and partition calls."""
    rows = {key: [] for key in ("APP_PARAMS", "CLI_COMMAND", "APPLICATION", "MESSAGE", "OBJECT_ID",
                                "OBSERVATION_ID", "ORIGIN", "PRIORITY", "TIME")}
    for entry in first["hdus"]:
        if entry["name"] != "UV_DATA":
            continue
        header = entry["header"]
        date = str(header.get("DATE-OBS", "")).strip() or str(header.get("DATE", "")).strip() or "2000-01-01"
        stamp = _fits_date_seconds(date)
        messages = _history_cards(header)
        for _ in messages:                                     # rows are added at the end ...
            for key, value in (("APP_PARAMS", None), ("CLI_COMMAND", None), ("APPLICATION", ""), ("MESSAGE", ""),
                               ("OBJECT_ID", 0), ("OBSERVATION_ID", 0), ("ORIGIN", ""), ("PRIORITY", ""),
                               ("TIME", 0.0)):
                rows[key].append(value)
        for row, message in enumerate(messages):               # ... but written from row 0 on
            rows["OBSERVATION_ID"][row] = 0
            rows["TIME"][row] = stamp
            rows["PRIORITY"][row] = "NORMAL"
            rows["ORIGIN"][row] = "FITSIDItoMS1::fillObsTables"
            rows["APPLICATION"][row] = "ms"
            rows["MESSAGE"][row] = message
    mms = ms_dir.parent.parent if ms_dir.parent.name == "SUBMSS" else ms_dir
    vis = params.get("vis", str(mms.with_name(mms.stem + ".ms")))
    version = params.get("version", HISTORY_VERSION)
    gap = float(params.get("scanreindexgap_s", 15.0))
    # importfitsidi pops the first file off its list argument before recording the parameters.
    import_params = [("fitsidifile", list(idi_files[1:])),
                     ("vis", vis), ("constobsid", True), ("scanreindexgap_s", gap), ("specframe", specframe),
                     ("coordframe", "")]
    calls = [("importfitsidi", import_params)]
    if ms_dir.parent.name == "SUBMSS":
        calls.append(("partition", [("vis", vis), ("outputvis", params.get("outputvis", str(mms)))]
                      + list(PARTITION_PARAMS)))
    for task, task_params in calls:
        for message in _task_messages(task, task_params, version):
            rows["APP_PARAMS"].append(np.array([""], dtype=object))
            rows["CLI_COMMAND"].append(np.array([""], dtype=object))
            rows["APPLICATION"].append("ms")
            rows["MESSAGE"].append(message)
            rows["OBJECT_ID"].append(0)
            rows["OBSERVATION_ID"].append(-1)
            rows["ORIGIN"].append(task)
            rows["PRIORITY"].append("INFO")
            rows["TIME"].append(_time.time() + UNIX_EPOCH_MJD_SECONDS)
    return rows


def write_part_history(first_part_dir, part_dir) -> None:
    """HISTORY of a further part of the Multi-MS: the first part's rows without partition's (partition writes
    its parameters through the Multi-MS, i.e. into the first part's HISTORY only)."""
    from ..casacore_format.writer import create_table, read_table_values, select_rows
    columns, values, nrow, keywords, info = read_table_values(Path(first_part_dir) / "HISTORY")
    keep = [k for k in range(nrow) if values["ORIGIN"][k] != "partition"] if nrow else []
    values = select_rows(values, keep) if nrow else {}
    for column in columns:
        column.shape_col_def, column.shape_col = False, ()
    create_table(Path(part_dir) / "HISTORY", columns, len(keep), values, keywords=keywords,
                 info=(info["type"], info["subType"]), readme=info["readme"], manager_name="StandardStMan")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _write(ms_dir: Path, name: str, nrow: int, values: dict, **column_options) -> None:
    """Write subtable ``name`` (no keywords, empty table.info, one StandardStMan) and its table comment."""
    from ..casacore_format.tabledat import read_table_dat, write_table_dat
    from ..casacore_format.writer import create_table
    columns = _columns(name, **column_options)
    names = [c.name for c in columns]
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"{name}: values for unknown columns {sorted(unknown)}")
    prepared = {}
    for column in columns:
        if column.name not in values:
            continue
        value = values[column.name]
        if column.is_array and not column.shape:
            prepared[column.name] = list(value)
        elif column.dtype == 11 and not column.is_array:
            prepared[column.name] = [str(v) for v in value]
        else:
            prepared[column.name] = np.asarray(value)
    create_table(ms_dir / name, columns, nrow, prepared, keywords={}, info=("", ""), readme="",
                 manager_name="StandardStMan")
    if name in TABLE_COMMENTS:
        dat = read_table_dat(ms_dir / name)
        dat.desc_comment = TABLE_COMMENTS[name]
        write_table_dat(ms_dir / name, dat)


def _concatenate(target: dict, extra: dict) -> None:
    """Append the rows of ``extra`` to ``target`` (both ``{column: list}``); the columns must be the same (a later
    file's table with other optional columns would leave cells casacore fills with defaults: not reproduced)."""
    if set(extra) != set(target):
        raise NotImplementedError(f"a later file's table has other columns ({sorted(set(extra) ^ set(target))})")
    for key, value in extra.items():
        target[key] = list(target[key]) + list(value)


def write_subtables(idi_files: list, ms_dir, setup: dict, summary: dict, *, specframe: str = "GEO",
                    history_params: Optional[dict] = None) -> list[str]:
    """Write every subtable of the first part of the Multi-MS into the existing directory ``ms_dir``.

    Parameters
    ----------
    idi_files : list of str
        The FITS-IDI files, in import order.
    ms_dir : path
        The first part (``<mms>/SUBMSS/<mms>.0000.ms``) or a plain MS; its main table is written elsewhere.
    setup : dict
        :func:`read_idi_setup` of the same files (checked against them: ``nspw``, ``antenna_ids``).
    summary : dict
        What the main-table pass knows: ``time_start`` (MJD s of the first row), ``time_end`` (last row time +
        its INTTIM, casacore's lastTime_p), ``nfield`` (max FIELD_ID + 1), ``nspw``, ``nscan``. The subtables
        take their times from the files themselves (first/last row of each UV_DATA table, as the filler does per
        file); the summary is only checked against them (a mismatch is logged).
    specframe : str
        importfitsidi ``specframe`` (MEAS_FREQ_REF of every subband).
    history_params : dict, optional
        Values recorded in HISTORY: ``vis`` (default ``<mms stem>.ms`` beside the Multi-MS), ``outputvis``
        (default the Multi-MS), ``scanreindexgap_s`` (default 15.0), ``version`` (default
        :data:`HISTORY_VERSION`).

    Returns
    -------
    list of str
        The subtables written, in the order of the main-table keywords of a part (after ``MS_VERSION``).
    """
    started = _time.perf_counter()
    ms_dir = Path(ms_dir)
    if not ms_dir.is_dir():
        raise FileNotFoundError(f"{ms_dir} is not a directory")
    frame = (specframe or "GEO").upper()
    if frame not in SPECTRAL_FRAMES:
        raise ValueError(f"specframe {specframe!r} invalid; use one of {sorted(SPECTRAL_FRAMES)}")
    files = [scan_idi_file(path) for path in idi_files]
    layouts = [_file_layout(one) for one in files]
    for k, later in enumerate(files[1:], start=1):
        _check_later_file(files[0], later, k)
    first, layout = files[0], layouts[0]
    numbering = _antenna_numbering(first)
    if setup.get("antenna_ids") and dict(setup["antenna_ids"]) != numbering:
        raise ValueError("setup antenna_ids do not match the FITS-IDI files")
    time_range = (min(one["start"] for one in layouts), max(one["end"] for one in layouts))
    if summary and (summary.get("time_start") != time_range[0] or summary.get("time_end") != time_range[1]):
        logger.warning("write_subtables: main-table time range {}..{} differs from the files' {}..{}",
                       summary.get("time_start"), summary.get("time_end"), time_range[0], time_range[1])
    written = []

    antenna = _antenna_rows(first)
    _write(ms_dir, "ANTENNA", len(antenna["NAME"]), antenna)
    written.append("ANTENNA")

    spw = _spectral_windows(_first(first, "FREQUENCY"))
    nspw = len(spw["ref_frequency"])
    if setup.get("nspw") is not None and int(setup["nspw"]) != nspw:
        raise ValueError(f"setup nspw {setup['nspw']} != {nspw} subbands in the FREQUENCY table")
    _write(ms_dir, "DATA_DESCRIPTION", nspw, {"FLAG_ROW": np.zeros(nspw, dtype=bool),
                                             "POLARIZATION_ID": np.zeros(nspw, dtype=np.int32),
                                             "SPECTRAL_WINDOW_ID": np.arange(nspw, dtype=np.int32)})
    written.append("DATA_DESCRIPTION")

    feed = _feed_rows(first, layout)
    for later, later_layout in zip(files[1:], layouts[1:]):
        _merge_feed(feed, _feed_rows(later, later_layout))
    _write(ms_dir, "FEED", len(feed["ANTENNA_ID"]), feed)
    written.append("FEED")

    flags = {}
    if _first(first, "FLAG") is not None:
        flags = _flag_cmd_rows(first, layout, numbering, spw["nfreqid"])
    _write(ms_dir, "FLAG_CMD", len(flags.get("TIME", [])), flags)
    written.append("FLAG_CMD")

    field = _field_rows(first, layout["start"])
    field_values = {key: field[key] for key in ("CODE", "FLAG_ROW", "NAME", "NUM_POLY", "SOURCE_ID", "TIME")}
    for column in ("DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR"):
        field_values[column] = [np.array(cell) for cell in field["DIR"]]
    _write(ms_dir, "FIELD", len(field["NAME"]), field_values, field_ref="ICRS" if field["_ref"] == "J2000"
           else field["_ref"])
    written.append("FIELD")

    history = _history_rows(first, [str(f) for f in idi_files], ms_dir, frame, dict(history_params or {}))
    _write(ms_dir, "HISTORY", len(history["MESSAGE"]), history)
    written.append("HISTORY")

    observation = _observation_row(first, time_range)
    _write(ms_dir, "OBSERVATION", 1, {"TIME_RANGE": np.array([observation["TIME_RANGE"]]), "LOG": [None],
                                      "SCHEDULE": [None], "FLAG_ROW": np.zeros(1, dtype=bool),
                                      "OBSERVER": [observation["OBSERVER"]], "PROJECT": [observation["PROJECT"]],
                                      "RELEASE_DATE": np.array([observation["RELEASE_DATE"]]),
                                      "SCHEDULE_TYPE": [""], "TELESCOPE_NAME": [observation["TELESCOPE_NAME"]]})
    written.append("OBSERVATION")

    correlations = _correlations(_first(first, "UV_DATA")["header"])
    ncorr = int(_first(first, "FREQUENCY")["header"].get("NO_STKD", len(correlations["corr_types"])))
    _write(ms_dir, "POLARIZATION", 1, {"CORR_TYPE": [np.array(correlations["corr_types"], dtype=np.int32)],
                                       "CORR_PRODUCT": [np.array(correlations["corr_product"], dtype=np.int32)],
                                       "FLAG_ROW": np.zeros(1, dtype=bool), "NUM_CORR": np.array([ncorr])})
    written.append("POLARIZATION")

    _write(ms_dir, "PROCESSOR", 0, {})
    written.append("PROCESSOR")

    nchan = spw["nchan"]
    _write(ms_dir, "SPECTRAL_WINDOW", nspw, {
        "MEAS_FREQ_REF": np.full(nspw, SPECTRAL_FRAMES[frame], dtype=np.int32),
        "CHAN_FREQ": spw["chan_freq"], "REF_FREQUENCY": np.array(spw["ref_frequency"]),
        "CHAN_WIDTH": [np.full(nchan, w) for w in spw["width"]],
        "EFFECTIVE_BW": [np.full(nchan, w) for w in spw["width"]],
        "RESOLUTION": [np.full(nchan, w) for w in spw["width"]],
        "FLAG_ROW": np.zeros(nspw, dtype=bool),
        "FREQ_GROUP": np.array([s // max(spw["nif"], 1) for s in range(nspw)], dtype=np.int32),
        "FREQ_GROUP_NAME": ["none"] * nspw,
        "IF_CONV_CHAIN": np.array([s % max(spw["nif"], 1) for s in range(nspw)], dtype=np.int32),
        "NAME": ["none"] * nspw, "NET_SIDEBAND": np.array(spw["sideband"], dtype=np.int32),
        "NUM_CHAN": np.full(nspw, nchan, dtype=np.int32), "TOTAL_BANDWIDTH": np.array(spw["total_bandwidth"])})
    written.append("SPECTRAL_WINDOW")

    for name in ("STATE", "SOURCE", "POINTING"):
        _write(ms_dir, name, 0, {})
        written.append(name)

    def merged(name: str, rows_of) -> Optional[dict]:
        """Rows of subtable ``name`` after concatenation: the first file's, then every later file's when the
        first file's table has rows (MSConcat ignores later tables otherwise); None without a first table."""
        source = {"SYSCAL": "SYSTEM_TEMPERATURE", "WEATHER": "WEATHER", "GAIN_CURVE": "GAIN_CURVE",
                  "PHASE_CAL": "PHASE-CAL", "EARTH_ORIENTATION": "CALC"}[name]
        if _first(first, source) is None:
            return None
        rows = rows_of(0)
        if not len(next(iter(rows.values()))):
            return rows
        for k in range(1, len(files)):
            if _first(files[k], source) is not None:
                _concatenate(rows, rows_of(k))
        return rows

    syscal = merged("SYSCAL", lambda k: _syscal_rows(files[k], layouts[k], _antenna_numbering(files[k])))
    if syscal is not None and len(syscal["TIME"]):
        _write(ms_dir, "SYSCAL", len(syscal["TIME"]), syscal)
        written.append("SYSCAL")

    weather_columns: tuple = ()
    if _first(first, "WEATHER") is not None:
        weather_columns = _weather_rows(first, layout, numbering)[1]
    weather = merged("WEATHER", lambda k: _weather_rows(files[k], layouts[k], _antenna_numbering(files[k]))[0])
    if weather is not None and len(weather["TIME"]):
        _write(ms_dir, "WEATHER", len(weather["TIME"]), weather, weather_columns=weather_columns)
        written.append("WEATHER")

    generic = {
        "EARTH_ORIENTATION": lambda k: _eop_rows(files[k], layouts[k], k),
        "GAIN_CURVE": lambda k: _gain_curve_rows(files[k], layouts[k], _antenna_numbering(files[k])),
        "PHASE_CAL": lambda k: _phase_cal_rows(files[k], layouts[k], _antenna_numbering(files[k])),
    }
    for entry in first["hdus"]:
        name = GENERIC_SUBTABLES.get(entry["name"])
        if name is None:
            continue
        rows = merged(name, generic[name])
        _write(ms_dir, name, len(rows["TIME"]), rows)
        written.append(name)

    logger.info("idi_subtables: {} subtables of {} written from {} file(s) in {:.2f}s", len(written), ms_dir.name,
                len(files), _time.perf_counter() - started)
    return written
