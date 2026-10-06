"""Writer/reader for CASA-compatible fringe-fit calibration tables (Type=Calibration, SubType="Fringe Jones").

The layout mirrors what CASA 6.7 ``fringefit`` writes (see /tmp/vlbipy_spec/fringe_caltable_desc.json):

- Main table: TIME, FIELD_ID, SPECTRAL_WINDOW_ID, ANTENNA1, ANTENNA2 (= refant), INTERVAL, SCAN_NUMBER,
  OBSERVATION_ID scalars; FPARAM/PARAMERR/SNR float32 (1, 8) and FLAG bool (1, 8) per row; WEIGHT float left
  undefined exactly like CASA does.  FPARAM[0, 4*p + k] = (phase [rad], delay [ns], rate [s/s], disp) for pol p.
- Subtables ANTENNA/FIELD/OBSERVATION are deep copies of the MS ones, HISTORY is empty, SPECTRAL_WINDOW has one
  channel per spw (the solution reference frequency).  Subtables are registered as table keywords with the
  casacore "Table: <absolute path>" string form.

Only the casatools ``table`` tool is used, imported lazily so this module imports without casatools.
casatools quirks handled here: ``putcol``/``getcol`` use Fortran (column-major) order, so a (nrow, 1, 8) array is
written as its transpose (8, 1, nrow) and read back with ``.T``; float columns are returned as float64 by getcol.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NPARAM = 8
MJD_UNIX_EPOCH_SECONDS = 40587.0 * 86400.0
MAIN_COLUMNS = ["TIME", "FIELD_ID", "SPECTRAL_WINDOW_ID", "ANTENNA1", "ANTENNA2", "INTERVAL", "SCAN_NUMBER",
                "OBSERVATION_ID", "FPARAM", "PARAMERR", "FLAG", "SNR", "WEIGHT"]
SPW_COPIED_SCALARS = ["REF_FREQUENCY", "MEAS_FREQ_REF", "FLAG_ROW", "FREQ_GROUP", "FREQ_GROUP_NAME",
                      "IF_CONV_CHAIN", "NAME", "NET_SIDEBAND", "TOTAL_BANDWIDTH"]
FREQ_MEASINFO = {"TabRefCodes": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 64], dtype=np.uint64),
                 "TabRefTypes": np.array(["REST", "LSRK", "LSRD", "BARY", "GEO", "TOPO", "GALACTO", "LGROUP",
                                          "CMB", "Undefined"]),
                 "VarRefCol": "MEAS_FREQ_REF", "type": "frequency"}


def _scalar_column(value_type, comment="", keywords=None, option=0):
    """Return a casacore column description for a scalar column stored in StandardStMan group MSMTAB."""
    return {"comment": comment, "dataManagerGroup": "MSMTAB", "dataManagerType": "StandardStMan",
            "keywords": dict(keywords or {}), "maxlen": 0, "option": option, "valueType": value_type}


def _array_column(value_type, comment="", keywords=None, ndim=-1, option=0, shape=None):
    """Return a casacore column description for an array column (ndim=-1 means variable shape and dimension)."""
    desc = _scalar_column(value_type, comment, keywords, option)
    desc["ndim"] = ndim
    if shape is not None:
        desc["shape"] = np.array(shape, dtype=np.int64)
    return desc


def _epoch_keywords():
    """Return the column keywords CASA uses for a UTC epoch column in seconds."""
    return {"MEASINFO": {"Ref": "UTC", "type": "epoch"}, "QuantumUnits": np.array(["s"])}


def main_table_desc():
    """Return the table description of the main fringe caltable (same columns/types/options as CASA fringefit)."""
    desc = {"TIME": _scalar_column("double", keywords=_epoch_keywords(), option=5),
            "FIELD_ID": _scalar_column("int", option=5),
            "SPECTRAL_WINDOW_ID": _scalar_column("int", option=5),
            "ANTENNA1": _scalar_column("int", option=5),
            "ANTENNA2": _scalar_column("int", option=5),
            "INTERVAL": _scalar_column("double", keywords={"QuantumUnits": np.array(["s"])}, option=5),
            "SCAN_NUMBER": _scalar_column("int", option=5),
            "OBSERVATION_ID": _scalar_column("int", option=5),
            "FPARAM": _array_column("float"),
            "PARAMERR": _array_column("float"),
            "FLAG": _array_column("boolean"),
            "SNR": _array_column("float"),
            "WEIGHT": _array_column("float")}
    return desc


def spectral_window_table_desc():
    """Return the table description of the caltable SPECTRAL_WINDOW subtable (standard MS v2 columns)."""
    hz = {"QuantumUnits": np.array(["Hz"])}
    hz_meas = {"MEASINFO": dict(FREQ_MEASINFO), "QuantumUnits": np.array(["Hz"])}
    return {"CHAN_FREQ": _array_column("double", "Center frequencies for each channel in the data matrix", hz_meas, 1),
            "CHAN_WIDTH": _array_column("double", "Channel width for each channel", hz, 1),
            "EFFECTIVE_BW": _array_column("double", "Effective noise bandwidth of each channel", hz, 1),
            "FLAG_ROW": _scalar_column("boolean", "Row flag"),
            "FREQ_GROUP": _scalar_column("int", "Frequency group"),
            "FREQ_GROUP_NAME": _scalar_column("string", "Frequency group name"),
            "IF_CONV_CHAIN": _scalar_column("int", "The IF conversion chain number"),
            "MEAS_FREQ_REF": _scalar_column("int", "Frequency Measure reference"),
            "NAME": _scalar_column("string", "Spectral window name"),
            "NET_SIDEBAND": _scalar_column("int", "Net sideband"),
            "NUM_CHAN": _scalar_column("int", "Number of spectral channels"),
            "REF_FREQUENCY": _scalar_column("double", "The reference frequency", hz_meas),
            "RESOLUTION": _array_column("double", "The effective noise bandwidth for each channel", hz, 1),
            "TOTAL_BANDWIDTH": _scalar_column("double", "The total bandwidth for this window", hz)}


def history_table_desc():
    """Return the table description of the (empty) HISTORY subtable."""
    return {"APPLICATION": _scalar_column("string", "Application name"),
            "APP_PARAMS": _array_column("string", "Application parameters", ndim=1),
            "CLI_COMMAND": _array_column("string", "CLI command sequence", ndim=1),
            "MESSAGE": _scalar_column("string", "Log message"),
            "OBJECT_ID": _scalar_column("int", "Originating ObjectID"),
            "OBSERVATION_ID": _scalar_column("int", "Observation id (index in OBSERVATION table)"),
            "ORIGIN": _scalar_column("string", "(Source code) origin from which message originated"),
            "PRIORITY": _scalar_column("string", "Message priority"),
            "TIME": _scalar_column("double", "Timestamp of message", _epoch_keywords())}


def _dminfo(columns):
    """Return a dminfo dict putting all `columns` in one StandardStMan group named MSMTAB like CASA does."""
    return {"*1": {"COLUMNS": np.array(sorted(columns)), "NAME": "MSMTAB", "SEQNR": 0, "SPEC": {},
                   "TYPE": "StandardStMan"}}


def _copy_subtable(ms_path, name, dest_dir):
    """Deep-copy MS subtable `name` into `dest_dir`/`name` and return its absolute path."""
    import casatools

    src = Path(ms_path) / name
    dest = Path(dest_dir) / name
    tb = casatools.table()
    tb.open(str(src))
    copied = tb.copy(str(dest), deep=True)
    copied.close()
    tb.close()
    return dest


def _read_ms_spw_scalars(ms_path):
    """Read the per-spw scalar columns copied verbatim from the MS SPECTRAL_WINDOW table. Returns (nspw, dict)."""
    import casatools

    tb = casatools.table()
    tb.open(str(Path(ms_path) / "SPECTRAL_WINDOW"))
    nspw = tb.nrows()
    present = set(tb.colnames())
    values = {col: tb.getcol(col) for col in SPW_COPIED_SCALARS if col in present}
    tb.close()
    return nspw, values


def _write_spectral_window(dest_dir, ms_path, spw_chan_freq, spw_chan_width):
    """Create the single-channel SPECTRAL_WINDOW subtable from the MS one; returns its absolute path."""
    import casatools

    chan_freq = np.asarray(spw_chan_freq, dtype=np.float64).reshape(-1)
    chan_width = np.asarray(spw_chan_width, dtype=np.float64).reshape(-1)
    nspw, scalars = _read_ms_spw_scalars(ms_path)
    if chan_freq.shape[0] != nspw or chan_width.shape[0] != nspw:
        raise ValueError(f"spw_chan_freq/spw_chan_width must have {nspw} entries (MS spws), got "
                         f"{chan_freq.shape[0]}/{chan_width.shape[0]}")
    dest = Path(dest_dir) / "SPECTRAL_WINDOW"
    desc = spectral_window_table_desc()
    tb = casatools.table()
    tb.create(str(dest), desc, dminfo=_dminfo(list(desc)))
    tb.addrows(nspw)
    defaults = {"REF_FREQUENCY": chan_freq, "MEAS_FREQ_REF": np.full(nspw, 5, dtype=np.int32),
                "FLAG_ROW": np.zeros(nspw, dtype=bool), "FREQ_GROUP": np.zeros(nspw, dtype=np.int32),
                "FREQ_GROUP_NAME": np.array([""] * nspw), "IF_CONV_CHAIN": np.zeros(nspw, dtype=np.int32),
                "NAME": np.array([""] * nspw), "NET_SIDEBAND": np.ones(nspw, dtype=np.int32),
                "TOTAL_BANDWIDTH": np.abs(chan_width)}
    for col in SPW_COPIED_SCALARS:
        tb.putcol(col, scalars.get(col, defaults[col]))
    tb.putcol("NUM_CHAN", np.ones(nspw, dtype=np.int32))
    # Fortran order: a (nspw, 1) per-row array column is passed as shape (1, nspw).
    tb.putcol("CHAN_FREQ", chan_freq.reshape(1, nspw))
    tb.putcol("CHAN_WIDTH", chan_width.reshape(1, nspw))
    tb.putcol("EFFECTIVE_BW", np.abs(chan_width).reshape(1, nspw))
    tb.putcol("RESOLUTION", np.abs(chan_width).reshape(1, nspw))
    tb.close()
    return dest


def _write_history_table(dest_dir):
    """Create an empty HISTORY subtable in `dest_dir`; returns its absolute path."""
    import casatools

    dest = Path(dest_dir) / "HISTORY"
    desc = history_table_desc()
    tb = casatools.table()
    tb.create(str(dest), desc, dminfo=_dminfo(list(desc)))
    tb.close()
    return dest


def _as_param_array(name, values, nrow, dtype):
    """Validate a (nrow, 8) parameter array and return it as a C-contiguous array of `dtype`."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    if arr.shape != (nrow, NPARAM):
        raise ValueError(f"{name} must have shape ({nrow}, {NPARAM}), got {arr.shape}")
    return arr


def _as_row_array(name, values, nrow, dtype):
    """Validate a per-row 1-D array and return it as `dtype`."""
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.shape[0] != nrow:
        raise ValueError(f"{name} must have {nrow} entries, got {arr.shape[0]}")
    return arr


def write_fringe_table(path, ms_path, *, times, field_ids, spw_ids, antenna_ids, refant_id, scan_numbers, fparam,
                       paramerr, flag, snr, intervals=None, spw_chan_freq, spw_chan_width, observation_ids=None,
                       casa_version="vlbipy") -> Path:
    """Write a CASA "Fringe Jones" calibration table at `path` (any existing table there is removed first).

    Inputs are one entry per solution row (nrow): `times` (MJD s), `field_ids`, `spw_ids`, `antenna_ids`,
    `scan_numbers`, optional `intervals` (default 0.0 like CASA for solint='inf') and `observation_ids` (default 0);
    `fparam`, `paramerr`, `snr` float (nrow, 8) and `flag` bool (nrow, 8).  `refant_id` (scalar or (nrow,)) fills ANTENNA2.
    `spw_chan_freq`/`spw_chan_width` (nspw,) give the single channel of each caltable spw (one row per MS spw).
    ANTENNA/FIELD/OBSERVATION are copied from `ms_path`.  WEIGHT is left undefined, as CASA does.
    Returns the absolute path of the written table.
    """
    import casatools

    path = Path(path).absolute()
    ms_path = Path(ms_path)
    nrow = int(np.asarray(times).reshape(-1).shape[0])
    columns = {"TIME": _as_row_array("times", times, nrow, np.float64),
               "FIELD_ID": _as_row_array("field_ids", field_ids, nrow, np.int32),
               "SPECTRAL_WINDOW_ID": _as_row_array("spw_ids", spw_ids, nrow, np.int32),
               "ANTENNA1": _as_row_array("antenna_ids", antenna_ids, nrow, np.int32),
               "ANTENNA2": np.broadcast_to(np.asarray(refant_id, dtype=np.int32), (nrow,)).copy(),
               "INTERVAL": (np.zeros(nrow) if intervals is None else _as_row_array("intervals", intervals, nrow,
                                                                                     np.float64)),
               "SCAN_NUMBER": _as_row_array("scan_numbers", scan_numbers, nrow, np.int32),
               "OBSERVATION_ID": (np.zeros(nrow, dtype=np.int32) if observation_ids is None else
                                  _as_row_array("observation_ids", observation_ids, nrow, np.int32))}
    params = {"FPARAM": _as_param_array("fparam", fparam, nrow, np.float32),
              "PARAMERR": _as_param_array("paramerr", paramerr, nrow, np.float32),
              "SNR": _as_param_array("snr", snr, nrow, np.float32),
              "FLAG": _as_param_array("flag", flag, nrow, bool)}

    if path.exists():
        shutil.rmtree(path)
    logger.info("Writing fringe caltable %s (%d rows, MS %s)", path, nrow, ms_path)
    desc = main_table_desc()
    tb = casatools.table()
    tb.create(str(path), desc, dminfo=_dminfo(MAIN_COLUMNS))
    tb.addrows(nrow)
    for name, values in columns.items():
        tb.putcol(name, values)
    # casatools expects Fortran order: cell shape (1, 8) over nrow rows is passed as (8, 1, nrow).
    for name, values in params.items():
        tb.putcol(name, values.reshape(nrow, 1, NPARAM).T)
    tb.putinfo({"type": "Calibration", "subType": "Fringe Jones", "readme": ""})
    tb.putkeyword("ParType", "Float")
    tb.putkeyword("MSName", ms_path.name)
    tb.putkeyword("VisCal", "Fringe Jones")
    tb.putkeyword("PolBasis", "unknown")
    tb.putkeyword("CASA_Version", str(casa_version))
    tb.close()

    subtables = {name: _copy_subtable(ms_path, name, path) for name in ("OBSERVATION", "ANTENNA", "FIELD")}
    subtables["SPECTRAL_WINDOW"] = _write_spectral_window(path, ms_path, spw_chan_freq, spw_chan_width)
    subtables["HISTORY"] = _write_history_table(path)
    tb.open(str(path), nomodify=False)
    for name, sub_path in subtables.items():
        tb.putkeyword(name, f"Table: {sub_path}")
    tb.close()
    return path


def read_fringe_table(path) -> dict:
    """Read a fringe caltable written by CASA or `write_fringe_table`.

    Returns a dict with every main column as a numpy array (FPARAM/PARAMERR/SNR float32 and FLAG bool back in
    C-order (nrow, 8); WEIGHT only if defined), plus "antenna_names" (ANTENNA subtable NAME), "spw_chan_freq"
    (nspw,) and "spw_chan_width" (nspw,) from the SPECTRAL_WINDOW subtable, "keywords" (table keywords) and
    "info" (table info dict).
    """
    import casatools

    path = Path(path).absolute()
    tb = casatools.table()
    tb.open(str(path))
    nrow = tb.nrows()
    out = {"nrow": nrow, "keywords": tb.getkeywords(), "info": tb.info()}
    for name in ("TIME", "INTERVAL"):
        out[name] = np.asarray(tb.getcol(name), dtype=np.float64)
    for name in ("FIELD_ID", "SPECTRAL_WINDOW_ID", "ANTENNA1", "ANTENNA2", "SCAN_NUMBER", "OBSERVATION_ID"):
        out[name] = np.asarray(tb.getcol(name), dtype=np.int32)
    for name, dtype in (("FPARAM", np.float32), ("PARAMERR", np.float32), ("SNR", np.float32), ("FLAG", bool)):
        if nrow == 0:
            out[name] = np.zeros((0, NPARAM), dtype=dtype)
            continue
        out[name] = np.ascontiguousarray(np.asarray(tb.getcol(name)).T.reshape(nrow, -1), dtype=dtype)
    if nrow > 0 and tb.iscelldefined("WEIGHT", 0):
        out["WEIGHT"] = np.ascontiguousarray(np.asarray(tb.getcol("WEIGHT")).T.reshape(nrow, -1), dtype=np.float32)
    tb.close()

    tb.open(str(path / "ANTENNA"))
    out["antenna_names"] = np.asarray(tb.getcol("NAME"), dtype=str)
    tb.close()
    tb.open(str(path / "SPECTRAL_WINDOW"))
    nspw = tb.nrows()
    out["spw_chan_freq"] = np.array([np.asarray(tb.getcell("CHAN_FREQ", i)).reshape(-1)[0] for i in range(nspw)])
    out["spw_chan_width"] = np.array([np.asarray(tb.getcell("CHAN_WIDTH", i)).reshape(-1)[0] for i in range(nspw)])
    tb.close()
    return out


def write_history(path, message, origin="vlbipy", application="vlbipy", priority="INFO"):
    """Append one row to the HISTORY subtable of the caltable at `path` with TIME = now (MJD seconds, UTC).

    APP_PARAMS and CLI_COMMAND (variable-length string arrays) are written as empty string arrays.
    """
    import casatools

    hist = Path(path).absolute() / "HISTORY"
    tb = casatools.table()
    tb.open(str(hist), nomodify=False)
    row = tb.nrows()
    tb.addrows(1)
    tb.putcell("APPLICATION", row, str(application))
    tb.putcell("MESSAGE", row, str(message))
    tb.putcell("ORIGIN", row, str(origin))
    tb.putcell("PRIORITY", row, str(priority))
    tb.putcell("TIME", row, MJD_UNIX_EPOCH_SECONDS + time.time())
    tb.putcell("OBSERVATION_ID", row, 0)
    tb.putcell("OBJECT_ID", row, 0)
    tb.putcell("APP_PARAMS", row, np.array([], dtype=str))
    tb.putcell("CLI_COMMAND", row, np.array([], dtype=str))
    tb.close()
    logger.info("Appended HISTORY row to %s: %s", hist, message)
