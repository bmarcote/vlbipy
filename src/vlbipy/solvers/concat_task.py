"""Concatenate measurement sets without casacore: what ``casatasks.concat(respectname=True, copypointing=False)``
writes for the per-source splits of several epochs, plus the per-epoch amplitude scaling of the campaign merge.

:func:`concat_ms` repeats the task (``task_concat.py``) and casacore's ``MSConcat::concatenate`` on plain
measurement sets, rule by rule as read off ``MSConcat.cc`` and the ``MS*Columns::match*`` functions:

* The inputs are processed in the order of their earliest main-table TIME (stable). The first is copied; each
  following one is appended to the result. The main table is NOT sorted: the rows of each input follow the
  previous ones in their own order.
* ANTENNA: an antenna matches the last unflagged row with the same NAME and STATION whose POSITION is within
  1 m on every axis (same reference frame); otherwise its row is appended (an antenna that moved gets a new
  ID). FEED: the feeds of a matched antenna are compared with casacore's ``matchFeed`` *including its quirks*:
  the candidate rows are searched with the antenna index of the appended MS (not the matched one) and a feed
  counts as present when its row number in the appended MS equals a matched row number of the result; feeds
  without a match are appended. A matched feed's TIME/INTERVAL grow to cover both validity ranges.
* SPECTRAL_WINDOW / POLARIZATION / DATA_DESCRIPTION: a window matches an unflagged row with the same NUM_CHAN
  and IF_CONV_CHAIN, TOTAL_BANDWIDTH within a quarter of it, the same frequency frame, REF_FREQUENCY and every
  CHAN_FREQ within 1 Hz; otherwise it is appended. A DATA_DESCRIPTION row is appended for every new
  (window, polarization) pair. Windows not used by any DATA_DESCRIPTION are matched / appended afterwards.
* FIELD (dirtol 1 mas, ``respectname``): a field matches a row whose REFERENCE/DELAY/PHASE_DIR are all within
  1 mas (great-circle separation as ``MVDirection::separation``), trying first the last row with the same
  name, then all rows from the last; with ``respectname`` the names must also agree. Otherwise appended.
* OBSERVATION: the rows are appended, then a later row equal to an earlier one in FLAG_ROW, OBSERVER, PROJECT,
  RELEASE_DATE, TELESCOPE_NAME and TIME_RANGE is removed and its ID mapped to the earlier one. SCAN_NUMBER of
  appended rows whose OBSERVATION_ID changed is offset by ``max(existing scans) + 1 - min(appended scans)``
  (casacore's ``scanOffsetForOid`` rules), so the scans of the epochs stay distinct.
* Appended main rows: ANTENNA1/2, DATA_DESC_ID, FIELD_ID, OBSERVATION_ID remapped; a baseline whose new
  antenna IDs come in decreasing order is swapped: UVW negated, visibilities conjugated and the correlations
  permuted (RL <-> LR) in DATA, CORRECTED_DATA, FLAG, WEIGHT(_SPECTRUM), SIGMA(_SPECTRUM), FLAG_CATEGORY,
  FEED1/FEED2 exchanged. WEIGHT_SPECTRUM / SIGMA_SPECTRUM / FLAG_CATEGORY are copied when row 0 of both MSs
  defines them (otherwise the cells stay undefined).
* SYSCAL, WEATHER, GAIN_CURVE, PHASE_CAL, EARTH_ORIENTATION: rows appended only when the result already has
  rows, with ANTENNA_ID (and SPECTRAL_WINDOW_ID, OBSERVATION_ID) remapped. casacore's quirks are kept: a
  window ID is remapped only when a new window was added, and then a table holding an ID absent from the map
  (a window that matched its own index) is emptied completely, as are tables with antenna IDs out of range.
* POINTING is emptied (``copypointing=False``); FLAG_CMD, HISTORY and any other subtable keep the rows of the
  first MS. HISTORY gets the parameter rows of ``write_history`` (taskname, version, one row per parameter);
  their TIME is the time of the run, and the version row names vlbipy instead of CASA.

The epoch scaling of ``CasaExportOps._scale_epochs`` is done while the rows are written: for each input in the
order given (not the sorted order), with factor ``s != 1``, the rows of the result whose TIME lies within that
input's [min - 1 s, max + 1 s] get DATA / CORRECTED_DATA x s, WEIGHT and WEIGHT_SPECTRUM x s**-2, SIGMA x s
(SIGMA_SPECTRUM is left alone, as there), in float32 as CASA multiplies them.

Not reproduced (:class:`NotImplementedError`, the caller runs CASA): Multi-MS inputs (CASA first converts them
with ``split(keepmms=False)``, whose mstransform re-sorts the rows and re-derives part of the WEIGHTs), inputs
with different main-table columns or differently defined WEIGHT_SPECTRUM/SIGMA_SPECTRUM/FLAG_CATEGORY,
MODEL_DATA / FLOAT_DATA / LAG_DATA, non-empty SOURCE, STATE or PROCESSOR tables, polarization setups that do
not match, spectral windows with different channel counts, reversed channel order, direction polynomials,
ephemerides, solar-system or variable direction frames, differing measure frames or units, POINTING rows with
``copypointing=True``, ambiguous OBSERVATION matches. The main table is written in the main process (one
table, one writer); bulk columns are copied in blocks of :data:`CONCAT_ROWS` rows.
"""
from __future__ import annotations

import copy
import shutil
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..casacore_format.aipsio import SCALAR_DTYPES, TP_STRING, TableRef
from ..casacore_format.table import Table
from ..casacore_format.tsm import TILED_TYPES
from ..casacore_format.writer import create_table, read_table_values
from ..logging_utils import get_logger

logger = get_logger()

#: casacore's default tolerances in concat: freqtol 1 Hz, dirtol 1 mas, antenna position 1 m.
FREQTOL_HZ = 1.0
DIRTOL_RAD = np.deg2rad(1e-3 / 3600.0)
ANTTOL_M = 1.0
#: Main-table rows copied per block.
CONCAT_ROWS = 50_000
#: Main-table columns kept in memory (remapped per appended row).
_SCALARS = ("ANTENNA1", "ANTENNA2", "ARRAY_ID", "DATA_DESC_ID", "EXPOSURE", "FEED1", "FEED2", "FIELD_ID",
            "FLAG_ROW", "INTERVAL", "OBSERVATION_ID", "PROCESSOR_ID", "SCAN_NUMBER", "STATE_ID", "TIME",
            "TIME_CENTROID", "UVW")
#: Main-table columns copied block by block (correlation axis last).
_BULK = ("DATA", "CORRECTED_DATA", "FLAG", "WEIGHT", "SIGMA", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "FLAG_CATEGORY")
#: Columns copied only when row 0 of both MSs defines them.
_OPTIONAL_CELLS = ("WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "FLAG_CATEGORY")
#: Subtables whose rows are appended with remapped antenna / window IDs: (name, antenna IDs may be -1).
_ID_TABLES = (("SYSCAL", False), ("WEATHER", True), ("GAIN_CURVE", False), ("PHASE_CAL", False))
_PLANETS = {"MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO", "SUN", "MOON", "COMET"}


# -- inputs -------------------------------------------------------------------------------------------------

def _inspect(path: Path) -> dict:
    """Main-table facts of one input: columns, rows, first TIME, time range, which optional cells row 0 defines.

    Raises NotImplementedError for a Multi-MS or for main-table columns this module does not copy.
    """
    if (path / "SUBMSS").is_dir():
        raise NotImplementedError(f"concat: {path.name} is a Multi-MS (CASA converts it with split first)")
    with Table(path) as table:
        if table.parts:
            raise NotImplementedError(f"concat: {path.name} is a concatenated table")
        names = table.colnames()
        extra = [n for n in names if n not in _SCALARS and n not in _BULK]
        if extra:
            raise NotImplementedError(f"concat: {path.name} has main-table columns {extra}")
        nrow = table.nrows()
        if nrow == 0:
            raise NotImplementedError(f"concat: {path.name} has no rows")
        time = np.asarray(table.getcol("TIME"), dtype=np.float64)
        defined = {n: _cell_defined(table, n) for n in _OPTIONAL_CELLS if n in names}
    return {"path": path, "columns": names, "nrow": nrow, "t_first": float(np.sort(time)[0]),
            "window": (float(time.min()) - 1.0, float(time.max()) + 1.0), "defined": defined}


def _cell_defined(table: Table, name: str) -> bool:
    """Whether row 0 of array column ``name`` holds a cell (casacore's ``isDefined(0)``)."""
    try:
        return np.asarray(table.getcol(name, 0, 1)[0]).size > 0
    except (ValueError, IndexError, TypeError):
        return False


def _load_sub(ms: Path, name: str) -> Optional[dict]:
    """A subtable as ``{columns, values (name -> list of cells), nrow, keywords, info, source, changed}``."""
    path = ms / name
    if not (path / "table.dat").is_file():
        return None
    columns, values, nrow, keywords, info = read_table_values(path)
    cells = {}
    for column in columns:
        value = values[column.name]
        cells[column.name] = [] if value is None or nrow == 0 else list(value)
    return {"columns": columns, "values": cells, "nrow": nrow, "keywords": keywords, "info": info,
            "source": path, "changed": False}


def _names(sub: dict) -> list[str]:
    """Column names of a loaded subtable."""
    return [c.name for c in sub["columns"]]


def _column(sub: dict, name: str):
    """The ColumnDesc of ``name`` in a loaded subtable."""
    return next(c for c in sub["columns"] if c.name == name)


def _empty(sub: Optional[dict]) -> bool:
    """casacore's ``isNull() || nrow() == 0``."""
    return sub is None or sub["nrow"] == 0


def _default_cell(column):
    """The value of a cell nobody wrote: 0 / "" for scalars, zeros for fixed-shape arrays, else undefined."""
    if not column.is_array:
        return "" if column.dtype == TP_STRING else SCALAR_DTYPES[column.dtype](0)
    if column.shape:
        shape = tuple(column.shape[::-1])
        return np.full(shape, "", dtype=object) if column.dtype == TP_STRING else np.zeros(
            shape, dtype=SCALAR_DTYPES[column.dtype])
    return None


def _append_row(sub: dict, other: dict, row: int, overrides: Optional[dict] = None) -> int:
    """``putMatchingFields``: append row ``row`` of ``other`` (columns both have), then ``overrides``."""
    other_names = set(_names(other))
    for column in sub["columns"]:
        if overrides and column.name in overrides:
            value = overrides[column.name]
        elif column.name in other_names:
            value = copy.deepcopy(other["values"][column.name][row])
            if value is None and column.is_array:
                # TableRow::put writes an undefined array cell as an empty (defined) array.
                value = np.zeros(0, dtype=object if column.dtype == TP_STRING else SCALAR_DTYPES[column.dtype])
        else:
            value = _default_cell(column)
        sub["values"][column.name].append(value)
    sub["nrow"] += 1
    sub["changed"] = True
    return sub["nrow"] - 1


def _remove_rows(sub: dict, rows) -> None:
    """Remove the given rows of a loaded subtable."""
    drop = set(int(r) for r in rows)
    if not drop:
        return
    keep = [r for r in range(sub["nrow"]) if r not in drop]
    for name in sub["values"]:
        sub["values"][name] = [sub["values"][name][r] for r in keep]
    sub["nrow"] = len(keep)
    sub["changed"] = True


def _units(sub: dict, name: str) -> tuple:
    """QuantumUnits of a column (empty tuple without)."""
    units = _column(sub, name).keywords.get("QuantumUnits")
    return tuple(str(u) for u in np.atleast_1d(units)) if units is not None else ()


def _measinfo(sub: dict, name: str) -> dict:
    """MEASINFO keywords of a column ({} without)."""
    return dict(_column(sub, name).keywords.get("MEASINFO") or {})


def _require_same_frames(this: dict, other: dict, table: str, columns: tuple) -> None:
    """Units and fixed measure frames of ``columns`` must agree between the two tables (casacore converts)."""
    for name in columns:
        if name not in _names(this) or name not in _names(other):
            continue
        if _units(this, name) != _units(other, name):
            raise NotImplementedError(f"concat: {table}.{name} units differ")
        a, b = _measinfo(this, name), _measinfo(other, name)
        if "VarRefCol" in a or "VarRefCol" in b:
            continue
        if str(a.get("Ref", "")) != str(b.get("Ref", "")):
            raise NotImplementedError(f"concat: {table}.{name} frames differ ({a.get('Ref')} / {b.get('Ref')})")


# -- spectral windows, polarizations, data descriptions ------------------------------------------------------

def _freq_frame(sub: dict, row: int):
    """The frequency frame of a SPECTRAL_WINDOW row (MEAS_FREQ_REF when the column carries it)."""
    info = _measinfo(sub, "REF_FREQUENCY")
    if "VarRefCol" in info:
        return int(sub["values"][str(info["VarRefCol"])][row])
    return str(info.get("Ref", ""))


def _match_spw(this: dict, other: dict, spw: int) -> int:
    """``MSSpWindowColumns::matchSpw`` with channel frequencies: the first matching row, or -1."""
    tv, ov = this["values"], other["values"]
    nchan = int(ov["NUM_CHAN"][spw])
    bandwidth = float(ov["TOTAL_BANDWIDTH"][spw])
    freqs = np.asarray(ov["CHAN_FREQ"][spw], dtype=np.float64)
    for row in range(this["nrow"]):
        if bool(tv["FLAG_ROW"][row]) or int(tv["NUM_CHAN"][row]) != nchan:
            continue
        if int(tv["IF_CONV_CHAIN"][row]) != int(ov["IF_CONV_CHAIN"][spw]):
            continue
        if abs(float(tv["TOTAL_BANDWIDTH"][row]) - bandwidth) > abs(bandwidth / 4.0):
            continue
        if _freq_frame(this, row) != _freq_frame(other, spw):
            continue
        if abs(float(tv["REF_FREQUENCY"][row]) - float(ov["REF_FREQUENCY"][spw])) > FREQTOL_HZ:
            continue
        if nchan == 1:
            return row
        mine = np.asarray(tv["CHAN_FREQ"][row], dtype=np.float64)
        if mine.shape == freqs.shape and np.all(np.abs(mine - freqs) <= FREQTOL_HZ):
            return row
        if mine.shape == freqs.shape and np.all(np.abs(mine - freqs[::-1]) <= FREQTOL_HZ):
            raise NotImplementedError("concat: a spectral window matches with reversed channels")
    return -1


def _merge_spw_pol(subs: dict, others: dict) -> tuple[list[int], dict, bool]:
    """``copySpwAndPol``: returns (DATA_DESC_ID map, window index map, whether a window was added)."""
    spw, pol, dd = subs["SPECTRAL_WINDOW"], subs["POLARIZATION"], subs["DATA_DESCRIPTION"]
    ospw, opol, odd = others["SPECTRAL_WINDOW"], others["POLARIZATION"], others["DATA_DESCRIPTION"]
    _require_same_frames(spw, ospw, "SPECTRAL_WINDOW", ("CHAN_FREQ", "REF_FREQUENCY", "TOTAL_BANDWIDTH"))
    nchans = {int(n) for n in spw["values"]["NUM_CHAN"]} | {int(n) for n in ospw["values"]["NUM_CHAN"]}
    if len(nchans) > 1:
        raise NotImplementedError(f"concat: spectral windows with different channel counts {sorted(nchans)}")
    spw_map: dict[int, int] = {}
    added = False
    found = set()
    dd_map = []
    for d in range(odd["nrow"]):
        other_spw = int(odd["values"]["SPECTRAL_WINDOW_ID"][d])
        found.add(other_spw)
        new_spw = _match_spw(spw, ospw, other_spw)
        matched = new_spw >= 0
        if not matched:
            new_spw = _append_row(spw, ospw, other_spw)
            spw_map[other_spw] = new_spw
            added = True
        elif new_spw != other_spw:
            spw_map[other_spw] = new_spw
        corr = np.asarray(opol["values"]["CORR_TYPE"][int(odd["values"]["POLARIZATION_ID"][d])])
        candidates = [r for r in range(pol["nrow"]) if not bool(pol["values"]["FLAG_ROW"][r])
                      and np.array_equal(np.asarray(pol["values"]["CORR_TYPE"][r]), corr)]
        if not candidates:
            raise NotImplementedError("concat: the polarization setups differ")
        new_pol = max(candidates)            # the while loop over tryRow ends on the last matching row
        rows = [r for r in range(dd["nrow"]) if int(dd["values"]["SPECTRAL_WINDOW_ID"][r]) == new_spw
                and int(dd["values"]["POLARIZATION_ID"][r]) == new_pol] if matched else []
        if rows:
            dd_map.append(rows[0])
        else:
            dd_map.append(_append_row(dd, {"columns": [], "values": {}},
                                      0, {"POLARIZATION_ID": np.int32(new_pol),
                                          "SPECTRAL_WINDOW_ID": np.int32(new_spw)}))
    for other_spw in range(ospw["nrow"]):
        if other_spw not in found and _match_spw(spw, ospw, other_spw) < 0:
            spw_map[other_spw] = _append_row(spw, ospw, other_spw)
            added = True
    return dd_map, spw_map, added


# -- antennas and feeds ---------------------------------------------------------------------------------------

def _match_antenna(ant: dict, name: str, station: str, position: np.ndarray) -> int:
    """``matchAntennaAndStation`` (1 m): the last unflagged row with the same name, station and position."""
    values = ant["values"]
    for row in range(ant["nrow"] - 1, -1, -1):
        if bool(values["FLAG_ROW"][row]) or values["NAME"][row] != name or values["STATION"][row] != station:
            continue
        if np.all(np.abs(np.asarray(values["POSITION"][row], dtype=np.float64) - position) <= ANTTOL_M):
            return row
    return -1


def _match_feed(feed: dict, ofeed: dict, k: int, antenna: int, spw: int, ignore: list) -> tuple[int, Optional[tuple]]:
    """``MSFeedColumns::matchFeed`` for row ``k`` of the appended FEED: (row or -1, new (TIME, INTERVAL) or None)."""
    fv, ov = feed["values"], ofeed["values"]
    time, half = float(ov["TIME"][k]), float(ov["INTERVAL"][k]) / 2.0
    nrec = int(ov["NUM_RECEPTORS"][k])
    position = np.asarray(ov["POSITION"][k], dtype=np.float64)
    focus = "FOCUS_LENGTH" in fv and "FOCUS_LENGTH" in ov
    for row in range(feed["nrow"] - 1, -1, -1):
        if row in ignore:
            continue
        if focus and float(fv["FOCUS_LENGTH"][row]) != float(ov["FOCUS_LENGTH"][k]):
            continue
        if not (int(fv["ANTENNA_ID"][row]) == antenna and int(fv["FEED_ID"][row]) == int(ov["FEED_ID"][k])
                and int(fv["SPECTRAL_WINDOW_ID"][row]) == spw and int(fv["NUM_RECEPTORS"][row]) == nrec
                and np.array_equal(np.asarray(fv["POSITION"][row], dtype=np.float64), position)):
            continue
        same = np.array_equal(np.asarray(fv["POL_RESPONSE"][row]), np.asarray(ov["POL_RESPONSE"][k]))
        offsets = (np.asarray(fv["BEAM_OFFSET"][row]), np.asarray(ov["BEAM_OFFSET"][k]))
        angles = (np.asarray(fv["RECEPTOR_ANGLE"][row]), np.asarray(ov["RECEPTOR_ANGLE"][k]))
        types = (np.asarray(fv["POLARIZATION_TYPE"][row]), np.asarray(ov["POLARIZATION_TYPE"][k]))
        matches = True
        for i in range(nrec):
            if not (offsets[0][i, 0] == offsets[1][i, 0] and offsets[0][i, 1] == offsets[1][i, 1]
                    and types[0][i] == types[1][i] and angles[0][i] == angles[1][i] and same):
                matches = False
                break
        if not matches:
            continue
        mine_time = float(fv["TIME"][row])
        mine_half = float(fv["INTERVAL"][row]) / 2.0 or 5e17          # 0 means forever
        if mine_time - mine_half <= time - half and mine_time + mine_half >= time + half:
            return row, None
        new_time = (mine_time + time) / 2.0
        new_interval = 2.0 * (max(mine_time + mine_half, time + half) - new_time)
        return row, (new_time, new_interval)
    return -1, None


def _feed_overrides(antenna: int, other_spw: int, spw_map: dict, spw_added: bool) -> dict:
    """ANTENNA_ID (and, when a window was added, the remapped SPECTRAL_WINDOW_ID: -1 when unmapped)."""
    out = {"ANTENNA_ID": np.int32(antenna)}
    if spw_added:
        out["SPECTRAL_WINDOW_ID"] = np.int32(spw_map.get(other_spw, -1))
    return out


def _merge_antennas(subs: dict, others: dict, spw_map: dict, spw_added: bool) -> list[int]:
    """``copyAntennaAndFeed``: the new index of every antenna of the appended MS."""
    ant, oant, feed, ofeed = subs["ANTENNA"], others["ANTENNA"], subs["FEED"], others["FEED"]
    _require_same_frames(ant, oant, "ANTENNA", ("POSITION",))
    if ("FOCUS_LENGTH" in _names(feed)) != ("FOCUS_LENGTH" in _names(ofeed)):
        raise NotImplementedError("concat: only one FEED table has FOCUS_LENGTH")
    ov = oant["values"]
    other_feed_ant = np.asarray(ofeed["values"]["ANTENNA_ID"], dtype=np.int64)
    ant_map = []
    for a in range(oant["nrow"]):
        position = np.asarray(ov["POSITION"][a], dtype=np.float64)
        new = _match_antenna(ant, ov["NAME"][a], ov["STATION"][a], position)
        to_compare = np.flatnonzero(other_feed_ant == a)
        if new >= 0:
            ant_map.append(new)
            mine = np.flatnonzero(np.asarray(feed["values"]["ANTENNA_ID"], dtype=np.int64) == new)
            ignore: list[int] = []
            if mine.size == to_compare.size:
                for k in to_compare:
                    other_spw = int(ofeed["values"]["SPECTRAL_WINDOW_ID"][k])
                    spw = spw_map.get(other_spw, -1) if spw_added else other_spw
                    row, validity = _match_feed(feed, ofeed, int(k), a, spw, ignore)
                    if row >= 0:
                        if validity is not None:
                            feed["values"]["TIME"][row] = np.float64(validity[0])
                            feed["values"]["INTERVAL"][row] = np.float64(validity[1])
                            feed["changed"] = True
                        ignore.append(row)
            if len(ignore) != to_compare.size:
                for k in to_compare:
                    if int(k) not in ignore:      # casacore compares appended-MS rows with result rows
                        other_spw = int(ofeed["values"]["SPECTRAL_WINDOW_ID"][k])
                        _append_row(feed, ofeed, int(k), _feed_overrides(new, other_spw, spw_map, spw_added))
            continue
        new = _append_row(ant, oant, a)
        ant_map.append(new)
        for k in to_compare:
            other_spw = int(ofeed["values"]["SPECTRAL_WINDOW_ID"][k])
            _append_row(feed, ofeed, int(k), _feed_overrides(new, other_spw, spw_map, spw_added))
        logger.info("concat: antenna {} of the appended MS has no counterpart; new ID {}", ov["NAME"][a], new)
    return ant_map


# -- fields ---------------------------------------------------------------------------------------------------

def _unit_vector(direction) -> np.ndarray:
    """Direction cosines of (longitude, latitude) in radians."""
    lon, lat = float(direction[0]), float(direction[1])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _separation(a, b) -> float:
    """``MVDirection::separation``: 2 asin(|a - b| / 2) of the unit vectors."""
    d = np.linalg.norm(_unit_vector(a) - _unit_vector(b)) / 2.0
    return 2.0 * np.arcsin(min(1.0, max(-1.0, d)))


def _field_dirs(sub: dict, row: int) -> list:
    """(REFERENCE_DIR, DELAY_DIR, PHASE_DIR) of a FIELD row without direction polynomial."""
    return [np.asarray(sub["values"][n][row], dtype=np.float64).reshape(-1, 2)[0]
            for n in ("REFERENCE_DIR", "DELAY_DIR", "PHASE_DIR")]


def _check_fields(sub: dict, table: str) -> None:
    """Raise for what copyField would treat specially: polynomials, ephemerides, planets, variable frames."""
    for name in ("REFERENCE_DIR", "DELAY_DIR", "PHASE_DIR"):
        info = _measinfo(sub, name)
        if "VarRefCol" in info or str(info.get("Ref", "")).upper() in _PLANETS:
            raise NotImplementedError(f"concat: {table} FIELD.{name} frame {info} is not reproduced")
    if any(int(n) != 0 for n in sub["values"]["NUM_POLY"]):
        raise NotImplementedError(f"concat: {table} has direction polynomials")
    if "EPHEMERIS_ID" in sub["values"] and any(int(e) > -1 for e in sub["values"]["EPHEMERIS_ID"]):
        raise NotImplementedError(f"concat: {table} has ephemeris fields")


def _merge_fields(subs: dict, others: dict, respectname: bool) -> list[int]:
    """``copyField`` (dirtol 1 mas): the new index of every field of the appended MS."""
    fld, ofld = subs["FIELD"], others["FIELD"]
    _check_fields(fld, "result")
    _check_fields(ofld, "appended MS")
    _require_same_frames(fld, ofld, "FIELD", ("REFERENCE_DIR", "DELAY_DIR", "PHASE_DIR"))
    fld_map = []
    for f in range(ofld["nrow"]):
        name = ofld["values"]["NAME"][f]
        wanted = _field_dirs(ofld, f)
        hint = max([r for r in range(fld["nrow"]) if fld["values"]["NAME"][r] == name], default=-1)
        order = ([hint] if hint >= 0 else []) + list(range(fld["nrow"] - 1, -1, -1))
        new = -1
        for row in order:
            if bool(fld["values"]["FLAG_ROW"][row]) or int(fld["values"]["NUM_POLY"][row]) != 0:
                continue
            if all(_separation(w, m) < DIRTOL_RAD for w, m in zip(wanted, _field_dirs(fld, row))):
                new = row
                break
        if new >= 0 and (not respectname or fld["values"]["NAME"][new] == name):
            fld_map.append(new)
            continue
        overrides = {"EPHEMERIS_ID": np.int32(-1)} if "EPHEMERIS_ID" in fld["values"] else None
        fld_map.append(_append_row(fld, ofld, f, overrides))
    return fld_map


# -- observations ---------------------------------------------------------------------------------------------

def _obs_equivalent(obs: dict, i: int, j: int) -> bool:
    """``obsRowsEquivalent``: FLAG_ROW, OBSERVER, PROJECT, RELEASE_DATE, TELESCOPE_NAME and TIME_RANGE equal."""
    v = obs["values"]
    return (bool(v["FLAG_ROW"][i]) == bool(v["FLAG_ROW"][j]) and v["OBSERVER"][i] == v["OBSERVER"][j]
            and v["PROJECT"][i] == v["PROJECT"][j] and float(v["RELEASE_DATE"][i]) == float(v["RELEASE_DATE"][j])
            and v["TELESCOPE_NAME"][i] == v["TELESCOPE_NAME"][j]
            and np.array_equal(np.asarray(v["TIME_RANGE"][i]), np.asarray(v["TIME_RANGE"][j])))


def _merge_observations(subs: dict, others: dict) -> tuple[dict, dict, set]:
    """``copyObservation(remRedunObsId=True)``: (ID map of the result, ID map of the appended MS, the result
    IDs whose observation was also in the appended MS)."""
    obs, oobs = subs["OBSERVATION"], others["OBSERVATION"]
    original = obs["nrow"]
    added = {k: _append_row(obs, oobs, k) for k in range(oobs["nrow"])}
    added_rows = set(added.values())
    redirect: dict[int, int] = {}
    counterpart: set[int] = set()
    removed: list[int] = []
    for j in range(obs["nrow"]):
        for k in range(j + 1, obs["nrow"]):
            if _obs_equivalent(obs, j, k):
                if k in removed:
                    raise NotImplementedError("concat: an OBSERVATION row matches several others")
                redirect[k] = j
                if k in added_rows:
                    counterpart.add(j)
                removed.append(k)
    map_a = {i: redirect[i] for i in range(original) if i in redirect}
    map_b = {k: redirect.get(row, row) for k, row in added.items()}
    _remove_rows(obs, removed)
    logger.info("concat: OBSERVATION {} rows added, {} matched", obs["nrow"] - original, len(removed))
    return map_a, map_b, counterpart


# -- calibration subtables ----------------------------------------------------------------------------------

def _merge_id_table(subs: dict, others: dict, name: str, allow_minus_one: bool, ant_map: list, spw_map: dict,
                    spw_added: bool) -> None:
    """copySysCal / copyWeather / copyGainCurve / copyPhaseCal: append with remapped antenna and window IDs."""
    this, other = subs.get(name), others.get(name)
    if _empty(this) or _empty(other):
        if _empty(this) and not _empty(other):
            logger.warning("concat: the result has no {} rows; those of the appended MS are dropped", name)
        return
    start = this["nrow"]
    for k in range(other["nrow"]):
        _append_row(this, other, k)
    ants = [int(a) for a in this["values"]["ANTENNA_ID"][start:]]
    low = -1 if allow_minus_one else 0
    if any(a < low or a > len(ant_map) - 1 for a in ants):
        logger.warning("concat: invalid antenna IDs in {}: the table is emptied (as casacore does)", name)
        _remove_rows(this, range(this["nrow"]))
        return
    this["values"]["ANTENNA_ID"][start:] = [np.int32(ant_map[a]) if a > -1 else np.int32(a) for a in ants]
    if spw_added and name != "WEATHER":
        spws = [int(s) for s in this["values"]["SPECTRAL_WINDOW_ID"][start:]]
        if any(s not in spw_map for s in spws):
            logger.warning("concat: unmapped spectral window IDs in {}: the table is emptied (as casacore does)", name)
            _remove_rows(this, range(this["nrow"]))
            return
        this["values"]["SPECTRAL_WINDOW_ID"][start:] = [np.int32(spw_map[s]) for s in spws]


def _merge_eop(subs: dict, others: dict, map_a: dict, map_b: dict) -> None:
    """copyEOP: append, OBSERVATION_ID remapped (result rows by the result map, appended rows by theirs)."""
    this, other = subs.get("EARTH_ORIENTATION"), others.get("EARTH_ORIENTATION")
    if _empty(this) or _empty(other):
        return
    start = this["nrow"]
    for k in range(other["nrow"]):
        _append_row(this, other, k)
    ids = this["values"]["OBSERVATION_ID"]
    for r in range(this["nrow"]):
        mapping = map_a if r < start else map_b
        if int(ids[r]) in mapping:
            ids[r] = np.int32(mapping[int(ids[r])])
    this["changed"] = True


# -- main table -----------------------------------------------------------------------------------------------

def _pol_swap(others: dict) -> np.ndarray:
    """The correlation permutation of a swapped baseline (RL <-> LR) from CORR_PRODUCT; one for every DDID."""
    pol, dd = others["POLARIZATION"], others["DATA_DESCRIPTION"]
    swaps = []
    for pid in sorted({int(p) for p in dd["values"]["POLARIZATION_ID"]}):
        products = np.asarray(pol["values"]["CORR_PRODUCT"][pid]).reshape(-1, 2)
        swap = []
        for i in range(len(products)):
            j = [j for j in range(len(products))
                 if products[i, 0] == products[j, 1] and products[i, 1] == products[j, 0]]
            if not j:
                raise NotImplementedError("concat: a correlation has no counterpart for swapped antennas")
            swap.append(j[0])
        swaps.append(tuple(swap))
    if len(set(swaps)) != 1:
        raise NotImplementedError("concat: polarization setups with different correlation products")
    return np.array(swaps[0], dtype=np.int64)


def _scan_offsets(main: dict, curr: int, other_scans: np.ndarray, counterpart: set) -> tuple[dict, int]:
    """casacore's scan offsets: ({result OBSERVATION_ID: offset}, default offset for new IDs)."""
    obs_ids = main["OBSERVATION_ID"][:curr]
    scans = main["SCAN_NUMBER"][:curr]
    distinct, mins, maxs = [], [], []
    for oid in obs_ids[np.sort(np.unique(obs_ids, return_index=True)[1])]:
        sel = scans[obs_ids == oid]
        distinct.append(int(oid))
        mins.append(int(sel.min()))
        maxs.append(int(sel.max()))
    max_this = max(0, int(scans.max()))
    min_other = int(other_scans.min())
    default = max(0, max_this + 1 - min_other)
    offsets = {}
    for i, oid in enumerate(distinct):
        offset = maxs[i - 1] + 1 - min_other if (oid in counterpart and i != 0) else mins[i] - 1
        offsets[oid] = max(0, offset)
    return offsets, default


def _append_main(main: dict, others_main: dict, maps: dict) -> np.ndarray:
    """Append the scalar columns of one MS to ``main`` (in place); returns the swapped-baseline mask."""
    ant_map = np.asarray(maps["antenna"], dtype=np.int32)
    a1 = ant_map[others_main["ANTENNA1"]]
    a2 = ant_map[others_main["ANTENNA2"]]
    swap = a1 > a2
    curr = len(main["TIME"])
    offsets, default = _scan_offsets(main, curr, others_main["SCAN_NUMBER"], maps["counterpart"])
    if maps["obs_a"]:
        main["OBSERVATION_ID"] = np.array([maps["obs_a"].get(int(o), int(o)) for o in main["OBSERVATION_ID"]],
                                          dtype=np.int32)
    obs_in = others_main["OBSERVATION_ID"]
    obs_out = np.array([maps["obs_b"].get(int(o), int(o)) for o in obs_in], dtype=np.int32)
    scan = others_main["SCAN_NUMBER"].copy()
    for oid in np.unique(obs_out[obs_out != obs_in]):
        offsets.setdefault(int(oid), default)
        changed = (obs_out == oid) & (obs_out != obs_in)
        scan[changed] += offsets[int(oid)]
        if offsets[int(oid)]:
            logger.info("concat: scan numbers of OBSERVATION_ID {} offset by {}", int(oid), offsets[int(oid)])
    new = dict(others_main)
    new["ANTENNA1"] = np.where(swap, a2, a1).astype(np.int32)
    new["ANTENNA2"] = np.where(swap, a1, a2).astype(np.int32)
    new["UVW"] = np.where(swap[:, None], -others_main["UVW"], others_main["UVW"])
    new["FEED1"] = np.where(swap, others_main["FEED2"], others_main["FEED1"])
    new["FEED2"] = np.where(swap, others_main["FEED1"], others_main["FEED2"])
    new["DATA_DESC_ID"] = np.asarray(maps["ddid"], dtype=np.int32)[others_main["DATA_DESC_ID"]]
    new["FIELD_ID"] = np.asarray(maps["field"], dtype=np.int32)[others_main["FIELD_ID"]]
    new["OBSERVATION_ID"] = obs_out
    new["SCAN_NUMBER"] = scan.astype(np.int32)
    for name in main:
        main[name] = np.concatenate([main[name], new[name]])
    return swap


def _read_scalars(path: Path, names: list) -> dict:
    """The in-memory main columns of one input."""
    with Table(path) as table:
        return {n: np.asarray(table.getcol(n)) for n in _SCALARS if n in names}


def _bulk_block(table: Table, name: str, start: int, count: int, swap: Optional[np.ndarray],
                polswap: Optional[np.ndarray], times: np.ndarray, windows: list) -> np.ndarray:
    """Rows [start, start + count) of a bulk column of one input, swapped and scaled as written to the result."""
    values = np.array(table.getcol(name, start, count))
    if swap is not None and swap[start:start + count].any():
        rows = np.flatnonzero(swap[start:start + count])
        moved = values[rows][..., polswap]
        values[rows] = np.conj(moved) if name in ("DATA", "CORRECTED_DATA") else moved
    powers = {"DATA": 1.0, "CORRECTED_DATA": 1.0, "WEIGHT": -2.0, "SIGMA": 1.0, "WEIGHT_SPECTRUM": -2.0}
    if name in powers:
        for (low, high), scale in windows:
            if scale == 1.0:
                continue
            rows = np.flatnonzero((times >= low) & (times <= high))
            if rows.size:
                # casatools reads float/complex cells as double precision: the product is rounded once to float32.
                factor = scale if powers[name] == 1.0 else scale ** powers[name]
                wide = np.complex128 if values.dtype.kind == "c" else np.float64
                values[rows] = (values[rows].astype(wide) * factor).astype(values.dtype)
    return values


def _history_rows(hist: dict, vislist: list, concatvis: str, respectname: bool, copypointing: bool,
                  weights: list) -> None:
    """The rows ``write_history`` adds after concat (TIME is now, the version row names vlbipy)."""
    from .. import __version__
    if hist is None:
        return
    now = (_time.time() / 86400.0 + 40587.0) * 86400.0
    lines = ["taskname=concat", f"version: vlbipy {__version__} (native concat)",
             f"{'vislist':<11} = {vislist!r}", f'{"concatvis":<11} = "{concatvis}"', f'{"freqtol":<11} = ""',
             f'{"dirtol":<11} = ""', f"{'respectname':<11} = {respectname}", f"{'timesort':<11} = False",
             f"{'copypointing':<11} = {copypointing}", f"{'visweightscale':<11} = {weights!r}",
             f'{"forcesingleephemfield":<11} = ""']
    empty = {"columns": [], "values": {}}
    for i, line in enumerate(lines):
        _append_row(hist, empty, 0, {"TIME": np.float64(now + 1e-6 * i), "OBSERVATION_ID": np.int32(-1),
                                     "MESSAGE": line, "PRIORITY": "INFO", "ORIGIN": "concat", "OBJECT_ID": np.int32(0),
                                     "APPLICATION": "ms", "CLI_COMMAND": np.array([""], dtype=object),
                                     "APP_PARAMS": np.array([""], dtype=object)})


# -- writing --------------------------------------------------------------------------------------------------

def _column_values(column, cells: list, nrow: int):
    """Cells of a loaded subtable column in the form create_table takes."""
    if nrow == 0:
        return None
    if not column.is_array:
        if column.dtype == TP_STRING:
            return [str(c) for c in cells]
        return np.asarray(cells, dtype=SCALAR_DTYPES[column.dtype])
    return list(cells)


def _write_sub(target: Path, sub: dict) -> None:
    """Write a loaded (changed) subtable at ``target`` and copy its own subtables from the source."""
    columns = copy.deepcopy(sub["columns"])
    for column in columns:
        column.shape_col_def, column.shape_col = False, ()
    values = {c.name: _column_values(c, sub["values"][c.name], sub["nrow"]) for c in columns}
    info = sub["info"]
    create_table(target, columns, sub["nrow"], values, keywords=sub["keywords"],
                 info=(info["type"], info["subType"]), readme=info["readme"])
    for child in Path(sub["source"]).iterdir():
        if child.is_dir() and (child / "table.dat").is_file():
            shutil.copytree(child, target / child.name)


def _write_main(target: Path, first: Path, pieces: list, main: dict, windows: list, defined: dict) -> None:
    """Create the main table of the result at ``target`` and copy the bulk columns block by block.

    ``defined`` tells which of WEIGHT_SPECTRUM / SIGMA_SPECTRUM / FLAG_CATEGORY row 0 of every input defines:
    the others are left undefined, as MSConcat does not copy them.
    """
    with Table(first) as template:
        columns = [copy.deepcopy(c) for c in template.dat.columns]
        keywords = dict(template.dat.keywords)
        info = template.info()
    subtables = [k for k, v in keywords.items() if isinstance(v, TableRef)]
    keywords.update({name: TableRef(f"././{name}") for name in subtables})
    nrow = len(main["TIME"])
    tiled, bulk_names = {}, []
    for column in columns:
        column.shape_col_def, column.shape_col = False, ()
        if column.name in _BULK and defined.get(column.name, True):
            bulk_names.append(column.name)
        if column.is_array and column.dm_type in TILED_TYPES:
            if not defined.get(column.name, True):
                tiled[column.name] = [None] * nrow
                continue
            shapes = []
            for piece in pieces:
                with Table(piece["path"]) as table:
                    shapes += [None if s is None else tuple(s) for s in table.cell_shapes(column.name)]
            tiled[column.name] = shapes
    values = {name: main[name] for name in main}
    small_bulk = [n for n in bulk_names if n not in tiled]
    for name in small_bulk:
        blocks = []
        for piece in pieces:
            with Table(piece["path"]) as table:
                blocks.append(_bulk_block(table, name, 0, piece["nrow"], piece["swap"], piece["polswap"],
                                          main["TIME"][piece["start"]:piece["start"] + piece["nrow"]], windows))
        values[name] = np.concatenate(blocks)
    create_table(target, columns, nrow, values, keywords=keywords, info=(info["type"], info["subType"]),
                 readme=info["readme"], tiled=tiled)
    with Table(target, readonly=False) as out:
        for name in [n for n in bulk_names if n in tiled]:
            for piece in pieces:
                with Table(piece["path"]) as table:
                    for start in range(0, piece["nrow"], CONCAT_ROWS):
                        count = min(CONCAT_ROWS, piece["nrow"] - start)
                        first_row = piece["start"] + start
                        block = _bulk_block(table, name, start, count, piece["swap"], piece["polswap"],
                                            main["TIME"][first_row:first_row + count], windows)
                        out.putcol(name, block, first_row)


# -- entry point ----------------------------------------------------------------------------------------------

def concat_ms(inputs, output, *, respectname: bool = True, copypointing: bool = False,
              scales: Optional[list] = None) -> dict:
    """Concatenate the plain measurement sets ``inputs`` into the new ``output`` (see the module docstring).

    Parameters
    ----------
    inputs : list of path
        Plain measurement sets (any order: they are processed by their earliest TIME, as concat does).
    output : path
        The result (replaced when it exists).
    respectname : bool
        Fields with different names are never merged (concat's ``respectname``).
    copypointing : bool
        concat's ``copypointing``; True is only reproduced when no input has POINTING rows.
    scales : list of float, optional
        One amplitude factor per input (in the order of ``inputs``): its rows get DATA x s, WEIGHT x s**-2,
        SIGMA x s, as ``CasaExportOps._scale_epochs`` does after concat.

    Returns
    -------
    dict
        ``rows`` (total), ``rows_per_input``, ``order`` (input paths as processed), ``antennas``, ``fields``,
        ``spectral_windows``, ``observations``, ``scaled`` and ``seconds``.

    Raises
    ------
    NotImplementedError
        For inputs or options this module does not reproduce (the caller then runs CASA).
    """
    started = _time.perf_counter()
    paths = [Path(p) for p in inputs]
    if not paths:
        raise ValueError("concat: no input measurement sets")
    if scales is not None and len(scales) != len(paths):
        raise ValueError("concat: scales must have one factor per input")
    facts = [_inspect(p) for p in paths]
    for fact in facts[1:]:
        if fact["columns"] != facts[0]["columns"]:
            raise NotImplementedError(f"concat: {fact['path'].name} has other main-table columns")
        if fact["defined"] != facts[0]["defined"]:
            raise NotImplementedError(f"concat: {fact['path'].name} defines WEIGHT/SIGMA_SPECTRUM differently")
    ordered = sorted(facts, key=lambda f: f["t_first"])
    windows = []
    if scales is not None and any(abs(float(s) - 1.0) > 1e-6 for s in scales):
        windows = [(fact["window"], float(s)) for fact, s in zip(facts, scales)]
    first = ordered[0]["path"]
    with Table(first) as table:
        sub_names = [k for k, v in table.dat.keywords.items() if isinstance(v, TableRef)]
    subs = {name: _load_sub(first, name) for name in sub_names}
    for name in ("ANTENNA", "FEED", "FIELD", "SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION", "OBSERVATION"):
        if subs.get(name) is None:
            raise NotImplementedError(f"concat: {first.name} has no {name} subtable")
    pointing = subs.get("POINTING")
    if pointing is not None and pointing["nrow"]:
        if copypointing:
            raise NotImplementedError("concat: copypointing=True with POINTING rows")
        _remove_rows(pointing, range(pointing["nrow"]))
    main = _read_scalars(first, ordered[0]["columns"])
    pieces = [{"path": first, "start": 0, "nrow": ordered[0]["nrow"], "swap": None, "polswap": None}]
    for fact in ordered[1:]:
        path = fact["path"]
        others = {name: _load_sub(path, name) for name in
                  set(sub_names) | {"ANTENNA", "FEED", "FIELD", "SPECTRAL_WINDOW", "POLARIZATION",
                                    "DATA_DESCRIPTION", "OBSERVATION", "STATE", "SOURCE", "PROCESSOR",
                                    "POINTING"} | {n for n, _ in _ID_TABLES} | {"EARTH_ORIENTATION"}}
        for name in ("STATE", "SOURCE", "PROCESSOR"):
            if not (_empty(subs.get(name)) and _empty(others.get(name))):
                raise NotImplementedError(f"concat: non-empty {name} tables are not reproduced")
        if copypointing and not _empty(others.get("POINTING")):
            raise NotImplementedError("concat: copypointing=True with POINTING rows")
        polswap = _pol_swap(others)
        ddid_map, spw_map, spw_added = _merge_spw_pol(subs, others)
        ant_map = _merge_antennas(subs, others, spw_map, spw_added)
        field_map = _merge_fields(subs, others, respectname)
        obs_a, obs_b, counterpart = _merge_observations(subs, others)
        for name, minus_one in _ID_TABLES:
            _merge_id_table(subs, others, name, minus_one, ant_map, spw_map, spw_added)
        _merge_eop(subs, others, obs_a, obs_b)
        start = len(main["TIME"])
        swap = _append_main(main, _read_scalars(path, fact["columns"]),
                            {"antenna": ant_map, "ddid": ddid_map, "field": field_map, "obs_a": obs_a,
                             "obs_b": obs_b, "counterpart": counterpart})
        pieces.append({"path": path, "start": start, "nrow": fact["nrow"], "swap": swap, "polswap": polswap})
        logger.info("concat: {} appended ({} rows, {} swapped baselines, new spw={}, antennas -> {})", path.name,
                    fact["nrow"], int(swap.sum()), spw_added, ant_map)
    weights = [np.float64(1.0)] * len(paths)
    _history_rows(subs.get("HISTORY"), [str(p) for p in inputs], str(output), respectname, copypointing, weights)
    target = Path(output)
    if target.exists():
        shutil.rmtree(target)
    _write_main(target, first, pieces, main, windows, ordered[0]["defined"])
    for name in sub_names:
        sub = subs[name]
        if sub is None:
            continue
        if sub["changed"]:
            _write_sub(target / name, sub)
        else:
            shutil.copytree(sub["source"], target / name)
    for child in first.iterdir():
        if child.is_dir() and (child / "table.dat").is_file() and not (target / child.name).exists():
            shutil.copytree(child, target / child.name)
    report = {"rows": int(len(main["TIME"])), "rows_per_input": [int(f["nrow"]) for f in ordered],
              "order": [str(f["path"]) for f in ordered], "antennas": subs["ANTENNA"]["nrow"],
              "fields": subs["FIELD"]["nrow"], "spectral_windows": subs["SPECTRAL_WINDOW"]["nrow"],
              "observations": subs["OBSERVATION"]["nrow"], "scaled": bool(windows),
              "seconds": _time.perf_counter() - started}
    logger.info("concat[dask-ms]: {} -> {} ({} rows, {} antennas, {} fields, {} spw, {} observations{}) in {:.1f}s",
                ", ".join(p.name for p in paths), target.name, report["rows"], report["antennas"], report["fields"],
                report["spectral_windows"], report["observations"],
                ", scaled " + ", ".join(f"{s:.3f}" for _, s in windows) if windows else "", report["seconds"])
    return report
