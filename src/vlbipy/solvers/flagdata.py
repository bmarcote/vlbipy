"""Native flagging for the dask-ms backend: ``flagdata``'s selection modes without a pass over the visibilities.

Almost every flag the pipeline writes is a pure *selection* — an antenna or a
baseline during a time range (the a-priori flag file, the off-source and outlier
commands), the start of every scan (quack), some channels of some subbands (the
band edges, the bandpass gaps), the autocorrelations. ``casatasks.flagdata``
walks the whole measurement set for each of them. Here the small metadata
columns of each table part are read once, every command becomes a row index
set (``searchsorted`` on merged time windows, not one pass per command), and
FLAG is read and written only for the row blocks that change. Each part of a
Multi-MS is handled by its own worker process.

The selection semantics are CASA's, checked one by one against
``flagdata`` on real data:

* ``antenna='X'`` selects the cross-correlations of X (not its autocorrelation);
  ``'X&Y'`` one baseline; ``,`` and ``;`` separate alternatives (a union);
* ``timerange='t0~t1'`` keeps the rows whose time centroid is inside
  ``[t0, t1]``, both ends included, at the millisecond precision of the string;
* ``spw='s:a~b;c~d'`` / ``correlation='LL'`` flag only those cells. FLAG_ROW is
  written only by commands with neither (set when flagging, cleared when
  unflagging); any spw or correlation selection, even a whole subband, leaves it
  as it was;
* ``mode='quack'`` (``quackmode='beg'``) flags the selected rows with
  ``time <= start + quackinterval``, where *start* is the first time of the
  *selected* rows in that scan (and field and subband);
* an empty antenna selection includes the autocorrelations; ``autocorr=True``
  selects only them.

Anything outside that vocabulary raises :class:`Unsupported`, and the caller
hands the command to CASA unchanged.

The flag statistics (:func:`count_flags`) read FLAG only. "Observable" data —
cross-correlations that are not exactly zero — needs DATA once; the rows that
are zero are cached next to each table part (``.vlbipy_observable.npz``), since
DATA does not change after import.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import time as _time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .msio import ms_parts, open_table, read_setup, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows read and written per FLAG block inside a worker.
FLAG_ROWS = 200_000
#: Name of the per-part cache of the rows whose DATA is exactly zero.
OBSERVABLE_CACHE = ".vlbipy_observable.npz"
#: Keys a command may carry that do not change what is flagged.
_IGNORED_KEYS = {"vis", "flagbackup", "reason", "savepars", "cmdreason", "datacolumn", "display", "action",
                 "quackmode", "quackincrement", "inpfile", "mode", "outfile", "tbuff"}
#: Selection keys this module implements.
_SELECTION_KEYS = {"antenna", "timerange", "spw", "correlation", "field", "scan", "autocorr", "quackinterval"}
#: CASA Stokes enumeration -> correlation label (circular and linear products).
_CORR_LABELS = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
_EPOCH = _dt.datetime(1858, 11, 17)
_TOKEN = re.compile(r"""(\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|(\S+))""")


class Unsupported(ValueError):
    """A command uses a selection or mode the native engine does not implement (CASA handles it instead)."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_command(text: str) -> dict:
    """Split one ``flagdata`` list-mode line (``key='value' key=value ...``) into a dict of strings."""
    text = str(text).strip()
    params = {}
    position = 0
    for match in _TOKEN.finditer(text):
        if text[position:match.start()].strip():
            raise Unsupported(f"cannot parse flag command {text!r}")
        key = match.group(1)
        value = next(group for group in match.groups()[1:] if group is not None)
        params[key] = value
        position = match.end()
    if text[position:].strip():
        raise Unsupported(f"cannot parse flag command {text!r}")
    return params


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "t", "yes")


def _parse_time(text: str, *, date: Optional[str] = None) -> tuple[float, str]:
    """``YYYY/MM/DD/hh:mm:ss[.f]`` (or ``hh:mm:ss[.f]`` on ``date``) -> (MJD seconds, its date part)."""
    text = text.strip()
    parts = text.split("/")
    if len(parts) == 4:
        date, clock = "/".join(parts[:3]), parts[3]
    elif len(parts) == 1 and date is not None:
        clock = parts[0]
    else:
        raise Unsupported(f"time {text!r} is not of the form YYYY/MM/DD/hh:mm:ss")
    try:
        year, month, day = (int(p) for p in date.split("/"))
        fields = clock.split(":")
        if len(fields) != 3:
            raise ValueError
        hours, minutes, seconds = int(fields[0]), int(fields[1]), float(fields[2])
    except ValueError as exc:
        raise Unsupported(f"cannot parse time {text!r}") from exc
    when = _dt.datetime(year, month, day) - _EPOCH
    return when.total_seconds() + 3600.0 * hours + 60.0 * minutes + seconds, date


def parse_timerange(text: str) -> list[tuple[float, float]]:
    """``'t0~t1[,t2~t3]'`` -> [(t0, t1), ...] in MJD seconds (inclusive)."""
    ranges = []
    for piece in (p for p in str(text).split(",") if p.strip()):
        if "~" not in piece:
            raise Unsupported(f"timerange {piece!r} is not a range")
        first, last = piece.split("~", 1)
        start, date = _parse_time(first)
        end, _ = _parse_time(last, date=date)
        ranges.append((start, end))
    return ranges


def _int_ranges(text: str, *, what: str) -> list[tuple[int, int]]:
    """``'1,3~5'`` -> [(1, 1), (3, 5)] (inclusive)."""
    out = []
    for piece in (p.strip() for p in str(text).split(",") if p.strip()):
        try:
            if "~" in piece:
                low, high = (int(v) for v in piece.split("~", 1))
            else:
                low = high = int(piece)
        except ValueError as exc:
            raise Unsupported(f"{what} selection {piece!r} is not a number or a range") from exc
        out.append((low, high))
    return out


def parse_spw(text: str, nspw: int, nchan: int) -> dict[int, np.ndarray]:
    """``'*:0~0;63~63'``, ``'5'``, ``'0~3:2~5'``, ``'1,2'`` -> {spw: channel mask (nchan,)}."""
    selected: dict[int, np.ndarray] = {}
    for piece in (p.strip() for p in str(text).split(",") if p.strip()):
        spw_part, _, chan_part = piece.partition(":")
        spw_part = spw_part.strip()
        if spw_part in ("*", ""):
            spws = list(range(nspw))
        else:
            spws = [s for low, high in _int_ranges(spw_part, what="spw") for s in range(low, high + 1)]
        mask = np.zeros(nchan, dtype=bool)
        if chan_part.strip():
            for chunk in (c.strip() for c in chan_part.split(";") if c.strip()):
                for low, high in _int_ranges(chunk, what="channel"):
                    mask[max(low, 0):min(high, nchan - 1) + 1] = True
        else:
            mask[:] = True
        for spw in spws:
            if not 0 <= spw < nspw:
                raise Unsupported(f"spw {spw} does not exist")
            selected[spw] = selected[spw] | mask if spw in selected else mask.copy()
    return selected


def parse_antenna(text: str, names: list[str]) -> tuple[set[int], set[tuple[int, int]]]:
    """``'EF'``, ``'EF&JB'``, ``'EF,MC'``, ``'EF&JB;MC'`` -> (antennas whose cross-correlations, baselines)."""
    index = {name.upper(): i for i, name in enumerate(names)}

    def resolve(token: str) -> int:
        token = token.strip()
        if token.upper() in index:
            return index[token.upper()]
        raise Unsupported(f"antenna {token!r} is not a known antenna name")

    antennas: set[int] = set()
    baselines: set[tuple[int, int]] = set()
    for term in (t.strip() for t in re.split(r"[;,]", str(text)) if t.strip()):
        if term.startswith("!") or "&&" in term or any(c in term for c in "*<>=^"):
            raise Unsupported(f"antenna selection {term!r} is not supported natively")
        if "&" in term:
            left, right = term.split("&", 1)
            a, b = resolve(left), resolve(right)
            if a == b:
                raise Unsupported(f"antenna selection {term!r} selects an autocorrelation")
            baselines.add((min(a, b), max(a, b)))
        else:
            antennas.add(resolve(term))
    return antennas, baselines


def compile_command(params: dict, setup: dict, *, unflag: bool = False) -> dict:
    """Turn one command (a dict of flagdata parameters) into a picklable selection for the part workers.

    Raises
    ------
    Unsupported
        When a key, mode or value is outside what this module reproduces exactly.
    """
    mode = str(params.get("mode", "manual") or "manual").lower()
    if mode not in ("manual", "quack", "unflag"):
        raise Unsupported(f"mode {mode!r} is not implemented natively")
    action = str(params.get("action", "apply") or "apply").lower()
    if action != "apply":
        raise Unsupported(f"action {action!r} is not implemented natively")
    unknown = set(params) - _SELECTION_KEYS - _IGNORED_KEYS
    if unknown:
        raise Unsupported(f"selection key(s) {', '.join(sorted(unknown))} not implemented natively")
    if mode == "quack":
        if str(params.get("quackmode", "beg") or "beg").lower() != "beg" or _truthy(params.get("quackincrement")):
            raise Unsupported("only quackmode='beg' without quackincrement is implemented natively")
    names, field_names = setup["antenna_names"], setup["field_names"]
    nspw, nchan = setup["chan_freq"].shape
    selection: dict = {"unflag": unflag or mode == "unflag", "antennas": None, "baselines": None,
                       "autocorr": False, "times": None, "fields": None, "scans": None, "spw": None,
                       "corr": None, "quack": None}
    antenna = str(params.get("antenna", "") or "").strip()
    if antenna:
        antennas, baselines = parse_antenna(antenna, names)
        selection["antennas"], selection["baselines"] = sorted(antennas), sorted(baselines)
    if _truthy(params.get("autocorr", False)):
        if antenna:
            raise Unsupported("autocorr together with an antenna selection is not implemented natively")
        selection["autocorr"] = True
    timerange = str(params.get("timerange", "") or "").strip()
    if timerange:
        selection["times"] = parse_timerange(timerange)
    field = str(params.get("field", "") or "").strip()
    if field:
        ids = set()
        for name in (n.strip() for n in field.split(",") if n.strip()):
            if name in field_names:
                ids.add(field_names.index(name))
            else:
                raise Unsupported(f"field {name!r} is not a field name")
        selection["fields"] = sorted(ids)
    scan = str(params.get("scan", "") or "").strip()
    if scan:
        selection["scans"] = sorted({s for low, high in _int_ranges(scan, what="scan") for s in range(low, high + 1)})
    spw = str(params.get("spw", "") or "").strip()
    if spw:
        selection["spw"] = {int(k): v for k, v in parse_spw(spw, nspw, nchan).items()}
    correlation = str(params.get("correlation", "") or "").strip()
    if correlation:
        labels = setup.get("corr_labels") or []
        mask = np.zeros(len(labels), dtype=bool)
        for label in (c.strip().upper() for c in correlation.split(",") if c.strip()):
            if label not in labels:
                raise Unsupported(f"correlation {label!r} is not one of {labels}")
            mask[labels.index(label)] = True
        selection["corr"] = mask
    if mode == "quack":
        interval = float(params.get("quackinterval", 0.0) or 0.0)
        if interval <= 0:
            raise Unsupported("quack needs a positive quackinterval")
        selection["quack"] = interval
    return selection


def compile_flagdata(kwargs: dict, setup: dict) -> list[dict]:
    """Compile a whole ``flagdata`` call (``mode='list'`` with ``inpfile`` lines, or one selection)."""
    mode = str(kwargs.get("mode", "manual") or "manual").lower()
    action = str(kwargs.get("action", "apply") or "apply").lower()
    if action != "apply":
        raise Unsupported(f"action {action!r} is not implemented natively")
    if mode != "list":
        return [compile_command(kwargs, setup)]
    unknown = set(kwargs) - _IGNORED_KEYS
    if unknown:
        raise Unsupported(f"list mode with extra selection key(s) {', '.join(sorted(unknown))}")
    source = kwargs.get("inpfile")
    if isinstance(source, (list, tuple)):
        lines = [str(line) for line in source]
    elif source and Path(str(source)).is_file():
        lines = Path(str(source)).read_text().splitlines()
    else:
        raise Unsupported(f"list mode needs an inpfile list or file (got {source!r})")
    commands = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return [compile_command(parse_command(line), setup) for line in commands]


def merge_selections(selections: list[dict]) -> list[dict]:
    """Fold selections that differ only in their time ranges into one with the ranges merged and sorted."""
    if any(selection["unflag"] for selection in selections):
        return list(selections)          # flagging and unflagging do not commute: keep the caller's order
    merged: dict[tuple, dict] = {}
    rest = []
    for selection in selections:
        if selection["times"] is None or selection["quack"] is not None:
            rest.append(selection)
            continue
        spw = selection["spw"]
        key = (selection["unflag"], tuple(selection["antennas"] or ()) if selection["antennas"] is not None else None,
               tuple(selection["baselines"] or ()) if selection["baselines"] is not None else None,
               selection["autocorr"], tuple(selection["fields"] or ()) if selection["fields"] is not None else None,
               tuple(selection["scans"] or ()) if selection["scans"] is not None else None,
               None if spw is None else tuple((k, v.tobytes()) for k, v in sorted(spw.items())),
               None if selection["corr"] is None else selection["corr"].tobytes())
        if key in merged:
            merged[key]["times"].extend(selection["times"])
        else:
            merged[key] = dict(selection, times=list(selection["times"]))
    out = list(rest)
    for selection in merged.values():
        ranges = sorted(selection["times"])
        joined = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= joined[-1][1]:
                joined[-1][1] = max(joined[-1][1], end)
            else:
                joined.append([start, end])
        selection["times"] = [tuple(r) for r in joined]
        out.append(selection)
    return out


# ---------------------------------------------------------------------------
# Row and cell selection inside one part
# ---------------------------------------------------------------------------
def _read_meta(handle) -> dict:
    return {"ant1": np.asarray(handle.getcol("ANTENNA1"), dtype=np.int64),
            "ant2": np.asarray(handle.getcol("ANTENNA2"), dtype=np.int64),
            "time": np.asarray(handle.getcol("TIME"), dtype=np.float64),
            "ddid": np.asarray(handle.getcol("DATA_DESC_ID"), dtype=np.int64),
            "field": np.asarray(handle.getcol("FIELD_ID"), dtype=np.int64),
            "scan": np.asarray(handle.getcol("SCAN_NUMBER"), dtype=np.int64)}


def _inside(times: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    """Rows whose time lies in one of the sorted, disjoint inclusive ``ranges``."""
    starts = np.array([r[0] for r in ranges])
    ends = np.array([r[1] for r in ranges])
    slot = np.searchsorted(starts, times, side="right") - 1
    ok = slot >= 0
    ok[ok] = times[ok] <= ends[slot[ok]]
    return ok


def _base_rows(selection: dict, meta: dict, ddid_to_spw: np.ndarray) -> np.ndarray:
    """Boolean row mask of a selection, before quack."""
    ant1, ant2 = meta["ant1"], meta["ant2"]
    cross = ant1 != ant2
    if selection["autocorr"]:
        rows = ~cross
    elif selection["antennas"] is None and selection["baselines"] is None:
        rows = np.ones(ant1.shape, dtype=bool)
    else:
        rows = np.zeros(ant1.shape, dtype=bool)
        if selection["antennas"]:
            wanted = np.asarray(selection["antennas"])
            rows |= cross & (np.isin(ant1, wanted) | np.isin(ant2, wanted))
        if selection["baselines"]:
            code = np.minimum(ant1, ant2) * 100_000 + np.maximum(ant1, ant2)
            rows |= np.isin(code, [a * 100_000 + b for a, b in selection["baselines"]])
    if selection["fields"] is not None:
        rows &= np.isin(meta["field"], selection["fields"])
    if selection["scans"] is not None:
        rows &= np.isin(meta["scan"], selection["scans"])
    if selection["spw"] is not None:
        allowed = [ddid for ddid, spw in enumerate(ddid_to_spw) if int(spw) in selection["spw"]]
        rows &= np.isin(meta["ddid"], allowed)
    if selection["times"] is not None:
        candidates = np.flatnonzero(rows)
        rows[candidates] = _inside(meta["time"][candidates], selection["times"])
    return rows


def _quack_keys(meta: dict, rows: np.ndarray) -> np.ndarray:
    """One integer per row identifying its (scan, field, data description) chunk."""
    return (meta["scan"][rows] * 10_000 + meta["field"][rows]) * 10_000 + meta["ddid"][rows]


def quack_starts_job(*, path: str, selections: list[dict], setup: dict, engine: str) -> list[dict]:
    """Worker job: the first selected time per (scan, field, ddid) chunk for each quack selection."""
    handle = open_table(path, engine=engine)
    try:
        meta = _read_meta(handle)
    finally:
        handle.close()
    out = []
    for selection in selections:
        rows = np.flatnonzero(_base_rows(selection, meta, setup["ddid_to_spw"]))
        keys = _quack_keys(meta, rows)
        starts: dict[int, float] = {}
        if rows.size:
            unique, inverse = np.unique(keys, return_inverse=True)
            first = np.full(unique.size, np.inf)
            np.minimum.at(first, inverse, meta["time"][rows])
            starts = dict(zip(unique.tolist(), first.tolist()))
        out.append(starts)
    return out


def _cell_mask(selection: dict, ddid_to_spw: np.ndarray, nchan: int, ncorr: int) -> Optional[np.ndarray]:
    """``(nddid, nchan, ncorr)`` cells a selection flags, or None for whole rows."""
    if selection["spw"] is None and selection["corr"] is None:
        return None
    mask = np.ones((len(ddid_to_spw), nchan, ncorr), dtype=bool)
    if selection["spw"] is not None:
        for ddid, spw in enumerate(ddid_to_spw):
            channels = selection["spw"].get(int(spw))
            mask[ddid] = channels[:, None] if channels is not None else False
    if selection["corr"] is not None:
        mask &= selection["corr"][None, None, :]
    return mask


def flag_part_job(*, path: str, selections: list[dict], setup: dict, engine: str,
                  chunk_rows: int = FLAG_ROWS) -> dict:
    """Worker job: apply ``selections`` (in order) to the FLAG/FLAG_ROW columns of one table part."""
    ddid_to_spw = setup["ddid_to_spw"]
    stats = {"rows": 0, "rows_touched": 0, "cells_changed": 0}
    handle = open_table(path, readonly=False, engine=engine)
    try:
        nrows = handle.nrows()
        stats["rows"] = nrows
        if nrows == 0:
            return stats
        meta = _read_meta(handle)
        has_flag_row = "FLAG_ROW" in handle.colnames()
        ncorr = int(setup["ncorr"])
        nchan = int(setup["chan_freq"].shape[1])
        # Each step: (unflag?, row mask, cell mask or None)
        steps = []
        for selection in selections:
            rows = _base_rows(selection, meta, ddid_to_spw)
            if selection["quack"] is not None:
                candidates = np.flatnonzero(rows)
                starts = selection.get("quack_starts", {})
                keys = _quack_keys(meta, candidates)
                first = np.array([starts.get(int(k), np.inf) for k in keys]) if keys.size else np.zeros(0)
                rows[candidates] = meta["time"][candidates] <= first + selection["quack"]
            if rows.any():
                steps.append((selection["unflag"], rows, _cell_mask(selection, ddid_to_spw, nchan, ncorr)))
        if not steps:
            return stats
        touched = np.zeros(nrows, dtype=bool)
        for _, rows, _ in steps:
            touched |= rows
        stats["rows_touched"] = int(touched.sum())
        for start in range(0, nrows, chunk_rows):
            stop = min(nrows, start + chunk_rows)
            if not touched[start:stop].any():
                continue
            flag = np.asarray(handle.getcol("FLAG", start, stop - start), dtype=bool)
            before = flag.copy()
            row_flag = row_before = None
            ddid = meta["ddid"][start:stop]
            for unflag, rows, cells in steps:
                local = np.flatnonzero(rows[start:stop])
                if not local.size:
                    continue
                if cells is None:
                    flag[local] = not unflag
                    # Like flagdata, FLAG_ROW follows whole-row selections only: a command with any spw or
                    # correlation selection leaves it alone, even when it completes or breaks a row.
                    if has_flag_row:
                        if row_flag is None:
                            row_before = np.asarray(handle.getcol("FLAG_ROW", start, stop - start), dtype=bool)
                            row_flag = row_before.copy()
                        row_flag[local] = not unflag
                elif unflag:
                    flag[local] &= ~cells[ddid[local]]
                else:
                    flag[local] |= cells[ddid[local]]
            changed = int(np.count_nonzero(flag != before))
            if changed:
                stats["cells_changed"] += changed
                handle.putcol("FLAG", flag, start)
            if row_flag is not None and not np.array_equal(row_flag, row_before):
                handle.putcol("FLAG_ROW", row_flag, start)
    finally:
        handle.close()
    return stats


def read_flag_setup(ms, *, engine: Optional[str] = None) -> dict:
    """:func:`~vlbipy.solvers.msio.read_setup` plus the correlation labels of the MS."""
    engine = engine or table_engine()
    setup = read_setup(ms, engine=engine)
    handle = open_table(Path(ms) / "POLARIZATION", engine=engine)
    try:
        corr_types = np.asarray(handle.getcol("CORR_TYPE")).reshape(-1)[:setup["ncorr"]]
    finally:
        handle.close()
    setup["corr_labels"] = [_CORR_LABELS.get(int(c), str(int(c))) for c in corr_types]
    return setup


def apply_flag_selections(ms, selections: list[dict], *, setup: Optional[dict] = None,
                          workers: Optional[int] = None) -> dict:
    """Apply compiled selections to every part of ``ms`` (one worker per part).

    Returns
    -------
    dict
        ``rows`` in the MS, ``rows_touched``, ``cells_changed``, ``parts``, ``commands`` and ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = setup or read_flag_setup(ms, engine=engine)
    parts = ms_parts(ms)
    selections = merge_selections(selections)
    quack = [s for s in selections if s["quack"] is not None]
    if quack:
        found = run_jobs("vlbipy.solvers.flagdata:quack_starts_job",
                         [{"path": p, "selections": quack, "setup": setup, "engine": engine} for p in parts], workers)
        for index, selection in enumerate(quack):
            starts: dict[int, float] = {}
            for per_part in found:
                for key, value in per_part[index].items():
                    starts[key] = min(value, starts.get(key, np.inf))
            selection["quack_starts"] = starts
    results = run_jobs("vlbipy.solvers.flagdata:flag_part_job",
                       [{"path": p, "selections": selections, "setup": setup, "engine": engine} for p in parts],
                       workers)
    total = {name: sum(r[name] for r in results) for name in ("rows", "rows_touched", "cells_changed")}
    total.update(parts=len(parts), commands=len(selections), seconds=_time.perf_counter() - started)
    return total


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _observable_cache(path: str, handle, nrows: int, chunk_rows: int) -> dict:
    """Rows of one part whose DATA is all zero, and per-cell masks of the rows that are partly zero (cached)."""
    cache = Path(path) / OBSERVABLE_CACHE
    if cache.is_file():
        try:
            with np.load(cache) as stored:
                if int(stored["nrows"]) == nrows:
                    return {name: stored[name] for name in ("zero_rows", "partial_rows", "partial_mask")}
        except (OSError, KeyError, ValueError):
            pass
    zero_rows, partial_rows, partial_mask = [], [], []
    for start in range(0, nrows, chunk_rows):
        data = handle.getcol("DATA", start, min(chunk_rows, nrows - start))
        zero = (data == 0).reshape(data.shape[0], -1)
        count = zero.sum(axis=1)
        zero_rows.append(np.flatnonzero(count == zero.shape[1]) + start)
        partial = np.flatnonzero((count > 0) & (count < zero.shape[1]))
        partial_rows.append(partial + start)
        partial_mask.append(~zero[partial].reshape((partial.size,) + data.shape[1:]))
    shape = (0,) + tuple(handle.getcol("FLAG", 0, 1).shape[1:]) if nrows else (0, 0, 0)
    result = {"zero_rows": np.concatenate(zero_rows) if zero_rows else np.zeros(0, np.int64),
              "partial_rows": np.concatenate(partial_rows) if partial_rows else np.zeros(0, np.int64),
              "partial_mask": np.concatenate(partial_mask) if partial_mask else np.zeros(shape, bool)}
    try:
        np.savez(cache, nrows=nrows, **result)
    except OSError as exc:
        logger.debug("could not cache the observable rows of {} ({})", path, exc)
    return result


def count_part_job(*, path: str, engine: str, nant: int, nddid: int, chunk_rows: int = FLAG_ROWS) -> dict:
    """Worker job: flagged and observable cross-correlation cells of one part per (antenna1, antenna2, ddid)."""
    flagged = np.zeros(nant * nant * nddid, dtype=np.int64)
    observable = np.zeros(nant * nant * nddid, dtype=np.int64)
    handle = open_table(path, engine=engine)
    try:
        nrows = handle.nrows()
        if nrows == 0:
            return {"flagged": flagged, "observable": observable}
        ant1 = np.asarray(handle.getcol("ANTENNA1"), dtype=np.int64)
        ant2 = np.asarray(handle.getcol("ANTENNA2"), dtype=np.int64)
        ddid = np.asarray(handle.getcol("DATA_DESC_ID"), dtype=np.int64)
        cache = _observable_cache(path, handle, nrows, chunk_rows)
        weight = np.ones(nrows, dtype=bool)              # rows whose every cell is observable
        weight[cache["zero_rows"]] = False
        weight[cache["partial_rows"]] = False
        weight &= ant1 != ant2
        partial_of = {int(r): i for i, r in enumerate(cache["partial_rows"])}
        key = (np.clip(ant1, 0, nant - 1) * nant + np.clip(ant2, 0, nant - 1)) * nddid + np.clip(ddid, 0, nddid - 1)
        for start in range(0, nrows, chunk_rows):
            stop = min(nrows, start + chunk_rows)
            flag = np.asarray(handle.getcol("FLAG", start, stop - start), dtype=bool)
            ncell = flag[0].size
            full = weight[start:stop]
            per_row = flag.reshape(flag.shape[0], -1).sum(axis=1)
            flagged += np.bincount(key[start:stop][full], weights=per_row[full],
                                   minlength=flagged.size).astype(np.int64)
            observable += np.bincount(key[start:stop][full], minlength=observable.size) * ncell
            partial = cache["partial_rows"]
            for row in partial[np.searchsorted(partial, start):np.searchsorted(partial, stop)]:
                if ant1[row] == ant2[row]:
                    continue
                cells = cache["partial_mask"][partial_of[int(row)]]
                flagged[key[row]] += int(np.count_nonzero(flag[row - start] & cells))
                observable[key[row]] += int(np.count_nonzero(cells))
    finally:
        handle.close()
    return {"flagged": flagged, "observable": observable}


def count_flags(ms, *, workers: Optional[int] = None) -> dict:
    """Flagged and observable cross-correlation cells per (antenna1, antenna2, data description).

    Returns
    -------
    dict
        ``flagged`` and ``observable`` arrays of shape ``(nant, nant, nddid)``, ``antenna_names`` and
        ``seconds``.
    """
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    nant, nddid = len(setup["antenna_names"]), len(setup["ddid_to_spw"])
    results = run_jobs("vlbipy.solvers.flagdata:count_part_job",
                       [{"path": p, "engine": engine, "nant": nant, "nddid": nddid} for p in ms_parts(ms)], workers)
    shape = (nant, nant, nddid)
    return {"flagged": sum(r["flagged"] for r in results).reshape(shape),
            "observable": sum(r["observable"] for r in results).reshape(shape),
            "antenna_names": setup["antenna_names"], "seconds": _time.perf_counter() - started}


# ---------------------------------------------------------------------------
# Channel-averaged amplitudes (for the slew measurement)
# ---------------------------------------------------------------------------
#: Rows per amplitude job: bounds a worker's memory (64 channels x 2 hands of complex64: ~200 MB).
AMPLITUDE_ROWS = 200_000


def row_amplitude_job(*, specs: list[dict], column: str, corrs: Optional[list[int]], ncorr: int,
                      engine: str) -> dict:
    """Worker job: the mean unflagged amplitude over channels and ``corrs`` of every cross-correlation row."""
    from .msio import read_runs
    block = read_runs(specs, [column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2"], engine=engine,
                      corrs=None if corrs is None else np.asarray(corrs), ncorr=ncorr)
    if not block:
        return {}
    values = block.pop(column)
    # In double precision and over the (correlation, channel, row) view: what casatools hands the CASA reader
    # (complex128, Fortran order), so the sums run in the same order and agree with it to the last bit.
    amplitude = np.hypot(values.real.astype(np.float64), values.imag.astype(np.float64))
    del values
    amplitude[np.asarray(block.pop("FLAG"), dtype=bool)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # rows with every sample flagged
        amplitude = np.nanmean(amplitude.T, axis=(0, 1))
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int32)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int32)
    keep = np.isfinite(amplitude) & (ant1 != ant2)
    return {"time": np.asarray(block["TIME"], dtype=np.float64)[keep], "ant1": ant1[keep], "ant2": ant2[keep],
            "ddid": np.asarray(block["DATA_DESC_ID"], dtype=np.int32)[keep],
            "amplitude": amplitude[keep].astype(np.float32)}


def read_row_amplitudes(ms, *, fields: Optional[list[str]] = None, column: str = "DATA",
                        corrs: Optional[list[int]] = None, workers: Optional[int] = None) -> dict:
    """Mean unflagged amplitude (over channels and the ``corrs`` correlations) of every cross-correlation row.

    Only the rows of ``fields`` are read, in parallel over row blocks of every table part.

    Returns
    -------
    dict
        Parallel arrays ``time``, ``ant1``, ``ant2``, ``ddid``, ``amplitude`` (float32), ordered by data
        description and then by time; rows without an unflagged sample are left out. ``seconds`` is the
        wall time.
    """
    from .msio import read_layout, run_specs, select_runs, split_rows
    started = _time.perf_counter()
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    field_ids = None
    if fields:
        field_ids = {setup["field_names"].index(name) for name in fields if name in setup["field_names"]}
    selected = select_runs(layout, fields=field_ids)
    jobs = [{"specs": chunk, "column": column, "corrs": corrs, "ncorr": int(setup["ncorr"]), "engine": engine}
            for chunk in split_rows(run_specs(layout, selected), AMPLITUDE_ROWS)]
    results = [r for r in run_jobs("vlbipy.solvers.flagdata:row_amplitude_job", jobs, workers) if r]
    names = ("time", "ant1", "ant2", "ddid", "amplitude")
    if not results:
        out = {name: np.zeros(0) for name in names}
    else:
        out = {name: np.concatenate([r[name] for r in results]) for name in names}
        order = np.lexsort((out["time"], out["ddid"]))
        out = {name: values[order] for name, values in out.items()}
    out["seconds"] = _time.perf_counter() - started
    return out


# ---------------------------------------------------------------------------
# CASA flagdata, one table part per worker
# ---------------------------------------------------------------------------
#: flagdata modes whose result on a Multi-MS is the union of their results on its parts: they judge each
#: (scan, field, subband) chunk or each row on its own, and no such chunk spans two parts.
PER_PART_MODES = {"tfcrop", "rflag", "clip", "manual", "list", "quack", "unflag"}


def casa_flagdata_job(*, kwargs: dict, logfile: str = "") -> str:
    """Worker job: run ``casatasks.flagdata(**kwargs)`` on one part; returns "" or the error text.

    The task runs in a clean subprocess: a worker that already imported python-casacore (the default
    table engine) cannot import casatools afterwards — the shared TaQL engine registers its functions
    twice (``User defined TaQL function derivedmscal.help already exists``).
    """
    import pickle
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile("wb", suffix=".pickle", delete=False) as handle:
        pickle.dump({"kwargs": kwargs, "logfile": logfile}, handle)
        spec = handle.name
    try:
        proc = subprocess.run([sys.executable, "-m", "vlbipy.solvers.flagdata", spec],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return (proc.stderr or f"exit status {proc.returncode}").strip().splitlines()[-1]
    finally:
        os.unlink(spec)
    return ""


def parallel_casa_flagdata(kwargs: dict, *, logfile: str = "", workers: Optional[int] = None) -> Optional[dict]:
    """Run a CASA flagdata call on every part of a Multi-MS at once; None when that would not be equivalent.

    Equivalent to the serial call for :data:`PER_PART_MODES` with ``action='apply'``: those modes flag every
    (scan, field, subband) chunk on its own data, and a Multi-MS never splits such a chunk between parts.
    """
    vis = kwargs.get("vis")
    mode = str(kwargs.get("mode", "manual") or "manual").lower()
    action = str(kwargs.get("action", "apply") or "apply").lower()
    if not vis or mode not in PER_PART_MODES or action != "apply":
        return None
    parts = ms_parts(vis)
    if len(parts) < 2:
        return None
    started = _time.perf_counter()
    jobs = [{"kwargs": dict(kwargs, vis=part, flagbackup=False), "logfile": logfile} for part in parts]
    results = run_jobs("vlbipy.solvers.flagdata:casa_flagdata_job", jobs, workers)
    # A part the selection does not reach has nothing to flag; on the whole MS that is an error only when no
    # part has any of the selected data.
    empty = [error for error in results if "MSSelectionNullSelection" in error]
    errors = [error for error in results if error and error not in empty]
    if len(empty) == len(parts):
        raise RuntimeError(f"flagdata selected no data: {empty[0]}")
    if errors:
        raise RuntimeError(f"flagdata failed on {len(errors)} of {len(parts)} part(s): {errors[0]}")
    return {"parts": len(parts), "seconds": _time.perf_counter() - started}


if __name__ == "__main__":
    # The one-part casatasks.flagdata runner of :func:`casa_flagdata_job` — a clean process, so
    # casatools never meets a python-casacore already loaded here.
    import pickle
    import sys

    import casatasks

    spec = pickle.load(open(sys.argv[1], "rb"))
    if spec["logfile"]:
        casatasks.casalog.setlogfile(spec["logfile"])
    try:
        casatasks.flagdata(**spec["kwargs"])
    except Exception as exc:  # noqa: BLE001 - the caller reports the text
        sys.stderr.write(f"{type(exc).__name__}: {exc}")
        sys.exit(3)
