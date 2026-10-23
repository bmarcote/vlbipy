"""What CORRECTED_DATA holds, or should hold: the calibration record of a measurement set (dask-ms backend).

The pipeline applies its table chain after every calibration pass and every self-calibration step, but
the steps in between only *read* the corrected data (flagging, plots, statwt, splits). Writing
CORRECTED_DATA each time is an unaveraged copy of the whole data set (about 60 GB of I/O for GL048,
15-25 minutes on a disk array), so an application can be *deferred* instead, the way AIPS keeps its
calibration in CL tables and applies it while reading (``DOCAL``):

* :func:`record_deferred` stores the chain of each field, with a frozen copy of its tables (later solves
  rewrite them in place), and :func:`vlbipy.solvers.applycal.defer_to_ms` writes the calibration flags
  (data without a valid solution), which only need FLAG and the tables. CORRECTED_DATA of those fields is
  then *pending*;
* a native reader of the corrected column asks :func:`view` about the fields it reads: the pending ones are
  read from DATA and corrected in memory with the code that writes CORRECTED_DATA
  (:func:`vlbipy.solvers.applycal.correct_rows`), so it sees what applycal would have written —
  visibilities, flags and weights;
* :func:`record_written` notes what an applycal wrote, and :func:`write_pending` writes the pending
  chains, which anything else reading CORRECTED_DATA needs first (CASA tasks, the ``ms`` tool, jplotter).

The record (``record.json``) and the frozen tables live in :data:`RECORD_DIR` inside the measurement set.
A field the record does not mention reads CORRECTED_DATA as it is.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger

logger = get_logger()

#: Directory inside the measurement set holding the record and the frozen tables of the deferred chains.
RECORD_DIR = ".vlbipy_corrected"
_RECORD = "record.json"
_TABLES = "tables"
#: Frozen tables already read in this process (they never change: a new version gets a new directory).
_FROZEN_CACHE: dict = {}


def _record_path(ms) -> Path:
    return Path(ms) / RECORD_DIR / _RECORD


def load_record(ms) -> dict:
    """The record of ``ms``: ``fields`` (``{field id: chain}``, what CORRECTED_DATA should hold) and
    ``written`` (``{field id: digest}``, what it holds). Empty when there is none."""
    path = _record_path(ms)
    if not path.is_file():
        return {"fields": {}, "written": {}}
    record = json.loads(path.read_text())
    return {"fields": {int(k): v for k, v in record.get("fields", {}).items()},
            "written": {int(k): v for k, v in record.get("written", {}).items()}}


def _save(ms, record: dict) -> None:
    path = _record_path(ms)
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"fields": {str(k): v for k, v in record["fields"].items()},
                                     "written": {str(k): v for k, v in record["written"].items()}}, indent=1))
    os.replace(temporary, path)


def table_signature(path) -> list:
    """``[relative path, size, mtime_ns]`` of every file of a table (lock files aside): changes when it is rewritten."""
    root = Path(path)
    return [[str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "table.lock"]


def _plain_entry(entry) -> dict:
    """An entry of :func:`vlbipy.solvers.apply.apply_tables` (dict, or tuple of ``normalise_entries``) as JSON."""
    if isinstance(entry, tuple):
        table, interp, spwmap, field_id, calwt = entry
        return {"path": str(Path(table.path).absolute()), "interp": interp, "spwmap": list(spwmap),
                "gainfield": [] if field_id is None else [int(f) for f in field_id], "calwt": bool(calwt)}
    gainfield = entry.get("gainfield", [])
    if gainfield is None:
        gainfield = []
    elif not isinstance(gainfield, str):
        gainfield = [int(f) for f in gainfield]
    source = entry.get("path", entry.get("table"))
    return {"path": str(Path(getattr(source, "path", source)).absolute()),
            "interp": str(entry.get("interp", "linear") or "linear"),
            "spwmap": [int(s) for s in (entry.get("spwmap") or [])], "gainfield": gainfield,
            "calwt": bool(entry.get("calwt", False))}


def chain_digest(entries: list, *, parang: bool, write_flags: bool) -> str:
    """Digest of a chain: the tables (paths and file signatures), their options, ``parang`` and ``write_flags``."""
    plain = [dict(_plain_entry(e), signature=table_signature(_plain_entry(e)["path"])) for e in entries]
    payload = json.dumps({"entries": plain, "parang": bool(parang), "write_flags": bool(write_flags)},
                         sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def begin_write(ms, field_ids) -> None:
    """CORRECTED_DATA of ``field_ids`` is about to be rewritten: until :func:`record_written` it holds nothing known."""
    record = load_record(ms)
    if not record["fields"] and not record["written"]:
        return
    for field_id in field_ids:
        record["written"][int(field_id)] = None
    _save(ms, record)


def record_written(ms, entries_by_field: dict, *, parang: bool, write_flags: bool) -> None:
    """applycal wrote CORRECTED_DATA of these fields with these chains (it is no longer pending).

    Nothing is recorded for a measurement set that has no record (no application was ever deferred on it).
    """
    record = load_record(ms)
    if not record["fields"] and not record["written"]:
        return
    for field_id, entries in entries_by_field.items():
        digest = chain_digest(entries, parang=parang, write_flags=write_flags)
        record["fields"][int(field_id)] = {"digest": digest, "entries": [_plain_entry(e) for e in entries],
                                           "parang": bool(parang), "write_flags": bool(write_flags),
                                           "weights": True}
        record["written"][int(field_id)] = digest
    _collect(ms, record)
    _save(ms, record)


def mark_written(ms, field_ids) -> None:
    """CORRECTED_DATA of ``field_ids`` now holds their recorded chain (see :func:`write_pending`)."""
    record = load_record(ms)
    for field_id in field_ids:
        record["written"][int(field_id)] = record["fields"][int(field_id)]["digest"]
    _save(ms, record)


def weights_set(ms, field_ids) -> None:
    """The WEIGHT columns of ``field_ids`` were written after their chain was recorded (statwt).

    Until the next application they are what the corrected data of those fields weigh: an applycal written
    before statwt would have left them so. The pending fields then read their weights as they are on disk,
    and :func:`write_pending` does not rebuild them.
    """
    record = load_record(ms)
    changed = False
    for field_id, chain in record["fields"].items():
        if field_id in {int(f) for f in field_ids} and chain.get("weights", True):
            chain["weights"] = False
            changed = True
    if changed:
        _save(ms, record)


def record_deferred(ms, entries_by_field: dict, *, parang: bool, write_flags: bool) -> list[int]:
    """Record the chains of these fields without writing CORRECTED_DATA; returns the fields now pending.

    Every table is frozen: copied under the record (once per table version), so that a later solve that
    rewrites it in place does not change what the corrected column of these fields is.
    """
    record = load_record(ms)
    frozen_root = Path(ms) / RECORD_DIR / _TABLES
    for field_id, entries in entries_by_field.items():
        plain = []
        for entry in entries:
            item = _plain_entry(entry)
            signature = table_signature(item["path"])
            version = hashlib.blake2b(json.dumps([item["path"], signature]).encode(), digest_size=8).hexdigest()
            frozen = frozen_root / version / Path(item["path"]).name
            if not frozen.is_dir():
                # With the lock files: a casacore lock file carries the table's synchronisation record, which
                # for some tables (the SPECTRAL_WINDOW of a CASA import) is the only place that holds the
                # current number of rows. Copied aside and renamed once complete, so that a copy cut short
                # is never taken for the table.
                partial = frozen.with_name(frozen.name + ".partial")
                shutil.rmtree(partial, ignore_errors=True)
                shutil.copytree(item["path"], partial)
                partial.rename(frozen)
            plain.append(dict(item, frozen=str(frozen.relative_to(Path(ms)))))
        record["fields"][int(field_id)] = {"digest": chain_digest(entries, parang=parang, write_flags=write_flags),
                                           "entries": plain, "parang": bool(parang),
                                           "write_flags": bool(write_flags), "weights": True}
    _collect(ms, record)
    _save(ms, record)
    return sorted(f for f in entries_by_field if record["written"].get(int(f)) != record["fields"][int(f)]["digest"])


def record_external(ms, field_ids, label: str) -> None:
    """CORRECTED_DATA of ``field_ids`` was written by something this record cannot describe (CASA applycal,
    an incremental application): those fields read it as it is."""
    record = load_record(ms)
    if not record["fields"] and not record["written"]:
        return
    for field_id in field_ids:
        record["fields"][int(field_id)] = {"digest": f"external:{label}", "entries": [], "parang": False,
                                           "write_flags": False}
        record["written"][int(field_id)] = f"external:{label}"
    _collect(ms, record)
    _save(ms, record)


def forget(ms) -> None:
    """Drop the record (CORRECTED_DATA was reset to DATA, or removed)."""
    shutil.rmtree(Path(ms) / RECORD_DIR, ignore_errors=True)


def _collect(ms, record: dict) -> None:
    """Remove the frozen tables no recorded chain refers to any more."""
    root = Path(ms) / RECORD_DIR / _TABLES
    if not root.is_dir():
        return
    used = {Path(e["frozen"]).parent.name for chain in record["fields"].values()
            for e in chain.get("entries", []) if "frozen" in e}
    for version in root.iterdir():
        if version.name not in used:
            shutil.rmtree(version, ignore_errors=True)


def pending(ms) -> dict:
    """``{field id: chain}`` of the fields whose CORRECTED_DATA does not hold their recorded chain."""
    record = load_record(ms)
    return {f: chain for f, chain in record["fields"].items() if record["written"].get(f) != chain["digest"]}


def frozen_entries(ms, chain: dict) -> list[dict]:
    """The entries of a recorded chain, pointing at its frozen tables (at the tables themselves if written)."""
    return [dict(e, path=str(Path(ms) / e["frozen"]) if "frozen" in e else e["path"]) for e in chain["entries"]]


def _main_columns(ms) -> list[str]:
    from .msio import ms_parts, open_table
    handle = open_table(ms_parts(ms)[0])
    try:
        return list(handle.colnames())
    finally:
        handle.close()


def view(ms, column: str = "CORRECTED_DATA", field_ids=None, *, setup: Optional[dict] = None,
         columns: Optional[list[str]] = None) -> Optional[dict]:
    """How a reader of ``column`` gets the corrected data of ``field_ids`` (None = every field).

    Returns
    -------
    dict or None
        None when nothing has to be corrected on the fly: ``column`` is not CORRECTED_DATA, or every
        selected field has its chain written. Otherwise ``entries_by_field`` (``{field id: normalised
        entries}`` of the pending fields), ``parang``, ``write_flags`` and ``weights`` (``{field id: bool}``;
        ``weights`` False when the WEIGHT columns are to be read as they are, see :func:`weights_set`), ``setup``
        (:func:`vlbipy.solvers.msio.read_setup`) and ``columns`` (main-table column names): what
        :func:`vlbipy.solvers.applycal.correct_block` needs. Picklable, so it can go to worker jobs.
    """
    if str(column).upper() != "CORRECTED_DATA":
        return None
    stale = pending(ms)
    if field_ids is not None:
        wanted = {int(f) for f in field_ids}
        stale = {f: chain for f, chain in stale.items() if f in wanted}
    if not stale:
        return None
    from .apply import normalise_entries
    from .msio import read_setup
    setup = setup or read_setup(ms)
    if len(_FROZEN_CACHE) > 32:
        _FROZEN_CACHE.clear()
    entries = {f: normalise_entries(frozen_entries(ms, chain), calwt=True, cache=_FROZEN_CACHE,
                                    field_dirs=setup["field_dirs"], target_field=f)
               for f, chain in stale.items()}
    return {"entries_by_field": entries, "parang": {f: bool(c["parang"]) for f, c in stale.items()},
            "write_flags": {f: bool(c["write_flags"]) for f, c in stale.items()},
            "weights": {f: bool(c.get("weights", True)) for f, c in stale.items()}, "setup": setup,
            "columns": list(columns) if columns is not None else _main_columns(ms)}


def write_pending(ms, field_ids=None, *, workers: Optional[int] = None) -> list[int]:
    """Write CORRECTED_DATA of the pending fields (of ``field_ids``, None = all) with their recorded chains.

    Returns the fields written (none when nothing was pending).
    """
    from .applycal import apply_to_ms
    stale = pending(ms)
    if field_ids is not None:
        stale = {f: c for f, c in stale.items() if f in {int(i) for i in field_ids}}
    stale = {f: c for f, c in stale.items() if c["entries"]}
    groups: dict = {}
    for field_id, chain in stale.items():
        key = (chain["parang"], chain["write_flags"], chain.get("weights", True))
        groups.setdefault(key, {})[field_id] = frozen_entries(ms, chain)
    for (parang, write_flags, weights), entries_by_field in groups.items():
        logger.info("applycal[dask-ms]: writing the deferred calibration of {} field(s) of {}",
                    len(entries_by_field), Path(ms).name)
        begin_write(ms, entries_by_field)
        apply_to_ms(ms, entries_by_field, parang=parang, applymode="calflagstrict" if write_flags else "calonly",
                    workers=workers, record=False, write_weights=weights)
        mark_written(ms, entries_by_field)
    return sorted(stale)
