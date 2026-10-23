"""Flag versions (CASA ``flagmanager``) without CASA: save, restore, delete, rename and list.

The files are CASA's, so either implementation reads what the other wrote: ``<vis>.flagversions/`` holds one
table ``flags.<name>`` per version — FLAG (TiledShapeStMan) and FLAG_ROW, one row per main-table row, the parts of a
Multi-MS concatenated in order — and ``FLAG_VERSION_LIST``, one ``"<name> : <comment>"`` line per version.
Saving over an existing name first renames it to ``<name>.old.<unix time>``, as the CASA task does.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from ..casacore_format.table import Table
from ..casacore_format.tabledat import parse_concat_dat, table_kind
from ..casacore_format.writer import array_column, create_table, scalar_column
from ..logging_utils import get_logger

logger = get_logger()

LIST_NAME = "FLAG_VERSION_LIST"


def versions_dir(vis) -> Path:
    """``<vis>.flagversions``."""
    vis = Path(vis)
    return vis.parent / f"{vis.name}.flagversions"


def _parts(vis) -> list[Path]:
    """The tables holding the main rows, in row order (the parts of a Multi-MS, as its ``table.dat`` lists them)."""
    vis = Path(vis)
    raw = (vis / "table.dat").read_bytes()
    if table_kind(raw) == "ConcatTable":
        return [vis / name.replace("././", "") for name in parse_concat_dat(raw).parts]
    return [vis]


def list_versions(vis) -> list[tuple[str, str]]:
    """``[(name, comment), ...]`` from ``FLAG_VERSION_LIST``, in order."""
    listing = versions_dir(vis) / LIST_NAME
    if not listing.is_file():
        return []
    out = []
    for line in listing.read_text().splitlines():
        name, _, comment = line.partition(" : ")
        if name.strip():
            out.append((name.strip(), comment.strip()))
    return out


def _write_list(vis, versions: list[tuple[str, str]]) -> None:
    folder = versions_dir(vis)
    folder.mkdir(exist_ok=True)
    (folder / LIST_NAME).write_text("".join(f"{name} : {comment}\n" for name, comment in versions))


def save_version(vis, versionname: str, comment: str = "") -> Path:
    """Copy FLAG and FLAG_ROW of every row of ``vis`` into ``flags.<versionname>``; returns the version table."""
    if not versionname:
        raise ValueError("Illegal empty versionname: ''")
    started = time.perf_counter()
    folder = versions_dir(vis)
    folder.mkdir(exist_ok=True)
    target = folder / f"flags.{versionname}"
    versions = list_versions(vis)
    if target.exists():
        old = f"{versionname}.old.{int(time.time())}"
        logger.warning("flag version {!r} already exists; renaming it to {}", versionname, old)
        target.rename(folder / f"flags.{old}")
        versions = [(old if name == versionname else name, c) for name, c in versions]
    parts = _parts(vis)
    shapes, flag_rows = [], []
    for part in parts:
        with Table(part) as table:
            nrow = table.nrows()
            shapes.extend(table.cell_shapes("FLAG") if nrow else [])
            flag_rows.append(np.asarray(table.getcol("FLAG_ROW"), dtype=bool) if nrow else np.zeros(0, bool))
    total = len(shapes)
    columns = [array_column("FLAG", "boolean", ndim=len(shapes[0]) if shapes else 2,
                            comment="The data flags, array of bools with same shape as data"),
               scalar_column("FLAG_ROW", "boolean", comment="Row flag - flag all data in this row if True")]
    create_table(target, columns, total, {"FLAG_ROW": np.concatenate(flag_rows) if flag_rows else np.zeros(0, bool)},
                 tiled={"FLAG": shapes})
    with Table(target, readonly=False) as out:
        row = 0
        for part in parts:
            with Table(part) as table:
                for start in range(0, table.nrows(), 100_000):
                    block = table.getcol("FLAG", start, min(100_000, table.nrows() - start))
                    out.putcol("FLAG", block, row)
                    row += len(block)
    versions = [(name, c) for name, c in versions if name != versionname] + [(versionname, comment)]
    _write_list(vis, versions)
    logger.info("flagmanager[native]: saved {} rows of {} as version {!r} in {:.2f}s", total, Path(vis).name,
                versionname, time.perf_counter() - started)
    return target


def restore_version(vis, versionname: str, merge: str = "replace") -> None:
    """Write the flags of version ``versionname`` back into ``vis`` (``merge``: replace, or, and)."""
    if not versionname:
        raise ValueError("Illegal versionname: ''")
    merge = (merge or "replace").lower()
    if merge not in ("replace", "or", "and"):
        raise ValueError(f"unknown merge mode {merge!r}")
    started = time.perf_counter()
    source = versions_dir(vis) / f"flags.{versionname}"
    if not (source / "table.dat").is_file():
        raise ValueError(f"no flag version {versionname!r} for {vis}")
    with Table(source) as saved:
        row = 0
        for part in _parts(vis):
            with Table(part, readonly=False) as table:
                nrow = table.nrows()
                for start in range(0, nrow, 100_000):
                    count = min(100_000, nrow - start)
                    flag = np.asarray(saved.getcol("FLAG", row, count), dtype=bool)
                    flag_row = np.asarray(saved.getcol("FLAG_ROW", row, count), dtype=bool)
                    if merge != "replace":
                        current = np.asarray(table.getcol("FLAG", start, count), dtype=bool)
                        current_row = np.asarray(table.getcol("FLAG_ROW", start, count), dtype=bool)
                        combine = np.logical_or if merge == "or" else np.logical_and
                        flag, flag_row = combine(current, flag), combine(current_row, flag_row)
                    table.putcol("FLAG", flag, start)
                    table.putcol("FLAG_ROW", flag_row, start)
                    row += count
    logger.info("flagmanager[native]: restored version {!r} into {} (merge={}) in {:.2f}s", versionname,
                Path(vis).name, merge, time.perf_counter() - started)


def delete_version(vis, versionname: str) -> None:
    """Remove version ``versionname`` (its table and its line in the list)."""
    if not versionname:
        raise ValueError("Illegal versionname: ''")
    shutil.rmtree(versions_dir(vis) / f"flags.{versionname}", ignore_errors=True)
    _write_list(vis, [(name, c) for name, c in list_versions(vis) if name != versionname])


def rename_version(vis, oldname: str, versionname: str, comment: str = "") -> None:
    """Rename version ``oldname`` to ``versionname`` (with a new comment)."""
    folder = versions_dir(vis)
    if not (folder / f"flags.{oldname}").is_dir():
        raise ValueError(f"No such flagversions: {oldname}")
    if (folder / f"flags.{versionname}").exists():
        raise ValueError(f"Flagversions {versionname} already exists!")
    (folder / f"flags.{oldname}").rename(folder / f"flags.{versionname}")
    _write_list(vis, [((versionname, comment) if name == oldname else (name, c)) for name, c in list_versions(vis)])


def flagmanager(vis=None, mode: str = "list", versionname: str = "", oldname: str = "", comment: str = "",
                merge: str = "replace", **_):
    """``casatasks.flagmanager`` with the same keywords and files. ``mode='list'`` returns CASA's dict."""
    if not vis or not Path(str(vis)).is_dir():
        raise ValueError("Visibility data set not found - please verify the name")
    mode = (mode or "list").lower()
    if mode == "save":
        save_version(vis, versionname, comment or "")
    elif mode == "restore":
        restore_version(vis, versionname, merge)
    elif mode == "delete":
        delete_version(vis, versionname)
    elif mode == "rename":
        rename_version(vis, oldname, versionname, comment or "")
    elif mode == "list":
        out = {k: {"name": name, "comment": c} for k, (name, c) in enumerate(list_versions(vis))}
        out["MS"] = str(Path(vis).absolute())
        return out
    else:
        raise ValueError(f"Unknown mode {mode}")
    return None
