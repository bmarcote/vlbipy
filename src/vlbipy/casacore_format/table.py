"""A casacore table opened without casacore: column reads, in-place writes, keywords, added columns.

:class:`Table` reads ``table.dat`` (:mod:`.tabledat`) and opens one reader per data manager
(:mod:`.tsm`, :mod:`.ssm`, :mod:`.ism`). Arrays are returned row-major ``(nrow, *cell)`` in C axis order
(``(nrow, nchan, ncorr)`` for DATA) and native byte order, exactly as python-casacore's ``getcol`` returns
them; string scalars as a list of str. A concatenated table (the main table of a Multi-MS) is read as the
concatenation of its parts.

Writes are in place: cells that already exist can be overwritten (tiled columns through a memory map,
fixed-size StandardStMan cells); :meth:`Table.add_column_like` adds a tiled column with the layout of
another (CORRECTED_DATA like DATA). Anything else raises :class:`UnsupportedTable`, and the caller falls back to
casacore. The table is not locked: the pipeline never has casacore and this module write the same table at the
same time.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .aipsio import TableRef
from .ism import IncrementalManager
from .ssm import StandardManager
from .tabledat import (ConcatDat, DataManagerEntry, LockInfo, TableDat, UnsupportedTable, effective_nrow,
                       parse_concat_dat, parse_table_dat, read_lock_info, table_kind, write_lock_info,
                       write_table_dat)
from .tsm import TILED_TYPES, Cube, TiledManager, create_tiled_shape_manager, write_header

SUPPORTED_MANAGERS = ("StandardStMan", "IncrementalStMan") + TILED_TYPES


def _open_manager(path: Path, dat: TableDat, manager: DataManagerEntry, writable: bool):
    columns = dat.columns_of(manager.seqnr)
    if manager.type_name in TILED_TYPES:
        return TiledManager(path, manager.seqnr, manager.type_name, columns, dat.big_endian, dat.nrow, writable)
    if manager.type_name == "StandardStMan":
        return StandardManager(path, manager, columns, dat.big_endian, dat.nrow, writable)
    if manager.type_name == "IncrementalStMan":
        return IncrementalManager(path, manager, columns, dat.big_endian, dat.nrow)
    raise UnsupportedTable(f"data manager {manager.type_name} is not supported")


class Table:
    """A casacore table on disk.

    Parameters
    ----------
    path : path
        The table directory.
    readonly : bool
        Open for reading only (default). ``False`` allows :meth:`putcol` and :meth:`add_column_like`.
    """

    def __init__(self, path, readonly: bool = True) -> None:
        self.path = Path(path)
        self.readonly = readonly
        raw = (self.path / "table.dat").read_bytes()
        kind = table_kind(raw)
        self.parts: list[Table] = []
        self.concat: Optional[ConcatDat] = None
        if kind == "ConcatTable":
            self.concat = parse_concat_dat(raw)
            self.parts = [Table(self.path / name.replace("././", ""), readonly) for name in self.concat.parts]
            self.dat = self.parts[0].dat if self.parts else None
            self._managers = {}
            return
        if kind != "PlainTable":
            raise UnsupportedTable(f"{self.path}: {kind} is not supported")
        self.dat = parse_table_dat(raw)
        self.dat.nrow = effective_nrow(self.path, self.dat)
        for manager in self.dat.managers:
            if manager.type_name not in SUPPORTED_MANAGERS and self.dat.columns_of(manager.seqnr):
                raise UnsupportedTable(f"{self.path}: data manager {manager.type_name} is not supported")
        self._managers: dict[int, object] = {}

    # -- description --
    def nrows(self) -> int:
        """Number of rows."""
        if self.parts:
            return sum(part.nrows() for part in self.parts)
        return int(self.dat.nrow)

    def colnames(self) -> list[str]:
        """Column names in description order."""
        return [column.name for column in self.dat.columns]

    def coldesc(self, name: str):
        """The :class:`~.tabledat.ColumnDesc` of a column."""
        return self.dat.column(name)

    def getcoldesc(self, name: str) -> dict:
        """The column description as python-casacore's ``getcoldesc`` returns it (main fields)."""
        column = self.dat.column(name)
        desc = {"valueType": column.value_type, "dataManagerType": column.dm_type,
                "dataManagerGroup": column.dm_group, "option": column.option, "maxlen": column.maxlen,
                "comment": column.comment, "keywords": dict(column.keywords)}
        if column.is_array:
            desc["ndim"] = column.ndim
            if column.shape:
                desc["shape"] = np.asarray(column.shape)
        return desc

    def getkeywords(self) -> dict:
        """Table keywords; a subtable is given as ``"Table: <absolute path>"``, as casatools and python-casacore
        give it (the raw :class:`~.aipsio.TableRef` values are in ``self.dat.keywords``)."""
        return {key: (f"Table: {self.subtable(key).absolute()}" if isinstance(value, TableRef) else value)
                for key, value in self.dat.keywords.items()}

    def info(self) -> dict:
        """``table.info`` as casatools returns it: ``{"type", "subType", "readme"}``."""
        path = self.path / "table.info"
        lines = path.read_text().split("\n") if path.is_file() else []
        table_type = lines[0].split("=", 1)[1].strip() if lines and "=" in lines[0] else ""
        sub_type = lines[1].split("=", 1)[1].strip() if len(lines) > 1 and "=" in lines[1] else ""
        return {"type": table_type, "subType": sub_type, "readme": "\n".join(lines[3:]) if len(lines) > 3 else ""}

    def getcolkeywords(self, name: str) -> dict:
        """Keywords of one column (units, measure info)."""
        return dict(self.dat.column(name).keywords)

    def subtable(self, name: str) -> Path:
        """The path of the subtable a table keyword refers to (e.g. ``"ANTENNA"``)."""
        ref = self.dat.keywords.get(name)
        if not isinstance(ref, TableRef):
            raise KeyError(f"{self.path}: no subtable keyword {name!r}")
        text = ref.name
        if text.startswith("././"):
            return self.path / text[4:]
        candidate = Path(text)
        return candidate if candidate.is_absolute() else self.path / text

    # -- reads --
    def _manager(self, seqnr: int):
        if seqnr not in self._managers:
            self._managers[seqnr] = _open_manager(self.path, self.dat, self.dat.manager(seqnr), not self.readonly)
        return self._managers[seqnr]

    def _rows(self, start: int, nrow: int) -> tuple[int, int]:
        total = self.nrows()
        if nrow is None or nrow < 0:
            nrow = total - start
        if start < 0 or start + nrow > total:
            raise IndexError(f"{self.path}: rows {start}..{start + nrow} outside the table ({total} rows)")
        return start, nrow

    def getcol(self, name: str, start: int = 0, nrow: int = -1):
        """Rows ``[start, start + nrow)`` of a column (``nrow=-1``: to the end)."""
        start, nrow = self._rows(start, nrow)
        if self.parts:
            return self._concat_read(lambda part, s, n: part.getcol(name, s, n), start, nrow)
        column = self.dat.column(name)
        return self._manager(column.seqnr).read(column, start, nrow)

    def getcolview(self, name: str, start: int = 0, nrow: int = -1):
        """Like :meth:`getcol` but, for tiled columns, a read-only view of the memory map where possible."""
        start, nrow = self._rows(start, nrow)
        column = self.dat.column(name)
        manager = self._manager(column.seqnr)
        if isinstance(manager, TiledManager):
            return manager.read(column, start, nrow, copy=False)
        return manager.read(column, start, nrow)

    def getcolslice(self, name: str, blc, trc, inc=None, start: int = 0, nrow: int = -1) -> np.ndarray:
        """A slice of the cells of an array column; ``blc``/``trc`` (inclusive) and ``inc`` in C axis order,
        e.g. ``[chan, corr]``, as python-casacore takes them."""
        start, nrow = self._rows(start, nrow)
        if self.parts:
            return self._concat_read(lambda part, s, n: part.getcolslice(name, blc, trc, inc, s, n), start, nrow)
        inc = inc or [1] * len(blc)
        cells = self.getcolview(name, start, nrow)
        # As in casacore: a negative blc starts at 0, a negative trc runs to the end of the axis.
        index = (slice(None),) + tuple(slice(max(int(b), 0), (int(t) + 1) if int(t) >= 0 else n, max(int(i), 1))
                                       for b, t, i, n in zip(blc, trc, inc, cells.shape[1:]))
        return np.array(cells[index])

    def getcell(self, name: str, row: int):
        """One cell."""
        values = self.getcol(name, row, 1)
        return values[0]

    def _concat_read(self, read, start: int, nrow: int):
        pieces = []
        offset = 0
        for part in self.parts:
            n = part.nrows()
            lo, hi = max(start, offset), min(start + nrow, offset + n)
            if lo < hi:
                pieces.append(read(part, lo - offset, hi - lo))
            offset += n
        if len(pieces) == 1:
            return pieces[0]
        if pieces and isinstance(pieces[0], list):
            return [value for piece in pieces for value in piece]
        return np.concatenate(pieces) if pieces else np.zeros(0)

    # -- writes --
    def putcol(self, name: str, values, start: int = 0, nrow: int = -1) -> None:
        """Overwrite rows ``[start, start + len(values))`` of a column (cells must already exist)."""
        if self.readonly:
            raise PermissionError(f"{self.path} was opened read-only")
        if self.parts:
            raise UnsupportedTable("writing through a concatenated table; open its parts")
        values = np.asarray(values)
        if nrow is not None and nrow >= 0 and nrow != len(values):
            raise ValueError(f"putcol({name}): {len(values)} rows of values for nrow={nrow}")
        column = self.dat.column(name)
        manager = self._manager(column.seqnr)
        if isinstance(manager, IncrementalManager):
            raise UnsupportedTable(f"in-place writes to IncrementalStMan column {name}")
        manager.write(column, start, values)

    def add_column_like(self, name: str, template: str) -> None:
        """Add column ``name`` with the description of array column ``template`` and every cell zero.

        A tiled template's tiling is copied; any other template gets a new TiledShapeStMan with one hypercube
        per cell shape. Writes the new data manager's header and zero-filled (sparse) tile files, then
        ``table.dat`` and the ``table.lock`` sync record.
        """
        if self.readonly:
            raise PermissionError(f"{self.path} was opened read-only")
        if name in self.colnames():
            raise ValueError(f"{self.path}: column {name} exists")
        source = self.dat.column(template)
        manager_entry = self.dat.manager(source.seqnr)
        seqnr = int(self.dat.seq_count)
        column = copy.deepcopy(source)
        column.name = column.original_name = name
        column.comment = f"The {name.lower().replace('_', ' ')} column"
        column.dm_group = f"Tiled{name.title().replace('_', '')}"
        column.seqnr = seqnr
        source_manager = self._manager(source.seqnr)
        if manager_entry.type_name not in TILED_TYPES or len(source_manager.columns) != 1:
            # Template in another storage manager (or sharing its tiles): a new TiledShapeStMan, cube per shape.
            if not source.is_array:
                raise UnsupportedTable(f"{template} is not an array column")
            column.dm_type = "TiledShapeStMan"
            column.option &= ~1                               # tiled cells are not "direct"
            create_tiled_shape_manager(self.path, seqnr, column, self.cell_shapes(template), self.dat.big_endian,
                                       column.dm_group)
            self._register_column(column, DataManagerEntry("TiledShapeStMan", seqnr, b""))
            return
        cubes = []
        for cube in source_manager.cubes:
            cubes.append(Cube(cube.shape, cube.tile_shape, cube.file_seqnr, cube.offset, cube.extensible,
                              dict(cube.values)))
        for file_seqnr, length in source_manager.files.items():
            target = self.path / f"table.f{seqnr}_TSM{file_seqnr}"
            with open(target, "wb") as handle:
                if length:
                    handle.truncate(int(length))
        write_header(self.path, seqnr, manager_entry.type_name, [column], self.dat.big_endian, self.dat.nrow,
                     cubes, dict(source_manager.files), column.dm_group, source_manager.default_tile_shape,
                     source_manager.row_map, source_manager.cube_map, source_manager.pos_map)
        self._register_column(column, DataManagerEntry(manager_entry.type_name, seqnr, manager_entry.blob))

    def cell_shapes(self, name: str) -> list:
        """The C-order cell shape of every row of array column ``name`` (None where a cell is undefined)."""
        column = self.dat.column(name)
        manager = self._manager(column.seqnr)
        if isinstance(manager, TiledManager):
            shapes = [None] * self.nrows()
            for cube_index, _, row, count in manager.locate(0, self.nrows()):
                shape = tuple(manager.cubes[cube_index].shape[:-1][::-1])
                for k in range(row, row + count):
                    shapes[k] = shape
            return shapes
        values = manager.read(column, 0, self.nrows())
        if isinstance(values, np.ndarray):
            return [values.shape[1:]] * len(values)
        return [None if cell is None else np.shape(cell) for cell in values]

    def _register_column(self, column, manager: DataManagerEntry) -> None:
        """Append a column and its new data manager to ``table.dat`` (and the sync record of ``table.lock``)."""
        seqnr = manager.seqnr
        self.dat.columns.append(column)
        self.dat.managers.append(manager)
        self.dat.seq_count = seqnr + 1
        write_table_dat(self.path, self.dat)
        info = read_lock_info(self.path)
        if info is not None:
            counters = tuple(info.manager_change_counters) + (0,) * (len(self.dat.managers)
                                                                     - len(info.manager_change_counters))
            write_lock_info(self.path, LockInfo(self.dat.nrow, len(self.dat.columns), info.modify_counter + 1,
                                                info.table_change_counter + 1, counters))

    def flush(self) -> None:
        """Write pending changes (memory maps, rewritten StandardStMan files)."""
        for part in self.parts:
            part.flush()
        for manager in self._managers.values():
            flush = getattr(manager, "flush", None)
            if flush is not None:
                flush()

    def close(self) -> None:
        """Flush and release the data managers."""
        self.flush()
        for part in self.parts:
            part.close()
        for manager in self._managers.values():
            manager.close()
        self._managers.clear()

    def __enter__(self) -> "Table":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def is_table(path) -> bool:
    """True when ``path`` is a casacore table directory."""
    return (Path(path) / "table.dat").is_file()


def can_open(path) -> bool:
    """True when this module can read the table at ``path`` (every data manager in it is supported)."""
    try:
        Table(path)
    except (UnsupportedTable, OSError, ValueError):
        return False
    return True


def table_endian_default() -> bool:
    """Byte order for new tables: the machine's (casacore's default, ``table.endianformat=local``)."""
    return sys.byteorder == "big"
