"""``table.dat``: the description of a casacore table (columns, keywords, data managers), read and written.

Layout (casacore PlainTable::putFile, TableDesc::putFile, ColumnSet::putFile), all AipsIO, big-endian:

* ``Table`` object (version 2, or 3 for more than 2**31 rows): ``nrow``, ``endian`` (0 big, 1 little: the byte
  order of the storage-manager files), ``"PlainTable"``;
* ``TableDesc``: name, version, comment, keywords (``TableRecord``), private keywords, then the columns, each
  ``[uint 1][class name][BaseColumnDesc]`` with name, comment, data-manager type and group, data type, option,
  ndim, shape (array columns), max string length, column keywords, and the class-specific tail (scalar default
  value, or the always-false "has default" switch of array columns);
* the column set: ``int -2`` (or ``-3`` + storage option), nrow, the data-manager sequence counter, the
  data-manager list ``(type, seqnr)``, per column ``[uint 2][original name][uint 1][seqnr]`` (+ for arrays
  ``bool shapeColDef`` and its ``IPosition``), and finally one blob per data manager (whatever the manager
  writes into ``table.dat``; most keep their state in their own ``table.f<seqnr>`` file).

:func:`read_table_dat` parses this into :class:`TableDat`; :func:`write_table_dat` writes it back so that an
unmodified description round-trips byte for byte (tested), and a modified one (an added column, a new table)
is what casacore itself would write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .aipsio import (TP_BOOL, TP_COMPLEX, TP_DCOMPLEX, TP_DOUBLE, TP_FLOAT, TP_INT, TP_INT64, TP_SHORT,
                     TP_STRING, TP_UCHAR, TP_UINT, AipsReader, AipsWriter, RecordField, fields_of)

#: Column option bits (casacore ColumnDesc::Option).
OPTION_DIRECT = 1
OPTION_UNDEFINED = 2
OPTION_FIXED_SHAPE = 4

#: The type id casacore puts in column class names (ValType::getTypeStr), keyed by data type.
TYPE_IDS = {TP_BOOL: "Bool    ", TP_UCHAR: "uChar   ", TP_SHORT: "Short   ", TP_INT: "Int     ",
            TP_UINT: "uInt    ", TP_INT64: "Int64   ", TP_FLOAT: "float   ", TP_DOUBLE: "double  ",
            TP_COMPLEX: "Complex ", TP_DCOMPLEX: "DComplex", TP_STRING: "String  "}
#: Names casacore (and python-casacore's ``getcoldesc``) use for the value types.
VALUE_TYPE_NAMES = {TP_BOOL: "boolean", TP_UCHAR: "uchar", TP_SHORT: "short", TP_INT: "int", TP_UINT: "uint",
                    TP_INT64: "int64", TP_FLOAT: "float", TP_DOUBLE: "double", TP_COMPLEX: "complex",
                    TP_DCOMPLEX: "dcomplex", TP_STRING: "string"}


@dataclass
class ColumnDesc:
    """One column of the table description.

    ``shape`` is the fixed cell shape in casacore (Fortran) axis order, e.g. ``(ncorr, nchan)`` for DATA, or
    None when not fixed; ``ndim`` is -1 for arrays of any dimensionality, 0 for scalars.
    """
    name: str
    dtype: int
    is_array: bool
    dm_type: str = "StandardStMan"
    dm_group: str = ""
    comment: str = ""
    option: int = 0
    ndim: int = 0
    shape: Optional[tuple] = None
    maxlen: int = 0
    keywords: dict = field(default_factory=dict)
    keyword_fields: Optional[list] = None
    class_name: str = ""
    default: Any = None
    seqnr: int = 0
    original_name: str = ""
    shape_col_def: bool = False
    shape_col: tuple = ()

    def __post_init__(self) -> None:
        if not self.class_name:
            prefix = "ArrayColumnDesc<" if self.is_array else "ScalarColumnDesc<"
            self.class_name = prefix + TYPE_IDS[self.dtype]
        if not self.original_name:
            self.original_name = self.name
        if self.default is None and not self.is_array:
            self.default = "" if self.dtype == TP_STRING else (False if self.dtype == TP_BOOL else 0)

    @property
    def value_type(self) -> str:
        """``"int"``, ``"double"``, ``"complex"``, ... (python-casacore naming)."""
        return VALUE_TYPE_NAMES.get(self.dtype, str(self.dtype))


@dataclass
class DataManagerEntry:
    """A data manager: its type name (``StandardStMan``, ``TiledShapeStMan``, ...), sequence number and blob."""
    type_name: str
    seqnr: int
    blob: bytes = b""


@dataclass
class TableDat:
    """The parsed ``table.dat`` of a plain table."""
    nrow: int
    big_endian: bool
    columns: list[ColumnDesc]
    managers: list[DataManagerEntry]
    keywords: dict = field(default_factory=dict)
    keyword_fields: Optional[list] = None
    private_keywords: dict = field(default_factory=dict)
    desc_name: str = ""
    desc_version: str = ""
    desc_comment: str = ""
    table_type: str = "PlainTable"
    seq_count: int = 0
    colset_version: int = 2
    storage_option: tuple = (0, 0)

    def column(self, name: str) -> ColumnDesc:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"no column {name!r}")

    def manager(self, seqnr: int) -> DataManagerEntry:
        for manager in self.managers:
            if manager.seqnr == seqnr:
                return manager
        raise KeyError(f"no data manager with sequence number {seqnr}")

    def columns_of(self, seqnr: int) -> list[ColumnDesc]:
        """The columns stored by data manager ``seqnr``, in description order."""
        return [column for column in self.columns if column.seqnr == seqnr]


class UnsupportedTable(ValueError):
    """The table uses a feature this reader does not implement (the caller falls back to casacore)."""


def _read_record_with_fields(reader: AipsReader) -> tuple[dict, list]:
    """A TableRecord plus its field list (kept so that writing it back preserves the exact description)."""
    record = reader.table_record()
    return record, record.fields


def _read_column(reader: AipsReader) -> ColumnDesc:
    reader.u32()                                             # ColumnDesc version
    class_name = reader.string()
    if not (class_name.startswith("ScalarColumnDesc<") or class_name.startswith("ArrayColumnDesc<")):
        raise UnsupportedTable(f"column class {class_name!r} is not supported")
    is_array = class_name.startswith("ArrayColumnDesc<")
    reader.u32()                                             # BaseColumnDesc version
    name = reader.string()
    comment = reader.string()
    dm_type = reader.string()
    dm_group = reader.string()
    dtype = reader.i32()
    option = reader.i32()
    ndim = reader.i32()
    shape = reader.ipos() if is_array else None
    maxlen = reader.u32()
    keywords, keyword_fields = _read_record_with_fields(reader)
    reader.u32()                                             # derived-class version
    default = None
    if is_array:
        reader.bool()                                        # obsolete "has default" switch
    else:
        default = reader.scalar(dtype)
    return ColumnDesc(name=name, dtype=dtype, is_array=is_array, dm_type=dm_type, dm_group=dm_group,
                      comment=comment, option=option, ndim=ndim, shape=tuple(shape) if shape else None,
                      maxlen=maxlen, keywords=keywords, keyword_fields=keyword_fields, class_name=class_name,
                      default=default)


@dataclass
class ConcatDat:
    """The ``table.dat`` of a concatenated table (the main table of a Multi-MS): its parts and the subtables
    they share, all relative to the table directory."""
    nrow: int
    parts: list[str]
    subtables: list[str]


def table_kind(data: bytes) -> str:
    """``"PlainTable"``, ``"ConcatTable"`` or ``"RefTable"`` for the bytes of a ``table.dat``."""
    reader = AipsReader(data, big_endian=True)
    _, version = reader.start("Table")
    reader.u64() if version > 2 else reader.u32()
    reader.u32()
    return reader.string()


def parse_concat_dat(data: bytes) -> ConcatDat:
    """Parse the ``table.dat`` of a concatenated table."""
    reader = AipsReader(data, big_endian=True)
    _, version = reader.start("Table")
    nrow = reader.u64() if version > 2 else reader.u32()
    reader.u32()
    if reader.string() != "ConcatTable":
        raise UnsupportedTable("not a concatenated table")
    reader.start("ConcatTable")
    parts = [reader.string() for _ in range(reader.u32())]
    subtables = list(reader.block(TP_STRING))
    reader.end()
    reader.end()
    return ConcatDat(nrow=nrow, parts=parts, subtables=subtables)


def write_concat_dat(table_path, nrow: int, parts: list[str], subtables: list[str] = ()) -> None:
    """Write the ``table.dat`` of a concatenated table (a Multi-MS root): ``parts`` relative to it, e.g.
    ``"SUBMSS/X.mms.0000.ms"`` (written as ``././SUBMSS/...``), and the names of the subtables they share."""
    writer = AipsWriter(big_endian=True)
    wide = nrow > 2147483647
    writer.start("Table", 3 if wide else 2)
    writer.u64(nrow) if wide else writer.u32(nrow)
    writer.u32(0)
    writer.string("ConcatTable")
    writer.start("ConcatTable", 0)
    writer.u32(len(parts))
    for part in parts:
        writer.string(part if part.startswith("././") else f"././{part}")
    writer.block(TP_STRING, list(subtables))
    writer.end()
    writer.end()
    path = Path(table_path)
    (path / "table.dat_tmp").write_bytes(writer.getvalue())
    (path / "table.dat_tmp").replace(path / "table.dat")


def parse_table_dat(data: bytes) -> TableDat:
    """Parse the bytes of a ``table.dat``."""
    reader = AipsReader(data, big_endian=True)
    _, version = reader.start("Table")
    nrow = reader.u64() if version > 2 else reader.u32()
    big_endian = reader.u32() == 0
    table_type = reader.string()
    if table_type != "PlainTable":
        raise UnsupportedTable(f"{table_type} is not a plain table")
    reader.start("TableDesc")
    desc_name = reader.string()
    desc_version = reader.string()
    desc_comment = reader.string()
    keywords, keyword_fields = _read_record_with_fields(reader)
    private_keywords = reader.table_record()
    columns = [_read_column(reader) for _ in range(reader.u32())]
    reader.end()

    first = reader.i32()
    colset_version = -first if first < 0 else 1
    if colset_version == 1:
        raise UnsupportedTable("column set version 1 (very old table) is not supported")
    storage_option = (0, 0)
    if colset_version >= 3:
        reader.u64()
        storage_option = (reader.i32(), reader.i32())
    else:
        reader.u32()
    seq_count = reader.u32()
    managers = [DataManagerEntry(reader.string(), reader.u32()) for _ in range(reader.u32())]
    for column in columns:
        if reader.u32() != 2:
            raise UnsupportedTable(f"old column format for {column.name}")
        column.original_name = reader.string()
        reader.u32()
        column.seqnr = reader.u32()
        if column.is_array:
            column.shape_col_def = reader.bool()
            if column.shape_col_def:
                column.shape_col = reader.ipos()
    for manager in managers:
        manager.blob = reader.raw(reader.u32())
    reader.end()
    return TableDat(nrow=nrow, big_endian=big_endian, columns=columns, managers=managers, keywords=keywords,
                    keyword_fields=keyword_fields, private_keywords=private_keywords, desc_name=desc_name,
                    desc_version=desc_version, desc_comment=desc_comment, table_type=table_type,
                    seq_count=seq_count, colset_version=colset_version, storage_option=storage_option)


def read_table_dat(table_path) -> TableDat:
    """Read and parse ``<table_path>/table.dat``."""
    return parse_table_dat((Path(table_path) / "table.dat").read_bytes())


def _write_column(writer: AipsWriter, column: ColumnDesc) -> None:
    writer.u32(1)
    writer.string(column.class_name)
    writer.u32(1)
    writer.string(column.name)
    writer.string(column.comment)
    writer.string(column.dm_type)
    writer.string(column.dm_group)
    writer.i32(column.dtype)
    writer.i32(column.option)
    writer.i32(column.ndim)
    if column.is_array:
        writer.ipos(column.shape or ())
    writer.u32(column.maxlen)
    writer.table_record(column.keywords, fields=record_fields(column.keywords, column.keyword_fields))
    writer.u32(1)
    if column.is_array:
        writer.bool(False)
    else:
        writer.scalar(column.dtype, column.default)


def serialize_table_dat(dat: TableDat) -> bytes:
    """The bytes of ``table.dat`` for a description."""
    writer = AipsWriter(big_endian=True)
    wide = dat.nrow > 2147483647
    writer.start("Table", 3 if wide else 2)
    writer.u64(dat.nrow) if wide else writer.u32(dat.nrow)
    writer.u32(0 if dat.big_endian else 1)
    writer.string(dat.table_type)
    writer.start("TableDesc", 2)
    writer.string(dat.desc_name)
    writer.string(dat.desc_version)
    writer.string(dat.desc_comment)
    writer.table_record(dat.keywords, fields=record_fields(dat.keywords, dat.keyword_fields))
    writer.table_record(dat.private_keywords)
    writer.u32(len(dat.columns))
    for column in dat.columns:
        _write_column(writer, column)
    writer.end()

    if dat.colset_version >= 3 or wide:
        writer.i32(-3)
        writer.u64(dat.nrow)
        writer.i32(dat.storage_option[0])
        writer.i32(dat.storage_option[1])
    else:
        writer.i32(-2)
        writer.u32(dat.nrow)
    writer.u32(dat.seq_count)
    writer.u32(len(dat.managers))
    for manager in dat.managers:
        writer.string(manager.type_name)
        writer.u32(manager.seqnr)
    for column in dat.columns:
        writer.u32(2)
        writer.string(column.original_name)
        writer.u32(1)
        writer.u32(column.seqnr)
        if column.is_array:
            writer.bool(column.shape_col_def)
            if column.shape_col_def:
                writer.ipos(column.shape_col)
    for manager in dat.managers:
        writer.u32(len(manager.blob))
        writer.raw(manager.blob)
    writer.end()
    return writer.getvalue()


def write_table_dat(table_path, dat: TableDat) -> None:
    """Write ``<table_path>/table.dat`` atomically (via ``table.dat_tmp`` and a rename, as casacore does)."""
    path = Path(table_path)
    tmp = path / "table.dat_tmp"
    tmp.write_bytes(serialize_table_dat(dat))
    tmp.replace(path / "table.dat")


#: Bytes of lock-request slots at the start of ``table.lock`` (casacore LockFile: (1 + 2 * 32) ints).
LOCK_REQUEST_BYTES = 260


@dataclass
class LockInfo:
    """The sync record casacore keeps in ``table.lock``: the row count other processes must use (it is newer
    than ``table.dat`` when only data, not the description, was flushed) and change counters."""
    nrow: int
    ncolumn: int
    modify_counter: int = 0
    table_change_counter: int = 0
    manager_change_counters: tuple = ()


def read_lock_info(table_path) -> Optional[LockInfo]:
    """The sync record of ``<table_path>/table.lock``, or None when there is none."""
    path = Path(table_path) / "table.lock"
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < LOCK_REQUEST_BYTES + 4:
        return None
    length = int.from_bytes(data[LOCK_REQUEST_BYTES:LOCK_REQUEST_BYTES + 4], "big")
    if length == 0 or len(data) < LOCK_REQUEST_BYTES + 4 + length:
        return None
    reader = AipsReader(data[LOCK_REQUEST_BYTES + 4:LOCK_REQUEST_BYTES + 4 + length], big_endian=True)
    _, version = reader.start("sync")
    nrow = reader.u32() if version == 1 else reader.u64()
    ncolumn = reader.i32()
    modify = reader.u32()
    if ncolumn < 0:
        return LockInfo(nrow, ncolumn, modify)
    table_counter = reader.u32()
    counters = tuple(int(v) for v in reader.block(TP_UINT))
    return LockInfo(nrow, ncolumn, modify, table_counter, counters)


def write_lock_info(table_path, info: LockInfo) -> None:
    """Write ``table.lock`` with an unlocked request area and the given sync record (in place when it exists, so
    a process holding an fcntl lock on it keeps it)."""
    writer = AipsWriter(big_endian=True)
    wide = info.nrow > 4294967295
    writer.start("sync", 2 if wide else 1)
    writer.u64(info.nrow) if wide else writer.u32(info.nrow)
    writer.i32(info.ncolumn)
    writer.u32(info.modify_counter)
    if info.ncolumn >= 0:
        writer.u32(info.table_change_counter)
        writer.block(TP_UINT, list(info.manager_change_counters))
    writer.end()
    record = writer.getvalue()
    path = Path(table_path) / "table.lock"
    body = len(record).to_bytes(4, "big") + record
    if path.is_file():
        with open(path, "r+b") as handle:
            handle.seek(LOCK_REQUEST_BYTES)
            handle.write(body)
            handle.truncate(LOCK_REQUEST_BYTES + len(body))
    else:
        path.write_bytes(bytes(LOCK_REQUEST_BYTES) + body)


def effective_nrow(table_path, dat: TableDat) -> int:
    """The row count casacore uses: the lock file's when it has one (non-zero), else ``table.dat``'s."""
    info = read_lock_info(table_path)
    if info is not None and info.nrow > 0:
        return int(info.nrow)
    return int(dat.nrow)


def record_fields(keywords: dict, fields: Optional[list]) -> list[RecordField]:
    """The field list to write a keyword record with: the parsed one when the keys are unchanged."""
    if fields is not None and [f.name for f in fields] == list(keywords):
        return fields
    return fields_of(keywords)
