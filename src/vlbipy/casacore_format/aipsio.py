"""AipsIO: casacore's object serialisation, read and written without casacore.

Every casacore persistent object (``table.dat``, storage-manager headers, records) is a tree of AipsIO objects:

* the outermost object starts with the magic word ``0xbebebebe``;
* every object is ``[uint32 length][string type][uint32 version][payload ...]``, where ``length`` counts the
  bytes from the length word itself to the end of the payload (nested objects included);
* scalars are fixed-size in the stream's byte order (``table.dat`` is always big-endian, "canonical"; the
  storage-manager files follow the table's endianness), a string is ``uint32 n`` + ``n`` bytes, a bool is one
  byte, and an array of ``n`` bools is ``ceil(n / 8)`` bytes with the first value in the lowest bit.

:class:`AipsReader` reads such a stream, :class:`AipsWriter` writes one (object lengths are patched in on
:meth:`AipsWriter.end`). Both also (de)serialise the composite objects the table system stores: ``IPosition``,
``Array<T>``, ``RecordDesc`` and ``TableRecord`` (keyword sets). Records are returned as plain dicts with numpy
arrays; a subtable keyword (``TpTable``) is returned as :class:`TableRef`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAGIC = 0xBEBEBEBE

#: casacore ``DataType`` enum values (casa/Utilities/DataType.h).
TP_BOOL, TP_CHAR, TP_UCHAR, TP_SHORT, TP_USHORT, TP_INT, TP_UINT, TP_FLOAT, TP_DOUBLE = range(9)
TP_COMPLEX, TP_DCOMPLEX, TP_STRING, TP_TABLE = 9, 10, 11, 12
TP_ARRAY_BOOL, TP_ARRAY_CHAR, TP_ARRAY_UCHAR, TP_ARRAY_SHORT, TP_ARRAY_USHORT = 13, 14, 15, 16, 17
TP_ARRAY_INT, TP_ARRAY_UINT, TP_ARRAY_FLOAT, TP_ARRAY_DOUBLE, TP_ARRAY_COMPLEX = 18, 19, 20, 21, 22
TP_ARRAY_DCOMPLEX, TP_ARRAY_STRING, TP_RECORD, TP_OTHER = 23, 24, 25, 26
TP_QUANTITY, TP_ARRAY_QUANTITY, TP_INT64, TP_ARRAY_INT64 = 27, 28, 29, 30

#: Scalar data type -> numpy dtype (native byte order; byte order is applied on read/write).
SCALAR_DTYPES = {TP_BOOL: np.bool_, TP_CHAR: np.int8, TP_UCHAR: np.uint8, TP_SHORT: np.int16,
                 TP_USHORT: np.uint16, TP_INT: np.int32, TP_UINT: np.uint32, TP_FLOAT: np.float32,
                 TP_DOUBLE: np.float64, TP_COMPLEX: np.complex64, TP_DCOMPLEX: np.complex128,
                 TP_INT64: np.int64}
#: Array data type -> its element (scalar) data type.
ARRAY_ELEMENT = {TP_ARRAY_BOOL: TP_BOOL, TP_ARRAY_CHAR: TP_CHAR, TP_ARRAY_UCHAR: TP_UCHAR,
                 TP_ARRAY_SHORT: TP_SHORT, TP_ARRAY_USHORT: TP_USHORT, TP_ARRAY_INT: TP_INT,
                 TP_ARRAY_UINT: TP_UINT, TP_ARRAY_FLOAT: TP_FLOAT, TP_ARRAY_DOUBLE: TP_DOUBLE,
                 TP_ARRAY_COMPLEX: TP_COMPLEX, TP_ARRAY_DCOMPLEX: TP_DCOMPLEX, TP_ARRAY_STRING: TP_STRING,
                 TP_ARRAY_INT64: TP_INT64}
ELEMENT_ARRAY = {element: array for array, element in ARRAY_ELEMENT.items()}
#: The AipsIO object name casacore writes for an ``Array<T>`` inside a record (RecordRep::putDataField).
ARRAY_OBJECT_NAMES = {TP_BOOL: "Array<void>", TP_UCHAR: "Array<uChar>", TP_SHORT: "Array<short>",
                      TP_INT: "Array<Int>", TP_UINT: "Array<uInt>", TP_INT64: "Array<Int64>",
                      TP_FLOAT: "Array<float>", TP_DOUBLE: "Array<double>", TP_COMPLEX: "Array<void>",
                      TP_DCOMPLEX: "Array<void>", TP_STRING: "Array<String>", TP_CHAR: "Array<Char>",
                      TP_USHORT: "Array<uShort>"}


class Record(dict):
    """A record read from AipsIO: a dict that remembers its field descriptions (``fields``), so that writing an
    unchanged record back reproduces the exact ``RecordDesc`` (e.g. an empty ``Array<String>``)."""
    fields: Optional[list] = None


@dataclass
class TableRef:
    """A subtable keyword: the table name as stored (relative names start with ``././``)."""
    name: str


@dataclass
class RecordField:
    """One field of a ``RecordDesc``: name, data type, shape (arrays), sub-description (records), comment."""
    name: str
    dtype: int
    shape: Optional[tuple] = None
    sub: Optional[list] = None
    table_desc: str = ""
    comment: str = ""


class AipsIOError(ValueError):
    """The bytes do not hold the AipsIO object that was expected."""


def _code(big_endian: bool) -> str:
    return ">" if big_endian else "<"


class AipsReader:
    """Sequential reader of an AipsIO stream held in memory.

    Parameters
    ----------
    data : bytes
        The stream (e.g. a whole ``table.dat``).
    big_endian : bool
        Byte order of the scalars (``table.dat`` and TSM/SSM/ISM headers are big-endian unless the table was
        written little-endian, see :attr:`vlbipy.casacore_format.tabledat.TableDat.big_endian`).
    offset : int
        Where reading starts.
    """

    def __init__(self, data: bytes, big_endian: bool = True, offset: int = 0) -> None:
        self.data = memoryview(data)
        self.big_endian = big_endian
        self.pos = offset
        self._ends: list[int] = []
        self._prefix = _code(big_endian)

    # -- primitives --
    def _unpack(self, fmt: str, size: int):
        value = struct.unpack_from(self._prefix + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def u8(self) -> int:
        return self._unpack("B", 1)

    def bool(self) -> bool:
        return bool(self._unpack("B", 1) & 1)

    def i16(self) -> int:
        return self._unpack("h", 2)

    def u16(self) -> int:
        return self._unpack("H", 2)

    def i32(self) -> int:
        return self._unpack("i", 4)

    def u32(self) -> int:
        return self._unpack("I", 4)

    def i64(self) -> int:
        return self._unpack("q", 8)

    def u64(self) -> int:
        return self._unpack("Q", 8)

    def f32(self) -> float:
        return self._unpack("f", 4)

    def f64(self) -> float:
        return self._unpack("d", 8)

    def string(self) -> str:
        n = self.u32()
        text = bytes(self.data[self.pos:self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return text

    def raw(self, n: int) -> bytes:
        """``n`` bytes as they are."""
        out = bytes(self.data[self.pos:self.pos + n])
        self.pos += n
        return out

    def values(self, dtype_code: int, n: int):
        """``n`` consecutive values of a scalar data type: a numpy array, or a list of str for strings."""
        if dtype_code == TP_STRING:
            return [self.string() for _ in range(n)]
        if dtype_code == TP_BOOL:
            nbytes = (n + 7) // 8
            bits = np.frombuffer(self.data, dtype=np.uint8, count=nbytes, offset=self.pos)
            self.pos += nbytes
            return np.unpackbits(bits, bitorder="little")[:n].astype(bool)
        dtype = np.dtype(SCALAR_DTYPES[dtype_code]).newbyteorder(self._prefix)
        out = np.frombuffer(self.data, dtype=dtype, count=n, offset=self.pos)
        self.pos += n * dtype.itemsize
        return out.astype(dtype.newbyteorder("="))

    def scalar(self, dtype_code: int):
        """One value of a scalar data type as a Python/numpy scalar."""
        if dtype_code == TP_STRING:
            return self.string()
        if dtype_code == TP_BOOL:
            return self.bool()
        return self.values(dtype_code, 1)[0]

    # -- objects --
    def start(self, expected: Optional[str] = None) -> tuple[str, int]:
        """Enter an object: returns ``(type, version)``; checks the type when ``expected`` is given."""
        if not self._ends:
            magic = self.u32()
            if magic != MAGIC:
                raise AipsIOError(f"no AipsIO magic value at offset {self.pos - 4} ({magic:#x})")
        begin = self.pos
        length = self.u32()
        kind = self.string()
        version = self.u32()
        if expected is not None and kind != expected:
            raise AipsIOError(f"expected AipsIO object {expected!r}, found {kind!r} at offset {begin}")
        self._ends.append(begin + length)
        return kind, version

    def peek_type(self) -> str:
        """The type of the next object, without entering it."""
        pos = self.pos + (4 if not self._ends else 0)
        n = struct.unpack_from(self._prefix + "I", self.data, pos + 4)[0]
        return bytes(self.data[pos + 8:pos + 8 + n]).decode()

    def end(self) -> None:
        """Leave the current object (anything not read is skipped)."""
        self.pos = self._ends.pop()

    def ipos(self) -> tuple:
        """An ``IPosition``."""
        _, version = self.start("IPosition")
        n = self.u32()
        values = tuple(self.i32() for _ in range(n)) if version == 1 else tuple(self.i64() for _ in range(n))
        self.end()
        return values

    def block(self, dtype_code: int):
        """A ``Block<T>`` object: numpy array (list for strings)."""
        self.start("Block")
        values = self.values(dtype_code, self.u32())
        self.end()
        return values

    def array(self, dtype_code: int):
        """An ``Array<T>`` object of element type ``dtype_code``: numpy array in C order (list for strings)."""
        _, version = self.start()
        ndim = self.i32()
        if version < 3:
            for _ in range(ndim):
                self.i32()                                   # origin, discarded by casacore too
        shape = tuple(self.u32() for _ in range(ndim))
        count = self.u32()
        values = self.values(dtype_code, count)
        self.end()
        if dtype_code == TP_STRING:
            if len(shape) <= 1:
                return list(values)
            return np.array(values, dtype=object).reshape(shape[::-1])
        return np.asarray(values).reshape(shape[::-1]) if shape else np.asarray(values)

    def record_desc(self) -> list[RecordField]:
        """A ``RecordDesc``: the list of fields."""
        _, version = self.start("RecordDesc")
        fields = []
        for _ in range(self.i32()):
            name = self.string()
            dtype = self.i32()
            field = RecordField(name=name, dtype=dtype)
            if dtype == TP_RECORD:
                field.sub = self.record_desc()
            elif dtype == TP_TABLE:
                field.table_desc = self.string()
            elif dtype in ARRAY_ELEMENT:
                field.shape = self.ipos()
            if version > 1:
                field.comment = self.string()
            fields.append(field)
        self.end()
        return fields

    def table_record(self) -> dict:
        """A ``TableRecord`` (keyword set) as a dict; nested records are dicts, subtables :class:`TableRef`."""
        kind, _ = self.start()
        if kind != "TableRecord":
            raise AipsIOError(f"old-style keyword set {kind!r} is not supported")
        fields = self.record_desc()
        self.i32()                                           # record type (fixed / variable)
        record = self._record_data(fields)
        self.end()
        record.fields = fields
        return record

    def record(self) -> dict:
        """A plain ``Record`` (as stored inside data-manager headers)."""
        self.start("Record")
        fields = self.record_desc()
        self.i32()
        record = self._record_data(fields, table_record=False)
        self.end()
        record.fields = fields
        return record

    def _record_data(self, fields: list[RecordField], table_record: bool = True) -> Record:
        out = Record()
        for field in fields:
            if field.dtype == TP_RECORD:
                if field.sub:
                    out[field.name] = self._record_data(field.sub, table_record)
                    out[field.name].fields = field.sub
                else:
                    out[field.name] = self.table_record() if table_record else self.record()
            elif field.dtype == TP_TABLE:
                out[field.name] = TableRef(self.string())
            elif field.dtype in ARRAY_ELEMENT:
                out[field.name] = self.array(ARRAY_ELEMENT[field.dtype])
            else:
                out[field.name] = self.scalar(field.dtype)
        return out


class AipsWriter:
    """Writer of an AipsIO stream into memory (``getvalue()`` returns the bytes).

    Parameters
    ----------
    big_endian : bool
        Byte order of the scalars.
    """

    def __init__(self, big_endian: bool = True) -> None:
        self.buffer = bytearray()
        self.big_endian = big_endian
        self._starts: list[int] = []
        self._prefix = _code(big_endian)

    def getvalue(self) -> bytes:
        if self._starts:
            raise AipsIOError("AipsIO object not closed")
        return bytes(self.buffer)

    def _pack(self, fmt: str, value) -> None:
        self.buffer += struct.pack(self._prefix + fmt, value)

    def u8(self, value: int) -> None:
        self._pack("B", value)

    def bool(self, value: bool) -> None:
        self._pack("B", 1 if value else 0)

    def i16(self, value: int) -> None:
        self._pack("h", value)

    def u16(self, value: int) -> None:
        self._pack("H", value)

    def i32(self, value: int) -> None:
        self._pack("i", value)

    def u32(self, value: int) -> None:
        self._pack("I", value)

    def i64(self, value: int) -> None:
        self._pack("q", value)

    def u64(self, value: int) -> None:
        self._pack("Q", value)

    def f32(self, value: float) -> None:
        self._pack("f", value)

    def f64(self, value: float) -> None:
        self._pack("d", value)

    def string(self, value: str) -> None:
        data = str(value).encode("utf-8")
        self.u32(len(data))
        self.buffer += data

    def raw(self, data: bytes) -> None:
        self.buffer += data

    def values(self, dtype_code: int, values) -> None:
        """Consecutive values of a scalar data type (no count)."""
        if dtype_code == TP_STRING:
            for value in values:
                self.string(value)
            return
        array = np.asarray(values).reshape(-1)
        if dtype_code == TP_BOOL:
            self.buffer += np.packbits(array.astype(bool), bitorder="little").tobytes()
            return
        dtype = np.dtype(SCALAR_DTYPES[dtype_code]).newbyteorder(self._prefix)
        self.buffer += array.astype(dtype).tobytes()

    def scalar(self, dtype_code: int, value) -> None:
        if dtype_code == TP_STRING:
            self.string(value)
        elif dtype_code == TP_BOOL:
            self.bool(bool(value))
        else:
            self.values(dtype_code, [value])

    def start(self, kind: str, version: int) -> None:
        """Begin an object (writes the magic word first at the outermost level)."""
        if not self._starts:
            self.u32(MAGIC)
        self._starts.append(len(self.buffer))
        self.u32(MAGIC)                                      # length placeholder, patched by end()
        self.string(kind)
        self.u32(version)

    def end(self) -> None:
        """Close the current object and patch its length."""
        begin = self._starts.pop()
        struct.pack_into(self._prefix + "I", self.buffer, begin, len(self.buffer) - begin)

    def ipos(self, values) -> None:
        values = [int(v) for v in values]
        wide = any(v > 2147483647 for v in values)
        self.start("IPosition", 2 if wide else 1)
        self.u32(len(values))
        for value in values:
            self.i64(value) if wide else self.i32(value)
        self.end()

    def block(self, dtype_code: int, values) -> None:
        """A ``Block<T>`` object."""
        values = list(values) if dtype_code == TP_STRING else np.asarray(values).reshape(-1)
        self.start("Block", 1)
        self.u32(len(values))
        self.values(dtype_code, values)
        self.end()

    def array(self, dtype_code: int, values, name: str = "Array") -> None:
        """An ``Array<T>`` object; ``values`` in C order (numpy) or a flat list of str."""
        if dtype_code == TP_STRING and not isinstance(values, np.ndarray):
            values = list(values)
            shape = (len(values),)
            flat = values
        else:
            array = np.asarray(values)
            shape = array.shape[::-1] if array.ndim else (1,)
            flat = array.reshape(-1).tolist() if dtype_code == TP_STRING else array
        self.start(name, 3)
        self.u32(len(shape))
        for n in shape:
            self.u32(int(n))
        count = int(np.prod(shape)) if shape else 1
        self.u32(count)
        self.values(dtype_code, flat)
        self.end()

    def record_desc(self, fields: list[RecordField]) -> None:
        self.start("RecordDesc", 2)
        self.i32(len(fields))
        for field in fields:
            self.string(field.name)
            self.i32(field.dtype)
            if field.dtype == TP_RECORD:
                self.record_desc(field.sub or [])
            elif field.dtype == TP_TABLE:
                self.string(field.table_desc)
            elif field.dtype in ARRAY_ELEMENT:
                self.ipos(field.shape if field.shape is not None else (-1,))
            self.string(field.comment)
        self.end()

    def table_record(self, record: dict, record_type: int = 1, fields: Optional[list] = None) -> None:
        """A ``TableRecord``; the description is derived from the values unless ``fields`` is given."""
        fields = fields if fields is not None else fields_of(record)
        self.start("TableRecord", 1)
        self.record_desc(fields)
        self.i32(record_type)
        self._record_data(record, fields, table_record=True)
        self.end()

    def record(self, record: dict, record_type: int = 1, fields: Optional[list] = None) -> None:
        """A plain ``Record`` (data-manager headers)."""
        fields = fields if fields is not None else fields_of(record)
        self.start("Record", 1)
        self.record_desc(fields)
        self.i32(record_type)
        self._record_data(record, fields, table_record=False)
        self.end()

    def _record_data(self, record: dict, fields: list[RecordField], table_record: bool) -> None:
        for field in fields:
            value = record[field.name]
            if field.dtype == TP_RECORD:
                if field.sub:
                    self._record_data(value, field.sub, table_record)
                elif table_record:
                    self.table_record(value)
                else:
                    self.record(value)
            elif field.dtype == TP_TABLE:
                self.string(value.name)
            elif field.dtype in ARRAY_ELEMENT:
                element = ARRAY_ELEMENT[field.dtype]
                self.array(element, value, ARRAY_OBJECT_NAMES.get(element, "Array"))
            else:
                self.scalar(field.dtype, value)


def fields_of(record: dict) -> list[RecordField]:
    """The description to write ``record`` with: the one it was read with while its keys are unchanged."""
    fields = getattr(record, "fields", None)
    if fields is not None and [f.name for f in fields] == list(record):
        return fields
    return describe_record(record)


def describe_record(record: dict, table_record: bool = True) -> list[RecordField]:
    """A ``RecordDesc`` for a dict of Python/numpy values (sub-records are written as variable records).

    Python ``int`` -> Int, ``float`` -> Double, ``complex`` -> DComplex, ``bool`` -> Bool, ``str`` -> String,
    numpy scalars and arrays by dtype, a list of str -> Array<String>, :class:`TableRef` -> Table.
    """
    fields = []
    for name, value in record.items():
        if isinstance(value, dict):
            fields.append(RecordField(name, TP_RECORD, sub=[]))
        elif isinstance(value, TableRef):
            fields.append(RecordField(name, TP_TABLE))
        elif isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim > 0):
            array = np.asarray(value)
            empty_list = isinstance(value, (list, tuple)) and not value
            element = TP_STRING if (array.dtype.kind in "USO" or empty_list) else dtype_code_of(array.dtype)
            fields.append(RecordField(name, ELEMENT_ARRAY[element], shape=(-1,)))
        elif isinstance(value, str):
            fields.append(RecordField(name, TP_STRING))
        elif isinstance(value, (bool, np.bool_)):
            fields.append(RecordField(name, TP_BOOL))
        elif isinstance(value, int):
            fields.append(RecordField(name, TP_INT))
        elif isinstance(value, float):
            fields.append(RecordField(name, TP_DOUBLE))
        elif isinstance(value, complex):
            fields.append(RecordField(name, TP_DCOMPLEX))
        elif isinstance(value, np.generic):
            fields.append(RecordField(name, dtype_code_of(np.asarray(value).dtype)))
        else:
            raise TypeError(f"cannot store keyword {name!r} of type {type(value).__name__}")
    return fields


def dtype_code_of(dtype) -> int:
    """The casacore scalar data type of a numpy dtype."""
    dtype = np.dtype(dtype)
    for code, candidate in SCALAR_DTYPES.items():
        if np.dtype(candidate) == dtype:
            return code
    if dtype.kind in "US":
        return TP_STRING
    if dtype == np.dtype(np.uint64):
        return TP_UINT                                       # records have no unsigned 64-bit type
    raise TypeError(f"no casacore data type for numpy dtype {dtype}")
