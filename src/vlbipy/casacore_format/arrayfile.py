"""StManArrayFile (``table.f<seqnr>i``): the indirect arrays of StandardStMan and IncrementalStMan.

Layout (casacore StManArrayFile), in the table's byte order: a 16-byte header ``[uint version][int64 length]
[int 0]``, then arrays at 8-byte aligned offsets, each ``[uint refcount (version 1)][uint ndim][int shape...]``
followed by the values (bools as bits; strings as uint offsets to ``[uint length][chars]`` records placed
further on). A row of an indirect column stores the offset of its array, 0 when the cell is undefined.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from .aipsio import SCALAR_DTYPES, TP_BOOL, TP_STRING


class ArrayFile:
    """Reader/appender of an indirect-array file.

    Parameters
    ----------
    path : path
        The ``table.f<seqnr>i`` file.
    big_endian : bool
        The table's byte order.
    """

    def __init__(self, path, big_endian: bool) -> None:
        self.path = Path(path)
        self.big_endian = big_endian
        self.prefix = ">" if big_endian else "<"
        self.data = self.path.read_bytes() if self.path.is_file() else b""
        self.version = struct.unpack_from(self.prefix + "I", self.data, 0)[0] if len(self.data) >= 4 else 1

    def shape(self, offset: int) -> tuple:
        """The shape of the array at ``offset`` in casacore (Fortran) axis order."""
        position = offset + (4 if self.version > 0 else 0)
        ndim = struct.unpack_from(self.prefix + "I", self.data, position)[0]
        return struct.unpack_from(self.prefix + "i" * ndim, self.data, position + 4)

    def read(self, offset: int, dtype_code: int):
        """The array at ``offset``, numpy in C order (shape reversed); list/object array for strings."""
        shape = self.shape(offset)
        position = offset + (4 if self.version > 0 else 0) + 4 + 4 * len(shape)
        count = int(np.prod(shape)) if shape else 1
        c_shape = tuple(shape[::-1])
        if dtype_code == TP_BOOL:
            bits = np.frombuffer(self.data, dtype=np.uint8, count=(count + 7) // 8, offset=position)
            return np.unpackbits(bits, bitorder="little")[:count].astype(bool).reshape(c_shape)
        if dtype_code == TP_STRING:
            pointers = struct.unpack_from(self.prefix + "I" * count, self.data, position)
            values = []
            for pointer in pointers:
                if pointer == 0:
                    values.append("")
                    continue
                n = struct.unpack_from(self.prefix + "I", self.data, pointer)[0]
                values.append(self.data[pointer + 4:pointer + 4 + n].decode("utf-8", errors="replace"))
            return np.array(values, dtype=object).reshape(c_shape)
        dtype = np.dtype(SCALAR_DTYPES[dtype_code]).newbyteorder(self.prefix)
        values = np.frombuffer(self.data, dtype=dtype, count=count, offset=position)
        return values.astype(dtype.newbyteorder("=")).reshape(c_shape)


class ArrayFileWriter:
    """Builds a new ``table.f<seqnr>i`` file in memory; :meth:`append` returns each array's offset."""

    def __init__(self, big_endian: bool) -> None:
        self.prefix = ">" if big_endian else "<"
        self.buffer = bytearray(struct.pack(self.prefix + "Iqi", 1, 16, 0))

    def append(self, values, dtype_code: int, shape: Optional[tuple] = None) -> int:
        """Store one array (``values`` in C order; ``shape`` in C order defaults to its own) and return its
        offset."""
        if dtype_code == TP_STRING:
            array = np.asarray(values, dtype=object)
        else:
            array = np.asarray(values)
        c_shape = tuple(shape) if shape is not None else array.shape
        fortran = tuple(c_shape[::-1])
        while len(self.buffer) % 8:
            self.buffer += b"\0"
        offset = len(self.buffer)
        self.buffer += struct.pack(self.prefix + "II", 1, len(fortran))
        self.buffer += struct.pack(self.prefix + "i" * len(fortran), *fortran)
        flat = array.reshape(-1)
        if dtype_code == TP_BOOL:
            self.buffer += np.packbits(flat.astype(bool), bitorder="little").tobytes()
        elif dtype_code == TP_STRING:
            table_at = len(self.buffer)
            self.buffer += b"\0" * (4 * flat.size)
            pointers = []
            for text in flat:
                data = str(text).encode("utf-8")
                pointers.append(len(self.buffer))
                self.buffer += struct.pack(self.prefix + "I", len(data)) + data
            struct.pack_into(self.prefix + "I" * flat.size, self.buffer, table_at, *pointers)
        else:
            dtype = np.dtype(SCALAR_DTYPES[dtype_code]).newbyteorder(self.prefix)
            self.buffer += flat.astype(dtype).tobytes()
        return offset

    def getvalue(self) -> bytes:
        """The file contents (the header's length field is set to the current size)."""
        out = bytearray(self.buffer)
        struct.pack_into(self.prefix + "q", out, 4, len(out))
        return bytes(out)
