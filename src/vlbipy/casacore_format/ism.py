"""IncrementalStMan (``table.f<seqnr>``): read only (values are stored once per change, so the MS index columns
TIME, SCAN_NUMBER, FIELD_ID, ... cost a few kilobytes).

File layout (casacore ISMBase/ISMBucket/ISMIndex), byte order of the table: an AipsIO header
``IncrementalStMan`` at offset 0 (bucket size, number of buckets, ...), buckets from offset 512, and the
``ISMIndex`` (first row and bucket number of every bucket) as AipsIO right after the last bucket. A bucket is
``[uint index offset (top bits: 64-bit row numbers)][data ...]`` followed, per column, by ``[uint n]``,
``n`` row numbers relative to the bucket's first row (uint32 or uint64) and ``n`` data offsets: the value at
data offset ``off`` holds from that row until the next listed one. Values are fixed size (bools as bits; arrays
of fixed shape as all their elements) or, for strings, ``[uint total length][chars]``; indirect arrays store
the int64 offset of the array in ``table.f<seqnr>i``.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from .aipsio import SCALAR_DTYPES, TP_BOOL, TP_INT64, TP_STRING, TP_UINT, AipsReader
from .arrayfile import ArrayFile
from .ssm import is_direct
from .tabledat import ColumnDesc, DataManagerEntry, UnsupportedTable

HEADER_SIZE = 512


class IncrementalManager:
    """An IncrementalStMan of one table, read-only."""

    def __init__(self, table_path, manager: DataManagerEntry, columns: list[ColumnDesc], big_endian: bool,
                 nrow: int) -> None:
        self.path = Path(table_path)
        self.seqnr = manager.seqnr
        self.order = [column.name for column in columns]
        self.big_endian = big_endian
        self.prefix = ">" if big_endian else "<"
        self.nrow = nrow
        self.file = self.path / f"table.f{self.seqnr}"
        self.data = self.file.read_bytes()
        reader = AipsReader(self.data, big_endian=big_endian)
        _, version = reader.start("IncrementalStMan")
        if version >= 5:
            reader.bool()
        self.bucket_size = reader.u32()
        self.nbuckets = reader.u32()
        reader.end()
        index = AipsReader(self.data, big_endian=big_endian, offset=HEADER_SIZE + self.nbuckets * self.bucket_size)
        _, index_version = index.start("ISMIndex")
        nused = index.u32()
        rows = index.block(TP_INT64) if index_version > 1 else index.block(TP_UINT)
        self.bucket_rows = np.asarray(rows, dtype=np.int64)[:nused + 1]
        self.bucket_numbers = np.asarray(index.block(TP_UINT), dtype=np.int64)[:nused]
        index.end()
        self._buckets: dict[int, tuple] = {}
        self._array_file: Optional[ArrayFile] = None

    def _bucket(self, number: int) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
        """``(data start, [(rows, offsets) per column])`` of bucket ``number``."""
        if number not in self._buckets:
            base = HEADER_SIZE + number * self.bucket_size
            raw = struct.unpack_from(self.prefix + "I", self.data, base)[0]
            wide = (raw & 0xF0000000) != 0
            offset = base + (raw & 0x0FFFFFFF)
            columns = []
            for _ in self.order:
                n = struct.unpack_from(self.prefix + "I", self.data, offset)[0]
                offset += 4
                row_dtype = np.dtype("u8" if wide else "u4").newbyteorder(self.prefix)
                rows = np.frombuffer(self.data, dtype=row_dtype, count=n, offset=offset).astype(np.int64)
                offset += n * row_dtype.itemsize
                offs = np.frombuffer(self.data, dtype=np.dtype("u4").newbyteorder(self.prefix), count=n,
                                     offset=offset).astype(np.int64)
                offset += 4 * n
                columns.append((rows, offs))
            self._buckets[number] = (base + 4, columns)
        return self._buckets[number]

    def _value(self, column: ColumnDesc, at: int):
        """The value stored at absolute byte offset ``at``."""
        if column.dtype == TP_STRING:
            total = struct.unpack_from(self.prefix + "I", self.data, at)[0]
            if column.is_array:
                raise UnsupportedTable(f"IncrementalStMan string array column {column.name}")
            return self.data[at + 4:at + total].decode("utf-8", errors="replace")
        if column.is_array and not is_direct(column):
            pointer = struct.unpack_from(self.prefix + "q", self.data, at)[0]
            if not pointer:
                return None
            if self._array_file is None:
                self._array_file = ArrayFile(self.path / f"table.f{self.seqnr}i", self.big_endian)
            return self._array_file.read(int(pointer), column.dtype)
        nelem = int(np.prod(column.shape)) if (column.is_array and column.shape) else 1
        cell = tuple(column.shape[::-1]) if (column.is_array and column.shape) else ()
        if column.dtype == TP_BOOL:
            bits = np.frombuffer(self.data, dtype=np.uint8, count=(nelem + 7) // 8, offset=at)
            values = np.unpackbits(bits, bitorder="little")[:nelem].astype(bool)
        else:
            dtype = np.dtype(SCALAR_DTYPES[column.dtype]).newbyteorder(self.prefix)
            values = np.frombuffer(self.data, dtype=dtype, count=nelem, offset=at).astype(dtype.newbyteorder("="))
        return values.reshape(cell) if cell else values[0]

    def read(self, column: ColumnDesc, start: int, nrow: int):
        """Rows ``[start, start + nrow)``: numpy array, list of str (strings), or list of cells (indirect)."""
        number = self.order.index(column.name)
        end = start + nrow
        runs = []                                             # (first row, last row, value)
        k = int(np.searchsorted(self.bucket_rows, start, side="right")) - 1
        while k < self.bucket_numbers.size and self.bucket_rows[k] < end:
            bucket_first, bucket_end = int(self.bucket_rows[k]), int(self.bucket_rows[k + 1])
            data_start, columns = self._bucket(int(self.bucket_numbers[k]))
            rows, offs = columns[number]
            for j in range(rows.size):
                first = bucket_first + int(rows[j])
                last = bucket_first + int(rows[j + 1]) - 1 if j + 1 < rows.size else bucket_end - 1
                if last < start or first >= end:
                    continue
                runs.append((max(first, start), min(last, end - 1), data_start + int(offs[j])))
            k += 1
        if column.dtype == TP_STRING or (column.is_array and not is_direct(column)):
            out = []
            for first, last, at in runs:
                value = self._value(column, at)
                out.extend([value] * (last - first + 1))
            if column.is_array:
                shapes = {np.shape(v) for v in out if v is not None}
                if len(shapes) == 1 and all(v is not None for v in out):
                    return np.stack(out)
            return out
        values = [self._value(column, at) for _, _, at in runs]
        counts = [last - first + 1 for first, last, _ in runs]
        if not values:
            dtype = SCALAR_DTYPES[column.dtype]
            cell = tuple(column.shape[::-1]) if (column.is_array and column.shape) else ()
            return np.zeros((0,) + cell, dtype=dtype)
        stacked = np.asarray(values)
        return np.repeat(stacked, counts, axis=0)

    def close(self) -> None:
        self._buckets.clear()
