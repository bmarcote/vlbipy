"""StandardStMan (``table.f<seqnr>``): read, update in place, and write new.

File layout (casacore SSMBase), byte order of the table unless stated:

* bytes 0..511: AipsIO header ``StandardStMan`` (bucket size, number of buckets, free list, index bucket(s),
  index length, last string bucket, number of indices);
* bucket ``b`` at ``512 + b * bucket_size``;
* the index (one ``SSMIndex`` per group of columns sharing buckets: rows per bucket, last row and bucket
  number of every data bucket) is an AipsIO stream stored in the index bucket(s), after an 8-byte big-endian
  prefix, or at ``idx_bucket_offset`` when it shares a bucket;
* a data bucket holds, per column, the values of consecutive rows from the column's offset (the ``SSM`` blob
  in ``table.dat`` gives every column's offset and index number): fixed-size values one after another, bools as
  bits; a variable string is a 12-byte entry ``(bucket, offset, length)`` pointing into the string buckets
  (strings of up to 8 bytes are stored in the entry itself), a fixed-length string ``maxlen`` bytes;
  an indirect array is the int64 offset of the array in ``table.f<seqnr>i``;
* string bucket: big-endian ``[int][used length][deleted][next bucket]`` then the characters; a string may
  continue in the next bucket.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .aipsio import SCALAR_DTYPES, TP_BOOL, TP_INT, TP_STRING, TP_UINT, AipsReader, AipsWriter
from .arrayfile import ArrayFile, ArrayFileWriter
from .tabledat import OPTION_DIRECT, OPTION_FIXED_SHAPE, ColumnDesc, DataManagerEntry, UnsupportedTable

HEADER_SIZE = 512


@dataclass
class SSMIndex:
    """Rows per bucket, last row of each bucket, and the bucket numbers, of one group of columns."""
    rows_per_bucket: int
    ncolumns: int
    last_row: np.ndarray
    bucket: np.ndarray
    free_space: dict


def _value_size(column: ColumnDesc) -> int:
    """External size in bytes of one cell (bools: bits are counted by the caller)."""
    if column.dtype == TP_STRING:
        return column.maxlen if (column.maxlen > 0 and not column.is_array) else 12
    nelem = _nelements(column)
    return nelem * np.dtype(SCALAR_DTYPES[column.dtype]).itemsize


def _nelements(column: ColumnDesc) -> int:
    if column.is_array and is_direct(column):
        return int(np.prod(column.shape)) if column.shape else 1
    return 1


def is_direct(column: ColumnDesc) -> bool:
    """Arrays stored in the bucket itself (option Direct, hence a fixed shape), not in ``table.f<n>i``."""
    return bool(column.option & OPTION_DIRECT)


def has_fixed_shape(column: ColumnDesc) -> bool:
    """A fixed-shape array column (its string cells are then stored without their shape)."""
    return bool(column.option & OPTION_FIXED_SHAPE) and bool(column.shape)


class StandardManager:
    """A StandardStMan of one table: column reads and in-place updates of fixed-size cells."""

    def __init__(self, table_path, manager: DataManagerEntry, columns: list[ColumnDesc], big_endian: bool,
                 nrow: int, writable: bool = False) -> None:
        self.path = Path(table_path)
        self.seqnr = manager.seqnr
        self.columns = {column.name: column for column in columns}
        self.order = [column.name for column in columns]
        self.big_endian = big_endian
        self.prefix = ">" if big_endian else "<"
        self.nrow = nrow
        self.writable = writable
        blob = AipsReader(manager.blob, big_endian=True)
        blob.start("SSM")
        self.name = blob.string()
        self.column_offset = np.asarray(blob.block(TP_UINT), dtype=np.int64)
        self.column_index = np.asarray(blob.block(TP_UINT), dtype=np.int64)
        blob.end()
        self.file = self.path / f"table.f{self.seqnr}"
        self.data = bytearray(self.file.read_bytes()) if writable else self.file.read_bytes()
        self._read_header()
        self._array_file: Optional[ArrayFile] = None
        self._dirty = False

    # -- header and index --
    def _read_header(self) -> None:
        reader = AipsReader(self.data, big_endian=self.big_endian)
        _, version = reader.start("StandardStMan")
        if version >= 3:
            reader.bool()
        self.bucket_size = reader.u32()
        self.nbuckets = reader.u32()
        reader.u32()                                          # persistent cache size
        reader.u32()                                          # free buckets
        reader.i32()                                          # first free bucket
        nidx_buckets = reader.u32()
        first_idx = reader.i32()
        idx_offset = reader.u32() if version >= 2 else 0
        reader.i32()                                          # last string bucket
        index_length = reader.u32()
        nindex = reader.u32()
        reader.end()
        raw = bytearray()
        bucket, remaining = first_idx, index_length
        for _ in range(nidx_buckets):
            start = HEADER_SIZE + bucket * self.bucket_size
            if idx_offset > 0:
                raw += self.data[start + idx_offset:start + idx_offset + index_length]
                break
            take = min(remaining, self.bucket_size - 8)
            raw += self.data[start + 8:start + 8 + take]
            remaining -= take
            bucket = struct.unpack_from(">i", self.data, start + 4)[0]
        index_reader = AipsReader(bytes(raw), big_endian=self.big_endian)
        self.indices = []
        for _ in range(nindex):
            index_reader.start("SSMIndex")
            nused = index_reader.u32()
            rows_per_bucket = index_reader.u32()
            ncolumns = index_reader.i32()
            index_reader.start("SimpleOrderedMap")
            index_reader.i32()
            nfree = index_reader.u32()
            index_reader.u32()
            free = {index_reader.i32(): index_reader.i32() for _ in range(nfree)}
            index_reader.end()
            last_row = np.asarray(index_reader.block(TP_UINT), dtype=np.int64)[:nused]
            buckets = np.asarray(index_reader.block(TP_UINT), dtype=np.int64)[:nused]
            index_reader.end()
            self.indices.append(SSMIndex(rows_per_bucket, ncolumns, last_row, buckets, free))

    def _pieces(self, name: str, start: int, nrow: int) -> list[tuple[int, int, int]]:
        """``(byte offset of the first value, first row in bucket, count)`` per bucket covering the rows."""
        number = self.order.index(name)
        index = self.indices[int(self.column_index[number])]
        offset = int(self.column_offset[number])
        pieces = []
        row, end = start, start + nrow
        while row < end:
            k = int(np.searchsorted(index.last_row, row, side="left"))
            if k >= index.last_row.size:
                raise IndexError(f"row {row} beyond the StandardStMan index of {self.file.name}")
            first = int(index.last_row[k - 1]) + 1 if k > 0 else 0
            count = min(int(index.last_row[k]), end - 1) - row + 1
            bucket_start = HEADER_SIZE + int(index.bucket[k]) * self.bucket_size + offset
            pieces.append((bucket_start, row - first, count))
            row += count
        return pieces

    # -- reads --
    def read(self, column: ColumnDesc, start: int, nrow: int):
        """Rows ``[start, start + nrow)``: numpy array ``(nrow, *cell)``; list of str for string scalars; list of
        per-row arrays (or None) when an indirect column's shapes differ."""
        if column.dtype == TP_STRING:
            return self._read_strings(column, start, nrow)
        if column.is_array and not is_direct(column):
            return self._read_indirect(column, start, nrow)
        nelem = _nelements(column)
        cell = tuple(column.shape[::-1]) if (column.is_array and column.shape) else ()
        if column.dtype == TP_BOOL:
            out = np.empty(nrow * nelem, dtype=bool)
            filled = 0
            for byte_start, first, count in self._pieces(column.name, start, nrow):
                bit0 = first * nelem
                nbits = count * nelem
                chunk = np.frombuffer(self.data, dtype=np.uint8, count=(bit0 + nbits + 7) // 8 - bit0 // 8,
                                      offset=byte_start + bit0 // 8)
                bits = np.unpackbits(chunk, bitorder="little")[bit0 % 8:bit0 % 8 + nbits]
                out[filled:filled + nbits] = bits.astype(bool)
                filled += nbits
            return out.reshape((nrow,) + cell)
        dtype = np.dtype(SCALAR_DTYPES[column.dtype]).newbyteorder(self.prefix)
        size = dtype.itemsize * nelem
        out = np.empty(nrow * nelem, dtype=dtype.newbyteorder("="))
        filled = 0
        for byte_start, first, count in self._pieces(column.name, start, nrow):
            values = np.frombuffer(self.data, dtype=dtype, count=count * nelem, offset=byte_start + first * size)
            out[filled:filled + count * nelem] = values
            filled += count * nelem
        return out.reshape((nrow,) + cell)

    def _entries(self, column: ColumnDesc, start: int, nrow: int) -> list[tuple[int, int, int, bytes]]:
        """The 12-byte ``(bucket, offset, length)`` string entries (with the raw entry bytes) of rows."""
        out = []
        for byte_start, first, count in self._pieces(column.name, start, nrow):
            for k in range(count):
                at = byte_start + (first + k) * 12
                bucket, offset, length = struct.unpack_from(self.prefix + "iii", self.data, at)
                out.append((bucket, offset, length, bytes(self.data[at:at + 8])))
        return out

    def _string_data(self, bucket: int, offset: int, length: int) -> bytes:
        """``length`` bytes from the string buckets, following the chain."""
        out = bytearray()
        while length > 0:
            base = HEADER_SIZE + bucket * self.bucket_size
            used, _, next_bucket = struct.unpack_from(">iii", self.data, base + 4)
            take = min(length, used - offset)
            out += self.data[base + 16 + offset:base + 16 + offset + take]
            length -= take
            bucket, offset = next_bucket, 0
        return bytes(out)

    def _read_strings(self, column: ColumnDesc, start: int, nrow: int):
        if not column.is_array and column.maxlen > 0:
            values = []
            for byte_start, first, count in self._pieces(column.name, start, nrow):
                for k in range(count):
                    at = byte_start + (first + k) * column.maxlen
                    values.append(bytes(self.data[at:at + column.maxlen]).split(b"\0")[0].decode())
            return values
        values = []
        for bucket, offset, length, raw in self._entries(column, start, nrow):
            if not column.is_array:
                if length <= 8:
                    values.append(raw[:length].decode("utf-8", errors="replace"))
                else:
                    values.append(self._string_data(bucket, offset, length).decode("utf-8", errors="replace"))
                continue
            values.append(self._string_array(column, bucket, offset, length))
        if column.is_array:
            shapes = {np.shape(v) for v in values if v is not None}
            if len(shapes) == 1 and all(v is not None for v in values):
                return np.array(values, dtype=object).reshape((nrow,) + shapes.pop())
        return values

    def _string_array(self, column: ColumnDesc, bucket: int, offset: int, length: int):
        """A string array cell from the string buckets (indirect cells carry their shape first)."""
        if length == 0:
            if has_fixed_shape(column):
                return np.full(tuple(column.shape[::-1]), "", dtype=object)
            return None
        data = self._string_data(bucket, offset, length)
        position = 0
        if has_fixed_shape(column):
            shape = tuple(column.shape)
        else:
            ndim = struct.unpack_from(">I", data, 0)[0]
            shape = struct.unpack_from(">" + "i" * ndim, data, 4)
            position = 4 + 4 * ndim + 4                       # shape, then the "filled" flag
        count = int(np.prod(shape)) if shape else 1
        values = []
        for _ in range(count):
            n = struct.unpack_from(">I", data, position)[0]
            values.append(data[position + 4:position + 4 + n].decode("utf-8", errors="replace"))
            position += 4 + n
        return np.array(values, dtype=object).reshape(tuple(shape[::-1]))

    def _read_indirect(self, column: ColumnDesc, start: int, nrow: int):
        if self._array_file is None:
            self._array_file = ArrayFile(self.path / f"table.f{self.seqnr}i", self.big_endian)
        offsets = []
        for byte_start, first, count in self._pieces(column.name, start, nrow):
            offsets.extend(np.frombuffer(self.data, dtype=np.dtype("i8").newbyteorder(self.prefix), count=count,
                                         offset=byte_start + first * 8).tolist())
        cells = [self._array_file.read(int(o), column.dtype) if o else None for o in offsets]
        shapes = {c.shape for c in cells if c is not None}
        if len(shapes) == 1 and all(c is not None for c in cells):
            return np.stack(cells) if cells else np.zeros((0,) + shapes.pop())
        return cells

    # -- in-place writes --
    def write(self, column: ColumnDesc, start: int, values) -> None:
        """Overwrite fixed-size cells of rows ``[start, start + len(values))`` (scalars, direct arrays)."""
        if not self.writable:
            raise PermissionError(f"{self.file.name} was opened read-only")
        if column.dtype == TP_STRING or (column.is_array and not is_direct(column)):
            raise UnsupportedTable(f"in-place write of {column.name} (strings / indirect arrays) is not supported")
        values = np.asarray(values)
        nrow = len(values)
        nelem = _nelements(column)
        flat = values.reshape(nrow * nelem)
        if column.dtype == TP_BOOL:
            done = 0
            for byte_start, first, count in self._pieces(column.name, start, nrow):
                bit0 = first * nelem
                nbits = count * nelem
                lo, hi = byte_start + bit0 // 8, byte_start + (bit0 + nbits + 7) // 8
                bits = np.unpackbits(np.frombuffer(bytes(self.data[lo:hi]), dtype=np.uint8), bitorder="little")
                bits[bit0 % 8:bit0 % 8 + nbits] = flat[done:done + nbits].astype(np.uint8)
                self.data[lo:hi] = np.packbits(bits, bitorder="little").tobytes()
                done += nbits
        else:
            dtype = np.dtype(SCALAR_DTYPES[column.dtype]).newbyteorder(self.prefix)
            size = dtype.itemsize * nelem
            done = 0
            for byte_start, first, count in self._pieces(column.name, start, nrow):
                encoded = flat[done:done + count * nelem].astype(dtype).tobytes()
                at = byte_start + first * size
                self.data[at:at + len(encoded)] = encoded
                done += count * nelem
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.file.write_bytes(bytes(self.data))
            self._dirty = False

    def close(self) -> None:
        self.flush()


# ---------------------------------------------------------------------------
# Writing a new StandardStMan
# ---------------------------------------------------------------------------
def _bits_per_row(column: ColumnDesc) -> int:
    if column.dtype == TP_STRING:
        return 8 * _value_size(column)                         # string arrays always use 12-byte entries
    if column.dtype == TP_BOOL and (not column.is_array or is_direct(column)):
        return _nelements(column)
    if column.is_array and not is_direct(column):
        return 64
    return 8 * _value_size(column)


def write_standard_manager(table_path, seqnr: int, name: str, columns: list[ColumnDesc], values: dict,
                           nrow: int, big_endian: bool, bucket_size: int = 0) -> bytes:
    """Write ``table.f<seqnr>`` (and ``table.f<seqnr>i`` for indirect arrays) holding ``values`` of ``columns``.

    ``values[name]`` is a numpy array ``(nrow, *cell)`` (C order), a list of str for string scalars, or a list
    of per-row arrays/None for indirect array columns. All columns share one index (one bucket layout, as
    casacore lays out a new StandardStMan). Returns the ``SSM`` blob for ``table.dat``.
    """
    prefix = ">" if big_endian else "<"
    bits = [_bits_per_row(column) for column in columns]
    total_bits = sum(bits) if bits else 8
    if bucket_size <= 0:
        bucket_size = max(1024, min(32768, ((max(nrow, 1) * total_bits + 7) // 8 + 1023) // 1024 * 1024))
    rows_per_bucket = max(1, (8 * bucket_size) // max(total_bits, 1))
    while rows_per_bucket > 1 and sum((rows_per_bucket * b + 7) // 8 for b in bits) > bucket_size:
        rows_per_bucket -= 1
    offsets, position = [], 0
    for b in bits:
        offsets.append(position)
        position += (rows_per_bucket * b + 7) // 8
    ndata = max(1, -(-nrow // rows_per_bucket)) if nrow else 0
    buckets = [bytearray(bucket_size) for _ in range(ndata)]
    string_buckets: list[bytearray] = []
    array_file = ArrayFileWriter(big_endian)
    has_indirect = any(c.is_array and not is_direct(c) for c in columns)

    def store_string(data: bytes) -> tuple[int, int]:
        """Append ``data`` to the string buckets; returns (bucket number, offset) of its first byte."""
        capacity = bucket_size - 16
        if not string_buckets or (len(data) > capacity - _used(string_buckets[-1])
                                  and capacity - _used(string_buckets[-1]) < 50):
            string_buckets.append(_new_string_bucket(bucket_size))
        first_bucket = ndata + len(string_buckets) - 1
        first_offset = _used(string_buckets[-1])
        remaining = data
        while remaining:
            bucket = string_buckets[-1]
            used = _used(bucket)
            take = min(len(remaining), capacity - used)
            bucket[16 + used:16 + used + take] = remaining[:take]
            _set_used(bucket, used + take)
            remaining = remaining[take:]
            if remaining:
                string_buckets.append(_new_string_bucket(bucket_size))
                struct.pack_into(">i", bucket, 12, ndata + len(string_buckets) - 1)
        return first_bucket, first_offset

    for column, offset in zip(columns, offsets):
        column_values = values[column.name]
        for b in range(ndata):
            first = b * rows_per_bucket
            count = min(rows_per_bucket, nrow - first)
            target = buckets[b]
            if column.dtype == TP_STRING:
                for k in range(count):
                    cell = column_values[first + k]
                    if column.is_array:
                        data = _encode_string_array(cell, with_shape=not has_fixed_shape(column))
                        if data:
                            sb, so = store_string(data)
                            struct.pack_into(prefix + "iii", target, offset + 12 * k, sb, so, len(data))
                        continue
                    data = str(cell).encode("utf-8")
                    if len(data) <= 8:
                        target[offset + 12 * k:offset + 12 * k + len(data)] = data
                        struct.pack_into(prefix + "i", target, offset + 12 * k + 8, len(data))
                    else:
                        sb, so = store_string(data)
                        struct.pack_into(prefix + "iii", target, offset + 12 * k, sb, so, len(data))
            elif column.is_array and not is_direct(column):
                pointers = []
                for k in range(count):
                    cell = column_values[first + k]
                    pointers.append(0 if cell is None else array_file.append(cell, column.dtype))
                target[offset:offset + 8 * count] = np.asarray(pointers, dtype=prefix + "i8").tobytes()
            elif column.dtype == TP_BOOL:
                flat = np.asarray(column_values[first:first + count], dtype=bool).reshape(-1)
                packed = np.packbits(flat, bitorder="little").tobytes()
                target[offset:offset + len(packed)] = packed
            else:
                dtype = np.dtype(SCALAR_DTYPES[column.dtype]).newbyteorder(prefix)
                encoded = np.asarray(column_values[first:first + count]).astype(dtype).tobytes()
                target[offset:offset + len(encoded)] = encoded

    index = AipsWriter(big_endian=big_endian)
    index.start("SSMIndex", 1)
    index.u32(ndata)
    index.u32(rows_per_bucket)
    index.i32(len(columns))
    index.start("SimpleOrderedMap", 1)
    index.i32(0)
    free_bytes = bucket_size - position
    index.u32(1 if free_bytes > 0 else 0)
    index.u32(1)
    if free_bytes > 0:
        index.i32(position)
        index.i32(free_bytes)
    index.end()
    last_rows = [min((b + 1) * rows_per_bucket, nrow) - 1 for b in range(ndata)]
    index.block(TP_UINT, np.asarray(last_rows, dtype=np.uint32))
    index.block(TP_UINT, np.arange(ndata, dtype=np.uint32))
    index.end()
    index_bytes = index.getvalue()

    all_buckets = buckets + string_buckets
    nidx_buckets = -(-len(index_bytes) // (bucket_size - 8))
    first_idx = len(all_buckets)
    for k in range(nidx_buckets):
        bucket = bytearray(bucket_size)
        nxt = first_idx + k + 1 if k + 1 < nidx_buckets else -1
        struct.pack_into(">ii", bucket, 0, nxt, nxt)
        chunk = index_bytes[k * (bucket_size - 8):(k + 1) * (bucket_size - 8)]
        bucket[8:8 + len(chunk)] = chunk
        all_buckets.append(bucket)

    header = AipsWriter(big_endian=big_endian)
    header.start("StandardStMan", 3)
    header.bool(big_endian)
    header.u32(bucket_size)
    header.u32(len(all_buckets))
    header.u32(2)                                             # persistent cache size
    header.u32(0)
    header.i32(-1)
    header.u32(nidx_buckets)
    header.i32(first_idx)
    header.u32(0)
    header.i32(ndata + len(string_buckets) - 1 if string_buckets else -1)
    header.u32(len(index_bytes))
    header.u32(1)
    header.end()
    head = header.getvalue()
    out = bytearray(HEADER_SIZE)
    out[:len(head)] = head
    for bucket in all_buckets:
        out += bucket
    path = Path(table_path)
    (path / f"table.f{seqnr}").write_bytes(bytes(out))
    if has_indirect:
        (path / f"table.f{seqnr}i").write_bytes(array_file.getvalue())

    blob = AipsWriter(big_endian=True)
    blob.start("SSM", 2)
    blob.string(name)
    blob.block(TP_UINT, np.asarray(offsets, dtype=np.uint32))
    blob.block(TP_UINT, np.zeros(len(columns), dtype=np.uint32))
    blob.end()
    return blob.getvalue()


def _new_string_bucket(bucket_size: int) -> bytearray:
    bucket = bytearray(bucket_size)
    struct.pack_into(">iiii", bucket, 0, 0, 0, bucket_size - 16, -1)
    return bucket


def _used(bucket: bytearray) -> int:
    return struct.unpack_from(">i", bucket, 4)[0]


def _set_used(bucket: bytearray, used: int) -> None:
    struct.pack_into(">ii", bucket, 4, used, len(bucket) - 16 - used)


def _encode_string_array(cell, with_shape: bool) -> bytes:
    """String-array cell as stored in the string buckets (cells of variable shape start with their shape)."""
    if cell is None:
        return b""
    array = np.asarray(cell, dtype=object)
    out = bytearray()
    if with_shape:
        fortran = array.shape[::-1] if array.ndim else (1,)
        out += struct.pack(">I", len(fortran)) + struct.pack(">" + "i" * len(fortran), *fortran) + struct.pack(">I", 1)
    for text in array.reshape(-1):
        data = str(text).encode("utf-8")
        out += struct.pack(">I", len(data)) + data
    return bytes(out)
