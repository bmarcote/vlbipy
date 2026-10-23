"""The tiled storage managers (TiledShapeStMan, TiledColumnStMan, TiledCellStMan): memory-mapped hypercubes.

Files: ``table.f<seqnr>`` is the AipsIO header (always big-endian), ``table.f<seqnr>_TSM<k>`` hold the tiles.
A hypercube has the shape ``(cell axes..., rows)`` in casacore (Fortran) axis order and is cut into tiles of
``tile_shape``; tile ``(t0, t1, ..., tn)`` is bucket ``t0 + n0 * (t1 + n1 * (...))`` at
``file_offset + bucket * bucket_size`` and holds its values in Fortran order (first axis fastest), each data
column of the manager one after the other (bools packed 8 per byte, lowest bit first). Values are in the
table's byte order (``TableDat.big_endian``).

TiledShapeStMan maps row intervals onto its cubes (``row_map``/``cube_map``/``pos_map``): interval ``i`` covers
rows ``(row_map[i-1], row_map[i]]``, stored in cube ``cube_map[i]`` at row position
``pos_map[i] - (row_map[i] - row)``. Cube 0 is a dummy for rows without a defined cell. TiledColumnStMan has a
single cube over all rows; TiledCellStMan one cube per row.

:class:`TiledManager` reads a manager's header and gives row-major numpy views of any row range
(:meth:`TiledManager.read`, :meth:`TiledManager.write`); when the tiles span whole cells the view is a plain
reshape of the memory map, with no copy.
"""
from __future__ import annotations

import mmap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .aipsio import SCALAR_DTYPES, TP_BOOL, TP_UINT, AipsReader, AipsWriter
from .tabledat import ColumnDesc, UnsupportedTable

TILED_TYPES = ("TiledShapeStMan", "TiledColumnStMan", "TiledCellStMan")


@dataclass
class Cube:
    """One hypercube: shape and tile shape in Fortran axis order (last axis = rows), file and offset."""
    shape: tuple
    tile_shape: tuple
    file_seqnr: int
    offset: int
    extensible: bool = False
    values: dict = field(default_factory=dict)

    @property
    def tiles_per_dim(self) -> tuple:
        return tuple(-(-s // t) if t else 0 for s, t in zip(self.shape, self.tile_shape))

    @property
    def tile_size(self) -> int:
        return int(np.prod(self.tile_shape)) if self.tile_shape else 0


def _column_bytes(dtype_code: int, npixel: int) -> int:
    if dtype_code == TP_BOOL:
        return (npixel + 7) // 8
    return npixel * np.dtype(SCALAR_DTYPES[dtype_code]).itemsize


class TiledManager:
    """A tiled storage manager of one table, opened for reading (``writable=False``) or in-place writing.

    Parameters
    ----------
    table_path : path
        The table directory.
    seqnr : int
        The data manager's sequence number (its files are ``table.f<seqnr>*``).
    type_name : str
        ``TiledShapeStMan``, ``TiledColumnStMan`` or ``TiledCellStMan``.
    columns : list of ColumnDesc
        The columns it stores, in table-description order (that is their order inside a tile).
    big_endian : bool
        Byte order of the tile data.
    nrow : int
        Rows in the table.
    """

    def __init__(self, table_path, seqnr: int, type_name: str, columns: list[ColumnDesc], big_endian: bool,
                 nrow: int, writable: bool = False) -> None:
        if type_name not in TILED_TYPES:
            raise UnsupportedTable(f"{type_name} is not a tiled storage manager")
        self.path = Path(table_path)
        self.seqnr = seqnr
        self.type_name = type_name
        self.columns = columns
        self.big_endian = big_endian
        self.nrow = nrow
        self.writable = writable
        self.cubes: list[Cube] = []
        self.files: dict[int, int] = {}
        self.row_map = np.zeros(0, dtype=np.int64)
        self.cube_map = np.zeros(0, dtype=np.int64)
        self.pos_map = np.zeros(0, dtype=np.int64)
        self.default_tile_shape: tuple = ()
        self._maps: dict[int, np.ndarray] = {}
        self._read_header()

    # -- header --
    def _read_header(self) -> None:
        reader = AipsReader((self.path / f"table.f{self.seqnr}").read_bytes(), big_endian=True)
        reader.start(self.type_name)
        if self.type_name == "TiledColumnStMan":
            self.default_tile_shape = reader.ipos()
        elif self.type_name == "TiledCellStMan":
            self.default_tile_shape = reader.ipos()
        _, version = reader.start("TiledStMan")
        big_endian = reader.bool() if version >= 2 else True
        if big_endian != self.big_endian:
            raise UnsupportedTable(f"TSM endian flag of table.f{self.seqnr} differs from the table's")
        reader.u32()                                        # sequence number
        reader.u64() if version >= 3 else reader.u32()      # rows
        ncol = reader.u32()
        for _ in range(ncol):
            reader.i32()                                    # data types
        self.hypercolumn = reader.string()
        reader.u64() if version >= 3 else reader.u32()      # max cache size
        self.ndim = reader.u32()
        nfile = reader.u64() if version >= 3 else reader.u32()
        for index in range(nfile):
            if reader.bool():
                file_version = reader.u32()
                file_seqnr = reader.u32()
                length = reader.u32() if file_version == 1 else reader.u64()
                self.files[file_seqnr] = length
        ncube = reader.u64() if version >= 3 else reader.u32()
        for _ in range(ncube):
            cube_version = reader.i32()
            values = reader.record()
            extensible = reader.bool()
            reader.u32()                                    # ndim
            shape = reader.ipos()
            tile_shape = reader.ipos()
            file_seqnr = reader.i32()
            offset = reader.u32() if cube_version == 1 else reader.i64()
            self.cubes.append(Cube(tuple(shape), tuple(tile_shape), file_seqnr, offset, extensible, values))
        reader.end()
        if self.type_name == "TiledShapeStMan":
            self.default_tile_shape = reader.ipos()
            nused = reader.u32()
            self.row_map = np.asarray(reader.block(TP_UINT), dtype=np.int64)[:nused]
            self.cube_map = np.asarray(reader.block(TP_UINT), dtype=np.int64)[:nused]
            self.pos_map = np.asarray(reader.block(TP_UINT), dtype=np.int64)[:nused]
        reader.end()

    # -- row -> (cube, position) --
    def locate(self, start: int, nrow: int) -> list[tuple[int, int, int, int]]:
        """Split rows ``[start, start + nrow)`` into pieces ``(cube index, cube row, table row, count)``."""
        if start < 0 or start + nrow > self.nrow:
            raise IndexError(f"rows {start}..{start + nrow} outside the table ({self.nrow} rows)")
        if self.type_name == "TiledColumnStMan":
            return [(0, start, start, nrow)]
        if self.type_name == "TiledCellStMan":
            return [(row, 0, row, 1) for row in range(start, start + nrow)]
        pieces = []
        row, end = start, start + nrow
        while row < end:
            index = int(np.searchsorted(self.row_map, row, side="left"))
            if index >= self.row_map.size:
                raise ValueError(f"row {row} of table.f{self.seqnr} has no defined cell")
            last = int(self.row_map[index])
            count = min(last, end - 1) - row + 1
            position = int(self.pos_map[index]) - (last - row)
            pieces.append((int(self.cube_map[index]), position, row, count))
            row += count
        return pieces

    def cell_shape(self, row: int) -> tuple:
        """The cell shape of ``row`` in row-major (C) order, e.g. ``(nchan, ncorr)``."""
        cube_index, _, _, _ = self.locate(row, 1)[0]
        return tuple(self.cubes[cube_index].shape[:-1][::-1])

    # -- data access --
    def _map(self, file_seqnr: int) -> np.ndarray:
        """The whole ``_TSM<k>`` file as a uint8 memory map (read-only unless opened writable)."""
        if file_seqnr not in self._maps:
            path = self.path / f"table.f{self.seqnr}_TSM{file_seqnr}"
            if path.stat().st_size == 0:
                self._maps[file_seqnr] = np.zeros(0, dtype=np.uint8)
            else:
                with open(path, "r+b" if self.writable else "rb") as handle:
                    buffer = mmap.mmap(handle.fileno(), 0,
                                       access=mmap.ACCESS_WRITE if self.writable else mmap.ACCESS_READ)
                self._maps[file_seqnr] = np.frombuffer(buffer, dtype=np.uint8)
        return self._maps[file_seqnr]

    def _value_dtype(self, column: ColumnDesc) -> np.dtype:
        dtype = np.dtype(SCALAR_DTYPES[column.dtype])
        return dtype.newbyteorder(">" if self.big_endian else "<")

    def _layout(self, cube: Cube) -> tuple[int, dict]:
        """Bucket size of a cube's tiles and the byte offset of each column inside a bucket."""
        offsets, size = {}, 0
        for column in self.columns:
            offsets[column.name] = size
            size += _column_bytes(column.dtype, cube.tile_size)
        return size, offsets

    def _column_chunks(self, cube: Cube, column: ColumnDesc, tile_start: int, tile_stop: int) -> np.ndarray:
        """The bytes of one column in the buckets of row tiles ``tile_start..tile_stop``: ``(ntiles, nbytes)``."""
        tiles = cube.tiles_per_dim
        per_row_tile = int(np.prod(tiles[:-1])) if len(tiles) > 1 else 1
        bucket_size, offsets = self._layout(cube)
        first, count = tile_start * per_row_tile, (tile_stop - tile_start) * per_row_tile
        begin = cube.offset + first * bucket_size
        buckets = self._map(cube.file_seqnr)[begin:begin + count * bucket_size].reshape(count, bucket_size)
        offset = offsets[column.name]
        return buckets[:, offset:offset + _column_bytes(column.dtype, cube.tile_size)]

    def _grid(self, cube: Cube, column: ColumnDesc, tile_start: int, tile_stop: int) -> np.ndarray:
        """Values of row tiles ``tile_start..tile_stop``: ``(row tiles, cell tiles C order..., tile axes C order)``
        (a view of the memory map unless the column shares its buckets or is boolean)."""
        tiles = cube.tiles_per_dim
        grid_shape = (tile_stop - tile_start,) + tuple(tiles[:-1][::-1])
        tile_c = tuple(cube.tile_shape[::-1])
        chunk = self._column_chunks(cube, column, tile_start, tile_stop)
        if column.dtype == TP_BOOL:
            bits = np.unpackbits(chunk, axis=1, bitorder="little")[:, :cube.tile_size]
            return bits.astype(bool).reshape(grid_shape + tile_c)
        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk)
        return chunk.reshape(-1).view(self._value_dtype(column)).reshape(grid_shape + tile_c)

    @staticmethod
    def _untile_axes(ncell: int) -> list[int]:
        """Transpose taking ``(rt, ct_{n-1}..ct_0, Tr, T_{n-1}..T_0)`` to ``(rt, Tr, ct_{n-1}, T_{n-1}, ..., ct_0, T_0)``."""
        axes = [0, ncell + 1]
        for k in range(ncell):
            axes += [1 + k, ncell + 2 + k]
        return axes

    def _row_bytes(self, cube: Cube, column: ColumnDesc) -> int:
        """Bytes per row when a bool column's tiles hold whole cells and each row starts on a byte, else 0."""
        cell = cube.shape[:-1]
        if column.dtype != TP_BOOL or any(t < s for t, s in zip(cube.tile_shape[:-1], cell)):
            return 0
        bits = int(np.prod(cube.tile_shape[:-1])) if len(cube.tile_shape) > 1 else 1
        return bits // 8 if bits % 8 == 0 else 0

    def read(self, column: ColumnDesc, start: int, nrow: int, copy: bool = True) -> np.ndarray:
        """Rows ``[start, start + nrow)`` of a column, row-major ``(nrow, *cell)`` in native byte order.

        Every row of the range must have the same cell shape. ``copy=False`` returns a read-only view of the
        memory map where that is possible (whole-cell tiles, native byte order), which costs nothing until
        it is used.
        """
        pieces = self.locate(start, nrow)
        out = None
        for cube_index, position, row, count in pieces:
            block = self._read_cube_rows(self.cubes[cube_index], column, position, count, copy)
            if len(pieces) == 1:
                return block
            if out is None:
                out = np.empty((nrow,) + block.shape[1:], dtype=block.dtype)
            elif block.shape[1:] != out.shape[1:]:
                raise ValueError(f"column {column.name}: rows {start}..{start + nrow} have different cell shapes")
            out[row - start:row - start + count] = block
        return out

    def _read_cube_rows(self, cube: Cube, column: ColumnDesc, position: int, count: int, copy: bool) -> np.ndarray:
        row_tile = cube.tile_shape[-1]
        t0, t1 = position // row_tile, -(-(position + count) // row_tile)
        skip = position - t0 * row_tile
        cell = tuple(cube.shape[:-1][::-1])
        tile_c = tuple(cube.tile_shape[::-1])
        row_bytes = self._row_bytes(cube, column)
        if row_bytes:
            chunk = self._column_chunks(cube, column, t0, t1)
            rows = chunk.reshape(-1, row_bytes)[skip:skip + count]
            bits = np.unpackbits(rows, axis=1, bitorder="little").astype(bool)
            return bits.reshape((count,) + tile_c[1:])[(slice(None),) + tuple(slice(0, n) for n in cell)]
        grid = self._grid(cube, column, t0, t1)
        if all(t >= s for t, s in zip(tile_c[1:], cell)):
            values = grid.reshape((grid.shape[0] * tile_c[0],) + tile_c[1:])
        else:
            grid_cells = grid.shape[1:1 + len(cell)]
            values = grid.transpose(self._untile_axes(len(cell))).reshape(
                (grid.shape[0] * tile_c[0],) + tuple(g * t for g, t in zip(grid_cells, tile_c[1:])))
        values = values[(slice(skip, skip + count),) + tuple(slice(0, n) for n in cell)]
        if not values.dtype.isnative:
            return values.astype(values.dtype.newbyteorder("="))
        return np.array(values) if copy else values

    def write(self, column: ColumnDesc, start: int, values: np.ndarray) -> None:
        """Overwrite rows ``[start, start + len(values))`` of a column whose cells already exist (in place).

        Only the bytes of those rows are written wherever the tiling allows (it does for MS columns), so
        processes writing different rows of one table never overwrite each other's values.
        """
        if not self.writable:
            raise PermissionError(f"table.f{self.seqnr} was opened read-only")
        values = np.asarray(values)
        for cube_index, position, row, count in self.locate(start, len(values)):
            self._write_cube_rows(self.cubes[cube_index], column, position, values[row - start:row - start + count])

    def _write_cube_rows(self, cube: Cube, column: ColumnDesc, position: int, values: np.ndarray) -> None:
        count = len(values)
        cell = tuple(cube.shape[:-1][::-1])
        if values.shape[1:] != cell:
            raise ValueError(f"column {column.name}: cells of shape {values.shape[1:]} written to cells of {cell}")
        row_tile = cube.tile_shape[-1]
        t0, t1 = position // row_tile, -(-(position + count) // row_tile)
        skip = position - t0 * row_tile
        tile_c = tuple(cube.tile_shape[::-1])
        whole_cells = all(t >= s for t, s in zip(tile_c[1:], cell))
        row_bytes = self._row_bytes(cube, column)
        if row_bytes:
            padded = np.zeros((count,) + tile_c[1:], dtype=bool)
            padded[(slice(None),) + tuple(slice(0, n) for n in cell)] = values      # tile padding stays 0
            chunk = self._column_chunks(cube, column, t0, t1)
            rows = chunk.reshape(-1, row_bytes)
            rows[skip:skip + count] = np.packbits(padded.reshape(count, -1), axis=1, bitorder="little")
            return
        bucket_size, _ = self._layout(cube)
        if column.dtype != TP_BOOL and whole_cells and bucket_size == _column_bytes(column.dtype, cube.tile_size):
            grid = self._grid(cube, column, t0, t1)
            target = grid.reshape((grid.shape[0] * tile_c[0],) + tile_c[1:])
            target[(slice(skip, skip + count),) + tuple(slice(0, n) for n in cell)] = values
            return
        self._write_tiles(cube, column, t0, t1, skip, values, cell, tile_c)

    def _write_tiles(self, cube: Cube, column: ColumnDesc, t0: int, t1: int, skip: int, values: np.ndarray,
                     cell: tuple, tile_c: tuple) -> None:
        """Read-modify-write whole tiles (cells split over several tiles, shared buckets, unaligned bools)."""
        grid = self._grid(cube, column, t0, t1)
        grid_cells = grid.shape[1:1 + len(cell)]
        axes = self._untile_axes(len(cell))
        region = grid.transpose(axes).reshape(((t1 - t0) * tile_c[0],)
                                              + tuple(g * t for g, t in zip(grid_cells, tile_c[1:]))).copy()
        region[(slice(skip, skip + len(values)),) + tuple(slice(0, n) for n in cell)] = values
        split = region.reshape((t1 - t0, tile_c[0]) + tuple(x for g, t in zip(grid_cells, tile_c[1:]) for x in (g, t)))
        tiles = split.transpose(np.argsort(axes)).reshape(-1, cube.tile_size)
        chunk = self._column_chunks(cube, column, t0, t1)
        if column.dtype == TP_BOOL:
            chunk[:] = np.packbits(tiles.astype(np.uint8), axis=1, bitorder="little")[:, :chunk.shape[1]]
        else:
            chunk[:] = tiles.astype(self._value_dtype(column)).view(np.uint8).reshape(len(tiles), -1)

    def flush(self) -> None:
        """Write memory-mapped changes to disk."""
        for array in self._maps.values():
            base = array.base
            if isinstance(base, mmap.mmap):
                base.flush()

    def close(self) -> None:
        """Release the memory maps (views handed out keep their own reference)."""
        self.flush()
        self._maps.clear()


def write_header(table_path, seqnr: int, type_name: str, columns: list[ColumnDesc], big_endian: bool, nrow: int,
                 cubes: list[Cube], files: dict, hypercolumn: str, default_tile_shape: tuple,
                 row_map=(), cube_map=(), pos_map=()) -> None:
    """Write the ``table.f<seqnr>`` header of a tiled storage manager (as casacore's ``flush`` does)."""
    writer = AipsWriter(big_endian=True)
    writer.start(type_name, 1)
    if type_name in ("TiledColumnStMan", "TiledCellStMan"):
        writer.ipos(default_tile_shape)
    if big_endian:
        writer.start("TiledStMan", 1)
    else:
        writer.start("TiledStMan", 2)
        writer.bool(False)
    writer.u32(seqnr)
    writer.u32(nrow)
    writer.u32(len(columns))
    for column in columns:
        writer.i32(column.dtype)
    writer.string(hypercolumn)
    writer.u32(0)                                           # persistent max cache size
    ndim = len(cubes[-1].shape) if cubes and cubes[-1].shape else (len(default_tile_shape) or 1)
    writer.u32(ndim)
    nfile = max(files) + 1 if files else 0
    writer.u32(nfile)
    for index in range(nfile):
        if index in files:
            writer.bool(True)
            writer.u32(1)
            writer.u32(index)
            writer.u32(int(files[index]))
        else:
            writer.bool(False)
    writer.u32(len(cubes))
    for cube in cubes:
        writer.i32(1)
        writer.record(cube.values or {})
        writer.bool(cube.extensible)
        writer.u32(len(cube.shape) if cube.shape else 0)
        writer.ipos(cube.shape)
        writer.ipos(cube.tile_shape)
        writer.i32(cube.file_seqnr)
        writer.u32(int(cube.offset))
    writer.end()
    if type_name == "TiledShapeStMan":
        writer.ipos(default_tile_shape)
        writer.u32(len(row_map))
        writer.block(TP_UINT, np.asarray(row_map, dtype=np.uint32))
        writer.block(TP_UINT, np.asarray(cube_map, dtype=np.uint32))
        writer.block(TP_UINT, np.asarray(pos_map, dtype=np.uint32))
    writer.end()
    (Path(table_path) / f"table.f{seqnr}").write_bytes(writer.getvalue())


def default_tile_rows(cell_fortran: tuple, dtype_code: int, nrow: int, target_bytes: int = 1048576) -> int:
    """Rows per tile for whole-cell tiles of about ``target_bytes`` (at least 1, at most ``nrow``)."""
    nelem = int(np.prod(cell_fortran)) if cell_fortran else 1
    nbytes = nelem / 8.0 if dtype_code == TP_BOOL else nelem * np.dtype(SCALAR_DTYPES[dtype_code]).itemsize
    return int(max(1, min(max(nrow, 1), target_bytes // max(nbytes, 1))))


def create_tiled_shape_manager(table_path, seqnr: int, column: ColumnDesc, row_shapes: list, big_endian: bool,
                               hypercolumn: str) -> None:
    """Write a new TiledShapeStMan (header and zero-filled tile files) for one column.

    ``row_shapes[i]`` is the C-order cell shape of row ``i`` (None: the cell stays undefined). Rows of one shape
    share a hypercube (cube ``k`` lives in file ``table.f<seqnr>_TSM<k>``); the row map follows casacore's
    (``row_map[i]`` = last row of interval ``i``, ``pos_map[i]`` = its position in cube ``cube_map[i]``).
    """
    nrow = len(row_shapes)
    shapes: list[tuple] = []
    counts: list[int] = []
    row_map, cube_map, pos_map = [], [], []
    for row, shape in enumerate(row_shapes):
        if shape is None:
            continue
        fortran = tuple(int(n) for n in shape[::-1])
        if fortran not in shapes:
            shapes.append(fortran)
            counts.append(0)
        cube = shapes.index(fortran) + 1
        position = counts[cube - 1]
        counts[cube - 1] += 1
        if row_map and cube_map[-1] == cube and pos_map[-1] + 1 == position and row_map[-1] + 1 == row:
            row_map[-1] += 1
            pos_map[-1] += 1
        else:
            row_map.append(row)
            cube_map.append(cube)
            pos_map.append(position)
    cubes = [Cube((), (), -1, 0)]
    files = {}
    for index, (fortran, count) in enumerate(zip(shapes, counts), start=1):
        rows_per_tile = default_tile_rows(fortran, column.dtype, count)
        tile = fortran + (rows_per_tile,)
        cube = Cube(fortran + (count,), tile, index, 0, True)
        cubes.append(cube)
        ntiles = int(np.prod(cube.tiles_per_dim))
        length = ntiles * _column_bytes(column.dtype, cube.tile_size)
        with open(Path(table_path) / f"table.f{seqnr}_TSM{index}", "wb") as handle:
            handle.truncate(length)
        files[index] = length
    # Without a defined cell the header still declares the rank casacore checks (cell ndim + the row axis).
    default_tile = cubes[1].tile_shape if len(cubes) > 1 else (1,) * (max(int(column.ndim), 0) + 1)
    write_header(table_path, seqnr, "TiledShapeStMan", [column], big_endian, nrow, cubes, files, hypercolumn,
                 default_tile, row_map, cube_map, pos_map)
