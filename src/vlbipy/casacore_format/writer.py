"""Creating casacore tables without casacore: calibration tables, split measurement sets, imported data.

:func:`create_table` writes a complete table directory: ``table.dat``, one StandardStMan (``table.f0``, plus
``table.f0i`` for indirect arrays) holding every small column, one TiledShapeStMan per bulk column named in
``tiled`` (their cells start at zero and are filled afterwards with :meth:`vlbipy.casacore_format.table.Table.putcol`),
``table.info`` and the sync record of ``table.lock``. :func:`copy_table` rewrites an existing table (e.g. an MS
subtable) value by value into a new one. Column descriptions are built with :func:`scalar_column` and
:func:`array_column`; shapes are given in C order (``(nchan, ncorr)``), as everywhere in this package.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .aipsio import SCALAR_DTYPES, TP_BOOL, TP_STRING, TableRef, dtype_code_of
from .ssm import is_direct, write_standard_manager
from .tabledat import (OPTION_DIRECT, OPTION_FIXED_SHAPE, ColumnDesc, DataManagerEntry, LockInfo, TableDat,
                       write_lock_info, write_table_dat)
from .tsm import create_tiled_shape_manager

#: casacore type names (python-casacore's ``valueType``) -> data type code.
VALUE_TYPES = {"boolean": 0, "bool": 0, "uchar": 2, "short": 3, "int": 5, "uint": 6, "float": 7, "double": 8,
               "complex": 9, "dcomplex": 10, "string": 11, "int64": 29}


def _code(value_type) -> int:
    return VALUE_TYPES[value_type] if isinstance(value_type, str) else int(value_type)


def scalar_column(name: str, value_type, *, comment: str = "", keywords: Optional[dict] = None,
                  option: int = 0, maxlen: int = 0) -> ColumnDesc:
    """A scalar column (``value_type`` as ``"int"``, ``"double"``, ... or a data type code)."""
    return ColumnDesc(name=name, dtype=_code(value_type), is_array=False, comment=comment,
                      keywords=dict(keywords or {}), option=option, maxlen=maxlen)


def array_column(name: str, value_type, *, shape: Optional[tuple] = None, ndim: int = -1, comment: str = "",
                 keywords: Optional[dict] = None, direct: bool = False) -> ColumnDesc:
    """An array column; ``shape`` (C order) makes it fixed-shape, ``direct`` stores fixed cells in the bucket."""
    option = 0
    fortran = None
    if shape is not None:
        fortran = tuple(int(n) for n in shape[::-1])
        ndim = len(fortran)
        option |= OPTION_FIXED_SHAPE | (OPTION_DIRECT if direct else 0)
    return ColumnDesc(name=name, dtype=_code(value_type), is_array=True, comment=comment,
                      keywords=dict(keywords or {}), option=option, ndim=ndim, shape=fortran)


def _default_values(column: ColumnDesc, nrow: int):
    """Values of a column nobody filled: zeros / empty strings, undefined cells for variable arrays."""
    if column.dtype == TP_STRING and not column.is_array:
        return [""] * nrow
    if column.is_array and not column.shape:
        return [None] * nrow
    cell = tuple(column.shape[::-1]) if column.is_array else ()
    if column.dtype == TP_STRING:
        return [np.full(cell, "", dtype=object) for _ in range(nrow)]
    return np.zeros((nrow,) + cell, dtype=SCALAR_DTYPES[column.dtype])


def _prepare(column: ColumnDesc, values, nrow: int):
    """Values in the form the StandardStMan writer takes (array, list of str, or list of cells)."""
    if values is None:
        return _default_values(column, nrow)
    if column.dtype == TP_STRING and not column.is_array:
        values = [str(v) for v in values]
    elif column.is_array and not (column.shape and is_direct(column)):
        if isinstance(values, np.ndarray) and values.dtype != object:
            values = [values[i] for i in range(len(values))]
        else:
            values = list(values)
    else:
        values = np.asarray(values, dtype=SCALAR_DTYPES.get(column.dtype, None) if column.dtype != TP_STRING
                            else object)
    if len(values) != nrow:
        raise ValueError(f"column {column.name}: {len(values)} values for {nrow} rows")
    return values


def write_table_info(path, table_type: str = "", sub_type: str = "", readme: str = "") -> None:
    """``table.info``: ``Type = ...`` and ``SubType = ...`` lines, an empty line, then the readme."""
    (Path(path) / "table.info").write_text(f"Type = {table_type}\nSubType = {sub_type}\n\n{readme}")


def create_table(path, columns: list[ColumnDesc], nrow: int, values: Optional[dict] = None, *,
                 keywords: Optional[dict] = None, info: tuple = ("", ""), readme: str = "",
                 tiled: Optional[dict] = None, big_endian: Optional[bool] = None, overwrite: bool = True,
                 manager_name: str = "SSM") -> Path:
    """Write a new table at ``path``.

    Parameters
    ----------
    path : path
        The table directory (removed first when it exists and ``overwrite``).
    columns : list of ColumnDesc
        The columns, in order.
    nrow : int
        Number of rows.
    values : dict
        ``{column: values}`` for the StandardStMan columns (arrays ``(nrow, *cell)``, lists of str, or lists of
        cells / None for variable-shape arrays); missing columns get zeros, empty strings or undefined cells.
    keywords : dict
        Table keywords (a :class:`~.aipsio.TableRef` value declares a subtable, e.g. ``TableRef("././ANTENNA")``).
    info : (str, str)
        ``table.info`` type and subtype.
    tiled : dict
        ``{column: [cell shape of each row (C order) or None]}`` (or one shape for every row): the bulk columns,
        each stored in its own TiledShapeStMan and zero until written with ``Table.putcol``.
    big_endian : bool
        Byte order of the data files (default: this machine's, like casacore).
    manager_name : str
        Name (and column group) of the StandardStMan (CASA calls it ``MSMTAB`` in calibration tables).

    Returns
    -------
    pathlib.Path
    """
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    big_endian = (sys.byteorder == "big") if big_endian is None else bool(big_endian)
    for column in columns:
        if column.is_array and column.shape and (column.option & OPTION_FIXED_SHAPE):
            column.shape_col_def, column.shape_col = True, tuple(column.shape)   # casacore requires it
    values = values or {}
    tiled = tiled or {}
    standard = [c for c in columns if c.name not in tiled]
    managers = []
    seqnr = 0
    if standard:
        for column in standard:
            column.dm_type, column.dm_group, column.seqnr = "StandardStMan", manager_name, seqnr
        blob = write_standard_manager(path, seqnr, manager_name, standard,
                                      {c.name: _prepare(c, values.get(c.name), nrow) for c in standard},
                                      nrow, big_endian)
        managers.append(DataManagerEntry("StandardStMan", seqnr, blob))
        seqnr += 1
    for column in columns:
        if column.name not in tiled:
            continue
        shapes = tiled[column.name]
        if shapes is None or (isinstance(shapes, tuple) and all(isinstance(n, (int, np.integer)) for n in shapes)):
            shapes = [tuple(shapes) if shapes is not None else tuple(column.shape[::-1])] * nrow
        column.dm_type, column.dm_group, column.seqnr = "TiledShapeStMan", f"Tiled{column.name.title()}", seqnr
        column.option &= ~OPTION_DIRECT
        create_tiled_shape_manager(path, seqnr, column, list(shapes), big_endian, column.dm_group)
        managers.append(DataManagerEntry("TiledShapeStMan", seqnr, b""))
        seqnr += 1
    # casacore's description of each tiled column's hypercube (it opens the table without them, but defines them).
    private = {f"Hypercolumn_{c.dm_group}": {"ndim": np.uint32((c.ndim if c.ndim > 0 else len(c.shape or ())) + 1),
                                               "data": [c.name], "coord": [], "id": []}
               for c in columns if c.name in tiled}
    dat = TableDat(nrow=nrow, big_endian=big_endian, columns=list(columns), managers=managers,
                   keywords=dict(keywords or {}), private_keywords=private, seq_count=seqnr)
    write_table_dat(path, dat)
    write_table_info(path, info[0], info[1], readme)
    write_lock_info(path, LockInfo(nrow, len(columns), 1, 1, tuple(1 for _ in managers)))
    for column in columns:
        if column.name in tiled and column.name in values:
            from .table import Table
            with Table(path, readonly=False) as table:
                table.putcol(column.name, values[column.name])
    return path


def copy_table(source, dest, *, keywords: Optional[dict] = None) -> Path:
    """Rewrite table ``source`` value by value as a new StandardStMan table at ``dest``.

    Column descriptions and keywords (subtable references excepted, unless given in ``keywords``) and
    ``table.info`` are kept; every column is stored in one StandardStMan, so what the source used (incremental
    or tiled managers) does not matter. Meant for subtables.
    """
    from .table import Table
    source, dest = Path(source), Path(dest)
    with Table(source) as table:
        nrow = table.nrows()
        columns = []
        values = {}
        for column in table.dat.columns:
            copied = ColumnDesc(name=column.name, dtype=column.dtype, is_array=column.is_array,
                                comment=column.comment, option=column.option, ndim=column.ndim,
                                shape=column.shape, maxlen=column.maxlen, keywords=column.keywords,
                                keyword_fields=column.keyword_fields, class_name=column.class_name,
                                default=column.default, shape_col_def=column.shape_col_def,
                                shape_col=column.shape_col)
            columns.append(copied)
            values[column.name] = table.getcol(column.name) if nrow else None
        own = {k: v for k, v in table.dat.keywords.items() if not isinstance(v, TableRef)}
        info_text = (source / "table.info").read_text() if (source / "table.info").is_file() else ""
    lines = info_text.split("\n")
    table_type = lines[0].split("=", 1)[1].strip() if lines and "=" in lines[0] else ""
    sub_type = lines[1].split("=", 1)[1].strip() if len(lines) > 1 and "=" in lines[1] else ""
    readme = "\n".join(lines[3:]) if len(lines) > 3 else ""
    own.update(keywords or {})
    return create_table(dest, columns, nrow, values, keywords=own, info=(table_type, sub_type), readme=readme)


def numpy_value_type(array) -> int:
    """The casacore data type of a numpy array's elements (strings -> String)."""
    array = np.asarray(array)
    return TP_STRING if array.dtype.kind in "USO" else dtype_code_of(array.dtype)


__all__ = ["create_table", "copy_table", "scalar_column", "array_column", "write_table_info", "TP_BOOL"]


def read_table_values(path) -> tuple:
    """Everything needed to write a table again: ``(columns, values, nrow, keywords, info)``.

    ``columns`` are copies of the column descriptions, ``values`` every column read in full (arrays, lists of str,
    lists of cells), ``keywords`` the raw table keywords (subtables as :class:`~.aipsio.TableRef`).
    """
    import copy

    from .table import Table
    with Table(path) as table:
        nrow = table.nrows()
        columns = [copy.deepcopy(column) for column in table.dat.columns]
        values = {column.name: (table.getcol(column.name) if nrow else None) for column in columns}
        keywords = dict(table.dat.keywords)
        info = table.info()
    return columns, values, nrow, keywords, info


def rewrite_table(path, transform) -> Path:
    """Rewrite the (small) table at ``path`` after ``transform(values, nrow) -> (values, nrow)`` changed its rows.

    Column descriptions, keywords and ``table.info`` are kept; everything is stored in one StandardStMan. Meant
    for subtables (FIELD reduced to one source, SPECTRAL_WINDOW with averaged channels, ...).
    """
    columns, values, nrow, keywords, info = read_table_values(path)
    for column in columns:
        if values[column.name] is None:
            values[column.name] = _default_values(column, nrow)
    values, nrow = transform(values, nrow)
    for column in columns:
        column.shape_col_def, column.shape_col = False, ()
    return create_table(path, columns, nrow, values, keywords=keywords, info=(info["type"], info["subType"]),
                        readme=info["readme"])


def select_rows(values: dict, rows) -> dict:
    """``values`` (as :func:`read_table_values` returns them) restricted to the row indices ``rows``."""
    rows = [int(r) for r in rows]
    out = {}
    for name, column in values.items():
        if isinstance(column, np.ndarray):
            out[name] = column[rows]
        else:
            out[name] = [column[r] for r in rows]
    return out
