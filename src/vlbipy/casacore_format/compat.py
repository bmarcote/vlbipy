"""A drop-in for ``casatools.table`` on the native table layer, for code written against casatools.

Same conventions as casatools: arrays come back with their axes reversed (``(ncorr, nchan, nrow)``) and promoted
(int -> int64, float -> float64, complex -> complex128), strings as numpy ``<U`` arrays, subtables in keywords
as ``"Table: <absolute path>"``, ``open`` returning True. Values given to ``putcol``/``putcell`` are in the same
reversed order.

Calls this module does not implement natively (TaQL queries, adding or removing rows, copies, ...) are handed to
a real ``casatools.table`` opened on the same table at that moment: the native handle is closed first and every
later call goes to casatools, so mixing the two can never show different contents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .table import Table
from .tabledat import UnsupportedTable

#: Promotion casatools applies to what it returns.
_PROMOTE = {"i": np.int64, "u": np.int64, "f": np.float64, "c": np.complex128}


def _promote(values):
    """casatools' types for a value read natively."""
    if isinstance(values, list):
        if not values or isinstance(values[0], str):
            return np.array(values, dtype=str)
        return values
    array = np.asarray(values)
    if array.dtype == object:
        return array.astype(str)
    target = _PROMOTE.get(array.dtype.kind)
    return array.astype(target) if target is not None and array.dtype != target else array


def _reverse(array: np.ndarray) -> np.ndarray:
    """Reverse all axes (row-major -> casatools' Fortran view)."""
    return np.ascontiguousarray(np.transpose(array)) if array.ndim > 1 else array


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    return value


class table:  # noqa: N801 - mirrors casatools.table
    """casatools-compatible table tool backed by :class:`vlbipy.casacore_format.table.Table`."""

    def __init__(self, tablename: str = "", nomodify: bool = True, **_) -> None:
        self._table: Optional[Table] = None
        self._casa = None
        self._path: Optional[Path] = None
        self._nomodify = True
        if tablename:
            self.open(tablename, nomodify=nomodify)

    # -- life cycle --
    def open(self, tablename: str = "", lockoptions=None, nomodify: bool = True) -> bool:
        """Open a table (read-only unless ``nomodify=False``)."""
        self.close()
        self._path = Path(str(tablename))
        self._nomodify = bool(nomodify)
        if not (self._path / "table.dat").is_file():
            raise RuntimeError(f"Table {self._path} does not exist")          # casatools raises RuntimeError
        try:
            self._table = Table(self._path, readonly=self._nomodify)
        except UnsupportedTable:
            self._promote_to_casatools()
        return True

    def close(self) -> bool:
        if self._table is not None:
            self._table.close()
            self._table = None
        if self._casa is not None:
            self._casa.close()
            self._casa = None
        return True

    def done(self) -> bool:
        return self.close()

    def isopened(self) -> bool:
        return self._table is not None or self._casa is not None

    def flush(self) -> bool:
        if self._table is not None:
            self._table.flush()
        elif self._casa is not None:
            self._casa.flush()
        return True

    def unlock(self) -> bool:
        return self.flush()

    def lock(self, write: bool = True, nattempts: int = 0) -> bool:
        return True

    def name(self) -> str:
        return str(self._path.absolute()) if self._path is not None else ""

    def _promote_to_casatools(self):
        """Hand this table over to a real casatools table tool from now on."""
        import casatools
        if self._table is not None:
            self._table.close()
            self._table = None
        self._casa = casatools.table()
        self._casa.open(str(self._path), nomodify=self._nomodify)
        return self._casa

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        casa = self._casa or self._promote_to_casatools()
        return getattr(casa, name)

    # -- description --
    def nrows(self) -> int:
        return self._casa.nrows() if self._casa is not None else self._table.nrows()

    def colnames(self) -> list:
        return self._casa.colnames() if self._casa is not None else self._table.colnames()

    def getkeywords(self) -> dict:
        if self._casa is not None:
            return self._casa.getkeywords()
        return {key: _keyword(value) for key, value in self._table.getkeywords().items()}

    def keywordnames(self) -> list:
        return list(self.getkeywords())

    def getkeyword(self, keyword):
        return self.getkeywords()[keyword]

    def getcolkeywords(self, columnname: str) -> dict:
        if self._casa is not None:
            return self._casa.getcolkeywords(columnname)
        return {key: _keyword(value) for key, value in self._table.getcolkeywords(columnname).items()}

    def getcolkeyword(self, columnname: str, keyword: str):
        return self.getcolkeywords(columnname)[keyword]

    def info(self) -> dict:
        return self._casa.info() if self._casa is not None else self._table.info()

    def getdesc(self) -> dict:
        if self._casa is not None:
            return self._casa.getdesc()
        return {name: self._table.getcoldesc(name) for name in self._table.colnames()}

    def getcoldesc(self, columnname: str) -> dict:
        return self._casa.getcoldesc(columnname) if self._casa is not None else self._table.getcoldesc(columnname)

    # -- reads --
    def getcol(self, columnname: str, startrow: int = 0, nrow: int = -1, rowincr: int = 1):
        if self._casa is not None:
            return self._casa.getcol(columnname, startrow, nrow, rowincr)
        values = self._table.getcol(columnname, int(startrow), int(nrow))
        if not len(values):                                                   # casatools: a flat empty array
            from .aipsio import SCALAR_DTYPES, TP_STRING
            code = self._table.coldesc(columnname).dtype
            return _promote(np.zeros(0, dtype=str if code == TP_STRING else SCALAR_DTYPES.get(code, np.float64)))
        if isinstance(values, list) and not isinstance(values[0], str):
            raise RuntimeError(f"getcol: column {columnname} has cells of different shapes or undefined cells")
        values = _promote(values)
        if rowincr != 1:
            values = values[::int(rowincr)]
        return _reverse(values)

    def getcell(self, columnname: str, rownr: int = 0):
        if self._casa is not None:
            return self._casa.getcell(columnname, rownr)
        value = self._table.getcell(columnname, int(rownr))
        if value is None:
            raise RuntimeError(f"getcell: no array in row {rownr} of column {columnname}")
        if isinstance(value, (str, bool)) or np.isscalar(value):
            return _scalar(value)
        return _reverse(_promote(value))

    def iscelldefined(self, columnname: str, rownr: int = 0) -> bool:
        if self._casa is not None:
            return self._casa.iscelldefined(columnname, rownr)
        try:
            return self._table.getcell(columnname, int(rownr)) is not None
        except (ValueError, IndexError):
            return False

    def getcolslice(self, columnname: str, blc, trc, incr=None, startrow: int = 0, nrow: int = -1, rowincr: int = 1):
        if self._casa is not None:
            return self._casa.getcolslice(columnname, blc, trc, incr, startrow, nrow, rowincr)
        incr = incr or [1] * len(blc)
        values = self._table.getcolslice(columnname, list(blc)[::-1], list(trc)[::-1], list(incr)[::-1],
                                         int(startrow), int(nrow))
        values = _promote(values)
        return _reverse(values[::int(rowincr)] if rowincr != 1 else values)

    # -- writes --
    def putcol(self, columnname: str, value, startrow: int = 0, nrow: int = -1, rowincr: int = 1) -> bool:
        if self._casa is not None:
            return self._casa.putcol(columnname, value, startrow, nrow, rowincr)
        if rowincr != 1:
            return self._promote_to_casatools().putcol(columnname, value, startrow, nrow, rowincr)
        column = self._table.coldesc(columnname)
        array = np.asarray(value)
        array = np.transpose(array) if array.ndim > 1 else array
        from .aipsio import SCALAR_DTYPES
        if column.dtype in SCALAR_DTYPES:
            array = np.ascontiguousarray(array, dtype=SCALAR_DTYPES[column.dtype])
        try:
            self._table.putcol(columnname, array, int(startrow))
        except UnsupportedTable:
            return self._promote_to_casatools().putcol(columnname, value, startrow, nrow, rowincr)
        return True

    def putcell(self, columnname: str, rownr, thevalue) -> bool:
        if self._casa is not None:
            return self._casa.putcell(columnname, rownr, thevalue)
        rows = [int(rownr)] if np.isscalar(rownr) else [int(r) for r in rownr]
        for row in rows:
            value = np.asarray(thevalue)
            self.putcol(columnname, value[..., None] if value.ndim else np.asarray([value]), row, 1)
        return True


def _keyword(value):
    """A keyword value as casatools returns it (lists of str as numpy string arrays, recursively)."""
    if isinstance(value, dict):
        return {key: _keyword(item) for key, item in value.items()}
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return np.array(value, dtype=str)
    if isinstance(value, np.generic):
        return value.item()
    return value


class LazyModule:
    """A module imported on first attribute access (``on_import(module)`` is then called once)."""

    def __init__(self, name: str, on_import=None) -> None:
        self._name = name
        self._module = None
        self._on_import = on_import

    def _load(self):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self._name)
            if self._on_import is not None:
                self._on_import(self._module)
        return self._module

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._load(), name)


class NativeTools:
    """``casatools`` with :class:`table` replaced by the native tool; every other tool (``ms``, ``msmetadata``,
    ``image``, ...) is casatools', imported the first time one is asked for."""

    table = table

    def __init__(self, casatools=None) -> None:
        self._casatools = casatools if casatools is not None else LazyModule("casatools")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._casatools, name)
