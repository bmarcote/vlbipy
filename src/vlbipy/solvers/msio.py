"""Measurement-set access for the dask-ms backend's solvers: layout index, chunk reads, column writes.

The measurement set is the single source of truth: CASA flags, images and
exports it, and the solvers here read and update the same columns. Three things
make that fast:

* the **layout** (:func:`read_layout`): the index columns are read once and cut
  into *runs* — maximal blocks of consecutive rows of one table part with the
  same field, scan and data description. A run is read with one contiguous
  ``getcol``, which is as fast as casacore gets; row-index (fancy) reads are not;
* a Multi-MS is addressed **part by part** (its ``SUBMSS/*.ms``), so different
  worker processes read — and, for the calibration application, write —
  different tables with no lock between them;
* visibility columns are read **sliced** (:func:`read_run`): a fringe fit needs
  the parallel hands of the central channels, a third of the bytes.

Table engine: python-casacore when it works on this machine, ``casatools.table``
otherwise (python-casacore segfaults on import on some macOS builds, so its health
is probed once in a subprocess; ``$VLBIPY_TABLE_ENGINE`` overrides the choice).
All arrays returned here are row-major ``(nrow, nchan, ncorr)`` in the on-disk
dtype, whichever engine read them.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

#: Environment variable forcing the table engine (``casacore`` or ``casatools``).
ENGINE_ENV = "VLBIPY_TABLE_ENGINE"
#: Columns whose cells are (nchan, ncorr) arrays and can therefore be read sliced.
CUBE_COLUMNS = ("DATA", "CORRECTED_DATA", "MODEL_DATA", "FLAG", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM")
#: casatools upcasts on read; these are the on-disk dtypes to cast back to.
_CASATOOLS_DTYPES = {"DATA": np.complex64, "CORRECTED_DATA": np.complex64, "MODEL_DATA": np.complex64,
                     "WEIGHT_SPECTRUM": np.float32, "SIGMA_SPECTRUM": np.float32, "WEIGHT": np.float32,
                     "SIGMA": np.float32, "FLAG": np.bool_, "FLAG_ROW": np.bool_}


@functools.cache
def table_engine() -> str:
    """Return the table engine to use: ``"casacore"`` when python-casacore imports cleanly, else ``"casatools"``.

    The import is probed in a subprocess because a broken python-casacore wheel
    segfaults, which cannot be caught in-process.
    """
    forced = os.environ.get(ENGINE_ENV, "").strip().lower()
    if forced in ("casacore", "casatools"):
        return forced
    try:
        probe = subprocess.run([sys.executable, "-c", "import casacore.tables"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return "casatools"
    return "casacore" if probe.returncode == 0 else "casatools"


class _CasacoreTable:
    """python-casacore table handle with the minimal row-major interface the solvers need."""

    def __init__(self, path: str, readonly: bool) -> None:
        from casacore.tables import table
        self._table = table(str(path), readonly=readonly, ack=False)

    def nrows(self) -> int:
        """Number of rows."""
        return int(self._table.nrows())

    def colnames(self) -> list[str]:
        """Column names."""
        return list(self._table.colnames())

    def getcol(self, name: str, start: int = 0, nrow: int = -1) -> np.ndarray:
        """Read ``nrow`` rows of a column from ``start``."""
        return self._table.getcol(name, start, nrow)

    def getcolslice(self, name: str, blc: list[int], trc: list[int], inc: list[int], start: int,
                    nrow: int) -> np.ndarray:
        """Read a (channel, correlation) slice of an array column (``blc``/``trc`` inclusive)."""
        return self._table.getcolslice(name, blc, trc, inc, start, nrow)

    def putcol(self, name: str, values: np.ndarray, start: int = 0) -> None:
        """Write ``values`` (row-major) to a column from row ``start``."""
        self._table.putcol(name, values, start, len(values))

    def add_column_like(self, name: str, template: str) -> None:
        """Add column ``name`` with the description and storage manager layout of ``template``."""
        from casacore.tables import makecoldesc
        description = dict(self._table.getcoldesc(template))
        description["comment"] = f"The {name.lower().replace('_', ' ')} column"
        manager = dict(self._table.getdminfo(template))
        manager["NAME"] = f"Tiled{name.title().replace('_', '')}"
        self._table.addcols(makecoldesc(name, description), manager)

    def close(self) -> None:
        """Flush and close."""
        self._table.close()


class _CasatoolsTable:
    """``casatools.table`` handle with the same interface (transposes the Fortran-order arrays)."""

    def __init__(self, path: str, readonly: bool) -> None:
        import casatools
        self._table = casatools.table()
        if not self._table.open(str(path), nomodify=readonly):
            raise OSError(f"could not open table {path}")

    def nrows(self) -> int:
        """Number of rows."""
        return int(self._table.nrows())

    def colnames(self) -> list[str]:
        """Column names."""
        return list(self._table.colnames())

    @staticmethod
    def _row_major(name: str, values) -> np.ndarray:
        """Transpose a casatools array to ``(nrow, ...)`` and restore the on-disk dtype."""
        values = np.ascontiguousarray(np.asarray(values).T)
        dtype = _CASATOOLS_DTYPES.get(name)
        return values.astype(dtype, copy=False) if dtype is not None else values

    def getcol(self, name: str, start: int = 0, nrow: int = -1) -> np.ndarray:
        """Read ``nrow`` rows of a column from ``start``."""
        return self._row_major(name, self._table.getcol(name, start, nrow))

    def getcolslice(self, name: str, blc: list[int], trc: list[int], inc: list[int], start: int,
                    nrow: int) -> np.ndarray:
        """Read a (channel, correlation) slice of an array column (``blc``/``trc`` inclusive)."""
        return self._row_major(name, self._table.getcolslice(name, blc[::-1], trc[::-1], inc[::-1], start, nrow))

    def putcol(self, name: str, values: np.ndarray, start: int = 0) -> None:
        """Write ``values`` (row-major) to a column from row ``start``."""
        self._table.putcol(name, np.asfortranarray(np.asarray(values).T), start, len(values))

    def add_column_like(self, name: str, template: str) -> None:
        """Add column ``name`` with the description and storage manager layout of ``template``."""
        description = {name: self._table.getcoldesc(template)}
        # casatools lists every data manager; take the one that stores the template column.
        managers = [dict(m) for m in self._table.getdminfo().values() if template in list(m.get("COLUMNS", []))]
        if not managers:
            raise OSError(f"no data manager found for column {template}")
        manager = managers[0]
        manager["NAME"] = f"Tiled{name.title().replace('_', '')}"
        manager["COLUMNS"] = np.array([name])
        manager.pop("SEQNR", None)
        self._table.addcols(description, {"*1": manager})

    def close(self) -> None:
        """Flush and close."""
        self._table.close()


def open_table(path, *, readonly: bool = True, engine: Optional[str] = None):
    """Open a casacore table with the selected engine; the handle reads and writes row-major arrays."""
    engine = engine or table_engine()
    return _CasacoreTable(str(path), readonly) if engine == "casacore" else _CasatoolsTable(str(path), readonly)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def ms_parts(ms) -> list[str]:
    """Return the tables holding the main rows: the sub-MSs of a Multi-MS, or the MS itself."""
    ms = Path(ms)
    subs = sorted(str(p) for p in (ms / "SUBMSS").glob("*.ms")) if (ms / "SUBMSS").is_dir() else []
    return subs or [str(ms)]


def read_setup(ms, *, engine: Optional[str] = None) -> dict:
    """Read the small subtables a solve needs: antenna/field names and positions, spw frequencies, DDID map.

    Returns
    -------
    dict
        ``antenna_names``, ``antenna_xyz`` (nant, 3), ``mounts``, ``field_names``, ``field_dirs`` (nfield, 2)
        [rad], ``chan_freq`` (nspw, nchan) [Hz], ``ddid_to_spw`` (nddid,), ``ncorr``.
    """
    ms = Path(ms)
    columns = {}
    for sub, names in (("ANTENNA", ("NAME", "POSITION", "MOUNT")), ("FIELD", ("NAME", "PHASE_DIR")),
                       ("SPECTRAL_WINDOW", ("CHAN_FREQ",)), ("DATA_DESCRIPTION", ("SPECTRAL_WINDOW_ID",
                                                                                  "POLARIZATION_ID")),
                       ("POLARIZATION", ("NUM_CORR",))):
        handle = open_table(ms / sub, engine=engine)
        try:
            present = handle.colnames()
            for name in names:
                if name in present:
                    columns[f"{sub}.{name}"] = np.asarray(handle.getcol(name))
        finally:
            handle.close()
    nfield = len(columns["FIELD.NAME"])
    chan_freq = np.asarray(columns["SPECTRAL_WINDOW.CHAN_FREQ"], dtype=np.float64)
    return {"antenna_names": [str(n) for n in columns["ANTENNA.NAME"]],
            "antenna_xyz": np.asarray(columns["ANTENNA.POSITION"], dtype=np.float64).reshape(-1, 3),
            "mounts": [str(m) for m in columns.get("ANTENNA.MOUNT", [])],
            "field_names": [str(n) for n in columns["FIELD.NAME"]],
            "field_dirs": np.asarray(columns["FIELD.PHASE_DIR"], dtype=np.float64).reshape(nfield, -1)[:, :2],
            "chan_freq": chan_freq.reshape(chan_freq.shape[0], -1),
            "ddid_to_spw": np.asarray(columns["DATA_DESCRIPTION.SPECTRAL_WINDOW_ID"], dtype=np.int64),
            "ncorr": int(np.asarray(columns["POLARIZATION.NUM_CORR"]).reshape(-1)[0])}


def read_layout(ms, *, engine: Optional[str] = None) -> dict:
    """Index the main table: one entry per run of consecutive rows sharing part, field, scan and DDID.

    Returns
    -------
    dict
        ``parts`` (list of table paths), ``columns`` (main-table column names) and parallel arrays over the
        runs: ``part`` (index into ``parts``), ``start``, ``nrow``, ``field``, ``scan``, ``ddid``, ``t_min``,
        ``t_max`` [MJD s]. ``nrows`` is the total row count.
    """
    parts = ms_parts(ms)
    runs = {name: [] for name in ("part", "start", "nrow", "field", "scan", "ddid", "t_min", "t_max")}
    columns: list[str] = []
    total = 0
    for index, part in enumerate(parts):
        handle = open_table(part, engine=engine)
        try:
            columns = columns or handle.colnames()
            nrows = handle.nrows()
            if nrows == 0:
                continue
            field, scan, ddid = (np.asarray(handle.getcol(name), dtype=np.int64)
                                 for name in ("FIELD_ID", "SCAN_NUMBER", "DATA_DESC_ID"))
            time = np.asarray(handle.getcol("TIME"), dtype=np.float64)
        finally:
            handle.close()
        total += nrows
        change = np.flatnonzero((np.diff(field) != 0) | (np.diff(scan) != 0) | (np.diff(ddid) != 0)) + 1
        starts = np.concatenate([[0], change])
        counts = np.diff(np.concatenate([starts, [nrows]]))
        runs["part"].append(np.full(starts.size, index, dtype=np.int64))
        runs["start"].append(starts)
        runs["nrow"].append(counts)
        runs["field"].append(field[starts])
        runs["scan"].append(scan[starts])
        runs["ddid"].append(ddid[starts])
        runs["t_min"].append(np.minimum.reduceat(time, starts))
        runs["t_max"].append(np.maximum.reduceat(time, starts))
    layout = {name: (np.concatenate(values) if values else np.zeros(0, dtype=np.int64))
              for name, values in runs.items()}
    layout.update(parts=parts, columns=columns, nrows=total)
    return layout


def select_runs(layout: dict, *, fields=None, scans=None, ddids=None, timerange=None) -> np.ndarray:
    """Indices of the layout runs matching the field / scan / DDID sets and overlapping the time range."""
    keep = np.ones(layout["start"].shape, dtype=bool)
    if fields is not None:
        keep &= np.isin(layout["field"], sorted(fields))
    if scans is not None:
        keep &= np.isin(layout["scan"], sorted(scans))
    if ddids is not None:
        keep &= np.isin(layout["ddid"], sorted(ddids))
    if timerange is not None:
        keep &= (layout["t_max"] >= timerange[0]) & (layout["t_min"] <= timerange[1])
    return np.flatnonzero(keep)


def run_specs(layout: dict, indices) -> list[dict]:
    """Picklable descriptions of the given runs for a worker job (path, row range, field, scan, DDID)."""
    return [{"path": layout["parts"][int(layout["part"][i])], "start": int(layout["start"][i]),
             "nrow": int(layout["nrow"][i]), "field": int(layout["field"][i]), "scan": int(layout["scan"][i]),
             "ddid": int(layout["ddid"][i])} for i in indices]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def _slice_bounds(index: Optional[np.ndarray], size: int) -> tuple[int, int, int, Optional[np.ndarray]]:
    """Turn an index array into ``(first, last, step, residual)`` for a sliced read.

    A regular progression is read exactly (``residual`` None); anything else is read
    from its first to its last element and then indexed with ``residual``.
    """
    if index is None:
        return 0, size - 1, 1, None
    index = np.asarray(index, dtype=np.int64)
    if index.size == 1:
        return int(index[0]), int(index[0]), 1, None
    steps = np.diff(index)
    if np.all(steps == steps[0]) and steps[0] > 0:
        return int(index[0]), int(index[-1]), int(steps[0]), None
    return int(index.min()), int(index.max()), 1, index - int(index.min())


def read_run(handle, start: int, nrow: int, columns: list[str], *, chans: Optional[np.ndarray] = None,
             corrs: Optional[np.ndarray] = None, nchan: int = 0, ncorr: int = 0) -> dict:
    """Read ``nrow`` consecutive rows of ``columns`` from an open table handle.

    Cube columns (:data:`CUBE_COLUMNS`) are restricted to the channel indices
    ``chans`` and correlation indices ``corrs`` (None = all; ``nchan``/``ncorr``
    are the full cell shape, needed only when one of them is given). Returns
    ``{column: row-major array}``.
    """
    out = {}
    sliced = chans is not None or corrs is not None
    if sliced:
        c0, c1, cstep, crest = _slice_bounds(chans, nchan)
        p0, p1, pstep, prest = _slice_bounds(corrs, ncorr)
    for name in columns:
        if sliced and name in CUBE_COLUMNS:
            values = handle.getcolslice(name, [c0, p0], [c1, p1], [cstep, pstep], start, nrow)
            if crest is not None:
                values = values[:, crest]
            if prest is not None:
                values = values[:, :, prest]
            out[name] = values
        elif sliced and name in ("WEIGHT", "SIGMA") and corrs is not None:
            out[name] = handle.getcol(name, start, nrow)[:, np.asarray(corrs, dtype=np.int64)]
        else:
            out[name] = handle.getcol(name, start, nrow)
    return out


def read_runs(specs: list[dict], columns: list[str], *, engine: str, chans_by_ddid: Optional[dict] = None,
              corrs: Optional[np.ndarray] = None, nchan: int = 0, ncorr: int = 0,
              timerange: Optional[tuple] = None) -> dict:
    """Read and concatenate several runs (all with the same channel count after selection).

    ``chans_by_ddid`` maps DATA_DESC_ID to the channel indices to read. With
    ``timerange`` only the rows inside ``(t0, t1)`` are read: TIME is read first
    and the cube columns are fetched for the enclosing row span only. Every run
    contributes ``FIELD_ID``/``SCAN_NUMBER``/``DATA_DESC_ID`` columns from its
    spec, so those need not be read. Returns ``{column: array}``; empty dict when
    no row survives.
    """
    pieces: dict[str, list] = {}
    handles: dict[str, object] = {}
    try:
        for spec in specs:
            handle = handles.get(spec["path"])
            if handle is None:
                handle = handles[spec["path"]] = open_table(spec["path"], engine=engine)
            start, nrow, keep = spec["start"], spec["nrow"], None
            if timerange is not None:
                time = np.asarray(handle.getcol("TIME", start, nrow), dtype=np.float64)
                inside = np.flatnonzero((time >= timerange[0]) & (time <= timerange[1]))
                if not inside.size:
                    continue
                start, nrow = start + int(inside[0]), int(inside[-1] - inside[0]) + 1
                if inside.size != nrow:
                    keep = inside - inside[0]
            chans = chans_by_ddid.get(spec["ddid"]) if chans_by_ddid is not None else None
            block = read_run(handle, start, nrow, columns, chans=chans, corrs=corrs, nchan=nchan, ncorr=ncorr)
            block["FIELD_ID"] = np.full(nrow, spec["field"], dtype=np.int64)
            block["SCAN_NUMBER"] = np.full(nrow, spec["scan"], dtype=np.int64)
            block["DATA_DESC_ID"] = np.full(nrow, spec["ddid"], dtype=np.int64)
            for name, values in block.items():
                pieces.setdefault(name, []).append(values if keep is None else values[keep])
    finally:
        for handle in handles.values():
            handle.close()
    return {name: (values[0] if len(values) == 1 else np.concatenate(values)) for name, values in pieces.items()}


def split_rows(specs: list[dict], max_rows: int) -> list[list[dict]]:
    """Group run specs into chunks of at most ``max_rows`` rows, splitting runs longer than that."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    filled = 0
    for spec in specs:
        offset = 0
        while offset < spec["nrow"]:
            take = min(spec["nrow"] - offset, max_rows - filled)
            current.append(dict(spec, start=spec["start"] + offset, nrow=take))
            filled += take
            offset += take
            if filled >= max_rows:
                chunks.append(current)
                current, filled = [], 0
    if current:
        chunks.append(current)
    return chunks
