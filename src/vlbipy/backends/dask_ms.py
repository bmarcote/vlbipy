"""Dask-MS backend: the CASA pipeline with the calibration engine replaced by fast numpy solvers.

The measurement set stays the single source of truth. CASA still imports, flags
and exports it; what changes is every step that used to dominate the run time:

* fringe fitting (single-band delay, multi-band delay, the per-scan SNR survey),
* bandpass and gain solves,
* the application of the calibration (``applycal``),
* imaging, which always runs in difmapy on the per-source splits (``imager =
  "difmap"`` on the backend): a request for tclean or WSClean is redirected.

Those read and write the measurement set directly, in parallel worker processes,
through :mod:`vlbipy.solvers` (``fringefit_task``, ``gain_task``, ``apply_task``),
and write ordinary CASA calibration tables, so the two backends can be mixed
freely on the same working directory. A request the fast engine does not cover
(a cal library, ``bandtype='BPOLY'``, ...) falls back to the CASA task with a
warning.

The visibilities are also exposed as lazy dask-ms datasets (``get_data``), from
the measurement set or, when one was written, from a *dask-ms zarr store*
(``<code>.zarr``: main table partitioned by FIELD_ID/DATA_DESC_ID plus all
subtables). The store is optional (``[import].zarr_store`` or ``vlbipy export``)
and read-only as far as the pipeline is concerned; it is enough on its own to
rebuild the metadata after the measurement set has been deleted.

Two conversion paths produce identical stores:

* **casacore path** — dask-ms's native reader (``xds_from_ms``), used when
  python-casacore works on this machine;
* **casatools path** — a chunked reader built on ``casatools.table``, used where
  python-casacore is broken (it segfaults on some macOS builds, so health is
  probed in a subprocess) or not installed.

"""
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import BackendError
from ..logging_utils import get_logger, warnings
from ..models import Antenna, FreqSetup, ObsMetadata, Scan, Stokes
from ..solvers.msio import table_engine
from .casa import CasaBackend, CasaCalibrationOps, CasaDataOps

logger = get_logger()

#: Main-table columns that can never be stored (variable shape / index-only).
_SKIP_MAIN_COLUMNS = {"FLAG_CATEGORY"}

#: Default number of main-table rows per dask/zarr chunk (~100 MB of DATA for 64ch x 4pol).
_DEFAULT_CHUNK_ROWS = 50_000

#: Standard dask-ms dimension names for well-known main-table columns. All other
#: columns get per-column dimension names ("<column>-1", ...) exactly like dask-ms
#: does, so columns with equal rank but different shapes never conflict.
_DIMS_BY_COLUMN = {
    "UVW": ("row", "uvw"),
    "WEIGHT": ("row", "corr"), "SIGMA": ("row", "corr"),
    "DATA": ("row", "chan", "corr"), "FLAG": ("row", "chan", "corr"),
    "MODEL_DATA": ("row", "chan", "corr"), "CORRECTED_DATA": ("row", "chan", "corr"),
    "WEIGHT_SPECTRUM": ("row", "chan", "corr"), "SIGMA_SPECTRUM": ("row", "chan", "corr"),
}

#: casacore column valueType -> numpy dtype (casatools getcol upcasts, so we cast back).
_VALUETYPE_DTYPES = {"complex": np.complex64, "dcomplex": np.complex128, "float": np.float32,
                     "double": np.float64, "int": np.int32, "uint": np.uint32,
                     "short": np.int16, "boolean": np.bool_}


def _casacore_is_healthy() -> bool:
    """Return True if python-casacore imports cleanly on this machine (see :func:`vlbipy.solvers.msio.table_engine`)."""
    return table_engine() == "casacore"


def _column_dims(name: str, ndim: int) -> tuple[str, ...]:
    """Return the dimension names for a column: standard names for known columns,
    per-column names otherwise (avoids same-rank shape conflicts within a Dataset)."""
    dims = _DIMS_BY_COLUMN.get(name)
    if dims is not None and len(dims) == ndim:
        return dims
    return ("row",) + tuple(f"{name}-{i}" for i in range(1, ndim))


def _zero_pad_cells(cells: list, dtype) -> np.ndarray:
    """Stack variably-shaped row cells into one array, zero-padding to the max shape.

    ``cells`` may contain None for undefined rows (stored as all-zero rows).
    Used for columns like GAIN_CURVE.GAIN whose per-row shape follows NUM_POLY.
    """
    shapes = [cell.shape for cell in cells if cell is not None]
    if not shapes:
        raise ValueError("no readable cells to pad")
    ndim = max(len(s) for s in shapes)
    max_shape = tuple(max(s[i] if i < len(s) else 1 for s in shapes) for i in range(ndim))
    out = np.zeros((len(cells),) + max_shape, dtype=dtype)
    for i, cell in enumerate(cells):
        if cell is not None:
            region = tuple(slice(0, n) for n in cell.shape)
            out[(i,) + region] = cell
    return out


def _casatools_read(ms: str, taql: str, column: str, startrow: int, nrow: int,
                    dtype=None) -> np.ndarray:
    """Read ``nrow`` rows of one column via casatools, returned row-major.

    casatools returns Fortran order (``(..., nrows)``); ``.T`` yields ``(nrows, ...)``.
    ``dtype`` restores the on-disk type (casatools upcasts to double precision).
    """
    import casatools
    table = casatools.table()
    table.open(ms)
    try:
        selection = table.query(taql) if taql else table
        try:
            values = np.asarray(selection.getcol(column, startrow, nrow)).T
            return values.astype(dtype, copy=False) if dtype is not None else values
        finally:
            if taql:
                selection.close()
    finally:
        table.close()


def _casatools_table_to_dataset(path: str, taql: str = "", chunk_rows: int = _DEFAULT_CHUNK_ROWS,
                                attrs: Optional[dict] = None, skip: set = frozenset(),
                                lazy: bool = True):
    """Build a dask-ms Dataset for one (sub)table / TAQL selection via casatools.

    With ``lazy=True`` (main table) columns become delayed chunked reads. With
    ``lazy=False`` (subtables, which are small) columns are read fully up front,
    so any read failure surfaces here instead of inside a later dask compute.

    Columns are skipped when the single-row probe fails (undefined cells, e.g.
    HISTORY.APP_PARAMS; debug log). Columns whose probe passes but whose full
    read fails have variable row shapes (e.g. GAIN_CURVE.GAIN, whose shape
    follows NUM_POLY): those are read row by row and zero-padded to the maximum
    shape so no data is lost. Returns None for empty selections.
    """
    import casatools
    import dask
    import dask.array as da
    from daskms import Dataset

    table_name = Path(path).name
    table = casatools.table()
    table.open(path)
    try:
        selection = table.query(taql) if taql else table
        try:
            nrows = selection.nrows()
            if nrows == 0:
                return None
            probes = {}
            dtypes = {}
            eager = {}
            for name in selection.colnames():
                if name in skip:
                    continue
                try:
                    probes[name] = np.asarray(selection.getcol(name, 0, 1)).T
                    value_type = str(selection.getcoldesc(name).get("valueType", "")).lower()
                    dtypes[name] = _VALUETYPE_DTYPES.get(value_type, probes[name].dtype)
                except RuntimeError as exc:
                    logger.debug("skipping column {} of {}: {}", name, table_name, exc)
                    continue
                if not lazy:
                    try:
                        eager[name] = np.asarray(selection.getcol(name)).T.astype(dtypes[name], copy=False)
                    except RuntimeError:
                        # Variable row shapes: read cell by cell and zero-pad to the max shape.
                        cells = []
                        for row in range(nrows):
                            try:
                                cells.append(np.asarray(selection.getcell(name, row)).T)
                            except RuntimeError:
                                cells.append(None)
                        try:
                            eager[name] = _zero_pad_cells(cells, dtypes[name])
                        except ValueError:
                            logger.debug("skipping column {} of {}: no readable cells", name, table_name)
                            continue
                        logger.info("{}: column {} has variable row shapes; zero-padded to {}",
                                    table_name, name, eager[name].shape[1:])
        finally:
            if taql:
                selection.close()
    finally:
        table.close()

    data_vars = {}
    if not lazy:
        for name, values in eager.items():
            data_vars[name] = (_column_dims(name, values.ndim), da.from_array(values, chunks=values.shape))
        return Dataset(data_vars, attrs=attrs or {}) if data_vars else None

    row_chunk = min(chunk_rows, nrows)
    row_chunks = [row_chunk] * (nrows // row_chunk)
    if nrows % row_chunk:
        row_chunks.append(nrows % row_chunk)

    for name, probe in probes.items():
        cell_shape = probe.shape[1:]
        dtype = dtypes[name]
        blocks = []
        start = 0
        for n in row_chunks:
            delayed_read = dask.delayed(_casatools_read)(path, taql, name, start, n, dtype)
            blocks.append(da.from_delayed(delayed_read, shape=(n,) + cell_shape, dtype=dtype))
            start += n
        data_vars[name] = (_column_dims(name, 1 + len(cell_shape)), da.concatenate(blocks, axis=0))
    return Dataset(data_vars, attrs=attrs or {})


def _list_subtables(ms: Path) -> list[str]:
    """Return the names of the subtables present in a measurement set directory."""
    return sorted(p.name for p in ms.iterdir() if p.is_dir() and (p / "table.dat").is_file())


def ms_to_daskms(ms, store, chunk_rows: int = _DEFAULT_CHUNK_ROWS) -> Path:
    """Convert a measurement set to a dask-ms zarr store (main table + subtables).

    Parameters
    ----------
    ms : str or pathlib.Path
        The input measurement set.
    store : str or pathlib.Path
        Output zarr store path (created; must not exist).
    chunk_rows : int
        Main-table rows per chunk.

    Returns
    -------
    pathlib.Path
        The store path.
    """
    import dask
    from daskms.experimental.zarr import xds_to_zarr

    ms, store = Path(ms), Path(store)
    if not ms.is_dir():
        raise BackendError(f"measurement set not found: {ms}")
    use_casacore = _casacore_is_healthy()
    logger.info("converting {} -> dask-ms store {} (reader: {})", ms.name, store.name,
                "casacore" if use_casacore else "casatools")

    if use_casacore:
        import daskms
        main_datasets = daskms.xds_from_ms(str(ms), chunks={"row": chunk_rows})
        dask.compute(xds_to_zarr(main_datasets, str(store)))
        # Subtables are converted and computed one by one so a bad table (e.g. a
        # variable-shape column) skips just that table, with its partial dir removed.
        for name in _list_subtables(ms):
            try:
                sub = daskms.xds_from_table(f"{ms}::{name}")
                dask.compute(xds_to_zarr(sub, f"{store}::{name}"))
            except Exception as exc:  # noqa: BLE001 - optional subtables must not abort conversion
                warnings.warn(f"could not convert subtable {name}: {exc}")
                shutil.rmtree(store / name, ignore_errors=True)  # drop any partial write
    else:
        keys = _casatools_read(str(ms), "", "FIELD_ID", 0, -1), _casatools_read(str(ms), "", "DATA_DESC_ID", 0, -1)
        pairs = sorted(set(zip(keys[0].tolist(), keys[1].tolist())))
        logger.info("  {} partitions (FIELD_ID x DATA_DESC_ID), {} rows", len(pairs), len(keys[0]))
        main_datasets = []
        for field_id, ddid in pairs:
            taql = f"FIELD_ID=={field_id} && DATA_DESC_ID=={ddid}"
            dataset = _casatools_table_to_dataset(str(ms), taql, chunk_rows,
                                                  attrs={"FIELD_ID": int(field_id), "DATA_DESC_ID": int(ddid)},
                                                  skip=_SKIP_MAIN_COLUMNS)
            if dataset is not None:
                main_datasets.append(dataset)
        # casatools table tools are not thread-safe: compute serially.
        dask.compute(xds_to_zarr(main_datasets, str(store)), scheduler="synchronous")
        # Subtables are read eagerly (they are small), so read failures surface here
        # per table instead of aborting the whole conversion.
        for name in _list_subtables(ms):
            try:
                sub = _casatools_table_to_dataset(str(ms / name), lazy=False)
                if sub is not None:
                    dask.compute(xds_to_zarr([sub], f"{store}::{name}"), scheduler="synchronous")
            except Exception as exc:  # noqa: BLE001 - optional subtables must not abort conversion
                warnings.warn(f"could not convert subtable {name}: {exc}")
                shutil.rmtree(store / name, ignore_errors=True)  # drop any partial write

    logger.info("dask-ms store written: {}", store)
    return store


class DaskMsDataOps(CasaDataOps):
    """CASA data operations plus lazy dask-ms access and the optional zarr store."""

    def _has_ms(self, project_code: str) -> bool:
        """True when the project's measurement set is on disk."""
        return self.backend.ms_path(project_code).is_dir()

    def is_imported(self, project_code: str) -> bool:
        """Return True if the measurement set or, failing that, the dask-ms store exists."""
        return self._has_ms(project_code) or self.backend.store_path(project_code).is_dir()

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False, mms: bool = True,
                    zarr_store: bool = False, **kwargs) -> None:
        """Import FITS-IDI files to a measurement set (CASA) and optionally convert it to a dask-ms store.

        Parameters
        ----------
        project_code : str
            Project code (names the MS and the store).
        source_names : list of str
            Passed through to the CASA import.
        scan_gap : int
            Scan-boundary gap in seconds (importfitsidi scanreindexgap_s).
        files : list of str
            The FITS-IDI files, in order.
        delete : bool
            Redo the MS (and the store) if they exist.
        mms : bool
            Write a Multi-MS (default). Its parts are what the calibration application parallelises over, so
            a plain MS makes that step several times slower.
        zarr_store : bool
            Also write ``<code>.zarr`` (``[import].zarr_store``): a compact copy for lazy access and for
            rebuilding the report after the MS is gone. The pipeline itself never reads it.
        """
        store = self.backend.store_path(project_code)
        if store.is_dir() and not self._has_ms(project_code) and not delete:
            logger.info("dask-ms store {} already exists and there is no measurement set; skipping import", store)
            return
        kwargs.pop("keep_ms", None)  # legacy option: the measurement set is always kept now
        super().import_data(project_code, source_names, scan_gap=scan_gap, files=files, delete=delete, mms=mms,
                            **kwargs)
        if store.is_dir() and delete:
            logger.warning("removing existing dask-ms store {}", store)
            shutil.rmtree(store)
        if zarr_store and not store.is_dir():
            ms_to_daskms(self.backend.ms_path(project_code), store)

    # -- data access --
    def _store_datasets(self, project_code: str) -> list:
        """Return the main-table partitions of the zarr store as lazy dask-ms datasets."""
        from daskms.experimental.zarr import xds_from_zarr
        store = self.backend.store_path(project_code)
        if not store.is_dir():
            raise BackendError(f"dask-ms store not found: {store} (run import_data first)")
        return xds_from_zarr(str(store))

    def datasets(self, project_code: str) -> list:
        """Return the visibilities as lazy dask-ms datasets: from the zarr store when there is one, else the MS.

        The store is a snapshot taken at import or export time; the measurement set carries the current flags
        and the corrected data, so read it (delete or re-export the store) when those matter.
        """
        if self.backend.store_path(project_code).is_dir():
            return self._store_datasets(project_code)
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"data not found for {project_code}: neither a dask-ms store nor a measurement "
                               "set exists (run import_data first)")
        if table_engine() != "casacore":
            raise BackendError("lazy access to a measurement set needs a working python-casacore; "
                               f"export a store instead: vlbipy export -p {project_code} --format dask-ms")
        import daskms
        return daskms.xds_from_ms(str(ms), chunks={"row": _DEFAULT_CHUNK_ROWS})

    # -- metadata --
    def get_metadata(self, project_code: str, source_names: list[str], observatory: str) -> ObsMetadata:
        """Read the observation metadata from the measurement set, or from the store when the MS is gone."""
        if self._has_ms(project_code):
            return super().get_metadata(project_code, source_names, observatory)
        return self._metadata_from_store(project_code)

    def _metadata_from_store(self, project_code: str) -> ObsMetadata:
        """Read the full observation metadata from the zarr store (no measurement set needed)."""
        import dask
        import datetime as dt

        antenna_table = self.backend.get_table(project_code, "ANTENNA")
        antenna_names = [str(n) for n in antenna_table.NAME.values]
        stations = [str(s) for s in antenna_table.STATION.values]
        diameters = np.asarray(antenna_table.DISH_DIAMETER.values, dtype=float)
        positions = np.asarray(antenna_table.POSITION.values, dtype=float)
        mounts = ([str(m) for m in antenna_table.MOUNT.values]
                  if hasattr(antenna_table, "MOUNT") else [])

        field_table = self.backend.get_table(project_code, "FIELD")
        field_names = [str(n) for n in field_table.NAME.values]
        phase_dir = np.asarray(field_table.PHASE_DIR.values, dtype=float).reshape(len(field_names), -1)
        source_coords = {name: (float(np.degrees(phase_dir[i, 0]) % 360.0),
                                float(np.degrees(phase_dir[i, 1])))
                         for i, name in enumerate(field_names)}

        spw = self.backend.get_table(project_code, "SPECTRAL_WINDOW")
        chan_freq = np.asarray(spw.CHAN_FREQ.values, dtype=float)
        freq = FreqSetup(ref_freq=float((chan_freq.min() + chan_freq.max()) / 2.0),
                         total_bandwidth=float(np.sum(np.asarray(spw.TOTAL_BANDWIDTH.values))),
                         n_subbands=int(chan_freq.shape[0]), n_channels=int(chan_freq.shape[1]),
                         channel_width=float(abs(np.asarray(spw.CHAN_WIDTH.values).flat[0])),
                         polarizations=[])
        try:
            corr_types = np.asarray(
                self.backend.get_table(project_code, "POLARIZATION").CORR_TYPE.values)[0]
            freq.polarizations = [Stokes(int(c)) for c in corr_types if 0 <= int(c) <= 12]
        except Exception as exc:  # noqa: BLE001 - optional table; metadata proceeds without it
            logger.debug("no POLARIZATION information in the store: {}", exc)

        time_range = (0.0, 0.0)
        obs_date = None
        try:
            obs_table = self.backend.get_table(project_code, "OBSERVATION")
            time_range = tuple(float(v) for v in np.asarray(obs_table.TIME_RANGE.values)[0][:2])
            obs_date = (dt.datetime(1858, 11, 17) + dt.timedelta(seconds=time_range[0])).date()
        except Exception as exc:  # noqa: BLE001 - optional table; metadata proceeds without it
            logger.debug("no OBSERVATION information in the store: {}", exc)

        # Scans from the main-table partitions (small columns only; computed lazily).
        per_scan: dict[int, dict] = {}
        observed_fields: set[int] = set()
        for dataset in self._store_datasets(project_code):
            field_id = int(dataset.attrs.get("FIELD_ID", 0))
            observed_fields.add(field_id)
            scan_no, time, ant1, ant2 = dask.compute(dataset.SCAN_NUMBER.data, dataset.TIME.data,
                                                     dataset.ANTENNA1.data, dataset.ANTENNA2.data)
            spw_id = int(dataset.attrs.get("DATA_DESC_ID", 0))
            for scan in np.unique(scan_no):
                rows = scan_no == scan
                entry = per_scan.setdefault(int(scan), {
                    "source": field_names[field_id] if field_id < len(field_names) else "",
                    "t0": np.inf, "t1": -np.inf, "ants": set(), "spws": set()})
                entry["t0"] = min(entry["t0"], float(time[rows].min()))
                entry["t1"] = max(entry["t1"], float(time[rows].max()))
                entry["ants"].update(int(a) for a in np.unique(ant1[rows]))
                entry["ants"].update(int(a) for a in np.unique(ant2[rows]))
                entry["spws"].add(spw_id)

        observed: set[str] = set()
        scan_counts: dict[str, int] = {}
        scans = []
        for number in sorted(per_scan):
            entry = per_scan[number]
            ants = sorted(antenna_names[i] for i in entry["ants"] if i < len(antenna_names))
            observed.update(ants)
            for ant in ants:
                scan_counts[ant] = scan_counts.get(ant, 0) + 1
            scans.append(Scan(scan_number=number, source=entry["source"], time_start=entry["t0"],
                              time_end=entry["t1"], antennas=ants,
                              subbands=tuple(sorted(entry["spws"]))))

        # The FIELD table lists every scheduled source; keep only those with actual data
        # (a store partition only exists for FIELD_IDs that have rows).
        source_ids = {field_names[i]: i for i in sorted(observed_fields) if i < len(field_names)}
        source_names = list(source_ids)
        source_coords = {name: source_coords[name] for name in source_names if name in source_coords}
        if len(source_names) < len(field_names):
            logger.info("FIELD table lists {} sources but only {} have data; keeping those",
                        len(field_names), len(source_names))

        antennas = {name: Antenna(name=name, fullname=stations[i] if i < len(stations) else "",
                                  diameter=float(diameters[i]) if i < len(diameters) else 0.0,
                                  position=tuple(positions[i]) if i < len(positions) else (0.0, 0.0, 0.0),
                                  observed=name in observed,
                                  mount=mounts[i] if i < len(mounts) else "",
                                  n_scans=scan_counts.get(name, 0))
                    for i, name in enumerate(antenna_names)}
        meta = ObsMetadata(project_code=project_code, obs_date=obs_date, time_range=time_range,
                           antennas=antennas, scans=scans, freq_setup=freq,
                           source_names=source_names, source_coords=source_coords,
                           source_ids=source_ids)
        logger.info("metadata from {}: {} antennas ({} observed), {} scans, {} sources, {:.3f} GHz",
                    self.backend.store_path(project_code).name, len(antennas), len(observed),
                    len(scans), len(source_names), freq.freq_ghz)
        return meta

    # -- inspection --
    def listobs(self, project_code: str, listfile: Optional[str] = None) -> dict:
        """Return the scan listing: CASA's listobs when the MS exists, else one built from the store."""
        if self._has_ms(project_code):
            return super().listobs(project_code, listfile)
        meta = self._metadata_from_store(project_code)
        listing = {f"scan_{s.scan_number}": {"source": s.source, "time_start": s.time_start,
                                             "time_end": s.time_end, "antennas": s.antennas}
                   for s in meta.scans}
        if listfile:
            lines = [f"{s.scan_number:4d}  {s.source:16s} {s.time_start:.1f} - {s.time_end:.1f}  "
                     f"({', '.join(s.antennas)})" for s in meta.scans]
            Path(listfile).write_text("\n".join(lines) + "\n")
            logger.info("listobs written to {}", listfile)
        return listing


class DaskMsCalibrationOps(CasaCalibrationOps):
    """CASA calibration operations whose solves and application run on the numpy engine.

    Only the engine hooks are replaced: every public method (``initial_calibration``, ``fringefit``,
    ``bandpass``, ``scalar_bandpass``, ``scan_snr``, ...) is inherited, so selection, table bookkeeping and
    the prior chain are exactly those of the CASA backend.
    """

    #: ``apply`` keywords the fast application understands; anything else is passed to CASA's applycal.
    _FAST_APPLY_KEYWORDS = ("applymode", "workers", "chunk_rows", "flagbackup")

    def _run_fast(self, label: str, function, params: dict, casa_task) -> None:
        """Run ``function(**params)``; if it does not cover the request, warn and run the CASA task instead."""
        try:
            function(**params)
        except NotImplementedError as exc:
            warnings.warn(f"{label}: {exc}; running the CASA task instead")
            casa_task(params)

    def _run_fringefit_task(self, params: dict) -> None:
        """Fringe fit with :func:`vlbipy.solvers.fringefit_task.run_fringefit` (same keywords, same table format)."""
        from ..solvers.fringefit_task import run_fringefit
        self._run_fast("fringefit", run_fringefit, params, super()._run_fringefit_task)

    def _run_bandpass_task(self, params: dict) -> None:
        """Solve the bandpass with :func:`vlbipy.solvers.gain_task.run_bandpass` (CASA "B Jones" output)."""
        from ..solvers.gain_task import run_bandpass
        self._run_fast("bandpass", run_bandpass, params, super()._run_bandpass_task)

    def _run_gaincal_task(self, params: dict) -> None:
        """Solve gains with :func:`vlbipy.solvers.gain_task.run_gaincal` (CASA "G Jones" output)."""
        from ..solvers.gain_task import run_gaincal
        self._run_fast("gaincal", run_gaincal, params, super()._run_gaincal_task)

    def apply(self, project_code: str, field: str, tables: list, *, gainfield: str = "", parang: bool = True,
              flagbackup: bool = False, callib: bool = False, **kwargs) -> None:
        """Apply the calibration tables to the measurement set (CORRECTED_DATA, flags, weights) in one pass.

        Same contract as :meth:`CasaCalibrationOps.apply`: ``field`` empty corrects every observed source,
        each with the table chain and field mapping resolved for it. All of them are corrected in a single
        pass over the data by :func:`vlbipy.solvers.apply_task.apply_to_ms`. A cal library (``callib=True``) or
        an applycal keyword the fast path does not know is handed to CASA instead.
        """
        from ..solvers.apply_task import APPLY_MODES, apply_to_ms
        from ..solvers.fringefit_task import _prior_entries
        from ..solvers.msio import read_setup

        unknown = sorted(set(kwargs) - set(self._FAST_APPLY_KEYWORDS))
        mode = str(kwargs.get("applymode", "calflagstrict") or "calflagstrict").lower()
        if callib or unknown or mode not in APPLY_MODES:
            reason = "a cal library" if callib else (f"keyword(s) {', '.join(unknown)}" if unknown
                                                     else f"applymode={mode!r}")
            warnings.warn(f"{project_code}: the dask-ms apply does not cover {reason}; using CASA applycal")
            return super().apply(project_code, field, tables, gainfield=gainfield, parang=parang,
                                 flagbackup=flagbackup, callib=callib,
                                 **{k: v for k, v in kwargs.items() if k not in ("workers", "chunk_rows")})
        ms = self.backend.ms_path(project_code)
        usable = [t for t in tables if t.path and Path(t.path).exists()]
        missing = [t.cal_type for t in tables if t not in usable]
        if missing:
            warnings.warn(f"{project_code}: skipping missing calibration table(s): {', '.join(missing)}")
        if not usable:
            raise BackendError(f"{project_code}: no calibration tables to apply")
        if not parang:
            warnings.warn(f"{project_code}: applycal without the parallactic-angle correction")
        setup = read_setup(ms)
        if field:
            apply_fields = [name.strip() for name in field.split(",") if name.strip()]
        else:
            meta = self.backend.data.get_metadata(project_code, [], "")
            apply_fields = sorted({scan.source for scan in meta.scans})
        entries_by_field = {}
        for apply_field in apply_fields:
            if apply_field not in setup["field_names"]:
                raise BackendError(f"{project_code}: unknown field {apply_field!r} in applycal")
            params = self._compile_apply_params(project_code, usable, apply_field, include_calwt=True,
                                                gainfield=gainfield)
            entries = _prior_entries(params["gaintable"], params["gainfield"], params["interp"], params["spwmap"],
                                     setup["field_names"])
            for entry, calwt in zip(entries, params["calwt"]):
                entry["calwt"] = bool(calwt)
            entries_by_field[setup["field_names"].index(apply_field)] = entries
            applied = self._tables_for_field(usable, apply_field)
            logger.info("applycal: {} -> field={} ({} tables: {})", ms.name, apply_field, len(applied),
                        ", ".join(t.cal_type for t in applied))
        if flagbackup or kwargs.get("flagbackup"):
            self.backend.tasks.flagmanager(vis=str(ms), mode="save", versionname="before_applycal", merge="replace")
        apply_to_ms(ms, entries_by_field, parang=parang, applymode=mode, workers=kwargs.get("workers"),
                    **({"chunk_rows": int(kwargs["chunk_rows"])} if "chunk_rows" in kwargs else {}))

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, **kwargs) -> list:
        """A-priori amplitude calibration (gencal Tsys + GC, + EOP for VLBA/LBA).

        Runs gencal on the measurement set like the CASA backend. When only a dask-ms store is left, tables
        generated earlier are registered as they are; without them the step cannot run.
        """
        if self.backend.ms_path(project_code).is_dir():
            return super().a_priori(project_code, field, needs_eop=needs_eop, eop_file=eop_file, **kwargs)
        from ..models import CalTable
        from .casa import APRIORI_TABLE_SPECS
        paths = self.backend.apriori_table_paths(project_code, needs_eop)
        if all(path.is_dir() for path in paths.values()):
            logger.info("a_priori: no measurement set; using the existing a-priori caltables")
            return [CalTable(cal_type=cal_type, path=str(path), field=field,
                             interp=APRIORI_TABLE_SPECS[cal_type][1])
                    for cal_type, path in paths.items()]
        missing = ", ".join(t for t, p in paths.items() if not p.is_dir())
        raise BackendError(f"{project_code}: a-priori caltables missing ({missing}) and there is no measurement "
                           "set; re-run import_data(force=True) to regenerate them")


class DaskMsBackend(CasaBackend):
    """CASA backend with numpy/dask-ms calibration (fringe fits, bandpass, gains, applycal) and difmapy imaging.

    Parameters
    ----------
    work_dir : str
        Directory holding the measurement set, the optional store, and products.
    """

    kind = "dask-ms"
    requires_data_files = True
    #: Imaging and self-calibration always run in difmapy on the per-source splits; tclean/WSClean are not used.
    imager = "difmap"

    data_ops = DaskMsDataOps
    calibration_ops = DaskMsCalibrationOps

    def __init__(self, work_dir: str = ".") -> None:
        try:
            import daskms  # noqa: F401
        except ImportError as exc:
            raise BackendError("the dask-ms backend requires dask-ms: "
                               "pip install vlbipy[daskms]") from exc
        super().__init__(work_dir)

    def store_path(self, project_code: str) -> Path:
        """Return the zarr-store path for a project (work_dir/<code>.zarr)."""
        return self.work_dir / f"{project_code}.zarr"

    def get_table(self, project_code: str, table: str):
        """Return one subtable of the zarr store (e.g. ``"ANTENNA"``) as a dask-ms dataset."""
        from daskms.experimental.zarr import xds_from_zarr
        datasets = xds_from_zarr(f"{self.store_path(project_code)}::{table}")
        if not datasets:
            raise BackendError(f"subtable {table} not present in {self.store_path(project_code)}")
        return datasets[0]
