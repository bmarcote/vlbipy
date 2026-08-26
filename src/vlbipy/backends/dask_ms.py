"""Dask-MS backend: the observation as lazy dask-ms datasets backed by a zarr store.

Importing with this backend produces, next to the measurement set, a *dask-ms
zarr store* (``<code>.zarr``) holding the full MS content (main table partitioned
by FIELD_ID/DATA_DESC_ID, plus all subtables). After conversion every operation
(metadata, ``get_data``) reads only the zarr store: lazy, chunked, and free of
any CASA/casacore dependency.

Two conversion paths produce identical stores:

* **casacore path** — dask-ms's native reader (``xds_from_ms``), used when
  python-casacore works on this machine;
* **casatools path** — a chunked reader built on ``casatools.table``, used where
  python-casacore is broken (it segfaults on some macOS builds, so health is
  probed in a subprocess) or not installed.

The FITS-IDI -> MS step itself is delegated to
:class:`~vlbipy.backends.casa.CasaBackend` (importfitsidi is CASA-only).
"""
import functools
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import BackendError
from ..logging_utils import get_logger, warnings
from ..models import Antenna, FreqSetup, ObsMetadata, Scan, Stokes
from .base import Backend, CalibrationOps, DataOps, PlotOps

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


@functools.cache
def _casacore_is_healthy() -> bool:
    """Return True if python-casacore imports cleanly (probed in a subprocess).

    A broken python-casacore wheel segfaults on import (seen on macOS), which
    cannot be caught in-process; the probe isolates the crash.
    """
    probe = subprocess.run([sys.executable, "-c", "import casacore.tables"],
                           capture_output=True, timeout=120)
    healthy = probe.returncode == 0
    logger.debug("python-casacore probe: {}", "healthy" if healthy else
                 f"unusable (exit {probe.returncode})")
    return healthy


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


class DaskMsDataOps(DataOps):
    """Import to a zarr store and read metadata from it (no CASA after import)."""

    def is_imported(self, project_code: str) -> bool:
        """Return True if the project's dask-ms store already exists."""
        return self.backend.store_path(project_code).is_dir()

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False,
                    keep_ms: bool = False, **kwargs) -> None:
        """Import FITS-IDI files to an MS (via CASA), convert to a dask-ms store, drop the MS.

        The zarr store is the primary data product: all further processing
        (metadata, calibration) reads it, so the intermediate MS is removed
        after a successful conversion unless ``keep_ms`` is set
        (``[import].keep_ms`` in the config).

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
            Redo both the MS and the store if they exist.
        keep_ms : bool
            Keep the intermediate measurement set instead of removing it.
        """
        store = self.backend.store_path(project_code)
        if store.is_dir():
            if not delete:
                logger.info("dask-ms store {} already exists; skipping import", store)
                return
            logger.warning("removing existing dask-ms store {}", store)
            shutil.rmtree(store)
        from .casa import CasaBackend
        casa = CasaBackend(work_dir=str(self.work_dir))
        kwargs.pop("mms", None)  # the intermediate MS is removed after conversion: plain MS suffices
        casa.data.import_data(project_code, source_names, scan_gap=scan_gap, files=files,
                              delete=delete, mms=False, **kwargs)
        ms = casa.ms_path(project_code)
        ms_to_daskms(ms, store)
        # gencal (the a_priori step) needs the MS, which is about to be removed:
        # pre-generate the a-priori caltables now; a_priori() registers them later.
        apriori_ok = True
        try:
            casa.calibrate.a_priori(project_code, field="", needs_eop=kwargs.get("needs_eop", False))
        except Exception as exc:  # noqa: BLE001 - keep the MS so a_priori can be retried
            apriori_ok = False
            warnings.anomaly(f"{project_code}: could not pre-generate the a-priori caltables "
                             f"({exc}); keeping the MS so the a_priori step can retry")
        if keep_ms or not apriori_ok:
            logger.info("keeping the intermediate MS {}", ms.name)
        elif not (store / "MAIN").is_dir():
            raise BackendError(f"store {store} has no MAIN table after conversion; "
                               f"keeping the MS {ms} for safety")
        else:
            shutil.rmtree(ms)
            logger.info("removed intermediate MS {}; the dask-ms store is now the primary "
                        "data product", ms.name)

    # -- data access --
    def datasets(self, project_code: str) -> list:
        """Return the main-table partitions as lazy dask-ms datasets."""
        from daskms.experimental.zarr import xds_from_zarr
        store = self.backend.store_path(project_code)
        if not store.is_dir():
            raise BackendError(f"dask-ms store not found: {store} (run import_data first)")
        return xds_from_zarr(str(store))

    # -- metadata --
    def get_metadata(self, project_code: str, source_names: list[str], observatory: str) -> ObsMetadata:
        """Read the full observation metadata from the zarr store (no CASA involved)."""
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
        for dataset in self.datasets(project_code):
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
        """Return a scan listing built from the store; optionally write it as text."""
        meta = self.get_metadata(project_code, [], "")
        listing = {f"scan_{s.scan_number}": {"source": s.source, "time_start": s.time_start,
                                             "time_end": s.time_end, "antennas": s.antennas}
                   for s in meta.scans}
        if listfile:
            lines = [f"{s.scan_number:4d}  {s.source:16s} {s.time_start:.1f} - {s.time_end:.1f}  "
                     f"({', '.join(s.antennas)})" for s in meta.scans]
            Path(listfile).write_text("\n".join(lines) + "\n")
            logger.info("listobs written to {}", listfile)
        return listing


class DaskMsCalibrationOps(CalibrationOps):
    """Calibration for the dask-ms backend (a-priori tables come from the import-time MS)."""

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, **kwargs) -> list:
        """A-priori amplitude calibration (gencal Tsys + GC, + EOP for VLBA/LBA).

        The caltables were pre-generated with gencal at import time (before the
        intermediate MS was removed); this step registers them. If they are
        missing but the MS was kept, gencal runs now via the CASA backend.
        """
        from ..models import CalTable
        from .casa import APRIORI_TABLE_SPECS, CasaBackend
        casa = CasaBackend(work_dir=str(self.work_dir))
        paths = casa.apriori_table_paths(project_code, needs_eop)
        if all(path.is_dir() for path in paths.values()):
            logger.info("a_priori: using the caltables pre-generated at import time")
            return [CalTable(cal_type=cal_type, path=str(path), field=field,
                             interp=APRIORI_TABLE_SPECS[cal_type][1])
                    for cal_type, path in paths.items()]
        if casa.ms_path(project_code).is_dir():
            return casa.calibrate.a_priori(project_code, field, needs_eop=needs_eop,
                                           eop_file=eop_file)
        missing = ", ".join(t for t, p in paths.items() if not p.is_dir())
        raise BackendError(f"{project_code}: a-priori caltables missing ({missing}) and the MS "
                           "was removed; re-run import_data(force=True) to regenerate them")


class DaskMsPlotOps(PlotOps):
    """Diagnostic plots for the dask-ms backend."""

    def caltable(self, project_code: str, caltable: str, cal_type: str = "") -> list[str]:
        """Plot a (pre-generated) calibration table to PNG(s) under <work_dir>/plots."""
        from ..plotting import CalTablePlotter
        plotter = CalTablePlotter(self.plot_dir("caltables"))
        return [str(p) for p in plotter.plot(caltable, cal_type=cal_type)]


class DaskMsBackend(Backend):
    """Backend exposing the observation as lazy dask-ms datasets in a zarr store.

    Import runs FITS-IDI -> MS (via the CASA backend) and then MS -> zarr; all
    reads afterwards touch only the zarr store.

    Parameters
    ----------
    work_dir : str
        Directory holding the store, the MS, and products.
    """

    kind = "dask-ms"
    requires_data_files = True

    data_ops = DaskMsDataOps
    calibration_ops = DaskMsCalibrationOps
    plot_ops = DaskMsPlotOps

    def __init__(self, work_dir: str = ".") -> None:
        try:
            import daskms  # noqa: F401
            from daskms.experimental import zarr  # noqa: F401
        except ImportError as exc:
            raise BackendError("the dask-ms backend requires dask-ms: "
                               "pip install vlbipy[daskms]") from exc
        super().__init__(work_dir)

    def store_path(self, project_code: str) -> Path:
        """Return the zarr-store path for a project (work_dir/<code>.zarr)."""
        return self.work_dir / f"{project_code}.zarr"

    def get_table(self, project_code: str, table: str):
        """Return one subtable (e.g. ``"ANTENNA"``) as a dask-ms dataset."""
        from daskms.experimental.zarr import xds_from_zarr
        datasets = xds_from_zarr(f"{self.store_path(project_code)}::{table}")
        if not datasets:
            raise BackendError(f"subtable {table} not present in {self.store_path(project_code)}")
        return datasets[0]
