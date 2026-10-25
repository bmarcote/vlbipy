"""Dask-MS backend: the CASA pipeline with the calibration engine replaced by fast numpy solvers.

The measurement set stays the single source of truth, written by the native
FITS-IDI import (:func:`vlbipy.solvers.idi_import.import_fitsidi` — CASA's
``importfitsidi`` + ``partition`` without CASA; a file the native importer does
not cover still goes through the CASA tasks). CASA still flags and exports the
measurement set; what changes is every step that used to dominate the run time:

* fringe fitting (single-band delay, multi-band delay, the per-scan SNR survey),
* bandpass and gain solves,
* the application of the calibration (``applycal``),
* imaging, which always runs in difmapy on the per-source splits (``imager =
  "difmap"`` on the backend): a request for tclean or WSClean is redirected.

Those read and write the measurement set directly, in parallel worker processes,
through :mod:`vlbipy.solvers` (``fringefit``, ``gain``, ``applycal``),
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
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import BackendError
from ..logging_utils import get_logger, warnings
from ..models import Antenna, FreqSetup, ObsMetadata, Scan, Stokes
from ..solvers.msio import table_engine
from ..tools import space_available_gb
from .casa import (IMPORT_FLAG_VERSION, CasaBackend, CasaCalibrationOps, CasaDataOps, CasaExportOps,
                   CasaFlagOps)

logger = get_logger()

#: casatools/casatasks are not thread-safe: the CASA fallbacks (a file the native FITS-IDI import does
#: not cover, a plain-MS request, an MS upgrade) go through this lock, so a parallel multi-epoch
#: fan-out runs them one project at a time.
_CASA_LOCK = threading.Lock()

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


class DaskMsPlotReads:
    """Parallel replacements of the CASA readers behind the slow diagnostic plots (see
    :mod:`vlbipy.solvers.plot_read`); mixed into :class:`DaskMsDataOps`."""

    def read_dynamic_spectra(self, project_code: str, *, field: str = "", column: str = "corrected",
                             max_time_bins: int = 200, stokes_i: bool = True,
                             metadata: Optional[ObsMetadata] = None) -> dict:
        from ..solvers.plot_read import read_dynamic_spectra
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        wanted = [f for f in field.split(",") if f]
        scans = [s for s in meta.scans if not wanted or s.source in wanted]
        if not scans:
            raise BackendError(f"{project_code}: no scans on {field!r}")
        n_time = max(1, int(max_time_bins))
        edges = np.linspace(min(s.time_start for s in scans), max(s.time_end for s in scans), n_time + 1)
        pol_indices, pol_labels = parallel_hand_indices(meta)
        result = read_dynamic_spectra(self.backend.ms_path(project_code), fields=wanted or None,
                                      column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                      corrs=list(pol_indices), edges=edges, stokes_i=stokes_i)
        names = list(meta.antennas)
        spectra = {(names[a], names[b]): grid for (a, b), grid in result["spectra"].items()}
        present = sorted({name for pair in spectra for name in pair}, key=names.index)
        logger.info("read_dynamic_spectra[dask-ms]: {} baseline(s) over {} time bins x {} channels ({}) in {:.1f}s",
                    len(spectra), n_time, meta.freq_setup.n_subbands * meta.freq_setup.n_channels,
                    "Stokes I" if stokes_i else ",".join(pol_labels), result["seconds"])
        return {"baselines": spectra, "antennas": present, "times": ((edges[:-1] + edges[1:]) / 2.0).tolist(),
                "frequencies_ghz": meta.freq_setup.frequencies_ghz(), "field": field, "column": column,
                "stokes_i": stokes_i}

    def _time_evolution_rows(self, project_code: str, meta: ObsMetadata, scans: list, refant: str,
                             refant_id: int, subband: int, first: int, last: int, column_name: str,
                             pol_indices: list[int]) -> tuple:
        """The same rows as the CASA reader, read in parallel (one subband, the central channels only)."""
        from ..solvers.msio import read_setup
        from ..solvers.plot_read import read_time_evolution_rows
        ms_path = self.backend.ms_path(project_code)
        ddid_to_spw = read_setup(ms_path)["ddid_to_spw"]
        ncorr = len(meta.freq_setup.polarizations)
        return read_time_evolution_rows(
            ms_path, column=column_name.upper(),
            corrs=None if list(pol_indices) == list(range(ncorr)) else list(pol_indices), refant=refant_id,
            ddids=[ddid for ddid, spw in enumerate(ddid_to_spw) if int(spw) == int(subband)],
            scans=[s.scan_number for s in scans], channels=list(range(first, last + 1)))

    def read_uvdistance(self, project_code: str, *, field: str = "", column: str = "corrected",
                        time_bin: float = 10.0, with_model: bool = False,
                        metadata: Optional[ObsMetadata] = None) -> dict:
        from ..solvers.plot_read import read_uvdistance
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        pol_indices, pol_labels = parallel_hand_indices(meta)
        ms_path = self.backend.ms_path(project_code)
        with_model = with_model and self._has_column(ms_path, "MODEL_DATA")
        wanted = [f for f in field.split(",") if f]
        starts = {s.scan_number: s.time_start for s in meta.scans if not wanted or s.source in wanted}
        result = read_uvdistance(ms_path, fields=wanted or None,
                                 column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                 corrs=list(pol_indices), time_bin=time_bin, scan_starts=starts,
                                 with_model=with_model)
        logger.info("read_uvdistance[dask-ms]: {} averaged point(s) on {} ({} s bins, per subband{}) in {:.1f}s",
                    result["uvdist_mlambda"].size, field or "all fields", time_bin,
                    ", with model" if with_model else "", result["seconds"])
        out = {"uvdist_mlambda": result["uvdist_mlambda"], "values": result["values"], "polarizations": pol_labels,
               "field": field, "column": column, "time_bin": time_bin}
        if with_model:
            out["model"] = {"uvdist_mlambda": result["uvdist_mlambda"], "values": result["model"]}
        return out


    def read_baseline_timeline(self, project_code: str, *, field: str, column: str = "corrected",
                               delay_search: bool = False, scans: Optional[list] = None,
                               metadata: Optional[ObsMetadata] = None) -> dict:
        """The band-averaged amplitude per baseline and integration, read in parallel (same output as
        :meth:`CasaDataOps.read_baseline_timeline`; 26 min there for a phase calibrator of GL048)."""
        from ..solvers.plot_read import read_baseline_timeline
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        wanted = {int(number) for number in scans} if scans else None
        field_scans = [s.scan_number for s in meta.scans if s.source == field and (wanted is None or
                                                                                    s.scan_number in wanted)]
        if not field_scans:
            raise BackendError(f"{project_code}: no scans on {field!r}")
        pol_indices, _ = parallel_hand_indices(meta)
        antenna_names = list(meta.antennas)
        n_antenna = len(antenna_names)
        try:
            result = read_baseline_timeline(self.backend.ms_path(project_code), field=field,
                                            column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                            corrs=list(pol_indices), scans=field_scans, nant=n_antenna,
                                            delay_search=delay_search)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
        if not result["time"].size:
            raise BackendError(f"{project_code}: no unflagged {column} data on {field!r}")
        times = np.unique(result["time"])
        index = np.searchsorted(times, result["time"])
        low, high = np.divmod(result["baseline"], n_antenna)
        sums = np.zeros((n_antenna, n_antenna, times.size), dtype=complex)
        counts = np.zeros((n_antenna, n_antenna, times.size))
        np.add.at(sums, (low, high, index), result["sum"])          # one vote per subband
        np.add.at(counts, (low, high, index), result["count"])
        with np.errstate(invalid="ignore", divide="ignore"):
            amplitude = np.where(counts > 0, np.abs(sums) / np.maximum(counts, 1.0), np.nan)
        amplitude = np.fmax(amplitude, np.transpose(amplitude, (1, 0, 2)))
        scan_of_time = np.zeros(times.size, dtype=int)
        scan_of_time[index] = result["scan"]
        steps = np.diff(times)
        integration = float(np.min(steps[steps > 0])) if (steps > 0).any() else 0.0
        logger.info("read_baseline_timeline[dask-ms]: {} on {}: {} integration(s) of {:.1f} s in {} scan(s), {:.1f}s",
                    column, field, times.size, integration, np.unique(scan_of_time).size, result["seconds"])
        return {"antennas": antenna_names, "times": times, "scans": scan_of_time, "amplitude": amplitude,
                "integration": integration, "field": field, "column": column}

    def read_total_visibility(self, project_code: str, *, fields: Optional[list[str]] = None,
                              column: str = "corrected", metadata: Optional[ObsMetadata] = None) -> dict:
        """The light curve of every source, read in one parallel pass (see :meth:`CasaDataOps.read_total_visibility`).

        The CASA reader selects every scan and subband separately through the ``ms`` tool: 22 000 selections
        and six hours on GL048 (a 22 M-row Multi-MS), against minutes here.
        """
        from ..solvers.plot_read import read_total_visibility
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        wanted = list(fields or meta.source_names)
        pol_indices, _ = parallel_hand_indices(meta)
        try:
            result = read_total_visibility(self.backend.ms_path(project_code), fields=wanted,
                                           column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                           corrs=list(pol_indices))
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
        names = result["field_names"]
        sources = {}
        keep = result["count"] > 0
        for field_id in np.unique(result["field"][keep]):
            rows = keep & (result["field"] == field_id)
            sources[names[int(field_id)]] = {"times": result["time"][rows], "vis_sum": result["sum"][rows],
                                             "n_vis": result["count"][rows].astype(float),
                                             "scans": result["scan"][rows].astype(np.int64)}
        time_start = min((float(e["times"].min()) for e in sources.values()),
                         default=float(meta.time_range[0]) if meta.time_range else 0.0)
        logger.info("read_total_visibility[dask-ms]: {} source(s), {} integration(s) from the {} column in {:.1f}s",
                    len(sources), sum(e["times"].size for e in sources.values()), column, result["seconds"])
        return {"sources": sources, "column": column, "time_start": time_start}


    def read_timeseries(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                        refant: str = "", column: str = "corrected", max_time_bins: int = 300,
                        metadata: Optional[ObsMetadata] = None) -> dict:
        """Amplitude/phase against time of the baselines to ``refant``, read in parallel (same output as
        :meth:`CasaDataOps.read_timeseries`, which reads every subband through the ``ms`` tool)."""
        from ..solvers.plot_read import read_timeseries
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        refant = refant or self.backend.calibrate.refant_chain(meta, "").split(",")[0]
        antenna_names = list(meta.antennas)
        if refant not in antenna_names:
            raise BackendError(f"{project_code}: reference antenna {refant!r} is not in the array")
        wanted = [f for f in field.split(",") if f]
        wanted_scans = {int(number) for number in (scans or [])}
        selected_scans = [s for s in meta.scans
                          if (not wanted or s.source in wanted)
                          and (not wanted_scans or s.scan_number in wanted_scans)]
        if not selected_scans:
            raise BackendError(f"{project_code}: no scans matching field={field!r}, scans={scans or 'all'}")
        # The bins of the CASA reader: within scans, never across a scan boundary.
        span = max(s.time_end for s in selected_scans) - min(s.time_start for s in selected_scans)
        bin_width = max(span / max(int(max_time_bins), 1), 1.0)
        edges: list[float] = []
        for scan in sorted(selected_scans, key=lambda s: s.time_start):
            n_scan_bins = max(1, int(np.ceil(scan.duration_sec / bin_width)))
            edges.extend(np.linspace(scan.time_start, scan.time_end, n_scan_bins + 1)[:-1])
            edges.append(scan.time_end)
        edges = np.asarray(edges, dtype=float)
        pol_indices, pol_labels = parallel_hand_indices(meta)
        result = read_timeseries(self.backend.ms_path(project_code), fields=wanted or None,
                                 scans=sorted(wanted_scans) or None,
                                 column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                 corrs=list(pol_indices), refant=antenna_names.index(refant), edges=edges)
        baselines = {f"{refant}-{antenna_names[other]}": values for other, values in result["baselines"].items()
                     if other < len(antenna_names)}
        logger.info("read_timeseries[dask-ms]: {} baseline(s) to {} over {} time bins in {:.1f}s",
                    len(baselines), refant, len(edges) - 1, result["seconds"])
        return {"baselines": dict(sorted(baselines.items())), "times": ((edges[:-1] + edges[1:]) / 2.0).tolist(),
                "polarizations": pol_labels, "refant": refant, "field": field, "column": column}

    # The CASA readers below read the corrected column through the ``ms`` tool: a deferred calibration
    # (vlbipy.solvers.corrected) is written first.
    def _corrected_written(self, project_code: str, column: str) -> None:
        if column == "corrected":
            self.backend.calibrate.ensure_corrected(project_code)

    def _selected_scans(self, meta: ObsMetadata, field: str, scans: Optional[list]) -> tuple[list, list]:
        """The field names and scan numbers a ``field``/``scans`` selection of the CASA readers covers."""
        wanted = [f for f in field.split(",") if f]
        numbers = sorted(int(s) for s in (scans or []))
        return wanted, numbers

    def read_spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                      refant: str = "", column: str = "corrected", all_pols: bool = False,
                      metadata: Optional[ObsMetadata] = None) -> dict:
        """Time-averaged spectra of the baselines to ``refant``, read in parallel (same output as
        :meth:`CasaDataOps.read_spectrum`, which reads through the ``ms`` tool)."""
        self._corrected_written(project_code, column)
        from ..solvers.plot_read import read_spectrum
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        refant = refant or self.backend.calibrate.refant_chain(meta, "").split(",")[0]
        antenna_names = list(meta.antennas)
        if refant not in antenna_names:
            raise BackendError(f"{project_code}: reference antenna {refant!r} is not in the array")
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        if all_pols:
            pol_indices = list(range(len(meta.freq_setup.polarizations)))
            pol_labels = [p.name for p in meta.freq_setup.polarizations]
        else:
            pol_indices, pol_labels = parallel_hand_indices(meta)
        wanted, numbers = self._selected_scans(meta, field, scans)
        result = read_spectrum(self.backend.ms_path(project_code), fields=wanted or None,
                               scans=numbers or None,
                               column={"corrected": "CORRECTED_DATA", "data": "DATA",
                                       "model": "MODEL_DATA"}[column],
                               corrs=list(pol_indices), refant=antenna_names.index(refant))
        spectra = {antenna_names[k]: v for k, v in result["spectra"].items() if k < len(antenna_names)}
        logger.info("read_spectrum[dask-ms]: {} baseline(s) to {} from the {} column ({} x {} channels) in {:.1f}s",
                    len(spectra), refant, column, n_spw, n_chan, result["seconds"])
        return {"antennas": sorted(spectra), "spectra": spectra, "refant": refant,
                "n_spw": n_spw, "n_channels": n_chan, "polarizations": pol_labels,
                "frequencies_ghz": [meta.freq_setup.frequencies_ghz(s) for s in range(n_spw)],
                "column": column, "field": field, "scans": list(scans or [])}

    def read_autocorr_spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                               column: str = "data", metadata: Optional[ObsMetadata] = None) -> dict:
        """Time-averaged autocorrelation amplitude spectra, read in parallel (same output as
        :meth:`CasaDataOps.read_autocorr_spectrum`)."""
        self._corrected_written(project_code, column)
        from ..solvers.plot_read import read_autocorr_spectrum
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        antenna_names = list(meta.antennas)
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        pol_indices, pol_labels = parallel_hand_indices(meta)
        wanted, numbers = self._selected_scans(meta, field, scans)
        result = read_autocorr_spectrum(self.backend.ms_path(project_code), fields=wanted or None,
                                        scans=numbers or None,
                                        column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                        corrs=list(pol_indices))
        spectra = {antenna_names[k]: v for k, v in result["spectra"].items() if k < len(antenna_names)}
        logger.info("read_autocorr_spectrum[dask-ms]: {} antenna(s) on {} scan(s) of {} from the {} column in "
                    "{:.1f}s", len(spectra), len(scans or []) or "all", field or "all fields", column,
                    result["seconds"])
        return {"antennas": sorted(spectra, key=antenna_names.index), "spectra": spectra,
                "n_spw": n_spw, "n_channels": n_chan, "polarizations": pol_labels,
                "frequencies_ghz": [meta.freq_setup.frequencies_ghz(s) for s in range(n_spw)],
                "scans": list(scans or []), "field": field, "column": column}

    def read_subband_phases(self, project_code: str, *, fields: list[str], refant: str,
                            column: str = "corrected", channel_fraction: float = 0.8,
                            metadata: Optional[ObsMetadata] = None) -> dict:
        """Per scan and baseline to ``refant`` the mean phase of every subband, read in parallel (same output as
        :meth:`CasaDataOps.read_subband_phases`)."""
        self._corrected_written(project_code, column)
        from ..solvers.plot_read import read_subband_phases
        from .casa import parallel_hand_indices
        meta = metadata or self.get_metadata(project_code, [], "")
        antenna_names = list(meta.antennas)
        if refant not in antenna_names:
            raise BackendError(f"{project_code}: reference antenna {refant!r} not in the data")
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        pol_indices, pol_labels = parallel_hand_indices(meta)
        scans = [s for s in meta.scans if s.source in set(fields)]
        margin = int(round(n_chan * (1.0 - channel_fraction) / 2.0))
        result = read_subband_phases(self.backend.ms_path(project_code), fields=list(fields),
                                     scans=[s.scan_number for s in scans],
                                     column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                                     corrs=list(pol_indices), refant=antenna_names.index(refant), first=margin,
                                     last=max(margin + 1, n_chan - margin))
        named = {}
        for other, per_scan in result["phases"].items():
            if other >= len(antenna_names):
                continue
            grid = np.full((len(scans), n_spw, len(pol_indices)), np.nan)
            for row, scan in enumerate(scans):
                if scan.scan_number in per_scan:
                    grid[row] = per_scan[scan.scan_number]
            named[antenna_names[other]] = grid
        logger.info("read_subband_phases[dask-ms]: {} baseline(s) to {} over {} scan(s) of {} ({} column) in {:.1f}s",
                    len(named), refant, len(scans), ",".join(fields), column, result["seconds"])
        return {"antennas": sorted(named, key=antenna_names.index), "phases": named, "refant": refant,
                "scans": [{"scan": s.scan_number, "source": s.source,
                           "time": 0.5 * (s.time_start + s.time_end)} for s in scans],
                "polarizations": pol_labels, "n_spw": n_spw, "column": column, "fields": list(fields)}

    def read_uv_coverage(self, project_code: str, *, field: str = "", max_points: int = 200_000,
                         metadata: Optional[ObsMetadata] = None) -> dict:
        """The sampled (u, v) points per source in Mlambda, read in parallel (same output as
        :meth:`CasaDataOps.read_uv_coverage`)."""
        from ..solvers.plot_read import read_uv_coverage
        meta = metadata or self.get_metadata(project_code, [], "")
        names = {fid: name for name, fid in meta.source_ids.items()}
        frequency = float(meta.freq_setup.ref_freq)
        to_mlambda = frequency / 299792458.0 / 1e6
        result = read_uv_coverage(self.backend.ms_path(project_code),
                                  fields=[f for f in field.split(",") if f] or None,
                                  max_points=int(max_points))
        fields = {names.get(fid, f"field{fid}"): {"u": (u * to_mlambda).tolist(), "v": (v * to_mlambda).tolist()}
                  for fid, (u, v) in result["fields"].items()}
        logger.info("read_uv_coverage[dask-ms]: {} field(s), {} point(s) on {} in {:.1f}s", len(fields),
                    sum(len(f["u"]) for f in fields.values()), field or "all fields", result["seconds"])
        return {"fields": fields, "unit": "Mlambda", "freq_ghz": frequency / 1e9}


class DaskMsDataOps(DaskMsPlotReads, CasaDataOps):
    """CASA data operations plus lazy dask-ms access and the optional zarr store."""

    def _has_ms(self, project_code: str) -> bool:
        """True when the project's measurement set is on disk."""
        return self.backend.ms_path(project_code).is_dir()

    def is_imported(self, project_code: str) -> bool:
        """Return True if the measurement set or, failing that, the dask-ms store exists."""
        return self._has_ms(project_code) or self.backend.store_path(project_code).is_dir()

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False, mms: bool = True,
                    zarr_store: bool = False, workers: Optional[int] = None, **kwargs) -> None:
        """Import FITS-IDI files natively (:func:`vlbipy.solvers.idi_import.import_fitsidi`) and optionally
        convert the measurement set to a dask-ms store.

        The native import writes the Multi-MS itself (the parts land in ``SUBMSS/`` as the worker
        processes write them), so no CASA task runs at all; a file the native importer does not cover —
        DiFX/VLBA correlator output needing van Vleck or weight corrections — falls back to CASA's
        ``importfitsidi`` + ``partition`` with a warning. Anything else (an existing plain MS to upgrade,
        ``mms=False``) keeps the CASA path.

        Parameters
        ----------
        project_code : str
            Project code (names the MS and the store).
        source_names : list of str
            Passed through to the CASA import (the native one needs no source list).
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
        workers : int
            Worker processes writing the parts (native import only).
        """
        store = self.backend.store_path(project_code)
        if store.is_dir() and not self._has_ms(project_code) and not delete:
            logger.info("dask-ms store {} already exists and there is no measurement set; skipping import", store)
            return
        kwargs.pop("keep_ms", None)  # legacy option: the measurement set is always kept now
        target = self.work_dir / f"{project_code}.mms"
        plain_ms = self.work_dir / f"{project_code}.ms"
        if mms and files and not plain_ms.is_dir() and (delete or not target.exists()):
            if target.exists():
                logger.warning("removing existing {}", target)
                shutil.rmtree(target)
            try:
                self._import_fitsidi_native(project_code, files, target, float(scan_gap), workers)
            except NotImplementedError as exc:
                warnings.warn(f"{project_code}: {exc}; importing with CASA instead")
                shutil.rmtree(target, ignore_errors=True)          # drop any partial import
                with _CASA_LOCK:
                    super().import_data(project_code, source_names, scan_gap=scan_gap, files=files,
                                        delete=delete, mms=mms, **kwargs)
        else:
            # An existing MS to upgrade or skip, a plain-MS request, or no files: casatools is not
            # thread-safe, so a parallel multi-epoch fan-out runs these one project at a time.
            with _CASA_LOCK:
                super().import_data(project_code, source_names, scan_gap=scan_gap, files=files,
                                    delete=delete, mms=mms, **kwargs)
        if store.is_dir() and delete:
            logger.warning("removing existing dask-ms store {}", store)
            shutil.rmtree(store)
        if zarr_store and not store.is_dir():
            ms_to_daskms(self.backend.ms_path(project_code), store)

    def _import_fitsidi_native(self, project_code: str, files: list[str], target: Path, scan_gap: float,
                               workers: Optional[int]) -> None:
        """Write ``<code>.mms`` with the native FITS-IDI import and save the as-imported flags."""
        from ..solvers.idi_import import import_fitsidi
        idi_size_gb = sum(Path(f).stat().st_size for f in files) / 1e9
        # The Multi-MS is written once (no transient plain MS): ~the size of the FITS-IDI data.
        needed = 1.5 * idi_size_gb
        if space_available_gb(self.work_dir) < needed:
            raise BackendError(f"not enough disk space in {self.work_dir} to create the MMS "
                               f"(~{needed:.0f} GB needed)")
        import_fitsidi([str(f) for f in files], target, scan_gap=scan_gap, workers=workers)
        # Keep the flags the import itself made (zero-weight data): a later reset returns to this state.
        try:
            self.backend.tasks.flagmanager(vis=str(target), mode="save", versionname=IMPORT_FLAG_VERSION,
                                           comment="flags as imported", merge="replace")
        except RuntimeError as exc:
            logger.warning("could not save the as-imported flags of {} ({})", target.name, exc)

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
        """Fringe fit with :func:`vlbipy.solvers.fringefit.fringefit` (same keywords, same table format)."""
        from ..solvers.fringefit import fringefit
        self._run_fast("fringefit", fringefit, params, super()._run_fringefit_task)

    def _run_gencal_task(self, params: dict) -> None:
        """``gencal(caltype='tsys')`` with :func:`vlbipy.solvers.apriori.tsys` (22 min -> 26 s on
        GL048) and ``caltype='eop'`` with :func:`vlbipy.solvers.eop.eop` (which also fixes a
        CASA eop.py extrapolation bug); the other caltypes, and a Tsys spectrum, with CASA."""
        caltype = params.get("caltype")
        if caltype == "eop":
            from ..solvers.eop import eop
            return self._run_fast("gencal eop", eop, params, super()._run_gencal_task)
        if caltype != "tsys" or params.get("uniform", True):
            return super()._run_gencal_task(params)
        from ..solvers.apriori import tsys
        self._run_fast("gencal tsys", lambda vis, caltable, **_: tsys(vis, caltable), params,
                       super()._run_gencal_task)

    def _run_accor_task(self, params: dict) -> None:
        """Solve the correlator amplitude correction with :func:`vlbipy.solvers.accor.accor`."""
        from ..solvers.accor import accor
        self._run_fast("accor", accor, params, super()._run_accor_task)

    def _run_smoothcal_task(self, params: dict) -> None:
        """Smooth a calibration table with :func:`vlbipy.solvers.accor.smoothcal`."""
        from ..solvers.accor import smoothcal
        self._run_fast("smoothcal", smoothcal, params, super()._run_smoothcal_task)

    def _set_model_column(self, ms: Path, field: str, model: str) -> None:
        """Fill MODEL_DATA natively (:func:`vlbipy.solvers.setjy.setjy`)."""
        from ..solvers.setjy import setjy
        setjy(ms, field, model)

    def _run_bandpass_task(self, params: dict) -> None:
        """Solve the bandpass with :func:`vlbipy.solvers.gain.bandpass` (CASA "B Jones" output)."""
        from ..solvers.gain import bandpass
        self._run_fast("bandpass", bandpass, params, super()._run_bandpass_task)

    def _run_gaincal_task(self, params: dict) -> None:
        """Solve gains with :func:`vlbipy.solvers.gain.gaincal` (CASA "G Jones" output)."""
        from ..solvers.gain import gaincal
        self._run_fast("gaincal", gaincal, params, super()._run_gaincal_task)

    def apply(self, project_code: str, field: str, tables: list, *, gainfield: str = "", parang: bool = True,
              flagbackup: bool = False, callib: bool = False, defer: bool = False, **kwargs) -> None:
        """Apply the calibration tables to the measurement set (CORRECTED_DATA, flags, weights) in one pass.

        Same contract as :meth:`CasaCalibrationOps.apply`: ``field`` empty corrects every observed source,
        each with the table chain and field mapping resolved for it. All of them are corrected in a single
        pass over the data by :func:`vlbipy.solvers.applycal.applycal`. A cal library (``callib=True``) or
        an applycal keyword the fast path does not know is handed to CASA instead.

        ``defer=True`` writes only the calibration flags and records the chains
        (:func:`vlbipy.solvers.applycal.defer_to_ms`): the readers of the corrected column then calibrate
        DATA as they read it, and CORRECTED_DATA is written when something needs the column itself
        (:meth:`ensure_corrected`).
        """
        from ..solvers.applycal import APPLY_MODES, apply_to_ms, defer_to_ms
        from ..solvers.fringefit import _prior_entries
        from ..solvers.msio import read_setup

        unknown = sorted(set(kwargs) - set(self._FAST_APPLY_KEYWORDS))
        mode = str(kwargs.get("applymode", "calflagstrict") or "calflagstrict").lower()
        ms = self.backend.ms_path(project_code)
        if callib or unknown or mode not in APPLY_MODES:
            reason = "a cal library" if callib else (f"keyword(s) {', '.join(unknown)}" if unknown
                                                     else f"applymode={mode!r}")
            warnings.warn(f"{project_code}: the dask-ms apply does not cover {reason}; using CASA applycal")
            # backend.tasks writes the deferred chains before CASA's applycal and records what it wrote.
            return super().apply(project_code, field, tables, gainfield=gainfield, parang=parang,
                                 flagbackup=flagbackup, callib=callib,
                                 **{k: v for k, v in kwargs.items() if k not in ("workers", "chunk_rows")})
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
        if defer:
            defer_to_ms(ms, entries_by_field, parang=parang, applymode=mode, workers=kwargs.get("workers"))
            return
        apply_to_ms(ms, entries_by_field, parang=parang, applymode=mode, workers=kwargs.get("workers"),
                    **({"chunk_rows": int(kwargs["chunk_rows"])} if "chunk_rows" in kwargs else {}))

    #: ``apply(defer=True)`` is understood (see :meth:`apply`).
    defers_apply = True

    def ensure_corrected(self, project_code: str) -> list[int]:
        """Write CORRECTED_DATA of every field whose calibration was deferred; returns those fields.

        For anything that reads the column itself rather than through the native readers: CASA tasks and
        tools, jplotter, a look at the data by hand.
        """
        from ..solvers.corrected import write_pending
        return write_pending(self.backend.ms_path(project_code))

    def reweight(self, project_code: str, *, column: str = "corrected", timebin: str = "",
                 minsamp: int = 2, flagbackup: bool = True, **kwargs) -> dict:
        """statwt with its default binning (one integration, a whole spw), computed natively in parallel.

        Same weights, flags and report as :meth:`CasaCalibrationOps.reweight` (see
        :mod:`vlbipy.solvers.statwt`); a coarser time binning or any other statwt option goes to CASA.
        """
        from ..solvers.statwt import statwt
        options = {k: v for k, v in kwargs.items() if v not in (None, "")}
        single_integration = str(timebin).strip() in ("", "1")
        if options or not single_integration or column not in ("corrected", "data"):
            # CASA's statwt reads the column itself (backend.tasks writes a deferred calibration first).
            return super().reweight(project_code, column=column, timebin=timebin, minsamp=minsamp,
                                    flagbackup=flagbackup, **kwargs)
        ms = self.backend.ms_path(project_code)
        logger.info("statwt[dask-ms]: datacolumn={!r} minsamp={} flagbackup={}", column, minsamp, flagbackup)
        if flagbackup:
            # CASA's statwt names its backups statwt_1, statwt_2, ...
            listing = Path(f"{ms}.flagversions") / "FLAG_VERSION_LIST"
            names = [line.split(":")[0].strip() for line in listing.read_text().splitlines()] if listing.is_file() else []
            number = 1 + max([int(n.split("_")[1]) for n in names if n.startswith("statwt_") and n[7:].isdigit()],
                             default=0)
            self.backend.tasks.flagmanager(vis=str(ms), mode="save", versionname=f"statwt_{number}")
        try:
            report = statwt(ms, column={"corrected": "CORRECTED_DATA", "data": "DATA"}[column],
                               minsamp=int(minsamp))
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: statwt failed: {exc}") from exc
        logger.info("statwt: new weights mean={} variance={}", report.get("mean"), report.get("variance"))
        return report

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, needs_tec: bool = False, **kwargs) -> list:
        """A-priori amplitude calibration (gencal Tsys + GC, + EOP for VLBA/LBA, + TEC below the
        observatory's ionosphere threshold).

        Runs gencal on the measurement set like the CASA backend — except the ionospheric table, which
        CASA gets wrong (its F Jones sign doubles the dispersive delay), so it is computed natively by
        :func:`vlbipy.solvers.tec.tec` from IONEX maps cached under ``<work_dir>/ionex``.
        When only a dask-ms store is left, tables generated earlier are registered as they are; without
        them the step cannot run.
        """
        from ..models import CalTable
        from .casa import APRIORI_TABLE_SPECS
        tec_path = self.backend.caldir() / f"{project_code}{APRIORI_TABLE_SPECS['tec'][0]}"
        if self.backend.ms_path(project_code).is_dir():
            tables = super().a_priori(project_code, field, needs_eop=needs_eop, eop_file=eop_file,
                                      needs_tec=False, **kwargs)
            if needs_tec:
                tables.append(self._tec_table(project_code, field, tec_path))
            return tables
        paths = self.backend.apriori_table_paths(project_code, needs_eop)
        if needs_tec:
            paths["tec"] = tec_path
        if all(path.is_dir() for path in paths.values()):
            logger.info("a_priori: no measurement set; using the existing a-priori caltables")
            return [CalTable(cal_type=cal_type, path=str(path), field=field,
                             interp=APRIORI_TABLE_SPECS[cal_type][1])
                    for cal_type, path in paths.items()]
        missing = ", ".join(t for t, p in paths.items() if not p.is_dir())
        raise BackendError(f"{project_code}: a-priori caltables missing ({missing}) and there is no measurement "
                           "set; re-run import_data(force=True) to regenerate them")

    def _tec_table(self, project_code: str, field: str, tec_path: "Path") -> "CalTable":
        """Write (or reuse) the ionospheric Fringe Jones table and return it as a CalTable."""
        from ..models import CalTable
        from ..solvers.tec import tec
        from .casa import APRIORI_TABLE_SPECS
        ionex_dir = Path(self.work_dir) / "ionex"
        ionex_dir.mkdir(parents=True, exist_ok=True)
        if tec_path.is_dir():
            shutil.rmtree(tec_path)
        report = tec(self.backend.ms_path(project_code), tec_path, ionex_dir=ionex_dir)
        logger.info("a_priori: TEC table {} (IONEX {}, {:.1f}-{:.1f} TECU)", tec_path.name,
                    report["product"], *report["stec_range"])
        return CalTable(cal_type="tec", path=str(tec_path), field=field,
                        interp=APRIORI_TABLE_SPECS["tec"][1])


#: casatasks that never read CORRECTED_DATA: they need no deferred calibration written first.
_TASKS_WITHOUT_CORRECTED = {"flagmanager", "listobs", "gencal", "fringefit", "gaincal", "bandpass", "smoothcal",
                            "accor", "importfitsidi", "importuvfits", "setjy", "ft", "plotants", "applycal",
                            "clearcal", "flagdata"}
#: casatasks whose data column defaults to DATA (the others default to the corrected data).
_TASKS_READING_DATA = {"plotms", "visstat"}


def _reads_corrected(task: str, kwargs: dict) -> bool:
    """Whether a casatasks call may read CORRECTED_DATA (its data column, or its default one)."""
    if task in _TASKS_WITHOUT_CORRECTED:
        return False
    columns = [str(kwargs[key]).lower() for key in ("datacolumn", "xdatacolumn", "ydatacolumn") if kwargs.get(key)]
    if not columns:
        return task not in _TASKS_READING_DATA      # tclean, mstransform, split, exportuvfits, statwt, partition...
    return any(c not in ("data", "model", "model_data", "float_data") for c in columns)


class NativeFlagTasks:
    """``casatasks`` with ``flagdata`` answered natively where it can be (see :mod:`vlbipy.solvers.flagdata`),
    and the deferred calibrations written before a task reads CORRECTED_DATA.

    Every flagging call of the CASA flag operations goes through ``backend.tasks.flagdata``, so wrapping it
    here moves all of them (a-priori file, autocorrelations, quack, band edges, bandpass gaps, off-source and
    outlier commands) off ``casatasks.flagdata`` at once. A call the native engine does not reproduce exactly
    (tfcrop, rflag, an unusual selection) is passed to CASA unchanged; every other attribute is casatasks'.

    A measurement set whose calibration was deferred (:mod:`vlbipy.solvers.corrected`) has a CORRECTED_DATA
    that is out of date: any other task that may read it gets it written first, and after CASA's applycal or
    clearcal the record says what the column holds.
    """

    def __init__(self, tasks) -> None:
        self._tasks = tasks

    def __getattr__(self, name):
        task = getattr(self._tasks, name)
        if not callable(task) or name.startswith("_"):
            return task

        def guarded(*args, **kwargs):
            from ..solvers import corrected
            vis = kwargs.get("vis", args[0] if args else None)
            if not isinstance(vis, (str, Path)) or not (Path(vis) / corrected.RECORD_DIR).is_dir():
                return task(*args, **kwargs)
            if name == "applycal" or _reads_corrected(name, kwargs):
                corrected.write_pending(vis)
            result = task(*args, **kwargs)
            if name == "clearcal":
                corrected.forget(vis)
            elif name == "applycal":
                corrected.record_external(vis, list(corrected.load_record(vis)["fields"]), "casa")
            return result

        return guarded

    def flagmanager(self, **kwargs):
        """Run ``flagmanager`` natively (:mod:`vlbipy.solvers.flagmanager` — same files), CASA as fallback."""
        from ..solvers.flagmanager import flagmanager
        try:
            return flagmanager(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a broken version table must not stop the pipeline
            logger.debug("flagmanager[native] failed ({}); using casatasks.flagmanager", exc)
            with _CASA_LOCK:
                return self._tasks.flagmanager(**kwargs)

    def flagdata(self, **kwargs):
        """Run ``flagdata`` natively when the call is a supported selection, otherwise through CASA."""
        from ..solvers.flagdata import Unsupported, apply_flag_selections, compile_flagdata, read_flag_setup
        vis = kwargs.get("vis")
        if vis and Path(str(vis)).is_dir():
            try:
                setup = read_flag_setup(vis)
                selections = compile_flagdata(kwargs, setup)
            except Unsupported as exc:
                logger.debug("flagdata: {}; using casatasks.flagdata", exc)
            else:
                if kwargs.get("flagbackup"):
                    self.flagmanager(vis=str(vis), mode="save", versionname=f"flagdata_{int(time.time())}")
                stats = apply_flag_selections(vis, selections, setup=setup)
                logger.info("flagdata[native]: {} command(s) -> {} selection(s), {} of {} row(s) touched, "
                            "{} cell(s) changed in {} part(s), {:.2f}s", len(selections) if kwargs.get("mode")
                            != "list" else _count_commands(kwargs), stats["commands"], stats["rows_touched"],
                            stats["rows"], stats["cells_changed"], stats["parts"], stats["seconds"])
                return {}
            from ..solvers.flagdata import parallel_casa_flagdata
            if "corrected" in str(kwargs.get("datacolumn", "")).lower():
                from ..solvers.corrected import write_pending
                write_pending(vis)
            if kwargs.get("flagbackup"):
                self.flagmanager(vis=str(vis), mode="save", versionname=f"flagdata_{int(time.time())}")
            try:
                logfile = str(self._tasks.casalog.logfile())
            except Exception:  # noqa: BLE001 - the parts then log to their own default file
                logfile = ""
            stats = parallel_casa_flagdata(dict(kwargs, flagbackup=False), logfile=logfile)
            if stats is not None:
                logger.info("flagdata[casa x{}]: mode={} run on every part in parallel, {:.1f}s", stats["parts"],
                            kwargs.get("mode", "manual"), stats["seconds"])
                return {}
        if vis and Path(str(vis)).is_dir() and "corrected" in str(kwargs.get("datacolumn", "")).lower():
            from ..solvers.corrected import write_pending
            write_pending(vis)
        return self._tasks.flagdata(**kwargs)


def _count_commands(kwargs: dict) -> int:
    source = kwargs.get("inpfile")
    if isinstance(source, (list, tuple)):
        return len(source)
    try:
        return sum(1 for line in Path(str(source)).read_text().splitlines() if line.strip())
    except OSError:
        return 0


class DaskMsFlagOps(CasaFlagOps):
    """CASA flag operations with native flag statistics and data reads; the flagdata calls themselves go
    through :class:`NativeFlagTasks`."""

    def _quack_series(self, project_code: str, meta: ObsMetadata, scans: list, wanted: list[str],
                      column_name: str, pol_indices: list[int]) -> dict:
        """Same series as the CASA reader, read in parallel over the table parts (one pass, wanted fields only)."""
        from ..solvers.flagdata import read_row_amplitudes
        if column_name.lower() == "corrected_data":
            self.backend.calibrate.ensure_corrected(project_code)     # read as it is on disk
        ncorr = len(meta.freq_setup.polarizations)
        rows = read_row_amplitudes(self.backend.ms_path(project_code), fields=wanted or None,
                                   column=column_name.upper(),
                                   corrs=None if list(pol_indices) == list(range(ncorr)) else list(pol_indices))
        logger.info("quack: read {} baseline sample(s) in {:.1f}s", rows["time"].size, rows["seconds"])
        starts = np.array([s.time_start for s in scans], dtype=float)
        ends = np.array([s.time_end for s in scans], dtype=float)
        times, ant1, ant2 = rows["time"], rows["ant1"].astype(np.int64), rows["ant2"].astype(np.int64)
        scan_index = np.searchsorted(starts, times, side="right") - 1
        inside = (scan_index >= 0) & (scan_index < len(scans))
        inside[inside] &= times[inside] <= ends[scan_index[inside]]
        keep = np.flatnonzero(inside)
        if not keep.size:
            return {}
        n_ant = max(len(meta.antennas), 1)
        codes = (scan_index[keep].astype(np.int64) * n_ant + ant1[keep]) * n_ant + ant2[keep]
        order = np.argsort(codes, kind="stable")          # keeps subband-then-time order inside each baseline
        keep, codes = keep[order], codes[order]
        unique, first = np.unique(codes, return_index=True)
        series = {}
        for code, group in zip(unique, np.split(keep, first[1:])):
            scan_id, pair = divmod(int(code), n_ant * n_ant)
            series[(scan_id, *divmod(pair, n_ant))] = (times[group] - starts[scan_id],
                                                       rows["amplitude"][group].astype(float))
        return series

    def flagged_fraction(self, project_code: str, *, where: str = "") -> float:
        if where:
            return super().flagged_fraction(project_code, where=where)
        from ..solvers.flagdata import count_flags
        counts = count_flags(self.backend.ms_path(project_code))
        observable = int(counts["observable"].sum())
        return int(counts["flagged"].sum()) / observable if observable else 0.0

    def summary(self, project_code: str, *, where: str = "") -> dict:
        if where:
            return super().summary(project_code, where=where)
        from ..solvers.flagdata import count_flags
        counts = count_flags(self.backend.ms_path(project_code))
        flagged, observable, names = counts["flagged"], counts["observable"], counts["antenna_names"]

        def entry(f: int, o: int) -> dict:
            return {"flagged": int(f), "observable": int(o), "fraction": f / o if o else 0.0}

        report = entry(flagged.sum(), observable.sum())
        report["spw"] = {ddid: entry(flagged[:, :, ddid].sum(), observable[:, :, ddid].sum())
                         for ddid in range(flagged.shape[2]) if observable[:, :, ddid].sum()}
        # Each baseline is stored once: an antenna's data is its row plus its column.
        per_flag = flagged.sum(axis=2)
        per_obs = observable.sum(axis=2)
        antenna_f = per_flag.sum(axis=1) + per_flag.sum(axis=0) - np.diag(per_flag)
        antenna_o = per_obs.sum(axis=1) + per_obs.sum(axis=0) - np.diag(per_obs)
        report["antenna"] = {name: entry(antenna_f[i], antenna_o[i]) for i, name in enumerate(names)}
        dead = [i for i, name in enumerate(names) if antenna_o[i] > 0 and antenna_f[i] == antenna_o[i]]
        alive = np.ones(len(names), dtype=bool)
        alive[dead] = False
        keep = alive[:, None] & alive[None, :]
        report["excluding_dead"] = entry(per_flag[keep].sum(), per_obs[keep].sum())
        report["excluding_dead"]["excluded_antennas"] = [names[i] for i in dead]
        return report


def _timebin_seconds(timebin) -> Optional[float]:
    """``"10s"``, ``"1min"``, ``"10"`` or ``""`` in seconds (0 = no averaging); None when not understood."""
    text = str(timebin or "").strip().lower()
    if not text:
        return 0.0
    for unit, factor in (("min", 60.0), ("s", 1.0), ("h", 3600.0)):
        if text.endswith(unit):
            text, scale = text[: -len(unit)].strip(), factor
            break
    else:
        scale = 1.0
    try:
        return float(text) * scale
    except ValueError:
        return None


class DaskMsExportOps(CasaExportOps):
    """CASA export with the per-source split done natively (see :mod:`vlbipy.solvers.split`)."""

    def split_band(self, project_code: str, spws: list[int], target: str) -> str:
        """Split the subbands ``spws`` natively (:func:`vlbipy.solvers.split.split`)."""
        from ..solvers.split import split
        split(self.backend.ms_path(project_code), target, field="", column="DATA", spws=spws)
        return str(target)

    def ms(self, project_code: str, source: str, *, datacolumn: str = "corrected",
           chanbin: int = -1, timebin: str = "", keepflags: bool = True,
           outputvis: str = "", metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Split one source like :meth:`CasaExportOps.ms`, as a plain measurement set written in parallel.

        mstransform took 12 minutes per source on GL048 (one read thread over a 66 GB Multi-MS); the
        averaging here gives the same rows (checked column by column). Options it does not reproduce
        (dropping flagged rows, other columns, extra mstransform keywords) go to mstransform.
        """
        from ..solvers.split import split
        seconds = _timebin_seconds(timebin)
        column = {"corrected": "CORRECTED_DATA", "data": "DATA"}.get(str(datacolumn).lower())
        # The output table is created with python-casacore; where it is broken (some macOS builds) use CASA.
        if kwargs or not keepflags or seconds is None or column is None or table_engine() != "casacore":
            return super().ms(project_code, source, datacolumn=datacolumn, chanbin=chanbin, timebin=timebin,
                              keepflags=keepflags, outputvis=outputvis, metadata=metadata, **kwargs)
        ms = self.backend.ms_path(project_code)
        outdir = self.work_dir / "calibrated_data"
        outdir.mkdir(parents=True, exist_ok=True)
        target = Path(outputvis) if outputvis else outdir / f"{project_code}_{source}.ms"
        n_channels = (metadata or self.backend.data.get_metadata(project_code, [], "")).freq_setup.n_channels
        chanbin = int(chanbin)
        if chanbin > n_channels:
            logger.info("split: {} has {} channel(s) per subband, fewer than chanbin={}; keeping them all",
                        source, n_channels, chanbin)
            chanbin = 1
        width = n_channels if chanbin == -1 else max(chanbin, 1)
        try:
            split(ms, target, source, column=column, chanbin=width, timebin=seconds)
        except (RuntimeError, ValueError) as exc:
            raise BackendError(f"{project_code}: split of {source!r} failed: {exc}") from exc
        try:
            from ..interactive import snapshot_flags
            snapshot_flags(str(target))      # reference for flag.from_split after difmapy editing
        except Exception as exc:  # noqa: BLE001 - the snapshot is a convenience, not a product
            logger.debug("could not snapshot the flags of {}: {}", target, exc)
        return str(target)


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
    flag_ops = DaskMsFlagOps
    export_ops = DaskMsExportOps

    def __init__(self, work_dir: str = ".") -> None:
        try:
            import daskms  # noqa: F401
        except ImportError as exc:
            raise BackendError("the dask-ms backend requires dask-ms: "
                               "pip install vlbipy[daskms]") from exc
        super().__init__(work_dir)
        self.tasks = NativeFlagTasks(self.tasks)

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
