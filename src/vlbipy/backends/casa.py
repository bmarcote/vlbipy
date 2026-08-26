"""CASA backend: import, inspection, a-priori calibration and diagnostics.

Implements the :mod:`~vlbipy.backends.base` components with casatools/casatasks.
Recycled from ``casa_pipeline`` (``Importing.evn_fitsidi``,
``Project.get_metadata_from_ms`` and the scan/antenna queries) with two changes:
metadata is read in a *single* msmd session instead of one open/close per query,
and the expensive per-visibility subband check is a separate, explicit call.

casatools/casatasks are imported lazily so the rest of vlbipy works without a
CASA installation; constructing this backend without them raises a clear
:class:`~vlbipy.errors.BackendError`.
"""
from __future__ import annotations

import datetime as dt
import math
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import BackendError
from ..logging_utils import get_logger, warnings
from ..models import (Antenna, CalTable, FreqSetup, ObsMetadata, Scan, ScanSNRSurvey, Stokes)
from ..tools import fetch_eop_file, mjdsec2datetime, space_available_gb
from .base import Backend, CalibrationOps, DataOps, ExportOps, FlagOps, PlotOps

logger = get_logger()

#: A-priori caltable name suffix and apply-interpolation per calibration type.
APRIORI_TABLE_SPECS = {"tsys": (".tsys", "nearest"), "gc": (".gcal", "nearest"),
                       "eop": (".eop", "linear")}

#: SNR value CASA writes for the reference antenna's own (trivial) solution.
_REFANT_SNR_SENTINEL = 999.0

#: Parallel-hand polarization pairs, by correlation basis.
_PARALLEL_HANDS = {Stokes.RR: "RR", Stokes.LL: "LL", Stokes.XX: "XX", Stokes.YY: "YY"}

#: Re-exported for backwards compatibility; the ordering now lives in vlbipy.selection.
from ..selection import REFANT_PRIORITY  # noqa: E402,F401

#: TaQL condition selecting the rows any flagging statistic is measured over. Autocorrelations
#: are never imaged, so flagging them says nothing about data quality.
_OBSERVABLE_ROWS = "ANTENNA1 != ANTENNA2"


def count_query_text(ms: str, column: str, *, where: str = "", groupby: str = "") -> str:
    """Build the TaQL that counts flagged and observable visibilities.

    Observable means a cross-correlation whose data is not exactly zero: zeros
    were never recorded (a station absent from a scan, or a subband it did not
    observe), so counting them would report the array's schedule rather than the
    quality of the data.

    Parameters
    ----------
    ms : str
        Path to the measurement set.
    column : str
        Data column deciding whether a visibility was recorded (``DATA``).
    where : str, optional
        Extra TaQL condition. It is parenthesised before being ANDed on, because
        TaQL binds ``AND`` tighter than ``OR``: a bare ``"A OR B"`` would
        otherwise re-admit the autocorrelations this statistic must exclude.
    groupby : str, optional
        Column to group by (e.g. ``DATA_DESC_ID``, ``ANTENNA1``).

    Returns
    -------
    str
        The query, selecting ``NFLAGGED`` and ``NOBSERVABLE`` (plus the grouping
        column when grouped).
    """
    condition = _OBSERVABLE_ROWS + (f" AND ({where})" if where else "")
    selected = f"{groupby}, " if groupby else ""
    return (f"SELECT {selected}gntrue(FLAG && {column} != 0) AS NFLAGGED, "
            f"gntrue({column} != 0) AS NOBSERVABLE FROM {ms} WHERE {condition}"
            + (f" GROUPBY {groupby}" if groupby else ""))


def parallel_hand_indices(metadata: ObsMetadata) -> tuple[list[int], list[str]]:
    """Return the indices and labels of the parallel-hand correlations.

    An MS stores correlations in their own order (typically RR, RL, LR, LL), so
    the parallel hands are not the first N entries. Filtering the label list
    without filtering the data axis renames the cross-hands — RL gets plotted as
    "LL" — which makes a perfectly calibrated dataset look broken.

    Returns
    -------
    tuple
        ``([indices into the correlation axis], [labels])``; falls back to every
        correlation when none is recognised as a parallel hand.
    """
    pairs = [(i, _PARALLEL_HANDS[p]) for i, p in enumerate(metadata.freq_setup.polarizations)
             if p in _PARALLEL_HANDS]
    if not pairs:
        return (list(range(len(metadata.freq_setup.polarizations))),
                [str(p.name) for p in metadata.freq_setup.polarizations])
    return [i for i, _ in pairs], [label for _, label in pairs]


def polarization_labels(metadata: ObsMetadata, n_pol: int) -> list[str]:
    """Return display labels for the parallel-hand polarizations, in solution order.

    Parameters
    ----------
    metadata : ObsMetadata
        Supplies the correlation products actually present.
    n_pol : int
        Number of polarizations to label; unnamed extras become ``P3``, ``P4``, ...
    """
    labels = [_PARALLEL_HANDS[p] for p in metadata.freq_setup.polarizations
              if p in _PARALLEL_HANDS]
    if len(labels) >= n_pol:
        return labels[:n_pol]
    return labels + [f"P{i + 1}" for i in range(len(labels), n_pol)]


def central_channel_selection(n_channels: int, fraction: float) -> str:
    """Return a CASA spw selection string for the central ``fraction`` of channels.

    Parameters
    ----------
    n_channels : int
        Channels per subband.
    fraction : float
        Fraction of channels to keep (1.0 or less than 2 channels -> all channels).

    Returns
    -------
    str
        e.g. ``"*:3~28"`` for 32 channels at 0.8, or ``"*"`` when nothing is trimmed.
    """
    if fraction >= 1.0 or n_channels < 4:
        return "*"
    n_edge = int(round(n_channels * (1.0 - fraction) / 2.0))
    if n_edge < 1:
        return "*"
    return f"*:{n_edge}~{n_channels - 1 - n_edge}"


class CasaDataOps(DataOps):
    """Import FITS-IDI/UVFITS into a measurement set and read its metadata."""

    def is_imported(self, project_code: str) -> bool:
        """Return True if the project's measurement set already exists."""
        return self.backend.ms_path(project_code).is_dir()

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False,
                    mms: bool = True, **kwargs) -> None:
        """Import FITS-IDI files via importfitsidi, then partition into a Multi-MS.

        The Multi-MS (``<code>.mms``, one sub-MS per scan/spw chunk) enables
        parallel I/O and is transparent to every CASA task and tool. A plain MS
        already present is upgraded in place (partitioned, then removed) without
        re-importing the FITS-IDI files.

        Parameters
        ----------
        project_code : str
            Project code (names the output MS/MMS).
        source_names : list of str
            Unused here (per-source split happens at export time).
        scan_gap : int
            Gap in seconds that defines a new scan boundary (scanreindexgap_s).
        files : list of str
            The FITS-IDI files, in order (required unless a plain MS already exists).
        delete : bool
            Overwrite an existing MS/MMS instead of skipping.
        mms : bool
            Partition into a Multi-MS (default True; ``[import].mms``).
        """
        work_dir = self.work_dir
        target = work_dir / (f"{project_code}.mms" if mms else f"{project_code}.ms")
        plain_ms = work_dir / f"{project_code}.ms"
        if target.exists():
            if not delete:
                logger.info("{} already exists; skipping import (use force=True to redo)", target)
                return
            logger.warning("removing existing {}", target)
            shutil.rmtree(target)
            if delete and mms and plain_ms.exists():
                shutil.rmtree(plain_ms)

        if not plain_ms.exists():
            if not files:
                raise BackendError(f"{project_code}: no FITS-IDI files provided to import")
            idi_size_gb = sum(Path(f).stat().st_size for f in files) / 1e9
            needed = (4.0 if mms else 2.5) * idi_size_gb  # partition transiently doubles the MS
            if space_available_gb(work_dir) < needed:
                raise BackendError(f"not enough disk space in {work_dir} to create the "
                                   f"{'MMS' if mms else 'MS'} (~{needed:.0f} GB needed)")
            logger.info("importfitsidi: {} file(s) -> {}", len(files), plain_ms)
            self.backend.tasks.importfitsidi(vis=str(plain_ms), fitsidifile=[str(f) for f in files],
                                             constobsid=True, scanreindexgap_s=float(scan_gap),
                                             specframe="GEO")
        elif mms:
            logger.info("found existing plain MS {}; partitioning it into a Multi-MS", plain_ms.name)

        if mms:
            logger.info("partition: {} -> {} (Multi-MS for parallel I/O)", plain_ms.name, target.name)
            self.backend.tasks.partition(vis=str(plain_ms), outputvis=str(target), createmms=True,
                                         separationaxis="auto", numsubms="auto", flagbackup=False,
                                         datacolumn="all")
            shutil.rmtree(plain_ms)
        logger.info("created {}", target)

    def import_uvfits(self, project_code: str, uvfits: str, *, delete: bool = False) -> None:
        """Import a UVFITS file into a measurement set, fixing AIPS numeric antenna names."""
        ms = self.backend.ms_path(project_code)
        if ms.exists():
            if not delete:
                logger.info("MS {} already exists; skipping import (use force=True to redo)", ms)
                return
            shutil.rmtree(ms)
        logger.info("importuvfits: {} -> {}", uvfits, ms)
        self.backend.tasks.importuvfits(fitsfile=str(uvfits), vis=str(ms), antnamescheme="new")
        self._fix_antenna_names(ms)

    def adopt_ms(self, project_code: str, ms: str) -> None:
        """Register an already-existing measurement set as this project's MS."""
        ms_path = Path(ms)
        if not ms_path.is_dir():
            raise BackendError(f"measurement set not found: {ms_path}")
        self.backend.register_ms(project_code, ms_path)
        logger.info("adopted existing MS {} for {}", ms_path, project_code)

    def reset_calibration(self, project_code: str, *, unflag: bool = True,
                          backup_flags: bool = True) -> dict:
        """Return an already-imported measurement set to its as-correlated state.

        Runs ``clearcal`` (resetting CORRECTED_DATA to DATA and dropping MODEL_DATA)
        and, unless disabled, ``flagdata(mode='unflag')``. Flags only ever
        accumulate across runs, so without this a re-run inherits every flag a
        previous attempt made and works on a progressively smaller array.

        The correlator flags that came in with the FITS-IDI are cleared too; the
        pipeline puts them back in the a-priori flagging step, which is the one
        place that decides what is bad.

        Parameters
        ----------
        project_code : str
            Project code.
        unflag : bool
            Clear every flag (``[import].unflag_existing``).
        backup_flags : bool
            Save a ``flagmanager`` version first (``[import].backup_flags``), so
            hand-made flags survive as a restorable version.

        Returns
        -------
        dict
            ``flagged_before`` / ``flagged_after`` fractions and ``backup``.
        """
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"{project_code}: measurement set {ms} not found")
        before = self.backend.flag.flagged_fraction(project_code)
        backup = ""
        if unflag and backup_flags:
            backup = f"{project_code}_before_reset"
            try:
                self.backend.tasks.flagmanager(vis=str(ms), mode="delete", versionname=backup)
            except RuntimeError:
                pass  # no previous backup of this name; nothing to replace
            self.backend.tasks.flagmanager(vis=str(ms), mode="save", versionname=backup)
            logger.info("saved current flags as version {!r} before resetting", backup)
        logger.info("clearcal: resetting the corrected data of {}", ms.name)
        try:
            self.backend.tasks.clearcal(vis=str(ms), addmodel=False)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: clearcal failed: {exc}") from exc
        if unflag:
            try:
                self.backend.tasks.flagdata(vis=str(ms), mode="unflag", flagbackup=False)
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: unflag failed: {exc}") from exc
        after = self.backend.flag.flagged_fraction(project_code)
        logger.info("reset {}: flagged {:.1%} -> {:.1%}; corrected data cleared{}",
                    ms.name, before, after,
                    f" (previous flags kept as {backup!r})" if backup else "")
        return {"flagged_before": before, "flagged_after": after, "backup": backup}

    def _fix_antenna_names(self, ms: Path) -> None:
        """Restore antenna names from the STATION column when AIPS left numbers in NAME."""
        table = self.backend.tools.table()
        if not table.open(str(ms / "ANTENNA"), nomodify=False):
            raise BackendError(f"could not open {ms}/ANTENNA")
        try:
            names = table.getcol("NAME")
            if len(names) and str(names[0]).isnumeric():
                table.putcol("NAME", list(table.getcol("STATION")))
                logger.info("restored antenna names from STATION column in {}", ms)
        finally:
            table.close()

    # -- metadata --
    def get_metadata(self, project_code: str, source_names: list[str], observatory: str) -> ObsMetadata:
        """Read the full observation metadata from the measurement set (one msmd session).

        Reads antennas (position, diameter, mount, scan count), scans (source,
        participating antennas, subbands, integration time), the frequency setup
        and the sources that actually have data. Array geometry (baseline lengths,
        resolution) is derived from this by :class:`~vlbipy.models.ObsMetadata`.
        """
        ms = self.backend.ms_path(project_code)
        msmd = self.backend.tools.msmetadata()
        if not msmd.open(str(ms)):
            raise BackendError(f"could not open MS {ms}")
        try:
            antenna_names = list(msmd.antennanames())
            telescope = msmd.observatorynames()[0] if msmd.observatorynames() else ""
            if telescope and observatory and telescope.upper() != observatory.upper():
                logger.warning("observatory mismatch: config says {} but the MS says {}",
                               observatory, telescope)

            field_names = list(msmd.fieldnames())
            source_coords: dict[str, tuple[float, float]] = {}
            for i, name in enumerate(field_names):
                center = msmd.phasecenter(i)
                to_deg = (180.0 / math.pi) if center["m0"].get("unit", "rad") == "rad" else 1.0
                ra_deg = float(center["m0"]["value"]) * to_deg
                dec_deg = float(center["m1"]["value"]) * to_deg
                source_coords[name] = (ra_deg % 360.0, dec_deg)

            timerange = msmd.timerangeforobs(0)
            t_start = float(timerange["begin"]["m0"]["value"]) * 86400.0
            t_end = float(timerange["end"]["m0"]["value"]) * 86400.0
            obs_date = (dt.datetime(1858, 11, 17) + dt.timedelta(seconds=t_start)).date()

            n_spw = int(msmd.nspw())
            mean_freq = (msmd.meanfreq(0) + msmd.meanfreq(n_spw - 1)) / 2.0
            bandwidths = msmd.bandwidths()
            freq = FreqSetup(ref_freq=float(mean_freq),
                             total_bandwidth=float(sum(bandwidths)) if hasattr(bandwidths, "__len__")
                             else float(bandwidths) * n_spw,
                             n_subbands=n_spw, n_channels=int(msmd.nchan(0)),
                             channel_width=float(abs(msmd.chanwidths(0)[0])),
                             polarizations=[Stokes(int(c)) for c in msmd.corrtypesforpol(0)
                                            if 0 <= int(c) <= 12],
                             channel_freqs=[[float(f) for f in msmd.chanfreqs(s)]
                                            for s in range(n_spw)])

            scans: list[Scan] = []
            observed: set[str] = set()
            observed_fields: set[int] = set()
            scan_counts: dict[str, int] = {}
            for scan_number in msmd.scannumbers():
                scan_fields = msmd.fieldsforscan(scan_number)
                observed_fields.update(int(f) for f in scan_fields)
                source = field_names[int(scan_fields[0])] if len(scan_fields) else ""
                times = msmd.timesforscan(scan_number)
                ants = [antenna_names[a] for a in msmd.antennasforscan(scan_number)]
                observed.update(ants)
                for ant in ants:
                    scan_counts[ant] = scan_counts.get(ant, 0) + 1
                scans.append(Scan(scan_number=int(scan_number), source=source,
                                  time_start=float(times.min()) if len(times) else 0.0,
                                  time_end=float(times.max()) if len(times) else 0.0,
                                  antennas=ants,
                                  integration_time=self._exposure_time(msmd, scan_number),
                                  subbands=tuple(int(s) for s in msmd.spwsforscan(scan_number))))
        finally:
            msmd.close()

        # The FIELD table lists every scheduled source; keep only those with actual data.
        source_ids = {field_names[i]: i for i in sorted(observed_fields) if i < len(field_names)}
        source_names = list(source_ids)
        source_coords = {name: source_coords[name] for name in source_names if name in source_coords}
        if len(source_names) < len(field_names):
            logger.info("FIELD table lists {} sources but only {} have data; keeping those",
                        len(field_names), len(source_names))

        antennas = self._read_antenna_table(ms, antenna_names, observed, scan_counts)
        meta = ObsMetadata(project_code=project_code, obs_date=obs_date, time_range=(t_start, t_end),
                           antennas=antennas, scans=scans, freq_setup=freq,
                           source_names=source_names, source_coords=source_coords,
                           source_ids=source_ids)
        logger.info("metadata from {}: {} antennas ({} observed), {} scans, {} sources, {:.3f} GHz",
                    ms.name, len(antennas), len(observed), len(scans), len(source_names),
                    freq.freq_ghz)
        logger.info("array: longest baseline {:.0f} km -> resolution ~{:.2f} mas; "
                    "shortest {:.0f} km -> largest scale ~{:.1f} mas",
                    meta.max_baseline / 1e3, meta.resolution_mas,
                    meta.min_baseline / 1e3, meta.largest_angular_scale_mas)
        return meta

    def _exposure_time(self, msmd, scan_number: int) -> float:
        """Return the correlator integration time of a scan in seconds (0.0 if unavailable)."""
        try:
            return float(msmd.exposuretime(scan=int(scan_number), spwid=0)["value"])
        except (RuntimeError, KeyError, TypeError, IndexError):
            return 0.0

    def _read_antenna_table(self, ms: Path, antenna_names: list[str], observed: set[str],
                            scan_counts: dict[str, int]) -> dict[str, Antenna]:
        """Read positions/diameters/mounts from the ANTENNA subtable into Antenna objects."""
        antennas: dict[str, Antenna] = {}
        table = self.backend.tools.table()
        stations, positions, diameters, mounts = [], [], [], []
        if table.open(str(ms / "ANTENNA")):
            try:
                stations = list(table.getcol("STATION"))
                positions = table.getcol("POSITION").T  # casatools returns (3, n)
                diameters = list(table.getcol("DISH_DIAMETER"))
                mounts = list(table.getcol("MOUNT"))
            finally:
                table.close()
        for i, name in enumerate(antenna_names):
            antennas[name] = Antenna(
                name=name, fullname=str(stations[i]) if i < len(stations) else "",
                diameter=float(diameters[i]) if i < len(diameters) else 0.0,
                position=tuple(float(v) for v in positions[i]) if i < len(positions) else (0.0, 0.0, 0.0),
                observed=name in observed, mount=str(mounts[i]) if i < len(mounts) else "",
                n_scans=scan_counts.get(name, 0))
        return antennas

    def get_subband_participation(self, project_code: str,
                                  antenna_names: list[str]) -> dict[str, tuple[int, ...]]:
        """Return ``{antenna: (subbands with unflagged data, ...)}`` via one TaQL pass.

        Heterogeneous arrays record different subband subsets per antenna. This
        reads the FLAG column, so it is the expensive part of inspection and is
        deliberately kept out of :meth:`get_metadata`.
        """
        ms = self.backend.ms_path(project_code)
        logger.info("inspecting subband participation ({} antennas; reads the FLAG column)",
                    len(antenna_names))
        participation: dict[str, set[int]] = {name: set() for name in antenna_names}
        table = self.backend.tools.table()
        if not table.open(str(ms)):
            raise BackendError(f"could not open MS {ms}")
        try:
            # An MS stores each baseline once, with ANTENNA1 < ANTENNA2, so an antenna's data
            # is split across both columns: query each in turn or the highest-numbered
            # antennas look empty.
            for column in ("ANTENNA1", "ANTENNA2"):
                query = (f"SELECT {column}, DATA_DESC_ID, gntrue(!FLAG) AS NVALID FROM {ms} "
                         f"WHERE ANTENNA1 != ANTENNA2 GROUPBY {column}, DATA_DESC_ID")
                try:
                    result = table.taql(query)
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: subband participation query failed: "
                                       f"{exc}") from exc
                # A GROUPBY result is a calculated table: getcol() raises "Unknown casa
                # DataType", so read it cell by cell. One row per (antenna, subband).
                try:
                    for i in range(result.nrows()):
                        antenna_id = int(result.getcell(column, i))
                        if int(result.getcell("NVALID", i)) <= 0 or antenna_id >= len(antenna_names):
                            continue
                        participation[antenna_names[antenna_id]].add(
                            int(result.getcell("DATA_DESC_ID", i)))
                finally:
                    result.close()
        finally:
            table.close()
        result_map = {name: tuple(sorted(spws)) for name, spws in participation.items()}
        for name, spws in result_map.items():
            if not spws:
                logger.warning("antenna {} has no unflagged data in any subband", name)
        return result_map

    def read_spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                      refant: str = "", column: str = "corrected", all_pols: bool = False,
                      metadata: Optional[ObsMetadata] = None) -> dict:
        """Read time-averaged visibility spectra on baselines to the reference antenna.

        Vector-averages every baseline over time within the selection, one
        subband at a time so a whole scan never has to sit in memory at once.
        Vector (not scalar) averaging is the point: it keeps the phase, so a
        residual delay shows as a slope across the band instead of averaging away.

        Parameters
        ----------
        field : str
            Field selection (comma-separated names; empty = all).
        scans : list, optional
            Scan numbers to average over (default: all scans on ``field``).
        refant : str
            Reference antenna; only baselines including it are returned.
        column : str
            ``"corrected"`` (post-applycal) or ``"data"`` (raw).

        Returns
        -------
        dict
            ``antennas`` (the other end of each baseline), ``spectra``
            (``{antenna: complex array (n_spw, n_chan, n_pol)}``), ``refant``,
            ``n_spw``, ``n_channels``, ``polarizations``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        refant = refant or self.backend.calibrate.refant_chain(meta, "").split(",")[0]
        antenna_names = list(meta.antennas)
        if refant not in antenna_names:
            raise BackendError(f"{project_code}: reference antenna {refant!r} is not in the array")
        refant_id = antenna_names.index(refant)
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        column_name = {"corrected": "corrected_data", "data": "data", "model": "model_data"}[column]
        if all_pols:
            # Full Stokes is only meaningful on the raw data, where the cross-hands
            # still carry the instrumental polarization signature.
            pol_indices = list(range(len(meta.freq_setup.polarizations)))
            pol_labels = [p.name for p in meta.freq_setup.polarizations]
        else:
            pol_indices, pol_labels = parallel_hand_indices(meta)

        accumulated: dict[int, np.ndarray] = {}
        counts: dict[int, np.ndarray] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(n_spw):
                ms_tool.selectinit(datadescid=spw)
                selection: dict = {"baseline": f"{refant}&*"}
                if field:
                    selection["field"] = [f for f in field.split(",") if f]
                if scans:
                    selection["scan_number"] = [int(s) for s in scans]
                try:
                    ms_tool.select(selection)
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: selecting {selection} failed: {exc}") from exc
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1", "antenna2"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read the {column} column "
                                       f"(has applycal run?): {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record["flag"], dtype=bool)
                values = np.where(flags, np.nan, values)          # (npol, nchan, nrow)
                # Cross-hands carry no useful phase on an unpolarized calibrator and
                # would dominate the plot with noise; keep the parallel hands only.
                values = values[pol_indices, :, :]
                ant1 = np.asarray(record["antenna1"])
                ant2 = np.asarray(record["antenna2"])
                other = np.where(ant1 == refant_id, ant2, ant1)
                for antenna_id in np.unique(other):
                    if int(antenna_id) == refant_id:
                        continue                                   # autocorrelation
                    rows = other == antenna_id
                    chunk = values[:, :, rows]
                    with np.errstate(invalid="ignore"):
                        summed = np.nansum(chunk, axis=2).T        # -> (nchan, npol)
                        valid = np.sum(np.isfinite(chunk), axis=2).T
                    key = int(antenna_id)
                    if key not in accumulated:
                        accumulated[key] = np.zeros((n_spw, n_chan, summed.shape[1]), dtype=complex)
                        counts[key] = np.zeros((n_spw, n_chan, summed.shape[1]), dtype=float)
                    accumulated[key][spw, :summed.shape[0]] += summed
                    counts[key][spw, :valid.shape[0]] += valid
        finally:
            ms_tool.close()

        spectra = {}
        for antenna_id, total in accumulated.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                spectra[antenna_names[antenna_id]] = np.where(counts[antenna_id] > 0,
                                                              total / counts[antenna_id], np.nan)
        logger.info("read_spectrum: {} baseline(s) to {} from the {} column ({} x {} channels)",
                    len(spectra), refant, column, n_spw, n_chan)
        return {"antennas": sorted(spectra), "spectra": spectra, "refant": refant,
                "n_spw": n_spw, "n_channels": n_chan, "polarizations": pol_labels,
                "frequencies_ghz": [meta.freq_setup.frequencies_ghz(s) for s in range(n_spw)],
                "column": column, "field": field, "scans": list(scans or [])}

    def read_dynamic_spectra(self, project_code: str, *, field: str = "",
                             column: str = "corrected", max_time_bins: int = 200,
                             stokes_i: bool = True,
                             metadata: Optional[ObsMetadata] = None) -> dict:
        """Read a time x frequency array for every antenna pair.

        Reads one subband at a time and bins in time on the fly, so a whole
        observation never sits in memory: at full resolution every baseline of a
        long phase-calibrator track would be gigabytes, while the binned result
        is a few tens of megabytes and shows the same structure.

        Parameters
        ----------
        field : str
            Field to read (normally the phase calibrator).
        column : str
            ``"corrected"`` (post-applycal) or ``"data"``.
        max_time_bins : int
            Time resolution of the output grid.
        stokes_i : bool
            Average the parallel hands into Stokes I; otherwise keep them separate.

        Returns
        -------
        dict
            ``baselines`` (``{(a, b): complex array (n_time, n_freq)}``),
            ``antennas``, ``times`` (MJD seconds, bin centres), ``frequencies_ghz``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        wanted = [f for f in field.split(",") if f]
        scans = [s for s in meta.scans if not wanted or s.source in wanted]
        if not scans:
            raise BackendError(f"{project_code}: no scans on {field!r}")
        t_start = min(s.time_start for s in scans)
        t_end = max(s.time_end for s in scans)
        n_time = max(1, int(max_time_bins))
        edges = np.linspace(t_start, t_end, n_time + 1)
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        n_freq = n_spw * n_chan
        pol_indices, pol_labels = parallel_hand_indices(meta)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        antenna_names = list(meta.antennas)

        sums: dict[tuple[str, str], np.ndarray] = {}
        counts: dict[tuple[str, str], np.ndarray] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(n_spw):
                ms_tool.selectinit(datadescid=spw)
                selection: dict = {}
                if wanted:
                    selection["field"] = wanted
                if selection:
                    ms_tool.select(selection)
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1", "antenna2", "time"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record["flag"], dtype=bool)
                values = np.where(flags, np.nan, values)[pol_indices, :, :]
                if stokes_i:                       # I = (RR + LL) / 2
                    values = np.nanmean(values, axis=0, keepdims=True)
                ant1 = np.asarray(record["antenna1"])
                ant2 = np.asarray(record["antenna2"])
                bins = np.clip(np.digitize(np.asarray(record["time"]), edges) - 1, 0, n_time - 1)
                pairs = np.stack([np.minimum(ant1, ant2), np.maximum(ant1, ant2)], axis=1)
                for pair in np.unique(pairs, axis=0):
                    a, b = int(pair[0]), int(pair[1])
                    if a == b or a >= len(antenna_names) or b >= len(antenna_names):
                        continue
                    rows = (pairs[:, 0] == a) & (pairs[:, 1] == b)
                    key = (antenna_names[a], antenna_names[b])
                    if key not in sums:
                        sums[key] = np.zeros((n_time, n_freq), dtype=complex)
                        counts[key] = np.zeros((n_time, n_freq), dtype=float)
                    chunk = values[0, :, rows]          # -> (n_rows, n_chan)
                    slot = slice(spw * n_chan, spw * n_chan + chunk.shape[1])
                    # Accumulate per time bin: the rows of a bin are averaged together.
                    for bin_index in np.unique(bins[rows]):
                        in_bin = bins[rows] == bin_index
                        block = chunk[in_bin]
                        ok = np.isfinite(block)
                        sums[key][bin_index, slot] += np.where(ok, block, 0.0).sum(axis=0)
                        counts[key][bin_index, slot] += ok.sum(axis=0)
        finally:
            ms_tool.close()

        spectra = {}
        for key, total in sums.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                spectra[key] = np.where(counts[key] > 0, total / np.maximum(counts[key], 1), np.nan)
        centres = (edges[:-1] + edges[1:]) / 2.0
        present = sorted({name for pair in spectra for name in pair},
                         key=lambda n: antenna_names.index(n))
        logger.info("read_dynamic_spectra: {} baseline(s) over {} time bins x {} channels ({})",
                    len(spectra), n_time, n_freq, "Stokes I" if stokes_i else ",".join(pol_labels))
        return {"baselines": spectra, "antennas": present, "times": centres.tolist(),
                "frequencies_ghz": meta.freq_setup.frequencies_ghz(), "field": field,
                "column": column, "stokes_i": stokes_i}

    def read_timeseries(self, project_code: str, *, field: str = "", refant: str = "",
                        column: str = "corrected", max_time_bins: int = 300,
                        metadata: Optional[ObsMetadata] = None) -> dict:
        """Read amplitude/phase vs time per baseline, averaged over frequency.

        The time companion to :meth:`read_spectrum`: same baselines to the
        reference antenna, but collapsed along frequency instead of time, and
        keeping the polarizations separate. Vector-averaged over channels, so a
        residual delay reduces the amplitude rather than being hidden.

        Parameters
        ----------
        field : str
            Field selection (empty = all).
        refant : str
            Reference antenna; only baselines including it are returned.
        max_time_bins : int
            Number of time bins spanning the selection.

        Returns
        -------
        dict
            ``baselines`` (``{name: complex array (n_time, n_pol)}``), ``times``
            (MJD seconds), ``polarizations``, ``refant``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        refant = refant or self.backend.calibrate.refant_chain(meta, "").split(",")[0]
        antenna_names = list(meta.antennas)
        if refant not in antenna_names:
            raise BackendError(f"{project_code}: reference antenna {refant!r} is not in the array")
        refant_id = antenna_names.index(refant)
        wanted = [f for f in field.split(",") if f]
        scans = [s for s in meta.scans if not wanted or s.source in wanted]
        if not scans:
            raise BackendError(f"{project_code}: no scans on {field!r}")
        # Bin *within* scans. A global time grid puts a bin across each scan boundary,
        # and vector-averaging over the minutes-long gap there collapses the amplitude —
        # which draws a dip at both ends of every scan that is an artefact of the binning,
        # not a property of the data.
        span = max(s.time_end for s in scans) - min(s.time_start for s in scans)
        bin_width = max(span / max(int(max_time_bins), 1), 1.0)
        edges: list[float] = []
        for scan in sorted(scans, key=lambda s: s.time_start):
            n_scan_bins = max(1, int(np.ceil(scan.duration_sec / bin_width)))
            edges.extend(np.linspace(scan.time_start, scan.time_end, n_scan_bins + 1)[:-1])
            edges.append(scan.time_end)
        edges = np.asarray(edges, dtype=float)
        n_time = len(edges) - 1
        pol_indices, pol_labels = parallel_hand_indices(meta)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, np.ndarray] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(meta.freq_setup.n_subbands):
                ms_tool.selectinit(datadescid=spw)
                selection: dict = {"baseline": f"{refant}&*"}
                if wanted:
                    selection["field"] = wanted
                ms_tool.select(selection)
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1", "antenna2", "time"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record["flag"], dtype=bool)
                values = np.where(flags, np.nan, values)[pol_indices, :, :]
                ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])
                other = np.where(ant1 == refant_id, ant2, ant1)
                bins = np.clip(np.digitize(np.asarray(record["time"]), edges) - 1, 0, n_time - 1)
                for antenna_id in np.unique(other):
                    if int(antenna_id) == refant_id or int(antenna_id) >= len(antenna_names):
                        continue
                    rows = other == antenna_id
                    name = f"{refant}-{antenna_names[int(antenna_id)]}"
                    if name not in sums:
                        sums[name] = np.zeros((n_time, len(pol_indices)), dtype=complex)
                        counts[name] = np.zeros((n_time, len(pol_indices)), dtype=float)
                    chunk = values[:, :, rows]                       # (n_pol, n_chan, n_rows)
                    for bin_index in np.unique(bins[rows]):
                        block = chunk[:, :, bins[rows] == bin_index]
                        ok = np.isfinite(block)
                        sums[name][bin_index] += np.where(ok, block, 0.0).sum(axis=(1, 2))
                        counts[name][bin_index] += ok.sum(axis=(1, 2))
        finally:
            ms_tool.close()

        baselines = {}
        for name, total in sums.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                baselines[name] = np.where(counts[name] > 0, total / np.maximum(counts[name], 1),
                                           np.nan)
        centres = (edges[:-1] + edges[1:]) / 2.0
        logger.info("read_timeseries: {} baseline(s) to {} over {} time bins",
                    len(baselines), refant, n_time)
        return {"baselines": dict(sorted(baselines.items())), "times": centres.tolist(),
                "polarizations": pol_labels, "refant": refant, "field": field, "column": column}

    def read_uvdistance(self, project_code: str, *, field: str = "", column: str = "corrected",
                        time_bin: float = 10.0, metadata: Optional[ObsMetadata] = None) -> dict:
        """Read visibilities against uv distance, averaged per subband and in time.

        Channels are averaged within each subband and samples within
        ``time_bin`` seconds are averaged per baseline — both vector averages, so
        a residual delay or phase drift reduces the amplitude instead of being
        hidden. Subbands stay separate because each has its own sky frequency,
        and uv distance is measured in wavelengths: the same physical baseline
        sits at a different uv distance in each subband.

        Parameters
        ----------
        field : str
            Field to read (one source).
        column : str
            ``"corrected"`` (post-applycal) or ``"data"``.
        time_bin : float
            Averaging interval in seconds.

        Returns
        -------
        dict
            ``uvdist_mlambda`` (1-D), ``values`` (``(n_points, n_pol)`` complex),
            ``polarizations``, ``field``, ``column``, ``time_bin``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        wanted = [f for f in field.split(",") if f]
        pol_indices, pol_labels = parallel_hand_indices(meta)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        light_speed = 299792458.0

        distances: list[np.ndarray] = []
        averaged: list[np.ndarray] = []
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(meta.freq_setup.n_subbands):
                ms_tool.selectinit(datadescid=spw)
                if wanted:
                    ms_tool.select({"field": wanted})
                try:
                    record = ms_tool.getdata([column_name, "flag", "uvw", "time",
                                              "antenna1", "antenna2"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record["flag"], dtype=bool)
                values = np.where(flags, np.nan, values)[pol_indices, :, :]
                with np.errstate(invalid="ignore"):
                    per_row = np.nanmean(values, axis=1).T        # channel average -> (nrow, npol)
                uvw = np.asarray(record["uvw"])                    # (3, nrow)
                times = np.asarray(record["time"], dtype=float)
                ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])

                # Group by baseline and time bin; a bin is one averaged point.
                bins = np.floor((times - times.min()) / max(time_bin, 1e-6)).astype(np.int64)
                keys = (ant1.astype(np.int64) << 40) + (ant2.astype(np.int64) << 24) + bins
                order = np.argsort(keys, kind="stable")
                keys_sorted = keys[order]
                starts = np.flatnonzero(np.r_[True, keys_sorted[1:] != keys_sorted[:-1]])
                groups = np.split(order, starts[1:])

                frequency = float(np.mean(meta.freq_setup.channel_freqs[spw])
                                  if meta.freq_setup.channel_freqs else meta.freq_setup.ref_freq)
                for group in groups:
                    block = per_row[group]
                    finite = np.isfinite(block)
                    if not finite.any():
                        continue
                    with np.errstate(invalid="ignore"):
                        mean_value = np.where(finite, block, 0.0).sum(axis=0) / np.maximum(
                            finite.sum(axis=0), 1)
                    mean_value = np.where(finite.any(axis=0), mean_value, np.nan)
                    u, v = uvw[0, group].mean(), uvw[1, group].mean()
                    distances.append(np.hypot(u, v) * frequency / light_speed / 1e6)
                    averaged.append(mean_value)
        finally:
            ms_tool.close()

        uvdist = np.asarray(distances, dtype=float)
        points = (np.asarray(averaged) if averaged
                  else np.empty((0, len(pol_indices)), dtype=complex))
        logger.info("read_uvdistance: {} averaged point(s) on {} ({} s bins, per subband)",
                    uvdist.size, field or "all fields", time_bin)
        return {"uvdist_mlambda": uvdist, "values": points, "polarizations": pol_labels,
                "field": field, "column": column, "time_bin": time_bin}

    def check_apriori_data(self, project_code: str) -> dict:
        """Report whether the MS carries usable Tsys and gain-curve information.

        ``gencal`` reads these from the MS SYSCAL and GAIN_CURVE subtables, which
        ``importfitsidi`` fills from the FITS-IDI ``SYSTEM_TEMPERATURE`` and
        ``GAIN_CURVE`` tables — themselves usually populated from the ``.antab``
        file before import. Amplitude calibration is meaningless without them, so
        this is checked explicitly before a-priori calibration runs.

        Returns
        -------
        dict
            ``has_tsys`` / ``has_gc`` (bool), ``antennas_with_tsys`` and
            ``antennas_without_tsys`` (lists of names), and ``gc_is_trivial``
            (True when every gain-curve coefficient is 1.0, i.e. a placeholder).
        """
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"{project_code}: measurement set {ms} not found")
        table = self.backend.tools.table()
        table.open(str(ms / "ANTENNA"))
        try:
            antenna_names = [str(n) for n in table.getcol("NAME")]
        finally:
            table.close()

        with_tsys: list[str] = []
        n_syscal = 0
        if (ms / "SYSCAL").is_dir():
            table.open(str(ms / "SYSCAL"))
            try:
                n_syscal = table.nrows()
                if n_syscal:
                    antenna_ids = np.asarray(table.getcol("ANTENNA_ID"))
                    tsys = np.asarray(table.getcol("TSYS"))
                    # TSYS is (npol, nrow) or (npol, nchan, nrow): the row axis is last.
                    valid_per_row = np.any(tsys > 0.0, axis=tuple(range(tsys.ndim - 1)))
                    for index, name in enumerate(antenna_names):
                        if np.any(valid_per_row & (antenna_ids == index)):
                            with_tsys.append(name)
            finally:
                table.close()

        n_gc, gc_trivial = 0, False
        if (ms / "GAIN_CURVE").is_dir():
            table.open(str(ms / "GAIN_CURVE"))
            try:
                n_gc = table.nrows()
                if n_gc:
                    gain = np.asarray(table.getcol("GAIN"))
                    gc_trivial = bool(np.all(gain == 1.0))
            finally:
                table.close()

        report = {"has_tsys": bool(with_tsys), "has_gc": n_gc > 0,
                  "antennas_with_tsys": with_tsys,
                  "antennas_without_tsys": [n for n in antenna_names if n not in with_tsys],
                  "gc_is_trivial": gc_trivial, "n_syscal_rows": n_syscal, "n_gc_rows": n_gc}
        logger.info("a-priori data in {}: SYSCAL {} rows ({}/{} antennas with valid Tsys), "
                    "GAIN_CURVE {} rows{}", ms.name, n_syscal, len(with_tsys), len(antenna_names),
                    n_gc, " (all coefficients 1.0 — placeholder)" if gc_trivial else "")
        return report

    # -- inspection --
    def listobs(self, project_code: str, listfile: Optional[str] = None) -> dict:
        """Run casatasks.listobs on the MS; write to <work_dir>/<code>-listobs.log by default."""
        ms = self.backend.ms_path(project_code)
        listfile = listfile or str(self.work_dir / f"{project_code}-listobs.log")
        Path(listfile).unlink(missing_ok=True)
        result = self.backend.tasks.listobs(vis=str(ms), listfile=listfile, overwrite=True)
        logger.info("listobs written to {}", listfile)
        return result


class CasaCalibrationOps(CalibrationOps):
    """Calibration solves via casatasks (a-priori and the SNR survey so far)."""

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, **kwargs) -> list[CalTable]:
        """A-priori amplitude calibration: gencal Tsys + gain curve (+ EOP where needed).

        Runs ``gencal(caltype='tsys', uniform=False)`` (from the SYSCAL subtable) and
        ``gencal(caltype='gc')`` (from the GAIN_CURVE subtable) for every network, and
        ``gencal(caltype='eop', infile=usno_finals.erp)`` for networks correlated with
        provisional Earth-orientation parameters (VLBA/LBA; not EVN).

        Parameters
        ----------
        project_code : str
            Project code (the MS must exist).
        field : str
            Recorded in the returned CalTables (gencal itself is field-independent).
        needs_eop : bool
            Whether EOP corrections apply (from the observatory handler).
        eop_file : str, optional
            Path to a usno_finals.erp file; auto-downloaded if omitted.

        Returns
        -------
        list of CalTable
            The generated calibration tables, in application order.
        """
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"{project_code}: measurement set {ms} not found; "
                               "run import_data() before a_priori calibration")
        self._verify_apriori_inputs(project_code, antab=kwargs.get("antab"))
        tables: list[CalTable] = []
        paths = self.backend.apriori_table_paths(project_code, needs_eop)

        for cal_type, table_path in paths.items():
            if table_path.exists():
                shutil.rmtree(table_path)
            gencal_kwargs = {"vis": str(ms), "caltable": str(table_path), "caltype": cal_type}
            if cal_type == "tsys":
                gencal_kwargs["uniform"] = False
            if cal_type == "eop":
                gencal_kwargs["infile"] = str(eop_file) if eop_file else str(fetch_eop_file(self.work_dir))
            logger.info("gencal caltype='{}' -> {}", cal_type, table_path.name)
            try:
                self.backend.tasks.gencal(**gencal_kwargs)
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: gencal caltype='{cal_type}' failed: {exc}") from exc
            tables.append(CalTable(cal_type=cal_type, path=str(table_path), field=field,
                                   interp=APRIORI_TABLE_SPECS[cal_type][1]))
        if not needs_eop:
            logger.info("EOP corrections not needed for this network; skipped")
        return tables

    def smooth(self, project_code: str, table: CalTable, *, threshold: float = 6.0,
               max_passes: int = 3, window: int = 5, **kwargs) -> CalTable:
        """De-spike a calibration table in place, per antenna / subband / polarization.

        Each antenna-subband-polarization time series is tested against its own
        median and MAD: points deviating by more than ``threshold`` robust sigmas
        are replaced with the local running median, repeating until a pass finds
        nothing (or ``max_passes``). Values that are non-positive to begin with
        (the -999.9 placeholders correlators write for stations without a
        measurement) cannot be repaired and are flagged instead.

        Parameters
        ----------
        project_code : str
            Project code (for log messages).
        table : CalTable
            The table to clean; modified on disk and returned.
        threshold : float
            Outlier cut in robust sigmas (``[calibration].tsys_outlier_sigma``).
        max_passes : int
            Maximum de-spiking passes (``[calibration].tsys_smooth_passes``).
        window : int
            Running-median window, in solutions, used for replacement values.

        Returns
        -------
        CalTable
            The same table, with ``snr`` left untouched.
        """
        from ..statistics import despike

        path = Path(table.path)
        if not path.is_dir():
            raise BackendError(f"{project_code}: calibration table {path} not found")
        handle = self.backend.tools.table()
        if not handle.open(str(path), nomodify=False):
            raise BackendError(f"could not open {path} for writing")
        try:
            param_column = "FPARAM" if "FPARAM" in handle.colnames() else "CPARAM"
            values = np.asarray(handle.getcol(param_column)).astype(float)  # (npar, nchan, nrow)
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
            antenna_ids = np.asarray(handle.getcol("ANTENNA1"))
            spw_ids = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
            times = np.asarray(handle.getcol("TIME"))

            # Non-positive Tsys is a placeholder, not a measurement: flag, never smooth.
            invalid = (values <= 0.0) & ~flags
            n_invalid = int(invalid.sum())
            flags |= invalid

            n_replaced = 0
            for antenna in np.unique(antenna_ids):
                for spw in np.unique(spw_ids):
                    rows = np.where((antenna_ids == antenna) & (spw_ids == spw))[0]
                    if rows.size < 3:
                        continue
                    order = rows[np.argsort(times[rows])]
                    for par in range(values.shape[0]):
                        for chan in range(values.shape[1]):
                            series = values[par, chan, order]
                            usable = ~flags[par, chan, order]
                            if usable.sum() < 3:
                                continue
                            masked = np.where(usable, series, np.nan)
                            cleaned, replaced = despike(masked, threshold=threshold,
                                                        max_passes=max_passes, window=window)
                            if not replaced.any():
                                continue
                            fixed = replaced & np.isfinite(cleaned)
                            values[par, chan, order[fixed]] = cleaned[fixed]
                            n_replaced += int(fixed.sum())
            if n_replaced or n_invalid:
                handle.putcol(param_column, values)
                handle.putcol("FLAG", flags)
                handle.flush()
        finally:
            handle.close()

        total = values.size
        logger.info("smooth[{}]: flagged {} non-positive and de-spiked {} of {} solutions "
                    "(>{:.0f} sigma MAD)", table.cal_type, n_invalid, n_replaced, total, threshold)
        if n_replaced > 0.1 * total:
            warnings.warn(f"{project_code}: {n_replaced / total:.0%} of the {table.cal_type} "
                          "solutions had to be de-spiked; check the table plot")
        return table

    def initial_calibration(self, project_code: str, field: str, refant: str, *,
                            scans: Optional[list] = None, gaintable: Optional[list] = None,
                            channel_fraction: float = 0.8, solint: str = "inf",
                            minsnr: float = 20.0, metadata: Optional[ObsMetadata] = None,
                            suffix: str = "sbd", **kwargs) -> CalTable:
        """Single-band (instrumental) delay calibration on the selected scan(s).

        Solves one delay per antenna, subband and polarization with the rates
        forced to zero: the instrumental delay is a fixed property of the signal
        path, so letting the rate float only adds noise. Only the central
        ``channel_fraction`` of each subband is used, since the edges roll off
        and have not been bandpass-corrected yet.

        Parameters
        ----------
        project_code : str
            Project code.
        field : str
            Calibrator field(s) to solve on.
        refant : str
            Reference antenna, or a comma-separated fallback chain.
        scans : list, optional
            Scan numbers to restrict the solve to (from the scan selection).
        gaintable : list, optional
            Prior calibration tables to apply on the fly.
        channel_fraction : float
            Fraction of central channels per subband to use.
        solint : str
            Solution interval; ``"inf"`` gives one solution per scan.
        minsnr : float
            Minimum SNR for a solution to be kept.
        suffix : str
            Table name suffix, so the post-bandpass re-run can be kept separately.

        Returns
        -------
        CalTable
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        spw = central_channel_selection(meta.freq_setup.n_channels, channel_fraction)
        table_path = self.backend.caldir() / f"{project_code}.{suffix}"
        params = {"caltable": str(table_path), "field": field, "spw": spw, "solint": solint,
                  "zerorates": True, "refant": self.refant_chain(meta, refant), "minsnr": minsnr,
                  "corrdepflags": True, "parang": True}
        if scans:
            params["scan"] = ",".join(str(s) for s in scans)
        logger.info("initial_calibration[{}]: fringefit field={} scans={} spw={} minsnr={}",
                    suffix, field, params.get("scan", "all"), spw, minsnr)
        self._fringefit(project_code, params, gaintable)
        return CalTable(cal_type=suffix, path=str(table_path), field=field, interp="nearest",
                        snr=self._median_table_snr(table_path))

    def bandpass(self, project_code: str, field: str, refant: str, *,
                 scans: Optional[list] = None, gaintable: Optional[list] = None,
                 solint: str = "inf", combine: str = "scan", minsnr: float = 3.0,
                 solnorm: bool = True, metadata: Optional[ObsMetadata] = None,
                 **kwargs) -> CalTable:
        """Bandpass calibration on the same scan(s) used for the instrumental delay.

        Uses every channel (the point is to measure the band shape, including the
        edges) and normalises the result so the bandpass carries the shape but
        not the flux scale.

        Parameters
        ----------
        combine : str
            Axes combined before solving; ``"scan"`` lets several selected scans
            contribute to one solution.
        solnorm : bool
            Normalise each solution to unit mean amplitude.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        table_path = self.backend.caldir() / f"{project_code}.bpass"
        if table_path.exists():
            shutil.rmtree(table_path)
        params = {"vis": str(self.backend.ms_path(project_code)), "caltable": str(table_path),
                  "field": field, "solint": solint, "combine": combine, "solnorm": solnorm,
                  "refant": self.refant_chain(meta, refant), "minsnr": minsnr, "bandtype": "B",
                  "corrdepflags": True, "parang": True, "fillgaps": 1}
        if scans:
            params["scan"] = ",".join(str(s) for s in scans)
        params.update(self._prior_callib(project_code, gaintable, table_path))
        logger.info("bandpass: field={} scans={} solint={} combine={} (prior: {})",
                    field, params.get("scan", "all"), solint, combine,
                    ", ".join(t.cal_type for t in gaintable or []) or "none")
        try:
            self.backend.tasks.bandpass(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: bandpass failed (field={field!r}): {exc}") from exc
        if not table_path.is_dir():
            raise BackendError(f"{project_code}: bandpass produced no table at {table_path}")
        return CalTable(cal_type="bpass", path=str(table_path), field=field,
                        interp="nearest,nearest", snr=self._median_table_snr(table_path))

    def fringefit(self, project_code: str, field: str, refant: str, *,
                  gaintable: Optional[list] = None, solint: str = "inf", combine: str = "spw",
                  minsnr: float = 5.0, zerorates: bool = False, suffix: str = "mbd",
                  dispersive: bool = False, metadata: Optional[ObsMetadata] = None,
                  **kwargs) -> CalTable:
        """Global (multi-band delay) fringe fit on the calibrators.

        Solves the time-variable phase, delay and delay rate left after the
        instrumental calibration. Combining the subbands (``combine='spw'``)
        pools the whole bandwidth into one solution, which is what makes weak
        calibrators detectable; the rates are solved for (unlike the SBD) because
        these residuals are atmospheric and vary through the observation.

        Because the solutions then live in a single subband, applying them needs
        an spw map pointing every subband at that one — built here from the table
        itself rather than assumed to be subband 0, since it is whichever subband
        the reference antenna actually had.

        Parameters
        ----------
        field : str
            Calibrator field(s) to solve on.
        refant : str
            Reference antenna or fallback chain.
        gaintable : list, optional
            Prior tables (Tsys, gain curve, SBD, bandpass).
        solint : str
            Solution interval; shorter tracks the atmosphere better but needs SNR.
        combine : str
            Axes combined before solving (``"spw"`` for the multi-band delay).
        minsnr : float
            Minimum SNR to keep a solution.
        dispersive : bool
            Also solve for the dispersive (ionospheric) delay. The ionosphere
            delays low frequencies more than high ones, so below roughly 6 GHz
            the residual is not a single non-dispersive delay and fitting one
            leaves a frequency-dependent phase behind. Adds a third free
            parameter, so it needs more SNR than a plain delay+rate solve.

        Returns
        -------
        CalTable
            With ``spwmap`` filled in ready for apply.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        table_path = self.backend.caldir() / f"{project_code}.{suffix}"
        params = {"caltable": str(table_path), "field": field, "solint": solint,
                  "combine": combine, "zerorates": zerorates, "corrdepflags": True,
                  "refant": self.refant_chain(meta, refant), "minsnr": minsnr, "parang": True}
        if dispersive:
            # paramactive = [delay, rate, dispersive delay]; zerorates is separate,
            # it zeroes the fitted rates in the output rather than not fitting them.
            params["paramactive"] = [True, True, True]
        logger.info("fringefit({}): field={} solint={} combine={} minsnr={} dispersive={} "
                    "(prior: {})", suffix, field, solint, combine, minsnr, dispersive,
                    ", ".join(t.cal_type for t in gaintable or []) or "none")
        self._fringefit(project_code, params, gaintable)
        spwmap = self._combined_spwmap(table_path, meta) if "spw" in combine else []
        return CalTable(cal_type=suffix, path=str(table_path), field=field, interp="linear",
                        spwmap=spwmap, snr=self._median_table_snr(table_path))

    def _combined_spwmap(self, table_path: Path, metadata: ObsMetadata) -> list[int]:
        """Return the spw map for a table solved with ``combine='spw'``.

        Every subband must point at the one the solutions were written to. That
        is not always subband 0 — it is the first subband the reference antenna
        had — so it is read back from the table.
        """
        handle = self.backend.tools.table()
        if not handle.open(str(table_path)):
            raise BackendError(f"could not open {table_path} to build the spw map")
        try:
            spw_ids = np.unique(np.asarray(handle.getcol("SPECTRAL_WINDOW_ID")))
        finally:
            handle.close()
        if spw_ids.size != 1:
            logger.info("fringefit: solutions span {} subbands; no spw map needed", spw_ids.size)
            return []
        reference = int(spw_ids[0])
        logger.info("fringefit: solutions stored in subband {}; mapping all {} subbands to it",
                    reference, metadata.freq_setup.n_subbands)
        return [reference] * int(metadata.freq_setup.n_subbands)

    def scalar_bandpass(self, project_code: str, field: str, refant: str, *,
                        gaintable: Optional[list] = None, solint: str = "inf",
                        combine: str = "scan", minsnr: float = 3.0, solnorm: bool = True,
                        metadata: Optional[ObsMetadata] = None, **kwargs) -> CalTable:
        """Solve one amplitude gain per antenna and subband (a "scalar bandpass").

        Even after the bandpass, the subbands of an antenna can sit at slightly
        different amplitude levels: the bandpass is normalised per subband, so it
        removes the *shape* within each one but not a constant offset between
        them. Those steps survive into the combined image as an effective
        mis-weighting of the band.

        Amplitude only (``calmode='a'``), one solution per antenna per subband
        for the whole observation, normalised so the flux scale set by the
        a-priori calibration is preserved rather than re-derived here.

        Returns
        -------
        CalTable
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        table_path = self.backend.caldir() / f"{project_code}.scalar_bp"
        if table_path.exists():
            shutil.rmtree(table_path)
        params = {"vis": str(self.backend.ms_path(project_code)), "caltable": str(table_path),
                  "field": field, "solint": solint, "combine": combine, "gaintype": "G",
                  "calmode": "a", "solnorm": solnorm, "minsnr": minsnr, "parang": True,
                  "refant": self.refant_chain(meta, refant)}
        params.update(self._prior_callib(project_code, gaintable, table_path))
        logger.info("scalar_bandpass: gaincal calmode='a' field={} solint={} combine={} "
                    "(one amplitude per antenna and subband)", field, solint, combine)
        try:
            self.backend.tasks.gaincal(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: scalar bandpass failed (field={field!r}): "
                               f"{exc}") from exc
        if not table_path.is_dir():
            raise BackendError(f"{project_code}: scalar bandpass produced no table")
        spread = self._subband_gain_spread(table_path)
        logger.info("scalar_bandpass: subband-to-subband amplitude spread was {:.1%} "
                    "(median over antennas)", spread)
        return CalTable(cal_type="scalar_bp", path=str(table_path), field=field,
                        interp="nearest", snr=self._median_table_snr(table_path))

    def _subband_gain_spread(self, table_path: Path) -> float:
        """Return the median across antennas of the relative spread of gain across subbands."""
        handle = self.backend.tools.table()
        if not handle.open(str(table_path)):
            return 0.0
        try:
            gains = np.abs(np.asarray(handle.getcol("CPARAM")))
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
            antennas = np.asarray(handle.getcol("ANTENNA1"))
        finally:
            handle.close()
        values = np.where(flags, np.nan, gains)
        spreads = []
        for antenna in np.unique(antennas):
            per_antenna = values[..., antennas == antenna]
            finite = per_antenna[np.isfinite(per_antenna)]
            if finite.size > 1 and np.median(finite) > 0:
                spreads.append(float(np.std(finite) / np.median(finite)))
        return float(np.median(spreads)) if spreads else 0.0

    def _fringefit(self, project_code: str, params: dict, gaintable: Optional[list]) -> None:
        """Run casatasks.fringefit with prior tables attached, replacing any existing table."""
        table_path = Path(params["caltable"])
        if table_path.exists():
            shutil.rmtree(table_path)
        params = dict(params)
        params["vis"] = str(self.backend.ms_path(project_code))
        params.update(self._prior_callib(project_code, gaintable, table_path))
        try:
            self.backend.tasks.fringefit(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: fringefit failed "
                               f"(field={params.get('field')!r}, scan={params.get('scan', 'all')}, "
                               f"refant={params.get('refant')!r}): {exc}") from exc
        if not table_path.is_dir():
            raise BackendError(f"{project_code}: fringefit produced no table at {table_path}")

    def _median_table_snr(self, table_path: Path) -> float:
        """Return the median SNR of the unflagged solutions in a calibration table."""
        handle = self.backend.tools.table()
        if not handle.open(str(table_path)):
            return 0.0
        try:
            if "SNR" not in handle.colnames() or not handle.nrows():
                return 0.0
            snr = np.asarray(handle.getcol("SNR")).astype(float)
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
        finally:
            handle.close()
        usable = snr[~flags & (snr > 0) & (snr != _REFANT_SNR_SENTINEL)]
        return float(np.median(usable)) if usable.size else 0.0

    def measure_edge_channels(self, project_code: str, table: CalTable, *, threshold: float = 6.0,
                              max_edge_fraction: float = 0.25, **kwargs) -> dict:
        """Measure how many channels roll off at each subband edge, from the bandpass table.

        Builds three per-channel profiles across every antenna, subband and
        polarization — median amplitude, phase scatter, and flagged fraction —
        and finds the flat interior of each. Subband edges show up as amplitude
        roll-off, rising phase scatter, or solutions the solver had to flag; the
        widest trim the three agree on is what needs flagging.

        The same trim is applied to every subband: they share a signal path
        shape, and a per-subband trim would leave the band with ragged,
        non-uniform channel coverage.

        Parameters
        ----------
        table : CalTable
            The bandpass table to analyse.
        threshold : float
            Deviation in robust sigmas beyond which an edge channel is rejected.
        max_edge_fraction : float
            Never trim more than this fraction of a subband from either edge.

        Returns
        -------
        dict
            ``n_edge`` (channels to flag at each edge), ``first``/``last`` (the
            flat range), ``n_channels``, and the three profiles.
        """
        from ..statistics import find_flat_range, mad_sigma

        handle = self.backend.tools.table()
        if not handle.open(str(table.path)):
            raise BackendError(f"could not open bandpass table {table.path}")
        try:
            values = np.asarray(handle.getcol("CPARAM"))          # (npol, nchan, nrow)
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
        finally:
            handle.close()
        if values.ndim != 3 or values.shape[1] < 8:
            raise BackendError(f"{project_code}: bandpass table has an unexpected shape "
                               f"{values.shape}; cannot measure the band edges")

        n_channels = values.shape[1]
        amplitude = np.abs(values).astype(float)
        phase = np.angle(values)
        amplitude[flags] = np.nan
        phase[flags] = np.nan

        # Collapse polarization and row (= antenna x subband) onto the channel axis.
        per_channel = amplitude.transpose(1, 0, 2).reshape(n_channels, -1)
        phase_channel = phase.transpose(1, 0, 2).reshape(n_channels, -1)
        with np.errstate(invalid="ignore"):
            amp_profile = np.nanmedian(per_channel, axis=1)
            phase_profile = np.array([mad_sigma(row[np.isfinite(row)]) for row in phase_channel])
        flagged_fraction = flags.transpose(1, 0, 2).reshape(n_channels, -1).mean(axis=1)

        amp_range = find_flat_range(amp_profile, threshold=threshold,
                                    max_edge_fraction=max_edge_fraction)
        phase_range = find_flat_range(phase_profile, threshold=threshold,
                                      max_edge_fraction=max_edge_fraction)
        # Channels the solver itself could not solve are edges too.
        max_trim = int(n_channels * max_edge_fraction)
        solved = np.where(flagged_fraction < 0.5)[0]
        solved_range = ((int(solved[0]), int(solved[-1])) if solved.size
                        else (0, n_channels - 1))
        first = max(amp_range[0], phase_range[0], min(solved_range[0], max_trim))
        last = min(amp_range[1], phase_range[1], max(solved_range[1], n_channels - 1 - max_trim))
        # One trim for the whole band: use the wider of the two edges.
        n_edge = min(max_trim, max(first, n_channels - 1 - last))
        logger.info("edge channels: amplitude flat over {}, phase over {}, solved over {} "
                    "-> flag {} channel(s) at each edge of all {} subbands",
                    amp_range, phase_range, solved_range, n_edge, values.shape[0] and "the")
        return {"n_edge": int(n_edge), "first": int(n_edge), "last": int(n_channels - 1 - n_edge),
                "n_channels": int(n_channels), "amplitude_profile": amp_profile.tolist(),
                "phase_profile": phase_profile.tolist(),
                "flagged_fraction": flagged_fraction.tolist()}

    def solution_coverage(self, project_code: str, table: CalTable,
                          metadata: Optional[ObsMetadata] = None) -> dict[str, set[int]]:
        """Return ``{antenna: {subbands with an unflagged solution}}`` for a calibration table.

        Used to verify that every antenna expected to be detected actually got a
        solution in every subband it recorded — a silent gap here becomes flagged
        data at apply time.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        handle = self.backend.tools.table()
        if not handle.open(str(table.path)):
            raise BackendError(f"could not open calibration table {table.path}")
        try:
            antenna_ids = np.asarray(handle.getcol("ANTENNA1"))
            spw_ids = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
        finally:
            handle.close()
        handle.open(str(Path(table.path) / "ANTENNA"))
        try:
            names = [str(n) for n in handle.getcol("NAME")]
        finally:
            handle.close()
        # A row is usable when at least one parameter survived; FLAG is (npar, nchan, nrow).
        usable_row = ~np.all(flags, axis=tuple(range(flags.ndim - 1)))
        coverage: dict[str, set[int]] = {name: set() for name in names}
        for row in np.where(usable_row)[0]:
            index = int(antenna_ids[row])
            if index < len(names):
                coverage[names[index]].add(int(spw_ids[row]))
        return {name: spws for name, spws in coverage.items() if name in meta.antennas}

    def apply(self, project_code: str, field: str, tables: list[CalTable], *,
              gainfield: str = "", parang: bool = True, flagbackup: bool = False,
              **kwargs) -> None:
        """Apply the accumulated calibration tables to a field (applycal).

        Each table carries its own ``interp`` and ``spwmap``, so the caller never
        has to assemble the parallel lists applycal expects.

        Parameters
        ----------
        project_code : str
            Project code.
        field : str
            Field selection to correct (empty = all fields).
        tables : list of CalTable
            Tables to apply, in order.
        gainfield : str
            Field whose solutions to transfer (empty = matching field).
        parang : bool
            Apply the parallactic-angle correction (always on for VLBI).
        """
        ms = self.backend.ms_path(project_code)
        usable = [t for t in tables if t.path and Path(t.path).exists()]
        missing = [t.cal_type for t in tables if t not in usable]
        if missing:
            warnings.warn(f"{project_code}: skipping missing calibration table(s): "
                          f"{', '.join(missing)}")
        if not usable:
            raise BackendError(f"{project_code}: no calibration tables to apply")
        if not parang:
            warnings.warn(f"{project_code}: applycal without the parallactic-angle correction")
        callib = self.write_callib(project_code, usable, gainfield=gainfield)
        params = {"vis": str(ms), "docallib": True, "callib": str(callib),
                  "parang": parang, "flagbackup": flagbackup, "applymode": "calflagstrict"}
        if field:
            params["field"] = field
        logger.info("applycal: {} -> field={} ({} tables: {})", ms.name, field or "all",
                    len(usable), ", ".join(t.cal_type for t in usable))
        try:
            self.backend.tasks.applycal(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: applycal failed (field={field!r}, tables="
                               f"{[t.cal_type for t in usable]}): {exc}") from exc

    def _prior_callib(self, project_code: str, tables: Optional[list], table_path: Path) -> dict:
        """Return the applycal-on-the-fly parameters for a solve's prior tables.

        Solving on top of earlier calibration uses the same cal library applycal
        does, rather than parallel ``gaintable``/``interp``/``spwmap``/``gainfield``
        lists: one declarative file per solve, named after the table it produces,
        so what each step was solved on top of stays readable next to the data.
        Returns an empty dict when there are no priors.
        """
        if not tables:
            return {}
        # Interactive callers may pass bare paths; treat those as plain tables.
        tables = [t if isinstance(t, CalTable) else CalTable(Path(str(t)).suffix.lstrip(".") or "prior",
                                                             path=str(t))
                  for t in tables]
        callib = self.write_callib(project_code, tables,
                                   filename=f"callibs/{table_path.name}.txt")
        return {"docallib": True, "callib": str(callib)}

    def write_callib(self, project_code: str, tables: list, *, gainfield: str = "",
                     filename: str = "caltables.txt") -> Path:
        """Write a CASA cal-library file listing the tables to apply, and return its path.

        The cal library is applycal's declarative form: one line per table with
        its own interpolation, field mapping and spectral-window mapping. Using
        it instead of parallel ``gaintable``/``interp``/``spwmap``/``gainfield``
        lists means the exact calibration applied is a readable artefact next to
        the data — which is the record you need months later to know what was
        done, and what a reviewer would ask for.

        ``fldmap`` is where phase referencing lives: a table carrying a field
        mapping has those solutions transferred to every field, which is how the
        phase calibrator's fringe solutions reach the target.

        ``calwt`` comes from each table and defaults to True, matching CASA's own
        default. Writing False instead would silently stop the weights being
        calibrated — on a heterogeneous VLBI array that misweights every baseline
        to a sensitive antenna, and it changes the solutions themselves, because
        the solvers are weighted fits.

        Parameters
        ----------
        tables : list of CalTable
            Tables in application order.
        gainfield : str
            Overrides every table's own field mapping when given.
        filename : str
            Output name inside the working directory.

        Returns
        -------
        pathlib.Path
            The written cal-library file.
        """
        path = self.work_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# vlbipy cal library for {project_code}",
                 "# one line per calibration table, in application order",
                 "# tinterp/finterp = time/frequency interpolation; fldmap = solutions to"
                 " transfer from; spwmap = subband mapping"]
        for table in tables:
            interp = [part.strip() for part in str(table.interp or "linear").split(",")]
            entry = [f"caltable='{table.path}'",
                     f"calwt={bool(getattr(table, 'calwt', True))}",
                     f"tinterp='{interp[0] or 'linear'}'"]
            if len(interp) > 1 and interp[1]:
                entry.append(f"finterp='{interp[1]}'")
            mapping = gainfield or table.gainfield
            if mapping:
                entry.append(f"fldmap='{mapping}'")
            if table.spwmap:
                entry.append(f"spwmap={list(table.spwmap)}")
            lines.append(" ".join(entry))
        path.write_text("\n".join(lines) + "\n")
        logger.info("cal library written: {} ({} table(s))", path, len(tables))
        return path

    def _verify_apriori_inputs(self, project_code: str, antab: Optional[str] = None) -> dict:
        """Fail early unless the MS carries the Tsys / gain-curve data gencal needs.

        Amplitude calibration silently produces garbage when SYSCAL is empty, so
        this refuses to continue instead. The MS is already built by this point,
        so a missing Tsys cannot be patched in place: the ``.antab`` has to be
        appended to the FITS-IDI files and the data re-imported. This reports
        exactly that, naming the ``.antab`` when one is on disk.

        Parameters
        ----------
        project_code : str
            Project code.
        antab : str, optional
            Path to the ``.antab`` file, when the caller located one.
        """
        report = self.backend.data.check_apriori_data(project_code)
        if not report["has_tsys"]:
            hint = (f"append it with casavlbitools and re-import: the file {antab} is on disk"
                    if antab else
                    f"no {project_code}.antab found either — download it from the archive "
                    "(EVN: the .antab lives next to the FITS-IDI files)")
            raise BackendError(
                f"{project_code}: the measurement set has no valid system-temperature data "
                f"(SYSCAL is empty or all-negative), so amplitudes cannot be calibrated. "
                f"Tsys must be in the FITS-IDI before import — {hint}, then re-run "
                f"import_data(force=True).")
        if not report["has_gc"]:
            raise BackendError(
                f"{project_code}: the measurement set has no gain-curve data (GAIN_CURVE is "
                f"empty). Append the .antab gain curves to the FITS-IDI files and re-run "
                f"import_data(force=True).")
        if report["gc_is_trivial"]:
            warnings.warn(f"{project_code}: every gain-curve coefficient is 1.0 — the elevation "
                          "dependence of the antenna gain will not be corrected")
        missing = report["antennas_without_tsys"]
        if missing:
            warnings.warn(f"{project_code}: no valid Tsys for {', '.join(missing)}; their "
                          "amplitude scale stays uncalibrated")
        return report

    def scan_snr(self, project_code: str, field: str, *, refant: str = "",
                 channel_fraction: float = 0.8, scans: Optional[list] = None,
                 metadata: Optional[ObsMetadata] = None, gaintable: Optional[list] = None,
                 minsnr: float = 0.0, max_scans: int = 0, **kwargs) -> ScanSNRSurvey:
        """Measure fringe SNR per scan, antenna and polarization on the calibrators.

        Runs one ``fringefit`` with ``solint='inf', combine='spw'`` over the central
        ``channel_fraction`` of each subband — one solution per scan, antenna and
        polarization — and reads the SNR column back into a
        :class:`~vlbipy.models.ScanSNRSurvey`. The resulting table is a diagnostic
        product (``<code>.snr``) and is *not* added to the gain-table chain.

        Parameters
        ----------
        project_code : str
            Project code (the MS must exist).
        field : str
            Calibrator field selection (comma-separated source names).
        refant : str
            Reference antenna; its own solutions carry a sentinel SNR and are masked.
        channel_fraction : float
            Fraction of central channels per subband to use (default 0.8).
        scans : list, optional
            Restrict to these scan numbers (default: every scan on ``field``).
        max_scans : int
            Sample at most this many scans, spread evenly over the observation
            (0 = every scan). A fringe fit per scan is the most expensive part of
            inspection, and the survey only needs *relative* SNR to rank
            antennas and scans, so a representative sample is enough.
        metadata : ObsMetadata, optional
            Reuse already-loaded metadata instead of re-reading the MS.
        gaintable : list, optional
            Prior calibration tables to apply on the fly (e.g. Tsys/gain curve).
        minsnr : float
            Minimum SNR for fringefit to keep a solution (0 = keep everything, so
            weak antennas appear in the survey rather than silently vanishing).

        Returns
        -------
        ScanSNRSurvey
        """
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"{project_code}: measurement set {ms} not found; "
                               "run import_data() before the SNR survey")
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        spw = central_channel_selection(meta.freq_setup.n_channels, channel_fraction)
        refant = self.refant_chain(meta, refant)
        scans = self._survey_scans(meta, field, scans, max_scans)
        table_path = self.backend.caldir() / f"{project_code}.snr"
        if table_path.exists():
            shutil.rmtree(table_path)

        ff_kwargs = {"vis": str(ms), "caltable": str(table_path), "field": field, "spw": spw,
                     "solint": "inf", "combine": "spw", "refant": refant, "zerorates": False,
                     "corrdepflags": True, "minsnr": float(minsnr), "parang": True}
        # Same cal library as every other solve: the parallel-list form this used to
        # build dropped spwmap and gainfield, which silently misapplies any prior
        # solved with combine='spw'.
        ff_kwargs.update(self._prior_callib(project_code, gaintable, table_path))
        if scans:
            ff_kwargs["scan"] = ",".join(str(s) for s in scans)
        logger.info("scan_snr: fringefit over {} (spw={}, refant chain {}, central {:.0%} of channels)",
                    field or "all fields", spw, refant, channel_fraction)
        try:
            self.backend.tasks.fringefit(**ff_kwargs)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: scan_snr fringefit failed "
                               f"(field={field!r}, refant={refant!r}): {exc}") from exc

        survey = self._read_snr_table(table_path, meta, refant, channel_fraction, field)
        detected = len(survey.values_for())
        total = len(survey.scan_numbers) * len(survey.antennas) * max(1, len(survey.polarizations))
        logger.info("scan_snr: {} scans x {} antennas x {} pols; {}/{} solutions found",
                    len(survey.scan_numbers), len(survey.antennas), len(survey.polarizations),
                    detected, total)
        for antenna in survey.dead_antennas():
            logger.warning("scan_snr: antenna {} has no usable fringes (median SNR < 3)", antenna)
        return survey

    def _survey_scans(self, metadata: ObsMetadata, field: str, scans: Optional[list],
                      max_scans: int) -> Optional[list]:
        """Return the scan numbers to survey, sampling evenly when there are too many.

        A per-scan fringe fit is the dominant cost of inspection, and the survey
        only needs relative SNR, so beyond ``max_scans`` an evenly spread sample
        over the observation carries the same information for a fraction of the
        time. What was dropped is logged: a silent cap would read as full coverage.
        """
        if scans:
            candidates = list(scans)
        else:
            wanted = [f for f in field.split(",") if f]
            candidates = [s.scan_number for s in metadata.scans
                          if not wanted or s.source in wanted]
        if not candidates or max_scans <= 0 or len(candidates) <= max_scans:
            return scans if scans else None
        step = len(candidates) / float(max_scans)
        sampled = [candidates[min(len(candidates) - 1, int(i * step))] for i in range(max_scans)]
        sampled = sorted(dict.fromkeys(sampled))
        logger.info("scan_snr: sampling {} of {} scan(s) on {} (every ~{:.1f}th; raise "
                    "[calibration].snr_max_scans to survey all)", len(sampled), len(candidates),
                    field or "all fields", step)
        return sampled

    def refant_chain(self, metadata: ObsMetadata, requested: str = "") -> str:
        """Return a comma-separated reference-antenna fallback chain for CASA.

        A single reference antenna is not safe: it may be flagged in exactly the
        scans being solved, and CASA then fails the whole solve with "No valid
        reference antenna supplied". Passing the full ranked chain lets CASA fall
        back per solution interval instead.

        The order is: antennas the caller asked for, then the built-in
        :data:`REFANT_PRIORITY` list, then everything else by scan coverage.
        Dish size is deliberately not the primary key — DISH_DIAMETER is often 0
        in FITS-IDI-derived measurement sets.

        Parameters
        ----------
        metadata : ObsMetadata
            Observation metadata (supplies the observed antennas).
        requested : str
            Preferred antenna(s), comma-separated; unknown names are dropped.
        """
        from ..selection import rank_reference_antennas

        chain = rank_reference_antennas(metadata, requested)
        if not chain:
            raise BackendError("cannot pick a reference antenna: no antenna has data")
        return ",".join(chain)

    def _read_snr_table(self, table_path: Path, metadata: ObsMetadata, refant: str,
                        channel_fraction: float, field: str) -> ScanSNRSurvey:
        """Read a fringe caltable's SNR column into a ScanSNRSurvey keyed by scan/antenna/pol."""
        table = self.backend.tools.table()
        if not table.open(str(table_path)):
            raise BackendError(f"could not open SNR caltable {table_path}")
        try:
            times = np.asarray(table.getcol("TIME"))
            antenna_ids = np.asarray(table.getcol("ANTENNA1"))
            # ANTENNA2 holds the reference antenna *actually used* for each solution, which
            # varies across scans whenever CASA falls back down the refant chain.
            refant_ids = np.asarray(table.getcol("ANTENNA2"))
            snr = np.asarray(table.getcol("SNR")).T     # -> (nrows, nchan, nparam)
            flags = np.asarray(table.getcol("FLAG")).T  # -> (nrows, nchan, nparam)
        finally:
            table.close()
        table.open(str(table_path / "ANTENNA"))
        try:
            table_antennas = [str(n) for n in table.getcol("NAME")]
        finally:
            table.close()

        n_param = snr.shape[2] if snr.ndim == 3 else 1
        # A Fringe Jones table holds (phase, delay, rate, disp) per polarization; the SNR is
        # replicated across the four, so stride to the first parameter of each polarization.
        stride = 4 if n_param % 4 == 0 else 1
        n_pol = max(1, n_param // stride)
        pol_labels = polarization_labels(metadata, n_pol)

        wanted_sources = [f for f in field.split(",") if f] if field else []
        scans = [s for s in metadata.scans if not wanted_sources or s.source in wanted_sources]
        scan_index = {s.scan_number: i for i, s in enumerate(scans)}
        antennas = [a.name for a in metadata.observed_antennas] or list(metadata.antennas)
        antenna_index = {name: i for i, name in enumerate(antennas)}

        nan = float("nan")
        matrices = {label: [[nan] * len(antennas) for _ in scans] for label in pol_labels}
        refants_used: dict[str, int] = {}
        for row, time in enumerate(times):
            scan = self._scan_for_time(scans, float(time))
            if scan is None:
                continue
            name = (table_antennas[int(antenna_ids[row])]
                    if int(antenna_ids[row]) < len(table_antennas) else "")
            if name not in antenna_index:
                continue
            reference = int(refant_ids[row]) if row < len(refant_ids) else -1
            if 0 <= reference < len(table_antennas):
                refants_used[table_antennas[reference]] = refants_used.get(
                    table_antennas[reference], 0) + 1
            # A solution against itself is the reference antenna's sentinel, not a measurement.
            if reference == int(antenna_ids[row]):
                continue
            for pol in range(n_pol):
                column = pol * stride
                value = float(snr[row, 0, column]) if snr.ndim == 3 else float(snr[row])
                flagged = bool(flags[row, 0, column]) if flags.ndim == 3 else bool(flags[row])
                if flagged or value <= 0.0 or value == _REFANT_SNR_SENTINEL:
                    continue
                matrices[pol_labels[pol]][scan_index[scan.scan_number]][antenna_index[name]] = value

        used = sorted(refants_used, key=lambda n: -refants_used[n])
        if len(used) > 1:
            logger.info("scan_snr: reference antenna varied across scans: {}",
                        ", ".join(f"{n} ({refants_used[n]})" for n in used))
        return ScanSNRSurvey(project_code=metadata.project_code,
                             scan_numbers=[s.scan_number for s in scans],
                             scan_sources=[s.source for s in scans], antennas=antennas,
                             snr=matrices, channel_fraction=channel_fraction,
                             refant=",".join(used) if used else refant.split(",")[0])

    def _scan_for_time(self, scans: list[Scan], time: float) -> Optional[Scan]:
        """Return the scan whose time range contains ``time`` (nearest within a scan length)."""
        for scan in scans:
            if scan.time_start <= time <= scan.time_end:
                return scan
        # Solution timestamps sit at interval centres; fall back to the nearest scan.
        nearest = min(scans, key=lambda s: abs((s.time_start + s.time_end) / 2.0 - time),
                      default=None)
        if nearest is None:
            return None
        midpoint = (nearest.time_start + nearest.time_end) / 2.0
        return nearest if abs(midpoint - time) <= max(nearest.duration_sec, 60.0) else None


class CasaFlagOps(FlagOps):
    """Flagging via casatasks.flagdata (plus AOFlagger when it is installed)."""

    #: flagdata parameters per mode; ``kind`` -> (mode, extra kwargs).
    MODES = {"autocorr": ("manual", {"autocorr": True}),
             "quack": ("quack", {"quackmode": "beg"}),
             "tfcrop": ("tfcrop", {"datacolumn": "corrected", "timecutoff": 4.0, "freqcutoff": 3.0}),
             "rflag": ("rflag", {"datacolumn": "corrected"}),
             "manual": ("manual", {}),
             "from_file": ("list", {})}

    def run(self, project_code: str, kind: str, *, field: str = "", **kwargs) -> float:
        """Run one flagging mode and return the fraction of observable data it newly flagged.

        The fraction is measured as the change in the total flagged fraction, so
        it reports the effect of *this* operation rather than the cumulative
        state. See :meth:`flagged_fraction` for what "observable" excludes.
        """
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"{project_code}: measurement set {ms} not found")
        before = self.flagged_fraction(project_code)
        if kind == "aoflagger":
            self._run_aoflagger(project_code, field=field, **kwargs)
        else:
            self._run_flagdata(project_code, kind, field=field, **kwargs)
        after = self.flagged_fraction(project_code)
        delta = max(0.0, after - before)
        if kind == "autocorr":
            # Autocorrelations are outside the statistic by definition, so this step can
            # only ever report 0%; say so rather than let it read as "did nothing".
            logger.info("flag[autocorr]: autocorrelations flagged (they are excluded from "
                        "the flagging statistics, so this reports 0% by construction)")
            return delta
        logger.info("flag[{}]: {:.2%} newly flagged ({:.1%} -> {:.1%} of observable data)",
                    kind, delta, before, after)
        return delta

    def _run_flagdata(self, project_code: str, kind: str, *, field: str = "", **kwargs) -> None:
        """Translate a vlbipy flagging mode into a casatasks.flagdata call."""
        ms = self.backend.ms_path(project_code)
        if kind not in self.MODES and kind != "edges":
            raise self._unsupported(f"run[{kind}]", f"known modes: {', '.join(self.MODES)}, edges")
        params: dict = {"vis": str(ms), "flagbackup": False}
        if field:
            params["field"] = field
        if kind == "edges":
            params.update(mode="manual", spw=self._edge_spw_selection(project_code, **kwargs))
        else:
            mode, extra = self.MODES[kind]
            params["mode"] = mode
            params.update(extra)
            if kind == "quack":
                params["quackinterval"] = float(kwargs.pop("interval", 0.0) or 0.0)
                if params["quackinterval"] <= 0.0:
                    logger.info("flag[quack]: interval is 0 s; nothing to do")
                    return
            if kind == "from_file":
                params["inpfile"] = str(kwargs.pop("flagfile"))
            if kind == "manual":
                params.update({k: v for k, v in kwargs.items() if v not in (None, "")})
        logger.info("flagdata {}", " ".join(f"{k}={v!r}" for k, v in params.items() if k != "vis"))
        try:
            self.backend.tasks.flagdata(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: flagdata[{kind}] failed: {exc}") from exc

    def outliers(self, project_code: str, *, field: str = "", threshold: float = 5.0,
                 column: str = "corrected", max_time_bins: int = 400, dry_run: bool = False,
                 metadata: Optional[ObsMetadata] = None, **kwargs) -> dict:
        """Flag visibilities that break the smoothness of their own baseline.

        Calibrated data varies slowly along both time and frequency, so a point
        that departs from its own baseline's median by more than ``threshold``
        robust sigmas is a defect rather than signal. Judging each baseline
        against *itself* is what makes this safe on real arrays: a short
        eMERLIN spacing legitimately shows amplitudes many times higher than an
        intercontinental one, and a global cut would flag the whole station.

        Amplitude drives the decision — it is where interference and correlator
        problems show up most cleanly — with the phase scatter reported for
        information.

        Parameters
        ----------
        field : str
            Restrict to one field (empty = all).
        threshold : float
            Robust-sigma cut on amplitude.
        dry_run : bool
            Measure and report without applying any flags.

        Returns
        -------
        dict
            ``n_outliers``, ``fraction``, ``per_baseline`` (worst first) and
            ``flagged_fraction_of_data`` when the flags were applied.
        """
        from ..statistics import mad_sigma

        spectra = self.backend.data.read_dynamic_spectra(
            project_code, field=field, column=column, max_time_bins=max_time_bins,
            stokes_i=True, metadata=metadata)
        baselines = spectra["baselines"]
        commands: list[str] = []
        per_baseline: list[tuple[str, float]] = []
        times = np.asarray(spectra["times"], dtype=float)
        total_points = 0
        total_outliers = 0

        for (first, second), values in baselines.items():
            amplitude = np.abs(values)
            finite = np.isfinite(amplitude)
            if finite.sum() < 20:
                continue
            centre = float(np.nanmedian(amplitude[finite]))
            sigma = mad_sigma(amplitude[finite])
            total_points += int(finite.sum())
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            bad = finite & (np.abs(amplitude - centre) > threshold * sigma)
            if not bad.any():
                continue
            total_outliers += int(bad.sum())
            per_baseline.append((f"{first}&{second}", float(bad.sum()) / float(finite.sum())))
            # One flag command per affected time bin, over the whole bin's width.
            for bin_index in np.unique(np.where(bad)[0]):
                if bin_index >= times.size:
                    continue
                half = (times[1] - times[0]) / 2.0 if times.size > 1 else 1.0
                start = mjdsec2datetime(float(times[bin_index]) - half)
                end = mjdsec2datetime(float(times[bin_index]) + half)
                stamp = "%Y/%m/%d/%H:%M:%S.%f"
                commands.append(f"antenna='{first}&{second}' "
                                f"timerange='{start.strftime(stamp)[:-3]}~{end.strftime(stamp)[:-3]}'"
                                + (f" field='{field}'" if field else ""))

        fraction = (total_outliers / total_points) if total_points else 0.0
        per_baseline.sort(key=lambda item: item[1], reverse=True)
        report = {"n_outliers": total_outliers, "n_points": total_points, "fraction": fraction,
                  "per_baseline": per_baseline, "threshold": threshold, "commands": len(commands)}
        logger.info("outliers: {} of {} points ({:.2%}) beyond {:.0f} sigma on {} baseline(s)",
                    total_outliers, total_points, fraction, threshold, len(per_baseline))
        for name, share in per_baseline[:5]:
            logger.info("  worst: {} {:.1%} of its points", name, share)
        if dry_run or not commands:
            return report
        before = self.backend.flag.flagged_fraction(project_code)
        try:
            self.backend.tasks.flagdata(vis=str(self.backend.ms_path(project_code)), mode="list",
                                        inpfile=commands, flagbackup=False, action="apply")
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: flagging outliers failed: {exc}") from exc
        after = self.backend.flag.flagged_fraction(project_code)
        report["flagged_fraction_of_data"] = max(0.0, after - before)
        logger.info("outliers: flagged {:.2%} of the data ({:.1%} -> {:.1%})",
                    report["flagged_fraction_of_data"], before, after)
        return report

    def measure_quack(self, project_code: str, *, field: str = "", column: str = "corrected",
                      sigma: float = 2.0, max_seconds: float = 120.0,
                      metadata: Optional[ObsMetadata] = None, **kwargs) -> dict:
        """Measure the per-antenna slew time from calibrated data, one antenna at a time.

        For each scan and each baseline, the level is the baseline's own median
        over that scan and the noise its MAD; a sample counts as off-source when
        it lies more than ``sigma`` MADs *below* that median. An antenna's slew
        in a scan is the leading stretch that is low on **all** of its baselines
        — the minimum over them. That minimum is what separates "this antenna is
        off source" from "the far end is": a genuinely off-source antenna drags
        down every baseline it is part of, while the far end only affects one.

        Antennas are then resolved one at a time, largest first: once an
        antenna's slew is known its samples are masked out, so the antennas
        still to be measured are no longer judged through baselines that were
        contaminated by it. Without that, a slow antenna makes every other
        antenna look slow too.

        The per-antenna result is averaged over its scans and applied uniformly
        to every scan of every source.

        Parameters
        ----------
        field : str
            Source to measure on — normally the phase calibrator, which is
            observed often enough to average over.
        sigma : float
            MADs below the scan median that count as off-source.
        max_seconds : float
            Refuse to report more than this (a longer ramp is a different fault).

        Returns
        -------
        dict
            ``per_antenna`` (seconds), ``per_scan`` (raw per-antenna lists),
            ``sigma``, ``integration_time``.
        """
        from ..statistics import mad_sigma

        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        wanted = [f for f in field.split(",") if f]
        scans = [s for s in meta.scans if not wanted or s.source in wanted]
        if not scans:
            raise BackendError(f"{project_code}: no scans on {field!r}")
        integration = max(0.5, min((s.integration_time for s in scans if s.integration_time),
                                   default=2.0))
        starts = np.array([s.time_start for s in scans], dtype=float)
        ends = np.array([s.time_end for s in scans], dtype=float)
        antenna_names = list(meta.antennas)
        pol_indices, _ = parallel_hand_indices(meta)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]

        # amplitude[(scan, a, b)] -> (offsets, amplitudes) for one baseline in one scan
        series: dict[tuple, list] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(meta.freq_setup.n_subbands):
                ms_tool.selectinit(datadescid=spw)
                if wanted:
                    ms_tool.select({"field": wanted})
                try:
                    record = ms_tool.getdata([column_name, "flag", "time", "antenna1", "antenna2"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record["flag"], dtype=bool)
                values = np.where(flags, np.nan, values)[pol_indices, :, :]
                with np.errstate(invalid="ignore"):
                    amplitude = np.nanmean(np.abs(values), axis=(0, 1))
                times = np.asarray(record["time"], dtype=float)
                ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])
                scan_index = np.searchsorted(starts, times, side="right") - 1
                inside = (scan_index >= 0) & (scan_index < len(scans))
                inside[inside] &= times[inside] <= ends[scan_index[inside]]
                usable = inside & np.isfinite(amplitude) & (ant1 != ant2)
                for row in np.flatnonzero(usable):
                    key = (int(scan_index[row]), int(ant1[row]), int(ant2[row]))
                    series.setdefault(key, []).append(
                        (times[row] - starts[scan_index[row]], float(amplitude[row])))
        finally:
            ms_tool.close()
        if not series:
            raise BackendError(f"{project_code}: no usable data on {field!r} to measure the slew")

        # Per baseline and scan: how long the leading low stretch lasts.
        def leading_low(entries: list, masked: dict, scan: int) -> float:
            offsets = np.array([e[0] for e in entries], dtype=float)
            amps = np.array([e[1] for e in entries], dtype=float)
            order = np.argsort(offsets)
            offsets, amps = offsets[order], amps[order]
            keep = np.ones(offsets.shape, dtype=bool)
            for antenna_id, seconds in masked.items():   # already-resolved antennas
                if antenna_id in scan_antennas.get(scan, ()) and seconds > 0:
                    keep &= ~(offsets < seconds)
            offsets, amps = offsets[keep], amps[keep]
            if amps.size < 5:
                return 0.0
            level, noise = float(np.median(amps)), mad_sigma(amps)
            if not np.isfinite(noise) or noise <= 0:
                return 0.0
            low = amps < level - sigma * noise
            if not low.size or not low[0]:
                return 0.0
            end = 0
            while end < low.size and low[end]:
                end += 1
            return float(offsets[end - 1] + integration)

        scan_antennas = {i: {int(a) for key in series if key[0] == i for a in key[1:]}
                         for i in range(len(scans))}
        per_scan: dict[str, list] = {}
        per_antenna: dict[str, float] = {}
        masked: dict[int, float] = {}
        candidates = {a for key in series for a in key[1:]}

        while candidates:
            estimates: dict[int, list] = {}
            for antenna_id in candidates:
                for scan in range(len(scans)):
                    baselines = [entries for key, entries in series.items()
                                 if key[0] == scan and antenna_id in key[1:]]
                    if len(baselines) < 2:
                        continue     # one baseline cannot separate the two ends
                    # Low on *all* baselines: the minimum is what this antenna owns.
                    lows = [leading_low(entries, masked, scan) for entries in baselines]
                    estimates.setdefault(antenna_id, []).append(min(lows))
            averaged = {a: float(np.mean(v)) for a, v in estimates.items() if v}
            if not averaged:
                break
            worst = max(averaged, key=lambda a: averaged[a])
            seconds = min(averaged[worst], max_seconds)
            name = antenna_names[worst] if worst < len(antenna_names) else str(worst)
            per_scan[name] = estimates.get(worst, [])
            candidates.discard(worst)
            if seconds <= 0:
                continue     # nothing to flag for this one; the rest will be smaller still
            masked[worst] = seconds
            per_antenna[name] = seconds
            logger.info("quack: {} slews for {:.0f}s on average over {} scan(s)",
                        name, seconds, len(estimates.get(worst, [])))

        logger.info("quack: measured on {} over {} scan(s); {} antenna(s) need trimming ({})",
                    field or "all fields", len(scans), len(per_antenna),
                    ", ".join(f"{k} {v:.0f}s" for k, v in sorted(per_antenna.items())) or "none")
        return {"per_antenna": per_antenna, "per_scan": per_scan, "sigma": sigma,
                "integration_time": integration}

    def quack(self, project_code: str, *, per_antenna: Optional[dict] = None,
              field: str = "", **kwargs) -> float:
        """Apply per-antenna quack flagging, measuring the intervals when not given."""
        measurement = None
        if per_antenna is None:
            measurement = self.measure_quack(project_code, field=field, **kwargs)
            per_antenna = measurement["per_antenna"]
        if not per_antenna:
            logger.info("quack: no antenna shows a settling ramp; nothing to flag")
            return 0.0
        before = self.flagged_fraction(project_code)
        ms = str(self.backend.ms_path(project_code))
        for antenna, seconds in sorted(per_antenna.items()):
            logger.info("quack: flagging the first {:.0f} s of each scan on {}", seconds, antenna)
            try:
                self.backend.tasks.flagdata(vis=ms, mode="quack", quackmode="beg",
                                            quackinterval=float(seconds), antenna=str(antenna),
                                            flagbackup=False)
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: quack flagging of {antenna} failed: "
                                   f"{exc}") from exc
        after = self.flagged_fraction(project_code)
        logger.info("quack: {:.2%} newly flagged ({:.1%} -> {:.1%})", after - before, before, after)
        return max(0.0, after - before)

    def _edge_spw_selection(self, project_code: str, *, edge_fraction: float = 0.1,
                            n_channels: int = 0, edge_channels: int = 0, **kwargs) -> str:
        """Build the spw selection flagging the outer channels of every subband.

        ``edge_channels`` (an explicit channel count, e.g. from the measured
        bandpass roll-off) wins over the ``edge_fraction`` default.
        """
        if not n_channels:
            n_channels = int(self.backend.data.get_metadata(
                project_code, [], "").freq_setup.n_channels)
        n_edge = int(edge_channels) or int(round(n_channels * float(edge_fraction)))
        n_edge = max(0, min(n_edge, n_channels // 2 - 1))
        if n_edge <= 0:
            return ""
        return f"*:0~{n_edge - 1};{n_channels - n_edge}~{n_channels - 1}"

    def _run_aoflagger(self, project_code: str, *, field: str = "", strategy: str = "default",
                       **kwargs) -> None:
        """Run the external AOFlagger binary, falling back to tfcrop when it is absent."""
        import shutil as sh
        import subprocess
        binary = sh.which("aoflagger")
        if not binary:
            logger.info("aoflagger not installed; falling back to flagdata(mode='tfcrop')")
            self._run_flagdata(project_code, "tfcrop", field=field)
            return
        command = [binary, "-v"]
        if strategy and strategy != "default":
            command += ["-strategy", str(strategy)]
        command.append(str(self.backend.ms_path(project_code)))
        logger.info("running {}", " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise BackendError(f"{project_code}: aoflagger failed ({result.returncode}): "
                               f"{result.stderr.strip()[:300]}")

    def _data_column(self, project_code: str) -> str:
        """Return the column that says whether a visibility was ever recorded.

        DATA holds the correlator output, so a zero there means nothing was
        recorded. CORRECTED_DATA is only a fallback for datasets split without
        the raw column.
        """
        table = self.backend.tools.table()
        if not table.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS {self.backend.ms_path(project_code)}")
        try:
            columns = set(table.colnames())
        finally:
            table.close()
        for name in ("DATA", "CORRECTED_DATA"):
            if name in columns:
                return name
        raise BackendError(f"{project_code}: no DATA or CORRECTED_DATA column to inspect")

    def _count_query(self, project_code: str, *, where: str = "", groupby: str = "") -> list[dict]:
        """Run one TaQL pass counting flagged and observable visibilities.

        Returns one dict per group with ``flagged``, ``observable`` and (when
        grouped) the grouping column. Reads FLAG and DATA, so it costs a pass
        over the visibilities — a few seconds even on a multi-GB set, which is
        negligible next to the flagdata runs it brackets.
        """
        ms = self.backend.ms_path(project_code)
        query = count_query_text(str(ms), self._data_column(project_code),
                                 where=where, groupby=groupby)
        table = self.backend.tools.table()
        if not table.open(str(ms)):
            raise BackendError(f"could not open MS {ms}")
        try:
            try:
                result = table.taql(query)
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: flag statistics query failed: {exc}") from exc
            # A GROUPBY result is a calculated table, so read it cell by cell (getcol raises).
            try:
                rows = []
                for i in range(result.nrows()):
                    row = {"flagged": int(result.getcell("NFLAGGED", i)),
                           "observable": int(result.getcell("NOBSERVABLE", i))}
                    if groupby:
                        row[groupby] = int(result.getcell(groupby, i))
                    rows.append(row)
                return rows
            finally:
                result.close()
        finally:
            table.close()

    def flagged_fraction(self, project_code: str, *, where: str = "") -> float:
        """Return the flagged fraction of the *observable* data.

        Autocorrelations and visibilities that were never recorded (exactly zero
        — a station absent from a scan, or a subband it did not observe) are
        excluded from both the numerator and the denominator. Counting them
        would make every fraction a statement about the array's schedule rather
        than about the data: on a typical VLBI set more than half the
        visibilities are zero-filled and already flagged, which pins the
        reported fraction near that floor no matter what the flagging did.

        Parameters
        ----------
        project_code : str
            Project code.
        where : str, optional
            Extra TaQL condition ANDed onto the selection, e.g.
            ``"ANTENNA1 == 3 OR ANTENNA2 == 3"``.

        Returns
        -------
        float
            Flagged fraction in [0, 1]; 0.0 when there is no observable data.
        """
        rows = self._count_query(project_code, where=where)
        if not rows or not rows[0]["observable"]:
            return 0.0
        return rows[0]["flagged"] / rows[0]["observable"]

    def summary(self, project_code: str, *, where: str = "") -> dict:
        """Return flagging statistics overall and per antenna and subband.

        Every count is over observable data only, on the same basis as
        :meth:`flagged_fraction` (no autocorrelations, no never-recorded
        visibilities), so an antenna reading 100% really has lost all the data
        it recorded rather than merely not having observed.

        Parameters
        ----------
        project_code : str
            Project code.
        where : str, optional
            Extra TaQL condition ANDed onto the selection.

        Returns
        -------
        dict
            ``{"flagged", "observable", "fraction",
            "antenna": {name: {...}}, "spw": {id: {...}}}``, where each nested
            entry carries the same three keys.
        """
        def entry(flagged: int, observable: int) -> dict:
            return {"flagged": flagged, "observable": observable,
                    "fraction": flagged / observable if observable else 0.0}

        totals = self._count_query(project_code, where=where)
        report = entry(totals[0]["flagged"] if totals else 0,
                       totals[0]["observable"] if totals else 0)
        report["spw"] = {row["DATA_DESC_ID"]: entry(row["flagged"], row["observable"])
                         for row in self._count_query(project_code, where=where,
                                                      groupby="DATA_DESC_ID")}
        # An MS stores each baseline once with ANTENNA1 < ANTENNA2, so an antenna's data is
        # split across both columns: group by each in turn and add, or the highest-numbered
        # antennas look empty.
        names = self._antenna_names(project_code)
        per_antenna: dict[str, list[int]] = {name: [0, 0] for name in names}
        for column in ("ANTENNA1", "ANTENNA2"):
            for row in self._count_query(project_code, where=where, groupby=column):
                if row[column] < len(names):
                    counts = per_antenna[names[row[column]]]
                    counts[0] += row["flagged"]
                    counts[1] += row["observable"]
        report["antenna"] = {name: entry(*counts) for name, counts in per_antenna.items()}
        return report

    def _antenna_names(self, project_code: str) -> list[str]:
        """Return antenna names in ANTENNA-table order (index = antenna ID)."""
        table = self.backend.tools.table()
        if not table.open(f"{self.backend.ms_path(project_code)}/ANTENNA"):
            raise BackendError(f"{project_code}: could not open the ANTENNA subtable")
        try:
            return [str(name) for name in table.getcol("NAME")]
        finally:
            table.close()


class CasaExportOps(ExportOps):
    """Split calibrated data per source and export it."""

    def ms(self, project_code: str, source: str, *, datacolumn: str = "corrected",
           chanbin: int = -1, timebin: str = "", keepflags: bool = True,
           outputvis: str = "", metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Split one source into its own measurement set under ``<work_dir>/calibrated_data``.

        Parameters
        ----------
        project_code : str
            Project code.
        source : str
            Field to split out.
        datacolumn : str
            Column to write into the output DATA column (``corrected`` after applycal).
        chanbin : int
            Channels to average together. ``-1`` (the default) collapses each
            subband to a single channel, which is what continuum work wants and
            what keeps the exported UVFITS small; ``0`` or ``1`` keeps every
            channel. Matches the ``casa_pipeline`` convention.
        timebin : str
            Time averaging interval (e.g. ``"4s"``; empty = none).
        keepflags : bool
            Keep flagged rows in the output (False drops them).
        outputvis : str
            Explicit output path; defaults to ``<code>_<source>.ms``.

        Returns
        -------
        str
            Path to the split measurement set.
        """
        ms = self.backend.ms_path(project_code)
        outdir = self.work_dir / "calibrated_data"
        outdir.mkdir(parents=True, exist_ok=True)
        target = Path(outputvis) if outputvis else outdir / f"{project_code}_{source}.ms"
        if target.exists():
            shutil.rmtree(target)
        params = {"vis": str(ms), "outputvis": str(target), "field": source,
                  "datacolumn": datacolumn, "keepflags": keepflags}
        n_channels = (metadata or self.backend.data.get_metadata(
            project_code, [], "")).freq_setup.n_channels
        chanbin = int(chanbin)
        if (chanbin == -1 or chanbin > 1) and n_channels > 1:
            params.update(chanaverage=True, chanbin=n_channels if chanbin == -1 else chanbin)
        else:
            params.update(chanaverage=False, chanbin=1)
        if timebin:
            params.update(timeaverage=True, timebin=str(timebin))
        logger.info("mstransform: {} field={} -> {}", ms.name, source, target.name)
        try:
            self.backend.tasks.mstransform(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: split of {source!r} failed: {exc}") from exc
        return str(target)

    def uvfits(self, project_code: str, source: str, *, multisource: bool = False,
               combinespw: bool = True, padwithflags: bool = True, overwrite: bool = True,
               split_ms: str = "", **kwargs) -> str:
        """Export one source to UVFITS, ready to load in Difmap.

        Uses the parameter combination ``casa_pipeline`` settled on, because it
        is the one that carries the flags across correctly:

        ``combinespw=True``
            merges the subbands into one spectral window, which Difmap expects.
        ``padwithflags=True``
            fills the gaps that leaves with flagged rows instead of dropping
            them — without it a partially-flagged dataset exports a ragged,
            unreadable UVFITS.
        ``multisource=False``
            single-source files, so Difmap needs no source selection.

        The data column is ``data``: the split already wrote the corrected
        visibilities there, and the split product has no CORRECTED column.

        Parameters
        ----------
        split_ms : str
            Use this already-split measurement set instead of splitting again.
        """
        source_ms = Path(split_ms) if split_ms else Path(self.ms(project_code, source, **kwargs))
        target = source_ms.with_suffix(".uvfits")
        if target.exists() and overwrite:
            target.unlink()
        logger.info("exportuvfits: {} -> {} (combinespw={}, padwithflags={})",
                    source_ms.name, target.name, combinespw, padwithflags)
        try:
            self.backend.tasks.exportuvfits(vis=str(source_ms), fitsfile=str(target),
                                            datacolumn="data", multisource=multisource,
                                            combinespw=combinespw, padwithflags=padwithflags,
                                            overwrite=overwrite)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: exportuvfits of {source!r} failed: {exc}") from exc
        return str(target)


class CasaPlotOps(PlotOps):
    """Diagnostic plots rendered from CASA tables with matplotlib."""

    def caltable(self, project_code: str, caltable: str, cal_type: str = "",
                 metadata: Optional[ObsMetadata] = None) -> list[str]:
        """Plot a calibration table to PNG(s) under <work_dir>/plots."""
        from ..plotting import CalTablePlotter
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        frequencies = {s: meta.freq_setup.frequencies_ghz(s)
                       for s in range(meta.freq_setup.n_subbands)}
        plotter = CalTablePlotter(self.plot_dir("caltables"), frequencies=frequencies)
        return [str(p) for p in plotter.plot(caltable, cal_type=cal_type)]

    def spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                 refant: str = "", column: str = "corrected", label: str = "",
                 all_pols: bool = False, metadata: Optional[ObsMetadata] = None,
                 **kwargs) -> list[str]:
        """Plot amplitude and phase vs frequency on baselines to the reference antenna."""
        from ..plotting import plot_corrected_spectrum
        spectrum = self.backend.data.read_spectrum(project_code, field=field, scans=scans,
                                                   refant=refant, column=column,
                                                   all_pols=all_pols, metadata=metadata)
        return [str(p) for p in plot_corrected_spectrum(
            spectrum, self.plot_dir(self.category_for_column(column)), project_code, label=label)]

    def radplot(self, project_code: str, *, field: str = "", column: str = "corrected",
                time_bin: float = 10.0, label: str = "",
                metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Plot amplitude and phase vs uv distance for one source."""
        from ..plotting import plot_radplot
        uvdata = self.backend.data.read_uvdistance(project_code, field=field, column=column,
                                                   time_bin=time_bin, metadata=metadata)
        return str(plot_radplot(uvdata, self.plot_dir(self.category_for_column(column)),
                                project_code, label=label))

    def timeseries(self, project_code: str, *, field: str = "", refant: str = "",
                   column: str = "corrected", label: str = "",
                   metadata: Optional[ObsMetadata] = None, **kwargs) -> list[str]:
        """Plot amplitude and phase vs time on baselines to the reference antenna."""
        from ..plotting import plot_baseline_timeseries
        series = self.backend.data.read_timeseries(project_code, field=field, refant=refant,
                                                   column=column, metadata=metadata, **kwargs)
        return [str(p) for p in plot_baseline_timeseries(
            series, self.plot_dir(self.category_for_column(column)), project_code, label=label)]

    def baseline_corner(self, project_code: str, *, field: str = "", column: str = "corrected",
                        quantity: str = "phase", max_time_bins: int = 200, label: str = "",
                        metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Plot the per-baseline time x frequency corner grid for one field."""
        from ..plotting import plot_baseline_corner
        spectra = self.backend.data.read_dynamic_spectra(
            project_code, field=field, column=column, max_time_bins=max_time_bins,
            stokes_i=True, metadata=metadata)
        return str(plot_baseline_corner(spectra, self.plot_dir(self.category_for_column(column)),
                                        project_code, quantity=quantity, label=label))

    def bandpass_profile(self, project_code: str, measurement: dict, **kwargs) -> str:
        """Plot the per-channel band profile that drove the edge-channel decision."""
        from ..plotting import plot_bandpass_profile
        return str(plot_bandpass_profile(measurement, self.plot_dir("caltables"), project_code))

    def scan_snr(self, project_code: str, survey: Optional[ScanSNRSurvey] = None,
                 *, field: str = "", **kwargs) -> list[str]:
        """Plot the scan/antenna fringe-SNR matrix, one PNG per polarization.

        Parameters
        ----------
        survey : ScanSNRSurvey, optional
            An existing survey; computed via
            :meth:`CasaCalibrationOps.scan_snr` when omitted.
        field : str
            Calibrator selection used when the survey has to be computed.
        """
        from ..plotting import ScanSNRPlotter
        if survey is None:
            survey = self.backend.calibrate.scan_snr(project_code, field, **kwargs)
        # The SNR survey measures the data before the main calibration is applied.
        plotter = ScanSNRPlotter(self.plot_dir("raw"))
        return [str(p) for p in plotter.plot(survey)]


class CasaBackend(Backend):
    """Backend using casatools/casatasks (import, metadata, a-priori, diagnostics).

    Parameters
    ----------
    work_dir : str
        Directory holding the measurement set(s) and products.
    """

    kind = "casa"
    requires_data_files = True

    data_ops = CasaDataOps
    calibration_ops = CasaCalibrationOps
    flag_ops = CasaFlagOps
    plot_ops = CasaPlotOps
    export_ops = CasaExportOps

    def __init__(self, work_dir: str = ".") -> None:
        try:
            import casatasks
            import casatools
        except ImportError as exc:
            raise BackendError("the CASA backend requires casatools/casatasks: "
                               "pip install vlbipy[casa]") from exc
        self.tasks = casatasks
        self.tools = casatools
        self._ms_overrides: dict[str, Path] = {}
        super().__init__(work_dir)
        self.redirect_casa_log()

    def log_dir(self) -> Path:
        """Return (and create) the log directory (work_dir/logs)."""
        path = self.work_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def redirect_casa_log(self) -> Path:
        """Point CASA's log at ``<work_dir>/logs`` and sweep up any stray logs.

        CASA opens a log named for the moment it starts, in whatever directory
        the process was launched from — and it does so on ``import casatasks``,
        before any of our code runs. So the redirect has to be followed by a
        sweep: the redirect stops new files appearing in the shell's working
        directory, and the sweep moves the one already created (plus any left by
        earlier runs) under the project directory, where the rest of the outputs
        live.

        Returns
        -------
        pathlib.Path
            The log file CASA now writes to.
        """
        from ..tools import collect_casa_logs

        log_dir = self.log_dir()
        previous = ""
        try:
            previous = str(self.tasks.casalog.logfile())
        except Exception as exc:  # noqa: BLE001 - logging must never break the backend
            logger.debug("could not read the current CASA log path: {}", exc)
        target = log_dir / (Path(previous).name if previous else "casa.log")
        try:
            self.tasks.casalog.setlogfile(str(target))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not redirect the CASA log to {} ({}); it will keep writing "
                           "to the current directory", target, exc)
            return Path(previous or target)

        # The old file is closed now that the logger moved: fold it in, then sweep.
        # The project directory is swept too — a log loose beside the data is as
        # much clutter as one in the shell's cwd.
        search_dirs = {Path.cwd(), self.work_dir}
        if previous:
            search_dirs.add(Path(previous).parent)
        collect_casa_logs(log_dir, search_dirs=search_dirs)
        logger.info("CASA log -> {}", target)
        return target

    def ms_path(self, project_code: str) -> Path:
        """Return the measurement-set path; the Multi-MS (<code>.mms) wins when present."""
        override = self._ms_overrides.get(project_code)
        if override is not None:
            return override
        mms = self.work_dir / f"{project_code}.mms"
        return mms if mms.is_dir() else self.work_dir / f"{project_code}.ms"

    def register_ms(self, project_code: str, ms_path: Path) -> None:
        """Record an explicit measurement-set path for a project (used by ``data.adopt_ms``)."""
        self._ms_overrides[project_code] = ms_path

    def caldir(self) -> Path:
        """Return (and create) the calibration-table directory (work_dir/caltables)."""
        path = self.work_dir / "caltables"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def apriori_table_paths(self, project_code: str, needs_eop: bool = False) -> dict[str, Path]:
        """Return the expected a-priori caltable paths, keyed by cal type."""
        types = ["tsys", "gc"] + (["eop"] if needs_eop else [])
        return {t: self.caldir() / f"{project_code}{APRIORI_TABLE_SPECS[t][0]}" for t in types}
