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
from ..models import (Antenna, CalTable, FreqSetup, ObsMetadata, QualityMetrics, Scan, ScanSNRSurvey,
                      Stokes)
from ..results import Image
from ..tools import fetch_eop_file, mjdsec2datetime, space_available_gb
from .base import Backend, CalibrationOps, DataOps, ExportOps, FlagOps, ImagingOps, PlotOps

logger = get_logger()

#: A-priori caltable name suffix and apply-interpolation per calibration type.
APRIORI_TABLE_SPECS = {"accor": (".accor", "nearest"), "tsys": (".tsys", "nearest"),
                       "gc": (".gcal", "nearest"), "eop": (".eop", "nearest")}

#: SNR value CASA writes for the reference antenna's own (trivial) solution.
_REFANT_SNR_SENTINEL = 999.0

#: Parallel-hand polarization pairs, by correlation basis.
_PARALLEL_HANDS = {Stokes.RR: "RR", Stokes.LL: "LL", Stokes.XX: "XX", Stokes.YY: "YY"}

#: Re-exported for backwards compatibility; the ordering now lives in vlbipy.selection.
from ..selection import REFANT_PRIORITY  # noqa: E402,F401

#: TaQL condition selecting the rows any flagging statistic is measured over. Autocorrelations
#: are never imaged, so flagging them says nothing about data quality.
_OBSERVABLE_ROWS = "ANTENNA1 != ANTENNA2"
# Scans read per subband at once when measuring the slew ramp: bounds the memory of
# one read to a few hundred MB instead of a whole subband of the experiment.
_QUACK_SCANS_PER_READ = 20
# Scans read per subband at once by the whole-field readers (dynamic spectra, uv distance):
# a full subband of a long target track is tens of GB once unpacked (EM163: 1.2M rows).
_SCANS_PER_READ = 10


#: Name of the flag version saved right after the import (see ``CasaDataOps.reset_calibration``).
IMPORT_FLAG_VERSION = "as_imported"

#: Telescope names a correlator may write for a network other than the network's own name.
_TELESCOPE_ALIASES = {"LBA": {"VLBA", "ATLBA"}}


#: Polynomial coefficients per polarization that CASA's gain-curve table (``gencal caltype='gc'``,
#: an "EPowerCurve") can hold. A longer polynomial is cut off there without any message.
_CASA_GC_MAX_COEFFICIENTS = 8


def refit_polynomial(coefficients, lo: float, hi: float, *, max_coefficients: int = _CASA_GC_MAX_COEFFICIENTS,
                     tolerance: float = 1e-3) -> tuple[np.ndarray, float]:
    """Approximate a power-series polynomial on ``[lo, hi]`` with at most ``max_coefficients`` terms.

    Returns the lowest-degree least-squares fit whose largest relative deviation
    from the original on the interval is within ``tolerance`` (the highest
    allowed degree when none is), as power-series coefficients ``c_0 .. c_n``,
    and that largest relative deviation.
    """
    from numpy.polynomial import Polynomial
    original = Polynomial(np.asarray(coefficients, dtype=float))
    x = np.linspace(float(lo), float(hi), 400)
    y = original(x)
    scale = np.maximum(np.abs(y), 1e-12)
    best: tuple[np.ndarray, float] = (np.asarray(coefficients, dtype=float)[:max_coefficients], float("inf"))
    for degree in range(0, max_coefficients):
        fit = Polynomial.fit(x, y, degree).convert()
        deviation = float(np.max(np.abs(fit(x) - y) / scale))
        if deviation < best[1]:
            best = (np.asarray(fit.coef, dtype=float), deviation)
        if deviation <= tolerance:
            break
    return best


def antenna_elevation_range(metadata, antenna: str, *, pad: float = 2.0,
                            default: tuple[float, float] = (8.0, 88.0)) -> tuple[float, float]:
    """Lowest and highest elevation (deg) at which ``antenna`` observed, padded by ``pad`` and kept in 1-90.

    Evaluated at the start, middle and end of every scan the antenna took part
    in. Falls back to ``default`` when the antenna position or the source
    coordinates are not known.
    """
    try:
        from astropy import units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        entry = metadata.antennas.get(antenna)
        position = entry.position if entry is not None else None
        coords = dict(getattr(metadata, "source_coords", {}) or {})
        if position is None or not any(position):
            return default
        location = EarthLocation.from_geocentric(*position, unit=u.m)
        times, ra, dec = [], [], []
        for scan in metadata.scans:
            if antenna not in scan.antennas or scan.source not in coords:
                continue
            for moment in (scan.time_start, 0.5 * (scan.time_start + scan.time_end), scan.time_end):
                times.append(moment / 86400.0)
                ra.append(coords[scan.source][0])
                dec.append(coords[scan.source][1])
        if not times:
            return default
        frame = AltAz(obstime=Time(np.asarray(times), format="mjd"), location=location)
        elevation = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg).transform_to(frame).alt.deg
        return float(max(1.0, np.min(elevation) - pad)), float(min(90.0, np.max(elevation) + pad))
    except Exception as exc:  # noqa: BLE001 - a missing ephemeris must not stop the calibration
        logger.debug("elevation range of {} not computed ({}); using {}", antenna, exc, default)
        return default


def _scan_chunks(scans, size: int = _SCANS_PER_READ) -> list[list[int]]:
    """Split the scan numbers of ``scans`` into consecutive groups of at most ``size``."""
    numbers = sorted({int(s.scan_number) for s in scans})
    return [numbers[i:i + size] for i in range(0, len(numbers), size)]


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


def select_ms_rows(ms_tool, *, field: str = "", baseline: str = "", scans: Optional[list] = None) -> None:
    """Restrict an open ms tool (after ``selectinit``) to a field / baseline / scan selection.

    Uses ``ms.msselect`` with CASA selection syntax: ``ms.select`` only knows
    column-style keys (``antenna1``, ``scan_number``) and silently *ignores*
    ``field`` or ``baseline``, which would return every row of the subband.

    Parameters
    ----------
    field : str
        Comma-separated field names (empty = all).
    baseline : str
        CASA baseline selection, e.g. ``"EF&*"`` or ``"*&&&"`` (autocorrelations).
    scans : list, optional
        Scan numbers to keep.
    """
    selection = {}
    if field:
        selection["field"] = ",".join(f for f in str(field).split(",") if f)
    if baseline:
        selection["baseline"] = baseline
    if scans:
        selection["scan"] = ",".join(str(int(s)) for s in scans)
    if selection:
        try:
            ms_tool.msselect(selection)
        except RuntimeError as exc:
            raise BackendError(f"selecting {selection} failed: {exc}") from exc


def _channel_average(values, flags: np.ndarray, pol_indices: list[int]) -> np.ndarray:
    """Vector-average a ``(npol, nchan, nrow)`` block over channels -> ``(nrow, n_selected_pol)``."""
    masked = np.where(flags, np.nan, np.asarray(values))[pol_indices, :, :]
    with np.errstate(invalid="ignore"):
        return np.nanmean(masked, axis=1).T


def _finite_mean(block: np.ndarray) -> np.ndarray:
    """Mean over axis 0 ignoring non-finite entries (``nan`` where a column has none)."""
    finite = np.isfinite(block)
    with np.errstate(invalid="ignore"):
        mean_value = np.where(finite, block, 0.0).sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
    return np.where(finite.any(axis=0), mean_value, np.nan)


def compact_subband_selection(subbands) -> str:
    """Return a CASA spw selection for ``subbands`` as contiguous runs, e.g. ``"1~3,5"``.

    Parameters
    ----------
    subbands : sequence of int
        Subband (spw) indices; empty gives ``""`` (meaning every subband).

    Returns
    -------
    str
    """
    ordered = sorted({int(s) for s in subbands or ()})
    if not ordered:
        return ""
    runs, start, previous = [], ordered[0], ordered[0]
    for spw in ordered[1:]:
        if spw != previous + 1:
            runs.append(f"{start}~{previous}" if previous > start else str(start))
            start = spw
        previous = spw
    runs.append(f"{start}~{previous}" if previous > start else str(start))
    return ",".join(runs)


def central_channel_selection(n_channels: int, fraction: float, subbands=()) -> str:
    """Return a CASA spw selection string for the central ``fraction`` of channels.

    Parameters
    ----------
    n_channels : int
        Channels per subband.
    fraction : float
        Fraction of channels to keep (1.0 or less than 4 channels -> all channels).
    subbands : sequence of int, optional
        Subbands to solve on (empty = every subband). The channel range is repeated for each
        contiguous run of them: CASA applies a channel range only to the subbands of the token
        it is attached to, so ``"1~3,5:4~27"`` would take every channel of subbands 1-3.

    Returns
    -------
    str
        e.g. ``"*:3~28"`` for 32 channels at 0.8, ``"1~6:3~28"`` when only subbands 1-6 are
        selected, or ``"*"`` when nothing is trimmed.
    """
    spw_part = compact_subband_selection(subbands) or "*"
    if fraction >= 1.0 or n_channels < 4:
        return spw_part
    n_edge = int(round(n_channels * (1.0 - fraction) / 2.0))
    if n_edge < 1:
        return spw_part
    channels = f"{n_edge}~{n_channels - 1 - n_edge}"
    return ",".join(f"{run}:{channels}" for run in spw_part.split(","))


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
            # Flag versions of a previous (deleted) MMS make partition refuse to write.
            stale_flags = target.with_name(f"{target.name}.flagversions")
            if stale_flags.exists():
                shutil.rmtree(stale_flags)
            logger.info("partition: {} -> {} (Multi-MS for parallel I/O)", plain_ms.name, target.name)
            self.backend.tasks.partition(vis=str(plain_ms), outputvis=str(target), createmms=True,
                                         separationaxis="auto", numsubms="auto", flagbackup=False,
                                         datacolumn="all")
            shutil.rmtree(plain_ms)
        # Keep the flags the import itself made (data the correlator gave zero weight): a
        # later reset must be able to return to exactly this state. DiFX carries them only
        # in the visibility weights, so nothing else can put them back.
        try:
            self.backend.tasks.flagmanager(vis=str(target), mode="save", versionname=IMPORT_FLAG_VERSION,
                                           comment="flags as imported", merge="replace")
        except RuntimeError as exc:
            logger.warning("could not save the as-imported flags of {} ({})", target.name, exc)
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

        The weights are put back to their as-imported values as well
        (:meth:`_reset_weights`): ``applycal`` and the ``statwt`` reweighting of a
        previous attempt leave WEIGHT holding that attempt's calibration, which a
        new a-priori calibration would then scale a second time.

        The flags go back to those of the import (data the correlator gave zero
        weight, saved as the ``as_imported`` flag version); every flag made since
        is cleared, and the a-priori flagging step applies the observatory's again.

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
            imported = ms.with_name(f"{ms.name}.flagversions") / f"flags.{IMPORT_FLAG_VERSION}"
            try:
                if imported.exists():
                    # Back to the flags of the import (zero-weight data), not to no flags at all.
                    self.backend.tasks.flagmanager(vis=str(ms), mode="restore", versionname=IMPORT_FLAG_VERSION)
                else:
                    logger.warning("{} has no saved as-imported flags (imported by an older version): clearing "
                                   "every flag, including those of data the correlator gave zero weight; "
                                   "re-import to recover them", ms.name)
                    self.backend.tasks.flagdata(vis=str(ms), mode="unflag", flagbackup=False)
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: unflag failed: {exc}") from exc
        self._reset_weights(ms)
        after = self.backend.flag.flagged_fraction(project_code)
        logger.info("reset {}: flagged {:.1%} -> {:.1%}; corrected data cleared{}",
                    ms.name, before, after,
                    f" (previous flags kept as {backup!r})" if backup else "")
        return {"flagged_before": before, "flagged_after": after, "backup": backup}

    def _reset_weights(self, ms: Path, *, chunk_rows: int = 200_000) -> None:
        """Put WEIGHT and SIGMA of ``ms`` back to their as-imported values and drop the per-channel weights.

        ``importfitsidi`` gives a visibility the weight ``2 x channel width x
        integration time`` (the number of independent samples, e.g. 2e6 for 0.5 MHz
        and 2 s), with ``SIGMA = 1 / sqrt(WEIGHT)``, and weight 0 / sigma 1 where the
        correlator recorded nothing. Fringe-fit SNRs are scaled by these weights, so
        they must not simply be set to 1: that makes every detection look a thousand
        times weaker and the antenna selection rejects the whole array.

        Rows with no data in a polarization (all-zero visibilities: an antenna that
        did not record that subband) get weight 0 again, so they can never be
        averaged in after an unflag. The correlator's valid-data fraction of
        partially filled integrations is the one thing that cannot be recovered
        without re-importing; those integrations get the full nominal weight.

        WEIGHT_SPECTRUM and SIGMA_SPECTRUM are not written by the import; ``statwt``
        adds them (tens of GB on a large dataset) and recreates them when it runs.
        """
        widths = self._channel_widths_by_ddid(ms)
        handle = self.backend.tools.table()
        if not handle.open(str(ms), nomodify=False):
            raise BackendError(f"could not open {ms} to reset its weights")
        try:
            dropped = [c for c in ("WEIGHT_SPECTRUM", "SIGMA_SPECTRUM") if c in handle.colnames()]
            if dropped:
                handle.removecols(dropped)
            n_rows, n_empty = int(handle.nrows()), 0
            for start in range(0, n_rows, chunk_rows):
                count = min(chunk_rows, n_rows - start)
                data = np.asarray(handle.getcol("DATA", startrow=start, nrow=count))           # (npol, nchan, nrow)
                recorded = (data != 0).any(axis=1)                                             # (npol, nrow)
                del data
                exposure = np.asarray(handle.getcol("EXPOSURE", startrow=start, nrow=count), dtype=float)
                ddid = np.asarray(handle.getcol("DATA_DESC_ID", startrow=start, nrow=count))
                nominal = 2.0 * widths[ddid] * exposure                                         # (nrow,)
                weight = np.where(recorded, nominal[np.newaxis, :], 0.0).astype(np.float32)
                with np.errstate(divide="ignore"):
                    sigma = np.where(weight > 0, 1.0 / np.sqrt(weight), 1.0).astype(np.float32)
                handle.putcol("WEIGHT", weight, startrow=start, nrow=count)
                handle.putcol("SIGMA", sigma, startrow=start, nrow=count)
                n_empty += int((~recorded).sum())
        finally:
            handle.close()
        logger.info("reset {}: WEIGHT = 2 x channel width x integration time over {} row(s) ({:.0%} of the "
                    "row/polarization cells carry no data and got weight 0){}", ms.name, n_rows,
                    n_empty / max(4 * n_rows, 1), f"; dropped {', '.join(dropped)}" if dropped else "")

    def _channel_widths_by_ddid(self, ms: Path) -> np.ndarray:
        """Channel width in Hz of every DATA_DESC_ID of ``ms`` (the first channel of its spectral window)."""
        handle = self.backend.tools.table()
        if not handle.open(str(ms / "DATA_DESCRIPTION")):
            raise BackendError(f"could not open {ms}/DATA_DESCRIPTION")
        try:
            spw_ids = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
        finally:
            handle.close()
        if not handle.open(str(ms / "SPECTRAL_WINDOW")):
            raise BackendError(f"could not open {ms}/SPECTRAL_WINDOW")
        try:
            widths = np.array([abs(float(np.asarray(handle.getcell("CHAN_WIDTH", int(row))).ravel()[0]))
                               for row in range(handle.nrows())])
        finally:
            handle.close()
        return widths[spw_ids]

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
            # DiFX labels every array it correlates "VLBA", so an LBA dataset saying so is expected.
            expected = {observatory.upper()} | _TELESCOPE_ALIASES.get(observatory.upper(), set())
            if telescope and observatory and telescope.upper() not in expected:
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
        """Return ``{antenna: (subbands with observable data, ...)}`` via one TaQL pass.

        Heterogeneous arrays record different subband subsets per antenna. This
        reads the FLAG and DATA columns, so it is the expensive part of
        inspection and is deliberately kept out of :meth:`get_metadata`.

        A subband only counts when it has *unflagged and recorded* data
        (``DATA != 0``), not merely unflagged rows: an antenna that never
        correlated a subband still has rows there (zero-filled), and after an
        unflag (``data.reset_calibration(unflag=True)``, e.g. a ``--scratch``
        re-run) those rows are indistinguishable from real data by FLAG alone.
        Counting them as "participation" would let an antenna that structurally
        never recorded a subband look like it took part in it.
        """
        ms = self.backend.ms_path(project_code)
        logger.info("inspecting subband participation ({} antennas; reads the FLAG/DATA columns)",
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
                query = (f"SELECT {column}, DATA_DESC_ID, gntrue(!FLAG && DATA != 0) AS NVALID "
                         f"FROM {ms} WHERE ANTENNA1 != ANTENNA2 GROUPBY {column}, DATA_DESC_ID")
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
                logger.warning("antenna {} has no observable data in any subband", name)
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
                select_ms_rows(ms_tool, field=field, baseline=f"{refant}&*", scans=scans)
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
            for spw, scan_chunk in ((spw, chunk) for spw in range(n_spw)
                                    for chunk in _scan_chunks(scans)):
                ms_tool.selectinit(datadescid=spw)
                try:
                    select_ms_rows(ms_tool, field=",".join(wanted), scans=scan_chunk)
                except BackendError:
                    ms_tool.reset()
                    continue     # nothing recorded in this subband for these scans
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1", "antenna2", "time"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.pop(column_name, None)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record.pop("flag"), dtype=bool)
                values = np.where(flags[pol_indices, :, :], np.nan, values[pol_indices, :, :])
                del flags
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

    def read_baseline_timeline(self, project_code: str, *, field: str, column: str = "corrected",
                               metadata: Optional[ObsMetadata] = None) -> dict:
        """Read one amplitude per baseline and integration, averaged over the whole band.

        Each visibility is vector-averaged over its channels and parallel hands and
        then over the subbands, so on calibrated data of a bright source every
        baseline gets its best signal-to-noise at the native time resolution - what
        is needed to see an antenna arrive on source a few seconds late. Read a few
        scans of one subband at a time, like the other whole-field readers.

        Parameters
        ----------
        field : str
            One field (a bright calibrator).
        column : str
            ``"corrected"`` (post-applycal) or ``"data"``.

        Returns
        -------
        dict
            ``antennas`` (every antenna name, the axis order), ``times`` (MJD seconds),
            ``scans`` (scan number per time), ``amplitude``
            (``(n_antenna, n_antenna, n_time)``, symmetric, ``nan`` where flagged) and
            ``integration`` (seconds).
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        scans = [s for s in meta.scans if s.source == field]
        if not scans:
            raise BackendError(f"{project_code}: no scans on {field!r}")
        pol_indices, _ = parallel_hand_indices(meta)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        antenna_names = list(meta.antennas)
        pieces: list[tuple] = []
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw, scan_chunk in ((spw, chunk) for spw in range(meta.freq_setup.n_subbands)
                                    for chunk in _scan_chunks(scans)):
                ms_tool.selectinit(datadescid=spw)
                try:
                    select_ms_rows(ms_tool, field=field, scans=scan_chunk)
                except BackendError:
                    ms_tool.reset()
                    continue     # nothing recorded in this subband for these scans
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1", "antenna2", "time", "scan_number"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.pop(column_name, None)
                if values is None or not values.size:
                    continue
                good = ~np.asarray(record.pop("flag"), dtype=bool)[pol_indices, :, :]
                count = good.sum(axis=(0, 1))
                total = np.where(good, values[pol_indices, :, :], 0.0).sum(axis=(0, 1))
                del values, good
                ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])
                keep = (count > 0) & (ant1 != ant2)
                pieces.append((ant1[keep], ant2[keep], np.asarray(record["time"], dtype=float)[keep],
                               np.asarray(record["scan_number"])[keep], total[keep] / count[keep]))
        finally:
            ms_tool.close()
        if not pieces:
            raise BackendError(f"{project_code}: no unflagged {column} data on {field!r}")
        ant1, ant2, row_time, row_scan, row_value = (np.concatenate([p[k] for p in pieces]) for k in range(5))
        times = np.unique(row_time)
        index = np.searchsorted(times, row_time)
        n_antenna = len(antenna_names)
        inside = (ant1 < n_antenna) & (ant2 < n_antenna)
        low, high = np.minimum(ant1, ant2)[inside], np.maximum(ant1, ant2)[inside]
        sums = np.zeros((n_antenna, n_antenna, times.size), dtype=complex)
        counts = np.zeros((n_antenna, n_antenna, times.size))
        np.add.at(sums, (low, high, index[inside]), row_value[inside])       # one vote per subband
        np.add.at(counts, (low, high, index[inside]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            amplitude = np.where(counts > 0, np.abs(sums) / np.maximum(counts, 1.0), np.nan)
        amplitude = np.fmax(amplitude, np.transpose(amplitude, (1, 0, 2)))
        scan_of_time = np.zeros(times.size, dtype=int)
        scan_of_time[index] = row_scan
        steps = np.diff(times)
        integration = float(np.min(steps[steps > 0])) if (steps > 0).any() else 0.0
        logger.info("read_baseline_timeline: {} on {}: {} integration(s) of {:.1f} s in {} scan(s)", column, field,
                    times.size, integration, np.unique(scan_of_time).size)
        return {"antennas": antenna_names, "times": times, "scans": scan_of_time, "amplitude": amplitude,
                "integration": integration, "field": field, "column": column}

    def read_timeseries(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                        refant: str = "", column: str = "corrected", max_time_bins: int = 300,
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
        wanted_scans = {int(number) for number in (scans or [])}
        selected_scans = [s for s in meta.scans
                          if (not wanted or s.source in wanted)
                          and (not wanted_scans or s.scan_number in wanted_scans)]
        if not selected_scans:
            raise BackendError(f"{project_code}: no scans matching field={field!r}, scans={scans or 'all'}")
        # Bin *within* scans. A global time grid puts a bin across each scan boundary,
        # and vector-averaging over the minutes-long gap there collapses the amplitude —
        # which draws a dip at both ends of every scan that is an artefact of the binning,
        # not a property of the data.
        span = max(s.time_end for s in selected_scans) - min(s.time_start for s in selected_scans)
        bin_width = max(span / max(int(max_time_bins), 1), 1.0)
        edges: list[float] = []
        for scan in sorted(selected_scans, key=lambda s: s.time_start):
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
                select_ms_rows(ms_tool, field=",".join(wanted), baseline=f"{refant}&*",
                               scans=sorted(wanted_scans))
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
                        time_bin: float = 10.0, with_model: bool = False,
                        metadata: Optional[ObsMetadata] = None) -> dict:
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
        with_model : bool
            Also bin the MODEL_DATA column (same bins, same flags) when the MS has one.

        Returns
        -------
        dict
            ``uvdist_mlambda`` (1-D), ``values`` (``(n_points, n_pol)`` complex),
            ``polarizations``, ``field``, ``column``, ``time_bin`` and, with a model,
            ``model`` (``{"uvdist_mlambda", "values"}``).
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        pol_indices, pol_labels = parallel_hand_indices(meta)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        light_speed = 299792458.0
        ms_path = self.backend.ms_path(project_code)
        with_model = with_model and self._has_column(ms_path, "MODEL_DATA")
        columns = [column_name, "flag", "uvw", "time", "antenna1", "antenna2"]
        if with_model:
            columns.append("model_data")

        distances: list[float] = []
        averaged: list[np.ndarray] = []
        model_averaged: list[np.ndarray] = []
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(ms_path)):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            wanted = [f for f in field.split(",") if f]
            chunks = _scan_chunks([s for s in meta.scans if not wanted or s.source in wanted])
            for spw, scan_chunk in ((spw, chunk) for spw in range(meta.freq_setup.n_subbands)
                                    for chunk in chunks):
                ms_tool.selectinit(datadescid=spw)
                try:
                    select_ms_rows(ms_tool, field=field, scans=scan_chunk)
                except BackendError:
                    ms_tool.reset()
                    continue     # nothing recorded in this subband for these scans
                try:
                    record = ms_tool.getdata(columns)
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                flags = np.asarray(record["flag"], dtype=bool)
                per_row = _channel_average(values, flags, pol_indices)          # (nrow, npol)
                model_row = (_channel_average(record["model_data"], flags, pol_indices)
                             if with_model else None)
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
                    if not np.isfinite(per_row[group]).any():
                        continue
                    u, v = uvw[0, group].mean(), uvw[1, group].mean()
                    distances.append(float(np.hypot(u, v) * frequency / light_speed / 1e6))
                    averaged.append(_finite_mean(per_row[group]))
                    if model_row is not None:
                        model_averaged.append(_finite_mean(model_row[group]))
        finally:
            ms_tool.close()

        uvdist = np.asarray(distances, dtype=float)
        empty = np.empty((0, len(pol_indices)), dtype=complex)
        points = np.asarray(averaged) if averaged else empty
        logger.info("read_uvdistance: {} averaged point(s) on {} ({} s bins, per subband{})",
                    uvdist.size, field or "all fields", time_bin, ", with model" if with_model else "")
        result = {"uvdist_mlambda": uvdist, "values": points, "polarizations": pol_labels,
                  "field": field, "column": column, "time_bin": time_bin}
        if with_model:
            result["model"] = {"uvdist_mlambda": uvdist,
                               "values": np.asarray(model_averaged) if model_averaged else empty}
        return result

    def _has_column(self, ms_path: Path, column: str) -> bool:
        """Return True if the main table of ``ms_path`` has ``column``."""
        table = self.backend.tools.table()
        if not table.open(str(ms_path)):
            return False
        try:
            return column in table.colnames()
        finally:
            table.close()

    def read_autocorr_spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                               column: str = "data", metadata: Optional[ObsMetadata] = None) -> dict:
        """Read the time-averaged autocorrelation amplitude spectrum of every antenna.

        Autocorrelations (``ANTENNA1 == ANTENNA2``) carry each station's own
        bandpass shape and RFI without depending on a fringe, so they are the
        first place to look at the raw data. Amplitudes (scalar average over
        time) of the parallel hands only, one subband at a time.

        Parameters
        ----------
        field : str
            Field selection (comma-separated names; empty = all).
        scans : list, optional
            Scan numbers to average over (default: all scans on ``field``).
        column : str
            ``"data"`` (raw, the default) or ``"corrected"``.

        Returns
        -------
        dict
            ``antennas``, ``spectra`` (``{antenna: (n_spw, n_chan, n_pol)}`` real
            amplitudes), ``n_spw``, ``n_channels``, ``polarizations``,
            ``frequencies_ghz``, ``scans``, ``field``, ``column``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        antenna_names = list(meta.antennas)
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        pol_indices, pol_labels = parallel_hand_indices(meta)
        accumulated: dict[int, np.ndarray] = {}
        counts: dict[int, np.ndarray] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(n_spw):
                ms_tool.selectinit(datadescid=spw)
                select_ms_rows(ms_tool, field=field, baseline="*&&&", scans=scans)
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                # Autocorrelations are shown unflagged: flag_autocorr marks every one of
                # them, so applying FLAG would blank the diagnostic entirely. Zero-filled
                # rows (a subband a station never recorded) still mask to NaN.
                amplitude = np.where(np.abs(values) == 0, np.nan, np.abs(values))[pol_indices, :, :]
                ant1 = np.asarray(record["antenna1"])
                for antenna_id in np.unique(ant1):
                    key = int(antenna_id)
                    if key >= len(antenna_names):
                        continue
                    chunk = amplitude[:, :, ant1 == antenna_id]          # (npol, nchan, nrow)
                    with np.errstate(invalid="ignore"):
                        summed = np.nansum(chunk, axis=2).T              # (nchan, npol)
                        valid = np.sum(np.isfinite(chunk), axis=2).T
                    if key not in accumulated:
                        accumulated[key] = np.zeros((n_spw, n_chan, summed.shape[1]), dtype=float)
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
        logger.info("read_autocorr_spectrum: {} antenna(s) on {} scan(s) of {} from the {} column",
                    len(spectra), len(scans or []) or "all", field or "all fields", column)
        return {"antennas": sorted(spectra, key=antenna_names.index), "spectra": spectra,
                "n_spw": n_spw, "n_channels": n_chan, "polarizations": pol_labels,
                "frequencies_ghz": [meta.freq_setup.frequencies_ghz(s) for s in range(n_spw)],
                "scans": list(scans or []), "field": field, "column": column}

    def read_subband_phases(self, project_code: str, *, fields: list[str], refant: str,
                            column: str = "corrected", channel_fraction: float = 0.8,
                            metadata: Optional[ObsMetadata] = None) -> dict:
        """Per scan and baseline to ``refant``, the mean phase of every subband (parallel hands).

        Vector-averages the central ``channel_fraction`` of each subband over
        time within a scan, so each (scan, antenna, subband, pol) collapses to
        one phase. Differences between subbands are the residual instrumental
        phase jumps the single-band delay should have removed.

        Returns
        -------
        dict
            ``antennas``, ``scans`` (``[{"scan", "source", "time"}, ...]``),
            ``phases`` (``{antenna: (n_scan, n_spw, n_pol)`` degrees, nan = no data),
            ``polarizations``, ``n_spw``, ``refant``, ``column``, ``fields``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        antenna_names = list(meta.antennas)
        if refant not in antenna_names:
            raise BackendError(f"{project_code}: reference antenna {refant!r} not in the data")
        refant_id = antenna_names.index(refant)
        n_spw, n_chan = meta.freq_setup.n_subbands, meta.freq_setup.n_channels
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        pol_indices, pol_labels = parallel_hand_indices(meta)
        scans = [s for s in meta.scans if s.source in set(fields)]
        scan_index = {s.scan_number: i for i, s in enumerate(scans)}
        margin = int(round(n_chan * (1.0 - channel_fraction) / 2.0))
        channels = slice(margin, max(margin + 1, n_chan - margin))
        phases: dict[int, np.ndarray] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(n_spw):
                ms_tool.selectinit(datadescid=spw)
                select_ms_rows(ms_tool, field=",".join(fields), baseline=f"{refant}&*",
                               scans=[s.scan_number for s in scans])
                try:
                    record = ms_tool.getdata([column_name, "flag", "antenna1", "antenna2", "scan_number"])
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: could not read the {column} column: {exc}") from exc
                ms_tool.reset()
                values = record.get(column_name)
                if values is None or not values.size:
                    continue
                values = np.where(np.asarray(record["flag"], dtype=bool), np.nan, values)
                values = values[pol_indices, channels, :]                 # (npol, nsel, nrow)
                ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])
                other = np.where(ant1 == refant_id, ant2, ant1)
                scan_col = np.asarray(record["scan_number"])
                for antenna_id in np.unique(other):
                    if int(antenna_id) == refant_id:
                        continue
                    key = int(antenna_id)
                    if key not in phases:
                        phases[key] = np.full((len(scans), n_spw, len(pol_indices)), np.nan)
                    for scan_number in np.unique(scan_col[other == antenna_id]):
                        rows = (other == antenna_id) & (scan_col == scan_number)
                        chunk = values[:, :, rows]
                        if not np.isfinite(chunk).any():
                            continue
                        with np.errstate(invalid="ignore"):
                            mean = np.nanmean(chunk.reshape(chunk.shape[0], -1), axis=1)
                        phases[key][scan_index[int(scan_number)], spw, :] = np.degrees(np.angle(mean))
        finally:
            ms_tool.close()
        named = {antenna_names[k]: v for k, v in phases.items()}
        logger.info("read_subband_phases: {} baseline(s) to {} over {} scan(s) of {} ({} column)",
                    len(named), refant, len(scans), ",".join(fields), column)
        return {"antennas": sorted(named, key=antenna_names.index), "phases": named, "refant": refant,
                "scans": [{"scan": s.scan_number, "source": s.source, "time": 0.5 * (s.time_start + s.time_end)}
                          for s in scans],
                "polarizations": pol_labels, "n_spw": n_spw, "column": column, "fields": list(fields)}

    def read_total_visibility(self, project_code: str, *, fields: Optional[list[str]] = None,
                              column: str = "corrected", metadata: Optional[ObsMetadata] = None) -> dict:
        """Coherently sum every unflagged cross-correlation per field and integration.

        For each timestamp the visibilities of all baselines, channels and
        parallel hands are added as complex numbers; on a calibrated point
        source that sum divided by the count is the source's flux density, so
        its run against time is the total-power light curve. Read one scan at
        a time so a long track never sits in memory at once.

        Parameters
        ----------
        fields : list of str, optional
            Sources to read (default: every source with scans).
        column : str
            ``"corrected"`` (post-applycal) or ``"data"``.

        Returns
        -------
        dict
            ``sources`` (``{name: {"times", "vis_sum", "n_vis", "scans"}}`` as numpy arrays),
            ``column``, ``time_start`` (MJD seconds of the first integration).
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        wanted = list(fields or meta.source_names)
        column_name = {"corrected": "corrected_data", "data": "data"}[column]
        pol_indices, _ = parallel_hand_indices(meta)
        per_source: dict[str, dict[str, list]] = {}
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for scan in sorted(meta.scans, key=lambda s: s.time_start):
                if scan.source not in wanted:
                    continue
                sums: dict[float, complex] = {}
                nums: dict[float, int] = {}
                for spw in range(meta.freq_setup.n_subbands):
                    ms_tool.selectinit(datadescid=spw)
                    select_ms_rows(ms_tool, field=scan.source, scans=[scan.scan_number])
                    try:
                        record = ms_tool.getdata([column_name, "flag", "time", "antenna1", "antenna2"])
                    except RuntimeError as exc:
                        raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                    ms_tool.reset()
                    values = record.get(column_name)
                    if values is None or not values.size:
                        continue
                    cross = np.asarray(record["antenna1"]) != np.asarray(record["antenna2"])
                    ok = ~np.asarray(record["flag"], dtype=bool)[pol_indices][:, :, cross]
                    chunk = np.where(ok, values[pol_indices][:, :, cross], 0.0)
                    times = np.asarray(record["time"], dtype=float)[cross]
                    row_sum, row_num = chunk.sum(axis=(0, 1)), ok.sum(axis=(0, 1))
                    for stamp in np.unique(times):
                        rows = times == stamp
                        sums[stamp] = sums.get(stamp, 0.0) + complex(row_sum[rows].sum())
                        nums[stamp] = nums.get(stamp, 0) + int(row_num[rows].sum())
                entry = per_source.setdefault(scan.source, {"times": [], "vis_sum": [], "n_vis": [],
                                                            "scans": []})
                for stamp in sorted(sums):
                    if nums[stamp] <= 0:
                        continue
                    entry["times"].append(stamp)
                    entry["vis_sum"].append(sums[stamp])
                    entry["n_vis"].append(nums[stamp])
                    entry["scans"].append(int(scan.scan_number))
        finally:
            ms_tool.close()

        sources = {name: {"times": np.asarray(e["times"], dtype=float),
                          "vis_sum": np.asarray(e["vis_sum"], dtype=complex),
                          "n_vis": np.asarray(e["n_vis"], dtype=float),
                          "scans": np.asarray(e["scans"], dtype=np.int64)}
                   for name, e in per_source.items() if e["times"]}
        time_start = min((float(e["times"].min()) for e in sources.values()),
                         default=float(meta.time_range[0]) if meta.time_range else 0.0)
        logger.info("read_total_visibility: {} source(s), {} integration(s) from the {} column",
                    len(sources), sum(e["times"].size for e in sources.values()), column)
        return {"sources": sources, "column": column, "time_start": time_start}

    def read_uv_coverage(self, project_code: str, *, field: str = "", max_points: int = 200_000,
                         metadata: Optional[ObsMetadata] = None) -> dict:
        """Read the sampled (u, v) points per source, in wavelengths.

        The UVW column is per row and identical across subbands, so only the
        first data description is read and the metres are converted with a
        single band-centre frequency (``freq_setup.ref_freq``): the per-subband
        spread in wavelengths is a few percent and irrelevant for a coverage plot.
        Autocorrelations and rows flagged entirely are dropped; fields with more
        than ``max_points`` samples keep every k-th row (deterministic).

        Parameters
        ----------
        field : str
            Comma-separated fields to read; empty means all observed sources.
        max_points : int
            Subsampling threshold per field.

        Returns
        -------
        dict
            ``fields`` (``{name: {"u": [...], "v": [...]}}``), ``unit`` (``"Mlambda"``),
            ``freq_ghz``.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        wanted = [f for f in field.split(",") if f]
        names = {fid: name for name, fid in meta.source_ids.items()}
        frequency = float(meta.freq_setup.ref_freq)
        to_mlambda = frequency / 299792458.0 / 1e6

        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            ms_tool.selectinit(datadescid=0)
            select_ms_rows(ms_tool, field=",".join(wanted))
            try:
                record = ms_tool.getdata(["uvw", "flag", "flag_row", "antenna1", "antenna2",
                                          "field_id"])
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: could not read uvw: {exc}") from exc
        finally:
            ms_tool.close()

        uvw = np.asarray(record.get("uvw", np.empty((3, 0))), dtype=float)
        fields: dict[str, dict[str, list[float]]] = {}
        if uvw.size:
            ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])
            flagged = np.asarray(record["flag_row"], dtype=bool) | np.asarray(
                record["flag"], dtype=bool).all(axis=(0, 1))
            keep = (ant1 != ant2) & ~flagged
            field_ids = np.asarray(record["field_id"])
            for fid in np.unique(field_ids[keep]):
                rows = np.flatnonzero(keep & (field_ids == fid))
                step = max(1, int(np.ceil(rows.size / max_points)))
                rows = rows[::step]
                fields[names.get(int(fid), f"field{int(fid)}")] = {
                    "u": (uvw[0, rows] * to_mlambda).tolist(),
                    "v": (uvw[1, rows] * to_mlambda).tolist()}
        logger.info("read_uv_coverage: {} field(s), {} point(s) on {}", len(fields),
                    sum(len(f["u"]) for f in fields.values()), field or "all fields")
        return {"fields": fields, "unit": "Mlambda", "freq_ghz": frequency / 1e9}

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
                    # GAIN cells are variable-shaped in some imported MSs, so getcol
                    # cannot stack them; read row by row instead.
                    try:
                        gain = np.asarray(table.getcol("GAIN"))
                    except Exception:  # noqa: BLE001
                        cells = [np.ravel(table.getcell("GAIN", r)) for r in range(n_gc)]
                        gc_trivial = all(np.all(c == 1.0) for c in cells)
                    else:
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
        provisional Earth-orientation parameters (VLBA/LBA; not EVN). With
        ``needs_accor=True`` (DiFX: VLBA/LBA) the correlator amplitude correction
        comes first, see :meth:`accor`; ``accor`` is a dict of its settings.

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
        self._fit_gain_curves_to_casa(project_code, metadata=kwargs.get("metadata"))
        tables: list[CalTable] = []
        paths = self.backend.apriori_table_paths(project_code, needs_eop)
        if kwargs.get("needs_accor"):
            accor = self.accor(project_code, field, **dict(kwargs.get("accor") or {}))
            if accor is not None:
                tables.append(accor)

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

    def accor(self, project_code: str, field: str = "", *, solint: str = "30s", smoothtype: str = "median",
              smoothtime: float = 1800.0, **kwargs) -> Optional[CalTable]:
        """Correlator amplitude correction from the autocorrelations (CASA ``accor`` + ``smoothcal``).

        DiFX (VLBA, LBA) does not normalise the cross-correlations by the
        digitiser statistics; the autocorrelations, which should be exactly
        unity, measure that error per antenna, subband and polarization.
        ``accor`` solves for it every ``solint`` into ``<code>.accor``, and
        ``smoothcal`` then smooths the solutions in time (``smoothtype``,
        ``smoothtime`` seconds) into ``<code>.accor_smooth``, the table that is
        applied. It must run while the autocorrelations are still unflagged.

        Parameters
        ----------
        project_code : str
            Project code (the MS must exist).
        field : str
            Recorded in the returned CalTable (the solve uses every field).
        solint : str
            Solution interval of ``accor``.
        smoothtype : str
            ``smoothcal`` filter: ``"median"`` or ``"mean"``.
        smoothtime : float
            ``smoothcal`` filter width in seconds; 0 applies the unsmoothed table.

        Returns
        -------
        CalTable or None
            The table to apply, or ``None`` when the data hold no usable
            autocorrelations (reported as an anomaly, not an error).
        """
        ms = self.backend.ms_path(project_code)
        raw = self.backend.caldir() / f"{project_code}{APRIORI_TABLE_SPECS['accor'][0]}"
        smoothed = raw.with_name(f"{raw.name}_smooth")
        for path in (raw, smoothed):
            if path.exists():
                shutil.rmtree(path)
        logger.info("accor solint='{}' -> {}", solint, raw.name)
        try:
            # corrdepflags: a station with one dead polarization has that autocorrelation flagged
            # (DiFX gives it zero weight). Without this, accor drops the whole row and the working
            # polarization gets no solution either, which then flags the antenna altogether.
            self.backend.tasks.accor(vis=str(ms), caltable=str(raw), solint=str(solint), corrdepflags=True)
        except RuntimeError as exc:
            warnings.anomaly(f"{project_code}: accor failed ({exc}); the correlator amplitude correction is "
                             "not applied")
            return None
        if not raw.is_dir() or self._all_flagged(raw):
            warnings.anomaly(f"{project_code}: accor found no usable autocorrelations (flagged or not in the "
                             "data); the correlator amplitude correction is not applied")
            return None
        applied = raw
        if float(smoothtime) > 0:
            logger.info("smoothcal smoothtype='{}' smoothtime={:g} s -> {}", smoothtype, float(smoothtime),
                        smoothed.name)
            try:
                self.backend.tasks.smoothcal(vis=str(ms), tablein=str(raw), caltable=str(smoothed),
                                             smoothtype=str(smoothtype), smoothtime=float(smoothtime))
                applied = smoothed
            except RuntimeError as exc:
                warnings.warn(f"{project_code}: smoothcal of the accor table failed ({exc}); applying it unsmoothed")
        return CalTable(cal_type="accor", path=str(applied), field=field, interp=APRIORI_TABLE_SPECS["accor"][1])

    def _all_flagged(self, table_path: Path) -> bool:
        """True when a calibration table has no row, or no unflagged solution."""
        handle = self.backend.tools.table()
        handle.open(str(table_path))
        try:
            return handle.nrows() == 0 or bool(np.all(handle.getcol("FLAG")))
        finally:
            handle.close()

    def _fit_gain_curves_to_casa(self, project_code: str, *, metadata: Optional[ObsMetadata] = None,
                                 tolerance: float = 1e-3) -> dict[str, dict]:
        """Shorten gain-curve polynomials that are longer than CASA can apply.

        ``gencal(caltype='gc')`` copies at most :data:`_CASA_GC_MAX_COEFFICIENTS`
        coefficients per polarization out of the GAIN_CURVE subtable and silently
        drops the rest. A station whose ``.antab`` polynomial has one term more
        (JB in EM163: nine) is then corrected with a curve that is right at low
        elevation and wrong by orders of magnitude towards the zenith - its
        amplitudes follow the source's elevation over the whole track.

        Such a polynomial is replaced, in the GAIN_CURVE subtable, by the
        lowest-degree fit that reproduces it within ``tolerance`` over the
        elevations at which the antenna actually observed (a high-order
        polynomial is only meaningful where it was fitted, and wild outside). The
        untouched subtable is saved once as ``<caltables>/<code>.gain_curve.original``.

        Returns
        -------
        dict
            ``{antenna: {"coefficients_before", "coefficients_after", "elevation_range",
            "max_deviation", "truncation_error"}}`` for the antennas that were refitted
            (``truncation_error`` is the largest factor by which CASA's cut-off would
            have changed the amplitudes).
        """
        ms = self.backend.ms_path(project_code)
        if not (ms / "GAIN_CURVE").is_dir():
            return {}
        names = self._table_antenna_names(ms)
        handle = self.backend.tools.table()
        if not handle.open(str(ms / "GAIN_CURVE"), nomodify=False):
            raise BackendError(f"could not open {ms}/GAIN_CURVE")
        report: dict[str, dict] = {}
        try:
            n_poly = np.asarray(handle.getcol("NUM_POLY"))
            too_long = np.flatnonzero(n_poly > _CASA_GC_MAX_COEFFICIENTS)
            if not too_long.size:
                return {}
            backup = self.backend.caldir() / f"{project_code}.gain_curve.original"
            if not backup.exists():
                handle.copy(str(backup), deep=True, valuecopy=True).close()
            antenna_ids = np.asarray(handle.getcol("ANTENNA_ID"))
            meta = metadata or self.backend.data.get_metadata(project_code, [], "")
            for row in too_long:
                name = names[int(antenna_ids[row])] if int(antenna_ids[row]) < len(names) else f"#{int(antenna_ids[row])}"
                kind = str(handle.getcell("TYPE", int(row))).upper()
                gain = np.asarray(handle.getcell("GAIN", int(row)), dtype=float)            # (npol, n_poly)
                el_lo, el_hi = antenna_elevation_range(meta, name)
                lo, hi = (90.0 - el_hi, 90.0 - el_lo) if "ZA" in kind else (el_lo, el_hi)
                fits = [refit_polynomial(coefficients, lo, hi, tolerance=tolerance) for coefficients in gain]
                width = max(len(coefficients) for coefficients, _ in fits)
                new_gain = np.zeros((gain.shape[0], width))
                for pol, (coefficients, _) in enumerate(fits):
                    new_gain[pol, :len(coefficients)] = coefficients
                x = np.linspace(lo, hi, 200)
                full = np.polynomial.polynomial.polyval(x, gain[0])
                cut = np.polynomial.polynomial.polyval(x, gain[0, :_CASA_GC_MAX_COEFFICIENTS])
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.sqrt(np.abs(cut / full)) if "POWER" in kind else np.abs(cut / full)
                handle.putcell("GAIN", int(row), new_gain)
                handle.putcell("NUM_POLY", int(row), int(width))
                report[name] = {"coefficients_before": int(gain.shape[1]), "coefficients_after": int(width),
                                "elevation_range": (float(el_lo), float(el_hi)),
                                "max_deviation": float(max(deviation for _, deviation in fits)),
                                "truncation_error": float(np.nanmax(np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-12))))}
        finally:
            handle.close()
        for name, entry in sorted(report.items()):
            logger.info("gain curve: {} has a {}-coefficient polynomial but CASA applies only the first {} (amplitudes "
                        "off by up to a factor {:.3g} at the elevations observed); refitted with {} coefficients over "
                        "{:.0f}-{:.0f} deg, within {:.2%} of the original", name, entry["coefficients_before"],
                        _CASA_GC_MAX_COEFFICIENTS, entry["truncation_error"], entry["coefficients_after"],
                        *entry["elevation_range"], entry["max_deviation"])
        return report

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
                            suffix: str = "sbd", stages: Optional[list] = None,
                            max_window_sec: float = 120.0, callib: bool = False,
                            **kwargs) -> CalTable:
        """Single-band (instrumental) delay calibration: one time-constant solution per antenna.

        Solves one delay per antenna, subband and polarization with the rates
        forced to zero: the instrumental delay is a fixed property of the signal
        path, so letting the rate float only adds noise. Only the central
        ``channel_fraction`` of each subband is used, since the edges roll off
        and have not been bandpass-corrected yet. The solve uses a *single* scan
        and at most ``max_window_sec`` of it: several scans would give several
        solutions in time and the applied phases would not be continuous. When
        ``stages`` chains more than one scan (no scan detects every antenna), the
        later stages are re-based onto the first reference; see :meth:`_staged_sbd`.

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
        stages : list of dict, optional
            ``[{"scan", "antennas", "refant"}, ...]`` from ``selection.plan_sbd_stages``.
        max_window_sec : float
            Longest stretch of a scan used for the solve (centred in the scan).
        callib : bool
            Use a CASA cal-library file for the on-the-fly priors instead of
            explicit parallel lists.

        Returns
        -------
        CalTable
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        spw = self._solve_spw(meta, channel_fraction)
        table_path = self.backend.caldir() / f"{project_code}.{suffix}"
        base = {"field": field, "spw": spw, "solint": solint, "zerorates": True, "minsnr": minsnr,
                "corrdepflags": True, "parang": True}
        if not stages:
            # No plan: one solve over the requested scans (or the whole field).
            params = dict(base, caltable=str(table_path), refant=self.refant_chain(meta, refant))
            if scans:
                params["scan"] = ",".join(str(s) for s in scans)
                params["timerange"] = self._solve_window(meta, scans[0], max_window_sec) if len(scans) == 1 else ""
            logger.info("initial_calibration[{}]: fringefit field={} scans={} timerange={} spw={} minsnr={}",
                        suffix, field, params.get("scan", "all"), params.get("timerange") or "all", spw, minsnr)
            self._fringefit(project_code, params, gaintable, field=field, callib=callib)
        else:
            self._staged_sbd(project_code, meta, stages, base, table_path, gaintable,
                             max_window_sec, suffix, callib=callib)
        return CalTable(cal_type=suffix, path=str(table_path), field=field, interp="nearest",
                        snr=self._median_table_snr(table_path))

    def _solve_window(self, meta: ObsMetadata, scan_number: int, max_window_sec: float) -> str:
        """CASA ``timerange`` for the central ``max_window_sec`` of a scan ("" = whole scan when shorter)."""
        scan = next((s for s in meta.scans if s.scan_number == scan_number), None)
        if scan is None or scan.duration_sec <= max_window_sec or max_window_sec <= 0:
            return ""
        mid = 0.5 * (scan.time_start + scan.time_end)
        start, end = mid - max_window_sec / 2.0, mid + max_window_sec / 2.0
        epoch = dt.datetime(1858, 11, 17)
        fmt = "%Y/%m/%d/%H:%M:%S"
        return f"{(epoch + dt.timedelta(seconds=start)).strftime(fmt)}~{(epoch + dt.timedelta(seconds=end)).strftime(fmt)}"

    def _staged_sbd(self, project_code: str, meta: ObsMetadata, stages: list[dict], base: dict,
                    table_path: Path, gaintable: Optional[list], max_window_sec: float, suffix: str,
                    *, callib: bool = False) -> None:
        """Solve the single-band delay stage by stage and merge into one table.

        Stage 1 solves on its scan for its antennas, referenced to its refant, and
        becomes ``table_path``. Every later stage solves only the baselines among
        its new antennas plus its refant (an antenna already solved), on its own
        scan, into a temporary table. Those solutions are referenced to that
        stage's refant, so the refant's own stage-1 solution is added to them
        (phase, delay and dispersive delay) before the rows are appended, which
        puts every antenna on the stage-1 reference. Nothing is applied to the
        data in between and nothing is flagged: the priors are attached on the
        fly for all stages alike.
        """
        for index, stage in enumerate(stages):
            first = index == 0
            antennas = list(stage["antennas"]) + ([] if first else [stage["refant"]])
            target = table_path if first else table_path.with_name(f"{table_path.name}.stage{index + 1}")
            params = dict(base, caltable=str(target), scan=str(stage["scan"]), refant=stage["refant"],
                          antenna=",".join(antennas) + "&",
                          timerange=self._solve_window(meta, stage["scan"], max_window_sec))
            logger.info("initial_calibration[{}] stage {}/{}: fringefit scan={} timerange={} antennas={} refant={}",
                        suffix, index + 1, len(stages), stage["scan"], params["timerange"] or "whole scan",
                        ",".join(antennas), stage["refant"])
            self._fringefit(project_code, params, gaintable,
                             field=base.get("field", ""), callib=callib)
            if not first:
                added = self._rebase_and_append(table_path, target, meta, stage["refant"], stage["antennas"])
                shutil.rmtree(target, ignore_errors=True)
                logger.info("initial_calibration[{}] stage {}: {} solution row(s) re-based onto the stage-1 "
                            "reference and appended", suffix, index + 1, added)

    def _rebase_and_append(self, main_path: Path, stage_path: Path, meta: ObsMetadata, refant: str,
                           antennas: list[str]) -> int:
        """Append ``stage_path`` rows for ``antennas`` to ``main_path``, re-based onto the main reference.

        A fringe table stores per row FPARAM ``(4 * npol, 1)`` = (phase, delay,
        rate, dispersive) per polarization, relative to the row's reference
        antenna. Adding the ``refant`` row of the main table (same subband)
        to each stage row expresses it relative to the main table's reference.
        Rows of the stage refant itself (zero by construction) are dropped.
        """
        table = self.backend.tools.table()
        table.open(str(main_path / "ANTENNA"))
        try:
            names = [str(n) for n in table.getcol("NAME")]
        finally:
            table.close()
        wanted = {names.index(a) for a in antennas if a in names}
        ref_id = names.index(refant)
        table.open(str(main_path))
        try:
            main_ant = np.asarray(table.getcol("ANTENNA1"))
            main_spw = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"))
            main_par = np.asarray(table.getcol("FPARAM"))          # (npar, nchan, nrow)
            main_flag = np.asarray(table.getcol("FLAG"))
            main_ref = np.asarray(table.getcol("ANTENNA2"))
        finally:
            table.close()
        offsets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for row in np.where(main_ant == ref_id)[0]:
            offsets[int(main_spw[row])] = (main_par[:, :, row], main_flag[:, :, row])
        if not offsets:
            raise BackendError(f"stage reference antenna {refant} has no solution in {main_path.name}")
        main_reference = int(np.bincount(main_ref[main_ref >= 0]).argmax()) if np.any(main_ref >= 0) else ref_id

        table.open(str(stage_path), nomodify=False)
        try:
            ant = np.asarray(table.getcol("ANTENNA1"))
            spw = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"))
            par = np.asarray(table.getcol("FPARAM"))
            flag = np.asarray(table.getcol("FLAG"))
            keep = [r for r in range(len(ant)) if int(ant[r]) in wanted and int(spw[r]) in offsets]
            for row in keep:
                ref_par, ref_flag = offsets[int(spw[row])]
                par[:, :, row] += ref_par
                flag[:, :, row] |= ref_flag
            # Wrap the phase terms (every 4th parameter starting at 0) back into (-pi, pi].
            par[0::4] = (par[0::4] + np.pi) % (2 * np.pi) - np.pi
            # An antenna already in the main table is here for the polarization an earlier
            # stage could not give it: fill only what is still flagged there, never replace.
            existing = {(int(a), int(w)): r for r, (a, w) in enumerate(zip(main_ant, main_spw))}
            merged = [row for row in keep if (int(ant[row]), int(spw[row])) in existing]
            for row in merged:
                target = existing[(int(ant[row]), int(spw[row]))]
                fill = main_flag[:, :, target] & ~flag[:, :, row]
                main_par[:, :, target][fill] = par[:, :, row][fill]
                main_flag[:, :, target][fill] = False
            table.putcol("FPARAM", par)
            table.putcol("FLAG", flag)
            table.putcol("ANTENNA2", np.full(len(ant), main_reference, dtype=main_ref.dtype))
            drop = [r for r in range(len(ant)) if r not in set(keep) or r in set(merged)]
            if drop:
                table.removerows(drop)
            table.flush()
            appended = table.nrows()
            if appended:
                table.copyrows(str(main_path))
        finally:
            table.close()
        if merged:
            table.open(str(main_path), nomodify=False)
            try:
                # Rows appended above come after the ones read earlier, so those are untouched.
                table.putcol("FPARAM", main_par, 0, main_par.shape[2])
                table.putcol("FLAG", main_flag, 0, main_flag.shape[2])
                table.flush()
            finally:
                table.close()
        return int(appended + len(merged))

    def bandpass(self, project_code: str, field: str, refant: str, *,
                 scans: Optional[list] = None, gaintable: Optional[list] = None,
                 solint: str = "inf", combine: str = "scan", minsnr: float = 3.0,
                 solnorm: bool = True, fillgaps: int = 8, bandtype: str = "B",
                 metadata: Optional[ObsMetadata] = None, callib: bool = False,
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
        fillgaps : int
            Interpolate over flagged channel gaps up to this width (channels), so a
            few RFI-flagged channels do not punch holes in the band shape.
        bandtype : str
            ``"B"`` (per channel) or ``"BPOLY"`` (polynomial).
        callib : bool
            Use a CASA cal-library file for the on-the-fly priors instead of
            explicit parallel lists.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        table_path = self.backend.caldir() / f"{project_code}.bpass"
        # Solve into a temporary table and only swap it in on success. A re-solve
        # (second/third pass) can legitimately produce nothing when cumulative
        # flagging has left the fringe finder with too few unflagged antennas;
        # deleting the existing table up front would then leave the run with no
        # bandpass at all. Keeping the previous table lets the caller fall back.
        new_path = self.backend.caldir() / f"{project_code}.bpass.new"
        if new_path.exists():
            shutil.rmtree(new_path)
        params = {"vis": str(self.backend.ms_path(project_code)), "caltable": str(new_path),
                  "field": field, "solint": solint, "combine": combine, "solnorm": solnorm,
                  "refant": self.refant_chain(meta, refant), "minsnr": minsnr, "bandtype": bandtype,
                  "corrdepflags": True, "parang": True, "fillgaps": int(fillgaps)}
        if scans:
            params["scan"] = ",".join(str(s) for s in scans)
        params.update(self._prior_callib(project_code, gaintable, new_path,
                                           field=field, callib=callib))
        logger.info("bandpass: field={} scans={} solint={} combine={} (prior: {})",
                    field, params.get("scan", "all"), solint, combine,
                    ", ".join(t.cal_type for t in gaintable or []) or "none")
        try:
            self._run_bandpass_task(params)
        except RuntimeError as exc:
            if new_path.exists():
                shutil.rmtree(new_path)
            raise BackendError(f"{project_code}: bandpass failed (field={field!r}): {exc}") from exc
        if not new_path.is_dir():
            raise BackendError(f"{project_code}: bandpass produced no table (field={field!r}); "
                               "the solve interval had too few unflagged antennas")
        # Success: replace the previous table atomically.
        if table_path.exists():
            shutil.rmtree(table_path)
        new_path.rename(table_path)
        return CalTable(cal_type="bpass", path=str(table_path), field=field,
                        interp="nearest,nearest", snr=self._median_table_snr(table_path))

    def fringefit(self, project_code: str, field: str, refant: str, *,
                  gaintable: Optional[list] = None, solint: str = "inf", combine: str = "spw",
                  minsnr: float = 5.0, zerorates: bool = False, suffix: str = "mbd",
                  dispersive: bool = False, channel_fraction: float = 0.8,
                  metadata: Optional[ObsMetadata] = None, callib: bool = False,
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
        channel_fraction : float
            Fraction of central channels per subband used for the solve (the
            solution is applied to the whole band). 1.0 uses every channel.
        callib : bool
            Use a CASA cal-library file for the on-the-fly priors instead of
            explicit parallel lists.

        Returns
        -------
        CalTable
            With ``spwmap`` filled in ready for apply.
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        table_path = self.backend.caldir() / f"{project_code}.{suffix}"
        # Solve on the central channels only, like the SBD: the subband edges roll off and
        # their phase is the least trustworthy part of the band, so letting them into the
        # fit biases the delay. The solution is still applied to every channel.
        spw = self._solve_spw(meta, channel_fraction)
        params = {"caltable": str(table_path), "field": field, "spw": spw, "solint": solint,
                  "combine": combine, "zerorates": zerorates, "corrdepflags": True,
                  "refant": self.refant_chain(meta, refant), "minsnr": minsnr, "parang": True}
        if dispersive:
            # paramactive = [delay, rate, dispersive delay]; zerorates is separate,
            # it zeroes the fitted rates in the output rather than not fitting them.
            params["paramactive"] = [True, True, True]
        logger.info("fringefit({}): field={} spw={} solint={} combine={} minsnr={} dispersive={} "
                    "(prior: {})", suffix, field, spw, solint, combine, minsnr, dispersive,
                    ", ".join(t.cal_type for t in gaintable or []) or "none")
        self._fringefit(project_code, params, gaintable, field=field, callib=callib)
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
                          combine: str = "scan", minsnr: float = 3.0, max_factor: float = 2.0,
                          metadata: Optional[ObsMetadata] = None, callib: bool = False,
                          **kwargs) -> CalTable:
        """Solve one amplitude gain per antenna and subband (a "scalar bandpass").

        Even after the bandpass, the subbands of an antenna can sit at slightly
        different amplitude levels: the bandpass is normalised per subband, so it
        removes the *shape* within each one but not a constant offset between
        them. Those steps survive into the combined image as an effective
        mis-weighting of the band.

        Amplitude only (``calmode='a'``), one solution per antenna per subband
        for the whole observation. The solutions are then normalised **per
        antenna and polarization across its subbands** (median = 1), so the table
        only levels the subbands of each antenna and the flux scale the a-priori
        calibration gave that antenna is untouched.

        CASA's own ``solnorm`` must not be used for this: it normalises each
        subband across the *antennas*. That keeps each antenna's absolute level
        from a solve against a 1 Jy point model - wrong by the source structure
        on every baseline of a resolved calibrator (EM163: JB 0.40, T6 0.63,
        EF 1.25) - and makes the normalisation itself jump between subbands
        whenever the set of antennas observing them differs, which puts the same
        step into every antenna.

        A subband whose gain is more than a factor ``max_factor`` above or below
        the antenna's own level is not a step to level but a broken subband (band
        edge, no signal): its solution is flagged, so ``applycal`` flags the data.
        Anything closer is corrected, however large: EF's top subband is 40% down
        in one polarization on EM163 and perfectly usable once levelled.

        Parameters
        ----------
        max_factor : float
            Largest ratio between a subband's gain and its antenna's median gain
            (either way) that is still corrected rather than flagged.
        callib : bool
            Use a CASA cal-library file for the on-the-fly priors instead of
            explicit parallel lists.

        Returns
        -------
        CalTable
        """
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        table_path = self.backend.caldir() / f"{project_code}.scalar_bp"
        # Solve into a temporary table and swap only on success: like the bandpass,
        # a re-solve on heavily-flagged data can produce nothing, and destroying the
        # previous table first would leave the run with no scalar bandpass to fall
        # back to (see CasaCalibrationOps.bandpass).
        new_path = self.backend.caldir() / f"{project_code}.scalar_bp.new"
        if new_path.exists():
            shutil.rmtree(new_path)
        params = {"vis": str(self.backend.ms_path(project_code)), "caltable": str(new_path),
                  "field": field, "solint": solint, "combine": combine, "gaintype": "G",
                  "calmode": "a", "solnorm": False, "minsnr": minsnr, "parang": True,
                  "refant": self.refant_chain(meta, refant)}
        params.update(self._prior_callib(project_code, gaintable, new_path,
                                           field=field, callib=callib))
        logger.info("scalar_bandpass: gaincal calmode='a' field={} solint={} combine={} "
                    "(one amplitude per antenna and subband)", field, solint, combine)
        try:
            self._run_gaincal_task(params)
        except RuntimeError as exc:
            if new_path.exists():
                shutil.rmtree(new_path)
            raise BackendError(f"{project_code}: scalar bandpass failed (field={field!r}): "
                               f"{exc}") from exc
        if not new_path.is_dir():
            raise BackendError(f"{project_code}: scalar bandpass produced no table (field={field!r}); "
                               "the solve interval had too few unflagged antennas")
        if table_path.exists():
            shutil.rmtree(table_path)
        new_path.rename(table_path)
        dropped = self._normalise_per_antenna(table_path, max_factor=max_factor)
        spread = self._subband_gain_spread(table_path)
        logger.info("scalar_bandpass: subband-to-subband amplitude spread was {:.1%} "
                    "(median over antennas), normalised per antenna", spread)
        if dropped:
            logger.info("scalar_bandpass: {} antenna/subband solution(s) more than a factor {:g} off their antenna's "
                        "level were flagged (the data will be flagged on apply): {}", len(dropped), max_factor,
                        ", ".join(dropped[:12]) + (" ..." if len(dropped) > 12 else ""))
        return CalTable(cal_type="scalar_bp", path=str(table_path), field=field,
                        interp="nearest", snr=self._median_table_snr(table_path))

    def _normalise_per_antenna(self, table_path: Path, *, max_factor: float = 2.0) -> list[str]:
        """Scale a gain table so each antenna/polarization has median amplitude 1 across its subbands.

        Solutions more than a factor ``max_factor`` above or below that median are
        flagged; solutions the solver already flagged are left as they are. Returns
        the labels of the newly flagged ones, ``"<antenna> spw <n> <pol index>"``.
        """
        handle = self.backend.tools.table()
        if not handle.open(str(table_path), nomodify=False):
            raise BackendError(f"could not open {table_path} to normalise it")
        try:
            gains = np.asarray(handle.getcol("CPARAM"))                 # (npol, 1, nrow)
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
            antennas = np.asarray(handle.getcol("ANTENNA1"))
            spws = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
            names = self._table_antenna_names(table_path)
            dropped: set[str] = set()
            for antenna in np.unique(antennas):
                rows = np.where(antennas == antenna)[0]
                for pol in range(gains.shape[0]):
                    amplitude = np.abs(gains[pol, 0, rows])
                    good = ~flags[pol, 0, rows] & np.isfinite(amplitude) & (amplitude > 0)
                    if not good.any():
                        continue
                    level = float(np.median(amplitude[good]))
                    gains[pol, 0, rows[good]] = gains[pol, 0, rows[good]] / level
                    ratio = amplitude / level
                    off = good & ((ratio > max_factor) | (ratio < 1.0 / max_factor))
                    flags[pol, 0, rows[off]] = True
                    name = names[int(antenna)] if int(antenna) < len(names) else f"#{int(antenna)}"
                    dropped.update(f"{name} spw {int(s)} pol {pol}" for s in spws[rows[off]])
            handle.putcol("CPARAM", gains)
            handle.putcol("FLAG", flags)
        finally:
            handle.close()
        return sorted(dropped, key=lambda label: (label.split()[0], int(label.split()[2]), label.split()[-1]))

    def _table_antenna_names(self, table_path: Path) -> list[str]:
        """Antenna names of a calibration table (empty when its ANTENNA subtable cannot be read)."""
        handle = self.backend.tools.table()
        if not handle.open(str(Path(table_path) / "ANTENNA")):
            return []
        try:
            return [str(n) for n in handle.getcol("NAME")]
        finally:
            handle.close()

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

    def _fringefit(self, project_code: str, params: dict, gaintable: Optional[list], *,
                   field: str = "", callib: bool = False) -> None:
        """Run casatasks.fringefit with prior tables attached, replacing any existing table."""
        table_path = Path(params["caltable"])
        if table_path.exists():
            shutil.rmtree(table_path)
        params = dict(params)
        params["vis"] = str(self.backend.ms_path(project_code))
        params.update(self._prior_callib(project_code, gaintable, table_path,
                                           field=field, callib=callib))
        try:
            self._run_fringefit_task(params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: fringefit failed "
                               f"(field={params.get('field')!r}, scan={params.get('scan', 'all')}, "
                               f"refant={params.get('refant')!r}): {exc}") from exc
        if not table_path.is_dir():
            raise BackendError(f"{project_code}: fringefit produced no table at {table_path}")

    def _run_fringefit_task(self, params: dict) -> None:
        """Run the fringe-fit engine with CASA ``fringefit`` task parameters (the single override point).

        ``params`` is the complete keyword set of ``casatasks.fringefit`` (vis, caltable,
        selection, solve options and the on-the-fly prior lists). Subclasses that solve
        with another engine replace this method only; everything around it (table paths,
        staging, spw maps, SNR reading) stays shared.
        """
        self.backend.tasks.fringefit(**params)

    def _run_bandpass_task(self, params: dict) -> None:
        """Run the bandpass solver (the override point for alternate engines)."""
        self.backend.tasks.bandpass(**params)

    def _run_gaincal_task(self, params: dict) -> None:
        """Run gaincal (the override point for alternate engines)."""
        self.backend.tasks.gaincal(**params)

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

    def reweight(self, project_code: str, *, column: str = "corrected", timebin: str = "",
                 minsamp: int = 2, flagbackup: bool = True, **kwargs) -> dict:
        """Recompute the visibility weights from the scatter of the calibrated data (statwt).

        The correlator weights only reflect the nominal integration; after the
        a-priori and fringe calibration the real per-baseline noise is known from
        the data itself. Anomalously high or low weights afterwards point at bad
        data that hid until now, so the pipeline flags again and re-solves after
        this step.

        Parameters
        ----------
        column : str
            Data column the scatter is measured on (``"corrected"`` after applycal).
        timebin : str
            Time window per weight estimate (empty = per scan/subband default).
        minsamp : int
            Minimum number of unflagged visibilities per estimate.
        flagbackup : bool
            Save a flag version first (statwt flags what it cannot weight).

        Returns
        -------
        dict
            statwt's own report (``mean`` and ``variance`` of the new weights).
        """
        params = {"vis": str(self.backend.ms_path(project_code)), "datacolumn": column,
                  "minsamp": int(minsamp), "flagbackup": bool(flagbackup)}
        if timebin:
            params["timebin"] = timebin
        params.update({k: v for k, v in kwargs.items() if v not in (None, "")})
        logger.info("statwt: {}", " ".join(f"{k}={v!r}" for k, v in params.items() if k != "vis"))
        try:
            result = self.backend.tasks.statwt(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: statwt failed: {exc}") from exc
        report = dict(result) if isinstance(result, dict) else {"result": result}
        logger.info("statwt: new weights mean={} variance={}", report.get("mean"), report.get("variance"))
        return report

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

    def _resolve_table_gainfield(self, table: CalTable, apply_field: str) -> str:
        """Return the ``gainfield`` entry for ``table`` when applying to ``apply_field``.

        Tables without a declared ``table.gainfield`` are not field-mapped; their
        ``table.field`` may contain provenance even when the CASA table itself is
        field-independent. When every requested field was solved into a transferable
        table, CASA's ``nearest`` mapping uses each field's own solution. Otherwise
        the stored ``gainfield`` supplies the phase-reference solution.
        """
        if not table.gainfield or not apply_field:
            return table.gainfield
        requested = {f.strip() for f in apply_field.split(",") if f.strip()}
        solved = {f.strip() for f in table.field.split(",") if f.strip()}
        if requested and requested <= solved:
            return "nearest"
        return table.gainfield

    @staticmethod
    def _resolve_table_interp(table: CalTable, mapping: str) -> str:
        """Return the ``interp`` entry for ``table`` given its resolved field ``mapping``.

        A field corrected with its own per-scan solutions (``mapping == "nearest"``)
        takes each scan's own solution, i.e. ``nearest`` in time. Interpolating
        linearly between scans of the same field instead makes CASA flag an antenna
        that has a single good solution on that field — one that joined only one
        scan of a fringe finder loses that scan, and with it its bandpass and
        instrumental delay (EM163: HH and IB on the only scan with every antenna).
        Phase referencing (a named ``gainfield``) keeps the table's own interpolation.
        """
        interp = [part.strip() for part in str(table.interp or "linear").split(",")]
        if table.gainfield and mapping == "nearest":
            interp[0] = "nearest"
        return ",".join(interp)

    @staticmethod
    def _tables_for_field(tables: list[CalTable], apply_field: str) -> list[CalTable]:
        """Tables that correct ``apply_field``: those without ``apply_to`` or whose ``apply_to`` names it.

        ``apply_to`` restricts a table to some fields (the phase calibrator's
        self-cal gains must not reach the fringe finders). An empty
        ``apply_field`` means no selection, so every table is kept.
        """
        wanted = {f for f in str(apply_field).split(",") if f}
        if not wanted:
            return list(tables)
        return [t for t in tables
                if not getattr(t, "apply_to", "") or wanted & {f for f in t.apply_to.split(",") if f}]

    def _compile_apply_params(self, project_code: str, tables: list[CalTable], apply_field: str, *,
                              include_calwt: bool = False, callib: bool = False,
                              filename: str = "caltables.txt", gainfield: str = "") -> dict:
        """Return the applycal parameters for ``tables`` when correcting ``apply_field``.

        By default (``callib=False``) the returned dict contains the explicit
        parallel ``gaintable``, ``gainfield``, ``interp`` and ``spwmap`` lists, plus
        ``calwt`` when ``include_calwt`` is true.  With ``callib=True`` a CASA cal
        library file is written with the same per-table, per-field resolution and
        the dict contains ``docallib=True`` and ``callib``.
        """
        if callib:
            callib_path = self.write_callib(project_code, tables, field=apply_field,
                                              filename=filename, gainfield=gainfield)
            return {"docallib": True, "callib": str(callib_path)}
        gaintable: list[str] = []
        resolved_gainfield: list[str] = []
        interp: list[str] = []
        spwmap: list[list[int]] = []
        calwt: list[bool] = []
        for table in self._tables_for_field(tables, apply_field):
            gaintable.append(str(table.path))
            mapping = gainfield or self._resolve_table_gainfield(table, apply_field)
            resolved_gainfield.append(mapping)
            interp.append(self._resolve_table_interp(table, mapping))
            spwmap.append(list(table.spwmap))
            if include_calwt:
                calwt.append(bool(getattr(table, "calwt", True)))
        params: dict = {"gaintable": gaintable, "gainfield": resolved_gainfield,
                        "interp": interp, "spwmap": spwmap}
        if include_calwt:
            params["calwt"] = calwt
        return params

    def apply(self, project_code: str, field: str, tables: list[CalTable], *,
              gainfield: str = "", parang: bool = True, flagbackup: bool = False,
              callib: bool = False, **kwargs) -> None:
        """Apply the accumulated calibration tables to a field (applycal).

        By default the explicit parallel ``gaintable``/``gainfield``/``interp``/
        ``spwmap``/``calwt`` lists are built from each :class:`~vlbipy.models.CalTable`
        and passed directly to CASA.  With ``callib=True`` a cal-library file is
        written instead and ``docallib=True`` is used.

        When ``field`` is empty the accumulated tables are applied once per
        observed source field so that phase-referencing mappings are resolved per
        source rather than globally.

        Parameters
        ----------
        project_code : str
            Project code.
        field : str
            Field selection to correct (empty = all observed source fields).
        tables : list of CalTable
            Tables to apply, in order.
        gainfield : str
            Optional global field mapping override (empty = resolved per table).
        parang : bool
            Apply the parallactic-angle correction (always on for VLBI).
        callib : bool
            Use a CASA cal-library file instead of explicit parallel lists.
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
        if field:
            apply_fields = [field]
        else:
            meta = self.backend.data.get_metadata(project_code, [], "")
            apply_fields = sorted({s.source for s in meta.scans})
            if not apply_fields:
                apply_fields = [""]
        for apply_field in apply_fields:
            params = self._compile_apply_params(
                project_code, usable, apply_field, include_calwt=True, callib=callib,
                filename=f"caltables/{apply_field or 'all'}.txt", gainfield=gainfield)
            params.update(kwargs)
            params.update({"vis": str(ms), "parang": parang})
            params.setdefault("flagbackup", flagbackup)
            params.setdefault("applymode", "calflagstrict")
            if apply_field:
                params["field"] = apply_field
            applied = self._tables_for_field(usable, apply_field)
            logger.info("applycal: {} -> field={} ({} tables: {})", ms.name, apply_field or "all",
                        len(applied), ", ".join(t.cal_type for t in applied))
            try:
                self.backend.tasks.applycal(**params)
            except RuntimeError as exc:
                raise BackendError(f"{project_code}: applycal failed (field={apply_field!r}, tables="
                                   f"{[t.cal_type for t in usable]}): {exc}") from exc

    def apply_callib(self, project_code: str, callib: str, *, field: str = "", parang: bool = True,
                     **kwargs) -> None:
        """Apply an existing CASA calibration-library file directly to a measurement set."""
        params = dict(kwargs)
        params.update({"vis": str(self.backend.ms_path(project_code)), "docallib": True,
                       "callib": str(callib), "parang": parang})
        params.setdefault("flagbackup", False)
        params.setdefault("applymode", "calflagstrict")
        if field:
            params["field"] = field
        try:
            self.backend.tasks.applycal(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: applycal failed (callib={callib!r}): {exc}") from exc

    def _prior_callib(self, project_code: str, tables: Optional[list], table_path: Path, *,
                      field: str = "", callib: bool = False) -> dict:
        """Return the applycal-on-the-fly parameters for a solve's prior tables.

        By default the explicit parallel ``gaintable``/``gainfield``/``interp``/
        ``spwmap`` lists are returned (no ``calwt``: solving tasks do not accept
        it).  With ``callib=True`` a CASA cal library is written instead, named after
        the table it produces, and ``docallib=True``/``callib`` are returned.  The
        ``field`` argument is the current solve field so the per-table mapping can
        resolve ``nearest`` for self-calibration tables.  Returns an empty dict when
        there are no priors.
        """
        if not tables:
            return {}
        # Interactive callers may pass bare paths; treat those as plain tables.
        tables = [t if isinstance(t, CalTable) else CalTable(Path(str(t)).suffix.lstrip(".") or "prior",
                                                             path=str(t))
                  for t in tables]
        return self._compile_apply_params(
            project_code, tables, field, include_calwt=False, callib=callib,
            filename=f"callibs/{table_path.name}.txt")

    def write_callib(self, project_code: str, tables: list, *, gainfield: str = "",
                     field: str = "", filename: str = "caltables.txt") -> Path:
        """Write a CASA cal-library file listing the tables to apply, and return its path.

        The cal library is applycal's declarative form: one line per table with
        its own interpolation, field mapping and spectral-window mapping. Using
        it instead of parallel ``gaintable``/``interp``/``spwmap``/``gainfield``
        lists means the exact calibration applied is a readable artefact next to
        the data — which is the record you need months later to know what was
        done, and what a reviewer would ask for.

        ``fldmap`` is resolved per table and per ``field``: a table solved on the
        requested field uses ``nearest``, a target field uses the table's own
        ``gainfield`` (phase referencing), and field-independent tables carry no
        ``fldmap`` at all.  A non-empty ``gainfield`` argument overrides that
        resolution for callers that need backward-compatible behaviour.

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
            Optional global field mapping override (empty = resolved per table).
        field : str
            The field being corrected, used to resolve the per-table mapping.
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
            mapping = gainfield or self._resolve_table_gainfield(table, field)
            interp = self._resolve_table_interp(table, mapping).split(",")
            entry = [f"caltable='{table.path}'",
                     f"calwt={bool(getattr(table, 'calwt', True))}",
                     f"tinterp='{interp[0] or 'linear'}'"]
            if len(interp) > 1 and interp[1]:
                entry.append(f"finterp='{interp[1]}'")
            if mapping:
                entry.append(f"fldmap='{mapping}'")
            if table.spwmap:
                entry.append(f"spwmap={list(table.spwmap)}")
            if getattr(table, "apply_to", ""):
                entry.insert(1, f"field='{table.apply_to}'")
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
                 minsnr: float = 0.0, max_scans: int = 0, callib: bool = False,
                 reuse: bool = False, **kwargs) -> ScanSNRSurvey:
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
        callib : bool
            Use a CASA cal-library file for the on-the-fly priors instead of
            explicit parallel lists.
        reuse : bool
            Read an existing ``<code>.snr`` table instead of solving again (a
            resumed run: the fringe fit is the expensive part).

        Returns
        -------
        ScanSNRSurvey
        """
        ms = self.backend.ms_path(project_code)
        if not ms.is_dir():
            raise BackendError(f"{project_code}: measurement set {ms} not found; "
                               "run import_data() before the SNR survey")
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        spw = self._solve_spw(meta, channel_fraction)
        refant = self.refant_chain(meta, refant)
        scans = self._survey_scans(meta, field, scans, max_scans)
        table_path = self.backend.caldir() / f"{project_code}.snr"
        if reuse and table_path.is_dir():
            logger.info("scan_snr: reading the existing survey table {}", table_path.name)
            return self._read_snr_table(table_path, meta, refant, channel_fraction, field)
        if table_path.exists():
            shutil.rmtree(table_path)

        ff_kwargs = {"vis": str(ms), "caltable": str(table_path), "field": field, "spw": spw,
                     "solint": "inf", "combine": "spw", "refant": refant, "zerorates": False,
                     "corrdepflags": True, "minsnr": float(minsnr), "parang": True}
        # The parallel-list form needs the spwmap and gainfield aligned per table;
        # the cal-library path resolves them per table into a file.  Both preserve
        # field-specific mappings (nearest for self-calibration, table.gainfield for
        # phase referencing) instead of a single global override.
        ff_kwargs.update(self._prior_callib(project_code, gaintable, table_path,
                                               field=field, callib=callib))
        if scans:
            ff_kwargs["scan"] = ",".join(str(s) for s in scans)
        logger.info("scan_snr: fringefit over {} (spw={}, refant chain {}, central {:.0%} of channels)",
                    field or "all fields", spw, refant, channel_fraction)
        try:
            self._run_fringefit_task(ff_kwargs)
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

    def _solve_spw(self, metadata: ObsMetadata, channel_fraction: float) -> str:
        """Return the CASA spw selection every solve uses: central channels of the usable band.

        The usable band is the subbands at least two antennas recorded
        (:attr:`~vlbipy.models.ObsMetadata.usable_subbands`). A subband a single antenna recorded
        has no baseline, so including it gives nothing to fit while widening the band a
        ``combine='spw'`` solve references its delay to. The selection is left as ``"*"`` when
        every subband is usable, which is the homogeneous-array case.

        Parameters
        ----------
        metadata : ObsMetadata
            Observation metadata (supplies the channel count and subband participation).
        channel_fraction : float
            Fraction of central channels of each subband to solve on.
        """
        usable, n_subbands = metadata.usable_subbands, metadata.freq_setup.n_subbands
        # No restriction when every subband is usable, nor when participation left nothing to go on.
        subbands = usable if 0 < len(usable) < n_subbands else ()
        if subbands:
            logger.info("solving on the usable band only: subband(s) {} of {} have at least two antennas",
                        compact_subband_selection(subbands), n_subbands)
        return central_channel_selection(metadata.freq_setup.n_channels, channel_fraction, subbands)

    def read_snr_table(self, project_code: str, metadata: ObsMetadata,
                       refant: str = "") -> Optional[ScanSNRSurvey]:
        """Reload the stored SNR survey table ``<caldir>/<code>.snr`` without re-running fringefit.

        Returns ``None`` when the table does not exist. The survey's field selection is
        not stored in the table, so every scan is kept (rows only exist for the
        surveyed calibrator scans anyway).
        """
        table_path = self.backend.caldir() / f"{project_code}.snr"
        if not table_path.is_dir():
            return None
        return self._read_snr_table(table_path, metadata, refant, channel_fraction=0.8, field="")

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


def _leading_low_seconds(offsets: np.ndarray, amps: np.ndarray, sigma: float,
                         integration: float, max_seconds: float = 120.0) -> float:
    """Return how many seconds of leading-low data exist in a single baseline/scan timeseries.

    Computes the stable level from samples beyond a guard region (avoiding the ramp itself
    depressing the median), then counts consecutive low samples from the start.

    Parameters
    ----------
    offsets : numpy.ndarray
        Time offsets from scan start (seconds), already sorted ascending.
    amps : numpy.ndarray
        Amplitude values corresponding to *offsets*, same length.
    sigma : float
        Number of MAD-based sigmas below the stable level that counts as "low".
    integration : float
        Integration time (seconds); added to the last low sample's offset to give the stretch.
    max_seconds : float
        Upper bound on the returned value.

    Returns
    -------
    float
        Duration of the leading low stretch in seconds, or 0.0 when none is detected.
    """
    from ..statistics import mad_sigma
    if offsets.size < 5:
        return 0.0
    # Guard region: estimate the stable level from samples well past any plausible ramp.
    scan_span = float(offsets[-1] - offsets[0]) + integration
    guard = max(3.0 * integration, min(max_seconds, scan_span / 4.0))
    stable_mask = offsets > guard
    if np.count_nonzero(stable_mask) >= 5:
        stable = amps[stable_mask]
        level = float(np.median(stable))
        noise = mad_sigma(stable)
    else:
        # Fall back to whole-scan median/MAD when too few stable samples.
        level = float(np.median(amps))
        noise = mad_sigma(amps)
    if not np.isfinite(noise) or noise <= 0:
        return 0.0
    low = amps < level - sigma * noise
    if not low.size or not low[0]:
        return 0.0
    end = 0
    while end < low.size and low[end]:
        end += 1
    return min(float(offsets[end - 1] + integration), max_seconds)


def _scan_consensus(lows: list[float], min_agree: int = 2) -> float:
    """Combine per-baseline leading-low durations into a single scan estimate via median consensus.

    A genuinely off-source antenna drags down *every* baseline it participates in.
    Requiring at least half (and at least *min_agree*) of the baselines to report a
    positive stretch separates "this antenna slewed" from "one far-end baseline had a
    glitch". The median of the positive values is robust to one outlier baseline that
    reports a much shorter or longer ramp.

    Parameters
    ----------
    lows : list[float]
        Per-baseline leading-low durations for one antenna in one scan (may contain zeros).
    min_agree : int
        Minimum number of baselines that must report a positive stretch for consensus.

    Returns
    -------
    float
        Median of the positive values if consensus is reached, otherwise 0.0.
    """
    if not lows:
        return 0.0
    positive = [v for v in lows if v > 0]
    n_pos = len(positive)
    # Require at least half of all baselines AND at least min_agree.
    if n_pos < min_agree or n_pos < len(lows) / 2.0:
        return 0.0
    return float(np.median(positive))


class CasaFlagOps(FlagOps):
    """Flagging via casatasks.flagdata (plus AOFlagger when it is installed)."""

    #: flagdata parameters per mode; ``kind`` -> (mode, extra kwargs).
    # Auto-flaggers must never extend flags across baselines: VLBI antenna sensitivities
    # differ by orders of magnitude, so what is an outlier on one baseline is signal
    # on another. tfcrop/rflag already judge each baseline on its own statistics.
    MODES = {"autocorr": ("manual", {"autocorr": True}),
             "quack": ("quack", {"quackmode": "beg"}),
             "tfcrop": ("tfcrop", {"datacolumn": "data", "timecutoff": 4.0, "freqcutoff": 3.0,
                                   "extendflags": False}),
             "rflag": ("rflag", {"datacolumn": "corrected", "extendflags": False}),
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
            per_antenna = kwargs.pop("per_antenna", None) or {}
            if per_antenna:
                # One command per antenna, so a station with a clean band keeps it.
                commands = []
                for name, (left, right) in sorted(per_antenna.items()):
                    spw = self._edge_spw_selection(project_code, left=left, right=right, **kwargs)
                    if spw:
                        commands.append(f"antenna='{name}' spw='{spw}'")
                if not commands:
                    logger.info("flag[edges]: no antenna needs trimming")
                    return
                logger.info("flagdata edges: {} per-antenna command(s), e.g. {}",
                            len(commands), commands[0])
                try:
                    self.backend.tasks.flagdata(vis=str(ms), mode="list", inpfile=commands,
                                                flagbackup=False, action="apply")
                except RuntimeError as exc:
                    raise BackendError(f"{project_code}: flagdata[edges] failed: {exc}") from exc
                return
            spw = self._edge_spw_selection(project_code, **kwargs)
            if not spw:
                logger.info("flag[edges]: nothing to trim (subbands are too narrow or flat)")
                return
            params.update(mode="manual", spw=spw)
        else:
            mode, extra = self.MODES[kind]
            params["mode"] = mode
            params.update(extra)
            if kind == "quack":
                params["quackinterval"] = float(kwargs.pop("interval", 0.0) or 0.0)
                if kwargs.get("antenna"):
                    params["antenna"] = str(kwargs["antenna"])
                if params["quackinterval"] <= 0.0:
                    logger.info("flag[quack]: interval is 0 s; nothing to do")
                    return
            if kind == "from_file":
                params["inpfile"] = str(kwargs.pop("flagfile"))
            if kind in ("manual", "tfcrop", "rflag"):
                # Caller overrides (e.g. datacolumn='corrected' once calibrated, or a
                # tighter cutoff) win over the mode defaults.
                params.update({k: v for k, v in kwargs.items() if v not in (None, "")})
        logger.info("flagdata {}", " ".join(f"{k}={v!r}" for k, v in params.items() if k != "vis"))
        try:
            self.backend.tasks.flagdata(**params)
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: flagdata[{kind}] failed: {exc}") from exc

    def _spike_departures(self, amplitude: np.ndarray, finite: np.ndarray, threshold: float,
                          smooth_bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the time bins where a baseline's amplitude spikes or drops, and every bin's departure.

        The second array is the fractional departure of each time bin from the local
        level (``nan`` where there is none), for callers that need its size as well.

        The decision is made on the baseline's *level* — the median over channels of
        each time bin — not on individual time/frequency cells. A single cell is one
        baseline, one channel, 25 seconds of data: thermal noise alone moves it by tens
        of percent (on rsm07, 90% of cells sit within 50% of the local level), so no
        per-cell cut can separate noise from defects. Averaging the channels first
        divides that noise by roughly the square root of their number and leaves a
        smooth series in which a real spike or drop stands out.

        The level is compared with its own running median rather than one value for the
        whole track: a baseline's amplitude drifts as the source rises and sets and as
        the gains wander — largest on the long baselines — and judging that drift
        against a single median condemns whole stretches of good data. Frequency
        structure (RFI) is left to the tfcrop/aoflagger steps, which run earlier.

        Parameters
        ----------
        amplitude : numpy.ndarray
            Amplitude array, time x frequency.
        finite : numpy.ndarray
            Boolean mask of usable samples.
        threshold : float
            Robust-sigma cut on the fractional departure from the local level.
        smooth_bins : int
            Window, in time bins, of the running median that tracks the level: long
            enough to average the noise, short enough to follow genuine drifts.

        Returns
        -------
        numpy.ndarray
            Indices of the offending time bins (empty when the baseline is clean).
        """
        from ..statistics import mad_sigma, running_median

        empty = (np.zeros(0, dtype=int), np.full(amplitude.shape[0], np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = np.nanmedian(np.where(finite, amplitude, np.nan), axis=1)
            level = running_median(profile, window=smooth_bins)
            level = np.where(np.isfinite(level) & (level > 0.0), level, np.nan)
            if not np.isfinite(level).any():
                return empty
            relative = profile / level - 1.0
        usable = np.isfinite(relative)
        if usable.sum() < 5:
            return empty[0], relative
        sigma = mad_sigma(relative[usable])
        if not np.isfinite(sigma) or sigma <= 0.0:
            return empty[0], relative
        return np.where(usable & (np.abs(relative) > threshold * sigma))[0], relative

    @staticmethod
    def _baseline_coherence(values: np.ndarray) -> float:
        """How much of a baseline's amplitude is signal: 1 for a strong source, towards 0 for noise.

        Per time bin, the amplitude of the channel-averaged visibility over the mean
        channel amplitude; the median over the bins. Noise averages down across the
        channels while a calibrated source does not.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(np.nanmean(values, axis=1)) / np.nanmean(np.abs(values), axis=1)
        ratio = ratio[np.isfinite(ratio)]
        return float(np.median(ratio)) if ratio.size else 0.0

    def _spike_bins(self, amplitude: np.ndarray, finite: np.ndarray, threshold: float, smooth_bins: int) -> np.ndarray:
        """Return only the time bins of :meth:`_spike_departures`."""
        return self._spike_departures(amplitude, finite, threshold, smooth_bins)[0]

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

        The decision is made per baseline, per field, on the baseline's channel-averaged
        level against its own running median (see :meth:`_spike_bins`), so what gets
        flagged is a time bin whose amplitude genuinely spikes or drops. Each part of
        that matters. Pooling the fields puts the brightest source far above the faint
        majority's median; judging against one median for the whole track condemns the
        slow drift of a long baseline; and judging individual time/frequency cells
        condemns ordinary thermal noise. Any of the three flags whole stretches of good
        data — and when that data belongs to the scan the instrumental delay is solved
        on, the next pass loses those antennas entirely and applycal then flags them
        across the whole observation.

        Amplitude drives the decision — it is where interference and correlator
        problems show up most cleanly — with the phase scatter reported for
        information.

        Parameters
        ----------
        field : str
            Restrict to one field (empty = all, each judged separately).
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
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        names = [f for f in field.split(",") if f] or sorted({s.source for s in meta.scans})
        smooth_bins = max(3, int(kwargs.pop("smooth_bins", 9)))
        max_baseline_fraction = float(kwargs.pop("max_baseline_fraction", 0.1))
        gross_departure = float(kwargs.pop("gross_departure", 0.25))
        gross_max_fraction = float(kwargs.pop("gross_max_fraction", 0.35))
        gross_min_bins = int(kwargs.pop("gross_min_bins", 30))
        gross_min_coherence = float(kwargs.pop("gross_min_coherence", 0.5))
        commands: list[str] = []
        per_baseline: list[tuple[str, float]] = []
        suspect: list[tuple[str, float]] = []
        total_points = 0
        total_outliers = 0

        for name in names:
            try:
                spectra = self.backend.data.read_dynamic_spectra(
                    project_code, field=name, column=column, max_time_bins=max_time_bins,
                    stokes_i=True, metadata=meta)
            except BackendError as exc:
                logger.info("outliers: skipping field {} ({})", name, exc)
                continue
            times = np.asarray(spectra["times"], dtype=float)
            half = (times[1] - times[0]) / 2.0 if times.size > 1 else 1.0
            stamp = "%Y/%m/%d/%H:%M:%S.%f"
            for (first, second), values in spectra["baselines"].items():
                amplitude = np.abs(values)
                finite = np.isfinite(amplitude)
                if finite.sum() < 20:
                    continue
                total_points += int(finite.sum())
                bad_bins, relative = self._spike_departures(amplitude, finite, threshold, smooth_bins)
                bad_bins = bad_bins[bad_bins < times.size]
                if not bad_bins.size:
                    continue
                # Spikes and drops are rare events. A criterion that wants to remove a
                # large share of one baseline has misfitted it — too few time bins to
                # measure a spread, or a level it cannot follow — and acting on that
                # verdict is how good antennas get erased from the calibration scan.
                # Report the baseline instead of gutting it.
                share = float(finite[bad_bins].sum()) / float(finite.sum())
                if share > max_baseline_fraction:
                    # Too many for the sigma cut to be trusted - but on a well-sampled baseline the
                    # bins that sit far from the local level in absolute terms (an antenna still
                    # slewing, a dropout) are bad whatever the statistics say. Keep only those.
                    # Only where the source itself sets the amplitude, though: on a baseline that
                    # is mostly noise (a faint target, a long spacing on a weak source) the level
                    # of a bin follows how many samples went into it, and cutting on it would
                    # flag good data and bias what is left.
                    populated = int(np.isfinite(relative).sum())
                    coherence = self._baseline_coherence(values)
                    gross = bad_bins[np.abs(relative[bad_bins]) > gross_departure]
                    gross_share = float(finite[gross].sum()) / float(finite.sum()) if gross.size else 0.0
                    if (populated < gross_min_bins or not gross.size or gross_share > gross_max_fraction
                            or coherence < gross_min_coherence):
                        logger.info("outliers: {} {}&{} looks {:.0%} anomalous — too much to be "
                                    "spikes; leaving it untouched", name, first, second, share)
                        suspect.append((f"{name} {first}&{second}", share))
                        continue
                    logger.info("outliers: {} {}&{} looks {:.0%} anomalous; flagging only the {} bin(s) more than "
                                "{:.0%} off the local level ({:.0%} of it)", name, first, second, share, gross.size,
                                gross_departure, gross_share)
                    bad_bins, share = gross, gross_share
                # One flag command per affected time bin, over the whole bin's width.
                flagged = int(finite[bad_bins].sum())
                total_outliers += flagged
                per_baseline.append((f"{name} {first}&{second}", share))
                for bin_index in bad_bins:
                    start = mjdsec2datetime(float(times[bin_index]) - half)
                    end = mjdsec2datetime(float(times[bin_index]) + half)
                    commands.append(f"antenna='{first}&{second}' field='{name}' "
                                    f"timerange='{start.strftime(stamp)[:-3]}~{end.strftime(stamp)[:-3]}'")

        fraction = (total_outliers / total_points) if total_points else 0.0
        per_baseline.sort(key=lambda item: item[1], reverse=True)
        suspect.sort(key=lambda item: item[1], reverse=True)
        report = {"n_outliers": total_outliers, "n_points": total_points, "fraction": fraction,
                  "per_baseline": per_baseline, "threshold": threshold, "commands": len(commands),
                  "left_untouched": suspect}
        logger.info("outliers: {} of {} points ({:.2%}) beyond {:.0f} sigma on {} baseline(s)",
                    total_outliers, total_points, fraction, threshold, len(per_baseline))
        if suspect:
            logger.info("outliers: {} baseline(s) looked broadly anomalous and were left "
                        "untouched (worst: {})", len(suspect),
                        ", ".join(f"{n} {s:.0%}" for n, s in suspect[:3]))
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

    def off_source(self, project_code: str, *, fields: list[str], transfer_fields: Optional[list[str]] = None,
                   level: float = 0.8, column: str = "corrected", gapless_seconds: float = 10.0,
                   margin: int = 1, min_share: float = 0.2, dry_run: bool = False,
                   metadata: Optional[ObsMetadata] = None, **kwargs) -> dict:
        """Flag antennas while they are not on source, measured on the calibrated data.

        The station flags that come with the data cover the slewing of most
        antennas, but not of all (no log at all for the e-MERLIN out-stations or
        IB in EM163) and not always to the end (WB arrives 4-14 s after its flag
        ends in nine scans out of ten). Before calibration this cannot be seen on a
        weak calibrator; after it, an antenna that is still slewing drags *all* its
        baselines down together, sample by sample.

        Two parts:

        ``fields`` (bright calibrators) are measured directly. Per integration the
        normalised amplitudes of every baseline are split into one factor per
        antenna (:func:`vlbipy.statistics.detect_off_source`); an antenna below
        ``level`` is flagged for that integration, plus ``margin`` integrations
        while it settles.

        ``transfer_fields`` (targets and other sources too faint to measure) get the
        time each antenna *typically* needs to arrive: the 90th percentile, over the
        calibrator scans of the same kind (starting right after the previous scan,
        i.e. containing the slew, or after a gap), of the time from the scan start
        until the antenna was on source. That time is a property of the slew and is
        tight (EM163: e-MERLIN stations 10-12 s in three scans out of four, WB
        34-38 s in all of them). An antenna that is rarely late (in fewer than
        ``min_share`` of those scans) gets nothing, and one whose own station flags
        already cover its slew is never late in this sense.

        Parameters
        ----------
        fields : list of str
            Fields bright enough to measure on.
        transfer_fields : list of str, optional
            Fields that receive the per-antenna typical times instead.
        level : float
            On-source fraction below which an antenna is flagged.
        gapless_seconds : float
            A scan starting within this many seconds of the previous scan's end contains the slew.
        margin : int
            Integrations flagged after the last off-source one.
        dry_run : bool
            Measure and report without flagging.

        Returns
        -------
        dict
            ``commands``, ``per_antenna`` (seconds flagged directly), ``typical``
            (arrival after the scan start, ``{"gapless": {antenna: seconds}, "gap": {...}}``), ``n_direct``,
            ``n_transfer`` and, when applied, ``flagged_fraction_of_data``.
        """
        from ..statistics import detect_off_source
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        ordered = sorted(meta.scans, key=lambda s: s.time_start)
        previous_end = {s.scan_number: (ordered[k - 1].time_end if k else -np.inf) for k, s in enumerate(ordered)}
        scan_info = {s.scan_number: s for s in meta.scans}
        stamp = "%Y/%m/%d/%H:%M:%S.%f"

        def command(antenna: str, start: float, end: float) -> str:
            first, last = mjdsec2datetime(start), mjdsec2datetime(end)
            return f"antenna='{antenna}' timerange='{first.strftime(stamp)[:-3]}~{last.strftime(stamp)[:-3]}'"

        commands: list[str] = []
        per_antenna: dict[str, float] = {}
        leading: dict[str, dict[str, list[float]]] = {"gapless": {}, "gap": {}}
        for field in fields:
            try:
                data = self.backend.data.read_baseline_timeline(project_code, field=field, column=column, metadata=meta)
            except BackendError as exc:
                logger.info("off source: skipping field {} ({})", field, exc)
                continue
            times, scan_of_time, step = data["times"], data["scans"], data["integration"] or 1.0
            found = detect_off_source(data["amplitude"], scan_of_time, level=level)
            for index, name in enumerate(data["antennas"]):
                judged, off = found["judged"][index], found["off"][index]
                if not judged.any():
                    continue
                for scan_number in np.unique(scan_of_time[judged]):
                    samples = np.flatnonzero((scan_of_time == scan_number) & judged)
                    if samples.size < 5:
                        continue
                    scan = scan_info.get(int(scan_number))
                    kind = "gapless" if scan and scan.time_start - previous_end[scan.scan_number] <= gapless_seconds else "gap"
                    bad = off[samples]
                    lead = int(np.argmin(bad)) if not bad.all() else bad.size        # length of the leading run
                    begin = (scan.time_start if scan else times[samples[0]]) - step / 2.0
                    arrival = (times[samples[lead - 1]] + step / 2.0 - begin) if lead else 0.0
                    leading[kind].setdefault(name, []).append(arrival)
                    # one command per run of consecutive off-source integrations, with the settling margin
                    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
                    for first, last in zip(edges[::2], edges[1::2]):
                        start = times[samples[first]] - step / 2.0
                        if first == 0 and scan:
                            start = min(start, scan.time_start - step / 2.0)
                        end = times[samples[last - 1]] + step / 2.0 + margin * step
                        commands.append(command(name, start, end))
                        per_antenna[name] = per_antenna.get(name, 0.0) + (last - first) * step
        n_direct = len(commands)

        # What each antenna typically needs, for the fields too faint to measure.
        typical: dict[str, dict[str, float]] = {"gapless": {}, "gap": {}}
        for kind, by_antenna in leading.items():
            for name, durations in by_antenna.items():
                values = np.asarray(durations)
                if values.size >= 5 and float((values > 0).mean()) >= min_share:
                    typical[kind][name] = float(np.percentile(values, 90))
        wanted = [f for f in (transfer_fields or []) if f not in fields]
        for scan in meta.scans:
            if scan.source not in wanted:
                continue
            kind = "gapless" if scan.time_start - previous_end[scan.scan_number] <= gapless_seconds else "gap"
            step = scan.integration_time or 2.0
            for name, seconds in typical[kind].items():
                if seconds <= 0 or name not in scan.antennas:
                    continue
                begin = scan.time_start - step / 2.0
                commands.append(command(name, begin, min(begin + seconds + margin * step, scan.time_end + step / 2.0)))
        report = {"commands": commands, "per_antenna": per_antenna, "typical": typical, "n_direct": n_direct,
                  "n_transfer": len(commands) - n_direct, "level": float(level)}
        if per_antenna:
            logger.info("off source: {} stretch(es) flagged on {}: seconds per antenna {}", n_direct, ", ".join(fields),
                        ", ".join(f"{n} {s:.0f}" for n, s in sorted(per_antenna.items(), key=lambda item: -item[1])))
        else:
            logger.info("off source: every antenna is on source throughout {}", ", ".join(fields))
        for kind, label in (("gapless", "scans that contain the slew"), ("gap", "scans that start after a gap")):
            if typical[kind]:
                logger.info("off source: typical arrival after the scan start in {} (applied to {}): {}", label,
                            ", ".join(wanted) or "no other field",
                            ", ".join(f"{n} {s:.0f} s" for n, s in sorted(typical[kind].items(), key=lambda item: -item[1])))
        if dry_run or not commands:
            return report
        before = self.flagged_fraction(project_code)
        try:
            self.backend.tasks.flagdata(vis=str(self.backend.ms_path(project_code)), mode="list", inpfile=commands,
                                        flagbackup=False, action="apply")
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: flagging the off-source antennas failed: {exc}") from exc
        after = self.flagged_fraction(project_code)
        report["flagged_fraction_of_data"] = max(0.0, after - before)
        logger.info("flag[off source]: {:.2%} newly flagged ({:.1%} -> {:.1%} of observable data; {} direct, {} "
                    "transferred command(s))", report["flagged_fraction_of_data"], before, after, n_direct,
                    report["n_transfer"])
        return report

    def measure_quack(self, project_code: str, *, field: str = "", column: str = "corrected",
                      sigma: float = 2.0, max_seconds: float = 120.0,
                      metadata: Optional[ObsMetadata] = None, **kwargs) -> dict:
        """Measure the per-antenna slew time from calibrated data, one antenna at a time.

        For each scan and each baseline, the stable level is estimated from
        samples beyond a guard region (so the ramp itself cannot depress the
        reference); a sample counts as off-source when it lies more than
        ``sigma`` MADs below that level.

        An antenna's slew in a scan is determined by **median consensus** over
        its baselines: each baseline yields a leading-low duration via
        :func:`_leading_low_seconds`; then :func:`_scan_consensus` requires at
        least half (and at least 2) of the baselines to report a positive
        stretch, returning the median of those values. This preserves the
        "off-source antenna drags down every baseline" invariant while tolerating
        one clean or dead baseline (which would veto detection under the old
        ``min`` rule).

        Antennas are then resolved one at a time, largest first: once an
        antenna's slew is known its samples are masked out, so the antennas
        still to be measured are no longer judged through baselines that were
        contaminated by it. Without that, a slow antenna makes every other
        antenna look slow too.

        The per-antenna result is the **median** over its scans (robust to scans
        where the antenna did not slew, unlike mean which dilutes with zeros)
        and applied uniformly to every scan of every source.

        Parameters
        ----------
        field : str
            Source to measure on — normally the phase calibrator, which is
            observed often enough to average over.
        sigma : float
            MADs below the stable level that count as off-source.
        max_seconds : float
            Refuse to report more than this (a longer ramp is a different fault).

        Returns
        -------
        dict
            ``per_antenna`` (seconds), ``per_scan`` (raw per-antenna lists),
            ``sigma``, ``integration_time``.
        """
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

        # amplitude[(scan, a, b)] -> (offsets, amplitudes) for one baseline in one scan.
        # Read a few scans of one subband at a time: a whole subband over every field
        # does not fit in memory on a long experiment (EM163: ~9 GB per subband).
        n_ant = max(len(antenna_names), 1)
        pieces: dict[tuple, list] = {}
        scan_numbers = [s.scan_number for s in scans]
        scan_chunks = [scan_numbers[i:i + _QUACK_SCANS_PER_READ]
                       for i in range(0, len(scan_numbers), _QUACK_SCANS_PER_READ)]
        ms_tool = self.backend.tools.ms()
        if not ms_tool.open(str(self.backend.ms_path(project_code))):
            raise BackendError(f"could not open MS for {project_code}")
        try:
            for spw in range(meta.freq_setup.n_subbands):
                for scan_chunk in scan_chunks:
                    ms_tool.selectinit(datadescid=spw)
                    try:
                        select_ms_rows(ms_tool, field=",".join(wanted), scans=scan_chunk)
                    except BackendError:
                        ms_tool.reset()
                        continue     # nothing recorded in this subband for these scans
                    try:
                        record = ms_tool.getdata([column_name, "flag", "time",
                                                  "antenna1", "antenna2"])
                    except RuntimeError as exc:
                        raise BackendError(f"{project_code}: could not read {column}: {exc}") from exc
                    ms_tool.reset()
                    values = record.get(column_name)
                    if values is None or not values.size:
                        continue
                    flags = np.asarray(record["flag"], dtype=bool)[pol_indices, :, :]
                    amplitude = np.abs(values[pol_indices, :, :])
                    del values, record["flag"]
                    amplitude[flags] = np.nan
                    with np.errstate(invalid="ignore"):
                        amplitude = np.nanmean(amplitude, axis=(0, 1))
                    times = np.asarray(record["time"], dtype=float)
                    ant1, ant2 = np.asarray(record["antenna1"]), np.asarray(record["antenna2"])
                    scan_index = np.searchsorted(starts, times, side="right") - 1
                    inside = (scan_index >= 0) & (scan_index < len(scans))
                    inside[inside] &= times[inside] <= ends[scan_index[inside]]
                    rows = np.flatnonzero(inside & np.isfinite(amplitude) & (ant1 != ant2))
                    if not rows.size:
                        continue
                    codes = (scan_index[rows].astype(np.int64) * n_ant + ant1[rows]) * n_ant + ant2[rows]
                    order = np.argsort(codes, kind="stable")
                    rows, codes = rows[order], codes[order]
                    unique, first = np.unique(codes, return_index=True)
                    for code, group in zip(unique, np.split(rows, first[1:])):
                        scan_id, pair = divmod(int(code), n_ant * n_ant)
                        key = (scan_id, *divmod(pair, n_ant))
                        pieces.setdefault(key, []).append(
                            (times[group] - starts[scan_id], amplitude[group].astype(np.float32)))
        finally:
            ms_tool.close()
        if not pieces:
            raise BackendError(f"{project_code}: no usable data on {field!r} to measure the slew")
        series = {key: (np.concatenate([p[0] for p in parts]),
                        np.concatenate([p[1] for p in parts]).astype(float))
                  for key, parts in pieces.items()}
        del pieces
        # (scan, antenna) -> baselines of that antenna in that scan
        by_scan_antenna: dict[tuple, list] = {}
        for key, entries in series.items():
            for antenna_id in key[1:]:
                by_scan_antenna.setdefault((key[0], antenna_id), []).append(entries)

        # Per baseline and scan: delegate to _leading_low_seconds with masked-resolution masking.
        def baseline_leading_low(entries: tuple, masked: dict, scan: int) -> float:
            """Measure leading-low seconds for one baseline in one scan, masking resolved antennas."""
            offsets, amps = entries
            order = np.argsort(offsets, kind="stable")
            offsets, amps = offsets[order], amps[order]
            keep = np.ones(offsets.shape, dtype=bool)
            for antenna_id, secs in masked.items():   # already-resolved antennas
                if antenna_id in scan_antennas.get(scan, ()) and secs > 0:
                    keep &= ~(offsets < secs)
            offsets, amps = offsets[keep], amps[keep]
            return _leading_low_seconds(offsets, amps, sigma, integration, max_seconds)

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
                    baselines = by_scan_antenna.get((scan, antenna_id), [])
                    if len(baselines) < 2:
                        continue     # one baseline cannot separate the two ends
                    # Median consensus: at least half (and >=2) baselines must agree.
                    lows = [baseline_leading_low(entries, masked, scan) for entries in baselines]
                    estimates.setdefault(antenna_id, []).append(_scan_consensus(lows))
            # Per-antenna: median over scans (robust to scans where the antenna did not slew).
            aggregated = {a: float(np.median(v)) for a, v in estimates.items() if v}
            if not aggregated:
                break
            worst = max(aggregated, key=lambda a: aggregated[a])
            seconds = min(aggregated[worst], max_seconds)
            name = antenna_names[worst] if worst < len(antenna_names) else str(worst)
            per_scan[name] = estimates.get(worst, [])
            candidates.discard(worst)
            if seconds <= 0:
                continue     # nothing to flag for this one; the rest will be smaller still
            masked[worst] = seconds
            per_antenna[name] = seconds
            logger.info("quack: {} slews for {:.0f}s (median over {} scan(s))",
                        name, seconds, len(estimates.get(worst, [])))

        logger.info("quack: measured on {} over {} scan(s); {} antenna(s) need trimming ({})",
                    field or "all fields", len(scans), len(per_antenna),
                    ", ".join(f"{k} {v:.0f}s" for k, v in sorted(per_antenna.items())) or "none")
        return {"per_antenna": per_antenna, "per_scan": per_scan, "sigma": sigma,
                "integration_time": integration}

    def quack(self, project_code: str, *, per_antenna: Optional[dict] = None,
              interval: float = 0.0, field: str = "", **kwargs) -> float:
        """Flag the start of every scan, per antenna.

        Resolution order: explicit ``per_antenna`` seconds win; otherwise
        ``interval`` (> 0) applies one value to the whole array; otherwise the
        ramp is measured from the data with :meth:`measure_quack`. The
        interval scope is the whole dataset — quacking only the calibrators would
        leave the same slewing data in the target.
        """
        per_antenna = dict(per_antenna or {})
        if not per_antenna and interval and float(interval) > 0:
            per_antenna = {"": float(interval)}
        if not per_antenna:
            per_antenna = self.measure_quack(project_code, field=field, **kwargs)["per_antenna"]
        if not per_antenna:
            logger.info("quack: no antenna shows a settling ramp; nothing to flag")
            return 0.0
        before = self.flagged_fraction(project_code)
        ms = str(self.backend.ms_path(project_code))
        for antenna, seconds in sorted(per_antenna.items()):
            logger.info("quack: flagging the first {:.0f} s of each scan on {}", seconds,
                        antenna or "all antennas")
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

    def measure_edge_channels(self, project_code: str, table: CalTable, *, threshold: float = 6.0,
                              max_edge_fraction: float = 0.25, **kwargs) -> dict:
        """Measure how many channels roll off at each subband edge, from the bandpass table.

        Builds three per-channel profiles across every antenna, subband and
        polarization — median amplitude, phase scatter, and flagged fraction —
        and finds the flat interior of each. Subband edges show up as amplitude
        roll-off, rising phase scatter, or solutions the solver had to flag; the
        widest trim the three agree on is what needs flagging.

        The same trim is applied to every subband of an antenna: they share a
        signal path shape, and a per-subband trim would leave the band with
        ragged, non-uniform channel coverage. Each *antenna* is measured
        separately though, and reported in ``per_antenna``: stations differ in
        where their band stops being usable, and trimming the whole array to the
        worst one throws away bandwidth the others recorded perfectly well —
        bandwidth the target image needs.

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
            ``n_edge`` (widest array-wide trim), ``first``/``last`` (the flat
            range), ``n_channels``, ``per_antenna`` (``{name: (left, right)}``)
            and the three profiles.
        """
        from ..statistics import find_flat_range, mad_sigma

        handle = self.backend.tools.table()
        if not handle.open(str(table.path)):
            raise BackendError(f"could not open bandpass table {table.path}")
        try:
            values = np.asarray(handle.getcol("CPARAM"))          # (npol, nchan, nrow)
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
            antenna_ids = np.asarray(handle.getcol("ANTENNA1"))
        finally:
            handle.close()
        if values.ndim != 3:
            raise BackendError(f"{project_code}: bandpass table has an unexpected shape "
                               f"{values.shape}; cannot measure the band edges")
        n_channels = values.shape[1]
        # Too few channels to have a measurable roll-off — a single-channel subband has no
        # edge to trim, and a handful cannot support a flat-interior fit. Keep every channel.
        if n_channels < 8:
            logger.info("edge channels: {} channel(s) per subband is too few to measure a "
                        "roll-off; keeping the whole band", n_channels)
            return {"n_edge": 0, "first": 0, "last": max(0, n_channels - 1),
                    "n_channels": int(n_channels), "per_antenna": {}, "method": "narrowband",
                    "amplitude_profile": [], "phase_profile": [], "flagged_fraction": []}

        amplitude = np.abs(values).astype(float)
        phase = np.angle(values)
        amplitude[flags] = np.nan
        phase[flags] = np.nan
        names_handle = self.backend.tools.table()
        antenna_names: list[str] = []
        if names_handle.open(str(Path(table.path) / "ANTENNA")):
            try:
                antenna_names = [str(n) for n in names_handle.getcol("NAME")]
            finally:
                names_handle.close()

        # Collapse polarization and row (= antenna x subband) onto the channel axis.
        per_channel = amplitude.transpose(1, 0, 2).reshape(n_channels, -1)
        phase_channel = phase.transpose(1, 0, 2).reshape(n_channels, -1)
        with np.errstate(invalid="ignore"):
            amp_profile = np.nanmedian(per_channel, axis=1)
            phase_profile = np.array([mad_sigma(row[np.isfinite(row)]) for row in phase_channel])
        flagged_fraction = flags.transpose(1, 0, 2).reshape(n_channels, -1).mean(axis=1)

        max_trim = int(n_channels * max_edge_fraction)
        overall = self._edge_trim(amp_profile, phase_profile, flagged_fraction,
                                  threshold=threshold, max_trim=max_trim, n_channels=n_channels)
        logger.info("edge channels: array-wide trim {} left / {} right of {} channels",
                    overall[0], overall[1], n_channels)

        # Per antenna: stations differ in where their band stops being usable, and trimming
        # every one to the worst station's roll-off throws away bandwidth the others have.
        per_antenna: dict[str, tuple[int, int]] = {}
        antenna_profiles: dict[str, np.ndarray] = {}
        for antenna_id in np.unique(antenna_ids):
            rows = antenna_ids == antenna_id
            if not rows.any():
                continue
            name = (antenna_names[int(antenna_id)] if int(antenna_id) < len(antenna_names)
                    else f"#{int(antenna_id)}")
            amp_rows = amplitude[:, :, rows].transpose(1, 0, 2).reshape(n_channels, -1)
            phase_rows = phase[:, :, rows].transpose(1, 0, 2).reshape(n_channels, -1)
            flag_rows = flags[:, :, rows].transpose(1, 0, 2).reshape(n_channels, -1)
            if not np.isfinite(amp_rows).any():
                continue                      # no solutions at all: nothing to measure
            with np.errstate(invalid="ignore"):
                amp_one = np.nanmedian(amp_rows, axis=1)
                phase_one = np.array([mad_sigma(r[np.isfinite(r)]) for r in phase_rows])
            trim = self._edge_trim(amp_one, phase_one, flag_rows.mean(axis=1),
                                   threshold=threshold, max_trim=max_trim, n_channels=n_channels)
            per_antenna[name] = trim
            antenna_profiles[name] = amp_one
        if per_antenna:
            logger.info("edge channels per antenna: {}",
                        ", ".join(f"{n} {l}/{r}" for n, (l, r) in sorted(per_antenna.items())))
        n_edge = max(overall)
        return {"n_edge": int(n_edge), "first": int(overall[0]),
                "last": int(n_channels - 1 - overall[1]), "n_channels": int(n_channels),
                "per_antenna": {n: (int(l), int(r)) for n, (l, r) in per_antenna.items()},
                "antennas": {n: {"amplitude_profile": p.tolist(), "n_edge": per_antenna[n]}
                             for n, p in antenna_profiles.items()},
                "amplitude_profile": amp_profile.tolist(),
                "phase_profile": phase_profile.tolist(),
                "flagged_fraction": flagged_fraction.tolist()}

    def bandpass_gaps(self, project_code: str, table: CalTable, *, min_gain: float = 0.5,
                      dry_run: bool = False, **kwargs) -> dict:
        """Flag the data of every antenna/subband/channel the bandpass cannot calibrate.

        Two kinds of channel: those whose bandpass solution is flagged, and those
        where the (per-subband normalised) bandpass amplitude is below ``min_gain``.
        ``applycal`` does not flag the first kind - it interpolates the bandpass in
        frequency and extends the nearest good solution outward - so a band-edge
        roll-off the solver gave up on comes through *partly* corrected and too low
        (EM163: channel 1 of every subband 30-40% down; 23 channels of CM's first
        subband). The second kind is corrected in amplitude but is mostly noise.

        Unlike the edge trim this is per antenna *and* per subband: a station
        whose band starts in the middle of one subband loses those channels only.

        Parameters
        ----------
        table : CalTable
            The bandpass table.
        min_gain : float
            Bandpass amplitude (subband median = 1) below which a channel is flagged.
        dry_run : bool
            Build and report the commands without flagging.

        Returns
        -------
        dict
            ``commands`` (flagdata list-mode strings), ``n_channels_flagged`` (antenna x
            subband x channel cells), ``per_antenna`` (``{name: cells}``) and, when the
            flags were applied, ``flagged_fraction_of_data``.
        """
        handle = self.backend.tools.table()
        if not handle.open(str(table.path)):
            raise BackendError(f"could not open bandpass table {table.path}")
        try:
            values = np.asarray(handle.getcol("CPARAM"))          # (npol, nchan, nrow)
            flags = np.asarray(handle.getcol("FLAG")).astype(bool)
            antenna_ids = np.asarray(handle.getcol("ANTENNA1"))
            spw_ids = np.asarray(handle.getcol("SPECTRAL_WINDOW_ID"))
        finally:
            handle.close()
        names = self.backend.calibrate._table_antenna_names(Path(table.path))
        n_channels = values.shape[1]
        commands: list[str] = []
        per_antenna: dict[str, int] = {}
        dead: dict[str, set[str]] = {}
        correlations = self._correlation_labels(project_code)
        # Feed labels in table order (R, L or X, Y), from the parallel hands of the data.
        hands = [c[0] for c in correlations if len(c) == 2 and c[0] == c[1]]
        for row in range(values.shape[2]):
            solved = ~flags[:, :, row]
            if not solved.any():
                continue                  # the antenna has no solution in this subband: no data to protect
            name = names[int(antenna_ids[row])] if int(antenna_ids[row]) < len(names) else str(int(antenna_ids[row]))
            # A polarization with no solution at all in this subband is a dead receiver
            # channel, not a gap in the band: only the correlations it enters are lost, and it
            # must not take the other polarization's channels with it.
            alive = solved.any(axis=1)
            for pol in np.flatnonzero(~alive):
                products = ",".join(c for c in correlations if hands[pol] in c) if pol < len(hands) else ""
                if products:
                    commands.append(f"antenna='{name}' spw='{int(spw_ids[row])}' correlation='{products}'")
                    dead.setdefault(name, set()).add(hands[pol])
            bad = (~solved | (np.abs(values[:, :, row]) < min_gain))[alive].any(axis=0)   # any live polarization
            if not bad.any():
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
            ranges = ";".join(f"{a}~{b - 1}" for a, b in zip(edges[::2], edges[1::2]))
            commands.append(f"antenna='{name}' spw='{int(spw_ids[row])}:{ranges}'")
            per_antenna[name] = per_antenna.get(name, 0) + int(bad.sum())
        if dead:
            logger.info("bandpass gaps: no solution in one polarization, only its correlations are flagged: {}",
                        ", ".join(f"{name} ({'/'.join(sorted(pols))})" for name, pols in sorted(dead.items())))
        report = {"commands": commands, "n_channels_flagged": int(sum(per_antenna.values())),
                  "per_antenna": per_antenna, "n_channels": int(n_channels), "min_gain": float(min_gain)}
        if not commands:
            logger.info("bandpass gaps: every channel has a usable bandpass solution; nothing to flag")
            return report
        logger.info("bandpass gaps: {} antenna/subband selection(s) without a usable bandpass (flagged solution or "
                    "gain < {:g}); channels per antenna: {}", len(commands), min_gain,
                    ", ".join(f"{n} {c}" for n, c in sorted(per_antenna.items(), key=lambda item: -item[1])))
        if dry_run:
            return report
        before = self.flagged_fraction(project_code)
        try:
            self.backend.tasks.flagdata(vis=str(self.backend.ms_path(project_code)), mode="list", inpfile=commands,
                                        flagbackup=False, action="apply")
        except RuntimeError as exc:
            raise BackendError(f"{project_code}: flagging the bandpass gaps failed: {exc}") from exc
        after = self.flagged_fraction(project_code)
        report["flagged_fraction_of_data"] = max(0.0, after - before)
        logger.info("flag[bandpass gaps]: {:.2%} newly flagged ({:.1%} -> {:.1%} of observable data)",
                    report["flagged_fraction_of_data"], before, after)
        return report

    def _correlation_labels(self, project_code: str) -> list[str]:
        """Return the correlation products of the data, in order (e.g. ``['RR', 'RL', 'LR', 'LL']``)."""
        codes = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
        handle = self.backend.tools.table()
        handle.open(str(self.backend.ms_path(project_code) / "POLARIZATION"))
        try:
            types = np.atleast_1d(np.asarray(handle.getcell("CORR_TYPE", 0)))
        finally:
            handle.close()
        return [codes.get(int(code), "") for code in types]

    def _edge_trim(self, amp_profile: np.ndarray, phase_profile: np.ndarray,
                   flagged_fraction: np.ndarray, *, threshold: float, max_trim: int,
                   n_channels: int) -> tuple[int, int]:
        """Return ``(left, right)`` channels to trim, from three independent indicators.

        Amplitude roll-off, rising phase scatter and channels the solver could not solve
        each vote on where the usable band starts and ends. The verdict is their *median*
        per side, not the widest of them: one indicator running away — phase scatter is
        noisy on a weak antenna — would otherwise throw away good bandwidth on the word of
        a single measurement, and every flagged channel costs sensitivity in the final
        image. Two indicators must agree before a channel is given up.

        Parameters
        ----------
        amp_profile, phase_profile, flagged_fraction : numpy.ndarray
            Per-channel median amplitude, phase scatter, and flagged fraction.
        threshold : float
            Deviation in robust sigmas defining the flat interior.
        max_trim : int
            Never trim more than this many channels from either edge.
        n_channels : int
            Channels per subband.

        Returns
        -------
        tuple
            ``(left, right)`` channel counts to flag.
        """
        from ..statistics import find_flat_range

        amp_range = find_flat_range(amp_profile, threshold=threshold,
                                    max_edge_fraction=max_trim / max(n_channels, 1))
        phase_range = find_flat_range(phase_profile, threshold=threshold,
                                      max_edge_fraction=max_trim / max(n_channels, 1))
        solved = np.where(np.asarray(flagged_fraction) < 0.5)[0]
        solved_range = (int(solved[0]), int(solved[-1])) if solved.size else (0, n_channels - 1)
        lefts = sorted((amp_range[0], phase_range[0], solved_range[0]))
        rights = sorted((n_channels - 1 - amp_range[1], n_channels - 1 - phase_range[1],
                         n_channels - 1 - solved_range[1]))
        return (int(min(max(lefts[1], 0), max_trim)), int(min(max(rights[1], 0), max_trim)))

    def _edge_spw_selection(self, project_code: str, *, edge_fraction: float = 0.1,
                            n_channels: int = 0, edge_channels: int = 0,
                            left: int = -1, right: int = -1, **kwargs) -> str:
        """Build the spw selection flagging the outer channels of every subband.

        ``left``/``right`` (asymmetric, measured per antenna) win over
        ``edge_channels`` (a symmetric count from the measured roll-off), which in
        turn wins over the ``edge_fraction`` default. Returns an empty string when
        there is nothing to trim — including a subband too narrow to have an edge,
        as with single-channel spectral windows.
        """
        if not n_channels:
            n_channels = int(self.backend.data.get_metadata(
                project_code, [], "").freq_setup.n_channels)
        if n_channels < 3:
            return ""      # one or two channels: no edge to trim without losing the band
        if left >= 0 or right >= 0:
            n_left, n_right = max(0, int(left)), max(0, int(right))
        else:
            symmetric = int(edge_channels) or int(round(n_channels * float(edge_fraction)))
            n_left = n_right = max(0, symmetric)
        limit = n_channels // 2 - 1
        n_left, n_right = min(n_left, limit), min(n_right, limit)
        parts = []
        if n_left > 0:
            parts.append(f"0~{n_left - 1}")
        if n_right > 0:
            parts.append(f"{n_channels - n_right}~{n_channels - 1}")
        return f"*:{';'.join(parts)}" if parts else ""

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

        # Count the surviving baselines directly. Summing per-antenna counts would count
        # every kept baseline twice and is therefore not the baseline-weighted health
        # statistic the report promises.
        dead_ids = [index for index, name in enumerate(names)
                    if per_antenna[name][1] > 0
                    and per_antenna[name][0] == per_antenna[name][1]]
        exclusion = (" AND ".join(
            f"{column} NOT IN [{','.join(str(index) for index in dead_ids)}]"
            for column in ("ANTENNA1", "ANTENNA2")) if dead_ids else "")
        surviving_where = " AND ".join(
            part for part in (f"({where})" if where else "", exclusion) if part)
        surviving = self._count_query(project_code, where=surviving_where)
        report["excluding_dead"] = entry(
            surviving[0]["flagged"] if surviving else 0,
            surviving[0]["observable"] if surviving else 0)
        report["excluding_dead"]["excluded_antennas"] = [names[index] for index in dead_ids]
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
            channel. A subband with fewer channels than ``chanbin`` is left
            unaveraged. Matches the ``casa_pipeline`` convention.
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
        if chanbin > n_channels:
            logger.info("split: {} has {} channel(s) per subband, fewer than chanbin={}; keeping them all",
                        source, n_channels, chanbin)
            chanbin = 1
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
        try:
            from ..interactive import snapshot_flags
            snapshot_flags(str(target))      # reference for flag.from_split after difmapy editing
        except Exception as exc:  # noqa: BLE001 - the snapshot is a convenience, not a product
            logger.debug("could not snapshot the flags of {}: {}", target, exc)
        return str(target)

    def merge(self, project_codes: list[str], source_names: list[str], *, inputs: Optional[list[str]] = None,
              outputvis: str = "", scales: Optional[list[float]] = None, **kwargs) -> str:
        """Concatenate the calibrated per-source measurement sets of several epochs into one.

        Parameters
        ----------
        project_codes : list of str
            Project codes of the epochs, in the order of ``inputs`` (for the log).
        source_names : list of str
            The source the inputs hold (one name; for the log).
        inputs : list of str
            The per-source split measurement sets, one per epoch.
        outputvis : str
            Path of the combined measurement set (replaced when it exists).
        scales : list of float, optional
            One amplitude factor per input: its visibilities are multiplied by it
            (and its weights divided by the square) in the combined file. Used to
            bring epochs in which a variable source had different flux densities
            to a common level, so that one model can describe all of them.

        Returns
        -------
        str
            Path of the combined measurement set.
        """
        if not inputs or not outputvis:
            raise BackendError("merge needs the per-epoch measurement sets and an output path")
        missing = [path for path in inputs if not Path(path).is_dir()]
        if missing:
            raise BackendError(f"merge: measurement set(s) not found: {', '.join(missing)}")
        target = Path(outputvis)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        logger.info("concat: {} [{}] -> {}", ",".join(source_names), ", ".join(project_codes), target.name)
        # A split of a Multi-MS is itself a Multi-MS, and concat cannot append to one: every
        # such input goes through a plain-MS copy first (these are small, averaged files).
        temporary: list[Path] = []
        try:
            plain = []
            for index, path in enumerate(inputs):
                if (Path(path) / "SUBMSS").is_dir():
                    copy = target.with_name(f"{target.name}.part{index}")
                    if copy.exists():
                        shutil.rmtree(copy)
                    # split(keepmms=False): mstransform on a Multi-MS writes a Multi-MS again.
                    self.backend.tasks.split(vis=str(path), outputvis=str(copy), datacolumn="data",
                                             keepmms=False)
                    temporary.append(copy)
                    plain.append(str(copy))
                else:
                    plain.append(str(path))
            if len(plain) == 1:
                shutil.copytree(plain[0], target)
            else:
                self.backend.tasks.concat(vis=plain, concatvis=str(target), respectname=True, copypointing=False)
        except RuntimeError as exc:
            raise BackendError(f"concat of {', '.join(source_names)} failed: {exc}") from exc
        finally:
            for copy in temporary:
                shutil.rmtree(copy, ignore_errors=True)
        if scales and any(abs(float(scale) - 1.0) > 1e-6 for scale in scales):
            self._scale_epochs(target, [str(path) for path in inputs], [float(scale) for scale in scales])
        return str(target)

    def _scale_epochs(self, combined: Path, inputs: list[str], scales: list[float]) -> None:
        """Multiply the visibilities of each epoch inside ``combined`` by its factor (rows found by time)."""
        handle = self.backend.tools.table()
        windows = []
        for path in inputs:
            handle.open(path)
            try:
                times = np.asarray(handle.getcol("TIME"))
            finally:
                handle.close()
            windows.append((float(times.min()) - 1.0, float(times.max()) + 1.0))
        handle.open(str(combined), nomodify=False)
        try:
            times = np.asarray(handle.getcol("TIME"))
            columns = [c for c in ("DATA", "CORRECTED_DATA") if c in handle.colnames()]
            for (start, end), scale in zip(windows, scales):
                rows = np.flatnonzero((times >= start) & (times <= end))
                if scale == 1.0 or rows.size == 0:
                    continue
                query = handle.selectrows(rows.tolist())
                try:
                    for column in columns:
                        query.putcol(column, np.asarray(query.getcol(column)) * scale)
                    for column, power in (("WEIGHT", -2.0), ("SIGMA", 1.0)):
                        query.putcol(column, np.asarray(query.getcol(column)) * scale ** power)
                    if "WEIGHT_SPECTRUM" in handle.colnames():
                        query.putcol("WEIGHT_SPECTRUM", np.asarray(query.getcol("WEIGHT_SPECTRUM")) * scale ** -2.0)
                finally:
                    query.close()
            handle.flush()
        finally:
            handle.close()
        logger.info("concat: epochs scaled to a common flux level ({})", ", ".join(f"{s:.3f}" for s in scales))

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


def robust_tag(robust: float) -> str:
    """Return the filesystem-safe image-name tag for a robust value (``robust-2``, ``robust0``)."""
    return f"robust{float(robust):g}"


def _angle_to_mas(quantity: dict) -> float:
    """Convert a CASA angle quantity (``{"value", "unit"}``) to milliarcseconds."""
    scale = {"mas": 1.0, "arcsec": 1e3, "arcmin": 6e4, "deg": 3.6e6, "rad": math.degrees(1.0) * 3.6e6}
    return float(quantity.get("value", 0.0)) * scale.get(str(quantity.get("unit", "arcsec")), 1e3)


def _angle_to_rad(quantity: dict) -> float:
    """Convert a CASA angle quantity to radians."""
    scale = {"rad": 1.0, "deg": math.pi / 180, "arcmin": math.pi / 10800,
             "arcsec": math.pi / 648000, "mas": math.pi / 6.48e8}
    return float(quantity.get("value", 0.0)) * scale.get(str(quantity.get("unit", "arcsec")), math.pi / 648000)


class CasaImagingOps(ImagingOps):
    """Deconvolution with ``tclean`` and image statistics from the CASA image tools."""

    #: Default image size in pixels when neither the caller nor the config says otherwise.
    DEFAULT_IMSIZE = 1024
    #: Pixels per synthesised beam (resolution / cell).
    PIXELS_PER_BEAM = 5.0

    def image_dir(self) -> Path:
        """Return (and create) the image directory (``<work_dir>/images``)."""
        path = self.work_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def imagename(self, project_code: str, source: str, robust: float) -> Path:
        """Return the tclean image-name prefix ``<work_dir>/images/<code>_<source>.robust<r>``."""
        return self.image_dir() / f"{project_code}_{source}.{robust_tag(robust)}"

    def _vis_for(self, project_code: str, source: str) -> tuple[str, str, str]:
        """Return ``(vis, field, datacolumn)``: the split calibrated MS when it exists, else the main MS."""
        split = self.work_dir / "calibrated_data" / f"{project_code}_{source}.ms"
        if split.is_dir():
            return str(split), "", "data"
        return str(self.backend.ms_path(project_code)), source, "corrected"

    def _cell_mas(self, metadata: Optional[ObsMetadata], imsize: int) -> float:
        """Pixel size: a fifth of the resolution, widened so the field spans twice the largest scale."""
        resolution = float(metadata.resolution_mas) if metadata is not None else 0.0
        if resolution <= 0:
            return 1.0
        cell = resolution / self.PIXELS_PER_BEAM
        largest = float(metadata.largest_angular_scale_mas)
        if largest > 0:
            cell = max(cell, min(2.0 * largest / imsize, resolution / 3.0))
        return cell

    def clean(self, project_code: str, source: str, *, robust: float = 0.0, weighting: str = "briggs",
              niter: int = 4000, threshold_sigma: float = 3.0, imsize: Optional[list[int]] = None,
              cell: Optional[str] = None, produce_fits: bool = True, produce_png: bool = True,
              metadata: Optional[ObsMetadata] = None, savemodel: str = "none", **kwargs) -> Image:
        """Image one source with tclean (mfs, hogbom, Stokes I) and return its :class:`Image`.

        Uses the split calibrated MS ``<work_dir>/calibrated_data/<code>_<source>.ms``
        when present (its DATA column is already calibrated), otherwise the main MS
        with ``field=source`` on CORRECTED_DATA. Cleaning stops at ``niter`` or when the
        residual drops below ``threshold_sigma`` times the dirty-map rms.

        Parameters
        ----------
        robust : float
            Briggs robust parameter.
        weighting : str
            tclean weighting scheme.
        niter, threshold_sigma : int, float
            Clean iterations cap and stopping threshold in residual-rms units.
        imsize : list of int, optional
            Image size in pixels (default 1024 x 1024).
        cell : str, optional
            Pixel size as a CASA quantity (default resolution / 5 from the metadata).
        produce_fits : bool
            Export ``<imagename>.fits``.
        produce_png : bool
            Recorded on the Image only; the PNG grid is rendered by the caller per source.
        savemodel : str
            tclean ``savemodel`` (``"none"`` | ``"modelcolumn"`` | ``"virtual"``).

        Returns
        -------
        Image
            ``paths`` has ``image``, ``residual``, ``psf``, ``model`` and (``fits``).
        """
        kwargs.pop("imager", None)
        meta = metadata or self.backend.data.get_metadata(project_code, [], "")
        vis, field, datacolumn = self._vis_for(project_code, source)
        size = [int(v) for v in imsize] if imsize else [self.DEFAULT_IMSIZE, self.DEFAULT_IMSIZE]
        if len(size) == 1:
            size = size * 2
        cell_value = cell or f"{self._cell_mas(meta, min(size)):.4g}mas"
        name = self.imagename(project_code, source, robust)
        for stale in name.parent.glob(f"{name.name}.*"):
            shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
        params = {"vis": vis, "imagename": str(name), "field": field, "datacolumn": datacolumn,
                  "specmode": "mfs", "deconvolver": "hogbom", "gridder": "standard", "stokes": "I",
                  "weighting": weighting, "robust": float(robust), "imsize": size,
                  "cell": [cell_value], "interactive": False, "parallel": False,
                  "savemodel": savemodel}
        params.update(kwargs)
        # ``nsigma`` makes tclean's minor cycle fail on data where its own noise estimate
        # is degenerate, so measure the dirty-map rms and pass an absolute threshold.
        try:
            self.backend.tasks.tclean(niter=0, **params)
            rms = float(np.max(self.backend.tasks.imstat(imagename=f"{name}.residual").get("rms", [0.0])))
        except Exception:  # noqa: BLE001 - without a dirty map just clean to niter
            rms = 0.0
        threshold = f"{threshold_sigma * rms}Jy" if rms > 0 else ""
        logger.info("tclean: {} field={!r} robust={:g} imsize={} cell={} niter={} threshold={}",
                    Path(vis).name, field or source, float(robust), size, cell_value, niter,
                    threshold or "niter")
        try:
            self.backend.tasks.tclean(niter=int(niter), threshold=threshold, **params)
        except Exception as exc:  # noqa: BLE001 - casatasks raise plain Exceptions
            raise BackendError(f"{project_code}: tclean of {source!r} (robust {robust:g}) failed: {exc}") from exc
        paths = {kind: f"{name}.{kind}" for kind in ("image", "residual", "psf", "model")}
        if not Path(paths["image"]).is_dir():
            raise BackendError(f"{project_code}: tclean produced no image for {source!r}")
        if produce_fits:
            paths["fits"] = f"{name}.fits"
            # tclean leaves an all-false pixel mask (mask0), which exportfits honours by
            # blanking every pixel to NaN. Reset the default mask to all-good first.
            try:
                ia = self.backend.tools.image()
                if ia.open(paths["image"]):
                    ia.calcmask(mask="T", asdefault=True)
                    ia.done()
            except Exception as exc:  # noqa: BLE001 - fall through and let exportfits try anyway
                logger.debug("could not clear the image mask on {}: {}", paths["image"], exc)
            try:
                self.backend.tasks.exportfits(imagename=paths["image"], fitsimage=paths["fits"],
                                              overwrite=True, dropdeg=True)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(f"{project_code}: exportfits of {source!r} failed: {exc}") from exc
        stats = self.statistics(paths["image"], residual=paths["residual"])
        logger.info("image {}: peak {:.4g} Jy/beam, rms {:.3g} Jy/beam, DR {:.0f}, flux {:.4g} Jy",
                    name.name, stats["peak"], stats["rms"], stats["dynamic_range"], stats["integrated_flux"])
        metrics = QualityMetrics(peak=stats["peak"], rms=stats["rms"], dynamic_range=stats["dynamic_range"],
                                 integrated_flux=stats["integrated_flux"], beam=stats["beam"])
        return Image(source=source, robust=float(robust), weighting=weighting, paths=paths, stats=metrics)

    def statistics(self, image_path: str, *, residual: str = "") -> dict:
        """Return ``peak``, ``rms``, ``dynamic_range``, ``integrated_flux`` (Jy) and ``beam`` (mas, mas, deg).

        The rms comes from the residual image when given (the restored image's
        rms is biased by the source itself); the integrated flux is the ``flux``
        of the restored image above three times that rms.
        """
        # imstat is unreliable on tclean images (empty returns when a mask is present),
        # so the statistics are computed from the pixels directly.
        ia = self.backend.tools.image()
        if not ia.open(image_path):
            raise BackendError(f"could not open image {image_path}")
        try:
            pixels = np.asarray(ia.getchunk(), dtype=float)
            coords = ia.coordsys()
            increment = [abs(v) for v in coords.increment().get("numeric", [])[:2]]
            restoring = ia.restoringbeam() or {}
        finally:
            ia.close()
        pixels = np.squeeze(pixels)
        finite = pixels[np.isfinite(pixels)]
        peak = float(np.max(finite)) if finite.size else 0.0
        rms = peak
        if residual and Path(residual).is_dir():
            ia.open(residual)
            try:
                res_pixels = np.squeeze(np.asarray(ia.getchunk(), dtype=float))
            finally:
                ia.close()
            res_pixels = res_pixels[np.isfinite(res_pixels)]
            if res_pixels.size:
                median = np.median(res_pixels)
                rms = float(1.4826 * np.median(np.abs(res_pixels - median))) or float(np.std(res_pixels))
        beam = (0.0, 0.0, 0.0)
        if "major" in restoring:
            beam = (_angle_to_mas(restoring["major"]), _angle_to_mas(restoring["minor"]),
                    float(restoring.get("positionangle", {}).get("value", 0.0)))
        flux = peak
        if rms > 0 and beam[0] > 0 and len(increment) >= 2 and all(increment):
            # Sum the >3-sigma emission and convert pixel sums to Jy via the beam area.
            bmaj = _angle_to_rad(restoring["major"]); bmin = _angle_to_rad(restoring["minor"])
            pixels_per_beam = math.pi / (4 * math.log(2)) * bmaj * bmin / (increment[0] * increment[1])
            if pixels_per_beam > 0:
                flux = float(np.sum(finite[finite > 3.0 * rms])) / pixels_per_beam
        return {"peak": peak, "rms": rms, "dynamic_range": peak / rms if rms > 0 else 0.0,
                "integrated_flux": flux, "beam": beam}


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

    def autocorr(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                 column: str = "data", label: str = "", metadata: Optional[ObsMetadata] = None,
                 **kwargs) -> str:
        """Plot the autocorrelation amplitude spectra (one panel per antenna) for a selection."""
        from ..plotting import plot_autocorr_spectrum
        spectrum = self.backend.data.read_autocorr_spectrum(project_code, field=field, scans=scans,
                                                            column=column, metadata=metadata)
        return str(plot_autocorr_spectrum(spectrum, self.plot_dir(self.category_for_column(column)),
                                          project_code, label=label))

    def radplot(self, project_code: str, *, field: str = "", column: str = "corrected",
                time_bin: float = 10.0, label: str = "", with_model: bool = False,
                metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Plot amplitude and phase vs uv distance for one source (model overlaid when asked)."""
        from ..plotting import plot_radplot
        uvdata = self.backend.data.read_uvdistance(project_code, field=field, column=column,
                                                   time_bin=time_bin, with_model=with_model,
                                                   metadata=metadata)
        return str(plot_radplot(uvdata, self.plot_dir(self.category_for_column(column)),
                                project_code, label=label))

    def lightcurve(self, project_code: str, *, fields: Optional[list[str]] = None,
                   column: str = "corrected", label: str = "", averaging_sec=None,
                   metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Plot the total coherent visibility amplitude vs time, one row per source."""
        from ..plotting import plot_total_lightcurve
        data = self.backend.data.read_total_visibility(project_code, fields=fields, column=column,
                                                       metadata=metadata)
        params = {"averaging_sec": tuple(averaging_sec)} if averaging_sec else {}
        return str(plot_total_lightcurve(data, self.plot_dir(self.category_for_column(column)),
                                         project_code, label=label, **params))

    def subband_phases(self, project_code: str, *, fields: list[str], refant: str, column: str = "corrected",
                       label: str = "", metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Plot the per-scan phase offsets between subbands on baselines to ``refant``."""
        from ..plotting import plot_subband_phase_jumps
        data = self.backend.data.read_subband_phases(project_code, fields=fields, refant=refant, column=column,
                                                     metadata=metadata)
        return str(plot_subband_phase_jumps(data, self.plot_dir(self.category_for_column(column)),
                                            project_code, label=label))

    def diagnostic(self, project_code: str, kind: str, *, field: str = "", label: str = "",
                   metadata: Optional[ObsMetadata] = None, **kwargs) -> str:
        """Produce one diagnostic plot; only ``uv_coverage`` (u vs v per source) is implemented."""
        if kind != "uv_coverage":
            raise self._unsupported(f"diagnostic[{kind}]", "only uv_coverage is implemented")
        from ..plotting import plot_uv_coverage
        data = self.backend.data.read_uv_coverage(project_code, field=field, metadata=metadata)
        suffix = f".{label}" if label else ""
        outfile = self.plot_dir("raw") / f"{project_code}{suffix}.uv_coverage.png"
        logger.info("uv coverage plot: {} field(s) -> {}", len(data.get("fields") or {}), outfile)
        return plot_uv_coverage(data, str(outfile), title=f"{project_code} — uv coverage")

    def timeseries(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                   refant: str = "", column: str = "corrected", label: str = "",
                   metadata: Optional[ObsMetadata] = None, **kwargs) -> list[str]:
        """Plot amplitude and phase vs time on baselines to the reference antenna."""
        from ..plotting import plot_baseline_timeseries
        series = self.backend.data.read_timeseries(project_code, field=field, scans=scans,
                                                   refant=refant, column=column,
                                                   metadata=metadata, **kwargs)
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

    def image_grid(self, project_code: str, images: dict, **kwargs) -> list[str]:
        """Plot the FITS images of each source side by side into ``<work_dir>/images``."""
        from ..plotting import plot_image_grid
        return [str(p) for p in plot_image_grid(images, self.backend.image.image_dir(), project_code)]

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
    imaging_ops = CasaImagingOps
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
