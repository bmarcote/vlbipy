"""Callable operation namespaces for vlbipy.

Each namespace is bound to a single :class:`~vlbipy.observation.Observation` and
implements the *callable-namespace* pattern: calling the namespace runs its
sensible default, while its methods run explicit variants. For example::

    obs.clean(...)            # default imager ([imaging].imager, difmapy)
    obs.clean.wsclean(...)    # explicit imager
    obs.calibrate()           # full default chain
    obs.calibrate.bandpass()  # one step

Namespaces translate high-level intent into calls on the observation's
:class:`~vlbipy.backends.base.Backend`; they never import a backend library and
never touch the filesystem directly.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional, Union

from .. import phaseref
from ..diagnostics import write_summary
from ..errors import BackendError, StepError
from ..logging_utils import get_logger, warnings
from ..models import CalTable, QualityMetrics
from ..plot_jobs import PlotJob, draw, plot_workers, run_plot_jobs
from ..results import Image, ImageSet, SelfcalResult
from ..sources import Source
from ..tools import natsort_key

logger = get_logger()

#: Flagged-fraction above which a flag step is flagged as an anomaly.
_HIGH_FLAG_FRACTION = 0.3

TargetLike = Union[str, Source, Image, None]


#: The instrumental calibration steps in solving order (:meth:`CalibrateNamespace.instrumental`); each is
#: solved through the tables of the steps before it.
INSTRUMENTAL_STEPS = ("initial_calibration", "fringefit", "bandpass", "initial_calibration_sbd2", "fringefit_mbd2")

class Namespace:
    """Base class for operation namespaces bound to one observation.

    Provides discoverability helpers so users can see, interactively, which
    operations a namespace offers (tab-completion lists the public methods, and
    :meth:`operations` / ``repr`` enumerate them explicitly).
    """

    def __init__(self, obs) -> None:
        self._obs = obs

    @property
    def _backend(self):
        return self._obs._backend

    @property
    def _code(self) -> str:
        return self._obs.project_code

    def operations(self) -> list[str]:
        """Return the names of the callable operations this namespace offers."""
        return [name for name in dir(self)
                if not name.startswith("_") and name != "operations"
                and callable(getattr(type(self), name, None))]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} operations: {', '.join(self.operations())}>"

    def _resolve_source_name(self, target: TargetLike) -> str:
        """Resolve a target given as name / Source / Image / None to a source name."""
        if target is None:
            return self._obs.sources.target.name
        if isinstance(target, Image):
            return target.source
        if isinstance(target, Source):
            return target.name
        return str(target)


class ImportDataNamespace(Namespace):
    """Ingest raw data: locate/download, prepare, import to the backend, load metadata."""

    def __call__(self, *, force: bool = False, files=None, **kwargs):
        """Run the default import: find-or-download the raw files, prepare and import them.

        Parameters
        ----------
        force : bool
            Re-run even if already imported (overwrites an existing MS).
        files : str or list, optional
            Explicit FITS-IDI file(s) or a glob pattern; skips find/download.
        """
        obs = self._obs
        if not obs._state.should_run("import_data", force=force):
            # Skipping the import must not skip *knowing* about the data: a resumed run
            # in a fresh process has no metadata yet, and every later step needs it.
            if obs._metadata is None:
                obs._metadata = obs._load_metadata_cache() or self._load_metadata()
            obs.restrict_sources_to_data()
            return obs.metadata
        imp_cfg = obs.config.get("import", {})
        scan_gap = imp_cfg.get("scan_gap", 15)
        inputs: list[str] = []
        if self._backend.requires_data_files and self._backend.data.is_imported(self._code) and not force:
            # Fresh process, product already on disk: skip file search/download entirely.
            logger.info("import_data: {} already imported; reading metadata", self._code)
            self._reset_existing_data(imp_cfg)
        elif self._backend.requires_data_files:
            file_list = self._locate_files(files, imp_cfg)
            file_list = obs._observatory_handler.prepare_for_import(
                file_list, obs.work_dir, project_code=self._code,
                replace_tsys=imp_cfg.get("replace_tsys", False))
            self._backend.data.import_data(self._code, obs.sources.names, scan_gap=scan_gap,
                                           files=file_list, delete=force,
                                           zarr_store=imp_cfg.get("zarr_store", False),
                                           mms=imp_cfg.get("mms", True),
                                           needs_eop=obs._observatory_handler.needs_eop, **kwargs)
            inputs = list(file_list)
        else:
            handler = obs._observatory_handler
            logger.info("import_data: {} {}", handler.name,
                        "auto-download available" if handler.auto_download else "is manual-download only")
            self._backend.data.import_data(self._code, obs.sources.names, scan_gap=scan_gap, **kwargs)
        obs._metadata = self._load_metadata()
        obs.restrict_sources_to_data()
        obs._state.mark_complete("import_data", outputs=[f"{self._code}.ms"], inputs=inputs)
        return obs._metadata

    def _reset_existing_data(self, imp_cfg: dict) -> None:
        """Clear a previous attempt's flags and corrected data from an existing dataset.

        Flags only ever accumulate, so re-running over a dataset an earlier
        attempt already touched would start from a smaller array than that
        attempt did — and the result would depend on how often the pipeline had
        been run before. Disable with ``[import].reset_existing = false`` when a
        dataset carries hand-made flags that must survive (they are saved as a
        restorable flag version either way).
        """
        if not self._obs.scratch:
            # Resuming: the corrected data and flags are this pipeline's own work in
            # progress, not a previous attempt's leftovers. Clearing them here would
            # silently undo every calibration step already completed.
            logger.info("import_data: resuming, so the existing flags and corrected data "
                        "are kept (use --scratch to start over)")
            return
        if not imp_cfg.get("reset_existing", True):
            logger.info("import_data: keeping the existing flags and corrected data "
                        "([import].reset_existing is false)")
            return
        if not self._backend.supports("data", "reset_calibration"):
            return
        try:
            self._backend.data.reset_calibration(
                self._code, unflag=imp_cfg.get("unflag_existing", True),
                backup_flags=imp_cfg.get("backup_flags", True))
        except Exception as exc:  # noqa: BLE001 - a failed reset must not block the run
            warnings.anomaly(f"{self._code}: could not reset the existing dataset ({exc}); "
                             "it still carries the previous attempt's flags")

    def _load_metadata(self):
        """Read the full observation metadata and enrich it before anything else runs.

        Beyond the backend's own metadata read this fills in source coordinates
        (for sources declared in the config) and per-antenna subband
        participation, so later steps never have to re-query the data. Backends
        that cannot report subband participation are skipped silently.
        """
        obs = self._obs
        metadata = self._backend.data.get_metadata(self._code, obs.sources.names, obs.observatory)
        self._update_source_coordinates(metadata)
        self._update_subband_participation(metadata)
        self._write_summary(metadata)
        self._write_metadata(metadata)
        return metadata

    def _write_metadata(self, metadata) -> None:
        """Persist the metadata to ``<work_dir>/.metadata.json`` so later tools (e.g. the
        dashboard) can rebuild it without re-reading the measurement set."""
        if not metadata or not self._backend.requires_data_files:
            return
        try:
            path = Path(self._obs.work_dir) / ".metadata.json"
            path.write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - a reporting cache; never abort the import
            warnings.warn(f"{self._code}: could not persist the observation metadata ({exc})")

    def _write_summary(self, metadata) -> None:
        """Write ``<work_dir>/summary.md`` describing what was imported.

        Backends without on-disk products have no work directory to write into,
        and a failure here is a reporting problem, not an import problem, so it
        must never abort a run that otherwise succeeded.
        """
        if not metadata or not self._backend.requires_data_files:
            return
        obs = self._obs
        try:
            write_summary(metadata, Path(obs.work_dir) / "summary.md", obs.sources)
        except Exception as exc:  # noqa: BLE001 - reporting only; the import itself is fine
            warnings.warn(f"{self._code}: could not write the observation summary ({exc})")

    def _update_subband_participation(self, metadata) -> None:
        """Fill Antenna.subbands from the data (heterogeneous arrays record different subsets)."""
        if not metadata or not self._backend.supports("data", "get_subband_participation"):
            return
        try:
            participation = self._backend.data.get_subband_participation(
                self._code, list(metadata.antennas))
        except Exception as exc:  # noqa: BLE001 - inspection detail; never block the import
            warnings.warn(f"{self._code}: could not determine subband participation ({exc})")
            return
        for name, subbands in participation.items():
            if name in metadata.antennas:
                metadata.antennas[name].subbands = subbands
        partial = [f"{n} ({len(s)}/{metadata.freq_setup.n_subbands})"
                   for n, s in participation.items()
                   if s and len(s) < metadata.freq_setup.n_subbands]
        if partial:
            logger.info("heterogeneous array: {} recorded only some subbands", ", ".join(partial))
        self._report_usable_band(metadata)

    def _report_usable_band(self, metadata) -> None:
        """Log the usable band and warn about the subbands and antennas that carry no baseline.

        A subband fewer than two antennas recorded has no cross-correlation, so it cannot be
        fringe-fitted and is excluded from every solve (see
        :attr:`~vlbipy.models.ObsMetadata.usable_subbands`). An antenna whose subbands are all
        of that kind has no baseline at all and cannot be calibrated.
        """
        unusable = metadata.unusable_subbands
        if not unusable:
            return
        per_subband = metadata.antennas_per_subband()
        detail = ", ".join(f"{spw} ({', '.join(per_subband[spw]) or 'no antenna'})" for spw in unusable)
        warnings.anomaly(f"{self._code}: subband(s) {detail} have fewer than two antennas, so they carry no "
                         f"baseline; the usable band is subband(s) {list(metadata.usable_subbands)} of "
                         f"{metadata.freq_setup.n_subbands} and every solve is restricted to it")
        usable = set(metadata.usable_subbands)
        for antenna in metadata.observed_antennas:
            if antenna.subbands and not usable & set(antenna.subbands):
                warnings.anomaly(f"{self._code}: {antenna.name} recorded only subband(s) "
                                 f"{list(antenna.subbands)}, which no other antenna recorded: it has no "
                                 f"baseline and cannot be calibrated")

    def _locate_files(self, files, imp_cfg: dict) -> list[str]:
        """Resolve the raw data files: explicit argument, disk search, or archive download."""
        obs = self._obs
        Path(obs.work_dir).mkdir(parents=True, exist_ok=True)
        if files:
            if isinstance(files, (str, Path)):
                expanded = sorted(glob.glob(str(files)), key=natsort_key)
                if not expanded:
                    raise FileNotFoundError(f"no files match {files!r}")
                return expanded
            return [str(f) for f in files]
        handler = obs._observatory_handler
        found = handler.find_data_files(self._code, obs.work_dir)
        if found:
            logger.info("import_data: found {} raw data file(s) in {}", len(found), obs.work_dir)
            return found
        if not handler.auto_download:
            raise Exception(f"no raw data found in {obs.work_dir} and {handler.name} has no "
                              f"auto-download. {handler.manual_download_instructions()}")
        return handler.download_data(self._code, obs.work_dir,
                                     obsdate=imp_cfg.get("obsdate") or None,
                                     username=imp_cfg.get("username") or None,
                                     password=imp_cfg.get("password") or None)

    def _update_source_coordinates(self, metadata) -> None:
        """Fill in source coordinates from the data for sources declared in the config."""
        if not metadata or not metadata.source_coords:
            return
        from astropy import coordinates as coord
        for source in self._obs.sources:
            radec = metadata.source_coords.get(source.name)
            if radec is not None and source.coordinates is None:
                source.coordinates = coord.SkyCoord(ra=radec[0], dec=radec[1], unit="deg")

    def from_fitsidi(self, files, **kwargs):
        """Import from explicit FITS-IDI file(s) (list, single path, or glob pattern)."""
        logger.info("import_data.from_fitsidi: {}", files)
        return self.__call__(files=files, **kwargs)

    def from_uvfits(self, uvfits, *, force: bool = False, **kwargs):
        """Import from a UVFITS file."""
        obs = self._obs
        if not obs._state.should_run("import_data", force=force):
            return obs.metadata
        logger.info("import_data.from_uvfits: {}", uvfits)
        if self._backend.supports("data", "import_uvfits"):
            self._backend.data.import_uvfits(self._code, str(uvfits), delete=force)
        else:
            self._backend.data.import_data(self._code, obs.sources.names, **kwargs)
        obs._metadata = self._load_metadata()
        obs._state.mark_complete("import_data", outputs=[f"{self._code}.ms"], inputs=[str(uvfits)])
        return obs._metadata

    def from_ms(self, ms, *, force: bool = False, **kwargs):
        """Adopt an already-imported measurement set."""
        obs = self._obs
        if not obs._state.should_run("import_data", force=force):
            return obs.metadata
        logger.info("import_data.from_ms: {}", ms)
        if self._backend.supports("data", "adopt_ms"):
            self._backend.data.adopt_ms(self._code, str(ms))
        else:
            self._backend.data.import_data(self._code, obs.sources.names, **kwargs)
        obs._metadata = self._load_metadata()
        obs._state.mark_complete("import_data", outputs=[str(ms)], inputs=[str(ms)])
        return obs._metadata


class CalibrateNamespace(Namespace):
    """Calibration: a-priori, single-band delay, bandpass, global fringe fit, apply."""

    def __call__(self, *, force: bool = False) -> list[CalTable]:
        """Run the full default calibration chain in order."""
        self.a_priori(force=force)
        self.initial_calibration(force=force)
        self.bandpass(force=force)
        self.fringefit(force=force)
        self.apply(force=force)
        return self._obs.gaintables

    def a_priori(self, *, force: bool = False, smooth: Optional[bool] = None,
                 plot: Optional[bool] = None) -> list[CalTable]:
        """A-priori calibration: gencal Tsys + gain curve (+ ACCOR and EOP for VLBA/LBA).

        For DiFX data (``needs_accor``) the correlator amplitude correction is
        solved first from the autocorrelations (``accor`` + ``smoothcal``,
        ``[calibration.accor]``), which are flagged only afterwards.

        Refuses to run unless the measurement set actually carries Tsys and
        gain-curve data (those come from the ``.antab`` at import time). The
        resulting tables are then de-spiked and, when needed, smoothed, and
        plotted per antenna.

        Parameters
        ----------
        force : bool
            Re-run even if the step already completed.
        smooth : bool, optional
            Clean outliers out of the Tsys table (default ``[calibration].smooth_tsys``).
        plot : bool, optional
            Plot each produced table (default True).
        """
        obs = self._obs
        if not obs._state.should_run("a_priori", force=force):
            return obs.gaintables
        cal_cfg = obs.config.get("calibration", {})
        handler = obs._observatory_handler
        antab = handler.get_antab_file(self._code, obs.work_dir)
        accor_cfg = dict(cal_cfg.get("accor", {}))
        needs_accor = bool(handler.needs_accor and accor_cfg.pop("enabled", True))
        tables = self._backend.calibrate.a_priori(
            self._code, obs.calibrator_field, needs_eop=handler.needs_eop,
            eop_file=cal_cfg.get("eop_file") or None, antab=antab,
            needs_accor=needs_accor, accor=accor_cfg)
        if needs_accor and obs._state.status("flag_autocorr") != "done":
            # flag.apriori left the autocorrelations for accor; it has used them now.
            obs.flag.autocorr(force=force)
        if smooth if smooth is not None else cal_cfg.get("smooth_tsys", True):
            tables = [self._smooth_table(table, cal_cfg) for table in tables]
        for table in tables:
            obs.add_gaintable(table, "a_priori")
        if plot is not False:
            for table in tables:
                obs.plot.caltable(table)
        obs._state.mark_complete("a_priori", outputs=[t.cal_type for t in tables])
        return tables

    def _smooth_table(self, table: CalTable, cal_cfg: dict) -> CalTable:
        """De-spike one a-priori table, leaving it untouched if the backend cannot smooth."""
        if table.cal_type != "tsys" or not self._backend.supports("calibrate", "smooth"):
            return table
        smooth_cfg = {"threshold": cal_cfg.get("tsys_outlier_sigma", 6.0),
                      "max_passes": cal_cfg.get("tsys_smooth_passes", 3)}
        smooth_cfg.update(cal_cfg.get("tsys_smooth", {}))
        return self._backend.calibrate.smooth(self._code, table, **smooth_cfg)

    def select_calibration_data(self, *, min_snr: Optional[float] = None) -> tuple[list[str], list[int]]:
        """Choose the antennas and scan(s) the instrumental calibration is solved on.

        Antennas must have been detected above ``min_snr`` (partial-band antennas
        included); scans are then picked so that all of those antennas are
        detected, using several linked scans when no single one covers the array.
        The survey behind it only sees the usable band — the subbands at least two
        antennas recorded — and the sources it runs on are the declared roles, or
        every observed source when no role is declared (see
        :meth:`_calibration_source_groups`).

        Returns
        -------
        tuple
            ``(antenna names best-first, scan numbers)``.
        """
        from ..selection import plan_sbd_stages, select_antennas, without_weak_detections

        obs = self._obs
        # Decide the antenna/scan set once — on the first (cleanest) pass — and reuse it for
        # every later pass. Re-surveying on the progressively flagged data of a second/third
        # pass lets the detectable set shrink (e.g. 8 -> 6 -> 2 antennas on a sparse fringe
        # finder), and applycal (calflagstrict) then flags every antenna the shrunken solution
        # no longer covers — silently deleting good antennas. The calibratable array is a
        # property of the observation, not something a later pass should re-litigate on
        # dirtier data.
        if obs.cal_selection is not None:
            antennas, scans = obs.cal_selection
            # The one thing that reopens it: an antenna whose own stage scan has been flagged away.
            replanned = self._replan_lost_antennas(antennas, scans)
            if replanned is not None:
                return replanned
            logger.info("select_calibration_data: reusing the instrumental selection fixed on the "
                        "first pass — {} antenna(s) {} on scan(s) {} (not re-surveying the now-flagged "
                        "data)", len(antennas), ",".join(antennas), scans)
            return antennas, scans
        cal_cfg = obs.config.get("calibration", {})
        threshold = min_snr if min_snr is not None else cal_cfg.get("detection_snr", 7.0)
        self.infer_source_roles()
        # Survey one source group at a time, in the order the calibration falls back
        # through them. The fringe finder is both the right source to solve on and
        # usually the one with fewest scans, so this avoids fringe-fitting every
        # phase-calibrator scan just to end up not using any of them.
        attempted, best = [], (0.0, "")
        for names in self._calibration_source_groups():
            attempted.extend(names)
            survey = self.scan_snr(field=",".join(names))
            ranked = survey.rank_antennas()
            best = max([best] + [(snr, name) for name, snr in ranked if snr == snr and snr != float("inf")])
            antennas = select_antennas(survey, obs.metadata, min_snr=threshold)
            if not antennas:
                logger.info("select_calibration_data: no antenna qualifies on {}; trying the "
                            "next source group", ",".join(names))
                continue
            # Detected is not the same as on source: do not solve an antenna on a scan where it
            # sits far below its own level in the other scans.
            planning, _ = without_weak_detections(survey, self._weak_in_scans(names, survey), min_snr=threshold)
            stages = plan_sbd_stages(planning, antennas, min_snr=threshold, sources=names, metadata=obs.metadata)
            scans = [stage["scan"] for stage in stages]
            if scans:
                obs.set_cal_selection(antennas, scans, stages)
                logger.info("select_calibration_data: fixed the instrumental selection for the whole "
                            "run — {} antenna(s) {} on scan(s) {}; every pass solves on this set",
                            len(antennas), ",".join(antennas), scans)
                return antennas, scans
        best_snr, best_antenna = best
        detected = f"the best was {best_antenna} at {best_snr:.1f}" if best_antenna else "nothing was detected at all"
        usable = list(obs.metadata.usable_subbands) if obs.metadata else []
        band = f"subband(s) {usable} of {obs.metadata.freq_setup.n_subbands}" if obs.metadata else "unknown"
        raise StepError("select_calibration_data",
                        f"{self._code}: no scan on {', '.join(attempted)} detects an antenna above "
                        f"{threshold:g} sigma ({detected}); the usable band is {band}. Lower "
                        f"calibration.detection_snr or check the flagging; cannot solve the instrumental delay")

    def _weak_in_scans(self, names: list[str], survey) -> dict[int, list[str]]:
        """Scans of the survey in which an antenna is far below its own level in the other scans.

        The survey's fringe SNR cannot tell: an antenna in a sidelobe of a bright
        fringe finder is still detected, at nearly the same SNR (EM163 scan 103: CM
        at 6% of its amplitude, SNR 105/170 against 147/252 and 187/224 in its good
        scans). The raw amplitudes can. They are read per source for the surveyed
        scans (:meth:`read_baseline_timeline` with ``delay_search``, since nothing is
        fringe-fitted yet) and compared between scans of the same baseline
        (:func:`vlbipy.statistics.weak_scans_per_antenna`).
        ``[calibration].weak_scan_level`` (default 0.3; 0 disables) is the fraction of
        its best scan below which an antenna is not solved on a scan.

        Returns
        -------
        dict
            ``{scan: [antennas]}``; empty when the backend cannot read the data this
            way, when it is disabled, or when no source has two surveyed scans.
        """
        from ..statistics import weak_scans_per_antenna

        obs = self._obs
        level = float(obs.config.get("calibration", {}).get("weak_scan_level", 0.3))
        if level <= 0 or not self._backend.supports("data", "read_baseline_timeline"):
            return {}
        weak: dict[int, list[str]] = {}
        for field in names:
            scans = [scan for scan, source in zip(survey.scan_numbers, survey.scan_sources) if source == field]
            if len(scans) < 2:
                continue
            try:
                data = self._backend.data.read_baseline_timeline(self._code, field=field, column="data",
                                                                 delay_search=True, scans=scans, metadata=obs.metadata)
            except BackendError as exc:
                logger.info("scan selection: could not compare the amplitudes of {} between scans ({})", field, exc)
                continue
            found = weak_scans_per_antenna(data["amplitude"], data["scans"], level=level)
            for index, name in enumerate(data["antennas"]):
                for position in found["weak"][index].nonzero()[0]:
                    scan = int(found["scans"][position])
                    weak.setdefault(scan, []).append(name)
                    logger.info("scan selection: {} is at {:.0%} of its own best scan in scan {} of {}; not on "
                                "source there, whatever its fringe SNR", name, found["gains"][index][position],
                                scan, field)
        return weak

    def _replan_lost_antennas(self, antennas: list[str], scans: list[int], *,
                              limit: float = 0.95) -> Optional[tuple[list[str], list[int]]]:
        """Re-plan the instrumental stages when flagging has emptied an antenna's stage scan.

        The selection is fixed on the first pass and deliberately not re-surveyed
        afterwards. That leaves one hole: an antenna whose data in the scan it is
        solved on gets flagged later. EM163: CM was detected in the chosen
        fringe-finder scan, but at 6% of its amplitude (off source for the whole
        scan; the fringe SNR barely shows it), and the off-source flagging rightly
        removed it there. With the plan unchanged it then has no single-band delay
        or bandpass, and applycal flags *all* its data. Such an antenna is given
        another scan (:func:`vlbipy.selection.amend_sbd_stages`); the others keep
        theirs, and the antenna set does not change.

        Parameters
        ----------
        antennas, scans
            The selection in use.
        limit : float
            An antenna counts as lost in a scan when at least this fraction of the
            data it recorded there is flagged.

        Returns
        -------
        tuple or None
            The new ``(antennas, scans)``, also persisted; None when no antenna lost
            its scan (or the backend cannot count flags per scan).
        """
        from ..selection import amend_sbd_stages, plan_sbd_stages, without_weak_detections

        obs = self._obs
        stages = obs.cal_stages
        if not stages or [stage["scan"] for stage in stages] != list(scans):
            return None
        if not self._backend.supports("flag", "summary"):
            return None
        lost: dict[int, list[str]] = {}
        for stage in stages:
            scan = int(stage["scan"])
            try:
                counts = self._backend.flag.summary(self._code, where=f"SCAN_NUMBER == {scan}").get("antenna", {})
            except BackendError as exc:
                logger.warning("select_calibration_data: could not count the flags of scan {} ({}); keeping the "
                               "selection as it is", scan, exc)
                return None
            for name in dict.fromkeys([*stage["antennas"], stage["refant"]]):
                entry = counts.get(name)
                if entry and entry["observable"] and entry["fraction"] >= limit and name not in lost.get(scan, []):
                    lost.setdefault(scan, []).append(name)
        if not lost:
            return None
        described = "; ".join(f"{', '.join(names)} in scan {scan}" for scan, names in sorted(lost.items()))
        source_of = {scan.scan_number: scan.source for scan in obs.metadata.scans} if obs.metadata else {}
        used = {source_of.get(int(scan)) for scan in scans}
        names = next((group for group in self._calibration_source_groups() if used & set(group)), [])
        if not names:
            warnings.anomaly(f"{self._code}: flagging left no data for {described}, and the sources of the "
                             "instrumental selection are unknown; those antennas will have no instrumental solution")
            return None
        threshold = obs.config.get("calibration", {}).get("detection_snr", 7.0)
        logger.info("select_calibration_data: flagging left no data for {}; looking for another scan on {}",
                    described, ",".join(names))
        survey = self.scan_snr(field=",".join(names), force=True)
        survey, _ = without_weak_detections(survey, self._weak_in_scans(names, survey), min_snr=threshold)
        amended = amend_sbd_stages(stages, survey, lost, min_snr=threshold, sources=names, metadata=obs.metadata)
        if not amended:
            # A stage lost its own reference antenna, so nothing of the chain can be kept.
            amended = plan_sbd_stages(survey, antennas, min_snr=threshold, sources=names, metadata=obs.metadata)
        if not amended:
            warnings.anomaly(f"{self._code}: flagging left no data for {described} and no scan on "
                             f"{', '.join(names)} can replace it; keeping the selection as it is")
            return None
        new_scans = [stage["scan"] for stage in amended]
        unsolved = sorted(set(antennas) - {name for stage in amended for name in stage["antennas"]})
        if unsolved:
            warnings.anomaly(f"{self._code}: {', '.join(unsolved)} lost the scan of the instrumental calibration "
                             "to flagging and no other scan detects them; all their data will be flagged on apply")
        obs.set_cal_selection(antennas, new_scans, amended)
        logger.info("select_calibration_data: instrumental selection re-planned for the antennas that lost "
                    "their scan — {} antenna(s) on scan(s) {}", len(antennas), new_scans)
        return antennas, new_scans

    def infer_source_roles(self, *, force: bool = False) -> dict:
        """Give the sources in the data a role when the configuration declares none for them.

        Without a role every field selection falls back to "all sources", and the
        instrumental delay is then solved on whichever scan happens to cover the
        most antennas — possibly a weak target, in which case the fringe fit
        returns nothing above its minimum SNR. The roles are reconstructed from
        the data: the scan pattern and sky positions say which sources form a
        phase-referencing cycle, and a fringe SNR survey over *every* source says
        how well the array detects each one
        (:func:`~vlbipy.sources.infer_source_roles`).

        Nothing happens when every observed source already has a declared role;
        a declared role is never overridden. The survey this runs is the one the
        instrumental selection then reuses, so the extra fringe fit is not
        repeated per source group.

        Parameters
        ----------
        force : bool
            Re-infer even for sources that already have a role (the declared ones
            still win; this only re-runs the survey and the inference).

        Returns
        -------
        dict
            ``{source: SourceType}`` actually adopted (empty when nothing was missing).
        """
        from ..sources import infer_source_roles as infer

        obs = self._obs
        if obs.metadata is None:
            return {}
        observed = [name for name in (obs.metadata.source_names or []) if name]
        missing = [name for name in observed if name not in obs.sources]
        if not missing and not force:
            return {}
        # Inferring the roles costs a fringe fit per surveyed scan, so an earlier run's answer
        # is reused: the roles are a property of the observation, not of a pass. --scratch
        # and force discard it.
        if not force:
            stored, stored_phaseref = obs.load_source_roles()
            reusable = {name: role for name, role in stored.items() if name in missing}
            if reusable and not set(missing) - set(reusable):
                logger.info("source roles: reusing the roles inferred on an earlier run ({})",
                            ", ".join(f"{n} = {r.value}" for n, r in reusable.items()))
                obs.adopt_source_roles(reusable, stored_phaseref, persist=False)
                return reusable
        logger.info("source roles: {} of {} observed source(s) have no declared role ({}); inferring "
                    "them from the data", len(missing), len(observed), ", ".join(missing) or "none")
        survey = None
        try:
            survey = self.scan_snr(field=",".join(observed), scans=self._fair_scan_sample(observed),
                                   max_scans=0)
        except Exception as exc:  # noqa: BLE001 - the structural inference works without a survey
            warnings.warn(f"{self._code}: could not survey every source to rank them by SNR ({exc}); "
                          "inferring the source roles from the scan pattern alone")
        declared = {s.name: s.source_type for s in obs.sources}
        roles, phaseref = infer(obs.metadata, survey, declared=declared)
        obs.adopt_source_roles(roles, phaseref)
        return roles

    def _field_for_scans(self, scans: Optional[list]) -> str:
        """Return the field selection matching the scans the instrumental solve runs on.

        The field and the scan selection are combined by the solver, so they have
        to agree: solving ``field=<fringe finder>`` on a scan the selection took
        from the phase calibrator (because the fringe finder detected nothing)
        selects no data at all, and a staged solve may legitimately take its
        stages from different sources. The sources of the chosen scans are
        therefore what the field is built from, falling back to
        :attr:`~vlbipy.observation.Observation.calibrator_field` when no scan is known.

        Parameters
        ----------
        scans : list or None
            Scan numbers the solve is restricted to.

        Returns
        -------
        str
            Comma-separated source names.
        """
        obs = self._obs
        if not scans or obs.metadata is None:
            return obs.calibrator_field
        wanted = {int(scan) for scan in scans}
        names = [s.source for s in obs.metadata.scans if s.scan_number in wanted and s.source]
        return ",".join(dict.fromkeys(names)) or obs.calibrator_field

    def _fair_scan_sample(self, sources: list[str]) -> list[int]:
        """Return scan numbers spread evenly *within every source*, for the role-ranking survey.

        The ordinary survey sampler spreads its budget over the observation as a
        whole, which can drop a fringe finder's two scans entirely — and a source
        with no row in the survey looks undetected, which is exactly the wrong
        conclusion to draw about the brightest source of the run. Each source
        therefore gets its own even sample.

        Parameters
        ----------
        sources : list of str
            Sources to sample (every observed source).

        Returns
        -------
        list of int
            Scan numbers, sorted; empty when the metadata has no scan.
        """
        obs = self._obs
        budget = int(obs.config.get("calibration", {}).get("snr_max_scans", 24) or 24)
        per_source = max(2, budget // max(1, len(sources)))
        sampled: list[int] = []
        for name in sources:
            scans = [s.scan_number for s in obs.metadata.scans if s.source == name]
            if len(scans) <= per_source:
                sampled.extend(scans)
                continue
            step = len(scans) / float(per_source)
            sampled.extend(scans[min(len(scans) - 1, int(index * step))] for index in range(per_source))
        return sorted(dict.fromkeys(sampled))

    def _calibration_source_groups(self) -> list[list[str]]:
        """Source names to survey for the instrumental solve, best group first.

        Normally the declared roles, tried in the order the calibration falls back through them
        (fringe finders, then phase calibrators, then targets). An observation with no role
        declared at all — an external project with no ``[sources]`` configuration, or one whose
        declared names do not match the data and were all dropped by
        :meth:`~vlbipy.observation.Observation.restrict_sources_to_data` — falls back to every
        source the data holds, because the instrumental delay only needs one bright scan.

        Returns
        -------
        list of list of str
        """
        obs = self._obs
        groups = [[s.name for s in group] for group in
                  (obs.sources.fringe_finders, obs.sources.phase_calibrators, obs.sources.targets) if group]
        if groups:
            return groups
        observed = list(obs.metadata.source_names) if obs.metadata else []
        if not observed:
            raise StepError("select_calibration_data",
                            f"{self._code}: no source has a role (fringe_finders / phase_calibrators / "
                            f"targets) and the metadata lists no source either; declare [sources] in the "
                            f"configuration")
        warnings.anomaly(f"{self._code}: no source has a role (fringe_finders / phase_calibrators / targets) "
                         f"in this observation — check that the [sources] names match the data; surveying "
                         f"every observed source instead: {', '.join(observed)}")
        return [observed]

    def initial_calibration(self, *, force: bool = False, scans: Optional[list] = None,
                            antennas: Optional[list] = None, suffix: str = "sbd") -> CalTable:
        """Single-band delay (instrumental) calibration on the selected scan(s)."""
        obs = self._obs
        step = f"initial_calibration_{suffix}" if suffix != "sbd" else "initial_calibration"
        if not obs._state.should_run(step, force=force):
            return obs._table(suffix)
        if scans is None:
            antennas, scans = self.select_calibration_data()
        cal_cfg = obs.config.get("calibration", {})
        # Drop a table this same step produced before: re-running (e.g. a resumed process,
        # or calling instrumental() again) would otherwise pass the *old* table as one of
        # this solve's own on-the-fly priors while it is being overwritten at the same path.
        obs.drop_gaintables({step})
        # The whole [calibration.sbd] section is forwarded: its keys are the backend
        # function's parameters, so any of them can be tuned without a code change.
        settings = {"channel_fraction": cal_cfg.get("snr_channel_fraction", 0.8)}
        settings.update(cal_cfg.get("sbd", {}))
        # The stage plan (one scan per stage, chained through a shared antenna) is what
        # makes the delay a single time-constant solution per antenna; without a plan
        # (explicit scans from the caller) every scan is solved in one stage.
        stages = obs.cal_stages if obs.cal_stages and [st["scan"] for st in obs.cal_stages] == list(scans) else []
        # Preserve exactly what each instrumental solve saw.  A staged SBD may use a
        # different scan and reference antenna at every stage; plotting the combined
        # field with the observation-wide refant hides missing or weak stage links.
        plot_stages = stages or [{"scan": scan, "refant": (antennas or [obs.refant])[0]}
                                 for scan in scans]
        # Before the first applycal there is no corrected column to read: the first
        # instrumental solve sees the raw data.
        stage_column = "corrected" if obs._state.status("apply") == "done" else "data"
        for stage_index, stage in enumerate(plot_stages, start=1):
            scan_number = int(stage["scan"])
            scan = next((s for s in obs.metadata.scans if s.scan_number == scan_number), None)
            if scan is None:
                continue
            try:
                obs.plot.timeseries(field=scan.source, scans=[scan_number], column=stage_column,
                                    refant=str(stage["refant"]),
                                    label=f"{suffix}_stage{stage_index}_scan{scan_number}")
            except Exception as exc:  # noqa: BLE001 - reporting must not prevent calibration
                warnings.warn(f"{self._code}: SBD stage {stage_index} data plot failed ({exc})")
        table = self._backend.calibrate.initial_calibration(
            self._code, self._field_for_scans(scans), ",".join(antennas or []) or obs.refant,
            scans=scans, stages=stages, gaintable=list(obs.gaintables), metadata=obs.metadata,
            suffix=suffix, callib=cal_cfg.get("callib", False), **settings)
        obs.add_gaintable(table, step)
        obs._state.mark_complete(step, outputs=[suffix])
        return table

    def bandpass(self, *, force: bool = False, scans: Optional[list] = None,
                 antennas: Optional[list] = None) -> CalTable:
        """Bandpass calibration on the same scan(s) as the instrumental delay."""
        obs = self._obs
        if not obs._state.should_run("bandpass", force=force):
            return obs._table("bpass")
        if scans is None:
            antennas, scans = self.select_calibration_data()
        obs.drop_gaintables({"bandpass"})
        cal_cfg = obs.config.get("calibration", {})
        bp_cfg = dict(cal_cfg.get("bandpass", {}))
        try:
            table = self._backend.calibrate.bandpass(
                self._code, self._field_for_scans(scans), ",".join(antennas or []) or obs.refant,
                scans=scans, gaintable=list(obs.gaintables), metadata=obs.metadata,
                callib=cal_cfg.get("callib", False), **bp_cfg)
        except BackendError as exc:
            # A re-solve on already-flagged data can fail when the fringe finder has
            # too few unflagged antennas left (common with a single, sparse FF on a
            # third pass). The band shape is stable across passes, so fall back to the
            # bandpass the earlier pass solved rather than aborting the whole run.
            fallback = self._existing_table("bpass", interp="nearest,nearest")
            if fallback is None:
                raise
            warnings.warn(f"{self._code}: bandpass re-solve failed ({exc}); keeping the "
                          "previous bandpass table")
            table = fallback
        obs.add_gaintable(table, "bandpass")
        obs._state.mark_complete("bandpass", outputs=["bpass"])
        return table

    def _existing_table(self, cal_type: str, interp: str = "nearest") -> Optional[CalTable]:
        """Return a calibration table already on disk for ``cal_type``, or None.

        Used as the fallback when a re-solve cannot proceed: the backend leaves the
        previous (successfully-solved) table in place, and this rebuilds a CalTable
        pointing at it so it stays in the apply chain. The file name follows the
        backend convention ``<code>.<cal_type>``.
        """
        caldir = getattr(self._backend, "caldir", None)
        if caldir is None:
            return None
        path = caldir() / f"{self._code}.{cal_type}"
        if not path.is_dir():
            return None
        return CalTable(cal_type=cal_type, path=str(path), field=self._obs.calibrator_field,
                        interp=interp)

    def instrumental(self, *, force: bool = False, plot: bool = True) -> list[CalTable]:
        """Full instrumental calibration: SBD, MBD, bandpass, then SBD and MBD again.

        The order matters. The first single-band delay is solved on a band whose
        shape is still uncorrected, and the first multi-band delay inherits that
        bias — but the bandpass itself is best solved on data whose delays have
        already been removed, otherwise the residual delay slopes across each
        subband are absorbed into the band shape. The way out is to go round
        once: rough delays, bandpass with those applied, then re-solve both
        delays through the known bandpass.

        Final chain: ``sbd2`` and ``mbd2`` replace the first-pass ``sbd``/``mbd``,
        with the bandpass between them.

        Returns
        -------
        list of CalTable
            ``[bandpass, refined SBD, refined MBD]`` — what stays in the chain.
        """
        obs = self._obs
        antennas, scans = self.select_calibration_data()
        logger.info("instrumental calibration: {} antenna(s) {} on scan(s) {}",
                    len(antennas), ",".join(antennas), scans)

        # Each solve is made through the tables before it in this sequence and none after it. So
        # from the first step that has to run, that step and every later one come out of the chain
        # and are solved again; finished steps before it are kept (a resumed run skips them all).
        sequence = INSTRUMENTAL_STEPS
        pending = [force or obs._state.status(name) != "done" for name in sequence]
        first = pending.index(True) if True in pending else len(sequence)
        if first < len(sequence):
            stale = obs.drop_gaintables(sequence[first:])
            if stale:
                logger.info("instrumental calibration: re-solving from {}; dropped {} from the chain",
                            sequence[first], ", ".join(stale))
        rerun = [index >= first for index in range(len(sequence))]

        # Pass 1: rough delays, so the bandpass is solved on delay-corrected data.
        first_sbd = self.initial_calibration(force=rerun[0], scans=scans, antennas=antennas)
        first_mbd = self.fringefit(force=rerun[1], plot=False)

        # Bandpass, now that both delays are removed.
        bandpass = self.bandpass(force=rerun[2], scans=scans, antennas=antennas)

        # Pass 2: re-solve both delays *through* everything already in the chain. These
        # are incremental corrections on top of the first pass, not replacements — the
        # first-pass tables stay, and all of them are applied together.
        refined_sbd = self.initial_calibration(force=rerun[3], scans=scans, antennas=antennas,
                                               suffix="sbd2")
        refined_mbd = self.fringefit(force=rerun[4], suffix="mbd2", plot=plot)

        # Checked and plotted when solved: a resumed run that keeps the tables has already reported them.
        if rerun[3]:
            self.verify_solutions(refined_sbd, antennas)
        if plot:
            for table, solved in ((bandpass, rerun[2]), (refined_sbd, rerun[3])):
                if solved:
                    obs.plot.caltable(table)
        logger.info("instrumental: chain is now {}",
                    " -> ".join(t.cal_type for t in obs.gaintables))
        return [first_sbd, first_mbd, bandpass, refined_sbd, refined_mbd]

    def _solve_dispersive(self, cal_cfg: dict) -> bool:
        """Decide whether the fringe fit should also solve the dispersive delay.

        The ionosphere delays low frequencies more than high ones, so below
        ``[calibration].ionos_max_ghz`` (8 GHz by default) the residual delay is
        genuinely dispersive and fitting a single non-dispersive delay leaves a
        frequency-dependent phase behind. Above that the effect is negligible
        and the extra free parameter only costs SNR.

        Set ``[calibration].ionos = false`` (or ``--no-ionos``) to never solve
        for it, whatever the frequency.
        """
        obs = self._obs
        if not cal_cfg.get("ionos", True):
            logger.info("fringefit: dispersive delay disabled ([calibration].ionos is false)")
            return False
        freq_ghz = obs.metadata.freq_setup.freq_ghz if obs.metadata else 0.0
        threshold = float(cal_cfg.get("ionos_max_ghz", 8.0))
        if not freq_ghz:
            return False
        dispersive = freq_ghz < threshold
        logger.info("fringefit: observing at {:.3f} GHz, {} {:.0f} GHz -> {} solve for the "
                    "dispersive (ionospheric) delay", freq_ghz,
                    "below" if dispersive else "above", threshold,
                    "will" if dispersive else "will not")
        return dispersive

    def second_pass(self, *, force: bool = False, plot: bool = True,
                    step: str = "second_pass") -> list[CalTable]:
        """Re-solve the instrumental and fringe calibration on the now-flagged data.

        The first solutions were derived from data that still contained whatever
        the flagging steps later removed — band edges, RFI, outliers, slewing
        data — so those points biased them. Re-solving from the a-priori tables
        (Tsys / gain curve / ACCOR / EOP, which do not depend on the solutions) gives cleaner
        solutions. The data-derived tables are replaced, not stacked.

        Parameters
        ----------
        step : str
            State name recorded for this pass (``"second_pass"``; the pipeline
            uses ``"third_pass"`` for the re-solve after reweighting).
        """
        obs = self._obs
        if not obs._state.should_run(step, force=force):
            return obs.gaintables
        # Everything the a-priori step made stays (Tsys, gain curve, and for DiFX data ACCOR and
        # EOP); the types are named as well for chains recorded before tables carried their step.
        apriori = [t for t in obs.gaintables
                   if t.step == "a_priori" or t.cal_type in ("accor", "tsys", "gc", "eop")]
        dropped = [t.cal_type for t in obs.gaintables if t not in apriori]
        obs.set_gaintables(apriori)
        logger.info("{}: re-deriving {} from the a-priori tables on the flagged data",
                    step.replace("_", " "), ", ".join(dropped) or "nothing")
        obs._snr_surveys.clear()
        for name in ("scan_snr", "initial_calibration", "bandpass", "fringefit",
                     "initial_calibration_sbd2", "fringefit_mbd2", "scalar_bandpass"):
            obs._state.invalidate_downstream(name, [name])
        self.instrumental(force=True, plot=plot)
        obs._state.mark_complete(step, outputs=[t.cal_type for t in obs.gaintables])
        return obs.gaintables

    def reweight(self, *, force: bool = False, **kwargs) -> dict:
        """Recompute the visibility weights from the calibrated data (``statwt``).

        Run after the full chain has been applied. The new weights expose bad
        data that hid until now (anomalously high or low weights), so the
        pipeline follows this with another outlier flag and a full re-solve of
        the calibration (``second_pass(step="third_pass")``).
        """
        obs = self._obs
        if not obs._state.should_run("reweight", force=force):
            return {}
        if not self._backend.supports("calibrate", "reweight"):
            logger.info("reweight: backend {} does not implement it; skipping", self._backend.kind)
            return {}
        cfg = dict(obs.config.get("calibration", {}).get("reweight", {}))
        cfg.pop("enabled", None)
        cfg.update(kwargs)
        report = self._backend.calibrate.reweight(self._code, **cfg)
        obs._state.mark_complete("reweight", outputs=[f"mean={report.get('mean')}"])
        return report

    def scalar_bandpass(self, *, force: bool = False, plot: bool = True) -> Optional[CalTable]:
        """Solve one amplitude gain per antenna and subband, levelling the subbands."""
        obs = self._obs
        if not obs._state.should_run("scalar_bandpass", force=force):
            return obs._table("scalar_bp")
        if not self._backend.supports("calibrate", "scalar_bandpass"):
            logger.info("scalar_bandpass: backend {} does not implement it; skipping",
                        self._backend.kind)
            return None
        obs.drop_gaintables({"scalar_bandpass"})
        cal_cfg = obs.config.get("calibration", {})
        cfg = dict(cal_cfg.get("scalar_bandpass", {}))
        # Solve on the phase-reference sources (the phase calibrators, or the targets with inverse phase
        # referencing): gaincal assumes a point source, so a resolved fringe finder would have its
        # structure absorbed into the antenna gains.
        compact = obs.phaseref_plan().references or [s.name for s in obs.sources.calibrators]
        try:
            table = self._backend.calibrate.scalar_bandpass(
                self._code, ",".join(compact) or obs.calibrator_field,
                obs.refant, gaintable=list(obs.gaintables), metadata=obs.metadata,
                callib=cal_cfg.get("callib", False), **cfg)
            # The solve compares the two hands of every antenna on amplitudes: a much weaker one goes now,
            # before the next pass re-solves and the Stokes I self-calibration averages it in.
            hands = Path(table.path).with_name(f"{self._code}.scalar_bp.hands.json")
            if hands.is_file():
                obs.flag.weak_polarizations_from_gains(json.loads(hands.read_text()))
        except BackendError as exc:
            # Optional subband levelling: a re-solve on heavily-flagged data can fail. Keep
            # the previous pass's table if there is one; otherwise skip it rather than abort
            # the whole run — the calibration is still valid without the levelling.
            fallback = self._existing_table("scalar_bp")
            if fallback is None:
                warnings.warn(f"{self._code}: scalar bandpass could not be solved ({exc}) and "
                              "no earlier table exists; continuing without subband levelling")
                obs._state.mark_complete("scalar_bandpass", outputs=[])
                return None
            warnings.warn(f"{self._code}: scalar bandpass re-solve failed ({exc}); keeping the "
                          "previous scalar bandpass table")
            table = fallback
        obs.add_gaintable(table, "scalar_bandpass")
        if plot:
            obs.plot.caltable(table)
        obs._state.mark_complete("scalar_bandpass", outputs=["scalar_bp"])
        return table

    def verify_solutions(self, table: CalTable, antennas: list[str]) -> dict[str, set[int]]:
        """Warn about antennas missing solutions in subbands they actually recorded.

        A missing solution is not harmless: at apply time that antenna/subband is
        flagged, silently shrinking the array.
        """
        obs = self._obs
        if not self._backend.supports("calibrate", "solution_coverage"):
            return {}
        coverage = self._backend.calibrate.solution_coverage(self._code, table, obs.metadata)
        # Only the usable band can be solved: a subband with a single antenna has no baseline.
        usable = set(obs.metadata.usable_subbands) if obs.metadata else set()
        for name in antennas:
            recorded = set(obs.metadata.antennas[name].subbands) & usable if obs.metadata else set()
            solved = coverage.get(name, set())
            missing = sorted(recorded - solved)
            if missing:
                warnings.anomaly(f"{self._code}: {name} has no {table.cal_type} solution in "
                                 f"subband(s) {missing} although it recorded them")
        return coverage

    def fringefit(self, *, force: bool = False, plot: bool = True,
                  suffix: str = "mbd") -> CalTable:
        """Global (multi-band delay) fringe fit on the fringe finders and the phase-reference sources.

        The reference sources (:meth:`Observation.phaseref_plan`: the phase calibrators, or the targets with
        inverse phase referencing) are solved together and their solutions are what the other fields take.
        With the ``sequential`` strategy only the first (brightest) one is solved with the fringe finders and
        goes to every field; each next one is then solved on its own, on data corrected by the tables before
        it, into ``<suffix>_<source>``, which goes to itself, the later references and the receivers.
        """
        obs = self._obs
        step = "fringefit" if suffix == "mbd" else f"fringefit_{suffix}"
        if not obs._state.should_run(step, force=force):
            return obs._table(suffix)
        plan = obs.phaseref_plan()
        finders = [s.name for s in obs.sources.fringe_finders]
        first = plan.references[:1] if plan.strategy == "sequential" else plan.references
        names = list(dict.fromkeys(finders + first)) or [s.name for s in obs.sources.targets]
        obs.drop_gaintables({step})
        cal_cfg = obs.config.get("calibration", {})
        mbd_cfg = dict(cal_cfg.get("mbd", {}))
        mbd_cfg.setdefault("dispersive", self._solve_dispersive(cal_cfg))
        # The fringe solutions are what gets transferred to the other fields: tie them to the
        # reference source(s) so applycal maps those solutions onto every field.
        applied = self._fringefit_table(",".join(names), suffix, cal_cfg, mbd_cfg, gainfield=",".join(first),
                                        plot=plot, step=step)
        outputs = [suffix]
        if plan.strategy == "sequential":
            observed = obs.metadata.source_names if obs.metadata else list(obs.sources.names)
            for name in plan.references[1:]:
                logger.info("fringefit: {} solved on data already corrected by {} (sequential phase referencing)",
                            name, ", ".join(plan.references[:plan.references.index(name)]))
                self._fringefit_table(name, f"{suffix}_{name}", cal_cfg, mbd_cfg, gainfield=name,
                                      apply_to=plan.apply_to(name, observed) or "", plot=plot, step=step)
                outputs.append(f"{suffix}_{name}")
        obs._state.mark_complete(step, outputs=outputs)
        return applied

    def _fringefit_table(self, field: str, suffix: str, cal_cfg: dict, mbd_cfg: dict, *, gainfield: str,
                         step: str, apply_to: str = "", plot: bool = True) -> CalTable:
        """Solve one fringe table on ``field``, smooth it and append it to the chain; return the applied table."""
        obs = self._obs
        table = self._backend.calibrate.fringefit(
            self._code, field, obs.refant, gaintable=list(obs.gaintables),
            metadata=obs.metadata, suffix=suffix,
            callib=cal_cfg.get("callib", False), **mbd_cfg)
        if gainfield:
            table.gainfield = gainfield
            logger.info("fringefit: solutions will be applied from {}", table.gainfield)
        table.apply_to = apply_to
        if plot:
            obs.plot.caltable(table)
        applied = self._smooth_fringe_table(table)
        obs.add_gaintable(applied, step)
        if plot and applied.path != table.path:
            obs.plot.caltable(applied)
        return applied

    def _smooth_fringe_table(self, table: CalTable) -> CalTable:
        """Smooth a fringe table in time and return the table to apply.

        The solved table stays on disk and is plotted; it is the smoothed copy
        that enters the apply chain, because a per-interval delay and rate carry
        the noise of one interval while the atmospheric residual they measure
        drifts slowly — so the raw solutions transfer their own scatter onto the
        target. Configured by ``[calibration.mbd_smooth]``, whose keys go straight
        to the backend; ``window = 0`` applies the solved table instead.

        A backend that cannot smooth, or a failure part-way, leaves the solved
        table in the chain rather than aborting the run: an unsmoothed fringe
        solution is correct, just noisier.

        Parameters
        ----------
        table : CalTable
            The freshly solved fringe table.

        Returns
        -------
        CalTable
            The table to apply (the smoothed copy, or ``table`` itself).
        """
        obs = self._obs
        settings = dict(obs.config.get("calibration", {}).get("mbd_smooth", {}))
        if not self._backend.supports("calibrate", "smooth_fringe"):
            return table
        try:
            return self._backend.calibrate.smooth_fringe(self._code, table, metadata=obs.metadata,
                                                         **settings)
        except Exception as exc:  # noqa: BLE001 - the unsmoothed solution is still a valid one
            warnings.warn(f"{self._code}: could not smooth the {table.cal_type} solutions in time "
                          f"({exc}); applying them unsmoothed")
            return table

    def scan_snr(self, *, field: str = "", force: bool = False, **kwargs):
        """Measure fringe SNR per scan, antenna and polarization on the calibrators.

        A diagnostic fringe fit over the central channels of every calibrator
        scan (``[calibration].snr_channel_fraction``, default 0.8). The result is
        stored on ``obs.metadata.snr_survey`` for scan and reference-antenna
        selection, and is what ``obs.plot.scan_snr()`` renders.

        Parameters
        ----------
        field : str
            Source selection; defaults to every calibrator (fringe finders and
            phase calibrators), falling back to the targets when none is defined.
        force : bool
            Re-run even if the survey already exists.

        Returns
        -------
        ScanSNRSurvey
        """
        obs = self._obs
        cal_cfg = obs.config.get("calibration", {})
        calibrators = list(obs.sources.calibrators or obs.sources.targets)
        if obs.phaseref_settings.inverse:
            # Inverse phase referencing solves on the targets: their brightness decides the strategy.
            calibrators += [s for s in obs.sources.targets if s not in calibrators]
        selection = field or ",".join(s.name for s in calibrators)
        cached = obs._snr_surveys.get(selection)
        if cached is not None and not force:
            logger.info("scan_snr: reusing the survey of {}", selection)
            return cached
        wider = self._wider_survey(selection) if not force else None
        if wider is not None:
            return wider
        survey_cfg = {"channel_fraction": cal_cfg.get("snr_channel_fraction", 0.8),
                      "max_scans": cal_cfg.get("snr_max_scans", 24),
                      "callib": cal_cfg.get("callib", False)}
        survey_cfg.update(cal_cfg.get("snr_survey", {}))
        survey_cfg.update(kwargs)
        # A resumed run has the survey table on disk but not in memory: read it back
        # rather than repeat the fringe fit.
        if not force and not field and obs._state.status("scan_snr") == "done":
            survey_cfg.setdefault("reuse", True)
        survey = self._backend.calibrate.scan_snr(
            self._code, selection, refant=obs.refant, metadata=obs.metadata,
            gaintable=list(obs.gaintables), **survey_cfg)
        obs._snr_surveys[selection] = survey
        if obs._metadata is not None:
            obs._metadata.snr_survey = survey
        for antenna in survey.dead_antennas():
            warnings.warn(f"{self._code}: antenna {antenna} shows no usable fringes on {selection}")
        obs._state.mark_complete("scan_snr", outputs=["snr"])
        return survey

    def _wider_survey(self, selection: str):
        """Return a cached survey restricted to ``selection``, or None when none covers it.

        Role inference surveys every source; the instrumental selection then asks
        for one group of them. The wider survey already holds those rows, so it is
        sliced rather than solved again — one fringe fit per scan is the dominant
        cost of inspection.

        Parameters
        ----------
        selection : str
            Comma-separated source names being asked for.

        A wider survey only substitutes when it holds *every* scan the requested
        sources have: the role-ranking survey is sampled per source, and a thin
        sample of a phase calibrator's 31 scans is not the basis the instrumental
        selection should be planned on.

        Returns
        -------
        ScanSNRSurvey or None
        """
        obs = self._obs
        wanted = {name for name in selection.split(",") if name}
        if not wanted or obs.metadata is None:
            return None
        expected = {s.scan_number for s in obs.metadata.scans if s.source in wanted}
        if not expected:
            return None
        for cached_selection, survey in obs._snr_surveys.items():
            have = {name for name in cached_selection.split(",") if name}
            if not wanted <= have:
                continue
            restricted = survey.restricted_to(wanted)
            # The survey has a row for every scan of its sources, but a sampled survey only
            # fringe-fitted some of them: the others are all-nan and were never looked at.
            solved = set(restricted.solved_scans())
            if not expected <= solved:
                logger.info("scan_snr: the survey of {} solved only {} of the {} scan(s) on {}; "
                            "solving this group on its own", cached_selection,
                            len(solved & expected), len(expected), selection)
                continue
            logger.info("scan_snr: taking {} from the wider survey of {} ({} scan(s)); not "
                        "fringe-fitting again", selection, cached_selection, len(restricted.scan_numbers))
            obs._snr_surveys[selection] = restricted
            return restricted
        return None

    def apply(self, *, force: bool = False, field: str = "", defer: bool = False) -> None:
        """Apply the accumulated calibration tables to every source (or one ``field``).

        ``defer=True`` lets a backend that can (dask-ms) record the application without writing
        CORRECTED_DATA: the calibration flags are written, and the readers of the corrected data apply the
        tables while they read DATA (see :mod:`vlbipy.solvers.corrected`); :meth:`write_corrected` writes the
        column. Other backends apply as usual.
        """
        if not self._obs._state.should_run("apply", force=force):
            return
        # Empty selection = every observed source field: the targets need the
        # calibration too, and anything left uncorrected would silently be imaged
        # from raw data.  Per-source calls let the field mapping resolve nearest for
        # calibrators and the phase calibrator for targets.
        cal_cfg = self._obs.config.get("calibration", {})
        options = {"defer": True} if defer and getattr(self._backend.calibrate, "defers_apply", False) else {}
        self._backend.calibrate.apply(
            self._code, field, self.transfer_chain(),
            callib=cal_cfg.get("callib", False), **options)
        self._obs._state.mark_complete("apply")

    def transfer_chain(self) -> list[CalTable]:
        """The apply chain with the phase-reference transfer resolved (:func:`vlbipy.phaseref.transfer_chain`).

        With the ``interpolate`` strategy every receiver (target, check source; the phase calibrators with
        inverse phase referencing) takes tables interpolated between the two reference sources nearest to it,
        rebuilt whenever a table they come from changed; otherwise this is the chain as solved.
        """
        obs = self._obs
        tables = list(obs.gaintables)
        if not self._backend.requires_data_files:
            return tables
        plan = obs.phaseref_plan()
        if plan.strategy != "interpolate":
            return tables
        return phaseref.transfer_chain(tables, plan, obs.metadata, Path(obs.work_dir) / "caltables" / "phaseref")

    def write_corrected(self) -> None:
        """Write CORRECTED_DATA where an application was deferred (no-op when it is up to date, or on a
        backend that does not defer)."""
        ensure = getattr(self._backend.calibrate, "ensure_corrected", None)
        if ensure is not None:
            ensure(self._code)


class FlagNamespace(Namespace):
    """Flagging operations. Auto-flaggers run on calibrators only, never targets."""

    def __call__(self, *, force: bool = False) -> None:
        """Run the pre-calibration flag chain: a-priori flags + autocorr, quack, initial auto-flag."""
        self.apriori(force=force)
        self.quack(force=force)
        self.initial(force=force)

    def _run(self, kind: str, *, field: str = "", force: bool = False, **kwargs) -> float:
        if not self._obs._state.should_run(f"flag_{kind}", force=force):
            return 0.0
        frac = self._backend.flag.run(self._code, kind, field=field, **kwargs)
        if frac >= _HIGH_FLAG_FRACTION:
            warnings.anomaly(f"{self._code}: flag[{kind}] removed {frac:.1%} of data")
        self._obs._state.mark_complete(f"flag_{kind}")
        return frac

    def apriori(self, *, force: bool = False) -> float:
        """Apply the observatory's a-priori flags plus the autocorrelations.

        The a-priori flag table (``.uvflg`` for the EVN/LBA) marks data the
        station or correlator already knows to be bad — off-source slews,
        receiver problems, known RFI. VLBA data carries those flags inside the
        FITS-IDI, so ``importfitsidi`` has already applied them and only the
        autocorrelations remain.
        """
        obs = self._obs
        handler = obs._observatory_handler
        flagfile = self._ensure_flag_file()
        flagged = 0.0
        if flagfile:
            logger.info("flag.apriori: applying {}", Path(flagfile).name)
            flagged += self.from_file(str(flagfile), force=force)
        else:
            warnings.anomaly(
                f"{self._code}: no a-priori flag file for {handler.name}; off-source and slewing "
                "data will stay in the dataset (visible as low amplitudes at scan starts)")
        if self._accor_pending():
            logger.info("flag.apriori: the autocorrelations are kept for accor; calibrate.a_priori flags them")
            return flagged
        return flagged + self.autocorr(force=force)

    def _accor_pending(self) -> bool:
        """True when accor still has to run on this observation (it needs the autocorrelations unflagged)."""
        obs = self._obs
        enabled = obs.config.get("calibration", {}).get("accor", {}).get("enabled", True)
        return bool(obs._observatory_handler.needs_accor and enabled
                    and self._backend.requires_data_files and obs._state.status("a_priori") != "done")

    def _ensure_flag_file(self) -> Optional[str]:
        """Return the CASA-format a-priori flag file, fetching and converting it if needed.

        The ``.uvflg`` lives in the archive's pipeline area rather than beside the
        FITS-IDI files, so a project imported from a local copy usually has no
        flag table at all. Fetch it on demand, and convert the AIPS-format table
        to CASA flag commands, rather than silently proceeding without it.
        """
        obs = self._obs
        handler = obs._observatory_handler
        if not self._backend.requires_data_files:
            return None      # no real data on disk to flag, and nothing to fetch against
        existing = handler.get_flag_file(self._code, obs.work_dir)
        if existing and existing.endswith(".flag"):
            return existing
        if not existing:
            imp_cfg = obs.config.get("import", {})
            try:
                handler.fetch_apriori_files(self._code, obs.work_dir,
                                            obsdate=imp_cfg.get("obsdate") or None,
                                            username=imp_cfg.get("username") or None,
                                            password=imp_cfg.get("password") or None)
            except Exception as exc:  # noqa: BLE001 - offline or proprietary: not fatal
                warnings.warn(f"{self._code}: could not fetch the a-priori flag file ({exc})")
            existing = handler.get_flag_file(self._code, obs.work_dir)
        if not existing or not existing.endswith(".uvflg"):
            return existing
        # AIPS .uvflg needs converting to CASA flag commands against the FITS-IDI files.
        idi_files = handler.find_data_files(self._code, obs.work_dir)
        if not idi_files:
            # The files may have been imported from elsewhere (import_data(files=...)
            # pointing outside work_dir) — fall back to what import actually read, rather
            # than silently dropping the a-priori flags just because work_dir is empty.
            idi_files = [f for f in obs._state.inputs("import_data") if Path(f).is_file()]
        if not idi_files:
            warnings.warn(f"{self._code}: {Path(existing).name} found but no FITS-IDI files are "
                          "present to convert it against; a-priori flags not applied")
            return None
        handler.prepare_for_import(idi_files, obs.work_dir, project_code=self._code)
        return handler.get_flag_file(self._code, obs.work_dir)

    def autocorr(self, *, force: bool = False) -> float:
        """Flag autocorrelations."""
        return self._run("autocorr", force=force)

    def edges(self, *, force: bool = False, edge_channels: Optional[int] = None,
              table: Optional[CalTable] = None, plot: bool = True, **kwargs) -> dict:
        """Flag the subband edge channels, measuring how many from the bandpass when possible.

        Resolution order: an explicit ``edge_channels=N`` wins; otherwise, when a
        bandpass table exists and the backend can analyse it, the roll-off
        actually present in the data decides (``[flagging].edge_outlier_sigma``,
        ``[flagging].max_edge_fraction``); otherwise the blind
        ``[flagging].edge_channels_fraction`` is used. Every subband gets the
        same trim: they share a signal path, and a ragged per-subband trim would
        leave non-uniform channel coverage.

        Returns
        -------
        dict
            ``n_edge``, ``n_channels``, ``method`` (``explicit`` / ``measured`` /
            ``fraction``), ``flagged_fraction_of_data`` and, when measured, the
            per-channel profiles.
        """
        obs = self._obs
        if not obs._state.should_run("flag_edges", force=force):
            return {}
        cfg = obs.config.get("flagging", {})
        n_channels = int(obs.metadata.freq_setup.n_channels) if obs.metadata else 0
        table = table or obs._table("bpass")
        measurement: dict = {}
        if edge_channels is not None:
            measurement = {"n_edge": int(edge_channels), "n_channels": n_channels, "method": "explicit"}
        elif table is not None and self._backend.supports("flag", "measure_edge_channels"):
            measurement = self._backend.flag.measure_edge_channels(
                self._code, table, threshold=cfg.get("edge_outlier_sigma", 6.0),
                max_edge_fraction=cfg.get("max_edge_fraction", 0.25))
            # The backend may report its own verdict (e.g. "narrowband" when the subbands
            # are too narrow to have an edge); only label it "measured" when it did not.
            measurement.setdefault("method", "measured")
        else:
            fraction = float(cfg.get("edge_channels_fraction", 0.1))
            measurement = {"n_edge": int(round(n_channels * fraction)), "n_channels": n_channels,
                           "method": "fraction", "edge_fraction": fraction}
            logger.info("flag.edges: no bandpass to measure the roll-off from; flagging {:.0%} of "
                        "each subband edge", fraction)
        n_edge = int(measurement.get("n_edge", 0))
        per_antenna = measurement.get("per_antenna") or {}
        if n_edge > 0 or any(left or right for left, right in per_antenna.values()):
            measurement["flagged_fraction_of_data"] = self._backend.flag.run(
                self._code, "edges", edge_channels=n_edge, n_channels=measurement.get("n_channels", 0),
                edge_fraction=measurement.get("edge_fraction", 0.0), per_antenna=per_antenna,
                **kwargs)
            if measurement["flagged_fraction_of_data"] >= _HIGH_FLAG_FRACTION:
                warnings.anomaly(f"{self._code}: flag[edges] removed "
                                 f"{measurement['flagged_fraction_of_data']:.1%} of data")
        else:
            measurement["flagged_fraction_of_data"] = 0.0
            logger.info("flag.edges: the subbands are flat to the edges; nothing to flag")
        # The edge trim is one number per antenna; what the bandpass could not solve (or had to
        # boost several-fold) in a particular subband is flagged where it is.
        if table is not None and self._backend.supports("flag", "bandpass_gaps"):
            gaps = self._backend.flag.bandpass_gaps(self._code, table,
                                                    min_gain=float(cfg.get("bandpass_min_gain", 0.5)))
            measurement["bandpass_gaps"] = {k: v for k, v in gaps.items() if k != "commands"}
        if plot and measurement["method"] == "measured":
            obs.plot.bandpass_profile(measurement)
        obs._state.mark_complete("flag_edges", outputs=[f"edge={n_edge}", measurement["method"]])
        return measurement

    def quack(self, *, force: bool = False, per_antenna: Optional[dict] = None,
              interval: Optional[float] = None, field: str = "", column: str = "data",
              **kwargs) -> float:
        """Flag the slewing/settling time at the start of every scan, per antenna.

        Runs *before* calibration (SKILL step 7) so the instrumental solutions
        are never fitted on off-source data. Resolution order: ``per_antenna``
        (``{"EF": 4, ...}`` seconds, default ``[flagging.quack_antennas]``),
        then a single ``interval`` for the whole array (default
        ``[flagging].quack_interval``), and only when neither is configured is
        the ramp measured from the ``column`` data per antenna over every field
        (``[flagging].quack_sigma``, ``quack_max_seconds``).
        """
        obs = self._obs
        if not obs._state.should_run("flag_quack", force=force):
            return 0.0
        cfg = obs.config.get("flagging", {})
        per_antenna = per_antenna if per_antenna is not None else dict(cfg.get("quack_antennas", {}) or {})
        interval = float(interval if interval is not None else cfg.get("quack_interval", 0) or 0)
        # Antenna slewing depresses every baseline of that antenna regardless of the
        # source being observed, so the ramp is measured on every field: restricting
        # to the phase calibrator leaves too few scans for the per-scan median to
        # detect an antenna that only slews through part of the schedule (RSM07 WB).
        measure_on = list(obs.sources)
        if per_antenna:
            logger.info("flag.quack: configured per-antenna intervals {}",
                        ", ".join(f"{a} {s:g}s" for a, s in sorted(per_antenna.items())))
        elif interval > 0:
            logger.info("flag.quack: configured interval {:g} s for every antenna", interval)
        elif self._backend.supports("flag", "quack"):
            logger.info("flag.quack: no interval configured; measuring the ramp per antenna "
                        "on the {} column", column)
        else:
            logger.info("flag.quack: no interval configured and backend {} cannot measure it; "
                        "nothing to flag", self._backend.kind)
        flagged = self._backend.flag.quack(
            self._code, per_antenna=per_antenna or None, interval=interval,
            field=field or ",".join(s.name for s in measure_on), column=column,
            sigma=cfg.get("quack_sigma", 2.0), max_seconds=cfg.get("quack_max_seconds", 120.0),
            metadata=obs.metadata, **kwargs)
        if flagged >= _HIGH_FLAG_FRACTION:
            warnings.anomaly(f"{self._code}: flag[quack] removed {flagged:.1%} of data")
        obs._state.mark_complete("flag_quack")
        return flagged

    def tfcrop(self, *, force: bool = False, column: str = "data", field: str = "", **kwargs) -> float:
        """Time-frequency auto-flag on calibrators only, judged per baseline.

        ``column`` selects the data the statistics are computed on: ``"data"``
        before calibration, ``"corrected"`` afterwards. Flags are never extended
        across baselines (``[flagging].tfcrop`` may override cutoffs).
        """
        params = dict(self._obs.config.get("flagging", {}).get("tfcrop", {}))
        params.update(kwargs)
        field = field or ",".join(s.name for s in self._obs.sources.calibrators)
        return self._run("tfcrop", field=field, force=force, datacolumn=column, **params)

    def initial(self, *, force: bool = False) -> float:
        """Initial flagging of the calibrators before any solve (SKILL step 8).

        Only strong outliers are meant to go here — spikes, near-zero
        amplitudes, clearly bad scans — so the auto-flagger runs on the raw
        data with the default per-baseline cutoffs. Deep flagging waits until
        the data are calibrated (see :meth:`outliers`).
        """
        return self.tfcrop(force=force, column="data")

    def statistics(self) -> dict:
        """Flagging statistics: total and per antenna / subband, over observable data only.

        Excludes autocorrelations and never-recorded visibilities (a station
        absent from a scan, or a subband it did not observe), so a per-antenna
        fraction reports the data quality rather than the schedule. The result
        is kept on ``obs.flag_statistics`` and included in :meth:`Observation.report`.
        """
        obs = self._obs
        if not self._backend.supports("flag", "summary"):
            logger.info("flag.statistics: backend {} cannot count flags; skipping", self._backend.kind)
            return {}
        report = self._backend.flag.summary(self._code)
        obs.flag_statistics = report
        if self._backend.requires_data_files:
            try:
                path = Path(obs.work_dir) / ".flag_statistics.json"
                path.write_text(json.dumps(report, default=float), encoding="utf-8")
            except Exception:  # noqa: BLE001 - a reporting cache; statistics already succeeded
                logger.debug("flag.statistics: could not persist {}", path)
        worst = sorted(report.get("antenna", {}).items(), key=lambda kv: -kv[1]["fraction"])
        logger.info("flag.statistics: {:.1%} of observable data flagged; per antenna: {}",
                    report.get("fraction", 0.0),
                    ", ".join(f"{a} {v['fraction']:.0%}" for a, v in worst))
        for antenna, values in worst:
            if values["observable"] and values["fraction"] >= 0.9:
                warnings.anomaly(f"{self._code}: antenna {antenna} has {values['fraction']:.0%} of its "
                                 "recorded data flagged")
        return report

    def aoflagger(self, *, force: bool = False) -> float:
        """Run AOFlagger on calibrators only (targets are protected)."""
        field = ",".join(s.name for s in self._obs.sources.calibrators)
        return self._run("aoflagger", field=field, force=force)

    def outliers(self, *, field: str = "", threshold: Optional[float] = None,
                 dry_run: bool = False, force: bool = False, step: str = "flag_outliers",
                 **kwargs) -> dict:
        """Flag points that break their own baseline's smoothness (amplitude-driven).

        Run this after the full calibration is applied: on calibrated data each
        baseline should vary smoothly in time and frequency, so what stands out
        is a defect. Judging every baseline against itself keeps legitimately
        bright short spacings (eMERLIN within the EVN) from being flagged away.

        Parameters
        ----------
        step : str
            State name this run is recorded under, so the pipeline can flag
            outliers more than once (e.g. again after reweighting) and still
            resume correctly.
        """
        obs = self._obs
        if not dry_run and not obs._state.should_run(step, force=force):
            return {}
        cfg = obs.config.get("flagging", {})
        if not cfg.get("flag_outliers", True):
            logger.info("outliers: skipped ([flagging].flag_outliers is false)")
            if not dry_run:
                obs._state.mark_complete(step)
            return {}
        # An antenna still slewing at the start of a scan is not a statistical outlier of one baseline
        # but one antenna pulling all of its baselines down. Found once, on the first pass: the typical
        # times given to the faint fields count from each antenna's first unflagged sample, so a second
        # application would move that sample and flag again.
        if step == "flag_outliers" and not field and cfg.get("flag_off_source", True):
            self.off_source(dry_run=dry_run)
        report = self._backend.flag.outliers(
            self._code, field=field, dry_run=dry_run, metadata=obs.metadata,
            threshold=threshold if threshold is not None else cfg.get("outlier_sigma", 5.0),
            **{"gross_departure": cfg.get("outlier_gross_departure", 0.25),
               "gross_max_fraction": cfg.get("outlier_gross_max_fraction", 0.35), **kwargs})
        share = report.get("flagged_fraction_of_data", 0.0)
        if share >= _HIGH_FLAG_FRACTION:
            warnings.anomaly(f"{self._code}: outlier flagging removed {share:.1%} of the data")
        if not dry_run:
            obs._state.mark_complete(step)
        return report

    def off_source(self, *, dry_run: bool = False, **kwargs) -> dict:
        """Flag antennas while they are not on source, measured on the calibrated data.

        The fringe finders and phase calibrators are measured sample by sample; the
        targets and check sources, too faint for that, get the time each antenna
        typically arrives late (see the backend's ``flag.off_source``). Needs the
        calibration applied, so the pipeline runs it at the start of the first
        outlier pass. A failure is a warning, not a failed run.

        Returns
        -------
        dict
            The backend report (``commands``, ``per_antenna``, ``typical``, ...), empty when skipped.
        """
        obs = self._obs
        if not self._backend.supports("flag", "off_source"):
            return {}
        cfg = obs.config.get("flagging", {})
        with_data = list(obs.metadata.source_names) if obs.metadata else list(obs.sources.names)
        bright = [s.name for s in list(obs.sources.fringe_finders) + list(obs.sources.phase_calibrators)
                  if s.name in with_data]
        bright = list(dict.fromkeys(bright))
        if not bright:
            logger.info("off source: no fringe finder or phase calibrator with data to measure on; skipped")
            return {}
        try:
            return self._backend.flag.off_source(
                self._code, fields=bright, transfer_fields=[n for n in with_data if n not in bright],
                level=float(cfg.get("off_source_level", 0.8)), dry_run=dry_run, metadata=obs.metadata, **kwargs)
        except BackendError as exc:
            warnings.warn(f"{self._code}: off-source flagging failed ({exc}); continuing without it")
            return {}

    def from_file(self, path: str, *, force: bool = False) -> float:
        """Apply flags from an external flag command file."""
        return self._run("from_file", force=force, flagfile=path)

    def from_split(self, source: TargetLike = None, *, flag_backup: bool = True) -> int:
        """Carry flags edited in a per-source split (e.g. by difmapy) to the parent measurement set.

        Compares ``calibrated_data/<code>_<source>.ms`` with the FLAG snapshot taken
        when it was split and applies the newly flagged rows as ``flagdata``
        commands (baseline, subband, time range). Returns the number of commands.
        """
        from ..interactive import flag_commands_from_split, run_flag_commands
        obs = self._obs
        src = self._resolve_source_name(source)
        split = Path(obs.work_dir) / "calibrated_data" / f"{self._code}_{src}.ms"
        if not split.is_dir():
            raise StepError("flag_from_split", f"{self._code}: no split measurement set for {src} at {split}")
        commands = flag_commands_from_split(str(split), field=src)
        if commands:
            run_flag_commands(str(self._backend.ms_path(self._code)), commands, flag_backup=flag_backup)
            from ..interactive import snapshot_flags
            snapshot_flags(str(split))        # the split now matches the parent again
        logger.info("flag.from_split[{}]: {} command(s) applied to the parent measurement set", src, len(commands))
        return len(commands)

    def manual(self, *, force: bool = False, **selection) -> float:
        """Apply a manual flag selection (e.g. antenna/spw/timerange)."""
        return self._run("manual", force=force, **selection)

    def weak_polarizations(self, *, force: bool = False) -> dict[str, str]:
        """Flag the receptor of an antenna that shows no signal while its other one does (before any solve).

        Decided on the fringe-SNR survey of the calibrators (:meth:`ScanSNRSurvey.dead_hands`). A receptor
        that is detected but weak is caught later, on amplitudes, when the scalar bandpass has solved both
        hands of every antenna (:meth:`weak_polarizations_from_gains`). ``[flagging].weak_polarization_ratio
        = 0`` turns both off.

        Returns
        -------
        dict
            ``{antenna: flagged hand}``.
        """
        obs = self._obs
        step = "flag_weak_polarization"
        if not obs._state.should_run(step, force=force):
            return {}
        if float(obs.config.get("flagging", {}).get("weak_polarization_ratio", 5.0) or 0.0) <= 0 or \
                not self._backend.supports("calibrate", "scan_snr"):
            obs._state.mark_complete(step, outputs=[])
            return {}
        detection = float(obs.config.get("calibration", {}).get("detection_snr", 7.0))
        dead = obs.calibrate.scan_snr().dead_hands(detection=detection)
        flagged = {}
        for antenna, (hand, weak_snr, strong_snr) in dead.items():
            flagged[antenna] = self._flag_hand(antenna, hand, f"has no fringes (median SNR {weak_snr:.1f} on the "
                                                              f"calibrator scans where the other hand has "
                                                              f"{strong_snr:.0f})")
        if not flagged:
            logger.info("flag[weak_polarization]: both hands of every antenna have fringes")
        obs._state.mark_complete(step, outputs=[f"{a}:{h}" for a, h in flagged.items()])
        return flagged

    def weak_polarizations_from_gains(self, levels: dict) -> dict[str, str]:
        """Flag the receptor of an antenna whose amplitude is much below that of its other one.

        ``levels`` compares the two hands of every antenna in the scalar-bandpass solutions, before they are
        normalised (:meth:`CasaCalibrationOps._hand_levels`): one amplitude per hand, subband and calibrator.
        A hand ``[flagging].weak_polarization_ratio`` (5) times weaker in the median over the solutions both
        have, or solved nowhere while the other is solved in three or more, loses its parallel hand and both
        cross hands. Left in, a Stokes I calibration averages its near-zero signal with the good hand:
        V589A's CD, with an LCP 11 times below its RCP, had every baseline at ~0.55 of its RR.

        Returns
        -------
        dict
            ``{antenna: flagged hand}`` (only the ones flagged now).
        """
        ratio = float(self._obs.config.get("flagging", {}).get("weak_polarization_ratio", 5.0) or 0.0)
        flagged = {}
        if ratio <= 0:
            return flagged
        for antenna, level in (levels or {}).items():
            solved, value = level.get("solved", {}), level.get("ratio")
            if solved.get("LL", 0) == 0 and solved.get("RR", 0) >= 3:
                hand, how = "LL", "has no amplitude solution in any subband while its RCP has"
            elif solved.get("RR", 0) == 0 and solved.get("LL", 0) >= 3:
                hand, how = "RR", "has no amplitude solution in any subband while its LCP has"
            elif value and level.get("pairs", 0) >= 3 and (value >= ratio or value <= 1.0 / ratio):
                hand = "LL" if value >= ratio else "RR"
                how = f"is {max(value, 1.0 / value):.1f} times weaker than its {'RCP' if hand == 'LL' else 'LCP'} " \
                      "in amplitude (scalar bandpass)"
            else:
                continue
            flagged[antenna] = self._flag_hand(antenna, hand, how)
        return flagged

    def _flag_hand(self, antenna: str, hand: str, why: str) -> str:
        """Flag one receptor of ``antenna``: its parallel hand and both cross hands, on every baseline."""
        correlations = ",".join([hand, "RL", "LR"])
        self._backend.flag.run(self._code, "manual", antenna=antenna, correlation=correlations)
        done = self._obs.__dict__.setdefault("_flagged_hands", set())
        if (antenna, hand) not in done:
            done.add((antenna, hand))
            warnings.anomaly(f"{self._code}: {antenna} {'LCP' if hand == 'LL' else 'RCP'} {why}; flagged its "
                             f"{correlations} data")
        return hand


class PlotNamespace(Namespace):
    """Diagnostic plotting."""

    #: Maximum number of phase-calibrator scans the per-scan diagnostics fall back to.
    MAX_DIAGNOSTIC_SCANS = 5

    def __call__(self, *, column: str = "corrected", label: str = "") -> list[str]:
        """Produce the standard diagnostic set (see :meth:`diagnostics`)."""
        return self.diagnostics(column=column, label=label)

    def _run_jobs(self, jobs: list[PlotJob], label: str) -> list[str]:
        """Draw a plot set several plots at a time, warning on failures instead of aborting.

        Backend plots run in worker processes, ``call`` jobs here meanwhile
        (:func:`vlbipy.plot_jobs.run_plot_jobs`): at most ``[plotting].workers`` and the CPU count,
        and only as many as fit together in ``[plotting].memory_fraction`` of the free memory.
        Returns the paths written, in job order.
        """
        cfg = self._obs.config.get("plotting", {})
        return run_plot_jobs(self._backend, self._code, jobs, max_workers=plot_workers(cfg.get("workers", "auto")),
                             memory_fraction=float(cfg.get("memory_fraction", 0.8)), label=label)

    def _draw(self, job: PlotJob):
        """Draw one plot job in this process, now (the single-plot methods: a failure raises)."""
        return draw(self._backend, self._code, job)

    def diagnostics(self, *, column: str = "corrected", label: str = "",
                    force: bool = False) -> list[str]:
        """The standard diagnostic set on one data column (SKILL steps 7 and 15).

        On the raw data (``column="data"``) this shows which antennas actually
        observed, where signal exists and what needs flagging; on the calibrated
        data (``"corrected"``) the same plots must show flat phases near zero
        and stable amplitudes on the calibrators. The set: scan x antenna fringe
        SNR (tplot), per-scan autocorrelation and cross-correlation spectra on
        the fringe finders (raw only), amplitude/phase vs frequency and vs time
        on baselines to the reference antenna, per-baseline corner plots,
        amplitude/phase vs uv distance, the uv coverage (raw only; it does
        not change) and the time evolution of every baseline to the reference
        antenna over the whole observation (raw only, :meth:`time_evolution`).

        Plotting is reporting: a plot that fails is logged as a warning and the
        rest of the set (and the pipeline) carries on. ``[plotting].workers``
        plots are drawn at a time, each in a worker process (see :meth:`_run_jobs`).

        The raw set is recorded as the ``plot_raw`` step: it reads the whole
        dataset and does not change once the a-priori flags are in, so a resumed
        run keeps the plots already on disk (``force=True`` redraws them).

        Returns
        -------
        list of str
            Paths of the PNG files written.
        """
        obs = self._obs
        label = label or ("raw" if column == "data" else "calibrated")
        if column == "data" and label == "raw" and not obs._state.should_run("plot_raw", force=force):
            self._obs.calibrate.scan_snr()     # the survey still drives the calibration selection
            # A project whose raw set was drawn before the time-evolution figures existed gets them now.
            if self._lacks_time_evolution():
                job = PlotJob("time_evolution", call=lambda: self.time_evolution(column=column, label=label))
                self._run_jobs([job], label)
            return sorted(str(p) for p in self._backend.plot.plot_dir("raw").glob("*.png"))
        calibrators = obs.sources.calibrators or obs.sources.targets
        # Raw diagnostics trigger the calibrator-only survey used for calibration
        # selection. Corrected diagnostics only render the comprehensive survey that
        # the orchestrator measured after the final applycal. It goes first: the
        # survey it may measure picks the reference antenna of each per-scan plot.
        written: list[str] = []
        if column == "data" or (obs.metadata and obs.metadata.snr_survey is not None):
            written += self._run_jobs([PlotJob("scan_snr", call=lambda: self.scan_snr())], label)
        jobs: list[PlotJob] = []
        if column == "data" and self._backend.supports("plot", "diagnostic"):
            jobs.append(self._uv_coverage_job("uv_coverage"))
        if self._use_jplotter():
            # Per-scan spectra, per-field spectra and time series and the whole-track time evolution in one
            # batch of parallel jplotter sessions (seconds per figure instead of minutes).
            jobs.append(PlotJob("jplotter", call=lambda: self.jplotter_set(
                column=column, label=label, fields=[s.name for s in calibrators], per_scan=column == "data",
                time_evolution=column == "data" and self._backend.supports("plot", "time_evolution"))))
        else:
            if column == "data":
                jobs += self._scan_diagnostic_jobs(column=column, label=label)
                if self._backend.supports("plot", "time_evolution"):
                    jobs.append(PlotJob("time_evolution", call=lambda: self.time_evolution(column=column,
                                                                                           label=label)))
            jobs += [self._spectrum_job(f"spectrum[{s.name}]", field=s.name, column=column,
                                        label=f"{label}_{s.name}") for s in calibrators]
            jobs += [self._timeseries_job(f"timeseries[{s.name}]", field=s.name, column=column,
                                          label=f"{label}_{s.name}") for s in calibrators]
        jobs += [self._corners_job(f"corners[{s.name}]", field=s.name, column=column, label=f"{label}_{s.name}")
                 for s in calibrators]
        jobs += self._radplot_jobs(column=column, label=label)
        if column == "corrected":
            jobs.append(self._subband_phases_job("subband_phases", column=column, label=label))
        written += self._run_jobs(jobs, label)
        logger.info("plot.diagnostics[{}]: {} plot(s) on the {} column", label, len(written), column)
        if column == "data" and label == "raw":
            obs._state.mark_complete("plot_raw")
        return written

    def final_data(self) -> list[str]:
        """The final-data set on the calibrated column: every source, labelled ``final``.

        uv coverage of all sources in one figure plus a single-panel copy per
        source, amplitude/phase vs uv distance with the model overlaid (when a
        MODEL column exists), spectra and per-baseline light curves for every
        source, and the total coherent amplitude light curve. Failing plots warn
        and the set carries on.
        """
        names = self._all_source_names()
        jobs: list[PlotJob] = []
        if self._backend.supports("plot", "diagnostic"):
            jobs.append(self._uv_coverage_job("uv_coverage"))
        jobs += self._radplot_jobs(all_sources=True, column="corrected", label="final", with_model=True)
        jobs += [self._spectrum_job(f"spectrum[{name}]", field=name, column="corrected", label=f"final_{name}")
                 for name in names]
        jobs += [self._timeseries_job(f"timeseries[{name}]", field=name, column="corrected", label=f"final_{name}")
                 for name in names]
        jobs.append(self._lightcurve_job("lightcurve", column="corrected", label="final"))
        written = self._run_jobs(jobs, "final")
        logger.info("plot.final_data: {} plot(s) over {} source(s)", len(written), len(names))
        return written

    def _all_source_names(self) -> list[str]:
        """Every source with data: the metadata's source list, else the configured sources."""
        obs = self._obs
        return list(obs.metadata.source_names) if obs.metadata else list(obs.sources.names)

    def _plot(self, kind: str, **kwargs) -> str:
        return self._backend.plot.diagnostic(self._code, kind, **kwargs)

    def tplot(self) -> str:
        """Antenna-participation timeline."""
        return self._plot("tplot")

    def uv_coverage(self) -> str:
        """UV coverage of every source with data, one panel per source in a single figure."""
        return self._draw(self._uv_coverage_job("uv_coverage"))

    def _uv_coverage_job(self, name: str) -> PlotJob:
        """The :meth:`uv_coverage` plot as a job named ``name``."""
        return PlotJob(name, method="diagnostic", kwargs={"kind": "uv_coverage", "metadata": self._obs.metadata})

    def elevation(self) -> str:
        """Source elevation vs time."""
        return self._plot("elevation")

    def amp_vs_time(self) -> str:
        """Amplitude vs time."""
        return self._plot("amp_vs_time")

    def phase_vs_time(self) -> str:
        """Phase vs time."""
        return self._plot("phase_vs_time")

    def autocorr(self, *, field: str = "", scans: Optional[list] = None, column: str = "data",
                 label: str = "", **kwargs) -> str:
        """Autocorrelation amplitude spectra, one panel per antenna, for a field / scan selection."""
        return self._draw(self._autocorr_job("autocorr", field=field, scans=scans, column=column, label=label,
                                             **kwargs))

    def _autocorr_job(self, name: str, *, field: str = "", scans: Optional[list] = None, column: str = "data",
                      label: str = "", **kwargs) -> PlotJob:
        """The :meth:`autocorr` plot as a job named ``name``."""
        obs = self._obs
        return PlotJob(name, method="autocorr", kwargs=dict(field=field or obs.calibrator_field, scans=scans,
                                                            column=column, label=label, metadata=obs.metadata,
                                                            **kwargs))

    def crosscorr(self) -> str:
        """Cross-correlation spectra."""
        return self._plot("crosscorr")

    def scan_snr(self, **kwargs) -> list[str]:
        """Plot the scan x antenna fringe-SNR matrix, one PNG per polarization.

        Uses the survey on ``obs.metadata.snr_survey`` when present, otherwise
        runs ``obs.calibrate.scan_snr()`` first.
        """
        obs = self._obs
        survey = obs.metadata.snr_survey if obs.metadata else None
        if survey is None:
            survey = obs.calibrate.scan_snr()
        return self._backend.plot.scan_snr(self._code, survey, **kwargs)

    def snr_for_scans(self) -> dict[int, dict[str, Optional[float]]]:
        """Return ``{scan_number: {antenna: fringe SNR}}`` from the SNR survey (``None`` where absent).

        The SNR is the median over polarizations; an empty dict when no survey
        has been measured yet.
        """
        survey = self._obs.metadata.snr_survey if self._obs.metadata else None
        return survey.per_scan_antenna() if survey is not None else {}

    def spectrum(self, *, field: str = "", scans: Optional[list] = None, column: str = "corrected",
                 label: str = "", all_pols: bool = False, refant: str = "", **kwargs) -> list[str]:
        """Plot amplitude/phase vs channel of the calibrated data, per baseline to the refant.

        Defaults to the scan the instrumental calibration was solved on, since
        that is where the response should be flattest — but the calibration has
        been applied to every source, so any field or scan can be inspected.
        ``refant`` overrides the observation's reference antenna (a chain is
        reduced to its first member).
        """
        return self._draw(self._spectrum_job("spectrum", field=field, scans=scans, column=column, label=label,
                                             all_pols=all_pols, refant=refant, **kwargs))

    def _spectrum_job(self, name: str, *, field: str = "", scans: Optional[list] = None, column: str = "corrected",
                      label: str = "", all_pols: bool = False, refant: str = "", **kwargs) -> PlotJob:
        """The :meth:`spectrum` plot as a job named ``name``."""
        obs = self._obs
        return PlotJob(name, method="spectrum", kwargs=dict(
            field=field or obs.calibrator_field, scans=scans, refant=(refant or str(obs.refant)).split(",")[0],
            column=column, label=label, all_pols=all_pols, metadata=obs.metadata, **kwargs))

    def _refant_for_scan(self, scan) -> str:
        """Return a reference antenna that is present in ``scan``.

        The observation's refant (first member of a chain that observed the
        scan) when possible, else the highest-ranked antenna of the SNR survey
        present in the scan, else the scan's first antenna.
        """
        obs = self._obs
        present = list(scan.antennas)
        for name in str(obs.refant).split(","):
            if name in present:
                return name
        survey = obs.metadata.snr_survey if obs.metadata else None
        if survey is not None:
            for name, _ in survey.rank_antennas():
                if name in present:
                    return name
        return present[0] if present else str(obs.refant).split(",")[0]

    def _diagnostic_scans(self) -> list:
        """Pick the scans the per-scan diagnostics are made on.

        Every fringe-finder scan when there are fringe finders; otherwise up to
        :attr:`MAX_DIAGNOSTIC_SCANS` phase-calibrator scans spread evenly in
        time, then swapped/extended greedily so every observed antenna appears
        in at least one chosen scan when the schedule allows it.
        """
        obs = self._obs
        scans = sorted(obs.metadata.scans if obs.metadata else [], key=lambda s: s.time_start)
        finders = {s.name for s in obs.sources.fringe_finders}
        if finders:
            return [s for s in scans if s.source in finders]
        cals = {s.name for s in obs.sources.phase_calibrators} or {s.name for s in obs.sources.calibrators}
        candidates = [s for s in scans if s.source in cals] or scans
        cap = self.MAX_DIAGNOSTIC_SCANS
        if len(candidates) <= cap:
            return candidates
        picks = [candidates[round(i * (len(candidates) - 1) / (cap - 1))] for i in range(cap)]
        chosen = list({s.scan_number: s for s in picks}.values())
        observed = {a.name for a in obs.metadata.observed_antennas} if obs.metadata else set()
        missing = observed - {a for s in chosen for a in s.antennas}
        while missing:
            best = max((s for s in candidates if s not in chosen), key=lambda s: len(missing & set(s.antennas)),
                       default=None)
            if best is None or not missing & set(best.antennas):
                break
            if len(chosen) >= cap:
                # Drop the chosen scan whose antennas are all covered by the others.
                redundant = [s for s in chosen if not (set(s.antennas) - {a for o in chosen if o is not s
                                                                            for a in o.antennas})]
                if not redundant:
                    break
                chosen.remove(redundant[0])
            chosen.append(best)
            missing -= set(best.antennas)
        return sorted(chosen, key=lambda s: s.time_start)

    def scan_diagnostics(self, *, column: str = "data", label: str = "raw") -> list[str]:
        """Autocorrelation and cross-correlation spectra of individual scans.

        On each diagnostic scan (see :meth:`_diagnostic_scans`) the autocorrelations
        of every antenna and the cross-correlation spectra to a reference antenna
        present in that scan. On the raw column the cross-correlations are shown
        in full Stokes: before calibration the cross-hands carry the
        instrumental polarization signature, afterwards they are only noise.
        """
        return self._run_jobs(self._scan_diagnostic_jobs(column=column, label=label), label)

    def _scan_diagnostic_jobs(self, *, column: str, label: str) -> list[PlotJob]:
        """The plots of :meth:`scan_diagnostics`: autocorrelations, then cross-correlation spectra, per scan."""
        scans = self._diagnostic_scans()
        logger.info("scan_diagnostics: {} scan(s) ({})", len(scans),
                    ", ".join(str(s.scan_number) for s in scans) or "none")
        jobs: list[PlotJob] = []
        for scan in scans:
            tag = f"{label}_scan{scan.scan_number}"
            jobs.append(self._autocorr_job(f"autocorr[{scan.scan_number}]", field=scan.source,
                                           scans=[scan.scan_number], column=column, label=tag))
            jobs.append(self._spectrum_job(f"spectrum[{scan.scan_number}]", field=scan.source,
                                           scans=[scan.scan_number], column=column, all_pols=(column == "data"),
                                           label=tag, refant=self._refant_for_scan(scan)))
        return jobs

    def raw_stokes(self, **kwargs) -> list[str]:
        """Backwards-compatible alias of :meth:`scan_diagnostics` on the raw data."""
        return self.scan_diagnostics(column="data", label=kwargs.get("label", "raw"))

    def radplot(self, *, sources: Optional[list] = None, all_sources: bool = False, column: str = "corrected",
                time_bin: Optional[float] = None, label: str = "", with_model: bool = False,
                **kwargs) -> list[str]:
        """Amplitude and phase vs uv distance, one plot per source.

        Defaults to every calibrator: these are the sources whose structure the
        calibration depends on, so a resolved one showing a falling amplitude
        profile is something to know about before trusting its solutions.
        ``all_sources`` plots every source with data instead; ``with_model``
        overlays the MODEL column (calibrated data only, skipped when absent).
        """
        jobs = self._radplot_jobs(sources=sources, all_sources=all_sources, column=column, time_bin=time_bin,
                                  label=label, with_model=with_model, **kwargs)
        written = [self._draw(job) for job in jobs]
        logger.info("radplot: {} plot(s) for {}", len(written), ", ".join(job.kwargs["field"] for job in jobs))
        return written

    def _radplot_jobs(self, *, sources: Optional[list] = None, all_sources: bool = False, column: str = "corrected",
                      time_bin: Optional[float] = None, label: str = "", with_model: bool = False,
                      **kwargs) -> list[PlotJob]:
        """The plots of :meth:`radplot` (same arguments), one job per source, named ``radplot[<source>]``."""
        obs = self._obs
        if all_sources:
            names = self._all_source_names()
        else:
            names = list(sources or [s.name for s in obs.sources.calibrators]
                         or [s.name for s in obs.sources.targets])
        if time_bin is None:
            time_bin = obs.config.get("export", {}).get("radplot_time_bin", 10.0)
        return [PlotJob(f"radplot[{name}]", method="radplot", kwargs=dict(
                    field=name, column=column, label=label, time_bin=time_bin,
                    with_model=with_model and column == "corrected", metadata=obs.metadata, **kwargs))
                for name in names]

    def lightcurve(self, *, column: str = "corrected", label: str = "calibrated",
                   averaging_sec: Optional[list] = None, **kwargs) -> str:
        """Total coherent visibility amplitude vs time for every source, at several averaging scales.

        ``averaging_sec`` defaults to ``[export].lightcurve_averaging`` (``0`` =
        native integrations, positive = seconds within a scan, ``-1`` = per scan).
        """
        return self._draw(self._lightcurve_job("lightcurve", column=column, label=label, averaging_sec=averaging_sec,
                                               **kwargs))

    def _lightcurve_job(self, name: str, *, column: str = "corrected", label: str = "calibrated",
                        averaging_sec: Optional[list] = None, **kwargs) -> PlotJob:
        """The :meth:`lightcurve` plot as a job named ``name``."""
        obs = self._obs
        scales = averaging_sec or obs.config.get("export", {}).get("lightcurve_averaging", [0, 30, 120, -1])
        return PlotJob(name, method="lightcurve", kwargs=dict(fields=self._all_source_names(), column=column,
                                                              label=label, averaging_sec=list(scales),
                                                              metadata=obs.metadata, **kwargs))

    def subband_phases(self, *, column: str = "corrected", label: str = "calibrated", **kwargs) -> str:
        """Residual phase jumps between subbands, per calibrator scan and baseline to the refant.

        The verification of the single-band delay: on calibrated data the phase
        of every subband relative to the first should sit at zero on every scan.
        Goes through every scan on every calibrator (fringe finders and phase
        calibrators).
        """
        return self._draw(self._subband_phases_job("subband_phases", column=column, label=label, **kwargs))

    def _subband_phases_job(self, name: str, *, column: str = "corrected", label: str = "calibrated",
                            **kwargs) -> PlotJob:
        """The :meth:`subband_phases` plot as a job named ``name``."""
        obs = self._obs
        fields = [s.name for s in obs.sources.calibrators] or [s.name for s in obs.sources.targets]
        return PlotJob(name, method="subband_phases", kwargs=dict(
            fields=fields, refant=str(obs.refant).split(",")[0], column=column, label=label,
            metadata=obs.metadata, **kwargs))

    def timeseries(self, *, field: str = "", scans: Optional[list] = None,
                   column: str = "corrected", label: str = "", refant: str = "",
                   **kwargs) -> list[str]:
        """Amplitude and phase vs time per baseline for an optional exact scan selection."""
        return self._draw(self._timeseries_job("timeseries", field=field, scans=scans, column=column, label=label,
                                               refant=refant, **kwargs))

    def _timeseries_job(self, name: str, *, field: str = "", scans: Optional[list] = None, column: str = "corrected",
                        label: str = "", refant: str = "", **kwargs) -> PlotJob:
        """The :meth:`timeseries` plot as a job named ``name``."""
        obs = self._obs
        return PlotJob(name, method="timeseries", kwargs=dict(
            field=field or obs.calibrator_field, scans=scans, refant=(refant or str(obs.refant)).split(",")[0],
            column=column, label=label, metadata=obs.metadata, **kwargs))

    def time_evolution(self, *, column: str = "data", label: str = "raw") -> list[str]:
        """Amplitude and phase vs time over the whole observation, per baseline to a reference antenna.

        One figure per (reference antenna, subband) pair of :func:`~vlbipy.selection.plan_time_evolution`:
        normally a single one, on the observation's reference antenna in a subband every antenna
        recorded, coloured by source and vector-averaged over the central channels of the subband.
        The data of that first figure then tells where the reference antenna has no unflagged data or
        no fringes (:func:`~vlbipy.selection.reference_antenna_gaps`), which the metadata cannot; when
        it does, the plan is made again with that knowledge and further reference antennas are added,
        highest fringe SNR first. A figure that fails warns and the rest carries on.

        Returns
        -------
        list of str
            Paths of the PNG files written, ``<code>.<label>.time_evolution.ref<ANT>.spw<N>.png``.
        """
        obs = self._obs
        if obs.metadata is None:
            return []
        plan, read = self._time_evolution_plan(column)
        roles = {s.name: s.source_type.value for s in obs.sources}
        jobs = [PlotJob(f"time_evolution[{entry['refant']},spw{entry['subband']}]",
                        call=lambda e=entry: self._time_evolution_figure(e, read.get((e["refant"], e["subband"])),
                                                                         column, label, roles))
                for entry in plan]
        written = self._run_jobs(jobs, label)
        logger.info("plot.time_evolution: {} figure(s) of {} planned ({})", len(written), len(plan),
                    "; ".join(f"ref {e['refant']} subband {e['subband']}: {e['reason']}" for e in plan) or "none")
        return written

    def _time_evolution_plan(self, column: str) -> tuple[list[dict], dict]:
        """The time-evolution figures to make (:func:`~vlbipy.selection.plan_time_evolution`) and the data read.

        The data of the first planned figure tells where its reference antenna has no unflagged data or no
        fringes, which the metadata cannot; the plan is then made again with that knowledge.

        Returns
        -------
        (plan, read)
            The plan entries and ``{(refant, subband): series}`` of what was read on the way.
        """
        from ..selection import plan_time_evolution, reference_antenna_gaps
        obs = self._obs
        metadata = obs.metadata
        refant = str(obs.refant).split(",")[0]
        plan = plan_time_evolution(metadata, metadata.snr_survey, refant)
        read: dict[tuple[str, int], dict] = {}
        if plan and self._backend.supports("data", "read_time_evolution"):
            first = (plan[0]["refant"], plan[0]["subband"])
            read[first] = self._backend.data.read_time_evolution(self._code, refant=first[0], subband=first[1],
                                                                 column=column, metadata=metadata)
            gaps = reference_antenna_gaps(read[first], metadata)
            if gaps["no_data"] or gaps["no_fringes"]:
                plan = plan_time_evolution(metadata, metadata.snr_survey, refant,
                                           without_data={first[0]: gaps["no_data"]},
                                           without_fringes={first[0]: gaps["no_fringes"]})
        return plan, read

    def _use_jplotter(self) -> bool:
        """Whether the standard plots are drawn with jiveplot's jplotter (``[plotting].engine``)."""
        from .. import jplot
        engine = str(self._obs.config.get("plotting", {}).get("engine", "auto")).lower()
        if engine == "matplotlib" or not self._backend.supports("data", "is_imported"):
            return False
        if not self._backend.data.is_imported(self._code):
            return False
        if engine == "jplotter" and not jplot.available():
            warnings.warn(f"{self._code}: [plotting].engine is 'jplotter' but jiveplot cannot be imported; "
                          "drawing the standard plots with matplotlib")
        return jplot.available()

    def _common_subband(self) -> int:
        """The subband recorded by the most antennas (the first such)."""
        metadata = self._obs.metadata
        counts: dict[int, int] = {}
        for antenna in metadata.observed_antennas:
            for subband in antenna.subbands or range(metadata.freq_setup.n_subbands):
                counts[int(subband)] = counts.get(int(subband), 0) + 1
        return min(counts, key=lambda s: (-counts[s], s)) if counts else 0

    def jplotter_set(self, *, column: str = "data", label: str = "raw", fields: Optional[list[str]] = None,
                     per_scan: bool = True, time_evolution: bool = True) -> list[str]:
        """The standard per-scan, per-field and whole-track plots, drawn by jplotter in parallel sessions.

        The same figures and file names as :meth:`scan_diagnostics` (autocorrelation and cross-correlation
        spectra of each diagnostic scan), :meth:`spectrum` and :meth:`timeseries` (per field, to the
        reference antenna) and :meth:`time_evolution` (every baseline to each planned reference antenna over
        the whole observation), in jiveplot's ``standardplots`` style. See :mod:`vlbipy.jplot`.

        Returns
        -------
        list of str
            The PNG pages written.
        """
        from .. import jplot
        obs = self._obs
        plot = self._backend.plot
        folder = plot.plot_dir(plot.category_for_column(column))
        stem = lambda tail: str(folder / f"{self._code}.{tail}")       # noqa: E731
        refant = str(obs.refant).split(",")[0]
        figures = []
        if per_scan:
            for scan in self._diagnostic_scans():
                tag = f"{label}_scan{scan.scan_number}"
                figures.append(jplot.auto_spectrum(stem(f"{tag}.autocorr"), scans=[scan.scan_number]))
                figures.append(jplot.cross_spectrum(stem(f"{tag}.spectrum"), refant=self._refant_for_scan(scan),
                                                    scans=[scan.scan_number], all_pols=(column == "data")))
        subband = self._common_subband()
        for name in fields if fields is not None else [s.name for s in (obs.sources.calibrators
                                                                        or obs.sources.targets)]:
            figures.append(jplot.cross_spectrum(stem(f"{label}_{name}.spectrum"), refant=refant, source=name))
            figures.append(jplot.amp_phase_time(stem(f"{label}_{name}.timeseries"), refant=refant, subband=subband,
                                                source=name, by_source=False))
        if time_evolution and obs.metadata is not None:
            plan, _ = self._time_evolution_plan(column)
            for entry in plan:
                figures.append(jplot.amp_phase_time(
                    stem(f"{label}.time_evolution.ref{entry['refant']}.spw{entry['subband']}"),
                    refant=entry["refant"], subband=entry["subband"]))
        if column == "corrected":
            obs.calibrate.write_corrected()          # jplotter reads the column itself
        result = jplot.draw(self._backend.ms_path(self._code), figures,
                            column={"data": "DATA", "corrected": "CORRECTED_DATA"}[column],
                            log_dir=Path(obs.work_dir) / "logs")
        return result["files"]

    def _time_evolution_figure(self, entry: dict, series: Optional[dict], column: str, label: str,
                               roles: dict) -> list[str]:
        """Make one planned time-evolution figure.

        Parameters
        ----------
        entry : dict
            One item of the plan (``refant``, ``subband``, ``reason``, ``covers``).
        series : dict or None
            Its data when already read; otherwise it is read here (backends that can).
        column, label : str
            Data column and file-name label.
        roles : dict
            ``{source: role}`` for the legend.

        Returns
        -------
        list of str
            The PNG path; empty when a reference antenna added to cover some scans turns out to have no
            unflagged data in any of them (its figure would show nothing the first one does not).
        """
        metadata = self._obs.metadata
        if series is None and self._backend.supports("data", "read_time_evolution"):
            series = self._backend.data.read_time_evolution(self._code, refant=entry["refant"],
                                                            subband=entry["subband"], column=column,
                                                            metadata=metadata)
        if series is not None and entry["covers"] and not set(entry["covers"]) & set(series["scans_with_data"]):
            logger.info("plot.time_evolution: {} has no unflagged data in the scans it was added for; no figure",
                        entry["refant"])
            return []
        return [self._backend.plot.time_evolution(self._code, refant=entry["refant"], subband=entry["subband"],
                                                  column=column, label=label, series=series, metadata=metadata,
                                                  source_roles=roles, reason=entry["reason"])]

    def _lacks_time_evolution(self) -> bool:
        """True when the raw time-evolution figures can be made (data on disk) and none exists yet."""
        backend = self._backend
        if not backend.supports("plot", "time_evolution") or not backend.supports("data", "is_imported"):
            return False
        if not backend.data.is_imported(self._code):
            return False
        return not any(backend.plot.plot_dir("raw").glob("*.time_evolution.*.png"))

    def corner(self, *, field: str = "", quantity: str = "phase", column: str = "corrected",
               label: str = "", **kwargs) -> str:
        """Per-baseline time x frequency grid for one field, coloured by phase or amplitude.

        Defaults to the phase calibrator: that is the source whose calibrated
        phases should be flat everywhere, so any structure left in a cell is a
        calibration problem rather than the sky.
        """
        return self._draw(self._corners_job("corner", field=field, quantities=(quantity,), column=column,
                                            label=label, **kwargs))[0]

    def corners(self, *, field: str = "", column: str = "corrected", label: str = "", **kwargs) -> list[str]:
        """Both corner plots — phase and amplitude (Stokes I) — for one field, from a single read of its data."""
        return self._draw(self._corners_job("corners", field=field, column=column, label=label, **kwargs))

    def _corners_job(self, name: str, *, field: str = "", quantities=("phase", "amp"), column: str = "corrected",
                     label: str = "", **kwargs) -> PlotJob:
        """The corner plots of ``quantities`` (one read of the data for all of them) as a job named ``name``."""
        obs = self._obs
        if not field:
            group = obs.sources.phase_calibrators or obs.sources.calibrators or obs.sources.targets
            field = ",".join(s.name for s in group)
        return PlotJob(name, method="baseline_corners", kwargs=dict(field=field, quantities=list(quantities),
                                                                    column=column, label=label,
                                                                    metadata=obs.metadata, **kwargs))

    def image_grid(self, images: dict) -> list[str]:
        """Render the FITS images of each source side by side (``{source: {robust: fits}}``)."""
        return self._backend.plot.image_grid(self._code, images)

    def bandpass_profile(self, measurement: dict) -> str:
        """Plot the per-channel band profile behind the edge-channel decision."""
        return self._backend.plot.bandpass_profile(self._code, measurement)

    def caltable(self, table: CalTable) -> list[str]:
        """Plot one calibration table to PNG(s): Tsys/gain-curve/fringe/gain layouts.

        The observation's metadata goes along (cached, so it works without the measurement set): it
        gives the bandpass its frequency axis and the gain curves the elevations each antenna observed at.
        """
        return self._backend.plot.caltable(self._code, table.path, cal_type=table.cal_type,
                                           metadata=self._obs.metadata)

    def caltables(self) -> list[str]:
        """Plot every calibration table produced so far.

        Uses the in-memory ``obs.gaintables`` when populated; in a fresh process
        it falls back to the tables found on disk in <work_dir>/caltables.
        """
        tables = list(self._obs.gaintables)
        if not tables:
            caldir = Path(self._obs.work_dir) / "caltables"
            tables = [CalTable(cal_type=p.suffix.lstrip("."), path=str(p))
                      for p in sorted(caldir.glob("*")) if self._is_caltable(p)]
            if tables:
                logger.info("plot.caltables: found {} caltable(s) on disk", len(tables))
        paths: list[str] = []
        for table in tables:
            paths.extend(self.caltable(table))
        return paths

    @staticmethod
    def _is_caltable(path: Path) -> bool:
        """True for a directory holding a calibration table (its ``table.info`` says ``Type = Calibration``).

        ``caltables/`` also holds things that cannot be plotted as one: the copy of the GAIN_CURVE subtable
        saved before a gain-curve refit (``<code>.gain_curve.original``) and JSON records.
        """
        info = path / "table.info"
        return path.is_dir() and info.is_file() and "Type = Calibration" in info.read_text(errors="replace")


class CleanNamespace(Namespace):
    """Imaging. Callable runs the default imager; methods select a specific one."""

    default_imager = "difmap"

    def __call__(self, target: TargetLike = None, *, robust=None, imager: Optional[str] = None,
                 **kwargs) -> Union[Image, ImageSet]:
        """Image a source with the default imager (or ``imager=``); ``robust`` defaults to the config list."""
        default = self._obs.config.get("imaging", {}).get("imager") or self.default_imager
        if not self._backend.requires_data_files and (imager or default) == "difmap":
            default, imager = "wsclean", None       # in-memory backends have no split MS for difmapy
        return self._image(target, robust, imager or default, **kwargs)

    def _backend_imager(self, requested: str) -> str:
        """Return the imager to use: the backend's own (``Backend.imager``) when it has one, else ``requested``.

        The dask-ms backend images everything with difmapy; asking it for tclean or WSClean is not an error
        (configurations and scripts are shared between backends) but it is logged, so the redirect is visible.
        """
        forced = getattr(self._backend, "imager", None)
        if forced and requested != forced:
            logger.info("{}: the {} backend images with {}; ignoring imager={!r}", self._code, self._backend.kind,
                        forced, requested)
        return forced or requested

    def wsclean(self, target: TargetLike = None, *, robust=None, **kwargs) -> Union[Image, ImageSet]:
        """Image with WSClean (difmapy on a backend that fixes the imager, like dask-ms)."""
        return self._image(target, robust, "wsclean", **kwargs)

    def tclean(self, target: TargetLike = None, *, robust=None, **kwargs) -> Union[Image, ImageSet]:
        """Image with CASA tclean (difmapy on a backend that fixes the imager, like dask-ms)."""
        return self._image(target, robust, "tclean", **kwargs)

    def difmap(self, target: TargetLike = None, *, robust=None, **kwargs) -> Union[Image, ImageSet]:
        """Image with difmapy (CLEAN on the per-source split, no self-calibration)."""
        return self._image(target, robust, "difmap", **kwargs)

    def split_ms(self, source: str, *, force: bool = False) -> str:
        """Path of the per-source calibrated measurement set, splitting it when missing (or forced)."""
        obs = self._obs
        path = Path(obs.work_dir) / "calibrated_data" / f"{self._code}_{source}.ms"
        if force or not path.is_dir():
            timebin, chanbin = obs.export.averaging(source)
            path = Path(self._backend.export.ms(self._code, source, timebin=timebin, chanbin=chanbin,
                                                metadata=obs.metadata))
        return str(path)

    def _image_with_difmap(self, src: str, robust_values: list[float], img_cfg: dict, weighting: str,
                           niter: int, **kwargs) -> list[Image]:
        """CLEAN ``src`` with difmapy at every robust value (see :mod:`vlbipy.backends.difmap`).

        The source is first searched for in a wide dirty map and re-centred only when a
        significant peak lies far from the phase centre (``[imaging].search_fov``,
        ``search_sigma``, ``recentre_min_beams``).
        """
        from ..backends import difmap
        obs = self._obs
        split = self.split_ms(src)
        image_dir = Path(obs.work_dir) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        report = difmap.image_source(split, str(image_dir / f"{self._code}_{src}"), robust_values=robust_values,
                                     niter=int(niter), gain=float(img_cfg.get("clean_gain", 0.05)),
                                     threshold_sigma=float(kwargs.get("threshold_sigma", 3.0)),
                                     average_channels=bool(img_cfg.get("average_channels", True)),
                                     **_search_settings(img_cfg))
        _write_search_report(image_dir / f"{self._code}_{src}.search.json", report)
        return [_image_from_difmap(src, info, weighting) for info in report["images"].values()]

    def _image(self, target, robust, imager, *, imsize=None, weighting=None, niter=None,
               **kwargs) -> Union[Image, ImageSet]:
        """Image one source at every requested robust value, then render the PNG grid of the set.

        A single robust returns an :class:`Image`; a list (or the config default
        ``[imaging].robust``) returns an :class:`ImageSet`. The PNG preview is one
        figure per source with a panel per robust; its path is recorded on every
        image as ``paths["png"]``.
        """
        obs = self._obs
        src = self._resolve_source_name(target)
        imager = self._backend_imager(imager)
        img_cfg = obs.config.get("imaging", {})
        weighting = weighting or img_cfg.get("weighting", "briggs")
        niter = img_cfg.get("niter", 0) if niter is None else niter
        if robust is None:
            robust = img_cfg.get("robust", [0.0])
        robust_values = list(robust) if isinstance(robust, (list, tuple)) else [robust]
        kwargs.setdefault("threshold_sigma", img_cfg.get("threshold_sigma", 3.0))
        kwargs.setdefault("produce_fits", img_cfg.get("produce_fits", True))
        kwargs.setdefault("produce_png", img_cfg.get("produce_png", True))
        kwargs.setdefault("metadata", obs.metadata)
        if imager == "difmap":
            images = self._image_with_difmap(src, [float(r) for r in robust_values], img_cfg, weighting, niter,
                                             **kwargs)
        else:
            images = [self._backend.image.clean(self._code, src, robust=float(r), imager=imager,
                                                imsize=imsize, weighting=weighting, niter=niter, **kwargs)
                      for r in robust_values]
        fits_paths = {img.robust: img.paths["fits"] for img in images if img.paths.get("fits")}
        if kwargs["produce_png"] and fits_paths and self._backend.supports("plot", "image_grid"):
            try:
                png = self._backend.plot.image_grid(self._code, {src: fits_paths})
                for img in images:
                    img.paths["png"] = png[0] if png else img.paths.get("png", "")
            except Exception as exc:  # noqa: BLE001 - a missing preview must not lose the images
                warnings.warn(f"{self._code}: image preview of {src} failed ({exc})")
        obs._state.mark_complete(f"clean_{src}", outputs=[img.paths.get("fits", "") for img in images])
        return images[0] if len(images) == 1 else ImageSet(images)


def _search_settings(img_cfg: dict) -> dict:
    """difmapy keywords from ``[imaging]``: map width, and the search field, significance and re-centring distance."""
    return {"mapsize": int(img_cfg.get("mapsize", 8192)),
            "search_fov_mas": float(img_cfg.get("search_fov", 1000.0)),
            "search_sigma": float(img_cfg.get("search_sigma", 10.0)),
            "recentre_min_beams": float(img_cfg.get("recentre_min_beams", 10.0))}


def _write_search_report(path: Path, report: dict) -> None:
    """Save the source search of a difmapy report (peak, significance, offset, shift) as JSON, when there is one."""
    if report.get("search"):
        Path(path).write_text(json.dumps({"search": report["search"], "shift_mas": report.get("shift_mas", [0, 0])},
                                         indent=2), encoding="utf-8")


def _image_from_difmap(source: str, info: dict, weighting: str = "briggs") -> Image:
    """Build an :class:`Image` from a :func:`vlbipy.backends.difmap.clean_image` report."""
    beam = tuple(info.get("beam") or (0.0, 0.0, 0.0))
    metrics = QualityMetrics(peak=float(info["peak"]), rms=float(info["rms"]),
                             dynamic_range=float(info["dynamic_range"]),
                             integrated_flux=float(info.get("model_flux", 0.0)),
                             beam=(beam + (0.0, 0.0, 0.0))[:3])
    return Image(source=source, robust=float(info["robust"]), weighting=weighting,
                 paths={"fits": info["fits"]}, stats=metrics)


class SelfcalNamespace(Namespace):
    """Self-calibration with difmapy: modelfit, phase ladder, Bayesian amplitude gains.

    The calibrators are processed in the standard order: the fringe finder
    first, whose amplitude corrections go to every field; then the phase
    calibrator, whose amplitude *and* phase solutions go to itself, the
    targets and the check sources. Tables are CASA "G Jones" tables written by
    difmapy against the parent measurement set and appended to the apply chain,
    so ``calibrate.apply()`` puts them onto the data like any other table.
    """

    def __call__(self, target: TargetLike = None, **kwargs):
        """Self-calibrate a source (default: the fringe finder), a list of them, or an Image's source."""
        if isinstance(target, (list, tuple)):
            return [self(t, **kwargs) for t in target]
        if not self._backend.requires_data_files:
            # In-memory backends have no split measurement set for difmapy: use their own stand-in.
            image = target if isinstance(target, Image) else None
            src = self._resolve_source_name(target)
            result = self._backend.image.selfcal(self._code, src, image=image, **kwargs)
            self._obs._state.mark_complete(f"selfcal_{src}")
            return result
        return self.calibrator(target, **kwargs)

    def _config(self) -> dict:
        return dict(self._obs.config.get("selfcal", {}))

    def _apply_to(self, source: str) -> str:
        """Fields a calibrator's solutions are applied to: FF -> all; others -> everything but the FFs.

        A phase-reference source follows the plan (:meth:`vlbipy.phaseref.PhaseRefPlan.apply_to`):
        ``sequential`` leaves out the references solved before it, ``interpolate`` every other reference.
        """
        obs = self._obs
        finders = {s.name for s in obs.sources.fringe_finders}
        if source in finders:
            return ""
        names = [n for n in (obs.metadata.source_names if obs.metadata else list(obs.sources.names))
                 if n not in finders]
        planned = obs.phaseref_plan().apply_to(source, names)
        return ",".join(names) if planned is None else planned

    def calibrator(self, target: TargetLike = None, *, force: bool = False, transfer_phases: Optional[bool] = None,
                   **kwargs) -> SelfcalResult:
        """Run the difmapy sequence on one calibrator and register its gain tables.

        Parameters
        ----------
        transfer_phases : bool, optional
            Put the phase self-cal table into the apply chain (for the fields of
            :meth:`_apply_to`). Defaults to False for fringe finders, True otherwise.
        """
        from ..backends import difmap
        obs = self._obs
        src = self._resolve_source_name(target) if target is not None else \
            ([s.name for s in obs.sources.fringe_finders] or obs.phaseref_plan().references
             or [s.name for s in obs.sources.targets])[0]
        step = f"selfcal_{src}"
        if not obs._state.should_run(step, force=force):
            return obs._selfcal_results.get(src) or SelfcalResult(src, [], True)
        cfg = self._config()
        img_cfg = obs.config.get("imaging", {})
        is_finder = src in {s.name for s in obs.sources.fringe_finders}
        transfer = (not is_finder) if transfer_phases is None else bool(transfer_phases)
        obs.drop_gaintables({step})
        split = obs.clean.split_ms(src)
        out_dir = Path(obs.work_dir) / "selfcal"
        out_dir.mkdir(parents=True, exist_ok=True)
        image_dir = Path(obs.work_dir) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        solints = cfg.get("solints") or None
        logger.info("selfcal[{}]: difmapy on {} ({}; phases {})", src, Path(split).name,
                    "fringe finder" if is_finder else "phase calibrator", "transferred" if transfer else "kept local")
        report = difmap.calibrate_source(
            split, str(self._backend.ms_path(self._code)), str(out_dir / f"{self._code}.{src}"),
            robust_values=[float(r) for r in img_cfg.get("robust", [-2, 0, 2])],
            solints=list(solints) if solints else None,
            min_improvement=float(cfg.get("min_improvement", 0.002)),
            max_bad_fraction=float(cfg.get("max_bad_fraction", 0.25)),
            bayes_models=tuple(cfg.get("bayes_models", ["clean", "gauss1", "gauss2", "gauss3"])),
            prior_sigma=float(cfg.get("prior_sigma", 0.1)), outlier_sigma=float(cfg.get("outlier_sigma", 1.0)),
            workers=cfg.get("workers") or None,
            imagename=str(image_dir / f"{self._code}_{src}"), niter=int(img_cfg.get("niter", 4000)),
            gain=float(img_cfg.get("clean_gain", 0.05)), threshold_sigma=float(img_cfg.get("threshold_sigma", 3.0)),
            **{**_search_settings(img_cfg), **kwargs})
        _write_search_report(image_dir / f"{self._code}_{src}.search.json", report)
        apply_to = self._apply_to(src)
        amp_path = report["amplitude"].get("caltable", "")
        if amp_path and Path(amp_path).is_dir():
            obs.add_gaintable(CalTable(cal_type=f"selfamp_{src}", path=amp_path, field=src, gainfield=src,
                                       interp="nearest", apply_to=apply_to, snr=0.0), step)
        if transfer and report.get("phase_table") and Path(report["phase_table"]).is_dir():
            obs.add_gaintable(CalTable(cal_type=f"selfphase_{src}", path=report["phase_table"], field=src,
                                       gainfield=src, interp="linear", apply_to=apply_to, snr=0.0), step)
        images = [_image_from_difmap(src, info) for info in report["images"].values()]
        rounds = [dict(r, mode="phase") for r in report["rounds"]]
        result = SelfcalResult(src, rounds, any(r["accepted"] for r in rounds), images[0] if images else None)
        result.images = images
        result.report = report
        obs._selfcal_results[src] = result
        self._preview(src, images)
        obs._state.mark_complete(step, outputs=[t.path for t in obs.gaintables if t.step == step])
        if not result.converged:
            warnings.warn(f"{self._code}: no phase self-cal step improved the fit on {src}")
        logger.info("selfcal[{}]: {} accepted round(s); chain is now {}", src,
                    sum(r["accepted"] for r in rounds), " -> ".join(t.cal_type for t in obs.gaintables))
        return result

    def _preview(self, src: str, images: list) -> None:
        """Render the robust grid PNG for the difmapy images (best effort)."""
        if not images or not self._backend.supports("plot", "image_grid"):
            return
        try:
            png = self._backend.plot.image_grid(self._code, {src: {i.robust: i.paths["fits"] for i in images}})
            for image in images:
                image.paths["png"] = png[0] if png else ""
        except Exception as exc:  # noqa: BLE001 - a preview must not fail the calibration
            warnings.warn(f"{self._code}: image preview of {src} failed ({exc})")

    def run_all(self, *, force: bool = False) -> dict[str, SelfcalResult]:
        """The pipeline's self-calibration stage.

        1. fringe finder(s), one after the other: difmapy sequence; the amplitude table
           goes to every field, the phases stay local (a fringe finder is too far from
           the target for its phases to transfer);
        2. phase-reference sources (the phase calibrators, or the targets with inverse
           phase referencing), in the order of the plan (:meth:`Observation.phaseref_plan`):
           difmapy sequence on data that already carry those amplitudes; amplitude + phase
           tables -> the fields of :meth:`_apply_to`, never the fringe finders;
        3. apply the whole chain to every field, then re-split every source (and
           UVFITS) so imaging sees the final calibration.

        Each calibrator is re-calibrated and re-split on its own just before its
        session, so it sees the gains of the ones before it that apply to it: with the
        ``sequential`` strategy (and ``time``) its table is then a refinement on top of
        theirs; with ``interpolate`` every reference is self-calibrated independently and
        the receivers take the interpolation of their tables.
        """
        obs = self._obs
        results: dict[str, SelfcalResult] = {}
        if not self._config().get("enabled", True):
            logger.info("selfcal: disabled ([selfcal].enabled = false)")
            return results
        if not self._backend.requires_data_files:
            logger.info("selfcal: backend {} has no on-disk data for difmapy; skipping", self._backend.kind)
            return results
        with_data = set(obs.metadata.source_names) if obs.metadata else set(obs.sources.names)
        finders = [s.name for s in obs.sources.fringe_finders if s.name in with_data]
        references = [name for name in obs.phaseref_plan().references if name in with_data and name not in finders]
        for name in finders + references:
            if obs._state.should_run(f"selfcal_{name}", force=force):
                # Deferred: only the split reads it, and calibrates while it reads (dask-ms).
                obs.calibrate.apply(force=True, field=name, defer=True)
                obs.clean.split_ms(name, force=True)
            results[name] = self.calibrator(name, force=force)
        if results:
            obs.calibrate.apply(force=True)
            obs.export.per_source(force=True)
        return results


class ExportNamespace(Namespace):
    """Split/export per-source products (UVFITS, measurement set)."""

    def __call__(self, source: TargetLike = None, **kwargs) -> str:
        """Default export: per-source UVFITS."""
        return self.uvfits(source, **kwargs)

    def uvfits(self, source: TargetLike = None, *, time_average: str = "", channel_average: int = 1,
               **kwargs) -> str:
        """Export a source to UVFITS."""
        src = self._resolve_source_name(source)
        return self._backend.export.uvfits(self._code, src, time_average=time_average,
                                           channel_average=channel_average, **kwargs)

    def ms(self, source: TargetLike = None, **kwargs) -> str:
        """Split a source into its own measurement set."""
        src = self._resolve_source_name(source)
        return self._backend.export.ms(self._code, src, **kwargs)

    def averaging(self, source: str) -> tuple[str, int]:
        """Time and channel averaging of the split of ``source``: ``(timebin, chanbin)``.

        ``[export].time_average`` applies to every source. Calibrators and check
        sources are averaged to ``channel_average`` (default: one channel per
        subband), targets to ``target_channel_average`` (default 4), and
        ``average_targets = false`` keeps the target(s) at full resolution.
        """
        cfg = self._obs.config.get("export", {})
        timebin = str(cfg.get("time_average", "10s") or "")
        if source not in {s.name for s in self._obs.sources.targets}:
            return timebin, int(cfg.get("channel_average", -1))
        if not cfg.get("average_targets", True):
            return "", 1
        return timebin, int(cfg.get("target_channel_average", 4))

    def calibrators(self, **kwargs) -> dict[str, str]:
        """Split only the calibrators (fringe finders and phase calibrators)."""
        obs = self._obs
        names = [s.name for s in obs.sources.calibrators]
        if not names:
            raise StepError("split", f"{self._code}: no calibrators defined to split")
        return self.per_source(sources=names, **kwargs)

    def per_source(self, *, force: bool = False, sources: Optional[list] = None,
                   uvfits: bool = True, **kwargs) -> dict[str, str]:
        """Split every source with data into its own measurement set.

        The split writes the CORRECTED column into the output DATA column, so
        each file is standalone calibrated data — run this after ``calibrate``.

        Returns
        -------
        dict
            Mapping source name -> path of the split measurement set. Sources
            that fail to split are reported as warnings and omitted, so one bad
            field does not lose the rest.
        """
        obs = self._obs
        if not obs._state.should_run("split", force=force):
            return {}
        names = obs.metadata.source_names if obs.metadata else obs.sources.names
        if sources:
            names = [n for n in names if n in set(sources)]
        roles = {s.name: s.source_type.value for s in obs.sources}
        logger.info("split: writing {} per-source measurement set(s): {}", len(names),
                    ", ".join(f"{n} ({roles.get(n, 'unknown role')})" for n in names))
        kwargs.setdefault("metadata", obs.metadata)
        products: dict[str, str] = {}
        for name in names:
            timebin, chanbin = self.averaging(name)
            try:
                products[name] = self._backend.export.ms(
                    self._code, name, **{"timebin": timebin, "chanbin": chanbin, **kwargs})
            except Exception as exc:  # noqa: BLE001 - one bad field must not lose the others
                warnings.anomaly(f"{self._code}: could not split {name}: {exc}")
        logger.info("split: wrote {} of {} per-source measurement set(s)", len(products), len(names))
        if uvfits:
            for name, path in list(products.items()):
                try:
                    products[f"{name}.uvfits"] = self._backend.export.uvfits(
                        self._code, name, split_ms=path)
                except Exception as exc:  # noqa: BLE001 - one failed export must not lose the rest
                    warnings.anomaly(f"{self._code}: could not export {name} to UVFITS: {exc}")
        obs._state.mark_complete("split", outputs=list(products))
        return products


__all__ = [
    "Namespace", "ImportDataNamespace", "CalibrateNamespace", "FlagNamespace",
    "PlotNamespace", "CleanNamespace", "SelfcalNamespace", "ExportNamespace", "SelfcalResult",
]
