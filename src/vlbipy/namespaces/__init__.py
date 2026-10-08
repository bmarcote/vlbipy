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

from ..diagnostics import write_summary
from ..errors import BackendError, StepError
from ..logging_utils import get_logger, warnings
from ..models import CalTable, QualityMetrics
from ..results import Image, ImageSet, SelfcalResult
from ..sources import Source
from ..tools import natsort_key

logger = get_logger()

#: Flagged-fraction above which a flag step is flagged as an anomaly.
_HIGH_FLAG_FRACTION = 0.3

TargetLike = Union[str, Source, Image, None]


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
        from ..selection import plan_sbd_stages, select_antennas

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
            logger.info("select_calibration_data: reusing the instrumental selection fixed on the "
                        "first pass — {} antenna(s) {} on scan(s) {} (not re-surveying the now-flagged "
                        "data)", len(antennas), ",".join(antennas), scans)
            return antennas, scans
        cal_cfg = obs.config.get("calibration", {})
        threshold = min_snr if min_snr is not None else cal_cfg.get("detection_snr", 7.0)
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
            stages = plan_sbd_stages(survey, antennas, min_snr=threshold, sources=names, metadata=obs.metadata)
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
            self._code, obs.calibrator_field, ",".join(antennas or []) or obs.refant,
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
                self._code, obs.calibrator_field, ",".join(antennas or []) or obs.refant,
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

        # Pass 1: rough delays, so the bandpass is solved on delay-corrected data.
        first_sbd = self.initial_calibration(force=force, scans=scans, antennas=antennas)
        first_mbd = self.fringefit(force=force, plot=False)

        # Bandpass, now that both delays are removed.
        bandpass = self.bandpass(force=force, scans=scans, antennas=antennas)

        # Pass 2: re-solve both delays *through* everything already in the chain. These
        # are incremental corrections on top of the first pass, not replacements — the
        # first-pass tables stay, and all of them are applied together.
        refined_sbd = self.initial_calibration(force=True, scans=scans, antennas=antennas,
                                               suffix="sbd2")
        refined_mbd = self.fringefit(force=True, suffix="mbd2", plot=plot)

        self.verify_solutions(refined_sbd, antennas)
        if plot:
            for table in (bandpass, refined_sbd):
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
        # Solve on the phase calibrator: gaincal assumes a point source, so a resolved
        # fringe finder would have its structure absorbed into the antenna gains.
        compact = obs.sources.phase_calibrators or obs.sources.calibrators
        try:
            table = self._backend.calibrate.scalar_bandpass(
                self._code, ",".join(s.name for s in compact) or obs.calibrator_field,
                obs.refant, gaintable=list(obs.gaintables), metadata=obs.metadata,
                callib=cal_cfg.get("callib", False), **cfg)
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
        """Global (multi-band delay) fringe fit on all calibrators."""
        obs = self._obs
        step = "fringefit" if suffix == "mbd" else f"fringefit_{suffix}"
        if not obs._state.should_run(step, force=force):
            return obs._table(suffix)
        cals = obs.sources.calibrators or obs.sources.targets
        field = ",".join(s.name for s in cals)
        obs.drop_gaintables({step})
        cal_cfg = obs.config.get("calibration", {})
        mbd_cfg = dict(cal_cfg.get("mbd", {}))
        mbd_cfg.setdefault("dispersive", self._solve_dispersive(cal_cfg))
        table = self._backend.calibrate.fringefit(
            self._code, field, obs.refant, gaintable=list(obs.gaintables),
            metadata=obs.metadata, suffix=suffix,
            callib=cal_cfg.get("callib", False), **mbd_cfg)
        # The fringe solutions are what gets transferred to the target: tie them to the
        # phase calibrator(s) so applycal maps those solutions onto every field.
        phase_cals = obs.sources.phase_calibrators
        if phase_cals:
            table.gainfield = ",".join(s.name for s in phase_cals)
            logger.info("fringefit: solutions will be applied from {}", table.gainfield)
        obs.add_gaintable(table, step)
        if plot:
            obs.plot.caltable(table)
        obs._state.mark_complete(step, outputs=[suffix])
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
        calibrators = obs.sources.calibrators or obs.sources.targets
        selection = field or ",".join(s.name for s in calibrators)
        cached = obs._snr_surveys.get(selection)
        if cached is not None and not force:
            logger.info("scan_snr: reusing the survey of {}", selection)
            return cached
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

    def apply(self, *, force: bool = False, field: str = "") -> None:
        """Apply the accumulated calibration tables to every source (or one ``field``)."""
        if not self._obs._state.should_run("apply", force=force):
            return
        # Empty selection = every observed source field: the targets need the
        # calibration too, and anything left uncorrected would silently be imaged
        # from raw data.  Per-source calls let the field mapping resolve nearest for
        # calibrators and the phase calibrator for targets.
        cal_cfg = self._obs.config.get("calibration", {})
        self._backend.calibrate.apply(
            self._code, field, list(self._obs.gaintables),
            callib=cal_cfg.get("callib", False))
        self._obs._state.mark_complete("apply")


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
            gaps = self._backend.flag.bandpass_gaps(self._code, table, min_gain=float(cfg.get("bandpass_min_gain", 0.5)))
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


class PlotNamespace(Namespace):
    """Diagnostic plotting."""

    #: Maximum number of phase-calibrator scans the per-scan diagnostics fall back to.
    MAX_DIAGNOSTIC_SCANS = 5

    def __call__(self, *, column: str = "corrected", label: str = "") -> list[str]:
        """Produce the standard diagnostic set (see :meth:`diagnostics`)."""
        return self.diagnostics(column=column, label=label)

    def _run_jobs(self, jobs: list, label: str) -> list[str]:
        """Run ``(name, callable)`` plot jobs, warning on failures instead of aborting."""
        written: list[str] = []
        for name, job in jobs:
            try:
                result = job()
            except Exception as exc:  # noqa: BLE001 - a plot must never abort the reduction
                warnings.warn(f"{self._code}: {label} {name} plot failed ({exc})")
                continue
            written.extend(result if isinstance(result, list) else [result])
        return written

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
        amplitude/phase vs uv distance, and the uv coverage (raw only; it does
        not change).

        Plotting is reporting: a plot that fails is logged as a warning and the
        rest of the set (and the pipeline) carries on.

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
            return sorted(str(p) for p in self._backend.plot.plot_dir("raw").glob("*.png"))
        calibrators = obs.sources.calibrators or obs.sources.targets
        # Raw diagnostics trigger the calibrator-only survey used for calibration
        # selection. Corrected diagnostics only render the comprehensive survey that
        # the orchestrator measured after the final applycal.
        jobs = [("scan_snr", lambda: self.scan_snr())] if (
            column == "data" or (obs.metadata and obs.metadata.snr_survey is not None)) else []
        if column == "data":
            jobs += [("scan_diagnostics", lambda: self.scan_diagnostics(column=column, label=label))]
            if self._backend.supports("plot", "diagnostic"):
                jobs += [("uv_coverage", lambda: [self.uv_coverage()])]
        jobs += [(f"spectrum[{s.name}]", lambda source=s: self.spectrum(
                    field=source.name, column=column, label=f"{label}_{source.name}"))
                 for s in calibrators]
        jobs += [(f"timeseries[{s.name}]", lambda source=s: self.timeseries(
                    field=source.name, column=column, label=f"{label}_{source.name}"))
                 for s in calibrators]
        jobs += [("corners", lambda: [p for s in calibrators for p in
                                      self.corners(field=s.name, column=column, label=f"{label}_{s.name}")]),
                 ("radplot", lambda: self.radplot(column=column, label=label))]
        if column == "corrected":
            jobs.append(("subband_phases", lambda: [self.subband_phases(
                column=column, label=label)]))
        written = self._run_jobs(jobs, label)
        logger.info("plot.diagnostics[{}]: {} plot(s) on the {} column", label, len(written), column)
        if column == "data" and label == "raw":
            obs._state.mark_complete("plot_raw")
        return written

    def final_data(self) -> list[str]:
        """The final-data set on the calibrated column: every source, labelled ``final``.

        uv coverage of all sources in one figure, amplitude/phase vs uv distance
        with the model overlaid (when a MODEL column exists), spectra and
        per-baseline light curves for every source, and the total coherent
        amplitude light curve. Failing plots warn and the set carries on.
        """
        obs = self._obs
        names = self._all_source_names()
        jobs = []
        if self._backend.supports("plot", "diagnostic"):
            jobs.append(("uv_coverage", lambda: [self.uv_coverage()]))
        jobs += [("radplot", lambda: self.radplot(all_sources=True, column="corrected", label="final",
                                                  with_model=True))]
        jobs += [(f"spectrum[{name}]", lambda n=name: self.spectrum(field=n, column="corrected", label=f"final_{n}"))
                 for name in names]
        jobs += [(f"timeseries[{name}]", lambda n=name: self.timeseries(field=n, column="corrected",
                                                                         label=f"final_{n}"))
                 for name in names]
        jobs += [("lightcurve", lambda: self.lightcurve(column="corrected", label="final"))]
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
        return self._plot("uv_coverage", metadata=self._obs.metadata)

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
        obs = self._obs
        return self._backend.plot.autocorr(self._code, field=field or obs.calibrator_field, scans=scans,
                                           column=column, label=label, metadata=obs.metadata, **kwargs)

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
        obs = self._obs
        refant = (refant or str(obs.refant)).split(",")[0]
        return self._backend.plot.spectrum(self._code, field=field or obs.calibrator_field,
                                           scans=scans, refant=refant, column=column,
                                           label=label, all_pols=all_pols,
                                           metadata=obs.metadata, **kwargs)

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
        written: list[str] = []
        scans = self._diagnostic_scans()
        for scan in scans:
            tag = f"{label}_scan{scan.scan_number}"
            jobs = [(f"autocorr[{scan.scan_number}]",
                     lambda s=scan, t=tag: self.autocorr(field=s.source, scans=[s.scan_number], column=column,
                                                         label=t)),
                    (f"spectrum[{scan.scan_number}]",
                     lambda s=scan, t=tag: self.spectrum(field=s.source, scans=[s.scan_number], column=column,
                                                         all_pols=(column == "data"), label=t,
                                                         refant=self._refant_for_scan(s)))]
            written.extend(self._run_jobs(jobs, label))
        logger.info("scan_diagnostics: {} plot(s) over {} scan(s) ({})", len(written), len(scans),
                    ", ".join(str(s.scan_number) for s in scans) or "none")
        return written

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
        obs = self._obs
        if all_sources:
            names = self._all_source_names()
        else:
            names = list(sources or [s.name for s in obs.sources.calibrators]
                         or [s.name for s in obs.sources.targets])
        cfg = obs.config.get("export", {})
        written = []
        for name in names:
            written.append(self._backend.plot.radplot(
                self._code, field=name, column=column, label=label,
                time_bin=time_bin if time_bin is not None else cfg.get("radplot_time_bin", 10.0),
                with_model=with_model and column == "corrected", metadata=obs.metadata, **kwargs))
        logger.info("radplot: {} plot(s) for {}", len(written), ", ".join(names))
        return written

    def lightcurve(self, *, column: str = "corrected", label: str = "calibrated",
                   averaging_sec: Optional[list] = None, **kwargs) -> str:
        """Total coherent visibility amplitude vs time for every source, at several averaging scales.

        ``averaging_sec`` defaults to ``[export].lightcurve_averaging`` (``0`` =
        native integrations, positive = seconds within a scan, ``-1`` = per scan).
        """
        obs = self._obs
        scales = averaging_sec or obs.config.get("export", {}).get("lightcurve_averaging", [0, 30, 120, -1])
        return self._backend.plot.lightcurve(self._code, fields=self._all_source_names(), column=column,
                                             label=label, averaging_sec=list(scales), metadata=obs.metadata,
                                             **kwargs)

    def subband_phases(self, *, column: str = "corrected", label: str = "calibrated", **kwargs) -> str:
        """Residual phase jumps between subbands, per calibrator scan and baseline to the refant.

        The verification of the single-band delay: on calibrated data the phase
        of every subband relative to the first should sit at zero on every scan.
        Goes through every scan on every calibrator (fringe finders and phase
        calibrators).
        """
        obs = self._obs
        fields = [s.name for s in obs.sources.calibrators] or [s.name for s in obs.sources.targets]
        return self._backend.plot.subband_phases(self._code, fields=fields, refant=str(obs.refant).split(",")[0],
                                                 column=column, label=label, metadata=obs.metadata, **kwargs)

    def timeseries(self, *, field: str = "", scans: Optional[list] = None,
                   column: str = "corrected", label: str = "", refant: str = "",
                   **kwargs) -> list[str]:
        """Amplitude and phase vs time per baseline for an optional exact scan selection."""
        obs = self._obs
        return self._backend.plot.timeseries(
            self._code, field=field or obs.calibrator_field, scans=scans,
            refant=(refant or str(obs.refant)).split(",")[0], column=column, label=label,
            metadata=obs.metadata, **kwargs)

    def corner(self, *, field: str = "", quantity: str = "phase", column: str = "corrected",
               label: str = "", **kwargs) -> str:
        """Per-baseline time x frequency grid for one field, coloured by phase or amplitude.

        Defaults to the phase calibrator: that is the source whose calibrated
        phases should be flat everywhere, so any structure left in a cell is a
        calibration problem rather than the sky.
        """
        obs = self._obs
        if not field:
            group = obs.sources.phase_calibrators or obs.sources.calibrators or obs.sources.targets
            field = ",".join(s.name for s in group)
        return self._backend.plot.baseline_corner(self._code, field=field, quantity=quantity,
                                                  column=column, label=label,
                                                  metadata=obs.metadata, **kwargs)

    def corners(self, **kwargs) -> list[str]:
        """Both corner plots — phase and amplitude (Stokes I) — for one field."""
        return [self.corner(quantity=q, **kwargs) for q in ("phase", "amp")]

    def image_grid(self, images: dict) -> list[str]:
        """Render the FITS images of each source side by side (``{source: {robust: fits}}``)."""
        return self._backend.plot.image_grid(self._code, images)

    def bandpass_profile(self, measurement: dict) -> str:
        """Plot the per-channel band profile behind the edge-channel decision."""
        return self._backend.plot.bandpass_profile(self._code, measurement)

    def caltable(self, table: CalTable) -> list[str]:
        """Plot one calibration table to PNG(s): Tsys/gain-curve/fringe/gain layouts."""
        return self._backend.plot.caltable(self._code, table.path, cal_type=table.cal_type)

    def caltables(self) -> list[str]:
        """Plot every calibration table produced so far.

        Uses the in-memory ``obs.gaintables`` when populated; in a fresh process
        it falls back to the tables found on disk in <work_dir>/caltables.
        """
        tables = list(self._obs.gaintables)
        if not tables:
            caldir = Path(self._obs.work_dir) / "caltables"
            tables = [CalTable(cal_type=p.suffix.lstrip("."), path=str(p))
                      for p in sorted(caldir.glob("*")) if p.is_dir()]
            if tables:
                logger.info("plot.caltables: found {} caltable(s) on disk", len(tables))
        paths: list[str] = []
        for table in tables:
            paths.extend(self.caltable(table))
        return paths


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
        """Fields a calibrator's solutions are applied to: FF -> all; others -> everything but the FFs."""
        obs = self._obs
        finders = {s.name for s in obs.sources.fringe_finders}
        if source in finders:
            return ""
        names = obs.metadata.source_names if obs.metadata else list(obs.sources.names)
        return ",".join(n for n in names if n not in finders)

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
            (obs.sources.fringe_finders or obs.sources.phase_calibrators or obs.sources.targets)[0].name
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
            prior_sigma=float(cfg.get("prior_sigma", 0.1)), workers=cfg.get("workers") or None,
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
        2. phase calibrator(s): difmapy sequence on data that already carry those
           amplitudes; amplitude + phase tables -> the phase calibrator, the targets and
           the check sources, never the fringe finders;
        3. apply the whole chain to every field, then re-split every source (and
           UVFITS) so imaging sees the final calibration.

        Each calibrator is re-calibrated and re-split on its own just before its
        session, so it sees the gains of the ones before it: its table is then a
        refinement on top of theirs rather than a second copy of the same correction.
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
        phasecals = [s.name for s in obs.sources.phase_calibrators if s.name in with_data and s.name not in finders]
        for name in finders + phasecals:
            if obs._state.should_run(f"selfcal_{name}", force=force):
                obs.calibrate.apply(force=True, field=name)
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
