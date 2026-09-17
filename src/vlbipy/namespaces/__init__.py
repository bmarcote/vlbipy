"""Callable operation namespaces for vlbipy.

Each namespace is bound to a single :class:`~vlbipy.observation.Observation` and
implements the *callable-namespace* pattern: calling the namespace runs its
sensible default, while its methods run explicit variants. For example::

    obs.clean(...)            # default imager (WSClean)
    obs.clean.wsclean(...)    # explicit imager
    obs.calibrate()           # full default chain
    obs.calibrate.bandpass()  # one step

Namespaces translate high-level intent into calls on the observation's
:class:`~vlbipy.backends.base.Backend`; they never import a backend library and
never touch the filesystem directly.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Union

from ..diagnostics import write_summary
from ..errors import BackendError, StepError
from ..logging_utils import get_logger, warnings
from ..models import CalTable
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
                obs._metadata = self._load_metadata()
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
                                           keep_ms=imp_cfg.get("keep_ms", False),
                                           mms=imp_cfg.get("mms", True),
                                           needs_eop=obs._observatory_handler.needs_eop, **kwargs)
            inputs = list(file_list)
        else:
            handler = obs._observatory_handler
            logger.info("import_data: {} {}", handler.name,
                        "auto-download available" if handler.auto_download else "is manual-download only")
            self._backend.data.import_data(self._code, obs.sources.names, scan_gap=scan_gap, **kwargs)
        obs._metadata = self._load_metadata()
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
        return metadata

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
        """A-priori amplitude calibration: gencal Tsys + gain curve (+ EOP for VLBA/LBA).

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
        tables = self._backend.calibrate.a_priori(
            self._code, obs.calibrator_field, needs_eop=handler.needs_eop,
            eop_file=cal_cfg.get("eop_file") or None, antab=antab)
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

        Antennas must have recorded every subband and been detected above
        ``min_snr``; scans are then picked so that all of those antennas are
        detected, using several linked scans when no single one covers the array.

        Returns
        -------
        tuple
            ``(antenna names best-first, scan numbers)``.
        """
        from ..selection import select_antennas, select_calibration_scans

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
        attempted = []
        for group in (obs.sources.fringe_finders, obs.sources.phase_calibrators, obs.sources.targets):
            if not group:
                continue
            names = [s.name for s in group]
            attempted.extend(names)
            survey = self.scan_snr(field=",".join(names))
            antennas = select_antennas(survey, obs.metadata, min_snr=threshold,
                                       require_all_subbands=cal_cfg.get("require_all_subbands", True))
            if not antennas:
                logger.info("select_calibration_data: no antenna qualifies on {}; trying the "
                            "next source group", ",".join(names))
                continue
            scans = select_calibration_scans(survey, antennas, min_snr=threshold, sources=names)
            if scans:
                obs.set_cal_selection(antennas, scans)
                logger.info("select_calibration_data: fixed the instrumental selection for the whole "
                            "run — {} antenna(s) {} on scan(s) {}; every pass solves on this set",
                            len(antennas), ",".join(antennas), scans)
                return antennas, scans
        raise StepError(f"{self._code}: no scan on {', '.join(attempted) or 'any source'} detects "
                        f"a full-band antenna above {threshold:g} sigma; cannot solve the "
                        f"instrumental delay")

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
        table = self._backend.calibrate.initial_calibration(
            self._code, obs.calibrator_field, ",".join(antennas or []) or obs.refant,
            scans=scans, gaintable=list(obs.gaintables), metadata=obs.metadata,
            suffix=suffix, **settings)
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
        bp_cfg = dict(obs.config.get("calibration", {}).get("bandpass", {}))
        try:
            table = self._backend.calibrate.bandpass(
                self._code, obs.calibrator_field, ",".join(antennas or []) or obs.refant,
                scans=scans, gaintable=list(obs.gaintables), metadata=obs.metadata, **bp_cfg)
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
        (Tsys / gain curve / EOP, which do not depend on the data) gives cleaner
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
        keep = {"tsys", "gc", "eop"}
        dropped = [t.cal_type for t in obs.gaintables if t.cal_type not in keep]
        obs.set_gaintables([t for t in obs.gaintables if t.cal_type in keep])
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
        cfg = dict(obs.config.get("calibration", {}).get("scalar_bandpass", {}))
        # Solve on the phase calibrator: gaincal assumes a point source, so a resolved
        # fringe finder would have its structure absorbed into the antenna gains.
        compact = obs.sources.phase_calibrators or obs.sources.calibrators
        try:
            table = self._backend.calibrate.scalar_bandpass(
                self._code, ",".join(s.name for s in compact) or obs.calibrator_field,
                obs.refant, gaintable=list(obs.gaintables), metadata=obs.metadata, **cfg)
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
        for name in antennas:
            recorded = set(obs.metadata.antennas[name].subbands) if obs.metadata else set()
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
            metadata=obs.metadata, suffix=suffix, **mbd_cfg)
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
                      "max_scans": cal_cfg.get("snr_max_scans", 24)}
        survey_cfg.update(cal_cfg.get("snr_survey", {}))
        survey_cfg.update(kwargs)
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
        # Empty selection = every field: the targets need the calibration too, and
        # anything left uncorrected would silently be imaged from raw data.
        self._backend.calibrate.apply(self._code, field, list(self._obs.gaintables))
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
        return flagged + self.autocorr(force=force)

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
        the ramp measured from the ``column`` data per antenna
        (``[flagging].quack_sigma``, ``quack_max_seconds``).
        """
        obs = self._obs
        if not obs._state.should_run("flag_quack", force=force):
            return 0.0
        cfg = obs.config.get("flagging", {})
        per_antenna = per_antenna if per_antenna is not None else dict(cfg.get("quack_antennas", {}) or {})
        interval = float(interval if interval is not None else cfg.get("quack_interval", 0) or 0)
        measure_on = (obs.sources.phase_calibrators or obs.sources.calibrators or obs.sources.targets)
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
        report = self._backend.flag.outliers(
            self._code, field=field, dry_run=dry_run, metadata=obs.metadata,
            threshold=threshold if threshold is not None else cfg.get("outlier_sigma", 5.0),
            **kwargs)
        share = report.get("flagged_fraction_of_data", 0.0)
        if share >= _HIGH_FLAG_FRACTION:
            warnings.anomaly(f"{self._code}: outlier flagging removed {share:.1%} of the data")
        if not dry_run:
            obs._state.mark_complete(step)
        return report

    def from_file(self, path: str, *, force: bool = False) -> float:
        """Apply flags from an external flag command file."""
        return self._run("from_file", force=force, flagfile=path)

    def manual(self, *, force: bool = False, **selection) -> float:
        """Apply a manual flag selection (e.g. antenna/spw/timerange)."""
        return self._run("manual", force=force, **selection)


class PlotNamespace(Namespace):
    """Diagnostic plotting."""

    def __call__(self, *, column: str = "corrected", label: str = "") -> list[str]:
        """Produce the standard diagnostic set (see :meth:`diagnostics`)."""
        return self.diagnostics(column=column, label=label)

    def diagnostics(self, *, column: str = "corrected", label: str = "") -> list[str]:
        """The standard diagnostic set on one data column (SKILL steps 7 and 15).

        On the raw data (``column="data"``) this shows which antennas actually
        observed, where signal exists and what needs flagging; on the calibrated
        data (``"corrected"``) the same plots must show flat phases near zero
        and stable amplitudes on the calibrators. The set: scan x antenna fringe
        SNR (tplot), full-Stokes cross-correlation spectra on the fringe-finder
        scans (raw only), amplitude/phase vs frequency and vs time on baselines
        to the reference antenna, per-baseline corner plots, amplitude/phase vs
        uv distance, and the uv coverage (raw only; it does not change).

        Plotting is reporting: a plot that fails is logged as a warning and the
        rest of the set (and the pipeline) carries on.

        Returns
        -------
        list of str
            Paths of the PNG files written.
        """
        obs = self._obs
        label = label or ("raw" if column == "data" else "calibrated")
        phase_cals = obs.sources.phase_calibrators or obs.sources.calibrators
        jobs = [("scan_snr", lambda: self.scan_snr())] if column == "data" else []
        if column == "data":
            jobs += [("raw_stokes", lambda: self.raw_stokes()),
                     ("uv_coverage", lambda: [self.uv_coverage()])]
        jobs += [("spectrum", lambda: self.spectrum(column=column, label=label)),
                 ("timeseries", lambda: [p for s in phase_cals for p in
                                         self.timeseries(field=s.name, column=column, label=f"{label}_{s.name}")]),
                 ("corners", lambda: self.corners(column=column, label=label)),
                 ("radplot", lambda: self.radplot(column=column, label=label))]
        written: list[str] = []
        for name, job in jobs:
            if name == "uv_coverage" and not self._backend.supports("plot", "diagnostic"):
                continue
            try:
                result = job()
            except Exception as exc:  # noqa: BLE001 - a plot must never abort the reduction
                warnings.warn(f"{self._code}: {label} {name} plot failed ({exc})")
                continue
            written.extend(result if isinstance(result, list) else [result])
        logger.info("plot.diagnostics[{}]: {} plot(s) on the {} column", label, len(written), column)
        return written

    def _plot(self, kind: str, **kwargs) -> str:
        return self._backend.plot.diagnostic(self._code, kind, **kwargs)

    def tplot(self) -> str:
        """Antenna-participation timeline."""
        return self._plot("tplot")

    def uv_coverage(self) -> str:
        """UV-coverage plot."""
        return self._plot("uv_coverage")

    def elevation(self) -> str:
        """Source elevation vs time."""
        return self._plot("elevation")

    def amp_vs_time(self) -> str:
        """Amplitude vs time."""
        return self._plot("amp_vs_time")

    def phase_vs_time(self) -> str:
        """Phase vs time."""
        return self._plot("phase_vs_time")

    def autocorr(self) -> str:
        """Auto-correlation spectra."""
        return self._plot("autocorr")

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

    def spectrum(self, *, field: str = "", scans: Optional[list] = None, column: str = "corrected",
                 label: str = "", all_pols: bool = False, **kwargs) -> list[str]:
        """Plot amplitude/phase vs channel of the calibrated data, per baseline to the refant.

        Defaults to the scan the instrumental calibration was solved on, since
        that is where the response should be flattest — but the calibration has
        been applied to every source, so any field or scan can be inspected.
        """
        obs = self._obs
        return self._backend.plot.spectrum(self._code, field=field or obs.calibrator_field,
                                           scans=scans, refant=obs.refant, column=column,
                                           label=label, all_pols=all_pols,
                                           metadata=obs.metadata, **kwargs)

    def radplot(self, *, sources: Optional[list] = None, column: str = "corrected",
                time_bin: Optional[float] = None, label: str = "", **kwargs) -> list[str]:
        """Amplitude and phase vs uv distance, one plot per calibrator source.

        Defaults to every calibrator: these are the sources whose structure the
        calibration depends on, so a resolved one showing a falling amplitude
        profile is something to know about before trusting its solutions.
        """
        obs = self._obs
        names = list(sources or [s.name for s in obs.sources.calibrators]
                     or [s.name for s in obs.sources.targets])
        cfg = obs.config.get("export", {})
        written = []
        for name in names:
            written.append(self._backend.plot.radplot(
                self._code, field=name, column=column, label=label,
                time_bin=time_bin if time_bin is not None else cfg.get("radplot_time_bin", 10.0),
                metadata=obs.metadata, **kwargs))
        logger.info("radplot: {} plot(s) for {}", len(written), ", ".join(names))
        return written

    def timeseries(self, *, field: str = "", column: str = "corrected", label: str = "",
                   **kwargs) -> list[str]:
        """Amplitude and phase vs time per baseline, in the stacked two-panel layout."""
        obs = self._obs
        return self._backend.plot.timeseries(self._code, field=field or obs.calibrator_field,
                                             refant=obs.refant, column=column, label=label,
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

    def raw_stokes(self, **kwargs) -> list[str]:
        """Full-Stokes amplitude/phase spectra of the raw data, one plot per fringe-finder scan.

        Cross-hands are only shown here: before calibration they carry the
        instrumental polarization signature worth inspecting, whereas afterwards
        they are noise on an unpolarized calibrator and only obscure RR/LL.
        """
        obs = self._obs
        finders = obs.sources.fringe_finders or obs.sources.calibrators or obs.sources.targets
        names = [s.name for s in finders]
        written: list[str] = []
        for scan in (obs.metadata.scans if obs.metadata else []):
            if scan.source not in names:
                continue
            written.extend(self.spectrum(field=scan.source, scans=[scan.scan_number],
                                         column="data", all_pols=True,
                                         label=f"raw_scan{scan.scan_number}", **kwargs))
        logger.info("raw_stokes: {} plot(s) over {} fringe-finder scan(s)",
                    len(written), len(written) // 2)
        return written

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

    default_imager = "wsclean"

    def __call__(self, target: TargetLike = None, *, robust=0.0, imager: Optional[str] = None,
                 **kwargs) -> Union[Image, ImageSet]:
        """Image a source with the default imager (or ``imager=``)."""
        return self._image(target, robust, imager or self.default_imager, **kwargs)

    def wsclean(self, target: TargetLike = None, *, robust=0.0, **kwargs) -> Union[Image, ImageSet]:
        """Image with WSClean."""
        return self._image(target, robust, "wsclean", **kwargs)

    def tclean(self, target: TargetLike = None, *, robust=0.0, **kwargs) -> Union[Image, ImageSet]:
        """Image with CASA tclean."""
        return self._image(target, robust, "tclean", **kwargs)

    def _image(self, target, robust, imager, *, imsize=None, weighting=None, niter=None,
               **kwargs) -> Union[Image, ImageSet]:
        src = self._resolve_source_name(target)
        img_cfg = self._obs.config.get("imaging", {})
        weighting = weighting or img_cfg.get("weighting", "briggs")
        niter = img_cfg.get("niter", 0) if niter is None else niter
        robust_values = list(robust) if isinstance(robust, (list, tuple)) else [robust]
        images = [self._backend.image.clean(self._code, src, robust=float(r), imager=imager,
                                            imsize=imsize, weighting=weighting, niter=niter, **kwargs)
                  for r in robust_values]
        self._obs._state.mark_complete(f"clean_{src}")
        return images[0] if len(images) == 1 else ImageSet(images)


class SelfcalNamespace(Namespace):
    """Self-calibration (phase-only then amp+phase, with convergence checks)."""

    def __call__(self, target: TargetLike = None, **kwargs):
        """Run the default self-cal loop for a source, list of sources, or Image."""
        return self._run(target, mode="both", **kwargs)

    def phase(self, target: TargetLike = None, **kwargs):
        """Phase-only self-cal."""
        return self._run(target, mode="p", **kwargs)

    def ampphase(self, target: TargetLike = None, **kwargs):
        """Amplitude+phase self-cal."""
        return self._run(target, mode="ap", **kwargs)

    def _run(self, target, *, mode="both", phase_rounds=None, ampphase_rounds=None, **kwargs):
        # A list/iterable of sources -> self-cal each, return list.
        if isinstance(target, (list, tuple)):
            return [self._run(t, mode=mode, phase_rounds=phase_rounds,
                              ampphase_rounds=ampphase_rounds, **kwargs) for t in target]
        image = target if isinstance(target, Image) else None
        src = self._resolve_source_name(target)
        sc_cfg = self._obs.config.get("selfcal", {})
        if mode == "p":
            pr, ar = (phase_rounds if phase_rounds is not None else sc_cfg.get("phase_rounds", 4)), 0
        elif mode == "ap":
            pr, ar = 0, (ampphase_rounds if ampphase_rounds is not None else sc_cfg.get("ampphase_rounds", 5))
        else:
            pr = phase_rounds if phase_rounds is not None else sc_cfg.get("phase_rounds", 4)
            ar = ampphase_rounds if ampphase_rounds is not None else sc_cfg.get("ampphase_rounds", 5)
        result = self._backend.image.selfcal(self._code, src, image=image, phase_rounds=pr,
                                             ampphase_rounds=ar, **kwargs,
                                             threshold=sc_cfg.get("convergence_threshold", 0.05))
        self._obs._state.mark_complete(f"selfcal_{src}")
        if not result.converged:
            warnings.warn(f"{self._code}: self-cal on {src} did not improve dynamic range")
        return result


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

    def calibrators(self, **kwargs) -> dict[str, str]:
        """Split only the calibrators (fringe finders and phase calibrators)."""
        obs = self._obs
        names = [s.name for s in obs.sources.calibrators]
        if not names:
            raise StepError(f"{self._code}: no calibrators defined to split")
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
        export_cfg = obs.config.get("export", {})
        kwargs.setdefault("timebin", export_cfg.get("time_average", ""))
        kwargs.setdefault("chanbin", export_cfg.get("channel_average", -1))
        kwargs.setdefault("metadata", obs.metadata)
        products: dict[str, str] = {}
        for name in names:
            try:
                products[name] = self._backend.export.ms(self._code, name, **kwargs)
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
