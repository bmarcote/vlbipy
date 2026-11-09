"""VLBIObs — the single public entry point for vlbipy.

``VLBIObs`` represents one *or many* projects (a campaign). Given one project it
behaves like a single observation; given many it fans steps out across them
(managed parallelism) and :meth:`merge` combines them after calibration. It
exposes the same callable namespaces as :class:`~vlbipy.observation.Observation`;
pre-merge steps fan out, while imaging/export operate on the merged (or sole)
observation.
"""
# from __future__ import annotations
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional, Union
from .errors import CampaignError, ConfigError, StaleCalibrationError
from .config import _deep_merge, kwargs_to_overrides, load_config
from .logging_utils import get_logger, headline, run_logs, warnings
from .observation import Observation
from .results import ImageSet

logger = get_logger()

ProjectLike = Union[str, list, tuple]


class _Fanout:
    """Proxy that fans a namespace call/attribute out over several observations.

    Calling the proxy calls each child namespace; accessing an attribute returns
    a function that dispatches that method to each child. Results are returned as
    a list, or unwrapped to a single value when there is only one observation.

    The proxy is fully discoverable in interactive sessions: ``__dir__`` exposes
    the public operations of the underlying namespace (so tab-completion works),
    unknown names raise :class:`AttributeError` immediately, dispatched methods
    keep the original name/docstring/signature (so IPython ``?`` help works),
    and ``repr`` lists the available operations.
    """

    def __init__(self, children: list, parallel: bool = False, on_failure=None) -> None:
        self._children = children
        self._parallel = parallel
        self._on_failure = on_failure

    def _each(self, call) -> list:
        """``[call(child) for child in children]``, in parallel threads when this fan-out allows it.

        Parallel calls share the solver worker pool (their jobs interleave there) and the shared
        progress display (each child contributes its own bars); CASA fallbacks serialize internally.

        One project's failure never stops the others: every child runs. Then, with an ``on_failure``
        handler (:meth:`VLBIObs.run`), each failed child's observation goes to it - the run drops that
        project and carries on with the rest - and the results of the others are returned. Without one,
        the first failure is raised once all children have run, the others noted on it.
        """
        children = object.__getattribute__(self, "_children")
        if not object.__getattribute__(self, "_parallel") or len(children) <= 1:
            outcomes = [_outcome(call, child) for child in children]
        else:
            logger.info("running {} projects in parallel", len(children))
            with ThreadPoolExecutor(max_workers=len(children), thread_name_prefix="vlbipy-fanout") as pool:
                outcomes = list(pool.map(lambda child: _outcome(call, child), children))
        failed = [(child, error) for child, (ok, error) in zip(children, outcomes) if not ok]
        if failed:
            on_failure = object.__getattribute__(self, "_on_failure")
            owners = [getattr(child, "_obs", None) for child, _ in failed]
            if on_failure is None or any(owner is None for owner in owners):
                first = failed[0][1]
                for child, error in failed[1:]:
                    first.add_note(f"{_child_code(child)} failed too: {error}")
                raise first
            for owner, (_, error) in zip(owners, failed):
                on_failure(owner, error)
        return [value for ok, value in outcomes if ok]

    def __call__(self, *args, **kwargs):
        results = self._each(lambda child: child(*args, **kwargs))
        return results[0] if len(results) == 1 else results

    def __getattr__(self, name: str):
        children = object.__getattribute__(self, "_children")
        template = getattr(type(children[0]), name, None)
        if template is None:
            available = ", ".join(self.operations())
            raise AttributeError(f"{type(children[0]).__name__!r} has no operation {name!r} "
                                 f"(available: {available})")
        if not callable(template):
            results = [getattr(child, name) for child in children]
            return results[0] if len(results) == 1 else results

        def _dispatch(*args, **kwargs):
            results = self._each(lambda child: getattr(child, name)(*args, **kwargs))
            return results[0] if len(results) == 1 else results

        functools.update_wrapper(_dispatch, getattr(children[0], name))
        return _dispatch

    def operations(self) -> list[str]:
        """Return the names of the callable operations the wrapped namespace offers."""
        return self._children[0].operations()

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | {n for n in dir(self._children[0]) if not n.startswith("_")})

    def __repr__(self) -> str:
        return f"<fan-out over {len(self._children)} observation(s); operations: {', '.join(self.operations())}>"


def _outcome(call, child) -> tuple:
    """``(True, call(child))``, or ``(False, exception)`` when it raised one."""
    try:
        return True, call(child)
    except Exception as exc:  # noqa: BLE001 - handed back to the fan-out, which decides
        return False, exc


def _child_code(child) -> str:
    """The project code of a fan-out child (a namespace of one observation), for messages."""
    return getattr(getattr(child, "_obs", None), "project_code", None) or repr(child)


def _string_list(value: str | list | tuple | None) -> list:
    """Normalise a str/list/tuple/None argument to a list (None/empty -> [])."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return list(value)


class VLBIObs:
    """A campaign of one or more VLBI projects — the object users construct.

    Parameters
    ----------
    project : str or list of str
        One project code, or several (a campaign).
    network : str, optional
        Observatory: ``"EVN"``, ``"VLBA"`` or ``"LBA"``.
    backend : str, optional
        Backend name (``"dummy"``, ``"casa"`` or ``"aips"``). If omitted, the
        config value is used (``"casa"`` by default). Others not implemented.
    target, phasecal, fringe_finder, check_source : str or list, optional
        Convenience source-role declarations.
    refant : str or list, optional
        Preferred reference antenna(s).
    mode : str
        Observing mode (``"continuum"`` by default).
    config : str or pathlib.Path or dict, optional
        User TOML config path or dict.
    work_dir : str or pathlib.Path, optional
        Base working directory.
    **overrides
        Extra nested-config overrides (deep-merged last).
    """

    def __init__(self, project: ProjectLike, *, network: Optional[str] = None, backend: Optional[str] = None,
                 target=None, phasecal=None, fringe_finder=None, check_source=None,
                 refant=None, mode: str = "continuum", config=None, work_dir=None, **overrides) -> None:
        codes: list[str] = [project] if isinstance(project, str) else list(project)
        if not codes:
            raise ValueError("At least one project code is required")

        kw_over = kwargs_to_overrides(
            network=network, backend=backend, mode=mode,
            refant=refant if refant else None,
            target=_string_list(target) or None,
            phasecal=_string_list(phasecal) or None,
            fringe_finder=_string_list(fringe_finder) or None,
            check_source=_string_list(check_source) or None,
            phaseref_strategy=overrides.pop("phaseref_strategy", None),
            inverse_phaseref=overrides.pop("inverse_phaseref", None),
            phaseref_order=overrides.pop("phaseref_order", None),
        )
        merged_overrides = _deep_merge(kw_over, overrides) if overrides else kw_over
        self.config = load_config(config, overrides=merged_overrides)
        self._reject_unknown_overrides(overrides)

        base_work = Path(work_dir) if work_dir else None
        self._adopt_saved_sources(codes, base_work)
        self._observations: list[Observation] = []
        for code in codes:
            obs_work = (base_work / code) if base_work and len(codes) > 1 else base_work
            self._observations.append(Observation(code, self.config, work_dir=obs_work))

        self._merged: Optional[Observation] = None
        #: Inside :meth:`run`: a project that fails is set aside here and the others go on.
        self._isolating = False
        self._failed: dict[str, tuple[Observation, Exception]] = {}
        self._save_project(codes, base_work)
        logger.info(f"VLBIObs {', '.join(codes)}")

    #: File that lets a later process (the notebook, ``vlbipy report``) rebuild this object.
    PROJECT_FILENAME = ".project.json"

    #: Class default for runners built without ``__init__`` (the epoch subprocess): failures propagate.
    _isolating = False

    def _save_project(self, codes: list[str], base_work) -> None:
        """Persist the project codes and the fully-merged configuration next to the products.

        The archive password (``[import].password``) is blanked in the saved copy: it is only needed to
        download, and the file sits beside the products where anyone given the run can read it.
        """
        if not self._observations or not self._observations[0]._backend.requires_data_files:
            return
        root = Path(base_work) if base_work else Path(self._observations[0].work_dir)
        if len(codes) > 1 and not base_work:
            root = Path(self._observations[0].work_dir).parent
        saved = dict(self.config)
        if saved.get("import", {}).get("password"):
            saved["import"] = {**saved["import"], "password": ""}
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / self.PROJECT_FILENAME).write_text(json.dumps(
                {"projects": codes, "config": saved, "work_dir": str(root)}, indent=2, default=str))
        except OSError as exc:
            logger.warning("could not write {}: {}", root / self.PROJECT_FILENAME, exc)

    def _project_root(self, codes: list[str], base_work) -> Path:
        """Where :meth:`_save_project` keeps the project file (the same rule, before the observations exist)."""
        if base_work:
            return Path(base_work)
        first = Path(self.config.get("global", {}).get("work_dir", "") or Path.cwd() / codes[0])
        return first.parent if len(codes) > 1 else first

    def _adopt_saved_sources(self, codes: list[str], base_work) -> None:
        """Reuse the source roles of the previous run of the same projects when none are declared.

        Roles given on the command line live only in the configuration this process builds, so resuming without
        them (``--from-step``) would calibrate with no fringe finder or phase calibrator - and then save that
        empty set over the one the earlier run recorded.
        """
        if any(self.config.get("sources", {}).get(key) for key in
               ("fringe_finders", "phase_calibrators", "targets", "check_sources")):
            return
        path = self._project_root(codes, base_work) / self.PROJECT_FILENAME
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if list(saved.get("projects") or []) != list(codes):
            return
        sources = (saved.get("config") or {}).get("sources") or {}
        if not any(sources.get(key) for key in ("fringe_finders", "phase_calibrators", "targets", "check_sources")):
            return
        self.config["sources"] = {**self.config.get("sources", {}), **sources}
        if (saved.get("config") or {}).get("phase_referencing"):
            self.config["phase_referencing"] = saved["config"]["phase_referencing"]
        logger.info("no source roles given: using those of the previous run ({})",
                    ", ".join(f"{key} {', '.join(value)}" for key, value in sources.items() if value))

    @classmethod
    def load(cls, work_dir) -> "VLBIObs":
        """Rebuild a VLBIObs from the ``.project.json`` a previous process left in ``work_dir``."""
        root = Path(work_dir)
        data = json.loads((root / cls.PROJECT_FILENAME).read_text())
        return cls(data["projects"], config=data["config"], work_dir=data.get("work_dir") or str(root))

    def _reject_unknown_overrides(self, overrides: dict) -> None:
        """Fail on a keyword that matches no configuration section.

        ``**overrides`` exists to set config sections (``imaging={...}``), but it
        also swallows misspelled keywords silently — and a typo like
        ``fringefinder=`` instead of ``fringe_finder=`` means the pipeline
        calibrates on the wrong source without ever saying so.
        """
        unknown = [key for key in (overrides or {}) if key not in self.config]
        if unknown:
            known = ", ".join(sorted(self.config))
            raise ConfigError(f"unknown argument(s): {', '.join(sorted(unknown))}. Source roles are "
                              f"target, phasecal, fringe_finder, check_source; configuration "
                              f"sections are {known}.")

    # -- construction helpers --
    @classmethod
    def from_config(cls, config, **kwargs) -> "VLBIObs":
        """Build a VLBIObs entirely from a TOML config (project list included)."""
        cfg = load_config(config)
        project = cfg.get("global", {}).get("project", "")
        projects = cfg.get("global", {}).get("projects", [])
        codes = projects or ([project] if project else [])
        if not codes:
            raise ConfigError("config must set [global].project or [global].projects")
        return cls(codes, config=config, **kwargs)

    # -- internal targets --
    def _primary(self) -> Observation:
        """Return the merged observation if present, else the first observation."""
        return self._merged if self._merged is not None else self._observations[0]

    def _imaging_target(self) -> Observation:
        """Return the observation imaging/export act on; require merge for campaigns."""
        if self._merged is not None:
            return self._merged
        if len(self._observations) == 1:
            return self._observations[0]
        raise Exception("call merge() before imaging/exporting a multi-project campaign")

    # -- one project's failure does not stop the others --
    def _fanout(self, namespace: str, parallel: bool = False) -> _Fanout:
        """A fan-out of ``namespace`` over the observations still in the run (see :meth:`_drop_failed`)."""
        return _Fanout([getattr(o, namespace) for o in self._observations], parallel=parallel,
                       on_failure=self._drop_failed if self._isolating else None)

    def _drop_failed(self, obs: Observation, error: Exception) -> None:
        """Leave a failed project out of the rest of :meth:`run`; the others carry on to the end.

        The step it had in progress (the last one started) is recorded as failed in its state file. When it was
        the last project still running there is nothing left to do: its error is raised (all the failures, when
        several).
        """
        if obs._state._started:
            obs._state.mark_failed(next(reversed(obs._state._started)), str(error))
        self._failed[obs.project_code] = (obs, error)
        self._observations = [o for o in self._observations if o is not obs]
        if not self._observations:
            if len(self._failed) == 1:
                raise error
            raise CampaignError({code: exc for code, (_, exc) in self._failed.items()}) from error
        logger.error("[{}] failed: {}", obs.project_code, error)
        headline("[{}] failed and is left out; {} carr{} on", obs.project_code,
                 ", ".join(o.project_code for o in self._observations),
                 "ies" if len(self._observations) == 1 else "y")

    def _each_project(self, call) -> None:
        """``call(obs)`` for every observation still in the run, a failure handled like a fan-out's."""
        for obs in list(self._observations):
            try:
                call(obs)
            except Exception as exc:  # noqa: BLE001 - the other projects go on
                if not self._isolating:
                    raise
                self._drop_failed(obs, exc)

    # -- fan-out namespaces (pre-merge, per project) --
    @property
    def import_data(self) -> _Fanout:
        """Import namespace, fanned out across all observations; a campaign's projects import in parallel."""
        return self._fanout("import_data", parallel=True)

    @property
    def calibrate(self) -> _Fanout:
        """Calibration namespace, fanned out across all observations."""
        return self._fanout("calibrate")

    @property
    def flag(self) -> _Fanout:
        """Flagging namespace, fanned out across all observations."""
        return self._fanout("flag")

    @property
    def plot(self) -> _Fanout:
        """Plotting namespace, fanned out across all observations."""
        return self._fanout("plot")

    @property
    def ai(self) -> _Fanout:
        """AI-mode namespace (diagnostics, Tsys repair, distribution flagging), fanned out across all observations."""
        return self._fanout("ai")

    # -- delegated namespaces (post-merge / single target) --
    @property
    def _per_epoch_products(self) -> bool:
        """True for a campaign on real data: every epoch is split, self-calibrated and imaged on its own."""
        return len(self._observations) > 1 and self._observations[0]._backend.requires_data_files

    @property
    def clean(self):
        """Imaging namespace: every epoch of a campaign, else the merged (or sole) observation."""
        if self._per_epoch_products:
            return self._fanout("clean")
        return self._imaging_target().clean

    @property
    def selfcal(self):
        """Self-cal namespace: every epoch of a campaign, else the merged (or sole) observation."""
        if self._per_epoch_products:
            return self._fanout("selfcal")
        return self._imaging_target().selfcal

    @property
    def export(self):
        """Export namespace: every epoch of a campaign, else the merged (or sole) observation."""
        if self._per_epoch_products:
            return self._fanout("export")
        return self._imaging_target().export

    @property
    def sources(self):
        """The shared source set (from the primary observation)."""
        return self._primary().sources

    def per_project(self, name: str):
        """Return one observation's attribute, or ``{project_code: value}`` for a campaign.

        Every read-only property below funnels through this, so a single-project
        campaign — the common case — reads exactly like the underlying
        :class:`~vlbipy.observation.Observation` (``obs.metadata``), while a
        multi-project one keeps the values labelled by project instead of
        collapsing them into an anonymous list.

        Parameters
        ----------
        name : str
            Attribute name on :class:`~vlbipy.observation.Observation`.
        """
        values = {o.project_code: getattr(o, name) for o in self._observations}
        return next(iter(values.values())) if len(values) == 1 else values

    @property
    def metadata(self):
        """Observation metadata (``None`` until :meth:`import_data` has run)."""
        return self.per_project("metadata")

    @property
    def antennas(self):
        """Antennas, keyed by name (empty until imported)."""
        return self.per_project("antennas")

    @property
    def scans(self):
        """Scan list (empty until imported)."""
        return self.per_project("scans")

    @property
    def frequency(self):
        """Frequency setup (``None`` until imported)."""
        return self.per_project("frequency")

    @property
    def gaintables(self):
        """Calibration tables accumulated so far, in application order."""
        return self.per_project("gaintables")

    @property
    def refant(self):
        """Reference antenna(s) in use."""
        return self.per_project("refant")

    @property
    def calibrator_field(self):
        """Field used for calibration: fringe finders, else phase cals, else targets."""
        return self.per_project("calibrator_field")

    @property
    def observatory(self):
        """The VLBI network."""
        return self.per_project("observatory")

    @property
    def work_dir(self):
        """Working directory holding the data and products."""
        return self.per_project("work_dir")

    @property
    def project_code(self):
        """Project code(s) in this campaign."""
        return self.per_project("project_code")

    @property
    def snr_survey(self):
        """The most recent per-scan fringe SNR survey (``None`` until measured)."""
        return self.per_project("snr_survey")

    @property
    def data(self):
        """Lazy dask-ms datasets (dask-ms backend only)."""
        return self.per_project("data")

    @property
    def observations(self) -> list[Observation]:
        """The list of per-project observations."""
        return list(self._observations)

    # -- campaign operations --
    def merge(self, *, force: bool = False) -> Observation:
        """Combine calibrated data across projects (no-op for a single project).

        Returns
        -------
        Observation
            The merged observation (or the sole observation for one project).
        """
        if len(self._observations) == 1:
            logger.info("merge requested, but it is already a single project")
            return self._observations[0]
        if self._per_epoch_products:
            raise Exception("the epochs of a campaign on real data are combined with combine() "
                            "(joint calibrator model, per-epoch refinement, combined images), not merge()")

        codes = [o.project_code for o in self._observations]
        src_names = self._observations[0].sources.names
        self._observations[0]._backend.export.merge(codes, src_names)
        merged_code = "+".join(codes)
        merged = Observation(merged_code, self.config)
        merged._metadata = self._observations[0]._metadata
        self._merged = merged
        logger.info("merged {} projects -> {}", len(codes), merged_code)
        return merged

    def combine(self, *, force: bool = False) -> dict:
        """Combine the calibrated epochs of a campaign (see :mod:`vlbipy.campaign`).

        The phase calibrators are modelled from all epochs together, every epoch
        is refined against that model and re-imaged, and the epochs of every
        source are imaged together. Needs each epoch calibrated, self-calibrated
        and split first (what :meth:`run` does before calling this).

        Returns
        -------
        dict
            The campaign report (also written to ``combined/<name>.campaign.json``).
        """
        if len(self._observations) == 1:
            logger.info("combine requested, but it is a single project")
            return {}
        for observation in self._observations:
            self._require_imaging_policy(observation)
        from .campaign import combine_epochs
        return combine_epochs(self, force=force)

    def run(self, *, force: bool = False, scratch: bool = False, from_step: str = "") -> dict:
        """Run the full default pipeline end to end.

        Resumes from the last completed step by default: progress is recorded in
        ``<work_dir>/.pipeline_state.json``, so an interrupted run continues
        rather than redoing hours of calibration.

        In a campaign, a project that fails is left out and the others run to the end; the run then raises
        :class:`~vlbipy.errors.CampaignError` naming the failed projects (the error itself when a lone
        project fails, or the last one still running).

        Parameters
        ----------
        force : bool
            Re-run steps even when they are already recorded as done.
        scratch : bool
            Start over — forget all progress and reset the data itself.
        from_step : str
            Re-run from this step onward (see ``observation.STEP_ORDER``).

        Returns
        -------
        dict
            Mapping of target source name -> produced image(s).
        """
        self._isolating, self._failed = True, {}
        try:
            images = self._run_steps(force=force, scratch=scratch, from_step=from_step)
        finally:
            self._isolating = False
            self._observations = self._observations + [obs for obs, _ in self._failed.values()]
        for obs, _ in self._failed.values():
            try:
                obs.report()                                           # its summary.md says how far it got
            except Exception as exc:  # noqa: BLE001 - the report of a failed project is best effort
                logger.warning("[{}] report failed: {}", obs.project_code, exc)
        summary = warnings.summary()
        logs = ", ".join(str(path) for path in run_logs()) or "the run log"
        failed = ", ".join(self._failed)
        headline("run finished{}: {} warning(s){}; details in {} and summary.md",
                 f" for {', '.join(o.project_code for o in self._observations if o.project_code not in self._failed)}"
                 f"; {failed} FAILED" if failed else "",
                 len(summary), " (see below)" if summary else "", logs)
        if summary:
            for item in summary:
                logger.warning("  - {}", item)
        if self._failed:
            raise CampaignError({code: exc for code, (_, exc) in self._failed.items()})
        return images

    def _run_steps(self, *, force: bool, scratch: bool, from_step: str) -> dict:
        """The steps of :meth:`run`, over the projects still running."""
        if not scratch:
            for observation in self._observations:
                self._require_imaging_policy(observation)
        for observation in self._observations:
            observation.prepare_run(scratch=scratch, from_step=from_step)

        # The sequence follows the reduce-vlbi-data procedure step by step.
        self.import_data(force=force)                                  # steps 1-3
        self._split_frequency_bands(force=force)                       # multi-band (S/X): one pipeline per band
        self._each_project(lambda obs: obs.notebook())                 # first notebook: what is there so far
        self.flag.apriori(force=force)                                 # step 4
        # AI mode: what the raw data (a-priori flags only) say about every antenna and Tsys series; the Tsys
        # repair of the next step follows its plan.
        self.ai.prediagnose(force=force)
        self.calibrate.a_priori(force=force)                           # steps 5-6 (EOP, Tsys, GC)
        self.plot.diagnostics(column="data", label="raw", force=force)  # step 7
        self.flag.quack(force=force)                                   # step 7 (slewing)
        self.flag.initial(force=force)                                 # step 8
        # Steps 9-11: SBD -> MBD -> bandpass -> SBD -> MBD, then trim the band edges the
        # bandpass has revealed and put the instrumental chain onto every field.
        # Every application before the last one is deferred where the backend can (dask-ms): the
        # steps in between read the corrected data through readers that calibrate DATA as they
        # read it, and CORRECTED_DATA is written once, with the final chain (write_corrected).
        # A receptor much weaker than its pair goes first, so no solve averages it in.
        self.flag.weak_polarizations(force=force)
        self.calibrate.instrumental(force=force)
        self.flag.edges(force=force)
        self.calibrate.apply(force=True, defer=True)
        # Step 12: deeper flagging on calibrated data, then re-solve on the cleaner data
        # and level the subband amplitudes.
        # The band edges (and the bandpass gaps) are measured again on the bandpass of the last pass,
        # the one applied: it is solved on more heavily flagged data than the first, and channels it no
        # longer solves would otherwise be extrapolated by the apply rather than flagged.
        reweight = self.config.get("calibration", {}).get("reweight", {}).get("enabled", True)
        self.flag.outliers(force=force)
        self.calibrate.second_pass(force=force)
        if not reweight:
            self.flag.edges(force=force, step="flag_edges_final")
        self.calibrate.scalar_bandpass(force=force)
        self.calibrate.apply(force=True, defer=True)                   # step 13
        # Step 14: weights from the calibrated scatter expose more bad data; flag it and
        # re-run the whole chain from the a-priori tables.
        if reweight:
            self.calibrate.reweight(force=force)
            self.flag.outliers(force=force, step="flag_outliers_reweighted")
            self.calibrate.second_pass(force=force, step="third_pass")
            self.flag.edges(force=force, step="flag_edges_final")
            self.calibrate.scalar_bandpass(force=force)
            self.calibrate.apply(force=True, defer=True)
        # e-MERLIN flux scale: a model-based amplitude correction of the outstations,
        # appended to the chain before the split sees it.
        def emerlin_scale(obs: Observation) -> None:
            if obs.calibrate.emerlin_scale(force=force):
                obs.calibrate.apply(force=True, defer=True)

        self._each_project(emerlin_scale)
        # AI mode: whatever the calibrated data show outside the noise distribution (Rice amplitudes, von Mises
        # phases, one level for every subband) is flagged before self-calibration and imaging see it.
        self._each_project(lambda obs: obs.ai.flag_distribution(force=force)
                           if obs.ai.enabled and obs.config.get("ai", {}).get("flag_distribution", True) else None)
        self._survey_all_sources()                                      # final calibrated SNR coverage
        if self._per_epoch_products:
            images = self._finish_campaign(force=force)
        else:
            if len(self._observations) > 1:
                self.calibrate.write_corrected()                       # merge copies the column itself
                self.merge(force=force)
            # Step 16: calibrated, per-source measurement sets — the deliverable of the
            # calibration. After merge() for a campaign, so the split covers the combined data.
            if len(self._observations) > 1:
                self.export.per_source(force=force)
                self.flag.statistics()                                     # step 20 (report)
                self._selfcal_all(force=force)                             # step 17a (difmapy)
            else:
                self.flag.statistics()                                     # step 20 (report)
                self._selfcal_then_split(force=force)                      # steps 17a and 16
            self._calibrated_diagnostics()                                 # step 15
            images = self._image_all(force=force)                          # step 17b
            self._final_plots()                                            # step 18
            # A structured phase calibrator: its model goes into the fringe fit, and everything after it is
            # derived again (and self-calibrated against that model).
            images = self._model_calibration(force=force) or images
        for obs in self._observations:
            if obs.ai.enabled:
                try:
                    obs.ai.postdiagnose()                              # residual antenna gains of the result
                except Exception as exc:  # noqa: BLE001 - a report never fails the run
                    warnings.warn(f"{obs.project_code}: AI post-diagnostic failed: {exc}")
        self.report()
        return images

    def _model_calibration(self, obs: Optional[Observation] = None, *, force: bool = False) -> dict:
        """The model pass: re-derive the calibration with the calibrators' structure models in the solves.

        The first pass takes every calibrator as a point source in the fringe fit, and self-calibrates it against
        one Gaussian. When the phase calibrator turns out to be structured (its structure model, fitted at the end
        of its self-calibration, has visibility phases of ``[calibration.model_mbd].min_phase_deg`` or more on its
        baselines), its structure phase went into the delays and rates - and the phase self-cal - transferred to
        the targets. Then the calibrators' models are written to MODEL_DATA, the chain is solved again from the
        delays on (:meth:`CalibrateNamespace.model_pass`), the calibrators are self-calibrated against their
        structure models (``model_components``), and every source is imaged again. The first pass's figures of
        merit and the new ones are written side by side to ``<work_dir>/ai/<code>.model_pass.json``.

        ``[calibration.model_mbd].enabled``: ``"auto"`` (the default: only for a structured phase calibrator),
        ``true`` (always, with whatever models there are) or ``false``. Returns the new images, or ``{}`` when the
        pass did not run.
        """
        obs = obs or self._imaging_target()
        cfg = obs.config.get("calibration", {}).get("model_mbd", {})
        mode = str(cfg.get("enabled", "auto")).strip().lower()
        if mode in ("false", "0", "no", "off", "") or not obs._backend.requires_data_files:
            return {}
        if not obs._state.should_run("model_pass", force=force):
            return {}
        try:
            structure = obs.calibrate.calibrator_structure()
        except Exception as exc:  # noqa: BLE001 - the first pass is complete and stands without it
            warnings.warn(f"{obs.project_code}: could not measure the calibrators' structure ({exc}); "
                          "no model pass")
            return {}
        phasecals = {n: v for n, v in structure.items() if v["role"] == "phasecal"}
        if mode == "auto" and not any(v["structured"] for v in phasecals.values()):
            logger.info("model_mbd: no structured phase calibrator ({}); the point-source fringe fit stands",
                        ", ".join(f"{n}: {v['phase_p90']:.1f} deg" for n, v in phasecals.items()) or "no model")
            obs._state.mark_complete("model_pass", outputs=["not needed"])
            return {}
        models = {n: v["components"] for n, v in structure.items()
                  if mode != "auto" or v["structured"] or v["role"] == "phasecal"}
        if not models:
            return {}
        headline("model pass: {} in the fringe fit and the self-calibration",
                 ", ".join(f"{n} ({len(c)} components)" for n, c in models.items()))
        before = self._figures_of_merit(obs)
        components = int(cfg.get("components", 3))
        obs.calibrate.model_pass(models, force=True)
        obs.selfcal.run_all(force=True, model_components=components)
        self._calibrated_diagnostics(obs)
        images = self._image_all(obs, force=True)
        self._final_plots(obs)
        after = self._figures_of_merit(obs)
        record = {"structure": {n: {k: v for k, v in s.items() if k != "components"} | {"ncomp": len(s["components"])}
                                for n, s in structure.items()},
                  "models": {n: c for n, c in models.items()}, "before": before, "after": after}
        out = Path(obs.work_dir) / "ai" / f"{obs.project_code}.model_pass.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        for name in sorted(set(before) & set(after)):
            b, a = before[name], after[name]
            logger.info("model pass: {} peak {} -> {} Jy/beam, rms {} -> {} Jy/beam, dynamic range {} -> {}", name,
                        _num(b.get("peak")), _num(a.get("peak")), _num(b.get("rms")), _num(a.get("rms")),
                        _num(b.get("dynamic_range")), _num(a.get("dynamic_range")))
        return images

    @staticmethod
    def _figures_of_merit(obs: Observation) -> dict:
        """Peak, rms and dynamic range of every source's robust-0 image (from the imaging records), and the
        reduced chi-squared of the calibrators' self-calibrated models."""
        work = Path(obs.work_dir)
        out: dict = {}
        for path in sorted((work / "images").glob(f"{obs.project_code}_*.search.json")):
            name = path.name[len(obs.project_code) + 1:-len(".search.json")]
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            images = record.get("images") or {}
            image = images.get("0.0") or images.get("0") or next(iter(images.values()), {}) \
                if isinstance(images, dict) else {}
            search = record.get("search") or {}
            out[name] = {"peak": image.get("peak"), "rms": image.get("rms"),
                         "dynamic_range": image.get("dynamic_range"), "search_snr": search.get("snr"),
                         "search_peak": search.get("peak"),
                         "model_rchisq": (image.get("model") or {}).get("rchisq") if isinstance(image, dict) else None}
        for name, result in obs._selfcal_results.items():
            report = getattr(result, "report", None) or {}
            entry = out.setdefault(name, {})
            entry["model_rchisq"] = (report.get("image_model") or {}).get("rchisq")
            entry["ncomp"] = len((report.get("image_model") or {}).get("components") or [])
        return out

    def _split_frequency_bands(self, *, force: bool = False) -> None:
        """Expand every multi-band observation into one observation per band.

        A data set whose subbands form several widely separated frequency bands
        (e.g. simultaneous S/X) is calibrated band by band: each band is split
        into ``<code>_<band>/<code>_<band>.ms`` (DATA column, every field) and a
        new :class:`Observation` takes its place, marked ``_band_of`` the parent
        so the campaign combination is skipped for them later.
        """
        expanded: list[Observation] = []
        for obs in self._observations:
            freq = obs.metadata.freq_setup if obs.metadata and obs.metadata.freq_setup else None
            bands = freq.bands if freq else []
            if len(bands) <= 1:
                expanded.append(obs)
                continue
            labels = [freq.band_label(spws) for spws in bands]
            warnings.warn(f"{obs.project_code}: the subbands form {len(bands)} frequency bands "
                          f"({', '.join(labels)}); each band is calibrated independently as "
                          + ", ".join(f"{obs.project_code}_{label}" for label in labels))
            for spws, label in zip(bands, labels):
                band_code = f"{obs.project_code}_{label}"
                work = Path(obs.work_dir).parent / band_code
                work.mkdir(parents=True, exist_ok=True)
                band_ms = work / f"{band_code}.ms"
                if band_ms.exists() and not force:
                    logger.info("band split: {} already exists", band_ms)
                else:
                    obs._backend.export.split_band(obs.project_code, list(spws), str(band_ms))
                band_obs = Observation(band_code, obs.config, work_dir=work)
                band_obs._band_of = obs.project_code
                band_obs._backend.data.adopt_ms(band_code, str(band_ms))
                band_obs._metadata = band_obs._backend.data.get_metadata(
                    band_code, [], obs.observatory)
                band_obs._state.mark_complete("import_data", outputs=[str(band_ms)])
                expanded.append(band_obs)
            logger.info("{}: split into {} bands -> {}", obs.project_code, len(bands),
                        ", ".join(o.project_code for o in expanded[-len(bands):]))
        self._observations = expanded

    def _require_imaging_policy(self, obs: Observation) -> None:
        """Stop before changing products when a real difmap epoch still carries legacy calibration."""
        imager = getattr(obs._backend, "imager", None) or obs.config.get("imaging", {}).get("imager", "difmap")
        if obs._backend.requires_data_files and imager == "difmap":
            from .backends.difmap import imaging_settings
            from .namespaces import _require_current_selfcal
            _require_current_selfcal(obs, imaging_settings(obs.config.get("imaging", {})))

    def _finish_campaign(self, *, force: bool = False) -> dict:
        """Steps 16-21 for a campaign: split, self-calibrate and image every epoch, then combine them.

        Returns ``{project_code: {source: images}}`` plus the campaign report under ``"combined"``.
        """
        if all(obs._backend.requires_data_files and (getattr(obs._backend, "imager", "") or
               obs.config.get("imaging", {}).get("imager", "difmap")) == "difmap"
               for obs in self._observations):
            from .epoch_jobs import run_epochs
            failed: dict = {}
            images = run_epochs(self._observations, force=force, failed=failed if self._isolating else None)
            for obs in [o for o in self._observations if o.project_code in failed]:
                self._drop_failed(obs, RuntimeError(f"{obs.project_code}: epoch products failed: "
                                                    f"{failed[obs.project_code]}"))
        else:
            images = {}

            def finish(obs: Observation) -> None:
                images[obs.project_code] = self._finish_epoch(obs, force=force)

            self._each_project(finish)
        if len(self._observations) < 2:
            logger.info("campaign: only {} left; no multi-epoch combination",
                        ", ".join(o.project_code for o in self._observations))
            return images
        try:
            # Band splits share one epoch: combining them into a joint model would be wrong.
            if not all(getattr(o, "_band_of", None) for o in self._observations):
                images["combined"] = self.combine(force=force)
        except Exception as exc:  # noqa: BLE001 - the per-epoch products are complete and must survive
            warnings.warn(f"multi-epoch combination failed: {exc}")
        return images

    def _finish_epoch(self, obs: Observation, *, force: bool = False) -> dict:
        """Finish one epoch in dependency order; independent epochs may run in isolated processes."""
        self._require_imaging_policy(obs)
        obs.flag.statistics()
        self._selfcal_then_split(obs, force=force)
        self._calibrated_diagnostics(obs)
        images = self._image_all(obs, force=force)
        self._final_plots(obs)
        return images

    def _calibrated_diagnostics(self, obs: Optional[Observation] = None) -> None:
        """Step 15, the diagnostic plots of the calibrated data, on CORRECTED_DATA written with the final chain.

        They come after self-calibration: the corrected column is written once, at the end of the
        calibration (a backend that defers its applications writes it here), and the jplotter figures read
        it from disk.
        """
        def diagnostics(each: Observation) -> None:
            each.calibrate.write_corrected()
            each.plot.diagnostics(column="corrected", label="calibrated")

        if obs is not None:
            diagnostics(obs)
        else:
            self._each_project(diagnostics)

    def _survey_all_sources(self) -> None:
        """Measure final per-scan SNR on every observed source without changing solve selection."""
        for obs in self._observations:
            names = list(obs.metadata.source_names) if obs.metadata else []
            if not names:
                continue
            selection = ",".join(names)
            saved_selection = obs.cal_selection
            saved_stages = obs.cal_stages
            try:
                obs.calibrate.scan_snr(field=selection, force=True, max_scans=0, warn_dead=False)
            except Exception as exc:  # noqa: BLE001 - final diagnostics must not abort calibration
                warnings.warn(f"{obs.project_code}: all-source calibrated SNR survey failed ({exc})")
            finally:
                # scan_snr should not alter this, but make the invariant explicit: the
                # comprehensive report survey must never replace instrumental choices.
                if obs.cal_selection != saved_selection:
                    if saved_selection is None:
                        obs.clear_cal_selection()
                    else:
                        obs.set_cal_selection(*saved_selection, stages=saved_stages)

    def imaging_sources(self, obs: Optional[Observation] = None) -> list[str]:
        """Return the sources the pipeline images: ``[imaging].sources`` = ``"all"`` or ``"targets"``.

        ``"all"`` is every declared source that has data (calibrators included);
        sources declared in the config but absent from the metadata are skipped.
        """
        obs = obs or self._imaging_target()
        which = str(self.config.get("imaging", {}).get("sources", "all")).lower()
        names = [s.name for s in obs.sources.targets] if which == "targets" else list(obs.sources.names)
        if obs.metadata is not None and obs.metadata.source_names:
            names = [n for n in names if n in set(obs.metadata.source_names)]
        return names

    def _selfcal_then_split(self, obs: Optional[Observation] = None, *, force: bool = False) -> dict:
        """Self-calibrate (step 17a), then make sure the per-source products exist (step 16).

        The self-calibration stage ends by applying the whole chain and splitting every source again
        (``selfcal.run_all``), and each calibrator is split on its own before its session; a full split
        before it would only be thrown away (half a minute on an LBA epoch). It is made here only when the
        stage produced nothing: disabled, no on-disk data for difmapy, or failed.
        """
        obs = obs or self._imaging_target()
        results = self._selfcal_all(obs, force=force)
        if not results:
            obs.export.per_source(force=force)
        return results

    def _selfcal_all(self, obs: Optional[Observation] = None, *, force: bool = False) -> dict:
        """Self-calibrate the calibrators with difmapy and transfer the gains (``selfcal.run_all``).

        A failure here is a warning recorded as a failed ``selfcal`` step: the
        CASA calibration and the split products already exist and are still
        worth imaging.
        """
        obs = obs or self._imaging_target()
        try:
            return obs.selfcal.run_all(force=force)
        except StaleCalibrationError:
            raise
        except Exception as exc:  # noqa: BLE001 - self-cal must not abort the run
            obs._state.mark_failed("selfcal", str(exc))
            warnings.warn(f"{obs.project_code}: self-calibration stage failed: {exc}")
            return {}

    def _image_all(self, obs: Optional[Observation] = None, *, force: bool = False, reimage: bool = False) -> dict:
        """Image every source of :meth:`imaging_sources` at the configured robust values.

        Imaging is the last stage and must not throw away the calibration the
        run just produced (split MSs and UVFITS are already on disk): a clean that
        fails on one source is a warning recorded as a failed ``clean_<source>``
        step, not a failed run. A backend without imaging skips the step.
        """
        obs = obs or self._imaging_target()
        images: dict = {}
        imager = getattr(obs._backend, "imager", None) or self.config.get("imaging", {}).get(
            "imager", obs.clean.default_imager)
        difmap_imager = str(imager) == "difmap" and obs._backend.requires_data_files
        if not difmap_imager and not obs._backend.supports("image", "clean"):
            logger.warning("imaging skipped: backend {} does not implement image.clean", obs._backend.kind)
            return images
        robust = self.config.get("imaging", {}).get("robust", [-2, 0, 2])
        for name in self.imaging_sources(obs):
            # ``reimage``: the calibration changed since the self-cal session made its images.
            if not reimage and name in obs._selfcal_results and obs._selfcal_results[name].images:
                images[name] = ImageSet(obs._selfcal_results[name].images)   # imaged in its difmapy session
                obs._state.mark_complete(f"clean_{name}", outputs=[i.paths.get("fits", "") for i in images[name]])
                continue
            if not obs._state.should_run(f"clean_{name}", force=force):
                continue
            try:
                images[name] = obs.clean(target=name, robust=robust)
            except StaleCalibrationError:
                raise
            except Exception as exc:  # noqa: BLE001 - imaging failure must not abort the run
                obs._state.mark_failed(f"clean_{name}", str(exc))
                warnings.warn(f"imaging {name} failed: {exc}")
        return images

    def _final_plots(self, obs: Optional[Observation] = None) -> list[str]:
        """Produce the final-data plot set (``plot.final_data``), recorded as the ``final_plots`` step."""
        obs = obs or self._imaging_target()
        try:
            written = obs.plot.final_data()
        except Exception as exc:  # noqa: BLE001 - plotting must never abort the run
            obs._state.mark_failed("final_plots", str(exc))
            warnings.warn(f"{obs.project_code}: final-data plots failed: {exc}")
            return []
        obs._state.mark_complete("final_plots", outputs=written)
        return written

    def reset(self) -> None:
        """Clear state on every observation and drop any merge."""
        for obs in self._observations:
            obs.reset()

        self._merged = None

    clean_state = reset

    def listobs(self, listfile=None):
        """Return the scan/field listing(s) of the imported data (per observation)."""
        results = [o.listobs(listfile=listfile) for o in self._observations]
        return results[0] if len(results) == 1 else results

    def notebook(self, embed: bool = True) -> list[str]:
        """Write the interactive reduction notebook of every observation (see :mod:`vlbipy.notebook`).

        ``embed=True`` stores the plots in each notebook (self-contained); ``False`` links them by relative path.
        """
        return [o.notebook(embed=embed) for o in self._observations]

    def report(self) -> list[dict]:
        """Return per-observation report dicts (no files written in dummy mode)."""
        return [o.report() for o in self._observations]

    def summary(self) -> str:
        """Return a combined human-readable summary of all observations."""
        return "\n\n".join(o.summary() for o in self._observations)

    @property
    def state(self) -> dict:
        """Aggregate step state, keyed by project code."""
        return {o.project_code: o.state.as_dict() for o in self._observations}

    # -- container protocol --
    def __getitem__(self, code: str) -> Observation:
        for obs in self._observations:
            if obs.project_code == code:
                return obs

        raise KeyError(f"no project {code!r} (have: {', '.join(o.project_code for o in self._observations)})")

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._observations)

    def __len__(self) -> int:
        return len(self._observations)

    def __repr__(self) -> str:
        codes = ", ".join(o.project_code for o in self._observations)
        return f"VLBIObs([{codes}], backend={self._primary()._backend.kind})"


def _num(value) -> str:
    """A figure of merit for the log, or ``-``."""
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "-"
