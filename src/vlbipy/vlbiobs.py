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
from pathlib import Path
from typing import Iterator, Optional, Union
from .errors import ConfigError
from .config import _deep_merge, kwargs_to_overrides, load_config
from .logging_utils import get_logger, warnings
from .observation import Observation

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

    def __init__(self, children: list) -> None:
        self._children = children

    def __call__(self, *args, **kwargs):
        results = [child(*args, **kwargs) for child in self._children]
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
            results = [getattr(child, name)(*args, **kwargs) for child in children]
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
        )
        merged_overrides = _deep_merge(kw_over, overrides) if overrides else kw_over
        self.config = load_config(config, overrides=merged_overrides)
        self._reject_unknown_overrides(overrides)

        base_work = Path(work_dir) if work_dir else None
        self._observations: list[Observation] = []
        for code in codes:
            obs_work = (base_work / code) if base_work and len(codes) > 1 else base_work
            self._observations.append(Observation(code, self.config, work_dir=obs_work))

        self._merged: Optional[Observation] = None
        logger.info(f"VLBIObs {', '.join(codes)}")

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

    # -- fan-out namespaces (pre-merge, per project) --
    @property
    def import_data(self) -> _Fanout:
        """Import namespace, fanned out across all observations."""
        return _Fanout([o.import_data for o in self._observations])

    @property
    def calibrate(self) -> _Fanout:
        """Calibration namespace, fanned out across all observations."""
        return _Fanout([o.calibrate for o in self._observations])

    @property
    def flag(self) -> _Fanout:
        """Flagging namespace, fanned out across all observations."""
        return _Fanout([o.flag for o in self._observations])

    @property
    def plot(self) -> _Fanout:
        """Plotting namespace, fanned out across all observations."""
        return _Fanout([o.plot for o in self._observations])

    # -- delegated namespaces (post-merge / single target) --
    @property
    def clean(self):
        """Imaging namespace on the merged (or sole) observation."""
        return self._imaging_target().clean

    @property
    def selfcal(self):
        """Self-cal namespace on the merged (or sole) observation."""
        return self._imaging_target().selfcal

    @property
    def export(self):
        """Export namespace on the merged (or sole) observation."""
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

        codes = [o.project_code for o in self._observations]
        src_names = self._observations[0].sources.names
        self._observations[0]._backend.export.merge(codes, src_names)
        merged_code = "+".join(codes)
        merged = Observation(merged_code, self.config)
        merged._metadata = self._observations[0]._metadata
        self._merged = merged
        logger.info("merged {} projects -> {}", len(codes), merged_code)
        return merged

    def run(self, *, force: bool = False, scratch: bool = False, from_step: str = "") -> dict:
        """Run the full default pipeline end to end.

        Resumes from the last completed step by default: progress is recorded in
        ``<work_dir>/.pipeline_state.json``, so an interrupted run continues
        rather than redoing hours of calibration.

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
        for observation in self._observations:
            observation.prepare_run(scratch=scratch, from_step=from_step)

        # The sequence follows the reduce-vlbi-data procedure step by step.
        self.import_data(force=force)                                  # steps 1-3
        self.flag.apriori(force=force)                                 # step 4
        self.calibrate.a_priori(force=force)                           # steps 5-6 (EOP, Tsys, GC)
        self.plot.diagnostics(column="data", label="raw")              # step 7
        self.flag.quack(force=force)                                   # step 7 (slewing)
        self.flag.initial(force=force)                                 # step 8
        # Steps 9-11: SBD -> MBD -> bandpass -> SBD -> MBD, then trim the band edges the
        # bandpass has revealed and put the instrumental chain onto every field.
        self.calibrate.instrumental(force=force)
        self.flag.edges(force=force)
        self.calibrate.apply(force=True)
        # Step 12: deeper flagging on calibrated data, then re-solve on the cleaner data
        # and level the subband amplitudes.
        self.flag.outliers(force=force)
        self.calibrate.second_pass(force=force)
        self.calibrate.scalar_bandpass(force=force)
        self.calibrate.apply(force=True)                               # step 13
        # Step 14: weights from the calibrated scatter expose more bad data; flag it and
        # re-run the whole chain from the a-priori tables.
        if self.config.get("calibration", {}).get("reweight", {}).get("enabled", True):
            self.calibrate.reweight(force=force)
            self.flag.outliers(force=force, step="flag_outliers_reweighted")
            self.calibrate.second_pass(force=force, step="third_pass")
            self.calibrate.scalar_bandpass(force=force)
            self.calibrate.apply(force=True)
        self.plot.diagnostics(column="corrected", label="calibrated")  # step 15
        if len(self._observations) > 1:
            self.merge(force=force)
        # Step 16: calibrated, per-source measurement sets — the deliverable of the
        # calibration. After merge() for a campaign, so the split covers the combined data.
        self.export.per_source(force=force)
        self.flag.statistics()                                         # step 20 (report)

        robust = self.config.get("imaging", {}).get("robust", [0])
        images = {}
        # Imaging is optional: a backend without it must not throw away the
        # calibration a long run just produced.
        if self._imaging_target()._backend.supports("image", "clean"):
            for tgt in self.sources.targets:
                # Imaging is the last stage and must not throw away the calibration the
                # run just produced (split MSs and UVFITS are already on disk): a clean
                # that fails on one target is a warning, not a failed run.
                try:
                    images[tgt.name] = self.clean(target=tgt.name, robust=robust)
                except Exception as exc:  # noqa: BLE001 - imaging failure must not abort the run
                    warnings.warn(f"imaging {tgt.name} failed: {exc}")
        else:
            logger.warning("imaging skipped: backend {} does not implement image.clean yet",
                           self._imaging_target()._backend.kind)

        self.report()
        summary = warnings.summary()
        if summary:
            logger.warning("run finished with {} warning(s):", len(summary))
            for item in summary:
                logger.warning("  - {}", item)

        return images

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
