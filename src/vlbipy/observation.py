"""The per-project observation object.

``Observation`` is the internal unit held inside :class:`~vlbipy.vlbiobs.VLBIObs`
(one per project code). It owns that project's backend, observatory handler,
sources, step state, and metadata, and hosts the callable operation namespaces
(``import_data``, ``calibrate``, ``flag``, ``plot``, ``clean``, ``selfcal``,
``export``). Users normally drive it through :class:`VLBIObs`, but it is exposed
via ``vlbiobs["CODE"]`` / ``vlbiobs.observations`` for power users.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .backends import Backend, get_backend
from .errors import StepError
from .logging_utils import add_file_log, get_logger, warnings
from .models import CalTable, ObsMetadata
from .namespaces import (CalibrateNamespace, CleanNamespace, ExportNamespace, FlagNamespace,
                         ImportDataNamespace, PlotNamespace, SelfcalNamespace)
from .observatories import get_observatory_handler
from .sources import SourceSet
from .state import StepState

logger = get_logger()

#: File holding the calibration chain, so a resumed process can rebuild it.
CALTABLES_FILENAME = ".caltables.json"

#: Pipeline steps in execution order. Resuming from a step invalidates it and
#: everything after it, so this must match the order :meth:`VLBIObs.run` runs them
#: in and use the same names the steps record themselves under.
STEP_ORDER = [
    "import_data", "a_priori", "flag_apriori", "flag_from_file", "flag_autocorr",
    "flag_quack", "flag_tfcrop",
    "scan_snr", "initial_calibration", "fringefit", "bandpass",
    "initial_calibration_sbd2", "fringefit_mbd2", "flag_edges",
    "apply", "flag_aoflagger", "flag_outliers",
    "second_pass", "scalar_bandpass", "reweight", "flag_outliers_reweighted", "third_pass", "split",
]


class Observation:
    """A single VLBI observation (one project code).

    Parameters
    ----------
    project_code : str
        The project code.
    config : dict
        Fully-merged configuration (see :func:`vlbipy.config.load_config`).
    backend : Backend, optional
        A backend instance; if omitted, one is created from
        ``config['global']['backend']``.
    work_dir : str or pathlib.Path, optional
        Working directory; defaults to ``config['global']['work_dir']`` or
        ``./<project_code>``.
    """

    def __init__(self, project_code: str, config: dict, backend: Optional[Backend] = None,
                 work_dir=None) -> None:
        self.project_code = project_code
        self.config = config
        self.observatory = config.get("global", {}).get("observatory", "EVN")
        cfg_work = config.get("global", {}).get("work_dir", "")
        self.work_dir = str(work_dir or cfg_work or Path.cwd() / project_code)

        self._backend = backend or get_backend(config.get("global", {}).get("backend", "dummy"),
                                               work_dir=self.work_dir)
        self._observatory_handler = get_observatory_handler(self.observatory)
        self.sources = SourceSet.from_config(config.get("sources", {}), config.get("phase_referencing"))
        # State is persisted only for backends with on-disk products: resuming means
        # picking up artifacts a previous process wrote, and the dummy backend has none.
        self._state = StepState(project_code,
                                work_dir=self.work_dir if self._backend.requires_data_files else None)
        if self._backend.requires_data_files:
            add_file_log(Path(self.work_dir) / "logs", project_code)
        self._metadata: Optional[ObsMetadata] = None
        # The apply chain is persisted: resuming mid-pipeline in a fresh process must
        # solve on top of the tables earlier steps produced, not from scratch.
        self.gaintables: list[CalTable] = self._load_gaintables()
        self._snr_surveys: dict = {}   # per-field fringe SNR surveys (see plot/calibrate.scan_snr)
        self.flag_statistics: dict = {}  # last flag.statistics() result (per antenna / subband)
        self._scratch = False

        # Callable operation namespaces.
        self.import_data = ImportDataNamespace(self)
        self.calibrate = CalibrateNamespace(self)
        self.flag = FlagNamespace(self)
        self.plot = PlotNamespace(self)
        self.clean = CleanNamespace(self)
        self.selfcal = SelfcalNamespace(self)
        self.export = ExportNamespace(self)

    # -- the calibration chain --
    @property
    def _caltables_path(self) -> Optional[Path]:
        """Path of the persisted calibration chain, or ``None`` for in-memory backends."""
        if not self._backend.requires_data_files:
            return None
        return Path(self.work_dir) / CALTABLES_FILENAME

    def _load_gaintables(self) -> list[CalTable]:
        """Read the persisted calibration chain; a missing or corrupt file yields none."""
        path = self._caltables_path
        if path is None or not path.is_file():
            return []
        try:
            tables = [CalTable.from_dict(entry) for entry in json.loads(path.read_text())]
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("[{}] could not read {} ({}); the calibration chain starts empty",
                           self.project_code, path.name, exc)
            return []
        if tables:
            logger.info("[{}] calibration chain: {}", self.project_code,
                        " -> ".join(t.cal_type for t in tables))
        return tables

    def _save_gaintables(self) -> None:
        """Write the calibration chain (no-op for backends without on-disk products)."""
        path = self._caltables_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([t.to_dict() for t in self.gaintables], indent=2))
        except OSError as exc:
            logger.warning("[{}] could not write {} ({}); a resumed run will re-derive "
                           "the calibration chain", self.project_code, path.name, exc)

    def add_gaintable(self, table: CalTable, step: str) -> CalTable:
        """Append a table to the apply chain, tagging it with the step that made it."""
        table.step = step
        self.gaintables.append(table)
        self._save_gaintables()
        return table

    def set_gaintables(self, tables: list[CalTable]) -> list[CalTable]:
        """Replace the whole apply chain (used when a pass re-derives its tables)."""
        self.gaintables[:] = list(tables)
        self._save_gaintables()
        return self.gaintables

    def drop_gaintables(self, steps) -> list[str]:
        """Drop the tables produced by the given steps; return the cal types removed."""
        steps = set(steps)
        dropped = [t.cal_type for t in self.gaintables if t.step in steps]
        if dropped:
            self.gaintables[:] = [t for t in self.gaintables if t.step not in steps]
            self._save_gaintables()
        return dropped

    # -- read-only accessors --
    @property
    def metadata(self) -> Optional[ObsMetadata]:
        """Observation metadata (populated by :meth:`import_data`)."""
        return self._metadata

    @property
    def antennas(self) -> dict:
        """Antenna dict from metadata (empty until data is imported)."""
        return self._metadata.antennas if self._metadata else {}

    @property
    def frequency(self):
        """Frequency setup from metadata (``None`` until imported)."""
        return self._metadata.freq_setup if self._metadata else None

    @property
    def scans(self) -> list:
        """Scan list from metadata (empty until imported)."""
        return self._metadata.scans if self._metadata else []

    @property
    def data(self) -> list:
        """The imported visibilities as lazy dask-ms datasets (dask-ms backend only)."""
        return self._backend.data.datasets(self.project_code)

    @property
    def state(self) -> StepState:
        """The step-state tracker."""
        return self._state

    @property
    def snr_survey(self):
        """The most recent per-scan fringe SNR survey (``None`` until measured)."""
        return self._metadata.snr_survey if self._metadata else None

    @property
    def refant(self) -> str:
        """Reference antenna(s): configured preference, else the most sensitive antenna.

        Once a fringe SNR survey exists the ranking it produced decides, so
        diagnostics reference the same antenna the calibration solved against.
        Falling back to the first antenna in the table (as before any survey)
        picks whichever station happens to be listed first, which is arbitrary.
        """
        from .selection import rank_reference_antennas

        pref = self.config.get("global", {}).get("reference_antenna") or []
        if pref:
            return ",".join(pref)
        survey = self.snr_survey
        if survey is not None:
            # The survey's own reference antenna is the one every solution is tied to;
            # rank_antennas() cannot return it (its solutions are the masked sentinel).
            used = [name for name in survey.refant_names if name in self.antennas]
            if used:
                return used[0]
        if self._metadata and self._metadata.antennas:
            ranked = rank_reference_antennas(self._metadata)
            if ranked:
                return ranked[0]
        return "auto"

    @property
    def calibrator_field(self) -> str:
        """Comma-joined field for calibration: fringe finders, else phase cals, else targets."""
        for group in (self.sources.fringe_finders, self.sources.phase_calibrators, self.sources.targets):
            if group:
                return ",".join(s.name for s in group)
        return ",".join(self.sources.names)

    def _table(self, cal_type: str) -> Optional[CalTable]:
        """Return the most recent calibration table of a given type, if any."""
        for t in reversed(self.gaintables):
            if t.cal_type == cal_type:
                return t
        return None

    def prepare_run(self, *, scratch: bool = False, from_step: str = "") -> None:
        """Decide what this run redoes: everything, from a step, or only what is missing.

        Parameters
        ----------
        scratch : bool
            Forget all recorded progress so the whole pipeline runs again. The
            data itself is reset by the import step, which checks this flag.
        from_step : str
            Invalidate this step and everything after it, then resume there.
        """
        if scratch:
            logger.info("[{}] --scratch: discarding all recorded progress", self.project_code)
            self._state.reset()
            self.set_gaintables([])
            self._snr_surveys.clear()
            self._scratch = True
            return
        if from_step:
            if from_step not in STEP_ORDER:
                raise StepError(f"unknown step {from_step!r}; known steps: "
                                f"{', '.join(STEP_ORDER)}")
            logger.info("[{}] resuming from {}", self.project_code, from_step)
            self._state.invalidate_downstream(from_step, STEP_ORDER)
            # The tables those steps produced are about to be re-derived: keeping them
            # in the chain would apply the stale solution alongside its replacement.
            stale = self.drop_gaintables(STEP_ORDER[STEP_ORDER.index(from_step):])
            if stale:
                logger.info("[{}] dropped from the calibration chain: {}",
                            self.project_code, ", ".join(stale))

    @property
    def scratch(self) -> bool:
        """Whether this run was started with ``--scratch``."""
        return getattr(self, "_scratch", False)

    # -- actions --
    def listobs(self, listfile=None) -> dict:
        """Return a scan/field listing of the imported data (backend listobs).

        Parameters
        ----------
        listfile : str, optional
            Also write the listing to this file (backend-dependent default).
        """
        return self._backend.data.listobs(self.project_code, listfile=listfile)

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the observation."""
        lines = [f"Observation {self.project_code} ({self.observatory}, backend={self._backend.kind})"]
        if self._metadata:
            m = self._metadata
            duration_h = (m.time_range[1] - m.time_range[0]) / 3600.0
            if m.obs_date:
                lines.append(f"  observed: {m.obs_date.isoformat()} ({duration_h:.1f} h)")
            ant_marks = [a.name if a.observed else f"[{a.name}]" for a in m.antennas.values()]
            lines.append(f"  antennas ({m.n_antennas}, [] = no data): {', '.join(ant_marks)}")
            lines.append(f"  scans: {m.n_scans}; freq: {m.freq_setup.freq_ghz:.3f} GHz, "
                         f"BW {m.freq_setup.bandwidth_mhz:.0f} MHz "
                         f"({m.freq_setup.n_subbands} x {m.freq_setup.n_channels} ch)")
            if m.source_names:
                lines.append(f"  sources in data: {', '.join(m.source_names)}")
        else:
            lines.append("  (metadata not loaded; run import_data())")
        lines.append(f"  targets: {', '.join(s.name for s in self.sources.targets) or 'none'}")
        lines.append(f"  phase cals: {', '.join(s.name for s in self.sources.phase_calibrators) or 'none'}")
        lines.append(f"  fringe finders: {', '.join(s.name for s in self.sources.fringe_finders) or 'none'}")
        lines.append(f"  refant: {self.refant}")
        return "\n".join(lines)

    def report(self) -> dict:
        """Return a machine-readable report dict (no file is written in dummy mode)."""
        m = self._metadata
        rep = {
            "project": self.project_code,
            "observatory": self.observatory,
            "backend": self._backend.kind,
            "n_antennas": m.n_antennas if m else 0,
            "n_scans": m.n_scans if m else 0,
            "sources": {
                "targets": [s.name for s in self.sources.targets],
                "phase_calibrators": [s.name for s in self.sources.phase_calibrators],
                "fringe_finders": [s.name for s in self.sources.fringe_finders],
                "check_sources": [s.name for s in self.sources.check_sources],
            },
            "gaintables": [t.cal_type for t in self.gaintables],
            "flagging": self.flag_statistics,
            "steps": self._state.as_dict(),
            "warnings": warnings.summary(),
        }
        logger.info("report: {} (would write report.html + report.json)", self.project_code)
        return rep

    def reset(self) -> None:
        """Clear step state, calibration tables, and loaded metadata."""
        self._state.reset()
        self.set_gaintables([])
        self._metadata = None
        self.flag_statistics = {}
        logger.info("reset observation {}", self.project_code)

    def __repr__(self) -> str:
        return f"Observation({self.project_code!r}, {self.observatory}, backend={self._backend.kind})"
