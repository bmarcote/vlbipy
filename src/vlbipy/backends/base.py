"""The backend interface for vlbipy.

A backend hides all package-specific task invocation (CASA, dask-ms, AIPS,
WSClean, ...) behind one cohesive interface. The API layer (``VLBIObs`` /
``Observation`` and the operation namespaces) calls only these methods and never
imports a backend's underlying library.

The interface is split into **components**, one per operation group, mirroring
the user-facing namespaces so the two read the same way::

    obs.calibrate.bandpass(...)          # user-facing namespace
    backend.calibrate.bandpass(...)      # backend component

A :class:`Backend` composes six components:

===============  ==========================  ==================================
Attribute        Component                   Owns
===============  ==========================  ==================================
``.data``        :class:`DataOps`            import, metadata, inspection
``.calibrate``   :class:`CalibrationOps`     a-priori, SBD, bandpass, fringe fit
``.flag``        :class:`FlagOps`            every flagging mode
``.image``       :class:`ImagingOps`         deconvolution, self-cal, uv models
``.plot``        :class:`PlotOps`            diagnostic plots
``.export``      :class:`ExportOps`          split / export / concatenate
===============  ==========================  ==================================

Every operation has a default that raises ``NotImplementedError``; a concrete
backend overrides only what it supports, and :meth:`Backend.capabilities`
reports what that is. To write a new backend, copy ``backends/template.py`` —
it is a complete, commented skeleton of all six components.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..models import CalTable, ObsMetadata, ScanSNRSurvey
from ..results import Image, SelfcalResult

#: Attributes on a component that are not operations (excluded from introspection).
_NON_OPERATIONS = frozenset({"group", "kind", "work_dir", "backend", "operations",
                             "implemented", "supports"})


class BackendComponent:
    """Base class for one operation group of a backend.

    A component is bound to its owning :class:`Backend` at construction and
    reaches shared state (``work_dir``, the backend's library handles) through
    it, so components stay thin and independently regenerable.

    Parameters
    ----------
    backend : Backend
        The owning backend instance.
    """

    #: Short name of the operation group (used in error messages and capabilities).
    group: str = "component"

    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    @property
    def backend(self) -> "Backend":
        """The owning backend."""
        return self._backend

    @property
    def kind(self) -> str:
        """The owning backend's short identifier (e.g. ``"casa"``)."""
        return self._backend.kind

    @property
    def work_dir(self) -> Path:
        """The owning backend's working directory."""
        return self._backend.work_dir

    def _unsupported(self, operation: str, hint: str = "") -> NotImplementedError:
        """Return the error raised by an operation this backend does not implement."""
        message = f"backend {self.kind!r} does not implement {self.group}.{operation}()"
        return NotImplementedError(f"{message}: {hint}" if hint else message)

    # -- introspection --
    def operations(self) -> list[str]:
        """Return the names of every operation this component declares."""
        return sorted(name for name in dir(type(self))
                      if not name.startswith("_") and name not in _NON_OPERATIONS
                      and callable(getattr(type(self), name, None)))

    def _declaring_abc(self) -> type:
        """Return the component ABC this class derives from (e.g. :class:`DataOps`)."""
        for cls in reversed(type(self).__mro__):
            if issubclass(cls, BackendComponent) and cls is not BackendComponent:
                return cls
        return type(self)

    def supports(self, operation: str) -> bool:
        """Return True if this backend actually implements ``operation``.

        An operation counts as implemented when the concrete component overrides
        the stub declared on its ABC.
        """
        abc = self._declaring_abc()
        return getattr(type(self), operation, None) is not getattr(abc, operation, None)

    def implemented(self) -> list[str]:
        """Return the operations this backend implements (a subset of :meth:`operations`)."""
        return [name for name in self.operations() if self.supports(name)]

    def __repr__(self) -> str:
        done = self.implemented()
        return (f"<{type(self).__name__} ({self.kind}.{self.group}) "
                f"{len(done)}/{len(self.operations())} implemented: {', '.join(done) or 'none'}>")


class DataOps(BackendComponent):
    """Ingest raw data and read everything the rest of the pipeline needs to know.

    This component is the only one allowed to create the backend's native data
    product, and the single source of observation metadata.
    """

    group = "data"

    def is_imported(self, project_code: str) -> bool:
        """Return True if the project's data is already in the backend's native format."""
        return False

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False, **kwargs) -> None:
        """Import raw correlator output (FITS-IDI) into the backend's native format.

        Parameters
        ----------
        project_code : str
            Project code; names the output product.
        source_names : list of str
            Sources declared in the configuration.
        scan_gap : int
            Time gap in seconds that starts a new scan.
        files : list of str, optional
            The raw files, in order.
        delete : bool
            Overwrite an existing product instead of skipping.
        """
        raise self._unsupported("import_data")

    def import_uvfits(self, project_code: str, uvfits: str, *, delete: bool = False) -> None:
        """Import a UVFITS file into the backend's native format."""
        raise self._unsupported("import_uvfits")

    def adopt_ms(self, project_code: str, ms: str) -> None:
        """Register an already-existing measurement set as this project's data."""
        raise self._unsupported("adopt_ms")

    def reset_calibration(self, project_code: str, *, unflag: bool = True,
                          backup_flags: bool = True) -> dict:
        """Return an already-imported dataset to its as-correlated state.

        Re-running the pipeline over a dataset a previous attempt already touched
        otherwise inherits that attempt's flags and corrected data: flags only
        ever accumulate, so each run starts from a smaller array than the last
        and the result silently depends on run history.

        Parameters
        ----------
        project_code : str
            Project code.
        unflag : bool
            Clear every flag (the pipeline re-applies the a-priori flags itself).
        backup_flags : bool
            Save the current flags first so hand-made flags can be recovered.

        Returns
        -------
        dict
            ``flagged_before`` / ``flagged_after`` fractions and the backup name.
        """
        raise self._unsupported("reset_calibration")

    def get_metadata(self, project_code: str, source_names: list[str], observatory: str) -> ObsMetadata:
        """Return full observation metadata: antennas, scans, frequency setup, sources.

        Backends should populate as much of :class:`~vlbipy.models.ObsMetadata`
        as they can afford at inspection time — downstream steps (scan selection,
        reference-antenna ranking, imaging parameters) read it instead of
        re-querying the data.
        """
        raise self._unsupported("get_metadata")

    def get_subband_participation(self, project_code: str, antenna_names: list[str]) -> dict[str, tuple[int, ...]]:
        """Return ``{antenna: (subband indices with unflagged data, ...)}``.

        Heterogeneous VLBI arrays record different subband subsets per antenna;
        this is separate from :meth:`get_metadata` because it requires reading
        the flags and is therefore the expensive part of inspection.
        """
        raise self._unsupported("get_subband_participation")

    def read_spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                      refant: str = "", column: str = "corrected", all_pols: bool = False,
                      **kwargs) -> dict:
        """Return time-averaged spectra on baselines to the reference antenna."""
        raise self._unsupported("read_spectrum")

    def read_uvdistance(self, project_code: str, *, field: str = "", column: str = "corrected",
                        time_bin: float = 10.0, **kwargs) -> dict:
        """Return visibilities against uv distance, averaged per subband and in time."""
        raise self._unsupported("read_uvdistance")

    def read_uv_coverage(self, project_code: str, *, field: str = "", **kwargs) -> dict:
        """Return the sampled (u, v) points per source, in wavelengths.

        Only unflagged cross-correlation rows are included; the conjugate points
        ``(-u, -v)`` are *not* duplicated (the plot mirrors them). Fields with
        more than ~200k points are subsampled deterministically (every k-th row).

        Returns
        -------
        dict
            ``fields`` (``{source_name: {"u": list[float], "v": list[float]}}``),
            ``unit`` (``"Mlambda"``), ``freq_ghz`` (the reference frequency used).
        """
        raise self._unsupported("read_uv_coverage")

    def read_timeseries(self, project_code: str, *, field: str = "", refant: str = "",
                        column: str = "corrected", max_time_bins: int = 300, **kwargs) -> dict:
        """Return amplitude/phase vs time per baseline, averaged over frequency."""
        raise self._unsupported("read_timeseries")

    def read_dynamic_spectra(self, project_code: str, *, field: str = "",
                             column: str = "corrected", max_time_bins: int = 200,
                             stokes_i: bool = True, **kwargs) -> dict:
        """Return a binned time x frequency array per antenna pair."""
        raise self._unsupported("read_dynamic_spectra")

    def listobs(self, project_code: str, listfile: Optional[str] = None) -> dict:
        """Return a scan/field listing of the imported data (optionally written to a file)."""
        raise self._unsupported("listobs")

    def datasets(self, project_code: str) -> list:
        """Return the visibilities as lazy datasets (dask-ms style backends only)."""
        raise self._unsupported("datasets", "only dataset-oriented backends expose visibilities")


class CalibrationOps(BackendComponent):
    """Solve for and apply calibration.

    Every method returns the :class:`~vlbipy.models.CalTable` it produced (or a
    list of them) so the caller can accumulate the gain-table chain without
    knowing anything about the backend's table format.
    """

    group = "calibrate"

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, **kwargs) -> list[CalTable]:
        """Produce a-priori calibration tables (Tsys, gain curve, EOP, TEC, ...)."""
        raise self._unsupported("a_priori")

    def initial_calibration(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Single-band delay (instrumental) calibration on the fringe finder."""
        raise self._unsupported("initial_calibration")

    def bandpass(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Bandpass calibration."""
        raise self._unsupported("bandpass")

    def fringefit(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Global (multi-band delay) fringe fit.

        Recognised keyword arguments include ``dispersive``: when true, also solve
        the dispersive (ionospheric) delay. The pipeline sets it from the observing
        frequency (see ``[calibration].ionos``), so a backend that cannot fit a
        dispersive term should say so rather than silently ignoring it.
        """
        raise self._unsupported("fringefit")

    def scalar_bandpass(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Solve one amplitude gain per antenna and subband, removing inter-subband steps."""
        raise self._unsupported("scalar_bandpass")

    def scan_snr(self, project_code: str, field: str, *, refant: str = "",
                 channel_fraction: float = 0.8, scans: Optional[list] = None,
                 max_scans: int = 0, **kwargs) -> ScanSNRSurvey:
        """Measure fringe SNR per scan, antenna and polarization on the calibrators.

        A short fringe fit over the central ``channel_fraction`` of each subband,
        solved per scan, whose SNRs are read back into a
        :class:`~vlbipy.models.ScanSNRSurvey`. This is a *diagnostic* solve: its
        table is not added to the gain-table chain. Downstream it drives scan
        selection, reference-antenna ranking and the SNR matrix plot.

        Parameters
        ----------
        project_code : str
            Project code.
        field : str
            Calibrator field selection (comma-separated names).
        refant : str
            Reference antenna; its own (sentinel) solutions are masked out.
        channel_fraction : float
            Fraction of central channels per subband to use (default 0.8).
        scans : list, optional
            Restrict the survey to these scans (default: all scans on ``field``).
        """
        raise self._unsupported("scan_snr")

    def reweight(self, project_code: str, *, column: str = "corrected", **kwargs) -> dict:
        """Recompute the visibility weights from the scatter of the calibrated data.

        Returns a dict describing the resulting weights (backend-dependent). The
        pipeline flags anew and re-solves the whole chain after this, because
        the recomputed weights expose data that looked fine until now.
        """
        raise self._unsupported("reweight")

    def solution_coverage(self, project_code: str, table: CalTable, metadata=None) -> dict:
        """Return ``{antenna: {subbands with an unflagged solution}}`` for a table."""
        raise self._unsupported("solution_coverage")

    def smooth(self, project_code: str, table: CalTable, **kwargs) -> CalTable:
        """Smooth an existing calibration table (delay/rate noise reduction)."""
        raise self._unsupported("smooth")

    def apply(self, project_code: str, field: str, tables: list[CalTable], **kwargs) -> None:
        """Apply the accumulated calibration tables to a field."""
        raise self._unsupported("apply")

    def write_callib(self, project_code: str, tables: list[CalTable], **kwargs) -> "Path":
        """Write the list of tables to apply, in order, and return the file's path.

        Every point where calibration is applied — the apply step and the
        on-the-fly priors of each solve — should go through this rather than
        assembling parallel table/interpolation/mapping lists, so what was
        applied stays a readable artefact next to the data. Backends whose
        toolkit has no equivalent may leave this unimplemented; the pipeline
        only requires it for provenance, not for correctness.
        """
        raise self._unsupported("write_callib")


class FlagOps(BackendComponent):
    """Flagging.

    Backends only need to implement :meth:`run`; the named modes below delegate
    to it, so a backend gets the whole flagging surface from one method and can
    still override any individual mode that needs bespoke handling.
    """

    group = "flag"

    def run(self, project_code: str, kind: str, *, field: str = "", **kwargs) -> float:
        """Run one flagging mode and return the fraction of data it flagged.

        Parameters
        ----------
        project_code : str
            Project code.
        kind : str
            One of ``autocorr``, ``edges``, ``quack``, ``tfcrop``, ``rflag``,
            ``aoflagger``, ``from_file``, ``manual``.
        field : str
            Field selection (empty = all fields).
        """
        raise self._unsupported(f"run[{kind}]")

    def measure_edge_channels(self, project_code: str, table: CalTable, *,
                              threshold: float = 6.0, max_edge_fraction: float = 0.25,
                              **kwargs) -> dict:
        """Measure how many channels roll off at each subband edge, from a bandpass table.

        Returns a dict with at least ``n_edge`` (channels to flag at each edge)
        and ``n_channels``; the result is what :meth:`edges` flags.
        """
        raise self._unsupported("measure_edge_channels")

    def flagged_fraction(self, project_code: str, **kwargs) -> float:
        """Return the flagged fraction of the observable data (0-1)."""
        raise self._unsupported("flagged_fraction")

    def summary(self, project_code: str, **kwargs) -> dict:
        """Return flagging statistics: ``{"flagged", "observable", "fraction", "antenna": {...}, "spw": {...}}``.

        Counts must cover *observable* data only — no autocorrelations and no
        visibilities that were never recorded (antenna absent from a scan or a
        subband it did not observe) — so a per-antenna fraction describes data
        quality rather than the schedule.
        """
        raise self._unsupported("summary")

    def autocorr(self, project_code: str, **kwargs) -> float:
        """Flag autocorrelations (never used in VLBI imaging)."""
        return self.run(project_code, "autocorr", **kwargs)

    def edges(self, project_code: str, *, edge_fraction: float = 0.1, **kwargs) -> float:
        """Flag the outer channels of every subband."""
        return self.run(project_code, "edges", edge_fraction=edge_fraction, **kwargs)

    def quack(self, project_code: str, *, per_antenna: Optional[dict] = None,
              interval: float = 0.0, **kwargs) -> float:
        """Flag the first seconds of every scan.

        ``per_antenna`` (``{antenna: seconds}``) wins over the array-wide
        ``interval``. Backends that can *measure* the settling ramp override this
        and fall back to the measurement when neither is given; the default
        implementation can only apply what it is told and flags nothing otherwise.
        """
        per_antenna = dict(per_antenna or {})
        if not per_antenna and interval and float(interval) > 0:
            per_antenna = {"": float(interval)}
        return sum(self.run(project_code, "quack", interval=float(seconds), antenna=antenna)
                   for antenna, seconds in per_antenna.items())

    def tfcrop(self, project_code: str, *, field: str = "", **kwargs) -> float:
        """Time-frequency outlier flagging (calibrators only)."""
        return self.run(project_code, "tfcrop", field=field, **kwargs)

    def aoflagger(self, project_code: str, *, field: str = "", strategy: str = "default",
                  **kwargs) -> float:
        """Run AOFlagger (calibrators only; targets are protected)."""
        return self.run(project_code, "aoflagger", field=field, strategy=strategy, **kwargs)

    def measure_quack(self, project_code: str, *, field: str = "", threshold: float = 0.9,
                      **kwargs) -> dict:
        """Measure per antenna how long each scan takes to reach full amplitude."""
        raise self._unsupported("measure_quack")

    def outliers(self, project_code: str, *, field: str = "", threshold: float = 5.0,
                 dry_run: bool = False, **kwargs) -> dict:
        """Flag visibilities that break the smoothness of their own baseline."""
        raise self._unsupported("outliers")

    def from_file(self, project_code: str, flagfile: str, **kwargs) -> float:
        """Apply flags listed in an external flag-command file."""
        return self.run(project_code, "from_file", flagfile=flagfile, **kwargs)

    def manual(self, project_code: str, **selection) -> float:
        """Apply a manual flag selection (antenna / spw / timerange / ...)."""
        return self.run(project_code, "manual", **selection)


class ImagingOps(BackendComponent):
    """Deconvolution, uv-plane model fitting and self-calibration."""

    group = "image"

    def clean(self, project_code: str, source: str, *, robust: float = 0.0, imager: str = "wsclean",
              imsize: Optional[list[int]] = None, weighting: str = "briggs", niter: int = 0,
              **kwargs) -> Image:
        """Deconvolve a source and return the resulting :class:`~vlbipy.results.Image`."""
        raise self._unsupported("clean")

    def uvmodel(self, project_code: str, source: str, *, components: int = 1, **kwargs) -> Image:
        """Fit a source model directly in the uv plane (e.g. circular Gaussians).

        Preferred over CLEAN when building a compact calibrator model for
        self-calibration, where robustness matters more than image fidelity.
        """
        raise self._unsupported("uvmodel")

    def selfcal(self, project_code: str, source: str, *, image: Optional[Image] = None,
                phase_rounds: int = 4, ampphase_rounds: int = 5, threshold: float = 0.05,
                **kwargs) -> SelfcalResult:
        """Run the self-calibration loop for a source, discarding rounds that do not improve."""
        raise self._unsupported("selfcal")

    def statistics(self, image_path: str) -> dict:
        """Return image statistics (peak, rms, dynamic range, beam) for an existing image."""
        raise self._unsupported("statistics")


class PlotOps(BackendComponent):
    """Diagnostic plots produced from the backend's own data products.

    Plots are filed by *what they were made from*, under ``<work_dir>/plots``:

    ``raw/``         data as correlated, before any calibration is applied
    ``caltables/``   the calibration solutions themselves
    ``calibrated/``  data with the calibration applied

    Keeping the same plot of the same source in different folders before and
    after calibration is what makes the pair directly comparable — the whole
    point of a before/after diagnostic.
    """

    group = "plot"

    #: Sub-directories of ``<work_dir>/plots``, by what the plot was made from.
    CATEGORIES = ("raw", "caltables", "calibrated")

    def plot_dir(self, category: str = "") -> Path:
        """Return (and create) the plot directory for a category.

        Parameters
        ----------
        category : str
            One of :attr:`CATEGORIES`; empty puts the plot directly in ``plots/``.
        """
        if category and category not in self.CATEGORIES:
            raise ValueError(f"unknown plot category {category!r} "
                             f"(expected one of {', '.join(self.CATEGORIES)})")
        path = self.work_dir / "plots" / category if category else self.work_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def category_for_column(column: str) -> str:
        """Return the plot category matching a data column (``corrected`` -> calibrated)."""
        return "calibrated" if str(column).lower() in ("corrected", "corrected_data") else "raw"

    def diagnostic(self, project_code: str, kind: str, **kwargs) -> str:
        """Produce one diagnostic plot and return its path.

        Parameters
        ----------
        kind : str
            One of ``tplot``, ``uv_coverage``, ``elevation``, ``amp_vs_time``,
            ``phase_vs_time``, ``autocorr``, ``crosscorr``.
        """
        raise self._unsupported(f"diagnostic[{kind}]")

    def caltable(self, project_code: str, caltable: str, cal_type: str = "") -> list[str]:
        """Plot one calibration table to PNG file(s); return their paths."""
        raise self._unsupported("caltable")

    def scan_snr(self, project_code: str, survey: Optional[ScanSNRSurvey] = None,
                 **kwargs) -> list[str]:
        """Plot the scan/antenna fringe-SNR matrix, one PNG per polarization."""
        raise self._unsupported("scan_snr")

    def spectrum(self, project_code: str, *, field: str = "", scans: Optional[list] = None,
                 refant: str = "", column: str = "corrected", label: str = "", **kwargs) -> list[str]:
        """Plot amplitude/phase vs channel of the (calibrated) data, one panel per baseline."""
        raise self._unsupported("spectrum")

    def radplot(self, project_code: str, *, field: str = "", column: str = "corrected",
                time_bin: float = 10.0, label: str = "", **kwargs) -> str:
        """Plot amplitude and phase vs uv distance for one source."""
        raise self._unsupported("radplot")

    def timeseries(self, project_code: str, *, field: str = "", refant: str = "",
                   column: str = "corrected", label: str = "", **kwargs) -> list[str]:
        """Plot amplitude/phase vs time per baseline, in the stacked two-panel layout."""
        raise self._unsupported("timeseries")

    def baseline_corner(self, project_code: str, *, field: str = "", column: str = "corrected",
                        quantity: str = "phase", **kwargs) -> str:
        """Plot a time x frequency panel per antenna pair, coloured by phase or amplitude."""
        raise self._unsupported("baseline_corner")

    def bandpass_profile(self, project_code: str, measurement: dict, **kwargs) -> str:
        """Plot the per-channel amplitude / phase-scatter / flagged profile of the band."""
        raise self._unsupported("bandpass_profile")


class ExportOps(BackendComponent):
    """Split, export and concatenate calibrated data."""

    group = "export"

    def uvfits(self, project_code: str, source: str, **kwargs) -> str:
        """Export a source to UVFITS (Difmap-ready); return its path."""
        raise self._unsupported("uvfits")

    def ms(self, project_code: str, source: str, **kwargs) -> str:
        """Split a source into its own measurement set; return its path."""
        raise self._unsupported("ms")

    def merge(self, project_codes: list[str], source_names: list[str], **kwargs) -> str:
        """Concatenate calibrated data across projects/epochs; return a handle."""
        raise self._unsupported("merge")


class Backend:
    """Package-agnostic backend: a bundle of six operation components.

    Subclasses set :attr:`kind` and point the ``*_ops`` class attributes at their
    own component subclasses. Components are instantiated in ``__init__``, so a
    subclass that needs to open a library handle first should do that *before*
    calling ``super().__init__()``.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Directory holding this backend's data products.
    """

    #: Short identifier of the backend (overridden by subclasses).
    kind: str = "base"
    #: Whether import needs raw data files on disk (False only for the dummy backend).
    requires_data_files: bool = True

    #: Component classes; subclasses override with their own implementations.
    data_ops: type[DataOps] = DataOps
    calibration_ops: type[CalibrationOps] = CalibrationOps
    flag_ops: type[FlagOps] = FlagOps
    imaging_ops: type[ImagingOps] = ImagingOps
    plot_ops: type[PlotOps] = PlotOps
    export_ops: type[ExportOps] = ExportOps

    def __init__(self, work_dir: Union[str, Path] = ".") -> None:
        self.work_dir = Path(work_dir)
        self.data = self.data_ops(self)
        self.calibrate = self.calibration_ops(self)
        self.flag = self.flag_ops(self)
        self.image = self.imaging_ops(self)
        self.plot = self.plot_ops(self)
        self.export = self.export_ops(self)

    def components(self) -> dict[str, BackendComponent]:
        """Return the components keyed by their attribute name."""
        return {"data": self.data, "calibrate": self.calibrate, "flag": self.flag,
                "image": self.image, "plot": self.plot, "export": self.export}

    def capabilities(self) -> dict[str, list[str]]:
        """Return ``{component: [implemented operations]}`` for this backend.

        Lets callers (and users, interactively) see exactly what a backend
        supports without triggering ``NotImplementedError``.
        """
        return {name: component.implemented() for name, component in self.components().items()}

    def supports(self, component: str, operation: str) -> bool:
        """Return True if ``component.operation`` is implemented by this backend."""
        target = self.components().get(component)
        return bool(target and target.supports(operation))

    def __repr__(self) -> str:
        counts = ", ".join(f"{name}={len(ops)}" for name, ops in self.capabilities().items())
        return f"{type(self).__name__}(kind={self.kind!r}, implemented: {counts})"


__all__ = ["Backend", "BackendComponent", "DataOps", "CalibrationOps", "FlagOps",
           "ImagingOps", "PlotOps", "ExportOps"]
