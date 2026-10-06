"""Skeleton for a new vlbipy backend — copy this file and fill it in.

How to add a backend
====================

1. Copy this file to a new module (inside vlbipy or in your own package).
2. Replace ``Template`` with your backend's name and set ``kind``.
3. Implement the operations you support and **delete the ones you do not** —
   deleting a method restores the base stub, which raises a clear
   ``NotImplementedError`` and is reported by :meth:`Backend.capabilities`.
   A half-written method that silently returns nothing is far worse than an
   absent one.
4. Register it so vlbipy can find it by name. Pick **one** (or more) of:

   a. **Entry point** (recommended for published packages)::

          # pyproject.toml of your package
          [project.entry-points."vlbipy.backends"]
          mybackend = "mypkg.backend:MyBackend"

      Install the package and ``vlbipy pipeline --backend mybackend`` works.

   b. **Programmatic registration** (e.g. in your package's ``__init__.py``)::

          from vlbipy.registry import register_backend
          from .backend import MyBackend
          register_backend("mybackend", MyBackend)

   c. **Config import bridge** — list your module in the pipeline config so
      it is imported (and its registrations run) before name resolution::

          [global]
          plugins = ["mypkg.vlbipy_plugin"]
          backend = "mybackend"

   d. **Direct reference** — use ``module:qualname`` as the backend value::

          [global]
          backend = "mypkg.backend:MyBackend"

5. Add tests (construction + whatever it implements).

If your package also provides an observatory handler, register both from the
same module — see :mod:`vlbipy.observatories.base` for the observatory
contract.

Each component is independent: you can implement ``DataOps`` first and get a
working import/inspection backend, then add calibration later.

Contracts every backend must honour
===================================

* **Return the product, not None.** Calibration returns ``CalTable``, imaging
  returns ``Image``, exports return the path. The API layer tracks products by
  what these return.
* **Raise, never swallow.** On failure raise
  :class:`~vlbipy.errors.BackendError` with the operation, the selection, and
  the underlying message. A step that fails silently corrupts every later step.
* **Take primitives.** Components receive resolved values (project code, source
  names, config values), never ``Observation``/``VLBIObs`` objects, so backends
  stay independent of the API layer.
* **Log at boundaries.** One ``logger.info`` per external task invocation with
  its resolved parameters; that log is the only record of what actually ran.
* **Import your library lazily.** Do it in ``__init__`` and raise
  ``BackendError`` if it is missing, so vlbipy still imports without it.

The six components
==================

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
"""
from __future__ import annotations

from typing import Optional

from ..errors import BackendError
from ..logging_utils import get_logger
from ..models import CalTable, ObsMetadata, ScanSNRSurvey
from ..results import Image, SelfcalResult
from .base import Backend, CalibrationOps, DataOps, ExportOps, FlagOps, ImagingOps, PlotOps

logger = get_logger()


class TemplateDataOps(DataOps):
    """Import and inspection — override to connect your data format."""

    def is_imported(self, project_code: str) -> bool:
        """Return True if this project's native data product already exists on disk."""
        raise NotImplementedError

    def import_data(self, project_code: str, source_names: list[str], *, scan_gap: int = 15,
                    files: Optional[list[str]] = None, delete: bool = False, **kwargs) -> None:
        """Convert the raw correlator files (FITS-IDI) into the backend's native format.

        The pipeline calls this once; the result lives in ``work_dir``.
        """
        raise NotImplementedError

    def get_metadata(self, project_code: str, source_names: list[str], observatory: str) -> ObsMetadata:
        """Read antennas, scans, frequency setup and sources into an ObsMetadata.

        Populate as much as is affordable here: everything downstream (scan
        selection, refant ranking, imaging parameters) reads this instead of
        re-querying the data. Leave the expensive per-visibility work to
        :meth:`get_subband_participation`.
        """
        raise NotImplementedError

    def get_subband_participation(self, project_code: str,
                                  antenna_names: list[str]) -> dict[str, tuple[int, ...]]:
        """Return ``{antenna: (subbands with unflagged data, ...)}``."""
        raise NotImplementedError

    def listobs(self, project_code: str, listfile: Optional[str] = None) -> dict:
        """Return a scan/field listing of the imported data.

        Optionally write the listing to *listfile* (backend-dependent format).
        """
        raise NotImplementedError


class TemplateCalibrationOps(CalibrationOps):
    """Calibration solves — each method returns the CalTable(s) it produced."""

    def a_priori(self, project_code: str, field: str, *, needs_eop: bool = False,
                 eop_file: Optional[str] = None, **kwargs) -> list[CalTable]:
        """Produce a-priori calibration tables (Tsys, gain curve, EOP, TEC, ...).

        Return the list of CalTables created, in application order.
        """
        raise NotImplementedError

    def initial_calibration(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Solve the single-band (instrumental) delay on the fringe finder.

        Return the CalTable describing the solution.
        """
        raise NotImplementedError

    def bandpass(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Solve the bandpass response and return the CalTable."""
        raise NotImplementedError

    def fringefit(self, project_code: str, field: str, refant: str, **kwargs) -> CalTable:
        """Global multi-band delay fringe fit.

        ``kwargs`` carries the whole ``[calibration.mbd]`` config section plus
        ``dispersive`` (solve the ionospheric delay too, decided from the
        observing frequency) and ``suffix`` (``"mbd"`` or ``"mbd2"``).

        Return the CalTable describing the solution.
        """
        raise NotImplementedError

    def scan_snr(self, project_code: str, field: str, *, refant: str = "",
                 channel_fraction: float = 0.8, scans: Optional[list] = None,
                 max_scans: int = 0, **kwargs) -> ScanSNRSurvey:
        """Diagnostic per-scan/antenna/polarization fringe SNR survey.

        This is a diagnostic solve only: the resulting table is *not* added to
        the gain chain. Return a :class:`~vlbipy.models.ScanSNRSurvey`.
        """
        raise NotImplementedError

    def apply(self, project_code: str, field: str, tables: list[CalTable], **kwargs) -> None:
        """Apply the accumulated calibration tables to a field.

        *tables* is the full gain chain in application order.
        """
        raise NotImplementedError


class TemplateFlagOps(FlagOps):
    """Flagging — implementing ``run`` alone gives you every named mode."""

    def run(self, project_code: str, kind: str, *, field: str = "", **kwargs) -> float:
        """Dispatch one flagging mode and return the flagged fraction (0.0-1.0).

        *kind* is one of ``autocorr``, ``edges``, ``quack``, ``tfcrop``,
        ``rflag``, ``aoflagger``, ``from_file``, ``manual``.
        """
        raise NotImplementedError


class TemplateImagingOps(ImagingOps):
    """Imaging and self-calibration."""

    def clean(self, project_code: str, source: str, *, robust: float = 0.0, imager: str = "wsclean",
              imsize: Optional[list[int]] = None, weighting: str = "briggs", niter: int = 0,
              **kwargs) -> Image:
        """Deconvolve a source and return an :class:`~vlbipy.results.Image`.

        Fill in :attr:`Image.stats` (:class:`~vlbipy.models.QualityMetrics`)
        with at least ``peak``, ``rms`` and ``dynamic_range``.
        """
        raise NotImplementedError

    def selfcal(self, project_code: str, source: str, *, image: Optional[Image] = None,
                phase_rounds: int = 4, ampphase_rounds: int = 5, threshold: float = 0.05,
                **kwargs) -> SelfcalResult:
        """Run the self-calibration loop (phase-only then amp+phase rounds).

        Discard rounds that do not improve dynamic range by at least
        *threshold* (fractional). Return a :class:`~vlbipy.results.SelfcalResult`.
        """
        raise NotImplementedError


class TemplatePlotOps(PlotOps):
    """Diagnostic plots — produced from the backend's own data products."""

    def diagnostic(self, project_code: str, kind: str, **kwargs) -> str:
        """Produce one diagnostic plot and return its file path.

        *kind* is one of ``tplot``, ``uv_coverage``, ``elevation``,
        ``amp_vs_time``, ``phase_vs_time``, ``autocorr``, ``crosscorr``.
        """
        raise NotImplementedError

    def caltable(self, project_code: str, caltable: str, cal_type: str = "") -> list[str]:
        """Plot one calibration table to PNG file(s) and return their paths."""
        raise NotImplementedError


class TemplateExportOps(ExportOps):
    """Split and export calibrated data."""

    def uvfits(self, project_code: str, source: str, **kwargs) -> str:
        """Export one source to UVFITS (Difmap-ready) and return the path."""
        raise NotImplementedError

    def ms(self, project_code: str, source: str, **kwargs) -> str:
        """Split one source into its own measurement set and return the path."""
        raise NotImplementedError


class TemplateBackend(Backend):
    """A new backend: set ``kind`` and wire up the components you implemented.

    Parameters
    ----------
    work_dir : str
        Directory holding this backend's data products.
    """

    kind = "template"
    requires_data_files = True

    data_ops = TemplateDataOps
    calibration_ops = TemplateCalibrationOps
    flag_ops = TemplateFlagOps
    imaging_ops = TemplateImagingOps
    plot_ops = TemplatePlotOps
    export_ops = TemplateExportOps

    def __init__(self, work_dir: str = ".") -> None:
        """Instantiate the backend, importing the underlying library lazily.

        Raise :class:`~vlbipy.errors.BackendError` if the library is missing,
        so vlbipy still imports without it installed.
        """
        # Import the underlying library here so vlbipy still works without it.
        try:
            import your_library  # noqa: F401
        except ImportError as exc:
            raise BackendError("the template backend requires your_library: "
                               "pip install your_library") from exc
        super().__init__(work_dir)
