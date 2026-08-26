"""vlbipy — backend-agnostic VLBI data reduction suite (API-first design).

Public entry point::

    from vlbipy import VLBIObs

    obs = VLBIObs(project="RSM07", network="EVN", target="3C286")
    obs.import_data()
    obs.calibrate()
    img = obs.clean(target="3C286", robust=2)
    img.export_fits("3C286.image.fits")

``VLBIObs`` represents one or more projects (a campaign) and exposes the full
reduction through callable namespaces. In this milestone all work runs on an
in-memory ``dummy`` backend (no CASA, no files). See ``docs/PRD.md``.
"""
from .config import load_config
from .errors import BackendError, ConfigError, SourceNotFoundError, StepError
from .fitsidi import find_fitsidi_files, inspect_fitsidi
from .logging_utils import configure_logging
from .models import (Antenna, BackendKind, CalTable, FreqSetup, Mode, ObsMetadata,
                     Observatory, QualityMetrics, Scan, SourceType, Stokes)
from .observation import Observation
from .plotting import CalTablePlotter
from .results import Image, ImageSet, SelfcalResult
from .sources import Source, SourceSet
from .vlbiobs import VLBIObs

# Explicitly import backends so they're available
from .backends import get_backend  # noqa: F401

__version__ = "0.2.0"

__all__ = [
    "VLBIObs", "Observation", "load_config", "configure_logging",
    "inspect_fitsidi", "find_fitsidi_files", "get_backend", "CalTablePlotter",
    "Source", "SourceSet", "Image", "ImageSet", "SelfcalResult",
    "Antenna", "FreqSetup", "Scan", "ObsMetadata", "CalTable", "QualityMetrics",
    "Stokes", "Observatory", "BackendKind", "Mode", "SourceType",
    "ConfigError", "SourceNotFoundError", "BackendError", "StepError",
    "__version__",
]
