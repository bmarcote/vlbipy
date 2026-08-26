"""Observatory-handler interface for vlbipy.

An :class:`ObservatoryHandler` owns everything that differs between VLBI networks:
auto-download capability, a-priori file formats (``.antab``/``.uvflg``), and which
correlator-specific corrections (EOP, ACCOR, ionospheric TEC) apply. The core API
selects a handler from the ``network`` argument and never special-cases arrays.
"""
from __future__ import annotations

from typing import Optional

from ..logging_utils import get_logger

logger = get_logger()


class ObservatoryHandler:
    """Base class describing per-network behavior.

    Subclasses set the class attributes and override the methods they support.
    """

    #: Network name.
    name: str = "base"
    #: Whether the archive supports automated downloads.
    auto_download: bool = False
    #: Whether Earth-orientation-parameter corrections apply.
    needs_eop: bool = False
    #: Whether ACCOR (correlator amplitude) corrections apply.
    needs_accor: bool = False
    #: Apply ionospheric TEC correction when the lowest frequency is below this (GHz); 0 disables.
    tec_below_ghz: float = 0.0

    def find_data_files(self, project_code: str, directory: str) -> list[str]:
        """Return already-present raw data files for a project (empty if none)."""
        return []

    def download_data(self, project_code: str, directory: str, **kwargs) -> list[str]:
        """Download raw data for a project; return the file paths.

        Raises
        ------
        NotImplementedError
            For networks without automated downloads (see
            :meth:`manual_download_instructions`).
        """
        raise NotImplementedError(self.manual_download_instructions())

    def prepare_for_import(self, files: list[str], directory: str, **kwargs) -> list[str]:
        """Perform any pre-import preparation (e.g. append Tsys/GC); return files."""
        return list(files)

    def fetch_apriori_files(self, project_code: str, directory: str, **kwargs) -> dict:
        """Fetch the a-priori calibration/flag files (``.antab`` / ``.uvflg``).

        Returns ``{"antab": path_or_None, "uvflg": path_or_None}``; the default
        just reports what is already on disk, for networks with no archive API.
        """
        return {"antab": self.get_antab_file(project_code, directory),
                "uvflg": self.get_flag_file(project_code, directory)}

    def get_antab_file(self, project_code: str, directory: str) -> Optional[str]:
        """Return the path to the ANTAB (Tsys/gain-curve) file, if any."""
        return None

    def get_flag_file(self, project_code: str, directory: str) -> Optional[str]:
        """Return the path to the a-priori flag file, if any."""
        return None

    def manual_download_instructions(self) -> str:
        """Return human-readable instructions for retrieving data manually."""
        return f"No download instructions available for {self.name}."

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, auto_download={self.auto_download})"
