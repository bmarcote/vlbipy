"""LBA observatory handler (stub-level behavior for the API-first milestone)."""
from __future__ import annotations

from .base import ObservatoryHandler


class LBAObservatory(ObservatoryHandler):
    """Australian Long Baseline Array. No automated downloads yet."""

    name = "LBA"
    auto_download = False
    needs_eop = True
    needs_accor = True
    tec_below_ghz = 7.0

    def manual_download_instructions(self) -> str:
        return ("LBA: retrieve the correlator output from the ATOA archive "
                "(https://atoa.atnf.csiro.au) together with the .antab (Tsys) and .uvflg "
                "files, and place them in the working directory.")
