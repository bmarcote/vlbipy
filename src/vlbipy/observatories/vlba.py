"""VLBA observatory handler (stub-level behavior for the API-first milestone)."""
from __future__ import annotations

from .base import ObservatoryHandler


class VLBAObservatory(ObservatoryHandler):
    """Very Long Baseline Array. No automated downloads; needs EOP/ACCOR."""

    name = "VLBA"
    auto_download = False
    needs_eop = True
    needs_accor = True
    tec_below_ghz = 7.0

    def manual_download_instructions(self) -> str:
        return ("VLBA: NRAO does not support automated archive downloads. Retrieve the "
                "correlator output (FITS-IDI/UVFITS) and the .antab/.uvflg files manually "
                "from https://data.nrao.edu and place them in the working directory.")
