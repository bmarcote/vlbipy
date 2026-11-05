"""VLBA observatory handler: local FITS-IDI discovery and checks of what the file must already carry.

VLBA data come from the NRAO archive (no automated download) as one FITS-IDI file per experiment, expected
here as ``<CODE>.idifits`` (rename or symlink the archive's ``VLBA_<CODE>_..._<date>.idifits``) in the working
directory, its ``input_data/`` or its parent (:func:`vlbipy.fitsidi.search_dirs`).

Unlike the EVN and the LBA there are no ``.antab`` / ``.uvflg`` files: the FITS-IDI file holds the Tsys
(``SYSTEM_TEMPERATURE``), the gain curves (``GAIN_CURVE``) and the a-priori flags (``FLAG``). The import copies
the first two into the SYSCAL / GAIN_CURVE subtables that ``gencal`` reads, and the flags into ``FLAG_CMD``,
from where the a-priori flag step applies them (``flags_in_fitsidi``, see
:meth:`vlbipy.namespaces.FlagNamespace.apriori`). :meth:`VLBAObservatory.prepare_for_import` only checks that
the file is complete and carries those tables; it never modifies it.
"""
from __future__ import annotations

from pathlib import Path

from .. import fitsidi
from ..logging_utils import get_logger, warnings
from .base import ObservatoryHandler

logger = get_logger()

#: File-name suffix of the VLBA FITS-IDI file (matched case-insensitively): ``<CODE>.idifits``.
DATA_SUFFIX = ".idifits"


class VLBAObservatory(ObservatoryHandler):
    """Very Long Baseline Array. No automated downloads; needs EOP/ACCOR; Tsys, gain curves and flags come
    inside the FITS-IDI file."""

    name = "VLBA"
    auto_download = False
    needs_eop = True
    needs_accor = True
    flags_in_fitsidi = True

    def find_data_files(self, project_code: str, directory: str) -> list[str]:
        """Return ``[<CODE>.idifits]`` from the nearest of :func:`vlbipy.fitsidi.search_dirs` holding it
        (name compared case-insensitively; a symlink counts), or ``[]``."""
        wanted = f"{project_code}{DATA_SUFFIX}".lower()
        for search_dir in fitsidi.search_dirs(directory):
            matches = sorted(str(path) for path in search_dir.iterdir() if path.name.lower() == wanted
                             and path.is_file())
            if matches:
                return matches[:1]
        return []

    def prepare_for_import(self, files: list[str], directory: str, *, project_code: str = "", **kwargs) -> list[str]:
        """Check the FITS-IDI file before it is imported; returns ``files`` unchanged.

        Raises ``ValueError`` when the file is shorter than its headers say (a download in progress): its
        Tsys, gain-curve and flag tables sit after the visibilities and would be silently missing. Warns
        (anomaly) when the SYSTEM_TEMPERATURE, GAIN_CURVE or FLAG table is absent or empty, and names the
        antennas the Tsys and gain-curve tables do not cover.
        """
        for path in files:
            fitsidi.check_complete(path)
        code = project_code or Path(files[0]).name.split(".")[0]
        rows = fitsidi.table_rows(files[0])
        logger.info("VLBA {}: SYSTEM_TEMPERATURE {} row(s), GAIN_CURVE {} row(s), FLAG {} row(s)",
                    Path(files[0]).name, rows.get("SYSTEM_TEMPERATURE", 0), rows.get("GAIN_CURVE", 0),
                    rows.get("FLAG", 0))
        for extname, consequence in (("SYSTEM_TEMPERATURE", "amplitude calibration will fail"),
                                     ("GAIN_CURVE", "amplitude calibration will fail"),
                                     ("FLAG", "no a-priori flags (off-source, slewing) will be applied")):
            if not rows.get(extname, 0):
                warnings.anomaly(f"{code}: {Path(files[0]).name} has no {extname} table rows; {consequence}")
        if rows.get("SYSTEM_TEMPERATURE", 0) and rows.get("GAIN_CURVE", 0):
            for label, missing in fitsidi.apriori_coverage(files[0]).items():
                if missing:
                    warnings.anomaly(f"{code}: the FITS-IDI file has no {label} for {', '.join(missing)}; "
                                     "those antennas cannot be amplitude-calibrated")
        return list(files)

    def manual_download_instructions(self) -> str:
        return ("VLBA: NRAO does not support automated archive downloads. Retrieve the FITS-IDI file "
                "(VLBA_<CODE>_..._<date>.idifits) from https://data.nrao.edu and place it, renamed or symlinked "
                "as <CODE>.idifits, in the working directory or in its parent. It already carries the Tsys, "
                "gain curves and a-priori flags: no .antab or .uvflg file is needed.")
