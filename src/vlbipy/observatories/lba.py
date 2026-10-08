"""LBA observatory handler: local file discovery and FITS-IDI preparation.

LBA data are correlated with DiFX and retrieved by hand (no archive API), so
this handler only deals with what is on disk:

* locate the single FITS-IDI file of an experiment (``<CODE>.FITS``),
* append the Tsys and gain-curve tables of the ``.antab`` file to it — DiFX
  writes neither, only an empty ``GAIN_CURVE`` placeholder,
* convert the AIPS ``.uvflg`` a-priori flags into a CASA flag-command file.

The ``.antab`` and ``.uvflg`` of an LBA experiment are concatenations of
per-station files produced by different tools, so both are read tolerantly
(see :mod:`vlbipy.uvflg`): a station that reports Tsys for only part of the
band is given the level of the part it has, and flags for stations that are
not in the data are dropped.

A campaign of several epochs (``V589A``, ``V589B``, ...) usually keeps all the
raw files side by side while each epoch gets its own working directory, so the
parent of the working directory is searched as well.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import fitsidi
from ..logging_utils import get_logger, warnings
from ..tools import natsort_key
from .base import ObservatoryHandler

logger = get_logger()

#: File-name suffixes (lower case) a DiFX FITS-IDI file of an LBA experiment may carry.
_DATA_SUFFIXES = (".fits", ".fitsidi", ".idifits", ".idi")


class LBAObservatory(ObservatoryHandler):
    """Australian Long Baseline Array. No automated downloads; needs EOP corrections."""

    name = "LBA"
    auto_download = False
    needs_eop = True
    needs_accor = True
    tec_below_ghz = 7.0
    #: A-priori flag records longer than this many hours are not applied (0 = apply all). The
    #: station flag files mark slews; a record of hours is an interval the station log never
    #: closed (V589A: one "Mixed" record flags ATCA, the best antenna, for 12 of 13 hours).
    max_flag_hours = 2.0

    def _search_dirs(self, directory: str) -> list[Path]:
        """Directories searched for raw files: the working directory, its input_data/, and its parent."""
        root = Path(directory)
        return [d for d in (root, root / "input_data", root.resolve().parent) if d.is_dir()]

    def _find(self, project_code: str, directory: str, suffixes: tuple[str, ...]) -> list[str]:
        """Return the files named ``<code><anything><suffix>`` (case-insensitive), nearest directory first."""
        code = project_code.lower()
        for search_dir in self._search_dirs(directory):
            matches = [str(path) for path in search_dir.iterdir()
                       if path.is_file() and path.name.lower().startswith(code)
                       and path.name.lower().endswith(suffixes)
                       # V589A must not pick up V589AB.FITS: the code ends where the name does,
                       # or at a separator.
                       and not path.name[len(code):len(code) + 1].isalnum()]
            if matches:
                return sorted(matches, key=natsort_key)
        return []

    def find_data_files(self, project_code: str, directory: str) -> list[str]:
        """Return the FITS-IDI file(s) of the experiment (``<CODE>.FITS`` or EVN-style names)."""
        return self._find(project_code, directory, _DATA_SUFFIXES) or fitsidi.find_fitsidi_files(
            directory, project_code)

    def get_antab_file(self, project_code: str, directory: str) -> Optional[str]:
        """Return the ``.antab`` (Tsys/gain-curve) file of the experiment, if present."""
        matches = self._find(project_code, directory, (".antab",))
        return matches[0] if matches else None

    def get_flag_file(self, project_code: str, directory: str) -> Optional[str]:
        """Return the a-priori flag file: the CASA-style ``.flag`` if present, else the AIPS ``.uvflg``."""
        for suffix in (".flag", ".uvflg"):
            matches = self._find(project_code, directory, (suffix,))
            if matches:
                return matches[0]
        return None

    def prepare_for_import(self, files: list[str], directory: str, *, project_code: str = "",
                           replace_tsys: bool = False, **kwargs) -> list[str]:
        """Append Tsys/gain curves from the ``.antab`` and convert the ``.uvflg`` to a CASA ``.flag`` file.

        Both are skipped when already done (the FITS-IDI carries a Tsys table /
        the ``.flag`` file exists). The FITS-IDI file is modified in place.
        """
        if not files:
            return list(files)
        project_code = project_code or Path(files[0]).name.split(".")[0]
        self._append_antab(files, directory, project_code, replace_tsys)
        self._convert_flags(files, directory, project_code)
        return list(files)

    def _append_antab(self, files: list[str], directory: str, project_code: str, replace: bool) -> None:
        """Write the SYSTEM_TEMPERATURE and GAIN_CURVE tables of the ``.antab`` into the FITS-IDI file."""
        from ..casavlbitools import fitsidi as cvt

        antab = self.get_antab_file(project_code, directory)
        has_tsys, has_gc = fitsidi.has_tsys(files[0]), fitsidi.has_gain_curve(files[0])
        if not antab:
            if not has_tsys:
                warnings.anomaly(f"{project_code}: no .antab file found and the FITS-IDI file carries no "
                                 "Tsys table — amplitude calibration will fail")
            return
        if replace or not has_gc:
            # DiFX leaves an empty GAIN_CURVE placeholder that would block the append. It goes
            # first, while it is still the last table of the file: removing it is then a
            # truncation instead of a rewrite of the whole file.
            fitsidi.remove_empty_table(files[0], "GAIN_CURVE")
        if replace or not has_tsys:
            logger.info("appending Tsys from {} to {}", Path(antab).name, Path(files[0]).name)
            filled = cvt.append_tsys(str(antab), list(files), replace=replace and has_tsys, fill_bands=True)
            for station, per_pol in (filled or {}).items():
                bands = sorted({band for values in per_pol.values() for band in values})
                warnings.warn(f"{project_code}: the .antab gives {station} no Tsys for subband(s) "
                              f"{', '.join(map(str, bands))}; the level of its other subbands was used")
        if replace or not has_gc:
            logger.info("appending gain curves from {} to {}", Path(antab).name, Path(files[0]).name)
            cvt.append_gc(str(antab), files[0], replace=fitsidi.has_gain_curve(files[0]))
        self._report_antab_coverage(files[0], project_code)

    def _report_antab_coverage(self, idi_file: str, project_code: str) -> None:
        """Warn about antennas the appended tables say nothing about."""
        from astropy.io import fits
        import numpy as np

        with fits.open(idi_file) as hdulist:
            numbers = {int(n): str(name).strip() for name, n in zip(hdulist["ARRAY_GEOMETRY"].data["ANNAME"],
                                                                    hdulist["ARRAY_GEOMETRY"].data["NOSTA"])}
            with_tsys = set(np.unique(hdulist["SYSTEM_TEMPERATURE"].data["ANTENNA_NO"]).tolist()) \
                if "SYSTEM_TEMPERATURE" in hdulist else set()
            with_gc = set(np.unique(hdulist["GAIN_CURVE"].data["ANTENNA_NO"]).tolist()) \
                if "GAIN_CURVE" in hdulist else set()
        for label, present in (("Tsys", with_tsys), ("gain curve", with_gc)):
            missing = [name for number, name in sorted(numbers.items()) if number not in present]
            if missing:
                warnings.anomaly(f"{project_code}: the .antab has no {label} for {', '.join(missing)}; "
                                 "those antennas cannot be amplitude-calibrated (add nominal values "
                                 "to the .antab and re-import)")

    def _convert_flags(self, files: list[str], directory: str, project_code: str) -> None:
        """Convert ``<code>.uvflg`` to ``<work_dir>/<code>.flag`` (CASA flag commands)."""
        from ..uvflg import uvflg_to_casa

        flagfile = Path(directory) / f"{project_code.lower()}.flag"
        if flagfile.is_file():
            return
        matches = self._find(project_code, directory, (".uvflg",))
        if not matches:
            return
        meta = fitsidi.inspect_fitsidi(files[0], project_code)
        if meta.obs_date is None:
            warnings.warn(f"{project_code}: cannot date {Path(files[0]).name}; a-priori flags not converted")
            return
        result = uvflg_to_casa(matches[0], flagfile, meta.obs_date.year, antennas=list(meta.antennas),
                               max_seconds=3600.0 * self.max_flag_hours)
        for command in result["too_long"]:
            warnings.anomaly(f"{project_code}: a-priori flag longer than {self.max_flag_hours:g} h not applied "
                             f"(it would remove the antenna for most of the run; check the data): {command}")
        if not result["written"]:
            warnings.warn(f"{project_code}: {Path(matches[0]).name} holds no usable flag")

    def manual_download_instructions(self) -> str:
        return ("LBA: retrieve the correlator output (<CODE>.FITS) from the ATOA archive "
                "(https://atoa.atnf.csiro.au) together with the .antab (Tsys) and .uvflg "
                "files, and place them in the working directory or in its parent.")
