"""EVN observatory handler: archive downloads and FITS-IDI preparation.

Implements the real EVN data-retrieval flow, recycled from
``casa_pipeline.download.evn_data`` / ``obsdata.Importing.evn_download`` but
using urllib + hashlib (via :mod:`vlbipy.tools`) instead of wget/md5sum:

* locate FITS-IDI files already on disk,
* download FITS-IDI + ``.antab`` + ``.uvflg`` from https://archive.jive.eu,
  verifying the archive MD5 checksum file and retrying failed files once,
* pre-import preparation: append Tsys/gain-curve tables from the ``.antab``
  file and convert AIPS ``.uvflg`` flags to a CASA-style ``.flag`` file
  (both via the external ``casavlbitools`` package, when installed).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .. import fitsidi, tools
from ..logging_utils import get_logger, warnings
from .base import ObservatoryHandler

logger = get_logger()

#: Base URL of the EVN/JIVE archive experiment tree.
EVN_ARCHIVE_URL = "https://archive.jive.eu/exp"
#: Archive catalog page listing every experiment as ``arch.php?exp=<CODE>_<YYMMDD>`` links.
EVN_ARCHIVE_INDEX_URL = "https://archive.jive.eu/scripts/listarch.php"


def _import_casavlbitools():
    """Return the casavlbitools.fitsidi module (vendored copy, or an external install)."""
    try:
        from ..casavlbitools import fitsidi as cvt_fitsidi
        return cvt_fitsidi
    except ImportError:
        pass
    try:
        from casavlbitools import fitsidi as cvt_fitsidi
        return cvt_fitsidi
    except ImportError:
        warnings.warn("casavlbitools could not be imported: cannot append Tsys/GC to FITS-IDI "
                      "or convert .uvflg flags.")
        return None


class EVNObservatory(ObservatoryHandler):
    """European VLBI Network. Supports automated archive downloads."""
    name = "EVN"
    auto_download = True
    needs_eop = False
    needs_accor = False
    tec_below_ghz = 5.0

    def find_data_files(self, project_code: str, directory: str) -> list[str]:
        """Return the FITS-IDI files for the project already present in the directory."""
        return fitsidi.find_fitsidi_files(directory, project_code)

    def resolve_obsdate(self, project_code: str, directory: str) -> str:
        """Return the observing date in YYMMDD format.

        Resolution order: RDATE of a FITS-IDI file already on disk, then the
        archive index at https://archive.jive.eu/exp/ (latest matching entry).

        Raises
        ------
        Exception
            If the date cannot be determined (set ``[import].obsdate``).
        """
        files = self.find_data_files(project_code, directory)
        if files:
            return fitsidi.get_obs_date(files[0]).strftime("%y%m%d")
        logger.info("resolving obsdate of {} from the EVN archive catalog...", project_code)
        index = tools.fetch_url_text(EVN_ARCHIVE_INDEX_URL)
        dates = re.findall(rf"exp={re.escape(project_code.upper())}_(\d{{6}})", index)
        if dates:
            if len(set(dates)) > 1:
                logger.warning("multiple archive entries for {}: {}; using the latest",
                               project_code, ", ".join(sorted(set(dates))))
            return sorted(dates)[-1]
        raise Exception(f"cannot determine the observing date of {project_code} from the EVN "
                          "archive; set it explicitly via the [import].obsdate config key (YYMMDD)")

    def _verify_checksums(self, checksum_file: Path, directory: Path, base_url: str,
                          username: Optional[str], password: Optional[str]) -> None:
        """Verify downloaded files against the archive checksum file, re-downloading failures once."""
        expected: dict[str, str] = {}
        for line in checksum_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                expected[Path(parts[1].lstrip("*")).name] = parts[0]
        failed = [name for name, md5 in expected.items()
                  if (directory / name).is_file() and tools.md5sum(directory / name) != md5]
        for filename in failed:
            logger.warning("checksum mismatch for {}; re-downloading", filename)
            tools.download_file(f"{base_url}/fits/{filename}", directory / filename, username, password)
            if tools.md5sum(directory / filename) != expected[filename]:
                raise Exception(f"checksum verification failed twice for {filename}")
        if failed:
            logger.info("re-downloaded {} corrupted file(s) successfully", len(failed))

    def download_data(self, project_code: str, directory: str, *, obsdate: Optional[str] = None,
                      username: Optional[str] = None, password: Optional[str] = None,
                      **kwargs) -> list[str]:
        """Download FITS-IDI + .antab + .uvflg files for an EVN project from the archive.

        Parameters
        ----------
        project_code : str
            EVN project code (e.g. ``"EB032"``).
        directory : str
            Destination directory.
        obsdate : str, optional
            Observing date in YYMMDD format; auto-resolved from the archive index if omitted.
        username, password : str, optional
            Archive credentials for proprietary data; omit for public data.

        Returns
        -------
        list of str
            The naturally-sorted FITS-IDI files now present in the directory.
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        obsdate = obsdate or self.resolve_obsdate(project_code, directory)
        base_url = f"{EVN_ARCHIVE_URL}/{project_code.upper()}_{obsdate}"
        code = project_code.lower()

        logger.info("downloading EVN data for {} from {}/fits/", project_code, base_url)
        listing = tools.fetch_url_text(f"{base_url}/fits/", username, password)
        entries = re.findall(r'href="([^"?/]+)"', listing)
        wanted = sorted({e for e in entries if e.lower().startswith(code)}, key=tools.natsort_key)
        if not wanted:
            raise Exception(f"no files for {project_code} found at {base_url}/fits/ "
                              "(wrong project code or observing date?)")

        # Separate files already on disk from those that need downloading
        to_download = []
        for filename in wanted:
            dest = directory_path / filename
            if dest.is_file():
                logger.info("  {} already present; skipping", filename)
            else:
                to_download.append((filename, dest))

        # Download missing files in parallel (up to 4 concurrent)
        if to_download:
            logger.info("Fetching {} file(s) in parallel...", len(to_download))
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(tools.download_file, f"{base_url}/fits/{filename}",
                                    dest, username, password): filename
                    for filename, dest in to_download
                }
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        future.result()
                        logger.info("  {} downloaded", filename)
                    except Exception as exc:
                        raise Exception(f"failed to download {filename}: {exc}") from exc

        checksum_file = directory_path / f"{code}.checksum"
        if checksum_file.is_file():
            logger.info("verifying MD5 checksums...")
            self._verify_checksums(checksum_file, directory_path, base_url, username, password)
        else:
            warnings.warn(f"{project_code}: no checksum file in the archive; skipping verification")

        self.fetch_apriori_files(project_code, directory, obsdate=obsdate,
                                 username=username, password=password)

        files = self.find_data_files(project_code, directory)
        if not files:
            raise Exception(f"download finished but no FITS-IDI files found in {directory}")
        logger.info("EVN download complete: {} FITS-IDI file(s)", len(files))
        return files

    def fetch_apriori_files(self, project_code: str, directory: str, *,
                            obsdate: Optional[str] = None, username: Optional[str] = None,
                            password: Optional[str] = None) -> dict:
        """Fetch the ``.antab`` and ``.uvflg`` from the archive's pipeline area.

        These live under ``pipe/`` rather than with the FITS-IDI files, so a
        dataset that was downloaded (or copied) by hand usually arrives without
        them — and then the a-priori flags are silently never applied, leaving
        slewing data in the calibrated product. Kept separate from
        :meth:`download_data` so it can be run against an already-imported
        project.

        Missing files are reported, not fatal: proprietary or older experiments
        do not always publish both.

        Returns
        -------
        dict
            ``antab`` and ``uvflg`` paths (``None`` when unavailable).
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        code = project_code.lower()
        obsdate = obsdate or self.resolve_obsdate(project_code, directory)
        base_url = f"{EVN_ARCHIVE_URL}/{project_code.upper()}_{obsdate}"
        for filename in (f"{code}.antab.gz", f"{code}.uvflg"):
            plain = directory_path / filename.removesuffix(".gz")
            if plain.is_file():
                logger.info("{} already present", plain.name)
                continue
            dest = directory_path / filename
            try:
                logger.info("fetching {}/pipe/{}", base_url, filename)
                tools.download_file(f"{base_url}/pipe/{filename}", dest, username, password)
                if filename.endswith(".gz"):
                    tools.gunzip(dest)
            except Exception as exc:  # noqa: BLE001 - not every experiment publishes both
                warnings.warn(f"{project_code}: could not fetch {filename} from the archive ({exc})")
        return {"antab": self.get_antab_file(project_code, directory),
                "uvflg": self.get_flag_file(project_code, directory)}

    def get_antab_file(self, project_code: str, directory: str) -> Optional[str]:
        """Return the .antab (Tsys/gain-curve) file for the project, if present."""
        for search_dir in (Path(directory), Path(directory) / "input_data"):
            matches = sorted(search_dir.glob(f"{project_code.lower()}*.antab"))
            if matches:
                return str(matches[0])
        return None

    def get_flag_file(self, project_code: str, directory: str) -> Optional[str]:
        """Return the a-priori flag file: the CASA-style .flag if present, else the AIPS .uvflg."""
        for suffix in (".flag", ".uvflg"):
            for search_dir in (Path(directory), Path(directory) / "input_data"):
                matches = sorted(search_dir.glob(f"{project_code.lower()}*{suffix}"))
                if matches:
                    return str(matches[0])
        return None

    def prepare_for_import(self, files: list[str], directory: str, *, project_code: str = "",
                           replace_tsys: bool = False, **kwargs) -> list[str]:
        """Append Tsys/GC tables from .antab and convert .uvflg to a CASA .flag file.

        Both steps are no-ops when the FITS-IDI files already carry the tables /
        the .flag file already exists. Requires ``casavlbitools``; without it the
        step is skipped with a warning unless the FITS-IDI files already contain
        the Tsys tables (then nothing is needed).
        """
        if not files:
            return list(files)
        project_code = project_code or Path(files[0]).name.split("_")[0].split(".")[0]
        antab = self.get_antab_file(project_code, directory)
        already_has_tsys = fitsidi.has_tsys(files[0])

        if antab and (replace_tsys or not already_has_tsys):
            cvt = _import_casavlbitools()
            if cvt is not None:
                logger.info("appending Tsys from {} to the FITS-IDI files", antab)
                cvt.append_tsys(str(antab), files, replace=replace_tsys)
                logger.info("appending gain-curve information from {}", antab)
                cvt.append_gc(str(antab), files[0], replace=replace_tsys)
            elif not already_has_tsys:
                warnings.anomaly(f"{project_code}: FITS-IDI files lack Tsys and casavlbitools is "
                                 "unavailable — amplitude calibration will be wrong")
        elif not antab and not already_has_tsys:
            warnings.anomaly(f"{project_code}: no .antab file found and the FITS-IDI files carry "
                             "no Tsys tables — amplitude calibration will be wrong")

        uvflg = Path(directory) / f"{project_code.lower()}.uvflg"
        flagfile = Path(directory) / f"{project_code.lower()}.flag"
        if uvflg.is_file() and not flagfile.is_file():
            cvt = _import_casavlbitools()
            if cvt is not None:
                logger.info("converting AIPS flags {} -> {}", uvflg.name, flagfile.name)
                cvt.convert_flags(infile=str(uvflg), idifiles=files, outfile=str(flagfile))
        return list(files)

    def manual_download_instructions(self) -> str:
        return ("EVN data are retrieved from the JIVE archive at https://archive.jive.eu/ "
                "(FITS-IDI files under <PROJECT>_<YYMMDD>/fits/, .antab/.uvflg under pipe/).")
