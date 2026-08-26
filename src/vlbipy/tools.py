"""Small shared utilities for vlbipy (time conversion, downloads, file helpers).

Recycled and modernized from the earlier ``casa_pipeline`` project: the wget/md5sum
subprocess calls are replaced with urllib/hashlib so vlbipy has no shell-tool
dependencies. All network access in vlbipy goes through :func:`fetch_url_text`
and :func:`download_file` so tests can monkeypatch them.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union
from .logging_utils import get_logger

logger = get_logger()

#: Origin of the Modified Julian Date scale.
_MJD_ORIGIN = dt.datetime(1858, 11, 17, tzinfo=None)


def mjd2datetime(mjd: float) -> dt.datetime:
    """Convert a Modified Julian Date (in days) to a naive UTC datetime."""
    return _MJD_ORIGIN + dt.timedelta(days=float(mjd))


def mjdsec2datetime(mjd_seconds: float) -> dt.datetime:
    """Convert an MJD given in seconds (CASA convention) to a naive UTC datetime."""
    return _MJD_ORIGIN + dt.timedelta(seconds=float(mjd_seconds))


def datetime2mjd(date: dt.datetime) -> float:
    """Convert a naive UTC datetime to a Modified Julian Date (in days)."""
    delta = date - _MJD_ORIGIN
    return delta.days + delta.seconds / 86400.0


def natsort_key(text: str):
    """Return a natural-sort key so that 'IDI2' sorts before 'IDI10'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def chunkert(counter: int, max_length: int, increment: int):
    """Yield ``(start, n)`` chunks covering ``[counter, max_length)`` in steps of ``increment``."""
    while counter < max_length:
        this_increment = min(increment, max_length - counter)
        yield (counter, this_increment)
        counter += this_increment


def space_available_gb(path: Union[str, Path]) -> float:
    """Return the available disk space (in GB) on the filesystem containing ``path``."""
    results = os.statvfs(str(path))
    return results.f_frsize * results.f_bavail / 1e9


def collect_casa_logs(log_dir: Union[str, Path], search_dirs=None) -> list[Path]:
    """Move stray ``casa*.log`` files into ``log_dir``; return what was moved.

    CASA writes a log named for the moment it starts into whatever directory the
    process happens to be in, so a working directory accumulates one file per
    invocation. Everything a run produces belongs under the project directory,
    so these are swept up rather than left scattered around the shell's cwd.

    Existing names are never overwritten: a colliding file gets a numeric
    suffix, because two CASA sessions started in the same second would otherwise
    lose one of the two logs.

    Parameters
    ----------
    log_dir : str or pathlib.Path
        Destination directory (created if needed).
    search_dirs : iterable, optional
        Directories to sweep; defaults to the current working directory.

    Returns
    -------
    list of pathlib.Path
        The new locations of the moved files.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for directory in [Path(d) for d in (search_dirs or [Path.cwd()])]:
        if not directory.is_dir() or directory.resolve() == log_dir.resolve():
            continue
        for source in sorted(directory.glob("casa*.log")):
            target = log_dir / source.name
            counter = 1
            while target.exists():
                target = log_dir / f"{source.stem}.{counter}{source.suffix}"
                counter += 1
            try:
                shutil.move(str(source), str(target))
                moved.append(target)
            except OSError as exc:  # in use, or not ours to move: leave it alone
                logger.debug("could not move {}: {}", source, exc)
    if moved:
        logger.info("collected {} CASA log file(s) into {}", len(moved), log_dir)
    return moved


def md5sum(path: Union[str, Path]) -> str:
    """Return the hex MD5 checksum of a file (streamed, so large files are fine)."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_opener(url: str, username: Optional[str], password: Optional[str]) -> urllib.request.OpenerDirector:
    """Build a urllib opener, with HTTP basic auth when credentials are given."""
    if username and password:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, username, password)
        return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr))
    return urllib.request.build_opener()


def fetch_url_text(url: str, username: Optional[str] = None, password: Optional[str] = None,
                   timeout: float = 120.0) -> str:
    """Fetch a URL and return its body decoded as text.

    Raises
    ------
    ConnectionError
        On HTTP/network failure, with a hint about credentials on 401/403.
    """
    opener = _build_opener(url, username, password)
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        hint = " (are archive credentials required/correct?)" if exc.code in (401, 403) else ""
        raise ConnectionError(f"HTTP {exc.code} fetching {url}{hint}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"network error fetching {url}: {exc.reason}") from exc


def download_file(url: str, dest: Union[str, Path], username: Optional[str] = None,
                  password: Optional[str] = None, timeout: float = 600.0) -> Path:
    """Download a URL to ``dest`` (a file path), streaming to disk; return the path.

    Raises
    ------
    ConnectionError
        On HTTP/network failure, with a hint about credentials on 401/403.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = _build_opener(url, username, password)
    try:
        with opener.open(url, timeout=timeout) as response, open(dest, "wb") as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)
    except urllib.error.HTTPError as exc:
        dest.unlink(missing_ok=True)
        hint = " (are archive credentials required/correct?)" if exc.code in (401, 403) else ""
        raise ConnectionError(f"HTTP {exc.code} downloading {url}{hint}") from exc
    except urllib.error.URLError as exc:
        dest.unlink(missing_ok=True)
        raise ConnectionError(f"network error downloading {url}: {exc.reason}") from exc
    logger.debug("downloaded {} -> {} ({} bytes)", url, dest, dest.stat().st_size)
    return dest


#: HTTPS source(s) tried first for the USNO Earth-orientation-parameter file.
EOP_URLS = ("https://gemini.gsfc.nasa.gov/500/oper/solve_apriori_files/usno_finals.erp",)
#: CDDIS FTPS fallback (anonymous), fetched via curl exactly like the NRAO VLBI pipeline.
EOP_CDDIS_URL = "ftp://gdc.cddis.eosdis.nasa.gov/vlbi/gsfc/ancillary/solve_apriori/usno_finals.erp"


def fetch_eop_file(directory: Union[str, Path]) -> Path:
    """Fetch the USNO EOP file (``usno_finals.erp``) needed by gencal caltype='eop'.

    An already-present file in ``directory`` is reused. Otherwise the HTTPS
    mirrors are tried first, then the CDDIS FTPS archive via ``curl --ftp-ssl``.

    Raises
    ------
    ConnectionError
        If no source could provide the file (set ``[calibration].eop_file``).
    """
    dest = Path(directory) / "usno_finals.erp"
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("using existing EOP file {}", dest)
        return dest
    for url in EOP_URLS:
        try:
            logger.info("downloading EOP file from {}", url)
            return download_file(url, dest)
        except ConnectionError as exc:
            logger.warning("EOP download failed from {}: {}", url, exc)
    logger.info("falling back to CDDIS (curl --ftp-ssl) for the EOP file")
    result = subprocess.run(["curl", "-sS", "-u", "anonymous:daip@nrao.edu", "--ftp-ssl",
                             EOP_CDDIS_URL, "-o", str(dest)], capture_output=True, text=True, timeout=300)
    if result.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.unlink(missing_ok=True)
    raise ConnectionError("could not download usno_finals.erp from any source; download it manually "
                          "and point [calibration].eop_file at it")


def gunzip(path: Union[str, Path], remove_original: bool = True) -> Path:
    """Decompress a ``.gz`` file next to itself; return the decompressed path."""
    path = Path(path)
    if path.suffix != ".gz":
        return path
    out_path = path.with_suffix("")
    with gzip.open(path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    if remove_original:
        path.unlink()
    return out_path
