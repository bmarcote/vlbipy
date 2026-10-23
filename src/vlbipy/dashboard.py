"""Static HTML dashboard of a vlbipy reduction.

``build_dashboard(obs)`` writes ``<work_dir>/html/`` with one page per tab
(``index.html`` = overview, ``diagnostics.html``, ``calibration.html``,
``final_data.html``), a shared ``style.css``/``dashboard.js`` and a
``dashboard.json`` snapshot of everything the pages were rendered from. Plots are
referenced relatively (``../plots/...``, ``../images/...``), so the ``html/``
folder must travel together with the rest of the project directory.

The context also carries the general comments of the reduction
(:func:`vlbipy.comments.build_comments`): the Overview page shows them and
``build_dashboard`` rewrites ``<work_dir>/summary.md`` with the same list on top.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from typing import Optional

from astropy.time import Time
from loguru import logger

# The thresholds shared with the general comments are defined once, in vlbipy.comments.
from .comments import (FRINGE_DETECTION_SNR, GAIN_BAD_DEVIATION, GAIN_DECIMALS, GAIN_WARN_DEVIATION,
                       NO_DATA_FLAGGED_FRACTION, build_comments, group_comments, load_run_warnings)

#: (file stem, tab title) in display order. ``index`` is the overview.
PAGES: list[tuple[str, str]] = [("index", "Overview"), ("diagnostics", "Diagnostics"),
                                ("calibration", "Calibration"), ("final_data", "Final results")]

#: Fringe-SNR thresholds for the per-antenna chips in the scan table: (upper bound, css class).
SNR_CHIP_CLASSES: list[tuple[float, str]] = [(FRINGE_DETECTION_SNR, "snr-bad"), (6.0, "snr-low"), (10.0, "snr-mid"),
                                             (math.inf, "snr-good")]

#: Why a (scan, antenna) chip carries no SNR: state -> (css class, tooltip text). ``snr`` is
#: coloured by :func:`snr_class` instead and is deliberately absent from this map.
SCAN_CHIP_STATES: dict[str, tuple[str, str]] = {
    "absent": ("snr-absent", "not in this scan"),
    "refant": ("snr-ref", "reference antenna (no SNR of its own)"),
    "nodata": ("snr-nodata", "no data (fully flagged)"),
    "nofringe": ("snr-nofringe", "no fringe solution"),
    "unsurveyed": ("snr-none", "scan not surveyed"),
}

#: Amplitude-correction cells: (largest ``|gain - 1|`` of the class, css class). ``gain-none`` (no
#: solution) is assigned by :func:`gain_class` and is deliberately absent from this list. The bounds
#: (and ``GAIN_DECIMALS``, the decimals a correction is shown and judged with) come from
#: :mod:`vlbipy.comments`, so the table and the general comments call the same corrections large.
GAIN_CELL_CLASSES: list[tuple[float, str]] = [(GAIN_WARN_DEVIATION, "gain-good"), (GAIN_BAD_DEVIATION, "gain-warn"),
                                              (math.inf, "gain-bad")]

#: Chip class of a general comment per severity (see :mod:`vlbipy.comments`).
COMMENT_CHIP_CLASSES: dict[str, str] = {"problem": "crit", "warning": "warn", "note": "mut"}

#: Diagnostics sub-tabs: glob patterns relative to <work_dir>/plots.
AUTOCORR_PATTERNS: list[str] = ["raw/*_scan*.autocorr.png"]
CROSSCORR_PATTERNS: list[str] = ["raw/*_scan*.spectrum.png"]
#: ``<code>.raw.time_evolution.ref<ANT>.spw<N>.png``: one figure per reference antenna and subband.
TIME_EVOLUTION_PATTERNS: list[str] = ["raw/*.time_evolution.*.png"]

#: Calibration-table plot groups: (title, blurb, glob patterns relative to <work_dir>/plots).
CALIBRATION_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Gain curves", "Gain curve per antenna divided by its DPFU (or by its peak when the DPFU is not on record), "
     "against elevation; the shaded band marks the elevations observed.", ["caltables/*gcal*.png"]),
    ("System temperature", "Tsys vs time per antenna (one colour per polarization).", ["caltables/*tsys*.png"]),
    ("Bandpass", "Amplitude and phase vs frequency per antenna.", ["caltables/*.bpass*.png"]),
    ("SBD selected-scan data", "Amplitude and phase versus time for each selected SBD stage, one baseline to that stage's reference antenna per panel.",
     ["calibrated/*.sbd*_scan*.timeseries.png", "raw/*.sbd*_scan*.timeseries.png"]),
    ("Single-band delay (SBD)", "Instrumental phase, delay and rate solutions versus frequency/subband.", ["caltables/*.sbd.*.png", "caltables/*.sbd2.*.png"]),
    ("Multi-band delay (MBD)",
     "Global fringe-fit delay/phase/rate solutions, as solved and after the time smoothing that is "
     "what actually gets applied (<code>_smooth</code>).",
     ["caltables/*.mbd.*.png", "caltables/*.mbd2.*.png",
      "caltables/*.mbd_smooth.*.png", "caltables/*.mbd2_smooth.*.png"]),
    ("Scalar bandpass", "One amplitude level per antenna and subband.", ["caltables/*scalar_bp*amp*.png"]),
    ("Fringe-SNR survey", "Scan x antenna fringe SNR on the calibrators.", ["raw/*snr_matrix*.png"]),
    ("Calibrator amplitude and phase", "Calibrated amplitude and phase versus frequency and time for every fringe finder, phase calibrator and check source.",
     ["calibrated/*.calibrated_*.spectrum.png", "calibrated/*.calibrated_*.timeseries.png"]),
    ("Subband phase jumps", "Phase of every subband relative to the first, per calibrator scan and baseline to the reference antenna.",
     ["calibrated/*subband_phases*.png"]),
]

#: Final-data plot groups: (title, blurb, glob patterns, fallback patterns).
FINAL_GROUPS: list[tuple[str, str, list[str], list[str]]] = [
    ("UV coverage", "uv-plane sampling per source.", ["raw/*uv_coverage*.png"], []),
    ("Amplitude/phase vs uv distance", "Calibrated visibilities vs baseline length; the model is drawn as a line.",
     ["calibrated/*.final.radplot.*.png"], ["calibrated/*.calibrated.radplot.*.png"]),
    ("Calibrated spectrum", "Amplitude and phase vs frequency on baselines to the reference antenna.",
     ["calibrated/*final_*.spectrum.png"], ["calibrated/*.calibrated.spectrum.png"]),
    ("Light curve per baseline", "Amplitude and phase vs time per baseline.",
     ["calibrated/*final_*.timeseries.png"], ["calibrated/*calibrated_*.timeseries.png"]),
    ("Total-amplitude light curve", "Coherent sum of all baselines at several time averagings.",
     ["calibrated/*lightcurve*.png"], []),
]


# -- helpers -----------------------------------------------------------------------------------

def mjd_seconds_to_utc(seconds: float) -> str:
    """Format an MJD expressed in seconds as ``YYYY-MM-DD HH:MM:SS`` UTC.

    Returns an empty string for missing/implausible values (0, None, or anything
    before 1970 — a sentinel zero otherwise formats as 1858).
    """
    if not seconds or seconds / 86400.0 < 40587:
        return ""
    return Time(seconds / 86400.0, format="mjd").strftime("%Y-%m-%d %H:%M:%S")


def snr_class(snr: Optional[float]) -> str:
    """CSS class of an antenna chip for a fringe SNR (``snr-none`` when unknown)."""
    if snr is None or (isinstance(snr, float) and math.isnan(snr)):
        return "snr-none"
    return next(cls for bound, cls in SNR_CHIP_CLASSES if snr < bound)


def gain_class(gain: Optional[float]) -> str:
    """CSS class of an amplitude-correction cell.

    The class follows the value as it is displayed (rounded to :data:`GAIN_DECIMALS`), so a cell that
    reads ``1.10`` is never coloured as "more than 10% off".

    Parameters
    ----------
    gain : float or None
        Amplitude factor applied to the visibilities (1.0 = unchanged); None/NaN = no solution.

    Returns
    -------
    str
        ``gain-none`` without a solution, else the first class of :data:`GAIN_CELL_CLASSES` whose bound
        ``|gain - 1|`` does not exceed.
    """
    value = _finite_or_none(gain)
    if value is None:
        return "gain-none"
    deviation = abs(round(value, GAIN_DECIMALS) - 1.0)
    # The tolerance absorbs the binary representation of the bounds (abs(1.1 - 1.0) > 0.1 in floats).
    return next(cls for bound, cls in GAIN_CELL_CLASSES if deviation <= bound + 1e-9)


def human_size(n_bytes: Optional[int]) -> str:
    """Format a byte count as ``KiB``/``MiB``/``GiB`` (empty string when unknown).

    Parameters
    ----------
    n_bytes : int or None
        Size in bytes; None/negative (a directory, or a missing file) gives "".

    Returns
    -------
    str
        e.g. ``"43.3 MiB"``, ``"1.21 GiB"``, or ``""``.
    """
    if n_bytes is None or n_bytes < 0:
        return ""
    for unit, scale in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.2f} {unit}" if n_bytes / scale < 10 else f"{n_bytes / scale:.1f} {unit}"
    return f"{n_bytes} B"


_MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def format_obs_date(value) -> str:
    """Format an observing date as ``DD Mmm YYYY`` from a date, datetime or string.

    Accepts ``datetime.date``/``.datetime`` objects, ISO strings and other
    formats that :class:`~astropy.time.Time` can parse. Returns an empty string
    for missing/empty input and falls back to the original string when parsing
    fails.
    """
    if not value:
        return ""
    d: Optional[date] = None
    if isinstance(value, date):
        d = value.date() if isinstance(value, datetime) else value
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            d = date.fromisoformat(value)
        else:
            try:
                parsed = Time(value).to_datetime()
                d = parsed.date()
            except Exception:
                return value
    if d is None:
        try:
            d = date.fromisoformat(str(value))
        except Exception:
            return str(value)
    return f"{d.day:02d} {_MONTH_NAMES[d.month]} {d.year}"


def relative_href(target: Path, html_dir: Path) -> str:
    """Relative URL from ``html_dir`` to ``target`` with forward slashes."""
    import os
    return os.path.relpath(target, html_dir).replace("\\", "/")


def find_plots(plot_root: Path, patterns: list[str]) -> list[Path]:
    """Sorted unique files under ``plot_root`` matching any of the glob ``patterns``."""
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in plot_root.glob(pattern):
            found.setdefault(str(path), path)
    def plot_order(value: str) -> tuple:
        """Natural report order, with fringe parameters phase, delay, rate."""
        name = Path(value).name.lower()
        parameter = next((index for index, tag in enumerate((".phase.", ".delay.", ".rate."))
                          if tag in name), 99)
        return (re.sub(r"\.(phase|delay|rate)\.", ".parameter.", value), parameter, value)

    return [found[k] for k in sorted(found, key=plot_order)]


def human_title(path: Path, project_code: str) -> str:
    """Readable figure title from a plot file name."""
    stem = re.sub(r"\.png$", "", path.name)
    stem = re.sub(rf"^{re.escape(project_code)}[._]?", "", stem) if project_code else stem
    return stem.replace("_", " ").replace(".", " / ").strip() or path.name


def scan_number_from_name(path: Path) -> Optional[int]:
    """Scan number encoded as ``_scan<N>`` in a plot file name (None when absent)."""
    match = re.search(r"_scan(\d+)\.", path.name)
    return int(match.group(1)) if match else None


def time_evolution_key(path: Path) -> Optional[tuple[str, int]]:
    """Reference antenna and subband encoded as ``.time_evolution.ref<ANT>.spw<N>.png`` (None when absent)."""
    match = re.search(r"\.time_evolution\.ref([^.]+)\.spw(\d+)\.png$", path.name)
    return (match.group(1), int(match.group(2))) if match else None


# -- data collection ---------------------------------------------------------------------------

def _primary_observation(obs):
    """Return the single :class:`Observation` behind ``obs`` (a ``VLBIObs`` or an ``Observation``)."""
    if not hasattr(obs, "observations"):
        return obs
    primary = obs._primary
    return primary() if callable(primary) else primary


def _source_rows(observation, metadata) -> list[dict]:
    """Source table rows: type, name, field id, coordinates (sexagesimal and degrees) and scan count."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    rows = []
    names = list(metadata.source_names) if metadata else []
    known = {s.name: s for s in observation.sources}
    for name in names + [n for n in known if n not in names]:
        source = known.get(name)
        coords = source.coordinates if source is not None and source.coordinates is not None else None
        if coords is None and metadata and name in metadata.source_coords:
            ra, dec = metadata.source_coords[name]
            coords = SkyCoord(ra * u.deg, dec * u.deg)
        rows.append({"name": name, "type": source.source_type.value if source else "other",
                     "id": metadata.source_ids.get(name, "") if metadata else "",
                     "ra": coords.ra.to_string(unit=u.hourangle, sep=":", precision=4, pad=True) if coords else "",
                     "dec": coords.dec.to_string(unit=u.deg, sep=":", precision=3, alwayssign=True, pad=True) if coords else "",
                     "ra_deg": float(coords.ra.deg) if coords else None, "dec_deg": float(coords.dec.deg) if coords else None,
                     "n_scans": sum(1 for s in metadata.scans if s.source == name) if metadata else 0})
    return rows


def antenna_scan_state(name: str, in_scan: bool, snr: Optional[float], scan_surveyed: bool,
                       refant_names, no_data_antennas) -> str:
    """Classify one (scan, antenna) cell of the scan table.

    Parameters
    ----------
    name : str
        Antenna code.
    in_scan : bool
        Whether the antenna appears in the scan's antenna list.
    snr : float or None
        Finite fringe SNR of this antenna in this scan, None when there is none.
    scan_surveyed : bool
        Whether the fringe-SNR survey covered this scan at all (at least one
        antenna of the scan came back with a finite SNR).
    refant_names : iterable of str
        Reference antenna(s) of the survey; their own solutions are sentinels,
        so they have no SNR of their own.
    no_data_antennas : iterable of str
        Antennas that recorded no usable data at all over the whole observation.

    Returns
    -------
    str
        One of ``"absent"``, ``"snr"``, ``"refant"``, ``"nodata"``,
        ``"nofringe"``, ``"unsurveyed"``.
    """
    if not in_scan:
        return "absent"
    # A finite value outranks "refant": with a comma-separated fallback reference
    # antenna, a secondary refant does have real SNR in the scans it did not reference.
    if snr is not None:
        return "snr"
    if name in refant_names:
        return "refant"
    if name in no_data_antennas:
        return "nodata"
    return "nofringe" if scan_surveyed else "unsurveyed"


def _scan_rows(metadata, source_roles: Optional[dict[str, str]] = None,
               refant_names: Optional[list[str]] = None,
               no_data_antennas: Optional[set[str]] = None) -> list[dict]:
    """Scan table rows with per-antenna fringe SNR and an explicit state per antenna.

    Each row carries a slot for every antenna present in the metadata so the
    rendered scan chips line up vertically even when an antenna is absent from a
    particular scan. ``antenna_snrs`` holds the finite SNR (None when there is
    none) and ``antenna_states`` says *why* there is none - see
    :func:`antenna_scan_state`.

    Parameters
    ----------
    metadata : ObsMetadata or None
        Source of antennas, scans and the fringe-SNR survey.
    source_roles : dict, optional
        ``{source: role}`` used to tag each scan row.
    refant_names : list of str, optional
        Reference antenna(s); defaults to the survey's own ``refant_names``.
    no_data_antennas : set of str, optional
        Antennas with no usable data at all (fully flagged or never observed).

    Returns
    -------
    list of dict
        One JSON-safe row per scan.
    """
    survey = metadata.snr_survey if metadata else None
    per_scan = survey.per_scan_antenna() if survey is not None and hasattr(survey, "per_scan_antenna") else {}
    all_antennas = list(metadata.antennas) if metadata else []
    roles = source_roles or {}
    references = list(refant_names) if refant_names is not None else list(getattr(survey, "refant_names", []) or [])
    dead = set(no_data_antennas or ())
    rows = []
    for scan in (metadata.scans if metadata else []):
        raw = per_scan.get(scan.scan_number, {})
        snrs = {name: _finite_or_none(raw.get(name)) for name in all_antennas}
        # The survey table is reloaded with field="", so scan_numbers covers the whole
        # observation: only a finite SNR proves the fringe fit actually ran on this scan.
        surveyed = any(value is not None for value in snrs.values())
        present = set(scan.antennas)
        states = {name: antenna_scan_state(name, name in present, snrs[name], surveyed, references, dead)
                  for name in all_antennas}
        rows.append({"number": scan.scan_number, "source": scan.source,
                     "role": roles.get(scan.source, "other"),
                     "start_utc": mjd_seconds_to_utc(scan.time_start), "duration_sec": round(scan.duration_sec),
                     "antennas": list(scan.antennas), "surveyed": surveyed,
                     "antenna_snrs": snrs, "antenna_states": states})
    return rows


def _finite_or_none(value) -> Optional[float]:
    """Return ``value`` as float when finite, otherwise None."""
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _download_entry(path: Path, work_dir: Path, label: str, kind: str) -> dict:
    """One per-source data product: label, paths (absolute + relative to ``work_dir``) and size."""
    import os
    is_dir = path.is_dir()
    size = None if is_dir else path.stat().st_size
    return {"kind": kind, "label": label, "path": str(path),
            "relpath": os.path.relpath(path, work_dir).replace("\\", "/"),
            "is_dir": is_dir, "size_bytes": size, "size_human": human_size(size)}


def _source_downloads(work_dir: Path, project_code: str, source_names: list[str]) -> dict[str, list[dict]]:
    """Per-source data products that exist on disk, keyed by source name.

    Looks for ``calibrated_data/<code>_<source>.uvfits``,
    ``calibrated_data/<code>_<source>.ms`` (a directory) and every
    ``images/<code>_<source>.robust*.fits``. Missing files are simply omitted, so
    a source with nothing on disk maps to an empty list.

    Parameters
    ----------
    work_dir : Path
        Project directory holding ``calibrated_data/`` and ``images/``.
    project_code : str
        Project code prefixing every product file name.
    source_names : list of str
        Sources to look for, in the order they should be displayed.

    Returns
    -------
    dict
        ``{source: [entry, ...]}`` with one :func:`_download_entry` per product.
    """
    data_dir, image_dir = Path(work_dir) / "calibrated_data", Path(work_dir) / "images"
    images = sorted(p.name for p in image_dir.iterdir()) if image_dir.is_dir() else []
    out: dict[str, list[dict]] = {}
    for name in source_names:
        entries = []
        uvfits, measurement_set = data_dir / f"{project_code}_{name}.uvfits", data_dir / f"{project_code}_{name}.ms"
        if uvfits.is_file():
            entries.append(_download_entry(uvfits, work_dir, "Calibrated UVFITS", "uvfits"))
        if measurement_set.is_dir():
            entries.append(_download_entry(measurement_set, work_dir, "Calibrated measurement set", "ms"))
        robusts = []
        for file_name in images:
            match = re.fullmatch(rf"{re.escape(project_code)}_{re.escape(name)}\.robust(-?\d+)\.fits", file_name)
            if match:
                robusts.append((int(match.group(1)), match.group(1), file_name))
        for _, robust, file_name in sorted(robusts):
            label = f"Image FITS (robust {robust})"
            entries.append(_download_entry(image_dir / file_name, work_dir, label, "fits"))
        out[name] = entries
    return out


def _read_json_object(path: Path) -> dict:
    """JSON object stored in ``path``; an empty dict when it is missing, unreadable or not an object."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("dashboard: could not read {} ({})", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def match_ifs_to_subbands(if_freqs_hz: list, channel_freqs: list, n_ifs: int, n_subbands: int) -> list[Optional[int]]:
    """Subband index behind every IF of a self-calibrated split (None where it cannot be told).

    The split measurement set does not have to hold every subband of the observation, so IF ``k`` is not
    necessarily subband ``k``. When the IF frequencies and the channel frequencies of the metadata are both
    known, an IF is the subband whose channel range contains its frequency (the nearest centre when several
    do; a subband is used once). Without frequencies, or when not one IF falls in a subband, the position is
    used - which is only safe when the split kept every subband (``n_ifs == n_subbands``).

    Parameters
    ----------
    if_freqs_hz : list of float
        Frequency (Hz) of every IF as the gain record reports it; may be empty.
    channel_freqs : list of list of float
        Channel frequencies (Hz) per subband from the metadata; may be empty.
    n_ifs : int
        Number of IFs in the gain record.
    n_subbands : int
        Number of subbands of the observation.

    Returns
    -------
    list of int or None
        One entry per IF.
    """
    bands = []
    for index, channels in enumerate(channel_freqs or []):
        values = [float(f) for f in channels]
        if values:
            step = abs(values[1] - values[0]) if len(values) > 1 else 0.0
            bands.append((index, min(values) - 0.5 * step, max(values) + 0.5 * step, sum(values) / len(values)))
    freqs = [_finite_or_none(f) for f in (if_freqs_hz or [])]
    if bands and len(freqs) == n_ifs and all(f is not None for f in freqs):
        matched: list[Optional[int]] = []
        for freq in freqs:
            inside = [(abs(freq - centre), index) for index, low, high, centre in bands
                      if low <= freq <= high and index not in matched]
            matched.append(min(inside)[1] if inside else None)
        if any(subband is not None for subband in matched):
            return matched
    return list(range(n_ifs)) if n_ifs == n_subbands else [None] * n_ifs


def read_gain_record(selfcal_dir: Path, project_code: str, source: str) -> dict:
    """Amplitude corrections of one self-calibrated source as difmapy left them on disk.

    ``<code>.<source>.gains.json`` (``difmap.write_gain_record``) holds the factors that were applied. Runs
    made before that file existed only have the Bayesian report ``<code>.<source>.amp.json``, whose
    ``corrections_per_if`` are the same numbers. The report is read whenever it exists, because only it says
    which station and IF had a solution at all: the gain record stores 1.0 for "no correction" and for
    "no solution" alike.

    Parameters
    ----------
    selfcal_dir : Path
        ``<work_dir>/selfcal``.
    project_code : str
        Project code prefixing the file names.
    source : str
        Source name.

    Returns
    -------
    dict
        Empty when neither file holds per-IF factors. Otherwise ``file`` (name of the file the factors come
        from), ``if_freqs_hz``, ``gain_per_if`` (station -> list over IFs), ``sigma`` and ``p_correction``
        (station -> float or None), ``stations`` (station -> its entry of the report; empty without a
        report), ``findings`` (sentences of the report) and ``applied`` (bool, None when unknown).
    """
    stem = f"{project_code}.{source}"
    gains = _read_json_object(selfcal_dir / f"{stem}.gains.json")
    report = _read_json_object(selfcal_dir / f"{stem}.amp.json")
    from_gains = isinstance(gains.get("gain_per_if"), dict) and bool(gains["gain_per_if"])
    per_if = gains["gain_per_if"] if from_gains else report.get("corrections_per_if")
    if not isinstance(per_if, dict) or not per_if:
        return {}
    listed = report.get("stations") if isinstance(report.get("stations"), list) else []
    stations = {str(s["station"]): s for s in listed if isinstance(s, dict) and s.get("station")}
    uncertainty = {}
    for key in ("sigma", "p_correction"):
        values = gains.get(key) if from_gains and isinstance(gains.get(key), dict) else None
        values = values if values is not None else {name: s.get(key) for name, s in stations.items()}
        uncertainty[key] = {str(name): _finite_or_none(value) for name, value in values.items()}
    if_freqs = (gains.get("if_freqs_hz") if from_gains else None) or report.get("if_freqs_hz") or []
    applied = report.get("applied")
    return {"file": f"{stem}.{'gains' if from_gains else 'amp'}.json",
            "if_freqs_hz": [_finite_or_none(f) for f in if_freqs] if isinstance(if_freqs, list) else [],
            "gain_per_if": {str(name): list(values) for name, values in per_if.items() if isinstance(values, list)},
            "sigma": uncertainty["sigma"], "p_correction": uncertainty["p_correction"], "stations": stations,
            "findings": [str(f) for f in report.get("findings") or []],
            "applied": applied if isinstance(applied, bool) else None}


def _report_has_solution(station: Optional[dict], index: int) -> Optional[bool]:
    """Whether the Bayesian report solved IF ``index`` (0-based) of a station; None when it does not say.

    ``station`` is the station's entry of the report (None when there is no report): ``has_data`` false
    means no solution anywhere, and an IF whose ``per_if`` entry has a null ``correction`` has none.
    """
    if not station:
        return None
    if station.get("has_data") is False:
        return False
    per_if = station.get("per_if") if isinstance(station.get("per_if"), list) else []
    # Entries carry their 1-based IF number; fall back to the list position when they do not.
    by_index = {(int(entry["if"]) - 1 if isinstance(entry.get("if"), int) else position): entry
                for position, entry in enumerate(per_if) if isinstance(entry, dict)}
    if index not in by_index:
        return None
    return _finite_or_none(by_index[index].get("correction")) is not None


def _metadata_has_solution(antenna: Optional[dict], subband: Optional[int]) -> bool:
    """Whether the metadata allow a solution: the antenna observed and, when known, recorded the subband.

    ``antenna`` is its entry of the context ``antennas`` list (None = unknown station, which is not held
    against it); an empty ``subbands`` list means the participation was never determined.
    """
    if antenna is None:
        return True
    if not antenna.get("observed", True):
        return False
    recorded = antenna.get("subbands") or []
    return subband is None or not recorded or subband in recorded


def station_gains(name: str, record: dict, antenna: Optional[dict],
                  subbands: list[Optional[int]]) -> list[Optional[float]]:
    """Applied amplitude factor of one station per IF, None where it has no solution.

    A gain record stores 1.0 both for "no correction needed" and for "no solution", so a solved IF is told
    from an unsolved one with what else is known. The Bayesian report decides when it has the station
    (:func:`_report_has_solution`). Without it, a station the gain record gives no uncertainty for has no
    solution at all, and the metadata decide the rest (:func:`_metadata_has_solution`).

    Parameters
    ----------
    name : str
        Station code.
    record : dict
        Result of :func:`read_gain_record`.
    antenna : dict or None
        The station's entry of the context ``antennas`` list (``observed``, ``subbands``); None when unknown.
    subbands : list of int or None
        Subband index behind every IF (:func:`match_ifs_to_subbands`); its length is the number of IFs.

    Returns
    -------
    list of float or None
        One value per IF.
    """
    raw = record["gain_per_if"].get(name)
    if raw is None:
        return [None] * len(subbands)
    values = [_finite_or_none(value) for value in raw][:len(subbands)]
    values += [None] * (len(subbands) - len(values))
    station = record["stations"].get(name)
    sigma = record["sigma"]
    unsolved_station = station is None and bool(sigma) and sigma.get(name) is None
    gains = []
    for index, value in enumerate(values):
        solved = _report_has_solution(station, index)
        if solved is None:
            solved = not unsolved_station and _metadata_has_solution(antenna, subbands[index])
        gains.append(value if solved else None)
    return gains


def gain_correction_entry(record: dict, source: dict, antennas: list[dict], channel_freqs: list,
                          n_subbands: int) -> dict:
    """Amplitude-correction table of one source: applied factor per antenna and subband.

    Rows are the antennas of the observation in metadata order (then any station only the record knows).
    Columns are the IFs of the self-calibrated split, each labelled with the subband it is. The median of an
    antenna is taken over the subbands it has a solution in.

    Parameters
    ----------
    record : dict
        Result of :func:`read_gain_record` (not empty).
    source : dict
        Source row of the context (``name``, ``type``).
    antennas : list of dict
        Context ``antennas`` list (``name``, ``observed``, ``subbands``).
    channel_freqs : list of list of float
        Channel frequencies (Hz) per subband from the metadata, used to match IFs to subbands.
    n_subbands : int
        Number of subbands of the observation.

    Returns
    -------
    dict
        JSON-safe entry of ``ctx["gain_corrections"]``: ``source``, ``role``, ``file`` (relative to the work
        directory), ``applied`` (bool, None when unknown), ``antennas``, ``subbands`` (subband index per
        column, None when unknown), ``subband_labels``, ``if_freqs_hz``, ``gain_per_if`` and ``gain_median``
        (None = no solution), ``sigma``, ``p_correction``, ``has_data`` and ``naive_gscale`` (per antenna),
        and ``findings`` (sentences of the Bayesian report).
    """
    per_if = record["gain_per_if"]
    n_ifs = max((len(values) for values in per_if.values()), default=0)
    subbands = match_ifs_to_subbands(record["if_freqs_hz"], channel_freqs, n_ifs, n_subbands)
    known = {a["name"]: a for a in antennas}
    names = list(known) + [name for name in per_if if name not in known]
    gains = {name: station_gains(name, record, known.get(name), subbands) for name in names}
    solved = {name: [g for g in gains[name] if g is not None] for name in names}
    stations = record["stations"]
    return {"source": source["name"], "role": source.get("type", "other"), "file": f"selfcal/{record['file']}",
            "applied": record["applied"], "antennas": names, "subbands": subbands,
            "subband_labels": [f"SB {s}" if s is not None else f"IF {i + 1}" for i, s in enumerate(subbands)],
            "if_freqs_hz": (list(record["if_freqs_hz"]) + [None] * n_ifs)[:n_ifs],
            "gain_per_if": gains,
            "gain_median": {name: float(statistics.median(solved[name])) if solved[name] else None for name in names},
            "sigma": {name: record["sigma"].get(name) for name in names},
            "p_correction": {name: record["p_correction"].get(name) for name in names},
            "has_data": {name: bool(solved[name]) for name in names},
            "naive_gscale": {name: _finite_or_none((stations.get(name) or {}).get("naive_gscale")) for name in names},
            "findings": list(record["findings"])}


def _gain_corrections(work_dir: Path, project_code: str, sources: list[dict], antennas: list[dict],
                      channel_freqs: list, n_subbands: int) -> list[dict]:
    """Amplitude corrections of the difmapy self-calibration found under ``<work_dir>/selfcal``.

    Parameters
    ----------
    work_dir : Path
        Project directory.
    project_code : str
        Project code prefixing the self-calibration file names.
    sources : list of dict
        Context source rows; they set the order of the result.
    antennas : list of dict
        Context ``antennas`` list.
    channel_freqs : list of list of float
        Channel frequencies (Hz) per subband from the metadata.
    n_subbands : int
        Number of subbands of the observation.

    Returns
    -------
    list of dict
        One :func:`gain_correction_entry` per source that has a gain record or a Bayesian report; a source
        without either, or whose files cannot be used, is left out.
    """
    selfcal_dir = Path(work_dir) / "selfcal"
    if not selfcal_dir.is_dir():
        return []
    entries = []
    for source in sources:
        try:
            record = read_gain_record(selfcal_dir, project_code, source["name"])
            if record:
                entries.append(gain_correction_entry(record, source, antennas, channel_freqs, n_subbands))
        except Exception as exc:  # noqa: BLE001 - a malformed report must not cost the whole dashboard
            logger.warning("dashboard: amplitude corrections of {} not read ({})", source["name"], exc)
    return entries


def _calibration_selection(observation, scans: list[dict]) -> list[dict]:
    """Stages of the instrumental (single-band delay) selection, each with the source of its scan.

    Parameters
    ----------
    observation : Observation
        Its ``cal_stages`` (read back from ``.cal_selection.json``) are the stage plan. A selection saved
        before stages existed is only usable when it holds a single scan: with several scans it does not
        say which antenna was solved on which.
    scans : list of dict
        Context scan rows (``number``, ``source``).

    Returns
    -------
    list of dict
        ``{"scan", "source", "refant", "antennas"}`` per stage, in solving order; empty when undecided.
    """
    source_of = {scan["number"]: scan["source"] for scan in scans}
    stages = list(getattr(observation, "cal_stages", None) or [])
    selection = getattr(observation, "cal_selection", None)
    if not stages and selection and len(selection[1]) == 1:
        stages = [{"scan": selection[1][0], "antennas": list(selection[0]), "refant": ""}]
    selected = []
    for stage in stages:
        number = _finite_or_none(stage.get("scan"))
        if number is None:
            continue
        selected.append({"scan": int(number), "source": source_of.get(int(number), ""),
                         "refant": str(stage.get("refant") or ""),
                         "antennas": [str(name) for name in stage.get("antennas") or []]})
    return selected


def _polarization_snr(survey, metadata) -> dict:
    """Detections per antenna and parallel hand in the fringe-SNR survey.

    A value counts as a detection when it is finite and at least ``FRINGE_DETECTION_SNR``. Only surveyed
    scans (some antenna has a finite SNR) the antenna took part in are counted.

    Parameters
    ----------
    survey : ScanSNRSurvey or None
        The survey of the observation.
    metadata : ObsMetadata or None
        Its scan list says which antennas were in each scan.

    Returns
    -------
    dict
        ``{"hands": [...], "antennas": {name: {"scans": int, "detected": {hand: int},
        "median": {hand: float or None}}}}``; empty lists/dicts without a survey.
    """
    if survey is None or metadata is None or not getattr(survey, "snr", None):
        return {"hands": [], "antennas": {}}
    hands = list(survey.polarizations)
    present = {int(scan.scan_number): set(scan.antennas) for scan in metadata.scans}
    surveyed = [row for row in range(len(survey.scan_numbers))
                if any(_finite_or_none(value) is not None for hand in hands for value in survey.snr[hand][row])]
    antennas = {}
    for column, name in enumerate(survey.antennas):
        own = [row for row in surveyed if name in present.get(int(survey.scan_numbers[row]), ())]
        detected, median = {}, {}
        for hand in hands:
            values = [_finite_or_none(survey.snr[hand][row][column]) for row in own]
            values = [value for value in values if value is not None and value >= FRINGE_DETECTION_SNR]
            detected[hand] = len(values)
            median[hand] = float(statistics.median(values)) if values else None
        antennas[name] = {"scans": len(own), "detected": detected, "median": median}
    return {"hands": hands, "antennas": antennas}


def _step_statuses(observation) -> dict:
    """Recorded pipeline steps as ``{step: {"status": str, "error": str}}`` (``error`` only when one is stored)."""
    state = getattr(observation, "_state", None)
    steps = state.as_dict() if state is not None else {}
    statuses = {}
    for name, entry in steps.items():
        statuses[name] = {"status": str(entry.get("status") or "")}
        if entry.get("error"):
            statuses[name]["error"] = str(entry["error"])
    return statuses


def _source_results(work_dir: Path, project_code: str, sources: list[dict]) -> list[dict]:
    """What the imaging left on disk per source: the source search and the image statistics.

    Reads ``images/<code>_<source>.search.json`` (written for every source difmapy imaged, with the
    statistics of its latest images) and, for images made before that report carried them, the
    self-calibration report ``selfcal/<code>.<source>.json`` (calibrators only), whose ``images`` hold the
    peak, rms and dynamic range per robust value. The images themselves are never opened.

    Parameters
    ----------
    work_dir : Path
        Project directory.
    project_code : str
        Project code prefixing the file names.
    sources : list of dict
        Context source rows; they set the order of the result.

    Returns
    -------
    list of dict
        ``{"source", "role", "search", "images"}`` per source that has either: ``search`` is None or a dict
        with ``detected``, ``recentred`` (bool) and ``snr``, ``peak``, ``rms``, ``fov_mas``, ``offset_mas``,
        ``offset_beams`` (float or None); ``images`` is a list of ``{"robust", "peak", "rms",
        "dynamic_range"}`` ordered by robust.
    """
    results = []
    for source in sources:
        name = source["name"]
        found = _read_json_object(Path(work_dir) / "images" / f"{project_code}_{name}.search.json")
        report = _read_json_object(Path(work_dir) / "selfcal" / f"{project_code}.{name}.json")
        search = found.get("search") or report.get("search")
        if isinstance(search, dict) and search:
            numbers = ("snr", "peak", "rms", "fov_mas", "offset_mas", "offset_beams")
            search = {"detected": bool(search.get("detected")), "recentred": bool(search.get("recentred")),
                      **{key: _finite_or_none(search.get(key)) for key in numbers}}
        else:
            search = None
        # The image report is rewritten whenever a source is imaged (a campaign re-images after the self-cal).
        recorded = next((images for images in (found.get("images"), report.get("images"))
                         if isinstance(images, dict) and images), {})
        measured = [info for info in recorded.values()
                    if isinstance(info, dict) and _finite_or_none(info.get("peak")) is not None]
        images = [{key: _finite_or_none(info.get(key)) for key in ("robust", "peak", "rms", "dynamic_range")}
                  for info in measured]
        if search is None and not images:
            continue
        results.append({"source": name, "role": source.get("type", "other"), "search": search,
                        "images": sorted(images, key=lambda image: (image["robust"] is None, image["robust"] or 0.0))})
    return results


def collect(obs, *, flag_statistics: Optional[dict] = None) -> dict:
    """Gather everything the dashboard renders into one JSON-serialisable dict.

    Parameters
    ----------
    obs : VLBIObs or Observation
        Observation whose metadata, sources, calibration chain and flag statistics are read.
    flag_statistics : dict, optional
        Result of ``obs.flag.statistics()``; defaults to the last one stored on the observation.

    Returns
    -------
    dict
        The dashboard context. Besides the observation, antenna, flagging, source, scan, download and
        calibration-table entries it carries ``gain_corrections``: one :func:`gain_correction_entry` per
        source the difmapy self-calibration left amplitude corrections for (empty list when there are none).
        The inputs of the general comments follow - ``calibration_selection``
        (:func:`_calibration_selection`), ``polarization_snr`` (:func:`_polarization_snr`), ``steps``
        (:func:`_step_statuses`), ``source_results`` (:func:`_source_results`) and ``run_warnings``
        (:func:`vlbipy.comments.load_run_warnings`) - and ``comments``, the list
        :func:`vlbipy.comments.build_comments` derives from all of the above.
    """
    observation = _primary_observation(obs)
    metadata = observation.metadata
    if metadata is not None and metadata.snr_survey is None:
        loader = getattr(getattr(observation._backend, "calibrate", None), "read_snr_table", None)
        if loader is not None:
            try:
                metadata.snr_survey = loader(observation.project_code, metadata, observation.refant)
            except Exception:  # noqa: BLE001 - the chips just render "not surveyed"
                logger.debug("dashboard: could not reload the SNR survey table")
    freq = metadata.freq_setup if metadata else None
    stats = flag_statistics if flag_statistics is not None else (observation.flag_statistics or {})
    if not stats and observation._backend.requires_data_files:
        cached = Path(observation.work_dir) / ".flag_statistics.json"
        if cached.is_file():
            try:
                stats = json.loads(cached.read_text())
            except Exception:  # noqa: BLE001 - corrupt cache just means no flag panel
                logger.debug("dashboard: could not read {}", cached)
    antennas = []
    for ant_id, (name, ant) in enumerate((metadata.antennas if metadata else {}).items()):
        ant_stats = (stats.get("antenna", {}).get(name) or {}) if stats else {}
        antennas.append({"id": ant_id, "name": name, "observed": bool(ant.observed),
                         "subbands": list(ant.subbands or ()), "n_scans": int(ant.n_scans or 0),
                         "flagged": _finite_or_none(ant_stats.get("flagged")),
                         "observable": _finite_or_none(ant_stats.get("observable")),
                         "flagged_fraction": _finite_or_none(ant_stats.get("fraction"))})
    # New caches carry a direct baseline-weighted count with dead stations removed.
    # Do not reconstruct it from per-antenna totals for old caches: each surviving
    # baseline appears at both endpoints and such a sum is misleading.
    excluding_dead = stats.get("excluding_dead") if stats else None
    if excluding_dead is not None:
        excluding_dead = {"flagged": _finite_or_none(excluding_dead.get("flagged")),
                          "observable": _finite_or_none(excluding_dead.get("observable")),
                          "fraction": _finite_or_none(excluding_dead.get("fraction")),
                          "excluded_antennas": list(excluding_dead.get("excluded_antennas", [])),
                          "available": True}
    else:
        excluding_dead = {"flagged": None, "observable": None, "fraction": None,
                          "excluded_antennas": [], "available": False}
    start, end = metadata.time_range if metadata else (0.0, 0.0)
    sources = _source_rows(observation, metadata)
    source_roles = {s["name"]: s["type"] for s in sources}
    survey = metadata.snr_survey if metadata else None
    refant_names = list(getattr(survey, "refant_names", []) or [])
    no_data_antennas = {a["name"] for a in antennas if not a["observed"]
                        or (a["flagged_fraction"] is not None and a["flagged_fraction"] >= NO_DATA_FLAGGED_FRACTION)}
    context = {
        "project": observation.project_code, "observatory": observation.observatory,
        "backend": observation._backend.kind, "refant": observation.refant,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "observation": {
            "date": format_obs_date(metadata.obs_date) if metadata else "",
            "start_utc": mjd_seconds_to_utc(start), "end_utc": mjd_seconds_to_utc(end),
            "duration_hours": round((end - start) / 3600.0, 2) if end > start else 0.0,
            "ref_freq_ghz": round(freq.freq_ghz, 4) if freq else None,
            "bandwidth_mhz": round(freq.bandwidth_mhz, 2) if freq else None,
            "n_subbands": freq.n_subbands if freq else 0, "n_channels": freq.n_channels if freq else 0,
            "channel_width_khz": round(freq.channel_width / 1e3, 2) if freq else None,
            "polarizations": [str(getattr(p, "name", p)) for p in (freq.polarizations if freq else [])],
        },
        "antennas": antennas,
        "flagging": {"fraction": _finite_or_none(stats.get("fraction")),
                     "antenna": {a["name"]: {"flagged": a["flagged"], "observable": a["observable"],
                                             "fraction": a["flagged_fraction"]} for a in antennas},
                     "excluding_dead": excluding_dead},
        "sources": sources,
        "scans": _scan_rows(metadata, source_roles, refant_names, no_data_antennas),
        "downloads": _source_downloads(Path(observation.work_dir), observation.project_code,
                                       [s["name"] for s in sources]),
        "gain_corrections": _gain_corrections(Path(observation.work_dir), observation.project_code, sources, antennas,
                                              list(freq.channel_freqs) if freq else [], freq.n_subbands if freq else 0),
        "caltables": [t.to_dict() for t in observation.gaintables],
        "work_dir": str(observation.work_dir),
    }
    # What only the general comments need. Every piece comes from a cache on disk (or from the
    # observation object), never from the measurement set: the report is rebuilt after it is gone.
    context["calibration_selection"] = _calibration_selection(observation, context["scans"])
    context["polarization_snr"] = _polarization_snr(survey, metadata)
    context["steps"] = _step_statuses(observation)
    context["source_results"] = _source_results(Path(observation.work_dir), observation.project_code, sources)
    context["run_warnings"] = load_run_warnings(Path(observation.work_dir), observation.project_code)
    context["comments"] = build_comments(context)
    return context


# -- rendering ---------------------------------------------------------------------------------

CSS = """
:root{--ground:#f5f8fa;--panel:#fff;--panel2:#eef3f6;--border:#d7e0e6;--ink:#15222c;--muted:#5b6b78;
  --accent:#0c8f96;--accent-ink:#0a6b70;--good:#158a63;--warn:#a5741a;--crit:#c0384a;--mid:#c9a400;--role-ff:#c0303f;--role-pc:#2463c9;--role-tg:#d2690f;
  --shadow:0 1px 2px rgba(20,40,55,.06),0 8px 24px rgba(20,40,55,.06)}
:root[data-theme="dark"]{--ground:#080d12;--panel:#101922;--panel2:#0b141c;--border:#1d2b37;--ink:#d9e6ee;
  --muted:#7f94a3;--accent:#37d3d9;--accent-ink:#8ff0f3;--good:#3ad39a;--warn:#e7b755;--crit:#e5586b;--mid:#e8d64b;--role-ff:#ff6b7a;--role-pc:#6aa5ff;--role-tg:#ffa24a;
  --shadow:0 1px 0 rgba(0,0,0,.4),0 12px 32px rgba(0,0,0,.35)}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);line-height:1.5;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
a{color:var(--accent-ink);text-decoration:none}
.wrap{display:grid;grid-template-columns:230px 1fr;min-height:100vh}
.side{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;border-right:1px solid var(--border);
  background:var(--panel2);padding:22px 16px}
.brand{font-weight:700;font-size:15px}
.brand .code{color:var(--accent)}
.brand small{display:block;color:var(--muted);font-weight:500;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;margin-top:4px}
.tabs{margin-top:24px;display:flex;flex-direction:column;gap:2px}
.tabs a{display:block;padding:9px 12px;border-radius:8px;color:var(--ink);font-size:14px;font-weight:600}
.tabs a:hover{background:var(--panel)}
.tabs a.active{background:var(--panel);color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}
.nav{margin-top:18px;display:flex;flex-direction:column;gap:1px;border-top:1px solid var(--border);padding-top:12px}
.nav a{display:block;padding:5px 12px;border-radius:6px;color:var(--muted);font-size:12.5px}
.nav a:hover,.nav a.active{color:var(--accent)}
.main{padding:34px 40px 80px;max-width:1280px}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--accent);margin:0 0 6px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:0 0 4px}
h3{font-size:15px;margin:22px 0 4px}
.sub{color:var(--muted);margin:0 0 18px;max-width:80ch}
section{scroll-margin-top:18px;padding:26px 0 8px;border-top:1px solid var(--border)}
section:first-of-type{border-top:0;padding-top:8px}
.facts{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 8px}
.fact{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 14px;min-width:120px;
  box-shadow:var(--shadow)}
.fact .k{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.fact .v{font-size:15px;font-weight:600;margin-top:2px}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 12px;white-space:nowrap;vertical-align:middle}
thead th{background:var(--panel2);color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;border-bottom:1px solid var(--border)}
tbody tr{border-top:1px solid var(--border)}
tbody tr:nth-child(odd){background:var(--panel)}
td.num{font-variant-numeric:tabular-nums;text-align:right}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;margin:1px 2px}
.chip.good{background:var(--panel2);background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.chip.warn{background:var(--panel2);background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.chip.crit{background:var(--panel2);background:color-mix(in srgb,var(--crit) 20%,transparent);color:var(--crit)}
.chip.mut{background:var(--panel2);color:var(--muted)}
.chip.snr-good{background:var(--good);color:#fff}
.chip.snr-mid{background:var(--mid);color:#222}
.chip.snr-low{background:#e07b1a;color:#fff}
.chip.snr-bad{background:var(--crit);color:#fff}
.chip.snr-none{background:var(--panel2);color:var(--muted);border:1px solid var(--border)}
.chip.snr-absent{background:var(--panel2);color:var(--muted);border:1px solid var(--border)}
.chip.snr-ref{background:var(--panel2);background:color-mix(in srgb,var(--accent) 22%,transparent);color:var(--accent-ink);border:1px solid var(--accent)}
.chip.snr-ref .refmark{font-style:normal;font-size:8.5px;font-weight:700;letter-spacing:.04em;margin-left:2px;
  opacity:.9;text-transform:uppercase}
.chip.snr-nodata{background:var(--panel2);color:var(--muted);border:1px dashed var(--crit);text-decoration:line-through}
.chip.snr-nofringe{background:var(--panel2);background:color-mix(in srgb,var(--crit) 16%,transparent);color:var(--crit);border:1px solid var(--border)}
.chip.scan-chip{min-width:34px;height:20px;text-align:center;display:inline-flex;align-items:center;justify-content:center;vertical-align:middle}
.downloads{list-style:none;margin:8px 0 4px;padding:0;display:flex;flex-direction:column;gap:6px}
.downloads li{display:flex;flex-wrap:wrap;align-items:center;gap:8px;background:var(--panel);
  border:1px solid var(--border);border-radius:10px;padding:8px 12px;box-shadow:var(--shadow)}
.downloads a{font-weight:600}
.downloads .dl-path{color:var(--muted);font-size:12px;background:var(--panel2);border-radius:6px;padding:1px 6px}
.legend{display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:12px;color:var(--muted);margin:6px 0}
.badge{display:inline-block;padding:1px 8px;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:.04em;
  text-transform:uppercase}
.badge.fringefinder{background:var(--panel2);background:color-mix(in srgb,var(--role-ff) 20%,transparent);color:var(--role-ff)}
.badge.phasecal{background:var(--panel2);background:color-mix(in srgb,var(--role-pc) 20%,transparent);color:var(--role-pc)}
.badge.target{background:var(--panel2);background:color-mix(in srgb,var(--role-tg) 22%,transparent);color:var(--role-tg)}
.badge.other,.badge.checksource,.badge.polcal{background:var(--panel2);color:var(--muted)}
.sbbar{display:inline-flex;gap:2px;height:10px;width:160px;vertical-align:middle}
.sbbar i{flex:1;border-radius:2px;background:var(--panel2);border:1px solid var(--border)}
.sbbar i.on{background:var(--accent);border-color:var(--accent)}
.hbar{position:relative;display:inline-block;width:120px;height:9px;border-radius:5px;background:var(--panel2);
  overflow:hidden;vertical-align:middle}
.hbar>i{position:absolute;top:0;left:0;bottom:0;border-radius:5px}
.hbar>i.good{background:var(--good)} .hbar>i.warn{background:var(--warn)} .hbar>i.crit{background:var(--crit)}
.antlist{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(380px,100%),1fr));gap:16px;margin-top:14px}
.grid.wide{grid-template-columns:1fr}
figure{margin:0;background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:hidden;
  box-shadow:var(--shadow);display:flex;flex-direction:column}
figure button.imgbtn{border:0;padding:0;background:#0a1016;cursor:zoom-in;display:block}
figure img{width:100%;height:auto;display:block}
figcaption{padding:10px 13px}
figcaption .t{font-weight:600;font-size:13px}
figcaption .d{color:var(--muted);font-size:12px;margin-top:3px}
.skymap{max-width:460px;margin-top:8px;margin-bottom:14px}
.skymap svg{display:block;width:100%;height:auto}
.skymap .axis,.skymap .ring{fill:none;stroke:var(--border);stroke-width:1}
.skymap .ringlabel,.skymap .seplabel{fill:var(--muted);font-size:10.5px;font-family:ui-monospace,Menlo,monospace}
.skymap .compass{fill:var(--muted);font-weight:700;font-size:12px}
.skymap .pair{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 3}
.skymap .leader{stroke:var(--border);stroke-width:1}
.skymap .srclabel{fill:var(--ink);font-size:11.5px;font-weight:600}
.skymap text{paint-order:stroke;stroke:var(--panel);stroke-width:3px;stroke-linejoin:round}
.skymap .src{fill:var(--muted);stroke:var(--panel);stroke-width:1.5}
.skymap .src.fringefinder{fill:var(--role-ff)} .skymap .src.phasecal{fill:var(--role-pc)} .skymap .src.target{fill:var(--role-tg)}
.count{color:var(--muted);font-size:12.5px;font-weight:600;font-family:ui-monospace,Menlo,monospace}
.empty{color:var(--muted);font-style:italic;padding:8px 0}
.subtabs-wrap{margin:12px 0}
.subtabs-wrap .subtabs{margin:0}
.subtabs{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}
.subtabs button{border:1px solid var(--border);background:var(--panel);color:var(--ink);border-radius:8px;
  padding:6px 12px;font-size:13px;cursor:pointer}
.subtabs button.active{background:var(--accent);border-color:var(--accent);color:#fff}
.subpanel{display:none} .subpanel.active{display:block}
:root{--gain-good:#bfe6cf;--gain-good-ink:#15222c;--gain-warn:#ef9a30;--gain-warn-ink:#15222c;
  --gain-bad:#c0384a;--gain-bad-ink:#fff}
:root[data-theme="dark"]{--gain-good:#17503d;--gain-good-ink:#fff;--gain-warn:#a85f10;--gain-warn-ink:#fff;
  --gain-bad:#ff8090;--gain-bad-ink:#15222c}
.subtabs button.active .badge{background:rgba(255,255,255,.22);color:inherit}
.gainwrap{display:inline-block;max-width:100%;vertical-align:top;scrollbar-color:var(--muted) var(--panel2)}
.gaintable{width:auto}
.gaintable th,.gaintable td{text-align:center;padding:6px 8px;border:0}
.gaintable thead th{border-bottom:1px solid var(--border)}
.gaintable thead th small{display:block;font-size:10px;font-weight:500;letter-spacing:0;text-transform:none}
.gaintable tbody tr,.gaintable tbody tr:nth-child(odd){border-top:0;background:var(--panel)}
.gaintable .gain-ant{text-align:left;position:sticky;left:0;z-index:1}
.gaintable td.gain-ant{background:var(--panel)}
.gaintable td.gain{font-variant-numeric:tabular-nums;min-width:46px}
.gaintable .gain-median{position:sticky;right:0;z-index:1;font-weight:700;box-shadow:inset 2px 0 0 var(--ink)}
.gain-good{background:var(--gain-good);color:var(--gain-good-ink)}
.gain-warn{background:var(--gain-warn);color:var(--gain-warn-ink)}
.gain-bad{background:var(--gain-bad);color:var(--gain-bad-ink);font-weight:700}
.gain-none{background:var(--panel);color:var(--muted)}
.gainkey{display:inline-block;padding:1px 9px;font-size:11.5px;font-weight:600}
.gainkey.gain-none{border:1px solid var(--border)}
.comments{list-style:none;margin:6px 0 12px;padding:0;display:flex;flex-direction:column;gap:4px;max-width:1160px}
.comments li{display:flex;gap:10px;align-items:baseline;background:var(--panel);border:1px solid var(--border);
  border-left:3px solid var(--border);border-radius:8px;padding:6px 11px;font-size:13.5px}
.comments li.cmt-problem{border-left-color:var(--crit)}
.comments li.cmt-warning{border-left-color:var(--warn)}
.comments li .chip{flex:none;min-width:66px;text-align:center;margin:0}
.comments li .when{color:var(--muted);font-size:12px;white-space:nowrap;margin-left:6px}
.cmt-ant{display:grid;grid-template-columns:48px minmax(0,1fr);gap:0 10px;align-items:start;max-width:1160px}
.cmt-ant .cmt-name{font-weight:700;font-size:14px;padding-top:7px}
.cmt-ant .comments{margin:0 0 8px;max-width:none}
.cmt-srcs{display:flex;flex-direction:column;gap:10px;margin:6px 0 12px;max-width:1160px}
.cmt-src{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:9px 11px 3px}
.cmt-src-head{font-size:14px;display:flex;align-items:center;gap:8px}
.cmt-src .comments{margin:7px 0 6px;max-width:none}
.lb{position:fixed;top:0;right:0;bottom:0;left:0;background:rgba(4,8,11,.86);display:none;align-items:center;
  justify-content:center;padding:28px;z-index:50;cursor:zoom-out}
.lb.open{display:flex}
.lb img{max-width:96vw;max-height:92vh;border-radius:8px}
.lb .cap{position:fixed;bottom:16px;left:0;right:0;text-align:center;color:#cfe0ea;font-size:13px}
.themetoggle{margin-top:22px;width:100%;border:1px solid var(--border);background:var(--panel);color:var(--muted);
  border-radius:8px;padding:7px;font-size:12px;cursor:pointer}
@media (max-width:820px){.wrap{grid-template-columns:1fr}
  .side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--border);overflow:visible}
  .tabs,.nav{flex-direction:row;flex-wrap:wrap}
  .tabs a.active{box-shadow:inset 0 -3px 0 var(--accent)}
  .main{padding:24px 18px 60px}}
"""

JS = """
function vlbipyStorageGet(k){try{return localStorage.getItem(k)}catch(e){return null}}
function vlbipyStorageSet(k,v){try{localStorage.setItem(k,v)}catch(e){}}
function vlbipyShowPanel(panel){
  panel.classList.add('active');
  panel.querySelectorAll('img').forEach(function(i){
    i.loading='eager';
    if(!i.complete&&!i.src&&i.dataset.src)i.src=i.dataset.src;});}
const lb=document.getElementById('lb'),lbimg=document.getElementById('lbimg'),lbcap=document.getElementById('lbcap');
document.querySelectorAll('.imgbtn').forEach(b=>b.addEventListener('click',()=>{
  const im=b.querySelector('img');if(!im)return;
  lbimg.src=im.currentSrc||im.src;lbcap.textContent=b.dataset.cap||'';lb.classList.add('open');}));
lb.addEventListener('click',()=>lb.classList.remove('open'));
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('open');});
const links=[...document.querySelectorAll('.nav a')];
const secs=links.map(a=>document.querySelector(a.getAttribute('href'))).filter(Boolean);
const io=new IntersectionObserver(es=>{es.forEach(en=>{if(en.isIntersecting){
  links.forEach(l=>l.classList.toggle('active',l.getAttribute('href')==='#'+en.target.id));}});},
  {rootMargin:'-45% 0px -45% 0px'});
secs.forEach(s=>io.observe(s));
document.querySelectorAll('.subtabs').forEach(group=>{
  group.addEventListener('click',e=>{
    const btn=e.target.closest('button');if(!btn||!group.contains(btn))return;
    const wrap=group.closest('.subtabs-wrap')||group.parentElement;
    wrap.querySelectorAll('.subtabs button').forEach(b=>b.classList.toggle('active',b===btn));
    wrap.querySelectorAll('.subpanel').forEach(p=>{
      const on=p.id===btn.dataset.panel;p.classList.toggle('active',on);
      if(on)vlbipyShowPanel(p);});});});
const stored=vlbipyStorageGet('vlbipy-theme');
if(stored)document.documentElement.setAttribute('data-theme',stored);
else if(matchMedia('(prefers-color-scheme:dark)').matches)document.documentElement.setAttribute('data-theme','dark');
document.getElementById('themetoggle').addEventListener('click',()=>{
  const next=document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',next);vlbipyStorageSet('vlbipy-theme',next);});
"""


def _figure(path: Path, html_dir: Path, title: str, caption: str = "") -> str:
    """One ``<figure>`` with a click-to-zoom image referenced relatively from ``html_dir``."""
    href, cap = relative_href(path, html_dir), escape(f"{title} - {caption}" if caption else title)
    return (f'<figure><button class="imgbtn" data-cap="{cap}"><img loading="lazy" src="{escape(href)}" '
            f'alt="{escape(title)}"></button><figcaption><div class="t">{escape(title)}</div>'
            + (f'<div class="d">{escape(caption)}</div>' if caption else "") + "</figcaption></figure>")


def _plot_grid(paths: list[Path], html_dir: Path, project_code: str, caption: str = "", wide: bool = False) -> str:
    """Grid of figures, or an "empty" notice when there are no plots."""
    if not paths:
        return '<p class="empty">No plots available for this section.</p>'
    figs = "".join(_figure(p, html_dir, human_title(p, project_code), caption) for p in paths)
    return f'<div class="grid{" wide" if wide else ""}">{figs}</div>'


def _section(anchor: str, title: str, body: str, blurb: str = "", count: str = "") -> str:
    """One page section with an anchor for the in-page navigation."""
    badge = f' <span class="count">{escape(count)}</span>' if count else ""
    sub = f'<p class="sub">{escape(blurb)}</p>' if blurb else ""
    return f'<section id="{anchor}"><h2>{escape(title)}{badge}</h2>{sub}{body}</section>'


def _table(headers: list[str], rows: list[str]) -> str:
    """HTML table from header labels and pre-rendered ``<tr>`` rows."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


SKY_MAP_RING_STEPS_DEG = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90)
SKY_MAP_MAX_RINGS = 4


def _unit_vector(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    """Cartesian unit vector of a sky position given in degrees."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    return math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)


def angular_offset(ra0: float, dec0: float, ra: float, dec: float) -> tuple[float, float]:
    """Great-circle distance and position angle (east of north) of ``(ra, dec)`` seen from ``(ra0, dec0)``.

    All inputs and both outputs are in degrees; the position angle lies in ``[0, 360)``.
    """
    a0, d0, a, d = (math.radians(v) for v in (ra0, dec0, ra, dec))
    dra = a - a0
    north = math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(dra)
    east = math.cos(d) * math.sin(dra)
    rho = math.atan2(math.hypot(east, north), math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(dra))
    return math.degrees(rho), math.degrees(math.atan2(east, north)) % 360.0


def _format_separation(deg: float) -> str:
    """Angular separation as text: arcminutes below one degree, degrees above."""
    if deg < 1:
        return f"{deg * 60:.1f}\u2032"
    return f"{deg:.2f}\u00b0" if deg < 10 else f"{deg:.1f}\u00b0"


def _format_ring(deg: float) -> str:
    """Ring radius of the sky map as a round number: arcminutes below one degree, degrees above."""
    return f"{round(deg * 60, 3):g}\u2032" if deg < 1 else f"{deg:g}\u00b0"


def _sky_map_labels(names: list[str], points: list[tuple[float, float]], flip_x: float) -> list[tuple[float, float, str]]:
    """Place the source labels of the sky map so that they do not overlap.

    Parameters
    ----------
    names : list of str
        Label texts, one per point.
    points : list of (float, float)
        SVG positions of the sources.
    flip_x : float
        Points to the right of this x get their label on their left (``text-anchor="end"``).

    Returns
    -------
    list of (float, float, str)
        ``(x, y, text-anchor)`` per point, in input order. Labels are placed top to bottom and a label that would
        overlap one already placed moves down (then up) in steps of one line; the width is estimated from the
        number of characters.
    """
    placed_boxes, result = [], [None] * len(names)
    for i in sorted(range(len(names)), key=lambda k: points[k][1]):
        x, y = points[i]
        anchor = "end" if x > flip_x else "start"
        lx = x - 9 if anchor == "end" else x + 9
        width = 7.0 * len(names[i])
        x0, x1 = (lx - width, lx) if anchor == "end" else (lx, lx + width)
        ly = y + 4
        for shift in (0, 13, -13, 26, -26, 39, -39, 52):
            top = y + 4 + shift - 11
            if not any(x0 < bx1 and bx0 < x1 and top < by1 and by0 < top + 14 for bx0, bx1, by0, by1 in placed_boxes):
                ly = y + 4 + shift
                break
        placed_boxes.append((x0, x1, ly - 11, ly + 3))
        result[i] = (lx, ly, anchor)
    return result


def _sky_map(sources: list[dict]) -> str:
    """Inline SVG of the source positions relative to each other.

    Parameters
    ----------
    sources : list of dict
        Source rows from :func:`_source_rows`; rows without ``ra_deg``/``dec_deg`` (older ``dashboard.json``
        snapshots, unknown coordinates) are skipped.

    Returns
    -------
    str
        A ``<figure class="skymap">`` with an azimuthal equidistant projection about the mean direction of the
        sources (north up, east left), so the distance from the centre and the direction to every source are
        true. Rings mark angular distance from the centre and every target is joined to its nearest phase
        calibrator with the separation written along the line. Empty when fewer than two sources have
        coordinates.
    """
    placed = [s for s in sources if s.get("ra_deg") is not None and s.get("dec_deg") is not None]
    if len(placed) < 2:
        return ""
    vectors = [_unit_vector(s["ra_deg"], s["dec_deg"]) for s in placed]
    mx, my, mz = (sum(v[i] for v in vectors) for i in range(3))
    norm = math.sqrt(mx * mx + my * my + mz * mz)
    if norm < 1e-9:
        ra0, dec0 = placed[0]["ra_deg"], placed[0]["dec_deg"]
    else:
        ra0, dec0 = math.degrees(math.atan2(my, mx)) % 360.0, math.degrees(math.asin(mz / norm))
    offsets = [angular_offset(ra0, dec0, s["ra_deg"], s["dec_deg"]) for s in placed]
    max_rho = max(rho for rho, _ in offsets)
    step = next((st for st in SKY_MAP_RING_STEPS_DEG if math.ceil(max_rho / st) <= SKY_MAP_MAX_RINGS), 90)
    n_rings = max(1, math.ceil(max_rho / step - 1e-9))
    width, height, radius = 440, 360, 140
    cx, cy = width / 2, height / 2
    scale = radius / (n_rings * step)
    points = [(cx - rho * scale * math.sin(math.radians(pa)), cy - rho * scale * math.cos(math.radians(pa)))
              for rho, pa in offsets]

    parts = [f'<line class="axis" x1="{cx - radius}" y1="{cy}" x2="{cx + radius}" y2="{cy}"/>',
             f'<line class="axis" x1="{cx}" y1="{cy - radius}" x2="{cx}" y2="{cy + radius}"/>']
    for k in range(1, n_rings + 1):
        r = k * radius / n_rings
        parts.append(f'<circle class="ring" cx="{cx}" cy="{cy}" r="{r:.1f}"/>')
        parts.append(f'<text class="ringlabel" x="{cx + r * 0.707 + 3:.1f}" y="{cy - r * 0.707 - 3:.1f}">'
                     f'{escape(_format_ring(k * step))}</text>')
    parts.append(f'<text class="compass" x="{cx}" y="{cy - radius - 8}" text-anchor="middle">N</text>')
    parts.append(f'<text class="compass" x="{cx - radius - 8}" y="{cy + 4}" text-anchor="end">E</text>')

    calibrators = [i for i, s in enumerate(placed) if s.get("type") == "phasecal"]
    for i, source in enumerate(placed):
        if source.get("type") != "target" or not calibrators:
            continue
        sep, j = min((angular_offset(source["ra_deg"], source["dec_deg"], placed[j]["ra_deg"], placed[j]["dec_deg"])[0], j)
                     for j in calibrators)
        (x1, y1), (x2, y2) = points[i], points[j]
        parts.append(f'<line class="pair" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        parts.append(f'<text class="seplabel" x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 4:.1f}" '
                     f'text-anchor="middle">{escape(_format_separation(sep))}</text>')
    labels = _sky_map_labels([s["name"] for s in placed], points, cx + radius * 0.4)
    for source, (x, y), (lx, ly, anchor) in zip(placed, points, labels):
        role = escape(str(source.get("type") or "other"), quote=True)
        tip = f'{source["name"]} ({source.get("ra", "")}, {source.get("dec", "")})'
        if abs(ly - (y + 4)) > 1:
            parts.append(f'<line class="leader" x1="{x:.1f}" y1="{y:.1f}" x2="{lx + (2 if anchor == "end" else -2):.1f}" '
                         f'y2="{ly - 4:.1f}"/>')
        parts.append(f'<circle class="src {role}" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>{escape(tip)}</title></circle>')
        parts.append(f'<text class="srclabel" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">'
                     f'{escape(source["name"])}</text>')
    caption = (f"Source positions about their mean direction (RA {ra0 / 15:.2f} h, Dec {dec0:+.1f}\u00b0) in an "
               "azimuthal equidistant projection: distances from the centre and directions are true. North up, east "
               "left; dashed lines join each target to its nearest phase calibrator.")
    return (f'<figure class="skymap"><svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Sky positions of the observed sources">{"".join(parts)}</svg>'
            f'<figcaption><div class="d">{escape(caption)}</div></figcaption></figure>')


def _panel_ids(prefix: str, names: list[str]) -> list[str]:
    """Unique HTML ids ``<prefix>-<slug>`` for sub-tab panels; a repeated slug is numbered (``-2``, ``-3``...)."""
    ids, seen = [], {}
    for name in names:
        base = f"{prefix}-{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-') or 'panel'}"
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return ids


def _subtabs(tabs: list[tuple[str, str, str]]) -> str:
    """Horizontal sub-tabs inside a page.

    Parameters
    ----------
    tabs : list of (str, str, str)
        ``(panel id, button label as HTML, panel body as HTML)`` in display order; the ids must be unique
        within the page (:func:`_panel_ids`). The first tab starts active.

    Returns
    -------
    str
        The ``.subtabs-wrap`` block ``dashboard.js`` switches. Lazy images never load inside a hidden
        panel, so only the active panel keeps ``loading="lazy"``.
    """
    buttons, panels = [], []
    for index, (panel_id, label, body) in enumerate(tabs):
        pid = escape(panel_id, quote=True)
        if index != 0:
            body = body.replace(' loading="lazy"', "")
        buttons.append(f'<button class="{"active" if index == 0 else ""}" data-panel="{pid}">{label}</button>')
        panels.append(f'<div class="subpanel{" active" if index == 0 else ""}" id="{pid}">{body}</div>')
    return f'<div class="subtabs-wrap"><div class="subtabs">{"".join(buttons)}</div>{"".join(panels)}</div>'


def _page(ctx: dict, page: str, title: str, sections: list[tuple[str, str]], body: str) -> str:
    """Full HTML document for one tab; ``sections`` are (anchor, label) for the side navigation."""
    tabs = "".join(f'<a href="{stem}.html"{" class=\"active\"" if stem == page else ""}>{escape(name)}</a>'
                   for stem, name in PAGES)
    nav_links = "".join(f'<a href="#{a}">{escape(label)}</a>' for a, label in sections)
    nav = f'<nav class="nav">{nav_links}</nav>' if nav_links else ""
    code = escape(ctx["project"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{code} · {escape(title)} · vlbipy</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap">
<aside class="side">
  <div class="brand">vlbipy · <span class="code">{code}</span><small>Calibration dashboard</small></div>
  <nav class="tabs">{tabs}</nav>
  {nav}
  <button class="themetoggle" id="themetoggle">Toggle light / dark</button>
</aside>
<main class="main">
  <p class="eyebrow">{escape(ctx["observatory"])} · {code}</p>
  <h1>{escape(title)}</h1>
  <p class="sub">Generated {escape(ctx["generated_utc"])}. Click any plot to enlarge.</p>
  {body}
</main>
</div>
<div class="lb" id="lb"><img id="lbimg" alt=""><div class="cap" id="lbcap"></div></div>
<script src="dashboard.js"></script>
</body>
</html>
"""


def _comment_item(comment: dict) -> str:
    """One general comment as a list item: severity chip, sentence and, for a pipeline warning, its first-seen time."""
    severity = comment.get("severity") if comment.get("severity") in COMMENT_CHIP_CLASSES else "note"
    when = f'<span class="when">first seen {escape(str(comment["when"]))} UTC</span>' if comment.get("when") else ""
    return (f'<li class="cmt-{severity}"><span class="chip {COMMENT_CHIP_CLASSES[severity]}">{severity}</span>'
            f'<span>{escape(str(comment.get("text", "")))}{when}</span></li>')


def _comment_count(comments: Optional[list]) -> str:
    """Tally of the comments per severity for the section heading, e.g. ``"1 problem(s) · 4 warning(s)"``.

    The closing line of the antenna group ("Nothing to report for ...") is not a finding and is left out.
    """
    findings = [c for c in comments or [] if c.get("category") != "antenna" or c.get("antenna")]
    tally = [(sum(1 for c in findings if c.get("severity") == severity), severity) for severity in COMMENT_CHIP_CLASSES]
    return " · ".join(f"{count} {severity}(s)" for count, severity in tally if count)


def _comments_section(ctx: dict) -> str:
    """Body of the "General comments" section of the Overview page.

    Renders ``ctx["comments"]`` (:func:`vlbipy.comments.build_comments`) grouped as the plain-text block of
    ``summary.md`` is: data and calibration, one box per source headed by its name and role, one block per
    antenna with something to say (closed by the line naming the antennas with nothing to report), then the
    warnings the pipeline raised. A context
    saved before the comments existed has no such key and gets a notice instead.
    """
    comments = ctx.get("comments")
    if comments is None:
        return ('<p class="empty">This report was collected before the general comments existed; '
                'rebuild it with <code>vlbipy report</code>.</p>')
    groups = group_comments(comments)
    if not any(groups.values()):
        return '<p class="empty">Nothing to report.</p>'
    parts = []
    if groups["general"]:
        items = "".join(_comment_item(comment) for comment in groups["general"])
        parts.append(f'<h3>Data and calibration</h3><ul class="comments">{items}</ul>')
    if groups["sources"]:
        boxes = []
        for name, role, source_comments in groups["sources"]:
            items = "".join(_comment_item(comment) for comment in source_comments)
            boxes.append(f'<div class="cmt-src"><div class="cmt-src-head"><b>{escape(name)}</b> '
                         f'<span class="badge {escape(role, quote=True)}">{escape(role)}</span></div>'
                         f'<ul class="comments">{items}</ul></div>')
        parts.append(f'<h3>Sources</h3><div class="cmt-srcs">{"".join(boxes)}</div>')
    if groups["antennas"] or groups["antenna_notes"]:
        parts.append("<h3>Antennas</h3>")
        for name, antenna_comments in groups["antennas"]:
            items = "".join(_comment_item(comment) for comment in antenna_comments)
            parts.append(f'<div class="cmt-ant"><div class="cmt-name mono">{escape(name)}</div>'
                         f'<ul class="comments">{items}</ul></div>')
        for comment in groups["antenna_notes"]:
            parts.append(f'<p class="sub">{escape(str(comment.get("text", "")))}</p>')
    if groups["pipeline"]:
        items = "".join(_comment_item(comment) for comment in groups["pipeline"])
        parts.append(f'<h3>Warnings raised by the pipeline</h3><ul class="comments">{items}</ul>')
    return "".join(parts)


def render_overview(ctx: dict, html_dir: Path) -> str:
    """Overview tab: observation facts, general comments, antennas, sources and scan table."""
    obs = ctx["observation"]
    facts = [("Observing date", obs["date"]), ("Start (UTC)", obs["start_utc"]), ("End (UTC)", obs["end_utc"]),
             ("Duration", f'{obs["duration_hours"]} h'), ("Reference frequency", f'{obs["ref_freq_ghz"]} GHz'),
             ("Total bandwidth", f'{obs["bandwidth_mhz"]} MHz'), ("Subbands", str(obs["n_subbands"])),
             ("Channels per subband", str(obs["n_channels"])), ("Polarizations", ", ".join(obs["polarizations"]))]
    fact_html = "".join(f'<div class="fact"><div class="k">{escape(k)}</div><div class="v mono">{escape(v)}</div></div>'
                        for k, v in facts if v not in ("", "None GHz", "None MHz"))
    n_sb = max(obs["n_subbands"], 1)
    observed_count = no_data_count = no_signal_count = 0
    ant_rows = []
    for ant in ctx["antennas"]:
        segs = "".join(f'<i class="{"on" if i in ant["subbands"] else ""}" title="subband {i}"></i>' for i in range(n_sb))
        frac = ant["flagged_fraction"]
        if not ant["observed"]:
            no_data_count += 1
            status = '<span class="chip crit">no data</span>'
            health = '<span class="chip crit">no data</span>'
        elif frac is not None and frac >= 1.0:
            no_signal_count += 1
            status = '<span class="chip warn">no signal</span>'
            health = '<span class="chip warn">no signal</span>'
        else:
            observed_count += 1
            status = '<span class="chip good">observed</span>'
            if frac is None:
                health = '<span class="chip mut">not measured</span>'
            else:
                kept = max(0.0, 100 * (1 - frac))
                cls = "crit" if frac >= 0.95 else ("warn" if frac >= 0.7 else "good")
                health = (f'<span class="hbar"><i class="{cls}" style="width:{kept:.0f}%"></i></span> '
                          f'<span class="count">{kept:.0f}% kept / {100 * frac:.1f}% flagged</span>')
        ant_rows.append(f'<tr><td class="num">{ant["id"]}</td>'
                        f'<td class="mono"><b>{escape(ant["name"])}</b></td>'
                        f'<td><span class="sbbar">{segs}</span> <span class="count">{len(ant["subbands"])}/{n_sb}</span></td>'
                        f'<td class="num">{ant["n_scans"]}</td>'
                        f'<td>{status}</td><td>{health}</td></tr>')
    antenna_table = _table(["ID", "Antenna", "Subbands observed / total bandwidth", "Scans", "Status", "Data health"], ant_rows)

    frac = ctx["flagging"]["fraction"]
    excluding = ctx["flagging"].get("excluding_dead")
    flagging_parts = []
    if frac is not None:
        flagging_parts.append(f"{frac * 100:.1f}% of recorded cross-correlation visibilities is flagged overall")
    if excluding and excluding.get("fraction") is not None:
        names = ", ".join(excluding.get("excluded_antennas", [])) or "none"
        flagging_parts.append(
            f"{excluding['fraction'] * 100:.1f}% is flagged when antennas with no usable signal are excluded "
            f"(excluded: {names})")
    elif frac is not None and excluding and not excluding.get("available", False):
        flagging_parts.append("the exact baseline-weighted fraction excluding no-signal antennas is unavailable in this older cache")
    flagging_summary = ("; ".join(flagging_parts) + ".") if flagging_parts else "No flagging statistics recorded yet."
    summary = (f"{observed_count} antenna{'s' if observed_count != 1 else ''} observed, "
               f"{no_data_count} with no data, and {no_signal_count} observed but with no usable signal "
               f"(completely flagged). {flagging_summary}")

    src_rows = [f'<tr><td><span class="badge {escape(s["type"])}">{escape(s["type"])}</span></td>'
                f'<td><b>{escape(s["name"])}</b></td><td class="mono">{escape(str(s["id"]))}</td>'
                f'<td class="mono">{escape(s["ra"])}</td><td class="mono">{escape(s["dec"])}</td>'
                f'<td class="num">{s["n_scans"]}</td></tr>' for s in ctx["sources"]]
    source_table = _table(["Type", "Source", "ID", "RA (J2000)", "Dec (J2000)", "Scans"], src_rows)

    scan_rows = []
    for scan in ctx["scans"]:
        snrs = scan.get("antenna_snrs", {})
        states = scan.get("antenna_states", {})
        role = scan.get("role", "other")
        present = set(scan.get("antennas", []))
        chips = []
        for ant in ctx["antennas"]:
            name = ant["name"]
            snr = snrs.get(name)
            # Old dashboard.json snapshots have no states: fall back to value-or-unsurveyed.
            state = states.get(name) or ("snr" if snr is not None else
                                         ("absent" if name not in present else "unsurveyed"))
            if state == "snr":
                cls, tip, content = snr_class(snr), f"SNR {snr:.1f}", escape(name)
            else:
                cls, tip = SCAN_CHIP_STATES.get(state, SCAN_CHIP_STATES["unsurveyed"])
                content = "&nbsp;" if state == "absent" else escape(name)
                if state == "refant":
                    content += '<i class="refmark">ref</i>'
            chips.append(f'<span class="chip scan-chip {cls} mono" title="{escape(f"{name}: {tip}")}">{content}</span>')
        chips_html = "".join(chips)
        scan_rows.append(f'<tr><td class="num">{scan["number"]}</td>'
                         f'<td><span class="badge {escape(role)}">{escape(role)}</span> <b>{escape(scan["source"])}</b></td>'
                         f'<td class="mono">{escape(scan["start_utc"])}</td><td class="num">{scan["duration_sec"]} s</td>'
                         f'<td style="white-space:normal">{chips_html}</td></tr>')
    legend = ('<div class="legend">Fringe SNR per antenna: <span class="chip snr-bad">&lt; 3</span>'
              '<span class="chip snr-low">3 - 6</span><span class="chip snr-mid">6 - 10</span>'
              '<span class="chip snr-good">&ge; 10</span> · no SNR because: '
              '<span class="chip snr-ref">reference antenna</span>'
              '<span class="chip snr-nodata">no data (fully flagged)</span>'
              '<span class="chip snr-nofringe">no fringe solution</span>'
              '<span class="chip snr-none">scan not surveyed</span></div>')
    scan_table = legend + _table(["Scan", "Source", "Start (UTC)", "Duration", "Antennas"], scan_rows)

    body = (_section("observation", "Observation", f'<div class="facts">{fact_html}</div>',
                     "Setup of the observation and the calibration run.")
            + _section("comments", "General comments", _comments_section(ctx),
                       "Issues, decisions and results of this reduction, derived from the calibration products "
                       "on disk. The same list opens summary.md.", count=_comment_count(ctx.get("comments")))
            + _section("antennas", "Antennas", antenna_table, summary)
            + _section("sources", "Sources", _sky_map(ctx["sources"]) + source_table)
            + _section("scans", "Scans", scan_table))
    sections = [("observation", "Observation"), ("comments", "General comments"), ("antennas", "Antennas"),
                ("sources", "Sources"), ("scans", "Scans")]
    return _page(ctx, "index", "Overview", sections, body)


def _time_evolution_panel(paths: list[Path], html_dir: Path, project_code: str, refants: list[str]) -> str:
    """Time-evolution figures grouped by reference antenna, then ordered by subband.

    Parameters
    ----------
    paths : list of Path
        Files matching :data:`TIME_EVOLUTION_PATTERNS`.
    html_dir : Path
        Directory of the page, for the relative image links.
    project_code : str
        Project code, stripped from the title of a file whose name does not follow the convention.
    refants : list of str
        Reference antenna(s) of the observation; their groups come first, in this order, then the
        other reference antennas alphabetically.

    Returns
    -------
    str
        One heading and figure grid per reference antenna (a trailing "Other" group for names that do
        not parse), or the "empty" notice when there are no files.
    """
    if not paths:
        return '<p class="empty">No plots available for this section.</p>'
    caption = "Raw amplitude and phase vs time per baseline to the reference antenna, coloured by source."
    groups: dict[str, list[tuple[int, Path]]] = {}
    other: list[Path] = []
    for path in paths:
        key = time_evolution_key(path)
        if key is None:
            other.append(path)
        else:
            groups.setdefault(key[0], []).append((key[1], path))
    order = sorted(groups, key=lambda name: (refants.index(name) if name in refants else len(refants), name))
    out = []
    for name in order:
        figures = "".join(_figure(path, html_dir, f"Baselines to {name} · subband {subband}", caption)
                          for subband, path in sorted(groups[name], key=lambda item: (item[0], str(item[1]))))
        out.append(f'<h3>Reference antenna {escape(name)}</h3><div class="grid wide">{figures}</div>')
    if other:
        out.append("<h3>Other</h3>" + _plot_grid(other, html_dir, project_code, caption, wide=True))
    return "".join(out)


def render_diagnostics(ctx: dict, html_dir: Path, plot_root: Path) -> str:
    """Diagnostics tab: raw auto-correlations, cross-correlations and time evolution, one sub-tab each."""
    code = ctx["project"]
    auto = find_plots(plot_root, AUTOCORR_PATTERNS)
    cross = find_plots(plot_root, CROSSCORR_PATTERNS)
    evolution = find_plots(plot_root, TIME_EVOLUTION_PATTERNS)
    scan_source = {s["number"]: s["source"] for s in ctx["scans"]}

    def per_scan(paths: list[Path], caption: str) -> str:
        if not paths:
            return '<p class="empty">No plots available for this section.</p>'
        groups: dict[int, list[Path]] = {}
        for p in paths:
            groups.setdefault(scan_number_from_name(p) or 0, []).append(p)
        out = []
        for number in sorted(groups):
            label = f"Scan {number} · {scan_source.get(number, '')}".rstrip(" ·") if number else "Unspecified scan"
            out.append(f"<h3>{escape(label)}</h3>" + _plot_grid(groups[number], html_dir, code, caption, wide=True))
        return "".join(out)

    refants = [name for name in str(ctx.get("refant") or "").split(",") if name]
    tabs = [
        ("diag-autocorr", "Auto-correlations",
         _section("autocorr", "Auto-correlations", per_scan(auto, "Amplitude vs frequency per antenna (raw data)."),
                  "Auto-correlation spectra per antenna on the fringe-finder scans (phase calibrator when no fringe "
                  "finder was observed).")),
        ("diag-crosscorr", "Cross-correlations",
         _section("crosscorr", "Cross-correlations to the reference antenna",
                  per_scan(cross, "Amplitude and phase vs frequency per baseline to the reference antenna (raw data)."),
                  "Raw cross-correlation spectra per baseline to the reference antenna of each scan.")),
        ("diag-time-evolution", "Time evolution",
         _section("time-evolution", "Time evolution", _time_evolution_panel(evolution, html_dir, code, refants),
                  "Raw amplitude and phase versus time over the whole observation for every baseline to a reference "
                  "antenna, coloured by source and vector-averaged over the central channels of one subband. A further "
                  "reference antenna is added when the first one did not observe the whole run or had problems during "
                  "part of it, and a further subband when not every antenna recorded the first one.")),
    ]
    # The sections live in sub-tabs, so there are no in-page anchors for the side navigation.
    return _page(ctx, "diagnostics", "Diagnostics", [], _subtabs(tabs))


def _chain_table(caltables: list[dict]) -> str:
    """Calibration chain table (one row per calibration table applied)."""
    if not caltables:
        return '<p class="empty">No calibration tables recorded yet.</p>'
    rows = []
    for t in caltables:
        snr = t.get("snr") or 0
        chip = ('<span class="chip mut">-</span>' if not snr else
                f'<span class="chip {"good" if snr >= 100 else ("warn" if snr >= 10 else "crit")} mono">{snr:,.0f}</span>')
        rows.append(f'<tr><td class="mono"><b>{escape(str(t.get("cal_type", "")))}</b></td><td>{escape(str(t.get("field", "")))}</td>'
                    f'<td class="mono">{escape(str(t.get("interp", "")))}</td><td>{chip}</td>'
                    f'<td class="mono">{escape(str(t.get("step", "")))}</td><td class="mono">{escape(Path(str(t.get("path", ""))).name)}</td></tr>')
    return _table(["Table", "Field", "Interp", "Median SNR", "Step", "File"], rows)


def _gain_cell(name: str, column: str, gain: Optional[float], extra_class: str = "") -> str:
    """One ``<td>`` of the amplitude-correction table, coloured by :func:`gain_class`.

    Parameters
    ----------
    name : str
        Antenna code (tooltip only).
    column : str
        What the column is, for the tooltip (subband label and frequency, or the median).
    gain : float or None
        Amplitude factor; None renders a neutral dash ("no solution").
    extra_class : str
        Additional css class (``gain-median`` for the summary column).
    """
    value = _finite_or_none(gain)
    classes = f"gain {gain_class(value)}{' ' + extra_class if extra_class else ''}"
    if value is None:
        return f'<td class="{classes}" title="{escape(f"{name} · {column}: no solution")}">&ndash;</td>'
    tip = f"{name} · {column}: x{value:.4f} ({(value - 1.0) * 100:+.1f}%)"
    return f'<td class="{classes}" title="{escape(tip)}">{value:.{GAIN_DECIMALS}f}</td>'


def _frequency_labels(freqs_hz: list) -> list[str]:
    """Column frequencies in MHz with the fewest decimals (0 to 2) that still tell the columns apart.

    ``freqs_hz`` holds one frequency in Hz per column (None when unknown, which gives an empty label).
    """
    values = [_finite_or_none(freq) for freq in freqs_hz]
    known = [value / 1e6 for value in values if value]
    decimals = next((d for d in (0, 1, 2) if len({f"{mhz:.{d}f}" for mhz in known}) == len(known)), 2)
    return [f"{value / 1e6:.{decimals}f} MHz" if value else "" for value in values]


def _gain_table(entry: dict) -> str:
    """Amplitude-correction table of one source: a row per antenna, a column per subband, then the median.

    ``entry`` is one item of ``ctx["gain_corrections"]`` (:func:`gain_correction_entry`). The coloured cells
    touch each other (no borders or row striping, see ``.gaintable`` in the style sheet); the antenna and
    median columns stay in view when a wide table scrolls sideways.
    """
    labels = [str(label) for label in entry.get("subband_labels") or []]
    freqs = (list(entry.get("if_freqs_hz") or []) + [None] * len(labels))[:len(labels)]
    columns, head = [], ['<th class="gain-ant">Antenna</th>']
    for label, freq, short in zip(labels, freqs, _frequency_labels(freqs)):
        freq = _finite_or_none(freq)
        exact = f"{freq / 1e6:.2f}".rstrip("0").rstrip(".") + " MHz" if freq else ""
        columns.append(f"{label} ({exact})" if exact else label)
        head.append(f'<th title="{escape(columns[-1])}">{escape(label)}'
                    f'{f"<small>{escape(short)}</small>" if short else ""}</th>')
    head.append('<th class="gain-median">Median</th>')
    rows = []
    for name in entry.get("antennas") or []:
        values = list((entry.get("gain_per_if") or {}).get(name) or [])[:len(labels)]
        values += [None] * (len(labels) - len(values))
        cells = "".join(_gain_cell(name, column, value) for column, value in zip(columns, values))
        n_solved = sum(1 for value in values if _finite_or_none(value) is not None)
        median = _gain_cell(name, f"median of {n_solved} subband(s)", (entry.get("gain_median") or {}).get(name),
                            "gain-median")
        rows.append(f'<tr><td class="gain-ant mono"><b>{escape(name)}</b></td>{cells}{median}</tr>')
    return (f'<div class="tablewrap gainwrap"><table class="gaintable"><thead><tr>{"".join(head)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _gain_legend() -> str:
    """Colour key of the amplitude-correction table, worded from :data:`GAIN_CELL_CLASSES`."""
    keys, lower = [], 0.0
    for bound, cls in GAIN_CELL_CLASSES:
        if lower == 0.0:
            text = f"within {bound:.0%}"
        elif math.isinf(bound):
            text = f"beyond {lower:.0%}"
        else:
            text = f"{lower:.0%} - {bound:.0%}"
        keys.append(f'<span class="gainkey {cls}">{text}</span>')
        lower = bound
    keys.append('<span class="gainkey gain-none">&ndash; no solution</span>')
    return f'<div class="legend">Correction relative to 1.00: {"".join(keys)}</div>'


def _gain_panel(entry: dict) -> str:
    """Table of one source, a warning when its corrections were not applied, and the file they come from."""
    warning = ('<p><span class="chip warn">computed but not applied to the data</span></p>'
               if entry.get("applied") is False else "")
    note = f'<p class="count">{escape(str(entry["file"]))}</p>' if entry.get("file") else ""
    return warning + _gain_table(entry) + note


def _gain_section(ctx: dict) -> str:
    """Body of the "Amplitude corrections" section: the colour key and one table per self-calibrated source.

    A single source renders as one titled table, several get a sub-tab each. A context without
    ``gain_corrections`` (an older ``dashboard.json``), or with none, renders the "empty" notice.
    """
    entries = [e for e in (ctx.get("gain_corrections") or []) if isinstance(e, dict) and e.get("antennas")]
    if not entries:
        return '<p class="empty">No amplitude corrections from the self-calibration available.</p>'
    labels = []
    for entry in entries:
        source, role = escape(str(entry.get("source", ""))), escape(str(entry.get("role") or "other"))
        labels.append(f'{source} <span class="badge {role}">{role}</span>')
    if len(entries) == 1:
        return _gain_legend() + f"<h3>{labels[0]}</h3>" + _gain_panel(entries[0])
    ids = _panel_ids("gains", [str(entry.get("source", "")) for entry in entries])
    return _gain_legend() + _subtabs([(pid, label, _gain_panel(entry))
                                      for pid, label, entry in zip(ids, labels, entries)])


def render_calibration(ctx: dict, html_dir: Path, plot_root: Path) -> str:
    """Calibration tab: chain table, amplitude corrections, calibration-table plot groups and corner plots."""
    code = ctx["project"]
    sections = [("chain", "Calibration chain"), ("amplitude-corrections", "Amplitude corrections")]
    body = (_section("chain", "Calibration chain", _chain_table(ctx["caltables"]),
                     "Tables solved and applied in order: tsys -> gc -> sbd -> mbd -> bpass, then re-solved.")
            + _section("amplitude-corrections", "Amplitude corrections", _gain_section(ctx),
                       "Amplitude factor the difmapy gscale self-calibration applied to the visibilities of every "
                       "antenna and subband (1.00 = unchanged). The last column is the median over the subbands "
                       "the antenna has a solution in."))
    for title, blurb, patterns in CALIBRATION_GROUPS:
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        paths = find_plots(plot_root, patterns)
        sections.append((anchor, title))
        body += _section(anchor, title, _plot_grid(paths, html_dir, code, wide=len(paths) <= 2), blurb)
    corner_paths = find_plots(plot_root, ["calibrated/*corner*.png"])
    by_source: dict[str, list[Path]] = {}
    names = sorted((s["name"] for s in ctx["sources"]), key=len, reverse=True)
    for p in corner_paths:
        source = next((n for n in names if f"_{n}." in p.name or f".{n}." in p.name), "all calibrators")
        by_source.setdefault(source, []).append(p)
    if by_source:
        ordered = sorted(by_source.items())
        caption = "Per-baseline time x frequency grid; structure left in a cell is a calibration residual."
        ids = _panel_ids("corner", [source for source, _ in ordered])
        corners = _subtabs([(pid, escape(source), _plot_grid(paths, html_dir, code, caption))
                            for pid, (source, paths) in zip(ids, ordered)])
    else:
        corners = '<p class="empty">No corner plots available.</p>'
    sections.append(("corners", "Calibrator corner plots"))
    body += _section("corners", "Calibrator phases and amplitudes", corners,
                     "Calibrated per-baseline phase and amplitude grids, one sub-tab per source.")
    return _page(ctx, "calibration", "Calibration", sections, body)


def plot_source(path: Path, names: list[str]) -> Optional[str]:
    """Source a plot file belongs to, or None when its name mentions none of ``names``.

    Parameters
    ----------
    path : Path
        Plot file; only its name is inspected.
    names : list of str
        Candidate source names, longest first so ``3C39`` never steals ``3C395``.

    Returns
    -------
    str or None
        Matching source name, None for a file that belongs to no single source.
    """
    return next((n for n in names if f"_{n}." in path.name or f".{n}." in path.name), None)


def _downloads_block(entries: list[dict], html_dir: Path) -> str:
    """Download list for one source's products ("" when nothing exists on disk)."""
    if not entries:
        return ""
    items = []
    for entry in entries:
        href = relative_href(Path(entry["path"]), html_dir)
        tag = ' <span class="chip mut">directory</span>' if entry.get("is_dir") else ""
        size = f' <span class="count">{escape(entry.get("size_human") or "")}</span>' if entry.get("size_human") else ""
        items.append(f'<li><a href="{escape(href)}">{escape(entry["label"])}</a>{tag}{size}'
                     f'<code class="mono dl-path">{escape(entry.get("relpath") or href)}</code></li>')
    return f'<ul class="downloads">{"".join(items)}</ul>'


def _final_source_panel(ctx: dict, html_dir: Path, slug: str, entries: list[dict],
                        groups: list[tuple[str, str, list[Path]]], images: list[Path]) -> str:
    """Body of one source sub-tab: downloads, the FINAL_GROUPS sections that have a plot, then the images.

    A group with nothing for this source is left out rather than shown as an
    empty box: not every final plot exists per source (the total light curve is
    one figure for the whole observation), and five "no plots available" notices
    in every tab would bury the ones that do.
    """
    code = ctx["project"]
    parts = []
    if entries:
        parts.append(_section(f"{slug}-downloads", "Downloads", _downloads_block(entries, html_dir),
                              "Calibrated data and images of this source as written to the project directory."))
    for title, blurb, paths in groups:
        if not paths:
            continue
        anchor = f'{slug}-{re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")}'
        parts.append(_section(anchor, title, _plot_grid(paths, html_dir, code, wide=len(paths) <= 2), blurb))
    if images:
        parts.append(_section(f"{slug}-images", "Images",
                              _plot_grid(images, html_dir, code, "Images at robust -2 (uniform), 0 and 2 (natural).", wide=True),
                              "Images with Briggs robust -2 (uniform), 0 and +2 (natural) weighting: the model (Gaussians and the CLEAN of their residual, or CLEAN alone) restored with each beam, plus the residuals."))
    return "".join(parts)


def render_final_data(ctx: dict, html_dir: Path, plot_root: Path, image_root: Path) -> str:
    """Final-results tab: one sub-tab per source with its plots, images and downloads.

    Plots are attributed to a source by name (longest source name first); the
    ones naming no source land in a trailing "All sources" sub-tab. A source with
    neither plots nor downloads is skipped.
    """
    names = sorted((s["name"] for s in ctx["sources"]), key=len, reverse=True)
    grouped: list[tuple[str, str, dict[Optional[str], list[Path]]]] = []
    for title, blurb, patterns, fallback in FINAL_GROUPS:
        by_source: dict[Optional[str], list[Path]] = {}
        for path in find_plots(plot_root, patterns) or find_plots(plot_root, fallback):
            by_source.setdefault(plot_source(path, names), []).append(path)
        grouped.append((title, blurb, by_source))
    images = find_plots(image_root, ["*.images.*.png"]) if image_root.is_dir() else []
    if not images and image_root.is_dir():
        images = find_plots(image_root, ["**/*.png"])
    images_by_source: dict[Optional[str], list[Path]] = {}
    for path in images:
        images_by_source.setdefault(plot_source(path, names), []).append(path)
    downloads = ctx.get("downloads") or {}

    tabs: list[tuple[Optional[str], str]] = []
    for source in ctx["sources"]:
        name = source["name"]
        has_plots = any(by_source.get(name) for _, _, by_source in grouped) or bool(images_by_source.get(name))
        if has_plots or downloads.get(name):
            role = escape(source["type"])
            tabs.append((name, f'{escape(name)} <span class="badge {role}">{role}</span>'))
    if any(by_source.get(None) for _, _, by_source in grouped) or images_by_source.get(None):
        tabs.append((None, "All sources"))
    if not tabs:
        body = _section("final", "Per-source products", '<p class="empty">No final data products available yet.</p>',
                        "Calibrated visibilities, light curves and images per source.")
        return _page(ctx, "final_data", "Final results", [], body)

    panels = []
    for panel_id, (source, label) in zip(_panel_ids("final", [source or "all-sources" for source, _ in tabs]), tabs):
        groups = [(title, blurb, by_source.get(source) or []) for title, blurb, by_source in grouped]
        panels.append((panel_id, label, _final_source_panel(ctx, html_dir, panel_id, downloads.get(source) or [],
                                                           groups, images_by_source.get(source) or [])))
    return _page(ctx, "final_data", "Final results", [], _subtabs(panels))


def render_pages(ctx: dict, work_dir: Path) -> Path:
    """Write every page plus assets into ``<work_dir>/html`` and return the overview path."""
    html_dir = Path(work_dir) / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    plot_root, image_root = Path(work_dir) / "plots", Path(work_dir) / "images"
    pages = {"index": render_overview(ctx, html_dir), "diagnostics": render_diagnostics(ctx, html_dir, plot_root),
             "calibration": render_calibration(ctx, html_dir, plot_root),
             "final_data": render_final_data(ctx, html_dir, plot_root, image_root)}
    for stem, html in pages.items():
        (html_dir / f"{stem}.html").write_text(html, encoding="utf-8")
    (html_dir / "style.css").write_text(CSS, encoding="utf-8")
    (html_dir / "dashboard.js").write_text(JS, encoding="utf-8")
    (html_dir / "dashboard.json").write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    logger.bind(headline=True).info("report: {}", html_dir / "index.html")
    logger.info("dashboard: wrote {} page(s) to {}", len(pages), html_dir)
    return html_dir / "index.html"


def refresh_summary(obs, ctx: dict) -> Optional[Path]:
    """Rewrite ``<work_dir>/summary.md`` with the general comments of ``ctx`` as its first section.

    The import step writes the summary before anything is calibrated; this brings it up to date once
    there is something to say, with the very list the Overview page shows.

    Parameters
    ----------
    obs : VLBIObs or Observation
        Observation the context was collected from (its metadata and sources fill the other sections).
    ctx : dict
        Context of :func:`collect`.

    Returns
    -------
    pathlib.Path or None
        The file written; None when there is no metadata to summarise or the write failed (logged, never
        raised: the summary is a report, it must not cost the dashboard).
    """
    observation = _primary_observation(obs)
    metadata = observation.metadata
    if metadata is None:
        return None
    try:
        from .diagnostics import write_summary
        return write_summary(metadata, Path(ctx["work_dir"]) / "summary.md", observation.sources,
                             comments=ctx.get("comments") or [])
    except Exception as exc:  # noqa: BLE001 - reporting only
        logger.warning("dashboard: could not refresh summary.md ({})", exc)
        return None


def build_dashboard(obs, *, flag_statistics: Optional[dict] = None) -> Path:
    """Collect the observation state and render the multi-page dashboard; returns ``html/index.html``.

    Also rewrites ``<work_dir>/summary.md`` (:func:`refresh_summary`) so that its "General Comments" block
    is the list the Overview page was rendered from. This is the one path behind ``Observation.report()``,
    ``vlbipy report`` and :func:`serve_dashboard`.
    """
    ctx = collect(obs, flag_statistics=flag_statistics)
    index = render_pages(ctx, Path(ctx["work_dir"]))
    refresh_summary(obs, ctx)
    return index


def serve_dashboard(obs, *args, **kwargs) -> Path:
    """Build the static dashboard and open it in the default browser."""
    import webbrowser
    index = build_dashboard(obs)
    webbrowser.open(index.as_uri())
    return index
