"""Static HTML dashboard of a vlbipy reduction.

``build_dashboard(obs)`` writes ``<work_dir>/html/`` with one page per tab
(``index.html`` = overview, ``diagnostics.html``, ``calibration.html``,
``final_data.html``), a shared ``style.css``/``dashboard.js`` and a
``dashboard.json`` snapshot of everything the pages were rendered from. Plots are
referenced relatively (``../plots/...``, ``../images/...``), so the ``html/``
folder must travel together with the rest of the project directory.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from typing import Optional

from astropy.time import Time
from loguru import logger

#: (file stem, tab title) in display order. ``index`` is the overview.
PAGES: list[tuple[str, str]] = [("index", "Overview"), ("diagnostics", "Diagnostics"),
                                ("calibration", "Calibration"), ("final_data", "Final data")]

#: Fringe-SNR thresholds for the per-antenna chips in the scan table: (upper bound, css class).
SNR_CHIP_CLASSES: list[tuple[float, str]] = [(3.0, "snr-bad"), (6.0, "snr-low"), (10.0, "snr-mid"), (math.inf, "snr-good")]

#: Calibration-table plot groups: (title, blurb, glob patterns relative to <work_dir>/plots).
CALIBRATION_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Gain curves", "Elevation-dependent gain per antenna from the ANTAB/GC information.", ["caltables/*gcal*.png"]),
    ("System temperature", "Tsys vs time per antenna (one colour per polarization).", ["caltables/*tsys*.png"]),
    ("Bandpass", "Amplitude and phase vs frequency per antenna.", ["caltables/*.bpass*.png"]),
    ("SBD selected-scan data", "Amplitude and phase versus time for each selected SBD stage, one baseline to that stage's reference antenna per panel.",
     ["calibrated/*.sbd*_scan*.timeseries.png", "raw/*.sbd*_scan*.timeseries.png"]),
    ("Single-band delay (SBD)", "Instrumental phase, delay and rate solutions versus frequency/subband.", ["caltables/*.sbd.*.png", "caltables/*.sbd2.*.png"]),
    ("Multi-band delay (MBD)", "Global fringe-fit delay/phase/rate solutions.", ["caltables/*.mbd.*.png", "caltables/*.mbd2.*.png"]),
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


# -- data collection ---------------------------------------------------------------------------

def _primary_observation(obs):
    """Return the single :class:`Observation` behind ``obs`` (a ``VLBIObs`` or an ``Observation``)."""
    if not hasattr(obs, "observations"):
        return obs
    primary = obs._primary
    return primary() if callable(primary) else primary


def _source_rows(observation, metadata) -> list[dict]:
    """Source table rows: type, name, field id, coordinates and scan count."""
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
                     "n_scans": sum(1 for s in metadata.scans if s.source == name) if metadata else 0})
    return rows


def _scan_rows(metadata, source_roles: Optional[dict[str, str]] = None) -> list[dict]:
    """Scan table rows with per-antenna fringe SNR (None when not surveyed).

    Each row carries a slot for every antenna present in the metadata so the
    rendered scan chips line up vertically even when an antenna is absent from a
    particular scan.
    """
    survey = metadata.snr_survey if metadata else None
    per_scan = survey.per_scan_antenna() if survey is not None and hasattr(survey, "per_scan_antenna") else {}
    all_antennas = list(metadata.antennas) if metadata else []
    roles = source_roles or {}
    rows = []
    for scan in (metadata.scans if metadata else []):
        snrs = per_scan.get(scan.scan_number, {})
        rows.append({"number": scan.scan_number, "source": scan.source,
                     "role": roles.get(scan.source, "other"),
                     "start_utc": mjd_seconds_to_utc(scan.time_start), "duration_sec": round(scan.duration_sec),
                     "antennas": list(scan.antennas),
                     "antenna_snrs": {name: _finite_or_none(snrs.get(name)) for name in all_antennas}})
    return rows


def _finite_or_none(value) -> Optional[float]:
    """Return ``value`` as float when finite, otherwise None."""
    try:
        return float(value) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def collect(obs, *, flag_statistics: Optional[dict] = None) -> dict:
    """Gather everything the dashboard renders into one JSON-serialisable dict.

    Parameters
    ----------
    obs : VLBIObs or Observation
        Observation whose metadata, sources, calibration chain and flag statistics are read.
    flag_statistics : dict, optional
        Result of ``obs.flag.statistics()``; defaults to the last one stored on the observation.
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
    return {
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
        "scans": _scan_rows(metadata, source_roles),
        "caltables": [t.to_dict() for t in observation.gaintables],
        "work_dir": str(observation.work_dir),
    }


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
.chip.scan-chip{min-width:34px;height:20px;text-align:center;display:inline-flex;align-items:center;justify-content:center;vertical-align:middle}
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
.count{color:var(--muted);font-size:12.5px;font-weight:600;font-family:ui-monospace,Menlo,monospace}
.empty{color:var(--muted);font-style:italic;padding:8px 0}
.subtabs-wrap{margin:12px 0}
.subtabs-wrap .subtabs{margin:0}
.subtabs{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}
.subtabs button{border:1px solid var(--border);background:var(--panel);color:var(--ink);border-radius:8px;
  padding:6px 12px;font-size:13px;cursor:pointer}
.subtabs button.active{background:var(--accent);border-color:var(--accent);color:#fff}
.subpanel{display:none} .subpanel.active{display:block}
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


def _page(ctx: dict, page: str, title: str, sections: list[tuple[str, str]], body: str) -> str:
    """Full HTML document for one tab; ``sections`` are (anchor, label) for the side navigation."""
    tabs = "".join(f'<a href="{stem}.html"{" class=\"active\"" if stem == page else ""}>{escape(name)}</a>'
                   for stem, name in PAGES)
    nav = "".join(f'<a href="#{a}">{escape(label)}</a>' for a, label in sections)
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
  <nav class="nav">{nav}</nav>
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


def render_overview(ctx: dict, html_dir: Path) -> str:
    """Overview tab: observation facts, antennas, sources and scan table."""
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
    target_detected = any(
        scan.get("role") == "target"
        and any(s is not None and not math.isnan(s) and s >= 3.0
                for s in scan.get("antenna_snrs", {}).values())
        for scan in ctx["scans"])
    for scan in ctx["scans"]:
        snrs = scan.get("antenna_snrs", {})
        role = scan.get("role", "other")
        present = set(scan.get("antennas", []))
        chips = []
        for ant in ctx["antennas"]:
            name = ant["name"]
            snr = snrs.get(name)
            if (name in present and snr is not None and not math.isnan(snr)
                    and (role != "target" or target_detected)):
                cls = snr_class(snr)
            else:
                cls = "snr-none"
            title = f"{name}: SNR {'n/a' if snr is None else f'{snr:.1f}'}"
            content = escape(name) if name in present else "&nbsp;"
            chips.append(f'<span class="chip scan-chip {cls} mono" title="{title}">{content}</span>')
        chips_html = "".join(chips)
        scan_rows.append(f'<tr><td class="num">{scan["number"]}</td>'
                         f'<td><span class="badge {escape(role)}">{escape(role)}</span> <b>{escape(scan["source"])}</b></td>'
                         f'<td class="mono">{escape(scan["start_utc"])}</td><td class="num">{scan["duration_sec"]} s</td>'
                         f'<td style="white-space:normal">{chips_html}</td></tr>')
    legend = ('<div class="legend">Fringe SNR per antenna: <span class="chip snr-bad">&lt; 3</span>'
              '<span class="chip snr-low">3 - 6</span><span class="chip snr-mid">6 - 10</span>'
              '<span class="chip snr-good">&ge; 10</span><span class="chip snr-none">not surveyed</span></div>')
    scan_table = legend + _table(["Scan", "Source", "Start (UTC)", "Duration", "Antennas"], scan_rows)

    body = (_section("observation", "Observation", f'<div class="facts">{fact_html}</div>',
                     "Setup of the observation and the calibration run.")
            + _section("antennas", "Antennas", antenna_table, summary)
            + _section("sources", "Sources", source_table, count=f'{len(ctx["sources"])}')
            + _section("scans", "Scans", scan_table, count=f'{len(ctx["scans"])}'))
    sections = [("observation", "Observation"), ("antennas", "Antennas"),
                ("sources", "Sources"), ("scans", "Scans")]
    return _page(ctx, "index", "Overview", sections, body)


def render_diagnostics(ctx: dict, html_dir: Path, plot_root: Path) -> str:
    """Diagnostics tab: raw auto- and cross-correlation spectra per diagnostic scan."""
    code = ctx["project"]
    auto = find_plots(plot_root, ["raw/*_scan*.autocorr.png"])
    cross = find_plots(plot_root, ["raw/*_scan*.spectrum.png"])
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

    body = (_section("autocorr", "Auto-correlations", per_scan(auto, "Amplitude vs frequency per antenna (raw data)."),
                     "Auto-correlation spectra per antenna on the fringe-finder scans (phase calibrator when no fringe finder was observed).",
                     count=f"{len(auto)} plot(s)")
            + _section("crosscorr", "Cross-correlations to the reference antenna",
                       per_scan(cross, "Amplitude and phase vs frequency per baseline to the reference antenna (raw data)."),
                       "Raw cross-correlation spectra per baseline to the reference antenna of each scan.",
                       count=f"{len(cross)} plot(s)"))
    return _page(ctx, "diagnostics", "Diagnostics", [("autocorr", "Auto-correlations"), ("crosscorr", "Cross-correlations")], body)


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


def render_calibration(ctx: dict, html_dir: Path, plot_root: Path) -> str:
    """Calibration tab: chain table, every calibration-table plot group and per-source corner plots."""
    code = ctx["project"]
    sections = [("chain", "Calibration chain")]
    body = _section("chain", "Calibration chain", _chain_table(ctx["caltables"]),
                    "Tables solved and applied in order: tsys -> gc -> sbd -> mbd -> bpass, then re-solved.")
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
        buttons, panels, seen = [], [], {}
        for i, (source, paths) in enumerate(sorted(by_source.items())):
            base = f"corner-{re.sub(r'[^A-Za-z0-9]+', '-', source).strip('-') or 'panel'}"
            seen[base] = seen.get(base, 0) + 1
            pid = base if seen[base] == 1 else f"{base}-{seen[base]}"
            grid = _plot_grid(paths, html_dir, code,
                              "Per-baseline time x frequency grid; structure left in a cell is a calibration residual.")
            if i != 0:
                grid = grid.replace(' loading="lazy"', "")
            buttons.append(f'<button class="{"active" if i == 0 else ""}" data-panel="{escape(pid, quote=True)}">{escape(source)}</button>')
            panels.append(f'<div class="subpanel{" active" if i == 0 else ""}" id="{escape(pid, quote=True)}">{grid}</div>')
        corners = f'<div class="subtabs-wrap"><div class="subtabs">{"".join(buttons)}</div>{"".join(panels)}</div>'
    else:
        corners = '<p class="empty">No corner plots available.</p>'
    sections.append(("corners", "Calibrator corner plots"))
    body += _section("corners", "Calibrator phases and amplitudes", corners,
                     "Calibrated per-baseline phase and amplitude grids, one sub-tab per source.")
    return _page(ctx, "calibration", "Calibration", sections, body)


def render_final_data(ctx: dict, html_dir: Path, plot_root: Path, image_root: Path) -> str:
    """Final-data tab: uv coverage, radplots, spectra, light curves and images per source."""
    code = ctx["project"]
    sections, body = [], ""
    for title, blurb, patterns, fallback in FINAL_GROUPS:
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        paths = find_plots(plot_root, patterns) or find_plots(plot_root, fallback)
        sections.append((anchor, title))
        body += _section(anchor, title, _plot_grid(paths, html_dir, code, wide=len(paths) <= 2), blurb, count=f"{len(paths)} plot(s)")
    images = find_plots(image_root, ["*.images.*.png"]) if image_root.is_dir() else []
    if not images and image_root.is_dir():
        images = find_plots(image_root, ["**/*.png"])
    sections.append(("images", "Images"))
    body += _section("images", "Images", _plot_grid(images, html_dir, code, "CLEAN images at robust -2 (uniform), 0 and 2 (natural).", wide=True),
                     "CLEAN images per source with Briggs robust -2 (uniform), 0 and +2 (natural) weighting.", count=f"{len(images)} plot(s)")
    return _page(ctx, "final_data", "Final data", sections, body)


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
    logger.info("dashboard: wrote {} page(s) to {}", len(pages), html_dir)
    return html_dir / "index.html"


def build_dashboard(obs, *, flag_statistics: Optional[dict] = None) -> Path:
    """Collect the observation state and render the multi-page dashboard; returns ``html/index.html``."""
    ctx = collect(obs, flag_statistics=flag_statistics)
    return render_pages(ctx, Path(ctx["work_dir"]))


def serve_dashboard(obs, *args, **kwargs) -> Path:
    """Build the static dashboard and open it in the default browser."""
    import webbrowser
    index = build_dashboard(obs)
    webbrowser.open(index.as_uri())
    return index
