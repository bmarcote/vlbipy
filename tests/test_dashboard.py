"""Tests for the multi-page HTML dashboard (vlbipy.dashboard), dummy backend."""
from __future__ import annotations

import json
import re
from pathlib import Path

from vlbipy import VLBIObs
from vlbipy.dashboard import (AUTOCORR_PATTERNS, CALIBRATION_GROUPS, CROSSCORR_PATTERNS, FINAL_GROUPS, _sky_map,
                              TIME_EVOLUTION_PATTERNS, build_dashboard, collect, find_plots, format_obs_date,
                              gain_class, match_ifs_to_subbands, plot_source, read_gain_record, render_pages,
                              angular_offset, snr_class, time_evolution_key)


def make_obs(tmp_path, **roles):
    """A single-project dummy observation, already imported, living under tmp_path."""
    roles.setdefault("target", "3C395")
    obs = VLBIObs("db01", network="EVN", backend="dummy", work_dir=tmp_path / "db01", **roles)
    obs.import_data()
    return obs


def test_build_dashboard_writes_four_pages_and_shared_files(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    index = build_dashboard(obs)
    html = Path(index).parent
    assert Path(index).name == "index.html" and html.name == "html"
    for page in ("index.html", "diagnostics.html", "calibration.html", "final_data.html"):
        assert (html / page).is_file(), page
    for asset in ("style.css", "dashboard.js", "dashboard.json"):
        assert (html / asset).is_file(), asset


def test_pages_have_utf8_and_navigate_between_each_other(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    build_dashboard(obs)
    html = Path(obs["db01"].work_dir) / "html"
    for page in html.glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert '<meta charset="utf-8">' in text
        for other in ("index.html", "diagnostics.html", "calibration.html", "final_data.html"):
            assert f'href="{other}"' in text


def test_overview_content(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    build_dashboard(obs)
    text = (Path(obs["db01"].work_dir) / "html" / "index.html").read_text(encoding="utf-8")
    assert "Antennas" in text and "Scans" in text and "Sources" in text
    assert "db01" in text
    # the antenna subband coverage bars exist for every antenna
    assert text.count('class="sbbar"') >= len(obs["db01"].metadata.antennas)


def test_overview_sections_carry_no_item_counts_except_the_comments(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    build_dashboard(obs)
    text = (Path(obs["db01"].work_dir) / "html" / "index.html").read_text(encoding="utf-8")
    headings = re.findall(r'<section id="([^"]+)"><h2>(.*?)</h2>', text)
    assert [anchor for anchor, h2 in headings if 'class="count"' in h2] in ([], ["comments"])
    assert "<h2>Sources</h2>" in text and "<h2>Scans</h2>" in text


def test_angular_offset_distance_and_position_angle():
    rho, pa = angular_offset(10.0, 0.0, 11.0, 0.0)
    assert abs(rho - 1.0) < 1e-9 and abs(pa - 90.0) < 1e-9  # east of the centre
    rho, pa = angular_offset(10.0, 30.0, 10.0, 28.0)
    assert abs(rho - 2.0) < 1e-9 and abs(pa - 180.0) < 1e-9  # south of the centre
    rho, _ = angular_offset(0.0, 89.0, 180.0, 89.0)
    assert abs(rho - 2.0) < 1e-9  # across the pole


def test_sky_map_places_sources_and_joins_targets_to_their_nearest_phase_calibrator():
    sources = [{"name": "TGT", "type": "target", "ra_deg": 280.0, "dec_deg": 32.0, "ra": "", "dec": ""},
               {"name": "PC_NEAR", "type": "phasecal", "ra_deg": 281.0, "dec_deg": 32.0, "ra": "", "dec": ""},
               {"name": "PC_FAR", "type": "phasecal", "ra_deg": 290.0, "dec_deg": 40.0, "ra": "", "dec": ""},
               {"name": "FF", "type": "fringefinder", "ra_deg": 250.0, "dec_deg": 39.0, "ra": "", "dec": ""},
               {"name": "NOPOS", "type": "other", "ra_deg": None, "dec_deg": None}]
    svg = _sky_map(sources)
    assert svg.startswith('<figure class="skymap"><svg') and "NOPOS" not in svg
    for name, role in (("TGT", "target"), ("PC_NEAR", "phasecal"), ("FF", "fringefinder")):
        assert f'class="src {role}"' in svg and f">{name}</text>" in svg
    assert svg.count('class="pair"') == 1 and "50.9\u2032" in svg  # 1 deg in RA at Dec 32
    x = {m[1]: float(m[0]) for m in re.findall(r'<text class="srclabel" x="([-0-9.]+)"[^>]*>([^<]+)<', svg)}
    assert x["FF"] > x["PC_FAR"]  # east is left: the source at lower RA lies to the right
    assert _sky_map(sources[:1]) == "" and _sky_map([]) == ""


def test_format_obs_date_robustly():
    """Observing dates are rendered as ``DD Mmm YYYY`` from date, datetime or strings."""
    from datetime import date, datetime

    assert format_obs_date(date(2018, 10, 30)) == "30 Oct 2018"
    assert format_obs_date(datetime(2018, 10, 30, 14, 30, 0)) == "30 Oct 2018"
    assert format_obs_date("2018-10-30") == "30 Oct 2018"
    assert format_obs_date(None) == ""
    assert format_obs_date("") == ""


def test_overview_date_formatted_and_no_refant_fact(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    build_dashboard(obs)
    text = (Path(obs["db01"].work_dir) / "html" / "index.html").read_text(encoding="utf-8")
    assert "30 Oct 2018" in text
    assert "Reference antenna" not in text


def test_antenna_table_id_order_status_and_health(tmp_path):
    """Antennas keep metadata order, ID is first, statuses are coloured and data health is integrated."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    names = list(obs["db01"].metadata.antennas)
    # Force one antenna to have no data and another to be completely flagged.
    obs["db01"].metadata.antennas[names[0]].observed = False
    per_antenna = {}
    for i, name in enumerate(names):
        if i == 0:
            per_antenna[name] = {"flagged": 0, "observable": 0, "fraction": 0.0}
        elif i == 1:
            per_antenna[name] = {"flagged": 10000, "observable": 10000, "fraction": 1.0}
        else:
            per_antenna[name] = {"flagged": 1000 * (i + 1), "observable": 10000, "fraction": round((i + 1) * 0.1, 3)}
    stats = {"flagged": 20000, "observable": 100000, "fraction": 0.2, "antenna": per_antenna}
    build_dashboard(obs, flag_statistics=stats)
    text = (Path(obs["db01"].work_dir) / "html" / "index.html").read_text(encoding="utf-8")

    assert "<th>ID</th>" in text
    assert 'chip crit">no data' in text
    assert 'chip warn">no signal' in text
    assert 'chip good">observed' in text
    # Old statistics without the exact baseline aggregate are not approximated.
    assert "% of recorded cross-correlation visibilities is flagged overall" in text
    assert "exact baseline-weighted fraction excluding no-signal antennas is unavailable" in text
    # The count sentence lists all three groups.
    assert "with no data" in text
    assert "no usable signal (completely flagged)" in text


def test_exact_excluding_dead_aggregate_is_rendered(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    stats = obs["db01"].flag.statistics()
    stats["excluding_dead"] = {"flagged": 7, "observable": 20, "fraction": 0.35,
                               "excluded_antennas": ["TR"]}
    build_dashboard(obs, flag_statistics=stats)
    text = (Path(obs["db01"].work_dir) / "html" / "index.html").read_text()
    assert "35.0% is flagged when antennas with no usable signal are excluded" in text
    assert "excluded: TR" in text
    assert "double-count" not in text


def test_snr_chip_thresholds():
    assert snr_class(2.0) == "snr-bad"
    assert snr_class(4.0) == "snr-low"
    assert snr_class(8.0) == "snr-mid"
    assert snr_class(30.0) == "snr-good"


def test_collect_context_and_relative_plot_links(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    build_dashboard(obs)
    context = json.loads((Path(obs["db01"].work_dir) / "html" / "dashboard.json").read_text())
    assert context["project"] == "db01"
    assert context["antennas"] and context["scans"] and context["sources"]
    # every scan carries an antenna->snr map and source role
    assert all("antenna_snrs" in s and "role" in s for s in context["scans"])
    # antennas are in metadata insertion order and carry an id
    assert context["antennas"][0]["id"] == 0


def test_collect_without_metadata(tmp_path):
    """A project dir with no metadata still produces the four pages (empty sections)."""
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    obs["db01"]._metadata = None
    index = build_dashboard(obs)
    assert Path(index).is_file()


def test_scan_chips_fixed_width_slots_and_role_snr_fallback(tmp_path):
    """Scan chips keep one fixed-width slot per global antenna and use role-aware SNR colouring."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    # Survey only the calibrators, leaving target scans without significant detections.
    calibrator_names = [s.name for s in obs["db01"].sources if s.source_type.value != "target"]
    survey = obs["db01"]._backend.calibrate.scan_snr(
        "db01", field=",".join(calibrator_names), metadata=obs["db01"].metadata
    )
    obs["db01"].metadata.snr_survey = survey
    build_dashboard(obs)
    text = (Path(obs["db01"].work_dir) / "html" / "index.html").read_text(encoding="utf-8")

    n_global = len(obs["db01"].metadata.antennas)
    n_scans = len(obs["db01"].metadata.scans)
    assert text.count('class="chip scan-chip') == n_global * n_scans
    # Calibrator scans have SNR colours; target scans stay neutral because they were not surveyed.
    assert "snr-good" in text or "snr-mid" in text or "snr-low" in text
    assert "snr-none" in text


def test_calibration_subtabs_are_wrapped_for_scoping(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    # Create a synthetic corner plot so the calibration page builds a subtab group.
    plot_dir = Path(obs["db01"].work_dir) / "plots" / "calibrated"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "db01.corner.phase.png").write_bytes(b"")
    build_dashboard(obs)
    text = (Path(obs["db01"].work_dir) / "html" / "calibration.html").read_text(encoding="utf-8")
    assert 'class="subtabs-wrap"' in text


def test_dashboard_cross_browser_subtabs_and_css(tmp_path):
    """Generated assets avoid browser-fragile syntax (Safari compatibility regression)."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    plot_dir = Path(obs["db01"].work_dir) / "plots" / "calibrated"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name in ("db01.corner_3C345.phase.png", "db01.corner_J1848+3219.phase.png"):
        (plot_dir / name).write_bytes(b"")
    build_dashboard(obs)
    html_dir = Path(obs["db01"].work_dir) / "html"
    css = (html_dir / "style.css").read_text(encoding="utf-8")
    js = (html_dir / "dashboard.js").read_text(encoding="utf-8")
    page = (html_dir / "calibration.html").read_text(encoding="utf-8")

    # CSS: no `inset` shorthand, every color-mix rule preceded by a plain fallback.
    assert "inset:" not in css
    for line in css.splitlines():
        if "color-mix(" in line:
            assert line.index("background:") < line.index("color-mix(")
    # No padding/border shift on the active tab (keeps tab names visible in Safari).
    assert ".tabs a.active" in css and "padding-left:9px" not in css

    # JS: lazy images inside hidden panels are force-loaded on activation; storage guarded.
    assert "closest('button')" in js and "try{" in js
    assert "loading='eager'" in js

    # Panels: unique ids matching data-panel; hidden panels carry no lazy images.
    ids = re.findall(r'class="subpanel[^"]*" id="([^"]+)"', page)
    panels = re.findall(r'data-panel="([^"]+)"', page)
    assert len(ids) == len(set(ids)) == 2 and sorted(ids) == sorted(panels)
    wrap = page.split('<div class="subtabs-wrap">', 1)[1].split('</section>', 1)[0]
    segments = re.split(r'<div class="subpanel[^"]*"', wrap)[1:]
    headers = re.findall(r'<div class="subpanel([^"]*)"', wrap)
    for header, segment in zip(headers, segments):
        if "active" not in header:
            assert 'loading="lazy"' not in segment


def test_calibration_groups_order_and_no_plot_counts(tmp_path):
    """Calibration owns phase jumps/SBD data and keeps solution parameters in scientific order."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    root = Path(obs["db01"].work_dir) / "plots"
    for relative in ("caltables/db01.sbd.phase.png", "caltables/db01.sbd.delay.png",
                     "caltables/db01.sbd.rate.png", "calibrated/db01.calibrated.subband_phases.png"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    build_dashboard(obs)
    calibration = (Path(obs["db01"].work_dir) / "html" / "calibration.html").read_text()
    final = (Path(obs["db01"].work_dir) / "html" / "final_data.html").read_text()
    assert "Bandpass median profile" not in calibration
    assert "Subband phase jumps" in calibration and "Subband phase jumps" not in final
    assert "plot(s)" not in calibration
    assert calibration.index("sbd / phase") < calibration.index("sbd / delay") < calibration.index("sbd / rate")


def test_scan_chip_states_say_why_there_is_no_snr(tmp_path):
    """Every (scan, antenna) cell carries an explicit state instead of a blanket "not surveyed"."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    observation = obs["db01"]
    names = list(observation.metadata.antennas)
    refant, dead, no_fringe, absent = names[0], names[2], names[3], names[-1]
    # Survey the calibrators only, so the target scans are never covered by the fringe fit.
    calibrators = [s.name for s in observation.sources if s.source_type.value != "target"]
    survey = observation._backend.calibrate.scan_snr("db01", field=",".join(calibrators),
                                                     metadata=observation.metadata)
    for antenna in (dead, no_fringe):
        column = survey.antennas.index(antenna)
        for pol in survey.polarizations:
            for row in range(len(survey.scan_numbers)):
                survey.snr[pol][row][column] = float("nan")
    observation.metadata.snr_survey = survey
    observation.metadata.scans[0].antennas = [n for n in observation.metadata.scans[0].antennas if n != absent]
    per_antenna = {name: {"flagged": 0, "observable": 100, "fraction": 0.0} for name in names}
    per_antenna[dead] = {"flagged": 100, "observable": 100, "fraction": 1.0}
    build_dashboard(obs, flag_statistics={"flagged": 100, "observable": 800, "fraction": 0.125,
                                          "antenna": per_antenna})

    html_dir = Path(observation.work_dir) / "html"
    context = json.loads((html_dir / "dashboard.json").read_text())
    surveyed = next(s for s in context["scans"] if s["surveyed"])
    unsurveyed = next(s for s in context["scans"] if not s["surveyed"])
    assert surveyed["antenna_states"][refant] == "refant"
    assert surveyed["antenna_states"][dead] == "nodata"
    assert surveyed["antenna_states"][no_fringe] == "nofringe"
    assert unsurveyed["antenna_states"][no_fringe] == "unsurveyed"
    assert context["scans"][0]["antenna_states"][absent] == "absent"
    # The SNR values stay in the context next to the new states.
    assert any(v is not None for v in surveyed["antenna_snrs"].values())

    text = (html_dir / "index.html").read_text(encoding="utf-8")
    assert f'snr-ref mono" title="{refant}: reference antenna (no SNR of its own)"' in text
    assert f'snr-nodata mono" title="{dead}: no data (fully flagged)"' in text
    assert f'snr-nofringe mono" title="{no_fringe}: no fringe solution"' in text
    assert f'snr-none mono" title="{no_fringe}: scan not surveyed"' in text
    assert f'snr-absent mono" title="{absent}: not in this scan"' in text
    # The legend explains every chip the table can show.
    for label in ("reference antenna", "no data (fully flagged)", "no fringe solution", "scan not surveyed"):
        assert f">{label}</span>" in text


def test_plot_source_matches_the_longest_source_name_first():
    """``3C39`` never steals the files of ``3C395`` and unattributed files return None."""
    names = sorted(["3C395", "3C39", "J1848+3219"], key=len, reverse=True)
    assert plot_source(Path("rsm07.final.radplot.3C395.png"), names) == "3C395"
    assert plot_source(Path("rsm07.final_3C395.spectrum.png"), names) == "3C395"
    assert plot_source(Path("rsm07.final_3C39.spectrum.png"), names) == "3C39"
    assert plot_source(Path("rsm07.final.radplot.J1848+3219.png"), names) == "J1848+3219"
    assert plot_source(Path("rsm07.final.lightcurve.png"), names) is None


def test_final_results_page_has_one_subtab_per_source(tmp_path):
    """The renamed page keeps its file name and splits every plot group per source."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    plots = Path(obs["db01"].work_dir) / "plots" / "calibrated"
    plots.mkdir(parents=True, exist_ok=True)
    for name in ("db01.final.radplot.3C395.png", "db01.final_3C395.spectrum.png",
                 "db01.final_J1848+3219.spectrum.png", "db01.final.lightcurve.png"):
        (plots / name).write_bytes(b"")
    build_dashboard(obs)
    html_dir = Path(obs["db01"].work_dir) / "html"
    page = (html_dir / "final_data.html").read_text(encoding="utf-8")
    index = (html_dir / "index.html").read_text(encoding="utf-8")

    assert "<h1>Final results</h1>" in page
    assert '<a href="final_data.html">Final results</a>' in index
    # Same markup as the calibration corner subtabs, so dashboard.js needs no change.
    assert 'class="subtabs-wrap"' in page and 'class="subtabs"' in page
    ids = re.findall(r'class="subpanel[^"]*" id="([^"]+)"', page)
    assert ids == ["final-3C395", "final-J1848-3219", "final-all-sources"]
    assert sorted(ids) == sorted(re.findall(r'data-panel="([^"]+)"', page))
    assert 'data-panel="final-3C395">3C395 <span class="badge target">target</span>' in page

    bodies = dict(zip(ids, re.split(r'<div class="subpanel[^"]*" id="[^"]+">', page)[1:]))
    assert "db01.final_3C395.spectrum.png" in bodies["final-3C395"]
    assert "db01.final.radplot.3C395.png" in bodies["final-3C395"]
    assert "3C395.spectrum.png" not in bodies["final-J1848-3219"]
    assert "db01.final_J1848+3219.spectrum.png" in bodies["final-J1848-3219"]
    assert "db01.final.lightcurve.png" in bodies["final-all-sources"]
    # Lazy images never load inside a hidden panel (same Safari fix as the corner subtabs).
    assert 'loading="lazy"' in bodies["final-3C395"]
    assert 'loading="lazy"' not in bodies["final-J1848-3219"]
    # Only the sections this source actually has a plot for: an empty box per group would
    # bury the real ones (the total light curve, for one, is a single observation-wide figure).
    assert "Calibrated spectrum" in bodies["final-3C395"]
    assert "Amplitude/phase vs uv distance" in bodies["final-3C395"]
    for title in ("UV coverage", "Total-amplitude light curve"):
        assert title not in bodies["final-3C395"]
    assert 'class="empty"' not in bodies["final-3C395"]


def test_final_results_downloads_list_products_found_on_disk(tmp_path):
    """Per-source downloads are collected from disk into the context and linked relatively."""
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    work = Path(obs["db01"].work_dir)
    (work / "calibrated_data").mkdir(parents=True, exist_ok=True)
    (work / "calibrated_data" / "db01_3C395.uvfits").write_bytes(b"x" * 2 * 1024 * 1024)
    (work / "calibrated_data" / "db01_3C395.ms").mkdir(exist_ok=True)
    (work / "images").mkdir(parents=True, exist_ok=True)
    for name in ("db01_3C395.robust0.fits", "db01_3C395.robust-2.fits"):
        (work / "images" / name).write_bytes(b"y" * 1024)
    build_dashboard(obs)

    context = json.loads((work / "html" / "dashboard.json").read_text())
    entries = context["downloads"]["3C395"]
    assert [e["kind"] for e in entries] == ["uvfits", "ms", "fits", "fits"]
    assert [e["label"] for e in entries][2:] == ["Image FITS (robust -2)", "Image FITS (robust 0)"]
    assert entries[0]["size_human"] == "2.00 MiB" and entries[1]["is_dir"] is True
    assert entries[1]["size_human"] == "" and entries[1]["relpath"] == "calibrated_data/db01_3C395.ms"
    assert context["downloads"]["J1848+3219"] == []

    page = (work / "html" / "final_data.html").read_text(encoding="utf-8")
    assert 'href="../calibrated_data/db01_3C395.uvfits">Calibrated UVFITS</a>' in page
    assert "2.00 MiB" in page
    assert 'class="chip mut">directory</span>' in page
    assert ">calibrated_data/db01_3C395.ms</code>" in page
    # A source with neither plots nor downloads gets no tab at all.
    assert 'data-panel="final-J1848-3219"' not in page


def test_renderer_tolerates_a_context_without_states_or_downloads(tmp_path):
    """An old dashboard.json (no antenna_states, no downloads) still renders every page."""
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    build_dashboard(obs)
    context = json.loads((Path(obs["db01"].work_dir) / "html" / "dashboard.json").read_text())
    context.pop("downloads")
    for scan in context["scans"]:
        scan.pop("antenna_states")
        scan.pop("surveyed")
    out = tmp_path / "replay"
    index = render_pages(context, out)
    assert Path(index).is_file()
    text = (out / "html" / "index.html").read_text(encoding="utf-8")
    assert 'class="chip scan-chip' in text
    assert "Final results" in (out / "html" / "final_data.html").read_text(encoding="utf-8")


# -- amplitude corrections of the difmapy self-calibration --------------------------------------

def subband_centre(subband: int) -> float:
    """Centre frequency (Hz) of a subband of :func:`gain_obs`."""
    return 1.6e9 + 32e6 * subband + 15.5e6


def gain_obs(tmp_path):
    """Dummy observation with four subbands of known frequency and uneven antenna participation.

    Antennas (metadata order): EF, YS, O8, MC, TR, WB, JB, NT. YS recorded subbands 1 and 2 only and O8
    did not observe.
    """
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    metadata = obs["db01"].metadata
    metadata.freq_setup.n_subbands = 4
    metadata.freq_setup.channel_freqs = [[1.6e9 + 32e6 * sb + 1e6 * ch for ch in range(32)] for sb in range(4)]
    for antenna in metadata.antennas.values():
        antenna.subbands = (0, 1, 2, 3)
    metadata.antennas["YS"].subbands = (1, 2)
    metadata.antennas["O8"].observed = False
    return obs


def write_selfcal(work: Path, source: str, *, gains: dict | None = None, report: dict | None = None) -> None:
    """Write the difmapy gain record and/or Bayesian report of ``source`` under ``<work>/selfcal``."""
    selfcal = Path(work) / "selfcal"
    selfcal.mkdir(parents=True, exist_ok=True)
    if gains is not None:
        (selfcal / f"db01.{source}.gains.json").write_text(json.dumps(gains))
    if report is not None:
        (selfcal / f"db01.{source}.amp.json").write_text(json.dumps(report))


def phasecal_gains() -> dict:
    """A gain record (``gains.json``) on a split holding subbands 1-3; MC is missing from it."""
    return {"source": "J1848+3219", "antennas": ["EF", "YS", "O8", "TR", "WB", "JB", "NT"],
            "if_freqs_hz": [subband_centre(1), subband_centre(2), subband_centre(3)],
            "gain_per_if": {"EF": [1.05, 0.95, 1.20], "YS": [0.80, 1.0, 1.0], "O8": [1.0, 1.0, 1.0],
                            "TR": [1.0, 1.0, 1.0], "WB": [1.0, 1.02, 0.98], "JB": [1.5, 1.4, 0.6],
                            "NT": [1.0, 1.0, 1.0]},
            "sigma": {"EF": 0.02, "YS": 0.03, "O8": None, "TR": None, "WB": 0.01, "JB": 0.05, "NT": 0.02}}


def finder_report() -> dict:
    """A Bayesian report (``amp.json``) of an older run that wrote no gain record."""
    def per_if(*corrections):
        return [{"if": number, "correction": value} for number, value in enumerate(corrections, start=1)]

    return {"source": "3C345", "applied": True,
            "if_freqs_hz": [subband_centre(sb) for sb in range(4)],
            "corrections_per_if": {"EF": [0.9, 1.0, 1.1, 1.0], "YS": [1.0, 0.7, 0.75, 1.0], "JB": [1.0, 1.0, 1.0, 1.0]},
            "stations": [
                {"station": "EF", "has_data": True, "sigma": 0.04, "p_correction": 0.9, "naive_gscale": 1.3,
                 "per_if": per_if(0.9, None, 1.1, 1.0)},
                {"station": "YS", "has_data": True, "sigma": 0.1, "p_correction": 0.99, "naive_gscale": 0.6,
                 "per_if": per_if(None, 0.7, 0.75, None)},
                {"station": "JB", "has_data": False, "sigma": None, "p_correction": None, "naive_gscale": None,
                 "per_if": per_if(None, None, None, None)}],
            "findings": ["Best source model: clean (posterior probability 1.000)",
                         "YS: a correction is needed in IF2 x0.700 +- 4.0% (P = 0.999)"]}


def test_gain_class_follows_the_displayed_value_at_the_thresholds():
    """Green within 10%, orange within 30%, red beyond - judged on the two decimals the cell shows."""
    for gain in (1.0, 1.10, 0.90, 1.104, 0.896):
        assert gain_class(gain) == "gain-good", gain
    for gain in (1.11, 0.89, 1.30, 0.70, 1.304):
        assert gain_class(gain) == "gain-warn", gain
    for gain in (1.31, 0.69, 2.5, 0.1):
        assert gain_class(gain) == "gain-bad", gain
    for gain in (None, float("nan")):
        assert gain_class(gain) == "gain-none"


def test_ifs_are_matched_to_subbands_by_frequency_then_by_position():
    channel_freqs = [[1.6e9 + 32e6 * sb + 1e6 * ch for ch in range(32)] for sb in range(4)]
    centre = [subband_centre(sb) for sb in range(4)]
    # A split that kept two of the four subbands: IF 1 is subband 1 and IF 2 is subband 3.
    assert match_ifs_to_subbands([centre[1], centre[3]], channel_freqs, 2, 4) == [1, 3]
    # A frequency outside every subband stays unmatched instead of being guessed.
    assert match_ifs_to_subbands([centre[0], 9.9e9], channel_freqs, 2, 4) == [0, None]
    # Without frequencies the position is only trusted when the split kept every subband.
    assert match_ifs_to_subbands([], channel_freqs, 4, 4) == [0, 1, 2, 3]
    assert match_ifs_to_subbands(centre, [], 4, 4) == [0, 1, 2, 3]
    assert match_ifs_to_subbands([], [], 2, 4) == [None, None]
    # Frequencies on another scale fall in no subband at all: back to the position.
    assert match_ifs_to_subbands([c / 1e9 for c in centre], channel_freqs, 4, 4) == [0, 1, 2, 3]


def test_gain_record_prefers_the_applied_gains_and_falls_back_to_the_bayesian_report(tmp_path):
    selfcal = tmp_path / "selfcal"
    report = finder_report()
    report["corrections_per_if"] = {"EF": [9.0, 9.0, 9.0], "YS": [9.0, 9.0, 9.0]}
    write_selfcal(tmp_path, "J1848+3219", gains=phasecal_gains(), report=report)
    write_selfcal(tmp_path, "3C345", report=finder_report())

    both = read_gain_record(selfcal, "db01", "J1848+3219")
    assert both["file"] == "db01.J1848+3219.gains.json"
    assert both["gain_per_if"]["EF"] == [1.05, 0.95, 1.20]          # the applied gains, not the report's
    assert both["sigma"]["EF"] == 0.02 and both["sigma"]["TR"] is None
    assert set(both["stations"]) == {"EF", "YS", "JB"} and both["findings"] == report["findings"]

    old = read_gain_record(selfcal, "db01", "3C345")
    assert old["file"] == "db01.3C345.amp.json" and old["applied"] is True
    assert old["gain_per_if"]["YS"] == [1.0, 0.7, 0.75, 1.0]
    assert old["sigma"] == {"EF": 0.04, "YS": 0.1, "JB": None} and old["p_correction"]["YS"] == 0.99
    assert old["if_freqs_hz"] == [subband_centre(sb) for sb in range(4)]

    assert read_gain_record(selfcal, "db01", "3C395") == {}
    (selfcal / "db01.3C395.gains.json").write_text("not json")
    assert read_gain_record(selfcal, "db01", "3C395") == {}


def test_gain_corrections_blank_unsolved_cells_and_take_the_median_over_solved_ones(tmp_path):
    """A 1.0 where a station has no solution is a filler: it is neither shown nor counted in the median."""
    obs = gain_obs(tmp_path)
    work = Path(obs["db01"].work_dir)
    write_selfcal(work, "J1848+3219", gains=phasecal_gains())
    write_selfcal(work, "3C345", report=finder_report())
    context = collect(obs)
    json.dumps(context["gain_corrections"])                         # JSON-safe
    # One entry per self-calibrated source, in the order of the source table; the target has none.
    assert [s["name"] for s in context["sources"]] == ["3C395", "J1848+3219", "3C345"]
    phasecal, finder = context["gain_corrections"]
    names = ["EF", "YS", "O8", "MC", "TR", "WB", "JB", "NT"]

    # Gain record only: the metadata and the station uncertainties say where there is no solution.
    assert (phasecal["source"], phasecal["role"]) == ("J1848+3219", "phasecal")
    assert phasecal["file"] == "selfcal/db01.J1848+3219.gains.json" and phasecal["applied"] is None
    assert phasecal["antennas"] == names
    assert phasecal["subbands"] == [1, 2, 3] and phasecal["subband_labels"] == ["SB 1", "SB 2", "SB 3"]
    assert phasecal["if_freqs_hz"] == [subband_centre(1), subband_centre(2), subband_centre(3)]
    gains = phasecal["gain_per_if"]
    assert gains["EF"] == [1.05, 0.95, 1.20] and phasecal["gain_median"]["EF"] == 1.05
    assert gains["YS"] == [0.80, 1.0, None]                          # YS did not record subband 3
    assert abs(phasecal["gain_median"]["YS"] - 0.90) < 1e-12        # ... so its filler 1.0 is not counted
    assert gains["O8"] == [None, None, None]                         # did not observe
    assert gains["MC"] == [None, None, None]                         # not in the record at all
    assert gains["TR"] == [None, None, None]                         # observed, but no uncertainty = no solution
    assert gains["NT"] == [1.0, 1.0, 1.0] and phasecal["gain_median"]["NT"] == 1.0   # solved, and exactly 1
    assert phasecal["gain_median"]["JB"] == 1.4 and phasecal["gain_median"]["WB"] == 1.0
    assert [phasecal["gain_median"][n] for n in ("O8", "MC", "TR")] == [None, None, None]
    assert phasecal["has_data"] == {n: n not in ("O8", "MC", "TR") for n in names}
    assert phasecal["sigma"]["WB"] == 0.01 and phasecal["sigma"]["MC"] is None
    assert set(phasecal["p_correction"].values()) == {None} and set(phasecal["naive_gscale"].values()) == {None}
    assert phasecal["findings"] == []

    # Bayesian report only (older run): it decides, per station and per IF.
    assert (finder["source"], finder["role"]) == ("3C345", "fringefinder")
    assert finder["file"] == "selfcal/db01.3C345.amp.json" and finder["applied"] is True
    assert finder["subbands"] == [0, 1, 2, 3]
    assert finder["gain_per_if"]["EF"] == [0.9, None, 1.1, 1.0]     # IF 2 unsolved; the 1.0 of IF 4 is real
    assert finder["gain_median"]["EF"] == 1.0
    # YS has a solution in subbands 1 and 2 only.
    assert finder["gain_per_if"]["YS"] == [None, 0.7, 0.75, None]
    assert abs(finder["gain_median"]["YS"] - 0.725) < 1e-12
    assert finder["gain_per_if"]["JB"] == [None] * 4 and finder["has_data"]["JB"] is False
    assert finder["gain_per_if"]["WB"] == [None] * 4                 # not in the report
    assert finder["naive_gscale"]["EF"] == 1.3 and finder["naive_gscale"]["WB"] is None
    assert finder["sigma"]["YS"] == 0.1 and finder["p_correction"]["EF"] == 0.9
    assert finder["findings"][1].startswith("YS: a correction is needed")


def test_calibration_page_has_one_gain_table_per_source(tmp_path):
    obs = gain_obs(tmp_path)
    work = Path(obs["db01"].work_dir)
    write_selfcal(work, "J1848+3219", gains=phasecal_gains())
    write_selfcal(work, "3C345", report=finder_report())
    build_dashboard(obs)
    page = (work / "html" / "calibration.html").read_text(encoding="utf-8")
    css = (work / "html" / "style.css").read_text(encoding="utf-8")

    # The section sits right after the chain and is reachable from the side navigation.
    assert '<a href="#amplitude-corrections">Amplitude corrections</a>' in page
    assert page.index('<section id="chain">') < page.index('<section id="amplitude-corrections">') \
        < page.index('<section id="gain-curves">')
    section = page.split('<section id="amplitude-corrections">', 1)[1].split("</section>", 1)[0]
    for key in ('gainkey gain-good">within 10%', 'gainkey gain-warn">10% - 30%', 'gainkey gain-bad">beyond 30%',
                'gainkey gain-none">&ndash; no solution'):
        assert key in section, key

    # One sub-tab per source, in the order of the source table, with the usual sub-tab markup.
    ids = re.findall(r'class="subpanel[^"]*" id="([^"]+)"', section)
    assert ids == ["gains-J1848-3219", "gains-3C345"]
    assert ids == re.findall(r'data-panel="([^"]+)"', section)
    assert 'data-panel="gains-J1848-3219">J1848+3219 <span class="badge phasecal">phasecal</span>' in section
    tables = dict(zip(ids, re.split(r'<div class="subpanel[^"]*" id="[^"]+">', section)[1:]))
    phasecal, finder = tables["gains-J1848-3219"], tables["gains-3C345"]
    assert ">selfcal/db01.J1848+3219.gains.json</p>" in phasecal and ">selfcal/db01.3C345.amp.json</p>" in finder

    # Rows are antennas, columns the subbands of the split (labelled) and the median comes last.
    assert phasecal.count('<td class="gain-ant mono">') == 8 and finder.count('<td class="gain-ant mono">') == 8
    header = re.findall(r"<th(?: [^>]*)?>(.*?)</th>", phasecal)
    assert header == ["Antenna", "SB 1<small>1648 MHz</small>", "SB 2<small>1680 MHz</small>",
                      "SB 3<small>1712 MHz</small>", "Median"]
    assert '<th title="SB 1 (1647.5 MHz)">' in phasecal             # the exact frequency stays in the tooltip
    ef_row = phasecal.split("<b>EF</b>", 1)[1].split("</tr>", 1)[0]
    assert re.findall(r'<td class="([^"]+)"[^>]*>([^<]*)</td>', ef_row) == [
        ("gain gain-good", "1.05"), ("gain gain-good", "0.95"), ("gain gain-warn", "1.20"),
        ("gain gain-good gain-median", "1.05")]
    assert '<td class="gain gain-warn" title="EF · SB 3 (1711.5 MHz): x1.2000 (+20.0%)">1.20</td>' in phasecal
    jb_row = phasecal.split("<b>JB</b>", 1)[1].split("</tr>", 1)[0]
    assert re.findall(r'<td class="([^"]+)"', jb_row) == ["gain gain-bad"] * 3 + ["gain gain-bad gain-median"]
    assert 'title="JB · median of 3 subband(s): x1.4000 (+40.0%)">1.40</td>' in jb_row
    # No solution: a neutral dash, never a green 1.00 - in the cells and in the median alike.
    assert '<td class="gain gain-none" title="YS · SB 3 (1711.5 MHz): no solution">&ndash;</td>' in phasecal
    o8_row = phasecal.split("<b>O8</b>", 1)[1].split("</tr>", 1)[0]
    assert re.findall(r'<td class="([^"]+)"[^>]*>([^<]*)</td>', o8_row) == [("gain gain-none", "&ndash;")] * 3 + [
        ("gain gain-none gain-median", "&ndash;")]
    ys_row = finder.split("<b>YS</b>", 1)[1].split("</tr>", 1)[0]
    assert re.findall(r'<td class="([^"]+)"[^>]*>([^<]*)</td>', ys_row) == [
        ("gain gain-none", "&ndash;"), ("gain gain-warn", "0.70"), ("gain gain-warn", "0.75"),
        ("gain gain-none", "&ndash;"), ("gain gain-warn gain-median", "0.72")]

    # The coloured cells touch: no cell border, no row rule and no row striping inside this table.
    assert ".gaintable th,.gaintable td{text-align:center;padding:6px 8px;border:0}" in css
    assert ".gaintable tbody tr,.gaintable tbody tr:nth-child(odd){border-top:0;background:var(--panel)}" in css
    gain_rules = "".join(line for line in css.splitlines() if ".gain" in line)
    assert "border-radius" not in gain_rules and "border-left" not in gain_rules and "margin" not in gain_rules
    # The antenna and median columns stay in view when a table with many subbands scrolls sideways.
    assert ".gaintable .gain-ant{text-align:left;position:sticky;left:0;z-index:1}" in css
    assert ".gaintable .gain-median{position:sticky;right:0;" in css
    # Both themes define the three colours.
    for token in ("--gain-good:", "--gain-warn:", "--gain-bad:"):
        assert css.count(token) == 2, token


def test_calibration_page_with_a_single_gain_table_needs_no_subtabs(tmp_path):
    obs = gain_obs(tmp_path)
    work = Path(obs["db01"].work_dir)
    report = finder_report()
    report["applied"] = False
    write_selfcal(work, "3C345", report=report)
    build_dashboard(obs)
    section = (work / "html" / "calibration.html").read_text(encoding="utf-8")
    section = section.split('<section id="amplitude-corrections">', 1)[1].split("</section>", 1)[0]
    assert "subtabs" not in section and section.count('<table class="gaintable">') == 1
    assert '<h3>3C345 <span class="badge fringefinder">fringefinder</span></h3>' in section
    # Corrections the self-calibration computed but rejected are flagged as such.
    assert "computed but not applied to the data" in section


def test_calibration_page_without_gain_records_and_with_an_old_context(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219")
    build_dashboard(obs)
    work = Path(obs["db01"].work_dir)
    context = json.loads((work / "html" / "dashboard.json").read_text())
    assert context["gain_corrections"] == []
    notice = "No amplitude corrections from the self-calibration available."
    assert notice in (work / "html" / "calibration.html").read_text(encoding="utf-8")
    # A dashboard.json written before the key existed still renders.
    context.pop("gain_corrections")
    render_pages(context, tmp_path / "replay")
    assert notice in (tmp_path / "replay" / "html" / "calibration.html").read_text(encoding="utf-8")


# -- diagnostics sub-tabs -----------------------------------------------------------------------

def test_time_evolution_key_reads_reference_antenna_and_subband():
    assert time_evolution_key(Path("em163.raw.time_evolution.refEF.spw12.png")) == ("EF", 12)
    assert time_evolution_key(Path("rsm07.raw.time_evolution.refT6.spw0.png")) == ("T6", 0)
    assert time_evolution_key(Path("rsm07.raw.time_evolution.png")) is None
    assert time_evolution_key(Path("rsm07.raw_3C345.timeseries.png")) is None


def test_diagnostics_page_has_three_horizontal_subtabs(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs["db01"].work_dir)
    build_dashboard(obs)
    empty = (work / "html" / "diagnostics.html").read_text(encoding="utf-8")
    assert empty.count("No plots available for this section.") == 3

    raw = work / "plots" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("db01.raw_scan1.autocorr.png", "db01.raw_scan1.spectrum.png",
                 "db01.raw.time_evolution.refAR.spw0.png", "db01.raw.time_evolution.refEF.spw10.png",
                 "db01.raw.time_evolution.refEF.spw2.png", "db01.raw.time_evolution.notes.png"):
        (raw / name).write_bytes(b"")
    build_dashboard(obs)
    page = (work / "html" / "diagnostics.html").read_text(encoding="utf-8")

    assert re.findall(r'<button class="([^"]*)" data-panel="([^"]+)">([^<]+)</button>', page) == [
        ("active", "diag-autocorr", "Auto-correlations"), ("", "diag-crosscorr", "Cross-correlations"),
        ("", "diag-time-evolution", "Time evolution")]
    assert page.count('class="subtabs-wrap"') == 1
    ids = re.findall(r'class="subpanel[^"]*" id="([^"]+)"', page)
    assert ids == ["diag-autocorr", "diag-crosscorr", "diag-time-evolution"]
    bodies = dict(zip(ids, re.split(r'<div class="subpanel[^"]*" id="[^"]+">', page)[1:]))
    auto, cross, evolution = (bodies[i] for i in ids)

    # Every file lands in its own sub-tab only, each with its blurb and no plot count.
    assert "db01.raw_scan1.autocorr.png" in auto and "spectrum.png" not in auto and "time_evolution" not in auto
    assert "db01.raw_scan1.spectrum.png" in cross and "autocorr.png" not in cross and "time_evolution" not in cross
    assert "autocorr.png" not in evolution and "spectrum.png" not in evolution
    assert "plot(s)" not in page
    assert "Auto-correlation spectra per antenna" in auto and "vector-averaged over the central channels" in evolution
    # The observation's reference antenna (EF) leads although AR sorts first; subbands in numeric order;
    # a file that does not follow the naming convention is still shown, last.
    order = ["Reference antenna EF", "refEF.spw2.png", "refEF.spw10.png", "Reference antenna AR", "refAR.spw0.png",
             "<h3>Other</h3>", "time_evolution.notes.png"]
    positions = [evolution.index(text) for text in order]
    assert positions == sorted(positions)
    assert "Baselines to EF · subband 10" in evolution
    # Lazy images never load inside a hidden panel: only the active one keeps the attribute.
    assert 'loading="lazy"' in auto and 'loading="lazy"' not in cross and 'loading="lazy"' not in evolution
    # The sections live in sub-tabs now, so the side navigation has no in-page anchors.
    assert '<nav class="nav">' not in page


def test_time_evolution_files_belong_to_no_other_plot_group(tmp_path):
    root = tmp_path / "plots"
    (root / "raw").mkdir(parents=True)
    name = "db01.raw.time_evolution.refEF.spw3.png"
    (root / "raw" / name).write_bytes(b"")
    assert [p.name for p in find_plots(root, TIME_EVOLUTION_PATTERNS)] == [name]
    others = (AUTOCORR_PATTERNS + CROSSCORR_PATTERNS
              + [pattern for _, _, patterns in CALIBRATION_GROUPS for pattern in patterns]
              + [pattern for _, _, patterns, fallback in FINAL_GROUPS for pattern in patterns + fallback])
    assert find_plots(root, others) == []
    # ... and none of the plots already written to plots/raw is taken for a time-evolution figure.
    for other in ("db01.raw_scan1.autocorr.png", "db01.raw_scan1.spectrum.png", "db01.raw_3C345.timeseries.png",
                  "db01.sbd_stage1_scan63.timeseries.png", "db01.snr_matrix.RR.png", "db01.uv_coverage.png",
                  "db01.raw.radplot.3C345.png", "db01.raw_3C345.corner.phase.png"):
        (root / "raw" / other).write_bytes(b"")
    assert [p.name for p in find_plots(root, TIME_EVOLUTION_PATTERNS)] == [name]


# -- general comments ----------------------------------------------------------------------------

def commented_obs(tmp_path):
    """Dummy observation with something to comment on: O8 never observed, YS on part of the band, TR
    leaving after the third scan, and the instrumental delay solved on scan 3 (3C345)."""
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    observation = obs["db01"]
    metadata = observation.metadata
    metadata.antennas["O8"].observed = False
    metadata.antennas["YS"].subbands = (1, 2)
    for scan in metadata.scans[3:]:
        scan.antennas = [name for name in scan.antennas if name != "TR"]
    solved = [name for name in metadata.antennas if name != "O8"]
    observation.set_cal_selection(solved, [3], [{"scan": 3, "antennas": solved, "refant": "EF"}])
    return obs


def test_overview_shows_the_general_comments_between_observation_and_antennas(tmp_path):
    obs = commented_obs(tmp_path)
    build_dashboard(obs)
    html = Path(obs["db01"].work_dir) / "html"
    text = (html / "index.html").read_text(encoding="utf-8")
    assert text.index('id="observation"') < text.index('id="comments"') < text.index('id="antennas"')
    assert '<a href="#observation">Observation</a><a href="#comments">General comments</a><a href="#antennas">' in text
    section = text[text.index('id="comments"'):text.index('id="antennas"')]
    assert "<h3>Data and calibration</h3>" in section and "<h3>Antennas</h3>" in section
    assert ("Scan 3 (3C345) was selected for the instrumental delay (reference antenna EF): it was observed by all 7 "
            "antennas with usable data, and all were solved there.") in section
    # One block per antenna with something to say, its code prominent and the severity as a chip.
    for name in ("YS", "O8", "TR"):
        assert f'<div class="cmt-name mono">{name}</div>' in section
    assert '<div class="cmt-name mono">EF</div>' not in section
    assert '<li class="cmt-warning"><span class="chip warn">warning</span><span>no data in scans 4-9</span></li>' in section
    assert '<span class="chip mut">note</span><span>recorded only subbands 1-2 (2 of 8)</span>' in section
    assert "Nothing to report for: EF, MC, WB, JB, NT." in section
    assert "2 warning(s) · 2 note(s)" in section
    # The page shows exactly the list the context carries.
    context = json.loads((html / "dashboard.json").read_text())
    assert [c["category"] for c in context["comments"]] == ["calibration", "antenna", "antenna", "antenna", "antenna"]
    for comment in context["comments"]:
        assert comment["text"] in section


def test_collect_gathers_what_the_comments_need_from_the_caches(tmp_path):
    from vlbipy.models import ScanSNRSurvey
    obs = commented_obs(tmp_path)
    observation = obs["db01"]
    metadata, work = observation.metadata, Path(observation.work_dir)
    names, nan = list(metadata.antennas), float("nan")
    fringes = [[nan if name in ("EF", "O8") else 40.0 for name in names] for _ in metadata.scans]
    no_jb = [[nan if name == "JB" else value for name, value in zip(names, row)] for row in fringes]
    metadata.snr_survey = ScanSNRSurvey(project_code="db01", scan_numbers=[s.scan_number for s in metadata.scans],
                                        scan_sources=[s.source for s in metadata.scans], antennas=names,
                                        snr={"RR": fringes, "LL": no_jb}, refant="EF")
    (work / "images").mkdir(parents=True, exist_ok=True)
    (work / "selfcal").mkdir(parents=True, exist_ok=True)
    (work / "images" / "db01_3C395.search.json").write_text(json.dumps(
        {"search": {"detected": True, "snr": 26.6, "peak": 1.36e-4, "rms": 5e-6, "fov_mas": 1000.0, "offset_mas": 0.8,
                    "offset_beams": 0.3, "recentred": False}, "shift_mas": [0.0, 0.0]}))
    (work / "selfcal" / "db01.J1848+3219.json").write_text(json.dumps(
        {"images": {"0.0": {"robust": 0.0, "peak": 0.308, "rms": 3.3e-4, "dynamic_range": 933.0},
                    "-2.0": {"robust": -2.0, "peak": 0.30, "rms": 3.9e-4, "dynamic_range": 770.0}}}))
    (work / ".run_warnings.json").write_text(json.dumps(
        {"db01: antenna WB shows no usable fringes on 3C345": "2026-10-10 12:07:00",
         "ANOMALY: db01: flag[quack] removed 61.0% of data": "2026-10-10 12:08:00"}))

    context = collect(obs)
    assert context["calibration_selection"] == [{"scan": 3, "source": "3C345", "refant": "EF",
                                                 "antennas": [n for n in names if n != "O8"]}]
    assert context["steps"] == {"import_data": {"status": "done"}}
    assert context["polarization_snr"]["hands"] == ["RR", "LL"]
    assert context["polarization_snr"]["antennas"]["JB"] == {"scans": 9, "detected": {"RR": 9, "LL": 0},
                                                             "median": {"RR": 40.0, "LL": None}}
    assert context["polarization_snr"]["antennas"]["TR"]["scans"] == 3           # only the scans it was in
    assert [(r["source"], r["role"]) for r in context["source_results"]] == [("3C395", "target"),
                                                                             ("J1848+3219", "phasecal")]
    assert context["source_results"][0]["search"]["detected"] is True and context["source_results"][0]["images"] == []
    assert [image["robust"] for image in context["source_results"][1]["images"]] == [-2.0, 0.0]
    assert context["run_warnings"] == [
        {"message": "antenna WB shows no usable fringes on 3C345", "anomaly": False, "first_seen_utc": "2026-10-10 12:07:00"},
        {"message": "flag[quack] removed 61.0% of data", "anomaly": True, "first_seen_utc": "2026-10-10 12:08:00"}]
    json.dumps(context)                                               # everything stays JSON-safe

    by_antenna = {(c["antenna"], c["text"]) for c in context["comments"]}
    assert ("JB", "no signal in LCP (LL): no fringes in any of the 9 surveyed scans it took part in, while RCP (RR) "
                  "was detected in 9") in by_antenna
    assert ("WB", "shows no usable fringes on 3C345") in by_antenna
    per_source = {(c["source"], c["role"], c["text"]) for c in context["comments"] if c["category"] == "source"}
    assert ("3C395", "target", "Detected at 27 sigma in the search map (dirty-map peak 136 uJy/beam), 0.8 mas "
            "(0.3 beams) from the phase centre.") in per_source
    assert ("J1848+3219", "phasecal", "Image (robust 0): peak 308 mJy/beam, rms 330 uJy/beam, dynamic "
            "range 933.") in per_source

    page = render_pages(context, tmp_path / "replay")
    text = Path(page).read_text(encoding="utf-8")
    # One box per source, headed by its name in bold and its role badge, before the antennas.
    sources = text[text.index("<h3>Sources</h3>"):text.index("<h3>Antennas</h3>")]
    assert sources.count('<div class="cmt-src">') == 2
    assert '<div class="cmt-src-head"><b>3C395</b> <span class="badge target">target</span></div>' in sources
    table_order = [s["name"] for s in context["sources"] if f"<b>{s['name']}</b>" in sources]
    assert table_order == sorted(table_order, key=lambda name: sources.index(f"<b>{name}</b>"))
    assert len(table_order) == 2
    assert "<h3>Warnings raised by the pipeline</h3>" in text
    assert '<span class="when">first seen 2026-10-10 12:07:00 UTC</span>' in text
    assert '<li class="cmt-problem"><span class="chip crit">problem</span><span>flag[quack] removed 61.0% of data' in text


def test_build_dashboard_rewrites_the_summary_with_the_same_comments(tmp_path):
    obs = commented_obs(tmp_path)
    work = Path(obs["db01"].work_dir)
    assert not (work / "summary.md").exists()                         # the dummy import writes none
    build_dashboard(obs)
    summary = (work / "summary.md").read_text()
    assert summary.startswith("VLBI Observation Summary: db01")
    assert summary.index("General Comments for db01") < summary.index("Source Summary for db01")
    comments = json.loads((work / "html" / "dashboard.json").read_text())["comments"]
    flat = " ".join(summary.split())
    assert comments and all(" ".join(comment["text"].split()) in flat for comment in comments)
    assert "  TR  WARNING  no data in scans 4-9" in summary.splitlines()


def test_a_summary_that_cannot_be_written_does_not_cost_the_dashboard(tmp_path, monkeypatch):
    import vlbipy.diagnostics

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(vlbipy.diagnostics, "write_summary", broken)
    obs = commented_obs(tmp_path)
    index = build_dashboard(obs)
    assert Path(index).is_file() and not (Path(obs["db01"].work_dir) / "summary.md").exists()


def test_overview_of_a_context_without_comments(tmp_path):
    """A dashboard.json saved before the comments existed still renders; an empty list says so plainly."""
    obs = commented_obs(tmp_path)
    context = collect(obs)
    context.pop("comments")
    text = Path(render_pages(context, tmp_path / "old")).read_text(encoding="utf-8")
    assert 'id="comments"' in text and "collected before the general comments existed" in text
    context["comments"] = []
    text = Path(render_pages(context, tmp_path / "empty")).read_text(encoding="utf-8")
    assert '<p class="empty">Nothing to report.</p>' in text
