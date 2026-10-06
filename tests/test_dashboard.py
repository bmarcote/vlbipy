"""Tests for the multi-page HTML dashboard (vlbipy.dashboard), dummy backend."""
from __future__ import annotations

import json
import re
from pathlib import Path

from vlbipy import VLBIObs
from vlbipy.dashboard import build_dashboard, collect, format_obs_date, snr_class


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
