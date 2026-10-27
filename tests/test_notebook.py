"""Tests for the generated reduction notebook (vlbipy.notebook), dummy backend.

The dummy backend writes no plots, so each test creates small files with the names a real run leaves on disk.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest

nbformat = pytest.importorskip("nbformat")

from vlbipy import VLBIObs  # noqa: E402 - after the importorskip
from vlbipy.cli import _cmd_notebook, build_parser  # noqa: E402
from vlbipy.models import CalTable  # noqa: E402
from vlbipy.notebook import (IMAGE_LINK, build_notebook, image_markdown, notebook_summary,  # noqa: E402
                             plots_cell, selfcal_report_markdown, step_plots, write_notebook)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_obs(tmp_path, **roles):
    """A single-project dummy observation, already imported, living under tmp_path."""
    roles.setdefault("target", "3C395")
    campaign = VLBIObs("db01", network="EVN", backend="dummy", work_dir=tmp_path / "db01", **roles)
    campaign.import_data()
    return campaign["db01"]


def touch(work, *relatives) -> list[Path]:
    """Create plot files under the work directory and return their paths.

    Each holds the PNG signature followed by its own relative name, so the files are non-empty and distinct.
    """
    paths = [Path(work) / relative for relative in relatives]
    for path, relative in zip(paths, relatives):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_SIGNATURE + relative.encode())
    return paths


def cell_index(nb, heading: str) -> int:
    """Index of the markdown cell that starts with ``heading``."""
    return next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith(heading))


def image_urls(nb) -> list[str]:
    """Every image URL shown by the markdown cells."""
    return [url for c in nb.cells if c.cell_type == "markdown" for url in IMAGE_LINK.findall(c.source)]


def attachment_bytes(cell, url: str) -> bytes:
    """The decoded bytes of the attachment an ``attachment:<key>`` URL of ``cell`` points to."""
    assert url.startswith("attachment:")
    return base64.b64decode(cell["attachments"][url[len("attachment:"):]]["image/png"])


# ---- embedded plots (the default)

def test_plots_are_embedded_by_default_as_cell_attachments(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    profile, _ = touch(obs.work_dir, "plots/caltables/db01.bandpass_profile.png", "images/db01.images.3C395.png")
    obs._state.mark_complete("flag_edges", outputs=["edge=1", "measured"])
    nb = build_notebook(obs)
    nbformat.validate(nb)
    index = cell_index(nb, "### Subband edges")
    plots, rerun = nb.cells[index + 1], nb.cells[index + 2]
    assert "**bandpass profile** — `plots/caltables/db01.bandpass_profile.png`" in plots.source
    assert "![bandpass profile](attachment:0_db01.bandpass_profile.png)" in plots.source
    assert attachment_bytes(plots, "attachment:0_db01.bandpass_profile.png") == profile.read_bytes()
    assert rerun.cell_type == "code" and rerun.source == "obs.flag.edges(force=FORCE)"
    urls = image_urls(nb)
    assert len(urls) == 2 and all(url.startswith("attachment:") for url in urls)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            assert all(url[len("attachment:"):] in cell["attachments"] for url in IMAGE_LINK.findall(cell.source))
        assert all(output.output_type == "stream" for output in cell.get("outputs", []))
    assert nb.metadata["vlbipy"]["plots"] == 2 and nb.metadata["vlbipy"]["embedded"] is True


def test_attachment_key_has_no_plus_but_the_caption_keeps_the_path(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    path, = touch(obs.work_dir, "plots/calibrated/db01.calibrated_J1848+3219.spectrum.png")
    nb = build_notebook(obs)
    cell = nb.cells[cell_index(nb, "### Calibrators after calibration")]
    assert list(cell["attachments"]) == ["0_db01.calibrated_J1848_3219.spectrum.png"]
    assert "`plots/calibrated/db01.calibrated_J1848+3219.spectrum.png`" in cell.source
    assert IMAGE_LINK.findall(cell.source) == ["attachment:0_db01.calibrated_J1848_3219.spectrum.png"]
    assert attachment_bytes(cell, IMAGE_LINK.findall(cell.source)[0]) == path.read_bytes()


def test_same_file_name_in_two_directories_gets_two_keys(tmp_path):
    obs = make_obs(tmp_path)
    work = Path(obs.work_dir)
    raw, calibrated = touch(work, "plots/raw/db01.sbd_stage1_scan63.timeseries.png",
                            "plots/calibrated/db01.sbd_stage1_scan63.timeseries.png")
    cell = plots_cell([raw, calibrated], "db01", work, work, embed=True)
    urls = IMAGE_LINK.findall(cell.source)
    assert urls == ["attachment:0_db01.sbd_stage1_scan63.timeseries.png",
                    "attachment:1_db01.sbd_stage1_scan63.timeseries.png"]
    assert [attachment_bytes(cell, url) for url in urls] == [raw.read_bytes(), calibrated.read_bytes()]
    assert "`plots/raw/db01.sbd_stage1_scan63.timeseries.png`" in cell.source
    assert "`plots/calibrated/db01.sbd_stage1_scan63.timeseries.png`" in cell.source


def test_embedded_notebook_written_elsewhere_is_self_contained(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    touch(work, "plots/caltables/db01.tsys.png", "plots/calibrated/db01.final.radplot.J1848+3219.png",
          "images/db01.images.3C395.png", "summary.md")
    obs.add_gaintable(CalTable(cal_type="tsys", path=str(work / "caltables" / "db01.tsys")), "a_priori")
    obs._state.mark_complete("a_priori", outputs=["tsys"])
    written = write_notebook(obs, str(tmp_path / "elsewhere" / "nb.ipynb"))
    nb = nbformat.read(str(written), as_version=4)
    nbformat.validate(nb)
    urls = image_urls(nb)
    assert len(urls) == 3 and all(url.startswith("attachment:") for url in urls)
    assert not any("../" in c.source for c in nb.cells if c.cell_type == "markdown")
    assert notebook_summary(written)["plots"] == 3 and notebook_summary(written)["embedded"] is True


def test_empty_plot_file_falls_back_to_a_link(tmp_path):
    obs = make_obs(tmp_path)
    work = Path(obs.work_dir)
    empty = work / "plots" / "raw" / "db01.uv_coverage.png"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")
    good, = touch(work, "images/db01.images.3C395.png")
    cell = plots_cell([empty, good], "db01", work, work, embed=True)
    assert IMAGE_LINK.findall(cell.source) == ["plots/raw/db01.uv_coverage.png", "attachment:1_db01.images.3C395.png"]
    assert list(cell["attachments"]) == ["1_db01.images.3C395.png"]


def test_cli_link_option_turns_embedding_off():
    assert build_parser().parse_args(["notebook", "-p", "db01"]).link is False
    assert build_parser().parse_args(["notebook", "-p", "db01", "--link"]).link is True
    campaign = MagicMock()
    campaign.notebook.return_value = ["db01.ipynb"]
    assert _cmd_notebook(campaign, argparse.Namespace(link=True, open=False)) == 0
    campaign.notebook.assert_called_once_with(embed=False)
    campaign.notebook.reset_mock()
    assert _cmd_notebook(campaign, argparse.Namespace(link=False, open=False)) == 0
    campaign.notebook.assert_called_once_with(embed=True)


# ---- linked plots (embed=False)

def test_step_plots_are_linked_between_the_description_and_the_rerun_cell(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    touch(obs.work_dir, "plots/caltables/db01.bandpass_profile.png")
    obs._state.mark_complete("flag_edges", outputs=["edge=1", "measured"])
    nb = build_notebook(obs, embed=False)
    index = cell_index(nb, "### Subband edges")
    plots, rerun = nb.cells[index + 1], nb.cells[index + 2]
    assert plots.cell_type == "markdown"
    assert "![bandpass profile](plots/caltables/db01.bandpass_profile.png)" in plots.source
    assert rerun.cell_type == "code" and rerun.source == "obs.flag.edges(force=FORCE)"
    for cell in nb.cells:
        assert "attachments" not in cell
        assert all(output.output_type == "stream" for output in cell.get("outputs", []))
    assert "base64" not in nbformat.writes(nb)
    assert nb.metadata["vlbipy"]["embedded"] is False


def test_step_without_plots_has_no_plot_cell(tmp_path):
    obs = make_obs(tmp_path)
    obs._state.mark_complete("apply")
    nb = build_notebook(obs)
    index = cell_index(nb, "### Apply")
    assert nb.cells[index + 1].cell_type == "code"


def test_table_plots_match_the_exact_table_name(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    touch(work, "plots/caltables/db01.sbd.delay.png", "plots/caltables/db01.sbd2.phase.png",
          "plots/caltables/db01.sbd2.delay.png", "plots/caltables/db01.gcal.png")
    obs.add_gaintable(CalTable(cal_type="gc", path=str(work / "caltables" / "db01.gcal")), "a_priori")
    obs.add_gaintable(CalTable(cal_type="sbd", path=str(work / "caltables" / "db01.sbd")), "initial_calibration")
    sbd2 = CalTable(cal_type="sbd2", path=str(work / "caltables" / "db01.sbd2"))
    obs.add_gaintable(sbd2, "initial_calibration_sbd2")
    assert [p.name for p in step_plots(obs, "a_priori")] == ["db01.gcal.png"]
    assert [p.name for p in step_plots(obs, "initial_calibration")] == ["db01.sbd.delay.png"]
    assert [p.name for p in step_plots(obs, "initial_calibration_sbd2")] == ["db01.sbd2.phase.png",
                                                                             "db01.sbd2.delay.png"]


def test_solved_and_smoothed_fringe_tables_are_both_shown(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    touch(work, "plots/caltables/db01.mbd2.delay.png", "plots/caltables/db01.mbd2.phase.png",
          "plots/caltables/db01.mbd2_smooth.delay.png", "plots/caltables/db01.mbd.delay.png")
    # only the smoothed copy is applied, so only that one is in the chain
    smoothed = CalTable(cal_type="mbd2", path=str(work / "caltables" / "db01.mbd2_smooth"))
    obs.add_gaintable(smoothed, "fringefit_mbd2")
    assert [p.name for p in step_plots(obs, "fringefit_mbd2")] == ["db01.mbd2.phase.png", "db01.mbd2.delay.png",
                                                                   "db01.mbd2_smooth.delay.png"]
    assert [p.name for p in step_plots(obs, "fringefit")] == ["db01.mbd.delay.png"]


def test_calibrated_sbd_stage_plot_wins_over_the_raw_one(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    touch(work, "plots/raw/db01.sbd_stage1_scan63.timeseries.png", "plots/raw/db01.sbd2_stage1_scan63.timeseries.png",
          "plots/calibrated/db01.sbd_stage1_scan63.timeseries.png")
    assert step_plots(obs, "initial_calibration") == [work / "plots/calibrated/db01.sbd_stage1_scan63.timeseries.png"]
    assert step_plots(obs, "initial_calibration_sbd2") == [work / "plots/raw/db01.sbd2_stage1_scan63.timeseries.png"]


def test_raw_diagnostics_are_listed_once_in_reading_order(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    touch(obs.work_dir, "plots/raw/db01.raw_3C345.spectrum.png", "plots/raw/db01.raw_scan1.spectrum.png",
          "plots/raw/db01.raw_scan1.autocorr.png", "plots/raw/db01.snr_matrix.RR.png", "plots/raw/db01.uv_coverage.png")
    names = [p.name for p in step_plots(obs, "plot_raw")]
    assert names == ["db01.snr_matrix.RR.png", "db01.raw_scan1.autocorr.png", "db01.raw_scan1.spectrum.png",
                     "db01.raw_3C345.spectrum.png"]
    # The per-antenna figures replace the per-scan ones (which antenna_diagnostics deletes).
    for name in ("db01.raw_scan1.autocorr.png", "db01.raw_scan1.spectrum.png"):
        (Path(obs.work_dir) / "plots/raw" / name).unlink()
    touch(obs.work_dir, "plots/raw/db01.raw_antEF.autocorr.png", "plots/raw/db01.raw_antEF.spectrum.png")
    names = [p.name for p in step_plots(obs, "plot_raw")]
    assert names == ["db01.snr_matrix.RR.png", "db01.raw_antEF.autocorr.png", "db01.raw_antEF.spectrum.png",
                     "db01.raw_3C345.spectrum.png"]


def test_calibrated_data_section_lists_final_plots_and_images(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    touch(obs.work_dir, "plots/raw/db01.uv_coverage.png", "plots/calibrated/db01.final.radplot.3C395.png",
          "plots/calibrated/db01.calibrated.radplot.3C345.png", "plots/calibrated/db01.calibrated_3C345.timeseries.png",
          "plots/calibrated/db01.final.lightcurve.png", "images/db01.images.3C395.png")
    nb = build_notebook(obs, embed=False)
    start, end = cell_index(nb, "## Calibrated data"), cell_index(nb, "## Calibration tables")
    section = "\n".join(c.source for c in nb.cells[start:end])
    assert "(plots/raw/db01.uv_coverage.png)" in nb.cells[cell_index(nb, "### UV coverage")].source
    radplot = nb.cells[cell_index(nb, "### Amplitude/phase vs uv distance")].source
    assert "db01.final.radplot.3C395.png" in radplot and "calibrated.radplot" not in radplot
    # no final_* light curve on disk: the group falls back to the calibrators' one
    assert "db01.calibrated_3C345.timeseries.png" in nb.cells[cell_index(nb, "### Light curve per baseline")].source
    assert "db01.calibrated.radplot.3C345.png" in nb.cells[cell_index(nb, "### Calibrators after calibration")].source
    assert "(images/db01.images.3C395.png)" in nb.cells[cell_index(nb, "### Images")].source
    assert "### Calibrated spectrum" not in section
    assert start < cell_index(nb, "### Images") < end


def test_plus_in_a_source_name_is_encoded_in_the_url_only(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    path, = touch(work, "plots/calibrated/db01.calibrated_J1848+3219.spectrum.png")
    text = image_markdown(path, "db01", work)
    assert "`plots/calibrated/db01.calibrated_J1848+3219.spectrum.png`" in text
    assert "(plots/calibrated/db01.calibrated_J1848%2B3219.spectrum.png)" in text
    urls = image_urls(build_notebook(obs, embed=False))
    assert urls == ["plots/calibrated/db01.calibrated_J1848%2B3219.spectrum.png"]


def test_linked_notebook_written_elsewhere_links_back_to_the_plots(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    touch(work, "plots/caltables/db01.tsys.png", "plots/calibrated/db01.final.radplot.J1848+3219.png",
          "images/db01.images.3C395.png")
    obs.add_gaintable(CalTable(cal_type="tsys", path=str(work / "caltables" / "db01.tsys")), "a_priori")
    obs._state.mark_complete("a_priori", outputs=["tsys"])
    target = tmp_path / "elsewhere" / "nb.ipynb"
    written = write_notebook(obs, str(target), embed=False)
    nb = nbformat.read(str(written), as_version=4)
    nbformat.validate(nb)
    urls = image_urls(nb)
    assert len(urls) == 3 and all(url.startswith("../db01/") for url in urls)
    assert all((target.parent / unquote(url)).is_file() for url in urls)
    assert notebook_summary(written)["plots"] == 3 and notebook_summary(written)["embedded"] is False
    assert nb.metadata["vlbipy"]["plots"] == 3


def test_selfcal_step_shows_the_report_and_its_plots(tmp_path):
    obs = make_obs(tmp_path, phasecal="J1848+3219", fringe_finder="3C345")
    work = Path(obs.work_dir)
    touch(work, "selfcal/db01.3C345.amp.png", "images/db01.images.3C345.png")
    rounds = [{"solint": "30min", "chisq_before": 120844.5, "chisq_after": 79066.1, "improvement": 0.3457,
               "accepted": True},
              {"solint": "10min", "chisq_before": 61864.3, "chisq_after": 61862.3, "improvement": 3.2e-05,
               "accepted": False}]
    report = {"source": "3C345", "model_rchisq": 2.1689, "ladder": ["30min", "10min"], "rounds": rounds}
    (work / "selfcal" / "db01.3C345.json").write_text(json.dumps(report), encoding="utf-8")
    selfamp = CalTable(cal_type="selfamp_3C345", path=str(work / "selfcal" / "db01.3C345.amp.G"))
    obs.add_gaintable(selfamp, "selfcal_3C345")
    obs._state.mark_complete("selfcal_3C345")
    nb = build_notebook(obs, embed=False)
    index = cell_index(nb, "### Self-calibration of 3C345")
    description = nb.cells[index].source
    assert "Phase ladder: 30min, 10min; model reduced chi² 2.17." in description
    assert "| 30min | 120,844.5 | 79,066.1 | -34.6% | accepted |" in description
    assert "| 10min | 61,864.3 | 61,862.3 | -0.0% | rejected |" in description
    # the image grid belongs to the imaging step and the Calibrated data section, not to self-cal
    assert IMAGE_LINK.findall(nb.cells[index + 1].source) == ["selfcal/db01.3C345.amp.png"]
    assert step_plots(obs, "clean_3C345") == [work / "images/db01.images.3C345.png"]
    assert selfcal_report_markdown(work, "db01", "J1848+3219") == ""


def test_notebook_without_plots_still_builds_and_carries_the_summary(tmp_path):
    obs = make_obs(tmp_path)
    nb = build_notebook(obs)
    nbformat.validate(nb)
    assert nb.metadata["vlbipy"]["plots"] == 0
    setup = next(c for c in nb.cells if c.cell_type == "code")
    assert [(o.output_type, o.name, o.text) for o in setup.outputs] == [("stream", "stdout", obs.summary() + "\n")]
    assert "No calibrated-data plots exist yet" in nb.cells[cell_index(nb, "## Calibrated data") + 1].source
