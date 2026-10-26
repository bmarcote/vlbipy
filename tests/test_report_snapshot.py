"""Optional report-only integration against cached products, never against measurement-set visibilities."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from vlbipy import VLBIObs
from vlbipy.dashboard import build_dashboard


@pytest.mark.skipif(not os.environ.get("VLBIPY_REPORT_SNAPSHOT"), reason="no cached report snapshot supplied")
def test_cached_reports_rebuild_without_changing_the_originals(tmp_path):
    """Reconstruct reports in an isolated copy, with no main MS, and verify all original caches remain unchanged."""
    root = Path(os.environ["VLBIPY_REPORT_SNAPSHOT"])
    epochs = [root] if (root / ".metadata.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / ".metadata.json").is_file())
    assert epochs
    for source in epochs:
        inputs = [path for path in source.glob(".*.json") if path.is_file()]
        inputs.extend(path for path in (source / "html").glob("*") if path.is_file())
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        snapshot = tmp_path / source.name
        snapshot.mkdir()
        for path in source.glob(".*.json"):
            if path.is_file():
                shutil.copy2(path, snapshot / path.name)
        project = json.loads((snapshot / ".project.json").read_text())
        project["work_dir"] = str(snapshot)
        project["config"].setdefault("global", {})["work_dir"] = str(snapshot)
        (snapshot / ".project.json").write_text(json.dumps(project))
        for name in ("plots", "images", "selfcal", "calibrated_data", "caltables"):
            if (source / name).is_dir():
                (snapshot / name).symlink_to(source / name, target_is_directory=True)
        campaign = VLBIObs.load(snapshot)
        assert len(campaign.observations) == 1
        obs = campaign.observations[0]
        assert Path(obs.work_dir).resolve() == snapshot.resolve()
        assert not (snapshot / "data").exists()
        index = build_dashboard(obs)
        html = index.read_text()
        assert 'class="skymap"' in html and 'class="src target"' in html
        for page in ("index.html", "diagnostics.html", "calibration.html", "final_data.html"):
            assert (snapshot / "html" / page).is_file()
        assert (snapshot / "summary.md").is_file()
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        assert before == after
