"""Feature rows: what the small model sees of every finding, and the label it learns to predict.

A row is one decision unit - an antenna on a calibrator (pre-diagnostic), one Tsys series (antenna, subband,
hand), one baseline/hand of calibrated data - described by a fixed set of dimensionless numbers that do not
depend on the array, the frequency or the flux density of the source, so that what is learnt on one experiment
carries over to the next. Each row also carries the labels the expert rules gave it (:mod:`~vlbipy.ai.knowledge`)
and, when a human reviewed it, the human's label, which wins (:func:`annotate`).

Rows are appended to ``<work_dir>/ai/<code>.features.jsonl``; the annotated sets that ship with vlbipy live in
``vlbipy/data/ai/`` (:func:`vlbipy.ai.model.training_files`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import knowledge

#: Feature names per stage, in the order the model reads them.
FEATURES = {
    "tsys": ("log_level_vs_antenna", "scatter", "scatter_ratio", "unphysical_fraction", "trend", "nominal",
             "slope", "corr", "fast_slope", "fast_corr", "fast_rms", "raw_offset", "expected_offset"),
    "prediagnostic": ("scan_scatter", "scatter_vs_array", "trend", "max_drop", "max_rise", "range",
                      "trend_consistent_fields", "nominal_tsys", "tsys_trend", "residual_trend"),
    "calibrated": ("chi2r", "frac_out", "log_snr", "ks"),
}


def series_stats(time: np.ndarray, x: np.ndarray) -> dict:
    """Shape of one antenna's log-gain series over a field's scans.

    Returns ``n``, ``scan_scatter`` (robust scatter about a 5-scan running median), ``trend`` (median of the last
    15% minus the first 15%), ``max_drop`` / ``max_rise`` (extremes of the 5-scan running median about the
    series median) and ``range`` - all in natural-log units (~fractions for small values).
    """
    from scipy.ndimage import median_filter
    order = np.argsort(time)
    x = np.asarray(x, dtype=float)[order]
    n = x.size
    if n < 3:
        return {"n": int(n), "scan_scatter": np.nan, "trend": np.nan, "max_drop": np.nan, "max_rise": np.nan,
                "range": np.nan}
    smooth = median_filter(x, size=min(5, n), mode="nearest")
    resid = x - smooth
    # Scan-to-scan scatter needs scans to compare: a 5-scan median of 5 scans is the scans themselves.
    scatter = float(1.4826 * np.median(np.abs(resid - np.median(resid))) * 1.25) if n >= 7 else np.nan
    k = max(1, int(round(0.15 * n)))
    level = float(np.median(smooth))
    return {"n": int(n), "scan_scatter": scatter, "trend": float(np.median(x[-k:]) - np.median(x[:k])),
            "max_drop": float(np.min(smooth) - level), "max_rise": float(np.max(smooth) - level),
            "range": float(np.max(smooth) - np.min(smooth))}


def _vector(values: dict, names: tuple) -> list:
    out = []
    for name in names:
        value = values.get(name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        out.append(value if np.isfinite(value) else None)
    return out


def tsys_rows(code: str, tsys_report: dict) -> list[dict]:
    """One row per Tsys series of a pre-diagnostic report."""
    rows = []
    for f in tsys_report.get("findings", []):
        level = f.get("level_vs_antenna")
        values = {"log_level_vs_antenna": np.log(level) if level and level > 0 else None,
                  "scatter": f.get("scatter"),
                  "scatter_ratio": (f["scatter"] / max(f["partner_scatter"], 1e-3)
                                    if f.get("scatter") is not None and f.get("partner_scatter") is not None
                                    and np.isfinite(f.get("scatter", np.nan)) and np.isfinite(f.get("partner_scatter", np.nan))
                                    else None),
                  "unphysical_fraction": f.get("unphysical", 0) / max(f.get("n", 1) + f.get("unphysical", 0), 1),
                  "trend": f.get("trend"), "nominal": float("nominal" in f.get("labels", ())),
                  **{k: f.get(k) for k in ("slope", "corr", "fast_slope", "fast_corr", "fast_rms", "raw_offset",
                                           "expected_offset")}}
        labels = [r["label"] for r in f.get("rules", knowledge.label(f, "tsys"))]
        rows.append({"experiment": code, "stage": "tsys", "unit": f"{f['antenna']}:{f['spw']}:{f['hand']}",
                     "x": _vector(values, FEATURES["tsys"]), "rule_labels": labels,
                     "label": _primary(labels, ("noisy_tsys_hand", "spurious_high_tsys", "real_high_tsys",
                                                "tsys_undercorrects", "tsys_variation_real",
                                                "tsys_fluctuation_not_in_data", "nominal_tsys", "unphysical_tsys"))})
    return rows


def prediagnostic_rows(code: str, report: dict) -> list[dict]:
    """One row per (antenna, calibrator) of the raw-data pre-diagnostic, plus the Tsys rows."""
    rows = []
    nominal = {f["antenna"] for f in report.get("tsys", {}).get("findings", []) if "nominal" in f.get("labels", ())}
    for name, info in report.get("per_antenna", {}).items():
        for field, stats in info.get("fields", {}).items():
            array = report.get("array_scan_scatter", {}).get(field, 0.01) or 0.01
            values = dict(stats, scatter_vs_array=(stats["scan_scatter"] / max(array, 1e-3)
                                                   if np.isfinite(stats.get("scan_scatter", np.nan)) else np.nan),
                          trend_consistent_fields=info.get("trend_consistent_fields", 0),
                          nominal_tsys=float(name in nominal))
            finding = dict(values, array_scan_scatter=array)
            labels = [r["label"] for r in knowledge.label(finding, "prediagnostic")]
            rows.append({"experiment": code, "stage": "prediagnostic", "unit": f"{name}@{field}",
                         "x": _vector(values, FEATURES["prediagnostic"]), "rule_labels": labels,
                         "label": _primary(labels, ("antenna_amplitude_scatter", "antenna_sensitivity_trend",
                                                    "trend_explained_by_tsys"))})
    rows += tsys_rows(code, report.get("tsys", {}))
    return rows


def calibrated_rows(code: str, report: dict, names: list[str]) -> list[dict]:
    """One row per baseline/hand of a distribution report (is this baseline's noise distribution healthy?)."""
    rows = []
    for b in report.get("baselines", []):
        values = {"chi2r": b["chi2r"], "frac_out": b["frac_out"],
                  "log_snr": np.log(b["snr"]) if b.get("snr") and b["snr"] > 0 else None, "ks": b["ks"]}
        label = "bad_distribution" if (b["chi2r"] > 1.5 or b["frac_out"] > 0.01 or b["ks"] > 0.1) else "healthy"
        rows.append({"experiment": code, "stage": "calibrated",
                     "unit": f"{b.get('baseline', f'{names[b['ant1']]}-{names[b['ant2']]}')}:{'RL'[b['hand']]}",
                     "x": _vector(values, FEATURES["calibrated"]), "rule_labels": [label], "label": label})
    return rows


def _primary(labels: list[str], priority: tuple) -> str:
    """The one label a row is trained on: the first of ``priority`` among ``labels``, else ``"ok"``."""
    for name in priority:
        if name in labels:
            return name
    return "ok"


def write_rows(path, rows: list[dict], *, stage: str) -> Path:
    """Replace the rows of ``stage`` (the ``stage`` key of each row, or the call's) in a JSONL file."""
    path = Path(path)
    kept = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("written_by") != stage:
                kept.append(row)
    for row in rows:
        row["written_by"] = stage
    path.write_text("".join(json.dumps(r, default=float) + "\n" for r in kept + rows))
    return path


def read_rows(path) -> list[dict]:
    """The rows of a JSONL file."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def annotate(rows: list[dict], corrections: dict[str, str], *, reviewer: str = "human") -> list[dict]:
    """Overwrite the label of the rows whose ``unit`` (or ``stage/unit``) is in ``corrections``.

    The human label wins over the rules': this is how a review ("SR RCP Tsys is noise, use LCP") becomes
    training data even where no rule fired, or where one fired wrongly.
    """
    for row in rows:
        for key in (f"{row['stage']}/{row['unit']}", row["unit"]):
            if key in corrections:
                row["label"] = corrections[key]
                row["reviewed_by"] = reviewer
                break
    return rows


def apply_review(rows: list[dict], review: dict) -> list[dict]:
    """Apply a review file (``{"reviewer", "labels": {"<stage>/<unit pattern>": label}}``) to feature rows.

    Unit patterns may use ``*`` for any subband or hand (``"tsys/UR:*:R"``) and ``@*`` for any field.
    """
    import fnmatch
    corrections = review.get("labels", {})
    for row in rows:
        key = f"{row['stage']}/{row['unit']}"
        for pattern, label in corrections.items():
            if fnmatch.fnmatchcase(key, pattern):
                row["label"] = label
                row["reviewed_by"] = review.get("reviewer", "human")
    return rows


def matrix(rows: list[dict], stage: str) -> tuple[np.ndarray, list[str], list[dict]]:
    """``X`` (n, nfeat; nan for missing), the labels, and the rows of one stage."""
    chosen = [r for r in rows if r["stage"] == stage]
    X = np.array([[np.nan if v is None else float(v) for v in r["x"]] for r in chosen], dtype=float) \
        if chosen else np.zeros((0, len(FEATURES[stage])))
    return X, [r["label"] for r in chosen], chosen


def stage_of(row_stage: Optional[str]) -> str:
    """Feature set of a row stage (``ai_flag_distribution`` rows use the ``calibrated`` features)."""
    return row_stage if row_stage in FEATURES else "calibrated"
