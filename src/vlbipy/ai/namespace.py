"""``obs.ai``: the AI-mode steps of the pipeline (pre-diagnostic, Tsys repair, distribution flagging, annotation).

Every step writes what it found to ``<work_dir>/ai/``: a JSON report (the findings, the rules that fired and the
actions taken), the visibility cubes it computed, and the labelled feature rows of
:mod:`vlbipy.ai.features` that the small model of :mod:`vlbipy.ai.model` is trained on.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


from ..logging_utils import get_logger, warnings
from ..namespaces import Namespace
from . import antenna, distributions, features, knowledge, tsys, visstats

logger = get_logger()

#: Flag version holding the a-priori flags only, which the pre-diagnostic (and any later look at the raw
#: data "as it came") reads instead of the evolving FLAG column.
APRIORI_FLAGS = "ai_apriori"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _tsys_log_gain(syscal: dict, antenna_name: str, scans) -> np.ndarray:
    """``-0.5 log Tsys`` of one antenna per scan (median over its subbands and hands), median-subtracted."""
    names = list(syscal["antennas"])
    if antenna_name not in names:
        return np.full(len(scans), np.nan)
    rows = syscal["antenna"] == names.index(antenna_name)
    values = np.full(len(scans), np.nan)
    for i, scan in enumerate(scans):
        t = syscal["tsys"][rows & (syscal["scan"] == scan)]
        t = t[np.isfinite(t) & (t > 0)]
        if t.size:
            values[i] = -0.5 * np.log(np.median(t))
    if np.isfinite(values).any():
        values -= np.nanmedian(values)
    return values


class AINamespace(Namespace):
    """Data-driven diagnostics and decisions (see :mod:`vlbipy.ai`)."""

    def __call__(self, *, force: bool = False) -> dict:
        """Run the pre-diagnostic (the step that needs nothing but the imported, a-priori-flagged data)."""
        return self.prediagnose(force=force)

    # ------------------------------------------------------------------ helpers
    @property
    def _cfg(self) -> dict:
        return dict(self._obs.config.get("ai", {}))

    @property
    def enabled(self) -> bool:
        """``[ai].enabled`` and a backend with a measurement set on disk."""
        return bool(self._cfg.get("enabled", True)) and bool(getattr(self._backend, "requires_data_files", False)) \
            and hasattr(self._backend, "ms_path")

    @property
    def dir(self) -> Path:
        """``<work_dir>/ai``."""
        path = Path(self._obs.work_dir) / "ai"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ms(self) -> str:
        return str(self._backend.ms_path(self._code))

    def _calibrators(self) -> list[str]:
        obs = self._obs
        names = [s.name for s in obs.sources.phase_calibrators + obs.sources.fringe_finders]
        present = set(obs.metadata.source_names) if obs.metadata else set(names)
        return [n for n in dict.fromkeys(names) if n in present]

    def _write(self, name: str, report: dict) -> Path:
        path = self.dir / f"{self._code}.{name}.json"
        path.write_text(json.dumps(report, indent=1, default=_json_default))
        return path

    def load_report(self, name: str) -> dict:
        """A report written earlier (``{}`` if there is none)."""
        path = self.dir / f"{self._code}.{name}.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    # ------------------------------------------------------------------ pre-diagnostic
    def prediagnose(self, *, force: bool = False, binsec: Optional[float] = None) -> dict:
        """What the raw data say about every antenna, before any calibration (``ai_prediagnostic``).

        Run right after the a-priori flagging. Saves those flags as the version ``ai_apriori``, reduces the raw
        data of every field to the statistics cube (:mod:`~vlbipy.ai.visstats`), and reports per antenna:

        * the gain over time on each calibrator (sensitivity drops, scan-to-scan scatter), from the
          phase-independent amplitude, and whether two sources agree (instrumental) or not (structure);
        * the subband and hand levels of the raw correlation coefficients;
        * every Tsys series classified (:func:`vlbipy.ai.tsys.assess`) and checked against the raw amplitudes:
          whether a subband excess, a trend or the scan-to-scan fluctuations of the Tsys are real.

        The findings are labelled by :mod:`~vlbipy.ai.knowledge` and the expected consequences are logged.
        """
        obs = self._obs
        step = "ai_prediagnostic"
        if not self.enabled or not obs._state.should_run(step, force=force):
            return self.load_report("prediagnostic")
        from ..solvers.flagmanager import list_versions, save_version
        ms = self._ms()
        # The version must hold the a-priori flags only: saved right after flag.apriori, and reused when the
        # pre-diagnostic is re-run later on (by then FLAG carries every flag the pipeline added).
        exists = any(name == APRIORI_FLAGS for name, _ in list_versions(ms))
        later = any(obs._state.status(name) == "done" for name in ("flag_quack", "flag_tfcrop", "initial_calibration"))
        if not exists or not later:
            if later:
                warnings.warn(f"{self._code}: no '{APRIORI_FLAGS}' flag version and the pipeline has flagged since the "
                              "a-priori flags: the pre-diagnostic reads the current flags")
            save_version(ms, APRIORI_FLAGS, "a-priori flags only: the AI-mode pre-diagnostic reads the raw data "
                                            "with these")
        cfg = self._cfg
        cube = visstats.visibility_stats(ms, column="DATA", binsec=float(binsec or cfg.get("binsec", 30.0)),
                                         flag_version=APRIORI_FLAGS)
        visstats.save(cube, self.dir / f"{self._code}.raw_cube.npz")
        syscal = tsys.read_syscal(ms)
        report = self._diagnose_raw(cube, syscal)
        report["tsys"] = self._assess_tsys(cube, syscal)
        self._write("prediagnostic", report)
        rows = features.prediagnostic_rows(self._code, report)
        features.write_rows(self.dir / f"{self._code}.features.jsonl", rows, stage="prediagnostic")
        self._log_prediagnostic(report)
        try:
            from .plots import plot_prediagnostic
            plot_prediagnostic(report, Path(obs.work_dir) / "plots" / "ai", self._code)
        except Exception as exc:  # noqa: BLE001 - a plot never stops the pipeline
            logger.warning("ai: pre-diagnostic plot failed ({})", exc)
        obs._state.mark_complete(step, outputs=[str(self.dir / f"{self._code}.prediagnostic.json")])
        return report

    def _diagnose_raw(self, cube: dict, syscal: Optional[dict] = None) -> dict:
        names = [str(n) for n in cube["antennas"]]
        calibrators = [c for c in self._calibrators() if c in list(cube["field_names"])]
        gains = antenna.gains_vs_time(cube, fields=calibrators, coherent=False)
        per_antenna: dict = {n: {"fields": {}} for n in names}
        scatters = {}
        for field, g in gains.items():
            for a, name in enumerate(names):
                x = np.nanmean(g["x"][:, a, :], axis=1)
                ok = np.isfinite(x)
                if ok.sum() < 3:
                    continue
                stats = features.series_stats(g["time"][ok], x[ok])
                if syscal is not None:
                    # What the Tsys predicts for the same scans (-0.5 log Tsys, all subbands and hands), and the
                    # trend left once it is applied: the part the a-priori calibration cannot take out.
                    predicted = _tsys_log_gain(syscal, name, g["scan"][ok])
                    if np.isfinite(predicted).sum() >= 3:
                        tsys_stats = features.series_stats(g["time"][ok][np.isfinite(predicted)],
                                                           predicted[np.isfinite(predicted)])
                        resid = x[ok] - np.where(np.isfinite(predicted), predicted, 0.0)
                        stats["tsys_trend"] = tsys_stats["trend"]
                        stats["residual_trend"] = features.series_stats(g["time"][ok], resid)["trend"]
                per_antenna[name]["fields"][field] = stats
                if np.isfinite(stats["scan_scatter"]):
                    scatters.setdefault(field, []).append(stats["scan_scatter"])
        for name in names:
            fields = per_antenna[name]["fields"]
            trends = [f["trend"] for f in fields.values() if f.get("n", 0) >= 5]
            per_antenna[name]["trend_consistent_fields"] = int(sum(1 for t in trends if abs(t) > 0.1 and
                                                                   np.sign(t) == np.sign(trends[0])) >= 2)
        sb = antenna.subband_offsets(cube, fields=calibrators, coherent=False)
        hr = antenna.hand_ratios(cube, fields=calibrators, coherent=False)
        return {"project": self._code, "calibrators": calibrators, "antennas": names,
                "spws": [int(s) for s in sb["spws"]], "rho1": cube["rho1"].tolist(),
                "gains": {f: {"time": g["time"].tolist(), "scan": g["scan"].tolist(), "x": g["x"].tolist()}
                          for f, g in gains.items()},
                "array_scan_scatter": {f: float(np.nanmedian(v)) for f, v in scatters.items()},
                "per_antenna": per_antenna,
                "subband_offsets": sb["x"].tolist(), "hand_ratios": hr["x"].tolist()}

    def _assess_tsys(self, cube: dict, syscal: Optional[dict] = None) -> dict:
        ms = self._ms()
        syscal = syscal if syscal is not None else tsys.read_syscal(ms)
        with open(self.dir / f"{self._code}.syscal.pkl", "wb") as handle:
            pickle.dump(syscal, handle)
        names = [str(n) for n in cube["antennas"]]
        calibrators = [c for c in self._calibrators() if c in list(cube["field_names"])]
        findings = [f for f in tsys.assess(syscal) if f["n"] > 0]
        sb = antenna.subband_offsets(cube, fields=calibrators, coherent=False)
        tsys.check_against_raw(findings, sb, names)
        checks = {}
        reference = calibrators[0] if calibrators else None
        if reference:
            g = antenna.gains_vs_time(cube, fields=[reference], coherent=False, combine_spw=False)[reference]
            spws = [int(s) for s in np.unique(cube["spw"])]
            for f in findings:
                if f["spw"] not in spws:
                    continue
                k = spws.index(f["spw"])
                hand = 0 if f["hand"] == "R" else 1
                check = tsys.check_trend_against_raw(syscal, {"scan": g["scan"], "x": g["x"][:, :, :, k]},
                                                     f["antenna"], hand=hand, spw=f["spw"])
                checks[(f["antenna"], f["spw"], hand)] = check
                f.update({k2: check[k2] for k2 in ("slope", "corr", "fast_slope", "fast_corr", "fast_rms")})
        for f in findings:
            f["rules"] = knowledge.label(f, "tsys")
        decisions = tsys.plan(syscal, findings, checks)
        with open(self.dir / f"{self._code}.tsys_plan.pkl", "wb") as handle:
            pickle.dump({"findings": findings, "decisions": decisions}, handle)
        return {"reference_field": reference, "findings": findings,
                "decisions": {f"{k[0]}:{k[1]}:{'RL'[k[2]]}": v for k, v in decisions.items()}}

    def _log_prediagnostic(self, report: dict) -> None:
        for name, info in report["per_antenna"].items():
            for field, stats in info["fields"].items():
                array = report["array_scan_scatter"].get(field, 0.01)
                finding = dict(stats, array_scan_scatter=array, trend_consistent_fields=info["trend_consistent_fields"])
                for fired in knowledge.label(finding, "prediagnostic"):
                    if fired["label"] == "trend_explained_by_tsys":
                        logger.info("ai: {} on {}: raw amplitude {:+.0%} over the run, followed by its Tsys", name,
                                    field, np.expm1(stats["trend"]))
                        continue
                    detail = []
                    if np.isfinite(stats.get("scan_scatter", np.nan)):
                        detail.append(f"scan-to-scan scatter {100 * stats['scan_scatter']:.1f}% vs array "
                                      f"{100 * array:.1f}%")
                    detail.append(f"raw amplitude {np.expm1(stats['trend']):+.0%} over the run")
                    if np.isfinite(stats.get("residual_trend", np.nan)):
                        detail.append(f"{np.expm1(stats['residual_trend']):+.0%} after its Tsys")
                    warnings.warn(f"{self._code}: AI pre-diagnostic: {name} on {field}: {fired['label']} "
                                  f"({', '.join(detail)}) -> {fired['action']}")
        decisions = report["tsys"]["decisions"]
        changed = {k: v for k, v in decisions.items() if v["action"] in ("substitute", "rescale")}
        for key, decision in changed.items():
            logger.info("ai: Tsys {} -> {} ({})", key, decision["action"], decision["reason"])

    # ------------------------------------------------------------------ Tsys repair (a-priori step)
    def repair_tsys(self, table) -> Optional[dict]:
        """Repair the Tsys table with the plan of the pre-diagnostic (None when there is no plan)."""
        path = self.dir / f"{self._code}.tsys_plan.pkl"
        if not self.enabled or not self._cfg.get("repair_tsys", True) or not path.is_file():
            return None
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        counts = tsys.repair_table(table.path, saved["decisions"],
                                   despike_sigma=float(self._obs.config.get("calibration", {})
                                                       .get("tsys_outlier_sigma", 6.0)))
        self._write("tsys_repair", {"counts": counts,
                                    "decisions": {f"{k[0]}:{k[1]}:{'RL'[k[2]]}": v
                                                  for k, v in saved["decisions"].items()}})
        return counts

    # ------------------------------------------------------------------ calibrated data
    def calibrated_cube(self, *, fields: Optional[list] = None, binsec: Optional[float] = None,
                        name: str = "calibrated_cube") -> dict:
        """The statistics cube of the corrected data (the deferred chain applied on the fly)."""
        cube = visstats.visibility_stats(self._ms(), column="CORRECTED_DATA", fields=fields,
                                         binsec=float(binsec or self._cfg.get("calibrated_binsec", 10.0)))
        visstats.save(cube, self.dir / f"{self._code}.{name}.npz")
        return cube

    def flag_distribution(self, *, force: bool = False, step: str = "ai_flag_distribution",
                          fields: Optional[list] = None, apply: Optional[bool] = None) -> dict:
        """Flag what departs from the noise distribution in the calibrated data (:mod:`~vlbipy.ai.distributions`).

        Every field is tested (the calibrators by default plus, with ``[ai].flag_targets``, the targets: they are
        noise-dominated, which the test handles - Rayleigh amplitudes, uniform phases). The flags go in as
        ``flagdata`` list commands, so they are reversible from the flag versions; a report lists every finding.
        """
        obs = self._obs
        if not self.enabled or not obs._state.should_run(step, force=force):
            return self.load_report(step)
        cfg = self._cfg
        if fields is None:
            fields = self._calibrators()
            if cfg.get("flag_targets", True):
                present = set(obs.metadata.source_names) if obs.metadata else set()
                fields += [s.name for s in obs.sources.targets + obs.sources.check_sources if s.name in present]
        cube = self.calibrated_cube(fields=fields, name=f"{step}_cube")
        found = distributions.assess(cube, p_bin=float(cfg.get("p_outlier", distributions.P_BIN)))
        departures = distributions.amplitude_departures(cube, fields=self._calibrators())
        commands = distributions.flag_commands(found, cube, departures=departures)
        names = [str(n) for n in cube["antennas"]]
        channels = []
        try:
            refant = obs.refant[0] if isinstance(obs.refant, (list, tuple)) else obs.refant
            channels = distributions.channel_outliers(self._ms(), fields=self._calibrators(),
                                                      refant=names.index(str(refant).split(",")[0]))
        except Exception as exc:  # noqa: BLE001 - the channel test is an extra, never a blocker
            logger.warning("ai: channel test skipped ({})", exc)
        commands += [f"mode='manual' spw='{c['spw']}:{c['channel']}' reason='AI_CHANNEL'" for c in channels]
        report = {"fields": fields, "n_cells": int(cube["field"].size), "rho1": cube["rho1"].tolist(),
                  "outlier_bins": len(found["bins"]), "excess_scans": found["scans"], "subbands": found["spws"],
                  "departures": departures, "channels": channels,
                  "baselines": [dict(b, baseline=f"{names[b['ant1']]}-{names[b['ant2']]}") for b in found["baselines"]],
                  "commands": len(commands)}
        list_file = self.dir / f"{self._code}.{step}.flags.txt"
        list_file.write_text("\n".join(commands) + ("\n" if commands else ""))
        if commands and (apply if apply is not None else cfg.get("apply_flags", True)):
            from ..solvers.flagmanager import save_version
            save_version(self._ms(), f"before_{step}", "before the AI distribution flags")
            report["flagged_fraction"] = self._backend.flag.run(self._code, "from_file", flagfile=str(list_file))
            report["applied"] = True
            logger.info("ai: {} distribution flag command(s) applied from {} ({} outlier bins, {} scans with excess "
                        "scatter, {} subband departures, {} amplitude departures)", len(commands), list_file.name,
                        len(found["bins"]), len(found["scans"]), len(found["spws"]), len(departures))
        self._write(step, report)
        features.write_rows(self.dir / f"{self._code}.features.jsonl",
                            features.calibrated_rows(self._code, report, names), stage=step)
        obs._state.mark_complete(step, outputs=[str(list_file)])
        return report

    def postdiagnose(self, *, fields: Optional[list] = None, name: str = "postdiagnostic") -> dict:
        """Antenna-based residuals of the calibrated calibrators: time, subband and hand levels (report only)."""
        fields = fields or self._calibrators()
        cube = self.calibrated_cube(fields=fields, name=f"{name}_cube", binsec=30.0)
        names = [str(n) for n in cube["antennas"]]
        gains = antenna.gains_vs_time(cube, fields=fields, coherent=True)
        sb = antenna.subband_offsets(cube, fields=fields, coherent=True)
        hr = antenna.hand_ratios(cube, fields=fields, coherent=True)
        per_antenna = {}
        for field, g in gains.items():
            for a, ant in enumerate(names):
                x = np.nanmean(g["x"][:, a, :], axis=1)
                ok = np.isfinite(x)
                if ok.sum() >= 3:
                    per_antenna.setdefault(ant, {})[field] = features.series_stats(g["time"][ok], x[ok])
        report = {"fields": fields, "antennas": names, "spws": [int(s) for s in sb["spws"]],
                  "per_antenna": per_antenna, "subband_offsets": sb["x"].tolist(), "hand_ratios": hr["x"].tolist(),
                  "gains": {f: {"time": g["time"].tolist(), "scan": g["scan"].tolist(), "x": g["x"].tolist()}
                            for f, g in gains.items()}}
        self._write(name, report)
        for a, ant in enumerate(names):
            ratios = np.array(hr["x"])[:, a]
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size and np.max(np.abs(ratios)) > 0.05:
                logger.warning("ai: {} calibrated RR/LL still differ by up to {:+.0%} on the calibrators", ant,
                               float(np.exp(ratios[np.argmax(np.abs(ratios))]) - 1))
        try:
            from .plots import plot_postdiagnostic
            plot_postdiagnostic(report, Path(self._obs.work_dir) / "plots" / "ai", self._code, name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai: post-diagnostic plot failed ({})", exc)
        return report
