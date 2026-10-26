"""Multi-epoch combination: a joint calibrator model, per-epoch refinement, combined images.

Every epoch of a campaign is calibrated, self-calibrated and imaged on its own
first. What one epoch cannot give is a good model of the phase calibrator: its
uv coverage is that of a single track, with whatever antennas happened to work
that day. :func:`combine_epochs` uses all of them together:

1. the calibrated data of each phase calibrator are concatenated across the
   epochs (each brought to a common flux level, since calibrators vary) and
   modelled jointly with difmapy (:func:`vlbipy.backends.difmap.joint_model`);
2. each epoch of that calibrator is self-calibrated again against the joint
   model, scaled back to the epoch's own flux density; the resulting gains go
   into that epoch's calibration chain as one more table (``joint_<source>``),
   applied to the same fields as the calibrator's own self-calibration. The
   phase ladder first, then the station amplitudes with ``bayes_gscale``
   against the same fixed model: a calibrator that one track models poorly
   gives amplitude corrections that absorb its structure, and the joint model
   puts every epoch on the same one. Only stations clearly off are corrected
   (``[selfcal].prior_sigma``), and how much each station depends on the model
   is the uncertainty the epoch's own amplitude step measured (the joint model
   is not rebuilt without each station); ``[selfcal].joint_amplitude = false``
   keeps the epochs' own amplitudes;
3. every epoch is re-split and re-imaged with the refined calibration, and its
   plots of the calibrated data are redrawn;
4. the epochs of every source are concatenated and imaged together, which for
   the target is the deep image of the campaign.

Products go to ``<campaign directory>/combined`` (``data/``, ``selfcal/``,
``images/`` and ``<name>.campaign.json``, the record of all of the above with
the image statistics of every epoch before and after).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .logging_utils import get_logger, warnings
from .models import CalTable

logger = get_logger()


def campaign_name(codes: list[str]) -> str:
    """Name of a campaign: what its project codes have in common (``V589A..D`` -> ``V589``)."""
    prefix = os.path.commonprefix([str(code) for code in codes]).rstrip("_-. ")
    return prefix if len(prefix) >= 2 else "+".join(codes)


def campaign_dir(observations: list) -> Path:
    """Directory of the combined products: ``combined/`` next to the per-epoch working directories."""
    return Path(observations[0].work_dir).resolve().parent / "combined"


def fits_statistics(path: str) -> dict:
    """Peak and noise of a FITS image: ``peak``, ``rms`` (MAD-based, so the source does not count) and their ratio."""
    from astropy.io import fits
    with fits.open(str(path)) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=float))
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"peak": float("nan"), "rms": float("nan"), "dynamic_range": 0.0}
    rms = 1.4826 * float(np.median(np.abs(finite - np.median(finite))))
    peak = float(finite.max())
    return {"peak": peak, "rms": rms, "dynamic_range": peak / rms if rms > 0 else 0.0}


def epoch_image_statistics(obs) -> dict:
    """``{source: {robust: statistics}}`` of the images an epoch has on disk."""
    image_dir = Path(obs.work_dir) / "images"
    stats: dict = {}
    for name in (obs.metadata.source_names if obs.metadata else obs.sources.names):
        for path in sorted(image_dir.glob(f"{obs.project_code}_{name}.robust*.fits")):
            robust = path.name[len(f"{obs.project_code}_{name}.robust"):-len(".fits")]
            try:
                stats.setdefault(name, {})[robust] = fits_statistics(str(path))
            except Exception as exc:  # noqa: BLE001 - statistics are a report, not a product
                logger.debug("could not measure {}: {}", path, exc)
    return stats


def _epoch_flux(obs, source: str) -> Optional[float]:
    """Flux density of ``source`` in this epoch, from the fitted model of its own self-calibration."""
    path = Path(obs.work_dir) / "selfcal" / f"{obs.project_code}.{source}.json"
    try:
        images = json.loads(path.read_text()).get("images", {})
    except (OSError, json.JSONDecodeError):
        return None
    fluxes = [float(info["model_flux"]) for info in images.values() if float(info.get("model_flux", 0.0)) > 0]
    return float(np.median(fluxes)) if fluxes else None


def _own_model_sigma(obs, source: str) -> dict:
    """Station -> per-IF log-gain uncertainty that the epoch's own amplitude step found for ``source``.

    That step rebuilt its models without each station in turn, so it measured how much each station's gain
    depends on the source model (large for one whose baselines no other station constrains). Against the fixed
    joint model nothing measures it, so :func:`refine_calibrator` carries it over. Empty when there is no report.
    """
    path = Path(obs.work_dir) / "selfcal" / f"{obs.project_code}.{source}.amp.json"
    try:
        report = json.loads(path.read_text())
        sigma, names = report["log_gain"]["sigma"], report["antennas"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if sigma and isinstance(sigma[0], list):
        return {name: [row[a] for row in sigma] for a, name in enumerate(names)}
    return dict(zip(names, sigma))


def _epochs_changed_since(observations: list, report_path: Path) -> list[str]:
    """Project codes whose calibration chain was written after the campaign report.

    The combination itself is the last thing to touch an epoch's chain (it adds
    the ``joint_<source>`` tables) before it writes the report, so a chain newer
    than the report means the epoch was calibrated again since: its combined
    products describe data that no longer exist.
    """
    written = report_path.stat().st_mtime
    changed = []
    for obs in observations:
        chain = obs._caltables_path
        if chain is not None and chain.is_file() and chain.stat().st_mtime > written:
            changed.append(obs.project_code)
    return changed


def joint_model_settings(config: dict) -> dict:
    """Deep calibrator fitting limits, independent of the target's single-component starting policy."""
    from .backends.difmap import imaging_settings
    settings = imaging_settings(config.get("imaging", {}))
    cfg = config.get("selfcal", {})
    settings.update(max_components=int(cfg.get("joint_max_components", 10)),
                    model_rounds=int(cfg.get("joint_model_rounds", 3)))
    return settings


def _campaign_settings(config: dict) -> dict:
    """Scientific settings used to decide whether combined models and images can be reused."""
    from .backends.difmap import imaging_settings, method_keywords
    from .namespaces import _search_settings
    img_cfg, cfg = config.get("imaging", {}), config.get("selfcal", {})
    return {"imaging": {**method_keywords(imaging_settings(img_cfg)), **_search_settings(img_cfg),
                        "robust": [float(value) for value in img_cfg.get("robust", [-2, 0, 2])],
                        "average_channels": bool(img_cfg.get("average_channels", True))},
            "joint": {**method_keywords(joint_model_settings(config)), "solints": cfg.get("solints") or [],
                      "min_improvement": float(cfg.get("min_improvement", 0.002)),
                      "max_bad_fraction": float(cfg.get("max_bad_fraction", 0.25)),
                      "joint_amplitude": bool(cfg.get("joint_amplitude", False)),
                      "prior_sigma": float(cfg.get("prior_sigma", 0.1)),
                      "outlier_sigma": float(cfg.get("outlier_sigma", 1.0))},
            "phase_referencing": config.get("phase_referencing", {})}


def stale_reason(observations: list, report_path: Path, report: dict, settings: Optional[dict] = None) -> str:
    """Why an existing campaign report cannot be reused ("" when it can).

    Rebuild when calibration or scientific settings changed, a combination failed,
    or its calibrated data, fitted models or images are no longer present.
    """
    if "epochs" in report and set(report["epochs"]) != {obs.project_code for obs in observations}:
        return "the campaign epochs changed"
    outdated = _epochs_changed_since(observations, report_path)
    if outdated:
        return f"the calibration of {', '.join(outdated)} changed after the last combination"
    if report.get("failed"):
        return f"the last combination was incomplete ({'; '.join(report['failed'])})"
    if settings is not None and report.get("settings") != settings:
        return "the combined-model or imaging settings changed"
    if not report.get("combined"):
        return "no combined images were recorded"
    for source, record in report.get("calibrators", {}).items():
        if not record.get("model") or not Path(record["model"]).is_file():
            return f"the joint model of {source} is missing"
        for code, table in record.get("tables", {}).items():
            if not Path(table).is_dir():
                return f"the joint table of {source} in {code} is missing"
    for source, record in report["combined"].items():
        if not record.get("ms") or not Path(record["ms"]).is_dir():
            return f"the combined calibrated data of {source} are missing"
        if not record.get("images") or any(not info.get("fits") or not Path(info["fits"]).is_file()
                                            for info in record["images"].values()):
            return f"the combined images of {source} are missing"
    return ""


def _has_source(obs, source: str) -> bool:
    """An epoch participates only when it observed the source, not merely when the campaign declared it."""
    return source in (obs.metadata.source_names if obs.metadata is not None else obs.sources.names)


def _joint_step(source: str) -> str:
    """Pipeline step owning a calibrator's per-epoch joint-model corrections."""
    return f"joint_{source}"


def _reimage_reason(obs, steps: set[str], settings: dict) -> str:
    """Why the images an epoch has on disk cannot be its images before the combination ("" when they can).

    They can when none of them was made after ``steps`` (the ``joint_<source>`` tables just dropped from its
    chain) completed, and all were made with the imaging ``settings`` in force (as their image reports,
    ``images/<code>_<source>.search.json``, record). A combination interrupted after it refined the epochs
    leaves its tables in the chains but its images untouched: those are then the "before" images already,
    and only the calibrated data are redone.
    """
    from .backends.difmap import method_keywords
    state = obs._state.as_dict()
    joint = [entry["timestamp"] for step in steps if (entry := state.get(step, {})).get("status") == "done"
             and entry.get("timestamp")]
    wanted = json.loads(json.dumps(method_keywords(settings)))
    image_dir = Path(obs.work_dir) / "images"
    for name in (obs.metadata.source_names if obs.metadata else obs.sources.names):
        entry = state.get(f"clean_{name}", {})
        if entry.get("status") != "done":
            continue
        if joint and str(entry.get("timestamp", "")) > min(joint):
            return f"the images of {name} were made with the joint calibration"
        try:
            recorded = json.loads((image_dir / f"{obs.project_code}_{name}.search.json").read_text()).get("imaging")
        except (OSError, ValueError, AttributeError):
            recorded = None
        if recorded != wanted:
            return f"the images of {name} were made with other imaging settings"
    return ""


def refine_calibrator(observations: list, source: str, root: Path, name: str, config: dict) -> dict:
    """Model ``source`` from all the epochs that observed it and refine each epoch against that model.

    Returns the per-source record of the campaign report; ``tables`` maps the
    project codes that gained a ``joint_<source>`` table to its path.
    """
    from .backends import difmap

    cfg = dict(config.get("selfcal", {}))
    img_cfg = config.get("imaging", {})
    robust_values = [float(r) for r in img_cfg.get("robust", [-2, 0, 2])]
    solints = list(cfg.get("solints") or []) or None
    limits = {"min_improvement": float(cfg.get("min_improvement", 0.002)),
              "max_bad_fraction": float(cfg.get("max_bad_fraction", 0.25))}
    backend = observations[0]._backend
    codes = [obs.project_code for obs in observations]
    splits = [obs.clean.split_ms(source) for obs in observations]
    fluxes = [_epoch_flux(obs, source) for obs in observations]
    known = [flux for flux in fluxes if flux]
    reference = float(np.median(known)) if known else 1.0
    fluxes = [flux or reference for flux in fluxes]
    scales = [reference / flux for flux in fluxes]
    logger.info("joint model of {}: epochs {} with {} Jy; modelled at {:.3f} Jy", source, ", ".join(codes),
                ", ".join(f"{flux:.3f}" for flux in fluxes), reference)
    combined = backend.export.merge(codes, [source], inputs=splits, scales=scales,
                                    outputvis=str(root / "data" / f"{name}_{source}.joint.ms"))
    (root / "selfcal").mkdir(parents=True, exist_ok=True)
    model = difmap.joint_model(combined, str(root / "selfcal" / f"{name}.{source}.joint"),
                               robust_values=robust_values, solints=solints,
                               mapsize=int(img_cfg.get("mapsize") or 8192), **limits, **joint_model_settings(config))
    model_flux = float(model["model_flux"])
    if not np.isfinite(model_flux) or model_flux <= 0:
        raise ValueError(f"joint model of {source} has invalid total flux {model_flux}")
    record = {"epochs": codes, "flux_jy": dict(zip(codes, fluxes)), "reference_flux_jy": reference,
              "model": model["model"], "model_flux_jy": model["model_flux"], "joint_rounds": model["rounds"],
              "joint_images": {str(robust): info for robust, info in model["images"].items()},
              "joint_model": model.get("image_model", {}),
              "refinement": {}, "tables": {}, "failed_epochs": []}
    step = _joint_step(source)
    for obs, split, flux in zip(observations, splits, fluxes):
        code = obs.project_code
        try:
            out_dir = Path(obs.work_dir) / "selfcal"
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = out_dir / f"{code}.{source}.joint"
            # The joint model carries the structure; the flux density is this epoch's own.
            epoch_model = difmap.scale_model_file(model["model"], f"{prefix}.mod", flux / model_flux)
            report = difmap.refine_against_model(split, str(obs._backend.ms_path(code)), str(prefix), epoch_model,
                                                 solints=solints, amplitude=bool(cfg.get("joint_amplitude", False)),
                                                 prior_sigma=float(cfg.get("prior_sigma", 0.1)),
                                                 outlier_sigma=float(cfg.get("outlier_sigma", 1.0)),
                                                 model_sigma=_own_model_sigma(obs, source),
                                                 workers=cfg.get("workers") or None, **limits)
            if report["caltable"] and Path(report["caltable"]).is_dir():
                table = CalTable(cal_type=f"joint_{source}", path=report["caltable"], field=source,
                                 gainfield=source, interp="linear", apply_to=obs.selfcal._apply_to(source), snr=0.0)
                try:
                    obs.add_gaintable(table, step)
                finally:
                    if any(registered is table for registered in obs.gaintables):
                        record["tables"][code] = report["caltable"]
            accepted = [r["solint"] for r in report["rounds"] if r["accepted"]]
            record["refinement"][code] = {"accepted_phase_solints": accepted, "amplitude": report["amplitude"],
                                          "amplitude_report": report["amplitude_files"].get("json", ""),
                                          "chisq_before": report["chisq_before"], "chisq_after": report["chisq_after"]}
            amplitudes = ", ".join(f"{name} x{value:.2f}" for name, value in report["amplitude"].items())
            logger.info("[{}] {} against the joint model: phase steps accepted {}; amplitudes {}; chisq {:.4g} -> {:.4g}",
                        code, source, ", ".join(accepted) or "none", amplitudes or "kept",
                        report["chisq_before"], report["chisq_after"])
            obs._state.mark_complete(step, outputs=[report["caltable"]] if report["caltable"] else [])
        except Exception as exc:  # noqa: BLE001 - one epoch failing must not lose the others
            warnings.warn(f"{code}: refinement of {source} against the joint model failed ({exc})")
            obs._state.mark_failed(step, str(exc))
            record["failed_epochs"].append(code)
    return record


def image_combined(by_source: dict[str, list], root: Path, name: str, config: dict) -> dict:
    """Concatenate the epochs of every source and image each concatenation; return the image reports.

    ``by_source`` maps a source to the observations that have it. The concatenations are made one after
    the other, then imaged side by side (:func:`vlbipy.backends.difmap.run_image_jobs`).

    Returns
    -------
    dict
        Source -> its record (``epochs``, ``ms``, ``png``, ``shift_mas``, ``search``, ``model``, ``images``),
        or the exception that stopped it.
    """
    from .backends import difmap
    from .namespaces import _search_settings, _write_image_report
    from .plotting import plot_image_grid

    img_cfg = config.get("imaging", {})
    settings = difmap.imaging_settings(img_cfg)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    jobs: dict = {}
    for source, observations in by_source.items():
        codes = [obs.project_code for obs in observations]
        try:
            splits = [obs.clean.split_ms(source) for obs in observations]
            combined = observations[0]._backend.export.merge(codes, [source], inputs=splits,
                                                             outputvis=str(root / "data" / f"{name}_{source}.ms"))
        except Exception as exc:  # noqa: BLE001 - one source failing must not lose the rest
            results[source] = exc
            continue
        results[source] = {"epochs": codes, "ms": combined}
        jobs[source] = {"split_ms": combined, "imagename": str(image_dir / f"{name}_{source}"),
                        "robust_values": [float(r) for r in img_cfg.get("robust", [-2, 0, 2])],
                        "average_channels": bool(img_cfg.get("average_channels", True)),
                        **settings, **_search_settings(img_cfg)}
    for source, report in zip(jobs, difmap.run_image_jobs(list(jobs.values()))):
        if isinstance(report, Exception):
            results[source] = report
            continue
        _write_image_report(image_dir / f"{name}_{source}.search.json", report, settings)
        try:
            png = plot_image_grid({source: {robust: info["fits"] for robust, info in report["images"].items()}},
                                  image_dir, name)
        except Exception as exc:  # noqa: BLE001 - a preview must not lose the images
            warnings.warn(f"{name}: image preview of {source} failed ({exc})")
            png = []
        results[source].update(png=png[0] if png else "", shift_mas=report.get("shift_mas", [0.0, 0.0]),
                               search=report.get("search"), model=report.get("image_model") or report.get("model", {}),
                               images={str(robust): info for robust, info in report["images"].items()})
    return results


def combine_epochs(vlbi, *, force: bool = False) -> dict:
    """Run the multi-epoch stage of a campaign (see the module docstring); return its report.

    Parameters
    ----------
    vlbi : VLBIObs
        A campaign whose epochs have all been calibrated, self-calibrated and split.
    force : bool
        Redo the stage even when ``<name>.campaign.json`` already exists.
    """
    from .backends.difmap import imaging_settings
    observations = list(vlbi.observations)
    config = vlbi.config
    settings = imaging_settings(config.get("imaging", {}))
    policy = _campaign_settings(config)
    policy["reference_plans"] = {obs.project_code: list(obs.phaseref_plan().references) for obs in observations}
    codes = [obs.project_code for obs in observations]
    name = campaign_name(codes)
    root = campaign_dir(observations)
    report_path = root / f"{name}.campaign.json"
    if report_path.is_file() and not force:
        try:
            previous = json.loads(report_path.read_text())
        except (OSError, ValueError):
            previous = {}
        previous = previous if isinstance(previous, dict) else {}
        reason = stale_reason(observations, report_path, previous, settings=policy)
        if not reason:
            logger.info("campaign {}: already combined ({}); use force=True to redo", name, report_path)
            return previous
        logger.info("campaign {}: {}; combining again", name, reason)
    root.mkdir(parents=True, exist_ok=True)
    logger.info("campaign {}: combining {} epochs ({}) -> {}", name, len(codes), ", ".join(codes), root)
    # Whatever fails is recorded: a report with failures is a record of an attempt, and the
    # next run combines again instead of taking the stage for done.
    report: dict = {"name": name, "epochs": codes, "settings": policy,
                    "calibrators": {}, "combined": {}, "failed": []}

    # A previous combination left its tables in the chains: the joint model must be built from the
    # per-epoch calibration alone, so they come out first (and the splits are redone without them).
    stale: set[str] = set()
    # The phase-reference sources in each epoch's solve order (the targets with inverse phase referencing).
    calibrators: list[str] = []
    for obs in observations:
        for source in obs.phaseref_plan().references:
            if source not in calibrators:
                calibrators.append(source)
    joint_steps = {_joint_step(source) for source in calibrators}
    joint_steps.update(table.step for obs in observations for table in obs.gaintables
                       if table.step.startswith("joint_"))
    for obs in observations:
        dropped = obs.drop_gaintables(joint_steps)
        if dropped:
            stale.add(obs.project_code)
            obs.calibrate.apply(force=True)
            obs.export.per_source(force=True, uvfits=False)
        # The "before" of the comparison below must not contain a previous combination, and must be
        # imaged the way the "after" will be; images that already are stay.
        reason = _reimage_reason(obs, joint_steps if dropped else set(), settings)
        if reason:
            logger.info("campaign {}: imaging {} again before the combination: {}", name, obs.project_code, reason)
            vlbi._image_all(obs, force=True, reimage=True)
    report["images_before"] = {obs.project_code: epoch_image_statistics(obs) for obs in observations}

    changed: set[str] = set()
    for source in calibrators:
        having = [obs for obs in observations if _has_source(obs, source)]
        if len(having) < 2:
            logger.info("campaign {}: {} was observed in one epoch only ({}); no joint model", name, source,
                        ", ".join(obs.project_code for obs in having) or "none")
            continue
        # A calibrator refined earlier in this loop was applied to this one as well: its
        # split has to carry that before it is modelled, as in the single-epoch sequence.
        for obs in having:
            if obs.project_code in changed:
                obs.calibrate.apply(force=True, field=source)
                obs.clean.split_ms(source, force=True)
        try:
            record = refine_calibrator(having, source, root, name, config)
        except Exception as exc:  # noqa: BLE001 - one calibrator failing must not lose the rest
            warnings.warn(f"campaign {name}: joint model of {source} failed ({exc})")
            report["failed"].append(f"joint model of {source}")
            continue
        report["calibrators"][source] = record
        report["failed"] += [f"refinement of {source} in {code}" for code in record["failed_epochs"]]
        changed |= set(record["tables"])

    for obs in observations:
        if obs.project_code not in changed | stale:
            continue
        obs.calibrate.apply(force=True)
        obs.export.per_source(force=True)
        vlbi._image_all(obs, force=True, reimage=True)
        # The epoch's plots of its calibrated data (and so its pages) show the final calibration.
        vlbi._calibrated_diagnostics(obs)
        vlbi._final_plots(obs)
    report["images_after"] = {obs.project_code: epoch_image_statistics(obs) for obs in observations}

    sources: list[str] = []
    for obs in observations:
        ordered = [s.name for s in obs.sources.targets] + [s.name for s in obs.sources]
        for source in ordered:
            if source not in sources and (obs.metadata is None or source in obs.metadata.source_names):
                sources.append(source)
    by_source = {source: [obs for obs in observations if _has_source(obs, source)] for source in sources}
    for source, result in image_combined(by_source, root, name, config).items():
        if isinstance(result, Exception):
            warnings.warn(f"campaign {name}: combined image of {source} failed ({result})")
            report["failed"].append(f"combined image of {source}")
        else:
            report["combined"][source] = result

    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log_summary(report)
    logger.info("campaign {}: report written to {}", name, report_path)
    return report


def _log_summary(report: dict) -> None:
    """Log, per source, the image of every epoch before and after the joint refinement and the combined one."""
    for source, combined in report.get("combined", {}).items():
        for robust, info in sorted(combined.get("images", {}).items(), key=lambda item: float(item[0])):
            parts = []
            for code in report["epochs"]:
                before = report["images_before"].get(code, {}).get(source, {})
                after = report["images_after"].get(code, {}).get(source, {})
                tag = f"{float(robust):g}"
                if tag in after:
                    was = f"{before[tag]['dynamic_range']:.0f} -> " if tag in before else ""
                    parts.append(f"{code} {was}{after[tag]['dynamic_range']:.0f}")
            logger.info("campaign {}: {} robust {:+g}: combined peak {:.4g} Jy/beam, rms {:.3g}, DR {:.0f} "
                        "(per-epoch DR: {})", report["name"], source, float(robust), info["peak"], info["rms"],
                        info["dynamic_range"], "; ".join(parts) or "n/a")
