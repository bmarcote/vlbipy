"""Imaging and self-calibration with difmapy (pure difmapy; no CASA here).

The sequence for a calibrator follows the standard interactive difmap session:

1. load the single-source split measurement set (averaged in time and frequency
   by ``[export]``); Briggs weighting;
2. **modelfit** one circular Gaussian at the phase centre (the only starting
   model ever used: flux, position and size free);
3. **phase-only self-calibration** from long to short solution intervals
   (``30min -> 10min -> 5min -> scan`` and ``2min`` when the scans are longer
   than 5 min). Every step is checked - the weighted chi-squared must drop by
   ``min_improvement`` and no more than ``max_bad_fraction`` of the solutions
   may fail - and rejected steps are reverted (the observation is cloned
   before each attempt). The model is re-fitted after every accepted step;
4. **amplitude calibration** with the Bayesian gscale (``bayes_gscale``), which
   compares source models, leaves each station out in turn and shrinks the
   corrections towards the a-priori calibration;
5. **visibility-domain model fitting**, starting with one circular Gaussian; the fitted model plus untouched
   residuals are restored at the requested Briggs robust values and written as FITS. CLEAN is explicit opt-in only.

The phase and amplitude gains are exported as CASA "G Jones" tables that
refer to the *parent* multi-source measurement set, so the CASA side can apply
them to the other fields (``vlbipy.namespaces.SelfcalNamespace``).

A target or check source is only imaged (:func:`image_source`).

Every source is first looked for in a wide dirty map (:func:`locate_source`).
The phase centre is moved onto the peak only when that peak is significant *and*
far from the centre (more than ``recentre_min_beams`` synthesized beams), i.e.
when the regular map could miss it. A source that is merely a little off-centre
is left where it is.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import BackendError
from ..logging_utils import get_logger

logger = get_logger()

#: Default phase-only self-cal ladder (difmapy solint strings), long to short.
DEFAULT_SOLINTS = ("30min", "10min", "5min", "scan")
#: Added at the end of the ladder when the median scan is longer than this many seconds.
LONG_SCAN_SEC = 300.0
SHORT_SOLINT = "2min"
#: CLEAN stops (and undoes its last batch) when the residual rms grows by more than this fraction.
DIVERGENCE_TOLERANCE = 0.02
#: ... and does not stop on the residual-peak criterion while a batch still lowers the residual rms by more
#: than this fraction: the rms is then still made of the sidelobes of flux that has not been cleaned.
RMS_CONVERGENCE = 0.01
#: Pixels per resolution element in the wide dirty map searched for a target.
SEARCH_OVERSAMPLE = 3.0
#: Largest side (pixels) of that map; the cell grows instead when the field needs more.
SEARCH_MAX_PIXELS = 16384


#: Width in pixels of the map the final images are made on (difmapy's own default is 4096). Only the inner
#: half along each axis is valid and kept in the FITS file, so the images are IMAGE_PIXELS / 2 wide on disk.
IMAGE_PIXELS = 8192
MODEL_RESIDUAL_PRECISION = 8 * np.finfo(np.float32).eps


def _load_difmapy():
    """Import difmapy lazily so vlbipy imports without it; plots go off-screen."""
    import os
    os.environ.setdefault("DIFMAPY_PLOT_MODE", "inline")
    try:
        import difmapy  # noqa: WPS433 - optional dependency
    except ImportError as exc:
        raise BackendError("imaging and self-calibration need difmapy: pip install vlbipy[difmap]") from exc
    # A directory called "difmapy" on sys.path (a source checkout next to the project) imports as an empty
    # namespace package: say so instead of failing later with "module has no attribute 'load'".
    if not hasattr(difmapy, "load"):
        where = ", ".join(str(p) for p in getattr(difmapy, "__path__", [])) or "an unknown location"
        raise BackendError(f"difmapy is not installed: 'import difmapy' found only the directory {where}, not the "
                           "package. Install it (pip install vlbipy[difmap], or 'pip install -e' on its checkout)")
    return difmapy


def selfcal_ladder(scan_seconds: list[float], base: tuple = DEFAULT_SOLINTS) -> list[str]:
    """Return the phase self-cal solution intervals for these scan lengths (long to short).

    ``2min`` is appended when the median scan exceeds :data:`LONG_SCAN_SEC`;
    intervals longer than the whole track are dropped since they equal ``inf``.
    """
    ladder = list(base)
    if scan_seconds and float(np.median(scan_seconds)) > LONG_SCAN_SEC:
        ladder.append(SHORT_SOLINT)
    return ladder


def load(split_ms: str, *, robust: float = 0.0, mapsize: Optional[int] = None,
         cell_mas: Optional[float] = None, average_channels: bool = False):
    """Load a single-source measurement set into difmapy with Stokes I and Briggs weighting.

    The map size defaults to difmapy's ``auto_mapsize`` (resolution / 10 per pixel).
    ``average_channels`` collapses every IF to one channel while loading: difmapy
    images and self-calibrates per IF anyway, and the copies its self-cal makes
    (one per ``bayes_gscale`` thread) are then that many times smaller.
    """
    difmapy = _load_difmapy()
    obs = difmapy.load(str(split_ms), freqavg="all") if average_channels else difmapy.load(str(split_ms))
    if mapsize and cell_mas:
        obs.mapsize(int(mapsize), float(cell_mas))
    else:
        obs.auto_mapsize()
    obs.uvweight(robust=float(robust))
    logger.info("difmapy: loaded {} ({} antennas, {} IFs, {:.0%} flagged)", Path(split_ms).name,
                len(obs.antennas), obs.nif, obs.flagged_fraction)
    return obs


def scan_lengths(obs) -> list[float]:
    """Scan durations in seconds as difmapy delimits them."""
    return [float(s["tmax"] - s["tmin"]) for s in obs.scans()]


def fit_starting_model(obs) -> dict:
    """Fit a single circular Gaussian (flux, position, size free), seeded at the dirty-map peak.

    This is always the starting model: with no model difmapy seeds exactly that
    component at the residual peak (``seed_model``) and ``modelfit`` refines it.
    """
    obs.clrmod()
    result = _checked_modelfit(obs.modelfit(quiet=False, verbose=False))
    comp = result["components"][0]
    logger.info("difmapy modelfit: {:.3f} Jy circular Gaussian, FWHM {:.2f} mas at ({:+.2f}, {:+.2f}) mas, "
                "rchisq {:.3g}, converged={}", comp.get("flux", float("nan")), comp.get("major", float("nan")),
                comp.get("x", 0.0), comp.get("y", 0.0), result.get("rchisq", float("nan")),
                result.get("converged"))
    return result


def _bad_fraction(report: dict) -> float:
    """Fraction of solution bins the self-cal could not solve."""
    nbins = max(int(report.get("nbins", 0)), 1)
    return float(report.get("nbadsol", 0)) / nbins


def phase_selfcal(obs, solints: list[str], *, min_improvement: float = 0.002,
                  max_bad_fraction: float = 0.25, refit: bool = True) -> tuple[object, list[dict]]:
    """Phase-only self-calibration down the ``solints`` ladder with accept/revert per step.

    A step is accepted when the weighted chi-squared of the model fit drops by at
    least ``min_improvement`` (fractional; small on purpose, a 1% gain is worth keeping) and at most ``max_bad_fraction`` of
    the solution bins failed; the model is then re-fitted. A rejected step is
    reverted by discarding the attempt (the observation is cloned first) and
    ends the ladder: shorter intervals only add noise once the data stop asking
    for them. With ``refit=False`` the model is left exactly as it is (a model
    brought in from elsewhere, see :func:`refine_against_model`).

    Returns
    -------
    (observation, rounds)
        The (possibly replaced) observation object and one dict per attempted
        round: ``solint, chisq_before, chisq_after, improvement, nbins, nbadsol,
        bad_fraction, accepted``.
    """
    rounds: list[dict] = []
    for solint in solints:
        trial = obs.copy()
        report = trial.selfcal(phase=True, amp=False, solint=solint, quiet=False, verbose=False)
        before, after = float(report["fit_before"]["chisq"]), float(report["fit_after"]["chisq"])
        improvement = (before - after) / before if before > 0 else 0.0
        bad = _bad_fraction(report)
        accepted = improvement >= min_improvement and bad <= max_bad_fraction and math.isfinite(after)
        rounds.append({"solint": solint, "chisq_before": before, "chisq_after": after,
                       "improvement": improvement, "nbins": int(report.get("nbins", 0)),
                       "nbadsol": int(report.get("nbadsol", 0)), "bad_fraction": bad, "accepted": accepted})
        logger.info("difmapy selfcal phase solint={}: chisq {:.4g} -> {:.4g} ({:+.1%}), {} bad of {} bins -> {}",
                    solint, before, after, -improvement, rounds[-1]["nbadsol"], rounds[-1]["nbins"],
                    "accepted" if accepted else "rejected")
        if not accepted:
            break
        if refit:
            try:
                _checked_modelfit(trial.modelfit(quiet=False, verbose=False))
            except BackendError:
                rounds[-1].update(accepted=False, reason="modelfit refit failed")
                logger.warning("difmapy selfcal phase solint={}: model refit failed; keeping the previous round", solint)
                break
        obs = trial
    return obs, rounds


def amplitude_calibration(obs, prefix: str, parent_ms: str, *, models=("current", "gauss1", "gauss2", "gauss3"),
                          prior_sigma: float = 0.1, workers: Optional[int] = None, allow_clean: bool = False) -> dict:
    """Bayesian station amplitude calibration; writes ``<prefix>.json``, ``.png`` and the CASA table.

    The table refers to ``parent_ms`` (the multi-source measurement set), so
    CASA can apply the corrections to every field. Returns the JSON-able summary
    with the table path under ``"caltable"``.
    """
    models = tuple(models)
    if not allow_clean:
        models = tuple(model for model in models if not str(getattr(model, "kind", model)).strip().lower().startswith("clean"))
        models = models or ("current", "gauss1", "gauss2", "gauss3")
    result = obs.bayes_gscale(models=models, prior_sigma=prior_sigma, prefix=str(prefix),
                              outformat="CASA", ms=str(parent_ms), workers=workers, plot=False, quiet=False,
                              verbose=False)
    raw_files = dict(getattr(result, "files", {}) or {})
    files = {k: [str(x) for x in v] if isinstance(v, (list, tuple)) else str(v) for k, v in raw_files.items()}
    table = files.get("caltable") or files.get("casa") or f"{prefix}.G"
    table = table[0] if isinstance(table, list) else table
    summary = {"best_model": getattr(result, "best_model", None), "caltable": str(table), "files": files,
               "summary": result.summary() if callable(getattr(result, "summary", None)) else str(result)}
    try:
        summary["report"] = result.to_dict()
    except Exception:  # noqa: BLE001 - the JSON report on disk has the details anyway
        summary["report"] = {}
    logger.info("difmapy bayes_gscale: best model {}; table {}", summary["best_model"], table)
    return summary


def write_gain_record(obs, path: str, amplitude: Optional[dict] = None) -> dict:
    """Write the amplitude gain correction every station needed, per IF, as a small JSON record.

    The Bayesian report (``<prefix>.amp.json``) holds the same numbers among its diagnostics; this file is
    the plain answer to "what did each antenna need": one value per station and IF.

    Parameters
    ----------
    obs : difmapy Observation
        The self-calibrated observation (its accumulated gains are read with ``station_gains``).
    path : str
        Output JSON path.
    amplitude : dict, optional
        Summary of :func:`amplitude_calibration`; adds the per-station uncertainty and the probability that
        a correction was needed, when its report has them.

    Returns
    -------
    dict
        The record written: ``source``, ``antennas``, ``if_freqs_hz``, ``convention``, ``gain_per_if``
        (station -> list over IFs), ``gain_median`` (station -> float) and, when known, ``sigma`` and
        ``p_correction`` (station -> float).
    """
    per_if = {name: [float(v) for v in values] for name, values in obs.station_gains(per_if=True).items()}
    record = {"source": obs.source, "antennas": list(per_if),
              "if_freqs_hz": [float(f) for f in ((amplitude or {}).get("report", {}).get("if_freqs_hz") or [])],
              "convention": "amplitude factor the visibilities of the station were multiplied by; 1.0 = no "
                            "correction (or no solution). CASA tables store the reciprocal.",
              "gain_per_if": per_if,
              "gain_median": {name: float(np.median(values)) for name, values in per_if.items()}}
    stations = (amplitude or {}).get("report", {}).get("stations") or []
    for key in ("sigma", "p_correction"):
        # A station without a solution has no uncertainty: null in the file, not NaN (which is not JSON).
        values = {s["station"]: (float(s[key]) if math.isfinite(float(s[key])) else None)
                  for s in stations if s.get("station") and s.get(key) is not None}
        if values:
            record[key] = values
    Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    changed = {name: value for name, value in record["gain_median"].items() if abs(value - 1.0) > 0.005}
    logger.info("difmapy: station gain corrections of {} -> {} ({})", obs.source, Path(path).name,
                ", ".join(f"{name} {value:.3f}" for name, value in changed.items()) or "none above 0.5%")
    return record


def export_phase_table(obs, path: str, parent_ms: str, since) -> dict:
    """Write the gains accumulated since ``since`` (the phase self-cal) as a CASA table for ``parent_ms``."""
    info = obs.savecaltable(str(path), outformat="CASA", ms=str(parent_ms), since=since, quiet=False, verbose=False)
    logger.info("difmapy: phase self-cal table {} ({} rows)", info.get("path", path), info.get("nrows"))
    return dict(info)


def clean_image(obs, fits_path: str, *, robust: float, niter: int = 4000, gain: float = 0.05,
                threshold_sigma: float = 3.0, batch: int = 100) -> tuple[object, dict]:
    """CLEAN the current (calibrated) data at one robust value and write a restored FITS image.

    Returns ``(observation, info)``: the observation may be the pre-divergence clone.

    The model is cleared first (the CLEAN image should not inherit the
    modelfit component), then Högbom CLEAN runs in batches until the residual
    peak inside the inner quarter drops below ``threshold_sigma`` times its rms
    while the rms itself has stopped falling (see :data:`RMS_CONVERGENCE`), the
    rms starts to rise, or ``niter`` components. Returns peak / rms / dynamic
    range / model flux / beam and the FITS path.
    """
    obs.clrmod()
    obs.uvweight(robust=float(robust))
    obs.invert(verbose=False)
    done = 0
    rms = float(obs.imstat().get("rms", 0.0))
    stopped = ""
    while done < niter:
        checkpoint = obs.copy()
        obs.clean(min(batch, niter - done), gain, verbose=False)
        stats = obs.imstat()
        new_rms = float(stats.get("rms", 0.0))
        # Unwindowed Hogbom CLEAN on sparse VLBI coverage can run away: once the
        # residual rms stops falling the last batch is undone and the loop ends.
        if new_rms > rms * (1.0 + DIVERGENCE_TOLERANCE):
            obs = checkpoint
            stopped = f"residual rms rose ({rms:.3g} -> {new_rms:.3g})"
            break
        still_falling = rms > 0 and (rms - new_rms) > RMS_CONVERGENCE * rms
        done += batch
        rms = new_rms
        # The residual rms shrinks as the sidelobes are removed, so the stopping level
        # is measured on the current residual, not on the dirty map. For the same reason the
        # peak criterion alone is not trusted while the rms is still dropping: on a faint,
        # resolved source seen mostly by one sensitive baseline, the sidelobes of the
        # uncleaned flux fill the map and lift the rms until the peak already looks like noise
        # (V589A target, natural weighting: stopped after 100 components at twice the noise).
        if rms > 0 and abs(float(stats.get("max", 0.0))) < threshold_sigma * rms and not still_falling:
            stopped = f"residual peak below {threshold_sigma:g} sigma"
            break
    obs.restore(verbose=False)
    obs.wmap(str(fits_path), overwrite=True)
    _crop_to_valid(fits_path, obs.valid_slice)
    valid = obs.valid_slice
    peak = float(np.nanmax(np.asarray(obs.restored_map)[valid]))
    final = obs.imstat()
    noise = float(final.get("rms", rms)) or rms
    beam = obs.estimated_beam
    info = {"fits": str(fits_path), "robust": float(robust), "peak": peak, "rms": noise,
            "dynamic_range": peak / noise if noise > 0 else 0.0, "model_flux": float(obs.model_flux),
            "ncomponents": done, "beam": [float(b) for b in np.atleast_1d(beam)] if beam is not None else []}
    info["stopped"] = stopped or "niter reached"
    logger.info("difmapy clean robust={:+g}: peak {:.4g} Jy/beam, rms {:.3g}, DR {:.0f}, {:.4g} Jy in {} comps ({}) -> {}",
                robust, peak, noise, info["dynamic_range"], info["model_flux"], done, info["stopped"],
                Path(fits_path).name)
    return obs, info


def _crop_to_valid(fits_path: str, valid: tuple) -> None:
    """Keep only difmapy's valid inner quarter of the map in the FITS file (CRPIX shifted to match)."""
    from astropy.io import fits
    with fits.open(str(fits_path), mode="update") as hdul:
        hdu = hdul[0]
        data = hdu.data
        ys, xs = valid
        if data.ndim == 2:
            hdu.data = data[ys, xs]
        else:
            hdu.data = data[..., ys, xs]
        hdu.header["CRPIX1"] = float(hdu.header.get("CRPIX1", 0.0)) - xs.start
        hdu.header["CRPIX2"] = float(hdu.header.get("CRPIX2", 0.0)) - ys.start
        hdu.header["HISTORY"] = "vlbipy: cropped to the difmapy inner quarter (the valid map area)"
        hdul.flush()


def _checked_modelfit(result: dict) -> dict:
    """Reject failed or nonphysical visibility fits rather than replacing them with CLEAN."""
    components = result.get("components") or []
    valid = bool(components) and result.get("converged", False) and math.isfinite(float(result.get("rchisq", math.nan)))
    for component in components:
        valid = valid and all(math.isfinite(float(component.get(key, 0.0)))
                              for key in ("flux", "x", "y", "major"))
        valid = valid and float(component.get("flux", 0.0)) > 0 and float(component.get("major", 0.0)) >= 0
    if not valid:
        raise BackendError("difmapy modelfit did not produce a converged, finite positive-flux model; no CLEAN fallback")
    return result


def fit_image_model(obs, *, max_components: int = 1, component_sigma: float = 5.0,
                    component_improvement: float = 0.01) -> tuple[object, dict]:
    """Fit one Gaussian first; add components only above noise and float32 visibility precision, with fit gains."""
    if int(max_components) < 1 or component_sigma <= 0 or not 0 <= component_improvement < 1:
        raise ValueError("invalid modelfit component limit, significance or improvement threshold")
    fit = _checked_modelfit(obs.modelfit(quiet=False, verbose=False))
    stopped = "component limit reached"
    while len(obs.model) < int(max_components):
        obs.invert(verbose=False)
        rms = float(obs.noise_stats().get("rms", 0.0))
        _, peak = obs.peak_offset()
        floor = MODEL_RESIDUAL_PRECISION * abs(float(obs.model_flux))
        if not math.isfinite(rms) or rms <= 0 or not math.isfinite(float(peak)) or float(peak) < max(component_sigma * rms, floor):
            stopped = "no significant positive residual component"
            break
        trial = obs.copy()
        trial.seed_model(type="gauss", free=("flux", "pos", "major"))
        try:
            candidate = _checked_modelfit(trial.modelfit(quiet=False, verbose=False))
        except BackendError:
            stopped = "additional component fit rejected"
            break
        before, after = float(fit["rchisq"]), float(candidate["rchisq"])
        if before <= 0 or (before - after) / before < component_improvement:
            stopped = "additional component did not improve the fit"
            break
        obs, fit = trial, candidate
    report = {**fit, "components": list(obs.model), "method": "modelfit", "stopped": stopped}
    logger.info("difmapy modelfit model: {} component(s), rchisq {:.4g}, {}",
                len(report["components"]), float(fit["rchisq"]), stopped)
    return obs, report


def modelfit_image(obs, fits_path: str, *, robust: float, model_report: Optional[dict] = None,
                   max_components: int = 1, component_sigma: float = 5.0,
                   component_improvement: float = 0.01) -> tuple[object, dict]:
    """Restore the visibility-fitted model plus untouched residuals, without deconvolving the residual map."""
    if model_report is None:
        obs, model_report = fit_image_model(obs, max_components=max_components, component_sigma=component_sigma,
                                            component_improvement=component_improvement)
    obs.uvweight(robust=float(robust))
    obs.invert(verbose=False)
    stats = obs.imstat()
    noise = float(obs.noise_stats().get("rms", stats.get("rms", 0.0)))
    obs.restore(no_residual=False, verbose=False)
    obs.wmap(str(fits_path), overwrite=True)
    _crop_to_valid(fits_path, obs.valid_slice)
    peak = float(np.nanmax(np.asarray(obs.restored_map)[obs.valid_slice]))
    beam = obs.estimated_beam
    info = {"fits": str(fits_path), "robust": float(robust), "peak": peak, "rms": noise,
            "dynamic_range": peak / noise if noise > 0 else 0.0, "model_flux": float(obs.model_flux),
            "ncomponents": len(obs.model), "beam": [float(b) for b in np.atleast_1d(beam)] if beam is not None else [],
            "stopped": model_report["stopped"], "method": "modelfit", "model": model_report,
            "image_kind": "fitted model plus residuals"}
    logger.info("difmapy modelfit robust={:+g}: {:.4g} Jy in {} component(s), peak {:.4g} Jy/beam, rms {:.3g} -> {}",
                robust, info["model_flux"], info["ncomponents"], peak, noise, Path(fits_path).name)
    return obs, info


def image_all_robust(obs, imagename: str, robust_values, *, method: str = "modelfit", niter: int = 4000,
                     gain: float = 0.05, threshold_sigma: float = 3.0, max_components: int = 1,
                     component_sigma: float = 5.0, component_improvement: float = 0.01) -> dict[float, dict]:
    """Fit once in the visibility domain and render at each robustness; CLEAN requires an explicit opt-in."""
    method = str(method).strip().lower()
    if method not in ("modelfit", "clean"):
        raise ValueError(f"unknown imaging method {method!r}; choose modelfit or explicitly request clean")
    robust_values = list(robust_values)
    if not robust_values:
        return {}
    model_report = None
    if method == "modelfit":
        obs, model_report = fit_image_model(obs, max_components=max_components, component_sigma=component_sigma,
                                            component_improvement=component_improvement)
    images = {}
    for robust in robust_values:
        robust = float(robust)
        path = f"{imagename}.robust{robust:g}.fits"
        if method == "modelfit":
            obs, images[robust] = modelfit_image(obs, path, robust=robust, model_report=model_report)
        else:
            obs, images[robust] = clean_image(obs, path, robust=robust, niter=niter, gain=gain,
                                             threshold_sigma=threshold_sigma)
    return images


def calibrate_source(split_ms: str, parent_ms: str, prefix: str, *, robust_values=(-2.0, 0.0, 2.0),
                     solints: Optional[list[str]] = None, min_improvement: float = 0.002,
                     max_bad_fraction: float = 0.25, bayes_models=("current", "gauss1", "gauss2", "gauss3"),
                     prior_sigma: float = 0.1, workers: Optional[int] = None, imagename: Optional[str] = None,
                     search_fov_mas: float = 0.0, search_sigma: float = 10.0, recentre_min_beams: float = 10.0,
                     mapsize: int = IMAGE_PIXELS, fixed_antennas: tuple = (), deferred_antennas: tuple = (),
                     fix_stage1: bool = True, **imaging_kwargs) -> dict:
    """Full difmapy calibrator sequence: model, phase ladder, amplitude gains, images, tables.

    Products (``prefix`` is a path stem, e.g. ``<work_dir>/selfcal/<code>.<source>``):
    ``<prefix>.phase.G`` (CASA table of the phase self-cal), ``<prefix>.amp.G`` (+``.json``,
    ``.png`` from the Bayesian gscale), ``<prefix>.gains.json`` (the amplitude correction of every
    station and IF, see :func:`write_gain_record`) and ``<imagename>.robust<r>.fits``. Returns a
    JSON-able report with the rounds, both table paths and the image statistics.

    The self-calibration itself runs on difmapy's default map; the final images are made on a map
    ``mapsize`` pixels wide (same cell), since that is where a larger clean field pays off.

    The channels the split keeps are averaged per IF on load (see :func:`load`):
    a calibrator sits at the phase centre, so nothing is lost. The source is
    located first (:func:`locate_source`) and re-centred only when it lies far
    from the phase centre.
    """
    obs = load(split_ms, average_channels=True)
    known = {a.lower(): a for a in obs.antennas}
    fixed = [known[n] for n in (a.lower() for a in fixed_antennas) if n in known]
    deferred = [known[n] for n in (a.lower() for a in deferred_antennas) if n in known]
    if fixed:
        obs.selfant(fixed, fix=True, verbose=False)
        logger.info("difmapy: {} fixed in self-calibration (a-priori amplitudes held)", ", ".join(fixed))
    search, shift = locate_source(obs, imagename or prefix, search_fov_mas=search_fov_mas, search_sigma=search_sigma,
                                  recentre_min_beams=recentre_min_beams)
    # Stage 1 ignores the longest-baseline antennas: they see a resolved source and would bias the
    # model while it is still coarse. They join for stage 2, when the model is good enough to hold them.
    if deferred and len(deferred) < len(obs.antennas):
        obs.ignore(*deferred)
        logger.info("difmapy: stage 1 ignores the longest-baseline antenna(s): {}", ", ".join(deferred))
    model = fit_starting_model(obs)
    ladder = solints or selfcal_ladder(scan_lengths(obs))
    snapshot = obs.gain_snapshot()
    rounds: list[dict] = []
    obs, stage1 = phase_selfcal(obs, ladder, min_improvement=min_improvement, max_bad_fraction=max_bad_fraction)
    rounds += stage1
    if deferred and len(deferred) < len(obs.antennas):
        obs.unignore(*deferred)
        obs.modelfit(verbose=False)            # the model now has to describe the long baselines too
        if fix_stage1:
            stage1_names = [a for a in obs.antennas if a not in deferred]
            obs.selfant(stage1_names, fix=True, verbose=False)
        obs, stage2 = phase_selfcal(obs, ladder, min_improvement=min_improvement,
                                    max_bad_fraction=max_bad_fraction)
        for r in stage2:
            r["stage"] = "long-baselines"
        rounds += stage2
        if fix_stage1:
            obs.selfant(stage1_names, fix=False, verbose=False)
    phase_table = export_phase_table(obs, f"{prefix}.phase", parent_ms, snapshot) if any(r["accepted"] for r in rounds) \
        else {}
    amplitude = amplitude_calibration(obs, f"{prefix}.amp", parent_ms, models=bayes_models, prior_sigma=prior_sigma,
                                      workers=workers, allow_clean=imaging_kwargs.get("method", "modelfit") == "clean")
    gains = write_gain_record(obs, f"{prefix}.gains.json", amplitude)
    # One accumulated table (phase + amplitude corrections since the first model) is what
    # gets transferred to the targets and check sources.
    accumulated = ""
    try:
        info = obs.savecaltable(f"{prefix}.all", outformat="CASA", ms=str(parent_ms), since=snapshot,
                                quiet=False, verbose=False)
        accumulated = str(info.get("path", f"{prefix}.all"))
    except Exception as exc:  # noqa: BLE001 - the separate phase/amp tables still cover it
        logger.warning("difmapy: could not export the accumulated gain table ({})", exc)
    obs.auto_mapsize(npix=int(mapsize))
    images = image_all_robust(obs, imagename or prefix, robust_values, **imaging_kwargs)
    if any(shift):
        for info in images.values():
            _record_shift(info["fits"], shift[0], shift[1])
    report = {"source": obs.source, "split_ms": str(split_ms), "model": model.get("components", []),
              "search": search, "shift_mas": list(shift),
              "model_rchisq": model.get("rchisq"), "ladder": ladder, "rounds": rounds,
              "phase_table": phase_table.get("path", ""), "amplitude": amplitude, "images": images,
              "accumulated_table": accumulated,
              "image_model": next(iter(images.values()), {}).get("model", {}),
              "station_gains": {k: float(v) for k, v in obs.station_gains().items()},
              "station_gains_per_if": gains["gain_per_if"], "gains_file": f"{prefix}.gains.json"}
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def joint_model(combined_ms: str, prefix: str, *, robust_values=(-2.0, 0.0, 2.0), model_robust: float = 0.0,
                solints: Optional[list[str]] = None, min_improvement: float = 0.002,
                max_bad_fraction: float = 0.25, mapsize: int = IMAGE_PIXELS,
                model_rounds: int = 3, **imaging_kwargs) -> dict:
    """Model a calibrator from the data of several epochs together; write ``<prefix>.mod``.

    ``combined_ms`` is the concatenation of the calibrated per-epoch splits of
    one source, brought to a common flux level. The joint uv coverage is what
    one epoch lacks: structure that no single epoch constrains is fixed here.
    The sequence is the calibrator one (single Gaussian, phase self-cal ladder)
    without amplitude calibration - the stations kept their names but not
    their gains from one epoch to the next - followed by a visibility-domain
    modelfit. Up to ``model_rounds`` cycles alternate the phase ladder with deeper
    Gaussian fitting, so newly constrained structure enters the next solve. The
    default joint component limit is ten, independent of the target's one-component
    policy. Iteration stops when neither phase gains nor component count improve.
    Its parametric components are the joint model; CLEAN is never an automatic fallback.

    Returns a JSON-able report with ``model`` (the ``.mod`` file), ``model_flux``,
    the self-cal ``rounds`` and the ``images`` (one per robust value).
    """
    if int(model_rounds) < 1:
        raise ValueError("model_rounds must be at least one")
    method = str(imaging_kwargs.get("method", "modelfit")).strip().lower()
    if method not in ("modelfit", "clean"):
        raise ValueError(f"unknown imaging method {method!r}")
    options = {"max_components": int(imaging_kwargs.get("max_components", 10)),
               "component_sigma": float(imaging_kwargs.get("component_sigma", 5.0)),
               "component_improvement": float(imaging_kwargs.get("component_improvement", 0.01))}
    obs = load(combined_ms, average_channels=True)
    fit_starting_model(obs)
    ladder = solints or selfcal_ladder(scan_lengths(obs))
    image_model, rounds, history = {}, [], []
    for iteration in range(int(model_rounds) if method == "modelfit" else 1):
        before = len(obs.model)
        obs, phase_rounds = phase_selfcal(obs, ladder, min_improvement=min_improvement,
                                          max_bad_fraction=max_bad_fraction)
        rounds.extend(dict(round_, model_round=iteration + 1) for round_ in phase_rounds)
        if method != "modelfit":
            break
        obs, image_model = fit_image_model(obs, **options)
        accepted = sum(bool(round_["accepted"]) for round_ in phase_rounds)
        history.append({"round": iteration + 1, "ncomponents": len(obs.model),
                        "rchisq": image_model.get("rchisq"), "accepted_phase_steps": accepted})
        logger.info("difmapy joint model round {}: {} component(s), {} accepted phase steps",
                    iteration + 1, len(obs.model), accepted)
        if not accepted and len(obs.model) == before:
            break
    obs.auto_mapsize(npix=int(mapsize))
    # The model image goes last so that its CLEAN components are the ones left in the observation.
    requested = [float(r) for r in robust_values]
    order = [r for r in requested if r != float(model_robust)]
    if float(model_robust) in requested or method == "clean":
        order.append(float(model_robust))
    images = {}
    for robust in order:
        if method == "modelfit":
            obs, images[robust] = modelfit_image(obs, f"{prefix}.robust{robust:g}.fits", robust=robust,
                                                model_report=image_model)
        else:
            options = {key: imaging_kwargs[key] for key in ("niter", "gain", "threshold_sigma") if key in imaging_kwargs}
            obs, images[robust] = clean_image(obs, f"{prefix}.robust{robust:g}.fits", robust=robust, **options)
    model_path = f"{prefix}.mod"
    obs.wmodel(model_path)
    report = {"source": obs.source, "combined_ms": str(combined_ms), "model": model_path,
              "model_flux": float(obs.model_flux), "ladder": ladder, "rounds": rounds,
              "image_model": image_model, "method": method, "model_rounds": len(history),
              "model_iterations": history,
              "images": {robust: images[robust] for robust in sorted(images)}}
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("difmapy joint model of {}: {:.4g} Jy in {} components -> {}", obs.source, report["model_flux"],
                method, Path(model_path).name)
    return report


def scale_model_file(model_path: str, out_path: str, factor: float) -> str:
    """Write a copy of a difmap ``.mod`` file with every component flux multiplied by ``factor``."""
    lines = []
    for line in Path(model_path).read_text().splitlines():
        fields = line.split()
        if not fields or line.lstrip().startswith("!"):
            lines.append(line)
            continue
        # The flux is the first column; a trailing "v" marks it as free in modelfit.
        token = fields[0]
        suffix = "v" if token.endswith("v") else ""
        value = float(token.rstrip("v")) * float(factor)
        lines.append(line.replace(token, f"{value:.9g}{suffix}", 1))
    Path(out_path).write_text("\n".join(lines) + "\n")
    return str(out_path)


def refine_against_model(split_ms: str, parent_ms: str, prefix: str, model_path: str, *,
                         solints: Optional[list[str]] = None, min_improvement: float = 0.002,
                         max_bad_fraction: float = 0.25, amplitude: bool = True, prior_sigma: float = 0.1,
                         outlier_sigma: float = 1.0, model_sigma: Optional[dict] = None,
                         workers: Optional[int] = None) -> dict:
    """Self-calibrate one epoch of a calibrator against a fixed model; write ``<prefix>.G``.

    The model (a ``.mod`` file, e.g. the multi-epoch one of :func:`joint_model`
    scaled to this epoch's flux density) is not touched: only the antenna gains
    move. Phases go down the same accept/revert ladder as the calibrator
    sequence; then Bayesian station/IF amplitude corrections against that same
    fixed parametric model are kept if they improve the fit. The gains
    accumulated here are exported as one CASA table for ``parent_ms``.

    Returns a JSON-able report: ``caltable`` ("" when no step was accepted),
    ``rounds``, ``amplitude`` (the accepted station factors) and the fit
    ``chisq_before`` / ``chisq_after``.
    """
    obs = load(split_ms, average_channels=True)
    obs.clrmod()
    obs.rmodel(str(model_path))
    parametric = bool(obs.model) and all(component.get("type") in ("gauss", "gaussian") or component.get("freepar", 0)
                                        for component in obs.model)
    model_policy = {"method": "modelfit" if parametric else "unverified", "revision": MODELFIT_REVISION}
    start = float(obs.moddif()["chisq"])
    snapshot = obs.gain_snapshot()
    ladder = solints or selfcal_ladder(scan_lengths(obs))
    obs, rounds = phase_selfcal(obs, ladder, min_improvement=min_improvement, max_bad_fraction=max_bad_fraction,
                                refit=False)
    gains: dict = {}
    amplitude_files: dict = {}
    if amplitude:
        trial = obs.copy()
        trial._replace_model([{**component, "freepar": 0} for component in trial.model])
        before = float(trial.moddif()["chisq"])
        result = trial.bayes_gscale(models=("current",), prior_sigma=prior_sigma, outlier_sigma=outlier_sigma,
                                     model_sigma=model_sigma, workers=workers, nloop=0, apply=True,
                                     prefix=f"{prefix}.amp", outformat="CASA", ms=str(parent_ms), plot=False, quiet=False,
                                     verbose=False)
        amplitude_files = dict(getattr(result, "files", {}) or {})
        after = float(trial.moddif()["chisq"])
        accepted = math.isfinite(after) and before > 0 and (before - after) / before >= min_improvement
        logger.info("difmapy Bayesian amplitudes against the joint model: chisq {:.4g} -> {:.4g} -> {}", before, after,
                    "accepted" if accepted else "rejected")
        if accepted:
            obs = trial
            gains = {row["station"]: float(row["correction"]) for row in result.station_table()
                     if row.get("has_data") and math.isfinite(float(row.get("correction", math.nan)))}
    table = ""
    if gains or any(r["accepted"] for r in rounds):
        info = obs.savecaltable(f"{prefix}", outformat="CASA", ms=str(parent_ms), since=snapshot, quiet=False,
                                verbose=False)
        table = str(info.get("path", f"{prefix}"))
    report = {"source": obs.source, "split_ms": str(split_ms), "model": str(model_path), "caltable": table,
              "ladder": ladder, "rounds": rounds, "amplitude": gains, "amplitude_files": amplitude_files,
              "model_policy": model_policy, "chisq_before": start, "chisq_after": float(obs.moddif()["chisq"])}
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def find_peak(obs, *, fov_mas: float = 1000.0, threshold_sigma: float = 10.0, fits_path: str = "") -> dict:
    """Search a wide dirty map (no CLEAN) for the brightest point and judge its significance.

    The map is naturally weighted for sensitivity and twice ``fov_mas`` wide,
    because only the inner half of a difmapy map along each axis is valid. The
    noise is the sigma-clipped rms left after removing the dirty beam scaled to
    the peak: in a dirty map the sidelobes of a real source would otherwise
    pass for noise and hide the very detection being tested.

    The map size and weighting of ``obs`` are changed; callers set them again.

    Parameters
    ----------
    fov_mas : float
        Width of the searched field in mas.
    threshold_sigma : float
        Peak-to-noise ratio above which the peak counts as a detection.
    fits_path : str
        Write the dirty map here when given.

    Returns
    -------
    dict
        ``x, y`` (mas east / north of the phase centre), ``peak`` (Jy/beam),
        ``rms``, ``snr``, ``detected``, ``fov_mas``, ``cell_mas``, ``npix``, ``fits``.
    """
    cell = obs.estimated_resolution() / SEARCH_OVERSAMPLE
    npix = int(math.ceil(2.0 * float(fov_mas) / cell / 64.0)) * 64
    if npix > SEARCH_MAX_PIXELS:
        npix, cell = SEARCH_MAX_PIXELS, 2.0 * float(fov_mas) / SEARCH_MAX_PIXELS
    obs.clrmod()
    obs.uvweight(robust=2.0)
    obs.mapsize(npix, cell)
    obs.invert(verbose=False)
    (x, y), peak = obs.peak_offset()
    dirty, beam = np.asarray(obs.dmap), np.asarray(obs.dbeam)
    iy, ix = int(round(y / cell)) + npix // 2, int(round(x / cell)) + npix // 2
    by, bx = np.unravel_index(int(np.argmax(beam)), beam.shape)
    residual = dirty - peak * np.roll(beam, (iy - by, ix - bx), axis=(0, 1))
    rms = float(obs.noise_stats(image=residual[obs.valid_slice]).get("rms", 0.0))
    snr = peak / rms if rms > 0 else 0.0
    if fits_path:
        obs.wdmap(str(fits_path), overwrite=True)
    result = {"x": float(x), "y": float(y), "peak": float(peak), "rms": rms, "snr": float(snr),
              "detected": bool(snr >= threshold_sigma), "fov_mas": float(fov_mas), "cell_mas": float(cell),
              "npix": npix, "fits": str(fits_path)}
    logger.info("difmapy search: {:.0f} mas field ({} px of {:.2f} mas): peak {:.4g} Jy/beam at ({:+.1f}, {:+.1f}) mas, "
                "rms {:.3g}, {:.1f} sigma -> {}", fov_mas, npix, cell, peak, x, y, rms, snr,
                "detected" if result["detected"] else f"below {threshold_sigma:g} sigma")
    return result


def _record_shift(fits_path: str, east_mas: float, north_mas: float) -> None:
    """Move the reference coordinate of a FITS image to the shifted phase centre.

    difmapy writes the original phase centre into the header whatever the
    shift, so the image of a re-centred source would be labelled with the
    position it was moved away from.
    """
    from astropy.io import fits
    with fits.open(str(fits_path), mode="update") as hdul:
        header = hdul[0].header
        dec = float(header.get("CRVAL2", 0.0))
        header["CRVAL1"] = float(header.get("CRVAL1", 0.0)) + east_mas / 3.6e6 / max(math.cos(math.radians(dec)), 1e-6)
        header["CRVAL2"] = dec + north_mas / 3.6e6
        header["HISTORY"] = f"vlbipy: phase centre moved {east_mas:+.3f} mas east, {north_mas:+.3f} mas north"
        hdul.flush()


def locate_source(obs, imagename: str, *, search_fov_mas: float = 1000.0, search_sigma: float = 10.0,
                  recentre_min_beams: float = 10.0, robust: float = 0.0) -> tuple[dict, tuple[float, float]]:
    """Find the source in a wide dirty map and re-centre on it only when it is far from the phase centre.

    The phase centre moves onto the peak when the peak is above ``search_sigma``
    *and* further than ``recentre_min_beams`` synthesized beams (major axis of the
    search map) from the centre. Closer than that the regular map already contains
    the source, and an undetected source keeps the original centre. The map size
    and weighting are put back (``auto_mapsize``, Briggs ``robust``) afterwards.

    Returns
    -------
    (search, shift)
        The :func:`find_peak` report, extended with ``beam_mas``, ``offset_mas``,
        ``offset_beams`` and ``recentred``; and the applied (east, north) shift in
        mas, ``(0, 0)`` when nothing moved. ``({}, (0, 0))`` when the search is off.
    """
    if not search_fov_mas or search_fov_mas <= 0:
        return {}, (0.0, 0.0)
    search = find_peak(obs, fov_mas=float(search_fov_mas), threshold_sigma=float(search_sigma),
                       fits_path=f"{imagename}.search.fits" if imagename else "")
    beam = float(max(obs.estimated_beam[:2]))
    offset = math.hypot(search["x"], search["y"])
    beams = offset / beam if beam > 0 else 0.0
    recentre = bool(search["detected"] and beams > recentre_min_beams)
    search.update(beam_mas=beam, offset_mas=offset, offset_beams=beams, recentred=recentre)
    shift = (search["x"], search["y"]) if recentre else (0.0, 0.0)
    if recentre:
        obs.shift(-shift[0], -shift[1])     # the map contents move with the shift: this centres the peak
        logger.info("difmapy: {} re-centred on the peak, {:+.1f} mas east, {:+.1f} mas north ({:.0f} beams of {:.2f} mas "
                    "from the phase centre)", obs.source, shift[0], shift[1], beams, beam)
    elif search["detected"]:
        logger.info("difmapy: {} is {:.1f} mas ({:.1f} beams of {:.2f} mas) from the phase centre, within {:g} beams; "
                    "keeping the original phase centre", obs.source, offset, beams, beam, recentre_min_beams)
    else:
        logger.info("difmapy: {} not detected in the search map; keeping the original phase centre", obs.source)
    obs.auto_mapsize()
    obs.uvweight(robust=float(robust))
    return search, shift


def image_source(split_ms: str, imagename: str, *, robust_values=(-2.0, 0.0, 2.0), search_fov_mas: float = 0.0,
                 search_sigma: float = 10.0, recentre_min_beams: float = 10.0, average_channels: bool = True,
                 mapsize: int = IMAGE_PIXELS, **imaging_kwargs) -> dict:
    """Fit one Gaussian to an already-calibrated target/check source first, without self-calibration or CLEAN fallback.

    ``average_channels`` (default on) collapses every IF to one channel while loading, as the
    self-calibration path does: visibility fitting then works on a fraction of the visibilities. The price is
    bandwidth smearing of emission far from the phase centre, which is small inside the ~1 arcsec the
    source search covers; turn it off (``[imaging].average_channels = false``) for wider fields.
    The images are made on a map ``mapsize`` pixels wide (``[imaging].mapsize``), of which the FITS
    file keeps the valid inner half.

    With ``search_fov_mas`` > 0 the source is first located (:func:`locate_source`;
    the dirty map goes to ``<imagename>.search.fits``) and the phase centre moved
    onto it when it lies far outside the regular map. The report carries the
    search under ``"search"`` and the applied ``"shift_mas"`` (east, north).
    """
    obs = load(split_ms, average_channels=average_channels)
    search, shift = locate_source(obs, imagename, search_fov_mas=search_fov_mas, search_sigma=search_sigma,
                                  recentre_min_beams=recentre_min_beams)
    if str(imaging_kwargs.get("method", "modelfit")).strip().lower() == "modelfit":
        fit_starting_model(obs)
    obs.auto_mapsize(npix=int(mapsize))
    images = image_all_robust(obs, imagename, robust_values, **imaging_kwargs)
    if any(shift):
        for info in images.values():
            _record_shift(info["fits"], shift[0], shift[1])
    return {"source": obs.source, "split_ms": str(split_ms), "images": images, "search": search,
            "image_model": next(iter(images.values()), {}).get("model", {}), "shift_mas": list(shift)}


#: Revision tag recorded in the image report with the modelfit settings (bump when the defaults change).
MODELFIT_REVISION = 3
#: Peak memory of one imaging job per map pixel: 3.5 GB measured for an 8192-pixel map (the map, its beam and
#: the FFT grid, plus the double-precision copies the noise statistics make).
IMAGE_BYTES_PER_PIXEL = 52
#: Share of the free memory the imaging jobs running together may take.
IMAGE_MEMORY_FRACTION = 0.6


def imaging_settings(img_cfg: dict) -> dict:
    """The difmapy imaging keywords of an ``[imaging]`` table: the ``method`` and its limits."""
    return {"method": str(img_cfg.get("method", "modelfit")).strip().lower(),
            "niter": int(img_cfg.get("niter", 4000)), "gain": float(img_cfg.get("clean_gain", 0.05)),
            "threshold_sigma": float(img_cfg.get("threshold_sigma", 3.0)),
            "max_components": int(img_cfg.get("max_components", 1)),
            "component_sigma": float(img_cfg.get("component_sigma", 5.0)),
            "component_improvement": float(img_cfg.get("component_improvement", 0.01))}


def method_keywords(settings: dict) -> dict:
    """What makes images with ``settings`` what they are: the keywords of :func:`imaging_settings` its method
    takes (all of them for modelfit, with :data:`MODELFIT_REVISION`)."""
    if settings.get("method", "modelfit") == "clean":
        return {key: settings[key] for key in ("method", "niter", "gain", "threshold_sigma") if key in settings}
    return {**settings, "revision": MODELFIT_REVISION}


def image_in_worker(*, log: dict, **kwargs) -> dict:
    """The job run by each subprocess of :func:`run_image_jobs` (the module ``__main__``):
    :func:`image_source`, logging to the parent's run log."""
    from ..logging_utils import attach_logging
    attach_logging(log)
    return image_source(**kwargs)


def _image_subprocess(spec_path: str, result_path: str, threads: int) -> subprocess.Popen:
    """Start one ``image_in_worker`` subprocess (``threads`` CPUs for its numerics) on the spec file."""
    env = dict(os.environ)
    # The shared worker pool pins these to 1, which is what made difmapy crawl: its clean/bayes work is
    # threaded through BLAS, FFTW and ThreadPoolExecutors. Here each concurrent image gets its share.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
                 "VLBIPY_FFT_WORKERS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = str(threads)
    return subprocess.Popen([sys.executable, "-m", "vlbipy.backends.difmap", spec_path, result_path],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_image_jobs(jobs: list[dict], *, max_workers: Optional[int] = None) -> list:
    """Run :func:`image_source` for every job (its keyword arguments), several at once in subprocesses.

    As many at a time as there are CPUs and as fit together in :data:`IMAGE_MEMORY_FRACTION` of the
    free memory (:data:`IMAGE_BYTES_PER_PIXEL` per map pixel), at most ``max_workers``; each running
    job gets all usable CPUs for its numerical work. These do not go through the
    worker pool: its processes pin the numerical libraries to one thread, which difmapy needs.
    One job, or room for only one, runs here.

    Returns
    -------
    list
        One entry per job, in order: its report, or the exception it raised (a failed image must not
        lose the others).
    """
    import pickle
    import tempfile

    from ..logging_utils import logging_setup
    from ..resources import available_memory, usable_cpus
    if not jobs:
        return []
    per_job = IMAGE_BYTES_PER_PIXEL * max(int(job.get("mapsize", IMAGE_PIXELS)) for job in jobs) ** 2
    free = available_memory()
    fit = max(1, int(free * IMAGE_MEMORY_FRACTION // per_job)) if free is not None else len(jobs)
    workers = max(1, min(len(jobs), usable_cpus(), fit, int(max_workers) if max_workers else len(jobs)))
    if workers > 1:
        threads = usable_cpus()
        logger.info("difmapy: imaging {} source(s), {} at a time, {} thread(s) each ({:.1f} GB each, {} free)",
                    len(jobs), workers, threads, per_job / 1e9,
                    f"{free / 1e9:.1f} GB" if free is not None else "unknown")
        log = logging_setup()
        results: list = [None] * len(jobs)
        for first in range(0, len(jobs), workers):
            running = []
            for index, job in enumerate(jobs[first:first + workers], start=first):
                spec = tempfile.NamedTemporaryFile("wb", prefix="vlbipy_img_", suffix=".pickle", delete=False)
                pickle.dump(dict(job, log=log), spec)
                spec.close()
                result_path = spec.name + ".result"
                running.append((index, spec.name, result_path,
                                _image_subprocess(spec.name, result_path, threads)))
            for index, spec_path, result_path, proc in running:
                _, stderr = proc.communicate()
                try:
                    with open(result_path, "rb") as handle:
                        results[index] = pickle.load(handle)
                except OSError:
                    errors = stderr.decode(errors="replace").strip().splitlines() if stderr else []
                    results[index] = RuntimeError(f"imaging subprocess failed "
                                                  f"({errors[-1] if errors else f'exit status {proc.returncode}'})")
                for path in (spec_path, result_path):
                    os.unlink(path) if os.path.exists(path) else None
        return results
    results = []
    for job in jobs:
        try:
            results.append(image_source(**job))
        except Exception as exc:  # noqa: BLE001 - reported per source by the caller
            results.append(exc)
    return results


if __name__ == "__main__":
    # The one-image runner of :func:`run_image_jobs`: a clean process whose numerical libraries get a
    # fair thread share, unlike the worker pool that pins them to one thread.
    import pickle

    spec = pickle.load(open(sys.argv[1], "rb"))
    result = image_in_worker(**spec)
    with open(sys.argv[2], "wb") as handle:
        pickle.dump(result, handle)
