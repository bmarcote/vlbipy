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
5. **images** at the requested Briggs robust values, written as FITS: the source
   is modelled with circular Gaussians (``modelfit``), what they leave is CLEANed
   in windows, and the one model is restored at every robust value
   (:func:`model_images`; ``[imaging].method = "clean"`` CLEANs each weighting
   on its own instead, :func:`clean_image`).

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
#: Robust value of the maps the modelfit-first imaging works on (:func:`model_images`): the residuals the
#: next Gaussian starts on and the CLEAN of what the Gaussians leave. The model is the same at every weighting.
MODEL_ROBUST = 0.0
#: The first Gaussian starts at the peak of the naturally weighted map, the most sensitive one (as the search).
SEED_ROBUST = 2.0
#: Later Gaussians start within this many beams (at :data:`MODEL_ROBUST`) of the first, plus its size.
REGION_BEAMS = 3.0
#: Half-width in beams of a CLEAN window, around a Gaussian (plus its size) or a residual peak.
WINDOW_BEAMS = 4.0
#: Most CLEAN windows opened on residual emission away from the Gaussians.
MAX_WINDOWS = 10
#: Parameters a Gaussian component fits: a circular Gaussian of free flux, position and size.
GAUSS_FREE = ("flux", "pos", "major")
#: Revision of the modelfit imaging, recorded with the images: a change that alters them bumps it, so that a
#: campaign makes its "before" images again rather than compare them with images made another way.
MODELFIT_REVISION = 2
#: Peak memory of one imaging job per map pixel: 3.5 GB measured for an 8192-pixel map (the map, its beam and
#: the FFT grid, plus the double-precision copies the noise statistics make).
IMAGE_BYTES_PER_PIXEL = 52
#: Share of the free memory the imaging jobs running together may take.
IMAGE_MEMORY_FRACTION = 0.6
#: The worker job of :func:`run_image_jobs`, as :mod:`vlbipy.solvers.workers` addresses it.
IMAGE_JOB = "vlbipy.backends.difmap:image_in_worker"
#: Pixels per resolution element in the wide dirty map searched for a target.
SEARCH_OVERSAMPLE = 3.0
#: Largest side (pixels) of that map; the cell grows instead when the field needs more.
SEARCH_MAX_PIXELS = 16384


#: Width in pixels of the map the final images are made on (difmapy's own default is 4096). Only the inner
#: half along each axis is valid and kept in the FITS file, so the images are IMAGE_PIXELS / 2 wide on disk.
IMAGE_PIXELS = 8192


def _load_difmapy():
    """Import difmapy lazily so vlbipy imports without it; plots go off-screen."""
    import os
    os.environ.setdefault("DIFMAPY_PLOT_MODE", "inline")
    try:
        import difmapy  # noqa: WPS433 - optional dependency
    except ImportError as exc:
        # Say why: an installed difmapy whose compiled core does not load (built for another platform, a
        # missing library) is not the same problem as a missing package.
        raise BackendError(f"imaging and self-calibration need difmapy, which could not be imported ({exc}); "
                           "install it with pip install vlbipy[difmap] (or rebuild it: maturin develop "
                           "--release in its checkout)") from exc
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
    """Fit a single circular Gaussian (flux, position, size free) at the phase centre.

    This is always the starting model: with no model difmapy seeds exactly that
    component at the residual peak (``seed_model``) and ``modelfit`` refines it.
    """
    obs.clrmod()
    result = obs.modelfit(quiet=True)
    comp = result["components"][0] if result.get("components") else {}
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
        report = trial.selfcal(phase=True, amp=False, solint=solint, quiet=True)
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
        obs = trial
        if refit:
            obs.modelfit(quiet=True)
    return obs, rounds


def amplitude_calibration(obs, prefix: str, parent_ms: str, *, models=("clean", "gauss1", "gauss2", "gauss3"),
                          prior_sigma: float = 0.1, outlier_sigma: float = 1.0,
                          workers: Optional[int] = None) -> dict:
    """Bayesian station amplitude calibration; writes ``<prefix>.json``, ``.png`` and the CASA table.

    The table refers to ``parent_ms`` (the multi-source measurement set), so
    CASA can apply the corrections to every field. Returns the JSON-able summary
    with the table path under ``"caltable"``.
    """
    result = obs.bayes_gscale(models=tuple(models), prior_sigma=prior_sigma, outlier_sigma=outlier_sigma,
                              prefix=str(prefix),
                              outformat="CASA", ms=str(parent_ms), workers=workers, plot=False, quiet=True)
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
    info = obs.savecaltable(str(path), outformat="CASA", ms=str(parent_ms), since=since, quiet=True)
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
    obs.invert()
    done = 0
    rms = float(obs.imstat().get("rms", 0.0))
    stopped = ""
    while done < niter:
        checkpoint = obs.copy()
        obs.clean(min(batch, niter - done), gain, quiet=True)
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
    info = write_restored(obs, fits_path, robust, method="clean", ncomponents=done,
                          stopped=stopped or "niter reached")
    logger.info("difmapy clean robust={:+g}: peak {:.4g} Jy/beam, rms {:.3g}, DR {:.0f}, {:.4g} Jy in {} comps ({}) -> {}",
                robust, info["peak"], info["rms"], info["dynamic_range"], info["model_flux"], done, info["stopped"],
                Path(fits_path).name)
    return obs, info


def write_restored(obs, fits_path: str, robust: float, **extra) -> dict:
    """Restore the current model at ``robust`` (with its residuals) and write it as a FITS image.

    Returns the image statistics: ``fits``, ``robust``, ``peak`` (Jy/beam), ``rms`` of the residual,
    ``dynamic_range``, ``model_flux`` and ``beam``, plus ``extra``.
    """
    obs.uvweight(robust=float(robust))
    obs.invert()
    obs.restore()
    obs.wmap(str(fits_path), overwrite=True)
    _crop_to_valid(fits_path, obs.valid_slice)
    peak = float(np.nanmax(np.asarray(obs.restored_map)[obs.valid_slice]))
    noise = float(obs.imstat().get("rms", 0.0))
    beam = obs.estimated_beam
    return {"fits": str(fits_path), "robust": float(robust), "peak": peak, "rms": noise,
            "dynamic_range": peak / noise if noise > 0 else 0.0, "model_flux": float(obs.model_flux),
            "beam": [float(b) for b in np.atleast_1d(beam)] if beam is not None else [], **extra}


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


def set_model(obs, components: list[dict]) -> None:
    """Make ``components`` (dicts as ``obs.model`` lists them, free parameters included) the model of ``obs``.

    Undoes a fit or a CLEAN batch without copying the observation, whose maps take a gigabyte each at
    8192 pixels: modelfit and CLEAN change nothing else.
    """
    obs.clrmod()
    for comp in components:
        obs.addcmp(comp["flux"], comp["x"], comp["y"], type=comp["type"], major=comp["major"], ratio=comp["ratio"],
                   phi=comp["phi"], freq0=comp["freq0"], spcind=comp["spcind"], free=int(comp.get("freepar", 0)))


def _map_offsets(obs, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """East and north offsets (mas) of the columns and rows of a map of ``shape``, as ``peak_offset`` counts them."""
    ny, nx = shape
    east, _, _, north = obs.extent
    return ((np.arange(nx) - nx / 2) * east / (nx / 2), (np.arange(ny) - ny / 2) * north / (ny / 2))


def fit_components(obs, *, max_components: int = 10, component_sigma: float = 5.0,
                   component_improvement: float = 0.01) -> dict:
    """Model the source with circular Gaussians, added one at a time (difmap ``modelfit``); return a report.

    The first component starts at the peak of the naturally weighted map, as the source search does; each
    later one at the brightest residual (robust :data:`MODEL_ROBUST`) within :data:`REGION_BEAMS` beams of
    the first plus its size, and all of them are fitted again together. A component is kept when

    * the fit improves by ``component_sigma``: the drop of chi-squared, in units of the reduced chi-squared
      after the fit, is at least ``component_sigma**2``;
    * after the first, it also removes ``component_improvement`` of the chi-squared. On a bright source the
      calibration errors of the most sensitive baselines make any extra component pass the first test;
    * every component stays physical: positive flux, no larger than the region, its centre inside it.

    The fitting stops at the first component that fails (it is undone) or at ``max_components``. The kept
    ones are fixed afterwards, so nothing re-fits them by accident.

    Later components stay near the first on purpose. Further out the residual peaks of a bright source are
    mostly the sidelobes of its calibration errors (V589: one baseline, AT-PA, laying stripes 26 mas apart
    over the map), which the fit chases into Janskys of spurious flux. Emission away from the source (a
    jet knot further than that) is left to :func:`clean_residual`, which gives it a window of its own when
    it stands out of the noise.

    Returns
    -------
    dict
        ``components`` (flux in Jy, x/y/major in mas), ``flux``, ``rchisq`` of the final model (NaN without
        one) and why the fitting ``stopped``.
    """
    obs.clrmod()
    chisq = float(obs.moddif()["chisq"])
    rchisq = float("nan")
    region: Optional[tuple] = None
    stopped = f"{max_components} components"
    for number in range(1, int(max_components) + 1):
        kept = obs.model
        if region is None:
            obs.uvweight(robust=SEED_ROBUST)
            obs.invert()
            obs.seed_model(type="gauss", free=GAUSS_FREE)
        else:
            obs.uvweight(robust=MODEL_ROBUST)
            obs.invert()
            residual = np.asarray(obs.dmap)
            xs, ys = _map_offsets(obs, residual.shape)
            x0, y0, radius = region
            near = (xs[None, :] - x0) ** 2 + (ys[:, None] - y0) ** 2 <= radius ** 2
            (x, y), peak = obs.peak_offset(np.where(near, residual, -np.inf))
            obs.addcmp(peak, x, y, type="gauss", free=list(GAUSS_FREE))
        try:
            fit = obs.modelfit(quiet=True)
        except Exception as exc:  # noqa: BLE001 - a fit that cannot be done ends the model where it is
            set_model(obs, kept)
            stopped = f"component {number} could not be fitted ({exc})"
            break
        new = float(obs.moddif()["chisq"])
        sigma = math.sqrt(max(chisq - new, 0.0) / max(float(fit["rchisq"]), 1.0))
        if region is None:
            first = obs.model[0]
            obs.uvweight(robust=MODEL_ROBUST)
            region = (first["x"], first["y"], abs(first["major"]) + REGION_BEAMS * float(max(obs.estimated_beam[:2])))
        x0, y0, radius = region
        physical = all(np.isfinite(c["flux"]) and c["flux"] > 0 and np.isfinite(c["major"])
                       and abs(c["major"]) <= radius and math.hypot(c["x"] - x0, c["y"] - y0) <= radius
                       for c in obs.model)
        if not physical:
            stopped = f"component {number} came out unphysical"
        elif sigma < component_sigma:
            stopped = f"component {number} at {sigma:.1f} sigma"
        elif number > 1 and chisq - new < component_improvement * chisq:
            stopped = f"component {number} removes {(chisq - new) / chisq:.2%} of the chi-squared"
        if not physical or stopped.startswith(f"component {number} "):
            set_model(obs, kept)
            break
        chisq, rchisq = new, float(fit["rchisq"])
    set_model(obs, [dict(c, freepar=0) for c in obs.model])
    components = [{key: float(c[key]) for key in ("flux", "x", "y", "major")} for c in obs.model]
    return {"components": components, "flux": float(sum(c["flux"] for c in components)), "rchisq": rchisq,
            "stopped": stopped}


def clean_residual(obs, *, clean_sigma: float = 5.0, threshold_sigma: float = 3.0, niter: int = 4000,
                   gain: float = 0.05, batch: int = 100) -> dict:
    """CLEAN what the model of ``obs`` leaves, in windows: around the Gaussians and on residual emission.

    The first windows sit on the Gaussians (their size plus :data:`WINDOW_BEAMS` beams), and CLEAN runs in
    them at robust :data:`MODEL_ROBUST` as :func:`clean_image` does: in batches, until the residual peak in
    the windows is below ``threshold_sigma`` times the residual rms while that rms has stopped falling, the
    rms rises (the batch is undone), or ``niter`` components. A window where the first batch finds nothing
    (the rms does not fall and no peak stands above ``clean_sigma`` times it) gets no components. Then the
    brightest residual outside the windows is looked at: one above ``clean_sigma`` times the rms of the
    residual gets a window of its own, up to :data:`MAX_WINDOWS`. That rms includes the artefacts of the
    calibration errors, on purpose: a peak that does not stand out of them is most likely one of them (V589A,
    against the clipped noise: 0.7 Jy of CLEAN components on the stripes of its bad baselines, for a 1 Jy
    source). A source without a model starts there.

    Returns
    -------
    dict
        ``clean_components``, ``clean_flux`` (Jy), ``windows`` and why it stopped (``clean_stopped``).
    """
    obs.uvweight(robust=MODEL_ROBUST)
    obs.clear_windows()
    obs.invert()
    beam = float(max(obs.estimated_beam[:2]))
    residual = np.asarray(obs.dmap)
    xs, ys = _map_offsets(obs, residual.shape)
    valid = obs.valid_slice
    inside = np.zeros(residual.shape, dtype=bool)
    windows: list[tuple] = []

    def add_window(x: float, y: float, half: float) -> None:
        nonlocal inside
        obs.add_window(x - half, x + half, y - half, y + half)
        inside |= (np.abs(xs[None, :] - x) <= half) & (np.abs(ys[:, None] - y) <= half)
        windows.append((x, y, half))

    def emission_elsewhere() -> Optional[tuple]:
        """Position of the brightest residual outside the windows when it stands above ``clean_sigma``."""
        if len(windows) >= MAX_WINDOWS:
            return None
        rest = np.where(inside, -np.inf, residual)[valid]
        iy, ix = np.unravel_index(int(np.argmax(rest)), rest.shape)
        noise = float(obs.imstat().get("rms", 0.0))
        if not noise > 0 or rest[iy, ix] < clean_sigma * noise:
            return None
        return float(xs[ix + valid[1].start]), float(ys[iy + valid[0].start])

    gaussians = obs.model
    for comp in gaussians:
        add_window(comp["x"], comp["y"], abs(comp["major"]) + WINDOW_BEAMS * beam)
    if not windows and (spot := emission_elsewhere()):
        add_window(*spot, WINDOW_BEAMS * beam)
    rms = float(obs.imstat().get("rms", 0.0))
    done, fresh, stopped = 0, True, ""
    while windows and done < niter:
        before = obs.model
        step = min(batch, niter - done)
        obs.clean(step, gain, quiet=True)
        new_rms = float(obs.imstat().get("rms", 0.0))
        residual = np.asarray(obs.dmap)
        peak = float(np.max(np.where(inside, residual, -np.inf)[valid]))
        falling = rms > 0 and (rms - new_rms) > RMS_CONVERGENCE * rms
        rose = new_rms > rms * (1.0 + DIVERGENCE_TOLERANCE)
        if rose or (fresh and not falling and peak < clean_sigma * new_rms):
            # Worse, or nothing in the new window the model had not taken: undo the batch.
            set_model(obs, before)
            obs.invert()
            residual = np.asarray(obs.dmap)
            stopped = f"residual rms rose ({rms:.3g} -> {new_rms:.3g})" if rose else "nothing left to clean"
        else:
            done += step
            rms, fresh = new_rms, False
            if falling or peak >= threshold_sigma * rms:
                continue
            stopped = f"residual peak below {threshold_sigma:g} sigma"
        spot = emission_elsewhere()
        if spot is None:
            break
        add_window(*spot, WINDOW_BEAMS * beam)
        fresh = True
    else:
        if windows:
            stopped = "niter reached"
    obs.clear_windows()
    return {"clean_components": done, "clean_flux": float(obs.model_flux) - sum(c["flux"] for c in gaussians),
            "windows": len(windows), "clean_stopped": stopped or "no emission to clean"}


def model_images(obs, imagename: str, robust_values, *, fit: bool = True, max_components: int = 10,
                 component_sigma: float = 5.0, component_improvement: float = 0.01, clean_sigma: float = 5.0,
                 threshold_sigma: float = 3.0, niter: int = 4000, gain: float = 0.05) -> tuple[dict, dict]:
    """Model the source with Gaussians, CLEAN what they leave, and restore that model at every robust value.

    The default imaging (``[imaging].method = "modelfit"``): :func:`fit_components`, then
    :func:`clean_residual`, then one FITS image per robust value (``<imagename>.robust<r>.fits``), all of
    the same model, each with its own beam and residuals. ``fit=False`` (a source the search did not
    detect) gives the dirty maps, with neither Gaussians nor CLEAN.

    Returns
    -------
    (images, model)
        ``{robust: image statistics}`` (see :func:`write_restored`) and the report of the model: what
        :func:`fit_components` and :func:`clean_residual` return, together.
    """
    if fit:
        model = fit_components(obs, max_components=max_components, component_sigma=component_sigma,
                               component_improvement=component_improvement)
        model.update(clean_residual(obs, clean_sigma=clean_sigma, threshold_sigma=threshold_sigma, niter=niter,
                                    gain=gain))
        logger.info("difmapy modelfit: {} Gaussian(s), {:.4g} Jy, reduced chi-squared {:.4g} ({}); CLEAN of the "
                    "residual: {} component(s), {:.4g} Jy in {} window(s) ({})", len(model["components"]),
                    model["flux"], model["rchisq"], model["stopped"], model["clean_components"],
                    model["clean_flux"], model["windows"], model["clean_stopped"])
    else:
        obs.clrmod()
        model = {"components": [], "flux": 0.0, "rchisq": float("nan"), "stopped": "not detected: no model",
                 "clean_components": 0, "clean_flux": 0.0, "windows": 0, "clean_stopped": "not detected"}
        logger.info("difmapy: {} not detected; its images are the dirty maps", obs.source)
    images = {}
    for robust in robust_values:
        fits_path = f"{imagename}.robust{float(robust):g}.fits"
        images[float(robust)] = info = write_restored(
            obs, fits_path, float(robust), method="modelfit", gaussians=len(model["components"]),
            ncomponents=model["clean_components"], stopped=f"{model['stopped']}; {model['clean_stopped']}")
        logger.info("difmapy image robust={:+g}: peak {:.4g} Jy/beam, rms {:.3g}, DR {:.0f}, {:.4g} Jy -> {}",
                    float(robust), info["peak"], info["rms"], info["dynamic_range"], info["model_flux"],
                    Path(fits_path).name)
    return images, model


def image_all_robust(obs, imagename: str, robust_values, *, method: str = "modelfit", fit: bool = True,
                     niter: int = 4000, gain: float = 0.05, threshold_sigma: float = 3.0, **model_kwargs
                     ) -> tuple[dict[float, dict], dict]:
    """Images at every robust value, ``imagename`` + ``.robust<r>.fits``, with the ``method`` of ``[imaging]``.

    ``"modelfit"`` (default): :func:`model_images`, one model restored at every weighting; ``model_kwargs``
    are its limits. ``"clean"``: :func:`clean_image` at each robust value on its own (``fit`` does not apply).

    Returns
    -------
    (images, model)
        ``{robust: image statistics}`` and the model report (empty for ``"clean"``).
    """
    if method == "modelfit":
        return model_images(obs, imagename, robust_values, fit=fit, niter=niter, gain=gain,
                            threshold_sigma=threshold_sigma, **model_kwargs)
    if method != "clean":
        raise ValueError(f"unknown imaging method {method!r}: use 'modelfit' or 'clean'")
    images = {}
    for robust in robust_values:
        tag = f"robust{float(robust):g}"
        obs, images[float(robust)] = clean_image(obs, f"{imagename}.{tag}.fits", robust=float(robust),
                                                 niter=niter, gain=gain, threshold_sigma=threshold_sigma)
    return images, {}


def imaging_settings(img_cfg: dict) -> dict:
    """The difmapy imaging keywords of an ``[imaging]`` table: the ``method`` and its limits."""
    return {"method": str(img_cfg.get("method", "modelfit")).strip().lower(),
            "niter": int(img_cfg.get("niter", 4000)), "gain": float(img_cfg.get("clean_gain", 0.05)),
            "threshold_sigma": float(img_cfg.get("threshold_sigma", 3.0)),
            "max_components": int(img_cfg.get("max_components", 10)),
            "component_sigma": float(img_cfg.get("component_sigma", 5.0)),
            "component_improvement": float(img_cfg.get("component_improvement", 0.01)),
            "clean_sigma": float(img_cfg.get("clean_residual_sigma", 5.0))}


def method_keywords(settings: dict) -> dict:
    """What makes images with ``settings`` what they are: the keywords of :func:`imaging_settings` its method
    takes (all of them for modelfit, with :data:`MODELFIT_REVISION`)."""
    if settings.get("method", "modelfit") == "clean":
        return {key: settings[key] for key in ("method", "niter", "gain", "threshold_sigma") if key in settings}
    return {**settings, "revision": MODELFIT_REVISION}


def calibrate_source(split_ms: str, parent_ms: str, prefix: str, *, robust_values=(-2.0, 0.0, 2.0),
                     solints: Optional[list[str]] = None, min_improvement: float = 0.002,
                     max_bad_fraction: float = 0.25, bayes_models=("clean", "gauss1", "gauss2", "gauss3"),
                     prior_sigma: float = 0.1, outlier_sigma: float = 1.0, workers: Optional[int] = None,
                     imagename: Optional[str] = None,
                     search_fov_mas: float = 0.0, search_sigma: float = 10.0, recentre_min_beams: float = 10.0,
                     mapsize: int = IMAGE_PIXELS, **clean_kwargs) -> dict:
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
    search, shift = locate_source(obs, imagename or prefix, search_fov_mas=search_fov_mas, search_sigma=search_sigma,
                                  recentre_min_beams=recentre_min_beams)
    model = fit_starting_model(obs)
    ladder = solints or selfcal_ladder(scan_lengths(obs))
    snapshot = obs.gain_snapshot()
    obs, rounds = phase_selfcal(obs, ladder, min_improvement=min_improvement, max_bad_fraction=max_bad_fraction)
    phase_table = export_phase_table(obs, f"{prefix}.phase", parent_ms, snapshot) if any(r["accepted"] for r in rounds) \
        else {}
    amplitude = amplitude_calibration(obs, f"{prefix}.amp", parent_ms, models=bayes_models, prior_sigma=prior_sigma,
                                      outlier_sigma=outlier_sigma, workers=workers)
    gains = write_gain_record(obs, f"{prefix}.gains.json", amplitude)
    obs.auto_mapsize(npix=int(mapsize))
    images, image_model = image_all_robust(obs, imagename or prefix, robust_values, **clean_kwargs)
    if any(shift):
        for info in images.values():
            _record_shift(info["fits"], shift[0], shift[1])
    report = {"source": obs.source, "split_ms": str(split_ms), "model": model.get("components", []),
              "search": search, "shift_mas": list(shift),
              "model_rchisq": model.get("rchisq"), "ladder": ladder, "rounds": rounds,
              "phase_table": phase_table.get("path", ""), "amplitude": amplitude, "images": images,
              "image_model": image_model,
              "station_gains": {k: float(v) for k, v in obs.station_gains().items()},
              "station_gains_per_if": gains["gain_per_if"], "gains_file": f"{prefix}.gains.json"}
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def joint_model(combined_ms: str, prefix: str, *, robust_values=(-2.0, 0.0, 2.0), model_robust: float = 0.0,
                solints: Optional[list[str]] = None, min_improvement: float = 0.002,
                max_bad_fraction: float = 0.25, mapsize: int = IMAGE_PIXELS, method: str = "modelfit",
                **image_kwargs) -> dict:
    """Model a calibrator from the data of several epochs together; write ``<prefix>.mod``.

    ``combined_ms`` is the concatenation of the calibrated per-epoch splits of
    one source, brought to a common flux level. The joint uv coverage is what
    one epoch lacks: structure that no single epoch constrains is fixed here.
    The sequence is the calibrator one (single Gaussian, phase self-cal ladder)
    without amplitude calibration - the stations kept their names but not
    their gains from one epoch to the next - followed by the images. The model
    is what they are made of: the Gaussians and the CLEAN of their residual
    (``method="modelfit"``, :func:`model_images`), or with ``method="clean"``
    the CLEAN components of the ``model_robust`` image.

    Returns a JSON-able report with ``model`` (the ``.mod`` file), ``model_flux``,
    the self-cal ``rounds``, the ``images`` (one per robust value) and, for
    modelfit, the ``image_model`` report.
    """
    obs = load(combined_ms, average_channels=True)
    fit_starting_model(obs)
    ladder = solints or selfcal_ladder(scan_lengths(obs))
    obs, rounds = phase_selfcal(obs, ladder, min_improvement=min_improvement, max_bad_fraction=max_bad_fraction)
    obs.auto_mapsize(npix=int(mapsize))
    image_model: dict = {}
    if method == "modelfit":
        images, image_model = image_all_robust(obs, prefix, robust_values, method=method, **image_kwargs)
    else:
        # The model image goes last so that its CLEAN components are the ones left in the observation.
        order = [float(r) for r in robust_values if float(r) != float(model_robust)] + [float(model_robust)]
        clean_kwargs = {key: image_kwargs[key] for key in ("niter", "gain", "threshold_sigma") if key in image_kwargs}
        images = {}
        for robust in order:
            obs, images[robust] = clean_image(obs, f"{prefix}.robust{robust:g}.fits", robust=robust, **clean_kwargs)
    model_path = f"{prefix}.mod"
    obs.wmodel(model_path)
    report = {"source": obs.source, "combined_ms": str(combined_ms), "model": model_path,
              "model_flux": float(obs.model_flux), "ladder": ladder, "rounds": rounds,
              "images": {robust: images[robust] for robust in sorted(images)}, "image_model": image_model}
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("difmapy joint model of {}: {:.4g} Jy in {} -> {}", obs.source, report["model_flux"],
                f"{len(image_model['components'])} Gaussian(s) and {image_model['clean_components']} CLEAN "
                f"component(s)" if image_model else "CLEAN components", Path(model_path).name)
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
    sequence; then the station amplitudes (one factor per station and IF) are
    solved against the same model by ``bayes_gscale``, as in the epoch's own
    amplitude step: the median station holds the flux scale of every IF, a
    station within ``prior_sigma`` of it is left alone and one clearly off is
    corrected (``<prefix>.amp.json`` and ``.png``). The model is not rebuilt
    without each station, so how much a station's gain depends on it comes
    from outside: ``model_sigma`` (station -> per-IF log-gain uncertainty, the
    epoch's own amplitude step's). The gains accumulated here are exported as
    one CASA table for ``parent_ms``.

    Returns a JSON-able report: ``caltable`` ("" when nothing was accepted),
    ``rounds``, ``amplitude`` (station -> correction, median over IFs, for the
    stations corrected by more than 0.5%), ``amplitude_files`` and the fit
    ``chisq_before`` / ``chisq_after``.
    """
    obs = load(split_ms, average_channels=True)
    obs.clrmod()
    obs.rmodel(str(model_path))
    start = float(obs.moddif()["chisq"])
    snapshot = obs.gain_snapshot()
    ladder = solints or selfcal_ladder(scan_lengths(obs))
    obs, rounds = phase_selfcal(obs, ladder, min_improvement=min_improvement, max_bad_fraction=max_bad_fraction,
                                refit=False)
    gains: dict = {}
    amplitude_files: dict = {}
    if amplitude:
        # "current": the fixed model, so every station is measured against the structure all epochs constrain.
        result = obs.bayes_gscale(models=("current",), prior_sigma=prior_sigma, outlier_sigma=outlier_sigma,
                                  model_sigma=model_sigma, workers=workers, plot=False, quiet=True)
        amplitude_files = {key: str(path) for key, path in result.write(f"{prefix}.amp", quiet=True).items()}
        gains = {row["station"]: float(row["correction"]) for row in result.station_table()
                 if row["has_data"] and abs(math.log(row["correction"])) > 0.005}
        logger.info("difmapy bayes_gscale against the joint model: {}",
                    ", ".join(f"{name} x{value:.3f}" for name, value in gains.items()) or "no station off")
    table = ""
    if gains or any(r["accepted"] for r in rounds):
        info = obs.savecaltable(f"{prefix}", outformat="CASA", ms=str(parent_ms), since=snapshot, quiet=True)
        table = str(info.get("path", f"{prefix}"))
    report = {"source": obs.source, "split_ms": str(split_ms), "model": str(model_path), "caltable": table,
              "ladder": ladder, "rounds": rounds, "amplitude": gains, "amplitude_files": amplitude_files,
              "chisq_before": start, "chisq_after": float(obs.moddif()["chisq"])}
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
    obs.invert()
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
                 mapsize: int = IMAGE_PIXELS, **image_kwargs) -> dict:
    """Image an already-calibrated source (target / check source), no self-cal (see :func:`image_all_robust`).

    ``average_channels`` (default on) collapses every IF to one channel while loading, as the
    self-calibration path does: the imaging then works on a fraction of the visibilities. The price is
    bandwidth smearing of emission far from the phase centre, which is small inside the ~1 arcsec the
    source search covers; turn it off (``[imaging].average_channels = false``) for wider fields.
    The images are made on a map ``mapsize`` pixels wide (``[imaging].mapsize``), of which the FITS
    file keeps the valid inner half.

    With ``search_fov_mas`` > 0 the source is first located (:func:`locate_source`;
    the dirty map goes to ``<imagename>.search.fits``) and the phase centre moved
    onto it when it lies far outside the regular map. A source the search does not
    detect gets no model (modelfit imaging): its images are the dirty maps. The report
    carries the search under ``"search"``, the applied ``"shift_mas"`` (east, north)
    and the model under ``"model"``.
    """
    obs = load(split_ms, average_channels=average_channels)
    search, shift = locate_source(obs, imagename, search_fov_mas=search_fov_mas, search_sigma=search_sigma,
                                  recentre_min_beams=recentre_min_beams)
    obs.auto_mapsize(npix=int(mapsize))
    images, model = image_all_robust(obs, imagename, robust_values, fit=bool(search.get("detected", True)),
                                     **image_kwargs)
    if any(shift):
        for info in images.values():
            _record_shift(info["fits"], shift[0], shift[1])
    return {"source": obs.source, "split_ms": str(split_ms), "images": images, "search": search,
            "shift_mas": list(shift), "model": model}


def image_in_worker(*, log: dict, **kwargs) -> dict:
    """Worker job of :func:`run_image_jobs`: :func:`image_source`, logging to the parent's run log."""
    from ..logging_utils import attach_logging
    attach_logging(log)
    return image_source(**kwargs)


def run_image_jobs(jobs: list[dict], *, max_workers: Optional[int] = None) -> list:
    """Run :func:`image_source` for every job (its keyword arguments), several at once in worker processes.

    difmapy works on one core, and an epoch has several sources to image, so they are imaged side by
    side in the processes of :mod:`vlbipy.solvers.workers`: as many at a time as there are CPUs and as
    fit together in :data:`IMAGE_MEMORY_FRACTION` of the free memory (:data:`IMAGE_BYTES_PER_PIXEL`
    per map pixel), at most ``max_workers``. One job, or room for only one, runs here.

    Returns
    -------
    list
        One entry per job, in order: its report, or the exception it raised (a failed image must not
        lose the others).
    """
    from ..logging_utils import logging_setup
    from ..resources import available_memory, usable_cpus
    from ..solvers.workers import submit_jobs
    if not jobs:
        return []
    per_job = IMAGE_BYTES_PER_PIXEL * max(int(job.get("mapsize", IMAGE_PIXELS)) for job in jobs) ** 2
    free = available_memory()
    fit = max(1, int(free * IMAGE_MEMORY_FRACTION // per_job)) if free is not None else len(jobs)
    workers = max(1, min(len(jobs), usable_cpus(), fit, int(max_workers) if max_workers else len(jobs)))
    if workers > 1:
        logger.info("difmapy: imaging {} source(s), {} at a time ({:.1f} GB each, {} free)", len(jobs), workers,
                    per_job / 1e9, f"{free / 1e9:.1f} GB" if free is not None else "unknown")
        log = logging_setup()
        try:
            futures = submit_jobs(IMAGE_JOB, [dict(job, log=log) for job in jobs], workers)
        except Exception as exc:  # noqa: BLE001 - the images can still be made here, one at a time
            logger.warning("difmapy: the worker processes could not start ({}); imaging one source at a time", exc)
        else:
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - WorkerError: reported per source by the caller
                    results.append(exc)
            return results
    results = []
    for job in jobs:
        try:
            results.append(image_source(**job))
        except Exception as exc:  # noqa: BLE001 - reported per source by the caller
            results.append(exc)
    return results
