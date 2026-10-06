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
5. **CLEAN images** at the requested Briggs robust values, written as FITS.

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

from ..logging_utils import get_logger

logger = get_logger()

#: Default phase-only self-cal ladder (difmapy solint strings), long to short.
DEFAULT_SOLINTS = ("30min", "10min", "5min", "scan")
#: Added at the end of the ladder when the median scan is longer than this many seconds.
LONG_SCAN_SEC = 300.0
SHORT_SOLINT = "2min"
#: CLEAN stops (and undoes its last batch) when the residual rms grows by more than this fraction.
DIVERGENCE_TOLERANCE = 0.02
#: Pixels per resolution element in the wide dirty map searched for a target.
SEARCH_OVERSAMPLE = 3.0
#: Largest side (pixels) of that map; the cell grows instead when the field needs more.
SEARCH_MAX_PIXELS = 16384


def _load_difmapy():
    """Import difmapy lazily so vlbipy imports without it; plots go off-screen."""
    import os
    os.environ.setdefault("DIFMAPY_PLOT_MODE", "inline")
    import difmapy  # noqa: WPS433 - optional dependency
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
                  max_bad_fraction: float = 0.25) -> tuple[object, list[dict]]:
    """Phase-only self-calibration down the ``solints`` ladder with accept/revert per step.

    A step is accepted when the weighted chi-squared of the model fit drops by at
    least ``min_improvement`` (fractional; small on purpose, a 1% gain is worth keeping) and at most ``max_bad_fraction`` of
    the solution bins failed; the model is then re-fitted. A rejected step is
    reverted by discarding the attempt (the observation is cloned first) and
    ends the ladder: shorter intervals only add noise once the data stop asking
    for them.

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
        obs.modelfit(quiet=True)
    return obs, rounds


def amplitude_calibration(obs, prefix: str, parent_ms: str, *, models=("clean", "gauss1", "gauss2", "gauss3"),
                          prior_sigma: float = 0.1, workers: Optional[int] = None) -> dict:
    """Bayesian station amplitude calibration; writes ``<prefix>.json``, ``.png`` and the CASA table.

    The table refers to ``parent_ms`` (the multi-source measurement set), so
    CASA can apply the corrections to every field. Returns the JSON-able summary
    with the table path under ``"caltable"``.
    """
    result = obs.bayes_gscale(models=tuple(models), prior_sigma=prior_sigma, prefix=str(prefix),
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
    or ``niter`` components. Returns peak / rms / dynamic range / model flux /
    beam and the FITS path.
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
        done += batch
        rms = new_rms
        # The residual rms shrinks as the sidelobes are removed, so the stopping level
        # is measured on the current residual, not on the dirty map.
        if rms > 0 and abs(float(stats.get("max", 0.0))) < threshold_sigma * rms:
            stopped = f"residual peak below {threshold_sigma:g} sigma"
            break
    obs.restore()
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


def image_all_robust(obs, imagename: str, robust_values, **clean_kwargs) -> dict[float, dict]:
    """CLEAN images at every robust value; ``imagename`` gets ``.robust<r>.fits`` appended."""
    images = {}
    for robust in robust_values:
        tag = f"robust{float(robust):g}"
        obs, images[float(robust)] = clean_image(obs, f"{imagename}.{tag}.fits", robust=float(robust),
                                                 **clean_kwargs)
    return images


def calibrate_source(split_ms: str, parent_ms: str, prefix: str, *, robust_values=(-2.0, 0.0, 2.0),
                     solints: Optional[list[str]] = None, min_improvement: float = 0.002,
                     max_bad_fraction: float = 0.25, bayes_models=("clean", "gauss1", "gauss2", "gauss3"),
                     prior_sigma: float = 0.1, workers: Optional[int] = None, imagename: Optional[str] = None,
                     search_fov_mas: float = 0.0, search_sigma: float = 10.0, recentre_min_beams: float = 10.0,
                     **clean_kwargs) -> dict:
    """Full difmapy calibrator sequence: model, phase ladder, amplitude gains, images, tables.

    Products (``prefix`` is a path stem, e.g. ``<work_dir>/selfcal/<code>.<source>``):
    ``<prefix>.phase.G`` (CASA table of the phase self-cal), ``<prefix>.amp.G`` (+``.json``,
    ``.png`` from the Bayesian gscale) and ``<imagename>.robust<r>.fits``. Returns a
    JSON-able report with the rounds, both table paths and the image statistics.

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
                                      workers=workers)
    images = image_all_robust(obs, imagename or prefix, robust_values, **clean_kwargs)
    if any(shift):
        for info in images.values():
            _record_shift(info["fits"], shift[0], shift[1])
    report = {"source": obs.source, "split_ms": str(split_ms), "model": model.get("components", []),
              "search": search, "shift_mas": list(shift),
              "model_rchisq": model.get("rchisq"), "ladder": ladder, "rounds": rounds,
              "phase_table": phase_table.get("path", ""), "amplitude": amplitude, "images": images,
              "station_gains": {k: float(v) for k, v in obs.station_gains().items()}}
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
                 search_sigma: float = 10.0, recentre_min_beams: float = 10.0, **clean_kwargs) -> dict:
    """Image an already-calibrated source (target / check source) with CLEAN only, no self-cal.

    With ``search_fov_mas`` > 0 the source is first located (:func:`locate_source`;
    the dirty map goes to ``<imagename>.search.fits``) and the phase centre moved
    onto it when it lies far outside the regular map. The report carries the
    search under ``"search"`` and the applied ``"shift_mas"`` (east, north).
    """
    obs = load(split_ms)
    search, shift = locate_source(obs, imagename, search_fov_mas=search_fov_mas, search_sigma=search_sigma,
                                  recentre_min_beams=recentre_min_beams)
    images = image_all_robust(obs, imagename, robust_values, **clean_kwargs)
    if any(shift):
        for info in images.values():
            _record_shift(info["fits"], shift[0], shift[1])
    return {"source": obs.source, "split_ms": str(split_ms), "images": images, "search": search,
            "shift_mas": list(shift)}
