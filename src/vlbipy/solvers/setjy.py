"""Write MODEL_DATA for one field from a model image — the dask-ms backend's ``setjy``.

A CASA model image (a ``*.model`` / ``*.model.tt0`` directory, Jy/pixel: the e-MERLIN 3C286 models in
``vlbipy/data/emerlin_calibrator_models``) gives one point component per non-zero pixel, negative CLEAN
components included. A FITS image gives its pixels above a fraction of the peak. For a flux-density standard
(3C286: Perley & Butler 2017) the model is scaled channel by channel so that its total is the catalogue flux at
that frequency, as CASA's ``setjy`` does with ``model=`` and ``scalebychan=True``: that is what gives the solved
gains their absolute scale, the role ``setjy`` + ``getjy`` play in a CASA flow.

Visibilities are computed per row and channel as the direct sum
``V(u, v) = sum_k A_k exp(-2 pi i (u l_k + v m_k))`` with ``u, v`` in wavelengths and ``l`` growing to the
east (the image's own, signed, increments); amplitude corrections (the only use of this model in the pipeline)
do not depend on the phase sign convention. Parallel hands get ``V``, cross hands get 0 — the model is
Stokes I only.
"""
from __future__ import annotations

import math
import time as _time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..logging_utils import get_logger
from .applycal import ensure_column
from .msio import open_table, read_layout, read_setup, select_runs, table_engine

logger = get_logger()

SPEED_OF_LIGHT = 299792458.0
#: Parallel-hand Stokes-I correlation codes (MS POLARIZATION CORR_TYPE).
_CIRC = {"RR": 5, "RL": 6, "LR": 7, "LL": 8}


def pb17_flux_3c286(freq_hz: float) -> float:
    """Perley & Butler (2017) flux-density model of 3C286 in Jy at ``freq_hz``."""
    log_nu = math.log10(freq_hz / 1e9)
    return 10.0 ** (1.2515 - 0.4605 * log_nu - 0.1715 * log_nu**2 + 0.0336 * log_nu**3)


#: Field names 3C286 goes by.
_3C286_NAMES = {"3c286", "1331+305", "j1331+3030", "j1331+305", "1328+307", "b1328+307"}


def standard_flux(field: str) -> Optional[Callable[[float], float]]:
    """The catalogue flux-density model (Jy at a frequency in Hz) of a flux-density standard, by field name; None
    for any other source. Only 3C286 (Perley & Butler 2017) so far."""
    return pb17_flux_3c286 if str(field).strip().lower().replace(" ", "") in _3C286_NAMES else None


def casa_image_components(image) -> tuple:
    """Every non-zero pixel of a CASA model image (Jy/pixel) as (flux [Jy], l, m [rad], frequency [Hz]).

    ``l``, ``m`` are the offsets from the reference pixel along the image's own increments (RA decreasing with x),
    so ``l`` grows to the east. The frequency is the spectral reference value (None without a spectral axis).
    """
    from casacore.images import image as casa_image

    handle = casa_image(str(image))
    data = np.asarray(handle.getdata(), dtype=np.float64)
    coords = handle.coordinates()
    direction = coords.get_coordinate("direction")
    (ref_y, ref_x), (inc_y, inc_x) = direction.get_referencepixel(), direction.get_increment()
    units = [str(u) for u in direction.get_unit()]
    if any(u != "rad" for u in units):
        raise ValueError(f"{image}: direction axes in {units}, expected radians")
    try:
        freq = float(np.ravel(coords.get_coordinate("spectral").get_referencevalue())[0])
    except Exception:  # noqa: BLE001 - an image without a spectral axis
        freq = None
    plane = data.reshape(-1, data.shape[-2], data.shape[-1])[0]          # (y, x) of the first plane
    iy, ix = np.nonzero(plane)
    if not iy.size:
        raise ValueError(f"{image}: the model image is empty")
    flux = plane[iy, ix]
    l = (ix - float(ref_x)) * float(inc_x)                              # RA increment < 0: l grows to the east
    m = (iy - float(ref_y)) * float(inc_y)
    logger.info("model {}: {} components ({} negative), {:.3f} Jy at {}", Path(image).name, flux.size,
                int((flux < 0).sum()), flux.sum(), f"{freq / 1e9:.4f} GHz" if freq else "no frequency")
    return flux, l, m, freq


def model_components(image, *, threshold: float = 0.005) -> tuple:
    """(flux [Jy], l, m [rad], frequency [Hz] or None) of a model image: a CASA image directory (every non-zero
    pixel) or a FITS file (:func:`image_components`)."""
    if Path(str(image)).is_dir():
        return casa_image_components(image)
    flux, l, m = image_components(image, threshold=threshold)
    return flux, l, m, _fits_frequency(image)


def _fits_frequency(image) -> Optional[float]:
    """The FREQ axis reference value of a FITS image, or None."""
    from astropy.io import fits

    header = fits.getheader(str(image))
    for axis in range(1, int(header.get("NAXIS", 0)) + 1):
        if str(header.get(f"CTYPE{axis}", "")).upper().startswith("FREQ"):
            return float(header[f"CRVAL{axis}"])
    return None


def predict(uvw: np.ndarray, freqs: np.ndarray, flux: np.ndarray, l: np.ndarray, m: np.ndarray, *,
            scale: Optional[np.ndarray] = None, max_cells: int = 2_000_000) -> np.ndarray:
    """Model visibilities ``(rows, channels)`` of point components at ``uvw`` (m) and ``freqs`` (Hz).

    ``scale`` multiplies each channel (the spectrum of a flux-density standard). The sum runs over row blocks of
    at most ``max_cells`` row x component cells; on an evenly spaced band each channel's phasors are the previous
    ones times one step, so the exponentials are computed twice per block instead of once per channel.
    """
    uvw = np.asarray(uvw, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    out = np.empty((uvw.shape[0], freqs.size), dtype=np.complex128)
    if not uvw.shape[0] or not freqs.size:
        return out
    step = freqs[1] - freqs[0] if freqs.size > 1 else 0.0
    even = freqs.size > 1 and np.allclose(np.diff(freqs), step, rtol=1e-6, atol=1e-3)
    block = max(1, int(max_cells) // max(flux.size, 1))
    for start in range(0, uvw.shape[0], block):
        rows = slice(start, start + block)
        k = (-2j * np.pi / SPEED_OF_LIGHT) * (uvw[rows, 0:1] * l[None, :] + uvw[rows, 1:2] * m[None, :])
        if even:
            phasor, advance = np.exp(k * freqs[0]), np.exp(k * step)
            for channel in range(freqs.size):
                out[rows, channel] = phasor @ flux
                if channel + 1 < freqs.size:
                    phasor *= advance
        else:
            for channel, freq in enumerate(freqs):
                out[rows, channel] = np.exp(k * freq) @ flux
    if scale is not None:
        out *= np.asarray(scale, dtype=np.float64)[None, :]
    return out


def spectral_scale(freqs: np.ndarray, flux: np.ndarray, field: str, *, standard: bool = True,
                   flux_total: Optional[float] = None) -> Optional[np.ndarray]:
    """Per-channel factor that brings the model's total to the catalogue flux of a standard at each frequency
    (None: the image's own total). ``flux_total`` instead scales the total to that value in every channel."""
    total = float(np.sum(flux))
    if flux_total is not None:
        return np.full(np.shape(freqs), float(flux_total) / total)
    spectrum = standard_flux(field) if standard else None
    if spectrum is None:
        return None
    return np.array([spectrum(float(f)) for f in np.ravel(freqs)]) / total


def image_components(image, *, threshold: float = 0.005, flux_total: Optional[float] = None) -> tuple:
    """Pixels of the FITS image above ``threshold`` x peak as (flux [Jy], l, m [rad]).

    ``flux_total`` rescales the components so their sum equals it (the catalogue
    flux scale); None keeps the image's own Jy/beam sum.
    """
    from astropy.io import fits

    with fits.open(str(image)) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64).squeeze()
        header = hdul[0].header
    while data.ndim > 2:
        data = data[0]
    peak = float(np.nanmax(data))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"{image}: no positive pixel in the model image")
    iy, ix = np.nonzero(data >= threshold * peak)
    crpix = (float(header.get("CRPIX1", 0)), float(header.get("CRPIX2", 0)))
    cdelt = (float(header.get("CDELT1", header.get("CDELT1A", 0))),
             float(header.get("CDELT2", header.get("CDELT2A", 0))))
    if not all(crpix) or not all(cdelt):
        raise ValueError(f"{image}: CRPIX/CDELT missing; cannot place the model on the sky")
    # FITS pixels are 1-based; the increments keep their sign (RA decreases with x), so l grows to the east -
    # their absolute values mirrored the source in RA, which changes the visibility amplitudes of any
    # asymmetric source.
    l = np.radians((ix + 1 - crpix[0]) * cdelt[0])
    m = np.radians((iy + 1 - crpix[1]) * cdelt[1])
    flux = data[iy, ix]
    if flux_total is not None:
        flux = flux * float(flux_total) / flux.sum()
    logger.info("model {}: {} components above {:.1%} of the peak ({:.3f} Jy total)", Path(image).name,
                flux.size, threshold, flux.sum())
    return flux.astype(np.float64), l, m


def _corr_parallel(corr_types, ncorr: int) -> np.ndarray:
    """Bool per correlation slot: True for the parallel hands (RR/LL or XX/YY)."""
    types = list(corr_types or [])
    if not types or len(types) != ncorr:
        out = np.zeros(ncorr, dtype=bool)
        out[[0, ncorr - 1]] = True
        return out
    parallel = {_CIRC["RR"], _CIRC["LL"], 1, 4}     # XX=1, YY=4 in the Stokes code list
    return np.asarray([t in parallel for t in types])





def setjy(vis, field: str, image, *, flux_total: Optional[float] = None, standard: bool = True,
          threshold: float = 0.005, engine: Optional[str] = None, workers: Optional[int] = None) -> dict:
    """Write MODEL_DATA of ``field`` from the model ``image``; return a small report.

    Parameters
    ----------
    vis : str or Path
        Measurement set (or Multi-MS); MODEL_DATA is created if missing.
    field : str
        Field the model applies to.
    image : str or Path
        CASA model image (directory, Jy/pixel) or FITS image (pixels above ``threshold`` x peak).
    flux_total : float, optional
        Total flux density of the model in Jy in every channel. None: the catalogue flux of a standard at each
        channel's frequency (``standard``), else the image's own sum.
    standard : bool
        Scale a flux-density standard (3C286) to its catalogue spectrum, channel by channel.
    threshold : float
        Fraction of the image peak below which pixels are ignored.
    """
    started = _time.perf_counter()
    vis = str(vis)
    engine = engine or table_engine()
    setup = read_setup(vis, engine=engine)
    layout = read_layout(vis, engine=engine)
    if field not in setup["field_names"]:
        raise ValueError(f"{Path(vis).name} has no field {field!r}")
    field_id = setup["field_names"].index(field)
    flux, l, m, image_freq = model_components(image, threshold=threshold)
    chan_freq = setup["chan_freq"]                                            # (nspw, nchan)
    ncorr = int(setup["ncorr"])
    parallel = _corr_parallel(setup.get("corr_types"), ncorr)

    if ensure_column(vis, "MODEL_DATA", "DATA", engine=engine):
        # A new column has no array in its cells: every other field gets the unit model (CASA's setjy with
        # usescratch), or any later reader of MODEL_DATA fails on them (EM178: the calibrated SNR survey).
        unit_model(vis, layout, setup, skip_fields={field_id}, engine=engine)
    selected = select_runs(layout, fields={field_id})
    rows_written = 0
    # Model computation is a dense (row,chan) x component product — fast enough in one pass per part.
    for index in selected:
        part = layout["parts"][int(layout["part"][index])]
        handle = open_table(part, engine=engine, readonly=False)
        try:
            start, nrow = int(layout["start"][index]), int(layout["nrow"][index])
            ddids = np.asarray(handle.getcol("DATA_DESC_ID", start=start, nrow=nrow), dtype=np.int64)
            uvw = np.asarray(handle.getcol("UVW", start=start, nrow=nrow), dtype=np.float64)
            # Not read from MODEL_DATA: a column just added has no array in its cells yet (EM178: "TSM: no array
            # in row 0 of column MODEL_DATA").
            out = np.zeros((nrow, chan_freq.shape[1], ncorr), dtype=np.complex64)
            for ddid in np.unique(ddids):
                sel = np.flatnonzero(ddids == ddid)
                freqs = chan_freq[int(setup["ddid_to_spw"][int(ddid)])]
                scale = spectral_scale(freqs, flux, field, standard=standard, flux_total=flux_total)
                values = predict(uvw[sel], freqs, flux, l, m, scale=scale).astype(np.complex64)
                out[sel] = values[:, :, None] * parallel.astype(np.complex64)[None, None, :]
            handle.putcol("MODEL_DATA", out, start=start)
            rows_written += nrow
        finally:
            handle.close()
    freqs = np.asarray(chan_freq, dtype=np.float64)
    centre = float(np.mean(freqs[freqs > 0])) if np.any(freqs > 0) else 0.0
    scale = spectral_scale(np.array([centre]), flux, field, standard=standard, flux_total=flux_total)
    total = float(flux.sum() * (scale[0] if scale is not None else 1.0))
    report = {"rows": int(rows_written), "components": int(flux.size), "image_flux_jy": float(flux.sum()),
              "image_freq_hz": image_freq, "flux_jy": total, "centre_freq_hz": centre,
              "scaled": "catalogue" if scale is not None and flux_total is None else
                        ("flux_total" if flux_total is not None else "image"),
              "seconds": _time.perf_counter() - started}
    logger.info("setjy[native]: {} field={} <- {} ({} components; {:.3f} Jy at {:.4f} GHz ({}), image {:.3f} Jy at "
                "{}; {} rows) in {:.1f}s", Path(vis).name, field, Path(image).name, flux.size, total, centre / 1e9,
                report["scaled"], flux.sum(), f"{image_freq / 1e9:.4f} GHz" if image_freq else "?", rows_written,
                report["seconds"])
    return report


def unit_model(vis, layout: dict, setup: dict, *, skip_fields=(), engine: Optional[str] = None,
               max_rows: int = 20000) -> int:
    """Write the unit model (1 in the parallel hands, 0 in the cross hands) into MODEL_DATA of every field but
    ``skip_fields``; returns the rows written. MODEL_DATA must exist."""
    parallel = _corr_parallel(setup.get("corr_types"), int(setup["ncorr"])).astype(np.complex64)
    nchan = int(np.asarray(setup["chan_freq"]).shape[1])
    skip = {int(f) for f in skip_fields}
    written = 0
    for part_index, part in enumerate(layout["parts"]):
        runs = [i for i in np.flatnonzero(layout["part"] == part_index) if int(layout["field"][i]) not in skip]
        if not runs:
            continue
        handle = open_table(part, engine=engine, readonly=False)
        try:
            for index in runs:
                start, nrow = int(layout["start"][index]), int(layout["nrow"][index])
                for offset in range(0, nrow, max_rows):
                    count = min(max_rows, nrow - offset)
                    block = np.broadcast_to(parallel[None, None, :], (count, nchan, parallel.size))
                    handle.putcol("MODEL_DATA", np.ascontiguousarray(block), start=start + offset)
                    written += count
        finally:
            handle.close()
    return written


def write_component_models(vis, models: dict, *, engine: Optional[str] = None) -> dict:
    """Write MODEL_DATA from component models: ``models`` maps field name -> difmap(y) components.

    The components are evaluated with :func:`vlbipy.solvers.skymodel.predict` (the convention difmap fits them
    with) for the parallel hands; the cross hands get 0. Every other field gets a unit model (1 in the parallel
    hands), so a solve that divides by MODEL_DATA treats it as a point source, as CASA does after ``clearcal``.
    MODEL_DATA is created when missing. Returns ``{"rows", "fields": {name: rows}, "seconds"}``.
    """
    from .skymodel import usable

    started = _time.perf_counter()
    vis = str(vis)
    engine = engine or table_engine()
    setup = read_setup(vis, engine=engine)
    layout = read_layout(vis, engine=engine)
    names = setup["field_names"]
    unknown = [name for name in models if name not in names]
    if unknown:
        raise ValueError(f"{Path(vis).name} has no field(s) {', '.join(unknown)}")
    by_id = {names.index(name): usable(components) for name, components in models.items()}
    parallel = _corr_parallel(setup.get("corr_types"), int(setup["ncorr"])).astype(np.complex64)
    ensure_column(vis, "MODEL_DATA", "DATA", engine=engine)
    counts: dict = {}
    rows_written = 0
    for part_index, part in enumerate(layout["parts"]):
        runs = np.flatnonzero(layout["part"] == part_index)
        if not runs.size:
            continue
        handle = open_table(part, engine=engine, readonly=False)
        try:
            for index in runs:
                rows_written += _write_run_model(handle, layout, int(index), by_id, setup, parallel, counts, names)
        finally:
            handle.close()
    report = {"rows": int(rows_written), "fields": {name: int(counts.get(name, 0)) for name in models},
              "seconds": _time.perf_counter() - started}
    logger.info("model[native]: {} MODEL_DATA <- {} ({} rows, unit model elsewhere) in {:.1f}s", Path(vis).name,
                ", ".join(f"{name} ({len(by_id[names.index(name)])} components)" for name in models),
                rows_written, report["seconds"])
    return report


def _write_run_model(handle, layout: dict, index: int, by_id: dict, setup: dict, parallel: np.ndarray,
                     counts: dict, names: list) -> int:
    """Write MODEL_DATA of one run (rows of one field, scan and subband) of an open table; returns its rows."""
    from .skymodel import predict

    field_id = int(layout["field"][index])
    start, nrow = int(layout["start"][index]), int(layout["nrow"][index])
    chan_freq = setup["chan_freq"]
    components = by_id.get(field_id)
    out = np.empty((nrow, chan_freq.shape[1], parallel.size), dtype=np.complex64)
    if components:
        ddids = np.asarray(handle.getcol("DATA_DESC_ID", start=start, nrow=nrow), dtype=np.int64)
        uvw = np.asarray(handle.getcol("UVW", start=start, nrow=nrow), dtype=np.float64)
        for ddid in np.unique(ddids):
            sel = np.flatnonzero(ddids == ddid)
            values = predict(components, uvw[sel], chan_freq[int(setup["ddid_to_spw"][int(ddid)])])
            out[sel] = values.astype(np.complex64)[:, :, None] * parallel[None, None, :]
    else:
        out[:] = parallel[None, None, :]
    handle.putcol("MODEL_DATA", out, start=start)
    counts[names[field_id]] = counts.get(names[field_id], 0) + nrow
    return nrow
