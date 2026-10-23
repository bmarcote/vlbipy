"""Write MODEL_DATA for one field from a FITS model image — the dask-ms backend's ``setjy``.

The pixels of the image above a fraction of its peak are taken as point
components, each pixel carrying its Jy/beam value as flux density. Their total
is then normalised to the catalogue flux of the source at the band's centre
frequency (Perley & Butler 2017 for 3C286), which is what gives the solved gains
their absolute scale — the role ``setjy`` + ``getjy`` play in a CASA flow.

Visibilities are computed per row and channel as the direct sum
``V(u, v) = sum_k A_k exp(-2 pi i (u l_k + v m_k))`` with ``u, v`` in wavelengths;
amplitude corrections (the only use of this model in the pipeline) do not depend
on the phase sign convention. Parallel hands get ``V``, cross hands get 0 —
the model is Stokes I only.
"""
from __future__ import annotations

import math
import time as _time
from pathlib import Path
from typing import Optional

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
    cdelt = (abs(float(header.get("CDELT1", header.get("CDELT1A", 0)))),
             abs(float(header.get("CDELT2", header.get("CDELT2A", 0)))))
    if not all(crpix) or not all(cdelt):
        raise ValueError(f"{image}: CRPIX/CDELT missing; cannot place the model on the sky")
    l = np.radians((ix + 1 - crpix[0]) * cdelt[0])    # FITS pixels are 1-based
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





def setjy(vis, field: str, image, *, flux_total: Optional[float] = None,
                       threshold: float = 0.005, engine: Optional[str] = None,
                       workers: Optional[int] = None) -> dict:
    """Write MODEL_DATA of ``field`` from the FITS ``image``; return a small report.

    Parameters
    ----------
    vis : str or Path
        Measurement set (or Multi-MS); MODEL_DATA is created if missing.
    field : str
        Field the model applies to.
    image : str or Path
        FITS model image (Jy/beam pixels, position from CRPIX/CDELT).
    flux_total : float, optional
        Absolute flux density of the model in Jy at the band's centre frequency;
        the components are scaled to it. None uses the image's own sum.
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
    flux, l, m = image_components(image, threshold=threshold, flux_total=flux_total)
    chan_freq = setup["chan_freq"]                                            # (nspw, nchan)
    ncorr = int(setup["ncorr"])
    parallel = _corr_parallel(setup.get("corr_types"), ncorr)

    ensure_column(vis, "MODEL_DATA", "DATA", engine=engine)
    selected = select_runs(layout, fields={field_id})
    rows_written = 0
    # Model computation is a dense (row,chan) x component product — fast enough in one pass per part.
    for index in selected:
        part = layout["parts"][int(layout["part"][index])]
        handle = open_table(part, engine=engine, readonly=False)
        try:
            start, nrow = int(layout["start"][index]), int(layout["nrow"][index])
            ddids = np.asarray(handle.getcol("DATA_DESC_ID", start=start, nrow=nrow), dtype=np.int64)
            model = np.asarray(handle.getcol("MODEL_DATA", start=start, nrow=nrow))
            uvw = np.asarray(handle.getcol("UVW", start=start, nrow=nrow), dtype=np.float64)
            out = np.zeros_like(model)
            for ddid in np.unique(ddids):
                sel = np.flatnonzero(ddids == ddid)
                freqs = chan_freq[int(setup["ddid_to_spw"][int(ddid)])]
                u_l = uvw[sel, 0:1] * freqs[None, :] / SPEED_OF_LIGHT
                v_l = uvw[sel, 1:2] * freqs[None, :] / SPEED_OF_LIGHT
                phase = -2.0 * np.pi * (u_l[:, :, None] * l + v_l[:, :, None] * m)
                values = (flux * np.exp(1j * phase)).sum(axis=2).astype(np.complex64)
                out[sel] = values[:, :, None] * parallel.astype(np.complex64)[None, None, :]
            handle.putcol("MODEL_DATA", out, start=start, nrow=nrow)
            rows_written += nrow
        finally:
            handle.close()
    report = {"rows": int(rows_written), "components": int(flux.size),
              "flux_jy": float(flux.sum()), "seconds": _time.perf_counter() - started}
    logger.info("setjy[native]: {} field={} <- {} ({} components, {:.2f} Jy; {} rows) in {:.1f}s",
                Path(vis).name, field, Path(image).name, flux.size, flux.sum(), rows_written,
                report["seconds"])
    return report
