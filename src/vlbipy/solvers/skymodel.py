"""Visibilities of a component model (difmap/difmapy modelfit components), for the model overlays of the plots.

A component is the dict difmapy records in its image reports: ``flux`` (Jy), ``x`` / ``y`` (mas east / north
of the phase centre), ``major`` (FWHM, mas), ``ratio`` (minor / major), ``phi`` (position angle of the major
axis, degrees east of north), ``type`` (``"gauss"``; ``"delta"``/``"point"`` or a zero size for a point) and
optionally ``freq0`` (Hz) / ``spcind`` for a power-law spectrum. The visibility of the model is

    V(u, v) = sum_k S_k(nu) G_k(u, v) exp(+2 pi i (u x_k + v y_k))

with ``u, v`` the measurement set's UVW in wavelengths and ``G_k`` the Fourier transform of the elliptical
Gaussian, normalised to 1 at the origin. That is the convention difmap (and difmapy, which reads the UVW of
the measurement set unchanged) fits the components with, so it must be the one they are evaluated with: on
V589A the weighted visibilities of J1405-6118, 46 mas from the phase centre, add up to 0.63 mJy when rotated
with this sign and to 0.12 mJy (as if not rotated at all) with the other. It is the same for both parallel
hands (a Stokes I model).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

#: Speed of light (m/s).
C_M_S = 299792458.0
#: Radians per milliarcsecond.
MAS = math.pi / (180.0 * 3600.0 * 1000.0)
#: Component types drawn as a point.
POINT_TYPES = ("delta", "point")


def shifted(components: list[dict], shift_mas) -> list[dict]:
    """``components`` moved back to the original phase centre.

    The imaging re-centres on a source far from the phase centre (``obs.shift(-x, -y)`` in difmapy) and
    records the applied ``shift_mas`` (east, north): a component it reports at ``(x, y)`` is at
    ``(x + shift_east, y + shift_north)`` from the phase centre of the measurement set.
    """
    east, north = (float(v) for v in (list(shift_mas or ()) + [0.0, 0.0])[:2])
    if not east and not north:
        return [dict(c) for c in components]
    return [dict(c, x=float(c.get("x", 0.0)) + east, y=float(c.get("y", 0.0)) + north) for c in components]


def usable(components) -> list[dict]:
    """The components with a finite, non-zero flux density and finite position."""
    out = []
    for component in components or []:
        try:
            values = [float(component.get(key, 0.0) or 0.0) for key in ("flux", "x", "y")]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in values) and values[0] != 0.0:
            out.append(component)
    return out


def predict(components: list[dict], uvw: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Model visibilities ``(nrow, nchan)`` (complex) of ``components`` for rows with ``uvw`` [m] at
    ``frequencies`` [Hz], either one frequency vector for all rows ``(nchan,)`` or one per row ``(nrow, nchan)``.
    """
    uvw = np.asarray(uvw, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim == 1:
        freqs = np.broadcast_to(freqs, (uvw.shape[0], freqs.size))
    scale = freqs / C_M_S                                                   # wavelengths per metre
    u = uvw[:, 0:1] * scale
    v = uvw[:, 1:2] * scale
    total = np.zeros(u.shape, dtype=np.complex128)
    for component in usable(components):
        flux = float(component["flux"])
        x, y = float(component.get("x", 0.0)) * MAS, float(component.get("y", 0.0)) * MAS
        term = np.exp(2j * np.pi * (u * x + v * y))
        major = float(component.get("major", 0.0) or 0.0) * MAS
        if str(component.get("type", "gauss")).lower() not in POINT_TYPES and major > 0:
            ratio = float(component.get("ratio", 1.0) or 1.0)
            phi = math.radians(float(component.get("phi", 0.0) or 0.0))
            along = u * math.sin(phi) + v * math.cos(phi)                   # uv along the major axis
            across = u * math.cos(phi) - v * math.sin(phi)
            term = term * np.exp(-(math.pi ** 2) / (4.0 * math.log(2.0))
                                 * ((major * along) ** 2 + (major * ratio * across) ** 2))
        freq0, index = float(component.get("freq0", 0.0) or 0.0), float(component.get("spcind", 0.0) or 0.0)
        spectrum = (freqs / freq0) ** index if freq0 > 0 and index else 1.0
        total += flux * spectrum * term
    return total


def row_frequencies(chan_freq: np.ndarray, ddid_to_spw, ddid: np.ndarray) -> np.ndarray:
    """``(nrow, nchan)`` channel frequencies of rows with data description ``ddid``."""
    spw = np.asarray(ddid_to_spw)[np.asarray(ddid, dtype=np.int64)]
    return np.asarray(chan_freq, dtype=np.float64)[spw]


def masked_model(components: Optional[list[dict]], block: dict, flags: np.ndarray, chan_freq, ddid_to_spw,
                 rows: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Model visibilities of the rows of a read ``block`` with the data's flags applied (``nan`` where flagged).

    ``flags`` is ``(nrow, nchan, npol)`` of the selected rows (``rows``, a boolean or index selection of the
    block, all rows when None); the model is repeated over the hands. None without components.
    """
    if not usable(components):
        return None
    uvw = np.asarray(block["UVW"], dtype=np.float64)
    ddid = np.asarray(block["DATA_DESC_ID"], dtype=np.int64)
    if rows is not None:
        uvw, ddid = uvw[rows], ddid[rows]
    model = predict(components, uvw, row_frequencies(chan_freq, ddid_to_spw, ddid))
    model = np.repeat(model[:, :, None], flags.shape[2], axis=2)
    return np.where(flags, np.nan, model)


def recorded_model(work_dir, project_code: str, source: str) -> list[dict]:
    """The components of the latest model of ``source``, at their position from the original phase centre.

    Read from the record of the final imaging (``images/<code>_<source>.search.json``) or, for a calibrator
    imaged only by the self-calibration, from its report (``selfcal/<code>.<source>.json``). Empty when the
    source has not been imaged (or the record cannot be read).
    """
    import json
    from pathlib import Path

    candidates = (Path(work_dir) / "images" / f"{project_code}_{source}.search.json",
                  Path(work_dir) / "selfcal" / f"{project_code}.{source}.json")
    for path in candidates:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for model in (record.get("model"), record.get("image_model")):
            components = model.get("components") if isinstance(model, dict) else model
            if isinstance(components, list) and usable(components):
                return shifted(usable(components), record.get("shift_mas"))
    return []
