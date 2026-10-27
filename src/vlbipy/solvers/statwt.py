"""Visibility weights from the scatter of the data (the dask-ms backend's ``statwt``).

:func:`statwt` writes what ``casatasks.statwt`` writes with its default
binning (one integration per bin, ``chanbin='spw'``, correlations separately,
``statalg='classic'``): for every row and correlation, the unflagged channels of
the data column give the population variances of the real and imaginary parts,
and the weight is ``2 / (var_re + var_im)``. A row/correlation with fewer than
``minsamp`` unflagged channels gets weight 0 and is flagged. WEIGHT_SPECTRUM
(when present) holds the weight on every channel, WEIGHT the same value per
correlation, FLAG_ROW whether a row is now entirely flagged. Checked against
CASA on GL048 (relative difference 6e-8).

Unlike CASA, SIGMA (and SIGMA_SPECTRUM) is updated too, by the ratio of the old
to the new weight: ``sigma * sqrt(w_old / w_new)``. Every later applycal
rebuilds the weights from SIGMA, so CASA's statwt on the corrected column only
lasts until the chain is applied again — and the pipeline re-solves and applies
after reweighting (third pass, self-calibration). Carrying the measured excess
noise in SIGMA instead keeps it through any later chain: an antenna noisier than
its Tsys says (V589A KE, x6 off) otherwise returns to its nominal weight and
dominates the faint-target image with its noise.

Every row is independent, so the parts of a Multi-MS are processed by parallel
worker processes, each reading and writing its own table. When the calibration
of a field was deferred (:mod:`vlbipy.solvers.corrected`), its corrected data
are computed from DATA as they are read.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .calblock import sigma_to_weight
from .corrected import view as corrected_view, weights_set
from .msio import open_table, read_layout, table_engine
from .workers import run_jobs

logger = get_logger()

#: Rows per read/write block (64 channels x 4 correlations: 40 MB of visibilities).
STATWT_ROWS = 50_000


def row_weights(values: np.ndarray, flags: np.ndarray, minsamp: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """statwt's weight per row and correlation from ``(nrow, nchan, ncorr)`` values and flags.

    Returns
    -------
    (weight, starved)
        ``weight`` float32 (nrow, ncorr), 0 where there are fewer than ``minsamp`` unflagged channels or no
        scatter; ``starved`` bool (nrow, ncorr), the cells with fewer than ``minsamp`` unflagged channels.
    """
    good = ~flags
    count = good.sum(axis=1)
    safe = np.maximum(count, 1)
    real = np.where(good, values.real, 0).astype(np.float64)
    imag = np.where(good, values.imag, 0).astype(np.float64)
    variance = np.zeros(count.shape, dtype=np.float64)
    for part in (real, imag):
        mean = part.sum(axis=1) / safe
        variance += (np.where(good, part - mean[:, None, :], 0) ** 2).sum(axis=1) / safe
    starved = count < int(minsamp)
    with np.errstate(divide="ignore"):
        weight = np.where(starved | (variance <= 0), 0.0, 2.0 / np.where(variance > 0, variance, 1.0))
    return weight.astype(np.float32), starved


def rescaled_sigma(sigma: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """``sigma * sqrt(old / new)`` where both weights are positive, ``sigma`` unchanged elsewhere.

    ``sigma`` is (nrow, ncorr) or (nrow, nchan, ncorr); ``old`` and ``new`` are (nrow, ncorr).
    """
    sigma = np.asarray(sigma, dtype=np.float32)
    old, new = np.asarray(old, dtype=np.float64), np.asarray(new, dtype=np.float64)
    usable = (old > 0) & (new > 0)
    factor = np.sqrt(np.divide(old, new, out=np.ones_like(old), where=usable))
    if sigma.ndim == 3:
        factor = factor[:, None, :]
    return (sigma * factor).astype(np.float32)


def statwt_part_job(*, path: str, segments: list[dict], engine: str, column: str, minsamp: int,
                    has_weight_spectrum: bool, chunk_rows: int, view: Optional[dict] = None,
                    has_sigma_spectrum: bool = False) -> dict:
    """Worker job: reweight the row ranges ``segments`` of one table part in place.

    With ``view`` (:func:`vlbipy.solvers.corrected.view`), the segments of its fields are read from DATA and
    corrected as applycal would have written CORRECTED_DATA. SIGMA (and SIGMA_SPECTRUM) is rescaled by the
    ratio of the weights before and after (:func:`rescaled_sigma`).
    """
    from .applycal import corrected_chunks
    stats = {"rows": 0, "flagged_before": 0, "flagged_after": 0, "samples": 0,
             "wsum": 0.0, "wsum2": 0.0, "wcount": 0}
    live = set(view["entries_by_field"]) if view is not None else set()

    def reweight(start: int, nrow: int, values: np.ndarray, flags: np.ndarray, old: np.ndarray) -> None:
        before = int(np.count_nonzero(flags))
        weight, starved = row_weights(values, flags, minsamp)
        handle.putcol("SIGMA", rescaled_sigma(handle.getcol("SIGMA", start, nrow), old, weight), start)
        if has_sigma_spectrum:
            handle.putcol("SIGMA_SPECTRUM",
                          rescaled_sigma(handle.getcol("SIGMA_SPECTRUM", start, nrow), old, weight), start)
        flags |= starved[:, None, :]
        after = int(np.count_nonzero(flags))
        if after != before:
            handle.putcol("FLAG", flags, start)
        handle.putcol("FLAG_ROW", flags.all(axis=(1, 2)), start)
        handle.putcol("WEIGHT", weight, start)
        if has_weight_spectrum:
            handle.putcol("WEIGHT_SPECTRUM", np.ascontiguousarray(np.broadcast_to(weight[:, None, :], flags.shape)),
                          start)
        # CASA's summary counts every unflagged sample (channel) a weight was computed from.
        nonzero = weight > 0
        used = weight[nonzero].astype(np.float64)
        samples = (~flags).sum(axis=1)[nonzero]
        stats["rows"] += nrow
        stats["samples"] += int(flags.size)
        stats["flagged_before"] += before
        stats["flagged_after"] += after
        stats["wsum"] += float((used * samples).sum())
        stats["wsum2"] += float((used * used * samples).sum())
        stats["wcount"] += int(samples.sum())

    handle = open_table(path, readonly=False, engine=engine)
    try:
        for segment in segments:
            if int(segment.get("field", -1)) in live:
                continue
            for offset in range(0, segment["nrow"], chunk_rows):
                start, nrow = segment["start"] + offset, min(chunk_rows, segment["nrow"] - offset)
                reweight(start, nrow, handle.getcol(column, start, nrow),
                         np.asarray(handle.getcol("FLAG", start, nrow), dtype=bool),
                         np.asarray(handle.getcol("WEIGHT", start, nrow), dtype=np.float32))
        corrected = [s for s in segments if int(s.get("field", -1)) in live]
        if corrected:
            sigma_column = "SIGMA_SPECTRUM" if has_sigma_spectrum else "SIGMA"
            for start, nrow, vis, flag, cube, weighted, _ in corrected_chunks(
                    handle, corrected, entries_by_field=view["entries_by_field"], setup=view["setup"],
                    columns=view["columns"], parang=view["parang"], chunk_rows=chunk_rows, weights="sigma",
                    calflags=view["write_flags"]):
                # The weights the deferred applycal would write: calibrated where its chain scales them,
                # rebuilt from SIGMA elsewhere.
                old = sigma_to_weight(handle.getcol(sigma_column, start, nrow))
                old = old.mean(axis=1) if old.ndim == 3 else old
                if cube is not None:
                    rows = slice(None) if weighted is None else np.asarray(weighted, dtype=bool)
                    old[rows] = cube.mean(axis=1)[rows]
                reweight(start, nrow, vis, flag, old)
    finally:
        handle.close()
    return stats


def statwt(ms, *, column: str = "CORRECTED_DATA", minsamp: int = 2, workers: Optional[int] = None,
              chunk_rows: int = STATWT_ROWS) -> dict:
    """Recompute WEIGHT (and WEIGHT_SPECTRUM) of every row of ``ms`` from the scatter of ``column``.

    SIGMA (and SIGMA_SPECTRUM) is rescaled to match, so that the weights a later applycal rebuilds from it
    keep the measured noise (see the module docstring).

    Returns
    -------
    dict
        ``mean`` and ``variance`` of the new non-zero weights over the unflagged samples (as CASA reports them),
        ``rows``,
        ``flagged_before``/``flagged_after`` sample counts and ``seconds``.
    """
    if int(minsamp) < 2:
        raise ValueError("minsamp must be >= 2")
    started = _time.perf_counter()
    engine = table_engine()
    layout = read_layout(ms, engine=engine)
    view = corrected_view(ms, column, columns=layout["columns"])
    if column not in layout["columns"] and (view is None or set(np.unique(layout["field"]).tolist())
                                            - set(view["entries_by_field"])):
        raise RuntimeError(f"{Path(ms).name} has no {column} column")
    jobs = []
    for part_index, path in enumerate(layout["parts"]):
        runs = np.flatnonzero(layout["part"] == part_index)
        if not runs.size:
            continue
        jobs.append({"path": path, "engine": engine, "column": column, "minsamp": int(minsamp),
                     "has_weight_spectrum": "WEIGHT_SPECTRUM" in layout["columns"], "chunk_rows": int(chunk_rows),
                     "has_sigma_spectrum": "SIGMA_SPECTRUM" in layout["columns"],
                     "segments": [{"start": int(layout["start"][i]), "nrow": int(layout["nrow"][i]),
                                   "field": int(layout["field"][i]), "ddid": int(layout["ddid"][i])} for i in runs],
                     "view": view})
    results = run_jobs("vlbipy.solvers.statwt:statwt_part_job", jobs, workers)
    # A deferred calibration (vlbipy.solvers.corrected) now weighs its data with these weights, as an applycal
    # written before statwt would.
    weights_set(ms, np.unique(layout["field"]).tolist())
    total = {name: sum(r[name] for r in results) for name in results[0]} if results else {}
    count = max(total.get("wcount", 0), 1)
    mean = total.get("wsum", 0.0) / count
    variance = (total.get("wsum2", 0.0) - count * mean * mean) / max(count - 1, 1)
    report = {"mean": mean, "variance": variance,
              "rows": total.get("rows", 0), "flagged_before": total.get("flagged_before", 0),
              "flagged_after": total.get("flagged_after", 0), "seconds": _time.perf_counter() - started}
    logger.info("statwt[dask-ms]: {} — {} row(s) in {} part(s), flagged {:.1%} -> {:.1%}, {:.1f}s",
                Path(ms).name, report["rows"], len(jobs),
                report["flagged_before"] / max(total.get("samples", 0), 1),
                report["flagged_after"] / max(total.get("samples", 0), 1), report["seconds"])
    return report
