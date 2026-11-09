"""How well a flux calibrator's calibrated amplitudes match its model, per antenna, subband and hand.

The e-MERLIN flux scale (:meth:`vlbipy.namespaces.CalibrateNamespace.emerlin_scale`) puts the e-MERLIN antennas on
the absolute scale of a modelled calibrator (3C286). This module checks the result where it matters: the calibrated
visibilities of the calibrator (CORRECTED_DATA, computed on the fly when the calibration is deferred) on the
baselines among those antennas, against the model predicted at each row and channel
(:func:`vlbipy.solvers.setjy.predict`, scaled to the catalogue spectrum). The amplitude ratio of a baseline is
``sum |V| |M| / sum |M|^2`` over its unflagged cells - phase-blind, so a residual phase error does not read as a flux
error - and the antenna ratios follow from ``log r_ij = x_i + x_j`` by least squares, per subband and hand.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .applycal import read_view_runs
from .corrected import view as corrected_view
from .msio import open_table, read_layout, read_setup, run_specs, select_runs, split_rows, table_engine
from .setjy import _corr_parallel, model_components, predict, spectral_scale

logger = get_logger()

#: Rows read at a time.
CHECK_ROWS = 20000


def antenna_ratios(baselines: dict[tuple[int, int], tuple[float, float]]) -> dict[int, float]:
    """Antenna amplitude ratios from baseline ones: ``{(i, j): (ratio, weight)}`` -> ``{i: ratio}``.

    Solves ``log r_ij = x_i + x_j`` by weighted least squares. Needs a baseline graph with an odd cycle (three
    antennas in a triangle at least); otherwise the split between two antennas is not determined and nothing is
    returned.
    """
    pairs = [(i, j, r, w) for (i, j), (r, w) in baselines.items() if r > 0 and np.isfinite(r) and w > 0]
    antennas = sorted({i for i, *_ in pairs} | {j for _, j, *_ in pairs})
    if len(antennas) < 3:
        return {}
    index = {a: k for k, a in enumerate(antennas)}
    design = np.zeros((len(pairs), len(antennas)))
    target = np.zeros(len(pairs))
    weight = np.zeros(len(pairs))
    for row, (i, j, r, w) in enumerate(pairs):
        design[row, index[i]] = design[row, index[j]] = 1.0
        target[row], weight[row] = np.log(r), np.sqrt(w)
    if np.linalg.matrix_rank(design) < len(antennas):
        return {}
    solution, *_ = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)
    return {a: float(np.exp(solution[index[a]])) for a in antennas}


def model_check(ms, field: str, antennas: list[str], image, *, standard: bool = True,
                engine: Optional[str] = None, max_rows: int = CHECK_ROWS) -> dict:
    """Calibrated amplitude over model amplitude of ``field`` on the baselines among ``antennas``.

    Returns
    -------
    dict
        ``field``, ``antennas`` (the names checked), ``spws`` (subband -> hand label -> ``antennas`` {name: ratio},
        ``baselines`` {"A-B": ratio}, ``median`` (of the baselines)), ``rows`` (rows used) and ``seconds``.
        A ratio of 1 is a perfect match; 1.05 means the calibrated data are 5% above the model.
    """
    started = _time.perf_counter()
    ms = str(ms)
    engine = engine or table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    if field not in setup["field_names"]:
        raise ValueError(f"{Path(ms).name} has no field {field!r}")
    field_id = setup["field_names"].index(field)
    names = list(setup["antenna_names"])
    wanted = {name.lower() for name in antennas}
    ids = np.array([k for k, name in enumerate(names) if name.lower() in wanted], dtype=np.int64)
    flux, l, m, _ = model_components(image)
    ncorr = int(setup["ncorr"])
    corr_types = setup.get("corr_types") or _corr_types(ms, engine)
    hands = np.flatnonzero(_corr_parallel(corr_types, ncorr))
    labels = _hand_labels(corr_types, hands)
    view = corrected_view(ms, "CORRECTED_DATA", {field_id}, setup=setup, columns=layout["columns"])
    # (spw, hand, a1, a2) -> [sum |V||M|, sum |M|^2, cells]
    sums: dict[tuple[int, int, int, int], np.ndarray] = {}
    rows_used = 0
    selected = select_runs(layout, fields={field_id})
    for chunk in split_rows(run_specs(layout, selected), max_rows):
        block = read_view_runs(chunk, ["CORRECTED_DATA", "FLAG", "UVW", "ANTENNA1", "ANTENNA2"], engine=engine,
                               view=view)
        if not block:
            continue
        a1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
        a2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
        keep = np.isin(a1, ids) & np.isin(a2, ids) & (a1 != a2)
        if not keep.any():
            continue
        ddid = np.asarray(block["DATA_DESC_ID"], dtype=np.int64)
        uvw = np.asarray(block["UVW"], dtype=np.float64)
        data = np.asarray(block["CORRECTED_DATA"])
        flags = np.asarray(block["FLAG"], dtype=bool)
        for desc in np.unique(ddid[keep]):
            rows = np.flatnonzero(keep & (ddid == desc))
            spw = int(setup["ddid_to_spw"][int(desc)])
            freqs = np.asarray(setup["chan_freq"][spw], dtype=np.float64)
            amp_model = np.abs(predict(uvw[rows], freqs, flux, l, m,
                                       scale=spectral_scale(freqs, flux, field, standard=standard)))
            keys = a1[rows] * len(names) + a2[rows]
            unique, inverse = np.unique(keys, return_inverse=True)
            for hand in hands:
                values = data[rows, :, hand]
                good = ~flags[rows, :, hand] & np.isfinite(values) & (amp_model > 0)
                cross = np.where(good, np.abs(values) * amp_model, 0.0).sum(axis=1)
                power = np.where(good, amp_model ** 2, 0.0).sum(axis=1)
                cells = good.sum(axis=1).astype(np.float64)
                per = [np.bincount(inverse, weights=w, minlength=unique.size) for w in (cross, power, cells)]
                for k, key in enumerate(unique):
                    if per[2][k] <= 0:
                        continue
                    slot = (spw, int(hand), int(key) // len(names), int(key) % len(names))
                    sums[slot] = sums.get(slot, np.zeros(3)) + np.array([per[0][k], per[1][k], per[2][k]])
            rows_used += rows.size
    report: dict = {"field": field, "antennas": [names[k] for k in ids], "spws": {}, "rows": int(rows_used)}
    for spw in sorted({key[0] for key in sums}):
        for hand in hands:
            baselines = {(i, j): (float(v[0] / v[1]), float(v[2])) for (s, h, i, j), v in sums.items()
                         if s == spw and h == hand and v[1] > 0}
            if not baselines:
                continue
            ratios = antenna_ratios(baselines)
            report["spws"].setdefault(str(spw), {})[labels[int(hand)]] = {
                "antennas": {names[a]: r for a, r in sorted(ratios.items())},
                "baselines": {f"{names[i]}-{names[j]}": r for (i, j), (r, _) in sorted(baselines.items())},
                "median": float(np.median([r for r, _ in baselines.values()])),
            }
    report["seconds"] = _time.perf_counter() - started
    return report


def _corr_types(ms, engine: str) -> list[int]:
    """CORR_TYPE of the first POLARIZATION row (casacore Stokes codes); empty when it cannot be read."""
    try:
        handle = open_table(str(Path(ms) / "POLARIZATION"), engine=engine)
        try:
            return [int(code) for code in np.ravel(np.asarray(handle.getcol("CORR_TYPE", start=0, nrow=1)))]
        finally:
            handle.close()
    except Exception:  # noqa: BLE001 - the hands are then labelled by their slot
        return []


def _hand_labels(corr_types, hands) -> dict[int, str]:
    """Label (RR, LL, XX, YY) of each parallel-hand correlation slot."""
    codes = {5: "RR", 8: "LL", 9: "XX", 12: "YY", 1: "I"}
    types = list(corr_types or [])
    return {int(h): codes.get(int(types[h]), f"corr{int(h)}") if h < len(types) else f"corr{int(h)}" for h in hands}


def worst(report: dict) -> tuple[str, str, str, float]:
    """The antenna ratio furthest from 1 in a :func:`model_check` report: (antenna, subband, hand, ratio)."""
    found = ("", "", "", 1.0)
    for spw, per_hand in report.get("spws", {}).items():
        for hand, entry in per_hand.items():
            for name, ratio in entry.get("antennas", {}).items():
                if abs(np.log(ratio)) > abs(np.log(found[3])):
                    found = (name, spw, hand, float(ratio))
    return found
