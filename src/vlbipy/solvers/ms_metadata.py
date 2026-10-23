"""What ``casatools.msmetadata`` reports about a measurement set, read natively (:mod:`vlbipy.casacore_format`).

:func:`ms_summary` returns, in one pass over the index columns of the main table (every part of a Multi-MS) and
the small subtables, the quantities :meth:`vlbipy.backends.casa.CasaDataOps.get_metadata` asks msmd for, with
msmd's conventions: the observation's time range is the OBSERVATION table's TIME_RANGE, a scan's times are its
distinct TIME values, its antennas every ANTENNA1 and ANTENNA2 of its rows, its exposure the EXPOSURE of its
first row in subband 0, a field's phase centre the first term of PHASE_DIR (radians, as stored).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..casacore_format.table import Table
from .msio import ms_parts


def _subtable(ms: Path, name: str, columns: tuple) -> dict:
    """``{column: values}`` of a subtable (columns it lacks are left out)."""
    with Table(ms / name) as table:
        present = table.colnames()
        nrow = table.nrows()
        return {column: (table.getcol(column) if nrow else []) for column in columns if column in present}


def ms_summary(ms) -> dict:
    """The msmd quantities of a measurement set.

    Returns
    -------
    dict
        ``antenna_names``, ``telescopes`` (OBSERVATION TELESCOPE_NAME), ``field_names``, ``phase_dirs`` (nfield, 2)
        [rad], ``time_range`` (begin, end) [MJD s], ``chan_freq`` (list per spw, Hz), ``chan_width`` (list per spw),
        ``bandwidths`` (nspw,), ``corr_types`` (of polarization 0) and ``scans``: ``{scan: {"fields", "times",
        "antennas", "spws", "exposure"}}`` (sorted arrays; exposure 0.0 when the scan has no subband-0 row).
    """
    ms = Path(ms)
    antenna = _subtable(ms, "ANTENNA", ("NAME",))
    observation = _subtable(ms, "OBSERVATION", ("TELESCOPE_NAME", "TIME_RANGE"))
    field = _subtable(ms, "FIELD", ("NAME", "PHASE_DIR"))
    spw = _subtable(ms, "SPECTRAL_WINDOW", ("CHAN_FREQ", "CHAN_WIDTH", "TOTAL_BANDWIDTH"))
    description = _subtable(ms, "DATA_DESCRIPTION", ("SPECTRAL_WINDOW_ID",))
    polarization = _subtable(ms, "POLARIZATION", ("CORR_TYPE",))
    phase = field.get("PHASE_DIR", [])
    phase_dirs = np.array([np.asarray(p, dtype=np.float64).reshape(-1, 2)[0] for p in phase]).reshape(-1, 2)
    time_range = np.asarray(observation.get("TIME_RANGE", [[0.0, 0.0]]), dtype=np.float64).reshape(-1, 2)[0]
    ddid_spw = np.asarray(description.get("SPECTRAL_WINDOW_ID", []), dtype=np.int64)

    columns = {name: [] for name in ("SCAN_NUMBER", "FIELD_ID", "TIME", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID",
                                     "EXPOSURE")}
    for part in ms_parts(ms):
        with Table(part) as table:
            if not table.nrows():
                continue
            for name in columns:
                columns[name].append(np.asarray(table.getcol(name)))
    values = {name: (np.concatenate(v) if v else np.zeros(0)) for name, v in columns.items()}
    scan = values["SCAN_NUMBER"].astype(np.int64)
    order = np.argsort(scan, kind="stable")
    boundaries = np.flatnonzero(np.diff(scan[order])) + 1
    scans = {}
    for rows in np.split(order, boundaries) if order.size else []:
        spws = ddid_spw[values["DATA_DESC_ID"][rows].astype(np.int64)] if ddid_spw.size else \
            values["DATA_DESC_ID"][rows].astype(np.int64)
        in_spw0 = rows[spws == 0]
        exposure = 0.0
        if in_spw0.size:
            exposure = float(values["EXPOSURE"][in_spw0[np.argmin(values["TIME"][in_spw0])]])
        scans[int(scan[rows[0]])] = {
            "fields": np.unique(values["FIELD_ID"][rows].astype(np.int64)),
            "times": np.unique(values["TIME"][rows].astype(np.float64)),
            "antennas": np.unique(np.concatenate([values["ANTENNA1"][rows], values["ANTENNA2"][rows]]).astype(np.int64)),
            "spws": np.unique(spws), "exposure": exposure}
    return {"antenna_names": [str(n) for n in antenna.get("NAME", [])],
            "telescopes": [str(n) for n in observation.get("TELESCOPE_NAME", [])],
            "field_names": [str(n) for n in field.get("NAME", [])], "phase_dirs": phase_dirs,
            "time_range": (float(time_range[0]), float(time_range[1])),
            "chan_freq": [np.asarray(f, dtype=np.float64).reshape(-1) for f in spw.get("CHAN_FREQ", [])],
            "chan_width": [np.asarray(w, dtype=np.float64).reshape(-1) for w in spw.get("CHAN_WIDTH", [])],
            "bandwidths": np.asarray(spw.get("TOTAL_BANDWIDTH", []), dtype=np.float64),
            "corr_types": [int(c) for c in np.asarray(polarization.get("CORR_TYPE", [[]])[0]).reshape(-1)]
            if len(polarization.get("CORR_TYPE", [])) else [],
            "scans": scans}
