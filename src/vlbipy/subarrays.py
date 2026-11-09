"""Sub-arrays of a fringe fit: groups of antennas whose solutions are referenced to different antennas.

A phase calibrator that is heavily resolved - scatter-broadened at low frequency close to the Galactic plane -
is seen only on the shortest baselines. On EM178 (18 cm) J1832-1035 is a ~160 mas Gaussian: the e-MERLIN
baselines (JB, DA, PI, KN, DE: 10-60 km) detect it at fringe SNR 30-130 and EF-WB at ~150, while every baseline
longer than ~200 km, EF to the e-MERLIN stations included, is noise. A global fringe fit referenced to EF then
finds nothing for the e-MERLIN stations and their data are flagged, although the source calibrates their
mutual baselines perfectly well.

Solved on their own, with a reference antenna among them, those stations form a second sub-array. Its
solutions calibrate the baselines inside it and nothing else: a baseline between two sub-arrays joins
solutions referenced to two different antennas, whose atmospheric phases nobody measured, so its data are
flagged wherever those solutions are applied. The fringe fit writes the groups into its table directory
(:data:`SUBARRAY_FILE`, so a copy of the table - the smoothed one - carries them); the per-source split flags
the baselines between groups (:func:`flag_cross_subarrays`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from loguru import logger

#: The sub-array groups of a fringe table, ``{solution field: [[antenna, ...], ...]}``, inside the table directory.
#: Antennas listed in no group belong to the main array (the one of the global reference antenna).
SUBARRAY_FILE = "vlbipy.subarrays.json"


def write_subarrays(table_path, groups: dict[str, list[list[str]]]) -> None:
    """Record the sub-array groups of a fringe table (removes the record when there are none)."""
    path = Path(table_path) / SUBARRAY_FILE
    groups = {field: [sorted(group) for group in found if len(group) > 1] for field, found in groups.items()}
    groups = {field: found for field, found in groups.items() if found}
    if not groups:
        path.unlink(missing_ok=True)
        return
    if Path(table_path).is_dir():
        path.write_text(json.dumps(groups, indent=1, sort_keys=True))


def read_subarrays(table_path) -> dict[str, list[list[str]]]:
    """The sub-array groups recorded in a fringe table (``{}`` when there are none or the file is unreadable)."""
    path = Path(table_path) / SUBARRAY_FILE
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(field): [[str(name) for name in group] for group in found]
            for field, found in data.items() if isinstance(found, list)} if isinstance(data, dict) else {}


def station_clusters(positions: dict[str, Iterable[float]], max_km: float) -> list[list[str]]:
    """Groups of stations linked by baselines shorter than ``max_km`` (single linkage), largest first.

    Stations without a known position (all zeros) are left out. Singletons are not returned.
    """
    names = [name for name, xyz in positions.items() if np.any(np.asarray(xyz, float) != 0.0)]
    xyz = np.array([np.asarray(positions[name], float) for name in names]).reshape(-1, 3)
    parent = list(range(len(names)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.linalg.norm(xyz[i] - xyz[j]) <= max_km * 1e3:
                parent[root(i)] = root(j)
    clusters: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        clusters.setdefault(root(i), []).append(name)
    return sorted((c for c in clusters.values() if len(c) > 1), key=lambda c: (-len(c), c))


def group_labels(groups: list[list[str]], names: list[str]) -> np.ndarray:
    """Sub-array of every antenna of ``names``: 0 for the main array, ``k`` for the ``k``-th group."""
    labels = np.zeros(len(names), dtype=np.int64)
    upper = [str(name).strip().upper() for name in names]
    for index, group in enumerate(groups, start=1):
        for name in group:
            if str(name).upper() in upper:
                labels[upper.index(str(name).upper())] = index
    return labels


def cross_mask(antenna1: np.ndarray, antenna2: np.ndarray, names: list[str],
               group_sets: list[list[list[str]]]) -> np.ndarray:
    """Rows whose two antennas are in different sub-arrays of any of ``group_sets``."""
    mask = np.zeros(np.shape(antenna1), dtype=bool)
    for groups in group_sets:
        labels = group_labels(groups, names)
        mask |= labels[np.asarray(antenna1)] != labels[np.asarray(antenna2)]
    return mask


def groups_for_field(tables, field: str) -> list[list[list[str]]]:
    """The sub-array groups of the solutions that calibrate ``field``, one list per fringe table that has some.

    ``tables`` are the :class:`~vlbipy.models.CalTable` of the apply chain. A table solved on ``field`` calibrates
    it with its own solutions; otherwise a table with a ``gainfield`` transfers those fields' solutions to it.
    """
    out = []
    for table in tables:
        if not getattr(table, "path", "") or not (Path(table.path) / SUBARRAY_FILE).is_file():
            continue
        apply_to = {f for f in str(getattr(table, "apply_to", "") or "").split(",") if f}
        if apply_to and field not in apply_to:
            continue
        recorded = read_subarrays(table.path)
        solved = {f for f in str(table.field or "").split(",") if f}
        sources = [field] if field in solved else [f for f in str(table.gainfield or "").split(",") if f]
        for source in sources:
            groups = sorted(sorted(group) for group in recorded.get(source) or [])
            if groups and groups not in out:                    # the same groups from another fringe pass
                out.append(groups)
    return out


def flag_cross_subarrays(ms_path, group_sets: list[list[list[str]]], *, label: str = "") -> int:
    """Flag the rows of a (split) measurement set on baselines between sub-arrays; returns how many."""
    if not group_sets:
        return 0
    from casacore.tables import table
    with table(str(Path(ms_path) / "ANTENNA"), ack=False) as antennas:
        names = [str(n) for n in antennas.getcol("NAME")]
    flagged = 0
    with table(str(ms_path), readonly=False, ack=False) as data:
        if data.nrows() == 0:
            return 0
        rows = cross_mask(data.getcol("ANTENNA1"), data.getcol("ANTENNA2"), names, group_sets)
        if rows.any():
            flag = data.getcol("FLAG")
            already = flag[rows].all(axis=(1, 2))
            flag[rows] = True
            data.putcol("FLAG", flag)
            flag_row = data.getcol("FLAG_ROW")
            flag_row[rows] = True
            data.putcol("FLAG_ROW", flag_row)
            flagged = int((~already).sum())
    groups = "; ".join(" | ".join(",".join(g) for g in groups) for groups in group_sets)
    logger.info("split{}: flagged {} row(s) on baselines between the fringe-fit sub-arrays ({} and the main "
                "array): no solution relates their phases", f"[{label}]" if label else "", flagged, groups)
    return flagged


def describe(groups: Optional[dict[str, list[list[str]]]]) -> str:
    """``J1832-1035: CM,DA,DE,JB,KN,PI`` style summary of recorded groups."""
    return "; ".join(f"{field}: " + " | ".join(",".join(group) for group in found)
                     for field, found in sorted((groups or {}).items()))
