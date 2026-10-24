"""Split one source out of a measurement set with channel and time averaging (the dask-ms backend's split).

:func:`split` writes what ``casatasks.mstransform(field=..., datacolumn=..., chanaverage=True,
timeaverage=True, keepflags=True)`` writes, as a plain measurement set:

* time bins of ``timebin`` seconds per (scan, subband, baseline), counted from the first integration of
  the scan, and channel bins of ``chanbin`` channels;
* DATA: the WEIGHT_SPECTRUM-weighted mean of the unflagged samples of a bin, or of all of them when none
  is unflagged (the bin is then flagged); WEIGHT_SPECTRUM the sum of those weights and SIGMA_SPECTRUM
  ``1 / sqrt`` of it (-1 without weight); WEIGHT and SIGMA the means of the unflagged output channels (all of them when every
  one is flagged);
* TIME the mean of the integrations of a bin up to its last one with an unflagged sample, and INTERVAL the
  sum of theirs; EXPOSURE the sum over the integrations with an unflagged sample, and TIME_CENTROID and UVW
  their means weighted by the summed WEIGHT (all integrations when none has one); FLAG_ROW False, as
  mstransform leaves it;
* the subtables of the input, with FIELD reduced to the source (FIELD_ID 0) and SPECTRAL_WINDOW
  describing the averaged channels.

Every rule was read off mstransform's own output for GL048 and checked to float32 rounding. The input
is read by parallel worker processes (one job per group of scans); only the averaged rows, a few percent
of the input, come back to be sorted by time and baseline and written. When the calibration of the field
was deferred (:mod:`vlbipy.solvers.corrected`), the corrected column is computed from DATA as it is read,
the way AIPS's SPLIT applies its CL table.
"""
from __future__ import annotations

import shutil
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .applycal import read_view_runs
from .corrected import view as corrected_view
from .msio import read_layout, read_setup, run_specs, select_runs, table_engine
from .workers import run_jobs

logger = get_logger()

#: Input rows per worker job (whole scans are never split between jobs).
SPLIT_ROWS = 200_000
#: Input rows reduced at a time inside a job: the full-resolution float64/complex128 temporaries of the
#: channel averaging exist for this many rows only, not for the whole job (16 jobs of 200 000 rows at
#: 32 channels and 4 correlations held about 0.7 GB each).
REDUCE_ROWS = 20_000
#: Main-table scalar columns copied from the first integration of a bin.
_CARRIED = ("ANTENNA1", "ANTENNA2", "ARRAY_ID", "FEED1", "FEED2", "OBSERVATION_ID", "PROCESSOR_ID", "STATE_ID")
#: Main-table columns stored in tiled storage managers in the output (as CASA lays out an MS).
_TILED_OUTPUT = ("DATA", "FLAG", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "FLAG_CATEGORY", "WEIGHT", "SIGMA",
                 "UVW")
#: Tolerance on bin edges [s] (TIME values are not exact multiples of the integration time).
_EDGE = 1e-6


def _channel_bins(nchan: int, chanbin: int) -> np.ndarray:
    """Start channel of every output channel (the last bin may be narrower, as in mstransform)."""
    return np.arange(0, nchan, max(1, int(chanbin)))


def split_job(*, specs: list[dict], engine: str, column: str, columns: list[str], scan_starts: dict,
              timebin: float, chanbin: int, view: Optional[dict] = None) -> dict:
    """Worker job: average the rows of ``specs`` (whole scans of one field) into output rows.

    With ``view`` (:func:`vlbipy.solvers.corrected.view`) the corrected data, flags and weights of its fields
    are computed from DATA while they are read, as applycal would have written them.
    """
    weight_spectrum = "WEIGHT_SPECTRUM" in columns
    names = [column, "FLAG", "WEIGHT", "UVW", "TIME", "TIME_CENTROID", "EXPOSURE", "INTERVAL", "FIELD_ID",
             *_CARRIED]
    names += ["WEIGHT_SPECTRUM"] if weight_spectrum else []
    block = read_view_runs(specs, [n for n in names if n in columns or n == column], engine=engine, view=view)
    if not block:
        return {}
    vis = block.pop(column)
    flag = np.asarray(block.pop("FLAG"), dtype=bool)
    weight = np.asarray(block["WEIGHT"], dtype=np.float64)
    ws = (np.asarray(block.pop("WEIGHT_SPECTRUM"), dtype=np.float64) if weight_spectrum
          else np.broadcast_to(weight[:, None, :], vis.shape))
    nrow, nchan, ncorr = vis.shape
    time = np.asarray(block["TIME"], dtype=np.float64)
    scan = np.asarray(block["SCAN_NUMBER"], dtype=np.int64)
    ddid = np.asarray(block["DATA_DESC_ID"], dtype=np.int64)
    ant1 = np.asarray(block["ANTENNA1"], dtype=np.int64)
    ant2 = np.asarray(block["ANTENNA2"], dtype=np.int64)
    if timebin > 0:
        start = np.array([scan_starts[int(s)] for s in scan]) if scan.size else np.zeros(0)
        tbin = np.floor((time - start + _EDGE) / timebin).astype(np.int64)
    else:
        tbin = np.round(time * 1e3).astype(np.int64)        # every integration its own bin
    order = np.lexsort((tbin, ant2, ant1, ddid, scan))
    keys = np.stack([scan, ddid, ant1, ant2, tbin], axis=1)[order]
    starts = np.flatnonzero(np.concatenate([[True], np.any(keys[1:] != keys[:-1], axis=1)]))
    bin_of_row = np.empty(nrow, dtype=np.int64)
    bin_of_row[order] = np.repeat(np.arange(starts.size), np.diff(np.concatenate([starts, [nrow]])))

    def rows_sum(values):
        return np.add.reduceat(values[order], starts, axis=0)

    # Channel bins first (per row), then time bins (rows of a bin), REDUCE_ROWS input rows at a time so
    # that only output-sized sums are kept for the whole job.
    chan_starts = _channel_bins(nchan, chanbin)
    ok = ~flag
    shape = (starts.size, chan_starts.size, ncorr)
    num_ok, num_all = np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128)
    den_ok, den_all = np.zeros(shape), np.zeros(shape)
    cnt_ok = np.zeros(shape, dtype=np.int64)
    for first in range(0, nrow, REDUCE_ROWS):
        rows = slice(first, first + REDUCE_ROWS)
        v, w, flag_ok = vis[rows], ws[rows], ok[rows]
        w_ok = np.where(flag_ok, w, 0.0)
        # The rows of this block by output bin: one sum per bin present, added to that bin.
        bins = bin_of_row[rows]
        by_bin = np.argsort(bins, kind="stable")
        firsts = np.flatnonzero(np.concatenate([[True], bins[by_bin][1:] != bins[by_bin][:-1]]))
        present = bins[by_bin][firsts]
        for total, values in ((num_ok, w_ok * v), (den_ok, w_ok), (cnt_ok, flag_ok.astype(np.int64)),
                              (num_all, w * v), (den_all, w)):
            total[present] += np.add.reduceat(np.add.reduceat(values, chan_starts, axis=1)[by_bin], firsts, axis=0)
    out_flag = cnt_ok == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        data = np.where(out_flag, num_all / np.where(den_all > 0, den_all, 1.0),
                        num_ok / np.where(den_ok > 0, den_ok, 1.0))
        out_ws = np.where(out_flag, den_all, den_ok)
        # mstransform marks a sample without weight with a sigma of -1.
        out_ss = np.where(out_ws > 0, 1.0 / np.sqrt(np.where(out_ws > 0, out_ws, 1.0)), -1.0)
    good_chan = ~out_flag
    n_good = good_chan.sum(axis=1)
    has_good = n_good > 0
    out_weight = np.where(has_good, np.where(good_chan, out_ws, 0).sum(axis=1) / np.maximum(n_good, 1),
                          out_ws.mean(axis=1))
    out_sigma = np.where(has_good, np.where(good_chan, out_ss, 0).sum(axis=1) / np.maximum(n_good, 1),
                         out_ss.mean(axis=1))
    # Row (integration) level: what had an unflagged sample, weighted by its summed WEIGHT.
    row_ok = ok.any(axis=(1, 2))
    row_w = np.where(row_ok, weight.sum(axis=1), 0.0)
    count = rows_sum(np.ones(nrow))
    n_row_ok = rows_sum(row_ok.astype(np.float64))
    w_sum = rows_sum(row_w)
    centroid_ok = rows_sum(row_w * np.asarray(block["TIME_CENTROID"], dtype=np.float64))
    centroid_all = rows_sum(np.asarray(block["TIME_CENTROID"], dtype=np.float64))
    uvw = np.asarray(block["UVW"], dtype=np.float64)
    uvw_ok, uvw_all = rows_sum(row_w[:, None] * uvw), rows_sum(uvw)
    use_ok = w_sum > 0
    exposure = np.asarray(block["EXPOSURE"], dtype=np.float64)
    # TIME and INTERVAL cover the integrations of a bin up to its last one with data: flagged integrations
    # before it count, flagged ones after it do not (all of them when the bin has no data at all).
    last_ok = np.maximum.reduceat(np.where(row_ok, time, -np.inf)[order], starts)
    upto = (time <= last_ok[bin_of_row]) | ~np.isfinite(last_ok[bin_of_row])
    n_upto = rows_sum(upto.astype(np.float64))
    out = {
        "DATA": data.astype(np.complex64), "FLAG": out_flag,
        "WEIGHT_SPECTRUM": out_ws.astype(np.float32), "SIGMA_SPECTRUM": out_ss.astype(np.float32),
        "WEIGHT": out_weight.astype(np.float32), "SIGMA": out_sigma.astype(np.float32),
        "TIME": rows_sum(np.where(upto, time, 0.0)) / n_upto,
        "INTERVAL": rows_sum(np.where(upto, np.asarray(block["INTERVAL"], dtype=np.float64), 0.0)),
        "EXPOSURE": np.where(n_row_ok > 0, rows_sum(np.where(row_ok, exposure, 0.0)), rows_sum(exposure)),
        "TIME_CENTROID": np.where(use_ok, centroid_ok / np.where(use_ok, w_sum, 1.0), centroid_all / count),
        "UVW": np.where(use_ok[:, None], uvw_ok / np.where(use_ok, w_sum, 1.0)[:, None], uvw_all / count[:, None]),
        "SCAN_NUMBER": keys[starts, 0], "DATA_DESC_ID": keys[starts, 1],
        "FIELD_ID": np.asarray(block["FIELD_ID"], dtype=np.int32)[order][starts],
    }
    first = order[starts]
    for name in _CARRIED:
        if name in block:
            out[name] = np.asarray(block[name])[first]
    return out


def _write_ms(source_ms: Path, part: str, target: Path, rows: dict, field_id: int, setup: dict, chanbin: int,
              column_names: list[str], engine: str) -> None:
    """Create ``target`` with the main-table columns of ``part`` and the averaged ``rows``, then its subtables.

    Written with :mod:`vlbipy.casacore_format` (no casacore library): bulk columns in tiled storage managers,
    the scalars in a StandardStMan; subtables copied file by file, then FIELD reduced to the source and
    SPECTRAL_WINDOW describing the averaged channels.
    """
    import copy

    from ..casacore_format.aipsio import TableRef
    from ..casacore_format.table import Table
    from ..casacore_format.writer import create_table

    with Table(part) as template:
        descs = [copy.deepcopy(c) for c in template.dat.columns]
        keywords = dict(template.dat.keywords)
        subtables = {name: template.subtable(name) for name, value in keywords.items()
                     if isinstance(value, TableRef)}
        for child in Path(part).iterdir():
            if child.is_dir() and (child / "table.dat").is_file() and child.name not in subtables:
                subtables[child.name] = child
    dropped = {"CORRECTED_DATA", "MODEL_DATA"} | {c.name for c in descs if c.name not in column_names
                                                  and c.name != "FLAG_CATEGORY"}
    if "WEIGHT_SPECTRUM" not in rows:
        dropped |= {"WEIGHT_SPECTRUM", "SIGMA_SPECTRUM"}
    columns = [c for c in descs if c.name not in dropped]
    nrow = len(rows["TIME"])
    for column in columns:
        column.shape_col_def, column.shape_col = False, ()
        if column.name in ("DATA", "FLAG", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "FLAG_CATEGORY"):
            # The channel axis changes: cube columns declared with a fixed shape become variable-shaped.
            column.shape, column.option = None, 0
    values = {name: values for name, values in rows.items() if name in {c.name for c in columns}}
    if nrow:
        # field_id None: all fields split, keep the ids; else one source -> FIELD_ID 0.
        values["FIELD_ID"] = np.asarray(rows["FIELD_ID"], dtype=np.int32) if field_id is None \
            else np.zeros(nrow, dtype=np.int32)
        values["FLAG_ROW"] = np.zeros(nrow, dtype=bool)
    tiled = {}
    for column in columns:
        if column.name not in _TILED_OUTPUT:
            continue
        if column.name in values:
            tiled[column.name] = [tuple(np.shape(values[column.name])[1:])] * nrow
        else:
            tiled[column.name] = [None] * nrow                 # e.g. FLAG_CATEGORY: cells stay undefined
    keywords = {name: value for name, value in keywords.items() if name not in subtables}
    keywords.update({name: TableRef(f"././{name}") for name in subtables})
    create_table(target, columns, nrow, values, keywords=keywords, info=("Measurement Set", ""), tiled=tiled)
    for name, path in subtables.items():
        shutil.copytree(Path(path).resolve(), target / name)
    if field_id is not None:
        _reduce_field(target / "FIELD", field_id)
    _average_spectral_windows(target / "SPECTRAL_WINDOW", chanbin)


def _reduce_field(path: Path, field_id: int) -> None:
    """Keep only row ``field_id`` of the FIELD subtable (the split source becomes FIELD_ID 0)."""
    from ..casacore_format.writer import rewrite_table, select_rows
    rewrite_table(path, lambda values, nrow: (select_rows(values, [field_id]), 1))


def _average_spectral_windows(path: Path, chanbin: int) -> None:
    """Describe the averaged channels in SPECTRAL_WINDOW (frequency means, widths summed)."""
    from ..casacore_format.writer import rewrite_table

    def average(values: dict, nrow: int):
        names = [n for n in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION") if n in values]
        freqs, others = [], {n: [] for n in names}
        for row in range(nrow):
            freq = np.asarray(values["CHAN_FREQ"][row], dtype=np.float64).reshape(-1)
            starts = _channel_bins(freq.size, chanbin)
            counts = np.diff(np.concatenate([starts, [freq.size]]))
            freqs.append(np.add.reduceat(freq, starts) / counts)
            for n in names:
                others[n].append(np.add.reduceat(np.asarray(values[n][row], dtype=np.float64).reshape(-1),
                                                    starts))
        values["CHAN_FREQ"] = freqs
        values.update(others)
        values["NUM_CHAN"] = np.array([f.size for f in freqs], dtype=np.int32)
        values["REF_FREQUENCY"] = np.array([f[0] for f in freqs], dtype=np.float64)
        return values, nrow

    rewrite_table(path, average)


def split(ms, target, field: str = "", *, column: str = "CORRECTED_DATA", chanbin: int = 1,
                timebin: float = 0.0, spws=None, workers: Optional[int] = None) -> dict:
    """Average the rows of ``field`` into the new measurement set ``target`` (see the module docstring).

    Parameters
    ----------
    field : str
        Field to split; empty keeps every field (their FIELD_IDs are preserved, for band splits).
    column : str
        Input column written to the output DATA.
    chanbin : int
        Channels per output channel (1 keeps them all).
    timebin : float
        Seconds per output integration (0 keeps them all).
    spws : list of int, optional
        Restrict the split to these subbands (all fields).

    Returns
    -------
    dict
        ``rows_in``, ``rows_out`` and ``seconds``.
    """
    started = _time.perf_counter()
    ms, target = Path(ms), Path(target)
    engine = table_engine()
    setup = read_setup(ms, engine=engine)
    layout = read_layout(ms, engine=engine)
    field_id = None
    if field:
        if field not in setup["field_names"]:
            raise ValueError(f"{ms.name} has no field {field!r}")
        field_id = setup["field_names"].index(field)
    ddids = None
    if spws is not None:
        wanted = {int(s) for s in spws}
        ddids = {d for d, s in enumerate(setup["ddid_to_spw"]) if int(s) in wanted}
        if not ddids:
            raise ValueError(f"{ms.name} has no data in subband(s) {sorted(wanted)}")
    field_ids = {field_id} if field_id is not None else set(range(len(setup["field_names"])))
    # A deferred calibration (vlbipy.solvers.corrected) is applied while the data are read.
    view = corrected_view(ms, column, field_ids, setup=setup, columns=layout["columns"])
    if column not in layout["columns"] and view is None:
        raise RuntimeError(f"{ms.name} has no {column} column")
    selected = select_runs(layout, fields=field_ids, ddids=ddids)
    scan_starts: dict[int, float] = {}
    for index in selected:
        scan = int(layout["scan"][index])
        scan_starts[scan] = min(scan_starts.get(scan, np.inf), float(layout["t_min"][index]))
    # Whole scans per job: a time bin never straddles two jobs.
    by_scan: dict[int, list[int]] = {}
    for index in selected:
        by_scan.setdefault(int(layout["scan"][index]), []).append(int(index))
    groups, current, filled = [], [], 0
    for scan in sorted(by_scan):
        size = int(sum(layout["nrow"][i] for i in by_scan[scan]))
        if current and filled + size > SPLIT_ROWS:
            groups.append(current)
            current, filled = [], 0
        current += by_scan[scan]
        filled += size
    if current:
        groups.append(current)
    jobs = [{"specs": run_specs(layout, group), "engine": engine, "column": column, "columns": layout["columns"],
             "scan_starts": {s: scan_starts[s] for s in {int(layout["scan"][i]) for i in group}},
             "timebin": float(timebin), "chanbin": int(chanbin), "view": view} for group in groups]
    parts = [p for p in run_jobs("vlbipy.solvers.split:split_job", jobs, workers) if p]
    if not parts:
        raise RuntimeError(f"{ms.name}: no rows of {field!r}")
    rows = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    order = np.lexsort((rows["DATA_DESC_ID"], rows["ANTENNA2"], rows["ANTENNA1"], rows["TIME"]))
    rows = {name: values[order] for name, values in rows.items()}
    if "WEIGHT_SPECTRUM" not in layout["columns"]:
        rows.pop("WEIGHT_SPECTRUM", None)
        rows.pop("SIGMA_SPECTRUM", None)
    _write_ms(ms, layout["parts"][0], target, rows, field_id, setup, chanbin,
              [c for c in layout["columns"] if c not in ("CORRECTED_DATA", "MODEL_DATA")], engine)
    report = {"rows_in": int(layout["nrow"][selected].sum()), "rows_out": int(len(rows["TIME"])),
              "seconds": _time.perf_counter() - started}
    logger.info("split[dask-ms]: {} field={} -> {} ({} -> {} rows, chanbin={}, timebin={:g}s{}) in {:.1f}s",
                ms.name, field, target.name, report["rows_in"], report["rows_out"], chanbin, timebin,
                ", calibrated while read" if view is not None else "", report["seconds"])
    return report
