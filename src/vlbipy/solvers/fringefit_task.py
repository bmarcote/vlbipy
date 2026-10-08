"""A drop-in replacement for ``casatasks.fringefit`` for the dask-ms backend (numpy fringe solver).

:func:`run_fringefit` accepts the keyword set of the CASA task (``vis``, ``caltable``, the data
selection, the solve options and the on-the-fly prior lists). Each (field, scan) is one job run in
a worker process (:mod:`vlbipy.solvers.workers`): it reads the parallel hands of the selected
channels straight from the measurement set (:mod:`vlbipy.solvers.msio`), applies the priors, and
solves every solution interval with :mod:`vlbipy.solvers.fringe`. The parent only collects the
solutions and writes a CASA "Fringe Jones" table with
:func:`vlbipy.solvers.caltable.write_fringe_table`. The conventions (units, reference frequency
and time, parallactic angle, SNR) follow CASA so the table can be applied by ``applycal`` and read
by the rest of the pipeline unchanged.

Differences from CASA that are deliberate:

* the SNR column holds the AIPS FRING estimate of the FFT stage (the one ``minsnr`` is applied to);
  CASA writes a value that scales with the square root of the weights instead, so the two columns
  agree on what is detected but not on the numbers;
* the MODEL_DATA column is ignored (calibrators are treated as point sources, as the pipeline does);
* ``docallib=True`` (cal-library files) is not supported: pass the explicit parallel lists.
"""
from __future__ import annotations

import datetime as dt
import re
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .caltable import write_fringe_table
from .fringe import REFANT_SNR_SENTINEL, FringeData, fringefit_interval
from .msio import read_layout, read_setup, run_specs, select_runs, table_engine
from .workers import run_jobs

logger = get_logger()

_MJD_EPOCH = dt.datetime(1858, 11, 17)


# ---------------------------------------------------------------------------
# Selection parsing (the subset of the CASA MS-selection syntax the pipeline uses)
# ---------------------------------------------------------------------------
def _split_list(text) -> list[str]:
    """Split a comma-separated CASA selection string into stripped non-empty tokens."""
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def _expand_range(token: str, upper: int) -> list[int]:
    """Expand ``'a'``, ``'a~b'`` or ``'*'`` into integer ids (``upper`` = number of ids for ``'*'``)."""
    if token in ("*", ""):
        return list(range(upper))
    if "~" in token:
        lo, hi = token.split("~", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(token)]


def parse_names(selection, names: list[str]) -> list[int]:
    """Resolve a comma-separated list of names or ids (``'3C345,J1848+3219'``, ``'0,2'``) to ids; '' = all."""
    tokens = _split_list(selection)
    if not tokens:
        return list(range(len(names)))
    ids = []
    for token in tokens:
        if token in names:
            ids.append(names.index(token))
        elif re.fullmatch(r"\d+(~\d+)?", token):
            ids.extend(_expand_range(token, len(names)))
        else:
            raise ValueError(f"unknown selection {token!r}; known names: {names}")
    return ids


def parse_spw(selection, nspw: int, nchan: int) -> dict[int, np.ndarray]:
    """Parse a CASA spw/channel selection into ``{spw_id: channel indices}``.

    Supports ``''`` (everything), ``'*:6~57'``, ``'0,1'``, ``'0~2:4~60'`` and ``';'``-separated channel
    ranges; channel ranges are inclusive like CASA's.
    """
    tokens = _split_list(selection)
    if not tokens:
        return {s: np.arange(nchan) for s in range(nspw)}
    out: dict[int, np.ndarray] = {}
    for token in tokens:
        spw_part, _, chan_part = token.partition(":")
        chans = np.arange(nchan) if not chan_part else np.unique(np.concatenate(
            [np.asarray(_expand_range(part, nchan)) for part in chan_part.split(";")]))
        chans = chans[(chans >= 0) & (chans < nchan)]
        for spw in _expand_range(spw_part, nspw):
            out[spw] = np.union1d(out[spw], chans) if spw in out else chans
    return out


def parse_scans(selection) -> Optional[set[int]]:
    """Parse ``'1,2,4'`` / ``'3~8'`` into a set of scan numbers; '' -> None (all scans)."""
    tokens = _split_list(selection)
    if not tokens:
        return None
    scans: set[int] = set()
    for token in tokens:
        scans.update(_expand_range(token, 0))
    return scans


def parse_timerange(selection) -> Optional[tuple[float, float]]:
    """Parse ``'YYYY/MM/DD/hh:mm:ss[.s]~YYYY/MM/DD/hh:mm:ss[.s]'`` into MJD seconds; '' -> None."""
    text = str(selection or "").strip()
    if not text:
        return None
    if "~" not in text:
        raise ValueError(f"unsupported timerange {text!r} (expected 'start~end')")
    values = []
    for part in text.split("~", 1):
        part = part.strip()
        fmt = "%Y/%m/%d/%H:%M:%S.%f" if "." in part.split("/")[-1] else "%Y/%m/%d/%H:%M:%S"
        values.append((dt.datetime.strptime(part, fmt) - _MJD_EPOCH).total_seconds())
    return values[0], values[1]


def parse_antennas(selection, names: list[str]) -> tuple[Optional[set[int]], bool]:
    """Parse ``'EF,JB,WB&'`` -> (antenna ids, among_only). '' -> (None, False).

    A trailing ``&`` restricts the solve to the baselines *among* the listed antennas (CASA
    semantics); without it, any baseline that includes one of them is kept.
    """
    text = str(selection or "").strip()
    if not text:
        return None, False
    among = text.endswith("&")
    ids = set(parse_names(text.rstrip("&"), names))
    return ids, among


def parse_solint(solint) -> float:
    """Return the solution interval in seconds (``inf`` -> inf, ``'int'`` -> 0, ``'60s'``, ``'2min'``, 30)."""
    text = str(solint or "inf").strip().lower()
    if text in ("inf", ""):
        return float("inf")
    if text == "int":
        return 0.0
    match = re.fullmatch(r"([0-9.]+)\s*(s|sec|min|h)?", text)
    if not match:
        raise ValueError(f"unsupported solint {solint!r}")
    value = float(match.group(1))
    return value * {"s": 1.0, "sec": 1.0, None: 1.0, "min": 60.0, "h": 3600.0}[match.group(2)]


def _interval_edges(times: np.ndarray, solint_sec: float) -> list[tuple[float, float]]:
    """Split the sorted unique ``times`` of a scan into solution intervals of ``solint_sec`` (inf = one)."""
    if not np.isfinite(solint_sec):
        return [(times[0], times[-1])]
    if solint_sec <= 0:
        return [(t, t) for t in times]
    edges = []
    start = times[0]
    while start <= times[-1]:
        inside = times[(times >= start) & (times < start + solint_sec)]
        if inside.size:
            edges.append((inside[0], inside[-1]))
        start += solint_sec
    return edges


def _first_refant_with_data(chain: list[int], antenna1: np.ndarray, antenna2: np.ndarray, flag: np.ndarray) -> int:
    """First antenna of the chain that has unflagged data in the interval (CASA findRefAntWithData); -1 if none."""
    good = ~flag.all(axis=(1, 2))
    present = set(antenna1[good]) | set(antenna2[good])
    for ant in chain:
        if ant in present:
            return ant
    return -1


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------
def _solve_block(block: dict, setup: dict, spw_sel: dict, spw_groups: list[list[int]], refant_chain: list[int],
                 solint_sec: float, solve_kwargs: dict, field_id: int, scan: int) -> list[dict]:
    """Solve every (interval, spw group) of one scan block; returns the table rows as dicts of arrays.

    ``block`` comes from :func:`vlbipy.solvers.calblock.load_block`: its cubes already hold only the selected
    channels (``block["chan_freq"]`` are their frequencies) and the parallel hands.
    """
    nant = len(setup["antenna_names"])
    rows = []
    unique_times = np.unique(block["time"])
    for t_lo, t_hi in _interval_edges(unique_times, solint_sec):
        in_time = (block["time"] >= t_lo) & (block["time"] <= t_hi)
        for group in spw_groups:
            keep = in_time & np.isin(block["spw"], group)
            if not keep.any():
                continue
            # The frequency grid of this group: the selected channels of every spw in it.
            chans = spw_sel[group[0]]
            if any(not np.array_equal(spw_sel[s], chans) for s in group):
                raise ValueError("combine='spw' needs the same channel selection in every subband")
            freq = block["chan_freq"]
            refant = _first_refant_with_data(refant_chain, block["antenna1"][keep], block["antenna2"][keep],
                                             block["flag"][keep])
            if refant < 0:
                logger.warning("fringefit: scan {} spw {}: no reference antenna with data; interval skipped",
                               scan, group)
                continue
            data = FringeData.from_baselines(block["vis"][keep], block["flag"][keep], block["weight"][keep],
                                             block["antenna1"][keep], block["antenna2"][keep], block["time"][keep],
                                             block["spw"][keep], freq,
                                             nant=nant, f_ref_hz=group_reference_freq(freq, group))
            t_sol = 0.5 * (t_lo + t_hi)
            solution = fringefit_interval(data, refant, t_sol=t_sol, **solve_kwargs)
            table = solution.to_fparam()
            rows.append({"time": t_sol, "interval": 0.0 if not np.isfinite(solint_sec) else solint_sec,
                         "field_id": field_id, "scan": scan, "spw": int(min(group)), "refant": refant,
                         "f_ref_hz": solution.f_ref_hz, "nchan_sel": int(chans.size * len(group)), **table,
                         "n_ok": int((~solution.flag).sum()), "snr_median": float(np.median(
                             solution.snr[~solution.flag])) if (~solution.flag).any() else 0.0})
    return rows


def group_reference_freq(chan_freq, group: list[int]) -> float:
    """Return the reference frequency of one solve group: the centre of its selected channels [Hz].

    It comes from the *selection*, not from the subbands that happen to hold data in one solution
    interval. ``combine='spw'`` pools several subbands into a single solution whose delay is
    referenced to this frequency and whose table stores one value for it, so an interval missing
    the antenna that holds an edge subband must not reference its delay to a narrower band.

    Parameters
    ----------
    chan_freq : float array (nspw, nchan_sel)
        Channel frequencies per subband [Hz], already restricted to the selected channels
        (:func:`vlbipy.solvers.calblock.load_block` output; rows of unselected subbands are zero).
    group : list of int
        Subbands solved together, all of them selected (one entry unless ``combine='spw'``).

    Returns
    -------
    float
    """
    freqs = np.concatenate([np.asarray(chan_freq[spw], dtype=np.float64).reshape(-1) for spw in group])
    return 0.5 * (float(freqs.min()) + float(freqs.max()))


def solve_scan_job(*, specs: list[dict], setup: dict, engine: str, columns: list[str], data_column: str,
                   spw_sel: dict, spw_groups: list[list[int]], refant_chain: list[int], solint_sec: float,
                   solve_kwargs: dict, antennas: Optional[set], among: bool, timerange: Optional[tuple],
                   parang: bool, priors: list) -> list[dict]:
    """Worker job: read one (field, scan), apply the priors and fringe fit its solution intervals.

    Returns the table rows of the scan as dicts of small arrays (see :func:`_solve_block`).
    """
    from .calblock import load_block
    block = load_block(specs, setup=setup, engine=engine, columns=columns, data_column=data_column, chans=spw_sel,
                       parallel_hands=True, timerange=timerange, antennas=antennas, among=among, parang=parang,
                       priors=priors)
    if block is None:
        return []
    return _solve_block(block, setup, spw_sel, spw_groups, refant_chain, solint_sec, solve_kwargs, block["field"],
                        int(specs[0]["scan"]))


def run_fringefit(vis: str, caltable: str, *, field="", spw="", scan="", timerange="", antenna="",
                  solint="inf", combine="", refant="", minsnr=3.0, zerorates=False, globalsolve=True, niter=100,
                  paramactive=None, delaywindow=None, ratewindow=None, parang=False, gaintable=None,
                  gainfield=None, interp=None, spwmap=None, docallib=False, workers: Optional[int] = None,
                  data_column: str = "DATA", **ignored) -> Path:
    """Fringe fit ``vis`` into the CASA-format table ``caltable`` (``casatasks.fringefit`` keyword set).

    Parameters follow the CASA task. ``paramactive`` is ``[delay, rate, dispersive]`` (default
    delay and rate). ``gaintable``/``gainfield``/``interp``/``spwmap`` are the explicit prior
    lists; ``docallib`` is rejected. ``workers`` is the number of worker processes (default
    :func:`vlbipy.solvers.workers.default_workers`; 1 solves in this process). Unknown keywords
    (``corrdepflags``, ``selectdata``, ...) are ignored. Returns the table path.
    """
    if docallib:
        raise NotImplementedError("run_fringefit: cal-library files are not supported; pass gaintable lists")
    t_start = _time.time()
    vis, caltable = str(vis), Path(caltable)
    engine = table_engine()
    setup = read_setup(vis, engine=engine)
    layout = read_layout(vis, engine=engine)
    nspw, nchan = setup["chan_freq"].shape
    field_ids = set(parse_names(field, setup["field_names"]))
    spw_sel = parse_spw(spw, nspw, nchan)
    scans = parse_scans(scan)
    trange = parse_timerange(timerange)
    antennas, among = parse_antennas(antenna, setup["antenna_names"])
    refant_chain = parse_names(refant, setup["antenna_names"]) if str(refant).strip() else []
    if not refant_chain:
        raise ValueError("run_fringefit: a reference antenna (chain) is required")
    solint_sec = parse_solint(solint)
    combine_spw = "spw" in str(combine)
    spw_groups = [sorted(spw_sel)] if combine_spw else [[s] for s in sorted(spw_sel)]
    active = tuple(bool(x) for x in (paramactive or [True, True, False]))
    solve_kwargs = {"active": active, "minsnr": float(minsnr), "zerorates": bool(zerorates),
                    "global_solve": bool(globalsolve), "max_iter": int(niter),
                    "delay_window_ns": tuple(delaywindow) if delaywindow else None,
                    "rate_window": tuple(ratewindow) if ratewindow else None}
    prior_entries = _prior_entries(gaintable, gainfield, interp, spwmap, setup["field_names"])
    # The prior tables are read once here (casatools) and travel to the workers as plain arrays. Every table
    # calibrates the weights, as in CASA: the fit itself works on unit vectors, but the amplitude tables (Tsys,
    # gain curve) set how much each baseline counts in it.
    from .apply import normalise_entries
    table_cache: dict = {}
    priors_by_field = {field_id: [entry[:4] + (True,) for entry in normalise_entries(
        prior_entries, cache=table_cache, field_dirs=setup["field_dirs"], target_field=field_id)]
        for field_id in sorted(field_ids)}
    logger.info("fringefit[dask-ms]: {} field={} spw={} scans={} solint={} combine={} refant={} minsnr={} "
                "paramactive={} zerorates={} parang={} priors={}", Path(vis).name, sorted(field_ids), spw or "all",
                scan or "all", solint, combine or "none", [setup["antenna_names"][a] for a in refant_chain],
                minsnr, list(active), zerorates, parang, [Path(str(e["path"])).name for e in prior_entries])

    ddids = {ddid for ddid, s in enumerate(setup["ddid_to_spw"]) if int(s) in spw_sel}
    selected = select_runs(layout, fields=field_ids, scans=scans, ddids=ddids, timerange=trange)
    if not selected.size:
        raise ValueError(f"run_fringefit: no data for field={field!r} scan={scan!r}")
    common = {"setup": setup, "engine": engine, "columns": layout["columns"], "data_column": data_column,
              "spw_sel": spw_sel, "spw_groups": spw_groups, "refant_chain": refant_chain, "solint_sec": solint_sec,
              "solve_kwargs": solve_kwargs, "antennas": antennas, "among": among, "timerange": trange,
              "parang": bool(parang)}
    keys = sorted({(int(layout["field"][i]), int(layout["scan"][i])) for i in selected})
    jobs = [dict(common, priors=priors_by_field[field_id],
                 specs=run_specs(layout, [i for i in selected if int(layout["field"][i]) == field_id
                                          and int(layout["scan"][i]) == scan_no]))
            for field_id, scan_no in keys]
    # Longest scans first: the pool then finishes with short jobs instead of waiting on one long straggler.
    order = np.argsort([-sum(spec["nrow"] for spec in job["specs"]) for job in jobs], kind="stable")
    results = run_jobs("vlbipy.solvers.fringefit_task:solve_scan_job", [jobs[i] for i in order], workers)
    rows = [r for group in results for r in group]
    if not rows:
        raise RuntimeError("run_fringefit: no solution interval had data")
    _write_table(caltable, vis, setup, rows, spw_sel, combine_spw)
    n_ok = sum(r["n_ok"] for r in rows)
    logger.info("fringefit[dask-ms]: {} interval(s) on {} scan(s), {} antenna solutions, median SNR {:.0f}; "
                "{} written in {:.2f}s", len(rows), len(jobs), n_ok,
                float(np.median([r["snr_median"] for r in rows if r["n_ok"]])) if n_ok else 0.0,
                caltable.name, _time.time() - t_start)
    return caltable


def _prior_entries(gaintable, gainfield, interp, spwmap, field_names: list[str]) -> list[dict]:
    """Turn the CASA parallel prior lists into :func:`vlbipy.solvers.apply.apply_tables` entries.

    A ``gainfield`` of names or ids becomes a list of field ids, ``''`` an empty list (all fields) and
    ``'nearest'`` stays ``"nearest"`` until the field being calibrated is known.
    """
    tables = [str(t) for t in (gaintable or [])] if not isinstance(gaintable, str) else _split_list(gaintable)
    entries = []
    for i, path in enumerate(tables):
        mapping = str((gainfield or [""] * len(tables))[i] if i < len(gainfield or []) else "").strip()
        if mapping.lower() == "nearest":
            fields = "nearest"
        else:
            fields = parse_names(mapping, field_names) if mapping else []
        entries.append({"path": path, "interp": (interp or [])[i] if i < len(interp or []) else "linear",
                        "spwmap": list((spwmap or [])[i]) if i < len(spwmap or []) and (spwmap or [])[i] else [],
                        "gainfield": fields, "calwt": False})
    return entries


def _write_table(caltable: Path, vis: str, setup: dict, rows: list[dict], spw_sel: dict, combine_spw: bool) -> None:
    """Write the collected interval solutions as one CASA table (one row per interval, spw and antenna)."""
    nant = len(setup["antenna_names"])
    nspw = setup["chan_freq"].shape[0]
    rows = sorted(rows, key=lambda r: (r["time"], r["spw"]))
    stack = {k: np.concatenate([r[k] for r in rows]) for k in ("fparam", "paramerr", "flag", "snr")}
    per_row = {"times": np.repeat([r["time"] for r in rows], nant),
               "field_ids": np.repeat([r["field_id"] for r in rows], nant),
               "spw_ids": np.repeat([r["spw"] for r in rows], nant),
               "scan_numbers": np.repeat([r["scan"] for r in rows], nant),
               "intervals": np.repeat([r["interval"] for r in rows], nant),
               "antenna_ids": np.tile(np.arange(nant), len(rows)),
               "refant_ids": np.repeat([r["refant"] for r in rows], nant)}
    # Table SPECTRAL_WINDOW: one channel per spw at the centre of the selected channels (the solver's f_ref).
    chan_freq, chan_width = setup["chan_freq"], np.abs(np.diff(setup["chan_freq"], axis=1)).mean(axis=1)
    spw_centre = np.array([0.5 * (chan_freq[s].min() + chan_freq[s].max()) for s in range(nspw)])
    spw_width = np.array([chan_freq.shape[1] * chan_width[s] for s in range(nspw)])
    for s, chans in spw_sel.items():
        if not combine_spw:
            spw_centre[s] = 0.5 * (chan_freq[s, chans].min() + chan_freq[s, chans].max())
        spw_width[s] = chans.size * chan_width[s]
    if combine_spw:
        # One solution per interval, stored in the lowest *selected* subband (not necessarily 0) and
        # referenced to the centre of the whole selection, which every interval shares.
        solved = int(rows[0]["spw"])
        spw_centre[solved] = rows[0]["f_ref_hz"]
        logger.info("fringefit: combined solutions stored in subband {} at reference frequency {:.6f} GHz",
                    solved, spw_centre[solved] / 1e9)
    write_fringe_table(caltable, vis, times=per_row["times"], field_ids=per_row["field_ids"],
                       spw_ids=per_row["spw_ids"], antenna_ids=per_row["antenna_ids"],
                       refant_id=per_row["refant_ids"], scan_numbers=per_row["scan_numbers"],
                       intervals=per_row["intervals"], fparam=stack["fparam"], paramerr=stack["paramerr"],
                       flag=stack["flag"], snr=stack["snr"], spw_chan_freq=spw_centre, spw_chan_width=spw_width)
    assert REFANT_SNR_SENTINEL == 999.0  # the pipeline's SNR reader masks this sentinel
