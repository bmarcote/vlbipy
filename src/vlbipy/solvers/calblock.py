"""Load one calibrated block of visibilities inside a worker: read, select, de-rotate, apply the priors.

Every solver job starts the same way — read some runs of the measurement set,
keep the wanted antennas and times, apply the parallactic-angle rotation and the
prior calibration tables on the fly — and differs only in what it then reduces
the block to (fringe solutions, baseline averages). That common start lives here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .apply import apply_tables
from .msio import read_runs


def weight_column(columns: list[str], sigma_weights: bool = False) -> str:
    """Name of the column the weights come from.

    ``sigma_weights`` selects SIGMA_SPECTRUM / SIGMA (weights are then 1 / sigma^2, see :func:`load_block`);
    otherwise WEIGHT_SPECTRUM when the MS has it, else WEIGHT.
    """
    if sigma_weights:
        return "SIGMA_SPECTRUM" if "SIGMA_SPECTRUM" in columns else "SIGMA"
    return "WEIGHT_SPECTRUM" if "WEIGHT_SPECTRUM" in columns else "WEIGHT"


def sigma_to_weight(sigma: np.ndarray) -> np.ndarray:
    """``1 / sigma^2`` as float32, 0 where sigma is not positive."""
    sigma = np.asarray(sigma, dtype=np.float32)
    return np.divide(1.0, sigma * sigma, out=np.zeros_like(sigma), where=sigma > 0)


def load_block(specs: list[dict], *, setup: dict, engine: str, columns: list[str], data_column: str = "DATA",
               chans: Optional[dict] = None, parallel_hands: bool = False, timerange: Optional[tuple] = None,
               antennas: Optional[set] = None, among: bool = False, parang: bool = False, priors: tuple = (),
               model: bool = False, sigma_weights: bool = True) -> Optional[dict]:
    """Read the runs ``specs`` (one field) and return the selected, calibrated block; None when nothing is left.

    Parameters
    ----------
    specs : list of dict
        Run descriptions from :func:`vlbipy.solvers.msio.run_specs`, all of the same field.
    setup : dict
        Output of :func:`vlbipy.solvers.msio.read_setup`.
    columns : list of str
        Main-table column names (decides WEIGHT_SPECTRUM vs WEIGHT, FLAG_ROW, MODEL_DATA).
    chans : dict, optional
        ``{spw: channel indices}`` to read; every spw in the block must select the same number of channels.
    parallel_hands : bool
        Read only the first and last correlation (RR, LL) — all a fringe or gain solve uses.
    timerange : (float, float), optional
        Keep rows with TIME inside it [MJD s].
    antennas, among : set of int, bool
        Antenna selection; ``among`` keeps only baselines between selected antennas, else any baseline with one.
    parang : bool
        Rotate by the feed angles (CASA ``parang=True``).
    priors : sequence
        Entries from :func:`vlbipy.solvers.apply.normalise_entries`, applied in order.
    model : bool
        Also return MODEL_DATA as ``"model"`` when the column exists.
    sigma_weights : bool
        Start the weights from ``1 / SIGMA^2`` (SIGMA_SPECTRUM when present) instead of the WEIGHT columns.
        This is what CASA does when it calibrates on the fly: WEIGHT holds whatever the last applycal or
        statwt left there, so starting from it would apply the weight calibration of the priors twice.

    Returns
    -------
    dict or None
        ``vis``, ``flag``, ``weight`` (nrow, nchan_sel, ncorr_sel), ``time``, ``antenna1``, ``antenna2``, ``spw``,
        ``scan`` (nrow,), ``field`` (int), ``chan_freq`` (nspw, nchan_sel) and optionally ``model``.
    """
    ncorr = setup["ncorr"]
    nchan = setup["chan_freq"].shape[1]
    corrs = np.array([0, ncorr - 1]) if parallel_hands and ncorr > 2 else None
    ddid_to_spw = setup["ddid_to_spw"]
    chans_by_ddid = None
    if chans is not None:
        chans_by_ddid = {ddid: chans[int(spw)] for ddid, spw in enumerate(ddid_to_spw) if int(spw) in chans}
    wcol = weight_column(columns, sigma_weights)
    names = [data_column, "FLAG", "TIME", "ANTENNA1", "ANTENNA2", wcol]
    if "FLAG_ROW" in columns:
        names.append("FLAG_ROW")
    if model and "MODEL_DATA" in columns:
        names.append("MODEL_DATA")
    raw = read_runs(specs, names, engine=engine, chans_by_ddid=chans_by_ddid, corrs=corrs, nchan=nchan, ncorr=ncorr,
                    timerange=timerange)
    if not raw:
        return None
    antenna1 = np.asarray(raw["ANTENNA1"], dtype=np.int64)
    antenna2 = np.asarray(raw["ANTENNA2"], dtype=np.int64)
    keep = antenna1 != antenna2
    if antennas is not None:
        in1, in2 = np.isin(antenna1, list(antennas)), np.isin(antenna2, list(antennas))
        keep &= (in1 & in2) if among else (in1 | in2)
    if not keep.any():
        return None
    everything = bool(keep.all())
    pick = (lambda values: values) if everything else (lambda values: values[keep])
    vis, flag = pick(raw[data_column]), pick(np.asarray(raw["FLAG"], dtype=bool))
    weight = pick(sigma_to_weight(raw[wcol]) if sigma_weights else raw[wcol])
    if "FLAG_ROW" in raw:
        row_flag = pick(np.asarray(raw["FLAG_ROW"], dtype=bool))
        if row_flag.any():
            flag = flag | row_flag[:, None, None]
    time = pick(np.asarray(raw["TIME"], dtype=np.float64))
    antenna1, antenna2 = pick(antenna1), pick(antenna2)
    spw = ddid_to_spw[pick(raw["DATA_DESC_ID"])]
    field = int(specs[0]["field"])
    selected = sorted(set(spw.tolist()))
    chan_freq = setup["chan_freq"]
    if chans is not None:
        sizes = {chans[s].size for s in selected}
        if len(sizes) != 1:
            raise ValueError("a block needs the same number of selected channels in every subband")
        sub = np.zeros((chan_freq.shape[0], sizes.pop()), dtype=np.float64)
        for s in selected:
            sub[s] = chan_freq[s, chans[s]]
        chan_freq = sub
    if parang:
        from .parang import apply_parang, feed_angles_for_ms
        times, time_index = np.unique(time, return_inverse=True)
        ra, dec = setup["field_dirs"][field]
        angles = feed_angles_for_ms(setup["antenna_xyz"], setup["mounts"], float(ra), float(dec), times)
        vis = apply_parang(vis, antenna1, antenna2, time, angles, time_index=time_index)
    if priors:
        vis, flag, weight = apply_tables(vis, flag, weight, antenna1, antenna2, time, spw, chan_freq, list(priors),
                                         antenna_xyz=setup["antenna_xyz"], field_dir=setup["field_dirs"][field],
                                         copy=False)
    elif weight.ndim == 2:
        weight = np.broadcast_to(weight[:, None, :], vis.shape)
    block = {"vis": vis, "flag": flag, "weight": weight, "time": time, "antenna1": antenna1, "antenna2": antenna2,
             "spw": spw, "scan": pick(raw["SCAN_NUMBER"]), "field": field, "chan_freq": chan_freq}
    if "MODEL_DATA" in raw:
        block["model"] = pick(raw["MODEL_DATA"])
    return block
