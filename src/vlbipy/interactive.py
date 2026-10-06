"""Interactive flagging widgets for the generated notebook (plotly + ipywidgets).

Two flaggers, both with the same shape - a scatter figure with box/lasso
selection, a "Flag selected" button that writes the flags, and an "Undo" -
plus the bridge that carries flags edited in a difmapy window back to the
multi-source measurement set:

* :func:`caltable_flagger` plots a CASA calibration table (Tsys, gain curve,
  fringe delays/rates/phases, bandpass or gain amplitudes) per antenna and
  flags solutions by setting the table's ``FLAG`` column. ``applycal``
  (``calflagstrict``) then flags the data those solutions covered.
* :func:`visibility_flagger` plots channel-averaged visibilities of one field
  on baselines to the reference antenna (amplitude / phase vs time) straight
  from the measurement set and flags them with ``flagdata``. It is the
  fallback when difmapy's own windows cannot be shown (no display).
* :func:`flag_commands_from_split` turns the flags a difmapy session saved
  into a per-source split (``obs.save_flags()``) into ``flagdata`` commands
  for the parent measurement set, so the main pipeline can be resumed on the
  edited data.

Everything here is optional: the module imports plotly/ipywidgets lazily and
the rest of vlbipy never depends on it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np

from .logging_utils import get_logger

logger = get_logger()

_MJD_EPOCH = dt.datetime(1858, 11, 17)


def _widgets():
    import ipywidgets as widgets
    import plotly.graph_objects as go
    return widgets, go


def mjd_seconds_to_datetime(seconds: np.ndarray) -> list:
    """MJD-in-seconds array -> list of datetimes (what plotly wants for a time axis)."""
    return [_MJD_EPOCH + dt.timedelta(seconds=float(s)) for s in np.asarray(seconds, dtype=float)]


def _casa_timerange(start: float, end: float) -> str:
    """CASA ``timerange`` string covering [start, end] MJD seconds (padded by half a second)."""
    fmt = "%Y/%m/%d/%H:%M:%S.%f"
    lo = (_MJD_EPOCH + dt.timedelta(seconds=start - 0.5)).strftime(fmt)[:-4]
    hi = (_MJD_EPOCH + dt.timedelta(seconds=end + 0.5)).strftime(fmt)[:-4]
    return f"{lo}~{hi}"


# --------------------------------------------------------------------------------------------
# calibration tables
# --------------------------------------------------------------------------------------------

#: Parameter layout of a fringe table: (phase, delay, rate, dispersive) per polarization.
FRINGE_PARAMS = ("phase (deg)", "delay (ns)", "rate (ps/s)", "dispersive")


def read_caltable(path: str) -> dict:
    """Read a CASA calibration table into arrays keyed by column, plus antenna names and a kind.

    Returns ``time`` (MJD s), ``antenna`` (ids), ``spw``, ``flag`` (npar, nchan, nrow),
    ``values`` (npar, nchan, nrow) real-valued, ``kind`` (``"fringe"``, ``"gain"``,
    ``"bandpass"``, ``"tsys"``, ``"gc"``), ``antenna_names`` and ``labels`` per parameter.
    """
    from casatools import table
    tb = table()
    tb.open(str(path))
    try:
        cols = tb.colnames()
        out = {"time": np.asarray(tb.getcol("TIME"), dtype=float), "antenna": np.asarray(tb.getcol("ANTENNA1")),
               "spw": np.asarray(tb.getcol("SPECTRAL_WINDOW_ID")), "flag": np.asarray(tb.getcol("FLAG"), dtype=bool)}
        if "FPARAM" in cols:
            values = np.asarray(tb.getcol("FPARAM"), dtype=float)
            info = tb.info()
            sub = str(info.get("subType", ""))
            kind = "fringe" if "Fringe" in sub else ("tsys" if "Tsys" in sub else ("gc" if "GainCurve" in sub or
                                                                                       "EPower" in sub else "real"))
        else:
            cparam = np.asarray(tb.getcol("CPARAM"))
            values = np.abs(cparam) if cparam.shape[1] > 1 else np.abs(cparam)
            out["phase"] = np.degrees(np.angle(cparam))
            kind = "bandpass" if cparam.shape[1] > 1 else "gain"
    finally:
        tb.close()
    tb.open(str(Path(path) / "ANTENNA"))
    try:
        out["antenna_names"] = [str(n) for n in tb.getcol("NAME")]
    finally:
        tb.close()
    out["values"] = values
    out["kind"] = kind
    npar = values.shape[0]
    if kind == "fringe":
        npol = max(1, npar // 4)
        out["labels"] = [f"{FRINGE_PARAMS[p % 4]} pol{p // 4}" for p in range(npar)]
        # phases in degrees, delays in ns, rates in ps/s for readability
        scale = np.array([180.0 / np.pi, 1e9, 1e12, 1.0] * npol)[:npar]
        out["values"] = values * scale[:, None, None]
    elif kind in ("gain", "bandpass"):
        out["labels"] = [f"amplitude pol{p}" for p in range(npar)]
    else:
        out["labels"] = [f"{kind} pol{p}" for p in range(npar)]
    return out


def flag_caltable_rows(path: str, rows, params=None) -> int:
    """Set ``FLAG`` for the given table rows (all parameters/channels, or only ``params``); returns count."""
    from casatools import table
    rows = sorted({int(r) for r in rows})
    if not rows:
        return 0
    tb = table()
    tb.open(str(path), nomodify=False)
    try:
        flag = np.asarray(tb.getcol("FLAG"), dtype=bool)
        for r in rows:
            if params is None:
                flag[:, :, r] = True
            else:
                for p in params:
                    flag[p, :, r] = True
        tb.putcol("FLAG", flag)
        tb.flush()
    finally:
        tb.close()
    logger.info("flagged {} row(s) in {}", len(rows), Path(path).name)
    return len(rows)


def unflag_caltable_rows(path: str, rows, snapshot: np.ndarray) -> None:
    """Restore the ``FLAG`` column of ``rows`` from ``snapshot`` (an earlier full FLAG array)."""
    from casatools import table
    tb = table()
    tb.open(str(path), nomodify=False)
    try:
        flag = np.asarray(tb.getcol("FLAG"), dtype=bool)
        for r in rows:
            flag[:, :, r] = snapshot[:, :, r]
        tb.putcol("FLAG", flag)
        tb.flush()
    finally:
        tb.close()


def caltable_flagger(path: str, *, param: int = 0, title: str = ""):
    """Interactive plot of one parameter of a calibration table vs time, one trace per antenna.

    Box/lasso-select points and press **Flag selected** to flag those solutions
    (``FLAG`` column); **Undo** reverts the last flagging. Flagged points are
    drawn hollow. Returns the ipywidgets box to display.
    """
    widgets, go = _widgets()
    data = read_caltable(path)
    values, flag = data["values"], data["flag"]
    npar = values.shape[0]
    channel = values.shape[1] // 2
    times = mjd_seconds_to_datetime(data["time"])
    fig = go.FigureWidget()
    row_index: list[np.ndarray] = []
    for ant_id in np.unique(data["antenna"]):
        rows = np.where(data["antenna"] == ant_id)[0]
        row_index.append(rows)
        name = data["antenna_names"][int(ant_id)] if int(ant_id) < len(data["antenna_names"]) else str(ant_id)
        flagged = flag[param, channel, rows]
        fig.add_scatter(x=[times[r] for r in rows], y=values[param, channel, rows], mode="markers", name=name,
                        customdata=rows, marker={"size": 6, "symbol": np.where(flagged, "circle-open", "circle")},
                        hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.4g}}<extra>spw %{{text}}</extra>",
                        text=[str(s) for s in data["spw"][rows]])
    fig.update_layout(title=title or f"{Path(path).name}: {data['labels'][param]}", height=420,
                      dragmode="select", xaxis_title="time (UTC)", yaxis_title=data["labels"][param],
                      legend={"orientation": "h"}, margin={"l": 60, "r": 20, "t": 50, "b": 40})
    selected: set[int] = set()
    history: list[tuple[list[int], np.ndarray]] = []

    def on_select(trace, points, selector):
        for i in points.point_inds:
            selected.add(int(trace.customdata[i]))

    for trace in fig.data:
        trace.on_selection(on_select)
    button = widgets.Button(description="Flag selected", button_style="danger")
    undo = widgets.Button(description="Undo")
    status = widgets.HTML(value="select points with the box/lasso tool, then flag")
    param_box = widgets.Dropdown(options=[(lab, i) for i, lab in enumerate(data["labels"])], value=param,
                                 description="parameter")

    def redraw():
        fresh = read_caltable(path)
        for trace, rows in zip(fig.data, row_index):
            flagged = fresh["flag"][param_box.value, channel, rows]
            with fig.batch_update():
                trace.y = fresh["values"][param_box.value, channel, rows]
                trace.marker.symbol = np.where(flagged, "circle-open", "circle")
        fig.layout.yaxis.title = fresh["labels"][param_box.value]

    def do_flag(_):
        if not selected:
            status.value = "nothing selected"
            return
        history.append((sorted(selected), read_caltable(path)["flag"].copy()))
        n = flag_caltable_rows(path, selected)
        status.value = f"flagged {n} solution(s); re-run <code>obs.calibrate.apply(force=True)</code> to propagate"
        selected.clear()
        redraw()

    def do_undo(_):
        if not history:
            status.value = "nothing to undo"
            return
        rows, snapshot = history.pop()
        unflag_caltable_rows(path, rows, snapshot)
        status.value = f"restored {len(rows)} solution(s)"
        redraw()

    button.on_click(do_flag)
    undo.on_click(do_undo)
    param_box.observe(lambda change: redraw() if change["name"] == "value" else None)
    return widgets.VBox([widgets.HBox([param_box, button, undo]), fig, status])


# --------------------------------------------------------------------------------------------
# visibilities (plotly fallback)
# --------------------------------------------------------------------------------------------

def read_visibilities(ms_path: str, *, field: str, refant: str, column: str = "corrected",
                      channel_fraction: float = 0.8) -> dict:
    """Channel-averaged parallel-hand visibilities on baselines to ``refant``, per integration.

    Returns arrays ``time``, ``baseline`` (names ``REF-ANT``), ``spw``, ``amp``, ``phase`` (deg), ``flag``.
    """
    from casatools import ms as ms_tool_cls, table
    tb = table()
    tb.open(str(Path(ms_path) / "ANTENNA"))
    try:
        names = [str(n) for n in tb.getcol("NAME")]
    finally:
        tb.close()
    ref_id = names.index(refant)
    ms_tool = ms_tool_cls()
    ms_tool.open(str(ms_path))
    out = {k: [] for k in ("time", "baseline", "spw", "amp", "phase", "flag")}
    try:
        tb.open(str(Path(ms_path) / "DATA_DESCRIPTION"))
        n_dd = tb.nrows()
        tb.close()
        col = "corrected_data" if column == "corrected" else "data"
        for dd in range(n_dd):
            ms_tool.selectinit(datadescid=dd)
            try:
                ms_tool.msselect({"field": field, "baseline": f"{refant}&*"})
            except RuntimeError:
                ms_tool.reset()
                continue
            rec = ms_tool.getdata([col, "flag", "antenna1", "antenna2", "time"])
            ms_tool.reset()
            vis = rec.get(col)
            if vis is None or not vis.size:
                continue
            flag = np.asarray(rec["flag"], dtype=bool)
            nchan = vis.shape[1]
            margin = int(round(nchan * (1 - channel_fraction) / 2))
            sel = slice(margin, max(margin + 1, nchan - margin))
            pols = [0, vis.shape[0] - 1] if vis.shape[0] > 1 else [0]
            with np.errstate(invalid="ignore"):
                masked = np.where(flag, np.nan, vis)[pols][:, sel, :]
                mean = np.nanmean(masked.reshape(-1, masked.shape[-1]), axis=0)
            other = np.where(rec["antenna1"] == ref_id, rec["antenna2"], rec["antenna1"])
            out["time"].append(np.asarray(rec["time"], dtype=float))
            out["baseline"].append(np.array([f"{refant}-{names[int(a)]}" for a in other]))
            out["spw"].append(np.full(mean.size, dd))
            out["amp"].append(np.abs(mean))
            out["phase"].append(np.degrees(np.angle(mean)))
            out["flag"].append(~np.isfinite(mean))
    finally:
        ms_tool.close()
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def visibility_flagger(ms_path: str, *, field: str, refant: str, column: str = "corrected",
                       quantity: str = "amp", flag_backup: bool = True):
    """Amplitude or phase vs time per baseline to ``refant`` with box-select flagging (``flagdata``).

    Selected points become ``flagdata(mode='list')`` commands (baseline, spw,
    timerange) on ``ms_path``. Returns the widget box.
    """
    widgets, go = _widgets()
    data = read_visibilities(ms_path, field=field, refant=refant, column=column)
    times = mjd_seconds_to_datetime(data["time"])
    fig = go.FigureWidget()
    for baseline in sorted(set(data["baseline"].tolist())):
        idx = np.where(data["baseline"] == baseline)[0]
        fig.add_scatter(x=[times[i] for i in idx], y=data[quantity][idx], mode="markers", name=baseline,
                        customdata=idx, marker={"size": 4}, text=[f"spw {s}" for s in data["spw"][idx]])
    fig.update_layout(title=f"{field}: {quantity} vs time ({column} data)", dragmode="select", height=450,
                      xaxis_title="time (UTC)", yaxis_title="amplitude" if quantity == "amp" else "phase (deg)")
    selected: set[int] = set()
    for trace in fig.data:
        trace.on_selection(lambda tr, pts, sel: [selected.add(int(tr.customdata[i])) for i in pts.point_inds])
    button = widgets.Button(description="Flag selected", button_style="danger")
    status = widgets.HTML(value="select points, then flag; each selection becomes flagdata commands")

    def do_flag(_):
        if not selected:
            status.value = "nothing selected"
            return
        commands = flag_commands_for_points(data, sorted(selected), field=field)
        run_flag_commands(ms_path, commands, flag_backup=flag_backup)
        status.value = f"flagged {len(selected)} point(s) with {len(commands)} flagdata command(s)"
        selected.clear()

    button.on_click(do_flag)
    return widgets.VBox([button, fig, status])


def flag_commands_for_points(data: dict, indices, *, field: str = "") -> list[str]:
    """Group selected points into ``flagdata`` list commands: one per (baseline, spw, contiguous time run)."""
    commands = []
    idx = np.asarray(sorted(indices), dtype=int)
    if idx.size == 0:
        return commands
    keys = {}
    for i in idx:
        keys.setdefault((str(data["baseline"][i]), int(data["spw"][i])), []).append(float(data["time"][i]))
    for (baseline, spw), times in keys.items():
        times = sorted(times)
        gap = np.median(np.diff(times)) * 1.5 if len(times) > 1 else 1.0
        start = prev = times[0]
        for t in times[1:] + [None]:
            if t is None or t - prev > max(gap, 1.0):
                sel = f"antenna='{baseline.replace('-', '&')}' spw='{spw}' timerange='{_casa_timerange(start, prev)}'"
                commands.append(sel + (f" field='{field}'" if field else ""))
                if t is not None:
                    start = t
            prev = t if t is not None else prev
    return commands


def run_flag_commands(ms_path: str, commands: list[str], *, flag_backup: bool = True) -> None:
    """Apply ``flagdata(mode='list')`` commands to ``ms_path``."""
    if not commands:
        return
    from casatasks import flagdata
    flagdata(vis=str(ms_path), mode="list", inpfile=list(commands), flagbackup=bool(flag_backup))
    logger.info("flagdata: {} command(s) applied to {}", len(commands), Path(ms_path).name)


# --------------------------------------------------------------------------------------------
# difmapy -> parent measurement set
# --------------------------------------------------------------------------------------------

def snapshot_flags(ms_path: str) -> str:
    """Save the FLAG/FLAG_ROW columns of a split MS next to it (``<ms>.flags0.npz``) as the reference state."""
    from casatools import table
    tb = table()
    tb.open(str(ms_path))
    try:
        flag = np.asarray(tb.getcol("FLAG"), dtype=bool)
        flag_row = np.asarray(tb.getcol("FLAG_ROW"), dtype=bool)
    finally:
        tb.close()
    target = f"{ms_path}.flags0.npz"
    np.savez_compressed(target, flag=flag, flag_row=flag_row)
    return target


def flag_commands_from_split(split_ms: str, *, field: str) -> list[str]:
    """``flagdata`` commands for the parent MS reproducing the flags *added* to ``split_ms``.

    Compares the split's FLAG column with the snapshot :func:`snapshot_flags`
    took when it was written; rows (or channels of a row) flagged since then
    are turned into (baseline, spw, timerange) commands. A missing snapshot
    means every flagged row is exported.
    """
    from casatools import table
    tb = table()
    tb.open(str(split_ms))
    try:
        flag = np.asarray(tb.getcol("FLAG"), dtype=bool)
        time = np.asarray(tb.getcol("TIME"), dtype=float)
        a1, a2 = np.asarray(tb.getcol("ANTENNA1")), np.asarray(tb.getcol("ANTENNA2"))
        dd = np.asarray(tb.getcol("DATA_DESC_ID"))
    finally:
        tb.close()
    tb.open(str(Path(split_ms) / "ANTENNA"))
    try:
        names = [str(n) for n in tb.getcol("NAME")]
    finally:
        tb.close()
    snap = Path(f"{split_ms}.flags0.npz")
    before = np.load(snap)["flag"] if snap.is_file() else np.zeros_like(flag)
    new = flag & ~before                                   # (npol, nchan, nrow)
    rows = np.where(new.all(axis=(0, 1)))[0]
    if rows.size == 0:
        return []
    data = {"baseline": np.array([f"{names[int(a1[r])]}-{names[int(a2[r])]}" for r in rows]),
            "spw": dd[rows], "time": time[rows]}
    commands = flag_commands_for_points(data, range(rows.size), field=field)
    logger.info("{}: {} newly flagged row(s) -> {} flagdata command(s)", Path(split_ms).name, rows.size,
                len(commands))
    return commands
