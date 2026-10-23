"""The Earth-orientation correction table (the dask-ms backend's ``gencal eop``), without CASA.

:func:`eop` writes what ``casatasks.gencal(caltype='eop', infile=<file>)`` writes: CASA's
``task_gencal.EOPGencal`` creates an empty ``Fringe Jones`` table (``calibrater.createcaltable(partype='Real',
caltype='Fringe Jones', singlechan=True)``) and ``casatasks/private/eop.py`` (``do_generate_eop``) fills it in
Python with erfa. This module is a port of that code, with the same erfa calls in the same order, so every cell
is bit-identical.

What CASA does, and this module repeats:

* per observation (OBSERVATION row) the "old" EOPs are the MS ``EARTH_ORIENTATION`` rows of that OBSERVATION_ID
  (table order; MJD = TIME / 86400, PM in arcsec, UT1_UTC); an observation without them, or without new EOPs in
  ``infile`` for ``[min MJD, max MJD]``, is skipped. ``infile`` is parsed as USNO finals 2.0 / 2.1 (first line
  ``EOP-MOD Ver 2.x``, TAI Julian dates converted to UTC with erfa) or else as IERS EOP C04 text;
* then per scan of the observation (sorted), per antenna of the scan (ANTENNA1 and ANTENNA2 of its rows, sorted),
  per field of the scan (sorted), over the distinct TIMEs of the scan (all its fields): a sample at the first
  time more than 60 s after the last sample (starting from t0 = 0) and one at the last time if it was not
  sampled. Each sample gives one row per subband of the scan (sorted), in that order (no re-sorting);
* delay = geometric delay with the new EOPs minus with the old ones (``erfa.c2t00a`` celestial-to-terrestrial
  matrix with linearly interpolated polar motion and UT1-UTC, ``pdp(antenna xyz, rotated phase centre) / c``),
  rate = the same difference of the centred 60 s finite differences. Polar motion and UT1-UTC are interpolated
  linearly between the two EOP entries around each time. **Here this module differs from CASA on purpose**:
  eop.py clips the interpolation index to ``len(dict) - 1 = 3`` (the number of keys of its EOP dict, not of
  entries), so times after the fourth EOP entry of an observation are *extrapolated* from entries 2 and 3 — on
  RSM07 that makes the delay corrections about four times too large. ``casa_compatible=True`` repeats it, for
  comparisons with CASA. ``FPARAM`` (1, 8) float32 per pol (two):
  phase ``2 pi delay f_spw``, delay [ns], rate, 0, with ``f_spw`` the single channel frequency of the caltable
  SPECTRAL_WINDOW (the mean channel frequency); ``PARAMERR = -1``, ``FLAG = False``, ``SNR = WEIGHT = 1``,
  ``INTERVAL = 0``, ``ANTENNA2 = SCAN_NUMBER = -1``;
* antenna positions and phase centres go through msmd's spherical form first (``atan2(y, x)``, ``asin(z / r)``,
  ``r``; the phase centre via its unit vector) and back with ``erfa.s2p`` / ``erfa.s2c``, as CASA does;
* subtables OBSERVATION, ANTENNA, FIELD and an empty HISTORY as :mod:`vlbipy.solvers.gaincurve` writes
  them, and SPECTRAL_WINDOW as ``NewCalTable::makeSpwSingleChan``: a subband with more than one channel gets
  the frequency of channel ``nchan // 2`` and ``TOTAL_BANDWIDTH`` as its width, effective bandwidth and
  resolution (one channel); the other columns, FLAG_ROW included, are the MS values. A table with no row is
  removed (CASA rmtree's it).

The intended differences: the correct interpolation above (unless ``casa_compatible``) and the ``CASA_Version``
keyword, which reads ``vlbipy``. ``infile=None`` (CASA then reads its own ``geodetic/IERSeop2000`` table) raises
NotImplementedError.
"""
from __future__ import annotations

import math
import re
import shutil
import time as _time
from pathlib import Path

import erfa
import numpy as np

from ..casacore_format.table import Table
from ..casacore_format.tabledat import UnsupportedTable
from ..logging_utils import get_logger
from .caltable import _create, _link_subtables, history_table_desc, main_table_desc
from .gaincurve import HISTORY_ORDER, SUBTABLE_DESCS, _cells, write_calibration_subtable
from .msio import ms_parts, open_table, table_engine

logger = get_logger()

VISCAL = "Fringe Jones"
CASA_VERSION = "vlbipy"
#: Sampling interval and finite-difference step of CASA's eop.py [s].
DELTA = 60
SPEED_OF_LIGHT = 299792458.0
ARCSEC = 0.00000484813681109535993589914098765
#: gencal keywords that EOPGencal never reads (any value gives the same table).
IGNORED_KEYWORDS = ("uniform", "endpoint", "timeout", "retry", "retry_wait_time", "ant_pos_time_limit", "spw",
                    "antenna", "pol", "parameter")
INDEX_COLUMNS = ("OBSERVATION_ID", "SCAN_NUMBER", "FIELD_ID", "TIME", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID")


def eop(vis, caltable, *, infile=None, engine=None, casa_compatible: bool = False,
                    **gencal_kwargs) -> dict:
    """Write the table ``gencal(vis=vis, caltable=caltable, caltype='eop', infile=infile)`` writes.

    Parameters
    ----------
    vis : str or pathlib.Path
        The measurement set (or Multi-MS) with an EARTH_ORIENTATION subtable. Only read.
    caltable : str or pathlib.Path
        The table to write (an existing one is replaced).
    infile : str or pathlib.Path
        The updated EOP file (USNO finals 2.0/2.1 or IERS C04).
    engine : str, optional
        Table engine for the main-table index columns (default :func:`vlbipy.solvers.msio.table_engine`).
    casa_compatible : bool
        Repeat CASA's interpolation-index bug (see the module docstring), for cell-by-cell comparisons only.
    **gencal_kwargs
        ``caltype`` must be ``'eop'``; the keywords of :data:`IGNORED_KEYWORDS` are accepted with any value.

    Returns
    -------
    dict
        ``rows``, ``samples`` (antenna/field/time samples), ``observations`` (with rows), ``skipped``
        (observations without old or new EOPs), ``seconds``, ``removed`` (True when no row was written and the
        table was deleted, as CASA does).

    Raises
    ------
    NotImplementedError
        Another caltype, an unknown keyword, ``infile`` None/'None' (CASA's casadata table), or an MS this
        package cannot read.
    """
    started = _time.perf_counter()
    vis, caltable = Path(vis), Path(caltable)
    _check_keywords(gencal_kwargs, infile)
    engine = engine or table_engine()
    try:
        positions = _antenna_positions(vis)
        phase_dirs = _phase_dirs(vis)
        nobs = _nrows(vis / "OBSERVATION")
        spw_values = _spw_values(vis)
        index = _read_index(vis, engine)
    except (UnsupportedTable, OSError) as exc:
        raise NotImplementedError(f"gencal eop: {vis} could not be read without casacore ({exc})") from exc
    reffreq = spw_values["CHAN_FREQ"][:, 0]
    pieces, skipped, nsample = [], 0, 0
    for obsid in range(nobs):
        old_eops = eops_from_ms(vis, obsid)
        if old_eops is None:
            logger.error("gencal[native] eop: no EOPs found in {} for observation {}", vis, obsid)
            skipped += 1
            continue
        mjd_min, mjd_max = np.min(old_eops["MJD"]), np.max(old_eops["MJD"])
        new_eops = eops_from_file(infile, mjd_min, mjd_max)
        if new_eops is None:
            logger.error("gencal[native] eop: no updated EOPs for MJD {}-{} in {}", mjd_min, mjd_max, infile)
            skipped += 1
            continue
        samples = observation_samples(index, obsid)
        nsample += len(samples["time"])
        pieces.append(_rows_for_samples(samples, obsid, positions, phase_dirs, reffreq, old_eops, new_eops,
                                        casa_compatible))
    rows = _concatenate(pieces)
    nrow = int(rows["TIME"].size)
    if caltable.exists():
        shutil.rmtree(caltable)
    removed = nrow == 0
    if not removed:
        _write_table(caltable, vis, rows, spw_values)
    report = {"rows": nrow, "samples": nsample, "observations": len(pieces), "skipped": skipped,
              "removed": removed, "seconds": _time.perf_counter() - started}
    logger.info("gencal[native] eop: {} -> {} ({} row(s), {} sample(s), {} observation(s), {} skipped{}) in {:.2f}s",
                vis.name, caltable.name, nrow, nsample, len(pieces), skipped, ", table removed" if removed else "",
                report["seconds"])
    return report


def _check_keywords(gencal_kwargs: dict, infile) -> None:
    """Raise NotImplementedError unless the gencal keywords ask for exactly the table this module writes."""
    kwargs = dict(gencal_kwargs)
    caltype = kwargs.pop("caltype", "eop")
    if str(caltype).lower() != "eop":
        raise NotImplementedError(f"gencal caltype={caltype!r} (only 'eop' is written here)")
    for name in IGNORED_KEYWORDS:
        kwargs.pop(name, None)
    if kwargs:
        raise NotImplementedError(f"gencal eop with keyword(s) {sorted(kwargs)}")
    if not infile or str(infile) == "None":
        raise NotImplementedError("gencal eop without infile reads CASA's geodetic/IERSeop2000 table")
    if not Path(infile).is_file():
        raise NotImplementedError(f"gencal eop: infile {infile} does not exist")


def _nrows(path: Path) -> int:
    """Number of rows of a table."""
    with Table(path) as table:
        return table.nrows()


def _antenna_positions(vis: Path) -> np.ndarray:
    """Antenna ITRF positions (nant, 3) [m] round-tripped through msmd's spherical form, as eop.py gets them."""
    with Table(vis / "ANTENNA") as table:
        xyz = np.asarray(table.getcol("POSITION"), dtype=np.float64).reshape(-1, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    radius = np.sqrt(x * x + y * y + z * z)
    return np.array([erfa.s2p(lon, lat, r) for lon, lat, r in zip(np.arctan2(y, x), np.arcsin(z / radius), radius)])


def _phase_dirs(vis: Path) -> np.ndarray:
    """Phase-centre unit vectors (nfield, 3) as eop.py gets them (msmd.phasecenter, then erfa.s2c)."""
    with Table(vis / "FIELD") as table:
        nfield = table.nrows()
        cells = table.getcol("PHASE_DIR") if nfield else []
    out = np.zeros((nfield, 3))
    for field, cell in enumerate(cells):
        lon, lat = np.asarray(cell, dtype=np.float64).reshape(-1, 2)[0]
        x, y, z = math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)
        radius = math.sqrt(x * x + y * y + z * z)
        out[field] = erfa.s2c(math.atan2(y, x), math.asin(z / radius))
    return out


def _spw_values(vis: Path) -> dict:
    """The single-channel SPECTRAL_WINDOW values of the caltable."""
    desc = SUBTABLE_DESCS["SPECTRAL_WINDOW"]
    with Table(vis / "SPECTRAL_WINDOW") as table:
        nspw = table.nrows()
        values = {name: table.getcol(name) for name in desc}
    return single_channel_spw(values, nspw)


def single_channel_spw(values: dict, nrow: int) -> dict:
    """SPECTRAL_WINDOW values reduced as ``NewCalTable::makeSpwSingleChan`` does (``createcaltable singlechan``).

    A subband with ``nchan > 1`` channels keeps the frequency of channel ``nchan // 2`` and gets its
    ``TOTAL_BANDWIDTH`` as channel width, effective bandwidth and resolution; a single-channel one is unchanged.
    """
    out = dict(values)
    freq, width, bw, res = (_cells(values[name], nrow) for name in ("CHAN_FREQ", "CHAN_WIDTH", "EFFECTIVE_BW",
                                                                    "RESOLUTION"))
    total = np.asarray(values["TOTAL_BANDWIDTH"], dtype=np.float64)
    out["CHAN_FREQ"] = np.array([[cell[cell.size // 2]] for cell in freq]).reshape(nrow, 1)
    out["CHAN_WIDTH"] = np.array([[total[k] if freq[k].size > 1 else width[k][0]] for k in range(nrow)]).reshape(nrow, 1)
    out["EFFECTIVE_BW"] = np.array([[total[k] if freq[k].size > 1 else bw[k][0]] for k in range(nrow)]).reshape(nrow, 1)
    out["RESOLUTION"] = np.array([[total[k] if freq[k].size > 1 else res[k][0]] for k in range(nrow)]).reshape(nrow, 1)
    out["NUM_CHAN"] = np.ones(nrow, dtype=np.int32)
    return out


def _read_index(vis: Path, engine: str) -> dict:
    """The main-table index columns of every part, concatenated; DATA_DESC_ID mapped to ``SPW``."""
    with Table(vis / "DATA_DESCRIPTION") as table:
        ddid_spw = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64)
    columns = {name: [] for name in INDEX_COLUMNS}
    for part in ms_parts(vis):
        handle = open_table(part, engine=engine)
        try:
            if handle.nrows() == 0:
                continue
            for name in INDEX_COLUMNS:
                columns[name].append(np.asarray(handle.getcol(name)))
        finally:
            handle.close()
    index = {name: (np.concatenate(v) if v else np.zeros(0)) for name, v in columns.items()}
    for name in INDEX_COLUMNS:
        if name != "TIME":
            index[name] = index[name].astype(np.int64)
    index["SPW"] = ddid_spw[index["DATA_DESC_ID"]] if index["DATA_DESC_ID"].size else index["DATA_DESC_ID"]
    return index


def observation_samples(index: dict, obsid: int) -> dict:
    """The (antenna, field, time, subbands) samples of one observation, in eop.py's loop order.

    Parameters
    ----------
    index : dict
        Main-table columns ``OBSERVATION_ID, SCAN_NUMBER, FIELD_ID, TIME, ANTENNA1, ANTENNA2, SPW`` (arrays).
    obsid : int
        The observation.

    Returns
    -------
    dict
        ``antenna``, ``field``, ``time`` (one entry per sample) and ``spws`` (list of sorted subband arrays).
    """
    rows = np.flatnonzero(index["OBSERVATION_ID"] == obsid)
    scans = index["SCAN_NUMBER"][rows]
    out = {"antenna": [], "field": [], "time": [], "spws": []}
    for scan in np.unique(scans):
        sel = rows[scans == scan]
        antennas = np.unique(np.concatenate([index["ANTENNA1"][sel], index["ANTENNA2"][sel]]))
        fields = np.unique(index["FIELD_ID"][sel])
        spws = np.unique(index["SPW"][sel])
        times = sample_times(np.unique(index["TIME"][sel]))
        for antenna in antennas:
            for field in fields:
                for t in times:
                    out["antenna"].append(int(antenna))
                    out["field"].append(int(field))
                    out["time"].append(float(t))
                    out["spws"].append(spws)
    return out


def sample_times(times) -> list:
    """The times eop.py samples from a scan's sorted distinct times (> 60 s apart, plus the last one)."""
    out = []
    t0 = 0
    t = 0
    for t in times:
        if t - t0 > DELTA:
            t0 = t
            out.append(t)
    if t != t0:
        out.append(t)
    return out


def _rows_for_samples(samples, obsid, positions, phase_dirs, reffreq, old_eops, new_eops,
                      casa_compatible: bool = False) -> dict:
    """Main-table columns of the rows of one observation's samples (eop.py ``add_row``)."""
    times = np.asarray(samples["time"], dtype=np.float64)
    antennas = np.asarray(samples["antenna"], dtype=np.int64)
    fields = np.asarray(samples["field"], dtype=np.int64)
    pos, pc = positions[antennas], phase_dirs[fields]
    tm, dtm = times / 86400, DELTA / 86400
    old = [calc_delay(old_eops, pos, pc, t, casa_compatible) for t in (tm - dtm, tm, tm + dtm)]
    new = [calc_delay(new_eops, pos, pc, t, casa_compatible) for t in (tm - dtm, tm, tm + dtm)]
    delay = new[1] - old[1]
    rate = (new[2] - new[0]) / (2 * DELTA) - (old[2] - old[0]) / (2 * DELTA)
    counts = np.array([len(s) for s in samples["spws"]], dtype=np.int64)
    spws = np.concatenate(samples["spws"]).astype(np.int64) if counts.size else np.zeros(0, dtype=np.int64)
    expand = np.repeat(np.arange(times.size), counts)
    nrow = int(spws.size)
    param = np.zeros((nrow, 1, 8), dtype=np.float32)
    phase = 2 * math.pi * delay[expand] * reffreq[spws]
    for pol in range(2):
        param[:, 0, 4 * pol] = phase
        param[:, 0, 4 * pol + 1] = delay[expand] * 1e9
        param[:, 0, 4 * pol + 2] = rate[expand]
    return {"TIME": times[expand], "ANTENNA1": antennas[expand].astype(np.int32), "FIELD_ID": fields[expand].astype(
        np.int32), "OBSERVATION_ID": np.full(nrow, obsid, dtype=np.int32), "SPECTRAL_WINDOW_ID": spws.astype(np.int32),
        "FPARAM": param}


def _concatenate(pieces: list) -> dict:
    """Concatenate the per-observation row columns (empty columns when there is none)."""
    names = ("TIME", "ANTENNA1", "FIELD_ID", "OBSERVATION_ID", "SPECTRAL_WINDOW_ID", "FPARAM")
    if not pieces:
        return {name: np.zeros(0) for name in names}
    return {name: np.concatenate([p[name] for p in pieces]) for name in names}


def calc_delay(eops: dict, pos, p, mjd, casa_compatible: bool = False):
    """Geometric delay [s] of antenna ``pos`` towards unit vector ``p`` at UTC MJD ``mjd`` (eop.py ``calc_delay``)."""
    xp, yp = pm_xy(eops, 2400000.5, mjd, casa_compatible)
    dut1 = ut1_utc(eops, 2400000.5, mjd, casa_compatible) / 86400
    tai = erfa.utctai(2400000.5, mjd)
    tt = erfa.taitt(tai[0], tai[1])
    r = erfa.c2t00a(tt[0], tt[1], 2400000.5 + mjd, dut1, xp, yp)
    p = erfa.rxp(r, p)
    return erfa.pdp(pos, p) / SPEED_OF_LIGHT


def _bracket(eops: dict, jd1, jd2, casa_compatible: bool = False):
    """Indices (i0, i1) of the EOP entries bracketing MJD ``jd1 - 2400000.5 + jd2`` (eop.py's searchsorted).

    The index is clipped to the last entry. eop.py clips with ``len(eops) - 1`` where ``eops`` is the dict of four
    lists, so ``i1 <= 3`` whatever the number of entries and later times extrapolate from entries 2 and 3;
    ``casa_compatible=True`` repeats that.
    """
    i = np.searchsorted(eops["MJD"], (jd1 - 2400000.5) + jd2, side="right")
    i1 = np.clip(i, 1, (len(eops) if casa_compatible else len(eops["MJD"])) - 1)
    if len(eops["MJD"]) < 2 or np.max(i1) >= len(eops["MJD"]):
        raise NotImplementedError(f"gencal eop: {len(eops['MJD'])} EOP entries only (CASA fails with an IndexError)")
    return i1 - 1, i1


def ut1_utc(eops: dict, jd1, jd2, casa_compatible: bool = False):
    """UT1 - UTC [s] linearly interpolated, leap-second jumps removed (eop.py ``ut1_utc``)."""
    mjd1, mjd2 = jd1 - 2400000.5, jd2
    i0, i1 = _bracket(eops, jd1, jd2, casa_compatible)
    mjd_0, mjd_1 = eops["MJD"][i0], eops["MJD"][i1]
    val_0, val_1 = eops["UT1_UTC"][i0], eops["UT1_UTC"][i1]
    d_val = val_1 - val_0
    d_val = d_val - np.round(d_val)
    return val_0 + (mjd1 - mjd_0 + mjd2) / (mjd_1 - mjd_0) * d_val


def pm_xy(eops: dict, jd1, jd2, casa_compatible: bool = False):
    """Polar motion (x, y) [rad] linearly interpolated (eop.py ``pm_xy``)."""
    mjd1, mjd2 = jd1 - 2400000.5, jd2
    i0, i1 = _bracket(eops, jd1, jd2, casa_compatible)
    mjd_0, mjd_1 = eops["MJD"][i0], eops["MJD"][i1]
    val_0, val_1 = eops["PM_x"][i0], eops["PM_x"][i1]
    xp = val_0 + (mjd1 - mjd_0 + mjd2) / (mjd_1 - mjd_0) * (val_1 - val_0)
    val_0, val_1 = eops["PM_y"][i0], eops["PM_y"][i1]
    yp = val_0 + (mjd1 - mjd_0 + mjd2) / (mjd_1 - mjd_0) * (val_1 - val_0)
    return xp * ARCSEC, yp * ARCSEC


def _as_arrays(eops: dict):
    """The EOP lists as float64 arrays, or None when there is no entry."""
    if len(eops["MJD"]) == 0:
        return None
    return {name: np.asarray(values, dtype=np.float64) for name, values in eops.items()}


def eops_from_ms(vis, obsid: int):
    """The MS EARTH_ORIENTATION values of one observation (eop.py ``get_eops_from_ms``), or None."""
    eops = {"MJD": [], "UT1_UTC": [], "PM_x": [], "PM_y": []}
    path = Path(vis) / "EARTH_ORIENTATION"
    if not path.is_dir():
        return None
    with Table(path) as table:
        if table.nrows() == 0:
            return None
        obs = np.asarray(table.getcol("OBSERVATION_ID"))
        times = np.asarray(table.getcol("TIME"), dtype=np.float64)
        pms = np.asarray(table.getcol("PM"), dtype=np.float64).reshape(-1, 2)
        ut1 = np.asarray(table.getcol("UT1_UTC"), dtype=np.float64)
    for row in np.flatnonzero(obs == obsid):
        eops["MJD"].append(times[row] / 86400)
        eops["PM_x"].append(pms[row, 0] / ARCSEC)
        eops["PM_y"].append(pms[row, 1] / ARCSEC)
        eops["UT1_UTC"].append(ut1[row])
    return _as_arrays(eops)


def eops_from_file(infile, mjd_min, mjd_max):
    """The EOPs of ``infile`` within ``[mjd_min, mjd_max]`` (eop.py ``get_eops_from_file``), or None.

    As CASA, a parse error gives None (the observation is then skipped).
    """
    with open(infile) as fp:
        try:
            line = fp.readline()
            if line.startswith("EOP-MOD Ver 2.0"):
                return _parse_usno(fp, mjd_min, mjd_max, (10, 17), (18, 25), (26, 35))
            if line.startswith("EOP-MOD Ver 2.1"):
                return _parse_usno(fp, mjd_min, mjd_max, (10, 18), (19, 27), (28, 39))
            return _parse_eopc04(fp, mjd_min, mjd_max)
        except Exception:                                   # noqa: BLE001 - CASA's bare except
            return None


def _parse_usno(fp, mjd_min, mjd_max, x_cols, y_cols, ut_cols):
    """USNO finals 2.0/2.1 lines (column ranges given) -> EOPs with UTC MJD (eop.py ``parse_usno_2_x``)."""
    eops = {"MJD": [], "UT1_UTC": [], "PM_x": [], "PM_y": []}
    for line in fp:
        if re.match(r"#", line):
            continue
        jd_tai = float(line[0:9])
        mjd = jd_tai - 2400000.5
        if mjd < mjd_min or mjd > mjd_max:
            continue
        pm_x = float(line[x_cols[0]:x_cols[1]]) / 10
        pm_y = float(line[y_cols[0]:y_cols[1]]) / 10
        ut1_tai = float(line[ut_cols[0]:ut_cols[1]]) / 1e6
        jd_utc = erfa.taiutc(jd_tai, 0)
        utc_tai = (jd_utc[1] + (jd_utc[0] - jd_tai)) * 86400
        eops["MJD"].append(jd_utc[1] + (jd_utc[0] - 2400000.5))
        eops["UT1_UTC"].append(ut1_tai - utc_tai)
        eops["PM_x"].append(pm_x)
        eops["PM_y"].append(pm_y)
    return _as_arrays(eops)


def _parse_eopc04(fp, mjd_min, mjd_max):
    """IERS EOP C04 lines -> EOPs (eop.py ``parse_eopc04``)."""
    eops = {"MJD": [], "UT1_UTC": [], "PM_x": [], "PM_y": []}
    for line in fp:
        if re.match(r"\s", line) or re.match(r"#", line):
            continue
        mjd = int(line[12:19])
        if mjd < mjd_min or mjd > mjd_max:
            continue
        eops["MJD"].append(mjd)
        eops["PM_x"].append(float(line[20:30]))
        eops["PM_y"].append(float(line[31:41]))
        eops["UT1_UTC"].append(float(line[42:53]))
    return _as_arrays(eops)


def _write_table(caltable: Path, vis: Path, rows: dict, spw_values: dict) -> None:
    """Write the main table (one StandardStMan, like CASA) and its five subtables."""
    nrow = int(rows["TIME"].size)
    shape = (nrow, 1, 8)
    values = {"TIME": rows["TIME"], "FIELD_ID": rows["FIELD_ID"], "SPECTRAL_WINDOW_ID": rows["SPECTRAL_WINDOW_ID"],
              "ANTENNA1": rows["ANTENNA1"], "ANTENNA2": np.full(nrow, -1, dtype=np.int32),
              "INTERVAL": np.zeros(nrow), "SCAN_NUMBER": np.full(nrow, -1, dtype=np.int32),
              "OBSERVATION_ID": rows["OBSERVATION_ID"], "FPARAM": rows["FPARAM"],
              "PARAMERR": np.full(shape, -1.0, dtype=np.float32), "FLAG": np.zeros(shape, dtype=bool),
              "SNR": np.ones(shape, dtype=np.float32), "WEIGHT": np.ones(shape, dtype=np.float32)}
    keywords = {"ParType": "Float", "MSName": vis.absolute().name, "VisCal": VISCAL, "PolBasis": "unknown",
                "CASA_Version": CASA_VERSION}
    subtables = ("OBSERVATION", "ANTENNA", "FIELD", "SPECTRAL_WINDOW", "HISTORY")
    keywords.update(_link_subtables(caltable, subtables))
    _create(caltable, main_table_desc(), nrow, values, keywords=keywords, info=("Calibration", VISCAL))
    for name in ("OBSERVATION", "ANTENNA", "FIELD"):
        write_calibration_subtable(vis / name, caltable / name, SUBTABLE_DESCS[name])
    write_calibration_subtable(vis / "SPECTRAL_WINDOW", caltable / "SPECTRAL_WINDOW", SUBTABLE_DESCS["SPECTRAL_WINDOW"],
                               transform=lambda v, n: spw_values)
    history = history_table_desc()
    _create(caltable / "HISTORY", {name: history[name] for name in HISTORY_ORDER}, 0, {})
