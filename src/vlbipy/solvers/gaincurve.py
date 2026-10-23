"""The gain-curve table of the a-priori amplitude calibration (the dask-ms backend's ``gencal gc``), without CASA.

:func:`gaincurve` writes what ``casatasks.gencal(caltype='gc')`` writes for a measurement set with a
``GAIN_CURVE`` subtable: an ``EPowerCurve`` table (CASA ``EJones.cc``, ``EPowerCurve::specify``), with the same
arithmetic, row order, columns, keywords and subtables.

What CASA does, and this module repeats:

* one row per ``GAIN_CURVE`` row whose ``SPECTRAL_WINDOW_ID`` is a row of the MS ``SPECTRAL_WINDOW`` table;
  ``FEED_ID`` is ignored;
* ``FPARAM`` is a ``(1, 16)`` cell: 8 coefficients per polarization (R then L), the POWER gain ``DPFU x POLY``
  in K/Jy as a polynomial in **zenith angle** [deg]. At most the first 8 terms of ``GAIN`` are used (a longer
  polynomial is cut off without a message). ``POWER(EL)``/``VOLTAGE(EL)`` polynomials are re-expanded in zenith
  angle with the matrix ``m[i, j] = C(j, i) (-1)^j (-90)^(j - i)`` (float32), ``POWER(ZA)``/``VOLTAGE(ZA)`` are
  copied (identity matrix), anything else makes CASA fail;
* a ``VOLTAGE`` polynomial is squared into a power polynomial, but CASA squares the first polynomial only: the
  coefficients of the second polarization come out as zero (repeated here as it is);
* each polarization block is multiplied by its ``SENSITIVITY`` (DPFU). A station with ``DPFU = 1, POLY = 1``
  (no real gain curve) therefore gets ``FPARAM = [1, 0, ..., 0]``;
* every product and sum is a float32 operation done in CASA's order (matrix-vector product row by row, terms
  added one at a time, no fused multiply-add), so the table is bit-identical to CASA's;
* ``TIME = (begin + end) / 2`` and ``INTERVAL = (end - begin) / 2`` (half the ``GAIN_CURVE`` interval) with
  ``begin/end = TIME -/+ INTERVAL / 2``; ``FIELD_ID``, ``SCAN_NUMBER``, ``OBSERVATION_ID`` and ``ANTENNA2`` are
  -1; ``PARAMERR`` and ``SNR`` are 0 (CASA writes the parameter errors into SNR as well); ``FLAG`` is False;
  ``WEIGHT`` is left undefined;
* rows sorted (stably) by spectral window, time, ANTENNA1, ANTENNA2 (``NewCalTable::writeToDisk``);
* subtables: OBSERVATION, ANTENNA, FIELD with the values of the MS rows (only the columns a calibration table
  has) under CASA's own column descriptions (``CT*::requiredTableDesc``: FIELD directions are labelled J2000
  whatever the MS says, as CASA copies values and not keywords), an empty HISTORY, and SPECTRAL_WINDOW with one
  channel per subband: the mean channel frequency and the summed channel widths, resolutions and effective
  bandwidths, ``TOTAL_BANDWIDTH = |sum of widths|``, ``FLAG_ROW`` set for the subbands with no row in the main
  table.

The only intended difference: the ``CASA_Version`` keyword reads ``vlbipy`` (as in the other tables written by
:mod:`vlbipy.solvers.caltable`).
"""
from __future__ import annotations

import math
import time as _time
from pathlib import Path

import numpy as np

from ..casacore_format.aipsio import TP_FLOAT
from ..casacore_format.table import Table, is_table
from ..casacore_format.tabledat import UnsupportedTable
from ..logging_utils import get_logger
from .caltable import (FREQ_MEASINFO, _array_column, _create, _epoch_keywords, _link_subtables, _scalar_column,
                       history_table_desc, main_table_desc)

logger = get_logger()

VISCAL = "EPowerCurve"
#: Polynomial coefficients per polarization in an EPowerCurve table (``EPowerCurve::nPar() / 2``).
NCOEF = 8
NPAR = 2 * NCOEF
ZA_TYPES = ("POWER(ZA)", "VOLTAGE(ZA)")
EL_TYPES = ("POWER(EL)", "VOLTAGE(EL)")
_ITRF = {"MEASINFO": {"Ref": "ITRF", "type": "position"}, "QuantumUnits": np.array(["m", "m", "m"])}
_J2000 = {"MEASINFO": {"Ref": "J2000", "type": "direction"}, "QuantumUnits": np.array(["rad", "rad"])}
_HZ = {"QuantumUnits": np.array(["Hz"])}
_FREQ = {"MEASINFO": dict(FREQ_MEASINFO), "QuantumUnits": np.array(["Hz"])}
#: Column descriptions of the calibration-table subtables, in CASA's order (``CT*::requiredTableDesc``, as
#: casatools reports them for a table written by gencal): the MS values are written under these, never under the
#: descriptions of the MS.
SUBTABLE_DESCS = {
    "OBSERVATION": {
        "TIME_RANGE": _array_column("double", "Start and end of observation", _epoch_keywords(), 1, 5, [2]),
        "LOG": _array_column("string", "Observing log", ndim=1),
        "SCHEDULE": _array_column("string", "Observing schedule", ndim=1),
        "FLAG_ROW": _scalar_column("boolean", "Row flag"),
        "OBSERVER": _scalar_column("string", "Name of observer(s)"),
        "PROJECT": _scalar_column("string", "Project identification string"),
        "RELEASE_DATE": _scalar_column("double", "Release date when data becomes public", _epoch_keywords()),
        "SCHEDULE_TYPE": _scalar_column("string", "Observing schedule type"),
        "TELESCOPE_NAME": _scalar_column("string", "Telescope Name (e.g. WSRT, VLBA)")},
    "ANTENNA": {
        "OFFSET": _array_column("double", "Axes offset of mount to FEED REFERENCE point", _ITRF, 1, 5, [3]),
        "POSITION": _array_column("double", "Antenna X,Y,Z phase reference position", _ITRF, 1, 5, [3]),
        "TYPE": _scalar_column("string", "Antenna type (e.g. SPACE-BASED)"),
        "DISH_DIAMETER": _scalar_column("double", "Physical diameter of dish", {"QuantumUnits": np.array(["m"])}),
        "FLAG_ROW": _scalar_column("boolean", "Flag for this row"),
        "MOUNT": _scalar_column("string", "Mount type e.g. alt-az, equatorial, etc."),
        "NAME": _scalar_column("string", "Antenna name, e.g. VLA22, CA03"),
        "STATION": _scalar_column("string", "Station (antenna pad) name")},
    "FIELD": {
        "DELAY_DIR": _array_column("double", "Direction of delay center (e.g. RA, DEC)as polynomial in time.", _J2000,
                                   2),
        "PHASE_DIR": _array_column("double", "Direction of phase center (e.g. RA, DEC).", _J2000, 2),
        "REFERENCE_DIR": _array_column("double", "Direction of REFERENCE center (e.g. RA, DEC).as polynomial in "
                                       "time.", _J2000, 2),
        "CODE": _scalar_column("string", "Special characteristics of field, e.g. Bandpass calibrator"),
        "FLAG_ROW": _scalar_column("boolean", "Row Flag"),
        "NAME": _scalar_column("string", "Name of this field"),
        "NUM_POLY": _scalar_column("int", "Polynomial order of _DIR columns"),
        "SOURCE_ID": _scalar_column("int", "Source id"),
        "TIME": _scalar_column("double", "Time origin for direction and rate", _epoch_keywords())},
    "SPECTRAL_WINDOW": {
        "MEAS_FREQ_REF": _scalar_column("int", "Frequency Measure reference"),
        "CHAN_FREQ": _array_column("double", "Center frequencies for each channel in the data matrix", _FREQ, 1),
        "REF_FREQUENCY": _scalar_column("double", "The reference frequency", _FREQ),
        "CHAN_WIDTH": _array_column("double", "Channel width for each channel", _HZ, 1),
        "EFFECTIVE_BW": _array_column("double", "Effective noise bandwidth of each channel", _HZ, 1),
        "RESOLUTION": _array_column("double", "The effective noise bandwidth for each channel", _HZ, 1),
        "FLAG_ROW": _scalar_column("boolean", "Row flag"),
        "FREQ_GROUP": _scalar_column("int", "Frequency group"),
        "FREQ_GROUP_NAME": _scalar_column("string", "Frequency group name"),
        "IF_CONV_CHAIN": _scalar_column("int", "The IF conversion chain number"),
        "NAME": _scalar_column("string", "Spectral window name"),
        "NET_SIDEBAND": _scalar_column("int", "Net sideband"),
        "NUM_CHAN": _scalar_column("int", "Number of spectral channels"),
        "TOTAL_BANDWIDTH": _scalar_column("double", "The total bandwidth for this window", _HZ)}}
#: Column order of CASA's (empty) HISTORY subtable.
HISTORY_ORDER = ("APP_PARAMS", "CLI_COMMAND", "APPLICATION", "MESSAGE", "OBJECT_ID", "OBSERVATION_ID", "ORIGIN",
                 "PRIORITY", "TIME")
#: Channel columns of SPECTRAL_WINDOW reduced to one channel, and how (``mean`` or ``sum`` over the channels).
SPW_CHANNEL_REDUCTION = {"CHAN_FREQ": "mean", "CHAN_WIDTH": "sum", "RESOLUTION": "sum", "EFFECTIVE_BW": "sum"}
#: gencal keywords that EPowerCurve never reads (any value gives the same table).
IGNORED_KEYWORDS = ("infile", "uniform", "endpoint", "timeout", "retry", "retry_wait_time", "ant_pos_time_limit")
#: gencal selection keywords: CASA parses them but EPowerCurve ignores them; only their empty defaults are taken.
SELECTION_KEYWORDS = ("spw", "antenna", "pol", "parameter")
CASA_VERSION = "vlbipy"


def gaincurve(vis, caltable, **gencal_kwargs) -> dict:
    """Write the ``EPowerCurve`` table ``gencal(vis=vis, caltable=caltable, caltype='gc')`` writes.

    Parameters
    ----------
    vis : str or pathlib.Path
        The measurement set (or Multi-MS). Only read.
    caltable : str or pathlib.Path
        The table to write (an existing one is replaced, as gencal does).
    **gencal_kwargs
        The other keywords of ``casatasks.gencal``: ``caltype`` must be ``'gc'``; ``spw``, ``antenna``, ``pol``
        and ``parameter`` must be empty (None, ``''`` or ``[]``); the keywords of :data:`IGNORED_KEYWORDS` are
        accepted with any value, as CASA ignores them for this table.

    Returns
    -------
    dict
        ``rows`` (rows written), ``antennas`` (distinct ANTENNA1), ``spws`` (subbands with rows),
        ``flagged_spws`` (subbands without), ``dropped`` (GAIN_CURVE rows of no MS subband), ``truncated``
        (rows whose polynomial had more than 8 terms), ``seconds``.

    Raises
    ------
    NotImplementedError
        For anything CASA does differently or fails on, so that the caller can run CASA instead: another
        caltype, a non-empty selection, an unknown keyword, an MS without a ``GAIN_CURVE`` keyword (CASA then
        uses the VLA gain-curve tables), a table this package cannot read, a gain-curve type CASA rejects, an
        antenna outside the ANTENNA table, a ``GAIN`` cell without two polarizations or with fewer terms than
        used, a ``SENSITIVITY`` cell with fewer than two values, or ephemeris fields.
    """
    started = _time.perf_counter()
    vis, caltable = Path(vis), Path(caltable)
    _check_keywords(gencal_kwargs)
    try:
        curves = read_gain_curve(vis)
        with Table(vis / "SPECTRAL_WINDOW") as table:
            nspw = table.nrows()
        with Table(vis / "ANTENNA") as table:
            nant = table.nrows()
    except (UnsupportedTable, OSError) as exc:
        raise NotImplementedError(f"gencal gc: {vis} could not be read without casacore ({exc})") from exc
    used = np.flatnonzero((curves["spw"] >= 0) & (curves["spw"] < nspw))
    _check_rows(curves, used, nant)
    order = used[np.lexsort((curves["antenna"][used], curves["time"][used], curves["spw"][used]))]
    params = gain_curve_parameters([curves["type"][k] for k in order], curves["num_poly"][order],
                                   [curves["gain"][k] for k in order], [curves["sensitivity"][k] for k in order])
    begin = curves["time"][order] - curves["interval"][order] / 2
    end = curves["time"][order] + curves["interval"][order] / 2
    spws = curves["spw"][order].astype(np.int32)
    _write_table(caltable, vis, times=(begin + end) / 2, intervals=(end - begin) / 2, spws=spws,
                 antennas=curves["antenna"][order].astype(np.int32), params=params)
    present = np.unique(spws)
    report = {"rows": int(order.size), "antennas": int(np.unique(curves["antenna"][order]).size),
              "spws": int(present.size), "flagged_spws": int(nspw - present.size),
              "dropped": int(curves["spw"].size - used.size),
              "truncated": int(np.sum(curves["num_poly"][order] > NCOEF)), "seconds": _time.perf_counter() - started}
    if report["truncated"]:
        logger.warning("gencal[native] gc: {} GAIN_CURVE row(s) of {} hold more than {} coefficients; only the first "
                       "{} are written, as CASA does", report["truncated"], vis.name, NCOEF, NCOEF)
    logger.info("gencal[native] gc: {} GAIN_CURVE row(s) -> {} ({} row(s), {} antenna(s), {} subband(s), {} subband(s) "
                "without a gain curve flagged, {} row(s) of no subband dropped) in {:.2f}s", curves["spw"].size,
                caltable.name, report["rows"], report["antennas"], report["spws"], report["flagged_spws"],
                report["dropped"], report["seconds"])
    return report


def _check_keywords(gencal_kwargs: dict) -> None:
    """Raise NotImplementedError unless ``gencal_kwargs`` ask for exactly the table this module writes."""
    kwargs = dict(gencal_kwargs)
    caltype = kwargs.pop("caltype", None)
    if str(caltype).lower() != "gc":
        raise NotImplementedError(f"gencal caltype={caltype!r} (only 'gc' is written natively)")
    for name in SELECTION_KEYWORDS:
        value = kwargs.pop(name, None)
        if value is not None and len(value) > 0:
            raise NotImplementedError(f"gencal gc with {name}={value!r} (only the empty default is supported)")
    for name in IGNORED_KEYWORDS:
        kwargs.pop(name, None)
    if kwargs:
        raise NotImplementedError(f"gencal gc with keyword(s) {sorted(kwargs)}")


def read_gain_curve(vis) -> dict:
    """Read the ``GAIN_CURVE`` subtable of ``vis`` with :mod:`vlbipy.casacore_format`.

    Returns
    -------
    dict
        Parallel per-row values: ``antenna``, ``spw``, ``num_poly`` (int64 arrays), ``time``, ``interval``
        (float64 arrays), ``type`` (list of str), ``gain`` (list of float32 cells in C order
        ``(n_coefficients, n_pol)``, None where undefined) and ``sensitivity`` (list of 1-D float32 arrays, None
        where undefined).

    Raises
    ------
    NotImplementedError
        When the MS has no ``GAIN_CURVE`` keyword or table (CASA would use another gain-curve source), when
        GAIN or SENSITIVITY is not a float column, or when an ephemeris field is present (CASA rewrites its
        directions in the FIELD subtable).
    """
    vis = Path(vis)
    with Table(vis) as main:
        keywords = dict(main.dat.keywords)
    if "GAIN_CURVE" not in keywords or not is_table(vis / "GAIN_CURVE"):
        raise NotImplementedError(f"{vis.name} has no GAIN_CURVE subtable (CASA uses its own gain-curve tables)")
    _check_fields(vis)
    with Table(vis / "GAIN_CURVE") as table:
        for name in ("GAIN", "SENSITIVITY"):
            column = table.coldesc(name)
            if column.dtype != TP_FLOAT or not column.is_array:
                raise NotImplementedError(f"{vis.name}/GAIN_CURVE: column {name} is not a float array")
        if table.nrows() == 0:
            return {"antenna": np.zeros(0, np.int64), "spw": np.zeros(0, np.int64),
                    "num_poly": np.zeros(0, np.int64), "time": np.zeros(0), "interval": np.zeros(0), "type": [],
                    "gain": [], "sensitivity": []}
        columns = {name: table.getcol(name) for name in ("ANTENNA_ID", "SPECTRAL_WINDOW_ID", "NUM_POLY", "TIME",
                                                         "INTERVAL", "TYPE", "GAIN", "SENSITIVITY")}
    return {"antenna": np.asarray(columns["ANTENNA_ID"], dtype=np.int64),
            "spw": np.asarray(columns["SPECTRAL_WINDOW_ID"], dtype=np.int64),
            "num_poly": np.asarray(columns["NUM_POLY"], dtype=np.int64),
            "time": np.asarray(columns["TIME"], dtype=np.float64),
            "interval": np.asarray(columns["INTERVAL"], dtype=np.float64),
            "type": [str(value) for value in columns["TYPE"]],
            "gain": [None if cell is None else np.asarray(cell, dtype=np.float32) for cell in columns["GAIN"]],
            "sensitivity": [None if cell is None else np.asarray(cell, dtype=np.float32).reshape(-1)
                            for cell in columns["SENSITIVITY"]]}


def _check_fields(vis: Path) -> None:
    """Raise NotImplementedError when the FIELD subtable has ephemeris objects (``EPHEMERIS_ID >= 0``)."""
    with Table(vis / "FIELD") as field:
        ephemeris = field.getcol("EPHEMERIS_ID") if "EPHEMERIS_ID" in field.colnames() and field.nrows() else []
    if len(ephemeris) and np.max(ephemeris) >= 0:
        raise NotImplementedError(f"{vis.name}: FIELD has ephemeris objects (CASA rewrites their directions)")


def _check_rows(curves: dict, rows: np.ndarray, nant: int) -> None:
    """Raise NotImplementedError for a GAIN_CURVE row CASA rejects or cannot read safely (see the caller)."""
    for k in rows:
        kind, gain, sensitivity = curves["type"][k], curves["gain"][k], curves["sensitivity"][k]
        if kind not in ZA_TYPES + EL_TYPES:
            raise NotImplementedError(f"GAIN_CURVE row {k}: gain-curve type {kind!r} (CASA does not support it)")
        if not 0 <= curves["antenna"][k] < nant:
            raise NotImplementedError(f"GAIN_CURVE row {k}: ANTENNA_ID {curves['antenna'][k]} is not in ANTENNA")
        used = min(int(curves["num_poly"][k]), NCOEF)
        if gain is None or gain.ndim != 2 or gain.shape[1] < 2 or gain.shape[0] < used:
            raise NotImplementedError(f"GAIN_CURVE row {k}: GAIN cell {None if gain is None else gain.shape} "
                                      f"does not hold 2 polarizations x {used} coefficients")
        if sensitivity is None or sensitivity.size < 2:
            raise NotImplementedError(f"GAIN_CURVE row {k}: SENSITIVITY has fewer than two values")


def el_to_za_matrix() -> np.ndarray:
    """The float32 ``(16, 16)`` matrix CASA re-expands elevation polynomials in zenith angle with.

    Block diagonal, one ``8 x 8`` block per polarization, ``m[i, j] = C(j, i) (-1)^j (-90)^(j - i)`` for
    ``j >= i`` (0 below the diagonal), computed in double precision and rounded to float32. Applied to the
    coefficients ``c_j`` of ``P(el) = sum c_j el^j`` it gives those of ``P(90 - za)`` in powers of ``za``.
    """
    block = np.zeros((NCOEF, NCOEF), dtype=np.float32)
    for i in range(NCOEF):
        for j in range(i, NCOEF):
            block[i, j] = np.float32(math.comb(j, i) * (-1.0) ** j * (-90.0) ** (j - i))
    matrix = np.zeros((NPAR, NPAR), dtype=np.float32)
    matrix[:NCOEF, :NCOEF] = block
    matrix[NCOEF:, NCOEF:] = block
    return matrix


def gain_curve_parameters(types, num_poly, gains, sensitivities) -> np.ndarray:
    """The FPARAM vectors of GAIN_CURVE rows, with CASA's float32 arithmetic (``EPowerCurve::specify``).

    Parameters
    ----------
    types : list of str
        TYPE of each row (one of :data:`ZA_TYPES` or :data:`EL_TYPES`).
    num_poly : array of int
        NUM_POLY of each row; only the first ``min(NUM_POLY, 8)`` coefficients are used.
    gains : list of float32 arrays
        GAIN cells in C order ``(n_coefficients, n_pol)``; polarizations 0 and 1 are used.
    sensitivities : list of float32 arrays
        SENSITIVITY cells (DPFU of polarizations 0 and 1 [K/Jy]).

    Returns
    -------
    numpy.ndarray
        float32 ``(nrow, 16)``: the zenith-angle power polynomial times DPFU, 8 coefficients per polarization.
    """
    nrow = len(types)
    coefficients = np.zeros((nrow, NPAR), dtype=np.float32)
    for row in range(nrow):
        used = max(min(int(num_poly[row]), NCOEF), 0)
        coefficients[row, :used] = gains[row][:used, 0]
        coefficients[row, NCOEF:NCOEF + used] = gains[row][:used, 1]
    matrices = np.empty((nrow, NPAR, NPAR), dtype=np.float32)
    matrices[:] = np.eye(NPAR, dtype=np.float32)
    elevation = np.array([kind in EL_TYPES for kind in types], dtype=bool)
    matrices[elevation] = el_to_za_matrix()
    params = _matrix_vector_float32(matrices, coefficients)
    voltage = np.array([kind.startswith("VOLTAGE") for kind in types], dtype=bool)
    if voltage.any():
        params[voltage] = _square_first_polynomial(params[voltage])
    dpfu = np.array([cell[:2] for cell in sensitivities], dtype=np.float32).reshape(nrow, 2)
    params[:, :NCOEF] *= dpfu[:, :1]
    params[:, NCOEF:] *= dpfu[:, 1:]
    return params


def _matrix_vector_float32(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """``matrices @ vectors`` per row as casacore's ``product(Matrix, Vector)``: float32, terms added in order.

    Each element is ``s = 0; s += m[i, j] * v[j]`` for ``j = 0 .. n-1``, the product and the sum each rounded to
    float32 (CASA's build does not fuse them), which numpy's float32 ufuncs reproduce exactly.
    """
    out = np.zeros(vectors.shape, dtype=np.float32)
    for j in range(vectors.shape[1]):
        out = out + matrices[:, :, j] * vectors[:, j:j + 1]
    return out


def _square_first_polynomial(params: np.ndarray) -> np.ndarray:
    """CASA's voltage -> power step: the square of the first polynomial; the second polarization becomes zero."""
    squared = np.zeros(params.shape, dtype=np.float32)
    for i in range(NCOEF):
        for j in range(NCOEF):
            if i + j < NCOEF:
                squared[:, i + j] = squared[:, i + j] + params[:, i] * params[:, j]
    return squared


def _write_table(caltable: Path, vis: Path, *, times, intervals, spws, antennas, params) -> None:
    """Write the main table (one StandardStMan, like CASA) and its five subtables."""
    nrow = int(spws.size)
    cells = params.reshape(nrow, 1, NPAR)                                     # (1 channel, 16) per row
    values = {"TIME": np.asarray(times, dtype=np.float64), "FIELD_ID": np.full(nrow, -1, dtype=np.int32),
              "SPECTRAL_WINDOW_ID": spws, "ANTENNA1": antennas, "ANTENNA2": np.full(nrow, -1, dtype=np.int32),
              "INTERVAL": np.asarray(intervals, dtype=np.float64), "SCAN_NUMBER": np.full(nrow, -1, dtype=np.int32),
              "OBSERVATION_ID": np.full(nrow, -1, dtype=np.int32), "FPARAM": cells,
              "PARAMERR": np.zeros(cells.shape, dtype=np.float32), "FLAG": np.zeros(cells.shape, dtype=bool),
              "SNR": np.zeros(cells.shape, dtype=np.float32)}
    keywords = {"ParType": "Float", "MSName": vis.absolute().name, "VisCal": VISCAL, "PolBasis": "unknown",
                "CASA_Version": CASA_VERSION}
    subtables = ("OBSERVATION", "ANTENNA", "FIELD", "SPECTRAL_WINDOW", "HISTORY")
    keywords.update(_link_subtables(caltable, subtables))
    _create(caltable, main_table_desc(), nrow, values, keywords=keywords, info=("Calibration", VISCAL))
    for name in ("OBSERVATION", "ANTENNA", "FIELD"):
        write_calibration_subtable(vis / name, caltable / name, SUBTABLE_DESCS[name])
    write_calibration_subtable(vis / "SPECTRAL_WINDOW", caltable / "SPECTRAL_WINDOW", SUBTABLE_DESCS["SPECTRAL_WINDOW"],
                               transform=lambda values, nrow: _single_channel(values, nrow, spws))
    history = history_table_desc()
    _create(caltable / "HISTORY", {name: history[name] for name in HISTORY_ORDER}, 0, {})


def write_calibration_subtable(source: Path, dest: Path, desc: dict, transform=None) -> Path:
    """Write the rows of MS subtable ``source`` as a calibration-table subtable ``dest`` described by ``desc``.

    As CASA's ``TableCopy::copyRows`` into a ``CT*::requiredTableDesc`` table: the values of the ``desc`` columns
    are copied, the descriptions (and their measure keywords) are those of ``desc``, the other columns of the MS
    are left out, and the table has no keywords and an empty ``table.info``. ``transform(values, nrow) ->
    values`` may change the values before they are written.

    Raises
    ------
    NotImplementedError
        When the source lacks one of the ``desc`` columns.
    """
    with Table(source) as table:
        missing = sorted(set(desc) - set(table.colnames()))
        if missing:
            raise NotImplementedError(f"{source}: no column(s) {missing}")
        nrow = table.nrows()
        values = {name: (table.getcol(name) if nrow else None) for name in desc}
    if transform is not None:
        values = transform(values, nrow)
    return _create(dest, desc, nrow, {name: value for name, value in values.items() if value is not None})


def _single_channel(values: dict, nrow: int, spws: np.ndarray) -> dict:
    """SPECTRAL_WINDOW values reduced to one channel per subband and flagged where the table has no row.

    CASA (``SolvableVisCal::createMemCalTable``): the mean channel frequency, the summed channel widths,
    resolutions and effective bandwidths, one channel, ``TOTAL_BANDWIDTH = |sum of widths|``; then
    ``NewCalTable::flagAbsentSpws``. Sums run channel by channel (``std::accumulate``), as numpy's cumsum does.
    """
    out = dict(values)
    for name, how in SPW_CHANNEL_REDUCTION.items():
        reduced = np.zeros((nrow, 1), dtype=np.float64)
        for row, cell in enumerate(_cells(values[name], nrow)):
            total = np.cumsum(cell)[-1] if cell.size else 0.0
            reduced[row, 0] = total / (1.0 * cell.size) if how == "mean" else total
        out[name] = reduced
    out["NUM_CHAN"] = np.ones(nrow, dtype=np.int32)
    out["TOTAL_BANDWIDTH"] = np.abs(out["CHAN_WIDTH"][:, 0])
    out["FLAG_ROW"] = ~np.isin(np.arange(nrow), spws)
    return out


def _cells(column, nrow: int) -> list:
    """The cells of an array column (an ``(nrow, n)`` array or a list of arrays) as 1-D float64 arrays."""
    if column is None:
        return [np.zeros(0)] * nrow
    return [np.zeros(0) if cell is None else np.asarray(cell, dtype=np.float64).reshape(-1) for cell in column]
