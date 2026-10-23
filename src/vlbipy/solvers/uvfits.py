"""Export a measurement set to UVFITS without casacore: what ``casatasks.exportuvfits`` writes.

:func:`exportuvfits` reproduces casacore's ``MSFitsOutput`` (the engine behind ``exportuvfits`` and
``ms.tofits``) for single-source exports of every channel of every spectral window; the file is written
card by card and byte by byte as casacore writes it, so it is identical to CASA's except for the AIPS AN
keywords that casacore takes from its measures tables (see below). The rules, all read off
``MSFitsOutput.cc`` and checked against ``casatasks.exportuvfits`` on the RSM07 split measurement sets:

* Rows are sorted (stably) by TIME_CENTROID, ANTENNA1, ANTENNA2, FIELD_ID, plus DATA_DESC_ID without
  ``combinespw``. The random parameters of an output row come from the first sorted row of its group:
  UU, VV, WW = UVW * (1 / c) in seconds; DATE, DATE = Julian day number and day fraction of TIME_CENTROID;
  BASELINE = 256 * ant1 + ant2 + 0.01 * ARRAY_ID (antennas numbered from 1, or by their names when every
  name is an integer), or SUBARRAY, ANTENNA1, ANTENNA2 when an antenna number exceeds 255; FREQSEL.
* ``combinespw``: the spectral windows become the IFs of one group. A group ("time-baseline-field") is a run
  of at most one row per IF of one baseline whose TIME_CENTROID lies within 0.2 INTERVAL of the first; its rows
  are ordered by IF and copied while the IF numbers increase. With ``padwithflags`` missing IFs are padded
  with (0, 0, -0.0); CASA's quirks are kept: an IF that the first row of the *next* group fills after the
  group's last IF is copied from it as well, and a group with a repeated IF spills into extra rows that are
  cut at the expected row count. Without ``padwithflags`` every group must hold each IF exactly once.
* Data (re, im, weight) per IF, channel and correlation, the correlations in FITS Stokes order. The weight is
  WEIGHT_SPECTRUM (WEIGHT / nchan without it); a flagged sample keeps its visibility with weight -w (0 when
  w <= 0); FLAG_ROW makes every weight -|w|. Values pass through float32 additions as in casacore, so -0.0
  visibilities are written as +0.0.
* Header keywords and extension tables: AIPS FQ (one row, or one per spectral window without
  ``combinespw``), AIPS AN (one per OBSERVATION row) and AIPS WX when the measurement set has a WEATHER
  subtable.

AIPS AN keywords that casacore computes with its measures tables are computed here from the IERS-A table
bundled with astropy (never downloaded): GSTIA0 and DEGPDY with the IAU 1982 GMST (``erfa.gmst82``),
POLARX/POLARY (polar motion at 0h UTC of the day, times 6356752.31 m), UT1UTC (at the first TIME) and IATUTC
(``erfa.dat``, exact, where casacore's double-precision difference can be 1e-6 s off). On RSM07 (2025-09-16)
they agree with CASA 6.7 to 2e-8 deg (GSTIA0), 1.2e-7 deg (DEGPDY), 0.6 mm (POLARX/Y), 1.5e-5 s (UT1UTC)
and 1e-6 s (IATUTC); the differences come from the IERS tables, not from the formulas.

Options or data that this module does not reproduce raise :class:`NotImplementedError` (the caller falls back
to CASA): ``multisource=True`` or several fields, telescope names with special code paths in casacore
(WSRT, ALMA, IRAM PDB), DATA_DESC_ID rows that do not map onto their own spectral window, direction
polynomials, variable measure references, non-UTC times, non-ITRF antenna positions, Stokes types without a
FITS code and partly defined WEIGHT_SPECTRUM cells. Data that CASA itself refuses to export raise
:class:`UvfitsExportError`.
"""
from __future__ import annotations

import math
import re
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger

logger = get_logger()

#: Speed of light [m/s] (casacore ``C::c``); UVW is multiplied by its inverse, as casacore does.
SPEED_OF_LIGHT = 299792458.0
SECONDS_PER_DAY = 86400.0
#: Julian date of MJD 0.
JD_OF_MJD0 = 2400000.5
#: Earth polar radius [m] that casacore multiplies the polar motion angles by (POLARX, POLARY).
POLAR_RADIUS = 6356752.31
#: Size of a FITS block [bytes].
FITS_BLOCK = 2880
#: Output buffer size per write [bytes]; bounds the memory of large exports.
CHUNK_BYTES = 64 * 1024 * 1024
#: Telescopes for which casacore writes something else (WSRT: another writer; ALMA, IRAM PDB: FQ quirks).
SPECIAL_TELESCOPES = ("WSRT", "ALMA", "IRAM PDB", "IRAM_PDB")
#: casacore Stokes type -> FITS Stokes value (Stokes::FITSValue), for the types it maps unambiguously.
STOKES_FITS = {1: 1, 2: 2, 3: 3, 4: 4, 5: -1, 6: -3, 7: -4, 8: -2, 9: -5, 10: -7, 11: -8, 12: -6}
#: MFrequency type code -> SPECSYS keyword value (MFrequency::showType, then casacore's FITS renaming).
SPECSYS = {0: "SOURCE", 1: "LSRK", 2: "LSRD", 3: "BARYCENT", 4: "GEOCENTR", 5: "TOPOCENT", 6: "GALACTOC",
           7: "LGROUP", 8: "CMB"}
#: Mount substrings -> AIPS MNTSTA code, checked in this order (casacore _writeAN).
MOUNT_CODES = (("ALT-AZ+NASMYTH-R", 4), ("ALT-AZ+NASMYTH-L", 5), ("ALT-AZ+BWG-R", 6), ("ALT-AZ+BWG-L", 7),
               ("ALT-AZ", 0), ("EQUATORIAL", 1), ("ORBIT", 2), ("X-Y", 3), ("SPACE-HALCA", 9), ("BIZARRE", 8))
#: Pressure units -> factor to Pa (WEATHER PRESSURE is converted to mbar).
PRESSURE_UNITS = {"Pa": 1.0, "hPa": 100.0, "kPa": 1000.0, "mbar": 100.0, "bar": 1.0e5}


class UvfitsExportError(RuntimeError):
    """The data cannot be exported (CASA's ``exportuvfits`` refuses them too)."""


def exportuvfits(vis, fitsfile, *, datacolumn: str = "data", multisource: bool = False, combinespw: bool = True,
                  padwithflags: bool = True, overwrite: bool = True) -> dict:
    """Write ``vis`` as a single-source UVFITS file, as ``casatasks.exportuvfits`` does with these options.

    Parameters
    ----------
    vis : path
        The measurement set (a plain or Multi-MS); every row is exported.
    fitsfile : path
        The UVFITS file to write.
    datacolumn : str
        ``"data"``/``"observed"``, ``"corrected"``/``"corrected_data"`` or ``"model"``/``"model_data"``
        (case-insensitive); a missing CORRECTED_DATA/MODEL_DATA column, or any other name, exports DATA, as CASA.
    multisource : bool
        Only ``False`` is supported (``True`` raises NotImplementedError).
    combinespw : bool
        Write the spectral windows as IFs of one frequency group (True) or as frequency groups (False).
    padwithflags : bool
        With ``combinespw``: fill IFs missing in a time-baseline group with flagged zeros.
    overwrite : bool
        Replace an existing ``fitsfile``; otherwise an existing file raises FileExistsError.

    Returns
    -------
    dict
        ``fitsfile``, ``rows`` (input rows), ``groups`` (output rows), ``padded_fraction`` (share of the
        output IF slots padded with flags), ``seconds``.

    Raises
    ------
    NotImplementedError
        Options or data this writer does not reproduce exactly (fall back to CASA).
    UvfitsExportError
        Data CASA cannot export either (varying channel counts, Stokes not representable, ...).
    """
    from ..casacore_format.table import Table

    start = _time.perf_counter()
    if multisource:
        raise NotImplementedError("exportuvfits: multisource=True (AIPS SU table) is not implemented natively")
    vis, fitsfile = Path(vis), Path(fitsfile)
    logger.info("exportuvfits: {} -> {} (datacolumn={}, combinespw={}, padwithflags={})", vis.name, fitsfile.name,
                datacolumn, combinespw, padwithflags)
    with Table(vis) as table:
        main = _read_main(table)
        column = _data_column(datacolumn, table.colnames())
        sub = _read_subtables(table)
        setup = _spectral_setup(sub, main["used_ddids"], combinespw)
        antnums = _antenna_numbers(sub["antenna"]["NAME"], sub["telescope"])
        sources = table.parts or [table]
        weight_spectrum = _weight_spectrum_defined(sources, setup["numchan"], setup["numcorr"])
        if_of_row = setup["if_of_ddid"][main["DATA_DESC_ID"]]
        order = _sorted_rows(main, combinespw)
        index, first, ntbf = _output_rows(main, order, if_of_row, setup["nif"], combinespw, padwithflags)
        params, names, comments = _random_parameters(main, first, antnums, combinespw, setup)
        padded = float(np.mean(index < 0)) if index.size else 0.0
        if fitsfile.exists():
            if not overwrite:
                raise FileExistsError(f"{fitsfile} exists (overwrite=False)")
            fitsfile.unlink()
        fitsfile.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(fitsfile, "wb") as handle:
                handle.write(_header_bytes(_primary_cards(main, sub, setup, names, comments, len(first), column,
                                                          table)))
                _write_groups(handle, sources, column, main, index, params, setup, weight_spectrum)
                handle.write(_fq_table(sub, setup, combinespw))
                for arraynum in range(len(sub["observation"]["TELESCOPE_NAME"])):
                    handle.write(_an_table(main, sub, setup, antnums, arraynum))
                if sub["weather"] is not None:
                    handle.write(_wx_table(main, sub["weather"], antnums))
        except BaseException:
            fitsfile.unlink(missing_ok=True)
            raise
    seconds = _time.perf_counter() - start
    logger.info("exportuvfits: {} rows -> {} groups ({} time-baseline groups, {:.2f}% of IFs padded) in {:.2f} s",
                main["nrow"], len(first), ntbf, 100.0 * padded, seconds)
    return {"fitsfile": str(fitsfile), "rows": int(main["nrow"]), "groups": int(len(first)),
            "padded_fraction": padded, "seconds": seconds}


# -- inputs ---------------------------------------------------------------------------------------------------

def _data_column(datacolumn: str, colnames: list) -> str:
    """The main-table column CASA exports for ``datacolumn`` (DATA when the requested column is missing)."""
    name = str(datacolumn).upper()
    if name in ("OBSERVED", "DATA"):
        return "DATA"
    if name in ("CORRECTED", "CORRECTED_DATA", "MODEL", "MODEL_DATA"):
        wanted = name if name.endswith("_DATA") else f"{name}_DATA"
        if wanted in colnames:
            return wanted
        logger.info("exportuvfits: {} does not exist, writing DATA", wanted)
        return "DATA"
    logger.warning("exportuvfits: unrecognized column {}, writing DATA", datacolumn)
    return "DATA"


def _read_main(table) -> dict:
    """The main-table scalar columns the export needs, plus ``nrow`` and ``used_ddids`` (sorted unique)."""
    nrow = table.nrows()
    if nrow == 0:
        raise UvfitsExportError(f"{table.path}: empty measurement set")
    if nrow > np.iinfo(np.int32).max:
        raise UvfitsExportError(f"{table.path}: too many rows for a FITS file ({nrow})")
    names = ("TIME", "TIME_CENTROID", "INTERVAL", "ANTENNA1", "ANTENNA2", "ARRAY_ID", "FIELD_ID", "DATA_DESC_ID",
             "FLAG_ROW")
    main = {name: np.asarray(table.getcol(name)) for name in names}
    main["UVW"] = np.asarray(table.getcol("UVW"), dtype=np.float64)
    main["nrow"] = nrow
    main["used_ddids"] = np.unique(main["DATA_DESC_ID"])
    if not np.all(main["FIELD_ID"] == main["FIELD_ID"][0]):
        raise NotImplementedError("exportuvfits: several fields (CASA writes a multi-source file)")
    if main["FIELD_ID"][0] != 0:
        raise NotImplementedError("exportuvfits: the only field is not FIELD_ID 0")
    for name in ("TIME", "TIME_CENTROID"):
        ref = table.getcolkeywords(name).get("MEASINFO", {}).get("Ref", "UTC")
        if ref != "UTC":
            raise NotImplementedError(f"exportuvfits: {name} in {ref}, not UTC")
    main["DATA_DESC_ID"] = main["DATA_DESC_ID"].astype(np.int64)
    return main


def _unit(table, column: str, default: str) -> str:
    """The first QuantumUnits entry of a column (``default`` without one)."""
    units = table.getcolkeywords(column).get("QuantumUnits")
    return str(units[0]) if units is not None and len(units) else default


def _require_unit(table, column: str, unit: str) -> None:
    """Raise NotImplementedError unless ``column`` is in ``unit`` (no unit conversions are implemented)."""
    found = _unit(table, column, unit)
    if found != unit:
        raise NotImplementedError(f"exportuvfits: {column} in {found}, not {unit}")


def _read_subtables(table) -> dict:
    """Everything read from the subtables: spectral windows, polarizations, field, antennas, feeds, ...

    Raises NotImplementedError for the layouts this writer does not reproduce (see the module docstring).
    """
    from ..casacore_format.aipsio import TableRef
    from ..casacore_format.table import Table

    def open_sub(name):
        return Table(table.subtable(name))

    sub = {}
    with open_sub("DATA_DESCRIPTION") as dd:
        sub["dd_spw"] = np.asarray(dd.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64)
        sub["dd_pol"] = np.asarray(dd.getcol("POLARIZATION_ID"), dtype=np.int64)
    with open_sub("SPECTRAL_WINDOW") as spw:
        nspw = spw.nrows()
        sub["chan_freq"] = [np.asarray(spw.getcell("CHAN_FREQ", i), dtype=np.float64).reshape(-1)
                            for i in range(nspw)]
        for name in ("NUM_CHAN", "TOTAL_BANDWIDTH", "NET_SIDEBAND", "MEAS_FREQ_REF"):
            sub[name] = np.asarray(spw.getcol(name))
    with open_sub("POLARIZATION") as pol:
        sub["corr_type"] = [np.asarray(pol.getcell("CORR_TYPE", i), dtype=np.int64).reshape(-1)
                            for i in range(pol.nrows())]
        sub["num_corr"] = np.asarray(pol.getcol("NUM_CORR"), dtype=np.int64)
    with open_sub("FIELD") as field:
        measinfo = field.getcolkeywords("PHASE_DIR").get("MEASINFO", {})
        if "VarRefCol" in measinfo:
            raise NotImplementedError("exportuvfits: FIELD PHASE_DIR with a variable reference frame")
        _require_unit(field, "PHASE_DIR", "rad")
        if int(field.getcell("NUM_POLY", 0)) != 0:
            raise NotImplementedError("exportuvfits: FIELD PHASE_DIR given as a polynomial (NUM_POLY > 0)")
        sub["phase_dir"] = np.asarray(field.getcell("PHASE_DIR", 0), dtype=np.float64).reshape(-1, 2)[0]
        sub["direction_ref"] = str(measinfo.get("Ref", ""))
        sub["field_name"] = str(field.getcell("NAME", 0))
    with open_sub("OBSERVATION") as obs:
        if obs.nrows() == 0:
            raise UvfitsExportError(f"{table.path}: no OBSERVATION rows")
        sub["observation"] = {"TELESCOPE_NAME": [str(v) for v in obs.getcol("TELESCOPE_NAME")],
                              "OBSERVER": [str(v) for v in obs.getcol("OBSERVER")]}
    sub["telescope"] = sub["observation"]["TELESCOPE_NAME"][0]
    if sub["telescope"] in SPECIAL_TELESCOPES:
        raise NotImplementedError(f"exportuvfits: telescope {sub['telescope']} has its own code paths in casacore")
    with open_sub("ANTENNA") as ant:
        if ant.getcolkeywords("POSITION").get("MEASINFO", {}).get("Ref", "ITRF") != "ITRF":
            raise NotImplementedError("exportuvfits: ANTENNA POSITION not in ITRF")
        for name in ("POSITION", "OFFSET", "DISH_DIAMETER"):
            _require_unit(ant, name, "m")
        sub["antenna"] = {"NAME": [str(v) for v in ant.getcol("NAME")],
                          "STATION": [str(v) for v in ant.getcol("STATION")],
                          "MOUNT": [str(v) for v in ant.getcol("MOUNT")],
                          "POSITION": np.asarray(ant.getcol("POSITION"), dtype=np.float64).reshape(-1, 3),
                          "OFFSET": np.asarray(ant.getcol("OFFSET"), dtype=np.float64).reshape(-1, 3),
                          "DISH_DIAMETER": np.asarray(ant.getcol("DISH_DIAMETER"), dtype=np.float64)}
    with open_sub("FEED") as feed:
        _require_unit(feed, "RECEPTOR_ANGLE", "rad")
        nfeed = feed.nrows()
        sub["feed"] = {"ANTENNA_ID": np.asarray(feed.getcol("ANTENNA_ID"), dtype=np.int64) if nfeed else
                       np.zeros(0, dtype=np.int64),
                       "SPECTRAL_WINDOW_ID": np.asarray(feed.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64)
                       if nfeed else np.zeros(0, dtype=np.int64),
                       "POLARIZATION_TYPE": [[str(v) for v in np.asarray(feed.getcell("POLARIZATION_TYPE", i),
                                                                         dtype=object).reshape(-1)]
                                             for i in range(nfeed)],
                       "RECEPTOR_ANGLE": [np.asarray(feed.getcell("RECEPTOR_ANGLE", i), dtype=np.float64).reshape(-1)
                                          for i in range(nfeed)]}
    sub["rest_freq"] = 0.0
    if isinstance(table.dat.keywords.get("SOURCE"), TableRef):
        with open_sub("SOURCE") as source:
            if source.nrows() > 0 and "REST_FREQUENCY" in source.colnames():
                shapes = source.cell_shapes("REST_FREQUENCY")
                if shapes[0] is not None and int(np.prod(shapes[0])) > 0:
                    sub["rest_freq"] = float(np.asarray(source.getcell("REST_FREQUENCY", 0)).reshape(-1)[0])
    sub["weather"] = None
    if isinstance(table.dat.keywords.get("WEATHER"), TableRef):
        sub["weather"] = _read_weather(open_sub("WEATHER"))
    return sub


def _read_weather(weather) -> Optional[dict]:
    """The WEATHER columns the AIPS WX table takes (``None`` columns are absent; values converted later)."""
    with weather:
        if not weather.colnames():
            return None
        names = weather.colnames()
        out = {"nrow": weather.nrows(), "TIME": None, "INTERVAL": None, "ANTENNA_ID": None}
        for name in ("TIME", "INTERVAL", "ANTENNA_ID", "TEMPERATURE", "PRESSURE", "DEW_POINT", "WIND_SPEED",
                     "WIND_DIRECTION", "H2O", "IONOS_ELECTRON"):
            out[name] = np.asarray(weather.getcol(name)) if name in names and weather.nrows() else None
        if out["PRESSURE"] is not None:
            unit = _unit(weather, "PRESSURE", "hPa")
            if unit not in PRESSURE_UNITS:
                raise NotImplementedError(f"exportuvfits: WEATHER PRESSURE in {unit}")
            out["pressure_to_mbar"] = PRESSURE_UNITS["mbar"] / PRESSURE_UNITS[unit]
        if out["WIND_DIRECTION"] is not None:
            _require_unit(weather, "WIND_DIRECTION", "rad")
        return out


# -- frequency / polarization axes ----------------------------------------------------------------------------

def _near(a: float, b: float, tol: float) -> bool:
    """casacore's relative ``near(a, b, tol)``."""
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(b if a == 0 else a) <= (1 + tol) * np.finfo(np.float64).tiny
    if (a > 0) != (b > 0):
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b))


def _stokes_axis(corr_types: np.ndarray) -> tuple:
    """``(stokes_index, crval3, cdelt3)``: the input correlation of each output one and the STOKES axis."""
    for code in corr_types:
        if int(code) not in STOKES_FITS:
            raise NotImplementedError(f"exportuvfits: correlation type {int(code)} has no FITS Stokes code here")
    stokes = np.array([STOKES_FITS[int(c)] for c in corr_types], dtype=np.int64)
    index = np.argsort(stokes, kind="stable") if stokes[0] >= 0 else np.argsort(-stokes, kind="stable")
    ordered = stokes[index]
    if ordered.size > 2 and np.any(np.diff(ordered) != ordered[1] - ordered[0]):
        raise UvfitsExportError(f"exportuvfits: Stokes {stokes.tolist()} are not representable in FITS")
    cdelt = float(ordered[1] - ordered[0]) if ordered.size > 1 else 1.0
    return index, float(ordered[0]), cdelt


def _spectral_setup(sub: dict, used_ddids: np.ndarray, combinespw: bool) -> dict:
    """The frequency and polarization axes as casacore's ``_writeMain`` derives them.

    Returns ``if_of_ddid`` (IF / frequency-group index of every DATA_DESC_ID, -1 unused), ``spw_ids`` (used
    spectral windows), ``nif``, ``numchan``, ``numcorr``, ``stokes_index``, ``crval3``, ``cdelt3``,
    ``ref_freq``, ``ref_pix``, ``bw0`` (output channel width, signed) and ``meas_freq``.
    """
    ndds = len(sub["dd_spw"])
    if used_ddids.min() < 0 or used_ddids.max() >= ndds:
        raise UvfitsExportError(f"exportuvfits: DATA_DESC_ID outside the DATA_DESCRIPTION table ({ndds} rows)")
    # casacore indexes its spectral-window map with DATA_DESC_ID: only DATA_DESC_ID == SPECTRAL_WINDOW_ID, with
    # every window in use, is exported faithfully.
    if np.any(sub["dd_spw"][used_ddids] != used_ddids) or used_ddids.size != len(sub["chan_freq"]):
        raise NotImplementedError("exportuvfits: DATA_DESC_IDs are not the spectral windows, all of them in use")
    if_of_ddid = np.full(ndds, -1, dtype=np.int64)
    if_of_ddid[used_ddids] = np.arange(used_ddids.size)
    numcorr0 = numchan0 = 0
    f0 = bw0 = chanbw = delta = 0.0
    stokes, meas_freq = None, -1
    for i in used_ddids.tolist():
        s, p = int(sub["dd_spw"][i]), int(sub["dd_pol"][i])
        freqs = sub["chan_freq"][s]
        delta = float(freqs[1] - freqs[0]) if freqs.size > 1 else float(sub["TOTAL_BANDWIDTH"][0])
        if numcorr0 == 0:
            numcorr0, numchan0 = int(sub["num_corr"][p]), int(sub["NUM_CHAN"][s])
            if numcorr0 <= 0 or numchan0 <= 0:
                raise UvfitsExportError("exportuvfits: number of correlations or channels is zero")
            f0, bw0, chanbw = float(freqs[0]), delta, abs(delta)
            stokes, meas_freq = sub["corr_type"][p], int(sub["MEAS_FREQ_REF"][s])
        if int(sub["num_corr"][p]) != numcorr0:
            raise UvfitsExportError("exportuvfits: number of correlations varies in the MS")
        if int(sub["NUM_CHAN"][s]) != numchan0:
            raise UvfitsExportError("exportuvfits: number of channels varies between spectral windows")
        if not np.array_equal(stokes, sub["corr_type"][p]):
            raise UvfitsExportError("exportuvfits: Stokes types vary between spectral windows")
        if not _near(abs(delta), chanbw, 1.0e-5):
            raise UvfitsExportError("exportuvfits: bandwidth varies across spectral windows")
        if numchan0 > 1:
            delta = float(freqs[1] - freqs[0])
            steps = np.diff(freqs[:numchan0])
            if not all(_near(delta, float(step), 1.0e-5) for step in steps):
                raise UvfitsExportError("exportuvfits: channel width varies across the band")
        if int(sub["MEAS_FREQ_REF"][s]) != meas_freq:
            raise UvfitsExportError("exportuvfits: frequency frame varies in the MS")
    ref_pix = 1 + numchan0 // 2
    ref_freq = (f0 + bw0 / 2.0 - delta / 2.0) if ref_pix == 1 else (f0 + (ref_pix - 1) * bw0)
    stokes_index, crval3, cdelt3 = _stokes_axis(stokes)
    if meas_freq not in SPECSYS:
        raise NotImplementedError(f"exportuvfits: MEAS_FREQ_REF {meas_freq}")
    return {"if_of_ddid": if_of_ddid, "spw_ids": used_ddids.copy(), "nif": int(used_ddids.size) if combinespw else 1,
            "nspw": int(used_ddids.size), "numchan": numchan0, "numcorr": numcorr0, "stokes_index": stokes_index,
            "crval3": crval3, "cdelt3": cdelt3, "ref_freq": ref_freq, "ref_pix": ref_pix, "bw0": bw0,
            "meas_freq": meas_freq}


def _antenna_numbers(names: list, telescope: str) -> np.ndarray:
    """AIPS antenna numbers: the names when every one is an integer (VLA: after the EA/VA prefix), else index + 1."""
    numbers = np.arange(1, len(names) + 1, dtype=np.int64)
    parsed = []
    for name in names:
        if "VLA" in telescope:
            digits = re.search(r"[0-9]", name)
            name = name[digits.start():] if digits else name
        if re.fullmatch(r"[-+]?[0-9]+", name) is None:
            return numbers
        parsed.append(int(name))
    return np.asarray(parsed, dtype=np.int64) if parsed else numbers


def _weight_spectrum_defined(sources: list, nchan: int, ncorr: int) -> bool:
    """True when every WEIGHT_SPECTRUM cell is defined with the data shape (then it gives the weights).

    False without the column or with no cell defined (WEIGHT / nchan is used); a partly defined column raises
    NotImplementedError (casacore would mix both row by row).
    """
    if "WEIGHT_SPECTRUM" not in sources[0].colnames():
        return False
    shapes = [shape for part in sources for shape in part.cell_shapes("WEIGHT_SPECTRUM")]
    defined = [shape is not None and tuple(shape) == (nchan, ncorr) for shape in shapes]
    if all(defined):
        return True
    if not any(shape is not None for shape in shapes):
        return False
    raise NotImplementedError("exportuvfits: WEIGHT_SPECTRUM is defined for some rows only")


# -- row layout -----------------------------------------------------------------------------------------------

def _sorted_rows(main: dict, combinespw: bool) -> np.ndarray:
    """Row numbers sorted (stably) by TIME_CENTROID, ANTENNA1, ANTENNA2, FIELD_ID (+ DATA_DESC_ID if not combined)."""
    keys = [main["FIELD_ID"], main["ANTENNA2"], main["ANTENNA1"], main["TIME_CENTROID"]]
    if not combinespw:
        keys.insert(0, main["DATA_DESC_ID"])
    return np.lexsort(keys)


def _tbf_ends(tc: np.ndarray, width: np.ndarray, ant1: np.ndarray, ant2: np.ndarray, nif: int) -> np.ndarray:
    """Last sorted position of each time-baseline group, groups formed greedily from position 0.

    A group starting at ``i`` extends over the following rows of the same baseline whose TIME_CENTROID is at
    most ``tc[i] + 0.2 * INTERVAL[i]``, ``nif`` rows at most (casacore ``get_tbf_end``).
    """
    nrow = tc.size
    positions = np.arange(nrow)
    by_time = np.searchsorted(tc, tc + 0.2 * width, side="right") - 1
    change = np.flatnonzero((ant1[1:] != ant1[:-1]) | (ant2[1:] != ant2[:-1]))
    bounds = np.concatenate([change, [nrow - 1]])
    by_baseline = bounds[np.searchsorted(bounds, positions)]
    end_from = np.maximum(positions, np.minimum(np.minimum(positions + nif - 1, by_time), by_baseline)).tolist()
    ends = []
    position = 0
    while position < nrow:
        end = end_from[position]
        ends.append(end)
        position = end + 1
    return np.asarray(ends, dtype=np.int64)


def _output_rows(main: dict, order: np.ndarray, if_of_row: np.ndarray, nif: int, combinespw: bool,
                 padwithflags: bool) -> tuple:
    """Which input row fills each (output row, IF), and the row giving each output row's random parameters.

    Returns ``(index, first, ngroups)``: ``index`` (nout, nif) input row numbers (-1: padded with flags),
    ``first`` (nout,) input row numbers, ``ngroups`` the number of time-baseline groups (CASA's row count).
    """
    nrow = order.size
    if not combinespw or nif < 2:
        return order[:, None].copy(), order.copy(), nrow
    sorted_if = if_of_row[order]
    if not padwithflags:
        return _rows_without_padding(order, sorted_if, nif)
    ends = _tbf_ends(main["TIME_CENTROID"][order], np.asarray(main["INTERVAL"], dtype=np.float64)[order],
                     main["ANTENNA1"][order], main["ANTENNA2"][order], nif)
    group = np.repeat(np.arange(ends.size), np.diff(np.concatenate([[-1], ends])))
    stream = np.lexsort((sorted_if, group))                      # IF order inside each group (stable)
    stream_if = sorted_if[stream]
    new = np.ones(nrow, dtype=bool)
    new[1:] = (group[1:] != group[:-1]) | (stream_if[1:] <= stream_if[:-1])
    chunk = np.cumsum(new) - 1
    index = np.full((int(chunk[-1]) + 1, nif), -1, dtype=np.int64)
    index[chunk, stream_if] = order[stream]
    # casacore reads on into the next group for the IFs after the last one of a group, without consuming it.
    tail = ends[ends + 1 < nrow]
    leak = tail[stream_if[tail + 1] > stream_if[tail]]
    index[chunk[leak], stream_if[leak + 1]] = order[stream[leak + 1]]
    first = order[np.flatnonzero(new)]
    if index.shape[0] > ends.size:
        logger.warning("exportuvfits: repeated IFs in time-baseline groups; {} rows beyond the {} expected are "
                       "dropped, as CASA does", index.shape[0] - ends.size, ends.size)
        index, first = index[:ends.size], first[:ends.size]
    return index, first, int(ends.size)


def _rows_without_padding(order: np.ndarray, sorted_if: np.ndarray, nif: int) -> tuple:
    """``combinespw`` without ``padwithflags``: consecutive blocks of ``nif`` rows, each holding every IF once."""
    nrow = order.size
    counts = np.bincount(sorted_if, minlength=nif)
    if np.any(counts != counts[0]):
        raise UvfitsExportError(f"exportuvfits: the number of rows per spectral window varies ({counts.tolist()}); "
                                "the spectral windows cannot be combined without padwithflags")
    block = np.arange(nrow) // nif
    stream = np.lexsort((sorted_if, block))
    if np.any(sorted_if[stream].reshape(-1, nif) != np.arange(nif)):
        raise UvfitsExportError("exportuvfits: a DATA_DESC_ID appeared out of the expected order; data with "
                                "varying spectral windows cannot be exported with combinespw")
    return order[stream].reshape(-1, nif), order[::nif].copy(), nrow // nif


def _random_parameters(main: dict, first: np.ndarray, antnums: np.ndarray, combinespw: bool, setup: dict) -> tuple:
    """``(values (nout, pcount) float32, names, comments)`` of the random parameters of every output row."""
    uvw = main["UVW"][first] * (1.0 / SPEED_OF_LIGHT)
    jd = np.asarray(main["TIME_CENTROID"], dtype=np.float64)[first] / SECONDS_PER_DAY + JD_OF_MJD0
    day = np.trunc(jd)
    columns = [uvw[:, 0], uvw[:, 1], uvw[:, 2], day, jd - np.floor(jd)]
    names, comments = ["UU", "VV", "WW", "DATE", "DATE"], ["", "", "", "Day number", "Day fraction"]
    ant1, ant2 = antnums[main["ANTENNA1"][first]], antnums[main["ANTENNA2"][first]]
    array_id = np.asarray(main["ARRAY_ID"], dtype=np.float64)[first]
    if antnums.max() < 256:
        columns.append((ant1 * 256 + ant2).astype(np.float64) + array_id * 0.01)
        names.append("BASELINE")
    else:
        columns += [array_id + 1, ant1.astype(np.float64), ant2.astype(np.float64)]
        names += ["SUBARRAY", "ANTENNA1", "ANTENNA2"]
    if combinespw:
        columns.append(np.ones(first.size))
    else:
        columns.append(1.0 + setup["if_of_ddid"][main["DATA_DESC_ID"][first]])
    names.append("FREQSEL")
    comments += [""] * (len(names) - len(comments))
    return np.stack([np.asarray(c, dtype=np.float64) for c in columns], axis=1).astype(np.float32), names, comments


# -- primary data ---------------------------------------------------------------------------------------------

def _gather(sources: list, views: list, name: str, rows: np.ndarray) -> np.ndarray:
    """Cells ``rows`` (global row numbers) of column ``name`` from the parts of a (concatenated) table."""
    out = None
    offset = 0
    for part, part_views in zip(sources, views):
        count = part.nrows()
        inside = (rows >= offset) & (rows < offset + count)
        if np.any(inside):
            if name not in part_views:
                part_views[name] = part.getcolview(name)
            values = np.asarray(part_views[name][rows[inside] - offset])
            if out is None:
                out = np.empty((rows.size,) + values.shape[1:], dtype=values.dtype)
            out[inside] = values
        offset += count
    return out


def _visibilities(sources: list, views: list, column: str, main: dict, rows: np.ndarray, setup: dict,
                  weight_spectrum: bool) -> np.ndarray:
    """``(n, nchan, ncorr, 3)`` float32 (re, im, weight) of input ``rows``, correlations in FITS order."""
    corr = setup["stokes_index"]
    data = _gather(sources, views, column, rows)[:, :, corr].astype(np.complex64)
    flag = _gather(sources, views, "FLAG", rows)[:, :, corr]
    if weight_spectrum:
        weight = _gather(sources, views, "WEIGHT_SPECTRUM", rows)[:, :, corr].astype(np.float32)
    else:
        scalar = _gather(sources, views, "WEIGHT", rows).astype(np.float32)[:, corr] / np.float32(setup["numchan"])
        weight = np.broadcast_to(scalar[:, None, :], data.shape)
    zero = np.float32(0.0)
    out = np.empty(data.shape + (3,), dtype=np.float32)
    # casacore accumulates into float32 zeros: x + 0.0f turns -0.0 into +0.0, and a flagged weight <= 0 is 0.
    out[..., 0] = data.real + zero
    out[..., 1] = data.imag + zero
    summed = weight + zero
    out[..., 2] = np.where(flag, np.where(summed > zero, -summed, zero), summed)
    row_flag = np.asarray(main["FLAG_ROW"], dtype=bool)[rows]
    out[row_flag, :, :, 2] = -np.abs(out[row_flag, :, :, 2])
    return out


def _write_groups(handle, sources: list, column: str, main: dict, index: np.ndarray, params: np.ndarray,
                  setup: dict, weight_spectrum: bool) -> None:
    """Write the random-groups data (big-endian float32) and its zero padding, ``CHUNK_BYTES`` at a time."""
    nout, nif = index.shape
    nchan, ncorr = setup["numchan"], setup["numcorr"]
    pcount = params.shape[1]
    group_floats = pcount + nif * nchan * ncorr * 3
    step = max(1, CHUNK_BYTES // (4 * group_floats))
    views = [{} for _ in sources]
    pad = np.zeros((nchan, ncorr, 3), dtype=np.float32)
    pad[..., 2] = -0.0                                            # padded IFs: casacore writes -|0|
    written = 0
    for lo in range(0, nout, step):
        hi = min(nout, lo + step)
        rows = index[lo:hi]
        vis = np.broadcast_to(pad, rows.shape + pad.shape).copy()
        present = rows >= 0
        if np.any(present):
            vis[present] = _visibilities(sources, views, column, main, rows[present], setup, weight_spectrum)
        block = np.empty((hi - lo, group_floats), dtype=">f4")
        block[:, :pcount] = params[lo:hi]
        block[:, pcount:] = vis.reshape(hi - lo, -1)
        handle.write(block.tobytes())
        written += block.nbytes
    handle.write(b"\0" * ((-written) % FITS_BLOCK))


# -- headers --------------------------------------------------------------------------------------------------

def _format_value(value) -> str:
    """A keyword value as casacore writes it: logical/integer right-aligned in 20 columns, %20.12E, 'quoted'."""
    if isinstance(value, (bool, np.bool_)):
        return f"{'T' if value else 'F':>20}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):>20}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):20.12E}"
    return "'" + str(value).replace("'", "''").ljust(8) + "'"


def _card(key: str, value=None, comment: str = "") -> str:
    """One 80-character header card (``key`` "" gives a blank card, ``HISTORY`` takes its text as value)."""
    if not key:
        return " " * 80
    if key == "HISTORY":
        return f"HISTORY {value}".ljust(80)[:80]
    text = _format_value(value)
    card = f"{key:<8}= {text:<20} /{comment}" if comment else f"{key:<8}= {text}"
    if len(card) > 80:
        raise UvfitsExportError(f"exportuvfits: header card {key} = {value!r} is longer than 80 characters")
    return card.ljust(80)


def _header_bytes(cards: list) -> bytes:
    """The header cards, END and the blank padding to a full FITS block."""
    text = "".join(cards) + "END".ljust(80)
    text += " " * ((-len(text)) % FITS_BLOCK)
    return text.encode("ascii")


def _fits_date(seconds: float) -> str:
    """MJD seconds as casacore's FITS date ``YYYY-MM-DDThh:mm:ss.ssssss`` (rounded to the microsecond)."""
    days = float(seconds) / SECONDS_PER_DAY
    day = math.floor(days)
    micro = round((days - day) * SECONDS_PER_DAY * 1e6)
    if micro >= SECONDS_PER_DAY * 1e6:
        day, micro = day + 1, micro - int(SECONDS_PER_DAY * 1e6)
    import erfa
    year, month, mday, _ = erfa.jd2cal(JD_OF_MJD0, float(day))
    second, micro = divmod(int(micro), 1_000_000)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"{int(year):04d}-{int(month):02d}-{int(mday):02d}T{hour:02d}:{minute:02d}:{second:02d}.{micro:06d}"


def _primary_cards(main: dict, sub: dict, setup: dict, names: list, comments: list, gcount: int, column: str,
                   table) -> list:
    """The primary header cards (random groups), in casacore's order and layout."""
    nif, nchan, ncorr = setup["nif"], setup["numchan"], setup["numcorr"]
    ra, dec = sub["phase_dir"] * (180.0 / math.pi)
    if ra < 0:
        ra += 360.0
    if "2000" in sub["direction_ref"]:
        epoch = 2000.0
    elif "1950" in sub["direction_ref"]:
        epoch = 1950.0
    else:
        logger.warning("exportuvfits: cannot deduce the epoch of {} directions, assuming J2000", sub["direction_ref"])
        epoch = 2000.0
    bunit = table.getcolkeywords(column).get("QuantumUnit")
    bunit = bunit.upper() if isinstance(bunit, str) else "UNCALIB"
    cards = [_card("SIMPLE", True, "Standard FITS format"), _card("BITPIX", -32, "Floating point values"),
             _card("NAXIS", 7), _card("NAXIS1", 0, "Random groups, NOT image")]
    cards += [_card(f"NAXIS{i}", n) for i, n in zip(range(2, 8), (3, ncorr, nchan, nif, 1, 1))]
    cards += [_card("EXTEND", True, "Tables may follow"), _card("BLOCKED", True, "File may be blocked"),
              _card("GROUPS", True, "Random Group UV data"), _card("PCOUNT", len(names), "Number of random parameters"),
              _card("GCOUNT", gcount, "Number of groups (rows) in the file"), _card(""),
              _card("EPOCH", epoch), _card("BSCALE", 1.0), _card("BZERO", 0.0), _card("BUNIT", bunit)]
    axes = (("COMPLEX", 1.0, 1.0, 1.0), ("STOKES", setup["crval3"], setup["cdelt3"], 1.0),
            ("FREQ", setup["ref_freq"], setup["bw0"], float(setup["ref_pix"])), ("IF", 1.0, 1.0, 1.0),
            ("RA", float(ra), 1.0, 1.0), ("DEC", float(dec), 1.0, 1.0))
    for axis, (ctype, crval, cdelt, crpix) in enumerate(axes, start=2):
        cards += [_card(f"CTYPE{axis}", ctype), _card(f"CRVAL{axis}", float(crval)), _card(f"CDELT{axis}", float(cdelt)),
                  _card(f"CRPIX{axis}", float(crpix)), _card(f"CROTA{axis}", 0.0)]
    for i, (name, comment) in enumerate(zip(names, comments), start=1):
        cards += [_card(f"PTYPE{i}", name, comment), _card(f"PSCAL{i}", 1.0), _card(f"PZERO{i}", 0.0)]
    observation = sub["observation"]
    cards += [_card("OBJECT", sub["field_name"]), _card("DATE-OBS", _fits_date(main["TIME"][0])),
              _card("TELESCOP", observation["TELESCOPE_NAME"][0]), _card("INSTRUME", observation["TELESCOPE_NAME"][0]),
              _card("OBSERVER", observation["OBSERVER"][0]), _card("SORTORD", "TB"),
              _card("SPECSYS", SPECSYS[setup["meas_freq"]])]
    if sub["rest_freq"] > 0.0:
        cards.append(_card("RESTFREQ", sub["rest_freq"]))
    cards += [_card("HISTORY", "AIPS WTSCAL = 1.0"), _card(""), _card("ORIGIN", "casacore "), _card("")]
    return cards


# -- binary tables --------------------------------------------------------------------------------------------

def _binary_table(keywords: list, columns: list, nrow: int) -> bytes:
    """A BINTABLE extension in casacore's layout.

    ``keywords`` are cards; ``columns`` are ``(name, tform, unit, values)`` with ``values`` a numpy array of
    ``nrow`` rows already in the FITS (big-endian) type, or a list of str for ``nA`` columns (NUL padded).
    """
    pieces = []
    for name, tform, _, values in columns:
        if tform.endswith("A"):
            width = int(tform[:-1])
            raw = np.array([str(v).encode("ascii")[:width].ljust(width, b"\0") for v in values], dtype=f"S{width}")
            pieces.append(raw.view(np.uint8).reshape(nrow, width))
        else:
            array = np.ascontiguousarray(values)
            width = array.nbytes // nrow if nrow else 0
            pieces.append(np.frombuffer(array.tobytes(), dtype=np.uint8).reshape(nrow, width))
    rows = np.concatenate(pieces, axis=1) if pieces else np.zeros((nrow, 0), dtype=np.uint8)
    cards = [_card("XTENSION", "BINTABLE", "Binary Table Extension"), _card("BITPIX", 8, "Character Information"),
             _card("NAXIS", 2, "Two-dimensional table"), _card("NAXIS1", rows.shape[1], "Number of bytes per row"),
             _card("NAXIS2", nrow, "Number of rows"), _card("PCOUNT", 0, "No random parameters"),
             _card("GCOUNT", 1, "Only one group"), _card("TFIELDS", len(columns), "Number of columns"),
             _card(""), _card("")] + keywords + [_card("")]
    for i, (name, tform, unit, _) in enumerate(columns, start=1):
        cards += [_card(f"TTYPE{i}", name), _card(f"TFORM{i}", tform)]
        if unit:
            cards.append(_card(f"TUNIT{i}", unit))
    cards.append(_card(""))
    data = rows.tobytes()
    return _header_bytes(cards) + data + b"\0" * ((-len(data)) % FITS_BLOCK)


def _fq_table(sub: dict, setup: dict, combinespw: bool) -> bytes:
    """The AIPS FQ table: IF offsets from the reference frequency, channel widths, bandwidths, sidebands."""
    spws = setup["spw_ids"].tolist()
    centre = setup["numchan"] // 2                               # 0-based reference channel of every window
    offsets = [float(sub["chan_freq"][s][centre]) - setup["ref_freq"] for s in spws]
    widths = [setup["bw0"] if sub["chan_freq"][s].size > 1 else float(sub["TOTAL_BANDWIDTH"][s]) for s in spws]
    totals = [float(sub["TOTAL_BANDWIDTH"][s]) for s in spws]
    sidebands = [int(sub["NET_SIDEBAND"][s]) for s in spws]
    if combinespw:
        nrow, nif = 1, len(spws)
        frqsel = np.array([1], dtype=">i4")
        shape = (1, nif)
    else:
        nrow, nif = len(spws), 1
        frqsel = (1 + setup["if_of_ddid"][setup["spw_ids"]]).astype(">i4")
        shape = (nrow, 1)
    columns = [("FRQSEL", "1J", "", frqsel),
               ("IF FREQ", f"{nif}D", "HZ", np.asarray(offsets, dtype=">f8").reshape(shape)),
               ("CH WIDTH", f"{nif}E", "HZ", np.asarray(widths, dtype=">f4").reshape(shape)),
               ("TOTAL BANDWIDTH", f"{nif}E", "HZ", np.asarray(totals, dtype=">f4").reshape(shape)),
               ("SIDEBAND", f"{nif}J", "", np.asarray(sidebands, dtype=">i4").reshape(shape))]
    keywords = [_card("EXTNAME", "AIPS FQ"), _card("EXTVER", 1), _card("NO_IF", nif)]
    return _binary_table(keywords, columns, nrow)


def _earth_orientation(time_utc: float) -> dict:
    """GSTIA0, DEGPDY, POLARX, POLARY, UT1UTC, IATUTC for the first TIME (MJD seconds, UTC).

    From the IERS-A table bundled with astropy (no download): GMST (IAU 1982) at 0h UTC of the IAT day and the
    next, polar motion at 0h UTC of the UTC day, UT1 - UTC at ``time_utc``; TAI - UTC from ``erfa.dat``.
    Outside the table UT1 - UTC and the polar motion are 0 (logged).
    """
    import erfa
    from astropy.time import Time
    from astropy.utils import iers

    with iers.conf.set_temp("auto_download", False):
        table = iers.IERS_A.open(iers.IERS_A_FILE)

    def ut1_utc(mjd: float) -> float:
        value, status = table.ut1_utc(Time(mjd, format="mjd", scale="utc"), return_status=True)
        if status < 0:
            logger.warning("exportuvfits: MJD {} outside the bundled IERS table, UT1 - UTC = 0", mjd)
            return 0.0
        return float(value.to_value("s"))

    days = float(time_utc) / SECONDS_PER_DAY
    year, month, mday, fraction = erfa.jd2cal(JD_OF_MJD0, days)
    iat_utc = float(erfa.dat(year, month, mday, fraction))
    utc_day = math.floor(days)
    iat_day = math.floor((float(time_utc) + iat_utc) / SECONDS_PER_DAY)
    gst = [erfa.gmst82(JD_OF_MJD0, d + ut1_utc(d) / SECONDS_PER_DAY) for d in (iat_day, iat_day + 1)]
    gstdeg = math.degrees(gst[0])
    degpdy = 360.0 + (math.degrees(gst[1]) - gstdeg) % 360.0
    pm_x, pm_y, status = table.pm_xy(Time(float(utc_day), format="mjd", scale="utc"), return_status=True)
    if status < 0:
        logger.warning("exportuvfits: MJD {} outside the bundled IERS table, polar motion = 0", utc_day)
        polar = (0.0, 0.0)
    else:
        polar = (float(pm_x.to_value("rad")) * POLAR_RADIUS, float(pm_y.to_value("rad")) * POLAR_RADIUS)
    return {"GSTIA0": gstdeg, "DEGPDY": degpdy, "POLARX": polar[0], "POLARY": polar[1], "UT1UTC": ut1_utc(days),
            "IATUTC": iat_utc}


def _mount_code(mount: str) -> int:
    """AIPS MNTSTA code of an ANTENNA MOUNT string (9 when unknown)."""
    upper = mount.upper()
    for text, code in MOUNT_CODES:
        if text in upper:
            return code
    return 9


def _feed_polarizations(sub: dict, nant: int) -> list:
    """Per antenna ``(POLTYA, POLAA, POLTYB, POLAB)`` from its first FEED row (window -1 or any listed window).

    As in casacore, the types are reset for every antenna but the angles are not: an antenna without a FEED
    row (or with one receptor) keeps the angle of the antenna before it.
    """
    feed = sub["feed"]
    nspw = len(sub["chan_freq"])
    out = []
    polaa = polab = 0.0
    for antenna in range(nant):
        poltya = poltyb = " "
        rows = [row for row, (ant, spw) in enumerate(zip(feed["ANTENNA_ID"].tolist(),
                                                         feed["SPECTRAL_WINDOW_ID"].tolist()))
                if ant == antenna and (spw == -1 or 0 <= spw < nspw)]
        if rows:
            types, angles = feed["POLARIZATION_TYPE"][rows[0]], feed["RECEPTOR_ANGLE"][rows[0]] / (math.pi / 180.0)
            if len(types) >= 1:
                poltya, polaa = types[0], float(angles[0])
            if len(types) >= 2:
                poltyb, polab = types[1], float(angles[1])
        else:
            logger.warning("exportuvfits: no polarization types for antenna {}", antenna)
        out.append((poltya, polaa, poltyb, polab))
    return out


def _an_table(main: dict, sub: dict, setup: dict, antnums: np.ndarray, arraynum: int) -> bytes:
    """The AIPS AN table of OBSERVATION row ``arraynum``: ITRF positions, mounts, offsets, feeds."""
    antenna = sub["antenna"]
    nant = len(antenna["NAME"])
    names = antenna["NAME"]
    if (len(set(names)) == 1) and len(set(antenna["STATION"])) > 1:
        names = antenna["STATION"]
    eop = _earth_orientation(float(main["TIME"][0]))
    date = _fits_date(main["TIME"][0])
    telescope = sub["observation"]["TELESCOPE_NAME"][arraynum]
    keywords = [_card("EXTNAME", "AIPS AN"), _card("EXTVER", arraynum + 1), _card("ARRAYX", 0), _card("ARRAYY", 0),
                _card("ARRAYZ", 0), _card("GSTIA0", eop["GSTIA0"]), _card("DEGPDY", eop["DEGPDY"]),
                _card("FREQ", setup["ref_freq"]), _card("RDATE", date), _card("POLARX", eop["POLARX"]),
                _card("POLARY", eop["POLARY"]), _card("UT1UTC", eop["UT1UTC"]), _card("IATUTC", eop["IATUTC"]),
                _card("TIMSYS", "UTC"), _card("ARRNAM", telescope), _card("NUMORB", 0), _card("NOPCAL", 0),
                _card("POLTYPE", "        "), _card("XYZHAND", "RIGHT"), _card("FRAME", "ITRF")]
    feeds = _feed_polarizations(sub, nant)
    columns = [("ANNAME", "8A", "", names),
               ("STABXYZ", "3D", "METERS", antenna["POSITION"].astype(">f8")),
               ("ORBPARM", "0D", "", np.zeros((nant, 0), dtype=">f8")),
               ("NOSTA", "1J", "", antnums.astype(">i4")),
               ("MNTSTA", "1J", "", np.array([_mount_code(m) for m in antenna["MOUNT"]], dtype=">i4")),
               ("STAXOF", "1E", "METERS", antenna["OFFSET"][:, 0].astype(">f4")),
               ("POLTYA", "1A", "", [f[0] for f in feeds]),
               ("POLAA", "1E", "DEGREES", np.array([f[1] for f in feeds], dtype=">f4")),
               ("POLCALA", "1E", "", np.zeros(nant, dtype=">f4")),
               ("POLTYB", "1A", "", [f[2] for f in feeds]),
               ("POLAB", "1E", "DEGREES", np.array([f[3] for f in feeds], dtype=">f4")),
               ("POLCALB", "1E", "", np.zeros(nant, dtype=">f4")),
               ("DIAMETER", "1E", "", antenna["DISH_DIAMETER"].astype(">f4"))]
    return _binary_table(keywords, columns, nant)


def _wx_table(main: dict, weather: dict, antnums: np.ndarray) -> bytes:
    """The AIPS WX table (weather per antenna and time; absent WEATHER columns give zeros)."""
    nrow = weather["nrow"]
    if nrow == 0:
        logger.warning("exportuvfits: WEATHER table has no rows, no AIPS WX table written")
        return b""
    ref_time = math.floor(float(main["TIME"][0]) / SECONDS_PER_DAY) * SECONDS_PER_DAY

    def optional(name, convert):
        values = weather.get(name)
        if values is None:
            return np.zeros(nrow, dtype=">f4")
        return convert(np.asarray(values, dtype=np.float64)).astype(">f4")

    antenna_id = np.asarray(weather["ANTENNA_ID"], dtype=np.int64)
    antenna_no = np.where(antenna_id == -1, 0, antnums[np.clip(antenna_id, 0, None)])
    keywords = [_card("EXTNAME", "AIPS WX"), _card("EXTVER", 1), _card("OBSCODE", ""),
                _card("RDATE", _fits_date(main["TIME"][0])), _card("TABREV", 3)]
    columns = [("TIME", "1D", "DAYS", ((np.asarray(weather["TIME"], dtype=np.float64) - ref_time)
                                        / SECONDS_PER_DAY).astype(">f8")),
               ("TIME_INTERVAL", "1E", "DAYS", (np.asarray(weather["INTERVAL"], dtype=np.float64)
                                                / SECONDS_PER_DAY).astype(">f4")),
               ("ANTENNA_NO", "1J", "", antenna_no.astype(">i4")),
               ("SUBARRAY", "1J", "", np.zeros(nrow, dtype=">i4")),
               ("TEMPERATURE", "1E", "CENTIGRADE", optional("TEMPERATURE", lambda v: v - 273.15)),
               ("PRESSURE", "1E", "MILLIBAR", optional("PRESSURE", lambda v: v / weather["pressure_to_mbar"])),
               ("DEWPOINT", "1E", "CENTIGRADE", optional("DEW_POINT", lambda v: v - 273.15)),
               ("WIND_VELOCITY", "1E", "M/SEC", optional("WIND_SPEED", lambda v: v)),
               ("WIND_DIRECTION", "1E", "DEGREES", optional("WIND_DIRECTION", lambda v: v / (math.pi / 180.0))),
               ("WVR_H2O", "1E", "m-2", optional("H2O", lambda v: v)),
               ("IONOS_ELECTRON", "1E", "m-2", optional("IONOS_ELECTRON", lambda v: v))]
    return _binary_table(keywords, columns, nrow)
