"""Import FITS-IDI files into a Multi-MS without CASA: what ``importfitsidi`` followed by ``partition`` writes.

The visibilities are never decoded row by row: every UV_DATA table is memory-mapped as a structured numpy array
(big-endian, as FITS stores it), the small columns (time, baseline, source, band) are read once to plan the
output, and parallel worker processes each write one part of the Multi-MS (:mod:`vlbipy.casacore_format`), taking
the FLUX cells of their rows straight from the memory map. The rules are casacore's FITSIDItoMS1 filler
(``fillMSMainTable``), CASA's ``importfitsidi`` task (several files concatenated, scan re-indexing by time gaps
per field, one observation id) and ``partition`` (whole (scan, subband) chunks per part; WEIGHT recomputed from
SIGMA):

* one MS row per UV_DATA row and band (``DATA_DESC_ID = (FREQID - 1) * nband + band``); ANTENNA1 <= ANTENNA2 — a
  baseline given the other way round is swapped, its UVW negated, its visibility taken as is and its cross hands
  exchanged, every other visibility is conjugated;
* TIME = DATE + TIME of the row in MJD seconds, TIME_CENTROID = TIME + INTERVAL / 2, INTERVAL = EXPOSURE = INTTIM,
  UVW = (UU, VV, WW) x c;
* a FITS weight <= 0 flags the sample and keeps its magnitude; WEIGHT_SPECTRUM = weight x VIS_SCAL x INTTIM x
  channel width (x 2 for cross-correlations), SIGMA_SPECTRUM = 1 / sqrt(WEIGHT_SPECTRUM) (1 when that is 0),
  SIGMA = the median (casacore's: the lower middle value) of SIGMA_SPECTRUM over channels, WEIGHT = 1 / SIGMA^2,
  FLAG_ROW = every sample flagged;
* SCAN_NUMBER restarts whenever a field's integrations are more than ``scan_gap`` seconds apart.

DiFX and VLBA correlator output (``WEIGHTYP = CORRELAT``) gets the filler's digital corrections
(:func:`convert_block`): the van-Vleck correction of the autocorrelation spectra (``rho_2``/``rho_4``
per the antennas' NO_LEVELS, skipped when the primary HDU sets VANVLECK), the amplitude factor
``bfactc`` on every other visibility, and the ``vis / (VIS_SCAL x weight)`` rescaling that turns the
DiFX correlation coefficients into flux densities. DiFX writes the weights in the WEIGHT random
parameter (one value per band and correlation, no channel axis) instead of a third COMPLEX element.
"""
from __future__ import annotations

import os
import shutil
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .workers import run_jobs

logger = get_logger()

#: Speed of light [m/s] (casacore C::c).
SPEED_OF_LIGHT = 299792458.0
#: JD of MJD 0.
JD_MJD0 = 2400000.5
#: Output rows written at a time inside a part job.
BLOCK_ROWS = 20_000
#: FITS binary-table TFORM letters -> big-endian numpy types.
_TFORM_TYPES = {"D": ">f8", "E": ">f4", "J": ">i4", "I": ">i2", "K": ">i8", "B": "u1", "L": "u1", "A": "S1"}
#: readme CASA's partition writes into the table.info of each part.
PART_README = ("This is a MeasurementSet Table holding measurements from a Telescope\n"
               "This is a measurement set Table holding astronomical observations\n")


# ---------------------------------------------------------------------------
# UV_DATA as a memory map
# ---------------------------------------------------------------------------
def _uv_dtype(header) -> tuple[np.dtype, dict]:
    """The structured big-endian dtype of a UV_DATA row and the FLUX cell shape (C order)."""
    names, formats = [], []
    shapes = {}
    for k in range(1, int(header["TFIELDS"]) + 1):
        name = str(header[f"TTYPE{k}"]).strip()
        form = str(header[f"TFORM{k}"]).strip()
        count = int(form[:-1]) if len(form) > 1 else 1
        kind = _TFORM_TYPES.get(form[-1])
        if kind is None:
            raise NotImplementedError(f"UV_DATA column {name} has unsupported TFORM {form}")
        dims = header.get(f"TDIM{k}")
        if dims:
            shape = tuple(int(n) for n in str(dims).strip("() ").split(","))[::-1]
        elif count == 0:
            shape = (0,)                                # a zero-count column (e.g. DiFX's GATEID) is 0 bytes
        else:
            shape = (count,) if count > 1 else ()
        if kind == "S1":
            kind, shape = f"S{count}", ()
        names.append(name)
        formats.append((kind, shape) if shape else kind)
        shapes[name] = shape
    dtype = np.dtype({"names": names, "formats": formats})
    if dtype.itemsize != int(header["NAXIS1"]):
        raise NotImplementedError(f"UV_DATA row size {dtype.itemsize} != NAXIS1 {header['NAXIS1']}")
    return dtype, shapes


def uv_tables(idi_files: list) -> list[dict]:
    """Every UV_DATA table of the files, in order: ``path``, ``offset`` (bytes), ``nrow``, ``dtype``, ``header``."""
    from astropy.io import fits
    tables = []
    for path in idi_files:
        with fits.open(path, memmap=True) as hdus:
            for hdu in hdus:
                if hdu.name != "UV_DATA":
                    continue
                dtype, _ = _uv_dtype(hdu.header)
                tables.append({"path": str(path), "offset": int(hdu.fileinfo()["datLoc"]),
                               "nrow": int(hdu.header["NAXIS2"]), "dtype": dtype, "header": hdu.header.copy()})
    if not tables:
        raise ValueError("no UV_DATA table in the FITS-IDI files")
    return tables


def _rows(table: dict) -> np.ndarray:
    """The UV_DATA rows of ``table`` as a read-only structured memory map."""
    return np.memmap(table["path"], dtype=table["dtype"], mode="r", offset=table["offset"], shape=(table["nrow"],))


def _column(rows: np.ndarray, *fragments: str) -> str:
    """The first column whose name contains one of ``fragments`` (``UU---SIN`` for ``UU``)."""
    for fragment in fragments:
        for name in rows.dtype.names:
            if name == fragment or fragment in name:
                return name
    raise ValueError(f"UV_DATA has no {'/'.join(fragments)} column")


# ---------------------------------------------------------------------------
# Plan: the small columns of every row
# ---------------------------------------------------------------------------
def read_index(tables: list[dict], setup: dict) -> dict:
    """Per UV_DATA row (all tables concatenated): ``table``, ``row``, ``time`` [MJD s], ``interval``, ``ant1``,
    ``ant2`` (MS ids, ANTENNA1 <= ANTENNA2), ``swapped``, ``field``, ``freqid`` (0-based), ``array``."""
    pieces = {key: [] for key in ("table", "row", "time", "interval", "ant1", "ant2", "swapped", "field", "freqid",
                                  "array")}
    numbers = np.array(sorted(setup["antenna_ids"]), dtype=np.int64)
    ids = np.array([setup["antenna_ids"][n] for n in numbers], dtype=np.int64)
    for index, table in enumerate(tables):
        rows = _rows(table)
        time = (np.asarray(rows["DATE"], dtype=np.float64) - JD_MJD0)
        if "TIME" in rows.dtype.names:
            time = time + np.asarray(rows["TIME"], dtype=np.float64)
        baseline = np.asarray(rows["BASELINE"]).astype(np.float64)
        raw1 = (baseline.astype(np.int64)) // 256
        raw2 = baseline.astype(np.int64) - raw1 * 256
        position1, position2 = np.searchsorted(numbers, raw1), np.searchsorted(numbers, raw2)
        position1, position2 = np.minimum(position1, len(numbers) - 1), np.minimum(position2, len(numbers) - 1)
        if np.any(numbers[position1] != raw1) or np.any(numbers[position2] != raw2):
            raise ValueError("UV_DATA refers to an ANTENNA_NO missing from the ANTENNA table")
        ant1, ant2 = ids[position1], ids[position2]
        swapped = ant1 > ant2
        source = _column(rows, "SOURCE_ID", "SOURCE")
        field = np.asarray(rows[source]).astype(np.int64) - 1
        freqid = (np.asarray(rows["FREQID"]).astype(np.int64) - 1) if "FREQID" in rows.dtype.names else np.zeros(
            len(rows), dtype=np.int64)
        if "INTTIM" not in rows.dtype.names:
            raise NotImplementedError("UV_DATA without INTTIM")
        pieces["table"].append(np.full(len(rows), index, dtype=np.int64))
        pieces["row"].append(np.arange(len(rows), dtype=np.int64))
        pieces["time"].append(time * 86400.0)
        pieces["interval"].append(np.asarray(rows["INTTIM"], dtype=np.float32))
        pieces["ant1"].append(np.where(swapped, ant2, ant1))
        pieces["ant2"].append(np.where(swapped, ant1, ant2))
        pieces["swapped"].append(swapped)
        pieces["field"].append(field)
        pieces["freqid"].append(freqid)
        pieces["array"].append((100.0 * (baseline - np.floor(baseline) + 0.001)).astype(np.int64))
    return {key: np.concatenate(values) for key, values in pieces.items()}


def scan_numbers(time: np.ndarray, field: np.ndarray, array: np.ndarray, gap: float) -> np.ndarray:
    """CASA importfitsidi's scan re-indexing: in time order, a new scan starts when a field's integration is more
    than ``gap`` seconds after that field's previous one, or the array changes; every row of a field takes the
    field's current scan number."""
    order = np.argsort(time, kind="stable")
    scans = np.zeros(time.size, dtype=np.int32)
    current, last_time, last_scan = 0, {}, {}
    previous_array = int(array[order[0]]) if time.size else 0
    for i in order:
        f, t, a = int(field[i]), float(time[i]), int(array[i])
        if t - last_time.get(f, 0.0) > gap or a != previous_array:
            current += 1
            last_scan[f] = current
        scans[i] = last_scan.get(f, 0)
        last_time[f] = t
        previous_array = a
    return scans


def plan_parts(index: dict, scans: np.ndarray, nband: int, nparts: int, nvis_per_row: int = 1) -> list[list[dict]]:
    """Cut the output into parts of whole (scan, subband) chunks exactly as CASA's ``partition(separationaxis=
    'auto')`` does (casatasks ``partitionhelper.getPartitionMap`` with axes field, spw and scan).

    Pairs are taken by decreasing visibility count (then decreasing scan and subband *as strings*, CASA's
    keys); each goes to the part with the largest summed gap below the per-scan, per-subband and per-field
    reference levels (ties: the part with fewest visibilities so far). A pair's visibilities are, as CASA counts
    them, the scan's rows divided evenly over its subbands times ``nvis_per_row`` (channels x correlations).
    At most ``nparts`` parts (fewer when there are fewer pairs); each part's chunks in (scan, subband) order,
    which is the order partition writes rows in (it sorts by scan, subband and time). A chunk is ``{"rows",
    "band", "spw", "scan"}`` with ``rows`` indexing ``index``, in input order.
    """
    chunks: dict = {}
    nvis: dict = {}                              # {scan: {ddi: visibilities}} with string keys, as CASA's
    fields: dict = {}
    for scan in np.unique(scans):
        rows = np.flatnonzero(scans == scan)
        freqids = np.unique(index["freqid"][rows])
        ddis = [int(f) * nband + band for f in freqids for band in range(nband)]
        per_ddi = (rows.size * nband) / len(ddis) * nvis_per_row       # nRow / len(DDIds) * nchan * ncorr
        key = str(int(scan))
        nvis[key] = {str(ddi): per_ddi for ddi in sorted(ddis)}
        fields[key] = str(int(index["field"][rows[np.argmin(index["time"][rows])]]))
        for freqid in freqids:
            selected = rows[index["freqid"][rows] == freqid]
            for band in range(nband):
                spw = int(freqid) * nband + band
                chunks[(key, str(spw))] = {"rows": selected, "band": band, "spw": spw, "scan": int(scan)}
    pairs = [(scan, ddi) for scan in sorted(nvis) for ddi in nvis[scan]]
    nparts = min(int(nparts), len(pairs))
    per_scan = {scan: sum(nvis[scan].values()) for scan in nvis}
    per_ddi: dict = {}
    per_field: dict = {}
    for scan, ddi in pairs:
        per_ddi[ddi] = per_ddi.get(ddi, 0.0) + nvis[scan][ddi]
        per_field[fields[scan]] = per_field.get(fields[scan], 0.0) + nvis[scan][ddi]
    order = np.lexsort((np.array([p[1] for p in pairs]), np.array([p[0] for p in pairs]),
                        np.array([nvis[s][d] for s, d in pairs])))[::-1]
    scan_load = {scan: np.zeros(nparts) for scan in nvis}
    ddi_load = {ddi: np.zeros(nparts) for ddi in per_ddi}
    field_load = {field: np.zeros(nparts) for field in per_field}
    part_load = np.zeros(nparts)
    parts: list[list[dict]] = [[] for _ in range(nparts)]
    for k in order:
        scan, ddi = pairs[k]
        field, count = fields[scan], nvis[scan][ddi]
        gap = np.zeros(nparts)
        for total, load in ((per_scan[scan], scan_load[scan]), (per_ddi[ddi], ddi_load[ddi]),
                            (per_field[field], field_load[field])):
            gap = gap + max(total // nparts, load.max()) - load
        candidates = np.where(gap == gap.max())[0]
        target = int(candidates[np.argmin(part_load[candidates])]) if len(candidates) > 1 else int(candidates[0])
        parts[target].append(chunks[(scan, ddi)])
        part_load[target] += count
        scan_load[scan][target] += count
        ddi_load[ddi][target] += count
        field_load[field][target] += count
    return [sorted(p, key=lambda c: (c["scan"], c["spw"])) for p in parts if p]


# ---------------------------------------------------------------------------
# Writing one part
# ---------------------------------------------------------------------------
def _main_columns(ncorr: int, nchan: int) -> list:
    """The main-table columns of a part, described as CASA's partition writes them."""
    from ..casacore_format.writer import array_column, scalar_column
    seconds = {"QuantumUnits": ["s"]}
    epoch = {"QuantumUnits": ["s"], "MEASINFO": {"type": "epoch", "Ref": "UTC"}}
    columns = [
        array_column("UVW", "double", shape=(3,), direct=True, comment="Vector with uvw coordinates (in meters)",
                     keywords={"QuantumUnits": ["m", "m", "m"], "MEASINFO": {"type": "uvw", "Ref": "ITRF"}}),
        array_column("FLAG", "boolean", ndim=2, comment="The data flags, array of bools with same shape as data"),
        array_column("FLAG_CATEGORY", "boolean", ndim=3, comment="The flag category, NUM_CAT flags for each datum",
                     keywords={"CATEGORY": ["FLAG_CMD", "ORIGINAL", "USER"]}),
        array_column("WEIGHT", "float", ndim=1, comment="Weight for each polarization spectrum"),
        array_column("SIGMA", "float", ndim=1, comment="Estimated rms noise for channel with unity bandpass response"),
        scalar_column("ANTENNA1", "int", comment="ID of first antenna in interferometer"),
        scalar_column("ANTENNA2", "int", comment="ID of second antenna in interferometer"),
        scalar_column("ARRAY_ID", "int", comment="ID of array or subarray"),
        scalar_column("DATA_DESC_ID", "int", comment="The data description table index"),
        scalar_column("EXPOSURE", "double", comment="The effective integration time", keywords=seconds),
        scalar_column("FEED1", "int", comment="The feed index for ANTENNA1"),
        scalar_column("FEED2", "int", comment="The feed index for ANTENNA2"),
        scalar_column("FIELD_ID", "int", comment="Unique id for this pointing"),
        scalar_column("FLAG_ROW", "boolean", comment="Row flag - flag all data in this row if True"),
        scalar_column("INTERVAL", "double", comment="The sampling interval", keywords=seconds),
        scalar_column("OBSERVATION_ID", "int", comment="ID for this observation, index in OBSERVATION table"),
        scalar_column("PROCESSOR_ID", "int", comment="Id for backend processor, index in PROCESSOR table"),
        scalar_column("SCAN_NUMBER", "int", comment="Sequential scan number from on-line system"),
        scalar_column("STATE_ID", "int", comment="ID for this observing state"),
        scalar_column("TIME", "double", comment="Modified Julian Day", keywords=epoch),
        scalar_column("TIME_CENTROID", "double", comment="Modified Julian Day", keywords=epoch),
        array_column("DATA", "complex", ndim=2, comment="The data column"),
        array_column("WEIGHT_SPECTRUM", "float", ndim=2, comment="Weight for each data point"),
        array_column("SIGMA_SPECTRUM", "float", ndim=2, comment="Estimated rms noise for each data point")]
    return columns


def _rho_4(r: np.ndarray) -> np.ndarray:
    """Van Vleck relation for 4-level (2-bit) sampling: Fred Schwab's rational approximation
    (VLBA Correlator Memo 75), as casacore's ``rho_4``."""
    rr = r * r
    num = 1.1329552 - 3.1056902 * rr + 2.9296994 * rr * rr - 0.90122460 * rr * rr * rr
    den = 1 - 2.7056559 * rr + 2.5012473 * rr * rr - 0.73985978 * rr * rr * rr
    return (num / den) * r


def _redft00(n: int) -> np.ndarray:
    """The REDFT00 (unshifted DCT-I) transform matrix for an ``n``-point series: the forward matrix F of
    FFTW's ``plan_redft00``, ``y[k] = x[0] + (-1)^k x[n-1] + 2 sum_j x[j] cos(pi j k / (n-1))``.
    The same matrix applied to the output inverts it up to a scale of ``2 (n-1)``."""
    k, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    t = 2.0 * np.cos(np.pi * j * k / (n - 1))
    t[:, 0] = 1.0
    t[:, n - 1] = (-1.0) ** np.arange(n)
    return t


def _digital_correction(data: np.ndarray, wmag: np.ndarray, ant1: np.ndarray, ant2: np.ndarray,
                        cross: np.ndarray, setup: dict) -> np.ndarray:
    """The DiFX/VLBA digital corrections of casacore's filler, applied in place to ``data``.

    Cross-correlations (and every row of an antenna pair without a rho model, and everything when the
    file's VANVLECK keyword is set) are scaled by ``bfactc = gamma^2 / (A Rm alfa H)`` with the
    constants of the baseline's 2-bit/4-bit combination (VLBA Scientific Memo 12). Autocorrelation
    spectra in the parallel hands get the van Vleck correction instead: cosine transform to the lag
    domain, ``rho`` of the measured lag correlation, transform back (real-valued, as the filler writes
    it). ``wmag`` is the magnitude of the raw FITS weight per (row, chan, corr) — for VLBA it enters
    the saturation factor.
    """
    nchan, ncorr = data.shape[1], data.shape[2]
    digi = setup.get("digi_levels") or {}
    lev1 = np.fromiter((digi.get(int(a), 0) for a in ant1), np.int64, count=ant1.size)
    lev2 = np.fromiter((digi.get(int(a), 0) for a in ant2), np.int64, count=ant2.size)
    cat44 = (lev1 == 4) & (lev2 == 4)
    cat22 = (lev1 == 2) & (lev2 == 2)
    cat24 = ((lev1 == 2) & (lev2 == 4)) | ((lev1 == 4) & (lev2 == 2))
    a_coef, h_coef = 5.36, 0.87890625
    gm = 3.335875 * 64.0 / 63.0
    rm = np.select([cat44, cat22, cat24], [4.3048, 1.0, 5.8784], default=1.0 / (a_coef * h_coef))
    alfa = np.select([cat44, cat22, cat24], [0.882518, 2.0 / np.pi, 0.882518], default=1.0)
    gamma = np.select([cat44, cat22, cat24], [gm, 64.0 / 63.0, gm], default=1.0)
    has_rho = cat22 | cat44
    if float(setup.get("vanvleck", 0.0)) != 0.0:
        alfa = np.ones_like(alfa)
        has_rho = np.zeros_like(cat44)
    bfactc = (gamma * gamma) / (a_coef * rm * alfa * h_coef)
    bfacta = (gamma * gamma) / (a_coef * rm * h_coef)
    scalar = cross | ~has_rho                          # cross rows, and autos without a rho model
    data[scalar] *= bfactc[scalar, None, None]
    usable = np.flatnonzero(~cross & has_rho)          # autos that get the van Vleck correction
    if not usable.size:
        return data
    same_pol = np.array([product[0] == product[1] for product in setup["corr_product"]])
    for pol in np.flatnonzero(~same_pol):              # cross hands of an auto: the scalar factor
        data[usable, :, pol] *= bfactc[usable, None]
    # The filler cosine-transforms each parallel-hand autocorrelation spectrum to the lag domain,
    # corrects the measured correlation by rho(r), and transforms back (FFTW REDFT00 twice, output
    # divided by 2 nchan). The dropped last spectral point is extrapolated first.
    transform = _redft00(nchan + 1)
    interior = np.arange(1, nchan)
    wt = (1.0 - interior / nchan)[None, :]             # 1 - chan / nchan
    correlator = str(setup["correlator"]).upper()
    for pol in np.flatnonzero(same_pol):
        spectrum = np.empty((usable.size, nchan + 1), dtype=np.float64)
        spectrum[:, :nchan] = bfacta[usable, None] * data.real[usable, :, pol]
        spectrum[:, nchan] = (2.0 * spectrum[:, nchan - 1] - spectrum[:, nchan - 2]) if nchan > 1 \
            else spectrum[:, nchan - 1]
        good = np.abs(spectrum[:, 0]) >= 1e-20         # the filler leaves such spectra untouched
        rows = usable[good]
        if not rows.size:
            continue
        lags = spectrum[good] @ transform.T
        scale = wt * lags[:, 0][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = lags[:, 1:nchan] / scale
        corrected = np.empty_like(ratio)
        four = cat44[rows]                             # cat22 rows take rho_2
        if (~four).any():
            corrected[~four] = np.sin(np.pi * ratio[~four] / 2.0)
        if four.any():
            corrected[four] = _rho_4(ratio[four])
        lags[:, 1:nchan] = scale * corrected
        lags[:, nchan] = 0.0
        back = lags @ transform.T
        if correlator == "VLBA":
            sat = 1.0 + (0.25 if ncorr > 2 else 0.125) * wmag[rows][:, :, pol]
        else:
            sat = 1.0                                  # DiFX doesn't suffer from saturation
        data[rows, :, pol] = (sat * back[:, :nchan] / (2.0 * nchan)).astype(np.complex64)
    return data


def convert_block(flux: np.ndarray, band: int, swapped: np.ndarray, cross: np.ndarray, interval: np.ndarray,
                  setup: dict, spw: int, weights: Optional[np.ndarray] = None,
                  ant1: Optional[np.ndarray] = None, ant2: Optional[np.ndarray] = None) -> dict:
    """DATA, FLAG, WEIGHT_SPECTRUM, SIGMA_SPECTRUM, SIGMA, WEIGHT and FLAG_ROW of rows of one band.

    Parameters
    ----------
    flux : numpy.ndarray
        FLUX cells of the rows, (nrow, nband, nchan, nstokes, 2 or 3) as read (big-endian float32 allowed).
    band : int
        The band (IF) of the output rows.
    swapped, cross : numpy.ndarray of bool
        Rows whose antennas were exchanged / that are cross-correlations.
    interval : numpy.ndarray
        INTTIM of the rows (float32).
    setup : dict
        ``corr_index``, ``corr_swap_index``, ``vis_scale``, ``eff_chan_bw``, ``uv_data_has_weights``,
        ``corr_product``, ``correlator``, ``corver``, ``vanvleck``, ``digi_levels``, ``weightyp``.
    spw : int
        The output subband (selects the channel width).
    weights : numpy.ndarray, optional
        The WEIGHT random parameter of the rows, (nrow, nband x nstokes); the weights of DiFX data,
        which carries none inside FLUX.
    ant1, ant2 : numpy.ndarray, optional
        MS antenna ids of the rows (needed for the digitizer levels of DiFX/VLBA data).
    """
    cells = np.asarray(flux[:, band], dtype=np.float32)                      # (nrow, nchan, nstokes, k)
    nrow, nchan, nstokes = cells.shape[:3]
    order = np.empty(nstokes, dtype=np.int64)
    order[np.asarray(setup["corr_index"])] = np.arange(nstokes)               # MS position -> FITS position
    swap_order = np.empty(nstokes, dtype=np.int64)
    swap_order[np.asarray(setup["corr_swap_index"])] = np.arange(nstokes)
    take = np.where(swapped[:, None], swap_order[None, :], order[None, :])     # (nrow, ncorr)
    rows = np.arange(nrow)[:, None, None]
    chans = np.arange(nchan)[None, :, None]
    picked = cells[rows, chans, take[:, None, :]]                              # (nrow, nchan, ncorr, k)
    real, imag = picked[..., 0], picked[..., 1]
    data = np.empty((nrow, nchan, nstokes), dtype=np.complex64)
    data.real = real
    data.imag = np.where(swapped[:, None, None], imag, -imag)
    if setup["uv_data_has_weights"]:
        raw = picked[..., 2]
    elif weights is not None:
        # The WEIGHT random parameter: one weight per (band, FITS correlation), constant over channels.
        raw = np.broadcast_to(weights[np.arange(nrow)[:, None], band * nstokes + take][:, None, :],
                              (nrow, nchan, nstokes)).astype(np.float32)
    else:
        raw = np.ones((nrow, nchan, nstokes), dtype=np.float32)
    flag = raw <= 0.0
    weight = np.where(flag, -raw, raw).astype(np.float32)
    correlator = str(setup.get("correlator", "")).upper()
    if correlator in ("DIFX", "VLBA"):
        data = _digital_correction(data, weight, ant1, ant2, cross, setup)
    if str(setup.get("weightyp", "")).upper() == "CORRELAT":
        # WEIGHTYP=CORRELAT: vis /= weightScale x weightSpec (raw weight magnitude), where weightScale is
        # VIS_SCAL, times the integration for VLBA correlator versions >= 4.17.
        scale = np.full(nrow, float(setup["vis_scale"]), dtype=np.float64)
        if correlator == "VLBA" and float(setup.get("corver", 0.0)) >= 4.17:
            scale *= interval.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            data /= (scale[:, None, None] * weight).astype(np.complex64)
    factor = (float(setup["vis_scale"]) * np.where(cross, 2.0, 1.0) * interval.astype(np.float64)
              * float(setup["eff_chan_bw"][spw]))
    weight = (weight.astype(np.float64) * factor[:, None, None]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_spectrum = np.where(weight > 0.0, np.float32(1.0) / np.sqrt(weight), np.float32(1.0)).astype(np.float32)
    sigma = np.sort(sigma_spectrum, axis=1)[:, (nchan - 1) // 2, :]            # casacore median: lower middle
    with np.errstate(divide="ignore"):
        weight_row = (np.float32(1.0) / (sigma * sigma)).astype(np.float32)   # partition: WEIGHT from SIGMA
    return {"DATA": data, "FLAG": flag, "WEIGHT_SPECTRUM": weight, "SIGMA_SPECTRUM": sigma_spectrum,
            "SIGMA": sigma, "WEIGHT": weight_row, "FLAG_ROW": flag.all(axis=(1, 2))}


def _gather(tables: list[dict], maps: dict, index: dict, block: np.ndarray, setup: dict) -> tuple:
    """FLUX cells ``(n, nband, nchan, nstokes, k)``, UVW [s] ``(n, 3)`` and the WEIGHT random parameter
    ``(n, nband x nstokes)`` — None when the tables have none — of the input rows ``block`` (indices into
    ``index``), in that order, read from the memory maps of their tables."""
    flux, uvw, weights = None, np.empty((block.size, 3), dtype=np.float64), None
    shape = (int(setup["nif"]), int(setup["nchan"]), int(setup["ncorr"]), -1)
    for table_index in np.unique(index["table"][block]):
        where = np.flatnonzero(index["table"][block] == table_index)
        if table_index not in maps:
            maps[table_index] = _rows(tables[table_index])
        rows = maps[table_index]
        source_rows = index["row"][block[where]]
        cells = rows[_column(rows, "FLUX")][source_rows].reshape((where.size,) + shape)
        if flux is None:
            flux = np.empty((block.size,) + cells.shape[1:], dtype=cells.dtype)
        flux[where] = cells
        for axis, name in enumerate(("UU", "VV", "WW")):
            uvw[where, axis] = rows[_column(rows, name)][source_rows]
        if "WEIGHT" in rows.dtype.names:
            row_weights = np.asarray(rows["WEIGHT"][source_rows]).reshape(where.size, -1)
            if weights is None:
                weights = np.empty((block.size,) + row_weights.shape[1:], dtype=row_weights.dtype)
            weights[where] = row_weights
    return flux, uvw, weights


def _declare_empty_cube_rank(path: Path, column: str, template: str) -> None:
    """Rewrite the TiledShapeStMan header of ``column``, whose cells are all undefined, with the rank casacore
    expects (the column's ndim + 1) and partition's default tile shape (``template``'s tile with a unit axis
    inserted before the row axis: (ncorr, nchan, 1, rows) for FLAG_CATEGORY), so that the part is the one
    partition writes (:func:`vlbipy.casacore_format.tsm.create_tiled_shape_manager` declares a unit tile)."""
    from ..casacore_format.table import Table
    from ..casacore_format.tsm import Cube, write_header
    with Table(path) as table:
        desc = table.coldesc(column)
        tile = tuple(table._manager(table.coldesc(template).seqnr).default_tile_shape)
        nrow, big_endian = table.nrows(), table.dat.big_endian
    write_header(path, desc.seqnr, "TiledShapeStMan", [desc], big_endian, nrow, [Cube((), (), -1, 0)], {},
                 desc.dm_group, tile[:-1] + (1,) + tile[-1:])


def write_part_job(*, path: str, tables: list[dict], chunks: list[dict], index: dict, setup: dict) -> dict:
    """Worker job: write one part (main table only) from its chunks; returns ``rows`` and ``seconds``."""
    from ..casacore_format.table import Table
    from ..casacore_format.writer import create_table
    started = _time.perf_counter()
    ncorr, nchan = int(setup["ncorr"]), int(setup["nchan"])
    rows = np.concatenate([c["rows"] for c in chunks])
    nrow = int(rows.size)
    spw = np.concatenate([np.full(c["rows"].size, c["spw"], dtype=np.int32) for c in chunks])
    scan = np.concatenate([np.full(c["rows"].size, c["scan"], dtype=np.int32) for c in chunks])
    time = index["time"][rows]
    interval = index["interval"][rows].astype(np.float64)
    scalars = {"ANTENNA1": index["ant1"][rows].astype(np.int32), "ANTENNA2": index["ant2"][rows].astype(np.int32),
               "ARRAY_ID": index["array"][rows].astype(np.int32), "DATA_DESC_ID": spw, "EXPOSURE": interval,
               "FEED1": np.zeros(nrow, dtype=np.int32), "FEED2": np.zeros(nrow, dtype=np.int32),
               "FIELD_ID": index["field"][rows].astype(np.int32), "FLAG_ROW": np.zeros(nrow, dtype=bool),
               "INTERVAL": interval, "OBSERVATION_ID": np.zeros(nrow, dtype=np.int32),
               "PROCESSOR_ID": np.full(nrow, -1, dtype=np.int32), "SCAN_NUMBER": scan,
               "STATE_ID": np.full(nrow, -1, dtype=np.int32), "TIME": time, "TIME_CENTROID": time + interval / 2.0}
    cube = (nchan, ncorr)
    tiled = {"UVW": (3,), "FLAG": cube, "WEIGHT": (ncorr,), "SIGMA": (ncorr,), "DATA": cube,
             "WEIGHT_SPECTRUM": cube, "SIGMA_SPECTRUM": cube, "FLAG_CATEGORY": [None] * nrow}
    # The scalars are written with the table; FLAG_ROW is rewritten once the flags are known.
    create_table(Path(path), _main_columns(ncorr, nchan), nrow, scalars, keywords={"MS_VERSION": np.float32(2.0)},
                 info=("Measurement Set", "UVFITS"), readme=PART_README, tiled=tiled)
    _declare_empty_cube_rank(Path(path), "FLAG_CATEGORY", "DATA")
    flag_row = scalars["FLAG_ROW"]
    maps: dict = {}
    with Table(Path(path), readonly=False) as out:
        start = 0
        for chunk in chunks:
            for first in range(0, chunk["rows"].size, BLOCK_ROWS):
                block = chunk["rows"][first:first + BLOCK_ROWS]
                flux, uvw, weights = _gather(tables, maps, index, block, setup)
                swapped = index["swapped"][block]
                cross = index["ant1"][block] != index["ant2"][block]
                values = convert_block(flux, chunk["band"], swapped, cross, index["interval"][block], setup,
                                       chunk["spw"], weights=weights,
                                       ant1=index["ant1"][block], ant2=index["ant2"][block])
                for name in ("DATA", "FLAG", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "SIGMA", "WEIGHT"):
                    out.putcol(name, values[name], start)
                out.putcol("UVW", np.where(swapped[:, None], -uvw, uvw) * SPEED_OF_LIGHT, start)
                flag_row[start:start + block.size] = values["FLAG_ROW"]
                start += block.size
        out.putcol("FLAG_ROW", flag_row, 0)
    return {"rows": nrow, "seconds": _time.perf_counter() - started}


# ---------------------------------------------------------------------------
# The whole import
# ---------------------------------------------------------------------------
def import_fitsidi(idi_files: list, mms, *, scan_gap: float = 15.0, nparts: Optional[int] = None,
                   workers: Optional[int] = None, specframe: str = "GEO", overwrite: bool = False) -> dict:
    """Convert FITS-IDI files into the Multi-MS ``mms`` (``importfitsidi(constobsid=True, scanreindexgap_s=
    scan_gap, specframe=specframe)`` + ``partition(createmms=True)``).

    Parameters
    ----------
    idi_files : list of str
        The FITS-IDI files, in order.
    mms : path
        The Multi-MS to create (``<code>.mms``).
    scan_gap : float
        Seconds between two integrations of a field that start a new scan.
    nparts : int
        Parts of the Multi-MS (default: 8, as partition's ``numsubms='auto'``; fewer when there are fewer
        (scan, subband) pairs).
    workers : int
        Worker processes writing parts (default: the pool's size).

    Returns
    -------
    dict
        ``rows``, ``parts``, ``scans``, ``seconds``.
    """
    from ..casacore_format.aipsio import TableRef
    from ..casacore_format.tabledat import read_table_dat, write_concat_dat, write_table_dat
    from ..casacore_format.writer import write_table_info
    from .idi_subtables import read_idi_setup, write_part_history, write_subtables

    started = _time.perf_counter()
    mms = Path(mms)
    if mms.exists():
        if not overwrite:
            raise FileExistsError(f"{mms} exists")
        shutil.rmtree(mms)
    setup = read_idi_setup([str(f) for f in idi_files])
    if str(setup.get("weightyp", "")).upper() not in ("", "NORMAL", "CORRELAT"):
        raise NotImplementedError(f"unsupported FITS-IDI WEIGHTYP {setup['weightyp']!r}")
    tables = uv_tables(idi_files)
    index = read_index(tables, setup)
    scans = scan_numbers(index["time"], index["field"], index["array"], float(scan_gap))
    nband = int(setup["nif"])
    nparts = int(nparts or 8)                      # partition's numsubms='auto' without an MPI cluster
    parts = plan_parts(index, scans, nband, nparts, int(setup["nchan"]) * int(setup["ncorr"]))
    submss = mms / "SUBMSS"
    submss.mkdir(parents=True)
    names = [f"{mms.name}.{k:04d}.ms" for k in range(len(parts))]
    jobs = [{"path": str(submss / name), "tables": tables, "chunks": chunks, "index": index, "setup": setup}
            for name, chunks in zip(names, parts)]
    reports = run_jobs("vlbipy.solvers.idi_import:write_part_job", jobs, workers,
                       desc="importfitsidi: writing parts")
    written = time_index_summary(index, setup, scans)
    first = submss / names[0]
    # In CASA's keyword order (write_subtables returns it; WEATHER and PHASE_CAL depend on the files).
    subtables = write_subtables([str(f) for f in idi_files], first, setup, written, specframe=specframe,
                                history_params={"scanreindexgap_s": float(scan_gap)})
    for name in names:
        part = submss / name
        if part != first:
            for sub in subtables:
                if sub == "HISTORY":                 # every part has its own, without partition's rows
                    write_part_history(first, part)
                elif sub == "SOURCE":                # partition writes a SOURCE table in every part
                    shutil.copytree(first / sub, part / sub, symlinks=True)
                else:
                    os.symlink(f"../{names[0]}/{sub}", part / sub)
        dat = read_table_dat(part)
        dat.keywords.update({sub: TableRef(f"././{sub}") for sub in subtables})
        write_table_dat(part, dat)
    for sub in subtables:
        os.symlink(f"SUBMSS/{names[0]}/{sub}", mms / sub)
    total = int(sum(r["rows"] for r in reports))
    write_concat_dat(mms, total, [f"SUBMSS/{name}" for name in names])
    readme = (f"AxisType = scan,spw\n{PART_README}Virtual concatenation of the following tables:\n"
              + "".join(f"  SUBMSS/{name}\n" for name in names) + "\n")
    write_table_info(mms, "Measurement Set", "CONCATENATED", readme)
    report = {"rows": total, "parts": len(names), "scans": int(np.unique(scans).size),
              "seconds": _time.perf_counter() - started}
    logger.info("importfitsidi[native]: {} file(s) -> {} ({} rows in {} part(s), {} scans) in {:.1f}s",
                len(idi_files), mms.name, total, len(names), report["scans"], report["seconds"])
    return report


def time_index_summary(index: dict, setup: dict, scans: np.ndarray) -> dict:
    """What only the main-table pass knows, for the subtables: time range, fields and subbands written."""
    return {"time_start": float(index["time"][0]),
            "time_end": float(index["time"][-1] + float(index["interval"][-1])),
            "nfield": int(index["field"].max()) + 1 if index["field"].size else 0,
            "nspw": int(setup["nspw"]), "nscan": int(np.unique(scans).size)}
