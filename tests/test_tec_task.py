"""The ionospheric (TEC) correction (``vlbipy.ionex``, ``vlbipy.solvers.tec_task``): IONEX names, parsing and
interpolation on synthetic maps, the TECOR geometry, the exact Fringe Jones representation, the table of a
synthetic measurement set applied by the native apply, and (when the data are here) RSM07 with CASA's applycal.
No test uses the network."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vlbipy import ionex
from vlbipy.casacore_format.aipsio import TableRef
from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.writer import create_table, scalar_column
from vlbipy.solvers import tec_task
from vlbipy.solvers.apply import antenna_gains, load_caltable
from vlbipy.solvers.caltable import _columns
from vlbipy.solvers.fringe import k_disp
from vlbipy.solvers.gaincurve_task import SUBTABLE_DESCS

RSM07_MMS = Path(__file__).resolve().parents[1] / "test" / "RSM07_import_ref" / "RSM07.mms"
#: 2025-09-16 13:00 UTC, the start of RSM07 [MJD seconds].
T0 = (60934 + 13.0 / 24.0) * 86400.0
ANTENNA_XYZ = np.array([[4033947.07, 486990.99, 4900431.12],      # EF
                        [3828750.31, 442589.71, 5064921.82],      # WB
                        [5085442.78, 2668263.48, -2768697.03]])   # HH
FIELD_RADEC = np.radians([[250.745, 39.810], [282.092, 32.317]])
CHAN_FREQ = np.array([1594.99e6 + 0.5e6 * np.arange(8), 1658.99e6 + 0.5e6 * np.arange(8)])
#: (field, scan, first time, last time) relative to T0 [s]; integrations every 2 s.
SCANS = [(0, 1, 0.0, 120.0), (1, 2, 150.0, 270.0), (0, 3, 300.0, 420.0)]


def linear_tec(lat, lon, hours):
    """A TEC field [TECU] that bilinear and IONEX-eq.3 interpolation reproduce exactly (rotation cancels too)."""
    return 20.0 + 0.1 * np.asarray(lat) + 0.05 * np.asarray(lon) + 2.0 * np.asarray(hours)


def write_ionex(path: Path, day: dt.date, values, *, step_hours: int = 2, dlat: float = -17.5, dlon: float = 30.0,
                exponent: int = -2, missing=None, rms: bool = True) -> Path:
    """A small global IONEX 1.0 file: maps every ``step_hours`` from 00:00 to 24:00 of ``day`` holding
    ``values(lat, lon, hours since 00:00 of day)`` [TECU]; ``missing`` = (map, lat, lon) written as 9999."""
    lat = np.arange(87.5, -87.5 - 1e-9, dlat)
    lon = np.arange(-180.0, 180.0 + 1e-9, dlon)
    nmap = 24 // step_hours + 1
    lines = [f"{1.0:8.1f}{'':12s}{'I':1s}{'':19s}{'GPS':3s}{'':17s}IONEX VERSION / TYPE",
             f"{day.year:6d}{day.month:6d}{day.day:6d}{0:6d}{0:6d}{0:6d}{'':24s}EPOCH OF FIRST MAP",
             f"{nmap:6d}{'':54s}# OF MAPS IN FILE",
             f"{'  THIN SHELL':60s}MAPPING FUNCTION",
             f"{6371.0:8.1f}{'':52s}BASE RADIUS",
             f"{2:6d}{'':54s}MAP DIMENSION",
             f"  {450.0:6.1f}{450.0:6.1f}{0.0:6.1f}{'':40s}HGT1 / HGT2 / DHGT",
             f"  {lat[0]:6.1f}{lat[-1]:6.1f}{dlat:6.1f}{'':40s}LAT1 / LAT2 / DLAT",
             f"  {lon[0]:6.1f}{lon[-1]:6.1f}{dlon:6.1f}{'':40s}LON1 / LON2 / DLON",
             f"{exponent:6d}{'':54s}EXPONENT",
             f"{'':60s}END OF HEADER"]
    for kind in ("TEC", "RMS") if rms else ("TEC",):
        for k in range(nmap):
            moment = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=step_hours * k)
            lines.append(f"{k + 1:6d}{'':54s}START OF {kind} MAP")
            lines.append(f"{moment.year:6d}{moment.month:6d}{moment.day:6d}{moment.hour:6d}{0:6d}{0:6d}{'':24s}"
                         "EPOCH OF CURRENT MAP")
            for i, row_lat in enumerate(lat):
                lines.append(f"  {row_lat:6.1f}{lon[0]:6.1f}{lon[-1]:6.1f}{dlon:6.1f}{450.0:6.1f}{'':28s}"
                             "LAT/LON1/LON2/DLON/H")
                field = values(row_lat, lon, step_hours * k) if kind == "TEC" else np.full(lon.size, 1.5)
                counts = np.rint(np.asarray(field) / 10.0 ** exponent).astype(int)
                if missing is not None and kind == "TEC" and missing[0] == k and missing[1] == i:
                    counts[missing[2]] = 9999
                for start in range(0, lon.size, 16):
                    lines.append("".join(f"{c:5d}" for c in counts[start:start + 16]))
            lines.append(f"{k + 1:6d}{'':54s}END OF {kind} MAP")
    lines.append(f"{'':60s}END OF FILE")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def make_ms(path: Path) -> Path:
    """A measurement set with the index columns only: 3 antennas, 2 fields in 3 scans, 2 subbands of 8 channels."""
    rows = []
    for field, scan, first, last in SCANS:
        for spw in range(2):
            for t in T0 + np.arange(first, last + 0.5, 2.0):
                rows += [(t, i, j, field, scan, spw) for i in range(3) for j in range(i + 1, 3)]
    time, ant1, ant2, field, scan, ddid = (np.array(column) for column in zip(*rows))
    nrow = time.size
    main = [scalar_column(name, kind) for name, kind in (
        ("TIME", "double"), ("ANTENNA1", "int"), ("ANTENNA2", "int"), ("FIELD_ID", "int"), ("SCAN_NUMBER", "int"),
        ("DATA_DESC_ID", "int"))]
    values = {"TIME": time, "ANTENNA1": ant1.astype(np.int32), "ANTENNA2": ant2.astype(np.int32),
              "FIELD_ID": field.astype(np.int32), "SCAN_NUMBER": scan.astype(np.int32),
              "DATA_DESC_ID": ddid.astype(np.int32)}
    names = ("ANTENNA", "FIELD", "OBSERVATION", "SPECTRAL_WINDOW", "DATA_DESCRIPTION", "POLARIZATION")
    create_table(path, main, nrow, values, keywords={name: TableRef(f"././{name}") for name in names})
    antennas = ["EF", "WB", "HH"]
    create_table(path / "ANTENNA", _columns(SUBTABLE_DESCS["ANTENNA"]), 3, {
        "OFFSET": np.zeros((3, 3)), "POSITION": ANTENNA_XYZ, "TYPE": ["GROUND-BASED"] * 3,
        "DISH_DIAMETER": np.full(3, 25.0), "FLAG_ROW": np.zeros(3, bool), "MOUNT": ["ALT-AZ"] * 3,
        "NAME": antennas, "STATION": antennas})
    directions = [d.reshape(1, 2) for d in FIELD_RADEC]
    create_table(path / "FIELD", _columns(SUBTABLE_DESCS["FIELD"]), 2, {
        "DELAY_DIR": directions, "PHASE_DIR": directions, "REFERENCE_DIR": directions, "CODE": ["", ""],
        "FLAG_ROW": np.zeros(2, bool), "NAME": ["3C345", "J1848+3219"], "NUM_POLY": np.zeros(2, np.int32),
        "SOURCE_ID": np.arange(2, dtype=np.int32), "TIME": np.full(2, T0)})
    create_table(path / "OBSERVATION", _columns(SUBTABLE_DESCS["OBSERVATION"]), 1, {
        "TIME_RANGE": np.array([[T0, T0 + 420.0]]), "LOG": [np.array([""], dtype=object)],
        "SCHEDULE": [np.array([""], dtype=object)], "FLAG_ROW": np.zeros(1, bool), "OBSERVER": ["me"],
        "PROJECT": ["SIM"], "RELEASE_DATE": np.zeros(1), "SCHEDULE_TYPE": [""], "TELESCOPE_NAME": ["EVN"]})
    width = np.full(CHAN_FREQ.shape, 0.5e6)
    create_table(path / "SPECTRAL_WINDOW", _columns(SUBTABLE_DESCS["SPECTRAL_WINDOW"]), 2, {
        "MEAS_FREQ_REF": np.full(2, 5, np.int32), "CHAN_FREQ": CHAN_FREQ, "REF_FREQUENCY": CHAN_FREQ[:, 0],
        "CHAN_WIDTH": width, "EFFECTIVE_BW": width, "RESOLUTION": width, "FLAG_ROW": np.zeros(2, bool),
        "FREQ_GROUP": np.zeros(2, np.int32), "FREQ_GROUP_NAME": ["", ""], "IF_CONV_CHAIN": np.arange(2, dtype=np.int32),
        "NAME": ["IF0", "IF1"], "NET_SIDEBAND": np.ones(2, np.int32), "NUM_CHAN": np.full(2, 8, np.int32),
        "TOTAL_BANDWIDTH": np.full(2, 4e6)})
    create_table(path / "DATA_DESCRIPTION", [scalar_column("SPECTRAL_WINDOW_ID", "int"),
                                             scalar_column("POLARIZATION_ID", "int")], 2,
                 {"SPECTRAL_WINDOW_ID": np.arange(2, dtype=np.int32), "POLARIZATION_ID": np.zeros(2, np.int32)})
    create_table(path / "POLARIZATION", [scalar_column("NUM_CORR", "int")], 1, {"NUM_CORR": np.array([2], np.int32)})
    return path


@pytest.fixture
def native_engine(monkeypatch):
    """Read the synthetic tables with the native engine."""
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")


@pytest.fixture
def no_network(monkeypatch):
    """Any download attempt fails (and is recorded)."""
    calls = []

    def refuse_curl(url, dest, timeout):
        calls.append(url)
        return "network disabled in tests", False

    def refuse_download(url, dest, **kwargs):
        calls.append(url)
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(ionex, "_curl", refuse_curl)
    monkeypatch.setattr("vlbipy.tools.download_file", refuse_download)
    return calls


# ----------------------------------------------------------------------------------------------------------------
# IONEX names, download fallbacks, parsing, interpolation
# ----------------------------------------------------------------------------------------------------------------
def test_ionex_file_names_follow_the_aips_switch_dates():
    assert ionex.ionex_file_names("jplg", 2023, 219)[0] == "jplg2190.23i.Z"
    assert ionex.ionex_file_names("JPLG", 2023, 220)[:2] == ["JPL0OPSFIN_20232200000_01D_02H_GIM.INX.gz",
                                                             "JPL0OPSFIN_20232200000_01D_01H_GIM.INX.gz"]
    assert ionex.ionex_file_names("emrg", 2025, 259)[0] == "EMR0OPSFIN_20252590000_01D_01H_GIM.INX.gz"
    assert ionex.ionex_file_names("codg", 2022, 330)[0] == "codg3300.22i.Z"
    assert ionex.ionex_file_names("casg", 2024, 5)[0] == "CAS0OPSFIN_20240050000_01D_30M_GIM.INX.gz"
    assert ionex.ionex_file_names("igrg", 2025, 1)[0] == "IGS0OPSRAP_20250010000_01D_02H_GIM.INX.gz"
    assert ionex.ionex_file_names("uqrg", 2025, 1)[0] == "uqrg0010.25i.Z"
    with pytest.raises(ValueError, match="unknown IONEX product"):
        ionex.ionex_file_names("xxxg", 2025, 1)
    # An observation crossing midnight needs both days (and the new year).
    end_of_year = (dt.date(2024, 12, 31) - dt.date(1858, 11, 17)).days * 86400.0
    assert ionex.observation_days(end_of_year + 80000.0, end_of_year + 90000.0) == [(2024, 366), (2025, 1)]


def test_fetch_uses_the_cache_and_reports_every_failure(tmp_path, no_network):
    cached = tmp_path / "EMR0OPSFIN_20252590000_01D_01H_GIM.INX"
    write_ionex(cached, dt.date(2025, 9, 16), linear_tec)
    product, files = ionex.fetch_ionex(T0, T0 + 3600.0, tmp_path, model="emrg")
    assert product == "emrg" and files == [cached] and not no_network
    # Next day: nothing cached and no network -> a clear error naming the products and URLs tried.
    with pytest.raises(FileNotFoundError) as error:
        ionex.fetch_ionex(T0 + 86400.0, T0 + 86400.0 + 60.0, tmp_path, model="jplg")
    message = str(error.value)
    assert "jplg" in message and "JPL0OPSFIN_20252600000_01D_02H_GIM.INX.gz" in message and "uqrg" in message
    assert any("sideshow.jpl.nasa.gov" in url for url in no_network)
    # Without fallback only the requested product is tried.
    no_network.clear()
    with pytest.raises(FileNotFoundError):
        ionex.fetch_ionex(T0 + 86400.0, T0 + 86400.0 + 60.0, tmp_path, model="codg", fallback=False)
    assert no_network and all("COD" in url or "codg" in url for url in no_network)


def test_an_unreachable_cddis_is_asked_once(tmp_path, monkeypatch):
    calls = []

    def unreachable(url, dest, timeout):
        calls.append(url)
        return "Operation timed out", url.startswith("ftp://")
    monkeypatch.setattr(ionex, "_curl", unreachable)
    monkeypatch.setattr("vlbipy.tools.download_file", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("off")))
    write_ionex(tmp_path / "JPL0OPSFIN_20252590000_01D_02H_GIM.INX", dt.date(2025, 9, 16), linear_tec)
    product, files = ionex.fetch_ionex(T0, T0 + 60.0, tmp_path, model="emrg")
    assert product == "jplg" and files[0].name == "JPL0OPSFIN_20252590000_01D_02H_GIM.INX"
    assert sum(url.startswith("ftp://") for url in calls) == 1


def test_read_ionex_values_grid_missing_and_merge(tmp_path):
    day = dt.date(2025, 9, 16)
    first = ionex.read_ionex(write_ionex(tmp_path / "a.inx", day, linear_tec, missing=(1, 2, 3)))
    assert first.epochs.size == 13 and first.epochs[0] == pytest.approx(60934 * 86400.0)
    assert np.all(np.diff(first.lat) > 0) and first.lat[0] == -87.5 and first.lon[-1] == 180.0
    assert first.height_km == 450.0 and first.base_radius_km == 6371.0 and first.rms is not None
    expected = linear_tec(first.lat[:, None], first.lon[None, :], 4.0)
    assert np.allclose(first.tec[2], expected, atol=5e-3)
    lat_of_missing = 87.5 - 17.5 * 2
    assert np.isnan(first.tec[1, np.flatnonzero(first.lat == lat_of_missing)[0], 3])
    # The next day repeats the midnight map with other values: the first one read is kept (AIPS).
    second = ionex.read_ionex(write_ionex(tmp_path / "b.inx", day + dt.timedelta(days=1),
                                          lambda la, lo, h: linear_tec(la, lo, h) + 5.0))
    merged = ionex.merge_maps([second, first])
    assert merged.epochs.size == 25 and np.all(np.diff(merged.epochs) > 0)
    assert np.allclose(merged.tec[12], first.tec[12], equal_nan=True)
    assert np.allclose(merged.tec[13], second.tec[1], equal_nan=True)


def test_vertical_tec_is_exact_on_linear_maps_and_follows_ionex_eq3(tmp_path):
    from scipy.interpolate import RegularGridInterpolator
    day = dt.date(2025, 9, 16)
    maps = ionex.read_ionex(write_ionex(tmp_path / "lin.inx", day, linear_tec))
    rng = np.random.default_rng(3)
    lat, lon = rng.uniform(-87.5, 87.5, 500), rng.uniform(-150.0, 150.0, 500)
    hours = rng.uniform(0.0, 24.0, 500)
    times = maps.epochs[0] + 3600.0 * hours
    for rotation in (0.0, 0.5, 1.0):
        assert np.allclose(ionex.vertical_tec(maps, lat, lon, times, rotation=rotation),
                           linear_tec(lat, lon, hours), atol=1e-9)
    assert np.isnan(ionex.vertical_tec(maps, 0.0, 0.0, maps.epochs[-1] + 1.0))
    assert np.allclose(ionex.vertical_tec(maps, 10.0, 10.0, maps.epochs[0], use_rms=True), 1.5)

    # A non-linear field against an independent evaluation of IONEX eq. 3 (rotated maps, linear in time).
    def bump(la, lo, h):
        return 30.0 + 20.0 * np.cos(np.radians(la)) ** 2 * np.cos(np.radians(lo - 15.0 * h)) + h
    maps = ionex.read_ionex(write_ionex(tmp_path / "bump.inx", day, bump))
    grids = [RegularGridInterpolator((maps.lat, maps.lon), maps.tec[k]) for k in range(maps.epochs.size)]
    k = np.minimum((hours // 2).astype(int), 11)
    dt1, dt2 = (hours - 2.0 * k) / 24.0, (2.0 * (k + 1) - hours) / 24.0

    def wrap(x):
        return (x + 180.0) % 360.0 - 180.0
    reference = np.array([(dt2[i] * grids[k[i]]([lat[i], wrap(lon[i] + 360.0 * dt1[i])])[0]
                           + dt1[i] * grids[k[i] + 1]([lat[i], wrap(lon[i] - 360.0 * dt2[i])])[0]) / (dt1[i] + dt2[i])
                          for i in range(lat.size)])
    assert np.allclose(ionex.vertical_tec(maps, lat, lon, times, rotation=1.0), reference, atol=1e-9)


# ----------------------------------------------------------------------------------------------------------------
# TECOR geometry and the Fringe Jones representation
# ----------------------------------------------------------------------------------------------------------------
def test_pierce_point_and_mapping_function_follow_tecor():
    height = 450e3
    lat_ipp, dlon, za_ipp = tec_task.pierce_points(np.radians(50.0), np.radians(30.0), np.radians(90.0), height)
    assert lat_ipp == pytest.approx(np.radians(50.0)) and dlon == pytest.approx(0.0) and za_ipp == pytest.approx(0.0)
    ratio = 6378.0 / (6378.0 + 450.0)
    assert tec_task.slant_factor(0.0, height) == pytest.approx(1.0 / np.sqrt(1.0 - ratio ** 2))
    assert tec_task.slant_factor(np.radians(90.0), height) == pytest.approx(1.0)
    # The thin-shell factor is 1 / cos(zenith angle at the shell).
    el = np.radians(20.0)
    assert tec_task.slant_factor(el, height) == pytest.approx(
        1.0 / np.cos(tec_task.pierce_points(0.0, 0.0, el, height)[2]))
    petrov = tec_task.slant_factor(el, height, **tec_task.PETROV_MAPPING)
    ratio = 6378.0 / (6378.0 + 450.0 + 56.7)
    assert petrov == pytest.approx(0.85 / np.sqrt(1.0 - (ratio * np.cos(0.9782 * el)) ** 2))
    # A source due north (azimuth 0) pierces the shell north of the antenna, at the same longitude.
    lat_ipp, dlon, _ = tec_task.pierce_points(np.radians(50.0), 0.0, np.radians(30.0), height)
    assert lat_ipp > np.radians(50.0) and dlon == pytest.approx(0.0, abs=1e-12)
    lat_ipp, dlon, _ = tec_task.pierce_points(np.radians(50.0), np.radians(90.0), np.radians(30.0), height)
    assert dlon > 0.0


def test_fringe_parameters_reproduce_the_dispersive_phase_on_every_channel():
    rng = np.random.default_rng(7)
    for _ in range(20):
        fmin = rng.uniform(0.3e9, 8e9)
        freq = fmin + rng.uniform(4e6, 128e6) * np.linspace(0.0, 1.0, 33)
        f_ref = freq[16] + rng.uniform(-1e6, 1e6)
        stec = rng.uniform(1.0, 150.0)
        phase, delay, disp = tec_task.fringe_parameters(stec, freq[0], freq[-1], f_ref)
        model = phase + 2e-9 * np.pi * delay * (freq - f_ref) + disp * k_disp(freq, freq[0], freq[-1])
        exact = tec_task.ionospheric_phase(stec, freq)
        assert np.abs(np.angle(np.exp(1j * (model - exact)))).max() < 1e-7
    # 1 TECU at 1.6 GHz is 5.28 rad; the sign is IONOSPHERE_SIGN (+: measured on RSM07).
    assert tec_task.ionospheric_phase(1.0, 1.6e9) == pytest.approx(tec_task.IONOSPHERE_SIGN * 5.27998, rel=1e-5)
    assert np.sign(tec_task.fringe_parameters(10.0, 1.6e9, 1.632e9, 1.616e9)[1]) == -tec_task.IONOSPHERE_SIGN
    # Applied to a channel selection the error is a nearly constant phase of ~1e-3 rad per 10 TECU at L band.
    freq = 1594.99e6 + 0.5e6 * np.arange(64)
    error = tec_task.selection_error(10.0, freq[0], freq[-1], freq[4], freq[59], freq[4:60])
    assert np.abs(error).max() < 2e-3 and np.ptp(error) < 1e-4


def test_row_times_cover_every_scan():
    assert np.allclose(tec_task.row_times(0.0, 100.0, 30.0), [0.0, 30.0, 60.0, 90.0, 100.0])
    assert np.allclose(tec_task.row_times(0.0, 60.0, 30.0), [0.0, 30.0, 60.0])
    assert np.allclose(tec_task.row_times(5.0, 5.0, 30.0), [5.0])


# ----------------------------------------------------------------------------------------------------------------
# The table of a synthetic measurement set
# ----------------------------------------------------------------------------------------------------------------
def test_write_tec_table_applies_each_field_its_own_correction(tmp_path, native_engine, no_network):
    ms = make_ms(tmp_path / "sim.ms")
    write_ionex(tmp_path / "ionex" / "EMR0OPSFIN_20252590000_01D_01H_GIM.INX", dt.date(2025, 9, 16), linear_tec)
    report = tec_task.write_tec_table(ms, tmp_path / "sim.tec", ionex_dir=tmp_path / "ionex", interval=30.0)
    assert report["product"] == "emrg" and report["fields"] == ["3C345", "J1848+3219"] and not no_network
    assert report["shell_height_km"] == 450.0 and not report["simultaneous_fields"]
    ntime = sum(tec_task.row_times(T0 + lo, T0 + hi, 30.0).size for _, _, lo, hi in SCANS)
    assert report["nrow"] == ntime * 2 * 3
    with Table(tmp_path / "sim.tec") as table:
        keywords = table.getkeywords()
        field, scan = np.asarray(table.getcol("FIELD_ID")), np.asarray(table.getcol("SCAN_NUMBER"))
        flags = np.asarray(table.getcol("FLAG"))
    assert keywords["VisCal"] == "Fringe Jones" and not flags.any()
    assert set(zip(scan, field)) == {(s, f) for f, s, _, _ in SCANS}

    table = load_caltable(tmp_path / "sim.tec")
    maps = ionex.read_ionex(tmp_path / "ionex" / "EMR0OPSFIN_20252590000_01D_01H_GIM.INX")
    for field_id, _, first, last in SCANS:
        times = T0 + np.arange(first, last + 0.5, 2.0)
        stec = tec_task.slant_tec(maps, ANTENNA_XYZ, *FIELD_RADEC[field_id], times)["stec"]
        other = tec_task.slant_tec(maps, ANTENNA_XYZ, *FIELD_RADEC[1 - field_id], times)["stec"]
        for spw in range(2):
            exact = tec_task.ionospheric_phase(stec[:, :, None], CHAN_FREQ[spw][None, None, :])
            for gainfield in ([field_id], None):          # the field's own rows, or every row: the same
                gains, gflags = antenna_gains(table, np.arange(3), times, spw, CHAN_FREQ[spw], field_id=gainfield)
                assert not gflags.any()
                error = np.angle(gains[..., 0] * np.exp(-1j * exact))
                assert np.abs(error).max() < 2e-3
                assert np.allclose(gains[..., 0], gains[..., 1])
            # The other field's direction gives another correction: the table is direction-dependent.
            wrong = tec_task.ionospheric_phase(other[:, :, None], CHAN_FREQ[spw][None, None, :])
            assert np.abs(np.angle(np.exp(1j * (wrong - exact)))).max() > 0.1
    assert report["interp_error"] < 2e-3


def test_write_tec_table_fails_clearly_without_maps(tmp_path, native_engine, no_network):
    ms = make_ms(tmp_path / "sim.ms")
    with pytest.raises(FileNotFoundError, match="no IONEX product covers days 2025/259"):
        tec_task.write_tec_table(ms, tmp_path / "sim.tec", ionex_dir=tmp_path / "ionex")
    # Maps that end before the data do not cover them.
    write_ionex(tmp_path / "early.inx", dt.date(2025, 9, 15), linear_tec)
    with pytest.raises(ValueError, match="not the data"):
        tec_task.write_tec_table(ms, tmp_path / "sim.tec", ionex_dir=tmp_path / "ionex",
                                 ionex_files=[tmp_path / "early.inx"])


# ----------------------------------------------------------------------------------------------------------------
# RSM07 (EVN, 1.6 GHz), when the data are here
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not RSM07_MMS.is_dir() or os.environ.get("VLBIPY_SKIP_REALDATA"), reason="RSM07 not available")
def test_rsm07_table_native_apply_and_casa_applycal_agree(tmp_path, monkeypatch, native_engine):
    """The table of one RSM07 scan (synthetic maps on the real geometry) matches the exact phase at every
    integration and channel, and CASA's applycal applies it as the native apply does."""
    monkeypatch.chdir(tmp_path)                                   # CASA writes its log in the working directory
    casatasks = pytest.importorskip("casatasks")
    import casatools
    from vlbipy.solvers.apply import apply_tables
    from vlbipy.solvers.msio import read_setup
    ionex_dir = tmp_path / "ionex"
    write_ionex(ionex_dir / "EMR0OPSFIN_20252590000_01D_01H_GIM.INX", dt.date(2025, 9, 16),
                lambda la, lo, h: linear_tec(la, lo, h) + 10.0 * np.cos(np.radians(lo - 15.0 * h)))
    clone = tmp_path / "rsm07.mms"
    if subprocess.run(["cp", "-cR", str(RSM07_MMS), str(clone)], capture_output=True).returncode != 0:
        pytest.skip("cannot clone the RSM07 measurement set")
    small = tmp_path / "scan47.ms"
    casatasks.split(vis=str(clone), outputvis=str(small), scan="47", datacolumn="data", keepmms=False)
    shutil.rmtree(clone, ignore_errors=True)
    report = tec_task.write_tec_table(small, tmp_path / "scan47.tec", ionex_dir=ionex_dir, fallback=False)
    assert report["fields"] == ["3C395"] and report["interp_error"] < 0.01
    casatasks.applycal(vis=str(small), gaintable=[str(tmp_path / "scan47.tec")], interp=["linear"], calwt=[False],
                       parang=False, flagbackup=False)
    tb = casatools.table()
    tb.open(str(small))
    data, corrected = tb.getcol("DATA"), tb.getcol("CORRECTED_DATA")
    a1, a2, time, ddid = (tb.getcol(c) for c in ("ANTENNA1", "ANTENNA2", "TIME", "DATA_DESC_ID"))
    tb.close()
    setup = read_setup(small)
    spw = setup["ddid_to_spw"][ddid]
    vis = np.ascontiguousarray(data.transpose(2, 1, 0))
    native, _, _ = apply_tables(vis, np.zeros(vis.shape, bool), None, a1, a2, time, spw, setup["chan_freq"],
                                [{"path": str(tmp_path / "scan47.tec"), "interp": "linear"}])
    casa = np.ascontiguousarray(corrected.transpose(2, 1, 0))
    good = (np.abs(vis) > 0) & (a1 != a2)[:, None, None]
    assert np.abs(np.angle(native[good] * np.conj(casa[good]))).max() < 1e-3
    # Against the exact phase at every integration and channel (antenna-based, both hands alike).
    maps = ionex.read_ionex(ionex_dir / "EMR0OPSFIN_20252590000_01D_01H_GIM.INX")
    utimes, index = np.unique(time, return_inverse=True)
    field = setup["field_names"].index("3C395")
    stec = tec_task.slant_tec(maps, setup["antenna_xyz"], *setup["field_dirs"][field], utimes)["stec"]
    exact = tec_task.ionospheric_phase((stec[a1, index] - stec[a2, index])[:, None], setup["chan_freq"][spw])
    removed = np.angle(vis * np.conj(native))[..., 0]
    assert np.abs(np.angle(np.exp(1j * (removed - exact))))[good[..., 0]].max() < 5e-3
