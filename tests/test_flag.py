"""Tests for the native flagging engine of the dask-ms backend (vlbipy.solvers.flagdata).

The expected behaviour is CASA's ``flagdata``, measured selection by selection on
real data (GL048): see the module docstring of :mod:`vlbipy.solvers.flagdata`.
These tests pin those semantics down on a small synthetic measurement set, and
check the statistics and amplitudes against brute-force numpy.
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

from vlbipy.solvers import flagdata  # noqa: E402
from vlbipy.solvers.flagdata import (Unsupported, apply_flag_selections, compile_flagdata,  # noqa: E402
                                      count_flags, parse_antenna, parse_command, parse_spw, parse_timerange,
                                      read_flag_setup, read_row_amplitudes)

NANT, NSPW, NCHAN = 4, 2, 8
NAMES = ["EF", "JB", "MC", "O6"]
T0, DT = 5.0e9, 2.0
NTIME = 10                     # per scan
SCAN_GAP = 100.0


def _scalar(value_type):
    return {"valueType": value_type, "dataManagerType": "StandardStMan", "dataManagerGroup": "SSM", "option": 0,
            "maxlen": 0, "comment": "", "keywords": {}}


def _array(value_type, shape):
    desc = _scalar(value_type)
    desc["ndim"], desc["option"], desc["shape"] = len(shape), 5, np.array(shape)
    return desc


def _create(path, desc, nrow, columns):
    tb = casatools.table()
    tb.create(str(path), desc)
    tb.addrows(nrow)
    for name, values in columns.items():
        values = np.asarray(values)
        tb.putcol(name, np.asfortranarray(values.T) if values.ndim > 1 else values)
    tb.close()


def _main_rows(scans):
    """Rows (scan, spw, time step, ant1, ant2) with autocorrelations, in MS order (time-major)."""
    pairs = [(i, j) for i in range(NANT) for j in range(i, NANT)]
    return [(scan, spw, t, i, j) for scan in scans for spw in range(NSPW) for t in range(NTIME) for i, j in pairs]


def _write_main(path, rows, rng):
    scan, spw, step, ant1, ant2 = (np.array(column) for column in zip(*rows))
    nrow = len(rows)
    time = T0 + DT * step + SCAN_GAP * (scan - 1)
    data = (rng.uniform(0.5, 1.5, (nrow, NCHAN, 2)) * np.exp(1j * rng.uniform(-3, 3, (nrow, NCHAN, 2))))
    data = data.astype(np.complex64)
    data[(ant1 == 2) & (ant2 == 3) & (scan == 2)] = 0          # a baseline never recorded in scan 2
    partial = np.flatnonzero((ant1 == 0) & (ant2 == 1) & (spw == 1))[:3]
    data[partial, :2, 1] = 0                                     # rows that are only partly zero
    flag = np.zeros(data.shape, dtype=bool)
    flag[rng.uniform(size=data.shape) < 0.05] = True
    main = {"TIME": _scalar("double"), "ANTENNA1": _scalar("int"), "ANTENNA2": _scalar("int"),
            "FIELD_ID": _scalar("int"), "SCAN_NUMBER": _scalar("int"), "DATA_DESC_ID": _scalar("int"),
            "INTERVAL": _scalar("double"), "FLAG_ROW": _scalar("boolean"), "DATA": _array("complex", [2, NCHAN]),
            "FLAG": _array("boolean", [2, NCHAN])}
    _create(path, main, nrow, {
        "TIME": time, "ANTENNA1": ant1.astype(np.int32), "ANTENNA2": ant2.astype(np.int32),
        "FIELD_ID": (scan % 2).astype(np.int32), "SCAN_NUMBER": scan.astype(np.int32),
        "DATA_DESC_ID": spw.astype(np.int32), "INTERVAL": np.full(nrow, DT), "FLAG_ROW": np.zeros(nrow, dtype=bool),
        "DATA": data, "FLAG": flag})


def _write_subtables(ms):
    names = np.array(NAMES)
    _create(ms / "ANTENNA", {"NAME": _scalar("string"), "POSITION": _array("double", [3]),
                             "MOUNT": _scalar("string")}, NANT,
            {"NAME": names, "POSITION": np.ones((NANT, 3)), "MOUNT": np.array(["ALT-AZ"] * NANT)})
    _create(ms / "FIELD", {"NAME": _scalar("string"), "PHASE_DIR": _array("double", [2, 1])}, 2,
            {"NAME": np.array(["TARGET", "CAL"]), "PHASE_DIR": np.zeros((2, 1, 2))})
    _create(ms / "SPECTRAL_WINDOW", {"CHAN_FREQ": _array("double", [NCHAN])}, NSPW,
            {"CHAN_FREQ": np.array([[1e9 + 1e6 * c + 1e8 * s for c in range(NCHAN)] for s in range(NSPW)])})
    _create(ms / "DATA_DESCRIPTION", {"SPECTRAL_WINDOW_ID": _scalar("int"), "POLARIZATION_ID": _scalar("int")},
            NSPW, {"SPECTRAL_WINDOW_ID": np.arange(NSPW, dtype=np.int32),
                   "POLARIZATION_ID": np.zeros(NSPW, dtype=np.int32)})
    _create(ms / "POLARIZATION", {"NUM_CORR": _scalar("int"), "CORR_TYPE": _array("int", [2])}, 1,
            {"NUM_CORR": np.array([2], dtype=np.int32), "CORR_TYPE": np.array([[5, 8]], dtype=np.int32)})


@pytest.fixture
def ms(tmp_path):
    """A single-table MS: 4 antennas (with autocorrelations), 2 subbands x 8 channels, RR/LL, 3 scans."""
    path = tmp_path / "flag.ms"
    _write_main(path, _main_rows([1, 2, 3]), np.random.default_rng(5))
    _write_subtables(path)
    return path


@pytest.fixture
def mms(tmp_path):
    """The same layout as a Multi-MS of two parts (scans 1-2 and scan 3)."""
    path = tmp_path / "flag.mms"
    (path / "SUBMSS").mkdir(parents=True)
    rng = np.random.default_rng(5)
    _write_main(path / "SUBMSS" / "flag.mms.0000.ms", _main_rows([1, 2]), rng)
    _write_main(path / "SUBMSS" / "flag.mms.0001.ms", _main_rows([3]), rng)
    _write_subtables(path)
    return path


def _read(path):
    tb = casatools.table()
    tb.open(str(path))
    try:
        out = {name: np.asarray(tb.getcol(name)) for name in
               ("FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2", "TIME", "SCAN_NUMBER", "DATA_DESC_ID", "DATA")}
    finally:
        tb.close()
    out["FLAG"] = np.ascontiguousarray(out["FLAG"].T)
    out["DATA"] = np.ascontiguousarray(out["DATA"].T)
    return out


def _flag(path, **kwargs):
    setup = read_flag_setup(path)
    return apply_flag_selections(path, compile_flagdata(dict(kwargs, vis=str(path)), setup), setup=setup, workers=1)


def _stamp(seconds):
    return (dt.datetime(1858, 11, 17) + dt.timedelta(seconds=float(seconds))).strftime("%Y/%m/%d/%H:%M:%S.%f")[:-3]


# ----------------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------------
def test_parse_command_reads_quoted_and_bare_values():
    assert parse_command("antenna='EF' timerange=\"a~b\" autocorr=True reason='x y'") == {
        "antenna": "EF", "timerange": "a~b", "autocorr": "True", "reason": "x y"}
    with pytest.raises(Unsupported):
        parse_command("antenna='EF' stray")


def test_parse_spw_channel_ranges():
    selected = parse_spw("*:0~0;6~7,1:3", 2, 8)
    assert sorted(selected) == [0, 1]
    assert np.flatnonzero(selected[0]).tolist() == [0, 6, 7]
    assert np.flatnonzero(selected[1]).tolist() == [0, 3, 6, 7]
    assert parse_spw("1", 2, 8)[1].all()
    with pytest.raises(Unsupported):
        parse_spw("5", 2, 8)


def test_parse_timerange_and_antenna():
    (start, end), = parse_timerange("2025/11/05/06:03:48.5~06:04:09")
    assert end - start == pytest.approx(20.5)
    assert parse_antenna("EF,MC", NAMES) == ({0, 2}, set())
    assert parse_antenna("EF&JB;O6", NAMES) == ({3}, {(0, 1)})
    for bad in ("!EF", "EF&&JB", "EF&EF", "XX", "E*"):
        with pytest.raises(Unsupported):
            parse_antenna(bad, NAMES)


def test_modes_and_keys_outside_the_native_vocabulary_are_unsupported(ms):
    setup = read_flag_setup(ms)
    for kwargs in (dict(mode="tfcrop"), dict(mode="manual", uvrange=">1km"), dict(mode="manual", action="calculate"),
                   dict(mode="quack", quackinterval=4.0, quackmode="end")):
        with pytest.raises(Unsupported):
            compile_flagdata(kwargs, setup)


# ----------------------------------------------------------------------------------------------------------------
# Selections (CASA flagdata semantics)
# ----------------------------------------------------------------------------------------------------------------
def test_antenna_selects_its_cross_correlations_and_sets_flag_row(ms):
    before = _read(ms)
    _flag(ms, mode="manual", antenna="JB")
    after = _read(ms)
    jb = (before["ANTENNA1"] == 1) | (before["ANTENNA2"] == 1)
    cross = before["ANTENNA1"] != before["ANTENNA2"]
    assert after["FLAG"][jb & cross].all() and after["FLAG_ROW"][jb & cross].all()
    np.testing.assert_array_equal(after["FLAG"][~(jb & cross)], before["FLAG"][~(jb & cross)])  # autocorr kept


def test_timerange_keeps_centroids_inside_both_ends(ms):
    before = _read(ms)
    times = np.unique(before["TIME"])
    _flag(ms, mode="list", inpfile=[f"antenna='EF&MC' timerange='{_stamp(times[3])}~{_stamp(times[5])}'",
                                    f"antenna='EF&MC' timerange='{_stamp(times[7] + 0.5)}~{_stamp(times[9] - 0.5)}'"])
    after = _read(ms)
    baseline = (before["ANTENNA1"] == 0) & (before["ANTENNA2"] == 2)
    flagged = np.unique(after["TIME"][baseline & after["FLAG"].all(axis=(1, 2))])
    assert set(times[3:6]) <= set(flagged) and times[2] not in flagged and times[6] not in flagged
    assert times[8] in flagged and times[7] not in flagged and times[9] not in flagged


def test_channel_and_correlation_selections_leave_flag_row_alone(ms):
    before = _read(ms)
    _flag(ms, mode="manual", antenna="MC", spw="1", correlation="RR,LL")         # a whole row, but via spw/corr
    _flag(ms, mode="manual", antenna="O6", spw="0:2~3", correlation="LL")
    after = _read(ms)
    mc = ((before["ANTENNA1"] == 2) | (before["ANTENNA2"] == 2)) & (before["ANTENNA1"] != before["ANTENNA2"])
    assert after["FLAG"][mc & (before["DATA_DESC_ID"] == 1)].all()
    assert not after["FLAG_ROW"].any()
    o6 = ((before["ANTENNA1"] == 3) | (before["ANTENNA2"] == 3)) & (before["ANTENNA1"] != before["ANTENNA2"])
    rows = o6 & (before["DATA_DESC_ID"] == 0)
    assert after["FLAG"][rows][:, 2:4, 1].all()
    changed = after["FLAG"] != before["FLAG"]
    changed[rows, 2:4, 1] = False
    changed[mc & (before["DATA_DESC_ID"] == 1)] = False
    assert not changed.any()


def test_unflag_whole_rows_clears_flag_row_but_a_channel_unflag_does_not(ms):
    _flag(ms, mode="manual", antenna="EF")
    _flag(ms, mode="unflag", antenna="EF", spw="0")
    after = _read(ms)
    ef = ((after["ANTENNA1"] == 0) | (after["ANTENNA2"] == 0)) & (after["ANTENNA1"] != after["ANTENNA2"])
    spw0 = after["DATA_DESC_ID"] == 0
    assert not after["FLAG"][ef & spw0].any() and after["FLAG_ROW"][ef & spw0].all()   # as flagdata leaves it
    _flag(ms, mode="unflag", antenna="EF")
    after = _read(ms)
    assert not after["FLAG"][ef].any() and not after["FLAG_ROW"][ef].any()


def test_autocorr_flags_only_the_autocorrelations(ms):
    before = _read(ms)
    _flag(ms, mode="manual", autocorr=True)
    after = _read(ms)
    auto = before["ANTENNA1"] == before["ANTENNA2"]
    assert after["FLAG"][auto].all()
    np.testing.assert_array_equal(after["FLAG"][~auto], before["FLAG"][~auto])


def test_quack_counts_from_the_first_selected_time_of_each_scan(ms):
    before = _read(ms)
    times = np.unique(before["TIME"][before["SCAN_NUMBER"] == 1])
    _flag(ms, mode="quack", quackinterval=4.0, antenna="MC", scan="1",
          timerange=f"{_stamp(times[3])}~{_stamp(times[-1])}")
    after = _read(ms)
    mc = ((before["ANTENNA1"] == 2) | (before["ANTENNA2"] == 2)) & (before["ANTENNA1"] != before["ANTENNA2"])
    flagged = np.unique(after["TIME"][mc & after["FLAG"].all(axis=(1, 2)) & ~before["FLAG"].all(axis=(1, 2))])
    assert flagged.tolist() == times[3:6].tolist()            # start + 0, 2 and 4 s: the end is included


def test_quack_with_no_antenna_includes_the_autocorrelations_on_every_part(mms):
    _flag(mms, mode="quack", quackinterval=2.0)
    for part in sorted((mms / "SUBMSS").glob("*.ms")):
        after = _read(part)
        for scan in np.unique(after["SCAN_NUMBER"]):
            rows = after["SCAN_NUMBER"] == scan
            early = rows & (after["TIME"] <= after["TIME"][rows].min() + 2.0)
            assert after["FLAG"][early].all() and after["FLAG_ROW"][early].all()
            assert not after["FLAG_ROW"][rows & ~early].any()


def test_list_of_commands_equals_applying_them_one_by_one(ms, tmp_path):
    import shutil
    other = tmp_path / "copy.ms"
    shutil.copytree(ms, other)
    times = np.unique(_read(ms)["TIME"])
    commands = [f"antenna='EF' timerange='{_stamp(times[k])}~{_stamp(times[k + 1])}'" for k in (0, 1, 5, 20)]
    commands += ["antenna='JB' spw='1:0~1'", "antenna='MC&O6' scan='3'", "antenna='O6' field='CAL'"]
    _flag(ms, mode="list", inpfile=commands)
    for command in commands:
        _flag(other, **parse_command(command))
    np.testing.assert_array_equal(_read(ms)["FLAG"], _read(other)["FLAG"])
    np.testing.assert_array_equal(_read(ms)["FLAG_ROW"], _read(other)["FLAG_ROW"])


# ----------------------------------------------------------------------------------------------------------------
# Statistics and amplitudes
# ----------------------------------------------------------------------------------------------------------------
def test_count_flags_counts_observable_cross_correlations_and_caches_the_zero_rows(mms):
    counts = count_flags(mms, workers=1)
    flagged = observable = 0
    for part in sorted((mms / "SUBMSS").glob("*.ms")):
        main = _read(part)
        cross = main["ANTENNA1"] != main["ANTENNA2"]
        recorded = main["DATA"] != 0
        flagged += int((main["FLAG"] & recorded)[cross].sum())
        observable += int(recorded[cross].sum())
        assert (part / flagdata.OBSERVABLE_CACHE).is_file()
    assert int(counts["flagged"].sum()) == flagged and int(counts["observable"].sum()) == observable
    assert counts["observable"][2, 3].sum() < counts["observable"][1, 3].sum()     # 2&3 not recorded in scan 2
    again = count_flags(mms, workers=1)
    np.testing.assert_array_equal(again["flagged"], counts["flagged"])


def test_row_amplitudes_average_the_unflagged_samples(ms):
    rows = read_row_amplitudes(ms, fields=["CAL"], workers=1)
    main = _read(ms)
    keep = (main["ANTENNA1"] != main["ANTENNA2"]) & (main["SCAN_NUMBER"] % 2 == 1)
    amplitude = np.where(main["FLAG"], np.nan, np.abs(main["DATA"].astype(np.complex128)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # rows with every sample flagged
        expected = np.nanmean(amplitude, axis=(1, 2))
    keep &= np.isfinite(expected)
    order = np.lexsort((main["TIME"][keep], main["DATA_DESC_ID"][keep]))
    np.testing.assert_allclose(rows["amplitude"], expected[keep][order].astype(np.float32), rtol=1e-6)
    np.testing.assert_array_equal(rows["time"], main["TIME"][keep][order])


# ----------------------------------------------------------------------------------------------------------------
# Backend wiring
# ----------------------------------------------------------------------------------------------------------------
def test_native_summary_matches_the_taql_summary(ms, monkeypatch):
    pytest.importorskip("daskms")
    from vlbipy.backends.casa import CasaFlagOps
    from vlbipy.backends.dask_ms import DaskMsBackend
    backend = DaskMsBackend(str(Path(ms).parent))
    monkeypatch.setattr(backend, "ms_path", lambda code: Path(ms))
    assert backend.flag.summary("p") == CasaFlagOps(backend).summary("p")
    assert backend.flag.flagged_fraction("p") == pytest.approx(CasaFlagOps(backend).flagged_fraction("p"))


def test_flagdata_falls_back_to_casa_for_what_it_cannot_do_natively(ms, monkeypatch):
    pytest.importorskip("daskms")
    from vlbipy.backends.dask_ms import NativeFlagTasks
    calls = []

    class Tasks:
        def flagdata(self, **kwargs):
            calls.append(kwargs)

    tasks = NativeFlagTasks(Tasks())
    tasks.flagdata(vis=str(ms), mode="manual", antenna="EF", flagbackup=False)
    assert calls == []
    tasks.flagdata(vis=str(ms), mode="tfcrop", datacolumn="data", flagbackup=False)
    assert calls and calls[0]["mode"] == "tfcrop"
