"""Fringe-fit sub-arrays: stations that see a resolved calibrator only among themselves."""
from types import SimpleNamespace

import numpy as np
import pytest

from vlbipy.models import CalTable
from vlbipy.subarrays import (SUBARRAY_FILE, cross_mask, flag_cross_subarrays, groups_for_field, read_subarrays,
                              station_clusters, write_subarrays)

from test_backends import _FakeCalTables, _fringe_store

#: AA and BB far apart (the main array); CC and DD 50 km from each other and thousands of km from the rest.
POSITIONS = {"AA": (6.4e6, 0.0, 0.0), "BB": (6.4e6, 1.0e6, 0.0), "CC": (0.0, 6.4e6, 0.0), "DD": (0.0, 6.4e6, 5.0e4)}


def test_station_clusters_link_nearby_stations_only():
    e_merlin = {"JB": (3822626.0, -154105.0, 5086486.0), "PI": (3817549.0, -163031.0, 5089896.0),
                "DA": (3828714.0, -169453.0, 5080649.0), "CM": (3920356.0, 2542.0, 5014284.0)}
    far = {"EF": (4033947.0, 486990.0, 4900431.0), "HH": (5085442.0, 2668263.0, -2768697.0)}
    clusters = station_clusters({**e_merlin, **far, "XX": (0.0, 0.0, 0.0)}, 300.0)
    assert clusters == [["JB", "PI", "DA", "CM"]]                     # EF is 600 km away; XX has no position
    assert station_clusters({**e_merlin, **far}, 0.0) == []


def test_cross_mask_flags_baselines_between_groups_only():
    names = ["AA", "BB", "CC", "DD"]
    a1, a2 = np.array([0, 0, 2, 1, 0]), np.array([1, 2, 3, 3, 3])
    assert list(cross_mask(a1, a2, names, [[["CC", "DD"]]])) == [False, True, False, True, True]
    assert not cross_mask(a1, a2, names, []).any()


def test_groups_follow_the_solutions_each_field_is_calibrated_with(tmp_path):
    mbd = tmp_path / "X.mbd_smooth"
    mbd.mkdir()
    write_subarrays(mbd, {"PC": [["CC", "DD"]], "FF": [["CC"]]})        # a one-antenna group is no sub-array
    assert read_subarrays(mbd) == {"PC": [["CC", "DD"]]}
    table = CalTable(cal_type="mbd", path=str(mbd), field="FF,PC", gainfield="PC")
    tsys = CalTable(cal_type="tsys", path=str(tmp_path / "X.tsys"))
    assert groups_for_field([tsys, table], "TGT") == [[["CC", "DD"]]]  # transferred from PC
    assert groups_for_field([tsys, table], "PC") == [[["CC", "DD"]]]   # its own solutions
    assert groups_for_field([tsys, table], "FF") == []                 # its own solutions: one array
    write_subarrays(mbd, {})
    assert not (mbd / SUBARRAY_FILE).exists() and groups_for_field([table], "TGT") == []


def _tiny_ms(path, names, antenna1, antenna2):
    from casacore.tables import makearrcoldesc, makescacoldesc, maketabdesc, table
    nrow = len(antenna1)
    desc = maketabdesc([makescacoldesc("ANTENNA1", 0), makescacoldesc("ANTENNA2", 0),
                        makescacoldesc("FLAG_ROW", False), makearrcoldesc("FLAG", False, ndim=2)])
    with table(str(path), desc, nrow=nrow, ack=False) as main:
        main.putcol("ANTENNA1", np.asarray(antenna1, np.int32))
        main.putcol("ANTENNA2", np.asarray(antenna2, np.int32))
        main.putcol("FLAG_ROW", np.zeros(nrow, bool))
        main.putcol("FLAG", np.zeros((nrow, 4, 2), bool))
    with table(str(path / "ANTENNA"), maketabdesc([makescacoldesc("NAME", "")]), nrow=len(names), ack=False) as ant:
        ant.putcol("NAME", list(names))


def test_a_split_loses_the_baselines_between_sub_arrays(tmp_path):
    pytest.importorskip("casacore.tables")
    from casacore.tables import table
    from vlbipy.namespaces import _flag_split_subarrays
    ms = tmp_path / "X_TGT.ms"
    _tiny_ms(ms, ["AA", "BB", "CC", "DD"], [0, 0, 2, 1, 0], [1, 2, 3, 3, 3])
    mbd = tmp_path / "X.mbd"
    mbd.mkdir()
    write_subarrays(mbd, {"PC": [["CC", "DD"]]})
    obs = SimpleNamespace(project_code="X", gaintables=[CalTable(cal_type="mbd", path=str(mbd), field="FF,PC",
                                                                 gainfield="PC")])
    assert _flag_split_subarrays(obs, "TGT", ms) == 3
    with table(str(ms), ack=False) as main:
        assert list(main.getcol("FLAG_ROW")) == [False, True, False, True, True]
        assert list(main.getcol("FLAG").all(axis=(1, 2))) == [False, True, False, True, True]
    assert (tmp_path / "X_TGT.ms.flags0.npz").is_file()                 # the reference for later flag transfers
    assert _flag_split_subarrays(obs, "FF", ms) == 0
    assert flag_cross_subarrays(ms, [[["CC", "DD"]]]) == 0              # already flagged


def _sub_array_case(tmp_path, monkeypatch, dd_delays):
    """AA (reference) and BB detect CAL; CC and DD, 50 km apart, see it only between themselves."""
    from vlbipy.backends.casa import CasaCalibrationOps
    from vlbipy.models import Antenna, ObsMetadata, Scan
    ops = CasaCalibrationOps.__new__(CasaCalibrationOps)
    ops._backend = type("B", (), {"tools": type("T", (), {"table": staticmethod(_FakeCalTables)})()})()
    monkeypatch.setattr(ops, "_table_antenna_names", lambda path: ["AA", "BB", "CC", "DD"])
    main, retry, sub = tmp_path / "X.mbd", tmp_path / "X.mbd.retry", tmp_path / "X.mbd.subarray"
    main.mkdir()
    scans = [1, 2, 3, 4]
    antennas = np.repeat([0, 1, 2, 3], 4)
    _FakeCalTables.stores = {str(main): _fringe_store(antennas, scans * 4, [1] * 8 + [0] * 8, fparam=1.0,
                                                      delays=[0.0] * 4 + [1.0] * 4 + [0.0] * 8)}
    meta = ObsMetadata.__new__(ObsMetadata)
    meta.scans = [Scan(scan_number=n, source="CAL", antennas=["AA", "BB", "CC", "DD"]) for n in scans]
    meta.antennas = {name: Antenna(name=name, position=xyz, observed=True) for name, xyz in POSITIONS.items()}
    calls = []

    def fake_fringefit(project_code, params, gaintable, *, field="", callib=False):
        calls.append(dict(params))
        if params["caltable"] == str(retry):                            # the SNR-3 retry finds nothing more
            retry.mkdir()
            _FakeCalTables.stores[str(retry)] = _fringe_store(antennas, scans * 4, [0] * 16)
            return
        sub.mkdir()
        store = _fringe_store(antennas, scans * 4, [0] * 8 + [1] * 8, fparam=2.0,
                              delays=[0.0] * 12 + list(dd_delays))
        store["ANTENNA2"] = np.full(16, 2)                              # referenced to CC
        _FakeCalTables.stores[str(sub)] = store

    monkeypatch.setattr(ops, "_fringefit", fake_fringefit)
    adopted = ops._retry_weak_antennas("X", {"caltable": str(main), "minsnr": 5.0, "refant": "AA,BB,CC,DD",
                                             "field": "CAL"}, [], meta, field="CAL", callib=False, minsnr=3.0,
                                       keep_fraction=0.5, max_delay_step=2.0, subarray_km=300.0)
    return adopted, calls, _FakeCalTables.stores[str(main)], main, sub


def test_antennas_that_see_the_calibrator_only_among_themselves_become_a_sub_array(tmp_path, monkeypatch):
    adopted, calls, store, main, sub = _sub_array_case(tmp_path, monkeypatch, [3.0, 3.1, 3.2, 3.1])
    assert adopted == []
    assert calls[-1]["antenna"] == "CC,DD&" and calls[-1]["refant"] == "CC,DD"
    assert calls[-1]["field"] == "CAL" and calls[-1]["scan"] == "1,2,3,4"
    solved = ~store["FLAG"].reshape(-1, 16).all(axis=0)
    assert solved.all()                                                 # CC and DD now have solutions
    assert list(store["ANTENNA2"][8:]) == [2] * 8 and list(store["FPARAM"][0, 0, 8:]) == [2.0] * 8
    assert list(store["FPARAM"][0, 0, :8]) == [1.0] * 8                 # the main array is untouched
    assert read_subarrays(main) == {"CAL": [["CC", "DD"]]} and not sub.exists()


def test_a_sub_array_of_noise_is_not_adopted(tmp_path, monkeypatch):
    adopted, calls, store, main, sub = _sub_array_case(tmp_path, monkeypatch, [80.0, -120.0, 30.0, 140.0])
    assert calls[-1]["antenna"] == "CC,DD&"
    assert store["FLAG"].reshape(-1, 16).all(axis=0)[8:].all()          # CC and DD stay without solutions
    assert read_subarrays(main) == {} and not sub.exists()
