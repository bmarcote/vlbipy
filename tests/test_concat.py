"""Native concat (solvers/concat.py): a synthetic two-epoch case and a comparison with casatasks.concat."""
import filecmp
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vlbipy.casacore_format.aipsio import TableRef
from vlbipy.casacore_format.table import Table
from vlbipy.casacore_format.writer import array_column, create_table, scalar_column
from vlbipy.solvers.concat import concat

RSM07_MMS = Path(__file__).resolve().parent.parent / "test" / "RSM07_import_ref" / "RSM07.mms"
NCHAN, NCORR = 4, 4
DAY = 86400.0


def _write_ms(path: Path, antennas: list, t0: float, seed: int) -> dict:
    """A small plain MS: ``antennas`` [(name, x)], 2 subbands x 4 channels, RR/RL/LR/LL, one field, 2 scans."""
    rng = np.random.default_rng(seed)
    nant = len(antennas)
    baselines = [(i, j) for i in range(nant) for j in range(i + 1, nant)]
    rows = [(t0 + 10.0 * k, scan, dd, i, j) for k, scan in ((0, 1), (1, 1), (2, 2)) for dd in (0, 1)
            for i, j in baselines]
    nrow = len(rows)
    time, scan, ddid, ant1, ant2 = (np.array([r[c] for r in rows]) for c in range(5))
    values = {"TIME": time, "TIME_CENTROID": time, "INTERVAL": np.full(nrow, 10.0), "EXPOSURE": np.full(nrow, 10.0),
              "ANTENNA1": ant1.astype(np.int32), "ANTENNA2": ant2.astype(np.int32),
              "FEED1": np.zeros(nrow, np.int32), "FEED2": np.zeros(nrow, np.int32),
              "DATA_DESC_ID": ddid.astype(np.int32), "SCAN_NUMBER": scan.astype(np.int32),
              "FIELD_ID": np.zeros(nrow, np.int32), "ARRAY_ID": np.zeros(nrow, np.int32),
              "OBSERVATION_ID": np.zeros(nrow, np.int32), "PROCESSOR_ID": np.full(nrow, -1, np.int32),
              "STATE_ID": np.full(nrow, -1, np.int32), "FLAG_ROW": np.zeros(nrow, bool),
              "UVW": rng.normal(size=(nrow, 3)) * 1e6,
              "DATA": (rng.normal(size=(nrow, NCHAN, NCORR)) + 1j * rng.normal(size=(nrow, NCHAN, NCORR))).astype(
                  np.complex64),
              "FLAG": rng.random((nrow, NCHAN, NCORR)) < 0.1,
              "WEIGHT": rng.random((nrow, NCORR)).astype(np.float32),
              "SIGMA": rng.random((nrow, NCORR)).astype(np.float32)}
    columns = [scalar_column(n, "double") for n in ("TIME", "TIME_CENTROID", "INTERVAL", "EXPOSURE")]
    columns += [scalar_column(n, "int") for n in ("ANTENNA1", "ANTENNA2", "FEED1", "FEED2", "DATA_DESC_ID",
                                                  "SCAN_NUMBER", "FIELD_ID", "ARRAY_ID", "OBSERVATION_ID",
                                                  "PROCESSOR_ID", "STATE_ID")]
    columns += [scalar_column("FLAG_ROW", "boolean"), array_column("UVW", "double", shape=(3,)),
                array_column("DATA", "complex", ndim=2), array_column("FLAG", "boolean", ndim=2),
                array_column("WEIGHT", "float", ndim=1), array_column("SIGMA", "float", ndim=1)]
    subs = ["ANTENNA", "FEED", "FIELD", "SPECTRAL_WINDOW", "POLARIZATION", "DATA_DESCRIPTION", "OBSERVATION"]
    tiled = {"DATA": (NCHAN, NCORR), "FLAG": (NCHAN, NCORR), "WEIGHT": (NCORR,), "SIGMA": (NCORR,)}
    create_table(path, columns, nrow, values, keywords={s: TableRef(f"././{s}") for s in subs},
                 info=("Measurement Set", ""), tiled=tiled)
    create_table(path / "ANTENNA", [scalar_column("NAME", "string"), scalar_column("STATION", "string"),
                                    array_column("POSITION", "double", shape=(3,)),
                                    scalar_column("FLAG_ROW", "boolean")],
                 nant, {"NAME": [a[0] for a in antennas], "STATION": [a[0] for a in antennas],
                        "POSITION": np.array([[x, 0.0, 0.0] for _, x in antennas]), "FLAG_ROW": np.zeros(nant, bool)})
    nfeed = nant * 2
    create_table(path / "FEED", [scalar_column(n, "int") for n in ("ANTENNA_ID", "FEED_ID", "SPECTRAL_WINDOW_ID",
                                                                     "NUM_RECEPTORS")]
                 + [scalar_column("TIME", "double"), scalar_column("INTERVAL", "double"),
                    array_column("POSITION", "double", shape=(3,)), array_column("BEAM_OFFSET", "double", ndim=2),
                    array_column("POL_RESPONSE", "complex", ndim=2), array_column("RECEPTOR_ANGLE", "double", ndim=1),
                    array_column("POLARIZATION_TYPE", "string", ndim=1)], nfeed,
                 {"ANTENNA_ID": np.repeat(np.arange(nant), 2).astype(np.int32), "FEED_ID": np.zeros(nfeed, np.int32),
                  "SPECTRAL_WINDOW_ID": np.tile([0, 1], nant).astype(np.int32),
                  "NUM_RECEPTORS": np.full(nfeed, 2, np.int32), "TIME": np.full(nfeed, t0),
                  "INTERVAL": np.full(nfeed, 100.0), "POSITION": np.zeros((nfeed, 3)),
                  "BEAM_OFFSET": [np.zeros((2, 2))] * nfeed, "POL_RESPONSE": [np.eye(2, dtype=np.complex64)] * nfeed,
                  "RECEPTOR_ANGLE": [np.zeros(2)] * nfeed, "POLARIZATION_TYPE": [np.array(["R", "L"], object)] * nfeed})
    direction = np.array([[1.0, 0.5]])
    create_table(path / "FIELD", [scalar_column("NAME", "string"), scalar_column("FLAG_ROW", "boolean"),
                                  scalar_column("NUM_POLY", "int")]
                 + [array_column(n, "double", ndim=2) for n in ("REFERENCE_DIR", "DELAY_DIR", "PHASE_DIR")], 1,
                 {"NAME": ["SRC"], "FLAG_ROW": np.zeros(1, bool), "NUM_POLY": np.zeros(1, np.int32),
                  "REFERENCE_DIR": [direction], "DELAY_DIR": [direction], "PHASE_DIR": [direction]})
    freqs = [1.6e9 + 32e6 * s + 8e6 * np.arange(NCHAN) for s in range(2)]
    create_table(path / "SPECTRAL_WINDOW", [array_column("CHAN_FREQ", "double", ndim=1),
                                            scalar_column("REF_FREQUENCY", "double"), scalar_column("NUM_CHAN", "int"),
                                            scalar_column("TOTAL_BANDWIDTH", "double"),
                                            scalar_column("IF_CONV_CHAIN", "int"),
                                            scalar_column("FLAG_ROW", "boolean")],
                 2, {"CHAN_FREQ": freqs, "REF_FREQUENCY": np.array([f[0] for f in freqs]),
                     "NUM_CHAN": np.full(2, NCHAN, np.int32), "TOTAL_BANDWIDTH": np.full(2, 32e6),
                     "IF_CONV_CHAIN": np.zeros(2, np.int32), "FLAG_ROW": np.zeros(2, bool)})
    create_table(path / "POLARIZATION", [array_column("CORR_TYPE", "int", ndim=1),
                                         array_column("CORR_PRODUCT", "int", ndim=2), scalar_column("NUM_CORR", "int"),
                                         scalar_column("FLAG_ROW", "boolean")], 1,
                 {"CORR_TYPE": [np.array([5, 6, 7, 8], np.int32)],
                  "CORR_PRODUCT": [np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.int32)],
                  "NUM_CORR": np.array([4], np.int32), "FLAG_ROW": np.zeros(1, bool)})
    create_table(path / "DATA_DESCRIPTION", [scalar_column("SPECTRAL_WINDOW_ID", "int"),
                                             scalar_column("POLARIZATION_ID", "int"),
                                             scalar_column("FLAG_ROW", "boolean")],
                 2, {"SPECTRAL_WINDOW_ID": np.array([0, 1], np.int32), "POLARIZATION_ID": np.zeros(2, np.int32),
                     "FLAG_ROW": np.zeros(2, bool)})
    create_table(path / "OBSERVATION", [array_column("TIME_RANGE", "double", shape=(2,)),
                                        scalar_column("FLAG_ROW", "boolean"), scalar_column("OBSERVER", "string"),
                                        scalar_column("PROJECT", "string"), scalar_column("RELEASE_DATE", "double"),
                                        scalar_column("TELESCOPE_NAME", "string")], 1,
                 {"TIME_RANGE": np.array([[t0, t0 + 30.0]]), "FLAG_ROW": np.zeros(1, bool), "OBSERVER": ["me"],
                  "PROJECT": ["P1"], "RELEASE_DATE": np.zeros(1), "TELESCOPE_NAME": ["VLBI"]})
    return values


def _read(path: Path, names: list) -> dict:
    with Table(path) as table:
        return {n: np.asarray(table.getcol(n)) for n in names}


def test_synthetic_two_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")
    t0 = 5.0e9
    first = _write_ms(tmp_path / "e1.ms", [("AA", 1e6), ("BB", 2e6), ("CC", 3e6)], t0, seed=1)
    # Epoch 2 lists CC first (its baselines to AA/BB come out swapped) and BB moved by 0.5 m (still matched).
    second = _write_ms(tmp_path / "e2.ms", [("CC", 3e6), ("AA", 1e6), ("BB", 2e6 + 0.5)], t0 + 30 * DAY, seed=2)
    # Given in reverse time order: concat processes them by their first TIME.
    report = concat([tmp_path / "e2.ms", tmp_path / "e1.ms"], tmp_path / "out.ms", scales=[2.0, 1.0])
    assert report["rows"] == 2 * len(first["TIME"]) and report["antennas"] == 3 and report["observations"] == 2
    assert report["spectral_windows"] == 2 and report["fields"] == 1
    out = _read(tmp_path / "out.ms", ["ANTENNA1", "ANTENNA2", "SCAN_NUMBER", "OBSERVATION_ID", "UVW", "DATA",
                                      "WEIGHT", "SIGMA", "TIME"])
    n = len(first["TIME"])
    np.testing.assert_array_equal(out["DATA"][:n], first["DATA"])          # epoch 1 copied unchanged
    assert set(out["SCAN_NUMBER"][n:]) == {3, 4} and set(out["OBSERVATION_ID"][n:]) == {1}
    ant_map = np.array([2, 0, 1])
    a1, a2 = ant_map[second["ANTENNA1"]], ant_map[second["ANTENNA2"]]
    swap = a1 > a2
    assert swap.any() and (~swap).any()
    np.testing.assert_array_equal(out["ANTENNA1"][n:], np.minimum(a1, a2))
    np.testing.assert_array_equal(out["UVW"][n:][swap], -second["UVW"][swap])
    expected = second["DATA"].copy()
    expected[swap] = np.conj(expected[swap][..., [0, 2, 1, 3]])
    expected = (expected.astype(np.complex128) * 2.0).astype(np.complex64)
    np.testing.assert_array_equal(out["DATA"][n:], expected)
    weight = second["WEIGHT"].copy()
    weight[swap] = weight[swap][:, [0, 2, 1, 3]]
    np.testing.assert_array_equal(out["WEIGHT"][n:], (weight.astype(np.float64) * 0.25).astype(np.float32))
    with Table(tmp_path / "out.ms" / "FEED") as feed:
        # Identical feeds match (casacore compares them under the appended MS's antenna index, which is AA, BB
        # or CC here alike): no row is added, but every validity range grows to cover both epochs.
        assert feed.nrows() == 6
        assert np.all(np.asarray(feed.getcol("INTERVAL")) > 30 * DAY)


def test_multi_ms_is_left_to_casa(tmp_path):
    (tmp_path / "x.ms" / "SUBMSS").mkdir(parents=True)
    with pytest.raises(NotImplementedError):
        concat([tmp_path / "x.ms", tmp_path / "x.ms"], tmp_path / "out.ms")


def _shift_epoch(source: Path, dest: Path, seconds: float) -> None:
    """Clone a split MS and move it ``seconds`` later (main TIME/TIME_CENTROID and OBSERVATION TIME_RANGE)."""
    subprocess.run(["cp", "-cR", str(source), str(dest)], check=True)
    with Table(dest, readonly=False) as table:
        for name in ("TIME", "TIME_CENTROID"):
            table.putcol(name, np.asarray(table.getcol(name)) + seconds)
    with Table(dest / "OBSERVATION", readonly=False) as table:
        table.putcol("TIME_RANGE", np.asarray(table.getcol("TIME_RANGE")) + seconds)


def test_rsm07_concat_matches_casa(tmp_path, monkeypatch):
    pytest.importorskip("casatasks")
    if not RSM07_MMS.is_dir():
        pytest.skip("RSM07 reference data not available")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VLBIPY_TABLE_ENGINE", "native")
    from casatasks import concat as casa_concat

    from vlbipy.solvers.split import split
    from vlbipy.solvers.uvfits import exportuvfits
    split(RSM07_MMS, tmp_path / "e1.ms", "3C345", column="DATA", chanbin=4, timebin=8.0, workers=4)
    _shift_epoch(tmp_path / "e1.ms", tmp_path / "e2.ms", 30 * DAY)
    inputs = [str(tmp_path / "e1.ms"), str(tmp_path / "e2.ms")]
    casa_concat(vis=inputs, concatvis=str(tmp_path / "casa.ms"), respectname=True, copypointing=False)
    concat(inputs, tmp_path / "native.ms")
    names = ["TIME", "ANTENNA1", "ANTENNA2", "SCAN_NUMBER", "OBSERVATION_ID", "DATA_DESC_ID", "FIELD_ID", "UVW",
             "DATA", "FLAG", "WEIGHT", "SIGMA", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM", "EXPOSURE", "TIME_CENTROID"]
    casa, native = _read(tmp_path / "casa.ms", names), _read(tmp_path / "native.ms", names)
    for name in names:
        np.testing.assert_array_equal(native[name], casa[name], err_msg=name)
    for sub in ("ANTENNA", "FEED", "OBSERVATION", "SPECTRAL_WINDOW", "EARTH_ORIENTATION", "GAIN_CURVE", "SYSCAL"):
        with Table(tmp_path / "casa.ms" / sub) as a, Table(tmp_path / "native.ms" / sub) as b:
            assert a.nrows() == b.nrows(), sub
    exportuvfits(tmp_path / "casa.ms", tmp_path / "casa.uvfits")
    exportuvfits(tmp_path / "native.ms", tmp_path / "native.uvfits")
    assert filecmp.cmp(tmp_path / "casa.uvfits", tmp_path / "native.uvfits", shallow=False)
    shutil.rmtree(tmp_path / "casa.ms")
