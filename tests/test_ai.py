"""AI mode (vlbipy.ai): antenna decomposition, noise estimation, Tsys assessment and repair plan,
distribution tests, the decision forest and the rule base - on synthetic data with known answers."""
import numpy as np
import pytest

from vlbipy.ai import antenna, distributions, features, knowledge, model, tsys, visstats


def test_solve_antenna_terms_recovers_gains_and_rejects_a_bad_baseline():
    rng = np.random.default_rng(1)
    nant = 6
    truth = np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.05])
    a1, a2 = np.triu_indices(nant, 1)
    values = truth[a1] + truth[a2] + rng.normal(0, 0.01, a1.size)
    values[3] += 1.0                                       # one baseline-based problem
    res = antenna.solve_antenna_terms(values, np.full(a1.size, 1e4), a1, a2, nant)
    assert np.allclose(res["x"], truth, atol=0.03)
    assert np.nanargmax(np.abs(res["resid"])) == 3         # reported, not forced onto the stations


def _cube(nscan=6, nbin=10, nant=4, nspw=2, amp=1.0, sigma=0.05, gains=None, hand_ratio=None, seed=0):
    """A calibrated cube: point source of ``amp``, antenna amplitude gains (per scan) and hand ratios."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(nant, 1)
    rows = []
    for s in range(nscan):
        for b in range(a1.size):
            for spw in range(nspw):
                for t in range(nbin):
                    rows.append((s, a1[b], a2[b], spw, t))
    rows = np.array(rows)
    n = rows.shape[0]
    g = np.ones((nscan, nant)) if gains is None else np.asarray(gains)
    hr = np.zeros(nant) if hand_ratio is None else np.asarray(hand_ratio)
    signal = amp * g[rows[:, 0], rows[:, 1]] * g[rows[:, 0], rows[:, 2]]
    vec = np.stack([signal * np.exp(0.5 * (hr[rows[:, 1]] + hr[rows[:, 2]])),
                    signal * np.exp(-0.5 * (hr[rows[:, 1]] + hr[rows[:, 2]]))], axis=1).astype(complex)
    vec += sigma * (rng.normal(size=vec.shape) + 1j * rng.normal(size=vec.shape)) / np.sqrt(2)
    count = np.full((n, 2), 100.0)
    return {"field": np.zeros(n, int), "scan": rows[:, 0], "ant1": rows[:, 1], "ant2": rows[:, 2], "spw": rows[:, 3],
            "tbin": rows[:, 0] * 1000 + rows[:, 4], "time": rows[:, 0] * 600.0 + rows[:, 4] * 10.0,
            "uvw": np.zeros((n, 3)), "vec": vec, "power": np.abs(vec) ** 2 + sigma ** 2, "count": count,
            "noise2": np.full((n, 2), sigma ** 2 * 100), "noise2_avg": np.full((n, 2), sigma ** 2),
            "antennas": np.array([f"A{i}" for i in range(nant)]), "field_names": np.array(["CAL"]),
            "spw_freq": np.arange(nspw) * 32e6 + 5e9, "hands": np.array(["RR", "LL"]),
            "column": np.array("CORRECTED_DATA"), "binsec": np.array(10.0), "rho1": np.array([0.0, 0.0])}


def test_gains_vs_time_finds_a_sensitivity_drop_and_hand_ratios():
    gains = np.ones((6, 4))
    gains[3:, 2] = 0.5                                     # antenna 2 loses sensitivity half way
    cube = _cube(gains=gains, hand_ratio=[0.0, 0.1, 0.0, 0.0])
    g = antenna.gains_vs_time(cube, coherent=True)["CAL"]
    x2 = np.nanmean(g["x"][:, 2, :], axis=1)
    assert np.allclose(np.exp(x2[3:] - x2[:3].mean()), 0.5, atol=0.03)
    hr = antenna.hand_ratios(cube, coherent=True, per_spw=False)
    assert hr["x"][0, 1] == pytest.approx(0.1, abs=0.01) and abs(hr["x"][0, 0]) < 0.01


def test_noise_from_time_differences_handles_correlated_channels():
    """Adjacent-channel correlation 0.5 (SFXC): time differences give the per-sample variance, rho1 the rest."""
    rng = np.random.default_rng(2)
    nrow, nchan = 400, 40
    white = (rng.normal(size=(nrow, nchan + 1, 1)) + 1j * rng.normal(size=(nrow, nchan + 1, 1))) / np.sqrt(2)
    v = (white[:, 1:] + white[:, :-1]) / np.sqrt(2)          # unit variance, rho1 = 0.5
    keys = np.repeat(np.arange(20), 20)
    unique = np.arange(20)
    moments = visstats._time_differences(v, np.ones(v.shape, bool), keys, unique)
    sums = {"t1sum": moments["t1sum"], "t1n": moments["t1n"], "t2sum": moments["t2sum"], "t2n": moments["t2n"],
            "d2sum": np.ones((20, 1)), "d2n": np.ones((20, 1))}
    noise2 = visstats._noise2(sums, "DATA")
    assert np.median(noise2) == pytest.approx(1.0, rel=0.1)
    rho1 = np.median((moments["t1c"] / moments["t1cn"]) / (moments["t1sum"] / moments["t1n"]))
    assert rho1 == pytest.approx(0.5, abs=0.05)


def _syscal():
    """Two antennas, two subbands: AA steady both hands; BB RCP noisy, BB spw 1 a nominal constant."""
    rng = np.random.default_rng(3)
    n = 300
    t = np.arange(n) * 60.0
    rows = []
    for antenna_id in (0, 1):
        for spw in (0, 1):
            base = 50 + 5 * np.sin(t / 5000.0)
            r = base * (1 + 0.003 * rng.normal(size=n))
            ll = base * (1 + 0.003 * rng.normal(size=n))
            if antenna_id == 1 and spw == 0:
                r = base * (1 + 0.15 * rng.normal(size=n))
            if antenna_id == 1 and spw == 1:
                r = np.full(n, 26.0)
                ll = np.full(n, 26.0)
            rows.append((np.full(n, antenna_id), np.full(n, spw), t, np.stack([r, ll], 1)))
    return {"antenna": np.concatenate([r[0] for r in rows]), "spw": np.concatenate([r[1] for r in rows]),
            "time": np.concatenate([r[2] for r in rows]), "tsys": np.concatenate([r[3] for r in rows]),
            "scan": np.concatenate([np.repeat(np.arange(30), 10)] * 4),
            "field": np.concatenate([np.repeat(np.arange(30) % 2, 10)] * 4), "antennas": np.array(["AA", "BB"])}


def test_tsys_assess_and_plan():
    syscal = _syscal()
    findings = tsys.assess(syscal)
    labels = {(f["antenna"], f["spw"], f["hand"]): set(f["labels"]) for f in findings}
    assert "noisy_hand" in labels[("BB", 0, "R")] and "noisy_hand" not in labels[("BB", 0, "L")]
    assert "nominal" in labels[("BB", 1, "R")] and "nominal" in labels[("BB", 1, "L")]
    assert not labels[("AA", 0, "R")]
    checks = {("BB", 0, 0): {"slope": 0.0, "corr": 0.0, "fast_slope": 0.01, "fast_corr": 0.0, "fast_rms": 0.06}}
    plan = tsys.plan(syscal, findings, checks)
    assert plan[("BB", 0, 0)]["action"] == "substitute" and plan[("BB", 0, 0)]["source_hand"] == 1
    assert plan[("BB", 1, 0)]["window"] == 0.0             # nominal: nothing to smooth
    assert plan[("AA", 0, 0)]["combine_fields"]            # a faint source does not raise the Tsys


def test_tsys_rules_opacity_and_spurious_subband():
    real = {"slope": 2.1, "corr": 0.98}
    assert any(r["label"] == "tsys_undercorrects" for r in knowledge.label(real, "tsys"))
    spurious = {"labels": ["high_subband"], "tsys_real": False}
    assert [r["action"] for r in knowledge.label(spurious, "tsys")] == ["rescale_subband"]


def test_distribution_test_is_calibrated_on_noise_and_finds_injected_problems():
    cube = _cube(amp=0.0, sigma=0.05, seed=5)              # pure noise: no outlier, chi2r ~ 1
    found = distributions.assess(cube)
    assert len(found["bins"]) == 0 and len(found["scans"]) == 0
    assert np.median([b["chi2r"] for b in found["baselines"]]) == pytest.approx(1.0, abs=0.15)
    cube = _cube(amp=1.0, sigma=0.05, seed=6)
    spike = np.flatnonzero((cube["scan"] == 2) & (cube["ant1"] == 0) & (cube["ant2"] == 1) & (cube["spw"] == 0))[4]
    cube["vec"][spike, 0] *= 0.3                            # a dropout
    jump = (cube["scan"] == 4) & (cube["ant1"] == 1) & (cube["ant2"] == 2) & (cube["spw"] == 1)
    cube["vec"][jump, 1] *= np.exp(1j * np.random.default_rng(7).uniform(-np.pi, np.pi, jump.sum()))
    found = distributions.assess(cube)
    assert any(b["scan"] == 2 and b["ant1"] == 0 and b["ant2"] == 1 and b["hand"] == 0 for b in found["bins"])
    assert any(s["scan"] == 4 and s["ant1"] == 1 and s["ant2"] == 2 and s["spw"] == 1 and s["hand"] == 1
               for s in found["scans"])
    commands = distributions.flag_commands(found, cube)
    assert any("scan='4'" in c and "A1&A2" in c and "correlation='LL'" in c for c in commands)


def test_forest_learns_threshold_rules_and_round_trips(tmp_path):
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 1, size=(300, 3))
    X[rng.random(X.shape) < 0.05] = np.nan                 # some missing features
    labels = ["noisy" if (x[0] > 0.6 and not x[1] > 0.5) else "ok" for x in X]
    forest = model.Forest.fit(X, labels, "tsys", n_trees=25)
    accuracy = np.mean([p == lab for p, lab in zip(forest.predict(X), labels)])
    assert accuracy > 0.9
    again = model.Forest.from_dict(forest.to_dict())
    assert again.predict(X[:20]) == forest.predict(X[:20])


def test_feature_rows_and_annotation():
    report = {"per_antenna": {"T6": {"fields": {"PC": {"n": 60, "scan_scatter": 0.01, "trend": -0.9,
                                                       "max_drop": -0.3, "max_rise": 0.9, "range": 1.2}},
                                     "trend_consistent_fields": 1}},
              "array_scan_scatter": {"PC": 0.01}, "tsys": {"findings": []}}
    rows = features.prediagnostic_rows("TEST", report)
    assert rows[0]["label"] == "antenna_sensitivity_trend"
    rows = features.annotate(rows, {"T6@PC": "antenna_sensitivity_drop"}, reviewer="test")
    assert rows[0]["label"] == "antenna_sensitivity_drop" and rows[0]["reviewed_by"] == "test"


def test_scalar_bandpass_levels_the_two_hands(tmp_path, monkeypatch):
    """With level_hands, RCP 10% below LCP on an antenna comes out level; the antenna's Stokes I level stays."""
    from vlbipy.backends.casa import CasaCalibrationOps

    class FakeTable:
        store: dict = {}

        def open(self, path, nomodify=True):
            return True

        def getcol(self, name):
            return FakeTable.store[name].copy()

        def putcol(self, name, value):
            FakeTable.store[name] = np.array(value)

        def close(self):
            pass

    rcp = np.array([0.45, 0.45, 0.45, 0.45])
    lcp = rcp / 0.9
    FakeTable.store.update(CPARAM=np.stack([rcp, lcp])[:, None, :].astype(complex), FLAG=np.zeros((2, 1, 4), bool),
                           ANTENNA1=np.zeros(4, int), SPECTRAL_WINDOW_ID=np.arange(4))
    ops = CasaCalibrationOps.__new__(CasaCalibrationOps)
    ops._backend = type("B", (), {"tools": type("T", (), {"table": staticmethod(FakeTable)})()})()
    monkeypatch.setattr(ops, "_table_antenna_names", lambda path: ["AA"])
    ops._normalise_per_antenna(tmp_path, level_hands=True)
    g = np.abs(FakeTable.store["CPARAM"][:, 0, 0])
    corrected = np.array([rcp[0], lcp[0]]) / g                 # CASA divides the data by the gains
    assert corrected[0] == pytest.approx(corrected[1])          # hands level
    assert np.sqrt(g[0] * g[1]) == pytest.approx(1.0)           # Stokes I level untouched


def test_review_patterns_and_shipped_review_file():
    """A review file's patterns (``tsys/UR:*:R``, ``prediagnostic/T6@*``) override the rule labels; the EB104G
    review that ships with vlbipy parses and names labels the feature sets know."""
    import json
    rows = [{"stage": "tsys", "unit": f"UR:{s}:{h}", "label": "ok"} for s in range(3) for h in "RL"]
    rows.append({"stage": "prediagnostic", "unit": "T6@J1031+7441", "label": "ok"})
    review = {"reviewer": "test", "labels": {"tsys/UR:*:R": "noisy_tsys_hand",
                                             "prediagnostic/T6@*": "antenna_sensitivity_trend"}}
    rows = features.apply_review(rows, review)
    assert [r["label"] for r in rows if r["unit"].endswith(":R")] == ["noisy_tsys_hand"] * 3
    assert all(r["label"] == "ok" for r in rows if r["unit"].endswith(":L"))
    assert rows[-1]["label"] == "antenna_sensitivity_trend" and rows[-1]["reviewed_by"] == "test"
    shipped = json.loads((model.DATA_DIR / "EB104G.review.json").read_text())
    assert shipped["experiment"] == "EB104G" and "tsys/SR:5:R" in shipped["labels"]


def test_shipped_model_loads_and_predicts():
    forests = model.load()
    if not forests:
        pytest.skip("no trained model in vlbipy/data/ai")
    rows = features.read_rows(model.DATA_DIR / "EB104G.features.jsonl")
    rows = model.predict_rows(rows, forests)
    tsys_rows = [r for r in rows if r["stage"] == "tsys" and "model_label" in r]
    assert tsys_rows and all(0.0 <= r["model_p"] <= 1.0 for r in tsys_rows)
    noisy = [r for r in tsys_rows if r["unit"] in ("SR:5:R", "SR:6:R", "SR:7:R")]
    assert all(r["model_label"] == "noisy_tsys_hand" for r in noisy)     # it learnt the review
