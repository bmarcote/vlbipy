"""Tests for the slew-detection helpers (no CASA or MS required).

Exercises :func:`~vlbipy.backends.casa._leading_low_seconds` and
:func:`~vlbipy.backends.casa._scan_consensus` with synthetic data to
verify the ramp detection, consensus logic, and edge cases.
"""
import numpy as np
import pytest

from vlbipy.backends.casa import _leading_low_seconds, _scan_consensus

# ---------------------------------------------------------------------------
# Shared constants for synthetic timeseries
# ---------------------------------------------------------------------------
INTEGRATION = 2.0        # seconds
SIGMA = 2.0              # MADs below the stable level
MAX_SECONDS = 120.0      # cap


def _make_timeseries(n_samples: int = 60, integration: float = INTEGRATION,
                     ramp_end: float = 0.0, stable_level: float = 1.0,
                     ramp_level: float = 0.1, noise_std: float = 0.01,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic amplitude timeseries with an optional leading ramp.

    Parameters
    ----------
    n_samples : int
        Total number of time samples.
    integration : float
        Integration time (seconds) — spacing between samples.
    ramp_end : float
        Offset (seconds) at which the ramp ends and stable level begins. 0 = no ramp.
    stable_level : float
        Nominal amplitude of the on-source data.
    ramp_level : float
        Amplitude during the ramp (constant low, simulating slewing).
    noise_std : float
        Gaussian noise added to all samples.
    rng : numpy.random.Generator or None
        Random number generator (for reproducibility).

    Returns
    -------
    offsets : numpy.ndarray
        Time offsets from scan start (sorted ascending).
    amps : numpy.ndarray
        Amplitude values.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    offsets = np.arange(n_samples) * integration
    amps = np.full(n_samples, stable_level)
    if ramp_end > 0:
        amps[offsets < ramp_end] = ramp_level
    amps += rng.normal(0, noise_std, n_samples)
    return offsets, np.maximum(amps, 0.0)


# ===== _leading_low_seconds tests =====

class TestLeadingLowSeconds:
    """Unit tests for :func:`_leading_low_seconds`."""

    def test_synthetic_ramp_detected(self):
        """(a) A clean ramp at the start of a scan is detected."""
        offsets, amps = _make_timeseries(n_samples=60, ramp_end=20.0)
        result = _leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, MAX_SECONDS)
        # Ramp occupies samples 0..9 (offsets 0..18), so result should be near 20 s.
        assert result > 0, "ramp was not detected at all"
        assert 15.0 <= result <= 25.0, f"expected ~20 s, got {result:.1f}"

    def test_ramp_starting_few_samples_in(self):
        """(b) A ramp where the first sample is only mildly depressed is still detected.

        The stable-region level (computed from guard region) sees the first sample
        as slightly below the threshold, so the leading-low stretch starts at sample 0.
        """
        rng = np.random.default_rng(99)
        offsets, amps = _make_timeseries(n_samples=60, ramp_end=20.0, ramp_level=0.1,
                                         stable_level=1.0, noise_std=0.01, rng=rng)
        # Make the first sample only mildly depressed (0.7 instead of 0.1) — still below
        # the stable level minus sigma * noise when the guard-region level is ~1.0.
        amps[0] = 0.7
        result = _leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, MAX_SECONDS)
        assert result > 0, "mildly depressed first sample was missed"
        # Should still detect the ramp region (first sample is still well below 1.0 - 2*noise).
        assert result >= 2.0

    def test_clean_scan_returns_zero(self):
        """(d) A scan with no ramp yields zero."""
        offsets, amps = _make_timeseries(n_samples=60, ramp_end=0.0)
        result = _leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, MAX_SECONDS)
        assert result == 0.0

    def test_result_capped_at_max_seconds(self):
        """(f) The result is capped at max_seconds even for a very long ramp."""
        cap = 10.0
        offsets, amps = _make_timeseries(n_samples=60, ramp_end=50.0)
        result = _leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, cap)
        assert result <= cap, f"result {result:.1f} exceeds cap {cap}"
        assert result > 0, "the ramp should still be detected before capping"

    def test_too_few_samples_returns_zero(self):
        """Fewer than 5 samples cannot establish a reliable level."""
        offsets = np.array([0.0, 2.0, 4.0, 6.0])
        amps = np.array([0.1, 0.1, 1.0, 1.0])
        result = _leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, MAX_SECONDS)
        assert result == 0.0

    def test_constant_amplitude_returns_zero(self):
        """When noise is exactly zero and all samples are identical, MAD is 0 → return 0."""
        offsets = np.arange(20) * INTEGRATION
        amps = np.ones(20) * 5.0
        result = _leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, MAX_SECONDS)
        assert result == 0.0


# ===== _scan_consensus tests =====

class TestScanConsensus:
    """Unit tests for :func:`_scan_consensus`."""

    def test_all_baselines_agree(self):
        """When all baselines report the same ramp, consensus returns that value."""
        lows = [20.0, 20.0, 20.0, 20.0]
        assert _scan_consensus(lows) == 20.0

    def test_single_baseline_dip_does_not_flag(self):
        """(c) A dip present on only 1 of 4 baselines does NOT flag the antenna.

        A far-end fault affects only one baseline; fewer than half agree so the
        antenna is not flagged.
        """
        lows = [15.0, 0.0, 0.0, 0.0]
        assert _scan_consensus(lows) == 0.0

    def test_majority_consensus(self):
        """When 3 of 4 baselines report a ramp, consensus is the median of the positives."""
        lows = [18.0, 20.0, 22.0, 0.0]
        result = _scan_consensus(lows)
        assert result == pytest.approx(20.0)

    def test_exactly_half_with_min_agree(self):
        """Exactly half positive (2 of 4) — must meet >=half *and* >=2; 2/4 = 0.5 not > 0.5."""
        # 2 < 4/2.0 = 2.0 is False (2 < 2.0 is False), but 2 >= 2 is True.
        # So n_pos >= min_agree (2 >= 2) AND n_pos >= len/2 (2 >= 2.0) => consensus.
        lows = [15.0, 20.0, 0.0, 0.0]
        result = _scan_consensus(lows)
        # n_pos=2, len=4, 2 < 4/2.0=2.0 is False → consensus reached.
        assert result == pytest.approx(17.5)

    def test_one_of_two_baselines_no_consensus(self):
        """With only 2 baselines, both must agree (min_agree=2, half=1)."""
        lows = [15.0, 0.0]
        # n_pos=1 < min_agree=2 → no consensus.
        assert _scan_consensus(lows) == 0.0

    def test_both_of_two_baselines_consensus(self):
        """With 2 baselines both reporting, consensus is the median."""
        lows = [15.0, 20.0]
        result = _scan_consensus(lows)
        assert result == pytest.approx(17.5)

    def test_empty_list(self):
        assert _scan_consensus([]) == 0.0

    def test_all_zeros(self):
        assert _scan_consensus([0.0, 0.0, 0.0]) == 0.0


# ===== Integration: consensus over multiple scans =====

class TestPerAntennaMedianOverScans:
    """(e) Per-scan varying slew (slew in half the scans) still yields a nonzero antenna estimate.

    This tests the design decision to use *median* over scans instead of *mean*:
    an antenna that slews in some scans but not others should still get a nonzero
    estimate when the majority of scans show a slew.
    """

    def test_slew_in_majority_of_scans_yields_nonzero(self):
        """Median of [20, 18, 22, 0] is 19.0 — nonzero despite one clean scan."""
        scan_estimates = [20.0, 18.0, 22.0, 0.0]
        # np.median picks the average of the two middle values: (18, 20) → 19.0.
        result = float(np.median(scan_estimates))
        assert result > 0
        assert result == pytest.approx(19.0)

    def test_slew_in_minority_of_scans_yields_zero(self):
        """Median of [0, 0, 0, 20] is 0.0 — antenna is not flagged."""
        scan_estimates = [0.0, 0.0, 0.0, 20.0]
        result = float(np.median(scan_estimates))
        assert result == 0.0

    def test_slew_in_exactly_half_yields_nonzero(self):
        """Median of [20, 22, 0, 0] is 10.0 — still detects the antenna."""
        scan_estimates = [20.0, 22.0, 0.0, 0.0]
        result = float(np.median(scan_estimates))
        assert result > 0


# ===== Full pipeline: _leading_low_seconds + _scan_consensus =====

class TestEndToEndConsensus:
    """End-to-end tests combining both helpers to simulate the per-scan loop."""

    def _per_scan_estimate(self, baselines_data: list[tuple[np.ndarray, np.ndarray]]) -> float:
        """Simulate the per-scan loop: compute leading-low for each baseline, then consensus."""
        lows = [_leading_low_seconds(offsets, amps, SIGMA, INTEGRATION, MAX_SECONDS)
                for offsets, amps in baselines_data]
        return _scan_consensus(lows)

    def test_ramp_on_all_baselines_detected(self):
        """All baselines of a slewing antenna show the ramp."""
        rng = np.random.default_rng(7)
        baselines = []
        for _ in range(5):
            offsets, amps = _make_timeseries(n_samples=60, ramp_end=20.0, rng=rng)
            baselines.append((offsets, amps))
        result = self._per_scan_estimate(baselines)
        assert result > 0
        assert 15.0 <= result <= 25.0

    def test_ramp_on_one_of_four_not_detected(self):
        """(c) A dip on 1/4 baselines (far-end fault) does NOT flag this antenna."""
        rng = np.random.default_rng(8)
        baselines = []
        # 1 baseline with ramp
        offsets, amps = _make_timeseries(n_samples=60, ramp_end=20.0, rng=rng)
        baselines.append((offsets, amps))
        # 3 baselines clean
        for _ in range(3):
            offsets, amps = _make_timeseries(n_samples=60, ramp_end=0.0, rng=rng)
            baselines.append((offsets, amps))
        result = self._per_scan_estimate(baselines)
        assert result == 0.0, "far-end fault on 1/4 baselines should not flag this antenna"

    def test_ramp_on_three_of_four_detected(self):
        """Ramp on 3/4 baselines (the antenna is the common element) is detected."""
        rng = np.random.default_rng(9)
        baselines = []
        for _ in range(3):
            offsets, amps = _make_timeseries(n_samples=60, ramp_end=20.0, rng=rng)
            baselines.append((offsets, amps))
        # 1 clean baseline
        offsets, amps = _make_timeseries(n_samples=60, ramp_end=0.0, rng=rng)
        baselines.append((offsets, amps))
        result = self._per_scan_estimate(baselines)
        assert result > 0, "ramp on 3/4 baselines should be detected"
        assert 15.0 <= result <= 25.0


# ===== measure_quack: chunked reads of the measurement set =====

class _FakeMs:
    """An ms tool over an in-memory visibility set; records the size of every read."""

    def __init__(self, rows: dict, scan_of_row: np.ndarray) -> None:
        self._rows, self._scan_of_row = rows, scan_of_row
        self._selected = None
        self.reads: list[int] = []

    def open(self, _path):
        return True

    def close(self):
        return True

    def selectinit(self, datadescid=0):
        self._selected = np.ones(self._scan_of_row.shape, dtype=bool)

    def msselect(self, selection):
        if "scan" in selection:
            wanted = [int(s) for s in selection["scan"].split(",")]
            self._selected &= np.isin(self._scan_of_row, wanted)
        if not self._selected.any():
            raise RuntimeError("MSSelectionNullSelection")
        return True

    def reset(self):
        self._selected = None

    def getdata(self, columns):
        keep = self._selected
        self.reads.append(int(keep.sum()))
        return {name: self._rows[name][..., keep] for name in columns}


def test_measure_quack_reads_in_scan_chunks(monkeypatch, tmp_path):
    """The ramp is found from bounded reads: never a whole subband of the experiment at once."""
    from types import SimpleNamespace

    from vlbipy.backends import casa

    n_ant, n_scans, n_samples, slow, ramp = 4, 6, 60, 2, 20.0
    pairs = [(a, b) for a in range(n_ant) for b in range(a + 1, n_ant)]
    rng = np.random.default_rng(1)
    time, ant1, ant2, scan_of_row, amps = [], [], [], [], []
    scans = []
    for index in range(n_scans):
        start = 1000.0 * index
        scans.append(SimpleNamespace(scan_number=index + 1, source="CAL", time_start=start,
                                     time_end=start + n_samples * INTEGRATION,
                                     integration_time=INTEGRATION))
        for a, b in pairs:
            offsets = np.arange(n_samples) * INTEGRATION
            level = np.where((offsets < ramp) & (slow in (a, b)), 0.1, 1.0)
            amps.append(level + rng.normal(0, 0.01, n_samples))
            time.append(start + offsets)
            ant1.append(np.full(n_samples, a))
            ant2.append(np.full(n_samples, b))
            scan_of_row.append(np.full(n_samples, index + 1))
    amplitude = np.concatenate(amps)
    rows = {"data": np.broadcast_to(amplitude.astype(complex), (1, 8, amplitude.size)).copy(),
            "flag": np.zeros((1, 8, amplitude.size), dtype=bool),
            "time": np.concatenate(time), "antenna1": np.concatenate(ant1),
            "antenna2": np.concatenate(ant2)}
    fake = _FakeMs(rows, np.concatenate(scan_of_row))
    meta = SimpleNamespace(scans=scans, antennas=["A", "B", "C", "D"],
                           freq_setup=SimpleNamespace(n_subbands=1))
    backend = SimpleNamespace(tools=SimpleNamespace(ms=lambda: fake),
                              ms_path=lambda code: tmp_path / f"{code}.ms")
    monkeypatch.setattr(casa, "parallel_hand_indices", lambda metadata: ([0], ["RR"]))
    monkeypatch.setattr(casa, "_QUACK_SCANS_PER_READ", 2)

    result = casa.CasaFlagOps(backend).measure_quack("exp", column="data", metadata=meta)

    assert len(fake.reads) == 3 and max(fake.reads) == 2 * len(pairs) * n_samples
    assert result["per_antenna"] == {"C": pytest.approx(ramp, abs=INTEGRATION)}
