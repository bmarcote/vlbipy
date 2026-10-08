"""Tests for the calibrated-data off-source detection (vlbipy.statistics.detect_off_source)."""
from __future__ import annotations

import numpy as np

from vlbipy.statistics import antenna_on_source_fraction, detect_off_source


def make_track(n_antenna: int = 8, n_scans: int = 12, per_scan: int = 20, noise: float = 0.03, seed: int = 0):
    """Calibrated amplitudes of a point source: every baseline at its own level, plus noise."""
    rng = np.random.default_rng(seed)
    n_time = n_scans * per_scan
    level = rng.uniform(0.1, 0.4, (n_antenna, n_antenna))
    level = (level + level.T) / 2
    amplitude = level[:, :, None] * (1.0 + noise * rng.standard_normal((n_antenna, n_antenna, n_time)))
    amplitude = (amplitude + amplitude.transpose(1, 0, 2)) / 2
    amplitude[np.arange(n_antenna), np.arange(n_antenna)] = np.nan
    return amplitude, np.repeat(np.arange(1, n_scans + 1), per_scan)


def put_off_source(amplitude: np.ndarray, antenna: int, samples, factor: float = 0.2) -> None:
    amplitude[antenna, :, samples] *= factor
    amplitude[:, antenna, samples] *= factor


def test_a_late_antenna_is_found_and_its_partners_are_not():
    amplitude, scans = make_track()
    late = [s * 20 + k for s in range(12) for k in range(3)]        # first three samples of every scan
    put_off_source(amplitude, 2, late)
    result = detect_off_source(amplitude, scans)
    assert result["off"][2, late].all()
    assert result["off"][2].sum() == len(late)
    assert not np.delete(result["off"], 2, axis=0).any()            # nobody else is blamed for its baselines


def test_several_antennas_late_together_are_all_found():
    amplitude, scans = make_track()
    for antenna in (1, 4, 6):
        put_off_source(amplitude, antenna, [40, 41])
    result = detect_off_source(amplitude, scans)
    assert result["off"][[1, 4, 6]][:, [40, 41]].all()
    assert not result["off"][[0, 2, 3, 5, 7]].any()


def test_clean_data_is_left_alone():
    amplitude, scans = make_track(noise=0.08)
    assert not detect_off_source(amplitude, scans)["off"].any()


def test_a_scan_that_is_bad_throughout_is_still_found():
    amplitude, scans = make_track()
    whole = np.flatnonzero(scans == 5)
    put_off_source(amplitude, 3, whole, factor=0.3)
    result = detect_off_source(amplitude, scans)
    assert result["off"][3, whole].all() and result["off"][3].sum() == whole.size


def test_noisy_baselines_do_not_vote_and_flagged_samples_are_undecided():
    amplitude, scans = make_track()
    rng = np.random.default_rng(3)
    amplitude[0, 1] = amplitude[1, 0] = 0.2 * np.abs(1 + rng.standard_normal(amplitude.shape[2]))   # pure noise
    amplitude[5, :, 100:104] = np.nan
    amplitude[:, 5, 100:104] = np.nan                                 # antenna 5 flagged for four samples
    result = detect_off_source(amplitude, scans)
    assert not result["usable"][0, 1]
    assert not result["judged"][5, 100:104].any() and not result["off"].any()


def test_on_source_fraction_recovers_the_antenna_factors():
    amplitude, scans = make_track(noise=0.0)
    ratio = amplitude / np.nanmedian(amplitude, axis=2, keepdims=True)
    ratio[3, :, 7] *= 0.5
    ratio[:, 3, 7] *= 0.5
    gains, n_baselines = antenna_on_source_fraction(ratio, np.isfinite(ratio[:, :, 0]))
    assert abs(gains[3, 7] - 0.5) < 0.03 and np.allclose(np.delete(gains[:, 7], 3), 1.0, atol=0.03)
    assert (n_baselines == 7).all()


# -- the backend step: flag commands on the calibrator and the arrival time handed to the faint field --

def test_backend_flags_the_late_antenna_and_transfers_its_arrival_time():
    import re
    from types import SimpleNamespace
    from vlbipy.backends.casa import CasaFlagOps
    from vlbipy.models import Antenna, ObsMetadata, Scan

    names = [f"A{k}" for k in range(8)]
    step, per_scan, t0 = 2.0, 30, 5.0e9
    scans, cursor = [], t0
    for number in range(1, 25):                       # calibrator, target, calibrator, ... back to back
        source = "CAL" if number % 2 else "TGT"
        gap = 0.0 if (number % 4 == 1 or source == "TGT") else 30.0     # every other calibrator scan follows a gap
        start = cursor + gap + step
        scans.append(Scan(scan_number=number, source=source, time_start=start, time_end=start + (per_scan - 1) * step,
                          antennas=list(names), integration_time=step))
        cursor = scans[-1].time_end
    meta = ObsMetadata(antennas={n: Antenna(name=n) for n in names}, scans=scans, source_names=["CAL", "TGT"])
    cal = [s for s in scans if s.source == "CAL"]
    amplitude, scan_index = make_track(n_antenna=8, n_scans=len(cal), per_scan=per_scan)
    times = np.concatenate([s.time_start + step * np.arange(per_scan) for s in cal])
    scan_index = np.repeat([s.scan_number for s in cal], per_scan)
    previous_end = {s.scan_number: (scans[k - 1].time_end if k else -np.inf) for k, s in enumerate(scans)}
    slewing = [s for s in cal if s.time_start - previous_end[s.scan_number] <= 10.0]
    for s in slewing:                                  # antenna 3 arrives 6 s late whenever the scan contains the slew
        put_off_source(amplitude, 3, np.flatnonzero(scan_index == s.scan_number)[:3])
    timeline = {"antennas": names, "times": times, "scans": scan_index, "amplitude": amplitude, "integration": step}
    ops = CasaFlagOps.__new__(CasaFlagOps)
    ops._backend = SimpleNamespace(data=SimpleNamespace(read_baseline_timeline=lambda *a, **k: timeline))
    report = ops.off_source("X", fields=["CAL"], transfer_fields=["TGT"], dry_run=True, metadata=meta)

    assert report["n_direct"] == len(slewing) and set(report["per_antenna"]) == {"A3"}
    assert report["typical"]["gapless"] == {"A3": 6.0} and report["typical"]["gap"] == {}
    targets = [s for s in scans if s.source == "TGT"]
    assert report["n_transfer"] == len(targets)
    assert all("antenna='A3'" in c for c in report["commands"])
    # on the target: from the scan start for the 6 s of the slew plus one integration to settle
    first, last = re.search(r"timerange='([^~]+)~([^']+)'", report["commands"][report["n_direct"]]).groups()
    seconds = lambda text: sum(float(x) * m for x, m in zip(text.split("/")[-1].split(":"), (3600, 60, 1)))   # noqa: E731
    assert abs((seconds(last) - seconds(first)) % 86400 - 8.0) < 0.01
