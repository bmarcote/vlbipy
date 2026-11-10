# Flagging and noise (S4, S8, S12, S14, S18)

## Estimating the noise

- **From time differences, never channel differences.** Adjacent channels are correlated: SFXC's windowed FFT gives an
  adjacent-channel correlation rho1 ~ 0.5, so channel differences underestimate the variance several times (~3x) and
  the variance of an average over n channels is `sigma^2 (1 + 2 rho1) / n`, not `sigma^2 / n`. Measure rho1 per hand
  if you need channel averages; use the time-difference estimate for every SNR, z-score and weight.
- A phase-independent amplitude `sqrt(power - noise^2)` is only as good as that noise estimate: a wrong noise invents
  "decorrelation" on the noisiest antenna.
- Re-weighting tools that take the scatter across channels inherit the same bias in absolute terms; relative weights
  between baselines are less affected. Check weights against time and uv distance per baseline.

## Per-baseline statistics

VLBI sensitivities differ by orders of magnitude between baselines (a 100-m dish to a 25-m dish vs two small dishes):
every statistic is computed **per baseline** (and per subband and hand), never array-wide. Readers of raw data must
drop exact zeros as well as flagged samples: a subband an antenna never recorded is zero-filled rows, and it must not
count as "flagged" in the statistics either.

## Initial flagging (S8)

Only strong outliers before any solve: RFI spikes, zero amplitudes, dead subbands or hands, station-reported bad
scans. Time-frequency croppers that work per baseline (tfcrop, AOFlagger with a VLBI/e-MERLIN strategy) are fine.
Never deep-flag the target at this stage. Save a flag version first.

## Distribution flagging of calibrated data (S12, S18)

Calibrated data on a baseline should look like a constant (or slowly varying, for a resolved source) signal plus
thermal noise: Rice-distributed amplitudes and von Mises phases around the signal.

1. **Detect.** Compare each baseline with **its own** behaviour: a running complex median along time as the signal,
   the model noise (time-difference estimate) as the width, and a fitted fractional jitter for slow real variations.
   Bins and scans that depart from that distribution are candidates.
2. **Require absolute badness at high S/N.** At S/N ~ 100 a 2-degree atmospheric wobble is "significant" but harmless:
   also require large departures in absolute terms (e.g. scan jitter > 20%, bins departing > 10%), or a third of the
   best baselines get flagged.
3. **Grow the outlier into a region.** From each outlier, widen in **time** (neighbouring integrations and scans) and
   in **frequency** (neighbouring channels and subbands), both ways, until the data return to the noise distribution.
   The result is a time range and a channel/subband range.
4. **Attribute it to an antenna.** For the two antennas of the baseline, look at the same time and frequency range on
   each antenna's **other** baselines. The antenna whose other baselines are also bad is responsible: flag that antenna
   (all its baselines) over the region. If neither antenna shows it elsewhere, it is a baseline-based problem: flag
   only that baseline.
5. **Flag the whole region**, not just the outlying bins.
6. **Keep phase referencing consistent.** If a phase-calibrator scan is flagged for an antenna, also flag that antenna
   on the adjacent target (and check-source) scans that would be calibrated from it; otherwise the target is
   calibrated by extrapolating across the gap and phase referencing does not work there.
7. Record the flag commands; a re-run of the same flagging restores the previous flag version first rather than
   flagging on top of itself.

## Quack (S4)

The first seconds of a scan are low in amplitude with scattered phases while an antenna is still settling. Measure per
antenna: an antenna's slew is the leading stretch that is low on **all** its baselines (one noisy baseline must not
extend it). Flag the same on the target scans (symmetric slews). Typical 2-10 s; e-MERLIN and Westerbork often need it
beyond the a-priori flags.

## Flag statistics (S20)

Per antenna and subband, over the data that antenna could have observed: never count subbands an antenna did not
record (zero-filled) as flagged.
