# Fringe fitting and phase transfer (S9, S10, S13)

## Instrumental delay ("manual phase-cal", S9)

- **Physics:** each subband passes through different electronics (BBCs, filters, cables): a constant phase offset and
  delay per subband and hand, stable over the experiment. One short, bright solution is enough.
- **Scan choice:** one fringe-finder scan where every working antenna has strong, stable fringes (1-2 min of it is
  enough). Rank candidates by **source strength first** (e.g. the median over antennas of each antenna's best fringe
  SNR on the source, an undetected antenna counting as zero) and antenna coverage second.
- **No single scan with everyone:** solve on the scan with most antennas and the best refant; then a second scan,
  same refant if possible or an already-corrected antenna as reference, solving only the missing antennas; iterate.
  An antenna with one dead polarisation, or present only in another scan, keeps what it has.
- **Solve:** per subband and hand, delay + phase, no rate, central ~70-80% of the channels, S/N >~ 10 (a strong
  fringe finder gives S/N >> 100, so failures mean a real problem). With pulse-cals (VLBA), use them and verify with a
  manual phase-cal.
- **Good:** phases on the solved scan flat and continuous across subbands (a common slope is a residual multi-band
  delay); another scan equally aligned apart from its rate. **Bad:** missing solutions (the antenna is lost), jumps on
  another scan (instrumental phases changed - split the solution in time), very different R and L delays.

## Global fringe fit (S10)

- **Physics:** time-variable residual delays and rates (clocks, troposphere, ionosphere, geometry). Schwab & Cotton
  (1983): FFT search per baseline, then antenna-based least squares against a reference antenna.
- **Sources:** fringe finders + phase calibrators (+ anything you will self-calibrate independently). Never the
  phase-referenced target when astrometry matters.
- **Combine subbands** (after S9) for one multi-band delay and rate per antenna and hand; central channels only.
- **Solution interval:** <= scan length, <= coherence time (phase-reference calibrator scans are ~1-3 min; 1-2 min at
  K band).
- **S/N threshold:** ~5-7 for faint calibrators, >=10-30 for bright ones; below ~5 false detections dominate. Lower it
  (towards 3-5) only when a faint calibrator otherwise loses a large fraction of the antennas, and check the extra
  solutions. Solutions at the edge of the search window are suspect.
- **Search windows:** restrict delay and rate (e.g. +-100-200 ns, +-25-50 mHz at L band), checking real solutions are
  not clipped.
- **Dispersive delay:** fit it below ~8 GHz (and/or apply TEC maps first), especially with wide fractional bandwidth.
- **Rates on a distant/weak calibrator:** zero them if noisy.
- **An antenna undetected on the phase calibrator** (its delays look like noise there) may still be detected at high
  S/N (>~20) on the fringe finders: keep those solutions; dropping them can empty the bandpass and later solves.
- **Never hand on a table whose solutions are all flagged**: keep the previous pass's table and report.
- **Smoothing:** clip outliers and smooth **delays and rates** over a few intervals (up to ~1 h at cm wavelengths),
  per antenna, subband, hand **and field** (a fringe finder's direction must not leak into the phase calibrator's).
  Never smooth phases (they wrap); careful at >=22 GHz where the troposphere changes fast. Smoothing that is done right
  does not decorrelate the calibrators: if coherence drops, look elsewhere.
- **Count expected solutions** (antennas x hands x subband groups x intervals) and compare with good/failed counts:
  it separates "no data" from real failures.
- **Good:** smooth delays and rates per antenna, similar trends in both hands, few and understood failures,
  calibrated calibrator phases near 0 (structure can move the long baselines). **Bad:** noise-like solutions (S/N too
  low or wrong interval), jumps (clock breaks: split), a whole station failing (check the refant baselines, the
  subband coverage, the search window).

## Resolved phase calibrators

**Symptom:** the signal falls systematically with baseline length; the long baselines find no fringes while the short
ones detect the calibrator well. The baseline length where it disappears varies widely from case to case (a few
hundred km is one example, it depends on frequency, scattering and the source).

**Two causes:**
- **Scatter broadening** (near the Galactic plane, worse at low frequency, ~nu^-2): calibrators close on the sky
  behave alike.
- **Intrinsic structure:** each source behaves differently; another calibrator may be fine.

**What to do:** re-solve the nearby undetected antennas as a **sub-array** among themselves (referenced to one of
them) instead of losing them; flag the baselines between sub-arrays in the per-source data (they have no phase
connection). A uv taper cannot help: the long baselines have no phase solutions at all. A source that is resolved for
the long baselines limits what the target image can achieve on them.

## Transfer to the targets (S13)

- Interpolate each calibrator's delay, rate and phase in time, rate-aware ("ambiguity-aware": the rate connects the
  phases across the gap). Each target takes **its own** phase calibrator only.
- **Several calibrators per target:**
  - one much brighter (but farther): solve on it, apply to the secondary calibrator and the target, then refine with a
    fringe fit on the secondary and transfer that to the target;
  - two comparable calibrators bracketing the target in every cycle (1D MultiView): solve both and interpolate along
    the line joining them (projected position of the target); if each cycle has only one of them, check the target
    position from each subset;
  - several calibrators per cycle (2D MultiView): fit a plane in the sky per time.
- When a phase-calibrator scan is flagged for an antenna, flag that antenna on the adjacent target scans that would
  interpolate from it, rather than extrapolating across the gap.

## Inverse phase referencing

Fringe-fit on the target and transfer to the calibrator/check source (no phase-referencing cycle, or a target brighter
and more compact than the calibrator). Decide on the **target's fringe-fit signal-to-noise**, not its flux density: S/N
> 7 is recommended; it can go down to > 3 if that recovers clearly more baselines (check those solutions for false
detections). A target the fringe fit does not detect makes every solution fail. The target position then comes from
the calibrator's apparent offset, sign reversed, and target self-calibration is part of the main calibration.
