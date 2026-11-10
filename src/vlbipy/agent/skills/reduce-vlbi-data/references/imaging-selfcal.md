# Imaging and self-calibration (S17-S19)

## Principle

Self-calibration treats the source as its own calibrator: assume a sky model, solve the antenna gains that best match
the data to it, re-image, repeat. It works because there are many more baselines than antennas (closure quantities
constrain the structure). A wrong model is imposed on the data, and through the transferred gains on the target too:
always compare model and data before solving.

## Imaging parameters

- **Beam:** theta ~ lambda / B_max (from the maximum uv distance in wavelengths).
- **Cell:** ~theta/10 for imaging. **Never coarser than ~theta/4.6 in difmap**: difmap grids only
  |u|, |v| < 1/(4 cell) x (1 - 8/npix), so a coarse cell (e.g. a quick search map at theta/3) silently drops every
  baseline beyond ~0.7 of the longest - and with them all the data of a target seen only on long baselines.
- **Image size:** large enough for the source and its brightest sidelobes (and for the search, S18); FFT-friendly
  (2^n or 2^n 3^m 5^k); 4096 is a good default for calibrators.
- **Weighting:** natural = sensitivity and extended emission; uniform = resolution; robust in between. Heterogeneous
  arrays (EF, e-MERLIN, big dishes) make natural weighting dominated by a few baselines. For calibrators start uniform
  and move towards natural.
- **Deconvolver:** Hogbom/Clark for compact emission and fractional bandwidth <~10-15%; multi-scale for extended
  emission; MT-MFS (nterms=2) when BW/nu >~15% or the spectral index matters (EVN L band ~16% is on the limit).
- **Windows:** tight and conservative; VLBI's sparse uv coverage makes sidelobes look like components.
- **difmap and flagged channels:** difmap flags a whole IF average when any selected channel in it is flagged. Load
  channel-averaged data (or select only unflagged channels).
- **Image S/N = peak / rms.** Quote that, not "dynamic range".

## Model fitting first

Fitting components directly to the visibilities is usually better than CLEAN for compact VLBI sources: it does not
depend on the weighting, does not inflate sizes, and allows sizes down to a fraction of the beam (S/N dependent).
- Start from **one circular Gaussian** (calibrators and targets).
- Add a component only when significant positive residuals remain **and** the fit improves (a model-comparison
  criterion), never just to lower the residuals.
- Keep calibrators parametric through phase self-cal, amplitude self-cal and joint multi-epoch models.
- Restore the same fitted model at each weighting with the untouched residuals added.
- A fit that fails or is unphysical is reported as such, not silently replaced by CLEAN.

## Self-calibration sequence

1. **Initial model:** a point source or a single Gaussian at the phase centre (calibrators), the peak of a dirty image,
   or a first fit. Flux from amplitude vs uv distance. Only self-calibrate once the model tracks the data (amplitude and
   phase vs uv distance, closure phases).
2. **Phase only** first (the atmosphere dominates). Long intervals at first (~30 min, or the whole run if the model is
   poor), shortening (15 -> 5 min -> scan length; shorter only at high frequency or with scans much longer than the
   coherence time). The phase should change by less than a few tens of degrees within an interval, and S/N per antenna
   per interval >~ 3-5.
3. **Amplitude + phase** only when the model amplitudes follow the data. First **one solution per antenna for the whole
   run** (gscale), then per subband, then time-dependent (~120 -> 60 -> 30 min) while solutions stay consistent.
   Normalise (no net flux change), keep amplitudes from wandering (uv-range limits, mean gain constrained, restart from
   the original data with the best model).
4. Re-image after each round. Rebuild the model from scratch after large corrections, or refit.

**Stop** when S/N plateaus, solutions get noisier than in the previous round, or amplitude corrections stay >20% from
unity (the model misses flux). Self-calibration is second-order: amplitudes should not grow significantly.

## Amplitude corrections: what to solve and where they go

- **Order:** the fringe finders first, each on data carrying the amplitude corrections of the earlier ones (a fringe
  finder that cannot be self-calibrated is skipped); then the phase calibrator (the target, with inverse phase
  referencing).
- **What transfers:**

| Correction | Applies to |
|---|---|
| a fringe finder's amplitude table (gscale per station and subband) | every field |
| a fringe finder's phase table and time-dependent amplitude curve | itself only |
| the phase calibrator's phase x amplitude table | itself and the fields it calibrates |

  The corrections multiply: the total per antenna and subband is scalar bandpass x every self-cal amplitude table
  applied to the target.
- **Per-station level with per-subband shrinkage:** solve one level per station and let each subband deviate around it
  with a prior (e.g. sigma ~ 0.1, a Bayesian gscale). Weak stations' per-subband noise collapses onto the station level,
  while real subband steps (5-15% left by a scalar bandpass a weak calibrator could not solve) are still corrected.
- **Time-dependent amplitude self-cal** (per scan, median-normalised) follows what the Tsys cannot (constant-Tsys/SEFD
  stations, absorption). Let the mean scale **float**: a solver that holds the mean gain of each IF at 1 lets one
  station needing a large correction drag all the others (chi-squared up by a large factor). Corrections more than ~x5
  from a station's median mean no signal: flag them, do not apply them.
- **Transferring the time-dependent curve to targets is opt-in, not default:** a per-scan solve against the
  calibrator's model carries the calibrator's structure; in one tested case transferring it made the check source worse
  (reduced chi-squared ~29 -> ~35, image S/N ~204 -> ~180). Verify on a check source before transferring.
- **e-MERLIN antennas** whose flux scale came from 3C286 hold their amplitudes (no correction) while their phases solve
  (`amplitude-calibration.md`).
- **Typical EVN factors 0.8-1.2.** Larger ones point to station problems (pointing, Tsys, polarisation swap). Record
  the factors per antenna and subband (JSON); flag cells >10% (and >30%) in reports. A gain record that stores 1.0 for
  both "no correction" and "no solution" must say which is which.
- **Compare before and after:** keep the binned amplitude vs uv distance of the data before the first self-cal and plot
  it under the final radplot; shifts forced by self-cal show up as offsets.

## Strategies

- **Outlier antennas:** exclude from the solve, calibrate the rest, re-include them.
- **Short/long baseline gap (EVN + e-MERLIN):** start with short-to-intermediate baselines, then add everything; the
  longest baselines may resolve the source or lack redundancy.
- **Heterogeneous weights:** if convergence is poor, use antenna weighting options or 1/sigma instead of 1/sigma^2.
- **At the end:** a calibrator model plus gains down to scan length; apply to the target (S18). Optionally repeat
  S9-S17 using the calibrator models.

## Target and check source (S18-S19)

- Search: large natural-weighted image around the expected position; S/N >~ 8 for an unknown position (~5 known).
  The search map's cell must respect the difmap limit above. Shift the phase centre to an off-centre detection.
- Phase-referencing quality: coherence losses (peak << integrated, fuzzy) mean imperfect phase transfer.
- Target self-cal only if phase-referenced S/N >~ 20; measure the astrometric position before it.
- Check sources: offset from the catalogue = systematic astrometric error; their residual phases after self-cal show
  the phase-referencing errors.
