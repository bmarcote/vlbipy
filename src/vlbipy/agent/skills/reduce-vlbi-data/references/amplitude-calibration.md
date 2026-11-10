# Amplitude calibration (S6, S12, S17)

## A-priori: Tsys x gain curve

- `SEFD = Tsys / (DPFU * g(elevation))`; the visibility in Jy is `rho * sqrt(SEFD_i SEFD_j)` (times the digitisation
  factor). Tsys tracks atmosphere, spillover and receiver; the gain curve tracks the dish efficiency with elevation.
- **Opacity** matters at >~22 GHz when Tsys is not measured against a load above the atmosphere (Tsys*).
- **Sampler statistics:** 2-bit quantisation bias gives 5-20% RR/LL offsets when the correlator does not correct it
  (DiFX: VLBA, LBA). Correct from the autocorrelations before discarding them.
- **EVN:** attach the archive `.antab` even when the FITS-IDI already carry Tsys/GC tables (embedded since January
  2023): the embedded tables had appending issues; until further notice download the `.antab` (it is in the archive's
  pipeline area, not beside the FITS-IDI files) and use it.

### Judge the Tsys against the raw data

Read the raw data with the **a-priori flags only** (keep that flag version; never use flags made later, which already
depend on the calibration) and compare, per antenna, subband and hand:

- **A raw amplitude drop the Tsys follows** is real and will be calibrated away: nothing to do.
- **Tsys variation absent from the raw amplitudes** is measurement noise: smooth the Tsys over a long window instead
  of imprinting it on the data.
- A useful number: the slope of raw log-amplitude against `-0.5 log Tsys`, on slow trends and scan to scan. ~1 = the
  Tsys variation is real; ~0 = Tsys measurement noise; >1.2 with a high correlation = absorption the ground Tsys
  misses (consider an opacity correction or time-dependent amplitude self-cal).
- **One hand much noisier** than the other: substitute the good hand's Tsys (scaled) rather than flag.
- **One subband with a spuriously high Tsys** (the raw amplitudes are not lower there): rescale that subband. A
  subband whose raw amplitudes are lower too is real: keep it.
- **Unphysical values** (zero, negative, constant placeholders, spikes): flag those samples or interpolate.
- **Nominal Tsys** (no measurement): the a-priori scale is uncertain; rely on amplitude self-calibration.
- **Constant Tsys with DPFU = 1** (some stations, e.g. Jb, Tianma, Urumqi and the e-MERLIN outstations in EVN runs):
  that is an SEFD, so the a-priori calibration carries **no time dependence**; a time-dependent amplitude self-cal on
  the calibrators must supply it.

### Gain curves

- The curve must stay **positive and smooth over the elevations each antenna actually observed**; a curve that dips to
  zero or below there is wrong (bad polynomial, wrong variable) and must be replaced or flagged. Outside the observed
  range it does not matter, but polynomials diverge there, so plot only the observed range.
- CASA stores the gain curve (`gencal caltype='gc'`) as the **power** gain `DPFU x POLY` in K/Jy, a polynomial in
  **zenith angle** (degrees), re-expanded from the ANTAB elevation polynomial, and applies its square root. Never
  evaluate those coefficients against elevation.
- CASA keeps 8 coefficients: longer ANTAB polynomials must be refitted over the observed elevation range (check the
  refit stays positive and within ~1% there).
- Stations with `DPFU = 1, POLY = 1` have no real gain curve: their Tsys column carries the whole scale.

Expected accuracy of the a-priori calibration: ~5-10% for well-behaved EVN/VLBA stations, worse with nominal values.

## Scalar bandpass (S12)

After the complex bandpass (normalised per subband) the subband levels of an antenna can still differ (per-subband
gains not captured by the Tsys, a weak phase calibrator that could not solve them). Solve one amplitude per antenna
and subband (zero phase) from the calibrated calibrators, and **level both hands together**: nothing downstream that
solves one gain per station (difmap) can correct an RCP/LCP offset.

## e-MERLIN outstations in EVN runs (flux scale on 3C286)

The outstations (Cm, Da, De, Kn, Pi) lack reliable a-priori Tsys and their subbands often differ by large factors.

- **When:** only if e-MERLIN outstations are in the array and 3C286 was observed. Jb alone does **not** trigger it
  (Jb observes in plain EVN runs); Jb is excluded only from that presence check.
- **Model:** 3C286 is slightly resolved on e-MERLIN baselines: use the e-MERLIN model image of the band (CASA
  `*.model.tt0` or AIPS `.ICL001` images; shipped with the eMCP pipeline and with vlbipy), scaled channel by channel to
  the Perley & Butler (2017) spectrum.
  Check the model orientation when converting between formats: a reader that takes |CDELT| mirrors the image in RA.
- **Solve:** phases first (short interval, so the amplitudes average coherently), then **one amplitude gain per
  antenna, subband and hand over all 3C286 scans**, on the baselines among the e-MERLIN antennas **including Jb**
  (Jb is solved with the outstations, not held at 1). Subbands where e-MERLIN did not observe take the nearest
  observed subband's gains (spwmap).
- **Check:** after applying, the calibrated 3C286 amplitudes on those baselines should match the model in every
  subband and hand within a few per cent. Transfer to the other calibrators and compare with their known fluxes.
- **Resolved-out flux of 3C286:** the model is scaled to the full Perley-Butler flux, but e-MERLIN resolves out part
  of it even on its shortest baseline. The e-MERLIN CASA pipeline (eMCP, current `casa6` branch, `calc_eMfactor`)
  multiplies the fluxes bootstrapped to the other calibrators by `eMfactor = 1 / (1 + rho)`, with
  `rho = 0.04 x (f / 5 GHz x L / 11.237 km)^2`, f the central frequency and L the shortest projected baseline to 3C286
  (Lovell excluded); rho = 0.04 for Mk-Ta at 5 GHz. Older tutorials used a fixed 0.938 (Fenech et al.). Whether it
  applies depends on the procedure: it corrects a bootstrap through `fluxscale`; a solve made directly against the
  scaled model carries the same bias in its gains. vlbipy does not apply it (yet).
- **Afterwards:** the e-MERLIN antennas **hold their amplitudes** during the full-array self-calibration (they carry
  each subband's flux scale; the rest of the array is scaled to them) while **their phases are still solved**. A
  self-cal "fix antenna" option that holds phases too (difmap's `selfant` fix) is not what you want.

## Amplitude corrections from self-calibration (S17)

See `imaging-selfcal.md`: per-station gscale, per-subband shrinkage, time-dependent curves and what transfers to
which field.
