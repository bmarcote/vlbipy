# Special observing modes

## Phase referencing and astrometry

- Errors scale with target-calibrator separation, cycle time, elevation and atmospheric conditions: troposphere
  dominates at high frequency, ionosphere at low frequency.
- Improvements: TEC-map corrections, geodetic blocks / zenith-delay fitting, closer (in-beam) calibrators, MultiView
  (spatial interpolation from several calibrators), check sources to quantify the residual.
- The absolute target position is tied to the **phase-calibrator catalogue position**: report the catalogue and epoch.
- Inverse phase referencing: see `fringe-fitting.md` (decide on the target's fringe-fit S/N: > 7, down to > 3).

## Correlator passes

Multiple SFXC passes are `exp_N`:
- **continuum + spectral line:** same sources, many more channels in the line pass (often fewer subbands);
- **multiple phase centres:** different targets per pass (headers list all sources), calibrators often only in pass 1;
- **pulsar gating/binning:** identical setup and target, different pulse phases.

## Multi-band data

One dataset may hold two bands (S/X). Calibrate each band separately (split early), unless a combined dispersive
delay fit is wanted. Simultaneous triple-band data (e.g. 20/40/80 GHz) are calibrated at the lowest band first and
extrapolated upwards step by step.

## Spectral line (masers, HI absorption)

- Delays, bandpass and fringe fit on **continuum calibrators only**; the line source's phase changes quickly across
  the line (this also limits inverse phase referencing on it).
- Apply calibration from the continuum pass to the line pass (spwmap if the layout differs).
- Doppler corrections and regridding to a fixed frame (LSRK Galactic, barycentric extragalactic) with the right rest
  frequency and velocity convention.
- Strong lines: template-spectrum amplitude calibration (autocorrelations scaled to a reference station) beats Tsys.
- Residual rates on a strong line channel, self-cal on that channel, transfer to all channels (positions then relative
  to that feature unless phase-referenced).
- Continuum subtraction in the uv plane from line-free channels; cubes, moments, spectra.

## Polarimetry

1. Parallactic-angle correction (S5) - mandatory.
2. R-L delay and phase at the reference antenna from a polarised calibrator; keep hands separate in the fringe fit and
   do not re-reference solutions.
3. D-terms from an unpolarised compact calibrator or a polarised one with wide parallactic-angle coverage (>~3 h);
   several calibrators jointly are more robust. EVN leakages can be several per cent (Westerbork up to tens of per cent).
4. EVPA from a source with a known angle (connected-interferometer or single-dish monitoring near the epoch and
   frequency). An antenna recorded in linear polarisation and converted with PolConvert provides it directly when used
   as reference (stated in the PI letter for EVN).
5. Stokes I, Q, U (V) images; check residuals, closure traces and D-term consistency.

## Wide field / multi-phase-centre

Calibrate once on the calibrator (identical in all passes), apply to every phase centre, apply **primary-beam
corrections** per antenna for sources away from the pointing centre (large dishes like EF have narrow beams), and use
in-beam bright sources as secondary calibrators (model with all sources in all centres).

## Multi-epoch (S21)

- Calibrate each epoch independently; compare calibrator fluxes between epochs.
- Joint phase-calibrator model from all epochs together, **scaled per epoch by epoch flux / fitted joint-model flux**
  (not by the normalisation reference: extra structure changes the fitted total).
- Refine each epoch's gains against the fixed joint model; leave joint amplitude solving off so each epoch keeps its
  own flux scale; re-apply to the targets.
- Track the check-source position per epoch (systematic drift); combine target epochs only when flux, size and
  position are consistent.

## Frequency notes

- **<=2 GHz:** RFI, ionosphere (TEC maps, dispersive delay), wide fractional bandwidth (MT-MFS), large fields (smearing,
  primary beam), scatter broadening near the Galactic plane.
- **5-8 GHz:** usually the most benign.
- **>=22 GHz:** opacity, short coherence times, pointing losses on big dishes, short solution intervals, fast cycles.
