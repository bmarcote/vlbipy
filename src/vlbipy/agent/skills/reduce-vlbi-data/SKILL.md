---
name: reduce-vlbi-data
description: Science and reasoning behind VLBI data reduction (centimetre-wavelength observations with the EVN, VLBA, LBA, e-MERLIN+EVN or global arrays) - what each calibration and imaging step corrects, why, how to judge the result and how to diagnose problems. Use for any VLBI calibration or imaging question; pair with /reduce-vlbi-vlbipy, /reduce-vlbi-casa or /reduce-vlbi-aips for the commands.
---

# Reduce VLBI data - the knowledge

This skill holds the **what and why** of VLBI data reduction: the physics each step corrects, the decisions to make,
the rules of thumb, and how to tell good results from bad. It is software-agnostic.

- **Commands:** `/reduce-vlbi-vlbipy` (vlbipy + difmapy), `/reduce-vlbi-casa` (CASA) and `/reduce-vlbi-aips`
  (AIPS / ParselTongue) follow the **same step numbers (S0-S21)** as this skill. Default order: vlbipy first, then
  CASA, then AIPS, unless the user says otherwise. For the science this skill wins over a command skill; for what
  a given vlbipy version does, its documentation and code win.
- **Golden rule:** *know your data.* VLBI calibration is sequential: verify each step before the next, because errors
  carry forward silently. Keep the raw data untouched, add corrections as new tables, and write down which table
  contains what.
- **Details live in `references/`** (read them when the step needs it):

| File | Content |
|---|---|
| `references/amplitude-calibration.md` | Tsys and gain curves and how to judge them against the raw data, scalar bandpass, e-MERLIN flux scale |
| `references/fringe-fitting.md` | instrumental delays, global fringe fit, resolved calibrators, phase transfer, MultiView, inverse phase referencing |
| `references/flagging-and-noise.md` | noise estimation, per-baseline statistics, distribution flagging and how to grow and attribute flags |
| `references/imaging-selfcal.md` | imaging parameters, model fitting, self-calibration strategy, amplitude corrections, imaging pitfalls |
| `references/special-modes.md` | phase referencing and astrometry, spectral line, polarimetry, wide field, multi-epoch, frequency notes |
| `references/arrays-and-stations.md` | SFXC vs DiFX, archives and file formats, station notes (dated field notes) |
| `references/diagnostics.md` | symptom -> likely cause -> action |
| `references/quick-numbers.md` | the numbers to carry around |

---

## Part A - Foundations

### A1. What the correlator gives you, and why VLBI needs extra steps

- The correlator delivers complex visibilities `V_ij` per baseline, time and channel, after removing the geometric
  delay model. Their amplitudes are only correct *relative to each other*: the true flux density needs each antenna's
  sensitivity, `S_ij ~ rho_ij * sqrt(SEFD_i * SEFD_j)`, with `SEFD = Tsys / (DPFU * g(elevation))` (and a
  digitisation factor).
- VLBI feeds are circular. Data are dual polarisation (RR, LL) or full polarisation (RR, LL, RL, LR); some antennas
  record a single hand. Cross-hands should be well below the parallel hands (>~3x, typically >~10x) unless leakage is
  large or the antenna observed linear polarisation.
- Not every antenna records the whole band (e-MERLIN outstations, a failed BBC): count coverage per antenna.
- Each station has its own clock, frequency standard and atmosphere. The residual errors are **antenna-based**:
  `dphi(t, nu) ~ phi0 + 2 pi tau (nu - nu0) + 2 pi r (t - t0) [+ dispersive term ~ 1/nu]`, with `tau` the **delay**
  (phase slope in frequency) and `r` the **rate** (phase slope in time). VLBI must fit these derivatives because the
  phases wind too fast to calibrate directly. The dispersive (ionospheric) delay matters below ~8 GHz.
- Phases may jump between subbands (different electronic paths). The correlator normalises each subband, so the
  subband amplitude levels are not physical and strong RFI changes them; atmospheric absorption lowers the amplitudes
  smoothly at low elevation.
- **Decorrelation:** averaging data that still carry a phase slope lowers the amplitude (360 deg across the band means
  `tau = 1/BW`, e.g. 2 ns across 512 MHz). **Correct the phases before averaging in time or frequency.**
- **Antenna-based vs baseline-based:** N unknowns against N(N-1)/2 measurements, so antenna errors can be solved.
  Closure phases (triangles) and closure amplitudes (quadrangles) are immune to them: use them to check data and model
  independently of the calibration. Non-closing errors come from an uncalibrated or averaged-over variable bandpass, or
  the correlator.
- **Coherence time** sets the longest solution interval: ~4-6 min at 1.4-8 GHz, ~1-2 min at 22 GHz, <~30 s at 43 GHz,
  shorter in bad weather and at low elevation.
- **Noise comes from time differences, never from channel differences.** Adjacent channels are correlated (SFXC: ~0.5
  from the windowed FFT), so channel differences underestimate the variance several times and a channel average has
  variance `sigma^2 (1 + 2 rho1) / n`. Every SNR, weight or outlier test must use a time-difference noise estimate
  (`references/flagging-and-noise.md`).

### A2. Source roles

| Role | Purpose | Requirements |
|---|---|---|
| **Fringe finder** | instrumental delays, bandpass, first check that every station has fringes | very bright (>~1 Jy correlated on long baselines), mostly compact, ideally observed with all stations up |
| **Bandpass calibrator** | per-channel response | high S/N **per channel**; usually the fringe finder, or the phase calibrator combining all its scans |
| **Phase calibrator** | time-variable delay, rate and phase transferred to the target | compact, >~50-100 mJy correlated (ideally a few hundred), close to the target (ideally <~2-3 deg), cycle shorter than the coherence time; there may be several |
| **Check source** | independent test of phase-referencing quality and astrometric error | compact, known VLBI position, similar separation and geometry |
| **Amplitude/flux calibrator** | absolute flux scale where Tsys is unreliable (e-MERLIN) | well-modelled source (3C286 with an e-MERLIN model image) |
| **Polarisation calibrators** | D-terms and EVPA | unpolarised compact source, or polarised with wide parallactic-angle coverage, plus a known-EVPA source |
| **Target** | science | - |

Roles come from the PI, the schedule (`.vex`/`.key`) or the observation summary, and can be **inferred from the
schedule structure**: fringe finders get a few scans spread across the run; the phase calibrator opens and closes each
cycle, is visited more often and with shorter scans than its target; check sources are bracketed by the calibrator but
visited less often. Do not decide roles by brightness alone: calibrators are normally the brighter source, but a target
brighter than its calibrator does occur, and that is the case for inverse phase referencing (C1), not a reason to swap
the roles. **Never fringe-fit on the phase-referenced target when astrometry matters.** Without phase calibrators,
calibrate on the target; without fringe finders, do everything on the phase calibrators. Separate blocks of
calibrator/target cycles are treated independently.

### A3. Book-keeping (any package)

1. The original visibilities are never modified until the final apply + split; every correction lives in a table.
2. Build calibration incrementally: a-priori amplitudes -> model corrections -> instrumental delay -> fringe fit ->
   bandpass -> amplitude refinement -> self-cal. Record what each table contains and what it was derived from. Loops
   over part of this path (with better calibrator models) are normal.
3. Save **flag versions** before every flagging step. Keep the a-priori-only flag state: diagnostics of the raw data
   are read with those flags, never with later ones.
4. Keep **one reference antenna** for the whole calibration where possible: sensitive, present throughout, stable in
   both hands, central (EF for EVN, a central VLBA station, ATCA/Parkes for LBA). Check that it really has data in
   every scan; a fall-back list is fine for fringe fitting.
5. Log every decision (bad stations, flagged subbands, gain factors) in a text/JSON record.

---

## Part B - The workflow (steps shared with the command skills)

Each step: **purpose -> decisions -> checks (good vs bad)**. The detail is in the reference named at the step.

### S0 - Resources
Raw VLBI data are 10-100+ GB; processing needs about 3x that. Average early only after the phases and bandpass are
calibrated and the field of view is checked (S16).

### S1 - Retrieve the files and the observing context
- **EVN:** FITS-IDI (`exp_N_1.IDI*`), `exp.antab` (Tsys + gain curves), `exp.uvflg` (a-priori flags), the pipeline
  outputs (TASAV tables, plots) and the station feedback; the schedule if possible. Read the pipeline report and the
  station feedback first: failed stations, missing subbands, polarisation problems.
- **Use the archive `.antab` for EVN data even when the FITS-IDI carry Tsys/GC tables** (embedded since January 2023):
  the embedded tables had appending issues, so until further notice download the `.antab` and attach it.
- **VLBA:** FITS-IDI (`.idifits`) from the NRAO archive, calibration attached (its FLAG table must be applied
  explicitly, see `references/arrays-and-stations.md`). **LBA:** FITS plus per-antenna Tsys/flag files.
- Note the setup (subbands x channels x polarisations), integration time, subband coverage per antenna and the roles.

### S2 - Prepare the a-priori information
Attach Tsys and gain curves when needed and convert the AIPS-format flag file for your package. FITS-IDI from
EVN/LBA carry no dish diameters or axis offsets; not needed for calibration (the correlator applied the geometry), only
for primary-beam work, which also needs the pointing centre from the `.vex`. Stations without Tsys get nominal values
and their amplitudes are less certain.

### S3 - Import and inspect
Confirm antennas, scans against the schedule, subbands, channels, polarisations, integration time, sources and roles.
Plot antenna positions and uv coverage, auto-correlations (all antennas and hands), cross-correlation spectra to the
reference antenna and amplitude/phase against time for the whole run. **Choose the reference antenna and the
fringe-finder scan(s) for the instrumental delay now:** one scan (a couple of minutes is enough) where every antenna
has strong, stable fringes; otherwise the minimum chain of scans, each sharing an already-solved antenna with the next.
Rank candidate scans by source strength first and antenna coverage second: a weak source that happens to have one
more antenna gives a delay solution the later fringe fit cannot recover from.

### S4 - A-priori flagging
- Apply the correlator/station flags; check they do not remove long stretches that show good fringes (a wrong end time
  can extend for hours).
- Run autocorrelation-based amplitude corrections first (sampler bias, ACCOR-type; ACSCL-type after the bandpass), then
  ignore the autocorrelations.
- **Edge channels:** ~5-10% per edge on the polyphase/FFT roll-off; measured best from the bandpass (S11), where they
  become outliers.
- **Quack:** antennas still slewing at scan start (low amplitude, scattered phase, then steady). Often missing for
  e-MERLIN and Westerbork. Identify per antenna (different slew speeds; a scan counts against an antenna only when it
  is low on all its baselines), flag a few seconds, and flag the same on the target scans.
- Low elevation (<~10 deg) if the data look bad there.

### S5 - Model corrections (before any data-based phase calibration)
- **EOPs:** DiFX (VLBA, LBA) correlates with predicted EOPs - apply final ones. SFXC uses final EOPs; only real-time
  e-EVN runs need a correction, and only when the EOPs used are recorded in the data (since January 2026).
- **Ionosphere (TEC maps):** apply below ~8 GHz (always below 3 GHz) for phase referencing; coarse, removes only part
  of the effect. CASA's implementation has bugs for VLBI; AIPS and vlbipy work.
- **Parallactic angle:** geometric, before any phase calibration; essential for polarimetry and phase referencing.
  Apply it once (the EVN pipeline tables already contain it; split data have it).
- **Troposphere (astrometry, >=22 GHz):** residual zenith delays from geodetic-style blocks.

### S6 - A-priori amplitude calibration -> `references/amplitude-calibration.md`
Tsys (atmosphere, spillover, receiver) times the gain curve (DPFU x g(elevation)) gives the SEFD per antenna, subband,
hand and time; opacity matters at >~22 GHz; DiFX data need the sampler correction (5-20% RR/LL offsets).
- **Judge the Tsys against the raw amplitudes**, read with the a-priori flags only: a raw amplitude drop the Tsys
  follows is calibrated away; Tsys variations absent from the raw data are measurement noise (smooth them); raw drops
  deeper than the Tsys predicts are absorption the ground Tsys misses.
- **Gain curves** must stay positive (and smooth) over the elevations each antenna observed; outside that range they
  do not matter. A station with DPFU = 1 and a constant Tsys is really an SEFD: no time dependence in its a-priori
  calibration, so amplitude self-calibration must supply it.
- Expected accuracy ~5-10% for good EVN/VLBA stations; refine with calibrator models (S17) and record the factors.

### S7 - Diagnostic plots on uncalibrated data

| Plot | What to look for |
|---|---|
| SNR per antenna per scan (fringe detections) | antennas without signal, dropouts, a missing polarisation |
| cross-power spectra to the refant on the fringe-finder scan | bandpass shape, dead subbands, delays and subband phase jumps, the scan(s) with all antennas |
| amplitude/phase vs time to the refant | dropouts, slewing at scan start, phase winding (rates); a second refant when the first has gaps |
| amplitude vs channel per subband | edge roll-off, RFI, dead channels |
| time x frequency per baseline | RFI patterns, bad periods |
| uv coverage per source | coverage, missing baselines, the e-MERLIN short-spacing gap |
| autocorrelation spectra | sampler problems |

Raw-data readers must drop exact zeros as well as flags: a subband an antenna never recorded is zero-filled rows.

### S8 - Initial flagging -> `references/flagging-and-noise.md`
Only obvious problems: strong RFI, zero amplitudes, dead subbands or hands, station-reported bad scans. **Statistics
per baseline**, never array-wide (sensitivities differ by orders of magnitude). Don't deep-flag the target yet. Save a
flag version.

### S9 - Instrumental delay ("manual phase-cal") -> `references/fringe-fitting.md`
Per subband and hand, delay + phase, **no rate**, on the chosen fringe-finder scan(s) over ~1-2 min, central ~70-80%
of the channels, S/N threshold >~10. Good: phases on that scan flat and continuous across subbands (an overall slope is
a residual multi-band delay); another scan equally aligned (with rates still present). Bad: missing solutions (that
antenna is lost), jumps on another scan (instrumental phases changed: split in time), very different R/L delays.

### S10 - Global fringe fit (multi-band delay, rate, phase) -> `references/fringe-fitting.md`
Schwab & Cotton: FFT search then least squares, antenna-based against a reference. Fringe finders + phase calibrators,
never the astrometric target. Combine subbands (needs S9); solution interval <= scan and <= coherence time; S/N
threshold ~5-7 faint, >=10 bright (below ~5 false detections dominate); restricted search windows; dispersive delay
below ~8 GHz. Smooth delays (not phases) per field over up to ~1 h at cm wavelengths. Never apply a table whose
solutions are all flagged. Count expected vs obtained solutions. Good: smooth delays and rates, similar in both hands,
calibrated calibrator phases near zero. Bad: noise-like solutions, jumps (clock breaks), a whole station failing.
**Resolved calibrators:** if the signal drops systematically with baseline length and the long baselines find no
fringes while the short ones do, the calibrator is resolved (scatter broadening near the Galactic plane at low
frequency, or intrinsic structure - then other sources behave differently); solve the short-baseline antennas as a
sub-array. Each target takes **its own** phase calibrator's solutions, interpolated in time.

### S11 - Bandpass
Antenna-based complex response per channel, stable in time: combine all bandpass-calibrator scans. Remove delays and
rates first (or solve on the instrumental-delay data). Normalise over the **full** usable band (an inner portion alone
biases amplitudes up to ~15%); interpolate over flagged channels; apply any smoothing used in the solve. A weak
bandpass calibrator adds noise to every source. Good: smooth, repeating shapes, flat phases. Bad: noisy per-channel
scatter, spikes (RFI), ripples in one hand only. Flag the edge channels the bandpass reveals and solve again.
Integrating over an uncalibrated variable bandpass is a main source of non-closing errors.

### S12 - Further flagging, re-calibration and scalar bandpass
With a first calibration applied, flag outliers **outside each baseline's own noise distribution** (grow each outlier
in time and frequency, attribute it to the right antenna, and flag the adjacent target scans when a phase-calibrator
scan goes; `references/flagging-and-noise.md`). Redo S9-S11 on the cleaner data. Then a **scalar bandpass**: one
amplitude per antenna and subband (zero phase) levelling the subbands, with **both hands levelled together** -
station-based gains downstream (difmap) cannot correct an RCP/LCP offset.

### S13 - Apply the calibration
Bandpass and instrumental delays to everyone; calibrators take their own phase solutions; targets and check sources
take their phase calibrator's solutions interpolated in time (rate-aware). Several calibrators around a target
(MultiView) need interpolation in position too.

### S14 - Re-weighting
Weights should reflect each baseline's real noise. Recompute them from the scatter of the calibrated data along time
(not channels, A1); look at weight vs time and uv distance per baseline - outliers reveal bad data. Flag, and redo
S9-S13 if the flagging was substantial.

### S15 - Verify the calibrated data
Redo the S7 plots on the calibrated data: calibrator phases ~0 and flat, amplitudes stable and consistent with a
simple source; amplitude vs uv distance per station (a station high or low = amplitude error, fixed in S17); closure
quantities consistent with the calibrator structure, especially on the phase calibrator.

### S16 - Split and average
Split per source (or calibrator-target pairs). Smearing limits: frequency averaging broadens radially by
~(dnu/nu) r, time averaging tangentially by ~omega_E dt r (omega_E = 7.3e-5 rad/s); keep both << beam over the field
you care about (5 mas beam at 1 arcsec: dnu/nu << 5e-3, dt << 70 s). A compact source at the phase centre can be
averaged to one channel per subband - only after delays and bandpass are calibrated.

### S17 - Image and self-calibrate the calibrators -> `references/imaging-selfcal.md`
Fringe finders first (each on data carrying the earlier ones' amplitude corrections), then the phase calibrator.
Start from a simple model (a circular Gaussian fitted to the visibilities is often best), compare model and data before
solving, phase-only first with long intervals shortening as the model improves, amplitude only when the model follows
the data, starting with one solution per antenna for the whole run (gscale) then per subband. Judge by image S/N =
peak/rms, residuals and quieter solutions. Typical EVN amplitude factors 0.8-1.2; record them (JSON). The calibrator's
amplitude table goes to every field; its phase and time-dependent amplitude curves stay with the source they were
solved on unless the transfer is shown to help. **e-MERLIN outstations in EVN runs:** set their flux scale on 3C286 with
the e-MERLIN model first (Jb solved with them), then hold their amplitudes while their phases self-calibrate
(`references/amplitude-calibration.md`).

### S18 - Image the target
1. Flag deeply: calibrated target data should look like noise plus signal per baseline.
2. Find the source in a large natural-weighted dirty image: S/N >~ 8 for an unknown position (~5 for a known one);
   shift the phase centre if off-centre; try other weightings and tapers before quoting an upper limit (e.g. 5 sigma).
3. Coherence losses (peak << integrated flux, fuzzy) = imperfect phase transfer.
4. Self-cal the target only if bright (phase-referenced S/N >~ 20, S/N >~ 3-5 per antenna per interval); **measure the
   astrometric position before** self-calibrating.
5. Several weightings; rms, peak, integrated flux, position, sizes (prefer uv-plane model fitting).

### S19 - Check sources
Image like the target; the offset from the catalogue position is the systematic astrometric error at that separation.
If bright, self-calibrate: its residual phases show the phase-referencing errors.

### S20 - Final products and error budget
Images per source and weighting (FITS), measurements with uncertainties: flux = statistical (+) calibration (~5-10%);
position = thermal (~beam / (2 S/N)) (+) check-source systematic (+) calibrator position; size = deconvolved Gaussians
with an S/N-dependent resolution limit (Marti-Vidal et al. 2012). Light curves from vector-averaged visibilities; flag
statistics per antenna (excluding subbands never observed); gain-factor JSON. Keep raw data, tables, final per-source
data and scripts/logs.

### S21 - Multi-epoch observations
Calibrate each epoch independently; check the calibrators did not vary; build a joint phase-calibrator model from all
epochs (scaled to each epoch's flux), refine each epoch against it, re-apply to targets, track check-source positions
per epoch, combine target epochs only when flux, size and position are consistent (`references/special-modes.md`).

---

## Step contract (the command skills follow these numbers)

| Step | vlbipy step(s) | CASA | AIPS |
|---|---|---|---|
| S1-S3 | `import_data`, `plot_raw` | `importfitsidi`, `listobs` | FITLD, INDXR, LISTR |
| S4 | `flag_apriori`, `flag_from_file`, `flag_autocorr`, `flag_quack` | `flagdata` list/quack | UVFLG, QUACK |
| S5-S6 | `ai_prediagnostic`, `a_priori` | `gencal` tsys/gc/eop, `accor` | ANTAB, APCAL, VLBATECR, CLCOR PANG |
| S7-S8 | `plot_raw`, `scan_snr`, `flag_tfcrop` | `plotms`, `flagdata` tfcrop | POSSM, VPLOT, SPFLG |
| S9 | `initial_calibration`, `initial_calibration_sbd2` | `fringefit` (sbd) | FRING per IF |
| S10 | `fringefit`, `fringefit_mbd2` | `fringefit` (mbd) | FRING, SNSMO, CLCAL |
| S11 | `bandpass`, `flag_edges` | `bandpass` | BPASS |
| S12 | `flag_outliers`, `second_pass`, `scalar_bandpass`, `ai_flag_distribution` | `flagdata`, `gencal` amp | UVFLG, CLCOR GAIN |
| S13-S14 | `apply`, `reweight`, `third_pass` | `applycal`, `statwt` | CLCAL |
| S16 | `split` | `mstransform` | SPLIT, FITTP |
| S17-S19 | `emerlin_scale`, `selfcal_<source>`, `model_pass`, `clean_<source>` | `gaincal`, `tclean` | CALIB, IMAGR (or Difmap) |
| S20-S21 | `epoch_products`, `final_plots`, `combine`, `report` | `exportfits` | FITTP |

## References
- ERIS 2026 tutorials: https://eris2026.ira.inaf.it/
- DARA EVN continuum workshop, Parts 1-3: https://www.jb.man.ac.uk/DARA/unit4/Workshops/EVN_continuum.html
- EVN data reduction and imaging tutorial (CASA): https://evlbi.org/evn-data-reduction-and-imaging-tutorial
- AIPS Cookbook, Ch. 9 "Reducing VLBI data in AIPS" and Appendix C
- Thompson, Moran & Swenson 2017, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.
- Walker, "Practical VLBI Imaging" (VLBI and the VLBA, 1995); Schwab & Cotton 1983, AJ 88, 688 (global fringe
  fitting); Cornwell & Fomalont (self-calibration, Synthesis Imaging II); VLBA Sci. Memo 37 (amplitude calibration)
- Reid & Honma 2014, ARA&A 52, 339 (VLBI astrometry); Pradel, Charlot & Lestrade 2006, A&A 452, 1099 (phase-referencing
  astrometric accuracy); Rioja & Dodson 2020, A&ARv 28, 6 (precise astrometry, MultiView)
- Marti-Vidal, Perez-Torres & Lobanov 2012, A&A 541, A135 (over-resolution of compact sources)
- e-MERLIN amplitude calibration: Perley & Butler 2017, ApJS 230, 7 (3C286 flux scale); Moldon 2021, eMCP: e-MERLIN
  CASA pipeline, ascl:2109.006 (https://github.com/e-merlin/eMERLIN_CASA_pipeline, `casa6` branch: 3C286 model
  images and the resolved-flux factor `calc_eMfactor`); e-MERLIN Data
  School tutorial (https://www.e-merlin.ac.uk/DataSchool_tutorial.pdf); DARA/ERIS e-MERLIN calibration tutorial
  (https://www.jb.man.ac.uk/DARA/ERIS22/calibration.html: flux resolved out by e-MERLIN on 3C286, Fenech et al.)
