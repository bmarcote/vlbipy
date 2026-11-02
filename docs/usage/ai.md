# AI mode

The AI mode (`vlbipy.ai`, `obs.ai`, `vlbipy ai`) turns the measurement set and its calibration tables into
compact, *labelled* diagnostics, and lets the pipeline act on them: it repairs the system temperatures before
they are applied, flags whatever the calibrated data show outside the noise distribution, and writes every
decision down as a feature row a small model learns from. It is on by default (`[ai].enabled`) and needs a
backend with a measurement set on disk (CASA, dask-ms).

It answers the questions a careful observer asks by eye, with numbers:

| Question | Where | Answer from |
|---|---|---|
| Did an antenna lose sensitivity, or jump from scan to scan? | raw data, before calibration | antenna gain over time per calibrator (`antenna.gains_vs_time`) |
| Is a Tsys hand noise? Is a subband's high Tsys real? Does the Tsys follow the amplitude? | Tsys table vs raw data | `tsys.assess`, `tsys.check_against_raw`, `tsys.check_trend_against_raw` |
| Which step introduced a time dependence? | any partial chain | `visstats.visibility_stats(..., chain=[...])` |
| Are RR and LL at the same level? Are the subbands level? | raw and calibrated data | `antenna.hand_ratios`, `antenna.subband_offsets` |
| Do the calibrated visibilities follow Rice / von Mises statistics? | calibrated data | `distributions.assess` |

## The physics it relies on

* SFXC (and DiFX after ACCOR) delivers **correlation coefficients**, `rho_ij = S / sqrt(SEFD_i SEFD_j)`, with
  `SEFD = Tsys / (DPFU * gain_curve)`. A real Tsys change moves the raw amplitude by `-0.5 log Tsys`; one the raw
  data do not show is a measurement error. Raw amplitudes are measured without any phase calibration with the
  phase-independent estimator `sqrt(<|V|^2> - sigma^2)`.
* The two circular hands see the same sky and atmosphere, and calibrators are unpolarized in circular hands:
  `|RR| = |LL|`, and both hands' Tsys move together.
* A VLBI band is far too narrow for a spectral index: every subband has the same calibrated amplitude.
* Calibrated visibilities are a slow signal plus complex Gaussian noise. The noise is measured from the data
  (differences of consecutive integrations), including the **correlation between adjacent channels** that
  SFXC's windowed FFT introduces (`rho1 = 0.50` measured on EB104G): ignoring it makes every channel average
  look `sqrt(2)` less noisy than it is, and channel-difference noise estimates three times too small.
* Source structure is baseline-based and slow; anything antenna-based or fast is instrumental. Two sources that
  show the same antenna trend settle it.

## In the pipeline

```
flag.apriori ─► ai.prediagnose ─► calibrate.a_priori (Tsys repaired) ─► ... ─► final apply
            ─► ai.flag_distribution ─► self-calibration (incl. time-dependent amplitudes) ─► images ─► ai.postdiagnose
```

**`ai.prediagnose`** (step `ai_prediagnostic`) runs on the raw data with the a-priori flags only (saved as the
flag version `ai_apriori`). It writes `<work_dir>/ai/<code>.prediagnostic.json`, the raw statistics cube
(`.raw_cube.npz`), the figures `plots/ai/<code>.ai.raw_gains.png` / `.raw_levels.png`, and a Tsys plan. Antenna
problems are reported as warnings (*"JB on J1031+7441: antenna_amplitude_scatter (scan scatter 11.3% vs array
0.6%)"*).

**Tsys repair** (`[ai].repair_tsys`) replaces the uniform per-field smoothing of the Tsys table, series by series:

| Finding | Repair |
|---|---|
| noisy hand (scatter ≥2.5× its partner's, raw data do not follow it) | the steady hand's time dependence, scaled to the noisy hand's median |
| spurious high subband (raw amplitudes not lower by the matching sqrt) | rescaled to the level the raw data imply |
| real variation that *under*-corrects (raw slope 1.2–3, corr > 0.9) | `Tsys (Tsys/median)^(s-1)`: absorption the ground-level Tsys misses |
| scan-to-scan fluctuations the raw data do not show | 30-min smoothing |
| real fast variation | 5-min smoothing |
| unphysical values | flagged |
| nominal (constant) Tsys | left alone; the time-dependent amplitude self-calibration takes over |

Fields are smoothed together unless the source measurably raises the Tsys (a significant step at field changes).

**Subband and hand levelling**: the scalar bandpass normalises both hands of an antenna together
(`[calibration.scalar_bandpass].level_hands`), so it also corrects the RCP/LCP ratio - which nothing else can:
difmap solves one gain per station.

**`ai.flag_distribution`** (step `ai_flag_distribution`, after the final apply, before self-calibration): every
field's calibrated cube (10 s bins) is compared with a running complex median along time. The expected scatter is
the bin's thermal noise, the noise of the median, and a fractional jitter fitted per baseline so that the median
`|z|^2` is that of an exponential law. Flagged, as `flagdata` list commands (`<work_dir>/ai/<code>.ai_flag_distribution.flags.txt`,
reversible: the version `before_ai_flag_distribution` is saved):

* single bins beyond `p < 1e-7` that also depart by more than 10% of the signal;
* scans of a baseline/subband/hand whose own jitter exceeds 20% (high SNR; phase jumps) or whose reduced
  chi-squared exceeds 2 (low SNR);
* subbands off the band median by >15% and >6 sigma;
* scans where a baseline's amplitude leaves its own slow trend by >20% and >5 sigma.

**Self-calibration** (difmapy): the Bayesian gscale gives one correction per station (`[selfcal].bayes_per_if =
false`; the scalar bandpass levels the subbands), then an amplitude self-calibration per scan
(`[selfcal].time_amplitude`), solved with a floating flux scale (difmap's default holds the mean gain of each IF,
and one far-off station drags the rest), kept when the fit improves by 1% and normalised to median 1 per station -
only the time dependence the Tsys cannot follow is added; a correction more than x5 from the median flags the data
instead (no signal). The curve corrects the calibrator it was solved on; transferring it to the targets is
opt-in (`[selfcal].time_amplitude_transfer`), because a per-scan solve against the calibrator's model carries some
of its structure: on EB104G the transfer made the check source worse (reduced chi-squared 28.9 -> 34.9, image SNR
204 -> 180).

**Model pass** (`[calibration.model_mbd]`, step `model_pass`, after the first images): every calibrator's
difmapy session ends by fitting up to `[selfcal].structure_components` Gaussians (3 by default: nearly every
calibrator is one Gaussian, a core-jet two or three) to its self-calibrated data, a measurement of its structure. When the phase calibrator's structure model has visibility phases of
`min_phase_deg` (5°) or more on its baselines (90th percentile, measured from its brightest component), the
point-source fringe fit has been putting that structure phase into the delays and rates transferred to the
targets. The calibrators' models are then written to MODEL_DATA (a unit model for every other field); the fringe
fits (dask-ms: `V / M`, weights `w |M|^2`, the model scaled to median amplitude 1 per scan; CASA: natively) and the
bandpass divide by them; the chain is re-derived from the delays on; the calibrators are self-calibrated against
their structure models (`components`, so the phase self-cal transferred to the targets does not absorb the
structure either); every source is imaged again. The flags written by the AI distribution flagging are undone
before the re-solve and the flagging is redone afterwards: they were judged against residuals of the point-source
calibration, and on EB104G they removed every TR baseline of a delay scan, which left TR without a single delay
solution once the chain was re-derived on them. Before/after figures of merit: `<work_dir>/ai/<code>.model_pass.json`.
On EB104G (phase calibrator compact, forced pass) nothing changed beyond the noise: check source 69.25 -> 69.29
mJy/beam, target 9.8σ both times.

**`ai.postdiagnose`** writes the residual antenna gains over time, subband offsets and hand ratios of the
calibrated calibrators (`<code>.postdiagnostic.json`, `plots/ai/<code>.ai.postdiagnostic_*.png`).

## Command line

```bash
vlbipy ai -p EB104G                 # pre-diagnostic (needs only the import and the a-priori flags)
vlbipy ai tsys -p EB104G            # the Tsys plan, series by series
vlbipy ai calibrated -p EB104G      # residual antenna gains of the corrected data
vlbipy ai flag --dry-run -p EB104G  # distribution flags: write the list and the report only
vlbipy ai rules -p any              # the rule base (Markdown)
vlbipy ai train -p any              # fit the model on the shipped annotated sets
vlbipy ai predict -p EB104G         # model labels for this project's feature rows
```

## Rules, features, model

`vlbipy.ai.knowledge` holds the expert rules, each with the evidence and the experiment that taught it
(`vlbipy ai rules`). Every finding becomes a **feature row** (`<work_dir>/ai/<code>.features.jsonl`): a few
dimensionless numbers that do not depend on the array, the frequency or the flux density (scatter ratios, slopes
against the Tsys, trends, reduced chi-squared, KS distance), the labels the rules gave it, and - where a person
reviewed it - the person's label, which wins (`features.annotate`).

`vlbipy.ai.model` trains one small forest of shallow decision trees per stage (Tsys series, antenna on a
calibrator, calibrated baseline) in pure numpy and stores it as JSON in `vlbipy/data/ai/model.json`: ~200 kB,
about a millisecond per prediction, no GPU and no dependency beyond numpy - it runs on a laptop next to the pipeline.
The annotated training sets ship in `vlbipy/data/ai/*.features.jsonl`; adding an experiment is running the
pre-diagnostic on it, correcting the labels that are wrong, and dropping the file there before `vlbipy ai train`.

### The model that ships (2026-11-01)

Trained on 1729 rows from six experiments: EB104G (reviewed against the PI's review and the raw data) and EB104B,
EB104D, EK053A, EK053B (EVN, 1.6 and 5 GHz) and GL048 (global, 22 GHz, 28 antennas), reduced from scratch with the
current pipeline and labelled by the rules, read-only. 80 trees of depth 6 per stage.

| stage | rows | stratified 5-fold (balanced) | leave one experiment out (balanced) |
|---|---|---|---|
| Tsys series | 1186 | 91% (84%) | 59% (58%) |
| antenna on a calibrator | 83 | 81% (76%) | 70% (59%) |
| calibrated baseline | 460 | 100% (100%) | 99% (99%) |

Held out one experiment at a time (trained on the others, predicting the rules' labels of the one left out; the
EVN rows are from the five-experiment model, before GL048 was added):

| experiment | Tsys series | antenna on a calibrator | calibrated baseline |
|---|---|---|---|
| EB104B | 73% | 93% | 98% |
| EB104D | 88% | 62% | 96% |
| EK053A | 65% | 100% | 95% |
| EK053B | 79% | 71% | 100% |
| GL048 (22 GHz) | 40% | 26% | 98% |

The calibrated-baseline stage carries over to a new band; the Tsys and antenna stages do not: at 22 GHz the Tsys
swings with elevation and weather, and a model that has only seen 1.6-5 GHz calls most of those series
under-correcting or not real (`ok -> tsys_undercorrects`, `ok -> tsys_variation_real`). That is also why the
leave-one-experiment-out Tsys score fell once GL048 joined (it is the one fold with no band like it): the model
needs reviewed experiments in every band (or band-aware features), not more capacity. Within a band the recurring
Tsys confusion is "fluctuations the raw data do not show" against `unphysical_tsys` / noisy hand, the borderline of
a smoothing choice. The rules remain the decision-maker in the pipeline and the model's labels are reported next to
theirs (`vlbipy ai predict`).

## EB104G, the experiment it was built on

EB104G (EVN, 5 GHz, 9 antennas; R4_D phase-referenced to J1031+7441, fringe finder J1048+7143) was reduced end to
end, and the review of the first calibration listed oscillating calibrator amplitudes, an offset between the two
hands, low subbands, scattered short-baseline phases and an unexplained UR-T6 drop. What the AI mode found:

| Finding | Evidence | Fix |
|---|---|---|
| T6 loses sensitivity x2.5-5 during the run | every T6 baseline (UR-T6 at 50 Mλ like YS-T6 at 150 Mλ) drops by the same factor, on both calibrators; nominal Tsys (26 K, DPFU 1) | time-dependent amplitude self-cal on the calibrators; its dead integrations flagged |
| JB scan-to-scan scatter 11%, wander ±30% | raw data, nominal Tsys (80 K, DPFU 1) | time-dependent amplitude self-cal on the calibrators (drift 80% -> 1%) |
| SR RCP Tsys spw 5-7 is noise | 15% point-to-point vs 0% in LCP; raw slope 0.00; injected 5-6% per-scan noise | LCP time dependence at RCP's level |
| UR RCP Tsys noisy | 3.8% vs 0.3% point-to-point; raw fast slope ~0.1 | LCP time dependence |
| WB spw6 RCP, spw7 LCP high Tsys are not real | x1.67 / x1.36 Tsys, raw amplitudes -4% / +1% (expected -23% / -14%) | rescaled |
| EF spw6 LCP high Tsys is real | x1.32, raw -10% (expected -13%) | kept |
| YS Tsys x2.4 in the last 1.5 h is real but under-corrects | raw amplitude slope 2.1 vs Tsys (corr 0.98): weather absorbs the signal as it raises the Tsys | opacity-like exponent |
| RR/LL offsets UR -12%, O8 -6%, SR -5% after calibration | UR DPFU R/L = 1.13 (looks swapped) and RCP Tsys 12% low; O8 RCP Tsys 10% low; raw hands equal | hand-joint scalar bandpass |
| short-baseline phases off the distribution | TR phase jumps of ±180° within scans (scans 1, 32, 64, 96, 141, 142); TR antenna-based | distribution flags |
| per-IF Bayesian corrections of a weak station are noise | T6 on the fringe finder 5.3-7.6 across IFs, carried to every field | one correction per station |
| band-edge channels left ~14% low | channel test on the calibrators' spectra | flagged (`AI_CHANNEL`) |
| the phase calibrator's time curve does not transfer | check source: reduced chi-squared 28.9 -> 34.9, SNR 204 -> 180 with it | applied to the calibrator only (opt-in transfer) |
| difmap amplitude self-cal diverged (chi-squared x60) | the mean-gain normalisation dragged by T6 | solved with a floating scale, normalised per station |
