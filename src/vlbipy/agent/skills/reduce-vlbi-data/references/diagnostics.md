# Diagnostics: symptom -> likely cause -> action

| Symptom | Likely cause | Action |
|---|---|---|
| No fringes to one station on any scan | station failure, wrong polarisation labels, clock error beyond the window | station feedback, widen the search window, check the autocorrelations |
| Fringes only in some subbands for a station | the station observed part of the band (e-MERLIN, a BBC down) | flag the missing subbands for it; don't count them as flagged |
| Long baselines never fringe on the phase calibrator, short ones do (signal falls with baseline length) | calibrator resolved: scatter broadening (similar for nearby calibrators) or intrinsic structure (source-specific) | sub-array fringe fit for the short-baseline antennas, flag inter-sub-array baselines; another calibrator if intrinsic |
| Phase jumps between subbands after the instrumental delay | instrumental phases changed (station reset), or the delay scan too short | solve on a scan near the jump, split in time |
| Delay/rate solutions scatter like noise | low S/N, wrong interval, weak calibrator | combine subbands, adjust the interval, raise the threshold, restrict windows |
| An antenna's solutions are noise on the phase calibrator but fine on the fringe finders | faint calibrator for that antenna | keep its high-S/N fringe-finder solutions; don't drop the antenna everywhere |
| A fringe-fit or bandpass table with every solution flagged | wrong scan/source choice, no data in the selection | never apply it; keep the previous table, re-plan the scans |
| The instrumental delay was solved on a weak source and the fringe fit later finds nothing | scan chosen by coverage before strength | re-choose the scan by source strength first |
| Inverse phase referencing: every solution fails | the target is not detected by the fringe fit (S/N too low) | normal phase referencing, or lower the S/N threshold towards 3 only if it recovers more baselines |
| Amplitudes of one station systematically off on calibrators | Tsys/gain-curve error, pointing | gain factor from a calibrator model (S17) |
| Raw amplitude drops that the Tsys does not follow | absorption the ground Tsys misses, or the Tsys is a constant SEFD | time-dependent amplitude self-cal on the calibrators; opacity correction at high frequency |
| Tsys wiggles absent from the raw data | Tsys measurement noise | smooth the Tsys over a long window |
| Gain curve near zero or negative over the observed elevations | bad polynomial, evaluated against the wrong variable (elevation vs zenith angle) | replace/refit over the observed range, re-check positivity |
| RR/LL amplitude offset for a station | sampler bias (DiFX), receiver problem | sampler correction, per-hand factor; a scalar bandpass must level the hands |
| Amplitude jumps between subbands for one antenna after the bandpass | per-subband gains not captured by Tsys | scalar bandpass (S12) or per-subband gscale |
| "Decorrelation" on the noisiest antenna only | noise estimated from channel differences (correlated channels) | estimate the noise from time differences |
| Distribution flagging removes a third of the best baselines | significance without an absolute threshold at high S/N | require absolute departures (jitter > 20%, bins > 10%) |
| Target images degrade after a calibrator scan was flagged | target scans next to the flagged calibrator scan kept (extrapolated phases) | flag the adjacent target scans for that antenna |
| Stripes / symmetric patterns in the image | residual antenna errors, bad baselines or time ranges | residual visibilities, flag, self-cal |
| Non-closing errors, dynamic range stuck | uncalibrated or averaged-over variable bandpass, correlator problem | better bandpass, closure phases per baseline |
| Target fuzzy, peak << integrated flux | phase-referencing decorrelation | shorter intervals, TEC corrections, check source, target self-cal if bright |
| Self-cal lowers peak and rms together | wrong model absorbed into the gains | back to the original data, simpler model, longer intervals |
| Amplitude gains drift in self-cal | model missing short-spacing flux | uv-range limits, mean gain constrained, include extended flux |
| Time-dependent amplitude self-cal: all stations shifted together | mean gain of each IF held at 1 while one station needs a large correction | float the mean scale; flag corrections > ~x5 from a station's median |
| Check source worse after transferring a calibrator's time curve | the per-scan curve carries the calibrator's structure | apply the time curve to the calibrator only |
| Target missing on the long baselines, or "no data in uv range" in difmap | map cell too coarse (difmap grids only within 1/(4 cell)) | cell <= beam/4.6 (search maps too) |
| difmap loses whole IFs | one flagged channel flags the IF average | load channel-averaged data |
| e-MERLIN 3C286 amplitudes off after the flux-scale step, mirrored structure | model image read with the wrong RA orientation | check CDELT signs when converting the model |
| e-MERLIN amplitudes moved by full-array self-cal | their amplitudes not held, or held with their phases | hold e-MERLIN amplitudes, solve their phases |
| Low amplitudes and scattered phases at scan starts | antenna still slewing | quack per antenna (S4) |
| A-priori flags remove hours of good data | wrong end time in the flag file | check the flag time ranges against the data |
| VLBA data unflagged despite a FLAG table in the FITS-IDI | the import did not apply it (or applied it untimed) | convert to time-ranged flag commands and apply |
| Check-source offset >~ a fraction of the beam | phase-referencing systematics | troposphere/ionosphere corrections, closer calibrator; report as systematic error |
