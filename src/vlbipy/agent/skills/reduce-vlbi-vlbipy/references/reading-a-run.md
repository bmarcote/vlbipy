# Reading a vlbipy run

Everything here is rebuilt by `vlbipy report -p CODE` from caches on disk (`.metadata.json`,
`.flag_statistics.json`, the `.snr` table, `.cal_selection.json`, `selfcal/*.json`, `images/*.search.json`,
`.pipeline_state.json`, `.run_warnings.json`), so a run can be reviewed after its measurement set is deleted.
`html/` links plots by relative path: keep it next to `plots/` and `images/`.

## General comments (`summary.md` first block, Diagnostics page first section)

Each comment has a severity (`problem`, `warning`, `note`) and a group:

| Group | Typical comments | What to check / do |
|---|---|---|
| Data and calibration | scan(s) of the instrumental delay and which antennas were solved there; poor tables; failed steps | was the delay solved on a strong source with every antenna? a failed step stops what follows |
| Sources | search and image results per source; self-cal remarks (poorly constrained, not applied, Bayesian findings) | detection S/N (> 8 unknown position); self-cal that was not applied usually means the model did not fit |
| Antennas | no data / fully or heavily flagged; partial band; stretches of >= 3 scans without data or fringes (UTC range, scans); a dead polarisation; amplitude corrections beyond 10% / 30%; "the unconstrained gscale wants > 10% but the applied factor stayed within 5%" | station feedback; Tsys vs raw data; a red correction needs an explanation (pointing, Tsys, swap) |
| Warnings raised by the pipeline | everything logged as a warning during the run (accumulated across resumes) | read them all once |

A scan counts against an antenna ("no fringes") only when at least half of the other antennas fringed there. The
"no fringes" and dead-polarisation rules see every scan at the end of a run but only the calibration-selection scans
in a later `vlbipy report` (the all-source survey is in memory).

## Pages

- **Overview:** observation setup; antennas (subbands recorded, data health); sources with a sky map (each target
  joined to its nearest phase calibrator with the separation); scans as a grid, one square per antenna. Chip states:
  coloured by SNR (< 3 red, 3-10 orange, > 10 green), `refant` (no SNR of its own), `nodata` (fully flagged),
  `nofringe` (surveyed, no solution), `unsurveyed`, `absent` (not in the scan). An antenna header in red delivered no
  data at all.
- **Diagnostics:** auto- and cross-correlations of the diagnostic scans; **time evolution**: raw amplitude and phase
  vs time on every baseline to a reference antenna, coloured by source, one subband, both hands. A second reference
  antenna appears when the first has gaps (no data, or lag-1 coherence < 0.45), a second subband when not every
  antenna recorded the first.
- **Calibration:** the chain table; instrumental-delay scan(s); flagged data; the **amplitude-correction** table per
  self-calibrated source (difmapy gscale per antenna and subband, 1.00 = unchanged; unsolved cells empty; last column
  = median over solved subbands); caltable plots: Tsys (already de-spiked/repaired), gain curves (against true
  elevation, observed range shaded; must be positive there), SBD (no rates), MBD (solved and smoothed), bandpass,
  scalar bandpass (one level per subband), self-cal tables.
- **Final results:** per source: downloads (calibrated UVFITS, split MS, FITS per weighting), uv coverage, radplots
  (pre-self-cal data in grey underneath: an offset means self-cal moved the amplitudes), spectra, light curves,
  images; result cards with image S/N (peak/rms), offset from the phase centre and the detection test.

## Files to open when a comment needs evidence

| Question | File |
|---|---|
| Was the Tsys repaired, and why? | `ai/CODE.prediagnostic.json`, `plots/ai/` |
| What did the distribution flagging remove? | `ai/CODE.ai_flag_distribution.flags.txt` (+ report) |
| Which scan and antennas solved the delays? | `.cal_selection.json` |
| Self-cal corrections per antenna/subband | `selfcal/CODE.SOURCE.gains.json` |
| e-MERLIN flux scale check | `emerlin_check_<stage>.json` |
| Model pass before/after | `ai/CODE.model_pass.json` |
| Step status and outputs | `.pipeline_state.json` (or `vlbipy pipeline --list-steps -p CODE`) |
| Multi-epoch fluxes and image statistics | `combined/<name>.campaign.json` |

## Copies of a run

`VLBIObs.load(dir)` uses the `work_dir` stored inside `.project.json`, and constructing the object rewrites that file.
To review a copy, point that field at the copy first, or the original run is written to.
