# vlbipy development notes

## Build & test

- Editable install in `/home/marcote/.venv` (`uv pip install -e .`); `difmapy` is the sibling checkout.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q` (zarr's auto-loaded pytest plugin breaks start-up
  with an older numpy). The AI mode has its own synthetic-data tests in `tests/test_ai.py`.
- `vlbipy -p EB104G ...` uses `./EB104G` as its work dir. Raw files (FITS-IDI, `.antab`, `.uvflg`) are looked
  for in the work dir, its `input_data/` and its parent (`fitsidi.search_dirs`, nearest first), so files
  downloaded by hand into the directory where vlbipy is started are found; the converted `.flag` and all
  products go into the work dir.

## AI mode (`src/vlbipy/ai`)

Data-driven diagnostics and decisions; user documentation in `docs/usage/ai.md`.

| module | role |
|---|---|
| `visstats` | the statistics cube: per (field, scan, baseline, spw, hand, time bin) vector/scalar averages, power, noise; reads any column or **any partial calibration chain on the fly** (`chain=`), or the flags of a saved flag version (`flag_version=`) |
| `antenna` | robust antenna-based decomposition of baseline log-amplitudes: gains vs time, subband offsets, hand ratios |
| `tsys` | Tsys assessment, checks against the raw amplitudes, the repair plan and `repair_table` |
| `distributions` | Rice / von Mises tests of calibrated data and the `flagdata` commands they imply |
| `knowledge` | the expert rules, each with its evidence (`vlbipy ai rules`) |
| `features` / `model` | labelled feature rows, review files, the numpy decision forest (`data/ai/model.json`) |
| `namespace` | `obs.ai`: `prediagnose`, `repair_tsys`, `flag_distribution`, `postdiagnose` |

Invariants worth keeping:

- **Noise comes from time differences, never from channel differences.** SFXC channels have adjacent-channel
  noise correlation 0.50 (windowed FFT); channel differences underestimate the variance ~3x, and the variance of a
  channel average is `sigma^2 (1 + 2 rho1) / n`. `visstats` measures `rho1` per hand and stores `noise2_avg`;
  every SNR/z-score must use it. The phase-independent amplitude `sqrt(power - noise2)` is only as good as
  `noise2` - with the channel estimator it invented a "decorrelation" of T6 on EB104G.
- The pre-diagnostic reads the raw data with the **a-priori flags only** (flag version `ai_apriori`, saved right
  after `flag.apriori`, reused afterwards - never re-saved from a later FLAG).
- Raw-data amplitude tests are only meaningful against the Tsys: a raw drop the Tsys follows is calibrated away.
  `tsys.check_trend_against_raw` returns the slow and the scan-to-scan slope of raw log-amplitude against
  `-0.5 log Tsys`: ~1 real, ~0 measurement noise, >1.2 with high correlation absorption the ground Tsys misses.
- Distribution tests compare each baseline with **its own** behaviour (running complex median along time, model
  noise, a fitted fractional jitter); "bad" must be bad in absolute terms at high SNR (scan jitter > 20%, bins
  departing > 10%), or a 2-degree atmospheric wobble at SNR 100 flags a third of the best baselines.
- Never write into another experiment's directory to collect training data: use the read-only path
  (`visstats.visibility_stats` + `tsys.read_syscal` + `AINamespace._diagnose_raw/_assess_tsys` on a stub
  observation whose `work_dir` is elsewhere).

Adding an experiment to the training set: run `vlbipy ai -p CODE` (or the read-only path), review the rows in
`<work_dir>/ai/CODE.features.jsonl`, write a review file like `data/ai/EB104G.review.json` for the labels that are
wrong, copy the reviewed rows to `data/ai/CODE.features.jsonl` and run `vlbipy ai train -p any`.

## Calibration lessons (EB104G, 2026-10-30)

- The scalar bandpass levels the two hands together (`level_hands`): nothing downstream can correct an RCP/LCP
  offset (difmap gains are per station).
- Bayesian gscale per station (`[selfcal].bayes_per_if = false`); a time-dependent, median-normalised amplitude
  self-cal per scan follows what the Tsys cannot (`[selfcal].time_amplitude`). A fringe finder's time curve
  applies to itself only.
- The time-dependent amplitude solve uses `float_scale=True`: difmap's default holds the mean gain of each IF at
  one, and one station that needs a large correction (T6) drags all the others (chi-squared x60 on EB104G).
  Corrections more than x5 from a station's median are written flagged (no signal), not applied.
- The time-dependent curve corrects the calibrator it was solved on; transfer to targets is opt-in
  (`[selfcal].time_amplitude_transfer`): on EB104G it made the check source worse (reduced chi-squared 28.9 ->
  34.9, image SNR 204 -> 180) because a per-scan solve against the calibrator's model carries its structure.
- A constant Tsys with DPFU = 1 (JB, T6) is an SEFD: the a-priori calibration then carries no time dependence.
- `mbd_smooth` (delay/rate smoothing) was suspected of decorrelating the calibrators and cleared: identical
  coherence with and without it.
