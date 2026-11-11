---
name: reduce-vlbi-vlbipy
description: Run and review VLBI reductions with vlbipy (and difmapy) - install, project layout, configuration, the CLI and Python API, the pipeline steps and how to resume them, the AI mode, campaigns, and how to read a finished run (dashboard, summary.md, comments, amplitude corrections) and act on it. Use when a user reduces EVN/VLBA/LBA/e-MERLIN data with vlbipy, asks what a vlbipy step or setting does, or wants a vlbipy run checked; the science is in /reduce-vlbi-data.
---

# VLBI reduction with vlbipy

vlbipy drives a whole VLBI reduction (import -> a-priori -> flagging -> instrumental and global fringe fit ->
bandpass -> scalar bandpass -> reweight -> split -> self-calibration and imaging with difmapy -> report) through one
`VLBIObs` object, from the CLI or Python. `/reduce-vlbi-data` explains why each step exists and how to judge it (same
step numbers, S0-S21); `/reduce-vlbi-casa` and `/reduce-vlbi-aips` give the manual equivalents.

**Before answering about behaviour, check the installed version** (`vlbipy --version`) and, when in doubt, its code
and `docs/`: this skill describes the version it ships with.

## Install and environment

```bash
pip install -e ".[daskms]"       # from a checkout: CASA backend + numpy calibration + difmapy imaging (recommended)
pip install -e .                 # CASA backend only (no imaging/self-calibration)
pip install -e ".[all]"          # + AIPS (ParselTongue, stub), notebook extras
vlbipy skills install            # make these skills available to your coding agent (see `vlbipy skills --help`)
```
Python >= 3.12. `~/.casa/data` must exist before casatools is imported (vlbipy creates it). Backends: `casa`
(default), `dask-ms` (CASA for import/flagging/export, numpy solvers, difmapy imaging; set `[global] backend =
"dask-ms"` or `--backend dask-ms`), `aips` (stub), `dummy` (synthetic, for tests).

## Project layout

`vlbipy -p CODE ...` works in `./CODE` (or `[global] work_dir`). Raw files (FITS-IDI, `.antab`, `.uvflg`; VLBA
`<CODE>.idifits`; LBA `<CODE>.FITS`) are looked for in the work dir, its `input_data/` and its parent, nearest first;
EVN data, `.antab` and `.uvflg` are downloaded from the JIVE archive when missing (`--username/--password` for
proprietary data, `--obsdate` if the date cannot be resolved). Products go into the work dir:

| Path | Content |
|---|---|
| `CODE.ms` (Multi-MS) | imported data; CORRECTED_DATA written once, by the final apply |
| `caltables/`, `.caltables.json`, `callibs/` | calibration tables, the persisted chain, per-solve priors |
| `plots/`, `plots/ai/` | diagnostic and calibration-table plots |
| `ai/` | AI-mode reports (`*.prediagnostic.json`, `*.features.jsonl`, distribution flag lists) |
| `selfcal/` | difmapy sessions, gain records (`*.gains.json`), pre-self-cal radplot snapshots |
| `calibrated_data/`, `images/` | per-source split MS/UVFITS, FITS images and search results |
| `html/`, `summary.md`, `CODE.ipynb` | report, summary and notebook |
| `.pipeline_state.json`, `.project.json`, `.run_warnings.json` | step state, project record, accumulated warnings |

## Configuration

Three layers, later winning: built-in `templates/defaults.toml` -> your TOML (`--config file.toml`) -> CLI flags.
Every key in a `[calibration.*]` section is passed to the backend function of that name. Read the installed
defaults (fully commented) with:
```bash
python -c "import vlbipy, pathlib; print((pathlib.Path(vlbipy.__file__).parent / 'templates/defaults.toml').read_text())"
```
```toml
[global]
project = "exp"
observatory = "EVN"           # EVN | VLBA | LBA
backend = "dask-ms"
reference_antenna = ["EF"]

[sources]
fringe_finders    = ["3C345"]
phase_calibrators = ["PHASECAL"]
targets           = ["TARGET"]
check_sources     = []
```
Undeclared roles are inferred from the schedule structure (`.source_roles.json`). Settings worth knowing:

| Setting | Meaning |
|---|---|
| `[import] use_embedded_tsys` | EVN: keep the Tsys/GC embedded in the FITS-IDI instead of the archive `.antab` (default false: the archive `.antab` replaces them) |
| `[import] replace_tsys` | force replacing the Tsys/GC in the FITS-IDI with the ANTAB ones |
| `[calibration] ionos`, `ionos_max_ghz` | dispersive delay in the fringe fit below 8 GHz (`--no-ionos` disables) |
| `[calibration.mbd] minsnr`, `subarray_km` | fringe-fit threshold; sub-array re-solve for antennas a resolved calibrator leaves unsolved |
| `[calibration.mbd_smooth] window` | delay/rate smoothing window per field (0 = apply unsmoothed) |
| `[calibration.scalar_bandpass] level_hands` | level both hands together (corrects RCP/LCP ratios) |
| `[calibration.emerlin] enabled`, `hold_reference` | e-MERLIN flux scale on 3C286; Jb is solved with the outstations unless `hold_reference` |
| `[phase_referencing] strategy`, `inverse` | how calibrator phases reach targets; inverse phase referencing (`--inverse-phaseref`) |
| `[selfcal] bayes_per_if`, `bayes_if_sigma` | per-station Bayesian gscale with per-subband shrinkage |
| `[selfcal] time_amplitude`, `time_amplitude_transfer` | per-scan amplitude self-cal; transfer of the calibrator's curve to targets (off by default) |
| `[imaging] method`, `max_components` | `modelfit` with one Gaussian by default; CLEAN only on explicit request |
| `[ai] enabled`, `repair_tsys`, `flag_distribution` | AI mode switches |

## Running

```bash
vlbipy -p exp -t TARGET --phasecal PHASECAL --fringe-finder 3C345   # whole pipeline (= vlbipy pipeline ...)
vlbipy pipeline --list-steps -p exp                                  # steps in order, with their status
vlbipy pipeline -p exp --from-step fringefit                         # redo from a step onward
vlbipy pipeline -p exp --scratch                                     # start over (resets flags and CORRECTED_DATA)
vlbipy import -p exp                                                 # import and print the summary
vlbipy calibrate a_priori -p exp                                     # single stages: calibrate, flag, plot, export
vlbipy report -p exp --open                                          # rebuild html/ and summary.md
vlbipy notebook -p exp                                               # the reduction notebook
vlbipy applycal target.ms selfcal/exp.SRC.phase                      # apply tables to any MS
vlbipy combine --config campaign.toml -p EXPA EXPB EXPC              # multi-epoch stage
```
Python (same object, same state on disk):
```python
from vlbipy import VLBIObs
obs = VLBIObs("exp", network="EVN", target="TARGET", phasecal="PHASECAL", fringe_finder="3C345")
obs.run()                                   # or step by step:
obs.import_data(); print(obs.summary())
survey = obs.calibrate.scan_snr(); survey.rank_antennas()
obs.calibrate.a_priori(); obs.flag.apriori(); obs.plot.diagnostics(column="data")
[t.cal_type for t in obs.gaintables]        # the chain so far
obs.calibrate                               # a namespace lists its operations
```

## Pipeline steps (`--from-step` values) and their S-numbers

| Step | S | What it does |
|---|---|---|
| `import_data` | S1-S3 | find/download FITS-IDI, `.antab`, `.uvflg`; attach the archive `.antab` (EVN); import to a Multi-MS |
| `ai_prediagnostic` | S6 | raw data with the a-priori flags only (flag version `ai_apriori`): antenna problems, Tsys plan |
| `a_priori` | S5-S6 | Tsys (repaired by the AI plan) + gain curve; EOP and ACCOR for DiFX; TEC below 8 GHz |
| `flag_apriori`, `flag_from_file`, `flag_autocorr` | S4 | observatory flags, user flag file, autocorrelations |
| `plot_raw` | S7 | raw diagnostics: scan SNR, auto/cross spectra, time evolution |
| `flag_quack`, `flag_tfcrop`, `flag_smart_precal` | S4, S8 | measured or configured quack; on the raw calibrators the smart (antenna-based) outlier pass on dask-ms, else per-baseline tfcrop |
| `scan_snr` | S7 | fringe SNR per scan, antenna and hand |
| `initial_calibration` | S9 | instrumental delay (SBD) on the selected scan(s) |
| `fringefit` | S10 | global fringe fit (MBD), smoothed copy applied |
| `bandpass`, `flag_edges` | S11 | complex bandpass; edge channels from its roll-off |
| `initial_calibration_sbd2`, `fringefit_mbd2` | S9-S10 | SBD and MBD again on bandpass-corrected data |
| `apply`, `flag_aoflagger`, `flag_outliers`, `flag_smart_phasecal` | S12-S13 | apply; off-source flags, then outlier flagging of the calibrated data (smart pass on the phase-reference sources on dask-ms) |
| `second_pass` | S12 | the whole chain re-solved on the cleaner data |
| `scalar_bandpass` | S12 | one amplitude per antenna and subband, hands together |
| `reweight`, `flag_outliers_reweighted`, `flag_smart_phasecal_reweighted`, `third_pass` | S14 | statwt, flag, re-solve |
| `flag_edges_final` | S11 | final edge flags |
| `emerlin_scale` | S17 | e-MERLIN flux scale on 3C286 (only with outstations present), then `emerlin_check` |
| `ai_flag_distribution` | S12/S18 | distribution flags of the calibrated data |
| `flag_smart_target` | S12 | smart outlier pass on the targets and check sources with their final calibration, before the split |
| `split` | S16 | per-source MS + UVFITS |
| `model_pass` | S10-S17 | re-solve against the calibrators' structure models when the phase calibrator is structured |

After the split, per-source steps run (`selfcal_<source>`, `joint_<source>`, `clean_<source>`, `epoch_products`,
`final_plots`). Resuming with `--from-step` forgets them and drops their tables. A re-run of the AI distribution
flagging restores its previous flags first. On dask-ms every `apply` but the last is deferred (readers apply the
chain on the fly), so CORRECTED_DATA is only complete at the end.

## AI mode

```bash
vlbipy ai -p exp                    # pre-diagnostic of the raw data + Tsys plan (needs import + a-priori flags)
vlbipy ai tsys -p exp               # the Tsys decisions, series by series
vlbipy ai calibrated -p exp         # residual antenna gains of the corrected data
vlbipy ai flag --dry-run -p exp     # distribution flags: list and report only
vlbipy flag smart-outliers -p exp --field PHASECAL --dry-run    # antenna-based outliers; report in flags/
vlbipy ai rules -p any              # the expert rules and their evidence
```
The rules decide; the shipped decision forest only reports its labels next to theirs (`vlbipy ai predict`). Every
decision is a feature row in `ai/CODE.features.jsonl`.

## Reviewing a run

Read `references/reading-a-run.md` for the report layout and what each comment means. The short version:
1. `summary.md` first block = the **General comments** (problems, warnings, notes; per antenna, per source, pipeline
   warnings). Each problem names its evidence (UTC ranges, scan numbers, subbands).
2. Overview: antennas with no data / partial band; the scan x antenna grid (SNR < 3 red, 3-10 orange, > 10 green).
3. Calibration page: the chain, the instrumental-delay scan(s), the **amplitude corrections** table (green within 10%,
   orange within 30%, red beyond), every caltable plot.
4. Final results: image S/N (peak/rms), offsets from the phase centre, detections above `[imaging] search_sigma`
   (8), radplots with the pre-self-cal data in grey underneath.
5. Act with `reduce-vlbi-data/references/diagnostics.md` (the /reduce-vlbi-data skill), then re-run with
   `--from-step` (flags on calibration solutions -> `apply`; flags on calibrator data -> `second_pass`; a new
   reference antenna or scan choice -> `initial_calibration`).
