# Command line

The CLI runs one stage at a time, so you can inspect the products before going
on. Every subcommand takes the same project and source arguments.

```bash
vlbipy <command> -p <project> [-t TARGET] [--phasecal NAME] [--fringe-finder NAME] [--check-source NAME]
```

| Command | Purpose |
|---|---|
| `run` | the whole pipeline (see [Full pipeline](pipeline.md)) |
| `import` | download / import raw data and read metadata |
| `calibrate` | run calibration steps, or one named step |
| `flag` | run flagging modes |
| `plot` | produce diagnostic plots |
| `export` | split per source and export |
| `applycal` | apply calibration tables to any measurement set (see [below](#applying-tables-to-any-measurement-set)) |
| `report` | rebuild the HTML report and `summary.md` from the products on disk (see [below](#the-report)) |
| `notebook` | write the reduction notebook `<work_dir>/<code>.ipynb` (see [below](#the-reduction-notebook)) |

`vlbipy --version` prints the installed version, which is the `version` written
in `pyproject.toml`.

## A worked reduction

```bash
# 1. Import: finds FITS-IDI locally or downloads from the EVN archive,
#    fetches .antab / .uvflg, imports, and prints the observation summary.
vlbipy import -p rsm07 -t 3C395 --phasecal 'J1848+3219' --fringe-finder '3C345'

# 2. A-priori amplitude calibration (Tsys + gain curve), de-spiked and plotted.
vlbipy calibrate a_priori -p rsm07 ...

# 3. Observatory flags plus autocorrelations.
vlbipy flag apriori -p rsm07 ...

# 4. Instrumental calibration: antenna/scan selection, SBD -> bandpass -> SBD.
vlbipy calibrate instrumental -p rsm07 ...

# 5. Global fringe fit and apply.
vlbipy calibrate fringefit -p rsm07 ...
vlbipy calibrate apply -p rsm07 ...

# 6. Diagnostics, then the calibrated products.
vlbipy plot corner spectrum -p rsm07 ...
vlbipy export -p rsm07 ...
```

## Applying tables to any measurement set

`vlbipy applycal` applies calibration tables to a measurement set without a
project: a self-calibration table from difmapy, or a table solved on another
set. The set and the tables can be given in any order.

```bash
vlbipy applycal GL048.mms selfcal/GL048.J1305-1033.phase
vlbipy applycal sbd.K bpass.B target.ms --interp linear nearest,linear --gainfield J1305-1033 ''
```

- On a set that has CORRECTED_DATA, the tables are applied **on top of it**:
  their corrections multiply what the column holds, their flags are added and
  the weights are scaled. Without one, they are applied to DATA and written
  to a new CORRECTED_DATA, as CASA's `applycal` does. `--from-data` forces
  the latter on a set that has the column, replacing it.
- The parallactic-angle correction is off by default, as in CASA: tables
  applied this way usually go on calibrated data (CORRECTED_DATA, or the
  DATA of a split), which have it already. `--parang` turns it on for data
  that are not calibrated yet.
- A table solved on another set (a split, another epoch) is matched to this
  one by antenna name, and `--gainfield` names the table's own fields.
- `--gainfield`, `--interp`, `--spwmap` and `--calwt` take one value per table
  (one value is used for all); `--field`, `--spw` and `--scan` restrict what
  is corrected. The `--intent`, `--timerange`, `--antenna` and
  `--observation` selections and `--callib` go through CASA's `applycal`,
  which always starts from DATA (`--from-data` is then required on a set
  with CORRECTED_DATA).

## Controlling what re-runs

Steps already recorded as complete are skipped:

```bash
--force                # re-run this step even if it is done
--from-step bandpass   # invalidate that step and everything after it
--scratch              # forget all progress and reset the data
```

## The ionosphere

Below 6 GHz the fringe fit also solves for the dispersive (ionospheric) delay,
because at those frequencies the residual delay is genuinely frequency-dependent
and fitting a single non-dispersive delay leaves phase behind. Above 6 GHz it is
skipped — the effect is negligible there and the extra free parameter only costs
SNR. Every run logs which way it went. To turn it off regardless of frequency:

```bash
vlbipy run -p rsm07 -t 3C395 --no-ionos
```

Because state persists in `.pipeline_state.json`, stopping and resuming days
later works, and so does mixing the CLI with a Python session on the same
project directory. The calibration chain itself is persisted alongside it in
`.caltables.json`, so a resumed step solves on top of the tables the earlier
steps produced rather than starting from uncalibrated data. Resuming from a step
drops the tables that step and the ones after it had produced, since they are
about to be re-derived.

## The report

```bash
vlbipy report -p rsm07            # rebuild <work_dir>/html/ and <work_dir>/summary.md
vlbipy report -p rsm07 --open     # ... and open the overview in the browser
```

The same report is written at the end of every `run`. It is built from the
products and caches on disk, so it can be rebuilt after the measurement set
has been deleted. `html/` links the plots by relative path and has to stay
next to `plots/` and `images/`.

| Page | Content |
|---|---|
| Overview | observation setup, **general comments**, antennas (subbands, data health), sources, scans with the fringe SNR of every antenna |
| Diagnostics | three sub-tabs on the raw data: auto-correlations and cross-correlations of the diagnostic scans, and the **time evolution** of the whole observation |
| Calibration | the calibration chain, the **amplitude corrections** of the self-calibration, every calibration-table plot, calibrated calibrator data |
| Final results | per source: downloads, uv coverage, radplots, spectra, light curves, images |

**General comments** lists what the reduction found, and opens `summary.md`
as well: which scan the instrumental delay was solved on and which antennas
were solved there; per antenna, missing data or missing fringes with their UTC
range and scan numbers, a partial band, a dead polarization, heavy flagging,
amplitude corrections beyond 10% (and corrections the self-calibration could
not constrain); per source, the detection and image statistics; and the
warnings the pipeline raised during the run (kept in `.run_warnings.json`).
Before the pipeline has run, `summary.md` only says that the section is still
to come.

**Amplitude corrections** is one table per self-calibrated source: the factor
the difmapy gscale applied to each antenna in each subband (1.00 = unchanged),
green within 10%, orange within 30%, red beyond, with the median over the
subbands the antenna has a solution in as the last column.

**Time evolution** shows the raw amplitude and phase against time on every
baseline to a reference antenna, coloured by source, vector-averaged over the
central channels of one subband, with the two parallel hands side by side. A
second reference antenna (the one with the highest fringe SNR) is added when
the first did not observe the whole time or has stretches without data or
fringes, and a second subband when not every antenna recorded the first.

## The reduction notebook

```bash
vlbipy notebook -p rsm07          # write <work_dir>/rsm07.ipynb and print its path
vlbipy notebook -p rsm07 --open   # ... and launch JupyterLab on it
vlbipy notebook -p rsm07 --link   # link the plots instead of embedding them
```

The same notebook is written after the import and at the end of every `run`. It
needs no kernel to be read: each step that ran is a section with what it did,
its status and the plots it produced, followed by the calibrated data (uv
coverage, radplots, spectra, light curves, images). The plots are embedded, so
the notebook is self-contained (typically 10–30 MB) and can be moved or sent on
its own; it shows them as they were when it was written. `--link` links them by
relative path instead: a small notebook that has to stay next to `plots/`,
`images/` and `selfcal/`. A repeated step overwrites its plots, so a section
always shows the latest ones.

Executing cells is only needed to act on the data: every step has the call that
re-runs it (`FORCE = False` keeps *Run All* safe), and the last sections flag
calibration-table solutions and visibilities interactively and resume the
pipeline. This needs the `notebook` extra (`pip install "vlbipy[notebook]"`).

## Configuration file

Anything settable on the command line — and much that is not — can go in a TOML
file:

```bash
vlbipy run --config rsm07.toml
```

```toml
[global]
project = "rsm07"
observatory = "EVN"
reference_antenna = ["EF"]

[sources]
fringe_finders    = ["3C345"]
phase_calibrators = ["J1848+3219"]
targets           = ["3C395"]

[calibration.mbd]
solint = "30s"        # tighter than the default for a fast-varying atmosphere
minsnr = 4.0
```

Values resolve in three layers, later winning: built-in defaults → your TOML →
command-line flags. See [Configuration](../configuration.md) for the full list.
