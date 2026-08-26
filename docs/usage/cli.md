# Command line

The CLI runs one stage at a time, so you can inspect the products before going
on. Every subcommand takes the same project and source arguments.

```bash
vlbipy <command> -p <project> [-t TARGET] [--phasecal NAME] [--fringe-finder NAME]
```

| Command | Purpose |
|---|---|
| `run` | the whole pipeline (see [Full pipeline](pipeline.md)) |
| `import` | download / import raw data and read metadata |
| `calibrate` | run calibration steps, or one named step |
| `flag` | run flagging modes |
| `plot` | produce diagnostic plots |
| `export` | split per source and export |

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

## Configuration file

Anything settable on the command line — and much that is not — can go in a TOML
file:

```bash
vlbipy run -c rsm07.toml
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
