# CLI

Command-line interface for vlbipy. Entry point for running the VLBI pipeline from the command line. Supports TOML input file, CLI parameter overrides, or a combination.

## Usage

```bash
# Full pipeline from a TOML config file
vlbipy run --config my_project.toml

# CLI parameters (they override the config file)
vlbipy run -p rsm07 -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345

# Summary only
vlbipy run -p rsm07 --summary-only

# Control what re-runs
vlbipy run -p rsm07 --from-step bandpass   # redo that step and everything after
vlbipy run -p rsm07 --scratch              # forget all progress and start over

# Skip the dispersive (ionospheric) delay in the fringe fit
vlbipy run -p rsm07 -t 3C395 --no-ionos
```

See [CLI usage](../usage/cli.md) for the full walkthrough.

## API Reference

::: vlbipy.cli
