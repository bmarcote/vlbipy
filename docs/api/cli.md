# CLI

Command-line interface for vlbipy: one subcommand per pipeline stage, each a
thin translation of flags into `VLBIObs` namespace calls. Supports a TOML
config file, CLI parameter overrides, or a combination.

## Usage

```bash
# Full pipeline (config file, CLI parameters, or both)
vlbipy pipeline -p rsm07 -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345
vlbipy pipeline -p rsm07 --config my_project.toml

# vlbipy -p ... (no subcommand) is shorthand for `vlbipy pipeline -p ...`
vlbipy -p rsm07 -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345

# One stage at a time
vlbipy import -p rsm07 -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345
vlbipy calibrate bandpass fringefit -p rsm07 ...   # selected steps, in order given
vlbipy flag aoflagger -p rsm07 ...
vlbipy plot caltables uv_coverage -p rsm07 ...
vlbipy export -p rsm07 ...

# Summary only
vlbipy summary -p rsm07

# Control what re-runs
vlbipy pipeline -p rsm07 --from-step bandpass   # redo that step and everything after
vlbipy pipeline -p rsm07 --scratch              # forget all progress and start over

# Skip the dispersive (ionospheric) delay in the fringe fit
vlbipy pipeline -p rsm07 -t 3C395 --no-ionos
```

See [CLI usage](../usage/cli.md) for the full walkthrough.

## API Reference

::: vlbipy.cli
