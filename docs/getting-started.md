# Getting Started

This guide walks you through installing vlbipy and running your first VLBI
data reduction.

!!! warning "Work in progress"

    The calibration chain runs end to end and produces calibrated, per-source
    data on the CASA backend. Imaging and self-calibration are not
    implemented yet. See [Status](usage/status.md) for the full inventory.

## Prerequisites

- **Python >= 3.12**
- A data reduction backend — **CASA** (`casatools`/`casatasks`) is the only
  one implemented today. AIPS (`parseltongue`) exists as an interface but
  every method is a stub.

## Installation

### With the CASA backend (recommended)

```bash
pip install "vlbipy[casa]"
```

### With the dask-ms reader

```bash
pip install "vlbipy[daskms]"
```

Lazy, distributed-ready access to data already exported to a zarr store —
a-priori calibration and import only, no full calibration chain.

### Everything

```bash
pip install "vlbipy[all]"
```

### Development

```bash
pip install "vlbipy[dev]"
```

Includes `pytest` and `ruff` for testing and linting.

## Your first reduction

Everything hangs off one object, `VLBIObs`, driven the same way whether you
start it from the command line or from Python.

### 1. From the command line

```bash
vlbipy pipeline -p rsm07 -n EVN \
    -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345
```

This locates or downloads the raw data, imports it, runs the full
calibration chain, and exports calibrated per-source data. Progress is
recorded in `<work_dir>/.pipeline_state.json`, so a repeated run resumes
rather than redoing finished steps:

```bash
# Re-run the bandpass step and everything after it
vlbipy pipeline -p rsm07 ... --from-step bandpass

# Forget all progress and start over
vlbipy pipeline -p rsm07 ... --scratch
```

Or run stages one at a time — the normal way to drive a reduction
interactively (see [Command line](usage/cli.md)):

```bash
vlbipy import -p rsm07 -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345
vlbipy calibrate -p rsm07 ...
vlbipy flag -p rsm07 ...
vlbipy export -p rsm07 ...
```

### 2. Using a TOML config file

Anything settable on the command line — and much that is not — can go in a
TOML file instead:

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

```bash
vlbipy pipeline -p rsm07 --config rsm07.toml
```

Values resolve in three layers, later winning: built-in defaults → your TOML
→ command-line flags. See [Configuration](configuration.md) for the full
reference.

### 3. From Python

```python
from vlbipy import VLBIObs

obs = VLBIObs("rsm07", network="EVN",
              target="3C395", phasecal="J1848+3219", fringe_finder="3C345")
obs.run()
print(obs.summary())
```

Or step by step, inspecting each result as you go:

```python
obs.import_data()
obs.calibrate.a_priori()
obs.flag.apriori()
obs.calibrate.instrumental()
obs.flag.edges()
obs.calibrate.apply(force=True)
obs.calibrate.fringefit()
obs.calibrate.apply(force=True)
obs.export.per_source()
```

See [Interactive Python](usage/python.md) for the full walkthrough, including
inspecting metadata before calibrating and reading the visibilities directly.

## Next Steps

- [Full pipeline](usage/pipeline.md) — what `run()` does, in order
- [Command line](usage/cli.md) — driving a reduction stage by stage
- [Interactive Python](usage/python.md) — the object model, and where files are written
- [Configuration](configuration.md) — full reference for the TOML configuration file
- [Observatories](observatories/index.md) — observatory-specific details
- [Backends](backends/index.md) — backend-specific details
- [Status](usage/status.md) — what works, what's partial, what's missing
