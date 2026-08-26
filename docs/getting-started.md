# Getting Started

This guide walks you through installing vlbipy and running your first VLBI data reduction pipeline.

## Prerequisites

- **Python >= 3.12**
- At least one data reduction backend:
  - **CASA**: `casatools` and `casatasks` (recommended for most users)
  - **AIPS**: `parseltongue` (for users with existing AIPS workflows)

## Installation

### Core package (no backend)

```bash
pip install vlbipy
```

This installs the core package without any backend. Useful for inspecting data metadata, generating summaries, or developing custom workflows.

### With CASA backend

```bash
pip install "vlbipy[casa]"
```

Installs `casatools` and `casatasks` alongside vlbipy. This is the recommended setup for most users.

### With AIPS backend

```bash
pip install "vlbipy[aips]"
```

Installs `parseltongue` for interfacing with AIPS. Requires a working AIPS installation on the system.

### Both backends

```bash
pip install "vlbipy[all]"
```

### Development

```bash
pip install "vlbipy[dev]"
```

Includes `pytest` and `ruff` for testing and linting.

## Your First Pipeline Run

### 1. Create a TOML input file

Create a file called `my_project.toml`:

```toml
[global]
project_name = "EG078B"
observatory = "EVN"
backend = "CASA"
reference_antenna = "EF"
obsdate = "231015"

[sources]
target = ["J1234+5678"]
phasecal = ["J1230+5600"]
fringefinder = ["3C345", "4C39.25"]

[flagging]
edge_channels_fraction = 0.05
outlier_sigma = 5.0

[calibration]
ionos = true                 # solve the dispersive delay below 6 GHz

[calibration.sbd]
solint = "inf"

[calibration.bandpass]
solint = "inf"
combine = "scan"

[calibration.mbd]
solint = "inf"

[imaging]
niter = 500
imsize = [512, 512]
cell = "0.5mas"
```

### 2. Run the pipeline

```bash
vlbipy run --config my_project.toml
```

This will execute all 15 pipeline steps: from setting up directories, through calibration, to final imaging.

### 3. Control what re-runs

Progress is recorded in `<work_dir>/.pipeline_state.json`, so a repeated run
picks up where the last one stopped. To redo part of it:

```bash
# Re-run the bandpass and every step after it
vlbipy run --config my_project.toml --from-step bandpass
```

```bash
# Forget all progress and start over
vlbipy run --config my_project.toml --scratch
```

### 4. View a summary

```bash
vlbipy run --config my_project.toml --summary-only
```

## Using the Python API

For interactive or scripted use:

```python
from vlbipy import Project
from vlbipy.pipeline import run_pipeline

# Create the project
project = Project(
    project_code="EG078B",
    observatory="EVN",
    backend="CASA",
    input_file="my_project.toml",
)

# Full pipeline
run_pipeline(project)

# Or step-by-step
from vlbipy import pipeline

pipeline.step_setup(project)
data_files = pipeline.step_find_data(project)
data_files = pipeline.step_prepare(project, data_files)
pipeline.step_import_data(project, data_files)
pipeline.step_load_metadata(project)
print(project.summary())
```

## Next Steps

- [Pipeline Workflow](pipeline.md) — detailed description of each pipeline step
- [Configuration](configuration.md) — full reference for the TOML configuration file
- [Observatories](observatories/index.md) — observatory-specific details
- [Backends](backends/index.md) — backend-specific details
