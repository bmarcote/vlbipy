# vlbipy

Backend-agnostic VLBI data calibration and imaging pipeline. **Under development and far from being usable**

Aims to support **EVN**, **VLBA**, and **LBA** observations with different backends like **CASA** or **AIPS** (via ParselTongue), or others, selectable at runtime. Starting with just CASA and external MS-related tools.

## Installation

```bash
# Core (no backend)
pip install .

# With CASA backend
pip install ".[casa]"

# With AIPS backend
pip install ".[aips]"

# Both backends
pip install ".[all]"
```

## Usage

### CLI

```bash
# Full pipeline from TOML input file
vlbipy -i my_project.toml

# CLI parameters (override input file)
vlbipy -p EG078B -n EVN --backend CASA -t J1234+5678 -pcal J1230+5600 -ff 3C345

# Summary only
vlbipy -p EG078B --summary-only

# Run specific steps
vlbipy -i my_project.toml --start-step 5 --end-step 11
```

### Python API

```python
from vlbipy import Project
from vlbipy import pipeline

project = Project(project_code="EG078B", observatory="EVN",
    backend="CASA", input_file="my_project.toml")

# Full pipeline
pipeline.run_pipeline(project)

# Or step-by-step
pipeline.step_setup(project)
pipeline.step_load_metadata(project)
print(project.summary())
```

## Configuration

Configuration uses a three-layer cascade (later overrides earlier):

1. **defaults.toml** — built-in defaults
2. **User input file** — project-specific TOML file
3. **CLI arguments** — command-line overrides

See `src/vlbipy/templates/example_input.toml` for a complete example.

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Setup project directories |
| 2 | Download / find data files |
| 3 | Observatory-specific preparation (ANTAB append, etc.) |
| 4 | Import into MS (CASA) or UVDATA (AIPS) |
| 5 | Load metadata (sources, antennas, scans, frequency setup) |
| 6 | Initial flagging (a-priori flags, autocorr, edge channels, quack) |
| 7 | Initial diagnostic plots |
| 8 | A-priori calibration (Tsys, gain curve) |
| 9 | Instrumental delay calibration (SBD) |
| 10 | Bandpass calibration |
| 11 | Global fringe fitting (MBD) |
| 12 | Post-calibration flagging |
| 13 | Apply calibration and split sources |
| 14 | Dirty images |
| 15 | Final imaging with multiple weightings |

## Architecture

```
vlbipy/
├── models.py          # Data model (Source, Antenna, FreqSetup, Scan, etc.)
├── project.py         # Project orchestrator
├── pipeline.py        # Pipeline step functions
├── config.py          # TOML config loading
├── cli.py             # CLI entry point
├── backends/
│   ├── base.py        # Abstract backend interfaces
│   ├── casa/          # CASA backend (casatools/casatasks)
│   └── aips/          # AIPS backend (ParselTongue)
├── observatories/
│   ├── evn.py         # EVN: download, ANTAB, uvflg
│   ├── vlba.py        # VLBA specifics
│   └── lba.py         # LBA specifics
├── diagnostics.py     # Summary reports, data checks
├── plotting.py        # Diagnostic plots
└── templates/         # Default and example TOML configs
```

## License

MIT
