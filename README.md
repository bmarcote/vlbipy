# vlbipy

Backend-agnostic VLBI data calibration and imaging pipeline.

**Work in progress.** The calibration chain runs end to end and produces
calibrated, per-source data on the CASA backend. Imaging and self-calibration
are not implemented yet. See [Status](docs/usage/status.md) for the honest
inventory of what works.

Aims to support **EVN**, **VLBA**, and **LBA** observations through different
backends — **CASA**, **AIPS** (via ParselTongue), or a lazy **dask-ms**
reader — selectable at runtime. Only CASA is implemented today.

## Installation

Requires Python >= 3.12.

```bash
# Core, no backend (inspect metadata, build summaries, develop against the dummy backend)
pip install .

# With the CASA backend (recommended — the only backend implemented so far)
pip install ".[casa]"

# With the dask-ms reader (lazy, distributed-ready access to exported data)
pip install ".[daskms]"

# Everything (CASA + AIPS + dask-ms)
pip install ".[all]"
```

## Usage

vlbipy exposes the same pipeline three ways — CLI, a full-pipeline call, or
step-by-step from Python — all driving the same `VLBIObs` object, so a run
started on the command line can be inspected and continued from Python.

### CLI

```bash
# Import (download if needed) + calibrate + flag + export, end to end
vlbipy pipeline -p rsm07 -n EVN -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345

# Or run individual stages
vlbipy import -p rsm07 -t 3C395 --phasecal J1848+3219 --fringe-finder 3C345
vlbipy calibrate a_priori instrumental fringefit -p rsm07 ...
vlbipy flag apriori quack -p rsm07 ...
vlbipy plot corner spectrum -p rsm07 ...
vlbipy export -p rsm07 ...

# Just the summary
vlbipy summary -p rsm07

# Resume, or redo from a step
vlbipy pipeline -p rsm07 ... --from-step bandpass
vlbipy pipeline -p rsm07 ... --scratch
```

See `vlbipy --help` for every subcommand and option (`import`, `calibrate`,
`flag`, `image`, `plot`, `summary`, `export`, `pipeline`).

### Python API

The full pipeline, run from Python:

```python
from vlbipy import VLBIObs

obs = VLBIObs("rsm07", network="EVN", work_dir="rsm07",
              target="3C395", phasecal="J1848+3219", fringe_finder="3C345")
obs.run()
```

Or drive it interactively, inspecting each result before deciding the next
step — the mode most useful while working out what a dataset needs:

```python
from vlbipy import VLBIObs

obs = VLBIObs("rsm07", network="EVN", work_dir="rsm07",
              target="3C395", phasecal="J1848+3219", fringe_finder="3C345")

# Locate or download the raw data (FITS-IDI + .antab + .uvflg for EVN), import, read metadata
obs.import_data()
print(obs.summary())

# A-priori amplitude calibration, then flag the observatory .uvflg + autocorrelations
obs.calibrate.a_priori()
obs.flag.apriori()

# Instrumental calibration: antenna/scan selection, then SBD -> bandpass -> SBD
antennas, scans = obs.calibrate.select_calibration_data()
obs.calibrate.instrumental()
obs.calibrate.edge_channels()
obs.calibrate.apply(force=True)

# Global fringe fit, then apply the full chain
obs.calibrate.fringefit()
obs.calibrate.apply(force=True)

# Look at what came out
obs.plot.spectrum(field="3C345", scans=scans)
obs.plot.caltables()
[t.cal_type for t in obs.gaintables]   # ['tsys', 'gc', 'bpass', 'sbd2', 'mbd']

# Finish: outlier flagging, per-source export
obs.flag.outliers()
obs.export.per_source()
```

Namespaces are discoverable — calling one runs its sensible default, and it
lists its own methods so a typo never turns into a silent `NotImplementedError`:

```python
obs.calibrate            # <CalibrateNamespace operations: a_priori, apply, ...>
obs["rsm07"]._backend.capabilities()
```

Full walkthrough, including the object model and where files are written:
[Interactive Python](docs/usage/python.md).

## Configuration

Configuration uses a three-layer cascade (later overrides earlier):

1. **`defaults.toml`** — built-in defaults (`src/vlbipy/templates/defaults.toml`)
2. **User input file** — project-specific TOML file, or a `dict` passed to `VLBIObs(config=...)`
3. **Constructor/CLI arguments** — explicit overrides

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
solint = "30s"
minsnr = 4.0
```

See [Configuration](docs/configuration.md) for the full reference.

## Pipeline steps

`VLBIObs.run()` chains these namespace operations, in order (each is also
callable on its own, and resumable via a `.pipeline_state.json` in the
working directory):

| Namespace call | What happens |
|---|---|
| `import_data` | find or download FITS-IDI + `.antab`/`.uvflg`; import to a Multi-MS; read metadata |
| `calibrate.a_priori` | Tsys + gain curve calibration, de-spiked |
| `flag.apriori` | observatory `.uvflg` flags + autocorrelations |
| `calibrate.instrumental` | antenna/scan selection, then SBD -> bandpass -> SBD |
| `calibrate.edge_channels` | measure and flag the subband roll-off |
| `calibrate.fringefit` | global (multi-band delay) fringe fit |
| `flag.quack` | measured per-antenna slew-time trim |
| `flag.outliers` | per-baseline robust outlier flagging |
| `calibrate.second_pass` | re-solve the full chain on the now-flagged data |
| `calibrate.scalar_bandpass` | one amplitude per antenna/subband, levelling the subbands |
| plots | corner, spectrum, time series, radplot, all calibration tables |
| `export.per_source` | split per source; UVFITS export |
| imaging | *not implemented yet* (skipped with a warning) |

See [Pipeline Workflow](docs/pipeline.md) for the detail behind each step.

## Architecture

```
src/vlbipy/
├── vlbiobs.py          # VLBIObs — the public entry point (one project, or a campaign)
├── observation.py       # Observation — per-project state, backend, namespaces
├── namespaces/          # import_data, calibrate, flag, plot, clean, selfcal, export
├── models.py             # Data model (Antenna, Source, FreqSetup, Scan, CalTable, ...)
├── config.py             # Three-layer TOML config loading
├── cli.py                 # CLI entry point (one subcommand per stage)
├── sources.py             # SourceSet: roles, phase referencing
├── state.py               # Persisted step state (resume / --from-step / --scratch)
├── plotting.py            # Diagnostic plots
├── diagnostics.py         # Scan/quality statistics, summaries
├── fitsidi.py              # FITS-IDI discovery and inspection
├── backends/
│   ├── base.py             # Abstract interfaces (DataOps, CalibrationOps, FlagOps, PlotOps, ImagingOps, ExportOps)
│   ├── casa.py              # CASA backend (casatools/casatasks) — implemented
│   ├── dask_ms.py           # dask-ms backend — a-priori/import only
│   ├── aips.py               # AIPS backend (ParselTongue) — stub
│   └── dummy.py               # In-memory synthetic backend, no external dependency
├── observatories/
│   ├── evn.py               # EVN: archive download, ANTAB append, .uvflg
│   ├── vlba.py                # VLBA specifics
│   └── lba.py                  # LBA specifics
└── templates/               # defaults.toml
```

## Documentation

Full docs (getting started, usage, configuration, observatories, backends,
API reference) are built with [Zensical](https://zensical.org/) from `docs/`:

```bash
pip install zensical "mkdocstrings[python]"
zensical build     # -> site/
zensical serve      # live preview
```

## License

MIT
