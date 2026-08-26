# Pipeline Workflow

The vlbipy pipeline consists of 15 discrete steps that take raw VLBI data from an observatory archive through to calibrated, imaged radio maps. Each step is implemented as a standalone function in `vlbipy.pipeline` that takes a `Project` object and modifies it in place.

## Pipeline Overview

```text
Raw data → Import → Metadata → Flagging → A-priori cal → Plots → SBD/BP/SBD → MBD Fringe fit → Apply & Split → Imaging → Selfcal → Final images
```

| Step | Function | Description |
| ---: | --- | --- |
| 1 | `step_setup` | Create project directory structure |
| 2 | `step_find_data` | Download or locate raw data files |
| 3 | `step_prepare` | Observatory-specific preparation (e.g. ANTAB append) |
| 4 | `step_import_data` | Import FITS-IDI into MS (CASA) or UVDATA (AIPS) |
| 5 | `step_load_metadata` | Read metadata: sources, antennas, scans, frequency setup |
| 6 | `step_initial_flagging` | A-priori flags, autocorrelations, edge channels, quack |
| 7 | `step_a_priori_calibration` | System temperature (Tsys) and gain curve calibration |
| 8 | `step_initial_plots` | Antenna participation and autocorrelation diagnostic plots |
| 9 | `step_instrumental_corrections` | SBD → Bandpass → refined SBD (combined step) |
| 10 | `step_global_fringefit` | Multi-band delay (MBD) fringe fitting on all calibrators |
| 11 | `step_post_cal_flagging` | Post-calibration flagging (tfcrop on corrected data) |
| 12 | `step_apply_and_split` | Apply all calibration and split individual sources |
| 13 | `step_initial_imaging` | Initial (dirty) images of all split sources |
| 14 | `step_selfcal` | Self-calibrate phase calibrator source(s) |
| 15 | `step_final_imaging` | Final imaging with multiple Briggs robust weightings |

## Running the Pipeline

### Full pipeline

```python
from vlbipy import Project
from vlbipy.pipeline import run_pipeline

project = Project(project_code="EG078B", observatory="EVN", backend="CASA", input_file="my_project.toml")
run_pipeline(project)
```

### Partial pipeline

```python
run_pipeline(project, start_step=5, end_step=11)
```

### Smart resume

By default (`smart=True`), the pipeline detects whether any calibration tables have been modified since the last run (e.g. by manual flagging) and automatically re-runs all downstream steps. To disable this and always re-run all steps in range:

```python
run_pipeline(project, smart=False)
```

### Step-by-step

```python
from vlbipy import pipeline

pipeline.step_setup(project)
data_files = pipeline.step_find_data(project)
data_files = pipeline.step_prepare(project, data_files)
pipeline.step_import_data(project, data_files)
pipeline.step_load_metadata(project)
pipeline.step_initial_flagging(project)
pipeline.step_a_priori_calibration(project)
pipeline.step_initial_plots(project)
pipeline.step_instrumental_corrections(project)
pipeline.step_global_fringefit(project)
pipeline.step_post_cal_flagging(project)
splits = pipeline.step_apply_and_split(project)
pipeline.step_initial_imaging(project, splits)
selfcal_tables = pipeline.step_selfcal(project, splits)
pipeline.step_final_imaging(project, splits, selfcal_tables)
```

---

## Step Details

### Step 1: Setup

Creates the standard project directory layout:

```text
<working_dir>/
├── input_data/           # Raw FITS-IDI files and ancillary files
├── calibration_tables/   # Calibration tables (.tsys, .gc, .sbd, .bpass, .mbd)
├── plots/
│   ├── pre/              # Diagnostic plots on uncalibrated data
│   └── calibrated/       # Diagnostic plots on calibrated data
├── calibrated_data/      # Split measurement sets per source
├── images/
│   └── initial/          # Initial images before selfcal
└── logs/                 # Pipeline log files
```

The `summary.md` file is written to the working directory after step 5.

### Step 2: Find Data

Searches `input_data/`, then the current working directory (moving any found files into `input_data/`). Falls back to automatic download for observatories that support it:

- **EVN**: Automatically downloads FITS-IDI, ANTAB, and `.uvflg` files from the [JIVE archive](http://archive.jive.nl/) given a project code and `obsdate`.
- **VLBA**: Download is not automated — data must be retrieved manually from the [NRAO archive](https://data.nrao.edu/portal/).
- **LBA**: Experimental automated download from [ATOA](https://atoa.atnf.csiro.au/).

### Step 3: Observatory-Specific Preparation

Pre-import processing that varies by observatory:

- **EVN**: Appends Tsys and gain curve data from the ANTAB file into the FITS-IDI file headers using `casavlbitools`.
- **VLBA**: Pass-through — calibration metadata is embedded in the FITS-IDI headers by the DiFX correlator.
- **LBA**: Similar to EVN; ANTAB appending is performed if an ANTAB file is found.

### Step 4: Import

Converts raw data files into the backend's native format:

- **CASA**: Imports FITS-IDI into a MeasurementSet (`{project}.ms`) using `importfitsidi`.
- **AIPS**: Loads FITS-IDI into a UVDATA file using `FITLD`.

If the MS already exists, this step is skipped.

### Step 5: Load Metadata

Reads the imported data to extract:

- **Source catalog**: Names, coordinates, and roles (target, calibrator, etc.)
- **Antenna table**: Station names, positions, diameters
- **Scan list**: Scan numbers, source associations, time ranges, participating antennas
- **Frequency setup**: Reference frequency, bandwidth, subbands, channels, polarizations

Writes a `summary.md` to the project directory. Also auto-selects the reference antenna based on observatory-specific priority lists (e.g. Effelsberg first for EVN, Pie Town first for VLBA).

### Step 6: Initial Flagging

Applies several layers of flagging to remove known bad data:

1. **A-priori flag file**: Observatory-provided flags (e.g. `.uvflg` for EVN).
2. **Autocorrelations**: Flagged as they carry no interferometric information.
3. **Edge channels**: A configurable fraction of channels at each spectral window edge are flagged to remove bandpass roll-off (default: 10% per edge).
4. **Quack**: The first N seconds of each scan are flagged while antennas settle on source.

A flag backup named `before_initial_flagging` is created before any flagging, so the pre-flagging state can be restored with:

```python
project.flag.flag_restore(project.msfile, "before_initial_flagging")
```

### Step 7: A-Priori Calibration

Generates amplitude calibration tables from metadata appended during step 3:

- **Tsys table** (`{project}.tsys`): Converts correlation coefficients to flux density (Jy) using measured system temperatures.
- **Gain curve table** (`{project}.gc`): Compensates for elevation-dependent antenna gain variations.

Both tables are added to `project.gaintables`.

### Step 8: Initial Diagnostic Plots

Generates diagnostic plots saved to `plots/pre/`:

- **tplot**: A time-vs-antenna participation grid showing which antennas were present in each scan, color-coded by source.
- **Autocorrelation spectra**: Amplitude vs. frequency for fringe finder sources, verifying bandpass shapes.
- **Cross-correlation spectra**: Amplitude vs. frequency on short baselines for fringe finders.

### Step 9: Instrumental Corrections (SBD → Bandpass → SBD)

A combined three-pass calibration step using fringe finder source(s):

1. **SBD pass 1** (`{project}.sbd`): Fringe fitting with `zerorates=True` on the fringe finder scan with the most antenna participation. Measures single-band delays — the constant instrumental delay offset per antenna.
2. **Bandpass** (`{project}.bpass`): Solves for the frequency-dependent complex gain (amplitude and phase) of each antenna. Typically uses `solint='inf'` and `combine='scan'`.
3. **SBD pass 2** (`{project}.sbd2`): A refined SBD solution computed after bandpass correction, to remove any residual delay not captured in pass 1.

The gain curve table (`gc`) is excluded from the fringe-fit pre-apply if it has missing antennas (to avoid GSL solver crashes).

### Step 10: Global Fringe Fitting (MBD)

The core VLBI calibration step. Solves simultaneously for **residual delays**, **delay rates**, and **phases** across all baselines for fringe finder and phase calibrator sources:

- Produces the MBD solution table (`{project}.mbd`)
- Uses `zerorates=False` to solve for fringe rates
- Solutions for the phase calibrator are interpolated and transferred to the target in step 12

Optionally solves for ionospheric dispersive delay at low frequencies (≤5 GHz) when enabled in config.

### Step 11: Post-Calibration Flagging

After applying all calibration solutions, runs automated flagging on calibrators:

- **tfcrop**: Time-frequency crop on corrected data, identifying outliers in calibrated visibility amplitudes.

### Step 12: Apply Calibration and Split

Applies the full calibration table chain to all data, then splits individual sources:

- All source types (targets, check sources, phase calibrators, fringe finders) are split
- Each source is written to `calibrated_data/{project}_{source}.ms`
- UVFITS files are exported alongside each split MS (configurable via `[export] export_uvfits`)

### Step 13: Initial Imaging

Creates zero-iteration ("dirty") images of all split sources with natural weighting, saved to `images/initial/`. These verify that the calibration has produced coherent fringes before self-calibration.

### Step 14: Self-Calibration

Runs iterative self-calibration on the phase calibrator source(s):

1. **Phase-only rounds** (default: 4): Solve for phase corrections with decreasing solution intervals.
2. **Amplitude+phase rounds** (default: 5): Solve for amplitude and phase after the phase-only rounds converge.

Self-calibration solutions from the phase calibrator are transferred to targets in step 15. Can be disabled with `[selfcal] enabled = false`.

### Step 15: Final Imaging

Images all sources (targets and check sources) with multiple Briggs robust weighting values:

- Default robust values: `[-2, -1, 0, 1, 2]`
- Applies selfcal solutions from step 14 to targets before imaging
- Images are saved to `images/{project}_{source}_r{robust}.image`
- FITS and PNG exports are produced when enabled in config

## Smart Pipeline Mode

The pipeline tracks calibration table file timestamps in a `pipeline_state.json` file in the project directory. When a table is modified (e.g. after manual flagging or re-calibration), all downstream steps are automatically invalidated and re-run. This means you can:

1. Run the full pipeline
2. Inspect the results and manually edit a calibration table
3. Re-run with `run_pipeline(project)` — only the steps after the modified table are re-executed

## Self-Calibration (Module)

Self-calibration can also be invoked directly via `vlbipy.selfcal.selfcal_loop` for custom workflows outside the main pipeline. See the [Self-Calibration API reference](api/selfcal.md) for details.
