# CASA Backend

## Overview

**CASA** (Common Astronomy Software Applications) is the primary data reduction package for modern radio interferometry, developed by NRAO, ESO, NAOJ, and CSIRO[^1]. It provides a comprehensive suite of tools for calibration, imaging, and analysis of radio interferometric data from instruments including the VLA, ALMA, and VLBI arrays.

vlbipy uses CASA through its modular Python packages **`casatools`** (low-level C++ tools with Python bindings) and **`casatasks`** (high-level task wrappers). This allows vlbipy to run without a full CASA installation — only the pip-installable packages are required.

[^1]: CASA Team, et al. (2022). "CASA, the Common Astronomy Software Applications for Radio Astronomy." *PASP*, 134, 114501. [doi:10.1088/1538-3873/ac9642](https://doi.org/10.1088/1538-3873/ac9642)

## Installation

```bash
pip install "vlbipy[casa]"
```

This installs `casatools` and `casatasks` alongside vlbipy. Requires Python >= 3.12.

## Data Format

CASA uses the **MeasurementSet (MS)** format, a directory-based table structure built on the `casacore` table system. Key characteristics:

- MS files are directories containing binary tables
- Data columns: `DATA` (raw), `CORRECTED_DATA` (calibrated), `MODEL_DATA` (model visibilities)
- Calibration tables are stored as separate MS-like directories
- Native support for FITS-IDI import via `importfitsidi`

## CASA Backend Components

### CasaDataBackend

Reads observation metadata from MeasurementSets using `casatools.table` and `casatools.msmetadata`:

- **`get_metadata()`**: Extracts sources, antennas, scans, and frequency setup
- **`split()`**: Wraps `casatasks.split` to extract single-source datasets
- **`export_uvfits()`**: Wraps `casatasks.exportuvfits`

### CasaImportBackend

- **`import_fitsidi()`**: Wraps `casatasks.importfitsidi` with optional Tsys and gain curve appending
- **`import_uvfits()`**: Wraps `casatasks.importuvfits`

### CasaCalibrationBackend

- **`fringefit()`**: Wraps `casatasks.fringefit` for delay and rate calibration
- **`bandpass()`**: Wraps `casatasks.bandpass`
- **`gaincal()`**: Wraps `casatasks.gaincal` for amplitude/phase self-calibration
- **`applycal()`**: Wraps `casatasks.applycal` using the `GainTableLibrary` to manage table chains
- **`a_priori_cal()`**: Generates Tsys and gain curve calibration tables

### CasaFlaggingBackend

- **`apply_flags_from_file()`**: Wraps `casatasks.flagdata(mode='list')`
- **`flag_autocorrelations()`**: Flags autocorrelation data
- **`flag_edge_channels()`**: Flags spectral window edges
- **`quack()`**: Flags scan beginnings
- **`tfcrop()`**: Time-frequency automatic flagger
- **`aoflagger()`**: Calls external AOFlagger binary

### CasaImagingBackend

- **`tclean()`**: Wraps `casatasks.tclean` for CLEAN deconvolution
- **`wsclean()`**: Calls external WSClean binary as a subprocess

## Gain Table Management

CASA's `applycal` requires parallel lists of `gaintable`, `gainfield`, `interp`, and `spwmap` parameters. vlbipy's `GainTableLibrary` manages this bookkeeping automatically:

```python
# The library accumulates tables as calibration progresses
project.gaintables.add("EG078B.sbd", cal_type="sbd", interp="linear")
project.gaintables.add("EG078B.bp", cal_type="bp", interp="linear,linear")

# When applying, it generates the correct parameter lists
project.calibrate.applycal(project.msfile, field="", gaintables=project.gaintables)
```

## References

- CASA documentation: [https://casa.nrao.edu/](https://casa.nrao.edu/)
- casatools PyPI: [https://pypi.org/project/casatools/](https://pypi.org/project/casatools/)
- casatasks PyPI: [https://pypi.org/project/casatasks/](https://pypi.org/project/casatasks/)
- CASA Team, et al. (2022). "CASA, the Common Astronomy Software Applications for Radio Astronomy." *PASP*, 134, 114501. [doi:10.1088/1538-3873/ac9642](https://doi.org/10.1088/1538-3873/ac9642)
