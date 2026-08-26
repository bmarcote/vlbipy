# AIPS Backend

## Overview

**AIPS** (Astronomical Image Processing System) is the classic data reduction package for radio interferometry, developed and maintained by NRAO since the 1980s[^1]. It has been the standard tool for VLBI data reduction for decades and remains widely used, particularly for specialized VLBI calibration tasks.

vlbipy interfaces with AIPS through **ParselTongue**, a Python scripting interface that provides programmatic access to AIPS tasks and data structures[^2]. This allows vlbipy to drive AIPS calibration and imaging from Python without manual interaction with the AIPS command-line interface.

[^1]: Greisen, E. W. (2003). "AIPS, the VLA, and the VLBA." In *Information Handling in Astronomy — Historical Vistas*, Astrophysics and Space Science Library, Vol. 285, p. 109. [doi:10.1007/0-306-48080-8_7](https://doi.org/10.1007/0-306-48080-8_7)

[^2]: Kettenis, M., et al. (2006). "ParselTongue: AIPS Talking Python." In *Astronomical Data Analysis Software and Systems XV*, ASP Conference Series, Vol. 351, p. 497.

## Installation

```bash
pip install "vlbipy[aips]"
```

This installs `parseltongue`. A working AIPS installation must be available on the system, with the `AIPS_ROOT` environment variable set.

## Data Format

AIPS uses a **catalog-based** data system:

- Visibility data is stored as **UVDATA** entries in the AIPS catalog
- Calibration is applied through **CL (calibration)** and **SN (solution)** tables attached to the UVDATA
- Bandpass corrections are stored in **BP** tables
- Flag information is stored in **FG** tables
- The AIPS catalog is indexed by user number and disk number

## AIPS Backend Components

### AipsDataBackend

Reads observation metadata from UVDATA catalog entries using ParselTongue's `AIPSUVData` interface:

- **`get_metadata()`**: Reads the AN (antenna), SU (source), and NX (index) tables
- **`split()`**: Wraps the AIPS `SPLIT` task
- **`export_uvfits()`**: Wraps the AIPS `FITTP` task

### AipsImportBackend

- **`import_fitsidi()`**: Wraps the AIPS `FITLD` task to load FITS-IDI data
- **`import_uvfits()`**: Wraps `FITLD` for UVFITS format

### AipsCalibrationBackend

- **`fringefit()`**: Wraps the AIPS `FRING` task for delay and rate calibration
- **`bandpass()`**: Wraps the AIPS `BPASS` task
- **`gaincal()`**: Wraps the AIPS `CALIB` task
- **`applycal()`**: Manages CL table application via `CLCAL`
- **`a_priori_cal()`**: Wraps `ANTAB` and `APCAL` for Tsys and gain curve calibration

### AipsFlaggingBackend

- **`apply_flags_from_file()`**: Wraps the AIPS `UVFLG` task
- **`flag_autocorrelations()`**: Uses `UVFLG` with `OPCODE='FLAG'` on autocorrelations
- **`flag_edge_channels()`**: Uses `UVFLG` to flag edge channels
- **`quack()`**: Wraps the AIPS `QUACK` task
- **`tfcrop()`**: Falls back to `RFLAG` or logs a warning (AIPS has limited auto-flagging)
- **`aoflagger()`**: Calls external AOFlagger binary, requires UVFITS export/reimport

### AipsImagingBackend

- **`tclean()`**: Wraps the AIPS `IMAGR` task for CLEAN deconvolution
- **`wsclean()`**: Exports to UVFITS, calls WSClean, and reimports the result

## AIPS vs CASA Differences

| Aspect | CASA | AIPS |
| --- | --- | --- |
| **Data format** | MeasurementSet (directory) | AIPS catalog entry |
| **Calibration model** | Separate table files applied via `applycal` | CL/SN table chain attached to UVDATA |
| **Fringe fitting** | `fringefit` task | `FRING` task |
| **Bandpass** | `bandpass` task | `BPASS` task |
| **Imaging** | `tclean` (multi-scale, wideband) | `IMAGR` (classic CLEAN) |
| **Scripting** | Native Python (`casatasks`) | ParselTongue Python bindings |
| **Parallelism** | MPI support in `tclean` | Limited |

## When to Use AIPS

- Legacy VLBI workflows and scripts
- Comparison with historical AIPS-based results
- Specialized AIPS tasks not yet available in CASA (e.g. some polarization calibration routines)
- Environments where AIPS is already installed and configured

## References

- AIPS homepage: [http://www.aips.nrao.edu/](http://www.aips.nrao.edu/)
- AIPS Cookbook: [http://www.aips.nrao.edu/cook.html](http://www.aips.nrao.edu/cook.html)
- ParselTongue documentation: [https://www.jive.eu/jivewiki/doku.php?id=parseltongue:parseltongue](https://www.jive.eu/jivewiki/doku.php?id=parseltongue:parseltongue)
- Greisen, E. W. (2003). "AIPS, the VLA, and the VLBA." *Astrophysics and Space Science Library*, Vol. 285, p. 109.
- Kettenis, M., et al. (2006). "ParselTongue: AIPS Talking Python." *ASP Conference Series*, Vol. 351, p. 497.
