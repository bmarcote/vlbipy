# Scientific Background

## Very Long Baseline Interferometry

**Very Long Baseline Interferometry (VLBI)** is a technique in radio astronomy that combines signals from radio telescopes separated by hundreds or thousands of kilometres to synthesize an instrument with an effective aperture equal to the maximum baseline length between antennas. This yields the highest angular resolution achievable in astronomy — routinely reaching sub-milliarcsecond scales at centimetre wavelengths.

Unlike connected-element interferometers (e.g. the VLA or ALMA), VLBI stations record data independently to local media using hydrogen maser frequency standards for time and phase coherence. The recorded data are shipped (or electronically transferred via high-speed networks in e-VLBI mode) to a central **correlator**, which cross-correlates the signals from all antenna pairs to produce **visibilities** — the fundamental measurement of an interferometer.

### Key references

- Thompson, A. R., Moran, J. M., & Swenson, G. W. (2017). *Interferometry and Synthesis in Radio Astronomy* (3rd ed.). Springer. [doi:10.1007/978-3-319-44431-4](https://doi.org/10.1007/978-3-319-44431-4)
- Walker, R. C. (1995). "Very Long Baseline Interferometry". In *Synthesis Imaging in Radio Astronomy II*, ASP Conference Series, Vol. 180, p. 433.

## The VLBI Calibration Problem

Raw VLBI visibilities are corrupted by several effects that must be removed before scientifically useful images can be produced:

1. **Instrumental delays**: Each station introduces a constant delay offset due to cable lengths, electronics, and clock offsets. These must be measured and removed (single-band delay, or SBD calibration).

2. **Atmospheric and ionospheric phase**: The troposphere and ionosphere introduce time-variable path-length changes. At frequencies below ~5 GHz, ionospheric dispersive delays become significant.

3. **Bandpass shape**: Each antenna's receiver and signal chain imprints a frequency-dependent gain (amplitude and phase) on the data. Bandpass calibration flattens these across the observing band.

4. **Fringe rates**: Earth rotation causes the geometric delay between antennas to change with time, producing a time-variable fringe rate. Global fringe fitting (multi-band delay, or MBD) solves for residual delays and rates across all baselines simultaneously.

5. **Amplitude calibration**: The system equivalent flux density (SEFD) of each antenna varies with elevation, weather, and receiver temperature. A-priori amplitude calibration uses system temperature (Tsys) measurements and antenna gain curves to convert correlation coefficients into physical flux density units (Jy).

6. **Radio frequency interference (RFI)**: Terrestrial signals can contaminate the data and must be identified and flagged.

The vlbipy pipeline addresses each of these effects through a well-defined sequence of calibration steps, following standard VLBI reduction practices as described in the [EVN Data Analysis Guide](https://www.evlbi.org/evn-data-analysis-guide) and the [AIPS Cookbook](http://www.aips.nrao.edu/cook.html).

## Source Types in VLBI Observations

A typical VLBI experiment observes several types of sources:

- **Target**: The science target — the source you want to image. Often weak and potentially resolved.
- **Phase calibrator**: A compact, nearby source observed frequently to track atmospheric phase variations. Used for phase-referencing to transfer calibration solutions to the target.
- **Fringe finder**: A strong, compact source (often a well-known quasar like 3C345, 3C273, or 4C39.25) used to determine instrumental delays and verify that all stations have fringes.
- **Polarization calibrator**: A source with known polarization properties, used to calibrate the instrumental polarization leakage (D-terms) and absolute polarization angle.
- **Check source**: An additional calibrator observed to verify the quality of the phase-referencing transfer, without being used in the calibration solutions.

## Self-Calibration

After the initial calibration, residual phase and amplitude errors may remain. **Self-calibration** is an iterative technique where the source's own structure (from a preliminary image) is used as a model to derive additional calibration corrections. This is performed in alternating imaging and calibration cycles:

1. **Phase-only self-calibration**: Corrects residual phase errors with progressively shorter solution intervals.
2. **Amplitude and phase self-calibration**: Also corrects residual amplitude errors, typically with longer solution intervals to avoid introducing noise.

Self-calibration is most effective for sources that are sufficiently bright and compact to produce a reliable initial model.

## The vlbipy Approach

vlbipy implements the standard VLBI calibration workflow as a reproducible, automated pipeline while preserving the ability to intervene at any step. The key design decisions are:

- **Observatory abstraction**: Each observatory (EVN, VLBA, LBA) has different data formats, archive systems, and auxiliary file conventions. vlbipy handles these differences in dedicated observatory handler classes, so the calibration logic remains observatory-independent.

- **Backend abstraction**: CASA and AIPS implement the same radio interferometric algorithms with different APIs and data formats. vlbipy defines abstract backend interfaces for data access, import, calibration, flagging, and imaging. Concrete implementations for CASA and AIPS plug into these interfaces.

- **Configuration cascade**: Pipeline parameters are specified through a three-layer system — built-in defaults, a user TOML configuration file, and command-line overrides — ensuring reproducibility while allowing quick parameter changes.

- **Gain table bookkeeping**: The `GainTableLibrary` tracks all calibration tables in order, automatically managing the `gaintable`, `gainfield`, `interp`, and `spwmap` parameters that CASA's `applycal` requires.
