# Status

vlbipy is **work in progress**. This page is the honest inventory: what works
on real data, what exists but is untested, and what is not written yet.

Everything below is validated against a 2.8 h EVN L-band dataset (rsm07, 14
antennas, 4 × 64 channels) on the CASA backend unless noted.

## Working

| Area | Notes |
|---|---|
| Import | FITS-IDI → Multi-MS; EVN archive download of data, `.antab` and `.uvflg` |
| Inspection | antennas, scans, frequency setup, subband participation, baseline geometry |
| Fringe SNR survey | per scan / antenna / polarization, with a scan × antenna matrix plot |
| Antenna & scan selection | full-band coverage + SNR; linked covering set when no single scan works |
| A-priori calibration | Tsys + gain curve, per-antenna plots; with the AI mode the Tsys is repaired series by series from the raw data (noisy hands, spurious subbands, absorption, adaptive smoothing) |
| AI mode | raw-data pre-diagnostic per antenna, Tsys checks against the raw amplitudes, distribution-based flagging of the calibrated data, rule base, numpy decision forest ([AI mode](ai.md)) |
| Flagging | `.uvflg`, autocorrelations, measured band edges, measured quack, per-baseline outliers |
| Instrumental calibration | SBD → bandpass → SBD, with solution-coverage verification |
| Global fringe fit | multi-band delay, with the spw map read back from the table |
| Second pass | full re-solve on the flagged data |
| Scalar bandpass | one amplitude per antenna and subband, both hands levelled together (corrects RCP/LCP offsets) |
| Apply | via a CASA cal library (`caltables.txt`); each solve declares its own priors in `callibs/` |
| Export | per-source measurement sets and Difmap-ready UVFITS |
| Diagnostics | spectrum, time series, per-baseline corner, radplot, all calibration tables |
| Global fringe fit | dispersive (ionospheric) delay solved below 8 GHz (`[calibration].ionos_max_ghz`); `--no-ionos` to disable |
| Imaging and self-calibration | difmapy (dask-ms backend): visibility model fitting first, phase self-cal, Bayesian amplitude gscale per station and subband, time-dependent amplitude self-cal, e-MERLIN flux scale from 3C286 |
| LBA (DiFX) | `<CODE>.FITS` + `.antab` + `.uvflg` found beside the working directory; Tsys / gain curves appended in place; ACCOR and EOP corrections; tested on V589 (4 epochs, 8.4 GHz, dask-ms backend) |
| Polarization-aware instrumental calibration | an antenna with one dead polarization, or one that appears only in other scans, keeps what it has |
| Campaigns | per-epoch calibration, joint phase-calibrator model, per-epoch refinement, combined images (`combine()`) |
| Bookkeeping | persistent step state and calibration chain, resume / `--from-step` / `--scratch`, run log |
| Dashboard & notebook | static HTML report (`vlbipy report`) and a Jupyter notebook (`vlbipy notebook`) showing every step with its plots; both rebuilt at the end of each run |

## Not implemented

- **Imaging and self-calibration on the CASA backend** — `image.clean` is not
  written for CASA (`run()` skips it with a warning); use the dask-ms backend,
  which images and self-calibrates with difmapy.
- **Spectral line, multi-phase-centre, pulsar binning** modes.
- **Polarization calibration** (RL delay, RL phase, D-terms).
- **AIPS backend** — component interfaces exist; every method is a stub.
- **VLBA / LBA automatic download** — manual-download instructions only. The
  VLBA path (ACCOR, EOP) shares the DiFX code tested on LBA data but has not been
  run on VLBA data.

## Known limitations

- **Quack is deliberately conservative.** An antenna's slew is the leading
  stretch that is low on *all* its baselines. A station with one noisy baseline
  is therefore under-trimmed rather than over-trimmed — the safer error, but it
  leaves some settling data in for the longest baselines.
- **`applymode='calflagstrict'`** discards data with no valid calibration. On
  rsm07 that removes the stations that genuinely never produced fringes; on
  other datasets it may be stricter than you want.
- **Testing is two-tier.** Pure-Python logic has unit tests that run anywhere
  (116 at the time of writing); anything touching CASA is verified by running it
  against real data, not in CI.

## Backend support

| | CASA | dask-ms | AIPS | dummy |
|---|---|---|---|---|
| Import & metadata | yes | via CASA; lazy data access: yes (MS or zarr store) | stub | synthetic |
| Calibration | yes | fringe fit, bandpass, gain, applycal: yes (numpy, parallel); the rest via CASA | stub | synthetic |
| Flagging | yes | via CASA | stub | synthetic |
| Imaging | — | yes (difmapy, always) | stub | synthetic |
| Export | yes | via CASA | stub | synthetic |

`dask-ms` is the CASA backend with the calibration engine replaced; see
[Backends](../backends/index.md#the-dask-ms-backend).

`backend.capabilities()` reports this at runtime for whatever you have
installed.
