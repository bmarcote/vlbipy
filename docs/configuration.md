# Configuration

vlbipy uses a **three-layer configuration cascade** where later layers override earlier ones:

1. **`defaults.toml`** — built-in defaults shipped with the package (at `src/vlbipy/templates/defaults.toml`)
2. **User input file** — a project-specific TOML file
3. **CLI arguments** — command-line overrides for quick changes

This design ensures reproducibility (the full configuration is captured in a TOML file) while allowing rapid iteration from the command line.

## Configuration File Format

Configuration files use [TOML](https://toml.io/) syntax. Here is a complete example:

```toml
[global]
project = "EG078B"
observatory = "EVN"          # EVN, VLBA, or LBA
backend = "CASA"             # CASA or AIPS
reference_antenna = ["EF"]   # List of preferred reference antennas (auto-prioritised if empty)
obsdate = "231015"           # YYMMDD format (required for EVN auto-download)
pwd = ""                     # Working directory override; empty = cwd

[sources]
fringe_finders = ["3C345", "4C39.25"]
phase_calibrators = ["J1230+5600"]
targets = ["J1234+5678"]
check_sources = []
polarization_calibrators = []

[import]
scan_gap = 15                # Seconds gap threshold for defining new scans in importfitsidi

[flagging]
flag_autocorrelations = true
edge_channels_fraction = 0.05 # Fallback if the subband edges cannot be measured
outlier_sigma = 5.0          # Robust-sigma cut for per-baseline outlier flagging
aoflagger_strategy = "default"
tfcrop_winsize = 3
tfcrop_timecutoff = 4.5
tfcrop_freqcutoff = 4.5

[calibration]
parang = true                # Apply parallactic angle correction throughout
ionos = true                 # Solve the dispersive delay below ionos_max_ghz
ionos_max_ghz = 8.0          # The ionosphere matters below this observing frequency
detection_snr = 7.0          # Fringe SNR above which an antenna counts as detected

[calibration.sbd]
solint = "inf"               # Solution interval for the instrumental delay
minsnr = 10.0                # Minimum SNR for a valid fringe-fit solution
channel_fraction = 0.7       # Central channels used (the edges are not yet bandpass-corrected)
zerorates = true             # The instrumental delay is constant in time: do not fit a rate

[calibration.bandpass]
solint = "inf"               # Solution interval (inf = average all)
combine = "scan"             # Combine across scans when solving
minsnr = 3.0
solnorm = true               # Normalise: keep the band shape, not the flux scale

[calibration.mbd]
solint = "inf"               # Solution interval for the global fringe fit
combine = "spw"              # Pool the subbands for sensitivity
minsnr = 5.0
zerorates = false            # Atmospheric residuals vary in time: fit the rate

[selfcal]
enabled = true
selfcal_target = false       # Also self-calibrate target sources
phase_rounds = 4             # Number of phase-only selfcal iterations
ampphase_rounds = 5          # Number of amplitude+phase selfcal iterations
convergence_threshold = 0.05 # Stop if dynamic range improves by < 5%

[imaging]
weighting = "briggs"
robust = [-2, -1, 0, 1, 2]  # Briggs robust values to image at
pixels_per_beam = 10         # Used to auto-compute cell size if not set
niter = 4000                 # CLEAN iterations
threshold_sigma = 3.0        # CLEAN threshold in units of image RMS
deconvolver = "hogbom"
produce_fits = true
produce_png = true

[export]
split_per_source = true
export_uvfits = true
time_average = ""            # No time averaging by default
channel_average = 1          # Channels per output subband (1 = average all)
```

## Section Reference

### `[global]`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `project` | string | `""` | Project code (e.g. `"EG078B"`) |
| `observatory` | string | `"EVN"` | VLBI network: `EVN`, `VLBA`, or `LBA` |
| `backend` | string | `"CASA"` | Data reduction backend: `CASA` or `AIPS` |
| `reference_antenna` | list[str] | `[]` | Preferred reference antennas in priority order. Auto-selected if empty. |
| `obsdate` | string | `""` | Observation date in YYMMDD format (required for EVN auto-download) |
| `pwd` | string | `""` | Working directory override. Defaults to current working directory. |

### `[sources]`

Source names must exactly match those in the data file. Both the new-style keys below and the legacy aliases (`target`, `phasecal`, `fringefinder`, `polcal`, `checksource`) are supported.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `fringe_finders` | list[str] | `[]` | Fringe finder source names |
| `phase_calibrators` | list[str] | `[]` | Phase calibrator source names |
| `targets` | list[str] | `[]` | Target source names |
| `check_sources` | list[str] | `[]` | Check source names |
| `polarization_calibrators` | list[str] | `[]` | Polarization calibrator source names |

### `[import]`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `scan_gap` | int | `15` | Time gap in seconds that defines a new scan boundary during import |

### `[flagging]`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `flag_autocorrelations` | bool | `true` | Flag autocorrelation data |
| `edge_channels_fraction` | float | `0.05` | Fraction of channels flagged on each subband edge (used only when the edges are not measured) |
| `quack_interval` | float | `0` | Seconds to flag at the start of every scan for every antenna; `[flagging.quack_antennas]` (`EF = 4`) overrides per antenna; when neither is set the slew is measured from the data |
| `edge_outlier_sigma` | float | `6.0` | MAD sigmas defining the flat interior of a subband when measuring the edges |
| `max_edge_fraction` | float | `0.25` | Never trim more than this fraction from either subband edge |
| `quack_sigma` | float | `2.0` | MADs below the baseline's scan median that count as off-source when measuring the slew |
| `quack_max_seconds` | float | `50.0` | Never trim more than this from a scan start |
| `outlier_sigma` | float | `5.0` | Robust-sigma cut for per-baseline outlier flagging |
| `aoflagger_strategy` | string | `"default"` | AOFlagger strategy file name |
| `tfcrop_winsize` | int | `3` | Window size for the tfcrop auto-flagger |
| `tfcrop_timecutoff` | float | `4.5` | Sigma cutoff along time axis |
| `tfcrop_freqcutoff` | float | `4.5` | Sigma cutoff along frequency axis |

### `[calibration]`

Settings shared by every calibration step.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `parang` | bool | `true` | Apply the parallactic-angle correction in all calibration steps |
| `refant_snr_select` | bool | `true` | Rank reference antennas from the measured fringe SNR rather than the static preference list |
| `eop_file` | string | `""` | `usno_finals.erp` path for EOP corrections (VLBA/LBA); empty auto-downloads |
| `ionos` | bool | `true` | Solve the dispersive (ionospheric) delay in the fringe fit below `ionos_max_ghz`. Set to `false` (or pass `--no-ionos`) to never solve it. |
| `ionos_max_ghz` | float | `8.0` | Observing frequency below which the ionosphere matters enough to fit |
| `snr_channel_fraction` | float | `0.7` | Central fraction of each subband used by the per-scan SNR survey |
| `snr_max_scans` | int | `24` | Cap on scans fringe-fitted by the survey (`0` = all); it only needs relative SNR |
| `smooth_tsys` | bool | `true` | De-spike the Tsys table after `gencal` |
| `tsys_outlier_sigma` | float | `6.0` | MAD sigmas beyond which a Tsys solution is an outlier |
| `tsys_smooth_passes` | int | `3` | Maximum de-spiking passes |
| `detection_snr` | float | `7.0` | Fringe SNR above which an antenna counts as detected |
| `require_all_subbands` | bool | `true` | Only calibrate on antennas that recorded the whole band |

!!! note "Passing anything the backend accepts"
    Every key in a `[calibration.*]` section below is forwarded verbatim to the
    backend function of the same name, so any parameter that function accepts can
    be tuned from the config file even if it is not listed here.

### `[calibration.sbd]`

Single-band (instrumental) delay. Runs twice: once on the selected fringe-finder
scan, and again as `sbd2` after the bandpass.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `solint` | string | `"inf"` | Solution interval |
| `minsnr` | float | `10.0` | Minimum SNR for a valid solution |
| `channel_fraction` | float | `0.7` | Central fraction of channels used; the edges are not yet bandpass-corrected |
| `zerorates` | bool | `true` | The instrumental delay is constant in time, so do not fit a rate |

### `[calibration.bandpass]`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `solint` | string | `"inf"` | Solution interval |
| `combine` | string | `"scan"` | Axes to combine when solving |
| `minsnr` | float | `3.0` | Minimum signal-to-noise ratio |
| `solnorm` | bool | `true` | Normalise: keep the band shape, not the flux scale |
| `corrdepflags` | bool | `true` | Respect per-correlation flags |
| `bandtype` | string | `"B"` | `B` (per-channel) or `BPOLY` |

### `[calibration.mbd]`

Multi-band delay (global fringe fit). Runs twice: `mbd` and, after the bandpass
and `sbd2`, `mbd2`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `solint` | string | `"inf"` | Solution interval |
| `combine` | string | `"spw"` | Pool the subbands for sensitivity (needs an spw map at apply time) |
| `minsnr` | float | `5.0` | Minimum signal-to-noise ratio |
| `zerorates` | bool | `false` | Atmospheric residuals vary in time, so fit the rate |
| `dispersive` | bool | *auto* | Also solve the dispersive delay. Left unset it follows `[calibration].ionos` and the observing frequency; set it here to decide explicitly. |

#### The dispersive (ionospheric) delay

The ionosphere delays low frequencies more than high ones, so its contribution is
*dispersive*: below roughly 6 GHz the residual is not a single non-dispersive
delay, and fitting one leaves a frequency-dependent phase behind. The fringe fit
therefore solves for the dispersive delay as well whenever the observing
frequency is below `[calibration].ionos_max_ghz`, and skips it above — where the
effect is negligible and the extra free parameter only costs SNR.

The decision is logged on every run, for example:

```
fringefit: observing at 1.658 GHz, below 6 GHz -> will solve for the dispersive (ionospheric) delay
```

To turn it off regardless of frequency:

```toml
[calibration]
ionos = false
```

or on the command line:

```bash
vlbipy run -p rsm07 -t 3C395 --no-ionos
```

### `[calibration.scalar_bandpass]`

A single amplitude per antenna and subband for the whole run, solved on the phase
calibrator — it levels the subbands relative to one another (the equivalent of
Difmap's per-antenna gain corrections).

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `solint` | string | `"inf"` | One solution for the whole run |
| `combine` | string | `"scan"` | Axes to combine when solving |
| `minsnr` | float | `3.0` | Minimum signal-to-noise ratio |
| `solnorm` | bool | `true` | Keep the a-priori flux scale; level the subbands only |

### `[calibration.tsys_smooth]`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `window` | int | `5` | Running-median window, in solutions, used to de-spike Tsys |

### `[calibration.snr_survey]`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `minsnr` | float | `0.0` | Keep every solution so weak antennas still appear in the survey |

### `[selfcal]`

Controls `obs.selfcal`. Not implemented for the CASA backend yet — see
[Self-Calibration](api/selfcal.md) and [Status](usage/status.md).

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enable self-calibration step |
| `selfcal_target` | bool | `false` | Also self-calibrate target sources (in addition to phase calibrators) |
| `phase_rounds` | int | `4` | Number of phase-only self-calibration iterations |
| `ampphase_rounds` | int | `5` | Number of amplitude+phase self-calibration iterations |
| `convergence_threshold` | float | `0.05` | Stop iterating when dynamic range improvement falls below this fraction |

### `[imaging]`

Controls `obs.clean`. Not implemented for the CASA backend yet — see
[Status](usage/status.md). Final imaging uses Briggs weighting with multiple
robust values.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `weighting` | string | `"briggs"` | Visibility weighting scheme |
| `robust` | list[float] | `[-2, -1, 0, 1, 2]` | Briggs robust values to image at (−2 = uniform, +2 = natural) |
| `pixels_per_beam` | int | `10` | Used to auto-compute cell size from the synthesised beam |
| `niter` | int | `4000` | Number of CLEAN deconvolution iterations |
| `threshold_sigma` | float | `3.0` | CLEAN threshold in units of the estimated image RMS |
| `deconvolver` | string | `"hogbom"` | CLEAN deconvolution algorithm (`hogbom`, `clark`, `multiscale`) |
| `produce_fits` | bool | `true` | Export CLEAN images as FITS files |
| `produce_png` | bool | `true` | Export CLEAN images as PNG thumbnails |

### `[export]`

Controls `obs.export.per_source`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `split_per_source` | bool | `true` | Write each source to its own MS file |
| `export_uvfits` | bool | `true` | Export a UVFITS file alongside each split MS |
| `time_average` | string | `""` | Time averaging interval (e.g. `"10s"`). Empty = no averaging. |
| `channel_average` | int | `1` | Channels per output subband. `1` = average all channels within each subband. |

## CLI Parameter Mapping

CLI arguments override the corresponding TOML parameters:

| CLI Flag | TOML Equivalent |
| --- | --- |
| `-n` / `--network` | `[global] observatory` |
| `--backend` | `[global] backend` |
| `--refant` | `[global] reference_antenna` |
| `-t` / `--target` | `[sources] targets` |
| `-pcal` / `--phasecal` | `[sources] phase_calibrators` |
| `-ff` / `--fringe-finder` | `[sources] fringe_finders` |
| `--pwd` | `[global] pwd` |
| `--no-ionos` | `[calibration] ionos = false` |
| `--from-step` | *(run control, no TOML equivalent)* |
| `--scratch` | *(run control, no TOML equivalent)* |
| `--force` | *(run control, no TOML equivalent)* |
