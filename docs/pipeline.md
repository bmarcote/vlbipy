# Pipeline Workflow

`VLBIObs.run()` chains together the same callable namespaces you can call
individually — `import_data`, `calibrate`, `flag`, `plot`, `export` — in a
fixed order. This page is the detailed walkthrough of what each step does
and why it's ordered the way it is; for the short version see
[Full pipeline](usage/pipeline.md), and for calling these interactively see
[Interactive Python](usage/python.md).

!!! warning "Work in progress"

    Calibration through to per-source calibrated data works end to end on
    the CASA backend. Imaging and self-calibration are not implemented yet —
    `run()` skips them with a warning. See [Status](usage/status.md).

## Overview

```text
Raw data → Import → Metadata → A-priori cal → A-priori flags
    → Instrumental cal (SBD → bandpass → SBD) → Edge-channel flags → Apply
    → Global fringe fit → Apply → Quack → Outlier flags
    → Second pass (re-solve on clean data) → Scalar bandpass → Apply
    → Plots → Export per source → [Imaging — not implemented]
```

Progress is recorded per step in `<work_dir>/.pipeline_state.json`; a step is
skipped on a repeated run unless `force=True` (`--force` on the CLI) or a
step later in the chain invalidates it. The calibration chain itself is
persisted in `<work_dir>/.caltables.json`, so a resumed process solves on top
of the tables earlier steps produced rather than starting from raw data.

## Step by step

### `import_data`

Locates FITS-IDI files already on disk, or downloads them (EVN: the JIVE
archive, including `.antab` and `.uvflg`; VLBA/LBA: mostly manual — see
[Observatories](observatories/index.md)). Imports into the backend's native
format (a CASA Multi-MS for the CASA backend) and reads metadata: sources,
antennas, scans, frequency setup.

### `calibrate.a_priori`

Amplitude calibration from the Tsys and gain-curve data appended to the
FITS-IDI at import time (`gencal`, plus an EOP table for VLBA/LBA). The Tsys
table is de-spiked (and optionally smoothed) before it enters the chain, and
each table is plotted per antenna. Refuses to run if the data has no Tsys /
gain-curve information to calibrate from.

### `flag.apriori`

Applies the observatory's a-priori flag table (`.uvflg` for EVN/LBA — VLBA's
equivalent is already inside the FITS-IDI, so `importfitsidi` has applied it
by the time this runs) and flags the autocorrelations, which carry no
interferometric information.

### `calibrate.select_calibration_data`

Not a pipeline step on its own, but the decision every instrumental step
below depends on: which antennas and scan(s) to solve the instrumental
calibration on. A short fringe fit (`calibrate.scan_snr`) surveys every
calibrator scan; antennas that recorded every subband and were detected
above `[calibration].detection_snr` (default 7σ) qualify, and scans are then
chosen so that all of those antennas are covered — using several linked
scans when no single scan covers the whole array. Falls back from fringe
finders to phase calibrators to targets if the fringe finders don't yield a
usable scan.

### `calibrate.instrumental`

Single-band delay (SBD) → bandpass → SBD again, solved on the antennas and
scan(s) above:

1. A first SBD pass gives rough per-antenna delays.
2. A first fringe fit refines them into a rough multi-band delay.
3. The bandpass is then solved on data with those delays already removed —
   otherwise residual delay slopes across each subband get absorbed into the
   band shape.
4. Both delays are re-solved (`sbd2`, `mbd2`) *through* the bandpass, as
   incremental corrections on top of the first pass — the first-pass tables
   stay in the chain, everything applies together.

`calibrate.verify_solutions` then checks that every antenna that recorded a
subband actually got a solution in it — a silent gap here means that
antenna/subband is flagged at apply time, shrinking the array without
telling you.

### `calibrate.edge_channels`

Measures the bandpass roll-off at each subband edge (rather than flagging a
blind configured fraction) and flags that many channels at both edges of
every subband, via `flag.edges`.

### `calibrate.apply`

Applies the accumulated calibration chain to every field (or one `field=`).
Uses a CASA cal library rather than parallel `gaintable`/`interp`/`spwmap`
lists — see [The cal libraries](usage/index.md#the-cal-libraries) for the
format.

### `calibrate.fringefit`

The global (multi-band delay) fringe fit, on every calibrator. Below
`[calibration].ionos_max_ghz` (6 GHz by default) it also solves for the
dispersive (ionospheric) delay, since at those frequencies the residual
delay is genuinely frequency-dependent; above it, that term is skipped
(`--no-ionos` disables it unconditionally). The resulting table is tied to
the phase calibrator(s) (`gainfield`), which is how those fringe solutions
reach the target at apply time.

### `flag.quack`

Measures each antenna's settling time after a slew and trims it. Uses the
stretch that is low on *all* of an antenna's baselines, so a station with
one noisy baseline is under-trimmed rather than over-trimmed.

### `flag.outliers`

Per-baseline robust (median/MAD) outlier flagging on calibrated amplitudes.

### `calibrate.second_pass`

Drops every data-derived table (SBD, bandpass, MBD, ...) back to the
a-priori ones (Tsys, gain curve, EOP — which don't depend on the data) and
re-solves the whole instrumental + fringe-fit chain on the now-flagged data.
The first pass's solutions were biased by whatever the flagging steps above
later removed (band edges, outliers, slewing data); re-solving from a clean
a-priori baseline avoids inheriting that bias.

### `calibrate.scalar_bandpass`

Solves one amplitude gain per antenna and subband on the phase calibrator
(a point source — a resolved fringe finder would have its structure absorbed
into the antenna gains), levelling the subband amplitudes before the final
apply.

### Diagnostics

`plot.corners`, `plot.spectrum`, `plot.timeseries` (per phase calibrator or
target), and `plot.radplot` on the calibrated data.

### `export.per_source`

Splits the calibrated data per source and writes UVFITS alongside each MS —
the deliverable of the calibration, and the input to imaging/self-cal
outside vlbipy in the meantime.

### Imaging

*Not implemented.* `run()` checks whether the active backend implements
`image.clean`; if not (true of CASA today), it logs a warning and returns
without images rather than failing the whole run.

## Campaigns

Give `VLBIObs` several project codes and every step above fans out across
them; `merge()` combines the calibrated data before `export.per_source()`.
See [Full pipeline — Campaigns](usage/pipeline.md#campaigns).

## Self-calibration

`obs.selfcal.phase()` / `obs.selfcal.ampphase()` exist on the object model
today but only run on the dummy backend — see the
[Self-Calibration API reference](api/selfcal.md) and
[Status](usage/status.md).
