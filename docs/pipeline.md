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
skipped on a repeated run unless `force=True` (`--scratch` on the CLI) or a
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
FITS-IDI at import time (`gencal`; for DiFX data, VLBA/LBA, also `accor` + `smoothcal` on the
autocorrelations and an EOP table, see [LBA](observatories/lba.md#accor-and-eop-difx-data)). The Tsys
table is de-spiked (and optionally smoothed) before it enters the chain, and
each table is plotted per antenna. Refuses to run if the data has no Tsys /
gain-curve information to calibrate from.

The gain-curve plot shows each curve divided by the DPFU of its antenna, against
elevation: a healthy curve sits near 1 whatever the size of the dish. The
elevations the antenna actually observed are shaded; a polynomial that had to
be shortened for CASA is only meaningful there. The DPFUs are kept in
`caltables/<code>.gcal.dpfu.json`; for a table made before that file existed
the curve is divided by its peak instead, and the plot says so.

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

Detected is not the same as on source. An antenna sitting in a sidelobe of a
bright fringe finder still gives fringes, and its fringe SNR hardly drops
(the survey's SNR follows the coherence of the phases, not the amplitude). So
the amplitudes are compared as well: each antenna's raw amplitude in a scan
against its own best scan of the same source, split per antenna so that a
weak partner does not count against it. Below `[calibration].weak_scan_level`
(default 0.3) the antenna is not solved on that scan and gets another one.

The choice is made once, on the first pass, and reused by the later passes
(re-surveying data that has been flagged in the meantime only makes the
detectable set shrink). One thing reopens it: an antenna whose data in the
scan it is solved on has since been flagged — typically an antenna that was
detected there but not properly on source, which the off-source flagging
removes. Left alone it would get no instrumental solution and lose all its
data when the calibration is applied. It is given another scan instead,
linked through an antenna that is still solved; the other antennas keep
theirs.

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

### `flag.edges`

Measures the bandpass roll-off at each subband edge (rather than flagging a
blind configured fraction) and flags that many channels at both edges of
every subband. Falls back to `[flagging].edge_channels_fraction` when no
bandpass table exists yet; `edge_channels=N` forces an explicit count.
Everything about deciding *what* to flag lives in the flag namespace, even
when the decision is read from a calibration table.

### `calibrate.apply`

Applies the accumulated calibration chain to every field (or one `field=`).
Uses a CASA cal library rather than parallel `gaintable`/`interp`/`spwmap`
lists — see [The cal libraries](usage/index.md#the-cal-libraries) for the
format.

`apply(defer=True)`, which the pipeline uses for every application before
the last one, lets the dask-ms backend skip writing CORRECTED_DATA: it writes
the calibration flags and records the chain, and the readers of the corrected
data calibrate DATA as they read it (see
[Deferred application](backends/index.md#deferred-application)).
`calibrate.write_corrected()` writes the column; the pipeline calls it before
the calibrated diagnostics. Other backends apply as usual.

### `calibrate.fringefit`

The global (multi-band delay) fringe fit, on every calibrator. Below
`[calibration].ionos_max_ghz` (8 GHz by default) it also solves for the
dispersive (ionospheric) delay, since at those frequencies the residual
delay is genuinely frequency-dependent; above it, that term is skipped
(`--no-ionos` disables it unconditionally). The resulting table is tied to
the phase calibrator(s) (`gainfield`), which is how those fringe solutions
reach the target at apply time.

### `flag.quack`

Runs *before* any solve, so the instrumental delays are never fitted on
slewing data. Per-antenna seconds from `[flagging.quack_antennas]` win, then
`[flagging].quack_interval` for the whole array; only when neither is
configured is the settling time measured from the raw data per antenna,
using the stretch that is low on *all* of an antenna's baselines.

### `flag.initial`

Initial auto-flagging of the calibrators (tfcrop, per baseline, flags never
extended) on the raw data — strong outliers only. Deep flagging waits for
calibrated data.

### `flag.outliers`

Per-baseline robust (median/MAD) outlier flagging on calibrated amplitudes.

### `calibrate.reweight`

`statwt` on the calibrated data. The new weights expose bad data that hid
until now, so the pipeline flags outliers again and re-runs the whole chain
(`second_pass(step="third_pass")` + `scalar_bandpass`). Disable with
`[calibration.reweight].enabled = false`.

### `flag.statistics`

Flagged fraction overall and per antenna / subband, counted over observable
data only (no autocorrelations, no never-recorded visibilities), so an
antenna at 100% really lost the data it recorded. Included in `report()`.

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

`plot.diagnostics(column=...)` runs the standard set twice: on the raw data
before any flagging or solve (scan SNR matrix, full-Stokes cross-correlation
spectra on the fringe-finder scans, spectra and time series on baselines to
the reference antenna, corner plots, radplot, uv coverage) and again on the
calibrated data, after self-calibration, from CORRECTED_DATA written with the
final chain. A failing plot is reported as a warning; it never aborts the
reduction.

The raw set also holds the time evolution of the whole observation
(`plot.time_evolution()`): amplitude and phase against time on every baseline
to the reference antenna, coloured by source, vector-averaged over the central
channels of a subband every antenna recorded. When the reference antenna did
not observe the whole time, or its data show stretches without unflagged
samples or without fringes, the same figure is made for a second (or third)
reference antenna, the one with the highest fringe SNR among those that cover
the gap; when no subband was recorded by every antenna, further subbands are
added until each antenna appears. A resumed run whose raw diagnostics were
made before this figure existed adds it once.

### `export.per_source`

Splits the calibrated data per source and writes UVFITS alongside each MS —
the deliverable of the calibration, and the input to imaging/self-cal
outside vlbipy in the meantime.

### Imaging

*Not implemented.* `run()` checks whether the active backend implements
`image.clean`; if not (true of CASA today), it logs a warning and returns
without images rather than failing the whole run.

## Campaigns

Give `VLBIObs` several project codes and every step above runs on each epoch
on its own, up to and including its self-calibration and images. The epochs
are then combined (`combine()`, see [Full pipeline — Campaigns](usage/pipeline.md#campaigns)):

1. the calibrated data of each phase calibrator are concatenated across the
   epochs, each scaled to a common flux density (calibrators vary), and
   modelled together: phase self-calibration, then CLEAN. The joint uv coverage
   constrains structure that no single track does;
2. each epoch of that calibrator is self-calibrated against the joint model
   (scaled back to that epoch's flux density, and never re-fitted): the phase
   ladder, each step kept only if it improves the fit. Station amplitudes are
   left as the epoch's own self-calibration set them unless
   `[selfcal].joint_amplitude = true` (the model is built from other epochs of
   a variable source; amplitudes fitted to it made the V589 targets worse). The gains become a `joint_<source>` table in that
   epoch's chain, applied to the fields the calibrator's own self-calibration
   goes to;
3. every epoch is re-applied, re-split and re-imaged;
4. the epochs of every source are concatenated and imaged together.

A source declared for the campaign but absent from an epoch is ignored in that
epoch, and a calibrator observed in a single epoch gets no joint model.

## Self-calibration

`obs.selfcal.phase()` / `obs.selfcal.ampphase()` exist on the object model
today but only run on the dummy backend — see the
[Self-Calibration API reference](api/selfcal.md) and
[Status](usage/status.md).
