# Backends

vlbipy separates the **pipeline logic** (what calibration steps to perform) from the **backend implementation** (how each step is executed). This allows the same pipeline to drive either CASA or AIPS transparently.

## Architecture

The backend system is built on abstract base classes (`BackendComponent`
subclasses) defined in `vlbipy.backends.base`:

| Interface | Responsibility |
| --- | --- |
| `DataOps` | Read metadata, listobs, read visibilities |
| `CalibrationOps` | A-priori/instrumental/fringe-fit calibration, applycal |
| `FlagOps` | Flag management (file-based, autocorr, edges incl. their measurement, quack, tfcrop, aoflagger, outliers, statistics) |
| `PlotOps` | Backend-side data needed for diagnostic plots |
| `ImagingOps` | CLEAN deconvolution (tclean, wsclean) |
| `ExportOps` | Per-source split, UVFITS/MS export, merge |

Each backend (`vlbipy.backends.casa`, `.aips`, `.dask_ms`, `.dummy`)
implements some or all of these; `backend.capabilities()` reports at runtime
which operations a given backend actually supports for each interface, so a
missing one raises a clear error instead of `NotImplementedError` deep in a
call stack. `vlbipy.backends.get_backend()` looks up and instantiates the
right one from a name (`"casa"`, `"aips"`, `"dask-ms"`, `"dummy"`); `Observation`
(one per project inside `VLBIObs`) calls it once at construction time.

Two backends are implemented in full today: `casa`, and `dask-ms`, which is
the CASA backend with the calibration engine replaced by fast numpy solvers
and difmapy imaging (see [The dask-ms backend](#the-dask-ms-backend)). AIPS is a stub — every method raises; `dummy` is an
in-memory synthetic backend used for tests and examples with no external
dependency.

## Backend Selection

The backend is selected when `VLBIObs` is constructed:

```python
# Via Python API
from vlbipy import VLBIObs
obs = VLBIObs("EG078B", network="EVN", backend="CASA")

# Via CLI
vlbipy pipeline -p EG078B -n EVN --backend CASA
```

```toml
# Via TOML configuration
[global]
backend = "casa"   # casa (default) | dask-ms (CASA + fast numpy calibration) | dummy | aips
```

## The dask-ms backend

`backend = "dask-ms"` is a complete pipeline backend. It runs the CASA
backend's pipeline, but the tasks that do the work are replaced by native
Python/numpy code that reads and writes the measurement set and the
calibration tables directly, in parallel worker processes, without
casacore, casatools or casatasks. CASA is imported only when a step still
needs one of the few tasks that have no native version yet. All imaging is
done by difmapy.

Imaging and self-calibration always run in difmapy on the per-source split
measurement sets under this backend: a request for `tclean` or `wsclean`
(`[imaging] imager`, `--imager`, `obs.clean.tclean()`) is redirected to
difmapy with a log message. difmapy is installed by the `daskms` extra.

```toml
[global]
backend = "dask-ms"
```

```bash
vlbipy pipeline -p EG078B -n EVN --backend dask-ms
```

Install it in addition to CASA:

```bash
pip install "vlbipy[casa,daskms]"
```

### What it replaces

| CASA task or tool | Native replacement | Where it is used |
| --- | --- | --- |
| `importfitsidi` + `partition` | `solvers/idi_import.py`, `solvers/idi_subtables.py` | the import: FITS-IDI straight to the Multi-MS, one worker per sub-MS (`VLBIPY_NATIVE_IMPORT=0` uses CASA) |
| `fringefit` | `solvers/fringefit_task.py` | single-band delay (`initial_calibration`), multi-band delay (`fringefit`), the per-scan SNR survey (`scan_snr`) |
| `bandpass` | `solvers/gain_task.py` | the instrumental bandpass |
| `gaincal` | `solvers/gain_task.py` | the pipeline's `scalar_bandpass` |
| `applycal` | `solvers/apply_task.py` | `calibrate.apply` |
| `gencal` (`tsys`, `gc`, `eop`) | `solvers/apriori_task.py`, `solvers/gaincurve_task.py`, `solvers/eop_task.py` | the a-priori calibration (the EOP delays are interpolated between the right daily entries, where CASA's `eop.py` extrapolates from the second and third entry for any time after the fourth) |
| `flagdata` (`manual`, `list`, `quack`, `unflag`; the flag statistics) | `solvers/flag_task.py` | every selection-based flagging step |
| `flagdata(mode='tfcrop')` | `solvers/tfcrop_task.py` | the `flag_tfcrop` autoflagging (CASA's flags, cell for cell) |
| `flagmanager` | `solvers/flagversions.py` | flag versions (same files as CASA's) |
| `statwt` (default binning) | `solvers/statwt_task.py` | `reweight` |
| `accor`, `smoothcal` (of the accor table) | `solvers/accor_task.py` | the correlator amplitude correction (VLBA, LBA) |
| `clearcal` (no selection) | `solvers/clearcal_task.py` | resetting a dataset for a `--scratch` run |
| `mstransform`/`split` | `solvers/split_task.py` | the per-source splits |
| `exportuvfits` | `solvers/uvfits_task.py` | the UVFITS files for difmap |
| `msmetadata`, the `ms` tool readers | `solvers/ms_metadata.py`, `solvers/plot_read.py` | metadata, the diagnostic and final plots |
| `casatools.table`, python-casacore | `vlbipy/casacore_format/` | every table read and write |

Still CASA: `rflag` and `clip` flagging (run on every sub-MS in parallel),
`listobs`, `importuvfits` and `concat`. A call the native code does not reproduce exactly (an unusual
selection, a cal library, `bandtype='BPOLY'`...) is handed to the CASA task
with a warning.

### How it works

The measurement set stays the single source of truth. The solvers read the
visibilities from it and write ordinary CASA calibration tables (Fringe
Jones, B Jones, G Jones), so `casa` and `dask-ms` can be mixed on the same
working directory, and CASA tools (`plotms`, `applycal`) can read the tables.

Each step is cut into jobs — a job reads its part of the data, reduces it and
returns a small result — and the jobs run inside a persistent pool of worker
processes. Processes rather than threads are used because casacore I/O does
not scale with threads. The array work inside a job does: each job gets its
share of the cores (all of them when fewer jobs than cores run, as with one
job per sub-MS), which its numpy blocks and compiled loops use as threads.

The calibration application parallelises over the sub-MSs of a Multi-MS (the
import default, `[import] mms = true`). A plain measurement set is corrected
by a single worker and is several times slower for that step.

### Deferred application

The pipeline applies its calibration chain after every calibration pass and
every self-calibration step, but the steps in between only *read* the
corrected data. Writing CORRECTED_DATA each time copies the whole data set:
for GL048 (22 M rows, a 66 GB Multi-MS) that is about 60 GB of disk traffic
per application, 15-25 minutes on a disk array, seven times per run. The
dask-ms backend defers those applications instead, the way AIPS keeps its
calibration in CL tables and applies it while a task reads (`DOCAL`):

- `calibrate.apply(defer=True)` writes only the calibration flags (the data
  without a valid solution, from FLAG and the tables alone: a few hundred MB)
  and records the chain of every field in `<ms>/.vlbipy_corrected/`, with a
  copy of each table, since later solves rewrite them in place;
- the dask-ms readers of the corrected column (the flagging and plot readers,
  `statwt`, the per-source split) read DATA and apply the recorded chain as
  they read it, with the code that writes CORRECTED_DATA: they see the same
  visibilities, flags and weights an `applycal` would have left, bit for bit;
- CORRECTED_DATA is written by `calibrate.write_corrected()`, which the
  pipeline calls once, at the end of the calibration, and before anything
  reads the column itself: a CASA task, the `ms` tool, jplotter.

A `statwt` between two applications is kept: its weights stay those of the
corrected data until the next application rebuilds them, as on disk.

### Cost of applying calibration

Calibrating rows takes two steps, and both are vectorized:

- **The gains.** A row needs the chain's gains at its two antennas, its
  subband and its time. Those points are listed once per block, and each
  table is evaluated at all of them in one pass, with no loop over antennas or
  subbands: a table's rows are indexed once, and one search finds the
  solutions around every point. A bandpass is mapped onto the data
  channels once per distinct solution. Each point is evaluated on its own, so
  a row gets the same gains in any block it is read with.
- **The correction.** Feed rotation, division by the gains, flags and weights
  are one pass over the visibilities.

When numba is installed (it comes with the `daskms` extra), the per-value
work (multiplying the factors out on every channel, and the correction) runs
as compiled loops on all the cores of the job, with numpy's own formulas, so
the results match numpy's (bit for bit on the machine measured below).
Without numba, numpy does the same work in threads.

Measured on GL048 (the full 12-table chain with fringe, bandpass, Tsys and
gain-curve tables, 16 subbands x 64 channels, 2 correlations), on a 2010
Xeon E5620 (no AVX):

| Per row | before (1 core) | numpy (1 core) | numba (1 core) | numba (8 threads) |
| --- | --- | --- | --- | --- |
| Gains, 400k-row batch | 24 µs | 5.7 µs | 3.0 µs | 2.0 µs |
| Gains, one scan (900 rows) | 113 µs | 12 µs | 10 µs | 10 µs |
| Correction | 7 µs | 4.6 µs | 2.1 µs | 0.75 µs |

A one-scan read pays a fixed cost of about 10 ms per call for the gains.

A GPU does not help here. On this machine's TITAN X the correction kernel takes
0.02 µs per row, but moving the visibilities, flags and weights over PCIe and
back takes 0.56 µs per row (6 GB/s), about what this CPU needs with 8 threads.
A modern laptop CPU is several times faster per core than this one, and the
disk, not the arithmetic, sets the pace of a pass over the data.

The engine lives in `vlbipy.solvers`:

| Module | Role |
| --- | --- |
| `solvers/workers.py` | worker pool |
| `solvers/msio.py` | measurement-set layout index, chunked reads and writes |
| `solvers/calblock.py` | read and calibrate one block of data |
| `solvers/fringefit_task.py`, `solvers/fringe.py` | fringe fitting |
| `solvers/gain_task.py` | bandpass and gaincal |
| `solvers/apply_task.py`, `solvers/apply.py` | applycal and calibration-table interpolation |
| `solvers/kernels.py` | compiled loops of the calibration (numba, optional) |
| `solvers/corrected.py` | the record of what CORRECTED_DATA holds (deferred applications) |
| `solvers/parang.py` | parallactic angle |
| `solvers/caltable.py` | calibration tables |

### Configuration

| Setting | Effect |
| --- | --- |
| `VLBIPY_WORKERS=N` (environment) | Number of worker processes. Default: the number of CPU cores, capped at 16. `1` runs everything in-process. |
| `VLBIPY_THREADS=N` (environment) | Threads a job uses for its array work. Default: its share of the cores (the cores divided by the jobs running at once). |
| `VLBIPY_NUMBA=0` (environment) | Uses numpy for the calibration loops even when numba is installed. |
| `VLBIPY_TABLE_ENGINE=native`, `casacore` or `casatools` (environment) | Forces the table I/O library. Under the dask-ms backend the default is the native reader/writer (`vlbipy.casacore_format`), which opens a table it cannot handle (an unusual storage manager) with python-casacore when it works on the machine, otherwise with `casatools.table`. Under the `casa` backend the default stays python-casacore / `casatools.table`. |
| `VLBIPY_NATIVE_IMPORT=0` (environment) | Imports the FITS-IDI files with CASA's `importfitsidi` + `partition` instead of the native import. The native import also hands over to CASA by itself for data it does not cover (DiFX/VLBA weights, files whose tables do not match one to one, Stokes I/Q/U/V). |
| `[import] mms` | `true` (default) imports to a Multi-MS, which lets `applycal` run in parallel over its sub-MSs. |
| `[import] zarr_store` | `true` also writes the zarr store `<code>.zarr` (default `false`). |

The zarr store is optional and is no longer written at import. It is a
read-only snapshot for lazy access (`obs.data`), and is enough on its own to
rebuild the report and metadata after the measurement set has been deleted.
Besides `[import] zarr_store = true`, it is produced by
`vlbipy export -p CODE --format dask-ms`. The measurement set is always kept;
the former `[import] keep_ms` key no longer exists.

### Performance

Measured on the RSM07 test observation (EVN, 14 antennas, 4 subbands x 64
channels, 1.4 M rows, 2.8 h; 12-core desktop, warm page cache; CASA 6.7.3):

| Step (same call, same data) | CASA task | dask-ms backend | speed-up |
|---|---|---|---|
| Single-band delay, 1 scan (fringefit) | 7-8 s | 0.2 s | ~40x |
| Multi-band delay, 33 scans (fringefit, combine=spw, dispersive) | 87-103 s | 1.3-1.6 s | ~65x |
| SNR survey, 63 scans (fringefit) | 237-278 s | 2.5-3.3 s | ~85x |
| Bandpass, 1 scan | 1.7-2.0 s | 0.3 s | ~6x |
| Scalar bandpass (gaincal calmode='a', whole phase calibrator) | 19-23 s | 1.3-1.6 s | ~14x |
| applycal, 3 fields, 8 tables | 72 s | 8-14 s | 5-9x |

`applycal` is bound by how fast the disk takes the roughly 3-5 GB of
corrected data and weights it has to write. The fringe fits were where the
CASA pipeline spent most of its time: with three calibration passes, roughly
20 of the about 25 minutes of calibration on this dataset.

### Agreement with CASA

The solvers were verified against the CASA tasks on RSM07.

**applycal** — checked table by table and for the full 8-table chain against
`casatasks.applycal` on the whole measurement set: identical flags; corrected
visibilities equal to within 4e-4 (relative, worst case; median 2e-5; phases
within 0.02 deg); identical weights on unflagged data. Weights of *flagged*
samples can differ from CASA's. The following CASA behaviour is reproduced:

- flagged solutions are not skipped in time interpolation (the data are
  flagged instead);
- flagged bandpass channels are interpolated across, unless a `...flag`
  frequency interpolation mode is used;
- fringe solutions are interpolated in parameter space, with CASA's
  rate-aware phase blend;
- gain curves are evaluated in zenith angle;
- the parallactic angle uses the direction of date and casacore's AZEL
  (geocentric-latitude) convention;
- weights are rebuilt from `SIGMA` and scaled by the Tsys, gain-curve and G
  tables only (never by bandpasses), so re-applying does not calibrate them
  twice.

**Fringe fit** — the same solutions are detected and flagged as by CASA in
every case tested (single-band, multi-band, survey). Delays agree to about
0.1 ps (median) and 1.5 ps (worst), single-band phases to about 1 deg. Two
known differences:

1. The delay rate of a multi-band (`combine='spw'`) solve is referred to the
   centre frequency of the band; CASA's is referred to the first channel. The
   tabulated rates therefore differ by the ratio of those frequencies (3.6% at
   L band for this dataset). The table is applied with the centre frequency by
   both CASA and vlbipy, so the vlbipy value is the self-consistent one.
2. The SNR column holds the AIPS FRING estimate from the FFT stage (the value
   `minsnr` is applied to), whereas CASA writes a number that scales with the
   square root of the data weights. Both agree on what is a detection, but
   the numbers are not comparable.

**bandpass** — same flagged channels as CASA, amplitudes within about 0.1%,
per-channel phases within 0.02 deg apart from a constant phase per antenna
and subband of a few degrees. CASA applies the prior fringe table slightly
differently while solving than `applycal` does; vlbipy uses the `applycal`
behaviour in both places, and the constant is absorbed by the next delay
solve. The SNR follows CASA's convention (formal SNR divided by the square
root of the reduced chi-square) to about 5%.

**gaincal** (amplitude, one solution per antenna and subband) — same flags,
amplitudes within about 1%.

### Fall-backs to CASA

Anything the fast engine does not cover runs the CASA task instead, with a
warning:

- cal-library files (`--callib`);
- `bandtype='BPOLY'`;
- gain types other than `G`;
- `combine='spw'` in `bandpass` or `gaincal`;
- an `applymode` other than `calflag`, `calflagstrict` or `calonly`.

## Adding a New Backend

To add support for a new backend:

1. Add a new module under `vlbipy/backends/` (e.g. `vlbipy/backends/newbackend.py`)
2. Implement the `BackendComponent` interfaces from `vlbipy.backends.base` it can support
3. Add the backend's name to `BackendKind` in `vlbipy.models`
4. Register it in `get_backend()` (`vlbipy/backends/__init__.py`)

See the [API reference](../api/backends.md) for the full interface definitions.
