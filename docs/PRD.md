# PRD: vlbipy — Backend-Agnostic VLBI Reduction Suite (API-First Design)

> **Status:** API-first design milestone. This PRD is a **ground-up redesign** of
> `vlbipy` and supersedes the previous `Project`-centric implementation, which is
> retained only as reference material. The near-term goal is a **robust, complete,
> playable Python API** backed by an in-memory *dummy* backend. Real backends (CASA,
> WSClean, AOFlagger, AIPS) and their actual task calls are deliberately deferred to
> later phases.

---

## 1. Problem Statement

VLBI (Very Long Baseline Interferometry) data reduction is one of the most manual,
expert-dependent workflows in radio astronomy. Reducing a single EVN, VLBA, or LBA
observation currently requires an experienced astronomer to hand-run a long sequence
of CASA (or AIPS) tasks — downloading raw correlator output, appending Tsys/gain-curve
information, flagging, fringe-fitting, bandpass calibration, self-calibration, and
imaging — while manually tracking which steps have run, which intermediate products are
stale, and which parameters worked for a given array/frequency combination. Every
observatory (EVN, VLBA, LBA), every observing mode (continuum, spectral line,
multi-phase-center, multi-epoch astrometry), and every downstream tool (WSClean,
eht-imager, Difmap) needs bespoke, one-off scripting.

There is no single tool a user can point at a project code and trust to produce a
scientifically sound, science-ready dataset without manual babysitting, while still
allowing an expert to intervene, inspect, and safely re-run only what changed — from a
CLI, an interactive Python/Jupyter session, or a dashboard, using the *same* object model.

## 2. Solution

`vlbipy` is a modular, backend-agnostic suite for VLBI data. The **primary product is a
high-level Python API** whose central object, `VLBIObs`, represents one *or many*
projects (a campaign) and exposes the entire reduction — import, calibration, flagging,
plotting, imaging, self-calibration, export, and reporting — through a small set of
task-oriented, callable namespaces.

The guiding principle is to change the interaction model **from "scientific software"
to "ordinary software"**: sensible defaults everywhere, the minimum possible required
input, and internal complexity (measurement sets, per-source splits, calibration-table
bookkeeping, spw maps) hidden from the user unless they ask.

```python
from vlbipy import VLBIObs

obs = VLBIObs(project="RSM07", network="EVN", target="3C286")
obs.import_data()
obs.calibrate()                     # full default chain — or step-by-step below
img = obs.clean(target="3C286", robust=2, imsize=8192)
img.export_fits("3C286.image.fits")
```

The same `VLBIObs` object is the single entry point for the CLI, interactive Python, and
(later) the dashboard. Backends and observatories plug in behind stable interfaces; the
API layer never imports `casatasks`/`casatools`, so a new backend or array can be added
without touching the user-facing surface or orchestration logic.

### 2.1 Scope of this milestone

- **In scope now:** the complete public Python API surface (classes, namespaces, method
  signatures, returned objects, config schema, source model, logging/error model) and a
  **`DummyBackend`** that makes the entire API runnable in-memory — it logs the
  operations it *would* perform, returns realistic synthetic result objects, mutates
  in-memory state, and writes **no files**. This lets us "play with" and iterate on the
  ergonomics before committing to any backend.
- **Deferred (documented, not built now):** real CASA/WSClean/AOFlagger/AIPS calls, real
  archive downloads, the CLI and dashboard implementations (specified here as thin
  wrappers), multi-epoch scientific merge, spectral-line/polarization/multi-phase-center
  modes, and containerization. See §11 Phased Roadmap.

## 3. Goals and Non-Goals

### Goals
- **G1.** A single, discoverable public API centered on `VLBIObs`, importable as
  `from vlbipy import VLBIObs`, usable identically for one project or a campaign of many.
- **G2.** Task-oriented **callable namespaces** (`obs.clean(...)`, `obs.clean.wsclean(...)`,
  `obs.calibrate.bandpass()`, `obs.flag.aoflagger()`, `obs.plot.tplot()`) that read like
  the astronomer's mental model, not like CASA task names.
- **G3.** **Zero backend leakage:** the user never constructs, sees, or calls a backend.
  All CASA/AIPS/WSClean specifics live behind a `Backend` interface.
- **G4.** **Minimum required input:** a project code plus (optionally) a network and a
  target is enough; everything else is defaulted or auto-derived, overridable via TOML or
  keyword arguments.
- **G5.** **Managed parallelism:** campaign-level calls fan out across projects
  automatically; the user writes one call, not an executor loop.
- **G6.** **Rich returned objects** (`Image`, `ImageSet`, `CalTable`, `QualityMetrics`)
  that can be inspected, exported, and fed back into the pipeline (`obs.selfcal(img)`).
- **G7.** A **`DummyBackend`** implementation that exercises the entire API in-memory so
  the design can be validated and unit-tested with no CASA and no data.

### Non-Goals (for this milestone)
- **N1.** Real calibration/imaging correctness — no actual CASA/WSClean/AOFlagger calls.
- **N2.** Real data downloads or file I/O of scientific products.
- **N3.** CLI and dashboard *implementations* (their API-surface contracts are specified).
- **N4.** Multi-epoch astrometry science, spectral-line cubes, polarization, and
  multi-phase-center handling (API shape must not preclude them; behavior is deferred).

## 4. Design Principles

1. **API first, backend later.** The public surface is frozen and fully tested against
   `DummyBackend` before any real backend is wired in.
2. **One object, three surfaces.** `VLBIObs` drives CLI, Python/Jupyter, and dashboard.
3. **Hide the internals.** Measurement sets, per-source splits, `mstransform`, spw maps,
   `gaintable`/`interp` lists, and calibration-table ordering are never required inputs.
4. **Defaults that "just work," overrides that are trivial.** Any default is overridable
   via TOML, per-call keyword, or constructor argument, using identical key names.
5. **Callable namespaces.** A verb group is itself callable (runs the sensible default)
   and also exposes explicit variants: `obs.clean(...)` == default imager;
   `obs.clean.wsclean(...)` / `obs.clean.tclean(...)` == explicit choice.
6. **Regenerable modules.** Flat, explicit code; each module can be rewritten from scratch
   against stable interfaces (per repo `AGENTS.md`).
7. **Explicit, observable behavior.** Every operation logs at its boundary; failures raise
   with full context; nothing is silently skipped.

## 5. Personas and Primary Journeys

- **P1 — Staff astronomer, standard reduction.** Wants a calibrated, imaged EVN continuum
  dataset from a project code with no scripting. Journey: `VLBIObs(...).run()` (or the CLI
  one-liner) → review `report.html`.  Optionally, manual edit the calibration tables and re-run all affected steps after such tables.
- **P2 — Expert, step-by-step control.** Wants to run stages individually, inspect
  diagnostics, hand-edit a calibration table, and re-run only what changed. Journey:
  interactive Python/Jupyter, calling namespace methods, relying on smart re-run.
- **P3 — Multi-epoch/campaign scientist.** Wants several projects reduced in parallel and
  then merged for joint imaging/astrometry. Journey: `VLBIObs(project=[...])` → per-epoch
  calibration → `obs.merge()` → campaign imaging and calibrating together.
- **P4 — Tool/pipeline developer.** Wants to add a backend, observatory, or imager without
  touching orchestration. Journey: implement a `Backend`/`ObservatoryHandler`; the API
  layer is untouched.

### The CLI one-liner (specified; implemented in a later phase)
```bash
vlbipy -p RSM07 --network EVN --target 3C286
```
is exactly equivalent to:
```python
VLBIObs(project="RSM07", network="EVN", target="3C286").run()
```

## 6. API Architecture

### 6.1 Object model

| Object | Role | User-facing? |
|---|---|---|
| **`VLBIObs`** | **The single public entry point.** A *campaign* of 1..N projects. Exposes every operation namespace and fans out across its observations. | **Yes — the only class users normally construct.** |
| `Observation` | The internal per-project unit (one project code / one MS lineage). Holds that project's sources, antennas, frequency setup, scans, calibration state. | Accessible (`obs["RSM07"]`, `obs.observations`) for power users; not the recommended constructor. |
| `SourceSet` | Role-aware collection of `Source` objects with attribute accessors. | Yes (`obs.sources`). |
| `Image` / `ImageSet` | Returned imaging products with stats and export methods. | Yes (returned by `obs.clean(...)`). |
| `CalTable` | A produced calibration product (type, field, interp, synthetic SNR). | Yes (returned by calibrate namespace; usually ignored). |
| `QualityMetrics` | Peak, rms, dynamic range, integrated flux, beam. | Yes (`img.stats`). |
| `Backend` / `ObservatoryHandler` | Hidden implementation interfaces. | **No.** |

**Key semantics:**
- `VLBIObs` given **one** project is a campaign-of-one and behaves exactly like a single
  observation. `merge()` is a no-op in that case.
- `VLBIObs` given **many** projects fans out every step across its `Observation` members
  (managed parallelism, §6.7) and `merge()` performs the cross-project/epoch combination.
- Every operation namespace exists on `VLBIObs` and delegates to each contained
  `Observation`; the two classes share the namespace surface (§6.3).

### 6.2 Construction and configuration

```python
VLBIObs(
    project: str | list[str],          # one code or many; many -> a campaign
    network: str | None = None,        # "EVN" | "VLBA" | "LBA"; may be per-project inferred
    backend: str = "dummy",            # "dummy" now; "casa" later. Never a backend object.
    target: str | list[str] = (),      # convenience role declaration
    phasecal: str | list[str] = (),
    fringe_finder: str | list[str] = (),
    check_source: str | list[str] = (),
    refant: str | list[str] = (),      # preferred reference antenna(s); auto-ranked if empty
    mode: str = "continuum",           # continuum | spectral-line | pulsar-binning | multi-phase-center
    config: str | Path | dict | None = None,  # TOML path or dict overrides
    work_dir: str | Path | None = None,
    **overrides,                       # any defaults.toml key, dotted (e.g. calibration_sbd_minsnr=...)
)

VLBIObs.from_config("campaign.toml")   # everything (incl. project list) from TOML
```

Configuration keeps the **three-layer TOML cascade** (built-in `defaults.toml` → user
file → constructor/CLI overrides; later layers win). Constructor keyword names mirror the
config keys, so anything expressible in TOML is expressible in Python and on the CLI.

### 6.3 Operation namespaces (the callable-namespace pattern)

Each namespace is an attribute of `VLBIObs`/`Observation`. Calling the namespace runs its
**default action**; its methods run explicit variants. All namespace methods accept a
`force: bool = False` keyword (bypass smart-skip, §6.8) and operation-specific parameters
that default from config.

| Namespace | Call (default) | Notable methods |
|---|---|---|
| `obs.import_data(...)` | download (if supported) + import to MS + per-source split | `import_data.from_fitsidi(...)`, `import_data.from_uvfits(...)`, `import_data.from_ms(...)` |
| `obs.calibrate(...)` | full default calibration chain (§6.3.1) | `.a_priori_gain_calibration()`, `.bandpass()`, `.fringefit()` (called also from `.sbd()`, and `.mbd()`), `.ionospheric_correction()`, `.apply()` |
| `obs.flag(...)` | default flag chain (autocorr → edges → quack → auto-flagger on calibrators) | `a_priori_flagging()`, `.autocorr()`, `.edges()`, `.quack()`, `.tfcrop()`, `.aoflagger()`, `.from_file()`, `.manual(...)`, `.unflag()` |
| `obs.plot(...)` | produce the standard diagnostic set | `.tplot()`, `.uvplot()`, `.elevation()`, `.amptime()`, `.phasetime()`, `anptime()`, `.autocorr()`, `.ampfreq()`, `.phasefreq()`, `anpfreq()`, `.caltable(...)` |
| `obs.clean(...)` | image with the default imager (WSClean) → `Image`/`ImageSet` | `.wsclean(...)`, `.clean(...)` (that would call either `.tclean()` in UNIX or `.iclean()` in macOS) |
| obs.image(...) | Plots the image (Image/ImageSet) with the default tool (Carta) | `.cleaned()`, `.dirty()`, `.residuals()`, `.beam()`  |
| `obs.selfcal(...)` | phase-only then amp+phase self-cal, convergence-checked | `.phase()`, `.ampphase()` |
| `obs.export(...)` | per-source split MS + UVFITS | `.uvfits(...)`, `.ms(...)` |

Top-level (non-namespace) methods on `VLBIObs`/`Observation`:
`summary()` / `inspect()`, `report()`, `merge()`, `run()`, `reset()` / `clean_state()`,
and read-only accessors (§6.6, §6.9).

#### 6.3.1 The default calibration chain

`obs.calibrate()` runs, in order (mirroring the validated pipeline):
* `a_priori` (Tsys, gain curve, TEC/EOP/ACCOR as applicable) to all sources.
* `initial_calibration` (single-band delay, SBD) → `bandpass` → refined SBD on the fringe finder sources (if they exist and all antennas observed at least one scan of them, otherwise in the phase calibrator sources if they exist, otherwise directly on the targets).
* Continue only on the fringe finders: `fringefit` (global/multi-band delay, MBD, smoothed) → post-calibration flagging on the selected sources → a second calibration pass with scalar amplitude normalization →  transfer the SBDs, bandpasses and scalar amplitude normalizations to the other sources (phase calibrators and target sources). In the flagging, estimate the edge channels that should be flagged (from the ones deviating from the average amplitudes and phases), and flag them to all data (all sources).
* Calibrate the phase calibrators (if they exist): MBD, bandpass. Flag the data and re-run this calibration. Transfer the solutions to the target sources. If multiple phase calibrators are associated to a specific target source, then the solutions between all calibrators should be interpolated to be applied to the target.
*


### 6.4 Imaging and returned objects

```python
img  = obs.clean(target="3C286", robust=2, imsize=8192)          # default imager -> Image
img  = obs.clean.wsclean(target=obs.sources.target, robust=-2)   # explicit imager
img  = obs.clean.tclean(target="3C286", robust=0, weighting="briggs")
imgs = obs.clean(target=obs.sources.target, robust=[-2, -1, 0, 1, 2])       # a sweep -> ImageSet
```

- `target` accepts a source name (`str`), a `Source`, or a role accessor
  (`obs.sources.target`). Imaging parameters (`cell`, `imsize`, `weighting`, `robust`,
  `niter`, `threshold`) default from the array geometry/frequency and `[imaging]` config.
- A scalar `robust` returns one `Image`; a list returns an `ImageSet`.

**`Image`** (returned):
- Attributes: `source`, `robust`, `weighting`, `paths` (image/residual/psf/model/fits/png),
  `stats: QualityMetrics`.
- Methods: `export_fits(outfile)`, `preview()` (PNG path), `open()` (viewer hook, later).

**`ImageSet`**: iterable of `Image`; `set[robust]`, `set.best()` (by dynamic range),
`set.by_robust` (dict), `set.export_all(dir)`.

**`QualityMetrics`**: `peak`, `rms`, `dynamic_range`, `snr`, `integrated_flux`,
`beam` (`bmaj`, `bmin`, `bpa`).

### 6.5 Self-calibration and export

```python
obs.selfcal(img)    # self-cal the source's MS using img's model
obs.selfcal("3C286", phase_rounds=4, ampphase_rounds=5)
result = obs.selfcal(obs.sources.phase_calibrators)   # -> SelfcalResult (rounds, DR gain, accepted/discarded)

obs.export.uvfits(source="3C286", time_average="4s", channel_average=1)
obs.export.ms(source="3C286")
img.export_fits("3C286.image.fits")
```

`selfcal` accepts an `Image`, a source name/`Source`, or a role accessor. Rounds that
reduce peak flux or raise noise are automatically discarded (convergence via dynamic-range
improvement threshold, from `[selfcal]` config).

### 6.6 Sources and phase referencing (config roles + accessors)

Roles are declared in config (or via constructor convenience kwargs) and read back through
`obs.sources`:

```python
obs.sources                     # SourceSet
obs.sources.target              # list[Source]
obs.sources.phasecal            # list[Source]
obs.sources.fringe_finder       # list[Source]
obs.sources.check_source        # list[Source]
obs.sources.calibrators         # phase cals + fringe finders
obs.sources.ampcals             # eMERLIN amplitude calibrator (3C286)
obs.sources["3C286"]             # by name
for s in obs.sources: ...        # iteration
```

Calibrator→target relationships (phase referencing) are declared in TOML:

```toml
[sources]
targets = ["3C286", "R20181030"]
phase_calibrators = ["J1048+7143", "J1027+7428"]
fringe_finders = ["3C345"]
check_sources = ["J1041+..."]

[phase_referencing]
"3C286"     = "J1048+7143"                    # direct mapping
"R20181030" = ["J1027+7428", "J1048+7143"]    # interpolation across multiple calibrators
```

**Empty-role fallback rule** (applied automatically and logged): a stage needing a role
that has no sources falls back through fringe finder → phase calibrator → target. E.g. with
no fringe finder, bandpass/instrumental steps use a phase calibrator; with neither, the
target itself.

### 6.7 Campaign: managed parallelism and merge

```python
obs = VLBIObs(project=["RSM07", "RSM08", "RSM09"], network="EVN", target="3C286")

obs.import_data()      # fans out across the 3 observations, run concurrently (managed)
obs.calibrate()        # each observation calibrated independently, in parallel
obs.merge()            # AFTER calibration: concatenate per-source across projects/epochs
img = obs.clean(target="3C286", robust=0)   # image the merged, campaign-level data

obs.observations       # list[Observation]
obs["RSM07"]           # a single Observation (same namespaces, acts on that project only)
obs.map(fn)            # advanced: apply a callable to each Observation
```

- Fan-out is **managed**: campaign-level namespace calls apply to every contained
  observation concurrently; the user never writes an executor loop. The concurrency
  primitive (threads/processes/Dask) is an implementation detail (see Open Questions); the
  `DummyBackend` runs sequentially but preserves call/return semantics.
- `merge()` is only meaningful after calibration. For a one-project `VLBIObs` it is a
  logged no-op. Pre-calibration steps (`import_data`, `calibrate`, `flag`) run per project;
  post-merge steps (`clean`, `export`, `report`) operate on the merged product.

### 6.8 State and smart re-run

- Each `Observation` owns a **state record** (`obs["CODE"].state`) of completed steps with
  timestamps and input/output tracking. `VLBIObs.state` aggregates across observations.
- Every namespace method consults state before running: if outputs are newer than inputs it
  **skips** (logged); a `force=True` keyword forces re-run; editing/deleting an intermediate
  invalidates and re-runs downstream steps.
- `obs.reset()` / `obs.clean_state()` clears state for a fresh start.
- Under `DummyBackend`, there is no real filesystem product; the state machine still runs
  and **logs its decisions** (e.g. `would skip: bandpass (up-to-date)`), so the smart-rerun
  logic is exercised and testable without CASA.

### 6.9 Backend abstraction and the `DummyBackend` contract

- A single cohesive **`Backend`** interface defines every low-level operation the namespaces
  need (import, metadata read, a-priori cal, fringefit, bandpass, gaincal, applycal, flag
  operations, clean/wsclean/tclean, split, export_uvfits). Real backends may internally
  compose sub-handlers, but the API layer sees one `Backend`.
- `VLBIObs`/`Observation` hold a `Backend` instance chosen by the `backend=` argument;
  **users never touch it.** This is the deliberate departure from the previous design, where
  `project.cal.fringefit(datafile=..., caltable=..., refant=..., solint=...)` exposed the
  backend and low-level paths directly.
- **`DummyBackend` (the only implementation in this milestone):**
  - Writes **no files** and requires no external software or data.
  - Logs each operation with its resolved parameters and data selection
    (e.g. `[dummy] fringefit field=3C345 refant=EF solint=inf combine=spw`).
  - Returns realistic synthetic result objects (`CalTable`, `Image`, `ImageSet`,
    `QualityMetrics`) and synthetic metadata (a configurable fake array/sources/scans/freq
    setup) so `summary()`, `sources`, plotting, and imaging all return sensible values.
  - Mutates in-memory state so smart re-run, role resolution, and merge behave correctly.

### 6.10 Observatory abstraction

An **`ObservatoryHandler`** interface owns everything that differs by array: auto-download
capability and URLs, a-priori file formats (`.antab`, `.uvflg`), and EOP/ACCOR/TEC
applicability. `VLBIObs` selects a handler from `network=`. For this milestone, handlers are
stubs: EVN reports auto-download-capable; VLBA/LBA report manual-download with clear guidance
(VLBA: NRAO has no automated download; LBA: retrieve from ATOA plus `.antab`/`.uvflg`
instructions). The `DummyBackend` synthesizes "found" files so `import_data` completes.

### 6.11 Logging and error model

- Structured `INFO`/`WARNING`/`ERROR`/`DEBUG` logging at every operation boundary, to console
  and (later, with real products) a per-run log file. Terminal styling: **orange** warnings,
  **red bold** errors; concise progress `INFO` otherwise.
- Prominent, real-time surfacing of anomalies (e.g. unusually high flagged fraction), plus an
  end-of-run summary of all warnings (also embedded in the report).
- Explicit exception hierarchy with context: `VlbipyError` (base), `ConfigError`,
  `SourceNotFoundError`, `BackendError`, `StepError` (records step, parameters, data
  selection). Failed steps are marked failed in state, never silently skipped.

### 6.12 CLI and dashboard (thin wrappers — specified, implemented later)

- **CLI:** `vlbipy` maps flags to `VLBIObs` construction and calls `.run()` (or a
  `--summary-only`, `--dashboard`, `--from-step/--to-step/--step`, `--force`, `--clean`,
  `--no-smart` mode). It adds **no** logic beyond argument translation.
- **Dashboard:** launched via `vlbipy --dashboard -p PROJECT`; reads the same `VLBIObs`
  state/products. Interactivity (interactive flagging, live re-runs, CARTA) is a later phase.
- Both are contract-specified here so the API is shaped to support them, but neither is
  implemented in this milestone.

## 7. End-to-End Examples

### 7.1 Single project, step-by-step (matches the Pre-PRD example line-for-line)
```python
from vlbipy import VLBIObs

obs = VLBIObs(project="RSM07", network="EVN", target="3C286")
obs.import_data()
obs.calibrate.initial_calibration()
obs.calibrate.bandpass()
obs.flag.aoflagger()
obs.plot.tplot()
img = obs.clean(target="3C286", robust=2, imsize=8192)
img = obs.clean.wsclean(target=obs.sources.target, robust=-2, imsize=8192)
obs.selfcal(img)                     # self-calibrates the source's data using img's model
img.export_fits(outfile="3C286.image.fits")
```

### 7.2 Single project, fully automated
```python
obs = VLBIObs(project="RSM07", network="EVN", target="3C286")
obs.run()                            # import -> calibrate -> flag -> image -> selfcal -> export -> report
print(obs.summary())
```

### 7.3 Campaign of many projects
```python
obs = VLBIObs(project=["RSM07", "RSM08", "RSM09"], network="EVN", target="3C286")
obs.import_data()
obs.calibrate()                      # parallel per-project
obs.merge()                          # cross-epoch concatenation per source
imgs = obs.clean(target="3C286", robust=[-2, -1, 0, 1, 2])   # ImageSet on merged data
imgs.best().export_fits("3C286.campaign.fits")
obs.report()
```

## 8. Module Design

Fresh package under `src/vlbipy/` (import name unchanged: `vlbipy`). Flat, regenerable
modules with stable interfaces.

- **`__init__.py`** — public exports: `VLBIObs`, `load_config`, `__version__`.
  (`Observation` and result types importable for power users/type hints.)
- **`vlbiobs.py`** — `VLBIObs`: construction, config resolution, observation fan-out,
  `merge()`, `run()`, aggregate state; hosts the namespace objects and delegates to members.
- **`observation.py`** — `Observation`: per-project unit; holds backend/observatory refs,
  resolved metadata, per-project state; the object namespaces actually act on.
- **`namespaces/`** — one module per callable namespace (`calibrate.py`, `flag.py`,
  `plot.py`, `clean.py`, `selfcal.py`, `export.py`, `import_data.py`). Each defines a
  callable object whose `__call__` runs the default and whose methods run variants; all
  translate high-level intent into `Backend` calls. No CASA imports.
- **`sources.py`** — `Source`, `SourceType`, `SourceSet` (role accessors, name lookup,
  iteration), phase-referencing map resolution, empty-role fallback.
- **`metadata.py`** — domain dataclasses: `Antenna`, `FreqSetup`, `Scan`, `ObsMetadata`,
  `CalTable`, `QualityMetrics`, plus enums (`Stokes`, `Observatory`, `Backend`, `Mode`).
- **`images.py`** — `Image`, `ImageSet`, and their methods.
- **`backends/base.py`** — the `Backend` interface (ABC/Protocol).
- **`backends/dummy.py`** — `DummyBackend` (in-memory + logging; synthetic metadata/results).
- **`observatories/base.py`** + `evn.py`/`vlba.py`/`lba.py` — `ObservatoryHandler` interface
  and stub handlers.
- **`config.py`** — three-layer TOML cascade; constructor/CLI override merge; role parsing.
- **`state.py`** — step/state tracking, smart-skip decisions, downstream invalidation
  (logs decisions in dummy mode).
- **`logging_utils.py`** — colored/structured logging setup; warning aggregation.
- **`errors.py`** — exception hierarchy.
- **`cli.py`** — argument translation → `VLBIObs` (thin; implemented in a later phase).
- **`dashboard.py`** — dashboard entry (thin; implemented in a later phase).
- **`templates/defaults.toml`** — built-in defaults (carried forward and extended with a
  `[phase_referencing]` section and `backend = "dummy"` default for this milestone).

## 9. Data Models (summary)

- **`Source`** — `name`, `coordinates` (astropy `SkyCoord`), `source_type`, `separation(other)`.
- **`SourceSet`** — role accessors (`target`, `targets`, `phase_calibrators`,
  `fringe_finders`, `check_sources`, `polarization_calibrators`, `calibrators`), `__getitem__`
  by name, `__iter__`, phase-referencing resolution.
- **`Antenna`** — `name`, `fullname`, `diameter`, `position` (ITRF x,y,z), `observed`,
  `subbands`.
- **`FreqSetup`** — `ref_freq`, `total_bandwidth`, `n_subbands`, `n_channels`,
  `channel_width`, `polarizations`; `freq_ghz`, `bandwidth_mhz`.
- **`Scan`** — `scan_number`, `source`, `time_start`, `time_end`, `antennas`; `duration_sec`.
- **`ObsMetadata`** — per-observation aggregate (`project_code`, `obs_date`, `time_range`,
  `sources`, `antennas`, `scans`, `freq_setup`).
- **`CalTable`** — `cal_type`, `path`, `field`, `interp`, `spwmap`, `snr` (synthetic in
  dummy mode).
- **`Image` / `ImageSet` / `QualityMetrics`** — as in §6.4–6.5.

All are plain, serializable dataclasses with no backend dependencies.

## 10. Testing Strategy

- **Contract tests against `DummyBackend` (no CASA, no data) — the core of this milestone.**
  Exercise the whole public surface: construct `VLBIObs` (single and multi-project); call
  every namespace default and variant; assert returned types/shapes (`clean` scalar→`Image`,
  list→`ImageSet`; `img.stats` is a `QualityMetrics`; `selfcal`→`SelfcalResult`); assert
  role resolution and empty-role fallback; assert `merge()` is a no-op for one project and
  combines for many; assert smart-skip/`force`/downstream-invalidation decisions are logged;
  assert operations are logged with resolved parameters.
- **Pure-Python unit tests:** config cascade and override precedence, source-role and
  phase-referencing parsing, state machine transitions, model (de)serialization.
- **Example-coverage test:** the Pre-PRD example (§7.1) executes end-to-end under
  `DummyBackend` without error, guaranteeing the documented ergonomics stay valid.
- Real-backend integration testing (CASA against a real dataset such as `rsm07`) is defined
  with the Phase 1 backend work, not here.

## 11. Phased Roadmap

- **Phase 0 — Dummy API (this design's implementation target; follow-up to this PRD).**
  Full public API + `DummyBackend` + config/source/state/logging modules + contract tests.
  No CASA, no files. Deliverable: a `vlbipy` you can `pip install` and "play with."
- **Phase 1 — CASA backend, single-epoch EVN continuum.** Implement `Backend` for CASA
  (import, a-priori incl. Tsys filtering + TEC, SBD→BP→refined-SBD→MBD, post-cal flagging +
  scalar amplitude normalization, phase referencing, self-cal, imaging via WSClean with
  tclean fallback, split/export, HTML+JSON report). Validate against `rsm07`.
- **Phase 2 — CLI + dashboard + notebook export.** Implement the thin CLI/dashboard wrappers;
  wire dashboard tabs to real products; generate `{project}_reduction.ipynb`.
- **Phase 3 — Multi-epoch campaign science.** Real cross-epoch merge, joint self-cal, apply
  combined model per epoch, astrometry (proper motion/parallax) reporting.
- **Phase 4 — VLBA/LBA automatic downloads.** Replace manual-download guidance with real
  `data.nrao.edu`/ATOA integration.
- **Phase 5 — Spectral-line and multi-phase-center/pulsar-binning modes.**
- **Phase 6 — Polarization calibration** (RL delay, RL phase, D-terms).
- **Phase 7 — Containerization** (Singularity image; `oxkat` as reference model only).
- Backends behind the same interface (AIPS, eht-imager, Difmap export) can be added in any
  phase without API changes.

## 12. Out of Scope (this milestone)

- Real CASA/AIPS/WSClean/AOFlagger task execution and any real scientific correctness.
- Real archive downloads and real product file I/O.
- CLI and dashboard *implementations* (contracts only).
- Multi-epoch astrometry science, spectral-line cubes, multi-phase-center/pulsar-binning,
  polarization calibration, MultiView 2D calibrator-plane interpolation.
- Containerization/packaging.

## 13. Open Questions

- **`obs.sources.target` when multiple targets exist.** Return the first, raise, or require
  `obs.sources.targets`? Implement: only `.target` will exist. Same as for other source types. They can be either a string | Source if single source exists, or list[str], list[Source] if multiple exists.
- **Campaign concurrency primitive.** Threads vs processes vs Dask for real fan-out (CASA is
  not thread-safe and is process-heavy). Decision: use multiple processes for multiple observations to run in parallel; do in serial for single epochs.
- **Merge semantics detail.** Exact per-source cross-epoch concatenation model (and whether
  frequency/uv alignment is required) is deferred to Phase 3; shape the `merge()` signature
  to accept it. Owner: user + maintainer.
- **Default imager.** WSClean by default with tclean fallback (per Pre-PRD and prior spec).
- **`VLBIObs` naming semantics.** The public class represents a campaign of 1..N projects but
  is named for a single observation (per user preference and the Pre-PRD). Documented as an
  intentional convention; revisit only if it confuses users.
- **Namespace verb naming.** Standardized on snake_case (`import_data`,
  `initial_calibration`); confirm no preferred aliases (e.g. `importdata`). Owner: user.

## 14. Further Notes

- The previous `Project`-centric code (`project.py`, `pipeline.py`, `backends/casa/*`,
  `runner.py`, etc.) is **reference material only** and is superseded by this design. Useful
  validated logic (refant SNR selection, Tsys filtering, scan/solint selection, TEC wiring)
  should be ported *behind the new `Backend` interface* during Phase 1 rather than reused as-is.
- `INSTRUCTIONS_CLAUDE.md` remains the full-detail source spec (exact CASA signatures,
  parameter defaults) to consult when implementing real backends in Phase 1+.
- Per repo `AGENTS.md`/`CLAUDE.md`: prefer full-file rewrites over micro-edits, keep
  parameters/elements on one line unless they exceed 120 characters, and document every
  function's current inputs/outputs.
