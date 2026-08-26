# vlbipy — Epic Backlog

Epic-level breakdown of [`PRD.md`](./PRD.md), spanning the full roadmap (Phases 0–7).
Each `##` heading is one epic (a self-contained chunk of work), intentionally kept
coarse-grained; fine-grained tasks are expected to be split out per epic when work
begins. Every epic is written to be copy-pasteable into GitHub Issues / Linear later.

**How to use:** implement top-to-bottom by priority. **Phase 0 is the active target**
(the playable dummy API); Phases 1–7 are sequenced but not yet scheduled.

**Legend** — Priority: `P0` = now, `P1` = next, `P2` = later, `P3` = future.
Status: `todo` unless noted.

## Overview

 | ID   | Epic                                            | Phase   | Priority   | Depends on       |
 | ---- | ------                                          | ------- | ---------- | ------------     |
 | E0.1 | Package foundation & domain models              | 0       | P0         | —                |
 | E0.2 | Core objects & callable namespaces              | 0       | P0         | E0.1             |
 | E0.3 | DummyBackend & observatory stubs                | 0       | P0         | E0.1             |
 | E0.4 | Config, state, logging & errors                 | 0       | P0         | E0.1             |
 | E0.5 | Contract tests & example validation             | 0       | P0         | E0.2, E0.3, E0.4 |
 | E1   | CASA backend — single-epoch EVN continuum       | 1       | P1         | E0.*             |
 | E2   | CLI, dashboard & notebook export                | 2       | P2         | E1               |
 | E3   | Multi-epoch campaign science                    | 3       | P2         | E1               |
 | E4   | VLBA/LBA automatic downloads                    | 4       | P2         | E1               |
 | E5   | Spectral-line & multi-phase-center/pulsar modes | 5       | P3         | E1               |
 | E6   | Polarization calibration                        | 6       | P3         | E1               |
 | E7   | Containerization                                | 7       | P3         | E1, E3           |
 | E8   | Additional backends (AIPS, eht-imager, Difmap)  | any     | P3         | E0.3             |

---

# Phase 0 — Playable Dummy API (active target)

Goal: a fresh `src/vlbipy` package exposing the **complete public API** backed by a
`DummyBackend` (in-memory + logging only, **no files written**), so the API can be
exercised, tested, and iterated on before any real backend exists. PRD §2.1, §6, §10.

## E0.1 — Package foundation & domain models
**Phase 0 · P0 · depends: —**

Stand up the fresh package skeleton and all backend-independent data types.

**Scope**
- New `src/vlbipy` package (import name unchanged); `pyproject.toml`; `__init__.py`
  exporting `VLBIObs`, `load_config`, `__version__` (plus `Observation` and result
  types for power users / type hints).
- `models.py`: `Antenna`, `FreqSetup`, `Scan`, `ObsMetadata`, `CalTable`,
  `QualityMetrics`, and enums (`Stokes`, `Observatory`, `Backend`, `Mode`).
- `sources.py`: `Source`, `SourceType`, `SourceSet` (role accessors `target`/`targets`/
  `phase_calibrators`/`fringe_finders`/`check_sources`/`polarization_calibrators`/
  `calibrators`, name lookup, iteration), phase-referencing resolution, empty-role
  fallback (fringe finder → phase cal → target).
- `results.py`: `Image`, `ImageSet`, `SelfcalResult` and their methods
  (`export_fits`, `preview`, `best`, `by_robust`, `export_all`).

**Acceptance criteria**
- All dataclasses import, construct, and round-trip (de)serialize with no CASA/backend imports.
- `SourceSet` resolves roles and phase-ref maps; ambiguous `.target` behavior matches the
  decision recorded in PRD §13 (Open Questions).

**PRD refs:** §6.1, §6.4, §6.6, §8, §9.

## E0.2 — Core objects & callable namespaces
**Phase 0 · P0 · depends: E0.1**

Implement the user-facing objects and the callable-namespace machinery.

**Scope**
- `vlbiobs.py`: `VLBIObs` — construction (single code or list), config resolution,
  observation fan-out, `merge()`, `run()`, aggregate state, `observations`, `__getitem__`,
  `map()`.
- `observation.py`: `Observation` — per-project unit holding backend/observatory refs,
  resolved metadata, per-project state.
- `namespaces/`: one module each for `import_data`, `calibrate`, `flag`, `plot`, `clean`,
  `selfcal`, `export`. Each is a **callable** whose `__call__` runs the default and whose
  methods run explicit variants (e.g. `clean.wsclean` / `clean.tclean`,
  `calibrate.a_priori/initial_calibration/bandpass/fringefit/phase_reference/apply`).
  Namespaces translate intent into `Backend` calls only — no CASA imports.
- Top-level methods: `summary()`/`inspect()`, `report()`, `reset()`/`clean_state()`.

**Acceptance criteria**
- Single- and multi-project construction works; campaign-level calls delegate to each
  `Observation`; `merge()` is a logged no-op for one project.
- Every namespace default and variant is callable and returns the documented object types
  (`clean` scalar robust → `Image`, list → `ImageSet`; `selfcal` → `SelfcalResult`).

**PRD refs:** §6.1–6.7.

## E0.3 — DummyBackend & observatory stubs
**Phase 0 · P0 · depends: E0.1**

Provide the only backend for this milestone plus observatory handler stubs.

**Scope**
- `backends/base.py`: single cohesive `Backend` interface (import, metadata read,
  a-priori cal, fringefit, bandpass, gaincal, applycal, flag ops, clean/wsclean/tclean,
  split, export_uvfits).
- `backends/dummy.py`: `DummyBackend` — writes no files; logs each op with resolved
  parameters and data selection; returns synthetic `CalTable`/`Image`/`ImageSet`/
  `QualityMetrics`; synthesizes configurable fake array/sources/scans/freq setup; mutates
  in-memory state.
- `observatories/base.py` + `evn.py`/`vlba.py`/`lba.py`: `ObservatoryHandler` interface and
  stub handlers (EVN = auto-download-capable; VLBA/LBA = clear manual-download guidance;
  dummy "found" files so `import_data` completes).

**Acceptance criteria**
- The full API runs end-to-end with `backend="dummy"` and no external software/data/files.
- Backend is never exposed to users (no public attribute returns a backend instance).

**PRD refs:** §6.9, §6.10.

## E0.4 — Config, state, logging & errors
**Phase 0 · P0 · depends: E0.1**

The cross-cutting infrastructure the API relies on.

**Scope**
- `config.py`: three-layer TOML cascade (defaults → user file → constructor/CLI overrides);
  constructor-kwarg ↔ config-key mirroring; role and `[phase_referencing]` parsing.
- `templates/defaults.toml`: carried forward from the old package and extended with a
  `[phase_referencing]` section and `backend = "dummy"` default.
- `state.py`: per-observation step/state record, smart-skip decisions, `force=`,
  downstream invalidation; in dummy mode it **logs decisions** (e.g. `would skip: bandpass`).
- `logging_utils.py`: structured `INFO`/`WARNING`/`ERROR`/`DEBUG`; **orange** warnings,
  **red bold** errors; anomaly surfacing + end-of-run warning summary.
- `errors.py`: `VlbipyError`, `ConfigError`, `SourceNotFoundError`, `BackendError`,
  `StepError` (records step/params/selection).

**Acceptance criteria**
- Override precedence verified (CLI/kwargs > user TOML > defaults).
- Smart-skip / force / invalidation decisions are observable via logs; failed steps are
  marked failed in state, never silently skipped.

**PRD refs:** §6.2, §6.8, §6.11.

## E0.5 — Contract tests & example validation
**Phase 0 · P0 · depends: E0.2, E0.3, E0.4**

Freeze the public API behavior with tests that need no CASA and no data.

**Scope**
- Contract suite against `DummyBackend`: construct `VLBIObs` (single + multi-project);
  call every namespace default and variant; assert returned types/shapes; assert role
  resolution + empty-role fallback; assert `merge()` no-op vs combine; assert
  smart-skip/`force`/invalidation are logged; assert ops log resolved parameters.
- Pure-Python unit tests: config cascade/precedence, source-role & phase-ref parsing,
  state transitions, model (de)serialization.
- Example-coverage test: the Pre-PRD example (PRD §7.1) runs end-to-end without error.

**Acceptance criteria**
- `pytest` green in a CASA-free environment; the §7.1 example is a passing test.

**PRD refs:** §10.

---

# Phases 1–7 — Real backends & advanced modes (sequenced, not yet scheduled)

## E1 — CASA backend: single-epoch EVN continuum
**Phase 1 · P1 · depends: E0.***

Implement `Backend` for CASA behind the frozen interface and deliver a real end-to-end
EVN continuum reduction, validated against `rsm07`.

**Scope**
- Import: FITS-IDI → MS, per-source split, Dask-friendly I/O; skip-if-exists.
- A-priori: Tsys (+ outlier filtering/smoothing), gain curve, TEC (freq-gated), EOP/ACCOR
  where applicable; warn on antennas missing Tsys where they have data.
- Main calibration chain: SBD → bandpass → refined SBD → global MBD (smoothed), with
  data-driven refant-by-SNR selection, SBD/bandpass scan selection by fringe SNR, and MBD
  solint optimization (port validated logic from the old package **behind the interface**).
- Post-cal flagging (edge channels; AOFlagger/tfcrop on calibrators only) + scalar
  amplitude normalization + second calibration pass.
- Phase referencing (time-interpolated calibrator solutions, derived spwmap) + self-cal
  loop (phase → amp+phase, convergence via dynamic-range improvement, bad-round discard).
- Imaging: WSClean default with tclean fallback; auto cell/imsize/weighting from geometry;
  robust sweep; wide-FOV search for uncertain target positions; image products
  (clean/residual/psf/model, FITS, PNG, stats).
- Split/export (per-source MS + UVFITS, averaging + smearing-loss warning); real working
  directory layout; real state (mtime/checksum); HTML + JSON report with anomaly summary.

**Acceptance criteria**
- End-to-end `rsm07` reduction produces calibrated data + images + report; the public API
  and CLI one-liner are unchanged from Phase 0.

**PRD refs:** §2.1, §6.3.1, §11 (Phase 1), §14 (porting notes).

## E2 — CLI, dashboard & notebook export
**Phase 2 · P2 · depends: E1**

Deliver the non-Python surfaces as thin wrappers over `VLBIObs`.

**Scope**
- CLI (`vlbipy`): flags → `VLBIObs` construction → `.run()`; modes `--summary-only`,
  `--dashboard`, `--from-step/--to-step/--step`, `--force`, `--clean`, `--no-smart`,
  `--verbose`. No logic beyond argument translation.
- Dashboard (`vlbipy --dashboard -p PROJECT`): reads the same `VLBIObs` state/products;
  interactive flagging + live smart re-runs; calibrated-data and results tabs.
- Notebook export: generate `{project}_reduction.ipynb` reproducing the reduction.

**Acceptance criteria**
- CLI one-liner equals the documented Python one-liner; dashboard renders real products.

**PRD refs:** §5, §6.12, §11 (Phase 2).

## E3 — Multi-epoch campaign science
**Phase 3 · P2 · depends: E1**

Make `merge()` scientifically real and add astrometry reporting.

**Scope**
- Real cross-epoch, per-source concatenation (with frequency/uv alignment handling).
- Joint self-cal across epochs; apply combined model back to each epoch.
- Check-source astrometric offset measurement (mas) and proper-motion/parallax reporting.

**Acceptance criteria**
- A multi-project `VLBIObs` calibrates per epoch, merges, and images the combined data;
  astrometric metrics appear in the report.

**PRD refs:** §6.7, §11 (Phase 3); resolves the merge-semantics Open Question (§13).

## E4 — VLBA/LBA automatic downloads
**Phase 4 · P2 · depends: E1**

Replace manual-download guidance with real archive integration.

**Scope**
- `data.nrao.edu` (VLBA) and ATOA (LBA) download flows in the respective
  `ObservatoryHandler`s, including `.antab`/`.uvflg` retrieval.

**Acceptance criteria**
- VLBA/LBA `import_data()` fetches data automatically where the archive allows; clear,
  actionable messaging otherwise.

**PRD refs:** §6.10, §11 (Phase 4).

## E5 — Spectral-line & multi-phase-center/pulsar-binning modes
**Phase 5 · P3 · depends: E1**

Support non-continuum observing modes.

**Scope**
- Channel-dependent calibration; spectral cube production + cube plotting.
- Detection/handling of multiple correlation passes (multi-phase-center, pulsar binning).
- S/X frequency splitting (e.g. ~2.3 GHz vs ~8.7 GHz) prior to imaging.

**Acceptance criteria**
- `mode="spectral-line"` / `"multi-phase-center"` / `"pulsar-binning"` produce the
  corresponding products through the same API.

**PRD refs:** §11 (Phase 5).

## E6 — Polarization calibration
**Phase 6 · P3 · depends: E1**

Add full polarization calibration.

**Scope**
- RL delay, RL phase (`polcal`), D-terms, applied between global fringe fit and final split;
  wire the `polarization_calibrators` role.

**Acceptance criteria**
- Polarization products validated against a known polarized calibrator.

**PRD refs:** §11 (Phase 6).

## E7 — Containerization
**Phase 7 · P3 · depends: E1, E3**

Package `vlbipy` for reproducible deployment.

**Scope**
- Singularity/Apptainer image bundling `vlbipy` + backends (`oxkat` as reference model only).

**Acceptance criteria**
- A container runs a full reduction reproducibly from a project code.

**PRD refs:** §11 (Phase 7).

## E8 — Additional backends (AIPS, eht-imager, Difmap)
**Any phase · P3 · depends: E0.3**

Prove the `Backend` interface generalizes beyond CASA/WSClean.

**Scope**
- Additional `Backend` implementations (AIPS calibration; eht-imager imaging; Difmap
  export), each selectable via `backend=` with no API-surface changes.

**Acceptance criteria**
- At least one additional backend produces valid products through the unchanged API.

**PRD refs:** §6.9, §11 (backend expansion).

---

## Open decisions (from PRD §13) — resolve as the relevant epics start

- `obs.sources.target` when multiple targets exist — proposed: `.target` raises if
  ambiguous, `.targets` is the plural accessor. *(E0.1)*
- Campaign concurrency primitive (threads/processes/Dask; CASA is not thread-safe) —
  design fan-out now, pick executor in Phase 1. *(E0.2 shape, E1 executor)*
- Merge semantics detail (per-source cross-epoch concatenation, freq/uv alignment). *(E3)*
- Default imager = WSClean with tclean fallback — confirm across networks/modes. *(E1)*
- `VLBIObs` naming semantics (campaign named for a single observation) — documented
  convention. *(E0.2)*
- Namespace verb naming — snake_case standard; confirm any aliases (e.g. `importdata`). *(E0.2)*
