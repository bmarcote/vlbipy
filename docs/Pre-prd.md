# VLIPY - Pre-PRD

VLBI (Very Long Baseline Interferometry) data reduction is one of the most manual, expert-dependent workflows in radio astronomy. Reducing a single EVN, VLBA, or LBA observation currently requires an experienced astronomer to hand-run a long sequence of CASA (or AIPS) tasks — downloading raw correlator output, appending Tsys/gain-curve information, flagging, fringe-fitting, bandpass calibration, self-calibration, and imaging — while manually tracking which steps have already been run, which intermediate products are stale, and which parameters worked for this array/frequency combination. Every observatory (EVN, VLBA, LBA), every observing mode (continuum, spectral line, multi-phase-center, multi-epoch astrometry), and every downstream tool (WSClean, eht-imager, Difmap) currently needs bespoke, one-off scripting. There is no single tool that a user can point at a project code and trust to produce a scientifically sound, science-ready dataset without manual babysitting, while still allowing an expert to intervene, inspect, and safely re-run only what changed.

**vlbipy** aims to be a suite for dealing with VLBI data in different ways:

- Allows to run a start-to-end pipeline to reduce data (import, calibrate, image, and analysis) from VLBI observations, via a CLI, an interactive Python session or via a Jupyter notebook, or via a dashboard.
- Allows a step-by-step run of different calibration or imaging steps by the user, in an interactive way via a Python session, or a Jupyter notebook.
- Allows to run some standard analyses in the resulting (calibrated) data.

**vlbipy** shall be as (backend) package agnostic as possible. The first implementation will mostly rely on the NRAO CASA software package to calibrate the data, but it will also be able to run other packages or independent components (as AOFlagger, WSClean, EHT-Imagers). The higher level must thus be independent and easily implementable for other future packages.

I want a development to-to-bottom, first defining the correct API and user interface, and later on implementing the lower levels and connectors with existing software.

The API, user interface, user-called instructions must be as simple as possible. The main goal of vlbipy is to change the standard approach from scientific software to something more standard programs in society. Most parts should be pre-defined and the user-required inputs should be as minimum as possible.



## Data Reduction Pipeline via CLI

`vlbipy` is a single, modular, backend-agnostic pipeline that takes a user from a
project code (or a minimal TOML config) to calibrated measurement sets, UVFITS
exports, science-ready images, and a self-contained HTML/report package — with no
required manual intervention, but full support for expert step-by-step control.

When complete, a user will be able to launch the VLBI data reduction suite in different ways:

When **running via the CLI**, the user will be able to run:

```bash
vlbipy -p RSM07 --network EVN --target 3C286
```

and get a fully calibrated, imaged, quality-assessed EVN continuum dataset. The same
`Project` object is usable from the CLI, from an interactive Python/Jupyter session,
or (in a later phase) from a web dashboard. Every step checks whether its inputs are
newer than its outputs before re-running, so editing a calibration table by hand and
re-running the pipeline only redoes the steps that depend on it. New backends
(LINC, or others in the future) and new imaging tools (WSClean today; eht-imager and others later)
plug in behind stable abstract interfaces without touching pipeline orchestration
logic. New observatories plug in behind an `ObservatoryHandler` interface.



## Data Reduction via Python interface

When **running inside a Python or Notebook session**, the user will be able to run (for example):

`````python
from vlbipy import VLBIObs

obs = VLBIObs(project='RSM07', network='EVN', target='3C286')
obs.importdata()
obs.calibrate.initial_calibration()
obs.calibrate.bandpass()
obs.flag.aoflagger()
obs.plot.tplot()
img = obs.clean(target='3C286', robust=2, imsize=8192)
img = obs.clean.wsclean(target=obs.sources.target, robust=-2, imsize=8192)
obs.selfcal(img)   # self-calibrates the MS data based on the image for that source
img.export_fits(outfile='3C286.image.fits')
`````

`vlbipy` is here mainly a wrapper to put different functions together and offering to the user a bit higher-level interface, ignoring the lower-level commands.



## API

I still do not have a clear picture of the API and how should be the classes and the user interface. Give me best ideas.  The standard workflow that the user would encounter is the following:

- A user starts specifying a project, or list of projects to analyze.
- Then all the following steps would run for each of the projects in parallel. All the initial steps will run independently. Only after all data are calibrated, the different projects will be merged into a single campaign.
- The internal structure is hidden for the user: when loaded, each project would be converted into a MS file, and this would be split per source (to speed up the analysis).
- In general, for each project there will be a MS file per source. All calibration/flagging/imaging steps will be running on these sources.
- However, after the data are calibrated, the data will be split again (with `mstransform`) into a MS per source. And for each, multiple images may be produced (e.g. for different robust parameters; in the pipeline to cover robust from -2, -1, 0, 1, 2 on CASA standard values). This needs to be supported by an easy interface with a syntax equivalent to the one I quoted above.



Feel free to re-think the syntax of these classes and modules for the user. Give me ideas of a consistent API (that later will be used for the pipeline). Make the full API architecture (after asking for solving the doubts), leave the backend implementation for later. I want to play with a API (implemented for dummy calls/outputs simulating for future real data).
