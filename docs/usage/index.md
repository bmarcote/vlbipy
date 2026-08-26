# Three ways to use vlbipy

!!! warning "Work in progress"

    vlbipy is under active development. Calibration through to per-source
    calibrated data works end to end on the CASA backend; imaging and
    self-calibration are not implemented yet. See [Status](status.md) for what
    is done, what is partial, and what is missing.

vlbipy exposes the same pipeline three ways. They are not different programs —
all three drive the same `VLBIObs` object, so a run started on the command line
can be inspected and continued from Python, and vice versa.

<div class="grid cards" markdown>

-   :material-play-box: **[Full pipeline](pipeline.md)**

    One command, raw data to calibrated products. Suited to batch and server
    execution. *Calibration works today; imaging is still to come.*

-   :material-console: **[Command line](cli.md)**

    Run individual stages, resume where you stopped, re-run from a chosen step.
    The normal way to drive a reduction interactively.

-   :material-language-python: **[Interactive Python](python.md)**

    A notebook or IPython session. Every step returns an object you can look
    at — metadata, calibration tables, SNR surveys, images.

</div>

## The object model

Everything hangs off one object:

```python
from vlbipy import VLBIObs

obs = VLBIObs("rsm07", network="EVN",
              target="3C395", phasecal="J1848+3219", fringe_finder="3C345")
```

`VLBIObs` represents one project or a campaign of several. Operations are
grouped into *callable namespaces*: calling the namespace runs its sensible
default, and its methods run explicit variants.

```python
obs.calibrate()              # the whole calibration chain
obs.calibrate.bandpass()     # just the bandpass step
obs.plot.corner()            # one diagnostic
```

| Namespace | What it does |
|---|---|
| `obs.import_data` | locate or download raw data, import, read metadata |
| `obs.calibrate` | a-priori, instrumental, fringe fit, scalar bandpass |
| `obs.flag` | a-priori flags, autocorrelations, edges, quack, outliers |
| `obs.plot` | every diagnostic plot |
| `obs.export` | per-source split and UVFITS export |
| `obs.clean`, `obs.selfcal` | imaging *(not implemented yet)* |

Read-only state is available directly on the object — `obs.metadata`,
`obs.antennas`, `obs.scans`, `obs.frequency`, `obs.refant`, `obs.gaintables`,
`obs.snr_survey`. For a multi-project campaign each returns a dict keyed by
project code; for a single project it returns the value itself.

## Where things are written

```
<work_dir>/
├── <code>.ms                 measurement set
├── caltables/                calibration tables
├── caltables.txt             CASA cal library: what was applied, in order
├── callibs/                  one cal library per solve: what it solved on top of
├── calibrated_data/          per-source MS + UVFITS
├── plots/
│   ├── raw/                  before calibration
│   ├── caltables/            the solutions themselves
│   └── calibrated/           after calibration
├── logs/                     vlbipy and CASA logs
├── input_data/               FITS-IDI, .antab, .uvflg
├── .pipeline_state.json      which steps have completed
└── .caltables.json           the calibration chain, so a resumed run can rebuild it
```

Nothing is written outside the project directory: CASA's log is redirected into
`logs/` and any stray `casa*.log` is swept up, so the directory you launch from
stays clean.

### The cal libraries

Every point where calibration is applied — both `applycal` and the on-the-fly
priors of each solve — uses a CASA cal library rather than parallel
`gaintable` / `interp` / `spwmap` / `gainfield` lists. One line per table, with
its own interpolation, field mapping and subband mapping:

```
# vlbipy cal library for rsm07
# one line per calibration table, in application order
# tinterp/finterp = time/frequency interpolation; fldmap = solutions to transfer from; spwmap = subband mapping
caltable='rsm07/caltables/rsm07.tsys' calwt=True tinterp='nearest'
caltable='rsm07/caltables/rsm07.gcal' calwt=True tinterp='nearest'
caltable='rsm07/caltables/rsm07.sbd' calwt=True tinterp='nearest'
caltable='rsm07/caltables/rsm07.mbd' calwt=True tinterp='linear' fldmap='J1848+3219' spwmap=[0, 0, 0, 0]
```

`caltables.txt` records the full chain applied to the data; `callibs/<table>.txt`
records what each individual solve was solved on top of. Both are plain text next
to the data, which is the record you need months later — and what a referee would
ask for — rather than something reconstructible only by re-reading the code.

`fldmap` is where phase referencing lives: a table carrying a field mapping has
those solutions transferred to every field, which is how the phase calibrator's
fringe solutions reach the target.

`calwt` calibrates the visibility weights along with the data. It defaults to
True, matching CASA — after Tsys and the gain curve the weights should track each
antenna's real sensitivity, which matters on an array mixing a 100 m dish with
25 m ones. It is per table, so `CalTable(..., calwt=False)` opts one out.
