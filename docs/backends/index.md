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

Only the CASA backend is implemented in full today. `dask-ms` implements
import and a-priori calibration for lazy, distributed-ready reads; AIPS is a
stub — every method raises; `dummy` is an in-memory synthetic backend used
for tests and examples with no external dependency.

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
backend = "casa"   # casa (default) | dask-ms | dummy | aips
```

## Adding a New Backend

To add support for a new backend:

1. Add a new module under `vlbipy/backends/` (e.g. `vlbipy/backends/newbackend.py`)
2. Implement the `BackendComponent` interfaces from `vlbipy.backends.base` it can support
3. Add the backend's name to `BackendKind` in `vlbipy.models`
4. Register it in `get_backend()` (`vlbipy/backends/__init__.py`)

See the [API reference](../api/backends.md) for the full interface definitions.
