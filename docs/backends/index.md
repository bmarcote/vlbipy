# Backends

vlbipy separates the **pipeline logic** (what calibration steps to perform) from the **backend implementation** (how each step is executed). This allows the same pipeline to drive either CASA or AIPS transparently.

## Architecture

The backend system is built on five abstract base classes defined in `vlbipy.backends.base`:

| Interface | Responsibility |
| --- | --- |
| `DataBackend` | Read metadata, split sources, export UVFITS |
| `ImportBackend` | Import FITS-IDI/UVFITS into native format |
| `CalibrationBackend` | Fringe fitting, bandpass, gain calibration, applycal |
| `FlaggingBackend` | Flag management (file-based, autocorr, edge, quack, tfcrop, aoflagger) |
| `ImagingBackend` | CLEAN deconvolution (tclean, wsclean) |

Each backend (CASA, AIPS) provides concrete implementations of all five interfaces. The `Project` class lazily instantiates the correct backend objects based on the `backend` parameter.

## Backend Selection

The backend is selected at project creation:

```python
# Via Python API
project = Project(project_code="EG078B", observatory="EVN", backend="CASA")

# Via CLI
vlbipy -p EG078B -n EVN --backend CASA
```

```toml
# Via TOML configuration
[global]
backend = "CASA"
```

## Lazy Initialization

Backend instances are created on first access, not at project initialization. This means:

- The core vlbipy package can be imported without any backend installed
- Backend-specific imports happen only when needed
- You can inspect project metadata and configuration without a backend

```python
project = Project(project_code="EG078B", observatory="EVN", backend="CASA")
# No casatools import has happened yet

project.data  # <- CasaDataBackend is created here, casatools imported
```

## Adding a New Backend

To add support for a new backend (e.g. a future Python-native correlator):

1. Create a new subpackage under `vlbipy/backends/` (e.g. `vlbipy/backends/newbackend/`)
2. Implement all five abstract interfaces from `vlbipy.backends.base`
3. Add the backend to the `Backend` enum in `vlbipy.models`
4. Register it in the `Project._create_*_backend()` factory methods

See the [API reference](../api/backends.md) for the full interface definitions.
