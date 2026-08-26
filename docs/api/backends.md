# Backends

Abstract base classes defining the backend-agnostic interface. Every backend (CASA, AIPS) must implement these ABCs. The pipeline code operates exclusively through these interfaces so backends are swappable.

## Backend Registry

::: vlbipy.backends

## Abstract Interfaces

::: vlbipy.backends.base.DataOps

::: vlbipy.backends.base.ExportOps

::: vlbipy.backends.base.CalibrationOps

::: vlbipy.backends.base.FlagOps

::: vlbipy.backends.base.ImagingOps
