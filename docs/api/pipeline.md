# Namespaces

There is no separate `pipeline` module — the full reduction is
`VLBIObs.run()` (see [Pipeline Workflow](../pipeline.md)), which chains
these same callable namespaces in order. Each namespace groups one stage of
the reduction; calling it runs its sensible default, and its methods are the
explicit variants (`obs.calibrate()` runs the full chain,
`obs.calibrate.bandpass()` runs just that step).

::: vlbipy.namespaces
