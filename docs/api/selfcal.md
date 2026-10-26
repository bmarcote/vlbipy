# Self-Calibration

The real-data pipeline uses Difmapy to fit a circular Gaussian to the visibilities first, then
refine calibrator phases and Bayesian station amplitudes against parametric source models.
Bayesian candidates default to `current`, `gauss1`, `gauss2` and `gauss3`, not CLEAN models.
Joint-epoch calibrator modelling and fixed-model refinement follow the same policy.

Targets and check sources start with a single fitted component and are not self-calibrated by
this imaging path. The fitted model is convolved with the restoring beam at each requested
Briggs weighting, adding the untouched residual map. No residual CLEAN runs automatically.
A nonconverged or nonphysical fit is reported as a failure, not replaced with CLEAN.

The default `[imaging]` settings are `method = "modelfit"` and `max_components = 1`.
Increasing the component limit permits additional Gaussian components only when residual
significance and fit improvement justify them. CLEAN requires an explicit method override.

Multi-epoch finishing uses memory-aware batches of isolated processes. Each epoch receives the
full usable CPU thread allowance; `epoch_workers` and `epoch_memory_fraction` limit admission,
not its numerical thread count. Plot workers also respect the epoch's memory allowance.

Legacy or unverified selfcal/joint tables cannot be reused under the current modelfit policy.
`StaleCalibrationError` stops the run before product generation; rebuild from pre-selfcal data
with `--scratch`. `force=True` alone cannot establish safe calibration provenance. A running
process must be restarted to use changed Python code or the rebuilt native extension.

## Multi-epoch refinement

The campaign concatenates every observed epoch of each phase-reference source after per-epoch
calibration. Calibrator inputs are normalized to a common flux level for modelling only. A single
Gaussian is fitted first, then significant additional structure may be fitted while alternating with
the phase-only self-calibration ladder. These deeper limits do not change target imaging's one-component
starting policy:

```toml
[selfcal]
joint_max_components = 10
joint_model_rounds = 3
joint_amplitude = false
```

The resulting sky model is scaled by `epoch_flux / fitted_joint_model_flux` for each epoch. Gains are
then solved separately on that epoch against the fixed structure, retaining the epoch's own antenna
IDs, times and flux scale. They follow its phase-reference plan to the target/check sources; combined
selfcal gains are never applied indiscriminately across epochs. Accumulated `selfcal_*` and `joint_*`
tables both participate in two-reference interpolation when that strategy is selected.

After gain application the epoch splits and images are refreshed, then calibrated target, check-source
and calibrator data are concatenated into the final campaign products. Previous joint tables are removed
from the chain before rebuilding a joint model, so repeated runs do not double-apply old corrections.
Campaign cache reuse requires matching scientific settings and existing model/MS/FITS products.

::: vlbipy.namespaces.SelfcalNamespace
