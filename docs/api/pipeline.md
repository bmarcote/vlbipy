# Pipeline

The pipeline module defines the full automated calibration and imaging pipeline as a sequence of discrete step functions. Each step takes a `Project` and modifies it in place. The pipeline can be run end-to-end via `run_pipeline()` or step-by-step.

