"""AI mode: data-driven diagnostics of a VLBI calibration, and the small model that learns from them.

The modules here turn a measurement set and its calibration tables into compact, labelled *features* —
per antenna, subband, hand and time — and decide from them what a careful observer would decide by eye:
an antenna that loses sensitivity, a Tsys hand that is noise, a subband that sits low, a baseline whose
calibrated visibilities do not follow the noise distribution they must follow.

* :mod:`~vlbipy.ai.visstats` — the visibility-statistics cube every diagnostic reads;
* :mod:`~vlbipy.ai.antenna` — antenna-based decomposition of baseline amplitudes (gains vs time / subband / hand);
* :mod:`~vlbipy.ai.tsys` — quality of the system-temperature table, hand by hand;
* :mod:`~vlbipy.ai.distributions` — Rice / Rayleigh / von Mises tests of calibrated data, and the flags they imply;
* :mod:`~vlbipy.ai.knowledge` — the expert rules, written down with the experiment that taught each of them;
* :mod:`~vlbipy.ai.model` — a numpy-only classifier trained on the annotated findings, small enough to ship.
"""
