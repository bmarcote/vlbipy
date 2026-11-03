"""The expert rules of the AI mode, each written down with the evidence and the experiment that taught it.

A rule maps a *finding* (a dict of features computed by :mod:`~vlbipy.ai.tsys`, :mod:`~vlbipy.ai.antenna` or
:mod:`~vlbipy.ai.distributions`) to a *label* and an *action* the pipeline can take. The rules are the teacher:
:func:`label` applies them to every finding of an experiment, and the labelled findings - corrected by hand
where a human disagreed - are the training set of the small model (:mod:`~vlbipy.ai.model`), which learns to
make the same calls from the features alone and generalises where no rule was written.

Physics every rule rests on (EVN, SFXC; the same holds for DiFX after ACCOR):

* the raw cross-correlations are correlation coefficients, ``rho_ij = S / sqrt(SEFD_i SEFD_j)``, with
  ``SEFD = Tsys / (DPFU * gain_curve(elevation))``. A *real* Tsys change moves the raw amplitude by
  ``-0.5 log(Tsys)``; a Tsys change the raw data do not show is a measurement error;
* the two circular hands see the same sky: for an unpolarized calibrator ``|RR| = |LL|``, and a common
  atmospheric/elevation term moves both hands' Tsys together;
* a VLBI band is far too narrow for a spectral index: every subband has the same calibrated amplitude;
* calibrated visibilities are a slowly varying signal plus complex Gaussian noise: Rice amplitudes, von Mises
  phases (uniform at low SNR). Source structure is baseline-based and slow; anything antenna-based or fast is
  instrumental.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Rule:
    """One expert decision: ``when(finding)`` -> ``label`` and ``action``."""

    id: str
    stage: str                 # "tsys", "prediagnostic", "calibration", "calibrated"
    label: str
    action: str
    when: Callable[[dict], bool]
    why: str
    learned_from: tuple = ()
    params: dict = field(default_factory=dict)


def _f(finding: dict, key: str, default=np.nan) -> float:
    value = finding.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


RULES: tuple[Rule, ...] = (
    # ------------------------------------------------------------------ Tsys table
    Rule("tsys.nominal", "tsys", "nominal_tsys", "rely_on_amplitude_selfcal",
         lambda f: "nominal" in f.get("labels", ()),
         "A constant Tsys (with DPFU = 1 it is an SEFD) gives the flux scale but no elevation or weather "
         "dependence. The station's raw amplitude trend survives calibration and must be taken out by a "
         "time-dependent amplitude self-calibration on the phase calibrator.",
         ("EB104G: JB 80 K, T6 26 K; JB 6% scan-to-scan scatter, T6 -25% over the track in raw data",)),
    Rule("tsys.noisy_hand", "tsys", "noisy_tsys_hand", "substitute_hand",
         lambda f: "noisy_hand" in f.get("labels", ()) and not (_f(f, "fast_slope") > 0.5),
         "One receptor's Tsys jumps from sample to sample far more than its partner's, and the raw amplitudes do "
         "not follow the jumps (scan-to-scan slope ~0). Both hands see the same atmosphere: use the steady hand's "
         "time dependence, scaled to the noisy hand's median level.",
         ("EB104G: SR RCP spw5-7 (15% point-to-point vs 0%; 5-6% per-scan noise injected, raw slope 0.00); "
          "UR RCP spw4-7 (3.8% vs 0.3%)",)),
    Rule("tsys.unreal_tsys_variation", "tsys", "tsys_fluctuation_not_in_data", "smooth_long",
         lambda f: _f(f, "fast_rms") > 0.005 and abs(_f(f, "fast_slope", 1.0)) < 0.3,
         "Scan-to-scan Tsys fluctuations that the raw correlation coefficients do not show are measurement noise: "
         "smooth the series over a long window (~30 min) so that they do not reach the calibrated amplitudes.",
         ("EB104G: SR LCP (1% per scan, slope ~0), UR RCP (0.4-1.2%, slope ~0)",)),
    Rule("tsys.real_variation", "tsys", "tsys_variation_real", "keep_time_resolution",
         lambda f: 0.7 < _f(f, "slope") <= 1.2 and _f(f, "corr") > 0.9,
         "The raw amplitudes follow -0.5 log Tsys: the variation is real (elevation, weather, RFI) and the Tsys "
         "must be applied at full time resolution - neither smoothed away nor flagged.",
         ("EB104G: TR spw6-7 bumps (slope 0.84-0.88, corr 0.72-0.74)",)),
    Rule("tsys.undercorrects", "tsys", "tsys_undercorrects", "opacity_correction",
         lambda f: 1.2 < _f(f, "slope") < 3.0 and _f(f, "corr") > 0.9,
         "The raw amplitudes move with the Tsys, but by more than -0.5 log Tsys: the atmosphere that raises the Tsys "
         "also absorbs the source (rain, clouds, low elevation), and a Tsys measured at the ground does not include "
         "that attenuation (Tsys* = Tsys e^tau does). Correct the Tsys by the exponent the raw data show.",
         ("EB104G: YS Tsys x2.4 in the last 1.5 h (weather), raw slope 2.1-2.2 (corr 0.98): calibrated YS still "
          "-40% after a plain Tsys correction; O8 slope 1.3 (corr 0.99) as it sets",)),
    Rule("tsys.high_subband_unreal", "tsys", "spurious_high_tsys", "rescale_subband",
         lambda f: "high_subband" in f.get("labels", ()) and f.get("tsys_real") is False,
         "A subband's Tsys sits well above the antenna's other subbands but its raw amplitudes are not lower by the "
         "matching sqrt: the excess is not real, and applying it would inflate that subband.",
         ("EB104G: WB spw6 RCP x1.67 (raw -4%, expected -23%), WB spw7 LCP x1.36 (raw +1%, expected -14%)",)),
    Rule("tsys.high_subband_real", "tsys", "real_high_tsys", "keep",
         lambda f: "high_subband" in f.get("labels", ()) and f.get("tsys_real") is True,
         "A subband's Tsys excess is matched by lower raw amplitudes (RFI or a noisier IF): the Tsys is right.",
         ("EB104G: EF spw6 LCP x1.32 (raw -10%, expected -13%)",)),
    Rule("tsys.unphysical", "tsys", "unphysical_tsys", "flag_samples",
         lambda f: _f(f, "unphysical", 0) > 0 and _f(f, "n", 0) > 0,
         "Tsys values far below the receiver temperature (or non-positive placeholders) are not measurements.",
         ("EB104G: EF LCP 3.3 K in the first 20 samples",)),
    # ------------------------------------------------------------------ raw data (pre-diagnostic)
    Rule("raw.gain_scatter", "prediagnostic", "antenna_amplitude_scatter", "amplitude_selfcal_scan",
         lambda f: _f(f, "scan_scatter") > 3.0 * max(_f(f, "array_scan_scatter", 0.01), 0.005),
         "An antenna's raw amplitude jumps from scan to scan several times more than the rest of the array's "
         "(pointing, a gain-unstable receiver). The a-priori calibration cannot follow it.",
         ("EB104G: JB 6.2% vs <=1.6% for every other station on J1031+7441",)),
    Rule("raw.gain_trend_tsys", "prediagnostic", "trend_explained_by_tsys", "apply_tsys",
         lambda f: abs(_f(f, "trend")) > 0.1 and abs(_f(f, "residual_trend", np.inf)) <= 0.1,
         "The antenna's raw amplitude drifts, and its Tsys drifts the matching way: the a-priori calibration "
         "takes it out.",
         ("EB104G: O8 -17% raw, Tsys +25% as it sets",)),
    Rule("raw.gain_trend", "prediagnostic", "antenna_sensitivity_trend", "amplitude_selfcal_slow",
         lambda f: abs(_f(f, "trend")) > 0.1 and _f(f, "trend_consistent_fields", 0) >= 1
         and not abs(_f(f, "residual_trend", np.inf)) <= 0.1,
         "An antenna's raw amplitude drifts by more than 10% over the track on more than one source: "
         "instrumental, not structure. Check whether its Tsys/gain curve follows it; if not, self-calibrate it.",
         ("EB104G: T6 -25% (J1031+7441) and -50% (J1048+7143); UR -20%; YS -30% in the last 1.5 h "
          "(explained by its Tsys)",)),
    Rule("raw.low_subband", "prediagnostic", "low_sensitivity_subband", "scalar_bandpass",
         lambda f: _f(f, "raw_subband_offset") < -0.1 and not f.get("tsys_explains", False),
         "A subband of an antenna is >10% low in the raw data with an unremarkable Tsys: lower sensitivity the "
         "a-priori calibration does not know about. The subband levelling (scalar bandpass) corrects it.",
         ("EB104G: EF spw2 -16%/-12%, UR spw2/spw4 -10%",)),
    # ------------------------------------------------------------------ calibration
    Rule("cal.hand_offset", "calibration", "polarization_amplitude_offset", "solve_hand_ratio",
         lambda f: abs(_f(f, "hand_ratio")) > 0.03,
         "The calibrated RR and LL of an antenna differ on unpolarized calibrators: one hand's Tsys (or DPFU) is "
         "off. Neither difmap (one gain per station) nor a per-hand normalisation can see it: the subband "
         "levelling must normalise both hands of an antenna together.",
         ("EB104G: UR -8..-15% (RCP Tsys 12-18% low), O8 -4..-7%, SR -6..+9%, WB spw6 +27%",)),
    Rule("cal.introduced_variation", "calibration", "calibration_introduced_variation", "bisect_chain",
         lambda f: _f(f, "calibrated_scatter") > 1.5 * max(_f(f, "raw_scatter", 0.0), 0.005),
         "An antenna is more variable after a calibration step than before it: that step injected noise (a noisy "
         "Tsys hand, a smoothing across fields, a table applied to fields it was not solved for).",
         ("EB104G: SR scan scatter 0.5% raw -> 1.8% after Tsys",)),
    Rule("cal.time_amplitude_transfer", "calibration", "time_amplitude_not_transferable", "apply_to_calibrator_only",
         lambda f: _f(f, "check_rchisq_with") > 1.05 * _f(f, "check_rchisq_without", np.inf),
         "A time-dependent amplitude self-calibration solved per scan on the phase calibrator improves the phase "
         "calibrator but makes the check source worse: the curves absorbed the calibrator's structure (an antenna "
         "on the longest baselines sees a different structure on each source). Keep them on the calibrator; transfer "
         "only what the check source confirms.",
         ("EB104G: check source reduced chi-squared 28.9 -> 34.9 and image SNR 204 -> 180 with the transfer; T6's "
          "curve x4.6 on J1031+7441 vs x2.1 on J1027+7428",)),
    # ------------------------------------------------------------------ calibrated data
    Rule("vis.outlier_bin", "calibrated", "outlier", "flag",
         lambda f: _f(f, "z2") > -np.log(1e-7),
         "A channel-averaged bin whose residual from the local signal is beyond the noise distribution "
         "(|z|^2 > 16, p < 1e-7 for an exponential law).",
         ("EB104G",)),
    Rule("vis.excess_scatter", "calibrated", "excess_scatter", "flag_scan",
         lambda f: _f(f, "chi2r") > 2.0 and _f(f, "n", 0) >= 4,
         "A scan of one baseline/subband/hand scatters more than its own thermal noise (reduced chi-squared > 2): "
         "phase jumps, RFI, a receiver dropping in and out.",
         ("EB104G",)),
    Rule("vis.subband_departure", "calibrated", "subband_departure", "flag_subband_scan",
         lambda f: abs(_f(f, "ratio", 1.0) - 1) > 0.15 and abs(_f(f, "nsigma")) > 6,
         "One subband's amplitude departs from the band median of its scan by >15% and >6 sigma: no source has a "
         "spectral index across a VLBI band.",
         ("EB104G",)),
    Rule("vis.amplitude_departure", "calibrated", "amplitude_departure", "flag_baseline_scan",
         lambda f: abs(_f(f, "ratio", 1.0) - 1) > 0.2 and abs(_f(f, "nsigma")) > 5,
         "A baseline's amplitude jumps away from its own slow trend in one scan: source structure changes "
         "slowly along the uv track, a failing receiver does not.",
         ("EB104G: Ur-T6 drop on J1031+7441",)),
)


def rules_for(stage: str) -> list[Rule]:
    """The rules of one stage."""
    return [rule for rule in RULES if rule.stage == stage]


def label(finding: dict, stage: str) -> list[dict]:
    """Every rule of ``stage`` that fires on ``finding``: ``[{"rule", "label", "action"}]``."""
    fired = []
    for rule in rules_for(stage):
        try:
            if rule.when(finding):
                fired.append({"rule": rule.id, "label": rule.label, "action": rule.action})
        except Exception:  # noqa: BLE001 - a malformed finding never fires a rule
            continue
    return fired


def describe() -> str:
    """The rule base as Markdown (for the documentation and the report)."""
    lines = ["| rule | stage | label | action | why | learned from |", "|---|---|---|---|---|---|"]
    for rule in RULES:
        lines.append(f"| `{rule.id}` | {rule.stage} | {rule.label} | {rule.action} | {rule.why} | "
                     f"{'; '.join(rule.learned_from)} |")
    return "\n".join(lines)
