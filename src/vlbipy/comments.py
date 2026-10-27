"""General comments of a reduction: the issues, decisions and results worth reading first.

``build_comments(context)`` turns the JSON-safe context of the HTML report
(:func:`vlbipy.dashboard.collect`) into a flat list of comments. The Overview page of the
report and the top of ``summary.md`` render that same list, so the two can never disagree.
Every rule reads the context alone - i.e. the caches a reduction leaves on disk - so the
comments can be rebuilt after the measurement set has been deleted.

A comment is a plain dict:

* ``severity``: ``"problem"`` (something is wrong), ``"warning"`` (look at it) or ``"note"``;
* ``category``: ``"calibration"``, ``"result"``, ``"antenna"`` or ``"pipeline"``;
* ``antenna``: station code, ``""`` when the comment is not about one station;
* ``text``: the sentence;
* ``when`` (warnings recorded by the pipeline only): UTC time the warning was first seen;
* ``data`` (calibration and result comments only): the numbers behind the sentence, for the boxes of the
  Overview page - ``kind`` (``"flagging"``, ``"instrumental_delay"``, ``"selfcal"``, ``"table"`` or
  ``"source"``) and the fields of that kind (see the rule that writes it).

The module also owns the small file that keeps the warnings of the pipeline between
processes (``.run_warnings.json``), since those warnings live in memory otherwise.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

logger = get_logger()

#: Severities, most serious first (the order comments are listed in within a group).
SEVERITIES: tuple[str, ...] = ("problem", "warning", "note")
#: Fixed-width label of each severity in the plain-text block.
SEVERITY_LABELS: dict[str, str] = {"problem": "PROBLEM", "warning": "WARNING", "note": "note"}

#: A flagged fraction at or above this counts as "no usable data at all".
NO_DATA_FLAGGED_FRACTION: float = 0.999
#: An antenna flagged this much more (absolute fraction) than the median antenna is pointed out.
HEAVY_FLAGGING_EXCESS: float = 0.15
#: Overall flagged fraction (dead antennas left out when known) from which the flagging is a warning.
HIGH_FLAGGED_FRACTION: float = 0.5

#: Fringe SNR below which an antenna counts as not detected in a scan.
FRINGE_DETECTION_SNR: float = 3.0
#: Source roles whose scans say something about an antenna: bright enough for every station to fringe.
CALIBRATOR_ROLES: tuple[str, ...] = ("fringefinder", "phasecal")
#: A scan is informative for one antenna only when this fraction of the other antennas was detected.
INFORMATIVE_DETECTED_FRACTION: float = 0.5
#: A scan on a source that is not a calibrator counts for an antenna only when the antenna is detected in
#: at least this fraction of the surveyed scans of that source (i.e. the source is bright for it).
USUAL_DETECTION_FRACTION: float = 0.8
#: Shortest stretch of consecutive scans (without data, or without fringes) reported on its own.
MIN_RUN_SCANS: int = 3
#: Most stretches listed per antenna and kind; the others are folded into a count.
MAX_RUNS_LISTED: int = 3
#: Scattered scans are mentioned only from this fraction of the scans considered (and at least MIN_RUN_SCANS).
SCATTERED_FRACTION: float = 0.1
#: A polarization is called dead only when the other hand was detected in at least this many scans.
MIN_DETECTIONS_OTHER_HAND: int = 3
#: Names of the parallel hands in the comments.
HAND_NAMES: dict[str, str] = {"RR": "RCP", "LL": "LCP"}

#: Decimals an amplitude correction is shown with; its level is decided on the value as displayed.
GAIN_DECIMALS: int = 2
#: Largest ``|gain - 1|`` of a correction that needs no remark (the green cells of the gain table).
GAIN_WARN_DEVIATION: float = 0.10
#: Largest ``|gain - 1|`` of a moderate correction (orange cells); beyond it the correction is large (red).
GAIN_BAD_DEVIATION: float = 0.30
#: An applied correction within this of 1.0 "stayed at 1" although the unconstrained gscale wanted more.
GAIN_UNFIXED_DEVIATION: float = 0.05
#: Fewest antennas with an unconstrained gscale for their median to stand for the array's flux scale.
MIN_ANTENNAS_FOR_SCALE: int = 4
#: Most single subbands listed per antenna in the "single subband beyond 30%" comment.
MAX_SUBBANDS_LISTED: int = 4
#: From this many "off but not corrected" antennas (and at least half of those solved) the source itself
#: is reported as poorly constrained, instead of every antenna.
MIN_UNFIXED_FOR_SOURCE: int = 3
#: A sentence of the Bayesian report listing a value per IF is cut down to their range from this many IFs.
CONDENSE_FROM_IFS: int = 3

#: Median solution SNR below which a calibration table is pointed out (0 means "not measured").
POOR_TABLE_SNR: float = 10.0
#: Dynamic range of the CLEAN image below which a self-calibrated calibrator is pointed out.
LOW_CALIBRATOR_DYNAMIC_RANGE: float = 50.0
#: A source found closer than this (mas) to the phase centre in the search map is said to be "at" it.
CENTRED_OFFSET_MAS: float = 0.05

#: Most warnings of the pipeline listed; the rest is a count.
MAX_PIPELINE_WARNINGS: int = 30
#: File (in the work directory) keeping the warnings the pipeline raised: ``{message: first-seen UTC}``.
RUN_WARNINGS_FILENAME: str = ".run_warnings.json"
#: Prefix the warning collector puts on a prominent anomaly.
ANOMALY_PREFIX: str = "ANOMALY: "
#: Pipeline warnings an antenna rule already words better; dropped for an antenna that rule reported.
DUPLICATED_WARNINGS: tuple[str, ...] = ("recorded only subband", "of its recorded data flagged")

#: Width of the ruler of the plain-text block (the same as the widest block of ``summary.md``).
TEXT_WIDTH: int = 100


# -- small helpers -----------------------------------------------------------------------------

def make_comment(severity: str, category: str, text: str, antenna: str = "", when: str = "",
                 data: Optional[dict] = None) -> dict:
    """Build one comment dict (see the module docstring); ``when`` and ``data`` are only stored when given."""
    comment = {"severity": severity, "category": category, "antenna": antenna, "text": text}
    if when:
        comment["when"] = when
    if data:
        comment["data"] = data
    return comment


def _finite(value) -> Optional[float]:
    """``value`` as a float when it is a finite number, otherwise None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_ranges(values) -> str:
    """Compact text of a set of integers: ``[0, 1, 2, 3, 8]`` -> ``"0-3, 8"``."""
    numbers = sorted({int(v) for v in values})
    groups: list[list[int]] = []
    for number in numbers:
        if groups and number == groups[-1][1] + 1:
            groups[-1][1] = number
        else:
            groups.append([number, number])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in groups)


def compact_antennas(names, everyone) -> str:
    """The shortest of ``"all"``, ``"all but X, Y"`` and the list itself for ``names`` out of ``everyone``."""
    names, everyone = list(names), list(everyone)
    missing = [name for name in everyone if name not in names]
    if everyone and not missing:
        return "all"
    listed = ", ".join(names)
    if everyone and len(missing) < len(names) and set(everyone) & set(names):
        return f"all but {', '.join(missing)}"
    return listed


def join_names(names: list[str]) -> str:
    """``["A", "B", "C"]`` -> ``"A, B and C"``."""
    names = list(names)
    if len(names) < 2:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + " and " + names[-1]


def format_flux(jy: float) -> str:
    """Flux density with three significant digits in Jy, mJy or uJy (ASCII only: it goes to ``summary.md``)."""
    value = abs(float(jy))
    if value >= 1.0:
        return f"{jy:.3g} Jy"
    if value >= 1e-3:
        return f"{jy * 1e3:.3g} mJy"
    return f"{jy * 1e6:.3g} uJy"


def gain_level(gain) -> Optional[str]:
    """Size of an amplitude correction: ``"good"``, ``"warn"``, ``"bad"``, or None without a value.

    Judged on the value as it is displayed (rounded to :data:`GAIN_DECIMALS`), with the bounds
    :data:`GAIN_WARN_DEVIATION` and :data:`GAIN_BAD_DEVIATION` the gain table of the report colours its
    cells with: a correction that reads ``1.10`` is never called "beyond 10%".
    """
    value = _finite(gain)
    if value is None:
        return None
    deviation = abs(round(value, GAIN_DECIMALS) - 1.0)
    # The tolerance absorbs the binary representation of the bounds (abs(1.1 - 1.0) > 0.1 in floats).
    if deviation <= GAIN_WARN_DEVIATION + 1e-9:
        return "good"
    return "warn" if deviation <= GAIN_BAD_DEVIATION + 1e-9 else "bad"


def _dead_antennas(context: dict) -> set[str]:
    """Antennas without usable data: never observed, or with all of their data flagged."""
    dead = set()
    for antenna in context.get("antennas") or []:
        fraction = _finite(antenna.get("flagged_fraction"))
        if not antenna.get("observed", True) or (fraction is not None and fraction >= NO_DATA_FLAGGED_FRACTION):
            dead.add(antenna["name"])
    return dead


def usable_antennas(context: dict) -> list[str]:
    """Antennas that observed and kept some unflagged data, in the order of the context."""
    dead = _dead_antennas(context)
    return [a["name"] for a in context.get("antennas") or [] if a["name"] not in dead]


def _reference_antennas(context: dict) -> set[str]:
    """Reference antenna(s): the configured chain plus any antenna a scan marks as its reference."""
    names = {name for name in str(context.get("refant") or "").split(",") if name and name != "auto"}
    for scan in context.get("scans") or []:
        names.update(name for name, state in (scan.get("antenna_states") or {}).items() if state == "refant")
    return names


# -- scans in time -----------------------------------------------------------------------------

def _parse_utc(text) -> Optional[datetime]:
    """``"YYYY-MM-DD HH:MM:SS"`` as a datetime, None when it is empty or malformed."""
    try:
        return datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def ordered_scans(context: dict) -> list[dict]:
    """Scans of the context in time order, each with ``start`` / ``end`` datetimes (None when unknown)."""
    scans = []
    for scan in context.get("scans") or []:
        start = _parse_utc(scan.get("start_utc"))
        end = start + timedelta(seconds=float(scan.get("duration_sec") or 0.0)) if start else None
        scans.append({**scan, "start": start, "end": end})
    return sorted(scans, key=lambda s: (s["start"] or datetime.min, s.get("number", 0)))


def _clock(moment: datetime, first_day) -> str:
    """``HH:MM`` of a moment, with ``(+Nd)`` when it falls N days after the first day of the observation."""
    days = (moment.date() - first_day).days
    return moment.strftime("%H:%M") + (f" (+{days}d)" if days else "")


def _scan_label(first: dict, last: dict) -> str:
    """``"scan 5"`` or ``"scans 5-9"`` for a stretch of scans."""
    a, b = first.get("number"), last.get("number")
    return f"scan {a}" if a == b else f"scans {a}-{b}"


def describe_stretch(scans: list[dict], first: int, last: int, *, open_ended: bool) -> str:
    """Words for the stretch ``scans[first..last]`` of a time-ordered scan list.

    Parameters
    ----------
    scans : list of dict
        Result of :func:`ordered_scans` (or a subset of it, still in time order).
    first, last : int
        Indices of the first and last scan of the stretch.
    open_ended : bool
        Say "until ..." / "... to the end of the observation" when the stretch touches either end of
        ``scans``. Only right when ``scans`` is the whole observation.

    Returns
    -------
    str
        e.g. ``"from 13:02 to 14:31 UTC (scans 131-276)"``; just the scan numbers when times are unknown.
    """
    label = _scan_label(scans[first], scans[last])
    start, end = scans[first]["start"], scans[last]["end"]
    day = next((s["start"].date() for s in scans if s["start"]), None)
    if start is None or end is None or day is None:
        return f"in {label}"
    if open_ended and first == 0 and last < len(scans) - 1 and scans[last + 1]["start"]:
        return f"until {_clock(scans[last + 1]['start'], day)} UTC ({label})"
    if open_ended and last == len(scans) - 1 and first > 0:
        return f"from {_clock(start, day)} UTC to the end of the observation ({label})"
    return f"from {_clock(start, day)} to {_clock(end, day)} UTC ({label})"


def find_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Maximal stretches of consecutive True values as ``(first index, last index)`` pairs."""
    runs, start = [], None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def _longest_runs(runs: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
    """The stretches worth listing (longest first, capped) and the number of scans in all the others."""
    ranked = sorted(runs, key=lambda run: (run[0] - run[1], run[0]))
    listed = [run for run in ranked if run[1] - run[0] + 1 >= MIN_RUN_SCANS][:MAX_RUNS_LISTED]
    others = sum(b - a + 1 for a, b in runs if (a, b) not in listed)
    return listed, others


# -- rules: antennas ---------------------------------------------------------------------------

def data_comments(context: dict) -> list[dict]:
    """Antennas with no data, with all data flagged, or flagged far more than the rest of the array."""
    antennas = context.get("antennas") or []
    dead = _dead_antennas(context)
    fractions = [_finite(a.get("flagged_fraction")) for a in antennas if a["name"] not in dead]
    fractions = [f for f in fractions if f is not None]
    typical = statistics.median(fractions) if fractions else None
    comments = []
    for antenna in antennas:
        name, fraction = antenna["name"], _finite(antenna.get("flagged_fraction"))
        if not antenna.get("observed", True):
            comments.append(make_comment("warning", "antenna", "no data: it is in the array but recorded nothing",
                                         name))
        elif name in dead:
            comments.append(make_comment("problem", "antenna", "no usable signal: all of its data is flagged", name))
        elif typical is not None and fraction is not None and fraction - typical >= HEAVY_FLAGGING_EXCESS:
            comments.append(make_comment("warning", "antenna", f"{fraction:.0%} of its data is flagged, against "
                                         f"{typical:.0%} for the median antenna", name))
    return comments


def band_comments(context: dict) -> list[dict]:
    """Antennas that recorded only part of the band, naming the others in the same situation."""
    n_subbands = int((context.get("observation") or {}).get("n_subbands") or 0)
    dead = _dead_antennas(context)
    partial: dict[tuple, list[str]] = {}
    for antenna in context.get("antennas") or []:
        recorded = tuple(sorted(set(antenna.get("subbands") or ())))
        if antenna["name"] not in dead and recorded and len(recorded) < n_subbands:
            partial.setdefault(recorded, []).append(antenna["name"])
    comments = []
    for recorded, names in partial.items():
        plural = "s" if len(recorded) > 1 else ""
        for name in names:
            others = [other for other in names if other != name]
            same = f", like {join_names(others)}" if others else ""
            comments.append(make_comment("note", "antenna", f"recorded only subband{plural} {format_ranges(recorded)} "
                                         f"({len(recorded)} of {n_subbands}){same}", name))
    return comments


def gap_comments(context: dict) -> list[dict]:
    """Stretches of the observation an antenna has no data in (it is absent from consecutive scans).

    Stretches of at least :data:`MIN_RUN_SCANS` scans are worded one by one (longest first, at most
    :data:`MAX_RUNS_LISTED`); shorter gaps are only counted, and antennas that merely skip isolated scans
    are mentioned only when that adds up to :data:`SCATTERED_FRACTION` of the observation.
    """
    scans = ordered_scans(context)
    dead = _dead_antennas(context)
    comments = []
    for antenna in context.get("antennas") or []:
        name = antenna["name"]
        absent = [name not in (scan.get("antennas") or []) for scan in scans]
        if name in dead or not any(absent) or all(absent):
            continue
        listed, others = _longest_runs(find_runs(absent))
        if listed:
            text = "no data " + ", and ".join(describe_stretch(scans, a, b, open_ended=True) for a, b in listed)
            if others:
                text += f"; absent from {others} other scan{'s' if others != 1 else ''}"
            comments.append(make_comment("warning", "antenna", text, name))
        elif others >= max(MIN_RUN_SCANS, SCATTERED_FRACTION * len(scans)):
            comments.append(make_comment("note", "antenna", f"absent from {others} of the {len(scans)} scans, never "
                                         f"more than {MIN_RUN_SCANS - 1} in a row", name))
    return comments


def _fringe_state(scan: dict, name: str) -> str:
    """State of an antenna in a scan (``antenna_states``), derived from the SNR for contexts without states."""
    state = (scan.get("antenna_states") or {}).get(name)
    if state:
        return state
    if name not in (scan.get("antennas") or []):
        return "absent"
    if _finite((scan.get("antenna_snrs") or {}).get(name)) is not None:
        return "snr"
    return "nofringe" if scan.get("surveyed") else "unsurveyed"


def _is_detected(scan: dict, name: str) -> bool:
    """Whether the antenna has a fringe solution of at least :data:`FRINGE_DETECTION_SNR` in the scan."""
    snr = _finite((scan.get("antenna_snrs") or {}).get(name))
    return _fringe_state(scan, name) == "snr" and snr is not None and snr >= FRINGE_DETECTION_SNR


def _detection_rates(scans: list[dict], skipped: set[str]) -> dict[tuple[str, str], float]:
    """Fraction of the surveyed scans of each source an antenna was detected in: ``{(antenna, source): rate}``."""
    seen: dict[tuple[str, str], int] = {}
    hits: dict[tuple[str, str], int] = {}
    for scan in scans:
        if not scan.get("surveyed"):
            continue
        for name in scan.get("antennas") or []:
            if name not in skipped:
                key = (name, str(scan.get("source")))
                seen[key] = seen.get(key, 0) + 1
                hits[key] = hits.get(key, 0) + (1 if _is_detected(scan, name) else 0)
    return {key: hits[key] / count for key, count in seen.items()}


def fringe_comments(context: dict) -> list[dict]:
    """Stretches of scans in which an antenna was present but gave no fringes while the array did.

    Only informative scans count for an antenna: surveyed, with at least
    :data:`INFORMATIVE_DETECTED_FRACTION` of the *other* antennas detected (a scan where nobody fringes
    says nothing about one station), and on a source the antenna can be expected to fringe on - a
    calibrator (any source when no role is declared), or another source on which this antenna is detected
    in at least :data:`USUAL_DETECTION_FRACTION` of the surveyed scans (a bright target). The reference
    antenna(s) and the antennas without usable data are never judged.

    Missing fringes are a problem (a stretch of scans, or every scan); scans missed here and there, never
    :data:`MIN_RUN_SCANS` in a row, are a warning.
    """
    scans = ordered_scans(context)
    skipped = _dead_antennas(context) | _reference_antennas(context)
    declared = any(scan.get("role") in CALIBRATOR_ROLES for scan in scans)
    rates = _detection_rates(scans, skipped)
    history: dict[str, list[tuple[dict, bool]]] = {}
    for scan in scans:
        if not scan.get("surveyed"):
            continue
        calibrator = not declared or scan.get("role") in CALIBRATOR_ROLES
        judged = [name for name in (scan.get("antennas") or []) if name not in skipped]
        detected = {name for name in judged if _is_detected(scan, name)}
        for name in judged:
            if not calibrator and rates.get((name, str(scan.get("source"))), 0.0) < USUAL_DETECTION_FRACTION:
                continue
            others = [other for other in judged if other != name]
            if others and len(detected & set(others)) >= INFORMATIVE_DETECTED_FRACTION * len(others):
                history.setdefault(name, []).append((scan, name not in detected))
    comments = []
    for antenna in context.get("antennas") or []:
        name = antenna["name"]
        entries = history.get(name) or []
        failing = [failed for _, failed in entries]
        if not any(failing):
            continue
        own = [scan for scan, _ in entries]
        if all(failing):
            where = describe_stretch(own, 0, len(own) - 1, open_ended=False)
            many = f"any of the {len(own)} scans" if len(own) > 1 else "the one scan"
            comments.append(make_comment("problem", "antenna", f"no fringes in {many} "
                                         f"it should have been detected in, while the rest of the array fringes "
                                         f"({where})", name))
            continue
        listed, others = _longest_runs(find_runs(failing))
        if listed:
            text = "no fringes " + ", and ".join(describe_stretch(own, a, b, open_ended=False) for a, b in listed)
            if others:
                text += f"; none either in {others} other scan{'s' if others != 1 else ''} where the array fringes"
            comments.append(make_comment("problem", "antenna", text, name))
        elif others >= max(MIN_RUN_SCANS, SCATTERED_FRACTION * len(own)):
            comments.append(make_comment("warning", "antenna", f"no fringes in {others} of the {len(own)} scans where the "
                                         f"rest of the array fringes, never more than {MIN_RUN_SCANS - 1} in a row",
                                         name))
    return comments


def polarization_comments(context: dict) -> list[dict]:
    """Antennas with one dead polarization: no detection in one hand while the other fringes repeatedly."""
    survey = context.get("polarization_snr") or {}
    skipped = _dead_antennas(context) | _reference_antennas(context)
    comments = []
    for name, entry in (survey.get("antennas") or {}).items():
        detected = entry.get("detected") or {}
        if name in skipped or len(detected) < 2:
            continue
        best_hand = max(detected, key=lambda hand: detected[hand])
        for hand, count in detected.items():
            if count == 0 and detected[best_hand] >= MIN_DETECTIONS_OTHER_HAND:
                scans = int(entry.get("scans") or 0)
                comments.append(make_comment(
                    "problem", "antenna", f"no signal in {HAND_NAMES.get(hand, hand)} ({hand}): no fringes in any of "
                    f"the {scans} surveyed scans it took part in, while {HAND_NAMES.get(best_hand, best_hand)} "
                    f"({best_hand}) was detected in {detected[best_hand]}", name))
    return comments


# -- rules: calibration ------------------------------------------------------------------------

def instrumental_delay_comments(context: dict) -> list[dict]:
    """Which scan the instrumental (single-band) delay was solved on, and which antennas it covers.

    Only checkable facts are stated: whether every antenna with usable data is in the scan and was solved
    there, which antennas needed a later stage, and which ones no stage covers.
    """
    stages = context.get("calibration_selection") or []
    usable = usable_antennas(context)
    scans = {scan.get("number"): scan for scan in context.get("scans") or []}
    comments, covered = [], set()
    for index, stage in enumerate(stages):
        scan = scans.get(stage.get("scan"))
        source = stage.get("source") or (scan or {}).get("source") or ""
        label = f"Scan {stage.get('scan')}" + (f" ({source})" if source else "")
        reference = f" (reference antenna {stage['refant']})" if stage.get("refant") else ""
        solved = [name for name in stage.get("antennas") or []]
        covered.update(solved)
        data = {"kind": "instrumental_delay", "stage": index + 1, "scan": stage.get("scan"), "source": source,
                "role": (scan or {}).get("role") or "", "refant": stage.get("refant") or "",
                "antennas": compact_antennas(solved, usable)}
        if index > 0:
            comments.append(make_comment("note", "calibration", f"{label} was used for the instrumental delay of "
                                         f"{join_names(solved)}{reference}, which the first stage does not cover.",
                                         data=data))
            continue
        present = set(scan.get("antennas") or []) if scan else None
        not_in_scan = [name for name in usable if present is not None and name not in present]
        not_solved = [name for name in usable if name not in solved and name not in not_in_scan]
        if present is not None and not not_in_scan and not not_solved:
            reason = f"it was observed by all {len(usable)} antennas with usable data, and all were solved there"
        else:
            reason = f"{sum(1 for name in usable if name in solved)} of the {len(usable)} antennas with usable data " \
                     "were solved there"
            if not_in_scan:
                reason += f"; not in that scan: {join_names(not_in_scan)}"
            if not_solved:
                reason += f"; not solved there: {join_names(not_solved)}"
        comments.append(make_comment("note", "calibration",
                                     f"{label} was selected for the instrumental delay{reference}: {reason}.",
                                     data=data))
    if stages:
        for name in usable:
            if name not in covered:
                comments.append(make_comment("warning", "antenna", "no stage of the instrumental-delay selection "
                                             "covers it: no single-band delay was solved for this antenna", name))
    return comments


def _condense_finding(text: str) -> str:
    """Shorten a sentence of the Bayesian report: a value per IF becomes the range of the values.

    ``"Net flux-scale change: IF1 -0.83%, IF2 -1.24%, ..."`` reads ``"net flux-scale change: -1.87% to
    -0.83% over the 16 IFs"``. Pointers into the JSON report (``"(see runs)"``) are dropped and a leading
    capital is lowered, since the sentence is quoted inside another one.
    """
    text = re.sub(r"\s*\(see [^)]*\)", "", text.replace("`", "")).strip().rstrip(".")
    values = [float(value) for value in re.findall(r"IF\d+ ([+-]?\d+(?:\.\d+)?)%", text)]
    if len(values) >= CONDENSE_FROM_IFS and ":" in text:
        text = f"{text.split(':', 1)[0]}: {min(values):+.2f}% to {max(values):+.2f}% over the {len(values)} IFs"
    return text[:1].lower() + text[1:] if text[:1].isupper() and text[1:2].islower() else text


def _general_findings(entry: dict) -> list[str]:
    """Sentences of the Bayesian report about the source as a whole (not one station, not the model choice)."""
    stations = set(entry.get("antennas") or [])
    kept = []
    for finding in entry.get("findings") or []:
        text = str(finding).strip()
        prefix = re.match(r"^([A-Za-z0-9]+):", text)
        if text and not text.startswith("Best source model") and not (prefix and prefix.group(1) in stations):
            kept.append(_condense_finding(text))
    return kept


def _entry_gains(entry: dict) -> dict:
    """Reading of one source of ``gain_corrections``, antenna by antenna.

    Returns
    -------
    dict
        ``solved`` (antennas with a median correction), ``medians`` (``{antenna: (median, level)}`` for the
        medians beyond :data:`GAIN_WARN_DEVIATION`), ``outliers`` (``{antenna: [(subband label, gain)]}``:
        single subbands beyond :data:`GAIN_BAD_DEVIATION` when the median is not), ``off`` (``{antenna:
        {"naive", "applied", "sigma", "p"}}``: the unconstrained gscale differs from 1 by more than
        :data:`GAIN_WARN_DEVIATION` while the applied factor stayed within :data:`GAIN_UNFIXED_DEVIATION`
        of 1), ``trusted`` (antennas whose unconstrained and applied factors agree within
        :data:`GAIN_UNFIXED_DEVIATION`, i.e. whose gain this source pins down) and ``misscaled``
        (``{antenna: {"relative", "applied"}}``, see below).

    ``misscaled`` is judged on the unconstrained gscale *relative to the median of the array*: one
    station with a wrong a-priori scale pulls the source model towards its own error, and against that
    model every *other* antenna comes out off by a common factor, which says nothing about them (EB104D:
    T6 ~20x low, every other antenna "x0.4-0.7"). An antenna is misscaled when, relative to the array, it
    is beyond :data:`GAIN_BAD_DEVIATION` and the correction applied recovered less than half of it (in
    log): the Bayesian self-calibration shrinks every correction to its ~10% prior, so a station-scale
    error is never fixed there. Such an antenna is left out of ``off``.
    """
    reading = {"solved": [], "medians": {}, "outliers": {}, "off": {}, "trusted": set(), "misscaled": {}}
    labels = entry.get("subband_labels") or []
    for name in entry.get("antennas") or []:
        median = _finite((entry.get("gain_median") or {}).get(name))
        if median is None:
            continue
        reading["solved"].append(name)
        level = gain_level(median)
        if level != "good":
            reading["medians"][name] = (median, level)
        if level != "bad":
            per_subband = zip(labels, (entry.get("gain_per_if") or {}).get(name) or [])
            outliers = [(label, float(gain)) for label, gain in per_subband if gain_level(gain) == "bad"]
            if outliers:
                reading["outliers"][name] = outliers
        naive = _finite((entry.get("naive_gscale") or {}).get(name))
        if naive is None:
            continue
        if abs(naive - median) <= GAIN_UNFIXED_DEVIATION:
            reading["trusted"].add(name)
        elif gain_level(naive) != "good" and abs(median - 1.0) <= GAIN_UNFIXED_DEVIATION:
            reading["off"][name] = {"naive": naive, "applied": median,
                                    "sigma": _finite((entry.get("sigma") or {}).get(name)),
                                    "p": _finite((entry.get("p_correction") or {}).get(name))}
    naive_gains = {name: _finite((entry.get("naive_gscale") or {}).get(name)) for name in reading["solved"]}
    naive_gains = {name: value for name, value in naive_gains.items() if value is not None and value > 0}
    if len(naive_gains) >= MIN_ANTENNAS_FOR_SCALE:
        scale = statistics.median(naive_gains.values())
        for name, naive in naive_gains.items():
            relative = naive / scale
            applied = _finite((entry.get("gain_median") or {}).get(name)) or 1.0
            if gain_level(relative) == "bad" and applied > 0 \
                    and abs(math.log(applied)) < 0.5 * abs(math.log(relative)):
                reading["misscaled"][name] = {"relative": relative, "applied": applied}
                reading["off"].pop(name, None)
    return reading


def _unfixed_text(cases: list[tuple[str, dict]]) -> str:
    """Sentence for an antenna that looks off in amplitude on ``[(source, values)]`` but was not corrected."""
    parts = []
    for source, values in cases:
        trust = [f"uncertainty {values['sigma']:.0%}"] if values["sigma"] is not None else []
        trust += [f"probability that a correction is needed {values['p']:.2f}"] if values["p"] is not None else []
        trusted = f" ({', '.join(trust)})" if trust else ""
        parts.append(f"on {source} an unconstrained gscale asks for x{values['naive']:.{GAIN_DECIMALS}f} but "
                     f"x{values['applied']:.{GAIN_DECIMALS}f} was applied{trusted}")
    return ("seems to be off in amplitude but could not be fixed, as neither the other antennas nor another "
            f"source constrain its gain: {'; '.join(parts)}; take a careful look at its amplitudes")


def _misscaled_text(cases: list[tuple[str, dict]]) -> str:
    """Sentence for an antenna whose amplitude scale is off against the rest of the array on ``[(source, values)]``."""
    wanted = ", ".join(f"x{values['relative']:.{GAIN_DECIMALS}f} on {source}" for source, values in cases)
    applied = max((values["applied"] for _, values in cases), key=lambda gain: abs(math.log(gain)))
    return (f"amplitude scale looks wrong against the rest of the array: relative to the array median, an "
            f"unconstrained gscale asks for {wanted}, but at most x{applied:.{GAIN_DECIMALS}f} was applied (the "
            "self-calibration keeps its corrections near the a-priori calibration); check the a-priori amplitude "
            "calibration of this station (ANTAB Tsys, DPFU, FT)")


def gain_comments(context: dict) -> list[dict]:
    """Amplitude corrections of the self-calibration that deserve a remark, one line per antenna and kind.

    * the median correction of an antenna beyond :data:`GAIN_WARN_DEVIATION` (warning) or
      :data:`GAIN_BAD_DEVIATION` (problem), over every self-calibrated source;
    * single subbands beyond :data:`GAIN_BAD_DEVIATION` when the median is not;
    * antennas whose amplitude scale is off against the rest of the array and that the self-calibration
      did not correct (``misscaled`` of :func:`_entry_gains`): a problem, on every source it shows on;
    * antennas that look off but were not corrected (``off`` of :func:`_entry_gains`). When that is true of
      at least half the antennas of a source (and of :data:`MIN_UNFIXED_FOR_SOURCE`), the source is
      reported as poorly constrained instead of every antenna; and an antenna whose gain another source
      pins down (``trusted``) is left alone, since it is the source, not the antenna, that is unconstrained;
    * the sentences of the Bayesian report about the source as a whole, and corrections never applied.
    """
    readings = [(entry, _entry_gains(entry)) for entry in context.get("gain_corrections") or []]
    medians: dict[str, list[tuple[str, float, str]]] = {}
    outliers: dict[str, list[tuple[str, str, float]]] = {}
    unfixed: dict[str, list[tuple[str, dict]]] = {}
    misscaled: dict[str, list[tuple[str, dict]]] = {}
    comments = []
    for entry, reading in readings:
        source = entry.get("source", "")
        for name, (median, level) in reading["medians"].items():
            medians.setdefault(name, []).append((source, median, level))
        for name, values in reading["outliers"].items():
            outliers.setdefault(name, []).extend((source, label, gain) for label, gain in values)
        for name, values in reading["misscaled"].items():
            misscaled.setdefault(name, []).append((source, values))
        off, solved = reading["off"], reading["solved"]
        if len(off) >= MIN_UNFIXED_FOR_SOURCE and len(off) >= 0.5 * len(solved):
            wanted = ", ".join(f"{name} x{values['naive']:.{GAIN_DECIMALS}f}" for name, values in off.items())
            blamed = list(reading["misscaled"])
            cause = (f" The common offset is most likely the model following {' and '.join(blamed)}, whose own "
                     f"{'scale is' if len(blamed) == 1 else 'scales are'} off against the rest of the array (see "
                     f"{'that antenna' if len(blamed) == 1 else 'those antennas'})." if blamed else "")
            comments.append(make_comment(
                "warning", "calibration", data={"kind": "selfcal", "source": source},
                text=f"The amplitude self-calibration of {source} is poorly constrained: an "
                f"unconstrained gscale would change {len(off)} of its {len(solved)} antennas by more than "
                f"{GAIN_WARN_DEVIATION:.0%} ({wanted}), but the corrections applied stayed within "
                f"{GAIN_UNFIXED_DEVIATION:.0%} of 1.{cause} Treat the amplitude scale of this source, and its "
                "model, with care."))
        else:
            for name, values in off.items():
                if not any(name in other["trusted"] for _, other in readings if other is not reading):
                    unfixed.setdefault(name, []).append((source, values))
        if entry.get("applied") is False:
            comments.append(make_comment("warning", "calibration", f"The amplitude corrections of {source} were "
                                         "computed but not applied to the data.",
                                         data={"kind": "selfcal", "source": source}))
        findings = _general_findings(entry)
        if findings:
            comments.append(make_comment("note", "calibration",
                                         f"Amplitude self-calibration of {source}: {'; '.join(findings)}.",
                                         data={"kind": "selfcal", "source": source}))
    for name, values in medians.items():
        large = any(level == "bad" for _, _, level in values)
        listed = ", ".join(f"x{median:.{GAIN_DECIMALS}f} on {source}" for source, median, _ in values)
        lead = "required a large amplitude correction" if large else \
            f"amplitude correction beyond {GAIN_WARN_DEVIATION:.0%}"
        comments.append(make_comment("problem" if large else "warning", "antenna",
                                     f"{lead}: {listed} (median over its subbands)", name))
    for name, values in outliers.items():
        listed = ", ".join(f"{label} x{gain:.{GAIN_DECIMALS}f} on {source}"
                           for source, label, gain in values[:MAX_SUBBANDS_LISTED])
        more = f" and {len(values) - MAX_SUBBANDS_LISTED} more" if len(values) > MAX_SUBBANDS_LISTED else ""
        comments.append(make_comment("warning", "antenna", f"single subbands needed a correction beyond "
                                     f"{GAIN_BAD_DEVIATION:.0%} although the median did not: {listed}{more}", name))
    for name, cases in unfixed.items():
        comments.append(make_comment("warning", "antenna", _unfixed_text(cases), name))
    for name, cases in misscaled.items():
        comments.append(make_comment("problem", "antenna", _misscaled_text(cases), name))
    return comments


# -- rules: results ----------------------------------------------------------------------------

def flagging_comments(context: dict) -> list[dict]:
    """Share of the cross-correlation data that is flagged, with and without the dead antennas."""
    flagging = context.get("flagging") or {}
    fraction = _finite(flagging.get("fraction"))
    if fraction is None:
        return []
    excluding = flagging.get("excluding_dead") or {}
    without, excluded = _finite(excluding.get("fraction")), list(excluding.get("excluded_antennas") or [])
    text = f"{fraction:.1%} of the cross-correlation data is flagged"
    if without is not None and excluded:
        text += f" ({without:.1%} without the antennas that have no usable signal: {join_names(excluded)})"
    effective = without if without is not None and excluded else fraction
    data = {"kind": "flagging", "fraction": fraction, "without": without if excluded else None, "excluded": excluded}
    return [make_comment("warning" if effective >= HIGH_FLAGGED_FRACTION else "note", "result", text + ".",
                         data=data)]


def _search_text(search: dict, with_peak: bool) -> str:
    """Words for the result of the source search in the wide dirty map.

    ``with_peak`` adds the peak of that map: worth saying only when no CLEAN image statistics are on record.
    """
    snr, fov = _finite(search.get("snr")), _finite(search.get("fov_mas"))
    if not search.get("detected"):
        field = f"the {fov:.0f} mas search map" if fov is not None else "the search map"
        return f"not detected in {field}" + (f" (brightest peak at {snr:.1f} sigma)" if snr is not None else "")
    offset, beams = _finite(search.get("offset_mas")) or 0.0, _finite(search.get("offset_beams")) or 0.0
    peak = _finite(search.get("peak"))
    text = f"detected at {snr:.0f} sigma in the search map" if snr is not None else "detected in the search map"
    text += f" (dirty-map peak {format_flux(peak)}/beam)" if with_peak and peak is not None else ""
    centred = offset < CENTRED_OFFSET_MAS
    text += ", at the phase centre" if centred else f", {offset:.1f} mas ({beams:.1f} beams) from the phase centre"
    return text + (" and re-centred on the peak" if search.get("recentred") else "")


def source_comments(context: dict) -> list[dict]:
    """What came out per source: the search result and the statistics of its CLEAN image (robust 0 if made)."""
    comments = []
    for entry in context.get("source_results") or []:
        role = entry.get("role") or "other"
        parts, severity = [], "note"
        search = entry.get("search")
        images = [image for image in entry.get("images") or [] if _finite(image.get("peak")) is not None]
        data = {"kind": "source", "source": entry.get("source", ""), "role": role,
                "detected": bool(search.get("detected")) if search else None, "snr": None, "snr_label": "",
                "flux": None, "flux_label": "", "offset_mas": _finite((search or {}).get("offset_mas"))}
        if search:
            parts.append(_search_text(search, with_peak=not images))
            if not search.get("detected"):
                severity = "warning"
            data.update(snr=_finite(search.get("snr")), snr_label="peak SNR in the search map")
            if _finite(search.get("peak")) is not None:
                data.update(flux=_finite(search.get("peak")), flux_label="peak in the search map (Jy/beam)")
        if images:
            image = next((i for i in images if _finite(i.get("robust")) == 0.0), images[0])
            peak, rms, dynamic = _finite(image["peak"]), _finite(image.get("rms")), _finite(image.get("dynamic_range"))
            if dynamic is not None:
                data.update(snr=dynamic, snr_label=f"dynamic range (robust {float(image.get('robust') or 0.0):g})")
            model_flux = _finite(image.get("model_flux"))
            if model_flux is not None and model_flux > 0:
                data.update(flux=model_flux, flux_label="total flux density of the model (Jy)")
            elif peak is not None:
                data.update(flux=peak, flux_label="image peak (Jy/beam)")
            text = f"CLEAN image (robust {float(image.get('robust') or 0.0):g}): peak {format_flux(peak)}/beam"
            text += f", rms {format_flux(rms)}/beam" if rms is not None else ""
            if dynamic is not None:
                text += f", dynamic range {dynamic:.0f}"
                if role in CALIBRATOR_ROLES and dynamic < LOW_CALIBRATOR_DYNAMIC_RANGE:
                    text += " - low for a calibrator, check its calibration and model"
                    severity = "warning"
            parts.append(text)
        if parts:
            comments.append(make_comment(severity, "result",
                                         f"{entry.get('source', '')} ({role}): {'; '.join(parts)}.", data=data))
    return comments


def table_comments(context: dict) -> list[dict]:
    """Calibration tables whose median solution SNR is poor (tables without a measured SNR are skipped)."""
    comments = []
    for table in context.get("caltables") or []:
        snr = _finite(table.get("snr"))
        if snr is not None and 0.0 < snr < POOR_TABLE_SNR:
            comments.append(make_comment("warning", "calibration", f"The {table.get('cal_type', '?')} table (step "
                                         f"{table.get('step', '?')}) has a low median solution SNR: {snr:.1f}.",
                                         data={"kind": "table", "table": str(table.get("cal_type", "?")),
                                               "snr": snr}))
    return comments


def step_comments(context: dict) -> list[dict]:
    """Pipeline steps recorded as failed."""
    comments = []
    for name, entry in (context.get("steps") or {}).items():
        if (entry or {}).get("status") == "failed":
            error = str(entry.get("error") or "").strip()
            comments.append(make_comment("problem", "pipeline",
                                         f"Step {name} failed" + (f": {error}" if error else ".")))
    return comments


# -- rules: warnings recorded by the pipeline --------------------------------------------------

def _warning_antenna(message: str, names: set[str]) -> str:
    """Antenna a pipeline warning is about ("" when it does not clearly name one of ``names``)."""
    match = re.match(r"^antenna (\S+?)[ ,:]", message) or re.match(r"^(\S+) (?:has|recorded|lost|shows) ", message)
    return match.group(1) if match and match.group(1) in names else ""


def pipeline_comments(context: dict, reported: set[str]) -> list[dict]:
    """The warnings the pipeline recorded, each with the time it was first seen.

    Parameters
    ----------
    context : dict
        Dashboard context; ``run_warnings`` is the list :func:`load_run_warnings` returns.
    reported : set of str
        Antennas the other rules already commented on. A warning of a kind those rules word better
        (:data:`DUPLICATED_WARNINGS`) is dropped for such an antenna.

    Returns
    -------
    list of dict
        At most :data:`MAX_PIPELINE_WARNINGS` comments (anomalies first), plus a count of the rest. A
        warning that clearly names one antenna is filed under that antenna.
    """
    names = {a["name"] for a in context.get("antennas") or []}
    comments = []
    for warning in context.get("run_warnings") or []:
        message = str(warning.get("message") or "").strip()
        antenna = _warning_antenna(message, names)
        if not message or (antenna in reported and any(marker in message for marker in DUPLICATED_WARNINGS)):
            continue
        text = re.sub(r"^antenna \S+ ", "", message) if antenna and message.startswith("antenna ") else message
        text = text[len(antenna) + 1:] if antenna and text.startswith(antenna + " ") else text
        comments.append(make_comment("problem" if warning.get("anomaly") else "warning",
                                     "antenna" if antenna else "pipeline", text, antenna,
                                     when=str(warning.get("first_seen_utc") or "")))
    comments.sort(key=lambda c: SEVERITIES.index(c["severity"]))
    hidden = len(comments) - MAX_PIPELINE_WARNINGS
    if hidden > 0:
        comments = comments[:MAX_PIPELINE_WARNINGS]
        comments.append(make_comment("note", "pipeline", f"{hidden} more warning{'s' if hidden != 1 else ''} of the "
                                     f"pipeline are not listed here; all of them are in {RUN_WARNINGS_FILENAME}."))
    return comments


# -- assembly ----------------------------------------------------------------------------------

def build_comments(context: dict) -> list[dict]:
    """Every general comment of a reduction, from the dashboard context alone.

    Parameters
    ----------
    context : dict
        JSON-safe context of :func:`vlbipy.dashboard.collect`. Missing keys simply silence the rules that
        need them, so an old or partial context is fine.

    Returns
    -------
    list of dict
        Comments (see the module docstring): calibration and results first, then the antennas in the order
        of the context with one closing line naming those with nothing to report, then the pipeline. Within
        a group and an antenna the most serious comments come first.
    """
    general = (instrumental_delay_comments(context) + gain_comments(context) + flagging_comments(context)
               + source_comments(context) + table_comments(context) + step_comments(context)
               + data_comments(context) + band_comments(context) + gap_comments(context)
               + fringe_comments(context) + polarization_comments(context))
    reported = {c["antenna"] for c in general if c["category"] == "antenna" and c["antenna"]}
    everything = general + pipeline_comments(context, reported)
    order = {a["name"]: index for index, a in enumerate(context.get("antennas") or [])}
    rank = SEVERITIES.index
    shared = sorted((c for c in everything if c["category"] in ("calibration", "result")),
                    key=lambda c: rank(c["severity"]))
    stations = sorted((c for c in everything if c["category"] == "antenna"),
                      key=lambda c: (order.get(c["antenna"], len(order)), rank(c["severity"])))
    pipeline = sorted((c for c in everything if c["category"] == "pipeline"), key=lambda c: rank(c["severity"]))
    commented = {c["antenna"] for c in stations}
    quiet = [name for name in order if name not in commented]
    if quiet:
        stations.append(make_comment("note", "antenna", f"Nothing to report for: {', '.join(quiet)}."))
    return shared + stations + pipeline


def group_comments(comments: list[dict]) -> dict:
    """Arrange a comment list the way it is displayed.

    Returns
    -------
    dict
        ``general`` (calibration and result comments), ``antennas`` (``[(station, [comments])]`` in order of
        first appearance), ``antenna_notes`` (antenna comments about no single station, e.g. the closing
        "nothing to report" line) and ``pipeline``.
    """
    groups = {"general": [], "antennas": [], "antenna_notes": [], "pipeline": []}
    by_station: dict[str, list[dict]] = {}
    for comment in comments or []:
        category = comment.get("category")
        if category == "antenna" and comment.get("antenna"):
            by_station.setdefault(comment["antenna"], []).append(comment)
        elif category == "antenna":
            groups["antenna_notes"].append(comment)
        elif category == "pipeline":
            groups["pipeline"].append(comment)
        else:
            groups["general"].append(comment)
    groups["antennas"] = list(by_station.items())
    return groups


def _text_line(lead: str, comment: dict, width: int) -> list[str]:
    """One comment as wrapped text lines: ``lead``, the severity label, then the sentence."""
    label = SEVERITY_LABELS.get(comment.get("severity"), "note")
    when = f" [first seen {comment['when']} UTC]" if comment.get("when") else ""
    head = f"{lead}{label:<7}  "
    return textwrap.wrap(str(comment.get("text", "")) + when, width=width, initial_indent=head,
                         subsequent_indent=" " * len(head), break_long_words=False, break_on_hyphens=False) or [head]


def comments_text(comments: Optional[list[dict]], project_code: str, width: int = TEXT_WIDTH) -> str:
    """The "General Comments" block of ``summary.md``, in its fixed-width plain-text style.

    Parameters
    ----------
    comments : list of dict or None
        Result of :func:`build_comments`. None means the pipeline has not run yet (import time): the block
        then only says when it gets filled in.
    project_code : str
        Project code for the heading.
    width : int
        Width of the ruler and of the wrapped lines.

    Returns
    -------
    str
        Multi-line block without a trailing newline.
    """
    lines = [f"General Comments for {project_code}", "=" * width]
    if comments is None:
        return "\n".join(lines + ["  Not written yet: this section is filled in once the pipeline has run "
                                  "(or by `vlbipy report`)."])
    groups = group_comments(comments)
    if not any(groups.values()):
        return "\n".join(lines + ["  Nothing to report."])
    if groups["general"]:
        lines.append("Calibration and results")
        for comment in groups["general"]:
            lines.extend(_text_line("  ", comment, width))
    if groups["antennas"] or groups["antenna_notes"]:
        lines.append("Antennas")
        name_width = max([len(name) for name, _ in groups["antennas"]] + [2])
        for name, items in groups["antennas"]:
            for index, comment in enumerate(items):
                lines.extend(_text_line(f"  {name if index == 0 else '':<{name_width}}  ", comment, width))
        for comment in groups["antenna_notes"]:
            lines.extend(textwrap.wrap(str(comment.get("text", "")), width=width, initial_indent="  ",
                                       subsequent_indent="    "))
    if groups["pipeline"]:
        lines.append("Warnings raised by the pipeline")
        for comment in groups["pipeline"]:
            lines.extend(_text_line("  ", comment, width))
    return "\n".join(lines)


# -- the warnings of the pipeline, kept between processes --------------------------------------

def _read_warning_file(path: Path) -> dict[str, str]:
    """``{message: first-seen UTC}`` stored in ``path``; empty when it is missing or unreadable."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("could not read {} ({})", path, exc)
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _belongs_to(message: str, project_code: str) -> bool:
    """Whether a recorded warning starts with the prefix ``"<project_code>: "`` (after the anomaly mark)."""
    text = message[len(ANOMALY_PREFIX):] if message.startswith(ANOMALY_PREFIX) else message
    return text.startswith(f"{project_code}: ")


def merge_run_warnings(work_dir, messages: list[str], other_projects=()) -> dict[str, str]:
    """Add the warnings a process recorded to ``<work_dir>/.run_warnings.json``; return what is stored.

    The file accumulates over runs: a resumed run skips the steps that raised a warning, so the warning
    would otherwise vanish from the report. A message already stored keeps its first-seen time.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Project directory.
    messages : list of str
        Messages of the warning collector (``logging_utils.warnings.summary()``).
    other_projects : iterable of str
        Project codes of the same run; a message prefixed with one of them belongs to that project and is
        left out.

    Returns
    -------
    dict
        ``{message: first-seen UTC "YYYY-MM-DD HH:MM:SS"}`` as written. Nothing is written when there is
        nothing to store.
    """
    path = Path(work_dir) / RUN_WARNINGS_FILENAME
    stored = _read_warning_file(path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    added = False
    for message in messages:
        if message not in stored and not any(_belongs_to(message, other) for other in other_projects):
            stored[message] = now
            added = True
    if added:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not write {} ({}); the warnings of this run will be missing from later reports",
                           path, exc)
    return stored


def load_run_warnings(work_dir, project_code: str) -> list[dict]:
    """The warnings kept in ``<work_dir>/.run_warnings.json``, ready for the dashboard context.

    Returns
    -------
    list of dict
        ``{"message", "anomaly", "first_seen_utc"}`` per warning, oldest first, with the anomaly mark and the
        ``"<project_code>: "`` prefix removed from the message. Empty when there is no file.
    """
    warnings = []
    for message, seen in _read_warning_file(Path(work_dir) / RUN_WARNINGS_FILENAME).items():
        anomaly = message.startswith(ANOMALY_PREFIX)
        text = message[len(ANOMALY_PREFIX):] if anomaly else message
        text = text[len(project_code) + 2:] if project_code and text.startswith(f"{project_code}: ") else text
        warnings.append({"message": text, "anomaly": anomaly, "first_seen_utc": seen})
    return sorted(warnings, key=lambda w: w["first_seen_utc"])
