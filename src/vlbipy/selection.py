"""Antenna and scan selection for instrumental calibration (pure logic, no CASA).

The single-band delay and bandpass solutions are derived from a small amount of
data — ideally one scan on a bright fringe finder — so *which* data is picked
decides the quality of everything downstream. Both choices are made from the
per-scan fringe SNR survey (:class:`~vlbipy.models.ScanSNRSurvey`) plus the
subband participation recorded in the metadata:

* :func:`select_antennas` keeps antennas that detected fringes, ranked by SNR
  (partial-band antennas included: their solutions cover the subbands they have).
* :func:`select_calibration_scans` finds the scan where every selected antenna
  was detected, or — when no single scan covers the array — the smallest set of
  scans that does, with at least one antenna shared between them so the separate
  solutions can be tied to a common reference.

Scans on the source the array detects best are preferred (:func:`_source_priority`):
the solve needs signal-to-noise far above the survey's detection threshold, so a
marginal gain in coverage on a weak source is worth less than a hundred-fold
stronger signal on a bright one.

The raw "time evolution" figures of the report are planned here too
(:func:`plan_time_evolution`): which reference antennas and subbands are needed so
that every antenna and every scan appears in at least one figure.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .logging_utils import get_logger
from .models import ObsMetadata, ScanSNRSurvey

logger = get_logger()

#: Detection threshold in fringe SNR: below this an antenna is not usable for calibration.
DEFAULT_MIN_SNR = 7.0


#: Fallback reference-antenna priority: the most sensitive dishes of each network, in
#: order. Used when the user configures none — DISH_DIAMETER is frequently 0 in
#: FITS-IDI-derived measurement sets, so dish size alone cannot rank the array, and
#: "whichever antenna the table happens to list first" is arbitrary.
REFANT_PRIORITY = ("EF", "YY", "AT", "PA", "PT", "GB", "YS", "O8", "LA", "MP", "MC", "TR", "WB")


def rank_reference_antennas(metadata: ObsMetadata, requested: str = "") -> list[str]:
    """Return candidate reference antennas, best first.

    Order: antennas the caller asked for, then :data:`REFANT_PRIORITY`, then the
    rest by how many scans they took part in, and last the antennas whose subbands
    all fall outside the usable band — they have no baseline, so they can reference
    nothing. They are kept (ranked last) rather than dropped so the chain is never empty.
    """
    observed = [a.name for a in metadata.observed_antennas] or list(metadata.antennas)
    upper = {name.upper(): name for name in observed}
    ordered: list[str] = []
    for name in (n.strip() for n in str(requested).split(",")):
        match = upper.get(name.upper())
        if match and match not in ordered:
            ordered.append(match)
    for name in REFANT_PRIORITY:
        match = upper.get(name)
        if match and match not in ordered:
            ordered.append(match)
    for name in sorted(observed, key=lambda n: -metadata.antennas[n].n_scans):
        if name not in ordered:
            ordered.append(name)
    usable = set(metadata.usable_subbands)
    if not usable:
        return ordered
    stranded = [n for n in ordered if metadata.antennas[n].subbands
                and not usable.intersection(metadata.antennas[n].subbands)]
    if stranded:
        logger.info("reference antenna: {} recorded no subband that has a baseline; ranked last",
                    ", ".join(stranded))
        ordered = [n for n in ordered if n not in stranded] + stranded
    return ordered


def select_antennas(survey: ScanSNRSurvey, metadata: ObsMetadata, *,
                    min_snr: float = DEFAULT_MIN_SNR, max_antennas: int = 0) -> list[str]:
    """Return the antennas usable for instrumental calibration, best first.

    An antenna qualifies when its median fringe SNR clears ``min_snr``. Antennas
    that recorded only some subbands are kept — heterogeneous arrays are common
    and their solutions simply cover the subbands they have. The survivors are
    ordered by median SNR, so the caller can take the head of the list as the
    reference-antenna preference.

    Parameters
    ----------
    survey : ScanSNRSurvey
        Per-scan fringe SNR survey over the calibrators.
    metadata : ObsMetadata
        Observation metadata; supplies subband participation per antenna.
    min_snr : float
        Minimum median fringe SNR for an antenna to be considered detected.
    max_antennas : int
        Keep at most this many antennas (0 = no limit).

    Returns
    -------
    list of str
        Antenna names ordered by decreasing median SNR (empty if none qualify).
    """
    # The reference antenna has no SNR of its own (its solutions are the sentinel it
    # is referenced against), so ranking alone would drop the one antenna every other
    # solution is tied to. It is detected by definition: put it first.
    reference = [name for name in survey.refant_names if name in metadata.antennas]
    ranked = [(name, float("inf")) for name in reference] + [
        entry for entry in survey.rank_antennas() if entry[0] not in reference]
    selected: list[str] = []
    rejected: list[str] = []
    for name, median_snr in ranked:
        if name not in metadata.antennas:
            continue
        if median_snr < min_snr:
            rejected.append(f"{name} (SNR {median_snr:.0f} < {min_snr:.0f})")
            continue
        selected.append(name)
    if max_antennas and len(selected) > max_antennas:
        selected = selected[:max_antennas]
    logger.info("antenna selection: kept {} of {} ({})", len(selected), len(ranked),
                ", ".join(selected) or "none")
    if rejected:
        logger.info("antenna selection: rejected {}", "; ".join(rejected))
    return selected


def detected_antennas_per_scan(survey: ScanSNRSurvey, antennas: list[str],
                               min_snr: float = DEFAULT_MIN_SNR) -> dict[int, set[str]]:
    """Return ``{scan_number: {antennas detected above min_snr}}`` restricted to ``antennas``.

    An antenna counts as detected in a scan when it clears ``min_snr`` in *every*
    polarization that has a solution there — a one-handed detection is not a
    usable basis for an instrumental-delay solution.
    """
    wanted = set(antennas)
    detected: dict[int, set[str]] = {}
    for index, scan in enumerate(survey.scan_numbers):
        found = set()
        reference = set(survey.refant_names)
        for column, name in enumerate(survey.antennas):
            if name not in wanted:
                continue
            if name in reference:
                # Every solution in the scan is referenced to it, so if the scan
                # produced anything at all the reference antenna was detected.
                if any(survey.snr[pol][index][c] == survey.snr[pol][index][c]
                       for pol in survey.polarizations for c in range(len(survey.antennas))):
                    found.add(name)
                continue
            values = [survey.snr[pol][index][column] for pol in survey.polarizations]
            usable = [v for v in values if v == v]  # drop nan
            if usable and min(usable) >= min_snr:
                found.add(name)
        detected[scan] = found
    return detected


def detected_hands_per_scan(survey: ScanSNRSurvey, antennas: list[str],
                            min_snr: float = DEFAULT_MIN_SNR) -> dict[int, set[tuple[str, str]]]:
    """Return ``{scan_number: {(antenna, polarization) detected above min_snr}}`` restricted to ``antennas``.

    The unit is one polarization of one antenna, not the antenna: a station with
    a dead receiver channel, or one that recorded a polarization only part of
    the time, is common in heterogeneous arrays and must neither be rejected
    for the hand it lacks nor lose the hand it has.
    """
    wanted = set(antennas)
    reference = set(survey.refant_names)
    detected: dict[int, set[tuple[str, str]]] = {}
    for index, scan in enumerate(survey.scan_numbers):
        found: set[tuple[str, str]] = set()
        for pol in survey.polarizations:
            row = survey.snr[pol][index]
            solved = any(value == value for value in row)
            for column, name in enumerate(survey.antennas):
                if name not in wanted:
                    continue
                if name in reference:
                    # Every solution of this hand is referenced to it, so if the scan
                    # produced any the reference antenna was detected in that hand.
                    if solved:
                        found.add((name, pol))
                elif row[column] == row[column] and row[column] >= min_snr:
                    found.add((name, pol))
        detected[scan] = found
    return detected


def _antenna_priority(survey: ScanSNRSurvey) -> list[str]:
    """Antennas best-first: the survey's reference antenna(s), then by median fringe SNR."""
    ordered = list(survey.refant_names)
    ordered += [name for name, _ in survey.rank_antennas() if name not in ordered]
    return ordered


def _source_priority(survey: ScanSNRSurvey) -> dict[int, int]:
    """Return ``{scan_number: source rank}``, higher for a scan on a better-detected source.

    The instrumental solve needs real signal-to-noise, not a bare detection: its
    ``minsnr`` is far above the detection threshold the survey is read with. A
    weak source where every antenna sits just above that threshold must therefore
    not be preferred over a bright one that is a hundred times stronger merely
    because it has one more antenna in a scan — the solve on the weak source
    returns nothing at all, which is the "no solutions with high enough SNR"
    failure this ranking prevents. The missing antenna is picked up by a later
    stage instead. Sources are ranked by :meth:`~vlbipy.models.ScanSNRSurvey.rank_sources`,
    which already counts an antenna a source never detected as zero, so a source
    only a few baselines see cannot win on brightness alone.
    """
    rank = {source: -index for index, (source, _, _) in enumerate(survey.rank_sources())}
    unknown = -len(rank) - 1
    return {scan: rank.get(source, unknown) for scan, source in survey.source_of_scan().items()}


def plan_sbd_stages(survey: ScanSNRSurvey, antennas: list[str], *, min_snr: float = DEFAULT_MIN_SNR,
                    sources: list[str] | None = None, metadata: ObsMetadata | None = None) -> list[dict]:
    """Plan the single-band-delay solve as a chain of single-scan stages.

    The instrumental delay is constant in time, so it must come from *one*
    scan: several scans give several independent solutions whose phases are
    not continuous once applied. Stage 1 is the best scan detecting the most
    antenna polarizations (ideally all of them). Only when some remain
    uncovered are further scans added, each detecting at least one
    already-solved antenna that becomes that stage's reference, so its
    solutions can be re-based onto the first stage's reference. As few scans
    as possible are used. Antennas that no remaining scan ties to the solved
    ones (a sub-array that never observed together with the rest, as in
    staggered global VLBI) start a separate chain on their own reference; such
    a stage is marked ``"independent": True`` and is not re-based.

    Coverage is counted per polarization of each antenna (see
    :func:`detected_hands_per_scan`): an antenna whose second polarization only
    shows up in another scan gets that hand from a later stage, and so appears
    in more than one stage.

    Parameters
    ----------
    survey : ScanSNRSurvey
        Per-scan fringe SNR survey.
    antennas : list of str
        Antennas that must be covered (from :func:`select_antennas`).
    min_snr : float
        Detection threshold.
    sources : list of str, optional
        Restrict to scans on these sources (e.g. the fringe finders).
    metadata : ObsMetadata, optional
        When given, an antenna only counts as detected in a scan it took part in
        (the survey marks its reference antenna detected wherever any solution exists).

    Returns
    -------
    list of dict
        ``[{"scan": int, "antennas": [solved here], "refant": str}, ...]`` in
        solve order. Empty when nothing is detected at all.
    """
    if not antennas:
        return []
    detected = detected_hands_per_scan(survey, antennas, min_snr)
    if metadata is not None:
        present = {scan.scan_number: set(scan.antennas) for scan in metadata.scans}
        detected = {scan: {hand for hand in hands if scan not in present or hand[0] in present[scan]}
                    for scan, hands in detected.items()}
    if sources:
        allowed = {scan for scan, source in zip(survey.scan_numbers, survey.scan_sources)
                   if source in sources}
        detected = {scan: hands for scan, hands in detected.items() if scan in allowed}
    detected = {scan: hands for scan, hands in detected.items() if hands}
    if not detected:
        return []
    required = set().union(*detected.values())
    priority = _antenna_priority(survey)
    by_source = _source_priority(survey)

    def rank(name: str) -> int:
        return priority.index(name) if name in priority else len(priority)

    def score(scan: int) -> float:
        value = survey.median_snr(scan_number=scan)
        return value if value == value else 0.0

    def hands_of(name: str, hands: set) -> set[str]:
        return {pol for antenna, pol in hands if antenna == name}

    stages: list[dict] = []
    covered: set[tuple[str, str]] = set()
    remaining = dict(detected)
    while remaining and not required <= covered:
        references: dict[int, list[str]] = {}
        for scan, found in remaining.items():
            gained_pols = {pol for _, pol in found - covered}
            names = {antenna for antenna, _ in found}
            if not stages:
                # The first reference only has to be there in every hand being solved.
                usable = [n for n in names if gained_pols <= hands_of(n, found)]
            else:
                # A later one must already be solved in those hands, to tie the chain.
                usable = [n for n in names if gained_pols <= hands_of(n, found) & hands_of(n, covered)]
            if usable and found - covered:
                references[scan] = sorted(usable, key=rank)
        independent = False
        if not references and stages:
            # Nothing left shares a solved antenna: a sub-array that never observed together
            # with the rest (staggered global VLBI). Its baselines never involve the solved
            # antennas, so its own reference serves just as well: start a separate chain.
            independent = True
            for scan, found in remaining.items():
                gained_pols = {pol for _, pol in found - covered}
                usable = [n for n in {antenna for antenna, _ in found} if gained_pols <= hands_of(n, found)]
                # One antenna on its own has no baseline: nothing to solve.
                if usable and len({antenna for antenna, _ in found - covered}) >= 2:
                    references[scan] = sorted(usable, key=rank)
        if not references:
            break
        # Best source first, then the widest coverage on it, then its strongest scan.
        best = max(references, key=lambda s: (by_source.get(s, 0), len(remaining[s] - covered), score(s)))
        gained = remaining[best] - covered
        gained_names = sorted({antenna for antenna, _ in gained}, key=rank)
        stage = {"scan": best, "antennas": gained_names, "refant": references[best][0]}
        if independent:
            stage["independent"] = True
        stages.append(stage)
        covered |= gained
        remaining.pop(best)

    n_antennas = len({antenna for antenna, _ in required})
    ranked_sources = [entry for entry in survey.rank_sources() if entry[0] in set(survey.scan_sources)]
    if len(ranked_sources) > 1:
        logger.info("scan selection: source ranking by array-wide fringe SNR: {}",
                    "; ".join(f"{name} ({snr:.0f} over {found} antenna(s))" for name, snr, found in ranked_sources))
    scan_source = survey.source_of_scan()
    if len(stages) == 1:
        logger.info("scan selection: scan {} on {} detects all {} antennas in every polarization they have "
                    "(median SNR {:.0f}); single-stage SBD", stages[0]["scan"],
                    scan_source.get(stages[0]["scan"], "?"), n_antennas, score(stages[0]["scan"]))
    else:
        logger.info("scan selection: no single scan covers the array; SBD in {} stage(s): {}", len(stages),
                    "; ".join(f"scan {s['scan']} on {scan_source.get(s['scan'], '?')} "
                              f"({','.join(s['antennas'])}; ref {s['refant']}{', separate sub-array' if s.get('independent') else ''})"
                              for s in stages))
    one_handed = sorted(name for name in {antenna for antenna, _ in required}
                        if len(hands_of(name, required)) < len(survey.polarizations))
    if one_handed:
        logger.warning("scan selection: detected in one polarization only: {}",
                       ", ".join(f"{name} ({'/'.join(sorted(hands_of(name, required)))})" for name in one_handed))
    missing = sorted(f"{antenna} {pol}" for antenna, pol in required - covered)
    missing += sorted(set(antennas) - {antenna for antenna, _ in required})
    if missing:
        logger.warning("scan selection: no scan detects {} above {:.0f} sigma; they will have no "
                       "instrumental-delay solution", ", ".join(missing), min_snr)
    return stages


def select_calibration_scans(survey: ScanSNRSurvey, antennas: list[str], *,
                             min_snr: float = DEFAULT_MIN_SNR, sources: list[str] | None = None,
                             metadata: ObsMetadata | None = None) -> list[int]:
    """Return the scan numbers of :func:`plan_sbd_stages`, in solve order."""
    return [stage["scan"] for stage in plan_sbd_stages(survey, antennas, min_snr=min_snr, sources=sources,
                                                       metadata=metadata)]


def without_weak_detections(survey: ScanSNRSurvey, weak: dict[int, list[str]], *,
                            min_snr: float = DEFAULT_MIN_SNR) -> tuple[ScanSNRSurvey, dict[int, list[str]]]:
    """Blank the detections of antennas in the scans where they are far below their own level.

    The fringe SNR of the survey says an antenna was detected, not that it was
    properly on source: see :func:`vlbipy.statistics.weak_scans_per_antenna`. A
    plan made from the survey returned here does not solve an antenna on such a
    scan. A reference antenna of the survey that is weak takes its whole scan
    out, since every solution of that scan hangs on it. An antenna that no other
    scan detects keeps its detection: a poor solution is better than none.

    Parameters
    ----------
    survey : ScanSNRSurvey
        The survey the plan would be made from.
    weak : dict
        ``{scan: [antennas far below their best scan]}``.
    min_snr : float
        Detection threshold, to tell whether another scan detects the antenna.

    Returns
    -------
    (ScanSNRSurvey, dict)
        A copy of the survey with those values set to ``nan`` (the input is left
        untouched) and ``{scan: [antennas]}`` of the detections actually removed.
    """
    from dataclasses import replace

    nan = float("nan")
    rows = {scan: row for row, scan in enumerate(survey.scan_numbers)}
    columns = {name: column for column, name in enumerate(survey.antennas)}

    def detected(name: str, row: int) -> bool:
        return any(survey.snr[pol][row][columns[name]] >= min_snr for pol in survey.polarizations)

    snr = {pol: [list(values) for values in matrix] for pol, matrix in survey.snr.items()}
    removed: dict[int, list[str]] = {}
    for scan, names in sorted(weak.items()):
        row = rows.get(int(scan))
        if row is None:
            continue
        for name in names:
            if name not in columns or not (detected(name, row) or name in survey.refant_names):
                continue
            elsewhere = [other for other_scan, other in rows.items()
                         if other != row and name not in weak.get(other_scan, []) and detected(name, other)]
            if not elsewhere and name not in survey.refant_names:
                logger.warning("scan selection: {} is weak in scan {} but no other scan detects it; solving it "
                               "there all the same", name, scan)
                continue
            blanked = range(len(survey.antennas)) if name in survey.refant_names else [columns[name]]
            for pol in snr:
                for column in blanked:
                    snr[pol][row][column] = nan
            removed.setdefault(int(scan), []).append(name)
    return replace(survey, snr=snr), removed


def amend_sbd_stages(stages: list[dict], survey: ScanSNRSurvey, lost: dict[int, list[str]], *,
                     min_snr: float = DEFAULT_MIN_SNR, sources: list[str] | None = None,
                     metadata: ObsMetadata | None = None) -> list[dict]:
    """Give antennas whose stage scan has since been flagged another scan to be solved on.

    The plan of :func:`plan_sbd_stages` is fixed on the first pass, before the
    flagging that follows it. An antenna that turns out not to have been on source
    in its stage scan (detected, but at a few per cent of its amplitude) is then
    flagged there, and with the plan unchanged it would get no instrumental
    solution at all and lose *all* its data on apply.

    Antennas that still have their data keep their stage. Each lost one gets a
    further stage: the scan that detects the most lost antennas together with an
    antenna the plan still solves, which becomes that stage's reference.

    Parameters
    ----------
    stages : list of dict
        The plan in use (``[{"scan", "antennas", "refant"}, ...]``).
    survey : ScanSNRSurvey
        Per-scan fringe SNR survey to pick the new scans from.
    lost : dict
        ``{scan: [antennas with no usable data left in that scan]}``; an antenna is
        never given a scan it is listed under.
    min_snr, sources, metadata
        As in :func:`plan_sbd_stages`.

    Returns
    -------
    list of dict
        The amended plan. Antennas no other scan detects are simply left out (and
        reported). Empty when a stage lost its own reference antenna: the chain is
        broken and the caller has to plan afresh.
    """
    gone = {(int(scan), name) for scan, names in lost.items() for name in names}
    if any((int(stage["scan"]), stage["refant"]) in gone for stage in stages):
        return []
    kept = []
    for stage in stages:
        names = [name for name in stage["antennas"] if (int(stage["scan"]), name) not in gone]
        if names:
            kept.append({**stage, "antennas": names})
    solved = {name for stage in kept for name in stage["antennas"]}
    waiting = {name for _, name in gone} - solved
    detected = detected_hands_per_scan(survey, sorted(solved | waiting), min_snr)
    if metadata is not None:
        present = {scan.scan_number: set(scan.antennas) for scan in metadata.scans}
        detected = {scan: {hand for hand in hands if scan not in present or hand[0] in present[scan]}
                    for scan, hands in detected.items()}
    if sources:
        allowed = {scan for scan, source in zip(survey.scan_numbers, survey.scan_sources) if source in sources}
        detected = {scan: hands for scan, hands in detected.items() if scan in allowed}
    names_in = {scan: {antenna for antenna, _ in hands if (scan, antenna) not in gone}
                for scan, hands in detected.items()}
    priority = _antenna_priority(survey)
    by_source = _source_priority(survey)

    def rank(name: str) -> int:
        return priority.index(name) if name in priority else len(priority)

    def score(scan: int) -> float:
        value = survey.median_snr(scan_number=scan)
        return value if value == value else 0.0

    amended = list(kept)
    while waiting:
        # A scan helps when it detects a waiting antenna and one already solved (its reference).
        helpful = {scan: names for scan, names in names_in.items() if names & waiting and names & solved}
        if not helpful:
            break
        best = max(helpful, key=lambda s: (by_source.get(s, 0), len(helpful[s] & waiting), score(s)))
        gained = sorted(helpful[best] & waiting, key=rank)
        same_scan = next((stage for stage in amended if stage["scan"] == best), None)
        if same_scan is not None:
            # Already a stage of the plan: solve them there, against that stage's reference.
            same_scan["antennas"] = same_scan["antennas"] + gained
        else:
            amended.append({"scan": best, "antennas": gained, "refant": sorted(helpful[best] & solved, key=rank)[0]})
        solved |= set(gained)
        waiting -= set(gained)
        names_in.pop(best)
    described = "; ".join(f"scan {s['scan']} ({','.join(s['antennas'])}; ref {s['refant']})" for s in amended)
    logger.info("scan selection: re-planned after flagging; SBD in {} stage(s): {}", len(amended), described)
    if waiting:
        logger.warning("scan selection: no other scan detects {} above {:.0f} sigma together with a solved "
                       "antenna; they will have no instrumental-delay solution", ", ".join(sorted(waiting)), min_snr)
    return amended


# -- raw time-evolution figures ------------------------------------------------------------------

#: Most reference antennas and subbands a time-evolution plan holds (one figure per pair of them).
TIME_EVOLUTION_MAX_REFANTS = 3
TIME_EVOLUTION_MAX_SUBBANDS = 3
#: Lag coherence (:func:`lag_coherence`) from which a scan clearly shows fringes on a baseline ...
FRINGE_COHERENCE = 0.7
#: ... and below which, on every baseline to an antenna, the scan shows none. Noise alone reaches ~0.4
#: once in a thousand scans of 60 integrations, so the two thresholds are kept well apart.
NOISE_COHERENCE = 0.45
#: Pairs of consecutive integrations a baseline needs in a scan before its coherence is trusted.
MIN_COHERENCE_PAIRS = 30


def format_runs(numbers, max_runs: int = 4) -> str:
    """Write integers compactly, consecutive ones as a range: ``[2, 3, 4, 7]`` -> ``"2-4, 7"``.

    Parameters
    ----------
    numbers : iterable of int
        Scan or subband numbers, in any order; duplicates are ignored.
    max_runs : int
        Runs written out; the numbers of the remaining runs are only counted (``"... and 5 more"``).

    Returns
    -------
    str
        The compact text, ``""`` for no numbers.
    """
    runs: list[list[int]] = []
    for number in sorted({int(n) for n in numbers}):
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    texts = [str(first) if first == last else f"{first}-{last}" for first, last in runs]
    if len(texts) <= max_runs:
        return ", ".join(texts)
    hidden = sum(last - first + 1 for first, last in runs[max_runs:])
    return ", ".join(texts[:max_runs]) + f" and {hidden} more"


def _counted(noun: str, numbers) -> str:
    """``"scan 7"`` or ``"scans 2-4, 7"``: the noun in the right number, followed by :func:`format_runs`."""
    numbers = list(numbers)
    return f"{noun}{'' if len(set(numbers)) == 1 else 's'} {format_runs(numbers)}"


def _recorded_subbands(metadata: ObsMetadata) -> dict[str, set[int]]:
    """Subbands recorded by every antenna with data, in antenna-table order.

    An antenna whose participation was never determined (no subband listed) counts as having recorded
    all of them, as in :meth:`~vlbipy.models.ObsMetadata.antennas_per_subband`.
    """
    everything = set(range(metadata.freq_setup.n_subbands))
    return {antenna.name: (set(antenna.subbands) & everything) or set(everything)
            for antenna in metadata.observed_antennas}


def plan_time_evolution_subbands(metadata: ObsMetadata, max_subbands: int = TIME_EVOLUTION_MAX_SUBBANDS, *,
                                 reference: str = "", wanted=None) -> list[tuple[int, str]]:
    """Pick the subbands of the time-evolution figures so that every antenna with data is in one.

    The first is the usable subband recorded by the most antennas (ideally all of them; ties go to the
    one nearest the band centre, then to the lower index). While an antenna is in none of the chosen
    subbands, the subband recorded by the most of the missing antennas is added.

    Parameters
    ----------
    metadata : ObsMetadata
        Supplies the antennas with data and the subbands each recorded.
    max_subbands : int
        Most subbands to return.
    reference : str
        Reference antenna of the figures: only subbands it recorded are considered, since a figure
        shows the baselines to it. Empty considers every usable subband.
    wanted : iterable of str, optional
        The antennas that have to be shown (default: every antenna with data).

    Returns
    -------
    list of tuple
        ``[(subband, reason), ...]`` in the order chosen, the reason in plain words. Empty when no
        subband (of the reference antenna) has a baseline to a wanted antenna.
    """
    recorded = _recorded_subbands(metadata)
    usable = [subband for subband in metadata.usable_subbands
              if not reference or subband in recorded.get(reference, ())]
    restricted = len(usable) < len(metadata.usable_subbands)
    centre = (metadata.freq_setup.n_subbands - 1) / 2.0
    chosen: list[tuple[int, str]] = []
    missing = set(recorded if wanted is None else wanted) - {reference}
    while missing and len(chosen) < max_subbands:
        taken = [subband for subband, _ in chosen]
        left = [subband for subband in usable if subband not in taken]
        if not left:
            break
        best = max(left, key=lambda s: (sum(1 for n in missing if s in recorded[n]),
                                        sum(1 for n in recorded if s in recorded[n]), -abs(s - centre), -s))
        gained = [name for name in recorded if name in missing and best in recorded[name]]
        if not gained:
            break
        if not chosen:
            n_recording = sum(1 for name in recorded if best in recorded[name])
            reason = (f"subband {best} is recorded by all {len(recorded)} antennas with data"
                      if n_recording == len(recorded) else
                      f"subband {best} is recorded by {n_recording} of the {len(recorded)} antennas with data, "
                      f"the most of any subband{f' {reference} recorded' if restricted else ''}")
        else:
            reason = f"subband {best} because {', '.join(gained)} did not record {_counted('subband', taken)}"
        chosen.append((int(best), reason))
        missing -= set(gained)
    return chosen


def _reference_order(survey, antennas: list[str]) -> list[str]:
    """Antennas a survey can vouch for as a reference, best first.

    The survey's own reference antenna(s) lead (they have no SNR of their own but every detection was
    made against them), then the detected antennas by decreasing fringe SNR. Antennas the survey never
    detected are left out. Without a survey every antenna is returned, in the order given.
    """
    if survey is None:
        return list(antennas)
    ordered = list(survey.refant_names) + [name for name, _ in survey.rank_antennas()]
    return [name for name in dict.fromkeys(ordered) if name in antennas]


def _gap_reason(name: str, numbers: list[int], present: set[int], no_data: set[int], no_fringes: set[int]) -> str:
    """Say in plain words why ``name`` does not cover the scans ``numbers``.

    Parameters
    ----------
    name : str
        Antenna that leaves the scans uncovered.
    numbers : list of int
        The scans in question.
    present, no_data, no_fringes : set of int
        Scans the antenna took part in, and those of them where it has no unflagged data / no fringes.

    Returns
    -------
    str
        e.g. ``"EF is absent from scans 2-21, has no unflagged data in scan 40"``.
    """
    absent = [n for n in numbers if n not in present]
    flagged = [n for n in numbers if n in present and n in no_data]
    silent = [n for n in numbers if n in present and n not in no_data and n in no_fringes]
    parts = []
    if absent:
        parts.append(f"is absent from {_counted('scan', absent)}")
    if flagged:
        parts.append(f"has no unflagged data in {_counted('scan', flagged)}")
    if silent:
        parts.append(f"shows no fringes in {_counted('scan', silent)}")
    return f"{name} " + ", ".join(parts)


def plan_time_evolution(metadata: ObsMetadata, survey: Optional[ScanSNRSurvey] = None, refant: str = "", *,
                        without_data: Optional[dict] = None, without_fringes: Optional[dict] = None,
                        max_refants: int = TIME_EVOLUTION_MAX_REFANTS,
                        max_subbands: int = TIME_EVOLUTION_MAX_SUBBANDS) -> list[dict]:
    """Decide which (reference antenna, subband) pairs the raw time-evolution figures are made for.

    One figure shows every baseline to one reference antenna in one subband over the whole observation.
    The plan is the smallest set of figures in which every antenna and every scan appears:

    * subbands: those of the observation's reference antenna that between them show every other
      antenna (:func:`plan_time_evolution_subbands`);
    * reference antennas: the observation's reference antenna first. It covers the scans it took part
      in, minus the ones it is known to have no unflagged data or no fringes in (``without_data`` /
      ``without_fringes``, which only the data itself can tell: see :func:`reference_antenna_gaps`).
      While scans with a baseline remain uncovered, the antenna with the highest fringe SNR among those
      that cover some of them is added (:func:`_reference_order`; without a survey, the one covering
      the most of them);
    * antennas that share no subband with the reference antenna can be in none of its figures: they
      get one on the best-ranked antenna that recorded a subband of theirs.

    A pair is planned only when the antenna recorded the subband; an added reference antenna that
    recorded none of the chosen subbands gets the best-populated subband it did record.

    Parameters
    ----------
    metadata : ObsMetadata
        Antennas, their subbands, and the antennas of every scan.
    survey : ScanSNRSurvey, optional
        Fringe-SNR survey used to rank the additional reference antennas.
    refant : str
        The observation's reference antenna (a comma-separated chain is reduced to its first member
        with data; empty picks one as :func:`rank_reference_antennas` does).
    without_data, without_fringes : dict, optional
        ``{antenna: scan numbers}`` in which the antenna is listed but has no unflagged data, or has
        data but shows no fringes on any baseline.
    max_refants, max_subbands : int
        Most reference antennas and subbands (so at most their product in figures).

    Returns
    -------
    list of dict
        ``[{"refant", "subband", "reason", "covers"}, ...]``, the observation's reference antenna first.
        ``reason`` says in plain words why the pair exists; ``covers`` lists the scans an additional
        reference antenna was added for (empty for the first one). Empty when nothing can be plotted.
    """
    recorded = _recorded_subbands(metadata)
    antennas = list(recorded)
    primary = next((name for name in rank_reference_antennas(metadata, refant) if name in recorded), None)
    subbands = plan_time_evolution_subbands(metadata, max_subbands, reference=primary) if primary else []
    if len(antennas) < 2 or not subbands:
        logger.info("time evolution: nothing to plan ({} antenna(s) with data, {} usable subband(s))",
                    len(antennas), len(metadata.usable_subbands))
        return []
    scans = [scan for scan in metadata.scans if len(scan.antennas) >= 2]
    present = {name: {scan.scan_number for scan in scans if name in scan.antennas} for name in antennas}
    no_data = {name: {int(n) for n in numbers} for name, numbers in (without_data or {}).items()}
    no_fringes = {name: {int(n) for n in numbers} for name, numbers in (without_fringes or {}).items()}
    covered = {name: present[name] - no_data.get(name, set()) - no_fringes.get(name, set()) for name in antennas}

    references: list[tuple[str, str, list[int]]] = [(primary, "reference antenna", [])]
    uncovered = {scan.scan_number for scan in scans} - covered[primary]
    order = _reference_order(survey, antennas)
    while uncovered and len(references) < max_refants:
        taken = [name for name, _, _ in references]
        candidates = [name for name in order if name not in taken and covered[name] & uncovered]
        if not candidates:
            break
        # With a survey ``order`` is by fringe SNR, so the first candidate is the best detected one.
        best = candidates[0] if survey is not None else max(candidates, key=lambda n: len(covered[n] & uncovered))
        gained = sorted(covered[best] & uncovered)
        why = _gap_reason(primary, gained, present[primary], no_data.get(primary, set()),
                          no_fringes.get(primary, set()))
        if len(taken) > 1:
            why += f" (not covered by {', '.join(taken[1:])} either)"
        references.append((best, why, gained))
        uncovered -= set(gained)

    chosen = [subband for subband, _ in subbands]
    plan: list[dict] = []
    for name, why, gained in references:
        own = [(subband, text) for subband, text in subbands if subband in recorded[name]]
        if not own:
            own = [(subband, f"subband {subband} because {name} did not record {_counted('subband', chosen)}")
                   for subband, _ in plan_time_evolution_subbands(metadata, 1, reference=name)]
        plan.extend({"refant": name, "subband": int(subband), "reason": f"{why}; {text}", "covers": list(gained)}
                    for subband, text in own)

    # Antennas that share no subband with the reference antenna are in none of the figures above.
    with_baseline = set(metadata.usable_subbands)
    stranded = [name for name in antennas if not recorded[name] & recorded[primary] & with_baseline]
    helpers = {name for name, _, _ in references}
    while stranded:
        reach = set().union(*(recorded[name] for name in stranded)) & with_baseline
        helper = next((name for name in order if name != primary and recorded[name] & reach
                       and (name in helpers or len(helpers) < max_refants)), None)
        if helper is None:
            break
        others = [name for name in stranded if name != helper]
        picked = (plan_time_evolution_subbands(metadata, 1, reference=helper, wanted=others) if others else []) \
            or plan_time_evolution_subbands(metadata, 1, reference=helper)
        if not picked:
            break
        subband = picked[0][0]
        shown = [name for name in stranded if name == helper or subband in recorded[name]]
        if not shown:
            break
        plan.append({"refant": helper, "subband": int(subband), "covers": [],
                     "reason": f"{', '.join(shown)} share{'s' if len(shown) == 1 else ''} no subband with {primary}; "
                               f"subband {subband} on {helper}"})
        helpers.add(helper)
        stranded = [name for name in stranded if name not in shown]

    logger.info("time evolution: {} figure(s) planned: {}", len(plan),
                "; ".join(f"ref {entry['refant']} subband {entry['subband']}" for entry in plan))
    if uncovered:
        logger.info("time evolution: {} left without a reference antenna", _counted("scan", uncovered))
    if stranded:
        logger.info("time evolution: {} left out (no reference antenna shares a subband with them)",
                    ", ".join(stranded))
    return plan


def lag_coherence(values) -> float:
    """Coherence between consecutive samples of one complex time series, between 0 and 1.

    ``|sum(v[t] * conj(v[t+1]))| / sum(|v[t]| * |v[t+1]|)``: a signal whose phase turns by the same small
    angle from one integration to the next adds up whatever its fringe rate, so the result is near 1 for
    a detection and near ``1 / sqrt(N)`` for noise. Unlike a plain vector average over the scan it does
    not need the residual rate to be removed first, which it is not in raw data.

    Parameters
    ----------
    values : array-like of complex
        Samples in time order; non-finite ones are skipped.

    Returns
    -------
    float
        The coherence, or ``nan`` with fewer than :data:`MIN_COHERENCE_PAIRS` pairs or no power at all.
    """
    series = np.asarray(values, dtype=complex).ravel()
    series = series[np.isfinite(series)]
    if series.size < MIN_COHERENCE_PAIRS + 1:
        return float("nan")
    products = series[:-1] * np.conj(series[1:])
    power = float(np.abs(products).sum())
    return float(np.abs(products.sum()) / power) if power > 0.0 else float("nan")


def reference_antenna_gaps(series: dict, metadata: ObsMetadata) -> dict[str, list[int]]:
    """Find the scans in which the reference antenna of a time-evolution series has no data, or no fringes.

    This is what the metadata cannot say: a station listed in a scan may have every visibility flagged
    (it was off source), or record noise only. Both are read off the series itself.

    * no data: scans the antenna is listed in (with at least one other antenna) that hold no unflagged
      sample on any baseline to it (``series["scans_with_data"]``);
    * no fringes: per scan, the highest :func:`lag_coherence` over all baselines and hands. Only sources
      the antenna clearly sees are judged - at least half of their scans reach :data:`FRINGE_COHERENCE` -
      and of those the scans below :data:`NOISE_COHERENCE` count. A weak target on which no baseline
      fringes says nothing about the antenna.

    Parameters
    ----------
    series : dict
        Output of ``DataOps.read_time_evolution``: ``refant``, ``baselines`` (``{antenna: {"scans",
        "vis"}}`` with ``vis`` complex ``(n_sample, n_hand)`` in time order), ``scans_with_data`` and
        ``scan_sources`` (``{scan number: source}``).
    metadata : ObsMetadata
        Supplies the scans the antenna is listed in.

    Returns
    -------
    dict
        ``{"no_data": [scan numbers], "no_fringes": [scan numbers]}``, both sorted.
    """
    refant = series["refant"]
    listed = {scan.scan_number for scan in metadata.scans if refant in scan.antennas and len(scan.antennas) >= 2}
    no_data = sorted(listed - {int(n) for n in series.get("scans_with_data") or []})
    best: dict[int, float] = {}
    for entry in (series.get("baselines") or {}).values():
        scans, vis = np.asarray(entry["scans"]), np.asarray(entry["vis"])
        for number in np.unique(scans):
            rows = scans == number
            for hand in range(vis.shape[1] if vis.ndim > 1 else 0):
                coherence = lag_coherence(vis[rows, hand])
                if coherence == coherence:
                    best[int(number)] = max(best.get(int(number), 0.0), coherence)
    sources = {int(number): source for number, source in (series.get("scan_sources") or {}).items()}
    no_fringes: list[int] = []
    for source in dict.fromkeys(sources.values()):
        judged = [number for number in best if sources.get(number) == source]
        if judged and 2 * sum(1 for number in judged if best[number] >= FRINGE_COHERENCE) >= len(judged):
            no_fringes.extend(number for number in judged if best[number] < NOISE_COHERENCE)
    return {"no_data": no_data, "no_fringes": sorted(no_fringes)}
