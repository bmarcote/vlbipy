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
"""
from __future__ import annotations

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
    rest by how many scans they took part in. Only antennas with data are kept.
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
    as possible are used.

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
        if not references:
            break
        best = max(references, key=lambda s: (len(remaining[s] - covered), score(s)))
        gained = remaining[best] - covered
        gained_names = sorted({antenna for antenna, _ in gained}, key=rank)
        stages.append({"scan": best, "antennas": gained_names, "refant": references[best][0]})
        covered |= gained
        remaining.pop(best)

    n_antennas = len({antenna for antenna, _ in required})
    if len(stages) == 1:
        logger.info("scan selection: scan {} detects all {} antennas in every polarization they have "
                    "(median SNR {:.0f}); single-stage SBD", stages[0]["scan"], n_antennas, score(stages[0]["scan"]))
    else:
        logger.info("scan selection: no single scan covers the array; SBD in {} stage(s): {}", len(stages),
                    "; ".join(f"scan {s['scan']} ({','.join(s['antennas'])}; ref {s['refant']})" for s in stages))
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
