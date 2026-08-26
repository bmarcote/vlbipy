"""Antenna and scan selection for instrumental calibration (pure logic, no CASA).

The single-band delay and bandpass solutions are derived from a small amount of
data — ideally one scan on a bright fringe finder — so *which* data is picked
decides the quality of everything downstream. Both choices are made from the
per-scan fringe SNR survey (:class:`~vlbipy.models.ScanSNRSurvey`) plus the
subband participation recorded in the metadata:

* :func:`select_antennas` keeps antennas that recorded the whole frequency range
  and detected fringes, ranked by SNR (the approach used by the Avica pipeline,
  simplified to the two criteria that actually drive the result).
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
                    min_snr: float = DEFAULT_MIN_SNR, require_all_subbands: bool = True,
                    max_antennas: int = 0) -> list[str]:
    """Return the antennas usable for instrumental calibration, best first.

    An antenna qualifies when it recorded every subband (so its solutions cover
    the full band) and its median fringe SNR clears ``min_snr``. The survivors
    are ordered by median SNR, so the caller can take the head of the list as the
    reference-antenna preference.

    Parameters
    ----------
    survey : ScanSNRSurvey
        Per-scan fringe SNR survey over the calibrators.
    metadata : ObsMetadata
        Observation metadata; supplies subband participation per antenna.
    min_snr : float
        Minimum median fringe SNR for an antenna to be considered detected.
    require_all_subbands : bool
        Require the antenna to have data in every subband. Turn this off for
        arrays that are heterogeneous by design, at the cost of solutions that
        do not span the band.
    max_antennas : int
        Keep at most this many antennas (0 = no limit).

    Returns
    -------
    list of str
        Antenna names ordered by decreasing median SNR (empty if none qualify).
    """
    n_subbands = metadata.freq_setup.n_subbands
    # The reference antenna has no SNR of its own (its solutions are the sentinel it
    # is referenced against), so ranking alone would drop the one antenna every other
    # solution is tied to. It is detected by definition: put it first.
    reference = [name for name in survey.refant_names if name in metadata.antennas]
    ranked = [(name, float("inf")) for name in reference] + [
        entry for entry in survey.rank_antennas() if entry[0] not in reference]
    selected: list[str] = []
    rejected: list[str] = []
    for name, median_snr in ranked:
        antenna = metadata.antennas.get(name)
        subbands = antenna.subbands if antenna else ()
        if median_snr < min_snr:
            rejected.append(f"{name} (SNR {median_snr:.0f} < {min_snr:.0f})")
            continue
        if require_all_subbands and n_subbands and subbands and len(subbands) < n_subbands:
            rejected.append(f"{name} ({len(subbands)}/{n_subbands} subbands)")
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


def select_calibration_scans(survey: ScanSNRSurvey, antennas: list[str], *,
                             min_snr: float = DEFAULT_MIN_SNR,
                             sources: list[str] | None = None) -> list[int]:
    """Return the scan(s) to solve the instrumental delay and bandpass on.

    Prefers a single scan in which every antenna in ``antennas`` was detected. If
    no such scan exists, builds the smallest covering set greedily, requiring
    each added scan to share at least one antenna with the scans already chosen:
    that shared antenna is what lets solutions from different scans be referred
    to a common phase, without which they cannot be combined.

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

    Returns
    -------
    list of int
        Scan numbers, best first. Empty when nothing is detected at all.
    """
    if not antennas:
        return []
    detected = detected_antennas_per_scan(survey, antennas, min_snr)
    if sources:
        allowed = {scan for scan, source in zip(survey.scan_numbers, survey.scan_sources)
                   if source in sources}
        detected = {scan: ants for scan, ants in detected.items() if scan in allowed}
    if not detected:
        return []
    required = set(antennas)
    quality = {scan: survey.median_snr(scan_number=scan) for scan in detected}

    def score(scan: int) -> float:
        value = quality.get(scan, float("nan"))
        return value if value == value else 0.0

    complete = [scan for scan, found in detected.items() if required <= found]
    if complete:
        best = max(complete, key=score)
        logger.info("scan selection: scan {} has all {} antennas detected (median SNR {:.0f})",
                    best, len(required), score(best))
        return [best]

    # No single scan covers the array: greedily cover it, keeping the scans linked.
    chosen: list[int] = []
    covered: set[str] = set()
    remaining = dict(detected)
    while remaining:
        if not chosen:
            candidates = remaining
        else:
            # Only scans sharing an antenna with what is already covered can be tied in.
            candidates = {scan: found for scan, found in remaining.items() if found & covered}
            if not candidates:
                break
        best = max(candidates, key=lambda s: (len(candidates[s] - covered), score(s)))
        gained = candidates[best] - covered
        if not gained and chosen:
            break
        chosen.append(best)
        covered |= candidates[best]
        remaining.pop(best)
        if required <= covered:
            break
    missing = sorted(required - covered)
    logger.info("scan selection: no single scan covers the array; using {} scan(s) {} "
                "covering {}/{} antennas", len(chosen), chosen, len(covered & required), len(required))
    if missing:
        logger.warning("scan selection: no scan detects {} above {:.0f} sigma; they will have no "
                       "instrumental-delay solution", ", ".join(missing), min_snr)
    return chosen
