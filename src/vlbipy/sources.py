"""Sources and source-role handling for vlbipy.

Defines the :class:`Source` domain object and :class:`SourceSet`, a role-aware
collection with attribute accessors (``targets``, ``phase_calibrators`` ...),
name lookup, and phase-referencing resolution with an automatic empty-role
fallback (fringe finder -> phase calibrator -> target).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from astropy import coordinates as coord
from astropy import units as u

from .errors import SourceNotFoundError
from .logging_utils import get_logger
from .models import SourceType
from .phaseref import SETTING_KEYS

logger = get_logger()

# Config role-key -> SourceType.
_ROLE_KEYS = {
    "targets": SourceType.TARGET,
    "phase_calibrators": SourceType.PHASE_CALIBRATOR,
    "fringe_finders": SourceType.FRINGE_FINDER,
    "check_sources": SourceType.CHECK_SOURCE,
    "polarization_calibrators": SourceType.POLARIZATION_CALIBRATOR,
}


def _as_list(value) -> list[str]:
    """Normalise a str or list value to a list of non-empty stripped strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


@dataclass
class Source:
    """A radio source with a name, sky position, and observational role.

    Parameters
    ----------
    name : str
        Source name as it appears in the data.
    coordinates : astropy.coordinates.SkyCoord, optional
        Sky position.
    source_type : SourceType
        Role of the source.
    calcode : str
        Calibrator code, if any.
    protected : bool
        Whether archive access is credential-protected.
    """

    name: str
    coordinates: Optional[coord.SkyCoord] = None
    source_type: SourceType = SourceType.OTHER
    calcode: str = ""
    protected: bool = False

    def separation(self, other: "Source") -> u.Quantity:
        """Return the angular separation to another source.

        Parameters
        ----------
        other : Source
            The other source.

        Returns
        -------
        astropy.units.Quantity
            Angular separation in degrees.

        Raises
        ------
        ValueError
            If either source has no coordinates.
        """
        if self.coordinates is None or other.coordinates is None:
            raise ValueError(f"coordinates not set for {self.name!r} or {other.name!r}")
        return self.coordinates.separation(other.coordinates).to(u.deg)

    def __str__(self) -> str:
        return f"{self.name} ({self.source_type.value})"


class SourceSet:
    """An ordered, role-aware collection of :class:`Source` objects.

    Parameters
    ----------
    sources : iterable of Source
        The sources to hold (order preserved).
    phase_referencing : dict, optional
        Mapping of target name -> ordered list of calibrator names.
    """

    def __init__(self, sources: Iterable[Source],
                 phase_referencing: dict[str, list[str]] | None = None) -> None:
        self._sources: list[Source] = list(sources)
        self.phase_referencing: dict[str, list[str]] = phase_referencing or {}

    @classmethod
    def from_config(cls, sources_cfg: dict, phaseref_cfg: dict | None = None) -> "SourceSet":
        """Build a SourceSet from configuration dictionaries.

        Parameters
        ----------
        sources_cfg : dict
            Keys are role names (``targets``, ``phase_calibrators`` ...); values
            are a source name or list of names.
        phaseref_cfg : dict, optional
            The ``[phase_referencing]`` section: mapping target name -> calibrator name or list of names,
            next to the strategy settings (:data:`vlbipy.phaseref.SETTING_KEYS`), which are left out.

        Returns
        -------
        SourceSet
        """
        sources: list[Source] = []
        seen: set[str] = set()
        for key, stype in _ROLE_KEYS.items():
            for name in _as_list((sources_cfg or {}).get(key)):
                if name in seen:
                    continue
                seen.add(name)
                sources.append(Source(name=name, source_type=stype))
        phaseref = {str(t): _as_list(c) for t, c in (phaseref_cfg or {}).items() if t not in SETTING_KEYS}
        return cls(sources, phase_referencing=phaseref)

    # -- role accessors --
    def _by_type(self, stype: SourceType) -> list[Source]:
        return [s for s in self._sources if s.source_type == stype]

    @property
    def targets(self) -> list[Source]:
        """All target sources."""
        return self._by_type(SourceType.TARGET)

    @property
    def phase_calibrators(self) -> list[Source]:
        """All phase calibrators."""
        return self._by_type(SourceType.PHASE_CALIBRATOR)

    @property
    def fringe_finders(self) -> list[Source]:
        """All fringe finders."""
        return self._by_type(SourceType.FRINGE_FINDER)

    @property
    def check_sources(self) -> list[Source]:
        """All check sources."""
        return self._by_type(SourceType.CHECK_SOURCE)

    @property
    def polarization_calibrators(self) -> list[Source]:
        """All polarization calibrators."""
        return self._by_type(SourceType.POLARIZATION_CALIBRATOR)

    @property
    def calibrators(self) -> list[Source]:
        """Phase calibrators plus fringe finders."""
        return self.phase_calibrators + self.fringe_finders

    @property
    def target(self) -> Source:
        """The single target source.

        Returns
        -------
        Source

        Raises
        ------
        SourceNotFoundError
            If there are zero targets or more than one (ambiguous); use
            :attr:`targets` or index by name in the ambiguous case.
        """
        tgts = self.targets
        if not tgts:
            raise SourceNotFoundError("no target source defined")
        if len(tgts) > 1:
            names = ", ".join(t.name for t in tgts)
            raise SourceNotFoundError(f"multiple targets ({names}); use .targets or select by name")
        return tgts[0]

    @property
    def names(self) -> list[str]:
        """Names of all sources, in order."""
        return [s.name for s in self._sources]

    def add(self, source: Source) -> bool:
        """Append a source unless its name is already present; True when it was added.

        Used to adopt roles inferred from the data (:func:`infer_source_roles`)
        without ever overriding a declared one.
        """
        if source.name in self:
            return False
        self._sources.append(source)
        return True

    def restrict_to(self, observed: Iterable[str]) -> list[str]:
        """Drop the sources that are not among ``observed``; return the names dropped.

        The epochs of a campaign share one source declaration, but not every
        epoch observes every source (a calibrator tried once and abandoned, for
        instance). A declared source with no data in an epoch must not reach that
        epoch's field selections: one unknown name makes a CASA selection fail as
        a whole.
        """
        present = set(observed)
        dropped = [s.name for s in self._sources if s.name not in present]
        if dropped:
            self._sources = [s for s in self._sources if s.name in present]
            self.phase_referencing = {
                target: [c for c in calibrators if c in present]
                for target, calibrators in self.phase_referencing.items() if target in present}
        return dropped

    # -- phase referencing --
    def calibrators_for(self, target_name: str) -> list[Source]:
        """Return the calibrators phase-referencing a given target.

        Uses the explicit ``phase_referencing`` mapping if present; otherwise
        falls back through fringe finders -> phase calibrators -> the target.

        Parameters
        ----------
        target_name : str
            Name of the target.

        Returns
        -------
        list of Source
        """
        mapped = self.phase_referencing.get(target_name)
        if mapped:
            return [self[name] for name in mapped]
        if self.fringe_finders:
            logger.warning("no phase-ref mapping for {!r}; falling back to fringe finders", target_name)
            return self.fringe_finders
        if self.phase_calibrators:
            logger.warning("no phase-ref mapping for {!r}; falling back to phase calibrators", target_name)
            return self.phase_calibrators
        logger.warning("no calibrators available for {!r}; falling back to the target itself", target_name)
        return [self[target_name]]

    # -- container protocol --
    def __getitem__(self, name: str) -> Source:
        for s in self._sources:
            if s.name == name:
                return s
        raise SourceNotFoundError(f"source {name!r} not found (have: {', '.join(self.names) or 'none'})")

    def __iter__(self) -> Iterator[Source]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, name: object) -> bool:
        return any(s.name == name for s in self._sources)

    def __repr__(self) -> str:
        return f"SourceSet({len(self._sources)} sources: {', '.join(self.names) or 'none'})"


# -- role inference ----------------------------------------------------------------------------

#: A source observed in at most this many scans (or this fraction of them, whichever is larger)
#: is a bright snapshot — a fringe finder — rather than part of a phase-referencing cycle.
FINDER_MAX_SCANS = 3
FINDER_MAX_FRACTION = 0.1

#: Two sources further apart than this on the sky are not a phase-referencing pair, however
#: their scans interleave: phase referencing only transfers over a few degrees.
PAIR_MAX_SEPARATION_DEG = 10.0

#: Fraction of the consecutive scan pairs of two sources that must change source for them to
#: count as interleaved (a perfectly alternating cycle gives 1.0, two separate blocks ~0.0).
PAIR_MIN_ALTERNATION = 0.6


def _scans_per_source(metadata) -> dict[str, list]:
    """Return ``{source: [its scans, in time order]}`` from an observation's metadata."""
    per_source: dict[str, list] = {}
    for scan in sorted(metadata.scans, key=lambda s: s.time_start):
        per_source.setdefault(scan.source, []).append(scan)
    return per_source


def _alternation(metadata, first: str, second: str) -> float:
    """Fraction of consecutive scans on ``first``/``second`` that switch between the two.

    A phase-referencing cycle is ``cal, target, cal, target, ...``, so every
    consecutive pair changes source and this is ~1. Two sources observed in
    separate blocks give ~0. Returns 0.0 when there are too few scans to tell.
    """
    sequence = [s.source for s in sorted(metadata.scans, key=lambda s: s.time_start)
                if s.source in (first, second)]
    if len(sequence) < 4:
        return 0.0
    changes = sum(1 for index in range(1, len(sequence)) if sequence[index] != sequence[index - 1])
    return changes / float(len(sequence) - 1)


def _separation_deg(metadata, first: str, second: str) -> Optional[float]:
    """Angular separation of two sources in degrees, or None when either has no coordinates."""
    coords = getattr(metadata, "source_coords", {}) or {}
    if first not in coords or second not in coords:
        return None
    one = coord.SkyCoord(coords[first][0] * u.deg, coords[first][1] * u.deg)
    other = coord.SkyCoord(coords[second][0] * u.deg, coords[second][1] * u.deg)
    return float(one.separation(other).to(u.deg).value)


def _is_pair(metadata, first: str, second: str) -> bool:
    """Whether two sources look like a phase-referencing pair: interleaved and close on the sky."""
    if _alternation(metadata, first, second) < PAIR_MIN_ALTERNATION:
        return False
    separation = _separation_deg(metadata, first, second)
    if separation is not None and separation > PAIR_MAX_SEPARATION_DEG:
        logger.info("source roles: {} and {} alternate but are {:.1f} deg apart; not a phase-reference pair",
                    first, second, separation)
        return False
    return True


def _calibrator_of_pair(metadata, survey, first: str, second: str) -> tuple[str, str]:
    """Return ``(phase calibrator, target)`` for an interleaved pair of sources.

    Brightness alone cannot decide this: plenty of targets are brighter than
    their calibrator. Three independent structural signatures of a
    phase-referencing cycle are used instead and the majority wins —

    * the cycle opens on the calibrator, so its first scan comes first;
    * the calibrator is visited at least as often as the target (``cal, target,
      ..., cal`` has one scan more);
    * the calibrator's scans are shorter — the integration time goes on the target.

    Three signals cannot tie, so no further tiebreak is needed. The array-wide
    fringe SNR of both sources is logged for the reader but deliberately not
    voted on: it decides which source the *instrumental* solve uses (see
    :meth:`~vlbipy.models.ScanSNRSurvey.rank_sources`), while a target brighter
    than its own calibrator is entirely ordinary.
    """
    per_source = _scans_per_source(metadata)
    one, other = per_source.get(first, []), per_source.get(second, [])
    snr = {name: value for name, value, _ in (survey.rank_sources() if survey is not None else [])}
    votes = [one[0].time_start < other[0].time_start if one and other else False,
             len(one) >= len(other),
             _median_duration(one) < _median_duration(other)]
    first_is_cal = sum(votes) >= 2
    logger.info("source roles: {} vs {} — opens the cycle/more scans/shorter scans = {}, array SNR "
                "{:.0f} vs {:.0f} -> phase calibrator {}", first, second, votes, snr.get(first, 0.0),
                snr.get(second, 0.0), first if first_is_cal else second)
    return (first, second) if first_is_cal else (second, first)


def _median_duration(scans: list) -> float:
    """Median scan duration in seconds (0.0 for an empty list)."""
    durations = sorted(s.duration_sec for s in scans)
    if not durations:
        return 0.0
    mid = len(durations) // 2
    return durations[mid] if len(durations) % 2 else (durations[mid - 1] + durations[mid]) / 2.0


def infer_source_roles(metadata, survey=None, *, declared: dict[str, SourceType] | None = None
                       ) -> tuple[dict[str, SourceType], dict[str, list[str]]]:
    """Infer the role of every source in the data that the configuration did not declare.

    An observation imported without a ``[sources]`` section — an external
    project, or one whose declared names do not match the data — has no role at
    all, and every field selection then falls back to "all sources". The
    instrumental delay is then solved on whichever scan happens to cover the most
    antennas, which may be a weak target, and the fringe fit returns no solution
    above its minimum SNR. The roles are therefore reconstructed from what the
    data itself shows: how often each source was visited, how its scans
    interleave with the others, how far apart they are on the sky, and — when a
    fringe SNR survey is available — how well the array detects each one.

    The rules, in order:

    1. Sources visited in at most :data:`FINDER_MAX_SCANS` scans (or
       :data:`FINDER_MAX_FRACTION` of them, whichever is more) are bright
       snapshots: fringe finders. One the array does not detect at all is left
       as ``OTHER`` rather than put first in the calibration's fallback chain.
    2. The remaining, repeatedly-visited sources are paired up into
       phase-referencing cycles (interleaved in time and within
       :data:`PAIR_MAX_SEPARATION_DEG`); :func:`_calibrator_of_pair` says which
       member is the calibrator. An unpaired one is a target.
    3. An observation where no source is revisited at all (a snapshot survey)
       makes its best-detected source the fringe finder and the rest targets.
    4. A single source is its own target: it is self-calibrated, and the
       calibration's fallback chain uses the target when no calibrator exists.

    A role that is already declared is never overridden, and it constrains the
    inference: the partner of a declared target is the phase calibrator whatever
    the structural vote says.

    Parameters
    ----------
    metadata : ObsMetadata
        Observation metadata; supplies the scans, source names and coordinates.
    survey : ScanSNRSurvey, optional
        Per-scan fringe SNR survey, used to rank the sources by how well the
        whole array detects them. Without it the inference is purely structural.
    declared : dict, optional
        ``{source: SourceType}`` already declared in the configuration. These are
        kept as they are and left out of the result.

    Returns
    -------
    (dict, dict)
        ``{source: SourceType}`` for the sources whose role was *not* declared,
        and ``{target: [phase calibrator]}`` for the pairs found among them.
    """
    fixed = dict(declared or {})
    names = [n for n in (metadata.source_names or list(_scans_per_source(metadata))) if n]
    if not names:
        return {}, {}
    per_source = _scans_per_source(metadata)
    limit = max(FINDER_MAX_SCANS, int(FINDER_MAX_FRACTION * len(metadata.scans)))
    array_snr = {name: value for name, value, _ in (survey.rank_sources() if survey is not None else [])}
    # Best-detected first, so the brightest snapshot leads the fringe-finder group and the
    # calibration tries it before anything else.
    ordered = sorted(names, key=lambda n: (array_snr.get(n, 0.0), len(per_source.get(n, []))), reverse=True)
    few = [n for n in ordered if len(per_source.get(n, [])) <= limit]
    many = [n for n in ordered if len(per_source.get(n, [])) > limit]

    roles: dict[str, SourceType] = {}
    phaseref: dict[str, list[str]] = {}
    if many:
        for name in few:
            detected = array_snr.get(name, None)
            roles[name] = SourceType.OTHER if detected == 0.0 else SourceType.FRINGE_FINDER
            if roles[name] is SourceType.OTHER:
                logger.info("source roles: {} was observed {} time(s) but no antenna detects it; "
                            "not used as a fringe finder", name, len(per_source.get(name, [])))
        waiting = list(many)
        while waiting:
            name = waiting.pop(0)
            partner = next((other for other in waiting if _is_pair(metadata, name, other)), None)
            if partner is None:
                roles[name] = SourceType.TARGET
                continue
            waiting.remove(partner)
            calibrator, target = _declared_pair_roles(fixed, name, partner)
            if calibrator is None:
                calibrator, target = _calibrator_of_pair(metadata, survey, name, partner)
            roles[calibrator] = SourceType.PHASE_CALIBRATOR
            roles[target] = SourceType.TARGET
            phaseref[target] = [calibrator]
    elif len(ordered) == 1:
        roles[ordered[0]] = SourceType.TARGET
    else:
        roles[ordered[0]] = SourceType.FRINGE_FINDER
        for name in ordered[1:]:
            roles[name] = SourceType.TARGET
    inferred = {name: role for name, role in roles.items() if name not in fixed}
    phaseref = {target: cals for target, cals in phaseref.items() if target in inferred or target in fixed}
    if inferred:
        logger.info("source roles inferred from the data: {}",
                    ", ".join(f"{name} = {role.value}" for name, role in inferred.items()))
    return inferred, phaseref


def _declared_pair_roles(declared: dict[str, SourceType], first: str,
                         second: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(calibrator, target)`` of a pair when the configuration fixes one member's role.

    The user naming the target settles the pair: its interleaved partner is the
    phase calibrator, whatever the structural vote would have said. ``(None, None)``
    when neither member has a declared role to go on.
    """
    for name, other in ((first, second), (second, first)):
        if declared.get(name) is SourceType.TARGET:
            return other, name
        if declared.get(name) in (SourceType.PHASE_CALIBRATOR, SourceType.FRINGE_FINDER):
            return name, other
    return None, None
