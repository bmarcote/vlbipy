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
            Mapping target name -> calibrator name or list of names.

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
        phaseref = {str(t): _as_list(c) for t, c in (phaseref_cfg or {}).items()}
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
