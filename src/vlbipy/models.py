"""Core domain data models for vlbipy.

These are plain, serializable dataclasses and enums with no backend or CASA
dependencies. They describe an observation's antennas, frequency setup, and
scans, plus the products of calibration/imaging (calibration tables, image
quality metrics). Every other module is written against these types.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
from dataclasses import dataclass, field as dcfield
from enum import Enum, IntEnum
from typing import Optional

#: Speed of light in m/s (avoids a scipy/astropy import in this dependency-free module).
_C_M_S = 299792458.0


class Stokes(IntEnum):
    """Stokes / correlation types following the casacore (MS) convention."""

    Undefined = 0
    I = 1  # noqa: E741
    Q = 2
    U = 3
    V = 4
    RR = 5
    RL = 6
    LR = 7
    LL = 8
    XX = 9
    XY = 10
    YX = 11
    YY = 12


class Observatory(Enum):
    """Supported VLBI networks."""

    EVN = "EVN"
    VLBA = "VLBA"
    LBA = "LBA"


class BackendKind(Enum):
    """Available data-reduction backends."""

    DUMMY = "dummy"
    CASA = "casa"
    AIPS = "aips"


class Mode(Enum):
    """Observing / processing mode."""

    CONTINUUM = "continuum"
    SPECTRAL_LINE = "spectral-line"
    PULSAR_BINNING = "pulsar-binning"
    MULTI_PHASE_CENTER = "multi-phase-center"


class SourceType(Enum):
    """Role of a source in the observation."""

    TARGET = "target"
    PHASE_CALIBRATOR = "phasecal"
    FRINGE_FINDER = "fringefinder"
    CHECK_SOURCE = "checksource"
    POLARIZATION_CALIBRATOR = "polcal"
    OTHER = "other"


@dataclass
class Antenna:
    """A single antenna / station in the array.

    Parameters
    ----------
    name : str
        Short antenna code (e.g. ``"EF"``).
    fullname : str
        Full station name.
    diameter : float
        Dish diameter in metres.
    position : tuple of float
        ITRF geocentric position ``(x, y, z)`` in metres.
    observed : bool
        Whether the antenna produced data.
    subbands : tuple of int
        Indices of the subbands the antenna observed (empty = not determined).
    mount : str
        Mount type as recorded in the data (``"ALT-AZ"``, ``"EQUATORIAL"``, ``"X-Y"``, ...).
    n_scans : int
        Number of scans the antenna took part in.
    """

    name: str
    fullname: str = ""
    diameter: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    observed: bool = False
    subbands: tuple[int, ...] = ()
    mount: str = ""
    n_scans: int = 0

    def __str__(self) -> str:
        return self.name


@dataclass
class FreqSetup:
    """Frequency setup of the observation.

    Parameters
    ----------
    ref_freq : float
        Reference frequency in Hz.
    total_bandwidth : float
        Total bandwidth in Hz.
    n_subbands : int
        Number of spectral windows / IFs.
    n_channels : int
        Channels per subband.
    channel_width : float
        Channel width in Hz.
    polarizations : list of Stokes
        Correlation products present, in the order the data stores them
        (typically RR, RL, LR, LL — the parallel hands are not the first two).
    channel_freqs : list of list of float
        Sky frequency in Hz of every channel, per subband. Spectral plots use
        this so their x axis is physical frequency rather than a channel index,
        which matters because subbands are not necessarily contiguous or ordered.
    """

    ref_freq: float = 0.0
    total_bandwidth: float = 0.0
    n_subbands: int = 0
    n_channels: int = 0
    channel_width: float = 0.0
    polarizations: list[Stokes] = dcfield(default_factory=list)
    channel_freqs: list[list[float]] = dcfield(default_factory=list)

    @property
    def freq_ghz(self) -> float:
        """Reference frequency in GHz."""
        return self.ref_freq / 1e9

    @property
    def bandwidth_mhz(self) -> float:
        """Total bandwidth in MHz."""
        return self.total_bandwidth / 1e6

    def frequencies_ghz(self, subband: Optional[int] = None) -> list[float]:
        """Return channel frequencies in GHz for one subband, or all of them in order.

        Falls back to a linear ramp built from ``ref_freq`` and ``channel_width``
        when the real frequencies were not recorded, so plots still have a
        sensible axis rather than failing.
        """
        if self.channel_freqs:
            if subband is None:
                return [f / 1e9 for spw in self.channel_freqs for f in spw]
            if subband < len(self.channel_freqs):
                return [f / 1e9 for f in self.channel_freqs[subband]]
        n_chan = max(1, self.n_channels)
        span = self.channel_width * n_chan
        start = self.ref_freq - self.total_bandwidth / 2.0
        subbands = range(self.n_subbands) if subband is None else [subband]
        return [(start + s * span + c * self.channel_width) / 1e9
                for s in subbands for c in range(n_chan)]


@dataclass
class Scan:
    """A single scan.

    Parameters
    ----------
    scan_number : int
        Scan identifier.
    source : str
        Name of the observed source.
    time_start, time_end : float
        Start / end time in MJD seconds.
    antennas : list of str
        Antenna codes present in the scan.
    integration_time : float
        Correlator integration (dump) time in seconds.
    subbands : tuple of int
        Indices of the subbands recorded in this scan.
    """

    scan_number: int
    source: str = ""
    time_start: float = 0.0
    time_end: float = 0.0
    antennas: list[str] = dcfield(default_factory=list)
    integration_time: float = 0.0
    subbands: tuple[int, ...] = ()

    @property
    def duration_sec(self) -> float:
        """Scan duration in seconds."""
        return self.time_end - self.time_start


@dataclass
class QualityMetrics:
    """Image quality metrics.

    Parameters
    ----------
    peak : float
        Peak brightness in Jy/beam.
    rms : float
        Off-source noise in Jy/beam.
    dynamic_range : float
        Peak / rms.
    integrated_flux : float
        Integrated flux density in Jy.
    beam : tuple of float
        Synthesised beam ``(bmaj_mas, bmin_mas, bpa_deg)``.
    """

    peak: float = 0.0
    rms: float = 0.0
    dynamic_range: float = 0.0
    integrated_flux: float = 0.0
    beam: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __str__(self) -> str:
        return (f"peak={self.peak:.4g} Jy/beam, rms={self.rms:.4g} Jy/beam, DR={self.dynamic_range:.4g}, "
                f"S_int={self.integrated_flux:.4g} Jy, beam={self.beam[0]:.3g}x{self.beam[1]:.3g} mas "
                f"@ {self.beam[2]:.1f} deg")


@dataclass
class CalTable:
    """A calibration product produced by a backend.

    Parameters
    ----------
    cal_type : str
        Kind of calibration (e.g. ``"tsys"``, ``"sbd"``, ``"bpass"``, ``"mbd"``).
    path : str
        Path to the table (synthetic under the dummy backend).
    field : str
        Field(s) the table was solved on (provenance only).
    gainfield : str
        Field selection to use when *applying* the table. Empty means "no field
        selection", which is what field-independent tables (Tsys, gain curve,
        EOP) require — selecting a field on them matches zero rows and applycal
        fails. Set this only to transfer one field's solutions to another, as
        phase referencing does.
    interp : str
        Interpolation mode for apply.
    spwmap : list of int
        Spectral-window mapping (empty = identity).
    snr : float
        Representative solution SNR.
    step : str
        Pipeline step that produced the table. Recorded so a resumed run can
        rebuild the apply chain and drop the tables belonging to steps it is
        about to redo.
    calwt : bool
        Calibrate the visibility weights along with the data. True is CASA's own
        default and what an amplitude calibration requires: after Tsys and the
        gain curve the weights should track each antenna's real sensitivity,
        which matters on a heterogeneous VLBI array. False leaves the weights as
        imported.
    """

    cal_type: str
    path: str = ""
    field: str = ""
    gainfield: str = ""
    interp: str = "linear"
    spwmap: list[int] = dcfield(default_factory=list)
    snr: float = 0.0
    step: str = ""
    calwt: bool = True

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of every field."""
        return {"cal_type": self.cal_type, "path": self.path, "field": self.field,
                "gainfield": self.gainfield, "interp": self.interp,
                "spwmap": list(self.spwmap), "snr": self.snr, "step": self.step,
                "calwt": self.calwt}

    @classmethod
    def from_dict(cls, data: dict) -> "CalTable":
        """Rebuild a table from :meth:`to_dict` output, ignoring unknown keys."""
        known = {"cal_type", "path", "field", "gainfield", "interp", "spwmap", "snr",
                 "step", "calwt"}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:
        return f"CalTable({self.cal_type}, field={self.field!r}, snr={self.snr:.1f})"


@dataclass
class ScanSNRSurvey:
    """Per-scan, per-antenna, per-polarization fringe signal-to-noise on the calibrators.

    Produced by a backend's ``calibrate.scan_snr()`` (a short fringe fit over the
    central channels of every calibrator scan) and consumed by the diagnostics
    plots and by scan / reference-antenna selection. All values are plain floats
    so the survey stays JSON-serializable; ``nan`` marks an antenna that was
    absent from a scan or whose solution failed.

    Parameters
    ----------
    project_code : str
        Project the survey belongs to.
    scan_numbers : list of int
        Scan identifiers, one per row of each SNR matrix.
    scan_sources : list of str
        Source observed in each scan (parallel to ``scan_numbers``).
    antennas : list of str
        Antenna codes, one per column of each SNR matrix.
    snr : dict
        Mapping polarization label (e.g. ``"RR"``) -> matrix of shape
        ``(n_scans, n_antennas)`` as nested lists.
    channel_fraction : float
        Fraction of central channels used in the fringe fit.
    refant : str
        Reference antenna used (its own solutions are masked out as ``nan``).
    """

    project_code: str = ""
    scan_numbers: list[int] = dcfield(default_factory=list)
    scan_sources: list[str] = dcfield(default_factory=list)
    antennas: list[str] = dcfield(default_factory=list)
    snr: dict[str, list[list[float]]] = dcfield(default_factory=dict)
    channel_fraction: float = 0.8
    refant: str = ""

    @property
    def polarizations(self) -> list[str]:
        """Polarization labels present in the survey."""
        return list(self.snr)

    def matrix(self, polarization: str) -> list[list[float]]:
        """Return the ``(n_scans, n_antennas)`` SNR matrix for one polarization."""
        if polarization not in self.snr:
            raise KeyError(f"no SNR matrix for polarization {polarization!r} "
                           f"(have: {', '.join(self.polarizations) or 'none'})")
        return self.snr[polarization]

    def values_for(self, *, antenna: Optional[str] = None, scan_number: Optional[int] = None,
                   polarization: Optional[str] = None) -> list[float]:
        """Return the finite SNR values matching an antenna / scan / polarization selection."""
        pols = [polarization] if polarization else self.polarizations
        rows = ([self.scan_numbers.index(scan_number)] if scan_number is not None
                else range(len(self.scan_numbers)))
        cols = ([self.antennas.index(antenna)] if antenna else range(len(self.antennas)))
        out = [self.snr[p][r][c] for p in pols for r in rows for c in cols]
        return [v for v in out if v == v]  # drop nan

    def median_snr(self, *, antenna: Optional[str] = None, scan_number: Optional[int] = None,
                   polarization: Optional[str] = None) -> float:
        """Return the median finite SNR over a selection (``nan`` if nothing matched)."""
        values = sorted(self.values_for(antenna=antenna, scan_number=scan_number,
                                        polarization=polarization))
        if not values:
            return float("nan")
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

    def rank_scans(self) -> list[tuple[int, float]]:
        """Return ``(scan_number, median_snr)`` pairs, best scan first.

        The median is taken over all antennas and polarizations of the scan, so
        a scan is ranked highly only when the *array as a whole* detected fringes.
        """
        ranked = [(scan, self.median_snr(scan_number=scan)) for scan in self.scan_numbers]
        return sorted([r for r in ranked if r[1] == r[1]], key=lambda r: r[1], reverse=True)

    def rank_antennas(self) -> list[tuple[str, float]]:
        """Return ``(antenna, median_snr)`` pairs, most sensitive antenna first."""
        ranked = [(ant, self.median_snr(antenna=ant)) for ant in self.antennas]
        return sorted([r for r in ranked if r[1] == r[1]], key=lambda r: r[1], reverse=True)

    @property
    def refant_names(self) -> list[str]:
        """Reference antenna(s) used, as a list (the solve may fall back between scans)."""
        return [name for name in str(self.refant).split(",") if name]

    def dead_antennas(self, threshold: float = 3.0) -> list[str]:
        """Return antennas whose median SNR is below ``threshold`` (or entirely absent).

        The reference antenna is never included: its own solutions are a sentinel
        rather than a measurement, so it has no SNR of its own and would always
        look dead — while being, by definition, the best-detected antenna.
        """
        reference = set(self.refant_names)
        dead = []
        for ant in self.antennas:
            if ant in reference:
                continue
            median = self.median_snr(antenna=ant)
            if median != median or median < threshold:
                dead.append(ant)
        return dead

    def __str__(self) -> str:
        return (f"ScanSNRSurvey({self.project_code}: {len(self.scan_numbers)} scans x "
                f"{len(self.antennas)} antennas x {len(self.polarizations)} pols)")


@dataclass
class ObsMetadata:
    """Aggregate metadata for a single observation (one project code).

    Parameters
    ----------
    project_code : str
        Project code.
    obs_date : datetime.date, optional
        Observing date.
    time_range : tuple of float
        ``(start_mjd_sec, end_mjd_sec)``.
    antennas : dict
        Mapping antenna code -> :class:`Antenna`.
    scans : list of Scan
        All scans.
    freq_setup : FreqSetup
        Frequency setup.
    source_names : list of str
        Names of sources present in the data.
    source_coords : dict
        Mapping source name -> ``(ra_deg, dec_deg)`` as read from the data.
    source_ids : dict
        Mapping source name -> FIELD_ID in the data (empty when the backend
        cannot report it).
    snr_survey : ScanSNRSurvey, optional
        Per-scan fringe SNR survey, once ``calibrate.scan_snr()`` has run.
    """

    project_code: str = ""
    obs_date: Optional[dt.date] = None
    time_range: tuple[float, float] = (0.0, 0.0)
    antennas: dict[str, Antenna] = dcfield(default_factory=dict)
    scans: list[Scan] = dcfield(default_factory=list)
    freq_setup: FreqSetup = dcfield(default_factory=FreqSetup)
    source_names: list[str] = dcfield(default_factory=list)
    source_coords: dict[str, tuple[float, float]] = dcfield(default_factory=dict)
    source_ids: dict[str, int] = dcfield(default_factory=dict)
    snr_survey: Optional[ScanSNRSurvey] = None

    @property
    def n_antennas(self) -> int:
        """Number of antennas."""
        return len(self.antennas)

    @property
    def n_scans(self) -> int:
        """Number of scans."""
        return len(self.scans)

    @property
    def observed_antennas(self) -> list[Antenna]:
        """Antennas that actually produced data, in table order."""
        return [a for a in self.antennas.values() if a.observed]

    @property
    def duration_hours(self) -> float:
        """Total span of the observation in hours."""
        return (self.time_range[1] - self.time_range[0]) / 3600.0

    def baseline_lengths(self) -> dict[tuple[str, str], float]:
        """Return the projected-free geometric length in metres of every antenna pair.

        Only antennas with a non-zero ITRF position and actual data are included,
        so the result reflects the array as observed rather than as scheduled.
        """
        usable = [a for a in self.antennas.values() if a.observed and any(a.position)]
        lengths: dict[tuple[str, str], float] = {}
        for first, second in itertools.combinations(usable, 2):
            offset = [p - q for p, q in zip(first.position, second.position)]
            lengths[(first.name, second.name)] = math.sqrt(sum(v * v for v in offset))
        return lengths

    @property
    def max_baseline(self) -> float:
        """Longest baseline in metres (0.0 if positions are unknown)."""
        lengths = self.baseline_lengths()
        return max(lengths.values()) if lengths else 0.0

    @property
    def min_baseline(self) -> float:
        """Shortest baseline in metres (0.0 if positions are unknown)."""
        lengths = self.baseline_lengths()
        return min(lengths.values()) if lengths else 0.0

    @property
    def resolution_mas(self) -> float:
        """Approximate synthesised-beam FWHM in mas: lambda / max_baseline.

        Returns 0.0 when the longest baseline or the reference frequency is unknown.
        """
        if not self.max_baseline or not self.freq_setup.ref_freq:
            return 0.0
        wavelength = _C_M_S / self.freq_setup.ref_freq
        return math.degrees(wavelength / self.max_baseline) * 3.6e6

    @property
    def largest_angular_scale_mas(self) -> float:
        """Approximate largest recoverable angular scale in mas: lambda / min_baseline."""
        if not self.min_baseline or not self.freq_setup.ref_freq:
            return 0.0
        wavelength = _C_M_S / self.freq_setup.ref_freq
        return math.degrees(wavelength / self.min_baseline) * 3.6e6

    def scans_for_source(self, source: str) -> list[Scan]:
        """Return all scans on a given source."""
        return [s for s in self.scans if s.source == source]

    def time_on_source(self, source: str) -> float:
        """Return the total on-source time in seconds for a given source."""
        return sum(s.duration_sec for s in self.scans_for_source(source))

    def antenna_scan_matrix(self) -> dict[str, list[bool]]:
        """Return ``{antenna: [participated_in_scan_0, ...]}`` over all scans."""
        return {name: [name in scan.antennas for scan in self.scans] for name in self.antennas}
