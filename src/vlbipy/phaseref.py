"""Phase referencing: which sources the phases are solved on, in which order, and how they reach the others.

The settings live in ``[phase_referencing]``, next to the optional target -> calibrator map (its other keys)::

    [phase_referencing]
    inverse = false          # solve on the target(s), transfer to the phase calibrators
    strategy = "auto"        # "auto" | "sequential" | "interpolate" | "time"
    brightness_ratio = 3.0   # "auto": sequential when the brightest reference is this many times brighter
    order = []               # "sequential": solve order (default: brightest first)
    max_extrapolation = 1.0  # "interpolate": furthest beyond either reference, in units of their separation
    max_gap = 600.0          # "interpolate": both references observed within this many seconds of a receiver time

The *reference sources* are the ones the phases are solved on and transferred from: the phase calibrators,
or, with ``inverse``, the targets (inverse phase referencing, for a target bright enough to self-calibrate).
Only the calibration roles swap: "target" still names the source of interest in every product and report.
The *receivers* are every other observed field except the fringe finders.

With several reference sources:

``sequential``
    The brightest is solved first and its solutions go to every field. Each next one is then solved on data
    already corrected by the ones before it, and its solutions - a second-order correction - go to itself,
    the later references and the receivers.
``interpolate``
    Every reference is solved on its own. A receiver takes the solutions of the two references nearest to it
    combined linearly along the line joining them on the sky: with ``x`` the projection of the receiver onto
    that line (0 at the first reference, 1 at the second), phase, delay, rate and dispersive delay are
    ``(1 - x) A + x B`` (the phase difference unwrapped in time), extrapolated when ``x`` is outside [0, 1].
    Amplitudes are interpolated with ``x`` clipped to [0, 1]: they are not direction dependent. Each
    reference's solutions have to be carried in time to the receiver first, so a receiver time either
    reference was not observed within ``max_gap`` of (before and after) takes the time interpolation instead:
    at X band, carrying a reference over 15 min costs more than the geometry gains (V589A).
``time``
    Every receiver takes the solutions of whichever reference was observed closest in time (CASA's linear
    time interpolation across all of them), as vlbipy did before these strategies existed.
``auto``
    ``sequential`` when the brightest reference has at least ``brightness_ratio`` times the fringe SNR of the
    next one; otherwise ``interpolate`` when both references nearest each receiver bracket its scans within
    ``max_gap`` (median over the scans), ``time`` when they do not (a schedule that pairs every target scan
    with one calibrator).

The strategy decides how the references are solved (fringe fit, self-calibration) and how their tables are
applied: ``sequential`` and ``interpolate`` need a re-run from the fringe fit when changed. The receivers of
``interpolate`` get *derived* tables (:func:`interpolated_table`), built when the chain is applied
(:func:`transfer_chain`), so both backends and CASA's own applycal apply them like any other table.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .errors import ConfigError
from .logging_utils import get_logger
from .models import CalTable

logger = get_logger()

#: Keys of ``[phase_referencing]`` that are settings, not target names of the target -> calibrator map.
SETTING_KEYS = ("inverse", "strategy", "brightness_ratio", "order", "max_extrapolation", "max_gap")
STRATEGIES = ("auto", "sequential", "interpolate", "time")
#: Self-calibration tables registered once per reference source, as ``<kind>_<source>``.
PER_SOURCE_KINDS = ("selfcal", "selfamp", "selfphase", "joint")
#: Spacing of the rows of a derived table inside each scan of its field [s].
DERIVED_CADENCE = 30.0
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class PhaseRefSettings:
    """The settings of ``[phase_referencing]`` (see the module docstring)."""

    inverse: bool = False
    strategy: str = "auto"
    brightness_ratio: float = 3.0
    order: tuple = ()
    max_extrapolation: float = 1.0
    max_gap: float = 600.0

    @classmethod
    def from_config(cls, section: Optional[dict]) -> "PhaseRefSettings":
        """Read and check the settings of a ``[phase_referencing]`` section (other keys are ignored)."""
        section = dict(section or {})
        strategy = str(section.get("strategy", "auto")).strip().lower()
        if strategy not in STRATEGIES:
            raise ConfigError(f"[phase_referencing] strategy = {section.get('strategy')!r}; use one of "
                              f"{', '.join(STRATEGIES)}")
        ratio = float(section.get("brightness_ratio", 3.0))
        if ratio < 1.0:
            raise ConfigError(f"[phase_referencing] brightness_ratio must be >= 1, got {ratio}")
        extrapolation = float(section.get("max_extrapolation", 1.0))
        if extrapolation < 0:
            raise ConfigError(f"[phase_referencing] max_extrapolation must be >= 0, got {extrapolation}")
        max_gap = float(section.get("max_gap", 600.0))
        if max_gap <= 0:
            raise ConfigError(f"[phase_referencing] max_gap must be positive (seconds), got {max_gap}")
        order = section.get("order") or ()
        order = (order,) if isinstance(order, str) else tuple(str(name) for name in order)
        inverse = section.get("inverse", False)
        if not isinstance(inverse, bool):
            raise ConfigError(f"[phase_referencing] inverse must be true or false, got {inverse!r}")
        return cls(inverse=inverse, strategy=strategy, brightness_ratio=ratio, order=order,
                   max_extrapolation=extrapolation, max_gap=max_gap)

    def to_dict(self) -> dict:
        return {"inverse": self.inverse, "strategy": self.strategy, "brightness_ratio": self.brightness_ratio,
                "order": list(self.order), "max_extrapolation": self.max_extrapolation, "max_gap": self.max_gap}


def target_map(section: Optional[dict]) -> dict:
    """The target -> calibrator(s) entries of a ``[phase_referencing]`` section (the settings left out)."""
    return {key: value for key, value in (section or {}).items() if key not in SETTING_KEYS}


@dataclass
class PhaseRefPlan:
    """How the phases of one observation are solved and transferred.

    Attributes
    ----------
    strategy : str
        ``"single"`` (one reference source: nothing to choose), ``"sequential"``, ``"interpolate"`` or
        ``"time"``.
    references : list of str
        The sources the phases are solved on, in solve order (brightest first unless ``order`` says otherwise).
    receivers : list of str
        The fields that only receive solutions (fringe finders excluded).
    inverse : bool
        Whether the references are the targets.
    brightness : dict
        Fringe SNR of each reference that the decision used (empty when unknown).
    reason : str
        Why this strategy, for the log and the report.
    max_extrapolation, max_gap : float
        As in the settings.
    """

    strategy: str
    references: list = field(default_factory=list)
    receivers: list = field(default_factory=list)
    inverse: bool = False
    brightness: dict = field(default_factory=dict)
    reason: str = ""
    max_extrapolation: float = 1.0
    max_gap: float = 600.0

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "references": list(self.references), "receivers": list(self.receivers),
                "inverse": self.inverse, "brightness": dict(self.brightness), "reason": self.reason,
                "max_extrapolation": self.max_extrapolation, "max_gap": self.max_gap}

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseRefPlan":
        return cls(strategy=data["strategy"], references=list(data.get("references", [])),
                   receivers=list(data.get("receivers", [])), inverse=bool(data.get("inverse", False)),
                   brightness=dict(data.get("brightness", {})), reason=str(data.get("reason", "")),
                   max_extrapolation=float(data.get("max_extrapolation", 1.0)),
                   max_gap=float(data.get("max_gap", 600.0)))

    def apply_to(self, source: str, others: Iterable[str]) -> Optional[str]:
        """The fields the self-calibration tables of reference ``source`` go to; None = the default ones.

        ``sequential``: itself, the references after it and the receivers. ``interpolate``: itself and the
        receivers (each reference is calibrated on its own; :func:`transfer_chain` replaces the receivers'
        copies by the interpolated ones). Otherwise None: every field ``others`` lists.
        """
        if source not in self.references or self.strategy not in ("sequential", "interpolate"):
            return None
        later = self.references[self.references.index(source) + 1:] if self.strategy == "sequential" else []
        wanted = {source, *later, *self.receivers}
        return ",".join(name for name in others if name in wanted)


def resolve_plan(settings: PhaseRefSettings, references: list, receivers: list,
                 brightness: Optional[dict] = None, sampling: Optional[float] = None) -> PhaseRefPlan:
    """Decide the strategy and the solve order of ``references`` (see the module docstring).

    ``brightness`` maps a source to its fringe SNR (any quantity proportional to its flux density will do);
    sources it does not have are of unknown brightness. ``sampling`` is how closely the references bracket
    the receivers (:func:`median_sampling`, seconds; None = unknown).
    """
    brightness = {name: float(value) for name, value in (brightness or {}).items() if name in references}
    known = all(brightness.get(name, 0.0) > 0 for name in references)
    ranked = sorted(references, key=lambda name: -brightness[name]) if known else list(references)
    listed = [name for name in settings.order if name in references]
    order = listed + [name for name in ranked if name not in listed]
    common = dict(references=order, receivers=list(receivers), inverse=settings.inverse,
                  brightness=brightness, max_extrapolation=settings.max_extrapolation,
                  max_gap=settings.max_gap)
    if len(order) <= 1:
        return PhaseRefPlan("single", reason="one reference source", **common)
    if settings.strategy != "auto":
        reason = f"strategy '{settings.strategy}' requested"
        if settings.strategy == "sequential" and not listed and not known:
            reason += "; brightness unknown, solving in the configured order"
        return PhaseRefPlan(settings.strategy, reason=reason, **common)
    if known:
        ratio = brightness[ranked[0]] / brightness[ranked[1]]
        if ratio >= settings.brightness_ratio:
            return PhaseRefPlan("sequential", reason=f"{ranked[0]} is {ratio:.1f}x brighter than {ranked[1]} "
                                                     f"(>= {settings.brightness_ratio:g})", **common)
        reason = f"{ranked[0]} is only {ratio:.1f}x brighter than {ranked[1]} (< {settings.brightness_ratio:g})"
    else:
        reason = "brightness of the references unknown"
    if sampling is not None and sampling > settings.max_gap:
        return PhaseRefPlan("time", reason=f"{reason}; but the references bracket the receiver scans only "
                                           f"within {sampling / 60:.0f} min (median; max_gap "
                                           f"{settings.max_gap / 60:.0f} min), too sparse to interpolate", **common)
    return PhaseRefPlan("interpolate", reason=reason, **common)


# ----------------------------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------------------------
def _tangent(ra_deg, dec_deg, ra0_deg, dec0_deg) -> np.ndarray:
    """Gnomonic (tangent-plane) offsets [rad] of a direction from the centre (ra0, dec0)."""
    ra, dec, ra0, dec0 = np.radians([ra_deg, dec_deg, ra0_deg, dec0_deg])
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    x = np.cos(dec) * np.sin(ra - ra0) / cos_c
    y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    return np.array([x, y])


def separation_deg(a, b) -> float:
    """Angular distance [deg] between two (ra, dec) [deg]."""
    ra1, dec1, ra2, dec2 = np.radians([a[0], a[1], b[0], b[1]])
    cos = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def projection(a, b, receiver) -> tuple[float, float]:
    """Position of ``receiver`` along the line from ``a`` to ``b`` (all (ra, dec) [deg]).

    Returns ``(x, offset)``: ``x`` is 0 at ``a`` and 1 at ``b`` (the projection on the tangent plane centred
    on the receiver, where the 1-D phase model is evaluated) and ``offset`` the distance [deg] of the
    receiver from that line.
    """
    pa = _tangent(a[0], a[1], receiver[0], receiver[1])
    pb = _tangent(b[0], b[1], receiver[0], receiver[1])
    line = pb - pa
    length2 = float(line @ line)
    if length2 == 0:
        raise ValueError("the two reference sources are at the same position")
    x = float(-pa @ line) / length2
    offset = float(abs(line[0] * -pa[1] - line[1] * -pa[0]) / np.sqrt(length2))
    return x, float(np.degrees(offset))


def interpolation_pair(references: list, coords: dict, receiver: str,
                       max_extrapolation: float = 1.0) -> tuple[str, str, float, float]:
    """The two references nearest ``receiver`` on the sky and its weight ``x`` between them.

    ``x`` is clipped to ``[-max_extrapolation, 1 + max_extrapolation]``. Returns ``(a, b, x, offset_deg)``.
    """
    usable = [name for name in references if name in coords]
    if len(usable) < 2 or receiver not in coords:
        raise ValueError(f"interpolating to {receiver} needs the positions of it and of two references")
    a, b = sorted(usable, key=lambda name: separation_deg(coords[name], coords[receiver]))[:2]
    x, offset = projection(coords[a], coords[b], coords[receiver])
    clipped = float(np.clip(x, -max_extrapolation, 1.0 + max_extrapolation))
    if clipped != x:
        logger.warning("phaseref: {} projects at x = {:.2f} between {} and {}; extrapolation limited to {:.2f}",
                       receiver, x, a, b, clipped)
    return a, b, clipped, offset


# ----------------------------------------------------------------------------------------------------------------
# Derived tables
# ----------------------------------------------------------------------------------------------------------------
def receiver_times(scans, source: str, cadence: float = DERIVED_CADENCE) -> np.ndarray:
    """Times [MJD s] for the rows of a derived table: every ``cadence`` seconds through each scan of ``source``
    (its start and end included)."""
    times = []
    for scan in scans:
        if scan.source != source:
            continue
        start, end = float(scan.time_start), float(max(scan.time_end, scan.time_start))
        steps = max(1, int(np.ceil((end - start) / cadence)))
        times.append(np.linspace(start, end, steps + 1))
    return np.unique(np.concatenate(times)) if times else np.zeros(0)


def bracket_gaps(solution_times, times) -> np.ndarray:
    """How far in time each of ``times`` is from the solutions around it: the larger of the distances to the
    last solution before and the first one after [s]; infinite where there is none on one side."""
    solution_times = np.unique(np.asarray(solution_times, dtype=np.float64))
    times = np.asarray(times, dtype=np.float64)
    if not solution_times.size:
        return np.full(times.shape, np.inf)
    after = np.searchsorted(solution_times, times, side="left")
    before = np.searchsorted(solution_times, times, side="right") - 1
    to_before = np.where(before >= 0, times - solution_times[np.maximum(before, 0)], np.inf)
    to_after = np.where(after < solution_times.size,
                        solution_times[np.minimum(after, solution_times.size - 1)] - times, np.inf)
    return np.maximum(to_before, to_after)


def median_sampling(scans, references: list, receivers: list, coords: dict) -> Optional[float]:
    """How closely the references bracket the receivers in time: over every receiver scan, the larger
    :func:`bracket_gaps` of the two references nearest the receiver on the sky (at the scan centres); the
    median over all those scans [s], or None when there is nothing to measure."""
    def centres(name):
        return np.array([(s.time_start + s.time_end) / 2 for s in scans if s.source == name], dtype=np.float64)

    gaps = []
    for receiver in receivers:
        try:
            a, b, _, _ = interpolation_pair(references, coords, receiver, np.inf)
        except ValueError:
            continue
        own = centres(receiver)
        if own.size:
            gaps.append(np.maximum(bracket_gaps(centres(a), own), bracket_gaps(centres(b), own)))
    return float(np.median(np.concatenate(gaps))) if gaps else None


def _unwrap_in_time(delta, flags, group, times, max_gap: float = 3 * DERIVED_CADENCE):
    """The phase difference ``delta`` [rad] between two references, made continuous where it can be.

    Inside a stretch of points no more than ``max_gap`` apart (a scan of the receiver, sampled every
    :data:`DERIVED_CADENCE` seconds) it is unwrapped, so the interpolated phase has no jump within the scan.
    Between stretches it is not: two references a few degrees apart differ by a hundred degrees and more from
    one scan to the next at X band (V589), and unwrapping across those gaps adds spurious turns. Each stretch
    is put on the branch whose median lies in (-pi, pi], the smallest phase gradient between the references.
    """
    out = np.array(delta, dtype=np.float64)
    order = np.lexsort((times, group))
    breaks = (np.diff(group[order]) != 0) | (np.diff(times[order]) > max_gap)
    for rows in np.split(order, np.flatnonzero(breaks) + 1):
        good = rows[~flags[rows]]
        if good.size:
            unwrapped = np.unwrap(np.angle(np.exp(1j * out[good])))
            out[good] = unwrapped - TWO_PI * np.round(np.median(unwrapped) / TWO_PI)
    return out


def _fringe_rates(table, antenna, times, spw_t, field_id):
    """Rate [s/s] of both hands at points, interpolated linearly in time as the other fringe parameters are."""
    from .solvers.apply import FRINGE_NPARAM_PER_POL, _brackets, _lookup
    look = _lookup(table, field_id)
    rates = np.zeros((times.size, 2))
    if not look["rows"].size:
        return rates
    i0, i1, w, _ = _brackets(look, antenna, spw_t, times, "linear")
    row0, row1 = look["rows"][i0], look["rows"][i1]
    for hand in range(2):
        column = FRINGE_NPARAM_PER_POL * hand + 2
        r0 = np.real(table.param[row0, 0, column]).astype(np.float64)
        r1 = np.real(table.param[row1, 0, column]).astype(np.float64)
        rates[:, hand] = (1.0 - w) * r0 + w * r1
    return rates


def _evaluate(table, field_ids, antenna, times, spw):
    """Parameters of ``table`` (rows of the fields ``field_ids``, interpolated linearly in time across them) at
    points given in its own spws: (npoint, 2) arrays."""
    from .solvers.apply import _fringe_grid, _table_gains
    identity = np.arange(table.spw_chan_freq.shape[0])       # the points are in table spws already
    if table.kind == "Fringe Jones":
        phase, delay, disp, flag, _ = _fringe_grid(table, antenna, times, spw, identity, field_id=list(field_ids),
                                                   interp="linear")
        rate = _fringe_rates(table, antenna, times, spw, list(field_ids))
        return {"phase": phase, "delay": delay, "rate": rate, "disp": disp, "flag": flag}
    values, flag, _ = _table_gains(table, antenna, times, spw, identity, table.spw_chan_freq,
                                   field_id=list(field_ids), interp="linear")
    return {"gain": values, "flag": flag}


def _combine(first: dict, second: dict, x: float, kind: str, group, times) -> dict:
    """``(1 - x) first + x second`` per point and hand; a point only one of them has takes that one."""
    both = ~first["flag"] & ~second["flag"]
    only_second = first["flag"] & ~second["flag"]
    flag = first["flag"] & second["flag"]
    out = {"flag": flag}
    if kind == "Fringe Jones":
        delta = np.zeros_like(first["phase"])
        for hand in range(2):
            raw = second["phase"][:, hand] - first["phase"][:, hand]
            delta[:, hand] = _unwrap_in_time(raw, ~both[:, hand], group, times)
        out["phase"] = np.where(both, first["phase"] + x * delta, np.where(only_second, second["phase"],
                                                                           first["phase"]))
        for name in ("delay", "rate", "disp"):
            out[name] = np.where(both, (1.0 - x) * first[name] + x * second[name],
                                 np.where(only_second, second[name], first[name]))
        return out
    weight = float(np.clip(x, 0.0, 1.0))
    ga, gb = first["gain"], second["gain"]
    delta = np.zeros(ga.shape)
    for hand in range(2):
        raw = np.angle(gb[:, hand] * np.conj(ga[:, hand]))
        delta[:, hand] = _unwrap_in_time(raw, ~both[:, hand], group, times)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.exp((1.0 - weight) * np.log(np.abs(ga)) + weight * np.log(np.abs(gb)))
    blended = amplitude * np.exp(1j * (np.angle(ga) + x * delta))
    out["gain"] = np.where(both, blended, np.where(only_second, gb, ga))
    return out


def interpolated_table(first: tuple, second: tuple, x: float, receiver: str, times: np.ndarray,
                       out_path, max_gap: Optional[float] = None) -> dict:
    """Write the table ``receiver`` takes between two references, and return a summary of it.

    Parameters
    ----------
    first, second : tuple
        ``(table path, field name)`` of the solutions of each reference (the same table for a fringe fit
        solved on both, two tables for two self-calibrations).
    x : float
        Weight of ``second`` (0 = only ``first``; outside [0, 1] extrapolates the phases).
    receiver : str
        Field the rows are written for (looked up in the FIELD subtable of ``first``).
    times : array
        Row times [MJD s] (:func:`receiver_times`).
    out_path : path
        The table to write; it is a copy of ``first`` (keywords, subtables, column layout) with new rows.
    max_gap : float, optional
        A time either reference has no solution within this many seconds of, before and after
        (:func:`bracket_gaps`), takes the time interpolation instead: of both references' solutions together
        when they are one table (what the ``time`` strategy applies), of the reference observed closer in time
        otherwise.

    Returns
    -------
    dict
        ``rows``, ``flagged`` (fraction of rows neither reference could give), ``one_sided`` (fraction only
        one could), ``time_interpolated`` (fraction beyond ``max_gap``) and ``kind``.
    """
    import casatools
    from .solvers.apply import load_caltable
    tables = {}
    for path, _ in (first, second):
        if path not in tables:
            tables[path] = load_caltable(path)
    ta, tb_ = tables[first[0]], tables[second[0]]
    if ta.kind != tb_.kind:
        raise ValueError(f"cannot interpolate a {ta.kind} table with a {tb_.kind} one")
    if ta.kind not in ("Fringe Jones", "G Jones", "T Jones") or ta.nchan_t != 1:
        raise ValueError(f"{ta.path}: only Fringe Jones and channel-less G/T Jones tables can be interpolated")
    if list(ta.antenna_names) != list(tb_.antenna_names):
        raise ValueError("the two tables have different antennas")
    names_a, names_b = [str(n) for n in ta.field_names], [str(n) for n in tb_.field_names]
    if first[1] not in names_a or second[1] not in names_b or receiver not in names_a:
        raise ValueError(f"field {first[1]!r}, {second[1]!r} or {receiver!r} missing from the tables' FIELD")
    id_a, id_b, id_out = names_a.index(first[1]), names_b.index(second[1]), names_a.index(receiver)
    rows_a = ta.field_id == id_a
    rows_b = tb_.field_id == id_b
    antennas = np.union1d(ta.antenna1[rows_a], tb_.antenna1[rows_b])
    spws = np.union1d(ta.spw_id[rows_a], tb_.spw_id[rows_b])
    antennas, spws = antennas[antennas >= 0], spws[spws >= 0]
    times = np.asarray(times, dtype=np.float64)
    grid_a, grid_s, grid_t = np.meshgrid(antennas, spws, times, indexing="ij")
    antenna, spw, t = grid_a.ravel(), grid_s.ravel(), grid_t.ravel()
    group = antenna * (int(spws.max(initial=0)) + 1) + spw
    va = _evaluate(ta, [id_a], antenna, t, spw)
    vb = _evaluate(tb_, [id_b], antenna, t, spw)
    out = _combine(va, vb, x, ta.kind, group, t)
    sparse = np.zeros(t.shape, dtype=bool)
    if max_gap is not None:
        gap_a, gap_b = bracket_gaps(ta.time[rows_a], t), bracket_gaps(tb_.time[rows_b], t)
        sparse = (gap_a > max_gap) | (gap_b > max_gap)
        if sparse.any():
            if first[0] == second[0]:
                fallback = _evaluate(ta, [id_a, id_b], antenna[sparse], t[sparse], spw[sparse])
            else:
                closer = (gap_b < gap_a)[sparse][:, None]
                fallback = {key: np.where(closer, vb[key][sparse], va[key][sparse]) for key in va}
            for key in out:
                out[key][sparse] = fallback[key]
    nrow = t.size
    flag = out["flag"]
    if ta.kind == "Fringe Jones":
        param = np.zeros((nrow, 1, 8))
        for hand in range(2):
            base = 4 * hand
            param[:, 0, base] = np.angle(np.exp(1j * out["phase"][:, hand]))
            param[:, 0, base + 1] = out["delay"][:, hand]
            param[:, 0, base + 2] = out["rate"][:, hand]
            param[:, 0, base + 3] = out["disp"][:, hand]
        flags = np.repeat(flag, 4, axis=1)[:, None, :]
        column = "FPARAM"
    else:
        param = out["gain"][:, None, :]
        flags = flag[:, None, :]
        column = "CPARAM"
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(first[0]).absolute()
    handle = casatools.table()
    handle.open(str(source))
    try:
        # The main table without its rows; ``norows`` empties the subtables too, so they are copied whole.
        handle.copy(str(out_path.absolute()), deep=True, valuecopy=True, norows=True).close()
        subtables = [name for name, value in handle.getkeywords().items()
                     if isinstance(value, str) and value.startswith("Table:")]
    finally:
        handle.close()
    for name in subtables:
        shutil.rmtree(out_path / name, ignore_errors=True)
        handle.open(str(source / name))
        try:
            handle.copy(str((out_path / name).absolute()), deep=True, valuecopy=True).close()
        finally:
            handle.close()
    handle = casatools.table()
    handle.open(str(out_path), nomodify=False)
    try:
        handle.addrows(nrow)
        handle.putcol("TIME", t)
        handle.putcol("INTERVAL", np.zeros(nrow))
        handle.putcol("FIELD_ID", np.full(nrow, id_out, dtype=np.int32))
        handle.putcol("SPECTRAL_WINDOW_ID", spw.astype(np.int32))
        handle.putcol("ANTENNA1", antenna.astype(np.int32))
        handle.putcol("ANTENNA2", np.full(nrow, -1, dtype=np.int32))
        handle.putcol("SCAN_NUMBER", np.zeros(nrow, dtype=np.int32))
        handle.putcol("OBSERVATION_ID", np.zeros(nrow, dtype=np.int32))
        # casatools takes Fortran order: (npar, nchan, nrow).
        dtype = np.float32 if column == "FPARAM" else np.complex64
        handle.putcol(column, param.astype(dtype).transpose(2, 1, 0))
        handle.putcol("PARAMERR", np.zeros(param.shape, dtype=np.float32).transpose(2, 1, 0))
        handle.putcol("SNR", np.where(flags, 0.0, 1.0).astype(np.float32).transpose(2, 1, 0))
        handle.putcol("FLAG", flags.transpose(2, 1, 0))
    finally:
        handle.close()
    one_sided = float(np.mean((va["flag"] ^ vb["flag"]))) if nrow else 0.0
    summary = {"rows": int(nrow), "kind": ta.kind, "flagged": float(np.mean(flag)) if nrow else 0.0,
               "one_sided": one_sided, "time_interpolated": float(np.mean(sparse)) if nrow else 0.0}
    logger.info("phaseref: {} -> {} ({} rows, x = {:.3f} between {} and {}; {:.0%} beyond max_gap, so time "
                "interpolated; {:.0%} from one reference only, {:.0%} from none)",
                Path(first[0]).name if first[0] == second[0] else f"{Path(first[0]).name} + {Path(second[0]).name}",
                receiver, nrow, x, first[1], second[1], summary["time_interpolated"], summary["one_sided"],
                summary["flagged"])
    return summary


# ----------------------------------------------------------------------------------------------------------------
# The apply chain
# ----------------------------------------------------------------------------------------------------------------
def _signature(paths, extra) -> str:
    """Digest of the inputs of a derived table: the files of its source tables and its parameters."""
    from .solvers.corrected import table_signature
    payload = [table_signature(path) for path in paths] + [extra]
    return hashlib.md5(json.dumps(payload, default=str).encode()).hexdigest()


def _derive_cached(sources: tuple, x: float, receiver: str, times: np.ndarray, out_path: Path,
                   max_gap: Optional[float] = None) -> bool:
    """Build a derived table unless the one on disk was made from exactly these inputs; True when usable."""
    stamp = out_path.with_name(out_path.name + ".json")
    signature = _signature(sorted({path for path, _ in sources}),
                           [sources, round(x, 9), receiver, times.size, float(times[0]), float(times[-1]), max_gap])
    if out_path.is_dir() and stamp.is_file() and json.loads(stamp.read_text()).get("signature") == signature:
        return True
    try:
        summary = interpolated_table(sources[0], sources[1], x, receiver, times, out_path, max_gap=max_gap)
    except Exception as exc:  # noqa: BLE001 - the receiver keeps the tables it had
        logger.warning("phaseref: could not interpolate {} to {} ({}); it keeps the time-interpolated "
                       "solutions", ", ".join(f"{Path(p).name}[{f}]" for p, f in sources), receiver, exc)
        return False
    stamp.write_text(json.dumps({"signature": signature, "x": x, "sources": sources, "summary": summary},
                                indent=1))
    return True


def _without(apply_to: str, every: list, dropped: str) -> str:
    """``apply_to`` (empty = every field) with ``dropped`` removed."""
    names = [name for name in (apply_to.split(",") if apply_to else every) if name]
    return ",".join(name for name in names if name != dropped)


def transfer_chain(tables: list, plan: PhaseRefPlan, metadata, out_dir) -> list:
    """The apply chain with the receivers' solutions interpolated between two references (``interpolate``).

    Two kinds of tables are interpolated: a table solved on several references (a fringe fit: its
    ``gainfield`` names them) and the per-reference self-calibration tables (``selfcal_<source>``,
    ``selfamp_<source>``, ``selfphase_<source>``, ``joint_<source>``), paired by kind. For every receiver a derived table
    (:func:`interpolated_table`) is appended with ``apply_to`` that receiver, and the receiver is removed from
    the ``apply_to`` of the tables it replaces. Any other strategy returns ``tables`` unchanged, and so does a
    receiver whose derived table could not be made.
    """
    if plan.strategy != "interpolate" or metadata is None or len(plan.references) < 2:
        return list(tables)
    coords = dict(getattr(metadata, "source_coords", {}) or {})
    every = list(getattr(metadata, "source_names", []) or [])
    refs = set(plan.references)
    out_dir = Path(out_dir)
    chain = [replace(table) for table in tables]
    added: list = []
    per_kind: dict = {}
    for table in chain:
        for kind in PER_SOURCE_KINDS:
            source = table.cal_type[len(kind) + 1:] if table.cal_type.startswith(f"{kind}_") else ""
            if source in refs and table.gainfield == source:
                per_kind.setdefault(kind, {})[source] = table
    for receiver in plan.receivers:
        if receiver not in every:
            continue
        try:
            a, b, x, offset = interpolation_pair(plan.references, coords, receiver, plan.max_extrapolation)
        except ValueError as exc:
            logger.warning("phaseref: {}", exc)
            continue
        times = receiver_times(metadata.scans, receiver)
        if not times.size:
            continue
        logger.info("phaseref: {} at x = {:.3f} between {} and {} ({:.2f} deg off their line)", receiver, x, a,
                    b, offset)
        for table in chain:
            solved = {name.strip() for name in table.gainfield.split(",") if name.strip()}
            if not ({a, b} <= solved and solved <= refs):
                continue
            if table.apply_to and receiver not in table.apply_to.split(","):
                continue
            path = out_dir / f"{Path(table.path).name}.{receiver}"
            if _derive_cached(((table.path, a), (table.path, b)), x, receiver, times, path, plan.max_gap):
                table.apply_to = _without(table.apply_to, every, receiver)
                added.append(CalTable(cal_type=f"{table.cal_type}>{receiver}", path=str(path), field=f"{a},{b}",
                                      gainfield=receiver, interp=table.interp, spwmap=list(table.spwmap),
                                      step=table.step, calwt=table.calwt, apply_to=receiver))
        for kind, by_source in per_kind.items():
            if a not in by_source or b not in by_source:
                continue
            ta, tb_ = by_source[a], by_source[b]
            path = out_dir / f"{kind}.{a}+{b}.{receiver}"
            if _derive_cached(((ta.path, a), (tb_.path, b)), x, receiver, times, path, plan.max_gap):
                for table in (ta, tb_):
                    table.apply_to = _without(table.apply_to, every, receiver)
                added.append(CalTable(cal_type=f"{kind}>{receiver}", path=str(path), field=f"{a},{b}",
                                      gainfield=receiver, interp="linear", step=ta.step, calwt=ta.calwt,
                                      apply_to=receiver))
    return chain + added
