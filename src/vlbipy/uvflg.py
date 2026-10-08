"""Tolerant reader for AIPS ``UVFLG`` text files, and their conversion to CASA flag commands.

The a-priori flag file of an EVN experiment is written by one program and is
regular. An LBA one is the concatenation of per-station files written by
different tools, and mixes dialects freely within one file::

    ant_name='AT' timerang=82,10,00,00, 82,10,01,46 reason='SLEWING' /
    opcode = 'FLAG'
    dtimrang = 1  timeoff = 0
    ant_name='Ho' timerang= 82,11,05,59,  82,11,06,02  reason='Slewing expected.' /
    ant_name ='PA', timerang=082,23,03,06, 082,23,03,08 reason='DISH IS STATIONARY' /

(header keywords with no terminating slash, commas between keywords, zero-padded
day numbers, mixed-case station codes). A strict key-file parser gives up on
those, so this one reads each ``/``-terminated record on its own and keeps only
what a flag needs: the antenna, the time range, and optionally the subbands,
channels and reason.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterable, Optional, Union

from .logging_utils import get_logger

logger = get_logger()

#: ``key = value`` pairs: a quoted string, or a comma-separated list of numbers.
_PAIR = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*('[^']*'|\"[^\"]*\"|[-+0-9.eE]+(?:\s*,\s*[-+0-9.eE]+)*)")


def _parse_value(text: str):
    """Return a quoted string unquoted, or a number / list of numbers."""
    if text[:1] in "'\"":
        return text[1:-1].strip()
    numbers = [float(v) for v in text.split(",")]
    return numbers[0] if len(numbers) == 1 else numbers


def parse_uvflg(path: Union[str, Path]) -> list[dict]:
    """Read the flag records of an AIPS ``UVFLG`` text file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.uvflg`` file.

    Returns
    -------
    list of dict
        One dict per record that names an antenna, with upper-case keys
        (``ANT_NAME``, ``TIMERANG``, ``BIF``, ``EIF``, ``BCHAN``, ``ECHAN``,
        ``REASON``, ...). Records that cannot be read are counted and skipped.
    """
    records: list[dict] = []
    skipped = 0
    pending = ""
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        pending = f"{pending} {line}" if pending else line
        if not line.endswith("/"):
            # A header keyword line (opcode, dtimrang, ...) has no slash and no antenna: it
            # must not be glued onto the record that follows it.
            if "ant_name" not in pending.lower():
                pending = ""
            continue
        record = {key.upper(): _parse_value(value) for key, value in _PAIR.findall(pending)}
        pending = ""
        if "ANT_NAME" in record:
            records.append(record)
        else:
            skipped += 1
    if skipped:
        logger.warning("{}: {} record(s) without an antenna name were skipped", Path(path).name, skipped)
    return records


def _timerange(values, year: int) -> Optional[str]:
    """Return a CASA time range for an AIPS ``d,h,m,s, d,h,m,s`` range (days are day-of-year).

    ``""`` means the whole observation, ``None`` a range that selects nothing.
    """
    if not isinstance(values, list) or len(values) != 8:
        return ""
    if values[:4] == [0.0] * 4 and values[4] >= 400.0:
        return ""
    start_of_year = dt.datetime(year, 1, 1)
    edges = [start_of_year + dt.timedelta(days=values[i] - 1, hours=values[i + 1], minutes=values[i + 2],
                                          seconds=values[i + 3]) for i in (0, 4)]
    if edges[1] <= edges[0]:
        return None
    return "~".join(edge.strftime("%Y/%m/%d/%H:%M:%S") for edge in edges)


def uvflg_to_casa(path: Union[str, Path], outfile: Union[str, Path], year: int, *,
                  antennas: Optional[Iterable[str]] = None, max_seconds: float = 0.0) -> dict:
    """Convert an AIPS ``UVFLG`` text file into a CASA flag-command file (``flagdata(mode='list')``).

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.uvflg`` file.
    outfile : str or pathlib.Path
        The CASA flag-command file to write.
    year : int
        Year of the observation (the file only gives the day of the year).
    antennas : iterable of str, optional
        Antenna names present in the data. Records naming any other station are
        dropped: a single unknown antenna makes ``flagdata`` reject the whole list.
    max_seconds : float
        Records longer than this are not converted (0 = no limit). These files flag
        slews and settling, seconds to minutes long; a station log that leaves an
        interval open produces one record running to the end of the experiment,
        which would delete the antenna. An antenna that really was absent for hours
        has no fringes and is dropped by the calibration anyway.

    Returns
    -------
    dict
        ``written`` (number of commands), ``per_antenna`` (commands per antenna),
        ``unknown`` (stations dropped), ``empty`` (zero-length ranges dropped),
        ``seconds`` (total flagged time per antenna) and ``too_long`` (the commands
        left out for exceeding ``max_seconds``).
    """
    known = {name.upper(): name for name in antennas} if antennas is not None else None
    per_antenna: dict[str, int] = {}
    seconds: dict[str, float] = {}
    unknown: set[str] = set()
    empty = 0
    lines = []
    too_long: list[str] = []
    for record in parse_uvflg(path):
        name = str(record["ANT_NAME"]).upper()
        if known is not None:
            if name not in known:
                unknown.add(name)
                continue
            name = known[name]
        timerange = _timerange(record.get("TIMERANG"), year)
        if timerange is None:
            empty += 1
            continue
        command = f"antenna='{name}'"
        if timerange:
            command += f" timerange='{timerange}'"
            start, end = (dt.datetime.strptime(edge, "%Y/%m/%d/%H:%M:%S") for edge in timerange.split("~"))
            duration = (end - start).total_seconds()
            if max_seconds > 0 and duration > max_seconds:
                too_long.append(f"{command} reason='{record.get('REASON', '')}' ({duration / 3600.0:.1f} h)")
                continue
            seconds[name] = seconds.get(name, 0.0) + duration
        spw = ""
        if "BIF" in record and record["BIF"] > 0:
            spw = f"{int(record['BIF']) - 1}~{int(record.get('EIF', record['BIF'])) - 1}"
        if "BCHAN" in record and record["BCHAN"] > 0:
            spw = f"{spw or '*'}:{int(record['BCHAN']) - 1}~{int(record.get('ECHAN', record['BCHAN'])) - 1}"
        if spw:
            command += f" spw='{spw}'"
        reason = re.sub(r"[^A-Za-z0-9]+", "_", str(record.get("REASON", ""))).strip("_")
        if reason:
            command += f" reason='{reason}'"
        lines.append(command)
        per_antenna[name] = per_antenna.get(name, 0) + 1
    Path(outfile).write_text("\n".join(lines) + ("\n" if lines else ""))
    if unknown:
        logger.warning("{}: flags for station(s) not in the data were dropped: {}", Path(path).name,
                       ", ".join(sorted(unknown)))
    logger.info("{} -> {}: {} flag command(s) ({})", Path(path).name, Path(outfile).name, len(lines),
                ", ".join(f"{name} {count}" for name, count in sorted(per_antenna.items())) or "none")
    return {"written": len(lines), "per_antenna": per_antenna, "unknown": sorted(unknown), "empty": empty,
            "seconds": seconds, "too_long": too_long}
