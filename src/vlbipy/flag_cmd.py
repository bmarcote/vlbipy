"""The correlator flags of a FITS-IDI file, as ``flagdata`` list-mode commands.

``importfitsidi`` (and the native import, :mod:`vlbipy.solvers.idi_subtables`) copies the FITS-IDI ``FLAG``
table into the ``FLAG_CMD`` subtable of the measurement set, one row per flag, with ``APPLIED = False``: the
visibilities themselves stay unflagged. VLBA data carry their a-priori flags (off-source, slewing, recorder
problems) only there, so they have to be applied after the import.

CASA's ``flagcmd(inpmode='table')`` cannot do it: it applies the ``COMMAND`` string of each row as it is,
and the filler writes the time of a flag only in the ``TIME`` / ``INTERVAL`` columns, never into the
command, so every flag would cover the whole observation. :func:`commands_from_flag_cmd` builds complete
commands instead:

* ``timerange`` from ``TIME - INTERVAL / 2`` to ``TIME + INTERVAL / 2`` (MJD seconds, UTC);
* antenna and field IDs replaced by their names, which both CASA and the native flagger
  (:mod:`vlbipy.solvers.flagdata`) accept;
* the stray closing quote casacore writes after a channel range (``spw='0:3~5''``) removed.
"""
from __future__ import annotations

import re
from typing import Iterable

from .logging_utils import get_logger
from .tools import mjdsec2datetime

logger = get_logger()

#: ``key='value'`` pairs of a FLAG_CMD command (the filler always quotes its values).
_PAIR = re.compile(r"(\w+)\s*=\s*'([^']*)'")


def format_time(mjd_seconds: float) -> str:
    """MJD seconds -> ``YYYY/MM/DD/hh:mm:ss.sss`` (CASA time selection, UTC)."""
    moment = mjdsec2datetime(mjd_seconds)
    return moment.strftime("%Y/%m/%d/%H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


def _antenna_names(text: str, names: list[str]) -> str:
    """``'3'`` / ``'3&5'`` (MS antenna IDs) -> ``'PT'`` / ``'PT&LA'``; raises ``ValueError`` on an unknown ID."""
    return "&".join(names[int(token)] for token in text.split("&"))


def command_from_row(time: float, interval: float, command: str, antenna_names: list[str],
                     field_names: list[str]) -> str:
    """One FLAG_CMD row -> one ``flagdata`` list-mode command with its time range and names.

    Parameters
    ----------
    time, interval : float
        ``TIME`` (centre, MJD seconds) and ``INTERVAL`` (seconds) of the row.
    command : str
        ``COMMAND`` of the row, as the FITS-IDI filler wrote it (IDs, no time range).
    antenna_names, field_names : list of str
        The ANTENNA and FIELD names of the measurement set, indexed by ID.

    Raises
    ------
    ValueError
        An antenna or field ID that the measurement set does not have.
    """
    text = str(command).replace("''", "'")
    parts = []
    for key, value in _PAIR.findall(text):
        if key == "antenna":
            value = _antenna_names(value, antenna_names)
        elif key == "field":
            value = ",".join(field_names[int(token)] for token in value.split(","))
        parts.append(f"{key}='{value}'")
    half = 0.5 * float(interval)
    parts.append(f"timerange='{format_time(time - half)}~{format_time(time + half)}'")
    return " ".join(parts)


def commands_from_flag_cmd(times: Iterable[float], intervals: Iterable[float], commands: Iterable[str],
                           antenna_names: list[str], field_names: list[str]) -> dict:
    """Every FLAG_CMD row -> ``flagdata`` list-mode commands (:func:`command_from_row`).

    Rows that cannot be converted (an ID the measurement set does not have) and rows without a positive
    interval are left out and counted, never applied over the whole observation.

    Returns
    -------
    dict
        ``commands`` (list of str), ``skipped`` (rows left out, with the reason) and ``per_antenna``
        (commands per antenna name; ``"all"`` for flags without an antenna).
    """
    out, skipped, per_antenna = [], [], {}
    for row, (time, interval, command) in enumerate(zip(times, intervals, commands)):
        if not float(interval) > 0.0:
            skipped.append(f"row {row}: interval {interval} s ({command!r})")
            continue
        try:
            line = command_from_row(time, interval, command, antenna_names, field_names)
        except (ValueError, IndexError) as exc:
            skipped.append(f"row {row}: {exc} ({command!r})")
            continue
        out.append(line)
        match = re.search(r"antenna='([^']*)'", line)
        for name in (match.group(1).split("&") if match else ["all"]):
            per_antenna[name] = per_antenna.get(name, 0) + 1
    return {"commands": out, "skipped": skipped, "per_antenna": per_antenna}
