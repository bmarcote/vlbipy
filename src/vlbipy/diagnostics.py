"""Human-readable observation summaries (the ``summary.md`` report).

Renders the metadata read at import time as fixed-width text tables: sources,
frequency setup, antenna participation, and a scan-by-scan listing. The report is
written once per project by the import step and is the first thing to look at
when a run behaves unexpectedly, so everything here is derived from
:class:`~vlbipy.models.ObsMetadata` alone and never queries the data again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from astropy import coordinates as coord

from .logging_utils import get_logger
from .models import ObsMetadata
from .sources import SourceSet

logger = get_logger()

#: Placeholder for an antenna that did not take part in a scan (keeps its column).
MISSING_ANTENNA = "--"

#: Minimum ruler width, so short reports keep the familiar layout.
_RULER = 100


def scan_summary(metadata: ObsMetadata) -> str:
    """Render the scan-by-scan table: number, source, duration, and antennas.

    Every antenna that observed at least one scan gets a fixed column, so a
    station always appears in the same position on every row and a gap is read
    down the column rather than by comparing names. Antennas that missed a scan
    are shown as ``--`` instead of being skipped. Stations with no data at all
    are left out entirely (they would be an empty column); the antenna table
    above still lists them.

    Parameters
    ----------
    metadata : ObsMetadata
        Populated observation metadata.

    Returns
    -------
    str
        Multi-line table.
    """
    columns = sorted({ant for scan in metadata.scans for ant in scan.antennas})
    cell_width = max([len(name) for name in columns] + [len(MISSING_ANTENNA)])
    header = f"{'Scan':>5}  {'Source':<20}  {'Duration':>8}  {'#Ant':>5}  {'Antennas'}"
    rows = []
    for scan in sorted(metadata.scans, key=lambda s: s.scan_number):
        present = set(scan.antennas)
        cells = " ".join(f"{name if name in present else MISSING_ANTENNA:<{cell_width}}"
                         for name in columns)
        duration = f"{scan.duration_sec:.0f}s"
        rows.append(f"{scan.scan_number:>5}  {scan.source:<20}  {duration:>8}  "
                    f"{len(scan.antennas):>5}  {cells}".rstrip())
    ruler = max([_RULER] + [len(line) for line in [header] + rows])
    return "\n".join([f"Scan Summary for {metadata.project_code}", "=" * ruler,
                      header, "-" * ruler] + rows)


def antenna_summary(metadata: ObsMetadata) -> str:
    """Render antenna participation: subbands recorded and number of scans.

    Scan counts are recomputed from the scan list rather than read from
    ``Antenna.n_scans`` so this table and :func:`scan_summary` can never disagree.

    Parameters
    ----------
    metadata : ObsMetadata
        Populated observation metadata.

    Returns
    -------
    str
        Multi-line table, including antennas that recorded no data.
    """
    lines = [f"Antenna Summary for {metadata.project_code}", "=" * 60,
             f"{'Antenna':<10}  {'Subbands':<20}  {'Scans':>6}", "-" * 60]
    for name in sorted(metadata.antennas):
        antenna = metadata.antennas[name]
        n_scans = sum(1 for scan in metadata.scans if name in scan.antennas)
        subbands = "[" + ",".join(str(s) for s in antenna.subbands) + "]" if antenna.subbands else "[]"
        lines.append(f"{name:<10}  {subbands:<20}  {n_scans:>6}")
    return "\n".join(lines)


def frequency_summary(metadata: ObsMetadata) -> str:
    """Render the frequency setup: reference frequency, bandwidth, subbands, channels.

    Parameters
    ----------
    metadata : ObsMetadata
        Populated observation metadata.

    Returns
    -------
    str
        Multi-line block.
    """
    freq = metadata.freq_setup
    return "\n".join([
        f"Frequency Setup for {metadata.project_code}", "=" * 60,
        f"  Reference frequency:  {freq.freq_ghz:.4f} GHz",
        f"  Total bandwidth:      {freq.bandwidth_mhz:.1f} MHz",
        f"  Number of subbands:   {freq.n_subbands}",
        f"  Channels per subband: {freq.n_channels}",
        f"  Channel width:        {freq.channel_width / 1e3:.1f} kHz",
        f"  Polarizations:        {', '.join(p.name for p in freq.polarizations)}",
    ])


def source_summary(metadata: ObsMetadata, sources: Optional[SourceSet] = None) -> str:
    """Render the source table: field ID, name, role, coordinates, and scan count.

    Only sources with at least one scan are listed, matching what the data
    actually contains rather than what was scheduled.

    Parameters
    ----------
    metadata : ObsMetadata
        Populated observation metadata.
    sources : SourceSet, optional
        Configured sources, used to label each source with its role. Sources not
        declared there are reported as ``other``.

    Returns
    -------
    str
        Multi-line table.
    """
    lines = [f"Source Summary for {metadata.project_code}", "=" * 80,
             f"{'ID':>3}  {'Source':<20}  {'Type':<15}  {'RA':<18}  {'Dec':<18}  {'Scans':>5}",
             "-" * 80]
    for name in sorted(metadata.source_names):
        n_scans = sum(1 for scan in metadata.scans if scan.source == name)
        if n_scans == 0:
            continue
        radec = metadata.source_coords.get(name)
        if radec is not None:
            position = coord.SkyCoord(ra=radec[0], dec=radec[1], unit="deg")
            ra_str = position.ra.to_string(unit="hourangle", sep=":", precision=5, pad=True)
            dec_str = position.dec.to_string(sep=":", precision=5, pad=True, alwayssign=True)
        else:
            ra_str = dec_str = "N/A"
        source_type = sources[name].source_type.value if sources and name in sources else "other"
        source_id = metadata.source_ids.get(name, "?")
        lines.append(f"{source_id:>3}  {name:<20}  {source_type:<15}  "
                     f"{ra_str:<18}  {dec_str:<18}  {n_scans:>5}")
    return "\n".join(lines)


def full_summary(metadata: ObsMetadata, sources: Optional[SourceSet] = None) -> str:
    """Assemble the complete observation summary from all the sections above.

    Parameters
    ----------
    metadata : ObsMetadata
        Populated observation metadata.
    sources : SourceSet, optional
        Configured sources, used to label source roles.

    Returns
    -------
    str
        The full report.
    """
    start_mjd, end_mjd = (t / 86400.0 for t in metadata.time_range)
    return "\n".join([
        f"VLBI Observation Summary: {metadata.project_code}",
        f"Observing date: {metadata.obs_date}",
        f"Time range: MJD {start_mjd:.6f} - {end_mjd:.6f}",
        "",
        source_summary(metadata, sources),
        "",
        frequency_summary(metadata),
        "",
        antenna_summary(metadata),
        "",
        scan_summary(metadata),
    ])


def write_summary(metadata: ObsMetadata, output_file, sources: Optional[SourceSet] = None) -> Path:
    """Write the full summary to a file, creating parent directories as needed.

    Parameters
    ----------
    metadata : ObsMetadata
        Populated observation metadata.
    output_file : str or pathlib.Path
        Destination file (conventionally ``<work_dir>/summary.md``).
    sources : SourceSet, optional
        Configured sources, used to label source roles.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(full_summary(metadata, sources) + "\n")
    logger.info("wrote observation summary to {}", path)
    return path
