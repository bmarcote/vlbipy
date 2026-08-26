# Observatories

vlbipy supports three major VLBI networks, each with dedicated observatory handler classes that manage data download, format-specific preparation, and auxiliary file handling.

The observatory is selected at project creation time via the `observatory` parameter (or `-n` CLI flag) and determines:

- **Data download source** and authentication requirements
- **Pre-import preparation** (e.g. ANTAB appending for EVN)
- **A-priori flag file** format and location
- **Reference antenna priority list** for automatic refant selection

## Supported Observatories

| Observatory | Handler Class | Archive | Correlator |
| --- | --- | --- | --- |
| [EVN](evn.md) | `EvnHandler` | [JIVE Archive](http://archive.jive.nl/) | SFXC at JIVE, Dwingeloo |
| [VLBA](vlba.md) | `VlbaHandler` | [NRAO Archive](https://data.nrao.edu/portal/) | DiFX at Socorro |
| [LBA](lba.md) | `LbaHandler` | [ATOA](https://atoa.atnf.csiro.au/) | DiFX at Curtin University |

## Observatory Handler Interface

All observatory handlers implement the `ObservatoryHandler` abstract base class, which defines:

- `download_data()` — Retrieve raw data files from the observatory archive
- `find_data_files()` — Locate existing data files in a local directory
- `prepare_for_import()` — Perform observatory-specific pre-import steps
- `get_flag_file()` — Find the a-priori flag file
- `get_antab_file()` — Find the ANTAB calibration metadata file

See the [API reference](../api/observatories.md) for full method signatures.
