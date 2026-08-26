# Very Long Baseline Array (VLBA)

## Overview

The **Very Long Baseline Array (VLBA)** is a dedicated VLBI instrument operated by the **National Radio Astronomy Observatory (NRAO)** in the United States. It consists of ten identical 25-metre radio telescopes distributed across the US from the Virgin Islands to Hawaii, providing baselines up to ~8,600 km.

The VLBA is unique among VLBI networks in that it is a **homogeneous, purpose-built array** — all antennas are identical in design and are operated centrally from the Array Operations Center in Socorro, New Mexico. This simplifies calibration compared to heterogeneous arrays like the EVN.

The VLBA observes at frequencies from 312 MHz (P-band) to 86 GHz (W-band) and is available for proposals year-round through the NRAO proposal system.

## Stations

| Code | Station | Location | Diameter |
| --- | --- | --- | --- |
| **BR** | Brewster | Washington | 25 m |
| **FD** | Fort Davis | Texas | 25 m |
| **HN** | Hancock | New Hampshire | 25 m |
| **KP** | Kitt Peak | Arizona | 25 m |
| **LA** | Los Alamos | New Mexico | 25 m |
| **MK** | Mauna Kea | Hawaii | 25 m |
| **NL** | North Liberty | Iowa | 25 m |
| **OV** | Owens Valley | California | 25 m |
| **PT** | Pie Town | New Mexico | 25 m |
| **SC** | Saint Croix | US Virgin Islands | 25 m |

## Correlator

VLBA data is correlated using the **DiFX software correlator**[^1] located at the NRAO Array Operations Center in Socorro, New Mexico. The DiFX correlator produces FITS-IDI output files with calibration metadata (Tsys, weather, gain curves) **already embedded** in the file headers.

[^1]: Deller, A. T., et al. (2011). "DiFX-2: A More Flexible, Efficient, Robust, and Powerful Software Correlator." *PASP*, 123, 275. [doi:10.1086/658907](https://doi.org/10.1086/658907)

## Data Format and Archive

VLBA data is available through the **NRAO Science Data Archive** at [https://data.nrao.edu/portal/](https://data.nrao.edu/portal/).

Key differences from EVN data:

- **Self-contained FITS-IDI files**: Tsys, weather, and gain curve information is embedded in the FITS-IDI headers by the correlator. No separate ANTAB file is needed.
- **No programmatic download**: The NRAO archive requires manual download through the web portal. vlbipy's `VlbaHandler.download_data()` raises `NotImplementedError` with instructions.
- **VLBA pipeline products**: The NRAO also provides pipeline-calibrated data products. These are useful for quick-look purposes but may not be optimal for all science cases.

## vlbipy VLBA Workflow

The `VlbaHandler` class handles VLBA-specific steps:

1. **Data location**: Searches the `uv/` directory for FITS-IDI files matching the project code (patterns: `<project>.idifits`, `VLBA_<project>*`, etc.).
2. **No preparation needed**: Since calibration metadata is embedded in the FITS-IDI files, `prepare_for_import()` is a pass-through.
3. **Flag file**: Looks for `<project>.flag` in the data directory.
4. **No ANTAB file**: `get_antab_file()` returns `None` since calibration data is in the FITS-IDI headers.

### Reference antenna priority

The default reference antenna priority for VLBA is:

```text
PT, LA, FD, KP, OV, BR, NL, MK, SC, HN
```

Pie Town is preferred due to its central location and consistent performance.

## References

- VLBA homepage: [https://science.nrao.edu/facilities/vlba](https://science.nrao.edu/facilities/vlba)
- VLBA Observational Status Summary: [https://science.nrao.edu/facilities/vlba/docs](https://science.nrao.edu/facilities/vlba/docs)
- NRAO Data Archive: [https://data.nrao.edu/portal/](https://data.nrao.edu/portal/)
- AIPS Cookbook (VLBA calibration): [http://www.aips.nrao.edu/cook.html](http://www.aips.nrao.edu/cook.html)
- Deller, A. T., et al. (2011). "DiFX-2: A More Flexible, Efficient, Robust, and Powerful Software Correlator." *PASP*, 123, 275. [doi:10.1086/658907](https://doi.org/10.1086/658907)
- Napier, P. J., et al. (1994). "The Very Long Baseline Array." *Proceedings of the IEEE*, 82(5), 658–672. [doi:10.1109/5.284733](https://doi.org/10.1109/5.284733)
