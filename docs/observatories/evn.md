# European VLBI Network (EVN)

## Overview

The **European VLBI Network (EVN)** is the world's most sensitive VLBI array. It is a distributed network of radio telescopes across Europe, Asia, and Africa that combines signals from up to ~20 stations to achieve angular resolutions of a few milliarcseconds at centimetre wavelengths. The EVN observes at frequencies from 1.4 GHz (L-band) to 43 GHz (Q-band), with some stations capable of millimetre-VLBI observations at 86 GHz.

The EVN operates in **session mode**, with three regular observing sessions per year (typically February/March, May/June, and October/November), plus ad-hoc e-VLBI runs using real-time data transfer over high-speed networks.

## Stations

The EVN comprises telescopes from multiple countries and institutions. Some of the primary stations include:

| Code | Station | Country | Diameter | Notes |
| --- | --- | --- | --- | --- |
| **EF** | Effelsberg | Germany | 100 m | MPIfR; one of the world's largest fully steerable dishes |
| **JB** | Jodrell Bank (Lovell) | UK | 76 m | University of Manchester |
| **WB** | Westerbork | Netherlands | 25 m (phased array) | ASTRON; single dish or phased array of 14×25 m |
| **MC** | Medicina | Italy | 32 m | INAF |
| **NT** | Noto | Italy | 32 m | INAF |
| **TR** | Toruń | Poland | 32 m | Nicolaus Copernicus University |
| **YS** | Yebes | Spain | 40 m | IGN; excellent high-frequency performance |
| **O8** | Onsala (20 m) | Sweden | 20 m | Chalmers University |
| **SR** | Sardinia | Italy | 64 m | INAF; one of the newest EVN stations |
| **UR** | Urumqi | China | 25 m | XAO, Chinese Academy of Sciences |
| **SH** | Shanghai (Sheshan) | China | 25 m | SHAO, Chinese Academy of Sciences |
| **HH** | Hartebeesthoek | South Africa | 26 m | HartRAO |
| **IR** | Irbene | Latvia | 32 m | VIRAC |
| **GB** | Green Bank | USA | 100 m | GBO; guest station for transatlantic baselines |

## Correlator

EVN data is correlated at the **Joint Institute for VLBI ERIC (JIVE)** in Dwingeloo, the Netherlands, using the **SFXC** software correlator[^1]. The correlator produces FITS-IDI output files.

[^1]: Keimpema, A., et al. (2015). "The SFXC software correlator for very long baseline interferometry: algorithms and implementation." *Experimental Astronomy*, 39, 259–279. [doi:10.1007/s10686-015-9446-1](https://doi.org/10.1007/s10686-015-9446-1)

## Data Format and Archive

EVN data products are distributed through the **JIVE archive** at [http://archive.jive.nl/](http://archive.jive.nl/). For each experiment, the archive provides:

- **FITS-IDI files**: The correlated visibility data, typically split into multiple files numbered `<project>_1_1.IDI1`, `<project>_1_1.IDI2`, etc.
- **ANTAB file** (`<project>.antab`): Contains system temperature (Tsys) measurements and gain curve polynomials for each antenna. This must be appended to the FITS-IDI headers before import.
- **Flag file** (`<project>.uvflg`): A-priori flags in AIPS UVFLG format, marking known bad data (antenna slewing, known RFI, equipment failures).

Protected experiments (those with proprietary periods) require username/password authentication.

## vlbipy EVN Workflow

The `EvnHandler` class in vlbipy handles all EVN-specific steps:

1. **Download**: Automatically fetches FITS-IDI, ANTAB, and uvflg files from the JIVE archive given a project code and observing date.
2. **ANTAB append**: Injects Tsys and gain curve tables into the FITS-IDI file headers using the `antab_editor` tool (or falls back to manual instructions).
3. **Flag conversion**: Converts AIPS-format `.uvflg` flags to CASA `flagdata` commands when using the CASA backend.

### Reference antenna priority

The default reference antenna priority for EVN is:

```text
EF, YS, O8, GB, JB, SR, MC, TR, WB, NT, UR
```

Effelsberg is preferred as the reference antenna due to its large collecting area and stable performance.

## References

- EVN homepage: [https://www.evlbi.org/](https://www.evlbi.org/)
- EVN Data Analysis Guide: [https://www.evlbi.org/evn-data-analysis-guide](https://www.evlbi.org/evn-data-analysis-guide)
- JIVE archive: [http://archive.jive.nl/](http://archive.jive.nl/)
- EVN Status Table (station capabilities): [https://www.evlbi.org/evn-status-table](https://www.evlbi.org/evn-status-table)
- Keimpema, A., et al. (2015). "The SFXC software correlator." *Exp. Astron.*, 39, 259. [doi:10.1007/s10686-015-9446-1](https://doi.org/10.1007/s10686-015-9446-1)
