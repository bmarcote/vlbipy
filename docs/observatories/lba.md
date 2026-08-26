# Long Baseline Array (LBA)

## Overview

The **Long Baseline Array (LBA)** is the Australian VLBI network, operated under the auspices of **CSIRO Astronomy and Space Science (CASS)**. It is not a dedicated VLBI instrument but rather a coordinated network of independently operated radio telescopes across Australia (and occasionally New Zealand) that are combined for VLBI observations during scheduled sessions.

The LBA provides baselines up to ~3,500 km within Australia and achieves angular resolutions comparable to other VLBI networks at its observing frequencies. It observes at frequencies from 1.4 GHz to 22 GHz, with some stations capable of higher frequencies.

## Stations

| Code | Station | Location | Diameter | Operator |
| --- | --- | --- | --- | --- |
| **AT** | ATCA (phased array) | Narrabri, NSW | 6×22 m | CSIRO |
| **PA** | Parkes (Murriyang) | Parkes, NSW | 64 m | CSIRO |
| **MP** | Mopra | Coonabarabran, NSW | 22 m | CSIRO |
| **HO** | Hobart | Hobart, Tasmania | 26 m | University of Tasmania |
| **CD** | Ceduna | Ceduna, SA | 30 m | University of Tasmania |
| **TI** | Tidbinbilla (DSS-43) | Canberra, ACT | 70 m | NASA/CDSCC |
| **WW** | Warkworth | Warkworth, NZ | 12 m | AUT University |

The **Australia Telescope Compact Array (ATCA)** can be used as a phased array, combining the signals of its six 22-metre antennas to act as a single, more sensitive element.

**Tidbinbilla** (the 70-metre NASA Deep Space Network antenna) is available on a best-effort basis and provides exceptional sensitivity when included.

## Correlator

LBA data is correlated using the **DiFX software correlator**[^1] operated at **Curtin University** in Perth, Western Australia. The correlator produces FITS-IDI output files.

[^1]: Deller, A. T., et al. (2011). "DiFX-2: A More Flexible, Efficient, Robust, and Powerful Software Correlator." *PASP*, 123, 275. [doi:10.1086/658907](https://doi.org/10.1086/658907)

## Data Format and Archive

LBA data is available through the **Australia Telescope Online Archive (ATOA)** at [https://atoa.atnf.csiro.au/](https://atoa.atnf.csiro.au/).

LBA data characteristics:

- **Multiple data formats**: FITS-IDI is the standard output, but legacy observations may use RPFITS or UVFITS format.
- **ANTAB handling**: Similar to the EVN, Tsys and gain curve data may be distributed as a separate ANTAB file that needs to be appended to the FITS-IDI headers before import.
- **Experimental download**: Automated download from ATOA is experimental in vlbipy. If it fails, data should be downloaded manually from the archive.

## vlbipy LBA Workflow

The `LbaHandler` class handles LBA-specific steps:

1. **Data download**: Attempts to retrieve data from ATOA. Falls back to searching for already-downloaded files.
2. **File discovery**: Searches for multiple formats: FITS-IDI, UVFITS, RPFITS, and FITS files matching the project code.
3. **ANTAB append**: If an ANTAB file is found, it can be appended to FITS-IDI headers (similar to EVN workflow).
4. **Flag file**: Looks for `<project>.flag`, `<project>.uvflg`, or `<project>.flags`.

### Reference antenna priority

The default reference antenna priority for LBA is:

```text
AT, PA, MP, HO, CD, TI
```

The ATCA phased array is preferred due to its high sensitivity. Parkes is the second choice for similar reasons.

## References

- LBA homepage: [https://www.atnf.csiro.au/vlbi/](https://www.atnf.csiro.au/vlbi/)
- Australia Telescope Online Archive (ATOA): [https://atoa.atnf.csiro.au/](https://atoa.atnf.csiro.au/)
- ATCA documentation: [https://www.narrabri.atnf.csiro.au/](https://www.narrabri.atnf.csiro.au/)
- Deller, A. T., et al. (2011). "DiFX-2: A More Flexible, Efficient, Robust, and Powerful Software Correlator." *PASP*, 123, 275. [doi:10.1086/658907](https://doi.org/10.1086/658907)
- Edwards, P. G., & Phillips, C. (2015). "The Long Baseline Array." Proceedings of the 12th Asian-Pacific Regional IAU Meeting. [arXiv:1501.04070](https://arxiv.org/abs/1501.04070)
