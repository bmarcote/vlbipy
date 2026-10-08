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

`LBAObservatory` (`network="LBA"`) works from files on disk; nothing is downloaded.

1. **File discovery**: `<CODE>.FITS` (any case; also `.fitsidi` / `.idifits`), `<code>.antab`
   and `<code>.uvflg` are looked for in the working directory, in its `input_data/`, and in
   its parent — the usual layout of a multi-epoch project, with all raw files side by side
   and one working directory per epoch.
2. **ANTAB append** (in place, into the FITS-IDI file). DiFX writes no `SYSTEM_TEMPERATURE`
   table and an empty `GAIN_CURVE` one; the empty table is removed and both are written from
   the `.antab`. LBA `.antab` files are concatenations of per-station files, so the reader
   accepts what they contain: `INDEX` given as ranges (`'R1:4'`) or lists (`'R1|L1'`),
   one- or two-digit hours, stamps such as `07:60.00`, a single `DPFU`. A station whose
   `INDEX` covers only part of the band gets the level of the subbands it has for the rest
   (with a warning) instead of losing them. Stations absent from the `.antab` are reported:
   add nominal values for them and re-import.
3. **Flags**: the `.uvflg` (also a concatenation of dialects: header keywords without a
   terminating slash, commas between keywords, zero-padded day numbers) is converted to CASA
   flag commands in `<work_dir>/<code>.flag` by `vlbipy.uvflg`. Records for stations that are
   not in the data are dropped; zero-length ranges are skipped. A record longer than
   `LBAObservatory.max_flag_hours` (2 h) is not applied and is reported instead: these files flag
   slews, and a record of hours is an interval the station log never closed (in V589A one such
   record would have removed ATCA, the most sensitive antenna, for 12 of the 13 hours).
4. **Import and a-priori calibration**: `importfitsidi`, then ACCOR, Tsys, gain curve and the
   EOP correction (see below). The flags the
   import makes from the DiFX weights are saved as the `as_imported` flag version, which a
   `--scratch` run returns to.

DiFX labels the array `VLBA` in the data; that is expected and not reported.

### ACCOR and EOP (DiFX data)

Both apply to anything correlated with DiFX (LBA and VLBA), and both tables are applied with
`nearest` interpolation.

- **ACCOR** corrects the cross-correlation amplitudes for the digitiser statistics, measured
  on the autocorrelations: `accor(solint='30s')` writes `<code>.accor`, and
  `smoothcal(smoothtype='median', smoothtime=1800.0)` writes `<code>.accor_smooth`, the table
  in the chain. Because it needs the autocorrelations, `flag.apriori` leaves them unflagged and
  `calibrate.a_priori` flags them once ACCOR is done. Data without usable autocorrelations
  skip the step with a warning. `accor` runs with `corrdepflags=True`: DiFX gives the
  autocorrelation of a dead polarization zero weight, and without it the working
  polarization of that station would get no solution either (and be flagged with it).
  The table is made by the a-priori step, so the second and third calibration passes keep
  it like Tsys and EOP. Settings:

  ```toml
  [calibration.accor]
  enabled = true
  solint = "30s"
  smoothtype = "median"
  smoothtime = 1800.0      # seconds; 0 applies the table unsmoothed
  ```

- **EOP**: DiFX correlates with predicted Earth-orientation parameters. `usno_finals.erp` is
  fetched into the working directory the standard way,

  ```bash
  curl -u anonymous:<e-mail> --ftp-ssl \
      ftp://gdc.cddis.eosdis.nasa.gov/vlbi/gsfc/ancillary/solve_apriori/usno_finals.erp > usno_finals.erp
  ```

  (an HTTPS mirror is tried if CDDIS cannot be reached; `[calibration].eop_file` points at a
  file you already have), and `gencal(caltype='eop', infile='usno_finals.erp')` writes
  `<code>.eop`.

### Amplitude scale

Most LBA stations come with nominal values (`Tsys = 1.0` and a `DPFU` that stands for the
SEFD), and phased ATCA is recorded in its own units. After the a-priori calibration the
amplitudes are therefore off by factors of a few per antenna, not by the ~10% the
self-calibration assumes by default. Widen the prior of the Bayesian amplitude step for LBA
data, or it will correct only a fraction of the error:

```toml
[selfcal]
prior_sigma = 1.0      # width of the prior on the log-amplitude corrections (default 0.1)
```

The flux scale is then only as good as the average of the nominal values.

### Antennas with one polarization

Stations with a dead receiver channel, or that recorded one polarization for part of the
run, are common. The instrumental calibration works per polarization: such a station keeps
the hand it has (the scan selection may take a second scan to pick up the other one), and
only the correlations involving the missing hand are flagged.

Subbands need not be in frequency order (V589: 8393, 8425, 8457, 8489, 8409, ... MHz); the
multi-band delay uses the true frequencies.

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
