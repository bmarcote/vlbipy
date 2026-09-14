# Full pipeline

!!! warning "Partially implemented"

    The calibration chain runs end to end and produces calibrated per-source
    data. **Imaging and self-calibration are not implemented**, so `run()`
    stops after the export and logs that it skipped imaging.

One command takes a project from raw correlator output to calibrated,
Difmap-ready data:

```bash
vlbipy run -p rsm07 -t 3C395 --phasecal 'J1848+3219' --fringe-finder '3C345'
```

or from Python:

```python
from vlbipy import VLBIObs

obs = VLBIObs("rsm07", network="EVN",
              target="3C395", phasecal="J1848+3219", fringe_finder="3C345")
obs.run()
```

## What it does, in order

| Step | What happens |
|---|---|
| `import_data` | find or download FITS-IDI, `.antab`, `.uvflg`; import to a Multi-MS; read metadata |
| `flag.apriori` | apply the observatory `.uvflg`; flag autocorrelations |
| `a_priori` | `gencal` Tsys + gain curve (+ EOP for VLBA/LBA); de-spike Tsys |
| `plot.diagnostics("data")` | raw data: scan SNR, full-Stokes cross-correlations, spectra, time series, corners, radplot, uv coverage |
| `flag.quack` | trim the slewing time per antenna (`[flagging.quack_antennas]`, `quack_interval`, or measured) |
| `flag.initial` | tfcrop on the calibrators, per baseline, strong outliers only |
| `scan_snr` | short fringe fit per scan: SNR per scan, antenna and polarization |
| `instrumental` | select antennas and scan(s), then SBD → MBD → bandpass → SBD → MBD again |
| `flag.edges` | measure the subband roll-off from the bandpass and flag it |
| `apply` | apply the instrumental chain to every field |
| `flag.outliers` | per-baseline robust outlier flagging on calibrated data |
| `second_pass` | re-solve the whole chain on the now-clean data |
| `scalar_bandpass` | one amplitude per antenna and subband, levelling the subbands |
| `apply` | full calibration |
| `reweight` | `statwt` from the calibrated scatter; then `flag.outliers` and a third pass |
| `plot.diagnostics("corrected")` | the same plots on the calibrated data |
| `export.per_source` | split per source, export UVFITS |
| `flag.statistics` | flagged fraction per antenna / subband over observable data; goes into the report |

A representative run on a 2.8 h, 14-antenna EVN dataset (8.3 GB) takes about
25 minutes, dominated by the two fringe-fitting passes.

## Resuming

Progress is recorded in `.pipeline_state.json`, so a run resumes rather than
repeating work:

```bash
vlbipy run -p rsm07 ...                        # continue where it stopped
vlbipy run -p rsm07 ... --from-step fringefit  # redo from a step onward
vlbipy run -p rsm07 ... --scratch              # start over, reset the data
```

`--scratch` also runs `clearcal` and unflags the measurement set, so a fresh run
does not inherit a previous attempt's flags. Without it, resuming deliberately
*keeps* the flags and corrected data, because those are the run's own work in
progress. Existing flags are saved as a restorable flag version first.

## Campaigns

Give several project codes and steps fan out across them, with `merge()`
combining them after calibration:

```python
obs = VLBIObs(["ek048a", "ek048b", "ek048c"], network="EVN", target="3C84")
obs.run()
```

Multi-epoch combination beyond the merge is not implemented yet.
