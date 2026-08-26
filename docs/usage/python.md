# Interactive Python

The most useful mode while working out what a dataset needs. Every step returns
an object, so you can look at the result before deciding the next move.

```python
from vlbipy import VLBIObs

obs = VLBIObs("rsm07", network="EVN", work_dir="rsm07",
              target="3C395", phasecal="J1848+3219", fringe_finder="3C345")
obs.import_data()
print(obs.summary())
```

```
Observation rsm07 (EVN, backend=casa)
  observed: 2025-09-16 (2.8 h)
  antennas (14, [] = no data): JB, WB, EF, MC, O8, T6, [TR], HH, IR, CM, [DA], KN, PI, DE
  scans: 63; freq: 1.658 GHz, BW 128 MHz (4 x 64 ch)
  sources in data: 3C345, J1848+3219, 3C395
  refant: EF
```

## Inspecting before calibrating

```python
obs.metadata.max_baseline / 1e3      # 10157 km
obs.metadata.resolution_mas          # 3.67 mas
obs.antennas["EF"].subbands          # (0, 1, 2, 3)
obs.metadata.time_on_source("3C395") / 60
```

The per-scan fringe SNR survey shows what is actually detectable before any
calibration decision is made:

```python
survey = obs.calibrate.scan_snr()
survey.rank_antennas()      # [('EF', 4780), ('T6', 3445), ('HH', 3152), ...]
survey.rank_scans()[:3]     # best scans to solve on
survey.dead_antennas()      # nothing usable
obs.plot.scan_snr()         # the scan x antenna SNR matrix
```

## Calibrating step by step

```python
obs.calibrate.a_priori()            # Tsys + gain curve, de-spiked, plotted
obs.flag.apriori()                  # .uvflg + autocorrelations

antennas, scans = obs["rsm07"].calibrate.select_calibration_data()
obs.calibrate.instrumental()        # SBD -> bandpass -> SBD, on that selection
obs.calibrate.edge_channels()       # measure and flag the band edges
obs.calibrate.apply(force=True)
obs.calibrate.fringefit()           # global fringe fit
obs.calibrate.apply(force=True)
```

Each returns its `CalTable`, and `obs.gaintables` is the chain built so far:

```python
[t.cal_type for t in obs.gaintables]
# ['tsys', 'gc', 'bpass', 'sbd2', 'mbd']
```

## Looking at the data

```python
obs.plot.spectrum(field="3C345", scans=scans)   # amp+phase vs frequency
obs.plot.timeseries(field="J1848+3219")         # amp+phase vs time
obs.plot.corners()                              # per-baseline time x frequency
obs.plot.radplot()                              # amp+phase vs uv distance
obs.plot.caltables()                            # every solution table
```

You can also read the visibilities yourself, without going through a plot:

```python
backend = obs["rsm07"]._backend
spec = backend.data.read_spectrum("rsm07", field="3C345", column="corrected")
spec["spectra"]["JB"].shape          # (n_spw, n_chan, n_pol)
```

## Flagging and finishing

```python
obs.flag.outliers(dry_run=True)     # measure before committing
obs.flag.quack()                    # per-antenna slew time, measured
obs.calibrate.second_pass()         # re-solve on the clean data
obs.calibrate.scalar_bandpass()     # level the subbands
obs.calibrate.apply(force=True)
obs.export.per_source()             # per-source MS + UVFITS
```

## Discoverability

Namespaces list their own operations, and backends report what they actually
implement — so `NotImplementedError` is never a surprise:

```python
obs.calibrate                       # <CalibrateNamespace operations: a_priori, apply, ...>
obs["rsm07"]._backend.capabilities()
# {'data': [...], 'calibrate': [...], 'image': [], ...}
```

An empty list means that part is not built yet.
