# Arrays, correlators, files and station notes

## SFXC (EVN and global, correlated at JIVE) vs DiFX (VLBA, LBA)

| Item | SFXC | DiFX |
|---|---|---|
| EOPs at correlation | final/measured; real-time e-EVN runs may need a correction, only when the EOPs used are in the data (since January 2026) | predicted -> **post-correlation EOP correction** for phase referencing/astrometry |
| 2-bit sampler bias | corrected at the correlator | not corrected -> ACCOR-type correction (VLBA, LBA) |
| A-priori Tsys/GC | `.antab` from the EVN archive; embedded in the FITS-IDI since January 2023, but **use the archive `.antab`** (the embedded tables had appending issues) | VLBA: attached to the FITS-IDI; LBA: per-antenna files combined into an ANTAB |
| A-priori flags | `.uvflg` (AIPS UVFLG format) from the archive | VLBA: FITS-IDI FLAG table; LBA: per-antenna flag files combined |
| Pulse-cal | generally unused -> manual phase-cal on a fringe-finder scan | VLBA: PC table sometimes usable; manual phase-cal as fallback |
| Pipeline products | EVN pipeline (AIPS): TASAV with CL/FG/BP tables, plots, quick-look images | VLBARUN diagnostic run |

## Files and archives

- **EVN:** `exp_N_1.IDI*` beside the experiment in the JIVE archive; `.antab` and `.uvflg` sit in the pipeline area
  (`pipe/`), so a dataset copied by hand usually arrives without them - fetch them, or the a-priori flags silently never
  get applied. A global experiment's `.uvflg` can contain sections written by other networks in their own dialect.
- **VLBA:** FITS-IDI (`.idifits`; the archive name is `VLBA_<CODE>_..._<date>.idifits`), calibration attached. Refuse a
  file shorter than its headers announce (download in progress: FLAG, SYSTEM_TEMPERATURE and GAIN_CURVE come after the
  UV data). **The FITS-IDI FLAG table is not applied by the import**: in CASA it lands in FLAG_CMD (not applied) with
  the time in TIME/INTERVAL, so applying that subtable as it is flags each row for the whole run. Convert each row to a
  time-ranged flag command (TIME +- INTERVAL/2, antenna and field names) and apply that.
- **LBA:** `EXP.FITS` plus per-antenna Tsys and flag files; stations without Tsys need nominal values.
- Dish diameters and axis offsets are absent from EVN/LBA FITS-IDI; only primary-beam work needs them.

## Naming of passes

`exp` is the experiment; SFXC passes are `exp_N` (continuum/line, phase centres, pulsar bins; see `special-modes.md`).

## Station field notes (dated; check that they still hold)

- **DPFU = 1, POLY = 1 stations** (e.g. Tianma, Urumqi, the e-MERLIN outstations in EVN runs): no real gain curve; the
  Tsys carries the scale. With a constant Tsys as well (e.g. Jb in some runs) the a-priori calibration is a constant
  SEFD with no time dependence (2026).
- **e-MERLIN outstations (Cm, Da, De, Kn, Pi):** unreliable a-priori Tsys, large subband-to-subband amplitude
  differences, often only part of the EVN band; set their scale on 3C286 (`amplitude-calibration.md`). Their
  baselines among themselves are short: they can detect a scatter-broadened calibrator the long baselines cannot (2026).
- **e-MERLIN and Westerbork:** slewing at scan start often not covered by the a-priori flags (quack).
- **Westerbork:** polarisation leakage can be tens of per cent.
- **Large dishes (EF, FAST, the GBT):** narrow primary beams (wide-field work) and pointing losses at
  high frequency.
- **A station noise-like on a faint phase calibrator but detected at S/N > 1000 on the fringe finders** happened to
  several antennas in one L-band run (2026): keep the fringe-finder solutions.
- **Ionosphere sign:** a TEC table must reduce the residual dispersive delay found by the fringe fit; verify the sign on
  your data before trusting a new implementation.
