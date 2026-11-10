---
name: reduce-vlbi-casa
description: CASA command reference for calibrating and imaging VLBI data (EVN, VLBA, LBA, EVN+e-MERLIN) - importfitsidi, a-priori gencal, fringefit, bandpass, applycal, statwt, split, tclean and gaincal self-calibration, step by step. Use when running or scripting VLBI reduction in CASA; the science behind each step is in /reduce-vlbi-data and the automated route in /reduce-vlbi-vlbipy.
---

# VLBI reduction in CASA - the commands

The **CASA commands** for each step of `/reduce-vlbi-data` (same step numbers, S0-S21). Load `/reduce-vlbi-data` for
why a step exists and how to judge it; `/reduce-vlbi-aips` for AIPS; `/reduce-vlbi-vlbipy` for the pipeline that wraps
these calls (read its code before writing new steps and match its conventions).

**Conventions**
- CASA >= 6.5 (6.7+ recommended). Modular CASA (`casatasks`, `casatools`) works for scripting.
- EVN a-priori helpers: `casavlbitools` (https://github.com/jive-vlbi/casa-vlbi); a copy ships inside vlbipy as
  `vlbipy.casavlbitools`.
- Experiment `exp`, passes `exp_N`. Tables in `caltables/`, plots in `plots/`, per-source MSes in `msfiles/`, images in
  `images/`.
- **Script structure (DARA/ERIS style):** number the steps in a dict and run `runsteps=[...]` so any block can be
  re-run. Always `rm -rf` a table before re-creating it.
- **Interactive cleaning on macOS:** the old viewer is gone; use `iclean(...)` (CASA >= 6.7.2, cubevis) or draw masks in
  CARTA and pass them to `tclean(mask=...)`. Fill the MODEL column with `ft(vis, model=..., usescratch=True)` if needed.

```python
import os, glob
exp          = 'exp'                    # experiment code (lower case)
msfile       = f'{exp}.ms'
refant       = 'EF'                     # or 'EF,O8,MC' as a priority list for fringefit
fringefinder = ['3C345']
phasecal     = ['PHASECAL']
target       = ['TARGET']
checksrc     = []
calibrators  = fringefinder + phasecal
for d in ('caltables', 'plots', 'msfiles', 'images', 'logs'):
    os.makedirs(d, exist_ok=True)
```

---

## S0 - Resources
```bash
df -h .
```

## S1 - Retrieve files
- **EVN:** `exp_*_1.IDI*` from https://archive.jive.eu, plus `exp.antab` and `exp.uvflg` from the experiment's
  pipeline area (`pipe/`). **Always use the archive `.antab`**, even when the FITS-IDI already carry Tsys/GC (since
  January 2023): the embedded tables had appending issues.
- **VLBA:** `<CODE>.idifits` (FITS-IDI) from https://data.nrao.edu, calibration attached. **LBA:** `EXP.FITS` plus
  per-antenna Tsys/flag files.

## S2 - Prepare the a-priori information (EVN/LBA)
Do this **before** `importfitsidi`. For EVN, replace any embedded Tsys/GC with the archive `.antab`:
```python
from casavlbitools.fitsidi import append_tsys, append_gc, convert_flags
idifiles = sorted(glob.glob(f'{exp}_*_1.IDI*'))
append_tsys(antabfile=f'{exp}.antab', idifiles=idifiles, replace=True)   # replace=True: overwrite embedded tables
append_gc(antabfile=f'{exp}.antab', idifile=idifiles[0], replace=True)
convert_flags(infile=f'{exp}.uvflg', idifiles=idifiles, outfile=f'{exp}.flag')   # AIPS UVFLG -> CASA flag list
```
Shell equivalents: `append_tsys.py exp.antab exp_1_1.IDI*`, `append_gc.py exp.antab exp_1_1.IDI1`,
`flag.py exp.uvflg exp_1_1.IDI* > exp.flag`. Spectral line: use each pass's ANTAB. External GC table:
`gc.py exp.antab caltables/exp.gc`.

## S3 - Import and inspect
```python
importfitsidi(fitsidifile=idifiles, vis=msfile, constobsid=True, scanreindexgap_s=15.0, specframe='GEO')
# VLBA: importfitsidi(fitsidifile=[f'{exp}.idifits'], vis=msfile, constobsid=True, scanreindexgap_s=15.0)
listobs(vis=msfile, listfile=f'logs/{exp}.listobs', overwrite=True)
plotants(vis=msfile, figfile=f'plots/{exp}_ants.png')
msmd.open(msfile)
ants = msmd.antennanames(); fields = msmd.fieldnames(); nspw = msmd.nspw()
nchan = msmd.nchan(0); scans = msmd.scannumbers(); msmd.close()
```
EVN/LBA FITS-IDI carry no dish diameters or axis offsets: not needed for calibration; fill the MS ANTENNA table
(`DISH_DIAMETER`, `OFFSET`) only for primary-beam work.

## S4 - A-priori flagging
```python
flagmanager(vis=msfile, mode='save', versionname='original')
flagdata(vis=msfile, mode='list', inpfile=f'{exp}.flag', reason='any', action='apply', flagbackup=True)
# Autocorrelation-based corrections first (S6: accor), then drop the autocorrelations
flagdata(vis=msfile, mode='manual', autocorr=True, flagbackup=False)
flagdata(vis=msfile, mode='manual', spw='*:0~3;28~31', flagbackup=False)          # edges (32 ch/spw example)
flagdata(vis=msfile, mode='quack', quackinterval=5.0, quackmode='beg', flagbackup=False)
flagdata(vis=msfile, mode='quack', antenna='WB', quackinterval=10., quackmode='beg')   # slow station
flagdata(vis=msfile, mode='elevation', lowerlimit=10.)
flagdata(vis=msfile, mode='manual', antenna='SV', correlation='RR')               # dead hand
flagdata(vis=msfile, mode='manual', antenna='CM,DA,DE,KN,PI', spw='0,3')          # subbands e-MERLIN never observed
flagmanager(vis=msfile, mode='save', versionname='apriori_flags')                 # keep: raw diagnostics use it
```
**VLBA FITS-IDI FLAG table:** the import puts it in the FLAG_CMD subtable, unapplied, with the time in
TIME/INTERVAL. `flagcmd(inpmode='table')` then flags each row for the **whole run**. Convert each row to a time-ranged
command (`timerange` = TIME +- INTERVAL/2, antenna/field IDs mapped to names) and apply with `flagdata(mode='list')`.

Keep manual flags in a text file (`mode='list'`), one command per line:
```
mode='manual' antenna='HH' scan='62~65' reason='station_log'
mode='manual' antenna='SH' spw='2' timerange='14:10:00~14:25:00'
```

## S5 - Model corrections
**EOP [DiFX only]:** `gencal(vis=msfile, caltable=f'caltables/{exp}.eop', caltype='eop', infile='usno_finals.erp')`.
**Ionosphere (<~8 GHz):** CASA's `tec_maps.create()` + `gencal(caltype='tecim')` have bugs for VLBI: don't rely on
them; do the TEC correction in vlbipy (or AIPS), or fit the dispersive delay in `fringefit` (S10).
**Parallactic angle:** `parang=True` in every `fringefit`/`gaincal`/`bandpass`/`applycal` on the raw multi-source
MS; `parang=False` after splitting CORRECTED data.

## S6 - A-priori amplitude calibration
```python
gencal(vis=msfile, caltable=f'caltables/{exp}.tsys', caltype='tsys', uniform=False)
gencal(vis=msfile, caltable=f'caltables/{exp}.gcal', caltype='gc')
plotms(vis=f'caltables/{exp}.tsys', xaxis='time', yaxis='tsys', coloraxis='corr',
       iteraxis='antenna', gridrows=3, gridcols=3, plotfile=f'plots/{exp}_tsys.png')
apriori    = [f'caltables/{exp}.gcal', f'caltables/{exp}.tsys']
apr_interp = ['nearest', 'nearest,nearest']
```
- The `gc` table holds the **power** gain `DPFU x POLY` as a polynomial in **zenith angle** (degrees), 8 coefficients
  per antenna/spw/pol; never evaluate it against elevation. Longer ANTAB polynomials must be refitted over the observed
  elevations; check the curve stays positive over the elevations each antenna observed.
- Compare the Tsys with the raw amplitudes before trusting it (`/reduce-vlbi-data` S6).
- Missing GC for a station -> `gencal` fails for it: provide an external GC table or treat its amplitudes as uncertain.
- **Sampler bias [DiFX: VLBA/LBA]:** `accor` (with `corrdepflags=True`) before removing the autocorrelations.
- Opacity (>~22 GHz): `gencal(caltype='opac', parameter=[tau per spw])`; use Tsys* from ANTAB where available.

## S7 - Diagnostic plots on raw data
```python
ff_scan_tr = '13:53:20~13:54:20'        # fringe-finder scan with all antennas
plotms(vis=msfile, xaxis='frequency', yaxis='phase', ydatacolumn='data', antenna=f'{refant}&*',
       correlation='RR,LL', timerange=ff_scan_tr, averagedata=True, avgtime='120',
       coloraxis='baseline', iteraxis='baseline')
plotms(vis=msfile, xaxis='channel', yaxis='amp', field=fringefinder[0], avgtime='3600',
       antenna=f'{refant}&*', correlation='RR,LL', iteraxis='spw', coloraxis='antenna2')
plotms(vis=msfile, xaxis='time', yaxis='amp', field=','.join(calibrators), spw='*:5~26',
       avgchannel='32', antenna=f'{refant}&*', coloraxis='baseline')
plotms(vis=msfile, xaxis='uwave', yaxis='vwave', field=target[0], coloraxis='spw')
```
For large MSes, `shadems`, `jplotter` or vlbipy plotting are much faster than `plotms`.

## S8 - Initial flagging (per baseline)
```python
flagdata(vis=msfile, mode='tfcrop', field=','.join(calibrators), datacolumn='data',
         timecutoff=4.0, freqcutoff=3.0, extendflags=False, flagbackup=True, action='apply')
# preview first with action='calculate', display='both' on a subset
```
`tfcrop`/`rflag` work per baseline (essential for VLBI). AOFlagger with a VLBI/e-MERLIN strategy is an alternative.

## S9 - Instrumental delays (sbd, "manual phase-cal")
```python
central = '5~26'                          # central channels of each spw
fringefit(vis=msfile, caltable=f'caltables/{exp}.sbd', field=fringefinder[0], timerange=ff_scan_tr,
          solint='inf', zerorates=True, refant=refant, minsnr=10, corrdepflags=True,
          spw=f'*:{central}', gaintable=apriori, interp=apr_interp, parang=True)
```
- Second scan needed: another `fringefit` on that timerange with `antenna` = refant + missing stations,
  `append=True`, same refant.
- Choose the scan on a strong source first, coverage second. Check: apply to the calibrators and plot CORRECTED phase
  vs frequency on that scan and on a **different** one.
```python
applycal(vis=msfile, field=','.join(calibrators), gaintable=apriori+[f'caltables/{exp}.sbd'],
         interp=apr_interp+['nearest'], parang=True, calwt=False, flagbackup=False)
```

## S10 - Global fringe fit (mbd: delay, rate, phase)
```python
tb.open(f'caltables/{exp}.sbd'); spws = tb.getcol('SPECTRAL_WINDOW_ID'); tb.close()
SPW_MBD = int(sorted(set(spws))[0])       # combine='spw' solutions live on the first spw
fringefit(vis=msfile, caltable=f'caltables/{exp}.mbd', field=','.join(calibrators),
          solint='inf', combine='spw', zerorates=False, refant=refant, minsnr=7,
          corrdepflags=True, spw=f'*:{central}',
          gaintable=apriori+[f'caltables/{exp}.sbd'], gainfield=['', '', fringefinder[0]],
          interp=apr_interp+['nearest'], parang=True)
```
- Below ~8 GHz: `paramactive=[True, True, True]` (delay, rate, dispersive); plot `yaxis='disp delay'`.
- Weak calibrators: `delaywindow=[-100,100]` (ns), `ratewindow=[-0.05,0.05]` (Hz).
- A table with every solution flagged must not enter the apply chain: check `FLAG` before using it.
- **Smoothing:** don't use `smoothcal` on fringe-fit tables (it targets G/T tables, its rate smoothing is experimental
  and it unflags solutions). Smooth delays and rates yourself (per antenna, spw, pol **and field**, flagged solutions
  excluded, phases never smoothed) or with vlbipy, writing a new table.
- Resolved phase calibrator (long baselines undetected): solve the short-baseline antennas in a separate `fringefit`
  with one of them as `refant` and `antenna='A,B,C&'`, and flag the inter-sub-array baselines in the split data.
```python
for y in ('delay', 'rate', 'phase'):
    plotms(vis=f'caltables/{exp}.mbd', xaxis='time', yaxis=y, coloraxis='corr', iteraxis='antenna',
           gridrows=2, gridcols=3, plotfile=f'plots/{exp}_mbd_{y}.png')
```
Compare the logger's solution counts with the expected ones (antennas x pols x intervals).

## S11 - Bandpass
```python
bandpass(vis=msfile, caltable=f'caltables/{exp}.bp', field=','.join(fringefinder), solint='inf',
         combine='scan', refant=refant, bandtype='B', solnorm=True, fillgaps=8, corrdepflags=True, minsnr=3,
         gaintable=apriori+[f'caltables/{exp}.sbd', f'caltables/{exp}.mbd'],
         gainfield=['', '', ','.join(fringefinder), ','.join(calibrators)],
         interp=apr_interp+['nearest', 'linear'], spwmap=[[], [], [], nspw*[SPW_MBD]], parang=True)
plotms(vis=f'caltables/{exp}.bp', xaxis='freq', yaxis='amp', coloraxis='corr', iteraxis='antenna',
       gridrows=3, gridcols=3)
```
- A weak fringe finder can be supplemented with phase-calibrator scans (`combine='scan,field'`).
- Flag the edge channels the bandpass reveals (`flagdata spw='*:0~N;M~last'`) and re-run `bandpass`.

## S12 - Re-flag, re-solve, scalar bandpass
Apply S9-S11, inspect CORRECTED data of the calibrators, flag outliers per baseline (grow each in time and
frequency, attribute it to the right antenna, flag the adjacent target scans when a phase-calibrator scan goes), then
**re-run sbd -> mbd -> bandpass from scratch**. Scalar bandpass: one amplitude per antenna and spw from median
calibrated amplitudes, **the same value for both hands**:
```python
gencal(vis=msfile, caltable=f'caltables/{exp}.scalar', caltype='amp', antenna='EF,WB', spw='0,1',
       parameter=[1.02, 0.97, 1.05, 0.99])   # one per (antenna, spw) as listed; same for R and L
```

## S13 - Apply calibration
```python
caltabs = apriori + [f'caltables/{exp}.sbd', f'caltables/{exp}.mbd', f'caltables/{exp}.bp']
for cal in calibrators:
    applycal(vis=msfile, field=cal, gaintable=caltabs, gainfield=['', '', fringefinder[0], cal, ''],
             interp=apr_interp+['nearest', 'linear', 'nearest,nearest'], spwmap=[[], [], [], nspw*[SPW_MBD], []],
             parang=True, calwt=False, applymode='calflag', flagbackup=False)
for tgt, pcal in zip(target+checksrc, phasecal*len(target+checksrc)):
    applycal(vis=msfile, field=tgt, gaintable=caltabs, gainfield=['', '', fringefinder[0], pcal, ''],
             interp=apr_interp+['nearest', 'linear', 'nearest,nearest'], spwmap=[[], [], [], nspw*[SPW_MBD], []],
             parang=True, calwt=False, applymode='calflag', flagbackup=True)
```
- `applymode`: `'calonly'` keeps data with failed solutions (tests); `'calflag'` flags them; `'calflagstrict'` also
  flags data with no calibration at all (drops stations that never fringed).
- Apply per field (one `applycal` per field and its `gainfield`), not a comma-separated field list sharing one
  gainfield.
- A cal library (`docallib=True, callib='caltables/callib.txt'`) keeps field/interp/spwmap per table in one place.
- MultiView / 2-D interpolation between calibrators is not native in CASA (write a combined table, or vlbipy).

## S14 - Re-weight
```python
statwt(vis=msfile, datacolumn='corrected', timebin='60s', chanbin='spw')
plotms(vis=msfile, xaxis='uvwave', yaxis='wt', field=target[0], coloraxis='baseline')
```
`statwt` estimates the scatter across channels and time in each bin; adjacent SFXC channels are correlated, so the
absolute sigma is underestimated (relative weights between baselines are what matter). Flag weight outliers; redo
S9-S13 if that changed a lot.

## S15 - Replot calibrated data
Repeat S7 with `ydatacolumn='corrected'`, plus amp vs `uvwave` per source (`coloraxis='antenna1'` shows station scale
errors).

## S16 - Split per source (and average)
```python
for src in target + checksrc + phasecal + fringefinder:
    mstransform(vis=msfile, field=src, outputvis=f'msfiles/{exp}_{src}.ms', datacolumn='corrected',
                correlation='RR,LL', keepflags=False, chanaverage=True, chanbin=nchan_avg,
                timeaverage=True, timebin='4s', usewtspectrum=True)
```
Check the smearing limits (`/reduce-vlbi-data` S16) before averaging wide fields.

## S17 - Image and self-calibrate the calibrators
```python
import numpy as np
ms.open(f'msfiles/{exp}_{src}.ms'); d = ms.getdata(['u','v','axis_info']); ms.close()
freq = d['axis_info']['freq_axis']['chan_freq'].max()
uvmax_lambda = np.hypot(d['u'], d['v']).max() * freq / 299792458.
beam_mas = np.degrees(1/uvmax_lambda)*3.6e6
cell = f'{beam_mas/10:.3f}mas'; imsize = 4096

def clean(vis, name, cell, imsize=1024, weighting='natural', robust=0.5, niter=1500, deconvolver='hogbom', mask=''):
    for ext in ('image','model','residual','psf','pb','sumwt','mask'):
        os.system(f'rm -rf images/{name}.{ext}')
    tclean(vis=vis, imagename=f'images/{name}', specmode='mfs', stokes='pseudoI', imsize=[imsize, imsize],
           cell=cell, weighting=weighting, robust=robust, deconvolver=deconvolver, niter=niter, cycleniter=300,
           threshold=0, psfcutoff=0.5, mask=mask, savemodel='modelcolumn')

def snr(name, box_src, box_noise):
    peak = imstat(f'images/{name}.image', box=box_src)['max'][0]
    rms  = imstat(f'images/{name}.image', box=box_noise)['rms'][0]
    print(f'{name}: peak {peak*1e3:.2f} mJy/b  rms {rms*1e3:.3f} mJy/b  S/N {peak/rms:.0f}')
```
- `stokes='pseudoI'` keeps single-polarisation antennas; `mtmfs, nterms=2` when BW/nu >~15%; multiscale for
  extended emission.
- Prefer a visibility model fit (`uvmodelfit`, or difmapy) for compact sources: one circular Gaussian first.
- Compare model and data before each self-cal (`plotms ... ydatacolumn='corrected'` vs `'model'`).

**Self-cal loop** (phase 30 -> 15 -> 5 min -> scan length; amplitude: one solution for the whole run first, then per
spw and longer-to-shorter intervals only while consistent):
```python
vis = f'msfiles/{exp}_{src}.ms'
gaincal(vis=vis, caltable=f'caltables/{src}.sc_p1', calmode='p', solint='30min', refant='EF', minblperant=3,
        minsnr=5, parang=False)
applycal(vis=vis, gaintable=[f'caltables/{src}.sc_p1'], calwt=False, parang=False)
# ... re-image, repeat with shorter solint into sc_p2 ...
gaincal(vis=vis, caltable=f'caltables/{src}.sc_ap', calmode='ap', solint='inf', combine='scan', solnorm=True,
        refant='EF', minblperant=3, minsnr=5, gaintable=[f'caltables/{src}.sc_p1', f'caltables/{src}.sc_p2'],
        interp=['linear', 'linear'])
```
- Time-dependent amplitude solves: don't force the mean gain of each spw to 1 when one station needs a large
  correction (it drags the others); flag corrections more than ~x5 from a station's median.
- The calibrator's constant amplitude table goes to every field; its phase and time-dependent tables stay with it
  unless the transfer is verified on a check source.
- Exclude a bad antenna from a solve with `antenna='!KN'`, then re-include it.
- **Export gain factors** (JSON, and to transfer as `gencal(caltype='amp')`):
```python
tb.open(f'caltables/{src}.sc_ap'); g = tb.getcol('CPARAM'); a = tb.getcol('ANTENNA1')
s = tb.getcol('SPECTRAL_WINDOW_ID'); fl = tb.getcol('FLAG'); tb.close()
# median |g| per (antenna, spw, pol) over unflagged solutions -> json
```

**e-MERLIN flux scale with 3C286 (EVN + e-MERLIN).** Only when e-MERLIN outstations are present (Jb alone does not
count) and 3C286 was observed. Jb is solved **with** the outstations:
```python
emerlin = 'JB,CM,DA,DE,KN,PI'                       # as named in the data
model = '3C286_L.clean.model.tt0'                   # e-MERLIN model (eMCP calibrator_models, or vlbipy's
                                                    # vlbipy/data/emerlin_calibrator_models)
setjy(vis=msfile, field='3C286', standard='Perley-Butler 2017', model=model, scalebychan=True, usescratch=True)
prior = caltabs                                     # the chain so far (with its gainfield/interp/spwmap)
gaincal(vis=msfile, caltable='caltables/emgain_phase', field='3C286', antenna=f'{emerlin}&', refant='JB',
        calmode='p', solint='30s', minsnr=3, parang=True, gaintable=prior)
gaincal(vis=msfile, caltable='caltables/emgain', field='3C286', antenna=f'{emerlin}&', refant='JB',
        calmode='a', solint='inf', combine='scan', minsnr=3, parang=True,
        gaintable=prior+['caltables/emgain_phase'])
```
- One amplitude per antenna, spw and hand over all 3C286 scans; spws without e-MERLIN data take the nearest observed
  spw (`spwmap`); every non-e-MERLIN antenna gets unit gain so the table can sit in the apply chain.
- After applying, the calibrated 3C286 amplitudes on e-MERLIN baselines should match the model per spw and hand;
  check the model orientation (a FITS model read with |CDELT| is mirrored in RA). e-MERLIN resolves out a few per cent
  of 3C286: eMCP multiplies the `fluxscale` results by `eMfactor = 1 / (1 + 0.04 (f/5 GHz x L/11.237 km)^2)` (L = the
  shortest projected baseline to 3C286, Lovell excluded).
- In the full-array self-cal, hold the e-MERLIN **amplitudes** but still solve their **phases**.

## S18 - Target
```python
applycal(vis=f'msfiles/{exp}_{tgt}.ms',
         gaintable=[f'caltables/{pcal}.sc_p1', f'caltables/{pcal}.sc_p2', f'caltables/{pcal}.sc_ap'],
         gainfield=[pcal, pcal, pcal], interp=['linear','linear','linear'], calwt=False)
```
(Works on a pair MS or matching antenna/spw indexing; otherwise apply to the multi-source MS with `field=tgt,
gainfield=pcal` and re-split.)
- Search: `tclean(..., niter=0, imsize=[8192,8192], weighting='natural')`; off-centre detection ->
  `phaseshift(vis, outputvis, phasecenter='J2000 ...')` and re-image.
- Position (`imfit`, or `uvmodelfit`) **before** any target self-cal. Image set: natural, robust 0 and 0.5, uniform.

## S19 - Check sources
As S18; compare the position with the catalogue; self-cal if S/N > ~20.

## S20 - Products
```python
exportfits(imagename=f'images/{name}.image', fitsimage=f'images/{name}.fits', dropdeg=True)
imfit(imagename=f'images/{name}.image', box=box_src, logfile=f'logs/{name}_imfit.log')
flagdata(vis=msfile, mode='summary', action='calculate')     # per antenna/spw; exclude spws never observed
```

## Special modes (commands)
- **Spectral line:** calibrate the continuum pass, apply to the line pass (`spwmap` if needed);
  `mstransform(regridms=True, outframe='LSRK', restfreq='22.23508GHz', mode='velocity', veltype='radio')`; fringe-fit a
  maser channel (`zerorates=False, solint='60s'`), self-cal on it, transfer to all channels; `uvcontsub`;
  `tclean(specmode='cube', restfreq=..., outframe='LSRK')`.
- **Polarimetry:** `parang=True` on the raw MS; cross-hand delay `gaincal(gaintype='KCROSS', combine='scan',
  field=polcal)`; D-terms with `polsolve` on a split MS with a full-Stokes model; EVPA from a known-EVPA source.
- **Primary beam (wide field):** per-antenna beam corrections per field (`gencal(caltype='amp')` or vlbipy).

## CASA pitfalls
- `spwmap` needs one entry per `gaintable` (`[]` for none) and `nspw` entries for the mapped table.
- `interp` for Tsys: `'nearest,nearest'`; for bandpass `'nearest,nearest'` or `'linear,linear'`.
- Don't reuse a table name without `rm -rf`.
- `parang=True` only on the raw MS.
- `refant` in `fringefit` can be a priority list; in `gaincal` add `refantmode='strict'` to fix the phase origin.
- `corrdepflags=True` keeps single-polarisation stations.
- `smoothcal` unflags solutions and is not for fringe-fit tables.
- Large MS: `mstransform(createmms=True)` (`mpicasa`) or dask-ms in vlbipy.

## References
ERIS 2026 (https://eris2026.ira.inaf.it/), DARA EVN continuum Parts 1-3
(https://www.jb.man.ac.uk/DARA/unit4/Workshops/EVN_continuum.html), EVN data reduction and imaging tutorial (CASA,
https://evlbi.org/evn-data-reduction-and-imaging-tutorial), casa-vlbi (https://github.com/jive-vlbi/casa-vlbi), CASA docs
(https://casadocs.readthedocs.io), eMCP (https://github.com/e-merlin/eMERLIN_CASA_pipeline), e-MERLIN calibration
tutorial (https://www.jb.man.ac.uk/DARA/ERIS22/calibration.html).
