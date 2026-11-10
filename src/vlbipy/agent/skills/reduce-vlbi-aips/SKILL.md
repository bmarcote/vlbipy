---
name: reduce-vlbi-aips
description: AIPS task recipes (and ParselTongue equivalents) for calibrating and imaging VLBI data, especially EVN and EVN+e-MERLIN - FITLD, TASAV/TACOP pipeline tables, UVFLG, CLCOR, VLBATECR, SETJY/CALIB/GETJY, FRING, SNSMO, CLCAL, BPASS, SPLIT, FITTP, IMAGR, following a real EVN+e-MERLIN reduction. Use when working in AIPS or reading AIPS logs; the science is in /reduce-vlbi-data.
---

# VLBI reduction in AIPS - the recipe

The **AIPS commands** for VLBI calibration. It follows a real EVN + e-MERLIN L-band phase-referencing reduction and
fills in the steps that log skipped, from the AIPS Cookbook Ch. 9. The science and the checks behind each step are in
`/reduce-vlbi-data` (step numbers in brackets, e.g. **[S11]**). For CASA see `/reduce-vlbi-casa`; for the automated
route `/reduce-vlbi-vlbipy`.

Placeholders: experiment `EXP`; sources `FRINGEF` (fringe finder), `PHASECAL`, `TARGET`, `CHECK`, and `3C286`.
Antenna numbers in the examples: 1 Jb, 3 Ef, 4 Mc, 6 O8, 10-14 Cm Da De Kn Pi (check yours with `prtan`).

---

## 1. AIPS essentials

- Start: `aips tv=local` (or `aips notv`). Define data areas first: `export PWD=$(pwd)`, then
  `datain 'PWD:file.IDI'` (a leading quote with no closing quote keeps lower case).
- **Verbs:** `default <task>`, `tget <task>`, `inp`, `go <task>`, `wait` (chain as `wait; go clcor`), `tput`,
  `explain <task>` / `help <task>`.
- **Catalogue:** `pcat`/`ucat`/`mcat`; `getn n`, `get2n n` (second input, e.g. a model image), `geton n`; `imh`
  (header + table versions); `zap`; `clr2name`.
- **Tables are the calibration:** CL (applied with `docal 1; gainuse n`), SN (solutions), FG (flags; `flagver`), BP
  (`doband 1; bpver n`), TY/GC (Tsys/gain curve), NX, PC (pulse-cal), CC (clean components). **Never delete CL1.**
  Delete wrong tables with `extdest` (`inext 'SN'; invers n`), then reset `inext ''; invers 0`.
- **Snapshots:** `tasav` copies all tables into a data-less file; `tacop` copies tables between files.
- **Procedures:** `run vlbautil` loads `VLBATECR`, `VLBAPANG`, `VLBAFIX`, `VLBAMPCL`, `VLBAFRGP`, `VLBAAMP`,
  `VLBACCOR`, ...; `run vlbarun` loads the VLBA pipeline.
- **History:** `hinote 'text'`, `prthi`. Keep a run log with a header: experiment, date, band, antenna numbers, sources
  and roles, known problems.
- **Scripting:** ParselTongue (`from AIPSTask import AIPSTask; t = AIPSTask('fring'); t.indata = uvdata;
  t.aparm[1:] = [...]; t()`), as used by the EVN pipeline. Array adverbs are **1-indexed** (`t.aparm[1]` is APARM(1)).

### Table-version plan (write it at the top of the log)

| Table | Contents | Made by |
|---|---|---|
| CL1 | correlator model, unit gains | FITLD/INDXR |
| CL2 | + a-priori amplitude (Tsys x GC) + parallactic angle | EVN pipeline (TACOP from TASAV) or ANTAB->APCAL->CLCAL + CLCOR 'PANG' |
| CL3 | + ionosphere (TEC maps) [+ known gain factors] | VLBATECR / TECOR [+ CLCOR 'GAIN'] |
| CL4 | + e-MERLIN amplitude scale from 3C286 (CALP: other antennas untouched) | CALIB->SN2->CLCAL |
| CL5 | + instrumental delays/phases (manual phase-cal) | FRING short scan->SN3->CLCAL |
| CL6 | + global fringe fit; targets from their phase calibrator | FRING->SN4 (SNSMO->SN5)->CLCAL |
| CL7 | + phase-calibrator self-cal (amp & phase) transferred to the target | CALIB with model->SN6->CLCAL |
| BP1 | bandpass | EVN pipeline (TACOP) or BPASS |
| FG1 | correlator + station + pipeline flags (+ own UVFLG) | TACOP / UVFLG |

---

## 2. Step by step

### 2.1 Load data **[S1, S3]**
```
default fitld
datain 'PWD:exp_1_1.IDI      (first file; ncount = number of IDI files)
outna 'EXP'
ncount 8
doconcat 1                     (concatenate all IDI files into one catalogue entry)
dowait -1
go fitld
```
FITLD tips: `clint 0.25` (CL interval, min), `digicor -1` (SFXC already corrected), `douvcomp 1`.
Rebuild the index after concatenation:
```
default extdest; getn 1; inext 'NX'; go extdest
default indxr;   getn 1; go indxr
```
Inspect: `imh`, `prtan`, `listr optype 'SCAN'`, `dtsum`, `possm`, `vplot`, `uvplt` (`bparm 6 7 2` uv coverage).

### 2.2 A-priori tables **[S2, S4, S6]**
The EVN pipeline TASAV contains the a-priori amplitude calibration (+ parallactic angle) in CL2, the pipeline flags in
FG1 and the bandpass in BP1:
```
tget fitld; datain 'PWD:exp.tasav.FITS; ncount 1; doconcat -1; go fitld    (-> slot 2)
default tacop; getn 2; geton 1; ncount 1
inext 'cl'; invers 2; outvers 2; go tacop
inext 'fg'; invers 1; outvers 1; go tacop
inext 'bp'; invers 1; outvers 1; go tacop
getn 1; imh
```
**Without pipeline tables** (or to redo them) - for EVN use the archive `.antab` even if the FITS-IDI carry TY/GC
tables (embedded since January 2023, with appending issues): delete the loaded TY/GC (`extdest`) and run ANTAB:
```
default antab; getn 1; calin 'PWD:exp.antab'; tyver 0; gcver 0; offset 0.3; go antab
default snplt; getn 1; inext 'TY'; optype 'TSYS'; nplots 9; dotv 1; go snplt
default apcal; getn 1; tyver 1; gcver 1; snver 1; go apcal      (dofit 1 + opcode for opacity at >=22 GHz)
default clcal; getn 1; snver 1; gainver 1; gainuse 2; interpol 'self'; dobtween 1; go clcal
default clcor; getn 1; opcode 'PANG'; clcorprm 1 0; gainver 2; gainuse 2; go clcor
default uvflg; getn 1; intext 'PWD:exp.uvflg'; outfgver 1; go uvflg
```
(`run vlbautil; vlbapang` does PANG. `tysmo` clips bad Tsys; `tycop` copies Tsys between IFs/pols when one is bad.)
Compare the Tsys with the raw amplitudes (`/reduce-vlbi-data` S6); gain curves must stay positive over the observed
elevations. **VLBA/DiFX only:** `vlbaccor` straight after loading, `ACSCL` after BPASS, `vlbaeops` for EOPs.

Load the models needed later (the e-MERLIN 3C286 model ships as `3C286_<band>.ICL001` with vlbipy, in
`vlbipy/data/emerlin_calibrator_models`):
```
tget fitld; datain 'PWD:3C286_L.ICL001'; outname '3C286'; go fitld
```

### 2.3 Flagging **[S4, S8]**
```
default uvflg; getn 1
timer 0; sour ''; basel 0; stokes ''
anten 10 11 12 13 14 0         (Cm Da De Kn Pi)
bif 2; eif 2; bchan 0; echan 30
opcode 'FLAG'; reason 'eMERLIN IF2 edge'
go uvflg
```
- Undo with `opcode 'UFLG'` (same selection). Single hand: `stokes 'RR'`. Elevation: `aparm 0 0 0 0 10`.
- Interactive: `spflg` (time-channel), `tvflg` (time-baseline), `editr` (per baseline, good for sparse arrays),
  `edita` (Tsys-based). Scan edges: `quack` (`opcode 'BEG'`, `aparm 0 t_min`; `'TAIL'` for scan ends).
- Statistics per baseline; when a phase-calibrator scan is flagged for an antenna, flag that antenna on the adjacent
  target scans too.
- Record known problems in the log header (wavy bandpass in one hand, weak stations, lost scans, partial bands).

### 2.4 Constant gain corrections (known station factors) **[S12, S17]**
From an earlier calibration (amplitude self-cal of a compact calibrator, or Difmap `gscale`), voltage gain factors
per antenna/IF:
```
default clcor; getn 1
gainver 2; gainuse 3           (first run creates CL3 from CL2)
opcode 'GAIN'; stokes ''
anten 1 0; bif 1; eif 1; clcorprm 1.10 0; go clcor
```
Later corrections must modify the **same** table: `gainver 3; gainuse 3` (CLCOR writes in place only when GAINVER =
GAINUSE != 0). Factors are **voltage** gains: a baseline amplitude ratio R gives sqrt(R) per antenna when shared
equally. A scalar bandpass applies the same factor to both hands (`stokes ''`).

### 2.5 Ionosphere **[S5]**
```
run vlbautil
task 'vlbatecr'; getn 1; go vlbatecr      (downloads IGS TEC maps and runs TECOR -> new CL)
```
Manual: `default tecor; infile '/path/jplgDDD0.YYi'; nfiles 1..; gainver 2; gainuse 3; aparm(1) 1; go tecor`. Below
~8 GHz, always for L-band phase referencing.

### 2.6 e-MERLIN amplitude scale with 3C286 (EVN + e-MERLIN) **[S17]**
Only when e-MERLIN outstations are in the array (Jb alone does not count) and 3C286 was observed. The outstations lack
reliable a-priori Tsys: tie them **and Jb** to 3C286 with the e-MERLIN model, solving all of them on the baselines among
them:
```
default setjy; getn 1
source '3C286'; optype 'CALC'; bif 2; eif 3       (IFs where e-MERLIN observed)
go setjy

default calib; getn 1; get2n 3                    (get2n = 3C286 model image)
calsour '3C286' ''
docalib 1; gainuse 3; doband 1; bpver 1
snver 2
refant 1                       (Jb: phase reference only)
antenna 1 10 11 12 13 14       (Jb + e-MERLIN)
basel   1 10 11 12 13 14       (only baselines among these)
dofit   1 10 11 12 13 14       (solve Jb and the outstations)
cmethod 'dft'
solint 2                       (minutes; average the amplitudes over all 3C286 scans afterwards)
aparm 4 0 0 0 0 3 3            (aparm(1)=4 min antennas, aparm(6)=3 print, aparm(7)=3 min SNR)
soltype 'L1R'; solmode 'A&P'
weightit 1; antwt 0; doflag -1
nmaps 1; ncomp 1000 0
go calib
```
- Solve the phases first if they wander within a scan, then one amplitude per antenna, IF and hand over all 3C286 scans
  (smooth/average the SN amplitudes, e.g. `snsmo` with long windows, before CLCAL). IFs without e-MERLIN data take the
  nearest IF's solution.
- Check after applying: calibrated 3C286 amplitudes on those baselines vs the model per IF and hand. Check the model
  orientation (RA sign) if it came from another format. e-MERLIN resolves out a few per cent of 3C286: the e-MERLIN
  pipeline scales bootstrapped fluxes by `1 / (1 + 0.04 (f/5 GHz x L/11.237 km)^2)` (L = shortest baseline; see
  `reduce-vlbi-data/references/amplitude-calibration.md`).

Transfer to the other calibrators (point models, model cleared), then GETJY and CLCAL with CALP on the e-MERLIN
antennas only (CALP passes the others through unchanged):
```
calsour 'FRINGEF', 'PHASECAL' ''; clr2name; wait; go calib      (appends to SN2)
default getjy; getn 1; sour 'FRINGEF', 'PHASECAL'; calsour '3C286' ''; bif 2; eif 3; snver 2; go getjy
default clcal; getn 1
snver 2; gainver 3; gainuse 4; refant 1
antennas 1 10 11 12 13 14
opcode 'CALP'
calsour '3C286', 'FRINGEF', 'PHASECAL', ''; sour calsour; interpol 'self'
go clcal
calsour 'PHASECAL' ''; sour 'TARGET', 'CHECK', ''; interpol 'ambg'
wait; go clcal
```
Later self-calibration must keep these amplitudes (exclude the e-MERLIN antennas from amplitude solutions, or solve
them with very long intervals) while their phases are solved.

### 2.7 Instrumental delays: manual phase-cal **[S9]**
One short fringe-finder scan where all antennas have fringes (`possm` it first), on a strong source:
```
default fring; getn 1
calsour 'FRINGEF' ''
timer 0 21 31 0 0 21 41 59     (the scan)
anten 0
docal 1; gainuse 4; doband 1; bpver 1
bchan 7; echan 121             (avoid edge channels)
cmethod 'dft'; weightit 1
solint 11                      (= scan length, one solution)
refant 3                       (Ef)
search 1 4 6                   (fallback refs: Jb, Mc, O8)
aparm 2 0 0 0 0 1 10           (aparm(1)=2 min antennas, aparm(5)=0 per-IF, aparm(6)=1 print, aparm(7)=10 SNR)
dparm(9) 1                     (do not fit rates)
snver 3
go fring; wait
```
Compare good/failed counts with the expected (stations x pols x IFs actually observed). Apply to all sources:
```
default clcal; getn 1
source ''; calsour 'FRINGEF' ''
snver 3; gainver 4; gainuse 5; refant 3
interpol ''; opcode 'CALI'
go clcal; wait
```
No scan with all antennas: a second FRING on another scan with `anten` = refant + missing stations into the same SN,
then one CLCAL (`vlbampcl` does both). **VLBA:** pulse-cals with `pccor`/`vlbapcor`, checked with a manual phase-cal.
**Check:** `possm` with `docal 1; gainuse 5; aparm(9) 1` on a *different* scan: phases continuous across IFs.

### 2.8 Global fringe fitting **[S10]**
```
default fring; getn 1; clr2name
calsour 'FRINGEF', '3C286', 'PHASECAL' ''
timer 0; anten 0
docal 2; gainuse 5             (docal 2: calibrate data and weights)
doband 1; bpver 1; flagver 0
bchan 7; echan 121
cmethod 'dft'; weightit 1
solint 1                       (min; <= calibrator scan, <= coherence time)
refant 3; search 1 4 6
aparm(5) 1                     (one multi-band delay + rate across all IFs)
aparm(9) 1                     (exhaustive baseline search: finds weak stations)
aparm(10) 1                    (dispersive delay too, below ~8 GHz)
dparm 1 200 50 2 0 0 1         (1 baseline combo; delay window 200 ns; rate window 50 mHz; 2 s; no re-referencing)
snver 4
go fring; wait
```
`aparm(7)` = SNR cutoff (default ~5; the EVN pipeline uses 7-9; >=3 minimum). `aparm(3) 1` averages RR+LL (not for
polarimetry). `dparm(8)` zeroes rates/delays after the fit. Failures include no-data intervals: check with SNPLT.

Inspect:
```
default snplt; getn 1; inext 'SN'; invers 4; optype 'DELA'; nplots 9; dotv 1; go snplt
optype 'RATE'; go snplt;  optype 'PHAS'; go snplt;  optype 'SNR'; go snplt
```
Smooth/clip the delays before interpolation (per source: don't mix fringe-finder and phase-calibrator directions):
```
default snsmo; getn 1
invers 4; outvers 5; refant 3
smotype 'DELA'; samptype 'MWF'
bparm 2 0                      (smoothing widths, hours)
cparm 2 0 1 1 0 2 0 10 20 0    (clip limits)
doblank -1
go snsmo
```
Check `EXPLAIN SNSMO` for the BPARM/CPARM index mapping in your version; `smotype 'VLDE'` (with `npiece`) when FRING
combined IFs. An antenna noise-like on the phase calibrator but detected on the fringe finders keeps the latter.

Apply: calibrators with their own solutions, targets from **their** phase calibrator only:
```
default clcal; getn 1
snver 5; gainver 5; gainuse 6; refant 3
doblank -1; dobtwe 1
sour 'FRINGEF', '3C286', 'PHASECAL' ''; calsour sour
opcode 'CALI'; interp 'self'
go clcal
calsour 'PHASECAL' ''          (restrict to the phase calibrator!)
sour 'TARGET', 'CHECK' ''
interp 'ambg'                  (rate-aware phase connection)
wait; go clcal
```
`vlbafrgp` runs FRING + CLCAL for phase referencing in one go. Resolved phase calibrator (long baselines undetected): a
separate FRING with `anten` = the short-baseline antennas and one of them as `refant`, and flag the inter-sub-array
baselines in the split data.

### 2.9 Bandpass **[S11]**
After the instrumental delays and the global fringe fit (phases coherent):
```
default bpass; getn 1
calsour 'FRINGEF' ''
docal 1; gainuse 6; refant 3
solint -1                      (combine all scans)
bpassprm(5) 1                  (do not divide by channel 0)
bpassprm(9) 1                  (interpolate over flagged channels)
bpassprm(10) 6                 (normalise with power, whole band)
ichansel 7 121 1 0             (normalise over the full usable band; the default inner 75% can bias up to 15%)
weightit 1; bpver 2
go bpass
```
Plot with `possm aparm(8) 2`; edit with `bpedt`; apply the same `smooth` wherever the table is used. Flag the edge
channels the BP table reveals (UVFLG) and solve again; from then on `doband 1; bpver 2`.

### 2.10 Check the calibration **[S15]**
```
default possm; getn 1
sour 'PHASECAL' ''
docal 1; gainuse 6; doband 1; bpver 1
bchan 7; echan 121
aparm 0 1 0 0 -180 180; aparm(9) 1
stokes 'half'; solint 0; nplots 9; dotv 1
tvini; go possm
```
Calibrated calibrator phases flat (~0) across IFs and channels. `vplot` (`bparm 0 -1 1 0 0 0 0 -180 180`) for
amplitude/phase vs time; `snplt inext 'CL'; dodelay 1` for interpolated delays.

### 2.11 Split, average, export **[S16]**
```
default split; getn 1
sour ''; bif 0; eif 0
bchan 7; echan 121
docal 2; gainuse 6; doband 1; bpver 1; flagver 0
aparm 2 0                      (average channels within each IF; see EXPLAIN SPLIT)
nchav 0; stokes 'half'
go split; wait
default fittp; getn 8; dataout 'PWD:EXP.PHASECAL.SPLIT.uvfits'; go fittp
```
`aparm(2) 2` corrects amplitudes for the integration time; `aparm(1) 3` also averages IFs (not for self-cal). Time
averaging: `uvavg` or `splat`.

### 2.12 Image and self-calibrate the phase calibrator **[S17]**
**Difmap route:** image and self-calibrate the split UVFITS in Difmap (or difmapy; load it channel-averaged, keep the
cell <= beam/4.6), save the model image, and bring it back:
```
default fitld; datain 'PWD:PHASECAL.difmap.fits'; outname 'PHASECAL'; outclas ''; ncount 1; go fitld
default puthead; getn 6; keyword 'CRVAL4'; keyval 1 0; puthead   (Stokes axis = I so CALIB accepts it)
```
Difmap's `gscale` gives per-station/IF factors (apply with CLCOR, 2.4, when not self-calibrating the multi-source file
directly). Self-calibrate the phase calibrator in the multi-source file against the model, transfer to the target:
```
default calib; getn 1; get2n 7
calsour 'PHASECAL' ''
docal 1; gainuse 6
invers 0; nmaps 1; ncomp 0
refant 3; search 1 4 6
weightit 1; soltype 'L1R'; cmethod 'dft'
solint 2; aparm 3 0
solmode 'A&P'
snver 6
go calib

tget clcal; getn 1
snver 6; gainv 6; gainu 7
sour 'PHASECAL' ''; calsour sour; opcode 'CALI'; interpol 'self'; refant 3
go clcal
sour 'TARGET', 'CHECK' ''; interp 'ambg'
wait; go clcal
tget split; docal 1; gainu 7; sour 'PHASECAL', 'TARGET', 'CHECK' ''; go split
```
**AIPS-only route:** IMAGR -> CALIB (`solmode 'P'` first, point or CC model via `get2n`, low `aparm(7)` at first),
iterate, then `'A&P'` with a long solint and `cparm(2) 1` (mean gain = 1) or `solmode 'GCON'` + `gainerr`. Intervals:
phase 30 -> 15 -> 5 min -> scan; A&P first one solution for the whole run (station gains, like `gscale`), then
120 -> 60 -> 30 min while consistent (the 2-min A&P above is much shorter than this). `SCMAP`/`SCIMG` automate the
loop. Self-calibrate the *original* SPLIT with the newest model to avoid gain wander.

### 2.13 Imaging and measurements **[S18-S20]**
```
default imagr; getn 8
stokes 'I'
cellsize 4e-4                  (arcsec; ~beam/10)
imsize 4096
bif 1; eif 4; nchav 1
robust 5                       (+5 ~ natural, -5 ~ uniform)
niter 1000; gain 0.05
cmethod 'dft'
dotv 1
go imagr
```
Other adverbs: `uvtaper`, `uvrange`, `uvwtfn 'NA'|'UN'|'UV'`, `clbox`/`boxfile`, `rashift/decshift`,
`overlap`/`nfield`. Statistics: `imstat`, `tvstat`, `imean`. Fits: `jmfit` (image plane), `uvfit` (uv plane; can also
fit antenna gains with `gainerr 1`), `imfit`. Plots: `kntr`, `lwpla`. Export: `fittp`.

### 2.14 Polarisation and spectral line
- **Polarisation:** `clcor opcode 'PANG'` early; FRING with `aparm(3) 0`, `dparm(7) 1`; `rldly` (R-L delay);
  `lpcal`/`pcal` (D-terms, Stokes-I CC model); `clcor 'PHAS'` on RL/LR for the EVPA; `imagr stokes 'QU'`.
- **Spectral line:** delays/bandpass on continuum calibrators; `cvel` (Doppler), `acfit` (template amplitude
  calibration); fringe-fit a maser channel for rates, `calib` on it, apply to all; `uvlsf`; cube with `imagr nchav 1`.

---

## 3. Condensed flow of the reference reduction
1. FITLD 8 IDI files (doconcat) -> EXTDEST NX -> INDXR.
2. FITLD pipeline TASAV -> TACOP CL2, FG1, BP1.
3. FITLD e-MERLIN 3C286 L-band model.
4. UVFLG the first 30 channels of IF2 for Cm, Da, De, Kn, Pi.
5. VLBATECR -> ionosphere CL.
6. SETJY 3C286 (IF2-3) -> CALIB 3C286 (Jb + e-MERLIN solved) + other calibrators -> SN2 -> GETJY -> CLCAL CALP
   (Jb + e-MERLIN) -> CL4.
7. FRING manual phase-cal on the fringe finder (per IF, no rates) -> SN3 -> CLCAL -> CL5.
8. FRING global on the calibrators (1 min, combined IFs, exhaustive) -> SN4 -> SNSMO -> SN5 -> CLCAL (self for
   calibrators, ambg for target/check from the phase calibrator) -> CL6.
9. POSSM/SNPLT checks -> SPLIT (1 ch/IF) -> FITTP.
10. Difmap self-cal of the phase calibrator -> FITLD model, PUTHEAD CRVAL4 -> CALIB A&P -> CLCAL to target/check ->
    SPLIT -> FITTP -> IMAGR target.

## 4. Common AIPS pitfalls
- **Version collisions:** reusing SN/CL numbers mixes solutions or fails silently. `imh` before each solve.
- **CLCAL with SNVER 0** merges *all* SN tables: give the explicit version.
- **CLCAL AMBG with many CALSOUR:** the target takes the nearest solution from any listed calibrator - restrict CALSOUR
  to its phase calibrator.
- **After SNSMO** apply the smoothed table (OUTVERS).
- **CLCOR** creates a new CL each run unless GAINVER = GAINUSE.
- **BPASS normalisation** over the inner 75% (default) -> up to 15% amplitude error; set ICHANSEL to the full band.
- `docal 1` calibrates data only; `docal 2` also the weights (use 2 for FRING/SPLIT after amplitude calibration).
- Adverbs persist: `default <task>` or `tget` deliberately; reset `inext ''; invers 0` after EXTDEST.
- CALIB/FRING with a model image: `clr2name` afterwards, or later runs silently use it.
- Copy-pasted blocks from old projects: re-check every `getn`/`get2n` against `pcat`.

## References
AIPS Cookbook (31DEC23), Ch. 9 + App. C (http://www.aips.nrao.edu/cook.html); `EXPLAIN <task>`; the EVN pipeline
(ParselTongue); DARA/ERIS tutorials for the concepts; e-MERLIN calibration tutorial
(https://www.jb.man.ac.uk/DARA/ERIS22/calibration.html).
