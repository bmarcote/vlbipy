#! /usr/bin/env python3
import sys
import copy
import glob
import string
import logging
import argparse
import subprocess
import functools
from pathlib import Path
import datetime as dt
import numpy as np
from astropy.io import fits
from typing import Optional, Union
from rich import box, text
from rich.table import Table
from rich.console import Console
from rich import print as rprint
from AIPS import AIPS
from AIPSTask import AIPSTask
from AIPSData import AIPSUVData, AIPSImage, AIPSCat

"""Part of this code uses or is based on the EVN Pypeline code
written in ParselTongue from JIVE.
"""

__version__ = "1.0-dev1"



def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        message = f"{func.__name__}("
        for arg in args:
            message += "\n    "
            if isinstance(arg, AIPSUVData):
                message += f"UVData({arg.name}, {arg.klass}, {arg.seq})"
            elif isinstance(arg, AIPSImage):
                message += f"Image({arg.name}, {arg.klass}, {arg.seq})"
            else:
                message += str(arg)

        messval = ""
        for kwarg, kwval in kwargs.items():
            if isinstance(kwval, AIPSUVData):
                messval += f"UVData({kwval.name}, {kwval.klass}, {kwval.seq})"
            elif isinstance(kwval, AIPSImage):
                messval += f"Image({kwval.name}, {kwval.klass}, {kwval.seq})"
            else:
                messval += str(kwval)

            message += "\n    " + f"{kwarg} = {kwval}"
        message += ")\n"
        logging.debug(message)
        return func(*args, **kwargs)

    return wrapper


def aipsno_from_project(projectname: str):
    """Follows a personal way to get an AIPS no from EVN experiments which is:
    for XXAAAB (e.g. EM123C), it takes the numeric part (123) and the epoch is
    converted to a number following the alphabet, so the resulting AIPS no is 1233.

    Otherwise it returns 101.
    """
    alphabet = list(string.ascii_lowercase)
    if projectname[2:5].isnumeric():
        aipsno = projectname[2:5]
        if len(projectname) == 6:
            aipsno += str(alphabet.index(projectname[5].lower()) + 1)
    else:
        aipsno = '101'

    return int(aipsno)


def fix_difmap_image(fitsimage: Union[str, Path], output_verify='ignore'):
    """When using Difmap 'select pi', the stokes parameter in the stored FITS image
    is unexpected for both AIPS and CASA. This script fixes the metadata in the file
    """
    try:
        with fits.open(fitsimage, mode='update') as ffile:
            if 'CRVAL4' in ffile[0].header:
                if -9.0001 < ffile[0].header['CRVAL4'] < -8.999:
                    ffile[0].header['CRVAL4'] = 1.0
                    ffile.flush(output_verify=output_verify)
    except fits.VerifyError as e:
        print(f"WARNING: VerifyWarning: {e}")
        print("Continuing from here...")


def get_antenna_number(uvdata: AIPSUVData, antenna_name: str) -> int:
    """Returns the AIPS antenna number for a given antenna name.
    """
    assert uvdata.exists(), "The provided uvdata does not exist."
    try:
        ant_no = [ant.upper() for ant in uvdata.antennas].index(antenna_name.upper()) + 1
        logging.debug(f"Antenna {antenna_name} is ID {ant_no}.")
    except ValueError as verr:
        rprint(f"[red bold]Antenna {antenna_name} is not in UVDATA antenna table[/red bold]")
        logging.debug(f"Antenna {antenna_name} is not in UVDATA({uvdata.name}, " \
                      f"{uvdata.klass}, {uvdata.disk}, {uvdata.seq}).")
        raise ValueError(verr)

    return [ant.upper() for ant in uvdata.antennas].index(antenna_name.upper()) + 1


def highest_table(uvdata: AIPSUVData, inext: str) -> int:
    """Returns the highest number of the given AIPS table in the specified UVDATA.
    It returns zero if there is no entry for that table.
    """
    assert uvdata.exists(), "The provided uvdata does not exist."
    max_no: int = 0
    for table in uvdata.tables:
        if (table[1][-2:] == inext.upper()) and (table[0] > max_no):
            max_no = int(table[0])

    logging.debug(f"Table {inext} no {max_no} is the highest in UVDATA({uvdata.name}, " \
                  f"{uvdata.klass}, {uvdata.disk}, {uvdata.seq}).")
    return max_no


def pcat(userno: int, disk: int = 0, verbose: bool = True) -> dict:
    """Runs a PCAT in AIPS and returns all entries in the disk.

    Inputs
        - userno : int
            AIPS user number
        - disk : int  (default = 0)
            The disk to examinate. If 0, then it will return all available disks.
        - verbose : bool  (default True)
            Print in the STDOUT the output in a nice way.

    Returns
        - AIPS catalogue : dict
            Dictionary with the selected disk number as key, and a list of all files
            available on the given disk, in AIPSUVdata or AIPSImage format.
    """
    AIPS.userno = userno
    entries = AIPSCat(disk)
    for adisk in entries:
        for i,an_entry in enumerate(entries[adisk]):
            if an_entry['type'] == 'UV':
                entries[adisk][i] = AIPSUVData(an_entry['name'], an_entry['klass'],
                                               adisk, an_entry['seq'])
            elif an_entry['type'] == 'MA':
                entries[adisk][i] = AIPSImage(an_entry['name'], an_entry['klass'],
                                              adisk, an_entry['seq'])
            else:
                rprint(f"[yellow]WARNING: format of data {an_entry} not recognized[/yellow]")

        logentries = '\n '.join([ str(i+1) + '   ' + an_entry.name + '  ' + an_entry.klass + \
                     '  ' + str(adisk) + '   ' + str(an_entry.seq) \
                     for i,an_entry in enumerate(entries[adisk])])
        logging.debug(f"AIPS Catalog for user {AIPS.userno} - disk {adisk}:"
                      " N   Name     Class    Disk   Seq.\n" \
                      f"{logentries}")
    if verbose:
        other_text = ''
        tables = []
        for adisk in entries:
            if len(entries[adisk]) > 0:
                table = Table(title=text.Text(f"Catalog -- user {AIPS.userno} -- Disk {adisk}",
                                              style='green'),
                              box=box.SIMPLE, show_edge=False, )
                table.add_column("N", justify="right", no_wrap=True)
                table.add_column("Name", justify="left", no_wrap=True)
                table.add_column("Class", justify="left", no_wrap=True)
                table.add_column("Disk", justify="center", no_wrap=True)
                table.add_column("Seq.", justify="right", no_wrap=True)
                for i,an_entry in enumerate(entries[adisk]):
                    table.add_row(str(i+1), an_entry.name, an_entry.klass, str(adisk),
                                  str(an_entry.seq))
                tables.append(table)
            else:
                other_text = f"[green]No entries for userno {AIPS.userno}, disk {adisk}.[/green]"

        Console().print(*tables)
        rprint(other_text + '\n')

    return entries


def zap_all(disk: int = 0) -> bool:
    """Removes all data entries in the given disk.
    Returns True if the operation successed.
    """
    logging.info(f"Removing all entries in userno {AIPS.userno}, disk {disk}.")
    AIPSCat(disk).zap(force=True)
    return True


@log
def fitld(projectname: str, datain: str, ncount: int = 1, doconcat: int = 1,
          outclass: str = "UVDATA", outseq: int = 1, digicor=-1, douvcomp: int = -1,
          clint: float = 0.25, replace=True, optype: str = 'UV') -> Union[AIPSUVData, AIPSImage]:
    """Imports the associated FITS-IDI or UVFITS from a given project into AIPS.
    Note that one and only one of the 'fitsidifiles' or 'uvfits; parameters needs to be provided.

    Inputs
        - projectname : str  (max len <= 6)
            Name that will be associated with the created data.
        - fitsidifiles : str (optional)
            Preffix common to all the FITS-IDI files to be imported.
        - uvfits : str (optional)
            UVFITS file to be imported.

    Returns
        - uvdata : AIPSUVData or AIPSImage
            AIPSUVData that will be created from the imported data. If exists, it will be replaced.
    """
    if optype == 'UV':
        uvdata = AIPSUVData(projectname, outclass, 1, outseq)
    elif optype == 'IM':
        uvdata = AIPSImage(projectname, outclass, 1, outseq)
    else:
        raise ValueError(f"OPTYPE can only be 'UV' or 'IM', but {optype} found.")

    if uvdata.exists():
        if replace:
            uvdata.zap(force=True)
        else:
            return uvdata

    fitld = AIPSTask('fitld')
    # fitld.outdata = uvdata
    fitld.outname = uvdata.name
    fitld.outclass = uvdata.klass
    fitld.outseq = uvdata.seq
    fitld.douvcomp = douvcomp
    fitld.clint = clint
    fitld.digicor = digicor
    fitld.doconcat = doconcat
    fitld.ncount = ncount
    fitld.datain = str(Path(datain).absolute())
    fitld()
    return uvdata


@log
def indxr(uvdata, cparm=[0, 22, 0.25, 0]):
    """Runs INDXR in the associated UVDATA.
    """
    indxr = AIPSTask('indxr')
    indxr.indata = uvdata
    indxr.cparm[1:] = cparm
    indxr()


@log
def tacop(uvdata: AIPSUVData, fromuvdata: AIPSUVData, inext: str, inver: int, ncount: int,
          outver: int):
    """Copies a table version from 'fromuvdata' into 'uvdata'.
    """
    tacop = AIPSTask('tacop')
    tacop.indata = fromuvdata
    tacop.outdata = uvdata
    tacop.inext = inext
    tacop.inver = inver
    tacop.outver = outver
    tacop.ncount = ncount
    tacop()


@log
def ionos(uvdata: AIPSUVData, aparm=[1, 0, 0, 0.85, 56.7, 0.9782]):
    # Replicates the VLBATECOR lines to grap the IONEX files
    assert uvdata.exists()
    nx = uvdata.table('NX', 1)
    date_obs = dt.datetime.strptime(uvdata.header['date_obs'], '%Y-%m-%d')
    dt0 = date_obs + dt.timedelta(days=nx[0]['time'])
    dt1 = date_obs + dt.timedelta(days=nx[-1]['time'])
    ionex_files = []
    for a_day in range(dt0.day, dt1.day+1):
        a_date = date_obs + dt.timedelta(days=a_day-date_obs.day)
        ionex_files.append(f"jplg{a_date.strftime('%j')}0.{a_date.strftime('%y')}i")
        if not Path(ionex_files[-1]).exists():
            curl_cmd = ["curl", "-u", "anonymous:daip@nrao.edu", "--ftp-ssl",
                        "ftp://gdc.cddis.eosdis.nasa.gov/gps/products/ionex/" \
                        f"{a_date.year}/{a_date.strftime('%j')}/" \
                        f"jplg{a_date.strftime('%j')}0.{a_date.strftime('%y')}i.Z", "-o",
                        f"./jplg{a_date.strftime('%j')}0.{a_date.strftime('%y')}i.Z"]
            # print(f"Trying {' '.join(curl_cmd)}")
            r = subprocess.run(curl_cmd, shell=False, stdout=None, stderr=subprocess.STDOUT)
            r.check_returncode()
            r = subprocess.run(["gunzip", ionex_files[-1]+".Z"], shell=False, stdout=None,
                               stderr=subprocess.STDOUT)
            r.check_returncode()

    tecor = AIPSTask('tecor')
    tecor.indata = uvdata
    tecor.nfiles = len(range(dt0.day, dt1.day+1))
    tecor.gainver = highest_table(uvdata, 'CL')
    tecor.gainuse = highest_table(uvdata, 'CL') + 1
    tecor.aparm[1:] = aparm
    tecor.infile = f"PWD:{ionex_files[0]}"
    tecor()


@log
def fring(uvdata: AIPSUVData, calsour: list, solint: float, refant: list, docalib: int = 2,
          parmode: Optional[str] = None, aparm: Optional[list] = None,
          dparm: Optional[list] = None, snr: int = 0,
          gainuse: Optional[int] = None, flagver: Optional[int] = 0, snver: Optional[int] = None,
          doband: int = -1, bpver: int = -1, weightit: int = 1, model : Optional[AIPSImage] = None,
          timer: list = [0, 0, 0, 0, 0, 0, 0, 0], cmethod: str = 'dft', bchan: int = 0,
          echan: int = 0, antenna: list = []):
    """Runs FRING on the given data.
    Special functions:
    parmode : str 'mbd' or 'sbd' (default None)
        Automatically selects the aparm/dparm appropate for the instrumental delay fring ('sbd')
        or the multi-band fring ('mbd'). If None, it will pick the provided values for aparm/dparm.
    """
    fring = AIPSTask('fring')
    fring.indata = uvdata
    fring.calsour[1:] = calsour
    fring.timer[1:] = timer
    fring.antenna[1:] = [get_antenna_number(uvdata, a) if type(a) is str else a for a in antenna]
    fring.docalib = docalib
    if gainuse is None:
        fring.gainuse = highest_table(uvdata, 'CL')
    else:
        fring.gainuse = gainuse

    fring.doband = doband
    fring.bpver = bpver
    fring.bchan = bchan
    fring.echan = echan
    # TODO:
    # fring.log = open('fringe.log', 'a')
    fring.cmethod = cmethod
    fring.weightit = weightit
    fring.solint = solint
    fring.refant = get_antenna_number(uvdata, refant[0]) if type(refant[0]) is str else refant[0]
    fring.search[1:] = [get_antenna_number(uvdata, a) if type(a) is str else a for a in refant[1:]]
    if snver is None:
        fring.snver = highest_table(uvdata, 'SN') + 1
    else:
        fring.snver = snver

    if flagver is None:
        fring.flagver = highest_table(uvdata, 'FG')
    else:
        fring.flagver = flagver

    if parmode == 'sbd':
        fring.aparm[1:] = [2, 0, 0, 0, 0, 1, 10 if snr == 0 else snr]
        fring.dparm[1:] = [1, 200, 50, 2, 0, 0, 1, 0, 1]
    elif parmode == 'mbd':
        fring.aparm[1:] = [2, 0, 0, 0, 1, 1, 3 if snr == 0 else snr, 0, 1]
        fring.dparm[1:] = [1, 200, 50, 2, 0, 0, 1, 0, 0]
    elif parmode is None:
        fring.aparm[1:] = aparm if aparm is not None else [2, 0, 0, 0, 1, 1, 3 if snr == 0 else snr, 0, 1]
        fring.dparm[1:] = dparm if dparm is not None else [1, 200, 50, 2, 0, 0, 1, 0, 0]
    else:
        raise ValueError(f"'parmode' can only be 'sbd' or 'mbd' (or None). But is {parmode}.")

    if model is not None:
        fring.in2data = model

    fring()


@log
def clcal(uvdata: AIPSUVData, calsour: list, refant: list, snver: int, gainver: int, gainuse: int,
          opcode: str = 'CALI', interpol: str = '2PT', sources: list[str] = [''], samptype: str = '',
          doblank: int = 0, dobtween: int = 0, cutoff: int = 0):
    clcal = AIPSTask('clcal')
    clcal.indata = uvdata
    clcal.calsour[1:] = calsour
    clcal.sources[1:] = sources
    clcal.opcode = opcode
    clcal.interpol = interpol
    #clcal.intparm = 0.00001
    clcal.samptype = samptype
    clcal.doblank = doblank
    clcal.dobtween = dobtween
    clcal.refant = get_antenna_number(uvdata, refant[0]) if type(refant[0]) is str else refant[0]
    clcal.snver = snver
    clcal.gainver = gainver
    clcal.gainuse = gainuse
    clcal.cutoff = cutoff
    clcal()


@log
def bpass(uvdata: AIPSUVData, calsour: list, refant: list, gainuse: int,
          flagver: int = 0, docalib: int = 1, solint: int = -1,
          soltype: str = 'L1R', bpver: int = 0, smooth: list = [1,],
          bpassprm: list = [0, 0, 0, 0, 1, 0, 0, 0, 1], weightit: int = 1):
    bpass = AIPSTask('bpass')
    bpass.indata = uvdata
    bpass.calsour[1:] = calsour
    bpass.flagver = flagver
    bpass.docalib = docalib
    bpass.gainuse = gainuse
    bpass.solint = solint
    bpass.refant = get_antenna_number(uvdata, refant[0]) if type(refant[0]) is str else refant[0]
    bpass.soltype = soltype
    bpass.smooth[1:] = smooth
    bpass.bpassprm[1:] = bpassprm
    bpass.bpver = bpver
    bpass.weightit = weightit
    bpass()


@log
def calib(uvdata: AIPSUVData, refant: list, calsour: list = [''],
          timerang: list = [0, 0, 0, 0, 0, 0, 0, 0], antennas: list = [], dofit: list = [],
          antuse: list = [], uvrange: list = [0, 0], weightit: int = 1, docalib: int = -1,
          gainuse: int = 0, flagver: int = 0, doband: int = -1, bpver: int = 0,
          smooth: list = [0, 0, 0], in2data: Optional[AIPSImage] = None, invers: int = 0,
          ncomp: list = [0], flux: float = 0.0, nmaps: int = 1, cmethod: str = 'dft',
          cmodel: str = 'comp', solint: float = 0.0, aparm: list = [2, 0, 1, 0, 0, 3, 3],
          doflag: int = 1, soltype: str = 'L1R', normaliz: int = 0, snver: int = 0,
          solmode: str = 'a&p', cparm: list = [0, 2, 0]):
    calib = AIPSTask('calib')
    calib.indata = uvdata
    if in2data is not None:
        calib.in2data = in2data

    calib.refant = get_antenna_number(uvdata, refant[0])
    calib.calsour[1:] = calsour
    calib.timerang[1:] = timerang
    calib.antennas[1:] = [get_antenna_number(uvdata, ant) if type(ant) is str \
                          else ant for ant in antennas]
    calib.dofit[1:] = dofit
    calib.antuse[1:] = antuse
    calib.uvrange[1:] = uvrange
    calib.weightit = weightit
    calib.docalib = docalib
    calib.gainuse = gainuse
    calib.flagver = flagver
    calib.doband = doband
    calib.bpver = bpver
    calib.smooth[1:] = smooth
    calib.invers = invers
    calib.ncomp[1:] = ncomp
    calib.flux = flux
    calib.nmaps = nmaps
    calib.cmethod = cmethod
    calib.cmodel = cmodel
    assert solmode.upper() in ('A&P', 'P', 'P!A', 'GCON')
    calib.solmode = solmode
    calib.solint = solint
    calib.aparm[1:] = aparm
    calib.cparm[1:] = cparm
    calib.doflag = doflag
    calib.soltype = soltype
    calib.normaliz = normaliz
    calib.snver = snver
    calib()
    return AIPSUVData(uvdata.name, 'CALIB', 1, 1)


@log
def splat(uvdata: AIPSUVData, sources: list = [''], bif: int = 0, eif: int = 0, bchan: int = 0,
          echan: int = 0, docal: int = 1, gainuse: int = 0, flagver: int = 0, bpver: int = 0,
          doband: int = 1, aparm: list = [0], stokes: str = '', ichansel: list = [0, 0, 0, 0],
          channel: int = 0, chinc: int = 1, solint: int = 0):
    splat = AIPSTask('splat')
    splat.indata = uvdata
    # TODO: outname!!
    splat.sources[1:] = sources
    splat.bif = bif
    splat.eif = eif
    splat.bchan = bchan
    splat.echan = echan
    splat.docal = docal
    splat.gainuse = gainuse
    splat.flagver = flagver
    splat.doband = doband
    splat.bpver = bpver
    splat.aparm[1:] = aparm
    splat.stokes = stokes
    # TODO: somehow this is not accepting a [0], nor [0, 0,]... Maybe now that is [[], []]
    splat.ichansel[1][1:] = ichansel
    splat.channel = channel
    splat.chinc = chinc
    splat.solint = solint
    splat()

@log
def split(uvdata: AIPSUVData, sources: list = [''], bif: int = 0, eif: int = 0, bchan: int = 0,
          echan: int = 0, docal: int = 1, gainuse: int = 0, flagver: int = 0, bpver: int = 0,
          doband: int = 1, aparm: list = [2, 0, 0, 1], nchav: int = 0):
    split = AIPSTask('split')
    split.indata = uvdata
    split.sources[1:] = sources
    split.bif = bif
    split.eif = eif
    split.bchan = bchan
    split.echan = echan
    split.docal = docal
    split.gainuse = gainuse
    split.flagver = flagver
    split.doband = doband
    split.bpver = bpver
    split.aparm[1:] = aparm
    split.nchav = nchav
    split()


@log
def multi(uvdata: AIPSUVData, outname: str = '', outclass: str = 'MULTI', outseq: int = 0,
          outdisk: int = 1, srcname: str = '', aparm: list = [0]):
    """Task to convert single-source to multi-source UV data
    """
    multi = AIPSTask('multi')
    multi.indata = uvdata
    while AIPSUVData(outname if outname != '' else uvdata.name, outclass,
                     outdisk, outseq+1).exists():
        outseq += 1

    multi.outdata = AIPSUVData(outname if outname != '' else uvdata.name, outclass,
                               outdisk, outseq+1)
    multi.srcname = srcname
    multi.aparm[1:] = aparm
    multi()
    return AIPSUVData(outname if outname != '' else uvdata.name, outclass,
                               outdisk, outseq+1)


@log
def uvflg(uvdata: AIPSUVData, intext: Optional[str] = None,
          sources: Optional[list[str]] = None, timerange: Optional[list[int]] = None,
          bchan: int = 0, echan: int = 0, bif: int = 0, eif: int = 0,
          antennas: Optional[Union[str, list[Union[str, int]]]] = None,
          baselines: Optional[Union[str, list[Union[str, int]]]] = None, stokes: str = '',
          min_elevation: int = 0, max_elevation: int = 0, opcode: str = 'flag',
          reason: str = ''):
    """Flags selected UV data.
    """
    uvflg = AIPSTask('uvflg')
    uvflg.indata = uvdata
    if intext is not None:
        uvflg.intext = intext

    uvflg.sources[1:] = sources if sources is not None else ['']
    uvflg.timerang[1:] = timerange if timerange is not None else [0]
    uvflg.bchan = bchan
    uvflg.echan = echan
    uvflg.bif = bif
    uvflg.eif = eif
    if antennas is not None:
        uvflg.antennas[1:] = [get_antenna_number(uvdata, a) if type(a) is str  \
                             else a for a in antennas]
    else:
        uvflg.antennas[1:] = [0,]

    if baselines is not None:
        uvflg.baseline[1:] = [get_antenna_number(uvdata, a) if type(a) is str  \
                             else a for a in baselines]
    else:
        uvflg.baseline[1:] = [0,]

    uvflg.stokes = stokes.upper()
    uvflg.aparm[1:] = [min_elevation, max_elevation]
    uvflg.opcode = opcode.upper()
    uvflg.reason = reason
    uvflg()

@log
def quack(uvdata: AIPSUVData, stokes: str = '', bif: int = 0, eif: int = 0,
          sources: Optional[list[str]] = None, timerange: Optional[list[int]] = None,
          antennas: Optional[list[Union[str, int]]] = None,
          baselines: Optional[list[Union[str, int]]] = None, flagver: int = 0,
          opcode: str = 'BEG', reason: str = '', endcut_min: float = 0.0,
          begcut_min: float = 0.1):
    """Flags specified portion of scans of UV data
    """
    quack = AIPSTask('quack')
    quack.indata = uvdata
    quack.stokes = stokes
    quack.bif = bif
    quack.eif = eif
    quack.sources[1:] = sources if sources is not None else ['']
    quack.timerang[1:] = timerange if timerange is not None else [0]
    if antennas is not None:
        quack.antennas[1:] = [get_antenna_number(uvdata, a) if type(a) is str  \
                             else a for a in antennas]
    else:
        quack.antennas[1:] = [0,]

    if baselines is not None:
        quack.baseline[1:] = [get_antenna_number(uvdata, a) if type(a) is str  \
                             else a for a in baselines]
    else:
        quack.baseline[1:] = [0,]

    quack.flagver = flagver
    quack.opcode = opcode.upper()
    quack.reason = reason
    quack.aparm[1:] = [endcut_min, begcut_min]
    quack()


@log
def fittp(uvdata: Union[AIPSUVData, AIPSImage], dataout: str):
    fittp = AIPSTask('fittp')
    fittp.indata = uvdata
    fittp.dataout = str(Path(dataout).absolute())
    fittp()


@log
def antab(uvdata: AIPSUVData, calin: str):
    """Reads amplitude calibration info from an ANTAB file and incorporates it
    into the uvdata.
    """
    antab = AIPSTask('antab')
    antab.indata = uvdata
    if not Path(calin).exists():
        rprint(f"[bold red]The ANTAB file {calin} could not be found[/bold red]")
        raise FileNotFoundError(f"The ANTAB file {calin} could not be found")

    antab.calin = calin
    antab()

@log
def apcal(uvdata: AIPSUVData, antennas: Optional[list[Union[str, int]]] = None,
          stokes: str = '', sources: list = [''], bif: int = 0, eif: int = 0,
          timerange: Optional[list[int]] = None, tyver: int = 0, gcver: int = 0,
          snver: int = 0, opcode: str = '', invers: int = 0, calin: str = '',
          tau0: list = [0,], dofit: list = [0,], aparm: list = [0], prtlev: int = 0,
          dotv: int = -1, ltype: int = 3, grchan: int = 0, solint: int = 0):
    """Generates an amplitude calibration SN table
    """
    apcal = AIPSTask('apcal')
    apcal.indata = uvdata
    if antennas is not None:
        apcal.antennas[1:] = [get_antenna_number(uvdata, a) if type(a) is str  \
                             else a for a in antennas]
    else:
        apcal.antennas[1:] = [0,]

    apcal.stokes = stokes
    apcal.sources = sources
    apcal.bif = bif
    apcal.eif = eif
    apcal.timerang[1:] = timerange if timerange is not None else [0]
    apcal.tyver = tyver
    apcal.gcver = gcver
    apcal.snver = snver
    apcal.opcode = opcode
    apcal.aparm[1:] = aparm
    apcal.solint = solint
    apcal.invers = invers
    apcal.calin = calin
    # apcal.trecvr[1:] = trecvr
    apcal.tau0[1:] = tau0
    apcal.dofit[1:] = dofit
    apcal.prtlev = prtlev
    apcal.dotv = dotv
    apcal.ltype = ltype
    apcal.grchan = grchan
    apcal()

@log
def clcor(uvdata: AIPSUVData, sources: Optional[list[str]] = None, stokes: str = '',
          bif: int = 0, eif: int = 0, timerange: Optional[list[int]] = None,
          antennas: Optional[list[Union[str, int]]] = None, gainver: int = 0,
          gainuse: int = 0, opcode: str = '', clcorprm: Optional[list] = None,
          infile: str = ''):
    """Applies various corrections to CL tables.
    """
    clcor = AIPSTask('clcor')
    clcor.indata = uvdata
    clcor.sources[1:] = sources if sources is not None else ['']
    clcor.timerang[1:] = timerange if timerange is not None else [0]
    if antennas is not None:
        clcor.antennas[1:] = [get_antenna_number(uvdata, a) if type(a) is str  \
                             else a for a in antennas]
    else:
        clcor.antennas[1:] = [0,]

    clcor.stokes = stokes
    clcor.bif = bif
    clcor.eif = eif
    clcor.gainver = gainver
    clcor.gainuse = gainuse
    clcor.opcode = opcode
    clcor.clcorprm[1:] = clcorprm
    clcor.infile = infile
    clcor()



"""XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
From here we define the functions that will process the data
(one level higher than the actual AIPS tasks)
"""

def download_evn(projectname: str, obsdate: Union[int, str], username: Optional[str] = None,
                 password: Optional[str] = None) -> bool:
    """Downloads all relevant files from the EVN archive required to run the data reduction.

    Inputs
    ------
        - projectname : str
            Name that will be assigned to the imported data (must be <8 characters)
        - obsdate : str or int
            Epoch of the observation in YYMMDD format.
        - username : str  (default None)
            Username to access the data if still propietary. If None but password given, it
            will be assumed to be the projectname in lower cases.
        - password : str  (default None)
            Password to access the data if still propietary.
    """
    params = ["wget"]
    if password is not None:
        params += ['--http-user', username if username is not None else projectname.lower(),
                   '--http-passwd', password]

    params += ["-t45", "-l1", "-r", "-nd", "archive.jive.nl/exp/" \
                   f"{projectname.upper()}_{obsdate}/fits -A '{projectname.lower()}*'"]
    try:
        subprocess.run(params, shell=True, stdout=None, check=True)
        subprocess.run(["md5sum", "-c", f"{projectname.lower()}.checksum"], shell=True, stdout=None, check=True)
        # subprocess.run(["wget", f"http://archive.jive.nl/exp/{projectname.upper()}_{obsdate}/" \
        #                 f"pipe/{projectname.lower()}.antab.gz"], shell=True, stdout=None, check=True)
        # subprocess.run(["gunzip", f"{projectname.lower()}.antab.gz"], shell=True, stdout=None, check=True)
        subprocess.run(["wget", f"http://archive.jive.nl/exp/{projectname.upper()}_{obsdate}/" \
                        f"pipe/{projectname.lower()}.uvflg"], shell=True, stdout=None, check=True)
        subprocess.run(["wget", f"http://archive.jive.nl/exp/{projectname.upper()}_{obsdate}/" \
                        f"pipe/{projectname.lower()}.tasav.FITS.gz"], shell=True, stdout=None, check=True)
        subprocess.run(["gunzip", f"{projectname.lower()}.tasav.FITS.gz"], shell=True, stdout=None, check=True)
    except ValueError as err:
        rprint(f"[bold red]ERROR: you may require credentials to download the data[/bold red]")
        rprint(f"[red]{err}[/red]")

    return all(f.exists() for f in (Path(f"pipe/{projectname.lower()}.tasav.FITS"),
                                    Path(f"pipe/{projectname.lower()}.uvflg"))) \
           and (len(glob.glob(f"{projectname.lower()}_*_1.IDI*")) > 0)



def importdata(projectname: str, disk: int = 1, replace: bool = True,
                  fitsidifiles: Optional[str] = None,
                  uvfits: Optional[str] = None) -> AIPSUVData:
    """Imports the data to AIPS, either from FITS-IDI files or from UVFITS.
    For the former ones, an associated 'projectname'.tasav.FITS file is expected,
    and will be used to copy the a-priori calibration and flagging to the data.

    For UVFITS, then both a 'projectname'.antab file and .uvflg are expected for
    the same reason.

    If replace is False and the task runs on existing data, it will only remove
    all calibration tables that it may have.

    Inputs
    ------
        - projectname : str
            Name that will be assigned to the imported data (must be <8 characters)
        - replace : bool  (default True)
            If the data exists, then it will be overwritten.
            Otherwise, the imported data will stay but the extra calibration tables
            will be removed.
        - fitsidifiles : str  (optional, default None)
            Common name to the FITS-IDI files to be imported (e.g. without the trailing
            number). For EVN data, it is expected to be 'exp_N_1.IDI'.
        - uvfits : str  (optional, default None)
            UVFITS file to be imported.
        One and only one of the two 'fitsidifiles' or 'uvfits' parameters must be provided.

    Returns
    -------
       AIPSUVData : imported data
    """
    assert len(projectname) <= 8, "The length of the project name must be <8 characters"
    uv = AIPSUVData(projectname.upper(), "UVDATA", disk, 1)
    if replace:
        logging.debug(f"Zapping all data.")
        zap_all()
    elif uv.exists():
        uv.zap_table('SN', -1)
        uv.zap_table('BP', -1)
        while  highest_table(uv, 'CL') > 2:
            uv.zap_table('CL', 0)

        return uv

    assert (fitsidifiles, uvfits).count(None) == 1, "One and only one of the 'fitsidifiles' or" \
            " 'uvfits' parameters need to be provided."
    if fitsidifiles is not None:
        ncount = len(glob.glob(f"{fitsidifiles}*"))
        if ncount == 0:
            raise FileNotFoundError(f"Could not find files with the name {fitsidifiles}")

        uvtasav = AIPSUVData(projectname.upper(), "TASAV", disk, 1)
        if uvtasav.exists() and replace:
            uvtasav.zap(force=True)

        if not uvtasav.exists():
            uv = fitld(projectname.upper(), datain=fitsidifiles, outclass="UVDATA",
                       replace=replace, doconcat=1, ncount=ncount, digicor=-1, douvcomp=1)
            if not Path(f"{projectname.lower()}.tasav.FITS").exists():
                rprint(f"\n\n[bold red]The file {projectname.lower()}.tasav.FITS should " \
                       "exist but is not found[/bold red]")
                sys.exit(1)
            uvtasav = fitld(projectname.upper(), datain=f"{projectname.lower()}.tasav.FITS",
                            outclass="TASAV", replace=replace, doconcat=-1, ncount=1,
                            digicor=-1, douvcomp=1)
            indxr(uv)
            tacop(uv, fromuvdata=uvtasav, inext='CL', inver=2, ncount=1, outver=2)
            # Not really necessary as I will apply the uvflg file later
            tacop(uv, fromuvdata=uvtasav, inext='FG', inver=1, ncount=1, outver=1)
    elif uvfits is not None:
        if not uv.exists():
            uv = fitld(projectname.upper(), datain=uvfits, outclass="UVDATA",
                           replace=replace, doconcat=-1, ncount=1, digicor=-1)
            indxr(uv)
            antab(uv, f"{projectname.lower()}.antab")
            apcal(uv, snver=1)
            clcal(uv, calsour=[''], refant=[0], snver=1, gainver=1, gainuse=2)
            # Parallactic angle correction
            clcor(uv, clcorprm=[0, 1], opcode='PANG', gainver=2, gainuse=2)

    if not uv.exists():
        rprint(f"[bold red]It seems like {uv} does not exist and should be![/bold red]")
        raise FileNotFoundError(f"It seems like {uv} does not exist and should be!")

    if Path(f"{projectname.lower()}.uvflg").exists():
        # It will apply the a-priori flagging again as there may be some extra flags added
        # by the user
        uvflg(uv, intext=f"{projectname.lower()}.uvflg", reason='Flags from file')

    return uv


def main_calibration(projectname: str, refant: list, solint: float,
                     target: list, bpsour : list, phaseref: Optional[list] = None,
                     sbd_timerange: Optional[list] = None,
                     model: Optional[Union[str, list]] = None,
                     bchan: int = 0, echan: int = 0, tecor: bool = True):
    """bpsour are the fringe finders and calsour should contain all calibrators

    avgchan_split True/False if all channels within a IF will be averaged or not.
    """
    assert len(target) > 0 and len(bpsour) > 0
    uvdata = AIPSUVData(projectname.upper(), "UVDATA", 1, 1)
    assert uvdata.exists(), "The UVDATA should already exist in AIPS as importing has been set."

    if model is not None:
        # model = AIPSImage()
        raise NotImplementedError("Running FRING with a src model is not implemented yet " \
                                  "in main_calibration")

    if tecor:
        ionos(uvdata)

    for iteration in range(2):
        cl_last = highest_table(uvdata, "CL")
        sn_last = highest_table(uvdata, "SN")
        bp_last = highest_table(uvdata, "BP")
        if (sbd_timerange is not None) and (len(sbd_timerange) != 8):
            raise NotImplementedError("Only one time stamp for sbd is supported for now.")
            # fring(uvdata, calsour=[""], solint=10, refant=refant, parmode='sbd', gainuse=cl_last,
            #       flagver=0, snver=sn_last+1, bpver = -1 if bp_last == 0 else bp_last, #model=model,
            #       timer=sbd_timerange[:8])
            # clcal(uvdata, calsour=calsour, refant=refant, snver=sn_last+1, gainver=cl_last,
            #       gainuse=cl_last+1, opcode='CALP')
            # # TODO Double-check this was the right way to do it
            # fring(uvdata, calsour=[""], solint=10, refant=refant, parmode='sbd', gainuse=cl_last+1,
            #       flagver=0, snver=sn_last+2, bpver = -1 if bp_last == 0 else bp_last, #model=model,
            #       timer=sbd_timerange[8:])
            # clcal(uvdata, calsour=calsour, refant=refant, snver=sn_last+2, gainver=cl_last+1,
            #       gainuse=cl_last+2, opcode='CALP')
        elif sbd_timerange is not None:
            fring(uvdata, calsour=[""], solint=10, refant=refant, parmode='sbd', gainuse=cl_last,
                  flagver=0, snver=sn_last + 1, bpver = bp_last, bchan=bchan, echan=echan, #model=model,
                  doband=1 if bp_last > 0 else -1, timer=sbd_timerange)
            clcal(uvdata, calsour=[""], refant=refant, snver=sn_last+1, gainver=cl_last,
                  gainuse=cl_last+1, opcode='CALP', interpol='2PT')   # VERIFTY THE ITERPOLS HERE AND AFTER

        cl_last = highest_table(uvdata, "CL")
        sn_last = highest_table(uvdata, "SN")
        fring(uvdata, calsour=bpsour + (target if phaseref is None else phaseref), solint=solint,
              refant=refant, parmode='mbd', gainuse=cl_last, flagver=0, snver=sn_last+1, bchan=bchan,
              echan=echan, bpver=bp_last, doband=1 if bp_last > 0 else -1) # model=model
        for a_source in bpsour + (target if phaseref is None else phaseref):
            clcal(uvdata, calsour=[a_source], sources=[a_source], refant=refant, snver=sn_last+1,
                  gainver=cl_last, gainuse=cl_last+1, opcode='CALI', interpol='SELF')

        if phaseref is not None:
            assert len(phaseref) in (1, len(target)), "The number of provided phase calibrators" \
                    " must match the number of target sources, or being only one."
            if len(phaseref) == 1:
                clcal(uvdata, calsour=phaseref, sources=target, refant=refant, snver=sn_last+1, samptype='EXP',
                      gainver=cl_last, gainuse=cl_last+1, opcode='CALI', interpol='AMBG', cutoff=8)
            else:
                for a_target, a_phaseref in zip(target, phaseref):
                    clcal(uvdata, calsour=[a_phaseref], sources=[a_target], refant=refant,
                          snver=sn_last+1, gainver=cl_last, gainuse=cl_last+1, opcode='CALI', samptype='EXP',
                          interpol='AMBG')

        if iteration == 0:
            bpass(uvdata, gainuse=highest_table(uvdata, "CL"), calsour=bpsour, refant=refant)



# def selfcalibration_on_splits(aipsid: int, projectname: str, refant: list, imagefile: str, target: list,
#                     phaseref: Optional[list] = None, calsour: Optional[str] = None,
#                     solmode: str = 'a&p', solint: float = 0.0, soltype: str = 'L1R', **kwargs):
#     AIPS.userno = aipsid
#      # Guesses the source name from the image file to import
#     uvimage = None
#     if calsour is not None:
#         uvimage = fitld(calsour, datain=f"PWD:{imagefile}", doconcat=-1, outclass='PICLN',
#                         replace=True, ncount=1, digicor=-1)
#     elif ('pcal' in imagefile) and (len(phaseref) == 1):
#         uvimage = fitld(phaseref[0], datain=f"PWD:{imagefile}", doconcat=-1, outclass='PICLN',
#                         replace=True, ncount=1, digicor=-1)
#     else:
#         for a_src in (target + phaseref if phaseref is not None else []):
#             if a_src in imagefile:
#                 print(f"It will be : '{a_src}', 'PICLN'.")
#                 uvimage = fitld(a_src, datain=f"PWD:{imagefile}", doconcat=-1, outclass='PICLN',
#                                 replace=True, ncount=1, digicor=-1)
#                 break
#
#     if uvimage is None:
#         print("The filename must contain the name of the source in the data. " \
#               "Otherwise 'calsour' must be provided.")
#         raise ValueError("The filename must contain the name of the source in the data. " \
#                          "Otherwise 'calsour' must be provided.")
#
#     uvsplit = AIPSUVData(uvimage.name, "SPLIT", 1, 1)
#     if not uvsplit.exists():
#         raise FileNotFoundError(f"With the given Difmap FITS image, a SPLIT for {uvimage.name} " \
#                                 "was expected but not found.")
#
#     while AIPSUVData(uvimage.name, "SPLIT", 1, uvsplit.seq + 1).exists():
#         uvsplit = AIPSUVData(uvimage.name, "SPLIT", 1, uvsplit.seq + 1)
#
#     _ = calib(uvsplit, in2data=uvimage, refant=refant, calsour=[uvimage.name], solmode=solmode,
#               solint=solint, soltype=soltype, **kwargs)
#     if uvimage.name in phaseref:
#         for a_target in target:
#             target_split = AIPSUVData(a_target, "SPLIT", 1, 1)
#             if not target_split.exists():
#                 raise FileNotFoundError(f"A SPLIT for {a_target} could not be found.")
#
#             while AIPSUVData(a_target, "SPLIT", 1, target_split.seq + 1).exists():
#                 target_split = AIPSUVData(a_target, "SPLIT", 1, target_split.seq + 1)
#
#             highest_seq = target_split.seq + 1
#             target_split = multi(uvdata=target_split)
#             sn2copy = highest_table(uvsplit, 'SN')
#             sn2create = highest_table(target_split, 'SN') + 1
#             tacop(uvdata=target_split, fromuvdata=uvsplit, inext='SN', inver=sn2copy, ncount=1,
#                   outver=sn2create)
#             clcal(target_split, calsour=[''], refant=refant, snver=sn2create, gainver=1,
#                   gainuse=2, opcode='CALI', interpol='ambg')
#             split(target_split, sources=[''], gainuse=highest_table(target_split, "CL"),
#                   doband=-1, aparm=[2, 0, 0, 1])
#             new_split = AIPSUVData(a_target, "SPLIT", 1, target_split.seq + 1)
#             if not new_split.exists():
#                 # It may have happened that AIPS cut the source name to 8 characters
#                 tmp_split = AIPSUVData(a_target[:8], "SPLIT", 1, 1)
#                 while tmp_split.exists():
#                     tmp_split = AIPSUVData(a_target[:8], "SPLIT", 1, tmp_split.seq + 1)
#
#                 new_split = AIPSUVData(a_target[:8], "SPLIT", 1, tmp_split.seq - 1)
#                 assert new_split.exists(), f"The SPLIT file for {a_target} should have been " \
#                                            "created in AIPS but is not found."
#                 new_split.rename(name=a_target, klass="SPLIT", seq=highest_seq + 1)
#
#             outfile, niter = f"{projectname}.{a_target}.SPLIT.selfcalNN.UVFITS", 1
#             while Path(outfile.replace('NN', str(niter))).exists():
#                 niter += 1
#
#             fittp(new_split, f"PWD:{outfile.replace('NN', str(niter))}")


def selfcalibration(projectname: str, refant: list, imagefile: str, calsour: str,
                    sources: list[str], solmode: str = 'a&p', solint: float = 0.0,
                    soltype: str = 'L1R', weightit: int = 1, normaliz: int = 1,
                    minsnr: int = 3, avgpol: bool = False, avgspw: bool = False, minant: int = 4):
    """Runs a cycle of self-calibration on the UVDATA with the specified parameters.
    """
    uvdata: AIPSUVData = AIPSUVData(projectname.upper(), "UVDATA", 1, 1)
    if not uvdata.exists():
        raise FileNotFoundError(f"The UVDATA for {projectname} was expected but not found.")

    # Guesses the source name from the image file to import
    uvimage: AIPSImage = AIPSImage(calsour, 'PICLN', 1, 1)
    if not uvimage.exists():
        # fixing the 'pa' stokes from Difmap
        fix_difmap_image(imagefile)
        uvimage = fitld(calsour, datain=imagefile, doconcat=-1, outclass='PICLN',
                        replace=True, ncount=1, digicor=-1, optype='IM')
        # if uvimage.header['crval'][uvimage.header['ctype'].index('STOKES')] == -9:
        #     new_crval = uvimage.header['crval']
        #     new_crval[uvimage.header['ctype'].index('STOKES')] = 1
        #     uvimage.header.update({'crval', new_crval})
    cltable: int = highest_table(uvdata, 'CL')
    bptable: int = highest_table(uvdata, 'BP')
    sntable: int = highest_table(uvdata, 'SN')

    calib(uvdata, in2data=uvimage, refant=refant, calsour=[calsour], solmode=solmode, docalib=1,
          solint=solint, soltype=soltype, gainuse=cltable, snver=sntable + 1, weightit=weightit,
          aparm=[minant, 0, 1 if avgpol else 0, 0, 1 if avgspw else 0, 3, minsnr],
          normaliz=normaliz, cparm=[0, 1, 0], doband=1 if bptable > 0 else -1, bpver=bptable)
    clcal(uvdata, calsour=[calsour], sources=[calsour], refant=refant, snver=sntable + 1,
          gainver=cltable, gainuse=cltable + 1, opcode='CALI', interpol='self')
    clcal(uvdata, calsour=[calsour], sources=[s for s in sources if s != calsour],
          refant=refant, snver=sntable + 1, gainver=cltable, gainuse=cltable + 1, opcode='CALI',
          interpol='ambg', samptype='EXP')


def split_sources(projectname: str, sources: Optional[list[str]] = None, bchan: int = 0,
                  echan: int = 0, avgchan: int = 0, avgtime: int = 0, replace: bool = True):
    """Splits the sources with the highest calibration table available.
    If replace, then it will remove the highest SPLIT files, otherwise creates new splits.
    """
    uvdata: AIPSUVData = AIPSUVData(projectname.upper(), "UVDATA", 1, 1)
    splits = {}

    for a_src in sources if sources is not None else uvdata.sources:
        if replace:
            temp_uvdata = AIPSUVData(a_src, "SPLIT", 1, 1)
            while temp_uvdata.exists():
                temp_uvdata.zap(force=True)
                temp_uvdata = AIPSUVData(a_src, "SPLIT", 1, temp_uvdata.seq + 1)

            splits[a_src] = AIPSUVData(a_src, "SPLIT", 1, 1)
        else:
            splits[a_src] = AIPSUVData(a_src, "SPLIT", 1, 1)
            while splits[a_src].exists():
                splits[a_src] = AIPSUVData(a_src, "SPLIT", 1, splits[a_src].seq + 1)

    split(uvdata, sources=sources if sources is not None else uvdata.sources,
          bchan=bchan, echan=echan, docal=1, gainuse=highest_table(uvdata, 'CL'),
          doband=1 if highest_table(uvdata, 'BP') > 0 else -1,
          bpver=highest_table(uvdata, 'BP'),
          aparm=[2 if avgchan == -1 else 1, avgtime, 0, 1, 0, 1])

    for a_src, a_split in splits.items():
        if a_split.exists():
            outfile = Path(f"{projectname}.{a_src}.SPLIT.{a_split.seq}.UVFITS")
            outfile.unlink(missing_ok=True)
            fittp(a_split, str(outfile))
        else:
            rprint(f"[yellow]A SPLIT file for {a_src} was not generated.[/yellow]")


def transfercal(projectname: str, uvfile: str, source: list, refant: list,
                ncount: int = 1, avgchan_split: bool = True):
    """TODO: THIS IS OLD CODE, NEEDS TO BE UPDATED TO MATCH THE OTHER FUNCTIONS
    """
    oriuvdata = AIPSUVData(projectname, "UVDATA", 1, 1)
    newuvdata = AIPSUVData(projectname, "UVDATA", 1, 2)
    while newuvdata.exists():
        newuvdata = AIPSUVData(projectname, "UVDATA", 1, newuvdata.seq + 1)

    fitld(newuvdata.name, datain=f"PWD:{uvfile}", ncount=ncount, doconcat=1,
          outclass=newuvdata.klass, outseq=newuvdata.seq, digicor=-1, douvcomp=1)

    # the first oriuvdata will only have 1 FG and 2,3 CL
    for n_cl in range(2, highest_table(oriuvdata, 'CL')+1):
        tacop(newuvdata, fromuvdata=oriuvdata, inext='CL', inver=n_cl, ncount=1, outver=n_cl)

    for n_sn in range(1, highest_table(oriuvdata, 'SN')+1):
        tacop(newuvdata, fromuvdata=oriuvdata, inext='SN', inver=n_sn, ncount=1, outver=n_sn)

    for n_fg in range(1, highest_table(oriuvdata, 'FG')+1):
        tacop(newuvdata, fromuvdata=oriuvdata, inext='FG', inver=n_fg, ncount=1, outver=n_fg)

    for n_bp in range(1, highest_table(oriuvdata, 'BP')+1):
        tacop(newuvdata, fromuvdata=oriuvdata, inext='BP', inver=n_bp, ncount=1, outver=n_bp)

    # sanity check because of AIPS SPLIT issues with src names larger than 8
    split_sources = copy.deepcopy(source)
    split_sources += [src[:8] for src in split_sources if len(src) > 8]
    split(newuvdata, sources=split_sources, gainuse=highest_table(newuvdata, "CL"),
          doband=1, bpver=highest_table(newuvdata, "BP"), aparm=[2 if avgchan_split else 0, 0, 0, 1])

    uvsplits = {}
    for a_source in source:
        n_seq = 1
        str_calibration = 'cal'
        if AIPSUVData(a_source[:8], "SPLIT", 1, 1).exists():
            n_seq_cut = 1
            while AIPSUVData(a_source[:8], "SPLIT", 1, n_seq_cut).exists():
                    # the first oriuvdata will only have 1 FG and 2,3 CL
                n_seq_cut += 1

            split_src = AIPSUVData(a_source[:8], "SPLIT", 1, n_seq_cut - 1)
            while AIPSUVData(a_source, "SPLIT", 1, n_seq).exists():
                n_seq += 1

            split_src.rename(name=a_source, klass="SPLIT", seq=n_seq)
            uvsplits[a_source] = split_src
            selfcal_split = AIPSUVData(a_source, "SPLIT", 1, n_seq - 1)
        else:
            while AIPSUVData(a_source, "SPLIT", 1, n_seq).exists():
                n_seq += 1

            if n_seq == 1:
                raise FileNotFoundError(f"A SPLIT file for {a_source} was expected but not found.")

            uvsplits[a_source] = AIPSUVData(a_source, "SPLIT", 1, n_seq - 1)
            selfcal_split = AIPSUVData(a_source, "SPLIT", 1, n_seq - 2)

        if highest_table(selfcal_split, inext='SN') > 0:
            # Also transfer the selfcal solutions
            tacop(uvsplits[a_source], fromuvdata=selfcal_split, inext='SN',
                  inver=highest_table(selfcal_split, 'SN'), ncount=1, outver=1)

            target_split = multi(uvdata=uvsplits[a_source])
            clcal(target_split, calsour=[''], refant=refant, snver=1, gainver=1,
                  gainuse=2, opcode='CALI', interpol='ambg')
            split(target_split, sources=[''], gainuse=highest_table(target_split, "CL"),
                  doband=-1, aparm=[2, 0, 0, 1])
            uvsplits[a_source] = AIPSUVData(a_source, "SPLIT", 1, n_seq)
            str_calibration = 'selfcal'

        outfile, niter = f"{projectname}.{a_source}.SPLIT.{str_calibration}NN.UVFITS", 1
        while Path(outfile.replace('NN', str(niter))).exists():
            niter += 1

        fittp(uvsplits[a_source], f"PWD:{outfile.replace('NN', str(niter))}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs a ParselTongue AIPS calibration.",
                                     prog="parseltongue.py", usage="%(prog)s [-h] [options]")
    parser.add_argument('projectname', type=str, help="Codename of the project.")
    parser.add_argument('--aipsid', type=int, default=None, help="AIPS ID to use for the processing. " \
                        "It not given, taken from the project code.")
    parser.add_argument('--disk', type=int, default=1, help="AIPS disk to use for the processing. ")
    parser.add_argument('--version', action="version", version=__version__)
    parser.add_argument('--log', type=str, default='parseltongue.log', help="Logging file.")
    parser.add_argument('--refant', type=str, default='',
                        help="Reference antennas (comma-separated list).")
    parser.add_argument('--username', type=str, default=None,
                        help="'username' required to download the data, if needed.")
    parser.add_argument('--password', type=str, default=None,
                        help="'password' required to download the data, if needed.")
    parser.add_argument('--obsdate', type=str, default=None,
                        help="Epoch of observation in YYMMDD format. Only needed if the data are downloaded.")
    parser.add_argument('--bchan', type=int, default=0, help="BCHAN parameter for all calibration and split.")
    parser.add_argument('--echan', type=int, default=0, help="ECHAN parameter for all calibration and split.")
    parser.add_argument('--fchan', type=float, default=None,
                        help="Sets BCHAN/ECHAN parameters to led out the fraction " \
                        "of edge channels given by this (e.g. 0.1 will flag 10% of the edge channels.")
    parser.add_argument('--minsnr', type=int, default=3,
                        help="Minimum SNR during the global fringe and/or self-calibration.")
    parser.add_argument('--replace', default=False, action="store_true",
                        help="If already exists, it will overwrite the existing files in AIPS.")
    parser.add_argument('--quack', type=str, default=None, help="Antennas and time (in seconds) to flag " \
            "at the beginning of each scan, in the form '{ant1}:{int1};{ant2}:{int2}': list of pairs " \
            "of antenna and time to flag, separated by ';'.")

    initial = parser.add_argument_group("Import and a-priori calibration")
    initial.add_argument('-i', '--initial', default=False, action="store_true",
                        help="Imports the data and runs the a-priori calibration.")
    initial.add_argument('--fitsidi', type=str, default=None, help="FITS-IDI root names to be " \
                         "imported (following AIPS naming as used in FITLD).")
    initial.add_argument('--uvfits', type=str, default=None, help="UVFITS to be imported.")

    pars_calib = parser.add_argument_group("Main calibration (fringe, bandpass)")
    pars_calib.add_argument('-c', '--calib', default=False, action="store_true",
                        help="Runs the main calibration.")
    pars_calib.add_argument('--solint', type=float, default=5, help="solint for the global fringe.")
    pars_calib.add_argument('--target', type=str, default=None,
                        help="comma-separated list of target sources")
    pars_calib.add_argument('--phaseref', type=str, default=None,
                        help="comma-separated list of phase-referencing sources")
    pars_calib.add_argument('--bpsour', type=str, default=None,
                        help="comma-separated list of fringe-finder or bandpass sources")
    pars_calib.add_argument('--sbdtime', type=str, default=None, help="Time in AIPS format " \
                        "(comma separated, no spaces) for the time range to use for the " \
                        "intrumental delay correction.")
    pars_calib.add_argument('--iono', default=False, action="store_true",
                        help="Runs the ionospheric corrections.")

    pars_export = parser.add_argument_group("Exporting to SPLIT files and UVFITS")
    pars_export.add_argument('-e', '--export', default=False, action="store_true",
                        help="Sports to SPLIT and UVFITS after each mean calibration step.")
    pars_export.add_argument('--avgchan', default=False, action="store_true",
                        help="Averages the data when splitting into a single-channel per IF.")

    pars_sc = parser.add_argument_group("Self-calibrate the data")
    pars_sc.add_argument('-s', '--selfcal', type=str, default=None,
                        help="Run a self-calibration using the given Difmap FITS image file.")
    pars_sc.add_argument('--sc_calsour', type=str, default=None,
                        help="Name of the source to use as model and to calibrate the data.")
    pars_sc.add_argument('--sc_target', type=str, default=None,
                        help="Sources to transfer the calibration solutions computed during self-calibration.")
    pars_sc.add_argument('--sc_avgpol', default=False, action="store_true",
                        help="Averages the two polarizations to compute the self-calibration.")
    pars_sc.add_argument('--sc_solmode', type=str, default='A&P:5',
                        help="Solution mode for the self-calibration (if selected): 'P' only phase,." \
                             "'A&P' amplitude and phase, and integration time (both values separated by ':'. " \
                             "If multiple runs of selfcal are desired, then it can be a ';' separated list.")
    # pars_sc.add_argument('--transfer', type=str, default=None,
    #                     help="Transfers the calibration to a new data set and splits the target "
    #                          "sources. Useful e.g. in multiple-pass observations (with e.g. "
    #                          "exp_1_1.IDI* and exp_2_1.IDI."
    #                          "It needs to be the AIPS name to import the file as provided to FITLD")

    args = parser.parse_args()
    logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logging.info(f"--- Starting ParselTongue reduction on {dt.datetime.now().strftime('%d %b %Y  %H:%M')}\n\n")

    if args.aipsid is None:
        args.aipsid = aipsno_from_project(args.projectname)

    assert (args.aipsid >= 100) and (args.aipsid < 100000)
    AIPS.userno = args.aipsid
    assert len(args.projectname) <= 6

    # Are the data already available?
    if glob.glob(f"{args.projectname.lower()}*IDI*") == 0 and \
            (glob.glob(f"{args.fitsidi}*") == 0 if args.fitsidi is not None else True) and \
            (glob.glob(args.uvfits) == 0 if args.uvfits is not None else True):
        if args.obsdate is None:
            rprint("\n\n[bold red]No FITS-IDI data found. To download the data --obsdate must be set[/bold red]")
            sys.exit(1)

        rprint("\n[bold]No FITS-IDI found. Procedding to download the data[bold]")
        download_evn(args.projectname, args.obsdate, args.username, args.password)


    if args.initial:
        fitsidifiles = args.fitsidi if (args.fitsidi, args.uvfits).count(None) < 2 \
                       else f"{args.projectname.lower()}_1_1.IDI"
        importdata(args.projectname, args.disk, args.replace, fitsidifiles, args.uvfits)

    uvdata: AIPSUVData = AIPSUVData(args.projectname.upper(), "UVDATA", 1, 1)
    if not uvdata.exists():
        rprint(f"\n\n[bold red]The UVDATA should have been imported at this stage[/bold red]")
        sys.exit(1)

    if args.quack is not None:
        assert ':' in args.quack, "Wrong format. It needs to be 'ant1:solint_s;ant2:solint_s;...'"
        for quack_pair in args.quack.split(';'):
            quack_ant, quack_time = quack_pair.split(':')
            quack(uvdata, antennas=[quack_ant], begcut_min=float(quack_time)/60.0)

    nchans = uvdata.header['naxis'][uvdata.header['ctype'].index('FREQ')]
    if args.fchan is not None:
        args.bchan = int(np.floor(nchans*args.fchan))
        args.echan = int(np.ceil(nchans*(1 - args.fchan)))

    if (args.bchan > 0) and (args.echan > 0):
        uvflg(uvdata, bchan=0, echan=args.bchan)
        uvflg(uvdata, bchan=args.echan, echan=nchans)

    if args.calib:
        assert args.target is not None, "The target source(s) must be provided."
        assert args.bpsour is not None, "The bandpass (or aka fringe-finder) source(s) must be provided."
        main_calibration(args.projectname, refant=args.refant.split(','), solint=args.solint,
                         target=args.target.split(','),
                         bpsour=args.bpsour.split(','),
                         phaseref=None if args.phaseref is None else args.phaseref.split(','),
                         sbd_timerange=None if args.sbdtime is None else [int(t) for t in args.sbdtime.split(',')],
                         model=None, bchan=args.bchan, echan=args.echan, tecor=args.iono)

    if args.export:
        split_sources(args.projectname, sources=args.target.split(',') + args.bpsour.split(',') + \
                      (args.phaseref.split(',') if args.phaseref is not None else []), bchan=args.bchan,
                      echan=args.echan, avgchan=-1 if args.avgchan else 0, replace=True)

    if args.selfcal is not None:
        first_amp_sc = True
        for sc_loop in args.sc_solmode.split(';'):
            sc_mode, sc_int = sc_loop.split(':')
            selfcalibration(args.projectname, refant=args.refant.split(','), imagefile=args.selfcal,
                            calsour=args.sc_calsour,
                            sources=[args.sc_calsour] + args.sc_target.split(',') \
                                    if args.sc_target is not None else [],
                            solmode=sc_mode.upper(), solint=float(sc_int),
                            minsnr=args.minsnr, avgpol=args.sc_avgpol, minant=2,
                            normaliz=1 if first_amp_sc and ('a' in sc_mode.lower()) else 0)

            if 'a' in sc_mode.lower():
                first_amp_sc = False

    if args.export:
        split_sources(args.projectname, sources=[args.sc_calsour] + args.sc_target.split(',') \
                                    if args.sc_target is not None else [], bchan=args.bchan,
                      echan=args.echan, avgchan=-1 if args.avgchan else 0, replace=False)

    rprint(f"\n\n[green]AIPS-ParselTongue appears to have ended successfully[/green]")
