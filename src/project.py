#!/usr/bin/env python3.11
"""Defines a project that specifies an interferometric radio observation.
This module defines all usual metadata and formats that are spected for such data
and the actions that the user may expect to perform on it.

A project is expected to be related to some observations carried out by e.g. EVN, GMRT,...
and defines the set of tasks required during a normal data reduction as calibration, plotting.
"""
from __future__ import annotations
import os
import copy
import time
import pickle
import logging
from collections import defaultdict
import datetime as dt
from pathlib import Path
from typing import Optional, Union, Iterable #, Tuple, NoReturn, List, Tuple
# import blessed
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import yaml
import numpy as np
from astropy import units as u
from astropy import coordinates as coord
from astropy.io import fits
from rich import print as rprint
from rich import progress
import casatasks
from casatools import msmetadata as msmd
from casatools import table as tb
from . import tools
from .obsdata import ObsEpoch as ObsEpoch
from .obsdata import Stokes as Stokes
from .obsdata import Source as Source
from .obsdata import Sources as Sources
from .obsdata import SourceType as SourceType
from .obsdata import Antenna as Antenna
from .obsdata import Antennas as Antennas
from .obsdata import FreqSetup as FreqSetup
from .project import Project as Project
from .evn import calibration as evn_calibration
from .flagging import Flagging
from .plotting import Plotting
from .imaging import Imaging
# # TOOD: same for GMRT



_SCI_PACKAGE = ('CASA', 'AIPS')
_OBSERVATORIES = ('EVN', 'LBA', 'GMRT')


class Project(object):
    """Defines a radio observation with all relevant metadata.
    """
    @property
    def projectname(self) -> str:
        """Name of the project associated to the observation.
        """
        return self._projectname

    @property
    def cwd(self) -> Path:
        """Returns the working directory for the project.
        """
        return self._cwd

    @property
    def observatory(self) -> str:
        """Returns the name of the observatory that conducted the observations.
        """
        return self._observatory

    @property
    def sci_package(self) -> str:
        """Returns the name of the software package that will be used to conduct the main
        calibration of the data (either CASA or AIPS).
        """
        return self._scipackage

    @property
    def msfile(self) -> Path:
        """Returns the name of the MS file (pathlib.Path object) associated to this correlator pass.
        """
        return self._msfile

    @property
    def uvfitsfile(self) -> Union[Path, None]:
        """Returns the UV FITS file associated to this observation, if any.
        """
        return self._uvfitsfile

    # @uvfitsfile.setter
    # def uvfitsfile(self, new_uvfitsfile: Union[Path, str]):
    #     if isinstance(new_uvfitsfile, Path):
    #         self._uvfitsfile = new_uvfitsfile
    #     elif isinstance(new_uvfitsfile, str):
    #         self._uvfitsfile = Path(new_uvfitsfile)
    #     else:
    #         TypeError(f"uvfitsfile() expected a str or Path type. Found {new_uvfitsfile} "
    #                   f"({type(new_uvfitsfile)}).")

    @property
    def time(self) -> ObsEpoch:
        """Returns an ObsEpoch object containing the times associated to the observation.
        """
        return self._time

    @property
    def antennas(self) -> Antennas:
        """Returns an Antennas object containing all antennas related to the given observation.
        """
        return self._antennas

    @property
    def refant(self) -> list:
        """Returns a sorted list of the antenna names to be used as reference in calibration.
        """
        return copy.deepcopy(self._refant)

    @refant.setter
    def refant(self, antennas: Union[str, list]):
        """Defines the antenna to use as reference during calibration.
        It can be either a str (with a single antenna, or comma-separated antennas),
        or a list with the antenna(s) to use as reference.
        """
        if isinstance(antennas, str):
            self._refant = [ant.strip() for ant in antennas.split(',')]
        elif isinstance(antennas, list):
            self._refant = copy.deepcopy(antennas)

    @property
    def freqsetup(self) -> Union[FreqSetup, None]:
        """Returns a FreqSetup object (if defined) containing the frequency information of
        the observation data.
        """
        return self._freqsetup

    @property
    def sources(self) -> Sources:
        """Returns a Sources object containing the information of all sources scheduled
        in the observation.
        """
        return self._sources

    @property
    def logdir(self) -> Path:
        """Returns the directory where the log outputs from the process are stored.
        """
        return self._logdir

    @property
    def caldir(self) -> Path:
        """Returns the directory where the calibration table files are stored.
        """
        return self._caldir

    @property
    def outdir(self) -> Path:
        """Returns the output directory where images, plots and other final products are stored.
        """
        return self._outdir

    @property
    def params(self) -> dict:
        return self._args

    @params.setter
    def params(self, new_params: dict):
        assert isinstance(new_params, dict), \
               f"{new_params} should be a dict but is {type(new_params)}"
        self._args = new_params

    # @property
    # def last_step(self):
    #     return self._last_step
    #
    # @last_step.setter
    # def last_step(self, last_step):
    #     self._last_step = last_step

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def splits(self) -> dict:
        return self._splits

    @property
    def calibrate(self):
    # def calibrate(self) -> evn_calibration.Calibration:
        #TODO: there will be more possibilities
        # likely to do a parent class called Calibration?
        return self._calibration

    @property
    def flag(self) -> flagging.Flagging:
    # def flag(self):
        return self._flagging

    @property
    def plot(self) -> plotting.Plotting:
    # def plot(self):
        return self._plotting

    @property
    def image(self) -> imaging.Imaging:
    # def image(self):
        return self._imaging

    @property
    def importdata(self) -> obsdata.Importing:
    # def importdata(self):
        #TODO: importing should also be a sub-class (observatory dependend)
        return self._importing


    def __init__(self, projectname: str, observatory: str = 'EVN', sci_package: str = 'CASA',
                 params: Optional[Union[dict, str, Path]] = None,
                 cwd: Optional[Union[Path, str]] = None, logging_level=logging.INFO):
        """Initializes a project related to a single radio observation.

        Inputs:
            projectname : str
               The name of the project (case insensitive).

            observatory : str  [default = '']
                Name of the observatory associated to this project.
                If not specified, it will be read from the MS when exists.

            sci_package: str  [default = 'CASA']
                The scientific package to be used for the main calibration tasks.
                It can be either 'CASA' or 'AIPS'. Note that the later is only available for VLBI
                data and it will use AIPS only for the a-priori gain calibration, ionospheric
                corrections if needed, instrumental delay and multi-band fringe, and bandpass.

            params : dict/Path/str  [optional]
                Default parameters read from the pipeline input file and are loaded here.
                If it is a Path or str, it is assumed to be the file name containing the
                input parameters.

            cwd : Path or str  [optional. Default = $PWD]
                The default directory where the pipeline will run, search for and create the files.
        """
        print(f"projectname: '{projectname}'")
        if (projectname == '') or (not isinstance(projectname, str)):
               raise ValueError("The project name needs to be a non-empty string.")

        self._projectname = projectname

        if observatory.upper().strip() not in _OBSERVATORIES:
            rprint(f"\n[bold red]The requested observatory ({observatory}) "
                   "is not supported[/bold red]")
            rprint(f"[red]Only {', '.join(_OBSERVATORIES)} are accepted.[/red]")
            raise ValueError("The requested observatory is not implemented.")

        self._observatory = observatory.upper().strip()

        self._logging_level = logging_level
        logging.basicConfig(level=logging_level, format="%(asctime)s %(levelname)s:\n%(message)s",
                            datefmt="%Y-%m-%d %H:%M")
        self._logger = logging.getLogger(self.projectname)
        self.logger.debug(f"Initializing project '{self.projectname}'.")


        if sci_package.upper().strip() not in _SCI_PACKAGE:
            raise ValueError(f"The sci_package {sci_package} is not known. " \
                       f"Only {', '.join(_SCI_PACKAGE)} are available.")

        self._scipackage = sci_package
        self.logger.debug(f"The package {sci_package} will be used for data reduction.")
        if (self._scipackage == 'AIPS') and (not tools.aips_exists()):
            rprint("\n[bold red]AIPS is set to be used, but no AIPS environment found[/bold red]")
            rprint("[red]Did you run the AIPS LOGIN.SH (or CSH) script?[/red]")
            raise OSError("No AIPS environment is found but AIPS will be used.")

        if cwd is None:
            self._cwd = Path('./')
        elif isinstance(cwd, str):
            self._cwd = Path(cwd)
        elif isinstance(cwd, Path):
            self._cwd = cwd
        else:
            raise TypeError(f"The working directory ({cwd}) should be either None, Path or str.")

        if params is not None:
            if isinstance(params, dict):
                self._args = params
            elif isinstance(params, Path) or isinstance(params, str):
                with open(params, mode="rt", encoding="utf-8") as fp:
                    self._args = yaml.safe_load(fp)

                self.logger.debug(f"Parameters successfully read from the file {params}.")
        else:
            self._args = {}

        if 'epoch' in self.params:
            self._time = ObsEpoch(dt.datetime.strptime(str(self.params['epoch']), "%y%m%d"))
        else:
            self._time = ObsEpoch(None)

        self._msfile = self.cwd / Path(f"{self.projectname}.ms")
        self._uvfitsfile = self.cwd / self.msfile.name.replace('.ms', '.uvfits')
        # if "logdir" not in params:
        self._logdir = self.cwd / 'log'
        self._caldir = self.cwd / 'caltables'
        self._outdir = self.cwd / 'results'
        self._sources = Sources()
        self._antennas = Antennas()
        self._splits: dict[str, list] = defaultdict(list)
        self._freqsetup = None
        self._last_step = None

        if 'sources' in self._args:
            for a_src in self._args['sources']['target']:
                self._sources.add(Source(name=a_src, sourcetype=SourceType.target,
                                              coordinates=None))

            for a_src in self._args['sources']['phaseref']:
                self._sources.add(Source(name=a_src, sourcetype=SourceType.calibrator,
                                              coordinates=None))

            for a_src in self._args['sources']['fringefinder']:
                self._sources.add(Source(name=a_src, sourcetype=SourceType.fringefinder,
                                              coordinates=None))

        for a_dir in (self.cwd, self.logdir, self.caldir, self.outdir):
            a_dir.mkdir(exist_ok=True)

        assert isinstance(sci_package, str) and sci_package.upper().strip() in _SCI_PACKAGE, \
               "Unexpected value of scipackage ({package})."
        self._scipackage = sci_package.upper().strip()

        #TODO: use json for this. Load and save as its own functions
        self._local_copy = self.cwd / Path(f"{self.projectname.lower()}.obj")
        # if self.exists_local_copy():
        #     self.load(self._local_copy)

        self.refant = self._args['reference_antenna'] if 'reference_antenna' in self._args else []

        if 'observatory' not in self._args:
            self._args['observatory'] = observatory

        if observatory.upper().strip() not in _OBSERVATORIES:
            ValueError(f"The observatory {observatory} is not known. " \
                       f"Only {', '.join(_OBSERVATORIES)} are available.")

        self._observatory = observatory.upper().strip()
        self.logger.debug(f"Data from observatory {observatory}.")

        # TODO: differentiate as function of the observatory
        if self.observatory.lower() == 'evn':
            self._calibration = evn_calibration.Calibration(self, self.caldir)
        elif (self.observatory.lower() == 'gmrt'):
            raise NotImplementedError("Data reduction for GMRT has not been implemented yet.")
        else:
            raise NotImplementedError(f"Data reduction for {self.observatory} " \
                                      "has not been implemented yet.")

        self._importing = obsdata.Importing(self)
        self._flagging = flagging.Flagging(self)
        self._plotting = plotting.Plotting(self)
        self._imaging = imaging.Imaging(self)
        # # self._last_step = None
        if self.msfile.exists():
            self.get_metadata_from_ms()
        # self.store()


    def get_metadata_from_ms(self, chunks: int = 100):
        """Recovers all useful metadata from the MS file and stores it in the object.

        Inputs
        ------
            - chunks : int  (default = 100)
              When reading through the MS, it reads it in chunks of the specified amount.
              A value of 100 has been observed to provided the fastest I/O times.

        -- May Raise
        It may raise KeyError if the type of sources (target, phaseref, fringefinder)
        are not specified in the self.params dict.
        """
        m = msmd(str(self.msfile))
        if not m.open(str(self.msfile)):
            return ValueError(f"The MS file {self.msfile} could not be openned.")

        try:
            self._antennas = Antennas()
            antenna_names = m.antennanames()
            for ant_name in antenna_names:
                ant = Antenna(name=ant_name, observed=False)
                self.antennas.add(ant)

            spw_names = range(m.nspw())
            telescope_name = m.observatorynames()[0]
            if self.observatory == '':
                self._observatory_name = telescope_name
            elif self.observatory != telescope_name:
                rprint("[yellow]WARNING: the observatory name in MS does not match the one "
                       f"provided in the project ({self.observatory} vs {telescope_name}).[/yellow]")

            self._sources = Sources()
            src_names = m.fieldnames()
            src_coords = [m.phasecenter(s) for s in range(len(src_names)) ]
            for a_name, a_coord in zip(src_names, src_coords):
                try:
                    if a_name in self.params['sources']['target']:
                        a_type = SourceType.target
                    elif a_name in self.params['sources']['phaseref']:
                        a_type = SourceType.calibrator
                    elif a_name in self.params['sources']['fringefinder']:
                        a_type = SourceType.fringefinder
                    else:
                        a_type = SourceType.other
                except KeyError:
                    rprint("[bold yellow]-- No source type information has been " \
                           "found --[bold yellow]")
                    rprint("You better define manually or through the inputs which sources " \
                           "are target/calibrator/etc.")
                    a_type = SourceType.other

                self.sources.append(Source(a_name, a_type,
                                                coord.SkyCoord(ra=a_coord['m0']['value'],
                                                               dec=a_coord['m1']['value'],
                                                               unit=(a_coord['m0']['unit'],
                                                                     a_coord['m1']['unit']),
                                                               equinox=a_coord['refer'])))

            timerange = m.timerangeforobs(0)
            self._time = ObsEpoch(
                    start_datetime=tools.mjd2date(timerange['begin']['m0']['value']),
                    end_datetime=tools.mjd2date(timerange['end']['m0']['value']))

            mean_freq = (m.meanfreq(spw_names[-1]) + m.meanfreq(0)) / 2.0
            self._freqsetup = FreqSetup(m.nchan(0), m.nspw(), mean_freq,
                                             m.bandwidths()[0])

            nrows = int(m.nrows())

            # To be able  to get the parallel hands, either circular or linear
            corr_order = [Stokes(i) for i in m.corrtypesforpol(0)]
            corr_pos = []
            try:
                corr_pos.append(corr_order.index(Stokes.RR))
                corr_pos.append(corr_order.index(Stokes.LL))
            except ValueError:
                try:
                    corr_pos.append(corr_order.index(Stokes.XX))
                    corr_pos.append(corr_order.index(Stokes.YY))
                except ValueError:
                    rprint("[bold red]The associated MS does not have neither circular nor " \
                           "linear-based polarization information[/bold red]")
        finally:
            m.close()


        ants_observed = set()
        scans = self.scan_numbers()
        scans_ants = self.antennas_in_scans(scans)
        for scan in scans_ants:
            for ant in scan:
                ants_observed.add(ant)

        for ant in ants_observed:
            self.antennas[ant].observed = True

        # Now to get which subbands each antenna observed
        m = tb(str(self.msfile))
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            ant_subband = defaultdict(set)
            rprint('\n[bold]Reading the MS to find which antennas actually observed...[/bold]')
            start_time = time.time()
            with progress.Progress() as progress_bar:
                task = progress_bar.add_task("[yellow]Reading MS...", total=nrows)
                for (start, nrow) in tools.chunkert(0, nrows, chunks):
                    ants1 = m.getcol('ANTENNA1', startrow=start, nrow=nrow)
                    ants2 = m.getcol('ANTENNA2', startrow=start, nrow=nrow)
                    spws = m.getcol('DATA_DESC_ID', startrow=start, nrow=nrow)
                    msdata = m.getcol('DATA', startrow=start, nrow=nrow)

                    for ant_i,antenna_name in enumerate(antenna_names):
                        for spw in spw_names:
                            cond = np.where(((ants1 == ant_i) | (ants2 == ant_i)) & (spws == spw))
                            if not (abs(msdata[corr_pos][:, :, cond[0]]) < 1e-5).all():
                                ant_subband[antenna_name].add(spw)
                            # testing a much faster check...  But it picks everything
                            # if len(cond[0]) > 0:
                            #     ant_subband[antenna_name].add(spw)

                    progress_bar.update(task, advance=nrow)
        finally:
            m.close()

        rprint(f"[green]Total time elapsed: {(time.time()-start_time)/60:.2f} min.[/green]")
        for antenna_name in self.antennas.names:
            self.antennas[antenna_name].subbands = tuple(ant_subband[antenna_name])
            # this is the same as two segments before, but in case one antenna just sent zero data
            self.antennas[antenna_name].observed = len(self.antennas[antenna_name].subbands) > 0

        self.listobs()
        # self.summary()


    def scan_numbers(self) -> np.ndarray:
        """Returns the list of scans that were observed in the observation.
        """
        m = msmd(str(self.msfile))
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            scannumbers = m.scannumbers()
        finally:
            m.close()

        return scannumbers


    def antennas_in_scan(self, scanno: int) -> list[str]:
        """Returns a list with all antennas that observed the given scan.
        """
        m = msmd(str(self.msfile))
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            antenna_names = m.antennanames()
            return [antenna_names[a] for a in m.antennasforscan(scanno)]
        finally:
            m.close()


    def antennas_in_scans(self, scanno: Union[list[int], np.ndarray]) -> list[list[str]]:
        """Returns a list with all antennas that observed each given scan.
        """
        m = msmd(str(self.msfile))
        scan_list = []
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            antenna_names = m.antennanames()
            for scan in scanno:
                scan_list.append([antenna_names[ant] for ant in m.antennasforscan(scan)])

            return scan_list
        finally:
            m.close()


    def scans_in_antennas(self) -> dict[str, list[int]]:
        """Returns a dictionary with all antennas that observed as keys and a list of all scans
        that they observed as values.
        """
        all_scans = self.scan_numbers()
        scan_list = self.antennas_in_scans(all_scans)
        ant_list = {ant: [] for ant in self.antennas.names}
        for ant in ant_list:
            for i,scan  in enumerate(scan_list):
                if ant in scan:
                    ant_list[ant].append(all_scans[i])

        return ant_list


    def scans_with_source(self, source_name: str) -> np.ndarray:
        """Returns a list with the numbers of the scans where the given source was observed.
        """
        m = msmd(str(self.msfile))
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            return m.scansforfield(source_name)
        finally:
            m.close()


    def times_for_scans(self, scanno: Union[list[int], np.ndarray]) -> np.ndarray:
        """Returns the time for the specified scan number.
        """
        if len(scanno) == 0:
            raise ValueError("There should be at least one scan to check in the list.")

        m = msmd(str(self.msfile))
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            times_casa: np.ndarray = m.timesforscans(scanno)
            return dt.datetime(1858, 11, 17, 0, 0, 2) + times_casa*dt.timedelta(seconds=1)
        finally:
            m.close()


    def times_for_scan(self, scanno: int) -> np.ndarray:
        """Returns the time for the specified scan number.
        """
        m = msmd(str(self.msfile))
        try:
            if not m.open(str(self.msfile)):
                raise ValueError(f"The MS file {self.msfile} could not be openned.")

            time_casa = m.timesforscan(scanno)
            return dt.datetime(1858, 11, 17, 0, 0, 2) + time_casa*dt.timedelta(seconds=1)
        finally:
            m.close()


    def best_scan_from_source(self, source_name: str, verbose: bool = True) -> Optional[tuple]:
        """Returns the scan from the given source where more antennas participated.
        """
        #TODO: in a future iteraction, do statistics on the data to also determine which one
        # has a lower dispersion (thus no RFI, constant amp/[phase?]...
        scans = self.scans_with_source(source_name)
        ants = self.antennas_in_scans(scans)
        max_ants = -1
        best_scan : dict[int, list] = {}
        for scan, ant in zip(scans, ants):
            if len(ant) >= max_ants:
                best_scan[scan] = ant
                max_ants = len(ant)

        if len(best_scan) == 1:
            best_scan_tuple = next(iter(best_scan.items()))
            if verbose:
                rprint(f"\n[bold]The scan {best_scan_tuple[0]} is the best one as " \
                       f"{max_ants} antennas observed it.[/bold]\n")

            return best_scan_tuple
        elif len(best_scan) > 1:
            best_scan_keys = tuple(best_scan.keys())
            best_scan_vals = tuple(best_scan.values())
            if verbose:
                rprint(f"\n[bold]The scans {', '.join((str(s) for s in best_scan_keys))} "
                       f"are the best one as {max_ants} antennas observed it.[/bold]\n")
            return (best_scan_keys, best_scan_vals)
        else:
            rprint("[yellow]No scan found for the given source.[/yellow]")
            return None


    def listobs(self, listfile: Optional[Union[Path, str]] = None, overwrite: bool = True) -> dict:
        listfile = self.outdir / f"{self.projectname}-listobs.log" if listfile is None else listfile
        return casatasks.listobs(self.msfile.name, listfile=str(listfile), overwrite=overwrite)


    def split(self, sources: Optional[Union[Iterable[str], str, None]] = None,
              datacolumn='corrected', keepflags=False, chanbin: int = -1,
              inplace=False, **kwargs) -> Optional[dict[str, Project]]:
        """Splits all the data from all calibrated sources.
        If sources is None, then all sources will be split.

        Params:
            - chanbin : int  (default = -1)
              Width (bin) of input channels to average to for an output channel.
              If -1, then if will create a single-output channel per spw.
              A value of 0 or 1 will not do any averaging.
            - inplace : bool  (default False)
              In case one wants this to be the new MS associated to the current project,
              instead of creating a new one.

        It returns a dict with all split source names as keys. The values are the new Ms objects.
        """
        np.int = int
        np.float = float  # these two is because CASA uses deprecated types in newer numpy ver!

        splits = {}

        if isinstance(sources, str):
            sources = [source.strip() for source in sources.split(',')]

        if sources is not None:
            for source in sources:
                # I don't think it should be so strict.
                # assert source in self.sources, \
                       # f"The passed source {source} is not in the MS {self.msfile}."
                rprint(f"[yellow]Note that the source {source} is not present " \
                       f"in the MS {self.msfile}.[/yellow]")
                self.logger.warning(f"Note that the source {source} is not present " \
                                    f"in the MS {self.msfile}.")
        else:
            sources = self.sources.names

        assert self.freqsetup is not None, \
               "FreqSetup should already be known for the given MS when split."
        if ((chanbin == -1) or (chanbin > 1)) and (self.freqsetup.channels > 1):
            kwargs['chanaverage'] = True
            kwargs['chanbin'] = self.freqsetup.channels  # TODO: it should get the chanbin number if > 1, instead of .channels
        else:
            kwargs['chanaverage'] = False
            kwargs['chanbin'] = 1

        if inplace:
            suffix = 1
            while os.path.isdir(f"{self.projectname}{'' if suffix == 1 else '.'+str(suffix)}.ms"):
                suffix += 1

            ms_name = f"{self.projectname}{'' if suffix == 1 else '.'+str(suffix)}.ms"
            casatasks.mstransform(vis=str(self.msfile), outputvis=ms_name,
                                  keepflags=keepflags, datacolumn=datacolumn, **kwargs)
            self._msfile = Path(ms_name)
            return None
        else:
            suffix = 1
            while any([os.path.isdir(f"{self.projectname}.{a_source}" \
                       f"{'' if suffix == 1 else '.'+str(suffix)}.ms") for a_source in sources]):
                suffix += 1

            for a_source in sources:
                try:
                    ms_name = f"{self.projectname}.{a_source}" \
                              f"{'' if suffix == 1 else '.'+str(suffix)}.ms"
                    casatasks.mstransform(vis=str(self.msfile), outputvis=ms_name,
                                          field=a_source, keepflags=keepflags,
                                          datacolumn=datacolumn, **kwargs)
                    splits[a_source] = Project(ms_name.replace('.ms', ''), cwd=self.cwd,
                                          params=self.params, logging_level=self._logging_level)
                    self.splits[a_source].append(splits[a_source])
                except RuntimeError as e:
                    rprint(f"[bold red]Could not create a split MS file for {a_source}.[/bold red]")
                    rprint(f"[red]{e}[/red]")

            return splits


    def export_uvfits(self, outfitsfilename: Union[str, None] = None, overwrite=True,
                      datacolumn='corrected', combinespw=True, padwithflags=True):
        """Export the available MS file into a UVFITS file, per source.
        """
        if outfitsfilename is None:
            outfitsfilename = str(self.msfile).replace(".ms", ".uvfits")

        casatasks.exportuvfits(vis=str(self.msfile), fitsfile=outfitsfilename,
                               datacolumn=datacolumn, multisource=(len(self.sources) > 1),
                               combinespw=combinespw, padwithflags=padwithflags,
                               overwrite=overwrite)

        # assert self.freqsetup is not None
        # # TODO: check that this does the correct thing
        # fits.setval(outfitsfilename, keyword='OLDRFQ  ', value=f"{self.freqsetup.frequency_range[0].to(u.GHz).value}D+09")
        # if self.observatory.upper() == 'EVN':
        #     fits.setval(outfitsfilename, keyword='CORRELAT', value=f"SFXC")


    def exists_local_copy(self) -> bool:
        """Checks if there is a local copy of the Experiment object stored in a local file.
        """
        return self._local_copy.exists()


    def store(self, path : Optional[Union[Path, str]] = None):
        """Stores the current Experiment into a file in the indicated path. If not provided,
        it will be '.{projectname.lower()}.obj' where exp is the name of the experiment.
        """
        if path is not None:
            self._local_copy = path if isinstance(path, Path) else Path(path)

        with open(self._local_copy, 'wb') as file:
            pickle.dump(self, file)

        self.logger.debug(f"Local copy of {self.projectname} stored at {self._local_copy}.")


    # TODO: make this function static
    def load(self, path: Optional[Union[Path, str]] = None) -> bool:
        """Loads the current Experiment that was stored in a file in the indicated path.
        If path is None, it assumes the standard path of '.{exp}.obj', where exp is the
        name of the experiment.
        """
        if path is not None:
            self._local_copy = path if isinstance(path, Path) else Path(path)

        with open(self._local_copy, 'rb') as file:
            obj = pickle.load(file)

        self._observatory = obj._observatory
        self._logger = obj._logger
        self._args = obj._args
        self._last_step = obj._last_step
        self._ms = obj._ms
        self._calibration = obj._calibration
        self._flagging = obj._flagging
        self._plotting = obj._plotting
        self._imaging = obj._imaging
        self._importing = obj._importing
        self.logger.info(f"Loaded Project object from the stored local copy at {self._local_copy}.")
        return True


    def summary(self, outfile: Union[str, Path, None] = None):
    #     TODO:  this is commented out because I have issues installing blessed within the CASA
    #     environment in my desktop
    #     term = blessed.Terminal(force_styling=True)
        s_file = []
    #     with term.fullscreen(), term.cbreak():
    #         s = term.white_on_red(term.center(term.bold(f"--- {self.projectname} ---")))
        s_file += [f"# {self.projectname.upper()}"]
    #         s += f"{term.normal}\n\n{term.normal}"
    #         s += term.bold_green('General Information\n')
    #         s += ["## General Information"]
        if self.observatory != '':
    #             s += term.bright_black('Observatory: ') + f"{self.observatory}\n"
            s_file += [f"Observatory: {self.observatory}"]
    #         s += term.bold_green('SOURCES\n')

        if self.freqsetup is not None:
            s_file += [f"Central frequency: {self.freqsetup.frequency:.2}"]
            s_file += [f"With a bandwidth of {self.freqsetup.bandwidth.to(u.MHz)} divided in " \
                       f"{self.freqsetup.n_subbands} x " \
                       f"{self.freqsetup.bandwidth_per_subband.to(u.MHz)}" \
                       f" subbands."]
            s_file += [f"{self.freqsetup.channels} spectral channels per subband.\n"]
        else:
            s_file += ["Frequency setup: not available.\n"]

        s_file += [f"Time range: {self.time.starttime} to {self.time.endtime}\n"]

        s_file += ["## Sources"]
    #         s += term.bright_black('Fringe finders: ') + \
    #              f"{', '.join([s.name for s in self.sources.fringe_finders])}\n"
        s_file += ["Fringe finders: " \
                   f"{', '.join([s.name for s in self.sources.fringe_finders])}"]
    #         s += term.bright_black('Phase calibrators: ') + \
    #              f"{', '.join([s.name for s in self.sources.phase_calibrators])}\n"
        s_file += ["Phase calibrators: " \
                   f"{', '.join([s.name for s in self.sources.phase_calibrators])}"]
    #         s += term.bright_black('Target sources: ') + \
    #              f"{', '.join([s.name for s in self.sources.targets])}\n\n"
        s_file += [f"Target sources: {', '.join([s.name for s in self.sources.targets])}\n"]
        longest_src_name = max([len(s) for s in self.sources.names])
        for src in self.sources:
    #             s += term.bright_black(f"{src.name}: ") + f"src.coordinates.\n"
            s_file += [f"{src.name}:{' '*(longest_src_name-len(src.name))} " \
                    f"{src.coordinates.to_string('hmsdms') if src.coordinates is not None else ''}"]


        s_file += ["\n## Antennas"]
        s_file += ["   Did Observe?  Subbands"]
        for ant in self.antennas:
            if ant.observed:
                try:
                    s_file += [f"{ant.name} yes " \
                               f"{' '*(3*(ant.subbands[0]))}{ant.subbands}"]
                except IndexError:
                    s_file += [f"{ant.name} yes " \
                               f"{' '*3}{ant.subbands}"]
            else:
                s_file += [f"{ant.name} no"]

        if outfile is not None:
            with open(outfile, 'w') as fout:
                fout.write('\n'.join(s_file))

        print('\n'.join(s_file))

        # s_file += ["## Data files"]
    #         s += term.bright_black('') + f"\n"
    #         s += term.bright_black('') + f"\n"
    #         s += term.bright_black('') + f"\n"
    #         s += term.bright_black('') + f"\n"
    #         s += term.bright_black('') + f"\n"
    #         s += term.bright_black('') + f"\n"
    #         s += term.bright_black('') + f"\n"


    def __repr__(self) -> str:
        return f"casa_pipeline.project.Project({self.projectname})"

    def __str__(self) -> str:
        return f"<Project {self.projectname}>"
