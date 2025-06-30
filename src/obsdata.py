#!/usr/bin/env python3
"""Defines all clases related to a VLBI experiment with all the relevant metadata.
The metadata is obtained from the MS itself.
"""
from __future__ import annotations
import os
import copy
import glob
import shutil
import subprocess
import datetime as dt
import numpy as np
from enum import IntEnum
from pathlib import Path
# from concurrent import futures
# from threading import Lock
from typing import Optional, Iterable, NoReturn, List, Union
from dataclasses import dataclass
from natsort import natsort_keygen
from enum import Enum
from astropy.io import fits
from astropy import units as u
from astropy import coordinates as coord
from rich import print as rprint
# import blessed
import casatasks
from casatools import table as tb
from .casavlbitools import fitsidi
from . import check_antab_idi
# from .project import Project
from . import tools
# from .. import casa_pipeline as capi


# Because as far as I know CASA folks do not have a straight away to get this simple information
class Stokes(IntEnum):
    """The Stokes types defined as in the enum class from casacore code.
    """
    Undefined = 0 # Undefined value
    I = 1 # standard stokes parameters
    Q = 2
    U = 3
    V = 4
    RR = 5 # circular correlation products
    RL = 6
    LR = 7
    LL = 8
    XX = 9 # linear correlation products
    XY = 10
    YX = 11
    YY = 12
    RX = 13 # mixed correlation products
    RY = 14
    LX = 15
    LY = 16
    XR = 17
    XL = 18
    YR = 19
    YL = 20
    PP = 21 # general quasi-orthogonal correlation products
    PQ = 22
    QP = 23
    QQ = 24
    RCircular = 25 # single dish polarization types
    LCircular = 26
    Linear = 27
    Ptotal = 28 # Polarized intensity ((Q^2+U^2+V^2)^(1/2))
    Plinear = 29 #  Linearly Polarized intensity ((Q^2+U^2)^(1/2))
    PFtotal = 30 # Polarization Fraction (Ptotal/I)
    PFlinear = 31 # linear Polarization Fraction (Plinear/I)
    Pangle = 32 # linear polarization angle (0.5  arctan(U/Q)) (in radians)



class ObsEpoch(object):
    """Time information from the observation.
    It provides different time values like the epoch of the observation (in different formats),
    and the time range (or duration) of the observation.
    """
    @property
    def epoch(self) -> Optional[dt.date]:
        """Returns the epoch at which the observations were conducted, in datetime.time format.
        """
        return self._epoch


    @property
    def ymd(self) -> Optional[str]:
        """Returns the epoch at the (start of) observations, in YYMMDD format.
        """
        return self.epoch.strftime('%y%m%d') if self.epoch is not None else None


    @property
    def mjd(self) -> Optional[float]:
        """Returns the Modifyed Julian Day (MJD) relative to the start of the observation.
        """
        if self.starttime is not None:
            return float(tools.date2mjd(self.starttime))
        elif self.epoch is not None:
            return float(tools.date2mjd(dt.datetime(*self.epoch.timetuple()[:6])))

        return None


    @property
    def doy(self) -> Optional[int]:
        """Returns the day of the year (DOY) related to the start of the observation.
        """
        return int(self.epoch.strftime('%j')) if self.epoch is not None else None


    @property
    def starttime(self) -> Optional[dt.datetime]:
        """Returns the start time of the observation in datetime format.
        """
        return self._starttime


    @starttime.setter
    def starttime(self, start_datetime: dt.datetime):
        if self.endtime is not None:
            assert (self.endtime - start_datetime) > dt.timedelta(days=0), \
                   "Starting time needs to be earlier than ending time."

        self._starttime = start_datetime
        self._epoch = start_datetime.date()


    @property
    def endtime(self) -> Optional[dt.datetime]:
        """Returns the ending time of the observation in datetime format.
        """
        return self._endtime


    @endtime.setter
    def endtime(self, end_datetime: dt.datetime):
        if self.starttime is not None:
            assert (end_datetime - self.starttime) > dt.timedelta(days=0), \
                   "Ending time needs to be later than starting time."

        self._endtime = end_datetime


    @property
    def duration(self) -> Optional[u.Quantity]:
        return ((self.endtime - self.starttime).total_seconds()*u.s).to(u.hour) \
                if None not in (self.endtime, self.starttime) else None


    def __init__(self, start_datetime: Optional[dt.datetime] = None,
                 end_datetime: Optional[dt.datetime] = None) -> None:
        self._starttime = start_datetime
        self._endtime = end_datetime
        self._epoch = self.starttime.date() if self.starttime is not None else None



class SourceType(Enum):
    """Type of source (target, calibrator, fringefinder, or other)
    """
    target = 0
    calibrator = 1
    fringefinder = 2
    other = 3


class Source(object):
    """Defines a source by name, type (i.e. target, reference, fringefinder, other)
    and if it must be protected or not (password required to get its data).
    """
    @property
    def name(self) -> str:
        """Name of the source.
        """
        return self._name

    @property
    def coordinates(self) -> Optional[coord.SkyCoord]:
        """Coordinates of the source, as astropy.coordinate.SkyCoord type.
        Returns None if they haven't yet been initialized.
        """
        return self._coordinates

    @property
    def type(self) -> SourceType:
        """Type of the source (SorceType object).
        """
        return self._type

    @type.setter
    def type(self, new_type: Union[str, SourceType]):
        """Sets the type of the given source.

        new_type : [str/SourceType]
            The new type for the given source. If string, it needs to match
            'target', 'phaseref', 'fringefinder' or 'other'.
            Otherwise, it is expected to be a 'SourceType' parameter.
        """
        if isinstance(new_type, SourceType):
            self._type =new_type
        elif isinstance(new_type, str):
            if new_type == 'target':
                self._type = SourceType.target
            elif new_type == 'phaseref':
                self._type = SourceType.calibrator
            elif new_type == 'fringefinder':
                self._type = SourceType.fringefinder
            elif new_type == 'other':
                self._type = SourceType.other
            else:
                raise ValueError(f"The new type given for the source {self.name} " \
                                 f"({new_type}) has an unexpected value, " \
                                 f"while 'target'/'phaseref'/'fringefinder'/'other' were expected.")
        else:
            raise ValueError(f"The new type given for the source {self.name} " \
                             f"({new_type}) has an unexpected type {type(new_type)}, " \
                             "while str/SourceType were expected.")


    @property
    def model(self) -> Optional[Path]:
        """Source model file (generated after imaging or by other tasks;
        it can be used for calibration).
        """
        return self._model


    @model.setter
    def model(self, new_model: Optional[Union[Path, str]]):
        if isinstance(new_model, str):
            new_model = Path(new_model)

        self._model = new_model


    def __init__(self, name: str, sourcetype: SourceType,
                 coordinates: Optional[coord.SkyCoord] = None,
                 model: Optional[Union[Path, str]] = None):
        assert isinstance(name, str), f"The name of the source must be a string (currrently {name})"
        assert isinstance(sourcetype, SourceType), \
               f"The name of the source must be a SourceType object (currrently {sourcetype})"
        assert isinstance(coordinates, coord.SkyCoord) or coordinates is None, \
               "The coordinates of the source must " \
               f"be an astropy.coordinates.SkyCoord object (currently {coordinates})"
        self._name = name
        self._type = sourcetype
        self._coordinates = coordinates
        self.model = model


    def __iter__(self):
        for key in ('name', 'type', 'protected'):
            yield key, getattr(self, key)


    def __str__(self) -> str:
        if self.coordinates is not None:
            return f"Source({self.name}, {self.type.name}, " \
                   f"at {self.coordinates.to_string('hmsdms')})"
        else:
            return f"Source({self.name}, {self.type.name}, unknown coordinates)"


    def __repr__(self) -> str:
        return self.__str__()


    def json(self) -> dict:
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = dict()
        for key, val in self.__iter__():
            if isinstance(val, SourceType):
                d[key] = val.name
            elif isinstance(val, coord.SkyCoord):
                d[key] = val.to_string('hmsdms')
            else:
                d[key] = str(val)

        return d



class Sources(object):
    """Defines a collection of sources.
    """
    def __init__(self, sources: Optional[Iterable[Source]] = None):
        if sources is not None:
            self._sources = copy.deepcopy(list(sources))
        else:
            self._sources = []


    def append(self, new_source: Source) -> None:
        """Adds a new source to the collection of sources.

        Note that this function and "add" are completely equivalent.
        """
        if new_source.name in self.names:
            raise ValueError("The specified source is already in the Sources list.")

        self._sources.append(new_source)


    def add(self, new_source: Source) -> None:
        """Adds a new source to the collection of sources.

        Note that this function and "add" are completely equivalent.
        """
        self.append(new_source)


    @property
    def names(self) -> List[str]:
        """Returns the names of all sources included in Sources.
        """
        return [s.name for s in self._sources]


    @property
    def types(self) -> List[SourceType]:
        """Returns the types of all sources in Sources.
        """
        return [s.type for s in self._sources]


    @property
    def coordinates(self) -> List[Optional[coord.SkyCoord]]:
        """Returns the coordinates of all sources in Sources.
        """
        return [s.coordinates for s in self._sources]


    def change_type(self, source: str, new_source_type: SourceType):
        """Changes the type of the given source to be a target source.
        The source must already be in the list of sources.
        """
        if source not in self.names:
            raise ValueError("The introduced source ({source}) is not part of the observation.")

        self[source].type = new_source_type

    @property
    def targets(self) -> Sources:
        """Returns the target sources.
        """
        return Sources([s for s in self._sources if s.type == SourceType.target])

    @targets.setter
    def targets(self, new_target: str):
        """Changes the type of the given source to be a target source.
        The source must already be in the list of sources.
        """
        self.change_type(new_target, SourceType.target)

    @property
    def phase_calibrators(self) -> Sources:
        """Returns the phase calibrator sources.
        """
        return Sources([s for s in self._sources if s.type == SourceType.calibrator])

    @phase_calibrators.setter
    def phase_calibrators(self, new_target: str):
        """Changes the type of the given source to be a phase calibrator source.
        The source must already be in the list of sources.
        """
        self.change_type(new_target, SourceType.calibrator)

    @property
    def all_calibrators(self) -> Sources:
        """Returns all calibrator sources (including phase calibrators and fringe finders).
        """
        return Sources([s for s in self._sources if (s.type == SourceType.calibrator) or \
                       (s.type == SourceType.fringefinder)])

    @property
    def other_sources(self) -> Sources:
        """Returns the sources catalogued as "others".
        """
        return Sources([s for s in self._sources if s.type == SourceType.other])

    @other_sources.setter
    def other_sources(self, new_other_source: str):
        """Changes the type of the given source to be a "other" source.
        The source must already be in the list of sources.
        """
        self.change_type(new_other_source, SourceType.other)

    @property
    def fringe_finders(self) -> Sources:
        """Returns the fringe-finder sources.
        """
        return Sources([s for s in self._sources if s.type == SourceType.fringefinder])

    @fringe_finders.setter
    def fringe_finders(self, new_fringefinder: str):
        """Changes the type of the given source to be a fringe finder source.
        The source must already be in the list of sources.
        """
        self.change_type(new_fringefinder, SourceType.fringefinder)

    def __len__(self):
        return len(self._sources)

    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, int):
            return self._sources[key]
        else:
            return self._sources[self.names.index(key)]

    def __delitem__(self, key: Union[int, str]):
        if isinstance(key, int):
            return self._sources.remove(self[key])
        else:
            return self._sources.remove(self[key])

    def __iter__(self):
        return self._sources.__iter__()

    def __reversed__(self):
        return self._sources[::-1]

    def __contains__(self, key):
        return key in self.names

    def __str__(self):
        return f"Sources([{', '.join(self.names)}])"

    def __repr__(self):
        return f"Sources([{', '.join(self.names)}])"

    def json(self):
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = list()
        for source in self.__iter__():
            d.append(source.json())

        return d


@dataclass
class Antenna:
    """Defines an antenna.
    It has three parameters:
        name : str
            Name of the antenna
        observed : bool
            If the antenna has observed (has no-null data).
        subbands : tuple
            Tuple with the subbands numbers where the antenna observed.
            It may be all subbands covered in the observation or a subset of them.
    """
    name: str
    observed: bool = True
    subbands: tuple = ()


class Antennas(object):
    """List of antennas (Antenna class)
    """
    def __init__(self, antennas: Optional[Iterable[Antenna]] = None):
        if antennas is not None:
            self._antennas = copy.deepcopy(list(antennas))
        else:
            self._antennas = []

        self._niter = -1


    def append(self, new_antenna: Antenna):
        """Adds a new antenna to the collection of antennas.

        Note that this function and "add" are completely equivalent.
        """
        self.add(new_antenna)


    def add(self, new_antenna: Antenna):
        """Adds a new antenna to the collection of antennas.

        Note that this function and "add" are completely equivalent.
        """
        assert isinstance(new_antenna, Antenna)
        self._antennas.append(new_antenna)


    @property
    def names(self) -> List[str]:
        """Returns the name of the antenna
        """
        return [a.name for a in self._antennas]


    @property
    def observed(self) -> List[str]:
        """Returns the antenna names that have data (have observed) in the observation
        """
        return [a.name for a in self._antennas if a.observed]


    @property
    def subbands(self):
        """Returns the subbands that each antenna observed.
        """
        return [a.subbands for a in self._antennas if a.observed]

    def __len__(self) -> int:
        return len(self._antennas)

    def __getitem__(self, key) -> Antenna:
        return self._antennas[self.names.index(key)]

    def __delitem__(self, key):
        return self._antennas.remove(self[key])

    def __iter__(self) -> Iterable[Antenna]:
        self._niter = -1
        for ant in self._antennas:
            yield ant


    def __next__(self) -> Antenna:
        if self._niter < self.__len__()-1:
            self._niter += 1
            return self._antennas[self._niter]

        raise StopIteration


    def __reversed__(self) -> list[Antenna]:
        return self._antennas[::-1]

    def __contains__(self, key) -> bool:
        return key in self.names

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "Antennas:\n" \
           f"([{', '.join([ant if ant in self.observed else '['+ant+']' for ant in self.names])})"

    def json(self) -> list[dict]:
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = list()
        for ant in self.__iter__():
            d.append({'name': ant.name, 'observed': ant.observed, 'subbands': ant.subbands})

        return d


class FreqSetup(object):
    """Defines the frequency setup of a given observation with the following data:
        - n_subbands :  int
            Number of subbands.
        - channels : int
            Number of channels per subband.
        - frequencies : array-like
            Reference frequency for each channel and subband (NxM array, with N
            number of subbands, and M number of channels per subband).
        - central_freq : astropy.units.Quantity
            Central frequency of the observation.
        - bandwidths : astropy.units.Quantity or float
            Total bandwidth for each subband.
    """
    @property
    def n_subbands(self) -> int:
        """Returns the number of subbands (spectral windows) of the observation.
        """
        return self._n_subbands


    @property
    def channels(self) -> int:
        """Returns the number of channels per subband of the observation.
        Assumes that all subbands have the same number of channels.
        """
        return self._channels


    @property
    def frequency(self) -> u.Quantity:
        """Returns the central frequency of the observation, in a astropy.units.Quantity format.
        """
        return self._frequency.to(u.GHz)


    @property
    def bandwidth_per_subband(self) -> u.Quantity:
        """Returns the total bandwidth per subband
        """
        return self._bandwidth_spw.to(u.MHz)


    @property
    def bandwidth(self) -> u.Quantity:
        """Returns the total bandwidth of the observation
        """
        return self._bandwidth_spw.to(u.MHz) * self.n_subbands


    @property
    def frequency_range(self) -> tuple:
        """Returns the lowest and highest frequency observed as a tuple.
        """
        return ((self.frequency - self.bandwidth/2).to(u.GHz),
                (self.frequency + self.bandwidth/2).to(u.GHz))


    def __init__(self, channels: int, n_subbands: int, central_frequency: Union[float, u.Quantity],
                 bandwidth_spw: Union[float, u.Quantity]):
        """Inputs:
            - channels : int
                Number of channels per subband.
            - n_subbands : int
                Number of subbands.
            - central_frequency: float or astropy.units.Quantity
                Central frequency of the observation. If no units are provided, Hz are assumed.
            - bandwidth_spw : float or astropy.units.Quantity
                Total bandwidth for each subband. If no units are provided, Hz are assumed.
        """
        if n_subbands <= 0:
            raise ValueError("n_subbands (currently {n_subbands}) must be a positive integer")

        self._channels = int(channels)
        self._n_subbands = n_subbands

        if isinstance(central_frequency, float):
            self._frequency = u.Quantity(central_frequency, u.Hz)
        else:
            self._frequency = central_frequency

        if isinstance(bandwidth_spw, float):
            self._bandwidth_spw = u.Quantity(bandwidth_spw*1e-9, u.GHz)
        else:
            self._bandwidth_spw = bandwidth_spw


    def __iter__(self):
        for key in ('n_subbands', 'channels', 'bandwidth_spw', 'frequency_range'):
            yield key, getattr(self, key)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"<FreqSetup>\n{self.frequency:.3f} central frequency.\n{self.n_subbands} x " \
               f"{self.bandwidth_per_subband.to(u.MHz):.1f} " \
               f"subbands. {self.channels} channels per subband."

    def json(self) -> dict:
        """Returns a dict with all attributes of the object.
        I define this method to use instead of .__dict__ as the later only reporst
        the internal variables (e.g. _username instead of username) and I want a better
        human-readable output.
        """
        d = dict()
        for key, val in self.__iter__():
            if isinstance(val, u.Quantity):
                d[key] = val.to(u.Hz).value
            else:
                d[key] = val

        return d


class Importing(object):
    """Class that contains different static functions to import data from different observatories
    into a MS. Each function would conduct the necessary steps to get a properly prepared MS file.
    """

    def __init__(self, ms: Project):
        self._ms = ms

    def lba_fits(self, fitsfile: str):
        # TODO: LBA import in CASA?
        raise NotImplementedError

    def evn_download(self, obsdate: Optional[str] = None, projectcode: Optional[str] = None,
                     username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """Downloads the data files associated with the given EVN project.
        It will download the associated FITS-IDI files, and the associated .antab and .uvflg files.

        Inputs
            obsdate : str  (default = None, assumes the info is written in the Project)
                String with the observing date, in the format YYMMDD.
            projectcode : str  (default = None)
                Project code of the experiment to download. If None, assumes the same as the
                project code.
            username : str  (default: None)
                Username required to download the data, if they are still private.
                If None, assumes that no credentials are needed because the proprieatary period
                already expired.
            password : str  (default: None)
                Password required to download the data, if they are still private.
                If None, assumes that no credentials are needed because the proprieatary period
                already expired.
        """
        params = []
        expname = projectcode if projectcode is not None else self._ms.projectname
        if obsdate is None:
            try:
                obsdate = self._ms.time.ymd if self._ms.time.ymd is not None else \
                          self._ms.params['epoch']
            except KeyError:
                rprint("[red bold]In order to download the EVN data, the 'epoch' (YYMMDD format) " \
                       "needs to be provided in the initial parameters.[/red bold]")

        if None not in (username, password):
            params += ["--http-user", username, "--http-passwd", password]

        params += ["-t45", "-l1", "-r", "-nd", "archive.jive.nl/exp/" \
                   f"{expname.upper()}_{obsdate}/fits -A '{expname.lower()}*'"]

        try:
            tools.shell_command("wget", params)
            tools.shell_command("md5sum", ["-c", f"{expname.lower()}.checksum"])
            # TODO: verify here that all files are OK!
            tools.shell_command("wget",
                                     [f"http://archive.jive.nl/exp/{expname.upper()}_{obsdate}/" \
                                      f"pipe/{expname.lower()}.antab.gz"])
            tools.shell_command("gunzip", [f"{expname.lower()}.antab.gz"])
            # TODO: if ERROR 404: Not Found, then go for the _1, _2,...
            tools.shell_command("wget",
                                     [f"http://archive.jive.nl/exp/{expname.upper()}_{obsdate}/" \
                                      f"pipe/{expname.lower()}.uvflg"])
            # TODO: this only if I need AIPS
            tools.shell_command("wget",
                                     [f"http://archive.jive.nl/exp/{expname.upper()}_{obsdate}/" \
                                      f"pipe/{expname.lower()}.tasav.FITS.gz"])
            tools.shell_command("gunzip", [f"{expname.lower()}.tasav.FITS.gz"])
        except ValueError as err:
            rprint(f"[bold red]ERROR: you may require credentials to download the data[/bold red]")
            rprint(f"[red]{err}[/red]")

        return all(f.exists() for f in (Path(f"{expname.lower()}.antab"),
                                               Path(f"pipe/{expname.lower()}.tasav.FITS"),
                                               Path(f"pipe/{expname.lower()}.uvflg"))) \
                    and (len(glob.glob(f"{expname.lower()}_*_1.IDI*")) > 0)


    def get_obsdate_from_fitsidi(self) -> dt.date:
        """Returns the observing epoch in datetime format as read from the FITS-IDI.
        """
        if self._ms.time.epoch is not None:
            return self._ms.time.epoch

        p = lambda x: Path(f"{self._ms.projectname}_1_1.IDI{x}")
        a_fitsidi = p("1") if p("1").exists() else p("")
        assert a_fitsidi.exists()
        with fits.open(a_fitsidi) as hdu:
            return dt.datetime.strptime(hdu[1].header['RDATE'], "%Y-%m-%d").date()


    def get_freq_from_fitsidi(self) -> u.Quantity:
        """Returns the central frequency of the observations as read from the FITS-IDI files.
        """
        if self._ms.freqsetup is not None:
            return self._ms.freqsetup.frequency

        p = lambda x: Path(f"{self._ms.projectname}_1_1.IDI{x}")
        a_fitsidi = p("1") if p("1").exists() else p("")
        assert a_fitsidi.exists()
        with fits.open(a_fitsidi) as hdu:
            return (hdu[1].header['REF_FREQ']*u.Hz).to(u.GHz)


    def evn_fitsidi(self, fitsidifiles: Union[list, str, None] = None, ignore_antab: bool = False,
                    ignore_uvflg: bool = False, delete: bool = False, replace_tsys: bool = False):
        """Imports the provided FITS-IDI files from an EVN observation into a single MS file.
        If checks if the FITS-IDI files already contain the Tsys and GC tables. Otherwise, it
        will first append such information and then import the files.

        If a .uvflg file exists (AIPS-format flag file), it will convert it to a CASA-compatible
        .flag file.

        Inputs
            fitsidifiles : list or str  (default: None)
                If str, it will retrieve all files that match the given path/name.
                If list, it must be an ordered list of all FITS-IDI files that will be imported.
                If None, then it will assume that the FITS-IDI files are in the current directory
                with the name '<projectname>_1_1.IDI*'.
            ignore_antab : bool (default False)
                If the FITS-IDI files should be imported into a MS without caring about the ANTAB
                information (not recommended).
            ignore_uvflg : bool (default False)
                If the FITS-IDI files should be imported into a MS without caring about the .uvflg
                file (AIPS-format a-priori flag table) information (not recommended).
            delete : bool (default False)
                Overwrites the MS if existing.
            replace_tsys : bool (default False)
                Overwrites the Tsys/GC information, if exists, in the FITS-IDI files.
        """
        if self._ms.msfile.exists():
            if delete:
                shutil.rmtree(self._ms.msfile)
            else:
                rprint(f"[bold yellow]The MS file {self._ms.msfile} already exists "
                       "and you told me to not remove it[/bold yellow]")
                raise FileExistsError

        # First run the Tsys and GC appending to the FITS-IDI. These functions will do nothing
        # if the information is already there.
        if fitsidifiles is None:
            fitsidifiles = sorted(glob.glob(f"{self._ms.projectname.lower()}_1_1.IDI*"),
                                  key=natsort_keygen())
        elif isinstance(fitsidifiles, str):
            fitsidifiles = sorted(glob.glob(fitsidifiles), key=natsort_keygen())
        elif isinstance(fitsidifiles, list):
            fitsidifiles = sorted(fitsidifiles, key=natsort_keygen())
        else:
            raise TypeError("The argument fitsidifiles should be either a list or str. "
                            f"Currently is {fitsidifiles} (type {type(fitsidifiles)})")

        self._ms.logger.debug(f"FITS-IDI files to read: {', '.join(fitsidifiles)}.")
        if len(fitsidifiles) == 0:
                raise FileNotFoundError(f"No files matching {fitsidifiles} could be found.")

        for a_fitsidi in fitsidifiles:
            if not os.path.isfile(a_fitsidi):
                raise FileNotFoundError(f"The file {a_fitsidi} could not be found.")

        if tools.space_available(self._ms.cwd) <= u.Quantity(1.55*3, u.kbit)* \
                    int(subprocess.run(f"du -sc {' '.join(fitsidifiles)}", shell=True,
                                       capture_output=True).stdout.decode().split()[-2]):
            rprint("\n\n[bold red]There is no enough space in the computer to create " \
                   "the MS file and perform the data reduction[/bold red]")
            raise IOError("Not enough disk space to create the MS file.")

        antabfile = self._ms.cwd / Path(f"{self._ms.projectname.lower()}.antab")
        uvflgfile = self._ms.cwd / Path(f"{self._ms.projectname.lower()}.uvflg")
        flagfile = self._ms.cwd / Path(f"{self._ms.projectname.lower()}.flag")
        if not ignore_antab:
            if antabfile.exists():
                rprint("[bold]Appending the Tsys values to the FITS-IDI files[/bold]")
                try:
                    fitsidi.append_tsys(str(antabfile), fitsidifiles, replace=replace_tsys)
                    rprint("[bold]Appending the GC values to the FITS-IDI files[/bold]")
                    fitsidi.append_gc(str(antabfile), fitsidifiles[0], replace=replace_tsys)
                except RuntimeError as e:
                    rprint(f"[yellow]{e}[/yellow]")
            else:
                assert all(map(check_antab_idi.check_consistency,
                               [afile for afile in fitsidifiles if afile.endswith('IDI1') \
                                                                or afile.endswith('IDI')])), \
                        "The FITS-IDI files do not have stored the Tsys/Gain Curve information, " \
                        "and no '.antab' file has been found in the current directory."

        if (not ignore_uvflg) and (not flagfile.exists()):
            assert uvflgfile.exists(), \
                   f"The associated file {uvflgfile} should exist but was not found."
            rprint("[bold]Converting the AIPS-style flags into CASA-style flags[/bold]")
            rprint(f"[green]The file {self._ms.projectname.lower()}.flag containing the "
                   "a-priori flagging has been created.[/green]")
            fitsidi.convert_flags(infile=str(uvflgfile), idifiles=fitsidifiles,
                                  outfile=str(flagfile))

        rprint("[bold]Importing the FITS-IDI files into a MS[/bold]")
        casatasks.importfitsidi(vis=str(self._ms.msfile), fitsidifile=fitsidifiles, constobsid=True,
                                scanreindexgap_s=8.0, specframe='GEO')
        rprint(f"[green]The file {self._ms.msfile} has been created[/green]")
        self._ms.get_metadata_from_ms()


    def _fix_ant_names(self, msdata: Optional[Union[str, Path]] = None):
        """Fixes the antenna table in a MS.
        When imported from a UVFITS from AIPS, the antenna names are dropped and the antenna
        name column contains numbers intead. Luckily, the antenna names are preserved in the site
        column. Which this function recovers to write them again into the antenna names.
        """
        msdata = self._ms.msfile if msdata is None else msdata

        m = tb(str(msdata))
        if not m.open(str(msdata)):
            raise ValueError(f"The MS file {msdata} could not be openned.")

        msanttable = m.getkeyword('ANTENNA').replace('Table: ', '')

        m = tb(str(msanttable))
        if not m.open(msanttable, nomodify=False):
            raise ValueError(f"The MS file {msanttable} could not be openned.")

        antnames = m.getcol('NAME')
        if antnames[0].isnumeric():
            m.putcol('NAME', list(m.getcol('STATION')))

        m.done()  # just because I would not trust the CASA code...


    def import_uvfits(self, uvfitsfile: Union[str, Path],
                      delete: bool = False):
        """Imports a UVFITS file into a MS.
        """
        if self._ms.msfile.exists():
            if delete:
                shutil.rmtree(self._ms.msfile)
            else:
                rprint(f"[bold yellow]The MS file {self._ms.msfile} already exists "
                       "and you told me to not remove it[/bold yellow]")
                raise FileExistsError

        casatasks.importuvfits(fitsfile=str(uvfitsfile), vis=str(self._ms.msfile),
                               antnamescheme='new')
        # Fixing the antenna name columns... from AIPS the antenna names are moved to numbers
        self._fix_ant_names(str(self._ms.msfile))
        self._ms.get_metadata_from_ms()



