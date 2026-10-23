"""The casacore table format (measurement sets, calibration tables) read and written without casacore.

Pure Python + numpy: ``table.dat`` and the storage managers that CASA writes (StandardStMan, IncrementalStMan,
the tiled storage managers, StManAipsIO) are parsed directly and the bulk columns are memory-mapped. See
:mod:`vlbipy.casacore_format.table` for the table handle and :mod:`vlbipy.casacore_format.tabledat` for the
description file.
"""
