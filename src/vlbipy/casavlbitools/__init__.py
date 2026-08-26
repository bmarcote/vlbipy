"""Vendored copy of JIVE's casavlbitools (from the casa-vlbi repository).

Bundled so vlbipy can append Tsys/gain-curve tables to FITS-IDI files
(:func:`fitsidi.append_tsys` / :func:`fitsidi.append_gc`) and convert AIPS
``.uvflg`` flags to CASA format (:func:`fitsidi.convert_flags`) without an
external install. Only the ``from casavlbitools import key`` lines were changed
to relative imports; the original LGPL license headers are preserved in each
module.
"""
