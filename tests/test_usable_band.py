"""Tests for the usable band: the subbands that carry a baseline, and the selections built from it.

A heterogeneous array can record a subband with a single antenna, which therefore has no
cross-correlation at all. Such a subband must stay out of every solve: there is nothing to fit in
it, and a ``combine='spw'`` solve would reference its delay to a band wider than the real one.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from vlbipy.models import Antenna, FreqSetup, ObsMetadata

# One antenna on the first subband, one on the last, the rest in between: the EK053B layout.
EK053B = {"EF": (0,), "WB": (1, 2, 3, 4, 5, 6), "JB": (1, 2, 3, 4, 5, 6), "MC": (1, 2, 3, 4, 5), "T6": (7,)}


def metadata(participation: dict, n_subbands: int = 8) -> ObsMetadata:
    """Metadata whose antennas recorded the given subbands."""
    antennas = {name: Antenna(name=name, observed=True, subbands=tuple(subbands))
                for name, subbands in participation.items()}
    return ObsMetadata(project_code="p", antennas=antennas,
                       freq_setup=FreqSetup(n_subbands=n_subbands, n_channels=32))


def test_a_subband_with_a_single_antenna_is_not_usable():
    meta = metadata(EK053B)
    assert meta.usable_subbands == (1, 2, 3, 4, 5, 6)
    assert meta.unusable_subbands == (0, 7)
    assert meta.antennas_per_subband()[0] == ["EF"]
    assert meta.antennas_per_subband()[5] == ["WB", "JB", "MC"]


def test_undetermined_participation_leaves_the_whole_band_usable():
    """A backend that cannot report participation must not shrink the band to nothing."""
    meta = metadata({"EF": (), "WB": ()}, n_subbands=4)
    assert meta.usable_subbands == (0, 1, 2, 3)
    assert meta.unusable_subbands == ()


def test_unobserved_antennas_do_not_make_a_subband_usable():
    meta = metadata({"EF": (0, 1), "WB": (0, 1)}, n_subbands=2)
    meta.antennas["WB"] = replace(meta.antennas["WB"], observed=False)
    assert meta.usable_subbands == ()


def test_selection_strings_restrict_the_solve_to_the_usable_band():
    from vlbipy.backends.casa import central_channel_selection, compact_subband_selection

    assert compact_subband_selection([]) == ""
    assert compact_subband_selection([1, 2, 3, 4, 5, 6]) == "1~6"
    assert compact_subband_selection([7, 1, 3, 2, 5]) == "1~3,5,7"
    assert central_channel_selection(32, 0.8) == "*:3~28"
    assert central_channel_selection(32, 0.8, metadata(EK053B).usable_subbands) == "1~6:3~28"
    # The channel range is repeated per contiguous run: CASA attaches it only to its own token,
    # so "1~3,5:3~28" would silently take every channel of subbands 1 to 3.
    assert central_channel_selection(32, 0.8, (1, 2, 3, 5)) == "1~3:3~28,5:3~28"
    assert central_channel_selection(32, 1.0, (1, 2)) == "1~2"
    assert central_channel_selection(2, 0.8, (1, 2)) == "1~2"


def test_combined_reference_frequency_comes_from_the_selection():
    """``combine='spw'`` references its delay to the centre of the selected band, not of one interval."""
    from vlbipy.solvers.fringefit_task import group_reference_freq

    chan_freq = np.array([[1.0e9, 1.2e9], [2.0e9, 2.2e9], [5.0e9, 5.4e9]])
    assert group_reference_freq(chan_freq, [0, 1, 2]) == 3.2e9
    assert group_reference_freq(chan_freq, [1, 2]) == 3.7e9
    assert group_reference_freq(chan_freq, [1]) == 2.1e9


def test_a_fixed_reference_frequency_overrides_the_grid_centre():
    """So every solution interval of a combined solve shares the frequency its table stores."""
    from vlbipy.solvers.fringe import FringeData

    data = FringeData(vis=np.zeros((1, 1, 2, 1), dtype=np.complex64), weight=np.zeros((1, 1, 2, 1)),
                      flag=np.ones((1, 1, 2, 1), dtype=bool), antenna1=np.array([0]), antenna2=np.array([1]),
                      time=np.array([0.0]), freq=np.array([1.0e9, 1.2e9]), spw_of_chan=np.zeros(2, dtype=int),
                      chan_offset=np.arange(2), nant=2)
    assert data.f_ref_hz == 1.1e9
    assert replace(data, f_ref_fixed=1.5e9).f_ref_hz == 1.5e9


def test_sources_without_a_role_are_still_surveyed():
    """An external project with no [sources] roles must calibrate, not fail on an empty selection."""
    from vlbipy import VLBIObs

    collection = VLBIObs("noroles", network="EVN", backend="dummy")
    collection.import_data()
    obs = collection["noroles"]
    assert not obs.sources.names
    groups = obs.calibrate._calibration_source_groups()
    assert groups == [list(obs.metadata.source_names)] and groups[0]


def test_an_antenna_outside_the_usable_band_is_not_the_preferred_reference():
    """EF tops REFANT_PRIORITY but here it recorded only the subband nobody else has."""
    from vlbipy.selection import rank_reference_antennas

    meta = metadata(EK053B)
    # MC then WB are the next two of REFANT_PRIORITY; EF and T6 fall to the end, in their old order.
    assert rank_reference_antennas(meta) == ["MC", "WB", "JB", "EF", "T6"]
