"""The native gain-curve table (``gencal caltype='gc'``) against CASA's arithmetic and CASA's table for RSM07."""
from pathlib import Path

import numpy as np
import pytest

from vlbipy.casacore_format.table import Table
from vlbipy.solvers.gaincurve_task import NCOEF, el_to_za_matrix, gain_curve_parameters, write_gain_curve_table

RSM07 = Path(__file__).resolve().parents[1] / "test" / "RSM07"


def test_elevation_polynomial_is_re_expanded_in_zenith_angle():
    coefficients = np.zeros(NCOEF)
    coefficients[:3] = [0.85, 5.0e-3, -5.5e-5]                     # a typical ANTAB POLY(el), peak near 45 deg
    za = np.linspace(0.0, 80.0, 9)
    expanded = el_to_za_matrix()[:NCOEF, :NCOEF].astype(float) @ coefficients
    assert np.allclose(np.polyval(expanded[::-1], za), np.polyval(coefficients[::-1], 90.0 - za), atol=1e-6)


def test_power_curve_is_scaled_by_dpfu_and_a_voltage_curve_squares_only_its_first_polarization():
    gain = np.zeros((3, 2), dtype=np.float32)
    gain[:, 0] = [1.0, 0.5, 0.0]
    gain[:, 1] = [2.0, 0.0, 0.0]
    dpfu = np.array([0.1, 0.2], dtype=np.float32)
    power = gain_curve_parameters(["POWER(ZA)"], np.array([3]), [gain], [dpfu])[0]
    assert np.allclose(power[:3], [0.1, 0.05, 0.0]) and np.allclose(power[NCOEF:NCOEF + 3], [0.4, 0.0, 0.0])
    voltage = gain_curve_parameters(["VOLTAGE(ZA)"], np.array([3]), [gain], [dpfu])[0]
    assert np.allclose(voltage[:3], 0.1 * np.array([1.0, 1.0, 0.25]))      # (1 + 0.5 za)^2
    assert np.all(voltage[NCOEF:] == 0.0)                                   # CASA's quirk, kept on purpose


@pytest.mark.skipif(not (RSM07 / "RSM07.mms").is_dir() or not (RSM07 / "caltables" / "RSM07.gcal").is_dir(),
                    reason="RSM07 test data not available")
def test_rsm07_gain_curve_table_is_identical_to_casa(tmp_path):
    native = tmp_path / "RSM07.gcal"
    report = write_gain_curve_table(RSM07 / "RSM07.mms", native, caltype="gc")
    assert report["rows"] == 48
    with Table(RSM07 / "caltables" / "RSM07.gcal") as casa, Table(native) as ours:
        assert ours.colnames() == casa.colnames() and ours.nrows() == casa.nrows()
        for column in casa.colnames():
            if column == "WEIGHT":                                          # undefined in both
                continue
            assert np.array_equal(np.asarray(ours.getcol(column)), np.asarray(casa.getcol(column))), column
        assert ours.info() == casa.info()
        assert sorted(ours.getkeywords()) == sorted(casa.getkeywords())
        for name in ("OBSERVATION", "ANTENNA", "FIELD", "SPECTRAL_WINDOW"):
            with Table(native / name) as sub_ours, Table(RSM07 / "caltables" / "RSM07.gcal" / name) as sub_casa:
                assert sub_ours.colnames() == sub_casa.colnames(), name
                for column in sub_casa.colnames():
                    left, right = sub_ours.getcol(column), sub_casa.getcol(column)
                    assert (left == right if isinstance(right, list) else np.array_equal(left, right)), (name, column)
