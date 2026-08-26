"""Tests for the TaQL behind the flagging statistics (no CASA needed)."""
import pytest

casa = pytest.importorskip("vlbipy.backends.casa")


def test_query_excludes_autocorrelations_and_unrecorded_data():
    query = casa.count_query_text("/tmp/x.ms", "DATA")
    assert "ANTENNA1 != ANTENNA2" in query          # no autocorrelations
    assert "gntrue(DATA != 0) AS NOBSERVABLE" in query   # denominator: recorded data only
    assert "gntrue(FLAG && DATA != 0) AS NFLAGGED" in query   # numerator on the same basis
    assert "GROUPBY" not in query


def test_caller_condition_is_parenthesised():
    """AND binds tighter than OR, so a bare disjunction would re-admit autocorrelations."""
    query = casa.count_query_text("/tmp/x.ms", "DATA", where="ANTENNA1 == 2 OR ANTENNA2 == 2")
    assert "ANTENNA1 != ANTENNA2 AND (ANTENNA1 == 2 OR ANTENNA2 == 2)" in query


def test_grouping_selects_and_groups_by_the_column():
    query = casa.count_query_text("/tmp/x.ms", "DATA", groupby="DATA_DESC_ID")
    assert query.startswith("SELECT DATA_DESC_ID, ")
    assert query.endswith(" GROUPBY DATA_DESC_ID")


def test_alternative_data_column_is_used_throughout():
    query = casa.count_query_text("/tmp/x.ms", "CORRECTED_DATA")
    assert "DATA != 0" not in query.replace("CORRECTED_DATA != 0", "")
