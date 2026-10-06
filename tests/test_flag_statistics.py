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


def test_summary_counts_surviving_baselines_directly_and_preserves_where(monkeypatch):
    ops = object.__new__(casa.CasaFlagOps)
    monkeypatch.setattr(ops, "_antenna_names", lambda project: ["EF", "WB", "TR"])
    calls = []

    def count(project, *, where="", groupby=""):
        calls.append((where, groupby))
        if groupby == "DATA_DESC_ID":
            return [{"DATA_DESC_ID": 0, "flagged": 5, "observable": 20}]
        if groupby == "ANTENNA1":
            return [{"ANTENNA1": 0, "flagged": 2, "observable": 10},
                    {"ANTENNA1": 2, "flagged": 10, "observable": 10}]
        if groupby == "ANTENNA2":
            return [{"ANTENNA2": 1, "flagged": 3, "observable": 10}]
        if "NOT IN" in where:
            return [{"flagged": 4, "observable": 20}]
        return [{"flagged": 15, "observable": 30}]

    monkeypatch.setattr(ops, "_count_query", count)
    report = ops.summary("P", where="FIELD_ID == 7 OR FIELD_ID == 9")
    assert report["excluding_dead"] == {"flagged": 4, "observable": 20, "fraction": 0.2,
                                         "excluded_antennas": ["TR"]}
    surviving_where = [where for where, groupby in calls if "NOT IN" in where and not groupby][0]
    assert "(FIELD_ID == 7 OR FIELD_ID == 9)" in surviving_where
    assert "ANTENNA1 NOT IN [2]" in surviving_where
    assert "ANTENNA2 NOT IN [2]" in surviving_where
