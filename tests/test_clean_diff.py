import numpy as np
import pandas as pd
import pytest

from keylogging_analysis.clean import clean_events
from keylogging_analysis.config import MetricConfig
from keylogging_analysis.diff import derive_edits, edit_between
from keylogging_analysis.metrics._common import dist_stats, per_minute, safe_ratio, span_ms
from keylogging_analysis.schema import KeylogData, validate
from helpers import MSG_A, MSG_B, edits_of, events_of, messages_of


def _clean(*messages, config=None):
    config = config or MetricConfig()
    data = validate(KeylogData(events_of(*messages), messages_of([m for m, _ in messages])))
    return clean_events(data.events, config)


@pytest.mark.parametrize("prev,cur,expected", [
    ("", "H", (0, 0, 1)),            # first keystroke
    ("Hi", "Hi ", (2, 0, 1)),        # append
    ("Hi th", "Hi t", (4, 1, 0)),    # backspace at the end
    ("I am", "I sm", (2, 1, 1)),     # mid-text replacement
    ("ab", "b", (0, 1, 0)),          # delete first character
    ("aa", "a", (1, 1, 0)),          # ambiguous: counts are right, pos is the later one
    ("abc", "abc", (3, 0, 0)),       # no change
    ("I", "I am", (1, 0, 3)),        # multi-character insert
    ("hello world", "hello", (5, 6, 0)),
])
def test_edit_between(prev, cur, expected):
    assert edit_between(prev, cur) == expected


def test_edit_between_non_ascii():
    assert edit_between("café", "café’") == (4, 0, 1)
    assert edit_between("ok 😀", "ok 😀!") == (4, 0, 1)
    assert edit_between("naïve", "naive") == (2, 1, 1)


def test_clean_orders_by_time_and_counts_regressions():
    # seq order: a (0 ms), abc (300 ms), ab (200 ms) -> time goes back once
    events, flags, report = _clean(("E", [(0, "a"), (300, "abc"), (200, "ab")]))
    assert events["text"].tolist() == ["a", "ab", "abc"]
    assert flags.loc["E", "n_time_regressions"] == 1
    assert report.n_messages_with_time_regression == 1


def test_clean_ties_broken_by_seq():
    events, _, _ = _clean(("T", [(0, "a"), (0, "ab")]))
    assert events["text"].tolist() == ["a", "ab"]


def test_clean_drops_nochange_events_including_empty_first_state():
    events, flags, report = _clean(("N", [(0, ""), (10, "a"), (20, "a"), (30, "a"), (40, "ab")]))
    assert events["text"].tolist() == ["a", "ab"]
    assert flags.loc["N", "n_nochange_dropped"] == 3
    assert (report.n_events_in, report.n_nochange_dropped, report.n_events_out) == (5, 3, 2)


def test_clean_keeps_nochange_when_configured():
    events, flags, _ = _clean(("N", [(0, "a"), (10, "a")]),
                              config=MetricConfig(drop_nochange_events=False))
    assert len(events) == 2
    assert flags.loc["N", "n_nochange_dropped"] == 0


def test_clean_flags_cover_messages_whose_events_all_drop():
    # every state of Z equals its predecessor ("" before the first): Z leaves `events`
    # entirely, but its flags row must remain so the drop is still reported
    events, flags, _ = _clean(("Z", [(0, ""), (10, "")]), ("K", [(0, "k")]))
    assert "Z" not in set(events["message_id"])
    assert flags.loc["Z", "n_nochange_dropped"] == 2


def test_derive_edits_message_a():
    e = edits_of(("A", MSG_A))
    assert e["op"].tolist() == ["insert"] * 5 + ["delete"] * 2 + ["insert"] * 2
    assert e["pos"].tolist() == [0, 1, 2, 3, 4, 4, 3, 3, 4]
    assert e["n_ins"].tolist() == [1, 1, 1, 1, 1, 0, 0, 1, 1]
    assert e["n_del"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert e["at_end"].tolist() == [True] * 9
    assert e["prev_char"].tolist() == ["", "H", "i", " ", "t", "t", " ", " ", "y"]
    assert e["ins_first"].tolist() == ["H", "i", " ", "t", "h", "", "", "y", "o"]
    assert np.isnan(e["iki_ms"].iloc[0])
    assert e["iki_ms"].iloc[1:].tolist() == [100, 200, 2200, 100, 100, 100, 300, 400]
    assert not e["bulk"].any()


def test_derive_edits_message_b_bulk_replace_and_dropped_nochange():
    e = edits_of(("B", MSG_B))
    assert e["text"].tolist() == ["I", "I am", "I sm", "I sm!"]
    assert e["op"].tolist() == ["insert", "insert", "replace", "insert"]
    assert e["n_ins"].tolist() == [1, 3, 1, 1]
    assert e["n_del"].tolist() == [0, 0, 1, 0]
    assert e["at_end"].tolist() == [True, True, False, True]
    assert e["bulk"].tolist() == [False, True, False, False]
    assert e["iki_ms"].iloc[1:].tolist() == [150, 150, 300]   # 450 ms state was dropped


def test_derive_edits_resets_between_messages():
    e = edits_of(("A", MSG_A), ("B", MSG_B))
    first_b = e.index[e["message_id"] == "B"][0]
    assert np.isnan(e.loc[first_b, "iki_ms"])
    assert e.loc[first_b, "pos"] == 0 and e.loc[first_b, "prev_char"] == ""


def test_derive_edits_empty():
    e = edits_of()
    assert len(e) == 0
    assert {"pos", "n_ins", "n_del", "op", "at_end", "iki_ms", "bulk"} <= set(e.columns)


def test_common_helpers():
    e = edits_of(("A", MSG_A), ("B", MSG_B))
    sp = span_ms(e)
    assert sp["A"] == 3500 and sp["B"] == 600
    assert per_minute(pd.Series({"A": 7.0}), pd.Series({"A": 3500.0}))["A"] == pytest.approx(120.0)
    assert np.isnan(per_minute(pd.Series({"A": 1.0}), pd.Series({"A": 0.0}))["A"])
    assert np.isnan(safe_ratio(pd.Series([1.0]), pd.Series([0.0]))[0])
    st = dist_stats(e["iki_ms"], e["message_id"], "iki")
    assert st.loc["B", "iki_mean"] == pytest.approx(200.0)
    assert st.loc["B", "iki_median"] == pytest.approx(150.0)
    assert st.loc["B", "iki_iqr"] == pytest.approx(75.0)
    assert st.loc["B", "iki_mad"] == pytest.approx(0.0)
