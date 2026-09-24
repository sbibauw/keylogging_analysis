import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.metrics.product import product_metrics
from helpers import MSG_A, MSG_B, MSG_C, edits_of, isnan, messages_of, row

MSGS = messages_of(["A", "B", "C"], sent_text=["Hi yo", "I am!", None],
                   response_delay_s=[10.0, None, None])

# A: 9 events, 7 inserts / 2 deletes, 7 chars in / 2 out, final "Hi yo" (5), span 3500 ms
#    cpm_product = 5 / (3500/60000) ; cpm_process = 7 / (3500/60000) = 120
#    with a 10 s response delay: 5 / (10/60) = 30 ; 7 / (10/60) = 42
# B: I, I am (bulk, 3 chars), I sm (replace), I sm! -> 6 chars in, 1 out, final 5, span 600
#    sent "I am!" differs from the final state "I sm!"


def _validated(msgs):
    """Messages with the types validate() gives them (as compute_message_metrics passes them)."""
    from keylogging_analysis.schema import KeylogData, validate
    from helpers import events_of
    return validate(KeylogData(events_of(), msgs)).messages


def test_product_message_a():
    r = row(product_metrics(edits_of(("A", MSG_A)), _validated(MSGS), MetricConfig()), "A")
    assert (r["n_events"], r["n_insert"], r["n_delete"], r["n_replace"]) == (9, 7, 2, 0)
    assert (r["chars_inserted"], r["chars_deleted"], r["final_length"]) == (7, 2, 5)
    assert r["process_product_ratio"] == pytest.approx(1.4)
    assert r["cpm_product"] == pytest.approx(5 * 60000 / 3500)
    assert r["cpm_process"] == pytest.approx(120.0)
    assert r["cpm_product_with_thinking"] == pytest.approx(30.0)
    assert r["cpm_process_with_thinking"] == pytest.approx(42.0)
    assert (r["n_bulk_inserts"], r["chars_bulk_inserted"], r["has_bulk_insert"]) == (0, 0, False)
    assert r["final_matches_sent"] == True  # noqa: E712


def test_product_message_b():
    r = row(product_metrics(edits_of(("B", MSG_B)), _validated(MSGS), MetricConfig()), "B")
    assert (r["n_events"], r["n_insert"], r["n_delete"], r["n_replace"]) == (4, 3, 0, 1)
    assert (r["chars_inserted"], r["chars_deleted"], r["final_length"]) == (6, 1, 5)
    assert r["process_product_ratio"] == pytest.approx(1.2)
    assert r["cpm_product"] == pytest.approx(500.0)
    assert r["cpm_process"] == pytest.approx(600.0)
    assert isnan(r["cpm_product_with_thinking"]) and isnan(r["cpm_process_with_thinking"])
    assert (r["n_bulk_inserts"], r["chars_bulk_inserted"], r["has_bulk_insert"]) == (1, 3, True)
    assert r["final_matches_sent"] == False  # noqa: E712


def test_product_single_event_and_unknown_sent_text():
    r = row(product_metrics(edits_of(("C", MSG_C)), _validated(MSGS), MetricConfig()), "C")
    assert (r["n_events"], r["final_length"]) == (1, 1)
    assert isnan(r["cpm_product"]) and isnan(r["cpm_process"])
    assert isnan(r["final_matches_sent"])


def test_single_bulk_event_message():
    msgs = _validated(messages_of(["P"], sent_text=["pasted whole message"]))
    r = row(product_metrics(edits_of(("P", [(0, "pasted whole message")])), msgs, MetricConfig()), "P")
    assert (r["n_bulk_inserts"], r["chars_bulk_inserted"], r["has_bulk_insert"]) == (1, 20, True)
    assert isnan(r["cpm_product"])          # span 0 -> NaN, never inf
    assert r["final_matches_sent"] == True  # noqa: E712


def test_bulk_threshold_follows_config():
    msgs = _validated(messages_of(["Q"]))
    e = edits_of(("Q", [(0, "a"), (10, "abc")]), config=MetricConfig(bulk_insert_min=3))
    assert row(product_metrics(e, msgs, MetricConfig(bulk_insert_min=3)), "Q")["n_bulk_inserts"] == 0
    e = edits_of(("Q", [(0, "a"), (10, "abcd")]))
    assert row(product_metrics(e, msgs, MetricConfig()), "Q")["n_bulk_inserts"] == 1


def test_product_without_optional_message_columns():
    msgs = _validated(messages_of(["A"]))
    r = row(product_metrics(edits_of(("A", MSG_A)), msgs, MetricConfig()), "A")
    assert isnan(r["cpm_product_with_thinking"])
    assert isnan(r["final_matches_sent"])
