import numpy as np
import pytest

from keylogging_analysis import KeylogData, MetricConfig, compute_message_metrics
from helpers import abc_data, events_of, isnan, messages_of, row


def test_one_row_per_message_including_empty():
    table, report = compute_message_metrics(abc_data())
    assert table["message_id"].tolist() == ["A", "B", "C", "D"]
    assert report.n_messages == 4
    assert (report.n_events_in, report.n_nochange_dropped, report.n_events_out) == (15, 1, 14)


def test_column_order_messages_first():
    table, _ = compute_message_metrics(abc_data())
    cols = table.columns.tolist()
    assert cols[:5] == ["message_id", "user_id", "session_id", "sent_text", "response_delay_s"]
    assert cols.index("n_events") < cols.index("typing_span_ms") < cols.index("n_bursts_200") \
        < cols.index("n_rbursts") < cols.index("n_time_regressions")
    assert cols[-2:] == ["n_time_regressions", "n_nochange_dropped"]


def test_values_flow_through_from_each_module():
    table, _ = compute_message_metrics(abc_data())
    a, b = row(table, "A"), row(table, "B")
    assert a["n_events"] == 9 and a["iki_mean"] == pytest.approx(437.5)
    assert a["n_bursts_200"] == 5 and a["n_rbursts"] == 1
    assert a["cpm_process_with_thinking"] == pytest.approx(42.0)
    assert b["has_bulk_insert"] is True or b["has_bulk_insert"] == True  # noqa: E712
    assert b["n_nochange_dropped"] == 1


def test_message_without_events():
    table, _ = compute_message_metrics(abc_data())
    d = row(table, "D")
    for c in ["n_events", "n_insert", "chars_inserted", "n_pauses_200", "n_bursts_200",
              "n_rbursts", "n_revisions", "n_bulk_inserts", "n_time_regressions",
              "n_nochange_dropped"]:
        assert d[c] == 0, c
    assert d["has_bulk_insert"] == False  # noqa: E712
    # spec §5: a message without events (fewer than 2, per §4.1.3) has NaN for
    # everything but the explicit counts/flags above -- including pause_time_ms,
    # not just typing_span_ms.
    for c in ["final_length", "typing_span_ms", "iki_median", "cpm_product",
              "burst_size_max_200", "pause_time_ms_200"]:
        assert isnan(d[c]), c


def test_all_nochange_message_behaves_like_empty():
    data = KeylogData(events_of(("Z", [(0, ""), (10, ""), (20, "")]), ("K", [(0, "k")])),
                      messages_of(["Z", "K"]))
    table, _ = compute_message_metrics(data)
    z = row(table, "Z")
    assert z["n_events"] == 0 and z["n_nochange_dropped"] == 3
    assert isnan(z["typing_span_ms"])


def test_count_columns_are_int64():
    table, _ = compute_message_metrics(abc_data())
    for c in ["n_events", "n_bursts_2000", "n_rbursts", "n_time_regressions"]:
        assert table[c].dtype == np.int64, c


def test_custom_config_changes_columns():
    table, _ = compute_message_metrics(abc_data(), MetricConfig(pause_thresholds_ms=(500,)))
    assert "n_bursts_500" in table.columns and "n_bursts_200" not in table.columns


def test_no_events_at_all():
    table, report = compute_message_metrics(KeylogData(events_of(), messages_of(["X", "Y"])))
    assert table["n_events"].tolist() == [0, 0]
    assert report.n_events_in == 0
