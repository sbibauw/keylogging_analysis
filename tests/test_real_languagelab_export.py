"""End-to-end run on a real LanguageLab export. Skipped unless KEYLOG_TEST_DATA is set.

    KEYLOG_TEST_DATA=/path/to/export uv run pytest tests/test_real_languagelab_export.py -v -s
"""
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from keylogging_analysis import compute_message_metrics, get_adapter

DATA = os.environ.get("KEYLOG_TEST_DATA")
FILTER = {"class_name": ["eB 2025-2026", "TI 2025-2026"]}
pytestmark = pytest.mark.skipif(not DATA, reason="set KEYLOG_TEST_DATA to a LanguageLab export directory")


@pytest.fixture(scope="module")
def run():
    t0 = time.perf_counter()
    res = get_adapter("languagelab_export")(Path(DATA), FILTER)
    table, report = compute_message_metrics(res.data, adapter_counts=res.counts)
    elapsed = time.perf_counter() - t0
    print(f"\nreal export: {len(table)} messages, {report.n_events_out} events, {elapsed:.1f} s")
    return res, table, report, elapsed


def test_runtime_under_60_seconds(run):
    assert run[3] < 60


def test_one_row_per_filtered_message(run):
    res, table, _, _ = run
    assert len(table) == res.counts["messages_after_filter"]


def test_event_count_matches_export_count_for_unique_keys(run):
    # For messages whose client key is unique, every state of that key is linked
    # to this message, so kept + dropped must equal the export's own count.
    _, table, _, _ = run
    u = table[table["client_key_is_unique_to_message"] == "t"]
    expected = pd.to_numeric(u["textarea_state_count_for_client_key"]).fillna(0).astype(int)
    assert ((u["n_events"] + u["n_nochange_dropped"]) == expected).mean() == 1.0


def test_final_state_agrees_with_export_flag(run):
    # Same ordering rule (client time, then id) as the export's own flag.
    _, table, _, _ = run
    u = table[table["last_state_by_client_time_matches_student_message"].isin(["t", "f"])]
    ours = u["final_matches_sent"].fillna(False).astype(bool)
    theirs = u["last_state_by_client_time_matches_student_message"] == "t"
    assert (ours == theirs).mean() >= 0.999


def test_indicators_are_plausible(run):
    _, table, _, _ = run
    typed = table[table["n_events"] >= 2]
    assert typed["iki_median"].between(30, 2000).mean() > 0.95
    assert (typed["n_bursts_200"] >= 1).all()
    assert (typed["burst_size_max_2000"] >= typed["burst_size_max_200"]).all()
    assert table["has_bulk_insert"].mean() < 0.2
