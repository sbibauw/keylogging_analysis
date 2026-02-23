"""Smoke tests for the core pipeline: load data, add IKI, add pause, add pburst, add action."""
import pandas as pd
import numpy as np
import pytest
from keylogging_analysis import KeyLoggingDataFrame


@pytest.fixture
def sample_kldf():
    """Create a minimal KeyLoggingDataFrame with synthetic data for 2 messages."""
    data = {
        "key_id": list(range(1, 11)),
        "message_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "content": ["H", "He", "Hel", "Hell", "Hello", "B", "Bo", "Bon", "Bonj", "Bonjour"],
        "event_for_message_type": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "key_time": pd.to_datetime([
            "2024-01-01 10:00:00.000",
            "2024-01-01 10:00:00.100",
            "2024-01-01 10:00:00.200",
            "2024-01-01 10:00:00.350",
            "2024-01-01 10:00:00.500",
            "2024-01-01 10:00:01.000",
            "2024-01-01 10:00:01.120",
            "2024-01-01 10:00:01.230",
            "2024-01-01 10:00:01.800",
            "2024-01-01 10:00:01.900",
        ]),
        "message_content": ["Hello", "Hello", "Hello", "Hello", "Hello",
                           "Bonjour", "Bonjour", "Bonjour", "Bonjour", "Bonjour"],
        "user_status": ["L"] * 10,
        "session_id": [1] * 10,
    }
    kldf = KeyLoggingDataFrame(data)
    kldf.system = "lh"
    return kldf


def test_create_empty():
    kldf = KeyLoggingDataFrame()
    assert kldf.empty


def test_add_iki(sample_kldf):
    sample_kldf.add_iki(colname="iki")
    assert "iki" in sample_kldf.columns
    # First key of each message should be NaN
    msg1 = sample_kldf[sample_kldf["message_id"] == 1].sort_values("key_time")
    assert pd.isna(msg1.iloc[0]["iki"])
    # Second key should have a valid IKI
    assert msg1.iloc[1]["iki"] > 0


def test_add_pause_fixed(sample_kldf):
    sample_kldf.add_iki(colname="iki")
    sample_kldf.add_pause(colname="pause", method="fixed", threshold=200, iki_colname="iki")
    assert "pause" in sample_kldf.columns
    # Check that pause values are boolean or NaN
    non_null = sample_kldf["pause"].dropna()
    assert non_null.dtype == "boolean" or set(non_null.unique()).issubset({True, False})


def test_add_pburst(sample_kldf):
    sample_kldf.add_iki(colname="iki")
    sample_kldf.add_pause(colname="pause", method="fixed", threshold=200, iki_colname="iki")
    sample_kldf.add_pburst(pause_colname="pause")
    assert "pburst" in sample_kldf.columns
    assert set(sample_kldf["pburst"].dropna().unique()).issubset({"B", "I", "O"})


def test_add_action(sample_kldf):
    sample_kldf.add_action()
    assert "action" in sample_kldf.columns
    # Most actions should be "addition" since content grows
    actions = sample_kldf["action"].dropna()
    assert "addition" in actions.values


def test_add_iki_duplicate_colname_raises(sample_kldf):
    sample_kldf.add_iki(colname="iki")
    with pytest.raises(ValueError, match="already exists"):
        sample_kldf.add_iki(colname="iki")
