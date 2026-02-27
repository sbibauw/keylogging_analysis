"""Smoke tests for the core pipeline: load data, add IKI, add pause, add pburst, add action."""
import pandas as pd
import numpy as np
import pytest
from keylogging_analysis import KeyLoggingDataFrame, __version__


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
    kldf = KeyLoggingDataFrame(data, system="lh")
    return kldf


def test_create_empty():
    kldf = KeyLoggingDataFrame()
    assert kldf.empty


def test_version():
    assert __version__ == "0.0.2"


def test_df_accessible(sample_kldf):
    """The underlying DataFrame should be accessible via .df"""
    assert isinstance(sample_kldf.df, pd.DataFrame)
    assert len(sample_kldf.df) == 10


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


def test_system_metadata_preserved(sample_kldf):
    """system metadata should survive add_iki and other transforms."""
    assert sample_kldf.system == "lh"
    sample_kldf.add_iki(colname="iki")
    assert sample_kldf.system == "lh"
    sample_kldf.add_pause(colname="pause", method="fixed", threshold=200, iki_colname="iki")
    assert sample_kldf.system == "lh"
    sample_kldf.add_action()
    assert sample_kldf.system == "lh"


def test_method_chaining(sample_kldf):
    """add_* methods should return self for chaining."""
    result = sample_kldf.add_iki(colname="iki")
    assert result is sample_kldf
    result2 = sample_kldf.add_action()
    assert result2 is sample_kldf


# ── fluency_metrics tests ────────────────────────────────────────

@pytest.fixture
def fluency_kldf():
    """Two users, two sessions, four messages with typing + deletions."""
    data = {
        "key_id": list(range(1, 25)),
        "message_id": [1]*6 + [2]*6 + [3]*6 + [4]*6,
        "content": [
            # msg 1 — user 1, session 1: type "Helo" then delete "o", type "lo"
            "H", "He", "Hel", "Helo", "Hel", "Hello",
            # msg 2 — user 1, session 1: type "Bon"
            "B", "Bo", "Bon", "Bonj", "Bonjou", "Bonjour",
            # msg 3 — user 2, session 2: type "Hi" then delete, retype
            "H", "Hi", "H", "Hi", "Hi!", "Hi!?",
            # msg 4 — user 2, session 2: straight typing
            "O", "Ou", "Oui", "Oui!", "Oui!!", "Oui!!!",
        ],
        "event_for_message_type": [0]*24,
        "key_time": pd.to_datetime([
            # msg 1
            "2024-01-01 10:00:00.000", "2024-01-01 10:00:00.100",
            "2024-01-01 10:00:00.200", "2024-01-01 10:00:00.350",
            "2024-01-01 10:00:00.600", "2024-01-01 10:00:00.700",
            # msg 2
            "2024-01-01 10:00:01.000", "2024-01-01 10:00:01.120",
            "2024-01-01 10:00:01.230", "2024-01-01 10:00:01.340",
            "2024-01-01 10:00:01.450", "2024-01-01 10:00:01.560",
            # msg 3
            "2024-01-01 11:00:00.000", "2024-01-01 11:00:00.100",
            "2024-01-01 11:00:00.400", "2024-01-01 11:00:00.500",
            "2024-01-01 11:00:00.600", "2024-01-01 11:00:00.700",
            # msg 4
            "2024-01-01 11:00:01.000", "2024-01-01 11:00:01.080",
            "2024-01-01 11:00:01.160", "2024-01-01 11:00:01.240",
            "2024-01-01 11:00:01.320", "2024-01-01 11:00:01.400",
        ]),
        "message_content": ["Hello"]*6 + ["Bonjour"]*6 + ["Hi!?"]*6 + ["Oui!!!"]*6,
        "user_status": ["L"]*24,
        "session_id": [1]*12 + [2]*12,
        "user_id": [10]*12 + [20]*12,
    }
    kldf = KeyLoggingDataFrame(data, system="ll")
    kldf.add_iki(colname="iki")
    kldf.add_pause(colname="pause", method="fixed", threshold=200, iki_colname="iki")
    kldf.add_pburst(pause_colname="pause")
    kldf.add_action(colname="action")
    kldf.add_rburst(colname="rburst", action_colname="action")
    return kldf


def test_fluency_message_shape(fluency_kldf):
    result = fluency_kldf.fluency_metrics(level="message")
    assert len(result) == 4  # 4 messages
    assert "message_id" in result.columns


def test_fluency_message_columns(fluency_kldf):
    result = fluency_kldf.fluency_metrics(level="message")
    expected = [
        "message_id", "user_id", "session_id",
        "iki_mean", "iki_md", "iki_sd", "iki_iqr", "iki_mad",
        "pauses_n", "pause_ratio", "pause_duration_total",
        "pause_duration_mean", "pause_duration_md",
        "pbursts_n", "pburst_length_mean",
        "rbursts_n", "rburst_length_mean",
        "revisions_n", "revision_ratio",
        "keystrokes_n", "final_text_length", "duration_total", "production_rate",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_fluency_session_shape(fluency_kldf):
    result = fluency_kldf.fluency_metrics(level="session")
    assert len(result) == 2  # 2 sessions
    assert "session_id" in result.columns
    assert "messages_n" in result.columns
    assert (result["messages_n"] == 2).all()


def test_fluency_user_shape(fluency_kldf):
    result = fluency_kldf.fluency_metrics(level="user")
    assert len(result) == 2  # 2 users
    assert "user_id" in result.columns
    assert "messages_n" in result.columns


def test_fluency_invalid_level(fluency_kldf):
    with pytest.raises(ValueError, match="Invalid level"):
        fluency_kldf.fluency_metrics(level="invalid")


def test_fluency_missing_columns():
    kldf = KeyLoggingDataFrame({"key_id": [1], "event_for_message_type": [0]}, system="ll")
    with pytest.raises(ValueError, match="Missing required columns"):
        kldf.fluency_metrics()


def test_fluency_spot_check_keystrokes(fluency_kldf):
    result = fluency_kldf.fluency_metrics(level="message")
    # Each message has 6 keystrokes
    assert (result["keystrokes_n"] == 6).all()


def test_fluency_user_id_normalization_lh():
    """LH system uses PERSONA_ID internally but outputs user_id."""
    data = {
        "key_id": list(range(1, 7)),
        "message_id": [1]*6,
        "content": ["H", "He", "Hel", "Hell", "Hello", "Hello!"],
        "event_for_message_type": [0]*6,
        "key_time": pd.to_datetime([
            "2024-01-01 10:00:00.000", "2024-01-01 10:00:00.100",
            "2024-01-01 10:00:00.200", "2024-01-01 10:00:00.300",
            "2024-01-01 10:00:00.400", "2024-01-01 10:00:00.500",
        ]),
        "message_content": ["Hello!"]*6,
        "user_status": ["L"]*6,
        "session_id": [1]*6,
        "PERSONA_ID": [99]*6,
    }
    kldf = KeyLoggingDataFrame(data, system="lh")
    kldf.add_iki(colname="iki")
    kldf.add_pause(colname="pause", method="fixed", threshold=200, iki_colname="iki")
    kldf.add_pburst(pause_colname="pause")
    kldf.add_action(colname="action")
    kldf.add_rburst(colname="rburst", action_colname="action")
    result = kldf.fluency_metrics(level="message")
    assert "user_id" in result.columns
    assert result["user_id"].iloc[0] == 99
