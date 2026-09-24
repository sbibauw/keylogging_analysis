"""Hand-worked fixtures shared by the engine tests.

MSG_A: typing with pauses and a two-character correction at the leading edge.
MSG_B: a 3-character bulk insert, a mid-text replacement, one no-change state.
MSG_C: a single event.
Message D (in abc_data) has no events at all.
Expected values for each are worked out in the tests that use them.
"""
import math

import pandas as pd

from keylogging_analysis.schema import KeylogData, validate

MSG_A = [(1000, "H"), (1100, "Hi"), (1300, "Hi "), (3500, "Hi t"), (3600, "Hi th"),
         (3700, "Hi t"), (3800, "Hi "), (4100, "Hi y"), (4500, "Hi yo")]
MSG_B = [(0, "I"), (150, "I am"), (300, "I sm"), (450, "I sm"), (600, "I sm!")]
MSG_C = [(500, "x")]


def events_of(*messages):
    """events_of(("A", MSG_A), ("B", MSG_B)) -> events frame, seq in listed order."""
    rows, seq = [], 0
    for mid, steps in messages:
        for t, text in steps:
            rows.append({"message_id": mid, "t_ms": float(t), "text": text, "seq": seq})
            seq += 1
    return pd.DataFrame(rows, columns=["message_id", "t_ms", "text", "seq"])


def messages_of(ids, **cols):
    df = pd.DataFrame({"message_id": list(ids), "user_id": "u1", "session_id": "s1"})
    for k, v in cols.items():
        df[k] = v
    return df


def abc_data():
    return KeylogData(
        events=events_of(("A", MSG_A), ("B", MSG_B), ("C", MSG_C)),
        messages=messages_of(["A", "B", "C", "D"],
                             sent_text=["Hi yo", "I am!", None, None],
                             response_delay_s=[10.0, None, None, None]),
    )


def edits_of(*messages, config=None):
    """Validated, cleaned, diffed events for the given (id, steps) messages."""
    from keylogging_analysis.clean import clean_events
    from keylogging_analysis.config import MetricConfig
    from keylogging_analysis.diff import derive_edits

    config = config or MetricConfig()
    data = validate(KeylogData(events_of(*messages), messages_of([m for m, _ in messages])))
    events, _, _ = clean_events(data.events, config)
    return derive_edits(events, config)


def row(df, message_id):
    """One message's metrics as a plain dict (df indexed or keyed by message_id)."""
    if "message_id" in df.columns:
        df = df.set_index("message_id")
    return df.loc[message_id].to_dict()


def isnan(x):
    return x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x))
