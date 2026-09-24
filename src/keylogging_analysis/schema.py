"""Canonical tables: one row per text state (events) and one per message."""
from dataclasses import dataclass

import pandas as pd

GROUP = "message_id"
EVENT_COLUMNS = ["message_id", "t_ms", "text", "seq"]
MESSAGE_REQUIRED = ["message_id", "user_id", "session_id"]


class SchemaError(ValueError):
    """Input does not satisfy the canonical format."""


@dataclass
class KeylogData:
    events: pd.DataFrame
    messages: pd.DataFrame


def _require(df: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing required column(s): {', '.join(missing)}")


def validate(data: KeylogData) -> KeylogData:
    """Check the canonical contract and return a type-coerced copy."""
    ev, ms = data.events, data.messages
    _require(ev, EVENT_COLUMNS, "events")
    _require(ms, MESSAGE_REQUIRED, "messages")

    ev = ev[EVENT_COLUMNS].copy()
    ev["message_id"] = ev["message_id"].astype("string")
    ev["t_ms"] = pd.to_numeric(ev["t_ms"]).astype("float64")
    ev["text"] = ev["text"].astype("string").fillna("")
    ev["seq"] = pd.to_numeric(ev["seq"]).astype("int64")
    if ev["t_ms"].isna().any():
        raise SchemaError(f"{int(ev['t_ms'].isna().sum())} events have a missing t_ms")

    ms = ms.copy()
    for c in MESSAGE_REQUIRED:
        ms[c] = ms[c].astype("string")
    for c in ("task_id", "sent_text"):
        if c in ms.columns:
            ms[c] = ms[c].astype("string")
    if "response_delay_s" in ms.columns:
        ms["response_delay_s"] = pd.to_numeric(ms["response_delay_s"]).astype("float64")

    dup = ms["message_id"][ms["message_id"].duplicated()]
    if len(dup):
        raise SchemaError(f"duplicate message_id in messages: {dup.unique()[:5].tolist()}")
    orphan = ~ev["message_id"].isin(ms["message_id"])
    if orphan.any():
        raise SchemaError(f"{int(orphan.sum())} events reference unknown message_id, "
                          f"e.g. {ev.loc[orphan, 'message_id'].unique()[:5].tolist()}")
    return KeylogData(events=ev.reset_index(drop=True), messages=ms.reset_index(drop=True))
