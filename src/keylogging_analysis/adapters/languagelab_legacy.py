"""Legacy LanguageLab format (ll_default.csv): one row per key event with full content."""
from pathlib import Path

import pandas as pd

from ..schema import KeylogData
from .base import AdapterResult, apply_filters, require_file

EPOCH = pd.Timestamp("1970-01-01")


def _ms(series: pd.Series) -> pd.Series:
    """Milliseconds since the epoch; NaN where the timestamp does not parse."""
    t = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    return (t - EPOCH) / pd.Timedelta(milliseconds=1)


def load(path: Path, filters=None) -> AdapterResult:
    path = require_file(Path(path))
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    counts = {"rows_read": len(raw)}
    raw = apply_filters(raw, filters)
    counts["rows_after_filter"] = len(raw)

    messages = (raw.drop_duplicates("message_id")
                .rename(columns={"message_content": "sent_text"})
                [["message_id", "user_id", "session_id", "sent_text", "message_uid",
                  "message_created_at", "user_status"]]
                .reset_index(drop=True))
    typing = raw[raw["event_for_message_type"] == "0"]
    t_ms = _ms(typing["key_time"])
    counts["events_bad_time"] = int(t_ms.isna().sum())      # unparseable key_time: dropped, counted
    typing, t_ms = typing[t_ms.notna()], t_ms[t_ms.notna()]
    counts["events_kept"] = len(typing)
    events = pd.DataFrame({
        "message_id": typing["message_id"],
        "t_ms": t_ms,
        "text": typing["content"].fillna(""),
        "seq": typing["key_id"].astype("int64"),
    }).reset_index(drop=True)
    return AdapterResult(KeylogData(events, messages), [path], counts)
