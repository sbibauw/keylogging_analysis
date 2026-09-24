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


def first_of_group(key: pd.Series) -> pd.Series:
    """True at each group's first row.

    ``key`` is the ``string``-dtype id column, and ``key.ne(key.shift())`` is
    <NA> at row 0 (no previous value to compare against) instead of True,
    under either string storage:

    - pyarrow storage (pandas 3.x's default): the result is an arrow
      ``bool[pyarrow]`` Series, and that extension dtype does not support
      ``cumsum`` at all, so the very first call raises ``TypeError``.
    - python storage (``mode.string_storage="python"``): the result is a nullable
      ``boolean`` Series, whose ``cumsum`` *does* run, but the leading <NA>
      propagates into the cumulative burst/run id, and pandas' ``groupby``
      drops NA-keyed rows by default — so the first event of the first
      message is silently dropped from its burst instead of raising.

    Row 0 has no previous value, which unambiguously makes it a group start,
    so NA is filled True; the plain numpy bool cast then makes the result
    cumsum-able (and dtype-stable) under both storages.
    """
    return key.ne(key.shift()).fillna(True).astype(bool)


def _require(df: pd.DataFrame, columns, name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing required column(s): {', '.join(missing)}")


def _to_id_string(s: pd.Series) -> pd.Series:
    """Coerce an id column to string, treating whole-number floats as ints.

    Without this, an id that happens to arrive as float64 (e.g. from an Excel
    read) stringifies as "1.0" while the same id as int64 stringifies as "1",
    so identical ids compare unequal across events/messages and look orphaned.
    """
    if pd.api.types.is_float_dtype(s):
        non_na = s.dropna()
        if len(non_na) and (non_na == non_na.round()).all():
            s = s.astype("Int64")
    return s.astype("string")


def _require_no_na(df: pd.DataFrame, column: str, name: str) -> None:
    n = int(df[column].isna().sum())
    if n:
        raise SchemaError(f"{name}.{column} has {n} missing value(s)")


def validate(data: KeylogData) -> KeylogData:
    """Check the canonical contract and return a type-coerced copy."""
    ev, ms = data.events, data.messages
    _require(ev, EVENT_COLUMNS, "events")
    _require(ms, MESSAGE_REQUIRED, "messages")

    ev = ev[EVENT_COLUMNS].copy()
    ev["message_id"] = _to_id_string(ev["message_id"])
    ev["t_ms"] = pd.to_numeric(ev["t_ms"]).astype("float64")
    ev["text"] = ev["text"].astype("string").fillna("")
    ev["seq"] = pd.to_numeric(ev["seq"])
    if ev["seq"].isna().any():
        raise SchemaError(f"{int(ev['seq'].isna().sum())} events have a missing seq")
    ev["seq"] = ev["seq"].astype("int64")
    if ev["t_ms"].isna().any():
        raise SchemaError(f"{int(ev['t_ms'].isna().sum())} events have a missing t_ms")
    _require_no_na(ev, "message_id", "events")

    ms = ms.copy()
    for c in MESSAGE_REQUIRED:
        ms[c] = _to_id_string(ms[c])
        _require_no_na(ms, c, "messages")
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
