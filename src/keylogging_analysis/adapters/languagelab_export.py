"""LanguageLab analyst export (2025-26 onwards): messages.csv + textarea_states.csv.

See the export's own notice (notice_analyste.md). Only states whose link to a
message is unique (message_link_status == 'matched_unique') are used.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from ..schema import KeylogData
from .base import AdapterResult, apply_filters, require_file

RENAME = {
    "chat_message_id": "message_id",
    "student_id": "user_id",
    "conversation_id": "session_id",
    "scenario_name": "task_id",
    "student_message": "sent_text",
    "student_response_delay_s": "response_delay_s",
}
PASSTHROUGH = [
    "wave_number", "section", "class_name", "class_id", "scenario_id",
    "message_index_in_conversation", "conversation_is_finished", "student_message_is_empty",
    "trace_coverage", "client_key_is_unique_to_message", "student_finished_at_utc",
    "textarea_state_count_for_client_key", "last_state_by_client_time_matches_student_message",
]
STATE_COLUMNS = ["textarea_state_id", "chat_message_id", "message_link_status",
                 "text_content", "client_timestamp_raw"]


def _read_csv(path: Path, columns, types, *, strings_can_be_null: bool = False) -> pd.DataFrame:
    # strings_can_be_null distinguishes an unquoted empty field (the export's
    # psql-NULL convention: sent text is unknown) from a quoted "" (a message
    # that really is empty), instead of collapsing both to "". It only matters
    # for messages.csv's sent_text; state text_content is read with the
    # default (an empty text state is a real, meaningful empty string).
    table = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(include_columns=columns, include_missing_columns=True,
                                             column_types=types,
                                             strings_can_be_null=strings_can_be_null,
                                             quoted_strings_can_be_null=False),
    )
    return table.to_pandas()


def load(path: Path, filters=None) -> AdapterResult:
    path = Path(path)
    msg_file = require_file(path / "messages.csv")
    state_file = require_file(path / "textarea_states.csv")
    counts = {}

    msg_cols = list(RENAME) + PASSTHROUGH
    raw = _read_csv(msg_file, msg_cols, {c: pa.string() for c in msg_cols if c != "student_response_delay_s"}
                    | {"student_response_delay_s": pa.float64()}, strings_can_be_null=True)
    counts["messages_read"] = len(raw)
    raw = apply_filters(raw, filters)
    counts["messages_after_filter"] = len(raw)
    messages = raw.rename(columns=RENAME)[list(RENAME.values()) + PASSTHROUGH].reset_index(drop=True)

    st = _read_csv(state_file, STATE_COLUMNS,
                   {"textarea_state_id": pa.int64(), "chat_message_id": pa.string(),
                    "message_link_status": pa.string(), "text_content": pa.string(),
                    "client_timestamp_raw": pa.float64()})
    counts["states_read"] = len(st)
    unique = st["message_link_status"].eq("matched_unique").fillna(False)
    counts["states_not_matched_unique"] = int((~unique).sum())
    st = st[unique]
    in_scope = st["chat_message_id"].isin(messages["message_id"])
    counts["states_outside_filter"] = int((~in_scope).sum())
    st = st[in_scope]
    counts["states_kept"] = len(st)

    events = pd.DataFrame({
        "message_id": st["chat_message_id"],
        "t_ms": st["client_timestamp_raw"],
        "text": st["text_content"],
        "seq": st["textarea_state_id"],
    }).reset_index(drop=True)
    return AdapterResult(KeylogData(events, messages), [msg_file, state_file], counts)
