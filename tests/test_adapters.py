from pathlib import Path

import pandas as pd
import pytest

import keylogging_analysis
from keylogging_analysis.adapters import ADAPTERS, get_adapter
from keylogging_analysis.adapters.base import apply_filters
from keylogging_analysis.schema import validate

DATA = Path(keylogging_analysis.__file__).parent / "data"

MESSAGES_CSV = '''chat_message_id,student_id,wave_number,section,class_name,conversation_id,scenario_id,scenario_name,student_message,student_response_delay_s,client_key_is_unique_to_message,textarea_state_count_for_client_key,last_state_by_client_time_matches_student_message
11,S_a,1,eB,eB 2025-2026,C_1,7,"ACTIVITY 1: \r
MEETING",Hi,12.5,t,3,t
12,S_a,1,eB,eB 2025-2026,C_1,7,"ACTIVITY 1: \r
MEETING",,,t,0,
13,S_b,,,Demo !,C_2,9,Demo,"x, ""y""",3.0,t,1,t
'''

STATES_CSV = '''textarea_state_id,chat_message_id,message_link_status,text_content,client_timestamp_raw
101,11,matched_unique,H,1000.5
102,11,matched_unique,Hi,1100.0
103,11,matched_unique,Hi,1150.0
104,,matched_ambiguous,zz,5.0
105,13,matched_unique,"x, ""y""",7.0
'''


@pytest.fixture
def export_dir(tmp_path):
    (tmp_path / "messages.csv").write_text(MESSAGES_CSV, encoding="utf-8")
    (tmp_path / "textarea_states.csv").write_text(STATES_CSV, encoding="utf-8")
    return tmp_path


def test_registry():
    assert set(ADAPTERS) == {"languagelab_export", "languagelab_legacy", "language_hero"}
    with pytest.raises(ValueError, match="languagelab_export"):
        get_adapter("nope")


def test_apply_filters():
    df = pd.DataFrame({"a": ["x", "y", "z"], "n": [1, 2, 3]})
    assert apply_filters(df, None).equals(df)
    assert apply_filters(df, {"a": ["x", "z"]})["a"].tolist() == ["x", "z"]
    assert apply_filters(df, {"n": ["2"]})["n"].tolist() == [2]


def test_filter_unknown_column_raises():
    with pytest.raises(ValueError, match="no_such_col"):
        apply_filters(pd.DataFrame({"a": [1]}), {"no_such_col": ["1"]})


def test_languagelab_export_mapping(export_dir):
    res = get_adapter("languagelab_export")(export_dir)
    data = validate(res.data)
    ms = data.messages.set_index("message_id")
    assert ms.loc["11", "user_id"] == "S_a"
    assert ms.loc["11", "session_id"] == "C_1"
    assert ms.loc["11", "task_id"] == "ACTIVITY 1: \r\nMEETING"
    assert ms.loc["11", "sent_text"] == "Hi"
    assert ms.loc["11", "response_delay_s"] == 12.5
    assert ms.loc["13", "sent_text"] == 'x, "y"'
    assert ms.loc["11", "class_name"] == "eB 2025-2026"          # passthrough
    ev = data.events
    assert ev["message_id"].tolist() == ["11", "11", "11", "13"]
    assert ev["seq"].tolist() == [101, 102, 103, 105]
    assert ev["t_ms"].tolist() == [1000.5, 1100.0, 1150.0, 7.0]
    assert res.counts == {"messages_read": 3, "messages_after_filter": 3, "states_read": 5,
                          "states_not_matched_unique": 1, "states_outside_filter": 0,
                          "states_kept": 4}
    assert [p.name for p in res.inputs] == ["messages.csv", "textarea_states.csv"]


def test_languagelab_export_filter(export_dir):
    res = get_adapter("languagelab_export")(export_dir, {"class_name": ["eB 2025-2026"]})
    assert sorted(res.data.messages["message_id"].tolist()) == ["11", "12"]
    assert res.counts["states_outside_filter"] == 1
    assert res.counts["states_kept"] == 3


def test_languagelab_export_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="textarea_states.csv"):
        (tmp_path / "messages.csv").write_text(MESSAGES_CSV)
        get_adapter("languagelab_export")(tmp_path)


def test_languagelab_legacy_default_dataset():
    res = get_adapter("languagelab_legacy")(DATA / "ll_default.csv")
    data = validate(res.data)
    assert sorted(data.messages["message_id"].tolist()) == ["1", "2", "3"]
    ev = data.events
    assert (ev["message_id"] == "3").sum() == 12           # the type-5 start marker is excluded
    first_1 = ev[ev["message_id"] == "1"]
    assert first_1["t_ms"].diff().iloc[1] == pytest.approx(123.0)   # 07.999 -> 08.122
    assert data.messages.set_index("message_id").loc["3", "sent_text"] == "How are you?"


def test_language_hero_default_dataset():
    res = get_adapter("language_hero")(DATA / "lh_default.csv")
    data = validate(res.data)
    assert data.messages["user_id"].notna().all()           # USER_ID, not the empty PERSONA_ID
    assert len(data.messages) == data.messages["message_id"].nunique()
    # 521 203 rows have event_for_message_type == 0; any with an unparseable time are counted
    assert res.counts["events_kept"] + res.counts["events_bad_time"] == 521203
