import json

import numpy as np
import pandas as pd
import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.schema import KeylogData, SchemaError, first_of_group, validate
from helpers import events_of, messages_of


def test_config_defaults():
    c = MetricConfig()
    assert c.pause_thresholds_ms == (200, 2000)
    assert c.bulk_insert_min == 3
    assert c.drop_nochange_events is True
    assert " " in c.between_word_chars and "." in c.between_word_chars


def test_config_sorts_thresholds_and_rejects_bad_values():
    assert MetricConfig(pause_thresholds_ms=[2000, 200]).pause_thresholds_ms == (200, 2000)
    with pytest.raises(ValueError):
        MetricConfig(pause_thresholds_ms=())
    with pytest.raises(ValueError):
        MetricConfig(pause_thresholds_ms=(200, 200))
    with pytest.raises(ValueError):
        MetricConfig(pause_thresholds_ms=(0,))
    with pytest.raises(ValueError):
        MetricConfig(bulk_insert_min=1)


def test_config_json_roundtrip(tmp_path):
    c = MetricConfig(pause_thresholds_ms=(300,), bulk_insert_min=4)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(c.to_dict()))
    assert MetricConfig.from_json(p) == c


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="bogus"):
        MetricConfig.from_dict({"bogus": 1})


def test_config_from_dict_partial_uses_defaults():
    assert MetricConfig.from_dict({"bulk_insert_min": 5}) == MetricConfig(bulk_insert_min=5)


def test_validate_coerces_types():
    ev = pd.DataFrame({"message_id": [1, 1], "t_ms": [0, 100], "text": ["a", None], "seq": ["3", "4"],
                       "extra": [1, 2]})
    ms = pd.DataFrame({"message_id": [1], "user_id": [7], "session_id": [9], "response_delay_s": ["2.5"]})
    out = validate(KeylogData(ev, ms))
    assert list(out.events.columns) == ["message_id", "t_ms", "text", "seq"]
    assert out.events["message_id"].tolist() == ["1", "1"]
    assert out.events["t_ms"].dtype == np.float64
    assert out.events["text"].tolist() == ["a", ""]
    assert out.events["seq"].dtype == np.int64
    assert out.messages["user_id"].tolist() == ["7"]
    assert out.messages["response_delay_s"].dtype == np.float64


def test_validate_rejects_missing_columns():
    ev = pd.DataFrame({"message_id": ["a"], "t_ms": [0.0], "seq": [0]})
    with pytest.raises(SchemaError, match="text"):
        validate(KeylogData(ev, messages_of(["a"])))


def test_validate_rejects_duplicate_messages_and_orphan_events():
    with pytest.raises(SchemaError, match="duplicate"):
        validate(KeylogData(events_of(), messages_of(["a", "a"])))
    with pytest.raises(SchemaError, match="unknown message_id"):
        validate(KeylogData(events_of(("zz", [(0, "x")])), messages_of(["a"])))


def test_validate_rejects_missing_time():
    ev = events_of(("a", [(0, "x")]))
    ev.loc[0, "t_ms"] = np.nan
    with pytest.raises(SchemaError, match="t_ms"):
        validate(KeylogData(ev, messages_of(["a"])))


def test_validate_rejects_missing_seq():
    ev = events_of(("a", [(0, "x")]))
    ev.loc[0, "seq"] = np.nan
    with pytest.raises(SchemaError, match="seq"):
        validate(KeylogData(ev, messages_of(["a"])))


def test_validate_coerces_whole_number_float_id_like_int():
    ev = pd.DataFrame({"message_id": [1.0], "t_ms": [0.0], "text": ["a"], "seq": [0]})
    ms = messages_of([1])
    out = validate(KeylogData(ev, ms))
    assert out.events["message_id"].tolist() == ["1"]
    assert out.messages["message_id"].tolist() == ["1"]


@pytest.mark.parametrize("storage", ["pyarrow", "python"])
def test_first_of_group_true_at_each_groups_first_row(storage):
    key = pd.Series(["a", "a", "b", "b", "b", "c"], dtype=pd.StringDtype(storage))
    out = first_of_group(key)
    assert out.tolist() == [True, False, True, False, False, True]
    assert out.dtype == bool


def test_first_of_group_empty_series():
    key = pd.Series([], dtype="string")
    out = first_of_group(key)
    assert out.tolist() == []
    assert out.dtype == bool


def test_validate_rejects_missing_required_ids():
    ev_bad = events_of(("a", [(0, "x")]))
    ev_bad.loc[0, "message_id"] = None
    with pytest.raises(SchemaError, match="message_id"):
        validate(KeylogData(ev_bad, messages_of(["a"])))

    ev = events_of(("a", [(0, "x")]))
    ms_bad = messages_of(["a"])
    ms_bad.loc[0, "session_id"] = None
    with pytest.raises(SchemaError, match="session_id"):
        validate(KeylogData(ev, ms_bad))
