import json

import pandas as pd
import pytest

from keylogging_analysis.cli import main, parse_filters

MESSAGES_CSV = '''chat_message_id,student_id,conversation_id,scenario_name,student_message,student_response_delay_s,class_name
1,S_a,C_1,T1,"line one
line two, ""quoted""",4.0,eB
2,S_b,C_2,T1,ok,2.0,TI
'''
STATES_CSV = '''textarea_state_id,chat_message_id,message_link_status,text_content,client_timestamp_raw
1,1,matched_unique,l,0
2,1,matched_unique,"line one
line two, ""quoted""",900
3,2,matched_unique,o,0
4,2,matched_unique,ok,120
'''


@pytest.fixture
def export_dir(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    (d / "messages.csv").write_text(MESSAGES_CSV, encoding="utf-8")
    (d / "textarea_states.csv").write_text(STATES_CSV, encoding="utf-8")
    return d


def test_parse_filters():
    assert parse_filters(["class_name=eB 2025-2026", "class_name=TI", "a=b=c"]) == \
        {"class_name": ["eB 2025-2026", "TI"], "a": ["b=c"]}
    with pytest.raises(SystemExit):
        parse_filters(["no_equals_sign"])


def test_cli_writes_csv_and_provenance(export_dir, tmp_path):
    out = tmp_path / "out" / "metrics.csv"
    assert main(["languagelab_export", str(export_dir), "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert len(table) == 2
    prov = json.loads((tmp_path / "out" / "metrics.provenance.json").read_text())
    assert prov["adapter"] == "languagelab_export"
    assert prov["rows_out"] == 2
    assert {i["path"].split("/")[-1] for i in prov["inputs"]} == {"messages.csv", "textarea_states.csv"}
    assert prov["cleaning"]["adapter_counts"]["states_kept"] == 4


def test_cli_roundtrip_text_with_newlines_and_quotes(export_dir, tmp_path):
    out = tmp_path / "m.csv"
    main(["languagelab_export", str(export_dir), "--out", str(out)])
    table = pd.read_csv(out, dtype={"message_id": str})
    sent = table.set_index("message_id").loc["1", "sent_text"]
    assert sent == 'line one\nline two, "quoted"'
    assert bool(table.set_index("message_id").loc["1", "final_matches_sent"]) is True


def test_cli_filter_and_config(export_dir, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"pause_thresholds_ms": [100]}))
    out = tmp_path / "f.csv"
    main(["languagelab_export", str(export_dir), "--out", str(out), "--config", str(cfg),
          "--filter", "class_name=TI"])
    table = pd.read_csv(out)
    assert table["user_id"].tolist() == ["S_b"]
    assert "n_bursts_100" in table.columns


def test_cli_zero_messages_writes_header_only(export_dir, tmp_path):
    out = tmp_path / "z.csv"
    assert main(["languagelab_export", str(export_dir), "--out", str(out),
                 "--filter", "class_name=nobody"]) == 0
    table = pd.read_csv(out)
    assert len(table) == 0 and "n_events" in table.columns
