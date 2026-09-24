# Text-State Metrics Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vectorised engine to `keylogging_analysis` that turns "full text at each event" logs into one row of fluency indicators per message, and wire it into the EPHEC R pipeline for cohort 2025-26.

**Architecture:** Adapters turn each source format into a canonical `KeylogData(events, messages)`. The engine validates, cleans, derives one edit per event (what changed, where, after how long), then four independent metric modules each return a frame indexed by `message_id`; an assembly step joins them onto the messages. A CLI writes the table plus a provenance JSON. The legacy `KeyLoggingDataFrame` is not touched.

**Tech Stack:** Python ≥3.10 (dev on 3.14), pandas ≥2.2 (dev on 3.0.6), numpy ≥2, pyarrow ≥17 (CSV reading), pytest, uv. R 4.6 (EPHEC side).

**Spec:** `docs/superpowers/specs/2026-09-24-text-state-engine-design.md` — read it before any task. Indicator names and definitions there are normative.

## Execution waves

Tasks in the same wave touch disjoint files and can run in parallel, each in its own git worktree branched from `engine-v0.1`, merged back into `engine-v0.1` after review.

| Wave | Tasks | Notes |
|---|---|---|
| 1 (serial) | 1 → 2 | foundation: every later task imports these |
| 2 (parallel) | 3, 4, 5, 6, 7 | timing, bursts, product, adapters, provenance |
| 3 (serial) | 8 → 9 | assembly + CLI, then real data + performance |
| 4 (serial) | 10 → 11 | EPHEC integration, then release docs |

Package repo: `/Users/serge/dev-gitonly/keylogging_analysis` (branch `engine-v0.1`).
EPHEC repo: `/Users/serge/Library/CloudStorage/Dropbox/dev/LanguageLab/researchdata/ephec-bertrand` (work on a new branch `text-state-metrics`, never on `main`).

## Global Constraints

- Run everything through uv from the package root: `uv run pytest ...`, `uv run python ...`. Add `--offline` if the network is unavailable; every dependency is in the uv cache.
- Runtime dependencies are exactly `pandas>=2.2`, `numpy>=2`, `pyarrow>=17`. Do not add any other runtime dependency. Dev dependency: `pytest`.
- `requires-python = ">=3.10"`: no syntax newer than 3.10.
- Code must behave the same under pandas 2.2 and 3.x: convert text columns explicitly with `.astype("string")` and never rely on the default string dtype.
- Do not modify `src/keylogging_analysis/classes.py`, `src/keylogging_analysis/help_functions.py`, or the legacy tests in `tests/test_smoke.py` (except the version assertion in Task 11).
- Real study data never enters this repo, including in test fixtures, docstrings or commit messages. Tests on real data read the directory given by `KEYLOG_TEST_DATA` and are skipped when it is unset.
- Indicator column names are exactly those of spec §5; θ-dependent names end in `_<threshold>` (e.g. `n_bursts_200`).
- Defaults (spec §6/§12): `pause_thresholds_ms = (200, 2000)`, `bulk_insert_min = 3`, `drop_nochange_events = True`, `between_word_chars = " \t\r\n.,;:!?\"'()-"`.
- A pause is `iki_ms >= θ` (inclusive).
- Per-minute rates and ratios are NaN, never ±inf, when the denominator is 0 or missing.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MZtzfSA5GFrTSaVzBP9f6x
  ```
- Never push, tag-push, or merge to `main` in either repo. The human does that.

## Review Focus

1. **A message whose states are all identical** (every event is a no-change): all its events are dropped. It must come out like a message with no events (counts 0, rest NaN), not crash or vanish. → Task 8, `test_all_nochange_message_behaves_like_empty`.
2. **Non-ASCII text** (é, ’, emoji): the diff must count characters in Python code points, and pause location must classify `’` as a letter (not in the separator set) without errors. → Task 2, `test_edit_between_non_ascii`.
3. **A filter that matches nothing, or names an unknown column**: an unknown column is an error naming the column; zero matches gives a valid, header-only CSV. → Task 6, `test_filter_unknown_column_raises`, and Task 8, `test_cli_zero_messages_writes_header_only`.
4. **A whole message pasted in one event**: flagged as bulk; span is 0, so rates are NaN (not inf); one burst of size 1. → Task 5, `test_single_bulk_event_message`.
5. **Text containing newlines, quotes and commas** passed through (`sent_text`): the CSV must round-trip exactly, because R reads it next. → Task 8, `test_cli_roundtrip_text_with_newlines_and_quotes`.

---

### Task 1: Configuration, schema, test helpers

**Files:**
- Create: `src/keylogging_analysis/config.py`
- Create: `src/keylogging_analysis/schema.py`
- Create: `tests/helpers.py`
- Test: `tests/test_config_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MetricConfig` (frozen dataclass): fields `pause_thresholds_ms: tuple[int, ...]`, `bulk_insert_min: int`, `drop_nochange_events: bool`, `between_word_chars: str`; methods `to_dict() -> dict`, `MetricConfig.from_dict(d: dict) -> MetricConfig`, `MetricConfig.from_json(path) -> MetricConfig`.
  - `KeylogData` (dataclass): `events: pd.DataFrame`, `messages: pd.DataFrame`.
  - `SchemaError(ValueError)`.
  - `validate(data: KeylogData) -> KeylogData`: returns a coerced copy. `events` has exactly the columns `message_id` (string), `t_ms` (float64), `text` (string, no NA), `seq` (int64). `messages` has `message_id`, `user_id`, `session_id` as string plus any other columns; `task_id`/`sent_text` are string if present; `response_delay_s` is float64 if present.
  - `GROUP = "message_id"` in `schema.py`.
  - `tests/helpers.py`: `MSG_A`, `MSG_B`, `MSG_C`, `events_of(*messages)`, `messages_of(ids, **cols)`, `abc_data()`, `edits_of(*messages, config=None)`, `row(df, message_id) -> dict`. `edits_of` imports `clean_events`/`derive_edits` from Task 2 lazily (inside the function), so this file is importable after Task 1.

- [ ] **Step 1: Write the failing tests**

`tests/test_config_schema.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.schema import KeylogData, SchemaError, validate
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
```

`tests/helpers.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_schema.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'keylogging_analysis.config'`.

- [ ] **Step 3: Implement `config.py`**

```python
"""Metric configuration: every analytic choice that changes an indicator's value."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class MetricConfig:
    pause_thresholds_ms: tuple = (200, 2000)
    bulk_insert_min: int = 3
    drop_nochange_events: bool = True
    between_word_chars: str = " \t\r\n.,;:!?\"'()-"

    def __post_init__(self):
        th = tuple(int(x) for x in self.pause_thresholds_ms)
        if not th:
            raise ValueError("pause_thresholds_ms must not be empty")
        if any(x <= 0 for x in th):
            raise ValueError(f"pause thresholds must be positive, got {th}")
        if len(set(th)) != len(th):
            raise ValueError(f"pause thresholds must be distinct, got {th}")
        object.__setattr__(self, "pause_thresholds_ms", tuple(sorted(th)))
        if int(self.bulk_insert_min) < 2:
            raise ValueError("bulk_insert_min must be >= 2 (a 1-character insert is ordinary typing)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pause_thresholds_ms"] = list(self.pause_thresholds_ms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MetricConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        d = dict(d)
        if "pause_thresholds_ms" in d:
            d["pause_thresholds_ms"] = tuple(d["pause_thresholds_ms"])
        return cls(**d)

    @classmethod
    def from_json(cls, path) -> "MetricConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
```

- [ ] **Step 4: Implement `schema.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_schema.py -v`
Expected: 9 passed. Then `uv run pytest -q`: all tests pass (18 legacy + 9 new).

- [ ] **Step 6: Commit**

```bash
git add src/keylogging_analysis/config.py src/keylogging_analysis/schema.py tests/helpers.py tests/test_config_schema.py
git commit -m "feat: MetricConfig and canonical KeylogData schema with validation"
```

---

### Task 2: Cleaning, edit derivation, shared metric helpers

**Files:**
- Create: `src/keylogging_analysis/clean.py`
- Create: `src/keylogging_analysis/diff.py`
- Create: `src/keylogging_analysis/metrics/__init__.py` (docstring only in this task)
- Create: `src/keylogging_analysis/metrics/_common.py`
- Test: `tests/test_clean_diff.py`

**Interfaces:**
- Consumes: `MetricConfig`, `KeylogData`, `validate`, `GROUP` (Task 1).
- Produces:
  - `CleaningReport` (dataclass): `n_messages: int`, `n_events_in: int`, `n_nochange_dropped: int`, `n_messages_with_time_regression: int`, `n_events_out: int`, `adapter_counts: dict[str, int]`; `to_dict() -> dict`.
  - `clean_events(events: pd.DataFrame, config: MetricConfig) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]`: returns (cleaned events sorted by `message_id, t_ms, seq` with a fresh RangeIndex; per-message flags indexed by `message_id` with int64 columns `n_time_regressions`, `n_nochange_dropped`, covering every message that had events *before* cleaning; report).
  - `edit_between(prev: str, cur: str) -> tuple[int, int, int]` returning `(pos, n_del, n_ins)`.
  - `derive_edits(events: pd.DataFrame, config: MetricConfig) -> pd.DataFrame`: input is `clean_events` output; returns it plus columns `pos` (int64), `n_ins` (int64), `n_del` (int64), `op` (string: insert/delete/replace/none), `at_end` (bool), `prev_char` (string, "" at pos 0), `ins_first` (string, "" when nothing inserted), `iki_ms` (float64, NaN on each message's first event), `bulk` (bool).
  - `metrics/_common.py`: `span_ms(edits) -> pd.Series`, `per_minute(x, span) -> pd.Series`, `safe_ratio(num, den) -> pd.Series`, `dist_stats(values, keys, prefix) -> pd.DataFrame` (columns `<prefix>_mean/_median/_sd/_iqr/_mad`).

- [ ] **Step 1: Write the failing tests**

`tests/test_clean_diff.py`:

```python
import numpy as np
import pandas as pd
import pytest

from keylogging_analysis.clean import clean_events
from keylogging_analysis.config import MetricConfig
from keylogging_analysis.diff import derive_edits, edit_between
from keylogging_analysis.metrics._common import dist_stats, per_minute, safe_ratio, span_ms
from keylogging_analysis.schema import KeylogData, validate
from helpers import MSG_A, MSG_B, edits_of, events_of, messages_of


def _clean(*messages, config=None):
    config = config or MetricConfig()
    data = validate(KeylogData(events_of(*messages), messages_of([m for m, _ in messages])))
    return clean_events(data.events, config)


@pytest.mark.parametrize("prev,cur,expected", [
    ("", "H", (0, 0, 1)),            # first keystroke
    ("Hi", "Hi ", (2, 0, 1)),        # append
    ("Hi th", "Hi t", (4, 1, 0)),    # backspace at the end
    ("I am", "I sm", (2, 1, 1)),     # mid-text replacement
    ("ab", "b", (0, 1, 0)),          # delete first character
    ("aa", "a", (1, 1, 0)),          # ambiguous: counts are right, pos is the later one
    ("abc", "abc", (3, 0, 0)),       # no change
    ("I", "I am", (1, 0, 3)),        # multi-character insert
    ("hello world", "hello", (5, 6, 0)),
])
def test_edit_between(prev, cur, expected):
    assert edit_between(prev, cur) == expected


def test_edit_between_non_ascii():
    assert edit_between("café", "café’") == (4, 0, 1)
    assert edit_between("ok 😀", "ok 😀!") == (4, 0, 1)
    assert edit_between("naïve", "naive") == (2, 1, 1)


def test_clean_orders_by_time_and_counts_regressions():
    # seq order: a (0 ms), abc (300 ms), ab (200 ms) -> time goes back once
    events, flags, report = _clean(("E", [(0, "a"), (300, "abc"), (200, "ab")]))
    assert events["text"].tolist() == ["a", "ab", "abc"]
    assert flags.loc["E", "n_time_regressions"] == 1
    assert report.n_messages_with_time_regression == 1


def test_clean_ties_broken_by_seq():
    events, _, _ = _clean(("T", [(0, "a"), (0, "ab")]))
    assert events["text"].tolist() == ["a", "ab"]


def test_clean_drops_nochange_events_including_empty_first_state():
    events, flags, report = _clean(("N", [(0, ""), (10, "a"), (20, "a"), (30, "a"), (40, "ab")]))
    assert events["text"].tolist() == ["a", "ab"]
    assert flags.loc["N", "n_nochange_dropped"] == 3
    assert (report.n_events_in, report.n_nochange_dropped, report.n_events_out) == (5, 3, 2)


def test_clean_keeps_nochange_when_configured():
    events, flags, _ = _clean(("N", [(0, "a"), (10, "a")]),
                              config=MetricConfig(drop_nochange_events=False))
    assert len(events) == 2
    assert flags.loc["N", "n_nochange_dropped"] == 0


def test_clean_flags_cover_messages_whose_events_all_drop():
    # every state of Z equals its predecessor ("" before the first): Z leaves `events`
    # entirely, but its flags row must remain so the drop is still reported
    events, flags, _ = _clean(("Z", [(0, ""), (10, "")]), ("K", [(0, "k")]))
    assert "Z" not in set(events["message_id"])
    assert flags.loc["Z", "n_nochange_dropped"] == 2


def test_derive_edits_message_a():
    e = edits_of(("A", MSG_A))
    assert e["op"].tolist() == ["insert"] * 5 + ["delete"] * 2 + ["insert"] * 2
    assert e["pos"].tolist() == [0, 1, 2, 3, 4, 4, 3, 3, 4]
    assert e["n_ins"].tolist() == [1, 1, 1, 1, 1, 0, 0, 1, 1]
    assert e["n_del"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert e["at_end"].tolist() == [True] * 9
    assert e["prev_char"].tolist() == ["", "H", "i", " ", "t", "t", " ", " ", "y"]
    assert e["ins_first"].tolist() == ["H", "i", " ", "t", "h", "", "", "y", "o"]
    assert np.isnan(e["iki_ms"].iloc[0])
    assert e["iki_ms"].iloc[1:].tolist() == [100, 200, 2200, 100, 100, 100, 300, 400]
    assert not e["bulk"].any()


def test_derive_edits_message_b_bulk_replace_and_dropped_nochange():
    e = edits_of(("B", MSG_B))
    assert e["text"].tolist() == ["I", "I am", "I sm", "I sm!"]
    assert e["op"].tolist() == ["insert", "insert", "replace", "insert"]
    assert e["n_ins"].tolist() == [1, 3, 1, 1]
    assert e["n_del"].tolist() == [0, 0, 1, 0]
    assert e["at_end"].tolist() == [True, True, False, True]
    assert e["bulk"].tolist() == [False, True, False, False]
    assert e["iki_ms"].iloc[1:].tolist() == [150, 150, 300]   # 450 ms state was dropped


def test_derive_edits_resets_between_messages():
    e = edits_of(("A", MSG_A), ("B", MSG_B))
    first_b = e.index[e["message_id"] == "B"][0]
    assert np.isnan(e.loc[first_b, "iki_ms"])
    assert e.loc[first_b, "pos"] == 0 and e.loc[first_b, "prev_char"] == ""


def test_derive_edits_empty():
    e = edits_of()
    assert len(e) == 0
    assert {"pos", "n_ins", "n_del", "op", "at_end", "iki_ms", "bulk"} <= set(e.columns)


def test_common_helpers():
    e = edits_of(("A", MSG_A), ("B", MSG_B))
    sp = span_ms(e)
    assert sp["A"] == 3500 and sp["B"] == 600
    assert per_minute(pd.Series({"A": 7.0}), pd.Series({"A": 3500.0}))["A"] == pytest.approx(120.0)
    assert np.isnan(per_minute(pd.Series({"A": 1.0}), pd.Series({"A": 0.0}))["A"])
    assert np.isnan(safe_ratio(pd.Series([1.0]), pd.Series([0.0]))[0])
    st = dist_stats(e["iki_ms"], e["message_id"], "iki")
    assert st.loc["B", "iki_mean"] == pytest.approx(200.0)
    assert st.loc["B", "iki_median"] == pytest.approx(150.0)
    assert st.loc["B", "iki_iqr"] == pytest.approx(75.0)
    assert st.loc["B", "iki_mad"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_clean_diff.py -v`
Expected: collection error `No module named 'keylogging_analysis.clean'`.

- [ ] **Step 3: Implement `clean.py`**

```python
"""Event cleaning. Every step is counted, nothing is dropped silently."""
from dataclasses import asdict, dataclass, field

import pandas as pd

from .config import MetricConfig
from .schema import GROUP


@dataclass
class CleaningReport:
    n_messages: int = 0
    n_events_in: int = 0
    n_nochange_dropped: int = 0
    n_messages_with_time_regression: int = 0
    n_events_out: int = 0
    adapter_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _first_of_message(ev: pd.DataFrame) -> pd.Series:
    return ev[GROUP].ne(ev[GROUP].shift())


def clean_events(events: pd.DataFrame, config: MetricConfig):
    """Order events by client time and drop states that change nothing.

    Returns (events, flags, report). ``events`` is sorted by message_id, t_ms,
    seq. ``flags`` has one row per message that had events before cleaning.
    """
    report = CleaningReport(n_events_in=len(events))

    by_source = events.sort_values([GROUP, "seq"], kind="stable")
    regress = by_source.groupby(GROUP, sort=False)["t_ms"].diff().lt(0)
    n_reg = regress.groupby(by_source[GROUP], sort=False).sum().astype("int64")

    ev = events.sort_values([GROUP, "t_ms", "seq"], kind="stable").reset_index(drop=True)
    first = _first_of_message(ev)
    prev_text = ev["text"].shift().where(~first, "").fillna("")
    nochange = ev["text"].eq(prev_text)
    if not config.drop_nochange_events:
        nochange = pd.Series(False, index=ev.index)
    n_drop = nochange.groupby(ev[GROUP], sort=False).sum().astype("int64")
    ev = ev[~nochange].reset_index(drop=True)

    flags = pd.DataFrame({"n_time_regressions": n_reg, "n_nochange_dropped": n_drop})
    flags.index.name = GROUP
    flags = flags.fillna(0).astype("int64")

    report.n_nochange_dropped = int(n_drop.sum())
    report.n_messages_with_time_regression = int((n_reg > 0).sum())
    report.n_events_out = len(ev)
    return ev, flags, report
```

- [ ] **Step 4: Implement `diff.py`**

```python
"""Per-event edit derivation: what each text state changed, where, after how long.

The diff is common-prefix / common-suffix. It gets every count right; with runs
of identical characters ("aa" -> "a") the reported position is the later one.
"""
import numpy as np
import pandas as pd

from .config import MetricConfig
from .schema import GROUP

EDIT_COLUMNS = ["pos", "n_ins", "n_del", "op", "at_end", "prev_char", "ins_first", "iki_ms", "bulk"]


def _common_prefix_len(a: str, b: str) -> int:
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:                       # binary search on slices: C-speed compares
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _common_suffix_len(a: str, b: str) -> int:
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:] == b[len(b) - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def edit_between(prev: str, cur: str):
    """Return (pos, n_del, n_ins): the edit that turns ``prev`` into ``cur``."""
    if cur.startswith(prev):                     # typing at the end: the common case
        return len(prev), 0, len(cur) - len(prev)
    if prev.startswith(cur):                     # backspacing at the end
        return len(cur), len(prev) - len(cur), 0
    p = _common_prefix_len(prev, cur)
    a, b = prev[p:], cur[p:]
    s = _common_suffix_len(a, b)
    return p, len(a) - s, len(b) - s


def derive_edits(events: pd.DataFrame, config: MetricConfig) -> pd.DataFrame:
    """Add edit columns to cleaned events (``clean_events`` output, already sorted)."""
    ev = events.reset_index(drop=True).copy()
    first = ev[GROUP].ne(ev[GROUP].shift())
    prev = ev["text"].shift().where(~first, "").fillna("").tolist()
    cur = ev["text"].tolist()

    triples = [edit_between(a, b) for a, b in zip(prev, cur)]
    pos = np.fromiter((t[0] for t in triples), dtype=np.int64, count=len(triples))
    n_del = np.fromiter((t[1] for t in triples), dtype=np.int64, count=len(triples))
    n_ins = np.fromiter((t[2] for t in triples), dtype=np.int64, count=len(triples))

    ev["pos"] = pos
    ev["n_ins"] = n_ins
    ev["n_del"] = n_del
    ev["op"] = pd.Series(
        np.select([(n_ins > 0) & (n_del == 0), (n_del > 0) & (n_ins == 0), (n_ins > 0) & (n_del > 0)],
                  ["insert", "delete", "replace"], "none"),
        index=ev.index, dtype="string")
    prev_len = np.fromiter((len(a) for a in prev), dtype=np.int64, count=len(prev))
    ev["at_end"] = (pos + n_del) == prev_len
    ev["prev_char"] = pd.Series([a[p - 1] if p > 0 else "" for a, p in zip(prev, pos)],
                                index=ev.index, dtype="string")
    ev["ins_first"] = pd.Series([b[p] if k > 0 else "" for b, p, k in zip(cur, pos, n_ins)],
                                index=ev.index, dtype="string")
    ev["iki_ms"] = ev["t_ms"].diff().where(~first).astype("float64")
    ev["bulk"] = (ev["op"] == "insert").to_numpy(dtype=bool) & (n_ins >= config.bulk_insert_min)
    return ev
```

- [ ] **Step 5: Implement `metrics/__init__.py` and `metrics/_common.py`**

`metrics/__init__.py` (Task 8 fills it in):

```python
"""Message-level indicators computed from derived edits (see the design spec, §5)."""
```

`metrics/_common.py`:

```python
"""Helpers shared by the metric modules. All results are indexed by message_id."""
import numpy as np
import pandas as pd

from ..schema import GROUP


def span_ms(edits: pd.DataFrame) -> pd.Series:
    g = edits.groupby(GROUP, sort=False)["t_ms"]
    return (g.max() - g.min()).astype("float64")


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0 or missing (never inf)."""
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den).where(den > 0, np.nan)


def per_minute(x: pd.Series, span: pd.Series) -> pd.Series:
    return safe_ratio(x, span / 60000.0)


def dist_stats(values: pd.Series, keys: pd.Series, prefix: str) -> pd.DataFrame:
    """mean, median, sd (n-1), iqr (linear quantiles), mad (unscaled) per key."""
    g = values.groupby(keys, sort=False)
    dev = (values - g.transform("median")).abs()
    out = pd.DataFrame({
        f"{prefix}_mean": g.mean(),
        f"{prefix}_median": g.median(),
        f"{prefix}_sd": g.std(),
        f"{prefix}_iqr": g.quantile(0.75) - g.quantile(0.25),
        f"{prefix}_mad": dev.groupby(keys, sort=False).median(),
    })
    out.index.name = GROUP
    return out
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_clean_diff.py -v`
Expected: all passed. Then `uv run pytest -q`: everything green.

- [ ] **Step 7: Commit**

```bash
git add src/keylogging_analysis/clean.py src/keylogging_analysis/diff.py src/keylogging_analysis/metrics tests/test_clean_diff.py
git commit -m "feat: event cleaning report, per-event edit derivation, metric helpers"
```

---

### Task 3: Timing and pause metrics (wave 2)

**Files:**
- Create: `src/keylogging_analysis/metrics/timing.py`
- Test: `tests/test_metrics_timing.py`

**Interfaces:**
- Consumes: `derive_edits` output columns (Task 2); `span_ms`, `per_minute`, `safe_ratio`, `dist_stats` (Task 2); `MetricConfig`, `GROUP` (Task 1).
- Produces: `timing_metrics(edits: pd.DataFrame, config: MetricConfig) -> pd.DataFrame`, indexed by `message_id` (one row per message present in `edits`), columns in this order: `typing_span_ms`, `iki_mean`, `iki_median`, `iki_sd`, `iki_iqr`, `iki_mad`, then for each θ: `n_pauses_θ` (int64), `pause_time_ms_θ` (float64), `pauses_per_min_θ`, `share_pauses_between_words_θ`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics_timing.py`:

```python
import statistics

import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.metrics.timing import timing_metrics
from helpers import MSG_A, MSG_B, MSG_C, edits_of, isnan, row

# Message A, cleaned (9 events):
#   IKIs = 100, 200, 2200, 100, 100, 100, 300, 400 ; span = 4500 - 1000 = 3500 ms
#   sorted 100,100,100,100,200,300,400,2200 -> mean 437.5, median 150
#   linear quantiles: q25 = 100, q75 = 300 + 0.25*(400-300) = 325 -> iqr 225
#   |x - 150| = 50,50,50,50,50,150,250,2050 -> mad 50
#   pauses >= 200 at events 2 (200, inserts " "), 3 (2200, after " "),
#   7 (300, after " "), 8 (400, "o" after "y") -> 4 pauses, 3100 ms, 3 between words
#   pauses >= 2000: event 3 only
# Message B, cleaned (4 events; the 450 ms no-change state is gone):
#   IKIs = 150, 150, 300 ; span 600 ; one pause >= 200 (300 ms, inserts "!")
# Message C: one event -> no IKI, span 0


def test_timing_message_a():
    r = row(timing_metrics(edits_of(("A", MSG_A)), MetricConfig()), "A")
    assert r["typing_span_ms"] == 3500
    assert r["iki_mean"] == pytest.approx(437.5)
    assert r["iki_median"] == pytest.approx(150)
    assert r["iki_sd"] == pytest.approx(statistics.stdev([100, 200, 2200, 100, 100, 100, 300, 400]))
    assert r["iki_iqr"] == pytest.approx(225)
    assert r["iki_mad"] == pytest.approx(50)
    assert r["n_pauses_200"] == 4
    assert r["pause_time_ms_200"] == pytest.approx(3100)
    assert r["pauses_per_min_200"] == pytest.approx(4 * 60000 / 3500)
    assert r["share_pauses_between_words_200"] == pytest.approx(0.75)
    assert r["n_pauses_2000"] == 1
    assert r["pause_time_ms_2000"] == pytest.approx(2200)
    assert r["pauses_per_min_2000"] == pytest.approx(60000 / 3500)
    assert r["share_pauses_between_words_2000"] == pytest.approx(1.0)


def test_timing_message_b_punctuation_counts_as_between_words():
    r = row(timing_metrics(edits_of(("B", MSG_B)), MetricConfig()), "B")
    assert r["typing_span_ms"] == 600
    assert r["iki_mean"] == pytest.approx(200)
    assert r["n_pauses_200"] == 1
    assert r["pauses_per_min_200"] == pytest.approx(100.0)
    assert r["share_pauses_between_words_200"] == pytest.approx(1.0)
    assert r["n_pauses_2000"] == 0
    assert r["pause_time_ms_2000"] == 0
    assert r["pauses_per_min_2000"] == 0
    assert isnan(r["share_pauses_between_words_2000"])


def test_timing_single_event_is_nan_not_inf():
    r = row(timing_metrics(edits_of(("C", MSG_C)), MetricConfig()), "C")
    assert r["typing_span_ms"] == 0
    assert all(isnan(r[k]) for k in ["iki_mean", "iki_median", "iki_sd", "iki_iqr", "iki_mad"])
    assert r["n_pauses_200"] == 0
    assert isnan(r["pauses_per_min_200"])
    assert isnan(r["share_pauses_between_words_200"])


def test_timing_pause_exactly_at_threshold_counts():
    r = row(timing_metrics(edits_of(("P", [(0, "a"), (200, "ab")])), MetricConfig()), "P")
    assert r["n_pauses_200"] == 1


def test_timing_pause_before_deletion_excluded_from_location_share():
    # pause 500 ms before a backspace, then 500 ms before "c" typed after "a"
    r = row(timing_metrics(edits_of(("D", [(0, "a"), (10, "ab"), (510, "a"), (1010, "ac")])),
                           MetricConfig()), "D")
    assert r["n_pauses_200"] == 2
    assert r["share_pauses_between_words_200"] == pytest.approx(0.0)


def test_timing_custom_thresholds_and_column_order():
    out = timing_metrics(edits_of(("A", MSG_A)), MetricConfig(pause_thresholds_ms=(300,)))
    assert list(out.columns) == ["typing_span_ms", "iki_mean", "iki_median", "iki_sd", "iki_iqr",
                                 "iki_mad", "n_pauses_300", "pause_time_ms_300",
                                 "pauses_per_min_300", "share_pauses_between_words_300"]
    assert out.loc["A", "n_pauses_300"] == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics_timing.py -v`
Expected: `ModuleNotFoundError: No module named 'keylogging_analysis.metrics.timing'`.

- [ ] **Step 3: Implement `metrics/timing.py`**

```python
"""Timing indicators: inter-event intervals, pauses and pause location (spec §5)."""
import pandas as pd

from ..config import MetricConfig
from ..schema import GROUP
from ._common import dist_stats, per_minute, safe_ratio, span_ms


def timing_metrics(edits: pd.DataFrame, config: MetricConfig) -> pd.DataFrame:
    key = edits[GROUP]
    iki = edits["iki_ms"]
    out = pd.DataFrame({"typing_span_ms": span_ms(edits)})
    out = out.join(dist_stats(iki, key, "iki"))

    seps = list(config.between_word_chars)
    between = (edits["pos"].eq(0)
               | edits["prev_char"].isin(seps).fillna(False)
               | edits["ins_first"].isin(seps).fillna(False))
    before_insert = edits["n_ins"] > 0

    for th in config.pause_thresholds_ms:
        pause = iki.ge(th).fillna(False)
        n = pause.groupby(key, sort=False).sum().astype("int64")
        out[f"n_pauses_{th}"] = n
        out[f"pause_time_ms_{th}"] = iki.where(pause, 0.0).groupby(key, sort=False).sum()
        out[f"pauses_per_min_{th}"] = per_minute(n, out["typing_span_ms"])
        den = (pause & before_insert).groupby(key, sort=False).sum()
        num = (pause & before_insert & between).groupby(key, sort=False).sum()
        out[f"share_pauses_between_words_{th}"] = safe_ratio(num, den)
    out.index.name = GROUP
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_metrics_timing.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/keylogging_analysis/metrics/timing.py tests/test_metrics_timing.py
git commit -m "feat: timing metrics - IKI distribution, pauses per threshold, pause location"
```

---

### Task 4: P-burst and R-burst metrics (wave 2)

**Files:**
- Create: `src/keylogging_analysis/metrics/bursts.py`
- Test: `tests/test_metrics_bursts.py`

**Interfaces:**
- Consumes: `derive_edits` output (Task 2), `MetricConfig`, `GROUP`.
- Produces:
  - `pburst_metrics(edits, config) -> pd.DataFrame` indexed by `message_id`; for each θ in order: `n_bursts_θ` (int64), `burst_size_max_θ`, `burst_size_mean_θ`, `burst_size_median_θ`, `burst_chars_mean_θ`.
  - `rburst_metrics(edits, config) -> pd.DataFrame` indexed by `message_id`: `n_rbursts` (int64), `rburst_size_max`, `rburst_size_median`, `n_revisions` (int64), `n_revisions_leading_edge` (int64).

- [ ] **Step 1: Write the failing test**

`tests/test_metrics_bursts.py`:

```python
import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.metrics.bursts import pburst_metrics, rburst_metrics
from helpers import MSG_A, MSG_B, MSG_C, edits_of, isnan, row

# Message A: bursts start at event 0 and at every pause.
#   θ=200: starts 0,2,3,7,8 -> [0,1] [2] [3,4,5,6] [7] [8]; sizes 2,1,4,1,1
#          net chars (ins - del): 2, 1, 0 (t,h in; h,t out), 1, 1
#   θ=2000: starts 0,3 -> [0..2] size 3 net 3 ; [3..8] size 6 net 4-2 = 2
#   R: runs [0..4] typing, [5,6] revision (starts at the end), [7,8] typing (last: not an R-burst)
# Message B (cleaned): I, I am, I sm (replace mid-text), I sm!
#   θ=200: pause before "!" -> [0,1,2] size 3 net 1+3+0 = 4 ; [3] size 1 net 1
#   θ=2000: one burst of 4, net 5
#   R: [0,1] typing, [2] revision (mid-text), [3] typing (last)


def test_pbursts_message_a():
    r = row(pburst_metrics(edits_of(("A", MSG_A)), MetricConfig()), "A")
    assert r["n_bursts_200"] == 5
    assert r["burst_size_max_200"] == 4
    assert r["burst_size_mean_200"] == pytest.approx(1.8)
    assert r["burst_size_median_200"] == pytest.approx(1)
    assert r["burst_chars_mean_200"] == pytest.approx(1.0)
    assert r["n_bursts_2000"] == 2
    assert r["burst_size_max_2000"] == 6
    assert r["burst_size_mean_2000"] == pytest.approx(4.5)
    assert r["burst_size_median_2000"] == pytest.approx(4.5)
    assert r["burst_chars_mean_2000"] == pytest.approx(2.5)


def test_pbursts_message_b_and_c():
    out = pburst_metrics(edits_of(("B", MSG_B), ("C", MSG_C)), MetricConfig())
    b, c = row(out, "B"), row(out, "C")
    assert (b["n_bursts_200"], b["burst_size_max_200"]) == (2, 3)
    assert b["burst_chars_mean_200"] == pytest.approx(2.5)
    assert (b["n_bursts_2000"], b["burst_size_max_2000"]) == (1, 4)
    assert b["burst_chars_mean_2000"] == pytest.approx(5.0)
    assert (c["n_bursts_200"], c["burst_size_max_200"], c["burst_size_median_200"]) == (1, 1, 1)


def test_pbursts_column_order():
    out = pburst_metrics(edits_of(("A", MSG_A)), MetricConfig(pause_thresholds_ms=(200,)))
    assert list(out.columns) == ["n_bursts_200", "burst_size_max_200", "burst_size_mean_200",
                                 "burst_size_median_200", "burst_chars_mean_200"]


def test_rbursts_message_a_and_b():
    out = rburst_metrics(edits_of(("A", MSG_A), ("B", MSG_B)), MetricConfig())
    a, b = row(out, "A"), row(out, "B")
    assert (a["n_rbursts"], a["rburst_size_max"], a["rburst_size_median"]) == (1, 5, 5)
    assert (a["n_revisions"], a["n_revisions_leading_edge"]) == (1, 1)
    assert (b["n_rbursts"], b["rburst_size_max"], b["rburst_size_median"]) == (1, 2, 2)
    assert (b["n_revisions"], b["n_revisions_leading_edge"]) == (1, 0)


def test_rbursts_none_without_revision():
    r = row(rburst_metrics(edits_of(("C", MSG_C)), MetricConfig()), "C")
    assert (r["n_rbursts"], r["n_revisions"], r["n_revisions_leading_edge"]) == (0, 0, 0)
    assert isnan(r["rburst_size_max"]) and isnan(r["rburst_size_median"])


def test_rbursts_two_revisions():
    # ab, abc, ab (rev 1), abd, abde, abd, ab (rev 2, two deletions), abx
    steps = [(0, "a"), (10, "ab"), (20, "abc"), (30, "ab"), (40, "abd"), (50, "abde"),
             (60, "abd"), (70, "ab"), (80, "abx")]
    r = row(rburst_metrics(edits_of(("R", steps)), MetricConfig()), "R")
    # typing runs before a revision: [a, ab, abc] size 3, [abd, abde] size 2
    assert (r["n_rbursts"], r["rburst_size_max"], r["rburst_size_median"]) == (2, 3, 2.5)
    assert (r["n_revisions"], r["n_revisions_leading_edge"]) == (2, 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics_bursts.py -v`
Expected: `ModuleNotFoundError: No module named 'keylogging_analysis.metrics.bursts'`.

- [ ] **Step 3: Implement `metrics/bursts.py`**

```python
"""Burst indicators (spec §5).

P-burst: a run of events started by the message's first event or by a pause >= θ.
R-burst: a run of non-revision events that is ended by a revision (a maximal
run of delete/replace events). The final run of a message is ended by sending,
not by a revision, so it is not an R-burst.
"""
import pandas as pd

from ..config import MetricConfig
from ..schema import GROUP

REVISION_OPS = ["delete", "replace"]


def pburst_metrics(edits: pd.DataFrame, config: MetricConfig) -> pd.DataFrame:
    key = edits[GROUP]
    first = key.ne(key.shift())
    net = edits["n_ins"] - edits["n_del"]
    frames = []
    for th in config.pause_thresholds_ms:
        start = first | edits["iki_ms"].ge(th).fillna(False)
        bursts = (pd.DataFrame({GROUP: key, "bid": start.cumsum(), "net": net})
                  .groupby("bid", sort=False)
                  .agg(**{GROUP: (GROUP, "first"), "size": (GROUP, "size"), "net": ("net", "sum")}))
        g = bursts.groupby(GROUP, sort=False)
        frames.append(pd.DataFrame({
            f"n_bursts_{th}": g.size().astype("int64"),
            f"burst_size_max_{th}": g["size"].max(),
            f"burst_size_mean_{th}": g["size"].mean(),
            f"burst_size_median_{th}": g["size"].median(),
            f"burst_chars_mean_{th}": g["net"].mean(),
        }))
    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    out.index.name = GROUP
    return out


def rburst_metrics(edits: pd.DataFrame, config: MetricConfig) -> pd.DataFrame:
    key = edits[GROUP]
    is_rev = edits["op"].isin(REVISION_OPS).fillna(False)
    first = key.ne(key.shift())
    run_start = first | is_rev.ne(is_rev.shift())
    runs = (pd.DataFrame({GROUP: key, "rid": run_start.cumsum(), "is_rev": is_rev,
                          "at_end": edits["at_end"]})
            .groupby("rid", sort=False)
            .agg(**{GROUP: (GROUP, "first"), "is_rev": ("is_rev", "first"),
                    "size": ("is_rev", "size"), "starts_at_end": ("at_end", "first")}))
    runs["is_last"] = runs[GROUP].ne(runs[GROUP].shift(-1))

    idx = pd.Index(key.drop_duplicates().tolist(), name=GROUP, dtype=key.dtype)
    rb = runs[~runs["is_rev"] & ~runs["is_last"]].groupby(GROUP, sort=False)["size"]
    rev = runs[runs["is_rev"]].groupby(GROUP, sort=False)
    out = pd.DataFrame(index=idx)
    out["n_rbursts"] = rb.size().reindex(idx, fill_value=0).astype("int64")
    out["rburst_size_max"] = rb.max().reindex(idx).astype("float64")
    out["rburst_size_median"] = rb.median().reindex(idx).astype("float64")
    out["n_revisions"] = rev.size().reindex(idx, fill_value=0).astype("int64")
    out["n_revisions_leading_edge"] = (rev["starts_at_end"].sum()
                                       .reindex(idx, fill_value=0).astype("int64"))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_metrics_bursts.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/keylogging_analysis/metrics/bursts.py tests/test_metrics_bursts.py
git commit -m "feat: P-burst metrics per pause threshold and R-burst/revision metrics"
```

---

### Task 5: Volume, product, rate and quality metrics (wave 2)

**Files:**
- Create: `src/keylogging_analysis/metrics/product.py`
- Test: `tests/test_metrics_product.py`

**Interfaces:**
- Consumes: `derive_edits` output (Task 2); `span_ms`, `per_minute`, `safe_ratio` (Task 2); validated `messages` frame (Task 1); `MetricConfig`, `GROUP`.
- Produces: `product_metrics(edits: pd.DataFrame, messages: pd.DataFrame, config: MetricConfig) -> pd.DataFrame` indexed by `message_id` (messages present in `edits`), columns in order: `n_events`, `n_insert`, `n_delete`, `n_replace`, `chars_inserted`, `chars_deleted` (all int64), `final_length` (int64), `process_product_ratio`, `cpm_product`, `cpm_process`, `cpm_product_with_thinking`, `cpm_process_with_thinking`, `n_bulk_inserts` (int64), `chars_bulk_inserted` (int64), `has_bulk_insert` (bool), `final_matches_sent` (pandas `boolean`, NA when `sent_text` is unknown or absent).

- [ ] **Step 1: Write the failing test**

`tests/test_metrics_product.py`:

```python
import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.metrics.product import product_metrics
from helpers import MSG_A, MSG_B, MSG_C, edits_of, isnan, messages_of, row

MSGS = messages_of(["A", "B", "C"], sent_text=["Hi yo", "I am!", None],
                   response_delay_s=[10.0, None, None])

# A: 9 events, 7 inserts / 2 deletes, 7 chars in / 2 out, final "Hi yo" (5), span 3500 ms
#    cpm_product = 5 / (3500/60000) ; cpm_process = 7 / (3500/60000) = 120
#    with a 10 s response delay: 5 / (10/60) = 30 ; 7 / (10/60) = 42
# B: I, I am (bulk, 3 chars), I sm (replace), I sm! -> 6 chars in, 1 out, final 5, span 600
#    sent "I am!" differs from the final state "I sm!"


def _validated(msgs):
    """Messages with the types validate() gives them (as compute_message_metrics passes them)."""
    from keylogging_analysis.schema import KeylogData, validate
    from helpers import events_of
    return validate(KeylogData(events_of(), msgs)).messages


def test_product_message_a():
    r = row(product_metrics(edits_of(("A", MSG_A)), _validated(MSGS), MetricConfig()), "A")
    assert (r["n_events"], r["n_insert"], r["n_delete"], r["n_replace"]) == (9, 7, 2, 0)
    assert (r["chars_inserted"], r["chars_deleted"], r["final_length"]) == (7, 2, 5)
    assert r["process_product_ratio"] == pytest.approx(1.4)
    assert r["cpm_product"] == pytest.approx(5 * 60000 / 3500)
    assert r["cpm_process"] == pytest.approx(120.0)
    assert r["cpm_product_with_thinking"] == pytest.approx(30.0)
    assert r["cpm_process_with_thinking"] == pytest.approx(42.0)
    assert (r["n_bulk_inserts"], r["chars_bulk_inserted"], r["has_bulk_insert"]) == (0, 0, False)
    assert r["final_matches_sent"] == True  # noqa: E712


def test_product_message_b():
    r = row(product_metrics(edits_of(("B", MSG_B)), _validated(MSGS), MetricConfig()), "B")
    assert (r["n_events"], r["n_insert"], r["n_delete"], r["n_replace"]) == (4, 3, 0, 1)
    assert (r["chars_inserted"], r["chars_deleted"], r["final_length"]) == (6, 1, 5)
    assert r["process_product_ratio"] == pytest.approx(1.2)
    assert r["cpm_product"] == pytest.approx(500.0)
    assert r["cpm_process"] == pytest.approx(600.0)
    assert isnan(r["cpm_product_with_thinking"]) and isnan(r["cpm_process_with_thinking"])
    assert (r["n_bulk_inserts"], r["chars_bulk_inserted"], r["has_bulk_insert"]) == (1, 3, True)
    assert r["final_matches_sent"] == False  # noqa: E712


def test_product_single_event_and_unknown_sent_text():
    r = row(product_metrics(edits_of(("C", MSG_C)), _validated(MSGS), MetricConfig()), "C")
    assert (r["n_events"], r["final_length"]) == (1, 1)
    assert isnan(r["cpm_product"]) and isnan(r["cpm_process"])
    assert isnan(r["final_matches_sent"])


def test_single_bulk_event_message():
    msgs = _validated(messages_of(["P"], sent_text=["pasted whole message"]))
    r = row(product_metrics(edits_of(("P", [(0, "pasted whole message")])), msgs, MetricConfig()), "P")
    assert (r["n_bulk_inserts"], r["chars_bulk_inserted"], r["has_bulk_insert"]) == (1, 20, True)
    assert isnan(r["cpm_product"])          # span 0 -> NaN, never inf
    assert r["final_matches_sent"] == True  # noqa: E712


def test_bulk_threshold_follows_config():
    msgs = _validated(messages_of(["Q"]))
    e = edits_of(("Q", [(0, "a"), (10, "abc")]), config=MetricConfig(bulk_insert_min=3))
    assert row(product_metrics(e, msgs, MetricConfig(bulk_insert_min=3)), "Q")["n_bulk_inserts"] == 0
    e = edits_of(("Q", [(0, "a"), (10, "abcd")]))
    assert row(product_metrics(e, msgs, MetricConfig()), "Q")["n_bulk_inserts"] == 1


def test_product_without_optional_message_columns():
    msgs = _validated(messages_of(["A"]))
    r = row(product_metrics(edits_of(("A", MSG_A)), msgs, MetricConfig()), "A")
    assert isnan(r["cpm_product_with_thinking"])
    assert isnan(r["final_matches_sent"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics_product.py -v`
Expected: `ModuleNotFoundError: No module named 'keylogging_analysis.metrics.product'`.

- [ ] **Step 3: Implement `metrics/product.py`**

```python
"""Volume, product, rate and data-quality indicators (spec §5)."""
import numpy as np
import pandas as pd

from ..config import MetricConfig
from ..schema import GROUP
from ._common import per_minute, safe_ratio, span_ms


def product_metrics(edits: pd.DataFrame, messages: pd.DataFrame,
                    config: MetricConfig) -> pd.DataFrame:
    key = edits[GROUP]
    g = edits.groupby(GROUP, sort=False)
    op = edits["op"]

    def count(mask: pd.Series) -> pd.Series:
        return mask.fillna(False).groupby(key, sort=False).sum().astype("int64")

    out = pd.DataFrame({
        "n_events": g.size().astype("int64"),
        "n_insert": count(op.eq("insert")),
        "n_delete": count(op.eq("delete")),
        "n_replace": count(op.eq("replace")),
        "chars_inserted": g["n_ins"].sum().astype("int64"),
        "chars_deleted": g["n_del"].sum().astype("int64"),
    })
    last_text = g["text"].last()
    out["final_length"] = last_text.str.len().astype("int64")
    out["process_product_ratio"] = safe_ratio(out["chars_inserted"], out["final_length"])

    span = span_ms(edits)
    out["cpm_product"] = per_minute(out["final_length"], span)
    out["cpm_process"] = per_minute(out["chars_inserted"], span)

    ms = messages.set_index(GROUP)
    if "response_delay_s" in ms.columns:
        delay_min = ms["response_delay_s"].reindex(out.index).astype("float64") / 60.0
    else:
        delay_min = pd.Series(np.nan, index=out.index)
    out["cpm_product_with_thinking"] = safe_ratio(out["final_length"], delay_min)
    out["cpm_process_with_thinking"] = safe_ratio(out["chars_inserted"], delay_min)

    bulk = edits["bulk"]
    out["n_bulk_inserts"] = count(bulk)
    out["chars_bulk_inserted"] = (edits["n_ins"].where(bulk, 0)
                                  .groupby(key, sort=False).sum().astype("int64"))
    out["has_bulk_insert"] = out["n_bulk_inserts"] > 0

    if "sent_text" in ms.columns:
        sent = ms["sent_text"].reindex(out.index).astype("string")
        out["final_matches_sent"] = (last_text.astype("string") == sent).astype("boolean")
    else:
        out["final_matches_sent"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out.index.name = GROUP
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_metrics_product.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/keylogging_analysis/metrics/product.py tests/test_metrics_product.py
git commit -m "feat: volume, product, rate and bulk-insert metrics"
```

---

### Task 6: Adapters (wave 2)

**Files:**
- Create: `src/keylogging_analysis/adapters/__init__.py`
- Create: `src/keylogging_analysis/adapters/base.py`
- Create: `src/keylogging_analysis/adapters/languagelab_export.py`
- Create: `src/keylogging_analysis/adapters/languagelab_legacy.py`
- Create: `src/keylogging_analysis/adapters/language_hero.py`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Consumes: `KeylogData`, `validate`, `GROUP` (Task 1).
- Produces:
  - `AdapterResult` (dataclass in `base.py`): `data: KeylogData`, `inputs: list[Path]`, `counts: dict[str, int]`.
  - `apply_filters(df: pd.DataFrame, filters: dict[str, list[str]] | None) -> pd.DataFrame` in `base.py` (string comparison; unknown column → `ValueError` naming it).
  - Each adapter module exposes `load(path: Path, filters: dict[str, list[str]] | None = None) -> AdapterResult`.
  - `adapters/__init__.py`: `ADAPTERS: dict[str, Callable]` with keys `"languagelab_export"`, `"languagelab_legacy"`, `"language_hero"`; `get_adapter(name) -> Callable` (unknown name → `ValueError` listing valid names).
  - `languagelab_export` counts keys: `messages_read`, `messages_after_filter`, `states_read`, `states_not_matched_unique`, `states_outside_filter`, `states_kept`.

Real-data facts the export adapter must handle (spec §7; profiled 2026-09-24):
- Quoted fields contain line breaks (scenario names such as `"ACTIVITY 1: \r\nMEETING…"`) → pyarrow `ParseOptions(newlines_in_values=True)`.
- `chat_message_id` is null on non-unique states → read ids as strings.
- `client_timestamp_raw` is in ms (median interval 185 ms, same as the keydown stream).
- `textarea_state_id` is the unique database id → use it as `seq`.
- `PERSONA_ID` in `lh_default.csv` is 100 % empty; the Language Hero user is `USER_ID`.
- Legacy datasets mark typing events with `event_for_message_type == "0"`; other types are start/stop markers whose `content` is not text.

- [ ] **Step 1: Write the failing tests**

`tests/test_adapters.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_adapters.py -v`
Expected: `ModuleNotFoundError: No module named 'keylogging_analysis.adapters'`.

- [ ] **Step 3: Implement `adapters/base.py`**

```python
"""What every adapter returns, and the shared message filter."""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..schema import KeylogData


@dataclass
class AdapterResult:
    data: KeylogData
    inputs: list = field(default_factory=list)      # files read, for provenance checksums
    counts: dict = field(default_factory=dict)      # rows read / kept / dropped, per reason


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    """Keep rows where each filter column's value (as text) is one of the listed values."""
    if not filters:
        return df
    for col, values in filters.items():
        if col not in df.columns:
            raise ValueError(f"filter column '{col}' not found; available: {', '.join(map(str, df.columns))}")
        df = df[df[col].astype("string").isin([str(v) for v in values]).fillna(False)]
    return df


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"expected input file not found: {path}")
    return path
```

- [ ] **Step 4: Implement `adapters/languagelab_export.py`**

```python
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


def _read_csv(path: Path, columns, types) -> pd.DataFrame:
    table = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(include_columns=columns, include_missing_columns=True,
                                             column_types=types),
    )
    return table.to_pandas()


def load(path: Path, filters=None) -> AdapterResult:
    path = Path(path)
    msg_file = require_file(path / "messages.csv")
    state_file = require_file(path / "textarea_states.csv")
    counts = {}

    msg_cols = list(RENAME) + PASSTHROUGH
    raw = _read_csv(msg_file, msg_cols, {c: pa.string() for c in msg_cols if c != "student_response_delay_s"}
                    | {"student_response_delay_s": pa.float64()})
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
```

Note: `dict | dict` needs Python 3.9+, fine under the ≥3.10 constraint.

- [ ] **Step 5: Implement `adapters/languagelab_legacy.py`**

```python
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
```

- [ ] **Step 6: Implement `adapters/language_hero.py`**

```python
"""Language Hero format (lh_default.csv): one row per key event with full content.

The user is USER_ID. PERSONA_ID is empty throughout lh_default.csv, although
the legacy KeyLoggingDataFrame uses it. The legacy anonymised/nonsense/native
filters are not applied here (use `filters` or filter downstream).
"""
from pathlib import Path

import pandas as pd

from ..schema import KeylogData
from .base import AdapterResult, apply_filters, require_file

EPOCH = pd.Timestamp("1970-01-01")


def load(path: Path, filters=None) -> AdapterResult:
    path = require_file(Path(path))
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    counts = {"rows_read": len(raw)}
    raw = apply_filters(raw, filters)
    counts["rows_after_filter"] = len(raw)

    messages = (raw.drop_duplicates("message_id")
                .rename(columns={"USER_ID": "user_id", "TASK_ID": "task_id",
                                 "message_content": "sent_text"})
                [["message_id", "user_id", "session_id", "task_id", "sent_text", "user_status",
                  "username", "SCENARIO_ID", "application", "message_created_at"]]
                .reset_index(drop=True))
    typing = raw[raw["event_for_message_type"] == "0"]
    t = pd.to_datetime(typing["key_time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    counts["events_bad_time"] = int(t.isna().sum())          # unparseable key_time: dropped, counted
    typing, t = typing[t.notna()], t[t.notna()]
    counts["events_kept"] = len(typing)
    events = pd.DataFrame({
        "message_id": typing["message_id"],
        "t_ms": (t - EPOCH) / pd.Timedelta(milliseconds=1),
        "text": typing["content"].fillna(""),
        "seq": typing["key_id"].astype("int64"),
    }).reset_index(drop=True)
    return AdapterResult(KeylogData(events, messages), [path], counts)
```

- [ ] **Step 7: Implement `adapters/__init__.py`**

```python
"""Source-format adapters. Each returns an AdapterResult holding canonical KeylogData."""
from . import language_hero, languagelab_export, languagelab_legacy
from .base import AdapterResult, apply_filters

ADAPTERS = {
    "languagelab_export": languagelab_export.load,
    "languagelab_legacy": languagelab_legacy.load,
    "language_hero": language_hero.load,
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown adapter '{name}'; choose one of: {', '.join(sorted(ADAPTERS))}") from None


__all__ = ["ADAPTERS", "AdapterResult", "apply_filters", "get_adapter"]
```

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/test_adapters.py -v`
Expected: 9 passed. If `test_languagelab_legacy_default_dataset` fails on the 123 ms interval, print the first rows of `ll_default.csv` and check the `key_time` format before changing the parser. Do not loosen the assertion.

- [ ] **Step 9: Commit**

```bash
git add src/keylogging_analysis/adapters tests/test_adapters.py
git commit -m "feat: adapters for the LanguageLab export, legacy LanguageLab and Language Hero"
```

---

### Task 7: Provenance (wave 2)

**Files:**
- Create: `src/keylogging_analysis/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: `MetricConfig` (Task 1), `CleaningReport` (Task 2).
- Produces:
  - `sha256_file(path: Path) -> str`
  - `provenance_path(out_csv: Path) -> Path` → `<stem>.provenance.json` next to the CSV.
  - `build_provenance(*, adapter: str, inputs: list[Path], config: MetricConfig, report: CleaningReport, n_rows_out: int, argv: list[str] | None = None) -> dict` with keys `engine`, `version`, `git` (`{"commit": str|None, "dirty": bool|None}`), `created_utc`, `python`, `pandas`, `adapter`, `inputs` (list of `{"path", "sha256", "bytes"}`), `config`, `cleaning`, `rows_out`, `argv`.
  - `write_provenance(prov: dict, path: Path) -> None` (UTF-8, indent 2, sorted keys).

- [ ] **Step 1: Write the failing test**

`tests/test_provenance.py`:

```python
import hashlib
import json
from pathlib import Path

from keylogging_analysis.clean import CleaningReport
from keylogging_analysis.config import MetricConfig
from keylogging_analysis.provenance import (build_provenance, provenance_path, sha256_file,
                                            write_provenance)


def test_sha256_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_provenance_path():
    assert provenance_path(Path("/a/b/metrics.csv")) == Path("/a/b/metrics.provenance.json")


def test_build_and_write_provenance(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("hello")
    report = CleaningReport(n_messages=2, n_events_in=5, adapter_counts={"states_kept": 5})
    prov = build_provenance(adapter="languagelab_export", inputs=[src],
                            config=MetricConfig(), report=report, n_rows_out=2,
                            argv=["keylog-metrics", "languagelab_export", str(tmp_path)])
    assert prov["engine"] == "keylogging_analysis"
    assert prov["adapter"] == "languagelab_export"
    assert prov["inputs"] == [{"path": str(src), "sha256": sha256_file(src), "bytes": 5}]
    assert prov["config"]["pause_thresholds_ms"] == [200, 2000]
    assert prov["cleaning"]["adapter_counts"] == {"states_kept": 5}
    assert prov["rows_out"] == 2
    assert set(prov["git"]) == {"commit", "dirty"}
    out = tmp_path / "m.provenance.json"
    write_provenance(prov, out)
    assert json.loads(out.read_text(encoding="utf-8")) == prov


def test_git_state_in_checkout():
    # the test suite runs from a git checkout of this repo
    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None or len(prov["git"]["commit"]) == 40
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_provenance.py -v`
Expected: `ModuleNotFoundError: No module named 'keylogging_analysis.provenance'`.

- [ ] **Step 3: Implement `provenance.py`**

```python
"""What produced an output file: engine version and commit, config, inputs, cleaning counts."""
import datetime as dt
import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance_path(out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    return out_csv.with_name(out_csv.stem + ".provenance.json")


def _pkg_version() -> str:
    try:
        return version("keylogging-analysis")
    except PackageNotFoundError:
        from . import __version__
        return __version__


def _git_state() -> dict:
    if not (REPO_ROOT / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                                 "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_provenance(*, adapter, inputs, config, report, n_rows_out, argv=None) -> dict:
    return {
        "engine": "keylogging_analysis",
        "version": _pkg_version(),
        "git": _git_state(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "adapter": adapter,
        "inputs": [{"path": str(p), "sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                   for p in inputs],
        "config": config.to_dict(),
        "cleaning": report.to_dict(),
        "rows_out": int(n_rows_out),
        "argv": list(argv) if argv is not None else None,
    }


def write_provenance(prov: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(prov, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                          encoding="utf-8")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_provenance.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/keylogging_analysis/provenance.py tests/test_provenance.py
git commit -m "feat: provenance record (version, commit, config, input checksums, counts)"
```

---

### Task 8: Assembly, public API and CLI (wave 3, after all of wave 2 is merged)

**Files:**
- Modify: `src/keylogging_analysis/metrics/__init__.py`
- Modify: `src/keylogging_analysis/cli.py` (replace the placeholder)
- Modify: `src/keylogging_analysis/__init__.py`
- Test: `tests/test_engine.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7 (exact names listed there).
- Produces:
  - `compute_message_metrics(data: KeylogData, config: MetricConfig | None = None, adapter_counts: dict | None = None) -> tuple[pd.DataFrame, CleaningReport]` — one row per message (all messages, including those with no events), message columns first, then product, timing, P-burst, R-burst metrics, then `n_time_regressions`, `n_nochange_dropped`.
  - `count_columns(config) -> list[str]` (int64 columns filled with 0 for messages without events).
  - Package exports: `from keylogging_analysis import compute_message_metrics, MetricConfig, KeylogData, get_adapter, KeyLoggingDataFrame, __version__`.
  - CLI `keylog-metrics ADAPTER INPUT --out CSV [--config JSON] [--filter COL=VALUE ...]`; `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
import numpy as np
import pytest

from keylogging_analysis import KeylogData, MetricConfig, compute_message_metrics
from helpers import abc_data, events_of, isnan, messages_of, row


def test_one_row_per_message_including_empty():
    table, report = compute_message_metrics(abc_data())
    assert table["message_id"].tolist() == ["A", "B", "C", "D"]
    assert report.n_messages == 4
    assert (report.n_events_in, report.n_nochange_dropped, report.n_events_out) == (15, 1, 14)


def test_column_order_messages_first():
    table, _ = compute_message_metrics(abc_data())
    cols = table.columns.tolist()
    assert cols[:5] == ["message_id", "user_id", "session_id", "sent_text", "response_delay_s"]
    assert cols.index("n_events") < cols.index("typing_span_ms") < cols.index("n_bursts_200") \
        < cols.index("n_rbursts") < cols.index("n_time_regressions")
    assert cols[-2:] == ["n_time_regressions", "n_nochange_dropped"]


def test_values_flow_through_from_each_module():
    table, _ = compute_message_metrics(abc_data())
    a, b = row(table, "A"), row(table, "B")
    assert a["n_events"] == 9 and a["iki_mean"] == pytest.approx(437.5)
    assert a["n_bursts_200"] == 5 and a["n_rbursts"] == 1
    assert a["cpm_process_with_thinking"] == pytest.approx(42.0)
    assert b["has_bulk_insert"] is True or b["has_bulk_insert"] == True  # noqa: E712
    assert b["n_nochange_dropped"] == 1


def test_message_without_events():
    table, _ = compute_message_metrics(abc_data())
    d = row(table, "D")
    for c in ["n_events", "n_insert", "chars_inserted", "n_pauses_200", "n_bursts_200",
              "n_rbursts", "n_revisions", "n_bulk_inserts", "n_time_regressions",
              "n_nochange_dropped"]:
        assert d[c] == 0, c
    assert d["pause_time_ms_200"] == 0
    assert d["has_bulk_insert"] == False  # noqa: E712
    for c in ["final_length", "typing_span_ms", "iki_median", "cpm_product", "burst_size_max_200"]:
        assert isnan(d[c]), c


def test_all_nochange_message_behaves_like_empty():
    data = KeylogData(events_of(("Z", [(0, ""), (10, ""), (20, "")]), ("K", [(0, "k")])),
                      messages_of(["Z", "K"]))
    table, _ = compute_message_metrics(data)
    z = row(table, "Z")
    assert z["n_events"] == 0 and z["n_nochange_dropped"] == 3
    assert isnan(z["typing_span_ms"])


def test_count_columns_are_int64():
    table, _ = compute_message_metrics(abc_data())
    for c in ["n_events", "n_bursts_2000", "n_rbursts", "n_time_regressions"]:
        assert table[c].dtype == np.int64, c


def test_custom_config_changes_columns():
    table, _ = compute_message_metrics(abc_data(), MetricConfig(pause_thresholds_ms=(500,)))
    assert "n_bursts_500" in table.columns and "n_bursts_200" not in table.columns


def test_no_events_at_all():
    table, report = compute_message_metrics(KeylogData(events_of(), messages_of(["X", "Y"])))
    assert table["n_events"].tolist() == [0, 0]
    assert report.n_events_in == 0
```

`tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_engine.py tests/test_cli.py -v`
Expected: `ImportError: cannot import name 'compute_message_metrics'`.

- [ ] **Step 3: Implement `metrics/__init__.py`**

```python
"""Message-level indicators computed from derived edits (see the design spec, §5)."""
import pandas as pd

from ..clean import CleaningReport, clean_events
from ..config import MetricConfig
from ..diff import derive_edits
from ..schema import GROUP, KeylogData, validate
from .bursts import pburst_metrics, rburst_metrics
from .product import product_metrics
from .timing import timing_metrics

STATIC_COUNTS = ["n_events", "n_insert", "n_delete", "n_replace", "chars_inserted",
                 "chars_deleted", "n_bulk_inserts", "chars_bulk_inserted", "n_rbursts",
                 "n_revisions", "n_revisions_leading_edge", "n_time_regressions",
                 "n_nochange_dropped"]


def count_columns(config: MetricConfig) -> list:
    per_theta = [f"{p}_{th}" for th in config.pause_thresholds_ms for p in ("n_pauses", "n_bursts")]
    return STATIC_COUNTS + per_theta


def compute_message_metrics(data: KeylogData, config: MetricConfig = None,
                            adapter_counts: dict = None):
    """One row of indicators per message. Returns (table, CleaningReport)."""
    config = config or MetricConfig()
    data = validate(data)
    events, flags, report = clean_events(data.events, config)
    report.n_messages = len(data.messages)
    report.adapter_counts = dict(adapter_counts or {})
    edits = derive_edits(events, config)

    parts = [product_metrics(edits, data.messages, config), timing_metrics(edits, config),
             pburst_metrics(edits, config), rburst_metrics(edits, config)]
    metrics = pd.concat(parts, axis=1)
    out = data.messages.set_index(GROUP)
    out = out.join(metrics, how="left").join(flags, how="left")

    for c in count_columns(config):
        out[c] = out[c].fillna(0).astype("int64")
    for th in config.pause_thresholds_ms:
        out[f"pause_time_ms_{th}"] = out[f"pause_time_ms_{th}"].fillna(0.0)
    out["has_bulk_insert"] = out["has_bulk_insert"].astype("boolean").fillna(False).astype(bool)
    out["final_matches_sent"] = out["final_matches_sent"].astype("boolean")
    return out.reset_index(), report


__all__ = ["compute_message_metrics", "count_columns"]
```

Column-order note: `pd.concat(parts, axis=1)` keeps each module's column order, and joining `flags` last puts `n_time_regressions`, `n_nochange_dropped` at the end. Messages with no events are rows missing from every part, so the left join gives NaN, fixed above.

- [ ] **Step 4: Implement `cli.py`**

```python
"""keylog-metrics: message-level fluency indicators from a keystroke-logging export."""
import argparse
import sys
from pathlib import Path

from .adapters import ADAPTERS, get_adapter
from .config import MetricConfig
from .metrics import compute_message_metrics
from .provenance import build_provenance, provenance_path, write_provenance


def parse_filters(items) -> dict:
    filters = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--filter expects COL=VALUE, got '{item}'")
        col, value = item.split("=", 1)
        filters.setdefault(col, []).append(value)
    return filters


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="keylog-metrics", description=__doc__)
    p.add_argument("adapter", choices=sorted(ADAPTERS), help="input format")
    p.add_argument("input", type=Path, help="export directory or file, depending on the adapter")
    p.add_argument("--out", type=Path, required=True, help="output CSV (provenance JSON is written next to it)")
    p.add_argument("--config", type=Path, help="MetricConfig as JSON (defaults otherwise)")
    p.add_argument("--filter", action="append", default=[], metavar="COL=VALUE",
                   help="keep messages whose COL equals VALUE; repeat for several values")
    args = p.parse_args(argv)

    config = MetricConfig.from_json(args.config) if args.config else MetricConfig()
    res = get_adapter(args.adapter)(args.input, parse_filters(args.filter) or None)
    table, report = compute_message_metrics(res.data, config, adapter_counts=res.counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    full_argv = ["keylog-metrics", *(argv if argv is not None else sys.argv[1:])]
    prov = build_provenance(adapter=args.adapter, inputs=res.inputs, config=config,
                            report=report, n_rows_out=len(table), argv=full_argv)
    write_provenance(prov, provenance_path(args.out))
    print(f"keylog-metrics: {len(table)} messages, {report.n_events_out} events "
          f"({report.n_nochange_dropped} no-change dropped) -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Update `src/keylogging_analysis/__init__.py`**

```python
from .adapters import get_adapter
from .classes import KeyLoggingDataFrame
from .config import MetricConfig
from .metrics import compute_message_metrics
from .schema import KeylogData

__version__ = "0.0.2"
__all__ = ["KeyLoggingDataFrame", "KeylogData", "MetricConfig", "compute_message_metrics",
           "get_adapter", "__version__"]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest -v`
Expected: every test passes, legacy included. Then `uv run keylog-metrics --help` prints the usage.

- [ ] **Step 7: Commit**

```bash
git add src/keylogging_analysis/metrics/__init__.py src/keylogging_analysis/cli.py src/keylogging_analysis/__init__.py tests/test_engine.py tests/test_cli.py
git commit -m "feat: compute_message_metrics assembly, public API and keylog-metrics CLI"
```

---

### Task 9: Real data, cross-checks and performance (wave 3)

**Files:**
- Create: `tests/test_real_languagelab_export.py`
- Modify (only if the performance check fails): `src/keylogging_analysis/diff.py`

**Interfaces:**
- Consumes: `get_adapter`, `compute_message_metrics` (Task 8).
- Produces: a skipped-by-default integration test and a recorded runtime.

The real export is at
`/Users/serge/Library/CloudStorage/Dropbox/dev/LanguageLab/researchdata/ephec-bertrand/data/02_pseudonymised/25-26/`.
It is confidential. Read it in place; never copy it, excerpt it into the repo, or paste message text into commits, logs or reports.

- [ ] **Step 1: Write the test**

`tests/test_real_languagelab_export.py`:

```python
"""End-to-end run on a real LanguageLab export. Skipped unless KEYLOG_TEST_DATA is set.

    KEYLOG_TEST_DATA=/path/to/export uv run pytest tests/test_real_languagelab_export.py -v -s
"""
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from keylogging_analysis import compute_message_metrics, get_adapter

DATA = os.environ.get("KEYLOG_TEST_DATA")
FILTER = {"class_name": ["eB 2025-2026", "TI 2025-2026"]}
pytestmark = pytest.mark.skipif(not DATA, reason="set KEYLOG_TEST_DATA to a LanguageLab export directory")


@pytest.fixture(scope="module")
def run():
    t0 = time.perf_counter()
    res = get_adapter("languagelab_export")(Path(DATA), FILTER)
    table, report = compute_message_metrics(res.data, adapter_counts=res.counts)
    elapsed = time.perf_counter() - t0
    print(f"\nreal export: {len(table)} messages, {report.n_events_out} events, {elapsed:.1f} s")
    return res, table, report, elapsed


def test_runtime_under_60_seconds(run):
    assert run[3] < 60


def test_one_row_per_filtered_message(run):
    res, table, _, _ = run
    assert len(table) == res.counts["messages_after_filter"]


def test_event_count_matches_export_count_for_unique_keys(run):
    # For messages whose client key is unique, every state of that key is linked
    # to this message, so kept + dropped must equal the export's own count.
    _, table, _, _ = run
    u = table[table["client_key_is_unique_to_message"] == "t"]
    expected = pd.to_numeric(u["textarea_state_count_for_client_key"]).fillna(0).astype(int)
    assert ((u["n_events"] + u["n_nochange_dropped"]) == expected).mean() == 1.0


def test_final_state_agrees_with_export_flag(run):
    # Same ordering rule (client time, then id) as the export's own flag.
    _, table, _, _ = run
    u = table[table["last_state_by_client_time_matches_student_message"].isin(["t", "f"])]
    ours = u["final_matches_sent"].fillna(False).astype(bool)
    theirs = u["last_state_by_client_time_matches_student_message"] == "t"
    assert (ours == theirs).mean() >= 0.999


def test_indicators_are_plausible(run):
    _, table, _, _ = run
    typed = table[table["n_events"] >= 2]
    assert typed["iki_median"].between(30, 2000).mean() > 0.95
    assert (typed["n_bursts_200"] >= 1).all()
    assert (typed["burst_size_max_2000"] >= typed["burst_size_max_200"]).all()
    assert table["has_bulk_insert"].mean() < 0.2
```

- [ ] **Step 2: Run it against the real export**

Run:
```bash
KEYLOG_TEST_DATA="/Users/serge/Library/CloudStorage/Dropbox/dev/LanguageLab/researchdata/ephec-bertrand/data/02_pseudonymised/25-26" uv run pytest tests/test_real_languagelab_export.py -v -s
```
Expected: 5 passed, runtime printed (about 15 k messages, about 1.39 M events).

If a cross-check fails, **do not loosen the assertion**. Find out why (systematic-debugging skill): count and describe the disagreeing messages, using ids and counts only, never text, and report to the human. A disagreement is a finding about the data or the engine, and the human decides which.

- [ ] **Step 3: Performance, only if `test_runtime_under_60_seconds` failed**

Profile first:
```bash
KEYLOG_TEST_DATA=... uv run python -m cProfile -s cumtime -m pytest tests/test_real_languagelab_export.py -k runtime 2>&1 | head -40
```
Optimise only the top entry. The expected hotspot is the list comprehension over `edit_between` in `derive_edits`: the `startswith` fast paths cover about 99.5 % of events, so first check that they are hit (count the fall-throughs). Rerun the full suite after any change.

- [ ] **Step 4: Run the whole suite without the env var**

Run: `uv run pytest -q`
Expected: all pass, and `test_real_languagelab_export.py` shows as skipped.

- [ ] **Step 5: Commit**

```bash
git add tests/test_real_languagelab_export.py src/keylogging_analysis/diff.py
git commit -m "test: end-to-end run and cross-checks on a real LanguageLab export (opt-in)"
```

---

### Task 10: EPHEC integration (wave 4)

Work in the EPHEC repo on a new branch: `git switch -c text-state-metrics` (from `main`). The repo lives in Dropbox. After any `git switch`/`merge`, run `git status` and `git diff --stat HEAD` to confirm the working tree matches HEAD: on 2026-09-24 Dropbox reverted a git-written file within a second.

**Files (EPHEC repo):**
- Modify: `R/00_setup.R` (COHORT_FILES `25-26` entry; `KEYLOG_ENGINE`; `keylog_command()`)
- Create: `R/05a_text_state_metrics.R`
- Modify: `CLAUDE.md` (run order, engine note), `ANALYSIS-LOG.md` (dated entry with the run's numbers)
- Commit the pending `ANALYSIS-PLAN.md` edit (25-26 status + R-burst attribution reminder) on the same branch.

**Interfaces:**
- Consumes: the `keylog-metrics` CLI (Task 8), run with `uv`.
- Produces: `data/03_derived/25-26/message_metrics.csv` + `message_metrics.provenance.json`; the R helpers `keylog_command()` and `KEYLOG_ENGINE`.

- [ ] **Step 1: Register cohort 25-26 and the engine in `R/00_setup.R`**

Add a comma after the `24-25` entry's closing parenthesis and replace the commented-out `25-26` placeholder line below it, so `COHORT_FILES` reads:

```r
COHORT_FILES <- list(
  `24-25` = list(
    ...unchanged...
  ),
  # 25-26 arrived as a raw event export, not precomputed indicators: R/05a
  # computes message-level indicators with the keylogging_analysis engine.
  `25-26` = list(
    messages        = "messages.csv",
    textarea_states = "textarea_states.csv",
    keystrokes      = "keystrokes.csv",          # keydown stream, unused by v0.1 of the engine
    checksums       = "SHA256SUMS",
    id_mapping      = "student_id_mapping.csv",  # in 01_raw_identified/: S_... -> Moodle username
    chatbot_classes = c("eB 2025-2026", "TI 2025-2026")
  )
)
```

Then, right after the `IKI_MAX_MS` block and before the final `message(...)` call, add:

```r
# ---------------------------------------------------------------------------
# Text-state metrics engine (Python package keylogging_analysis, run with uv).
#
# While the engine is developed, KEYLOG_ENGINE is a local checkout and runs in
# that checkout's own environment (outside Dropbox, deliberately: no .venv
# here). Once released, set it to a pinned git URL, e.g.
#   git+https://github.com/sbibauw/keylogging_analysis@v0.1.0
# Either way the provenance JSON next to each output records the exact commit.
# ---------------------------------------------------------------------------
if (!exists("KEYLOG_ENGINE")) {
  KEYLOG_ENGINE <- Sys.getenv("EPHEC_KEYLOG_ENGINE",
                              path.expand("~/dev-gitonly/keylogging_analysis"))
}
keylog_command <- function(engine = KEYLOG_ENGINE) {
  if (dir.exists(engine)) c("run", "--project", shQuote(engine), "keylog-metrics")
  else                    c("tool", "run", "--from", shQuote(engine), "keylog-metrics")
}
```

- [ ] **Step 2: Create `R/05a_text_state_metrics.R`**

```r
# 05a_text_state_metrics.R — message-level fluency indicators from raw text states.
#
# From 2025-26 the chatbot data arrive as a raw event export (one row per
# textarea `input` event, with the full text) instead of precomputed
# indicators. This step runs the keylogging_analysis engine on it and writes
# one row per message to
#   data/03_derived/<cohort>/message_metrics.csv   (+ .provenance.json)
# Indicator definitions: docs/superpowers/specs/2026-09-24-text-state-engine-design.md
# in the keylogging_analysis repo. Cohorts without an event export (24-25) are
# skipped, so the full pipeline can run this for every cohort.

source("R/00_setup.R")

run_text_state_metrics <- function() {
  if (is.null(COHORT_FILES[[COHORT]]$textarea_states)) {
    message("05a: cohort ", COHORT, " has no raw event export -- skipped.")
    return(invisible(NULL))
  }

  # 1. The files on disk must be the export that SHA256SUMS describes.
  old <- setwd(DIR_PSEUDO)
  check <- system2("shasum", c("-a", "256", "-c", cohort_file("checksums")),
                   stdout = TRUE, stderr = TRUE)
  setwd(old)
  if (!is.null(attr(check, "status"))) {
    stop("05a: checksum mismatch in ", DIR_PSEUDO, ":\n", paste(check, collapse = "\n"))
  }

  # 2. Run the engine on this cohort's classes only. RStudio launched from the
  #    Dock may not have Homebrew's PATH; run from a terminal if uv is missing.
  if (!nzchar(Sys.which("uv"))) {
    stop("05a: 'uv' not found on PATH (Sys.getenv('PATH') = ", Sys.getenv("PATH"), ")")
  }
  out <- derived_path("message_metrics", "csv")
  filters <- as.vector(rbind("--filter",
                             shQuote(paste0("class_name=", cohort_file("chatbot_classes")))))
  args <- c(keylog_command(), "languagelab_export", shQuote(normalizePath(DIR_PSEUDO)),
            "--out", shQuote(normalizePath(out, mustWork = FALSE)), filters)
  status <- system2("uv", args)
  if (status != 0) stop("05a: keylog-metrics failed (exit status ", status, ")")

  # 3. What came out.
  mm <- read_csv(out, show_col_types = FALSE, guess_max = 1e5,
                 col_types = cols(message_id = col_character(), .default = col_guess()))
  cat("\n=== 05a text-state metrics:", COHORT, "===\n")
  cat("messages:", nrow(mm), "  with >= 2 events:", sum(mm$n_events >= 2),
      "  without events:", sum(mm$n_events == 0), "\n")
  # pandas writes booleans as True/False; as.logical() reads those, readr may not.
  cat("bulk-insert (paste candidate) messages:", sum(as.logical(mm$has_bulk_insert)), "\n")
  cat("final state == sent text:",
      round(mean(as.logical(mm$final_matches_sent), na.rm = TRUE), 4), "\n")
  print(mm %>% count(class_name, wave_number, name = "messages"))
  print(summary(mm$iki_median))
  invisible(mm)
}

run_text_state_metrics()
```

- [ ] **Step 3: Run it for 25-26 and check 24-25 is untouched**

Run (from the EPHEC root):
```bash
EPHEC_COHORT=25-26 Rscript R/05a_text_state_metrics.R
Rscript R/05a_text_state_metrics.R          # 24-25: prints "skipped"
```
Expected for 25-26: checksums pass. The CSV and provenance JSON are written to `data/03_derived/25-26/`. About 15 k messages, with the class/wave counts printed. Then confirm `git status` shows no change under `output/`.

- [ ] **Step 4: Document**

- `CLAUDE.md`, "Running things": add `source("R/05a_text_state_metrics.R")  # raw text states -> message indicators (25-26+)` after the 05 line, with one sentence: 05a is a no-op for 24-25 and needs `uv` plus the engine at `KEYLOG_ENGINE`.
- `ANALYSIS-LOG.md`: new dated entry "2026-MM-DD · 25-26 text-state indicators (`R/05a`)" with the numbers printed in Step 3 (messages, events, no-change drops, time regressions, bulk inserts, final-matches share, runtime) and the engine commit from the provenance JSON. Counts only, never message text.

- [ ] **Step 5: Commit on the branch**

```bash
git add R/00_setup.R R/05a_text_state_metrics.R CLAUDE.md ANALYSIS-LOG.md ANALYSIS-PLAN.md
git commit -m "Add R/05a: 25-26 message indicators from raw text states"
```

Do not merge into `main`; the human decides.

---

### Task 11: Release notes and version (wave 4)

**Files (package repo):**
- Modify: `pyproject.toml` (`version = "0.1.0"`)
- Modify: `src/keylogging_analysis/__init__.py` (`__version__ = "0.1.0"`)
- Modify: `tests/test_smoke.py` (only `test_version`: `assert __version__ == "0.1.0"`)
- Modify: `README.md` (new top section "Text-state engine (v0.1)")
- Modify: `CLAUDE.md` (architecture section for the new modules; legacy marked as such; uv commands)

- [ ] **Step 1: Bump the version**

Edit the three places above. Run `uv sync --offline && uv run pytest -q`: all pass.

- [ ] **Step 2: README section** (insert before "# How to install?")

````markdown
# Text-state engine (v0.1)

`keylogging_analysis` 0.1 adds an engine that computes one row of writing-process
indicators per message from logs that record the **full text at each event**.
Definitions of every indicator: `docs/superpowers/specs/2026-09-24-text-state-engine-design.md`.

```bash
uv run keylog-metrics languagelab_export path/to/export --out metrics.csv \
    --filter class_name="eB 2025-2026"
```

```python
from keylogging_analysis import get_adapter, compute_message_metrics, MetricConfig

res = get_adapter("languagelab_export")("path/to/export")
table, report = compute_message_metrics(res.data, MetricConfig(pause_thresholds_ms=(200, 2000)))
```

Adapters: `languagelab_export` (2025-26 analyst export), `languagelab_legacy`
(`ll_default.csv`), `language_hero` (`lh_default.csv`). Each run writes
`metrics.provenance.json` (engine version and commit, config, input checksums,
cleaning counts) next to the CSV. The earlier `KeyLoggingDataFrame` API below is
unchanged.
````

- [ ] **Step 3: CLAUDE.md**

Replace the "Build, Install and Test" block with the uv commands (`uv sync`, `uv run pytest`, and the `KEYLOG_TEST_DATA` opt-in test), and add a section "Text-state engine" that lists the modules from spec §3 with their one-line roles, states that `classes.py`/`help_functions.py` are legacy and not to be extended, and records the known legacy bug: `_user_id_col` uses `PERSONA_ID`, which is empty in `lh_default.csv`.

- [ ] **Step 4: Commit, then stop for the human**

```bash
git add pyproject.toml uv.lock src/keylogging_analysis/__init__.py tests/test_smoke.py README.md CLAUDE.md
git commit -m "release: v0.1.0 text-state engine; docs"
```

Then **stop**. Tagging `v0.1.0`, pushing to GitHub (`origin` = sbibauw fork, `upstream` = Thonissen), merging `engine-v0.1` into `main`, and switching EPHEC's `KEYLOG_ENGINE` to the git URL are the human's decisions.
