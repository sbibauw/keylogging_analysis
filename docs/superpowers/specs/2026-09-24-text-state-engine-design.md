# Text-state metrics engine — design

**Date:** 2026-09-24 · **Status:** approved 2026-09-24 · **Branch:** `engine-v0.1`
**First consumer:** EPHEC written-fluency study (`ephec-bertrand` repo), cohort 2025-26.

## 1. Purpose

Compute message-level writing-process (fluency) indicators from keystroke
logging data with **one documented, tested implementation** that several
projects share (EPHEC, LanguageLab, Language Hero, future datasets).

For EPHEC this engine is what the paper's measurement claims rest on. So:

- every indicator has a written definition (§5) and a hand-computed unit test;
- every output file carries the engine version, config and input checksums;
- nothing is silently dropped: each cleaning step is counted in a report.

**Success criteria for v0.1**

1. The 2025-26 LanguageLab export (`textarea_states.csv`, ~1.4 M EPHEC events,
   ~15 k messages) goes in; one row per message comes out, with every §5
   indicator. Runtime under 60 s on a laptop.
2. Every indicator in §5 has a unit test with a hand-computed expected value.
3. The EPHEC pipeline calls it through one command and gets a CSV plus a
   provenance JSON in `data/03_derived/25-26/`.
4. The legacy `KeyLoggingDataFrame` API still imports and its tests still run.

**Decision C (agreed 2026-09-24).** The engine is built so it *could* recompute
the 2024-25 cohort if the raw event stream is recovered: that only needs one
more adapter. Until then, 2024-25 keeps the platform's precomputed indicators.

## 2. Canonical data format: text states

The default input format is **full text at each event**: one row per event,
carrying the complete text of the input field at that moment. This matches
what the package has always assumed (`content` per key event) and what the
2025-26 export provides directly (`textarea_states.csv`, one row per browser
`input` event).

Profiled on 2025-26 EPHEC data (unique-linked, 1.39 M states, 15 319
messages): 99.5 % of events change the length by exactly ±1 character, the
median interval is 185 ms (keydown stream: 183 ms), 0.28 % of events leave the
text unchanged, 0.06 % insert ≥ 5 characters at once, and 205 rows show a
backwards client clock. Text states are therefore keystroke-level for our
purposes.

Sources that log **only keys** (not text) need text reconstruction first. That
is a separate module (`reconstruct`, §8), out of scope for v0.1.

### 2.1 Tables

A `KeylogData` dataclass holds two pandas DataFrames.

**`events`** — one row per text state

| column | type | meaning |
|---|---|---|
| `message_id` | string | message the event belongs to |
| `t_ms` | float64 | client time in ms; only differences *within a message* are meaningful |
| `text` | string | full field content after the event |
| `seq` | int64 | source order (database id / row order), tie-breaker |

**`messages`** — one row per message

| column | type | required | meaning |
|---|---|---|---|
| `message_id` | string | yes | key |
| `user_id` | string | yes | participant |
| `session_id` | string | yes | conversation / session |
| `task_id` | string | no | scenario / task |
| `sent_text` | string | no | text actually sent |
| `response_delay_s` | float64 | no | server time from prompt shown to message sent (includes reading and thinking) |
| any other | any | no | passed through unchanged to the output (class, wave, section…) |

Adapters return exactly this. The engine never reads source-specific columns.

## 3. Architecture

```
src/keylogging_analysis/
  __init__.py          exports the new API + legacy KeyLoggingDataFrame
  classes.py           LEGACY, untouched (KeyLoggingDataFrame)
  help_functions.py    LEGACY, untouched
  schema.py            KeylogData, column contracts, validate()
  config.py            MetricConfig (frozen dataclass, JSON round-trip)
  clean.py             ordering, duplicate/no-change handling, CleaningReport
  diff.py              per-event edit derivation (what changed, where)
  metrics/
    __init__.py        compute_message_metrics(data, config) -> DataFrame
    timing.py          intervals, pauses, pause location
    bursts.py          P-bursts (per threshold), R-bursts
    product.py         volume, revision, process/product, rates, quality flags
  adapters/
    __init__.py        registry: name -> loader
    languagelab_export.py   2025-26 export (messages/textarea_states CSVs)
    languagelab_legacy.py   old `ll` schema (ll_default.csv)
    language_hero.py        `lh` schema (lh_default.csv)
  provenance.py        version, git commit, config, input checksums
  cli.py               `keylog-metrics` entry point
```

Data flow: `adapter -> KeylogData -> validate -> clean -> diff -> metrics -> CSV + provenance JSON`.

Each unit is a pure function on DataFrames, independently testable. The legacy
class is left as is: it has external users (Thonissen) and replacing it is not
needed for v0.1.

## 4. Processing steps

### 4.1 Cleaning (`clean.py`)

Applied per message, each step counted in `CleaningReport`:

1. **Order** events by `t_ms`, then `seq`. Messages where time order differs
   from source order are counted (`n_time_regressions`) and flagged, not dropped.
2. **Drop no-change events**: an event whose `text` equals the previous
   event's text is not a keystroke that produced anything. Counted.
3. **Messages with fewer than 2 events** keep a row with NaN timing metrics.

### 4.2 Edit derivation (`diff.py`)

For each event, compare `text` with the previous event's text (empty string for
the first event) using common-prefix/common-suffix matching:

| derived column | meaning |
|---|---|
| `pos` | index where the change starts |
| `n_ins` | characters inserted |
| `n_del` | characters removed |
| `op` | `insert` (n_del = 0), `delete` (n_ins = 0), `replace` (both > 0) |
| `at_end` | change is at the end of the previous text (the leading edge) |
| `iki_ms` | `t_ms` − previous event's `t_ms`; NaN for the first event |
| `bulk` | `op == insert` and `n_ins >= config.bulk_insert_min` |

Common-prefix/suffix diff cannot tell which of several identical characters was
removed ("aa" → "a"). This affects `pos`, `prev_char`, `ins_first` and `at_end`
(and hence, rarely, pause location and leading-edge classification, both of
which read `prev_char`/`ins_first`/`at_end`) — never counts (`n_ins`, `n_del`).

## 5. Indicator definitions (message level)

`θ` ranges over `config.pause_thresholds_ms` (default `[200, 2000]`); every
θ-dependent indicator is emitted once per threshold with suffix `_θ`
(e.g. `n_bursts_200`). "Events" are cleaned, text-changing events.

**Volume and product**
- `n_events`; `n_insert`, `n_delete`, `n_replace` (event counts)
- `chars_inserted`, `chars_deleted` (sum of `n_ins`, `n_del`)
- `final_length` — length of the last state
- `process_product_ratio` — `chars_inserted / final_length`

**Timing**
- `typing_span_ms` — last `t_ms` − first `t_ms` (excludes time before the first event)
- `iki_mean`, `iki_median`, `iki_sd`, `iki_iqr`, `iki_mad` over non-NaN `iki_ms`
  (`iki_mad` is the **unscaled** median absolute deviation: R's `mad()`
  multiplies by 1.4826 by default to estimate a normal SD; use
  `mad(x, constant = 1)` on the R side to match this column)
- Per §4.1.3: messages with fewer than 2 events are NaN for every metric in
  this section, including `typing_span_ms` and `pause_time_ms_θ` below (not
  just the IKI-distribution stats) — a one-event message has no interval to
  measure, not a zero-length one.

**Pauses** (a pause is an event with `iki_ms >= θ`)
- `n_pauses_θ`, `pause_time_ms_θ` (sum of those intervals), `pauses_per_min_θ`
- `share_pauses_between_words_θ` — share of pauses before an insertion that fall
  between words. A pause is between words when the insertion point is at the
  start of the text, **or** the character just before it is a separator
  (`between_word_chars`), **or** the first inserted character is a separator
  (Inputlog convention: a pause after a finished word, before its space, is a
  between-word pause). Pauses before deletions are excluded from both counts.
  NaN when there is no pause before an insertion.

**P-bursts** (per θ). A burst starts at the first event and at every pause ≥ θ,
and runs until the next pause.
- `n_bursts_θ`
- `burst_size_max_θ`, `burst_size_mean_θ`, `burst_size_median_θ` — size in
  **events** (the 2024-25 platform's unit: "nombre de keystrokes du burst")
- `burst_chars_mean_θ` — net characters inserted per burst (inserted − deleted)

**R-bursts.** A run of events ended by a revision. A revision is a maximal run
of consecutive `delete`/`replace` events; the revision's own events belong to
no burst. The final run of a message, ended by sending rather than by a
revision, is not an R-burst (so a message without revisions has none).
- `n_rbursts`, `rburst_size_max`, `rburst_size_median` (events)
- `n_revisions` (number of revision runs), `n_revisions_leading_edge` (runs
  starting `at_end`)

**Rates** (characters per minute)
- `cpm_product` = `final_length / typing_span`
- `cpm_process` = `chars_inserted / typing_span`
- `cpm_product_with_thinking` = `final_length / response_delay` (NaN if no
  `response_delay_s`)
- `cpm_process_with_thinking` = `chars_inserted / response_delay`

The two "with thinking" rates mix a server-clock denominator with client-side
counts. That is deliberate: the client clock cannot see the time before the
first keystroke. Only the denominator comes from the server.

**Quality flags**
- `n_bulk_inserts`, `chars_bulk_inserted`, `has_bulk_insert` (paste /
  autocomplete candidates)
- `n_time_regressions`, `n_nochange_dropped`
- `final_matches_sent` — last state equals `sent_text` (NaN if unknown)

**Carried through:** every `messages` column.

**Messages without events** (in `messages` but with no text state, or whose
states were all dropped as no-change) still get a row: counts are 0,
`has_bulk_insert` is False, everything else is NaN. Rates and per-minute values
are NaN, never infinite, when their denominator is 0.

### 5.1 Relation to the 2024-25 platform indicators

| 2024-25 platform | engine | status |
|---|---|---|
| `num_bursts_200/2000` | `n_bursts_200/2000` | same definition, up to how the platform treated the first burst and non-text keys |
| `max/mean/median_burst_size_*` | `burst_size_max/mean/median_*` | same unit (events ≈ keystrokes) |
| `r_burst_*` | `n_rbursts`, `rburst_size_*` | **platform definition unknown**; ours follows the P-/R-burst distinction of Chenoweth & Hayes (2001) |
| 4 cpm rates | 4 `cpm_*` rates | same numerators; the platform's "thinking time" start point is unknown |
| `total_keystrokes` | no equivalent | counts non-text keys (arrows, Shift…), which text states cannot see |

Pooling the two cohorts on these names therefore needs a caveat until the
platform code or the 2024-25 raw stream allows a direct check (decision C).

## 6. Configuration and provenance

`MetricConfig` (frozen dataclass, defaults shown; written into every output):

```
pause_thresholds_ms   = (200, 2000)
bulk_insert_min       = 3       # chars inserted in one event -> bulk/paste flag
drop_nochange_events  = True
between_word_chars    = whitespace + ".,;:!?\"'()-"
```

Every CLI run writes `<out>.csv` and `<out>.provenance.json` containing the
package version, git commit (if run from a checkout), the full config, the
adapter name, SHA-256 of each input file, the `CleaningReport`, and row counts
in and out.

## 7. Adapters and CLI

**`languagelab_export`** reads `messages.csv` + `textarea_states.csv`:

- CSV traps: quoted fields contain line breaks (scenario names), so use pyarrow
  with `newlines_in_values=True`. Read only the needed columns (the file is
  ~800 MB).
- Keep only `message_link_status == matched_unique` states (the notice's
  recommendation). Ambiguous keys are counted in the report, not guessed.
- `t_ms` ← `client_timestamp_raw` (ms, confirmed by comparing with the keydown
  stream); `seq` ← `textarea_state_id`.
- `session_id` ← `conversation_id`; `task_id` ← `scenario_name`;
  `response_delay_s` ← `student_response_delay_s`.
- Optional `--filter class_name=...` so a study reads only its classes.

**`languagelab_legacy`** and **`language_hero`** map the existing default
datasets (content per key event) onto the same tables. They are small, and
they prove the format is not EPHEC-specific.

**CLI**

```
keylog-metrics <adapter> <input_dir> --out <path.csv> [--config cfg.json] [--filter col=value ...]
```

## 8. Out of scope for v0.1 (named so they are not forgotten)

- `reconstruct`: rebuild text states from key-only logs (24-25 raw stream if
  recovered; copy-task JSON).
- Using the keydown stream (non-text keys, the 2 students with "Unidentified"
  keys).
- Session/user aggregation (EPHEC aggregates in R; the legacy class still has it).
- Copy-task metrics for 2025-26 (the copy task is a chatbot scenario there; the
  same engine applies, but the trial split is EPHEC-specific).
- Removing or rewriting the legacy class.

## 9. Environment

- uv-managed project: `uv sync` builds `.venv` from `uv.lock`. Python pinned
  in `.python-version` to 3.14 (installed; the old `.venv` was a stale
  `python -m venv` from a previous path and is replaced).
- Runtime dependencies: `pandas>=3.0` (hence Python >=3.11), `numpy>=2`,
  `pyarrow>=17`. pandas 2.x support was dropped after the final review
  (2026-09-24): it could not be tested offline, and every consumer runs from
  `uv.lock`. Code stays independent of the string storage (pyarrow default,
  python if the caller sets `mode.string_storage`). Resolved
  versions come from the local uv cache where possible (pandas 3.0.5,
  numpy 2.5.x, pyarrow 25.0.1 are cached). Dev group: `pytest`.
- Build backend stays hatchling.

## 10. Testing

- **Unit tests per indicator**: tiny hand-written messages (3–10 events) with
  expected values worked out by hand in the test file's comments. Every §5
  indicator is covered, including edge cases: single event, all deletions,
  replace-only (autocorrect), bulk insert, a pause exactly at θ, identical
  timestamps, clock regression.
- **Adapter tests** on synthetic CSVs that reproduce the traps (multiline
  quoted field, ambiguous links, missing `response_delay`).
- **Integration test on real data** is enabled only when
  `KEYLOG_TEST_DATA=<dir>` is set, and skipped otherwise. Real data never enters
  this repo (it is confidential, and the EPHEC texts may contain personal data).
- **Legacy tests** keep passing.
- **Performance check**: the 2025-26 EPHEC subset runs in under 60 s.

## 11. EPHEC integration (in `ephec-bertrand`)

- **No Python environment inside the EPHEC repo.** It lives in Dropbox, whose
  sync has already reverted a git-written file once (2026-09-24); a `.venv` of
  thousands of files there is asking for more of that. Instead `R/00_setup.R`
  holds `KEYLOG_ENGINE`: a local checkout path during development (run with
  `uv run --project <path> keylog-metrics`, which uses the package's own
  environment outside Dropbox) or, once released, a pinned git URL (run with
  `uv tool run --from git+…@v0.1.0 keylog-metrics`). The provenance JSON
  records the exact commit either way.
- `R/00_setup.R`: add the `25-26` entry to `COHORT_FILES`.
- New `R/05a_text_state_metrics.R`: checks `SHA256SUMS`, runs
  `uv run keylog-metrics languagelab_export ...` with the EPHEC class filter,
  and writes `data/03_derived/25-26/message_metrics.csv` + provenance.
- **Not in this spec:** adapting `R/05` to read these metrics (scenario
  mapping, filters, 2024-25 name mapping), the 25-26 copy task and vocabulary.
  Those are the next spec.

## 12. Decisions (confirmed by S. Bibauw, 2026-09-24)

1. **No-change events are dropped** (0.28 %). Alternative: keep them as
   zero-size events (they would still split IKIs).
2. **Bulk-insert threshold = 3 characters** in a single event. It is flagged, not
   removed; the study decides what to exclude. (Autocorrect is a `replace`, not
   an insert, so it is not flagged by this rule.)
3. **Burst size is counted in events**, like 2024-25, with net characters as a
   second column.
4. **R-burst definition** follows Chenoweth & Hayes (2001) (bursts ended by a revision). The platform's is unknown; cite-check before the paper.
5. **Clock regressions**: sort by client time and flag. Alternative: exclude
   the message.
6. **Pause location** uses the character before the insertion point *and* the
   first inserted character (between-word vs within-word), as in §5 and the
   code — not the character before the insertion point alone. Finer
   categories (sentence boundary, etc.) are left for later.
