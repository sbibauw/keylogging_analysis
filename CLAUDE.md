# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`keylogging_analysis` is a Python package for processing and analyzing keylogging data from two educational platforms: **Language Hero** ("lh") and **Language Lab** ("ll"). It provides writing process metrics (IKI, pauses, bursts, revisions) for L2 language fluency research.

## Build, Install and Test

```bash
uv sync                                  # install/update the environment
uv run pytest -q                         # run the test suite (95 passed + 5 skipped)
uv run keylog-metrics --help             # CLI entry point
```

Add `--offline` to any `uv run ...` command when the network is unavailable; every
dependency needed is already in the uv cache.

Opt-in real-data test (skipped unless `KEYLOG_TEST_DATA` is set — never points at
data inside this repo):

```bash
KEYLOG_TEST_DATA=<export dir> uv run pytest tests/test_real_languagelab_export.py -v -s
```

Build system: Hatchling (configured in `pyproject.toml`). Runtime dependencies:
`pandas>=2.2`, `numpy>=2`, `pyarrow>=17`. Dev dependency: `pytest`.

## Project Structure

- `src/keylogging_analysis/classes.py` — Core `KeyLoggingDataFrame` class (composition wrapper around `pd.DataFrame`). Contains all data loading, preprocessing, and analysis methods.
- `src/keylogging_analysis/help_functions.py` — Utility functions: column name generation, action/span detection helpers, burst metrics.
- `src/keylogging_analysis/__init__.py` — Exports `KeyLoggingDataFrame` and `__version__`.
- `src/keylogging_analysis/data/` — Default CSV datasets (`lh_default.csv` ~588k rows, `ll_default.csv` ~23 rows) and JSON filter lists (`lh_nonsense_message_ids.json`, `lh_native_message_ids.json`).
- `tests/test_smoke.py` — Smoke tests for the core pipeline.

## Text-state engine

Added in v0.1 (spec: `docs/superpowers/specs/2026-09-24-text-state-engine-design.md`;
plan: `docs/superpowers/plans/2026-09-24-text-state-engine.md`). It computes one row of
writing-process indicators per message from logs that record the full text at each
event ("text states"), as a set of pure functions on DataFrames, independent of the
legacy class below:

- `schema.py` — `KeylogData`, column contracts, `validate()`
- `config.py` — `MetricConfig` (frozen dataclass, JSON round-trip)
- `clean.py` — event ordering, duplicate/no-change handling, `CleaningReport`
- `diff.py` — per-event edit derivation (`edit_between`, `derive_edits`): what changed, where
- `metrics/timing.py` — intervals, pauses, pause location
- `metrics/bursts.py` — P-bursts (per pause threshold), R-bursts
- `metrics/product.py` — volume, revision, process/product, rates, quality flags
- `metrics/__init__.py` — `compute_message_metrics(data, config)`, `count_columns`
- `adapters/__init__.py` — registry: adapter name -> loader (`ADAPTERS`, `get_adapter`)
- `adapters/base.py` — shared `AdapterResult`, `apply_filters`, `require_file`
- `adapters/languagelab_export.py` — 2025-26 LanguageLab analyst export
- `adapters/languagelab_legacy.py` — old `ll` schema (`ll_default.csv`)
- `adapters/language_hero.py` — `lh` schema (`lh_default.csv`)
- `provenance.py` — engine version, git commit, config, input checksums, cleaning counts
- `cli.py` — `keylog-metrics` entry point

`classes.py` (`KeyLoggingDataFrame`) and `help_functions.py` are **legacy**: they have
external users (Thonissen) and are not to be extended — new writing-process metrics go
in the modules above.

Known legacy bug (not fixed, recorded here so it isn't rediscovered):
`KeyLoggingDataFrame._user_id_col` resolves the user id column to `PERSONA_ID` for
`system="lh"`, but `PERSONA_ID` is empty throughout `lh_default.csv` — `USER_ID` is the
column that is actually filled. The new `language_hero` adapter uses `USER_ID`.

Known pandas-3 pitfall: with pyarrow-backed string ids, `s.ne(s.shift())` yields `<NA>`
at row 0 (not `True`), and `bool[pyarrow]` has no `.cumsum()`. `schema.first_of_group()`
(`.fillna(True).astype(bool)`) is the one safe helper, used by `clean.py`, `diff.py`
and `metrics/bursts.py`; cast
aggregation outputs explicitly so dtypes match under both pandas 2.2 and 3.x.

## Architecture

The entire API is the `KeyLoggingDataFrame` class, which wraps a `pd.DataFrame` via **composition** (`self.df`). Convenience proxies (`__getitem__`, `__setitem__`, `__len__`, `columns`, `empty`, `shape`) delegate to `self.df`.

### Data loading — factory classmethods

- `KeyLoggingDataFrame.from_default(system, ...)` — load bundled default dataset
- `KeyLoggingDataFrame.from_files(system, path_keys, path_messages, ...)` — load from CSV files

Both return a new `KeyLoggingDataFrame` instance.

### Pattern for all `add_*` methods

1. Validate parameters (check `self.df.columns`, `self.df.empty`)
2. `df = self.df.copy()`, sort by `['message_id', 'key_time']`
3. Compute the metric on the plain DataFrame copy
4. Merge result back via `self._merge_and_update(df, colnames)` — merges on `key_id`
5. `return self` (enables method chaining)

### `_merge_and_update(self, computed_df, colnames)`

Merges computed columns from a plain DataFrame back into `self.df` on `key_id`. No reinit needed.

### `_compute_spans` static method

Span computation is extracted into `@staticmethod _compute_spans(df, selection, colnames)` so both `add_span` and internal callers (`add_length`, `add_distance_to_end`) can use it on plain DataFrames without needing a `KeyLoggingDataFrame` instance.

### Two data systems

The `system` parameter (`"lh"` or `"ll"`) determines column mappings, filtering options, and date parsing. Language Hero data has additional filtering (anonymized, nonsense, native speaker removal). Language Lab data has synthetic `event_for_message_type` and `user_status` columns added during loading.

### Key columns after loading

- `key_id`, `message_id`, `session_id` — identifiers
- `content` — cumulative message text at each keystroke
- `key_time` — timestamp (datetime)
- `event_for_message_type` — 0 = typing event, 5 = first event (non-typing events filtered in most methods)
- `message_content`, `user_status`

### Method chain

Typical usage: `from_default`/`from_files` -> `add_iki()` -> `add_pause()` -> `add_pburst()` -> `add_action()` -> `add_rburst()` -> `add_span()` -> analysis methods (`burst_dataframe()`, `pause_analysis()`, `fluency_metrics()`).

### `fluency_metrics(level=...)`

Extracts aggregated fluency profiles at three granularity levels:
- `"message"` — one row per message with IKI, pause, burst, revision, and production metrics
- `"session"` — groups message-level metrics by `[user_id, session_id]` with mean/median/std
- `"user"` — groups by `[user_id]`

Requires `add_iki`, `add_pause`, `add_pburst`, `add_action`, `add_rburst` to be called first. Uses `_user_id_col` property to resolve system-specific user ID column (PERSONA_ID for lh, user_id for ll).

### IOB tagging

Burst columns (`pburst`, `rburst`) use IOB format: `'B'` (beginning), `'I'` (inside), `'O'` (outside/NA).

### Logging

Progress messages use `logging.info()`. Implicit default warnings use `warnings.warn()`.

## Development Status

Version 0.1.0. This section covers the legacy `KeyLoggingDataFrame` API only —
see "Text-state engine" above for the new module set. Remaining stubs (raise
`NotImplementedError`) in the legacy class:
- `pause_dataframe()`, `revision_dataframe()`, `pburst_analysis()`
- `_drop_native()` — message IDs need verification (currently same as nonsense list; marked with TODO)

## Language

Code and API are in English. The research context involves French/Dutch L2 learners.
