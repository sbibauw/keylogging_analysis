# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`keylogging_analysis` is a Python package for processing and analyzing keylogging data from two educational platforms: **Language Hero** ("lh") and **Language Lab** ("ll"). It provides writing process metrics (IKI, pauses, bursts, revisions) for L2 language fluency research.

## Build, Install and Test

```bash
pip install -e .                # editable install for development
python -m pytest tests/ -v      # run tests
```

Build system: Hatchling (configured in `pyproject.toml`). Dependencies: pandas, numpy.

## Project Structure

- `src/keylogging_analysis/classes.py` — Core `KeyLoggingDataFrame` class (extends `pd.DataFrame`). Contains all data loading, preprocessing, and analysis methods.
- `src/keylogging_analysis/help_functions.py` — Utility functions: column name generation, action/span detection helpers, burst metrics.
- `src/keylogging_analysis/__init__.py` — Exports `KeyLoggingDataFrame`.
- `src/keylogging_analysis/data/` — Default CSV datasets (`lh_default.csv` ~588k rows, `ll_default.csv` ~23 rows).
- `tests/test_smoke.py` — Smoke tests for the core pipeline.

## Architecture

The entire API is the `KeyLoggingDataFrame` class, which subclasses `pd.DataFrame` with `_metadata = ['system']`. The pattern for all `add_*` methods is:

1. Validate parameters
2. Copy `self`, sort by `['message_id', 'key_time']`
3. Compute the metric on the copy
4. Merge result back into `self` on `key_id`
5. Reinitialize via `self.__init__(merged_df)` and return `self`

This in-place mutation via `__init__` is a deliberate pattern used throughout — preserve it when adding new methods.

### Two data systems

The `system` parameter (`"lh"` or `"ll"`) determines column mappings, filtering options, and date parsing. Language Hero data has additional filtering (anonymized, nonsense, native speaker removal). Language Lab data has synthetic `event_for_message_type` and `user_status` columns added during loading.

### Key columns after loading

- `key_id`, `message_id`, `session_id` — identifiers
- `content` — cumulative message text at each keystroke
- `key_time` — timestamp (datetime)
- `event_for_message_type` — 0 = typing event, 5 = first event (non-typing events filtered in most methods)
- `message_content`, `user_status`

### Method chain

Typical usage: `load_data` -> `add_iki()` -> `add_pause()` -> `add_pburst()` -> `add_action()` -> `add_span()` -> analysis methods (`burst_dataframe()`, `pause_analysis()`).

### IOB tagging

Burst columns (`pburst`, `rburst`) use IOB format: `'B'` (beginning), `'I'` (inside), `'O'` (outside/NA).

## Development Status

Version 0.0.1. Remaining stubs/incomplete methods:
- `revision_dataframe()` — stub (`pass`)
- `pause_dataframe()` — no implementation body
- `_drop_native()` — message IDs list needs verification (currently same as nonsense list; marked with TODO)

## Language

Code and API are in English. The research context involves French/Dutch L2 learners.
