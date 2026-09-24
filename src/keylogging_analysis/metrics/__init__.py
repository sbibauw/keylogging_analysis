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
    # pause_time_ms_<theta> is left as NaN (not 0) for messages with fewer than
    # 2 events, including messages without events at all -- spec §4.1.3/§5:
    # it is a timing metric, not a count, so it follows the same NaN rule as
    # typing_span_ms rather than the STATIC_COUNTS fillna(0) above.
    out["has_bulk_insert"] = out["has_bulk_insert"].astype("boolean").fillna(False).astype(bool)
    out["final_matches_sent"] = out["final_matches_sent"].astype("boolean")
    return out.reset_index(), report


__all__ = ["compute_message_metrics", "count_columns"]
