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
