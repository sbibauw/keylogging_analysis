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
