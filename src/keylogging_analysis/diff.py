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
