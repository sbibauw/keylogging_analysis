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


def _first_of_group(key: pd.Series) -> pd.Series:
    """True at each group's first row.

    ``key`` is the (pyarrow-backed, under pandas 3.x) ``string`` id column, so
    ``key.ne(key.shift())`` comes back as an arrow ``bool[pyarrow]`` Series with
    <NA> at row 0 (no previous value to compare against) instead of True, and
    that extension dtype does not support ``cumsum``. Row 0 has no previous
    value, which unambiguously makes it a group start, so NA is filled True;
    the plain numpy bool cast then makes the result cumsum-able.
    """
    return key.ne(key.shift()).fillna(True).astype(bool)


def pburst_metrics(edits: pd.DataFrame, config: MetricConfig) -> pd.DataFrame:
    key = edits[GROUP]
    first = _first_of_group(key)
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
    first = _first_of_group(key)
    run_start = first | is_rev.ne(is_rev.shift())
    runs = (pd.DataFrame({GROUP: key, "rid": run_start.cumsum(), "is_rev": is_rev,
                          "at_end": edits["at_end"]})
            .groupby("rid", sort=False)
            .agg(**{GROUP: (GROUP, "first"), "is_rev": ("is_rev", "first"),
                    "size": ("is_rev", "size"), "starts_at_end": ("at_end", "first")}))
    # Same arrow-bool/<NA> issue as _first_of_group: the frame's last run has no
    # next row to compare against, and is unambiguously the last run of its message.
    runs["is_last"] = runs[GROUP].ne(runs[GROUP].shift(-1)).fillna(True).astype(bool)

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
