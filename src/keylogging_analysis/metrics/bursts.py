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

    ``key`` is the ``string``-dtype id column, and ``key.ne(key.shift())`` is
    <NA> at row 0 (no previous value to compare against) instead of True,
    under either string storage:

    - pyarrow storage (pandas 3.x's default): the result is an arrow
      ``bool[pyarrow]`` Series, and that extension dtype does not support
      ``cumsum`` at all, so the very first call raises ``TypeError``.
    - python storage (pandas 2.2's default): the result is a nullable
      ``boolean`` Series, whose ``cumsum`` *does* run, but the leading <NA>
      propagates into the cumulative burst/run id, and pandas' ``groupby``
      drops NA-keyed rows by default — so the first event of the first
      message is silently dropped from its burst instead of raising.

    Row 0 has no previous value, which unambiguously makes it a group start,
    so NA is filled True; the plain numpy bool cast then makes the result
    cumsum-able (and dtype-stable) under both storages.
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
        # The "size" NamedAgg is sourced from the (string-dtype) GROUP column, so
        # under pyarrow string storage (pandas 3.x default) its aggregation
        # result comes back as nullable Int64 rather than plain int64, and that
        # extension dtype then leaks into every downstream max/mean/median below
        # — giving Int64/Float64 outputs under pyarrow storage but plain
        # int64/float64 under python storage (pandas 2.2 default) for the exact
        # same input. Casting right after the agg makes the dtype independent
        # of string storage.
        bursts["size"] = bursts["size"].astype("int64")
        g = bursts.groupby(GROUP, sort=False)
        frames.append(pd.DataFrame({
            f"n_bursts_{th}": g.size().astype("int64"),
            f"burst_size_max_{th}": g["size"].max().astype("int64"),
            f"burst_size_mean_{th}": g["size"].mean().astype("float64"),
            f"burst_size_median_{th}": g["size"].median().astype("float64"),
            f"burst_chars_mean_{th}": g["net"].mean().astype("float64"),
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
