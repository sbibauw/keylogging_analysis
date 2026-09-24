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
