"""Helpers shared by the metric modules. All results are indexed by message_id."""
import numpy as np
import pandas as pd

from ..schema import GROUP


def span_ms(edits: pd.DataFrame) -> pd.Series:
    g = edits.groupby(GROUP, sort=False)["t_ms"]
    return (g.max() - g.min()).astype("float64")


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0 or missing (never inf)."""
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den).where(den > 0, np.nan)


def per_minute(x: pd.Series, span: pd.Series) -> pd.Series:
    return safe_ratio(x, span / 60000.0)


def dist_stats(values: pd.Series, keys: pd.Series, prefix: str) -> pd.DataFrame:
    """mean, median, sd (n-1), iqr (linear quantiles), mad (unscaled) per key."""
    g = values.groupby(keys, sort=False)
    dev = (values - g.transform("median")).abs()
    out = pd.DataFrame({
        f"{prefix}_mean": g.mean(),
        f"{prefix}_median": g.median(),
        f"{prefix}_sd": g.std(),
        f"{prefix}_iqr": g.quantile(0.75) - g.quantile(0.25),
        f"{prefix}_mad": dev.groupby(keys, sort=False).median(),
    })
    out.index.name = GROUP
    return out
