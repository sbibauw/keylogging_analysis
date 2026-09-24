"""What every adapter returns, and the shared message filter."""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..schema import KeylogData


@dataclass
class AdapterResult:
    data: KeylogData
    inputs: list = field(default_factory=list)      # files read, for provenance checksums
    counts: dict = field(default_factory=dict)      # rows read / kept / dropped, per reason


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    """Keep rows where each filter column's value (as text) is one of the listed values."""
    if not filters:
        return df
    for col, values in filters.items():
        if col not in df.columns:
            raise ValueError(f"filter column '{col}' not found; available: {', '.join(map(str, df.columns))}")
        df = df[df[col].astype("string").isin([str(v) for v in values]).fillna(False)]
    return df


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"expected input file not found: {path}")
    return path
