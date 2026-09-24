"""Metric configuration: every analytic choice that changes an indicator's value."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class MetricConfig:
    pause_thresholds_ms: tuple = (200, 2000)
    bulk_insert_min: int = 3
    drop_nochange_events: bool = True
    # ' plus its typographic stand-ins (U+2019 right single quotation mark,
    # U+00B4 acute accent typed alone): all apostrophes separate words.
    between_word_chars: str = " \t\r\n.,;:!?\"'\u2019\u00b4()-"

    def __post_init__(self):
        th = tuple(int(x) for x in self.pause_thresholds_ms)
        if not th:
            raise ValueError("pause_thresholds_ms must not be empty")
        if any(x <= 0 for x in th):
            raise ValueError(f"pause thresholds must be positive, got {th}")
        if len(set(th)) != len(th):
            raise ValueError(f"pause thresholds must be distinct, got {th}")
        object.__setattr__(self, "pause_thresholds_ms", tuple(sorted(th)))
        if int(self.bulk_insert_min) < 2:
            raise ValueError("bulk_insert_min must be >= 2 (a 1-character insert is ordinary typing)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pause_thresholds_ms"] = list(self.pause_thresholds_ms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MetricConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        d = dict(d)
        if "pause_thresholds_ms" in d:
            d["pause_thresholds_ms"] = tuple(d["pause_thresholds_ms"])
        return cls(**d)

    @classmethod
    def from_json(cls, path) -> "MetricConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
