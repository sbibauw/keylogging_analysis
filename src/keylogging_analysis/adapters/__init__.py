"""Source-format adapters. Each returns an AdapterResult holding canonical KeylogData."""
from . import language_hero, languagelab_export, languagelab_legacy
from .base import AdapterResult, apply_filters

ADAPTERS = {
    "languagelab_export": languagelab_export.load,
    "languagelab_legacy": languagelab_legacy.load,
    "language_hero": language_hero.load,
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown adapter '{name}'; choose one of: {', '.join(sorted(ADAPTERS))}") from None


__all__ = ["ADAPTERS", "AdapterResult", "apply_filters", "get_adapter"]
