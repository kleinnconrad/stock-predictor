"""
Central access to the user-defined parameters in config/settings.yaml.
"""
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"


@lru_cache(maxsize=1)
def _read_settings() -> Dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> Dict[str, Any]:
    """
    Loads config/settings.yaml independent of the current working directory.

    Returns:
        Dict[str, Any]: A copy of the parsed settings, safe for the caller to modify.
    """
    return copy.deepcopy(_read_settings())
