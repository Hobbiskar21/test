"""
src/utils/config_loader.py
----------------------------
Loads config/config.yaml once and caches it.
Every module calls get_config() instead of reading YAML themselves.
"""

import yaml
import os
from functools import lru_cache

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "config.yaml"
)


@lru_cache(maxsize=1)
def get_config() -> dict:
    path = os.path.abspath(CONFIG_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found at: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg