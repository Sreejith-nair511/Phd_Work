"""Configuration loader with OmegaConf / YAML support."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file, resolving 'defaults' inheritance.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Merged configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Resolve 'defaults' inheritance
    if "defaults" in raw:
        base_configs = raw.pop("defaults")
        merged: Dict[str, Any] = {}
        for base_ref in base_configs:
            base_path = config_path.parent / f"{base_ref}.yaml"
            base = load_config(base_path)
            merged = _deep_merge(merged, base)
        raw = _deep_merge(merged, raw)

    return raw


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config to dot-separated keys (for logging)."""
    items: Dict[str, Any] = {}
    for k, v in config.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key))
        else:
            items[new_key] = v
    return items


def get_modality_list(config: Dict[str, Any]) -> list[str]:
    """Return list of enabled modality names from config."""
    modalities = config.get("modalities", {})
    return [m for m, enabled in modalities.items() if enabled]
