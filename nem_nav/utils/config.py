"""YAML config loading with dotted-key access and CLI overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        bases = [base] if isinstance(base, str) else list(base)
        merged: Dict[str, Any] = {}
        for b in bases:
            sub = load_yaml(path.parent / b)
            _deep_update(merged, sub)
        _deep_update(merged, cfg)
        return merged
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply CLI overrides of the form 'a.b.c=value'."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override: {ov!r}")
        key, val = ov.split("=", 1)
        try:
            val_parsed = yaml.safe_load(val)
        except Exception:
            val_parsed = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val_parsed
    return cfg


@dataclass
class RunConfig:
    cfg: Dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
