"""Deterministic seeding helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class SeededRng:
    """Reproducible numpy.random.Generator wrapper with a stable child-stream API."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def child(self, salt: str) -> "SeededRng":
        h = (self.seed * 1_000_003) ^ (abs(hash(salt)) & 0xFFFFFFFF)
        return SeededRng(int(h) & 0x7FFFFFFF)

    def integers(self, low, high=None, size=None):
        return self.rng.integers(low, high, size=size)

    def random(self, size=None):
        return self.rng.random(size=size)

    def choice(self, a, size=None, replace=True, p=None):
        return self.rng.choice(a, size=size, replace=replace, p=p)
