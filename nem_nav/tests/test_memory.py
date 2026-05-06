"""Unit tests for EpisodicMemory."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from nem_nav.models.memory.episodic import EpisodicMemory


def _rand_unit(dim: int, n: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_add_and_retrieve_returns_top_match():
    mem = EpisodicMemory(dim=8)
    embs = _rand_unit(8, n=5, seed=1)
    for i, e in enumerate(embs):
        mem.add(e, {"i": i})
    out = mem.retrieve(embs[3], k=1)
    assert len(out) == 1
    assert out[0][0] == 3
    assert out[0][1] > 0.99


def test_metadata_alignment_after_prune():
    mem = EpisodicMemory(dim=8, max_items=3)
    embs = _rand_unit(8, n=10, seed=2)
    for i, e in enumerate(embs):
        mem.add(e, {"i": i, "utility": float(i)})
    assert len(mem) == 3
    # After utility-aware pruning the top-3 utilities (i=7,8,9) should remain.
    kept = sorted(m["i"] for m in mem._metadata)
    assert kept == [7, 8, 9]
    # Each kept embedding should still match its metadata index by retrieval.
    for emb, meta in zip(mem._embeddings, mem._metadata):
        out = mem.retrieve(emb, k=1)
        assert out[0][2]["i"] == meta["i"]


def test_save_load_roundtrip():
    mem = EpisodicMemory(dim=8)
    embs = _rand_unit(8, n=4, seed=3)
    for i, e in enumerate(embs):
        mem.add(e, {"i": i})
    with tempfile.TemporaryDirectory() as td:
        mem.save(td)
        mem2 = EpisodicMemory.load(td)
    assert len(mem2) == 4
    out = mem2.retrieve(embs[2], k=1)
    assert out[0][2]["i"] == 2


def test_empty_retrieval_is_safe():
    mem = EpisodicMemory(dim=4)
    assert mem.retrieve(np.ones(4, dtype=np.float32), k=3) == []


def test_dim_mismatch_raises():
    mem = EpisodicMemory(dim=4)
    with pytest.raises(ValueError):
        mem.add(np.ones(5, dtype=np.float32), {})
