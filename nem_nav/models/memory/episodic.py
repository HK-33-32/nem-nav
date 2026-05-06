"""FAISS-backed episodic memory with strict metadata alignment.

The store keeps a parallel list of metadata dicts and exposes a thin
add/retrieve/prune/save/load API. Pruning is implemented by rebuilding the
underlying index, which guarantees metadata indices stay aligned with
embedding rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False


@dataclass
class MemoryItem:
    item_id: int
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Cosine-similarity vector store with metadata.

    Implementation details:

    * Embeddings are stored L2-normalised; cosine = inner product.
    * If FAISS is available we use ``IndexFlatIP``; otherwise a numpy
      fallback computes the same result. Both paths are exercised by tests.
    * Pruning by ``utility`` keeps the top-N items; the index is rebuilt
      to keep memory ↔ index alignment trivially correct.
    """

    def __init__(self, dim: int, max_items: int = 10_000, use_faiss: bool = True):
        self.dim = int(dim)
        self.max_items = int(max_items)
        self.use_faiss = bool(use_faiss and _HAS_FAISS)
        self._embeddings: Optional[np.ndarray] = None  # (n, dim) float32
        self._metadata: List[Dict[str, Any]] = []
        self._index = None
        if self.use_faiss:
            self._index = faiss.IndexFlatIP(self.dim)

    # ----- Core API ------------------------------------------------------
    def __len__(self) -> int:
        return 0 if self._embeddings is None else int(self._embeddings.shape[0])

    def add(self, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        emb = self._prepare(embedding)
        meta = dict(metadata or {})
        meta.setdefault("utility", 0.0)
        if self._embeddings is None:
            self._embeddings = emb[None, :]
        else:
            self._embeddings = np.concatenate([self._embeddings, emb[None, :]], axis=0)
        self._metadata.append(meta)
        if self.use_faiss:
            self._index.add(emb[None, :])
        if len(self) > self.max_items:
            self.prune(self.max_items)
        return len(self) - 1

    def retrieve(self, query: np.ndarray, k: int = 5,
                 exclude_ids: Optional[List[int]] = None
                 ) -> List[Tuple[int, float, Dict[str, Any]]]:
        if len(self) == 0:
            return []
        q = self._prepare(query)[None, :]
        k_eff = min(k + (len(exclude_ids) if exclude_ids else 0), len(self))
        if self.use_faiss:
            D, I = self._index.search(q, k_eff)
            ids = I[0].tolist()
            sims = D[0].tolist()
        else:
            sims_all = (self._embeddings @ q[0])
            order = np.argsort(-sims_all)[:k_eff]
            ids = order.tolist()
            sims = sims_all[order].tolist()
        out: List[Tuple[int, float, Dict[str, Any]]] = []
        excl = set(exclude_ids or [])
        for idx, sim in zip(ids, sims):
            if idx < 0:
                continue
            if idx in excl:
                continue
            out.append((int(idx), float(sim), dict(self._metadata[idx])))
            if len(out) >= k:
                break
        return out

    def update_utility(self, item_id: int, delta: float) -> None:
        if 0 <= item_id < len(self):
            self._metadata[item_id]["utility"] = float(self._metadata[item_id].get("utility", 0.0)) + float(delta)

    def prune(self, max_items: int) -> int:
        """Keep top-``max_items`` items by utility (ties broken by recency)."""
        if len(self) <= max_items:
            return 0
        utilities = np.array([m.get("utility", 0.0) for m in self._metadata], dtype=np.float64)
        recency = np.arange(len(self), dtype=np.float64) / max(len(self), 1)
        keep_score = utilities + 1e-3 * recency
        keep_idx = np.argsort(-keep_score)[:max_items]
        keep_idx_sorted = np.sort(keep_idx)
        new_emb = self._embeddings[keep_idx_sorted].copy()
        new_meta = [self._metadata[i] for i in keep_idx_sorted]
        self._embeddings = new_emb
        self._metadata = new_meta
        if self.use_faiss:
            self._index.reset()
            self._index.add(self._embeddings)
        return int(len(keep_idx_sorted))

    # ----- Persistence ---------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self._embeddings if self._embeddings is not None else np.zeros((0, self.dim), dtype=np.float32))
        with (path / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, default=float)
        cfg = {"dim": self.dim, "max_items": self.max_items, "use_faiss": self.use_faiss}
        with (path / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh)

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicMemory":
        path = Path(path)
        with (path / "config.json").open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        store = cls(dim=cfg["dim"], max_items=cfg["max_items"], use_faiss=cfg["use_faiss"])
        emb = np.load(path / "embeddings.npy")
        with (path / "metadata.json").open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if emb.shape[0] > 0:
            store._embeddings = emb.astype(np.float32)
            store._metadata = [dict(m) for m in meta]
            if store.use_faiss:
                store._index.add(store._embeddings)
        return store

    # ----- Internal ------------------------------------------------------
    def _prepare(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        if v.shape[0] != self.dim:
            raise ValueError(f"Embedding dim {v.shape[0]} != {self.dim}")
        return v
