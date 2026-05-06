"""Vision-language encoder.

Two backends are exposed under the same interface:

* ``OntologyVLMEncoder`` (default in this codebase): derives fixed semantic
  embeddings for the ontology via SVD of the room/object affinity matrix.
  Produces cosine-comparable text and "image" (set-of-objects) embeddings
  with realistic semantic structure but zero external dependencies.
* ``OpenCLIPEncoder``: a thin wrapper around ``open_clip`` activated only if
  the package is importable. Provided so a real CLIP backbone can be
  swapped in without changing any downstream code.

Both backends return L2-normalised vectors of identical dimensionality so
that retrieval, planner, and graph code is encoder-agnostic.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ...env.ontology import OBJECT_CLASSES, ROOM_TYPES, affinity_matrix


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


class OntologyVLMEncoder:
    """Deterministic semantic encoder for the controlled testbed.

    Embedding construction:

    1. Build the (|objects|, |rooms|) affinity matrix A.
    2. SVD: A = U Σ V^T, take ``U Σ`` truncated to ``embed_dim`` columns as
       object embeddings, and ``V Σ`` truncated similarly as room
       embeddings, then L2-normalise.
    3. Words outside the ontology are mapped to the centroid of objects/rooms
       whose name appears as a substring; otherwise zero.

    The structure of the embedding space is fully determined by the ontology
    (no randomness), so all reported numbers are deterministic w.r.t. seed.
    """

    def __init__(self, embed_dim: int = 32):
        self.embed_dim = int(embed_dim)
        A = affinity_matrix()  # (n_obj, n_rooms)
        # Augment with row-normalised version to encourage room-relative
        # similarity (so two kitchen objects are close even if their primary
        # room weight differs).
        A = A + 0.1
        A = A / A.sum(axis=1, keepdims=True)

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        d = min(self.embed_dim, S.shape[0])
        obj_emb = U[:, :d] * S[:d]
        room_emb = Vt[:d, :].T * S[:d]
        # Pad to embed_dim if needed.
        if d < self.embed_dim:
            pad = self.embed_dim - d
            obj_emb = np.concatenate([obj_emb, np.zeros((obj_emb.shape[0], pad))], axis=1)
            room_emb = np.concatenate([room_emb, np.zeros((room_emb.shape[0], pad))], axis=1)
        self._obj_emb = _normalize(obj_emb.astype(np.float32))
        self._room_emb = _normalize(room_emb.astype(np.float32))
        self._obj_index = {o: i for i, o in enumerate(OBJECT_CLASSES)}
        self._room_index = {r: i for i, r in enumerate(ROOM_TYPES)}

    # -- Object/room atomic lookups --------------------------------------
    def object_embedding(self, obj_class: str) -> np.ndarray:
        return self._obj_emb[self._obj_index[obj_class]]

    def room_embedding(self, room_type: str) -> np.ndarray:
        return self._room_emb[self._room_index[room_type]]

    # -- Public CLIP-like API --------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        """Embed a short natural-language goal of the form 'find a <obj>'."""
        words = text.lower().replace("_", " ").split()
        # Try direct object match first.
        for w in words[::-1]:  # back-to-front (e.g. "find a bed")
            if w in self._obj_index:
                return self._obj_emb[self._obj_index[w]]
            if w in self._room_index:
                return self._room_emb[self._room_index[w]]
        # Fallback: average of any tokens we recognise.
        embs = []
        for w in words:
            if w in self._obj_index:
                embs.append(self._obj_emb[self._obj_index[w]])
            elif w in self._room_index:
                embs.append(self._room_emb[self._room_index[w]])
        if not embs:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return _normalize(np.mean(embs, axis=0))

    def encode_image(self, observation: Dict) -> np.ndarray:
        """Aggregate visible objects and rooms into a single scene embedding."""
        embs: List[np.ndarray] = []
        weights: List[float] = []
        for obj_class, dr, dc in observation.get("visible_objects", []):
            if obj_class in self._obj_index:
                # Closer objects matter more (1 / (1 + dist)).
                w = 1.0 / (1.0 + math.hypot(dr, dc))
                embs.append(self._obj_emb[self._obj_index[obj_class]])
                weights.append(w)
        local_room = observation.get("local_room")
        if local_room in self._room_index:
            embs.append(self._room_emb[self._room_index[local_room]])
            weights.append(1.5)  # local room context is informative
        for rt in observation.get("visible_room_types", []):
            if rt in self._room_index:
                embs.append(self._room_emb[self._room_index[rt]])
                weights.append(0.05)
        if not embs:
            return np.zeros(self.embed_dim, dtype=np.float32)
        E = np.stack(embs, axis=0)
        w = np.asarray(weights, dtype=np.float32)[:, None]
        agg = (E * w).sum(axis=0) / max(w.sum(), 1e-8)
        return _normalize(agg.astype(np.float32))

    # -- Convenience helpers ---------------------------------------------
    def cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(_normalize(a), _normalize(b)))

    @property
    def dim(self) -> int:
        return self.embed_dim


def build_encoder(name: str = "ontology", **kwargs) -> OntologyVLMEncoder:
    name = name.lower()
    if name == "ontology":
        return OntologyVLMEncoder(**kwargs)
    if name == "openclip":
        try:
            from .openclip_backend import OpenCLIPEncoder  # noqa: F401
            return OpenCLIPEncoder(**kwargs)  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenCLIP backend unavailable; install `open_clip` and ensure "
                "`nem_nav.models.perception.openclip_backend` is importable."
            ) from e
    raise ValueError(f"Unknown encoder backend: {name!r}")
