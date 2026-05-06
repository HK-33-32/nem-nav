"""Retrieval scoring on top of EpisodicMemory.

Given a query embedding (typically the current observation), the scorer
returns top-k memory items ranked by a hybrid criterion:

    score(i) = sim_cos(q, e_i) +
               alpha_utility * tanh(utility_i) +
               alpha_goal * sim_cos(goal_emb, e_i) -
               beta_recency * recency_penalty(i) -
               beta_diversity * redundancy_with_already_chosen

Recency penalty up-weights more recent memories slightly. Diversity term
discourages near-duplicate retrievals (MMR-style).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .episodic import EpisodicMemory


@dataclass
class RetrievalConfig:
    top_k: int = 5
    alpha_utility: float = 0.2
    alpha_goal: float = 0.3
    beta_recency: float = 0.0          # 0 by default: keep cosine pure
    beta_diversity: float = 0.4        # MMR strength
    pool_multiplier: int = 4           # candidate pool size = top_k * mult


class HybridRetriever:
    def __init__(self, memory: EpisodicMemory, config: Optional[RetrievalConfig] = None):
        self.memory = memory
        self.config = config or RetrievalConfig()

    def retrieve(self, query: np.ndarray, goal_embedding: Optional[np.ndarray] = None,
                 exclude_ids: Optional[List[int]] = None
                 ) -> List[Tuple[int, float, Dict[str, Any]]]:
        cfg = self.config
        if len(self.memory) == 0:
            return []
        pool = self.memory.retrieve(query,
                                    k=max(cfg.top_k * cfg.pool_multiplier, cfg.top_k),
                                    exclude_ids=exclude_ids)
        if not pool:
            return []
        # Compute hybrid scores in numpy for clarity. The query embedding
        # itself is only used inside ``self.memory.retrieve`` above; the
        # hybrid score below is computed against the goal embedding.
        if goal_embedding is not None:
            g = self.memory._prepare(goal_embedding)
        else:
            g = None

        emb = self.memory._embeddings  # (n, dim)
        ids = np.array([i for i, _, _ in pool])
        sims = np.array([s for _, s, _ in pool])
        utilities = np.array([float(m.get("utility", 0.0)) for _, _, m in pool])
        # Recency: 0 (oldest) -> 1 (newest)
        recency = ids / max(len(self.memory) - 1, 1)
        if g is not None:
            goal_sims = emb[ids] @ g
        else:
            goal_sims = np.zeros_like(sims)

        base = (
            sims
            + cfg.alpha_utility * np.tanh(utilities)
            + cfg.alpha_goal * goal_sims
            - cfg.beta_recency * (1.0 - recency)
        )

        # MMR re-ranking on top of `base` and pool embeddings.
        chosen: List[int] = []
        chosen_emb = []
        remaining = list(range(len(ids)))
        while remaining and len(chosen) < cfg.top_k:
            if not chosen_emb:
                pick_local = int(np.argmax(base[remaining]))
            else:
                ce = np.stack(chosen_emb, axis=0)
                rem_emb = emb[ids[remaining]]
                redundancy = (rem_emb @ ce.T).max(axis=1)
                mmr = base[remaining] - cfg.beta_diversity * redundancy
                pick_local = int(np.argmax(mmr))
            pick = remaining.pop(pick_local)
            chosen.append(pick)
            chosen_emb.append(emb[ids[pick]])

        out: List[Tuple[int, float, Dict[str, Any]]] = []
        for c in chosen:
            mid = int(ids[c])
            out.append((mid, float(base[c]), dict(self.memory._metadata[mid])))
        return out
