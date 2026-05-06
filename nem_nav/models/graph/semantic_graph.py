"""Semantic topological graph over episodic states.

Nodes summarise *clusters* of nearby observations sharing a common
semantic context (typically a single room). Edges represent observed
spatial transitions (the agent walked from one node region to another)
plus optional semantic-affinity links (cosine similarity above a
threshold).

The graph compresses repeated experiences (multiple visits to the same
kitchen collapse into one node) and provides:

* a *graph prior* for action scoring (distance from the agent's current
  node to a target-related node),
* coarse-grained context for the planner,
* a way to ablate "no graph" trivially (just don't query it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover
    nx = None
    _HAS_NX = False


@dataclass
class GraphNode:
    node_id: int
    centroid_pose: Tuple[float, float]
    centroid_emb: np.ndarray
    room_label: Optional[str]
    member_memory_ids: List[int] = field(default_factory=list)
    visit_count: int = 0


class SemanticGraph:
    def __init__(self, sim_threshold: float = 0.92, pose_radius: float = 3.0):
        if not _HAS_NX:
            raise RuntimeError("networkx is required for SemanticGraph")
        self.sim_threshold = float(sim_threshold)
        self.pose_radius = float(pose_radius)
        self.g = nx.Graph()
        self._nodes: Dict[int, GraphNode] = {}
        self._next_id = 0
        self._last_node_id: Optional[int] = None

    # ---- Construction --------------------------------------------------
    def observe(self, embedding: np.ndarray, pose: Tuple[int, int],
                room_label: Optional[str], memory_id: int) -> int:
        """Either attach observation to a matching node or create a new one."""
        node_id = self._best_match(embedding, pose, room_label)
        if node_id is None:
            node_id = self._create_node(embedding, pose, room_label, memory_id)
        else:
            self._update_node(node_id, embedding, pose, memory_id)
        # Connect to previous node if it differs.
        if self._last_node_id is not None and self._last_node_id != node_id:
            if not self.g.has_edge(self._last_node_id, node_id):
                self.g.add_edge(self._last_node_id, node_id, weight=1.0, kind="transition")
            else:
                self.g[self._last_node_id][node_id]["weight"] += 1.0
        self._last_node_id = node_id
        return node_id

    def add_semantic_edges(self, sim_topk: int = 3) -> None:
        """Add semantic-affinity edges between nodes whose centroids are
        very similar (avoids creating dense graphs in small environments)."""
        if len(self._nodes) < 2:
            return
        ids = list(self._nodes.keys())
        E = np.stack([self._nodes[i].centroid_emb for i in ids], axis=0)
        S = E @ E.T
        for i_idx, ni in enumerate(ids):
            order = np.argsort(-S[i_idx])
            added = 0
            for j_idx in order:
                if int(j_idx) == i_idx:
                    continue
                if S[i_idx, j_idx] < self.sim_threshold:
                    break
                nj = ids[int(j_idx)]
                if not self.g.has_edge(ni, nj):
                    self.g.add_edge(ni, nj, weight=float(S[i_idx, j_idx]), kind="semantic")
                added += 1
                if added >= sim_topk:
                    break

    # ---- Queries -------------------------------------------------------
    def neighborhood(self, node_id: int, hops: int = 1) -> List[int]:
        if node_id not in self._nodes:
            return []
        if hops <= 0:
            return [node_id]
        seen = {node_id}
        frontier = [node_id]
        for _ in range(hops):
            new_frontier = []
            for n in frontier:
                for m in self.g.neighbors(n):
                    if m not in seen:
                        seen.add(m)
                        new_frontier.append(m)
            frontier = new_frontier
        return list(seen)

    def shortest_path_length(self, src: int, dst: int) -> Optional[int]:
        if src not in self._nodes or dst not in self._nodes:
            return None
        try:
            return nx.shortest_path_length(self.g, src, dst)
        except Exception:
            return None

    def best_node_for_query(self, query_emb: np.ndarray) -> Optional[int]:
        if not self._nodes:
            return None
        ids = list(self._nodes.keys())
        E = np.stack([self._nodes[i].centroid_emb for i in ids], axis=0)
        sims = E @ self._normalize(query_emb)
        return int(ids[int(np.argmax(sims))])

    def stats(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self._nodes),
            "n_edges": self.g.number_of_edges(),
            "avg_visits": float(np.mean([n.visit_count for n in self._nodes.values()])) if self._nodes else 0.0,
        }

    # ---- Internals -----------------------------------------------------
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x)
        return x / max(n, 1e-8)

    def _best_match(self, embedding: np.ndarray, pose: Tuple[int, int],
                    room_label: Optional[str]) -> Optional[int]:
        if not self._nodes:
            return None
        emb = self._normalize(embedding)
        best_id = None
        best_score = -math.inf
        for nid, node in self._nodes.items():
            if room_label is not None and node.room_label is not None and node.room_label != room_label:
                continue
            if math.hypot(node.centroid_pose[0] - pose[0],
                          node.centroid_pose[1] - pose[1]) > self.pose_radius:
                continue
            sim = float(np.dot(self._normalize(node.centroid_emb), emb))
            if sim >= self.sim_threshold and sim > best_score:
                best_score = sim
                best_id = nid
        return best_id

    def _create_node(self, embedding: np.ndarray, pose: Tuple[int, int],
                     room_label: Optional[str], memory_id: int) -> int:
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = GraphNode(
            node_id=nid,
            centroid_pose=(float(pose[0]), float(pose[1])),
            centroid_emb=self._normalize(embedding).copy(),
            room_label=room_label,
            member_memory_ids=[memory_id],
            visit_count=1,
        )
        self.g.add_node(nid)
        return nid

    def _update_node(self, node_id: int, embedding: np.ndarray,
                     pose: Tuple[int, int], memory_id: int) -> None:
        node = self._nodes[node_id]
        n = node.visit_count
        node.centroid_pose = (
            (node.centroid_pose[0] * n + pose[0]) / (n + 1),
            (node.centroid_pose[1] * n + pose[1]) / (n + 1),
        )
        new_emb = (node.centroid_emb * n + embedding) / (n + 1)
        node.centroid_emb = self._normalize(new_emb)
        node.member_memory_ids.append(memory_id)
        node.visit_count += 1

    def get_node(self, node_id: int) -> Optional[GraphNode]:
        return self._nodes.get(node_id)
