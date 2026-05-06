"""Navigation policy.

Action selection combines a small set of interpretable scores:

    score(cell) = w_frontier * frontier_gain
                + w_semantic * cos(goal_emb, expected_view_emb(cell))
                + w_memory   * memory_score(cell)
                + w_graph    * graph_prior(cell)
                + w_planner  * planner_bias_alignment(cell)

The policy then turns toward the chosen cell and moves forward whenever
forward is navigable, else turns toward it.

Modes (set via ``PolicyConfig``):

* ``baseline``     – only frontier_gain.
* ``random``       – uniform random over actions (sanity check).
* ``nem_nav``      – all components.
* ``no_memory``    – skip memory and retrieval contributions.
* ``no_llm``       – skip planner_bias.
* ``no_graph``     – skip graph_prior.
* ``retrieval_only`` – frontier + memory.
* ``graph_retrieval`` – frontier + memory + graph.
* ``planner_no_memory`` – frontier + planner_bias only.

All modes share the same code path; only weights differ.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...env.semantic_world import (
    FORWARD, STOP, TURN_LEFT, TURN_RIGHT, _DIRS, SemanticHomeWorld
)
from ..memory.episodic import EpisodicMemory
from ..memory.retrieval import HybridRetriever, RetrievalConfig
from ..planner.planner import DeterministicPlanner, LLMPlanner, Plan
from ..graph.semantic_graph import SemanticGraph
from ..perception.encoder import OntologyVLMEncoder
from ...utils.seed import SeededRng


@dataclass
class PolicyConfig:
    mode: str = "nem_nav"
    use_memory: bool = True
    use_graph: bool = True
    use_planner: bool = True
    planner_every_n_steps: int = 8
    retrieval_top_k: int = 5
    memory_budget: int = 10_000
    candidate_radius: int = 6  # how far to score candidate cells
    w_frontier: float = 1.0
    w_semantic: float = 1.5
    w_memory: float = 1.0
    w_graph: float = 0.5
    w_planner: float = 0.8
    stop_threshold: float = 1.5  # stop when target cell is within and goal class visible
    seed: int = 0
    # LLM-backed planner. "none" keeps the rule-based DeterministicPlanner.
    llm_backend: str = "none"               # "none" | "echo" | "llama_cpp"
    llm_model_path: Optional[str] = None    # GGUF path for llama_cpp backend

    @classmethod
    def for_mode(cls, mode: str, **kwargs) -> "PolicyConfig":
        cfg = cls(mode=mode, **kwargs)
        if mode == "baseline":
            cfg.use_memory = False
            cfg.use_graph = False
            cfg.use_planner = False
            cfg.w_semantic = 0.0
            cfg.w_memory = 0.0
            cfg.w_graph = 0.0
            cfg.w_planner = 0.0
        elif mode == "frontier_semantic":
            cfg.use_memory = False
            cfg.use_graph = False
            cfg.use_planner = False
            cfg.w_memory = 0.0
            cfg.w_graph = 0.0
            cfg.w_planner = 0.0
        elif mode == "random":
            cfg.use_memory = False
            cfg.use_graph = False
            cfg.use_planner = False
        elif mode == "nem_nav":
            pass  # all on
        elif mode == "no_memory":
            cfg.use_memory = False
            cfg.w_memory = 0.0
        elif mode == "no_llm":
            cfg.use_planner = False
            cfg.w_planner = 0.0
        elif mode == "no_graph":
            cfg.use_graph = False
            cfg.w_graph = 0.0
        elif mode == "retrieval_only":
            cfg.use_planner = False
            cfg.use_graph = False
            cfg.w_planner = 0.0
            cfg.w_graph = 0.0
        elif mode == "graph_retrieval":
            cfg.use_planner = False
            cfg.w_planner = 0.0
        elif mode == "planner_no_memory":
            cfg.use_memory = False
            cfg.use_graph = False
            cfg.w_memory = 0.0
            cfg.w_graph = 0.0
        else:
            raise ValueError(f"unknown policy mode: {mode!r}")
        return cfg


class NavigationPolicy:
    """End-to-end agent.

    Owns the encoder, episodic memory, retriever, planner, and graph.
    Exposes ``act(observation)`` for the runner. The policy maintains an
    egocentric occupancy/observed map plus a coarse semantic memory of
    visited cells; this is independent of the FAISS episodic memory which
    stores rich per-step embeddings.
    """

    def __init__(self, env: SemanticHomeWorld, encoder: OntologyVLMEncoder,
                 config: Optional[PolicyConfig] = None):
        self.env = env
        self.encoder = encoder
        self.cfg = config or PolicyConfig()
        self.rng = SeededRng(self.cfg.seed)

        self.memory = EpisodicMemory(dim=encoder.dim, max_items=self.cfg.memory_budget)
        self.retriever = HybridRetriever(
            self.memory,
            RetrievalConfig(top_k=self.cfg.retrieval_top_k),
        )
        self.planner = self._build_planner(encoder)
        self.graph = SemanticGraph()

        # Per-episode state
        self.observed: np.ndarray = np.zeros((env.size, env.size), dtype=bool)
        self.visited: np.ndarray = np.zeros((env.size, env.size), dtype=bool)
        # Cell-level semantic cache: maps cell -> last embedding seen there
        self._cell_emb_cache: Dict[Tuple[int, int], np.ndarray] = {}
        # Persistent (cross-episode) summaries — only the components that
        # rely on `use_memory` consume these, so ablations remain clean.
        self._persistent_observed: np.ndarray = np.zeros((env.size, env.size), dtype=bool)
        self._persistent_cell_emb: Dict[Tuple[int, int], np.ndarray] = {}
        self._persistent_cell_objects: Dict[Tuple[int, int], List[Tuple[str, int, int]]] = {}
        self._step_count: int = 0
        self._last_plan: Optional[Plan] = None
        self._last_planner_step: int = -10**9
        self._target_cell: Optional[Tuple[int, int]] = None
        self._stuck_counter: int = 0
        # Per-episode latency accumulators (milliseconds). The runner reads
        # these at the end of each episode and clears them via reset_episode.
        self._episode_retrieval_ms: float = 0.0
        self._episode_planner_ms: float = 0.0

    # ----- Planner construction ----------------------------------------
    def _build_planner(self, encoder: OntologyVLMEncoder):
        """Pick LLMPlanner if cfg.llm_backend != 'none', else DeterministicPlanner."""
        backend = (self.cfg.llm_backend or "none").lower()
        if backend == "none":
            return DeterministicPlanner(encoder)
        from ..planner.llm_backends import build_llm_fn
        kwargs: Dict[str, Any] = {}
        if backend == "llama_cpp":
            kwargs["model_path"] = self.cfg.llm_model_path
            kwargs["seed"] = self.cfg.seed
        llm_fn = build_llm_fn(backend, **kwargs)
        return LLMPlanner(llm_fn=llm_fn, encoder=encoder)

    # ----- Lifecycle ----------------------------------------------------
    def reset_episode(self) -> None:
        # Memory is shared across episodes within a scene, mirroring the
        # "lifelong memory" setting.
        self.observed = np.zeros((self.env.size, self.env.size), dtype=bool)
        self.visited = np.zeros((self.env.size, self.env.size), dtype=bool)
        self._cell_emb_cache.clear()
        self._step_count = 0
        self._last_plan = None
        self._last_planner_step = -10**9
        self._target_cell = None
        self._stuck_counter = 0
        self._episode_retrieval_ms = 0.0
        self._episode_planner_ms = 0.0

    def reset_lifelong(self) -> None:
        self.memory = EpisodicMemory(dim=self.encoder.dim, max_items=self.cfg.memory_budget)
        self.retriever = HybridRetriever(
            self.memory, RetrievalConfig(top_k=self.cfg.retrieval_top_k)
        )
        self.graph = SemanticGraph()
        self._persistent_observed = np.zeros((self.env.size, self.env.size), dtype=bool)
        self._persistent_cell_emb = {}
        self._persistent_cell_objects = {}
        self.reset_episode()

    # ----- Observation ingestion ---------------------------------------
    def _ingest(self, obs: Dict, goal_emb: np.ndarray) -> Tuple[int, np.ndarray]:
        emb = self.encoder.encode_image(obs)
        ar, ac, _ = obs["pose"]
        for cell in obs.get("visible_cells", []):
            self.observed[cell] = True
            self._cell_emb_cache[cell] = emb
            if self.cfg.use_memory:
                self._persistent_observed[cell] = True
                self._persistent_cell_emb[cell] = emb
        if self.cfg.use_memory:
            # Track per-cell visible objects (helps retrieve "I saw X here").
            for obj_class, dr, dc in obs.get("visible_objects", []):
                key = (ar + dr, ac + dc)
                self._persistent_cell_objects.setdefault(key, []).append((obj_class, 0, 0))
        self.visited[ar, ac] = True

        mid = -1
        if self.cfg.use_memory:
            metadata = {
                "pose": (ar, ac),
                "visible_objects": list(obs.get("visible_objects", [])),
                "local_room": obs.get("local_room"),
                "step": int(obs.get("step", self._step_count)),
                "utility": 0.0,
            }
            mid = self.memory.add(emb, metadata)
            if self.cfg.use_graph:
                self.graph.observe(emb, (ar, ac), obs.get("local_room"), mid)
        return mid, emb

    # ----- Core action selection ---------------------------------------
    def act(self, obs: Dict) -> Tuple[int, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        if self.cfg.mode == "random":
            return int(self.rng.choice(
                np.array([FORWARD, TURN_LEFT, TURN_RIGHT, STOP]),
                p=[0.45, 0.25, 0.25, 0.05])), info

        goal_emb = self.encoder.encode_text(obs["goal_text"])
        mid, current_emb = self._ingest(obs, goal_emb)

        # Check stop condition: any visible object equals the goal class and
        # adjacent.
        goal_class = obs.get("goal_class")
        ar, ac, ah = obs["pose"]
        if goal_class is not None:
            for cls, dr, dc in obs.get("visible_objects", []):
                if cls == goal_class and abs(dr) + abs(dc) <= self.env.success_radius:
                    info["stop_reason"] = "goal_visible_adjacent"
                    return STOP, info

        # Periodic planner refresh.
        if self.cfg.use_planner and (
            self._last_plan is None
            or self._step_count - self._last_planner_step >= self.cfg.planner_every_n_steps
        ):
            if self.cfg.use_memory:
                _t = time.perf_counter()
                retrieved = self.retriever.retrieve(current_emb, goal_emb)
                self._episode_retrieval_ms += (time.perf_counter() - _t) * 1000.0
            else:
                retrieved = []
            _t = time.perf_counter()
            self._last_plan = self.planner.plan(obs["goal_text"], retrieved, current_pose=(ar, ac))
            self._episode_planner_ms += (time.perf_counter() - _t) * 1000.0
            # If retrieval was empty but the graph has nodes from
            # already-visited (non-target) rooms, derive an explore-away
            # bias direction: head away from the centroid of visited rooms.
            if (
                self._last_plan.bias_direction is None
                and self.cfg.use_graph
                and self.graph._nodes
            ):
                centroids = np.array([n.centroid_pose for n in self.graph._nodes.values()],
                                     dtype=np.float32)
                cx, cy = centroids.mean(axis=0)
                dr, dc = ar - cx, ac - cy
                n = float(np.hypot(dr, dc))
                if n > 1e-6:
                    self._last_plan.bias_direction = (dr / n, dc / n)
                    self._last_plan.confidence = max(self._last_plan.confidence, 0.4)
                    self._last_plan.rationale += "; explore-away bias from graph"
            self._last_planner_step = self._step_count
            info["planner"] = self._last_plan.to_dict()

        # If memory remembers a cell that contained the exact goal class,
        # head toward a navigable cell adjacent to it. This is the canonical
        # "memory pays off" behaviour: the agent saw a bed in episode 1 and
        # is asked to find a bed in episode 2.
        remembered_goal = None
        if self.cfg.use_memory and goal_class is not None:
            best_dist = float("inf")
            for cell, objs in self._persistent_cell_objects.items():
                if not any(o[0] == goal_class for o in objs):
                    continue
                # Snap to the nearest navigable neighbour we know exists.
                approach = self._nearest_navigable_neighbor(cell)
                if approach is None:
                    continue
                d = abs(approach[0] - ar) + abs(approach[1] - ac)
                if d < best_dist:
                    best_dist = d
                    remembered_goal = approach

        # Pick or refresh a target cell among reachable candidates.
        if (
            self._target_cell is None
            or (ar, ac) == self._target_cell
            or self._stuck_counter > 6
            or (remembered_goal is not None and self._target_cell != remembered_goal)
        ):
            if remembered_goal is not None:
                self._target_cell = remembered_goal
                info["target_cell"] = self._target_cell
                info["target_source"] = "memory_recall"
            else:
                self._target_cell = self._select_target(obs, current_emb, goal_emb)
                info["target_cell"] = self._target_cell
                info["target_source"] = "frontier"
                if hasattr(self, "_last_score_breakdown"):
                    info["scores"] = dict(self._last_score_breakdown)
            self._stuck_counter = 0

        # Convert target cell into a low-level action.
        action = self._action_toward(self._target_cell, (ar, ac, ah))
        # Track stuck-ness: if we tried FORWARD but the env will block, our
        # path planner already returns TURN; nonetheless track repeated
        # non-progress.
        if action == FORWARD:
            dr, dc = _DIRS[ah]
            if not self.env._is_navigable(ar + dr, ac + dc):
                action = TURN_RIGHT
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        else:
            self._stuck_counter += 1
        self._step_count += 1
        return action, info

    # ----- Candidate scoring -------------------------------------------
    def _select_target(self, obs: Dict, current_emb: np.ndarray,
                       goal_emb: np.ndarray) -> Tuple[int, int]:
        ar, ac, _ = obs["pose"]
        cands = self._candidate_cells((ar, ac))
        if not cands:
            return (ar, ac)

        # Frontier gain: count of unobserved neighbours.
        f_gain = np.array([self._frontier_gain(c) for c in cands], dtype=np.float32)

        # Semantic score: cosine between goal_emb and the encoded view that
        # has been seen most recently at that cell or its 1-hop neighbours.
        s_score = np.array([self._semantic_score(c, goal_emb) for c in cands], dtype=np.float32)

        # Memory score: retrieval similarity if we treat the candidate's
        # cached embedding as a probe; falls back to current_emb if missing.
        if self.cfg.use_memory and len(self.memory) > 0:
            m_score = np.array([self._memory_score(c, goal_emb, current_emb) for c in cands],
                               dtype=np.float32)
        else:
            m_score = np.zeros(len(cands), dtype=np.float32)

        # Graph prior: shorter graph distance from current node to a
        # target-related node => better.
        if self.cfg.use_graph:
            g_score = self._graph_scores(cands, goal_emb, (ar, ac))
        else:
            g_score = np.zeros(len(cands), dtype=np.float32)

        # Planner alignment.
        if self.cfg.use_planner and self._last_plan is not None and self._last_plan.bias_direction is not None:
            p_score = np.array([self._planner_alignment(c, (ar, ac), self._last_plan)
                                for c in cands], dtype=np.float32)
        else:
            p_score = np.zeros(len(cands), dtype=np.float32)

        # Distance penalty: prefer closer candidates to avoid yo-yo behavior.
        dist = np.array([abs(c[0] - ar) + abs(c[1] - ac) for c in cands], dtype=np.float32)
        d_pen = -0.05 * dist

        scores = (
            self.cfg.w_frontier * f_gain
            + self.cfg.w_semantic * s_score
            + self.cfg.w_memory * m_score
            + self.cfg.w_graph * g_score
            + self.cfg.w_planner * p_score
            + d_pen
        )
        # Tie-break with deterministic noise.
        scores = scores + 1e-6 * self.rng.rng.standard_normal(scores.shape)
        # Expose interpretable score breakdown so the runner / gallery
        # script can render per-step contributions for figures and
        # supplementary analysis.
        self._last_score_breakdown = {
            "frontier": float(f_gain.max()) if f_gain.size else 0.0,
            "semantic": float(s_score.max()) if s_score.size else 0.0,
            "memory": float(m_score.max()) if m_score.size else 0.0,
            "graph": float(g_score.max()) if g_score.size else 0.0,
            "planner": float(p_score.max()) if p_score.size else 0.0,
        }
        return cands[int(np.argmax(scores))]

    def _candidate_cells(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Frontier candidates: cells on the cumulative explored boundary.

        With ``use_memory`` enabled the cumulative map spans previous
        episodes, so the agent can target rooms it has only ever seen
        before — this is the entire point of episodic memory.
        """
        observed = self._persistent_observed if self.cfg.use_memory else self.observed
        seen = {pos}
        q = deque([(pos, 0)])
        cands: List[Tuple[int, int]] = []
        max_d = self.cfg.candidate_radius * 4 if self.cfg.use_memory else self.cfg.candidate_radius
        while q:
            (r, c), d = q.popleft()
            if d > 0 and self._is_frontier_in((r, c), observed):
                cands.append((r, c))
            if d >= max_d:
                continue
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in seen:
                    continue
                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size):
                    continue
                if self.env.grid[nr, nc] == 1:
                    continue
                if (nr, nc) in self.env.object_at and (nr, nc) != pos:
                    continue
                if not observed[nr, nc]:
                    continue
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
        if not cands:
            cands = [c for c in seen if c != pos and self.env._is_navigable(*c)]
            if not cands:
                cands = [pos]
        return cands

    def _nearest_navigable_neighbor(self, cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        for dr, dc in _DIRS:
            nr, nc = cell[0] + dr, cell[1] + dc
            if self.env._is_navigable(nr, nc):
                return (nr, nc)
        return None

    def _is_frontier_in(self, cell: Tuple[int, int], observed: np.ndarray) -> bool:
        if not observed[cell] or not self.env._is_navigable(*cell):
            return False
        for dr, dc in _DIRS:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < self.env.size and 0 <= nc < self.env.size and not observed[nr, nc]:
                return True
        return False

    def _is_frontier(self, cell: Tuple[int, int]) -> bool:
        if not self.observed[cell] or not self.env._is_navigable(*cell):
            return False
        for dr, dc in _DIRS:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < self.env.size and 0 <= nc < self.env.size and not self.observed[nr, nc]:
                return True
        return False

    def _frontier_gain(self, cell: Tuple[int, int]) -> float:
        observed = self._persistent_observed if self.cfg.use_memory else self.observed
        gain = 0
        vr = self.env.view_radius
        for dr in range(-vr, vr + 1):
            for dc in range(-vr, vr + 1):
                if dr * dr + dc * dc > vr * vr:
                    continue
                rr, cc = cell[0] + dr, cell[1] + dc
                if 0 <= rr < self.env.size and 0 <= cc < self.env.size:
                    if not observed[rr, cc] and self.env.grid[rr, cc] == 0:
                        gain += 1
        return float(gain) / max(1, vr * vr)

    def _semantic_score(self, cell: Tuple[int, int], goal_emb: np.ndarray) -> float:
        emb = self._cell_emb_cache.get(cell)
        if emb is None:
            # Use the average embedding of observed neighbours.
            embs = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    e = self._cell_emb_cache.get((cell[0] + dr, cell[1] + dc))
                    if e is not None:
                        embs.append(e)
            if not embs:
                return 0.0
            emb = np.mean(embs, axis=0)
            emb = emb / max(np.linalg.norm(emb), 1e-8)
        return float(np.dot(emb, goal_emb))

    def _memory_score(self, cell: Tuple[int, int], goal_emb: np.ndarray,
                      current_emb: np.ndarray) -> float:
        emb = self._cell_emb_cache.get(cell, current_emb)
        # Top-k retrieval averaged similarity to goal.
        _t = time.perf_counter()
        retrieved = self.memory.retrieve(emb, k=self.cfg.retrieval_top_k)
        self._episode_retrieval_ms += (time.perf_counter() - _t) * 1000.0
        if not retrieved:
            return 0.0
        sims = []
        for mid, score, meta in retrieved:
            mem_emb = self.memory._embeddings[mid]
            sims.append(float(np.dot(mem_emb, goal_emb)))
        return float(np.mean(sims))

    def _graph_scores(self, cands: List[Tuple[int, int]],
                      goal_emb: np.ndarray, current_pos: Tuple[int, int]) -> np.ndarray:
        if not self.graph._nodes:
            return np.zeros(len(cands), dtype=np.float32)
        target_node = self.graph.best_node_for_query(goal_emb)
        if target_node is None:
            return np.zeros(len(cands), dtype=np.float32)
        scores = np.zeros(len(cands), dtype=np.float32)
        # For each candidate, find the node whose centroid is closest in
        # Euclidean distance and use its graph-distance to the target node
        # as the prior (closer => higher score).
        node_ids = list(self.graph._nodes.keys())
        node_centroids = np.array(
            [self.graph._nodes[nid].centroid_pose for nid in node_ids], dtype=np.float32
        )
        for i, (r, c) in enumerate(cands):
            d = np.hypot(node_centroids[:, 0] - r, node_centroids[:, 1] - c)
            nearest = node_ids[int(np.argmin(d))]
            spl = self.graph.shortest_path_length(nearest, target_node)
            if spl is None:
                continue
            scores[i] = 1.0 / (1.0 + spl)
        return scores

    def _planner_alignment(self, cell: Tuple[int, int],
                           current_pos: Tuple[int, int], plan: Plan) -> float:
        if plan.bias_direction is None:
            return 0.0
        dr = cell[0] - current_pos[0]
        dc = cell[1] - current_pos[1]
        n = math.hypot(dr, dc)
        if n < 1e-6:
            return 0.0
        ux, uy = dr / n, dc / n
        bx, by = plan.bias_direction
        return float(plan.confidence * (ux * bx + uy * by))

    # ----- Low-level action -------------------------------------------
    def _action_toward(self, target: Tuple[int, int], pose: Tuple[int, int, int]) -> int:
        ar, ac, ah = pose
        if (ar, ac) == target:
            self._target_cell = None  # force re-selection
            return TURN_RIGHT
        path = self._bfs_path((ar, ac), target)
        if (path is None or len(path) < 2) and self.cfg.use_memory:
            path = self._bfs_path((ar, ac), target, observed_mask=self._persistent_observed)
        if path is None or len(path) < 2:
            path = self._bfs_path((ar, ac), target, restrict_to_observed=False)
        if path is None or len(path) < 2:
            # Unreachable target — drop it, head to nearest current frontier.
            self._target_cell = None
            return TURN_RIGHT
        nxt = path[1]
        dr, dc = nxt[0] - ar, nxt[1] - ac
        try:
            desired_h = _DIRS.index((dr, dc))
        except ValueError:
            return TURN_RIGHT
        if desired_h == ah:
            return FORWARD
        diff = (desired_h - ah) % 4
        if diff == 1:
            return TURN_RIGHT
        if diff == 3:
            return TURN_LEFT
        return TURN_RIGHT

    def _bfs_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  restrict_to_observed: bool = True,
                  observed_mask: Optional[np.ndarray] = None,
                  ) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return [start]
        if observed_mask is None:
            observed_mask = self.observed
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                path = [cur]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            for dr, dc in _DIRS:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < self.env.size and 0 <= nc < self.env.size):
                    continue
                if (nr, nc) in prev:
                    continue
                if not self.env._is_navigable(nr, nc):
                    continue
                if restrict_to_observed and not observed_mask[nr, nc]:
                    continue
                prev[(nr, nc)] = cur
                q.append((nr, nc))
        return None
