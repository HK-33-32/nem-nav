"""Episode loop runner. Pure-Python; no global state."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


from ..env.semantic_world import SemanticHomeWorld
from ..env.world import build_world
from ..models.perception.encoder import build_encoder
from ..models.policy.policy import NavigationPolicy, PolicyConfig
from .metrics import EpisodeRecord, aggregate


@dataclass
class RunSpec:
    mode: str
    n_scenes: int = 5
    episodes_per_scene: int = 10
    scene_size: int = 24
    n_rooms: int = 6
    view_radius: int = 4
    max_steps: int = 200
    memory_budget: int = 10_000
    retrieval_top_k: int = 5
    planner_every_n_steps: int = 8
    seed: int = 0
    goal_classes: Optional[List[str]] = None  # restrict goal sampling
    encoder: str = "ontology"                  # "ontology" | "openclip"
    llm_backend: str = "none"                  # "none" | "echo" | "llama_cpp"
    llm_model_path: Optional[str] = None       # GGUF path for llama_cpp backend
    env: str = "semantic_home"                 # "semantic_home" | "habitat"
    env_kwargs: Optional[Dict[str, Any]] = None  # extra kwargs for non-default env


def run_one_episode(env: SemanticHomeWorld, policy: NavigationPolicy,
                    episode_id: int, goal_class: Optional[str] = None,
                    log_actions: bool = False
                    ) -> Tuple[EpisodeRecord, Dict[str, Any]]:
    ep = env.sample_episode(episode_id, goal_class=goal_class)
    obs = env.reset(ep)
    policy.reset_episode()

    success = 0
    steps = 0
    actions: List[int] = []
    info_trace: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    while True:
        action, ainfo = policy.act(obs)
        actions.append(action)
        if log_actions:
            info_trace.append(ainfo)
        obs, reward, done, sinfo = env.step(action)
        steps += 1
        if done:
            success = int(bool(sinfo.get("success", False)))
            break
    dt = time.perf_counter() - t0

    dtg = env.shortest_path_length_from_pos()
    if not math.isfinite(dtg):
        dtg = float(env.size)

    record = EpisodeRecord(
        success=success,
        shortest_path_length=ep.shortest_path_length,
        path_length=env.path_length,
        distance_to_goal=float(dtg),
        steps=steps,
        collisions=env.collisions,
        retrieval_latency_ms=float(getattr(policy, "_episode_retrieval_ms", 0.0)),
        planner_latency_ms=float(getattr(policy, "_episode_planner_ms", 0.0)),
    )
    extra = {
        "scene_id": ep.scene_id,
        "goal_class": ep.goal_class,
        "wall_time_s": dt,
        "actions": actions if log_actions else [],
        "memory_size": len(policy.memory),
        "graph_stats": policy.graph.stats() if policy.cfg.use_graph else {},
        "trace": info_trace if log_actions else [],
    }
    return record, extra


def _build_env_for_scene(spec: RunSpec, scene_index: int):
    """Construct an environment for one scene of an experiment matrix run.

    For ``semantic_home`` we keep the historical seeding scheme so that
    pinned reference numbers in the paper remain reproducible. For other
    backends (e.g. ``habitat``) we forward ``spec.env_kwargs`` and let the
    backend handle scene selection internally.
    """
    if spec.env == "semantic_home":
        return SemanticHomeWorld(
            size=spec.scene_size,
            n_rooms_target=spec.n_rooms,
            view_radius=spec.view_radius,
            max_steps=spec.max_steps,
            seed=spec.seed * 1000 + scene_index,
        )
    kwargs = dict(spec.env_kwargs or {})
    kwargs.setdefault("seed", spec.seed * 1000 + scene_index)
    kwargs.setdefault("max_steps", spec.max_steps)
    return build_world(spec.env, **kwargs)


def run_matrix(spec: RunSpec, logger=None) -> Dict[str, Any]:
    """Execute an experiment configuration end-to-end and return aggregates."""
    encoder = build_encoder(spec.encoder)
    cfg = PolicyConfig.for_mode(
        spec.mode,
        memory_budget=spec.memory_budget,
        retrieval_top_k=spec.retrieval_top_k,
        planner_every_n_steps=spec.planner_every_n_steps,
        seed=spec.seed,
        llm_backend=spec.llm_backend,
        llm_model_path=spec.llm_model_path,
    )

    all_records: List[EpisodeRecord] = []
    first_encounter_records: List[EpisodeRecord] = []
    revisit_records: List[EpisodeRecord] = []
    cold_records: List[EpisodeRecord] = []  # first episode in each scene
    per_scene: List[Dict[str, Any]] = []
    per_class: Dict[str, List[EpisodeRecord]] = {}

    for s in range(spec.n_scenes):
        env = _build_env_for_scene(spec, scene_index=s)
        policy = NavigationPolicy(env, encoder, cfg)
        policy.reset_lifelong()
        scene_records: List[EpisodeRecord] = []
        seen_classes: set = set()
        for e in range(spec.episodes_per_scene):
            goal = None
            if spec.goal_classes:
                goal = spec.goal_classes[e % len(spec.goal_classes)]
                if goal not in env.list_goal_classes():
                    continue
            rec, extra = run_one_episode(env, policy, episode_id=e, goal_class=goal)
            gcls = extra["goal_class"]
            is_first = gcls not in seen_classes
            extra["first_encounter"] = is_first
            seen_classes.add(gcls)
            scene_records.append(rec)
            (first_encounter_records if is_first else revisit_records).append(rec)
            if e == 0:
                cold_records.append(rec)
            per_class.setdefault(gcls, []).append(rec)
            if logger is not None:
                logger.log_episode({
                    "mode": spec.mode,
                    "seed": spec.seed,
                    "scene": s,
                    "episode": e,
                    **extra,
                    "metrics": rec.__dict__,
                })
        all_records.extend(scene_records)
        per_scene.append({
            "scene": s,
            **aggregate(scene_records),
        })
        if policy.cfg.use_graph:
            policy.graph.add_semantic_edges()

    summary = aggregate(all_records)
    summary["per_scene"] = per_scene
    summary["per_goal_class"] = {k: aggregate(v) for k, v in per_class.items()}
    summary["first_encounter"] = aggregate(first_encounter_records)
    summary["revisit"] = aggregate(revisit_records)
    summary["cold_start"] = aggregate(cold_records)
    summary["n_records"] = len(all_records)
    summary["mode"] = spec.mode
    summary["seed"] = spec.seed
    return summary
