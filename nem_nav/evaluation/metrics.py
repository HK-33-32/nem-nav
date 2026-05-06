"""Standard embodied-navigation metrics.

Definitions follow Anderson et al. 2018 ("On Evaluation of Embodied
Navigation Agents"):

* Success: 1 if the agent issued STOP within ``success_radius`` of any
  goal target, else 0.
* SPL: Success-weighted by (normalized) inverse Path Length:
  ``SPL = (1/N) Σ S_i * l_i / max(p_i, l_i)`` where ``l_i`` is the
  shortest path length and ``p_i`` is the path actually traversed.
* DTG: Distance to goal at episode end (in grid cells).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class EpisodeRecord:
    success: int
    shortest_path_length: float
    path_length: float
    distance_to_goal: float
    steps: int
    collisions: int
    retrieval_latency_ms: float = 0.0
    planner_latency_ms: float = 0.0


def compute_success(records: Iterable[EpisodeRecord]) -> float:
    recs = list(records)
    if not recs:
        return 0.0
    return sum(r.success for r in recs) / len(recs)


def compute_spl(records: Iterable[EpisodeRecord]) -> float:
    recs = list(records)
    if not recs:
        return 0.0
    total = 0.0
    for r in recs:
        if r.success and r.path_length > 0 and r.shortest_path_length > 0:
            total += r.shortest_path_length / max(r.path_length, r.shortest_path_length)
    return total / len(recs)


def compute_dtg(records: Iterable[EpisodeRecord]) -> float:
    recs = list(records)
    if not recs:
        return 0.0
    return sum(r.distance_to_goal for r in recs) / len(recs)


def compute_episode_length(records: Iterable[EpisodeRecord]) -> float:
    recs = list(records)
    if not recs:
        return 0.0
    return sum(r.steps for r in recs) / len(recs)


def aggregate(records: Iterable[EpisodeRecord]) -> dict:
    recs = list(records)
    n = len(recs)
    return {
        "n_episodes": n,
        "success": compute_success(recs),
        "spl": compute_spl(recs),
        "dtg": compute_dtg(recs),
        "ep_len": compute_episode_length(recs),
        "collisions": (sum(r.collisions for r in recs) / n) if recs else 0.0,
        "retrieval_latency_ms_mean":
            (sum(r.retrieval_latency_ms for r in recs) / n) if recs else 0.0,
        "planner_latency_ms_mean":
            (sum(r.planner_latency_ms for r in recs) / n) if recs else 0.0,
    }
