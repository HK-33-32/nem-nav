"""Unit tests for metrics."""
from __future__ import annotations

from nem_nav.evaluation.metrics import (
    EpisodeRecord, compute_spl, compute_success, aggregate,
)


def _r(success, spl, p_len, dtg=0.0, steps=10):
    return EpisodeRecord(success=success, shortest_path_length=spl,
                         path_length=p_len, distance_to_goal=dtg, steps=steps,
                         collisions=0)


def test_success_zero_when_no_records():
    assert compute_success([]) == 0.0


def test_spl_matches_anderson_definition():
    # SPL = (1/N) Σ S_i * l_i / max(p_i, l_i)
    recs = [_r(1, 10, 10), _r(1, 10, 20), _r(0, 10, 10)]
    expected = (1 * 10 / 10 + 1 * 10 / 20 + 0) / 3
    assert abs(compute_spl(recs) - expected) < 1e-9


def test_aggregate_returns_all_keys():
    recs = [_r(1, 5, 7, dtg=0)]
    out = aggregate(recs)
    for k in ("n_episodes", "success", "spl", "dtg", "ep_len", "collisions",
              "retrieval_latency_ms_mean", "planner_latency_ms_mean"):
        assert k in out
    assert out["n_episodes"] == 1
    assert out["success"] == 1.0


def test_aggregate_includes_latency_means():
    rec = EpisodeRecord(
        success=1, shortest_path_length=5.0, path_length=5.0,
        distance_to_goal=0.0, steps=5, collisions=0,
        retrieval_latency_ms=2.5, planner_latency_ms=10.0,
    )
    out = aggregate([rec, rec])
    assert abs(out["retrieval_latency_ms_mean"] - 2.5) < 1e-9
    assert abs(out["planner_latency_ms_mean"] - 10.0) < 1e-9


def test_unsuccessful_episode_does_not_contribute_to_spl():
    recs = [_r(0, 10, 10)]
    assert compute_spl(recs) == 0.0
    assert compute_success(recs) == 0.0
