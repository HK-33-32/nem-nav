"""Contract tests for the ``World`` protocol.

Every backend must satisfy the same structural contract — these tests run
without any optional dependency so they catch interface drift in the
default ``SemanticHomeWorld`` even when habitat-sim is not installed.
"""
from __future__ import annotations

from nem_nav.env.semantic_world import SemanticHomeWorld
from nem_nav.env.world import World, build_world


def test_semantic_home_satisfies_world_protocol():
    env = SemanticHomeWorld(size=14, n_rooms_target=3, max_steps=20, seed=0)
    assert isinstance(env, World)


def test_build_world_default_returns_semantic_home():
    env = build_world("semantic_home", size=14, n_rooms_target=3,
                      max_steps=20, seed=1)
    assert isinstance(env, World)
    assert env.size >= 14


def test_build_world_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        build_world("nope")


def test_world_episode_runs_one_step():
    env = build_world("semantic_home", size=14, n_rooms_target=3,
                      max_steps=20, seed=2)
    ep = env.sample_episode(0)
    obs = env.reset(ep)
    assert "pose" in obs and "goal_text" in obs
    obs2, _, done, info = env.step(0)
    assert "pose" in obs2
    assert isinstance(done, bool)
    assert isinstance(info, dict)
