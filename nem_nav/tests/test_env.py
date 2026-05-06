"""Smoke and structural tests for SemanticHomeWorld."""
from __future__ import annotations

import math


from nem_nav.env.semantic_world import SemanticHomeWorld, FORWARD, TURN_LEFT


def test_scene_is_fully_connected():
    env = SemanticHomeWorld(size=20, n_rooms_target=5, seed=42)
    assert env._is_fully_connected()
    assert len(env.rooms) >= 2
    assert len(env.objects) >= len(env.rooms)


def test_episode_has_finite_shortest_path():
    env = SemanticHomeWorld(size=20, n_rooms_target=5, seed=7)
    ep = env.sample_episode(0)
    assert math.isfinite(ep.shortest_path_length)
    assert ep.shortest_path_length >= 1


def test_step_does_not_crash_through_walls():
    env = SemanticHomeWorld(size=18, n_rooms_target=4, seed=11)
    ep = env.sample_episode(0)
    obs = env.reset(ep)
    for _ in range(50):
        obs, _, done, _ = env.step(FORWARD)
        if done:
            break
    # Agent's position must remain inside the grid and on a free cell.
    r, c, _ = obs["pose"]
    assert 0 <= r < env.size and 0 <= c < env.size
    assert env.grid[r, c] == 0


def test_view_cone_is_directional():
    env = SemanticHomeWorld(size=18, n_rooms_target=4, view_radius=4, seed=3)
    ep = env.sample_episode(0)
    env.reset(ep)
    cells_a = set(env._visible_cells())
    env.step(TURN_LEFT)
    cells_b = set(env._visible_cells())
    # Turning should change the visible cone unless the agent is in a tiny room.
    assert cells_a != cells_b
