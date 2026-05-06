"""Smoke tests for the Habitat backend.

These tests are skipped automatically when ``habitat-sim`` is not
installed or when no scene/episodes path is configured via the
``NEM_NAV_HABITAT_SCENE`` and ``NEM_NAV_HABITAT_EPISODES`` environment
variables. They are intended to be run by hand on a workstation that has
the dataset set up; the default GitHub Actions matrix skips them.
"""
from __future__ import annotations

import os

import pytest

try:
    import habitat_sim  # noqa: F401
    _HAS_HABITAT = True
except Exception:
    _HAS_HABITAT = False

from nem_nav.env.world import World

pytestmark = pytest.mark.habitat


@pytest.mark.skipif(not _HAS_HABITAT, reason="habitat-sim not installed")
@pytest.mark.skipif(not os.environ.get("NEM_NAV_HABITAT_SCENE"),
                    reason="NEM_NAV_HABITAT_SCENE not set")
def test_habitat_world_smoke_episode():
    from nem_nav.env.habitat_world import HabitatWorld
    env = HabitatWorld(
        scene_path=os.environ["NEM_NAV_HABITAT_SCENE"],
        episodes_path=os.environ.get("NEM_NAV_HABITAT_EPISODES"),
        max_steps=10,
    )
    assert isinstance(env, World)
    try:
        ep = env.sample_episode(0)
        obs = env.reset(ep)
        assert obs.get("rgb") is not None or obs.get("pose") is not None
        for _ in range(3):
            obs, reward, done, info = env.step(0)
            if done:
                break
    finally:
        env.close()
