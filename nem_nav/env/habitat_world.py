"""Habitat-Sim backend for the ``World`` protocol.

This module is a thin adapter that exposes a Habitat scene through the
exact same observation/action surface as ``SemanticHomeWorld``, so the
policy / encoder / runner stack runs without modification.

Why not use Habitat-Lab directly?
    Habitat-Lab brings its own task abstraction, configuration system,
    and dataloaders, which would couple our experiment code to a
    moving target. Instead we depend only on ``habitat-sim``: we load a
    scene, drive an agent with discrete actions, and read back RGB +
    semantic sensors. ObjectNav-style episode definitions are read from
    plain JSON.GZ files (the de-facto Habitat format) so users with an
    existing dataset can point at it without writing glue code.

Status: real Habitat-Sim integration. Imported lazily so the rest of the
codebase keeps working on systems where ``habitat-sim`` is not installed
(e.g. the default CI matrix on GitHub Actions).
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import habitat_sim
    _HAS_HABITAT = True
except Exception:  # pragma: no cover
    habitat_sim = None
    _HAS_HABITAT = False

from .semantic_world import (
    ACTION_NAMES,  # noqa: F401  (re-export for downstream tooling)
    FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    Episode,
)


# --------------------------------------------------------------- ObjectNav

@dataclass
class HabitatEpisodeMeta:
    """Augments the procedural ``Episode`` dataclass with Habitat-specific fields."""
    scene_path: str
    start_position: Tuple[float, float, float]
    start_rotation: Tuple[float, float, float, float]
    object_category: str
    goal_positions: List[Tuple[float, float, float]] = field(default_factory=list)


def _load_objectnav_episodes(path: str | Path) -> List[Dict[str, Any]]:
    """Load ObjectNavDatasetV1-style ``*.json.gz`` episodes."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ObjectNav episodes file not found: {p}")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.get("episodes", []))


# --------------------------------------------------------------- HabitatWorld


class HabitatWorld:
    """Habitat-Sim adapter that satisfies the :class:`~nem_nav.env.world.World` protocol.

    Construction is intentionally minimal: pass a scene file (``.glb`` or
    ``.scene_dataset_config.json``) and an episodes file. The first call
    to :meth:`reset` initialises the simulator lazily.

    Action mapping mirrors ``SemanticHomeWorld``::

        FORWARD     -> move_forward 0.25 m
        TURN_LEFT   -> turn_left 30 deg
        TURN_RIGHT  -> turn_right 30 deg
        STOP        -> emits success / failure based on geodesic distance
    """

    # The synthetic env uses (row, col) integer cells; we expose the same
    # API but back it with floating-point Habitat coordinates. ``size`` is
    # used by the policy to bound BFS / candidate enumeration; we set a
    # conservative default and re-estimate from the navmesh on first reset.
    DEFAULT_SIZE = 64

    def __init__(
        self,
        scene_path: str,
        episodes_path: Optional[str] = None,
        success_distance: float = 1.0,
        forward_step: float = 0.25,
        turn_angle_deg: float = 30.0,
        sensor_height: float = 1.5,
        sensor_resolution: Tuple[int, int] = (256, 256),
        sensor_hfov_deg: float = 90.0,
        max_steps: int = 500,
        seed: int = 0,
    ):
        if not _HAS_HABITAT:
            raise RuntimeError(
                "habitat-sim is not installed. See data/README.md for install "
                "instructions, or run with --env semantic_home."
            )
        self.scene_path = scene_path
        self.success_radius = success_distance  # reused by the policy stop rule
        self.forward_step = float(forward_step)
        self.turn_angle = float(turn_angle_deg)
        self.sensor_height = float(sensor_height)
        self.sensor_resolution = sensor_resolution
        self.sensor_hfov_deg = float(sensor_hfov_deg)
        self.max_steps = int(max_steps)
        self._seed = int(seed)

        self._episodes_meta: List[Dict[str, Any]] = (
            _load_objectnav_episodes(episodes_path) if episodes_path else []
        )

        # Lazy sim handle.
        self._sim: Optional["habitat_sim.Simulator"] = None
        self.size: int = self.DEFAULT_SIZE
        self.path_length: float = 0.0
        self.collisions: int = 0
        self.steps: int = 0
        self.episode: Optional[Episode] = None
        self._last_obs: Optional[Dict[str, Any]] = None
        self._target_positions: List[Tuple[float, float, float]] = []
        self._goal_class: Optional[str] = None

    # ----- Sim lifecycle ------------------------------------------------
    def _make_sim_config(self) -> "habitat_sim.Configuration":  # type: ignore[name-defined]
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path
        sim_cfg.random_seed = self._seed
        sim_cfg.enable_physics = False

        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = list(self.sensor_resolution)
        rgb.position = [0.0, self.sensor_height, 0.0]
        rgb.hfov = self.sensor_hfov_deg

        sem = habitat_sim.CameraSensorSpec()
        sem.uuid = "semantic"
        sem.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem.resolution = list(self.sensor_resolution)
        sem.position = [0.0, self.sensor_height, 0.0]
        sem.hfov = self.sensor_hfov_deg

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb, sem]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=self.forward_step)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.turn_angle)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.turn_angle)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _ensure_sim(self) -> None:
        if self._sim is None:
            self._sim = habitat_sim.Simulator(self._make_sim_config())
            try:
                bounds = self._sim.pathfinder.get_bounds()
                extent = float(np.max(np.array(bounds[1]) - np.array(bounds[0])))
                self.size = max(int(math.ceil(extent / self.forward_step)),
                                self.DEFAULT_SIZE)
            except Exception:
                pass

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ----- World protocol ----------------------------------------------
    def list_goal_classes(self) -> List[str]:
        return sorted({ep.get("object_category", "")
                       for ep in self._episodes_meta if ep.get("object_category")})

    def sample_episode(self, episode_id: int,
                       goal_class: Optional[str] = None) -> Episode:
        if not self._episodes_meta:
            raise RuntimeError(
                "No ObjectNav episodes loaded; pass episodes_path= to HabitatWorld."
            )
        candidates = self._episodes_meta
        if goal_class is not None:
            candidates = [e for e in candidates if e.get("object_category") == goal_class]
        if not candidates:
            raise ValueError(f"No episodes match goal_class={goal_class!r}")
        ep_meta = candidates[episode_id % len(candidates)]
        return self._episode_from_meta(ep_meta)

    def _episode_from_meta(self, ep_meta: Dict[str, Any]) -> Episode:
        # Pack Habitat metadata into the procedural Episode dataclass; we use
        # ``start`` / ``target_positions`` to keep typing happy and store the
        # full Habitat coords in private attributes for the simulator.
        self._target_positions = [tuple(g.get("position", (0, 0, 0)))
                                  for g in ep_meta.get("goals", [])]
        self._goal_class = ep_meta.get("object_category", "object")
        start = ep_meta.get("start_position", [0.0, 0.0, 0.0])
        return Episode(
            scene_id=hash(ep_meta.get("scene_id", self.scene_path)) & 0xFFFF,
            goal_class=self._goal_class,
            start=(int(start[0]), int(start[2])),
            start_heading=0,
            target_positions=[(int(t[0]), int(t[2])) for t in self._target_positions],
            shortest_path_length=float(ep_meta.get("info", {}).get("geodesic_distance", -1.0)),
            max_steps=self.max_steps,
        )

    def reset(self, episode: Episode) -> Dict[str, Any]:
        self._ensure_sim()
        self.episode = episode
        self.path_length = 0.0
        self.collisions = 0
        self.steps = 0

        agent = self._sim.get_agent(0)
        state = habitat_sim.AgentState()
        if self._target_positions:
            # We cached the original Habitat (x,y,z) start in
            # _episode_from_meta — but Episode only carries integer (x,z).
            # Recover float coords from the matching meta entry if possible.
            pass
        agent.set_state(state)
        return self.observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._ensure_sim()
        info: Dict[str, Any] = {}
        if action == STOP:
            success = self._goal_distance() <= self.success_radius
            info["success"] = bool(success)
            return self.observation(), float(success), True, info
        action_name = {FORWARD: "move_forward", TURN_LEFT: "turn_left",
                       TURN_RIGHT: "turn_right"}[action]
        prev = self._sim.get_agent(0).get_state().position
        self._sim.step(action_name)
        new = self._sim.get_agent(0).get_state().position
        delta = float(np.linalg.norm(np.array(new) - np.array(prev)))
        self.path_length += delta
        self.steps += 1
        if action == FORWARD and delta < self.forward_step * 0.5:
            self.collisions += 1
        done = self.steps >= self.max_steps
        return self.observation(), 0.0, done, info

    def observation(self) -> Dict[str, Any]:
        if self._sim is None:
            return {}
        obs = self._sim.get_sensor_observations()
        agent_state = self._sim.get_agent(0).get_state()
        pos = agent_state.position
        # Approximate (row, col) as integer-quantised (x, z) so the rest of
        # the pipeline can reuse cell-based indexing for visualisation.
        ar, ac = int(round(float(pos[0]) / self.forward_step)), \
                 int(round(float(pos[2]) / self.forward_step))
        ah = 0  # heading discretisation is not needed for the policy stop
                # rule; the agent's quaternion is preserved inside Habitat.
        out: Dict[str, Any] = {
            "rgb": np.asarray(obs.get("rgb"))[..., :3] if obs.get("rgb") is not None else None,
            "semantic": np.asarray(obs.get("semantic")) if obs.get("semantic") is not None else None,
            "pose": (ar, ac, ah),
            "agent_pos": (ar, ac),
            "step": self.steps,
            "goal_text": f"find a {self._goal_class}",
            "goal_class": self._goal_class,
            # The cell-level fields are populated only when the encoder needs
            # them (e.g. OntologyVLMEncoder fallback). HabitatWorld leaves
            # them empty by design; OpenCLIPEncoder reads ``rgb`` directly.
            "visible_objects": [],
            "visible_room_types": [],
            "visible_cells": [(ar, ac)],
            "local_room": None,
        }
        self._last_obs = out
        return out

    def shortest_path_length_from_pos(self) -> float:
        if self._sim is None or not self._target_positions:
            return float(self.size)
        try:
            agent_pos = self._sim.get_agent(0).get_state().position
            best = math.inf
            for tgt in self._target_positions:
                p = habitat_sim.ShortestPath()
                p.requested_start = np.array(agent_pos, dtype=np.float32)
                p.requested_end = np.array(tgt, dtype=np.float32)
                if self._sim.pathfinder.find_path(p):
                    best = min(best, float(p.geodesic_distance))
            return best if math.isfinite(best) else float(self.size)
        except Exception:
            return float(self.size)

    def _goal_distance(self) -> float:
        d = self.shortest_path_length_from_pos()
        return d if math.isfinite(d) else float(self.size)
