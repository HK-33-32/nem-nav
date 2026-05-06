"""SemanticHomeWorld: a procedurally generated semantic-navigation testbed.

The environment models a small apartment as an axis-aligned partition of a
square grid into rectangular rooms with semantic types and typical objects.
The agent has a discrete pose (cell + heading), receives a partial cone
observation, and must reach any cell adjacent to a target object class.

Design goals:

* Deterministic and seedable.
* Real semantics (objects co-occur with their typical rooms, with controlled
  noise) so that retrieval, graph abstraction and language-conditioned
  planning are *meaningfully* exercised rather than degenerated.
* Real shortest-path supervision via BFS for SPL.
* Cheap enough that the entire experiment matrix runs on CPU in minutes.

The interface deliberately mirrors a Habitat-style wrapper. To swap in
Habitat one only needs to re-implement ``reset``/``step``/``observation``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ontology import (
    OBJECT_CLASSES,
    ROOM_TYPES,
    objects_for_room,
)
from ..utils.seed import SeededRng


# Discrete actions
FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
STOP = 3
ACTIONS = [FORWARD, TURN_LEFT, TURN_RIGHT, STOP]
ACTION_NAMES = {FORWARD: "forward", TURN_LEFT: "turn_left",
                TURN_RIGHT: "turn_right", STOP: "stop"}

# Headings: 0=N, 1=E, 2=S, 3=W. Δ(row, col) for FORWARD.
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


@dataclass
class Room:
    room_id: int
    room_type: str
    r0: int
    c0: int
    r1: int  # exclusive
    c1: int  # exclusive

    @property
    def cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.r0, self.r1) for c in range(self.c0, self.c1)]

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.r0 + self.r1) // 2, (self.c0 + self.c1) // 2)


@dataclass
class WorldObject:
    obj_class: str
    pos: Tuple[int, int]
    room_id: int


@dataclass
class Episode:
    scene_id: int
    goal_class: str
    start: Tuple[int, int]
    start_heading: int
    target_positions: List[Tuple[int, int]]
    shortest_path_length: float
    max_steps: int


class SemanticHomeWorld:
    """A procedurally generated semantic apartment.

    Coordinates: (row, col) grid; a cell is navigable iff it is not a wall and
    not occupied by an object. Doorways (1 cell wide) connect adjacent rooms.

    A *scene* is generated once per episode-set seed; the agent then plays
    multiple goal-conditioned episodes inside that scene. This mirrors
    object-goal evaluation in indoor environments.
    """

    def __init__(
        self,
        size: int = 24,
        n_rooms_target: int = 6,
        view_radius: int = 4,
        view_cone_deg: float = 90.0,
        max_steps: int = 200,
        success_radius: int = 1,
        object_density: float = 0.7,
        noise_object_prob: float = 0.1,
        seed: int = 0,
    ):
        self.size = size
        self.n_rooms_target = n_rooms_target
        self.view_radius = view_radius
        self.view_cone_deg = view_cone_deg
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.object_density = object_density
        self.noise_object_prob = noise_object_prob
        self._seed = int(seed)
        self.rng = SeededRng(self._seed)

        # Filled by _generate_scene
        self.grid: np.ndarray  # 0=free, 1=wall
        self.rooms: List[Room] = []
        self.cell_room: np.ndarray  # int per cell (-1 if wall)
        self.objects: List[WorldObject] = []
        self.object_at: Dict[Tuple[int, int], WorldObject] = {}
        self.scene_id: int = 0
        self._generate_scene()

        # Episode state
        self.episode: Optional[Episode] = None
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_heading: int = 0
        self.steps: int = 0
        self.path_length: float = 0.0
        self.collisions: int = 0
        self.visited: np.ndarray = np.zeros((size, size), dtype=bool)
        self.observed: np.ndarray = np.zeros((size, size), dtype=bool)

    # ------------------------------------------------------------------
    # Scene generation
    # ------------------------------------------------------------------
    def _generate_scene(self) -> None:
        rng = self.rng.child("scene")
        N = self.size
        self.grid = np.ones((N, N), dtype=np.int8)  # walls everywhere
        self.cell_room = -np.ones((N, N), dtype=np.int32)
        self.rooms = []

        # Recursive partitioning. Each rectangle (r0, c0, r1, c1) is the
        # half-open *room* itself (inclusive r0/c0, exclusive r1/c1). Splits
        # consume one row/column for the shared wall, so adjacent rooms are
        # separated by exactly one wall cell — easy to carve doorways through.
        rects: List[Tuple[int, int, int, int]] = [(1, 1, N - 1, N - 1)]
        attempts = 0
        while len(rects) < self.n_rooms_target and attempts < 64:
            attempts += 1
            idx = max(range(len(rects)),
                      key=lambda i: (rects[i][2] - rects[i][0]) * (rects[i][3] - rects[i][1]))
            r0, c0, r1, c1 = rects[idx]
            h, w = r1 - r0, c1 - c0
            if h < 6 and w < 6:
                break  # cannot subdivide any further
            split_axis = "h" if h >= w else "w"
            if split_axis == "h" and h < 6:
                split_axis = "w"
            if split_axis == "w" and w < 6:
                split_axis = "h"
            if split_axis == "h":
                if h < 6:
                    break
                split = int(rng.integers(r0 + 3, r1 - 2))
                rects.pop(idx)
                rects.append((r0, c0, split, c1))         # upper room
                rects.append((split + 1, c0, r1, c1))     # lower room (split row is wall)
            else:
                if w < 6:
                    break
                split = int(rng.integers(c0 + 3, c1 - 2))
                rects.pop(idx)
                rects.append((r0, c0, r1, split))
                rects.append((r0, split + 1, r1, c1))

        room_types = list(ROOM_TYPES)
        rng.rng.shuffle(room_types)
        for k, (r0, c0, r1, c1) in enumerate(rects):
            if r1 <= r0 or c1 <= c0:
                continue
            rtype = room_types[k % len(room_types)]
            room = Room(room_id=k, room_type=rtype, r0=r0, c0=c0, r1=r1, c1=c1)
            self.rooms.append(room)
            for r in range(room.r0, room.r1):
                for c in range(room.c0, room.c1):
                    self.grid[r, c] = 0
                    self.cell_room[r, c] = room.room_id

        # Carve doorways between every pair of adjacent rooms (a wall cell
        # whose two opposite-axis neighbours belong to two different rooms).
        connected = self._connect_rooms(rng)
        if not connected:
            self._force_full_connectivity(rng)

        # Populate objects.
        self.objects = []
        self.object_at = {}
        for room in self.rooms:
            primary_objs = objects_for_room(room.room_type)
            n_objs = max(1, int(round(len(primary_objs) * self.object_density)))
            n_objs = min(n_objs, max(1, len(room.cells) // 4))
            free_cells = [(r, c) for (r, c) in room.cells if self._is_truly_free(r, c)]
            rng.rng.shuffle(free_cells)
            chosen = free_cells[:n_objs]
            for cell, obj_cls in zip(chosen, primary_objs[:n_objs]):
                self._place_object(obj_cls, cell, room.room_id)
            # Add a small amount of cross-room "noise" objects for realism.
            if rng.random() < self.noise_object_prob and len(free_cells) > n_objs:
                noise_cls = rng.choice(OBJECT_CLASSES)
                self._place_object(str(noise_cls), free_cells[n_objs], room.room_id)

        self.scene_id = self._seed

    def _is_truly_free(self, r: int, c: int) -> bool:
        return self.grid[r, c] == 0 and (r, c) not in self.object_at

    def _place_object(self, obj_cls: str, cell: Tuple[int, int], room_id: int) -> None:
        obj = WorldObject(obj_class=obj_cls, pos=cell, room_id=room_id)
        self.objects.append(obj)
        self.object_at[cell] = obj

    def _connect_rooms(self, rng: SeededRng) -> bool:
        # Build adjacency by scanning for shared wall segments.
        room_walls: Dict[Tuple[int, int], List[int]] = {}
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                if self.grid[r, c] == 1:
                    neigh_rooms = set()
                    for dr, dc in _DIRS:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.size and 0 <= cc < self.size and self.cell_room[rr, cc] >= 0:
                            neigh_rooms.add(int(self.cell_room[rr, cc]))
                    if len(neigh_rooms) >= 2:
                        a, b = sorted(list(neigh_rooms))[:2]
                        room_walls.setdefault((a, b), []).append(r * self.size + c)

        # For each adjacent pair, knock down one wall cell.
        for (a, b), cells in room_walls.items():
            chosen = int(rng.choice(np.array(cells)))
            r, c = chosen // self.size, chosen % self.size
            self.grid[r, c] = 0
            self.cell_room[r, c] = a  # arbitrarily assign to a

        # Check connectivity via BFS from any free cell.
        return self._is_fully_connected()

    def _force_full_connectivity(self, rng: SeededRng) -> None:
        # Very simple fallback: scan walls, for each wall cell with two
        # different room neighbors, knock down. Repeat until connected.
        for _ in range(8):
            if self._is_fully_connected():
                return
            for r in range(1, self.size - 1):
                for c in range(1, self.size - 1):
                    if self.grid[r, c] == 1:
                        neigh_rooms = set()
                        for dr, dc in _DIRS:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < self.size and 0 <= cc < self.size and self.cell_room[rr, cc] >= 0:
                                neigh_rooms.add(int(self.cell_room[rr, cc]))
                        if len(neigh_rooms) >= 2:
                            self.grid[r, c] = 0
                            self.cell_room[r, c] = list(neigh_rooms)[0]

    def _is_fully_connected(self) -> bool:
        free = list(zip(*np.where(self.grid == 0)))
        if not free:
            return False
        seen = {free[0]}
        q = deque([free[0]])
        while q:
            r, c = q.popleft()
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size and self.grid[rr, cc] == 0 and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    q.append((rr, cc))
        return len(seen) == len(free)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def list_goal_classes(self) -> List[str]:
        return sorted({o.obj_class for o in self.objects})

    def sample_episode(self, episode_id: int, goal_class: Optional[str] = None) -> Episode:
        rng = self.rng.child(f"episode_{episode_id}")
        # Choose a goal class that exists in the scene.
        present = self.list_goal_classes()
        if goal_class is None:
            goal_class = str(rng.choice(present))
        elif goal_class not in present:
            raise ValueError(f"Goal class {goal_class!r} not present in scene")

        target_positions = [o.pos for o in self.objects if o.obj_class == goal_class]

        # Sample a start cell that is free and at least 5 cells away from any target.
        free_cells = [(r, c) for (r, c) in zip(*np.where(self.grid == 0)) if (r, c) not in self.object_at]
        far_enough: List[Tuple[int, int]] = []
        for cell in free_cells:
            d = min(self._bfs_dist(cell, t) for t in target_positions)
            if d == math.inf:
                continue
            if d >= 5:
                far_enough.append(cell)
        if not far_enough:
            far_enough = free_cells
        start_idx = int(rng.integers(0, len(far_enough)))
        start = far_enough[start_idx]
        start_heading = int(rng.integers(0, 4))
        spl = min(self._bfs_dist(start, t) for t in target_positions)
        ep = Episode(
            scene_id=self.scene_id,
            goal_class=goal_class,
            start=start,
            start_heading=start_heading,
            target_positions=target_positions,
            shortest_path_length=float(spl),
            max_steps=self.max_steps,
        )
        return ep

    def reset(self, episode: Episode) -> Dict:
        self.episode = episode
        self.agent_pos = episode.start
        self.agent_heading = episode.start_heading
        self.steps = 0
        self.path_length = 0.0
        self.collisions = 0
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        self.observed = np.zeros((self.size, self.size), dtype=bool)
        self.visited[self.agent_pos] = True
        self._update_observed()
        return self.observation()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert self.episode is not None
        info: Dict = {}
        reward = 0.0
        done = False
        if action == FORWARD:
            dr, dc = _DIRS[self.agent_heading]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if self._is_navigable(nr, nc):
                self.agent_pos = (nr, nc)
                self.path_length += 1.0
                self.visited[nr, nc] = True
            else:
                self.collisions += 1
                reward -= 0.1
        elif action == TURN_LEFT:
            self.agent_heading = (self.agent_heading - 1) % 4
        elif action == TURN_RIGHT:
            self.agent_heading = (self.agent_heading + 1) % 4
        elif action == STOP:
            done = True
        else:
            raise ValueError(action)

        self.steps += 1
        self._update_observed()
        success = self._at_goal()
        if action == STOP:
            info["success"] = bool(success)
            reward += 1.0 if success else -0.5
        if self.steps >= self.episode.max_steps:
            done = True
        if done and "success" not in info:
            info["success"] = bool(self._at_goal())
        return self.observation(), reward, done, info

    def _at_goal(self) -> bool:
        if self.episode is None:
            return False
        ar, ac = self.agent_pos
        for tr, tc in self.episode.target_positions:
            if abs(ar - tr) + abs(ac - tc) <= self.success_radius:
                return True
        return False

    def _is_navigable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.grid[r, c] == 1:
            return False
        if (r, c) in self.object_at:
            # Objects block movement; the agent reaches the goal when
            # adjacent (success_radius >= 1).
            return False
        return True

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _update_observed(self) -> None:
        for cell in self._visible_cells():
            self.observed[cell] = True

    def _visible_cells(self) -> List[Tuple[int, int]]:
        ar, ac = self.agent_pos
        h = self.agent_heading
        cells = []
        # Forward cone of `view_radius` along heading direction with given
        # cone half-angle. Use simple ray-casting along discrete cells.
        cone = math.radians(self.view_cone_deg) / 2.0
        # Heading angle in radians, where 0=N -> -pi/2, 1=E -> 0, etc.
        heading_angle = {0: -math.pi / 2, 1: 0.0, 2: math.pi / 2, 3: math.pi}[h]
        for dr in range(-self.view_radius, self.view_radius + 1):
            for dc in range(-self.view_radius, self.view_radius + 1):
                if dr == 0 and dc == 0:
                    cells.append((ar, ac))
                    continue
                if dr * dr + dc * dc > self.view_radius * self.view_radius:
                    continue
                # angle of (dc, dr) — note row grows southward
                ang = math.atan2(dr, dc)
                diff = (ang - heading_angle + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > cone:
                    continue
                r, c = ar + dr, ac + dc
                if not (0 <= r < self.size and 0 <= c < self.size):
                    continue
                # ray-cast: check no wall in between
                if self._line_of_sight(ar, ac, r, c):
                    cells.append((r, c))
        return cells

    def _line_of_sight(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        # Bresenham's line algorithm; blocked by any wall.
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if (r, c) != (r0, c0) and (r, c) != (r1, c1):
                if self.grid[r, c] == 1:
                    return False
            if (r, c) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def observation(self) -> Dict:
        ar, ac = self.agent_pos
        visible_cells = self._visible_cells()
        visible_objects: List[Tuple[str, int, int]] = []
        visible_room_types: List[str] = []
        for r, c in visible_cells:
            obj = self.object_at.get((r, c))
            if obj is not None:
                visible_objects.append((obj.obj_class, r - ar, c - ac))
            rid = self.cell_room[r, c]
            if rid >= 0:
                visible_room_types.append(self.rooms[int(rid)].room_type)
        local_room = None
        rid = self.cell_room[ar, ac]
        if rid >= 0:
            local_room = self.rooms[int(rid)].room_type
        return {
            "pose": (ar, ac, self.agent_heading),
            "visible_objects": visible_objects,
            "visible_cells": visible_cells,
            "visible_room_types": visible_room_types,
            "local_room": local_room,
            "goal_text": f"find a {self.episode.goal_class}" if self.episode else "",
            "goal_class": self.episode.goal_class if self.episode else None,
            "step": self.steps,
        }

    # ------------------------------------------------------------------
    # Geometry helpers used by baselines and metrics
    # ------------------------------------------------------------------
    def _bfs_dist(self, start: Tuple[int, int], target: Tuple[int, int]) -> float:
        """Length of the shortest path from start to a cell adjacent to target."""
        if start == target:
            return 0.0
        seen = {start}
        q = deque([(start, 0)])
        while q:
            (r, c), d = q.popleft()
            if abs(r - target[0]) + abs(c - target[1]) <= self.success_radius and (r, c) != target:
                return float(d)
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    if (rr, cc) in seen:
                        continue
                    if self._is_navigable(rr, cc):
                        seen.add((rr, cc))
                        q.append(((rr, cc), d + 1))
        return math.inf

    def shortest_path_length_from_pos(self) -> float:
        if self.episode is None:
            return math.inf
        return min(self._bfs_dist(self.agent_pos, t) for t in self.episode.target_positions)

    def navigable_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        out = []
        for dr, dc in _DIRS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if self._is_navigable(nr, nc):
                out.append((nr, nc))
        return out
