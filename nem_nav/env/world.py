"""World protocol and factory.

This module defines the contract every environment backend must satisfy so
the policy / encoder / runner stack can swap ``SemanticHomeWorld`` for a
real Habitat scene without changing any other code.

The protocol is **structural** (PEP 544): ``SemanticHomeWorld`` already
satisfies it without inheriting from anything, and a future
``HabitatWorld`` only has to expose the same names.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:  # avoid circular import at runtime
    from .semantic_world import Episode


@runtime_checkable
class World(Protocol):
    """Minimal interface used by ``NavigationPolicy`` and the runner."""

    # Geometry
    size: int
    path_length: float
    collisions: int

    def reset(self, episode: "Episode") -> Dict[str, Any]: ...
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: ...
    def observation(self) -> Dict[str, Any]: ...
    def list_goal_classes(self) -> List[str]: ...
    def shortest_path_length_from_pos(self) -> float: ...
    def sample_episode(self, episode_id: int,
                       goal_class: str | None = None) -> "Episode": ...


def build_world(name: str = "semantic_home", **kwargs) -> World:
    """Factory for World implementations.

    ``name``:
        - ``"semantic_home"`` — synthetic procedural testbed (default).
        - ``"habitat"``       — real Habitat-Sim wrapper. Requires the
          optional ``habitat`` extra and an HM3D/MP3D dataset on disk.
    """
    name = name.lower()
    if name == "semantic_home":
        from .semantic_world import SemanticHomeWorld
        return SemanticHomeWorld(**kwargs)
    if name == "habitat":
        from .habitat_world import HabitatWorld
        return HabitatWorld(**kwargs)
    raise ValueError(f"Unknown world backend: {name!r}")
