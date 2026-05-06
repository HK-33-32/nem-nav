"""Render a procedural scene with rooms, objects, and a sample
trajectory for the qualitative figure in the paper.

Outputs ``paper/figures/scene_trajectory.pdf`` (or .png).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from nem_nav.env.semantic_world import (
    SemanticHomeWorld,
)
from nem_nav.models.perception.encoder import build_encoder
from nem_nav.models.policy.policy import NavigationPolicy, PolicyConfig


_ROOM_COLOURS = {
    "kitchen": "#fde2a7",
    "bedroom": "#c2e0f4",
    "bathroom": "#d8f0e0",
    "living_room": "#f6c9c9",
    "office": "#e0d4f0",
    "dining_room": "#f7e6d0",
    "hallway": "#e8e8e8",
}


def render(env: SemanticHomeWorld, traj: List[Tuple[int, int]],
           goal_class: str, target_positions: List[Tuple[int, int]],
           out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
    # Walls
    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r, c] == 1:
                ax.add_patch(patches.Rectangle((c, env.size - 1 - r), 1, 1,
                                               facecolor="#3a3a3a", edgecolor="none"))
    # Rooms
    for room in env.rooms:
        col = _ROOM_COLOURS.get(room.room_type, "#dddddd")
        for r in range(room.r0, room.r1):
            for cc in range(room.c0, room.c1):
                if env.grid[r, cc] == 0:
                    ax.add_patch(patches.Rectangle((cc, env.size - 1 - r), 1, 1,
                                                   facecolor=col, edgecolor="none"))
        rc, cc = room.center
        ax.text(cc + 0.5, env.size - 1 - rc + 0.5, room.room_type,
                ha="center", va="center", fontsize=6, color="#333333", alpha=0.6)
    # Objects
    for obj in env.objects:
        r, c = obj.pos
        marker_face = "#d65a31" if obj.obj_class == goal_class else "#777777"
        ax.plot(c + 0.5, env.size - 1 - r + 0.5, "o",
                markerfacecolor=marker_face, markeredgecolor="white",
                markersize=4 if obj.obj_class != goal_class else 7)
    # Trajectory
    if traj:
        xs = [c + 0.5 for r, c in traj]
        ys = [env.size - 1 - r + 0.5 for r, c in traj]
        ax.plot(xs, ys, "-", color="#1f4e79", linewidth=1.6)
        ax.plot(xs[0], ys[0], "s", color="#1f4e79", markersize=5)
        ax.plot(xs[-1], ys[-1], "*", color="#d65a31", markersize=10)
    # Goal targets
    for tr, tc in target_positions:
        ax.add_patch(patches.Circle((tc + 0.5, env.size - 1 - tr + 0.5),
                                    0.45, fill=False, edgecolor="#d65a31",
                                    linewidth=1.4))
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Goal: find a {goal_class}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode", type=int, default=2,
                   help="episode index within the scene "
                        "(memory has accumulated by then)")
    p.add_argument("--mode", default="nem_nav")
    p.add_argument("--out", default="paper/figures/scene_trajectory.pdf")
    p.add_argument("--scene_size", type=int, default=24)
    p.add_argument("--n_rooms", type=int, default=6)
    p.add_argument("--max_steps", type=int, default=200)
    args = p.parse_args()

    env = SemanticHomeWorld(size=args.scene_size, n_rooms_target=args.n_rooms,
                            max_steps=args.max_steps, seed=args.seed)
    encoder = build_encoder("ontology")
    cfg = PolicyConfig.for_mode(args.mode, seed=args.seed)
    policy = NavigationPolicy(env, encoder, cfg)
    policy.reset_lifelong()
    # Warm up the memory by running prior episodes.
    for e in range(args.episode):
        ep = env.sample_episode(e)
        obs = env.reset(ep)
        policy.reset_episode()
        for _ in range(args.max_steps):
            a, _ = policy.act(obs)
            obs, _, done, _ = env.step(a)
            if done:
                break
    ep = env.sample_episode(args.episode)
    obs = env.reset(ep)
    policy.reset_episode()
    traj = [env.agent_pos]
    for _ in range(args.max_steps):
        a, _ = policy.act(obs)
        obs, _, done, _ = env.step(a)
        traj.append(env.agent_pos)
        if done:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(env, traj, ep.goal_class, ep.target_positions, out)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
