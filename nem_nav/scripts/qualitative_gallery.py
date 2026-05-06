"""Generate a gallery of qualitative figures for the paper.

For each requested (scene_seed, episode) pair we render a multi-panel
figure that shows what the system actually did:

  Panel 1 (top-left)    Scene + agent trajectory + goal markers.
  Panel 2 (top-right)   Top-K retrieved memories projected onto the scene
                        (cells, colour-coded by retrieval rank).
  Panel 3 (bottom-left) Per-step planner output (target_room + confidence).
  Panel 4 (bottom-right) Per-step interpretable score breakdown
                        (frontier / semantic / memory / graph / planner).

Output: ``paper/figures/gallery_<seed>_ep<ep>.pdf`` and a single
``gallery_overview.pdf`` with a 2x4 grid of small thumbnails for the main
paper. Per-figure metadata (success/SPL/path_length) is written next to
each PDF as a sibling JSON for the supplementary appendix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nem_nav.env.semantic_world import SemanticHomeWorld  # noqa: E402
from nem_nav.models.perception.encoder import build_encoder  # noqa: E402
from nem_nav.models.policy.policy import NavigationPolicy, PolicyConfig  # noqa: E402


_ROOM_COLOURS = {
    "kitchen": "#fde2a7", "bedroom": "#c2e0f4", "bathroom": "#d8f0e0",
    "living_room": "#f6c9c9", "office": "#e0d4f0",
    "dining_room": "#f7e6d0", "hallway": "#e8e8e8",
}


def _render_scene(ax, env: SemanticHomeWorld, traj, goal_class: str,
                  target_positions, retrieved_cells=None,
                  show_room_labels: bool = True) -> None:
    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r, c] == 1:
                ax.add_patch(patches.Rectangle((c, env.size - 1 - r), 1, 1,
                                               facecolor="#3a3a3a", edgecolor="none"))
    for room in env.rooms:
        col = _ROOM_COLOURS.get(room.room_type, "#dddddd")
        for r in range(room.r0, room.r1):
            for cc in range(room.c0, room.c1):
                if env.grid[r, cc] == 0:
                    ax.add_patch(patches.Rectangle((cc, env.size - 1 - r), 1, 1,
                                                   facecolor=col, edgecolor="none"))
        if show_room_labels:
            rc, cc = room.center
            ax.text(cc + 0.5, env.size - 1 - rc + 0.5, room.room_type,
                    ha="center", va="center", fontsize=5,
                    color="#333333", alpha=0.55)
    for obj in env.objects:
        r, c = obj.pos
        face = "#d65a31" if obj.obj_class == goal_class else "#777777"
        ax.plot(c + 0.5, env.size - 1 - r + 0.5, "o",
                markerfacecolor=face, markeredgecolor="white",
                markersize=4 if obj.obj_class != goal_class else 7)
    if retrieved_cells:
        for rank, (rr, cc) in enumerate(retrieved_cells):
            alpha = max(0.2, 1.0 - 0.15 * rank)
            ax.add_patch(patches.Rectangle((cc, env.size - 1 - rr), 1, 1,
                                           facecolor="#ffd166",
                                           edgecolor="#b07d1c",
                                           linewidth=0.5,
                                           alpha=alpha))
    if traj:
        xs = [c + 0.5 for r, c in traj]
        ys = [env.size - 1 - r + 0.5 for r, c in traj]
        ax.plot(xs, ys, "-", color="#1f4e79", linewidth=1.4)
        ax.plot(xs[0], ys[0], "s", color="#1f4e79", markersize=5)
        ax.plot(xs[-1], ys[-1], "*", color="#d65a31", markersize=10)
    for tr, tc in target_positions:
        ax.add_patch(patches.Circle((tc + 0.5, env.size - 1 - tr + 0.5),
                                    0.45, fill=False, edgecolor="#d65a31",
                                    linewidth=1.4))
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _run_episode_with_traces(
    env: SemanticHomeWorld, policy: NavigationPolicy, episode_id: int,
    max_steps: int,
) -> Dict[str, Any]:
    ep = env.sample_episode(episode_id)
    obs = env.reset(ep)
    policy.reset_episode()
    traj: List[Tuple[int, int]] = [env.agent_pos]
    planner_trace: List[Dict[str, Any]] = []
    score_trace: List[Dict[str, float]] = []
    for _ in range(max_steps):
        action, info = policy.act(obs)
        if "planner" in info:
            planner_trace.append({"step": env.steps, **info["planner"]})
        if "scores" in info:
            score_trace.append({"step": env.steps, **info["scores"]})
        obs, _, done, sinfo = env.step(action)
        traj.append(env.agent_pos)
        if done:
            break
    success = bool(sinfo.get("success", False)) if 'sinfo' in dir() else False
    # Top-K memory cells (latest retrieval).
    retrieved_cells: List[Tuple[int, int]] = []
    if policy.cfg.use_memory and len(policy.memory) > 0:
        goal_emb = policy.encoder.encode_text(ep.goal_class)
        retrieved = policy.memory.retrieve(goal_emb, k=8)
        for mid, _, meta in retrieved:
            pose = meta.get("pose")
            if pose is not None:
                retrieved_cells.append(tuple(pose))
    return {
        "episode": ep,
        "trajectory": traj,
        "success": success,
        "steps": env.steps,
        "path_length": env.path_length,
        "shortest_path_length": ep.shortest_path_length,
        "planner_trace": planner_trace,
        "score_trace": score_trace,
        "retrieved_cells": retrieved_cells,
    }


def _render_panel(out_path: Path, env: SemanticHomeWorld, trace: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0))
    ep = trace["episode"]

    _render_scene(axes[0, 0], env, trace["trajectory"], ep.goal_class,
                  ep.target_positions)
    axes[0, 0].set_title(
        f"Goal: {ep.goal_class}   "
        f"{'SUCCESS' if trace['success'] else 'FAIL'} · "
        f"steps={trace['steps']} · "
        f"SPL={(ep.shortest_path_length / max(trace['path_length'], ep.shortest_path_length, 1e-6)) if trace['success'] else 0:.2f}",
        fontsize=9,
    )

    _render_scene(axes[0, 1], env, trace["trajectory"], ep.goal_class,
                  ep.target_positions, retrieved_cells=trace["retrieved_cells"],
                  show_room_labels=False)
    axes[0, 1].set_title(
        f"Top-{len(trace['retrieved_cells'])} retrieved memory cells "
        "(orange = high rank)", fontsize=9
    )

    # Panel 3: planner timeline.
    if trace["planner_trace"]:
        steps = [p["step"] for p in trace["planner_trace"]]
        confs = [float(p.get("confidence", 0)) for p in trace["planner_trace"]]
        rooms = [str(p.get("target_room")) for p in trace["planner_trace"]]
        axes[1, 0].plot(steps, confs, "-o", color="#1f4e79", markersize=4)
        for s, c, room in zip(steps, confs, rooms):
            axes[1, 0].annotate(room, (s, c), fontsize=6,
                                xytext=(0, 5), textcoords="offset points",
                                ha="center")
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].set_xlabel("step")
        axes[1, 0].set_ylabel("planner confidence")
        axes[1, 0].set_title("Planner output over time", fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "No planner calls (mode disables planner).",
                        ha="center", va="center", fontsize=9)
        axes[1, 0].axis("off")

    # Panel 4: score breakdown (averaged in a sliding window if available).
    if trace["score_trace"]:
        steps = [s["step"] for s in trace["score_trace"]]
        keys = [k for k in trace["score_trace"][0].keys() if k != "step"]
        for k in keys:
            ys = [float(s.get(k, 0)) for s in trace["score_trace"]]
            axes[1, 1].plot(steps, ys, "-", linewidth=1.0, label=k)
        axes[1, 1].set_xlabel("step")
        axes[1, 1].set_ylabel("max score")
        axes[1, 1].set_title("Score components over time", fontsize=9)
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5, 0.5,
            "Score components not exposed by the policy in this run.\n"
            "Re-run with policy.act(..., return_scores=True) to populate.",
            ha="center", va="center", fontsize=8, color="#666666",
        )
        axes[1, 1].axis("off")

    fig.suptitle(f"Episode @ seed={env._seed} (idx)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_overview(out_path: Path, panels: List[Tuple[Path, Dict[str, Any]]]) -> None:
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_2d(axes)
    for i, (env, trace) in enumerate(panels):
        ax = axes[i // cols, i % cols]
        _render_scene(ax, env, trace["trajectory"], trace["episode"].goal_class,
                      trace["episode"].target_positions, show_room_labels=False)
        ax.set_title(
            f"{trace['episode'].goal_class} · "
            f"{'OK' if trace['success'] else 'X'}",
            fontsize=8,
        )
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes_per_scene", type=int, default=8,
                   help="Number of episodes to warm memory before the rendered episode.")
    p.add_argument("--render_episode", type=int, default=5,
                   help="Index of the episode to render (0-based).")
    p.add_argument("--mode", default="nem_nav")
    p.add_argument("--out_dir", default="paper/figures/gallery")
    p.add_argument("--scene_size", type=int, default=24)
    p.add_argument("--n_rooms", type=int, default=6)
    p.add_argument("--max_steps", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder("ontology")
    overview: List[Tuple[SemanticHomeWorld, Dict[str, Any]]] = []
    summary: List[Dict[str, Any]] = []

    for seed in args.seeds:
        env = SemanticHomeWorld(size=args.scene_size, n_rooms_target=args.n_rooms,
                                max_steps=args.max_steps, seed=seed)
        cfg = PolicyConfig.for_mode(args.mode, seed=seed)
        policy = NavigationPolicy(env, encoder, cfg)
        policy.reset_lifelong()
        # Warm up memory.
        for e in range(args.render_episode):
            _ = _run_episode_with_traces(env, policy, e, args.max_steps)
        # Render this episode.
        trace = _run_episode_with_traces(env, policy, args.render_episode,
                                         args.max_steps)
        panel_path = out_dir / f"seed{seed}_ep{args.render_episode}.pdf"
        _render_panel(panel_path, env, trace)
        meta = {
            "seed": seed, "episode": args.render_episode,
            "goal_class": trace["episode"].goal_class,
            "success": trace["success"], "steps": trace["steps"],
            "path_length": trace["path_length"],
            "shortest_path_length": trace["episode"].shortest_path_length,
        }
        meta_path = panel_path.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        summary.append({"path": str(panel_path), **meta})
        overview.append((env, trace))
        print(f"[done] {panel_path}  goal={meta['goal_class']}  "
              f"success={meta['success']}  steps={meta['steps']}")

    _render_overview(out_dir.parent / "gallery_overview.pdf", overview)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] wrote {len(overview)} panels + overview")


if __name__ == "__main__":
    main()
