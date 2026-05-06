"""Failure-mode taxonomy from per-episode JSONL logs.

A* reviewers expect a clear, quantified failure analysis. This script
walks an experiment-matrix run directory, classifies every failed
episode into a small taxonomy, and writes:

  paper/tables/failure_taxonomy.tex
  paper/figures/failure_taxonomy.pdf
  outputs/matrix/<run_id>/failures.json

Failure classes (mutually exclusive, evaluated in order):

  TIMEOUT          Episode hit max_steps without STOP.
  STOP_TOO_FAR     Agent emitted STOP but the goal was outside
                   the success radius.
  STUCK_LOOP       Agent wandered (path_length >> shortest_path)
                   and never found the goal.
  WRONG_ROOM       Agent finished close to a wrong-class object
                   that ontologically lives in a different room
                   from the goal class (planner / retrieval mismatch).
  COLD_START       Failed on the very first episode of a scene
                   (no memory available — expected blind-spot).
  OTHER            Anything not above.

The taxonomy is intentionally coarse: too-fine a split makes the
table noisy at n=5 seeds.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nem_nav.env.ontology import primary_room  # noqa: E402


_FAILURE_ORDER = ["TIMEOUT", "STOP_TOO_FAR", "STUCK_LOOP", "WRONG_ROOM",
                  "COLD_START", "OTHER"]
_FAILURE_DESC = {
    "TIMEOUT":      "Hit max\\_steps without STOP",
    "STOP_TOO_FAR": "Stopped, goal not within radius",
    "STUCK_LOOP":   "Path length $>3\\times$ shortest, never reached goal",
    "WRONG_ROOM":   "Ended near wrong-room object (planner/retrieval mismatch)",
    "COLD_START":   "First episode in scene (no memory yet)",
    "OTHER":        "Other / unclassified",
}


def _classify(rec: Dict[str, Any], max_steps: int) -> str:
    metrics = rec.get("metrics", {})
    if int(metrics.get("success", 0)) == 1:
        return ""
    steps = int(metrics.get("steps", 0))
    if rec.get("first_encounter") and int(rec.get("episode", 0)) == 0:
        return "COLD_START"
    pl = float(metrics.get("path_length", 0))
    spl_target = float(metrics.get("shortest_path_length", 0))
    if steps >= max_steps:
        return "TIMEOUT"
    if spl_target > 0 and pl > 3.0 * spl_target:
        return "STUCK_LOOP"
    if steps < max_steps:
        # The agent stopped voluntarily but episode failed → STOP_TOO_FAR or
        # WRONG_ROOM (we can't fully tell without action trace; default to
        # STOP_TOO_FAR unless retrieval clearly pointed at a sibling object).
        plan_target = (rec.get("trace", []) or [{}])[-1].get("planner", {}).get("target_room")
        goal_room = primary_room(rec.get("goal_class", "")) if rec.get("goal_class") else None
        if plan_target and goal_room and plan_target != goal_room:
            return "WRONG_ROOM"
        return "STOP_TOO_FAR"
    return "OTHER"


def collect_failures(run_root: Path, mode: str = "nem_nav",
                     max_steps: int = 200) -> Dict[str, Any]:
    counts: Counter = Counter()
    per_class: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_total = 0
    n_failed = 0
    for seed_dir in sorted((run_root / "runs").glob(f"{mode}_seed*")):
        ep_path = seed_dir / "episodes.jsonl"
        if not ep_path.exists():
            continue
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n_total += 1
            cls = _classify(rec, max_steps)
            if not cls:
                continue
            n_failed += 1
            counts[cls] += 1
            per_class[cls][rec.get("goal_class", "?")] += 1
            if len(examples[cls]) < 3:
                examples[cls].append({
                    "scene": rec.get("scene"), "episode": rec.get("episode"),
                    "seed": rec.get("seed"), "goal_class": rec.get("goal_class"),
                    "metrics": rec.get("metrics", {}),
                })
    return {"total": n_total, "failed": n_failed, "counts": dict(counts),
            "per_class_goals": {k: dict(v) for k, v in per_class.items()},
            "examples": dict(examples)}


def write_table(stats: Dict[str, Any], out_path: Path) -> None:
    rows: List[str] = []
    rows.append(r"\begin{tabular}{l@{\hspace{8pt}}rrl}")
    rows.append(r"\toprule")
    rows.append(r"\textbf{Failure mode} & \textbf{\#} & \textbf{\% of fails} & \textbf{Definition} \\")
    rows.append(r"\midrule")
    n_failed = max(stats["failed"], 1)
    for cls in _FAILURE_ORDER:
        n = stats["counts"].get(cls, 0)
        pct = 100.0 * n / n_failed
        cls_tex = cls.replace("_", r"\_")
        rows.append(
            f"{cls_tex} & {n} & {pct:.1f}\\,\\% "
            f"& {_FAILURE_DESC[cls]} \\\\"
        )
    rows.append(r"\midrule")
    rows.append(r"\textbf{Total failures} & " + str(stats["failed"])
                + r" & 100.0\,\% & --- \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"% Total episodes considered: " + str(stats["total"]))
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_taxonomy(stats: Dict[str, Any], out_path: Path) -> None:
    counts = [stats["counts"].get(cls, 0) for cls in _FAILURE_ORDER]
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.0))
    ax.bar(range(len(_FAILURE_ORDER)), counts, color="#4a6fa5")
    ax.set_xticks(range(len(_FAILURE_ORDER)))
    ax.set_xticklabels(_FAILURE_ORDER, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("# episodes")
    ax.set_title(f"NEM-Nav failure taxonomy (n_failed={stats['failed']} of {stats['total']})",
                 fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="main")
    p.add_argument("--mode", default="nem_nav")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--paper_dir", default="paper")
    args = p.parse_args()

    run_root = Path("outputs/matrix") / args.run_id
    if not run_root.exists():
        raise SystemExit(f"missing {run_root}; run experiment matrix first")

    stats = collect_failures(run_root, mode=args.mode, max_steps=args.max_steps)
    paper_dir = Path(args.paper_dir)
    (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_table(stats, paper_dir / "tables" / "failure_taxonomy.tex")
    plot_taxonomy(stats, paper_dir / "figures" / "failure_taxonomy.pdf")
    (run_root / "failures.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"[done] failures.json written to {run_root}; "
          f"{stats['failed']}/{stats['total']} episodes failed")


if __name__ == "__main__":
    main()
