"""Lightweight evaluator that loads a run directory and prints aggregates.

Usage:
    python -m nem_nav.evaluate --run_dir outputs/nem_nav_seed0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()
    run = Path(args.run_dir)
    summary_path = run / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Run: {run}")
    for k in ("mode", "seed", "n_episodes", "success", "spl", "dtg",
              "ep_len", "collisions"):
        print(f"  {k:12s} {summary.get(k)}")
    if "first_encounter" in summary:
        fe = summary["first_encounter"]
        print(f"  first-encounter: SR={fe['success']:.3f} SPL={fe['spl']:.3f} (n={fe['n_episodes']})")
    if "revisit" in summary:
        rv = summary["revisit"]
        print(f"  revisit:         SR={rv['success']:.3f} SPL={rv['spl']:.3f} (n={rv['n_episodes']})")


if __name__ == "__main__":
    main()
