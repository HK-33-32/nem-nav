"""Run the *training-free* baselines reported in the paper.

These are the modes that do not require any learned weights and are
therefore reproducible from a clean checkout:

    random              - uniform random sanity floor
    baseline            - frontier-only exploration
    frontier_semantic   - frontier + goal-text similarity (no memory)

Outputs land under ``outputs/<mode>_seed<seed>/`` exactly as
``nem_nav.run`` would write them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


MODES = ["random", "baseline", "frontier_semantic"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_scenes", type=int, default=5)
    p.add_argument("--episodes_per_scene", type=int, default=15)
    p.add_argument("--max_steps", type=int, default=200)
    args = p.parse_args()

    for seed in args.seeds:
        for mode in MODES:
            cmd = [
                sys.executable, "-m", "nem_nav.run",
                "--mode", mode,
                "--seed", str(seed),
                "--n_scenes", str(args.n_scenes),
                "--episodes_per_scene", str(args.episodes_per_scene),
                "--max_steps", str(args.max_steps),
            ]
            print("RUN:", " ".join(cmd))
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
