"""Single-mode runner.

Examples:
    python -m nem_nav.run --mode baseline
    python -m nem_nav.run --mode nem_nav --seed 0 --n_scenes 5 --episodes_per_scene 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation.runner import RunSpec, run_matrix
from .utils.logging import RunLogger
from .utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEM-Nav single-mode runner")
    p.add_argument("--mode", required=True,
                   choices=["baseline", "random", "frontier_semantic", "nem_nav",
                            "no_memory", "no_llm", "no_graph", "retrieval_only",
                            "graph_retrieval", "planner_no_memory"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_scenes", type=int, default=5)
    p.add_argument("--episodes_per_scene", type=int, default=10)
    p.add_argument("--scene_size", type=int, default=24)
    p.add_argument("--n_rooms", type=int, default=6)
    p.add_argument("--view_radius", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--memory_budget", type=int, default=10000)
    p.add_argument("--retrieval_top_k", type=int, default=5)
    p.add_argument("--planner_every_n_steps", type=int, default=8)
    p.add_argument("--encoder", type=str, default="ontology",
                   choices=["ontology", "openclip"],
                   help="Perception backend (default: ontology). Use 'openclip' "
                        "with `pip install -e .[openclip]` for a real CLIP backbone.")
    p.add_argument("--llm_backend", type=str, default="none",
                   choices=["none", "echo", "llama_cpp"],
                   help="Planner LLM backend (default: none → DeterministicPlanner).")
    p.add_argument("--llm_model_path", type=str, default=None,
                   help="Path to GGUF checkpoint for --llm_backend llama_cpp. "
                        "If omitted, NEM_NAV_LLM_PATH env var is used.")
    p.add_argument("--env", type=str, default="semantic_home",
                   choices=["semantic_home", "habitat"],
                   help="Environment backend (default: semantic_home). Use "
                        "'habitat' with `pip install -e .[habitat]` and an "
                        "HM3D/MP3D dataset; configure paths via "
                        "--env_config nem_nav/configs/env/habitat.yaml.")
    p.add_argument("--env_config", type=str, default=None,
                   help="YAML file with extra kwargs for the selected --env "
                        "(applies only when --env is not 'semantic_home').")
    p.add_argument("--out", type=str, default=None,
                   help="Output run directory (default: outputs/<mode>_seed<seed>)")
    return p.parse_args()


def _load_env_kwargs(path: str | None) -> dict | None:
    if not path:
        return None
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)
    out = args.out or f"outputs/{args.mode}_seed{args.seed}"
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(out_path, name=args.mode)
    logger.write_metadata(vars(args), seed=args.seed)
    logger.info(f"Running mode={args.mode} seed={args.seed}")

    spec = RunSpec(
        mode=args.mode,
        n_scenes=args.n_scenes,
        episodes_per_scene=args.episodes_per_scene,
        scene_size=args.scene_size,
        n_rooms=args.n_rooms,
        view_radius=args.view_radius,
        max_steps=args.max_steps,
        memory_budget=args.memory_budget,
        retrieval_top_k=args.retrieval_top_k,
        planner_every_n_steps=args.planner_every_n_steps,
        seed=args.seed,
        encoder=args.encoder,
        llm_backend=args.llm_backend,
        llm_model_path=args.llm_model_path,
        env=args.env,
        env_kwargs=_load_env_kwargs(args.env_config),
    )
    summary = run_matrix(spec, logger=logger)
    logger.write_summary(summary)
    logger.info(json.dumps({k: v for k, v in summary.items()
                            if k not in ("per_scene", "per_goal_class")}, indent=2))
    logger.close()


if __name__ == "__main__":
    main()
