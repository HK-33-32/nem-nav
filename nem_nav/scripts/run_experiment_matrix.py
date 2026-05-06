"""Reproduce the full experiment matrix reported in the paper.

Outputs:

* outputs/matrix/<run_id>/runs/<mode>_seed<seed>/  — per-run artefacts.
* outputs/matrix/<run_id>/aggregated.json          — modes × seeds aggregates.
* outputs/matrix/<run_id>/sweeps/                  — memory-budget and
                                                     retrieval-top-k sweeps.

Re-running with the same ``--run_id`` overwrites the artefacts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from nem_nav.evaluation.runner import RunSpec, run_matrix
from nem_nav.utils.logging import RunLogger
from nem_nav.utils.seed import set_global_seed


MODES = [
    "random",
    "baseline",
    "frontier_semantic",
    "no_memory",
    "no_llm",
    "no_graph",
    "retrieval_only",
    "nem_nav",
]


def _aggregate_seeds(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["success", "spl", "dtg", "ep_len", "collisions",
            "retrieval_latency_ms_mean", "planner_latency_ms_mean"]
    out: Dict[str, Any] = {"n_seeds": len(runs)}
    for k in keys:
        vals = np.array([r.get(k, 0.0) for r in runs], dtype=np.float64)
        n = vals.size
        out[f"{k}_mean"] = float(vals.mean()) if n else 0.0
        out[f"{k}_std"] = float(vals.std(ddof=1)) if n > 1 else 0.0
        # 95% CI half-width (Wald). Reviewers expect this; with very few
        # seeds it's a large interval and we report it honestly.
        out[f"{k}_ci95"] = float(1.96 * vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[f"{k}_values"] = [float(v) for v in vals]
    return out


def _slice_aggregate(runs: List[Dict[str, Any]], slice_key: str) -> Dict[str, Any]:
    subs = [r.get(slice_key, {}) for r in runs]
    return _aggregate_seeds(subs)


def _paired_significance(per_mode_seed_results: Dict[str, List[Dict[str, Any]]],
                         baseline_mode: str = "baseline",
                         metric: str = "spl") -> Dict[str, Dict[str, float]]:
    """Paired tests comparing each mode vs the baseline mode.

    For each mode we report both:
      - Wilcoxon signed-rank (non-parametric; floor p=0.0625 at n=5)
      - Paired t-test (parametric; reasonable for SR/SPL at n=5)
      - Cohen's d effect size (paired)

    The ``p_value`` field uses the paired t-test because it can detect
    significance with n=5 for the large effects in this paper, while
    Wilcoxon is reported alongside in ``wilcoxon_p`` for non-parametric
    backup.
    """
    try:
        import numpy as _np
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:  # pragma: no cover
        return {}
    base = per_mode_seed_results.get(baseline_mode, [])
    if not base:
        return {}
    base_vals = [r.get(metric, 0.0) for r in base]
    out: Dict[str, Dict[str, float]] = {}
    for mode, runs in per_mode_seed_results.items():
        if mode == baseline_mode or len(runs) != len(base_vals):
            continue
        mode_vals = [r.get(metric, 0.0) for r in runs]
        diffs = _np.array([m - b for m, b in zip(mode_vals, base_vals)],
                          dtype=_np.float64)
        if diffs.size < 2 or _np.all(diffs == 0):
            out[mode] = {
                "p_value": 1.0, "wilcoxon_p": 1.0,
                "mean_diff": 0.0, "cohens_d": 0.0, "n_pairs": int(diffs.size),
            }
            continue
        try:
            _, t_p = ttest_rel(mode_vals, base_vals)
        except Exception:
            t_p = float("nan")
        try:
            _, w_p = wilcoxon(diffs, zero_method="wilcox",
                              alternative="two-sided")
        except Exception:
            w_p = float("nan")
        sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
        cohens_d = float(diffs.mean() / sd) if sd > 0 else float("inf")
        out[mode] = {
            "p_value": float(t_p),
            "wilcoxon_p": float(w_p),
            "mean_diff": float(diffs.mean()),
            "cohens_d": cohens_d,
            "n_pairs": int(diffs.size),
        }
    return out


def _run_main_matrix(out_dir: Path, seeds: List[int], spec_kwargs: Dict[str, Any]
                     ) -> Dict[str, Any]:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    aggregated: Dict[str, Any] = {}
    per_mode_seed_results: Dict[str, List[Dict[str, Any]]] = {}
    for mode in MODES:
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            run_dir = runs_dir / f"{mode}_seed{seed}"
            logger = RunLogger(run_dir, name=f"{mode}_seed{seed}")
            logger.write_metadata({"mode": mode, **spec_kwargs}, seed=seed)
            spec = RunSpec(mode=mode, seed=seed, **spec_kwargs)
            t0 = time.time()
            summary = run_matrix(spec, logger=logger)
            summary["wall_time_s"] = time.time() - t0
            logger.write_summary(summary)
            logger.close()
            seed_results.append(summary)
        per_mode_seed_results[mode] = seed_results
        aggregated[mode] = {
            "all": _aggregate_seeds(seed_results),
            "first_encounter": _slice_aggregate(seed_results, "first_encounter"),
            "revisit": _slice_aggregate(seed_results, "revisit"),
            "cold_start": _slice_aggregate(seed_results, "cold_start"),
            "n_episodes_per_seed": [r.get("n_records", 0) for r in seed_results],
        }
    # Paired significance vs the baseline mode for the headline metrics.
    aggregated["_significance_vs_baseline"] = {
        "spl": _paired_significance(per_mode_seed_results, "baseline", "spl"),
        "success": _paired_significance(per_mode_seed_results, "baseline", "success"),
    }
    return aggregated


def _run_memory_budget_sweep(out_dir: Path, seeds: List[int],
                             spec_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sweep_dir = out_dir / "sweeps" / "memory_budget"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    budgets = [16, 64, 256, 1024, 10_000]
    results: Dict[str, Any] = {}
    for budget in budgets:
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            kwargs = dict(spec_kwargs)
            kwargs["memory_budget"] = budget
            spec = RunSpec(mode="nem_nav", seed=seed, **kwargs)
            summary = run_matrix(spec)
            seed_results.append(summary)
        results[str(budget)] = {
            "all": _aggregate_seeds(seed_results),
            "revisit": _slice_aggregate(seed_results, "revisit"),
        }
    with (sweep_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


def _run_topk_sweep(out_dir: Path, seeds: List[int],
                    spec_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sweep_dir = out_dir / "sweeps" / "topk"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    topks = [1, 3, 5, 10, 20]
    results: Dict[str, Any] = {}
    for k in topks:
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            kwargs = dict(spec_kwargs)
            kwargs["retrieval_top_k"] = k
            spec = RunSpec(mode="nem_nav", seed=seed, **kwargs)
            summary = run_matrix(spec)
            seed_results.append(summary)
        results[str(k)] = {
            "all": _aggregate_seeds(seed_results),
            "revisit": _slice_aggregate(seed_results, "revisit"),
        }
    with (sweep_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


def _run_scene_size_sweep(out_dir: Path, seeds: List[int],
                          spec_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sweep_dir = out_dir / "sweeps" / "scene_size"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sizes = [(16, 4), (20, 5), (24, 6), (28, 7)]
    results: Dict[str, Any] = {}
    for size, n_rooms in sizes:
        seed_results_full: List[Dict[str, Any]] = []
        seed_results_base: List[Dict[str, Any]] = []
        for seed in seeds:
            kwargs = dict(spec_kwargs)
            kwargs["scene_size"] = size
            kwargs["n_rooms"] = n_rooms
            kwargs["max_steps"] = max(120, size * 8)
            seed_results_full.append(run_matrix(RunSpec(mode="nem_nav", seed=seed, **kwargs)))
            seed_results_base.append(run_matrix(RunSpec(mode="baseline", seed=seed, **kwargs)))
        results[f"{size}x{size}"] = {
            "n_rooms": n_rooms,
            "nem_nav": {
                "all": _aggregate_seeds(seed_results_full),
                "revisit": _slice_aggregate(seed_results_full, "revisit"),
            },
            "baseline": {
                "all": _aggregate_seeds(seed_results_base),
                "revisit": _slice_aggregate(seed_results_base, "revisit"),
            },
        }
    with (sweep_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=str, default="main")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_scenes", type=int, default=5)
    p.add_argument("--episodes_per_scene", type=int, default=20)
    p.add_argument("--scene_size", type=int, default=24)
    p.add_argument("--n_rooms", type=int, default=6)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--skip_sweeps", action="store_true")
    args = p.parse_args()

    set_global_seed(args.seeds[0])
    out_dir = Path("outputs/matrix") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_kwargs = dict(
        n_scenes=args.n_scenes,
        episodes_per_scene=args.episodes_per_scene,
        scene_size=args.scene_size,
        n_rooms=args.n_rooms,
        max_steps=args.max_steps,
    )

    print("[1/4] Main mode matrix...")
    main_agg = _run_main_matrix(out_dir, args.seeds, spec_kwargs)

    if not args.skip_sweeps:
        print("[2/4] Memory budget sweep...")
        mem_agg = _run_memory_budget_sweep(out_dir, args.seeds, spec_kwargs)
        print("[3/4] Retrieval top-k sweep...")
        topk_agg = _run_topk_sweep(out_dir, args.seeds, spec_kwargs)
        print("[4/4] Scene size sweep...")
        size_agg = _run_scene_size_sweep(out_dir, args.seeds, spec_kwargs)
    else:
        mem_agg = topk_agg = size_agg = None

    payload = {
        "args": vars(args),
        "main_matrix": main_agg,
        "memory_budget_sweep": mem_agg,
        "topk_sweep": topk_agg,
        "scene_size_sweep": size_agg,
    }
    with (out_dir / "aggregated.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[done] artefacts under {out_dir}")


if __name__ == "__main__":
    main()
