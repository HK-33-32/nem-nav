"""Generate paper-ready tables (.tex) and figures (.pdf/.png) from the
aggregated experiment outputs.

Reads ``outputs/matrix/<run_id>/aggregated.json`` (and optional sweep
JSONs under ``sweeps/``) and writes:

* paper/tables/main_results.tex
* paper/tables/first_vs_revisit.tex
* paper/tables/scene_size_sweep.tex (if available)
* paper/figures/learning_curve_revisit.pdf (or .png if pdf backend missing)
* paper/figures/memory_budget_sweep.pdf
* paper/figures/topk_sweep.pdf

Designed to be deterministic and idempotent.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_MODE_ORDER = [
    "random", "baseline", "frontier_semantic",
    "no_memory", "no_llm", "no_graph",
    "retrieval_only", "nem_nav",
]

_PRETTY = {
    "random": "Random",
    "baseline": "Frontier",
    "frontier_semantic": "Frontier+Sem",
    "no_memory": "NEM-Nav $-$ Mem",
    "no_llm": "NEM-Nav $-$ LLM",
    "no_graph": "NEM-Nav $-$ Graph",
    "retrieval_only": "Retrieval-only",
    "nem_nav": "NEM-Nav (full)",
}


def _fmt(mean: float, spread: float, decimals: int = 2) -> str:
    """Format ``mean ± spread`` (where spread is std or 95% CI half-width)."""
    if math.isnan(mean):
        return "--"
    return f"{mean:.{decimals}f}\\,$\\pm$\\,{spread:.{decimals}f}"


def _significance_marker(mode: str, sig: Dict[str, Dict[str, float]]) -> str:
    """Return an asterisk-superscript marking p-value vs the baseline.

    ``*`` for $p < 0.05$, ``**`` for $p < 0.01$, ``\\dagger`` for $p < 0.001$.
    """
    if not sig or mode not in sig:
        return ""
    p = float(sig[mode].get("p_value", 1.0))
    if p < 0.001:
        return r"$^{\dagger}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def write_main_table(payload: Dict[str, Any], out_path: Path) -> None:
    matrix = payload["main_matrix"]
    sig_spl = matrix.get("_significance_vs_baseline", {}).get("spl", {})
    n_seeds = next(
        (matrix[m]["all"].get("n_seeds", 0) for m in matrix if m in _MODE_ORDER),
        0,
    )
    rows: List[str] = []
    rows.append(r"\begin{tabular}{l@{\hspace{6pt}}cccc}")
    rows.append(r"\toprule")
    rows.append(r"\textbf{Method} & \textbf{SR} $\uparrow$ "
                r"& \textbf{SPL} $\uparrow$ "
                r"& \textbf{DTG} $\downarrow$ "
                r"& \textbf{Steps} $\downarrow$ \\")
    rows.append(r"\midrule")
    for mode in _MODE_ORDER:
        if mode not in matrix:
            continue
        agg = matrix[mode]["all"]
        # Use 95% CI half-width when available, fall back to std.
        sr_spread = agg.get("success_ci95", agg.get("success_std", 0))
        spl_spread = agg.get("spl_ci95", agg.get("spl_std", 0))
        dtg_spread = agg.get("dtg_ci95", agg.get("dtg_std", 0))
        len_spread = agg.get("ep_len_ci95", agg.get("ep_len_std", 0))
        marker = _significance_marker(mode, sig_spl)
        rows.append(
            f"{_PRETTY[mode]}{marker} & "
            f"{_fmt(agg['success_mean'], sr_spread)} "
            f"& {_fmt(agg['spl_mean'], spl_spread)} "
            f"& {_fmt(agg['dtg_mean'], dtg_spread, decimals=1)} "
            f"& {_fmt(agg['ep_len_mean'], len_spread, decimals=0)} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    if n_seeds > 0:
        rows.append(
            r"% n_seeds=" + str(n_seeds)
            + r"; values are mean $\pm$ 95\% CI; "
            + r"$^{*}p<0.05$, $^{**}p<0.01$, $^{\dagger}p<0.001$ "
            + r"(paired t-test vs Frontier baseline on per-seed SPL; "
            + r"Wilcoxon $p$ and Cohen's $d$ in the supplementary)"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_first_vs_revisit_table(payload: Dict[str, Any], out_path: Path) -> None:
    matrix = payload["main_matrix"]
    rows: List[str] = []
    rows.append(r"\begin{tabular}{l@{\hspace{4pt}}cc@{\hspace{8pt}}cc@{\hspace{8pt}}cc}")
    rows.append(r"\toprule")
    rows.append(r"& \multicolumn{2}{c}{\textbf{Cold start}}"
                r" & \multicolumn{2}{c}{\textbf{First encounter}}"
                r" & \multicolumn{2}{c}{\textbf{Revisit}} \\")
    rows.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    rows.append(r"\textbf{Method} & SR $\uparrow$ & SPL $\uparrow$ & SR $\uparrow$ & SPL $\uparrow$ & SR $\uparrow$ & SPL $\uparrow$ \\")
    rows.append(r"\midrule")
    for mode in _MODE_ORDER:
        if mode not in matrix:
            continue
        cold = matrix[mode].get("cold_start", {})
        fe = matrix[mode]["first_encounter"]
        rv = matrix[mode]["revisit"]
        rows.append(
            f"{_PRETTY[mode]} "
            f"& {_fmt(cold.get('success_mean', float('nan')), cold.get('success_ci95', cold.get('success_std', 0)))} "
            f"& {_fmt(cold.get('spl_mean', float('nan')), cold.get('spl_ci95', cold.get('spl_std', 0)))} "
            f"& {_fmt(fe['success_mean'], fe.get('success_ci95', fe.get('success_std', 0)))} "
            f"& {_fmt(fe['spl_mean'], fe.get('spl_ci95', fe.get('spl_std', 0)))} "
            f"& {_fmt(rv['success_mean'], rv.get('success_ci95', rv.get('success_std', 0)))} "
            f"& {_fmt(rv['spl_mean'], rv.get('spl_ci95', rv.get('spl_std', 0)))} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_scene_size_table(payload: Dict[str, Any], out_path: Path) -> None:
    sweep = payload.get("scene_size_sweep")
    if not sweep:
        return
    rows: List[str] = []
    rows.append(r"\begin{tabular}{l@{\hspace{6pt}}cccc}")
    rows.append(r"\toprule")
    rows.append(r"\textbf{Scene} & \multicolumn{2}{c}{\textbf{Frontier}} & \multicolumn{2}{c}{\textbf{NEM-Nav}} \\")
    rows.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    rows.append(r" & SR $\uparrow$ & SPL $\uparrow$ & SR $\uparrow$ & SPL $\uparrow$ \\")
    rows.append(r"\midrule")
    for size_key, vals in sweep.items():
        b = vals["baseline"]["all"]
        n = vals["nem_nav"]["all"]
        rows.append(
            f"{size_key} ({vals['n_rooms']} rooms) "
            f"& {_fmt(b['success_mean'], b['success_std'])} "
            f"& {_fmt(b['spl_mean'], b['spl_std'])} "
            f"& {_fmt(n['success_mean'], n['success_std'])} "
            f"& {_fmt(n['spl_mean'], n['spl_std'])} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_memory_budget_sweep(payload: Dict[str, Any], out_path: Path) -> None:
    sweep = payload.get("memory_budget_sweep")
    if not sweep:
        return
    budgets = sorted([int(k) for k in sweep.keys()])
    means_sr = [sweep[str(b)]["all"]["success_mean"] for b in budgets]
    stds_sr = [sweep[str(b)]["all"]["success_std"] for b in budgets]
    means_spl = [sweep[str(b)]["all"]["spl_mean"] for b in budgets]
    stds_spl = [sweep[str(b)]["all"]["spl_std"] for b in budgets]
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    ax.errorbar(budgets, means_sr, yerr=stds_sr, marker="o", label="SR", linewidth=1.5)
    ax.errorbar(budgets, means_spl, yerr=stds_spl, marker="s", label="SPL", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Memory budget $|\\mathcal{M}|$")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_topk_sweep(payload: Dict[str, Any], out_path: Path) -> None:
    sweep = payload.get("topk_sweep")
    if not sweep:
        return
    ks = sorted([int(k) for k in sweep.keys()])
    means_sr = [sweep[str(k)]["all"]["success_mean"] for k in ks]
    stds_sr = [sweep[str(k)]["all"]["success_std"] for k in ks]
    means_spl = [sweep[str(k)]["all"]["spl_mean"] for k in ks]
    stds_spl = [sweep[str(k)]["all"]["spl_std"] for k in ks]
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    ax.errorbar(ks, means_sr, yerr=stds_sr, marker="o", label="SR", linewidth=1.5)
    ax.errorbar(ks, means_spl, yerr=stds_spl, marker="s", label="SPL", linewidth=1.5)
    ax.set_xlabel("Retrieval top-$k$")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_learning_curve(run_root: Path, out_path: Path) -> None:
    """Plot mean SPL vs episode index (within-scene), aggregated across
    seeds and scenes, comparing nem_nav, no_memory, baseline."""
    targets = ["baseline", "no_memory", "no_llm", "no_graph", "nem_nav"]
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, mode in enumerate(targets):
        per_ep_spl: Dict[int, List[float]] = {}
        per_ep_sr: Dict[int, List[float]] = {}
        for seed_dir in sorted((run_root / "runs").glob(f"{mode}_seed*")):
            ep_path = seed_dir / "episodes.jsonl"
            if not ep_path.exists():
                continue
            for line in ep_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ep = rec.get("episode")
                m = rec.get("metrics", {})
                spl = m.get("shortest_path_length", 0)
                pl = m.get("path_length", 0)
                s = int(m.get("success", 0))
                if ep is None:
                    continue
                if pl > 0 and spl > 0 and s == 1:
                    per_ep_spl.setdefault(ep, []).append(spl / max(pl, spl))
                else:
                    per_ep_spl.setdefault(ep, []).append(0.0)
                per_ep_sr.setdefault(ep, []).append(s)
        if not per_ep_spl:
            continue
        eps = sorted(per_ep_spl.keys())
        mean_spl = [float(np.mean(per_ep_spl[e])) for e in eps]
        ax.plot(eps, mean_spl, label=_PRETTY[mode], color=color_cycle[i % len(color_cycle)],
                linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Episode index within scene")
    ax.set_ylabel("Mean SPL")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="main")
    p.add_argument("--paper_dir", default="paper")
    args = p.parse_args()

    run_root = Path("outputs/matrix") / args.run_id
    agg_path = run_root / "aggregated.json"
    if not agg_path.exists():
        raise SystemExit(f"missing {agg_path}; run experiment matrix first")
    payload = json.loads(agg_path.read_text(encoding="utf-8"))

    paper_dir = Path(args.paper_dir)
    tables = paper_dir / "tables"
    figs = paper_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    write_main_table(payload, tables / "main_results.tex")
    write_first_vs_revisit_table(payload, tables / "first_vs_revisit.tex")
    write_scene_size_table(payload, tables / "scene_size_sweep.tex")

    try:
        plot_memory_budget_sweep(payload, figs / "memory_budget_sweep.pdf")
    except Exception:
        plot_memory_budget_sweep(payload, figs / "memory_budget_sweep.png")
    try:
        plot_topk_sweep(payload, figs / "topk_sweep.pdf")
    except Exception:
        plot_topk_sweep(payload, figs / "topk_sweep.png")
    try:
        plot_learning_curve(run_root, figs / "learning_curve.pdf")
    except Exception:
        plot_learning_curve(run_root, figs / "learning_curve.png")

    print(f"[done] tables and figures written to {paper_dir}")


if __name__ == "__main__":
    main()
