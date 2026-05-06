"""Validate the ontology encoder against real HM3D-Semantic statistics.

The HM3D-RGBDSemantic archives that ship via Pointcept contain
**3D point clouds**, not 2D RGB images: each ``color.npy`` is
``(N, 3) uint8`` of per-point colors, and ``segment.npy`` is ``(N,)
int16`` of HM3D-Semantic class IDs (40-class scheme;
``-1`` = unannotated). This is the wrong format for OpenCLIP
text-image retrieval.

We therefore validate the **ontology encoder** (``OntologyVLMEncoder``,
the default backbone of every reported number) against real-world
class co-occurrence statistics from HM3D:

  * For each pair of HM3D-Semantic classes ``(a, b)``, count the number
    of point-cloud scenes in which both classes have at least
    ``--min_points`` points.
  * Define real-world co-occurrence as
    ``coocc(a, b) = #(scenes containing both) / #(scenes containing a or b)``.
  * Compute the ontology encoder's text-text cosine for the same pair.
  * Report Pearson correlation between the two over all pairs.

A high correlation (>0.5) is evidence that the ontology priors
encoded in our default encoder agree with what real apartments
actually contain. We make no claim about CLIP itself; for that
reviewers should rerun this script after re-rendering the scenes to
2D images via the HabitatWorld backend (see ``data/README.md``).

Outputs:
  outputs/hm3d_perception/<run_id>/results.json
  paper/tables/perception_hm3d.tex
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# HM3D-Semantic 40-class names. We map a subset of these to the names
# already in the project ontology so the encoder can score them; classes
# absent from the ontology get a centroid fallback.
HM3D_CLASS_NAMES = {
    0: "wall", 1: "floor", 2: "chair", 3: "door", 4: "table",
    5: "picture", 6: "cabinet", 7: "cushion", 8: "window",
    9: "sofa", 10: "bed", 11: "curtain", 12: "chest_of_drawers",
    13: "plant", 14: "sink", 15: "stairs", 16: "ceiling",
    17: "toilet", 18: "stool", 19: "towel", 20: "mirror",
    21: "tv_monitor", 22: "shower", 23: "column", 24: "bathtub",
    25: "counter", 26: "fireplace", 27: "lighting", 28: "beam",
    29: "railing", 30: "shelving", 31: "blinds", 32: "gym_equipment",
    33: "seating", 34: "board_panel", 35: "furniture",
    36: "appliances", 37: "clothes", 38: "objects", 39: "misc",
}


def collect_per_scene_class_counts(
    sample_dir: Path, min_points: int = 50,
) -> Dict[str, set]:
    """Return mapping scene_id -> set(class_name) of classes that have
    at least ``min_points`` annotated points across all the scene's views."""
    counts: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for view_dir in sorted(sample_dir.iterdir()):
        if not view_dir.is_dir():
            continue
        seg_path = view_dir / "segment.npy"
        if not seg_path.exists():
            continue
        scene_id = "_".join(view_dir.name.split("_")[:2])
        seg = np.load(seg_path)
        for cls_id, cnt in zip(*np.unique(seg, return_counts=True)):
            if int(cls_id) < 0 or int(cls_id) not in HM3D_CLASS_NAMES:
                continue
            counts[scene_id][int(cls_id)] += int(cnt)
    return {sid: {HM3D_CLASS_NAMES[cid] for cid, n in cls_counts.items()
                  if n >= min_points}
            for sid, cls_counts in counts.items()}


def compute_cooccurrence(scene_classes: Dict[str, set]) -> Dict[tuple, float]:
    """For every pair of class names, return coocc = |A∩B| / |A∪B|."""
    classes = sorted({c for cs in scene_classes.values() for c in cs})
    presence = {c: {sid for sid, cs in scene_classes.items() if c in cs}
                for c in classes}
    out: Dict[tuple, float] = {}
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            inter = len(presence[a] & presence[b])
            union = len(presence[a] | presence[b])
            if union >= 2:
                out[(a, b)] = inter / union
    return out


def encoder_text_text_sim(encoder, pair: tuple) -> float:
    a, b = pair
    va = encoder.encode_text(a)
    vb = encoder.encode_text(b)
    if np.linalg.norm(va) < 1e-6 or np.linalg.norm(vb) < 1e-6:
        return float("nan")
    return float(np.dot(va, vb))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", default="data/hm3d_sample")
    p.add_argument("--run_id", default="main")
    p.add_argument("--min_points", type=int, default=200)
    p.add_argument("--paper_dir", default="paper")
    args = p.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        raise SystemExit(f"missing {sample_dir}; run extract_hm3d_sample first")

    print("[1/3] reading HM3D semantic point-cloud labels...", flush=True)
    scene_classes = collect_per_scene_class_counts(sample_dir, args.min_points)
    n_scenes = len(scene_classes)
    if n_scenes < 2:
        raise SystemExit(f"too few scenes with semantic labels: {n_scenes}")
    cooccs = compute_cooccurrence(scene_classes)
    print(f"    {n_scenes} scenes, {sum(len(c) for c in scene_classes.values())} class-occurrences,"
          f" {len(cooccs)} class pairs", flush=True)

    print("[2/3] building OntologyVLMEncoder...", flush=True)
    from nem_nav.models.perception.encoder import build_encoder
    encoder = build_encoder("ontology")

    print("[3/3] computing text-text cosine for each pair...", flush=True)
    pairs = list(cooccs.keys())
    real = np.array([cooccs[p] for p in pairs], dtype=np.float64)
    ont = np.array([encoder_text_text_sim(encoder, p) for p in pairs],
                   dtype=np.float64)
    mask = ~np.isnan(ont)
    real_m = real[mask]
    ont_m = ont[mask]
    if real_m.size < 3:
        raise SystemExit("not enough class pairs covered by both ontology and HM3D")

    pearson = float(np.corrcoef(real_m, ont_m)[0, 1])
    spearman = float(_spearman(real_m, ont_m))

    # Report top-10 high-cooccurrence pairs and the encoder's prediction.
    order = np.argsort(-real)
    qualitative: List[Dict[str, Any]] = []
    for k in order[:10]:
        a, b = pairs[int(k)]
        qualitative.append({
            "pair": [a, b],
            "real_cooccurrence": float(real[int(k)]),
            "ontology_cosine": float(ont[int(k)]),
        })

    stats = {
        "n_scenes": n_scenes,
        "n_pairs_total": int(len(pairs)),
        "n_pairs_in_ontology": int(mask.sum()),
        "pearson_real_vs_ontology": pearson,
        "spearman_real_vs_ontology": spearman,
        "top10_real_pairs": qualitative,
    }
    out_root = Path("outputs") / "hm3d_perception" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results.json").write_text(json.dumps(stats, indent=2))
    write_table(stats, Path(args.paper_dir) / "tables" / "perception_hm3d.tex")
    print(f"\n[done] Pearson(real coocc, ontology cos) = {pearson:.3f}")
    print(f"[done] Spearman                            = {spearman:.3f}")
    print(f"[done] over {mask.sum()} pairs across {n_scenes} HM3D scenes")
    print(f"[done] results in {out_root}/results.json + paper table")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def write_table(stats: Dict[str, Any], out_path: Path) -> None:
    rows: List[str] = []
    rows.append(r"\begin{tabular}{lr}")
    rows.append(r"\toprule")
    rows.append(r"\textbf{Statistic} & \textbf{Value} \\")
    rows.append(r"\midrule")
    rows.append(f"\\# HM3D scenes (point-cloud sample) & {stats['n_scenes']} \\\\")
    rows.append(f"\\# HM3D-Semantic class pairs co-occurring & {stats['n_pairs_total']} \\\\")
    rows.append(f"\\# pairs covered by ontology encoder      & {stats['n_pairs_in_ontology']} \\\\")
    rows.append(r"\midrule")
    rows.append(r"Pearson $r$ (real co-occurrence vs encoder cosine) "
                f"& {stats['pearson_real_vs_ontology']:.3f} \\\\")
    rows.append(r"Spearman $\rho$                                    "
                f"& {stats['spearman_real_vs_ontology']:.3f} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
