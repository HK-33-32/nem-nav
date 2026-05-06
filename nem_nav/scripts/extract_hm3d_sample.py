"""Stream-extract a small sample of HM3D-RGBDSemantic view directories
without unpacking the full 60 GB archive.

The tarball at ``data/hm3d/hm3d_1.tar.gz`` contains
``./train/<scene_view_id>/{color,coord,instance,normal,segment}.npy``
per RGB-D view. We walk it sequentially and stop once we have
``--n_files`` files extracted (or earlier, if no progress is made).

Files for a single view are written contiguously inside the tarball,
so a flat "first N files" strategy gives us complete view directories.
"""
from __future__ import annotations

import argparse
import tarfile
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive",
                   default="data/hm3d/hm3d_1.tar.gz",
                   help="Path to one of the HM3D-RGBDSemantic .tar.gz files.")
    p.add_argument("--out_dir", default="data/hm3d_sample")
    p.add_argument("--n_files", type=int, default=200,
                   help="Number of files to extract before stopping. "
                        "Each view contributes 5 files (color/coord/instance"
                        "/normal/segment.npy), so 200 ~= 40 views.")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    seen_views: Counter = Counter()

    with tarfile.open(args.archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            # Format: ./train/<view_id>/<filename>.npy
            if len(parts) < 4 or not parts[-1].endswith(".npy"):
                continue
            view_id = parts[-2]
            target = out_root / view_id / parts[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted += 1
            seen_views[view_id] += 1
            if (extracted % 25) == 0:
                print(f"  extracted {extracted} files / {len(seen_views)} views",
                      flush=True)
            if extracted >= args.n_files:
                break

    n_views = len(seen_views)
    n_complete = sum(1 for v in seen_views.values() if v == 5)
    n_with_segment = sum(1 for v_id in seen_views
                         if (out_root / v_id / "segment.npy").exists())
    n_scenes = len({v.rsplit("_", 2)[0] for v in seen_views})
    print(f"\n[done] extracted {extracted} files into {out_root}")
    print(f"[done] {n_views} view-dirs, {n_complete} complete, "
          f"{n_with_segment} with segment.npy, {n_scenes} unique scene-prefixes")


if __name__ == "__main__":
    main()
