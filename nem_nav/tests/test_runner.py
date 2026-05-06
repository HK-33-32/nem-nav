"""End-to-end smoke test for the runner."""
from __future__ import annotations

import json

from nem_nav.evaluation.runner import RunSpec, run_matrix
from nem_nav.utils.logging import RunLogger


def test_baseline_runs_end_to_end():
    spec = RunSpec(mode="baseline", n_scenes=1, episodes_per_scene=2,
                   max_steps=60, scene_size=18, n_rooms=4)
    out = run_matrix(spec)
    assert out["n_episodes"] == 2
    assert 0.0 <= out["success"] <= 1.0
    # Baseline disables memory and planner, so latency means must be zero.
    assert out["retrieval_latency_ms_mean"] == 0.0
    assert out["planner_latency_ms_mean"] == 0.0


def test_full_nem_nav_runs_end_to_end():
    spec = RunSpec(mode="nem_nav", n_scenes=1, episodes_per_scene=2,
                   max_steps=60, scene_size=18, n_rooms=4)
    out = run_matrix(spec)
    assert out["n_episodes"] == 2
    assert 0.0 <= out["spl"] <= 1.0
    # Full system uses retrieval and planner — both should record nonzero
    # latency. We use > 0.0 (microsecond resolution from perf_counter).
    assert out["retrieval_latency_ms_mean"] > 0.0
    assert out["planner_latency_ms_mean"] > 0.0


def test_runlogger_records_peak_ram(tmp_path):
    spec = RunSpec(mode="nem_nav", n_scenes=1, episodes_per_scene=1,
                   max_steps=30, scene_size=14, n_rooms=3)
    logger = RunLogger(tmp_path / "run", name="t")
    logger.write_metadata({"mode": spec.mode}, seed=spec.seed)
    summary = run_matrix(spec, logger=logger)
    logger.write_summary(summary)
    logger.close()
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    # psutil is a hard requirement, so peak_ram_mb must be populated.
    assert isinstance(meta["peak_ram_mb"], float)
    assert meta["peak_ram_mb"] > 0.0
    # peak_vram_mb is None on CPU-only environments — that's fine.
    assert meta["peak_vram_mb"] is None or isinstance(meta["peak_vram_mb"], float)
