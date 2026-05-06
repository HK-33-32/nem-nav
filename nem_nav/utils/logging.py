"""Run-scoped structured logging."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _peak_ram_mb() -> Optional[float]:
    """Return resident-set-size of the current process in MB (psutil)."""
    if not _HAS_PSUTIL:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _peak_vram_mb() -> Optional[float]:
    """Return peak CUDA memory in MB across all visible devices, or None."""
    if not _HAS_TORCH:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        peak = max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count()))
        return float(peak) / (1024 * 1024)
    except Exception:
        return None


class RunLogger:
    """Writes JSONL episode logs and a textual summary to a per-run directory."""

    def __init__(self, run_dir: str | Path, name: str = "run"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.log_path = self.run_dir / f"{name}.log"
        self._ep_fh = self.episodes_path.open("a", encoding="utf-8")
        self._setup_text_logger()

    def _setup_text_logger(self) -> None:
        self.text = logging.getLogger(f"nem_nav.{self.name}")
        self.text.setLevel(logging.INFO)
        for h in list(self.text.handlers):
            self.text.removeHandler(h)
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.text.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.text.addHandler(sh)

    def info(self, msg: str) -> None:
        self.text.info(msg)

    def write_metadata(self, cfg: Dict[str, Any], seed: int) -> None:
        # Reset CUDA peak counter so peak_vram_mb measures only this run.
        if _HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    for d in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(d)
            except Exception:
                pass
        self._metadata = {
            "config": cfg,
            "seed": seed,
            "git_commit": _git_commit(),
            "started_at": time.time(),
            "python": sys.version,
            "platform": sys.platform,
            # peak_ram_mb / peak_vram_mb are filled in by write_summary
            "peak_ram_mb": None,
            "peak_vram_mb": None,
        }
        self._metadata_path = self.run_dir / "metadata.json"
        with self._metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, indent=2, default=str)

    def log_episode(self, record: Dict[str, Any]) -> None:
        self._ep_fh.write(json.dumps(record, default=float) + "\n")
        self._ep_fh.flush()

    def write_summary(self, summary: Dict[str, Any]) -> None:
        # Refresh peak resource usage and rewrite metadata.json so callers
        # have a single source of truth for {git, seed, peak_ram, peak_vram}.
        meta = getattr(self, "_metadata", None)
        if meta is not None:
            meta["peak_ram_mb"] = _peak_ram_mb()
            meta["peak_vram_mb"] = _peak_vram_mb()
            meta["finished_at"] = time.time()
            with self._metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)
        with self.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)

    def close(self) -> None:
        try:
            self._ep_fh.close()
        except Exception:
            pass
