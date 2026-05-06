"""Deterministic structured planner with an interface for swapping in a
local quantized LLM.

The planner consumes a goal text and a list of retrieved memory items and
produces a structured plan with the following keys:

    {
        "target_room": <room_type or None>,
        "confidence": <float in [0, 1]>,
        "rationale": <short string>,
        "bias_direction": <(dr, dc) unit vector or None>,
    }

Two backends are supplied:

* ``DeterministicPlanner`` (default): rule-based aggregation over the
  retrieved memory metadata + ontology priors. Always returns a
  parse-safe, JSON-serializable plan, mirroring the structured output a
  prompted LLM would produce after parsing.
* ``LLMPlanner`` (interface only): a stub that demonstrates how to call a
  local quantized model via a user-supplied callable. The stub gracefully
  falls back to ``DeterministicPlanner`` if the callable raises or returns
  an invalid plan, matching the "deterministic fallback when LLM output is
  invalid" requirement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...env.ontology import OBJECT_CLASSES, ROOM_TYPES, primary_room
from ..perception.encoder import OntologyVLMEncoder


@dataclass
class Plan:
    target_room: Optional[str]
    confidence: float
    rationale: str
    bias_direction: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_room": self.target_room,
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "bias_direction": list(self.bias_direction) if self.bias_direction else None,
        }


def parse_plan(text: str) -> Optional[Plan]:
    """Parse a JSON plan string. Returns None on any parse failure."""
    try:
        obj = json.loads(text)
        return Plan(
            target_room=obj.get("target_room"),
            confidence=float(obj.get("confidence", 0.0)),
            rationale=str(obj.get("rationale", "")),
            bias_direction=tuple(obj["bias_direction"]) if obj.get("bias_direction") else None,
        )
    except Exception:
        return None


class DeterministicPlanner:
    """Rule-based planner driven by ontology + retrieved memory.

    Decision procedure:

    1. From the goal text, recover the target object class (last lexical
       match).
    2. Use the ontology to obtain its primary room.
    3. Inspect retrieved memories: if any are co-located with the goal
       class or a sibling object, raise confidence and (if the memory
       carries a stored "pose") suggest a bias direction toward the cluster
       mean of those poses.
    4. Compose a one-line rationale.
    """

    name = "deterministic"

    def __init__(self, encoder: OntologyVLMEncoder, sim_threshold: float = 0.45):
        self.encoder = encoder
        self.sim_threshold = float(sim_threshold)

    def _goal_class(self, goal_text: str) -> Optional[str]:
        for tok in goal_text.lower().replace("_", " ").split()[::-1]:
            if tok in OBJECT_CLASSES:
                return tok
            if tok in ROOM_TYPES:
                return tok
        return None

    def plan(self, goal_text: str, retrieved_memories: List[Tuple[int, float, Dict[str, Any]]],
             current_pose: Optional[Tuple[int, int]] = None) -> Plan:
        target = self._goal_class(goal_text)
        if target is None:
            return Plan(target_room=None, confidence=0.0,
                        rationale="goal text unrecognised; defaulting to exploration")
        target_room = primary_room(target) if target in OBJECT_CLASSES else target

        # Inspect retrieved memories.
        relevant: List[Dict[str, Any]] = []
        for mid, score, meta in retrieved_memories:
            mem_objs = meta.get("visible_objects", [])
            mem_room = meta.get("local_room")
            sim = float(score)
            useful = False
            if any(c == target for c, _, _ in mem_objs):
                sim += 0.5
                useful = True
            if mem_room == target_room:
                sim += 0.2
                useful = True
            if useful or sim >= self.sim_threshold:
                rec = dict(meta)
                rec["_score"] = sim
                relevant.append(rec)

        if not relevant:
            return Plan(target_room=target_room, confidence=0.3,
                        rationale=f"no useful memories yet; head toward likely {target_room}")

        # Compute mean pose of relevant memories with poses.
        poses = [m.get("pose") for m in relevant if m.get("pose") is not None]
        bias = None
        confidence = min(1.0, 0.4 + 0.1 * len(relevant))
        if poses and current_pose is not None:
            arr = np.array([(p[0], p[1]) for p in poses], dtype=np.float32)
            mean = arr.mean(axis=0)
            dr = mean[0] - current_pose[0]
            dc = mean[1] - current_pose[1]
            n = float(np.hypot(dr, dc))
            if n > 1e-6:
                bias = (dr / n, dc / n)
        return Plan(
            target_room=target_room,
            confidence=confidence,
            rationale=f"{len(relevant)} memories support {target_room} (target={target})",
            bias_direction=bias,
        )

    def emit_json(self, plan: Plan) -> str:
        return json.dumps(plan.to_dict())


_DEFAULT_PROMPT_TEMPLATE = (
    "You are a deterministic indoor-navigation planner. "
    "Given a goal and a list of relevant past observations, "
    "respond with ONLY a JSON object with keys "
    "target_room, confidence, rationale, bias_direction.\n\n"
    "Goal: {goal}\n"
    "Retrieved memories:\n{memories}\n"
    "Output JSON now:"
)


class LLMPlanner:
    """Wrapper around a user-supplied local-LLM callable.

    The callable receives a deterministic prompt string and is expected to
    return raw text. We parse it as JSON; any failure falls back to the
    deterministic planner. This satisfies the spec's requirement for
    "deterministic fallback when LLM output is invalid" without coupling
    the rest of the system to any specific LLM runtime.

    The prompt template can be overridden via ``prompt_template`` (raw
    string with ``{goal}`` and ``{memories}`` placeholders) or via
    ``prompt_template_path`` (a file containing the same template). When
    neither is given, the bundled default in
    ``nem_nav/data/prompts/planner.txt`` is loaded if present, else a
    hard-coded fallback string is used.
    """

    name = "llm"

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        encoder: OntologyVLMEncoder,
        prompt_template: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
    ):
        from pathlib import Path
        self.llm_fn = llm_fn
        self.fallback = DeterministicPlanner(encoder)
        if prompt_template is not None:
            self._template = prompt_template
        elif prompt_template_path is not None:
            self._template = Path(prompt_template_path).read_text(encoding="utf-8")
        else:
            bundled = Path(__file__).resolve().parents[2] / "data" / "prompts" / "planner.txt"
            self._template = (
                bundled.read_text(encoding="utf-8")
                if bundled.exists() else _DEFAULT_PROMPT_TEMPLATE
            )

    def _build_prompt(self, goal_text: str,
                      retrieved_memories: List[Tuple[int, float, Dict[str, Any]]]) -> str:
        mem_lines = []
        for mid, score, meta in retrieved_memories[:5]:
            mem_lines.append(
                f"  - id={mid} score={score:.2f} room={meta.get('local_room')} "
                f"objects={[c for c, _, _ in meta.get('visible_objects', [])]}"
            )
        memories = "\n".join(mem_lines) if mem_lines else "  (no relevant memories)"
        return self._template.format(goal=goal_text, memories=memories)

    def plan(self, goal_text: str, retrieved_memories: List[Tuple[int, float, Dict[str, Any]]],
             current_pose: Optional[Tuple[int, int]] = None) -> Plan:
        prompt = self._build_prompt(goal_text, retrieved_memories)
        try:
            text = self.llm_fn(prompt)
            parsed = parse_plan(text)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        return self.fallback.plan(goal_text, retrieved_memories, current_pose)
