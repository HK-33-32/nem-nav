"""Tests for the local-LLM backends and the LLMPlanner integration."""
from __future__ import annotations

import json
import os

import pytest

from nem_nav.models.perception.encoder import OntologyVLMEncoder
from nem_nav.models.planner.llm_backends import build_llm_fn
from nem_nav.models.planner.planner import LLMPlanner, parse_plan

try:
    import llama_cpp  # noqa: F401
    _HAS_LLAMA_CPP = True
except Exception:
    _HAS_LLAMA_CPP = False


# ---------------------------------------------------------------- echo backend

def test_echo_backend_returns_canned_json():
    fn = build_llm_fn("echo", canned={"target_room": "kitchen",
                                      "confidence": 0.9, "rationale": "test",
                                      "bias_direction": [1.0, 0.0]})
    out = fn("any prompt")
    parsed = json.loads(out)
    assert parsed["target_room"] == "kitchen"
    assert parsed["bias_direction"] == [1.0, 0.0]
    plan = parse_plan(out)
    assert plan is not None and plan.target_room == "kitchen"


def test_echo_backend_default_payload_is_parseable():
    fn = build_llm_fn("echo")
    plan = parse_plan(fn("anything"))
    assert plan is not None
    assert plan.target_room is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_llm_fn("nonsense")


# ---------------------------------------------------------------- LLMPlanner

def test_llm_planner_uses_echo_backend():
    encoder = OntologyVLMEncoder()
    fn = build_llm_fn("echo", canned={"target_room": "bedroom",
                                      "confidence": 0.8, "rationale": "echo says bedroom",
                                      "bias_direction": None})
    planner = LLMPlanner(llm_fn=fn, encoder=encoder)
    plan = planner.plan("find a bed", retrieved_memories=[])
    assert plan.target_room == "bedroom"
    assert "echo" in plan.rationale


def test_llm_planner_falls_back_when_echo_returns_garbage():
    encoder = OntologyVLMEncoder()

    def garbage(_prompt: str) -> str:
        return "{this is not valid JSON"

    planner = LLMPlanner(llm_fn=garbage, encoder=encoder)
    plan = planner.plan("find a bed", retrieved_memories=[])
    # Falls back to DeterministicPlanner — bedroom is the primary room for
    # the goal class "bed" in the bundled ontology.
    assert plan.target_room == "bedroom"


def test_llm_planner_uses_bundled_prompt_template():
    encoder = OntologyVLMEncoder()

    captured = {}

    def capture(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps({"target_room": "kitchen", "confidence": 0.5,
                           "rationale": "ok", "bias_direction": None})

    planner = LLMPlanner(llm_fn=capture, encoder=encoder)
    planner.plan("find a chair", retrieved_memories=[])
    assert "find a chair" in captured["prompt"]
    assert "JSON" in captured["prompt"]


# ---------------------------------------------------------------- llama_cpp

@pytest.mark.llm
@pytest.mark.skipif(not _HAS_LLAMA_CPP, reason="llama-cpp-python not installed")
@pytest.mark.skipif(not os.environ.get("NEM_NAV_LLM_PATH"),
                    reason="NEM_NAV_LLM_PATH not set")
def test_llama_cpp_smoke():
    fn = build_llm_fn("llama_cpp", model_path=os.environ["NEM_NAV_LLM_PATH"],
                      max_tokens=64)
    text = fn("Reply with the JSON {\"target_room\": null, \"confidence\": 0.0, "
              "\"rationale\": \"ok\", \"bias_direction\": null}.")
    parsed = parse_plan(text)
    # The deterministic fallback inside the planner takes over if parse fails;
    # for the raw backend test we just check that we got a string back.
    assert isinstance(text, str)
    # Soft assertion — the model may or may not echo cleanly, but if it does
    # we should be able to parse it.
    if parsed is not None:
        assert parsed.confidence >= 0.0
