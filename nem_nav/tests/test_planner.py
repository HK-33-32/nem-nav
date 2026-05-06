"""Unit tests for the planner."""
from __future__ import annotations



from nem_nav.models.perception.encoder import OntologyVLMEncoder
from nem_nav.models.planner.planner import (
    DeterministicPlanner, LLMPlanner, parse_plan,
)


def test_parse_plan_handles_garbage():
    assert parse_plan("not json") is None


def test_parse_plan_handles_partial():
    p = parse_plan('{"target_room": "kitchen"}')
    assert p is not None
    assert p.target_room == "kitchen"
    assert p.confidence == 0.0


def test_deterministic_planner_picks_room_from_goal():
    enc = OntologyVLMEncoder()
    pl = DeterministicPlanner(enc)
    plan = pl.plan("find a bed", retrieved_memories=[])
    assert plan.target_room == "bedroom"
    assert 0.0 <= plan.confidence <= 1.0


def test_deterministic_planner_uses_memory_signal():
    enc = OntologyVLMEncoder()
    pl = DeterministicPlanner(enc)
    mem_meta = {"visible_objects": [("bed", 0, 0)],
                "local_room": "bedroom",
                "pose": (5, 5)}
    plan = pl.plan("find a bed",
                   retrieved_memories=[(0, 0.9, mem_meta)],
                   current_pose=(0, 0))
    assert plan.target_room == "bedroom"
    assert plan.confidence > 0.4
    assert plan.bias_direction is not None
    bd = plan.bias_direction
    # Pointing toward (5,5) from (0,0) means both components positive.
    assert bd[0] > 0 and bd[1] > 0


def test_llm_planner_falls_back_on_garbage():
    enc = OntologyVLMEncoder()

    def bad_llm(prompt: str) -> str:
        return "definitely not json"

    pl = LLMPlanner(bad_llm, enc)
    plan = pl.plan("find a bed", retrieved_memories=[])
    assert plan.target_room == "bedroom"
