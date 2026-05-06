"""Unit tests for the SemanticGraph."""
from __future__ import annotations

import numpy as np

from nem_nav.models.graph.semantic_graph import SemanticGraph


def _e(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def test_repeated_observations_collapse_into_one_node():
    g = SemanticGraph(sim_threshold=0.5, pose_radius=2.0)
    e = _e(0)
    n1 = g.observe(e, (5, 5), "kitchen", 0)
    n2 = g.observe(e + 1e-3 * _e(1), (5, 6), "kitchen", 1)
    assert n1 == n2
    assert g._nodes[n1].visit_count == 2


def test_distinct_rooms_yield_distinct_nodes():
    g = SemanticGraph(sim_threshold=0.95, pose_radius=2.0)
    e1, e2 = _e(0), _e(1)
    n1 = g.observe(e1, (1, 1), "kitchen", 0)
    n2 = g.observe(e2, (10, 10), "bedroom", 1)
    assert n1 != n2
    assert g.g.has_edge(n1, n2)


def test_neighborhood_expands_with_hops():
    g = SemanticGraph(sim_threshold=0.95, pose_radius=1.5)
    for i in range(4):
        g.observe(_e(i), (i * 3, 0), f"r{i}", i)
    nb1 = set(g.neighborhood(0, hops=1))
    nb2 = set(g.neighborhood(0, hops=2))
    assert nb1.issubset(nb2)
