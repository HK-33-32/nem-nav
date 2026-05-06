"""Tests for the OpenCLIP perception backend.

These tests are skipped automatically when ``open_clip_torch`` is not
installed (i.e. anyone who runs only the core dependencies). The CI matrix
job that exercises the optional ``openclip`` extra picks them up.
"""
from __future__ import annotations

import numpy as np
import pytest

open_clip = pytest.importorskip("open_clip", reason="open_clip_torch not installed")

from nem_nav.models.perception.encoder import build_encoder  # noqa: E402
from nem_nav.models.perception.openclip_backend import OpenCLIPEncoder  # noqa: E402


pytestmark = pytest.mark.openclip


@pytest.fixture(scope="module")
def encoder() -> OpenCLIPEncoder:
    return build_encoder("openclip")


def test_dim_matches_vit_b_32(encoder):
    assert encoder.dim == 512


def test_encode_text_is_unit_norm(encoder):
    v = encoder.encode_text("a photo of a chair")
    assert v.shape == (encoder.dim,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_encode_image_falls_back_when_no_rgb(encoder):
    # SemanticHomeWorld observations have no `rgb` key — the backend should
    # delegate to the ontology fallback instead of crashing.
    obs = {"visible_objects": [("chair", 0, 0)], "local_room": "living_room",
           "visible_room_types": []}
    v = encoder.encode_image(obs)
    assert v.shape == (encoder.dim,)
    # Fallback returns either a unit-norm vector or zeros for empty obs.
    n = float(np.linalg.norm(v))
    assert n == 0.0 or abs(n - 1.0) < 1e-3
