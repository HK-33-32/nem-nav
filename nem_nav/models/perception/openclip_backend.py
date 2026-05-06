"""OpenCLIP perception backend.

Wraps an `open_clip` model so it satisfies the same interface as
``OntologyVLMEncoder``:

* ``encode_image(observation: dict) -> np.ndarray``
* ``encode_text(text: str) -> np.ndarray``
* attribute ``dim``

If the observation does not contain an ``rgb`` array (e.g. when the
environment is the synthetic ``SemanticHomeWorld`` rather than a Habitat /
real-image source), the call falls back to the ontology encoder so the rest
of the pipeline keeps working without changes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import open_clip
    import torch
    _HAS_OPEN_CLIP = True
except Exception:  # pragma: no cover
    open_clip = None
    torch = None
    _HAS_OPEN_CLIP = False

from .encoder import OntologyVLMEncoder, _normalize


class OpenCLIPEncoder:
    """Thin OpenCLIP wrapper with a deterministic ontology fallback."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        precision: str = "fp32",
        ontology_fallback: Optional[OntologyVLMEncoder] = None,
    ):
        if not _HAS_OPEN_CLIP:
            raise RuntimeError(
                "open_clip is not installed; run `pip install -e .[openclip]`."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            tokens = self.tokenizer(["a photo"]).to(self.device)
            probe = self.model.encode_text(tokens)
        self._dim = int(probe.shape[-1])

        # The fallback handles synthetic observations that have no `rgb`.
        # Built lazily so we don't pay SVD cost when the user never needs it.
        self._fallback = ontology_fallback

    # ----- Public API ----------------------------------------------------
    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            feat = self.model.encode_text(tokens)
        v = feat[0].detach().cpu().float().numpy()
        return _normalize(v.astype(np.float32))

    def encode_image(self, observation: Dict[str, Any]) -> np.ndarray:
        rgb = observation.get("rgb")
        if rgb is None:
            # No real image — defer to the deterministic ontology encoder so
            # SemanticHomeWorld observations still produce comparable vectors.
            return self._get_fallback().encode_image(observation)
        # rgb is HxWx3 uint8 (or float in [0,1]); normalise via preprocess.
        from PIL import Image
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(x)
        v = feat[0].detach().cpu().float().numpy()
        return _normalize(v.astype(np.float32))

    # ----- Internal ------------------------------------------------------
    def _get_fallback(self) -> OntologyVLMEncoder:
        if self._fallback is None:
            # Use the OpenCLIP model's dim so embeddings are interchangeable
            # within the same EpisodicMemory instance.
            self._fallback = OntologyVLMEncoder(embed_dim=self._dim)
        return self._fallback
