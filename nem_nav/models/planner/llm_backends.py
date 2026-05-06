"""Local-LLM callables for ``LLMPlanner``.

Each backend is exposed as a plain ``Callable[[str], str]`` so the planner
itself stays decoupled from any specific runtime. The factory
``build_llm_fn(backend, **kwargs)`` selects the implementation.

Two backends are provided:

* ``"llama_cpp"`` — wraps `llama_cpp.Llama` over a GGUF checkpoint
  (CPU/GPU). The default temperature is 0 to match the spec's
  determinism requirement.
* ``"echo"``     — returns a canned string. Used by tests and by
  ``--llm_backend echo`` for smoke runs without external deps.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional


def build_llm_fn(backend: str, **kwargs) -> Callable[[str], str]:
    backend = backend.lower()
    if backend == "echo":
        return _echo_fn(kwargs.get("canned"))
    if backend == "llama_cpp":
        return _llama_cpp_fn(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r}")


# ---------------------------------------------------------------------- echo

def _echo_fn(canned: Optional[Dict[str, Any]] = None) -> Callable[[str], str]:
    """Return a callable that ignores its prompt and replies with JSON.

    If ``canned`` is None, returns a permissive JSON plan that the
    deterministic fallback would also accept.
    """
    if canned is None:
        canned = {
            "target_room": None,
            "confidence": 0.0,
            "rationale": "echo backend",
            "bias_direction": None,
        }
    payload = json.dumps(canned)

    def _fn(prompt: str) -> str:  # noqa: ARG001
        return payload

    return _fn


# ----------------------------------------------------------------- llama_cpp

def _llama_cpp_fn(
    model_path: Optional[str] = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int = 0,
) -> Callable[[str], str]:
    """Build a llama-cpp-python backed callable.

    ``model_path`` defaults to ``$NEM_NAV_LLM_PATH`` when not provided. Raises
    ``RuntimeError`` if neither is set, or if the GGUF file is missing.
    """
    try:
        from llama_cpp import Llama
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "llama-cpp-python is not installed. Run `pip install -e .[llm]`."
        ) from e

    path = model_path or os.environ.get("NEM_NAV_LLM_PATH")
    if not path:
        raise RuntimeError(
            "No GGUF model path provided. Pass --llm_model_path PATH or set "
            "the NEM_NAV_LLM_PATH environment variable."
        )
    if not os.path.exists(path):
        raise RuntimeError(f"GGUF model file not found: {path}")

    llm = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
        verbose=False,
    )

    def _fn(prompt: str) -> str:
        out = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["```", "\n\n"],
        )
        # llama-cpp returns the standard chat-completion JSON shape.
        try:
            return out["choices"][0]["text"]
        except Exception:  # pragma: no cover
            return ""

    return _fn
