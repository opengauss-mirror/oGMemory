"""Single source of truth for M3's LLM provider dispatch.

The QA / episodic-memory / judge / smoke stages used to ship their own
copy of ``build_llm`` / ``_build_llm`` / ``build_judge_llm``. They
diverged in subtle ways — the QA builder supported an OpenAI-compatible
fallback for ``volcengine`` / ``deepseek`` / etc. (commit 57c68af0),
the episodic-memory builder raised ``unknown LLM backend`` on the same
input (fixed only ad-hoc), and the judge builder had its own
``OGMEM_MINIMAX_API_KEY`` fallback. Consolidating here so future
provider support lands in one place.

Three roles, all sharing the same dispatch:

* ``qa_llm`` — answer generation in ``m3.qa`` and ``m3.eval`` adapter.
  Configured via ``m3.qa_llm`` in ``deploy/ogmemory.yaml`` (merged on
  top of the top-level ``llm`` block).
* ``episodic_llm`` — multimodal clip summarisation in
  ``m3.episodic_memory``. Configured via ``m3.episodic_llm``.
* ``judge_llm`` — judge scoring in ``m3.eval``. Configured via
  ``m3.judge_llm``.

The provider name is a **label**, not a class. ``volcengine`` /
``deepseek`` / ``dashscope`` / ``vllm`` / ``ollama`` / custom are all
OpenAI-compatible Chat Completions endpoints; the
``(base_url, api_key, model)`` triple is what actually distinguishes
them. The dispatch is:

1. ``"mock"`` → ``MockLLM`` (no real network).
2. ``"minimax"`` → ``MiniMaxLLM`` (the project-specific client with
   baked-in defaults).
3. Anything else → ``OpenAILLM`` (works for every
   OpenAI-compatible endpoint as long as ``base_url`` + ``api_key``
   + ``model`` are set in YAML).

The CLI exposes ``--llm {mock,minimax,openai,volcengine,...}`` only for
documentation; the dispatch above ignores the value when the config supplies
a fully-resolved ``(base_url, api_key, model)`` triple.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Per-role default model — only used when neither CLI nor YAML supplies one.
# Operators normally pin this in ``deploy/ogmemory.yaml`` so the defaults
# below are belt-and-suspenders.
DEFAULT_MODELS: dict[str, str] = {
    "qa_llm": "MiniMax-M3",
    "episodic_llm": "MiniMax-M3",
    "judge_llm": "MiniMax-M3",
}

# Roles supported by ``build_m3_llm``. Adding a new role = adding a new
# ``m3.<role>_llm`` block in ``deploy/ogmemory.yaml`` and a default
# model entry above.
SUPPORTED_ROLES = ("qa_llm", "episodic_llm", "judge_llm")


def _env_api_key_fallback(role: str, explicit: str | None) -> str | None:
    """Pick up deploy-time env vars the operator may have set.

    Used only for the judge role today (other roles are configured in
    YAML and dispatched through the same path). The env-var fallback
    is intentionally narrow: it only kicks in for the keys this
    project has historically used, so a leaked ``OPENAI_API_KEY`` from
    a developer's shell doesn't silently override the bench config.
    """
    if explicit:
        return explicit
    if role == "judge_llm":
        return (
            os.environ.get("OGMEM_MINIMAX_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
    return None


def build_m3_llm(
    provider: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    config_path: str | None = None,
    config_role: str = "qa_llm",
    json_mode: bool = True,
) -> Any:
    """Build an M3 LLM from CLI overrides + ``deploy/ogmemory.yaml``.

    Resolution order (later items override earlier):

    1. ``m3.<config_role>`` block in the YAML (already merged on top of
       the top-level ``llm`` block by :func:`m3.config.load_m3_runtime_config`).
    2. CLI overrides via the explicit kwargs (``provider``, ``model``,
       ``base_url``, ``api_key``).
    3. Built-in default model for the role (see :data:`DEFAULT_MODELS`).

    Dispatch:

    * ``"mock"`` → :class:`MockLLM`.
    * ``"minimax"`` → :class:`MiniMaxLLM` (uses its own env-var-driven
      defaults if ``api_key`` is unset).
    * Anything else (``"openai"``, ``"volcengine"``, ``"deepseek"``,
      ``"dashscope"``, ``"vllm"``, ``"ollama"``, custom label) →
      :class:`OpenAILLM` against the resolved ``(base_url, api_key,
      model)``. **The provider label is informational; the triple
      decides the actual endpoint.**

    Raises:
        ValueError: if ``config_role`` isn't in :data:`SUPPORTED_ROLES`,
            or if the resolved provider requires ``api_key`` and none
            was supplied (CLI / YAML / env-var fallback).
    """
    if config_role not in SUPPORTED_ROLES:
        raise ValueError(
            f"unknown config_role {config_role!r}; expected one of "
            f"{SUPPORTED_ROLES}"
        )

    # Defer import to avoid pulling ogmemory.yaml readers into scripts
    # that don't need them (e.g. unit tests of unrelated modules).
    from m3.config import load_m3_runtime_config

    runtime = load_m3_runtime_config(config_path)
    configured = getattr(runtime, config_role)

    provider = provider or configured.provider
    model = model or configured.model
    base_url = base_url or configured.base_url
    api_key = api_key or configured.api_key
    max_tokens = max_tokens or configured.max_tokens

    if provider == "mock":
        from providers.llm import MockLLM
        return MockLLM()

    if provider == "minimax":
        from providers.llm import get_minimax_llm
        MiniMaxLLM = get_minimax_llm()
        return MiniMaxLLM(
            api_key=api_key,
            base_url=base_url,
            model=model or DEFAULT_MODELS[config_role],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            request_timeout_seconds=configured.request_timeout_seconds,
        )

    # OpenAI-compatible fallback. Covers openai / volcengine / deepseek /
    # dashscope / vllm / ollama / custom labels. The provider string is
    # just a label; the (base_url, api_key, model) triple decides the
    # endpoint. Mirrors the dispatch in commit 57c68af0 (qa.py) and the
    # follow-up fix in episodic_memory.py.
    api_key = _env_api_key_fallback(config_role, api_key)
    if not api_key:
        raise ValueError(
            f"{config_role}: api_key is required when provider={provider!r}. "
            f"Set m3.{config_role}.api_key in ogmemory.yaml or pass "
            f"api_key=... to the builder."
        )

    from providers.llm import get_openai_llm
    OpenAILLM, _ = get_openai_llm()
    return OpenAILLM(
        api_key=api_key,
        base_url=base_url,
        model=model or DEFAULT_MODELS[config_role],
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        request_timeout_seconds=configured.request_timeout_seconds,
    )


# --- Per-role convenience wrappers ---------------------------------------
#
# Each is a thin alias so call sites read naturally without an explicit
# ``config_role=`` argument. Backwards-compatible names (``build_llm``
# etc.) live in their respective modules and delegate here, so existing
# imports keep working.


def build_qa_llm(
    provider: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    config_path: str | None = None,
) -> Any:
    """Build the QA answer LLM. Reads ``m3.qa_llm`` from YAML."""
    return build_m3_llm(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        config_path=config_path,
        config_role="qa_llm",
    )


def build_episodic_llm(
    provider: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    config_path: str | None = None,
) -> Any:
    """Build the episodic-memory LLM. Reads ``m3.episodic_llm`` from YAML."""
    return build_m3_llm(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        config_path=config_path,
        config_role="episodic_llm",
    )


def build_judge_llm(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    config_path: str | None = None,
) -> Any:
    """Build the eval judge LLM. Reads ``m3.judge_llm`` from YAML.

    Note: ``temperature`` defaults to 0.0 (deterministic for scoring).
    Other roles default to 0.2.
    """
    return build_m3_llm(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        config_path=config_path,
        config_role="judge_llm",
    )
