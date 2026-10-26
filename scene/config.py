"""Versioned scene configuration with atomic hot reload."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneConfig:
    """Runtime limits and quality thresholds for one scene-analysis batch."""

    analysis_input_budget_tokens: int = 8192
    analysis_trigger_tokens: int = 4096
    max_pending_nodes: int = 32
    min_evidence_nodes: int = 3
    min_independent_sources: int = 2
    max_anchor_count: int = 4
    candidate_card_budget_tokens: int = 2048
    max_candidate_scenes: int = 8
    max_react_rounds: int = 2
    max_llm_calls_per_batch: int = 3
    max_new_scenes_per_batch: int = 1
    max_members_per_scene: int = 64
    max_active_scenes_per_space: int = 256
    neighbor_k: int = 8
    similarity_threshold: float = 0.72
    strong_pair_threshold: float = 0.86
    min_quality: float = 0.78
    query_threshold: float = 0.75
    lexical_enabled: bool = False
    scene_boost: float = 0.50
    direct_evidence_reserve: float = 0.90
    member_decay: float = 0.85
    identity_overlap: float = 0.45
    retry_limit: int = 3
    max_wait_seconds: int = 1800

    def validate(self) -> None:
        if not isinstance(self.lexical_enabled, bool):
            raise ValueError("lexical_enabled must be boolean")
        positive_ints = (
            "analysis_input_budget_tokens",
            "analysis_trigger_tokens",
            "max_pending_nodes",
            "min_evidence_nodes",
            "min_independent_sources",
            "max_anchor_count",
            "candidate_card_budget_tokens",
            "max_candidate_scenes",
            "max_react_rounds",
            "max_llm_calls_per_batch",
            "max_new_scenes_per_batch",
            "max_members_per_scene",
            "max_active_scenes_per_space",
            "neighbor_k",
            "retry_limit",
            "max_wait_seconds",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.analysis_trigger_tokens > self.analysis_input_budget_tokens:
            raise ValueError("analysis_trigger_tokens must not exceed analysis_input_budget_tokens")
        if self.min_evidence_nodes > self.max_members_per_scene:
            raise ValueError("min_evidence_nodes must not exceed max_members_per_scene")
        if self.max_llm_calls_per_batch < 2 or self.max_llm_calls_per_batch > 4:
            raise ValueError("max_llm_calls_per_batch must be in [2, 4]")
        for name in (
            "similarity_threshold",
            "strong_pair_threshold",
            "min_quality",
            "query_threshold",
            "scene_boost",
            "direct_evidence_reserve",
            "member_decay",
            "identity_overlap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number in [0, 1]")
        if self.strong_pair_threshold < self.similarity_threshold:
            raise ValueError("strong_pair_threshold must be >= similarity_threshold")

    def merged(self, overrides: Mapping[str, object]) -> SceneConfig:
        fields = set(self.__dataclass_fields__)
        unknown = set(overrides) - fields
        if unknown:
            raise ValueError(f"unknown scene configuration: {sorted(unknown)}")
        candidate = replace(self, **dict(overrides))
        candidate.validate()
        return candidate


@dataclass(frozen=True)
class SceneConfigSnapshot:
    config: SceneConfig
    revision: int
    source: str


class SceneConfigProvider(Protocol):
    def get_effective(self, account_id: str, owner_space: str) -> SceneConfigSnapshot: ...

    def publish(
        self,
        account_id: str,
        owner_space: str,
        expected_revision: int,
        overrides: Mapping[str, object],
        *,
        source: str = "owner_space",
    ) -> SceneConfigSnapshot: ...

    def rollback(
        self,
        account_id: str,
        owner_space: str,
        target_revision: int,
    ) -> SceneConfigSnapshot: ...


class ConfigRevisionError(RuntimeError):
    """Raised when a configuration update races with another writer."""


ConfigRevisionConflict = ConfigRevisionError


class InMemorySceneConfigProvider:
    """Reference provider used by adapters and tests.

    The provider models the production contract: updates are CAS-protected,
    snapshots are immutable, and listeners receive the complete new snapshot.
    """

    def __init__(self, default: SceneConfig | None = None) -> None:
        self._default = default or SceneConfig()
        self._default.validate()
        self._snapshots: dict[tuple[str, str], SceneConfigSnapshot] = {}
        self._history: dict[tuple[str, str], dict[int, SceneConfigSnapshot]] = {}
        self._listeners: list[Callable[[str, str, SceneConfigSnapshot], None]] = []
        self._lock = RLock()

    def get_effective(self, account_id: str, owner_space: str) -> SceneConfigSnapshot:
        key = (account_id, owner_space)
        with self._lock:
            return self._snapshots.get(
                key,
                SceneConfigSnapshot(self._default, 0, "system"),
            )

    def publish(
        self,
        account_id: str,
        owner_space: str,
        expected_revision: int,
        overrides: Mapping[str, object],
        *,
        source: str = "owner_space",
    ) -> SceneConfigSnapshot:
        key = (account_id, owner_space)
        with self._lock:
            current = self._snapshots.get(key, SceneConfigSnapshot(self._default, 0, "system"))
            if current.revision != expected_revision:
                raise ConfigRevisionConflict(
                    f"configuration revision changed: expected={expected_revision}, "
                    f"actual={current.revision}"
                )
            next_snapshot = SceneConfigSnapshot(
                current.config.merged(overrides), current.revision + 1, source
            )
            self._snapshots[key] = next_snapshot
            self._history.setdefault(key, {})[next_snapshot.revision] = next_snapshot
            listeners = tuple(self._listeners)
        self._notify(listeners, account_id, owner_space, next_snapshot)
        return next_snapshot

    def rollback(
        self,
        account_id: str,
        owner_space: str,
        target_revision: int,
    ) -> SceneConfigSnapshot:
        key = (account_id, owner_space)
        with self._lock:
            history = self._history.get(key, {})
            target = (
                SceneConfigSnapshot(self._default, 0, "system")
                if target_revision == 0 else history.get(target_revision)
            )
            if target is None:
                raise ValueError(f"unknown configuration revision: {target_revision}")
            current = self.get_effective(account_id, owner_space)
            next_snapshot = SceneConfigSnapshot(target.config, current.revision + 1, "rollback")
            self._snapshots[key] = next_snapshot
            self._history.setdefault(key, {})[next_snapshot.revision] = next_snapshot
            listeners = tuple(self._listeners)
        self._notify(listeners, account_id, owner_space, next_snapshot)
        return next_snapshot

    def _notify(
        self,
        listeners: tuple[Callable[[str, str, SceneConfigSnapshot], None], ...],
        account_id: str,
        owner_space: str,
        snapshot: SceneConfigSnapshot,
    ) -> None:
        for listener in listeners:
            try:
                listener(account_id, owner_space, snapshot)
            except Exception:
                logger.exception("Scene configuration notification failed at revision %s", snapshot.revision)

    def subscribe(self, listener: Callable[[str, str, SceneConfigSnapshot], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def export(self, account_id: str, owner_space: str) -> dict[str, object]:
        snapshot = self.get_effective(account_id, owner_space)
        return {
            "revision": snapshot.revision,
            "source": snapshot.source,
            "values": asdict(snapshot.config),
        }


class SceneConfigCache:
    """Process-local cache with atomic snapshot replacement."""

    def __init__(self, provider: SceneConfigProvider) -> None:
        self._provider = provider
        self._snapshots: dict[tuple[str, str], SceneConfigSnapshot] = {}
        self._lock = RLock()
        subscribe = getattr(provider, "subscribe", None)
        if callable(subscribe):
            subscribe(self._replace)

    def get(self, account_id: str, owner_space: str) -> SceneConfigSnapshot:
        key = (account_id, owner_space)
        latest = self._provider.get_effective(account_id, owner_space)
        with self._lock:
            current = self._snapshots.get(key)
            if current is None or latest.revision >= current.revision:
                self._snapshots[key] = latest
                return latest
            return current

    def _replace(self, account_id: str, owner_space: str, snapshot: SceneConfigSnapshot) -> None:
        key = (account_id, owner_space)
        with self._lock:
            current = self._snapshots.get(key)
            if current is None or snapshot.revision >= current.revision:
                self._snapshots[key] = snapshot
