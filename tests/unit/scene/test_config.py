from concurrent.futures import ThreadPoolExecutor
from threading import Event, local

import pytest

from scene.config import InMemorySceneConfigProvider, SceneConfig, SceneConfigCache


def test_config_defaults_and_scoped_overrides():
    defaults = SceneConfig()
    defaults.validate()
    configured = defaults.merged({"lexical_enabled": True, "neighbor_k": 12, "max_llm_calls_per_batch": 4})
    configured.validate()
    assert configured.lexical_enabled is True
    assert configured.neighbor_k == 12
    assert configured.max_llm_calls_per_batch == 4
    assert defaults.max_llm_calls_per_batch == 3
    assert defaults.neighbor_k == 8


def test_config_publish_advances_revision_and_refreshes_cache():
    provider = InMemorySceneConfigProvider()
    cache = SceneConfigCache(provider)
    original = cache.get("acct", "user:alice")
    published = provider.publish(
        "acct", "user:alice", original.revision,
        {"max_candidate_scenes": 4}, source="ui",
    )
    assert published.revision == 1
    assert published.source == "ui"
    assert cache.get("acct", "user:alice") == published
    assert published.config.max_candidate_scenes == 4
    assert original.config.max_candidate_scenes == 8
    assert original.revision == 0
    exported = provider.export("acct", "user:alice")
    assert exported["values"]["max_candidate_scenes"] == 4
    assert exported["revision"] == 1


def test_config_owner_and_account_scopes_advance_independently():
    provider = InMemorySceneConfigProvider()
    for account, owner, neighbors in (
        ("acct", "user:alice", 4),
        ("acct", "user:bob", 6),
        ("other", "user:alice", 12),
    ):
        provider.publish(account, owner, 0, {"neighbor_k": neighbors})
    assert provider.get_effective("acct", "user:alice").config.neighbor_k == 4
    assert provider.get_effective("acct", "user:bob").config.neighbor_k == 6
    assert provider.get_effective("other", "user:alice").config.neighbor_k == 12


def test_config_rollback_publishes_new_revision_to_cache_and_listeners():
    provider = InMemorySceneConfigProvider()
    cache = SceneConfigCache(provider)
    notifications = []
    provider.subscribe(lambda *event: notifications.append(event))
    first = provider.publish("acct", "user:alice", 0, {"scene_boost": 0.3})
    second = provider.publish("acct", "user:alice", first.revision, {"scene_boost": 0.6})
    restored = provider.rollback("acct", "user:alice", first.revision)
    assert second.revision == 2
    assert restored.revision == 3
    assert restored.source == "rollback"
    assert restored.config.scene_boost == 0.3
    assert cache.get("acct", "user:alice") == restored
    assert notifications[-1] == ("acct", "user:alice", restored)


@pytest.mark.parametrize("default", [None, SceneConfig(neighbor_k=12, scene_boost=0.2)])
def test_first_update_rolls_back_to_initial_configuration(default):
    provider = InMemorySceneConfigProvider(default)
    cache = SceneConfigCache(provider)
    initial = cache.get("acct", "user:alice")
    updated = provider.publish("acct", "user:alice", initial.revision, {"neighbor_k": 16})
    restored = provider.rollback("acct", "user:alice", initial.revision)

    assert restored.config == initial.config
    assert restored.revision == updated.revision + 1
    assert restored.source == "rollback"
    assert cache.get("acct", "user:alice") == restored
    assert provider.rollback("acct", "user:alice", updated.revision).config == updated.config


@pytest.mark.parametrize("operation", ["publish", "rollback"])
def test_committed_update_reaches_subscribers_after_callback_error(operation, caplog):
    provider = InMemorySceneConfigProvider()
    first = provider.publish("acct", "user:alice", 0, {"neighbor_k": 12})
    second = provider.publish("acct", "user:alice", first.revision, {"neighbor_k": 16})

    def unavailable_subscriber(*_):
        raise RuntimeError("subscriber temporarily unavailable")

    provider.subscribe(unavailable_subscriber)
    cache = SceneConfigCache(provider)
    notifications = []
    provider.subscribe(lambda *event: notifications.append(event))
    if operation == "publish":
        committed = provider.publish("acct", "user:alice", second.revision, {"neighbor_k": 20})
        expected_neighbors = 20
    else:
        committed = provider.rollback("acct", "user:alice", first.revision)
        expected_neighbors = 12

    assert committed.revision == 3
    assert committed.config.neighbor_k == expected_neighbors
    assert provider.get_effective("acct", "user:alice") == committed
    assert cache.get("acct", "user:alice") == committed
    assert notifications == [("acct", "user:alice", committed)]
    assert any(record.exc_info for record in caplog.records)
    next_update = provider.publish("acct", "user:alice", committed.revision, {"neighbor_k": 24})
    assert next_update.revision == 4
    assert notifications[-1] == ("acct", "user:alice", next_update)


def test_cache_read_keeps_latest_revision_across_reordered_notifications(monkeypatch):
    provider = InMemorySceneConfigProvider()
    first_notification = Event()
    release_notification = Event()
    first_read = Event()
    release_read = Event()
    reader_state = local()

    def delay_first_notification(account, owner, snapshot):
        if snapshot.revision == 1:
            first_notification.set()
            assert release_notification.wait(10)

    provider.subscribe(delay_first_notification)
    cache = SceneConfigCache(provider)
    cache.get("acct", "user:alice")
    get_effective = provider.get_effective

    def delayed_read(account, owner):
        snapshot = get_effective(account, owner)
        if getattr(reader_state, "pause", False):
            first_read.set()
            assert release_read.wait(10)
        return snapshot

    def read_cached():
        reader_state.pause = True
        return cache.get("acct", "user:alice")

    monkeypatch.setattr(provider, "get_effective", delayed_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(provider.publish, "acct", "user:alice", 0, {"neighbor_k": 12})
        try:
            assert first_notification.wait(10)
            reader = executor.submit(read_cached)
            assert first_read.wait(10)
            newest = provider.publish("acct", "user:alice", 1, {"neighbor_k": 16})
            release_notification.set()
            assert writer.result(timeout=10).revision == 1
            release_read.set()
            assert reader.result(timeout=10) == newest
        finally:
            release_notification.set()
            release_read.set()
    assert cache.get("acct", "user:alice") == newest
