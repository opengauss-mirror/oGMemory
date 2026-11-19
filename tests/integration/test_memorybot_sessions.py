from tests.integration.test_memorybot_http import client as client
from tests.integration.test_memorybot_http import headers
from tests.integration.test_memorybot_service import service as service


def create(http, title, key):
    response = http.post("/api/v1/bot/sessions", headers=headers(key), json={"title": title})
    assert response.status_code == 201
    return response.json["data"]


def test_session_rename_archive_restore_and_search(client):
    http, _, _ = client
    session = create(http, "本周进展", "create")
    path = f"/api/v1/bot/sessions/{session['session_id']}"
    changed = http.patch(
        path,
        headers=headers("rename"),
        json={"expected_revision": 1, "title": " 青禾_100% 进展 "},
    )
    assert changed.status_code == 200
    assert changed.json["data"]["title"] == "青禾_100% 进展"
    assert changed.json["data"]["revision"] == 2
    matched = http.get("/api/v1/bot/sessions?q=_100%25", headers=headers("search"))
    assert [item["session_id"] for item in matched.json["data"]["items"]] == [session["session_id"]]
    archived = http.patch(
        path,
        headers=headers("archive"),
        json={"expected_revision": 2, "archived": True},
    )
    assert archived.status_code == 200
    assert archived.json["data"]["archived"] is True
    replay = http.patch(
        path,
        headers=headers("archive"),
        json={"expected_revision": 2, "archived": True},
    )
    assert replay.json == archived.json
    listed = http.get("/api/v1/bot/sessions?archived=true", headers=headers("archives"))
    assert listed.json["data"]["items"][0]["session_id"] == session["session_id"]
    restored = http.patch(
        path,
        headers=headers("restore"),
        json={"expected_revision": 3, "archived": False},
    )
    assert restored.json["data"]["revision"] == 4
    visible = http.get("/api/v1/bot/sessions", headers=headers("active"))
    assert visible.json["data"]["items"][0]["session_id"] == session["session_id"]
    assert visible.json["data"]["items"][0]["archived"] is False


def test_recent_session_order_follows_conversation_activity(client):
    http, _, _ = client
    first = create(http, "先建的对话", "first")
    second = create(http, "后建的对话", "second")
    sent = http.post(
        f"/api/v1/bot/sessions/{first['session_id']}/tasks",
        headers=headers("message"),
        json={"text": "继续上次的工作"},
    )
    assert sent.status_code == 202
    latest = http.get("/api/v1/bot/sessions?limit=1", headers=headers("latest")).json["data"]
    assert latest["items"][0]["session_id"] == first["session_id"]
    older = http.get(
        "/api/v1/bot/sessions",
        query_string={"limit": 1, "cursor": latest["next_cursor"]},
        headers=headers("older"),
    ).json["data"]
    assert older["items"][0]["session_id"] == second["session_id"]


def test_recent_history_pages_preserve_chronological_display(client):
    http, _, _ = client
    session = create(http, "多轮对话", "history")
    path = f"/api/v1/bot/sessions/{session['session_id']}"
    for index in range(5):
        submitted = http.post(
            path + "/tasks", headers=headers(f"turn-{index}"), json={"text": f"第{index + 1}轮"}
        )
        assert submitted.status_code == 202
    cursor = None
    pages = []
    for _ in range(3):
        query = {"limit": 2, "history_order": "desc"}
        if cursor:
            query["cursor"] = cursor
        response = http.get(path, query_string=query, headers=headers("read"))
        assert response.status_code == 200
        page = response.json["data"]
        pages.append([task["seq"] for task in page["messages"]])
        cursor = page["next_cursor"]
    assert pages == [[4, 5], [2, 3], [1]]
    assert cursor is None
