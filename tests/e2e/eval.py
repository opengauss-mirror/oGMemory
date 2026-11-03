"""Memory evaluation: ingest, QA, judge and statistics for supported datasets and hosts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from judge_parallel import PROTOCOL_VERSION


def load_dataset(
    path: str | Path,
    dataset_format: str = "auto",
    tail: str | None = None,
) -> list[dict]:
    """Load source data as ordered sessions and questions."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON array")
    if dataset_format == "auto":
        first = data[0]
        if isinstance(first, dict) and "conversation" in first:
            dataset_format = "locomo"
        elif isinstance(first, dict) and "sessions" in first:
            dataset_format = "memory"
        else:
            raise ValueError("Dataset sample must include conversation or sessions")
    if dataset_format == "locomo":
        return normalize_dataset(data, tail)
    if dataset_format == "memory":
        return _normalize_memory_data(data, tail)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def build_session_messages(
    normalized_item: dict,
    session_range: tuple[int, int] | None = None,
    tail: str | None = None,
) -> list[dict]:
    """Select sessions by inclusive index and preserve their source text."""
    if "conversation" in normalized_item:
        return build_locomo_session_messages(
            normalized_item, session_range, tail="[]" if tail is None else tail
        )
    sessions = []
    for session in normalized_item["sessions"]:
        index = session["meta"]["session_index"]
        if session_range and not session_range[0] <= index <= session_range[1]:
            continue
        sessions.append({
            "message": _append_tail(session["message"], tail),
            "meta": dict(session["meta"]),
        })
    return sessions


def parse_session_range(s: str) -> tuple[int, int]:
    """Parse '1-4' or '3' into an inclusive range."""
    if "-" in s:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    n = int(s)
    return n, n


def _append_tail(message: str, tail: str | None) -> str:
    return f"{message}\n\n{tail}" if tail else message


def _normalize_memory_data(data: list[dict], tail: str | None) -> list[dict]:
    samples = []
    sample_ids = set()
    for sample_index, item in enumerate(data):
        prefix = f"Invalid memory item at index {sample_index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}: expected an object")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{prefix}: sample_id must be a non-empty string")
        if sample_id in sample_ids:
            raise ValueError(f"{prefix}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        source_sessions = item.get("sessions")
        if not isinstance(source_sessions, list) or not source_sessions:
            raise ValueError(f"{prefix}: sessions must be a non-empty array")
        sessions, session_ids = [], set()
        for index, session in enumerate(source_sessions, start=1):
            session_prefix = f"{prefix}, session at index {index}"
            if not isinstance(session, dict):
                raise ValueError(f"{session_prefix}: expected an object")
            session_id = session.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError(f"{session_prefix}: session_id must be a non-empty string")
            if session_id in session_ids:
                raise ValueError(f"{session_prefix}: duplicate session_id {session_id!r}")
            session_ids.add(session_id)
            message = session.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"{session_prefix}: message must be a non-empty string")
            meta = {
                "sample_id": sample_id,
                "session_key": session_id,
                "session_index": index,
            }
            if "date_time" in session:
                if not isinstance(session["date_time"], str):
                    raise ValueError(f"{session_prefix}: date_time must be a string")
                meta["date_time"] = session["date_time"]
            sessions.append({"message": _append_tail(message, tail), "meta": meta})
        source_questions = item.get("questions")
        if not isinstance(source_questions, list) or not source_questions:
            raise ValueError(f"{prefix}: questions must be a non-empty array")
        questions = []
        for index, qa in enumerate(source_questions):
            qa_prefix = f"{prefix}, question at index {index}"
            if not isinstance(qa, dict):
                raise ValueError(f"{qa_prefix}: expected an object")
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise ValueError(f"{qa_prefix}: question must be a non-empty string")
            answer = qa.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                raise ValueError(f"{qa_prefix}: answer must be a string or number")
            if isinstance(answer, str) and not answer.strip():
                raise ValueError(f"{qa_prefix}: answer must not be empty")
            questions.append({
                "question": qa["question"],
                "answer": answer,
                "category": qa.get("category", ""),
                "evidence": qa.get("evidence", []),
            })
        samples.append({"sample_id": sample_id, "sessions": sessions, "questions": questions})
    return samples


def format_locomo_message(msg: dict) -> str:
    """Format a single LoCoMo message into a natural chat-style string.

    Output format:
        Speaker: text here
        image_url: caption
    """
    speaker = msg.get("speaker", "unknown")
    text = msg.get("text", "")
    line = f"{speaker}: {text}"

    img_urls = msg.get("img_url", [])
    if isinstance(img_urls, str):
        img_urls = [img_urls]
    blip = msg.get("blip_caption", "")

    if img_urls:
        for url in img_urls:
            caption = f": {blip}" if blip else ""
            line += f"\n{url}{caption}"
    elif blip:
        line += f"\n({blip})"

    return line


def load_locomo_data(
    path: str,
    sample_index: int | None = None,
) -> list[dict]:
    """Load LoCoMo JSON and optionally filter to one sample."""
    data = validate_dataset(Path(path))
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(data):
            raise ValueError(f"sample index {sample_index} out of range")
        return [data[sample_index]]
    return data


def build_locomo_session_messages(
    item: dict,
    session_range: tuple[int, int] | None = None,
    tail: str = "[]",
) -> list[dict]:
    """Build bundled session messages for one LoCoMo sample.

    Returns list of dicts with keys: message, meta.
    """
    conv = item["conversation"]
    speakers = f"{conv['speaker_a']} & {conv['speaker_b']}"

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions = []
    for sk in session_keys:
        sess_num = int(sk.split("_")[1])
        if session_range:
            lo, hi = session_range
            if sess_num < lo or sess_num > hi:
                continue

        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")

        parts = [f"[group chat conversation: {date_time}]"]
        for msg in conv[sk]:
            parts.append(format_locomo_message(msg))
        if tail:
            parts.append(tail)
        combined = "\n\n".join(parts)

        sessions.append(
            {
                "message": combined,
                "meta": {
                    "sample_id": item["sample_id"],
                    "session_key": sk,
                    "session_index": sess_num,
                    "date_time": date_time,
                    "speakers": speakers,
                },
            }
        )

    return sessions


def normalize_dataset(data: list[dict], tail: str | None = None) -> list[dict]:
    """Normalize LoCoMO sessions and evaluated questions."""
    return [
        {
            "sample_id": item["sample_id"],
            "sessions": build_locomo_session_messages(item, tail="[]" if tail is None else tail),
            "questions": [
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "category": qa.get("category", ""),
                    "evidence": qa.get("evidence", []),
                }
                for qa in item["qa"]
                if str(qa.get("category", "")) != "5"
            ],
        }
        for item in _validate_data(data)
    ]


def validate_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"LoCoMo dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _validate_data(data)


def _validate_data(data: list[dict]) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise ValueError("LoCoMo dataset must be a non-empty JSON array")
    sample_ids: set[str] = set()
    for index, item in enumerate(data):
        prefix = f"Invalid LoCoMo item at index {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}: expected an object")
        if not isinstance(item.get("sample_id"), str) or not item["sample_id"].strip():
            raise ValueError(f"{prefix}: sample_id must be a non-empty string")
        if item["sample_id"] in sample_ids:
            raise ValueError(f"{prefix}: duplicate sample_id {item['sample_id']!r}")
        sample_ids.add(item["sample_id"])
        conversation = item.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"{prefix}: conversation must be an object")
        for speaker_key in ("speaker_a", "speaker_b"):
            if not isinstance(conversation.get(speaker_key), str):
                raise ValueError(f"{prefix}: conversation.{speaker_key} must be a string")
        session_count = 0
        for session_key, messages in conversation.items():
            if not session_key.startswith("session_") or session_key.endswith("_date_time"):
                continue
            if not session_key.removeprefix("session_").isdigit() or not isinstance(messages, list):
                raise ValueError(f"{prefix}: conversation.{session_key} must be a message array")
            for message_index, message in enumerate(messages):
                message_prefix = (
                    f"{prefix}: conversation.{session_key} message at index {message_index}"
                )
                if not isinstance(message, dict):
                    raise ValueError(f"{message_prefix} must be an object")
                for field in ("speaker", "text"):
                    if not isinstance(message.get(field), str) or not message[field].strip():
                        raise ValueError(f"{message_prefix}: {field} must be a non-empty string")
            session_count += 1
        if not session_count:
            raise ValueError(f"{prefix}: conversation must include at least one session_N array")
        qas = item.get("qa")
        if not isinstance(qas, list):
            raise ValueError(f"{prefix}: qa must be an array")
        evaluated_qa_count = 0
        for qa_index, qa in enumerate(qas):
            qa_prefix = f"{prefix}, QA at index {qa_index}"
            if not isinstance(qa, dict):
                raise ValueError(f"{qa_prefix}: expected an object")
            if str(qa.get("category", "")) == "5":
                continue
            evaluated_qa_count += 1
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise ValueError(f"{qa_prefix}: question must be a non-empty string")
            answer = qa.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                raise ValueError(f"{qa_prefix}: answer must be a string or number")
            if isinstance(answer, str) and not answer.strip():
                raise ValueError(f"{qa_prefix}: answer must not be empty")
        if not evaluated_qa_count:
            raise ValueError(f"{prefix}: qa must include at least one evaluated question")
    return data


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def plugin_timeout_ms(value: str) -> int:
    result = int(value)
    if not 100 <= result <= 60000:
        raise argparse.ArgumentTypeError("must be between 100 and 60000 milliseconds")
    return result


def build_parser(
    agent: str | None = None, *, transports: tuple[str, ...] | None = None
) -> argparse.ArgumentParser:
    default_agent = agent or "openclaw"
    parser = argparse.ArgumentParser(description="Memory evaluation: ingest, QA, judge and statistics")
    parser.set_defaults(agent=default_agent)
    parser.add_argument("mode", choices=("ingest", "qa", "judge", "stat"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--dataset-format", choices=("auto", "locomo", "memory"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample", type=int, default=None)
    selection.add_argument("--all-samples", action="store_true")
    parser.add_argument("--sessions", help="Ingest session range, for example 1-4")
    parser.add_argument("--count", type=positive_int)
    parser.add_argument("--tail", default=None)
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--clear-ingest-record", action="store_true")
    parser.add_argument("--ingest-record", type=Path, default=Path(".ingest_record.json"))
    parser.add_argument("--verify-writes", action="store_true")
    parser.add_argument("--read-only-qa", action="store_true")
    parser.add_argument("--ogmemory-url", default="http://127.0.0.1:8090")
    parser.add_argument("--account-id", default="locomo-agent-eval")
    parser.add_argument("--user", "--user-id", dest="user_id", default="eval-1")
    parser.add_argument("--memory-agent-id", default=default_agent)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:8])
    parser.add_argument("--ogmemory-api-key-env", default="OGMEM_LOCOMO_OGMEMORY_API_KEY")
    parser.add_argument("--llm-api-key-env", default="OGMEM_API_KEY")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--claude-base-url")
    parser.add_argument("--claude-auth-token", action="store_true")
    parser.add_argument("--judge-api-key-env", default="OGMEM_API_KEY")
    parser.add_argument(
        "--api-base", "--judge-base-url", dest="judge_base_url", default="https://api.openai.com/v1"
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-profile", choices=("locomo",), default="locomo")
    parser.add_argument("--judge-disable-thinking", action="store_true")
    parser.add_argument("--judge-parallel", type=positive_int, default=1)
    parser.add_argument("--judge-max-attempts", type=positive_int, default=1)
    parser.add_argument(
        "-p", "--parallel", "--qa-parallel", dest="qa_parallel", type=positive_int, default=2
    )
    parser.add_argument("--agent-timeout", type=int, default=1200)
    parser.add_argument("--disable-host-tools", action="store_true")
    parser.add_argument("--compose-timeout-ms", type=plugin_timeout_ms, default=60000)
    parser.add_argument("--trace-timeout-ms", type=plugin_timeout_ms, default=60000)
    parser.add_argument("--agent-attempts", type=positive_int, default=3)
    parser.add_argument("--rate-limit-retry-delay", type=int, default=30)
    parser.add_argument("--stop-on-rate-limit", action="store_true")
    parser.add_argument("--write-wait-timeout", type=int, default=1200)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--openclaw-command", default="openclaw")
    parser.add_argument("--openclaw-port", type=int, default=18791)
    parser.add_argument(
        "--openclaw-provider-api",
        choices=("openai-responses", "openai-completions"),
        default="openai-responses",
    )
    parser.add_argument("--opencode-command", default="opencode")
    parser.add_argument("--claude-command", default="claude")
    parser.add_argument("--hermes-command", default="hermes")
    parser.add_argument("--dsh-command", default="dsh")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--qwenpaw-command", default="qwenpaw")
    parser.add_argument("--qwenpaw-port", type=int, default=18891)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--agent",
        choices=(agent,) if agent else ("openclaw", "opencode", "claude-code", "hermes", "dsh", "codex", "qwenpaw"),
        default=default_agent,
    )
    parser.add_argument(
        "--transport",
        choices=transports or ("ce", "plugin", "mcp", "gateway"),
        default=transports[0] if transports else None,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18789")
    parser.add_argument("--gateway-state-dir", type=Path)
    parser.add_argument("--token", default=os.environ.get("OPENCLAW_GATEWAY_TOKEN", ""))
    parser.add_argument("--agent-id", default="main")
    parser.add_argument("--judge-api-mode", choices=("chat", "responses"), default="chat")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def row_usage(row: dict) -> dict[str, int]:
    """Read token counters from nested plugin or flat gateway CSV rows."""
    usage = row.get("usage") or row
    if isinstance(usage, str):
        usage = json.loads(usage)
    return {key: int(usage.get(key, 0) or 0) for key in (
        "input_tokens", "output_tokens", "total_tokens", "cacheRead", "cacheWrite"
    )}


def qa_prompt(question: str) -> str:
    return f"Answer the question directly: {question}"


def session_key(sample_id: str, label: str, args) -> str:
    identity = (args.account_id, args.user_id, args.memory_agent_id, args.run_id, sample_id)
    namespace = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]
    if label.startswith("ingest-"):
        return f"ingest-{namespace}"
    return f"qa-{namespace}-q{label.removeprefix('qa-')}"


def memory_headers(args) -> dict[str, str]:
    return {
        "X-API-Key": args.ogmemory_api_key,
        "X-Account-ID": args.account_id,
        "X-User-ID": args.user_id,
        "X-Agent-ID": args.memory_agent_id,
    }


def memory_call(args, method: str, sid: str, **extra) -> dict:
    response = requests.post(
        f"{args.ogmemory_url}/api/v1/call/{method}",
        headers=memory_headers(args),
        json={
            "sessionId": sid,
            "accountId": args.account_id,
            "userId": args.user_id,
            "agentId": args.memory_agent_id,
            **extra,
        },
        timeout=args.write_wait_timeout,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("ok") is False or result.get("retryable") is False or result.get("status") == "failed":
        raise RuntimeError(f"{method} rejected for {sid}: {result}")
    return result


def wait_idle(args, sid: str) -> dict:
    deadline = time.monotonic() + args.write_wait_timeout
    while time.monotonic() < deadline:
        result = memory_call(
            args, "wait_until_idle", sid, timeoutSeconds=5, drainOutbox=False, waitOutbox=True
        )
        if result.get("idle") is True:
            outbox = result.get("outbox", {})
            if (outbox.get("error") or outbox.get("supported") is False
                    or any(outbox.get(key, 0) != 0 for key in ("pending", "processing", "total"))):
                raise RuntimeError(f"outbox completion unavailable: {outbox}")
            return result
        time.sleep(1)
    raise TimeoutError(f"source settlement timed out: {sid}")


def flush_source(args, sid: str) -> dict:
    started = time.monotonic()
    wait_idle(args, sid)
    state = source_state(args, sid)
    if not (state.get("message_count", 0) or state.get("commit_count", 0)
            or source_state(args, sid, context=True).get("archive_count", 0)):
        raise RuntimeError(f"source write evidence unavailable: {sid}")
    if state.get("message_count", 0):
        compacted = memory_call(args, "compact", sid)
        if compacted.get("compacted") is not True:
            raise RuntimeError(f"source compaction incomplete: {sid}: {compacted}")
        wait_idle(args, sid)
    deadline = started + args.write_wait_timeout
    while time.monotonic() < deadline:
        result = memory_call(args, "dispose", sid)
        if result.get("reason") == "below_token_threshold":
            raise RuntimeError(f"source extraction below completion threshold: {sid}")
        if result.get("disposed") is True:
            break
        time.sleep(1)
    else:
        raise TimeoutError(f"source flush timed out: {sid}")
    idle = wait_idle(args, sid)
    state = source_state(args, sid)
    archives = source_state(args, sid, context=True)
    if state.get("ok") is not True or state.get("pending_tokens") != 0:
        raise RuntimeError(f"source buffer remains after dispose: {sid}: {state}")
    if not archives.get("archive_count", 0):
        raise RuntimeError(f"source archive unavailable after dispose: {sid}")
    return {
        "session_id": sid,
        "idle": idle,
        "pending_tokens": 0,
        "archive_count": archives["archive_count"],
        "flush_seconds": time.monotonic() - started,
    }


def source_state(args, sid: str, *, context=False) -> dict:
    response = requests.get(
        f"{args.ogmemory_url}/api/v1/sessions/{quote(sid, safe='')}"
        + ("/context" if context else ""),
        params={
            "accountId": args.account_id,
            "userId": args.user_id,
            "agentId": args.memory_agent_id,
        },
        headers=memory_headers(args),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def create_adapter(args):
    common = sys.modules[__name__]
    if args.transport == "ce":
        from run_openclaw_plugin_locomo import CeAdapter

        return CeAdapter(args, common)
    if args.transport == "plugin":
        from agent_plugin_locomo import PluginAdapter

        return PluginAdapter(args, common)
    from run_mcp_locomo import GatewayAdapter, McpAdapter

    adapter = McpAdapter if args.transport == "mcp" else GatewayAdapter
    return adapter(args, common)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def questions(samples, count=None):
    for index, item in samples:
        for qi, qa in enumerate(item["questions"][:count], start=1):
            yield index, item["sample_id"], qi, qa


def summarize(rows, timings):
    categories, tokens = {}, {}
    for row in rows:
        bucket = categories.setdefault(str(row.get("category", "")), {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += row.get("is_correct") == "CORRECT"
        for key, value in row_usage(row).items():
            tokens[key] = tokens.get(key, 0) + value
    for bucket in categories.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]
    correct = sum(row.get("is_correct") == "CORRECT" for row in rows)
    judged = sum(row.get("is_correct") in ("CORRECT", "WRONG") for row in rows)
    return {
        "questions": len(rows),
        "judged": judged,
        "unjudged": len(rows) - judged,
        "correct": correct,
        "wrong": judged - correct,
        "empty_responses": sum(not row.get("response", "").strip() for row in rows),
        "accuracy": correct / len(rows) if rows else 0,
        "by_category": categories,
        "tokens": tokens,
        "timing_seconds": timings,
    }


def successful_answer(text):
    text = text.strip()
    return bool(text) and not text.startswith("[ERROR]")


def answer_questions(args, samples, adapter, rows, checkpoint):
    completed = {(row["sample_id"], int(row["qi"])) for row in rows if successful_answer(row["response"])}

    def answer(entry):
        index, sample_id, qi, qa = entry
        key = session_key(sample_id, f"qa-{qi}", args)
        started = time.monotonic()
        usage = {}
        for attempt in range(getattr(args, "agent_attempts", 3)):
            try:
                response, usage = adapter.answer(qa_prompt(qa["question"]), key)
                if not successful_answer(response):
                    raise RuntimeError("Agent returned an unsuccessful QA response")
                break
            except Exception as error:
                response = f"[ERROR] {error}"
                print(f"{key}: attempt {attempt + 1}: {error}", file=sys.stderr)
        return {
            "sample_id": sample_id,
            "sample_idx": index,
            "qi": qi,
            "question": qa["question"],
            "expected": str(qa["answer"]),
            "response": response,
            "category": str(qa.get("category", "")),
            "evidence": json.dumps(qa.get("evidence", []), ensure_ascii=False),
            **row_usage(usage),
            "qa_seconds": time.monotonic() - started,
        }

    for sample in samples:
        pending = [q for q in questions([sample], args.count) if (q[1], q[2]) not in completed]
        with ThreadPoolExecutor(max_workers=args.qa_parallel) as pool:
            futures = []
            for entry in pending:
                futures.append(pool.submit(answer, entry))
                if getattr(args, "delay", 0):
                    time.sleep(args.delay)
            for future in as_completed(futures):
                row = future.result()
                rows[:] = [prior for prior in rows
                           if (prior["sample_id"], int(prior["qi"])) != (row["sample_id"], row["qi"])]
                rows.append(row)
                rows.sort(key=lambda row: (int(row["sample_idx"]), int(row["qi"])))
                checkpoint(rows)
                if not successful_answer(row["response"]):
                    for pending_future in futures:
                        pending_future.cancel()
                    raise RuntimeError(f"QA failed for {row['sample_id']}/q{row['qi']}")
    return rows


def judge(args, input_path, output_path):
    environment = os.environ.copy()
    environment["OGMEM_API_KEY"] = os.environ.get(args.judge_api_key_env, "") or args.llm_api_key
    command = [
        sys.executable,
        str(Path(__file__).with_name("judge_parallel.py")),
        str(input_path),
        "--output",
        str(output_path),
        "--api-base",
        args.judge_base_url,
        "--judge-model",
        args.judge_model,
        "--judge-profile",
        getattr(args, "judge_profile", "locomo"),
        "--api-mode",
        args.judge_api_mode,
        "--parallel",
        str(args.judge_parallel),
        "--max-attempts",
        str(args.judge_max_attempts),
        "--retry-delay",
        str(args.rate_limit_retry_delay),
    ]
    if args.judge_disable_thinking:
        command.append("--disable-thinking")
    if args.stop_on_rate_limit:
        command.append("--fail-fast-rate-limit")
    subprocess.run(command, env=environment, cwd=Path(__file__).parent, check=True)


def selected_samples(args):
    data = load_dataset(
        args.dataset, getattr(args, "dataset_format", "auto"), tail=args.tail
    )
    if args.sample is None or args.all_samples:
        return list(enumerate(data, start=1))
    if not 0 <= args.sample < len(data):
        raise ValueError(f"sample index out of range: {args.sample}")
    return [(1, data[args.sample])]


def ingest(args, samples, adapter):
    record_path = args.ingest_record
    record = (
        json.loads(record_path.read_text())
        if record_path.exists() and not args.clear_ingest_record
        else {}
    )
    if args.clear_ingest_record:
        write_json(record_path, record)
    session_range = parse_session_range(args.sessions) if args.sessions else None
    prior_path = args.output_dir / "ingest.json"
    rows = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else []
    selected_source_ids = set()
    selected_record_keys = set()
    adapter.open("ingest")
    for index, item in samples:
        sid = item["sample_id"]
        for session in build_session_messages(item, session_range):
            name = session["meta"]["session_key"]
            key = session_key(sid, "ingest-1", args)
            source_sha256 = hashlib.sha256(session["message"].encode("utf-8")).hexdigest()
            record_key = f"{args.transport}:{args.agent}:{args.account_id}:{args.memory_agent_id}:{args.agent_id}:{args.user_id}:{sid}:{name}"
            selected_record_keys.add(record_key)
            prior = record.get(record_key, {})
            if not args.force_ingest and (
                prior.get("success") or (args.verify_writes and prior.get("write_verified"))
            ):
                if args.verify_writes and (
                    not prior.get("write_verified") or not prior.get("source_ids")
                    or prior.get("source_sha256") != source_sha256 or prior.get("session_key") != key
                ):
                    raise RuntimeError(
                        f"Saved ingest lacks matching write verification for {sid}/{name}; "
                        "use a fresh identity and --force-ingest"
                    )
                selected_source_ids.update(prior.get("source_ids", []))
                continue
            before = time.monotonic()
            source_ids, usage = [], {}
            failure = None
            record.pop(record_key, None)
            try:
                reply, usage, source_ids = adapter.ingest(session["message"], key)
                if not reply.strip():
                    raise RuntimeError("Agent returned empty ingest response")
                if args.verify_writes and not source_ids:
                    raise RuntimeError("Verified ingest returned no source sessions")
                record[record_key] = {
                    "success": not args.verify_writes, "session_key": key, "source_ids": source_ids,
                    "write_verified": args.verify_writes,
                    "source_sha256": source_sha256,
                }
                selected_source_ids.update(source_ids)
            except Exception as error:
                failure = error
                reply = f"[ERROR] {error}"
                print(f"{sid}/{name}: {error}", file=sys.stderr)
            write_json(record_path, record)
            row = {
                "sample_id": sid,
                "sample_idx": index,
                "session": name,
                "session_key": key,
                "user": args.user_id,
                "source_ids": source_ids,
                "reply": reply,
                "usage": usage,
                "write_seconds": time.monotonic() - before,
            }
            previous = next(
                (i for i, prior in enumerate(rows)
                 if prior["sample_id"] == sid and prior["session"] == name), None
            )
            if previous is None:
                rows.append(row)
            else:
                rows[previous] = row
            write_json(args.output_dir / "ingest.json", rows)
            print(f"ingest {sid}/{name}", flush=True)
            if failure is not None:
                raise RuntimeError(f"Ingest failed for {sid}/{name}") from failure
    if args.verify_writes:
        if not selected_source_ids:
            raise RuntimeError("Write verification requires at least one source session")
        checks = [adapter.settle(sid) for sid in sorted(selected_source_ids)]
        write_json(args.output_dir / "source-completion.json", {"sources": checks})
        for key in selected_record_keys:
            record[key]["success"] = True
        write_json(record_path, record)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(f"[{r['sample_id']}/{r['session']}] {r['reply']}" for r in rows) + "\n",
            encoding="utf-8",
        )
        write_json(Path(str(args.output) + ".json"), rows)
    return rows


def run(args, adapter_factory=None, judge_fn=judge):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "judge":
        output = args.output or args.dataset.with_name(args.dataset.stem + "_judged.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        judge_fn(args, args.dataset, output)
        return 0
    if args.mode == "stat":
        if not args.dataset.is_file():
            raise FileNotFoundError(args.dataset)
        result = summarize(read_csv(args.dataset), {})
        write_json(args.output or args.output_dir / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    samples = selected_samples(args)
    adapter = (adapter_factory or create_adapter)(args)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"ogmemory_api_key", "llm_api_key", "token"}
    }
    settings.update(
        protocol=PROTOCOL_VERSION,
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        adapter=adapter.description,
    )
    manifest_path = args.output_dir / f"{args.mode}-manifest.json"
    qa_path = Path(str(args.output) + ".csv") if args.output else args.output_dir / "qa_results.csv"
    if args.mode == "qa" and read_csv(qa_path):
        if not manifest_path.is_file():
            raise ValueError("Existing QA results require their original qa-manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        fields = (
            "protocol", "dataset_sha256", "dataset_format", "tail", "transport", "agent",
            "account_id", "user_id", "memory_agent_id", "agent_id", "run_id",
            "ogmemory_url", "base_url", "llm_model", "llm_base_url", "claude_base_url",
            "disable_host_tools", "read_only_qa", "openclaw_provider_api",
            "compose_timeout_ms", "trace_timeout_ms",
            "sample", "all_samples", "count", "verify_writes", "adapter",
        )
        changed = [key for key in fields if previous.get(key) != settings.get(key)]
        if changed:
            raise ValueError(f"QA resume settings changed: {', '.join(changed)}; use a new output")
    write_json(manifest_path, settings)
    started = time.monotonic()
    try:
        if args.preflight_only:
            adapter.open(args.mode)
        elif args.mode == "ingest":
            ingest(args, samples, adapter)
        else:
            adapter.open("qa")
            rows = answer_questions(
                args, samples, adapter, read_csv(qa_path), lambda value: write_csv(qa_path, value)
            )
            result = {
                "questions": len(rows),
                "errors": sum(r["response"].startswith("[ERROR]") for r in rows),
                "empty_responses": sum(not r["response"].strip() for r in rows),
            }
            if args.output:
                write_json(args.output, result)
            if result["errors"] or result["empty_responses"]:
                raise RuntimeError("QA contains unsuccessful responses; see the saved CSV")
        write_json(
            args.output_dir / f"{args.mode}-timing.json", {"seconds": time.monotonic() - started}
        )
        return 0
    finally:
        adapter.close()


def main(argv=None, *, agent=None, transports=None):
    parser = build_parser(agent, transports=transports)
    args = parser.parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve() if args.output else None
    args.output_dir = (
        args.output_dir
        or (
            args.output.parent / (args.output.stem + "-artifacts")
            if args.output
            else Path(__file__).parent / "result"
        )
    ).resolve()
    args.transport = args.transport or ("ce" if args.agent == "openclaw" else "plugin")
    if args.read_only_qa and args.transport not in ("plugin", "mcp"):
        parser.error("--read-only-qa requires --transport plugin or mcp")
    args.ogmemory_api_key = os.environ.get(args.ogmemory_api_key_env, "")
    args.llm_api_key = os.environ.get(args.llm_api_key_env, "")
    if args.transport in ("ce", "mcp", "gateway") and args.agent != "openclaw":
        raise ValueError("CE/MCP/gateway require OpenClaw")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
