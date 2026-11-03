"""Memory evaluation judge with explicit rubrics and resumable results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROTOCOL_VERSION = "memory-eval-v1"

LOCOMO_PROTOCOL_VERSION = "locomo-standard-v3"

JUDGE_SYSTEM = "You are an expert grader."

JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer

The point of the question is to ask about something one user should know about the other based on prior conversations.
Be generous with grading - as long as it touches on the same topic as the gold answer, count as CORRECT.
For time questions, accept same date/time even if format differs.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}"""

JUDGE_PROFILES = {"locomo": JUDGE_PROMPT}

thread_local = threading.local()
RESUME_FIELDS = ("sample_id", "qi", "question", "expected", "response")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def get_client(api_key: str, base_url: str):
    from openai import OpenAI

    client = getattr(thread_local, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)
        thread_local.client = client
    return client


def build_prompt(row: dict[str, str], profile: str = "locomo") -> str:
    return JUDGE_PROFILES[profile].format(
        question=row["question"],
        gold_answer=row["expected"],
        generated_answer=row["response"],
    )


def build_judge_signature(
    base_url: str, model: str, api_mode: str, disable_thinking: bool = False,
    profile: str = "locomo",
) -> str:
    payload = json.dumps(
        {
            "base_url": base_url.rstrip("/"),
            "model": model,
            "api_mode": api_mode,
            "prompt": JUDGE_PROFILES[profile],
            "protocol": (
                LOCOMO_PROTOCOL_VERSION
                if profile == "locomo" else PROTOCOL_VERSION
            ),
            "disable_thinking": disable_thinking,
            "system": JUDGE_SYSTEM,
            "temperature": 0,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def judge_row(
    index: int,
    row: dict[str, str],
    api_key: str,
    base_url: str,
    model: str,
    api_mode: str,
    max_attempts: int = 3,
    retry_delay: float = 2.0,
    stop_event: threading.Event | None = None,
    disable_thinking: bool = False,
    profile: str = "locomo",
) -> tuple[int, str, str, bool, bool]:
    if row.get("is_correct") in ("CORRECT", "WRONG"):
        return index, row["is_correct"], row.get("reasoning", ""), False, False

    max_attempts = max(1, max_attempts)
    reasoning = ""
    for attempt in range(1, max_attempts + 1):
        if stop_event is not None and stop_event.is_set():
            return index, "", "[CANCELLED] rate limit detected", True, False
        try:
            client = get_client(api_key, base_url)
            if api_mode == "responses":
                resp = client.responses.create(model=model, instructions=JUDGE_SYSTEM, input=build_prompt(row, profile), temperature=0, timeout=60)
                content = resp.output_text.strip()
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": build_prompt(row, profile)}],
                    temperature=0,
                    timeout=60,
                    **({"extra_body": {"thinking": {"type": "disabled"}}} if disable_thinking else {}),
                )
                content = resp.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"judge returned no JSON object: {content[:100]}")
            result = json.loads(content[start : end + 1])
            is_correct = str(result.get("label", result.get("is_correct", ""))).strip().upper()
            reasoning = result.get("reasoning", "")
            if is_correct not in ("CORRECT", "WRONG"):
                raise ValueError(f"judge returned invalid label: {is_correct!r}")
            return index, is_correct, reasoning, False, False
        except Exception as exc:
            reasoning = f"[ERROR attempt {attempt}/{max_attempts}] {exc}"
            rate_limited = (
                getattr(exc, "status_code", None) == 429
                or "429" in str(exc)
                or "rate limit" in str(exc).lower()
                or "too many requests" in str(exc).lower()
            )
            if rate_limited and stop_event is not None:
                stop_event.set()
                break
            if attempt < max_attempts and retry_delay > 0:
                if stop_event is None:
                    time.sleep(retry_delay)
                else:
                    stop_event.wait(retry_delay)
    return index, "", reasoning, True, rate_limited


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_existing(rows: list[dict[str, str]], output_path: str, signature: str) -> None:
    if not os.path.exists(output_path):
        return
    existing = load_rows(output_path)
    by_key = {
        tuple(row.get(field) for field in RESUME_FIELDS): row
        for row in existing
        if row.get("is_correct") in ("CORRECT", "WRONG") and row.get("judge_signature") == signature
    }
    for row in rows:
        prior = by_key.get(tuple(row.get(field) for field in RESUME_FIELDS))
        if prior:
            row["is_correct"] = prior.get("is_correct", "")
            row["reasoning"] = prior.get("reasoning", "")


def write_rows(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp_path = f"{path}.tmp"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-profile", choices=tuple(JUDGE_PROFILES), default="locomo")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--api-mode", choices=("chat", "responses"), default="chat")
    parser.add_argument("--api-key", default=os.environ.get("OGMEM_API_KEY", ""))
    parser.add_argument("-p", "--parallel", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=positive_int, default=50)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--fail-fast-rate-limit", action="store_true")
    args = parser.parse_args(argv)

    if args.disable_thinking and args.api_mode != "chat":
        parser.error("--disable-thinking requires chat completions")
    if not args.api_key:
        print("Error: missing API key", file=sys.stderr)
        return 1

    rows = load_rows(args.input)
    signature = build_judge_signature(
        args.api_base, args.judge_model, args.api_mode, args.disable_thinking, args.judge_profile
    )
    for row in rows:
        if row.get("judge_signature") != signature:
            row["is_correct"] = ""
            row["reasoning"] = ""
        row["judge_signature"] = signature
    merge_existing(rows, args.output, signature)
    fieldnames = list(rows[0].keys()) if rows else []
    if "is_correct" not in fieldnames:
        fieldnames.append("is_correct")
    if "reasoning" not in fieldnames:
        fieldnames.append("reasoning")
    if "judge_signature" not in fieldnames:
        fieldnames.append("judge_signature")

    total = len(rows)
    done = sum(1 for row in rows if row.get("is_correct") in ("CORRECT", "WRONG"))
    correct = sum(1 for row in rows if row.get("is_correct") == "CORRECT")
    wrong = sum(1 for row in rows if row.get("is_correct") == "WRONG")
    errors = 0
    processed = done

    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("is_correct") not in ("CORRECT", "WRONG")
    ]
    print(
        f"Judging {total} responses with {args.judge_model} using {args.parallel} workers...",
        file=sys.stderr,
    )
    print(f"  Resuming with {done}/{total} already judged", file=sys.stderr)

    stop_event = threading.Event() if args.fail_fast_rate_limit else None
    executor = ThreadPoolExecutor(max_workers=max(1, min(args.parallel, 40)))
    wait_for_workers = True
    try:
        futures = [
            executor.submit(
                judge_row,
                index,
                row,
                args.api_key,
                args.api_base,
                args.judge_model,
                args.api_mode,
                args.max_attempts,
                args.retry_delay,
                stop_event,
                args.disable_thinking,
                args.judge_profile,
            )
            for index, row in pending
        ]
        for future in as_completed(futures):
            index, is_correct, reasoning, had_error, rate_limited = future.result()
            rows[index]["is_correct"] = is_correct
            rows[index]["reasoning"] = reasoning
            processed += 1
            if had_error:
                errors += 1
            else:
                done += 1
                if is_correct == "CORRECT":
                    correct += 1
                else:
                    wrong += 1

            if processed % 10 == 0 or processed == total:
                accuracy = f"{correct / done:.1%}" if done else "N/A"
                print(
                    f"  [{processed}/{total}] accuracy={accuracy} "
                    f"(correct={correct}, wrong={wrong}, errors={errors})",
                    file=sys.stderr,
                )
            if processed % args.checkpoint_every == 0 or processed == total:
                write_rows(args.output, rows, fieldnames)
            if rate_limited and args.fail_fast_rate_limit:
                wait_for_workers = False
                for pending_future in futures:
                    pending_future.cancel()
                write_rows(args.output, rows, fieldnames)
                print("Rate limit detected; stopping judge.", file=sys.stderr)
                return 2
    finally:
        executor.shutdown(wait=wait_for_workers, cancel_futures=not wait_for_workers)

    write_rows(args.output, rows, fieldnames)
    print("\n=== Judge Results ===", file=sys.stderr)
    print(f"  Total:     {done}", file=sys.stderr)
    print(f"  Correct:   {correct}", file=sys.stderr)
    print(f"  Wrong:     {wrong}", file=sys.stderr)
    print(f"  Errors:    {errors}", file=sys.stderr)
    print(f"  Accuracy:  {correct / done:.1%}" if done else "  Accuracy: N/A", file=sys.stderr)
    print(f"  Output:    {args.output}", file=sys.stderr)
    return 1 if errors or done != total else 0


if __name__ == "__main__":
    raise SystemExit(main())
