"""Jev protocol validation and an injectable, single-request HTTP client."""

import json
import math
import time
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from core.errors import OperationDeadlineExceededError
from core.operation_deadline import check_deadline, remaining_seconds

from .config import DecisionConfig


class DecisionError(ValueError):
    """Safe protocol/transport failure, without source text or credentials."""


@dataclass(frozen=True)
class DecisionBatch:
    answers: dict
    choices: dict
    raw: dict
    seconds: float


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_once(request, timeout):
    return build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


def _is_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _validate_questions(state, questions, kind):
    if not isinstance(state, (dict, str)) or not isinstance(questions, dict):
        raise DecisionError("Decision state/questions must be JSON state and a question mapping")
    for name, question in questions.items():
        if not isinstance(name, str) or not name or not isinstance(question, dict):
            raise DecisionError("Decision questions require names and objects")
        if question.get("type") != kind:
            raise DecisionError("Invalid decision question type")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise DecisionError("Decision instructions are required")
        criteria = question.get("criteria")
        if kind == "choice" and (
            not isinstance(criteria, dict)
            or not criteria
            or any(not isinstance(key, str) or not key for key in criteria)
        ):
            raise DecisionError("Choice requires named criteria")


def _parse_choice(answer, criteria):
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
        raise DecisionError("Decision response choice criteria do not match")
    if not all(_is_probability(value) for value in probabilities.values()):
        raise DecisionError("Invalid decision choice probability")
    total = math.fsum(probabilities.values())
    if not math.isclose(total, 1, abs_tol=0.02):
        raise DecisionError("Decision choice probabilities do not sum to one")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in probabilities:
        raise DecisionError("Invalid decision choice")
    if not math.isclose(probabilities[choice], max(probabilities.values()), abs_tol=1e-12):
        raise DecisionError("Decision choice is not a maximum probability")
    return {key: value / total for key, value in probabilities.items()}, choice


def _parse_response(raw, questions, kind):
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise DecisionError("Decision response must contain an answer mapping")
    if set(raw["answers"]) != set(questions):
        raise DecisionError("Decision response question identifiers do not match")
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        raise DecisionError("Invalid decision usage metadata")
    if (
        usage.get("truncated")
        or usage.get("truncated_questions")
        or usage.get("state_tokens_dropped", 0)
    ):
        raise DecisionError("Decision response reports truncated input")
    answers, choices = {}, {}
    for name, question in questions.items():
        answer = raw["answers"][name]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise DecisionError("Decision response answer type does not match")
        if kind == "choice":
            answers[name], choices[name] = _parse_choice(answer, question["criteria"])
        else:
            score = answer.get("noul")
            if not _is_probability(score):
                raise DecisionError("Invalid decision relevance probability")
            answers[name] = float(score)
    return answers, choices


class JevDecisionClient:
    def __init__(self, config: DecisionConfig, *, transport=None):
        self.config = config
        self._transport = transport if transport is not None else self._post
        base = config.base_url.rstrip("/")
        self.endpoint = (
            base
            if base.endswith("/v1/systemone")
            else base + ("/systemone" if base.endswith("/v1") else "/v1/systemone")
        )

    def _post(self, payload):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
            headers=headers,
            method="POST",
        )
        remaining = remaining_seconds()
        timeout = (
            self.config.timeout_seconds
            if remaining is None
            else min(self.config.timeout_seconds, remaining)
        )
        with _open_once(request, timeout=timeout) as response:
            return json.load(response)

    def _request(self, payload):
        check_deadline()
        started = time.perf_counter()
        try:
            raw = self._transport(payload)
        except OperationDeadlineExceededError:
            raise
        except Exception as exc:
            check_deadline()
            error = DecisionError("Decision request failed")
            error.error_type = type(exc).__name__
            root = getattr(exc, "reason", None)
            error.root_error_type = (
                type(root).__name__ if isinstance(root, BaseException) else error.error_type
            )
            error.status_code = exc.code if isinstance(exc, HTTPError) else None
            raise error from None
        check_deadline()
        return raw, time.perf_counter() - started

    def evaluate(self, state, questions):
        return self._evaluate(state, questions, "noul")

    def evaluate_choice(self, state, questions):
        return self._evaluate(state, questions, "choice")

    def _evaluate(self, state, questions, kind):
        _validate_questions(state, questions, kind)
        try:
            payload = json.loads(
                json.dumps(
                    {"model": self.config.model, "state": state, "questions": questions},
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, OverflowError):
            raise DecisionError("Decision input must be finite JSON") from None
        if not questions:
            return DecisionBatch({}, {}, {}, 0.0)
        raw, seconds = self._request(payload)
        answers, choices = _parse_response(raw, questions, kind)
        return DecisionBatch(answers, choices, raw, seconds)
