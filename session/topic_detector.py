# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Lightweight session topic detection."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from session.models import SessionMessage


@dataclass
class TopicDetection:
    label: str
    confidence: float
    message_count: int
    changed: bool = False


class TopicDetector:
    def __init__(self, embedder: Any | None = None, similarity_threshold: float = 0.80):
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold

    def detect(
        self,
        messages: list[SessionMessage],
        previous: TopicDetection | None = None,
    ) -> TopicDetection:
        user_messages = [m for m in messages if m.role == "user" and m.content.strip()]
        if not user_messages:
            result = TopicDetection(label="general", confidence=0.0, message_count=0)
        elif self._embedder is not None and len(user_messages) > 1:
            result = self._detect_with_embeddings(user_messages)
        else:
            result = self._detect_with_keywords(user_messages)
        if previous is not None:
            result.changed = previous.label != result.label
        return result

    def _detect_with_keywords(self, messages: list[SessionMessage]) -> TopicDetection:
        label_counts: dict[str, int] = {
            "debugging": 0,
            "planning": 0,
            "code_review": 0,
        }
        label_scores: dict[str, int] = {
            "debugging": 0,
            "planning": 0,
            "code_review": 0,
        }
        for message in messages:
            label, score = self._classify_text(message.content)
            if label in label_counts and score > 0:
                label_counts[label] += 1
                label_scores[label] += score

        label = self._pick_label(label_counts, label_scores)
        if label == "general":
            generic_topic = self._detect_generic_keyword_topic(messages)
            if generic_topic is not None:
                return generic_topic
            return TopicDetection(label="general", confidence=0.1, message_count=len(messages))
        message_count = label_counts[label]
        confidence = max(0.1, min(1.0, message_count / max(1, len(messages))))
        return TopicDetection(label=label, confidence=confidence, message_count=message_count)

    def _detect_with_embeddings(self, messages: list[SessionMessage]) -> TopicDetection:
        texts = [message.content for message in messages]
        try:
            embeddings = self._embedder.embed_texts(texts)
        except Exception:
            return self._detect_with_keywords(messages)

        if not isinstance(embeddings, list) or len(embeddings) != len(messages):
            return self._detect_with_keywords(messages)
        valid_embeddings = [
            vector for vector in embeddings
            if isinstance(vector, list) and vector
        ]
        if len(valid_embeddings) != len(messages):
            return self._detect_with_keywords(messages)
        dims = len(valid_embeddings[0])
        if any(len(vector) != dims for vector in valid_embeddings):
            return self._detect_with_keywords(messages)

        clusters: list[list[int]] = []
        centroids: list[list[float]] = []
        for idx, vector in enumerate(embeddings):
            if not isinstance(vector, list) or not vector:
                continue
            best_idx = -1
            best_score = -1.0
            for cluster_idx, centroid in enumerate(centroids):
                score = self._cosine_similarity(vector, centroid)
                if score > best_score:
                    best_idx = cluster_idx
                    best_score = score
            if best_idx >= 0 and best_score >= self._similarity_threshold:
                clusters[best_idx].append(idx)
                centroids[best_idx] = self._centroid([
                    embeddings[i] for i in clusters[best_idx]
                ])
            else:
                clusters.append([idx])
                centroids.append(vector)

        if not clusters:
            return self._detect_with_keywords(messages)

        best_cluster = max(clusters, key=lambda c: (len(c), max(c)))
        cluster_messages = [messages[i] for i in best_cluster]
        label = self._label_cluster(cluster_messages)
        confidence = max(0.1, min(1.0, len(best_cluster) / max(1, len(messages))))
        return TopicDetection(
            label=label,
            confidence=confidence,
            message_count=len(best_cluster),
        )

    def _label_cluster(self, messages: list[SessionMessage]) -> str:
        keyword_topic = self._detect_with_keywords(messages)
        if keyword_topic.label != "general":
            return keyword_topic.label

        counts = self._generic_keyword_message_counts(messages)
        if not counts:
            return "general"
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _detect_generic_keyword_topic(
        self,
        messages: list[SessionMessage],
    ) -> TopicDetection | None:
        counts = self._generic_keyword_message_counts(messages)
        if not counts:
            return None
        label, message_count = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        if message_count < 2:
            return None
        confidence = max(0.1, min(1.0, message_count / max(1, len(messages))))
        return TopicDetection(label=label, confidence=confidence, message_count=message_count)

    @staticmethod
    def _generic_keyword_message_counts(messages: list[SessionMessage]) -> dict[str, int]:
        stopwords = {
            "the", "and", "for", "with", "this", "that", "please", "can",
            "you", "how", "what", "why", "when", "from", "into", "about",
        }
        counts: dict[str, int] = {}
        for message in messages:
            message_words: set[str] = set()
            for raw in message.content.lower().replace("_", " ").split():
                word = "".join(ch for ch in raw if ch.isalnum())
                if len(word) < 4 or word in stopwords:
                    continue
                message_words.add(word)
            for word in message_words:
                counts[word] = counts.get(word, 0) + 1
        return counts

    def _classify_text(self, text: str) -> tuple[str, int]:
        lowered = (text or "").lower()
        labels = [
            ("debugging", ("debug", "traceback", "failing", "failure", "bug", "fix")),
            ("planning", ("plan", "prd", "requirements", "implementation plan")),
            ("code_review", ("review", "risk", "regression", "finding")),
        ]
        best_label = "general"
        best_score = 0
        for label, keywords in labels:
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label, best_score

    @staticmethod
    def _pick_label(label_counts: dict[str, int], label_scores: dict[str, int]) -> str:
        best_label = "general"
        best_count = 0
        best_score = 0
        for label in ("debugging", "planning", "code_review"):
            count = label_counts.get(label, 0)
            score = label_scores.get(label, 0)
            if count > best_count or (count == best_count and score > best_score):
                best_label = label
                best_count = count
                best_score = score
        return best_label

    @staticmethod
    def _centroid(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dims = len(vectors[0])
        return [
            sum(vector[idx] for vector in vectors) / len(vectors)
            for idx in range(dims)
        ]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)
