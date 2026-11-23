# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared identity and governance rules for generated Agent Skills."""

from __future__ import annotations

import hashlib
import re
from typing import Any

MAX_SKILL_NAME_LENGTH = 64
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_skill_publication(publication: dict) -> dict:
    """Return a normalized, bounded publication payload."""

    if not isinstance(publication, dict) or not publication:
        raise ValueError("publication must be a non-empty object")
    normalized = dict(publication)
    capability_key = normalized.get("capability_key")
    if capability_key is not None and (
        not isinstance(capability_key, str)
        or not capability_key
        or len(capability_key) > MAX_SKILL_NAME_LENGTH
    ):
        raise ValueError("publication capability_key must be a bounded string")
    publication_uris = normalized.get("evidence_uris", [])
    if not isinstance(publication_uris, list) or any(
        not isinstance(uri, str) or not uri for uri in publication_uris
    ):
        raise ValueError("publication evidence URIs must be a string list")
    values = normalized.get("example_evidence_keys", [])
    if not isinstance(values, list) or len(values) > 3:
        raise ValueError("publication example evidence keys must be a bounded list")
    allowed_uris = set(publication_uris)
    examples = []
    seen = set()
    for item in values:
        example_id = item.get("example_id") if isinstance(item, dict) else None
        example_hash = item.get("example_hash") if isinstance(item, dict) else None
        if (
            example_id not in {"primary", "branch-1", "branch-2"}
            or not isinstance(example_hash, str)
            or len(example_hash) != 64
            or any(char not in "0123456789abcdef" for char in example_hash)
        ):
            raise ValueError("invalid publication example evidence key")
        key = (example_id, example_hash)
        if key in seen:
            raise ValueError("duplicate publication example evidence key")
        evidence_uris = item.get("evidence_uris", publication_uris)
        if (
            not isinstance(evidence_uris, list)
            or len(evidence_uris) > len(publication_uris)
            or any(
                not isinstance(uri, str) or uri not in allowed_uris
                for uri in evidence_uris
            )
            or len(set(evidence_uris)) != len(evidence_uris)
        ):
            raise ValueError("invalid publication example evidence URIs")
        seen.add(key)
        examples.append({
            "example_id": example_id,
            "example_hash": example_hash,
            "evidence_uris": evidence_uris,
        })
    if "example_evidence_keys" in normalized:
        normalized["example_evidence_keys"] = examples
    return normalized


def normalize_skill_name(value: Any) -> str:
    """Return the canonical, bounded kebab-case artifact name."""

    normalized = (
        re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(value).lower()))
        .strip("-")[:MAX_SKILL_NAME_LENGTH]
        .rstrip("-")
    )
    # Truncation can otherwise leave a trailing separator, producing a value
    # that normalize_skill_name itself cannot pass to validate_skill_name.
    if normalized:
        validate_skill_name(normalized)
    return normalized


def validate_skill_name(name: str) -> str:
    """Validate a public skill identifier and return it unchanged."""

    if len(name) > MAX_SKILL_NAME_LENGTH or not _SKILL_NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")
    return name


def internal_skill_uri(account_id: str, agent_id: str, name: str) -> str:
    return (
        f"ctx://{account_id}/agents/{agent_id}/skills/generated/"
        f"{validate_skill_name(name)}"
    )


def legacy_internal_skill_uri(account_id: str, agent_id: str, name: str) -> str:
    return f"ctx://{account_id}/agents/{agent_id}/skills/{validate_skill_name(name)}"


def internal_managed_skill_uri(account_id: str, agent_id: str, name: str) -> str:
    from skills.management import validate_managed_skill_name

    return (
        f"ctx://{account_id}/agents/{agent_id}/skills/imported/"
        f"{validate_managed_skill_name(name)}"
    )


def canonical_skill_uri(name: str, source: str | None = None) -> str:
    if source not in {None, "generated", "imported"}:
        raise ValueError("source must be generated or imported")
    if source == "imported":
        from skills.management import validate_managed_skill_name

        validate_managed_skill_name(name)
        directory = source
    else:
        validate_skill_name(name)
        directory = "generated"
    return f"ogmemory://skills/{directory}/{name}"


def is_governed_skill(node: Any) -> bool:
    """Return whether a node belongs to the reviewed generated catalog."""

    metadata = getattr(node, "metadata", {}) or {}
    # Publication is an immutable event. A later configuration change must not
    # make an already-reviewed artifact disappear from MCP/CLI discovery.
    return (
        metadata.get("ogmemory_generated") is True
        and metadata.get("publication_status") == "published"
    )


def is_managed_skill(node: Any) -> bool:
    """Return whether a skill was explicitly installed through management APIs."""

    metadata = getattr(node, "metadata", {}) or {}
    return (
        metadata.get("ogmemory_managed") is True
        and metadata.get("publication_status") == "published"
    )


def is_published_skill(node: Any) -> bool:
    return is_governed_skill(node) or is_managed_skill(node)


def artifact_body_hash(content: str) -> str:
    """Hash the complete portable artifact; governance lives outside SKILL.md."""

    normalized = "\n".join(line.rstrip() for line in content.strip().splitlines()) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
