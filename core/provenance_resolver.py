# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Provenance ID construction and parsing.

Provenance IDs trace any memory or relationship back to its original source.

Format:
    prov:{version}:{source_type}:{urlencoded(source_id)}:{urlencoded(detail)}

source_id and detail are URL-encoded so that `:` can serve as a safe delimiter.
This allows source_id to contain colons (e.g., memory URIs like ctx://...)
and detail to contain colons (e.g., field:subfield) without ambiguity.

For archive type, detail is a list of message IDs (joined/split by the resolver
internally). Other types use plain string detail.

Wire format examples:
    prov:1:archive:20260513_100000_a1b2:msg_a3f8%2Cmsg_b7c1
    prov:1:archive:20260513_100000_a1b2:
    prov:1:memory:ctx%3A%2F%2Facme%2Fusers%2Fmemories%2Fentities%2Frust:
    prov:1:dream:20260513_dream_001:

Human-readable form (for logs/debug only, never stored or parsed):
    prov:1:archive:20260513_100000_a1b2:msg_a3f8,msg_b7c1
    prov:1:memory:ctx://acme/users/memories/entities/rust:
"""

from __future__ import annotations

from urllib.parse import quote, unquote

VALID_SOURCE_TYPES = {"archive", "memory", "dream", "graph"}


class ProvenanceResolver:
    """Build and parse Provenance IDs with URL-encoded fields."""

    @staticmethod
    def validate_input(source_type: str, detail: str | list[str] = "") -> None:
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {source_type}. Must be one of {VALID_SOURCE_TYPES}"
            )
        if isinstance(detail, list) and source_type != "archive":
            raise ValueError(
                "List detail is only supported for archive source_type"
            )

    @staticmethod
    def build_id(
        source_type: str,
        source_id: str,
        detail: str | list[str] = "",
    ) -> str:
        """Build a provenance ID with URL-encoded source_id and detail.

        For archive type, detail can be a list of message IDs — the resolver
        joins them with comma internally. Other types accept plain string.
        """
        ProvenanceResolver.validate_input(source_type, detail)
        if isinstance(detail, list):
            detail = ",".join(detail)
        enc_id = quote(source_id, safe="")
        enc_detail = quote(detail, safe="")
        return f"prov:1:{source_type}:{enc_id}:{enc_detail}"

    @staticmethod
    def parse_id(pid: str) -> dict:
        """Parse a provenance ID back into its components.

        For archive type, detail is returned as a list of message IDs.
        Other types return detail as a plain string.
        """
        if not pid.startswith("prov:"):
            raise ValueError(f"Invalid provenance ID: {pid}")

        parts = pid.split(":")
        if len(parts) < 5:
            raise ValueError(f"Invalid provenance ID format: {pid}")

        version_str = parts[1]
        source_type = parts[2]
        # With encoding, `:` never appears inside source_id or detail,
        # so split(":") always yields exactly 5 parts.
        enc_source_id = parts[3]
        enc_detail = parts[4]

        ProvenanceResolver.validate_input(source_type)

        try:
            version = int(version_str)
        except ValueError:
            raise ValueError(f"Invalid version in provenance ID: {version_str}")

        detail_raw = unquote(enc_detail)

        # Archive detail is a comma-separated message ID list
        if source_type == "archive":
            detail: str | list[str] = detail_raw.split(",") if detail_raw else []
        else:
            detail = detail_raw

        return {
            "version": version,
            "source_type": source_type,
            "source_id": unquote(enc_source_id),
            "detail": detail,
        }

    @staticmethod
    def display_id(pid: str) -> str:
        """Return a human-readable provenance ID for logs and debugging.

        Decodes URL-encoded fields back to their original form so colons
        and other special characters are visible.  NOT suitable for storage
        or parsing — use the raw `pid` for that.
        """
        parsed = ProvenanceResolver.parse_id(pid)
        d = parsed["detail"]
        detail_str = ",".join(d) if isinstance(d, list) else d
        return (
            f"prov:{parsed['version']}:{parsed['source_type']}"
            f":{parsed['source_id']}:{detail_str}"
        )
