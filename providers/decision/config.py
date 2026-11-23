# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Explicit opt-in configuration for the Jev-compatible decision protocol."""

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DecisionConfig:
    base_url: str
    model: str
    provider: str = "jev"
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if self.provider != "jev":
            raise ValueError("decision.provider must be jev")
        if not isinstance(self.base_url, str):
            raise ValueError("decision.base_url must be an HTTP(S) URL")
        try:
            parts = urlsplit(self.base_url)
            valid = (
                parts.scheme in ("http", "https")
                and bool(parts.hostname)
                and not parts.username
                and not parts.password
            )
            valid = valid and not parts.query and not parts.fragment
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                "decision.base_url must be an HTTP(S) URL without credentials/query/fragment"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("decision.model is required")
        if not isinstance(self.api_key, str):
            raise ValueError("decision.api_key must be a string")
        if (
            type(self.timeout_seconds) not in (float, int)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("decision.timeout_seconds must be positive and finite")

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict) or not value:
            raise ValueError("decision must be a non-empty mapping or null")
        try:
            return cls(**value)
        except TypeError:
            raise ValueError("decision has missing or unsupported configuration fields") from None
