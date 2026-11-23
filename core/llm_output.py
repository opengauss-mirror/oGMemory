# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Normalize visible model output without accepting embedded reasoning."""

import re


def final_text(content: str | None) -> str:
    """Remove reasoning blocks; fail closed on an unfinished reasoning block.

    Some compatible endpoints return these blocks in content rather than in a
    separate reasoning field. A truncated block is not a final answer.
    """
    if not isinstance(content, str):
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", content,
                  flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE):
        return ""
    # A response can start mid-block; discard the prefix through its close.
    if re.search(r"</think\s*>", text, flags=re.IGNORECASE):
        text = re.split(r"</think\s*>", text, flags=re.IGNORECASE)[-1]
    return text.strip()
