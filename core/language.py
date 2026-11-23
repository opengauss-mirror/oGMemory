# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Language detection utility for ContextEngine.

Single implementation used across extraction, indexing, and compression.
Supports: zh-CN, ja, ko, ru, en (default).
"""

import re


def detect_language(text: str) -> str:
    """Detect language from text.

    Args:
        text: Text to analyze (can combine multiple strings with join).

    Returns:
        Language code: "zh-CN", "ja", "ko", "ru", or "en" (default).
    """
    if not text.strip():
        return "en"

    total = len(re.findall(r"\S", text))
    if total == 0:
        return "en"

    # Japanese: Hiragana/Katakana are unique to Japanese
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    # Chinese: CJK ideographs
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-CN"
    # Korean: need at least 2 characters and >=10% of text
    ko = len(re.findall(r"[\uac00-\ud7af]", text))
    if ko >= 2 and ko / total >= 0.10:
        return "ko"
    # Russian: need at least 2 characters and >=10% of text
    ru = len(re.findall(r"[\u0400-\u04ff]", text))
    if ru >= 2 and ru / total >= 0.10:
        return "ru"

    return "en"
