# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""LLM providers for structured text generation.

Includes mock implementations for testing and production implementations.
"""

from core.interfaces import LLM

from .mock_llm import MockLLM

# Model constants (safe to import - no openai dependency)
GPT_4O_MINI = "gpt-4o-mini"
GPT_4O = "gpt-4o"

def get_openai_llm():
    """Lazy import OpenAILLM to avoid hard openai dependency."""
    from .openai_llm import CachedOpenAILLM, OpenAILLM
    return OpenAILLM, CachedOpenAILLM


__all__ = [
    "LLM",
    # Mock implementations
    "MockLLM",
    # Lazy loader for OpenAI
    "get_openai_llm",
    # Model constants
    "GPT_4O_MINI",
    "GPT_4O",
]
