"""Mock LLM implementation for testing.

Returns pre-defined fixture data based on the prompt content.
Useful for unit tests and integration tests without real LLM calls.
"""


from core.interfaces import LLM


class MockLLM(LLM):
    """Mock LLM that returns fixture JSON responses.

    This mock analyzes the prompt for keywords and returns appropriate
    fixture data. Useful for testing extractors and planners.

    Example:
        llm = MockLLM()
        result = llm.complete_json("Extract profile", {...schema})
    """

    def __init__(self, fixtures: dict[str, dict] | None = None):
        """Initialize MockLLM with optional fixtures.

        Args:
            fixtures: Custom fixture dict mapping prompt keywords to JSON responses
        """
        self._fixtures = fixtures or {}
        self._call_count = 0
        self._last_prompt = ""  # Track last prompt for testing

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Complete a prompt and return JSON matching schema.

        Analyzes prompt for keywords and returns appropriate fixture data.

        Args:
            prompt: The input prompt (analyzed for keywords)
            schema: JSON schema for response (used to infer type)

        Returns:
            JSON dict matching the schema
        """
        self._call_count += 1
        prompt_lower = prompt.lower()

        # Schema-validated extraction fallback. Native tool extraction tests use
        # an empty operation list by default.
        if "operations" in schema.get("properties", {}):
            return {"operations": []}

        # Span identification (Phase 1 of two-phase extractor)
        if "span" in prompt_lower and "message ranges" in prompt_lower:
            return {"spans": [{"start": 0, "end": 0, "reason": "test span", "categories": ["profile"]}]}

        # Check custom fixtures first
        for keyword, fixture in self._fixtures.items():
            if keyword.lower() in prompt_lower:
                return fixture

        # Profile extraction
        if "profile" in prompt_lower:
            return {
                "candidates": [
                    {
                        "category": "profile",
                        "owner_scope": "user",
                        "routing_key": "profile",
                        "abstract": "User profile information",
                        "overview": "Name, preferences, and background",
                        "content": "User profile data extracted from conversation",
                        "confidence": 0.9,
                    }
                ]
            }

        # Preference extraction
        if "preference" in prompt_lower:
            return {
                "candidates": [
                    {
                        "category": "preference",
                        "owner_scope": "user",
                        "routing_key": "coffee",
                        "abstract": "User's coffee preferences",
                        "overview": "Likes dark roast, drinks 2-3 cups daily",
                        "content": "Prefers dark roast coffee, drinks 2-3 cups per day",
                        "confidence": 0.85,
                    }
                ]
            }

        # Entity extraction
        if "entity" in prompt_lower:
            return {
                "candidates": [
                    {
                        "category": "entity",
                        "owner_scope": "user",
                        "routing_key": "coffee_shop_downtown",
                        "abstract": "Downtown coffee shop",
                        "overview": "User's favorite coffee shop location",
                        "content": "Coffee shop downtown that user visits frequently",
                        "confidence": 0.8,
                    }
                ]
            }

        # Event extraction
        if "event" in prompt_lower:
            return {
                "candidates": [
                    {
                        "category": "event",
                        "owner_scope": "user",
                        "routing_key": "visit_20250315",
                        "abstract": "Coffee shop visit on March 15",
                        "overview": "User visited downtown coffee shop",
                        "content": "User visited the coffee shop on March 15, 2025",
                        "confidence": 0.9,
                    }
                ]
            }

        # Pattern/skill extraction (agent scope)
        if "pattern" in prompt_lower or "skill" in prompt_lower:
            return {
                "candidates": [
                    {
                        "category": "pattern" if "pattern" in prompt_lower else "skill",
                        "owner_scope": "agent",
                        "routing_key": "error_handling",
                        "abstract": "Common error handling pattern",
                        "overview": "Retry with exponential backoff",
                        "content": "When errors occur, implement retry with exponential backoff",
                        "confidence": 0.85,
                    }
                ]
            }

        # Default empty response
        return {"candidates": []}

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> list[dict]:
        """Mock implementation of tool-use for testing.

        Returns mock tool calls from _mock_tool_calls attribute if set,
        otherwise returns empty list (LLM decides no tools needed).

        Args:
            prompt: The input prompt (stored for test verification)
            tools: List of tool definitions (not used in mock)
            tool_choice: Tool choice setting (not used in mock)

        Returns:
            List of tool call results: [{"tool": str, "input": dict}, ...]
        """
        self._call_count += 1
        self._last_prompt = prompt  # Track for testing

        # Return mock tool calls if set via attribute
        return getattr(self, "_mock_tool_calls", [])

    def complete_with_tools_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[list[dict], str]:
        """Mock implementation of messages-based tool use."""
        self._call_count += 1
        self._last_messages = messages

        # Return mock tool calls if set via attribute
        mock_calls = getattr(self, "_mock_tool_calls", [])
        if mock_calls:
            return mock_calls, ""

        # Return mock content if set via attribute
        mock_content = getattr(self, "_mock_content", "")
        if mock_content:
            return [], mock_content

        return [], ""

    def complete_multimodal(self, messages: list[dict], schema: dict) -> dict:
        """Mock implementation of multimodal completion.

        Returns either an explicit ``_mock_multimodal_response`` if the
        test set it (preferred for assertions), or a deterministic
        canned response derived from the schema so ad-hoc tests still
        get something useful. Tracks the multimodal message list on
        ``_last_multimodal_messages`` so tests can verify what was sent.
        """
        self._call_count += 1
        self._last_messages = messages
        self._last_multimodal_messages = messages
        self._last_multimodal_schema = schema

        canned = getattr(self, "_mock_multimodal_response", None)
        if canned is not None:
            return canned

        # Build a deterministic stub from the schema so callers always
        # get a structurally valid response. Production-style smoke runs
        # can rely on this when they don't want to wire fixtures.
        result: dict = {}
        for prop, spec in (schema.get("properties") or {}).items():
            ptype = spec.get("type") if isinstance(spec, dict) else None
            if ptype == "string":
                # Pull the first text content part's text for a
                # recognisable canned value when available.
                text_hint = ""
                for m in messages:
                    content = m.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_hint = part.get("text", "")
                                break
                    if text_hint:
                        break
                result[prop] = (text_hint[:80] if text_hint else f"mock_{prop}")
            elif ptype == "array":
                result[prop] = []
            elif ptype == "object":
                result[prop] = {}
            elif ptype == "number":
                result[prop] = 0
            elif ptype == "integer":
                result[prop] = 0
            elif ptype == "boolean":
                result[prop] = True
            else:
                result[prop] = None
        return result

    @property
    def call_count(self) -> int:
        """Get the number of times complete_json was called."""
        return self._call_count

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0
