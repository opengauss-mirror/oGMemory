"""Search failures and corrected retries over the real MCP transport."""
import anyio

from tests.unit.server.test_mcp_code_edit_transport import _stdio_session


def test_stdio_search_rejects_invalid_arguments_and_accepts_retry(tmp_path):
    (tmp_path / "config.py").write_text("def parse_config(value):\n    return value\n")
    async def exercise():
        async with _stdio_session(tmp_path) as client:
            valid = {"query": "parse_config", "scope": {"paths": ["config.py"]}}
            tools = await client.list_tools()
            schema = next(t.input_schema for t in tools.tools if t.name == "ogm_search_code")
            assert schema["additionalProperties"] is False
            result = await client.call_tool("ogm_search_code", valid)
            assert not result.is_error
            for arguments, field in [
                ({}, "query"),
                ({**valid, "query": ""}, "query"),
                ({**valid, "query": " \t\n\u3000"}, "query"),
                ({**valid, "query": "\u001c\t\u001d\u001e\u001f\u3000"}, "query"),
                ({**valid, "unexpected_parameter": True}, "unexpected_parameter"),
                ({**valid, "budget": {"result_limit": 0}}, "budget.result_limit"),
                ({**valid, "budget": {"result_limit": 1.5}}, "budget.result_limit"),
                ({**valid, "scope": {"unknown": True}}, "scope.unknown"),
                ({**valid, "budget": {"unknown": True}}, "budget.unknown"),
            ]:
                result = await client.call_tool("ogm_search_code", arguments)
                text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
                assert result.is_error
                assert field in text
                assert "internal_error" not in text and "# hit 1" not in text
            result = await client.call_tool("ogm_search_code", {**valid, "query": "\u001c parse_config \u001f"})
            text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
            assert not result.is_error
            assert "# hit_count: 1" in text and "def parse_config(value):" in text
    anyio.run(exercise)


def test_stdio_zero_match_absolute_path_and_successful_retry(tmp_path):
    (tmp_path / "forms.py").write_text('class AccountForm:\n    help_text = "../password/"\n')
    (tmp_path / "admin.py").write_text('to_field = "id"\n')
    (tmp_path / "__init__.py").write_text("")
    async def exercise():
        async with _stdio_session(tmp_path) as client:
            tools = await client.list_tools()
            search_tool = next(t for t in tools.tools if t.name == "ogm_search_code")
            # Validate against the schema actually exposed over MCP.
            from jsonschema import validate
            def text(result):
                return "\n".join(b.text for b in result.content if hasattr(b, "text"))
            result = await client.call_tool("ogm_search_code", {
                "query": "AccountForm", "scope": {"grep_terms": ["AccountForm", "to_field"]}})
            assert not result.is_error
            assert "no_same_file_grep_match" in text(result)
            assert "literal AND" in text(result)
            assert "# hit 1" not in text(result)
            for path in [str(tmp_path), "../outside", "C:/private/repo", "ogmemory://code/repo"]:
                result = await client.call_tool("ogm_search_code", {
                    "query": "AccountForm", "scope": {"paths": [path]}})
                assert result.is_error
                assert "invalid_search_scope" in text(result)
                assert "workspace-relative" in text(result)
                assert str(tmp_path) not in text(result)
            result = await client.call_tool("ogm_search_code", {
                "query": "AccountForm", "scope": {"paths": ["repo/forms.py"]}})
            assert not result.is_error
            assert "no_files_in_scope" in text(result)
            assert "without the root slug" in text(result)
            assert "root_slugs" not in text(result)
            retry = {"query": "AccountForm", "scope": {
                "paths": ["forms.py"], "grep_terms": ["AccountForm"]}}
            validate(retry, search_tool.input_schema)
            result = await client.call_tool("ogm_search_code", retry)
            assert not result.is_error
            assert "status: ok" in text(result)
            assert "repo/forms.py" in text(result)
            assert "../password/" in text(result)
    anyio.run(exercise)
