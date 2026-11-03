"""Search failures and corrected retries over the real MCP transport."""
import anyio

from tests.unit.server.test_mcp_code_edit_transport import _stdio_session


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
