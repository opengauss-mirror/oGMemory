from memorybot.files import _PageText, parse_text
from memorybot.templates import TemplateRegistry, summarize_csv


def test_csv_decimal_groups_and_exported_text():
    result = summarize_csv([{"text": "项目,金额\n青禾,10.10\n青禾,0.20\n山川,2.35\n"}])
    assert result["total"] == "12.65"
    assert "青禾,10.30" in result["csv"]
    assert result["rows"] == 3


def test_utf8_bom_markdown_and_html_main_text():
    parsed = parse_text(b"\xef\xbb\xbf" + "青禾\n完成。".encode(), "notes.md")
    assert parsed["text"] == "青禾\n完成。"
    assert parsed["parsed_ranges"] == [{"unit": "line", "start": 1, "end": 2}]
    page = _PageText()
    page.feed(
        "<html><head><title>青禾进展</title></head><body><h1>青禾</h1><p>完成接口。</p></body></html>"
    )
    assert "完成接口。" in "".join(page.parts)


def test_template_version_and_frozen_csv_executor():
    registry = TemplateRegistry()
    documents = [{"text": "项目,金额\n青禾,3\n", "mime_type": "text/csv"}]
    frozen = registry.validate("csv_summary", "", documents, "csv")
    assert len(frozen["revision"]) == 64
    assert registry.execute_frozen(frozen, documents)["total"] == "3"
    assert {item["id"] for item in registry.list()} == {
        "weekly_report",
        "handover",
        "handbook",
        "release_check",
        "csv_summary",
    }
