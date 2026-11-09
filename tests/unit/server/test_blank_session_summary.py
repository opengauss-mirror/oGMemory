"""An empty auxiliary summary must not become an invalid embedding job."""
from unittest.mock import Mock, patch

import pytest

from server.memory_service import MemoryService


@pytest.mark.parametrize("formatted", ["", " \n\t"])
def test_blank_summary_does_not_write_or_build_candidate(formatted):
    service = MemoryService.__new__(MemoryService)
    service.get_llm = Mock(return_value=Mock())
    writer = Mock()
    with patch("extraction.summary_generator.SessionSummaryGenerator") as factory:
        generator = factory.return_value
        generator.generate.return_value = {"topics": []}
        generator.format_summary.return_value = formatted
        service._generate_and_write_summary([{"role": "user", "content": "source"}],
            None, Mock(), "source-session", writer)
        generator.build_candidate.assert_not_called()
        writer.write_memory.assert_not_called()


def test_nonempty_summary_is_written_unchanged():
    service = MemoryService.__new__(MemoryService)
    service.get_llm = Mock(return_value=Mock())
    writer, ctx = Mock(), Mock()
    with patch("extraction.summary_generator.SessionSummaryGenerator") as factory:
        generator = factory.return_value
        generator.generate.return_value = {"topics": ["source"]}
        generator.format_summary.return_value = "Source summary"
        service._generate_and_write_summary([{"role": "user", "content": "source"}],
            None, ctx, "source-session", writer)
        assert generator.build_candidate.call_args.args[1] == "Source summary"
        writer.write_memory.assert_called_once_with(generator.build_candidate.return_value, ctx)
