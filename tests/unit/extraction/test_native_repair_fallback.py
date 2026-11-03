from unittest.mock import Mock

import pytest

from core.errors import ToolCallProtocolError
from extraction.tools import Extractor
from tests.unit.extraction.test_native_validation_recovery import ctx, operation, prepared


def test_overlong_repair_uses_one_full_source_fallback():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(abstract="a" * 230)]
    llm.complete_json.side_effect = [
        {"repairs": [{"operation_index": 0, "abstract": "b" * 220}]},
        {"operations": [operation()]},
    ]
    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), "archive")
    assert len(result) == 1 and result[0].abstract == operation()["input"]["abstract"]
    assert llm.complete_json.call_count == 2
    assert prepared().prompt in llm.complete_json.call_args.args[0]


def test_invalid_target_is_regenerated_not_guessed_or_dropped():
    llm = Mock()
    llm.complete_with_tools.return_value = [operation(), operation("bad", target_uri="", expected_version=1)]
    llm.complete_json.return_value = {"operations": [operation(), operation("second")]}
    result = Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), "archive")
    assert len(result) == 2
    llm.complete_json.assert_called_once()


def test_invalid_fallback_remains_rejected_without_more_attempts():
    llm = Mock()
    bad = operation(target_uri="", expected_version=1)
    llm.complete_with_tools.return_value = [bad]
    llm.complete_json.return_value = {"operations": [bad]}
    with pytest.raises(ToolCallProtocolError):
        Extractor(llm)._structure_prepared_span_dual(prepared(), ctx(), "archive")
    llm.complete_json.assert_called_once()
