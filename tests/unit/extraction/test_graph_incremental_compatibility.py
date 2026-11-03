"""Opt-in descriptors accompany dev edits without weakening read validation."""
from unittest.mock import Mock

import pytest

from core.errors import ToolCallProtocolError
from extraction import Extractor
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools, parse_tool_call
from tests.unit.extraction.test_incremental_edit_flow import profile_read, profile_update
from tests.unit.extraction.test_relation_proposals import PROPOSAL


@pytest.mark.parametrize('name',['edit_memory','append_memory'])
def test_incremental_graph_descriptor_keeps_read_bound_identity_and_version(name):
    ctx,_,_,uri,node=profile_read('reader')
    inputs={**profile_update(name,uri),'update_activation':'Nora lives in Utrecht and has updated work details.'}
    candidate=parse_tool_call(name,inputs,SchemaRegistry(),read_nodes={uri:node},ctx=ctx,
                              activation_enabled=True)[2]
    assert candidate.target_uri==uri and candidate.expected_version==1
    assert candidate.metadata['update_activation']==inputs['update_activation']
    assert 'update_activation' in inputs  # caller input is never mutated
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call(name,{**inputs,'expected_version':2},SchemaRegistry(),
                        read_nodes={uri:node},ctx=ctx,activation_enabled=True)
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call(name,{**inputs,'owner_scope':'other'},SchemaRegistry(),
                        read_nodes={uri:node},ctx=ctx,activation_enabled=True)


def test_graph_incremental_tools_require_descriptor_only_when_enabled():
    for enabled in (False,True):
        tools=build_extraction_tools(SchemaRegistry(),activation_enabled=enabled)
        for tool in tools:
            if tool['name'] in {'edit_memory','append_memory'}:
                assert ('update_activation' in tool['input_schema']['required']) is enabled
                assert ('update_activation' in tool['input_schema']['properties']) is enabled


def test_lazy_native_relation_is_recorded_without_creating_a_memory():
    extractor=Extractor(Mock(),activation_enabled=True)
    loop=ExtractionReActLoop(Mock(),Mock(),SchemaRegistry(),Mock(),
                            extractor._prompt_manager,activation_enabled=True)
    assert loop._execute_tool('extract_relation',PROPOSAL,Mock())=={'status':'recorded'}


def test_graph_eager_prompt_keeps_relationship_guidance_and_dev_update_contract():
    graph=Extractor(Mock(),activation_enabled=True)
    prompt=graph._build_prompt('English','Source conversation')
    assert graph._prompt_manager.render('extraction_graph','lazy_tools_instruction') in prompt
    assert graph._update_instruction in prompt
    assert 'Source conversation' in prompt
