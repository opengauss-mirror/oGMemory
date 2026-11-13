from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event, Lock, get_ident, local
from unittest.mock import Mock

import pytest

from core.errors import LLMTransientError, ToolCallProtocolError
from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction import prompt_assembly
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prefetch import PrefetchResult
from extraction.prompts import PromptManager
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _Span

CTX = RequestContext('acct', 'user', 'agent', 'session', 'trace')
TIME = datetime(2026, 11, 5)
MESSAGES = [{'role': 'user', 'content': 'Project Beacon uses PostgreSQL.'}]
SHARED_TOOLS = {'get_relations', 'get_access_stats'}


def make_extractor(llm, *, mode='lazy', fs=None, **kwargs):
    registry = SchemaRegistry()
    return Extractor(llm, mode=mode, fs=Mock() if fs is None else fs, schema_registry=registry,
                     uri_resolver=URIResolver(registry), **kwargs)


def prepare(extractor):
    return extractor._prepare_span(_Span(0, 0, categories=['entity']), MESSAGES,
                                   'English', TIME, '', '', CTX)


@pytest.mark.parametrize('path', ['invalid', 'legacy', 'graph', 'unmatched_boundary'])
def test_full_lazy_preserves_runtime_suffix_in_provider_messages(path):
    class LegacyManager(PromptManager):
        def render(self, template_name, section, **kwargs):
            return super().render(template_name, section, **kwargs)

    llm = Mock()
    category = "entity" if path == "unmatched_boundary" else "unknown"
    llm.complete_json.return_value = {"categories": [category]}
    llm.complete_with_tools_messages.return_value = ([], "[]")
    manager = LegacyManager() if path == "legacy" else PromptManager()
    extractor = make_extractor(
        llm,
        node_processing_enabled=True,
        activation_enabled=path == "graph",
        prompt_manager=manager,
    )
    prepared = prepare(extractor)
    prepared.prompt += '\nCorrection marker: keep the original source facts.'
    if path == 'unmatched_boundary':
        prepared.span.prefetch_result = PrefetchResult(messages=['Retained prefetch snapshot'])
        prepared.prompt = 'Opaque input boundary\n' + prepared.prompt
    extractor._structure_prepared_span(prepared, CTX)
    request = llm.complete_with_tools_messages.call_args.kwargs
    tools = tuple(tool['name'] for tool in request['tools'])
    assert prepared.assembly_state.contract.scope == 'full'
    assert prepared.assembly_state.contract.tool_names == tools
    assert {"extract_profile", "read", "list", *SHARED_TOOLS} <= set(tools)
    assert "── profile (" in request["messages"][0]["content"]
    text = "\n".join(message["content"] for message in request["messages"])
    assert text.count("TOPIC_DRAFTS:") == 1
    assert text.count("Correction marker:") == 1
    assert MESSAGES[0]["content"] in text
    if path == "unmatched_boundary":
        assert "Opaque input boundary" in text
        assert text.count("Retained prefetch snapshot") == 1
        assert prepared.assembly_state.contract.reason == "unsupported_template"


def test_eager_recovery_does_not_replace_other_lazy_pass_tools(monkeypatch):
    llm = Mock()
    llm.complete_json.return_value = {"categories": ["entity"]}
    llm.complete_with_tools_messages.return_value = ([], "[]")
    extractor = make_extractor(llm)
    prepared = prepare(extractor)
    paused, released = Event(), Event()
    worker = local()
    original_assemble, original_run = prompt_assembly.assemble_prepared, ExtractionReActLoop.run

    def pause_after_assembly(*args, **kwargs):
        result = original_assemble(*args, **kwargs)
        if worker.waiting:
            paused.set()
            assert released.wait(5)
        return result

    def fail_one_lazy_pass(loop, *args, **kwargs):
        if not worker.waiting:
            raise ToolCallProtocolError('injected lazy output validation failure')
        return original_run(loop, *args, **kwargs)

    def native(waiting, temperature):
        worker.waiting = waiting
        return extractor._structure_prepared_span(prepared, CTX, temperature=temperature)

    monkeypatch.setattr(prompt_assembly, 'assemble_prepared', pause_after_assembly)
    monkeypatch.setattr(ExtractionReActLoop, 'run', fail_one_lazy_pass)
    eager = Mock(return_value=[])
    monkeypatch.setattr(extractor, '_structure_span_eager', eager)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(native, True, None)
        try:
            assert paused.wait(5)
            assert pool.submit(native, False, 0).result(timeout=5) == []
        finally:
            released.set()
        assert first.result(timeout=5) == []

    eager.assert_called_once()
    assert not SHARED_TOOLS.intersection(t['name'] for t in eager.call_args.kwargs['extraction_tools'])
    actual = {t['name'] for t in llm.complete_with_tools_messages.call_args.kwargs['tools']}
    assert SHARED_TOOLS <= actual
    recovery, = [attempt for attempt in prepared.assembly_state.attempts
                 if attempt.reason == 'lazy_eager_recovery']
    assert recovery.mode == 'eager' and not SHARED_TOOLS.intersection(recovery.tool_names)
    assert prepared.assembly_state.contract.mode == 'lazy'
    llm.complete_json.assert_called_once()


@pytest.mark.parametrize('thread_bound', [False, True])
def test_independent_span_selection_respects_storage_thread_boundary(thread_bound, monkeypatch):
    owner = get_ident()

    class Storage:
        def requires_thread_bound_access(self):
            return thread_bound

    llm = Mock()
    extractor = make_extractor(llm, mode='eager', fs=Storage())
    overlap, lock = Event(), Lock()
    active, peak = 0, 0
    prefetch_threads = []

    def select(*args, **kwargs):
        nonlocal active, peak
        if thread_bound:
            assert get_ident() == owner
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                overlap.set()
        if not thread_bound:
            overlap.wait(1)
        with lock:
            active -= 1
        return {"categories": ["entity"]}

    def prepare_prefetch(span, prompt, ctx):
        prefetch_threads.append(get_ident())
        return prompt

    llm.complete_json.side_effect = select
    monkeypatch.setattr(extractor, '_with_prefetch_context', prepare_prefetch)
    eager = Mock(return_value=[])
    monkeypatch.setattr(extractor, '_structure_span_eager', eager)
    messages = [*MESSAGES, *[{'role': 'assistant', 'content': f'Turn {i}'} for i in range(7)],
                {'role': 'user', 'content': 'Project Orion uses Redis.'}]
    spans = [_Span(i, i, categories=['entity']) for i in (0, 8)]
    assert extractor._structure_spans(spans, messages, 'English', TIME, '', '', CTX) == []
    assert llm.complete_json.call_count == 2
    assert eager.call_count == 4
    assert prefetch_threads == [owner, owner]
    assert peak == (1 if thread_bound else 2)


def test_selector_transport_failure_stops_dispatching_later_spans():
    llm = Mock()
    llm.complete_json.side_effect = LLMTransientError('provider unavailable')
    extractor = make_extractor(llm, mode='eager')
    messages = [{'role': 'user', 'content': f'Source message {i}.'} for i in range(9)]
    spans = [_Span(i, i, categories=['entity']) for i in range(len(messages))]

    with pytest.raises(LLMTransientError, match='provider unavailable'):
        extractor._structure_spans(spans, messages, 'English', TIME, '', '', CTX)

    assert 1 <= llm.complete_json.call_count <= 4
    llm.complete_with_tools.assert_not_called()


@pytest.mark.parametrize(
    'thread_bound, retain_ranges, tool_stats, categories, selection_calls, extraction_calls',
    [
        pytest.param(False, False, "", ["preference"], 1, 2, id="same-window"),
        pytest.param(True, False, "", ["preference"], 1, 2, id="owner-thread"),
        pytest.param(False, True, "", ["preference"], 1, 4, id="separate-source-ranges"),
        pytest.param(
            False, False, "search: 2 calls", ["preference"], 2, 4, id="different-tool-stats"
        ),
        pytest.param(False, False, "", [], 1, 0, id="no-task"),
        pytest.param(False, False, "", ["unknown"], 2, 2, id="full-fallback"),
    ],
)
def test_equal_selection_requests_across_distinct_spans(
    thread_bound, retain_ranges, tool_stats, categories, selection_calls, extraction_calls,
):
    owner = get_ident()

    class Storage:
        def requires_thread_bound_access(self):
            return thread_bound

    messages = [
        {'id': 'intro', 'role': 'assistant', 'content': 'Tell me about your interests.'},
        {'id': 'coffee', 'role': 'user', 'content': 'I like dark coffee.'},
        {'id': 'hiking', 'role': 'user', 'content': 'I prefer to hike on the weekends.'},
        {'id': 'ack', 'role': 'assistant', 'content': 'Thanks for sharing.'},
        {'id': 'end', 'role': 'assistant', 'content': 'Talk to you later.'},
    ]
    llm = Mock()
    requests = []

    def complete_json(prompt, schema):
        assert not thread_bound or get_ident() == owner
        if 'spans' in schema['properties']:
            return {'spans': [
                {'start': i, 'end': i, 'reason': 'explicit preference', 'categories': ['preference']}
                for i in (1, 2)
            ]}
        assert 'categories' in schema['properties']
        requests.append(prompt)
        return {"categories": categories}

    def complete_with_tools(**kwargs):
        assert not thread_bound or get_ident() == owner
        if categories == ['unknown']:
            assert any(tool['name'] == 'extract_event' for tool in kwargs['tools'])
        return [
            {'tool': 'extract_preference', 'input': {
                'routing_key': name, 'abstract': fact, 'overview': fact, 'content': fact, 'confidence': .95,
            }}
            for name, fact in [('coffee', 'Likes dark coffee'), ('hiking', 'Prefers hiking on weekends')]
        ]

    llm.complete_json.side_effect = complete_json
    llm.complete_with_tools.side_effect = complete_with_tools
    extractor = make_extractor(
        llm, mode='eager', fs=Storage(), source_occurrence_reconciliation_enabled=retain_ranges,
    )
    candidates = extractor.extract(messages, CTX, session_time=TIME, tool_stats_text=tool_stats,
                                   archive_id='archive')

    assert len(requests) == selection_calls
    assert len(set(requests)) == selection_calls
    assert llm.complete_with_tools.call_count == extraction_calls
    if extraction_calls:
        assert sorted(candidate.routing_key for candidate in candidates) == ['coffee', 'hiking']
        if extraction_calls == 2:
            assert all(candidate.provenance_ids == ['prov:1:archive:archive:coffee%2Chiking']
                       for candidate in candidates)
    else:
        assert candidates == []
