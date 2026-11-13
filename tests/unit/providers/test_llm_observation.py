from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from providers.llm_observation import capture_llm_events, observe_call
from providers.token_tracker import TokenTracker


def test_observation_tracks_physical_retries_and_unknown_usage_without_tool_duplication():
    tracker = TokenTracker('test')
    def provider():
        tracker.record_failure('llm')
        tracker.record_llm(11, 3, usage_known=True)
        return [{'tool': 'one'}, {'tool': 'two'}]
    with capture_llm_events() as events:
        result = observe_call('extraction', provider, correlation={'span_id': 'a'})
    usage = [e for e in events if e['event'] == 'usage']
    assert len(result) == 2
    assert len(usage) == 2
    assert usage[0]['total_tokens'] is None
    assert usage[1]['total_tokens'] == 14
    assert len({e['request_id'] for e in events}) == 1
    assert all(e['span_id'] == 'a' for e in events)


def test_content_capture_is_explicit_and_failure_does_not_leak_context():
    def fail(prompt):
        raise ValueError('original')
    with capture_llm_events() as events:
        with pytest.raises(ValueError, match='original'):
            observe_call('selection', fail, 'private source')
    assert 'private source' not in repr(events)
    with capture_llm_events(include_content=True) as events:
        assert observe_call("selection", lambda prompt: {"categories": []}, "private") == {
            "categories": []
        }
    assert events[0]["args"] == ("private",)
    assert events[-1]["result"] == {"categories": []}
    assert events[-1]["stage"] == "selection"


def test_parallel_contexts_keep_request_and_stage_isolated():
    tracker = TokenTracker('test')
    def run(index):
        return observe_call('selection' if index else 'recovery',
                            lambda: tracker.record_llm(index + 1, 2),
                            correlation={'span_id': str(index)})
    with capture_llm_events() as events:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, run, i) for i in range(2)]
            [f.result() for f in futures]
    usage = [e for e in events if e['event'] == 'usage']
    assert {(e['span_id'], e['stage'], e['total_tokens']) for e in usage} == {
        ('0', 'recovery', 3), ('1', 'selection', 4)}


def test_provider_without_usage_reporting_remains_explicitly_unknown():
    with capture_llm_events() as events:
        observe_call("selection", lambda: {"categories": []})
    assert events[-1]["usage_observed"] is False
    assert not [e for e in events if e["event"] == "usage"]


def test_extraction_scope_correlates_early_selection_and_nested_lazy_calls():
    from types import SimpleNamespace

    from providers.llm_observation import correlate_extraction_calls, record_assembly_event
    @correlate_extraction_calls
    def run(span, ctx, archive_id=None):
        observe_call('selection', lambda: 'selected')
        record_assembly_event(selection={'categories': ['entity']}, decision={'scope': 'scoped'})
        return observe_call('extraction', lambda: 'done')
    with capture_llm_events() as events:
        assert run(SimpleNamespace(start=2, end=5),
                   SimpleNamespace(session_id='session', user_id='user', account_id='account'), 'archive') == 'done'
    assert {e['session_id'] for e in events} == {'session'}
    assert {e['span_id'] for e in events} == {'2:5'}
    assert {e['archive_id'] for e in events} == {'archive'}
    assert any(e['event'] == 'assembly' for e in events)


@pytest.mark.parametrize('stage', ['qa', 'other'])
def test_shared_react_loop_does_not_charge_qa_as_extraction(stage):
    from tests.unit.core.test_react_loop import MockReactLoop, _make_llm
    loop = MockReactLoop(_make_llm())
    if stage == 'qa':
        loop._llm_observation_stage = 'qa'
    with capture_llm_events() as events:
        loop._call_llm([{'role': 'user', 'content': 'question'}])
    assert {e['stage'] for e in events} == {stage}


def test_same_session_operations_have_stable_distinct_correlation_without_archive():
    from types import SimpleNamespace

    from providers.llm_observation import correlate_extraction_calls
    @correlate_extraction_calls
    def run(ctx):
        observe_call('selection', lambda: None)
        observe_call('postprocess', lambda: None)
    ctx = SimpleNamespace(session_id='same-session', trace_id='trace')
    with capture_llm_events() as events:
        run(ctx)
        run(ctx)
    assert events[0]['operation_id'] == events[3]['operation_id']
    assert events[0]['operation_id'] != events[4]['operation_id']
    assert {e['trace_id'] for e in events} == {'trace'}
