"""Offline contracts for the portable, pinned BEAM grader (no paid calls)."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import beam_official
from scripts import grade_beam_answers as grader


@pytest.fixture(autouse=True)
def block_real_judge(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('An offline grader test attempted a provider request')

    monkeypatch.setattr('requests.post', denied)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('BEAM_JUDGE_API_KEY', raising=False)


def arguments(tmp_path, **changes):
    answers = tmp_path / 'qa' / 'answers.json'
    answers.parent.mkdir(exist_ok=True)
    data = {'information_extraction': [{
        'question_index': 1, 'question': 'What was recorded?',
        'response': 'A recorded answer.', 'rubric': ['Fact A', 'Fact B'],
    }]}
    answers.write_text(json.dumps(data), encoding='utf-8')
    return SimpleNamespace(**{
        'answers': answers, 'output': tmp_path / 'grade', 'questions': 1, 'rubrics': 2,
        'model': 'test-judge', 'base_url': 'https://judge.invalid/v1',
        'execute': False, 'resume': False, 'verify_offline': False, **changes,
    })


def provider_response(content, finish_reason='stop'):
    return {'choices': [{'finish_reason': finish_reason, 'message': {'content': content}}]}


def mock_judge(monkeypatch, contents):
    responses = iter(contents)
    calls = []

    def post(endpoint, *, headers, json, timeout):
        calls.append({'endpoint': endpoint, 'headers': headers, 'body': json})
        return SimpleNamespace(status_code=200, json=lambda: provider_response(next(responses)))

    monkeypatch.setattr('requests.post', post)
    monkeypatch.setenv('BEAM_JUDGE_API_KEY', 'offline-test-secret')
    return calls


def test_vendored_sources_and_prompt_match_pinned_release():
    hashes = beam_official.vendor_hashes()
    assert {k: hashes[k] for k in beam_official.PINNED_HASHES} == beam_official.PINNED_HASHES
    assert len(beam_official.CATEGORIES) == 10
    text = beam_official.prompt()
    assert '<rubric_item>' in text and '<llm_response>' in text


def test_vendor_tampering_fails_even_if_manifest_is_rehashed(tmp_path, monkeypatch):
    shutil.copytree(beam_official.VENDOR, tmp_path / 'vendor')
    monkeypatch.setattr(beam_official, 'VENDOR', tmp_path / 'vendor')
    source = beam_official.VENDOR / 'compute_metrics.py'
    source.write_text(source.read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='Vendored BEAM file changed'):
        beam_official.vendor_hashes()
    metadata = beam_official.VENDOR / 'provenance.json'
    data = json.loads(metadata.read_text())
    data['files']['compute_metrics.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unexpected BEAM upstream version'):
        beam_official.vendor_hashes()


def test_upstream_event_alignment_and_reporting_are_not_replaced():
    official = beam_official.load_metrics()
    # The extra item lowers F1; the official report nevertheless uses tau_norm,
    # not the event function's F1*tau final_score or its rubric average.
    class AlignmentJudge:
        def invoke(self, messages):
            text = messages[-1]['content']
            return SimpleNamespace(content='YES' if 'Second snippet: A' in text
                                   or 'Second snippet: B' in text else 'NO')

    metric = official['event_ordering_score'](['A', 'B'], ['A', 'B', 'extra'],
                                               'llm', AlignmentJudge())
    assert metric['final_score'] < metric['tau_norm']
    metric.update(llm_judge_score=0.5, llm_judge_responses=[{'score': 0.5}])
    summary = beam_official.summarize({
        'event_ordering': [metric],
        'information_extraction': [{'llm_judge_score': 0.0,
                                    'llm_judge_responses': [{'score': 0.0}]}],
    })
    assert summary['official_category_scores']['event_ordering'] == metric['tau_norm']
    assert summary['official_category_macro_mean'] == metric['tau_norm'] / 2
    assert summary['rubric_mean'] == 0.25


@pytest.mark.parametrize('category', beam_official.CATEGORIES)
def test_all_official_categories_receive_the_exact_upstream_rubric_prompt(category):
    official = beam_official.load_metrics()
    rubric, answer = ['A', 'B'], 'A\nB'
    expected = [beam_official.prompt().replace('<rubric_item>', item).replace(
        '<llm_response>', answer) for item in rubric]
    prompts = []

    class Judge:
        def invoke(self, prompt):
            if isinstance(prompt, str):
                prompts.append(prompt)
                return SimpleNamespace(content='{"score": 0.5, "reason": "Partial"}')
            return SimpleNamespace(content='YES')

    metric = official['evaluate_' + category](rubric, answer, 'Original probe', Judge())
    assert prompts == expected
    assert metric['llm_judge_score'] == 0.5
    assert len(metric['llm_judge_responses']) == 2
    if category == 'event_ordering':
        assert metric['tau_norm'] == 1.0


def test_prepare_does_not_call_provider_and_uses_exact_official_prompt(tmp_path):
    args = arguments(tmp_path)
    result = grader.run(args)
    assert result['status'] == 'prepared_offline'
    requests = json.loads((args.output / 'rubric-requests.json').read_text())
    assert len(requests) == 2
    assert requests[0] == grader.request_for(beam_official.prompt().replace(
        '<rubric_item>', 'Fact A').replace('<llm_response>', 'A recorded answer.'), 'test-judge')
    assert not (args.output / 'results.json').exists()


def test_execute_then_verify_offline_retains_raw_output_and_denominators(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    grader.run(args)
    calls = mock_judge(monkeypatch, ['{"score": 1, "reason": "Full"}',
                                    '{"score": 0.5, "reason": "Partial"}'])
    args.execute, args.resume = True, True
    result = grader.run(args)
    assert result['official_category_macro_mean'] == 0.75
    assert result['question_count'] == 1 and result['rubric_count'] == 2
    assert len(calls) == 2
    assert calls[0]['endpoint'] == 'https://judge.invalid/v1/chat/completions'
    assert calls[0]['body']['temperature'] == 0
    assert calls[0]['headers']['Authorization'] == 'Bearer offline-test-secret'
    assert 'offline-test-secret' not in ''.join(p.read_text() for p in args.output.rglob('*.json'))
    frozen = {p: p.read_bytes() for p in args.output.rglob('*.json')}
    args.execute, args.verify_offline = False, True
    monkeypatch.delenv('BEAM_JUDGE_API_KEY')
    replay = grader.run(args)
    assert replay == result and len(calls) == 2
    assert all(p.read_bytes() == value for p, value in frozen.items())


@pytest.mark.parametrize('score', [True, 0.4, 2, -1, None])
def test_invalid_judge_scores_never_become_completed_results(tmp_path, monkeypatch, score):
    args = arguments(tmp_path, execute=True)
    mock_judge(monkeypatch, [json.dumps({'score': score, 'reason': 'Invalid'}),
                             '{"score": 1, "reason": "Other"}'])
    with pytest.raises((ValueError, TypeError)):
        grader.run(args)
    assert not (args.output / 'results.json').exists()
    assert list(args.output.glob('calls/*/*.response.json'))


def test_truncated_judge_response_is_retained_without_a_score(tmp_path, monkeypatch):
    args = arguments(tmp_path, execute=True)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-test-secret')
    raw = provider_response('{"score":', finish_reason='length')
    monkeypatch.setattr('requests.post', lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: raw))
    with pytest.raises(ValueError, match='did not finish normally'):
        grader.run(args)
    assert json.loads(next(args.output.glob('calls/*/*.response.json')).read_text()) == raw
    assert not (args.output / 'results.json').exists()


@pytest.mark.parametrize('field, value', [('questions', 2), ('rubrics', 1)])
def test_expected_denominators_are_required(tmp_path, field, value):
    args = arguments(tmp_path, **{field: value})
    with pytest.raises(ValueError, match='denominators'):
        grader.run(args)
    assert not args.output.exists()


def test_changed_answers_or_configuration_cannot_resume(tmp_path):
    args = arguments(tmp_path)
    grader.run(args)
    args.resume = True
    args.model = 'a-different-judge'
    with pytest.raises(ValueError, match='configuration changed'):
        grader.run(args)
    args.model = 'test-judge'
    original = args.answers.read_text()
    args.answers.write_text(original.replace('recorded answer', 'changed answer'))
    with pytest.raises(ValueError, match='configuration changed'):
        grader.run(args)


def test_response_digest_detects_tampering(tmp_path, monkeypatch):
    judge = grader.CachedJudge(tmp_path / 'calls', 'test', 'https://judge.invalid', None)
    mock_judge(monkeypatch, ['{"score": 1}'])
    judge.key = 'offline-test-secret'
    judge.invoke('A prompt')
    response = next((tmp_path / 'calls').glob('*.response.json'))
    response.write_text(json.dumps(provider_response('{"score": 0}')))
    replay = grader.CachedJudge(tmp_path / 'calls', 'test', 'https://judge.invalid', None, offline=True)
    with pytest.raises(ValueError, match='binding changed'):
        replay.invoke('A prompt')


def test_offline_missing_response_does_not_fall_back_to_a_provider(tmp_path):
    judge = grader.CachedJudge(tmp_path / 'calls', 'test', 'https://judge.invalid', None,
                               offline=True)
    with pytest.raises(ValueError, match='Missing retained response'):
        judge.invoke('A prompt')


@pytest.mark.parametrize('endpoint', ['https://user:secret@judge.invalid',
                                     'https://judge.invalid/?api_key=secret',
                                     'file:///tmp/judge', 'https://judge.invalid/#secret'])
def test_endpoint_cannot_embed_credentials(tmp_path, endpoint):
    args = arguments(tmp_path, base_url=endpoint)
    with pytest.raises(ValueError, match='plain provider URL'):
        grader.run(args)
    assert not args.output.exists()


def test_offline_verification_requires_resume(tmp_path):
    with pytest.raises(ValueError, match='requires --resume'):
        grader.run(arguments(tmp_path, verify_offline=True))


def test_cli_help_is_portable_by_module_and_file(tmp_path):
    root = Path(grader.__file__).resolve().parent.parent
    for argv, cwd in [([sys.executable, '-m', 'scripts.grade_beam_answers', '--help'], root),
                      ([sys.executable, str(Path(grader.__file__).resolve()), '--help'], tmp_path)]:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert '--verify-offline' in result.stdout and '--execute' in result.stdout
