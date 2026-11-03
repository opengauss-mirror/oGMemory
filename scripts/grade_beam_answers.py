"""Pinned official BEAM grading with DeepSeek; offline by default.

Install: pip install requests json-repair scipy
Prepare: python -m scripts.grade_beam_answers answers.json NEW_DIR --rubrics 50
Grade: add --execute (DEEPSEEK_API_KEY required). Resume: add --resume.
Replay without API calls: --resume --verify-offline.

No added question, custom prompt, quote requirement, or score promotion.
Original answers and historical judgments are never overwritten.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import beam_official
from scripts.beam_answer_quality import summarize_answers

PROTOCOL = beam_official.PROTOCOL


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Retained artifact changed: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_questions(path, expected_questions, expected_rubrics):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or expected_questions < 1 or expected_rubrics < 1:
        raise ValueError('Expected category-to-questions object and positive denominators')
    questions, seen = [], set()
    for category, rows in data.items():
        if category not in beam_official.CATEGORIES or not isinstance(rows, list) or not rows:
            raise ValueError('Unknown or empty BEAM category')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Question must be an object')
            index = row.get('question_index')
            if type(index) is not int or index < 1 or (category, index) in seen:
                raise ValueError('Invalid or duplicate question identity')
            seen.add((category, index))
            if not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError('Missing original question')
            if not isinstance(row.get('response'), str):
                raise ValueError('Missing saved answer')
            rubrics = row.get('rubric')
            if not isinstance(rubrics, list) or not rubrics or any(
                    not isinstance(r, str) or not r.strip() for r in rubrics):
                raise ValueError('Invalid rubrics')
            questions.append({**row, 'category': category})
    if len(questions) != expected_questions or sum(len(q['rubric']) for q in questions) != expected_rubrics:
        raise ValueError('Incomplete or unexpected question/rubric denominators')
    return data, questions


def request_for(prompt, model):
    return {'model': model, 'temperature': 0,
            'messages': [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt}


class CachedJudge:
    def __init__(self, folder, model, endpoint, key, offline=False):
        self.folder, self.model, self.endpoint = Path(folder), model, endpoint
        self.key, self.offline, self.records = key, offline, []

    def invoke(self, prompt):
        body = request_for(prompt, self.model)
        prefix = self.folder / f'call-{len(self.records)+1:04d}'
        save(prefix.with_suffix('.request.json'), body)
        raw_path, receipt_path = prefix.with_suffix('.response.json'), prefix.with_suffix('.binding.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt != {'request_sha256': digest(body), 'response_sha256': file_hash(raw_path)}:
                raise ValueError('Retained provider response binding changed')
        if raw_path.exists():
            raw = json.loads(raw_path.read_text(encoding='utf-8'))
        elif self.offline:
            raise ValueError('Missing retained response during offline verification')
        else:
            import requests
            for attempt in range(5):
                try:
                    response = requests.post(self.endpoint,
                        headers={'Authorization': 'Bearer ' + self.key}, json=body, timeout=180)
                    if response.status_code != 200:
                        save(prefix.with_suffix(f'.http-error-{time.time_ns()}.json'),
                             {'status_code': response.status_code})
                        if response.status_code != 429 and response.status_code < 500:
                            raise ValueError(f'Judge HTTP {response.status_code}; no grade assigned')
                        raise requests.HTTPError('Retryable judge HTTP failure')
                    raw = response.json()
                    save(raw_path, raw)
                    break
                except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                    if attempt == 4:
                        raise RuntimeError('Judge transport failed; retained responses preserved') from None
                    time.sleep(2 ** attempt)
        save(receipt_path, {'request_sha256': digest(body), 'response_sha256': file_hash(raw_path)})
        choice = raw['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Judge did not finish normally; raw output retained')
        if not isinstance(choice.get('message', {}).get('content'), str):
            raise ValueError('Judge returned no text; raw output retained')
        self.records.append({'request': body, 'raw_response': raw, 'response_sha256': file_hash(raw_path)})
        return SimpleNamespace(content=choice['message']['content'])


def run(args):
    verify = getattr(args, 'verify_offline', False)
    if verify and (args.execute or not args.resume):
        raise ValueError('--verify-offline requires --resume and cannot be combined with --execute')
    answers_path, output = Path(args.answers).resolve(), Path(args.output).resolve()
    data, questions = load_questions(answers_path, args.questions, args.rubrics)
    if output == answers_path.parent or answers_path.parent in output.parents:
        raise ValueError('Use a separate grading directory outside the saved answers')
    endpoint = urlsplit(args.base_url)
    if (endpoint.scheme not in ('http', 'https') or not endpoint.hostname or endpoint.username or
            endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('Use a plain provider URL without credentials or query parameters')
    binding = {'protocol': PROTOCOL, 'upstream_commit': beam_official.COMMIT,
               'answers_sha256': file_hash(answers_path), 'grader_sha256': file_hash(__file__),
               'engine_sha256': file_hash(beam_official.__file__), 'vendor_hashes': beam_official.vendor_hashes(),
               'quality_sha256': file_hash(Path(__file__).with_name('beam_answer_quality.py')),
               'question_count': args.questions, 'rubric_count': args.rubrics,
               'model': args.model, 'temperature': 0,
               'endpoint': args.base_url.rstrip('/') + '/chat/completions'}
    if args.resume:
        if json.loads((output / 'binding.json').read_text(encoding='utf-8')) != binding:
            raise ValueError('Resume inputs, official protocol, code or judge configuration changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'binding.json', binding)
        save(output / 'answer-quality.json', summarize_answers(data))
    if not args.execute and not verify:
        template = beam_official.prompt()
        save(output / 'rubric-requests.json', [request_for(
            template.replace('<rubric_item>', rubric).replace('<llm_response>', q['response']), args.model)
            for q in questions for rubric in q['rubric']])
        return {'status': 'prepared_offline', **binding}
    key = (getattr(args, 'api_key', None) or os.environ.get('BEAM_JUDGE_API_KEY')
           or os.environ.get('DEEPSEEK_API_KEY'))
    if not verify and not key:
        raise ValueError('Set BEAM_JUDGE_API_KEY or DEEPSEEK_API_KEY for --execute')
    official = beam_official.load_metrics()
    metrics, judgments = {}, {}
    for qi, q in enumerate(questions, 1):
        category = q['category']
        client = CachedJudge(output / 'calls' / f'q{qi:03d}', args.model, binding['endpoint'], key, verify)
        metric = official['evaluate_' + category](rubric=q['rubric'], llm_response=q['response'],
                                                probing_question=q['question'], model=client)
        grades = metric['llm_judge_responses']
        if len(grades) != len(q['rubric']) or any(
                type(g.get('score')) is bool or float(g['score']) not in (0, .5, 1) for g in grades):
            raise ValueError('Invalid official judge scores; no rounding or promotion')
        if any(not math.isfinite(float(metric[k])) for k in ('llm_judge_score', 'tau_norm') if k in metric):
            raise ValueError('Undefined official metric; no substitute score assigned')
        save(output / 'questions' / f'q{qi:03d}.json', {'category': category,
             'question_index': q['question_index'], 'question': q['question'], 'metrics': metric})
        rows = []
        for ri, (rubric, grade, provider) in enumerate(zip(q['rubric'], grades, client.records[-len(grades):])):
            expected = request_for(official['unified_llm_judge_base_prompt'].replace(
                '<rubric_item>', rubric).replace('<llm_response>', q['response']), args.model)
            if provider['request'] != expected:
                raise ValueError('Rubric/provider association differs from upstream')
            record = {'protocol': PROTOCOL, 'category': category, 'question_index': q['question_index'],
                      'rubric_index': ri, 'rubric': rubric, **grade, 'score': float(grade['score']),
                      'raw_response': provider['raw_response'], 'response_sha256': provider['response_sha256']}
            save(output / 'grades' / f'q{qi:03d}-r{ri:03d}.json', record)
            rows.append(record)
        metrics.setdefault(category, []).append(metric)
        judgments.setdefault(category, []).append({'question_index': q['question_index'],
            'question': q['question'], 'rubric_results': rows, 'score': metric['llm_judge_score']})
    summary = beam_official.summarize(metrics)
    if summary['question_count'] != args.questions or summary['rubric_count'] != args.rubrics:
        raise ValueError('Incomplete official grading')
    result = {'protocol': PROTOCOL, 'upstream_commit': beam_official.COMMIT,
              'status': 'completed', 'primary_metric': 'official_category_macro_mean',
              'metric_note': 'Official category macro uses tau_norm for event_ordering; question_mean and rubric_mean are supplemental all-rubric averages.',
              **summary, 'judgments': judgments, 'official_results': metrics}
    if file_hash(answers_path) != binding['answers_sha256']:
        raise ValueError('Source answers changed during grading')
    save(output / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('answers', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--questions', type=int, default=20)
    parser.add_argument('--rubrics', type=int, default=50)
    parser.add_argument('--model', default=os.environ.get('BEAM_JUDGE_MODEL', 'deepseek-chat'))
    parser.add_argument('--base-url', default=os.environ.get('BEAM_JUDGE_BASE_URL',
                        os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')))
    parser.add_argument('--execute', action='store_true', help='Enable paid judge calls')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-offline', action='store_true', help='Replay saved responses without API calls')
    result = run(parser.parse_args())
    print(json.dumps({k: v for k, v in result.items() if k not in ('judgments', 'official_results')}, indent=2))


if __name__ == '__main__':
    main()
