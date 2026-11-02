"""Pinned BEAM functions and prompt, with no local scoring overrides.

Load the upstream functions without initializing unused embedding/BLEU models.
Install json-repair and scipy for the active LLM-alignment path. All scoring
function bodies and the judge prompt are preserved from the pinned MIT-licensed
upstream release in scripts/vendor/beam. This adapter skips upstream model and
NLTK download side effects, not any of the active scoring functions.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import List, Tuple

COMMIT = 'b2da22eac88bb0874c64665f13457eb99835774a'
PROTOCOL = 'beam-official-b2da22ea-deepseek-v1'
VENDOR = Path(__file__).with_name('vendor') / 'beam'
PINNED_HASHES = {
    'compute_metrics.py': 'e72bf25d36d521d166862063ec94f8e7875328193142d1edddb9b4677d47780d',
    'report_results.py': '386257d0a9b3b6e610d286ada9f7437cf4caed89702957cf8ccf30be9e006599',
    'LICENSE': '60c7b74ffaacaae383c3f06a48bc79d204b25bc9c1856df9711b3d290febd474',
    'unified_llm_judge_base_prompt.txt': '593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5',
}
CATEGORIES = ('abstention', 'contradiction_resolution', 'event_ordering',
              'information_extraction', 'instruction_following', 'knowledge_update',
              'multi_session_reasoning', 'preference_following', 'summarization', 'temporal_reasoning')


def vendor_hashes():
    metadata = json.loads((VENDOR / 'provenance.json').read_text(encoding='utf-8'))
    if metadata['commit'] != COMMIT or metadata['files'] != PINNED_HASHES:
        raise ValueError('Unexpected BEAM upstream version')
    hashes = {}
    for name, expected in metadata['files'].items():
        actual = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Vendored BEAM file changed: {name}')
        hashes[name] = actual
    hashes['provenance.json'] = hashlib.sha256((VENDOR / 'provenance.json').read_bytes()).hexdigest()
    return hashes


def prompt():
    vendor_hashes()
    return (VENDOR / 'unified_llm_judge_base_prompt.txt').read_text(encoding='utf-8')


def load_metrics():
    vendor_hashes()
    try:
        from json_repair import repair_json
        from scipy.stats import kendalltau
    except ImportError as exc:
        raise RuntimeError('BEAM grading requires json-repair and scipy; install the script requirements') from exc
    namespace = {'re': re, 'json': json, 'repair_json': repair_json,
                 'List': List, 'Tuple': Tuple, 'kendalltau': kendalltau,
                 'unified_llm_judge_base_prompt': prompt()}
    path = VENDOR / 'compute_metrics.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definitions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    exec(compile(definitions, str(path), 'exec'), namespace)
    return namespace


def summarize(metrics):
    """Upstream report aggregation; event ordering uses tau_norm, not F1*tau.

    question_mean and rubric_mean retain their historical all-rubric meaning.
    The official macro mean is explicitly the mean of the category columns.
    """
    if not metrics or any(not qs for qs in metrics.values()):
        raise ValueError('Cannot summarize missing BEAM judgments')
    categories = {c: sum(q['tau_norm'] if c == 'event_ordering' else q['llm_judge_score']
                         for q in qs) / len(qs) for c, qs in metrics.items()}
    questions = [q for qs in metrics.values() for q in qs]
    rubrics = [float(r['score']) for q in questions for r in q['llm_judge_responses']]
    return {'official_category_scores': categories,
            'official_category_macro_mean': sum(categories.values()) / len(categories),
            'question_mean': sum(q['llm_judge_score'] for q in questions) / len(questions),
            'rubric_mean': sum(rubrics) / len(rubrics),
            'question_count': len(questions), 'rubric_count': len(rubrics)}
