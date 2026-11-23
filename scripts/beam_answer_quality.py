# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Answer-text checks, separate from the session's actual-tool-execution audit."""
import re

VERSION = 'beam-answer-quality-v1'
TOOL_MARKUP = re.compile(r'DSML|<\s*(?:tool_call|function_call)\b', re.I)


def assess_answer(text):
    if not isinstance(text, str):
        raise ValueError('Answer must be a string')
    flags = []
    if not text.strip():
        flags.append('empty_answer')
    count = len(TOOL_MARKUP.findall(text))
    if count:
        flags.append('tool_markup_in_answer')
    return {'protocol': VERSION, 'flags': flags, 'requires_review': bool(flags),
            'literal_tool_markers': count}


def summarize_answers(answers):
    rows = [{'category': category, 'question_index': q['question_index'],
             **assess_answer(q['response'])}
            for category, questions in answers.items() for q in questions]
    return {'protocol': VERSION, 'question_count': len(rows),
            'flagged_question_count': sum(r['requires_review'] for r in rows),
            'questions': rows}
