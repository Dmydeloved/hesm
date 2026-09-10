"""Build smoke inputs, using one unchanged official LoCoMo conversation."""
import argparse
import copy
import json
from pathlib import Path

from experiments.settings import ROOT


def prepare(directory=None, locomo_source=None, conversation_index=0):
    root = Path(directory or ROOT / 'experiments/.runtime/smoke_data')
    root.mkdir(parents=True, exist_ok=True)
    source = Path(
        locomo_source
        or ROOT.parent / 'OmniMemEval/data/locomo/locomo10.json'
    ).expanduser().resolve(strict=True)
    source_data = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(source_data, list) or not source_data:
        raise ValueError('LoCoMo source must be a nonempty JSON array')
    if not 0 <= int(conversation_index) < len(source_data):
        raise IndexError(
            f'conversation_index must be between 0 and {len(source_data) - 1}'
        )
    # Deep-copy one complete record without changing conversation, QA, evidence,
    # summaries, observations, sample_id, or any other benchmark field.
    locomo = [copy.deepcopy(source_data[int(conversation_index)])]
    (root / 'locomo.json').write_text(
        json.dumps(locomo, ensure_ascii=False), encoding='utf-8'
    )
    sessions = [[{'role': 'user', 'content': 'I live in Berlin.', 'has_answer': False},
                 {'role': 'assistant', 'content': 'The access code is ambernine.', 'has_answer': False}],
                [{'role': 'user', 'content': 'I moved to Paris and no longer live in Berlin.', 'has_answer': True}]]
    lme = [{'question_id': 'hesm_smoke_update', 'question_type': 'knowledge-update',
            'question': 'Where do I live now?', 'answer': 'Paris', 'question_date': '2024/03/01 (Fri) 13:00',
            'haystack_sessions': sessions, 'haystack_dates': ['2024/01/01 (Mon) 13:00', '2024/02/01 (Thu) 13:00'],
            'haystack_session_ids': ['s1', 's2'], 'answer_session_ids': ['s2']}]
    (root / 'lme.json').write_text(json.dumps(lme), encoding='utf-8')
    beam = {'conversation_id': 'hesm_format_smoke', 'chat': [
        [{'role': 'user', 'content': 'I live in Berlin.', 'time_anchor': 'January-01-2024'},
         {'role': 'assistant', 'content': 'The access code is ambernine.', 'time_anchor': 'January-01-2024'}],
        [{'role': 'user', 'content': 'I moved to Paris and no longer live in Berlin.', 'time_anchor': 'February-01-2024'}]],
        'probing_questions': json.dumps({'knowledge_update': [{
            'question': 'Where does the user live now?', 'answer': 'Paris',
            'rubric': ['The user currently lives in Paris.'], 'difficulty': 'easy'}]})}
    (root / 'beam').mkdir(exist_ok=True)
    (root / 'beam/beam_100k.json').write_text(json.dumps(beam) + '\n', encoding='utf-8')
    print(
        f'Created one-conversation LoCoMo smoke input from {source} '
        f'(index={conversation_index}) and synthetic LME/BEAM fixtures: {root}'
    )
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--locomo-source')
    parser.add_argument('--conversation-index', type=int, default=0)
    args = parser.parse_args()
    prepare(args.output, args.locomo_source, args.conversation_index)
