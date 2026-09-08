"""Tiny synthetic format fixtures. These are NOT benchmark score datasets."""
import json
from pathlib import Path

from experiments.settings import ROOT


def prepare(directory=None):
    root = Path(directory or ROOT / 'experiments/.runtime/smoke_data')
    root.mkdir(parents=True, exist_ok=True)
    conversation = {
        'speaker_a': 'Mina', 'speaker_b': 'Jo',
        'session_1_date_time': '1:00 pm on 1 January, 2024',
        'session_1': [
            {'speaker': 'Mina', 'text': 'I live in Berlin. My favorite color is purple.', 'dia_id': 'D1:1'},
            {'speaker': 'Jo', 'text': 'My access code is ambernine.', 'dia_id': 'D1:2'}],
        'session_2_date_time': '1:00 pm on 1 February, 2024',
        'session_2': [{'speaker': 'Mina', 'text': 'I have moved to Paris and no longer live in Berlin.', 'dia_id': 'D2:1'}],
    }
    locomo = [{'sample_id': 'hesm_format_smoke', 'conversation': conversation, 'qa': [
        {'question': 'Where does Mina live now?', 'answer': 'Paris', 'category': 4, 'evidence': ['D2:1']},
        {'question': 'What is Jo\'s access code?', 'answer': 'ambernine', 'category': 4, 'evidence': ['D1:2']}
    ]}]
    (root / 'locomo.json').write_text(json.dumps(locomo), encoding='utf-8')
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
    print(f'Created synthetic smoke fixtures: {root} (not LoCoMo/LME/BEAM benchmark results)')
    return root


if __name__ == '__main__':
    prepare()
