"""Create independent local experiment credentials once, without displaying them."""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import yaml

from experiments.settings import ROOT, read_settings


def bootstrap(source, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite existing independent experiment credentials')
    config = yaml.safe_load(Path(source).read_text(encoding='utf-8'))
    memory, embedding = config['topic_extraction'], config['embedding']
    values = {
        'HESM_EVAL_URL': 'http://127.0.0.1:8766',
        'HESM_EVAL_TOKEN': secrets.token_urlsafe(32),
        'HESM_EVAL_MEMORY_API_KEY': memory['api_key'],
        'HESM_EVAL_MEMORY_BASE_URL': memory['base_url'],
        'HESM_EVAL_EMBEDDING_API_KEY': embedding['api_key'],
        'HESM_EVAL_EMBEDDING_BASE_URL': embedding['base_url'],
        'ANSWER_MODEL': 'gpt-4.1-mini-2025-04-14',
        'ANSWER_API_KEY': memory['api_key'], 'ANSWER_BASE_URL': memory['base_url'],
        'EVAL_MODEL': 'gpt-4o-mini-2024-07-18',
        'EVAL_API_KEY': memory['api_key'], 'EVAL_BASE_URL': memory['base_url'],
        'HESM_EVAL_TIMEOUT': '3600', 'LLM_WORKERS': '2', 'TOPK': '20',
        'ANONYMIZED_TELEMETRY': 'False', 'PYTHONUTF8': '1',
        'HESM_EVAL_CONFIG_FINGERPRINT': read_settings(ROOT / 'experiments/config/user_memory.yaml')['config_fingerprint'],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as f:
        f.write('# Independent credential copy. Runtime does not read production configuration.\n')
        for key, value in values.items():
            f.write(f'{key}={json.dumps(str(value))}\n')
    print(f'Created independent experiment env: {destination}; credential values omitted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-credentials', required=True, help='One-time credential source; models and tuning are not inherited')
    parser.add_argument('--output', default=str(ROOT / 'experiments/.env.user_memory'))
    args = parser.parse_args()
    bootstrap(args.source_credentials, args.output)
