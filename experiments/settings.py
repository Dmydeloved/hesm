"""Strict, independent configuration for experimental services."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_settings(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve(strict=True)
    if target == (ROOT / 'config/hesm.yaml').resolve():
        raise ValueError('The production HESM config cannot be used for experiments')
    data = yaml.safe_load(target.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Expected an explicit schema_version: 1 experiment config')
    required = {'profile', 'data_root', 'models', 'memory', 'retrieval', 'formal', 'api_token_env'}
    if required - data.keys():
        raise ValueError(f'Missing experiment settings: {sorted(required - data.keys())}')
    root = (target.parent / data['data_root']).resolve()
    production = (ROOT / 'memory').resolve()
    if root == production or production in root.parents or root in production.parents:
        raise ValueError('Experiment storage must be separate from production memory')
    if root == ROOT or root in ROOT.parents:
        raise ValueError('Experiment storage must be a dedicated directory')
    for section, keys in [('memory', ['segment_qa_limit', 'experience_summary_segment_threshold', 'input_chunk_tokens']),
                          ('retrieval', ['experience_candidates', 'segments_per_experience', 'qas_per_segment', 'context_tokens', 'max_top_k'])]:
        for key in keys:
            value = data[section].get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    for key in ['include_state', 'include_experience']:
        if type(data['retrieval'].get(key)) is not bool:
            raise ValueError(f'retrieval.{key} must be a boolean')
    if not data['retrieval']['include_state']:
        raise ValueError('The main experiment uses full HESM State; disabling State requires a separate ablation implementation')
    # The fingerprint records configuration, never credential values.
    data['config_fingerprint'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    data['config_path'] = str(target)
    data['data_root'] = str(root)
    return data


def connection(settings: dict, section: str) -> dict:
    model = settings['models'][section]
    values = {key: os.environ.get(model[f'{key}_env'], '').strip() for key in ['api_key', 'base_url']}
    if not all(values.values()) or not model.get('model'):
        raise ValueError(f'Missing independent {section} credentials/model; configure the experiment env file')
    return {**values, 'model': model['model'], 'timeout': float(model['timeout']),
            'max_retries': int(model['max_retries'])}


def load_env(path: str | Path) -> None:
    from dotenv import dotenv_values
    for key, value in dotenv_values(path).items():
        if value is not None:
            os.environ[key] = value
