"""Check independent settings and optionally probe the required model endpoints."""
import argparse
import json
import os
from pathlib import Path

from experiments.settings import read_settings, load_env, connection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--env', required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--direct-network', action='store_true', help='Ignore inherited proxy settings (run with network permission)')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.direct_network:
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
            os.environ.pop(key, None)
    load_env(args.env)
    settings = read_settings(args.config)
    result = {'profile': settings['profile'], 'config_fingerprint': settings['config_fingerprint'], 'checks': []}
    from openai import OpenAI
    import httpx
    for stage in ['memory', 'embedding', 'ANSWER', 'EVAL']:
        cfg = {}
        try:
            if stage in ['memory', 'embedding']:
                cfg = connection(settings, stage)
            else:
                cfg = {key: os.environ.get(f'{stage}_{key.upper()}', '') for key in ['model', 'api_key', 'base_url']}
                if not all(cfg.values()):
                    raise ValueError('Missing independent environment values')
            item = {'stage': stage, 'requested_model': cfg['model'], 'status': 'configured'}
            if args.live:
                client = OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'], timeout=45, max_retries=1,
                                http_client=httpx.Client(trust_env=not args.direct_network))
                if stage == 'embedding':
                    response = client.embeddings.create(model=cfg['model'], input='HESM evaluation connectivity check')
                    item['dimensions'] = len(response.data[0].embedding)
                else:
                    response = client.chat.completions.create(model=cfg['model'], messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=8, temperature=0)
                    item['returned_model'] = response.model
                item['status'] = 'ok'
            result['checks'].append(item)
        except Exception as exc:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            reason = str(cause)
            api_key = cfg.get('api_key')
            if api_key:
                reason = reason.replace(api_key, '<redacted>')
            reason = reason[:300]
            result['checks'].append({'stage': stage, 'status': 'failed', 'error_type': type(exc).__name__,
                                    'http_status': getattr(exc, 'status_code', None), 'network_reason': reason})
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if any(c['status'] == 'failed' for c in result['checks']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
