"""Dedicated HTTP service for OmniMemEval and OpenClaw; run with explicit config."""
from __future__ import annotations

import argparse
import hmac
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from experiments.backend import EvaluationBackend
from experiments.settings import read_settings, load_env


class Command(BaseModel):
    user_id: str
    messages: list[dict] = Field(default_factory=list)
    session_id: str = ''
    query: str = ''
    question_date: str | None = None
    top_k: int = 20
    complete: bool = False
    mode: str = ''
    snapshot_id: str = ''


def create_app(backend, token):
    app = FastAPI(title='HESM isolated evaluation API')

    def authorize(authorization):
        if not hmac.compare_digest(authorization or '', f'Bearer {token}'):
            raise HTTPException(401, 'Invalid experiment token')

    @app.get('/health')
    def health(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return backend.health()

    @app.post('/v1/{operation}')
    def command(operation: str, body: Command, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            if operation == 'add':
                return backend.add(body.user_id, body.messages, body.session_id)
            if operation == 'search':
                return backend.search(body.user_id, body.query, body.top_k, body.question_date)
            if operation == 'finalize':
                return backend.finalize(body.user_id, body.complete)
            if operation == 'delete':
                return backend.delete(body.user_id)
            if operation == 'mode':
                return backend.set_mode(body.user_id, body.mode)
            if operation == 'snapshot':
                return backend.snapshot(body.user_id)
            if operation == 'restore':
                return backend.restore(body.user_id, body.snapshot_id)
            raise HTTPException(404, 'Unknown operation')
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--env', required=True)
    args = parser.parse_args()
    load_env(args.env)
    settings = read_settings(args.config)
    token = os.environ.get(settings['api_token_env'], '')
    if len(token) < 24:
        raise SystemExit('Set a dedicated HESM_EVAL_TOKEN of at least 24 characters')
    # Production logging configuration is never initialized.
    logging.basicConfig(level=logging.WARNING)
    import uvicorn
    uvicorn.run(create_app(EvaluationBackend(settings), token),
                host=settings.get('host', '127.0.0.1'), port=settings['port'])


if __name__ == '__main__':
    main()
