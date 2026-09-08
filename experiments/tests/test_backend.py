import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.embedder import HashingEmbedder
from core.summarizer import TemplateSummarizer
from core.vector_store import ChromaVectorStore, _InMemoryCollection
from experiments.backend import EvaluationBackend
from experiments.components import EvaluationManager
from experiments.settings import ROOT, read_settings


class Collection(_InMemoryCollection):
    def get(self, *, limit, offset, include):
        items = sorted(self._items.values(), key=lambda x: x['id'])[offset:offset + limit]
        return {key: [i[field] for i in items] for key, field in [('ids', 'id'), ('documents', 'document'), ('embeddings', 'embedding'), ('metadatas', 'metadata')]}


class Vectors(ChromaVectorStore):
    def __init__(self, **kwargs):
        self.client = None
        self.collection = Collection()


class Extractor:
    def extract(self, text, **kwargs):
        if 'FAIL_MODEL' in text:
            raise RuntimeError('simulated model failure')
        return {'topic': 'personal facts', 'core_entity': 'person', 'intent': 'remember', 'entities': ['person'], 'confidence': 1., 'reasoning': 'fixture'}


class NoHistory:
    def recall(self, **kwargs):
        return {'history_experience': {}}


def components(settings, storage, vectors, directory):
    embedder = HashingEmbedder(128)
    manager = EvaluationManager(storage, vectors, embedder, TemplateSummarizer(),
        experience_recaller=NoHistory(), segment_summary_qa_threshold=2)
    return Extractor(), embedder, manager


@pytest.fixture
def backend(tmp_path):
    config = read_settings(ROOT / 'experiments/config/user_memory.yaml')
    config['data_root'] = str(tmp_path / 'memory')
    config['formal']['require_chroma'] = False
    config['formal']['require_real_models'] = False
    config['retrieval']['context_tokens'] = 4096
    return EvaluationBackend(config, component_factory=components, vector_factory=Vectors)


def message(content, date='2024-01-01T00:00:00Z', role='user'):
    return {'role': role, 'content': content, 'chat_time': date, 'has_answer': True, 'gold': 'DO_NOT_INGEST'}


def state(backend, user):
    storage, vectors, *_ = backend._open(backend._row(user))
    return '\n'.join(storage.connection.iterdump()), copy.deepcopy(vectors.collection._items)


def test_isolation_and_readonly_query_order(backend):
    backend.add('alice', [message('My secret is orchidseven')], 's1')
    backend.add('bob', [message('My secret is cobalteight')], 's1')
    before = state(backend, 'alice')
    first = backend.search('alice', 'What is my secret?')['context']
    backend.search('alice', 'An entirely new topic with no previous history')
    assert first == backend.search('alice', 'What is my secret?')['context']
    assert 'orchidseven' in first and 'cobalteight' not in first
    assert 'DO_NOT_INGEST' not in first
    assert state(backend, 'alice') == before


def test_idempotency_and_session_conflict(backend):
    messages = [message('I moved to Paris')]
    backend.add('u', messages, 's1')
    before = state(backend, 'u')
    assert backend.add('u', messages, 's1')['status'] == 'already_ingested'
    assert state(backend, 'u') == before
    with pytest.raises(ValueError, match='different content'):
        backend.add('u', [message('Different')], 's1')


def test_quarantine_partial_failure_and_reset(backend):
    with pytest.raises(RuntimeError, match='simulated'):
        backend.add('u', [message('first succeeds'), message('FAIL_MODEL')], 's1')
    with pytest.raises(RuntimeError, match='quarantined'):
        backend.search('u', 'first')
    backend.delete('u')
    backend.add('u', [message('replacement')], 's1')
    assert 'replacement' in backend.search('u', 'replacement')['context']


def test_snapshot_restore_and_frozen_test(backend):
    backend.add('u', [message('before snapshot')], 's1')
    snapshot = backend.snapshot('u')
    before = backend.search('u', 'snapshot')['context']
    backend.add('u', [message('after snapshot', '2024-02-01T00:00:00Z')], 's2')
    backend.set_mode('u', 'test')
    backend.restore('u', snapshot['snapshot_id'])
    assert backend.search('u', 'snapshot')['context'] == before
    with pytest.raises(PermissionError):
        backend.add('u', [message('test contamination')], 's3')
    with pytest.raises(PermissionError):
        backend.delete('u')
    with pytest.raises(ValueError, match='namespace'):
        backend.restore('other_user', snapshot['snapshot_id'])


def test_assistant_evidence_unknown_dates_and_budget(backend):
    backend.add('u', [message('Assistant says the access code is ambernine.', None, 'assistant')], 's1')
    result = backend.search('u', 'access code', question_date='2024-03-01')
    assert 'ambernine' in result['context']
    assert result['query_date'] == '2024-03-01'
    backend.settings['retrieval']['context_tokens'] = 80
    assert backend.search('u', 'access code')['context_tokens'] <= 80


def test_out_of_order_sessions_rejected(backend):
    backend.add('u', [message('newer', '2024-02-01T00:00:00Z')], 's2')
    with pytest.raises(ValueError, match='chronological'):
        backend.add('u', [message('older')], 's1')


def test_production_config_rejected():
    with pytest.raises(ValueError, match='production'):
        read_settings(ROOT / 'config/hesm.yaml')


def test_chunking_preserves_all_unicode(backend):
    backend.settings['memory']['input_chunk_tokens'] = 9
    text = '中文🙂é test' * 30
    chunks = list(backend._chunks(text))
    assert ''.join(chunks) == text
    assert all(len(backend.tokenizer.encode(c, disallowed_special=())) <= 9 for c in chunks)
