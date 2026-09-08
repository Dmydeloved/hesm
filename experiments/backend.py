"""Namespace-isolated evaluation backend; retrieval never calls write routing."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from core.storage import MemoryStorage
from core.time_utils import format_timestamp
from core.vector_store import ChromaVectorStore
from experiments.components import make_components, EvaluationVectors


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class EvaluationBackend:
    def __init__(self, settings, *, component_factory=make_components, vector_factory=EvaluationVectors):
        import tiktoken
        self.settings = settings
        self.root = Path(settings['data_root']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / 'experiment.json'
        if marker.exists():
            previous = json.loads(marker.read_text(encoding='utf-8'))
            if previous['config_fingerprint'] != settings['config_fingerprint']:
                raise ValueError('This data_root belongs to a different experiment config; use a new data_root')
        else:
            write_json(marker, settings)
        self.tokenizer = tiktoken.get_encoding(settings['retrieval']['tokenizer'])
        self.component_factory, self.vector_factory = component_factory, vector_factory
        self.lock = threading.RLock()
        self.handles = {}
        self.catalog = sqlite3.connect(self.root / 'catalog.sqlite3', check_same_thread=False)
        self.catalog.row_factory = sqlite3.Row
        self.catalog.execute('CREATE TABLE IF NOT EXISTS namespaces (id TEXT PRIMARY KEY, generation TEXT NOT NULL, mode TEXT NOT NULL)')
        self.catalog.commit()

    def _id(self, user_id):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError('An explicit nonempty user_id is required')
        return digest(user_id)

    def _row(self, user_id, *, create=False):
        key = self._id(user_id)
        row = self.catalog.execute('SELECT * FROM namespaces WHERE id=?', (key,)).fetchone()
        if row is None and create:
            self.catalog.execute('INSERT INTO namespaces VALUES (?,?,?)', (key, uuid.uuid4().hex, 'train'))
            self.catalog.commit()
            row = self.catalog.execute('SELECT * FROM namespaces WHERE id=?', (key,)).fetchone()
        return row

    def _open(self, row):
        key = row['generation']
        if key not in self.handles:
            directory = self.root / 'namespaces' / row['id'] / key
            directory.mkdir(parents=True, exist_ok=True)
            storage = MemoryStorage(directory / 'memory.sqlite3', check_same_thread=False)
            storage.connection.executescript('''
                CREATE TABLE IF NOT EXISTS eval_events (
                  source_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event_time TEXT,
                  content_hash TEXT NOT NULL, qa_ids TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS eval_sessions (
                  session_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, completed INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS eval_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            ''')
            storage.commit()
            vectors = self.vector_factory(persist_path=self.root / 'chroma', collection_name=f'hesm_{key}')
            if self.settings['formal']['require_chroma'] and vectors.client is None:
                storage.connection.close()
                raise RuntimeError('Real Chroma required; fallback vector collections cannot run formal experiments')
            extractor, embedder, manager = self.component_factory(self.settings, storage, vectors, directory)
            self.handles[key] = (storage, vectors, extractor, embedder, manager, directory)
        return self.handles[key]

    @staticmethod
    def _meta(storage, key, default=''):
        row = storage.connection.execute('SELECT value FROM eval_meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    @staticmethod
    def _set_meta(storage, key, value):
        storage.connection.execute('INSERT OR REPLACE INTO eval_meta VALUES (?,?)', (key, str(value)))
        storage.commit()

    def _chunks(self, text):
        # Split at Unicode characters, never between UTF-8 bytes in a token.
        limit = self.settings['memory']['input_chunk_tokens']
        remaining = text
        while remaining:
            lo, hi = 1, len(remaining)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(self.tokenizer.encode(remaining[:mid], disallowed_special=())) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            yield remaining[:lo]
            remaining = remaining[lo:]

    def add(self, user_id, messages, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('A stable session_id is required for ordered, idempotent ingestion')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty list')
        clean = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get('content'), str):
                raise ValueError('Each message requires string content')
            # Explicit whitelist: gold/evidence labels never reach memory models.
            clean.append({k: message[k] for k in ['role', 'name', 'content', 'chat_time', 'tools'] if k in message})
        payload_hash = digest(clean)
        with self.lock:
            row = self._row(user_id, create=True)
            if row['mode'] != 'train':
                raise PermissionError('Persistent test memory is frozen')
            storage, vectors, extractor, embedder, manager, directory = self._open(row)
            if self._meta(storage, 'quarantined'):
                raise RuntimeError('Namespace quarantined after partial ingestion; reset and replay the complete history')
            old = storage.connection.execute('SELECT * FROM eval_sessions WHERE session_id=?', (session_id,)).fetchone()
            if old:
                if old['payload_hash'] != payload_hash:
                    raise ValueError('session_id already exists with different content')
                if old['completed']:
                    return {'status': 'already_ingested', 'messages': len(clean)}
                raise RuntimeError('Interrupted ingestion detected; reset and replay this namespace')
            storage.connection.execute('INSERT INTO eval_sessions VALUES (?,?,0)', (session_id, payload_hash))
            storage.commit()
            touched = set()
            qa_count = 0
            try:
                for index, message in enumerate(clean):
                    if not message['content'].strip():
                        continue
                    event_time = format_timestamp(message['chat_time']) if message.get('chat_time') else None
                    previous_time = self._meta(storage, 'last_event_time')
                    if event_time and previous_time and event_time < previous_time:
                        raise ValueError('Sessions must be ingested in chronological order')
                    role = message.get('role', 'user')
                    speaker = message.get('name') or role
                    # Unknown dates retain an explicit label. The sentinel is only
                    # an internal storage requirement, never shown as event time.
                    stored_time = event_time or '1970-01-01 00:00:00'
                    for part, text in enumerate(self._chunks(message['content'])):
                        source = digest([session_id, index, part])
                        record_text = f'[{speaker}] {text}'
                        extraction = extractor.extract(record_text, context=f'Session: {session_id}; event time: {event_time or "unknown"}')
                        topics = extraction if isinstance(extraction, list) else [extraction]
                        ids = []
                        for topic in topics:
                            result = manager.add_qa(topic, record_text, tools=message.get('tools'),
                                                    timestamp=stored_time, state_key='evaluation', source_id=source)
                            ids.append(result['qa_id'])
                            touched.add(result['experience_id'])
                            qa_count += 1
                        storage.connection.execute('INSERT INTO eval_events VALUES (?,?,?,?,?)',
                                                   (source, session_id, event_time, digest(text), json.dumps(ids)))
                        storage.commit()
                    if event_time:
                        self._set_meta(storage, 'last_event_time', event_time)
                self._finalize_handle(storage, manager, touched)
                storage.connection.execute('UPDATE eval_sessions SET completed=1 WHERE session_id=?', (session_id,))
                storage.commit()
                return {'status': 'ingested', 'messages': len(clean), 'qas': qa_count}
            except Exception as exc:
                storage.rollback()
                self._set_meta(storage, 'quarantined', type(exc).__name__)
                raise

    def _finalize_handle(self, storage, manager, experience_ids, *, complete=False):
        for exp_id in sorted(experience_ids):
            exp = storage.get_experience(exp_id)
            for segment in storage.list_latest_segments(exp_id, self.settings['memory']['experience_summary_segment_threshold'] + 1):
                manager._summarize_segment(segment, segment['updated_at'], 'evaluation_finalize')
            manager._summarize_experience(exp, exp['updated_at'], 'evaluation_finalize')
            if complete:
                exp = storage.get_experience(exp_id)
                manager._mark_experience_completed(exp, exp['updated_at'])
        storage.commit()

    def finalize(self, user_id, complete=False):
        with self.lock:
            row = self._row(user_id, create=True)
            if row['mode'] == 'test':
                raise PermissionError('Cannot finalize frozen test memory')
            storage, _, _, _, manager, _ = self._open(row)
            if self._meta(storage, 'quarantined'):
                raise RuntimeError('Cannot finalize quarantined namespace')
            ids = [r[0] for r in storage.connection.execute("SELECT experience_id FROM experience_memory WHERE status='open'")]
            self._finalize_handle(storage, manager, ids, complete=complete)
            return {'status': 'ready', 'experiences': len(ids)}

    def search(self, user_id, query, top_k=20, question_date=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be nonempty')
        if type(top_k) is not int or not 1 <= top_k <= self.settings['retrieval']['max_top_k']:
            raise ValueError('top_k outside configured range')
        with self.lock:
            row = self._row(user_id)
            if row is None:
                raise KeyError('Unknown namespace: ingestion has not run')
            storage, vectors, _, embedder, _, directory = self._open(row)
            if self._meta(storage, 'quarantined') or storage.connection.execute('SELECT 1 FROM eval_sessions WHERE completed=0 LIMIT 1').fetchone():
                raise RuntimeError('Cannot search incomplete/quarantined memory')
            started = time.perf_counter()
            cfg = self.settings['retrieval']
            encoded_query = f'Question date: {question_date}\n{query}' if question_date else query
            embedding = embedder.embed(encoded_query)
            hits = vectors.query(embedding, 'experience', cfg['experience_candidates'])
            experiences = [storage.get_experience(x['metadata']['experience_id']) for x in hits]
            experiences = [e for e in experiences if e and e['status'] != 'deleted']
            segment_hits = []
            for experience in experiences:
                segment_hits.extend(vectors.query(embedding, 'segment', cfg['segments_per_experience'],
                                                  {'experience_id': experience['experience_id']}))
            segment_hits.sort(key=lambda h: (h['distance'], h['metadata']['segment_id']))
            segments = [storage.get_segment(h['metadata']['segment_id']) for h in segment_hits]
            segments = [s for s in segments if s and s['status'] != 'deleted']
            qa_hits = []
            for segment in segments:
                qa_hits.extend(vectors.query(embedding, 'qa', cfg['qas_per_segment'], {'segment_id': segment['segment_id']}))
            qa_hits.sort(key=lambda h: (h['distance'], h['metadata']['qa_id']))
            qas = [storage.get_qa(h['metadata']['qa_id']) for h in qa_hits[:top_k]]
            qas = [q for q in qas if q and q['status'] == 'open']
            blocks = []
            if cfg['include_experience']:
                for exp in experiences:
                    blocks.append(('experience', exp['experience_id'], json.dumps({
                        'level': 'Experience', 'id': exp['experience_id'], 'topic': exp['topic'],
                        'summary': exp['summary'], 'historical_experience': exp.get('history_experience'),
                        'updated_at': exp['updated_at']}, ensure_ascii=False)))
            for segment in segments:
                blocks.append(('segment', segment['segment_id'], json.dumps({
                    'level': 'Segment', 'id': segment['segment_id'], 'experience_id': segment['experience_id'],
                    'summary': segment['summary'], 'updated_at': segment['updated_at']}, ensure_ascii=False)))
            for qa in qas:
                event = storage.connection.execute('SELECT event_time,session_id FROM eval_events WHERE source_id=?', (qa.get('source_id'),)).fetchone()
                blocks.append(('qa', qa['qa_id'], json.dumps({
                    'level': 'QA', 'id': qa['qa_id'], 'source_id': qa.get('source_id'),
                    'session_id': event['session_id'] if event else None,
                    'event_time': event['event_time'] if event else None,
                    'content': qa['user_input'], 'assistant': qa['assistant_output']}, ensure_ascii=False)))
            # Budgets reserve room for raw evidence, rather than letting broad
            # summaries consume the entire context. All three stages are bounded.
            total = cfg['context_tokens']
            quotas = {'experience': int(total * .25), 'segment': int(total * .35), 'qa': total}
            used = {'experience': 0, 'segment': 0, 'qa': 0}
            selected, emitted = [], []
            for level, item_id, text in blocks:
                remaining = min(total - sum(used.values()), quotas[level] - used[level])
                if remaining <= 1:
                    continue
                tokens = self.tokenizer.encode(text + '\n', disallowed_special=())
                piece = self.tokenizer.decode(tokens[:remaining])
                selected.append(piece)
                emitted.append({'level': level, 'id': item_id, 'truncated': len(tokens) > remaining})
                used[level] += len(self.tokenizer.encode(piece, disallowed_special=()))
            context = ''.join(selected)
            tokens = self.tokenizer.encode(context, disallowed_special=())
            if len(tokens) > total:
                context = self.tokenizer.decode(tokens[:total])
            elapsed = (time.perf_counter() - started) * 1000
            result = {'context': context, 'context_tokens': len(self.tokenizer.encode(context, disallowed_special=())),
                      'retrieval_ms': elapsed, 'evidence': emitted, 'candidate_counts': {
                          'experiences': len(experiences), 'segments': len(segments), 'qas': len(qas)},
                      'query_date': question_date, 'config_fingerprint': self.settings['config_fingerprint']}
            # Telemetry is separate from the immutable business memory database.
            with (directory / 'searches.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'query_hash': digest(query), **result}, ensure_ascii=False) + '\n')
            return result

    def delete(self, user_id):
        with self.lock:
            row = self._row(user_id, create=True)
            if row['mode'] == 'test':
                raise PermissionError('Cannot clear frozen test memory')
            # Logical reset to a new generation. Old artifacts are retained for
            # audit/recovery, never recursively deleted while Chroma is open.
            self.catalog.execute('UPDATE namespaces SET generation=? WHERE id=?', (uuid.uuid4().hex, row['id']))
            self.catalog.commit()
            return {'status': 'reset'}

    def set_mode(self, user_id, mode):
        if mode not in ['train', 'test']:
            raise ValueError('mode must be train or test')
        with self.lock:
            row = self._row(user_id, create=True)
            self.catalog.execute('UPDATE namespaces SET mode=? WHERE id=?', (mode, row['id']))
            self.catalog.commit()
            return {'mode': mode}

    def snapshot(self, user_id):
        with self.lock:
            row = self._row(user_id)
            if row is None:
                raise KeyError('Unknown namespace')
            storage, vectors, _, _, _, _ = self._open(row)
            if self._meta(storage, 'quarantined') or storage.connection.execute('SELECT 1 FROM eval_sessions WHERE completed=0 LIMIT 1').fetchone():
                raise RuntimeError('Incomplete namespace cannot be snapshotted')
            snapshot_id = uuid.uuid4().hex
            dest = self.root / 'snapshots' / snapshot_id
            dest.mkdir(parents=True)
            with sqlite3.connect(dest / 'memory.sqlite3') as backup:
                storage.connection.backup(backup)
            count = vectors.count()
            sha = hashlib.sha256()
            with (dest / 'vectors.jsonl').open('wb') as f:
                for offset in range(0, count, 512):
                    records = vectors.collection.get(limit=512, offset=offset, include=['documents', 'embeddings', 'metadatas'])
                    for i, key in enumerate(records['ids']):
                        value = {name: records[name][i] for name in ['documents', 'embeddings', 'metadatas']}
                        if hasattr(value['embeddings'], 'tolist'):
                            value['embeddings'] = value['embeddings'].tolist()
                        line = (json.dumps({'id': key, **value}, ensure_ascii=False) + '\n').encode()
                        f.write(line)
                        sha.update(line)
            manifest = {'snapshot_id': snapshot_id, 'namespace': row['id'], 'vector_count': count,
                        'vectors_sha256': sha.hexdigest(), 'sqlite_sha256': hashlib.sha256((dest / 'memory.sqlite3').read_bytes()).hexdigest(),
                        'config_fingerprint': self.settings['config_fingerprint']}
            write_json(dest / 'manifest.json', manifest)
            return manifest

    def restore(self, user_id, snapshot_id):
        if not isinstance(snapshot_id, str) or len(snapshot_id) != 32 or any(c not in '0123456789abcdef' for c in snapshot_id):
            raise ValueError('Invalid snapshot_id')
        with self.lock:
            src = self.root / 'snapshots' / snapshot_id
            manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
            row = self._row(user_id, create=True)
            if manifest['namespace'] != row['id'] or manifest['config_fingerprint'] != self.settings['config_fingerprint']:
                raise ValueError('Snapshot namespace/config mismatch')
            for filename, key in [('memory.sqlite3', 'sqlite_sha256'), ('vectors.jsonl', 'vectors_sha256')]:
                sha = hashlib.sha256()
                with (src / filename).open('rb') as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        sha.update(chunk)
                if sha.hexdigest() != manifest[key]:
                    raise ValueError(f'Snapshot integrity check failed: {filename}')
            generation = uuid.uuid4().hex
            dest = self.root / 'namespaces' / row['id'] / generation
            dest.mkdir(parents=True)
            with sqlite3.connect(src / 'memory.sqlite3') as source, sqlite3.connect(dest / 'memory.sqlite3') as target:
                source.backup(target)
            proposed = {'id': row['id'], 'generation': generation, 'mode': row['mode']}
            _, vectors, _, _, _, _ = self._open(proposed)
            with (src / 'vectors.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    vectors.collection.upsert(ids=[item['id']], documents=[item['documents']],
                        embeddings=[item['embeddings']], metadatas=[item['metadatas']])
            if vectors.count() != manifest['vector_count']:
                raise RuntimeError('Restored vector count mismatch')
            self.catalog.execute('UPDATE namespaces SET generation=? WHERE id=?', (generation, row['id']))
            self.catalog.commit()
            return {'status': 'restored', 'snapshot_id': snapshot_id}

    def health(self):
        return {'status': 'ok', 'profile': self.settings['profile'],
                'config_fingerprint': self.settings['config_fingerprint'],
                'capabilities': ['namespace_isolation', 'readonly_search', 'idempotent_sessions', 'finalize', 'snapshot', 'restore', 'frozen_test_memory'],
                'context_tokens': self.settings['retrieval']['context_tokens']}
