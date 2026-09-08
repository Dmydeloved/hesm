"""Reuse HESM's extraction, summaries and manager with explicit dependencies.

Evaluation fails on model errors and preserves complete new evidence; it never
silently substitutes deterministic summaries or production model credentials.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.extractor import TopicExtractor
from core.manager import MemoryManager
from core.summarizer import LLMSummarizer, strip_markdown_code_fence
from core.prompts.topic_memory import build_segment_summary_prompt, build_experience_summary_prompt, build_historical_experience_prompt
from core.vector_store import build_vector_document, build_vector_metadata
from core.vector_store import ChromaVectorStore
from experiments.settings import connection


class EvaluationVectors(ChromaVectorStore):
    """Real Chroma only, with explicit telemetry and per-generation collections."""
    def __init__(self, persist_path, collection_name):
        import chromadb
        from chromadb.config import Settings
        self.persist_path = Path(persist_path)
        self.client = chromadb.PersistentClient(path=str(self.persist_path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})


class UsageClient:
    """Small OpenAI facade recording usage without logging credentials/content."""
    def __init__(self, client: Any, sink: Path, stage: str):
        self.client, self.sink, self.stage = client, sink, stage
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage.model_dump() if response.usage else None
        self.sink.parent.mkdir(parents=True, exist_ok=True)
        with self.sink.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'stage': self.stage, 'requested_model': kwargs['model'],
                                'returned_model': response.model, 'usage': usage}) + '\n')
        return response


class StrictSummarizer(LLMSummarizer):
    def _generate_summary(self, prompt, *, fallback='', task='summary'):
        response = self.client.chat.completions.create(
            model=self.model, messages=[{'role': 'user', 'content': prompt}], temperature=0)
        value = json.loads(strip_markdown_code_fence(response.choices[0].message.content or ''))
        if not isinstance(value, dict) or not value:
            raise ValueError(f'{task} returned an empty/non-object summary')
        return json.dumps(value, ensure_ascii=False)

    def summarize_segment(self, segment, qa_items):
        # Preserve all newly supplied messages; the production normalizer keeps
        # only five QAs and truncates assistant text, which loses benchmark facts.
        info = {k: v for k, v in segment.items() if k != 'qa_ids'}
        return self._generate_summary(build_segment_summary_prompt(info, qa_items), task='segment')

    def summarize_experience(self, experience, segments):
        info = {k: v for k, v in experience.items() if k not in {'segment_ids', 'history_experience'}}
        evidence = [{k: v for k, v in s.items() if k != 'qa_ids'} for s in segments]
        return self._generate_summary(build_experience_summary_prompt(info, evidence), task='experience')


class ExplicitEmbedder:
    def __init__(self, client, model, sink):
        self.client, self.model, self.sink = client, model, sink

    @lru_cache(maxsize=1024)
    def _embed(self, text):
        result = self.client.embeddings.create(model=self.model, input=text)
        with self.sink.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'stage': 'embedding', 'requested_model': self.model,
                                'usage': result.usage.model_dump() if result.usage else None}) + '\n')
        return tuple(result.data[0].embedding)

    def embed(self, text):
        return list(self._embed(text))


class ConfiguredRecaller:
    def __init__(self, storage, vectors, embedder, summarizer, limit=3):
        self.storage, self.vectors = storage, vectors
        self.embedder, self.summarizer, self.limit = embedder, summarizer, limit

    def recall(self, topic, core_entity, query, intent=''):
        hits = self.vectors.query(self.embedder.embed(f'{topic}\n{core_entity}\n{query}'),
                                  'experience', self.limit, {'status': 'completed'})
        experiences = [self.storage.get_experience(x['metadata']['experience_id']) for x in hits]
        experiences = [x for x in experiences if x]
        if not experiences:
            return {'history_experience': {}}
        segments = [s for e in experiences for s in self.storage.list_latest_segments(e['experience_id'], 3)]
        prompt = build_historical_experience_prompt(
            current_topic=topic, current_core_entity=core_entity, current_intent=intent,
            current_context=query, historical_experiences=experiences, historical_segments=segments)
        history = json.loads(self.summarizer._generate_summary(prompt, task='historical_experience'))
        return {'history_experience': history}


class EvaluationManager(MemoryManager):
    """Bounded segments and incremental summaries using the original HESM logic."""
    def _should_cut_segment(self, segment, intent):
        return len(segment.get('qa_ids', [])) >= self.segment_summary_qa_threshold or super()._should_cut_segment(segment, intent)

    def _summarize_segment(self, segment, now, reason):
        ids = segment.get('qa_ids', [])
        if int(segment.get('last_summarized_qa_count', 0)) == len(ids):
            self.storage.update_segment(segment)
            return
        super()._summarize_segment(segment, now, reason)

    def _summarize_experience(self, experience, now, reason):
        ids = experience.get('segment_ids', [])
        start = max(0, int(experience.get('last_summarized_segment_count', 0)) - 1)
        pending = ids[start:]
        width = self.experience_summary_segment_threshold
        for offset in range(0, len(pending), width):
            segments = [self.storage.get_segment(s) for s in pending[offset:offset + width]]
            experience['summary'] = self._summary_object(self.summarizer.summarize_experience(experience, [s for s in segments if s]))
        if pending:
            if (experience.get('summary', {}).get('current_state') or {}).get('status') == 'completed':
                experience['status'] = 'completed'
            experience['last_summarized_segment_count'] = len(ids)
            experience['updated_at'] = now
            experience['version'] = int(experience.get('version', 0)) + 1
            self.storage.update_experience(experience)
            self.upsert_experience_vector(experience['experience_id'])

    def upsert_experience_vector(self, experience_id):
        experience = self.storage.get_experience(experience_id)
        memory = {**experience, 'recent_segments': self.storage.list_latest_segments(experience_id, 3)}
        metadata = build_vector_metadata('experience', experience)
        for kind in ['experience', 'experience_route']:
            text = build_vector_document(kind, memory)
            self.vector_store.upsert(kind, experience_id, text, self.embedder.embed(text), experience['updated_at'], metadata)

    def _find_experience_by_vector(self, topic, core_entity):
        results = self.vector_store.query(self.embedder.embed(f'主题：{topic}\n核心实体：{core_entity}'),
                                          'experience_route', 5, {'status': 'open'})
        for item in results:
            if item['similarity'] > self.experience_similarity_threshold:
                value = self.storage.get_experience(item['metadata']['experience_id'])
                if value and value['status'] == 'open':
                    return value
        return None


def make_components(settings, storage, vectors, directory):
    from openai import OpenAI
    import httpx
    model = connection(settings, 'memory')
    embedding = connection(settings, 'embedding')
    directory.mkdir(parents=True, exist_ok=True)
    sink = directory / 'usage.jsonl'
    trust_env = os.getenv('HESM_EVAL_DIRECT_NETWORK', '0') != '1'
    raw = OpenAI(**{k: v for k, v in model.items() if k != 'model'}, http_client=httpx.Client(trust_env=trust_env))
    summary = StrictSummarizer(api_key=model['api_key'], model=model['model'], base_url=model['base_url'],
                               max_retries=model['max_retries'], retry_delay=1,
                               client=UsageClient(raw, sink, 'memory_summary'))
    extractor = TopicExtractor(api_key=model['api_key'], model=model['model'], base_url=model['base_url'],
                               max_retries=model['max_retries'] + 1, retry_delay=1)
    extractor.client = UsageClient(raw, sink, 'topic_extraction')
    embedder = ExplicitEmbedder(OpenAI(**{k: v for k, v in embedding.items() if k != 'model'},
                                      http_client=httpx.Client(trust_env=trust_env)), embedding['model'], sink)
    recaller = ConfiguredRecaller(storage, vectors, embedder, summary)
    cfg = settings['memory']
    manager = EvaluationManager(storage, vectors, embedder, summary,
        segment_summary_qa_threshold=cfg['segment_qa_limit'],
        experience_summary_segment_threshold=cfg['experience_summary_segment_threshold'],
        experience_similarity_threshold=cfg['experience_similarity_threshold'], experience_recaller=recaller)
    return extractor, embedder, manager
