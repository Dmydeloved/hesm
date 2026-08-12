# HESM

[中文文档](README_CN.md)

HESM is a hierarchical long-term memory system organized as **Experience →
Segment → QA**. It turns conversation turns into traceable structured memories,
maintains summaries at multiple levels, and retrieves a bounded hierarchy for
answer generation. This repository also contains a reproducible LoCoMo
benchmark pipeline with baselines, ablations, checkpoint resume, four-stage
logs, and report generation.

## Architecture

```text
Conversation turns
      │
      ▼
Topic / entity / intent extraction
      │
      ▼
Experience ── long-running topic and state
      └── Segment ── coherent intent or event phase
            └── QA ── original, traceable evidence turn
      │
      ▼
Structured + vector candidate recall
      │
      ▼
Optional LLM hierarchical reranking
      │
      ▼
Bounded Experience + Segment + QA context
```

The main implementation is under `hesm/`:

- `manager.py`: creates and updates the three memory levels.
- `storage.py`: SQLite persistence and relational lookup.
- `vector_store.py`: Chroma-compatible vector storage, with a local fallback.
- `extractor.py`: topic, entity, intent, and confidence extraction.
- `summarizer.py`: template or LLM summaries.
- `retriever.py`: hierarchical recall, fallback routing, and reranking.

Runtime concerns are deliberately separated:

- `config/hesm.yaml`: production HESM configuration only;
- `memory/`: production SQLite and Chroma memory only;
- `experiments/config/locomo.yaml`: self-contained experiment configuration;
- `experiments/outputs/`: all generated benchmark memories and results.

## LoCoMo benchmark

The benchmark compares four methods in the one-click suite:

| Method | Description |
|---|---|
| `vector_rag` | Flat vector search over raw dialogue turns |
| `mem0` | Mem0 memory and retrieval adapter |
| `amem` | A-MEM-style linked atomic memories |
| `hesm` | Full Experience → Segment → QA hierarchy |

`full_context` remains available from `run_main`, but is intentionally excluded
from `run_all` because the one-click comparison focuses on memory systems.

The default dataset path is `data/locomo10.json`. LoCoMo categories 1–4 are the
official evaluation set. Category 5 contains adversarial questions and should
not be mixed into the normal QA aggregate unless a dedicated evaluation policy
is defined.

### Metrics

- token-level answer precision, recall, and F1;
- binary LLM-as-a-Judge correctness;
- evidence Recall, Precision, F1, and Accuracy at K = 1, 3, 5;
- retrieved context tokens and compression ratio;
- per-query retrieval, answer, and Judge latency in checkpoints and logs.

## Requirements

- Python 3.10 or newer;
- an OpenAI-compatible chat completion endpoint;
- an embedding endpoint compatible with the configured provider;
- SQLite;
- ChromaDB and Mem0 for their corresponding benchmark methods.

Install the runtime packages in your environment:

```bash
pip install openai pyyaml tiktoken chromadb mem0ai
```

Production HESM reads only `config/hesm.yaml`. Keep real API keys outside
version control. Its relevant sections are:

- `topic_extraction`;
- `embedding`;
- `retrieval`;
- `summarization`;
- `memory_management`;
- `api`.

Experiments read only the self-contained `experiments/config/locomo.yaml`.
That file owns experiment model roles, paths, enabled methods, top-K values,
and ablation variants; experiment code never loads the production config.

## Public interfaces

The Python API exposes the two production operations through `HESMService`:

```python
from hesm import HESMService

service = HESMService()
service.add_memory(user_input="Alice moved to Paris.")
result = service.retrieve(question="Where did Alice move?")
```

The HTTP server exposes the same operations as `POST /api/memories` and
`POST /api/retrieve`:

```bash
python -m frontend.server
```

### Models used by Mem0, A-MEM, and HESM

The current configuration is:

| Component | Configuration section | Current model |
|---|---|---|
| Mem0 internal extraction/update LLM | `memory_methods.mem0.llm` | `gpt-5.4-nano-high` |
| Mem0 embeddings | `memory_methods.mem0.embedding` | `text-embedding-v4` |
| A-MEM keyword/context LLM | `memory_methods.amem.llm` | `gpt-5.4-nano-high` |
| A-MEM embeddings | `memory_methods.amem.embedding` | `text-embedding-v4` |
| HESM topic extraction | `memory_methods.hesm.topic_extraction` | `gpt-5.4-nano-high` |
| HESM hierarchical summarization | `memory_methods.hesm.summarization` | `gpt-5.4-nano-high` |
| HESM retrieval reranking | `memory_methods.hesm.retrieval` | `gpt-5.4-nano-high` |
| HESM embeddings | `memory_methods.hesm.embedding` | `text-embedding-v4` |
| Shared final answer generator | `answer_generation` | `gpt-5.4-mini` |
| Shared LLM Judge | `evaluation` | `gpt-5.4-mini` |

Each adapter reads its own `memory_methods` subsection from the experiment
configuration. YAML profiles centralize
the shared connection, model, retry, and embedding settings, while the existing
role paths remain available to the code. Change a profile once to update every
role that references it, or replace a method alias with a dedicated mapping.

## Running experiments

Run commands from the repository root.

### One-click suite

```bash
python -m experiments.locomo.run_all
```

This runs:

1. the main comparison: Vector RAG, Mem0, A-MEM, and HESM;
2. the HESM ablation study;
3. the cache latency evaluation.

Useful options:

```bash
# Quick run on the first conversation
python -m experiments.locomo.run_all --max-conversations 1

# Run only selected main methods
python -m experiments.locomo.run_all --methods vector_rag hesm

# Skip expensive parts
python -m experiments.locomo.run_all --skip-parts ablation cache

# Recommended: methods serial, four concurrent QA workers per method
python -m experiments.locomo.run_all --method-workers 1 --qa-workers 4
```

`method_workers` controls how many methods run at the same time. It defaults to
1 so latency numbers are not distorted by competition between methods.
`qa_workers` controls concurrent questions after a conversation memory is
ready; the default is 4. Every QA worker owns its retrieval reader, answer
client, and Judge client, while checkpoint writes stay in the parent thread.
Use `--method-workers 4 --qa-workers 1` if total completion time matters more
than isolated per-method latency. Avoid multiplying both values aggressively:
the approximate peak request concurrency is their product.

### Main experiment only

```bash
python -m experiments.locomo.run_main
python -m experiments.locomo.run_main --methods mem0 hesm
python -m experiments.locomo.run_main --max-conversations 2
```

To run the optional full-context upper bound independently:

```bash
python -m experiments.locomo.run_main --methods full_context
```

### Ablation study

```bash
python -m experiments.locomo.run_ablation
```

The complete ablation set is:

| Variant | Topic extraction | Memory layers | LLM reranker |
|---|---:|---|---:|
| `flat_memory` | No | Raw turns | No |
| `qa_only` | No at query time | QA | No |
| `qa_segment` | No at query time | QA + Segment | No |
| `full_hesm_no_reranker` | Yes | Experience + Segment + QA | No |
| `full_hesm` | Yes | Experience + Segment + QA | Yes |

Run a subset with:

```bash
python -m experiments.locomo.run_ablation \
  --variants qa_only qa_segment full_hesm_no_reranker full_hesm
```

QA and hierarchy variants reuse the HESM store created by the main experiment.
If it is missing, the standalone ablation runner builds it automatically.

### Cache evaluation

```bash
python -m experiments.locomo.run_cache --num-samples 50
```

## Four-stage method logs

Every main method and ablation variant writes one structured log file:

```text
experiments/outputs/locomo/logs/<method>.log
```

Each line is a UTF-8 JSON object. The `stage` field divides events into four
parts:

1. `BUILD`: memory creation or existing-store attachment;
2. `RETRIEVAL`: retrieved IDs, token count, context preview, and latency;
3. `ANSWER`: generated answer and latency;
4. `JUDGE`: ground truth, prediction, score, and latency.

Every part emits `STARTED` and `SUCCESS`, or `FAILED` with an error reason.
Builds that finish with item-level failures emit `PARTIAL` and the captured reasons.
Downstream stages emit `SKIPPED` when an earlier stage fails. Logs never include
API keys or complete configuration secrets.

Example log event:

```json
{
  "method": "hesm",
  "stage": "RETRIEVAL",
  "status": "SUCCESS",
  "conv_id": "conv-26",
  "query_index": 0,
  "retrieved_ids": ["D1:3"],
  "retrieved_tokens": 1082,
  "elapsed_ms": 18320.4
}
```

## Resume and retry behavior

Checkpoints are stored per method and conversation:

```text
experiments/outputs/locomo/answers/<method>_<conv_id>.json
```

A query is marked complete only when retrieval, answer generation, and Judge
all succeed. On rerun:

- successful queries are skipped immediately;
- retrieval, empty-answer, and Judge failures are retried;
- failed records and stage error messages remain in the checkpoint and log;
- legacy checkpoints are considered successful only when the prediction is
  non-empty and `judge_score >= 0`.
- Mem0 and A-MEM persist completed source `dia_id` values in
  `experiments/outputs/locomo/memory/mem0_<conv_id>/build_state.json` and
  `experiments/outputs/locomo/memory/amem_<conv_id>/build_state.json`; complete stores skip all memory-add
  calls, while partial stores add only missing turns.
- HESM validates stored `dia_id` values in SQLite: a complete store skips
  reconstruction, while a partial store processes only missing turns.

This makes an interrupted experiment resumable without paying for successful
queries again.

## Output layout

```text
experiments/outputs/locomo/
├── answers/   # per-method, per-conversation checkpoints and QA records
├── logs/      # one four-stage log per method or ablation variant
├── memory/    # isolated SQLite and vector stores
├── metrics/   # aggregated method metrics
└── tables/    # Markdown, CSV, and JSON result tables
```

Main tables are written as `main_results.{md,csv,json}`. Ablation and cache
tables use `ablation_results.*` and `cache_results.*`.

## Testing

The resume and logging tests use only the Python standard library:

```bash
python -m unittest tests.locomo.test_runner_resume_logging -v
```

Run the complete unittest discovery suite with:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility notes

- Run all compared methods with the same configuration snapshot and model
  versions.
- Do not compare partial runs with different Judge success sets.
- Keep category 5 outside the standard category 1–4 aggregate.
- Record the Git commit, configuration snapshot, and environment for published
  experiments.
- Evidence IDs evaluate traceable original turns. Summary-only evidence should
  retain source `dia_id` links if it is to be included in Evidence@K.

## Data license

The LoCoMo dataset has its own upstream license. Follow the upstream attribution
and non-commercial-use requirements when distributing or publishing dataset
artifacts. Repository code and dataset artifacts should not be assumed to share
the same license.
