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

The main implementation is under `memory/`:

- `manager.py`: creates and updates the three memory levels.
- `storage.py`: SQLite persistence and relational lookup.
- `vector_store.py`: Chroma-compatible vector storage, with a local fallback.
- `extractor.py`: topic, entity, intent, and confidence extraction.
- `summarizer.py`: template or LLM summaries.
- `retriever.py`: hierarchical recall, fallback routing, and reranking.

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

Configure model roles in `configs/config.yaml`. Keep real API keys outside
version control. The relevant sections are:

- `topic_extraction`;
- `answer_generation`;
- `embedding`;
- `retrieval`;
- `summarization`;
- `evaluation`.

Experiment paths, enabled methods, top-K values, and ablation variants are in
`experiments/locomo/config/experiment.yaml`.

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
```

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
outputs/locomo/logs/<method>.log
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
outputs/locomo/answers/<method>_<conv_id>.json
```

A query is marked complete only when retrieval, answer generation, and Judge
all succeed. On rerun:

- successful queries are skipped immediately;
- retrieval, empty-answer, and Judge failures are retried;
- failed records and stage error messages remain in the checkpoint and log;
- legacy checkpoints are considered successful only when the prediction is
  non-empty and `judge_score >= 0`.

This makes an interrupted experiment resumable without paying for successful
queries again.

## Output layout

```text
outputs/locomo/
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
