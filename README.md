# HESM

[中文文档](README_CN.md)

HESM is a hierarchical episodic memory system for conversational agents. It turns structured topic extraction results into a three-level memory graph—Experience, Segment, and QA—and combines structured filtering, vector recall, an LLM reranker, and runtime caches to retrieve evidence for later conversations.

## Memory hierarchy

```text
Experience (long-term topic + core entity / goal)
└── Segment (intent / subgoal)
    └── QA (original user-assistant evidence)
```

- **Experience** groups a long-running goal by `topic` and `core_entity` and maintains linked intents, state, and a rolling summary.
- **Segment** groups consecutive work under one `intent` and summarizes the corresponding subgoal.
- **QA** stores the original user input, assistant output, entities, tools, confidence, timestamp, and reasoning.

## Retrieval pipeline

`HybridRetriever` processes the hierarchy from Experience to Segment to QA:

1. Build one query text and compute its embedding once per `recall()` call.
2. Reuse that query vector for Experience, Segment, and QA vector searches.
3. Combine vector recall with structured candidates from SQLite.
4. Ask the retrieval LLM to select and score candidates at each layer.
5. Reuse cached Experience and Segment IDs when the topic/entity or intent is unchanged.
6. Return structured results plus a ready-to-use `context_text` block.

If query embedding or one vector search fails, retrieval continues with structured candidates. The LLM reranker remains required for candidate scoring.

## Project layout

| Path | Purpose |
| --- | --- |
| `memory/` | extraction, hierarchical memory management, embedding, storage, retrieval, and summarization |
| `prompts/` | extraction, retrieval, and summary prompt templates |
| `configs/config.yaml` | paths, models, endpoints, retry policy, and retrieval limits |
| `scripts/` | MultiWOZ preprocessing, topic extraction, memory building, interactive flow, and evaluation |
| `evaluation/` | evaluation helpers and tests |
| `tests/` | unit tests and local integration scripts |
| `data/` | raw/processed datasets, memory stores, and benchmark outputs |

## Requirements

- Python 3.10+
- `openai`
- `PyYAML`
- `chromadb`

Create an environment and install the runtime dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install openai PyYAML chromadb
```

## Configuration

Edit `configs/config.yaml` and provide the API keys/endpoints used by your environment:

- `topic_extraction`: structured topic extraction LLM
- `embedding`: embedding model used for memory writes and query vectors
- `retrieval`: per-layer retrieval reranker
- `summarization`: Segment and Experience summarization
- `evaluation`: answer generation and judging for end-to-end evaluation

All generated paths are configured under `paths`. To use another config file without editing the repository default, set `TOPIC_MEMORY_CONFIG` to its path.

Do not commit real API keys.

## Quick start

Run the end-to-end topic-memory flow for one user message:

```bash
python scripts/topic_memory_flow.py "Find me a hotel near the city centre" \
  --assistant-output "I can help with that." \
  --wait-memory-write
```

The flow extracts one or more topic records, retrieves existing memory in the foreground, and writes the new QA/Segment/Experience state through a background worker. Use `--verbose` for detailed logs.

## MultiWOZ data pipeline

The default paths below come from `configs/config.yaml`.

```bash
# 1. Convert raw MultiWOZ 2.2 files into scene-dialogue JSONL
python scripts/process_multiwoz_2_2.py

# 2. Extract structured topics (checkpoint/resume is enabled by default)
python scripts/multiwoz_topic_pipeline.py --continue-on-error

# 3. Build SQLite memory and Chroma vectors
python scripts/build_memory_from_multiwoz_topics.py

# 4. Build retrieval benchmark cases
python scripts/build_topic_benchmark.py

# 5. Evaluate hierarchical retrieval
python scripts/evaluate_retrieval.py

# 6. Evaluate end-to-end answer quality
python scripts/evaluate_end_to_end.py
```

Use each script's `--help` output to override input/output paths, limits, state keys, and retrieval sizes.

## Testing

Run the offline unit suite from the project root:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s evaluation/tests -p "test_*.py"
```

`tests/test_retriever.py` is a local integration script rather than an offline unit test. It expects a populated memory database and valid API configuration:

```bash
python tests/test_retriever.py
```

## Main retrieval result

`HybridRetriever.recall()` returns:

- `query`: normalized retrieval input
- `experiences`, `segments`, `qas`: selected records with scores
- `context_text`: formatted memory context for downstream generation
- `debug`: structured/vector candidate counts and cache hits
- `results`: backward-compatible QA/score entries
