# Evidence Topic Retrieval v2

This version keeps the experiment isolated from production and previous runs:

- Config: `experiments/config/evidence_topic_retrieval_v2.yaml`
- Runtime root: `experiments/.runtime/evidence_topic_retrieval_v2/<dataset_id>/`
- Each user's HESM database and Chroma store are under a SHA-256 user directory.
- Native HESM `add_memory` and `retrieve` calls write their Add/Retriever
  trajectories to that user's `hesm.sqlite3`; experiment request summaries
  remain in the dataset's `requests.sqlite3`.

Set `experiment.retrieval_topic_source` in the config to either:

- `llm`: always use HESM's native topic extractor.
- `evidence_topic_lookup`: look up topic/entity fields for only the first
  evidence key in the current user's SQL store. Empty evidence, missing rows,
  malformed lookup data, or a SQLite lookup error falls back to native LLM
  extraction and does not fail the search.

The run version is also part of the dataset ID, so use a new version name for
each independent run. Example:

```powershell
python -B -m experiments.run_user_memory `
  --benchmark locomo `
  --config experiments/config/evidence_topic_retrieval_v2.yaml `
  --env experiments/.env.user_memory `
  --version evidence_topic_fallback_v2 `
  --omnimemeval D:\code\OmniMemEval `
  --workers 1 --llm-workers 2
```

To use LLM extraction for every query, change `retrieval_topic_source` to
`llm`. The generated result manifest records the config fingerprint and the
dataset ID for reproducibility.
