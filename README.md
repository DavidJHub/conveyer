# conveyer

Clustering & analysis of skincare LLM conversations (English). Segments
conversations and measures how recommendation behaviour shows up per segment —
recommendation rate, brand amplification, and the top brands the LLM
introduces when asked for a recommendation.

## Layout

```
conveyer/
  config.py      PipelineConfig — column mapping, model choices, hyper-parameters
  ingest.py      load/parse data, derive features, English synthetic-data fallback
  models.py      embedding backends (Voyage > OpenAI > sentence-transformers > TF-IDF) + Anthropic LLM
  clustering.py  KMeans (+ silhouette sweep, stability) and BERTopic/HDBSCAN
  analysis.py    brand/recommendation features, intent, amplification, cluster summary, naming
  pipeline.py    end-to-end orchestration + CLI
notebooks/       exploratory notebook version of the same pipeline
```

## Quickstart

```bash
pip install -r requirements.txt          # core is enough; embedding/LLM backends optional

# synthetic data, best available embedding backend (auto-resolved)
python -m conveyer.pipeline

# real data
python -m conveyer.pipeline --data data/conversations.parquet --n-clusters 12
```

```python
from conveyer import PipelineConfig, run_pipeline

results = run_pipeline(PipelineConfig(data_path="data/conversations.parquet"))
results["summary"]                       # per-cluster characterisation
results["top_recommendations"]           # top LLM recommendations on reco/comparison turns
```

## Best model for this task

`embedding_backend="auto"` resolves, in order: **Voyage `voyage-3-large`**
(Anthropic-recommended, best-in-class English) if `VOYAGE_API_KEY` is set →
OpenAI `text-embedding-3-large` if `OPENAI_API_KEY` is set → a strong open model
via `sentence-transformers` → TF-IDF (always works, no deps). Cluster naming and
zero-shot intent use Claude (`ANTHROPIC_API_KEY`, model via `ANTHROPIC_MODEL`).

Outputs are written to `outputs/` as CSVs. Without a dataset the pipeline runs on
English synthetic conversations so you can validate it before plugging in real data.
```
