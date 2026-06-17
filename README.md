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
  clustering.py  method registry (kmeans/agglomerative/spectral/gmm/hdbscan), method
                 comparison, BERTopic, and an LLM-assisted layer (keyphrase expansion
                 + ClusterLLM-style granularity)
  analysis.py    brand/recommendation features, intent, amplification, cluster summary, naming
  viz.py         interactive Plotly dashboard (from memory or from saved CSVs)
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
from conveyer import PipelineConfig, run_pipeline, viz

results = run_pipeline(PipelineConfig(data_path="data/conversations.parquet"))
results["summary"]                       # per-cluster characterisation
results["method_comparison"]             # clustering methods scored against each other
results["top_recommendations"]           # top LLM recommendations on reco/comparison turns

viz.build_dashboard(results)             # interactive outputs/dashboard.html (from memory)
viz.build_dashboard(outputs_dir="outputs")   # ...or rebuilt from the saved CSVs
```

## Clustering beyond k-means

k-means assumes convex, equal-size, spherical clusters and a fixed `k` — a poor fit
for the embedding geometry of short conversational text. `cluster_method="auto"`
compares several methods and picks the best by cosine silhouette:

| method | why it can beat k-means |
|---|---|
| `agglomerative` (cosine, average linkage) | non-convex clusters; cut the hierarchy at any granularity |
| `spectral` | groups via the affinity graph, not Euclidean centroids |
| `gmm` | soft, elliptical clusters |
| `hdbscan` | density-based: no fixed `k`, explicit noise cluster |

`method_comparison.csv` reports silhouette, Davies-Bouldin, Calinski-Harabasz and
(if NIQ is present) ARI/NMI per method, so the choice is evidence-based.

**Reducing reliance on the embedder** (the harder problem for short text). Two
optional LLM-assisted steps, following ClusterLLM (Zhang et al., EMNLP 2023) and
few-shot clustering (Viswanathan et al., TACL), need `ANTHROPIC_API_KEY`:

- `--keyphrase-expansion`: the LLM enriches each short query with salient keyphrases
  before embedding, so the representation depends less on the base embedder.
- `--llm-granularity`: ClusterLLM-style — the LLM answers same/different on borderline
  pairs and the pipeline picks the hierarchy granularity that best agrees.

## Best model for this task

`embedding_backend="auto"` resolves, in order: **Voyage `voyage-3-large`**
(Anthropic-recommended, best-in-class English) if `VOYAGE_API_KEY` is set →
OpenAI `text-embedding-3-large` if `OPENAI_API_KEY` is set → a strong open model
via `sentence-transformers` → TF-IDF (always works, no deps). Cluster naming and
zero-shot intent use Claude (`ANTHROPIC_API_KEY`, model via `ANTHROPIC_MODEL`).

Outputs are written to `outputs/` as CSVs. Without a dataset the pipeline runs on
English synthetic conversations so you can validate it before plugging in real data.
```
