"""Central configuration for the conveyer clustering/analysis pipeline.

All modules read from a single :class:`PipelineConfig` instance so the column
mapping, model choices and hyper-parameters live in one place. Everything is
tuned for **English** skincare-conversation data.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineConfig:
    # --- Data ---
    data_path: str = "data/conversations.parquet"  # .parquet / .csv / .jsonl / .json (autodetected)
    use_synthetic_if_missing: bool = True
    synthetic_n: int = 1500

    # --- Column mapping (adjust to your real schema) ---
    col_question: str = "question"
    col_answer: str = "answer"
    col_brands_q: str = "brands_in_question"
    col_brands_a: str = "brands_in_answer"
    col_niq: str = "skin_care_categories"  # reference taxonomy used for external validation
    col_session: str = "session_id"
    col_session_pos: str = "session_pos"

    # --- Corpus to cluster ---
    # "question" -> user intent | "answer" -> LLM behaviour | "qa" -> both concatenated
    cluster_on: str = "question"
    first_turn_only: bool = True

    # --- Embeddings ---
    # "auto" resolves the best available backend at runtime:
    #   Voyage (VOYAGE_API_KEY) -> OpenAI (OPENAI_API_KEY) -> sentence-transformers -> TF-IDF.
    embedding_backend: str = "auto"  # auto | voyage | openai | sentence_transformers | tfidf
    voyage_model: str = "voyage-3-large"          # Anthropic-recommended embeddings, best-in-class English
    openai_model: str = "text-embedding-3-large"  # strong API alternative
    st_model: str = "BAAI/bge-large-en-v1.5"      # strong open English model (alt: Qwen/Qwen3-Embedding-0.6B)
    normalize_embeddings: bool = True
    embed_batch_size: int = 128

    # --- Clustering ---
    n_clusters: int = 12                       # 'n' for KMeans (explicit "n groups" request)
    k_sweep: Tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20)
    primary_label: str = "cluster_kmeans"      # cluster_kmeans | cluster_bertopic
    # BERTopic / HDBSCAN
    min_cluster_size: int = 40
    min_samples: int = 8
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    random_state: int = 42

    # --- LLM (cluster naming / zero-shot intent) ---
    use_llm_naming: bool = False
    llm_model: str = "claude-opus-4-8"  # override via ANTHROPIC_MODEL env var
    llm_max_tokens: int = 300

    # --- Outputs ---
    out_dir: str = "outputs"

    def text_column(self) -> str:
        return "_text"
