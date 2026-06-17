"""Clustering: embed the corpus and group it with two strong techniques.

- :func:`run_bertopic` — data-driven (BERTopic / HDBSCAN): discovers the number
  of topics and a noise cluster (``-1``). Degrades to plain HDBSCAN if BERTopic
  is not installed, and is skipped entirely if neither is available.
- :func:`kmeans_sweep` + :func:`run_kmeans` — KMeans with a fixed ``n`` plus a
  silhouette sweep to inform the choice of ``n``.

The number of clusters is a hyper-parameter, not a fact in the data; report the
sweep and (optionally) stability rather than a single ``n``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models import BaseEmbedder, build_embedder, detect_capabilities


def embed_corpus(texts: Sequence[str], cfg: PipelineConfig,
                 embedder: Optional[BaseEmbedder] = None) -> Tuple[np.ndarray, str]:
    """Embed ``texts`` with the configured (or provided) embedder."""
    embedder = embedder or build_embedder(cfg)
    emb = embedder.encode(list(texts))
    return emb, embedder.name


# --------------------------------------------------------------------------- #
# KMeans (fixed n) + silhouette sweep
# --------------------------------------------------------------------------- #
def kmeans_sweep(embeddings: np.ndarray, cfg: PipelineConfig) -> pd.DataFrame:
    """Sweep candidate ``k`` values, scoring each by cosine silhouette."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = embeddings.shape[0]
    rows = []
    for k in cfg.k_sweep:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, metric="cosine") if len(set(labels)) > 1 else float("nan")
        rows.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def run_kmeans(embeddings: np.ndarray, n_clusters: int, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init=10)
    return km.fit_predict(embeddings)


# --------------------------------------------------------------------------- #
# BERTopic / HDBSCAN (data-driven)
# --------------------------------------------------------------------------- #
def run_bertopic(texts: Sequence[str], embeddings: np.ndarray, cfg: PipelineConfig):
    """Return ``(labels, topic_model)``; ``labels`` is ``None`` if unavailable."""
    caps = detect_capabilities()
    if caps["bertopic"] and caps["umap"] and caps["hdbscan"]:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        umap_model = UMAP(n_neighbors=cfg.umap_n_neighbors, n_components=cfg.umap_n_components,
                          metric="cosine", random_state=cfg.random_state)
        hdb = HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples,
                      metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        topic_model = BERTopic(
            umap_model=umap_model, hdbscan_model=hdb,
            vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)),
            calculate_probabilities=False, verbose=False,
        )
        labels, _ = topic_model.fit_transform(list(texts), embeddings)
        return np.asarray(labels), topic_model

    if caps["hdbscan"]:
        import hdbscan

        clu = hdbscan.HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples,
                              metric="euclidean", cluster_selection_method="eom")
        return clu.fit_predict(embeddings), None

    return None, None


def noise_fraction(labels: np.ndarray) -> float:
    """Share of points assigned to the HDBSCAN noise cluster (``-1``)."""
    return float((labels == -1).mean()) if labels is not None else 0.0


def n_clusters_found(labels: np.ndarray) -> int:
    if labels is None:
        return 0
    uniq = set(labels.tolist())
    return len(uniq) - (1 if -1 in uniq else 0)


# --------------------------------------------------------------------------- #
# Representative documents
# --------------------------------------------------------------------------- #
def representative_docs(embeddings: np.ndarray, labels: Sequence[int], texts: Sequence[str],
                        top_n: int = 5) -> Dict[int, List[str]]:
    """For each cluster, the ``top_n`` documents closest to the centroid."""
    from sklearn.metrics.pairwise import cosine_similarity

    labels = np.asarray(labels)
    texts = list(texts)
    reps: Dict[int, List[str]] = {}
    for cl in sorted(set(labels.tolist())):
        idx = np.where(labels == cl)[0]
        if idx.size == 0:
            continue
        centroid = embeddings[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(embeddings[idx], centroid).ravel()
        order = idx[np.argsort(-sims)[:top_n]]
        reps[int(cl)] = [texts[i] for i in order]
    return reps


# --------------------------------------------------------------------------- #
# Stability across seeds
# --------------------------------------------------------------------------- #
def kmeans_stability(embeddings: np.ndarray, n_clusters: int, cfg: PipelineConfig,
                     seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> float:
    """Mean pairwise ARI of KMeans labelings across seeds (1.0 = identical)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    runs = [KMeans(n_clusters=n_clusters, random_state=s, n_init=10).fit_predict(embeddings) for s in seeds]
    scores = [adjusted_rand_score(runs[i], runs[j])
              for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return float(np.mean(scores)) if scores else float("nan")
