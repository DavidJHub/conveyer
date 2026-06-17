"""Clustering: embed the corpus and group it with two strong techniques.

- :func:`run_bertopic` — data-driven (BERTopic / HDBSCAN): discovers the number
  of topics and a noise cluster (``-1``). Degrades to plain HDBSCAN if BERTopic
  is not installed, and is skipped entirely if neither is available.
- :func:`kmeans_sweep` + :func:`run_kmeans` — KMeans with a fixed ``n`` plus a
  silhouette sweep to inform the choice of ``n``.

The number of clusters is a hyper-parameter, not a fact in the data; report the
sweep and (optionally) stability rather than a single ``n``.
"""

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


# --------------------------------------------------------------------------- #
# Alternatives to k-means
#
# k-means assumes isotropic, similarly-sized, convex clusters and a fixed k — a
# poor fit for embedding geometry of short conversational text. These methods fit
# that geometry better: agglomerative/spectral capture non-convex structure on a
# cosine/affinity graph, GMM allows elliptical soft clusters, and HDBSCAN is
# density-based (no fixed k, explicit noise).
# --------------------------------------------------------------------------- #
def run_agglomerative(embeddings: np.ndarray, n_clusters: int, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return model.fit_predict(embeddings)


def run_spectral(embeddings: np.ndarray, n_clusters: int, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    model = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                               n_neighbors=min(cfg.umap_n_neighbors, max(2, embeddings.shape[0] - 1)),
                               assign_labels="cluster_qr", random_state=cfg.random_state)
    return model.fit_predict(embeddings)


def run_gmm(embeddings: np.ndarray, n_clusters: int, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(n_components=n_clusters, covariance_type="full",
                            random_state=cfg.random_state)
    return model.fit_predict(embeddings)


def run_hdbscan(embeddings: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Density-based clustering. Prefers the `hdbscan` package, else sklearn's port."""
    try:
        import hdbscan

        clu = hdbscan.HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples,
                              metric="euclidean", cluster_selection_method="eom")
        return clu.fit_predict(embeddings)
    except ImportError:
        from sklearn.cluster import HDBSCAN  # available in scikit-learn >= 1.3

        clu = HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples)
        return clu.fit_predict(embeddings)


_PARTITIONAL = {"kmeans", "agglomerative", "spectral", "gmm"}


def cluster(method: str, embeddings: np.ndarray, cfg: PipelineConfig,
            n_clusters: Optional[int] = None) -> np.ndarray:
    """Dispatch to a clustering method by name."""
    n = n_clusters or cfg.n_clusters
    if method == "kmeans":
        return run_kmeans(embeddings, n, cfg)
    if method == "agglomerative":
        return run_agglomerative(embeddings, n, cfg)
    if method == "spectral":
        return run_spectral(embeddings, n, cfg)
    if method == "gmm":
        return run_gmm(embeddings, n, cfg)
    if method == "hdbscan":
        return run_hdbscan(embeddings, cfg)
    raise ValueError(f"Unknown clustering method: {method}")


# --------------------------------------------------------------------------- #
# Internal quality metrics & method comparison
# --------------------------------------------------------------------------- #
def internal_scores(embeddings: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    """Silhouette (cosine), Davies-Bouldin, Calinski-Harabasz; noise (-1) excluded."""
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                  silhouette_score)

    labels = np.asarray(labels)
    mask = labels != -1
    uniq = sorted(set(labels[mask].tolist()))
    base = {"n_clusters": len(uniq), "noise_frac": float((labels == -1).mean())}
    if len(uniq) < 2 or mask.sum() < 3:
        return {**base, "silhouette": float("nan"),
                "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    X, y = embeddings[mask], labels[mask]
    return {
        **base,
        "silhouette": float(silhouette_score(X, y, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
    }


def compare_methods(embeddings: np.ndarray, cfg: PipelineConfig,
                    niq: Optional[Sequence] = None) -> pd.DataFrame:
    """Run several methods and score them. Higher silhouette/CH and lower DB is better."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n = embeddings.shape[0]
    rows = []
    for m in cfg.compare_methods:
        if m in ("spectral", "agglomerative") and n > cfg.max_dense_n:
            rows.append({"method": m, "note": f"skipped (N={n} > max_dense_n)"})
            continue
        try:
            labels = cluster(m, embeddings, cfg)
        except Exception as exc:  # noqa: BLE001 - report and continue
            rows.append({"method": m, "note": f"error: {str(exc)[:50]}"})
            continue
        row = {"method": m, **internal_scores(embeddings, labels)}
        if niq is not None:
            row["ARI_vs_NIQ"] = float(adjusted_rand_score(niq, labels))
            row["NMI_vs_NIQ"] = float(normalized_mutual_info_score(niq, labels))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "silhouette" in df.columns:
        df = df.sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)
    return df


def select_best_method(comparison: pd.DataFrame, default: str = "kmeans") -> str:
    """Pick the method with the best cosine silhouette (ignoring failed/NaN rows)."""
    if "silhouette" not in comparison.columns:
        return default
    valid = comparison.dropna(subset=["silhouette"])
    valid = valid[valid.get("n_clusters", 0) >= 2]
    return valid.iloc[0]["method"] if len(valid) else default


# --------------------------------------------------------------------------- #
# LLM-assisted layer (reduces reliance on the raw embedder) — ClusterLLM / few-shot ideas
# --------------------------------------------------------------------------- #
def llm_keyphrase_expansion(texts: Sequence[str], llm, batch_size: int = 20) -> List[str]:
    """Append LLM-generated keyphrases to each (short) text before embedding.

    Enriching sparse short queries with salient keyphrases makes the representation
    less dependent on the base embedder (Viswanathan et al., few-shot clustering).
    Falls back to the original texts if the LLM is unavailable.
    """
    import json

    texts = list(texts)
    if llm is None:
        return texts
    out = list(texts)
    system = ("Add 3-6 salient keyphrases (skincare concern, intent, product type, brand) for each "
              "user question. Return ONLY a JSON list of strings, one per input, same order.")
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        prompt = "Questions:\n" + "\n".join(f"{j}. {t}" for j, t in enumerate(chunk))
        try:
            resp = llm.complete(prompt, system=system)
            phrases = json.loads(resp[resp.find("["): resp.rfind("]") + 1])
            for j, p in enumerate(phrases[:len(chunk)]):
                out[i + j] = f"{chunk[j]} || {p}"
        except Exception:  # noqa: BLE001 - keep original chunk on any failure
            continue
    return out


def _hierarchy_labels_for_ks(embeddings: np.ndarray, ks: Sequence[int]) -> Dict[int, np.ndarray]:
    """Cut a single average-linkage cosine hierarchy at several granularities."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    dist = pdist(embeddings, metric="cosine")
    Z = linkage(dist, method="average")
    return {k: fcluster(Z, t=k, criterion="maxclust") for k in ks}


def llm_select_granularity(embeddings: np.ndarray, texts: Sequence[str], cfg: PipelineConfig,
                           llm, candidate_ks: Optional[Sequence[int]] = None) -> dict:
    """ClusterLLM-style choice of n: ask the LLM same/different on borderline pairs,
    then pick the granularity whose hierarchy best agrees with those answers.

    Returns ``{"best_k", "agreement": DataFrame}``; falls back to ``cfg.n_clusters``.
    """
    candidate_ks = list(candidate_ks or cfg.k_sweep)
    n = embeddings.shape[0]
    if llm is None or n > cfg.max_dense_n or n < max(candidate_ks):
        return {"best_k": cfg.n_clusters, "agreement": pd.DataFrame()}

    labels_by_k = _hierarchy_labels_for_ks(embeddings, candidate_ks)

    rng = np.random.default_rng(cfg.random_state)
    pool = rng.integers(0, n, size=(min(cfg.llm_pairs_budget * 8, 2000), 2))
    pool = [(int(a), int(b)) for a, b in pool if a != b]
    # informative = same/diff assignment flips across candidate ks
    def informativeness(pair):
        same = [int(labels_by_k[k][pair[0]] == labels_by_k[k][pair[1]]) for k in candidate_ks]
        return np.var(same)
    pool.sort(key=informativeness, reverse=True)
    pairs = pool[:cfg.llm_pairs_budget]

    system = ("Decide if two skincare user questions belong to the SAME segment "
              "(same concern + intent). Answer ONLY 'yes' or 'no'.")
    llm_same = []
    for a, b in pairs:
        try:
            ans = llm.complete(f"A: {texts[a]}\nB: {texts[b]}", system=system).strip().lower()
            llm_same.append(1 if ans.startswith("y") else 0)
        except Exception:  # noqa: BLE001
            llm_same.append(np.nan)

    rows = []
    for k in candidate_ks:
        agree = [int((labels_by_k[k][a] == labels_by_k[k][b]) == bool(s))
                 for (a, b), s in zip(pairs, llm_same) if not (isinstance(s, float) and np.isnan(s))]
        rows.append({"k": k, "agreement": float(np.mean(agree)) if agree else float("nan")})
    agreement = pd.DataFrame(rows)
    best_k = int(agreement.sort_values("agreement", ascending=False)["k"].iloc[0]) if len(agreement) else cfg.n_clusters
    return {"best_k": best_k, "agreement": agreement}
