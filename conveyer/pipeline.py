"""End-to-end orchestration tying ingest -> models -> clustering -> analysis.

Run as a module::

    python -m conveyer.pipeline                 # synthetic data, best available backend
    python -m conveyer.pipeline --data data/conversations.parquet

or programmatically::

    from conveyer import PipelineConfig, run_pipeline
    results = run_pipeline(PipelineConfig(data_path="data/conversations.parquet"))
"""

import argparse
import os
from dataclasses import asdict
from typing import Dict

import pandas as pd

from . import analysis, clustering, ingest, viz
from .config import PipelineConfig
from .models import build_embedder, build_llm, detect_capabilities


def run_pipeline(cfg: PipelineConfig) -> Dict[str, object]:
    """Run the full pipeline and return a dict of artefacts."""
    caps = detect_capabilities()

    # 1) Ingest + features --------------------------------------------------- #
    df = ingest.ingest(cfg)
    df = analysis.add_brand_features(df)
    df = analysis.add_intent(df, cfg)
    print(f"[ingest] {df.attrs.get('source')} | rows={len(df)}")
    print(f"[features] recommendation_rate={df['is_recommendation'].mean():.1%} | "
          f"brands/answer={df['n_brands_answer'].mean():.2f}")
    acc = analysis.intent_accuracy(df)
    if acc is not None:
        print(f"[intent] rule accuracy vs ground truth (synthetic): {acc:.1%}")

    # 2) Corpus + optional LLM keyphrase expansion + embeddings ------------- #
    mask = (df[cfg.col_session_pos] == 1) if cfg.first_turn_only else pd.Series(True, index=df.index)
    work = df[mask].copy().reset_index(drop=True)
    texts = work[cfg.text_column()].tolist()

    llm = build_llm(cfg)
    embed_texts = texts
    if cfg.llm_keyphrase_expansion and llm is not None:
        embed_texts = clustering.llm_keyphrase_expansion(texts, llm)
        print("[llm] keyphrase expansion applied before embedding")

    embedder = build_embedder(cfg)
    embeddings, emb_name = clustering.embed_corpus(embed_texts, cfg, embedder=embedder)
    print(f"[embeddings] {embeddings.shape} via {emb_name} | docs={len(texts)}")

    # 3) Clustering: compare methods, pick granularity, pick method --------- #
    niq = work[cfg.col_niq].astype(str).tolist() if cfg.col_niq in work.columns else None
    sweep = clustering.kmeans_sweep(embeddings, cfg)
    comparison = clustering.compare_methods(embeddings, cfg, niq=niq)
    print("[methods]\n" + comparison.to_string(index=False))

    n_clusters = cfg.n_clusters
    granularity = None
    if cfg.llm_select_granularity and llm is not None:
        granularity = clustering.llm_select_granularity(embeddings, texts, cfg, llm)
        n_clusters = granularity["best_k"]
        print(f"[llm] ClusterLLM-style granularity -> n={n_clusters}")

    method = clustering.select_best_method(comparison) if cfg.cluster_method == "auto" else cfg.cluster_method
    work["cluster_kmeans"] = clustering.run_kmeans(embeddings, n_clusters, cfg)
    work["cluster"] = clustering.cluster(method, embeddings, cfg, n_clusters=n_clusters)
    stability = clustering.kmeans_stability(embeddings, n_clusters, cfg)

    bertopic_labels, _ = clustering.run_bertopic(texts, embeddings, cfg)
    if bertopic_labels is not None:
        work["cluster_bertopic"] = bertopic_labels
        print(f"[bertopic] clusters={clustering.n_clusters_found(bertopic_labels)} | "
              f"noise={clustering.noise_fraction(bertopic_labels):.1%}")

    print(f"[clustering] method={method} (cfg={cfg.cluster_method}) | n={n_clusters} | "
          f"kmeans stability(ARI)={stability:.3f}")

    # 4) Analysis ------------------------------------------------------------ #
    amp = analysis.brand_amplification(df)
    summary = analysis.cluster_summary(work, cfg)
    niq_scores = analysis.validate_against_niq(work, cfg)
    top_reco = analysis.top_recommendations(df)
    top_reco_cluster = analysis.top_recommendations_by_cluster(work)
    reps = clustering.representative_docs(embeddings, work["cluster"].to_numpy(), texts)

    if cfg.use_llm_naming and llm is not None:
        names = analysis.name_clusters_llm(reps, llm)
    else:
        names = analysis.name_clusters_keywords(work, texts)
    work["cluster_name"] = work["cluster"].map(lambda c: names.get(int(c), {}).get("label", str(c)))

    if niq_scores:
        print("[validation vs NIQ]", {k: {m: round(v, 3) for m, v in d.items()} for k, d in niq_scores.items()})

    # 5) Export + dashboard -------------------------------------------------- #
    os.makedirs(cfg.out_dir, exist_ok=True)
    labeled_cols = [cfg.col_question, "cluster", "cluster_kmeans", "cluster_name", "intent",
                    "asks_recommendation", "is_recommendation", "n_brands_answer"]
    if cfg.col_niq in work.columns:
        labeled_cols.append(cfg.col_niq)
    if "cluster_bertopic" in work.columns:
        labeled_cols.append("cluster_bertopic")
    labeled = work[labeled_cols]

    # 2D projection persisted so the disk-only dashboard can also draw the map
    xy = viz.project_2d(embeddings, cfg.random_state)
    proj_cols = [c for c in (cfg.col_question, "cluster", "cluster_name", "intent") if c in work.columns]
    projection = work[proj_cols].copy().rename(columns={cfg.col_question: "question"})
    projection["x"], projection["y"] = xy[:, 0], xy[:, 1]

    outputs = {
        "labeled_first_turn.csv": labeled,
        "cluster_summary.csv": summary,
        "brand_amplification.csv": amp,
        "top_recommendations.csv": top_reco,
        "top_recommendations_by_cluster.csv": top_reco_cluster,
        "kmeans_sweep.csv": sweep,
        "method_comparison.csv": comparison,
        "projection.csv": projection,
    }
    for fname, frame in outputs.items():
        frame.to_csv(os.path.join(cfg.out_dir, fname), index=False)
    print(f"[export] wrote {len(outputs)} files to {os.path.abspath(cfg.out_dir)}")

    results = {
        "config": asdict(cfg), "capabilities": caps, "df": df, "work": work,
        "embeddings": embeddings, "embedding_name": emb_name, "sweep": sweep,
        "method": method, "method_comparison": comparison, "stability": stability,
        "summary": summary, "amplification": amp, "niq_scores": niq_scores,
        "top_recommendations": top_reco, "top_recommendations_by_cluster": top_reco_cluster,
        "names": names, "reps": reps, "granularity": granularity, "projection": projection,
    }

    if cfg.build_dashboard:
        path = viz.build_dashboard(results, outputs_dir=cfg.out_dir)
        print(f"[dashboard] {os.path.abspath(path)}")

    return results


def _parse_args(argv=None) -> PipelineConfig:
    cfg = PipelineConfig()
    p = argparse.ArgumentParser(description="Conveyer skincare conversation clustering pipeline")
    p.add_argument("--data", default=cfg.data_path, help="Path to dataset (parquet/csv/jsonl/json)")
    p.add_argument("--cluster-on", default=cfg.cluster_on, choices=["question", "answer", "qa"])
    p.add_argument("--n-clusters", type=int, default=cfg.n_clusters)
    p.add_argument("--embedding-backend", default=cfg.embedding_backend,
                   choices=["auto", "voyage", "openai", "sentence_transformers", "tfidf"])
    p.add_argument("--cluster-method", default=cfg.cluster_method,
                   choices=["auto", "kmeans", "agglomerative", "spectral", "gmm", "hdbscan"],
                   help="Primary clustering method ('auto' picks best by silhouette)")
    p.add_argument("--all-turns", action="store_true", help="Use all turns, not just the first")
    p.add_argument("--llm-naming", action="store_true", help="Name clusters with an LLM (needs ANTHROPIC_API_KEY)")
    p.add_argument("--keyphrase-expansion", action="store_true",
                   help="LLM keyphrase expansion before embedding (needs ANTHROPIC_API_KEY)")
    p.add_argument("--llm-granularity", action="store_true",
                   help="ClusterLLM-style LLM choice of n (needs ANTHROPIC_API_KEY)")
    p.add_argument("--no-dashboard", action="store_true", help="Skip writing dashboard.html")
    p.add_argument("--out-dir", default=cfg.out_dir)
    a = p.parse_args(argv)
    return PipelineConfig(
        data_path=a.data, cluster_on=a.cluster_on, n_clusters=a.n_clusters,
        embedding_backend=a.embedding_backend, cluster_method=a.cluster_method,
        first_turn_only=not a.all_turns, use_llm_naming=a.llm_naming,
        llm_keyphrase_expansion=a.keyphrase_expansion, llm_select_granularity=a.llm_granularity,
        build_dashboard=not a.no_dashboard, out_dir=a.out_dir,
    )


if __name__ == "__main__":
    run_pipeline(_parse_args())
