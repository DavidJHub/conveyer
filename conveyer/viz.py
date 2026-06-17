"""Visualization: a self-contained interactive HTML dashboard for the pipeline
results, built from either an in-memory ``run_pipeline`` result dict or the CSVs
saved in ``outputs/``. Uses Plotly; no server required.

    from conveyer import viz
    viz.build_dashboard(results)                       # from memory
    viz.build_dashboard(outputs_dir="outputs")         # from saved CSVs
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Plotly is imported lazily inside functions so importing the package stays light.


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def project_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """2D projection for plotting: UMAP if available, else PCA."""
    try:
        from umap import UMAP

        return UMAP(n_components=2, metric="cosine", random_state=random_state).fit_transform(embeddings)
    except Exception:  # noqa: BLE001 - umap missing or failed
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(embeddings)


# --------------------------------------------------------------------------- #
# Loading results saved to disk
# --------------------------------------------------------------------------- #
_CSV_FILES = {
    "summary": "cluster_summary.csv",
    "amplification": "brand_amplification.csv",
    "top_recommendations": "top_recommendations.csv",
    "top_recommendations_by_cluster": "top_recommendations_by_cluster.csv",
    "sweep": "kmeans_sweep.csv",
    "method_comparison": "method_comparison.csv",
    "projection": "projection.csv",
    "labeled": "labeled_first_turn.csv",
}


def load_results_from_dir(outputs_dir: str) -> Dict[str, pd.DataFrame]:
    """Load whatever pipeline CSVs exist in ``outputs_dir`` into a dict."""
    data: Dict[str, pd.DataFrame] = {}
    for key, fname in _CSV_FILES.items():
        path = os.path.join(outputs_dir, fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    return data


# --------------------------------------------------------------------------- #
# Figure builders (each returns a plotly Figure or None if data is missing)
# --------------------------------------------------------------------------- #
def fig_projection(projection: Optional[pd.DataFrame]):
    if projection is None or not {"x", "y", "cluster"}.issubset(projection.columns):
        return None
    import plotly.express as px

    df = projection.copy()
    df["cluster"] = df["cluster"].astype(str)
    hover = [c for c in ("question", "intent", "cluster_name") if c in df.columns]
    fig = px.scatter(df, x="x", y="y", color="cluster", hover_data=hover,
                     title="Conversation map (2D projection, colored by cluster)",
                     opacity=0.75)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(legend_title_text="cluster", height=520)
    return fig


def fig_cluster_sizes(summary: Optional[pd.DataFrame]):
    if summary is None or "size" not in summary.columns:
        return None
    import plotly.express as px

    df = summary.copy()
    df["cluster"] = df["cluster"].astype(str)
    color = "intent_top" if "intent_top" in df.columns else None
    fig = px.bar(df.sort_values("size", ascending=False), x="cluster", y="size", color=color,
                 title="Cluster sizes (by dominant intent)", text="size")
    fig.update_layout(height=420, xaxis_title="cluster", yaxis_title="documents")
    return fig


def fig_intent_by_cluster(labeled: Optional[pd.DataFrame]):
    if labeled is None or not {"cluster", "intent"}.issubset(labeled.columns):
        return None
    import plotly.express as px

    ct = (pd.crosstab(labeled["cluster"].astype(str), labeled["intent"], normalize="index")
          .reset_index().melt(id_vars="cluster", var_name="intent", value_name="proportion"))
    fig = px.bar(ct, x="cluster", y="proportion", color="intent", barmode="stack",
                 title="Intent composition within each cluster")
    fig.update_layout(height=420, yaxis_tickformat=".0%")
    return fig


def fig_reco_rate(summary: Optional[pd.DataFrame]):
    if summary is None or "reco_rate" not in summary.columns:
        return None
    import plotly.express as px

    df = summary.copy()
    df["cluster"] = df["cluster"].astype(str)
    fig = px.bar(df.sort_values("reco_rate", ascending=False), x="cluster", y="reco_rate",
                 color="reco_rate", color_continuous_scale="Viridis",
                 title="Recommendation rate per cluster (answers introducing unsolicited brands)")
    fig.update_layout(height=420, yaxis_tickformat=".0%", coloraxis_showscale=False)
    return fig


def fig_amplification(amplification: Optional[pd.DataFrame], min_answer_count: int = 5):
    if amplification is None or not {"q_share", "a_share"}.issubset(amplification.columns):
        return None
    import plotly.express as px

    df = amplification.copy()
    if "a_count" in df.columns:
        df = df[df["a_count"] >= min_answer_count]
    size = "a_count" if "a_count" in df.columns else None
    fig = px.scatter(df, x="q_share", y="a_share", size=size, hover_name="brand",
                     color="amplification" if "amplification" in df.columns else None,
                     color_continuous_scale="RdBu_r",
                     title="Brand amplification: answer-share vs question-share (above diagonal = amplified)")
    lim = float(max(df["q_share"].max(), df["a_share"].max())) if len(df) else 1.0
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="gray"))
    fig.update_layout(height=460)
    return fig


def fig_top_recommendations(top_reco: Optional[pd.DataFrame], k: int = 15):
    if top_reco is None or "brand" not in top_reco.columns:
        return None
    import plotly.express as px

    val = "times_recommended" if "times_recommended" in top_reco.columns else top_reco.columns[1]
    df = top_reco.head(k).sort_values(val)
    fig = px.bar(df, x=val, y="brand", orientation="h",
                 title="Top LLM recommendations when a recommendation is requested")
    fig.update_layout(height=460)
    return fig


def fig_method_comparison(method_comparison: Optional[pd.DataFrame]):
    if method_comparison is None or "method" not in method_comparison.columns:
        return None
    import plotly.express as px

    df = method_comparison.copy()
    if "silhouette" not in df.columns:
        return None
    df = df.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False)
    text = df["n_clusters"].astype("Int64").astype(str) + " clusters" if "n_clusters" in df.columns else None
    fig = px.bar(df, x="method", y="silhouette", color="silhouette",
                 color_continuous_scale="Viridis", text=text,
                 title="Clustering method comparison (cosine silhouette — higher is better)")
    fig.update_layout(height=420, coloraxis_showscale=False)
    return fig


def fig_k_sweep(sweep: Optional[pd.DataFrame]):
    if sweep is None or not {"k", "silhouette"}.issubset(sweep.columns):
        return None
    import plotly.express as px

    fig = px.line(sweep, x="k", y="silhouette", markers=True,
                  title="Silhouette vs number of clusters (k sweep)")
    fig.update_layout(height=380)
    return fig


# --------------------------------------------------------------------------- #
# Dashboard assembly
# --------------------------------------------------------------------------- #
def _results_to_frames(results: dict, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """Adapt an in-memory run_pipeline result dict into the frame dict the
    figure builders expect, computing a 2D projection from embeddings if needed."""
    frames: Dict[str, pd.DataFrame] = {}
    mapping = {
        "summary": "summary", "amplification": "amplification",
        "top_recommendations": "top_recommendations",
        "top_recommendations_by_cluster": "top_recommendations_by_cluster",
        "sweep": "sweep", "method_comparison": "method_comparison",
    }
    for frame_key, res_key in mapping.items():
        if isinstance(results.get(res_key), pd.DataFrame):
            frames[frame_key] = results[res_key]

    work = results.get("work")
    if isinstance(work, pd.DataFrame):
        cols = [c for c in ("question", "intent", "cluster", "cluster_name") if c in work.columns]
        frames["labeled"] = work[cols].copy()

    if isinstance(results.get("projection"), pd.DataFrame):
        frames["projection"] = results["projection"]
    elif isinstance(work, pd.DataFrame):
        emb = results.get("embeddings")
        if emb is not None and "cluster" in work.columns:
            cols = [c for c in ("question", "intent", "cluster", "cluster_name") if c in work.columns]
            xy = project_2d(np.asarray(emb), random_state)
            proj = work[cols].copy().reset_index(drop=True)
            proj["x"], proj["y"] = xy[:, 0], xy[:, 1]
            frames["projection"] = proj
    return frames


def build_dashboard(results: Optional[dict] = None, outputs_dir: str = "outputs",
                    out_html: Optional[str] = None, title: str = "Conveyer — clustering dashboard",
                    random_state: int = 42) -> str:
    """Assemble all available panels into one self-contained HTML file.

    Pass ``results`` (from :func:`conveyer.run_pipeline`) to render from memory,
    otherwise the saved CSVs in ``outputs_dir`` are used. Returns the HTML path.
    """
    frames = _results_to_frames(results, random_state) if results is not None else load_results_from_dir(outputs_dir)

    builders = [
        (fig_projection, "projection"),
        (fig_method_comparison, "method_comparison"),
        (fig_cluster_sizes, "summary"),
        (fig_intent_by_cluster, "labeled"),
        (fig_reco_rate, "summary"),
        (fig_amplification, "amplification"),
        (fig_top_recommendations, "top_recommendations"),
        (fig_k_sweep, "sweep"),
    ]
    figures = []
    for builder, key in builders:
        fig = builder(frames.get(key))
        if fig is not None:
            figures.append(fig)

    out_html = out_html or os.path.join(outputs_dir, "dashboard.html")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)

    if not figures:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>{title}</h1><p>No data found to plot.</p></body></html>")
        return out_html

    blocks = [fig.to_html(full_html=False, include_plotlyjs=False, default_height="100%")
              for fig in figures]
    cards = "\n".join(f'<div class="card">{b}</div>' for b in blocks)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background:#f5f6f8; color:#1b1f24; }}
  header {{ padding: 18px 28px; background:#111827; color:#fff; }}
  header h1 {{ margin:0; font-size:20px; }}
  header p {{ margin:4px 0 0; color:#9ca3af; font-size:13px; }}
  .grid {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:18px; padding:24px; }}
  .card {{ background:#fff; border-radius:10px; box-shadow:0 1px 4px rgba(0,0,0,.08); padding:8px; min-height:380px; }}
  .card:first-child {{ grid-column: 1 / -1; }}
</style></head>
<body>
  <header><h1>{title}</h1><p>Skincare conversation segmentation — generated by conveyer</p></header>
  <div class="grid">{cards}</div>
</body></html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
