"""Analysis: brand/recommendation features, intent classification, brand
amplification, per-cluster characterisation, NIQ validation, the top LLM
recommendations, and cluster naming.

These are the deliverables; the clusters are just the segments the metrics are
read over. Everything assumes **English** text.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .ingest import normalize


# --------------------------------------------------------------------------- #
# Brand / recommendation features
# --------------------------------------------------------------------------- #
def brand_counts(series_of_sets) -> Counter:
    c: Counter = Counter()
    for s in series_of_sets:
        c.update(s)
    return c


def add_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recommendation events: brands the LLM introduces that the user did not ask for."""
    df = df.copy()
    df["_unsolicited"] = df.apply(lambda r: r["_brands_a_norm"] - r["_brands_q_norm"], axis=1)
    df["_endorsed"] = df.apply(lambda r: r["_brands_a_norm"] & r["_brands_q_norm"], axis=1)
    df["is_recommendation"] = df["_unsolicited"].apply(len) > 0
    df["n_brands_answer"] = df["_brands_a_norm"].apply(len)
    return df


# --------------------------------------------------------------------------- #
# Intent classification (rule-based; English schema)
# --------------------------------------------------------------------------- #
INTENT_PATTERNS: Dict[str, List[str]] = {
    "comparison": [r"\bvs\b", r"\bversus\b", r"which is better", r"better.*,", r"difference between",
                   r"\bor\b.*\?", r"compare"],
    "purchase": [r"best product", r"what do you recommend", r"recommend", r"where (can i|to) buy",
                 r"worth it", r"which.*should i (buy|get)", r"\bbuy\b", r"\bdupe\b"],
    "troubleshooting": [r"irritat", r"got worse", r"reaction", r"breaking out", r"can i (combine|mix|use)",
                        r"side effect", r"\bburn"],
    "routine": [r"routine", r"application order", r"morning and night", r"step by step", r"\border\b"],
    "informational": [r"what is", r"how does", r"good for", r"explain", r"what does"],
}
INTENT_PRIORITY = ["comparison", "purchase", "troubleshooting", "routine", "informational"]


def classify_intent(text: str) -> str:
    t = normalize(text)
    for intent in INTENT_PRIORITY:
        for pat in INTENT_PATTERNS[intent]:
            if re.search(pat, t):
                return intent
    return "informational"


def add_intent(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    df["intent"] = df[cfg.col_question].fillna("").apply(classify_intent)
    df["asks_recommendation"] = df["intent"].isin(["purchase", "comparison"])
    return df


def intent_accuracy(df: pd.DataFrame) -> Optional[float]:
    """If ground-truth intent is present (synthetic data), report rule accuracy."""
    if "_intent_true" in df.columns:
        return float((df["intent"] == df["_intent_true"]).mean())
    return None


# --------------------------------------------------------------------------- #
# Brand amplification
# --------------------------------------------------------------------------- #
def brand_amplification(df: pd.DataFrame) -> pd.DataFrame:
    """amplification = answer-share / question-share. >1 => the LLM amplifies the brand."""
    q_counts = brand_counts(df["_brands_q_norm"])
    a_counts = brand_counts(df["_brands_a_norm"])
    q_total = sum(q_counts.values()) or 1
    a_total = sum(a_counts.values()) or 1
    brands = sorted(set(q_counts) | set(a_counts))
    amp = pd.DataFrame({
        "brand": brands,
        "q_count": [q_counts[b] for b in brands],
        "a_count": [a_counts[b] for b in brands],
    })
    amp["q_share"] = amp["q_count"] / q_total
    amp["a_share"] = amp["a_count"] / a_total
    amp["amplification"] = amp["a_share"] / amp["q_share"].replace(0, np.nan)
    return amp.sort_values("a_count", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-cluster characterisation
# --------------------------------------------------------------------------- #
def _top_brands(sub: pd.DataFrame, col: str = "_brands_a_norm", k: int = 5) -> str:
    c = brand_counts(sub[col])
    return ", ".join(f"{b}({n})" for b, n in c.most_common(k)) if c else "-"


def cluster_summary(work: pd.DataFrame, cfg: PipelineConfig, label_col: str = "cluster") -> pd.DataFrame:
    """One row per cluster: size, intent mix, recommendation rate, brands, NIQ mode."""
    rows = []
    for cl, sub in work.groupby(label_col):
        intent_mix = sub["intent"].value_counts(normalize=True)
        niq_mode = sub[cfg.col_niq].astype(str).mode() if cfg.col_niq in sub.columns else pd.Series([], dtype=str)
        rows.append({
            "cluster": cl,
            "size": len(sub),
            "share": len(sub) / len(work),
            "niq_dominant": niq_mode.iloc[0] if len(niq_mode) else "-",
            "intent_top": intent_mix.index[0] if len(intent_mix) else "-",
            "pct_purchase": sub["intent"].eq("purchase").mean(),
            "pct_comparison": sub["intent"].eq("comparison").mean(),
            "reco_rate": sub["is_recommendation"].mean(),
            "brands_per_answer": sub["n_brands_answer"].mean(),
            "top_answer_brands": _top_brands(sub),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# External validation vs NIQ taxonomy
# --------------------------------------------------------------------------- #
def validate_against_niq(work: pd.DataFrame, cfg: PipelineConfig,
                         label_cols=("cluster_kmeans", "cluster_bertopic")) -> Dict[str, Dict[str, float]]:
    """ARI/NMI between cluster labels and the NIQ category column."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if cfg.col_niq not in work.columns:
        return {}
    niq = work[cfg.col_niq].astype(str).tolist()
    out: Dict[str, Dict[str, float]] = {}
    for col in label_cols:
        if col in work.columns:
            out[col] = {
                "ARI": float(adjusted_rand_score(niq, work[col])),
                "NMI": float(normalized_mutual_info_score(niq, work[col])),
            }
    return out


# --------------------------------------------------------------------------- #
# Top LLM recommendations when a recommendation is requested
# --------------------------------------------------------------------------- #
def top_recommendations(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Top unsolicited brands the LLM introduces on recommendation/comparison turns."""
    rec = df[df["asks_recommendation"]]
    counts = brand_counts(rec["_unsolicited"])
    return pd.DataFrame(counts.most_common(k), columns=["brand", "times_recommended"])


def top_recommendations_by_cluster(work: pd.DataFrame, label_col: str = "cluster", k: int = 5) -> pd.DataFrame:
    rows = []
    for cl, sub in work[work["asks_recommendation"]].groupby(label_col):
        counts = brand_counts(sub["_unsolicited"])
        top = ", ".join(f"{b}({n})" for b, n in counts.most_common(k)) if counts else "-"
        rows.append({"cluster": cl, "n_asks_reco": len(sub),
                     "reco_rate": sub["is_recommendation"].mean(), "top_recommendations": top})
    return pd.DataFrame(rows).sort_values("n_asks_reco", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cluster naming
# --------------------------------------------------------------------------- #
def name_clusters_keywords(work: pd.DataFrame, texts, label_col: str = "cluster") -> Dict[int, dict]:
    """Label clusters by their most distinctive TF-IDF terms (no LLM needed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = list(texts)
    labels = work[label_col].to_numpy()
    out: Dict[int, dict] = {}
    for cl in sorted(set(labels.tolist())):
        idx = np.where(labels == cl)[0]
        docs = [texts[i] for i in idx]
        if not docs:
            continue
        try:
            v = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english")
            X = v.fit_transform(docs)
            scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(v.get_feature_names_out())
            top = terms[np.argsort(-scores)[:4]]
            label = " / ".join(top)
        except ValueError:
            label = f"cluster {cl}"
        intent_mode = work.loc[work[label_col] == cl, "intent"].mode()
        out[int(cl)] = {"label": label,
                        "dominant_intent": intent_mode.iloc[0] if len(intent_mode) else "?"}
    return out


def name_clusters_llm(reps: Dict[int, List[str]], llm) -> Dict[int, dict]:
    """Label clusters with an LLM given representative questions per cluster."""
    system = ("You are a skincare research analyst. Given a cluster of user questions to an "
              "assistant, return ONLY JSON with keys: label (<=4 words), dominant_intent "
              "(informational|comparison|purchase|troubleshooting|routine), "
              "purchase_intent_heavy (boolean).")
    out: Dict[int, dict] = {}
    for cl, examples in reps.items():
        prompt = "Questions:\n- " + "\n- ".join(examples)
        txt = llm.complete(prompt, system=system)
        try:
            out[int(cl)] = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except (ValueError, json.JSONDecodeError):
            out[int(cl)] = {"label": txt[:40].strip(), "dominant_intent": "?", "purchase_intent_heavy": None}
    return out
