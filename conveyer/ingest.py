"""Data ingestion: load conversations, normalise brand columns, derive helper
fields, and (when no dataset is present) generate realistic **English**
synthetic data so the pipeline runs end-to-end.
"""
from __future__ import annotations

import ast
import os
from typing import Any, List, Set

import numpy as np
import pandas as pd

from .config import PipelineConfig


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def to_list(x: Any) -> List[str]:
    """Coerce a cell into a list of strings.

    Tolerates real Python lists, stringified lists (``"['CeraVe', ...]"``),
    simple comma-separated strings, and NaN/None.
    """
    if isinstance(x, list):
        return [str(b).strip() for b in x if str(b).strip()]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set)):
                    return [str(b).strip() for b in v if str(b).strip()]
            except (ValueError, SyntaxError):
                pass
        return [t.strip() for t in s.split(",") if t.strip()]
    return []


def normalize(s: Any) -> str:
    return str(s).strip().lower()


def as_brand_set(lst: List[str]) -> Set[str]:
    return set(normalize(b) for b in lst)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_any(path: str) -> pd.DataFrame:
    """Read a dataframe, autodetecting format by extension."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {path}")


def prepare(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Normalise brand columns, derive ``session_pos`` and build the text field."""
    df = df.copy()

    empty = pd.Series([[] for _ in range(len(df))], index=df.index)
    df["_brands_q"] = (df[cfg.col_brands_q] if cfg.col_brands_q in df.columns else empty).apply(to_list)
    df["_brands_a"] = (df[cfg.col_brands_a] if cfg.col_brands_a in df.columns else empty).apply(to_list)
    df["_brands_q_norm"] = df["_brands_q"].apply(as_brand_set)
    df["_brands_a_norm"] = df["_brands_a"].apply(as_brand_set)

    if cfg.col_session_pos not in df.columns:
        if cfg.col_session in df.columns:
            df[cfg.col_session_pos] = df.groupby(cfg.col_session).cumcount() + 1
        else:
            df[cfg.col_session_pos] = 1

    q = df[cfg.col_question].fillna("").astype(str)
    a = (df[cfg.col_answer] if cfg.col_answer in df.columns else pd.Series([""] * len(df), index=df.index))
    a = a.fillna("").astype(str)
    if cfg.cluster_on == "question":
        df["_text"] = q
    elif cfg.cluster_on == "answer":
        df["_text"] = a
    else:  # "qa"
        df["_text"] = (q + " [SEP] " + a).str.strip()

    return df


# --------------------------------------------------------------------------- #
# Synthetic English data
# --------------------------------------------------------------------------- #
_CATEGORIES = {
    "acne": (["acne", "breakouts", "blackheads", "pimples", "clogged pores"],
             ["CeraVe", "La Roche-Posay", "The Ordinary", "Paula's Choice"]),
    "anti-aging": (["wrinkles", "anti-aging", "fine lines", "firmness", "retinol"],
                   ["Olay", "RoC", "The Ordinary", "Drunk Elephant", "Estee Lauder"]),
    "sunscreen": (["sunscreen", "spf", "sun protection", "spf 50", "uv filter"],
                  ["La Roche-Posay", "ISDIN", "Bioderma", "EltaMD"]),
    "hyperpigmentation": (["dark spots", "hyperpigmentation", "uneven tone", "vitamin c"],
                          ["The Ordinary", "Good Molecules", "Murad", "SkinCeuticals"]),
    "hydration": (["hydration", "dry skin", "moisturizer", "hyaluronic acid"],
                  ["CeraVe", "Neutrogena", "Cetaphil", "Vichy"]),
    "sensitive": (["sensitive skin", "redness", "rosacea", "irritation"],
                  ["Avene", "Bioderma", "La Roche-Posay", "Cetaphil"]),
}

_INTENT_TEMPLATES = {
    "informational": ["what is {t}", "how does {t} work", "what is {t} good for", "explain {t}"],
    "comparison": ["{b1} vs {b2} for {t}", "which is better for {t}, {b1} or {b2}",
                   "difference between {b1} and {b2}"],
    "purchase": ["what is the best product for {t}", "what do you recommend for {t}",
                 "where can I buy something for {t}", "is {b1} worth it for {t}"],
    "troubleshooting": ["{b1} is irritating my skin, what should I do",
                        "my {t} got worse using {b1}", "can I combine {b1} with retinol"],
    "routine": ["build a routine for {t}", "application order for {t}",
                "morning and night routine for {t}"],
}
_INTENT_PROBS = [0.30, 0.18, 0.27, 0.15, 0.10]


def make_synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate plausible English skincare conversations for end-to-end testing."""
    rng = np.random.default_rng(seed)
    cat_names = list(_CATEGORIES.keys())
    intent_names = list(_INTENT_TEMPLATES.keys())
    rows = []
    for i in range(n):
        cat = rng.choice(cat_names)
        terms, brands = _CATEGORIES[cat]
        t = rng.choice(terms)
        intent = rng.choice(intent_names, p=_INTENT_PROBS)
        b1, b2 = rng.choice(brands, size=2, replace=False)
        question = rng.choice(_INTENT_TEMPLATES[intent]).format(t=t, b1=b1, b2=b2)

        if intent == "comparison":
            brands_q = [b1, b2]
        elif intent == "troubleshooting" or rng.random() < 0.25:
            brands_q = [b1]
        else:
            brands_q = []

        n_rec = int(rng.integers(1, 4))
        brands_a = list(rng.choice(brands, size=min(n_rec, len(brands)), replace=False))
        brands_a = list(dict.fromkeys(brands_q + brands_a))  # keep question brands, dedupe
        cites = bool(rng.random() < 0.4)
        answer = f"For {t}, consider {', '.join(brands_a)}." + (" [sources cited]" if cites else "")

        rows.append(dict(
            session_id=f"s{i // 3}", question=question, answer=answer,
            brands_in_question=brands_q, brands_in_answer=brands_a,
            skin_care_categories=cat, cites_sources=cites, _intent_true=intent,
        ))
    df = pd.DataFrame(rows)
    df["session_pos"] = df.groupby("session_id").cumcount() + 1
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def ingest(cfg: PipelineConfig) -> pd.DataFrame:
    """Load (or synthesise) the dataset and return a prepared dataframe."""
    if os.path.exists(cfg.data_path):
        df = read_any(cfg.data_path)
        source = cfg.data_path
    elif cfg.use_synthetic_if_missing:
        df = make_synthetic(cfg.synthetic_n, cfg.random_state)
        source = f"SYNTHETIC ({len(df)} rows; {cfg.data_path} not found)"
    else:
        raise FileNotFoundError(cfg.data_path)

    df = prepare(df, cfg)
    df.attrs["source"] = source
    return df
