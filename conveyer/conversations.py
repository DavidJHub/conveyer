"""MODULE 1 — conversation features: intent, sentiment, topics, brands, stages.

Turns the raw conversations table into the model-ready feature tables the
funnel model (module 3) consumes. Both the **user** side (what they asked, how
they felt) and the **agent** side (what it recommended, how enthusiastically)
are featurised, at two grains:

* ``fact_turn_features``          — one row per turn (``message_id``);
* ``fact_conversation_features``  — one row per session (``session_id``).

Feature families and how each is computed:

* **intent** — transparent rule patterns (comparison / purchase /
  troubleshooting / routine / informational) with a priority order; swap in an
  LLM or a fine-tuned classifier later without touching the schema;
* **sentiment** — a skincare-tuned lexicon scorer with negation handling by
  default, upgraded automatically to a HuggingFace pipeline when
  ``transformers`` is installed (``conveyer.models.build_sentiment_pipeline``);
* **brands** — mentions extracted with the shared canonical lexicon
  (:mod:`conveyer.brands`); the *unsolicited* set (agent-introduced, user never
  asked) is THE agent-recommendation exposure the funnel model weighs;
* **funnel stage** — per-turn keyword stages + the session-level HMM smoothing
  from :mod:`conveyer.funnel` (Awareness → … → Post-Purchase);
* **topics** — BERTopic when installed, else embeddings (auto-resolved
  backend) + KMeans, labelled by their most distinctive TF-IDF terms;
* **link behaviour** — counts from ``a_links_source`` / ``ai_click`` /
  ``next_10_urls`` (the behavioural hooks module 3 expands into events).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import pyarrow as pa

from . import funnel
from .brands import extract_brands
from .ingest import link_urls, load_conversations, trail_events
from .models import ModelConfig, build_embedder, build_sentiment_pipeline
from .schema_utils import to_dataframe, write_parquet

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ConversationConfig:
    data_path: str = "data/conversations.parquet"
    use_synthetic_if_missing: bool = True
    synthetic_n_sessions: int = 40
    seed: int = 42

    # topics
    topic_backend: str = "auto"      # auto | bertopic | kmeans | none
    n_topics: int = 8
    topic_on: str = "question"       # question | answer | qa
    min_docs_for_topics: int = 8

    # sentiment
    models: ModelConfig = field(default_factory=ModelConfig)

    out_dir: str = "outputs/conversations"

    def turns_path(self) -> str:
        import os
        return os.path.join(self.out_dir, "turn_features.parquet")

    def conversations_path(self) -> str:
        import os
        return os.path.join(self.out_dir, "conversation_features.parquet")


# --------------------------------------------------------------------------- #
# Intent (rule-based, transparent; priority order breaks ties)
# --------------------------------------------------------------------------- #
INTENT_PATTERNS: Dict[str, List[str]] = {
    "comparison": [r"\bvs\b", r"\bversus\b", r"which is better", r"difference between",
                   r"\bor\b.*\?", r"compare"],
    "purchase": [r"best product", r"what do you recommend", r"recommend", r"where (can i|to) buy",
                 r"worth it", r"which.*should i (buy|get)", r"\bbuy\b", r"\bdupe\b",
                 r"best .* for", r"what should i (use|try)"],
    "troubleshooting": [r"irritat", r"got worse", r"reaction", r"breaking out",
                        r"can i (combine|mix|use)", r"side effect", r"\bburn", r"how do i fix"],
    "routine": [r"routine", r"application order", r"morning and night", r"step by step"],
    "informational": [r"what is", r"how does", r"good for", r"explain", r"what does"],
}
INTENT_PRIORITY = ["comparison", "purchase", "troubleshooting", "routine", "informational"]


def classify_intent(text: str) -> str:
    t = str(text).lower()
    for intent in INTENT_PRIORITY:
        if any(re.search(p, t) for p in INTENT_PATTERNS[intent]):
            return intent
    return "informational"


# --------------------------------------------------------------------------- #
# Sentiment (lexicon default; transformers upgrade via models.py)
# --------------------------------------------------------------------------- #
_POS = {"love", "loved", "great", "amazing", "good", "works", "worked", "perfect",
        "gentle", "effective", "improved", "improvement", "happy", "best", "helps",
        "helped", "cleared", "glowing", "smooth", "recommend", "recommended",
        "favorite", "favourite", "nice", "excellent", "soothing", "hydrating"}
_NEG = {"hate", "hated", "bad", "terrible", "awful", "irritating", "irritated",
        "irritation", "burn", "burns", "burning", "breakout", "breakouts", "worse",
        "worsened", "reaction", "allergic", "dry", "drying", "sting", "stings",
        "redness", "itchy", "disappointed", "waste", "useless", "harsh", "greasy"}
_NEGATORS = {"not", "no", "never", "isn't", "wasn't", "doesn't", "didn't", "won't",
             "can't", "don't", "isnt", "wasnt", "doesnt", "didnt", "dont"}
_TOKEN = re.compile(r"[a-z']+")


def lexicon_sentiment(text: str) -> float:
    """Score in [-1, 1]: (pos − neg) hits over hits, negation flips polarity."""
    tokens = _TOKEN.findall(str(text).lower())
    pos = neg = 0
    for i, tok in enumerate(tokens):
        hit = 1 if tok in _POS else (-1 if tok in _NEG else 0)
        if not hit:
            continue
        if i > 0 and tokens[i - 1] in _NEGATORS:
            hit = -hit
        pos, neg = pos + (hit > 0), neg + (hit < 0)
    total = pos + neg
    return round((pos - neg) / total, 4) if total else 0.0


def _sentiment_series(texts: Sequence[str], cfg: ConversationConfig) -> tuple:
    """(scores, backend_name). Transformers pipeline when available."""
    pipe = build_sentiment_pipeline(cfg.models)
    if pipe is None:
        return [lexicon_sentiment(t) for t in texts], "lexicon"
    out = []
    for res in pipe([str(t)[:512] for t in texts], batch_size=16):
        score = float(res.get("score", 0.0))
        out.append(round(score if res.get("label", "").upper().startswith("POS") else -score, 4))
    return out, f"transformers:{cfg.models.sentiment_model}"


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #
def _have_bertopic() -> bool:
    import importlib
    try:
        importlib.import_module("bertopic")
        return True
    except ImportError:
        return False


def _tfidf_labels(texts: List[str], labels: np.ndarray) -> Dict[int, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    out: Dict[int, str] = {}
    for tid in sorted(set(int(x) for x in labels)):
        docs = [t for t, l in zip(texts, labels) if int(l) == tid]
        try:
            v = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words="english")
            X = v.fit_transform(docs)
            scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(v.get_feature_names_out())
            out[tid] = " / ".join(terms[np.argsort(-scores)[:3]])
        except ValueError:
            out[tid] = f"topic {tid}"
    return out


def add_topics(df: pd.DataFrame, cfg: ConversationConfig) -> tuple:
    """Assign ``topic_id`` + ``topic_label`` per turn. Returns (df, backend)."""
    df = df.copy()
    texts = {
        "question": df["question"],
        "answer": df["answer"],
        "qa": df["question"] + " " + df["answer"],
    }[cfg.topic_on].fillna("").astype(str).tolist()

    if cfg.topic_backend == "none" or len(df) < cfg.min_docs_for_topics:
        df["topic_id"], df["topic_label"] = -1, ""
        return df, "none"

    use_bertopic = cfg.topic_backend == "bertopic" or (
        cfg.topic_backend == "auto" and _have_bertopic() and len(df) >= 50)
    if use_bertopic and _have_bertopic():
        try:
            from bertopic import BERTopic

            model = BERTopic(nr_topics=cfg.n_topics, verbose=False)
            labels, _ = model.fit_transform(texts)
            df["topic_id"] = [int(t) for t in labels]
            info = {int(r["Topic"]): str(r["Name"]) for _, r in model.get_topic_info().iterrows()}
            df["topic_label"] = df["topic_id"].map(lambda t: info.get(t, ""))
            return df, "bertopic"
        except Exception as exc:
            print(f"[conversations] BERTopic failed ({exc}); falling back to kmeans.")

    from sklearn.cluster import KMeans

    emb = build_embedder(cfg.models).encode(texts)
    k = min(cfg.n_topics, max(2, len(df) // 4))
    labels = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit_predict(emb)
    names = _tfidf_labels(texts, labels)
    df["topic_id"] = [int(l) for l in labels]
    df["topic_label"] = df["topic_id"].map(names)
    return df, "kmeans"


# --------------------------------------------------------------------------- #
# Turn-level features
# --------------------------------------------------------------------------- #
def add_turn_features(df: pd.DataFrame, cfg: ConversationConfig) -> pd.DataFrame:
    df = df.copy()
    df["intent"] = df["question"].apply(classify_intent)
    df["asks_recommendation"] = df["intent"].isin(["purchase", "comparison"])

    q_sent, backend = _sentiment_series(df["question"].tolist(), cfg)
    a_sent, _ = _sentiment_series(df["answer"].tolist(), cfg)
    df["question_sentiment"], df["answer_sentiment"] = q_sent, a_sent
    df.attrs["sentiment_backend"] = backend

    df["brands_q"] = df["question"].apply(lambda t: sorted(extract_brands(t)))
    df["brands_a"] = df["answer"].apply(lambda t: sorted(extract_brands(t)))
    df["brands_unsolicited"] = df.apply(
        lambda r: sorted(set(r["brands_a"]) - set(r["brands_q"])), axis=1)
    df["brands_endorsed"] = df.apply(
        lambda r: sorted(set(r["brands_a"]) & set(r["brands_q"])), axis=1)
    df["n_brands_answer"] = df["brands_a"].apply(len)
    df["is_recommendation"] = df["brands_unsolicited"].apply(len) > 0

    df["funnel_stage"] = df["question"].apply(funnel.classify_stage)
    df["funnel_stage_idx"] = df["funnel_stage"].map(funnel.STAGE_INDEX)

    df["links_cited"] = df["a_links_source"].apply(link_urls)
    df["links_clicked"] = df["ai_click"].apply(link_urls)
    df["n_links_cited"] = df["links_cited"].apply(len)
    df["n_ai_clicks"] = df["links_clicked"].apply(len)
    df["n_trail_events"] = df["next_10_urls"].apply(lambda c: len(trail_events(c)))
    return df


def add_journey_stages(df: pd.DataFrame) -> pd.DataFrame:
    """HMM-smoothed journey stage per turn (see conveyer.funnel)."""
    _, decoded = funnel.fit_journey_hmm(df, session_col="session_id",
                                        pos_col="session_pos")
    return decoded


# --------------------------------------------------------------------------- #
# Conversation-level aggregation
# --------------------------------------------------------------------------- #
def conversation_features(turns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in turns.sort_values(["session_id", "session_pos"]).groupby("session_id"):
        stages = g["journey_stage"] if "journey_stage" in g.columns else g["funnel_stage"]
        path = list(dict.fromkeys(stages))          # collapse consecutive repeats
        dur = (g["prompt_dt"].max() - g["prompt_dt"].min()).total_seconds() \
            if g["prompt_dt"].notna().all() and len(g) > 1 else 0.0
        brands_a = sorted({b for bs in g["brands_a"] for b in bs})
        unsolicited = sorted({b for bs in g["brands_unsolicited"] for b in bs})
        topic_mode = g["topic_label"].mode() if "topic_label" in g.columns else pd.Series([], dtype=str)
        rows.append({
            "session_id": sid,
            "user_id": g["user_id"].iloc[0],
            "n_turns": len(g),
            "duration_seconds": float(dur),
            "first_intent": g["intent"].iloc[0],
            "dominant_intent": g["intent"].mode().iloc[0],
            "asks_recommendation_share": float(g["asks_recommendation"].mean()),
            "mean_question_sentiment": float(g["question_sentiment"].mean()),
            "mean_answer_sentiment": float(g["answer_sentiment"].mean()),
            "sentiment_trend": float(g["question_sentiment"].iloc[-1]
                                     - g["question_sentiment"].iloc[0]),
            "brands_discussed": brands_a,
            "brands_recommended_unsolicited": unsolicited,
            "n_brands_discussed": len(brands_a),
            "n_unsolicited_recommendations": len(unsolicited),
            "any_recommendation": bool(g["is_recommendation"].any()),
            "n_links_cited": int(g["n_links_cited"].sum()),
            "n_ai_clicks": int(g["n_ai_clicks"].sum()),
            "n_trail_events": int(g["n_trail_events"].sum()),
            "max_funnel_stage_idx": int(g["funnel_stage_idx"].max()),
            "max_funnel_stage": funnel.STAGES[int(g["funnel_stage_idx"].max())],
            "journey_path": " > ".join(path),
            "topic_label": topic_mode.iloc[0] if len(topic_mode) else "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Output schemas + orchestration
# --------------------------------------------------------------------------- #
_S, _I32, _F64, _B, _LS = (pa.string(), pa.int32(), pa.float64(), pa.bool_(),
                           pa.list_(pa.string()))

TURN_SCHEMA = pa.schema([
    ("message_id", _S), ("session_id", _S), ("user_id", _S), ("session_pos", _I32),
    ("prompt_datetime", _S), ("question", _S), ("answer", _S),
    ("intent", _S), ("asks_recommendation", _B),
    ("question_sentiment", _F64), ("answer_sentiment", _F64),
    ("brands_q", _LS), ("brands_a", _LS), ("brands_unsolicited", _LS),
    ("brands_endorsed", _LS), ("n_brands_answer", _I32), ("is_recommendation", _B),
    ("funnel_stage", _S), ("journey_stage", _S),
    ("topic_id", _I32), ("topic_label", _S),
    ("links_cited", _LS), ("links_clicked", _LS),
    ("n_links_cited", _I32), ("n_ai_clicks", _I32), ("n_trail_events", _I32),
])

CONVERSATION_SCHEMA = pa.schema([
    ("session_id", _S), ("user_id", _S), ("n_turns", _I32),
    ("duration_seconds", _F64), ("first_intent", _S), ("dominant_intent", _S),
    ("asks_recommendation_share", _F64),
    ("mean_question_sentiment", _F64), ("mean_answer_sentiment", _F64),
    ("sentiment_trend", _F64),
    ("brands_discussed", _LS), ("brands_recommended_unsolicited", _LS),
    ("n_brands_discussed", _I32), ("n_unsolicited_recommendations", _I32),
    ("any_recommendation", _B),
    ("n_links_cited", _I32), ("n_ai_clicks", _I32), ("n_trail_events", _I32),
    ("max_funnel_stage_idx", _I32), ("max_funnel_stage", _S),
    ("journey_path", _S), ("topic_label", _S),
])


def _turn_rows(turns: pd.DataFrame) -> List[dict]:
    rows = []
    for _, r in turns.iterrows():
        rows.append({
            "message_id": str(r["message_id"]), "session_id": str(r["session_id"]),
            "user_id": str(r["user_id"]), "session_pos": int(r["session_pos"]),
            "prompt_datetime": str(r.get("prompt_datetime", "") or ""),
            "question": r["question"], "answer": r["answer"],
            "intent": r["intent"], "asks_recommendation": bool(r["asks_recommendation"]),
            "question_sentiment": float(r["question_sentiment"]),
            "answer_sentiment": float(r["answer_sentiment"]),
            "brands_q": list(r["brands_q"]), "brands_a": list(r["brands_a"]),
            "brands_unsolicited": list(r["brands_unsolicited"]),
            "brands_endorsed": list(r["brands_endorsed"]),
            "n_brands_answer": int(r["n_brands_answer"]),
            "is_recommendation": bool(r["is_recommendation"]),
            "funnel_stage": r["funnel_stage"],
            "journey_stage": r.get("journey_stage", r["funnel_stage"]),
            "topic_id": int(r.get("topic_id", -1)),
            "topic_label": str(r.get("topic_label", "")),
            "links_cited": list(r["links_cited"]), "links_clicked": list(r["links_clicked"]),
            "n_links_cited": int(r["n_links_cited"]), "n_ai_clicks": int(r["n_ai_clicks"]),
            "n_trail_events": int(r["n_trail_events"]),
        })
    return rows


def run_conversations(cfg: Optional[ConversationConfig] = None,
                      df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Load (or take) the conversations table, featurise, write both parquets."""
    from .ingest import ingest as _ingest

    cfg = cfg or ConversationConfig()
    html_by_url = gt = None
    if df is None:
        df, html_by_url, gt = _ingest(cfg.data_path, cfg.synthetic_n_sessions, cfg.seed)
    else:
        df = load_conversations(df)
    print(f"[conversations] {df.attrs.get('source', 'provided df')} | turns={len(df)} | "
          f"sessions={df.session_id.nunique()}")

    turns = add_turn_features(df, cfg)
    turns, topic_backend = add_topics(turns, cfg)
    turns = add_journey_stages(turns)
    convs = conversation_features(turns)
    print(f"[conversations] sentiment={turns.attrs.get('sentiment_backend', '?')} | "
          f"topics={topic_backend} | reco_rate={turns['is_recommendation'].mean():.1%}")

    turns_df = to_dataframe(_turn_rows(turns), TURN_SCHEMA)
    convs_df = to_dataframe([{**r} for r in convs.to_dict("records")], CONVERSATION_SCHEMA)
    write_parquet(_turn_rows(turns), TURN_SCHEMA, cfg.turns_path())
    write_parquet(convs.to_dict("records"), CONVERSATION_SCHEMA, cfg.conversations_path())
    print(f"[export] {cfg.turns_path()} ({len(turns_df)}) | "
          f"{cfg.conversations_path()} ({len(convs_df)})")

    return {"turns": turns, "turns_out": turns_df, "conversations": convs_df,
            "html_by_url": html_by_url, "ground_truth": gt,
            "topic_backend": topic_backend}
