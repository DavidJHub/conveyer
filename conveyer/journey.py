"""MODULE 3 — the funnel: conversations × pages × clicks → conversion model.

Connects everything the other two modules produced and answers the project's
two headline questions:

1. **Conversion rate** — how often does a conversation end in (a proxy for) a
   purchase? With no purchase ground truth in the input schema, conversion is
   *reaching a Purchase-stage page* (cart / checkout / order) in the post-turn
   browsing trail — an explicit, documented proxy with adjustable depth
   (``JourneyConfig.conversion_stage``).
2. **Weight of agent recommendations** — are the brands the agent mentioned
   and the links it showed actually being clicked and followed, and how much
   does that exposure move conversion? Measured two ways: stratified
   conversion rates with Jeffreys credible intervals (exposed vs not), and the
   standardized coefficients of a holdout-validated logistic model.

The event grammar (grain = one behavioural event, table ``fact_funnel_event``):

* ``cited``       — the agent surfaced a link (``a_links_source``);
* ``ai_click``    — the user clicked a link straight from the answer;
* ``trail_visit`` — a post-turn navigation event (``next_10_urls``), with
  1-based position and the dwell until the next request.

Every event URL is joined to module 2's ``fact_scraped_page``; URLs that were
never scraped fall back to URL-only multimodal classification
(:func:`conveyer.scraping.classify_url`) so coverage is always 100%.
``brand_match`` ties the page's brand back to the conversation:
``unsolicited_rec`` (agent introduced it — the treatment of interest),
``endorsed``, ``requested``, or ``none``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from .brands import brand_for_domain, normalize_brand
from .ingest import trail_events
from .schema_utils import to_dataframe, write_parquet

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_DEPTH = {"shopping": 1, "cart": 2, "checkout": 3}


@dataclass
class JourneyConfig:
    conversion_stage: str = "cart"   # shopping | cart | checkout — proxy depth
    test_size: float = 0.3
    seed: int = 42
    out_dir: str = "outputs/journey"

    def path(self, name: str) -> str:
        import os
        return os.path.join(self.out_dir, name)


# commerce depth per page reading: checkout/order deepest, cart next, any
# shopping page counts as reaching the shop
def _commerce_depth(category: str, subtype: str) -> int:
    if subtype in ("checkout", "order"):
        return 3
    if subtype == "cart":
        return 2
    if category == "shopping":
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Event assembly
# --------------------------------------------------------------------------- #
def _page_lookup(pages: Optional[pd.DataFrame],
                 products: Optional[pd.DataFrame] = None) -> Dict[str, dict]:
    if pages is None or pages.empty:
        return {}
    cols = ["url", "page_category", "page_subtype", "funnel_stage", "seller_type",
            "primary_brand", "is_study_relevant"]
    have = [c for c in cols if c in pages.columns]
    lookup = {r["url"]: r for r in pages[have].to_dict("records")}
    # a retailer-hosted PDP carries its brand in the *product* markup, not the
    # domain — fold the extracted product brands in so brand_match can see them
    if products is not None and len(products):
        for url, g in products.groupby("url"):
            if url in lookup:
                lookup[url]["product_brands"] = sorted(
                    {str(b) for b in g["brand"] if str(b).strip()})
    return lookup


def _classify_uncovered(url: str) -> dict:
    from .scraping import classify_url

    c = classify_url(url)
    return {"url": url, "page_category": c.page_category, "page_subtype": c.page_subtype,
            "funnel_stage": c.funnel_stage, "seller_type": c.seller_type,
            "primary_brand": c.primary_brand, "is_study_relevant": c.is_study_relevant}


def _canonical_page_brands(info: dict) -> List[str]:
    """All canonical brands a page is evidence for: its own domain/OG brand
    plus the brands of the products extracted from it."""
    out: List[str] = []
    raw = str(info.get("primary_brand", "") or "")
    if raw:
        out.append(brand_for_domain(raw) or normalize_brand(raw))
    for b in info.get("product_brands", []) or []:
        canon = normalize_brand(b)
        if canon and canon not in out:
            out.append(canon)
    return out


def build_events(turns: pd.DataFrame, pages: Optional[pd.DataFrame] = None,
                 products: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per behavioural event, joined to its page classification."""
    lookup = _page_lookup(pages, products)
    cache: Dict[str, dict] = {}

    def info_for(url: str) -> Tuple[dict, str]:
        if url in lookup:
            return lookup[url], "pages"
        if url not in cache:
            cache[url] = _classify_uncovered(url)
        return cache[url], "url_only"

    rows: List[dict] = []
    for _, t in turns.iterrows():
        brands_uns = {normalize_brand(b) for b in t.get("brands_unsolicited", [])}
        brands_end = {normalize_brand(b) for b in t.get("brands_endorsed", [])}
        brands_q = {normalize_brand(b) for b in t.get("brands_q", [])}
        cited = list(t.get("links_cited", []) or [])
        clicked = set(t.get("links_clicked", []) or [])

        def emit(url, etype, position=None, dwell=None):
            info, coverage = info_for(url)
            pbrands = _canonical_page_brands(info)
            pbrand = pbrands[0] if pbrands else ""
            if any(b in brands_uns for b in pbrands):
                match = "unsolicited_rec"
                pbrand = next(b for b in pbrands if b in brands_uns)
            elif any(b in brands_end for b in pbrands):
                match = "endorsed"
            elif any(b in brands_q for b in pbrands):
                match = "requested"
            else:
                match = "none"
            rows.append({
                "session_id": str(t["session_id"]), "message_id": str(t["message_id"]),
                "event_type": etype, "url": url,
                "position": int(position) if position is not None else None,
                "dwell_seconds": float(dwell) if dwell is not None else None,
                "page_category": info.get("page_category", "unknown"),
                "page_subtype": info.get("page_subtype", "other"),
                "funnel_stage": info.get("funnel_stage", "Irrelevant"),
                "seller_type": info.get("seller_type", "na"),
                "page_brand": pbrand,
                "brand_match": match,
                # a *user action* on an agent-surfaced link: a direct click, or
                # a trail visit to a URL the answer cited. "cited" rows are
                # exposure, never behaviour, so they are always False here.
                "followed_agent_link": bool(
                    etype == "ai_click"
                    or (etype == "trail_visit" and (url in cited or url in clicked))),
                "commerce_depth": _commerce_depth(info.get("page_category", ""),
                                                  info.get("page_subtype", "")),
                "is_study_relevant": bool(info.get("is_study_relevant", False)),
                "page_coverage": coverage,
            })

        for url in cited:
            emit(url, "cited")
        for url in clicked:
            emit(url, "ai_click")
        for ev in trail_events(t.get("next_10_urls")):
            emit(ev["url"], "trail_visit", position=ev["position"],
                 dwell=ev["dwell_seconds"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Session outcomes + features
# --------------------------------------------------------------------------- #
def journey_features(conversations: pd.DataFrame, events: pd.DataFrame,
                     cfg: JourneyConfig) -> pd.DataFrame:
    """One row per session: exposure features + behavioural outcome."""
    need_depth = _DEPTH[cfg.conversion_stage]
    behav = events[events["event_type"].isin(["ai_click", "trail_visit"])]

    per_session: Dict[str, dict] = {}
    for sid, g in behav.groupby("session_id"):
        shop = g[g["commerce_depth"] >= 1]
        per_session[sid] = {
            "max_commerce_depth": int(g["commerce_depth"].max()) if len(g) else 0,
            "n_visits": int(len(g)),
            "n_shopping_visits": int(len(shop)),
            "n_relevant_visits": int(g["is_study_relevant"].sum()),
            "followed_agent_link": bool(g["followed_agent_link"].any()),
            "visited_recommended_brand": bool((g["brand_match"] == "unsolicited_rec").any()),
            "visited_requested_brand": bool(g["brand_match"].isin(["requested", "endorsed"]).any()),
            "total_dwell_shopping": float(shop["dwell_seconds"].fillna(0).sum()),
            "mean_dwell": float(g["dwell_seconds"].dropna().mean())
            if g["dwell_seconds"].notna().any() else 0.0,
        }

    rows = []
    for _, c in conversations.iterrows():
        b = per_session.get(str(c["session_id"]), {})
        depth = b.get("max_commerce_depth", 0)
        followed = bool(b.get("followed_agent_link", False)
                        or b.get("visited_recommended_brand", False))
        rows.append({
            "session_id": str(c["session_id"]),
            # conversation-side features (module 1)
            "n_turns": int(c["n_turns"]),
            "asks_recommendation_share": float(c["asks_recommendation_share"]),
            "mean_question_sentiment": float(c["mean_question_sentiment"]),
            "mean_answer_sentiment": float(c["mean_answer_sentiment"]),
            "n_unsolicited_recommendations": int(c["n_unsolicited_recommendations"]),
            "n_links_cited": int(c["n_links_cited"]),
            "max_funnel_stage_idx": int(c["max_funnel_stage_idx"]),
            # behaviour-side features (module 2 + trails)
            "n_visits": b.get("n_visits", 0),
            "n_shopping_visits": b.get("n_shopping_visits", 0),
            "n_relevant_visits": b.get("n_relevant_visits", 0),
            "total_dwell_shopping": b.get("total_dwell_shopping", 0.0),
            "mean_dwell": b.get("mean_dwell", 0.0),
            # exposure (the treatment)
            "followed_agent_link": bool(b.get("followed_agent_link", False)),
            "visited_recommended_brand": bool(b.get("visited_recommended_brand", False)),
            "followed_recommendation": followed,
            # outcome (the documented proxy)
            "max_commerce_depth": depth,
            "reached_shop": depth >= 1,
            "reached_cart": depth >= 2,
            "reached_checkout": depth >= 3,
            "converted": depth >= need_depth,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Conversion metrics (Jeffreys intervals) + funnel transitions
# --------------------------------------------------------------------------- #
def _jeffreys_ci(successes: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    a = (1 - level) / 2
    return (float(beta.ppf(a, successes + 0.5, n - successes + 0.5)),
            float(beta.ppf(1 - a, successes + 0.5, n - successes + 0.5)))


def conversion_report(features: pd.DataFrame,
                      strata: Tuple[str, ...] = ("followed_recommendation",
                                                 "followed_agent_link",
                                                 "visited_recommended_brand")) -> pd.DataFrame:
    """Conversion rate overall and by exposure stratum, with 95% CIs and lift."""
    rows = []

    def add(name, sub):
        n, s = len(sub), int(sub["converted"].sum())
        lo, hi = _jeffreys_ci(s, n)
        rows.append({"stratum": name, "n": n, "conversions": s,
                     "rate": round(s / n, 4) if n else None,
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    add("overall", features)
    for col in strata:
        add(f"{col}=True", features[features[col]])
        add(f"{col}=False", features[~features[col]])
    out = pd.DataFrame(rows)

    lifts = {}
    for col in strata:
        t = out.loc[out["stratum"] == f"{col}=True", "rate"]
        f = out.loc[out["stratum"] == f"{col}=False", "rate"]
        if len(t) and len(f) and f.iloc[0]:
            lifts[col] = round(float(t.iloc[0] / f.iloc[0]), 3)
    out.attrs["lift"] = lifts
    return out


def transition_matrix(turns: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Stage→stage transition counts across the whole journey: the smoothed
    conversation stages (in turn order), then the visited pages' stages (in
    trail order). Long form: from_stage, to_stage, count, prob."""
    from .funnel import STAGES

    order = {s: i for i, s in enumerate(STAGES)}
    counts: Dict[Tuple[str, str], int] = {}
    ev = events[events["event_type"] == "trail_visit"]
    for sid, g in turns.sort_values(["session_id", "session_pos"]).groupby("session_id"):
        stage_col = "journey_stage" if "journey_stage" in g.columns else "funnel_stage"
        seq = [s for s in g[stage_col] if s in order]
        page_seq = ev[ev["session_id"] == str(sid)].sort_values("position")
        seq += [s for s in page_seq["funnel_stage"] if s in order]
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    rows = [{"from_stage": a, "to_stage": b, "count": n} for (a, b), n in counts.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    totals = df.groupby("from_stage")["count"].transform("sum")
    df["prob"] = (df["count"] / totals).round(4)
    return df.sort_values(["from_stage", "to_stage"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The predictive model
# --------------------------------------------------------------------------- #
# The exposure model measures the recommendation weight WITHOUT post-treatment
# mediators: visit counts and dwell are *consequences* of following a
# recommendation, so controlling for them would absorb the very effect we want
# to read (the classic mediator trap). The full model adds them back for best
# prediction — use it to predict, not to interpret exposure.
EXPOSURE_FEATURES = [
    "n_turns", "asks_recommendation_share", "mean_question_sentiment",
    "mean_answer_sentiment", "n_unsolicited_recommendations", "n_links_cited",
    "max_funnel_stage_idx",
    "followed_agent_link", "visited_recommended_brand",
]
FULL_FEATURES = EXPOSURE_FEATURES + ["n_visits", "n_relevant_visits", "mean_dwell"]


def _fit_one(features: pd.DataFrame, cols: List[str], name: str,
             cfg: JourneyConfig) -> Dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = features[cols].astype(float).to_numpy()
    y = features["converted"].astype(int).to_numpy()
    stratify = y if min(np.bincount(y)) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size,
                                          random_state=cfg.seed, stratify=stratify)
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(scaler.transform(Xtr), ytr)

    def _auc(Xs, ys):
        if len(np.unique(ys)) < 2:
            return None
        return float(roc_auc_score(ys, model.predict_proba(scaler.transform(Xs))[:, 1]))

    coefs = pd.DataFrame({
        "model": name,
        "feature": cols,
        "coef_standardized": np.round(model.coef_[0], 4),
        "odds_ratio_per_sd": np.round(np.exp(model.coef_[0]), 4),
    }).sort_values("coef_standardized", key=np.abs, ascending=False).reset_index(drop=True)
    return {"model": model, "scaler": scaler, "coefficients": coefs,
            "auc_train": _auc(Xtr, ytr), "auc_test": _auc(Xte, yte),
            "n_train": int(len(ytr)), "n_test": int(len(yte))}


def fit_conversion_model(features: pd.DataFrame, cfg: JourneyConfig) -> Dict[str, object]:
    """Two logistic conversion models on standardized features.

    * ``exposure`` — conversation features + the exposure flags, **no
      post-treatment mediators**: its ``followed_agent_link`` /
      ``visited_recommended_brand`` coefficients are the headline "weight of
      agent recommendations";
    * ``full`` — adds visit counts and dwell for maximum predictive power.

    Returns {} when the outcome is degenerate (all converted or none).
    """
    y = features["converted"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 12:
        print("[journey] outcome degenerate or too few sessions — skipping model")
        return {}
    exposure = _fit_one(features, EXPOSURE_FEATURES, "exposure", cfg)
    full = _fit_one(features, FULL_FEATURES, "full", cfg)
    coefficients = pd.concat([exposure["coefficients"], full["coefficients"]],
                             ignore_index=True)
    return {"exposure": exposure, "full": full, "coefficients": coefficients,
            "auc_train": full["auc_train"], "auc_test": full["auc_test"],
            "n_train": full["n_train"], "n_test": full["n_test"]}


# --------------------------------------------------------------------------- #
# Output schemas + orchestration
# --------------------------------------------------------------------------- #
_S, _I32, _F64, _B = pa.string(), pa.int32(), pa.float64(), pa.bool_()

EVENT_SCHEMA = pa.schema([
    ("session_id", _S), ("message_id", _S), ("event_type", _S), ("url", _S),
    ("position", _I32), ("dwell_seconds", _F64),
    ("page_category", _S), ("page_subtype", _S), ("funnel_stage", _S),
    ("seller_type", _S), ("page_brand", _S), ("brand_match", _S),
    ("followed_agent_link", _B), ("commerce_depth", _I32),
    ("is_study_relevant", _B), ("page_coverage", _S),
])

JOURNEY_SCHEMA = pa.schema([
    ("session_id", _S), ("n_turns", _I32), ("asks_recommendation_share", _F64),
    ("mean_question_sentiment", _F64), ("mean_answer_sentiment", _F64),
    ("n_unsolicited_recommendations", _I32), ("n_links_cited", _I32),
    ("max_funnel_stage_idx", _I32),
    ("n_visits", _I32), ("n_shopping_visits", _I32), ("n_relevant_visits", _I32),
    ("total_dwell_shopping", _F64), ("mean_dwell", _F64),
    ("followed_agent_link", _B), ("visited_recommended_brand", _B),
    ("followed_recommendation", _B),
    ("max_commerce_depth", _I32), ("reached_shop", _B), ("reached_cart", _B),
    ("reached_checkout", _B), ("converted", _B),
])

TRANSITION_SCHEMA = pa.schema([
    ("from_stage", _S), ("to_stage", _S), ("count", _I32), ("prob", _F64),
])

COEFFICIENT_SCHEMA = pa.schema([
    ("model", _S), ("feature", _S), ("coef_standardized", _F64),
    ("odds_ratio_per_sd", _F64),
])


def run_journey(turns: pd.DataFrame, conversations: pd.DataFrame,
                pages: Optional[pd.DataFrame] = None,
                products: Optional[pd.DataFrame] = None,
                cfg: Optional[JourneyConfig] = None) -> Dict[str, object]:
    """Assemble events, outcomes, metrics and the model; write all parquets."""
    cfg = cfg or JourneyConfig()
    events = build_events(turns, pages, products)
    feats = journey_features(conversations, events, cfg)
    report = conversion_report(feats)
    transitions = transition_matrix(turns, events)
    model = fit_conversion_model(feats, cfg)

    write_parquet(events.to_dict("records"), EVENT_SCHEMA, cfg.path("funnel_events.parquet"))
    write_parquet(feats.to_dict("records"), JOURNEY_SCHEMA, cfg.path("journey_features.parquet"))
    if not transitions.empty:
        write_parquet(transitions.to_dict("records"), TRANSITION_SCHEMA,
                      cfg.path("funnel_transitions.parquet"))
    if model:
        write_parquet(model["coefficients"].to_dict("records"), COEFFICIENT_SCHEMA,
                      cfg.path("model_coefficients.parquet"))

    events = to_dataframe(events.to_dict("records"), EVENT_SCHEMA)
    feats = to_dataframe(feats.to_dict("records"), JOURNEY_SCHEMA)
    n_conv = int(feats["converted"].sum())
    print(f"[journey] events={len(events)} | sessions={len(feats)} | "
          f"conversions={n_conv} ({n_conv / max(1, len(feats)):.1%} @ "
          f"{cfg.conversion_stage}) | lift={report.attrs.get('lift', {})}")
    if model:
        exp = model["exposure"]
        print(f"[journey] full-model AUC train={model['auc_train']:.3f} "
              f"test={model['auc_test'] if model['auc_test'] is None else round(model['auc_test'], 3)} | "
              f"exposure-model weights (no mediators):\n"
              f"{exp['coefficients'].head(5).to_string(index=False)}")
    print(f"[export] {cfg.out_dir}/: funnel_events, journey_features, "
          f"funnel_transitions, model_coefficients")

    return {"events": events, "features": feats, "report": report,
            "transitions": transitions, "model": model, "config": cfg}


def evaluate_against_gt(features: pd.DataFrame, gt: pd.DataFrame) -> Dict[str, float]:
    """Synthetic-only: score the proxy outcome + exposure detection vs truth."""
    if gt is None or gt.empty:
        return {}
    m = features.merge(gt, on="session_id", how="inner")
    if m.empty:
        return {}
    return {
        "n": float(len(m)),
        "conversion_accuracy": float((m["converted"] == m["gt_converted"]).mean()),
        "followed_accuracy": float((m["followed_recommendation"] == m["gt_followed_rec"]).mean()),
    }
