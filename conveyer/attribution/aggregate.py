"""Session grain → the grain sales and MMM actually speak: brand-market-week.

Modules 1–3 answer *per journey*. RMS/Omnisales, CPS and every MMM ever built
answer *per brand, market, category, week*. Nothing can be joined until the
panel is aggregated up, and the aggregation is where three dimensions the
current pipeline never had get introduced:

* ``llm_platform`` — ChatGPT vs Gemini vs Claude vs Perplexity. Their answer
  styles, citation rates and referral behaviour differ enough that pooling
  them hides the effect; it is also the dimension the business will ask about
  first.
* ``category`` — face skincare today, more later. Brand lexicons, conversion
  levels and the offline share are category properties.
* ``market`` — the panel frame, the universe and the sales data are all
  per-market.

They are read from the input when present and filled from
:class:`AggregateConfig` when not, so a single-category ChatGPT US extract
still aggregates correctly and a multi-platform extract needs no code change.

Two tables come out:

``fact_ai_exposure_weekly``  grain = week × market × category × platform × brand
                             — the brand's presence in AI answers and the
                             behaviour that followed. Feeds the sales bridge
                             and the MMM's AI regressor.
``fact_ai_category_weekly``  grain = week × market × category × platform
                             — category-level conversation volume and reach.
                             The denominator for share-of-voice and the
                             top-down cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa

from ..brands import normalize_brand
from ..schema_utils import write_parquet
from .panel import PanelFrame, project_counts

DIMENSIONS = ("week", "market", "category", "llm_platform")


@dataclass
class AggregateConfig:
    """Defaults for dimensions the extract does not carry."""

    market: str = "US"
    category: str = "face skincare"
    llm_platform: str = "chatgpt"
    week_start: str = "W-MON"        # ISO weeks, Monday-anchored
    out_dir: str = "outputs/attribution"

    def path(self, name: str) -> str:
        import os
        return os.path.join(self.out_dir, name)


# --------------------------------------------------------------------------- #
# Dimension resolution
# --------------------------------------------------------------------------- #
def _dimension_frame(turns: pd.DataFrame, cfg: AggregateConfig) -> pd.DataFrame:
    """One row per session: its week + the three business dimensions.

    A session's week is the week of its **first** prompt: journeys that
    straddle a week boundary belong to the week they started, so a session is
    never split across two rows of a weekly model.
    """
    df = turns.copy()
    if "prompt_dt" not in df.columns:
        df["prompt_dt"] = pd.to_datetime(df.get("prompt_datetime"), errors="coerce")
    df["prompt_dt"] = pd.to_datetime(df["prompt_dt"], errors="coerce")
    if "user_id" not in df.columns:
        df["user_id"] = df["session_id"]

    for dim, default in (("market", cfg.market), ("category", cfg.category),
                         ("llm_platform", cfg.llm_platform)):
        if dim not in df.columns:
            df[dim] = default
        else:
            df[dim] = df[dim].fillna(default).replace("", default)

    first = (df.sort_values(["session_id", "prompt_dt"])
               .groupby("session_id", as_index=False)
               .agg(prompt_dt=("prompt_dt", "first"), user_id=("user_id", "first"),
                    market=("market", "first"), category=("category", "first"),
                    llm_platform=("llm_platform", "first")))
    period = first["prompt_dt"].dt.to_period(cfg.week_start)
    first["week"] = period.dt.start_time.dt.strftime("%Y-%m-%d").fillna("unknown")
    first["session_id"] = first["session_id"].astype(str)
    return first


# --------------------------------------------------------------------------- #
# Session → brand relations
# --------------------------------------------------------------------------- #
def _brand_set(cell) -> Set[str]:
    if cell is None:
        return set()
    if isinstance(cell, (list, tuple, np.ndarray, set)):
        return {normalize_brand(str(b)) for b in cell if str(b).strip()}
    return set()


def session_brand_relations(turns: pd.DataFrame, events: Optional[pd.DataFrame],
                            features: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Long table: one row per (session, brand) with what happened to it.

    ``mentioned`` / ``unsolicited`` come from the conversation (module 1),
    ``cited`` from links the answer surfaced, ``visited`` from behaviour
    (module 3). ``converted`` is attributed to **one** brand per session — the
    brand of the deepest commerce page the user reached — so summing
    conversions across brands cannot exceed the session count. Attributing it
    to every visited brand would inflate every brand's conversion count in
    exactly the multi-brand journeys that matter most.
    """
    rec: Dict[tuple, dict] = {}

    def cell(sid: str, brand: str) -> dict:
        key = (str(sid), brand)
        if key not in rec:
            rec[key] = {"session_id": str(sid), "brand": brand,
                        "mentioned": False, "unsolicited": False, "cited": False,
                        "visited": False, "followed": False, "converted": False,
                        "max_depth": 0}
        return rec[key]

    for sid, g in turns.groupby("session_id"):
        mentioned: Set[str] = set()
        unsolicited: Set[str] = set()
        for _, t in g.iterrows():
            mentioned |= _brand_set(t.get("brands_a"))
            unsolicited |= _brand_set(t.get("brands_unsolicited"))
        for b in (mentioned | unsolicited) - {""}:
            r = cell(sid, b)
            r["mentioned"] = b in mentioned
            r["unsolicited"] = b in unsolicited

    if events is not None and not events.empty:
        ev = events.copy()
        ev["brand"] = ev["page_brand"].fillna("").map(
            lambda b: normalize_brand(str(b)) if str(b).strip() else "")
        ev = ev[ev["brand"] != ""]
        behav_all = ev[ev["event_type"].isin(["ai_click", "trail_visit"])]
        for (sid, brand), g in ev.groupby(["session_id", "brand"]):
            r = cell(sid, brand)
            r["cited"] = r["cited"] or bool((g["event_type"] == "cited").any())
            behav = g[g["event_type"].isin(["ai_click", "trail_visit"])]
            if len(behav):
                r["visited"] = True
                r["followed"] = bool(behav["followed_agent_link"].any())
                r["max_depth"] = int(behav["commerce_depth"].max())

        # conversion → the single deepest-commerce brand of the session
        if features is not None and not features.empty:
            converted = set(features.loc[features["converted"].astype(bool),
                                         "session_id"].astype(str))
            sort_cols = ["commerce_depth"] + (
                ["position"] if "position" in behav_all.columns else [])
            for sid, g in behav_all.groupby("session_id"):
                if str(sid) not in converted:
                    continue
                g = g.sort_values(sort_cols,
                                  ascending=[False] + [True] * (len(sort_cols) - 1))
                if not len(g) or int(g.iloc[0]["commerce_depth"]) <= 0:
                    continue
                cell(sid, str(g.iloc[0]["brand"]))["converted"] = True

    return pd.DataFrame(list(rec.values()),
                        columns=["session_id", "brand", "mentioned", "unsolicited",
                                 "cited", "visited", "followed", "converted",
                                 "max_depth"])


# --------------------------------------------------------------------------- #
# The weekly fact tables
# --------------------------------------------------------------------------- #
def build_weekly(turns: pd.DataFrame, events: Optional[pd.DataFrame] = None,
                 features: Optional[pd.DataFrame] = None,
                 cfg: Optional[AggregateConfig] = None,
                 frames: Optional[Dict[str, PanelFrame]] = None,
                 ) -> Dict[str, pd.DataFrame]:
    """Aggregate a run into the two weekly fact tables (optionally projected).

    ``frames`` maps market → :class:`PanelFrame`; when given, every count also
    gets a ``*_projected`` column (and a 95% sampling interval where a
    denominator exists). Without frames the tables stay in panel units — the
    MMM can use them as an index, the sales bridge cannot.
    """
    cfg = cfg or AggregateConfig()
    dims = _dimension_frame(turns, cfg)
    rel = session_brand_relations(turns, events, features)

    feat = (features[["session_id", "converted", "followed_recommendation"]].copy()
            if features is not None and not features.empty
            else pd.DataFrame(columns=["session_id", "converted",
                                       "followed_recommendation"]))
    if not feat.empty:
        feat["session_id"] = feat["session_id"].astype(str)

    # -- category grain --------------------------------------------------- #
    cat_base = dims.merge(feat, on="session_id", how="left")
    for c in ("converted", "followed_recommendation"):
        cat_base[c] = cat_base[c].fillna(False).astype(bool)
    category = (cat_base.groupby(list(DIMENSIONS), as_index=False)
                .agg(n_sessions=("session_id", "nunique"),
                     n_users=("user_id", "nunique"),
                     n_sessions_followed=("followed_recommendation", "sum"),
                     n_sessions_converted=("converted", "sum")))
    for c in ("n_sessions_followed", "n_sessions_converted"):
        category[c] = category[c].astype(int)
    category["conversion_rate_proxy"] = (
        category["n_sessions_converted"] / category["n_sessions"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    # -- brand grain ------------------------------------------------------- #
    if rel.empty:
        brand = pd.DataFrame(columns=list(DIMENSIONS) + ["brand"])
    else:
        j = rel.merge(dims, on="session_id", how="inner")
        # a brand can be visited without ever being named in the chat (organic
        # navigation), so "visits per exposed session" needs the *conjunction*
        # in the numerator or the rate can exceed 1
        j["mentioned_and_visited"] = j["mentioned"] & j["visited"]
        brand = (j.groupby(list(DIMENSIONS) + ["brand"], as_index=False)
                  .agg(n_sessions_mentioned=("mentioned", "sum"),
                       n_sessions_mentioned_visited=("mentioned_and_visited", "sum"),
                       n_sessions_unsolicited=("unsolicited", "sum"),
                       n_sessions_cited=("cited", "sum"),
                       n_sessions_visited=("visited", "sum"),
                       n_sessions_followed=("followed", "sum"),
                       n_sessions_converted=("converted", "sum")))
        for c in brand.columns:
            if c.startswith("n_"):
                brand[c] = brand[c].astype(int)
        brand = brand.merge(category[list(DIMENSIONS) + ["n_sessions", "n_users"]],
                            on=list(DIMENSIONS), how="left")
        brand = brand.rename(columns={"n_sessions": "n_sessions_category"})
        # share of voice: the brand's presence among all brand mentions in the
        # cell — the identification-friendly MMM variable, because it is a
        # *share* and so does not simply track the AI-adoption trend
        totals = (brand.groupby(list(DIMENSIONS))["n_sessions_mentioned"]
                       .transform("sum").replace(0, np.nan))
        brand["ai_share_of_voice"] = (brand["n_sessions_mentioned"] / totals).fillna(0.0).round(4)
        brand["exposure_rate"] = (brand["n_sessions_mentioned"]
                                  / brand["n_sessions_category"].replace(0, np.nan)
                                  ).fillna(0.0).round(4)
        brand["visit_rate_given_exposure"] = (
            brand["n_sessions_mentioned_visited"]
            / brand["n_sessions_mentioned"].replace(0, np.nan)).fillna(0.0).round(4)

    # -- projection --------------------------------------------------------- #
    if frames:
        brand = _project_by_market(brand, frames,
                                   ["n_sessions_mentioned", "n_sessions_unsolicited",
                                    "n_sessions_visited", "n_sessions_converted"],
                                   base_col="n_sessions_category")
        category = _project_by_market(category, frames,
                                      ["n_sessions", "n_users",
                                       "n_sessions_converted"], base_col=None)

    return {"brand_weekly": brand, "category_weekly": category,
            "relations": rel, "config": cfg}


def _project_by_market(df: pd.DataFrame, frames: Dict[str, PanelFrame],
                       cols: List[str], base_col: Optional[str]) -> pd.DataFrame:
    if df.empty or "market" not in df.columns:
        return df
    out = []
    for market, g in df.groupby("market"):
        frame = frames.get(str(market)) or frames.get("*")
        if frame is None:
            g = g.copy()
            for c in cols:
                g[c + "_projected"] = np.nan   # unprojectable, and visibly so
            out.append(g)
            continue
        out.append(project_counts(g, cols, frame, base_col=base_col))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Schemas + persistence
# --------------------------------------------------------------------------- #
_S, _I32, _F64 = pa.string(), pa.int32(), pa.float64()

BRAND_WEEKLY_SCHEMA = pa.schema([
    ("week", _S), ("market", _S), ("category", _S), ("llm_platform", _S),
    ("brand", _S),
    ("n_sessions_category", _I32),
    ("n_sessions_mentioned", _I32), ("n_sessions_mentioned_visited", _I32),
    ("n_sessions_unsolicited", _I32),
    ("n_sessions_cited", _I32), ("n_sessions_visited", _I32),
    ("n_sessions_followed", _I32), ("n_sessions_converted", _I32),
    ("ai_share_of_voice", _F64), ("exposure_rate", _F64),
    ("visit_rate_given_exposure", _F64),
    ("n_sessions_mentioned_projected", _F64),
    ("n_sessions_unsolicited_projected", _F64),
    ("n_sessions_visited_projected", _F64),
    ("n_sessions_converted_projected", _F64),
])

CATEGORY_WEEKLY_SCHEMA = pa.schema([
    ("week", _S), ("market", _S), ("category", _S), ("llm_platform", _S),
    ("n_sessions", _I32), ("n_users", _I32), ("n_sessions_followed", _I32),
    ("n_sessions_converted", _I32), ("conversion_rate_proxy", _F64),
    ("n_sessions_projected", _F64), ("n_users_projected", _F64),
])


def write_weekly(tables: Dict[str, pd.DataFrame], cfg: AggregateConfig) -> List[str]:
    paths = []
    for key, schema, name in (("brand_weekly", BRAND_WEEKLY_SCHEMA,
                               "fact_ai_exposure_weekly.parquet"),
                              ("category_weekly", CATEGORY_WEEKLY_SCHEMA,
                               "fact_ai_category_weekly.parquet")):
        df = tables.get(key)
        if df is None or df.empty:
            continue
        rows = df.reindex(columns=[f.name for f in schema]).to_dict("records")
        paths.append(write_parquet(rows, schema, cfg.path(name)))
    return paths
