"""Where the URLs come from: the SimilarWeb clickstream star schema.

The "columns with links and web pages" the user pointed at live in the
clickstream dataset (see ``docs/DATA_DICTIONARY.md`` §2):

* ``dim_digital_site.url`` — the 265k distinct surfaced URLs, each with the
  vendor's ``page_type`` / ``seller_type`` / ``category`` / ``retailer_brand`` /
  ``extracted_product_skus`` (our classifier's prior + weak labels);
* ``fact_ai_click_through.surfaced_url`` — every surfacing event, carrying the
  ``message_id`` (which turn), ``was_recommended`` / ``was_visited`` /
  ``resulted_in_purchase`` (provenance);
* ``simweb_input_file.a_links_source`` / ``next_10_urls`` — links cited in the
  answer and the raw browsing trail.

This module distils those into one **candidate-URL table** (grain = URL, with
provenance counts and the vendor prior) and a **mentions map**
``message_id → [RecMention]`` built from ``fact_ai_recommendation`` +
``fact_ai_concept`` — the "product mentioned in the chat with the agent" that
``conveyer.scraping.products`` matches page products against.

It also works from **``simweb_input_file.parquet`` alone** (point
``clickstream_dir`` at the file): the trail in ``next_10_urls`` supplies the
URLs, and the ``request_time`` deltas between consecutive requests supply a
**dwell-time / retention signal** (`mean_dwell_seconds`,
`total_dwell_seconds`, `mean_trail_position`) — how long the user stayed on a
page before moving on. The last trail entry has no successor, so its dwell is
never guessed.
"""

from __future__ import annotations

import ast
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..ingest import normalize
from .classify import parse_url
from .config import ScrapeConfig
from .products import RecMention

# concept_name values that identify a product entity (see data dictionary §2.2)
_BRAND_CONCEPTS = {"BRAND"}
_SUBBRAND_CONCEPTS = {"SUB_BRAND"}
_CATEGORY_CONCEPTS = {"CATEGORY", "SUBCATEGORY", "APPLICATION_AREA",
                      "TARGET_SKIN_CONCERNCONDITION"}


@dataclass
class ScrapeSources:
    """Uniform hand-off to the pipeline (real *or* synthetic)."""
    urls: pd.DataFrame                                   # candidate-URL table
    mentions: Dict[str, List[RecMention]] = field(default_factory=dict)
    html_by_url: Optional[Dict[str, str]] = None        # offline corpus (synthetic/cache)
    source: str = ""
    ground_truth: Optional[pd.DataFrame] = None         # synthetic only: labels for evaluation


# --------------------------------------------------------------------------- #
# Loading the parquet star schema
# --------------------------------------------------------------------------- #
_TABLE_HINTS = {
    "dim_digital_site": ("dim_digital_site", "digital_site", "dim_site"),
    "click_through": ("click_through", "fact_ai_click_through", "clickthrough"),
    "recommendation": ("recommendation", "fact_ai_recommendation"),
    "concept": ("concept", "fact_ai_concept"),
    "input_file": ("input_file", "simweb_input_file", "input"),
}


def load_clickstream(cfg: ScrapeConfig) -> Optional[Dict[str, pd.DataFrame]]:
    """Load the clickstream parquet tables from ``cfg.clickstream_dir``.

    ``clickstream_dir`` may be the star-schema *directory* or a single parquet
    *file* (e.g. just ``simweb_input_file.parquet`` — the URLs in
    ``next_10_urls`` / ``a_links_source`` / ``ai_click`` are enough to run the
    whole pipeline). Returns None if nothing is found, so the pipeline can fall
    back to the synthetic corpus.
    """
    d = cfg.clickstream_dir
    if os.path.isfile(d) and d.endswith(".parquet"):
        try:
            return {"input_file": pd.read_parquet(d)}
        except Exception:
            return None
    if not os.path.isdir(d):
        return None
    files = glob.glob(os.path.join(d, "*.parquet")) + glob.glob(os.path.join(d, "*", "*.parquet"))
    out: Dict[str, pd.DataFrame] = {}
    for key, hints in _TABLE_HINTS.items():
        for f in files:
            base = os.path.basename(f).lower()
            if any(h in base for h in hints):
                try:
                    out[key] = pd.read_parquet(f)
                except Exception:
                    continue
                break
    return out or None


# --------------------------------------------------------------------------- #
# List-cell coercion + browsing-trail parsing
# --------------------------------------------------------------------------- #
def _as_records(cell: Any) -> List[Any]:
    """Coerce a link-list cell into a plain Python list.

    Real parquet gives ``next_10_urls`` back as a **numpy array of dicts**
    (``ingest.to_list`` would silently return [] for those); other exports give
    Python lists, or a stringified repr with single quotes. Handle all of them.
    """
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        return cell.tolist()
    if isinstance(cell, (list, tuple)):
        return list(cell)
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if isinstance(cell, str):
        s = cell.strip()
        if not s or s.lower() == "none":
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(s)
                if isinstance(v, (list, tuple)):
                    return list(v)
                return [v]
            except (ValueError, SyntaxError):
                continue
        return [s]
    return [cell]


def _event_url(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("requested_site") or item.get("url") or "").strip()
    return str(item or "").strip()


def _event_time_ms(item: Any) -> Optional[int]:
    if not isinstance(item, dict):
        return None
    t = item.get("request_time")
    try:
        return int(str(t))
    except (TypeError, ValueError):
        return None


def trail_events(cell: Any) -> List[Dict[str, Any]]:
    """Parse one ``next_10_urls`` cell into ordered events with dwell times.

    ``dwell_seconds`` for event *i* is the gap until request *i+1* — how long
    the user stayed before moving on. The **last** event has no successor, so
    its dwell is None (never guessed). Position is 1-based within the trail.
    """
    items = _as_records(cell)
    events = [{"url": _event_url(it), "t_ms": _event_time_ms(it)} for it in items]
    events = [e for e in events if e["url"]]
    if all(e["t_ms"] is not None for e in events) and len(events) > 1:
        events.sort(key=lambda e: e["t_ms"])
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(events):
        dwell = None
        if i + 1 < len(events) and e["t_ms"] is not None and events[i + 1]["t_ms"] is not None:
            delta = (events[i + 1]["t_ms"] - e["t_ms"]) / 1000.0
            dwell = delta if delta >= 0 else None
        out.append({"url": e["url"], "position": i + 1, "dwell_seconds": dwell})
    return out


# --------------------------------------------------------------------------- #
# Chat mentions (product mentioned with the agent)
# --------------------------------------------------------------------------- #
def build_mentions(recommendation: Optional[pd.DataFrame],
                   concept: Optional[pd.DataFrame],
                   cfg: ScrapeConfig) -> Dict[str, List[RecMention]]:
    """message_id -> [RecMention]. Concept attributes (BRAND/CATEGORY/…) are
    grouped onto their recommendation, then onto the turn."""
    if recommendation is None or recommendation.empty:
        return {}

    # concept: recommendation_id (=fact_id) -> {brands, sub_brands, categories}
    by_rec: Dict[str, Dict[str, set]] = {}
    if concept is not None and not concept.empty \
            and cfg.col_concept_fact_id in concept.columns:
        cname, cval = cfg.col_concept_name, cfg.col_concept_value
        for fid, name, val in zip(concept[cfg.col_concept_fact_id],
                                  concept.get(cname, ""), concept.get(cval, "")):
            slot = by_rec.setdefault(str(fid), {"brands": set(), "sub_brands": set(),
                                                 "categories": set()})
            n = str(name).upper()
            v = normalize(val)
            if not v or v == "none":
                continue
            if n in _BRAND_CONCEPTS:
                slot["brands"].add(v)
            elif n in _SUBBRAND_CONCEPTS:
                slot["sub_brands"].add(v)
            elif n in _CATEGORY_CONCEPTS:
                slot["categories"].add(v)

    mentions: Dict[str, List[RecMention]] = {}
    rid_col, mid_col = cfg.col_rec_id, cfg.col_rec_message_id
    ctx_col = cfg.col_rec_context
    for _, r in recommendation.iterrows():
        rid = str(r.get(rid_col, ""))
        mid = str(r.get(mid_col, ""))
        if not mid:
            continue
        attrs = by_rec.get(rid, {})
        m = RecMention(
            recommendation_id=rid, message_id=mid,
            entity_context=str(r.get(ctx_col, "") or ""),
            brands=set(attrs.get("brands", set())),
            sub_brands=set(attrs.get("sub_brands", set())),
            categories=set(attrs.get("categories", set())),
        )
        mentions.setdefault(mid, []).append(m)
    return mentions


# --------------------------------------------------------------------------- #
# Candidate URLs
# --------------------------------------------------------------------------- #
def _sites_prior(dim_site: Optional[pd.DataFrame], cfg: ScrapeConfig) -> Dict[int, dict]:
    if dim_site is None or dim_site.empty:
        return {}
    keep = [cfg.col_site_id, cfg.col_url, cfg.col_domain, cfg.col_page_type,
            cfg.col_seller_type, cfg.col_retailer_brand, cfg.col_site_category,
            cfg.col_extracted_skus]
    cols = [c for c in keep if c in dim_site.columns]
    prior: Dict[int, dict] = {}
    for row in dim_site[cols].itertuples(index=False):
        rec = dict(zip(cols, row))
        sid = rec.get(cfg.col_site_id)
        if sid is not None:
            prior[sid] = rec
    return prior


def candidate_urls(tables: Dict[str, pd.DataFrame], cfg: ScrapeConfig) -> pd.DataFrame:
    """Build the distinct-URL candidate table with provenance + vendor prior."""
    click = tables.get("click_through")
    dim_site = tables.get("dim_digital_site")
    prior = _sites_prior(dim_site, cfg)

    rows: Dict[str, dict] = {}

    def _bump(url: str, site_id=None, mid=None, recommended=False, visited=False,
              purchased=False, source="click_through", dwell=None, position=None):
        url = str(url or "").strip()
        if not url or url.lower() == "none":
            return
        r = rows.setdefault(url, {
            "url": url, "digital_site_id": site_id, "message_ids": set(),
            "times_surfaced": 0, "times_recommended": 0, "times_visited": 0,
            "resulted_in_purchase_any": False, "sources": set(),
            "_dwell_sum": 0.0, "_dwell_n": 0, "_pos_sum": 0.0, "_pos_n": 0,
        })
        r["times_surfaced"] += 1
        r["times_recommended"] += int(bool(recommended))
        r["times_visited"] += int(bool(visited))
        r["resulted_in_purchase_any"] = r["resulted_in_purchase_any"] or bool(purchased)
        r["sources"].add(source)
        if site_id is not None and r["digital_site_id"] is None:
            r["digital_site_id"] = site_id
        if mid:
            r["message_ids"].add(str(mid))
        if dwell is not None:
            r["_dwell_sum"] += float(dwell)
            r["_dwell_n"] += 1
        if position is not None:
            r["_pos_sum"] += float(position)
            r["_pos_n"] += 1

    if click is not None and not click.empty:
        cols = click.columns
        for row in click.itertuples(index=False):
            d = dict(zip(cols, row))
            _bump(d.get(cfg.col_surfaced_url), site_id=d.get(cfg.col_click_site_id),
                  mid=d.get(cfg.col_click_message_id),
                  recommended=bool(d.get(cfg.col_was_recommended, False)),
                  visited=bool(d.get(cfg.col_was_visited, False)),
                  purchased=bool(d.get(cfg.col_resulted_in_purchase, False)),
                  source="click_through")

    # answer-cited links, client-side clicks and the raw browsing trail
    inp = tables.get("input_file")
    if inp is not None and not inp.empty:
        mids = inp.get(cfg.col_click_message_id, pd.Series([""] * len(inp)))
        for col, src in (("a_links_source", "a_links_source"), ("ai_click", "ai_click")):
            if col not in inp.columns:
                continue
            for mid, cell in zip(mids, inp[col]):
                for item in _as_records(cell):
                    _bump(_event_url(item), mid=mid,
                          recommended=(src == "a_links_source"),
                          visited=(src == "ai_click"), source=src)
        if "next_10_urls" in inp.columns:
            # the trail: visited by construction; request_time deltas give the
            # dwell before the user moved on (relevancy / retention signal)
            for mid, cell in zip(mids, inp["next_10_urls"]):
                for ev in trail_events(cell):
                    _bump(ev["url"], mid=mid, visited=True, source="next_10_urls",
                          dwell=ev["dwell_seconds"], position=ev["position"])

    if not rows:
        return pd.DataFrame(columns=[
            "url", "digital_site_id", "domain", "page_type", "seller_type",
            "retailer_brand", "site_category", "extracted_skus", "message_ids",
            "times_surfaced", "times_recommended", "times_visited",
            "resulted_in_purchase_any", "sources", "mean_dwell_seconds",
            "total_dwell_seconds", "mean_trail_position"])

    out = []
    for url, r in rows.items():
        p = prior.get(r["digital_site_id"], {})
        u = parse_url(url)
        out.append({
            "url": url,
            "digital_site_id": r["digital_site_id"],
            "domain": p.get(cfg.col_domain) or u.registrable,
            "page_type": p.get(cfg.col_page_type),
            "seller_type": p.get(cfg.col_seller_type),
            "retailer_brand": _clean_none(p.get(cfg.col_retailer_brand)),
            "site_category": _clean_none(p.get(cfg.col_site_category)),
            "extracted_skus": _clean_none(p.get(cfg.col_extracted_skus)),
            "message_ids": sorted(r["message_ids"]),
            "times_surfaced": r["times_surfaced"],
            "times_recommended": r["times_recommended"],
            "times_visited": r["times_visited"],
            "resulted_in_purchase_any": r["resulted_in_purchase_any"],
            "sources": sorted(r["sources"]),
            "mean_dwell_seconds": round(r["_dwell_sum"] / r["_dwell_n"], 3) if r["_dwell_n"] else None,
            "total_dwell_seconds": round(r["_dwell_sum"], 3) if r["_dwell_n"] else None,
            "mean_trail_position": round(r["_pos_sum"] / r["_pos_n"], 3) if r["_pos_n"] else None,
        })
    df = pd.DataFrame(out)

    if cfg.only_recommended:
        df = df[df["times_recommended"] > 0].reset_index(drop=True)
    if cfg.dedupe_by == "domain":
        df = (df.sort_values("times_surfaced", ascending=False)
                .drop_duplicates("domain").reset_index(drop=True))
    df = df.sort_values("times_surfaced", ascending=False).reset_index(drop=True)
    if cfg.max_urls:
        df = df.head(cfg.max_urls).reset_index(drop=True)
    return df


def _clean_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return None if s == "" or s.lower() == "none" else s


def build_sources(cfg: ScrapeConfig) -> Optional[ScrapeSources]:
    """Real-data sources from the clickstream parquet dir, or None if absent."""
    tables = load_clickstream(cfg)
    if not tables:
        return None
    urls = candidate_urls(tables, cfg)
    mentions = build_mentions(tables.get("recommendation"), tables.get("concept"), cfg)
    return ScrapeSources(urls=urls, mentions=mentions, html_by_url=None,
                         source=cfg.clickstream_dir)


def chat_brands_for(url_row: pd.Series, mentions: Dict[str, List[RecMention]]) -> set:
    """Union of brands the agent mentioned across the turns this URL was surfaced on."""
    brands: set = set()
    for mid in url_row.get("message_ids", []) or []:
        for m in mentions.get(str(mid), []):
            brands |= m.brands | m.sub_brands
    return brands


def mentions_for(url_row: pd.Series, mentions: Dict[str, List[RecMention]]) -> List[RecMention]:
    out: List[RecMention] = []
    for mid in url_row.get("message_ids", []) or []:
        out.extend(mentions.get(str(mid), []))
    return out
