"""Output schema and parquet writer.

The user asked to "order this in a new data frame with its own schema and save
it as parquet". This is that schema — a small **two-table star** in the same
style as the SimilarWeb extracts it is derived from:

* ``fact_scraped_page`` — one row per scraped URL: URL parts, fetch metadata,
  everything extracted from the page, the classifier's verdict, the vendor prior
  and the provenance counts.
* ``fact_scraped_product`` — one row per product found on a page (FK ``page_id``):
  the extracted metadata (price / description / rating / category / …) and the
  match back to the chat recommendation (``coincides`` and why).

Types are pinned with explicit :mod:`pyarrow` schemas and every column is built
as a typed arrow array, so nullable integers and list columns round-trip through
parquet without pandas dtype surprises.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .classify import PageClass, parse_url
from .extract import PageContent
from .fetch import FetchResult
from .products import ProductRecord

_S = pa.string()
_I32 = pa.int32()
_I64 = pa.int64()
_F64 = pa.float64()
_B = pa.bool_()
_LS = pa.list_(pa.string())

PAGE_SCHEMA = pa.schema([
    ("page_id", _S), ("url", _S), ("final_url", _S),
    ("domain", _S), ("subdomain", _S), ("path", _S), ("query", _S),
    ("digital_site_id", _I64),
    # fetch
    ("fetch_status", _S), ("http_status", _I32), ("content_type", _S),
    ("fetched_at", _S), ("fetch_error", _S), ("from_cache", _B), ("parser", _S),
    # extracted page info
    ("lang", _S), ("title", _S), ("meta_description", _S), ("h1", _S),
    ("canonical_url", _S), ("og_type", _S), ("og_site_name", _S),
    ("word_count", _I32), ("n_links", _I32), ("n_images", _I32),
    ("n_products", _I32), ("has_price", _B), ("has_add_to_cart", _B),
    ("has_jsonld", _B), ("schema_types", _LS), ("text_excerpt", _S),
    # classification
    ("page_category", _S), ("page_subtype", _S), ("page_category_confidence", _F64),
    ("seller_type", _S), ("funnel_stage", _S), ("classifier_method", _S),
    ("classification_signals", _LS),
    ("skincare_relevance", _F64), ("is_study_relevant", _B),
    ("primary_brand", _S), ("brand_detected", _LS),
    # provenance + vendor prior
    ("source_tables", _LS), ("times_surfaced", _I32), ("times_recommended", _I32),
    ("times_visited", _I32), ("resulted_in_purchase_any", _B),
    ("n_message_ids", _I32), ("message_ids", _LS),
    ("prior_page_type", _S), ("prior_seller_type", _S),
    ("prior_retailer_brand", _S), ("prior_site_category", _S),
    # browsing-trail retention (from next_10_urls request_time deltas; the last
    # trail entry has no successor, so it never contributes a dwell)
    ("mean_dwell_seconds", _F64), ("total_dwell_seconds", _F64),
    ("mean_trail_position", _F64),
])

PRODUCT_SCHEMA = pa.schema([
    ("product_id", _S), ("page_id", _S), ("url", _S),
    ("name", _S), ("brand", _S), ("description", _S), ("category", _S),
    ("price", _F64), ("price_min", _F64), ("price_max", _F64),
    ("currency", _S), ("availability", _S),
    ("rating", _F64), ("rating_count", _I32), ("sku", _S), ("gtin", _S), ("image", _S),
    ("extraction_source", _S),
    # match to chat recommendation
    ("matched_message_id", _S), ("matched_recommendation_id", _S),
    ("matched_entity", _S), ("matched_brand", _S), ("matched_category", _S),
    ("match_type", _S), ("match_score", _F64), ("coincides", _B),
])


def page_id_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _product_id(page_id: str, rec: ProductRecord, idx: int) -> str:
    key = f"{page_id}|{rec.name}|{rec.sku}|{idx}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _as_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None and not (isinstance(x, float) and pd.isna(x)) else None
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
def page_row(url_row: dict, fetch: FetchResult, content: PageContent,
             cls: PageClass, n_products: int) -> dict:
    url = url_row.get("url") or fetch.url
    u = parse_url(url)
    return {
        "page_id": page_id_for(url),
        "url": url,
        "final_url": fetch.final_url or url,
        "domain": url_row.get("domain") or u.registrable,
        "subdomain": u.subdomain,
        "path": u.path,
        "query": u.query,
        "digital_site_id": _as_int(url_row.get("digital_site_id")),
        "fetch_status": fetch.status,
        "http_status": _as_int(fetch.http_status),
        "content_type": fetch.content_type,
        "fetched_at": fetch.fetched_at,
        "fetch_error": fetch.error,
        "from_cache": bool(fetch.from_cache),
        "parser": content.parser,
        "lang": content.lang,
        "title": content.title,
        "meta_description": content.meta_description,
        "h1": content.h1[0] if content.h1 else "",
        "canonical_url": content.canonical_url,
        "og_type": content.og.get("og:type", ""),
        "og_site_name": content.og.get("og:site_name", ""),
        "word_count": _as_int(content.word_count) or 0,
        "n_links": _as_int(content.n_links) or 0,
        "n_images": _as_int(content.n_images) or 0,
        "n_products": int(n_products),
        "has_price": bool(content.has_price),
        "has_add_to_cart": bool(content.has_add_to_cart),
        "has_jsonld": bool(content.has_jsonld),
        "schema_types": content.schema_types(),
        "text_excerpt": content.excerpt(2000),
        "page_category": cls.page_category,
        "page_subtype": cls.page_subtype,
        "page_category_confidence": float(cls.confidence),
        "seller_type": cls.seller_type,
        "funnel_stage": cls.funnel_stage,
        "classifier_method": cls.method,
        "classification_signals": list(cls.signals),
        "skincare_relevance": float(cls.skincare_relevance),
        "is_study_relevant": bool(cls.is_study_relevant),
        "primary_brand": cls.primary_brand,
        "brand_detected": list(cls.brand_detected),
        "source_tables": list(url_row.get("sources", []) or []),
        "times_surfaced": _as_int(url_row.get("times_surfaced")) or 0,
        "times_recommended": _as_int(url_row.get("times_recommended")) or 0,
        "times_visited": _as_int(url_row.get("times_visited")) or 0,
        "resulted_in_purchase_any": bool(url_row.get("resulted_in_purchase_any", False)),
        "n_message_ids": len(url_row.get("message_ids", []) or []),
        "message_ids": [str(m) for m in (url_row.get("message_ids", []) or [])],
        "prior_page_type": _str_or_empty(url_row.get("page_type")),
        "prior_seller_type": _str_or_empty(url_row.get("seller_type")),
        "prior_retailer_brand": _str_or_empty(url_row.get("retailer_brand")),
        "prior_site_category": _str_or_empty(url_row.get("site_category")),
        "mean_dwell_seconds": _f(url_row.get("mean_dwell_seconds")),
        "total_dwell_seconds": _f(url_row.get("total_dwell_seconds")),
        "mean_trail_position": _f(url_row.get("mean_trail_position")),
    }


def product_rows(page_id: str, url: str, products: List[ProductRecord]) -> List[dict]:
    rows = []
    for i, r in enumerate(products):
        rows.append({
            "product_id": _product_id(page_id, r, i),
            "page_id": page_id,
            "url": url,
            "name": r.name, "brand": r.brand, "description": r.description[:1000],
            "category": r.category,
            "price": _f(r.price), "price_min": _f(r.price_min), "price_max": _f(r.price_max),
            "currency": r.currency, "availability": r.availability,
            "rating": _f(r.rating), "rating_count": _as_int(r.rating_count),
            "sku": r.sku, "gtin": r.gtin, "image": r.image,
            "extraction_source": r.source,
            "matched_message_id": r.matched_message_id,
            "matched_recommendation_id": r.matched_recommendation_id,
            "matched_entity": r.matched_entity, "matched_brand": r.matched_brand,
            "matched_category": r.matched_category, "match_type": r.match_type,
            "match_score": float(r.match_score), "coincides": bool(r.coincides),
        })
    return rows


def _f(x) -> Optional[float]:
    try:
        return float(x) if x is not None and not (isinstance(x, float) and pd.isna(x)) else None
    except (ValueError, TypeError):
        return None


def _str_or_empty(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


# --------------------------------------------------------------------------- #
# Arrow / parquet
# --------------------------------------------------------------------------- #
def to_arrow_table(rows: List[dict], schema: pa.Schema) -> pa.Table:
    """Build a typed arrow table column-by-column (nullable-int / list safe)."""
    cols = {}
    for field in schema:
        vals = [r.get(field.name) for r in rows]
        cols[field.name] = pa.array(vals, type=field.type)
    return pa.table(cols, schema=schema)


def to_dataframe(rows: List[dict], schema: pa.Schema) -> pd.DataFrame:
    if not rows:
        return to_arrow_table([], schema).to_pandas()
    return to_arrow_table(rows, schema).to_pandas()


def write_parquet(rows: List[dict], schema: pa.Schema, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    table = to_arrow_table(rows, schema)
    pq.write_table(table, path)
    return path


def build_frames(page_rows: List[dict], product_rows_: List[dict]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (to_dataframe(page_rows, PAGE_SCHEMA),
            to_dataframe(product_rows_, PRODUCT_SCHEMA))
