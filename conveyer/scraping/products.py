"""Product extraction and matching to the chat recommendation.

Two jobs, both requested by the user:

1. **Extract product metadata** (price, description, rating, category, …) from a
   parsed :class:`~conveyer.scraping.extract.PageContent`. The primary source is
   schema.org ``Product`` markup in JSON-LD (the same structure Google Shopping
   reads), then OpenGraph ``product:*`` tags, then microdata, then a visible-text
   price heuristic — so we degrade instead of returning nothing.

2. **Relate each product to the product mentioned in the chat** and decide
   whether it *coincides*. A surfaced URL attaches to a turn (``message_id``);
   that turn's ``fact_ai_recommendation`` entities (with ``fact_ai_concept``
   BRAND / CATEGORY attributes) are the "product mentioned with the agent". We
   score brand / name / category / SKU agreement and set ``coincides`` when the
   blended score clears ``ScrapeConfig.coincide_threshold``. Brand normalisation
   reuses ``conveyer.ingest`` so it matches the rest of the codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..ingest import normalize
from .extract import PageContent, iter_jsonld_objects

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "with", "a", "an", "of", "to", "in", "on", "skin",
    "care", "skincare", "product", "products", "oz", "ml", "fl", "pack",
}
_PRICE_NUM = re.compile(r"(\d[\d,]*\.?\d*)")


def _tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(str(text).lower()) if t not in _STOP and len(t) > 1}


def _coerce_text(x: Any) -> str:
    """schema.org values arrive as str, {"@value":..}, {"name":..} or lists."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, dict):
        return _coerce_text(x.get("@value") or x.get("name") or x.get("value") or "")
    if isinstance(x, list):
        for item in x:
            t = _coerce_text(item)
            if t:
                return t
    return ""


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = _PRICE_NUM.search(str(x).replace(",", ""))
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Extracted product
# --------------------------------------------------------------------------- #
@dataclass
class ProductRecord:
    name: str = ""
    brand: str = ""
    description: str = ""
    category: str = ""
    price: Optional[float] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    currency: str = ""
    availability: str = ""
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    sku: str = ""
    gtin: str = ""
    image: str = ""
    source: str = ""          # jsonld | opengraph | microdata | heuristic

    # match-to-chat (populated by match_products)
    matched_message_id: str = ""
    matched_recommendation_id: str = ""
    matched_entity: str = ""
    matched_brand: str = ""
    matched_category: str = ""
    match_type: str = ""      # brand | name | category | sku | none
    match_score: float = 0.0
    coincides: bool = False


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _product_from_jsonld(obj: dict) -> Optional[ProductRecord]:
    types = obj.get("@type")
    types = [types] if isinstance(types, str) else (types or [])
    if not any("product" in str(t).lower() for t in types) and "offers" not in obj:
        return None

    rec = ProductRecord(source="jsonld")
    rec.name = _coerce_text(obj.get("name"))
    rec.brand = _coerce_text(obj.get("brand"))
    rec.description = _coerce_text(obj.get("description"))
    rec.category = _coerce_text(obj.get("category"))
    rec.sku = _coerce_text(obj.get("sku") or obj.get("mpn"))
    rec.gtin = _coerce_text(obj.get("gtin13") or obj.get("gtin12") or obj.get("gtin"))
    rec.image = _coerce_text(obj.get("image"))

    offers = obj.get("offers")
    offer_list = offers if isinstance(offers, list) else ([offers] if offers else [])
    prices: List[float] = []
    for off in offer_list:
        if not isinstance(off, dict):
            continue
        rec.currency = rec.currency or _coerce_text(off.get("priceCurrency"))
        rec.availability = rec.availability or _coerce_text(off.get("availability")).rsplit("/", 1)[-1]
        for key in ("price", "lowPrice", "highPrice"):
            v = _to_float(off.get(key))
            if v is not None:
                prices.append(v)
    if prices:
        rec.price_min, rec.price_max = min(prices), max(prices)
        rec.price = _to_float(offer_list[0].get("price")) if offer_list else None
        if rec.price is None:
            rec.price = rec.price_min

    agg = obj.get("aggregateRating")
    if isinstance(agg, dict):
        rec.rating = _to_float(agg.get("ratingValue"))
        rc = _to_float(agg.get("reviewCount") or agg.get("ratingCount"))
        rec.rating_count = int(rc) if rc is not None else None

    return rec if (rec.name or rec.price is not None) else None


def _products_from_itemlist(obj: dict) -> List[ProductRecord]:
    out: List[ProductRecord] = []
    elements = obj.get("itemListElement")
    if not isinstance(elements, list):
        return out
    for el in elements:
        item = el.get("item") if isinstance(el, dict) else None
        candidate = item if isinstance(item, dict) else (el if isinstance(el, dict) else None)
        if candidate:
            rec = _product_from_jsonld(candidate)
            if rec:
                out.append(rec)
    return out


def _product_from_opengraph(page: PageContent) -> Optional[ProductRecord]:
    og = page.og
    is_product = og.get("og:type", "").lower() == "product" or "product:price:amount" in og
    if not is_product:
        return None
    rec = ProductRecord(source="opengraph")
    rec.name = og.get("og:title", "") or page.title
    rec.description = og.get("og:description", "") or page.meta_description
    rec.brand = og.get("product:brand", "") or og.get("og:brand", "")
    rec.image = og.get("og:image", "")
    rec.price = _to_float(og.get("product:price:amount") or og.get("og:price:amount"))
    rec.currency = og.get("product:price:currency", "") or og.get("og:price:currency", "")
    rec.availability = og.get("product:availability", "")
    return rec if (rec.name or rec.price is not None) else None


def _product_from_microdata(page: PageContent) -> List[ProductRecord]:
    out: List[ProductRecord] = []
    for md in page.microdata:
        if "product" not in str(md.get("type", "")).lower():
            continue
        p = md.get("props", {})
        rec = ProductRecord(source="microdata")
        rec.name = _coerce_text(p.get("name"))
        rec.brand = _coerce_text(p.get("brand"))
        rec.description = _coerce_text(p.get("description"))
        rec.category = _coerce_text(p.get("category"))
        rec.sku = _coerce_text(p.get("sku"))
        rec.price = _to_float(p.get("price"))
        rec.currency = _coerce_text(p.get("priceCurrency"))
        rec.rating = _to_float(p.get("ratingValue"))
        rc = _to_float(p.get("reviewCount") or p.get("ratingCount"))
        rec.rating_count = int(rc) if rc is not None else None
        if rec.name or rec.price is not None:
            out.append(rec)
    return out


def _product_heuristic(page: PageContent) -> Optional[ProductRecord]:
    """Last resort: a page that clearly sells something but ships no markup."""
    if not page.has_price:
        return None
    from .extract import _PRICE_RE
    m = _PRICE_RE.search(page.text)
    rec = ProductRecord(source="heuristic")
    rec.name = (page.h1[0] if page.h1 else page.title)[:200]
    rec.description = page.meta_description
    rec.price = _to_float(m.group(0)) if m else None
    return rec if rec.name else None


def extract_products(page: PageContent, max_products: int = 40) -> List[ProductRecord]:
    """All products found on a page, best structured source first, de-duplicated."""
    out: List[ProductRecord] = []
    for obj in iter_jsonld_objects(page.jsonld):
        rec = _product_from_jsonld(obj)
        if rec:
            out.append(rec)
        out.extend(_products_from_itemlist(obj))

    out.extend(_product_from_microdata(page))
    og = _product_from_opengraph(page)
    if og and not any(_same_product(og, r) for r in out):
        out.append(og)
    if not out:
        h = _product_heuristic(page)
        if h:
            out.append(h)

    # de-duplicate on (name, price) keeping the richest record
    dedup: Dict[tuple, ProductRecord] = {}
    for r in out:
        key = (normalize(r.name), r.price)
        if key not in dedup or _richness(r) > _richness(dedup[key]):
            dedup[key] = r
    return list(dedup.values())[:max_products]


def _same_product(a: ProductRecord, b: ProductRecord) -> bool:
    return normalize(a.name) == normalize(b.name) and normalize(a.name) != ""


def _richness(r: ProductRecord) -> int:
    return sum(bool(x) for x in (r.name, r.brand, r.description, r.category,
                                 r.price, r.rating, r.sku, r.image))


# --------------------------------------------------------------------------- #
# Matching to the chat recommendation
# --------------------------------------------------------------------------- #
@dataclass
class RecMention:
    """One product entity the agent mentioned on the turn a URL is attached to."""
    recommendation_id: str = ""
    message_id: str = ""
    entity_context: str = ""     # LLM-written description
    brands: Set[str] = field(default_factory=set)      # normalized
    sub_brands: Set[str] = field(default_factory=set)  # normalized
    categories: Set[str] = field(default_factory=set)  # normalized
    skus: Set[str] = field(default_factory=set)        # normalized

    def name_tokens(self) -> Set[str]:
        return _tokens(self.entity_context) | {t for b in self.brands for t in _tokens(b)}


def _score_match(product: ProductRecord, page_brand: str, mention: RecMention) -> tuple:
    """Return (score, match_type) for one product against one mention."""
    pbrand = normalize(product.brand) or normalize(page_brand)
    # SKU: exact identity — strongest
    if product.sku and normalize(product.sku) in mention.skus:
        return 1.0, "sku"
    # Brand: normalized set membership / substring either direction
    if pbrand and (pbrand in mention.brands or pbrand in mention.sub_brands
                   or any(pbrand in b or b in pbrand for b in mention.brands if b)):
        base = 0.75
        # bump if the product name also shares tokens with the entity
        overlap = _token_overlap(product, mention)
        return min(1.0, base + 0.25 * overlap), "brand"
    # Name overlap with the entity description
    overlap = _token_overlap(product, mention)
    if overlap >= 0.34:
        return 0.4 + 0.5 * overlap, "name"
    # Category agreement — weak, corroborating only
    pcat = _tokens(product.category)
    if pcat and pcat & mention.categories_tokens():
        return 0.3, "category"
    if overlap > 0:
        return 0.2 * overlap, "name"
    return 0.0, "none"


def _token_overlap(product: ProductRecord, mention: RecMention) -> float:
    pt = _tokens(product.name)
    if not pt:
        return 0.0
    mt = mention.name_tokens()
    if not mt:
        return 0.0
    return len(pt & mt) / len(pt | mt)


# small helper attached to RecMention via monkey-free method
def _categories_tokens(self: RecMention) -> Set[str]:
    return {t for c in self.categories for t in _tokens(c)}


RecMention.categories_tokens = _categories_tokens  # type: ignore[attr-defined]


def match_products(products: List[ProductRecord], page_brand: str,
                   mentions: List[RecMention], message_id: str = "",
                   coincide_threshold: float = 0.5,
                   name_threshold: float = 0.34) -> List[ProductRecord]:
    """Annotate each product with its best-matching chat mention (in place)."""
    for product in products:
        best_score, best_type, best = 0.0, "none", None
        for m in mentions:
            score, mtype = _score_match(product, page_brand, m)
            if score > best_score:
                best_score, best_type, best = score, mtype, m
        product.match_score = round(best_score, 4)
        product.match_type = best_type if best else "none"
        product.coincides = best_score >= coincide_threshold
        product.matched_message_id = message_id
        if best is not None and best_score > 0:
            product.matched_recommendation_id = best.recommendation_id
            product.matched_entity = best.entity_context[:300]
            product.matched_brand = "; ".join(sorted(best.brands))
            product.matched_category = "; ".join(sorted(best.categories))
    return products
