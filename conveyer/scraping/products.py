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
    match_type: str = ""      # sku | brand+name | brand | name | category | none
    match_score: float = 0.0
    match_strength: str = "none"   # exact | strong | likely | none — only
                                   # exact/strong may set coincides
    match_signals: List[str] = field(default_factory=list)  # which evidence fired
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


def extract_products(page: PageContent, max_products: int = 40,
                     include_heuristic: bool = True) -> List[ProductRecord]:
    """All products found on a page, best structured source first, de-duplicated.

    ``include_heuristic=False`` disables the last-resort text-price heuristic —
    the pipeline passes that on transactional URLs (cart/checkout/order),
    where an h1 like "All Carts" plus an incidental "$10" in the copy would
    otherwise mint a phantom product."""
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
    if not out and include_heuristic:
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


# -- normalization & similarity primitives ---------------------------------- #
# Sizes and counts are packaging, not identity: "Moisturizing Cream 19 oz" and
# "Moisturizing Cream 453g" are the same product. SPF numbers ARE identity
# (SPF 30 vs SPF 60 are different products), so they survive cleaning.
_UNIT_NOISE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:ml|l|liter|litre|oz|fl ?oz|g|kg|mg|count|ct|pack|pk|pcs|pieces)\b")
_SPF_RE = re.compile(r"\bspf\s*(\d{1,3})\b")
# product "form" words: a cream and a lotion from the same brand line are
# different products — disagreement here must veto a coincide verdict.
# Synonym groups keep marketing variants of the SAME form from colliding
# ("Foaming Facial Cleanser" IS the "foaming face wash" that was mentioned).
_FORM_SYNONYMS = {
    "cream": "cream", "lotion": "lotion", "gel": "gel", "serum": "serum",
    "essence": "serum", "oil": "oil", "balm": "balm", "ointment": "balm",
    "wash": "cleanser", "cleanser": "cleanser", "foam": "cleanser",
    "soap": "cleanser", "spray": "spray", "mist": "spray", "stick": "stick",
    "bar": "bar", "mask": "mask", "conditioner": "conditioner",
    "shampoo": "shampoo", "toner": "toner", "exfoliant": "exfoliant",
    "peel": "exfoliant", "sunscreen": "sunscreen",
}
_FORM_TOKENS = set(_FORM_SYNONYMS)
# tokens too generic to identify a brand on their own ("JBCA Cosmetics" is
# identified by "jbca", never by "cosmetics"; common adjectives that double as
# brand names — Simple, Pure — must not match ordinary prose)
_GENERIC_BRAND_TOKENS = {"cosmetics", "beauty", "skincare", "labs", "lab",
                         "inc", "llc", "co", "company", "brand", "official",
                         "store", "shop", "the", "usa", "paris", "london",
                         "simple", "pure", "natural", "fresh", "true", "real",
                         "basic", "daily", "clean"}


def _clean_product_text(text: str) -> str:
    return _UNIT_NOISE.sub(" ", str(text).lower())


def _char_ngrams(s: str, n: int = 3) -> Dict[str, int]:
    s = re.sub(r"\s+", " ", s.strip())
    grams: Dict[str, int] = {}
    for i in range(max(0, len(s) - n + 1)):
        g = s[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _ngram_cosine(a: str, b: str, n: int = 3) -> float:
    """Character-trigram cosine — robust to typos and spelling variants
    ('moisturising' vs 'moisturizing') where token matching fails."""
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    dot = sum(c * gb.get(g, 0) for g, c in ga.items())
    na = sum(c * c for c in ga.values()) ** 0.5
    nb = sum(c * c for c in gb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _spf_numbers(text: str) -> Set[str]:
    return set(_SPF_RE.findall(str(text).lower()))


def _brand_signal(product: ProductRecord, page_brand: str,
                  mention: RecMention) -> tuple:
    """('agree' | 'conflict' | 'unknown', signal_name). Three ways to agree:
    the shared lexicon, fuzzy string identity, or — ONLY when the mention
    carries no brand concept at all — the product's distinctive brand token
    appearing inside the mention's entity text. That last route is what
    connects long-tail brands no lexicon knows (a 'JBCA' conditioner PDP to a
    turn that recommended 'the JBCA thickening conditioner'); it must never
    override an explicit, conflicting mention brand, and common-adjective
    brand names (Simple, Pure) are excluded from it."""
    pbrand = normalize(product.brand) or normalize(page_brand)
    mbrands = {b for b in (mention.brands | mention.sub_brands) if b}
    if not pbrand:
        return "unknown", ""
    if mbrands:
        if pbrand in mbrands or any(pbrand in b or b in pbrand for b in mbrands):
            return "agree", "brand_lexicon"
        if any(_ngram_cosine(pbrand, b) >= 0.72 for b in mbrands):
            return "agree", "brand_fuzzy"
        return "conflict", "brand_conflict"
    ent_tokens = _tokens(mention.entity_context)
    distinctive = {t for t in _tokens(pbrand)
                   if len(t) >= 3 and t not in _GENERIC_BRAND_TOKENS}
    if distinctive and distinctive & ent_tokens:
        return "agree", "brand_in_entity"
    return "unknown", ""


def _attr_relation(product: ProductRecord, mention: RecMention) -> str:
    """'agree' | 'conflict' | 'neutral' over identity-bearing attributes:
    SPF numbers and product-form words. Conflict vetoes a coincide verdict —
    an SPF 30 is not the SPF 60 that was recommended, a lotion is not the
    cream."""
    p_text = f"{product.name} {product.description}"
    p_spf = _spf_numbers(p_text)
    m_spf = _spf_numbers(mention.entity_context)
    if p_spf and m_spf:
        if p_spf & m_spf:
            return "agree"
        return "conflict"
    p_form = {_FORM_SYNONYMS[t] for t in
              _tokens(_clean_product_text(product.name)) & _FORM_TOKENS}
    m_form = {_FORM_SYNONYMS[t] for t in
              _tokens(_clean_product_text(mention.entity_context)) & _FORM_TOKENS}
    if p_form and m_form:
        return "agree" if p_form & m_form else "conflict"
    return "neutral"


def _score_match(product: ProductRecord, page_brand: str, mention: RecMention,
                 name_threshold: float = 0.34) -> tuple:
    """(score, match_type, match_strength, signals) for one product against
    one mention. Precision-first tiers:

    * **exact**  — SKU identity, or brand agreement + near-identical name;
    * **strong** — brand agreement + real name/attribute corroboration.
      Only exact/strong may set ``coincides``;
    * **likely** — brand alone, or name similarity alone: the same *brand* is
      not the same *product* (the old scorer's 0.75-for-brand-alone made a
      CeraVe cleanser 'coincide' with a CeraVe moisturizer mention);
    * **none**.

    A brand or attribute **conflict** (Cetaphil vs CeraVe; SPF 30 vs 60;
    lotion vs cream) caps the verdict below coincide no matter how similar
    the names read."""
    signals: List[str] = []
    if product.sku and normalize(product.sku) in mention.skus:
        return 1.0, "sku", "exact", ["sku_exact"]

    p_name = _clean_product_text(product.name)
    ent = _clean_product_text(mention.entity_context)
    pt = _tokens(p_name)
    et = _tokens(ent) | {t for b in mention.brands for t in _tokens(b)}
    # containment over the product name's CORE tokens — brand tokens excluded
    # from both sides of the ratio. Counting them let a same-brand sibling
    # ("CeraVe Foaming Facial Cleanser" vs a CeraVe cream mention) ride the
    # brand word into the exact tier on zero product-line agreement.
    brand_toks = (_tokens(product.brand) | _tokens(page_brand)
                  | {t for b in mention.brands for t in _tokens(b)})
    pt_core = pt - brand_toks
    containment = len(pt_core & et) / len(pt_core) if pt_core and et else 0.0
    # entity_context is a sentence; the product name typically leads it, so a
    # length-matched head window keeps the tail from diluting the similarity
    ng = 0.0
    if p_name and ent:
        ng = max(_ngram_cosine(p_name, ent),
                 _ngram_cosine(p_name, ent[:len(p_name)]))
    name_sim = max(containment, ng)

    brand_rel, brand_sig = _brand_signal(product, page_brand, mention)
    attr_rel = _attr_relation(product, mention)
    pcat = _tokens(product.category)
    cat_agree = bool(pcat and pcat & mention.categories_tokens())
    if brand_sig:
        signals.append(brand_sig)
    if attr_rel != "neutral":
        signals.append(f"attr_{attr_rel}")
    if containment >= name_threshold:
        signals.append("name_containment")
    if ng >= 0.45:
        signals.append("name_ngram")
    if cat_agree:
        signals.append("category")
    conflict = attr_rel == "conflict"
    # name corroboration is MANDATORY above the likely tier: brand + a shared
    # generic form word or category ("CeraVe Eye Repair Cream" vs the CeraVe
    # Moisturizing Cream mention — both creams, both moisturizers) is still a
    # different product. Attribute/category agreement corroborates the score,
    # never substitutes for the name. Containment only counts on names with
    # ≥ 2 core tokens — a bare "Moisturizer" hits 1.0 on any moisturizer
    # mention and identifies nothing.
    containment_ok = len(pt_core) >= 2
    name_evidence = containment_ok and (containment >= name_threshold or ng >= 0.45)

    if brand_rel == "agree":
        if not conflict and containment_ok and (containment >= 0.6 or ng >= 0.6):
            return round(min(1.0, 0.85 + 0.15 * name_sim), 4), "brand+name", "exact", signals
        if not conflict and name_evidence:
            bonus = 0.05 if (attr_rel == "agree" or cat_agree) else 0.0
            return round(min(0.84, 0.65 + 0.1 * name_sim + bonus), 4), \
                "brand+name", "strong", signals
        return 0.45, "brand", "likely", signals
    if brand_rel == "unknown":
        # name-only strong needs a name long enough to be identifying: a
        # 1-token "Moisturizer" hits containment 1.0 on any moisturizer mention
        if not conflict and ((len(pt_core) >= 3 and containment >= 0.85)
                             or (len(pt_core) >= 2 and ng >= 0.7)):
            return round(min(0.84, 0.6 + 0.2 * name_sim), 4), "name", "strong", signals
        if len(pt_core) >= 2 and (containment >= 0.5 or ng >= 0.5):
            return round(0.35 + 0.1 * name_sim, 4), "name", "likely", signals
        if cat_agree:
            return 0.3, "category", "likely", signals
        if name_sim > 0:
            return round(0.2 * name_sim, 4), "name", "none", signals
        return 0.0, "none", "none", signals
    # brand conflict: whatever the names look like, this is a different
    # brand's product — never allowed near a coincide verdict
    if name_sim >= 0.5 or cat_agree:
        return round(min(0.3, 0.3 * max(name_sim, 0.5)), 4), "name", "likely", signals
    return 0.0, "none", "none", signals


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


_STRENGTH_ORDER = {"none": 0, "likely": 1, "strong": 2, "exact": 3}


def match_products(products: List[ProductRecord], page_brand: str,
                   mentions: List[RecMention], message_id: str = "",
                   coincide_threshold: float = 0.5,
                   name_threshold: float = 0.34) -> List[ProductRecord]:
    """Annotate each product with its best-matching chat mention (in place).

    ``coincides`` requires an **exact or strong** tier AND the score
    threshold: brand-only and name-only similarity land in the auditable
    ``likely`` tier instead of polluting the coincide precision the funnel
    model depends on."""
    for product in products:
        best_score, best_type, best_strength = 0.0, "none", "none"
        best, best_signals = None, []
        for m in mentions:
            score, mtype, strength, signals = _score_match(
                product, page_brand, m, name_threshold)
            if (_STRENGTH_ORDER[strength], score) > (_STRENGTH_ORDER[best_strength], best_score):
                best_score, best_type, best_strength = score, mtype, strength
                best, best_signals = m, signals
        product.match_score = round(best_score, 4)
        product.match_type = best_type if best else "none"
        product.match_strength = best_strength
        product.match_signals = list(best_signals)
        product.coincides = (best_strength in ("exact", "strong")
                             and best_score >= coincide_threshold)
        product.matched_message_id = message_id
        if best is not None and best_score > 0:
            product.matched_recommendation_id = best.recommendation_id
            product.matched_entity = best.entity_context[:300]
            product.matched_brand = "; ".join(sorted(best.brands))
            product.matched_category = "; ".join(sorted(best.categories))
    return products
