"""Web-page scraping, page classification and on-page product extraction.

The clickstream dataset is full of URLs (`fact_ai_click_through.surfaced_url`,
`dim_digital_site.url`, `simweb_input_file.next_10_urls` / `a_links_source`).
This module turns those URLs into structured evidence:

1. **Fetch** politely (robots.txt, per-domain delay, size/time limits, retry,
   on-disk HTML cache) — `PoliteFetcher`.
2. **Parse** everything extractable from the page — title/meta/OpenGraph,
   JSON-LD blocks, microdata, headings, link/image/price-signal counts,
   text snippet — `parse_page`.
3. **Classify** the page into the funnel-aware taxonomy below —
   `classify_page`.
4. **Extract products** (schema.org Product via JSON-LD / microdata / OG
   fallbacks): name, brand, description, price, currency, availability,
   rating, category, SKU — `parse_page` output `products`.
5. **Match** on-page products against the entities the LLM named in the chat
   (`fact_ai_recommendation.entity_context`, or `brands_in_answer`) and flag
   whether they coincide — `match_products`.
6. **Persist** as parquet with a documented schema — `save_parquet`
   (`scraped_pages.parquet`, `scraped_products.parquet`,
   `product_entity_matches.parquet`).

Page taxonomy (page_class → funnel stage the visit evidences):

    brand_landing     brand-owned home/landing page          → Discovery
    catalogue_page    category/listing, many product cards   → Discovery
    search_results    search engine / site search results    → Discovery
    review_editorial  article, blog, "best X" review content → Evaluation
    forum_social      reddit / youtube / tiktok / instagram  → Evaluation
    product_page      single-product PDP with offer          → Intent
    checkout_or_cart  cart, checkout, order confirmation     → Purchase
    ai_assistant      chatgpt.com etc. (the chat itself)     → (n/a)
    unrelated         none of the above                      → (n/a)

The user's original three buckets map onto this: *catalogue / brand landing*
= {brand_landing, catalogue_page}; *shopping page* = {product_page,
checkout_or_cart} with `ownership` distinguishing brand-owned vs external
retailer/marketplace; *unrelated* = {unrelated} (plus `ai_assistant`,
`search_results`, `review_editorial`, `forum_social` as extra classes that
carry their own funnel signal — see docs/DATA_DICTIONARY.md §4).

Everything below `PoliteFetcher` is pure (html string in → dicts out), so the
classifier and extractors are unit-testable offline.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Taxonomy & domain knowledge
# --------------------------------------------------------------------------- #
PAGE_CLASSES: Dict[str, str] = {
    "brand_landing": "Discovery",
    "catalogue_page": "Discovery",
    "search_results": "Discovery",
    "review_editorial": "Evaluation",
    "forum_social": "Evaluation",
    "product_page": "Intent",
    "checkout_or_cart": "Purchase",
    "ai_assistant": "",
    "unrelated": "",
}

MARKETPLACE_DOMAINS = {
    "amazon.com", "walmart.com", "ebay.com", "etsy.com", "target.com",
    "aliexpress.com", "temu.com", "iherb.com", "shein.com",
}
RETAILER_DOMAINS = {
    "sephora.com", "ulta.com", "cvs.com", "walgreens.com", "boots.com",
    "dermstore.com", "lookfantastic.com", "yesstyle.com", "sokoglam.com",
    "stylevana.com", "oliveyoung.com", "costco.com", "riteaid.com",
}
SOCIAL_DOMAINS = {
    "youtube.com", "tiktok.com", "instagram.com", "facebook.com",
    "reddit.com", "pinterest.com", "x.com", "twitter.com", "quora.com",
}
SEARCH_DOMAINS = {"google.com", "bing.com", "duckduckgo.com", "search.yahoo.com"}
AI_DOMAINS = {"chatgpt.com", "chat.openai.com", "perplexity.ai", "gemini.google.com",
              "claude.ai", "copilot.microsoft.com"}

_PDP_PATH = re.compile(r"/(dp|gp/product|product|products|item|itm|p)/", re.I)
_CATALOG_PATH = re.compile(r"/(category|categories|collections?|shop|c|s\?|browse|catalog)(/|$|\?)", re.I)
_CHECKOUT_PATH = re.compile(r"/(cart|checkout|basket|order|payment)s?(/|$|\?)", re.I)
_SEARCH_PATH = re.compile(r"(/search|[?&](q|k|query|search_term)=)", re.I)
_REVIEW_HINT = re.compile(r"\b(best|top \d+|review|vs\.?|dupe|guide|how to|ranked)\b", re.I)
_PRICE_RE = re.compile(r"(?:[$€£]\s?\d{1,4}(?:[.,]\d{2})?)|(?:\d{1,4}[.,]\d{2}\s?(?:USD|EUR|GBP))")
_CART_RE = re.compile(r"add[ -]?to[ -]?(cart|bag|basket)|buy now|comprar|añadir al carrito", re.I)


def _registrable_domain(netloc: str) -> str:
    parts = netloc.lower().lstrip("www.").split(":")[0].split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc.lower()


# --------------------------------------------------------------------------- #
# 1 · Polite fetching
# --------------------------------------------------------------------------- #
@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    http_status: int = 0
    content_type: str = ""
    html: str = ""
    error: str = ""
    fetched_at: str = ""
    elapsed_ms: int = 0


class PoliteFetcher:
    """requests-based fetcher that honours robots.txt, rate-limits per domain,
    caps download size, retries transient failures, and caches HTML on disk so
    re-runs are free."""

    def __init__(self, user_agent: str = "conveyer-research-bot/0.2 (+github.com/DavidJHub/conveyer)",
                 delay_per_domain: float = 2.0, timeout: float = 15.0,
                 max_bytes: int = 2_500_000, retries: int = 2,
                 cache_dir: Optional[str] = None, respect_robots: bool = True):
        import requests

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.user_agent = user_agent
        self.delay = delay_per_domain
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.retries = retries
        self.respect_robots = respect_robots
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_hit: Dict[str, float] = {}
        self._robots: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}

    # -- helpers -------------------------------------------------------------
    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".html")

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        dom = urlparse(url).netloc
        if dom not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.set_url(f"{urlparse(url).scheme}://{dom}/robots.txt")
                rp.read()
                self._robots[dom] = rp
            except Exception:
                self._robots[dom] = None  # unreachable robots => default allow
        rp = self._robots[dom]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def _throttle(self, url: str) -> None:
        dom = urlparse(url).netloc
        wait = self._last_hit.get(dom, 0) + self.delay - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_hit[dom] = time.time()

    # -- main ----------------------------------------------------------------
    def fetch(self, url: str) -> FetchResult:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cp = self._cache_path(url)
        if cp and cp.exists():
            return FetchResult(url=url, final_url=url, http_status=200,
                               content_type="text/html; cache",
                               html=cp.read_text(errors="replace"),
                               fetched_at=now)
        if not self._allowed(url):
            return FetchResult(url=url, error="blocked_by_robots", fetched_at=now)

        err = ""
        for attempt in range(self.retries + 1):
            self._throttle(url)
            t0 = time.time()
            try:
                r = self.session.get(url, timeout=self.timeout, stream=True)
                ctype = r.headers.get("Content-Type", "")
                if "html" not in ctype and "text" not in ctype:
                    return FetchResult(url=url, final_url=r.url, http_status=r.status_code,
                                       content_type=ctype, error="non_html",
                                       fetched_at=now, elapsed_ms=int((time.time() - t0) * 1000))
                chunks, size = [], 0
                for chunk in r.iter_content(65536, decode_unicode=False):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > self.max_bytes:
                        break
                html = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
                res = FetchResult(url=url, final_url=r.url, http_status=r.status_code,
                                  content_type=ctype, html=html, fetched_at=now,
                                  elapsed_ms=int((time.time() - t0) * 1000))
                if r.status_code == 200 and cp:
                    cp.write_text(html)
                return res
            except Exception as e:  # noqa: BLE001 - record and retry
                err = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
        return FetchResult(url=url, error=err or "failed", fetched_at=now)


# --------------------------------------------------------------------------- #
# 2 · Parsing (pure: html in -> features + products out)
# --------------------------------------------------------------------------- #
def _soup(html: str):
    from bs4 import BeautifulSoup

    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def extract_jsonld(soup) -> List[dict]:
    out = []
    for tag in soup.find_all("script", type=re.compile("ld\\+json", re.I)):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        for obj in stack:
            if isinstance(obj, dict):
                out.append(obj)
                graph = obj.get("@graph")
                if isinstance(graph, list):
                    out.extend(o for o in graph if isinstance(o, dict))
    return out


def _jsonld_type(obj: dict) -> List[str]:
    t = obj.get("@type", [])
    return [t] if isinstance(t, str) else [str(x) for x in t]


def _first(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _product_from_jsonld(obj: dict) -> dict:
    offers = _first(obj.get("offers")) or {}
    if isinstance(offers, dict) and "offers" in offers:      # AggregateOffer
        offers = {**offers, **(_first(offers.get("offers")) or {})}
    rating = obj.get("aggregateRating") or {}
    brand = obj.get("brand")
    if isinstance(brand, dict):
        brand = brand.get("name")

    def _f(v):
        try:
            return float(str(v).replace(",", "").replace("$", ""))
        except (TypeError, ValueError):
            return np.nan

    return {
        "source": "jsonld",
        "name": _first(obj.get("name")),
        "brand": brand,
        "description": (obj.get("description") or "")[:600] or None,
        "price": _f(offers.get("price") or offers.get("lowPrice")),
        "price_currency": offers.get("priceCurrency"),
        "availability": str(offers.get("availability") or "").split("/")[-1] or None,
        "rating_value": _f(rating.get("ratingValue")),
        "rating_count": _f(rating.get("ratingCount") or rating.get("reviewCount")),
        "category": obj.get("category") if isinstance(obj.get("category"), str) else None,
        "sku": obj.get("sku") or obj.get("gtin13") or obj.get("gtin"),
        "image_url": _first(obj.get("image")) if isinstance(_first(obj.get("image")), str) else None,
    }


def _products_from_microdata(soup) -> List[dict]:
    out = []
    for scope in soup.find_all(attrs={"itemtype": re.compile("schema.org/Product", re.I)}):
        def _prop(name):
            el = scope.find(attrs={"itemprop": name})
            if el is None:
                return None
            return el.get("content") or el.get_text(" ", strip=True) or None

        price = _prop("price")
        try:
            price = float(str(price).replace(",", "").replace("$", "")) if price else np.nan
        except ValueError:
            price = np.nan
        out.append({"source": "microdata", "name": _prop("name"), "brand": _prop("brand"),
                    "description": (_prop("description") or "")[:600] or None,
                    "price": price, "price_currency": _prop("priceCurrency"),
                    "availability": _prop("availability"), "rating_value": np.nan,
                    "rating_count": np.nan, "category": _prop("category"),
                    "sku": _prop("sku"), "image_url": None})
    return out


def parse_page(url: str, html: str) -> Tuple[dict, List[dict]]:
    """Everything we can pull from one page: feature dict + product list."""
    soup = _soup(html)
    for tag in soup(["script", "style", "noscript"]):
        if not (tag.name == "script" and "ld+json" in (tag.get("type") or "")):
            tag.extract()
    jsonld = extract_jsonld(_soup(html))          # from untouched copy
    types = sorted({t for o in jsonld for t in _jsonld_type(o)})

    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    meta = {m.get("property") or m.get("name"): m.get("content", "")
            for m in _soup(html).find_all("meta") if m.get("content")}
    links = [a.get("href") or "" for a in _soup(html).find_all("a")]
    dom = _registrable_domain(urlparse(url).netloc)

    products = [_product_from_jsonld(o) for o in jsonld
                if "Product" in _jsonld_type(o)]
    if not products:
        products = _products_from_microdata(_soup(html))
    if not products and (meta.get("og:type") == "product" or "product:price:amount" in meta):
        try:
            price = float(meta.get("product:price:amount", "nan"))
        except ValueError:
            price = np.nan
        products = [{"source": "og_meta", "name": meta.get("og:title"),
                     "brand": meta.get("product:brand"), "description": meta.get("og:description"),
                     "price": price, "price_currency": meta.get("product:price:currency"),
                     "availability": meta.get("product:availability"), "rating_value": np.nan,
                     "rating_count": np.nan, "category": meta.get("product:category"),
                     "sku": None, "image_url": meta.get("og:image")}]
    products = [p for p in products if p.get("name")]

    h1 = _soup(html).find("h1")
    features = {
        "url": url,
        "domain": dom,
        "title": (_soup(html).title.get_text(strip=True) if _soup(html).title else "") or meta.get("og:title", ""),
        "meta_description": meta.get("description", "") or meta.get("og:description", ""),
        "og_type": meta.get("og:type", ""),
        "og_site_name": meta.get("og:site_name", ""),
        "canonical_url": next((l.get("href", "") for l in _soup(html).find_all("link")
                               if l.get("rel") == ["canonical"]), ""),
        "language": (_soup(html).html.get("lang") or "") if _soup(html).html else "",
        "h1": h1.get_text(" ", strip=True) if h1 else "",
        "n_links": len(links),
        "n_images": len(_soup(html).find_all("img")),
        "n_product_links": sum(bool(_PDP_PATH.search(l)) for l in links),
        "n_price_signals": len(_PRICE_RE.findall(text[:20000])),
        "has_add_to_cart": bool(_CART_RE.search(html[:200000])),
        "jsonld_types": ";".join(types),
        "n_jsonld_products": len([o for o in jsonld if "Product" in _jsonld_type(o)]),
        "text_chars": len(text),
        "text_snippet": text[:500],
    }
    return features, products


# --------------------------------------------------------------------------- #
# 3 · Classification
# --------------------------------------------------------------------------- #
def classify_page(feat: dict, brand_lexicon: Sequence[str] = ()) -> dict:
    """Rule-based, evidence-logged page classifier over `parse_page` features.

    Returns page_class, funnel_stage, ownership, confidence and the evidence
    trail. Swap-in point for an LLM/text classifier later — the interface
    (feature dict in, labelled dict out) stays the same.
    """
    dom, url = feat.get("domain", ""), feat.get("url", "")
    title = f"{feat.get('title','')} {feat.get('h1','')}"
    ev: List[str] = []

    def _done(cls, conf, ownership="n/a"):
        return {"page_class": cls, "funnel_stage": PAGE_CLASSES[cls],
                "ownership": ownership, "class_confidence": round(conf, 2),
                "class_evidence": ";".join(ev)}

    brand_hit = next((b for b in brand_lexicon
                      if re.sub(r"[^a-z0-9]", "", b.lower()) in dom.replace("-", "").replace(".", "")), "")
    ownership = ("marketplace" if dom in MARKETPLACE_DOMAINS
                 else "retailer" if dom in RETAILER_DOMAINS
                 else "brand_owned" if brand_hit else "unknown")
    if brand_hit:
        ev.append(f"brand-domain:{brand_hit}")

    if dom in AI_DOMAINS:
        ev.append("ai-domain")
        return _done("ai_assistant", 0.95)
    if dom in SOCIAL_DOMAINS:
        ev.append("social-domain")
        return _done("forum_social", 0.9)
    if dom in SEARCH_DOMAINS or _SEARCH_PATH.search(url):
        ev.append("search-url")
        return _done("search_results", 0.85)
    if _CHECKOUT_PATH.search(url) or re.search(r"\b(checkout|shopping cart)\b", title, re.I):
        ev.append("checkout-url/title")
        return _done("checkout_or_cart", 0.85, ownership)

    is_pdp = (feat.get("n_jsonld_products", 0) == 1
              or (feat.get("has_add_to_cart") and feat.get("n_jsonld_products", 0) <= 1
                  and bool(_PDP_PATH.search(url))))
    if is_pdp:
        ev.append("jsonld-product" if feat.get("n_jsonld_products") else "cart-button+pdp-url")
        return _done("product_page", 0.9 if feat.get("n_jsonld_products") else 0.7,
                     ownership if ownership != "unknown" else "retailer")

    many_products = (feat.get("n_jsonld_products", 0) > 1
                     or "ItemList" in feat.get("jsonld_types", "")
                     or "CollectionPage" in feat.get("jsonld_types", "")
                     or feat.get("n_product_links", 0) >= 8
                     or (_CATALOG_PATH.search(url) and feat.get("n_price_signals", 0) >= 4))
    if many_products:
        ev.append("itemlist/product-grid")
        return _done("catalogue_page", 0.8, ownership)

    if "Article" in feat.get("jsonld_types", "") or "NewsArticle" in feat.get("jsonld_types", "") \
            or (_REVIEW_HINT.search(title) and feat.get("text_chars", 0) > 2500):
        ev.append("article/review-title")
        return _done("review_editorial", 0.75)

    is_root = urlparse(url).path.strip("/") == ""
    if ownership == "brand_owned" and (is_root or feat.get("og_type") == "website"
                                       or "Organization" in feat.get("jsonld_types", "")
                                       or "WebSite" in feat.get("jsonld_types", "")):
        ev.append("brand-domain-root")
        return _done("brand_landing", 0.8, "brand_owned")
    if is_root and (feat.get("n_product_links", 0) >= 3 or feat.get("has_add_to_cart")):
        ev.append("shop-root")
        return _done("brand_landing", 0.6, ownership)

    ev.append("no-shopping-signals")
    return _done("unrelated", 0.5)


# --------------------------------------------------------------------------- #
# 4 · Product ↔ chat-entity matching
# --------------------------------------------------------------------------- #
def _norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).strip()


def _similarity(a: str, b: str) -> float:
    """Blend of char-level ratio and token overlap — robust to word order and
    packaging suffixes ('16 oz', 'fragrance free')."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    char = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    tok = len(ta & tb) / max(min(len(ta), len(tb)), 1)
    return 0.5 * char + 0.5 * tok


def match_products(products: pd.DataFrame, entities: pd.DataFrame,
                   entity_text_col: str = "entity_context",
                   entity_brand_col: str = "brand",
                   name_threshold: float = 0.55) -> pd.DataFrame:
    """Match every scraped product against every chat entity (per page or per
    turn if `message_id` is present in both), score similarity, and flag
    `coincides` when name similarity clears the threshold or brand matches
    with moderate name support.
    """
    rows = []
    join_on_turn = "message_id" in products.columns and "message_id" in entities.columns
    for _, p in products.iterrows():
        cand = entities
        if join_on_turn:
            cand = entities[entities["message_id"] == p["message_id"]]
        best, best_score, brand_ok = None, 0.0, False
        for _, e in cand.iterrows():
            s = _similarity(p.get("name"), e.get(entity_text_col))
            b = (_norm(p.get("brand")) != "" and entity_brand_col in e
                 and _norm(p.get("brand")) == _norm(e.get(entity_brand_col)))
            score = s + (0.15 if b else 0.0)
            if score > best_score:
                best, best_score, brand_ok = e, s, b
        if best is None:
            continue
        rows.append({
            "page_id": p.get("page_id"), "message_id": p.get("message_id"),
            "product_name": p.get("name"), "product_brand": p.get("brand"),
            "matched_entity": best.get(entity_text_col),
            "matched_entity_brand": best.get(entity_brand_col),
            "name_similarity": round(best_score, 3),
            "brand_match": bool(brand_ok),
            "coincides": bool(best_score >= name_threshold
                              or (brand_ok and best_score >= 0.35)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5 · Pipeline & parquet persistence
# --------------------------------------------------------------------------- #
PAGES_SCHEMA = [
    "page_id", "url", "final_url", "domain", "fetched_at", "http_status",
    "fetch_error", "content_type", "page_class", "funnel_stage", "ownership",
    "class_confidence", "class_evidence", "title", "meta_description",
    "og_type", "og_site_name", "canonical_url", "language", "h1", "n_links",
    "n_images", "n_product_links", "n_price_signals", "has_add_to_cart",
    "jsonld_types", "n_jsonld_products", "text_chars", "text_snippet",
]
PRODUCTS_SCHEMA = [
    "page_id", "url", "message_id", "source", "name", "brand", "description",
    "price", "price_currency", "availability", "rating_value", "rating_count",
    "category", "sku", "image_url",
]


def page_id_of(url: str) -> str:
    return hashlib.sha1(str(url).encode()).hexdigest()[:16]


def scrape_pages(urls: Iterable[str], fetcher: Optional[PoliteFetcher] = None,
                 brand_lexicon: Sequence[str] = (), max_pages: Optional[int] = None,
                 url_to_message: Optional[Dict[str, str]] = None,
                 progress: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch + parse + classify a URL list. Returns (pages_df, products_df).

    `url_to_message` optionally maps url -> message_id so scraped products can
    be matched to that turn's chat entities downstream.
    """
    fetcher = fetcher or PoliteFetcher()
    urls = list(dict.fromkeys(u for u in urls if isinstance(u, str) and u.startswith("http")))
    if max_pages:
        urls = urls[:max_pages]
    pages, products = [], []
    for i, url in enumerate(urls):
        res = fetcher.fetch(url)
        pid = page_id_of(url)
        if res.html:
            feat, prods = parse_page(url, res.html)
            feat.update(classify_page(feat, brand_lexicon))
        else:
            feat = {"url": url, "domain": _registrable_domain(urlparse(url).netloc),
                    "page_class": "unfetched", "funnel_stage": "", "ownership": "n/a",
                    "class_confidence": 0.0, "class_evidence": res.error}
            prods = []
        feat.update({"page_id": pid, "final_url": res.final_url,
                     "http_status": res.http_status, "fetch_error": res.error,
                     "content_type": res.content_type, "fetched_at": res.fetched_at})
        pages.append(feat)
        mid = (url_to_message or {}).get(url, "")
        for p in prods:
            products.append({**p, "page_id": pid, "url": url, "message_id": mid})
        if progress and (i + 1) % 25 == 0:
            print(f"  scraped {i + 1}/{len(urls)}")
    pages_df = pd.DataFrame(pages).reindex(columns=PAGES_SCHEMA)
    products_df = pd.DataFrame(products).reindex(columns=PRODUCTS_SCHEMA)
    return pages_df, products_df


def save_parquet(pages_df: pd.DataFrame, products_df: pd.DataFrame,
                 out_dir: str = "outputs/scraped",
                 matches_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"pages": str(out / "scraped_pages.parquet"),
             "products": str(out / "scraped_products.parquet")}
    pages_df.to_parquet(paths["pages"], index=False)
    products_df.to_parquet(paths["products"], index=False)
    if matches_df is not None:
        paths["matches"] = str(out / "product_entity_matches.parquet")
        matches_df.to_parquet(paths["matches"], index=False)
    return paths


# --------------------------------------------------------------------------- #
# CLI:  python -m conveyer.pages --urls <file.parquet|csv> --url-col url ...
# --------------------------------------------------------------------------- #
def _main(argv=None):
    ap = argparse.ArgumentParser(description="Scrape, classify and extract products from URLs")
    ap.add_argument("--urls", required=True, help="parquet/csv containing the URL column")
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--message-col", default=None, help="optional message_id column for matching")
    ap.add_argument("--out", default="outputs/scraped")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--cache", default="outputs/scraped/html_cache")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.urls) if args.urls.endswith(".parquet") else pd.read_csv(args.urls)
    mapping = (dict(zip(df[args.url_col], df[args.message_col].astype(str)))
               if args.message_col else None)
    fetcher = PoliteFetcher(delay_per_domain=args.delay, cache_dir=args.cache)
    pages, products = scrape_pages(df[args.url_col].dropna().tolist(), fetcher,
                                   max_pages=args.max_pages, url_to_message=mapping)
    paths = save_parquet(pages, products, args.out)
    print(f"pages: {len(pages)}  products: {len(products)}")
    print("class counts:\n", pages["page_class"].value_counts().to_string())
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    _main()
