"""Page classifier.

Given a parsed :class:`~conveyer.scraping.extract.PageContent` (plus the URL and,
when available, the SimilarWeb ``dim_digital_site`` prior), decide the headline
``page_category`` the user asked for — *catalogue / brand landing* (Discovery),
*shopping* (retailer or brand-owned), *unrelated* — and the four proposed extra
categories (*editorial, search, community, reference*).

The classifier is **multimodal**: four independent evidence channels vote and
any one of them can carry a page on its own —

1. **URL structure** — path/query tokens (``/cart/``, ``checkout``,
   ``/dp/…``, ``/search?q=``, ``ref_=nav_cart``, …);
2. **domain knowledge** — the registrable domain against curated retailer /
   marketplace / search / community / editorial / reference / brand lists
   (``amazon.com`` *is* a retailer, fetched or not);
3. **page content & markup** — schema.org ``@type``, OpenGraph, price /
   add-to-cart / product-count signals (only when the page was fetched);
4. **vendor prior** — ``dim_digital_site.page_type`` when available.

Votes are summed, the argmax wins with a softmax confidence, and
``PageClass.signals`` records which modalities actually fired so every label is
auditable. An optional LLM pass (``classifier="llm"`` / ``"auto"`` when
``ANTHROPIC_API_KEY`` is set) refines low-confidence pages.

Topical relevance is a **separate axis** from page structure, with collapse
rules that respect what each modality can know:

* **topic-neutral subtypes** (cart, checkout, order, search results,
  marketplace and storefront homepages) carry no topical tokens *by nature* —
  journey infrastructure, never collapsed for lacking skincare evidence. But
  neutrality must be *earned* by a structural vote (URL/markup/prior or a
  decisive domain role): the weak retailer catch-all does not turn
  ``sephora.com/careers`` into a journey page, and a SERP whose URL exposes
  its query is judged by the query text instead;
* **transactional URL tokens are self-evident commerce** on any domain — an
  unfetchable ``/checkouts/c/<token>`` on an unheard-of store is a checkout;
* **topical subtypes** (PDP, article, …) *with* fetched content and no skincare
  signal collapse to ``unrelated`` (a confident judgement);
* topical subtypes *without* content keep their *earned* structural category
  when the domain is a known one (an unfetched ``amazon.com/dp/…`` is still a
  shopping page — for *which* product is what ``is_study_relevant`` tracks),
  and fall to ``unknown`` when nothing earns the role.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from ..ingest import normalize
from .config import ScrapeConfig
from .extract import PageContent
from .taxonomy import (
    SIMILARWEB_PAGE_TYPE_TO_SUBTYPE, category_for_subtype, funnel_stage_for,
    is_commerce,
)

# --------------------------------------------------------------------------- #
# Curated domain lists (skincare / beauty commerce, US, Jan 2026)
# --------------------------------------------------------------------------- #
MARKETPLACE_DOMAINS = {
    "amazon", "walmart", "ebay", "etsy", "aliexpress", "target", "costco",
    "samsclub", "wish", "temu",
}
RETAILER_DOMAINS = MARKETPLACE_DOMAINS | {
    "sephora", "ulta", "cvs", "walgreens", "riteaid", "kohls", "macys",
    "nordstrom", "dermstore", "lookfantastic", "iherb", "beautylish",
    "cultbeauty", "spacenk", "bluemercury", "skinstore", "yesstyle",
    "stylevana", "soko-glam", "sokoglam", "boots", "superdrug", "chemistwarehouse",
    "well", "thirteenlune", "credobeauty", "shoppersdrugmart",
}
SEARCH_DOMAINS = {
    "google", "bing", "duckduckgo", "yahoo", "ecosia", "brave", "yandex",
    "baidu", "startpage", "kagi", "perplexity",
}
COMMUNITY_DOMAINS = {
    "reddit", "youtube", "tiktok", "instagram", "facebook", "pinterest",
    "quora", "twitter", "x", "threads", "makeupalley", "acne", "beautypedia",
    "trustpilot", "influenster",
}
EDITORIAL_DOMAINS = {
    "byrdie", "allure", "wirecutter", "nytimes", "cosmopolitan", "elle",
    "vogue", "harpersbazaar", "refinery29", "self", "glamour", "thecut",
    "goodhousekeeping", "prevention", "womenshealthmag", "instyle",
    "marieclaire", "realsimple", "buzzfeed", "businessinsider", "forbes",
    "rd", "today", "cnn", "nbcnews", "usatoday", "popsugar", "stylecraze",
    "who-what-wear", "whowhatwear", "dermstore-blog", "skincare",
}
REFERENCE_DOMAINS = {
    "wikipedia", "webmd", "mayoclinic", "healthline", "medicalnewstoday",
    "aad", "nih", "ncbi", "nlm", "cdc", "everydayhealth", "verywellhealth",
    "clevelandclinic", "hopkinsmedicine", "medlineplus", "fda", "paulaschoice-eu",
}
# Known brand-owned skincare storefronts (not exhaustive — the domain-vs-brand
# heuristic covers the long tail).
BRAND_DOMAINS = {
    "cerave", "theordinary", "deciem", "laroche-posay", "larocheposay",
    "olay", "neutrogena", "cetaphil", "paulaschoice", "drunkelephant",
    "tatcha", "kiehls", "clinique", "esteelauder", "glossier", "theinkeylist",
    "skinceuticals", "murad", "eltamd", "isdin", "bioderma", "avene", "vichy",
    "goodmolecules", "firstaidbeauty", "youthtothepeople", "summerfridays",
    "cocokind", "versedskin", "krave", "beautyofjoseon", "cosrx", "innisfree",
    "laneige", "farmacybeauty", "supergoop", "biossance", "eltamdskincare",
}

# Multi-label TLDs so registrable-domain extraction handles co.uk etc.
_MULTI_TLD = {
    "co.uk", "com.au", "co.jp", "com.br", "co.nz", "co.in", "com.mx",
    "co.kr", "com.tr", "co.za", "com.sg", "com.hk", "co.il",
}

_SKINCARE_KEYWORDS = re.compile(
    r"\b(skin ?care|skincare|serum|moisturi[sz]er|cleanser|sunscreen|spf|retinol|"
    r"niacinamide|hyaluronic|salicylic|glycolic|vitamin c|ceramide|toner|"
    r"exfoliant|acne|blackheads|wrinkles|dark spots|hyperpigmentation|dermatolog|"
    r"pore|breakout|rosacea|eczema|face ?cream|eye ?cream|peptide|azelaic)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #
@dataclass
class UrlParts:
    scheme: str = ""
    host: str = ""
    subdomain: str = ""
    registrable: str = ""   # e.g. amazon.co.uk
    core: str = ""          # e.g. amazon
    path: str = ""
    query: str = ""


def parse_url(url: str) -> UrlParts:
    try:
        s = urlsplit(url if "://" in url else "http://" + url)
    except ValueError:
        return UrlParts()
    host = (s.hostname or "").lower().lstrip(".")
    labels = host.split(".") if host else []
    registrable, subdomain = host, ""
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_TLD:
        registrable = ".".join(labels[-3:])
        subdomain = ".".join(labels[:-3])
    elif len(labels) >= 2:
        registrable = ".".join(labels[-2:])
        subdomain = ".".join(labels[:-2])
    core = registrable.split(".")[0] if registrable else ""
    if subdomain == "www":
        subdomain = ""
    return UrlParts(scheme=s.scheme, host=host, subdomain=subdomain,
                    registrable=registrable, core=core,
                    path=(s.path or "/").lower(), query=(s.query or "").lower())


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class PageClass:
    page_category: str = "unknown"     # final headline (collapses to 'unrelated' when off-topic)
    page_subtype: str = "other"        # structural subtype (pdp/collection/article/serp/...)
    seller_type: str = "na"            # brand_owned | retailer | na
    funnel_stage: str = "Irrelevant"
    confidence: float = 0.0
    method: str = "rule"               # rule | rule+prior | llm
    skincare_relevance: float = 0.0
    is_study_relevant: bool = False
    primary_brand: str = ""
    brand_detected: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)  # modalities that fired: url/domain/markup/prior/content


# Subtypes that carry no topical tokens by nature — journey infrastructure.
# Lack of skincare evidence must never collapse these (an Amazon cart URL says
# nothing about skincare and never will). Neutrality must be *earned* by a
# structural vote (URL/markup/prior, or a decisive domain role) — the weak
# retailer catch-all votes must not turn /careers or /help into journey pages.
TOPIC_NEUTRAL_SUBTYPES = {"cart", "checkout", "order", "serp", "site_search",
                          "marketplace", "homepage"}

# URL-token transactional subtypes are commerce infrastructure wherever they
# live: /checkouts/c/<token> on an unheard-of Shopify store is still a checkout.
TRANSACTIONAL_SUBTYPES = {"cart", "checkout", "order", "wishlist"}


def _known_domain(u: UrlParts) -> bool:
    """Domain modality has an opinion: the registrable domain is in one of the
    curated lists (or is .gov/.edu), so its role is known without fetching."""
    tld = u.registrable.rsplit(".", 1)[-1] if u.registrable else ""
    return (u.core in RETAILER_DOMAINS or u.core in BRAND_DOMAINS
            or u.core in SEARCH_DOMAINS or u.core in COMMUNITY_DOMAINS
            or u.core in EDITORIAL_DOMAINS or u.core in REFERENCE_DOMAINS
            or tld in ("gov", "edu"))


# --------------------------------------------------------------------------- #
# Rule scorer
# --------------------------------------------------------------------------- #
# Root-ish paths, shared by the homepage rule and the retailer-root boost so a
# locale root (/us, /en-us) never reads differently from "/".
_ROOT_PATHS = ("", "/", "/index.html", "/home", "/us", "/en", "/en-us")


def _url_subtype_votes(u: UrlParts) -> Dict[str, float]:
    """Vote for structural subtypes from the URL path/query."""
    v: Dict[str, float] = {}
    path, q = u.path, u.query

    def add(sub, w):
        v[sub] = v.get(sub, 0.0) + w

    if path in _ROOT_PATHS:
        add("homepage", 2.0)
    if re.search(r"/(dp|gp/product|ip|product|products|p|prod|pd|itm)(/|$|-)", path) or \
       re.search(r"[?&](sku|productid|pid|asin)=", "?" + q):
        add("pdp", 2.0)
    if re.search(r"/(collections?|categor(y|ies)|shop|browse|store|c|b|departments?)(/|$)", path):
        add("collection", 1.6)
    # cart/checkout/order: transactional tokens are decisive wherever they
    # appear — /gp/cart/view.html, /checkouts/c/<token>, basket.jsp, /gp/aw/c
    # (amazon mobile cart), ?ref_=nav_cart — and must outvote the single-letter
    # /c/ collection token (1.6).
    if re.search(r"/(cart|basket|(shopping-)?bag)(/|$|\.)", path) \
            or "/gp/cart" in path or "/gp/aw/c" in path:
        add("cart", 2.5)
    if re.search(r"/(checkouts?|payments?|buy)(/|$|\.)", path) or "/gp/buy" in path:
        add("checkout", 2.5)
    elif "checkout" in path:
        add("checkout", 1.8)
    if re.search(r"/(orders?|order-history|purchase-history)(/|$|\.)", path):
        add("order", 2.2)
    if re.search(r"/wishlist(/|$|\.)", path) or "/hz/wishlist" in path:
        add("wishlist", 2.2)
    if re.search(r"(^|[?&_=-])cart\b", q):
        add("cart", 1.0)          # e.g. ref_=nav_cart
    if "checkout" in q:
        add("checkout", 0.8)
    if re.search(r"/search", path) or re.search(r"[?&](q|k|query|search|keyword)=", "?" + q):
        add("serp", 2.0)
    if re.search(r"/(blog|article|news|reviews?|guide|guides|best|stories|tips)(/|$)", path) or \
       re.search(r"best-.*-for|-review(s)?(-|/|$)|top-\d+", path):
        add("article", 1.8)
    if re.search(r"/(wiki|health|conditions?|how-to|howto|learn|ingredient)(/|$)", path):
        add("wiki", 1.2)
    return v


def _domain_votes(u: UrlParts, page: PageContent, chat_brands: set) -> Dict[str, float]:
    v: Dict[str, float] = {}
    core, host = u.core, u.host

    def add(sub, w):
        v[sub] = v.get(sub, 0.0) + w

    tld = u.registrable.rsplit(".", 1)[-1] if u.registrable else ""
    if core in SEARCH_DOMAINS:
        add("serp", 3.0)
    if core in COMMUNITY_DOMAINS:
        add("forum", 3.0)
    if core in EDITORIAL_DOMAINS:
        add("article", 2.5)
    if core in REFERENCE_DOMAINS or tld in ("gov", "edu") or host.endswith(".gov") or host.endswith(".edu"):
        add("wiki", 2.5)
    if core in MARKETPLACE_DOMAINS:
        add("marketplace", 1.2)
    if core in RETAILER_DOMAINS:
        add("listing", 0.6)     # retailer, exact role decided by path/markup
        # a retailer's root page is its storefront entry (browse-many), not a
        # brand landing page — outvote the generic "homepage" reading
        if u.path in _ROOT_PATHS:
            add("marketplace", 2.6)
    # brand-owned: known brand domain, or the domain core matches a brand the
    # chat recommended / the page's own product brand
    page_brand_tokens = {normalize(page.og.get("product:brand", ""))} | {
        normalize(b) for b in chat_brands}
    if core in BRAND_DOMAINS or (core and any(core in b or b in core
                                              for b in page_brand_tokens if len(b) > 2)):
        add("brand_site", 1.5)
    return v


def _markup_votes(page: PageContent, n_products: int) -> Dict[str, float]:
    v: Dict[str, float] = {}
    types = set(page.schema_types())
    og_type = page.og.get("og:type", "").lower()

    def add(sub, w):
        v[sub] = v.get(sub, 0.0) + w

    if "product" in types or og_type == "product" or page.has_price:
        add("pdp", 1.5)
    if {"itemlist", "offercatalog", "collectionpage"} & types:
        add("collection", 1.8)
    if {"article", "blogposting", "newsarticle", "review", "webpage"} & types and og_type == "article":
        add("article", 1.5)
    elif {"article", "blogposting", "newsarticle", "review"} & types:
        add("article", 1.2)
    if "searchresultspage" in types:
        add("serp", 2.0)
    if {"qapage", "discussionforumposting", "socialmediaposting"} & types:
        add("forum", 2.0)
    if {"faqpage", "medicalwebpage", "howto"} & types:
        add("wiki", 1.2)
    if {"organization", "website"} & types and not (page.has_price or n_products):
        add("homepage", 0.8)
    if page.has_add_to_cart:
        add("pdp", 1.0)
    if n_products >= 3:
        add("collection", 1.2)
    elif n_products == 1:
        add("pdp", 1.0)
    return v


def _skincare_relevance(page: PageContent, u: UrlParts, chat_brands: set,
                        prior_category: str = "") -> float:
    # URL slugs are evidence too — critical when the body couldn't be fetched:
    # /product/cerave-moisturizing-cream is on-topic even with zero HTML.
    url_text = re.sub(r"[-_+/=&?.]", " ", f"{u.path} {u.query}")
    text = " ".join([page.title, page.meta_description, " ".join(page.h1),
                     page.text[:3000], url_text])
    hits = len(_SKINCARE_KEYWORDS.findall(text))
    score = min(1.0, hits / 4.0)
    # Only a skincare brand's own storefront is topical by domain alone; a
    # retailer (amazon, target) sells everything, so its domain must not confer
    # relevance — otherwise off-topic products on it look study-relevant.
    if u.core in BRAND_DOMAINS:
        score = max(score, 0.5)
    blob = normalize(text)
    if any(len(b) > 3 and b in blob for b in BRAND_DOMAINS):
        score = max(score, 0.4)
    if chat_brands:
        bt = {normalize(b) for b in chat_brands}
        if any(b and b in blob for b in bt):
            score = max(score, 0.6)
    if prior_category and prior_category.lower() in (
            "beauty_and_cosmetics", "health", "e-commerce_and_shopping"):
        score = max(score, 0.3)
    return round(score, 3)


def _softmax_conf(scores: Dict[str, float]) -> float:
    import math
    vals = [s for s in scores.values() if s > 0]
    if not vals:
        return 0.0
    top = max(vals)
    exps = [math.exp(s - top) for s in vals]
    return round(math.exp(top - top) / sum(exps), 3) if len(vals) > 1 else 1.0


def classify_rule(page: PageContent, url: str, cfg: ScrapeConfig,
                  prior: Optional[dict] = None, n_products: int = 0,
                  chat_brands: Optional[set] = None) -> PageClass:
    chat_brands = chat_brands or set()
    u = parse_url(url or page.url)
    prior = prior or {}

    # each modality votes independently; any one can carry the page alone
    modality_votes = {
        "url": _url_subtype_votes(u),
        "domain": _domain_votes(u, page, chat_brands),
        "markup": _markup_votes(page, n_products),
    }
    votes: Dict[str, float] = {}
    for src in modality_votes.values():
        for k, w in src.items():
            votes[k] = votes.get(k, 0.0) + w

    # SimilarWeb prior: nudge the mapped subtype
    prior_page_type = str(prior.get(cfg.col_page_type, "") or "").lower()
    prior_fired = cfg.use_similarweb_prior and prior_page_type in SIMILARWEB_PAGE_TYPE_TO_SUBTYPE
    if prior_fired:
        sub = SIMILARWEB_PAGE_TYPE_TO_SUBTYPE[prior_page_type]
        votes[sub] = votes.get(sub, 0.0) + 3.0 * cfg.prior_weight

    has_content = bool(page.title or page.text or page.og)
    signals = [name for name, v in modality_votes.items() if v]
    if prior_fired:
        signals.append("prior")
    if has_content and "markup" not in signals:
        signals.append("content")

    if not votes:
        return PageClass(page_category="unknown", page_subtype="other",
                         confidence=0.0, method="rule", signals=signals,
                         skincare_relevance=_skincare_relevance(page, u, chat_brands,
                                                                str(prior.get(cfg.col_site_category, ""))))

    subtype = max(votes, key=votes.get)
    category = category_for_subtype(subtype)
    # collapse subtype votes into category-level scores for confidence
    cat_scores: Dict[str, float] = {}
    for sub, w in votes.items():
        cat_scores[category_for_subtype(sub)] = cat_scores.get(category_for_subtype(sub), 0.0) + w
    confidence = _softmax_conf(cat_scores)

    relevance = _skincare_relevance(page, u, chat_brands,
                                    str(prior.get(cfg.col_site_category, "")))
    known = _known_domain(u)
    # the winning subtype must be *earned* by structural evidence: a URL/markup
    # vote, a matching vendor prior, or a decisive domain role (>= 2.0 — search
    # engine, community, editorial, reference, storefront root). The retailer
    # catch-alls (listing 0.6, bare marketplace 1.2) earn nothing on their own.
    prior_sub = SIMILARWEB_PAGE_TYPE_TO_SUBTYPE.get(prior_page_type) if prior_fired else None
    earned = (subtype in modality_votes["url"] or subtype in modality_votes["markup"]
              or subtype == prior_sub or modality_votes["domain"].get(subtype, 0.0) >= 2.0)
    # a SERP whose URL exposes the query is NOT topic-neutral — the query text
    # itself decides relevance (q=best+retinol vs q=gaming+laptops)
    serp_with_query = subtype in ("serp", "site_search") and \
        re.search(r"[?&](q|k|query|search|keyword)=[^&]+", "?" + u.query)
    neutral = subtype in TOPIC_NEUTRAL_SUBTYPES and earned and not serp_with_query
    # transactional URL tokens are self-evident commerce, known domain or not
    self_evident = subtype in TRANSACTIONAL_SUBTYPES and subtype in modality_votes["url"]
    # journey infrastructure (a cart on a known retailer, a storefront entry)
    # is relevant to the journey by virtue of being *in* it
    is_relevant = relevance >= 0.15 or (neutral and known) or self_evident

    # seller type (only meaningful for commerce categories)
    seller = "na"
    if is_commerce(category) or subtype in ("brand_site", "marketplace", "listing", "pdp", "collection"):
        if subtype == "brand_site" or u.core in BRAND_DOMAINS:
            seller = "brand_owned"
        elif u.core in RETAILER_DOMAINS or prior.get(cfg.col_retailer_brand):
            seller = "retailer"
        elif str(prior.get(cfg.col_seller_type, "")).lower() in ("1p", "3p"):
            seller = "retailer"

    # topical relevance overrides the headline bucket (user's "unrelated" case).
    # "unrelated" is a *confident* judgement — it needs fetched content to
    # stand on. Without content, an *earned* structural role stands in: an
    # unfetched amazon.com/dp/… is still a shopping page, an unfetched
    # /checkouts/c/<token> is still a checkout. A page whose only support is
    # the retailer catch-all (sephora.com/careers) tells us nothing → unknown.
    final_category = category
    if not is_relevant and category != "unknown":
        if has_content:
            final_category = "unrelated"
        elif not (earned and (known or self_evident)):
            final_category = "unknown"

    # page-intrinsic brand (the brand this page is *about*) — used for matching.
    # Deliberately NOT the chat brand: letting a brand-less product inherit the
    # chat brand would make everything trivially "coincide".
    page_brand = ""
    if u.core in BRAND_DOMAINS:
        page_brand = u.core
    elif page.og.get("product:brand"):
        page_brand = normalize(page.og["product:brand"])
    primary_brand = page_brand
    brands = sorted({b for b in chat_brands} | ({page_brand} if page_brand else set()))
    brands = [b for b in brands if b]

    method = "rule+prior" if prior_fired else "rule"
    return PageClass(
        page_category=final_category, page_subtype=subtype, seller_type=seller,
        funnel_stage=funnel_stage_for(final_category, subtype),
        confidence=confidence, method=method,
        skincare_relevance=relevance, is_study_relevant=is_relevant,
        primary_brand=primary_brand, brand_detected=brands,
        scores={k: round(v, 3) for k, v in sorted(cat_scores.items(), key=lambda x: -x[1])},
        signals=signals,
    )


# --------------------------------------------------------------------------- #
# Optional LLM refinement
# --------------------------------------------------------------------------- #
def _llm_available(cfg: ScrapeConfig) -> bool:
    import importlib
    try:
        importlib.import_module("anthropic")
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def classify_llm(page: PageContent, url: str, cfg: ScrapeConfig,
                 rule_result: PageClass) -> PageClass:
    """Refine a page label with an LLM. Falls back to the rule result on any error."""
    import json
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception:
        return rule_result

    system = (
        "You classify a web page surfaced during a skincare shopping conversation. "
        "Return ONLY JSON with keys: page_category (one of brand_landing, catalogue, "
        "shopping, editorial, search, community, reference, unrelated), seller_type "
        "(brand_owned, retailer, na), is_skincare (boolean), confidence (0..1). "
        "catalogue/brand_landing = Discovery; shopping = a page to buy a specific product."
    )
    prompt = (
        f"URL: {url}\nTitle: {page.title}\nMeta: {page.meta_description}\n"
        f"H1: {' | '.join(page.h1[:3])}\nSchema types: {', '.join(page.schema_types())}\n"
        f"og:type: {page.og.get('og:type', '')}\n"
        f"Text excerpt: {page.text[:800]}"
    )
    try:
        model = os.environ.get("ANTHROPIC_MODEL", cfg.llm_model)
        msg = client.messages.create(model=model, max_tokens=cfg.llm_max_tokens,
                                      system=system,
                                      messages=[{"role": "user", "content": prompt}])
        txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        data = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    except Exception:
        return rule_result

    cat = str(data.get("page_category", rule_result.page_category))
    sub = rule_result.page_subtype
    out = PageClass(
        page_category=cat if data.get("is_skincare", True) else "unrelated",
        page_subtype=sub, seller_type=str(data.get("seller_type", rule_result.seller_type)),
        funnel_stage=funnel_stage_for(cat, sub),
        confidence=float(data.get("confidence", rule_result.confidence)),
        method="llm", skincare_relevance=rule_result.skincare_relevance,
        is_study_relevant=bool(data.get("is_skincare", rule_result.is_study_relevant)),
        primary_brand=rule_result.primary_brand, brand_detected=rule_result.brand_detected,
        scores=rule_result.scores, signals=rule_result.signals + ["llm"],
    )
    return out


def classify_page(page: PageContent, url: str, cfg: ScrapeConfig,
                  prior: Optional[dict] = None, n_products: int = 0,
                  chat_brands: Optional[set] = None) -> PageClass:
    """Full classification with the configured strategy and graceful fallback."""
    result = classify_rule(page, url, cfg, prior=prior, n_products=n_products,
                           chat_brands=chat_brands)
    want_llm = cfg.classifier == "llm" or (cfg.classifier == "auto"
                                           and result.confidence < 0.55 and _llm_available(cfg))
    if want_llm and _llm_available(cfg):
        result = classify_llm(page, url, cfg, result)
    if result.confidence < cfg.min_confidence:
        result.page_category = "unknown"
    return result
