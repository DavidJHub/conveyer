"""Page classifier.

Given a parsed :class:`~conveyer.scraping.extract.PageContent` (plus the URL and,
when available, the SimilarWeb ``dim_digital_site`` prior), decide the headline
``page_category`` the user asked for — *catalogue / brand landing* (Discovery),
*shopping* (retailer or brand-owned), *unrelated* — and the four proposed extra
categories (*editorial, search, community, reference*).

The classifier is **multimodal**: independent evidence channels vote and any
one of them can carry a page on its own —

1. **URL structure** — path/query tokens (``/cart/``, ``checkout``,
   ``/dp/…``, ``/search?q=``, ``ref_=nav_cart``, …);
2. **service routing** — multi-service platforms (google, bing, yahoo,
   facebook, …) are routed by subdomain + first path segment BEFORE the
   generic domain vote: ``docs.google.com`` is a document tool, not "search";
   ``shopping.google.com`` is a marketplace; an *unrecognized* Google surface
   gets no domain opinion at all (see ``_PLATFORM_SERVICES``);
3. **domain knowledge** — the registrable domain against curated retailer /
   marketplace / search / community / editorial / reference / brand lists
   (``amazon.com`` *is* a retailer, fetched or not);
4. **page content & markup** — schema.org ``@type``, OpenGraph, price /
   add-to-cart / product-count signals (only when the page was fetched);
5. **vendor prior** — ``dim_digital_site.page_type`` when available;
6. **domain directory** — a :class:`~conveyer.scraping.directory.DomainEntry`
   (role + description) for domains the curated lists don't know, sourced
   from the built-in seed or the external file at
   ``ScrapeConfig.directory_path``. Its description can also stand in as
   content when the page is unfetchable (``content_scope="directory"``);
7. **hosting-platform fingerprint** — response headers / markup identify the
   commerce platform (Shopify, WooCommerce, …): storefront corroboration that
   survives a bot wall (see :mod:`conveyer.scraping.fingerprint`);
8. **learned model** — a self-trained multinomial logistic classifier over
   hashed URL/markup/text features (:mod:`conveyer.scraping.model`), voting
   subtype probabilities scaled by ``ScrapeConfig.model_weight``. Train it on
   the synthetic ground truth and/or your own labelled parquet:
   ``python -m conveyer.scraping.model train``.

Votes are summed, the argmax wins with a softmax confidence, and
``PageClass.signals`` records which modalities actually fired so every label is
auditable. An optional LLM pass (``classifier="llm"`` / ``"auto"`` when
``ANTHROPIC_API_KEY`` is set) refines low-confidence pages.

Topical relevance is a **separate axis** from page structure, with collapse
rules that respect what each modality can know:

* **topic-neutral subtypes** (cart, checkout, order, search results,
  marketplace and storefront homepages) carry no topical tokens *by nature* —
  journey infrastructure, never collapsed for lacking topical evidence. But
  neutrality must be *earned* by a structural vote (URL/markup/prior or a
  decisive domain role): the weak retailer catch-all does not turn
  ``sephora.com/careers`` into a journey page, and a SERP whose URL exposes
  its query is judged by the query text instead;
* **transactional URL tokens are self-evident commerce** on any domain — an
  unfetchable ``/checkouts/c/<token>`` on an unheard-of store is a checkout;
* **topical subtypes** (PDP, article, …) *with* fetched content and no beauty /
  personal-care signal collapse to ``unrelated`` (a confident judgement);
* topical subtypes *without* content keep their *earned* structural category
  when the domain is a known one (an unfetched ``amazon.com/dp/…`` is still a
  shopping page — for *which* product is what ``is_study_relevant`` tracks),
  and fall to ``unknown`` when nothing earns the role.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

if TYPE_CHECKING:  # runtime-free: directory.py imports classify, not vice versa
    from .directory import DomainEntry

from ..brands import BRAND_DOMAIN_CORES
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
# Known brand-owned skincare storefronts, derived from the canonical brand
# lexicon (conveyer.brands) so text mentions and domain matches resolve to the
# same brands. The domain-vs-brand heuristic covers the long tail.
BRAND_DOMAINS = set(BRAND_DOMAIN_CORES)

# Multi-label TLDs so registrable-domain extraction handles co.uk etc.
_MULTI_TLD = {
    "co.uk", "com.au", "co.jp", "com.br", "co.nz", "co.in", "com.mx",
    "co.kr", "com.tr", "co.za", "com.sg", "com.hk", "co.il",
}

# Topical relevance vocabulary. The study's umbrella is **beauty / personal
# care**, not skincare alone: the corpus routinely surfaces haircare, bodycare
# and cosmetics PDPs (a thickening conditioner is as on-topic as a retinol
# serum — the jbca.com counterexample scored 0.0 under the skincare-only list
# and collapsed to 'unrelated'). One hit already clears the relevance bar, so
# every term must be unambiguous on its own: common polysemes stay out —
# "foundation" (charity), "powder" (snow), "primer" (paint), "blush" (verb),
# "curl" (bicep), "cologne" (the city), "soap" (opera), bare "cosmetic"
# (superficial change), bare "beauty" (figurative) — compounds carry those
# meanings instead (curl cream, beauty routine, cosmetics). Extend per-run
# without editing code via ``ScrapeConfig.extra_relevance_terms``.
_RELEVANCE_KEYWORDS = re.compile(
    r"\b("
    # skincare (the original core)
    r"skin ?care|serums?|moisturi[sz]ers?|cleansers?|sunscreen|spf|retinol|"
    r"niacinamide|hyaluronic|salicylic|glycolic|vitamin c|ceramides?|toners?|"
    r"exfoliants?|acne|blackheads?|wrinkles?|dark spots?|hyperpigmentation|"
    r"dermatolog\w*|pores?|breakouts?|rosacea|eczema|face ?creams?|"
    r"eye ?creams?|peptides?|azelaic|anti[- ]ag(?:e?ing)|"
    # haircare ("air conditioner" / "air-conditioner" must not count)
    r"hair ?care|shampoos?|(?<!air )(?<!air-)conditioners?|"
    r"hair (?:masks?|oils?|serums?|sprays?|gels?|dyes?|loss|growth|types?|"
    r"routines?|styling)|scalp|dandruff|frizz|split ends|heat protectants?|"
    r"leave[- ]in|keratin|curl (?:creams?|defin\w*)|curly hair|"
    r"salon[- ]quality|"
    # body / personal care
    r"body ?care|personal care|body (?:wash(?:es)?|lotions?|butters?|scrubs?|"
    r"oils?)|hand creams?|lip balms?|deodorants?|antiperspirants?|"
    r"shower gels?|(?:sulfate|paraben|cruelty)[- ]free|shea butter|argan|"
    r"jojoba|aloe vera|"
    # cosmetics / fragrance ("makeup" one word only — "make up" is a verb)
    r"cosmetics|makeup|mascara|lipsticks?|lip gloss|eyeliners?|eyeshadows?|"
    r"concealers?|bronzers?|nail polish|setting sprays?|fragrances?|perfumes?|"
    r"eau de (?:parfum|toilette)|k[- ]beauty|"
    r"beauty (?:products?|routines?|brands?|editors?|tips?|essentials?)"
    r")\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=32)
def _extra_relevance_pattern(terms: Tuple[str, ...]) -> "Optional[re.Pattern]":
    """User-supplied vocabulary extensions (``ScrapeConfig.extra_relevance_terms``):
    plain case-insensitive phrases, matched whole-word with flexible internal
    whitespace — no regex knowledge required to add "beard oil"."""
    parts = [r"\s+".join(re.escape(w) for w in t.split())
             for t in terms if t and t.strip()]
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


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
    skincare_relevance: float = 0.0    # beauty/personal-care topical score
                                       # (column name kept for schema stability)
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

# A path-anchored transactional vote at or above this weight is treated as
# near-deterministic: /cart/, /checkouts/, /gp/cart, order-history. The weak
# query hints (ref_=nav_cart 1.0, checkout-in-query 0.8) stay advisory.
_TRANSACTIONAL_DECISIVE = 2.0


def _transactional_override(url_votes: Dict[str, float]) -> Optional[str]:
    """The transactional subtype the URL *path* proves, if any.

    A page under /cart/ or /checkouts/ IS a cart/checkout no matter how
    PDP-ish its body looks: cart pages necessarily show prices and
    add-to-cart/checkout phrases, so those markup signals must never outvote
    the URL (the amazon ``/cart/add-to-cart/…`` page scored pdp 3.5 vs cart
    2.5 from exactly that furniture and collapsed to 'unrelated')."""
    cands = {s: w for s, w in url_votes.items()
             if s in TRANSACTIONAL_SUBTYPES and w >= _TRANSACTIONAL_DECISIVE}
    if not cands:
        return None
    return max(cands, key=cands.get)


def is_transactional_url(url: str) -> bool:
    """URL-only: does the path carry a decisive cart/checkout/order/wishlist
    token? The pipeline uses this to suppress the *heuristic* product
    extractor on such pages — a cart's prices are its line items and page
    furniture, not evidence of a product-detail page."""
    return _transactional_override(_url_subtype_votes(parse_url(url))) is not None


_DECISIVE_ROLES = {"marketplace", "retailer", "brand", "search", "community",
                   "editorial", "reference"}


def _known_domain(u: UrlParts, directory_entry: "Optional[DomainEntry]" = None) -> bool:
    """Domain modality has an opinion: the registrable domain is in one of the
    curated lists (or is .gov/.edu, or the domain directory knows its role),
    so its role is known without fetching."""
    if directory_entry is not None and \
            getattr(directory_entry, "role", "") in _DECISIVE_ROLES:
        return True
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

# --------------------------------------------------------------------------- #
# Service-aware routing for multi-service platforms
# --------------------------------------------------------------------------- #
# One registrable domain, many products: "google ∈ SEARCH_DOMAINS ⇒ serp" is
# wrong for most Google URLs — docs.google.com is a document, maps.google.com
# is a places surface, news.google.com is editorial, accounts.google.com is an
# auth wall. The service (subdomain, or first path segment on the www host)
# decides; when a platform is listed here its service verdict REPLACES the
# generic domain vote, and an *unrecognized* service of a strict platform gets
# NO domain opinion at all (URL/markup channels must carry it) instead of a
# false "search" label. Weights ≥ 2.0 are decisive (they earn the subtype).
#
# Two fallback semantics, chosen per platform:
#   "fallback": {}   — strict: unmatched service ⇒ no domain opinion (google);
#   no "fallback" key — open: unmatched ⇒ the generic curated-list vote
#                       (amazon www pages keep behaving like a retailer).
_TOOL = {"tool": 2.5}
_ACCT = {"account": 2.5}
_WIKI = {"wiki": 2.5}
_NEWS = {"article": 2.5}
_PLATFORM_SERVICES: Dict[str, dict] = {
    "google": {
        "subdomains": {
            "shopping": {"marketplace": 2.6}, "store": {"collection": 2.0},
            "news": _NEWS, "support": _WIKI, "scholar": _WIKI, "books": _WIKI,
            "developers": _WIKI, "maps": {"local": 2.5}, "lens": {"serp": 2.5},
            "images": {"serp": 2.5}, "docs": _TOOL, "drive": _TOOL,
            "mail": _TOOL, "calendar": _TOOL, "meet": _TOOL, "photos": _TOOL,
            "translate": _TOOL, "pay": _TOOL, "gemini": _TOOL, "ads": _TOOL,
            "analytics": _TOOL, "cloud": _TOOL, "firebase": _TOOL,
            "colab": _TOOL, "sites": _TOOL, "groups": {"forum": 2.5},
            "accounts": _ACCT, "myaccount": _ACCT,
            "play": {"marketplace": 2.0},
        },
        "paths": {
            "search": {"serp": 3.0}, "webhp": {"serp": 3.0}, "imghp": {"serp": 3.0},
            "shopping": {"marketplace": 2.6}, "maps": {"local": 2.5},
            "travel": _TOOL, "flights": _TOOL, "finance": {"article": 2.0},
            "books": _WIKI, "forms": _TOOL, "sheets": _TOOL, "slides": _TOOL,
            "drive": _TOOL, "intl": _TOOL, "business": _TOOL, "chrome": _TOOL,
            "url": {"tool": 1.0},   # redirect wrapper; the target URL is the real page
        },
        "root": {"serp": 3.0},
        "fallback": {},
    },
    "bing": {
        "paths": {"search": {"serp": 3.0}, "images": {"serp": 2.5},
                  "videos": {"serp": 2.5}, "news": {"serp": 2.5},
                  "shop": {"marketplace": 2.4}, "maps": {"local": 2.5},
                  "ck": {"tool": 1.0}},   # /ck/a click-tracking redirect
        "root": {"serp": 3.0},
        "fallback": {"serp": 1.5},
    },
    "yahoo": {
        "subdomains": {"search": {"serp": 3.0}, "mail": _TOOL,
                       "finance": {"article": 2.0}, "news": _NEWS,
                       "sports": _NEWS, "shopping": {"marketplace": 2.4},
                       "login": _ACCT},
        "paths": {"news": _NEWS, "lifestyle": {"article": 2.0},
                  "entertainment": {"article": 2.0}},
        "root": {"serp": 3.0},
        "fallback": {},
    },
    "amazon": {   # open fallback: www pages keep the retailer behaviour
        "subdomains": {"aws": _TOOL, "music": _TOOL, "advertising": _TOOL,
                       "affiliate-program": _TOOL, "developer": _WIKI,
                       "sellercentral": _TOOL, "kdp": _TOOL, "read": _TOOL},
    },
    "facebook": {
        "subdomains": {"business": _TOOL, "developers": _WIKI, "ads": _TOOL},
        "paths": {"marketplace": {"marketplace": 2.6}, "business": _TOOL,
                  "help": _WIKI, "policies": _WIKI, "login": _ACCT},
    },
    "instagram": {
        "paths": {"shop": {"marketplace": 2.2}, "accounts": _ACCT},
    },
    "youtube": {
        "subdomains": {"music": _TOOL, "studio": _TOOL, "support": _WIKI},
        "paths": {"shopping": {"marketplace": 2.2}},
    },
    "x": {"subdomains": {"ads": _TOOL, "business": _TOOL, "help": _WIKI,
                         "developer": _WIKI}},
    "twitter": {"subdomains": {"ads": _TOOL, "business": _TOOL, "help": _WIKI,
                               "developer": _WIKI}},
    "apple": {
        "subdomains": {"support": _WIKI, "developer": _WIKI, "music": _TOOL,
                       "tv": _TOOL, "podcasts": _TOOL, "books": _TOOL,
                       "apps": {"marketplace": 2.0}, "appstoreconnect": _TOOL,
                       "id": _ACCT, "icloud": _TOOL},
        "paths": {"shop": {"collection": 2.0}, "store": {"collection": 2.0}},
    },
    "microsoft": {
        "subdomains": {"support": _WIKI, "learn": _WIKI, "docs": _WIKI,
                       "azure": _TOOL, "login": _ACCT, "account": _ACCT},
        "paths": {"store": {"marketplace": 2.0}},
    },
}

# subdomain labels that are presentation variants, not services
_TRANSPARENT_SUBDOMAINS = {"www", "www2", "m", "mobile", "amp", "en", "us", "l"}


def _deep_marketplace_demoted(votes: Dict[str, float], u: UrlParts) -> Dict[str, float]:
    """Topic-neutrality belongs to the marketplace SURFACE, not its items:
    ``facebook.com/marketplace`` is journey infrastructure, but
    ``facebook.com/marketplace/item/<id>`` is one specific listing — fetched
    off-topic content must still be able to collapse it. Deep paths keep the
    same decisive weight under the non-neutral ``listing`` subtype."""
    if "marketplace" not in votes:
        return votes
    segs = [s for s in u.path.strip("/").split("/") if s]
    if len(segs) >= 2:
        votes = dict(votes)
        votes["listing"] = max(votes.pop("marketplace"), votes.get("listing", 0.0))
    return votes


def _service_votes(u: UrlParts) -> Optional[Dict[str, float]]:
    """Service verdict for a multi-service platform URL.

    Returns ``None`` when the domain is not a listed platform or the platform
    is open and nothing matched (caller proceeds with the generic domain
    votes); ``{}`` when the platform matched but the service is unrecognized
    on a strict platform (caller suppresses the generic domain vote — an
    unknown Google surface must NOT read as "search"); otherwise the matched
    service's subtype votes."""
    plat = _PLATFORM_SERVICES.get(u.core)
    if plat is None:
        return None
    sub = u.subdomain.split(".")[-1] if u.subdomain else ""
    # locale labels (cn.bing.com, es.yahoo.com, en.m.wikipedia-style prefixes)
    # are presentation, not services — treat like www
    if sub in _TRANSPARENT_SUBDOMAINS or (len(sub) == 2 and sub.isalpha()):
        sub = ""
    if sub:
        rules = plat.get("subdomains", {})
        if sub in rules:
            return _deep_marketplace_demoted(dict(rules[sub]), u)
        return plat.get("fallback")   # unknown subdomain of the platform
    seg = u.path.strip("/").split("/", 1)[0] if u.path.strip("/") else ""
    prules = plat.get("paths", {})
    if seg and seg in prules:
        return _deep_marketplace_demoted(dict(prules[seg]), u)
    if u.path in _ROOT_PATHS and "root" in plat:
        return dict(plat["root"])
    return plat.get("fallback")


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
    # editorial slugs where the token sits INSIDE the hyphenated slug:
    # /how-to-choose-the-best-sunscreen/ carries both markers
    elif re.search(r"(^|/|-)best-[a-z0-9]", path) or \
            re.search(r"(^|/)how-to-[a-z0-9]", path):
        add("article", 1.5)
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


# directory role -> subtype votes, mirroring the curated-list weights in
# _domain_votes so an external-file retailer behaves exactly like a curated one
_ROLE_SUBTYPE_VOTES: Dict[str, Dict[str, float]] = {
    "search": {"serp": 3.0},
    "community": {"forum": 3.0},
    "editorial": {"article": 2.5},
    "reference": {"wiki": 2.5},
    "brand": {"brand_site": 1.5},
    "retailer": {"listing": 0.6},
    "marketplace": {"marketplace": 1.2, "listing": 0.6},
}


def _directory_votes(entry: "Optional[DomainEntry]", u: UrlParts) -> Dict[str, float]:
    """Votes from the domain directory's role — only consulted when the curated
    domain lists had nothing to say, so the two never double-count."""
    if entry is None:
        return {}
    role = getattr(entry, "role", "")
    v = dict(_ROLE_SUBTYPE_VOTES.get(role, {}))
    if role in ("retailer", "marketplace") and u.path in _ROOT_PATHS:
        v["marketplace"] = v.get("marketplace", 0.0) + 2.6
    return v


def _platform_votes(platform: str, u: "Optional[UrlParts]" = None) -> Dict[str, float]:
    """Corroborating votes from the hosting-platform fingerprint. A commerce
    platform (Shopify, WooCommerce, …) says "this domain is a storefront" —
    domain-level evidence, deliberately below the 2.0 'earned' bar: it
    corroborates a /products/… URL on a bot-walled store, it does not carry a
    page alone. On a CURATED retailer/marketplace domain the fingerprint adds
    nothing the lists don't already know — and its brand_site vote must never
    outvote the retailer role (credobeauty.com runs Shopify but is a curated
    retailer, not a DTC brand site)."""
    if not platform:
        return {}
    if u is not None and (u.core in RETAILER_DOMAINS or u.core in MARKETPLACE_DOMAINS):
        return {}
    from .fingerprint import is_commerce_platform
    if is_commerce_platform(platform):
        return {"brand_site": 1.2, "pdp": 0.4, "collection": 0.3}
    return {}


def _model_votes(page: Optional[PageContent], url: str, cfg: ScrapeConfig,
                 platform: str = "") -> Dict[str, float]:
    """The learned channel: subtype probabilities from the trained model (see
    :mod:`conveyer.scraping.model`), scaled by ``cfg.model_weight``. Returns
    ``{}`` whenever the model is disabled, missing, or fails — the rule
    channels never depend on it."""
    if not getattr(cfg, "use_learned_model", False):
        return {}
    try:
        from .model import predict_votes
        return predict_votes(page, url, cfg, platform=platform)
    except Exception:
        return {}


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


def _topical_relevance(page: PageContent, u: UrlParts, chat_brands: set,
                       prior_category: str = "",
                       extra_terms: Tuple[str, ...] = ()) -> float:
    """0–1 beauty / personal-care topical score. Stored in the
    ``skincare_relevance`` column — the name is kept for schema stability
    across existing parquets, but the vocabulary covers the whole umbrella
    (skincare, haircare, bodycare, cosmetics) plus ``extra_terms``."""
    # URL slugs are evidence too — critical when the body couldn't be fetched:
    # /product/cerave-moisturizing-cream is on-topic even with zero HTML.
    url_text = re.sub(r"[-_+/=&?.]", " ", f"{u.path} {u.query}")
    text = " ".join([page.title, page.meta_description, " ".join(page.h1),
                     page.text[:3000], url_text])
    hits = len(_RELEVANCE_KEYWORDS.findall(text))
    extra = _extra_relevance_pattern(tuple(extra_terms or ()))
    if extra is not None:
        hits += len(extra.findall(text))
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
                  chat_brands: Optional[set] = None,
                  domain_profile: Optional[dict] = None,
                  directory_entry: "Optional[DomainEntry]" = None,
                  content_scope: str = "page",
                  platform: str = "") -> PageClass:
    """``domain_profile`` is knowledge *learned from other pages of the same
    domain* ({"relevance": 0..1, "seller": ...}): it floors topical relevance
    and fills a missing seller_type, so URLs skipped by the fetch policy (or
    unreachable) still classify from what the domain already taught us —
    without ever copying a page-level label across the domain.
    ``directory_entry`` is offline knowledge about the *site* (see
    :mod:`conveyer.scraping.directory`): its role votes when the curated
    domain lists are silent, and its site category stands in for the vendor
    prior. ``content_scope`` says whose words ``page`` carries — "page" (its
    own), "stripped" (the query-stripped variant of the same document — also
    its own), "base" (the base URL's), "directory" (the entry's description)
    or "none": only the page's OWN content may prove it unrelated.
    ``platform`` is the hosting-platform fingerprint from the response headers
    (see :mod:`conveyer.scraping.fingerprint`) — a Shopify/WooCommerce store
    is storefront evidence even when the page itself was bot-walled."""
    chat_brands = chat_brands or set()
    u = parse_url(url or page.url)
    prior = prior or {}
    entry_role = getattr(directory_entry, "role", "") if directory_entry is not None else ""

    # each modality votes independently; any one can carry the page alone.
    # Service-aware routing replaces the generic domain vote on multi-service
    # platforms (docs.google.com must not read as "search"); the directory
    # only speaks when both the platform table and curated lists are silent.
    svc = _service_votes(u)
    service_votes = dict(svc) if svc else {}
    domain_votes = _domain_votes(u, page, chat_brands) if svc is None else {}
    modality_votes = {
        "url": _url_subtype_votes(u),
        "service": service_votes,
        "domain": domain_votes,
        "markup": _markup_votes(page, n_products),
        "platform": _platform_votes(platform, u),
        "directory": {} if (svc is not None or domain_votes)
        else _directory_votes(directory_entry, u),
        # the model sees only what training saw: the page's OWN content or the
        # bare URL — never a base-page/directory stand-in (train/serve match).
        # The query-stripped variant IS the page's own document.
        "model": _model_votes(page if content_scope in ("page", "stripped") else None,
                              url, cfg, platform),
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

    # the SimilarWeb site-category prior, with the directory's category as the
    # stand-in when the vendor table is silent about this domain
    prior_cat = str(prior.get(cfg.col_site_category, "") or "")
    if not prior_cat and directory_entry is not None:
        prior_cat = getattr(directory_entry, "category", "") or ""

    if not votes:
        return PageClass(page_category="unknown", page_subtype="other",
                         confidence=0.0, method="rule", signals=signals,
                         skincare_relevance=_topical_relevance(
                             page, u, chat_brands, prior_cat,
                             tuple(cfg.extra_relevance_terms or ())))

    subtype = max(votes, key=votes.get)
    # decisive transactional URL tokens win the subtype outright: prices and
    # checkout buttons are *expected furniture* on a cart page, so the markup
    # channel's pdp votes are not evidence against the URL's /cart/ path
    override = _transactional_override(modality_votes["url"])
    if override is not None and subtype != override:
        subtype = override
        signals.append("url_override")
    category = category_for_subtype(subtype)
    # collapse subtype votes into category-level scores for confidence
    cat_scores: Dict[str, float] = {}
    for sub, w in votes.items():
        cat_scores[category_for_subtype(sub)] = cat_scores.get(category_for_subtype(sub), 0.0) + w
    confidence = _softmax_conf(cat_scores)

    relevance = _topical_relevance(page, u, chat_brands, prior_cat,
                                   tuple(cfg.extra_relevance_terms or ()))
    profile_used = False
    if domain_profile:
        prof_rel = float(domain_profile.get("relevance", 0.0) or 0.0)
        if prof_rel > relevance:
            relevance, profile_used = round(prof_rel, 3), True
    known = _known_domain(u) or (domain_profile is not None
                                 and float(domain_profile.get("relevance", 0) or 0) >= 0.15)
    # a fingerprinted commerce platform vouches for the domain's STRUCTURAL
    # role only — an unfetchable /products/… on a Shopify store keeps the
    # shopping category, like an unfetched amazon.com/dp/…. It must NOT feed
    # the (neutral and known) relevance shortcut: an off-topic Shopify store's
    # homepage is not journey infrastructure just because Shopify hosts it.
    structural_known = known
    if not structural_known and platform:
        from .fingerprint import is_commerce_platform
        structural_known = is_commerce_platform(platform)
    # the winning subtype must be *earned* by structural evidence: a URL/markup
    # vote, a matching vendor prior, or a decisive domain/directory role
    # (>= 2.0 — search engine, community, editorial, reference, storefront
    # root). The retailer catch-alls (listing 0.6, bare marketplace 1.2) earn
    # nothing on their own.
    prior_sub = SIMILARWEB_PAGE_TYPE_TO_SUBTYPE.get(prior_page_type) if prior_fired else None
    earned = (subtype in modality_votes["url"] or subtype in modality_votes["markup"]
              or subtype == prior_sub or modality_votes["domain"].get(subtype, 0.0) >= 2.0
              or modality_votes["service"].get(subtype, 0.0) >= 2.0
              or modality_votes["directory"].get(subtype, 0.0) >= 2.0
              # model earns only with a real, confident vote (p >= 0.5); the
              # chained 0 < guard keeps model_weight=0 from making 0 >= 0 true
              or 0.0 < 0.5 * cfg.model_weight <= modality_votes["model"].get(subtype, 0.0))
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
        if subtype == "brand_site" or u.core in BRAND_DOMAINS or entry_role == "brand":
            seller = "brand_owned"
        elif (u.core in RETAILER_DOMAINS or prior.get(cfg.col_retailer_brand)
              or entry_role in ("retailer", "marketplace")):
            seller = "retailer"
        elif str(prior.get(cfg.col_seller_type, "")).lower() in ("1p", "3p"):
            seller = "retailer"
        elif domain_profile and domain_profile.get("seller") in ("brand_owned", "retailer"):
            seller = domain_profile["seller"]
            profile_used = True
        elif platform:
            # a self-hosted commerce platform (Shopify/WooCommerce/…) on a
            # domain no list knows is a DTC storefront in the overwhelming case
            from .fingerprint import platform_seller
            seller = platform_seller(platform) or seller

    # topical relevance overrides the headline bucket (user's "unrelated" case).
    # "unrelated" is a *confident* judgement — it needs the page's OWN content
    # to stand on; stand-in content (base page / directory description) is
    # domain-level evidence and cannot prove a deep link off-topic. Without
    # real content an *earned* structural role stands in: an unfetched
    # amazon.com/dp/… is still a shopping page, an unfetched
    # /checkouts/c/<token> is still a checkout. A page whose only support is
    # the weak retailer catch-all (sephora.com/careers) tells us nothing — and
    # a domain description alone must not mint a category for it → unknown.
    real_content = has_content and content_scope in ("page", "stripped")
    final_category = category
    if category != "unknown":
        if not is_relevant:
            if real_content:
                final_category = "unrelated"
            elif not (earned and (structural_known or self_evident)):
                final_category = "unknown"
        elif not real_content and not (earned or self_evident):
            final_category = "unknown"
    # invariant: a page whose FINAL headline is 'unrelated' (off-topic collapse
    # or an intrinsically off-journey subtype like tool/account) is never
    # study-relevant — a shared Google Doc that happens to mention retinol is
    # still not part of anyone's shopping journey
    if final_category == "unrelated":
        is_relevant = False

    # page-intrinsic brand (the brand this page is *about*) — used for matching.
    # Deliberately NOT the chat brand: letting a brand-less product inherit the
    # chat brand would make everything trivially "coincide".
    page_brand = ""
    if u.core in BRAND_DOMAINS or entry_role == "brand":
        page_brand = u.core
    elif page.og.get("product:brand"):
        page_brand = normalize(page.og["product:brand"])
    primary_brand = page_brand
    brands = sorted({b for b in chat_brands} | ({page_brand} if page_brand else set()))
    brands = [b for b in brands if b]

    if profile_used:
        signals.append("domain_profile")
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
                  chat_brands: Optional[set] = None,
                  domain_profile: Optional[dict] = None,
                  directory_entry: "Optional[DomainEntry]" = None,
                  content_scope: str = "page",
                  platform: str = "") -> PageClass:
    """Full classification with the configured strategy and graceful fallback."""
    result = classify_rule(page, url, cfg, prior=prior, n_products=n_products,
                           chat_brands=chat_brands, domain_profile=domain_profile,
                           directory_entry=directory_entry, content_scope=content_scope,
                           platform=platform)
    want_llm = cfg.classifier == "llm" or (cfg.classifier == "auto"
                                           and result.confidence < 0.55 and _llm_available(cfg))
    if want_llm and _llm_available(cfg):
        result = classify_llm(page, url, cfg, result)
    if result.confidence < cfg.min_confidence:
        result.page_category = "unknown"
    return result
