"""Page taxonomy for the scraped-page classifier.

The user asked to classify a surfaced page as one of three things — a
**catalogue / brand landing page** (the *Discovery* stage of the funnel), a
**shopping page** (external retailer *or* brand-owned), or **unrelated** to the
study — and to propose any further relevant categories. This module is that
proposal, made concrete.

We keep the three requested buckets and add four more that the SimilarWeb
`dim_digital_site.page_type` distribution (search 46k, marketplace 6.7k, …) and
the skincare-shopping literature make unavoidable — editorial "best-of"
listicles, search-engine result pages, community/UGC threads and reference/health
pages are where a large share of discovery and evaluation actually happens. Each
category carries two orthogonal attributes so the headline label stays simple:

* ``seller_type`` — only meaningful for commerce pages: who owns the storefront
  (``brand_owned`` DTC vs ``retailer`` third-party) — this is the "external
  retailer or brand owned" split the user asked for, lifted out of the category
  so ``shopping`` and ``catalogue`` don't each fork in two.
* ``funnel_stage`` — the ``dim_ai_funnel`` stage the page role maps to, so the
  page classifier plugs straight into the journey model (``conveyer.funnel``).

Everything here is plain data + pure functions so the classifier
(``conveyer.scraping.classify``) and the schema (``…schema``) can share one
source of truth.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- #
# Page categories (the headline label)
# --------------------------------------------------------------------------- #
# The three the user named, plus four proposed additions and an explicit
# "unknown" escape hatch for pages we could not fetch or that carry too little
# signal to classify (kept separate from "unrelated", which is a *confident*
# off-topic judgement).
PAGE_CATEGORIES: List[str] = [
    "brand_landing",  # brand homepage / campaign landing page — Discovery
    "catalogue",      # category / collection / listing of many products — Discovery
    "shopping",       # a page to buy a specific product: PDP, cart, checkout — Intent/Purchase
    "editorial",      # reviews, "best X for Y" listicles, blogs, magazines — Evaluation  [PROPOSED]
    "search",         # search-engine or on-site search results (SERP) — Intent           [PROPOSED]
    "community",      # forums / social / UGC (reddit, youtube, tiktok, q&a) — Evaluation  [PROPOSED]
    "reference",      # encyclopedic / health / how-to (wikipedia, .gov/.edu) — Awareness  [PROPOSED]
    "unrelated",      # confidently off-topic for a skincare-shopping study
    "unknown",        # could not fetch / not enough signal to decide
]
CATEGORY_INDEX: Dict[str, int] = {c: i for i, c in enumerate(PAGE_CATEGORIES)}

# One-line human definitions (surfaced in the schema doc + dashboards).
CATEGORY_DEFINITIONS: Dict[str, str] = {
    "brand_landing": "Brand homepage or campaign/landing page — introduces the brand, no single product to buy.",
    "catalogue": "Category, collection or listing page showing many products at once.",
    "shopping": "Transactional page for a specific product — product detail (PDP), cart or checkout.",
    "editorial": "Editorial content: reviews, 'best-of' listicles, buying guides, blog or magazine articles.",
    "search": "Search results — a search engine SERP or a retailer's on-site search listing.",
    "community": "User-generated discussion: forums, social video, Q&A, review communities.",
    "reference": "Reference / informational: encyclopedic, medical or how-to pages, not commerce.",
    "unrelated": "Not part of the skincare purchase journey (off-topic domain or content).",
    "unknown": "Fetch failed or the page carried too little signal to classify.",
}

# --------------------------------------------------------------------------- #
# Orthogonal attributes
# --------------------------------------------------------------------------- #
SELLER_TYPES: List[str] = ["brand_owned", "retailer", "na"]

# Map a page category to the SimilarWeb funnel stage it plays a role in
# (aligned with dim_ai_funnel / conveyer.funnel.STAGES). "unknown"/"unrelated"
# get the Irrelevant sentinel used by dim_ai_funnel (funnel_id -1).
CATEGORY_TO_FUNNEL_STAGE: Dict[str, str] = {
    "brand_landing": "Discovery",
    "catalogue": "Discovery",
    "editorial": "Evaluation",
    "community": "Evaluation",
    "reference": "Awareness",
    "search": "Intent",
    "shopping": "Intent",       # refined to Purchase for cart/checkout subtypes below
    "unrelated": "Irrelevant",
    "unknown": "Irrelevant",
}

# Finer page subtype -> (category, funnel_stage override or None). The subtype is
# what the URL/markup most specifically looks like; the category is the headline.
# This is also the bridge to dim_digital_site.page_type (pdp/search/checkout/
# marketplace/category/retailer_site/brand_site/other).
SUBTYPE_TO_CATEGORY: Dict[str, str] = {
    "homepage": "brand_landing",
    "landing": "brand_landing",
    "brand_site": "brand_landing",
    "collection": "catalogue",
    "category": "catalogue",
    "marketplace": "catalogue",
    "listing": "catalogue",
    "pdp": "shopping",
    "cart": "shopping",
    "checkout": "shopping",
    "order": "shopping",      # order status / history — Post-Purchase infrastructure
    "wishlist": "shopping",   # saved-for-later — Intent, NOT a purchase event
    "article": "editorial",
    "review": "editorial",
    "listicle": "editorial",
    "serp": "search",
    "site_search": "search",
    "forum": "community",
    "social": "community",
    "qa": "community",
    "wiki": "reference",
    "health": "reference",
    "howto": "reference",
    # Service-aware platform routing (multi-service domains like google.com):
    # a maps/places surface is informational; productivity / auth / utility
    # services are confidently off-journey — docs.google.com is not a search
    # page and never part of a shopping funnel, whatever the topic.
    "local": "reference",     # maps / places / store-locator surfaces
    "tool": "unrelated",      # docs, drive, mail, translate, cloud consoles …
    "account": "unrelated",   # sign-in / account management surfaces
    "other": "unknown",
}
# Subtypes that push a shopping page from Intent to Purchase / Post-Purchase.
# The bumps only apply to real shopping pages — an "unknown"/"unrelated" page
# with a cart-looking path must stay Irrelevant.
PURCHASE_SUBTYPES = {"cart", "checkout"}
ORDER_SUBTYPES = {"order"}

# Bridge from the vendor's dim_digital_site.page_type to our subtype, so the
# SimilarWeb label can act as a prior / weak supervision for the classifier.
SIMILARWEB_PAGE_TYPE_TO_SUBTYPE: Dict[str, str] = {
    "pdp": "pdp",
    "search": "serp",
    "checkout": "checkout",
    "marketplace": "marketplace",
    "category": "category",
    "retailer_site": "listing",
    "brand_site": "brand_site",
    "other": "other",
}


def funnel_stage_for(category: str, subtype: str | None = None) -> str:
    """Funnel stage for a (category, subtype), applying the cart/checkout bump."""
    if category == "shopping":
        if subtype in PURCHASE_SUBTYPES:
            return "Purchase"
        if subtype in ORDER_SUBTYPES:
            return "Post-Purchase"
    return CATEGORY_TO_FUNNEL_STAGE.get(category, "Irrelevant")


def category_for_subtype(subtype: str | None) -> str:
    """Headline category for a finer subtype (defaults to 'unknown')."""
    if not subtype:
        return "unknown"
    return SUBTYPE_TO_CATEGORY.get(subtype, "unknown")


def is_commerce(category: str) -> bool:
    """Whether a seller_type is meaningful for this category."""
    return category in ("brand_landing", "catalogue", "shopping")
