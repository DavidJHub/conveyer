"""Synthetic corpus so the whole pipeline runs offline, with ground truth.

Mirrors ``conveyer.ingest.make_synthetic``: when no clickstream data (and no
network) is available, generate a realistic set of skincare web pages — brand
landing pages, catalogues, retailer and brand-owned PDPs, editorial "best-of"
reviews, search-result pages, community threads, reference/health pages, and a
few deliberately off-topic pages — each with proper ``<title>`` / OpenGraph /
JSON-LD markup and a matching (or intentionally non-matching) chat
recommendation. Every page carries a ground-truth ``page_category`` /
``seller_type`` / ``coincides`` label so the classifier and the product matcher
can be scored end-to-end (see :func:`conveyer.scraping.pipeline.evaluate`).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .products import RecMention
from .schema import page_id_for  # noqa: F401  (kept for symmetry / external use)
from .sources import ScrapeSources

# --------------------------------------------------------------------------- #
# A small skincare catalog (brand, brand domain, product, category, price, rating)
# --------------------------------------------------------------------------- #
_CATALOG = [
    ("CeraVe", "cerave.com", "Moisturizing Cream", "moisturizer", 16.99, 4.7, 21843),
    ("The Ordinary", "theordinary.com", "Niacinamide 10% + Zinc 1%", "serum", 6.50, 4.5, 15230),
    ("La Roche-Posay", "laroche-posay.us", "Anthelios SPF 60 Sunscreen", "sunscreen", 35.99, 4.8, 9821),
    ("Paula's Choice", "paulaschoice.com", "2% BHA Liquid Exfoliant", "exfoliant", 34.00, 4.6, 18422),
    ("Cetaphil", "cetaphil.com", "Gentle Skin Cleanser", "cleanser", 13.49, 4.7, 30112),
    ("Drunk Elephant", "drunkelephant.com", "Protini Polypeptide Cream", "moisturizer", 68.00, 4.4, 5210),
    ("Neutrogena", "neutrogena.com", "Hydro Boost Water Gel", "moisturizer", 19.99, 4.6, 12040),
    ("The Inkey List", "theinkeylist.com", "Hyaluronic Acid Serum", "serum", 8.99, 4.5, 6733),
]
_RETAILERS = [("Amazon", "amazon.com"), ("Sephora", "sephora.com"),
              ("Ulta", "ulta.com"), ("Target", "target.com")]
_EDITORIAL = [("Byrdie", "byrdie.com"), ("Allure", "allure.com"),
              ("Wirecutter", "nytimes.com")]


def _jsonld(obj: dict) -> str:
    return f'<script type="application/ld+json">{json.dumps(obj)}</script>'


def _doc(lang: str, title: str, desc: str, og_type: str, canonical: str,
         head_extra: str, body: str, site_name: str = "") -> str:
    og = (f'<meta property="og:type" content="{og_type}">'
          f'<meta property="og:title" content="{title}">'
          f'<meta property="og:description" content="{desc}">'
          + (f'<meta property="og:site_name" content="{site_name}">' if site_name else ""))
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
            f'<title>{title}</title><meta name="description" content="{desc}">'
            f'<link rel="canonical" href="{canonical}">{og}{head_extra}</head>'
            f'<body>{body}</body></html>')


# --------------------------------------------------------------------------- #
# Page archetype builders -> (url, html, category, seller_type, subtype)
# --------------------------------------------------------------------------- #
def _brand_landing(brand, domain) -> Tuple[str, str, str, str, str]:
    url = f"https://www.{domain}/"
    ld = _jsonld({"@context": "https://schema.org", "@type": "Organization",
                  "name": brand, "url": url})
    body = (f"<h1>{brand} — Dermatologist-developed skincare</h1>"
            f"<p>Discover {brand} cleansers, moisturizers and sunscreen for every "
            f"skin concern. Explore our skincare routines.</p>"
            f'<a href="/collections/moisturizers">Shop moisturizers</a>'
            f'<a href="/collections/cleansers">Shop cleansers</a>')
    html = _doc("en", f"{brand} Official Site | Skincare", f"{brand} skincare official site.",
                "website", url, ld, body, site_name=brand)
    return url, html, "brand_landing", "brand_owned", "homepage"


def _catalogue(brand, domain, rec_product, seller_brand=None) -> Tuple[str, str, str, str, str]:
    host = domain if seller_brand is None else seller_brand[1]
    seller = "brand_owned" if seller_brand is None else "retailer"
    slug = rec_product[3]  # recommended product's category (e.g. serum)
    url = f"https://www.{host}/collections/{slug}s"
    # the recommended product first, then a couple of same-category peers
    items = [rec_product] + [c for c in _CATALOG if c[3] == slug and c[0] != brand][:3]
    ld = _jsonld({"@context": "https://schema.org", "@type": "ItemList",
                  "itemListElement": [
                      {"@type": "ListItem", "position": i + 1,
                       "item": {"@type": "Product", "name": f"{b} {p}", "brand": b,
                                "category": cat,
                                "offers": {"@type": "Offer", "price": price,
                                           "priceCurrency": "USD"}}}
                      for i, (b, dom, p, cat, price, rat, rc) in enumerate(items)]})
    body = (f"<h1>Best {slug.capitalize()}s for Every Skin Type</h1><p>Browse our full "
            f"range of {slug}s with hyaluronic acid, niacinamide and ceramides.</p>"
            + "".join(f'<a href="/products/{b.lower().replace(" ", "-")}">{b} {p}</a>'
                      for b, dom, p, cat, price, rat, rc in items))
    html = _doc("en", f"{slug.capitalize()}s — Shop All | Skincare",
                f"Shop all {slug}s.", "website", url, ld, body)
    return url, html, "catalogue", seller, "collection"


def _pdp(brand, domain, product, cat, price, rating, rc, retailer=None
         ) -> Tuple[str, str, str, str, str]:
    seller = "brand_owned" if retailer is None else "retailer"
    host = domain if retailer is None else retailer[1]
    slug = product.lower().replace(" ", "-").replace("%", "").replace("+", "")
    url = f"https://www.{host}/products/{brand.lower().replace(' ', '-')}-{slug}"
    ld = _jsonld({"@context": "https://schema.org", "@type": "Product",
                  "name": f"{brand} {product}", "brand": {"@type": "Brand", "name": brand},
                  "category": cat, "sku": f"{brand[:3].upper()}-{abs(hash(product)) % 100000}",
                  "description": f"{brand} {product} for {cat} — gentle, effective daily skincare.",
                  "aggregateRating": {"@type": "AggregateRating", "ratingValue": rating,
                                      "reviewCount": rc},
                  "offers": {"@type": "Offer", "price": price, "priceCurrency": "USD",
                             "availability": "https://schema.org/InStock"}})
    body = (f"<h1>{brand} {product}</h1>"
            f"<p>{brand} {product} is a {cat} for daily skincare. "
            f"Hydrating, dermatologist-tested. Price ${price:.2f}. Rated {rating}/5.</p>"
            f'<button>Add to cart</button><a href="/cart">Checkout</a>')
    html = _doc("en", f"{brand} {product} | Buy Online", f"Buy {brand} {product}.",
                "product", url, ld, body,
                site_name=(retailer[0] if retailer else brand))
    return url, html, "shopping", seller, "pdp"


def _editorial(pub, domain, focus_products) -> Tuple[str, str, str, str, str]:
    b0, dom0, p0, cat0, price0, rat0, rc0 = focus_products[0]
    url = f"https://www.{domain}/best-{cat0}s-for-dry-skin"
    ld = _jsonld({"@context": "https://schema.org",
                  "@graph": [
                      {"@type": "Article",
                       "headline": f"The 8 Best {cat0.capitalize()}s for Dry Skin, Tested",
                       "author": {"@type": "Person", "name": "Editorial Team"},
                       "publisher": {"@type": "Organization", "name": pub}},
                      {"@type": "ItemList", "itemListElement": [
                          {"@type": "ListItem", "position": i + 1,
                           "item": {"@type": "Product", "name": f"{b} {p}", "brand": b,
                                    "category": cat, "description": f"Editor pick: {b} {p}.",
                                    "aggregateRating": {"@type": "AggregateRating",
                                                        "ratingValue": rat, "reviewCount": rc},
                                    "offers": {"@type": "Offer", "price": price,
                                               "priceCurrency": "USD"}}}
                          for i, (b, dom, p, cat, price, rat, rc) in enumerate(focus_products)]}]})
    picks = "".join(f"<h2>{b} {p}</h2><p>Our review of {b} {p}, a {cat}. "
                    f"Best for hydration. ${price:.2f}.</p>"
                    for b, dom, p, cat, price, rat, rc in focus_products)
    body = ("<h1>The 8 Best Moisturizers for Dry Skin, Tested by Editors</h1>"
            "<p>We tested dozens of skincare moisturizers, serums and cleansers "
            "for dry, sensitive and acne-prone skin.</p>" + picks)
    html = _doc("en", "Best Moisturizers for Dry Skin (2026 Review)",
                "Editor-tested skincare reviews.", "article", url, ld, body, site_name=pub)
    return url, html, "editorial", "na", "article"


def _search(query) -> Tuple[str, str, str, str, str]:
    q = query.replace(" ", "+")
    url = f"https://www.google.com/search?q={q}"
    ld = _jsonld({"@context": "https://schema.org", "@type": "SearchResultsPage"})
    body = (f"<h1>{query}</h1><p>Search results for {query} skincare.</p>"
            f'<a href="https://cerave.com">CeraVe Official</a>'
            f'<a href="https://byrdie.com/best-moisturizers-for-dry-skin">Byrdie review</a>')
    html = _doc("en", f"{query} - Google Search", f"Search results for {query}.",
                "website", url, ld, body)
    return url, html, "search", "na", "serp"


def _community(product, brand) -> Tuple[str, str, str, str, str]:
    slug = product.lower().replace(" ", "_")
    url = f"https://www.reddit.com/r/SkincareAddiction/comments/{abs(hash(slug)) % 10**7}/review_{slug}"
    ld = _jsonld({"@context": "https://schema.org", "@type": "DiscussionForumPosting",
                  "headline": f"Review: {brand} {product}"})
    body = (f"<h1>Has anyone tried {brand} {product}?</h1>"
            f"<p>I've been using {brand} {product} for my acne-prone skin. "
            f"Thoughts on this serum vs other skincare?</p>")
    html = _doc("en", f"{brand} {product} review : SkincareAddiction",
                "Community discussion.", "article", url, ld, body, site_name="Reddit")
    return url, html, "community", "na", "forum"


def _reference(topic) -> Tuple[str, str, str, str, str]:
    slug = topic.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{slug}"
    ld = _jsonld({"@context": "https://schema.org", "@type": "MedicalWebPage",
                  "name": topic})
    body = (f"<h1>{topic}</h1><p>{topic} is a skincare ingredient. This article "
            f"explains how {topic} works for skin, its benefits and side effects. "
            f"Dermatology reference.</p>")
    html = _doc("en", f"{topic} - Wikipedia", f"{topic} reference article.",
                "website", url, ld, body, site_name="Wikipedia")
    return url, html, "reference", "na", "wiki"


def _unrelated(idx) -> Tuple[str, str, str, str, str]:
    # structurally a PDP / article, but off-topic → should collapse to 'unrelated'
    if idx % 2 == 0:
        url = f"https://www.amazon.com/products/gaming-laptop-{idx}"
        ld = _jsonld({"@context": "https://schema.org", "@type": "Product",
                      "name": "UltraBook Pro 16 Gaming Laptop", "brand": "TechCo",
                      "category": "Electronics",
                      "offers": {"@type": "Offer", "price": 1499.00, "priceCurrency": "USD"}})
        body = ("<h1>UltraBook Pro 16 Gaming Laptop</h1><p>Powerful laptop with RTX GPU, "
                "32GB RAM. Add to cart. Free shipping.</p><button>Add to cart</button>")
        html = _doc("en", "UltraBook Pro 16 Gaming Laptop | Buy", "Buy gaming laptop.",
                    "product", url, ld, body, site_name="Amazon")
    else:
        url = f"https://www.investopedia.com/articles/markets-{idx}"
        ld = _jsonld({"@context": "https://schema.org", "@type": "Article",
                      "headline": "How to Diversify Your Investment Portfolio"})
        body = ("<h1>How to Diversify Your Investment Portfolio</h1>"
                "<p>Stocks, bonds and ETFs for long-term investors.</p>")
        html = _doc("en", "Portfolio Diversification Guide", "Investing guide.",
                    "article", url, ld, body, site_name="Investopedia")
    return url, html, "unrelated", "na", ("pdp" if idx % 2 == 0 else "article")


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #
def make_corpus(n_pages: int = 60, seed: int = 42) -> ScrapeSources:
    rng = np.random.default_rng(seed)
    html_by_url: Dict[str, str] = {}
    mentions: Dict[str, List[RecMention]] = {}
    rows: Dict[str, dict] = {}   # keyed by url so duplicates merge, not explode
    gt: Dict[str, dict] = {}
    site_id = 1000

    def add(url, html, cat, seller, subtype, message_id, recommended, coincides_expected,
            page_type_prior=None, seller_prior=None, retailer_brand=None, site_category=None):
        nonlocal site_id
        if html is not None:      # html=None → unfetchable page (URL-only classification)
            html_by_url[url] = html
        if url in rows:  # same URL surfaced on another turn — merge provenance
            r = rows[url]
            if message_id not in r["message_ids"]:
                r["message_ids"].append(message_id)
            r["times_surfaced"] += int(rng.integers(1, 20))
            r["times_recommended"] += int(rng.integers(0, 4)) if recommended else 0
            if recommended and "a_links_source" not in r["sources"]:
                r["sources"].append("a_links_source")
            return
        site_id += 1
        visits = int(rng.integers(0, 30))
        # browsing-trail retention, as computed from next_10_urls request_time
        # deltas on real data (None when the URL was never in a trail)
        dwell = round(float(rng.lognormal(2.8, 0.9)), 3) if visits else None
        rows[url] = {
            "url": url, "digital_site_id": site_id, "domain": _domain_of(url),
            "page_type": page_type_prior, "seller_type": seller_prior,
            "retailer_brand": retailer_brand, "site_category": site_category,
            "extracted_skus": None, "message_ids": [message_id],
            "times_surfaced": int(rng.integers(1, 40)),
            "times_recommended": int(rng.integers(1, 6)) if recommended else 0,
            "times_visited": visits,
            "resulted_in_purchase_any": bool(rng.random() < 0.15),
            "sources": ["click_through"] + (["a_links_source"] if recommended else []),
            "mean_dwell_seconds": dwell,
            "total_dwell_seconds": round(dwell * max(1, visits), 3) if dwell else None,
            "mean_trail_position": round(float(rng.uniform(1, 10)), 2) if visits else None,
        }
        gt[url] = {"url": url, "gt_category": cat, "gt_seller_type": seller,
                   "gt_subtype": subtype, "gt_coincides": coincides_expected}

    turn = 0
    while len(rows) < n_pages:
        brand, domain, product, cat, price, rating, rc = _CATALOG[turn % len(_CATALOG)]
        retailer = _RETAILERS[turn % len(_RETAILERS)]
        pub = _EDITORIAL[turn % len(_EDITORIAL)]
        mid = f"m{turn:04d}"
        # the agent recommended this exact brand/product on this turn
        mentions[mid] = [RecMention(
            recommendation_id=f"r{turn:04d}", message_id=mid,
            entity_context=f"{brand} {product} — a {cat} the assistant recommended",
            brands={brand.lower()}, categories={cat})]

        # brand landing (Discovery) — no product to match
        u, h, c, s, st = _brand_landing(brand, domain)
        add(u, h, c, s, st, mid, recommended=True, coincides_expected=False,
            page_type_prior="brand_site", site_category="Beauty_and_Cosmetics")

        # brand-owned PDP for the recommended product — should coincide
        u, h, c, s, st = _pdp(brand, domain, product, cat, price, rating, rc)
        add(u, h, c, s, st, mid, recommended=True, coincides_expected=True,
            page_type_prior="pdp", seller_prior="1p", site_category="E-commerce_and_Shopping")

        # retailer PDP for the same product — should coincide
        u, h, c, s, st = _pdp(brand, domain, product, cat, price, rating, rc, retailer=retailer)
        add(u, h, c, s, st, mid, recommended=True, coincides_expected=True,
            page_type_prior="pdp", seller_prior="3p", retailer_brand=retailer[0].lower(),
            site_category="E-commerce_and_Shopping")

        # catalogue on the retailer listing the recommended product — Discovery
        u, h, c, s, st = _catalogue(brand, domain, _CATALOG[turn % len(_CATALOG)],
                                    seller_brand=retailer)
        add(u, h, c, s, st, mid, recommended=False, coincides_expected=True,
            page_type_prior="category")

        # editorial review featuring this product — coincides via brand
        u, h, c, s, st = _editorial(pub[0], pub[1], [_CATALOG[turn % len(_CATALOG)]])
        add(u, h, c, s, st, mid, recommended=True, coincides_expected=True)

        # search + community + reference (no vendor prior — classifier stands alone)
        u, h, c, s, st = _search(f"best {cat} for dry skin")
        add(u, h, c, s, st, mid, recommended=False, coincides_expected=False)
        u, h, c, s, st = _community(product, brand)
        add(u, h, c, s, st, mid, recommended=False, coincides_expected=False)
        u, h, c, s, st = _reference(cat.capitalize())
        add(u, h, c, s, st, mid, recommended=False, coincides_expected=False)

        # one off-topic page attached to the same turn
        u, h, c, s, st = _unrelated(turn)
        add(u, h, c, s, st, mid, recommended=False, coincides_expected=False,
            page_type_prior="pdp" if turn % 2 == 0 else None)

        # a cart page that can never be fetched (html=None → offline_miss):
        # the multimodal classifier must label it from URL + domain alone
        cart_url = (f"https://www.{retailer[1]}/gp/cart/view.html?ref_=nav_cart"
                    if retailer[1] == "amazon.com" else f"https://www.{retailer[1]}/cart")
        add(cart_url, None, "shopping", "retailer", "cart", mid,
            recommended=False, coincides_expected=False)

        # an unfetchable deep link on an UNKNOWN indie-brand domain whose BASE
        # page is reachable: the base-URL fallback must supply the topical
        # evidence (fetch_scope="base") that turns unknown into shopping
        indie = f"glowessence-{turn % 3}.com"
        base_url = f"https://www.{indie}/"
        base_body = (f"<h1>GlowEssence Skincare</h1><p>Clean skincare: serums, "
                     f"moisturizers and sunscreen for sensitive skin. Shop our "
                     f"hydrating collection.</p>")
        html_by_url[base_url] = _doc("en", "GlowEssence — Clean Skincare",
                                     "Indie skincare brand.", "website", base_url,
                                     "", base_body, site_name="GlowEssence")
        add(f"https://www.{indie}/products/hydra-serum-{turn}", None,
            "shopping", "na", "pdp", mid, recommended=False, coincides_expected=False)

        turn += 1

    keep = list(rows.keys())[:n_pages]
    urls = pd.DataFrame([rows[u] for u in keep]).reset_index(drop=True)
    gt_df = pd.DataFrame([gt[u] for u in keep]).reset_index(drop=True)
    return ScrapeSources(urls=urls, mentions=mentions, html_by_url=html_by_url,
                         source=f"SYNTHETIC ({len(urls)} pages)", ground_truth=gt_df)


def _domain_of(url: str) -> str:
    from .classify import parse_url
    return parse_url(url).registrable
