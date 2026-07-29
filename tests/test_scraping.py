"""Tests for conveyer.scraping.

Runs under pytest *or* standalone (``python tests/test_scraping.py``) so it works
in this repo, which has no pytest harness. Everything is offline — the fetcher
tests use a local 127.0.0.1 HTTP server, never the internet.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import numpy as np
import pandas as pd

from conveyer.scraping import ScrapeConfig, run_scrape
from conveyer.scraping.classify import classify_rule, parse_url
from conveyer.scraping.extract import extract_page
from conveyer.scraping.fetch import Fetcher
from conveyer.scraping.products import (ProductRecord, RecMention, extract_products,
                                        match_products)
from conveyer.scraping.schema import PAGE_SCHEMA, PRODUCT_SCHEMA, to_arrow_table
from conveyer.scraping.sources import _as_records, build_sources, trail_events
from conveyer.scraping.synthetic import make_corpus

PROD_HTML = """<!doctype html><html lang="en"><head><title>CeraVe Moisturizing Cream</title>
<meta name="description" content="Rich moisturizer for dry skin">
<meta property="og:type" content="product">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"CeraVe Moisturizing Cream",
 "brand":{"@type":"Brand","name":"CeraVe"},"category":"moisturizer","sku":"CRV-001",
 "description":"Ceramide moisturizer","aggregateRating":{"@type":"AggregateRating",
 "ratingValue":"4.7","reviewCount":"21843"},
 "offers":{"@type":"Offer","price":"16.99","priceCurrency":"USD","availability":"https://schema.org/InStock"}}
</script></head><body><h1>CeraVe Moisturizing Cream</h1>
<p>Best moisturizer with ceramides. $16.99.</p><button>Add to cart</button></body></html>"""

GRAPH_HTML = """<html><head><script type="application/ld+json">
{"@graph":[{"@type":"WebPage","name":"x"},{"@type":"Product","name":"The Ordinary Niacinamide",
"brand":"The Ordinary","offers":{"@type":"Offer","price":6.5,"priceCurrency":"USD"}}]}
</script></head><body>skincare serum</body></html>"""


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_basic():
    p = extract_page(PROD_HTML, "https://cerave.com/products/moisturizing-cream")
    _check(p.title == "CeraVe Moisturizing Cream", "title")
    _check(p.meta_description.startswith("Rich moisturizer"), "meta desc")
    _check(p.og.get("og:type") == "product", "og:type")
    _check("product" in p.schema_types(), f"schema types {p.schema_types()}")
    _check(p.has_price and p.has_add_to_cart, "price/cart signals")
    _check(p.h1 == ["CeraVe Moisturizing Cream"], f"h1 {p.h1}")


def test_extract_robust_to_garbage():
    for bad in ["", "<html><body><p>unclosed", "<<>>not html", "<script>bad json</script>"]:
        p = extract_page(bad, "http://x.com")
        _check(isinstance(p.schema_types(), list), "no crash on garbage")


def test_extract_graph_unwrap():
    prods = extract_products(extract_page(GRAPH_HTML, "http://x.com"))
    names = [pr.name for pr in prods]
    _check(any("Niacinamide" in n for n in names), f"@graph product not found: {names}")


def test_product_metadata():
    prods = extract_products(extract_page(PROD_HTML, "http://cerave.com/p"))
    _check(len(prods) >= 1, "one product")
    pr = [x for x in prods if x.price == 16.99][0]
    _check(pr.currency == "USD", "currency")
    _check(abs((pr.rating or 0) - 4.7) < 1e-6, "rating")
    _check(pr.rating_count == 21843, f"rating_count {pr.rating_count}")
    _check(pr.sku == "CRV-001", "sku")
    _check(pr.category == "moisturizer", "category")
    _check(pr.availability == "InStock", f"availability {pr.availability}")


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def test_matching_positive_and_negative():
    cerave = ProductRecord(name="CeraVe Moisturizing Cream", brand="CeraVe", price=16.99)
    laptop = ProductRecord(name="UltraBook Pro Laptop", brand="TechCo", price=1499.0)
    brandless = ProductRecord(name="Some Cream", brand="", price=9.0)
    mention = RecMention(recommendation_id="r1", message_id="m1",
                         entity_context="CeraVe Moisturizing Cream", brands={"cerave"},
                         categories={"moisturizer"})
    match_products([cerave, laptop, brandless], page_brand="", mentions=[mention],
                   message_id="m1", coincide_threshold=0.5)
    _check(cerave.coincides and cerave.match_type == "brand+name", "brand+name match")
    _check(cerave.match_strength == "exact", f"strength {cerave.match_strength}")
    _check(not laptop.coincides, "laptop must not coincide")
    _check(not brandless.coincides, "brandless product must not inherit chat brand")


# --------------------------------------------------------------------------- #
# URL parsing & classification
# --------------------------------------------------------------------------- #
def test_url_parsing():
    u = parse_url("https://www.amazon.co.uk/dp/B00X")
    _check(u.registrable == "amazon.co.uk", f"registrable {u.registrable}")
    _check(u.core == "amazon", f"core {u.core}")
    u2 = parse_url("https://shop.cerave.com/collections/moisturizers")
    _check(u2.core == "cerave" and u2.subdomain == "shop", f"{u2}")


def test_classify_categories():
    cfg = ScrapeConfig()
    cases = {
        "https://www.google.com/search?q=best+moisturizer": "search",
        "https://www.reddit.com/r/SkincareAddiction/comments/1/x": "community",
        "https://en.wikipedia.org/wiki/Retinol": "reference",
        "https://www.byrdie.com/best-moisturizers-review": "editorial",
    }
    for url, expected in cases.items():
        html = f"<html><head><title>skincare {expected}</title></head><body>skincare serum moisturizer acne retinol</body></html>"
        res = classify_rule(extract_page(html, url), url, cfg, chat_brands={"cerave"})
        _check(res.page_category == expected, f"{url} -> {res.page_category} (want {expected})")


def test_classify_unrelated_offtopic():
    cfg = ScrapeConfig()
    url = "https://www.amazon.com/products/gaming-laptop"
    html = ('<html><head><title>Gaming Laptop</title><meta property="og:type" content="product">'
            '<script type="application/ld+json">{"@type":"Product","name":"Laptop",'
            '"offers":{"price":1499,"priceCurrency":"USD"}}</script></head>'
            '<body><h1>Gaming Laptop</h1><p>RTX GPU. Add to cart.</p></body></html>')
    res = classify_rule(extract_page(html, url), url, cfg, chat_brands={"cerave"}, n_products=1)
    _check(res.page_category == "unrelated", f"off-topic -> {res.page_category}")
    _check(res.page_subtype == "pdp", f"subtype preserved: {res.page_subtype}")
    # unfetched, but the domain is a known retailer: the structural role holds
    # (it IS a shopping page — topicality is is_study_relevant's job)
    empty = classify_rule(extract_page("", url), url, cfg)
    _check(empty.page_category == "shopping", f"known-domain PDP -> {empty.page_category}")
    _check(not empty.is_study_relevant, "no topical evidence -> not study-relevant")
    cera_url = "https://www.sephora.com/product/cerave-moisturizing-cream"
    cera = classify_rule(extract_page("", cera_url), cera_url, cfg)
    _check(cera.page_category == "shopping", f"URL-slug evidence -> {cera.page_category}")
    _check(cera.is_study_relevant, "brand slug in URL -> study-relevant")
    # unfetched on an UNKNOWN domain: nothing to stand on -> unknown
    mystery = "https://someblog-nobody-knows.xyz/products/gaming-laptop"
    myst = classify_rule(extract_page("", mystery), mystery, cfg)
    _check(myst.page_category == "unknown", f"unknown domain, no content -> {myst.page_category}")


def test_multimodal_url_only():
    """Obvious URLs must classify correctly with ZERO fetched content —
    domain modality + URL tokens carry the page (the amazon-cart regression)."""
    cfg = ScrapeConfig()
    cases = {
        # url: (category, subtype, seller, funnel, study_relevant)
        # the reported regression: retailer cart, never fetchable
        "https://www.amazon.com/gp/cart/view.html?ref_=nav_cart":
            ("shopping", "cart", "retailer", "Purchase", True),
        "https://www.sephora.com/checkout": ("shopping", "checkout", "retailer", "Purchase", True),
        "https://www.ulta.com/cart": ("shopping", "cart", "retailer", "Purchase", True),
        "https://www.amazon.com/gp/aw/c": ("shopping", "cart", "retailer", "Purchase", True),
        # transactional tokens are self-evident commerce on ANY domain —
        # /checkouts/c/<token> must also beat the single-letter /c/ collection vote
        "https://glowlab.myshopify.com/checkouts/c/0a1b2c3d":
            ("shopping", "checkout", "na", "Purchase", True),
        # order history is Post-Purchase infrastructure, wishlist is Intent
        "https://www.amazon.com/gp/css/order-history":
            ("shopping", "order", "retailer", "Post-Purchase", True),
        "https://www.amazon.com/hz/wishlist/ls/ABCD123":
            ("shopping", "wishlist", "retailer", "Intent", True),
        # retailer roots (incl. locale roots) = storefront entry, not a brand landing
        "https://www.amazon.com/": ("catalogue", "marketplace", "retailer", "Discovery", True),
        "https://www.target.com/us": ("catalogue", "marketplace", "retailer", "Discovery", True),
        # opaque ASIN PDP: role holds from URL+domain; the product (and hence
        # topicality) stays unknown -> not study-relevant until fetched
        "https://www.amazon.com/dp/B00365FJ8K": ("shopping", "pdp", "retailer", "Intent", False),
        # a SERP that exposes its query is judged by the query text
        "https://www.google.com/search?q=best+gaming+laptops": ("search", "serp", "na", "Intent", False),
        "https://www.google.com/search?q=best+retinol": ("search", "serp", "na", "Intent", True),
    }
    for u, (cat, sub, seller, funnel, relevant) in cases.items():
        r = classify_rule(extract_page("", u), u, cfg)
        _check(r.page_category == cat, f"{u} -> {r.page_category} (want {cat})")
        _check(r.page_subtype == sub, f"{u} subtype {r.page_subtype} (want {sub})")
        _check(r.seller_type == seller, f"{u} seller {r.seller_type} (want {seller})")
        _check(r.funnel_stage == funnel, f"{u} funnel {r.funnel_stage} (want {funnel})")
        _check("url" in r.signals, f"{u} signals {r.signals} must include url")
        _check(r.is_study_relevant == relevant,
               f"{u}: is_study_relevant={r.is_study_relevant} (want {relevant})")


def test_neutrality_must_be_earned():
    """The weak retailer catch-all votes (listing 0.6 / marketplace 1.2) must
    not turn /careers, /help or /prime into relevant journey pages."""
    cfg = ScrapeConfig()
    for u in ("https://www.sephora.com/careers", "https://www.amazon.com/prime",
              "https://www.walmart.com/help"):
        r = classify_rule(extract_page("", u), u, cfg)
        _check(r.page_category == "unknown", f"{u} -> {r.page_category} (want unknown)")
        _check(not r.is_study_relevant, f"{u} must not be study-relevant")
    # with fetched off-topic content the judgement hardens to 'unrelated'
    jobs = ("<html><head><title>Careers at Sephora</title></head><body><h1>Careers</h1>"
            "<p>Join our team. Open roles in engineering and retail operations.</p></body></html>")
    u = "https://www.sephora.com/careers"
    r = classify_rule(extract_page(jobs, u), u, cfg)
    _check(r.page_category == "unrelated", f"fetched careers -> {r.page_category}")
    # an 'unknown' page with a cart-ish path must not be staged Purchase
    from conveyer.scraping.taxonomy import funnel_stage_for
    _check(funnel_stage_for("unknown", "cart") == "Irrelevant", "no Purchase bump off-shopping")
    _check(funnel_stage_for("shopping", "order") == "Post-Purchase", "order stage")


# --------------------------------------------------------------------------- #
# Browsing-trail parsing & dwell (next_10_urls)
# --------------------------------------------------------------------------- #
def test_as_records_real_world_shapes():
    recs = [{"request_time": "1", "requested_site": "https://a.com"}]
    _check(_as_records(np.array(recs, dtype=object)) == recs, "numpy array of dicts")
    _check(_as_records(recs) == recs, "plain list")
    _check(_as_records(None) == [] and _as_records(float("nan")) == [], "null-ish")
    parsed = _as_records("[{'request_time': '1', 'requested_site': 'https://a.com'}]")
    _check(parsed == recs, f"single-quoted repr string: {parsed}")


def test_trail_events_dwell():
    cell = [
        {"request_time": "1769448157367", "requested_site": "https://a.com/x"},
        {"request_time": "1769448158742", "requested_site": "https://b.com/y"},
        {"request_time": "1769448171980", "requested_site": "https://c.com/z"},
    ]
    ev = trail_events(cell)
    _check([e["url"] for e in ev] == ["https://a.com/x", "https://b.com/y", "https://c.com/z"],
           "order preserved")
    _check(abs(ev[0]["dwell_seconds"] - 1.375) < 1e-9, f"dwell0 {ev[0]['dwell_seconds']}")
    _check(abs(ev[1]["dwell_seconds"] - 13.238) < 1e-9, f"dwell1 {ev[1]['dwell_seconds']}")
    _check(ev[2]["dwell_seconds"] is None, "last entry has no successor -> no dwell")
    _check([e["position"] for e in ev] == [1, 2, 3], "positions")
    # unsorted timestamps get sorted; negative deltas are never emitted
    ev2 = trail_events(list(reversed(cell)))
    _check([e["url"] for e in ev2][0] == "https://a.com/x", "sorted by request_time")
    _check(all(d["dwell_seconds"] is None or d["dwell_seconds"] >= 0 for d in ev2), "no negative dwell")


def test_sources_input_file_only():
    """The pipeline must run from simweb_input_file.parquet alone."""
    tmp = tempfile.mkdtemp(prefix="conveyer_simweb_")
    try:
        df = pd.DataFrame({
            "message_id": ["m1", "m2"],
            "a_links_source": [["https://cerave.com/products/cream"], None],
            "ai_click": [None, None],
            "next_10_urls": [
                [{"request_time": "1000000000000", "requested_site": "https://cerave.com/products/cream"},
                 {"request_time": "1000000004000", "requested_site": "https://sephora.com/brand/cerave"}],
                [{"request_time": "2000000000000", "requested_site": "https://cerave.com/products/cream"},
                 {"request_time": "2000000002000", "requested_site": "https://x.com/"}],
            ],
        })
        path = os.path.join(tmp, "simweb_input_file.parquet")
        df.to_parquet(path)

        src = build_sources(ScrapeConfig(clickstream_dir=path))
        _check(src is not None and not src.urls.empty, "sources from single file")
        by_url = src.urls.set_index("url")
        cream = by_url.loc["https://cerave.com/products/cream"]
        # surfaced once via a_links_source (recommended) + twice via trails (visited)
        _check(int(cream["times_surfaced"]) == 3, f"surfaced {cream['times_surfaced']}")
        _check(int(cream["times_recommended"]) == 1, "recommended from a_links_source")
        _check(int(cream["times_visited"]) == 2, "trail entries are visited")
        # dwell: (4000ms + 2000ms)/2 = 3.0s across the two trail appearances
        _check(abs(cream["mean_dwell_seconds"] - 3.0) < 1e-9, f"dwell {cream['mean_dwell_seconds']}")
        _check(abs(cream["total_dwell_seconds"] - 6.0) < 1e-9, "total dwell")
        _check(set(cream["message_ids"]) == {"m1", "m2"}, "linked to both turns")
        # last-of-trail URLs must have no dwell, but still be candidates
        _check(pd.isna(by_url.loc["https://x.com/", "mean_dwell_seconds"]), "no dwell for trail tail")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Fetcher: hangs are impossible by construction
# --------------------------------------------------------------------------- #
def _start_dribble_server():
    """Local server whose page streams forever-ish: 4KB every 0.2s. A socket
    timeout never fires (bytes keep arriving) — only the wall-clock cap can
    stop it. robots.txt 404s instantly."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/robots.txt":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            try:
                for _ in range(300):                    # ~60s of dribble
                    self.wfile.write(b"x" * 4096)
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_fetch_hard_timeout_bounds_slow_hosts():
    srv, base = _start_dribble_server()
    try:
        cfg = ScrapeConfig(offline=False, timeout=2.0, hard_timeout=2.0, max_retries=2,
                           rate_limit_per_domain=0.0, use_cache=False, respect_robots=True)
        f = Fetcher(cfg)
        t0 = time.monotonic()
        res = f.fetch(base + "/slow")
        elapsed = time.monotonic() - t0
        _check(elapsed < 10.0, f"fetch must be bounded by hard_timeout, took {elapsed:.1f}s")
        _check(res.status in ("ok", "error"), f"status {res.status}")
        if res.status == "ok":
            _check("truncated" in res.error, f"expected truncation marker, got {res.error!r}")
    finally:
        srv.shutdown()


def test_throttle_is_per_domain_not_global():
    cfg = ScrapeConfig(offline=False, rate_limit_per_domain=0.6)
    f = Fetcher(cfg)
    f._throttle("a.com")
    t0 = time.monotonic()
    f._throttle("b.com")                     # different domain: no wait
    _check(time.monotonic() - t0 < 0.2, "other domains must not be blocked")
    t1 = time.monotonic()
    f._throttle("a.com")                     # same domain: waits out the interval
    _check(time.monotonic() - t1 >= 0.4, "same domain must be rate-limited")


def test_domain_circuit_breaker():
    """A domain that only ever fails must stop being fetched after the
    threshold — dead hosts can't cost hard_timeout for every URL."""
    # a port nobody listens on -> instant connection refused
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    base = f"http://127.0.0.1:{port}"

    cfg = ScrapeConfig(offline=False, timeout=1.0, hard_timeout=3.0, max_retries=0,
                       rate_limit_per_domain=0.0, use_cache=False,
                       respect_robots=False, domain_failure_threshold=2)
    f = Fetcher(cfg)
    t0 = time.monotonic()
    statuses = [f.fetch(f"{base}/p{i}").status for i in range(5)]
    elapsed = time.monotonic() - t0
    _check(statuses[:2] == ["error", "error"], f"first hits attempted: {statuses}")
    _check(statuses[2:] == ["circuit_open"] * 3, f"circuit must open: {statuses}")
    _check(elapsed < 5.0, f"circuit-open fetches must be instant, took {elapsed:.1f}s")


def test_smart_fetch_policy():
    """URL-decided pages are never fetched; content fetches are capped per
    domain; offline mode is exempt (corpus serving is free)."""
    from conveyer.scraping.pipeline import _fetch_decision

    online = ScrapeConfig(offline=False, fetch_policy="smart", max_fetch_per_domain=2)
    budget = {}
    fetch, why = _fetch_decision("https://www.amazon.com/gp/cart/view.html?ref_=nav_cart",
                                 online, budget)
    _check(not fetch and "URL-decided" in why, f"cart must not be fetched: {why}")
    fetch, _ = _fetch_decision("https://www.google.com/search?q=retinol", online, budget)
    _check(not fetch, "SERPs must not be fetched")
    decisions = [_fetch_decision(f"https://www.byrdie.com/article-{i}", online, budget)[0]
                 for i in range(4)]
    _check(decisions == [True, True, False, False],
           f"per-domain budget of 2 must cap fetches: {decisions}")
    offline = ScrapeConfig(offline=True, fetch_policy="smart", max_fetch_per_domain=1)
    _check(_fetch_decision("https://www.amazon.com/cart", offline, {})[0],
           "offline mode always 'fetches' (the corpus is free)")


def test_domain_profile_reuse():
    """Knowledge learned from fetched pages of a domain classifies its
    unfetched URLs — without copying page-level labels across the domain."""
    cfg = ScrapeConfig()
    url = "https://www.some-indie-brand.io/products/item-2841"
    bare = classify_rule(extract_page("", url), url, cfg)
    _check(bare.page_category == "unknown", f"no evidence -> {bare.page_category}")
    prof = {"relevance": 0.6, "seller": "brand_owned", "n_pages": 3}
    with_prof = classify_rule(extract_page("", url), url, cfg, domain_profile=prof)
    _check(with_prof.page_category == "shopping",
           f"profile relevance must keep the structural role: {with_prof.page_category}")
    _check(with_prof.is_study_relevant, "profile supplies topical evidence")
    _check(with_prof.seller_type == "brand_owned", "profile supplies the seller")
    _check("domain_profile" in with_prof.signals, f"signals {with_prof.signals}")
    # the profile must NOT overwrite what the URL says: a cart stays a cart
    cart = "https://www.some-indie-brand.io/cart"
    c = classify_rule(extract_page("", cart), cart, cfg, domain_profile=prof)
    _check(c.page_subtype == "cart" and c.funnel_stage == "Purchase",
           "page-level structure must never be domain-copied")


def test_base_url_fallback():
    """An unreachable deep link on an unknown domain must fall back to the base
    page (scheme://host/) for content, classify from base content + original
    URL tokens, and record fetch_scope='base'."""
    out = tempfile.mkdtemp(prefix="conveyer_basefb_")
    try:
        art = run_scrape(ScrapeConfig(synthetic_n_pages=30, out_dir=out, progress_every=0))
        pages = art["pages"]
        indie = pages[pages["url"].str.contains("glowessence")]
        _check(len(indie) > 0, "indie base-fallback pages present in corpus")
        _check((indie["fetch_scope"] == "base").all(),
               f"scope should be base: {indie['fetch_scope'].unique()}")
        _check((indie["page_category"] == "shopping").all(),
               f"base content + pdp URL -> shopping: {indie['page_category'].unique()}")
        _check(indie["is_study_relevant"].all(), "base content supplies topical evidence")
        _check(indie["classification_signals"].apply(lambda s: "base_content" in list(s)).all(),
               "base_content signal recorded")
        # products must never be attributed from a stand-in base page
        prods = art["products"]
        _check(not prods["url"].str.contains("glowessence").any(),
               "no products attributed from base pages")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Domain-directory fallback (robots_blocked / bot walls / dead hosts)
# --------------------------------------------------------------------------- #
def test_directory_seed_and_content():
    from conveyer.scraping.directory import directory_content, lookup
    e = lookup("https://www.amazon.co.uk/gp/cart/view.html")
    _check(e is not None and e.role == "marketplace", f"amazon entry: {e}")
    e2 = lookup("sephora.com")
    _check(e2 is not None and "skincare" in e2.description.lower(),
           "beauty retailer description must carry topical words")
    pc = directory_content(e2, "https://www.sephora.com/x")
    _check(bool(pc.text) and pc.parser == "directory", "stand-in content")
    _check(lookup("https://nobody-knows-this-site.xyz/") is None, "unknown -> None")


def test_directory_robots_blocked_fallback():
    """A robots-blocked URL on a known domain classifies from the directory's
    description (fetch_scope='directory'); the description supplies relevance
    but must never mint a category the URL didn't structurally earn."""
    from conveyer.scraping.fetch import FetchResult
    from conveyer.scraping.pipeline import _process_one
    from conveyer.scraping.sources import ScrapeSources

    cfg = ScrapeConfig()                      # offline, empty corpus
    src = ScrapeSources(urls=pd.DataFrame(), mentions={})
    fetcher = Fetcher(cfg, html_by_url={})    # base fallback will offline_miss

    url = "https://www.sephora.com/shop/face-cream"
    fr = FetchResult(url=url, status="robots_blocked")
    pr, prods = _process_one(cfg, src, {"url": url}, fr, fetcher)
    _check(pr["fetch_scope"] == "directory", f"scope {pr['fetch_scope']}")
    _check(pr["page_category"] == "catalogue", f"blocked /shop -> {pr['page_category']}")
    _check(pr["seller_type"] == "retailer", f"seller {pr['seller_type']}")
    _check(pr["is_study_relevant"], "beauty-retailer description supplies relevance")
    _check("directory_content" in list(pr["classification_signals"]),
           f"signals {pr['classification_signals']}")
    _check(prods == [], "no products may come from a directory stand-in")

    # no structural evidence in the URL -> the description alone is not enough
    url2 = "https://www.sephora.com/careers"
    pr2, _ = _process_one(cfg, src, {"url": url2},
                          FetchResult(url=url2, status="robots_blocked"), fetcher)
    _check(pr2["page_category"] == "unknown", f"blocked /careers -> {pr2['page_category']}")

    # blocked cart on a marketplace: transactional URL token stays decisive
    url3 = "https://www.amazon.com/gp/cart/view.html?ref_=nav_cart"
    pr3, _ = _process_one(cfg, src, {"url": url3},
                          FetchResult(url=url3, status="robots_blocked"), fetcher)
    _check(pr3["fetch_scope"] == "directory", f"scope {pr3['fetch_scope']}")
    _check(pr3["page_category"] == "shopping" and pr3["page_subtype"] == "cart",
           f"{pr3['page_category']}/{pr3['page_subtype']}")
    _check(pr3["funnel_stage"] == "Purchase", f"funnel {pr3['funnel_stage']}")


def test_directory_external_file_extends_lists():
    """External JSON entries behave like curated domains: role votes fire (the
    'directory' signal), seller_type resolves, and neutrality is earned."""
    from conveyer.scraping.directory import lookup
    from conveyer.scraping.extract import PageContent

    tmp = tempfile.mkdtemp(prefix="conveyer_dir_")
    try:
        path = os.path.join(tmp, "domain_directory.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"glowmarket.com": {
                "name": "GlowMarket", "role": "retailer",
                "category": "beauty_and_cosmetics",
                "description": "Online beauty retailer selling skincare and cosmetics."}}, fh)
        entry = lookup("https://glowmarket.com/cart", path)
        _check(entry is not None and entry.role == "retailer" and entry.source == "file",
               f"external entry: {entry}")
        cfg = ScrapeConfig(directory_path=path)
        url = "https://glowmarket.com/cart"
        r = classify_rule(PageContent(url=url), url, cfg, directory_entry=entry)
        _check(r.page_category == "shopping" and r.page_subtype == "cart",
               f"{r.page_category}/{r.page_subtype}")
        _check(r.seller_type == "retailer", f"seller {r.seller_type}")
        _check(r.funnel_stage == "Purchase", f"funnel {r.funnel_stage}")
        _check("directory" in r.signals, f"role votes must fire: {r.signals}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Transactional URLs must beat cart-furniture markup (the amazon add-to-cart
# regression: fetched cart page scored pdp 3.5 vs cart 2.5 and collapsed to
# 'unrelated' because a cart shows prices/checkout buttons by nature)
# --------------------------------------------------------------------------- #
AMAZON_CART_HTML = """<!doctype html><html lang="en-us"><head>
<title>Amazon.com Shopping Cart</title></head>
<body><h1>All Carts</h1><p>Your Amazon Cart is empty. Shop today's deals.
Under $10. Beauty &amp; Personal Care. Electronics. Toys &amp; Games.
Proceed to checkout. Check out Amazon Cart.</p>
<a href="/checkout">Proceed to checkout</a><a href="/deals">Shop deals</a>
</body></html>"""
AMAZON_CART_URL = "https://www.amazon.com/cart/add-to-cart/ref=dp_start-bbf_1_glance"


def test_transactional_url_beats_pdp_markup():
    cfg = ScrapeConfig()
    page = extract_page(AMAZON_CART_HTML, AMAZON_CART_URL)
    _check(page.has_price and page.has_add_to_cart,
           "the misleading cart furniture must be present for this test to bite")
    # even with a phantom product counted, the URL's /cart/ path must win
    r = classify_rule(page, AMAZON_CART_URL, cfg, n_products=1)
    _check(r.page_subtype == "cart", f"subtype {r.page_subtype} (want cart)")
    _check(r.page_category == "shopping", f"category {r.page_category}")
    _check(r.funnel_stage == "Purchase", f"funnel {r.funnel_stage}")
    _check(r.is_study_relevant, "a cart on a known retailer is journey infrastructure")
    _check("url_override" in r.signals, f"override must be audited: {r.signals}")
    # a real PDP with only a cart-ish QUERY hint must NOT be overridden
    pdp_url = "https://www.amazon.com/dp/B00365FJ8K?ref_=nav_cart"
    r2 = classify_rule(extract_page("", pdp_url), pdp_url, cfg)
    _check(r2.page_subtype == "pdp", f"query hint must stay advisory: {r2.page_subtype}")


def test_no_phantom_products_on_transactional_pages():
    from conveyer.scraping.classify import is_transactional_url
    _check(is_transactional_url(AMAZON_CART_URL), "cart path is transactional")
    _check(not is_transactional_url("https://www.amazon.com/dp/B00365FJ8K"),
           "pdp is not transactional")
    page = extract_page(AMAZON_CART_HTML, AMAZON_CART_URL)
    with_h = extract_products(page)
    _check(len(with_h) == 1 and with_h[0].source == "heuristic",
           f"the phantom exists without the guard: {[(p.name, p.source) for p in with_h]}")
    _check(extract_products(page, include_heuristic=False) == [],
           "no heuristic phantom on transactional pages")


def test_validate_repairs_unrelated_cart():
    """The URL-rule double check must flag and repair a stored
    unrelated/pdp/Irrelevant row whose URL alone proves shopping/cart/Purchase."""
    from conveyer.scraping.validate import apply_validation, validation_report, write_pages
    out = tempfile.mkdtemp(prefix="conveyer_validate_")
    try:
        art = run_scrape(ScrapeConfig(synthetic_n_pages=20, out_dir=out, progress_every=0))
        pages = art["pages"].copy()
        _check(validation_report(pages).empty, "a fresh run must validate clean")
        # corrupt one cart row the way the old classifier failed
        mask = pages["page_subtype"] == "cart"
        _check(mask.any(), "corpus has a cart page")
        idx = pages.index[mask][0]
        pages.loc[idx, ["page_category", "page_subtype", "funnel_stage",
                        "is_study_relevant"]] = ["unrelated", "pdp", "Irrelevant", False]
        rep = validation_report(pages)
        _check(len(rep) == 1 and rep.iloc[0]["mismatch"] == "category",
               f"corruption must be flagged: {rep.to_dict('records')}")
        fixed, rep2 = apply_validation(pages)
        row = fixed.loc[idx]
        _check(row["page_category"] == "shopping" and row["page_subtype"] == "cart",
               f"repaired to {row['page_category']}/{row['page_subtype']}")
        _check(row["funnel_stage"] == "Purchase" and bool(row["is_study_relevant"]),
               f"funnel {row['funnel_stage']}")
        _check("url_validated" in list(row["classification_signals"]), "repair audited")
        # repaired frame must round-trip through the pinned parquet schema
        path = os.path.join(out, "repaired.parquet")
        write_pages(fixed, path)
        back = pd.read_parquet(path)
        _check(len(back) == len(fixed) and
               back.set_index("page_id").loc[row["page_id"], "page_subtype"] == "cart",
               "parquet round-trip keeps the repair")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Relevance vocabulary: beauty/personal-care umbrella (the jbca.com haircare
# PDP scored skincare_relevance 0.0 under the skincare-only list and collapsed
# to 'unrelated' despite perfect pdp structure)
# --------------------------------------------------------------------------- #
HAIR_HTML = """<!doctype html><html lang="en"><head>
<title>Thickening Strengthening &amp; Restorative Conditioner - JBCA Cosmetics</title>
<meta name="description" content="Shop restorative thickening conditioner formulated to nourish, strengthen, and revive weakened hair.">
<script type="application/ld+json">{"@type":"ProductGroup","name":"Thickening Strengthening & Restorative Conditioner","offers":{"lowPrice":6.99,"highPrice":49.99,"priceCurrency":"USD"}}</script>
</head><body><h1>Thickening Strengthening &amp; Restorative Conditioner</h1>
<p>Shop by Category: Shampoos Conditioners Hair Masks Hair Styling Products.
Shop by Concern: Curl Definition Scalp Care. Fights dandruff, controls frizz, for curly hair.</p>
<p>$6.99 - $49.99. Salon Quality. Sulfate Free. Cruelty Free.</p>
<button>Add to cart</button></body></html>"""

HVAC_HTML = """<!doctype html><html lang="en"><head><title>Arctic 8000 BTU Portable Air Conditioner</title>
<meta name="description" content="This portable air conditioner cools rooms up to 350 sq ft.">
<script type="application/ld+json">{"@type":"Product","name":"Arctic 8000 Portable Air Conditioner","offers":{"price":329,"priceCurrency":"USD"}}</script>
</head><body><h1>Arctic 8000 Portable Air Conditioner</h1>
<p>The air-conditioner ships with a remote. Quiet air conditioner for bedrooms. $329. Add to cart.</p></body></html>"""


def test_haircare_pdp_is_relevant():
    """A haircare PDP on an unknown brand domain must classify shopping/pdp
    and be study-relevant — beauty/personal care is the topical umbrella, not
    skincare alone. An HVAC 'air conditioner' PDP must NOT ride along."""
    cfg = ScrapeConfig()
    url = "https://www.jbca.com/products/thickening-strengthening-restorative-conditioner"
    r = classify_rule(extract_page(HAIR_HTML, url), url, cfg, n_products=1)
    _check(r.page_category == "shopping", f"haircare PDP -> {r.page_category}")
    _check(r.page_subtype == "pdp" and r.funnel_stage == "Intent",
           f"{r.page_subtype}/{r.funnel_stage}")
    _check(r.skincare_relevance >= 0.5, f"relevance {r.skincare_relevance}")
    _check(r.is_study_relevant, "haircare is inside the study umbrella")
    # URL slug alone (unfetchable page) already carries the topic
    bare = classify_rule(extract_page("", url), url, cfg)
    _check(bare.is_study_relevant, f"slug-only relevance {bare.skincare_relevance}")
    # the lookbehind guard: 'air conditioner' / 'air-conditioner' earn nothing
    hvac_url = "https://coolhome.example/products/arctic-8000-portable-air-conditioner"
    h = classify_rule(extract_page(HVAC_HTML, hvac_url), hvac_url, cfg, n_products=1)
    _check(h.page_category == "unrelated", f"HVAC PDP -> {h.page_category}")
    _check(h.skincare_relevance < 0.15, f"HVAC relevance {h.skincare_relevance}")


def test_extra_relevance_terms_extend_vocabulary():
    """ScrapeConfig.extra_relevance_terms widens the topical umbrella per run
    without code edits — plain phrases, whole-word, flexible whitespace."""
    url = "https://groomedman.example/products/cedar-beard-oil"
    html = ('<html><head><title>Cedarwood Beard Oil</title>'
            '<script type="application/ld+json">{"@type":"Product","name":"Cedar Beard Oil",'
            '"offers":{"price":19,"priceCurrency":"USD"}}</script></head>'
            '<body><h1>Cedarwood Beard Oil</h1><p>Softens and tames. $19. Add to cart.</p></body></html>')
    base = classify_rule(extract_page(html, url), url, ScrapeConfig(), n_products=1)
    _check(base.page_category == "unrelated", f"default vocabulary -> {base.page_category}")
    cfg = ScrapeConfig(extra_relevance_terms=("beard oil",))
    ext = classify_rule(extract_page(html, url), url, cfg, n_products=1)
    _check(ext.page_category == "shopping" and ext.is_study_relevant,
           f"extended vocabulary -> {ext.page_category}, rel {ext.skincare_relevance}")


def test_reclassify_rescues_stale_unrelated():
    """After a vocabulary update, reclassify_pages must rescue stored
    'unrelated' rows from the cached HTML their html_path points at (even a
    Windows-style path written on another machine) or from the stored excerpt
    columns — and must leave genuinely off-topic rows untouched."""
    from conveyer.scraping.validate import reclassify_pages, write_pages
    out = tempfile.mkdtemp(prefix="conveyer_reclass_")
    try:
        cache = os.path.join(out, "cache")
        os.makedirs(cache)
        with open(os.path.join(cache, "hair01.json"), "w", encoding="utf-8") as fh:
            json.dump({"http_status": 200, "html": HAIR_HTML}, fh)
        with open(os.path.join(cache, "hvac01.json"), "w", encoding="utf-8") as fh:
            json.dump({"http_status": 200, "html": HVAC_HTML}, fh)
        common = dict(page_category="unrelated", page_subtype="pdp",
                      funnel_stage="Irrelevant", seller_type="na",
                      skincare_relevance=0.0, is_study_relevant=False,
                      classifier_method="rule", fetch_scope="page", n_products=1)
        pages = pd.DataFrame([
            # rescued from cache, via a foreign-OS relative path
            dict(common, page_id="p1",
                 url="https://www.jbca.com/products/thickening-strengthening-restorative-conditioner",
                 html_path="outputs\\scrape_cache\\hair01.json",
                 classification_signals=[["url", "markup"]][0],
                 title="", text_excerpt=""),
            # genuinely off-topic: must NOT be rescued
            dict(common, page_id="p2",
                 url="https://coolhome.example/products/arctic-8000-portable-air-conditioner",
                 html_path="outputs\\scrape_cache\\hvac01.json",
                 classification_signals=[["url", "markup"]][0],
                 title="", text_excerpt=""),
            # no cache file at all: rescued from the stored excerpt columns
            dict(common, page_id="p3",
                 url="https://brandx.example/products/repair-shampoo",
                 html_path="", classification_signals=[["url"]][0],
                 title="Repair Shampoo for damaged hair",
                 text_excerpt="Sulfate free shampoo and conditioner for dry scalp. $12. Add to cart."),
        ])
        cfg = ScrapeConfig(cache_dir=cache)
        fixed, report = reclassify_pages(pages, cfg)
        _check(len(report) == 2, f"rescued rows: {report['page_id'].tolist() if not report.empty else []}")
        by_id = fixed.set_index("page_id")
        _check(by_id.loc["p1", "page_category"] == "shopping"
               and by_id.loc["p1", "funnel_stage"] == "Intent",
               f"p1 -> {by_id.loc['p1', 'page_category']}")
        _check(float(by_id.loc["p1", "skincare_relevance"]) >= 0.5, "p1 relevance updated")
        _check("reclassified" in list(by_id.loc["p1", "classification_signals"]),
               "rescue audited in signals")
        _check("+reclassified" in str(by_id.loc["p1", "classifier_method"]), "method audited")
        _check(by_id.loc["p2", "page_category"] == "unrelated"
               and "reclassified" not in list(by_id.loc["p2", "classification_signals"]),
               "true off-topic row untouched")
        _check(by_id.loc["p3", "page_category"] == "shopping",
               f"excerpt-column fallback -> {by_id.loc['p3', 'page_category']}")
        src = report.set_index("page_id")["content_source"]
        _check(src["p1"] == "cache" and src["p3"] == "row", f"sources {src.to_dict()}")
        # rescued frame must round-trip through the pinned parquet schema
        path = os.path.join(out, "rescued.parquet")
        write_pages(fixed, path)
        back = pd.read_parquet(path)
        _check(back.set_index("page_id").loc["p1", "page_category"] == "shopping",
               "parquet round-trip keeps the rescue")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Service-aware platform routing (the "all google links became search" bug)
# --------------------------------------------------------------------------- #
def test_service_routing_multi_service_platforms():
    """One registrable domain, many products: the service (subdomain / first
    path segment) must decide, and an unrecognized Google surface must NOT
    default to 'search'."""
    from conveyer.scraping import classify_url
    cfg = ScrapeConfig(use_learned_model=False)   # isolate the routing rules
    expect = {
        "https://docs.google.com/document/d/abc/edit": ("unrelated", "tool"),
        "https://drive.google.com/file/d/x/view": ("unrelated", "tool"),
        "https://accounts.google.com/signin": ("unrelated", "account"),
        "https://aws.amazon.com/s3/": ("unrelated", "tool"),
        "https://mail.yahoo.com/d/folders/1": ("unrelated", "tool"),
        "https://maps.google.com/maps/place/Sephora": ("reference", "local"),
        "https://www.google.com/maps/place/Ulta": ("reference", "local"),
        "https://news.google.com/stories/xyz": ("editorial", "article"),
        "https://support.google.com/mail/answer/1": ("reference", "wiki"),
        # marketplace SURFACES are topic-neutral; deep item pages demote to
        # 'listing' so fetched off-topic content can still collapse them
        "https://shopping.google.com/": ("catalogue", "marketplace"),
        "https://shopping.google.com/product/123": ("catalogue", "listing"),
        "https://www.facebook.com/marketplace": ("catalogue", "marketplace"),
        "https://www.facebook.com/marketplace/item/9": ("catalogue", "listing"),
        "https://www.google.com/search?q=best+retinol": ("search", "serp"),
        "https://www.google.com/": ("search", "serp"),
        "https://search.yahoo.com/search?p=serum": ("search", "serp"),
    }
    for url, (cat, sub) in expect.items():
        r = classify_url(url, cfg)
        _check((r.page_category, r.page_subtype) == (cat, sub),
               f"{url} -> {r.page_category}/{r.page_subtype}, expected {cat}/{sub}")
        _check("service" in r.signals, f"service channel must fire for {url}")
    # unrecognized google surface: no false 'search', no invented category
    r = classify_url("https://www.google.com/randomservice/thing", cfg)
    _check(r.page_category != "search", f"unknown google surface -> {r.page_category}")
    # the retailer behaviour of www.amazon.com is untouched (open fallback)
    r = classify_url("https://www.amazon.com/dp/B00X", cfg)
    _check(r.page_category == "shopping" and r.page_subtype == "pdp",
           f"amazon pdp intact: {r.page_category}/{r.page_subtype}")


def test_learned_model_channel():
    """The self-trained model must autotrain, persist, vote as its own channel,
    and stay strictly optional."""
    import conveyer.scraping.model as M
    out = tempfile.mkdtemp(prefix="conveyer_model_")
    try:
        path = os.path.join(out, "page_model.npz")
        cfg = ScrapeConfig(model_path=path, model_autotrain=True)
        url = "https://www.sephora.com/product/cerave-moisturizing-cream"
        r = classify_rule(extract_page("", url), url, cfg)
        _check(os.path.exists(path), "autotrain must persist the model file")
        _check("model" in r.signals, f"model channel must fire: {r.signals}")
        # numpy-only inference round-trip
        m = M.PageModel.load(path)
        probs = m.predict_proba_one(M.featurize(url))
        _check(abs(sum(probs.values()) - 1.0) < 1e-4, "probabilities sum to 1")
        _check(max(probs, key=probs.get) == "pdp", f"top class {max(probs, key=probs.get)}")
        # the model knows the distilled service table standalone
        probs = m.predict_proba_one(M.featurize("https://docs.google.com/document/d/x"))
        _check(max(probs, key=probs.get) == "tool", f"docs.google -> {max(probs, key=probs.get)}")
        # disabled -> channel silent
        off = ScrapeConfig(model_path=path, use_learned_model=False)
        r2 = classify_rule(extract_page("", url), url, off)
        _check("model" not in r2.signals, "disabled model must not vote")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Block-resilient fetching: salvage headers/body, fingerprint the platform
# --------------------------------------------------------------------------- #
def _start_block_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    hits = {"n": 0, "agents": [], "paths": []}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["paths"].append(self.path)
            hits["agents"].append(self.headers.get("User-Agent", ""))
            if self.path == "/robots.txt":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"User-agent: *\nAllow: /\n")
                return
            hits["n"] += 1
            self.send_response(403)
            self.send_header("Content-Type", "text/html")
            self.send_header("X-Shopify-Stage", "production")
            self.send_header("Server", "cloudflare")
            self.end_headers()
            self.wfile.write(b"<html><body>Access denied</body></html>")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", hits


def test_blocked_fetch_salvages_headers_and_fingerprint():
    """A 403 is a wall, not a failure: one attempt (no hammering), headers +
    bounded body salvaged, platform fingerprinted from them."""
    from conveyer.scraping.fingerprint import detect_platform, platform_seller
    srv, base, hits = _start_block_server()
    try:
        cfg = ScrapeConfig(offline=False, timeout=3.0, hard_timeout=6.0, max_retries=2,
                           rate_limit_per_domain=0.0, use_cache=False)
        f = Fetcher(cfg)
        res = f.fetch(base + "/products/thing")
        _check(res.status == "blocked", f"403 -> blocked, got {res.status}")
        _check(res.http_status == 403, f"http_status {res.http_status}")
        _check(hits["n"] == 1, f"403 must not be retried, got {hits['n']} hits")
        _check(res.headers.get("x-shopify-stage") == "production",
               f"headers salvaged: {res.headers}")
        _check("denied" in res.error_body.lower(), "error body salvaged (bounded)")
        _check(not res.html, "error body must never masquerade as page content")
        plat = detect_platform(res.headers, res.error_body)
        _check(plat == "shopify", f"commerce platform wins over the WAF: {plat}")
        _check(platform_seller(plat) == "brand_owned", "shopify ⇒ DTC seller hint")
        # robots.txt was still consulted before the page (politeness unchanged)
        _check(hits["paths"][0] == "/robots.txt", f"robots first: {hits['paths']}")
        # classifier: bot-walled Shopify store + /products/ URL = shopping page
        cls = classify_rule(extract_page("", base + "/products/thing"),
                            base + "/products/thing",
                            ScrapeConfig(use_learned_model=False),
                            content_scope="none", platform=plat)
        _check(cls.page_category == "shopping" and cls.page_subtype == "pdp",
               f"blocked shopify pdp -> {cls.page_category}/{cls.page_subtype}")
        _check(cls.seller_type == "brand_owned", f"seller {cls.seller_type}")
        _check("platform" in cls.signals, f"signals {cls.signals}")
    finally:
        srv.shutdown()


def test_browser_headers_and_robots_compliance():
    """browser_headers=True presents a realistic Chrome profile (naive UA
    blocks don't fire); False keeps the descriptive bot UA. robots.txt is
    honored in both modes."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = {"agents": [], "paths": []}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["paths"].append(self.path)
            seen["agents"].append(self.headers.get("User-Agent", ""))
            body = b"<html><head><title>ok</title></head><body>fine</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        f = Fetcher(ScrapeConfig(offline=False, rate_limit_per_domain=0.0, use_cache=False))
        res = f.fetch(base + "/page")
        _check(res.status == "ok", f"fetch ok: {res.status} {res.error}")
        page_agents = [a for p, a in zip(seen["paths"], seen["agents"]) if p == "/page"]
        _check("Chrome" in page_agents[0], f"browser profile sent: {page_agents[0][:60]}")
        _check("/robots.txt" in seen["paths"], "robots.txt still consulted")
        _check(res.headers.get("content-type", "").startswith("text/html"),
               "response headers captured on OK fetches too")
        seen["agents"].clear(); seen["paths"].clear()
        f2 = Fetcher(ScrapeConfig(offline=False, browser_headers=False,
                                  rate_limit_per_domain=0.0, use_cache=False))
        f2.fetch(base + "/page2")
        page_agents = [a for p, a in zip(seen["paths"], seen["agents"]) if p == "/page2"]
        _check("conveyer-research-bot" in page_agents[0],
               f"bot UA when browser_headers=False: {page_agents[0][:60]}")
    finally:
        srv.shutdown()


def test_query_strip_fallback():
    """A stale-query link (?preview_id=…&preview_nonce=…) that errors must be
    retried WITHOUT its query; the stripped variant's content counts as the
    page's own (fetch_scope='stripped'). Identity-bearing queries (?q=,
    ?variant=…) must never be stripped — that would fetch a different page."""
    from conveyer.scraping.fetch import query_stripped_url
    from conveyer.scraping.pipeline import _process_one
    from conveyer.scraping.sources import ScrapeSources

    full = ("https://www.zicail.com/how-to-choose-the-best-sunscreen/"
            "?preview_id=10472&preview_nonce=fd5902c163")
    bare = "https://www.zicail.com/how-to-choose-the-best-sunscreen/"
    _check(query_stripped_url(full) == bare, f"strip -> {query_stripped_url(full)}")
    _check(query_stripped_url(bare) == "", "nothing to strip")
    _check(query_stripped_url("https://www.google.com/search?q=retinol") == "",
           "?q= selects the content — never stripped")
    _check(query_stripped_url("https://shop.example.com/p/x?variant=123") == "",
           "?variant= selects the content")
    _check(query_stripped_url("https://example.com/?utm_source=chat") == "",
           "root stripping is the base fallback's job")
    _check(query_stripped_url("https://www.byrdie.com/best-spf/?utm_source=chatgpt.com")
           == "https://www.byrdie.com/best-spf/", "tracking params strip fine")

    article = ('<html lang="en"><head><title>How to Choose the Best Sunscreen</title>'
               '<meta name="description" content="SPF, broad spectrum and skin type — '
               'how to pick a sunscreen."><meta property="og:type" content="article">'
               '<script type="application/ld+json">{"@type":"Article",'
               '"headline":"How to Choose the Best Sunscreen"}</script></head>'
               '<body><h1>How to Choose the Best Sunscreen</h1>'
               '<p>Dermatologists recommend SPF 30+ broad spectrum sunscreen, '
               'reapplied every two hours. Mineral vs chemical sunscreen.</p></body></html>')
    cfg2 = ScrapeConfig(offline=True, use_learned_model=False, use_cache=False)
    # the corpus knows ONLY the bare URL — the full link is an offline_miss,
    # exactly like the live 'preview nonce expired' error page
    fetcher = Fetcher(cfg2, html_by_url={bare: article})
    src = ScrapeSources(urls=pd.DataFrame([{"url": full}]), mentions={})
    fr = fetcher.fetch(full)
    _check(not fr.ok, f"the full link must fail first: {fr.status}")
    pr, prods = _process_one(cfg2, src, {"url": full, "message_ids": []}, fr, fetcher)
    _check(pr["fetch_scope"] == "stripped", f"scope {pr['fetch_scope']}")
    _check(pr["page_category"] == "editorial" and pr["page_subtype"] == "article",
           f"{pr['page_category']}/{pr['page_subtype']}")
    _check(pr["is_study_relevant"], "sunscreen article is on-topic")
    _check("query_stripped" in list(pr["classification_signals"]),
           f"audited: {pr['classification_signals']}")
    _check(pr["title"].startswith("How to Choose"), "content came from the stripped URL")
    # with the fallback disabled the same link stays content-less
    cfg3 = ScrapeConfig(offline=True, use_learned_model=False, use_cache=False,
                        query_strip_fallback=False, base_fallback=False,
                        directory_fallback=False)
    f3 = Fetcher(cfg3, html_by_url={bare: article})
    pr3, _ = _process_one(cfg3, src, {"url": full, "message_ids": []}, f3.fetch(full), f3)
    _check(pr3["fetch_scope"] == "none", f"knob off -> {pr3['fetch_scope']}")

    # THE WORDPRESS CASE: the preview link does not fail at the HTTP level —
    # it returns 200 with the ERROR PAGE as the body (or redirects to
    # wp-login). The soft-error detector must still trigger the strip.
    err_page = ('<html lang="en"><head><title>Page not found - Zicail</title></head>'
                '<body><h1>Sorry, you are not allowed to preview drafts.</h1>'
                '<p>Please log in and try again.</p></body></html>')
    f4 = Fetcher(cfg2, html_by_url={full: err_page, bare: article})
    fr4 = f4.fetch(full)
    _check(fr4.ok, "the soft-error fetch LOOKS successful")
    pr4, _ = _process_one(cfg2, src, {"url": full, "message_ids": []}, fr4, f4)
    _check(pr4["fetch_scope"] == "stripped" and pr4["page_category"] == "editorial",
           f"soft error rescued: {pr4['fetch_scope']}/{pr4['page_category']}")
    _check(pr4["title"].startswith("How to Choose"), "error shell replaced by the article")
    # a legit page with a strippable query is never second-guessed
    f5 = Fetcher(cfg2, html_by_url={full: article, bare: err_page})
    pr5, _ = _process_one(cfg2, src, {"url": full, "message_ids": []}, f5.fetch(full), f5)
    _check(pr5["fetch_scope"] == "page", f"healthy page untouched: {pr5['fetch_scope']}")
    # anti-bot challenge shells (202 + a few bytes) on BOTH variants: the
    # shell must never be classified as the page's content — it is dropped
    # and the URL slug still carries the topical/structural verdict
    shell = "<html><head><script>challenge()</script></head><body></body></html>"
    f6 = Fetcher(cfg2, html_by_url={full: shell, bare: shell})
    pr6, _ = _process_one(cfg2, src, {"url": full, "message_ids": []}, f6.fetch(full), f6)
    _check(pr6["fetch_scope"] not in ("page", "stripped"),
           f"challenge shell is not content: {pr6['fetch_scope']}")
    _check(pr6["page_category"] != "unrelated",
           f"shell must not prove off-topic: {pr6['page_category']}")


def test_fingerprint_detection():
    from conveyer.scraping.fingerprint import (detect_platform,
                                               is_commerce_platform)
    _check(detect_platform({"x-shopify-stage": "production"}) == "shopify", "shopify header")
    _check(detect_platform({"cf-ray": "abc"}, "") == "cloudflare", "waf-only -> infra")
    _check(not is_commerce_platform("cloudflare"), "infra is not commerce evidence")
    woo = '<html><head><meta name="generator" content="WooCommerce 8.1"></head></html>'
    _check(detect_platform({}, woo) == "woocommerce", "woocommerce markup")
    _check(detect_platform({}, "<html>plain</html>") == "", "no false positives")


# --------------------------------------------------------------------------- #
# Precision-first product ↔ chat matching
# --------------------------------------------------------------------------- #
def test_matching_precision_tiers():
    """Brand alone is not the product; conflicts veto; typos and long-tail
    brands still connect."""
    mention = RecMention(recommendation_id="r1", message_id="m1",
                         entity_context="CeraVe Moisturizing Cream — a moisturizer "
                                        "the assistant recommended",
                         brands={"cerave"}, categories={"moisturizer"})
    same_brand_other = ProductRecord(name="CeraVe Ultra Repair Lip Balm",
                                     brand="CeraVe", price=9.99)
    typo = ProductRecord(name="CereVe Moisturising Cream", brand="", price=16.0)
    rival = ProductRecord(name="Cetaphil Moisturizing Cream", brand="Cetaphil", price=13.0)
    match_products([same_brand_other, typo, rival], page_brand="",
                   mentions=[mention], message_id="m1")
    _check(not same_brand_other.coincides and same_brand_other.match_strength == "likely",
           f"same brand ≠ same product: {same_brand_other.match_strength}")
    _check(typo.coincides and "name_ngram" in typo.match_signals,
           f"typo variant connects via char-ngrams: {typo.match_strength} {typo.match_signals}")
    _check(not rival.coincides, "brand conflict vetoes however similar the name")
    # SPF numbers are identity: SPF 30 is not the SPF 60 that was recommended
    spf_mention = RecMention(recommendation_id="r2", message_id="m2",
                             entity_context="La Roche-Posay Anthelios SPF 60 Sunscreen",
                             brands={"la roche-posay"}, categories={"sunscreen"})
    spf30 = ProductRecord(name="La Roche-Posay Anthelios SPF 30 Sunscreen",
                          brand="La Roche-Posay", price=33.0)
    spf60 = ProductRecord(name="La Roche-Posay Anthelios Sunscreen SPF 60 5 oz",
                          brand="La Roche-Posay", price=35.99)
    match_products([spf30, spf60], page_brand="", mentions=[spf_mention], message_id="m2")
    _check(not spf30.coincides and "attr_conflict" in spf30.match_signals,
           f"SPF conflict caps the verdict: {spf30.match_strength} {spf30.match_signals}")
    _check(spf60.coincides, f"SPF agreement + brand + name: {spf60.match_strength}")
    # long-tail brand nobody's lexicon knows, named in the entity text (jbca case)
    jbca_mention = RecMention(recommendation_id="r3", message_id="m3",
                              entity_context="the JBCA thickening strengthening "
                                             "conditioner for weak hair",
                              brands=set(), categories={"conditioner"})
    jbca = ProductRecord(name="Thickening Strengthening & Restorative Conditioner",
                         brand="JBCA Cosmetics", price=6.99)
    match_products([jbca], page_brand="", mentions=[jbca_mention], message_id="m3")
    _check(jbca.coincides and "brand_in_entity" in jbca.match_signals,
           f"unknown brand connects via entity text: "
           f"{jbca.match_strength} {jbca.match_signals}")


def test_page_chat_match_rollup():
    """Pages carry the connection to the chat at the page grain."""
    out = tempfile.mkdtemp(prefix="conveyer_rollup_")
    try:
        art = run_scrape(ScrapeConfig(synthetic_n_pages=26, out_dir=out, progress_every=0))
        pages = art["pages"]
        _check("chat_match_strength" in pages.columns, "rollup column present")
        pdp = pages[(pages["page_subtype"] == "pdp") & (pages["page_category"] == "shopping")
                    & pages["url"].str.contains("moisturizing-cream")]
        _check((pdp["chat_match_strength"] == "exact").any(),
               f"recommended PDP rolls up exact: {pdp['chat_match_strength'].unique()}")
        lip = pages[pages["url"].str.contains("lip-balm")]
        _check(len(lip) > 0 and (lip["chat_match_strength"] != "exact").all()
               and (~lip["chat_match_strength"].isin(["strong"])).all(),
               f"hard negative stays below coincide: {lip['chat_match_strength'].unique()}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Adversarial-review regressions (each of these reproduced a real defect)
# --------------------------------------------------------------------------- #
def test_platform_fingerprint_limits():
    """The platform fingerprint is structural evidence only: it must not make
    off-topic stores study-relevant, and it must not outvote a curated
    retailer's role."""
    cfg = ScrapeConfig(use_learned_model=False)
    # fetched OFF-TOPIC Shopify store homepage: still collapses to unrelated
    html = ('<html lang="en"><head><title>Apex Gaming Gear | Official Site</title>'
            '<meta property="og:type" content="website">'
            '<script type="application/ld+json">{"@type":"Organization","name":"Apex"}</script>'
            '</head><body><h1>Apex Gaming Gear</h1>'
            '<p>Mechanical keyboards and mice. Free shipping.</p></body></html>')
    url = "https://apexgaminggear.example/"
    r = classify_rule(extract_page(html, url), url, cfg, platform="shopify")
    _check(r.page_category == "unrelated" and not r.is_study_relevant,
           f"off-topic shopify homepage -> {r.page_category}/{r.is_study_relevant}")
    # a curated retailer that happens to run Shopify keeps its retailer role
    about = "https://www.credobeauty.com/pages/about-us"
    html2 = ('<html lang="en"><head><title>About Us | Credo Beauty</title></head>'
             '<body><h1>About Credo</h1><p>Clean beauty skincare retailer.</p></body></html>')
    r2 = classify_rule(extract_page(html2, about), about, cfg, platform="shopify")
    _check(r2.seller_type != "brand_owned",
           f"curated retailer must not flip to brand_owned: {r2.seller_type}")
    # fetched off-topic marketplace ITEM page collapses (deep ≠ neutral surface)
    fb = "https://www.facebook.com/marketplace/item/123456"
    html3 = ('<html lang="en"><head><title>Used mountain bike - $250</title></head>'
             '<body><h1>Trek Marlin 5 mountain bike</h1><p>Good condition. $250.</p></body></html>')
    r3 = classify_rule(extract_page(html3, fb), fb, cfg)
    _check(r3.page_category == "unrelated",
           f"off-topic marketplace item -> {r3.page_category}")
    # locale subdomains of search engines are still the search engine
    r4 = classify_rule(extract_page("", "https://cn.bing.com/"), "https://cn.bing.com/", cfg)
    _check(r4.page_category == "search", f"cn.bing.com -> {r4.page_category}")
    # invariant: final 'unrelated' (e.g. a docs.google.com doc ABOUT skincare)
    # is never study-relevant
    doc = "https://docs.google.com/document/d/abc/edit"
    html5 = ('<html lang="en"><head><title>My skincare routine</title></head>'
             '<body><p>retinol serum moisturizer sunscreen niacinamide</p></body></html>')
    r5 = classify_rule(extract_page(html5, doc), doc, cfg)
    _check(r5.page_category == "unrelated" and not r5.is_study_relevant,
           f"tool page invariant: {r5.page_category}/{r5.is_study_relevant}")
    # model_weight=0 must not mint earned roles out of thin air (0 >= 0 bug)
    zero = ScrapeConfig(use_learned_model=False, model_weight=0.0)
    myst = "https://someblog-nobody-knows.xyz/interesting"
    r6 = classify_rule(extract_page("", myst), myst, zero)
    _check(r6.page_category == "unknown", f"model_weight=0 -> {r6.page_category}")


def test_matching_sibling_and_synonym_regressions():
    """Same-brand siblings sharing only generic words must not coincide;
    form synonyms (wash/cleanser) of the SAME product must."""
    mention = RecMention(recommendation_id="r1", message_id="m1",
                         entity_context="CeraVe Moisturizing Cream — a rich "
                                        "moisturizer the assistant recommended",
                         brands={"cerave"}, categories={"moisturizer"})
    sibling = ProductRecord(name="CeraVe Foaming Facial Cleanser", brand="CeraVe",
                            category="cleanser", price=15.0)
    eye = ProductRecord(name="CeraVe Eye Repair Cream", brand="CeraVe",
                        category="moisturizer", price=14.0)
    generic = ProductRecord(name="Moisturizer", brand="", price=9.0)
    match_products([sibling, eye, generic], page_brand="cerave",
                   mentions=[mention], message_id="m1")
    _check(not sibling.coincides, f"sibling cleanser: {sibling.match_strength}")
    _check(not eye.coincides,
           f"same brand+form, different product: {eye.match_strength}")
    _check(not generic.coincides and generic.match_strength != "strong",
           f"1-token generic name: {generic.match_strength}")
    # wash vs cleanser are the same form — a real match must survive
    wash_mention = RecMention(recommendation_id="r2", message_id="m2",
                              entity_context="the CeraVe foaming face wash for oily skin",
                              brands={"cerave"}, categories={"cleanser"})
    cleanser = ProductRecord(name="CeraVe Foaming Facial Cleanser", brand="CeraVe",
                             category="cleanser", price=15.0)
    match_products([cleanser], page_brand="", mentions=[wash_mention], message_id="m2")
    _check(cleanser.coincides,
           f"wash≡cleanser synonym must match: {cleanser.match_strength} {cleanser.match_signals}")
    # a common-adjective brand must not 'agree' via ordinary prose when the
    # mention explicitly names a different brand
    simple = ProductRecord(name="Kind To Skin Facial Wash", brand="Simple", price=6.0)
    adj_mention = RecMention(recommendation_id="r3", message_id="m3",
                             entity_context="a simple, fragrance-free cleanser "
                                            "like CeraVe's foaming wash",
                             brands={"cerave"}, categories={"cleanser"})
    match_products([simple], page_brand="", mentions=[adj_mention], message_id="m3")
    _check(not simple.coincides and "brand_in_entity" not in simple.match_signals,
           f"adjective ≠ brand agreement: {simple.match_signals}")


def test_retry_after_and_model_resilience():
    from conveyer.scraping.fetch import _parse_retry_after
    from conveyer.scraping.model import PageModel, _FAILED_PATHS, _model_for
    _check(_parse_retry_after("120") == 120.0, "delta-seconds form")
    _check(_parse_retry_after("") is None and _parse_retry_after("soon") is None,
           "unparseable -> None (falls back to backoff retries)")
    ra = _parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT")
    _check(ra is not None and ra >= 0, f"HTTP-date form parses: {ra}")
    # a corrupt model file must not silently kill the channel: retrain over it
    out = tempfile.mkdtemp(prefix="conveyer_modelcorrupt_")
    try:
        path = os.path.join(out, "model.npz")
        with open(path, "wb") as fh:
            fh.write(b"not an npz at all")
        cfg = ScrapeConfig(model_path=path, model_autotrain=True)
        m = _model_for(cfg)
        _check(m is not None, "corrupt file retrained")
        _check(PageModel.load(path) is not None, "replacement file is valid")
        _check(path not in _FAILED_PATHS, "path not marked failed after recovery")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Incremental persistence & resume
# --------------------------------------------------------------------------- #
def test_incremental_jsonl_and_resume():
    out = tempfile.mkdtemp(prefix="conveyer_scrape_out_")
    try:
        cfg = ScrapeConfig(synthetic_n_pages=12, out_dir=out, progress_every=0,
                           checkpoint_every=5)
        art1 = run_scrape(cfg)
        _check(art1["n_new"] == 12, f"first run n_new {art1['n_new']}")
        lines = open(cfg.pages_jsonl_path()).read().strip().splitlines()
        _check(len(lines) == 12, f"one JSONL line per page, got {len(lines)}")
        _check(all(json.loads(l)["page_id"] for l in lines), "lines are valid JSON")

        # simulate a crash that lost the last 7 pages
        with open(cfg.pages_jsonl_path(), "w") as fh:
            fh.write("\n".join(lines[:5]) + "\n" + '{"torn line')  # torn tail too
        art2 = run_scrape(cfg)
        _check(art2["n_new"] == 7, f"resume must only redo missing pages, n_new {art2['n_new']}")
        pages = art2["pages"]
        _check(len(pages) == 12 and pages["page_id"].nunique() == 12,
               f"final parquet complete & deduped: {len(pages)}")
        _check(art2["evaluation"]["category_accuracy"] == 1.0, "eval intact after resume")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_checkpoint_parts_bounded_memory():
    """Checkpoints must flush buffered rows to parquet part files (freeing
    memory) and the final parquet must be streamed from those parts. A
    pre-parts-format output (JSONL only) must migrate automatically."""
    from conveyer.scraping.schema import part_files
    out = tempfile.mkdtemp(prefix="conveyer_parts_")
    try:
        cfg = ScrapeConfig(synthetic_n_pages=12, out_dir=out, progress_every=0,
                           checkpoint_every=5)
        art = run_scrape(cfg)
        parts = part_files(cfg.pages_parts_dir())
        _check(len(parts) == 3, f"12 pages / checkpoint 5 -> 3 parts, got {len(parts)}")
        _check(len(art["pages"]) == 12, "final parquet complete")
        _check(art["pages"]["page_id"].nunique() == 12, "no duplicates")

        # rerun: nothing pending, nothing duplicated, parts untouched
        art2 = run_scrape(cfg)
        _check(art2["n_new"] == 0, f"nothing to redo, n_new {art2['n_new']}")
        _check(len(art2["pages"]) == 12 and art2["pages"]["page_id"].nunique() == 12,
               "rerun keeps the table intact")
        _check(len(part_files(cfg.pages_parts_dir())) == 3, "no empty/dup parts appended")

        # old-format migration: JSONL exists but parts are gone
        shutil.rmtree(cfg.pages_parts_dir())
        shutil.rmtree(cfg.products_parts_dir(), ignore_errors=True)
        art3 = run_scrape(cfg)
        _check(art3["n_new"] == 0, "JSONL alone is enough to resume")
        _check(len(art3["pages"]) == 12 and art3["pages"]["page_id"].nunique() == 12,
               f"parts rebuilt from the JSONL commit log: {len(art3['pages'])}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_offline_cache_is_lazy_and_html_path_recorded():
    """Offline mode must serve previously fetched pages from the per-URL disk
    cache (never preloading a corpus), and the page row must record where the
    raw HTML lives (html_path)."""
    from conveyer.scraping.fetch import FetchResult
    from conveyer.scraping.fetch import _cache_path as cache_path_for
    from conveyer.scraping.pipeline import _process_one
    from conveyer.scraping.sources import ScrapeSources

    tmp = tempfile.mkdtemp(prefix="conveyer_htmlcache_")
    try:
        cfg = ScrapeConfig(cache_dir=os.path.join(tmp, "cache"))
        url = "https://cerave.com/products/moisturizing-cream"
        # simulate a previous online run having cached the page
        writer = Fetcher(cfg)
        writer._write_cache(FetchResult(url=url, status="ok", http_status=200,
                                        final_url=url, content_type="text/html",
                                        html=PROD_HTML, fetched_at="t"))
        # a fresh offline fetcher with NO corpus must find it on disk, lazily
        reader = Fetcher(cfg, html_by_url={})
        fr = reader.fetch(url)
        _check(fr.ok and fr.from_cache, f"lazy disk-cache hit: {fr.status}")

        src = ScrapeSources(urls=pd.DataFrame(), mentions={})
        pr, prods = _process_one(cfg, src, {"url": url}, fr, reader)
        _check(pr["html_path"] == cache_path_for(cfg, url),
               f"html_path must point at the cache file: {pr['html_path']}")
        _check(os.path.exists(pr["html_path"]), "the file actually exists")
        _check(pr["page_category"] == "shopping" and len(prods) >= 1,
               "cached page classifies and yields products as if fetched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Schema & pipeline
# --------------------------------------------------------------------------- #
def test_schema_arrow_types():
    corpus = make_corpus(9, seed=1)
    _check(not corpus.urls.empty, "corpus urls")
    _check("mean_dwell_seconds" in corpus.urls.columns, "corpus carries dwell")
    t = to_arrow_table([], PAGE_SCHEMA)
    _check(t.num_rows == 0 and t.num_columns == len(PAGE_SCHEMA), "empty page table")
    tp = to_arrow_table([], PRODUCT_SCHEMA)
    _check(tp.num_columns == len(PRODUCT_SCHEMA), "empty product table")


def test_pipeline_end_to_end():
    out = tempfile.mkdtemp(prefix="conveyer_scrape_e2e_")
    try:
        art = run_scrape(ScrapeConfig(synthetic_n_pages=40, out_dir=out, progress_every=0))
        ev = art["evaluation"]
        _check(ev["category_accuracy"] >= 0.9, f"category acc {ev['category_accuracy']}")
        _check(ev["coincide_precision"] >= 0.8, f"coincide precision {ev['coincide_precision']}")
        _check(len(art["products"]) > 0, "products extracted")
        _check(art["pages"]["page_category"].nunique() >= 5, "category diversity")
        _check(art["pages"]["mean_dwell_seconds"].notna().any(), "dwell present in pages parquet")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
