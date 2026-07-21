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
    _check(cerave.coincides and cerave.match_type == "brand", "brand match")
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
    # a page we never fetched must NOT be confidently "unrelated" — but a
    # skincare-slugged URL is on-topic evidence even with zero HTML
    empty = classify_rule(extract_page("", url), url, cfg)
    _check(empty.page_category == "unknown", f"no content -> {empty.page_category}")
    cera_url = "https://www.sephora.com/product/cerave-moisturizing-cream"
    cera = classify_rule(extract_page("", cera_url), cera_url, cfg)
    _check(cera.page_category == "shopping", f"URL-slug evidence -> {cera.page_category}")


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
