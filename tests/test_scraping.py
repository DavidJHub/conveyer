"""Tests for conveyer.scraping.

Runs under pytest *or* standalone (``python tests/test_scraping.py``) so it works
in this repo, which has no pytest harness. Everything is offline and
dependency-light (standard library + pandas/pyarrow).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from conveyer.scraping import ScrapeConfig, run_scrape
from conveyer.scraping.classify import classify_rule, parse_url
from conveyer.scraping.extract import extract_page
from conveyer.scraping.products import (ProductRecord, RecMention, extract_products,
                                        match_products)
from conveyer.scraping.schema import PAGE_SCHEMA, PRODUCT_SCHEMA, to_arrow_table
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


def test_matching_positive_and_negative():
    cerave = ProductRecord(name="CeraVe Moisturizing Cream", brand="CeraVe", price=16.99)
    laptop = ProductRecord(name="UltraBook Pro Laptop", brand="TechCo", price=1499.0)
    brandless = ProductRecord(name="Some Cream", brand="", price=9.0)
    mention = RecMention(recommendation_id="r1", message_id="m1",
                         entity_context="CeraVe Moisturizing Cream", brands={"cerave"},
                         categories={"moisturizer"})
    # page_brand must NOT be a chat brand (would create false positives)
    match_products([cerave, laptop, brandless], page_brand="", mentions=[mention],
                   message_id="m1", coincide_threshold=0.5)
    _check(cerave.coincides and cerave.match_type == "brand", "brand match")
    _check(not laptop.coincides, "laptop must not coincide")
    _check(not brandless.coincides, "brandless product must not inherit chat brand")


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


def test_schema_arrow_types():
    corpus = make_corpus(9, seed=1)
    _check(not corpus.urls.empty, "corpus urls")
    # empty tables must still build with the right schema
    t = to_arrow_table([], PAGE_SCHEMA)
    _check(t.num_rows == 0 and t.num_columns == len(PAGE_SCHEMA), "empty page table")
    tp = to_arrow_table([], PRODUCT_SCHEMA)
    _check(tp.num_columns == len(PRODUCT_SCHEMA), "empty product table")


def test_pipeline_end_to_end():
    art = run_scrape(ScrapeConfig(synthetic_n_pages=40, out_dir="outputs/_test_scrape"))
    ev = art["evaluation"]
    _check(ev["category_accuracy"] >= 0.9, f"category acc {ev['category_accuracy']}")
    _check(ev["coincide_precision"] >= 0.8, f"coincide precision {ev['coincide_precision']}")
    _check(len(art["products"]) > 0, "products extracted")
    _check(art["pages"]["page_category"].nunique() >= 5, "category diversity")


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
