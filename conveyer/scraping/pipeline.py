"""End-to-end scrape → extract → classify → match → parquet.

Run as a module::

    python -m conveyer.scraping.pipeline                 # synthetic corpus, offline
    python -m conveyer.scraping.pipeline --clickstream-dir data/similarweb_clickstream_data --online

or programmatically::

    from conveyer.scraping import ScrapeConfig, run_scrape
    art = run_scrape(ScrapeConfig())          # offline synthetic by default
    art["pages"].head()                       # fact_scraped_page
    art["products"][art["products"].coincides].head()

Everything is safe and offline by default: with no data and no network it runs on
the synthetic corpus, classifies, matches products to the (synthetic) chat
recommendation and writes both parquet files, then scores itself against the
corpus ground truth.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from .classify import classify_page
from .config import ScrapeConfig
from .extract import PageContent, extract_page
from .fetch import Fetcher, FetchResult, _cache_path
from .products import extract_products, match_products
from .schema import PAGE_SCHEMA, PRODUCT_SCHEMA, build_frames, page_row, product_rows, write_parquet
from .sources import ScrapeSources, build_sources, chat_brands_for, mentions_for
from .synthetic import make_corpus


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
def _load_cache_corpus(cfg: ScrapeConfig, urls: List[str]) -> Dict[str, str]:
    import json
    corpus: Dict[str, str] = {}
    for url in urls:
        path = _cache_path(cfg, url)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    corpus[url] = json.load(fh).get("html", "")
            except Exception:
                continue
    return corpus


def _resolve_sources(cfg: ScrapeConfig, sources: Optional[ScrapeSources] = None) -> ScrapeSources:
    if sources is not None:
        return sources
    if os.path.isdir(cfg.clickstream_dir):
        real = build_sources(cfg)
        if real is not None and not real.urls.empty:
            if cfg.offline and real.html_by_url is None:
                real.html_by_url = _load_cache_corpus(cfg, real.urls["url"].tolist())
            return real
    if cfg.use_synthetic_if_missing:
        return make_corpus(cfg.synthetic_n_pages, cfg.synthetic_seed)
    raise FileNotFoundError(
        f"No clickstream data at {cfg.clickstream_dir} and use_synthetic_if_missing=False")


# --------------------------------------------------------------------------- #
# Core run
# --------------------------------------------------------------------------- #
def _prior_for(cfg: ScrapeConfig, row: dict) -> dict:
    return {
        cfg.col_page_type: row.get("page_type"),
        cfg.col_seller_type: row.get("seller_type"),
        cfg.col_retailer_brand: row.get("retailer_brand"),
        cfg.col_site_category: row.get("site_category"),
    }


def run_scrape(cfg: ScrapeConfig, sources: Optional[ScrapeSources] = None) -> Dict[str, object]:
    src = _resolve_sources(cfg, sources)
    urls_df = src.urls
    print(f"[sources] {src.source or cfg.clickstream_dir} | urls={len(urls_df)} | "
          f"turns_with_mentions={len(src.mentions)}")

    fetcher = Fetcher(cfg, html_by_url=src.html_by_url)
    url_list = urls_df["url"].tolist()
    results: List[FetchResult] = fetcher.fetch_many(url_list)
    ok = sum(r.ok for r in results)
    print(f"[fetch] mode={'offline' if cfg.offline else 'online'} | "
          f"ok={ok}/{len(results)}")

    page_rows: List[dict] = []
    product_rows_all: List[dict] = []
    for (_, row), fr in zip(urls_df.iterrows(), results):
        row_d = row.to_dict()
        url = row_d.get("url") or fr.url
        content = extract_page(fr.html, url=url, parser=cfg.html_parser) if fr.ok \
            else PageContent(url=url)
        products = extract_products(content, cfg.max_products_per_page) if fr.ok else []
        chat_brands = chat_brands_for(row, src.mentions)
        mentions = mentions_for(row, src.mentions)
        cls = classify_page(content, url, cfg, prior=_prior_for(cfg, row_d),
                            n_products=len(products), chat_brands=chat_brands)
        mid = (row_d.get("message_ids") or [""])[0]
        match_products(products, cls.primary_brand, mentions, message_id=str(mid),
                       coincide_threshold=cfg.coincide_threshold,
                       name_threshold=cfg.match_name_threshold)
        pr = page_row(row_d, fr, content, cls, len(products))
        page_rows.append(pr)
        product_rows_all.extend(product_rows(pr["page_id"], url, products))

    pages_df, products_df = build_frames(page_rows, product_rows_all)

    os.makedirs(cfg.out_dir, exist_ok=True)
    write_parquet(page_rows, PAGE_SCHEMA, cfg.pages_path())
    write_parquet(product_rows_all, PRODUCT_SCHEMA, cfg.products_path())
    print(f"[export] {cfg.pages_path()} ({len(pages_df)} pages) | "
          f"{cfg.products_path()} ({len(products_df)} products)")

    dist = pages_df["page_category"].value_counts()
    print("[categories]\n" + dist.to_string())
    n_coin = int(products_df["coincides"].sum()) if len(products_df) else 0
    print(f"[match] products={len(products_df)} | coincide={n_coin}")

    evaluation = evaluate(pages_df, products_df, src.ground_truth)
    if evaluation:
        print("[eval]", {k: round(v, 3) for k, v in evaluation.items()})

    return {"config": asdict(cfg), "sources": src, "pages": pages_df,
            "products": products_df, "fetch_results": results, "evaluation": evaluation}


# --------------------------------------------------------------------------- #
# Self-evaluation against synthetic ground truth
# --------------------------------------------------------------------------- #
def evaluate(pages: pd.DataFrame, products: pd.DataFrame,
             ground_truth: Optional[pd.DataFrame]) -> Dict[str, float]:
    if ground_truth is None or ground_truth.empty:
        return {}
    gt = ground_truth.set_index("url")
    merged = pages.set_index("url").join(gt, how="inner")
    if merged.empty:
        return {}
    out: Dict[str, float] = {}
    out["n"] = float(len(merged))
    out["category_accuracy"] = float((merged["page_category"] == merged["gt_category"]).mean())
    commerce = merged[merged["gt_seller_type"].isin(["brand_owned", "retailer"])]
    if len(commerce):
        out["seller_accuracy"] = float((commerce["seller_type"] == commerce["gt_seller_type"]).mean())

    # page-level coincidence: did any product on a page match, vs expectation
    if len(products):
        any_coin = products.groupby("url")["coincides"].any()
    else:
        any_coin = pd.Series(dtype=bool)
    pred = merged.index.to_series().map(any_coin).fillna(False).astype(bool)
    exp = merged["gt_coincides"].astype(bool)
    tp = int(((pred) & (exp)).sum()); fp = int(((pred) & (~exp)).sum())
    fn = int(((~pred) & (exp)).sum())
    out["coincide_accuracy"] = float((pred == exp).mean())
    out["coincide_precision"] = float(tp / (tp + fp)) if (tp + fp) else 1.0
    out["coincide_recall"] = float(tp / (tp + fn)) if (tp + fn) else 1.0
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv=None) -> ScrapeConfig:
    cfg = ScrapeConfig()
    p = argparse.ArgumentParser(description="Scrape + classify surfaced pages, extract products.")
    p.add_argument("--clickstream-dir", default=cfg.clickstream_dir)
    p.add_argument("--online", action="store_true", help="Actually fetch over the network (default: offline)")
    p.add_argument("--max-urls", type=int, default=cfg.max_urls)
    p.add_argument("--dedupe-by", default=cfg.dedupe_by, choices=["url", "domain"])
    p.add_argument("--only-recommended", action="store_true")
    p.add_argument("--classifier", default=cfg.classifier, choices=["auto", "rule", "llm", "embed"])
    p.add_argument("--parser", default=cfg.html_parser, choices=["auto", "stdlib", "bs4"])
    p.add_argument("--synthetic-pages", type=int, default=cfg.synthetic_n_pages)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=cfg.out_dir)
    a = p.parse_args(argv)
    return ScrapeConfig(
        clickstream_dir=a.clickstream_dir, offline=not a.online, max_urls=a.max_urls,
        dedupe_by=a.dedupe_by, only_recommended=a.only_recommended, classifier=a.classifier,
        html_parser=a.parser, synthetic_n_pages=a.synthetic_pages,
        use_cache=not a.no_cache, out_dir=a.out_dir,
    )


if __name__ == "__main__":
    run_scrape(_parse_args())
