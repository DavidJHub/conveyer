"""End-to-end scrape → extract → classify → match → parquet, **incrementally**.

Run as a module::

    python -m conveyer.scraping                       # synthetic corpus, offline
    python -m conveyer.scraping --clickstream-dir data/similarweb_clickstream_data/simweb_input_file.parquet \\
        --online --max-urls 2000

or programmatically::

    from conveyer.scraping import ScrapeConfig, run_scrape
    art = run_scrape(ScrapeConfig())          # offline synthetic by default
    art["pages"].head()                       # fact_scraped_page
    art["products"][art["products"].coincides].head()

Designed so a long real-data run can neither hang nor lose work:

* every URL is processed **as its fetch completes** (no batch barrier), under a
  per-URL wall-clock cap (``ScrapeConfig.hard_timeout``);
* every finished page is **appended immediately** to a line-per-record JSONL
  sidecar (products first, then the page line as the commit marker), so a crash
  or Ctrl-C loses at most the page in flight;
* the parquet files are refreshed every ``checkpoint_every`` pages and on exit
  (including on interrupt);
* with ``resume=True`` (default) a re-run skips URLs already in the JSONL and
  folds the previous results into the final parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .classify import PageClass, classify_page, is_transactional_url
from .config import ScrapeConfig
from .directory import directory_content, lookup
from .extract import PageContent, extract_page
from .fetch import Fetcher, FetchResult, _cache_path, base_url_of
from .products import extract_products, match_products
from .schema import (PAGE_SCHEMA, PRODUCT_SCHEMA, build_frames, page_row,
                     product_rows, write_parquet)
from .sources import ScrapeSources, build_sources, chat_brands_for, mentions_for
from .synthetic import make_corpus


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
def _load_cache_corpus(cfg: ScrapeConfig, urls: List[str]) -> Dict[str, str]:
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
    if os.path.exists(cfg.clickstream_dir):
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
# Incremental persistence (JSONL sidecar + parquet snapshots)
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> List[dict]:
    """Read a JSONL file, tolerating a torn final line from a crash."""
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_resume_state(cfg: ScrapeConfig) -> Tuple[List[dict], List[dict]]:
    """Previously completed (page_rows, product_rows); products whose page line
    never landed (crash between the two writes) are dropped so the page's
    re-scrape can't duplicate them."""
    pages = _load_jsonl(cfg.pages_jsonl_path())
    done_page_ids = {p.get("page_id") for p in pages}
    products = [r for r in _load_jsonl(cfg.products_jsonl_path())
                if r.get("page_id") in done_page_ids]
    return pages, products


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


def _process_one(cfg: ScrapeConfig, src: ScrapeSources, row_d: dict,
                 fr: FetchResult, fetcher: Fetcher) -> Tuple[dict, List[dict]]:
    """fetch result → (page_row, product_rows). Never raises: any extraction or
    classification error degrades to an 'unknown' page row so one bad page
    cannot kill a long run.

    Fallback chain (the classifier's multimodal design does the rest):
    1. the page itself (``fetch_scope="page"``);
    2. the **base URL** (``scheme://host/``) when the deep link is unreachable —
       x.com/…/status/… → x.com/ — so domain-level content still informs
       relevance/category (``fetch_scope="base"``; cache makes this ~free since
       base pages repeat across thousands of deep links);
    3. the **domain directory** when nothing is fetchable at all —
       robots_blocked, bot walls, dead hosts — the directory's description of
       the site stands in as domain-level content (``fetch_scope="directory"``;
       see :mod:`conveyer.scraping.directory`);
    4. URL + domain heuristics alone (``fetch_scope="none"``).
    """
    url = row_d.get("url") or fr.url
    try:
        scope = "page" if fr.ok else "none"
        content = extract_page(fr.html, url=url, parser=cfg.html_parser) if fr.ok \
            else PageContent(url=url)
        if not fr.ok and cfg.base_fallback:
            base = base_url_of(url)
            if base:
                frb = fetcher.fetch(base)
                if frb.ok:
                    content = extract_page(frb.html, url=url, parser=cfg.html_parser)
                    scope = "base"
        entry = lookup(url, cfg.directory_path) if cfg.directory_fallback else None
        if scope == "none" and entry is not None and entry.description:
            # the page is off-limits (robots/bot wall/dead host) but the *site*
            # is known — its directory description stands in as content
            content = directory_content(entry, url)
            scope = "directory"
        # products only from the page itself — a homepage's or the directory's
        # markup must not be attributed to a deep link it stands in for; and on
        # transactional URLs (cart/checkout) the text-price heuristic is off,
        # so cart furniture can't become a phantom product
        products = extract_products(content, cfg.max_products_per_page,
                                    include_heuristic=not is_transactional_url(url)) \
            if scope == "page" else []
        chat_brands = chat_brands_for(row_d, src.mentions)
        mentions = mentions_for(row_d, src.mentions)
        cls = classify_page(content, url, cfg, prior=_prior_for(cfg, row_d),
                            n_products=len(products), chat_brands=chat_brands,
                            content_scope=scope, directory_entry=entry)
        if scope == "base":
            cls.signals = [s for s in cls.signals if s != "content"] + ["base_content"]
        elif scope == "directory":
            cls.signals = [s for s in cls.signals if s != "content"] + ["directory_content"]
        mid = (row_d.get("message_ids") or [""])[0]
        match_products(products, cls.primary_brand, mentions, message_id=str(mid),
                       coincide_threshold=cfg.coincide_threshold,
                       name_threshold=cfg.match_name_threshold)
        pr = page_row(row_d, fr, content, cls, len(products), fetch_scope=scope)
        return pr, product_rows(pr["page_id"], url, products)
    except Exception as exc:
        err = FetchResult(url=url, status=fr.status, http_status=fr.http_status,
                          final_url=fr.final_url, content_type=fr.content_type,
                          error=(fr.error + f" | process: {type(exc).__name__}: {exc}").strip(" |"),
                          fetched_at=fr.fetched_at)
        pr = page_row(row_d, err, PageContent(url=url),
                      PageClass(page_category="unknown", method="error"), 0,
                      fetch_scope="none")
        return pr, []


def run_scrape(cfg: ScrapeConfig, sources: Optional[ScrapeSources] = None) -> Dict[str, object]:
    src = _resolve_sources(cfg, sources)
    urls_df = src.urls
    print(f"[sources] {src.source or cfg.clickstream_dir} | urls={len(urls_df)} | "
          f"turns_with_mentions={len(src.mentions)}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    page_rows_all, product_rows_all = ([], []) if not cfg.resume else _load_resume_state(cfg)
    done_urls = {r.get("url") for r in page_rows_all}
    if not cfg.resume:
        # a fresh run must not inherit a previous run's lines in the sidecars
        for p in (cfg.pages_jsonl_path(), cfg.products_jsonl_path()):
            if os.path.exists(p):
                os.remove(p)
    elif done_urls:
        print(f"[resume] {len(done_urls)} pages already done in "
              f"{cfg.pages_jsonl_path()} — skipping them")

    row_by_url: Dict[str, dict] = {}
    pending: List[str] = []
    for _, row in urls_df.iterrows():
        d = row.to_dict()
        u = d.get("url")
        if not u or u in row_by_url:
            continue
        row_by_url[u] = d
        if u not in done_urls:
            pending.append(u)

    fetcher = Fetcher(cfg, html_by_url=src.html_by_url)
    mode = "offline" if cfg.offline else "online"
    print(f"[fetch] mode={mode} | pending={len(pending)} | workers={cfg.max_workers} | "
          f"per-url cap={cfg.hard_timeout:.0f}s | line-by-line -> {cfg.pages_jsonl_path()}")

    n_new = n_ok = n_err = 0
    t_start = time.monotonic()
    pages_fh = open(cfg.pages_jsonl_path(), "a", encoding="utf-8")
    products_fh = open(cfg.products_jsonl_path(), "a", encoding="utf-8")
    try:
        for fr in fetcher.iter_fetch(pending):
            row_d = row_by_url.get(fr.url, {"url": fr.url})
            pr, prods = _process_one(cfg, src, row_d, fr, fetcher)

            # products first, page line last: the page line is the commit marker
            for p in prods:
                products_fh.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
            products_fh.flush()
            pages_fh.write(json.dumps(pr, ensure_ascii=False, default=str) + "\n")
            pages_fh.flush()

            page_rows_all.append(pr)
            product_rows_all.extend(prods)
            n_new += 1
            n_ok += int(fr.ok)
            n_err += int(not fr.ok)

            if cfg.progress_every and n_new % cfg.progress_every == 0:
                el = time.monotonic() - t_start
                rate = n_new / el if el > 0 else 0.0
                eta = (len(pending) - n_new) / rate if rate > 0 else float("inf")
                print(f"[progress] {n_new}/{len(pending)} pages | ok={n_ok} err={n_err} | "
                      f"{rate:.1f} pages/s | eta {eta/60:.1f} min", flush=True)
            if cfg.checkpoint_every and n_new % cfg.checkpoint_every == 0:
                write_parquet(page_rows_all, PAGE_SCHEMA, cfg.pages_path())
                write_parquet(product_rows_all, PRODUCT_SCHEMA, cfg.products_path())
                print(f"[checkpoint] parquet refreshed at {len(page_rows_all)} pages", flush=True)
    finally:
        pages_fh.close()
        products_fh.close()
        # snapshot whatever we have — also on crash/interrupt
        write_parquet(page_rows_all, PAGE_SCHEMA, cfg.pages_path())
        write_parquet(product_rows_all, PRODUCT_SCHEMA, cfg.products_path())

    pages_df, products_df = build_frames(page_rows_all, product_rows_all)
    print(f"[export] {cfg.pages_path()} ({len(pages_df)} pages) | "
          f"{cfg.products_path()} ({len(products_df)} products) | new this run: {n_new}")

    dist = pages_df["page_category"].value_counts()
    print("[categories]\n" + dist.to_string())
    n_coin = int(products_df["coincides"].sum()) if len(products_df) else 0
    print(f"[match] products={len(products_df)} | coincide={n_coin}")

    # URL-rule double check: any unrelated/unknown row a decisive URL verdict
    # contradicts is a classifier gap — surface it on every run
    try:
        from .validate import validation_report
        vrep = validation_report(pages_df, cfg)
        if len(vrep):
            print(f"[validate] {len(vrep)} label(s) disagree with the URL rules — "
                  f"inspect/repair: python -m conveyer.scraping.validate "
                  f"{cfg.pages_path()} --apply")
        else:
            print("[validate] all labels consistent with the URL rules")
    except Exception as exc:
        print(f"[validate] skipped: {type(exc).__name__}: {exc}")

    evaluation = evaluate(pages_df, products_df, src.ground_truth)
    if evaluation:
        print("[eval]", {k: round(v, 3) for k, v in evaluation.items()})

    return {"config": asdict(cfg), "sources": src, "pages": pages_df,
            "products": products_df, "evaluation": evaluation,
            "n_new": n_new, "n_fetch_errors": n_err}


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
    p.add_argument("--clickstream-dir", default=cfg.clickstream_dir,
                   help="Star-schema dir OR a single parquet file (e.g. simweb_input_file.parquet)")
    p.add_argument("--online", action="store_true", help="Actually fetch over the network (default: offline)")
    p.add_argument("--max-urls", type=int, default=cfg.max_urls)
    p.add_argument("--dedupe-by", default=cfg.dedupe_by, choices=["url", "domain"])
    p.add_argument("--only-recommended", action="store_true")
    p.add_argument("--classifier", default=cfg.classifier, choices=["auto", "rule", "llm", "embed"])
    p.add_argument("--parser", default=cfg.html_parser, choices=["auto", "stdlib", "bs4"])
    p.add_argument("--synthetic-pages", type=int, default=cfg.synthetic_n_pages)
    p.add_argument("--timeout", type=float, default=cfg.timeout, help="Socket timeout per request (s)")
    p.add_argument("--hard-timeout", type=float, default=cfg.hard_timeout,
                   help="Wall-clock cap per URL incl. robots + retries (s)")
    p.add_argument("--checkpoint-every", type=int, default=cfg.checkpoint_every)
    p.add_argument("--progress-every", type=int, default=cfg.progress_every)
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore + clear the JSONL sidecars instead of resuming")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=cfg.out_dir)
    a = p.parse_args(argv)
    return ScrapeConfig(
        clickstream_dir=a.clickstream_dir, offline=not a.online, max_urls=a.max_urls,
        dedupe_by=a.dedupe_by, only_recommended=a.only_recommended, classifier=a.classifier,
        html_parser=a.parser, synthetic_n_pages=a.synthetic_pages,
        timeout=a.timeout, hard_timeout=a.hard_timeout,
        checkpoint_every=a.checkpoint_every, progress_every=a.progress_every,
        resume=not a.no_resume, use_cache=not a.no_cache, out_dir=a.out_dir,
    )


if __name__ == "__main__":
    run_scrape(_parse_args())
