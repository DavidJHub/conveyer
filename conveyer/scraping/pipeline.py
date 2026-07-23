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

Designed so a long real-data run can neither hang, nor lose work, **nor grow
its memory with the number of pages**:

* every URL is processed **as its fetch completes** (no batch barrier), under a
  per-URL wall-clock cap (``ScrapeConfig.hard_timeout``);
* every finished page is **appended immediately** to a line-per-record JSONL
  sidecar (products first, then the page line as the commit marker), so a crash
  or Ctrl-C loses at most the page in flight;
* every ``checkpoint_every`` pages the in-memory buffer is **flushed to a
  parquet part file** (``scraped_pages_parts/part-NNNNNN.parquet``) and
  **cleared** — memory stays bounded by one checkpoint's worth of rows no
  matter how long the run is (the old design kept every row in RAM and rewrote
  the whole parquet at each checkpoint: O(n) memory, O(n²) writes);
* on exit (including interrupt) the part files are **streamed** into the final
  single-file parquet — one part in memory at a time. Mid-run progress is
  directly queryable too: ``pd.read_parquet("outputs/scrape/scraped_pages_parts")``;
* with ``resume=True`` (default) a re-run skips URLs already in the JSONL —
  the JSONL commit log is the source of truth, and any rows it holds that
  never made it into a part file (hard crash) are recovered into parts in
  bounded chunks. Old-format outputs (JSONL only, no parts) migrate
  automatically the same way;
* raw HTML is on disk, not in RAM: online fetches are cached one-file-per-URL
  under ``cache_dir`` (the page row's ``html_path`` points at the file), and
  offline replays read those files lazily instead of preloading the corpus.
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
from .schema import (PAGE_SCHEMA, PRODUCT_SCHEMA, finalize_from_parts,
                     next_part_seq, page_row, part_column_values, product_rows,
                     write_part)
from .sources import ScrapeSources, build_sources, chat_brands_for, mentions_for
from .synthetic import make_corpus


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
def _resolve_sources(cfg: ScrapeConfig, sources: Optional[ScrapeSources] = None) -> ScrapeSources:
    # NB: offline replays no longer preload every cached page's HTML into one
    # dict (2MB/page × 100k pages = machine-killing); the Fetcher now reads
    # the per-URL cache files lazily in offline mode.
    if sources is not None:
        return sources
    if os.path.exists(cfg.clickstream_dir):
        real = build_sources(cfg)
        if real is not None and not real.urls.empty:
            return real
    if cfg.use_synthetic_if_missing:
        return make_corpus(cfg.synthetic_n_pages, cfg.synthetic_seed)
    raise FileNotFoundError(
        f"No clickstream data at {cfg.clickstream_dir} and use_synthetic_if_missing=False")


# --------------------------------------------------------------------------- #
# Incremental persistence (JSONL commit log + parquet part files)
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: str):
    """Stream a JSONL file row by row, tolerating a torn final line from a
    crash. Never materializes the whole file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _recover_state(cfg: ScrapeConfig) -> Tuple[set, int]:
    """Reconcile the JSONL commit logs with the parquet part files, in bounded
    memory. Returns ``(done_urls, n_pages_done)``.

    The JSONL is the source of truth. Rows it holds that never reached a part
    file (hard crash between checkpoints — or a pre-parts-format output, which
    migrates automatically here) are streamed into new parts in
    ``checkpoint_every``-sized chunks. Part rows the JSONL does NOT vouch for
    (a truncated/edited log) invalidate the parts, which are rebuilt from the
    log. Products keep the commit rule: only rows whose page line landed."""
    pparts, prparts = cfg.pages_parts_dir(), cfg.products_parts_dir()
    chunk = max(int(cfg.checkpoint_every or 500), 1)

    # -- pages ------------------------------------------------------------- #
    part_pids = part_column_values(pparts, "page_id")
    jsonl_pids: set = set()
    done_urls: set = set()
    for row in _iter_jsonl(cfg.pages_jsonl_path()):
        pid, url = row.get("page_id"), row.get("url")
        if pid:
            jsonl_pids.add(pid)
        if url:
            done_urls.add(url)
    if part_pids - jsonl_pids:      # parts claim pages the truth log lost
        import shutil
        shutil.rmtree(pparts, ignore_errors=True)
        shutil.rmtree(prparts, ignore_errors=True)
        part_pids = set()

    seq = next_part_seq(pparts)
    buf: List[dict] = []
    recovered: set = set()
    for row in _iter_jsonl(cfg.pages_jsonl_path()):
        pid = row.get("page_id")
        if pid in part_pids or pid in recovered:
            continue
        recovered.add(pid)
        buf.append(row)
        if len(buf) >= chunk:
            seq = write_part(buf, PAGE_SCHEMA, pparts, seq)
            buf = []
    write_part(buf, PAGE_SCHEMA, pparts, seq)

    # -- products (only those whose page committed) ------------------------- #
    part_prids = part_column_values(prparts, "product_id")
    seq = next_part_seq(prparts)
    buf = []
    for row in _iter_jsonl(cfg.products_jsonl_path()):
        if row.get("product_id") in part_prids or row.get("page_id") not in jsonl_pids:
            continue
        buf.append(row)
        if len(buf) >= chunk:
            seq = write_part(buf, PRODUCT_SCHEMA, prparts, seq)
            buf = []
    write_part(buf, PRODUCT_SCHEMA, prparts, seq)

    return done_urls, len(jsonl_pids)


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
        base = ""
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
        # where the raw HTML lives on disk (the per-URL fetch cache), so any
        # page can be re-inspected / re-classified without re-fetching
        html_path = ""
        if cfg.use_cache:
            src_url = url if scope == "page" else (base if scope == "base" else "")
            if src_url:
                p = _cache_path(cfg, src_url)
                if os.path.exists(p):
                    html_path = p
        pr = page_row(row_d, fr, content, cls, len(products), fetch_scope=scope,
                      html_path=html_path)
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
    if not cfg.resume:
        # a fresh run must not inherit a previous run's state
        import shutil
        for p in (cfg.pages_jsonl_path(), cfg.products_jsonl_path()):
            if os.path.exists(p):
                os.remove(p)
        for d in (cfg.pages_parts_dir(), cfg.products_parts_dir()):
            shutil.rmtree(d, ignore_errors=True)
        done_urls: set = set()
        n_done = 0
    else:
        # bounded-memory recovery: JSONL truth -> parquet parts; we keep only
        # the done-URL set in RAM, never the previous rows themselves
        done_urls, n_done = _recover_state(cfg)
        if done_urls:
            print(f"[resume] {n_done} pages already done in "
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
    # only the rows since the last checkpoint live in memory; each checkpoint
    # flushes them to a parquet part file and clears the buffer
    pparts, prparts = cfg.pages_parts_dir(), cfg.products_parts_dir()
    seq_pages, seq_products = next_part_seq(pparts), next_part_seq(prparts)
    buf_pages: List[dict] = []
    buf_products: List[dict] = []
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

            buf_pages.append(pr)
            buf_products.extend(prods)
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
                seq_pages = write_part(buf_pages, PAGE_SCHEMA, pparts, seq_pages)
                seq_products = write_part(buf_products, PRODUCT_SCHEMA, prparts, seq_products)
                buf_pages, buf_products = [], []
                print(f"[checkpoint] {n_done + n_new} pages on disk in {pparts} "
                      f"— buffer flushed, memory freed", flush=True)
    finally:
        pages_fh.close()
        products_fh.close()
        # flush the tail buffer and stream all parts into the final parquet —
        # also on crash/interrupt; memory stays bounded by one part
        write_part(buf_pages, PAGE_SCHEMA, pparts, seq_pages)
        write_part(buf_products, PRODUCT_SCHEMA, prparts, seq_products)
        buf_pages, buf_products = [], []
        n_pages_total = finalize_from_parts(pparts, PAGE_SCHEMA, cfg.pages_path())
        n_products_total = finalize_from_parts(prparts, PRODUCT_SCHEMA, cfg.products_path())

    print(f"[export] {cfg.pages_path()} ({n_pages_total} pages) | "
          f"{cfg.products_path()} ({n_products_total} products) | new this run: {n_new}")
    # one arrow-backed copy for the return value / reports (the run itself
    # never held the full tables in memory)
    pages_df = pd.read_parquet(cfg.pages_path())
    products_df = pd.read_parquet(cfg.products_path())

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
