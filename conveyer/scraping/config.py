"""Configuration for the page-scraping / classification pipeline.

Mirrors :class:`conveyer.config.PipelineConfig`: one dataclass every module in
``conveyer.scraping`` reads, so column names, network etiquette, model choices
and output paths live in one place. Defaults are **safe and offline** — nothing
touches the network or a real dataset unless you ask it to, so the whole thing
runs end-to-end on the synthetic corpus with no keys, no libraries beyond the
standard library, and no data files present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ScrapeConfig:
    # --- Where the URLs come from (SimilarWeb clickstream tables) ------------ #
    clickstream_dir: str = "data/similarweb_clickstream_data"
    use_synthetic_if_missing: bool = True
    synthetic_n_pages: int = 60          # synthetic pages when no data/network
    synthetic_seed: int = 42

    # dim_digital_site columns (weak-label priors for the classifier)
    col_site_id: str = "digital_site_id"
    col_url: str = "url"
    col_domain: str = "domain"
    col_page_type: str = "page_type"     # pdp/search/checkout/marketplace/category/...
    col_seller_type: str = "seller_type"  # 1p / 3p / unknown
    col_retailer_brand: str = "retailer_brand"
    col_site_category: str = "category"
    col_extracted_skus: str = "extracted_product_skus"

    # fact_ai_click_through columns (provenance + turn linkage)
    col_click_message_id: str = "message_id"
    col_click_site_id: str = "digital_site_id"
    col_surfaced_url: str = "surfaced_url"
    col_was_recommended: str = "was_recommended"
    col_was_visited: str = "was_visited"
    col_resulted_in_purchase: str = "resulted_in_purchase"

    # chat context (fact_ai_recommendation + fact_ai_concept), for matching
    col_rec_message_id: str = "message_id"
    col_rec_id: str = "recommendation_id"
    col_rec_context: str = "entity_context"
    col_concept_fact_id: str = "fact_id"
    col_concept_name: str = "concept_name"
    col_concept_value: str = "concept_value"

    # --- What to scrape ----------------------------------------------------- #
    max_urls: int | None = None          # cap the number of distinct URLs (None = all)
    dedupe_by: str = "url"               # "url" or "domain" (one representative per domain)
    only_recommended: bool = False       # restrict to links the LLM actually surfaced

    # --- Fetching (safe by default) ----------------------------------------- #
    offline: bool = True                 # serve pages from the corpus, never the network
    respect_robots: bool = True
    user_agent: str = ("conveyer-research-bot/0.2 (+https://github.com/; "
                       "skincare agentic-commerce study; contact via repo)")
    timeout: float = 10.0                # socket connect/read timeout (per request)
    hard_timeout: float = 30.0           # wall-clock cap per URL: covers robots +
                                         # all retries + slow-dribble bodies. A URL
                                         # can never stall the pipeline longer than this.
    max_retries: int = 2
    retry_backoff: float = 2.0           # seconds; doubles each retry
    rate_limit_per_domain: float = 1.0   # min seconds between hits to one domain
    max_workers: int = 12                # concurrent fetches
    max_bytes: int = 2_000_000           # skip/truncate bodies larger than this
    cache_dir: str = "outputs/scrape_cache"
    use_cache: bool = True
    allowed_content_types: Tuple[str, ...] = ("text/html", "application/xhtml+xml")
    base_fallback: bool = True           # when a deep link is unreachable, fetch
                                         # scheme://host/ and classify from the
                                         # base page + the original URL's tokens
    directory_fallback: bool = True      # when nothing is fetchable at all
                                         # (robots_blocked, bot wall, dead host),
                                         # classify from the offline domain
                                         # directory's description of the site
                                         # (fetch_scope="directory")
    directory_path: str = "data/domain_directory.json"
                                         # optional external directory entries,
                                         # merged over the built-in seed (see
                                         # conveyer/scraping/directory.py)

    # --- Parsing / extraction ----------------------------------------------- #
    html_parser: str = "auto"            # auto | stdlib | bs4 (auto prefers bs4 if installed)
    text_excerpt_chars: int = 2000       # truncate stored page text
    max_products_per_page: int = 40

    # --- Classification ----------------------------------------------------- #
    classifier: str = "auto"             # auto | rule | llm | embed
    use_similarweb_prior: bool = True    # let dim_digital_site.page_type inform the label
    prior_weight: float = 0.5            # blend weight for the vendor prior (0..1)
    llm_model: str = "claude-opus-4-8"   # override via ANTHROPIC_MODEL
    llm_max_tokens: int = 400
    min_confidence: float = 0.0          # below this, fall back to 'unknown'

    # --- Product <-> chat matching ------------------------------------------ #
    match_name_threshold: float = 0.34   # token-overlap cutoff for a name match
    coincide_threshold: float = 0.5      # match_score >= this ⇒ coincides = True

    # --- Outputs (incremental: line-per-record JSONL + parquet snapshots) ---- #
    out_dir: str = "outputs/scrape"
    pages_filename: str = "scraped_pages.parquet"
    products_filename: str = "scraped_products.parquet"
    resume: bool = True                  # skip URLs already in the JSONL sidecar
    checkpoint_every: int = 500          # refresh the parquet files every N new pages
    progress_every: int = 25             # print a progress line every N pages

    def pages_path(self) -> str:
        import os
        return os.path.join(self.out_dir, self.pages_filename)

    def products_path(self) -> str:
        import os
        return os.path.join(self.out_dir, self.products_filename)

    def pages_jsonl_path(self) -> str:
        return self.pages_path().rsplit(".", 1)[0] + ".jsonl"

    def products_jsonl_path(self) -> str:
        return self.products_path().rsplit(".", 1)[0] + ".jsonl"
