"""conveyer.scraping — scrape surfaced pages, classify them, extract & match products.

The links surfaced during the skincare conversations (the SimilarWeb clickstream
``dim_digital_site.url`` / ``fact_ai_click_through.surfaced_url`` columns) are
fetched, parsed, and turned into a two-table parquet star:

* ``fact_scraped_page``   — one row per URL: extracted page info + a **page
  classification** (catalogue / brand-landing = Discovery, shopping = retailer or
  brand-owned, editorial / search / community / reference, or unrelated).
* ``fact_scraped_product`` — one row per product on a page: extracted metadata
  (price, description, rating, category) **matched to the product the agent
  mentioned** on the turn that surfaced the link, with a ``coincides`` verdict.

Everything runs offline on a synthetic corpus with only the standard library
(BeautifulSoup / Anthropic are optional upgrades), so it is testable before real
data or network access is plugged in.

Public API::

    from conveyer.scraping import ScrapeConfig, run_scrape
    art = run_scrape(ScrapeConfig())

Building blocks: :mod:`~conveyer.scraping.sources` (URLs from the clickstream),
:mod:`~conveyer.scraping.fetch`, :mod:`~conveyer.scraping.extract`,
:mod:`~conveyer.scraping.classify`, :mod:`~conveyer.scraping.products`,
:mod:`~conveyer.scraping.schema`, :mod:`~conveyer.scraping.taxonomy`.
"""

from .config import ScrapeConfig
from .pipeline import evaluate, run_scrape
from .resume import prepare_run, reset_run_state
from .taxonomy import CATEGORY_DEFINITIONS, PAGE_CATEGORIES, SELLER_TYPES


def classify_url(url: str, cfg: "ScrapeConfig | None" = None):
    """URL-only classification (no fetch): the multimodal rule scorer over URL
    tokens + domain knowledge + the domain directory's role. Used by
    :mod:`conveyer.journey` to cover trail URLs that weren't scraped, and handy
    interactively."""
    from .classify import classify_rule
    from .directory import lookup
    from .extract import PageContent

    cfg = cfg or ScrapeConfig()
    entry = lookup(url, cfg.directory_path) if cfg.directory_fallback else None
    return classify_rule(PageContent(url=url), url, cfg, directory_entry=entry)


__all__ = [
    "ScrapeConfig",
    "run_scrape",
    "prepare_run",
    "reset_run_state",
    "evaluate",
    "classify_url",
    "PAGE_CATEGORIES",
    "SELLER_TYPES",
    "CATEGORY_DEFINITIONS",
]
