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
from .taxonomy import CATEGORY_DEFINITIONS, PAGE_CATEGORIES, SELLER_TYPES

__all__ = [
    "ScrapeConfig",
    "run_scrape",
    "evaluate",
    "PAGE_CATEGORIES",
    "SELLER_TYPES",
    "CATEGORY_DEFINITIONS",
]
