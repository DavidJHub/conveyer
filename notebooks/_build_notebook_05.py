"""Builds notebooks/05_page_scraping.ipynb programmatically.

Run from the repo root or notebooks/:  python notebooks/_build_notebook_05.py
Then execute it:                       python notebooks/_build_notebook_05.py --execute
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
TARGET = HERE / "05_page_scraping.ipynb"

cells = []


def md(source: str):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# ============================================================================ #
md("""
# Scraping & classifying the pages the LLM surfaces

The clickstream dataframes carry **hundreds of thousands of links** — the URLs
the assistant cited in its answers and the pages users browsed right after each
turn (`dim_digital_site.url`, `fact_ai_click_through.surfaced_url`,
`simweb_input_file.a_links_source` / `next_10_urls`). This notebook demonstrates
`conveyer.scraping`, the module that turns those link columns into an analysable
dataset:

1. **Scrape** each URL (politely: robots.txt, per-domain rate limit, cache) and
   extract *everything the page offers* — title, meta/OpenGraph, headings, text,
   schema.org JSON-LD / microdata, links, images.
2. **Classify** the page into a funnel-mapped taxonomy — the three buckets the
   study asked for plus four proposed additions the data demands:

| `page_category` | Funnel stage | Origin |
|---|---|---|
| `brand_landing` | Discovery | requested |
| `catalogue` | Discovery | requested |
| `shopping` (seller: `brand_owned` / `retailer`) | Intent / Purchase | requested |
| `editorial` | Evaluation | **proposed** |
| `search` | Intent | **proposed** |
| `community` | Evaluation | **proposed** |
| `reference` | Awareness | **proposed** |
| `unrelated` / `unknown` | Irrelevant | requested / escape hatch |

3. **Extract the products** on each page (price, description, rating, category,
   SKU) and **match them to the product the agent mentioned** on the turn that
   surfaced the link (`fact_ai_recommendation` + `fact_ai_concept`), yielding a
   per-product `coincides` verdict.
4. **Persist** the result as a two-table parquet star with a pinned schema —
   `fact_scraped_page` + `fact_scraped_product` — documented in
   [`docs/SCRAPED_PAGES_SCHEMA.md`](../docs/SCRAPED_PAGES_SCHEMA.md).

Like every conveyer module, it runs end-to-end **offline** on a synthetic
ground-truth corpus when the real data (or the network) is absent — so this
notebook executes anywhere, and doubles as the module's acceptance test.
""")

# ---------------------------------------------------------------------------- #
md("""
## 1 · Setup & configuration

`ScrapeConfig` is safe by default: `offline=True` (no network), synthetic
fallback when `data/similarweb_clickstream_data/` is missing, vendor
`page_type` prior enabled. Flip `offline=False` (or run
`python -m conveyer.scraping --online`) for real fetching.
""")

code("""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
while not (ROOT / "conveyer").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from conveyer.scraping import ScrapeConfig, run_scrape, CATEGORY_DEFINITIONS

cfg = ScrapeConfig(
    clickstream_dir=str(ROOT / "data/similarweb_clickstream_data"),
    offline=True,                 # never touches the network in this demo
    synthetic_n_pages=60,
    out_dir=str(ROOT / "outputs/scrape"),
)
pd.set_option("display.max_colwidth", 60)
""")

md("""
## 2 · Run the pipeline

One call: resolve sources (real parquet if present, synthetic corpus otherwise)
→ fetch → extract → classify → extract & match products → write both parquet
files → self-evaluate against ground truth when the corpus provides it.
""")

code("""
art = run_scrape(cfg)
pages, products = art["pages"], art["products"]
""")

# ---------------------------------------------------------------------------- #
md("""
## 3 · `fact_scraped_page` — one row per URL

Identity + fetch metadata + extracted page info + classification + provenance.
The full 52-column schema is in `docs/SCRAPED_PAGES_SCHEMA.md`; here is the
analytical core:
""")

code("""
view_cols = ["url", "page_category", "page_subtype", "seller_type", "funnel_stage",
             "page_category_confidence", "skincare_relevance", "n_products",
             "has_price", "times_surfaced", "times_recommended", "prior_page_type"]
pages[view_cols].head(10)
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
HUE, INK = "#4878a8", "#444444"

for ax, col, title in ((axes[0], "page_category", "Pages by category"),
                       (axes[1], "funnel_stage", "Pages by funnel stage")):
    counts = pages[col].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color=HUE, height=0.62)
    for y, v in enumerate(counts.values):
        ax.text(v + max(counts.values) * 0.02, y, str(v), va="center",
                fontsize=9, color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.tick_params(labelsize=9, colors=INK, length=0)
    ax.set_xlim(0, max(counts.values) * 1.15)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_visible(False)
plt.tight_layout()
plt.show()
""")

md("""
### Seller split & the vendor prior

`seller_type` answers the requested "external retailer or brand owned" split as
an **orthogonal axis**, so `shopping` and `catalogue` don't each fork in two.
`prior_page_type` keeps SimilarWeb's own label verbatim: the classifier blends
it as a prior (`classifier_method = "rule+prior"`), and the crosstab below makes
every disagreement auditable.
""")

code("""
display(pd.crosstab(pages["page_category"], pages["seller_type"]))
with_prior = pages[pages["prior_page_type"] != ""]
pd.crosstab(with_prior["prior_page_type"], with_prior["page_subtype"])
""")

# ---------------------------------------------------------------------------- #
md("""
## 4 · `fact_scraped_product` — products on the pages, matched to the chat

Product metadata comes from schema.org `Product` JSON-LD first (what Google
Shopping reads), then OpenGraph `product:*`, microdata, and a visible-price
heuristic — `extraction_source` records which. Each product is scored against
the entities the agent mentioned on the turn(s) that surfaced the page
(SKU > brand > name-overlap > category), and `coincides` fires when the score
clears `cfg.coincide_threshold` (0.5).
""")

code("""
meta_cols = ["name", "brand", "price", "currency", "rating", "rating_count",
             "category", "extraction_source"]
products[meta_cols].head(8)
""")

code("""
match_cols = ["name", "matched_entity", "match_type", "match_score", "coincides"]
print(products["match_type"].value_counts().to_string(), "\\n")
print(f"coincides: {int(products['coincides'].sum())}/{len(products)} products")
products.loc[products["coincides"], match_cols].head(8)
""")

md("""
## 5 · Self-evaluation

The synthetic corpus ships ground-truth labels (category, seller, expected
coincidence), so the pipeline scores itself on every run. **Read this as a test
of the plumbing and rules, not real-world accuracy** — real pages add
JS-rendered content, bot walls and unmarked-up products.
""")

code("""
art["evaluation"]
""")

# ---------------------------------------------------------------------------- #
md("""
## 6 · The parquet artefacts

Both tables are written with explicit pyarrow schemas — nullable ints
(`http_status`, `rating_count`), `list<string>` columns (`schema_types`,
`message_ids`) and floats-with-null all round-trip cleanly.
""")

code("""
import pyarrow.parquet as pq

for f in ("scraped_pages.parquet", "scraped_products.parquet"):
    t = pq.read_table(Path(cfg.out_dir) / f)
    print(f"{f}: {t.num_rows} rows x {t.num_columns} cols")
print()
print(pq.read_table(Path(cfg.out_dir) / "scraped_pages.parquet").schema.to_string()[:600])
""")

md("""
## 7 · Running it for real

```bash
# real URLs from the clickstream, polite online fetching, vendor prior on
python -m conveyer.scraping --clickstream-dir data/similarweb_clickstream_data \\
    --online --max-urls 5000 --dedupe-by url

# only links the LLM actually recommended, one representative per domain
python -m conveyer.scraping --online --only-recommended --dedupe-by domain
```

Key `ScrapeConfig` knobs:

| knob | default | meaning |
|---|---|---|
| `offline` | `True` | never touch the network; serve corpus/cache |
| `respect_robots` / `rate_limit_per_domain` | `True` / 1.0s | politeness (per-domain throttle + robots.txt) |
| `use_cache` / `cache_dir` | `True` | every fetch cached; re-runs are free |
| `max_urls`, `only_recommended`, `dedupe_by` | all / off / url | scope control for the 265k-URL universe |
| `classifier` | `auto` | rule scorer; refines low-confidence pages with an LLM when `ANTHROPIC_API_KEY` is set |
| `use_similarweb_prior` / `prior_weight` | `True` / 0.5 | blend `dim_digital_site.page_type` into the vote |
| `coincide_threshold` | 0.5 | match score needed to declare product coincidence |

**Known limits & next steps.** (1) JS-rendered storefronts need a headless
browser — plug Playwright in at `Fetcher.fetch` if coverage demands it.
(2) `matched_recommendation_id` is inferred (the source
`click_through.recommendation_id` is 100% null) — keep `match_score` in
downstream models. (3) The scraped `funnel_stage` gives `conveyer.funnel`'s HMM
a second, behavioural emission channel: *where the user actually landed* vs
*what they asked*.
""")

# ============================================================================ #
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
})
nbf.write(nb, TARGET)
print(f"wrote {TARGET} ({len(cells)} cells)")

if "--execute" in sys.argv:
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, TARGET)
    print("executed OK")
