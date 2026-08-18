"""Builds notebooks/06_scrape_diagnostics.ipynb programmatically.

Run from the repo root or notebooks/:  python notebooks/_build_notebook_06.py
Then execute it:                       python notebooks/_build_notebook_06.py --execute
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
TARGET = HERE / "06_scrape_diagnostics.ipynb"

cells = []


def md(source: str):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# ============================================================================ #
md("""
# Scraper & classifier diagnostics — from an already-scraped parquet

This notebook diagnoses **module 2's two halves separately** — the *scraper*
(did the fetch work, and what did each label stand on?) and the *classifier*
(what did it decide, by which method, with which evidence?) — from a run that
is **already saved as parquet**. Nothing is scraped here.

> **Point it at your data:** it looks for `results/` at the repo root first
> (set `CONVEYER_RUN_DIR` to use any other directory or parquet file). It
> accepts a full `scraped_pages.parquet`, a run directory, or even a **bare
> URL-list parquet** — bare lists classify URL-only on the fly. Only if
> nothing is found does it fall back to generating the synthetic demo run
> into `results/`, so the notebook executes anywhere.

It ends with the **diagnostics dashboard** — every URL with its category,
subtype, method, evidence channels, confidence and suspicion score, plus
per-category example URLs — and answers the standing question:

> *"We are apparently not including a confidence score — which metric are we
> using now?"* → §3. Short version: `page_category_confidence` **is** in the
> parquet, but it saturates near 1.0 by construction; the operative triage
> metric is the **suspicion score**.
""")

# ---------------------------------------------------------------------------- #
md("""
## 0 · Load the saved run
""")

code("""
import os, sys, warnings
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "conveyer").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 96)
pd.set_option("display.width", 160)

from conveyer.scraping import ScrapeConfig
from conveyer.scraping.diagnostics import load_run

# ---- where the already-scraped parquet lives ------------------------------- #
PRODUCTS_FILE = ScrapeConfig().products_filename

def _usable(p: Path) -> bool:
    # load_run can read this: a .parquet file, or a directory holding at
    # least one parquet that is not the products table
    if p.is_file():
        return p.suffix == ".parquet"
    return p.is_dir() and any(f.name != PRODUCTS_FILE for f in p.glob("*.parquet"))

RUN = Path(os.environ.get("CONVEYER_RUN_DIR", "") or (ROOT / "results"))
if not _usable(RUN):
    for cand in (ROOT / "outputs/scrape", ROOT / "outputs/scrape_demo"):
        if _usable(cand):
            RUN = cand
            break
if not _usable(RUN):
    # last resort so the notebook executes anywhere: the synthetic demo run
    print("no saved parquet found — generating the synthetic demo run once…")
    from conveyer.scraping import run_scrape
    run_scrape(ScrapeConfig(out_dir=str(ROOT / "results"), synthetic_n_pages=120,
                            model_path=str(ROOT / "results/page_model.npz")))
    RUN = ROOT / "results"

CFG = ScrapeConfig(out_dir=str(RUN), model_path=str(Path(RUN if RUN.is_dir()
                   else RUN.parent) / "page_model.npz"))
pages, products, source_note = load_run(str(RUN), CFG)
print(f"run       : {RUN}")
print(f"pages     : {len(pages)} rows x {len(pages.columns)} cols")
print(f"products  : {len(products)} rows")
print(f"provenance: {source_note or 'no run_manifest.json (bare parquet)'}")
""")

code("""
# what this table gives us to diagnose with — grouped by concern
GROUPS = {
    "scraper / fetch": ["fetch_status", "http_status", "fetch_scope", "from_cache",
                        "parser", "fetch_error", "server_platform"],
    "extraction":      ["title", "word_count", "n_links", "n_images", "n_products",
                        "has_price", "has_add_to_cart", "has_jsonld"],
    "classification":  ["page_category", "page_subtype", "page_category_confidence",
                        "seller_type", "funnel_stage", "classifier_method",
                        "classification_signals", "skincare_relevance",
                        "is_study_relevant"],
    "provenance":      ["times_surfaced", "times_recommended", "times_visited",
                        "prior_page_type", "prior_seller_type"],
}
for group, cols in GROUPS.items():
    present = [c for c in cols if c in pages.columns]
    missing = [c for c in cols if c not in pages.columns]
    line = f"{group:<16} {len(present)}/{len(cols)} columns"
    if missing:
        line += f"  (missing: {', '.join(missing)})"
    print(line)
""")

# ---------------------------------------------------------------------------- #
md("""
## 1 · Scraper diagnostics — did the fetch work, and what did each label stand on?

Two views. `fetch_status` is the **transport** outcome (did an HTTP response
arrive); `fetch_scope` is the **evidence** outcome — whose content the
classifier actually used, after the fallback chain
(`page → stripped → base → directory → none`). A label is only as strong as
its scope.
""")

code("""
def share(s):
    out = s.value_counts(dropna=False).to_frame("pages")
    out["share"] = (out["pages"] / len(pages) * 100).round(1).astype(str) + "%"
    return out

print("== fetch_status (transport) ==")
print(share(pages["fetch_status"]).to_string())
print("\\n== fetch_scope (evidence the label stood on) ==")
print(share(pages["fetch_scope"]).to_string())
if pages["fetch_error"].astype(str).str.len().gt(0).any():
    print("\\n== top fetch errors ==")
    print(pages.loc[pages["fetch_error"].astype(str) != "", "fetch_error"]
          .value_counts().head(8).to_string())
""")

code("""
# per-domain health: fetch success and what came back
dom = (pages.assign(fetched=pages["fetch_scope"].isin(["page", "stripped"]))
       .groupby("domain")
       .agg(urls=("url", "size"), fetched=("fetched", "sum"),
            mean_words=("word_count", "mean"), products=("n_products", "sum"))
       .sort_values("urls", ascending=False))
dom["fetched"] = dom["fetched"].astype(int).astype(str) + "/" + dom["urls"].astype(str)
dom["mean_words"] = dom["mean_words"].round(0).astype(int)
print(f"{len(dom)} domains — top 15 by URL count")
dom.head(15)
""")

# ---------------------------------------------------------------------------- #
md("""
## 2 · Classifier results — categories, subtypes, methods

`page_category` is the headline label, derived from the finer structural
`page_subtype`. `classifier_method` says **who decided**:

| method | meaning |
|---|---|
| `rule` | the multimodal vote alone (URL / domain / markup / learned model / …) |
| `rule+prior` | vote **plus** the SimilarWeb `page_type` prior fired |
| `llm` | a low-confidence page refined by the Anthropic model (needs `ANTHROPIC_API_KEY`) |
| `human` | a relabel correction — confidence pinned to 1.0, immutable downstream |

Other values you may see: `error` (a row whose processing raised — category
falls back to `unknown`), a `+url_validated` / `+reclassified` suffix (rows
touched by the validate/reclassify repair passes), and `rule (url-only)`
(a bare URL-list input classified without fetching).

`classification_signals` lists every evidence channel that voted on the row —
the classifier is a vote among independent modalities, any one of which can
carry a page alone.
""")

code("""
cat = pages["page_category"].value_counts().to_frame("pages")
cat["share"] = (cat["pages"] / len(pages) * 100).round(1).astype(str) + "%"
from conveyer.scraping.taxonomy import CATEGORY_DEFINITIONS
cat["definition"] = [CATEGORY_DEFINITIONS.get(c, "") for c in cat.index]
cat
""")

code("""
# subtype x category — the structural label each headline derives from
pd.crosstab(pages["page_subtype"], pages["page_category"], margins=True,
            margins_name="total").sort_values("total", ascending=False)
""")

code("""
from collections import Counter
print("== classifier_method ==")
print(share(pages["classifier_method"]).to_string())
print("\\n== evidence channels (pages where each fired) ==")
sig = Counter(s for row in pages["classification_signals"] for s in list(row))
print(pd.Series(sig).sort_values(ascending=False).to_string())
print("\\n== funnel stage ==")
print(share(pages["funnel_stage"]).to_string())
commerce = pages[pages["page_category"].isin(["brand_landing", "catalogue", "shopping"])]
print(f"\\n== seller_type (commerce pages only, {len(commerce)} of {len(pages)}) ==")
print(commerce["seller_type"].value_counts().to_string())
""")

# ---------------------------------------------------------------------------- #
md("""
## 3 · The confidence question — which metric are we using?

**A confidence score *is* included.** Every row of `scraped_pages.parquet`
carries `page_category_confidence`, computed in
`conveyer/scraping/classify.py::_softmax_conf`: the subtype votes are
collapsed into **category-level score mass**, and confidence is the softmax
share of the winning category — `exp(s_top) / Σ exp(s_i)` over categories
with positive score.

Three things to know about it:

1. **It is a *separation* measure, not a calibrated probability.** It says how
   far ahead the winning category is — with only one positive category it is
   exactly 1.0 by definition, and with a clear winner it saturates ≥ 0.99.
2. **The full score vector is not persisted.** `PageClass.scores` (per-category
   vote mass) and the margin exist at classify time but only the collapsed
   confidence reaches the parquet — shown live below.
3. **Because it saturates, nothing downstream relies on it alone.** The
   `min_confidence` gate defaults to 0.0 (never fires) and LLM refinement
   triggers below 0.55 (almost never reached). The **operative triage metric
   is the `suspicion` score** (`conveyer/scraping/relabel.py`): a transparent
   sum of independent red flags that ranks the human review queue —
   `unknown` category **+3**, learned-model disagreement **+2**,
   unrelated-but-relevant **+2**, confidence < 0.9 **+2** (< 0.75 **+3**),
   no page content **+1**, ≤ 2 evidence channels **+1**.

Alongside them, each row carries two more quality axes: `skincare_relevance`
(0–1 topical score; drives the `unrelated` collapse at < 0.15) and the
`classification_signals` / `classifier_method` provenance pair.
""")

code("""
from conveyer.scraping.diagnostics import confidence_report

conf = pages["page_category_confidence"].astype(float)
rep = confidence_report(pages)
print(f"present in parquet : {bool(rep['present'])}")
print(f"mean / median / min: {rep['mean']:.3f} / {rep['median']:.3f} / {rep['min']:.3f}")
print(f"saturated (>=0.99) : {rep['share_saturated']:.0%}   exactly 1.0: {rep['share_exact_1']:.0%}")
print(f"< 0.90 (susp. flag): {rep['share_below_090']:.1%}")
print(f"< 0.55 (LLM gate)  : {rep['share_below_llm_gate']:.1%}  <- why llm-refinement ~never fires")

bands = pd.Series({                       # explicit bounds — no bin edge cases
    "= 1.00 exactly": int((conf == 1.0).sum()),
    "0.99 - <1.00":   int(((conf >= .99) & (conf < 1.0)).sum()),
    "0.90 - <0.99":   int(((conf >= .90) & (conf < .99)).sum()),
    "0.75 - <0.90":   int(((conf >= .75) & (conf < .90)).sum()),
    "0.55 - <0.75":   int(((conf >= .55) & (conf < .75)).sum()),
    "< 0.55":         int((conf < .55).sum()),
})
print("\\n== confidence bands ==")
print(bands.to_string())
assert bands.sum() == len(pages)
print("\\n== mean confidence by method / by category ==")
print(pages.groupby("classifier_method")["page_category_confidence"]
      .agg(["count", "mean", "min"]).round(3).to_string())
print()
print(pages.groupby("page_category")["page_category_confidence"]
      .agg(["count", "mean", "min"]).round(3).sort_values("count", ascending=False)
      .to_string())
""")

md("""
**What the parquet drops** — the score *vector*. Re-scoring a few URLs live
(URL-only, no fetch) shows the per-category vote mass and the margin between
the top two categories: a margin metric would separate rows the saturated
softmax cannot. This is what you'd persist if you wanted a sharper confidence.
""")

code("""
from conveyer.scraping import classify_url

sample = pages.sample(min(6, len(pages)), random_state=7)
rows = []
for url in sample["url"]:
    cls = classify_url(str(url), CFG)
    top2 = sorted(cls.scores.items(), key=lambda kv: -kv[1])[:2]
    margin = round(top2[0][1] - top2[1][1], 2) if len(top2) > 1 else float("inf")
    rows.append({"url": str(url)[:64], "scores (not persisted)": dict(cls.scores),
                 "top-2 margin": margin, "softmax conf": cls.confidence})
pd.DataFrame(rows)
""")

code("""
# the operative metric: suspicion — what the review queue actually ranks by
from conveyer.scraping.relabel import suspicion_report

sus = suspicion_report(pages, CFG)
print("== suspicion distribution ==")
print(sus["suspicion"].value_counts().sort_index().to_string())
print(f"\\nreview queue (suspicion >= 1): {int((sus['suspicion'] >= 1).sum())} of {len(sus)} rows")
sus.loc[sus["suspicion"] >= 1,
        ["url", "page_category", "page_subtype", "page_category_confidence",
         "suspicion", "reasons"]].head(10)
""")

# ---------------------------------------------------------------------------- #
md("""
## 4 · Examples — every category with its URLs
""")

code("""
examples = (pages.assign(conf=conf.round(2))
            .groupby("page_category", sort=False)
            [["url", "page_subtype", "classifier_method", "conf", "title"]]
            .head(4))
order = pages["page_category"].value_counts().index.tolist()
examples = examples.assign(_c=pages["page_category"]).sort_values(
    "_c", key=lambda s: s.map({c: i for i, c in enumerate(order)}))
examples.rename(columns={"_c": "page_category"}).set_index(
    ["page_category", "page_subtype"])
""")

# ---------------------------------------------------------------------------- #
md("""
## 5 · The dashboard — every URL on one page

`conveyer.scraping.diagnostics.build_diagnostics` renders all of the above as
one self-contained HTML page (inline SVG, no JS, light/dark aware): KPIs,
category / subtype / method / evidence charts, the confidence-vs-suspicion
pair, per-category example cards, and the **full URL table ordered most
suspicious first**. It is written next to the parquets so it travels with the
run.
""")

code("""
from conveyer.scraping.diagnostics import build_diagnostics, iframe

html = build_diagnostics(pages, products, CFG, source_note=source_note)
out = (RUN if RUN.is_dir() else RUN.parent) / "diagnostics.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
assert build_diagnostics(pages, products, CFG, source_note=source_note) == html, \\
    "pure function: same data, same page"
iframe(html, height=1500)
""")

# ---------------------------------------------------------------------------- #
md("""
## 6 · The same render from the shell — and what to do with a red row

```bash
# from a run directory (provenance banner from the run's own manifest)
python -m conveyer.scraping.diagnostics results

# from a bare parquet — even one that is just a URL list
python -m conveyer.scraping.diagnostics results/scraped_pages.parquet --out d.html

# cap the every-URL table on very large runs (the page says when it capped)
python -m conveyer.scraping.diagnostics results --max-table-rows 500
```

A row that looks wrong goes through the **relabel workflow**
([notebook 03](03_relabel_workflow.ipynb)): export the suspicion-ranked queue
to CSV, fill `correct_subtype`, apply (provenance-stamped, immutable), retrain
— then re-render this notebook; the page follows the data.

The narrative companion (funnel story, attention, product↔chat match) is the
**classifier readout**: `python -m conveyer.scraping.readout results`
([notebook 04](04_classifier_readout.ipynb)).
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
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, TARGET)
    print("executed OK")
