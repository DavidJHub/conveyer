"""Builds notebooks/04_classifier_readout.ipynb programmatically.

Run from the repo root or notebooks/:  python notebooks/_build_notebook_04.py
Then execute it:                       python notebooks/_build_notebook_04.py --execute
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
TARGET = HERE / "04_classifier_readout.ipynb"

cells = []


def md(source: str):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# ============================================================================ #
md("""
# The classifier readout — one page from any saved scrape run

`conveyer.scraping.readout` renders module 2's results as a **self-contained
HTML page** (inline SVG, no JS, no network, light/dark aware — attach it to
Confluence or email it): where the surfaced/visited URLs landed in the
taxonomy, how each label was earned, the product↔chat match, where attention
went, and the human review queue that feeds the
[relabel workflow](03_relabel_workflow.ipynb).

The contract that makes it re-runnable is the point of this notebook:

> **The readout is a pure function of the saved parquets.** No scraping
> happens here — it reads `scraped_pages.parquet` + `scraped_products.parquet`
> from a finished run, plus `run_manifest.json` for the provenance banner and
> the run's `page_model.npz` for the review queue. Render it twice on the same
> data and you get the same page; correct a label and re-render and the page
> follows. The page is a **view, never a store**.

```bash
python -m conveyer.scraping.readout outputs/scrape_demo            # dir → readout.html
python -m conveyer.scraping.readout PAGES.parquet --out r.html     # explicit file
```

*(Viewing this notebook on GitHub? Inline iframes below won't render there —
open the written `readout.html` files directly instead.)*
""")

# ---------------------------------------------------------------------------- #
md("""
## 1 · Point at a saved run

We use notebook 02's table (`outputs/scrape_demo`). If it isn't on disk yet,
the cell produces it with **the same config notebook 02 uses** — identical
config ⇒ identical run manifest ⇒ either notebook can resume the other's run
safely. On a machine where it already exists, nothing is scraped at all.
""")

code("""
import shutil, sys, warnings
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "conveyer").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

RUN = ROOT / "outputs/scrape_demo"
if not (RUN / "scraped_pages.parquet").exists():
    from conveyer.scraping import ScrapeConfig, run_scrape
    from conveyer.scraping.resume import prepare_run
    cfg = ScrapeConfig(clickstream_dir=str(ROOT / "data/conversations.parquet"),
                       offline=True, synthetic_n_pages=60,
                       out_dir=str(RUN), model_path=str(RUN / "page_model.npz"),
                       timeout=10.0, hard_timeout=30.0, max_urls=2000,
                       checkpoint_every=200, progress_every=25, resume=True)
    print(prepare_run(cfg).message)
    run_scrape(cfg)

pages = pd.read_parquet(RUN / "scraped_pages.parquet")
print(f"saved run: {RUN}")
print(f"  {len(pages)} pages · {pages['page_category'].nunique()} categories · "
      f"{pages['fetch_scope'].isin(['page', 'stripped']).sum()} content-backed")
""")

# ---------------------------------------------------------------------------- #
md("""
## 2 · Render it — one call

`readout_from_dir` reads the parquets, derives the **provenance banner** from
`run_manifest.json` (this run says, out loud, that it is the synthetic demo
corpus — the same banner names the input file on a real run), points the
review queue at the run's own model, writes `readout.html`, and returns the
HTML. The iframe sandboxes the page's CSS from the notebook's.
""")

code("""
from conveyer.scraping.readout import iframe, readout_from_dir

html = readout_from_dir(str(RUN), out_path=str(RUN / "readout.html"))
print(f"wrote {RUN / 'readout.html'} ({len(html):,} bytes)")
iframe(html, height=1250)
""")

# ---------------------------------------------------------------------------- #
md("""
## 3 · The page follows the data — correct a label, re-render

The readout carries a **human-corrected labels** KPI and the review queue.
Both must move the moment a correction lands, with no cache or state in
between. We prove it on a *working copy* of the run (the pristine
`scrape_demo` table is never touched): apply one human fix from the
[relabel workflow](03_relabel_workflow.ipynb), re-render, and diff.
""")

code("""
from conveyer.scraping.relabel import apply_corrections, is_human_labelled
from conveyer.scraping.validate import write_pages

WORK = ROOT / "outputs/readout_demo"
WORK.mkdir(parents=True, exist_ok=True)
for name in ("scraped_pages.parquet", "scraped_products.parquet",
             "page_model.npz", "run_manifest.json"):
    if (RUN / name).exists():
        shutil.copyfile(RUN / name, WORK / name)

# the top review-queue suspect gets the human's verdict
from conveyer.scraping import ScrapeConfig
from conveyer.scraping.relabel import export_review
work_cfg = ScrapeConfig(out_dir=str(WORK), model_path=str(WORK / "page_model.npz"))
work_pages = pd.read_parquet(WORK / "scraped_pages.parquet")
suspects = export_review(work_pages, str(WORK / "review.csv"), n=1, cfg=work_cfg)
target_url = suspects.iloc[0]["url"]

corrections = pd.DataFrame([{"url": target_url, "correct_subtype": "cart",
                             "note": "reviewer confirms: it is a cart"}])
corrected, report = apply_corrections(work_pages, corrections)
write_pages(corrected, str(WORK / "scraped_pages.parquet"))
print(report.to_string(index=False))
""")

code("""
html_after = readout_from_dir(str(WORK), out_path=str(WORK / "readout.html"))

n_human_before = int(sum(is_human_labelled(r) for r in
                         pages.to_dict("records")))
n_human_after = int(sum(is_human_labelled(r) for r in
                        corrected.to_dict("records")))
print(f"human-corrected labels: {n_human_before} -> {n_human_after}")
assert n_human_after == n_human_before + 1
assert html_after != html, "the page must change when the data changes"
assert ">%d</div>" % n_human_after not in html, "sanity: counts differ"
iframe(html_after, height=1250)
""")

# ---------------------------------------------------------------------------- #
md("""
## 4 · The same render from the shell

```bash
# from a run directory (banner + review queue come from the run's own files)
python -m conveyer.scraping.readout outputs/scrape_demo --out readout.html

# from a bare parquet (products picked up from the same directory if present)
python -m conveyer.scraping.readout outputs/scrape/scraped_pages.parquet

# more review-queue rows on the page
python -m conveyer.scraping.readout outputs/scrape_demo --queue-rows 25
```

**Cadence on real data.** After every scrape (or relabel pass): re-run the
readout, attach it to the wiki, and read the review queue top-down — the page
is cheap enough to regenerate that it should never be older than the parquet
it describes.
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
