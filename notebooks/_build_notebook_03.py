"""Builds notebooks/03_relabel_workflow.ipynb programmatically.

Run from the repo root or notebooks/:  python notebooks/_build_notebook_03.py
Then execute it:                       python notebooks/_build_notebook_03.py --execute
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
TARGET = HERE / "03_relabel_workflow.ipynb"

cells = []


def md(source: str):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# ============================================================================ #
md("""
# Human-in-the-loop relabelling — fix the classifier, then make it learn the fix

The learned channel of the page classifier **self-trains on its own confident
rows** (weak labels). That lets it absorb the real URL distribution for free —
and makes one failure permanent: a page the whole classifier got wrong is
*confidently* wrong, so the wrong label is exactly what the model trains on.
No amount of self-training can break that loop. A human can, and
`conveyer.scraping.relabel` is the smallest workflow that lets one:

| step | command | what it guarantees |
|---|---|---|
| **export** | `relabel export PAGES --out review.csv` | rows ranked by *independent red flags*, with reasons |
| **correct** | edit `review.csv` | filling `correct_subtype` is enough — the taxonomy derives the rest |
| **apply** | `relabel apply PAGES --corrections review.csv --apply` | taxonomy-validated or loudly rejected; provenance-stamped; **immutable downstream** |
| **retrain** | `relabel retrain PAGES` | human rows enter training as **gold, boosted ×3** |

This notebook runs the whole loop on the offline synthetic corpus and
**asserts every guarantee live** — like every conveyer notebook, it doubles as
an acceptance test. Companion docs:
[`docs/SCRAPING_MODULE.md`](../docs/SCRAPING_MODULE.md) §7 cheat sheet,
[`02_page_classifier.ipynb`](02_page_classifier.ipynb) for the classifier
itself.
""")

# ---------------------------------------------------------------------------- #
md("""
## 1 · Setup — a scrape run to correct

Offline synthetic corpus into its own directory (`outputs/relabel_demo`), so
nothing here touches notebook 02's table. Re-running this notebook **resumes**
the scrape (0 new pages) and then works on a **fresh working copy** of the
parquet each time — corrections never accumulate across runs, so every
execution starts from the same pristine labels.
""")

code("""
import os, shutil, sys, warnings
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / "conveyer").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from conveyer.scraping import ScrapeConfig, run_scrape
from conveyer.scraping.resume import prepare_run

OUT = ROOT / "outputs/relabel_demo"
cfg = ScrapeConfig(
    offline=True, synthetic_n_pages=40,
    out_dir=str(OUT), model_path=str(OUT / "page_model.npz"),
    resume=True, progress_every=0,
)
print(prepare_run(cfg).message)
art = run_scrape(cfg)

# the pristine table stays untouched; all corrections happen on a working copy
PRISTINE = OUT / "scraped_pages.parquet"
WORKING = OUT / "scraped_pages_working.parquet"
shutil.copyfile(PRISTINE, WORKING)
pages = pd.read_parquet(WORKING)
print(f"\\nworking copy: {len(pages)} pages, "
      f"{pages['page_category'].nunique()} categories")
""")

# ---------------------------------------------------------------------------- #
md("""
## 2 · The review queue — where a human's time is worth spending

`suspicion_report` scores every row by a plain **sum of independent red
flags** — transparent enough to argue with, which is the point:

| flag | weight |
|---|---|
| `page_category == "unknown"` (the classifier's own "can't tell") | +3 |
| the learned model disagrees with the stored subtype | +2 |
| `unrelated` yet topically relevant (the two axes conflict) | +2 |
| category confidence < 0.9 (< 0.75) | +2 (+3) |
| classified without page content (`fetch_scope` ≠ page) | +1 |
| ≤ 2 evidence channels fired | +1 |

Human-labelled rows score 0 by definition — they *are* the ground truth.
`export_review` writes the top suspects to a CSV with empty `correct_*`
columns; that CSV is the entire reviewer interface.
""")

code("""
from conveyer.scraping.relabel import export_review, suspicion_report

sus = suspicion_report(pages, cfg)
sus[["url", "page_category", "page_subtype", "page_category_confidence",
     "fetch_scope", "suspicion", "reasons"]].head(8)
""")

code("""
REVIEW = OUT / "review.csv"
exported = export_review(pages, str(REVIEW), n=8, cfg=cfg)
review = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
print("reviewer-facing columns:", [c for c in review.columns])
review[["url", "page_category", "page_subtype", "suspicion",
        "correct_subtype", "correct_category", "note"]].head(8)
""")

# ---------------------------------------------------------------------------- #
md("""
## 3 · Playing the reviewer — one good fix, three bad ones

In real use a person opens `review.csv` in a spreadsheet and fills
`correct_subtype` on the wrong rows. Here we simulate four edits that cover
the validation surface:

1. a **valid** correction — we deliberately relabel a `cart` as an editorial
   `article`, i.e. something the URL rules *actively disagree with*, so §4 can
   prove that a human label survives every automated pass;
2. an **unknown subtype** — rejected with the reason;
3. a **subtype/category mismatch** (`pdp` under `editorial`) — rejected: the
   taxonomy, not the CSV, owns that relationship;
4. a correction pointing at a **URL that isn't in the table** — rejected.

`apply_corrections` returns `(corrected pages, change report)`; nothing is
written until you choose to write it.
""")

code("""
from conveyer.scraping.relabel import apply_corrections

# in real use these edits happen in the CSV; here we build the same frame
# programmatically, against three distinct rows of the table
cart_url = pages.loc[pages["page_subtype"] == "cart", "url"].iloc[0]
others = pages.loc[pages["url"] != cart_url, "url"].iloc[:2].tolist()

review = pd.DataFrame([
    {"url": cart_url, "correct_subtype": "article",
     "note": "demo: human overrules the URL rules"},               # 1 · valid
    {"url": others[0], "correct_subtype": "not_a_subtype"},        # 2 · unknown subtype
    {"url": others[1], "correct_subtype": "pdp",
     "correct_category": "editorial"},                             # 3 · taxonomy mismatch
    {"url": "https://nowhere.example/x",
     "correct_subtype": "article"},                                # 4 · row not found
])

corrected, report = apply_corrections(pages, review)
report
""")

code("""
assert report.loc[report["url"] == cart_url, "applied"].all(), "the valid fix must apply"
assert not report.loc[report["detail"].str.contains("not_a_subtype"), "applied"].any()
assert (report["detail"].str.contains("belongs to")).any(), "mismatch must be explained"
assert (report["detail"].str.contains("not found")).any()

row = corrected.loc[corrected["url"] == cart_url].iloc[0]
print("the corrected row now carries full provenance:")
print(f"  label       : {row['page_category']} / {row['page_subtype']}"
      f"   (funnel: {row['funnel_stage']} — re-derived, never trusted from the CSV)")
print(f"  method      : {row['classifier_method']}")
print(f"  signals     : {list(row['classification_signals'])}")
print(f"  confidence  : {row['page_category_confidence']}")
""")

# ---------------------------------------------------------------------------- #
md("""
## 4 · The guarantees, proven live

A correction is only worth making if nothing can silently undo it. Three
things try:

* **a stale review CSV** — someone re-applies last month's export over a newer
  fix. Refused: once a row is human-labelled, only a deliberate change to the
  parquet moves it again;
* **`apply_validation`** — the URL-rules repair pass. Our URL still *says*
  cart (`/cart` is as decisive as URL tokens get), and the pass repairs
  exactly this kind of disagreement — but a human label outranks the rules;
* **`reclassify_pages`** — the content-aware rescue pass. Same contract.
""")

code("""
from conveyer.scraping.validate import apply_validation, reclassify_pages

_, stale = apply_corrections(corrected, review)     # same CSV, second time
stale_row = stale.loc[stale["url"] == cart_url]
print("re-apply of the same CSV ->", stale_row["detail"].iloc[0])
assert not stale_row["applied"].any()

v_pages, v_report = apply_validation(corrected, cfg)
r_pages, r_report = reclassify_pages(corrected, cfg)
for name, df in (("apply_validation", v_pages), ("reclassify_pages", r_pages)):
    got = df.loc[df["url"] == cart_url, "page_subtype"].iloc[0]
    assert got == "article", f"{name} overwrote the human label with {got!r}"
    print(f"{name:18s} -> still article  (human label untouched)")
""")

# ---------------------------------------------------------------------------- #
md("""
## 5 · Retrain — the fix becomes training signal

`model.train --pages` already self-trains on confident rows as weak labels.
Human rows now enter that same path as **gold**: exempt from the confidence
gate, and **duplicated ×3** (`HUMAN_BOOST`), so a handful of corrections can
out-vote the weak labels that taught the original mistake.

Honesty note: one gold row rarely flips a global model — the hashed-feature
logistic has seen hundreds of weak samples saying otherwise. What matters here
is the *mechanism* (watch the `+1 human-labelled rows (×3)` line and the vote
shift); at review scale — dozens of corrections per batch — the votes move.
""")

code("""
from conveyer.scraping.model import predict_votes, train
from conveyer.scraping.relabel import HUMAN_BOOST

votes_before = predict_votes(None, cart_url, cfg)

CORRECTED = OUT / "scraped_pages_corrected.parquet"
from conveyer.scraping.validate import write_pages
write_pages(corrected, str(CORRECTED))

cfg_v2 = ScrapeConfig(out_dir=str(OUT), model_path=str(OUT / "page_model_v2.npz"),
                      model_autotrain=False)
train(pages_parquet=str(CORRECTED), out_path=cfg_v2.model_path)

votes_after = predict_votes(None, cart_url, cfg_v2)
print(f"\\nlearned-channel votes for {cart_url}")
print(f"  before retrain: {votes_before}")
print(f"  after  retrain: {votes_after}")
print(f"  (gold rows are boosted x{HUMAN_BOOST}; 'article' mass should not shrink)")
""")

# ---------------------------------------------------------------------------- #
md("""
## 6 · The same loop from the shell

Everything above, as the commands a reviewer actually runs:

```bash
# 1 · rank & export the suspects
python -m conveyer.scraping.relabel export outputs/scrape/scraped_pages.parquet \\
    --out review.csv --n 40

# 2 · a human fills correct_subtype in review.csv (spreadsheet, editor, anything)

# 3 · validate + write back, in place (or --out NEW.parquet to keep the original)
python -m conveyer.scraping.relabel apply outputs/scrape/scraped_pages.parquet \\
    --corrections review.csv --apply

# 4 · retrain the learned channel on the corrected table
python -m conveyer.scraping.relabel retrain outputs/scrape/scraped_pages.parquet
```

**Where this fits.** On the first *real* SimilarWeb parquet, the intended
cadence is: run the scrape → `export` a 40-row queue → 20 minutes of human
review → `apply` → `retrain` → re-run `validate --reclassify` for the rows the
better model can now rescue. The queue ranks by expected error, so those 20
minutes land where the classifier is most likely wrong — and every correction
is an audit-trailed row, not an anonymous edit.
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
