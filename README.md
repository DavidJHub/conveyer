# conveyer

**From ChatGPT shopping conversations to a predictive funnel model.**

The dataset is a parquet of skincare conversations with an LLM agent, each turn
carrying the user's prompt, the agent's answer, the links the answer surfaced,
the links clicked directly from it, and the next 10 pages the user browsed
afterwards. The project's goal is a **funnel model** over that data: measure
the **conversion rate** of these journeys and the **weight of agent
recommendations** — the brands the agent names, the links it shows — in the
user's decision to buy.

🧭 **[docs/PROJECT_WIKI.md](docs/PROJECT_WIKI.md) — the project wiki**: full
description, architecture, glossary, FAQ, roadmap (editable Markdown;
pastes straight into Confluence) · 💶
[docs/SALES_ATTRIBUTION.md](docs/SALES_ATTRIBUTION.md) — **from journeys to
euros**: the assumption ledger, the sales bridge, the MMM connection, what is
still missing and who has to supply it · 📚
[docs/STATE_OF_THE_ART.md](docs/STATE_OF_THE_ART.md) — the literature this
sits in · 📖 [docs/FUNNEL_MODEL.md](docs/FUNNEL_MODEL.md) — the model design:
event grammar, conversion proxy, exposure definitions, limitations · 🗄️
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — input schema + every
output table.

## The three modules

```
input: data/conversations.parquet
  session_id · message_id · user_id · prompt_datetime · question · answer
  a_links_source · ai_click · next_10_urls ({request_time, requested_site})

conveyer/
  conversations.py   MODULE 1 — what was said.
                     Turn + session features: intent (rules), sentiment
                     (lexicon → transformers when installed), topics (BERTopic
                     → KMeans fallback), brand mentions via the shared lexicon
                     (unsolicited recommendations = the exposure treatment),
                     funnel stages (keyword classifier + HMM smoothing).
                     → outputs/conversations/{turn,conversation}_features.parquet

  scraping/          MODULE 2 — where the user landed.
                     Fetch + classify every surfaced/visited URL into a
                     funnel-mapped taxonomy (brand_landing/catalogue/shopping/
                     editorial/search/community/reference/unrelated/unknown),
                     extract product metadata, match products to the chat.
                     Classification is multimodal with a fallback chain:
                     page content → base-URL content → URL/domain heuristics —
                     an unfetchable amazon cart still classifies correctly.
                     Incremental (JSONL + resume), polite, hang-proof;
                     domains never re-worked (circuit breaker, per-domain
                     fetch budget, learned domain profiles).
                     → outputs/scrape/scraped_{pages,products}.parquet

  dashboard.py       insight dashboard — the objectives on one self-contained
                     HTML page: conversion + CIs, recommendation weights,
                     behavioural funnel, brand leaderboard, journey shape.
                     (python -m conveyer.dashboard --outputs outputs)

  journey.py         MODULE 3 — did it move the needle.
                     Joins both into behavioural events (cited / ai_click /
                     trail_visit with dwell), ties page brands back to chat
                     mentions, computes conversion (documented cart/checkout
                     proxy) with Jeffreys CIs by exposure stratum, funnel
                     transition matrix, and a holdout-validated logistic model
                     whose standardized coefficients weigh agent
                     recommendations against every other driver.
                     → outputs/journey/{funnel_events,journey_features,
                        funnel_transitions,model_coefficients}.parquet

  attribution/       MODULE 4 — how much of the *sales* was influenced.
                     Session grain → brand × market × category × platform ×
                     week; panel → universe projection; the euro bridge
                     computed two independent ways (bottom-up journey chain,
                     top-down market reach) over an explicit assumption
                     ledger, Monte-Carlo'd, with a tornado that names which
                     unknown owns the answer; and the AI variable inside a
                     marketing-mix model (share-of-voice regressor,
                     adstock/saturation, VIF diagnostics, a prior calibrated
                     from the bridge, media → AI-visibility → sales mediation).
                     Keeps *influenced* (touchpoint) and *incremental*
                     (causal) strictly apart.
                     → outputs/attribution/{fact_ai_exposure_weekly,
                        ai_influenced_sales, ai_attribution_*, mmm_*}.parquet

  support: ingest.py (loading, trail parsing, synthetic ground truth) ·
  brands.py (canonical lexicon shared by text & domains) · funnel.py (stage
  taxonomy + HMM) · models.py (auto-resolving embedding/sentiment/LLM backends)
```

## Quickstart

```bash
pip install -r requirements.txt

# end to end — synthetic + offline when data/network are absent, and then the
# run validates itself against the generator's ground truth
python -m conveyer.pipeline

# real data, polite online scraping, plus the insight dashboard
python -m conveyer.pipeline --data data/conversations.parquet --online --dashboard

# each module standalone
python -m conveyer.scraping --clickstream-dir data/conversations.parquet --online

# module 4 — sales attribution + MMM (synthetic market data when none is given)
python -m conveyer.attribution
python -m conveyer.attribution --journey-dir outputs/journey \
    --turns outputs/conversations/turn_features.parquet \
    --sales data/rms_brand_week.parquet --media data/media_spend.parquet
python -m conveyer.attribution --dump-ledger data/ledger.json   # edit, then --ledger
```

```python
from conveyer import run_pipeline
art = run_pipeline()
art["journey"]["report"]                       # conversion by exposure + CIs
art["journey"]["model"]["coefficients"]        # weight of agent recommendations
```

Notebooks (generated by `notebooks/_build_notebook_0*.py`, committed executed):

| notebook | contents |
|---|---|
| `01_funnel_pipeline.ipynb` | the whole story: modules 1→2→3, conversion lift, model weights, ground-truth validation |
| `02_page_classifier.ipynb` | deep dive on module 2: taxonomy, multimodal/URL-only classification, dwell ranking, incremental scraping |
| `03_relabel_workflow.ipynb` | human-in-the-loop retagging: the review queue, taxonomy-checked corrections, provenance guarantees proven live, gold-boosted retraining |
| `04_classifier_readout.ipynb` | the classifier readout re-rendered from saved parquets: one call per page, provenance banner from the run manifest, and proof the page follows a relabel correction |

## Design principles

* **Everything runs offline first.** No data, no network, no API keys — the
  synthetic generator produces conversations, trails and pages *with ground
  truth*, so every pipeline stage is testable (`tests/`, 36 tests) and every
  notebook executes anywhere. Real data slots in without code changes.
* **Graceful upgrades.** bs4/lxml, sentence-transformers, BERTopic,
  transformers-sentiment and the Anthropic API are optional; each resolves at
  runtime and degrades to a stdlib/sklearn fallback.
* **Explicit schemas.** Every output table is written with a pinned pyarrow
  schema (nullable ints, list columns round-trip cleanly) and documented in
  [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).
* **Honest measurement.** Conversion is a *proxy* (reaching cart/checkout in
  the trail); exposure ≠ causation; both caveats and the identification
  limits are spelled out in [docs/FUNNEL_MODEL.md](docs/FUNNEL_MODEL.md).
* **Assumptions are data.** Everything the sales estimate cannot observe — the
  cart-to-order rate, the online-to-offline multiplier, incrementality — is a
  declared factor with a range, a source and a status (`measured` … 
  `placeholder`), and every run reports which unknown owns the answer instead
  of burying constants in code.
