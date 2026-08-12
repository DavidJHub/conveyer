# Sales attribution — from AI conversations to euros, and into the MMM

> **What this page is.** The methodology and the build plan for the measurement
> approach the Innovation Lab asked for: *how much of our sales is influenced
> by AI conversations*. It states what the current pipeline can and cannot
> support, what the missing pieces are, how they are built (module 4,
> `conveyer/attribution/`, shipped with this page), how the marketing-mix model
> connects, and what has to come from data partners before any number can be
> published. Companion pages: [PROJECT_WIKI](PROJECT_WIKI.md) (the system
> today), [FUNNEL_MODEL](FUNNEL_MODEL.md) (module 3 methodology),
> [DATA_DICTIONARY](DATA_DICTIONARY.md) (all tables).

---

## 1 · Where we stand

Modules 1–3 answer, per journey: *what did the agent say*, *where did the user
go*, *did they reach a cart*. That is a complete, validated measurement of
**behaviour inside a desktop clickstream panel**.

The ask is a different quantity: **a share of market sales**. Between the two
sit five things the clickstream cannot see — whether a cart became an order,
whether the purchase happened on another device, whether it happened in a
shop, what it was worth, and whether it would have happened anyway. Every one
of them is a multiplication, so the panel's precision does not survive
contact with the question: a beautifully measured 35% cart-reach rate becomes
a sales number only after five factors nobody has measured yet.

That is why the current demo's sales figures are invented. The problem is not
that assumptions were made — they are unavoidable — but that they were
implicit. Module 4 makes them explicit, ranges them, and reports how much of
the answer each one owns.

### The distinction everything else hangs on

| | **AI-influenced sales** | **AI-incremental sales** |
|---|---|---|
| claim | the AI conversation was in the journey | the sale would not have happened without it |
| kind | touchpoint / reach | causal |
| size | large | several times smaller |
| measurable from | panel + calibration factors | experiments or an MMM — **never** the panel alone |
| who asks for it | brand teams, comms, the 3-month product | finance, budget allocation, the MMM |

The meeting asked for *influenced*, "not driven or done by AI… more loosely".
That is the right scope for the first release and it is achievable. But the
two numbers must never appear without their labels, because the moment an
influence number is used to justify investment it is being read as an
incremental one. Module 4 reports both, always, with the second explicitly
derived from the first by a factor whose only honest source is the MMM.

### The rule that resolves most design questions

> **The panel gives shares and relativities. Company data gives levels.**

Do not scale a desktop cart-reach rate up into an absolute sales figure and
present it as measurement. Use the panel for what it is unbiased-ish about —
*which* conversations convert more, *which* brands get recommended, *how*
exposure changes behaviour, *how* that moves week to week — and take the
absolute level from RMS / Omnisales / CPS. The bridge is built around this
rule; the top-down route exists precisely to keep the bottom-up route
anchored to a market quantity.

---

## 2 · What is missing, and where it now lives

| # | gap | why it blocks the ask | status |
|---|---|---|---|
| 1 | no `llm_platform` / `category` / `market` dimensions | data is arriving for Gemini, Claude, Perplexity and more categories; pooling them hides the effect | **built** — `attribution/aggregate.py`, defaults when the extract lacks them |
| 2 | no weekly aggregation | RMS, CPS and every MMM live at brand-market-week; nothing can be joined at session grain | **built** — `fact_ai_exposure_weekly`, `fact_ai_category_weekly` |
| 3 | no panel→universe projection | panel counts are not market counts; desktop-only is not all-device | **built** — `attribution/panel.py` (projection, Kish weights, design effect) |
| 4 | no declared assumptions | the sales estimate rested on invented constants inside code | **built** — `attribution/calibration.py`, the assumption ledger |
| 5 | no sales bridge | the actual missing span: journeys → euros | **built** — `attribution/bridge.py`, two independent routes + Monte Carlo |
| 6 | no uncertainty on the answer | a product of six uncertain factors needs an interval, not a point | **built** — MC intervals + a tornado ranking the assumptions |
| 7 | no MMM connection | incrementality cannot come from observational panel data | **built** — `attribution/mmm.py` (reference model + the regressor table to hand over) |
| 8 | **brand ↔ RMS/Omnisales hierarchy mapping** | the panel speaks the chat lexicon, sales speaks the product hierarchy; a string join is not a mapping | **not built — needs the sales extract**; the bridge fails loudly on it rather than silently mis-joining |
| 9 | **the factor values themselves** | 4 of 10 factors are placeholders | **not built — needs data**; §4 says which source retires which |
| 10 | demographics / geo | segmentation, and the representativeness correction | **not available**; the ledger carries a wide band instead of a fake weight |
| 11 | user-level longitudinal browsing | purchases after the click window are invisible; `window_recovery` is a guess | **not available**; ask the provider |
| 12 | a causal design | the only real source of incrementality | **roadmap** — §6 |

Everything marked *built* runs today, offline:

```bash
python -m conveyer.attribution                       # synthetic market, end to end
python -m conveyer.attribution --journey-dir outputs/journey \
       --turns outputs/conversations/turn_features.parquet \
       --sales data/rms_brand_week.parquet --media data/media_spend.parquet
python -m conveyer.attribution --dump-ledger data/ledger.json   # then edit + --ledger
python tests/test_attribution.py                     # 29 tests
```

---

## 3 · The bridge

### 3.1 Bottom-up — count journeys, convert them, value them

```
exposed        = projected AI sessions exposed to the brand
                 ÷ device_coverage × panel_representativeness × window_recovery
online_orders  = exposed × P(reach cart | exposed) × cart_to_order
                 × cross_device_completion
purchases      = online_orders × ropo_multiplier
influenced_€   = purchases × basket_value × brand_match_precision
share          = influenced_€ ÷ actual sales            (RMS / Omnisales)
```

`P(reach cart | exposed)` is the **only measured link** and comes from
module 3, per brand-week, as a Beta posterior shrunk toward the category rate
(thin brand-weeks borrow strength instead of producing 0% or 100% from three
sessions). Everything else is a declared factor.

### 3.2 Top-down — start from the market and work in

```
share = reach(AI conversations among category purchase occasions)
        × P(brand exposed | conversation)
        × influence_given_exposure
```

The two routes share almost no assumptions. **Agreement is evidence; a
`reconciliation_ratio` far from 1 is a finding, and it is reported, not
smoothed.** In practice the top-down route is also the sanity check that stops
the bottom-up route from implying more AI conversations than there are
shoppers — a failure mode that is invisible until you compute the ratio.

### 3.3 Uncertainty, and what to buy first

Every factor is drawn from its declared range (triangular over
`[low, value, high]` — the honest shape when you have a best guess and bounds).
The chain is Monte-Carlo'd, cells are summed *inside* each draw (the factors
are shared across brands, so per-brand intervals do not add in quadrature),
and `sensitivity()` re-runs it with one factor varying at a time.

The default ledger's tornado, on synthetic market data:

```
ropo_multiplier      30% of the interval   [placeholder]
incrementality       30%                   [placeholder]
basket_value         15%                   [placeholder]
…
panel_sampling        <5%                  [measured]
```

**The panel is not the bottleneck.** More panel data, more categories and more
LLM platforms will not narrow the answer; one CPS path-to-purchase study will.
That is the single most actionable output of this whole layer, and it should
drive the data-partnership conversation.

---

## 4 · Filling the ledger

Each row is a task with an owner, not a modelling choice.

| factor | status | what retires it |
|---|---|---|
| `ropo_multiplier` | placeholder | **CPS** claimed-source / path-to-purchase, or a bespoke survey module. The biggest single lever for a CPG whose volume is mostly offline. |
| `basket_value` | placeholder | **RMS / Omnisales** value per buying occasion, joined through the brand↔hierarchy mapping. A lookup, not an assumption — available as soon as gap #8 is closed. |
| `incrementality` | placeholder | **the MMM** (§5), then experiments. The panel's exposed/unexposed lift gives an *upper bound* (`incrementality_from_lift`) — recorded next to the factor, never used as its value. |
| `influence_given_exposure` | placeholder | a survey question ("did the AI answer affect what you bought?") on the panel or a brand tracker. Only the top-down route uses it. |
| `cart_to_order` | benchmark | **our own shop**: panel-observed cart reach on our D2C domain vs actual orders in the same window. This is the cheapest calibration available and it needs no partner. Do it first. |
| `device_coverage` | benchmark | provider mobile-panel coverage, or a cross-source triangulation of desktop share of LLM usage. |
| `window_recovery` | assumed | user-level longitudinal browsing (gap #11) — measure directly instead of assuming. |
| `panel_representativeness` | assumed | demographics → post-stratification weights (`panel.py` computes them; the wide band collapses to a number). |
| `cross_device_completion` | assumed | the managed panel (demographics + transaction fields) if that partnership signs. |
| `brand_match_precision` | estimated | hand-label a sample of real pages. The synthetic corpus scores 1.0 and **must not** be quoted as the real value. |

Two of these — `cart_to_order` against our own shop, and `basket_value` from
RMS — need no external partner and no signature. They should be in flight now.

---

## 5 · The MMM

### 5.1 Do not build a second model

The organisation already converts activity into incremental sales with MMMs.
The right move is to make the AI channel a **variable in those models**, not to
stand up a competing one. What this project ships to the MMM team is
`design_frame()` — the regressor table at brand × market × week — plus the
diagnostics that say whether the variable is usable. `attribution/mmm.py`
contains a small reference implementation for validating the variable and
sizing the effect internally; it is not a production MMM and does not pretend
to be.

### 5.2 Three ways an AI variable goes wrong

**There is no spend.** ROI is a ratio to cost and the cost of an assistant
naming your brand is zero. So the output is a *contribution* (euros, share of
sales), and any "return" statement has to name the investment it is a return
on — content, PR, retailer feeds, product data quality — not a media buy.

**It is a trend.** Assistant usage grows monotonically, and so does every
other digital variable in a two-year window. A conversation *count* is
collinear with time and with paid digital; the model cannot separate them.
This is not theoretical — run `compare_ai_variables()` and watch it happen:

| AI variable | contribution | VIF | verdict |
|---|---|---|---|
| `ai_share_of_voice` | +0.94% | 1.4 | well-conditioned |
| `exposure_rate` | +0.56% | 1.5 | well-conditioned |
| `n_sessions_mentioned_projected` | **−0.53%** | 3.4 | wrong sign — unusable |

The intuitive variable (how many AI conversations mentioned us) comes back
*negative*. That means "the model cannot see this", not "AI hurts sales" — and
it is exactly the result that would get an AI-in-MMM initiative killed in a
readout. The default regressor is therefore **share of voice**: the brand's
share of category brand mentions, which moves with competitive dynamics rather
than adoption.

The cost of that choice is honest and worth stating: share of voice *discards*
the adoption signal, so it systematically **under**-attributes a channel whose
effect is still growing. On synthetic data with a planted 2.0% contribution it
recovers ≈0.9%. Neither variable is right alone. The resolutions, in order of
practicality:

1. **cross-sectional variation** — fit across markets and brands together, so
   the common time trend is not the only thing moving. This is the main reason
   to push for more markets in the data deal.
2. **an external anchor on the coefficient** — `prior_from_bridge()` turns the
   panel bridge's incremental estimate into an informative prior on β_ai, and
   the fit shrinks toward it. Same logic as experiment-calibrated MMM, with the
   panel funnel standing in for a geo test until real experiments exist.
   Always report the free fit *and* the calibrated fit; a large gap between
   them is the finding.
3. **longer history**, which we do not have and cannot wait for.

**It is endogenous.** Assistants mention brands that sell well, are widely
distributed and are written about — reverse causality straight into the
coefficient. Defences: controls (price, distribution, seasonality, brand fixed
effects) and the two-stage decomposition below.

### 5.3 Two-stage: media → AI visibility → sales

```
stage 1   ai_share_of_voice ~ media + PR + distribution + content
stage 2   sales             ~ ai_share_of_voice + media + controls
indirect effect of driver c = β(c → AI visibility) × β(AI → sales)
```

This is the practically useful output. You cannot buy a slot in an LLM answer,
but you *can* move what it says — through content, PR, retailer product data,
review volume. The two-stage split quantifies the return on doing so, and it
turns "AI is a channel we can't control" into a measurable lever. Both stages
are correlational, and the mediation arithmetic assumes no unmeasured
confounding; treat it as a decomposition of the fitted model, not a causal
claim.

### 5.4 The loop

```
   panel funnel ──► bridge: influenced € ──► prior on β_ai ──► MMM
        ▲                                                       │
        └────────── incrementality factor ◄─────────────────────┘
```

The bridge gives the MMM a prior; the MMM gives the bridge its incrementality
factor. Each iteration replaces a placeholder with an estimate. This is the
whole architecture in one picture, and it is why the two halves ship together.

---

## 6 · What should take a different approach

Not everything in modules 1–3 survives contact with the sales question
unchanged.

1. **Stop treating cart-reach as the outcome; treat it as an index.** For a
   CPG skincare brand, desktop cart-reach is a thin, biased slice of demand.
   Use it for relative comparisons (which conversations, which brands, which
   platforms convert better) and take the absolute level from RMS/CPS. The
   bridge is built this way; analyses and dashboards should follow.

2. **Session grain is the wrong grain for purchase.** A purchase can happen
   days after the conversation. The next-N-clicks window is a *session*
   construct. Ask the provider for **user-level longitudinal browsing** and
   move to a user × window outcome (7 / 14 / 30 days post-conversation). This
   single change retires `window_recovery` and materially improves the
   estimate — it is the highest-value data ask on the list.

3. **The logistic conversion model must become hierarchical.** Pooling
   multiple categories, markets and platforms in one flat logistic regression
   is invalid: conversion levels differ by an order of magnitude across them.
   Brand/category/platform random effects, once volume allows.

4. **The brand lexicon must become a category-parameterised taxonomy.**
   `conveyer/brands.py` is tuned to skincare. Multi-category means a per-
   category brand list *plus* the mapping to the sales hierarchy (gap #8) —
   and that mapping, not the lexicon, is the keystone of the whole layer.

5. **Scraping should shift further from page classification to entity
   resolution.** With 50 clicks × several platforms × several categories, URL
   volume grows 10–50×. What the sales layer actually needs from a URL is
   *which retailer, which brand, which commerce depth* — the existing
   domain-profile and URL-only machinery already delivers that without
   fetching. Full page classification should become the exception, not the
   default.

6. **`next_10_urls` is a column name, not a contract.** The trail parser
   already handles N events; the naming and the docs should stop implying 10.

---

## 7 · Getting to a product in ~12 weeks

The meeting set a ~3-month expectation with a staggered basic-then-
sophisticated approach. This is the staggering.

| weeks | deliverable | depends on |
|---|---|---|
| 1–2 | brand ↔ RMS/Omnisales mapping table; ledger populated with the two no-partner factors (`cart_to_order` from our own shop, `basket_value` from RMS) | sales extract access |
| 1–2 | schema v2 in production: platform / category / market dimensions through modules 1–3 | — (module 4 already accepts them) |
| 3–5 | `fact_ai_exposure_weekly` delivered to the MMM team — the first tangible artefact, useful before any attribution number exists | panel data, panel frame from the provider |
| 3–5 | panel projection with the provider's real universe/panel figures; demographics if they land | partner |
| 5–8 | **release 1: influenced-sales share**, with intervals, the reconciliation check and the assumption ledger printed on every output | above |
| 7–10 | AI variable inside the production MMM; reference fit + two-stage as validation; first incrementality estimate replaces the placeholder | MMM team, media spend history |
| 9–12 | **release 2: incremental-sales share**, calibrated loop, sensitivity-driven data roadmap | above |
| ongoing | content/GEO experiments for real causal identification (§below) | brand teams |

**Release 1 is shippable without the data partnership closing.** It runs on
the existing sample plus own-asset calibration, and its outputs state exactly
which factors are still assumed. That protects the three-month commitment from
the partnership timeline — which, per the meeting, is the main external risk.

### Causal identification, eventually

You cannot A/B-test ChatGPT. But you can run **content/GEO experiments**:
change brand content, retailer product data or PR in some markets, measure the
change in AI answer share with the query data, and measure the sales change.
That is a real instrument for the AI variable and the only clean road to
incrementality. A **synthetic-control across brands** (brands whose AI mention
share jumped vs matched controls) is the cheaper observational fallback.

---

## 8 · Open questions

**For SimilarWeb**
1. **User-level longitudinal browsing**, not just the next-N clicks after a
   prompt — the single highest-value item on this list (§6.2).
2. Panel frame per market: panel size, universe definition, and any weights
   they already apply. Without these, projection is guesswork.
3. Device coverage: desktop share of AI usage, and any mobile roadmap.
4. Demographics / geo — confirmed, inferred, or absent; and if inferred, the
   accuracy, because inferred demographics must never be presented as
   observed.
5. Purchase-confirmation URL patterns per retailer — would tighten the outcome
   proxy from cart-reach to order.
6. Brand/entity extraction at the query level, or do we do it ourselves?

**For the managed panel ("Generate")**
7. Confirm what "demo and T.O." covers — demographics plus which transaction /
   outcome fields? If it includes observed purchases, it can calibrate
   `cart_to_order` and `cross_device_completion` directly, which is worth more
   than the extra volume.
8. Panel size, markets, categories, and overlap with the SimilarWeb panel.
9. Blob-storage access and the sample's schema, to size the ingest work.

**Internal**
10. RMS / Omnisales / CPS: which grain, which markets, which history, and who
    owns the extract? The bridge needs brand-market-week value; CPS needs to
    answer buyers/occasions and the offline share.
11. Existing MMMs: grain, vendor, model form, variable list, and whether a new
    regressor can be added mid-cycle. Which markets/brands have the longest
    clean history — those are where the AI variable has any chance.
12. Own D2C analytics access for the `cart_to_order` calibration (§4).
13. The brand↔product-hierarchy mapping — who owns it, does it exist already?

---

## 9 · Limitations to state on every output

* **Influenced ≠ incremental.** Repeated because it is the mistake that
  matters.
* The estimate is a **structure with declared assumptions**, not a
  measurement, until the placeholder factors are retired. Every report prints
  which ones are still open.
* Panel data is **desktop-only, observational, and self-selected into
  exposure**; the exposure effect is an upper bound.
* Factors are drawn **independently** — no dependence structure is known, and
  inventing a correlation matrix would be one more undeclared assumption. The
  consequence (variances add) is stated rather than hidden.
* The category denominator currently defaults to the sum of tracked brands,
  which understates the category and biases the top-down reach upward. A real
  RMS category total replaces it.
* Cells where the chain implies more influenced sales than the brand sold are
  **flagged and must not be published** — that is the factor chain failing, and
  it is a diagnostic, not a rounding issue.

---

## 10 · Output tables

| table | grain | what it is for |
|---|---|---|
| `fact_ai_exposure_weekly` | week × market × category × platform × brand | the panel's view of AI exposure; **the artefact to hand the MMM team** |
| `fact_ai_category_weekly` | week × market × category × platform | conversation volume and reach; the share-of-voice denominator and the top-down anchor |
| `ai_influenced_sales` | brand-week | influenced / incremental value and share with p05–p95, both routes, reconciliation ratio, publish flag |
| `ai_attribution_totals` | metric | the headline numbers with intervals |
| `ai_attribution_sensitivity` | factor | the tornado — which unknown owns the answer |
| `ai_attribution_factors` | factor | the ledger snapshot the run used: value, range, source, status |
| `mmm_coefficients` | feature | contributions and shares from the reference MMM |
| `mmm_diagnostics` | feature | VIF — whether the AI coefficient can be quoted at all |
