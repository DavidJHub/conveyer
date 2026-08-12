# Data dictionary — input & every output table

One conversation input (plus optional market data for module 4), sixteen
outputs. All outputs are parquet with **pinned pyarrow schemas**; regenerate
the funnel with `python -m conveyer.pipeline` and the attribution layer with
`python -m conveyer.attribution`.

```mermaid
flowchart LR
    IN[("data/conversations.parquet")] --> M1[conversations.py]
    M1 --> T1[/"turn_features"/] & C1[/"conversation_features"/]
    IN -. link columns .-> M2[scraping/]
    M2 --> P1[/"scraped_pages"/] & P2[/"scraped_products"/]
    T1 & C1 & P1 & P2 --> M3[journey.py]
    M3 --> E[/"funnel_events"/] & J[/"journey_features"/] & TR[/"funnel_transitions"/] & MC[/"model_coefficients"/]
```

---

## 0 · Input — `data/conversations.parquet` (grain = one user↔LLM turn)

| column | dtype | description |
|---|---|---|
| `session_id` | str | conversation/session identifier |
| `message_id` | str | unique id for the user/LLM turn |
| `user_id` | str | anonymised panelist identifier |
| `prompt_datetime` | str | prompt timestamp |
| `question` / `answer` | str | user prompt / model response |
| `a_links_source` | list | links surfaced in the response |
| `ai_click` | list | links clicked directly from the AI response |
| `next_10_urls` | list\<struct\> | post-prompt navigation: up to 10 `{request_time (epoch-ms str), requested_site}` |

Loading (`conveyer.ingest.load_conversations`) tolerates numpy-array,
Python-list and stringified-repr link cells, parses timestamps to
`prompt_dt`, and derives `session_pos` (1-based, time-ordered). When the file
is absent, `make_synthetic_conversations` produces the same schema **plus
ground truth** (`gt_archetype`, `gt_followed_rec`, `gt_converted`) and an
offline page corpus, so the full pipeline runs and scores itself.

---

## 1 · Module 1 outputs — `outputs/conversations/`

### `turn_features.parquet` (grain = turn; 25 cols)

| group | columns |
|---|---|
| identity | `message_id` PK · `session_id` · `user_id` · `session_pos` · `prompt_datetime` |
| text | `question` · `answer` |
| intent | `intent` (comparison/purchase/troubleshooting/routine/informational) · `asks_recommendation` |
| sentiment | `question_sentiment` · `answer_sentiment` (−1..1; lexicon or transformers) |
| brands | `brands_q` · `brands_a` · `brands_unsolicited` (agent-introduced — the exposure) · `brands_endorsed` · `n_brands_answer` · `is_recommendation` |
| stages | `funnel_stage` (keyword) · `journey_stage` (HMM-smoothed) |
| topics | `topic_id` · `topic_label` |
| links | `links_cited` · `links_clicked` · `n_links_cited` · `n_ai_clicks` · `n_trail_events` |

### `conversation_features.parquet` (grain = session; 22 cols)

`session_id` PK · `user_id` · `n_turns` · `duration_seconds` · `first_intent` ·
`dominant_intent` · `asks_recommendation_share` · `mean_question_sentiment` ·
`mean_answer_sentiment` · `sentiment_trend` · `brands_discussed` ·
`brands_recommended_unsolicited` · `n_brands_discussed` ·
`n_unsolicited_recommendations` · `any_recommendation` · `n_links_cited` ·
`n_ai_clicks` · `n_trail_events` · `max_funnel_stage_idx` · `max_funnel_stage` ·
`journey_path` (e.g. `Discovery > Evaluation > Intent`) · `topic_label`

---

## 2 · Module 2 outputs — `outputs/scrape/`

`scraped_pages.parquet` (57 cols, grain = URL) and
`scraped_products.parquet` (26 cols, grain = product-on-page), each with a
crash-safe `.jsonl` sidecar. Full column documentation, taxonomy, ER diagram
and caveats: **[SCRAPED_PAGES_SCHEMA.md](SCRAPED_PAGES_SCHEMA.md)**.

---

## 3 · Module 3 outputs — `outputs/journey/`

### `funnel_events.parquet` (grain = one behavioural event; 16 cols)

| column | type | description |
|---|---|---|
| `session_id` / `message_id` | str | which turn produced the event |
| `event_type` | str | `cited` (agent exposure) · `ai_click` (direct click) · `trail_visit` (post-turn navigation) |
| `url` | str | the page |
| `position` | int32 (nullable) | 1-based slot in the trail (trail visits only) |
| `dwell_seconds` | float64 (nullable) | gap to the next request; last event never gets one |
| `page_category` / `page_subtype` / `funnel_stage` / `seller_type` | str | module 2's classification (URL-only fallback ⇒ 100% coverage, see `page_coverage`) |
| `page_brand` | str | canonical brand the page is evidence for (domain, OG, or extracted product brands) |
| `brand_match` | str | vs the turn's mentions: `unsolicited_rec` · `endorsed` · `requested` · `none` |
| `followed_agent_link` | bool | user action on an agent-surfaced link (never true for `cited` rows) |
| `commerce_depth` | int32 | 0 none · 1 shopping page · 2 cart · 3 checkout/order |
| `is_study_relevant` | bool | module 2's topical verdict |
| `page_coverage` | str | `pages` (scraped) or `url_only` (heuristic fallback) |

### `journey_features.parquet` (grain = session; 21 cols) — the model table

| group | columns |
|---|---|
| conversation side | `n_turns` · `asks_recommendation_share` · `mean_question_sentiment` · `mean_answer_sentiment` · `n_unsolicited_recommendations` · `n_links_cited` · `max_funnel_stage_idx` |
| behaviour side | `n_visits` · `n_shopping_visits` · `n_relevant_visits` · `total_dwell_shopping` · `mean_dwell` |
| exposure (treatment) | `followed_agent_link` · `visited_recommended_brand` · `followed_recommendation` (either) |
| outcome (proxy) | `max_commerce_depth` · `reached_shop` · `reached_cart` · `reached_checkout` · `converted` (`≥ JourneyConfig.conversion_stage`, default cart) |

### `funnel_transitions.parquet` (long form)

`from_stage` · `to_stage` · `count` · `prob` — stage sequence = smoothed
conversation stages in turn order, then visited pages' stages in trail order;
`prob` rows sum to 1 per `from_stage`.

### `model_coefficients.parquet`

`feature` · `coef_standardized` · `odds_ratio_per_sd` — the logistic
conversion model on standardized features; the `followed_agent_link` /
`visited_recommended_brand` rows are the headline **weight of agent
recommendations** (see [FUNNEL_MODEL.md](FUNNEL_MODEL.md) for the
identification caveats).

---

## 4 · Module 4 outputs — `outputs/attribution/`

Methodology, the assumption ledger and the open data questions:
**[SALES_ATTRIBUTION.md](SALES_ATTRIBUTION.md)**.

### Extra inputs (optional; synthetic when absent)

| table | grain | columns |
|---|---|---|
| sales | brand-week | `week` · `market` · `category` · `brand` · `sales_value` · *(optional)* `price_index`, `category_sales_value`, `category_occasions` |
| media | brand-week | `week` · `market` · `brand` · any columns ending in `_spend` |

### `fact_ai_exposure_weekly.parquet` (grain = week × market × category × llm_platform × brand)

| group | columns |
|---|---|
| dimensions | `week` (session's **first** prompt week, Monday-anchored) · `market` · `category` · `llm_platform` |
| counts | `n_sessions_category` · `n_sessions_mentioned` · `n_sessions_mentioned_visited` · `n_sessions_unsolicited` · `n_sessions_cited` · `n_sessions_visited` · `n_sessions_followed` · `n_sessions_converted` |
| rates | `ai_share_of_voice` (sums to 1 within a cell) · `exposure_rate` · `visit_rate_given_exposure` |
| projected | `*_projected` (+ `_lo`/`_hi` where a denominator exists) — present **only** when a `PanelFrame` was supplied |

`n_sessions_converted` attributes each session's conversion to **one** brand
(the deepest-commerce page it reached), so brand conversions can never sum
above the session count.

### `fact_ai_category_weekly.parquet` (grain = week × market × category × llm_platform)

`n_sessions` · `n_users` · `n_sessions_followed` · `n_sessions_converted` ·
`conversion_rate_proxy` · `n_sessions_projected` · `n_users_projected`. The
share-of-voice denominator and the top-down route's reach anchor.

### `ai_influenced_sales.parquet` (grain = brand-week)

Dimensions + `sales_value` · the panel counts it used · and, for each of
`influenced_value`, `influenced_share`, `incremental_value`, `topdown_share`,
the `p05` / `p50` / `p95` of the Monte-Carlo distribution · plus
`reconciliation_ratio` (bottom-up ÷ top-down) and `share_exceeds_sales` —
**rows flagged true must not be published**; they mean the factor chain is
inconsistent for that cell.

### `ai_attribution_totals.parquet` · `ai_attribution_sensitivity.parquet` · `ai_attribution_factors.parquet`

`metric` · `unit` · `p05` · `p50` · `p95` — the headline numbers ·
`factor` · `status` · `low_x` · `high_x` · `interval_width_x` ·
`variance_share` — the tornado, "this assumption owns X% of the uncertainty" ·
`name` · `value` · `low` · `high` · `unit` · `status` · `source` · `note` —
the ledger snapshot the run used.

### `mmm_coefficients.parquet` · `mmm_diagnostics.parquet`

`feature` · `coef` · `contribution` · `contribution_share` ·
`feature` · `vif` · `flag`. A VIF above ~10 on the AI variable means the
coefficient is not separable from the rest of the mix and must not be quoted.

---

## Gotchas checklist

1. `next_10_urls` arrives as **numpy arrays of dicts** from parquet — always
   go through `ingest.as_records` / `trail_events`.
2. The **last trail entry has no dwell** (no successor); dwell includes idle
   time — treat as an upper bound on attention.
3. `converted` is a **proxy** (reaching cart/checkout in the trail), not a
   confirmed purchase.
4. Brand matching runs on the **canonical lexicon** (`conveyer.brands`);
   extend it before analysing a niche brand, or its mentions/pages won't link.
5. Exposure strata are observational — `followed_recommendation` users
   self-select (see FUNNEL_MODEL.md §limitations).
6. Module 4's euro figures are **not measurements** while the ledger still
   holds `placeholder` factors; every run prints which ones. `influenced` is a
   touchpoint claim, `incremental` a causal one — never quote one as the other.
7. The bridge joins the panel's brand lexicon to the sales hierarchy on a
   plain string key and **fails loudly** when nothing matches. That is the
   missing mapping table, not a bug to work around.
