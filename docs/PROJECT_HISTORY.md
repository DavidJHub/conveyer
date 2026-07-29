# Conveyer — Project History & Methods Recap

> **Purpose of this page.** The complete, meticulous record of how this project
> got to where it is: every approach we took, every method that failed, why it
> failed, and what replaced it. Written to be imported into Confluence as-is
> (plain Markdown: headings, tables, code blocks) and edited by anyone on the
> team. The companion page [`PROJECT_WIKI.md`](PROJECT_WIKI.md) describes the
> *current* system; this page explains **how and why it became that system**.
>
> Sibling references: [`SCRAPING_MODULE.md`](SCRAPING_MODULE.md) (module 2
> internals), [`SCRAPED_PAGES_SCHEMA.md`](SCRAPED_PAGES_SCHEMA.md) (output
> schemas), [`FUNNEL_MODEL.md`](FUNNEL_MODEL.md) (module 3 methodology),
> [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) (all tables),
> [`STATE_OF_THE_ART.md`](STATE_OF_THE_ART.md) + [`references.bib`](references.bib)
> (literature), notebooks
> [`01_funnel_pipeline.ipynb`](../notebooks/01_funnel_pipeline.ipynb) /
> [`02_page_classifier.ipynb`](../notebooks/02_page_classifier.ipynb).

---

## 1 · Mission

**From ChatGPT shopping conversations to a predictive funnel model.** Given a
corpus of agent-assisted skincare shopping conversations — each turn carrying
the user's question, the agent's answer, the links the answer surfaced, direct
clicks on them, and the next pages the user browsed — measure two things:

1. the **conversion rate** of these journeys, and
2. the **weight of agent recommendations** (the brands the agent names, the
   links it shows) in the user's decision to buy.

Context that makes this worth measuring: LLM referrals to e-commerce are now a
measurable acquisition channel (Kaiser & Schulze 2026), and LLM steering has
been shown to nearly triple sponsored-product selection (Salvi et al. 2026).
The full literature review (~35 references: agentic commerce, recommendation
inference, funnel modeling, web-page classification, weak supervision) lives
in `STATE_OF_THE_ART.md`.

### 1.1 The data

Two shapes of input, both supported by the same pipeline:

* **The single input file** (`simweb_input_file.parquet`) — one row per
  conversation turn:

  | column | meaning |
  |---|---|
  | `session_id`, `message_id`, `user_id`, `prompt_datetime` | identity & time |
  | `question`, `answer` | the turn's text |
  | `a_links_source` | links cited inside the answer (the recommendation exposure) |
  | `ai_click` | links clicked directly from the answer |
  | `next_10_urls` | the browsing trail: up to 10 `{request_time, requested_site}` events after the turn |

* **The SimilarWeb star schema** (when available) — `dim_digital_site` (URL →
  `page_type`/`category` weak labels), `fact_ai_click_through` (surfaced URL ↔
  turn linkage + `was_recommended`/`was_visited`/`resulted_in_purchase`),
  `fact_ai_recommendation` + `fact_ai_concept` (the entities the agent
  recommended, with `entity_context` text and BRAND/CATEGORY concepts),
  `dim_ai_funnel` (funnel stage vocabulary).

A structural fact that shaped the whole design:
`fact_ai_click_through.recommendation_id` is **100 % null** in the source
data, so a URL attaches to a *turn*, never directly to a recommended entity.
Connecting pages to the products mentioned in chat therefore has to be
**inferred** — which is what module 2's matcher does (§5.6).

### 1.2 A key early insight: dwell from the trail

`next_10_urls` events carry `request_time` timestamps. The gap from one
request to the next is how long the user stayed on a page before moving on —
a behavioural relevancy/retention signal that costs nothing to compute. Two
rules keep it honest: the **last** trail event has no successor, so it never
receives a dwell (null, not guessed); and dwell is documented as an **upper
bound** on attention (the clock runs while the user idles). Per URL the
pipeline aggregates `mean_dwell_seconds`, `total_dwell_seconds` and
`mean_trail_position` (1 = the first page opened after the answer).

---

## 2 · Prehistory — what the repo was before the funnel mission

The repository predates the funnel mission. Its first life (commits `9f73d0f`
→ `3bd12f7`) was a **skincare conversation clustering pipeline**: ingest +
embedding backends (`conveyer/models.py`: Voyage → OpenAI →
sentence-transformers → TF-IDF+SVD auto-resolution, a pattern the rest of the
project reuses), k-means and alternative clustering methods, a visualization
dashboard, then a reorganization around an ACES-simulator integration and a
research-question master notebook, a first project wiki with hypotheses and an
HMM spec (`46eb356`), and the state-of-the-art review (`cdc0af3`).

That era's lasting contributions: the **auto-resolving model layer**, the
**offline-first, synthetic-data-with-ground-truth** testing philosophy, the
**HMM journey-stage idea**, and the literature base. Much of the rest was
trial-and-error scaffolding, and when the funnel mission crystallized the
instruction was explicit: *"this whole project is a mess. clean everything
that is not being used."* The restructure (`d904119`) deleted everything not
serving the funnel model and organized the survivors into three modules.

---

## 3 · Architecture — three modules and why this decomposition

| module | question it answers | outputs (all pinned pyarrow schemas) |
|---|---|---|
| 1 · `conveyer.conversations` | **what was said** — intent, sentiment, topics, brand mentions (incl. *unsolicited* = the exposure), funnel stages (keywords + HMM smoothing) | `turn_features`, `conversation_features` |
| 2 · `conveyer.scraping` | **where the user landed** — fetch, classify, extract & match products for every surfaced/visited URL | `scraped_pages` (62 cols), `scraped_products` (28 cols) |
| 3 · `conveyer.journey` | **did it move the needle** — events, conversion with credible intervals, funnel transitions, the predictive models | `funnel_events`, `journey_features`, `funnel_transitions`, `model_coefficients` |

Plus `conveyer.pipeline` (the 1→2→3 orchestrator), `conveyer.dashboard`
(self-contained HTML insight dashboard, inline base64 charts, no JS),
`conveyer/brands.py` (the **shared canonical brand lexicon**, ~60 brands with
aliases and domain cores — the single place where text mentions and domains
resolve to the same brand), and the two executable notebooks (generated by
`notebooks/_build_notebook_0X.py`, executed in CI fashion on every rebuild).

**Why offline-first with synthetic ground truth.** Every module runs
end-to-end with no data files, no network and no API keys, on a synthetic
corpus that *plants known truth* (page categories, seller types, expected
product coincidences, session archetypes). This is not a convenience — it is
the project's validation instrument: the pipeline **scores itself against the
planted truth on every run**, so a regression in the measurement grammar
cannot ship silently. It caught real regressions repeatedly (§7).

**Why pinned pyarrow schemas everywhere.** Pandas round-trips silently mutate
types (nullable ints become float64-with-NaN, lists become numpy arrays).
Every output table is written with an explicit `pyarrow` schema and typed
arrays; `validate.py::_coerce_rows` re-pins types after any pandas round-trip.
One production bug (NaN → int32 arrow cast failure in journey) was fixed by
`_null_if_nan` coercion and the lesson generalized.

### 3.1 Module 1 — conversations

Turn-level: `intent` (informational / purchase / comparison), lexicon
sentiment (auto-upgrades to a transformers pipeline when installed), topics
(k-means on the embedding layer; BERTopic when installed), **brand
extraction** against the shared lexicon with the distinction that drives
everything downstream: `brands_unsolicited` (the agent introduced the brand,
user never asked — the treatment of interest) vs `brands_endorsed` vs
`brands_q`. Funnel stage per turn from keyword rules, then **HMM-smoothed**
into `journey_stage` (a user's stated stage is noisy turn-to-turn; the HMM
enforces plausible stage dynamics). Conversation-level aggregates include
`asks_recommendation_share`, sentiment means/trend, `journey_path` (the
smoothed stage sequence) and `max_funnel_stage_idx`.

### 3.2 Module 3 — journey & the funnel model

* **Events**: `cited` (exposure), `ai_click`, `trail_visit` (with dwell), each
  joined to module 2's page labels; URLs never scraped fall back to URL-only
  classification (`classify_url`) so coverage is 100 %.
* **`brand_match`** ties a visited page's brand back to the conversation:
  `unsolicited_rec` / `endorsed` / `requested` / `none`. Retailer PDPs carry
  the brand only in product JSON-LD, so product-level brands are folded into
  the page lookup (a real bug found when `visited_recommended_brand` lift came
  out 0/NaN).
* **Conversion** is an explicit, documented **proxy**: the trail reached
  commerce depth ≥ cart (`shopping`=1, `cart`=2, `checkout`=3; configurable
  via `JourneyConfig.conversion_stage`). No purchase confirmation exists in
  this schema; we say so everywhere instead of pretending.
* **Uncertainty**: conversion rates per exposure stratum carry **Jeffreys 95 %
  intervals**; `lift = rate(exposed)/rate(unexposed)`.
* **Two logistic models, deliberately**: the **exposure model** contains
  conversation features + exposure flags and **no post-treatment mediators**;
  the **full model** adds behavioural features (visits, dwell) for maximum
  predictive power. This split exists because the first single-model attempt
  produced *negative* exposure coefficients — visit counts are consequences of
  following a recommendation, and conditioning on them absorbed the effect
  being measured (a textbook post-treatment-bias failure we hit in practice,
  §7.5). Association, not causation, is stated on every surface.
* **Validation**: the synthetic generator plants five session archetypes
  (converter-via-rec 25 %, browser-via-rec 25 %, organic-converter 10 %,
  no-follow 30 %, researcher 10 %) with **deterministic largest-remainder
  assignment** (a random draw once skewed a 24-session run so badly the
  archetype table was unrecoverable). Current recovery at n=40:
  `conversion_accuracy = 1.0`, `followed_accuracy = 1.0`; lift
  `followed_recommendation 2.5 · followed_agent_link 2.5 ·
  visited_recommended_brand 2.2`.

---

## 4 · Module 2, part I — making the scraper survive reality

Module 2 consumed most of the project's engineering effort, because it is
where the real world pushes back. Chronological account:

### 4.1 First version (`1f3973a`) and the "runs forever" failure

The first scraper was conventional: `requests` + `robotparser`, a thread
pool, batch results, one parquet at the end. On real data it **hung
indefinitely** ("cell 2 runs forever without any error"). Post-mortem found
four independent causes, each fixed structurally (`3a12592`):

| failure | root cause | fix |
|---|---|---|
| infinite hang | stdlib `RobotFileParser.read()` uses urllib **with no timeout** — one dead host blocks forever | fetch robots.txt ourselves through the session with a bounded timeout; only *parse* with robotparser |
| slow-dribble hang | socket timeouts only bound the gap *between bytes*; a server trickling 4 KB/200 ms never times out | **`hard_timeout`: a wall-clock budget per URL** covering robots + all retries + body streaming |
| convoy effect | the per-domain rate limiter slept **while holding the global lock**, serializing every worker behind one domain | reserve-a-slot pattern: compute the slot under the lock, **sleep outside it** |
| all-or-nothing output | results collected as a batch; a crash lost everything | `as_completed` streaming + **line-per-record JSONL** persistence + `resume=True` |

A fifth, quieter bug from the same era: `next_10_urls` arrived as numpy
arrays from parquet and the naive `to_list` silently dropped them — the trail
(and all dwell data) vanished. `as_records` now handles ndarray / list /
str-repr forms explicitly.

### 4.2 Memory: the 10–26 GB lesson (`644033e`, `86e1606`)

Long real runs died at a consistent page count. Two causes:

1. **Batch materialization** — the final parquet was built from an in-memory
   list of every row. Fix: **parquet part files** — every `checkpoint_every`
   pages the buffer is flushed to `…_parts/part-NNNNNN.parquet` and cleared;
   on exit the parts are **streamed** through a `ParquetWriter` into the final
   file (never all in RAM); recovery reconciles JSONL ↔ parts in bounded
   chunks (`_recover_state`), and pre-parts outputs migrate automatically.
2. **The real OOM**: `iter_fetch` kept every completed `Future` in a list for
   the whole run — and a completed Future retains its result, which contains
   up to 2 MB of HTML. ~13k pages ≈ 10–26 GB. Fix: a **sliding submission
   window** (~2× workers in flight) that drops each Future reference the
   moment its result is yielded. Raw HTML lives on disk (one cache file per
   URL, `html_path` column points at it); offline replays read those files
   lazily instead of preloading a dict of the whole corpus (an earlier design
   that was itself a machine-killer).

Crash-consistency detail worth knowing: for each page, **product rows are
written first and the page line last** — the page line is the commit marker,
so a crash between writes can never leave orphan products, and recovery
re-admits only products whose page line landed.

### 4.3 Domain-level reuse: "it's taking really long" (`0b1c1a2`)

Real corpora hit the same domains thousands of times. Three layers keep long
runs fast **without ever copying a page-level label across a domain** (an
amazon cart is not an amazon PDP — only *domain-level* knowledge is reused):

1. **Circuit breaker** — after `domain_failure_threshold=3` consecutive
   failures with no success, a domain's remaining fetches are skipped
   instantly (`fetch_status="circuit_open"`).
2. **Smart fetch policy** — URL-decided pages (cart, checkout, order,
   wishlist, SERP, site-search — later also tool/account/local) are **never
   fetched**: their URL tokens already decide them and their bodies carry
   nothing extractable. Content fetches are capped at
   `max_fetch_per_domain=25`; the sample teaches the domain.
3. **Learned domain profiles** — `{relevance, seller, platform, n_pages}` per
   domain, learned from the pages that were fetched, persisted to
   `domain_profiles.json`, and used to classify the domain's remaining URLs
   network-free (`domain_profile` signal).

### 4.4 Block resilience: never walk away empty-handed

Added in the big upgrade (`9e6c380`) after the requirement *"we need the
scraping to avoid robot block… even at least a header from the html, anything
can be useful for the model"*:

* **Browser header profile by default** (`browser_headers=True`): a realistic
  Chrome header set. Many CDNs 403 unknown User-Agent strings at the edge
  before the origin sees the request. This is **presentation only** — robots
  compliance, rate limits, the circuit breaker and wall-clock caps are
  unchanged (and a latent bug was fixed en route: robots.txt fetches dropped
  the port from the URL).
* **Blocks are terminal**: 401/403/406/451 → `fetch_status="blocked"`, never
  retried (a wall answers the same way every time; retrying is hammering).
  429 honors `Retry-After` — including the RFC 9110 **HTTP-date** form; the
  first implementation parsed only bare seconds, and `float("")` on a missing
  header silently disabled *all* 429 retries (found in adversarial review,
  §6).
* **Salvage everything**: response headers are captured on every answer
  (`response_headers` column, JSON) plus a bounded error-body excerpt used
  **only** for fingerprinting — an error body must never masquerade as page
  content.
* **Platform fingerprinting** (`fingerprint.py`): headers and markup identify
  the hosting platform — `x-shopify-stage` ⇒ Shopify, `x-wix-request-id` ⇒
  Wix, `wp-content/plugins/woocommerce` ⇒ WooCommerce, etc. A commerce
  platform on an unknown domain is storefront evidence that **survives a bot
  wall**: a blocked Shopify store's `/products/…` URL still classifies
  `shopping · pdp · brand_owned`. The fingerprint is remembered in the domain
  profile so it keeps working after the circuit opens. Hard-won constraints
  from review (§6): the fingerprint is **structural evidence only** — it must
  never make an off-topic store study-relevant, and it is suppressed on
  curated retailer domains (Credo Beauty runs Shopify but is a retailer, not
  a DTC brand site). WAF vendors (Cloudflare, Akamai) are recorded as
  infrastructure and never count as commerce evidence.

### 4.5 The fallback chain (why nothing stays unclassified)

Orchestrated in `_process_one`, recorded in `fetch_scope`:

1. **`page`** — the URL itself was fetched; all channels play.
2. **`stripped`** — the exact link failed → retry **without the query
   string**. Motivating case: real trail URLs like
   `zicail.com/how-to-choose-the-best-sunscreen/?preview_id=10472&preview_nonce=…`
   error on a stale WordPress preview nonce while the bare article loads.
   Because the stripped variant is the *same document*, its content counts as
   the page's **own** (it can prove relevance or `unrelated`; products ARE
   extracted; the model channel sees it; `html_path` points at its cache).
   Guard: never fires when the query **selects** the content (`?q=`,
   `?variant=`, `?asin=`, `?page=`, `?v=` … — exact key match, so
   `preview_id` stays strippable).
   **"Failed" includes soft errors** — this was a second, live-diagnosed
   iteration (`4caf9e1`): the reported URL never fails at the HTTP level; it
   answers **HTTP 202 with a ~200-byte anti-bot challenge shell** (and
   WordPress preview failures elsewhere answer 200 with the error page, or
   redirect to `wp-login.php`). `_soft_error_page` detects auth-redirect
   final URLs, error-phrase titles on thin pages, and near-empty shells. A
   soft-error body is **never** classified as the page's content: if the
   stripped variant is healthy it takes over; otherwise the shell is dropped
   and the chain continues (a "Page not found" title must never label a page
   `unrelated`). Near-empty bodies are also never cached, so recoveries
   self-heal once a site stops challenging.
3. **`base`** — fetch `scheme://host/` (deep link on x.com → x.com/) so
   domain-level content still informs relevance/category. Products are
   deliberately **not** attributed from a stand-in homepage.
4. **`directory`** — nothing fetchable at all → the offline **domain
   directory** (`directory.py`: built-in seed + external JSON at
   `directory_path`) supplies a stored description of the *site* as stand-in
   content. Stand-ins supply relevance but can never prove `unrelated`,
   never yield products, and never mint a category the URL didn't earn.
5. **`none`** — URL + domain + prior heuristics alone.

---

## 5 · Module 2, part II — the classifier, iteration by iteration

### 5.1 The taxonomy

The study asked for three buckets — *catalogue / brand-landing* (Discovery),
*shopping* (external retailer **or** brand-owned), *unrelated* — and invited
proposals. We kept the three and added four the data demands (SimilarWeb's
own `page_type` distribution is dominated by search; editorial "best-of"
content and community/UGC are where evaluation actually happens):

| `page_category` | funnel stage | origin |
|---|---|---|
| `brand_landing`, `catalogue` | Discovery | requested |
| `shopping` | Intent (cart/checkout → Purchase; order → Post-Purchase) | requested |
| `editorial` | Evaluation | proposed |
| `search` | Intent | proposed |
| `community` | Evaluation | proposed |
| `reference` | Awareness | proposed |
| `unrelated` / `unknown` | Irrelevant | requested / escape hatch |

Two design decisions that recur everywhere:

* **`seller_type` is an orthogonal axis** (`brand_owned`/`retailer`/`na`), so
  `shopping` and `catalogue` don't each fork in two.
* **`unknown` ≠ `unrelated`**: `unrelated` is a *confident* off-topic
  judgement that requires the page's own content; `unknown` is the honest "we
  could not decide". Collapsing this distinction was the root of several
  early misclassifications.

Structural `page_subtype` (pdp, cart, checkout, order, wishlist, serp,
collection, marketplace, article, forum, wiki, homepage, … later + `local`,
`tool`, `account`) survives the `unrelated` collapse — structure and
topicality are separate axes.

### 5.2 Failure: the amazon cart classified `unknown` — and the multimodal redesign (`8c58acb`, `ed56666`)

Obvious URLs (`amazon.com/gp/cart/view.html?ref_=nav_cart`) came out
`unknown`: the then-current relevance collapse demanded skincare evidence
from pages that *by nature* carry none. The redesign made the classifier
**multimodal** — independent evidence channels each vote subtype weights, the
argmax wins with a softmax confidence over category-level sums, and
`classification_signals` records which channels fired so **every label is
auditable**. With it came the two rules that fixed the cart:

* **Topic-neutral subtypes** (cart, checkout, order, serp, site_search,
  marketplace, homepage) are journey infrastructure — never demoted for
  lacking beauty tokens. But neutrality must be **earned** by a structural
  vote (URL/markup/prior, or a decisive ≥ 2.0 domain/directory role): the
  weak retailer catch-all (0.6) must not turn `sephora.com/careers` into a
  journey page.
* **Transactional URL tokens are self-evident commerce** anywhere: an
  unfetchable `/checkouts/c/<token>` on an unheard-of Shopify store is a
  checkout.

An adversarial self-review of that redesign confirmed **a dozen defects**
before they shipped, among them: Shopify's `/checkouts/c/<token>` being
outvoted by the single-letter `/c/` collection token (fix: `checkouts?` in
the decisive regex); `listing`/`marketplace` sitting in the neutral set and
granting blanket relevance to `/careers` and `/prime`; the missing
`/gp/aw/c` mobile-cart token; the dropped `order` token (→ new `order`
subtype mapped to Post-Purchase); `wishlist` counted as Purchase (→ its own
subtype, Intent — saved-for-later is not a purchase event); divergent
root-path sets between rules; SERPs with a visible `?q=` being treated
topic-neutral instead of judged by their query text; and the Purchase funnel
bump applying to non-shopping pages. A companion rule with a story: **a
decisive transactional URL token wins the subtype outright**, because
Amazon's `/cart/add-to-cart/…` page scored `pdp 3.5` from its own furniture
(prices, checkout buttons are *expected* on a cart) vs `cart 2.5` and
collapsed to `unrelated`. The `validate.py` URL-rule double-check (report /
`--apply` repair, run automatically after every scrape) exists so that class
of regression can't ship silently again.

### 5.3 Failure: haircare classified `unrelated` — the relevance axis (`ca715b6`)

Real counterexample: `jbca.com/products/thickening-strengthening-restorative-conditioner`
— structurally perfect (`pdp`, ProductGroup JSON-LD, price, add-to-cart) but
`skincare_relevance = 0.0` → `unrelated`. Root cause: the topical vocabulary
knew only *skincare* words (serum, retinol, SPF…), and the page speaks pure
*haircare*. The study's umbrella is **beauty / personal care**.

The fix and its craft: the lexicon was widened to haircare, bodycare and
cosmetics/fragrance with the constraint that **every term must be unambiguous
at a single hit** (one keyword already clears the relevance bar), so common
polysemes stay out — *foundation* (charity), *powder* (snow), *primer*
(paint), *blush* (verb), *curl* (bicep), *cologne* (the city), *soap*
(opera), bare *cosmetic*/*beauty* — compounds carry those meanings instead
(`curl cream`, `beauty routine`, `cosmetics` plural only, `makeup` one word
only), and `conditioner` carries negative lookbehinds for `air `/`air-`.
`ScrapeConfig.extra_relevance_terms` lets a run extend the vocabulary with
plain phrases, no code edit. The stored column keeps its historical name
`skincare_relevance` **deliberately** (schema stability across existing
parquets); docs state the widened meaning.

**Retroactivity** mattered as much as the fix: labels are burned into the
parquet at scrape time and `resume` never revisits a done URL, so a
vocabulary improvement cannot reach old rows on its own. `validate.py
--reclassify` re-runs the full classifier over stored `unrelated`/`unknown`
rows using the **cached HTML each row's `html_path` points at** (tolerating
Windows `\` separators and moved cache dirs), falling back to the stored
title/excerpt columns — nothing re-fetched. Repairs are **rescue-only** (a
row changes only when the fresh verdict is study-relevant with a real
category) and audited (`+reclassified`). Chat brands are deliberately *not*
replayed from `brand_detected` during reclassification — that column mixes in
page brands, and replaying it would let a page vouch for itself.

### 5.4 Failure: every Google link is "search" — service routing (`9e6c380`)

`google` sat in the curated search-domain list, so **every** Google URL voted
`serp` — Docs, Drive, Maps, account sign-ins, all of it. The fix is the
**`_PLATFORM_SERVICES` table**: multi-service platforms (Google, Bing, Yahoo,
Amazon, Facebook, Instagram, YouTube, X, Apple, Microsoft) are routed by
**subdomain + first path segment** *before* the generic domain vote:

* a matched service's verdict **replaces** the domain vote
  (`docs.google.com` → `tool`, `maps.*` → `local`, `news.*` → editorial,
  `accounts.*` → `account`, `shopping.google.com` → marketplace,
  `aws.amazon.com` → `tool`, `facebook.com/marketplace` → marketplace);
* strict platforms (Google, Yahoo) return **no opinion** for unrecognized
  services — honest `unknown`, never a false "search"; open platforms
  (Amazon) fall through to their normal retailer behaviour;
* three new subtypes entered the taxonomy: `local` (→ reference), `tool` and
  `account` (→ unrelated, with the invariant that a final `unrelated` is
  never study-relevant — a shared Google Doc *about* retinol is still not a
  shopping journey);
* review-driven refinements (§6): marketplace **surfaces** stay topic-neutral
  but deep *item* pages demote to `listing` so a fetched off-topic Facebook
  Marketplace listing can still collapse; two-letter locale subdomains
  (`cn.bing.com`) are transparent.

### 5.5 "Even if we have to train it ourselves" — the learned channel (`9e6c380`, `model.py`)

A self-trained **multinomial logistic model** joined the committee as one
more voting channel. Every design choice serves robustness:

* **Feature hashing** (stable MD5-based signed hashing, 2¹⁶ dims): no
  vocabulary file to persist or drift; the same string hashes identically on
  any machine. `FEATURE_VERSION` is stored in the model file; when
  `featurize()` changes, stale files **retrain themselves transparently** on
  load (the alternative — old weights read through new feature hashes — is
  silent noise).
* **Features**: URL structure (core, subdomain, TLD, path segments and
  word-pieces, query keys, slug *shape* signatures — root-slug / `-vs-` /
  question-start), markup (schema.org types, og:type, price/cart flags,
  product-count bucket), text tokens (title, meta, h1, first words), vendor
  prior, platform fingerprint. **URL-only variants of every training page**
  are included so the model works on unfetchable URLs.
* **Training data** = synthetic ground truth + **distillation of the rule
  tables** (the service-routing map, transactional URL tokens, editorial slug
  patterns — so the model knows `docs.google.com` is a tool *standalone*) +
  optional **self-training on a real parquet** (confident rows as weak
  labels: `python -m conveyer.scraping.model train --pages …`). A distillation
  lesson: the first editorial-slug batch made beauty-token product names lean
  "article"; the fix was **counterweight samples** — the same vocabulary under
  `/product(s)/` labeled pdp, so the path segment stays the decider.
* **Persistence**: a plain `.npz` written **atomically** (tmp + `os.replace`;
  a torn write once meant a corrupt file that blocked autotrain forever while
  loading as None — found in review); corrupt/stale files retrain over
  themselves; training needs scikit-learn, **inference needs only numpy**.
* **Integration**: votes `model_weight (2.0) × probability` for the top
  subtypes — deliberately **below** the decisive 2.5–3.0 rule votes, so the
  model tips ties but cannot overrule a `/cart/` path; it "earns" a subtype
  only at p ≥ 0.5 (with an explicit `0 <` guard after review found
  `model_weight=0` made `0 ≥ 0` true for *every* subtype); fail-open (any
  exception ⇒ no opinion); it sees only the page's own content or the bare
  URL (never base/directory stand-ins — train/serve match); and it is
  explicitly **off** in the URL-rules-only validator and the smart-fetch gate
  (determinism: those verdicts must not depend on whatever CWD-relative model
  file exists).
* **Honest quality**: 5-fold CV ≈ 0.73 standalone on the ~560-sample seed
  frame — weak on `account`/`local` (few distinct URL shapes). That is fine
  *for a corroborator*; its value grows with self-training on real rows.

**Channel ablation** (ground-truth corpus, from notebook 02) shows where the
accuracy actually lives:

| variant | category acc | subtype acc |
|---|---|---|
| full (all channels) | 1.000 | 1.000 |
| no learned model | 1.000 | 0.933 |
| no vendor prior | 1.000 | 1.000 |
| URL/domain only (no content) | 0.842 | 0.933 |
| URL only, no model | 0.842 | 0.858 |

Reading: **content buys the last ~16 points of category accuracy** (the
`unrelated` collapse needs the page's own words); the **model earns +7pp on
subtypes**; the vendor prior is currently redundant (the other channels cover
it on synthetic data — it will matter on real data with missing markup).

The final channel roster (any one can carry a page alone): **URL tokens ·
service routing · curated domain lists · page markup · vendor prior · domain
directory · platform fingerprint · learned model**, plus an optional **LLM
refinement** pass (`classifier="auto"` + `ANTHROPIC_API_KEY`) for
low-confidence pages.

### 5.6 Failure: brand-only "coincides" — precision-first matching (`9e6c380`)

The original product↔chat matcher scored SKU 1.0 > brand 0.75 > name-overlap
> category 0.3 with `coincides = score ≥ 0.5`. The flaw: **brand alone
(0.75) coincided** — a CeraVe *cleanser* page "coincided" with a CeraVe
*moisturizer* mention. For a funnel model whose headline number is "did the
user land on the product the agent recommended", that is a systematic
false-positive machine.

The redesign is tiered (`match_strength`), with `coincides` requiring
**exact/strong** AND the score gate:

| tier | reached by | may coincide |
|---|---|---|
| **exact** | SKU identity; brand agreement + near-identical name | ✔ |
| **strong** | brand agreement + *name* corroboration (core-token containment ≥ 0.34 or char-trigram ≥ 0.45); or a near-identical multi-token name with no brand info | ✔ |
| **likely** | brand alone; brand + only a shared form/category word; moderate name similarity alone | ✘ |
| **none** | nothing meaningful | ✘ |

The precision machinery, each piece motivated by a concrete failure found in
adversarial review (§6):

* **Vetoes**: a brand conflict (Cetaphil vs a CeraVe mention) or an attribute
  conflict caps any match at `likely` no matter how similar the names read.
  Attributes = SPF numbers (SPF 30 ≠ SPF 60) and product-**form** words (a
  *lotion* is not the *cream*) — with **synonym groups** so marketing
  variants of the same form don't false-veto ("Foaming Facial Cleanser" IS
  the "foaming face wash" that was mentioned: wash≡cleanser≡foam).
* **Core-token containment**: name evidence is measured on the product name's
  tokens **minus brand words** (and minus size/count units — "19 oz" is
  packaging, not identity), requiring ≥ 2 core tokens. Both rules exist
  because review found same-brand siblings riding the brand token into the
  exact tier, and a bare "Moisturizer" hitting containment 1.0 against any
  moisturizer mention.
* **Char-trigram cosine** with a length-matched head window (the entity text
  is a sentence; the product name typically leads it) makes matching
  **typo-robust** ("CereVe Moisturising Cream" still connects) without an
  embedding dependency — embeddings were considered and rejected for short
  product strings (trigrams cover the same ground deterministically).
* **`brand_in_entity`** connects **long-tail brands no lexicon knows**: a
  JBCA conditioner PDP links to "the JBCA thickening conditioner" through the
  mention's own words. Review hardened it twice: it only applies when the
  mention carries *no* explicit brand (it must never override a conflicting
  mention brand), and common-adjective brand names (Simple, Pure) are
  excluded from the distinctive-token set.
* **`match_signals`** records exactly which evidence fired; pages carry the
  roll-up `chat_match_strength`/`chat_match_score` — the page↔chat product
  connection at the page grain, ready for the funnel model.
* The synthetic corpus gained **hard negatives** (same-brand different
  product; rival brand, same product line) that the old scorer would have
  failed; coincide precision/recall on ground truth remain 1.0.

### 5.7 Failure: informational slugs invisible to the URL channel (`c17b0f4`)

`zicail.com/eye-cream-vs-eye-serum/` collected **zero URL votes** (no
`/blog/` prefix, no `best-`/`how-to-` token) and — with the site serving
challenge shells to bots — fell to `unknown/other`. Three slug signatures now
vote `article`: **comparisons** (`[word]-vs-[word]`, a word required on both
sides so a `/vs-pink` brand line earns nothing), **question/explainer
phrases** (`what-is-`, `why-your-`, `when-to-`, `benefits-of-`,
`difference-between`, `-explained`, `-myths`, `-mistakes`…), and the
**WordPress-permalink shape** (a single root-level slug of ≥ 3 hyphenated
words; suppressed on curated retailer/marketplace domains, whose root slugs
are merch landing pages). Crucially, **relevance still gates the final
category**: an off-topic `/playstation-vs-xbox-which-to-buy/` on an unknown
domain keeps the article *reading* but stays `unknown` — the pattern never
mints an on-topic label. (Same commit: the zero-votes early return had never
set `is_study_relevant` even when slug relevance qualified — fixed.)

---

## 6 · Adversarial review as a working method

Twice in the project, a full multi-agent adversarial review ran over freshly
written work *before* it shipped, with verification agents instructed to
**refute** each claimed finding by reading and executing the actual code.
Both reviews earned their cost:

* the multimodal-classifier review (§5.2) confirmed **a dozen defects**;
* the big-upgrade review confirmed **16**, every one reproduced and fixed
  pre-ship, including: the platform fingerprint making off-topic Shopify
  homepages study-relevant; Shopify-hosted curated retailers flipping to
  `brand_owned`; deep marketplace items inheriting topic-neutrality;
  locale subdomains of search engines becoming `brand_landing`; the
  tool/account "unrelated-but-relevant" invariant break; HTTP-date
  `Retry-After` disabling 429 retries; blocked domains losing header salvage
  after the circuit opened (fixed via profile platform memory); the
  non-atomic model save; the `model_weight=0` earned-gate hole; the model
  seeing base-page stand-in content (train/serve mismatch); the validator
  depending on a CWD-relative model file; and five matcher precision holes
  (same-brand siblings reaching exact, `brand_in_entity` overriding explicit
  conflicts, form-synonym false vetoes, brand+category-only reaching strong,
  1-token names reaching strong).

The pattern that makes this work: findings must come with a concrete failure
scenario, verifiers default to "refuted" unless they can reproduce the logic,
and every confirmed finding lands as **both a fix and a pinned regression
test**.

The same philosophy runs continuously in-line: `run_scrape` prints a
`[validate]` URL-rule audit after every run, and the synthetic ground truth
scores category/seller/coincide accuracy on every pipeline invocation.

---

## 7 · Process failures — what went wrong outside the code

Honest inventory; each one changed how we work.

### 7.1 The auto-merge that broke main (`dc4e258` → repaired by `1552473`)

A GitHub auto-merge of PR #17 produced a **hybrid `run_scrape`** (new
preamble, old body referencing deleted symbols), dropped two parameters from
`classify_rule`/`classify_page` (`directory_entry`, `content_scope`) that the
merged bodies still used, and lost an import. The killer detail: the
pipeline's *defensive* degrade-to-unknown guard **swallowed the resulting
TypeErrors**, so instead of crashing, every page classified `unknown` —
journey accuracy quietly fell to 0.65 and the funnel model emptied. Lessons:
(a) a merge is a code change and needs the same test gate; (b) catch-all
error guards convert loud failures into silent data corruption — the
self-evaluation harness is what caught it.

### 7.2 The stale-kernel AttributeError

A `'PageClass' object has no attribute 'signals'` traceback in a user
notebook was not a code bug at all — a running Jupyter kernel holds old
module code after a pull. Now standard advice in every hand-off: **restart
the kernel after pulling**.

### 7.3 Committed conflict markers in a notebook (`030a34b`)

An executed `.ipynb` is JSON full of volatile fields (execution counts,
random cell ids, timestamps). Text-merging two executed notebooks produces
marker soup — and at one point a merge resolution **committed the markers
themselves** into `main`'s `02_page_classifier.ipynb` (108 marker lines; the
file wasn't valid JSON), after which every subsequent merge stacked new
conflicts inside old ones. Resolution: the branch's builder-generated
notebooks are the single source of truth; the healing merge took them
wholesale. Prevention: `.gitattributes` marks `*.ipynb -text -diff -merge`
(binary), so a future conflict is a whole-file pick — resolve by keeping
either side and regenerating with
`python notebooks/_build_notebook_0X.py --execute`. The notebooks being
**fully generated by scripts** is what makes this safe.

### 7.4 Resume/cache semantics vs "I re-ran it and nothing changed"

Twice, a fix appeared "not to work" because of persistence semantics, not
code: `resume=True` never revisits a URL already in the JSONL (old rows keep
old labels — that is what `--reclassify` is for), and the fetch cache
replays whatever was cached first (which once included a 202 challenge shell
— hence "never cache near-empty bodies"). Diagnosing before assuming: check
`fetch_status`/`html_path`/the JSONL row before concluding a classifier
regression.

### 7.5 Modeling failures worth remembering

* **Post-treatment bias in the flesh**: the first single logistic model
  produced negative exposure coefficients because visit counts (a
  consequence of following the recommendation) were controlled for → the
  two-model design (§3.2).
* **Zero-lift mystery**: `visited_recommended_brand` lift was 0/NaN because
  retailer PDPs carry the brand only in product JSON-LD → product brands
  folded into the page-brand lookup.
* **Leaky synthetic close-turns**: naming the brand in the generator's
  closing question turned `unsolicited` into `endorsed` and destroyed the
  planted archetypes → pronoun close-turns ("where can I buy it?").
* **Random archetype draws** skewed small runs → deterministic
  largest-remainder assignment.

---

## 8 · Rejected & superseded approaches (quick reference)

| approach | why it was abandoned |
|---|---|
| stdlib `robotparser.read()` | no timeout → infinite hangs on dead hosts |
| batch fetch → one parquet at the end | hangs look like crashes; crashes lose everything; RAM unbounded |
| keeping all Futures / preloading cache HTML | 10–26 GB RSS on real runs |
| `google` in a flat search-domain list | every Google service misread as "search" |
| skincare-only relevance lexicon | haircare/cosmetics PDPs collapsed to `unrelated` |
| brand-only match ⇒ coincides (0.75) | same brand ≠ same product; systematic false positives |
| Jaccard name overlap incl. brand tokens | brand word + generic words reached the exact tier |
| form-conflict veto without synonyms | "face wash" false-vetoed the matching "cleanser" |
| `float()` Retry-After parsing | HTTP-date form (RFC 9110) silently killed 429 retries |
| platform fingerprint ⇒ known ⇒ relevant | off-topic Shopify homepages became journey pages |
| marketplace neutrality for deep item URLs | fetched off-topic listings could never collapse |
| model earned-gate `≥ 0.5 × weight` unguarded | `model_weight=0` ⇒ `0 ≥ 0` earns every subtype |
| non-atomic model save + exists-only autotrain | one torn write disabled the channel forever, silently |
| trusting 2xx = content | 202 challenge shells and 200 error pages classified as the page |
| caching every ok body | cached shells poisoned all replays |
| text-merging executed notebooks | conflict-marker soup, eventually committed to main |
| embeddings for product-name matching | short strings; char-trigrams cover it deterministically, no dependency |
| renaming `skincare_relevance` | schema stability for existing parquets beat naming purity (documented instead) |

---

## 9 · What the numbers say right now (synthetic validation)

All figures from the executed notebooks / test suites at head; they validate
the **measurement machinery** — real data produces the real profile from the
same cells.

* **Self-evaluation**: category / seller / coincide accuracy, precision,
  recall all **1.0** on the ground-truth corpus (including matcher hard
  negatives). Journey: conversion accuracy 1.0, followed accuracy 1.0
  (n=40); lift 2.5 / 2.5 / 2.2 with non-overlapping Jeffreys directions.
* **Ablation**: see §5.5 — content ≈ +16pp category accuracy; model ≈ +7pp
  subtype.
* **Learned model standalone**: ~0.73 5-fold CV (corroborator by design).
* **Agent behaviour under similar prompts** (module 1): informational
  prompts draw the most unsolicited recommending (reco rate 0.86 vs 0.65 on
  purchase prompts); purchase prompts draw the most links (~0.96/turn);
  by stage, links concentrate at **Intent** (2.0/turn) while brand seeding
  happens at Awareness/Discovery.
* **Link relevancy**: editorial holds attention longest (mean dwell ≈ 28.5 s),
  then search ≈ 23.0, shopping ≈ 21.9, community ≈ 21.5; `unrelated` trails
  ≈ 14.1 s. Search/reference open first (trail position 1.0). Pages whose
  products match the chat mention out-dwell unmatched ones (≈ 23.0 vs
  20.2 s).
* **Susceptibility** (lift of following the agent, within segments): largest
  for users who **didn't ask** for recommendations (≈ 4.6 vs ≈ 1.9 for
  frequent askers) and for **short conversations** (≈ 4.6 vs ≈ 1.9 long) —
  askers convert at high rates regardless (they were going to buy); the
  agent's *marginal* weight peaks on unsolicited exposure to organic
  browsers. `followed_agent_link` is the top standardized exposure-model
  coefficient (odds ratio ≈ 2.9 per SD).

---

## 10 · Operational runbook

```bash
# full pipeline (offline demo / real data auto-detected)
python -m conveyer.pipeline --data data/conversations.parquet --online

# module 2 alone — polite, block-resilient, resumable
python -m conveyer.scraping --clickstream-dir data/conversations.parquet \
    --online --max-urls 2000 --hard-timeout 30 --progress-every 50
# interrupted? run the same command again — resumes from the JSONL

# repairs on an existing parquet (no re-scrape)
python -m conveyer.scraping.validate outputs/scrape/scraped_pages.parquet            # URL-rule report
python -m conveyer.scraping.validate outputs/scrape/scraped_pages.parquet --apply    # repair in place
python -m conveyer.scraping.validate outputs/scrape/scraped_pages.parquet --reclassify --apply
                                                    # content-aware rescue from cached HTML

# the learned model
python -m conveyer.scraping.model train --pages outputs/scrape/scraped_pages.parquet  # self-train
python -m conveyer.scraping.model eval                                                # per-subtype P/R

# notebooks are GENERATED — never hand-edit the .ipynb
python notebooks/_build_notebook_01.py --execute
python notebooks/_build_notebook_02.py --execute
```

Working agreements learned the hard way: restart Jupyter kernels after every
pull; treat merges as code changes (run the suites); notebooks conflict as
whole files — pick a side and regenerate; check `fetch_status`/`html_path`
before diagnosing a "classifier bug"; extend brands in `conveyer/brands.py`
and domains in `data/domain_directory.json` — never inline.

Test suites (all standalone-runnable, no pytest needed):
`tests/test_scraping.py` (45) · `tests/test_conversations.py` (7) ·
`tests/test_journey.py` (5) · `tests/test_dashboard.py` (2).

---

## 11 · Known limitations & roadmap

1. **JS-rendered storefronts** need a headless browser — the seam is
   `Fetcher.fetch` (plug Playwright there if coverage demands it).
2. **Matching is dormant without the entity tables** — in input-file-only
   mode there are no `fact_ai_recommendation`/`fact_ai_concept` rows, so
   products carry metadata with `match_type="none"`; the tiers light up the
   moment those tables are present. With module-1-derived (brand-level)
   mentions, page matches cap at `likely` by design.
3. **The conversion outcome is a proxy** (reached cart-or-deeper in a
   10-event window) and **exposure is observational** (users self-select into
   following recommendations) — stated on every surface; a causal design
   (e.g. within-user comparisons, instrumenting exposure) is future work.
4. **Synthetic validation certifies plumbing, not real-world accuracy** —
   the first real-data run should be followed by `model train --pages` (the
   self-training loop) and a manual audit of a labelled sample.
5. **Lexicon coverage**: brands and relevance terms are curated;
   `extra_relevance_terms`, the external domain directory, and
   `brand_in_entity` handle the long tail, but coverage review on real data
   is a standing task.
