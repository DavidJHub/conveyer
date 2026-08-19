# Classification & scraping — restructuring notes

Covers four things asked in this session: (1) the DNS-failure workaround in
`fetch.py`, (2) what `circuit_open` actually means, (3) a 45-domain
independent re-tag of the demo run and its agreement with the model, and
(4) whether the four label fields have enough categories.

---

## 1 · DNS failures (`fetch.py`)

### The error

```
ConnectionError: HTTPSConnectionPool(host='www.instagram.com', port=443):
Max retries exceeded with url: / (Caused by NameResolutionError(
"...Failed to resolve 'www.instagram.com' ([Errno 11001] getaddrinfo failed)"))
```

Before this change it was caught by the generic `except Exception` in the
retry loop, so it consumed **all `max_retries` attempts with exponential
backoff** — several seconds of sleeping to re-ask a question whose answer
cannot change — and then counted as one ordinary domain failure.

### The insight that shapes the fix

`www.instagram.com` obviously resolves for the rest of the world. So a DNS
failure has **two causes that `requests` reports identically**:

| Cause | Truth | Correct response |
|---|---|---|
| Host genuinely doesn't resolve (NXDOMAIN, dead/parked/typo domain, shut-down shortener) | permanent | never retry; open the circuit at once |
| **Our own** resolver is down / machine offline / VPN in the way | transient, and affects **every** domain | do not blame the domain, do not commit the row |

Conflating them is the dangerous case: a 30-second local DNS blip would mark
every domain dead, open every circuit, and — because the pipeline commits
rows to the JSONL log — **burn those verdicts in permanently**, since resume
skips anything already logged. A brief network hiccup would silently blank
part of the corpus.

### What was implemented

1. **`_is_dns_failure(exc)`** — detects resolution failures *structurally*
   first, walking `__cause__`/`__context__` for a `socket.gaierror`, because
   urllib3 buries it under `NameResolutionError → MaxRetryError →
   requests.ConnectionError`. Falls back to message markers (`getaddrinfo
   failed`, `[Errno 11001]`, `Temporary failure in name resolution`, …).
2. **`Fetcher._dns_alive()`** — a cached canary (`one.one.one.one`,
   `dns.google`, `example.com`, 30 s TTL) answering "can this machine resolve
   *anything*?". Costs one lookup per 30 s, not one per URL, and notices a
   resolver that comes back mid-run.
3. **Two new `fetch_status` values**:
   - `dns_error` — resolver works, this host does not. Permanent.
   - `dns_unavailable` — our resolver is down. The host is **not judged**.
4. **No retries** on either — the loop returns immediately.
5. **`_record_outcome(..., fatal=True)`** on `dns_error` opens the domain's
   circuit on the *first* observation instead of paying two more lookups to
   learn the same thing (and clears the success count, so a stale earlier
   success cannot keep the circuit closed).
6. **`dns_unavailable` rows are never committed** in `pipeline.handle()`, so a
   later run retries them. The run summary prints how many were deferred.

### Measured effect

```
dead host: status=dns_error in 0.11s   (was ~6s of futile retries)
circuit after 1 failure: True
false positives on ReadTimeout / ConnectionReset: none
```

---

## 2 · What `circuit_open` means

It is **not** a network error. It means *"we did not even try."*

`Fetcher` keeps `_domain_state: {domain: [consecutive_failures, successes]}`.
When a domain accumulates `domain_failure_threshold` (default **3**) straight
failures with **zero** successes, the domain's "circuit" opens, and every
remaining URL on that domain returns instantly:

```
fetch_status = "circuit_open"
fetch_error  = "domain circuit open (3 straight failures)"
```

The purpose is cost control. Without it, a dead or fully bot-walled host with
2,000 URLs in the worklist would cost `2000 × hard_timeout` = up to 16 hours
of waiting to learn the same thing 2,000 times. The name is borrowed from the
electrical/microservice circuit-breaker pattern: trip once, stop sending
current.

**In the demo data**, 69 rows are `circuit_open` — mostly `laroche-posay.us`
(28), `zicail.com` (11), `kerastase-usa.com` (8). Note that La Roche-Posay and
Kérastase are *exactly the brands the study cares about*: they bot-wall the
crawler, so their circuits open and their pages are classified from URL and
domain evidence alone.

**Important consequence:** `circuit_open` pages are not failures to be fixed
one by one — they are a *whole domain* that was written off. The page still
gets classified, just from the URL/domain rungs of the fallback ladder.

**Two defects to be aware of** (both in `docs/ENGINEERING_REVIEW.md`, F3):

- The state is **in-memory only**, so every new run re-probes every dead
  domain from zero. The README's "domains never re-worked" holds only within
  one process.
- The trip condition requires `oks == 0`, so **one** success ever disarms the
  breaker permanently. A retailer that serves its homepage and walls every
  deep link — Amazon, Sephora, Olive Young — can never trip it. The DNS path
  added above sidesteps this via `fatal=True`; the general case still needs a
  rolling window.

---

## 3 · Independent re-tag of 45 demo domains

### Method

`tools_sample_domains.py` draws a reproducible (`seed=7`) sample of **45
domains** out of the 1,071 distinct domains in
`outputs/scrape_demo/scraped_pages.parquet` (6,000 page rows), one
representative URL per domain (most-surfaced, tie-broken by URL), **stratified
by the model's own category** so the draw isn't 53 % `unrelated`/`unknown`.

`tools_review_sample.py` holds the reviewer's labels — assigned from URL +
title + known identity of each domain, using the *same* taxonomy — and scores
agreement. Output: **`outputs/sample_reviewed.parquet`** (45 rows, model
labels + review labels + per-field agreement flags).

### Agreement

| Field | Agreement |
|---|---|
| `page_category` | **53.3 %** (24/45) |
| `page_subtype` | **37.8 %** (17/45) |
| `seller_type` | 91.1 % (41/45) |
| `funnel_stage` | 84.4 % (38/45) |
| study relevance (binary) | 88.9 % (40/45) |

`seller_type` and relevance are respectable. `page_category` at 53 % and
`page_subtype` at 38 % are not usable as-is for a model that feeds euro
attribution.

Caveat: this is one reviewer on 45 domains, stratified (so not a
representative prevalence estimate), and the reviewer saw the model's label —
it is an error-*discovery* exercise, not a clean accuracy benchmark. The
failure *patterns* below are what matter, and they are systematic.

### Pattern A — `unknown` is a dumping ground (14 of 21 category misses)

Every one of these is a page whose identity is obvious from the URL:

| Domain | Model | Review | Why the model missed |
|---|---|---|---|
| `oliveyoung.com` | unknown | **shopping/pdp/retailer** | robots_blocked; major KR beauty retailer — a *relevant conversion page lost* |
| `zara.com` | unknown/article | shopping/pdp | fetched only at `base` scope |
| `hauteandwhatnot.com` | unknown | editorial | robots_blocked beauty blog |
| `zoho.com`, `similarweb.com`, `goflow.com`, `arras.io`, `benefitscal.com` | unknown | unrelated | robots_blocked, but trivially identifiable |
| `rosecoaudit.com`, `gov.co`, `globemagazine.com`, `realtor.com` | unknown | unrelated | fetch error |
| `dslhospice.com`, `makeugc.ai`, `miaprep.com`, `shellpointmtg.com` | unknown | unrelated | fetched OK but no rule fired |

The `oliveyoung.com` case is the expensive one: a bot-walled PDP at a major
beauty retailer becomes `unknown → Irrelevant`, so a genuine
Intent/Purchase-stage touchpoint drops out of the funnel entirely. This is
exactly the systematic under-count that biases attribution *downward* for
retailer journeys, and the bias is not random — it is concentrated on the
biggest, best-defended retailers.

### Pattern B — `pdp` is the default subtype for junk

Of the 17 domains the model calls `unrelated`, **13 carry subtype `pdp`** —
Telegram, Suno, Venice.ai, emojicombos, tokcomment, customer.io, joinf,
thirteen.org (PBS), amz123. The reviewer assigns `pdp` to 9 domains *in
total*, and to none of these.

Since `page_subtype` drives the Purchase/Post-Purchase funnel bump and is a
feature of the learned model, a `pdp` label sprayed across unrelated pages is
both a data-quality problem and a training-signal problem.

### Pattern C — confidence is anti-correlated with correctness

This empirically confirms finding **F5** of the engineering review:

| Confidence band | n | Category accuracy |
|---|---|---|
| 0.00 – 0.60 | 13 | **61.5 %** |
| 0.60 – 0.80 | 14 | 50.0 % |
| 0.80 – 0.95 | 9 | 55.6 % |
| 0.95 – 1.00 | 9 | **44.4 %** |

**The most confident predictions are the least accurate.** The cause is the
`else 1.0` branch in `_softmax_conf`: when only one channel votes, confidence
is hardcoded to 1.0. Nine of the sampled domains sit at conf ≥ 0.95 and fewer
than half are right. Because `model.py` self-trains on rows with
`page_category_confidence >= 0.75`, the pipeline is preferentially feeding
itself its **worst** labels as gold.

This single number is the strongest argument in this document for fixing F5
before any retraining.

### Pattern D — individual misfires worth fixing

- **`x.ai/account`** (an API sign-in page) → `community/forum`, confidence
  0.82. A sign-in page is not a forum.
- **`owndoc.shop`** → `brand_landing` from the title *"Our apologies but we
  can not ship to Colombia"* — a geo-block error shell treated as real
  content. This is finding **F10** (wall pages) observed live.
- **`drugs.com`** → `unrelated`. It is a health/drug reference site and
  skincare-adjacent; `reference` is the right label.
- **`dermcarecharlotte.com/schedule-an-appointment`** → `editorial/article`.
  It is a dermatology clinic booking page — a *local service*, and
  study-relevant.
- **`micoaes.com`** (B2B laser tattoo-removal machines) → `shopping/pdp` with
  `skincare_relevance = 1.0`. Professional equipment is not consumer skincare;
  a relevance false positive.

---

## 4 · Are the four label fields sufficient?

Short answer: **`seller_type` and `funnel_stage` are under-specified;
`page_category` has a structural flaw; `page_subtype` is roughly the right
size but mis-assigned.** The evidence below is from the sample, not just
theory.

### 4.1 `page_category` — the flaw is a conflation, not a shortage

Nine values is a reasonable count. The problem is that **`unrelated` currently
means two different things**:

1. *off-topic subject matter* — real content about pet sitting, real estate,
   video games; and
2. *non-content surfaces* — logins, dashboards, webmail, admin panels,
   utilities.

In the sample, **10 of 45 domains (22 %)** are category 2: `accela`,
`benefitscal`, `ca.gov`, `dslhospice`, `shellpointmtg`, `x.ai`, `zoho`,
`globemagazine` (wp-admin), `customer.io`, `similarweb`. The taxonomy already
has `tool` and `account` *subtypes*, but both map to category `unrelated`,
so the distinction is thrown away at the level the funnel actually uses.

This matters because the two behave differently in a journey: an off-topic
article is a genuine attention diversion, whereas a webmail tab is ambient
background noise from the clickstream. Lumping them inflates "distraction"
and adds noise to any dwell-based metric.

**Recommendation:** add a `utility` category (subtypes `tool`, `account`,
`admin`) mapping to funnel stage `Irrelevant` — cheap, since the subtypes
already exist — and reserve `unrelated` for genuinely off-topic *content*.

### 4.2 Missing categories the data actually demands

| Add | Evidence / rationale |
|---|---|
| **`video`** | `bilibili.com` in-sample; YouTube/TikTok reviews are a dominant beauty-discovery channel. Currently forced into `community` or `unrelated`. Distinct funnel role (Evaluation), distinct dwell profile. |
| **`social`** | Instagram/TikTok/Reddit are not the same as a forum. Reddit is genuine peer evaluation; an Instagram brand profile is closer to owned media. `community` conflates them. |
| **`local` / service** | `dermcarecharlotte.com` — clinics, salons, derm appointments. Real in skincare, currently `local` subtype → `reference` category, which reads oddly. |
| **`comparison` / review-aggregator** | "X vs Y", best-of round-ups and rating aggregators are the classic Evaluation surface; today they land in `editorial`. The URL rules already detect `-vs-` and `best-…-for`, so the signal exists and is being discarded. |
| **`deal` / coupon** | Distinct, high-intent, and heavily present in real clickstreams. |
| **`quiz` / diagnostic** | Skin-type quizzes and shade finders are a *signature* skincare funnel step and a strong intent signal. |
| **`b2b` / professional** | `micoaes.com` — professional equipment and ingredient suppliers (`lotioncrafter`) are not consumer journeys and currently pollute `shopping` with `relevance = 1.0`. |

### 4.3 `seller_type` — 3 values is too few

`brand_owned / retailer / na` cannot express distinctions the attribution
model needs, and the input data is already richer (SimilarWeb supplies
`1p/3p/unknown`). Recommended additions:

- **`marketplace_3p`** — an Amazon third-party seller is commercially very
  different from Amazon 1P; margin, authorisation and NIQ panel coverage all
  differ.
- **`aggregator` / affiliate** — comparison and cashback sites that sell
  nothing but route intent.
- **`unauthorized` / grey-market** — `fragranceresale.com` appears in the
  demo errors; resale channels matter for brand-safety reporting.
- **`pharmacy` / professional channel** — a regulated distinct channel in EU
  skincare (parapharmacie), and commercially separate from mass retail.

Also: `na` currently absorbs both "not a commerce page" and "commerce page,
seller unknown". Those should be separated, for the same reason
`unknown` ≠ `unrelated`.

### 4.4 `funnel_stage` — the real gap

Current stages: Awareness, Discovery, Evaluation, Intent, Purchase,
Post-Purchase, Irrelevant.

Contemporary marketing practice (McKinsey's consumer decision journey, and
Google's "messy middle" explore/evaluate loop) has largely moved away from
strictly linear funnels toward a **looping explore ↔ evaluate** model with an
explicit **loyalty/advocacy** tail. Two concrete problems here:

1. **No Loyalty / Advocacy / Retention stage.** `order` → Post-Purchase is the
   end of the road. Subscriptions and refills — central to skincare economics
   — have nowhere to go, and repeat purchase is invisible.
2. **`search` → Intent is too strong.** A Google SERP is usually *explore*,
   not intent-to-buy. Mapping it alongside `shopping` inflates the Intent
   stratum, which is precisely the stratum the conversion proxy conditions on.
   `site_search` (searching *within* a retailer) genuinely is Intent; a
   general web SERP is not. The taxonomy already distinguishes the two
   subtypes but maps both to the same stage.

Also worth reconsidering: `reference` → Awareness (a Wikipedia ingredient page
read mid-journey is Evaluation, not Awareness), and adding **Comparison** as
its own stage given how much of beauty research is head-to-head.

### 4.5 Management practice

Two-level `category → subtype` is the right shape and matches how vendors do
it (GA4 content grouping, schema.org `WebPage` subtypes such as
`CheckoutPage`/`CollectionPage`/`ItemPage`/`QAPage`/`SearchResultsPage`, IAB
content taxonomy tiers). Three things this project should adopt:

- **Keep `unknown` strictly separate from `other`/`unrelated`.** Already done,
  and done well — don't lose it.
- **Version the taxonomy.** Add a `taxonomy_version` column so a relabelled
  corpus can be told apart from an old one.
- **Measure agreement before trusting accuracy.** The 53 % figure above is a
  single reviewer; a real benchmark needs 300–500 pages and ideally two
  annotators with a Cohen's κ. That remains the highest-leverage open item.

---

## 5 · Suggested order of work

1. **Fix `_softmax_conf`'s `else 1.0`** — Pattern C shows it is actively
   poisoning self-training. One line, highest payoff.
2. **Stop `pdp` being the default subtype** for pages with no product
   evidence (Pattern B) — it corrupts both the funnel bump and the model's
   features.
3. **Rescue `unknown` for URL-decidable pages** (Pattern A). `oliveyoung.com`
   proves relevant retailer conversions are being dropped; the URL rules
   already know what a `/product/detail?prdtNo=` is.
4. **Add the `utility` category** (`tool`/`account`/`admin` already exist as
   subtypes) — 22 % of the sample, near-zero effort.
5. **Add `video` / `social` / `comparison`**, then `quiz`, `deal`, `b2b`.
6. **Split `search` → Intent** into `site_search` → Intent and web `serp` →
   Discovery/Explore.
7. **Add a Loyalty/Retention stage** and extend `seller_type` with
   `marketplace_3p` / `aggregator` / `pharmacy`.
8. Re-run the 45-domain check after each change — the harness is committed and
   takes seconds.

Files: `tools_sample_domains.py`, `tools_review_sample.py`,
`outputs/sample_for_review.csv`, `outputs/sample_reviewed.parquet`.
