# Engineering review — what this repository is, why it is built this way, and what is wrong with it

**Scope of this document.** Part 1 describes the repository as a whole and the
design axioms that explain *why* almost every file looks the way it does.
Part 2 is a deep read of **module 2, `conveyer/scraping/`** — the scraping and
page-classification subsystem — including a full control-flow walkthrough.
Part 3 is the critical part: **the flaws**, ranked by severity, each with a
file reference and quoted code. Part 4 is a sequenced improvement roadmap.
Part 5 records what was actually executed to verify the claims.

This is a *review*, not a tutorial. The plain-language walkthrough already
exists in [`SCRAPING_MODULE.md`](SCRAPING_MODULE.md), the output schema in
[`SCRAPED_PAGES_SCHEMA.md`](SCRAPED_PAGES_SCHEMA.md), and the input schema in
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). This document deliberately does
not repeat them; it explains the *reasoning* behind the architecture and then
attacks it.

Findings marked **[verified]** were reproduced directly against the working
tree during this review. Findings marked **[reported]** come from a full read
of the file but were not executed.

---

## Part 1 · The repository

### 1.1 The question the codebase exists to answer

The input is a parquet of skincare shopping conversations with an LLM agent.
Every turn carries the user's prompt, the agent's answer, the links that answer
surfaced, the links clicked out of it, and the next ten pages the user browsed.

The business question is a **funnel and attribution question**: *what fraction
of these journeys convert, and how much of that conversion is owed to the
agent's recommendations rather than to everything else?* That question is
what forces the four-module shape:

| Module | Package | Question it answers | Output |
|---|---|---|---|
| 1 | `conveyer/conversations.py` | *What was said?* | turn/conversation features: intent, sentiment, topics, brand mentions, funnel stage |
| 2 | `conveyer/scraping/` | *Where did the user land?* | `fact_scraped_page`, `fact_scraped_product` |
| 3 | `conveyer/journey.py` | *Did it move the needle?* | funnel events, conversion + CIs, transition matrix, logistic coefficients |
| 4 | `conveyer/attribution/` | *How many euros?* | exposure panel, sales bridge, MMM |

Module 2 is the load-bearing one for credibility. Modules 3 and 4 are
statistics over whatever module 2 says a page *was*. If a cart page is
mislabelled `unrelated`, the conversion proxy silently loses a conversion, and
every downstream euro figure inherits the error. **The scraping module is the
measurement instrument; everything after it is arithmetic on its readings.**

### 1.2 The six design axioms

Nearly every non-obvious decision in the codebase follows from one of these.
Recognising them makes the code read as intentional rather than idiosyncratic.

**A1 · Offline-first.** No data, no network, no API keys must still produce a
complete, self-scoring run. `conveyer/scraping/synthetic.py` generates a
corpus *with ground truth*, so `pipeline.evaluate()` can report accuracy on
every run. This is why `ScrapeConfig.offline` defaults to `True` and
`use_synthetic_if_missing` to `True`. The payoff is that every notebook
executes anywhere and every stage is testable before real data exists. The
cost is that the entire evaluation story is synthetic — see finding **F9**.

**A2 · Graceful degradation.** bs4, lxml, BERTopic, sentence-transformers,
transformers and the Anthropic SDK are all optional. Each resolves at runtime
and falls back to a stdlib/sklearn path. `conveyer/scraping/extract.py`
carries a complete hand-rolled `html.parser` extractor *in parallel* with the
bs4 one for exactly this reason. The payoff is portability; the cost is a
duplicated code path that must be kept in sync, and a habit of
`except Exception: pass` that hides real bugs — findings **F2**, **F13**.

**A3 · Explicit schemas.** Every output table is written through a pinned
pyarrow schema (`conveyer/scraping/schema.py`), so nullable ints and list
columns round-trip cleanly. This is why rows are dicts validated against
`PAGE_SCHEMA`/`PRODUCT_SCHEMA` rather than whatever pandas infers.

**A4 · Honest measurement.** Conversion is documented as a *proxy* (reaching
cart/checkout in the trail), exposure is documented as *not* causation, and
`unknown` is kept strictly distinct from `unrelated` so "we could not see"
never masquerades as "we looked and it was irrelevant". This distinction is
the single best idea in the classifier and is genuinely enforced — see §2.5.

**A5 · Assumptions are data.** Module 4 keeps an explicit ledger of every
unobservable factor with a range, a source and a status. Constants are not
buried in code. Module 2 does *not* honour this axiom nearly as well: its
decisive thresholds are hardcoded literals — finding **F6**.

**A6 · A long run must not hang, lose work, or grow.** This axiom is
responsible for most of the machinery in `pipeline.py`, `fetch.py` and
`resume.py`: per-URL wall-clock caps, a JSONL commit log, parquet part files,
a sliding fetch window, a circuit breaker. It is the most rigorously executed
axiom in the repository, and the code comments record the regressions that
motivated each control (e.g. "the old design kept every row in RAM and rewrote
the whole parquet at each checkpoint: O(n) memory, O(n²) writes").

---

## Part 2 · Module 2, the scraping module

### 2.1 The problem, precisely

Given ~10⁵ URLs of wildly heterogeneous provenance (agent citations, direct
clicks, and organic browsing trails), decide for each one:

1. **What kind of page is this?** → 9-label taxonomy → funnel stage.
2. **Who sells here?** → `brand_owned` / `retailer` / `na`.
3. **Does the product here match what the agent recommended?** → `coincides`.

with three hard constraints: most commercially interesting pages (Amazon,
Sephora, Ulta) will **refuse to be fetched**; the run must be resumable; and
the whole thing must work with the network unplugged.

### 2.2 Control flow, end to end

```
sources.build_sources()      URL worklist + per-turn brand/product mentions
        │                    (dedup by url|domain, sort by times_surfaced, cap)
        ▼
pipeline._fetch_decision()   "smart" gate — should we spend a request at all?
        │                     · skip URL-decided subtypes (cart/checkout/serp/…)
        │                     · skip past max_fetch_per_domain
        ▼
fetch.Fetcher.iter_fetch()   sliding-window ThreadPool, robots, per-domain
        │                    throttle, circuit breaker, hard wall-clock cap
        ▼
pipeline._process_one()      ── THE FALLBACK LADDER ──
        │   1. page              fetch ok                     scope="page"
        │   2. stripped          retry minus query string     scope="stripped"
        │   3. base              fetch scheme://host/         scope="base"
        │   4. directory         offline site description     scope="directory"
        │   5. none              URL + domain tokens only     scope="none"
        ▼
extract.extract_page()       bs4+lxml → bs4+html.parser → stdlib HTMLParser
        ▼
products.extract_products()  JSON-LD → microdata → OpenGraph → text heuristic
        ▼
classify.classify_page()     8 voting channels → subtype → category → stage
        ▼
products.match_products()    precision-first scorer → match_strength/coincides
        ▼
schema.page_row()            JSONL commit log → parquet parts → final parquet
        ▼
validate.validation_report() post-run URL-rule double check (report only)
```

### 2.3 Subsystem rationale

**`sources.py` — build the shopping list.** Pure pandas. Merges three
provenance channels (`fact_ai_click_through.surfaced_url`, the input file's
`a_links_source`/`ai_click` cells, and the `next_10_urls` trail) into one row
per URL carrying `times_surfaced`, `times_recommended`, `was_visited`, the
owning `message_id`s, and dwell seconds derived from consecutive
`request_time` gaps. Dwell is kept here because it is the only retention
signal available and module 3 needs it.

**`fetch.py` — spend requests carefully.** The controls exist in this shape
because of a specific adversary: CDNs that 403 unknown user-agents at the edge.
Hence `browser_headers=True` by default — presentation changes, compliance does
not: robots.txt is still evaluated against the *bot* identity
(`cached.can_fetch(self.cfg.user_agent, url)`). Two timeout layers exist
because `requests`' `timeout=` bounds the gap between chunks, not total body
time, so a slow-dribbling server needs a separate wall-clock `deadline`.
`401/403/406/451` are never retried because a wall answers identically every
time. Non-OK bodies are still kept, capped at 4 KB, purely to fingerprint the
hosting platform — a bot-walled Shopify store must still classify as a
storefront.

**`extract.py` — never crash on markup.** Two complete extractors (stdlib
`HTMLParser` and bs4) behind one `_assemble()`. The stdlib path is not a stub;
it reproduces title/meta/OG/JSON-LD/microdata/heading/text extraction by hand.

**`classify.py` — the heart.** Not a model, not a rule table: a **weighted
vote across eight independent channels** (URL structure, service routing,
curated domain lists, markup, SimilarWeb vendor prior, offline domain
directory, hosting-platform fingerprint, learned model). The argmax subtype
wins, subject to a transactional override so a cart page's "add to cart"
furniture cannot make it look like a PDP. Every channel is allowed to be
silent, which is precisely what makes the module survive unfetchable pages.
`PageClass.signals` records which channels fired, so every label is auditable.

**`model.py` — a soft advisor, not an authority.** Multinomial logistic
regression over MD5 feature hashing into 2¹⁶ dims, so no vocabulary file has
to be shipped and inference needs only numpy. Trained on synthetic ground
truth *plus distilled rule tables* plus optionally self-trained weak labels
from a real parquet, with human corrections repeated `HUMAN_BOOST` times.
Deliberately capped: it can only decide a subtype if its own probability
clears 0.5, so it tips ties but cannot overrule a `/cart/` path.

**`products.py` — precision over recall, by design.** The matcher is not
generic fuzzy matching. It strips packaging noise but keeps SPF numbers
(identity-bearing), treats brand disagreement and form/SPF disagreement as
hard vetoes, and excludes brand tokens from the name-similarity core so a
same-brand sibling cannot ride the shared brand word to a false match.
Brand-alone or name-alone evidence is capped at `likely` and excluded from
`coincides`. This bias is correct for the study: a false `coincides` directly
inflates the attributed conversion.

**`resume.py` / `schema.py` — never lose work.** The JSONL log is the source
of truth and the reader tolerates a torn final line. Products are written
*before* the page line, which acts as the commit marker. Every
`checkpoint_every` pages the buffer is flushed to a parquet part and cleared,
so RAM is bounded by one checkpoint regardless of run length. `prepare_run()`
fingerprints the input and refuses to silently merge two corpora into one
output table — a genuinely good idea most pipelines lack.

**`validate.py` / `relabel.py` / `readout.py` — the QA loop.** `validate`
re-derives labels from URL rules alone and flags disagreements. `relabel`
scores every row for suspicion, exports a review CSV, applies taxonomy-checked
corrections, and refuses to overwrite an existing human label. `readout`
renders a self-contained HTML report as a pure function of the saved parquets.

### 2.4 The two ideas that carry the design

**Idea 1 — the evidentiary hierarchy (`content_scope`).** The fallback ladder
is not just "try A then B". Each rung carries a different *epistemic weight*,
and `classify.py` enforces it:

```python
real_content = has_content and content_scope in ("page", "stripped")
```

Only a page's *own* document may prove it irrelevant. A base-page or directory
description may corroborate a domain-level role, but can never collapse a
specific deep link to `unrelated`. This is the mechanism that keeps A4 honest,
and it is genuinely well executed.

**Idea 2 — two orthogonal axes.** Structure (*what shape of page is this*) and
topic (*is this about skincare*) are scored independently. That is why "cart"
is topic-neutral infrastructure that needs no skincare keywords to stay valid,
while "article" must earn its topical relevance or fall back to `unknown`.

Both ideas are sound. Most of the findings below are failures of execution
*around* these two ideas rather than failures of the ideas themselves.

---

## Part 3 · What is wrong

Severity: **S1** = silently produces wrong data; **S2** = breaks a documented
guarantee or systematically biases results; **S3** = correctness/robustness
risk; **S4** = hygiene and maintainability.

F1–F15 came from reading the code; **F16–F18 came from actually running it**,
and F16 in particular is a live, reproducible test failure. If you read only
three findings, read **F1** (every European price is wrong), **F16** (offline
runs are not reproducible) and roadmap item **15** (there is no real ground
truth).

### F1 · [S1, verified] Every price on a European site is parsed wrong

`conveyer/scraping/products.py:36,59-66`:

```python
_PRICE_NUM = re.compile(r"(\d[\d,]*\.?\d*)")

def _to_float(x):
    ...
    m = _PRICE_NUM.search(str(x).replace(",", ""))
    return float(m.group(1)) if m else None
```

Commas are stripped *before* matching. Measured output of the real function:

```python
>>> from conveyer.scraping.products import _to_float
>>> _to_float("12,50 EUR")   # → 1250.0      should be 12.50   (100× too high)
>>> _to_float("9,90")        # → 990.0       should be 9.90    (100× too high)
>>> _to_float("1.234,56")    # → 1.23456     should be 1234.56 (1000× too low)
>>> _to_float("19.99")       # → 19.99       correct (US format)
>>> _to_float("1,234.56")    # → 1234.56     correct (US format)
```

Every US-format price is right and **every EU-format price is wrong**, by two
to three orders of magnitude, in whichever direction the site's formatting
happens to choose.

This is not a corner case: the project measures a European skincare market and
Part 4 of the pipeline converts journeys into **euros**. `_to_float` is the
single shared entry point for JSON-LD offers, microdata, OpenGraph and the
text heuristic, so *every* price path inherits it. There is no exception, no
log line, and no downstream sanity check.

**Fix.** Format-detect before stripping: if the last separator is a comma and
is followed by exactly 2 digits, treat comma as the decimal mark. Carry the
`priceCurrency` into the decision. Add a plausibility guard (a skincare SKU at
€1250 should be flagged, not silently stored).

### F2 · [S1, verified] The microdata parser never closes a scope

`conveyer/scraping/extract.py:270-273`:

```python
if self._md_stack and tag != "meta":
    # close the most recent itemscope on a matching-ish close; microdata
    # is best-effort, so we pop conservatively at container ends.
    pass
```

The comment describes behaviour the code does not implement. `_md_stack` only
ever grows. For nested `itemscope`s — the *normal* schema.org shape, a
`Product` containing a `Brand` — once the inner scope opens, `_md_stack[-1]`
stays the inner scope for the remainder of the document. Every subsequent
`itemprop` intended for the outer `Product` is attributed to the wrong item,
so `products._product_from_microdata` reads a `Brand` object that has absorbed
the product's price and name.

**Fix.** Track depth per scope: record the tag depth on `itemscope` push and
pop when that depth closes. Add a nested-microdata regression test.

### F3 · [S2, verified] The circuit breaker does not survive the process, and can be permanently disarmed

Two distinct defects in `conveyer/scraping/fetch.py`.

**(a) No persistence.** `fetch.py:186`:

```python
self._domain_state: Dict[str, list] = {}
```

Constructed fresh in `Fetcher.__init__`, never read from or written to disk.
The README promises "domains never re-worked (circuit breaker, per-domain
fetch budget, learned domain profiles)". That holds *within one process only*.
Resume — the entire operating model for a long run — restarts every dead
domain from zero failures, each re-probe costing up to `hard_timeout` (30 s).
Note the asymmetry: `domain_profiles.json` *is* persisted and *is* correctly
wiped by `reset_run_state`; the circuit state is simply missing that treatment.

**(b) One success disarms it forever.** `fetch.py:190-204`:

```python
def _circuit_open(self, domain):
    ...
    fails, oks = self._domain_state.get(domain, [0, 0])
    return oks == 0 and fails >= thr
```

The breaker requires `oks == 0`. A domain that answers *one* URL and then
fails on the next ten thousand can never trip, because `oks` is now 1. For a
large retailer that serves its homepage but bot-walls every deep link — the
exact profile of Amazon and Sephora, the highest-value domains in the corpus —
the breaker is inert precisely where it was needed.

**Fix.** Persist `_domain_state` to `out_dir/domain_state.json`, load it in
`__init__`, and add it to `reset_run_state`'s removal list. Replace the
`oks == 0` clause with a rolling window (e.g. open when the last N outcomes
are all failures, regardless of ancient successes).

### F4 · [S2, verified] No URL canonicalisation anywhere, so incrementality is unreliable

The worklist key in `sources.py` is `str(url or "").strip()`. The
already-done key in `resume.py:pages_done()` is the raw `url` field read back
from JSONL. Neither lowercases the host, drops the fragment, strips a default
port, collapses a trailing slash, or sorts/strips tracking parameters.

Therefore `https://X.com/a`, `https://x.com/a/` and
`https://x.com/a?utm_source=chatgpt` are three URLs, three fetches, three
rows, and three separate contributions to `max_fetch_per_domain`. This
inflates the workload, corrupts the per-domain budget accounting, and — worse
for the analysis — **splits one page's evidence across three rows**, so a
single landing page can appear as three distinct journey touchpoints in module
3. The existing `SCRAPING_MODULE.md` lists this as improvement #8; it is
materially more serious than "fewer fetches, cleaner joins" suggests, because
it biases the funnel counts, not just the runtime.

**Fix.** One shared `normalize_url()` used as the key in *both* `sources.py`
and `resume.py`. Keep the raw URL as a separate column for provenance.

### F5 · [S2, verified] Confidence is not a probability, and is hardcoded to 1.0 in the weakest case

`conveyer/scraping/classify.py:685-692`:

```python
def _softmax_conf(scores):
    vals = [s for s in scores.values() if s > 0]
    if not vals: return 0.0
    top = max(vals)
    exps = [math.exp(s - top) for s in vals]
    return round(math.exp(top - top) / sum(exps), 3) if len(vals) > 1 else 1.0
```

Two problems.

First, this is a softmax over **raw accumulated heuristic vote weights** —
2.0, 2.6, 1.6, 0.9, `model_weight × p` — that come from incommensurable
channels and were hand-picked, never fit. A confidence of 0.83 has no
statistical meaning; it reflects how many point-values the author assigned to
the winner versus the runner-up.

Second, the `else 1.0` branch: when exactly one category received any votes,
confidence is **1.0 regardless of how weak that single vote was**. A lone
0.9-weight WordPress-permalink guess reports maximum confidence. This is the
worst possible failure direction, because that number then gates real
decisions: `cfg.min_confidence`, the `classifier="auto"` LLM trigger, the
suspicion score in `relabel.py`, and the `>= 0.75` self-training cutoff in
`model.py`. **A page that is maximally under-evidenced is the one most likely
to be silently promoted into the training set as gold.** That closes a
feedback loop that entrenches the error (see F9).

**Fix.** At minimum, replace `1.0` with a saturating function of the winning
vote mass (e.g. `min(0.6, top/4)`), so a single weak vote cannot outrank a
contested one. Properly: fit a calibration map from
`(top − runner_up, n_signals)` to P(correct) on human labels.

### F6 · [S2, verified] The registrable-domain parser is hand-rolled and wrong for common suffixes

`conveyer/scraping/classify.py:127-130,206-211`:

```python
_MULTI_TLD = {
    "co.uk", "com.au", "co.jp", "com.br", "co.nz", "co.in", "com.mx",
    "co.kr", "com.tr", "co.za", "com.sg", "com.hk", "co.il",
}
...
if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_TLD:
    registrable = ".".join(labels[-3:])
elif len(labels) >= 2:
    registrable = ".".join(labels[-2:])
```

`.org.uk`, `.com.es`, `.com.pl`, `.com.ar`, `.co.id`, `.com.tw`, `.net.au`
and many others are absent. For `shop.example.org.uk` the parser computes
`registrable = "org.uk"`. That single wrong value then keys the throttle, the
circuit breaker, the domain budget, the domain profile, and every curated
domain-list lookup — so all pages under any `.org.uk` host collapse into one
shared, meaningless bucket. For a European study this is not exotic.

**Fix.** Use the Public Suffix List (`tldextract`). This is a one-line
substitution that removes a whole class of bug and stops the list needing
maintenance.

### F7 · [S2, reported] Brand matching uses bare substring containment

`conveyer/scraping/products.py:377-380`:

```python
if pbrand in mbrands or any(pbrand in b or b in pbrand for b in mbrands):
    return "agree", "brand_lexicon"
```

No word boundaries, no length floor. Short real skincare brands — **REN**,
**Pure**, **Simple**, **Olay** vs **Olay Regenerist**, **Nivea** vs **Nivea
Men** — will spuriously agree. The `_GENERIC_BRAND_TOKENS` stoplist that
exists precisely to prevent this is applied only to the *third* fallback
route, not to this primary check. Because a brand "agree" is what unlocks the
`strong`/`exact` tiers, a false agreement here converts directly into a false
`coincides`, which is the numerator of the attribution.

**Fix.** Require exact normalised equality or token-boundary containment with
a minimum token length; apply the stoplist to all three routes.

### F8 · [S2, verified] Text normalisation ignores accents, in a French-brand category

`conveyer/ingest.py`:

```python
def normalize(s): return str(s).strip().lower()
```

No NFKD, no punctuation handling, no internal whitespace collapse. In a
skincare corpus this is a guaranteed miss: **L'Oréal/L'Oreal**,
**Kérastase/Kerastase**, **La Roche-Posay/La Roche Posay**, **Caudalie**.
`normalize()` gates the product dedup key and the brand lexicon check, so the
same product appears twice and a correct brand match is scored as a conflict.

**Fix.** `unicodedata.normalize("NFKD", s)` with combining marks stripped,
apostrophe/hyphen folding, whitespace collapse.

### F9 · [S2, reported] The learned model is trained and evaluated on its own output

Three compounding issues in `conveyer/scraping/model.py`:

1. **Self-training loop.** Weak labels are rows where
   `page_category_confidence >= 0.75` — i.e. *the rule engine's own confident
   predictions*. The model is therefore trained to reproduce the rule engine's
   biases, and its votes then feed back into that same engine's vote sum.
   Combined with **F5** (weakest-evidence pages report confidence 1.0), the
   pages most likely to be wrong are the most likely to be promoted to gold.
2. **CV leakage.** `build_training_samples()` appends *two* rows per synthetic
   ground-truth URL — a content-bearing vector and a URL-only vector for the
   same URL and label. `evaluate_model()` then runs plain K-fold
   `cross_val_predict`, so these near-duplicates land in different folds. The
   reported accuracy is inflated by construction.
3. **No class balancing.** `LogisticRegression(max_iter=2000, C=2.0)` with no
   `class_weight`, over a mixture where hand-authored distillation samples for
   `tool`/`account` vastly outnumber naturally rare subtypes like `wishlist`.

**Fix.** `GroupKFold` keyed by URL; a frozen human-labelled holdout that is
never trained on; `class_weight="balanced"`; and cap distillation samples per
subtype.

### F10 · [S3, reported] Soft-404s, paywalls and cookie walls can be confidently labelled `unrelated`

`pipeline._soft_error_page()` catches a good set of shells, but the
`has_content` test in `classify.py` is `bool(page.title or page.text or
page.og)`. A soft-404 with a real `<title>` and boilerplate OG tags, a cookie
wall, or a paywall teaser all satisfy it, so `real_content` is `True` and the
page can be collapsed to `unrelated` — a *confident* off-topic judgement about
a page nobody ever read. Per F9, that verdict can then become training data.
This is the one place where the otherwise-excellent `unknown`/`unrelated`
discipline of axiom A4 leaks.

**Fix.** Add a `content_scope="wall"` rung. Detect it (thin body + known
consent/paywall vendor strings + `<title>` present) and exclude it from
`real_content`, exactly as `base`/`directory` already are.

### F11 · [S3, reported] Redirects are followed unconditionally and unaudited

`fetch.py` calls `self._session.get(url, ..., allow_redirects=True)` with no
redirect cap, no host allowlist, and — the analytically important part — the
throttle key and circuit-breaker key are computed from the **original** URL,
never re-derived from `resp.url`. So a short-link or tracking redirect is
rate-limited against the shortener's domain, not the destination, and
robots.txt for the destination host is never consulted. Classification also
receives the pre-redirect URL, so an `amzn.to` link classifies as an unknown
shortener rather than the Amazon PDP it resolves to. There is additionally no
private/link-local IP guard, so a redirect to `169.254.169.254` is followed.

**Fix.** Cap redirects, re-check robots/throttle/circuit against the final
host, block private and link-local address ranges, and pass `final_url` into
`classify_page`.

### F12 · [S3, reported] `coincide_threshold` is an inert knob

Every code path returning `strength="exact"` produces a score ≥ 0.85, and
every `strong` path ≥ 0.6. `match_products()` sets
`coincides = strength in ("exact","strong") and score >= coincide_threshold`,
with a default `coincide_threshold = 0.5`. The numeric test can therefore
never fail at the default. `coincides` is decided **entirely by
`match_strength`**; the config value that looks like the tuning dial for
attribution precision does nothing until raised above ~0.6.

**Fix.** Either document it as a tier gate, or re-scale scores so the
threshold is live across its nominal range.

### F13 · [S3, verified] Fail-open exception handling hides bugs, not just bad input

A consistent pattern, defensible individually, corrosive in aggregate:

| Location | Swallows | Consequence |
|---|---|---|
| `extract.py:280-283` | any parser exception | an `AttributeError` bug in the extractor is indistinguishable from malformed HTML |
| `classify.py` `_model_votes` | any model exception | a broken model channel contributes zero votes forever, silently, with no log |
| `classify.py` `classify_llm` | 3× any exception | auth failure, rate limit and "no LLM configured" are indistinguishable after the fact |
| `resume.py` `write_manifest` | `OSError` | the run-identity guard degrades silently |
| `resume.py` `iter_jsonl` | `JSONDecodeError` | intended for a torn final line, but silently drops corruption anywhere in the file, uncounted |
| `pipeline.py` validate hook | any exception | a broken validator prints one line and the run reports success |
| `relabel.py` | `(OSError, Exception)` | redundant and effectively bare |

Nothing in the module uses `logging`; diagnostics are `print()` only.

**Fix.** Keep failing open — that is the right call for a long batch run — but
*count and report*. A per-run counter of swallowed exceptions per site, printed
in the run summary and stored in the manifest, converts every one of these from
invisible to observable without changing control flow.

### F14 · [S3, reported] Assorted classifier brittleness

- **Single-letter path tokens.** `_url_subtype_votes` votes `collection` at
  weight 1.6 on `/c/` and `/b/`. These segments are extremely common as opaque
  IDs, locale codes and blog short-links on non-Amazon sites.
- **English-only relevance.** `_RELEVANCE_KEYWORDS` is entirely English, and it
  is the gate that authorises the `unrelated` collapse. A French or German
  skincare page with no brand-domain match scores 0 and is confidently
  discarded. `page.lang` is already extracted and fed to the *model* channel —
  the rule gate simply ignores it.
- **Config cannot override code.** `_directory_votes` is consulted only when
  the curated Python sets are silent, so a stale hardcoded entry can never be
  corrected by the external `data/domain_directory.json` without a redeploy —
  inverting the "knowledge as data" principle the directory exists to serve.
- **Duplicated magic number.** The relevance threshold `0.15` appears as a bare
  literal in three separate places in `classify.py`; changing one silently
  diverges behaviour. (Violates axiom A5.)
- **Non-deterministic tie-break.** `candidate_urls` sorts by `times_surfaced`
  with no secondary key, so ties reorder with upstream row order — which
  changes which URLs survive `max_urls`.

### F15 · [S4, reported] Human-label provenance rests on a string inside a list column

`is_human_labelled()` is `"human" in classification_signals`. That one token
is the sole enforcement of the immutability guarantee relied on by
`validate.validation_report`, `validate.reclassify_pages` and
`relabel.apply_corrections`. Any code path that rebuilds `classification_signals`
rather than appending, or any round-trip that yields `NaN` or a bare string,
silently re-opens every human correction to automated overwrite — with no error.

**Fix.** Add a first-class `is_human_labelled` boolean plus a
`labelled_at`/`labelled_by` pair to `PAGE_SCHEMA`, and treat the signals list
as display metadata only.

### F16 · [S2, verified] The test suite is not hermetic — and one test fails because of it

This was found by running the suite, and it is the most concrete defect in
this document because it reproduces in one command:

```
$ pytest tests/test_scraping.py -k test_directory_robots_blocked_fallback
E   AssertionError: scope base
1 failed
```

The test asserts that a robots-blocked Sephora URL falls all the way down the
ladder to `fetch_scope == "directory"`. It gets `"base"` instead. The cause is
not a bug in the ladder — it is that the test is not isolated:

```python
cfg = ScrapeConfig()                      # offline, empty corpus
fetcher = Fetcher(cfg, html_by_url={})    # base fallback will offline_miss
```

`ScrapeConfig()` uses the **default, repo-global** `cache_dir =
"outputs/scrape_cache"`. The comment's assumption — that the base fallback
will miss — is false on any machine that has ever run an online scrape,
because the offline fetcher reads the per-URL disk cache lazily. On this
working tree:

```
cache files: 22260
base cache path: outputs/scrape_cache\cf2ae1b3b3390f85d95720cf.json
EXISTS: True
```

`https://www.sephora.com/` is cached, so rung 3 (`base`) succeeds and rung 4
(`directory`) is never reached.

**Two separate problems follow, and the second is the serious one.**

**(a) Tests share mutable global state.** Tests instantiate `ScrapeConfig()`
with default paths rather than `tmp_path`, so they read and write the
developer's real `outputs/scrape` and `outputs/scrape_cache`. Results depend
on machine history, so the suite is green on a clean checkout and red on a
working machine — the failure mode most likely to train a team to ignore it.

**(b) "Offline" runs are not reproducible.** The same mechanism applies to
`run_scrape` itself, not just to tests. Which rung of the fallback ladder a
URL lands on — and therefore its `classification_signals`, its `fetch_scope`,
and potentially its label — depends on whatever happens to be sitting in
`outputs/scrape_cache` from previous, unrelated runs. Two analysts running the
same offline command on the same input can legitimately get different tables.
That directly contradicts axiom **A1**, and it means the "runs offline first,
deterministically" property the repository advertises is not guaranteed.

**Fix.** Point every test at `tmp_path` via a fixture that overrides
`out_dir`, `cache_dir` and `model_path`; make the offline fetcher's use of the
disk cache explicit (`offline_cache=True|False`) rather than implicit; and
record the cache generation in `run_manifest.json` so a run's provenance
includes what it was allowed to read.

### F17 · [S3, verified] Resume re-streams the entire commit log twice, every run

`pipeline._recover_state()` iterates `cfg.pages_jsonl_path()` in two separate
passes — once to build `jsonl_pids`/`done_urls`, then again to recover rows
not present in the parts. On this working tree that file is **160 MB**:

```
outputs/scrape/scraped_pages.jsonl     159,9 MB
outputs/scrape/scraped_products.jsonl   11,7 MB
outputs/scrape_cache                  22260 files,  8.6 GB
outputs/ (total)                                   ~9.4 GB
```

Every resumed run therefore pays a full double JSON-decode of the log before
issuing its first request, and that cost grows linearly with the lifetime of
the output directory. It is also why the suite takes >25 minutes on a working
machine while individual tests take ~1 s (F18): tests using default paths
trigger this recovery against the real 160 MB log.

**Fix.** Single pass — collect `done_urls`/`jsonl_pids` and buffer recoverable
rows in the same loop. Better, maintain a compact sidecar index (or use the
part files' own `page_id` column) so the full log is read only when the parts
are found to be inconsistent.

### F18 · [S4, verified] Repository hygiene

| Finding | Evidence |
|---|---|
| **No CI at all** | there is no `.github/` directory in the repo |
| **`pytest` is not a declared dependency** | absent from `requirements.txt`; the project venv did not have it installed |
| **`bs4`/`lxml` not installed in the working venv** | `import bs4` → `ModuleNotFoundError`, so the "preferred" parser path is not the one actually exercised locally |
| **README test count is stale** | README says "`tests/`, 36 tests"; there are **66** `def test_` functions |
| **The venv is not git-ignored** | `git status` lists `.conv/`, `.history/`, `.output/`, `.vscode/`, `output.png` as untracked; `.gitignore` covers `.venv`/`venv/`/`env/` but not `.conv/`. One `git add -A` commits the whole virtualenv |
| **The suite is slow and state-dependent** | the full `tests/test_scraping.py` ran past **25 minutes** and was aborted, while three trivial tests run in **1.0 s** and `test_learned_model_channel` alone takes **24.7 s** (`model_autotrain` triggers a real sklearn fit as a side effect of a classification call). The bulk of the remaining time is F17's recovery against a 160 MB log |
| **~9.4 GB of run artifacts live under `outputs/`** | not ignored per-subdirectory, and actively read by default-config runs and tests (F16, F17) |

The last two matter more than they look. A suite this slow, with no CI to run
it and a failure that only appears on machines with real data, is a suite that
will stop being run — and this repository's entire safety argument rests on it.

**Fix.** Add `pytest` to a `requirements-dev.txt`; add `.conv/`, `.history/`,
`.output/` to `.gitignore`; add a GitHub Actions workflow running the suite on
push (which would also have caught F16, since CI starts clean); pre-seed a
model fixture in `conftest.py` so `model_autotrain` never fires during tests;
correct the README count.

---

## Part 4 · Improvement roadmap

Ordered by (damage prevented) ÷ (effort). The first four are small, local
changes that remove entire classes of silent error.

### P0 — data-correctness, do these first

| # | Action | Files | Effort |
|---|---|---|---|
| 1 | Locale-aware price parsing + currency-aware plausibility guard | `products.py` `_to_float` | S |
| 2 | Pop the microdata scope stack by tag depth | `extract.py` | S |
| 3 | Swap `_MULTI_TLD` for `tldextract` (Public Suffix List) | `classify.py` `parse_url` | S |
| 4 | Unicode-aware `normalize()` (NFKD + punctuation folding) | `ingest.py` | S |
| 5 | Shared `normalize_url()` used by both the worklist and resume keys | `sources.py`, `resume.py` | M |
| 6 | Word-boundary brand matching; stoplist on all three routes | `products.py` `_brand_signal` | S |
| 7 | Isolate tests on `tmp_path`; make offline cache use explicit — fixes the failing test *and* offline determinism | `tests/conftest.py`, `fetch.py` | S |

### P1 — make the guarantees true

| # | Action | Files | Effort |
|---|---|---|---|
| 8 | Persist the circuit breaker; replace `oks == 0` with a rolling window | `fetch.py`, `resume.py` | M |
| 9 | Kill the `else 1.0` confidence branch; saturating fallback | `classify.py` `_softmax_conf` | S |
| 10 | Single-pass `_recover_state` (or a compact done-URL index) | `pipeline.py` | S |
| 11 | `content_scope="wall"` for paywalls/cookie walls/soft-404s | `pipeline.py`, `classify.py` | M |
| 12 | Re-derive throttle/robots/circuit keys from `resp.url`; cap redirects; block private IPs | `fetch.py` | M |
| 13 | Count and report swallowed exceptions in the run summary and manifest | module-wide | M |
| 14 | Promote `is_human_labelled` to a real schema column | `schema.py`, `relabel.py`, `validate.py` | M |

### P2 — make the measurement defensible

| # | Action | Effort |
|---|---|---|
| 15 | **Hand-label 300–500 real pages**, stratified by domain and category, as a frozen holdout | L |
| 16 | Re-point `evaluate()` and `evaluate_model()` at that holdout; `GroupKFold` by URL; `class_weight="balanced"` | M |
| 17 | Fit the vote weights and a real confidence calibration on it — the vote vectors are already logged | M |
| 18 | Per-language relevance vocabularies gated on `page.lang`; route non-English pages to LLM refinement instead of the keyword gate | M |
| 19 | Let `data/domain_directory.json` override curated lists, not only fill their gaps | S |
| 20 | CI + `requirements-dev.txt` + `.gitignore` + model fixture to de-slow the suite | S |

**Item 15 is the highest-leverage item in this document.** Findings F5, F9,
F14 and half of the roadmap are all downstream of the same root cause: *the
only ground truth this project has is synthetic, generated by the same
assumptions the classifier encodes.* Until a few hundred real pages are
labelled by hand, every accuracy number in the repository — including the
self-evaluation printed at the end of every run — is measuring the pipeline
against itself. Nothing else on this list changes that.

---

## Part 5 · What is genuinely good

A review that only lists faults misrepresents the codebase. These are above
the standard of comparable pipelines and should be protected during any
refactor:

- **The `unknown` vs `unrelated` distinction, enforced via `content_scope`.**
  Most pipelines conflate "we could not see" with "we saw nothing relevant".
  This one does not, and the mechanism is structural rather than
  conventional.
- **The commit-log persistence design.** Products before page, page line as
  commit marker, bounded-memory parts, torn-tail-tolerant reader. The comments
  even record the O(n²) regression that motivated it.
- **`prepare_run()`'s input fingerprinting**, which refuses to silently merge
  two corpora into one output table.
- **The precision-first matcher.** Hard vetoes on brand/SPF/form disagreement
  and brand-token exclusion from name similarity show real domain thinking.
- **`readout.py` as a pure function of saved parquets**, with a provenance
  banner that flags synthetic runs so demo numbers cannot be mistaken for real
  ones.
- **The density and honesty of the inline commentary.** Nearly every constant
  has a rationale attached. That is what made this review possible at all.

---

## Part 6 · Verification log

Executed against the working tree at commit `b7abb8f` (branch `main`) on
Windows, Python venv `.conv`:

| Check | Result |
|---|---|
| `git status --porcelain` | 3 modified notebooks; `.conv/`, `.history/`, `.output/`, `.vscode/`, `output.png` untracked and un-ignored |
| `Test-Path .github` | **false** — no CI |
| `pip list` | `pandas 3.0.3`, `numpy 2.4.6`, `pyarrow 24.0.0`, `scikit-learn 1.9.0`, `bertopic 0.17.4`, `sentence-transformers 5.6.0`; **no `pytest`**, **no `bs4`/`lxml`** |
| `import bs4` | `ModuleNotFoundError` |
| `Select-String "^def test_" tests\*.py` | **66** tests (README claims 36) |
| `pytest tests/test_scraping.py` | aborted after **>25 min**; 3 trivial tests run in 1.0 s, `test_learned_model_channel` alone 24.7 s |
| `pytest -k test_directory_robots_blocked_fallback` | **FAILED** — `AssertionError: scope base`, reproducible in isolation (F16) |
| `outputs/scrape_cache` | **22 260** files, 8.6 GB; `_cache_path("https://www.sephora.com/")` exists — the cause of the failure above |
| `outputs/scrape/scraped_pages.jsonl` | **159.9 MB**, re-streamed twice per resumed run (F17) |
| `outputs/` total | ~**9.4 GB** across 9 subdirectories |
| `_to_float` on EU price formats | executed: `"12,50 EUR"→1250.0`, `"9,90"→990.0`, `"1.234,56"→1.23456`; US formats correct (F1) |
| `extract.py:270-273` | confirmed literal `pass` (F2) |
| `fetch.py:186,190-196` | confirmed in-memory only; `oks == 0` clause confirmed (F3) |
| `classify.py:127-130` | confirmed `.org.uk` absent from `_MULTI_TLD` (F6) |
| `classify.py:685-692` | confirmed `else 1.0` branch (F5) |
| `config.py:126` `llm_model="claude-opus-4-8"` | **not a defect** — valid identifier in the installed `anthropic` SDK |

Note on dependency drift: `requirements.txt` pins `pandas>=2.0`, but the venv
runs **pandas 3.0.3**, which changed copy-on-write and several dtype
behaviours. Given F13's fail-open handlers, a pandas-3 incompatibility would
surface as silently missing data rather than as an exception. Pinning upper
bounds is worth doing before the next real run.
