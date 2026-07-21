# The funnel model — design, measurement and limits

Module 3 (`conveyer.journey`) answers the project's two headline questions on
the conversations dataset. This page is the specification: what is measured,
how, and what it can and cannot claim.

## 1 · Event grammar

Every turn contributes three kinds of behavioural evidence, normalised into
one event table (`fact_funnel_event`):

| event | source column | meaning |
|---|---|---|
| `cited` | `a_links_source` | the agent put a link in front of the user (**exposure**, not behaviour) |
| `ai_click` | `ai_click` | the user clicked a link straight from the answer (strongest follow signal) |
| `trail_visit` | `next_10_urls` | a post-turn navigation event, with 1-based position and dwell (`request_time` delta to the next request; the last event never gets one) |

Each event URL is classified by module 2 (scraped page → base-URL fallback →
URL/domain heuristics), so every event carries a `page_category`,
`funnel_stage`, `seller_type` and `commerce_depth`
(0 none · 1 shopping · 2 cart · 3 checkout/order).

## 2 · Linking behaviour to the conversation

Two independent bridges tie what the user *did* to what the agent *said*:

* **Link identity** — `followed_agent_link`: an `ai_click`, or a `trail_visit`
  to a URL the answer cited.
* **Brand identity** — `brand_match`: the page's canonical brand (domain, OG
  tag, or the brands of products extracted from it) against the turn's
  mentions. `unsolicited_rec` — the agent introduced the brand, the user never
  asked — is the exposure of interest; `endorsed` / `requested` mean the user
  already had the brand in mind (cf. the direct-vs-indirect split in paid-
  search attribution, Li et al. 2016).

`followed_recommendation = followed_agent_link ∨ visited_recommended_brand`
is the session-level treatment indicator.

## 3 · Outcome — an explicit proxy

The schema contains no purchase confirmation, so **conversion is defined as
reaching a Purchase-stage page in the post-turn trail**:
`converted := max_commerce_depth ≥ depth(conversion_stage)` with
`conversion_stage ∈ {shopping, cart, checkout}` (default **cart**). This is a
documented, adjustable proxy — cart-reach overstates purchase (abandonment)
and misses purchases outside the 10-event window or on other devices
(identity fragmentation; Lin & Misra 2022).

## 4 · Measurement

1. **Stratified conversion rates** with Jeffreys 95% intervals
   (`Beta(s+½, n−s+½)`) for each exposure stratum, plus the lift
   `rate(exposed) / rate(unexposed)`.
2. **Funnel transition matrix** — the journey as a stage sequence: the HMM-
   smoothed conversation stages (Awareness…Post-Purchase, `conveyer.funnel`)
   followed by the visited pages' stages in trail order; row-normalised
   transition probabilities.
3. **The predictive models** — two holdout-validated logistic regressions on
   standardized features:
   * the **exposure model** (the headline): conversation features (turns,
     intent, sentiment, unsolicited recommendations, cited links, max stage
     asked) + the treatment flags (`followed_agent_link`,
     `visited_recommended_brand`), with **no post-treatment mediators** —
     visit counts and dwell are consequences of following a recommendation,
     and controlling for them would absorb the effect being measured;
   * the **full model**: adds the behavioural features back for maximum
     predictive power — used to predict, not to interpret exposure.
   Standardized coefficients / odds-ratios-per-SD make the **weight of agent
   recommendations** directly comparable to every other driver.

On synthetic data the run closes the loop: generated archetypes carry ground
truth, and `evaluate_against_gt` reports conversion-proxy and exposure-
detection accuracy (both ≥ 0.95 in the test suite).

## 5 · Limitations (read before quoting numbers)

* **Association, not causation.** Users who follow recommendations
  self-select; the exposure coefficient is an upper bound on the causal
  effect. The dataset has no randomisation; ACES-style agent simulation or a
  holdout experiment would be needed for identification.
* **Proxy outcome** (§3) and a **10-event window** truncate journeys.
* **Brand lexicon coverage** bounds brand matching; extend
  `conveyer.brands.BRANDS` for niche brands.
* **Trail ≠ full clickstream** — only the 10 requests after each prompt;
  dwell includes idle time.
* Persuasion research (Salvi et al. 2026: LLM steering nearly triples
  sponsored selection; Meguellati et al. 2024/2025) implies exposure effects
  may be large and hard for users to detect — one more reason to keep the
  measurement observational-honest.

## 6 · Roadmap

* Covariate-dependent HMM transitions (recommendation events shifting
  P(Discovery→Evaluation)) — the machinery in `conveyer.funnel` supports it.
* Per-brand random effects once multiple sessions per brand accumulate.
* Purchase-confirmation enrichment (order-confirmation page patterns) to
  tighten the outcome proxy.

## Key references

Kaiser & Schulze (2026) *ChatGPT referrals to e-commerce websites*, Mark. Sci. ·
Salvi, Cuevas & Horta Ribeiro (2026) *Commercial persuasion in AI-mediated
conversations* · Cao & Hu (2026) *A solicit-then-suggest model of agentic
purchasing* · Wu & Bao (2025) *Advertising in AI systems* · Li, Kannan,
Viswanathan & Pani (2016) *Attribution strategies and ROI in paid search*,
Mark. Sci. · Lin & Misra (2022) *The identity fragmentation bias*, Mark. Sci.
Full annotated list: [STATE_OF_THE_ART.md](STATE_OF_THE_ART.md).
