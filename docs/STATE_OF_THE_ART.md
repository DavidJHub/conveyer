# State of the art — agentic commerce, inferring agent recommendations, and their impact on purchase decisions

Literature and industry review compiled July 2026 for the conveyer project.
Companion docs: [PIPELINE_AND_HYPOTHESES.md](PIPELINE_AND_HYPOTHESES.md) (our
architecture) and [notebook 04](../notebooks/04_aces_and_research_questions.ipynb)
(our prototypes). Sources are cited inline; industry/vendor figures are flagged
as such — they come from blogs and press with weaker methodology than the
academic anchors.

---

## 1 · Framing

Three questions organise the field, and this review:

1. **What do AI shopping agents actually do?** (simulators, audits, bias
   catalogues)
2. **How can agent/LLM recommendations be *inferred* at scale?** (observational
   logs, active probing, structural choice models)
3. **What is the causal impact of those recommendations on purchase
   decisions?** (platform analytics, experiments, attribution/MMM)

The field is young: the core references are 2024–2026, several are preprints,
and industry measurement practice is ahead of — and less rigorous than — the
academic literature.

---

## 2 · The agentic-commerce landscape (why this matters now)

**Transaction rails are being standardised.** OpenAI and Stripe co-developed
the **Agentic Commerce Protocol (ACP)** and launched *Instant Checkout* inside
ChatGPT (Sept 2025) with Etsy and Shopify merchants
([OpenAI](https://openai.com/index/buy-it-in-chatgpt/),
[Stripe](https://stripe.com/newsroom/news/stripe-openai-instant-checkout));
Stripe's *Shared Payment Tokens* scope an agent's payment authority to a
single, time-limited transaction
([Stripe](https://stripe.com/blog/introducing-our-agentic-commerce-solutions)).
Perplexity shipped *Instant Buy* with PayPal (Nov 2025); Google and Shopify
answered with the **Universal Commerce Protocol (UCP)** at NRF (Jan 2026), and
Visa/Mastercard are building their own agent-token rails
([overview](https://opascope.com/insights/ai-shopping-assistant-guide-2026-agentic-commerce-protocols/)).
Industry reporting suggests the first iteration of Instant Checkout was
sunset in early 2026 after limited merchant adoption
([Ekamoira](https://www.ekamoira.com/blog/chatgpt-instant-checkout-agentic-commerce-protocol-2026),
vendor source — treat the specifics with caution), which is itself a datapoint:
the *rails* are volatile, but every protocol iteration moves purchases further
inside the assistant, where clickstream-based measurement cannot follow.

**Traffic and conversion signals are material.** Adobe Analytics reports AI
traffic to US retail sites up **393% YoY in Q1 2026**, with AI-referred
visitors converting **42% better** than non-AI traffic in March 2026 — a
reversal from 38% *worse* twelve months earlier
([Yahoo Finance/Adobe](https://finance.yahoo.com/sectors/technology/articles/ai-traffic-us-retailers-jumps-160141756.html),
[analysis](https://www.digitalapplied.com/blog/ai-traffic-converts-42-percent-better-2026-channel-strategy)).
Similarweb panel data has ChatGPT referrals converting ~11.4% vs ~5.3% for
organic search on e-commerce sites
([Practical Ecommerce](https://www.practicalecommerce.com/mixed-reports-on-ai-ecommerce-traffic)).
The academic anchor is a **Marketing Science** study of first-party data from
973 websites (~$20B revenue, 50k+ ChatGPT-referred transactions): ChatGPT
referrals convert ~**2× better than paid social but ~13% below organic
search** ([Marketing Science 2026](https://pubsonline.informs.org/doi/10.1287/mksc.2025.0489)).
The spread across these three sources is the measurement problem in miniature:
*AI-referred traffic is selected* (users arrive pre-persuaded), so raw
conversion comparisons say little about incrementality (§5).

**Consumers already act on assistant advice without verifying.** Survey
evidence: ~18% of consumers have purchased on an AI recommendation without
checking via search first (2.5× higher among Gen Z/millennials), and 59% say
an AI mention makes them likely to visit a brand's site
([eMarketer](https://www.emarketer.com/content/exclusive--consumers-turn-ai-purchase-recommendations-without-using-traditional-search)).

---

## 3 · What AI shopping agents do: simulators and audits

### 3.1 ACES — the reference audit framework (our simulator)

[Allouah, Besbes, Figueroa, Kanoria & Kumar (arXiv:2508.02630, v3 Dec
2025)](https://arxiv.org/abs/2508.02630) pair a VLM shopping agent with a
fully programmable mock marketplace and run randomized trials. Findings that
matter for us:

- **Choice homogeneity / winner-take-most:** agents concentrate demand on a
  few "modal" products and ignore the rest — AI-mediated demand is *spikier*
  than human demand.
- **Preference instability:** provider **model updates drastically reshuffle
  market shares** — brand outcomes depend on a policy the brand cannot see.
  (For attribution this is a gift: releases are natural experiments, §5.3.)
- **Position bias is strong, provider-specific, and survives text-only
  interfaces** — all models favour the top row but *different models prefer
  different columns*, so there is no universal "rank 1".
- **Badge asymmetry:** sponsored tags are consistently *penalized*; platform
  endorsements ("Overall pick") are rewarded.
- **Heterogeneous sensitivities:** price/rating/review elasticities differ
  sharply across providers and versions.
- **Sellers can fight back:** a seller-side agent making simple,
  query-conditional description tweaks wins significant share — the
  GEO-for-agents arms race is already demonstrated in-sim.

Our `conveyer.aces` module replicates ACES's randomisation protocol and adds a
**prompt-perturbation treatment layer** (§4.3 below) plus an offline logit
agent with known ground truth for estimator validation.

### 3.2 Other environments

- **Magentic Marketplace** (Microsoft,
  [arXiv:2510.25779](https://arxiv.org/pdf/2510.25779)): two-sided LLM
  buyer/seller markets at scale. Frontier models approach optimal welfare only
  under ideal search; a severe **first-proposal bias** gives 10–30× advantage
  to response *speed* over quality; a "paradox of choice" — welfare *falls* as
  consideration sets grow; and manipulation tests (fake credentials, fake
  social proof) fully compromise some models
  ([MSR blog](https://www.microsoft.com/en-us/research/blog/magentic-marketplace-an-open-source-simulation-environment-for-studying-agentic-markets/)).
- **Adversarial ranking manipulation:** *StealthRank*
  ([arXiv:2504.05804](https://arxiv.org/pdf/2504.05804)) optimises stealthy
  product-description strings that push a target item to rank 1 — and
  documents extreme fragility: single-token changes flip a product from rank 1
  to omitted entirely.
- **Prompt sensitivity in general:** meaning-preserving rewrites reorder
  outputs across LLM tasks
  ([Quantifying LLMs' sensitivity, arXiv:2406.12334](https://arxiv.org/html/2406.12334v2));
  production RAG recommenders show paraphrase brittleness below rerun-stability
  baselines ([arXiv:2605.27440](https://arxiv.org/pdf/2605.27440)). This is
  the direct SOTA motivation for our perturbation library: *subtle prompt
  changes are a first-class treatment variable, not noise*.

### 3.3 Bias catalogue from LLM-recommendation audits

- **Incumbent advantage / brand-popularity bias:** LLMs systematically
  over-recommend well-known brands relative to equally good alternatives
  ([Incumbent Advantage, arXiv:2606.17443](https://arxiv.org/html/2606.17443v1)).
- **Global-vs-local and socio-economic bias:** US-centric models favour global
  brands; luxury brands recommended 88–100% of the time for high-income
  contexts vs 84–98% non-luxury for low-income
  ([arXiv:2406.13997](https://arxiv.org/pdf/2406.13997)).
- **Cultural/brand preference audits** as a method:
  [arXiv:2603.18300](https://arxiv.org/html/2603.18300v1).
- **Cognitive-bias transfer:** anchoring, decoys and framing implanted in
  product context shift LLM recommendations
  ([Bias Beware, arXiv:2502.01349](https://arxiv.org/html/2502.01349v3)).
- **Linguistic bias:** query phrasing style itself shifts what gets
  recommended ([arXiv:2604.25456](https://arxiv.org/pdf/2604.25456)).
- **Market-structure concerns:** early work on vertical tacit collusion in
  AI-mediated markets ([arXiv:2601.03061](https://arxiv.org/pdf/2601.03061)).

**Synthesis:** the SOTA consensus is that agent demand is (i) concentrated,
(ii) unstable across model versions, (iii) strongly influenced by
*presentation* variables (position, badges) the seller/platform controls, and
(iv) sensitive to prompt wording users don't even notice. Any impact model
that treats "the LLM's recommendation" as a stable exogenous variable is
mis-specified from the start.

---

## 4 · How to infer agent recommendations

Three methodological families, complementary and increasingly structural:

### 4.1 Observational: mining conversation/referral logs (what we do)

Extract recommendation events from real assistant traffic: entity extraction
over answers, requested-vs-recommended splits, rank within answer, funnel
stage per turn — exactly the SimilarWeb star schema this repo profiles
(`fact_ai_recommendation.mention_type/rank`, `fact_ai_funnel`), plus our
graph layer separating **direct** (user-initiated) from **indirect**
(LLM-created) brand exposure. Strengths: real behaviour, session context,
volume. Limits: panel coverage (desktop browser only — our Q4), no
counterfactuals, purchases censored.

The sequence-modelling side has precedent: HMM/latent-state models for
purchase readiness and journey stages are established in marketing science
(e.g. duration-dependent latent-state exit models,
[arXiv:2208.03937](https://arxiv.org/pdf/2208.03937)), and recent work couples
LLMs with latent-intent models for conversational recommendation
([LatentCRS, arXiv:2503.10703](https://arxiv.org/pdf/2503.10703); multi-view
intent alignment, [ACM TOIS](https://dl.acm.org/doi/10.1145/3719344)). Our
journey-HMM (wiki §5) sits squarely on this line.

### 4.2 Active probing: "Share of Model" panels and audits

The GEO industry has converged on **polling-style sampling**: define a
representative panel of 250–500 high-intent category queries, submit them
repeatedly across providers, and track *mention rate*, *recommendation share*,
*rank*, *sentiment*, *citations* — turning stochastic generation into stable
estimates by sampling at scale
([Share of Model](https://www.symphonicdigital.com/blog/understanding-share-of-model),
[measurement framework](https://almcorp.com/blog/llm-consistency-recommendation-share-measurement-framework/)).
Commercial platforms differ mainly in sampling frame: Evertune samples LLM
APIs at category scale with demographic weighting; Profound joins front-end
answers with server logs and downstream conversion
([comparison](https://www.evertune.ai/resources/insights-on-ai/evertune-vs-profound-feature-by-feature-geo-platform-comparison)).
One large audit spans 11.1M citations across 571k AI answers and 363 brands
([Wellows](https://wellows.com/blog/audit-brand-visibility-on-llms/), vendor).
Academic audits (§3.3) use the same design with randomization and controls.

*Relation to our stack:* our prompt-perturbation library **is** an audit query
panel with a factorial design — and augmenting the observational DB with the
same operators (`conveyer.augment`) keeps probe vocabulary and log vocabulary
identical, which the industry tools cannot do (they see no session logs).

### 4.3 Structural: choice models over audit/simulator output

The frontier is moving from *counting mentions* to *estimating the utility
function that generates them*: conditional/rank-ordered logit on observed
ranks and choices, recovering elasticities to position, price, rating, badges
— ACES does randomized one-factor experiments; our Q6 fits the full MNL and
recovers the planted ground truth, giving *why*-level answers ("this provider
weights rating 3× more than that one") instead of share dashboards. Nothing
in the surveyed industry practice does this yet; the closest academic
analogue is the elasticity analysis inside ACES itself
([arXiv:2508.02630](https://arxiv.org/abs/2508.02630)). This is a genuine gap
conveyer occupies.

---

## 5 · Impact of agent recommendations on purchase decisions

### 5.1 Evidence tiers available today

| Tier | Evidence | Key numbers | Caveat |
|---|---|---|---|
| Platform analytics | Adobe: AI traffic +393% YoY, converts +42% vs non-AI (Mar 2026); Similarweb: 11.4% vs 5.3% organic | selection bias: AI users arrive pre-persuaded | [Adobe/Yahoo](https://finance.yahoo.com/sectors/technology/articles/ai-traffic-us-retailers-jumps-160141756.html), [Practical Ecommerce](https://www.practicalecommerce.com/mixed-reports-on-ai-ecommerce-traffic) |
| First-party academic | ChatGPT referrals ≈2× paid-social conversion, ≈13% below organic search (973 sites, 50k+ transactions) | referral-click sample only — misses in-chat influence | [Marketing Science 2026](https://pubsonline.informs.org/doi/10.1287/mksc.2025.0489) |
| Survey | 18% bought on AI advice w/o verification; 59% visit brand site after AI mention | stated, not revealed, behaviour | [eMarketer](https://www.emarketer.com/content/exclusive--consumers-turn-ai-purchase-recommendations-without-using-traditional-search) |
| Behavioural experiments | chatbot response strategy, disclosure, trust and anthropomorphism shift purchase intention; consumers still prefer search engines for evaluation | lab settings, intention ≠ purchase | [MDPI](https://www.mdpi.com/0718-1876/20/2/93), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11851727/), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10206356/) |
| Simulation | ACES/Magentic: presentation levers (position, badges, description text) move agent market shares by large factors | sim-to-real transfer unproven | [arXiv:2508.02630](https://arxiv.org/abs/2508.02630), [arXiv:2510.25779](https://arxiv.org/pdf/2510.25779) |

### 5.2 The attribution problem is a "dark funnel" problem

The influence mostly happens **inside the chat**, before any measurable click:
recommendations shape the consideration set (§3), then the purchase happens
via organic search, direct visit, marketplace app, or offline — channels that
attribution systems credit instead. AI referral clicks (the only thing
GA4-style attribution sees) are the tip; the Marketing Science referral study
explicitly measures only that tip. Industry practice therefore treats AI
influence like TV/brand marketing: **measure it as a channel in MMM +
incrementality experiments**, not by last-click
([attribution vs MMM vs incrementality](https://www.haus.io/open-haus/attribution-vs-traditional-mmm-vs-incrementality-the-ultimate-measurement-showdown),
[causal MMM guide](https://lifesight.io/blog/causal-marketing-mix-modeling-guide/)).

### 5.3 Identification strategies (ranked for our setting)

1. **Model-release event studies / natural experiments.** ACES shows updates
   reshuffle agent market shares discontinuously; combined with §4.2 panels
   that timestamp recommendation-share jumps per brand, releases give
   difference-in-differences and synthetic-control designs on sales — no
   cooperation from the provider needed. (Our Q5 hypothesis.)
2. **Geo/holdout incrementality tests** where a brand can modulate its own
   AI-visible surface (content, feeds, ACP participation) by region — the
   standard geo-split design ([Lifesight](https://lifesight.io/blog/causal-marketing-mix-modeling-guide/)).
3. **Bayesian MMM with experiment-calibrated priors**: weekly
   recommendation-exposure volume (from logs/panels, poststratified to market
   — our Q4) enters with adstock+saturation next to search/social/offline;
   geo/event studies pin the channel coefficient's prior. This is the
   PyMC-Marketing-style workflow and matches our Q5 prototype.
4. **Structural composition** (our Q7): P(buy) = P(reach Intent) ×
   P(brand in consideration set) × P(choose | set) × P(convert | choose),
   with the middle two terms estimated from logs (HMM + graph) and simulator
   (MNL) respectively — this is what lets *offline* sales be attributed via
   the same consideration-set channel.

### 5.4 What no one has yet (the open gaps)

- **Sim-to-real validation:** nobody has shown that elasticities estimated in
  ACES-style sandboxes predict real assistant-referred purchase behaviour.
  Our loop (simulate → estimate → intervene → verify on logs) is designed to
  test exactly this.
- **In-chat exposure → offline purchase linkage** at the individual level
  remains unsolved (privacy + data access); MRP-style poststratification and
  MMM remain the honest tools.
- **Session-level journey dynamics:** the audit industry counts mentions per
  query; the academic funnel literature models clickstream. Joint
  chat+clickstream latent-state models over *real assistant sessions* (our
  Q1 IO-HMM) are essentially unpublished territory.
- **Stability accounting:** given documented model-update volatility, any
  reported "share of model" without a version/date dimension is already
  stale; treating the provider policy as a time-varying latent state is open.

---

## 6 · Positioning conveyer against the SOTA

| SOTA practice | conveyer's move |
|---|---|
| Mention-counting panels (GEO tools) | + structural MNL utility recovery (Q6) and Shapley intervention ranking (Q8) |
| One-factor randomized audits (ACES) | + factorial **prompt-perturbation arms** shared between simulator and DB augmentation |
| Referral-click conversion studies | + dark-funnel composition model (Q7) and Beta-binomial censored-purchase estimation (Q3) |
| Desktop-panel observational data | + MRP transportability to mobile / other providers (Q4) |
| Per-turn funnel tagging (`fact_ai_funnel`) | + session-level journey HMM with covariate-driven transitions (Q1) |
| MMM without an AI channel | + LLM-recommendation exposure as an adstocked, saturating channel calibrated on model-release events (Q5) |

---

## 7 · Curated reading list

**Core (read first)**
1. [What Is Your AI Agent Buying? (ACES) — arXiv:2508.02630](https://arxiv.org/abs/2508.02630)
2. [ChatGPT Referrals to E-Commerce Websites — Marketing Science 2026](https://pubsonline.informs.org/doi/10.1287/mksc.2025.0489)
3. [Magentic Marketplace — arXiv:2510.25779](https://arxiv.org/pdf/2510.25779)

**Biases & manipulation**
4. [Incumbent Advantage: Brand Bias in LLM Recommendation — arXiv:2606.17443](https://arxiv.org/html/2606.17443v1)
5. [Bias Beware: Cognitive Biases in LLM Recommendations — arXiv:2502.01349](https://arxiv.org/html/2502.01349v3)
6. [Global is Good, Local is Bad? Brand Bias in LLMs — arXiv:2406.13997](https://arxiv.org/pdf/2406.13997)
7. [StealthRank: Ranking Manipulation via Prompt Optimization — arXiv:2504.05804](https://arxiv.org/pdf/2504.05804)
8. [Auditing Preferences for Brands and Cultures — arXiv:2603.18300](https://arxiv.org/html/2603.18300v1)

**Sequence/intent modelling**
9. [Duration-Dependent Latent-State Exit Model — arXiv:2208.03937](https://arxiv.org/pdf/2208.03937)
10. [LatentCRS: Variational EM for LLM Conversational Recommendation — arXiv:2503.10703](https://arxiv.org/pdf/2503.10703)

**Measurement & attribution practice**
11. [Share of Model methodology](https://www.symphonicdigital.com/blog/understanding-share-of-model) ·
    [Evertune vs Profound](https://www.evertune.ai/resources/insights-on-ai/evertune-vs-profound-feature-by-feature-geo-platform-comparison)
12. [Causal MMM guide (Lifesight)](https://lifesight.io/blog/causal-marketing-mix-modeling-guide/) ·
    [Attribution vs MMM vs incrementality (Haus)](https://www.haus.io/open-haus/attribution-vs-traditional-mmm-vs-incrementality-the-ultimate-measurement-showdown)

**Rails & market context**
13. [Buy it in ChatGPT (OpenAI/ACP)](https://openai.com/index/buy-it-in-chatgpt/) ·
    [Stripe agentic commerce](https://stripe.com/blog/introducing-our-agentic-commerce-solutions)
14. [Adobe: AI traffic +393% Q1 2026](https://finance.yahoo.com/sectors/technology/articles/ai-traffic-us-retailers-jumps-160141756.html) ·
    [Mixed reports on AI e-commerce traffic](https://www.practicalecommerce.com/mixed-reports-on-ai-ecommerce-traffic)
15. [eMarketer: consumers buying on AI advice](https://www.emarketer.com/content/exclusive--consumers-turn-ai-purchase-recommendations-without-using-traditional-search)
