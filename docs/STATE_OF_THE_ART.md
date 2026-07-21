# State of the art — where this project sits

Annotated bibliography for the funnel-model mission: measuring conversion and
the weight of agent recommendations in LLM-mediated shopping. BibTeX for every
entry: [references.bib](references.bib).

## 1 · LLM traffic → e-commerce (the phenomenon we model)

* **Kaiser & Schulze (2026), *ChatGPT referrals to e-commerce websites*,
  Marketing Science** — first descriptive evidence at scale (973 shops):
  organic-LLM traffic differs from traditional channels in engagement and
  financial metrics. Our dataset is the *user side* of exactly this pipe:
  conversation → surfaced link → visit → (proxy) purchase.
* **Cao & Hu (2026), *A solicit-then-suggest model of agentic purchasing*** —
  theory for the conversational funnel: solicitation depth substitutes for
  assortment breadth (loss ~1/m vs ~k^(−2/d)). Motivates our per-turn stage +
  recommendation-count features: a few good questions move users down-funnel
  faster than more options.

## 2 · Commercial influence inside AI answers (why exposure weight matters)

* **Salvi, Cuevas & Horta Ribeiro (2026), *Commercial persuasion in
  AI-mediated conversations*** — preregistered (N=2,012): LLM steering nearly
  **triples** sponsored-product selection vs search placement (61% vs 22%),
  mostly undetected; disclosure labels barely help. If exposure effects are
  this large, quantifying them per journey (our module 3) is the core
  measurement problem.
* **Wu & Bao (2025), *Advertising in AI systems*** — design space and
  stakeholder incentives for commercial content in generative answers; argues
  provenance/transparency will lag. Our `brands_unsolicited` feature is the
  observable trace of exactly this channel.
* **Tang et al. (2025), *Ads that talk back*** — users struggle to detect ads
  embedded in chatbot responses and often prefer them.
* **Meguellati et al. (2024, 2025)** — LLM-generated ads match human experts
  on personalization and beat them on persuasion principles (authority,
  consensus) at near-zero marginal cost; Tohidi et al. (2025) show framing
  alone shifts attitudes. Together: the *content* of agent answers is a
  persuasion instrument, supporting sentiment/enthusiasm features.

## 3 · Monetization mechanics (the supply side of recommendations)

* **Xu et al. (2026), *Ad insertion in LLM-generated responses*** and
  **Zhao et al. (2025), *LLM-Auction*** — auction mechanisms for allocating
  ad content inside generated answers; **Hu et al. (2025), *GEM-Bench*** —
  benchmark for ad-injected response generation; **Yin (2025), *InfoBid*** —
  LLM simulation of information disclosure in auctions; **Wu & Zhu (2024)** —
  survey of large-model advertising auctions. These describe how
  recommendation slots may be *priced*; our model estimates what a slot is
  *worth* downstream (conversion weight).

## 4 · Agents as ad consumers (the mirror problem)

* **Nitu, Mühle & Stöckl (2025), *Machine-readable ads*** and **Stöckl & Nitu
  (2025), *Are AI agents interacting with online ads?*** — web agents ignore
  visual ads, favour structured data, and sometimes complete purchases
  uncritically. Relevant to our module 2: machine-readable structure
  (schema.org) is exactly what our scraper reads, and agent-driven traffic
  will increasingly share the clickstream with humans.

## 5 · Attribution & measurement methodology

* **Li, Kannan, Viswanathan & Pani (2016), *Attribution strategies and ROI in
  paid search*, Marketing Science** — attribution changes optimal spend; our
  direct (`requested`/`endorsed`) vs indirect (`unsolicited_rec`) brand-match
  split mirrors their channel-credit problem.
* **Lin & Misra (2022), *The identity fragmentation bias*, Marketing
  Science** — cross-device fragmentation biases effect estimates; our
  desktop-panel trail has the same blind spot (documented in
  [FUNNEL_MODEL.md](FUNNEL_MODEL.md) §5).
* **Chen et al. (2025/2026), *When ads become profiles*** — LLMs reconstruct
  private attributes from ad streams alone; a caution for how much a browsing
  trail reveals, and why panel data stays anonymised.

## 6 · The gap conveyer fills

Descriptive channel studies (§1) show LLM referrals exist; lab experiments
(§2) show steering works; mechanism papers (§3) price the slots. What's
missing is **journey-level measurement on observational data**: per-session
linkage of what the agent said → what the user visited → whether they reached
purchase, with the exposure weight estimated jointly with intent, sentiment
and stage. That table (`journey_features.parquet`) and its model are this
project's contribution.
