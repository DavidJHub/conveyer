# conveyer

Insights for **agentic e-commerce** from a database of skincare conversations
with ChatGPT (US, Jan 2026), joined with the SimilarWeb star schema
(funnel stages, topics, extracted product entities) and augmented with the
**[ACES](https://github.com/mycustomai/ACES) agentic-commerce simulator**
(arXiv:2508.02630) to study how subtle prompt changes move agent purchase
decisions.

## Project layout

```
conveyer/                     the Python package
  config.py                   PipelineConfig — column mapping, models, hyper-parameters
  ingest.py                   load/parse data, derive features, synthetic fallback
  models.py                   embedding backends + Anthropic LLM helpers
  clustering.py               kmeans/agglomerative/spectral/gmm/hdbscan + BERTopic + LLM-assisted
  analysis.py                 brand/recommendation features, intent, amplification
  viz.py                      interactive Plotly dashboard
  pipeline.py                 clustering pipeline orchestration + CLI
  ── new (v0.2) ──────────────────────────────────────────────────────────
  aces.py                     ACES integration: prompt perturbations, experiment grids in
                              ACES's local-dataset schema, offline logit agent with known
                              ground-truth utility weights, choice-share analysis
  augment.py                  DB augmentation with perturbed prompts + label-flip robustness
  funnel.py                   journey-stage inference: keyword classifier + discrete HMM (Q1)
  graphs.py                   user x session x entity network, direct/indirect exposure (Q2)
  attribution.py              Bayesian conversion (Q3), MRP extrapolation (Q4), MMM (Q5),
                              conditional-logit utility model (Q6), exact Shapley (Q8)

notebooks/
  01_text_clustering_pipeline.ipynb   conversation clustering & segment characterisation
  02_similarweb_star_schema.ipynb     profile of the SimilarWeb transformed data (7 tables)
  03_message_enrichment.ipynb         transformer classifiers: intent, sentiment, brand pairs
  04_aces_and_research_questions.ipynb  ⭐ master notebook — ACES integration, reorganised
                                      results, and a hypothesis + working prototype for each
                                      research question
  archive/                            earlier iterations kept for reference

data/                        datasets (not versioned) — see data/README.md
outputs/                     generated CSVs, ACES experiment exports, dashboards
```

## Quickstart

```bash
pip install -r requirements.txt

# master notebook (runs on synthetic data when the real files are absent)
jupyter lab notebooks/04_aces_and_research_questions.ipynb

# clustering pipeline CLI
python -m conveyer.pipeline --data data/conversations.parquet --n-clusters 12
```

## ACES integration

`conveyer.aces` connects the conversation data to the ACES simulator in both
directions:

1. **conveyer → ACES.** `product_catalog_from_conversations` turns the brands
   the LLM actually recommends into an ACES-style SKU catalog;
   `build_aces_experiments` replicates ACES's randomisation protocol
   (position permutation, log-normal price noise, rating shift, sponsored /
   overall-pick / low-stock badges) and `to_aces_csv` exports the grid in the
   exact schema `run.py --local-dataset` expects. Run it against real VLM
   agents with a clone of ACES:

   ```bash
   git clone https://github.com/mycustomai/ACES && cd ACES
   uv sync --all-packages && cp .env.sample .env   # add provider API keys
   uv run run.py --local-dataset <outputs/aces/conveyer_experiments.csv> --runtime-type screenshot
   ```

2. **Prompt dynamics.** `PERTURBATIONS` is a library of subtle prompt rewrites
   (politeness, urgency, budget, brand loyalty, ad skepticism, typos, social
   proof). They serve as treatment arms for ACES experiments and as the
   augmentation operators for the conversation DB (`conveyer.augment`).

3. **Offline fallback.** `AgentChoiceModel` is a multinomial-logit agent with
   *known* utility weights (position bias, price/rating sensitivity, badge
   effects, prompt-conditioned modifiers). The full chain — experiment grid →
   agent choices → choice shares → utility-model recovery → Shapley — runs
   with no API keys, and doubles as a test bench: estimators in
   `conveyer.attribution` must recover the simulator's ground truth.

## Research questions → where they live

| # | Question | Module | Notebook §
|---|----------|--------|-----------
| Q1 | Infer user purchase-journey stage | `funnel` (classifier + HMM) | 4.1
| Q2 | LLM turns + entities (+clickstream) as a network; direct vs indirect attribution | `graphs` | 4.2
| Q3 | Estimate conversion from clickstream | `attribution.beta_binomial_posterior` | 4.3
| Q4 | Extrapolate to unseen segments (mobile, other LLMs) | `attribution.shrink_segments` + `poststratify` | 4.4
| Q5 | Attribute online/offline sales to LLM recommendations | `attribution.fit_mmm` (+ causal design notes) | 4.5
| Q6 | Utility model of LLM product ranking | `attribution.fit_mnl` | 4.6
| Q7 | Conversion probability from intent x product x text | notebook prototype | 4.7
| Q8 | Shapley explanations + interventions | `attribution.exact_shapley` | 4.8

## Clustering pipeline (v0.1, still available)

`cluster_method="auto"` compares kmeans / agglomerative / spectral / gmm /
hdbscan by cosine silhouette; optional LLM-assisted steps (keyphrase
expansion, ClusterLLM-style granularity) need `ANTHROPIC_API_KEY`; embedding
backend auto-resolves Voyage → OpenAI → sentence-transformers → TF-IDF. See
`notebooks/01_text_clustering_pipeline.ipynb` and the module docstrings.

Without a dataset everything runs on English synthetic conversations, so the
whole repo is testable before plugging in real data.
