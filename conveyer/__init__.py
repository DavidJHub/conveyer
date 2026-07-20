"""conveyer — analysis of skincare LLM conversations for agentic e-commerce.

Public API:
    PipelineConfig   configuration (column mapping, models, hyper-parameters)
    run_pipeline     end-to-end clustering/analysis orchestration
    ingest, models, clustering, analysis, viz   the pipeline stages
    aces         ACES simulator integration (prompt perturbations, experiment
                 grids, offline logit agent)
    augment      DB augmentation with perturbed prompts
    funnel       purchase-journey stage inference (text classifier + HMM)
    graphs       user x session x entity interaction network
    attribution  Bayesian conversion, MRP extrapolation, MMM, MNL utility
                 model, Shapley explanations
    scraping     scrape surfaced pages, classify them (catalogue/landing/shopping/
                 …), extract product metadata and match it to the chat recommendation
"""
from . import (aces, analysis, attribution, augment, clustering, funnel,
               graphs, ingest, models, scraping, viz)
from .config import PipelineConfig
from .pipeline import run_pipeline

__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "ingest",
    "models",
    "clustering",
    "analysis",
    "viz",
    "aces",
    "augment",
    "funnel",
    "graphs",
    "attribution",
    "scraping",
]

__version__ = "0.2.0"
