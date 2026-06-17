"""conveyer — clustering & analysis of skincare LLM conversations (English).

Public API:
    PipelineConfig   configuration (column mapping, models, hyper-parameters)
    run_pipeline     end-to-end orchestration
    ingest, models, clustering, analysis   the individual stages
"""
from . import analysis, clustering, ingest, models, viz
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
]

__version__ = "0.1.0"
