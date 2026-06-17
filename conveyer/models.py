"""Model layer: embedding backends and the LLM client.

The embedder is chosen by :func:`build_embedder`. With ``embedding_backend="auto"``
it resolves the best available option at runtime, preferring API models when
their keys are present and degrading gracefully to a fully local TF-IDF backend
so nothing hard-fails in a minimal environment:

    Voyage (voyage-3-large)  ->  OpenAI (text-embedding-3-large)
        ->  sentence-transformers (open)  ->  TF-IDF + SVD
"""
from __future__ import annotations

import importlib
import os
from typing import List, Sequence

import numpy as np

from .config import PipelineConfig


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #
class BaseEmbedder:
    name: str = "base"

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI embeddings (Anthropic-recommended). Needs VOYAGE_API_KEY."""

    def __init__(self, cfg: PipelineConfig):
        import voyageai

        self.cfg = cfg
        self.client = voyageai.Client()
        self.name = f"voyage:{cfg.voyage_model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out: List[List[float]] = []
        bs = self.cfg.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            resp = self.client.embed(batch, model=self.cfg.voyage_model, input_type="document")
            out.extend(resp.embeddings)
        emb = np.asarray(out, dtype=np.float32)
        return _maybe_normalize(emb, self.cfg)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings. Needs OPENAI_API_KEY."""

    def __init__(self, cfg: PipelineConfig):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI()
        self.name = f"openai:{cfg.openai_model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        out: List[List[float]] = []
        bs = self.cfg.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            resp = self.client.embeddings.create(model=self.cfg.openai_model, input=batch)
            out.extend([d.embedding for d in resp.data])
        emb = np.asarray(out, dtype=np.float32)
        return _maybe_normalize(emb, self.cfg)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local open-source embeddings via sentence-transformers."""

    def __init__(self, cfg: PipelineConfig):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.st_model)
        self.name = f"sentence-transformers:{cfg.st_model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.cfg.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.cfg.normalize_embeddings,
        )
        return np.asarray(emb, dtype=np.float32)


class TfidfEmbedder(BaseEmbedder):
    """Fully local fallback: TF-IDF + TruncatedSVD (LSA)."""

    def __init__(self, cfg: PipelineConfig, n_components: int = 256):
        self.cfg = cfg
        self.n_components = n_components
        self.name = "tfidf+svd"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        texts = list(texts)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, stop_words="english")
        X = tfidf.fit_transform(texts)
        k = min(self.n_components, X.shape[1] - 1, max(2, X.shape[0] - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.cfg.random_state)
        emb = svd.fit_transform(X).astype(np.float32)
        self.name = f"tfidf+svd({k})"
        if self.cfg.normalize_embeddings:
            emb = normalize(emb)
        return emb


def _maybe_normalize(emb: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    if cfg.normalize_embeddings:
        from sklearn.preprocessing import normalize

        return normalize(emb)
    return emb


def _resolve_backend(cfg: PipelineConfig) -> str:
    if cfg.embedding_backend != "auto":
        return cfg.embedding_backend
    if os.environ.get("VOYAGE_API_KEY") and _have("voyageai"):
        return "voyage"
    if os.environ.get("OPENAI_API_KEY") and _have("openai"):
        return "openai"
    if _have("sentence_transformers"):
        return "sentence_transformers"
    return "tfidf"


def build_embedder(cfg: PipelineConfig) -> BaseEmbedder:
    """Instantiate the configured (or best available) embedding backend."""
    backend = _resolve_backend(cfg)
    builders = {
        "voyage": VoyageEmbedder,
        "openai": OpenAIEmbedder,
        "sentence_transformers": SentenceTransformerEmbedder,
        "tfidf": TfidfEmbedder,
    }
    if backend not in builders:
        raise ValueError(f"Unknown embedding_backend: {backend}")
    try:
        return builders[backend](cfg)
    except Exception as exc:  # missing dep / missing key at construction time
        if backend == "tfidf":
            raise
        print(f"[models] backend '{backend}' unavailable ({exc}); falling back to TF-IDF.")
        return TfidfEmbedder(cfg)


# --------------------------------------------------------------------------- #
# LLM client (cluster naming / zero-shot intent)
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, cfg: PipelineConfig):
        import anthropic

        self.cfg = cfg
        self.model = os.environ.get("ANTHROPIC_MODEL", cfg.llm_model)
        self.client = anthropic.Anthropic()

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs = dict(model=self.model, max_tokens=self.cfg.llm_max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        msg = self.client.messages.create(**kwargs)
        return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")


def build_llm(cfg: PipelineConfig) -> "AnthropicLLM | None":
    """Return an LLM client if any LLM-backed feature is enabled and usable."""
    wants_llm = cfg.use_llm_naming or cfg.llm_keyphrase_expansion or cfg.llm_select_granularity
    if not wants_llm:
        return None
    if not _have("anthropic") or not os.environ.get("ANTHROPIC_API_KEY"):
        print("[models] an LLM feature was requested but anthropic/ANTHROPIC_API_KEY is missing; skipping.")
        return None
    return AnthropicLLM(cfg)


def detect_capabilities() -> dict:
    """Report which optional dependencies are importable."""
    return {m: _have(m) for m in
            ("voyageai", "openai", "sentence_transformers", "bertopic", "umap", "hdbscan", "anthropic")}
