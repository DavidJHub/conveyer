"""Model layer: embedding backends, optional sentiment model, LLM client.

Every backend auto-resolves to the best available option and degrades
gracefully, so nothing here hard-fails in a minimal environment:

* embeddings: Voyage (VOYAGE_API_KEY) → OpenAI (OPENAI_API_KEY)
  → sentence-transformers (local) → TF-IDF + SVD (always works);
* sentiment: HuggingFace ``transformers`` pipeline when installed
  → the lexicon scorer in :mod:`conveyer.conversations` otherwise;
* LLM: Anthropic Messages API when ``anthropic`` + ``ANTHROPIC_API_KEY``
  are present, else ``None``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


@dataclass
class ModelConfig:
    """Knobs for the model layer (embedded by each module's own config)."""
    embedding_backend: str = "auto"   # auto | voyage | openai | sentence_transformers | tfidf
    voyage_model: str = "voyage-3-large"
    openai_model: str = "text-embedding-3-large"
    st_model: str = "BAAI/bge-large-en-v1.5"
    normalize_embeddings: bool = True
    embed_batch_size: int = 128
    tfidf_components: int = 256
    random_state: int = 42
    # sentiment
    sentiment_backend: str = "auto"   # auto | transformers | lexicon
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    # LLM
    llm_model: str = "claude-opus-4-8"  # override via ANTHROPIC_MODEL env var
    llm_max_tokens: int = 400


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #
class BaseEmbedder:
    name: str = "base"

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI embeddings (Anthropic-recommended). Needs VOYAGE_API_KEY."""

    def __init__(self, cfg: ModelConfig):
        import voyageai

        self.cfg = cfg
        self.client = voyageai.Client()
        self.name = f"voyage:{cfg.voyage_model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out: List[List[float]] = []
        bs = self.cfg.embed_batch_size
        for i in range(0, len(texts), bs):
            resp = self.client.embed(texts[i:i + bs], model=self.cfg.voyage_model,
                                     input_type="document")
            out.extend(resp.embeddings)
        return _maybe_normalize(np.asarray(out, dtype=np.float32), self.cfg)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings. Needs OPENAI_API_KEY."""

    def __init__(self, cfg: ModelConfig):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI()
        self.name = f"openai:{cfg.openai_model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        out: List[List[float]] = []
        bs = self.cfg.embed_batch_size
        for i in range(0, len(texts), bs):
            resp = self.client.embeddings.create(model=self.cfg.openai_model,
                                                 input=texts[i:i + bs])
            out.extend([d.embedding for d in resp.data])
        return _maybe_normalize(np.asarray(out, dtype=np.float32), self.cfg)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local open-source embeddings via sentence-transformers."""

    def __init__(self, cfg: ModelConfig):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.st_model)
        self.name = f"sentence-transformers:{cfg.st_model}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=self.cfg.embed_batch_size,
                                show_progress_bar=False,
                                normalize_embeddings=self.cfg.normalize_embeddings)
        return np.asarray(emb, dtype=np.float32)


class TfidfEmbedder(BaseEmbedder):
    """Fully local fallback: TF-IDF + TruncatedSVD (LSA)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = "tfidf+svd"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        texts = list(texts)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000,
                                stop_words="english")
        X = tfidf.fit_transform(texts)
        k = min(self.cfg.tfidf_components, X.shape[1] - 1, max(2, X.shape[0] - 1))
        emb = TruncatedSVD(n_components=k, random_state=self.cfg.random_state) \
            .fit_transform(X).astype(np.float32)
        self.name = f"tfidf+svd({k})"
        return normalize(emb) if self.cfg.normalize_embeddings else emb


def _maybe_normalize(emb: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    if cfg.normalize_embeddings:
        from sklearn.preprocessing import normalize

        return normalize(emb)
    return emb


def _resolve_backend(cfg: ModelConfig) -> str:
    if cfg.embedding_backend != "auto":
        return cfg.embedding_backend
    if os.environ.get("VOYAGE_API_KEY") and _have("voyageai"):
        return "voyage"
    if os.environ.get("OPENAI_API_KEY") and _have("openai"):
        return "openai"
    if _have("sentence_transformers"):
        return "sentence_transformers"
    return "tfidf"


def build_embedder(cfg: Optional[ModelConfig] = None) -> BaseEmbedder:
    """Instantiate the configured (or best available) embedding backend."""
    cfg = cfg or ModelConfig()
    backend = _resolve_backend(cfg)
    builders = {"voyage": VoyageEmbedder, "openai": OpenAIEmbedder,
                "sentence_transformers": SentenceTransformerEmbedder,
                "tfidf": TfidfEmbedder}
    if backend not in builders:
        raise ValueError(f"Unknown embedding_backend: {backend}")
    try:
        return builders[backend](cfg)
    except Exception as exc:  # missing dep / key at construction time
        if backend == "tfidf":
            raise
        print(f"[models] backend '{backend}' unavailable ({exc}); falling back to TF-IDF.")
        return TfidfEmbedder(cfg)


# --------------------------------------------------------------------------- #
# Optional transformer sentiment
# --------------------------------------------------------------------------- #
def build_sentiment_pipeline(cfg: Optional[ModelConfig] = None):
    """A HuggingFace sentiment pipeline, or None (caller falls back to lexicon)."""
    cfg = cfg or ModelConfig()
    if cfg.sentiment_backend == "lexicon":
        return None
    if not _have("transformers"):
        if cfg.sentiment_backend == "transformers":
            print("[models] transformers requested but not installed; using lexicon.")
        return None
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model=cfg.sentiment_model, truncation=True)
    except Exception as exc:
        print(f"[models] sentiment pipeline unavailable ({exc}); using lexicon.")
        return None


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, cfg: ModelConfig):
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
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def build_llm(cfg: Optional[ModelConfig] = None) -> "AnthropicLLM | None":
    cfg = cfg or ModelConfig()
    if not _have("anthropic") or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return AnthropicLLM(cfg)


def detect_capabilities() -> dict:
    """Report which optional dependencies are importable."""
    return {m: _have(m) for m in
            ("voyageai", "openai", "sentence_transformers", "bertopic",
             "transformers", "bs4", "anthropic")}
