"""Self-trained page classifier — the learned voting channel.

The rule channels are precise but enumerable; this module adds a **trainable
multinomial logistic model** over hashed features so the classifier can also
learn — from the synthetic ground truth out of the box, and from your own
labelled (or confidently rule-labelled) ``scraped_pages`` parquet as real data
accumulates. It predicts the structural **subtype** (pdp / serp / cart /
article / …); topical relevance stays the separate axis it always was.

Design choices, all in service of robustness:

* **Feature hashing** (stable MD5-based signed hashing, default 2^16 dims) —
  no vocabulary file to persist or drift; the same string always hashes the
  same way on any machine or Python version.
* **Features**: URL structure (domain core, subdomain, TLD, path segments,
  path word-pieces, query keys), markup (schema.org types, og:type,
  price/add-to-cart flags, product-count bucket), text tokens (title, meta,
  h1, first text words), vendor prior and hosting-platform fingerprint.
  URL-only variants of every training page are included so the model stays
  useful for pages that could not be fetched.
* **Distillation of the service table**: the ``_PLATFORM_SERVICES`` routing
  rules are compiled into training samples, so the learned model knows that
  ``docs.google.com`` is a tool and ``shopping.google.com`` a marketplace
  even standalone.
* **Persistence** as a plain ``.npz`` (float32 coefficient matrix + class
  list) — training needs scikit-learn, **inference needs only numpy**.
* **Fail-open**: the classify channel calls :func:`predict_votes` inside a
  try/except and treats any failure as "no opinion".

CLI::

    python -m conveyer.scraping.model train [--pages scraped_pages.parquet]
    python -m conveyer.scraping.model eval  [--pages scraped_pages.parquet]

Training on a parquet uses **confident rows as weak labels** (self-training:
``page_category_confidence ≥ 0.75``, content actually fetched), so the model
absorbs the real URL/markup distribution without any manual labelling; rows
you correct by hand are picked up the same way.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import ScrapeConfig
from .extract import PageContent, extract_page

DEFAULT_DIM = 2 ** 16
# bump when featurize() changes: a model trained on old features would read
# arbitrary weights for the new ones — a mismatched file retrains itself
FEATURE_VERSION = 2
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TRAIN_LOCK = threading.Lock()
_FAILED_PATHS: set = set()


# --------------------------------------------------------------------------- #
# Featurization (stable across machines — no vocabulary to persist)
# --------------------------------------------------------------------------- #
def _hash_feature(feature: str, dim: int) -> Tuple[int, float]:
    h = int.from_bytes(hashlib.md5(feature.encode("utf-8")).digest()[:8], "little")
    return h % dim, (1.0 if (h >> 62) & 1 == 0 else -1.0)


def featurize(url: str, page: Optional[PageContent] = None,
              prior_page_type: str = "", platform: str = "") -> List[str]:
    """Feature strings for one page. ``page=None`` ⇒ URL-only features."""
    from .classify import parse_url  # late: classify lazily imports this module

    u = parse_url(url)
    f: List[str] = [f"core:{u.core}", f"sub:{u.subdomain.split('.')[-1] if u.subdomain else ''}",
                    f"tld:{u.registrable.rsplit('.', 1)[-1] if u.registrable else ''}"]
    segs = [s for s in u.path.strip("/").split("/") if s][:4]
    f += [f"seg{i}:{s[:40]}" for i, s in enumerate(segs)]
    f += [f"pt:{p}" for s in segs for p in _TOKEN_RE.findall(s.lower())[:8]]
    f += [f"qk:{k.split('=', 1)[0][:20]}" for k in u.query.split("&") if k][:6]
    if not segs:
        f.append("path:root")
    # slug SHAPE — the blog-permalink / comparison / question signatures that
    # identify informational articles from the URL alone
    if len(segs) == 1 and segs[0].count("-") >= 2:
        f.append("shape:root_slug")
    if re.search(r"[a-z0-9]-vs-[a-z0-9]", u.path):
        f.append("shape:vs")
    if re.search(r"(^|/)(what|why|how|when|which|should|does|do|is|can)-", u.path):
        f.append("shape:question")
    if prior_page_type:
        f.append(f"prior:{prior_page_type.lower()}")
    if platform:
        f.append(f"plat:{platform}")

    if page is not None and (page.title or page.text or page.og or page.jsonld):
        f += [f"st:{t}" for t in page.schema_types()[:10]]
        og_type = page.og.get("og:type", "").lower()
        if og_type:
            f.append(f"og:{og_type}")
        if page.has_price:
            f.append("sig:price")
        if page.has_add_to_cart:
            f.append("sig:cart")
        if page.lang:
            f.append(f"lang:{page.lang.split('-')[0].lower()}")
        text = " ".join([page.title, page.meta_description,
                         " ".join(page.h1[:2]), page.text[:1000]]).lower()
        toks = _TOKEN_RE.findall(text)[:150]
        f += [f"w:{t}" for t in toks]
        title_toks = _TOKEN_RE.findall(page.title.lower())[:20]
        f += [f"b:{a}_{b}" for a, b in zip(title_toks, title_toks[1:])]
    else:
        f.append("sig:url_only")
    return f


def _vectorize_one(features: Sequence[str], dim: int) -> Dict[int, float]:
    vec: Dict[int, float] = {}
    for feat in features:
        idx, sign = _hash_feature(feat, dim)
        vec[idx] = vec.get(idx, 0.0) + sign
    norm = np.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {i: v / norm for i, v in vec.items()}


# --------------------------------------------------------------------------- #
# The persisted model (numpy-only at inference)
# --------------------------------------------------------------------------- #
class PageModel:
    def __init__(self, coef: np.ndarray, intercept: np.ndarray,
                 classes: List[str], dim: int):
        self.coef = coef.astype(np.float32)          # (n_classes, dim)
        self.intercept = intercept.astype(np.float32)
        self.classes = list(classes)
        self.dim = int(dim)

    def predict_proba_one(self, features: Sequence[str]) -> Dict[str, float]:
        vec = _vectorize_one(features, self.dim)
        if not vec:
            return {}
        idx = np.fromiter(vec.keys(), dtype=np.int64)
        val = np.fromiter(vec.values(), dtype=np.float32)
        scores = self.coef[:, idx] @ val + self.intercept
        scores -= scores.max()
        exp = np.exp(scores)
        probs = exp / exp.sum()
        return {c: float(p) for c, p in zip(self.classes, probs)}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # atomic: a crash mid-write must not leave a truncated npz that would
        # silently disable the channel forever (load fails ⇒ cached None while
        # the file's existence blocks autotrain)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, coef=self.coef, intercept=self.intercept,
                                classes=np.array(self.classes), dim=np.int64(self.dim),
                                feature_version=np.int64(FEATURE_VERSION))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "PageModel":
        with np.load(path, allow_pickle=False) as z:
            version = int(z["feature_version"]) if "feature_version" in z else 1
            if version != FEATURE_VERSION:
                raise ValueError(f"model feature_version {version} != "
                                 f"{FEATURE_VERSION} (featurize changed — retrain)")
            return cls(z["coef"], z["intercept"],
                       [str(c) for c in z["classes"]], int(z["dim"]))


@lru_cache(maxsize=4)
def _load_model(path: str, mtime: float) -> Optional[PageModel]:
    try:
        return PageModel.load(path)
    except Exception:
        return None


def _model_for(cfg: ScrapeConfig) -> Optional[PageModel]:
    path = cfg.model_path
    if not path or path in _FAILED_PATHS:
        return None
    if not os.path.exists(path):
        if not cfg.model_autotrain:
            return None
        with _TRAIN_LOCK:
            if path in _FAILED_PATHS:
                return None
            if not os.path.exists(path):   # double-checked under the lock
                try:
                    print(f"[model] no model at {path} — training on the "
                          f"synthetic ground truth (once)…")
                    train(out_path=path)
                except Exception as exc:
                    print(f"[model] autotrain failed ({exc}); "
                          f"continuing with rule channels only")
                    _FAILED_PATHS.add(path)
                    return None
    try:
        model = _load_model(path, os.path.getmtime(path))
    except OSError:
        return None
    if model is None:
        # the file exists but does not load (torn write, wrong format): a
        # corrupt file must not silently kill the channel — retrain over it
        # once, or mark the path failed so we never loop on it
        with _TRAIN_LOCK:
            if path in _FAILED_PATHS:
                return None
            if not cfg.model_autotrain:
                _FAILED_PATHS.add(path)
                return None
            try:
                print(f"[model] unreadable model at {path} — retraining")
                train(out_path=path)
                model = _load_model(path, os.path.getmtime(path))
            except Exception as exc:
                print(f"[model] retrain failed ({exc}); disabling the channel")
                model = None
            if model is None:
                _FAILED_PATHS.add(path)
    return model


def predict_votes(page: Optional[PageContent], url: str,
                  cfg: ScrapeConfig, prior_page_type: str = "",
                  platform: str = "") -> Dict[str, float]:
    """Subtype votes for the classify channel: ``model_weight × probability``
    for the top predictions. '{}' when no model is available."""
    model = _model_for(cfg)
    if model is None:
        return {}
    probs = model.predict_proba_one(featurize(url, page, prior_page_type, platform))
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    return {c: round(cfg.model_weight * p, 3)
            for c, p in top if p >= 0.15 and c != "other"}


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
def _service_distillation() -> List[Tuple[List[str], str]]:
    """Compile the platform-service routing table into URL-only samples, so
    the learned model knows the services standalone."""
    from .classify import _PLATFORM_SERVICES

    out: List[Tuple[List[str], str]] = []
    for core, plat in _PLATFORM_SERVICES.items():
        for sub, votes in plat.get("subdomains", {}).items():
            label = max(votes, key=votes.get)
            for path in ("/", "/some/page"):
                out.append((featurize(f"https://{sub}.{core}.com{path}"), label))
        for seg, votes in plat.get("paths", {}).items():
            label = max(votes, key=votes.get)
            out.append((featurize(f"https://www.{core}.com/{seg}"), label))
            out.append((featurize(f"https://www.{core}.com/{seg}/deeper/page"), label))
        root = plat.get("root")
        if root:
            label = max(root, key=root.get)
            out.append((featurize(f"https://www.{core}.com/"), label))
    return out


def _editorial_slug_distillation() -> List[Tuple[List[str], str]]:
    """Comparison / question / explainer slugs on blogs the lists don't know
    — the class of URL the corpus surfaces constantly
    (zicail.com/eye-cream-vs-eye-serum/) and the curated channels miss."""
    slugs = [
        "eye-cream-vs-eye-serum", "retinol-vs-retinal-which-is-better",
        "day-cream-vs-night-cream", "cleanser-vs-face-wash",
        "what-is-niacinamide", "what-are-ceramides",
        "why-your-moisturizer-pills", "when-to-apply-sunscreen",
        "benefits-of-hyaluronic-acid", "difference-between-toner-and-essence",
        "how-often-should-you-exfoliate", "sunscreen-myths-debunked",
        "skincare-routine-mistakes", "double-cleansing-explained",
    ]
    domains = ["zicail.com", "glowjournal.example", "beautynotes.example"]
    out = [(featurize(f"https://www.{d}/{s}/"), "article")
           for d in domains for s in slugs]
    # counterweight: the SAME beauty vocabulary under /product(s)/ is a PDP —
    # the path segment decides, the tokens must not drag product pages
    # into 'article'
    pdp_slugs = ["gentle-eye-cream", "hydrating-eye-serum",
                 "daily-face-moisturizer", "vitamin-c-serum",
                 "night-repair-cream", "brightening-eye-cream"]
    out += [(featurize(f"https://www.{d}/products/{s}"), "pdp")
            for d in domains for s in pdp_slugs]
    out += [(featurize(f"https://www.{d}/product/{s}"), "pdp")
            for d in domains for s in pdp_slugs[:3]]
    return out


def _transactional_distillation() -> List[Tuple[List[str], str]]:
    domains = ["amazon.com", "sephora.com", "target.com", "boots.co.uk",
               "brandstore.example"]
    paths = [("/cart", "cart"), ("/basket", "cart"), ("/gp/cart/view.html", "cart"),
             ("/checkout", "checkout"), ("/checkouts/c/a1b2c3", "checkout"),
             ("/orders", "order"), ("/order-history", "order"),
             ("/wishlist", "wishlist"), ("/search?q=retinol", "serp"),
             ("/s?k=moisturizer", "serp")]
    return [(featurize(f"https://www.{d}{p}"), label)
            for d in domains for p, label in paths]


def build_training_samples(pages_parquet: Optional[str] = None,
                           n_synthetic: int = 120, seed: int = 42,
                           min_confidence: float = 0.75
                           ) -> Tuple[List[List[str]], List[str]]:
    """(features, labels) from the synthetic ground truth (+ URL-only
    variants), the rule-table distillations, and optionally the confident rows
    of an existing parquet (weak labels / self-training)."""
    from .synthetic import make_corpus

    X: List[List[str]] = []
    y: List[str] = []
    corpus = make_corpus(n_synthetic, seed)
    gt = corpus.ground_truth.set_index("url")
    for url, row in gt.iterrows():
        label = str(row["gt_subtype"])
        html = corpus.html_by_url.get(url)
        if html:
            X.append(featurize(url, extract_page(html, url)))
            y.append(label)
        X.append(featurize(url))
        y.append(label)
    for feats, label in (_service_distillation() + _transactional_distillation()
                         + _editorial_slug_distillation()):
        X.append(feats)
        y.append(label)

    if pages_parquet and os.path.exists(pages_parquet):
        import pandas as pd

        from .relabel import HUMAN_BOOST, is_human_labelled
        from .validate import _content_for_row
        pages = pd.read_parquet(pages_parquet)
        cfg = ScrapeConfig()
        n_weak = n_gold = 0
        for _, r in pages.iterrows():
            row = r.to_dict()
            sub = str(row.get("page_subtype") or "")
            conf = float(row.get("page_category_confidence") or 0.0)
            gold = is_human_labelled(row)
            if sub in ("", "other") or (conf < min_confidence and not gold):
                continue
            if str(row.get("page_category")) == "unknown":
                continue
            pc, scope, _src = _content_for_row(row, cfg)
            page = pc if scope in ("page", "stripped") else None
            feats = featurize(str(row.get("url") or ""), page,
                              str(row.get("prior_page_type") or ""),
                              str(row.get("server_platform") or ""))
            # a human correction is gold: repeat it so a handful of fixed rows
            # can out-vote the weak labels that taught the original mistake
            repeats = HUMAN_BOOST if gold else 1
            for _ in range(repeats):
                X.append(feats)
                y.append(sub)
            n_weak += 0 if gold else 1
            n_gold += 1 if gold else 0
        print(f"[model] +{n_weak} weak-labelled rows"
              + (f" +{n_gold} human-labelled rows (×{HUMAN_BOOST})" if n_gold else "")
              + f" from {pages_parquet}")
    return X, y


def _to_csr(X: List[List[str]], dim: int):
    from scipy.sparse import csr_matrix

    indptr, indices, data = [0], [], []
    for feats in X:
        vec = _vectorize_one(feats, dim)
        indices.extend(vec.keys())
        data.extend(vec.values())
        indptr.append(len(indices))
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       np.asarray(indices, dtype=np.int64),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(X), dim))


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train(pages_parquet: Optional[str] = None, out_path: Optional[str] = None,
          dim: int = DEFAULT_DIM, n_synthetic: int = 120, seed: int = 42
          ) -> PageModel:
    from sklearn.linear_model import LogisticRegression

    X_feats, y = build_training_samples(pages_parquet, n_synthetic, seed)
    Xs = _to_csr(X_feats, dim)
    clf = LogisticRegression(max_iter=2000, C=2.0)
    clf.fit(Xs, y)
    model = PageModel(np.asarray(clf.coef_), np.asarray(clf.intercept_),
                      [str(c) for c in clf.classes_], dim)
    path = out_path or ScrapeConfig().model_path
    model.save(path)
    acc = float(clf.score(Xs, y))
    print(f"[model] trained on {len(y)} samples, {len(model.classes)} subtypes "
          f"(train acc {acc:.3f}) -> {path}")
    return model


def evaluate_model(pages_parquet: Optional[str] = None, dim: int = DEFAULT_DIM,
                   n_synthetic: int = 120, seed: int = 42, folds: int = 5) -> dict:
    """K-fold cross-validated report over the training frame."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_predict

    X_feats, y = build_training_samples(pages_parquet, n_synthetic, seed)
    Xs = _to_csr(X_feats, dim)
    y_arr = np.asarray(y)
    clf = LogisticRegression(max_iter=2000, C=2.0)
    pred = cross_val_predict(clf, Xs, y_arr, cv=folds)
    print(classification_report(y_arr, pred, zero_division=0))
    acc = float((pred == y_arr).mean())
    print(f"[model] {folds}-fold accuracy: {acc:.3f} on {len(y_arr)} samples")
    return {"accuracy": acc, "n": len(y_arr)}


def _main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train / evaluate the learned page classifier.")
    ap.add_argument("command", choices=["train", "eval"])
    ap.add_argument("--pages", default=None,
                    help="scraped_pages.parquet for weak-label self-training")
    ap.add_argument("--out", default=None, help="model output path (.npz)")
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--synthetic-pages", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if a.command == "train":
        train(a.pages, a.out, a.dim, a.synthetic_pages, a.seed)
    else:
        evaluate_model(a.pages, a.dim, a.synthetic_pages, a.seed)


if __name__ == "__main__":
    _main()
