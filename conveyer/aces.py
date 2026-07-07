"""ACES (Agentic e-CommercE Simulator) integration.

ACES — https://github.com/mycustomai/ACES (arXiv:2508.02630) — pairs a VLM
shopping agent with a programmable mock marketplace to study how AI agents
choose products under controlled perturbations (price, rating, position,
sponsored/badge treatments).

This module bridges the conveyer conversation data and ACES in both directions:

* **conveyer → ACES**: derive a product catalog from the brands/categories the
  LLM actually recommends in the conversation logs, generate perturbed
  experiment grids in ACES's *local-dataset CSV schema* (same columns and the
  same randomisation protocol as ACES `experiments/data_generation.py`), and
  emit the `uv run run.py --local-dataset ...` command to execute them against
  real VLM agents.
* **Prompt dynamics**: a library of *subtle prompt perturbations* (politeness,
  urgency, budget cues, brand loyalty, typos, ...) used (a) as the treatment
  arms of the ACES experiments and (b) by `conveyer.augment` to expand the
  conversation database.
* **Offline fallback**: `AgentChoiceModel`, a multinomial-logit agent with
  known ground-truth utility weights (position bias, price/rating sensitivity,
  badge effects, prompt-conditioned modifiers). It lets the full experiment →
  choice → analysis chain run without API keys, and — because the true
  parameters are known — validates the ranking/attribution estimators in
  `conveyer.attribution` (they should recover these weights).
"""

import shlex
import string
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1 · Subtle prompt perturbations
# --------------------------------------------------------------------------- #
# Each perturbation is a *minimal* rewrite of the base shopping query.
# `utility_modifiers` describe how a rational-ish agent could plausibly shift
# its weights in response — these drive the offline simulator, and are the
# hypotheses that a real ACES run tests.
PERTURBATIONS: Dict[str, dict] = {
    "baseline":    {"template": "{q}", "utility_modifiers": {}},
    "polite":      {"template": "Hi! Could you please help me find {q}? Thank you so much!",
                    "utility_modifiers": {}},
    "urgent":      {"template": "I need {q} TODAY, pick one fast.",
                    "utility_modifiers": {"position": -0.5, "low_stock": 0.4}},
    "budget":      {"template": "{q} — I'm on a tight budget, nothing expensive please.",
                    "utility_modifiers": {"log_price": -1.2}},
    "premium":     {"template": "{q} — money is no object, I want the best quality.",
                    "utility_modifiers": {"log_price": 0.8, "rating": 0.5}},
    "brand_loyal": {"template": "{q}. I usually buy {brand}, so I'd prefer that brand.",
                    "utility_modifiers": {"brand_match": 2.5}},
    "skeptical":   {"template": "{q}. Ignore sponsored results and ads, I only trust organic listings.",
                    "utility_modifiers": {"sponsored": -1.5}},
    "social_proof": {"template": "{q} — whatever most people buy and rate highly.",
                     "utility_modifiers": {"rating": 0.8, "log_rating_count": 0.6}},
    "typo":        {"template": "{q_typo}", "utility_modifiers": {}},
}


def _inject_typo(text: str, rng: np.random.Generator) -> str:
    """Swap two adjacent characters inside one random word (len >= 4)."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w.strip(string.punctuation)) >= 4]
    if not idx:
        return text
    i = int(rng.choice(idx))
    w = words[i]
    j = int(rng.integers(0, len(w) - 1))
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def perturb_prompt(query: str, kind: str, brand: str = "",
                   rng: Optional[np.random.Generator] = None) -> str:
    """Return the perturbed version of `query` for one perturbation kind."""
    rng = rng or np.random.default_rng(0)
    spec = PERTURBATIONS[kind]
    return spec["template"].format(q=query, brand=brand,
                                   q_typo=_inject_typo(query, rng))


def expand_prompts(query: str, kinds: Optional[Iterable[str]] = None,
                   brand: str = "", seed: int = 0) -> pd.DataFrame:
    """All perturbed variants of one query as a tidy frame."""
    rng = np.random.default_rng(seed)
    kinds = list(kinds or PERTURBATIONS)
    return pd.DataFrame({
        "perturbation": kinds,
        "prompt": [perturb_prompt(query, k, brand, rng) for k in kinds],
    })


# --------------------------------------------------------------------------- #
# 2 · Product catalog derived from the conversation data
# --------------------------------------------------------------------------- #
_PRODUCT_NOUNS = {
    "acne": "acne treatment gel", "anti-aging": "retinol night cream",
    "sunscreen": "SPF 50 facial sunscreen", "hyperpigmentation": "vitamin C serum",
    "hydration": "hyaluronic acid moisturizer", "sensitive": "soothing repair cream",
}


def product_catalog_from_conversations(df: pd.DataFrame, n_per_query: int = 8,
                                       category_col: str = "skin_care_categories",
                                       seed: int = 42) -> pd.DataFrame:
    """Build an ACES-style SKU catalog from the brands the LLM recommends.

    One shopping `query` per skincare category; the candidate products are the
    brands observed in `_brands_a` for that category (most-mentioned first).
    Price/rating/rating_count are synthesised deterministically — with real
    retail data, join them in here instead.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cat, sub in df.groupby(df[category_col].astype(str)):
        counts: Dict[str, int] = {}
        for lst in sub.get("_brands_a", sub.get("brands_in_answer", pd.Series(dtype=object))):
            for b in (lst if isinstance(lst, (list, set)) else []):
                counts[str(b)] = counts.get(str(b), 0) + 1
        brands = [b for b, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:n_per_query]
        noun = _PRODUCT_NOUNS.get(cat, f"{cat} skincare product")
        for b in brands:
            rows.append({
                "query": noun,
                "category": cat,
                "brand": b,
                "title": f"{b} {noun.title()}",
                "price": float(np.round(rng.uniform(8, 60), 2)),
                "rating": float(np.round(rng.uniform(3.5, 4.9), 1)),
                "rating_count": int(rng.integers(50, 20000)),
                "mention_count": counts[b],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3 · ACES experiment generation (same protocol as ACES data_generation.py)
# --------------------------------------------------------------------------- #
ACES_COLUMNS = [
    "query", "title", "brand", "price", "rating", "rating_count",
    "original_price", "original_rating", "original_rating_count",
    "sponsored", "overall_pick", "low_stock", "best_seller", "limited_time",
    "discounted", "stock_quantity", "experiment_label", "experiment_number",
    "position_in_experiment", "assigned_position", "desired_choice",
]


def _perturb_group(g: pd.DataFrame, experiment_id: int, label: str,
                   rng: np.random.Generator, permutation_only: bool) -> pd.DataFrame:
    """One randomised marketplace instance for one query — ACES's protocol:
    row permutation; log-normal price noise; rating shift r + α(5−r),
    α~U(−.4,.4); log-normal rating_count noise; Bernoulli(.5) sponsored /
    overall-pick / low-stock badges."""
    from scipy.stats import lognorm

    g = g.sample(frac=1, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
    g["original_price"], g["original_rating"], g["original_rating_count"] = \
        g["price"], g["rating"], g["rating_count"]
    for c in ("sponsored", "overall_pick", "low_stock", "best_seller",
              "limited_time", "discounted"):
        g[c] = False
    g["stock_quantity"] = 100

    if not permutation_only and experiment_id != 0:
        g["price"] = np.round(g["price"] * lognorm.rvs(s=0.3, scale=1, size=len(g),
                                                       random_state=rng.integers(0, 2**31)), 2)
        alpha = rng.uniform(-0.4, 0.4, len(g))
        g["rating"] = np.clip(np.round(g["rating"] + alpha * (5 - g["rating"]), 1), 0, 5)
        q = lognorm.rvs(s=1, scale=1, size=len(g), random_state=rng.integers(0, 2**31))
        g["rating_count"] = (g["rating_count"] * q).astype(int)

    if not permutation_only:
        if rng.random() < 0.5:
            k = int(rng.choice([1, 2, 3, 4]))
            g.loc[rng.choice(g.index, min(k, len(g)), replace=False), "sponsored"] = True
        if rng.random() < 0.5:
            elig = g.index[~g["sponsored"]]
            if len(elig):
                g.loc[rng.choice(elig), "overall_pick"] = True
        if rng.random() < 0.5:
            elig = g.index[~(g["sponsored"] | g["overall_pick"])]
            if len(elig):
                i = rng.choice(elig)
                g.loc[i, "low_stock"] = True
                g.loc[i, "stock_quantity"] = int(rng.integers(1, 5))

    g["experiment_label"] = label
    g["experiment_number"] = experiment_id
    g["position_in_experiment"] = range(len(g))
    g["assigned_position"] = range(len(g))
    g["desired_choice"] = -1
    return g


def build_aces_experiments(catalog: pd.DataFrame, n_experiments: int = 50,
                           label: str = "conveyer_v1", permutation_only: bool = False,
                           seed: int = 0) -> pd.DataFrame:
    """Replicated marketplace instances per query, ACES local-dataset schema."""
    out = []
    for i in range(n_experiments):
        rng = np.random.default_rng(seed + i)
        for q, g in catalog.groupby("query"):
            out.append(_perturb_group(g.copy(), i, label, rng, permutation_only))
    df = pd.concat(out, ignore_index=True)
    return df[[c for c in ACES_COLUMNS if c in df.columns]
              + [c for c in df.columns if c not in ACES_COLUMNS]]


def to_aces_csv(experiments: pd.DataFrame, path: str) -> str:
    experiments.to_csv(path, index=False)
    return path


def aces_run_command(dataset_csv: str, runtime: str = "screenshot",
                     include_models: Sequence[str] = ()) -> str:
    """The command to execute the exported grid against real VLM agents
    (requires a clone of https://github.com/mycustomai/ACES + `uv sync` +
    provider API keys in `.env`)."""
    cmd = ["uv", "run", "run.py", "--local-dataset", dataset_csv,
           "--runtime-type", runtime]
    for m in include_models:
        cmd += ["--include", m]
    return shlex.join(cmd)


# --------------------------------------------------------------------------- #
# 4 · Offline agent: multinomial-logit choice model with known ground truth
# --------------------------------------------------------------------------- #
FEATURES = ["position", "log_price", "rating", "log_rating_count",
            "sponsored", "overall_pick", "low_stock", "brand_match"]


def design_matrix(g: pd.DataFrame, preferred_brand: str = "") -> np.ndarray:
    """Feature matrix (one row per product in a choice set) for the utility model."""
    pb = str(preferred_brand).strip().lower()
    return np.column_stack([
        g["assigned_position"].to_numpy(float),
        np.log(np.maximum(g["price"].to_numpy(float), 0.01)),
        g["rating"].to_numpy(float),
        np.log1p(g["rating_count"].to_numpy(float)),
        g["sponsored"].to_numpy(float),
        g["overall_pick"].to_numpy(float),
        g["low_stock"].to_numpy(float),
        (g["brand"].astype(str).str.strip().str.lower() == pb).to_numpy(float) if pb
        else np.zeros(len(g)),
    ])


@dataclass
class AgentChoiceModel:
    """Softmax choice over product utilities, in the spirit of the biases the
    ACES paper measures (strong position bias, price sensitivity, badge
    effects). `PERTURBATIONS[kind]["utility_modifiers"]` shift the weights,
    so subtle prompt changes translate into measurable choice-share moves."""

    beta: Dict[str, float] = field(default_factory=lambda: {
        "position": -0.35,        # earlier slots win (position bias)
        "log_price": -0.9,        # cheaper wins
        "rating": 1.1,            # higher rating wins
        "log_rating_count": 0.35, # social proof
        "sponsored": -0.25,       # mild ad aversion
        "overall_pick": 0.9,      # platform endorsement helps
        "low_stock": 0.15,        # mild scarcity nudge
        "brand_match": 0.0,       # only active under brand-loyal prompts
    })
    temperature: float = 1.0

    def weights(self, perturbation: str = "baseline") -> np.ndarray:
        mods = PERTURBATIONS.get(perturbation, {}).get("utility_modifiers", {})
        return np.array([self.beta[f] + mods.get(f, 0.0) for f in FEATURES])

    def choice_probs(self, g: pd.DataFrame, perturbation: str = "baseline",
                     preferred_brand: str = "") -> np.ndarray:
        u = design_matrix(g, preferred_brand) @ self.weights(perturbation)
        u = (u - u.max()) / self.temperature
        p = np.exp(u)
        return p / p.sum()

    def choose(self, g: pd.DataFrame, perturbation: str = "baseline",
               preferred_brand: str = "", rng: Optional[np.random.Generator] = None) -> int:
        p = self.choice_probs(g, perturbation, preferred_brand)
        rng = rng or np.random.default_rng(0)
        return int(rng.choice(len(g), p=p))


def simulate(experiments: pd.DataFrame, perturbations: Sequence[str] = ("baseline",),
             model: Optional[AgentChoiceModel] = None, preferred_brand: str = "",
             seed: int = 0) -> pd.DataFrame:
    """Run the offline agent over every (query, experiment, perturbation) cell.

    Returns one row per *purchase decision*, mirroring what ACES's
    `experiment_logs/` aggregation gives you for real VLM agents — so all the
    downstream analysis (choice shares, utility-model recovery, Shapley) is
    identical whether the choices come from this simulator or from real runs.
    """
    model = model or AgentChoiceModel()
    rng = np.random.default_rng(seed)
    rows = []
    for (q, xnum), g in experiments.groupby(["query", "experiment_number"]):
        g = g.sort_values("assigned_position").reset_index(drop=True)
        for kind in perturbations:
            pb = preferred_brand or (g["brand"].iloc[0] if "brand" in g else "")
            probs = model.choice_probs(g, kind, pb if kind == "brand_loyal" else "")
            j = int(rng.choice(len(g), p=probs))
            r = g.iloc[j]
            rows.append({
                "query": q, "experiment_number": xnum, "perturbation": kind,
                "prompt": perturb_prompt(str(q), kind, pb, rng),
                "chosen_brand": r.get("brand", ""), "chosen_title": r.get("title", ""),
                "chosen_position": int(r["assigned_position"]),
                "chosen_price": float(r["price"]), "chosen_rating": float(r["rating"]),
                "chosen_sponsored": bool(r["sponsored"]),
                "choice_prob": float(probs[j]),
                "n_options": len(g),
            })
    return pd.DataFrame(rows)


def choice_shares(results: pd.DataFrame, by: str = "perturbation") -> pd.DataFrame:
    """Market share of each brand within each perturbation arm (or other cell)."""
    ct = (results.groupby([by, "chosen_brand"]).size().rename("choices").reset_index())
    ct["share"] = ct["choices"] / ct.groupby(by)["choices"].transform("sum")
    return ct.sort_values([by, "share"], ascending=[True, False]).reset_index(drop=True)
