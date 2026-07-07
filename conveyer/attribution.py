"""Estimation & attribution toolkit for the research questions.

Numpy/scipy implementations (no PyMC dependency) of the models each question
needs — every function is small enough to port 1:1 into PyMC/NumPyro when
moving from point estimates with analytic posteriors to full hierarchical
inference:

* Q3  `beta_binomial_posterior`  — conversion rate from sparse click/purchase
      counts (conjugate Bayesian; the honest object under heavy censoring).
* Q4  `shrink_segments`, `poststratify` — empirical-Bayes partial pooling +
      MRP-style extrapolation to unseen segments (mobile, other LLMs).
* Q5  `geometric_adstock`, `hill_saturation`, `fit_mmm` — marketing-mix-model
      building blocks to attribute online/offline sales to LLM-recommendation
      exposure.
* Q6  `fit_mnl` — conditional multinomial logit on choice sets: the utility
      model behind LLM/agent product rankings (recovers the ACES simulator's
      ground-truth weights).
* Q8  `exact_shapley` — exact Shapley decomposition of a product's choice
      probability over its utility features → intervention ranking.
"""

from itertools import permutations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Q3 · Bayesian conversion estimation
# --------------------------------------------------------------------------- #
def beta_binomial_posterior(successes: int, trials: int, a: float = 1.0,
                            b: float = 1.0, ci: float = 0.94) -> Dict[str, float]:
    """Posterior of a conversion rate under Beta(a,b) prior — returns mean and
    equal-tailed credible interval. With trials this small/censored, the CI is
    the deliverable, not the point estimate."""
    post = stats.beta(a + successes, b + trials - successes)
    lo, hi = post.ppf((1 - ci) / 2), post.ppf(1 - (1 - ci) / 2)
    return {"mean": float(post.mean()), "lo": float(lo), "hi": float(hi),
            "a_post": a + successes, "b_post": b + trials - successes}


def conversion_table(counts: pd.DataFrame, success_col: str = "purchases",
                     trial_col: str = "clicks", a: float = 1.0, b: float = 1.0) -> pd.DataFrame:
    """Beta-binomial posterior per row (e.g. per brand / per funnel stage)."""
    rows = [beta_binomial_posterior(int(r[success_col]), int(r[trial_col]), a, b)
            for _, r in counts.iterrows()]
    out = counts.copy().reset_index(drop=True)
    post = pd.DataFrame(rows)
    out[["cvr_mean", "cvr_lo", "cvr_hi"]] = post[["mean", "lo", "hi"]].round(4)
    return out


# --------------------------------------------------------------------------- #
# Q4 · Unseen segments: partial pooling + poststratification (MRP-lite)
# --------------------------------------------------------------------------- #
def shrink_segments(cells: pd.DataFrame, success_col: str, trial_col: str) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of per-cell rates toward the pooled rate.

    Method-of-moments Beta prior fitted on the observed cells; each cell's
    posterior mean is then `(a + s) / (a + b + n)` — cells with little data
    collapse to the global mean, big cells keep their own signal. This is the
    2-line version of a hierarchical Bayesian model (partial pooling)."""
    s, n = cells[success_col].to_numpy(float), cells[trial_col].to_numpy(float)
    p = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    mu = s.sum() / max(n.sum(), 1)
    var = max(p.var(), 1e-6)
    common = mu * (1 - mu) / var - 1
    a, b = max(mu * common, 0.5), max((1 - mu) * common, 0.5)
    out = cells.copy()
    out["rate_raw"] = np.round(p, 4)
    out["rate_shrunk"] = np.round((a + s) / (a + b + n), 4)
    out.attrs["prior"] = (float(a), float(b))
    return out


def poststratify(cell_rates: pd.DataFrame, population: pd.DataFrame,
                 on: List[str], rate_col: str = "rate_shrunk",
                 weight_col: str = "pop_share") -> Dict[str, float]:
    """MRP step 2: reweight modelled cell rates by the *target* population.

    `population` carries the cell shares of the segment we did NOT observe
    (mobile users, another LLM's user base — e.g. from SimilarWeb panel or
    survey data). The estimate for the unseen segment is the population-
    weighted average of the modelled cell rates — the standard survey answer
    to 'our sample is desktop-Chrome-US only'."""
    m = population.merge(cell_rates[on + [rate_col]], on=on, how="left")
    m[rate_col] = m[rate_col].fillna(cell_rates[rate_col].mean())
    w = m[weight_col] / m[weight_col].sum()
    return {"estimate": float((m[rate_col] * w).sum()),
            "coverage": float(m[rate_col].notna().mean())}


# --------------------------------------------------------------------------- #
# Q5 · MMM building blocks: adstock + saturation + regression
# --------------------------------------------------------------------------- #
def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Carry-over: today's effective exposure = x_t + decay * effective_{t-1}."""
    out = np.zeros_like(np.asarray(x, dtype=float))
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + decay * carry
        out[t] = carry
    return out


def hill_saturation(x: np.ndarray, half_sat: float, shape: float = 1.0) -> np.ndarray:
    """Diminishing returns: response saturates as exposure grows."""
    x = np.asarray(x, dtype=float)
    return x**shape / (x**shape + half_sat**shape)


def fit_mmm(sales: np.ndarray, channels: Dict[str, np.ndarray],
            decays: Sequence[float] = (0.0, 0.3, 0.5, 0.7, 0.9)) -> Dict[str, object]:
    """Tiny MMM: grid-search each channel's adstock decay, saturate at the
    channel median, then OLS on the transformed regressors.

    Returns coefficients, per-channel sales contributions and R². This is the
    skeleton to replace with a Bayesian MMM (PyMC-Marketing / LightweightMMM)
    once priors and calibration experiments are on the table."""
    from itertools import product as iproduct

    from sklearn.linear_model import LinearRegression

    names = list(channels)
    best = None
    for combo in iproduct(decays, repeat=len(names)):
        X = np.column_stack([
            hill_saturation(geometric_adstock(channels[c], d),
                            half_sat=max(np.median(channels[c]), 1e-9))
            for c, d in zip(names, combo)])
        reg = LinearRegression().fit(X, sales)
        r2 = reg.score(X, sales)
        if best is None or r2 > best["r2"]:
            best = {"r2": r2, "decays": dict(zip(names, combo)), "model": reg, "X": X}
    reg, X = best["model"], best["X"]
    contrib = {c: float(reg.coef_[i] * X[:, i].sum()) for i, c in enumerate(names)}
    total = sum(abs(v) for v in contrib.values()) or 1.0
    return {"r2": float(best["r2"]), "decays": best["decays"],
            "intercept": float(reg.intercept_),
            "coef": dict(zip(names, np.round(reg.coef_, 3))),
            "contribution": {k: round(v, 1) for k, v in contrib.items()},
            "contribution_share": {k: round(abs(v) / total, 3) for k, v in contrib.items()}}


# --------------------------------------------------------------------------- #
# Q6 · Conditional multinomial logit on choice sets (utility model)
# --------------------------------------------------------------------------- #
def fit_mnl(choice_sets: List[np.ndarray], chosen: Sequence[int],
            lr: float = 0.05, n_iter: int = 3000, l2: float = 1e-4,
            seed: int = 0) -> Dict[str, object]:
    """Maximum-likelihood conditional logit by full-batch gradient ascent.

    `choice_sets[i]` is the (n_options x n_features) design matrix of one
    displayed slate; `chosen[i]` the index picked. Returns coefficients and
    (inverse-Hessian) standard errors. Given ACES simulator output this
    recovers the agent's ground-truth utility weights — the estimator for
    'why does the LLM rank products in this order?'."""
    rng = np.random.default_rng(seed)
    k = choice_sets[0].shape[1]
    beta = rng.normal(0, 0.01, k)
    n = len(choice_sets)
    for _ in range(n_iter):
        grad = -l2 * beta
        for X, y in zip(choice_sets, chosen):
            u = X @ beta
            p = np.exp(u - u.max()); p /= p.sum()
            grad += X[y] - p @ X
        beta += lr * grad / n
    # observed information for standard errors
    H = np.eye(k) * l2
    ll = 0.0
    for X, y in zip(choice_sets, chosen):
        u = X @ beta
        p = np.exp(u - u.max()); p /= p.sum()
        ll += np.log(p[y] + 1e-300)
        xb = p @ X
        H += X.T @ (X * p[:, None]) - np.outer(xb, xb)
    se = np.sqrt(np.diag(np.linalg.pinv(H)))
    return {"beta": beta, "se": se, "loglik": float(ll), "n_obs": n,
            "z": beta / np.maximum(se, 1e-12)}


def mnl_table(fit: Dict[str, object], feature_names: Sequence[str],
              true_beta: Optional[Sequence[float]] = None) -> pd.DataFrame:
    t = pd.DataFrame({"feature": list(feature_names),
                      "beta_hat": np.round(fit["beta"], 3),
                      "se": np.round(fit["se"], 3),
                      "z": np.round(fit["z"], 1)})
    if true_beta is not None:
        t.insert(1, "beta_true", np.round(np.asarray(true_beta, dtype=float), 3))
    return t


# --------------------------------------------------------------------------- #
# Q8 · Exact Shapley values over utility features
# --------------------------------------------------------------------------- #
def exact_shapley(value_fn: Callable[[np.ndarray], float], x: np.ndarray,
                  baseline: np.ndarray, max_features: int = 10) -> np.ndarray:
    """Exact Shapley attribution of `value_fn(x) - value_fn(baseline)`.

    Coalitions are formed by switching features from `baseline` to `x`
    (interventional Shapley). Exact enumeration — fine for the <=8 utility
    features here; swap for sampling/KernelSHAP beyond ~10."""
    k = len(x)
    if k > max_features:
        raise ValueError(f"exact enumeration is O(k!): {k} features > {max_features}")
    phi = np.zeros(k)
    perms = list(permutations(range(k)))
    for order in perms:
        z = baseline.copy()
        prev = value_fn(z)
        for f in order:
            z[f] = x[f]
            cur = value_fn(z)
            phi[f] += cur - prev
            prev = cur
    return phi / len(perms)


def shapley_report(phi: np.ndarray, feature_names: Sequence[str],
                   x: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    return (pd.DataFrame({"feature": list(feature_names),
                          "value": np.round(x, 3),
                          "baseline": np.round(baseline, 3),
                          "shapley": np.round(phi, 4)})
            .sort_values("shapley", key=np.abs, ascending=False)
            .reset_index(drop=True))
