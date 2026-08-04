"""The sales bridge — from projected AI journeys to euros of influenced sales.

This is the module the whole project was missing: everything upstream measures
*behaviour in a desktop panel*, while the question is *what share of sales the
AI channel influences*. The two are separated by a chain of factors nobody can
observe in a clickstream, so the bridge does three things and refuses to blur
them:

**1 · It computes the same number two independent ways.**

*Bottom-up* — count the journeys, convert them, value them::

    exposed        = projected AI sessions exposed to the brand
                     ÷ device_coverage × panel_representativeness × window_recovery
    online_orders  = exposed × P(reach cart | exposed) × cart_to_order
                     × cross_device_completion
    purchases      = online_orders × ropo_multiplier          # online research, any till
    influenced_€   = purchases × basket_value × brand_match_precision
    share          = influenced_€ ÷ actual sales (RMS / Omnisales)

*Top-down* — start from the market and work in::

    share = reach(AI conversations among category purchase occasions)
            × P(brand exposed | conversation) × influence_given_exposure

The two routes share almost no assumptions. Agreement is evidence; a
``reconciliation_ratio`` far from 1 is a finding, reported rather than
smoothed away.

**2 · It separates *influenced* from *incremental*.** "Influenced" is a
touchpoint claim — the conversation was in the journey. "Incremental" is a
causal claim — the sale would not have happened otherwise. They differ by a
factor of several, and only the first is measurable from observational panel
data. ``incremental_value`` is always ``influenced_value × incrementality``,
and ``incrementality`` is a declared factor whose proper source is the MMM
(:mod:`conveyer.attribution.mmm`) or an experiment — never this module.

**3 · It says which unknown is costing the most.** Every factor is drawn from
its declared range (:mod:`conveyer.attribution.calibration`), so the output is
a distribution, and :func:`sensitivity` re-runs the chain with **one** factor
varying at a time. The resulting table ranks the assumptions by how much of
the final interval they own — i.e. which data to go buy first. On the current
default ledger the panel's own sampling error is nowhere near the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from ..schema_utils import write_parquet
from .aggregate import DIMENSIONS
from .calibration import Ledger, default_ledger

#: the factors that multiply into the bottom-up chain, in chain order
CHAIN_FACTORS = ("device_coverage", "panel_representativeness", "window_recovery",
                 "cart_to_order", "cross_device_completion", "ropo_multiplier",
                 "basket_value", "brand_match_precision")

KEY = list(DIMENSIONS) + ["brand"]


@dataclass
class BridgeConfig:
    n_draws: int = 2000
    seed: int = 7
    #: pseudo-observations of the pooled rate mixed into each cell's
    #: conversion rate — thin brand-weeks borrow strength from the category
    #: instead of producing 0% or 100% from three sessions
    shrinkage_prior: float = 20.0
    currency: str = "EUR"
    out_dir: str = "outputs/attribution"

    def path(self, name: str) -> str:
        import os
        return os.path.join(self.out_dir, name)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _require(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} is missing {missing}. "
            f"(Exposure counts must be projected to the universe first — pass "
            f"`frames=` to build_weekly; sales must carry brand-week value.)")


_COUNT_COLS = ("n_sessions_category", "n_sessions_mentioned", "n_sessions_unsolicited",
               "n_sessions_cited", "n_sessions_visited", "n_sessions_followed",
               "n_sessions_converted", "n_sessions", "n_users")


def join_key(sales: pd.DataFrame) -> List[str]:
    """The grain the two sides can actually meet on.

    Sales data has no ``llm_platform`` — no retailer records which assistant
    the shopper talked to. So unless the sales table carries a dimension, the
    exposure side is collapsed over it. Keeping the platform split alive
    through a join that cannot support it is how double-counted euros appear.
    """
    key = [d for d in DIMENSIONS if d in sales.columns] + ["brand"]
    missing = [c for c in ("week", "brand") if c not in key]
    if missing:
        raise ValueError(f"sales table must carry at least {missing}")
    return key


def collapse_exposure(exposure: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """Sum exposure counts to ``key``, re-deriving the rates from the sums.

    Rates are recomputed, never averaged: a share of voice averaged over
    platforms with wildly different panel volumes is not a share of anything.
    Dimensions dropped by the collapse are stamped ``"all"`` so the output
    still says what it covers.
    """
    key = list(key)
    if set(key) >= set(KEY):
        return exposure
    counts = [c for c in exposure.columns
              if c in _COUNT_COLS or c.endswith("_projected")
              or c.endswith("_projected_lo") or c.endswith("_projected_hi")]
    out = exposure.groupby(key, as_index=False)[counts].sum()
    for d in DIMENSIONS:
        if d not in key:
            out[d] = "all"
    if {"n_sessions_mentioned", "n_sessions_category"} <= set(out.columns):
        out["exposure_rate"] = (out["n_sessions_mentioned"]
                                / out["n_sessions_category"].replace(0, np.nan)
                                ).fillna(0.0).round(4)
        cell = [c for c in key if c != "brand"]
        totals = out.groupby(cell)["n_sessions_mentioned"].transform("sum")
        out["ai_share_of_voice"] = (out["n_sessions_mentioned"]
                                    / totals.replace(0, np.nan)).fillna(0.0).round(4)
    return out


def _joined(exposure: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Inner-join projected exposure to actual sales on the business key."""
    _require(exposure, KEY + ["n_sessions_mentioned", "n_sessions_converted",
                              "n_sessions_mentioned_projected", "exposure_rate"],
             "exposure table")
    key = join_key(sales)
    _require(sales, key + ["sales_value"], "sales table")
    j = collapse_exposure(exposure, key).merge(sales, on=key, how="inner",
                                               suffixes=("", "_sales"))
    if "category_sales_value" not in j.columns and not j.empty:
        # the top-down route needs a *category* denominator. Summing the
        # tracked brands understates it (untracked brands and private label
        # are missing), which biases the cross-check's reach upward — a real
        # RMS category total replaces this line.
        cell = [c for c in key if c != "brand"]
        j["category_sales_value"] = j.groupby(cell)["sales_value"].transform("sum")
    if j.empty:
        raise ValueError(
            "exposure and sales share no brand-weeks. The usual cause is a "
            "brand-naming mismatch: the panel speaks the chat lexicon "
            "(conveyer.brands), sales speaks the RMS/Omnisales hierarchy. "
            "They need an explicit mapping table, not a string join.")
    return j


# --------------------------------------------------------------------------- #
# The Monte Carlo chain
# --------------------------------------------------------------------------- #
def _conversion_draws(j: pd.DataFrame, cfg: BridgeConfig,
                      rng: np.random.Generator) -> np.ndarray:
    """(cells × draws) posterior draws of P(reach cart | exposed).

    Beta posterior per cell, shrunk toward the pooled rate. This is the only
    genuinely *measured* link in the chain, and giving it a distribution (not
    a point) is what lets the tornado compare panel sampling error against the
    assumed factors on the same axis.
    """
    n = j["n_sessions_mentioned"].astype(float).to_numpy()
    k = np.minimum(j["n_sessions_converted"].astype(float).to_numpy(), n)
    pooled = float(k.sum() / n.sum()) if n.sum() > 0 else 0.0
    m = cfg.shrinkage_prior
    a = k + m * pooled + 0.5
    b = (n - k) + m * (1 - pooled) + 0.5
    return rng.beta(a[:, None], b[:, None], size=(len(j), cfg.n_draws))


def _chain(j: pd.DataFrame, draws: pd.DataFrame, p_conv: np.ndarray) -> np.ndarray:
    """(cells × draws) influenced sales value, bottom-up."""
    def f(name: str) -> np.ndarray:
        return draws[name].to_numpy()[None, :]

    exposed = (j["n_sessions_mentioned_projected"].astype(float).to_numpy()[:, None]
               / np.maximum(f("device_coverage"), 1e-9)
               * f("panel_representativeness") * f("window_recovery"))
    online_orders = exposed * p_conv * f("cart_to_order") * f("cross_device_completion")
    purchases = online_orders * f("ropo_multiplier")
    return purchases * f("basket_value") * f("brand_match_precision")


def _topdown(j: pd.DataFrame, draws: pd.DataFrame) -> np.ndarray:
    """(cells × draws) influenced *share*, computed from the market side.

    Needs **category-level** projected reach (``n_users_projected``, joined in
    from ``fact_ai_category_weekly``): the route asks what share of category
    purchase occasions were preceded by an AI conversation at all, then how
    many of those saw this brand. Without it the answer is all-NaN — the
    brand-level count is not a substitute, and silently reusing it would
    multiply the brand exposure in twice.

    ``category_occasions`` is the number of purchase occasions the cell's
    sales represent. Prefer a CPS buyer/occasion count when it is joined in;
    otherwise it falls out of the same ``basket_value`` used bottom-up — the
    one place the two routes touch, and it is declared.
    """
    def f(name: str) -> np.ndarray:
        return draws[name].to_numpy()[None, :]

    if "n_users_projected" not in j.columns:
        return np.full((len(j), len(draws)), np.nan)
    users = j["n_users_projected"].astype(float).to_numpy()[:, None]
    users = users / np.maximum(f("device_coverage"), 1e-9) * f("panel_representativeness")

    if "category_occasions" in j.columns:
        occasions = j["category_occasions"].astype(float).to_numpy()[:, None] \
            * np.ones((1, draws.shape[0]))
    else:
        cat_value = (j["category_sales_value"] if "category_sales_value" in j.columns
                     else j["sales_value"]).astype(float).to_numpy()[:, None]
        occasions = cat_value / np.maximum(f("basket_value"), 1e-9)

    reach = np.divide(users, np.maximum(occasions, 1e-9))
    exposure_rate = j["exposure_rate"].astype(float).to_numpy()[:, None]
    return np.clip(reach, 0, 1) * exposure_rate * f("influence_given_exposure")


def _quantiles(mat: np.ndarray, qs=(0.05, 0.5, 0.95)) -> Dict[str, np.ndarray]:
    out = np.quantile(mat, qs, axis=1)
    return {f"p{int(q * 100):02d}": out[i] for i, q in enumerate(qs)}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def estimate_influenced_sales(exposure: pd.DataFrame, sales: pd.DataFrame,
                              ledger: Optional[Ledger] = None,
                              cfg: Optional[BridgeConfig] = None,
                              category_weekly: Optional[pd.DataFrame] = None,
                              ) -> Dict[str, object]:
    """Bottom-up + top-down AI-influenced sales per brand-week, with intervals.

    ``exposure`` is ``fact_ai_exposure_weekly`` **with projected counts**,
    ``sales`` is the brand-week value from RMS/Omnisales, and the optional
    ``category_weekly`` (``fact_ai_category_weekly``) enables the top-down
    cross-check. Returns ``{"cells", "totals", "sensitivity", "factors",
    "ledger", "headline"}``; ``cells`` is one row per brand-week with
    p05/p50/p95 of the influenced value, its share of that brand-week's actual
    sales, the incremental value, and the reconciliation ratio between routes.
    """
    ledger = ledger or default_ledger()
    cfg = cfg or BridgeConfig()
    j = _joined(exposure, sales).reset_index(drop=True)
    if category_weekly is not None and not category_weekly.empty:
        keep = [c for c in ("n_users_projected", "n_sessions_projected")
                if c in category_weekly.columns]
        cat_key = [d for d in join_key(sales) if d != "brand"]
        if keep:
            cat = collapse_exposure(category_weekly, cat_key + ["brand"]) \
                if "brand" in category_weekly.columns \
                else category_weekly.groupby(cat_key, as_index=False)[keep].sum()
            j = j.merge(cat[cat_key + keep], on=cat_key, how="left")

    rng = np.random.default_rng(cfg.seed)
    draws = ledger.draw(cfg.n_draws, seed=cfg.seed)
    p_conv = _conversion_draws(j, cfg, rng)

    value = _chain(j, draws, p_conv)
    sales_value = j["sales_value"].astype(float).to_numpy()[:, None]
    share = np.divide(value, np.maximum(sales_value, 1e-9))
    incremental = value * draws["incrementality"].to_numpy()[None, :]
    td_share = _topdown(j, draws)

    cells = j[KEY + ["sales_value", "n_sessions_mentioned", "n_sessions_converted",
                     "n_sessions_mentioned_projected", "exposure_rate"]].copy()
    for name, mat in (("influenced_value", value), ("influenced_share", share),
                      ("incremental_value", incremental),
                      ("topdown_share", td_share)):
        for q, col in _quantiles(mat).items():
            cells[f"{name}_{q}"] = col
    cells["reconciliation_ratio"] = (
        cells["influenced_share_p50"] / cells["topdown_share_p50"].replace(0, np.nan)
    ).round(3)
    cells["share_exceeds_sales"] = cells["influenced_share_p50"] > 1.0

    totals = _totals(j, value, incremental, td_share, cfg)
    sens = sensitivity(j, ledger, cfg, p_conv)

    head = _headline(totals, ledger, cells)
    return {"cells": cells, "totals": totals, "sensitivity": sens,
            "factors": ledger.to_frame(), "ledger": ledger, "headline": head,
            "config": cfg}


def _totals(j: pd.DataFrame, value: np.ndarray, incremental: np.ndarray,
            td_share: np.ndarray, cfg: BridgeConfig) -> pd.DataFrame:
    """Market-level roll-up: sum over cells inside each draw, then quantile.

    Summing first is what makes the interval right — the factors are shared
    across brands, so per-cell intervals do **not** add in quadrature.
    """
    sales_total = float(j["sales_value"].astype(float).sum())
    tot_v = value.sum(axis=0)
    tot_i = incremental.sum(axis=0)
    w = j["sales_value"].astype(float).to_numpy()[:, None]
    tot_td = (td_share * w).sum(axis=0) / max(w.sum(), 1e-9)

    def q(v, p):
        v = np.asarray(v, dtype=float)
        return float(np.nan) if np.all(np.isnan(v)) else float(np.nanquantile(v, p))

    rows = [
        {"metric": "sales_value", "unit": cfg.currency, "p05": sales_total,
         "p50": sales_total, "p95": sales_total},
        {"metric": "ai_influenced_value", "unit": cfg.currency,
         "p05": q(tot_v, .05), "p50": q(tot_v, .5), "p95": q(tot_v, .95)},
        {"metric": "ai_influenced_share", "unit": "share",
         "p05": q(tot_v, .05) / sales_total, "p50": q(tot_v, .5) / sales_total,
         "p95": q(tot_v, .95) / sales_total},
        {"metric": "ai_influenced_share_topdown", "unit": "share",
         "p05": q(tot_td, .05), "p50": q(tot_td, .5), "p95": q(tot_td, .95)},
        {"metric": "ai_incremental_value", "unit": cfg.currency,
         "p05": q(tot_i, .05), "p50": q(tot_i, .5), "p95": q(tot_i, .95)},
        {"metric": "ai_incremental_share", "unit": "share",
         "p05": q(tot_i, .05) / sales_total, "p50": q(tot_i, .5) / sales_total,
         "p95": q(tot_i, .95) / sales_total},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Where the uncertainty lives
# --------------------------------------------------------------------------- #
def sensitivity(j: pd.DataFrame, ledger: Ledger, cfg: BridgeConfig,
                p_conv: np.ndarray) -> pd.DataFrame:
    """One-factor-at-a-time tornado on the total influenced value.

    For each factor: re-run the chain with that factor drawn from its range
    and every other factor pinned at its central value, then report the
    resulting 90% interval as a ratio of the central total. ``panel_sampling``
    does the same for the measured conversion rate. ``variance_share``
    normalises the log-variances so the rows sum to 1 — read it as "this
    assumption owns X% of the uncertainty in the answer".
    """
    central = pd.DataFrame({n: [ledger.value(n)] for n in ledger.names})
    wide = central.loc[central.index.repeat(cfg.n_draws)].reset_index(drop=True)
    p_central = np.median(p_conv, axis=1)[:, None]
    base = float(_chain(j, central, p_central).sum())

    rows: List[dict] = []
    for i, name in enumerate(f for f in CHAIN_FACTORS if f in ledger):
        d = wide.copy()
        d[name] = ledger[name].draw(np.random.default_rng(cfg.seed + i), cfg.n_draws)
        rows.append(_sens_row(name, _chain(j, d, p_central).sum(axis=0), base,
                              ledger[name].status))

    # the measured link: factors pinned, only the panel's conversion posterior moves
    rows.append(_sens_row("panel_sampling", _chain(j, central, p_conv).sum(axis=0),
                          base, "measured"))

    # incrementality multiplies the finished chain, so its bar is its own range
    if "incrementality" in ledger:
        inc = ledger["incrementality"].draw(
            np.random.default_rng(cfg.seed + 991), cfg.n_draws)
        rows.append(_sens_row("incrementality", base * inc,
                              base * ledger.value("incrementality"),
                              ledger["incrementality"].status))

    out = pd.DataFrame(rows)
    v = out["log_variance"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["variance_share"] = (v / v.sum()).round(6) if v.sum() > 0 else 0.0
    return out.sort_values("variance_share", ascending=False).reset_index(drop=True)


def _sens_row(name: str, tot: np.ndarray, base: float, status: str) -> dict:
    tot = np.asarray(tot, dtype=float)
    lo, hi = float(np.quantile(tot, .05)), float(np.quantile(tot, .95))
    logs = np.log(np.maximum(tot, 1e-12))
    return {"factor": name, "status": status,
            "low_x": round(lo / base, 3) if base else None,
            "high_x": round(hi / base, 3) if base else None,
            "interval_width_x": round((hi - lo) / base, 3) if base else None,
            "log_variance": float(np.var(logs))}


def _headline(totals: pd.DataFrame, ledger: Ledger, cells: pd.DataFrame) -> str:
    def get(metric, q="p50"):
        s = totals.loc[totals["metric"] == metric, q]
        return float(s.iloc[0]) if len(s) else float("nan")

    bu, td = get("ai_influenced_share"), get("ai_influenced_share_topdown")
    lines = [
        f"AI-influenced sales share: {bu:.1%} "
        f"(90% CI {get('ai_influenced_share', 'p05'):.1%}–"
        f"{get('ai_influenced_share', 'p95'):.1%}, bottom-up)",
        f"Top-down cross-check: {td:.1%} — routes differ by "
        f"{(bu / td):.1f}×" if td and np.isfinite(td) and td > 0 else
        "Top-down cross-check: not computed (needs category-level projected "
        "reach — pass category_weekly)",
        f"AI-incremental sales share: {get('ai_incremental_share'):.1%} "
        f"(90% CI {get('ai_incremental_share', 'p05'):.1%}–"
        f"{get('ai_incremental_share', 'p95'):.1%}) — replace with the MMM's "
        f"coefficient as soon as one exists",
        ledger.headline(),
    ]
    n_over = int(cells["share_exceeds_sales"].sum())
    if n_over:
        lines.append(f"WARNING: {n_over} brand-weeks estimate more influenced "
                     f"sales than the brand sold — the factor chain is "
                     f"inconsistent for those cells, do not publish them.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Incrementality helpers
# --------------------------------------------------------------------------- #
def incrementality_from_lift(features: pd.DataFrame,
                             exposure_col: str = "followed_recommendation",
                             ) -> Optional[Dict[str, float]]:
    """Upper-bound incrementality from the panel's own exposed/unexposed lift.

    ``1 − 1/lift`` is the share of exposed conversions that exceed the
    unexposed base rate. It is an **upper bound**, not an estimate: users who
    follow recommendations were already more likely to buy (self-selection),
    so the whole selection effect is charged to the AI here. Use it to bound
    the ledger's ``incrementality``, never as its value.
    """
    if features is None or features.empty or exposure_col not in features.columns:
        return None
    ex = features[features[exposure_col].astype(bool)]
    un = features[~features[exposure_col].astype(bool)]
    if not len(ex) or not len(un):
        return None
    r_ex = float(ex["converted"].astype(bool).mean())
    r_un = float(un["converted"].astype(bool).mean())
    if r_un <= 0 or r_ex <= 0:
        return None
    lift = r_ex / r_un
    return {"rate_exposed": round(r_ex, 4), "rate_unexposed": round(r_un, 4),
            "lift": round(lift, 3), "incrementality_upper_bound": round(1 - 1 / lift, 4),
            "n_exposed": len(ex), "n_unexposed": len(un)}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
_S, _I32, _F64, _B = pa.string(), pa.int32(), pa.float64(), pa.bool_()

CELL_SCHEMA = pa.schema(
    [(d, _S) for d in DIMENSIONS] + [("brand", _S), ("sales_value", _F64),
     ("n_sessions_mentioned", _I32), ("n_sessions_converted", _I32),
     ("n_sessions_mentioned_projected", _F64), ("exposure_rate", _F64)]
    + [(f"{m}_{q}", _F64)
       for m in ("influenced_value", "influenced_share", "incremental_value",
                 "topdown_share")
       for q in ("p05", "p50", "p95")]
    + [("reconciliation_ratio", _F64), ("share_exceeds_sales", _B)])

TOTALS_SCHEMA = pa.schema([("metric", _S), ("unit", _S), ("p05", _F64),
                           ("p50", _F64), ("p95", _F64)])

SENSITIVITY_SCHEMA = pa.schema([("factor", _S), ("status", _S), ("low_x", _F64),
                                ("high_x", _F64), ("interval_width_x", _F64),
                                ("log_variance", _F64), ("variance_share", _F64)])

FACTOR_SCHEMA = pa.schema([("name", _S), ("value", _F64), ("low", _F64),
                           ("high", _F64), ("unit", _S), ("status", _S),
                           ("source", _S), ("note", _S)])


def write_bridge(result: Dict[str, object], cfg: Optional[BridgeConfig] = None
                 ) -> List[str]:
    cfg = cfg or result.get("config") or BridgeConfig()
    out = []
    for key, schema, name in (("cells", CELL_SCHEMA, "ai_influenced_sales.parquet"),
                              ("totals", TOTALS_SCHEMA, "ai_attribution_totals.parquet"),
                              ("sensitivity", SENSITIVITY_SCHEMA,
                               "ai_attribution_sensitivity.parquet"),
                              ("factors", FACTOR_SCHEMA,
                               "ai_attribution_factors.parquet")):
        df = result.get(key)
        if df is None or getattr(df, "empty", True):
            continue
        rows = df.reindex(columns=[f.name for f in schema]).to_dict("records")
        out.append(write_parquet(rows, schema, cfg.path(name)))
    return out
