"""Marketing-mix modelling with an AI channel — and the honest way to add one.

An MMM is how the business already converts activity into incremental sales,
so the AI channel should enter *that* model rather than get a parallel one of
its own. Three things make the AI variable different from a media channel, and
the design here is shaped by all three:

**There is no spend.** ROI is a ratio to cost, and the cost of an LLM naming
your brand is zero. So the AI variable is modelled as an *exposure index*, its
output is a **contribution** (euros of sales, share of total), and any
"return" statement has to name the investment it is a return on — content,
PR, retailer feeds — not a media buy.

**It is a trend.** Consumer LLM usage grows monotonically, and so does every
other digital variable in the window. A raw conversation count is collinear
with time and with paid digital; the model cannot tell them apart and will
hand back whatever the regulariser prefers. The default AI regressor is
therefore **share of voice** — the brand's share of category brand mentions —
which moves with competitive dynamics, not with adoption. :func:`diagnostics`
reports the VIF so this stops being a matter of opinion.

**It is endogenous.** Assistants mention brands that sell well, are widely
distributed and are written about — reverse causality straight into the
coefficient. Two defences are built in: controls (price, distribution,
seasonality, brand fixed effects) and the **two-stage** decomposition
(:func:`two_stage`), which models media/PR → AI visibility → sales, so paid
activity's effect *through* the assistant is separated from the assistant's
own effect.

Finally, :func:`prior_from_bridge` closes the loop with the panel: the sales
bridge's estimate becomes an informative prior on the AI coefficient, and
:func:`fit` shrinks toward it. That is the same calibration logic as
experiment-calibrated MMM — with the panel funnel standing in for a geo test
until real experiments exist.

The implementation is deliberately small (numpy + a closed-form penalised
regression): it is a reference model for validating the AI variable and
sizing the effect, not a replacement for the production MMM. What ships to the
MMM team is :func:`design_frame` — the regressor table at their grain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from ..schema_utils import write_parquet


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def geometric_adstock(x: Sequence[float], decay: float = 0.5) -> np.ndarray:
    """Carry-over: ``a[t] = x[t] + decay · a[t-1]``. ``decay=0`` is no memory."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def hill_saturation(x: Sequence[float], half: Optional[float] = None,
                    shape: float = 1.0) -> np.ndarray:
    """Diminishing returns: ``x^s / (x^s + half^s)``, in [0, 1).

    ``half`` defaults to the series' own mean, which keeps the curve in its
    informative region without hand-tuning per channel — the usual failure
    mode being a saturation point so high the transform is linear.
    """
    x = np.asarray(x, dtype=float)
    pos = x[x > 0]
    h = float(half if half is not None else (pos.mean() if pos.size else 1.0))
    h = max(h, 1e-9)
    xs = np.power(np.maximum(x, 0.0), shape)
    return xs / (xs + h ** shape)


def fourier_terms(week_index: Sequence[int], period: int = 52,
                  harmonics: int = 2) -> Dict[str, np.ndarray]:
    t = np.asarray(week_index, dtype=float)
    out: Dict[str, np.ndarray] = {}
    for k in range(1, harmonics + 1):
        out[f"seas_sin{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"seas_cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class MMMConfig:
    target: str = "sales_value"
    #: the AI regressor. share_of_voice by default — see the module docstring
    ai_col: str = "ai_share_of_voice"
    media_cols: Sequence[str] = ()
    control_cols: Sequence[str] = ()          # price_index, distribution, …
    adstock: Dict[str, float] = field(default_factory=dict)   # channel → decay
    default_decay: float = 0.4
    saturate_media: bool = True
    brand_fixed_effects: bool = True
    seasonality_harmonics: int = 2
    include_trend: bool = True
    ridge_alpha: float = 1.0
    holdout_weeks: int = 13                   # time-ordered, never random
    out_dir: str = "outputs/attribution"

    def path(self, name: str) -> str:
        import os
        return os.path.join(self.out_dir, name)


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
def design_frame(brand_weekly: pd.DataFrame, sales: pd.DataFrame,
                 media: Optional[pd.DataFrame] = None,
                 keys: Sequence[str] = ("week", "market", "brand"),
                 ) -> pd.DataFrame:
    """The deliverable for the MMM team: one row per brand-market-week.

    Joins the panel's AI exposure metrics onto the sales fact and, when given,
    the media spend table. Left-joined on sales so weeks with no AI
    conversations become explicit zeros rather than dropped rows — a gap in
    the panel is a zero of exposure, not missing data, and dropping those
    weeks would bias the coefficient upward.
    """
    keys = list(keys)
    ai_cols = [c for c in ("ai_share_of_voice", "exposure_rate",
                           "n_sessions_mentioned", "n_sessions_unsolicited",
                           "n_sessions_mentioned_projected",
                           "n_sessions_visited_projected")
               if c in brand_weekly.columns]
    # counts add across platforms; shares do not — averaging them is a rough
    # platform-equal-weight, and the alternative (a mention-weighted share) is
    # only right once platform panel sizes are comparable
    rates = {"ai_share_of_voice", "exposure_rate", "visit_rate_given_exposure"}
    ai = (brand_weekly.groupby(keys, as_index=False)
          .agg({c: ("mean" if c in rates else "sum") for c in ai_cols})
          if ai_cols else brand_weekly[keys].drop_duplicates())

    out = sales.merge(ai, on=keys, how="left")
    for c in ai_cols:
        out[c] = out[c].fillna(0.0)
    if media is not None and not media.empty:
        mkeys = [k for k in keys if k in media.columns]
        out = out.merge(media, on=mkeys, how="left")
        for c in media.columns:
            if c not in mkeys:
                out[c] = out[c].fillna(0.0)
    out = out.sort_values(keys).reset_index(drop=True)
    out["week_index"] = pd.factorize(out["week"].astype(str), sort=True)[0]
    return out


def build_matrix(frame: pd.DataFrame, cfg: MMMConfig
                 ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """(X, y, feature names, unpenalised feature names).

    Media columns are adstocked then saturated **within each brand-market
    series** — running the carry-over across a concatenated panel would leak
    one brand's spend into the next brand's first weeks.
    """
    df = frame.copy()
    cols: Dict[str, np.ndarray] = {}
    free: List[str] = []

    group_keys = [k for k in ("market", "brand") if k in df.columns]
    for ch in cfg.media_cols:
        if ch not in df.columns:
            continue
        decay = cfg.adstock.get(ch, cfg.default_decay)
        vals = np.zeros(len(df))
        it = df.groupby(group_keys).indices.items() if group_keys else [((), df.index)]
        for _, idx in it:
            idx = np.asarray(idx)
            order = idx[np.argsort(df.iloc[idx]["week_index"].to_numpy())]
            series = geometric_adstock(df.iloc[order][ch].to_numpy(), decay)
            vals[order] = series
        cols[f"{ch}_adstock"] = hill_saturation(vals) if cfg.saturate_media else vals

    if cfg.ai_col in df.columns:
        cols[cfg.ai_col] = df[cfg.ai_col].astype(float).to_numpy()
    for c in cfg.control_cols:
        if c in df.columns:
            cols[c] = df[c].astype(float).to_numpy()

    if cfg.seasonality_harmonics:
        for name, v in fourier_terms(df["week_index"], 52,
                                     cfg.seasonality_harmonics).items():
            cols[name] = v
            free.append(name)
    if cfg.include_trend:
        wi = df["week_index"].to_numpy(dtype=float)
        cols["trend"] = (wi - wi.mean()) / max(wi.std(), 1e-9)
        free.append("trend")

    if cfg.brand_fixed_effects and "brand" in df.columns:
        for b in sorted(df["brand"].unique())[1:]:      # first brand = reference
            cols[f"fe_brand_{b}"] = (df["brand"] == b).to_numpy(dtype=float)
            free.append(f"fe_brand_{b}")
    cols["intercept"] = np.ones(len(df))
    free.append("intercept")

    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    y = df[cfg.target].astype(float).to_numpy()
    return X, y, names, free


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _penalised_ls(X: np.ndarray, y: np.ndarray, lam: np.ndarray,
                  prior_idx: Optional[int] = None, prior_mean: float = 0.0,
                  prior_strength: float = 0.0) -> np.ndarray:
    """argmin ‖y − Xb‖² + Σλ b² + κ(b_i − m)²  — closed form.

    The third term is the calibration hook: it pulls one coefficient toward a
    value estimated outside the regression (the panel bridge), with κ setting
    how much the data has to fight to move it. κ = 0 is plain ridge.
    """
    A = X.T @ X + np.diag(lam)
    b = X.T @ y
    if prior_idx is not None and prior_strength > 0:
        A[prior_idx, prior_idx] += prior_strength
        b[prior_idx] += prior_strength * prior_mean
    return np.linalg.solve(A, b)


def fit(frame: pd.DataFrame, cfg: Optional[MMMConfig] = None,
        prior_mean: Optional[float] = None, prior_strength: float = 0.0,
        ) -> Dict[str, object]:
    """Fit the reference MMM and decompose sales into channel contributions.

    ``prior_mean`` (with ``prior_strength`` > 0) shrinks the AI coefficient
    toward an externally estimated value — use :func:`prior_from_bridge`.
    Validation is a **time-ordered holdout** (the last ``holdout_weeks``):
    random splits leak the future into training and flatter every MMM.
    """
    cfg = cfg or MMMConfig()
    X, y, names, free = build_matrix(frame, cfg)
    lam = np.array([0.0 if n in free else cfg.ridge_alpha for n in names])

    wi = frame["week_index"].to_numpy()
    cut = wi.max() - cfg.holdout_weeks
    train = wi <= cut if cfg.holdout_weeks and cut > wi.min() else np.ones(len(wi), bool)
    test = ~train

    ai_idx = names.index(cfg.ai_col) if cfg.ai_col in names else None
    beta = _penalised_ls(X[train], y[train], lam, ai_idx,
                         prior_mean or 0.0, prior_strength if prior_mean else 0.0)

    pred = X @ beta
    contrib = X * beta[None, :]
    coefs = pd.DataFrame({"feature": names, "coef": np.round(beta, 6),
                          "contribution": contrib.sum(axis=0)})
    total = float(pred.sum())
    coefs["contribution_share"] = (coefs["contribution"] / total).round(4) if total else np.nan

    fitres = {
        "coefficients": coefs.sort_values("contribution", key=np.abs,
                                          ascending=False).reset_index(drop=True),
        "r2_train": _r2(y[train], pred[train]),
        "r2_holdout": _r2(y[test], pred[test]) if test.any() else None,
        "mape_holdout": _mape(y[test], pred[test]) if test.any() else None,
        "n_train": int(train.sum()), "n_holdout": int(test.sum()),
        "prediction": pred, "feature_names": names, "beta": beta,
        "design": X, "target": y, "config": cfg,
        "prior_mean": prior_mean, "prior_strength": prior_strength,
    }
    if ai_idx is not None:
        ai_contrib = float(contrib[:, ai_idx].sum())
        fitres["ai_contribution"] = ai_contrib
        fitres["ai_contribution_share"] = ai_contrib / total if total else np.nan
        fitres["ai_coefficient"] = float(beta[ai_idx])
    fitres["diagnostics"] = diagnostics(X, names, cfg)
    return fitres


def _r2(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(y) < 2:
        return None
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else None


def _mape(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    m = np.abs(y) > 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m]))) if m.any() else None


def diagnostics(X: np.ndarray, names: List[str], cfg: MMMConfig) -> pd.DataFrame:
    """Variance-inflation per feature — the collinearity story, in numbers.

    A VIF above ~10 on the AI variable means the model cannot separate it from
    the rest of the mix; the coefficient may be precise-looking and
    meaningless. That is the single most likely way an AI-in-MMM result goes
    wrong, so it is computed every fit and belongs in every readout.
    """
    rows = []
    for i, name in enumerate(names):
        if name == "intercept":
            continue
        others = np.delete(X, i, axis=1)
        target = X[:, i]
        if np.allclose(target.std(), 0):
            rows.append({"feature": name, "vif": np.nan, "flag": "constant"})
            continue
        beta, *_ = np.linalg.lstsq(others, target, rcond=None)
        r2 = _r2(target, others @ beta) or 0.0
        vif = 1.0 / max(1 - r2, 1e-6)
        rows.append({"feature": name, "vif": round(float(vif), 2),
                     "flag": "collinear" if vif > 10 else ""})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


AI_VARIABLE_CANDIDATES = ("ai_share_of_voice", "exposure_rate",
                          "n_sessions_mentioned_projected",
                          "n_sessions_unsolicited")


def compare_ai_variables(frame: pd.DataFrame, cfg: Optional[MMMConfig] = None,
                         candidates: Sequence[str] = AI_VARIABLE_CANDIDATES,
                         ) -> pd.DataFrame:
    """Fit the same model with each candidate AI regressor and compare.

    The choice of AI variable is the single most consequential modelling
    decision in this whole layer, and it is not settled by taste — run this
    and read the table. The two failure modes it exposes:

    * **volume** variables (projected conversation counts) carry the adoption
      trend, so they are collinear with everything else that grew and can come
      back with the wrong sign — a negative coefficient here means "the model
      cannot see this", not "AI hurts sales";
    * **share** variables (share of voice, exposure rate) are trend-free and
      well-conditioned, but they discard the adoption signal, so they
      systematically *under*-attribute a channel whose effect is still
      growing.

    Neither is right on its own. The practical resolution is cross-sectional
    variation (several markets/brands, so the trend is not the only mover) plus
    an external anchor on the coefficient — which is what
    :func:`prior_from_bridge` supplies.
    """
    cfg = cfg or MMMConfig()
    rows = []
    for col in candidates:
        if col not in frame.columns:
            continue
        c = MMMConfig(**{**cfg.__dict__, "ai_col": col})
        f = fit(frame, c)
        vif = f["diagnostics"].set_index("feature")["vif"].get(col, np.nan)
        rows.append({"ai_variable": col,
                     "ai_contribution_share": f.get("ai_contribution_share"),
                     "coef_sign": "+" if f.get("ai_coefficient", 0) >= 0 else "−",
                     "vif": float(vif) if vif == vif else np.nan,
                     "r2_holdout": f.get("r2_holdout"),
                     "verdict": ("wrong sign — collinear, unusable"
                                 if f.get("ai_coefficient", 0) < 0 else
                                 "collinear — wide, fragile" if vif > 10 else
                                 "well-conditioned")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calibration from the panel bridge
# --------------------------------------------------------------------------- #
def prior_from_bridge(bridge_result: Dict[str, object], frame: pd.DataFrame,
                      cfg: Optional[MMMConfig] = None,
                      strength: float = 1.0) -> Dict[str, float]:
    """Turn the bridge's incremental-sales estimate into a prior on β_ai.

    The bridge says "AI drove ≈ X € of incremental sales over the period"; the
    MMM's AI contribution is ``β_ai · Σ ai_var``. Equating them gives the
    prior mean. ``strength`` is expressed in *pseudo-observations*: 1.0 lets
    the data override easily, 50 makes the model argue hard. Report both the
    free and the calibrated fit — a large gap between them is the finding.
    """
    cfg = cfg or MMMConfig()
    totals = bridge_result.get("totals")
    if totals is None or cfg.ai_col not in frame.columns:
        return {}
    row = totals.loc[totals["metric"] == "ai_incremental_value", "p50"]
    if not len(row):
        return {}
    denom = float(frame[cfg.ai_col].astype(float).sum())
    if denom <= 0:
        return {}
    return {"prior_mean": float(row.iloc[0]) / denom,
            "prior_strength": float(strength) * len(frame)}


# --------------------------------------------------------------------------- #
# Two-stage: media → AI visibility → sales
# --------------------------------------------------------------------------- #
def two_stage(frame: pd.DataFrame, cfg: Optional[MMMConfig] = None,
              visibility_drivers: Sequence[str] = ()) -> Dict[str, object]:
    """Decompose each driver's effect into direct and AI-mediated.

    Stage 1 regresses the AI variable on the drivers you can actually pull
    (media, PR, distribution, content); stage 2 is the ordinary MMM. The
    indirect effect of driver *c* is ``β_c→ai · β_ai→sales`` — the euros that
    the assistant passed through. This is the practically useful output: you
    cannot buy a slot in an LLM answer, but you can move what it says, and
    this quantifies the return on doing so.

    Both stages are correlational. The mediation arithmetic assumes the usual
    (untestable here) no-unmeasured-confounding condition; treat the split as
    a decomposition of the fitted model, not a causal claim.
    """
    cfg = cfg or MMMConfig()
    drivers = list(visibility_drivers) or [c for c in cfg.media_cols if c in frame.columns]
    if cfg.ai_col not in frame.columns or not drivers:
        return {}

    Xs = np.column_stack([frame[d].astype(float).to_numpy() for d in drivers]
                         + [np.ones(len(frame))])
    ys = frame[cfg.ai_col].astype(float).to_numpy()
    b1 = _penalised_ls(Xs, ys, np.array([cfg.ridge_alpha] * len(drivers) + [0.0]))
    stage1 = pd.DataFrame({"driver": drivers + ["intercept"],
                           "coef_to_ai_visibility": np.round(b1, 6)})

    sales_fit = fit(frame, cfg)
    b_ai = float(sales_fit.get("ai_coefficient", 0.0))
    coefs = sales_fit["coefficients"].set_index("feature")["coef"]

    rows = []
    for d, b in zip(drivers, b1[:-1]):
        direct = float(coefs.get(f"{d}_adstock", coefs.get(d, 0.0)))
        indirect = float(b) * b_ai
        rows.append({"driver": d, "direct_coef": round(direct, 6),
                     "indirect_via_ai": round(indirect, 6),
                     "share_via_ai": round(indirect / (direct + indirect), 4)
                     if (direct + indirect) else np.nan})
    return {"stage1": stage1, "stage1_r2": _r2(ys, Xs @ b1),
            "mediation": pd.DataFrame(rows), "sales_fit": sales_fit,
            "ai_coefficient": b_ai}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
_S, _F64 = pa.string(), pa.float64()

COEF_SCHEMA = pa.schema([("feature", _S), ("coef", _F64), ("contribution", _F64),
                         ("contribution_share", _F64)])
VIF_SCHEMA = pa.schema([("feature", _S), ("vif", _F64), ("flag", _S)])


def write_mmm(fit_result: Dict[str, object], cfg: Optional[MMMConfig] = None
              ) -> List[str]:
    cfg = cfg or fit_result.get("config") or MMMConfig()
    out = [write_parquet(fit_result["coefficients"].to_dict("records"),
                         COEF_SCHEMA, cfg.path("mmm_coefficients.parquet"))]
    diag = fit_result.get("diagnostics")
    if diag is not None and not diag.empty:
        out.append(write_parquet(diag.to_dict("records"), VIF_SCHEMA,
                                 cfg.path("mmm_diagnostics.parquet")))
    return out
