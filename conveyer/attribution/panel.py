"""Panel → market projection: from "N panelists did X" to "N people did X".

The clickstream is a **panel**, not a census. Every count coming out of
modules 1–3 is a sample statistic, and the sales bridge needs population
quantities. That conversion is three separate things, kept separate here
because they fail differently:

1. **Projection** — multiply by universe/panel. Mechanical, needs only the
   panel frame (panel size, universe size). Carries binomial sampling error.
2. **Post-stratification** — reweight so the panel's demographic mix matches
   the universe. Needs demographics we may not get; when they are absent the
   frame records ``demo_status="unweighted"`` and the ledger's
   ``panel_representativeness`` band (not a point correction) carries the risk.
   Nothing here ever silently pretends the panel is representative.
3. **Coverage grossing-up** — desktop-only observation of an all-device
   behaviour. That factor lives in the ledger (``device_coverage``), not here,
   because it is a benchmark rather than a property of the panel frame.

Uncertainty is reported as an *effective* sample size: weighting costs
precision (Kish), so a weighted count of 1 000 may carry the sampling error of
600. Quoting the raw n after weighting is the classic overstatement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PanelFrame:
    """What we know about the panel behind one market's data."""

    market: str = "US"
    panel_users: int = 50_000          # panelists contributing observations
    universe_users: int = 250_000_000  # online population the panel stands for
    #: observed → all-device grossing is a ledger factor; kept here only as a
    #: label so reports can say what the panel *did* observe
    devices: str = "desktop"
    #: "unweighted" (no demographics), "post_stratified", or "provider_weighted"
    demo_status: str = "unweighted"
    #: optional segment frame: columns [segment, panel_share, universe_share]
    demo_frame: Optional[pd.DataFrame] = None
    #: provider-supplied per-user weights, if any: {user_id: weight}
    user_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def projection_factor(self) -> float:
        """Universe members represented by one panelist."""
        if self.panel_users <= 0:
            raise ValueError("panel_users must be positive to project")
        return self.universe_users / self.panel_users

    def describe(self) -> str:
        return (f"{self.market}: {self.panel_users:,} {self.devices} panelists → "
                f"{self.universe_users:,} universe (×{self.projection_factor:,.0f}), "
                f"demographics: {self.demo_status}")


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def post_stratification_weights(frame: PanelFrame) -> Optional[pd.DataFrame]:
    """Segment weights = universe_share / panel_share.

    Returns None when no demographic frame is available — the expected state
    until the provider ships demographics. Callers must handle None rather
    than defaulting to 1.0 silently, so "we could not weight" stays visible.
    """
    df = frame.demo_frame
    if df is None or df.empty:
        return None
    need = {"segment", "panel_share", "universe_share"}
    if not need.issubset(df.columns):
        raise ValueError(f"demo_frame needs columns {sorted(need)}")
    out = df.copy()
    panel = out["panel_share"].replace(0, np.nan)
    out["weight"] = (out["universe_share"] / panel).fillna(1.0)
    # normalise so the weighted panel size equals the panel size
    out["weight"] = out["weight"] / (out["weight"] * out["panel_share"]).sum()
    return out[["segment", "panel_share", "universe_share", "weight"]]


def effective_n(weights: Sequence[float]) -> float:
    """Kish effective sample size ``(Σw)² / Σw²`` — the n that the weighted
    estimate really has. Equal weights give back the raw count."""
    w = np.asarray(list(weights), dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def design_effect(weights: Sequence[float]) -> float:
    """n / n_eff — how much precision the weighting costs (≥ 1)."""
    w = np.asarray(list(weights), dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 1.0
    return float(w.size / max(effective_n(w), 1e-9))


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def project_counts(df: pd.DataFrame, count_cols: Sequence[str], frame: PanelFrame,
                   base_col: Optional[str] = None, deff: float = 1.0,
                   suffix: str = "_projected") -> pd.DataFrame:
    """Project panel counts to the universe, with sampling intervals.

    ``count_cols`` are projected by the frame's factor. When ``base_col`` names
    the denominator each count was drawn from (e.g. sessions in the cell), the
    projection also gets a **95% interval** from the binomial error on the
    underlying proportion, widened by the design effect. Without a base there
    is no honest interval and the columns are left absent rather than filled
    with a fabricated one.
    """
    out = df.copy()
    factor = frame.projection_factor
    for col in count_cols:
        if col not in out.columns:
            continue
        n = out[col].astype(float)
        out[col + suffix] = n * factor
        if base_col and base_col in out.columns:
            base = out[base_col].astype(float).replace(0, np.nan)
            p = (n / base).clip(0, 1)
            se = np.sqrt(np.maximum(p * (1 - p), 0) / base * max(deff, 1.0))
            out[col + suffix + "_lo"] = ((p - 1.96 * se).clip(lower=0) * base * factor)
            out[col + suffix + "_hi"] = ((p + 1.96 * se).clip(upper=1) * base * factor)
    out.attrs["projection_factor"] = factor
    out.attrs["design_effect"] = float(deff)
    out.attrs["panel_frame"] = frame.describe()
    return out


def apply_user_weights(sessions: pd.DataFrame, frame: PanelFrame,
                       user_col: str = "user_id") -> pd.DataFrame:
    """Attach a per-session weight column (1.0 when the provider gave none).

    Also records the design effect on ``.attrs`` so downstream intervals can
    be widened by it instead of quoting the raw n.
    """
    out = sessions.copy()
    if frame.user_weights and user_col in out.columns:
        out["panel_weight"] = out[user_col].map(frame.user_weights).fillna(1.0)
    else:
        out["panel_weight"] = 1.0
    out.attrs["design_effect"] = design_effect(out["panel_weight"])
    return out
