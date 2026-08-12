"""Synthetic market data with a planted AI effect — so the layer runs today.

None of the market-side inputs exist yet: the data partnership is unsigned,
RMS/Omnisales/CPS extracts are not wired, and the MMM specs are still to
come. Following the project's offline-first rule, this module fabricates all
of them — sales, media spend, weekly AI exposure — from a **known** generating
process, so:

* the bridge and the MMM run end to end with one command, today;
* the MMM can be *validated*: the generator records the true AI contribution
  share, and the tests assert the fit recovers it;
* every fabricated table is produced by a function whose name says it is
  fabricated, instead of a spreadsheet nobody can trace.

Nothing here is a business estimate. The planted effect size is a dial, not a
finding; swap each ``make_*`` call for a real extract and the rest of the
package does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .panel import PanelFrame


@dataclass
class DemoSpec:
    n_weeks: int = 104
    brands: Sequence[str] = ("nivea", "cerave", "la roche-posay", "the ordinary")
    markets: Sequence[str] = ("US",)
    category: str = "face skincare"
    platforms: Sequence[str] = ("chatgpt", "gemini", "perplexity")
    #: the truth the MMM has to find back: AI's share of total sales
    true_ai_contribution_share: float = 0.02
    media_channels: Sequence[str] = ("tv_spend", "digital_spend", "retail_media_spend")
    media_contribution_share: float = 0.22
    seed: int = 11
    panel_users: int = 60_000
    universe_users: int = 210_000_000
    #: explicit week labels — set by :meth:`like` so fabricated sales line up
    #: with a real panel run instead of failing the join
    weeks: Optional[Sequence[str]] = None

    @classmethod
    def like(cls, brand_weekly: pd.DataFrame, **overrides) -> "DemoSpec":
        """A spec whose weeks, brands, markets and category match a real
        exposure table — the only way fabricated sales can be joined to a real
        panel run without a brand-mapping table."""
        def uniq(col, fallback):
            if col not in brand_weekly.columns:
                return fallback
            vals = [v for v in sorted(brand_weekly[col].dropna().unique()) if str(v)]
            return tuple(vals) or fallback
        weeks = [w for w in sorted(brand_weekly["week"].dropna().unique())
                 if str(w) != "unknown"] if "week" in brand_weekly.columns else None
        cat = uniq("category", ("face skincare",))
        return cls(weeks=weeks or None, n_weeks=len(weeks) if weeks else cls.n_weeks,
                   brands=uniq("brand", cls.brands), markets=uniq("market", ("US",)),
                   category=cat[0], platforms=uniq("llm_platform", cls.platforms),
                   **overrides)


def _weeks(n: int, start: str = "2025-01-06") -> List[str]:
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, periods=n, freq="W-MON")]


def make_market(spec: DemoSpec | None = None) -> Dict[str, object]:
    """Sales, media spend, weekly AI exposure and the planted ground truth.

    The generating process, in order: a brand base level with seasonality; a
    media response (adstocked, saturated); an AI response driven by the
    brand's **share of voice** in assistant answers, which follows a slow
    competitive random walk plus a category-wide adoption trend. Sales are
    then scaled so the AI block owns exactly ``true_ai_contribution_share``.

    The adoption trend is deliberately shared across brands and collinear with
    ``trend`` — that is the real identification problem, and a demo that
    avoided it would make the MMM look easier than it is.
    """
    spec = spec or DemoSpec()
    rng = np.random.default_rng(spec.seed)
    weeks = list(spec.weeks) if spec.weeks else _weeks(spec.n_weeks)
    n_weeks = len(weeks)
    t = np.arange(n_weeks)

    # category-wide LLM adoption: the trend the AI variable must NOT be
    adoption = 0.25 + 0.75 * t / max(n_weeks - 1, 1)

    sales_rows, media_rows, exposure_rows = [], [], []
    truth: Dict[str, float] = {}

    for market in spec.markets:
        for bi, brand in enumerate(spec.brands):
            # weekly brand sales in the market, chosen so that a plausible
            # panel reach (≈0.2–0.4 % of panelists per week) produces a
            # plausible influenced share — the two sides of the demo are
            # generated independently, and making them mutually absurd would
            # only exercise the warning paths
            base = 16_000_000 * (1.0 + 0.35 * bi)
            seas = 1 + 0.12 * np.sin(2 * np.pi * (t % 52) / 52 + bi)

            # competitive share of voice: mean-reverting walk, brand-specific
            sov = np.clip(0.15 + 0.10 * np.cumsum(rng.normal(0, 0.035, n_weeks))
                          - 0.02 * bi, 0.02, 0.6)
            sov = pd.Series(sov).rolling(5, min_periods=1).mean().to_numpy()

            spend = {}
            for ci, ch in enumerate(spec.media_channels):
                burst = (rng.random(n_weeks) < 0.28).astype(float)
                spend[ch] = np.round(burst * rng.uniform(40_000, 180_000, n_weeks)
                                     * (1.0 + 0.2 * ci), 0)

            from .mmm import geometric_adstock, hill_saturation
            media_effect = sum(hill_saturation(geometric_adstock(spend[ch], 0.45))
                               for ch in spec.media_channels) / len(spec.media_channels)
            ai_effect = sov * adoption

            m = spec.media_contribution_share
            a = spec.true_ai_contribution_share
            core = base * seas
            sales = (core * (1 - m - a)
                     + core.mean() * m * media_effect / max(media_effect.mean(), 1e-9)
                     + core.mean() * a * ai_effect / max(ai_effect.mean(), 1e-9))
            sales = sales * rng.normal(1.0, 0.03, n_weeks)
            truth[f"{market}|{brand}"] = a

            for i, wk in enumerate(weeks):
                sales_rows.append({
                    "week": wk, "market": market, "category": spec.category,
                    "brand": brand, "sales_value": float(round(sales[i], 2)),
                    "price_index": float(round(1 + 0.05 * np.sin(i / 9 + bi), 4)),
                })
                media_rows.append({"week": wk, "market": market, "brand": brand,
                                   **{ch: float(spend[ch][i])
                                      for ch in spec.media_channels}})
                # split the week's panel exposure across platforms
                shares = np.array([0.62, 0.23, 0.15])[:len(spec.platforms)]
                shares = shares / shares.sum()
                cat_sessions = int(rng.poisson(250 * adoption[i]))
                for p, sh in zip(spec.platforms, shares):
                    mentioned = int(rng.binomial(max(int(cat_sessions * sh), 0),
                                                 min(float(sov[i]), 0.95)))
                    visited = int(rng.binomial(mentioned, 0.38))
                    exposure_rows.append({
                        "week": wk, "market": market, "category": spec.category,
                        "llm_platform": p, "brand": brand,
                        "n_sessions_category": int(cat_sessions * sh),
                        "n_sessions_mentioned": mentioned,
                        "n_sessions_unsolicited": int(rng.binomial(mentioned, 0.55)),
                        "n_sessions_cited": int(rng.binomial(mentioned, 0.7)),
                        "n_sessions_visited": visited,
                        "n_sessions_followed": int(rng.binomial(visited, 0.6)),
                        "n_sessions_converted": int(rng.binomial(visited, 0.11)),
                    })

    exposure = pd.DataFrame(exposure_rows)
    dims = ["week", "market", "category", "llm_platform"]
    totals = exposure.groupby(dims)["n_sessions_mentioned"].transform("sum")
    exposure["ai_share_of_voice"] = (exposure["n_sessions_mentioned"]
                                     / totals.replace(0, np.nan)).fillna(0).round(4)
    exposure["exposure_rate"] = (exposure["n_sessions_mentioned"]
                                 / exposure["n_sessions_category"].replace(0, np.nan)
                                 ).fillna(0).round(4)
    exposure["visit_rate_given_exposure"] = (
        exposure["n_sessions_visited"]
        / exposure["n_sessions_mentioned"].replace(0, np.nan)).fillna(0).round(4)

    category = (exposure.groupby(dims, as_index=False)
                .agg(n_sessions=("n_sessions_category", "max"))
                .assign(n_users=lambda d: (d["n_sessions"] * 0.8).astype(int)))

    frame = PanelFrame(market=spec.markets[0], panel_users=spec.panel_users,
                       universe_users=spec.universe_users)
    from .panel import project_counts
    exposure = project_counts(exposure,
                              ["n_sessions_mentioned", "n_sessions_unsolicited",
                               "n_sessions_visited", "n_sessions_converted"],
                              frame, base_col="n_sessions_category")
    category = project_counts(category, ["n_sessions", "n_users"], frame)

    return {
        "sales": pd.DataFrame(sales_rows),
        "media": pd.DataFrame(media_rows),
        "brand_weekly": exposure,
        "category_weekly": category,
        "panel_frames": {spec.markets[0]: frame},
        "ground_truth": {"true_ai_contribution_share": spec.true_ai_contribution_share,
                         "true_media_contribution_share": spec.media_contribution_share,
                         "by_brand": truth},
        "spec": spec,
    }
