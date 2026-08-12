"""Orchestration: panel run (or demo) → weekly facts → euros → MMM → report.

One call wires the four pieces in the order the argument has to be made:

    aggregate   what the panel saw, at brand-market-week
    bridge      what that is worth, with declared assumptions and intervals
    mmm         how much of it is incremental, with the bridge as a prior
    report      the numbers, the assumptions, and what to go measure next

``run_attribution()`` with no arguments runs entirely on synthetic market data
(:mod:`conveyer.attribution.demo`) and scores the MMM against the planted
contribution — the same offline-first contract as the rest of the project.
Pass ``turns``/``events``/``features`` from a real pipeline run and the panel
side becomes real while the market side stays whatever you hand it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import bridge as bridge_mod
from . import mmm as mmm_mod
from .aggregate import AggregateConfig, build_weekly, write_weekly
from .calibration import Ledger, default_ledger
from .demo import DemoSpec, make_market
from .panel import PanelFrame


@dataclass
class AttributionConfig:
    out_dir: str = "outputs/attribution"
    ledger_path: Optional[str] = None      # JSON overrides for the ledger
    n_draws: int = 2000
    seed: int = 7
    #: pseudo-observation weight for the bridge-derived prior on β_ai.
    #: 0 = let the MMM speak freely (report both, always)
    prior_strength: float = 1.0
    write: bool = True


def run_attribution(turns: Optional[pd.DataFrame] = None,
                    events: Optional[pd.DataFrame] = None,
                    features: Optional[pd.DataFrame] = None,
                    sales: Optional[pd.DataFrame] = None,
                    media: Optional[pd.DataFrame] = None,
                    frames: Optional[Dict[str, PanelFrame]] = None,
                    ledger: Optional[Ledger] = None,
                    cfg: Optional[AttributionConfig] = None,
                    demo_spec: Optional[DemoSpec] = None,
                    ) -> Dict[str, object]:
    """Run the sales-attribution layer end to end.

    Any of the panel-side frames may be omitted; whatever is missing comes
    from the synthetic market so the run always completes and always says
    which half was fabricated.
    """
    cfg = cfg or AttributionConfig()
    ledger = ledger or (Ledger.load(cfg.ledger_path) if cfg.ledger_path
                        else default_ledger())
    agg_cfg = AggregateConfig(out_dir=cfg.out_dir)

    provenance = {"panel": "real" if turns is not None else "synthetic",
                  "market": "real" if sales is not None else "synthetic"}

    # -- 1 · weekly facts -------------------------------------------------- #
    if turns is not None:
        if not frames:
            # projection is required downstream; assuming a frame beats
            # failing, but the assumption goes in the provenance, not the fog
            frames = {"*": PanelFrame()}
            provenance["panel_frame"] = "ASSUMED DEFAULT — pass frames= with " \
                                        "the provider's real panel/universe sizes"
        tables = build_weekly(turns, events, features, agg_cfg, frames)
        brand_weekly, category_weekly = tables["brand_weekly"], tables["category_weekly"]
        demo = None
    else:
        demo = make_market(demo_spec)
        brand_weekly, category_weekly = demo["brand_weekly"], demo["category_weekly"]
        tables = {"brand_weekly": brand_weekly, "category_weekly": category_weekly}

    if sales is None:
        # fabricate the market side *around the observed panel* — same brands,
        # same weeks — so the join exercises the real keys instead of failing
        demo = demo or make_market(DemoSpec.like(brand_weekly))
        sales, media = demo["sales"], media if media is not None else demo["media"]

    # -- 2 · euros ---------------------------------------------------------- #
    # the panel's own exposed/unexposed lift bounds incrementality from above;
    # record it next to the assumption rather than quietly replacing it
    bound = bridge_mod.incrementality_from_lift(features) if features is not None else None
    if bound:
        ledger.set("incrementality",
                   high=max(ledger["incrementality"].high,
                            float(bound["incrementality_upper_bound"])),
                   note=ledger["incrementality"].note +
                   f" Panel lift bound this run: ≤{bound['incrementality_upper_bound']:.2f}.")

    bcfg = bridge_mod.BridgeConfig(n_draws=cfg.n_draws, seed=cfg.seed,
                                   out_dir=cfg.out_dir)
    br = bridge_mod.estimate_influenced_sales(brand_weekly, sales, ledger, bcfg,
                                              category_weekly=category_weekly)

    # -- 3 · MMM ------------------------------------------------------------ #
    media_cols = [c for c in (media.columns if media is not None else [])
                  if c.endswith("_spend")]
    mcfg = mmm_mod.MMMConfig(media_cols=media_cols,
                             control_cols=[c for c in ("price_index",)
                                           if c in sales.columns],
                             out_dir=cfg.out_dir)
    frame = mmm_mod.design_frame(brand_weekly, sales, media)
    free_fit = mmm_mod.fit(frame, mcfg)
    prior = mmm_mod.prior_from_bridge(br, frame, mcfg, cfg.prior_strength)
    cal_fit = mmm_mod.fit(frame, mcfg, **prior) if prior else None
    variables = mmm_mod.compare_ai_variables(frame, mcfg)

    # -- 4 · outputs -------------------------------------------------------- #
    if cfg.write:
        write_weekly(tables, agg_cfg)
        bridge_mod.write_bridge(br, bcfg)
        mmm_mod.write_mmm(cal_fit or free_fit, mcfg)

    evaluation = {}
    if demo is not None:
        truth = demo["ground_truth"]["true_ai_contribution_share"]
        evaluation = {"true_ai_contribution_share": truth,
                      "mmm_ai_contribution_share": free_fit.get("ai_contribution_share"),
                      "mmm_calibrated_ai_contribution_share":
                          cal_fit.get("ai_contribution_share") if cal_fit else None}

    report = build_report(br, free_fit, cal_fit, provenance, evaluation, variables)
    print(report)
    return {"weekly": tables, "bridge": br, "mmm": free_fit,
            "mmm_calibrated": cal_fit, "mmm_frame": frame,
            "ai_variable_comparison": variables,
            "ledger": ledger, "evaluation": evaluation,
            "provenance": provenance, "report": report}


def build_report(br: Dict[str, object], free_fit: Dict[str, object],
                 cal_fit: Optional[Dict[str, object]],
                 provenance: Dict[str, str],
                 evaluation: Dict[str, object],
                 variables: Optional[pd.DataFrame] = None) -> str:
    """The readout: the number, the caveat, and the next measurement to buy."""
    lines = ["", "=" * 72,
             "AI SALES ATTRIBUTION — panel: %(panel)s, market data: %(market)s"
             % provenance, "=" * 72, str(br["headline"]), ""]

    sens = br["sensitivity"]
    top = sens.head(3)
    lines.append("Uncertainty is owned by:")
    for _, r in top.iterrows():
        lines.append(f"  {r['factor']:<28} {r['variance_share']:>6.1%} of the "
                     f"interval   [{r['status']}]")
    worst = top.iloc[0]["factor"] if len(top) else None
    if worst:
        lines.append(f"  → measure `{worst}` first; nothing else moves the "
                     f"answer as much.")

    lines += ["", "MMM (reference implementation, not the production model):"]
    ai_share = free_fit.get("ai_contribution_share")
    if ai_share is not None:
        lines.append(f"  AI contribution share (free fit):        {ai_share:.1%}")
    if cal_fit is not None:
        lines.append(f"  AI contribution share (bridge-calibrated): "
                     f"{cal_fit.get('ai_contribution_share', float('nan')):.1%}")
    r2h = free_fit.get("r2_holdout")
    lines.append(f"  holdout R²: {r2h:.3f}" if r2h is not None
                 else "  holdout R²: n/a (series too short)")
    diag = free_fit.get("diagnostics")
    if diag is not None and not diag.empty:
        ai_row = diag[diag["feature"] == free_fit["config"].ai_col]
        if len(ai_row):
            vif = float(ai_row.iloc[0]["vif"])
            verdict = ("separable from the rest of the mix" if vif < 5 else
                       "hard to separate — read the coefficient with care"
                       if vif < 10 else
                       "NOT separable: do not quote this coefficient")
            lines.append(f"  VIF on the AI variable: {vif:.1f} — {verdict}")

    if variables is not None and not variables.empty:
        lines += ["", "Choice of AI regressor (same model, one variable swapped):"]
        for _, r in variables.iterrows():
            share = r["ai_contribution_share"]
            lines.append(f"  {r['ai_variable']:<32} "
                         f"{share:>7.2%}  VIF {r['vif']:>5.1f}  {r['verdict']}")

    if evaluation:
        t = evaluation.get("true_ai_contribution_share")
        e = evaluation.get("mmm_ai_contribution_share")
        if t is not None and e is not None:
            lines += ["", f"[synthetic check] planted AI contribution {t:.1%}, "
                          f"recovered {e:.1%}"]
    lines += ["", "Reminder: 'influenced' ≠ 'incremental'. The first is a "
                  "touchpoint claim the panel can support; the second is causal "
                  "and rests on the MMM.", "=" * 72, ""]
    return "\n".join(lines)
