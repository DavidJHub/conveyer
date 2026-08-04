"""MODULE 4 — sales attribution: from panel journeys to euros, and into the MMM.

Modules 1–3 stop at a behavioural proxy in a desktop panel ("this session
reached a cart"). The business question is a market quantity ("what share of
our sales does the AI channel influence?"). This package is the missing span,
built as four separable pieces so each can be replaced by better data without
touching the others:

``calibration``  the assumption ledger — every unobservable factor declared
                 with a range, a source and an honesty status. Placeholders
                 are labelled as such and reported in every output.
``panel``        panel → universe projection, weighting, effective sample size.
``aggregate``    session grain → week × market × category × platform × brand,
                 the grain sales data and MMMs actually use.
``bridge``       the chain to euros, computed two independent ways
                 (bottom-up and top-down), Monte-Carlo'd over the ledger, with
                 a tornado saying which unknown owns the uncertainty.
``mmm``          the AI variable inside a marketing-mix model: share-of-voice
                 regressor, adstock/saturation, collinearity diagnostics,
                 a prior calibrated from the bridge, and a two-stage
                 media → AI-visibility → sales decomposition.
``demo``         synthetic market data with a planted AI contribution, so all
                 of the above runs offline and the MMM can be scored.

Two claims are kept strictly apart throughout: **influenced** (the AI
conversation was in the journey — measurable, big) and **incremental** (the
sale would not have happened otherwise — causal, smaller, and the MMM's job).

    python -m conveyer.attribution            # synthetic end-to-end + report
"""

from .aggregate import AggregateConfig, build_weekly, write_weekly
from .bridge import (BridgeConfig, estimate_influenced_sales,
                     incrementality_from_lift, write_bridge)
from .calibration import Factor, Ledger, default_ledger
from .mmm import MMMConfig, design_frame, fit, prior_from_bridge, two_stage
from .panel import PanelFrame, effective_n, project_counts
from .run import AttributionConfig, run_attribution

__all__ = [
    "AttributionConfig", "run_attribution",
    "Ledger", "Factor", "default_ledger",
    "PanelFrame", "project_counts", "effective_n",
    "AggregateConfig", "build_weekly", "write_weekly",
    "BridgeConfig", "estimate_influenced_sales", "incrementality_from_lift",
    "write_bridge",
    "MMMConfig", "design_frame", "fit", "prior_from_bridge", "two_stage",
]
