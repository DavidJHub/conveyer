"""CLI: ``python -m conveyer.attribution`` — the sales-attribution layer."""

from __future__ import annotations

import argparse

import pandas as pd

from .calibration import Ledger, default_ledger
from .demo import DemoSpec
from .run import AttributionConfig, run_attribution


def _read(path):
    if not path:
        return None
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported input: {path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="AI sales attribution: panel journeys → influenced sales → MMM")
    p.add_argument("--journey-dir", default=None,
                   help="outputs/journey from a pipeline run (funnel_events + "
                        "journey_features); synthetic market data when absent")
    p.add_argument("--turns", default=None,
                   help="outputs/conversations/turn_features.parquet")
    p.add_argument("--sales", default=None,
                   help="brand-week sales: week, market, category, brand, sales_value")
    p.add_argument("--media", default=None,
                   help="brand-week media spend, columns ending in _spend")
    p.add_argument("--ledger", default=None,
                   help="JSON overrides for the assumption ledger")
    p.add_argument("--dump-ledger", default=None,
                   help="write the default ledger to this path and exit")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--weeks", type=int, default=104, help="synthetic weeks")
    p.add_argument("--out", default="outputs/attribution")
    p.add_argument("--no-write", action="store_true")
    a = p.parse_args(argv)

    if a.dump_ledger:
        path = default_ledger().save(a.dump_ledger)
        print(f"[attribution] default ledger written to {path} — edit values, "
              f"statuses and sources, then pass it with --ledger")
        return

    turns = _read(a.turns)
    events = features = None
    if a.journey_dir:
        import os
        ev = os.path.join(a.journey_dir, "funnel_events.parquet")
        jf = os.path.join(a.journey_dir, "journey_features.parquet")
        events = _read(ev) if os.path.exists(ev) else None
        features = _read(jf) if os.path.exists(jf) else None

    run_attribution(
        turns=turns, events=events, features=features,
        sales=_read(a.sales), media=_read(a.media),
        ledger=Ledger.load(a.ledger) if a.ledger else None,
        cfg=AttributionConfig(out_dir=a.out, n_draws=a.draws, write=not a.no_write),
        demo_spec=DemoSpec(n_weeks=a.weeks),
    )


if __name__ == "__main__":
    main()
