"""Tests for conveyer.journey (module 3) + the end-to-end pipeline.

Runs under pytest or standalone: ``python tests/test_journey.py``. Fully
offline: synthetic conversations → offline scrape → funnel model, validated
against the generator's ground truth.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import pandas as pd

from conveyer import run_pipeline
from conveyer.conversations import ConversationConfig
from conveyer.journey import JourneyConfig, _commerce_depth, _jeffreys_ci
from conveyer.scraping import ScrapeConfig


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _run(tmp, sessions=40):
    return run_pipeline(
        ConversationConfig(synthetic_n_sessions=sessions,
                           out_dir=os.path.join(tmp, "conv")),
        ScrapeConfig(out_dir=os.path.join(tmp, "scrape"), progress_every=0),
        JourneyConfig(out_dir=os.path.join(tmp, "journey")),
    )


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
def test_commerce_depth():
    _check(_commerce_depth("shopping", "checkout") == 3, "checkout deepest")
    _check(_commerce_depth("shopping", "cart") == 2, "cart")
    _check(_commerce_depth("shopping", "pdp") == 1, "pdp")
    _check(_commerce_depth("editorial", "article") == 0, "non-commerce")


def test_jeffreys_ci():
    lo, hi = _jeffreys_ci(5, 10)
    _check(0.15 < lo < 0.5 < hi < 0.85, f"CI around 0.5: ({lo}, {hi})")
    _check(_jeffreys_ci(0, 0) == (0.0, 1.0), "empty stratum -> vacuous CI")


# --------------------------------------------------------------------------- #
# End-to-end (the deliverable)
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end_recovers_ground_truth():
    tmp = tempfile.mkdtemp(prefix="conveyer_e2e_")
    try:
        art = _run(tmp)
        j, ev = art["journey"], art["evaluation"]

        # ground truth recovered: the conversion proxy and exposure detection
        _check(ev["conversion_accuracy"] >= 0.95, f"conversion acc {ev}")
        _check(ev["followed_accuracy"] >= 0.9, f"followed acc {ev}")

        # agent-recommendation exposure must lift conversion in this world
        lift = j["report"].attrs["lift"]
        _check(lift.get("followed_recommendation", 0) > 1.2, f"lift {lift}")

        # the models exist and generalise
        model = j["model"]
        _check(model and model["auc_test"] is not None and model["auc_test"] >= 0.7,
               f"AUC {model.get('auc_test') if model else None}")
        full = model["full"]["coefficients"].set_index("feature")["coef_standardized"]
        _check(full["n_visits"] > 0, "visit volume should predict conversion (full model)")
        # mediator-free exposure model: agent-link exposure must carry positive
        # weight in a world where following recommendations drives conversion
        expo = model["exposure"]["coefficients"].set_index("feature")["coef_standardized"]
        _check(expo["followed_agent_link"] > 0,
               f"exposure weight should be positive, got {expo['followed_agent_link']}")
        _check("model" in model["coefficients"].columns, "long-form coefficients")

        # events: full page coverage (scraped or URL-only), grammar respected
        events = j["events"]
        _check(set(events["event_type"]) <= {"cited", "ai_click", "trail_visit"}, "event types")
        _check((events[events["event_type"] == "cited"]["followed_agent_link"] == False).all(),  # noqa: E712
               "cited rows are exposure, not behaviour")
        _check(events["page_category"].notna().all(), "every event classified")

        # all four parquets written and readable
        for f in ("funnel_events", "journey_features", "funnel_transitions",
                  "model_coefficients"):
            path = os.path.join(tmp, "journey", f + ".parquet")
            _check(os.path.exists(path), f"{f} missing")
            _check(len(pd.read_parquet(path)) > 0, f"{f} empty")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transitions_are_probabilities():
    tmp = tempfile.mkdtemp(prefix="conveyer_trans_")
    try:
        art = _run(tmp, sessions=24)
        tr = art["journey"]["transitions"]
        _check(not tr.empty, "transitions built")
        sums = tr.groupby("from_stage")["prob"].sum()
        _check(((sums - 1.0).abs() < 0.01).all(), f"rows must sum to 1: {sums.to_dict()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_conversion_stage_knob():
    tmp = tempfile.mkdtemp(prefix="conveyer_stage_")
    try:
        art = run_pipeline(
            ConversationConfig(synthetic_n_sessions=24, out_dir=os.path.join(tmp, "c")),
            ScrapeConfig(out_dir=os.path.join(tmp, "s"), progress_every=0),
            JourneyConfig(out_dir=os.path.join(tmp, "j"), conversion_stage="shopping"),
        )
        f = art["journey"]["features"]
        # shallower bar -> at least as many conversions as the cart definition
        _check(int(f["converted"].sum()) >= int(f["reached_cart"].sum()),
               "shopping-depth conversions must dominate cart-depth")
        _check((f["converted"] == f["reached_shop"]).all(), "converted == reached_shop here")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
