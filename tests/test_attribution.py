"""Tests for MODULE 4 — the sales-attribution layer. Offline, no pytest needed.

    python tests/test_attribution.py

What is worth asserting here is not "the number is right" — the number rests on
declared assumptions — but that the *machinery* is honest: placeholders stay
visible, the two estimation routes are computed independently, projections are
not invented when the frame is missing, rates cannot exceed 1, and the MMM
tells you when its own coefficient is uninterpretable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conveyer.attribution import bridge as B          # noqa: E402
from conveyer.attribution import mmm as M             # noqa: E402
from conveyer.attribution.aggregate import (AggregateConfig, build_weekly,  # noqa: E402
                                            session_brand_relations)
from conveyer.attribution.calibration import Factor, Ledger, default_ledger  # noqa: E402
from conveyer.attribution.demo import DemoSpec, make_market  # noqa: E402
from conveyer.attribution.panel import (PanelFrame, design_effect,  # noqa: E402
                                        effective_n, post_stratification_weights,
                                        project_counts)
from conveyer.attribution.run import AttributionConfig, run_attribution  # noqa: E402

PASSED, FAILED = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  ok   {name}")
        except AssertionError as e:
            FAILED.append((name, str(e)))
            print(f"  FAIL {name}: {e}")
        except Exception as e:                                  # noqa: BLE001
            FAILED.append((name, repr(e)))
            print(f"  ERR  {name}: {e!r}")
        return fn
    return deco


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@check("ledger draws stay inside every declared range")
def _t1():
    led = default_ledger()
    d = led.draw(500, seed=1)
    for name in led.names:
        f = led[name]
        assert d[name].min() >= f.low - 1e-9, f"{name} below low"
        assert d[name].max() <= f.high + 1e-9, f"{name} above high"


@check("placeholders are named in the headline and never hidden")
def _t2():
    led = default_ledger()
    ph = led.placeholders()
    assert ph, "the default ledger must admit it contains placeholders"
    head = led.headline()
    assert "PLACEHOLDER" in head
    for name in ph:
        assert name in head, f"{name} missing from the headline"
    prov = led.provenance()
    assert set(prov["status"]) <= {"measured", "estimated", "benchmark",
                                   "assumed", "placeholder"}


@check("promoting a factor removes it from the placeholder list")
def _t3():
    led = default_ledger()
    led.set("ropo_multiplier", value=2.4, low=2.1, high=2.8,
            status="measured", source="CPS path-to-purchase 2026")
    assert "ropo_multiplier" not in led.placeholders()
    assert led.value("ropo_multiplier") == 2.4


@check("a factor whose value sits outside its range is rejected")
def _t4():
    try:
        Factor("bad", value=2.0, low=0.1, high=1.0)
    except ValueError:
        return
    raise AssertionError("out-of-range factor was accepted")


@check("ledger round-trips through JSON and overlays partial overrides")
def _t5():
    import json
    import tempfile

    led = default_ledger()
    with tempfile.TemporaryDirectory() as d:
        p = led.save(os.path.join(d, "ledger.json"))
        back = Ledger.load(p)
        assert back.names == led.names
        assert back.value("cart_to_order") == led.value("cart_to_order")

        partial = os.path.join(d, "override.json")
        with open(partial, "w", encoding="utf-8") as fh:
            json.dump([{"name": "cart_to_order", "value": 0.29, "low": 0.26,
                        "high": 0.33, "status": "measured",
                        "source": "own shop"}], fh)
        ov = Ledger.load(partial)
        assert ov.value("cart_to_order") == 0.29
        assert len(ov) == len(led), "a partial override must not drop factors"
        assert "cart_to_order" not in ov.placeholders()


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
@check("projection scales counts and widens with the design effect")
def _t6():
    frame = PanelFrame(panel_users=1_000, universe_users=1_000_000)
    df = pd.DataFrame({"n": [50], "base": [200]})
    tight = project_counts(df, ["n"], frame, base_col="base", deff=1.0)
    wide = project_counts(df, ["n"], frame, base_col="base", deff=4.0)
    assert tight["n_projected"].iloc[0] == 50_000
    w_t = tight["n_projected_hi"].iloc[0] - tight["n_projected_lo"].iloc[0]
    w_w = wide["n_projected_hi"].iloc[0] - wide["n_projected_lo"].iloc[0]
    assert w_w > w_t, "a larger design effect must widen the interval"


@check("no interval is invented when there is no denominator")
def _t7():
    frame = PanelFrame(panel_users=1_000, universe_users=1_000_000)
    out = project_counts(pd.DataFrame({"n": [50]}), ["n"], frame)
    assert "n_projected" in out.columns
    assert "n_projected_lo" not in out.columns


@check("Kish effective n and design effect behave")
def _t8():
    assert abs(effective_n([1, 1, 1, 1]) - 4.0) < 1e-9
    assert design_effect([1, 1, 1, 1]) == 1.0
    assert effective_n([1, 1, 1, 9]) < 4.0
    assert design_effect([1, 1, 1, 9]) > 1.0


@check("no demographic frame ⇒ no weights, stated rather than assumed")
def _t9():
    assert post_stratification_weights(PanelFrame()) is None
    demo = pd.DataFrame({"segment": ["a", "b"], "panel_share": [0.7, 0.3],
                         "universe_share": [0.5, 0.5]})
    w = post_stratification_weights(PanelFrame(demo_frame=demo,
                                               demo_status="post_stratified"))
    assert w is not None and (w["weight"] > 0).all()
    assert w.loc[w["segment"] == "b", "weight"].iloc[0] > \
        w.loc[w["segment"] == "a", "weight"].iloc[0], \
        "the under-represented segment must be weighted up"


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _tiny_run():
    turns = pd.DataFrame([
        {"session_id": "s1", "message_id": "s1m0", "user_id": "u1",
         "prompt_datetime": "2026-03-02T10:00:00", "brands_a": ["Nivea"],
         "brands_unsolicited": ["Nivea"]},
        {"session_id": "s2", "message_id": "s2m0", "user_id": "u2",
         "prompt_datetime": "2026-03-03T10:00:00", "brands_a": ["CeraVe", "Nivea"],
         "brands_unsolicited": ["CeraVe"]},
    ])
    events = pd.DataFrame([
        {"session_id": "s1", "event_type": "trail_visit", "page_brand": "Nivea",
         "followed_agent_link": True, "commerce_depth": 2, "position": 1},
        {"session_id": "s2", "event_type": "trail_visit", "page_brand": "CeraVe",
         "followed_agent_link": False, "commerce_depth": 1, "position": 1},
        {"session_id": "s2", "event_type": "trail_visit", "page_brand": "Nivea",
         "followed_agent_link": False, "commerce_depth": 3, "position": 2},
    ])
    features = pd.DataFrame([{"session_id": "s1", "converted": True,
                              "followed_recommendation": True},
                             {"session_id": "s2", "converted": True,
                              "followed_recommendation": False}])
    return turns, events, features


@check("a session's conversion is attributed to exactly one brand")
def _t10():
    turns, events, features = _tiny_run()
    rel = session_brand_relations(turns, events, features)
    per_session = rel.groupby("session_id")["converted"].sum()
    assert (per_session <= 1).all(), "conversions were double counted across brands"
    s2 = rel[(rel["session_id"] == "s2") & rel["converted"]]
    assert s2["brand"].tolist() == ["Nivea"], \
        "the conversion must land on the deepest-commerce brand"


@check("weekly aggregation carries the three new dimensions and sane rates")
def _t11():
    turns, events, features = _tiny_run()
    tables = build_weekly(turns, events, features,
                          AggregateConfig(market="DE", category="body care",
                                          llm_platform="gemini"))
    bw = tables["brand_weekly"]
    assert {"week", "market", "category", "llm_platform"} <= set(bw.columns)
    assert set(bw["market"]) == {"DE"} and set(bw["llm_platform"]) == {"gemini"}
    assert bw["visit_rate_given_exposure"].max() <= 1.0, "rate exceeded 1"
    for wk, g in bw.groupby(["week", "market", "category", "llm_platform"]):
        assert abs(g["ai_share_of_voice"].sum() - 1.0) < 1e-6, \
            f"share of voice must sum to 1 within {wk}"


@check("projection only happens when a panel frame is supplied")
def _t12():
    turns, events, features = _tiny_run()
    plain = build_weekly(turns, events, features)["brand_weekly"]
    assert "n_sessions_mentioned_projected" not in plain.columns
    proj = build_weekly(turns, events, features, None,
                        {"US": PanelFrame(panel_users=10, universe_users=1_000)}
                        )["brand_weekly"]
    assert proj["n_sessions_mentioned_projected"].sum() > 0


# --------------------------------------------------------------------------- #
# the bridge
# --------------------------------------------------------------------------- #
@check("the bridge refuses to guess when the join key cannot be met")
def _t13():
    d = make_market(DemoSpec(n_weeks=8))
    bad = d["sales"].rename(columns={"brand": "brand_name"})
    try:
        B.estimate_influenced_sales(d["brand_weekly"], bad)
    except ValueError as e:
        assert "brand" in str(e)
        return
    raise AssertionError("a sales table with no brand column was accepted")


@check("exposure collapses over platform because sales cannot carry it")
def _t14():
    d = make_market(DemoSpec(n_weeks=8))
    key = B.join_key(d["sales"])
    assert "llm_platform" not in key
    col = B.collapse_exposure(d["brand_weekly"], key)
    assert set(col["llm_platform"]) == {"all"}
    before = d["brand_weekly"]["n_sessions_mentioned"].sum()
    assert col["n_sessions_mentioned"].sum() == before, "counts lost in collapse"


@check("bottom-up and top-down are genuinely different computations")
def _t15():
    d = make_market(DemoSpec(n_weeks=26))
    res = B.estimate_influenced_sales(d["brand_weekly"], d["sales"],
                                      cfg=B.BridgeConfig(n_draws=300),
                                      category_weekly=d["category_weekly"])
    cells = res["cells"]
    assert cells["influenced_share_p50"].notna().all()
    assert cells["topdown_share_p50"].notna().all()
    assert not np.allclose(cells["influenced_share_p50"],
                           cells["topdown_share_p50"]), \
        "the cross-check must not be a restatement of the main estimate"
    assert (cells["influenced_share_p05"] <= cells["influenced_share_p50"]).all()
    assert (cells["influenced_share_p50"] <= cells["influenced_share_p95"]).all()


@check("top-down is skipped, not faked, without category reach")
def _t16():
    d = make_market(DemoSpec(n_weeks=8))
    res = B.estimate_influenced_sales(d["brand_weekly"], d["sales"],
                                      cfg=B.BridgeConfig(n_draws=100))
    assert res["cells"]["topdown_share_p50"].isna().all()
    assert "not computed" in res["headline"]


@check("incremental is always a fraction of influenced")
def _t17():
    d = make_market(DemoSpec(n_weeks=12))
    res = B.estimate_influenced_sales(d["brand_weekly"], d["sales"],
                                      cfg=B.BridgeConfig(n_draws=300))
    c = res["cells"]
    assert (c["incremental_value_p50"] <= c["influenced_value_p50"] + 1e-6).all()


@check("the tornado ranks the placeholders above the panel's sampling error")
def _t18():
    d = make_market(DemoSpec(n_weeks=26))
    res = B.estimate_influenced_sales(d["brand_weekly"], d["sales"],
                                      cfg=B.BridgeConfig(n_draws=400))
    s = res["sensitivity"]
    assert abs(s["variance_share"].sum() - 1.0) < 1e-3, s["variance_share"].sum()
    top = s.iloc[0]
    assert top["status"] == "placeholder", \
        f"expected an assumption to dominate, got {top['factor']}"
    rank = {r["factor"]: i for i, r in s.iterrows()}
    assert rank["panel_sampling"] > 0, "panel sampling should not dominate"


@check("a tighter range on the top factor tightens the answer")
def _t19():
    d = make_market(DemoSpec(n_weeks=12))
    loose = B.estimate_influenced_sales(d["brand_weekly"], d["sales"],
                                        cfg=B.BridgeConfig(n_draws=400))
    led = default_ledger().set("ropo_multiplier", value=3.2, low=3.0, high=3.4,
                               status="measured", source="test")
    tight = B.estimate_influenced_sales(d["brand_weekly"], d["sales"], led,
                                        B.BridgeConfig(n_draws=400))

    def width(res):
        t = res["totals"].set_index("metric")
        return float(t.loc["ai_influenced_share", "p95"]
                     - t.loc["ai_influenced_share", "p05"])
    assert width(tight) < width(loose), "measuring a factor must narrow the CI"


@check("panel lift gives an incrementality upper bound, not an estimate")
def _t20():
    feats = pd.DataFrame({
        "followed_recommendation": [True] * 20 + [False] * 20,
        "converted": [True] * 10 + [False] * 10 + [True] * 4 + [False] * 16,
    })
    out = B.incrementality_from_lift(feats)
    assert out is not None
    assert abs(out["lift"] - (0.5 / 0.2)) < 1e-6
    assert 0 < out["incrementality_upper_bound"] < 1


# --------------------------------------------------------------------------- #
# MMM
# --------------------------------------------------------------------------- #
@check("adstock carries over and saturation compresses")
def _t21():
    a = M.geometric_adstock([1, 0, 0, 0], 0.5)
    assert np.allclose(a, [1, 0.5, 0.25, 0.125])
    s = M.hill_saturation([0, 1, 10, 1000], half=10)
    assert s[0] == 0 and 0 < s[1] < s[2] < s[3] < 1


@check("adstock never leaks across brand series")
def _t22():
    frame = pd.DataFrame({
        "week": ["w1", "w2", "w1", "w2"], "market": ["US"] * 4,
        "brand": ["a", "a", "b", "b"], "week_index": [0, 1, 0, 1],
        "tv_spend": [100.0, 0.0, 0.0, 0.0], "sales_value": [1.0, 1.0, 1.0, 1.0],
        "ai_share_of_voice": [0.5, 0.5, 0.5, 0.5]})
    cfg = M.MMMConfig(media_cols=["tv_spend"], saturate_media=False,
                      brand_fixed_effects=False, seasonality_harmonics=0,
                      include_trend=False)
    X, _, names, _ = M.build_matrix(frame, cfg)
    tv = X[:, names.index("tv_spend_adstock")]
    assert tv[2] == 0 and tv[3] == 0, "brand a's spend leaked into brand b"


@check("weeks with no AI conversations become zeros, not dropped rows")
def _t23():
    d = make_market(DemoSpec(n_weeks=10))
    bw = d["brand_weekly"]
    trimmed = bw[bw["week"] != sorted(bw["week"].unique())[0]]
    frame = M.design_frame(trimmed, d["sales"], d["media"])
    assert len(frame) == len(d["sales"]), "sales rows were dropped by the join"
    assert frame["ai_share_of_voice"].isna().sum() == 0


@check("the MMM recovers a planted AI contribution with the right sign")
def _t24():
    d = make_market(DemoSpec(n_weeks=104))
    frame = M.design_frame(d["brand_weekly"], d["sales"], d["media"])
    cfg = M.MMMConfig(media_cols=[c for c in d["media"].columns
                                  if c.endswith("_spend")],
                      control_cols=["price_index"])
    fit = M.fit(frame, cfg)
    truth = d["ground_truth"]["true_ai_contribution_share"]
    est = fit["ai_contribution_share"]
    assert est > 0, "the AI contribution came back with the wrong sign"
    # attenuated by construction: share-of-voice discards the adoption signal
    assert 0.2 * truth < est < 2.5 * truth, f"{est:.4f} vs planted {truth}"
    assert fit["r2_holdout"] > 0.5


@check("the volume regressor is diagnosed as collinear, the share one is not")
def _t25():
    d = make_market(DemoSpec(n_weeks=104))
    frame = M.design_frame(d["brand_weekly"], d["sales"], d["media"])
    cfg = M.MMMConfig(media_cols=[c for c in d["media"].columns
                                  if c.endswith("_spend")])
    cmp = M.compare_ai_variables(frame, cfg).set_index("ai_variable")
    assert cmp.loc["ai_share_of_voice", "vif"] < \
        cmp.loc["n_sessions_mentioned_projected", "vif"], \
        "the volume variable should be the more collinear of the two"
    assert cmp.loc["ai_share_of_voice", "coef_sign"] == "+"


@check("the bridge prior moves the AI coefficient toward the panel estimate")
def _t26():
    d = make_market(DemoSpec(n_weeks=104))
    frame = M.design_frame(d["brand_weekly"], d["sales"], d["media"])
    cfg = M.MMMConfig(media_cols=[c for c in d["media"].columns
                                  if c.endswith("_spend")])
    br = B.estimate_influenced_sales(d["brand_weekly"], d["sales"],
                                     cfg=B.BridgeConfig(n_draws=300))
    prior = M.prior_from_bridge(br, frame, cfg, strength=50.0)
    assert prior, "no prior derived from the bridge"
    free = M.fit(frame, cfg)["ai_coefficient"]
    cal = M.fit(frame, cfg, **prior)["ai_coefficient"]
    assert abs(cal - prior["prior_mean"]) < abs(free - prior["prior_mean"]), \
        "calibration did not pull the coefficient toward the prior"


@check("two-stage splits a driver into direct and AI-mediated effects")
def _t27():
    d = make_market(DemoSpec(n_weeks=52))
    frame = M.design_frame(d["brand_weekly"], d["sales"], d["media"])
    cfg = M.MMMConfig(media_cols=[c for c in d["media"].columns
                                  if c.endswith("_spend")])
    out = M.two_stage(frame, cfg)
    assert not out["mediation"].empty
    assert {"direct_coef", "indirect_via_ai"} <= set(out["mediation"].columns)
    assert out["stage1_r2"] is not None


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@check("the whole layer runs offline and scores itself")
def _t28():
    res = run_attribution(cfg=AttributionConfig(write=False, n_draws=300),
                          demo_spec=DemoSpec(n_weeks=78))
    assert res["provenance"]["panel"] == "synthetic"
    assert "PLACEHOLDER" in res["report"]
    assert "influenced" in res["report"] and "incremental" in res["report"]
    ev = res["evaluation"]
    assert ev["mmm_ai_contribution_share"] is not None
    t = res["bridge"]["totals"].set_index("metric")
    inf = t.loc["ai_influenced_share", "p50"]
    inc = t.loc["ai_incremental_share", "p50"]
    assert 0 < inc < inf, "incremental must sit strictly below influenced"


@check("a real panel run drives the layer with fabricated market data only")
def _t29():
    from conveyer.ingest import make_synthetic_conversations
    from conveyer.conversations import ConversationConfig, run_conversations

    conv = run_conversations(ConversationConfig(synthetic_n_sessions=24,
                                                out_dir="/tmp/conveyer_attr_test"))
    res = run_attribution(turns=conv["turns"],
                          features=None, events=None,
                          cfg=AttributionConfig(write=False, n_draws=200))
    assert res["provenance"]["panel"] == "real"
    assert res["provenance"]["market"] == "synthetic"
    assert "ASSUMED DEFAULT" in res["provenance"].get("panel_frame", "")
    assert len(res["bridge"]["cells"]) > 0


if __name__ == "__main__":
    print("attribution tests")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name, err in FAILED:
        print(f"  - {name}: {err}")
    sys.exit(1 if FAILED else 0)
