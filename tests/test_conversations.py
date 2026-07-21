"""Tests for conveyer.conversations (module 1) + the shared brand lexicon.

Runs under pytest or standalone: ``python tests/test_conversations.py``.
Everything offline (lexicon sentiment, TF-IDF topics).
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

import pandas as pd

from conveyer.brands import brand_for_domain, extract_brands, normalize_brand
from conveyer.conversations import (ConversationConfig, classify_intent,
                                    lexicon_sentiment, run_conversations)
from conveyer.ingest import (as_records, load_conversations,
                             make_synthetic_conversations, trail_events)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# Brand lexicon
# --------------------------------------------------------------------------- #
def test_brand_extraction():
    got = extract_brands("Is CeraVe better than The Ordinary for acne? maybe la roche posay")
    _check(got == {"CeraVe", "The Ordinary", "La Roche-Posay"}, f"got {got}")
    # generic words must never read as brands
    _check(extract_brands("I love fresh air and simple routines and my hero") == set(),
           "generic words leaked as brands")
    _check(extract_brands("hero cosmetics mighty patch works") == {"Hero Cosmetics"},
           "qualified alias should match")
    _check(brand_for_domain("cerave") == "CeraVe", "domain -> canonical")
    _check(normalize_brand("CERAVE") == "CeraVe", "normalize")


# --------------------------------------------------------------------------- #
# Intent + sentiment
# --------------------------------------------------------------------------- #
def test_intent_rules():
    cases = {
        "cerave vs cetaphil which is better?": "comparison",
        "what do you recommend for acne": "purchase",
        "my skin got worse and is irritated": "troubleshooting",
        "build me a morning routine": "routine",
        "what is niacinamide": "informational",
    }
    for text, want in cases.items():
        got = classify_intent(text)
        _check(got == want, f"{text!r} -> {got} (want {want})")


def test_lexicon_sentiment():
    _check(lexicon_sentiment("I love it, my skin improved and looks amazing") > 0.5, "positive")
    _check(lexicon_sentiment("terrible, it burns and made my breakouts worse") < -0.5, "negative")
    _check(lexicon_sentiment("what is the weather") == 0.0, "neutral")
    _check(lexicon_sentiment("this is not good") < 0, "negation flips polarity")


# --------------------------------------------------------------------------- #
# Loading + synthetic
# --------------------------------------------------------------------------- #
def test_load_conversations_normalises():
    df = pd.DataFrame({
        "session_id": ["s1", "s1"], "message_id": ["m2", "m1"],
        "question": ["second?", "first?"], "answer": ["a2", "a1"],
        "prompt_datetime": ["2026-01-02T10:05:00", "2026-01-02T10:00:00"],
        "next_10_urls": ["[{'request_time': '1000', 'requested_site': 'https://a.com'}]", None],
    })
    out = load_conversations(df)
    _check(list(out["message_id"]) == ["m1", "m2"], "time-ordered within session")
    _check(list(out["session_pos"]) == [1, 2], "session_pos derived")
    _check(out["next_10_urls"].iloc[1][0]["requested_site"] == "https://a.com",
           "stringified trail parsed")
    _check(out["a_links_source"].apply(len).sum() == 0, "missing link col -> empty lists")


def test_synthetic_has_ground_truth_and_trails():
    df, html, gt = make_synthetic_conversations(20, seed=1)
    _check(df.session_id.nunique() == 20, "20 sessions")
    _check(set(gt.columns) >= {"session_id", "gt_archetype", "gt_followed_rec",
                               "gt_converted"}, "gt columns")
    _check(gt.gt_converted.any() and (~gt.gt_converted).any(), "both outcomes present")
    last = df[df.groupby("session_id")["session_pos"].transform("max") == df["session_pos"]]
    _check(all(len(t) > 0 for t in last["next_10_urls"]), "last turn carries the trail")
    some_trail = trail_events(last["next_10_urls"].iloc[0])
    _check(some_trail[0]["dwell_seconds"] is not None, "dwell computed")
    _check(any(u in html for u in [e["url"] for e in some_trail]),
           "trail pages exist in the offline corpus")


# --------------------------------------------------------------------------- #
# End-to-end module 1
# --------------------------------------------------------------------------- #
def test_run_conversations_end_to_end():
    out = tempfile.mkdtemp(prefix="conveyer_conv_")
    try:
        cfg = ConversationConfig(synthetic_n_sessions=16, out_dir=out)
        art = run_conversations(cfg)
        turns, convs = art["turns_out"], art["conversations"]
        _check(len(convs) == 16, "one row per session")
        _check(os.path.exists(cfg.turns_path()) and os.path.exists(cfg.conversations_path()),
               "parquets written")
        back = pd.read_parquet(cfg.turns_path())
        _check(len(back) == len(turns), "turn parquet round-trips")
        _check(turns["intent"].nunique() >= 2, "intent diversity")
        _check(turns["journey_stage"].notna().all(), "HMM journey stage assigned")
        _check((turns["brands_a"].apply(len) > 0).any(), "agent brand mentions extracted")
        _check(convs["n_unsolicited_recommendations"].sum() > 0, "unsolicited recs found")
        _check((convs["n_links_cited"] > 0).any(), "cited links counted")
        _check(convs["journey_path"].str.len().gt(0).all(), "journey paths built")
        _check("topic_label" in convs.columns, "topics present")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_as_records_shapes():
    import numpy as np

    recs = [{"request_time": "1", "requested_site": "https://a.com"}]
    _check(as_records(np.array(recs, dtype=object)) == recs, "ndarray")
    _check(as_records(None) == [] and as_records(float("nan")) == [], "nulls")
    _check(as_records(str(recs)) == recs, "repr string")


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
