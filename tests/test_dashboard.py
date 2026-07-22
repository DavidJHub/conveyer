"""Tests for conveyer.dashboard.

Runs under pytest or standalone: ``python tests/test_dashboard.py``. Offline.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from conveyer import run_pipeline
from conveyer.conversations import ConversationConfig
from conveyer.dashboard import (DashboardConfig, brand_leaderboard,
                                build_dashboard, from_output_dirs,
                                from_pipeline_art)
from conveyer.journey import JourneyConfig
from conveyer.scraping import ScrapeConfig


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _pipeline(tmp):
    return run_pipeline(
        ConversationConfig(synthetic_n_sessions=24, out_dir=os.path.join(tmp, "conversations")),
        ScrapeConfig(out_dir=os.path.join(tmp, "scrape"), progress_every=0),
        JourneyConfig(out_dir=os.path.join(tmp, "journey")),
    )


def test_dashboard_from_pipeline_and_from_outputs():
    tmp = tempfile.mkdtemp(prefix="conveyer_dash_")
    try:
        art = _pipeline(tmp)

        # from the live artifact dict
        p1 = build_dashboard(from_pipeline_art(art),
                             DashboardConfig(out_path=os.path.join(tmp, "d1.html")))
        html = open(p1, encoding="utf-8").read()
        _check(len(html) > 50_000, "self-contained page with inline charts")
        _check(html.count("data:image/png;base64,") >= 6, "charts embedded")
        for marker in ("sessions", "lift when following the agent",
                       "Recommended-brand leaderboard", "Journey shape",
                       "documented <b>proxy</b>"):
            _check(marker in html, f"missing section: {marker}")
        _check("<script" not in html.lower(), "no scripts — static and CSP-safe")

        # from saved parquet outputs, without re-running the pipeline
        p2 = build_dashboard(from_output_dirs(tmp),
                             DashboardConfig(out_path=os.path.join(tmp, "d2.html")))
        html2 = open(p2, encoding="utf-8").read()
        _check(html2.count("data:image/png;base64,") >= 6, "outputs-mode charts embedded")
        _check('alt="exposure model weights"' in html2,
               "exposure-model chart rebuilt from model_coefficients.parquet")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_brand_leaderboard_grammar():
    tmp = tempfile.mkdtemp(prefix="conveyer_dash_lb_")
    try:
        art = _pipeline(tmp)
        lb = brand_leaderboard(from_pipeline_art(art), top=10)
        _check(len(lb) > 0, "leaderboard has rows")
        _check(set(lb.columns) == {"brand", "sessions_recommended",
                                   "sessions_visited_brand", "sessions_followed_rec",
                                   "conversion_rate_when_recommended"}, f"cols {list(lb.columns)}")
        _check((lb["sessions_followed_rec"] <= lb["sessions_visited_brand"]).all(),
               "followers are a subset of visitors")
        _check(lb["conversion_rate_when_recommended"].between(0, 1).all(), "rates in [0,1]")
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
