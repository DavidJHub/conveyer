"""End-to-end orchestration: conversations → pages → funnel model.

Run as a module::

    python -m conveyer.pipeline                          # synthetic, offline
    python -m conveyer.pipeline --data data/conversations.parquet --online

or programmatically::

    from conveyer import run_pipeline
    art = run_pipeline()
    art["journey"]["report"]           # conversion by exposure, with CIs
    art["journey"]["model"]["coefficients"]   # weight of agent recommendations

The three modules stay independently usable; this wires them: module 1's turn
table supplies both the conversation features and the candidate URLs + brand
mentions for module 2 (the ``a_links_source`` / ``ai_click`` / ``next_10_urls``
columns are exactly the input-file shape the scraper consumes), and module 3
joins both outputs into the funnel tables and the conversion model. On
synthetic data the run scores itself against ground truth at the end.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import pandas as pd

from .brands import normalize_brand
from .conversations import ConversationConfig, run_conversations
from .journey import JourneyConfig, evaluate_against_gt, run_journey
from .scraping import ScrapeConfig, run_scrape
from .scraping.products import RecMention
from .scraping.sources import ScrapeSources, candidate_urls


def _mentions_from_turns(turns: pd.DataFrame) -> Dict[str, List[RecMention]]:
    """Turn module 1's brand mentions into the scraper's mention records so
    product↔chat matching works from conversation text alone."""
    out: Dict[str, List[RecMention]] = {}
    for _, t in turns.iterrows():
        ms = [RecMention(recommendation_id=f"{t['message_id']}:{b}",
                         message_id=str(t["message_id"]),
                         entity_context=f"{b} — recommended in the conversation",
                         brands={normalize_brand(b).lower()})
              for b in (t.get("brands_a") or [])]
        if ms:
            out[str(t["message_id"])] = ms
    return out


def run_pipeline(conv_cfg: Optional[ConversationConfig] = None,
                 scrape_cfg: Optional[ScrapeConfig] = None,
                 journey_cfg: Optional[JourneyConfig] = None) -> Dict[str, object]:
    conv_cfg = conv_cfg or ConversationConfig()
    scrape_cfg = scrape_cfg or ScrapeConfig()
    journey_cfg = journey_cfg or JourneyConfig()

    # 1 · conversations ---------------------------------------------------- #
    conv = run_conversations(conv_cfg)
    turns = conv["turns"]

    # 2 · pages — candidate URLs + mentions come straight from the turns --- #
    urls = candidate_urls({"input_file": turns}, scrape_cfg)
    src = ScrapeSources(urls=urls, mentions=_mentions_from_turns(turns),
                        html_by_url=conv.get("html_by_url"),
                        source="conversation link columns")
    scrape = run_scrape(scrape_cfg, sources=src)

    # 3 · the funnel -------------------------------------------------------- #
    journey = run_journey(turns, conv["conversations"], scrape["pages"],
                          scrape["products"], journey_cfg)

    gt = conv.get("ground_truth")
    evaluation = evaluate_against_gt(journey["features"], gt) if gt is not None else {}
    if evaluation:
        print("[pipeline eval vs ground truth]",
              {k: round(v, 3) for k, v in evaluation.items()})

    return {"conversations": conv, "scrape": scrape, "journey": journey,
            "evaluation": evaluation}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="conveyer end-to-end funnel pipeline")
    p.add_argument("--data", default="data/conversations.parquet",
                   help="conversations parquet (synthetic fallback when absent)")
    p.add_argument("--online", action="store_true", help="fetch pages over the network")
    p.add_argument("--max-urls", type=int, default=None)
    p.add_argument("--conversion-stage", default="cart",
                   choices=["shopping", "cart", "checkout"])
    p.add_argument("--sessions", type=int, default=40, help="synthetic session count")
    p.add_argument("--dashboard", action="store_true",
                   help="also render outputs/dashboard.html from this run")
    a = p.parse_args(argv)
    cfgs = (ConversationConfig(data_path=a.data, synthetic_n_sessions=a.sessions),
            ScrapeConfig(offline=not a.online, max_urls=a.max_urls),
            JourneyConfig(conversion_stage=a.conversion_stage))
    return cfgs, a.dashboard


if __name__ == "__main__":
    _cfgs, _dash = _parse_args()
    _art = run_pipeline(*_cfgs)
    if _dash:
        from .dashboard import DashboardConfig, build_dashboard, from_pipeline_art

        build_dashboard(from_pipeline_art(_art),
                        DashboardConfig(conversion_stage=_cfgs[2].conversion_stage))
