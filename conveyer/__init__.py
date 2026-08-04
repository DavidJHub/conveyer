"""conveyer — from ChatGPT shopping conversations to a predictive funnel model.

Input: one parquet of conversations (session_id, message_id, user_id,
prompt_datetime, question, answer, a_links_source, ai_click, next_10_urls).

Three modules, each writing parquet outputs with pinned schemas:

    conversations  MODULE 1 — turn + session features: intent, sentiment,
                   topics, brand mentions (the agent-recommendation exposure),
                   funnel stages (keyword classifier + HMM smoothing)
    scraping       MODULE 2 — fetch + classify every surfaced/visited URL
                   (multimodal: page content → base-URL fallback → URL/domain
                   heuristics), extract products, match them to the chat
    journey        MODULE 3 — connect both: behavioural events, conversion
                   rate (documented cart/checkout proxy) with credible
                   intervals, funnel transitions, and the logistic model whose
                   standardized coefficients weigh agent recommendations
                   against every other driver
    attribution    MODULE 4 — panel behaviour → market sales: weekly exposure
                   facts, panel projection, the euro bridge over a declared
                   assumption ledger, and the AI variable inside a
                   marketing-mix model (docs/SALES_ATTRIBUTION.md)

Support: ingest (loading + trail parsing + synthetic ground truth), brands
(the canonical lexicon both text and domains resolve to), funnel (stage
taxonomy + HMM), models (auto-resolved embedding/sentiment/LLM backends).

    from conveyer import run_pipeline
    art = run_pipeline()                       # synthetic + offline by default
"""
from . import (attribution, brands, conversations, funnel, ingest, journey,
               models, scraping)
from .attribution import AttributionConfig, run_attribution
from .conversations import ConversationConfig, run_conversations
from .journey import JourneyConfig, run_journey
from .pipeline import run_pipeline
from .scraping import ScrapeConfig, run_scrape

__all__ = [
    "run_pipeline",
    "ConversationConfig", "run_conversations",
    "ScrapeConfig", "run_scrape",
    "JourneyConfig", "run_journey",
    "AttributionConfig", "run_attribution",
    "ingest", "brands", "funnel", "models",
    "conversations", "scraping", "journey", "attribution",
]

__version__ = "0.3.0"
