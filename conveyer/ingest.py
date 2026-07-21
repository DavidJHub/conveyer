"""Data ingestion for the conversations parquet (the project's single input).

Expected schema (grain = one user↔LLM turn; see ``docs/DATA_DICTIONARY.md``):

    session_id, message_id, user_id, prompt_datetime, question, answer,
    a_links_source   (list of links surfaced in the response),
    ai_click         (list of links clicked directly from the AI response),
    next_10_urls     (list of {request_time, requested_site} navigation events)

This module

* loads + normalises that parquet (:func:`load_conversations`) — timestamps
  parsed, ``session_pos`` derived, the three link columns coerced to plain
  Python lists whatever pandas hands back (numpy arrays of dicts, lists,
  stringified reprs);
* owns the shared list/trail helpers (:func:`as_records`,
  :func:`trail_events`) used by every module — dwell times come from the
  ``request_time`` delta to the *next* request (the last event never gets one);
* generates a **synthetic dataset with ground truth**
  (:func:`make_synthetic_conversations`) so the whole three-module pipeline —
  conversation features → page classification → funnel model — runs and can be
  *validated* end-to-end with no data and no network.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("session_id", "message_id", "question", "answer")
LINK_COLUMNS = ("a_links_source", "ai_click", "next_10_urls")


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def normalize(s: Any) -> str:
    return str(s).strip().lower()


def to_list(x: Any) -> List[str]:
    """Coerce a cell into a list of strings (legacy helper, string-oriented)."""
    if isinstance(x, list):
        return [str(b).strip() for b in x if str(b).strip()]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set)):
                    return [str(b).strip() for b in v if str(b).strip()]
            except (ValueError, SyntaxError):
                pass
        return [t.strip() for t in s.split(",") if t.strip()]
    return []


def as_records(cell: Any) -> List[Any]:
    """Coerce a link-list cell into a plain Python list.

    Real parquet gives list<struct> columns back as **numpy arrays of dicts**;
    other exports give Python lists or a stringified repr with single quotes.
    Handle all of them — silently dropping these was a real production bug.
    """
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        return cell.tolist()
    if isinstance(cell, (list, tuple)):
        return list(cell)
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if isinstance(cell, str):
        s = cell.strip()
        if not s or s.lower() == "none":
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(s)
                return list(v) if isinstance(v, (list, tuple)) else [v]
            except (ValueError, SyntaxError):
                continue
        return [s]
    return [cell]


def event_url(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("requested_site") or item.get("url") or "").strip()
    return str(item or "").strip()


def event_time_ms(item: Any) -> Optional[int]:
    if not isinstance(item, dict):
        return None
    try:
        return int(str(item.get("request_time")))
    except (TypeError, ValueError):
        return None


def trail_events(cell: Any) -> List[Dict[str, Any]]:
    """Parse one ``next_10_urls`` cell into ordered events with dwell times.

    ``dwell_seconds`` for event *i* is the gap until request *i+1* — how long
    the user stayed before moving on. The **last** event has no successor, so
    its dwell is None (never guessed). Position is 1-based within the trail.
    """
    items = as_records(cell)
    events = [{"url": event_url(it), "t_ms": event_time_ms(it)} for it in items]
    events = [e for e in events if e["url"]]
    if len(events) > 1 and all(e["t_ms"] is not None for e in events):
        events.sort(key=lambda e: e["t_ms"])
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(events):
        dwell = None
        if i + 1 < len(events) and e["t_ms"] is not None and events[i + 1]["t_ms"] is not None:
            delta = (events[i + 1]["t_ms"] - e["t_ms"]) / 1000.0
            dwell = delta if delta >= 0 else None
        out.append({"url": e["url"], "t_ms": e["t_ms"], "position": i + 1,
                    "dwell_seconds": dwell})
    return out


def link_urls(cell: Any) -> List[str]:
    """Plain URL list from an ``a_links_source`` / ``ai_click`` cell."""
    return [u for u in (event_url(it) for it in as_records(cell)) if u]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_any(path: str) -> pd.DataFrame:
    """Read a dataframe, autodetecting format by extension."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {path}")


def load_conversations(source: "str | pd.DataFrame") -> pd.DataFrame:
    """Load + normalise the conversations table.

    Accepts a path or an already-loaded frame. Guarantees: required columns
    present, ``prompt_dt`` (pandas datetime, NaT-tolerant), ``session_pos``
    (1-based within session, time-ordered), link columns coerced to lists.
    """
    df = read_any(source) if isinstance(source, str) else source.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"conversations table is missing columns: {missing}")

    for col in LINK_COLUMNS:
        df[col] = df[col].apply(as_records) if col in df.columns \
            else [[] for _ in range(len(df))]

    if "prompt_datetime" in df.columns:
        df["prompt_dt"] = pd.to_datetime(df["prompt_datetime"], errors="coerce")
    else:
        df["prompt_dt"] = pd.NaT
    if "user_id" not in df.columns:
        df["user_id"] = df["session_id"]

    df["question"] = df["question"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    order = df.sort_values(["session_id", "prompt_dt", "message_id"]).index
    df = df.loc[order].reset_index(drop=True)
    df["session_pos"] = df.groupby("session_id").cumcount() + 1
    return df


# --------------------------------------------------------------------------- #
# Synthetic conversations with ground truth
# --------------------------------------------------------------------------- #
# Session archetypes: (name, share). Ground truth flags derive from these.
_ARCHETYPES = [
    ("converter_via_rec", 0.25),   # follows the agent's rec to cart+checkout
    ("browser_via_rec", 0.25),     # follows the rec, browses, never carts
    ("organic_converter", 0.10),   # converts on a brand the agent did NOT push
    ("no_follow", 0.30),           # wanders off-topic after the answer
    ("researcher", 0.10),          # reads reviews/reference, no shopping
]

_QUESTION_TEMPLATES = [
    "what should i use for {concern}?",
    "best {ptype} for {concern}?",
    "is {brand} good for {concern}?",
    "can you recommend a {ptype} for {concern}",
    "how do i fix {concern}",
]
_CONCERNS = ["dry skin", "acne", "dark spots", "redness", "oily skin", "fine lines"]


def make_synthetic_conversations(n_sessions: int = 40, seed: int = 42,
                                 ) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    """(conversations df, offline html corpus, ground-truth df).

    The trails visit pages generated by :mod:`conveyer.scraping.synthetic`, so
    module 2 can classify them offline; cart/checkout URLs are deliberately
    *unfetchable* (URL-only classification, like real carts). Ground truth per
    session: ``gt_followed_rec``, ``gt_converted``, ``gt_archetype``.
    """
    from .scraping.synthetic import (_CATALOG, _RETAILERS, _EDITORIAL, _pdp,
                                     _editorial, _search, _community,
                                     _reference, _unrelated)

    rng = np.random.default_rng(seed)
    # deterministic archetype shares (largest-remainder), shuffled order — a
    # random draw can skew tiny samples badly enough to distort the lift demo
    counts = {a: int(p * n_sessions) for a, p in _ARCHETYPES}
    for a, _ in sorted(_ARCHETYPES, key=lambda x: -x[1]):
        if sum(counts.values()) >= n_sessions:
            break
        counts[a] += 1
    arch_list = [a for a, c in counts.items() for _ in range(c)][:n_sessions]
    rng.shuffle(arch_list)

    html: Dict[str, str] = {}
    rows: List[dict] = []
    gt: List[dict] = []
    base_ms = 1_769_400_000_000

    def page(builder, *args, **kw) -> str:
        url, doc, *_ = builder(*args, **kw)
        html[url] = doc
        return url

    for s in range(n_sessions):
        sid, uid = f"s{s:04d}", f"u{s % max(4, n_sessions // 3):03d}"
        arch = arch_list[s]
        brand, domain, product, ptype, price, rating, rc = _CATALOG[s % len(_CATALOG)]
        alt = _CATALOG[(s + 3) % len(_CATALOG)]
        retailer = _RETAILERS[s % len(_RETAILERS)]
        pub = _EDITORIAL[s % len(_EDITORIAL)]
        concern = _CONCERNS[s % len(_CONCERNS)]

        rec_pdp = page(_pdp, brand, domain, product, ptype, price, rating, rc,
                       retailer=retailer)
        brand_pdp = page(_pdp, brand, domain, product, ptype, price, rating, rc)
        review = page(_editorial, pub[0], pub[1], [_CATALOG[s % len(_CATALOG)]])
        cart = f"https://www.{retailer[1]}/gp/cart/view.html?ref_=nav_cart" \
            if retailer[1] == "amazon.com" else f"https://www.{retailer[1]}/cart"
        checkout = f"https://www.{retailer[1]}/checkout"          # unfetchable on purpose

        # --- the trail this archetype produces --------------------------- #
        if arch == "converter_via_rec":
            trail = [rec_pdp, review, cart, checkout]
            clicks = [rec_pdp]
            followed, converted = True, True
        elif arch == "browser_via_rec":
            trail = [rec_pdp, brand_pdp, review]
            clicks = [rec_pdp] if rng.random() < 0.6 else []
            followed, converted = True, False
        elif arch == "organic_converter":
            b2, d2, p2, t2, pr2, ra2, rc2 = alt
            alt_pdp = page(_pdp, b2, d2, p2, t2, pr2, ra2, rc2, retailer=retailer)
            trail = [page(_search, f"best {t2} for {concern}"), alt_pdp, cart, checkout]
            clicks = []
            followed, converted = False, True
        elif arch == "researcher":
            trail = [page(_reference, ptype.capitalize()),
                     page(_community, product, brand), review]
            clicks = []
            followed, converted = False, False
        else:  # no_follow
            u1, h1, *_ = _unrelated(s)
            html[u1] = h1
            trail = [u1, "https://x.com/", f"https://x.com/somebody/status/{2_000_000_000 + s}"]
            clicks = []
            followed, converted = False, False

        # --- 2-3 turns of conversation ------------------------------------ #
        n_turns = int(rng.integers(2, 4))
        t0 = base_ms + s * 3_600_000
        for turn in range(n_turns):
            mid = f"{sid}m{turn}"
            q = _QUESTION_TEMPLATES[int(rng.integers(len(_QUESTION_TEMPLATES)))] \
                .format(concern=concern, ptype=ptype, brand=brand)
            if turn < n_turns - 1:
                a = (f"For {concern}, a good {ptype} is {brand} {product} — gentle and "
                     f"well reviewed ({rating}/5).")
                links, aic, trail_cell = [], [], []
            else:  # last turn carries the recommendation links + the trail
                a = (f"I'd recommend {brand} {product} for {concern}. "
                     f"You can find it here: {rec_pdp} or on the brand site {brand_pdp}.")
                links = [rec_pdp, brand_pdp]
                aic = clicks
                step = t0 + 60_000
                trail_cell = [{"request_time": str(step + i * int(rng.integers(2_000, 40_000))),
                               "requested_site": u} for i, u in enumerate(trail)]
            rows.append({
                "session_id": sid, "message_id": mid, "user_id": uid,
                "prompt_datetime": pd.Timestamp(t0 + turn * 90_000, unit="ms").isoformat(),
                "question": q, "answer": a,
                "a_links_source": links, "ai_click": aic, "next_10_urls": trail_cell,
            })
        gt.append({"session_id": sid, "gt_archetype": arch,
                   "gt_followed_rec": followed, "gt_converted": converted,
                   "gt_rec_brand": brand})

    df = load_conversations(pd.DataFrame(rows))
    return df, html, pd.DataFrame(gt)


def ingest(path: str = "data/conversations.parquet", n_synthetic: int = 40,
           seed: int = 42) -> Tuple[pd.DataFrame, Optional[Dict[str, str]],
                                    Optional[pd.DataFrame]]:
    """Load the real parquet if present, else synthesise.

    Returns ``(df, html_by_url, ground_truth)`` — the last two are None on
    real data (fetching happens online there).
    """
    if os.path.exists(path):
        df = load_conversations(path)
        df.attrs["source"] = path
        return df, None, None
    df, html, gt = make_synthetic_conversations(n_synthetic, seed)
    df.attrs["source"] = f"SYNTHETIC ({df.session_id.nunique()} sessions, {len(df)} turns)"
    return df, html, gt
