"""Interaction network: users x sessions x product entities (brands).

Research question Q2: model LLM interactions, extracted product entities and
(future) clickstream events as one network. Nodes are typed
(`user` / `session` / `brand`); edges are typed too:

* ``user —[has_session]→ session``
* ``session —[requested]→ brand``    (user named the brand)
* ``session —[recommended]→ brand``  (LLM introduced the brand — unsolicited)
* ``session —[endorsed]→ brand``     (user named it AND the LLM repeated it)
* ``session —[clicked]→ brand``      (when clickstream events are joined in)

Direct vs indirect exposure falls out of the edge types: a *direct* path to a
brand goes through `requested`/`endorsed` (user already knew it); an
*indirect* path goes through `recommended` (the LLM created the exposure).
That split is the attribution primitive for Q2/Q7.
"""

from typing import Dict, Optional

import networkx as nx
import pandas as pd

from .ingest import normalize


def build_interaction_graph(df: pd.DataFrame, session_col: str = "session_id",
                            user_col: str = "user_id",
                            clicks: Optional[pd.DataFrame] = None) -> nx.MultiDiGraph:
    """Typed multigraph over users, sessions and brand entities.

    `df` needs the conveyer-prepared brand sets (`_brands_q_norm`,
    `_brands_a_norm`; run `ingest.prepare` + `analysis.add_brand_features`).
    `clicks` (optional) is a frame with [session_col, brand, n_clicks] to
    attach clickstream edges.
    """
    G = nx.MultiDiGraph()
    has_user = user_col in df.columns
    for sid, sub in df.groupby(session_col):
        s = f"session:{sid}"
        G.add_node(s, kind="session")
        if has_user:
            u = f"user:{sub[user_col].iloc[0]}"
            G.add_node(u, kind="user")
            G.add_edge(u, s, kind="has_session")
        req, rec, endo = set(), set(), set()
        for _, r in sub.iterrows():
            bq, ba = r["_brands_q_norm"], r["_brands_a_norm"]
            req |= bq
            rec |= (ba - bq)
            endo |= (ba & bq)
        for b in req | rec | endo:
            G.add_node(f"brand:{b}", kind="brand")
        for b in req:
            G.add_edge(s, f"brand:{b}", kind="requested")
        for b in rec:
            G.add_edge(s, f"brand:{b}", kind="recommended")
        for b in endo:
            G.add_edge(s, f"brand:{b}", kind="endorsed")

    if clicks is not None:
        for _, r in clicks.iterrows():
            s, b = f"session:{r[session_col]}", f"brand:{normalize(r['brand'])}"
            if s in G:
                G.add_node(b, kind="brand")
                G.add_edge(s, b, kind="clicked", weight=float(r.get("n_clicks", 1)))
    return G


def graph_summary(G: nx.MultiDiGraph) -> pd.Series:
    kinds = pd.Series([d["kind"] for _, d in G.nodes(data=True)]).value_counts()
    ekinds = pd.Series([d["kind"] for _, _, d in G.edges(data=True)]).value_counts()
    return pd.concat([kinds.add_prefix("nodes_"), ekinds.add_prefix("edges_")])


def brand_exposure_table(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Per-brand session exposure split into direct vs indirect.

    * `direct` — sessions where the user brought the brand up (`requested` or
      `endorsed`).
    * `indirect` — sessions where only the LLM introduced it (`recommended`).
    * `llm_dependence = indirect / (direct + indirect)` — how much of a
      brand's LLM-surface exposure the LLM itself creates. High values mean
      the brand's visibility is at the mercy of the model's recommendation
      policy (the segment ACES perturbations move the most).
    """
    rows: Dict[str, Dict[str, set]] = {}
    for s, b, d in G.edges(data=True):
        if not b.startswith("brand:"):
            continue
        brand = b.split(":", 1)[1]
        e = rows.setdefault(brand, {"direct": set(), "indirect": set(), "clicked": set()})
        if d["kind"] in ("requested", "endorsed"):
            e["direct"].add(s)
        elif d["kind"] == "recommended":
            e["indirect"].add(s)
        elif d["kind"] == "clicked":
            e["clicked"].add(s)
    out = pd.DataFrame([
        {"brand": k, "direct": len(v["direct"]), "indirect": len(v["indirect"]),
         "clicked": len(v["clicked"])}
        for k, v in rows.items()
    ])
    out["exposed"] = out["direct"] + out["indirect"]
    out["llm_dependence"] = out["indirect"] / out["exposed"].replace(0, pd.NA)
    return out.sort_values("exposed", ascending=False).reset_index(drop=True)


def brand_centrality(G: nx.MultiDiGraph, top: int = 15) -> pd.DataFrame:
    """Degree + eigenvector centrality of brand nodes on the brand-session
    projection — which brands sit at the core of the recommendation network."""
    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d["kind"] in ("requested", "recommended", "endorsed", "clicked"):
            U.add_edge(u, v)
    deg = nx.degree_centrality(U)
    try:
        eig = nx.eigenvector_centrality(U, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eig = {n: float("nan") for n in U}
    rows = [{"brand": n.split(":", 1)[1], "degree": deg[n], "eigenvector": eig[n]}
            for n in U if n.startswith("brand:")]
    return (pd.DataFrame(rows).sort_values("degree", ascending=False)
            .head(top).reset_index(drop=True))


def cooccurrence_edges(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """Brand–brand co-recommendation edges (named in the same answer) — the
    substitution/consideration-set structure LLM answers induce."""
    from collections import Counter
    from itertools import combinations

    c: Counter = Counter()
    for ba in df["_brands_a_norm"]:
        for a, b in combinations(sorted(ba), 2):
            c[(a, b)] += 1
    rows = [{"brand_a": a, "brand_b": b, "count": n}
            for (a, b), n in c.items() if n >= min_count]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
