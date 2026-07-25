"""Dashboard generation — the project's headline insights as one HTML page.

Renders a **self-contained** dashboard (inline base64 charts, no JS, no CDN —
openable anywhere, attachable to a wiki) built directly around the project's
objectives:

* **Conversion** — the KPI row, the depth funnel (shop → cart → checkout) and
  conversion by exposure stratum with Jeffreys 95% intervals;
* **Weight of agent recommendations** — the exposure-model coefficients
  (mediator-free, see :mod:`conveyer.journey`) with the treatment bars
  highlighted, plus the recommended-brand leaderboard (recommended → visited →
  followed → converted);
* **Journey shape** — HMM journey-stage mix, intent mix, page-category mix and
  where attention goes (dwell by page category).

Two ways in::

    # from a live pipeline run
    from conveyer import run_pipeline
    from conveyer.dashboard import build_dashboard, from_pipeline_art
    art = run_pipeline()
    build_dashboard(from_pipeline_art(art))          # outputs/dashboard.html

    # from previously saved parquet outputs (no re-run needed)
    python -m conveyer.dashboard --outputs outputs   # reads conversations/ scrape/ journey/

Charts follow the house style of the notebooks: one calm hue, the orange
accent reserved for agent-recommendation exposure, direct value labels,
recessive axes.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .journey import conversion_report

HUE = "#4878a8"       # primary series
HUE2 = "#c17a3f"      # reserved: agent-recommendation exposure
INK = "#3d3d3d"
MUTED = "#767676"


# --------------------------------------------------------------------------- #
# Config + data container
# --------------------------------------------------------------------------- #
@dataclass
class DashboardConfig:
    out_path: str = "outputs/dashboard.html"
    title: str = "conveyer — agent-recommendation funnel"
    top_brands: int = 10
    conversion_stage: str = "cart"    # label only; the data already encodes it


@dataclass
class DashboardData:
    """Everything the dashboard reads, however it was obtained."""
    turns: pd.DataFrame
    conversations: pd.DataFrame
    pages: pd.DataFrame
    events: pd.DataFrame
    features: pd.DataFrame
    coefficients: Optional[pd.DataFrame] = None   # long form, model in {exposure, full}
    auc_test: Optional[float] = None
    source: str = ""
    extras: Dict[str, object] = field(default_factory=dict)


def from_pipeline_art(art: Dict[str, object]) -> DashboardData:
    """Adapter for the dict returned by :func:`conveyer.run_pipeline`."""
    j = art["journey"]
    model = j.get("model") or {}
    return DashboardData(
        turns=art["conversations"]["turns_out"],
        conversations=art["conversations"]["conversations"],
        pages=art["scrape"]["pages"],
        events=j["events"], features=j["features"],
        coefficients=model.get("coefficients"),
        auc_test=model.get("auc_test"),
        source=str(art["conversations"]["turns"].attrs.get("source", "pipeline run")),
    )


def from_output_dirs(root: str = "outputs") -> DashboardData:
    """Adapter for saved parquet outputs (conversations/ scrape/ journey/)."""
    def rd(*parts):
        return pd.read_parquet(os.path.join(root, *parts))

    coef_path = os.path.join(root, "journey", "model_coefficients.parquet")
    return DashboardData(
        turns=rd("conversations", "turn_features.parquet"),
        conversations=rd("conversations", "conversation_features.parquet"),
        pages=rd("scrape", "scraped_pages.parquet"),
        events=rd("journey", "funnel_events.parquet"),
        features=rd("journey", "journey_features.parquet"),
        coefficients=pd.read_parquet(coef_path) if os.path.exists(coef_path) else None,
        source=os.path.abspath(root),
    )


# --------------------------------------------------------------------------- #
# Chart helpers (matplotlib → inline base64)
# --------------------------------------------------------------------------- #
def _fig_to_uri(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _new_fig(w=6.4, h=3.2):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(w, h))


def _style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.tick_params(labelsize=9, colors=INK, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _barh(counts: pd.Series, title: str, hue: str = HUE, fmt="{:g}",
          accent_labels: tuple = ()) -> str:
    """Single-series horizontal bars. ``accent_labels`` get the orange accent
    (reserved for agent-recommendation exposure)."""
    counts = counts.sort_values()
    colors = [HUE2 if str(i) in accent_labels else hue for i in counts.index]
    fig, ax = _new_fig(6.4, 0.5 + 0.38 * len(counts))
    ax.barh([str(i) for i in counts.index], counts.values, color=colors, height=0.62)
    vmax = float(max(counts.values)) or 1.0
    for y, v in enumerate(counts.values):
        ax.text(v + vmax * 0.02, y, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, vmax * 1.18)
    ax.xaxis.set_visible(False)
    _style(ax, title)
    return _fig_to_uri(fig)


def _chart_conversion_by_exposure(report: pd.DataFrame) -> str:
    sub = report[report["stratum"].str.contains("overall|followed_recommendation")].copy()
    fig, ax = _new_fig(6.4, 2.6)
    ys = np.arange(len(sub))
    ax.barh(ys, sub["rate"], color=[HUE2 if "=True" in s else HUE for s in sub["stratum"]],
            height=0.55)
    ax.errorbar(sub["rate"], ys,
                xerr=[sub["rate"] - sub["ci_low"], sub["ci_high"] - sub["rate"]],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
    for y, (r, hi) in enumerate(zip(sub["rate"], sub["ci_high"])):
        ax.text(hi + 0.02, y, f"{r:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [s.replace("followed_recommendation=", "followed rec = ")
                       for s in sub["stratum"]], fontsize=9)
    ax.set_xlim(0, min(1.0, float(sub["ci_high"].max()) * 1.3))
    ax.xaxis.set_visible(False)
    _style(ax, "Conversion by agent-recommendation exposure (95% CI)")
    return _fig_to_uri(fig)


def _chart_exposure_weights(coefs: pd.DataFrame) -> str:
    exp = coefs[coefs["model"] == "exposure"].sort_values("coef_standardized") \
        if "model" in coefs.columns else coefs.sort_values("coef_standardized")
    fig, ax = _new_fig(6.4, 0.5 + 0.36 * len(exp))
    colors = [HUE2 if f in ("followed_agent_link", "visited_recommended_brand") else HUE
              for f in exp["feature"]]
    ax.barh(exp["feature"], exp["coef_standardized"], color=colors, height=0.6)
    ax.axvline(0, color=INK, lw=0.8)
    _style(ax, "Weight on conversion, per SD (orange = agent-recommendation exposure)")
    return _fig_to_uri(fig)


def _chart_depth_funnel(features: pd.DataFrame) -> str:
    n = max(1, len(features))
    stages = [("any visit", (features["n_visits"] > 0).mean()),
              ("reached shop", features["reached_shop"].mean()),
              ("reached cart", features["reached_cart"].mean()),
              ("reached checkout", features["reached_checkout"].mean())]
    fig, ax = _new_fig(6.4, 2.8)
    ys = np.arange(len(stages))[::-1]
    vals = [v for _, v in stages]
    ax.barh(ys, vals, color=HUE, height=0.58)
    for y, (label, v) in zip(ys, stages):
        ax.text(v + 0.02, y, f"{v:.0%}  ({int(round(v * n))})", va="center",
                fontsize=9, color=INK)
        ax.text(-0.02, y, label, va="center", ha="right", fontsize=9, color=INK)
    ax.set_xlim(0, 1.15)
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    _style(ax, "The behavioural funnel (share of sessions)")
    return _fig_to_uri(fig)


def _chart_dwell_by_category(events: pd.DataFrame) -> str:
    trail = events[(events["event_type"] == "trail_visit")
                   & events["dwell_seconds"].notna()]
    if trail.empty:
        return ""
    dwell = trail.groupby("page_category")["dwell_seconds"].mean().round(1)
    return _barh(dwell, "Mean dwell before moving on, by page category (s)",
                 fmt="{:.1f}")


# --------------------------------------------------------------------------- #
# Brand leaderboard
# --------------------------------------------------------------------------- #
def brand_leaderboard(data: DashboardData, top: int = 10) -> pd.DataFrame:
    """Per recommended brand: reach → behaviour → outcome."""
    convs, events, features = data.conversations, data.events, data.features
    behav = events[events["event_type"].isin(["ai_click", "trail_visit"])]
    conv_by_session = features.set_index("session_id")["converted"]

    rows = []
    rec_lists = convs.set_index("session_id")["brands_recommended_unsolicited"]
    all_brands: Dict[str, List[str]] = {}
    for sid, brands in rec_lists.items():
        for b in list(brands):
            all_brands.setdefault(b, []).append(sid)
    for brand, sessions in all_brands.items():
        visited = behav[(behav["page_brand"] == brand)]["session_id"].unique()
        followed = behav[(behav["page_brand"] == brand)
                         & (behav["brand_match"] == "unsolicited_rec")]["session_id"].unique()
        conv = conv_by_session.reindex(sessions).fillna(False)
        rows.append({
            "brand": brand,
            "sessions_recommended": len(sessions),
            "sessions_visited_brand": int(len(visited)),
            "sessions_followed_rec": int(len(followed)),
            "conversion_rate_when_recommended": round(float(conv.mean()), 3),
        })
    out = pd.DataFrame(rows).sort_values(
        ["sessions_recommended", "sessions_followed_rec"], ascending=False)
    return out.head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
_CSS = """
:root { --ink:#3d3d3d; --muted:#767676; --line:#e4e4e4; --card:#ffffff; --bg:#f6f7f9;
        --hue:#4878a8; --hue2:#c17a3f; }
* { box-sizing:border-box; margin:0; }
body { font:14px/1.5 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink); background:var(--bg); padding:28px; }
h1 { font-size:21px; font-weight:650; }
h2 { font-size:15px; font-weight:650; margin:26px 0 10px; }
.sub { color:var(--muted); font-size:12.5px; margin:4px 0 18px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.tile .v { font-size:22px; font-weight:650; }
.tile .v.accent { color:var(--hue2); }
.tile .k { color:var(--muted); font-size:11.5px; margin-top:2px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; overflow-x:auto; }
.card img { max-width:100%; height:auto; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th { text-align:left; color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); padding:6px 8px; }
td { border-bottom:1px solid var(--line); padding:6px 8px; }
tr:last-child td { border-bottom:none; }
.foot { color:var(--muted); font-size:11.5px; margin-top:26px; border-top:1px solid var(--line); padding-top:10px; }
"""


def _fmt_lift(v) -> str:
    try:
        return "–" if v is None or not np.isfinite(v) else f"{v:.2f}×"
    except TypeError:
        return "–"


def _tile(value: str, label: str, accent: bool = False) -> str:
    cls = "v accent" if accent else "v"
    return f'<div class="tile"><div class="{cls}">{value}</div><div class="k">{label}</div></div>'


def _card(inner: str) -> str:
    return f'<div class="card">{inner}</div>'


def _img(uri: str, alt: str = "chart") -> str:
    return f'<img src="{uri}" alt="{alt}">' if uri else ""


def build_dashboard(data: DashboardData, cfg: Optional[DashboardConfig] = None) -> str:
    """Render the dashboard and return its path."""
    cfg = cfg or DashboardConfig()
    f, ev = data.features, data.events
    report = conversion_report(f)
    lift = report.attrs.get("lift", {})
    n_sessions = len(f)
    n_conv = int(f["converted"].sum())
    rate = n_conv / n_sessions if n_sessions else 0.0
    overall = report[report["stratum"] == "overall"].iloc[0]
    exposure_rate = float(f["followed_recommendation"].mean()) if n_sessions else 0.0
    relevant_share = float(data.pages["is_study_relevant"].mean()) if len(data.pages) else 0.0

    tiles = [
        _tile(f"{n_sessions:,}", "sessions"),
        _tile(f"{len(data.turns):,}", "turns"),
        _tile(f"{rate:.0%}", f"conversion @ {cfg.conversion_stage} "
                             f"({overall['ci_low']:.0%}–{overall['ci_high']:.0%} CI)"),
        _tile(_fmt_lift(lift.get("followed_recommendation")),
              "lift when following the agent", accent=True),
        _tile(f"{exposure_rate:.0%}", "sessions that followed a recommendation", accent=True),
        _tile(f"{len(data.pages):,}", f"pages classified ({relevant_share:.0%} study-relevant)"),
    ]
    if data.auc_test is not None:
        tiles.append(_tile(f"{data.auc_test:.2f}", "conversion-model AUC (holdout)"))

    charts_main = [
        _card(_img(_chart_conversion_by_exposure(report), "conversion by exposure")),
        _card(_img(_chart_depth_funnel(f), "behavioural funnel")),
    ]
    if data.coefficients is not None and len(data.coefficients):
        charts_main.insert(1, _card(_img(_chart_exposure_weights(data.coefficients),
                                         "exposure model weights")))

    lb = brand_leaderboard(data, cfg.top_brands)
    lb_html = lb.to_html(index=False, border=0) if len(lb) else "<p class='sub'>no recommended brands found</p>"

    behav = ev[ev["event_type"].isin(["ai_click", "trail_visit"])]
    charts_shape = [
        _card(_img(_barh(data.turns["journey_stage"].value_counts(),
                         "Turns by journey stage (HMM-smoothed)"), "journey stages")),
        _card(_img(_barh(data.turns["intent"].value_counts(), "Turns by intent"), "intent mix")),
        _card(_img(_barh(data.pages["page_category"].value_counts(),
                         "Visited/surfaced pages by category"), "page categories")),
        _card(_img(_chart_dwell_by_category(ev), "dwell by category")),
        _card(_img(_barh(behav["brand_match"].value_counts(),
                         "Behavioural events by brand match",
                         accent_labels=("unsolicited_rec",)), "brand match mix")),
    ]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{cfg.title}</title><style>{_CSS}</style></head><body>
<h1>{cfg.title}</h1>
<p class="sub">Conversion of LLM shopping conversations and the weight of agent
recommendations · source: {data.source or "?"} · generated {now}</p>

<div class="tiles">{''.join(tiles)}</div>

<h2>Did following the agent move the needle?</h2>
<div class="grid">{''.join(charts_main)}</div>

<h2>Recommended-brand leaderboard <span class="sub">(reach → behaviour → outcome)</span></h2>
{_card(lb_html)}

<h2>Journey shape</h2>
<div class="grid">{''.join(charts_shape)}</div>

<div class="foot">Conversion is a documented <b>proxy</b> (reaching
{cfg.conversion_stage}/checkout pages in the post-turn trail), not a confirmed
purchase; exposure is observational (users self-select into following
recommendations) — see docs/FUNNEL_MODEL.md before quoting effect sizes.
Generated by <code>conveyer.dashboard</code>.</div>
</body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_path)) or ".", exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[dashboard] {os.path.abspath(cfg.out_path)} "
          f"({os.path.getsize(cfg.out_path) / 1024:.0f} KB)")
    return cfg.out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Generate the conveyer insight dashboard.")
    p.add_argument("--outputs", default=None,
                   help="Read saved parquet outputs from this root (conversations/ scrape/ journey/). "
                        "Omit to run the pipeline (synthetic fallback) first.")
    p.add_argument("--out", default="outputs/dashboard.html")
    p.add_argument("--title", default=DashboardConfig.title)
    a = p.parse_args(argv)

    if a.outputs:
        data = from_output_dirs(a.outputs)
    else:
        from .pipeline import run_pipeline

        data = from_pipeline_art(run_pipeline())
    build_dashboard(data, DashboardConfig(out_path=a.out, title=a.title))


if __name__ == "__main__":
    _main()
