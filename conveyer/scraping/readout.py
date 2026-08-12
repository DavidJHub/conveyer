"""The classifier readout — module 2's results as one self-contained HTML page.

Where :mod:`conveyer.dashboard` renders the *journey* story (module 3), this
renders the *classifier* story from a finished scrape run: where the surfaced
and visited URLs landed in the taxonomy, how each label was earned (evidence
channels, fallback chain, confidence), the commerce and product↔chat view,
where attention went, and the human review queue that feeds
:mod:`conveyer.scraping.relabel`.

Design contract, same as the insight dashboard: **pure function of the saved
parquets** — no network, no matplotlib, no JS; inline SVG charts with direct
labels; light/dark via CSS tokens; the whole page is one string you can write
to a file, attach to Confluence, or drop into a notebook iframe. Rendering
twice on the same data gives the same page, and rendering after a
:mod:`relabel` pass shows the corrections — the page is a view, never a store.

Provenance is part of the page: when built from a run directory the banner
reads ``run_manifest.json`` and says out loud whether the data is the
synthetic demo corpus or a real input file.

Use it three ways::

    python -m conveyer.scraping.readout outputs/scrape_demo          # dir → readout.html
    python -m conveyer.scraping.readout PAGES.parquet --out r.html   # explicit files

    from conveyer.scraping.readout import readout_from_dir
    html = readout_from_dir("outputs/scrape_demo")                   # notebook / script
"""

from __future__ import annotations

import html as _html
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ScrapeConfig
from .taxonomy import CATEGORY_DEFINITIONS

# layout constants for the bar charts
_BAR_H, _GAP, _LBL_W, _VAL_W = 22, 8, 168, 44

#: sequential-blue ordinal ramps (validated: monotone L, visible steps, ends
#: clear the surface — dataviz reference palette, light and dark modes)
ORDINAL_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORDINAL_DARK = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]

_CAT_LABELS = {"brand_landing": "Brand landing", "catalogue": "Catalogue",
               "shopping": "Shopping", "editorial": "Editorial",
               "search": "Search", "community": "Community",
               "reference": "Reference", "unrelated": "Unrelated",
               "unknown": "Unknown"}
_SIG_LABELS = {"url": "URL tokens", "domain": "Domain knowledge",
               "markup": "Page markup", "model": "Learned model",
               "prior": "Vendor prior", "service": "Service routing",
               "base_content": "Base-URL content",
               "directory_content": "Domain directory",
               "domain_profile": "Learned domain profile",
               "content": "Page content", "platform": "Platform fingerprint",
               "llm": "LLM refinement", "human": "Human correction",
               "url_override": "URL override", "query_stripped": "Query-strip",
               "url_validated": "URL-validated", "reclassified": "Reclassified"}


def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


# --------------------------------------------------------------------------- #
# SVG chart primitives (direct labels, recessive axes, native tooltips)
# --------------------------------------------------------------------------- #
def _hbars(rows: Sequence[Tuple[str, float, str]], width: int = 560,
           unit: str = "", ramp: Optional[Sequence[str]] = None,
           note_by_key: Optional[Dict[str, str]] = None,
           fmt: str = "{:g}") -> str:
    """Horizontal bars: one hue for magnitude, a ramp only when order carries
    meaning. rows = (label, value, tooltip)."""
    rows = list(rows)
    maxv = max((v for _, v, _ in rows), default=1) or 1
    plot_w = width - _LBL_W - _VAL_W
    height = len(rows) * (_BAR_H + _GAP) - _GAP
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'preserveAspectRatio="xMinYMin meet">']
    for i, (label, val, tip) in enumerate(rows):
        y = i * (_BAR_H + _GAP)
        w = max(round(plot_w * val / maxv, 1), 2)
        color = ramp[i] if ramp else "var(--series-1)"
        r = min(4.0, w / 2)
        out.append(
            f'<g class="bar"><title>{_esc(tip)}</title>'
            f'<text x="{_LBL_W - 10}" y="{y + _BAR_H / 2}" class="cat" '
            f'text-anchor="end" dominant-baseline="central">{_esc(label)}</text>'
            f'<path d="M{_LBL_W},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} '
            f'v{_BAR_H - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r} z" fill="{color}"/>'
            f'<text x="{_LBL_W + w + 8}" y="{y + _BAR_H / 2}" class="val" '
            f'dominant-baseline="central">{fmt.format(val)}{unit}</text>')
        if note_by_key and label in note_by_key:
            note = _esc(note_by_key[label])
            if width - (_LBL_W + w + 8 + _VAL_W) > 110:
                out.append(f'<text x="{_LBL_W + w + 8 + _VAL_W}" '
                           f'y="{y + _BAR_H / 2}" class="note" '
                           f'dominant-baseline="central">{note}</text>')
            else:
                out.append(f'<text x="{_LBL_W + w - 10}" y="{y + _BAR_H / 2}" '
                           f'class="note-in" text-anchor="end" '
                           f'dominant-baseline="central">{note}</text>')
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def _seg_strip(rows: Sequence[Tuple[str, int, str]], width: int = 560,
               height: int = 30) -> str:
    """One 100% strip with 2px gaps plus a full legend — a narrow segment must
    never be identified by color alone. rows = (label, value, color)."""
    total = sum(v for _, v, _ in rows) or 1
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'preserveAspectRatio="xMinYMin meet">']
    x = 0.0
    for label, val, color in rows:
        w = width * val / total
        if w <= 0:
            continue
        out.append(f'<g class="bar"><title>{_esc(label)}: {val} of {total}</title>'
                   f'<rect x="{x:.1f}" y="0" width="{max(w - 2, 1):.1f}" '
                   f'height="{height}" rx="3" fill="{color}"/></g>')
        x += w
    out.append("</svg>")
    legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{color}"></span>'
        f'{_esc(label)} · {val}</span>'
        for label, val, color in rows if val > 0)
    zero = [label for label, val, _ in rows if val == 0]
    if zero:
        legend += f'<span class="lg lg-zero">{_esc(" · ".join(zero))} · 0</span>'
    return "".join(out) + f'<div class="legend">{legend}</div>'


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def source_note_from_manifest(run_dir: str) -> Optional[str]:
    """The honesty banner, derived from the run's own manifest. None when no
    manifest exists (bare parquet — the caller can pass ``source_note``)."""
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    src = (manifest.get("input") or {}).get("source") or {}
    if src.get("kind") == "synthetic":
        return (f"<strong>Demo corpus, not market data.</strong> This run was "
                f"scraped from the <em>synthetic</em> ground-truth corpus "
                f"({src.get('n_pages', '?')} pages, seed {src.get('seed', '?')}) — "
                f"which is why any self-evaluation reads perfect. The page "
                f"regenerates unchanged from a real run.")
    name = _esc(os.path.basename(str(src.get("path", "input"))))
    return (f"<strong>Source:</strong> <span class='mono'>{name}</span> — "
            f"{manifest.get('pages', '?')} pages, labels computed "
            f"{_esc(manifest.get('updated_at', '?'))}.")


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def build_readout(pages: pd.DataFrame, products: Optional[pd.DataFrame] = None,
                  cfg: Optional[ScrapeConfig] = None,
                  source_note: Optional[str] = None,
                  queue_rows: int = 10) -> str:
    """Render the readout from loaded frames. Everything on the page is
    computed from the data — no number is hard-coded, so the same function
    renders the demo corpus and the first real run."""
    cfg = cfg or ScrapeConfig()
    if pages is None or pages.empty:
        raise ValueError("pages frame is empty — nothing to render")
    products = products if products is not None else pd.DataFrame()

    n = len(pages)
    cat_counts = pages["page_category"].value_counts()
    stages = pages["funnel_stage"].value_counts()
    sig_freq = Counter(s for row in pages["classification_signals"]
                       for s in list(row))
    scope = pages["fetch_scope"].value_counts()
    commerce = pages[pages["page_category"].isin(
        ["brand_landing", "catalogue", "shopping"])]
    seller = commerce["seller_type"].value_counts()
    conf = pages["page_category_confidence"].astype(float)
    conf_bands = [("≥ 0.99", int((conf >= 0.99).sum())),
                  ("0.90–0.99", int(((conf >= 0.90) & (conf < 0.99)).sum())),
                  ("0.75–0.90", int(((conf >= 0.75) & (conf < 0.90)).sum())),
                  ("< 0.75", int((conf < 0.75).sum()))]
    dwell = (pages.groupby("page_category")["mean_dwell_seconds"].mean()
             .dropna().sort_values(ascending=False))
    visits = pages.groupby("page_category")["times_visited"].sum()
    visited_total = int(pages["times_visited"].sum())
    surfaced_total = int(pages["times_surfaced"].sum())
    recommended_total = int(pages["times_recommended"].sum())
    fetched = int(pages["fetch_scope"].isin(["page", "stripped"]).sum())
    n_human = int(sum("human" in list(s) for s in pages["classification_signals"]))

    # review queue (fail-open: the page renders without a model file)
    from .relabel import suspicion_report
    sus = suspicion_report(pages, cfg)
    queue = sus[sus["suspicion"] >= 1]

    # -- charts ------------------------------------------------------------- #
    cat_rows = [(_CAT_LABELS.get(c, c), int(v),
                 f"{_CAT_LABELS.get(c, c)} — {CATEGORY_DEFINITIONS.get(c, '')}")
                for c, v in cat_counts.items()]
    shopping_subs = pages.loc[pages["page_category"] == "shopping",
                              "page_subtype"].value_counts()
    cat_notes = {}
    if len(shopping_subs):
        cat_notes["Shopping"] = " · ".join(
            f"{int(v)} {k}" for k, v in shopping_subs.items())

    stage_order = ["Awareness", "Discovery", "Evaluation", "Intent", "Purchase"]
    stage_rows = [(s, int(stages.get(s, 0)), f"{s}: {int(stages.get(s, 0))} pages")
                  for s in stage_order]
    stage_ramp = [f"var(--ord{i + 1})" for i in range(5)]
    for extra in ("Post-Purchase", "Irrelevant"):
        if stages.get(extra, 0):
            stage_rows.append((extra, int(stages[extra]),
                               f"{extra}: {int(stages[extra])} pages"))
            stage_ramp.append("var(--neutral-bar)")

    sig_rows = [(_SIG_LABELS.get(k, k), int(v),
                 f"{_SIG_LABELS.get(k, k)} fired on {v} of {n} pages")
                for k, v in sig_freq.most_common()]

    scope_rows = [("Page content",
                   int(scope.get("page", 0)) + int(scope.get("stripped", 0)),
                   "var(--ord3)"),
                  ("Base-URL fallback", int(scope.get("base", 0)), "var(--ord2)"),
                  ("Domain directory", int(scope.get("directory", 0)), "var(--ord1)"),
                  ("URL heuristics only", int(scope.get("none", 0)),
                   "var(--neutral-bar)")]
    conf_rows = [(b, v, f"category confidence {b}: {v} pages")
                 for b, v in conf_bands]

    seller_rows = [("Brand-owned (DTC)", int(seller.get("brand_owned", 0)),
                    "var(--series-1)"),
                   ("Retailer", int(seller.get("retailer", 0)), "var(--series-2)")]

    tier_order = [("exact", "Exact", "var(--ord4)"), ("strong", "Strong", "var(--ord3)"),
                  ("likely", "Likely", "var(--ord2)"), ("none", "None", "var(--neutral-bar)")]
    strength = (products["match_strength"].value_counts()
                if "match_strength" in products.columns else pd.Series(dtype=int))
    tier_rows = [(lab, int(strength.get(k, 0)),
                  f"{lab}: {int(strength.get(k, 0))} of {len(products)} products")
                 for k, lab, _ in tier_order]
    tier_ramp = [c for _, _, c in tier_order]
    coincide = (int(products["coincides"].sum())
                if "coincides" in products.columns else 0)

    dwell_rows = [(_CAT_LABELS.get(c, c), round(float(v), 1),
                   f"{_CAT_LABELS.get(c, c)}: mean dwell {v:.1f}s across "
                   f"{int(visits.get(c, 0))} visits")
                  for c, v in dwell.items()]

    # -- computed findings (guarded — no invented narrative) ---------------- #
    top_cat = cat_counts.index[0] if len(cat_counts) else None
    f_cat = ""
    if top_cat is not None:
        f_cat = (f"<b>{_CAT_LABELS.get(top_cat, top_cat)} leads</b> — "
                 f"{int(cat_counts.iloc[0])} of {n} pages "
                 f"({int(cat_counts.iloc[0]) * 100 // max(n, 1)}%)")
        if visited_total and top_cat in visits.index:
            f_cat += (f", absorbing {int(visits[top_cat])} of {visited_total} "
                      f"visit events.")
        else:
            f_cat += "."
    late = int(stages.get("Intent", 0)) + int(stages.get("Purchase", 0))
    f_stage = (f"Intent + Purchase hold <b>{late} of {n}</b> pages."
               if late else "")
    f_conf = (f"<b>{conf_bands[0][1]} of {n} labels sit ≥ 0.99</b>; the "
              f"{sum(v for _, v in conf_bands[1:])} below that line are what "
              f"the review queue surfaces first.")
    f_dwell = ""
    if len(dwell) >= 2 and "shopping" in dwell.index:
        top_d = dwell.index[0]
        if top_d != "shopping":
            f_dwell = (f"<b>{_CAT_LABELS.get(top_d, top_d)} holds attention "
                       f"longest</b> ({dwell.iloc[0]:.1f}s mean) vs "
                       f"{dwell['shopping']:.1f}s on shopping pages.")
    f_products = (f"<b>{coincide} of {len(products)} products coincide</b> with "
                  f"something the chat recommended (exact/strong only). "
                  f"Corpus totals: {surfaced_total} link surfacings → "
                  f"{visited_total} visits → {recommended_total} recommendation "
                  f"mentions." if len(products) else "")

    # -- review-queue table -------------------------------------------------- #
    qrows = []
    for _, r in queue.head(queue_rows).iterrows():
        url = str(r["url"])
        disp = url.replace("https://www.", "").replace("https://", "")
        qrows.append(
            f'<tr><td class="mono url" title="{_esc(url)}">{_esc(disp[:52])}</td>'
            f'<td><span class="chip">{_esc(r["page_category"])}/'
            f'{_esc(r["page_subtype"])}</span></td>'
            f'<td class="num">{float(r["page_category_confidence"]):.2f}</td>'
            f'<td class="num sus-{min(int(r["suspicion"]), 3)}">'
            f'{int(r["suspicion"])}</td>'
            f'<td class="reason">{_esc(r["reasons"])}</td></tr>')
    queue_table = (f'<div class="tbl-wrap"><table><thead><tr><th>URL</th>'
                   f'<th>Current label</th><th class="num">Conf</th>'
                   f'<th class="num">Susp.</th><th>Why it\'s queued</th></tr>'
                   f'</thead><tbody>{"".join(qrows)}</tbody></table></div>'
                   if qrows else
                   '<p class="desc">Queue is empty — nothing scored a red flag.</p>')

    kpis = [
        (f"{n}", "URLs classified",
         "100% coverage — the fallback chain never returns empty"),
        (f"{cat_counts.size}<span class='dim'>/9</span>", "categories observed",
         "of the taxonomy's nine headline labels"),
        (f"{fetched}<span class='dim'>/{n}</span>", "content-backed labels",
         "the rest classified from base-URL, directory or URL alone"),
        (f"{coincide}<span class='dim'>/{len(products)}</span>",
         "products match the chat", "precision-first tiers — only exact/strong count"),
        (f"{len(queue)}", "rows in the review queue",
         "suspicion ≥ 1 — exported for human retagging"),
        (f"{n_human}", "human-corrected labels",
         "classifier_method=human — immutable to automated repair passes"),
    ]
    kpi_html = "".join(
        f'<div class="kpi" title="{_esc(tip)}"><div class="kpi-v">{v}</div>'
        f'<div class="kpi-l">{_esc(l)}</div></div>' for v, l, tip in kpis)

    banner = (f'<div class="banner">{source_note}</div>' if source_note else "")

    return _TEMPLATE.format(
        n=n, banner=banner, kpi_html=kpi_html,
        cat_chart=_hbars(cat_rows, note_by_key=cat_notes),
        f_cat=f_cat,
        stage_chart=_hbars(stage_rows, ramp=stage_ramp),
        f_stage=f_stage,
        sig_chart=_hbars(sig_rows),
        scope_strip=_seg_strip(scope_rows),
        conf_chart=_hbars(conf_rows), f_conf=f_conf,
        n_commerce=len(commerce),
        seller_strip=_seg_strip(seller_rows, height=34),
        n_products=len(products),
        tier_chart=_hbars(tier_rows, ramp=tier_ramp),
        f_products=f_products,
        dwell_chart=_hbars(dwell_rows, unit="s", fmt="{:.1f}", width=760),
        f_dwell=f_dwell,
        queue_len=len(queue), queue_table=queue_table,
    )


def readout_from_dir(run_dir: str, out_path: Optional[str] = None,
                     queue_rows: int = 10) -> str:
    """Build the readout from a scrape run directory (the usual entry point).

    Reads ``scraped_pages.parquet`` / ``scraped_products.parquet``, points the
    suspicion model at the run's own ``page_model.npz``, and derives the
    provenance banner from ``run_manifest.json``. Returns the HTML; also
    writes it when ``out_path`` is given.
    """
    pages_path = os.path.join(run_dir, "scraped_pages.parquet")
    if not os.path.exists(pages_path):
        raise FileNotFoundError(f"no scraped_pages.parquet in {run_dir}")
    pages = pd.read_parquet(pages_path)
    prod_path = os.path.join(run_dir, "scraped_products.parquet")
    products = pd.read_parquet(prod_path) if os.path.exists(prod_path) else None
    cfg = ScrapeConfig(out_dir=run_dir,
                       model_path=os.path.join(run_dir, "page_model.npz"))
    html = build_readout(pages, products, cfg,
                         source_note=source_note_from_manifest(run_dir),
                         queue_rows=queue_rows)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return html


def iframe(html: str, height: int = 1400) -> "object":
    """Wrap the page for inline notebook display, sandboxed in its own frame
    so the readout's CSS and the notebook's CSS never fight."""
    from IPython.display import HTML
    return HTML(f'<iframe srcdoc="{_esc(html)}" style="width:100%;'
                f'height:{height}px;border:1px solid #8887;border-radius:8px;" '
                f'loading="lazy"></iframe>')


# --------------------------------------------------------------------------- #
# Template (tokens define light; dark redefines tokens only — both the OS
# preference and an explicit data-theme stamp are honored)
# --------------------------------------------------------------------------- #
_TEMPLATE = """<title>conveyer — page classifier readout</title>
<style>
:root {{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --human:#eb6834;
  --human-ink:#a8430f; --neutral-bar:#c3c2b7; --chip:#f0efec;
  --ord1:#86b6ef; --ord2:#5598e7; --ord3:#2a78d6; --ord4:#1c5cab; --ord5:#104281;
  --warnbg:#fdf6e7;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,.10);
    --series-1:#3987e5; --series-2:#d95926; --human:#d95926;
    --human-ink:#f0a077; --neutral-bar:#52514e; --chip:#262624;
    --ord1:#cde2fb; --ord2:#9ec5f4; --ord3:#6da7ec; --ord4:#3987e5; --ord5:#1c5cab;
    --warnbg:#26200f;
  }}
}}
:root[data-theme="dark"] {{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --human:#d95926;
  --human-ink:#f0a077; --neutral-bar:#52514e; --chip:#262624;
  --ord1:#cde2fb; --ord2:#9ec5f4; --ord3:#6da7ec; --ord4:#3987e5; --ord5:#1c5cab;
  --warnbg:#26200f;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.86em; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:36px 28px 56px; }}
header {{ margin-bottom:20px; }}
.eyebrow {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); margin:0 0 6px; }}
h1 {{ font-size:29px; line-height:1.15; margin:0 0 6px; letter-spacing:-.015em;
  text-wrap:balance; }}
.sub {{ color:var(--ink-2); max-width:64ch; margin:0; }}
.banner {{ margin:18px 0 24px; padding:12px 16px; border:1px solid var(--border);
  border-left:3px solid var(--human); border-radius:6px; background:var(--warnbg);
  color:var(--ink-2); font-size:13.5px; }}
.banner strong {{ color:var(--ink); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px; margin:0 0 26px; }}
.kpi {{ background:var(--surface); border:1px solid var(--border);
  border-radius:8px; padding:14px 16px 12px; }}
.kpi-v {{ font-size:26px; font-weight:650; letter-spacing:-.02em; }}
.kpi-v .dim {{ color:var(--muted); font-size:17px; font-weight:500; }}
.kpi-l {{ font-size:12.5px; color:var(--ink-2); margin-top:2px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
@media (max-width:860px) {{ .grid {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:18px 20px 16px; min-width:0; }}
.card.wide {{ grid-column:1 / -1; }}
.card h2 {{ font-size:15.5px; margin:0 0 2px; letter-spacing:-.01em; }}
.card .desc {{ font-size:13px; color:var(--ink-2); margin:0 0 14px;
  max-width:70ch; }}
.chart {{ width:100%; height:auto; display:block; }}
.chart .cat {{ font-size:12.5px; fill:var(--ink-2); }}
.chart .val {{ font-size:12.5px; font-weight:600; fill:var(--ink);
  font-variant-numeric:tabular-nums; }}
.chart .note {{ font-size:11.5px; fill:var(--muted); }}
.chart .note-in {{ font-size:11.5px; fill:var(--surface); }}
.legend {{ display:flex; flex-wrap:wrap; gap:4px 16px; margin-top:8px;
  font-size:12.5px; color:var(--ink-2); }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }}
.lg-zero {{ color:var(--muted); }}
.sw {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
.bar path, .bar rect {{ transition:opacity .12s; }}
.bar:hover path, .bar:hover rect {{ opacity:.82; }}
.finding {{ font-size:13px; color:var(--ink-2); border-top:1px solid var(--grid);
  margin-top:12px; padding-top:10px; }}
.finding b {{ color:var(--ink); }}
.section {{ margin:28px 0 12px; }}
.section h2 {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin:0; font-weight:600; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ text-align:left; font-size:11.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:600;
  padding:6px 10px; border-bottom:1px solid var(--grid); }}
td {{ padding:7px 10px; border-bottom:1px solid var(--grid);
  vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
.num {{ font-variant-numeric:tabular-nums; text-align:right; }}
.url {{ color:var(--ink-2); white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; max-width:340px; }}
.chip {{ background:var(--chip); border-radius:4px; padding:2px 7px;
  font-size:12px; white-space:nowrap; }}
.reason {{ color:var(--ink-2); }}
.sus-3, .sus-2 {{ color:var(--human-ink); font-weight:650; }}
.tbl-wrap {{ overflow-x:auto; }}
.steps {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
  margin-top:14px; }}
@media (max-width:860px) {{ .steps {{ grid-template-columns:1fr 1fr; }} }}
.step {{ border:1px solid var(--border); border-radius:8px; padding:12px 14px;
  background:var(--page); }}
.step .k {{ font-size:11.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--human-ink); font-weight:650; }}
.step .t {{ font-weight:600; font-size:13.5px; margin:3px 0 4px; }}
.step p {{ margin:0; font-size:12.5px; color:var(--ink-2); }}
.step code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11px; color:var(--ink); background:var(--chip); border-radius:4px;
  padding:1px 5px; display:inline-block; margin-top:6px;
  overflow-wrap:anywhere; }}
footer {{ margin-top:30px; color:var(--muted); font-size:12.5px;
  border-top:1px solid var(--grid); padding-top:14px; }}
footer code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px; }}
@media (prefers-reduced-motion: reduce) {{
  .bar path, .bar rect {{ transition:none; }} }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">conveyer · module 2 · page classifier</p>
  <h1>Where AI-recommended links actually lead</h1>
  <p class="sub">Every URL an assistant surfaced or a panelist visited,
  classified into a funnel-mapped taxonomy — with the evidence behind each
  label, the product↔chat match, and the human review queue that retrains the
  model.</p>
</header>
{banner}
<div class="kpis">{kpi_html}</div>

<div class="section"><h2>Where journeys land</h2></div>
<div class="grid">
  <div class="card">
    <h2>Page categories</h2>
    <p class="desc">The taxonomy's headline label per URL. Hover a bar for the
    category's definition.</p>
    {cat_chart}
    <p class="finding">{f_cat}</p>
  </div>
  <div class="card">
    <h2>Funnel stage of the visited pages</h2>
    <p class="desc">Each page role maps to the stage it plays in the journey;
    cart/checkout bump Shopping from Intent to Purchase.</p>
    {stage_chart}
    <p class="finding">{f_stage}</p>
  </div>
</div>

<div class="section"><h2>How labels are earned</h2></div>
<div class="grid">
  <div class="card">
    <h2>Evidence channels per label</h2>
    <p class="desc">The classifier is a vote among independent modalities; any
    one can carry a page alone. Counts = pages where the channel fired.</p>
    {sig_chart}
  </div>
  <div class="card">
    <h2>Content coverage — the fallback chain</h2>
    <p class="desc">Where the deciding content came from. Unfetchable pages
    (carts, checkouts) still classify from weaker scopes — coverage is 100%
    by construction.</p>
    {scope_strip}
    <h2 style="margin-top:18px">Label confidence</h2>
    <p class="desc">Softmax over category-level rule scores.</p>
    {conf_chart}
    <p class="finding">{f_conf}</p>
  </div>
</div>

<div class="section"><h2>Commerce &amp; products</h2></div>
<div class="grid">
  <div class="card">
    <h2>Who owns the storefront</h2>
    <p class="desc">Commerce pages only ({n_commerce} of {n}) — the
    brand-owned (DTC) vs external-retailer split.</p>
    {seller_strip}
  </div>
  <div class="card">
    <h2>Product ↔ conversation match</h2>
    <p class="desc">{n_products} products extracted from page markup, matched
    to chat mentions with precision-first tiers — brand alone never counts,
    attribute conflicts veto.</p>
    {tier_chart}
    <p class="finding">{f_products}</p>
  </div>
</div>

<div class="section"><h2>Attention</h2></div>
<div class="grid"><div class="card wide">
  <h2>Mean dwell by category</h2>
  <p class="desc">Seconds until the next request, averaged per category — an
  upper bound on attention (idle time included; the last trail event has no
  dwell).</p>
  {dwell_chart}
  <p class="finding">{f_dwell}</p>
</div></div>

<div class="section"><h2>Human review — retag &amp; retrain</h2></div>
<div class="grid"><div class="card wide">
  <h2>The review queue <span class="chip"
    style="color:var(--human-ink)">suspicion ≥ 1 · {queue_len} rows</span></h2>
  <p class="desc">Rows ranked by independent red flags — unknowns, low
  confidence, thin evidence, URL-only coverage, learned-model disagreement.
  This is the CSV a reviewer opens; filling
  <span class="mono">correct_subtype</span> is enough, everything else derives
  from the taxonomy.</p>
  {queue_table}
  <div class="steps">
    <div class="step"><div class="k">1 · Export</div>
      <div class="t">Rank &amp; export suspects</div>
      <p>Top-N suspicious rows to a review CSV with empty correction columns.</p>
      <code>python -m conveyer.scraping.relabel export PAGES.parquet --out review.csv</code></div>
    <div class="step"><div class="k">2 · Correct</div>
      <div class="t">A human fills one column</div>
      <p>Set <span class="mono">correct_subtype</span> on the wrong rows;
      category and funnel stage derive from the taxonomy on apply.</p></div>
    <div class="step"><div class="k">3 · Apply</div>
      <div class="t">Validated, provenance-stamped</div>
      <p>Taxonomy-checked or loudly rejected; the row becomes
      <span class="mono">classifier_method=human</span>, confidence 1.0, and is
      immune to every automated repair pass.</p>
      <code>… relabel apply PAGES.parquet --corrections review.csv --apply</code></div>
    <div class="step"><div class="k">4 · Retrain</div>
      <div class="t">Gold labels outvote weak ones</div>
      <p>Human rows enter model training ×3, so a handful of fixes can
      overturn the mistake that taught the model.</p>
      <code>… relabel retrain PAGES.parquet</code></div>
  </div>
</div></div>

<footer>Regenerate from any run:
<code>python -m conveyer.scraping.readout RUN_DIR --out readout.html</code> —
the page is a pure function of <code>scraped_pages.parquet</code> +
<code>scraped_products.parquet</code>. Taxonomy &amp; schema:
<code>docs/SCRAPED_PAGES_SCHEMA.md</code> · classifier internals:
<code>docs/SCRAPING_MODULE.md</code> · relabel workflow:
<code>conveyer/scraping/relabel.py</code>.</footer>
</div>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render the classifier readout from a saved scrape run.")
    ap.add_argument("source", help="run directory (or a scraped_pages.parquet)")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default: SOURCE_DIR/readout.html)")
    ap.add_argument("--queue-rows", type=int, default=10)
    a = ap.parse_args(argv)

    if os.path.isdir(a.source):
        out = a.out or os.path.join(a.source, "readout.html")
        readout_from_dir(a.source, out, queue_rows=a.queue_rows)
    else:
        pages = pd.read_parquet(a.source)
        prod_path = os.path.join(os.path.dirname(a.source),
                                 "scraped_products.parquet")
        products = (pd.read_parquet(prod_path)
                    if os.path.exists(prod_path) else None)
        out = a.out or os.path.join(os.path.dirname(a.source) or ".",
                                    "readout.html")
        html = build_readout(pages, products, queue_rows=a.queue_rows)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html)
    print(f"[readout] -> {out}")


if __name__ == "__main__":
    _main()
