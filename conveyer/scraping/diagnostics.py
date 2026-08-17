"""The scrape & classifier diagnostics page — every URL, every label, every reason.

Where :mod:`conveyer.scraping.readout` tells the classifier's *story* (where
journeys land, attention, the review loop), this page is its *diagnosis
companion*: the full URL inventory with the label, the method, the evidence
channels, the confidence and the suspicion score on every single row, plus
per-category example URLs — the page you open when you want to argue with the
classifier, not admire it.

It also answers the "which metric are we using?" question out loud: the pages
table **does** carry a confidence column — ``page_category_confidence``, a
softmax over the category-level vote mass (:func:`conveyer.scraping.classify.
_softmax_conf`). But it is a *separation* measure, not a calibrated
probability, and it saturates near 1.0 on clear winners — so operational
triage keys on the ``suspicion`` score (:func:`conveyer.scraping.relabel.
suspicion_report`) instead. The dashboard prints both, per row.

Design contract, same as the readout: **pure function of the saved parquets**
— no network, no matplotlib, no JS; inline SVG charts with direct labels;
light/dark via CSS tokens; one string you can write to a file or drop into a
notebook iframe. Rendering twice on the same data gives the same page.

Use it three ways::

    python -m conveyer.scraping.diagnostics results                 # dir → diagnostics.html
    python -m conveyer.scraping.diagnostics PAGES.parquet --out d.html

    from conveyer.scraping.diagnostics import diagnostics_from_dir
    html = diagnostics_from_dir("results")                          # notebook / script

A parquet that is only a **bare URL list** (no classification columns) also
works: :func:`ensure_pages_frame` classifies it URL-only on the fly
(:func:`conveyer.scraping.classify_url` — no fetching), so the dashboard can
diagnose a scrape *input* as well as a finished run.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import ScrapeConfig
from .readout import (_CAT_LABELS, _SIG_LABELS, _esc, _hbars, _seg_strip,
                      iframe, source_note_from_manifest)
from .taxonomy import CATEGORY_DEFINITIONS, PAGE_CATEGORIES

__all__ = ["build_diagnostics", "diagnostics_from_dir", "load_run",
           "ensure_pages_frame", "confidence_report", "iframe"]

#: what each classifier_method value means (shown as chart notes + tooltips)
METHOD_NOTES: Dict[str, str] = {
    "rule": "multimodal vote alone (URL / domain / markup / model …)",
    "rule+prior": "vote + the SimilarWeb page_type prior fired",
    "rule (url-only)": "URL + domain knowledge only — nothing was fetched",
    "llm": "low-confidence page refined by the Anthropic model",
    "human": "relabel correction — confidence pinned to 1.0, immutable",
    "error": "row processing raised — the label fell back to unknown",
}

#: repair-pass suffixes the validate/reclassify passes append to the method
_METHOD_SUFFIXES: Dict[str, str] = {
    "url_validated": "URL-validated by the repair pass",
    "reclassified": "reclassified by the repair pass",
}


def _method_note(m) -> str:
    """Human meaning of a ``classifier_method`` value, including the
    ``+url_validated`` / ``+reclassified`` suffixed variants the repair
    passes write (``rule+prior+reclassified`` → base note + suffix note)."""
    m = _text(m)
    if m in METHOD_NOTES:
        return METHOD_NOTES[m]
    parts = [p for p in m.split("+") if p]
    base = "+".join(p for p in parts if p not in _METHOD_SUFFIXES)
    notes = [METHOD_NOTES.get(base, "")] + \
        [_METHOD_SUFFIXES[p] for p in parts if p in _METHOD_SUFFIXES]
    return "; ".join(n for n in notes if n)


# category display order: taxonomy order, so runs are comparable page-to-page
_CAT_ORDER = {c: i for i, c in enumerate(PAGE_CATEGORIES)}


def _as_list(v) -> List[str]:
    """A parquet list column value as a plain list (None/NaN → [])."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    return [str(s) for s in list(v)]


def _signals(row) -> List[str]:
    return _as_list(row.get("classification_signals"))


def _num(v, default: float = 0.0) -> float:
    """A float with None/NaN/garbage collapsed to ``default`` — the ``or 0.0``
    idiom lets NaN through (NaN is truthy), so never use it on parquet values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f


def _text(v) -> str:
    """A string with None/NaN collapsed to '' (str(nan) would render 'nan')."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


# --------------------------------------------------------------------------- #
# Input tolerance: a finished pages table OR a bare URL list
# --------------------------------------------------------------------------- #
def ensure_pages_frame(df: pd.DataFrame,
                       cfg: Optional[ScrapeConfig] = None) -> pd.DataFrame:
    """Return a frame the dashboard can render.

    A finished ``fact_scraped_page`` table passes through untouched. A bare
    URL list (any frame with a ``url``-ish column but no ``page_category``)
    is classified **URL-only** via :func:`conveyer.scraping.classify_url` —
    no network, no fetching — and gets the minimal set of columns the page
    reads, with ``classifier_method="rule (url-only)"`` so the provenance is
    visible on every row.
    """
    if "page_category" in df.columns:
        return df
    url_col = next((c for c in ("url", "surfaced_url", "final_url")
                    if c in df.columns), None)
    if url_col is None:
        raise ValueError("frame has neither classification columns nor a "
                         "url/surfaced_url column — nothing to diagnose")
    from . import classify_url
    from .classify import parse_url
    cfg = cfg or ScrapeConfig()
    rows = []
    for url in df[url_col].astype(str):
        cls = classify_url(url, cfg)
        rows.append({
            "url": url, "domain": parse_url(url).registrable,
            "title": "", "page_category": cls.page_category,
            "page_subtype": cls.page_subtype,
            "page_category_confidence": float(cls.confidence),
            "seller_type": cls.seller_type, "funnel_stage": cls.funnel_stage,
            "classifier_method": "rule (url-only)",
            "classification_signals": list(cls.signals),
            "skincare_relevance": float(cls.skincare_relevance),
            "is_study_relevant": bool(cls.is_study_relevant),
            "fetch_status": "not_fetched", "fetch_scope": "none",
            "word_count": 0, "n_products": 0,
            "times_surfaced": 0, "times_visited": 0,
        })
    return pd.DataFrame(rows)


def load_run(source: str, cfg: Optional[ScrapeConfig] = None
             ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """(pages, products, source_note) from a run directory or a parquet file.

    Directories prefer the pipeline's own ``scraped_pages.parquet`` /
    ``scraped_products.parquet``; otherwise the first ``*.parquet`` found is
    treated as the pages input (and run through :func:`ensure_pages_frame`,
    so a bare URL list works too). Products are optional everywhere.
    """
    cfg = cfg or ScrapeConfig()
    note = None
    if os.path.isdir(source):
        note = source_note_from_manifest(source)
        pages_path = os.path.join(source, cfg.pages_filename)
        if not os.path.exists(pages_path):
            others = sorted(f for f in os.listdir(source)
                            if f.endswith(".parquet")
                            and f != cfg.products_filename)
            if not others:
                found = sorted(f for f in os.listdir(source)
                               if f.endswith(".parquet"))
                raise FileNotFoundError(
                    f"no pages parquet in {source}"
                    + (f" (found only {', '.join(found)} — a products file "
                       f"needs its pages table next to it)" if found else ""))
            pages_path = os.path.join(source, others[0])
        prod_path = os.path.join(source, cfg.products_filename)
    else:
        if not source.endswith(".parquet"):
            raise ValueError(f"{source} is neither a run directory nor a "
                             f".parquet file")
        pages_path = source
        prod_path = os.path.join(os.path.dirname(source) or ".",
                                 cfg.products_filename)
        note = source_note_from_manifest(os.path.dirname(source) or ".")
    pages = ensure_pages_frame(pd.read_parquet(pages_path), cfg)
    products = (pd.read_parquet(prod_path) if os.path.exists(prod_path)
                else pd.DataFrame())
    return pages, products, note


# --------------------------------------------------------------------------- #
# The confidence question, answered as data
# --------------------------------------------------------------------------- #
def confidence_report(pages: pd.DataFrame) -> Dict[str, float]:
    """The numbers behind "which metric are we using?" — see module docstring.

    Keys: ``present`` (the column exists), ``mean``/``median``/``min``,
    ``share_saturated`` (≥ 0.99), ``share_exact_1``, ``share_below_llm_gate``
    (< 0.55, the threshold that would trigger LLM refinement),
    ``share_below_090`` (the suspicion red-flag line).
    """
    if "page_category_confidence" not in pages.columns:
        return {"present": 0.0}
    c = pages["page_category_confidence"].astype(float).fillna(0.0)
    n = max(len(c), 1)
    return {
        "present": 1.0,
        "mean": round(float(c.mean()), 4),
        "median": round(float(c.median()), 4),
        "min": round(float(c.min()), 4),
        "share_saturated": round(float((c >= 0.99).sum()) / n, 4),
        "share_exact_1": round(float((c == 1.0).sum()) / n, 4),
        "share_below_llm_gate": round(float((c < 0.55).sum()) / n, 4),
        "share_below_090": round(float((c < 0.90).sum()) / n, 4),
    }


def _suspicion(pages: pd.DataFrame, cfg: ScrapeConfig) -> pd.DataFrame:
    """Suspicion-annotated copy of ``pages`` (fail-open: zeros on any error,
    so the dashboard renders even without a model file or relabel deps)."""
    try:
        from .relabel import suspicion_report
        return suspicion_report(pages, cfg)
    except Exception:
        out = pages.copy()
        out["suspicion"] = 0
        out["reasons"] = ""
        return out


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def build_diagnostics(pages: pd.DataFrame,
                      products: Optional[pd.DataFrame] = None,
                      cfg: Optional[ScrapeConfig] = None,
                      source_note: Optional[str] = None,
                      example_rows: int = 5,
                      max_table_rows: Optional[int] = None) -> str:
    """Render the diagnostics page from a pages frame (+ optional products).

    ``max_table_rows`` caps the every-URL table; when it bites, the page says
    so out loud (no silent truncation). Default ``None`` = all rows.
    """
    cfg = cfg or ScrapeConfig()
    if pages is None or pages.empty:
        raise ValueError("pages frame is empty — nothing to diagnose")
    if max_table_rows is not None and max_table_rows <= 0:
        raise ValueError("max_table_rows must be positive (or None for all rows)")
    pages = ensure_pages_frame(pages, cfg)
    products = products if products is not None else pd.DataFrame()

    n = len(pages)
    sus = _suspicion(pages, cfg)                 # sorted most-suspicious first
    cat_counts = pages["page_category"].value_counts()
    sub_counts = pages["page_subtype"].value_counts()
    method_counts = pages["classifier_method"].value_counts()
    sig_freq = Counter(s for row in pages["classification_signals"]
                       for s in _as_list(row))
    stages = pages["funnel_stage"].value_counts()
    scope = (pages["fetch_scope"].value_counts()
             if "fetch_scope" in pages.columns else pd.Series(dtype=int))
    conf = pages["page_category_confidence"].astype(float).fillna(0.0) \
        if "page_category_confidence" in pages.columns else pd.Series(dtype=float)
    rep = confidence_report(pages)
    n_domains = int(pages["domain"].nunique()) if "domain" in pages.columns else 0
    fetched = (int(pages["fetch_scope"].isin(["page", "stripped"]).sum())
               if "fetch_scope" in pages.columns else 0)
    queue_len = int((sus["suspicion"] >= 1).sum())

    # -- charts ------------------------------------------------------------- #
    cat_rows = [(_CAT_LABELS.get(c, c), int(v),
                 f"{_CAT_LABELS.get(c, c)} — {CATEGORY_DEFINITIONS.get(c, '')}")
                for c, v in cat_counts.items()]
    sub_rows = [(s, int(v), f"page_subtype={s}: {int(v)} of {n} pages")
                for s, v in sub_counts.items()]
    method_rows = [(m, int(v), f"{m} — {_method_note(m)} ({int(v)} pages)")
                   for m, v in method_counts.items()]
    method_notes = {m: _method_note(m) for m, _ in method_counts.items()}
    sig_rows = [(_SIG_LABELS.get(k, k), int(v),
                 f"{_SIG_LABELS.get(k, k)} fired on {v} of {n} pages")
                for k, v in sig_freq.most_common()]
    stage_order = ["Awareness", "Discovery", "Evaluation", "Intent",
                   "Purchase", "Post-Purchase", "Irrelevant"]
    stage_rows = [(s, int(stages[s]), f"{s}: {int(stages[s])} pages")
                  for s in stage_order if stages.get(s, 0)]

    scope_rows = [("Page content",
                   int(scope.get("page", 0)) + int(scope.get("stripped", 0)),
                   "var(--ord3)"),
                  ("Base-URL fallback", int(scope.get("base", 0)), "var(--ord2)"),
                  ("Domain directory", int(scope.get("directory", 0)), "var(--ord1)"),
                  ("URL heuristics only", int(scope.get("none", 0)),
                   "var(--neutral-bar)")]

    conf_bands = [("= 1.00 exactly", int((conf == 1.0).sum())),
                  ("0.99–1.00", int(((conf >= 0.99) & (conf < 1.0)).sum())),
                  ("0.90–0.99", int(((conf >= 0.90) & (conf < 0.99)).sum())),
                  ("0.75–0.90", int(((conf >= 0.75) & (conf < 0.90)).sum())),
                  ("0.55–0.75", int(((conf >= 0.55) & (conf < 0.75)).sum())),
                  ("< 0.55 (LLM gate)", int((conf < 0.55).sum()))]
    conf_rows = [(b, v, f"page_category_confidence {b}: {v} of {n} pages")
                 for b, v in conf_bands]

    sus_counts = sus["suspicion"].value_counts().sort_index()
    sus_rows = [(("clean (0)" if s == 0 else f"suspicion {int(s)}"), int(v),
                 f"suspicion={int(s)}: {int(v)} pages")
                for s, v in sus_counts.items()]

    # -- findings (computed, never invented) -------------------------------- #
    f_conf = ""
    if rep.get("present"):
        f_conf = (f"<b>The metric is <span class='mono'>page_category_confidence"
                  f"</span></b> — a softmax over the category-level vote mass "
                  f"(<span class='mono'>classify._softmax_conf</span>). It is a "
                  f"<i>separation</i> measure, not a calibrated probability, and "
                  f"it saturates: <b>{rep['share_saturated'] * 100:.0f}% of rows "
                  f"sit ≥ 0.99</b> ({rep['share_exact_1'] * 100:.0f}% exactly "
                  f"1.0; min {rep['min']:.2f}). Only {rep['share_below_llm_gate'] * 100:.1f}% "
                  f"fall under the 0.55 LLM-refinement gate — which is why triage "
                  f"runs on the <b>suspicion score</b> instead (right).")
    f_sus = (f"<b>{queue_len} of {n} rows carry at least one red flag</b> — "
             f"this is the review queue the relabel workflow exports. "
             f"Flags: unknown category +3, model disagreement +2, "
             f"unrelated-but-relevant +2, confidence &lt; 0.9 +2 (&lt; 0.75 +3), "
             f"no page content +1, ≤ 2 evidence channels +1.")
    f_method = ""
    if len(method_counts):
        top_m = method_counts.index[0]
        f_method = (f"<b>{_esc(top_m)}</b> decided "
                    f"{int(method_counts.iloc[0])} of {n} labels — "
                    f"{_esc(_method_note(top_m))}.")

    # -- per-category example cards ----------------------------------------- #
    ex_cards = []
    for c in sorted(cat_counts.index, key=lambda x: _CAT_ORDER.get(x, 99)):
        grp = pages[pages["page_category"] == c]
        rows_html = []
        for _, r in grp.head(example_rows).iterrows():
            url = _text(r["url"])
            disp = url.replace("https://www.", "").replace("https://", "")
            title = _text(r.get("title"))
            cv = _num(r.get("page_category_confidence"))
            rows_html.append(
                f'<div class="ex"><span class="mono url" title="{_esc(url)}'
                f'{" — " + _esc(title) if title else ""}">{_esc(disp[:64])}</span>'
                f'<span class="ex-meta"><span class="chip">{_esc(r["page_subtype"])}'
                f'</span> {_esc(_text(r.get("classifier_method")))}'
                f' · {cv:.2f}</span></div>')
        more = (f'<div class="ex-more">… and {len(grp) - example_rows} more</div>'
                if len(grp) > example_rows else "")
        ex_cards.append(
            f'<div class="card"><h2>{_esc(_CAT_LABELS.get(c, c))} '
            f'<span class="chip">{len(grp)}</span></h2>'
            f'<p class="desc">{_esc(CATEGORY_DEFINITIONS.get(c, ""))}</p>'
            f'{"".join(rows_html)}{more}</div>')

    # -- the every-URL table (most suspicious first) ------------------------- #
    table_df = sus
    cap_note = ""
    if max_table_rows is not None and len(table_df) > max_table_rows:
        cap_note = (f'<p class="desc"><b>Showing the {max_table_rows} most '
                    f'suspicious of {len(table_df)} rows</b> — the rest are in '
                    f'the parquet; re-render with a higher '
                    f'<span class="mono">max_table_rows</span> for all.</p>')
        table_df = table_df.head(max_table_rows)
    trs = []
    for i, (_, r) in enumerate(table_df.iterrows(), 1):
        url = _text(r["url"])
        disp = url.replace("https://www.", "").replace("https://", "")
        title = _text(r.get("title"))
        cv = _num(r.get("page_category_confidence"))
        rel = _num(r.get("skincare_relevance"))
        s = int(_num(r.get("suspicion")))
        sig = " · ".join(_signals(r.to_dict())) or "—"
        scope_v = _text(r.get("fetch_scope")) or "—"
        conf_cls = ' class="num lo"' if cv < 0.9 else ' class="num"'
        trs.append(
            f'<tr><td class="num idx">{i}</td>'
            f'<td class="mono url" title="{_esc(url)}'
            f'{" — " + _esc(title) if title else ""}">{_esc(disp[:56])}</td>'
            f'<td><span class="chip" title="funnel stage: '
            f'{_esc(_text(r.get("funnel_stage")))}">{_esc(r["page_category"])}/'
            f'{_esc(r["page_subtype"])}</span></td>'
            f'<td>{_esc(_text(r.get("classifier_method")))}</td>'
            f'<td{conf_cls}>{cv:.2f}</td>'
            f'<td class="num">{rel:.2f}</td>'
            f'<td class="sig">{_esc(sig)}</td>'
            f'<td>{_esc(scope_v)}</td>'
            f'<td class="num sus-{min(s, 3)}">{s}</td></tr>')
    url_table = (f'<div class="tbl-wrap"><table><thead><tr>'
                 f'<th class="num">#</th><th>URL</th><th>Label</th>'
                 f'<th>Method</th><th class="num">Conf</th>'
                 f'<th class="num">Rel.</th><th>Signals</th><th>Scope</th>'
                 f'<th class="num">Susp.</th></tr></thead>'
                 f'<tbody>{"".join(trs)}</tbody></table></div>')

    kpis = [
        (f"{n}", "URLs in the table", "one row per distinct scraped URL"),
        (f"{n_domains}", "distinct domains", "registrable domains"),
        (f"{cat_counts.size}<span class='dim'>/9</span>", "categories observed",
         "of the taxonomy's nine headline labels"),
        (f"{fetched}<span class='dim'>/{n}</span>", "content-backed labels",
         "fetch_scope = page/stripped; the rest classified from weaker scopes"),
        (f"{rep.get('median', 0):.2f}" if rep.get("present") else "—",
         "median confidence",
         "page_category_confidence — softmax of category vote mass"),
        (f"{queue_len}", "rows flagged for review",
         "suspicion ≥ 1 — the relabel queue, most suspicious first below"),
        (f"{len(products)}", "products extracted",
         "fact_scraped_product rows (0 when no products parquet)"),
    ]
    kpi_html = "".join(
        f'<div class="kpi" title="{_esc(tip)}"><div class="kpi-v">{v}</div>'
        f'<div class="kpi-l">{_esc(l)}</div></div>' for v, l, tip in kpis)

    banner = (f'<div class="banner">{source_note}</div>' if source_note else "")

    return _TEMPLATE.format(
        n=n, banner=banner, kpi_html=kpi_html,
        cat_chart=_hbars(cat_rows),
        sub_chart=_hbars(sub_rows),
        method_chart=_hbars(method_rows, note_by_key=method_notes, width=760),
        f_method=f_method,
        stage_chart=_hbars(stage_rows),
        sig_chart=_hbars(sig_rows),
        scope_strip=_seg_strip(scope_rows),
        conf_chart=_hbars(conf_rows), f_conf=f_conf,
        sus_chart=_hbars(sus_rows), f_sus=f_sus,
        ex_cards="".join(ex_cards),
        cap_note=cap_note, url_table=url_table,
        n_table=len(table_df),
    )


def diagnostics_from_dir(source: str, out_path: Optional[str] = None,
                         cfg: Optional[ScrapeConfig] = None,
                         example_rows: int = 5,
                         max_table_rows: Optional[int] = None) -> str:
    """Build the diagnostics page from a run directory or a pages parquet
    (the usual entry point). Returns the HTML; also writes it when
    ``out_path`` is given."""
    import dataclasses
    cfg = cfg or ScrapeConfig()
    pages, products, note = load_run(source, cfg)
    # keep the caller's cfg; only point the run-local fields (suspicion model,
    # out_dir) at the source's own directory — for files and dirs alike
    base = source if os.path.isdir(source) else (os.path.dirname(source) or ".")
    cfg = dataclasses.replace(cfg, out_dir=base,
                              model_path=os.path.join(base, "page_model.npz"))
    html = build_diagnostics(pages, products, cfg, source_note=note,
                             example_rows=example_rows,
                             max_table_rows=max_table_rows)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return html


# --------------------------------------------------------------------------- #
# Template — same tokens as the readout so the two pages read as one system
# (light defined on :root; dark redefines tokens under the OS preference and
# an explicit data-theme stamp; body background always painted).
# --------------------------------------------------------------------------- #
_TEMPLATE = """<title>conveyer — scrape &amp; classifier diagnostics</title>
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
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:10px; margin:0 0 26px; }}
.kpi {{ background:var(--surface); border:1px solid var(--border);
  border-radius:8px; padding:14px 16px 12px; }}
.kpi-v {{ font-size:26px; font-weight:650; letter-spacing:-.02em; }}
.kpi-v .dim {{ color:var(--muted); font-size:17px; font-weight:500; }}
.kpi-l {{ font-size:12.5px; color:var(--ink-2); margin-top:2px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
@media (max-width:960px) {{ .grid3 {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width:860px) {{ .grid, .grid3 {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:18px 20px 16px; min-width:0; }}
.card.wide {{ grid-column:1 / -1; }}
.card h2 {{ font-size:15.5px; margin:0 0 2px; letter-spacing:-.01em; }}
.card .desc {{ font-size:13px; color:var(--ink-2); margin:0 0 14px;
  max-width:76ch; }}
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
  padding:6px 8px; border-bottom:1px solid var(--grid); }}
th.num {{ text-align:right; }}
td {{ padding:6px 8px; border-bottom:1px solid var(--grid);
  vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
.num {{ font-variant-numeric:tabular-nums; text-align:right; }}
.num.lo {{ color:var(--human-ink); font-weight:650; }}
.idx {{ color:var(--muted); }}
.url {{ color:var(--ink-2); white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; max-width:270px; }}
.sig {{ color:var(--muted); font-size:11.5px; max-width:170px; }}
.chip {{ background:var(--chip); border-radius:4px; padding:2px 7px;
  font-size:12px; white-space:nowrap; }}
.sus-3, .sus-2 {{ color:var(--human-ink); font-weight:650; }}
.tbl-wrap {{ overflow-x:auto; max-height:720px; overflow-y:auto; }}
.ex {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline;
  padding:5px 0; border-bottom:1px solid var(--grid); }}
.ex:last-of-type {{ border-bottom:none; }}
.ex .url {{ max-width:56%; }}
.ex-meta {{ font-size:12px; color:var(--ink-2); white-space:nowrap; }}
.ex-more {{ font-size:12px; color:var(--muted); padding-top:6px; }}
footer {{ margin-top:30px; color:var(--muted); font-size:12.5px;
  border-top:1px solid var(--grid); padding-top:14px; }}
footer code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px; }}
@media (prefers-reduced-motion: reduce) {{
  .bar path, .bar rect {{ transition:none; }} }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">conveyer · module 2 · diagnostics</p>
  <h1>Every URL, every label, every reason</h1>
  <p class="sub">The scrape &amp; classifier forensics page: the full URL
  inventory with the label, method, evidence channels, confidence and
  suspicion score on each row — plus what the confidence metric actually is
  and why triage doesn't rely on it.</p>
</header>
{banner}
<div class="kpis">{kpi_html}</div>

<div class="section"><h2>What the classifier decided</h2></div>
<div class="grid">
  <div class="card">
    <h2>Page categories</h2>
    <p class="desc">The taxonomy's headline label per URL. Hover a bar for the
    category's definition.</p>
    {cat_chart}
  </div>
  <div class="card">
    <h2>Page subtypes</h2>
    <p class="desc">The finer structural label (pdp / serp / article / …) the
    category derives from.</p>
    {sub_chart}
  </div>
  <div class="card wide">
    <h2>Classifier methods</h2>
    <p class="desc">How each label was decided —
    <span class="mono">classifier_method</span> per row.</p>
    {method_chart}
    <p class="finding">{f_method}</p>
  </div>
  <div class="card">
    <h2>Funnel stages</h2>
    <p class="desc">Where each page role sits in the journey; cart/checkout
    bump Shopping from Intent to Purchase.</p>
    {stage_chart}
  </div>
  <div class="card">
    <h2>Evidence channels</h2>
    <p class="desc">Independent modalities that voted; any one can carry a
    page alone. Counts = pages where the channel fired.</p>
    {sig_chart}
  </div>
</div>

<div class="section"><h2>The confidence question</h2></div>
<div class="grid">
  <div class="card">
    <h2>Label confidence — <span class="mono">page_category_confidence</span></h2>
    <p class="desc">Softmax over the category-level vote mass. The per-category
    score vector (<span class="mono">PageClass.scores</span>) is computed but
    <b>not persisted</b> to the parquet — only this collapsed number is.</p>
    {conf_chart}
    <p class="finding">{f_conf}</p>
  </div>
  <div class="card">
    <h2>Suspicion — the operative triage metric</h2>
    <p class="desc">A transparent sum of independent red flags
    (<span class="mono">relabel.suspicion_report</span>); it ranks the review
    queue and orders the table below.</p>
    {sus_chart}
    <p class="finding">{f_sus}</p>
  </div>
  <div class="card wide">
    <h2>Content coverage — the fallback chain</h2>
    <p class="desc">Where each label's deciding content came from. Unfetchable
    pages still classify from weaker scopes — coverage is 100% by
    construction.</p>
    {scope_strip}
  </div>
</div>

<div class="section"><h2>Examples by category</h2></div>
<div class="grid3">
{ex_cards}
</div>

<div class="section"><h2>Every URL ({n_table} rows, most suspicious first)</h2></div>
<div class="grid"><div class="card wide">
  {cap_note}
  {url_table}
</div></div>

<footer>Regenerate from any run:
<code>python -m conveyer.scraping.diagnostics RUN_DIR --out diagnostics.html</code>
— a pure function of <code>scraped_pages.parquet</code>
(+ <code>scraped_products.parquet</code> when present). A bare URL-list
parquet works too: rows classify URL-only on the fly. Story view:
<code>python -m conveyer.scraping.readout RUN_DIR</code> · taxonomy &amp;
schema: <code>docs/SCRAPED_PAGES_SCHEMA.md</code> · review workflow:
<code>conveyer/scraping/relabel.py</code>.</footer>
</div>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Render the scrape & classifier diagnostics page from a "
                    "saved run (directory or pages parquet).")
    ap.add_argument("source", help="run directory or a scraped_pages.parquet "
                                   "(a bare URL-list parquet works too)")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default: SOURCE_DIR/diagnostics.html)")
    ap.add_argument("--example-rows", type=int, default=5)
    ap.add_argument("--max-table-rows", type=int, default=None,
                    help="cap the every-URL table (default: all rows)")
    a = ap.parse_args(argv)

    base = a.source if os.path.isdir(a.source) else (os.path.dirname(a.source) or ".")
    out = a.out or os.path.join(base, "diagnostics.html")
    diagnostics_from_dir(a.source, out, example_rows=a.example_rows,
                         max_table_rows=a.max_table_rows)
    print(f"[diagnostics] -> {out}")


if __name__ == "__main__":
    _main()
