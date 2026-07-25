"""Double-check stored page labels against the URL rules alone, and repair.

The URL channel is the most trustworthy voter for *structural* roles: a
``/cart/`` path is a cart whatever the body looked like, a ``?q=retinol`` SERP
is on-topic whatever was fetched. This module re-runs the **URL-only**
classifier over an existing ``scraped_pages`` table and flags rows where a
decisive URL verdict contradicts the stored label — the safety net for cases
like ``amazon.com/cart/add-to-cart/…`` being stored as ``unrelated/pdp``
because cart-page furniture (prices, checkout buttons) outvoted the URL.

Two mismatch kinds are actionable:

* ``category`` — stored ``unrelated``/``unknown`` (funnel ``Irrelevant``) but
  the URL alone proves a journey page (transactional token, earned neutral
  role on a known domain, or on-topic slug);
* ``subtype`` — stored ``shopping`` but with the wrong stage: the URL path
  carries a decisive cart/checkout/order token while the stored subtype says
  otherwise (Intent vs **Purchase** — this one moves the conversion proxy).

Use it on existing output (no re-scrape needed)::

    python -m conveyer.scraping.validate outputs/scrape/scraped_pages.parquet          # report
    python -m conveyer.scraping.validate outputs/scrape/scraped_pages.parquet --apply  # repair in place
    python -m conveyer.scraping.validate ... --out fixed.parquet                       # repair to a copy

or programmatically: :func:`validation_report` / :func:`apply_validation`.
``run_scrape`` also prints a ``[validate]`` summary after every run, so a
regression like this can't ship silently again.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import pandas as pd

from .classify import TRANSACTIONAL_SUBTYPES, PageClass, classify_rule
from .config import ScrapeConfig
from .directory import lookup
from .extract import PageContent
from .schema import PAGE_SCHEMA, write_parquet

_REPORT_COLUMNS = ["page_id", "url", "page_category", "page_subtype",
                   "funnel_stage", "url_category", "url_subtype", "url_funnel",
                   "url_relevant", "mismatch"]


def _classify_url_only(url: str, cfg: ScrapeConfig) -> PageClass:
    # same as conveyer.scraping.classify_url, imported piecewise to keep this
    # module usable from pipeline.py without a package-init import cycle
    entry = lookup(url, cfg.directory_path) if cfg.directory_fallback else None
    return classify_rule(PageContent(url=url), url, cfg, directory_entry=entry)


def validation_report(pages: pd.DataFrame,
                      cfg: Optional[ScrapeConfig] = None) -> pd.DataFrame:
    """One row per actionable disagreement between the stored label and the
    URL-only rules. Empty frame = every label is consistent with its URL."""
    cfg = cfg or ScrapeConfig()
    rows = []
    if pages is None or pages.empty or "url" not in pages.columns:
        return pd.DataFrame(columns=_REPORT_COLUMNS)
    for _, r in pages.iterrows():
        url = str(r.get("url") or "")
        if not url:
            continue
        v = _classify_url_only(url, cfg)
        stored_cat = str(r.get("page_category") or "")
        stored_sub = str(r.get("page_subtype") or "")
        # decisive URL verdicts only: the URL earns a real category AND proves
        # journey membership (transactional token / neutral role on a known
        # domain / on-topic slug). Anything weaker is left alone.
        url_decisive = v.page_category not in ("unknown", "unrelated") \
            and v.is_study_relevant
        transactional = v.page_subtype in TRANSACTIONAL_SUBTYPES and "url" in v.signals
        mismatch = ""
        if stored_cat in ("unrelated", "unknown") and url_decisive:
            mismatch = "category"
        elif stored_cat == "shopping" and transactional and stored_sub != v.page_subtype:
            mismatch = "subtype"
        if mismatch:
            rows.append({
                "page_id": str(r.get("page_id") or ""), "url": url,
                "page_category": stored_cat, "page_subtype": stored_sub,
                "funnel_stage": str(r.get("funnel_stage") or ""),
                "url_category": v.page_category, "url_subtype": v.page_subtype,
                "url_funnel": v.funnel_stage,
                "url_relevant": bool(v.is_study_relevant), "mismatch": mismatch,
            })
    return pd.DataFrame(rows, columns=_REPORT_COLUMNS)


def apply_validation(pages: pd.DataFrame,
                     cfg: Optional[ScrapeConfig] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (repaired copy of ``pages``, the report of what changed).

    Repaired rows take the URL verdict for ``page_category`` /
    ``page_subtype`` / ``funnel_stage`` / ``is_study_relevant``, get
    ``+url_validated`` appended to ``classifier_method`` and a
    ``url_validated`` entry in ``classification_signals`` — the fix stays
    auditable, nothing is overwritten silently."""
    cfg = cfg or ScrapeConfig()
    report = validation_report(pages, cfg)
    if report.empty:
        return pages, report
    fixed = pages.copy()
    by_url = report.set_index("url")
    for idx in fixed.index:
        url = str(fixed.at[idx, "url"] or "")
        if url not in by_url.index:
            continue
        v = by_url.loc[url]
        fixed.at[idx, "page_category"] = v["url_category"]
        fixed.at[idx, "page_subtype"] = v["url_subtype"]
        fixed.at[idx, "funnel_stage"] = v["url_funnel"]
        if "is_study_relevant" in fixed.columns:
            fixed.at[idx, "is_study_relevant"] = bool(v["url_relevant"])
        if "classifier_method" in fixed.columns:
            m = str(fixed.at[idx, "classifier_method"] or "rule")
            if "url_validated" not in m:
                fixed.at[idx, "classifier_method"] = m + "+url_validated"
        if "classification_signals" in fixed.columns:
            sig = fixed.at[idx, "classification_signals"]
            sig = list(sig) if sig is not None and not isinstance(sig, float) else []
            if "url_validated" not in sig:
                fixed.at[idx, "classification_signals"] = sig + ["url_validated"]
    return fixed, report


# --------------------------------------------------------------------------- #
# Writing back with the pinned schema
# --------------------------------------------------------------------------- #
def _coerce_rows(pages: pd.DataFrame) -> list:
    """DataFrame -> rows that satisfy PAGE_SCHEMA's types again (a parquet
    round-trip turns nullable ints into float64-with-NaN and list columns into
    numpy arrays; pyarrow needs them back as ints/lists/None)."""
    import pyarrow as pa
    rows = pages.to_dict("records")
    for row in rows:
        for field in PAGE_SCHEMA:
            val = row.get(field.name)
            is_na = val is None or (isinstance(val, float) and pd.isna(val))
            if pa.types.is_list(field.type):
                row[field.name] = [] if is_na else [str(x) for x in list(val)]
            elif pa.types.is_integer(field.type):
                row[field.name] = None if is_na else int(val)
            elif pa.types.is_floating(field.type):
                row[field.name] = None if is_na else float(val)
            elif pa.types.is_boolean(field.type):
                row[field.name] = False if is_na else bool(val)
            elif is_na:
                row[field.name] = ""
    return rows


def write_pages(pages: pd.DataFrame, path: str) -> str:
    return write_parquet(_coerce_rows(pages), PAGE_SCHEMA, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Double-check scraped_pages labels against the URL rules; "
                    "optionally repair the parquet.")
    ap.add_argument("pages", help="path to scraped_pages.parquet")
    ap.add_argument("--apply", action="store_true",
                    help="write the repaired table back to the same file")
    ap.add_argument("--out", default=None,
                    help="write the repaired table to this path instead")
    a = ap.parse_args(argv)

    pages = pd.read_parquet(a.pages)
    report = validation_report(pages)
    if report.empty:
        print(f"[validate] {len(pages)} pages — all labels agree with the URL rules")
        return
    print(f"[validate] {len(report)} disagreement(s) out of {len(pages)} pages:")
    summary = (report.groupby(["mismatch", "page_category", "url_category", "url_subtype"])
               .size().rename("n").reset_index())
    print(summary.to_string(index=False))
    print("\nexamples:")
    print(report[["url", "page_category", "page_subtype", "url_category",
                  "url_subtype", "url_funnel"]].head(10).to_string(index=False))

    if a.apply or a.out:
        fixed, _ = apply_validation(pages)
        out = a.out or a.pages
        write_pages(fixed, out)
        print(f"\n[validate] repaired {len(report)} rows -> {out}")
    else:
        print("\nre-run with --apply (in place) or --out PATH to repair these rows")


if __name__ == "__main__":
    _main()
