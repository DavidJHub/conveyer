"""Human-in-the-loop relabelling: export → correct → apply → retrain.

The learned channel self-trains on *weak* labels (its own confident rows), so
it can absorb the real URL distribution — but it can never fix a page the
whole classifier got wrong, because the wrong label is what it trains on.
Breaking that loop takes a human, and this module is the smallest workflow
that lets one intervene without touching a parquet by hand:

1. **export** — rank the table by suspicion (unknowns, low confidence, thin
   evidence, URL-only coverage, relevance conflicts, learned-model
   disagreement) and write the top rows to a review CSV with empty
   ``correct_*`` columns;
2. a human fills ``correct_subtype`` (and optionally ``correct_seller_type``
   / ``note``) for the rows that are wrong — everything else derives from the
   taxonomy, so one column is usually enough;
3. **apply** — validate the corrections against the taxonomy and write them
   back into the parquet with full provenance: ``classifier_method`` becomes
   ``human``, ``human`` joins ``classification_signals``, confidence goes to
   1.0. Human rows are **immutable downstream**: the validate/reclassify
   repair passes skip them, and re-applying a stale CSV cannot silently
   overwrite a newer correction;
4. **retrain** — ``model.train`` picks the corrected rows up through the
   normal weak-label path, and rows carrying the ``human`` signal are
   **boosted** (duplicated) so a handful of gold labels can outweigh hundreds
   of weak ones.

CLI::

    python -m conveyer.scraping.relabel export  PAGES.parquet --out review.csv [--n 40]
    python -m conveyer.scraping.relabel apply   PAGES.parquet --corrections review.csv --apply
    python -m conveyer.scraping.relabel retrain PAGES.parquet [--model page_model.npz]
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ScrapeConfig
from .taxonomy import (PAGE_CATEGORIES, SELLER_TYPES, SUBTYPE_TO_CATEGORY,
                       category_for_subtype, funnel_stage_for, is_commerce)

#: what the human edits; everything else in the CSV is context, not input
CORRECTION_COLUMNS = ["correct_subtype", "correct_category",
                      "correct_seller_type", "note"]

#: how many copies of a human-labelled sample enter model training — a gold
#: label should outweigh weak ones, and duplication is sample weighting that
#: needs no scikit-learn API beyond fit()
HUMAN_BOOST = 3


def _signals(row) -> List[str]:
    sig = row.get("classification_signals")
    if sig is None or (isinstance(sig, float) and pd.isna(sig)):
        return []
    return [str(s) for s in list(sig)]


def is_human_labelled(row) -> bool:
    """Whether a pages row carries a human correction (the immutability flag)."""
    return "human" in _signals(row if isinstance(row, dict) else row.to_dict())


# --------------------------------------------------------------------------- #
# 1 · Export: rank by suspicion
# --------------------------------------------------------------------------- #
def _model_predictions(pages: pd.DataFrame, cfg: ScrapeConfig) -> Optional[pd.Series]:
    """URL-only subtype prediction per row from the learned channel, or None
    when no model file loads. Disagreement with the stored label is one of the
    strongest review signals available for free."""
    try:
        from .model import _load_model, featurize
        model = _load_model(cfg.model_path, os.path.getmtime(cfg.model_path))
    except (OSError, Exception):                      # noqa: BLE001 — fail-open
        return None
    if model is None:
        return None

    def top(url: str) -> str:
        probs = model.predict_proba_one(featurize(str(url)))
        return max(probs, key=probs.get) if probs else ""

    return pages["url"].map(top)


def suspicion_report(pages: pd.DataFrame,
                     cfg: Optional[ScrapeConfig] = None) -> pd.DataFrame:
    """One row per page with a ``suspicion`` score and the reasons behind it.

    The score is a plain sum of independent red flags — transparent enough to
    argue with, which is the point of a review queue:

    * ``unknown`` category (the classifier's own "can't tell")        +3
    * learned model disagrees with the stored subtype                  +2
    * ``unrelated`` yet topically relevant (conflicting axes)          +2
    * category confidence below 0.9                                    +2 (·<0.75 ⇒ +3)
    * classified without page content (``fetch_scope`` ≠ page)         +1
    * ≤ 2 modalities fired                                             +1

    Human-labelled rows score 0 by definition — they *are* the ground truth.
    """
    cfg = cfg or ScrapeConfig()
    df = pages.copy()
    preds = _model_predictions(df, cfg)

    rows = []
    for idx, r in df.iterrows():
        row = r.to_dict()
        reasons: List[str] = []
        score = 0
        if is_human_labelled(row):
            rows.append({"suspicion": 0, "reasons": "human-labelled"})
            continue
        cat = str(row.get("page_category") or "")
        conf = float(row.get("page_category_confidence") or 0.0)
        sig = _signals(row)
        scope = str(row.get("fetch_scope") or "")
        rel = float(row.get("skincare_relevance") or 0.0)

        if cat == "unknown":
            score += 3; reasons.append("category=unknown")
        if preds is not None:
            pred = str(preds.loc[idx] or "")
            if pred and pred != str(row.get("page_subtype") or ""):
                score += 2; reasons.append(f"model says '{pred}'")
        if cat == "unrelated" and rel >= 0.5:
            score += 2; reasons.append(f"unrelated but relevance={rel:.2f}")
        if conf < 0.75:
            score += 3; reasons.append(f"confidence={conf:.2f}")
        elif conf < 0.9:
            score += 2; reasons.append(f"confidence={conf:.2f}")
        if scope not in ("page", "stripped"):
            score += 1; reasons.append(f"no page content ({scope or 'none'})")
        if len(sig) <= 2:
            score += 1; reasons.append(f"{len(sig)} signal(s)")
        rows.append({"suspicion": score, "reasons": "; ".join(reasons) or "clean"})

    out = df.assign(**pd.DataFrame(rows, index=df.index))
    return out.sort_values(["suspicion", "page_category_confidence"],
                           ascending=[False, True])


_EXPORT_CONTEXT = ["page_id", "url", "title", "page_category", "page_subtype",
                   "seller_type", "funnel_stage", "page_category_confidence",
                   "skincare_relevance", "fetch_scope", "classifier_method",
                   "suspicion", "reasons"]


def export_review(pages: pd.DataFrame, out_csv: str, n: int = 40,
                  min_suspicion: int = 1,
                  cfg: Optional[ScrapeConfig] = None) -> pd.DataFrame:
    """Write the top-``n`` suspects to a review CSV and return them.

    The CSV carries read-only context columns plus the four ``correct_*``
    columns, empty. Filling ``correct_subtype`` is enough for most fixes —
    category and funnel stage derive from the taxonomy on apply.
    """
    sus = suspicion_report(pages, cfg)
    picked = sus[sus["suspicion"] >= min_suspicion].head(n)
    out = picked.reindex(columns=_EXPORT_CONTEXT).copy()
    for c in CORRECTION_COLUMNS:
        out[c] = ""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[relabel] {len(out)} rows (suspicion ≥ {min_suspicion}) -> {out_csv}\n"
          f"[relabel] fill correct_subtype (one of: "
          f"{', '.join(sorted(SUBTYPE_TO_CATEGORY))}), then run apply")
    return out


# --------------------------------------------------------------------------- #
# 2 · Apply: taxonomy-checked, provenance-stamped
# --------------------------------------------------------------------------- #
def _clean(v) -> str:
    """CSV cell -> stripped string; the NaN a pandas read leaves in an empty
    cell must read as empty, not as the literal 'nan'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _validate_correction(row: dict) -> Tuple[Optional[dict], Optional[str]]:
    """(normalised correction, error). Exactly one of the two is None."""
    sub = _clean(row.get("correct_subtype")).lower()
    cat = _clean(row.get("correct_category")).lower()
    seller = _clean(row.get("correct_seller_type")).lower()
    if not sub and not cat:
        return None, None                     # untouched review row — skip
    if sub and sub not in SUBTYPE_TO_CATEGORY:
        return None, f"unknown subtype '{sub}'"
    if cat and cat not in PAGE_CATEGORIES:
        return None, f"unknown category '{cat}'"
    if sub and cat and category_for_subtype(sub) != cat:
        return None, (f"subtype '{sub}' belongs to "
                      f"'{category_for_subtype(sub)}', not '{cat}'")
    if not cat:
        cat = category_for_subtype(sub)
    if not sub:
        return None, (f"category '{cat}' alone is ambiguous — "
                      f"pick a subtype")
    if seller and seller not in SELLER_TYPES:
        return None, f"unknown seller_type '{seller}'"
    if seller and seller != "na" and not is_commerce(cat):
        return None, f"seller_type '{seller}' is meaningless for '{cat}'"
    return {"subtype": sub, "category": cat, "seller": seller,
            "note": _clean(row.get("note"))}, None


def apply_corrections(pages: pd.DataFrame, corrections: pd.DataFrame
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a filled review CSV; returns (corrected pages, change report).

    Guarantees: every correction is taxonomy-valid or loudly rejected;
    ``funnel_stage`` is re-derived, never trusted from the CSV; rows already
    human-labelled are not overwritten by a *stale* CSV (re-applying an old
    export is a no-op, not a regression); provenance is stamped so the model
    trainer and the repair passes can tell gold from weak.
    """
    fixed = pages.copy()
    by_key = {}
    for k in ("page_id", "url"):
        if k in fixed.columns:
            by_key.update({str(v): i for i, v in fixed[k].items() if str(v)})

    report = []
    for _, c in corrections.iterrows():
        row = c.to_dict()
        pid, curl = _clean(row.get("page_id")), _clean(row.get("url"))
        target = by_key.get(pid) if pid in by_key else by_key.get(curl)
        corr, err = _validate_correction(row)
        if corr is None and err is None:
            continue
        url = curl or pid or "?"
        if err:
            report.append({"url": url, "applied": False, "detail": err})
            continue
        if target is None:
            report.append({"url": url, "applied": False,
                           "detail": "row not found in the pages table"})
            continue
        if is_human_labelled(fixed.loc[target].to_dict()):
            report.append({"url": url, "applied": False,
                           "detail": "already human-labelled — newer correction wins"})
            continue

        old = (str(fixed.at[target, "page_category"]),
               str(fixed.at[target, "page_subtype"]))
        fixed.at[target, "page_category"] = corr["category"]
        fixed.at[target, "page_subtype"] = corr["subtype"]
        fixed.at[target, "funnel_stage"] = funnel_stage_for(corr["category"],
                                                            corr["subtype"])
        if corr["seller"]:
            fixed.at[target, "seller_type"] = corr["seller"]
        elif not is_commerce(corr["category"]):
            fixed.at[target, "seller_type"] = "na"
        if corr["category"] not in ("unrelated", "unknown") \
                and "is_study_relevant" in fixed.columns:
            fixed.at[target, "is_study_relevant"] = True
        fixed.at[target, "page_category_confidence"] = 1.0
        fixed.at[target, "classifier_method"] = "human"
        sig = _signals(fixed.loc[target].to_dict())
        fixed.at[target, "classification_signals"] = sig + ["human"]
        report.append({"url": url, "applied": True,
                       "detail": f"{old[0]}/{old[1]} -> "
                                 f"{corr['category']}/{corr['subtype']}"
                                 + (f" ({corr['note']})" if corr["note"] else "")})
    return fixed, pd.DataFrame(report, columns=["url", "applied", "detail"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Human relabelling for scraped_pages: export a review CSV, "
                    "apply the corrections, retrain the learned channel.")
    ap.add_argument("command", choices=["export", "apply", "retrain"])
    ap.add_argument("pages", help="scraped_pages.parquet")
    ap.add_argument("--out", default=None, help="export: CSV path / apply: new parquet")
    ap.add_argument("--corrections", default=None, help="apply: the filled review CSV")
    ap.add_argument("--apply", dest="in_place", action="store_true",
                    help="apply: overwrite the parquet in place")
    ap.add_argument("--n", type=int, default=40, help="export: max rows")
    ap.add_argument("--min-suspicion", type=int, default=1)
    ap.add_argument("--model", default=None, help="retrain: model .npz path")
    a = ap.parse_args(argv)

    pages = pd.read_parquet(a.pages)
    if a.command == "export":
        export_review(pages, a.out or "outputs/relabel_review.csv",
                      n=a.n, min_suspicion=a.min_suspicion)
        return
    if a.command == "retrain":
        from .model import train
        train(pages_parquet=a.pages, out_path=a.model)
        return
    # apply
    if not a.corrections:
        ap.error("apply needs --corrections REVIEW.csv")
    from .validate import write_pages
    fixed, report = apply_corrections(
        pages, pd.read_csv(a.corrections, dtype=str, keep_default_na=False))
    n_ok = int(report["applied"].sum()) if len(report) else 0
    print(report.to_string(index=False) if len(report) else "[relabel] no corrections found")
    if not n_ok:
        return
    if a.in_place:
        print(f"[relabel] {n_ok} corrections -> {write_pages(fixed, a.pages)}")
    elif a.out:
        print(f"[relabel] {n_ok} corrections -> {write_pages(fixed, a.out)}")
    else:
        print(f"[relabel] {n_ok} corrections validated — re-run with --apply "
              f"(in place) or --out PATH to write them")


if __name__ == "__main__":
    _main()
