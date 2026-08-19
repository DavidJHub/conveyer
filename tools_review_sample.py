"""Independent re-tagging of the sampled demo domains, and agreement scoring
against the model's own verdict.

Labels below are assigned by inspection of the URL + title + known identity of
each domain, using the SAME taxonomy the classifier uses
(conveyer/scraping/taxonomy.py) so the comparison is apples-to-apples.

`relevant` is the reviewer's judgement of whether the page belongs to the
skincare/beauty study at all — kept separate from the structural label,
mirroring the classifier's own two-axis design.
"""
import pandas as pd

from conveyer.scraping.taxonomy import funnel_stage_for

# domain -> (page_category, page_subtype, seller_type, relevant, note)
REVIEW = {
    "accela.com":            ("unrelated", "account",  "na", 0, "gov permit portal login"),
    "amz123.com":            ("unrelated", "homepage", "na", 0, "CN cross-border e-comm news portal"),
    "araks.com":             ("unrelated", "brand_site", "brand_owned", 0, "lingerie brand info page"),
    "arras.io":              ("unrelated", "other",    "na", 0, "browser game"),
    "benefitscal.com":       ("unrelated", "account",  "na", 0, "benefits portal password reset"),
    "bilibili.com":          ("unrelated", "homepage", "na", 0, "CN video platform (video gap)"),
    "ca.gov":                ("unrelated", "account",  "na", 0, "state timesheet portal"),
    "customer.io":           ("unrelated", "tool",     "na", 0, "martech SaaS dashboard"),
    "dermcarecharlotte.com": ("reference", "local",    "na", 1, "dermatology clinic booking - RELEVANT service"),
    "dior.com":              ("shopping",  "pdp",      "brand_owned", 1, "fragrance PDP"),
    "drugs.com":             ("reference", "homepage", "na", 1, "drug/health reference, skincare-adjacent"),
    "dslhospice.com":        ("unrelated", "account",  "na", 0, "patient login"),
    "emojicombos.com":       ("unrelated", "tool",     "na", 0, "emoji copy-paste tool"),
    "gamebanana.com":        ("unrelated", "other",    "na", 0, "game mod download"),
    "globemagazine.com":     ("unrelated", "account",  "na", 0, "wp-admin backend"),
    "goflow.com":            ("unrelated", "homepage", "na", 0, "ecommerce ops SaaS"),
    "gov.co":                ("unrelated", "article",  "na", 0, "consulate newsroom"),
    "hauteandwhatnot.com":   ("editorial", "homepage", "na", 1, "fashion/beauty blog"),
    "hwahae.com":            ("catalogue", "collection", "na", 1, "KR beauty awards listing"),
    "joinf.com":             ("unrelated", "tool",     "na", 0, "trade email SaaS"),
    "lotioncrafter.com":     ("shopping",  "cart",     "brand_owned", 1, "skincare ingredients cart"),
    "makeugc.ai":            ("unrelated", "tool",     "na", 0, "AI UGC generator"),
    "miaprep.com":           ("unrelated", "other",    "na", 0, "online curriculum"),
    "micoaes.com":           ("shopping",  "listing",  "brand_owned", 0, "B2B aesthetic LASER EQUIPMENT, not consumer skincare"),
    "oliveyoung.com":        ("shopping",  "pdp",      "retailer", 1, "major KR beauty retailer PDP - MISSED"),
    "owndoc.shop":           ("unknown",   "other",    "na", 0, "geo-block error shell, not real content"),
    "pacificabeauty.com":    ("shopping",  "pdp",      "brand_owned", 1, "beauty brand PDP"),
    "perplexity.ai":         ("search",    "serp",     "na", 0, "AI answer engine"),
    "realtor.com":           ("unrelated", "pdp",      "na", 0, "real-estate listing"),
    "rosecoaudit.com":       ("unrelated", "article",  "na", 0, "RU industrial audit"),
    "rover.com":             ("unrelated", "homepage", "na", 0, "pet sitting marketplace"),
    "secretnature.com":      ("unrelated", "pdp",      "brand_owned", 0, "CBD/THCA, borderline topical"),
    "shellpointmtg.com":     ("unrelated", "account",  "na", 0, "mortgage sign-in"),
    "similarweb.com":        ("unrelated", "tool",     "na", 0, "analytics dashboard"),
    "suno.com":              ("unrelated", "tool",     "na", 0, "AI music generator"),
    "telegram.org":          ("unrelated", "tool",     "na", 0, "messaging web app"),
    "thirteen.org":          ("unrelated", "homepage", "na", 0, "PBS public media"),
    "tokcomment.com":        ("unrelated", "tool",     "na", 0, "TikTok comment generator"),
    "tomokoshima.com":       ("shopping",  "pdp",      "brand_owned", 1, "salon haircare PDP"),
    "trulyfreehome.com":     ("unrelated", "pdp",      "brand_owned", 0, "home care, not skincare"),
    "venice.ai":             ("unrelated", "tool",     "na", 0, "AI chat app"),
    "verbproducts.com":      ("shopping",  "pdp",      "brand_owned", 1, "haircare PDP"),
    "x.ai":                  ("unrelated", "account",  "na", 0, "API account sign-in"),
    "zara.com":              ("shopping",  "pdp",      "brand_owned", 0, "apparel PDP - structurally shopping, off-topic"),
    "zoho.com":              ("unrelated", "tool",     "na", 0, "webmail compose"),
}

s = pd.read_csv("outputs/sample_for_review.csv", encoding="utf-8")
r = pd.DataFrame(
    [{"domain": d, "review_category": v[0], "review_subtype": v[1],
      "review_seller_type": v[2], "review_relevant": bool(v[3]), "review_note": v[4]}
     for d, v in REVIEW.items()])
r["review_funnel_stage"] = [funnel_stage_for(c, st)
                            for c, st in zip(r.review_category, r.review_subtype)]

m = s.merge(r, on="domain", how="inner", validate="1:1")
m["cat_agree"] = m.page_category == m.review_category
m["sub_agree"] = m.page_subtype == m.review_subtype
m["sell_agree"] = m.seller_type == m.review_seller_type
m["stage_agree"] = m.funnel_stage == m.review_funnel_stage
m["rel_agree"] = (m.skincare_relevance > 0) == m.review_relevant

m.to_parquet("outputs/sample_reviewed.parquet", index=False)

n = len(m)
print(f"=== AGREEMENT on {n} sampled domains ===")
for f, lab in [("cat_agree", "page_category"), ("sub_agree", "page_subtype"),
               ("sell_agree", "seller_type"), ("stage_agree", "funnel_stage"),
               ("rel_agree", "study relevance")]:
    print(f"  {lab:16} {m[f].mean():6.1%}  ({int(m[f].sum())}/{n})")

print("\n=== CATEGORY DISAGREEMENTS ===")
for _, x in m[~m.cat_agree].iterrows():
    print(f"  {x.domain:24.24} model={x.page_category:14.14} review={x.review_category:14.14} "
          f"conf={x.page_category_confidence:.2f} fetch={x.fetch_status:14.14} | {x.review_note}")

print("\n=== SUBTYPE DISAGREEMENTS (category agreed) ===")
for _, x in m[m.cat_agree & ~m.sub_agree].iterrows():
    print(f"  {x.domain:24.24} model={x.page_subtype:12.12} review={x.review_subtype:12.12} | {x.review_note}")

print("\n=== CONFIDENCE vs CORRECTNESS ===")
for lo, hi in [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]:
    b = m[(m.page_category_confidence >= lo) & (m.page_category_confidence < hi)]
    if len(b):
        print(f"  conf {lo:.2f}-{hi:.2f}: n={len(b):2d}  category accuracy {b.cat_agree.mean():5.1%}")

print("\n=== the 'pdp' default suspicion ===")
u = m[m.page_category == "unrelated"]
print(f"  model labels {len(u)} domains 'unrelated'; of those "
      f"{(u.page_subtype == 'pdp').sum()} carry subtype 'pdp'")
print("  reviewer assigns 'pdp' to only "
      f"{(m.review_subtype == 'pdp').sum()} domains overall")
print("\nwrote outputs/sample_reviewed.parquet")
