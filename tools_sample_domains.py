"""Draw a reproducible random sample of DOMAINS from the demo run for
independent re-tagging, and emit the model's own verdict alongside so the two
can be compared afterwards. One row per domain (the most-surfaced URL of that
domain represents it)."""
import pandas as pd

SEED = 7
N = 45

d = pd.read_parquet("outputs/scrape_demo/scraped_pages.parquet")

# one representative URL per domain: the most-surfaced, tie-broken by url so
# the draw is deterministic across machines
d = d.sort_values(["times_surfaced", "url"], ascending=[False, True])
rep = d.drop_duplicates("domain", keep="first")

# stratify by the model's category so every label is represented rather than
# drawing 45 rows that are 53% unrelated/unknown
frac = N / len(rep)
parts = [g.sample(max(1, round(len(g) * frac)), random_state=SEED)
         for _, g in rep.groupby("page_category")]
samp = pd.concat(parts)
if len(samp) > N:
    samp = samp.sample(N, random_state=SEED)
samp = samp.sort_values("domain")

cols = ["domain", "url", "title", "page_category", "page_subtype", "seller_type",
        "funnel_stage", "page_category_confidence", "fetch_status", "fetch_scope",
        "classifier_method", "skincare_relevance"]
out = samp[cols].reset_index(drop=True)
out.to_csv("outputs/sample_for_review.csv", index=False, encoding="utf-8")
print(f"{len(out)} domains sampled from {len(rep)} distinct domains "
      f"({len(d)} page rows)")
print(out["page_category"].value_counts().to_string())
print("\n--- domains ---")
for i, r in out.iterrows():
    print(f"{i:2d}  {r.domain:32.32}  {str(r.title)[:44]:44.44}  "
          f"{r.page_category}/{r.page_subtype}")
