"""Data augmentation for the conversation database.

Expands the observed conversations with *subtle prompt perturbations* (from
`conveyer.aces.PERTURBATIONS`) while preserving the original schema, so the
augmented rows flow through every existing pipeline stage (intent, funnel,
clustering, brand analysis) unchanged.

Two uses:
1. **Robustness measurement** — do intent/funnel/brand classifiers flip when
   the prompt changes trivially? (they shouldn't; where they do, that's a
   model-fragility finding).
2. **Counterfactual coverage** — the raw DB only contains prompts users
   happened to type. Perturbed variants populate the neighbourhood of each
   observed prompt, which is what the ACES simulator then scores.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .aces import PERTURBATIONS, perturb_prompt


def augment_conversations(df: pd.DataFrame, kinds: Optional[Iterable[str]] = None,
                          question_col: str = "question", n_variants: int = 3,
                          seed: int = 0) -> pd.DataFrame:
    """Return augmented copies of `df` rows with perturbed questions.

    Each original row spawns `n_variants` rows whose `question` is rewritten
    by a randomly drawn perturbation. New rows keep every other column and are
    tagged with `is_augmented=True`, `perturbation`, and a `source_row` index
    pointing back at the original.
    """
    rng = np.random.default_rng(seed)
    kinds = [k for k in (kinds or PERTURBATIONS) if k != "baseline"]
    out = []
    for idx, row in df.iterrows():
        q = str(row[question_col])
        brands = row.get("_brands_q") or row.get("brands_in_question") or []
        brand = str(list(brands)[0]) if len(brands) else "CeraVe"
        for kind in rng.choice(kinds, size=min(n_variants, len(kinds)), replace=False):
            new = row.copy()
            new[question_col] = perturb_prompt(q, str(kind), brand, rng)
            new["is_augmented"] = True
            new["perturbation"] = str(kind)
            new["source_row"] = idx
            out.append(new)
    aug = pd.DataFrame(out).reset_index(drop=True)
    return aug


def with_augmented(df: pd.DataFrame, aug: pd.DataFrame) -> pd.DataFrame:
    """Original + augmented rows in one frame (originals tagged baseline)."""
    base = df.copy()
    base["is_augmented"] = False
    base["perturbation"] = "baseline"
    base["source_row"] = base.index
    return pd.concat([base, aug], ignore_index=True)


def label_flip_report(combined: pd.DataFrame, label_col: str,
                      source_col: str = "source_row") -> pd.DataFrame:
    """How often does a classifier's label flip under each perturbation?

    `combined` must contain original + augmented rows with `label_col` already
    computed on the (possibly perturbed) question text.
    """
    base = (combined[~combined["is_augmented"]]
            .set_index(source_col)[label_col].rename("label_base"))
    aug = combined[combined["is_augmented"]].copy()
    aug = aug.join(base, on=source_col)
    aug["flipped"] = aug[label_col] != aug["label_base"]
    rep = (aug.groupby("perturbation")
              .agg(n=("flipped", "size"), flip_rate=("flipped", "mean"))
              .sort_values("flip_rate", ascending=False))
    return rep.round(3)
