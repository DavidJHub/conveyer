"""The assumption ledger — every number the sales estimate rests on, declared.

The sales bridge (:mod:`conveyer.attribution.bridge`) turns panel behaviour
into euros by multiplying a chain of factors. Most of those factors are *not*
measurable from the clickstream: what share of carts become orders, how many
online-researched purchases complete in a physical store, how much of an
influenced sale would have happened anyway. Hiding them inside code is how a
demo ends up "based on made-up data" without anyone being able to point at
*which* number was made up.

So every factor lives here as an explicit :class:`Factor` carrying

* a central value and a plausible ``[low, high]`` range,
* the **source** it came from, and
* a **status** — the honesty dial:

  ``measured``     computed from our own data (own-shop orders, RMS, CPS);
  ``estimated``    derived from the panel by this pipeline;
  ``benchmark``    an external published figure (industry study, vendor);
  ``assumed``      expert judgement, no data behind it yet;
  ``placeholder``  invented so the machinery runs — **demo only**.

:meth:`Ledger.provenance` reports how much of the answer rests on each status,
and the bridge's tornado chart says which factor to go buy data for first.
Replacing a placeholder is a one-line edit (or a JSON override file) — no code
change, and every downstream number moves with it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

STATUSES = ("measured", "estimated", "benchmark", "assumed", "placeholder")

#: statuses that mean "a real observation stands behind this number"
GROUNDED = ("measured", "estimated")


@dataclass
class Factor:
    """One multiplier in the panel → euros chain."""

    name: str
    value: float
    low: float
    high: float
    unit: str = "ratio"
    status: str = "assumed"
    source: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.name}: unknown status {self.status!r}")
        if not (self.low <= self.value <= self.high):
            raise ValueError(
                f"{self.name}: value {self.value} outside [{self.low}, {self.high}]")

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` Monte Carlo samples.

        Triangular over ``[low, value, high]`` — the honest shape when all we
        have is a best guess and a range: it puts the mode where the estimate
        is, goes to zero at the bounds, and needs no invented variance. A
        degenerate range (low == high) returns the constant.
        """
        if self.high <= self.low:
            return np.full(n, float(self.value))
        return rng.triangular(self.low, self.value, self.high, size=n)


# --------------------------------------------------------------------------- #
# The default ledger
# --------------------------------------------------------------------------- #
# Ranges are deliberately wide where nothing anchors them. Every `placeholder`
# is a task, not a result: see docs/SALES_ATTRIBUTION.md §"filling the ledger"
# for which data source retires each one.
def default_factors() -> List[Factor]:
    return [
        # --- panel → market ------------------------------------------------ #
        Factor("device_coverage", 0.42, 0.30, 0.55, "share", "benchmark",
               "desktop share of consumer LLM usage (vendor/industry reports)",
               "The panel is desktop-only. 1/device_coverage grosses the "
               "observed conversations up to all devices. Retire it when the "
               "provider ships mobile panel coverage."),
        Factor("panel_representativeness", 1.00, 0.80, 1.25, "ratio", "assumed",
               "no demographic weights available yet",
               "Correction for a panel that skews vs the online universe. "
               "Stays 1.0 with a wide band until demographics arrive; then it "
               "becomes a computed post-stratification weight (panel.py) and "
               "the band collapses."),
        # --- observation window ------------------------------------------- #
        Factor("window_recovery", 1.35, 1.10, 1.90, "ratio", "assumed",
               "trail truncation; to be measured on user-level longitudinal data",
               "Purchases that happen after the captured click window (next "
               "N clicks) are invisible. Measurable the moment the provider "
               "gives user-level longitudinal browsing instead of a fixed "
               "post-prompt trail."),
        # --- online conversion --------------------------------------------- #
        Factor("cart_to_order", 0.32, 0.20, 0.45, "share", "benchmark",
               "published desktop retail cart-abandonment benchmarks",
               "The panel outcome is *reaching* cart/checkout, not ordering. "
               "Calibrate against our own shop: panel-observed cart reach on "
               "our D2C domain vs actual orders in the same window."),
        Factor("cross_device_completion", 1.15, 1.00, 1.40, "ratio", "assumed",
               "identity fragmentation (Lin & Misra 2022)",
               "Journeys that start on desktop and finish in a retailer app or "
               "on mobile are counted as non-converting by the panel."),
        # --- online → offline ---------------------------------------------- #
        Factor("ropo_multiplier", 3.20, 1.50, 6.00, "ratio", "placeholder",
               "INVENTED — must come from CPS path-to-purchase or a survey",
               "The big one for CPG: most category volume is bought in store. "
               "This scales an online-influenced purchase to all-channel "
               "purchases influenced by the same conversation. Nothing in the "
               "clickstream can observe it; CPS claimed-source or a bespoke "
               "survey module retires it."),
        # --- value ---------------------------------------------------------- #
        Factor("basket_value", 24.0, 14.0, 38.0, "currency", "placeholder",
               "INVENTED — replace with RMS/Omnisales value per buying occasion",
               "Euro value of the brand's products in one influenced purchase "
               "occasion. RMS gives value/units by brand-market-period; CPS "
               "gives buying occasions. This is a lookup, not an assumption, "
               "as soon as the brand↔RMS hierarchy mapping exists."),
        # --- attribution quality -------------------------------------------- #
        Factor("brand_match_precision", 0.85, 0.70, 0.95, "share", "estimated",
               "brand lexicon + product-match tiers (module 2 self-evaluation)",
               "Share of brand↔page attributions that are correct. Measured on "
               "a hand-labelled sample of real pages; the synthetic corpus "
               "scores 1.0 and must not be quoted as the real value."),
        # --- influence → incremental ---------------------------------------- #
        Factor("incrementality", 0.25, 0.10, 0.45, "share", "placeholder",
               "INVENTED — this is exactly what the MMM must replace",
               "Share of AI-influenced sales that would not have happened "
               "without the AI conversation. The panel cannot identify it "
               "(exposure is self-selected). Sources, best first: MMM "
               "coefficient on the AI variable, geo/content experiments, "
               "matched-control panel lift (an upper bound)."),
        # --- top-down route -------------------------------------------------- #
        Factor("influence_given_exposure", 0.18, 0.06, 0.35, "share", "placeholder",
               "INVENTED — survey-calibratable ('did the AI answer affect what "
               "you bought?')",
               "Used only by the top-down cross-check: P(purchase choice was "
               "influenced | the user had an AI conversation about the "
               "category and was exposed to the brand)."),
    ]


class Ledger:
    """A named set of factors, with overrides, provenance and joint draws."""

    def __init__(self, factors: Optional[Iterable[Factor]] = None) -> None:
        self._f: Dict[str, Factor] = {}
        for f in (factors if factors is not None else default_factors()):
            self._f[f.name] = f

    # -- access ------------------------------------------------------------ #
    def __contains__(self, name: str) -> bool:
        return name in self._f

    def __getitem__(self, name: str) -> Factor:
        return self._f[name]

    def __len__(self) -> int:
        return len(self._f)

    @property
    def names(self) -> List[str]:
        return list(self._f)

    def value(self, name: str) -> float:
        return float(self._f[name].value)

    def set(self, name: str, **changes) -> "Ledger":
        """Replace fields of one factor in place (returns self, chainable).

        This is the supported way to promote a placeholder: pass the new
        value/range *and* the status and source that justify it, e.g.::

            ledger.set("cart_to_order", value=0.29, low=0.26, high=0.33,
                       status="measured", source="own-shop orders 2026-H1")
        """
        cur = asdict(self._f[name])
        cur.update(changes)
        self._f[name] = Factor(**cur)
        return self

    def add(self, factor: Factor) -> "Ledger":
        self._f[factor.name] = factor
        return self

    # -- persistence -------------------------------------------------------- #
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self._f.values()])

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(f) for f in self._f.values()], fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Ledger":
        """Load a full ledger, or overlay a partial override file on the
        defaults — a file holding just ``[{"name": "ropo_multiplier", ...}]``
        updates that factor and leaves the rest alone."""
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        led = cls()
        for row in rows:
            name = row["name"]
            if name in led and set(row) < set(asdict(led[name])):
                led.set(name, **{k: v for k, v in row.items() if k != "name"})
            else:
                led.add(Factor(**row))
        return led

    # -- Monte Carlo -------------------------------------------------------- #
    def draw(self, n: int, seed: int = 0) -> pd.DataFrame:
        """``n`` joint draws, one column per factor.

        Factors are drawn **independently**: no dependence structure is known,
        and inventing a correlation matrix would be one more undeclared
        assumption. The consequence — the chain's variance is the sum of
        independent variances — is stated in the docs.
        """
        rng = np.random.default_rng(seed)
        return pd.DataFrame({name: f.draw(rng, n) for name, f in self._f.items()})

    # -- honesty ------------------------------------------------------------ #
    def provenance(self) -> pd.DataFrame:
        """Factor count and mean relative width of the range, by status."""
        df = self.to_frame()
        if df.empty:
            return df
        df["rel_width"] = (df["high"] - df["low"]) / df["value"].replace(0, np.nan)
        out = (df.groupby("status")
                 .agg(n_factors=("name", "size"),
                      factors=("name", lambda s: ", ".join(sorted(s))),
                      mean_rel_width=("rel_width", "mean"))
                 .reset_index())
        order = {s: i for i, s in enumerate(STATUSES)}
        out["_o"] = out["status"].map(order)
        return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    def placeholders(self) -> List[str]:
        return [n for n, f in self._f.items() if f.status == "placeholder"]

    def headline(self) -> str:
        """One line for the top of any report built on this ledger."""
        ph = self.placeholders()
        grounded = sum(1 for f in self._f.values() if f.status in GROUNDED)
        if not ph:
            return (f"Ledger: {len(self)} factors, {grounded} grounded in our own "
                    f"data, no placeholders.")
        return (f"Ledger: {len(self)} factors, {grounded} grounded in our own data, "
                f"{len(ph)} PLACEHOLDER ({', '.join(ph)}) — the estimate is a "
                f"structure with declared assumptions, not a measurement.")


def default_ledger() -> Ledger:
    return Ledger()
