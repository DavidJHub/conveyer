"""Canonical skincare brand lexicon — the single source of truth.

Both sides of the funnel need to agree on what a "brand" is:

* **conversations** (module 1) extracts brand mentions from question/answer
  text — the *exposure* side ("which brands did the agent put in front of the
  user?");
* **scraping** (module 2) recognises brand-owned storefronts by domain and
  matches page products back to the chat — the *behaviour* side ("which
  brand's page did the user actually visit?").

Keeping one lexicon here means a mention in text and a visit to the brand's
site resolve to the same canonical key, which is what module 3 (``journey``)
joins on.

Each entry: canonical name → text aliases (safe for word-boundary matching in
prose) and web domain cores (the registrable-domain label, e.g. ``cerave`` for
``cerave.com``). Aliases that are common English words ("Fresh", "Simple",
"Hero", "Bubble", "Origins") are deliberately *excluded* from text matching
unless qualified ("hero cosmetics") — a false brand mention poisons the
exposure features downstream.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set

# canonical -> {"aliases": text-matchable strings, "domains": registrable cores}
BRANDS: Dict[str, dict] = {
    "CeraVe": {"aliases": ["cerave"], "domains": ["cerave"]},
    "The Ordinary": {"aliases": ["the ordinary"], "domains": ["theordinary", "deciem"]},
    "La Roche-Posay": {"aliases": ["la roche-posay", "la roche posay"],
                       "domains": ["laroche-posay", "larocheposay"]},
    "Cetaphil": {"aliases": ["cetaphil"], "domains": ["cetaphil"]},
    "Neutrogena": {"aliases": ["neutrogena"], "domains": ["neutrogena"]},
    "Olay": {"aliases": ["olay"], "domains": ["olay"]},
    "Paula's Choice": {"aliases": ["paula's choice", "paulas choice"], "domains": ["paulaschoice"]},
    "Drunk Elephant": {"aliases": ["drunk elephant"], "domains": ["drunkelephant"]},
    "The Inkey List": {"aliases": ["the inkey list", "inkey list"], "domains": ["theinkeylist"]},
    "Glossier": {"aliases": ["glossier"], "domains": ["glossier"]},
    "Kiehl's": {"aliases": ["kiehl's", "kiehls"], "domains": ["kiehls"]},
    "Clinique": {"aliases": ["clinique"], "domains": ["clinique"]},
    "Estée Lauder": {"aliases": ["estee lauder", "estée lauder"], "domains": ["esteelauder"]},
    "Tatcha": {"aliases": ["tatcha"], "domains": ["tatcha"]},
    "SkinCeuticals": {"aliases": ["skinceuticals"], "domains": ["skinceuticals"]},
    "Murad": {"aliases": ["murad"], "domains": ["murad"]},
    "EltaMD": {"aliases": ["eltamd", "elta md"], "domains": ["eltamd"]},
    "ISDIN": {"aliases": ["isdin"], "domains": ["isdin"]},
    "Bioderma": {"aliases": ["bioderma"], "domains": ["bioderma"]},
    "Avène": {"aliases": ["avene", "avène"], "domains": ["avene"]},
    "Vichy": {"aliases": ["vichy"], "domains": ["vichy"]},
    "Eucerin": {"aliases": ["eucerin"], "domains": ["eucerin"]},
    "Aveeno": {"aliases": ["aveeno"], "domains": ["aveeno"]},
    "Nivea": {"aliases": ["nivea"], "domains": ["nivea"]},
    "Garnier": {"aliases": ["garnier"], "domains": ["garnier"]},
    "L'Oréal": {"aliases": ["l'oreal", "loreal", "l'oréal"], "domains": ["loreal", "lorealparisusa"]},
    "Innisfree": {"aliases": ["innisfree"], "domains": ["innisfree"]},
    "COSRX": {"aliases": ["cosrx"], "domains": ["cosrx"]},
    "Laneige": {"aliases": ["laneige"], "domains": ["laneige"]},
    "Beauty of Joseon": {"aliases": ["beauty of joseon"], "domains": ["beautyofjoseon"]},
    "Supergoop": {"aliases": ["supergoop"], "domains": ["supergoop"]},
    "First Aid Beauty": {"aliases": ["first aid beauty"], "domains": ["firstaidbeauty"]},
    "Youth to the People": {"aliases": ["youth to the people"], "domains": ["youthtothepeople"]},
    "Good Molecules": {"aliases": ["good molecules"], "domains": ["goodmolecules"]},
    "Krave Beauty": {"aliases": ["krave beauty", "kravebeauty"], "domains": ["krave", "kravebeauty"]},
    "Farmacy": {"aliases": ["farmacy"], "domains": ["farmacybeauty"]},
    "Biossance": {"aliases": ["biossance"], "domains": ["biossance"]},
    "Summer Fridays": {"aliases": ["summer fridays"], "domains": ["summerfridays"]},
    "Versed": {"aliases": ["versed"], "domains": ["versedskin"]},
    "Cocokind": {"aliases": ["cocokind"], "domains": ["cocokind"]},
    "Naturium": {"aliases": ["naturium"], "domains": ["naturium"]},
    "Byoma": {"aliases": ["byoma"], "domains": ["byoma"]},
    "Hero Cosmetics": {"aliases": ["hero cosmetics", "mighty patch"], "domains": ["herocosmetics"]},
    "Tower 28": {"aliases": ["tower 28"], "domains": ["tower28beauty"]},
    "Dr. Jart+": {"aliases": ["dr. jart", "dr jart"], "domains": ["drjart"]},
    "Medicube": {"aliases": ["medicube"], "domains": ["medicube"]},
    "Anua": {"aliases": ["anua"], "domains": ["anua"]},
    "Skin1004": {"aliases": ["skin1004"], "domains": ["skin1004"]},
    "Torriden": {"aliases": ["torriden"], "domains": ["torriden"]},
    "Round Lab": {"aliases": ["round lab", "roundlab"], "domains": ["roundlab"]},
    "Shiseido": {"aliases": ["shiseido"], "domains": ["shiseido"]},
    "SK-II": {"aliases": ["sk-ii", "sk2"], "domains": ["sk-ii"]},
    "La Mer": {"aliases": ["la mer"], "domains": ["lamer"]},
    "Sunday Riley": {"aliases": ["sunday riley"], "domains": ["sundayriley"]},
    "Glow Recipe": {"aliases": ["glow recipe"], "domains": ["glowrecipe"]},
    "Topicals": {"aliases": ["topicals"], "domains": ["mytopicals"]},
    "Bubble Skincare": {"aliases": ["bubble skincare"], "domains": ["hellobubble"]},
    "e.l.f.": {"aliases": ["e.l.f."], "domains": ["elfcosmetics"]},
    "RoC": {"aliases": ["roc skincare"], "domains": ["rocskincare"]},
    "Vanicream": {"aliases": ["vanicream"], "domains": ["vanicream"]},
    "Stratia": {"aliases": ["stratia"], "domains": ["stratiaskin"]},
}

# --------------------------------------------------------------------------- #
# Derived lookups
# --------------------------------------------------------------------------- #
BRAND_DOMAIN_CORES: Set[str] = {d for spec in BRANDS.values() for d in spec["domains"]}

_DOMAIN_TO_CANONICAL: Dict[str, str] = {
    d: name for name, spec in BRANDS.items() for d in spec["domains"]}

_ALIAS_TO_CANONICAL: Dict[str, str] = {
    a: name for name, spec in BRANDS.items() for a in spec["aliases"]}

# one compiled alternation, longest aliases first so "the ordinary" beats "ordinary"
_ALIAS_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in
                      sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_brand(name: str) -> str:
    """Canonical form for a raw brand string (falls back to title-case)."""
    key = str(name).strip().lower()
    return _ALIAS_TO_CANONICAL.get(key) or _DOMAIN_TO_CANONICAL.get(key) or str(name).strip()


def extract_brands(text: str) -> Set[str]:
    """Canonical brand names mentioned in free text (word-boundary matches)."""
    if not text:
        return set()
    return {_ALIAS_TO_CANONICAL[m.group(1).lower()]
            for m in _ALIAS_RE.finditer(str(text))
            if m.group(1).lower() in _ALIAS_TO_CANONICAL}


def brand_for_domain(core: str) -> str:
    """Canonical brand for a registrable-domain core ('' when unknown)."""
    return _DOMAIN_TO_CANONICAL.get(str(core).strip().lower(), "")


def brands_in_any(texts: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for t in texts:
        out |= extract_brands(t)
    return out


def all_brand_names() -> List[str]:
    return sorted(BRANDS)
