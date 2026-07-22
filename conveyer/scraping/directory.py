"""Domain directory — offline knowledge about what a *site* is, keyed by domain.

The politeness rules mean some URLs can never be fetched: ``robots.txt``
disallows them (``fetch_status="robots_blocked"``), the host rejects bots
(HTTP 403), or it is simply dead. The page is off-limits — but *what the site
is* is public knowledge. This module is that knowledge, a small offline web
directory::

    >>> lookup("https://www.sephora.com/shop/moisturizer")
    DomainEntry(domain='sephora', name='Sephora', role='retailer',
                category='beauty_and_cosmetics', description='Specialty beauty
                retailer selling skincare, makeup, ...', source='seed')

An entry supplies three things to the classifier:

* a **description** that stands in for page text (``fetch_scope="directory"``)
  so the relevance scorer has words to read even when nothing was fetched;
* a **role** (retailer / marketplace / brand / search / community / editorial /
  reference) that votes like the curated domain lists do — for domains those
  lists don't know;
* a **site category** (e.g. ``beauty_and_cosmetics``) that acts like the
  SimilarWeb ``dim_digital_site.category`` prior.

Two layers, merged at load time:

1. a built-in **seed**, generated from the curated domain lists in
   :mod:`conveyer.scraping.classify` (every domain the classifier already
   recognises also gets a description), with hand-written entries for the
   highest-traffic sites;
2. an optional **external JSON file** (``ScrapeConfig.directory_path``,
   default ``data/domain_directory.json``) for the long tail. File entries
   win over the seed, so corrections and additions live in data, not code.

External file format — a mapping (or a list of objects with a ``domain`` key);
keys may be a bare core (``"sephora"``), a domain (``"sephora.com"``) or a
full URL::

    {
      "strawberrynet": {
        "name": "StrawberryNet",
        "role": "retailer",
        "category": "beauty_and_cosmetics",
        "description": "Online beauty retailer selling skincare and makeup."
      }
    }

Descriptions are written with intent: only genuinely beauty-focused sites
mention skincare (that wording is what earns topical relevance downstream);
broad marketplaces get deliberately generic descriptions so an off-topic
product on amazon never looks study-relevant by association.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .extract import PageContent

ROLES = ("marketplace", "retailer", "brand", "search", "community",
         "editorial", "reference", "other")

_ROLE_PHRASES = {
    "marketplace": "online marketplace",
    "retailer": "online retailer",
    "brand": "official brand site",
    "search": "search engine",
    "community": "community platform",
    "editorial": "online publisher",
    "reference": "reference site",
    "other": "website",
}


@dataclass(frozen=True)
class DomainEntry:
    domain: str               # registrable core, e.g. "sephora"
    name: str = ""
    role: str = "other"       # one of ROLES
    category: str = ""        # site-category prior, e.g. "beauty_and_cosmetics"
    description: str = ""
    source: str = "seed"      # seed | file


def _core_of(url_or_domain: str) -> str:
    """Normalise any key ('sephora', 'sephora.com', a full URL) to the core."""
    from .classify import parse_url
    s = str(url_or_domain or "").strip().lower()
    if not s:
        return ""
    if "/" not in s and "." not in s:
        return s
    return parse_url(s).core


# --------------------------------------------------------------------------- #
# Built-in seed
# --------------------------------------------------------------------------- #
# Retailers that are genuinely beauty-focused: their descriptions may claim
# skincare (and therefore confer topical relevance to blocked pages). The rest
# of RETAILER_DOMAINS (mass merchants, pharmacies, department stores) stay
# deliberately generic.
_BEAUTY_RETAILERS = {
    "sephora", "ulta", "dermstore", "lookfantastic", "cultbeauty", "spacenk",
    "bluemercury", "skinstore", "beautylish", "credobeauty", "thirteenlune",
    "sokoglam", "soko-glam", "stylevana", "yesstyle", "iherb",
}

# Hand-written entries for the highest-traffic domains; everything else in the
# curated lists gets a templated description below. Overrides win.
_OVERRIDES: Dict[str, dict] = {
    "amazon": dict(name="Amazon", role="marketplace", description=(
        "Global online marketplace selling products in every category, from "
        "electronics to groceries; beauty and personal care is one department "
        "among many. Product, cart and checkout pages live under /dp/, "
        "/gp/cart and /gp/buy paths.")),
    "walmart": dict(name="Walmart", role="marketplace", description=(
        "Mass-market retailer and online marketplace selling groceries, "
        "household goods, electronics and personal care.")),
    "target": dict(name="Target", role="marketplace", description=(
        "General-merchandise retailer selling household goods, groceries, "
        "apparel, electronics and personal care.")),
    "ebay": dict(name="eBay", role="marketplace", description=(
        "Consumer-to-consumer auction and shopping marketplace for new and "
        "used goods in every category.")),
    "etsy": dict(name="Etsy", role="marketplace", description=(
        "Marketplace for handmade, vintage and craft goods from independent "
        "sellers.")),
    "sephora": dict(name="Sephora", role="retailer",
                    category="beauty_and_cosmetics", description=(
        "Specialty beauty retailer selling skincare, makeup, haircare and "
        "fragrance from hundreds of brands, with product pages, ratings and "
        "reviews.")),
    "ulta": dict(name="Ulta Beauty", role="retailer",
                 category="beauty_and_cosmetics", description=(
        "Beauty retailer selling skincare, cosmetics, haircare and fragrance "
        "across drugstore and prestige brands.")),
    "cvs": dict(name="CVS Pharmacy", role="retailer", description=(
        "Drugstore chain selling medicine, health, beauty and personal-care "
        "products.")),
    "walgreens": dict(name="Walgreens", role="retailer", description=(
        "Drugstore chain selling medicine, health, beauty and personal-care "
        "products.")),
    "google": dict(name="Google", role="search", description=(
        "Web search engine; a results page lists links matching the user's "
        "query, on any topic.")),
    "bing": dict(name="Bing", role="search", description=(
        "Microsoft's web search engine; results pages list links matching "
        "the query.")),
    "perplexity": dict(name="Perplexity", role="search", description=(
        "AI answer engine that responds to queries with cited summaries and "
        "links.")),
    "reddit": dict(name="Reddit", role="community", description=(
        "Social news and discussion forum organised into topical communities "
        "(subreddits) of user posts and comment threads.")),
    "youtube": dict(name="YouTube", role="community", description=(
        "Video-sharing platform of user-uploaded videos, reviews, tutorials "
        "and vlogs on any topic.")),
    "tiktok": dict(name="TikTok", role="community", description=(
        "Short-form video platform of user-generated clips, trends and "
        "product reviews.")),
    "instagram": dict(name="Instagram", role="community", description=(
        "Photo and video social network of user posts, stories and reels.")),
    "pinterest": dict(name="Pinterest", role="community", description=(
        "Visual discovery board where users save and share images and ideas.")),
    "makeupalley": dict(name="MakeupAlley", role="community",
                        category="beauty_and_cosmetics", description=(
        "Beauty product review community: member-written skincare and makeup "
        "reviews and boards.")),
    "beautypedia": dict(name="Beautypedia", role="community",
                        category="beauty_and_cosmetics", description=(
        "Skincare and makeup product review site rating formulas and "
        "ingredients.")),
    "acne": dict(name="Acne.org", role="community",
                 category="beauty_and_cosmetics", description=(
        "Community and reference site about acne: treatment regimens, "
        "skincare routines and member discussions.")),
    "wikipedia": dict(name="Wikipedia", role="reference", description=(
        "Free crowd-written encyclopedia; articles explain topics neutrally "
        "with references.")),
    "webmd": dict(name="WebMD", role="reference", description=(
        "Consumer health information site: conditions, symptoms, drugs and "
        "wellness articles reviewed by physicians.")),
    "healthline": dict(name="Healthline", role="reference", description=(
        "Consumer health publisher: medically reviewed articles on "
        "conditions, nutrition and wellness.")),
    "mayoclinic": dict(name="Mayo Clinic", role="reference", description=(
        "Hospital-system health reference: diseases, symptoms, tests and "
        "treatments explained by clinicians.")),
    "byrdie": dict(name="Byrdie", role="editorial",
                   category="beauty_and_cosmetics", description=(
        "Beauty editorial site publishing skincare and makeup reviews, "
        "how-tos and 'best of' buying guides.")),
    "allure": dict(name="Allure", role="editorial",
                   category="beauty_and_cosmetics", description=(
        "Beauty magazine publishing skincare and cosmetics news, reviews and "
        "award roundups.")),
    "nytimes": dict(name="The New York Times / Wirecutter", role="editorial",
                    description=(
        "News publisher; its Wirecutter section publishes tested product "
        "recommendations and buying guides.")),
}


def _template(core: str, role: str, beauty: bool = False) -> dict:
    name = core.replace("-", " ").title()
    if role == "marketplace":
        desc = (f"{name} is an online marketplace / mass retailer selling "
                f"products across many categories.")
    elif role == "retailer" and beauty:
        return dict(name=name, role=role, category="beauty_and_cosmetics",
                    description=(f"{name} is a specialty beauty retailer "
                                 f"selling skincare, makeup and fragrance "
                                 f"from many brands."))
    elif role == "retailer":
        desc = (f"{name} is a retail chain selling products across categories "
                f"such as health, beauty, household and apparel.")
    elif role == "brand":
        return dict(name=name, role=role, category="beauty_and_cosmetics",
                    description=(f"{name} is the official site and storefront "
                                 f"of the {name} skincare brand — shop its "
                                 f"cleansers, serums, moisturizers and "
                                 f"sunscreens."))
    elif role == "search":
        desc = (f"{name} is a web search engine; results pages list links "
                f"matching the user's query.")
    elif role == "community":
        desc = (f"{name} is a social / community platform of user-generated "
                f"posts and discussions on any topic.")
    elif role == "editorial":
        desc = (f"{name} is an online publisher of articles, news, reviews "
                f"and buying guides.")
    elif role == "reference":
        desc = (f"{name} is a reference site with encyclopedic, medical or "
                f"how-to information.")
    else:
        desc = f"{name} is a website."
    return dict(name=name, role=role, category="", description=desc)


def _build_seed() -> Dict[str, DomainEntry]:
    # imported inside the function so this module stays import-light and free
    # of any import-cycle risk (classify never imports directory at runtime)
    from .classify import (BRAND_DOMAINS, COMMUNITY_DOMAINS, EDITORIAL_DOMAINS,
                           MARKETPLACE_DOMAINS, REFERENCE_DOMAINS,
                           RETAILER_DOMAINS, SEARCH_DOMAINS)

    spec: Dict[str, dict] = {}
    for core in MARKETPLACE_DOMAINS:
        spec[core] = _template(core, "marketplace")
    for core in RETAILER_DOMAINS - MARKETPLACE_DOMAINS:
        spec[core] = _template(core, "retailer", beauty=core in _BEAUTY_RETAILERS)
    for core in SEARCH_DOMAINS:
        spec[core] = _template(core, "search")
    for core in COMMUNITY_DOMAINS:
        spec[core] = _template(core, "community")
    for core in EDITORIAL_DOMAINS:
        spec[core] = _template(core, "editorial")
    for core in REFERENCE_DOMAINS:
        spec[core] = _template(core, "reference")
    for core in BRAND_DOMAINS:
        spec[core] = _template(core, "brand")
    for core, d in _OVERRIDES.items():
        spec[core] = {**spec.get(core, {}), **d}

    return {core: DomainEntry(domain=core, name=d.get("name", ""),
                              role=d.get("role", "other"),
                              category=d.get("category", ""),
                              description=d.get("description", ""),
                              source="seed")
            for core, d in spec.items()}


# --------------------------------------------------------------------------- #
# Loading (seed + optional external file), cached
# --------------------------------------------------------------------------- #
_seed_cache: Optional[Dict[str, DomainEntry]] = None
_file_cache: Dict[str, Tuple[float, Dict[str, DomainEntry]]] = {}
_warned_paths: set = set()


def _load_file(path: str) -> Dict[str, DomainEntry]:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    cached = _file_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    entries: Dict[str, DomainEntry] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.items() if isinstance(data, dict) else \
            ((d.get("domain", ""), d) for d in data if isinstance(d, dict))
        for key, d in items:
            if not isinstance(d, dict):
                continue
            core = _core_of(str(d.get("domain") or key))
            if not core:
                continue
            role = str(d.get("role", "other")).strip().lower()
            entries[core] = DomainEntry(
                domain=core, name=str(d.get("name", "")),
                role=role if role in ROLES else "other",
                category=str(d.get("category", "")).strip().lower(),
                description=str(d.get("description", "")).strip(),
                source="file")
    except Exception as exc:
        if path not in _warned_paths:
            _warned_paths.add(path)
            print(f"[directory] could not read {path}: {type(exc).__name__}: {exc}")
        return {}
    _file_cache[path] = (mtime, entries)
    return entries


def load_directory(path: Optional[str] = None) -> Dict[str, DomainEntry]:
    """The merged directory: built-in seed, overlaid by ``path`` if it exists."""
    global _seed_cache
    if _seed_cache is None:
        _seed_cache = _build_seed()
    if not path:
        return _seed_cache
    file_entries = _load_file(path)
    if not file_entries:
        return _seed_cache
    merged = dict(_seed_cache)
    merged.update(file_entries)
    return merged


def lookup(url_or_domain: str, path: Optional[str] = None) -> Optional[DomainEntry]:
    """The directory entry for a URL/domain's registrable core, or None."""
    core = _core_of(url_or_domain)
    if not core:
        return None
    return load_directory(path).get(core)


def directory_content(entry: DomainEntry, url: str = "") -> PageContent:
    """A :class:`PageContent` stand-in built from a directory entry — what the
    classifier reads when the page itself is off-limits. ``parser="directory"``
    marks the provenance in the output table."""
    title = f"{entry.name or entry.domain} — {_ROLE_PHRASES.get(entry.role, 'website')}"
    return PageContent(url=url, title=title, meta_description=entry.description,
                       text=entry.description, parser="directory")
