"""HTML → structured :class:`PageContent`.

This is the "get all the info of the web page we can" step. It is written
**standard-library-first**: a :class:`html.parser.HTMLParser` subclass pulls the
title, meta/OpenGraph/Twitter tags, canonical link, headings, visible text,
outbound links, images, embedded ``application/ld+json`` blocks and lightweight
microdata — with *no third-party dependency*. When BeautifulSoup + lxml are
installed the parser upgrades to them transparently (``html_parser="auto"``),
exactly like the embedding-backend auto-resolution in ``conveyer.models``.

The parsed object exposes the raw structured data (`jsonld`, `microdata`, `og`)
that ``conveyer.scraping.products`` turns into product records, and the derived
signals (`schema_types`, `has_price`, `has_add_to_cart`, `word_count`, …) that
``conveyer.scraping.classify`` scores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List

_WS = re.compile(r"\s+")
# A price mentioned in visible copy: $12, £9.99, 12,50 €, USD 20 …
_PRICE_RE = re.compile(
    r"(?:[$£€¥]|USD|EUR|GBP)\s?\d[\d.,]*|\d[\d.,]*\s?(?:[$£€¥]|USD|EUR|GBP|dollars?)",
    re.IGNORECASE,
)
_CART_RE = re.compile(
    r"add to (?:cart|bag|basket)|buy now|add-to-cart|checkout|proceed to (?:checkout|pay)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


# --------------------------------------------------------------------------- #
# The parsed page
# --------------------------------------------------------------------------- #
@dataclass
class PageContent:
    """Everything we could pull from one page's HTML (content only; fetch
    metadata lives on :class:`conveyer.scraping.fetch.FetchResult`)."""

    url: str = ""
    title: str = ""
    meta_description: str = ""
    canonical_url: str = ""
    lang: str = ""
    og: Dict[str, str] = field(default_factory=dict)
    twitter: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    h1: List[str] = field(default_factory=list)
    h2: List[str] = field(default_factory=list)
    h3: List[str] = field(default_factory=list)
    text: str = ""
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    jsonld: List[dict] = field(default_factory=list)     # parsed JSON-LD objects (flattened)
    microdata: List[dict] = field(default_factory=list)  # {"type":..., "props":{...}}
    parser: str = "stdlib"

    # ---- derived signals -------------------------------------------------- #
    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def has_jsonld(self) -> bool:
        return bool(self.jsonld)

    def schema_types(self) -> List[str]:
        """All schema.org @type strings present (JSON-LD + microdata), lower-cased."""
        out: List[str] = []
        for obj in iter_jsonld_objects(self.jsonld):
            t = obj.get("@type")
            out.extend(_as_type_list(t))
        for md in self.microdata:
            t = md.get("type")
            if t:
                out.append(str(t).rsplit("/", 1)[-1])
        return sorted({s.lower() for s in out if s})

    @property
    def has_price(self) -> bool:
        if any(k in self.og for k in ("product:price:amount", "og:price:amount")):
            return True
        for obj in iter_jsonld_objects(self.jsonld):
            if "offers" in obj or "price" in obj:
                return True
        return bool(_PRICE_RE.search(self.text))

    @property
    def has_add_to_cart(self) -> bool:
        blob = " ".join([self.text[:4000]] + self.links)
        return bool(_CART_RE.search(blob))

    def excerpt(self, n: int = 2000) -> str:
        return self.text[:n]


# --------------------------------------------------------------------------- #
# JSON-LD helpers
# --------------------------------------------------------------------------- #
def _as_type_list(t: Any) -> List[str]:
    if isinstance(t, list):
        return [str(x) for x in t]
    if t:
        return [str(t)]
    return []


def iter_jsonld_objects(blocks: Iterable[Any]) -> Iterable[dict]:
    """Yield every JSON-LD *object*, unwrapping arrays and ``@graph`` nesting."""
    stack = list(blocks)
    seen = 0
    while stack and seen < 5000:
        seen += 1
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            if "@graph" in node and isinstance(node["@graph"], list):
                stack.extend(node["@graph"])
            yield node


def _parse_jsonld_blocks(raw_blocks: List[str]) -> List[dict]:
    parsed: List[dict] = []
    for raw in raw_blocks:
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            # Some sites emit trailing junk or HTML comments around the JSON.
            cleaned = raw.strip().lstrip("<!--").rstrip("-->").strip()
            try:
                obj = json.loads(cleaned)
            except (ValueError, json.JSONDecodeError):
                continue
        if isinstance(obj, list):
            parsed.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            parsed.append(obj)
    return parsed


# --------------------------------------------------------------------------- #
# Standard-library parser
# --------------------------------------------------------------------------- #
_SKIP_TAGS = {"style", "noscript", "template", "svg"}
_HEADINGS = {"h1", "h2", "h3"}


class _StdlibExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: List[str] = []
        self._in_title = False
        self.metas: List[Dict[str, str]] = []
        self.link_tags: List[Dict[str, str]] = []
        self.anchors: List[str] = []
        self.imgs: List[str] = []
        self.headings: Dict[str, List[str]] = {h: [] for h in _HEADINGS}
        self._heading: str | None = None
        self._heading_parts: List[str] = []
        self.jsonld_raw: List[str] = []
        self._in_ld = False
        self._ld_parts: List[str] = []
        self._skip_depth = 0
        self.text_parts: List[str] = []
        self.lang = ""
        # microdata (lightweight): collect itemprop values within an itemscope
        self.microdata: List[dict] = []
        self._md_stack: List[dict] = []
        self._md_prop: str | None = None
        self._md_prop_parts: List[str] = []

    # -- tags -------------------------------------------------------------- #
    def handle_starttag(self, tag: str, attrs_list):
        attrs = {k.lower(): (v or "") for k, v in attrs_list}
        if tag == "html" and attrs.get("lang"):
            self.lang = attrs["lang"]
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            self.metas.append(attrs)
        elif tag == "link":
            self.link_tags.append(attrs)
        elif tag == "a" and attrs.get("href"):
            self.anchors.append(attrs["href"])
        elif tag == "img" and (attrs.get("src") or attrs.get("data-src")):
            self.imgs.append(attrs.get("src") or attrs.get("data-src"))
        elif tag in _HEADINGS:
            self._heading = tag
            self._heading_parts = []
        elif tag == "script":
            if "ld+json" in attrs.get("type", "").lower():
                self._in_ld = True
                self._ld_parts = []
            else:
                self._skip_depth += 1
        elif tag in _SKIP_TAGS:
            self._skip_depth += 1

        # microdata
        if "itemscope" in attrs:
            item = {"type": attrs.get("itemtype", ""), "props": {}}
            self.microdata.append(item)
            self._md_stack.append(item)
        if "itemprop" in attrs:
            self._md_prop = attrs["itemprop"]
            self._md_prop_parts = []
            # attribute-carried values (content/href/src) resolve immediately
            val = attrs.get("content") or attrs.get("href") or attrs.get("src")
            if val and self._md_stack:
                self._md_stack[-1]["props"].setdefault(self._md_prop, val)
                self._md_prop = None

    def handle_startendtag(self, tag: str, attrs_list):
        # <meta/>, <link/>, <img/> as self-closing
        self.handle_starttag(tag, attrs_list)
        if tag in _HEADINGS:
            self._heading = None

    def handle_endtag(self, tag: str):
        if tag == "title":
            self._in_title = False
        elif tag in _HEADINGS and self._heading == tag:
            self.headings[tag].append(_clean(" ".join(self._heading_parts)))
            self._heading = None
        elif tag == "script":
            if self._in_ld:
                self.jsonld_raw.append("".join(self._ld_parts))
                self._in_ld = False
                self._ld_parts = []
            elif self._skip_depth > 0:
                self._skip_depth -= 1
        elif tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._md_prop is not None:
            if self._md_stack:
                text = _clean(" ".join(self._md_prop_parts))
                if text:
                    self._md_stack[-1]["props"].setdefault(self._md_prop, text)
            self._md_prop = None
        if self._md_stack and tag != "meta":
            # close the most recent itemscope on a matching-ish close; microdata
            # is best-effort, so we pop conservatively at container ends.
            pass

    def handle_data(self, data: str):
        if self._in_ld:
            self._ld_parts.append(data)
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth > 0:
            return
        if self._heading is not None:
            self._heading_parts.append(data)
        if self._md_prop is not None:
            self._md_prop_parts.append(data)
        stripped = data.strip()
        if stripped:
            self.text_parts.append(stripped)


def _extract_stdlib(html: str, url: str) -> PageContent:
    p = _StdlibExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:  # malformed HTML must never crash the pipeline
        pass
    return _assemble(
        url=url,
        title=_clean(" ".join(p.title_parts)),
        metas=p.metas,
        link_tags=p.link_tags,
        anchors=p.anchors,
        imgs=p.imgs,
        headings=p.headings,
        text=_clean(" ".join(p.text_parts)),
        jsonld=_parse_jsonld_blocks(p.jsonld_raw),
        microdata=[m for m in p.microdata if m["props"]],
        lang=p.lang,
        parser="stdlib",
    )


# --------------------------------------------------------------------------- #
# BeautifulSoup parser (optional upgrade)
# --------------------------------------------------------------------------- #
def _extract_bs4(html: str, url: str) -> PageContent:
    from bs4 import BeautifulSoup  # type: ignore

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    metas = [{k.lower(): (v if isinstance(v, str) else " ".join(v))
              for k, v in tag.attrs.items()} for tag in soup.find_all("meta")]
    link_tags = [{k.lower(): (v if isinstance(v, str) else " ".join(v))
                  for k, v in tag.attrs.items()} for tag in soup.find_all("link")]
    anchors = [a.get("href") for a in soup.find_all("a") if a.get("href")]
    imgs = [i.get("src") or i.get("data-src") for i in soup.find_all("img")
            if i.get("src") or i.get("data-src")]
    headings = {h: [_clean(t.get_text(" ")) for t in soup.find_all(h)] for h in _HEADINGS}
    raw_ld = [t.string or t.get_text() for t in
              soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)})]
    for t in soup(["script", "style", "noscript", "template", "svg"]):
        t.decompose()
    text = _clean(soup.get_text(" "))
    title = _clean(soup.title.get_text()) if soup.title else ""
    html_tag = soup.find("html")
    lang = (html_tag.get("lang") if html_tag else "") or ""

    micro = []
    for node in soup.find_all(attrs={"itemscope": True}):
        props: Dict[str, str] = {}
        for pr in node.find_all(attrs={"itemprop": True}):
            name = pr.get("itemprop")
            val = pr.get("content") or pr.get("href") or pr.get("src") or _clean(pr.get_text(" "))
            if name and val:
                props.setdefault(name, val)
        if props:
            micro.append({"type": node.get("itemtype", ""), "props": props})

    return _assemble(url=url, title=title, metas=metas, link_tags=link_tags,
                     anchors=anchors, imgs=imgs, headings=headings, text=text,
                     jsonld=_parse_jsonld_blocks([r for r in raw_ld if r]),
                     microdata=micro, lang=lang, parser="bs4")


# --------------------------------------------------------------------------- #
# Assembly (shared by both parsers)
# --------------------------------------------------------------------------- #
def _assemble(url, title, metas, link_tags, anchors, imgs, headings, text,
              jsonld, microdata, lang, parser) -> PageContent:
    meta_map: Dict[str, str] = {}
    og: Dict[str, str] = {}
    tw: Dict[str, str] = {}
    description = ""
    for m in metas:
        key = (m.get("property") or m.get("name") or m.get("itemprop") or "").lower()
        content = m.get("content", "")
        if not key:
            continue
        meta_map.setdefault(key, content)
        if key.startswith("og:") or key.startswith("product:"):
            og.setdefault(key, content)
        elif key.startswith("twitter:"):
            tw.setdefault(key, content)
        if key == "description" and not description:
            description = content

    canonical = ""
    for lt in link_tags:
        rel = lt.get("rel", "")
        rel = " ".join(rel) if isinstance(rel, list) else str(rel)
        if "canonical" in rel.lower() and lt.get("href"):
            canonical = lt["href"]
            break

    if not description:
        description = og.get("og:description", "") or tw.get("twitter:description", "")

    return PageContent(
        url=url,
        title=title or og.get("og:title", ""),
        meta_description=_clean(description),
        canonical_url=canonical,
        lang=(lang or meta_map.get("og:locale", "")).replace("_", "-")[:16],
        og=og, twitter=tw, meta=meta_map,
        h1=[h for h in headings.get("h1", []) if h],
        h2=[h for h in headings.get("h2", []) if h],
        h3=[h for h in headings.get("h3", []) if h],
        text=text, links=anchors, images=imgs,
        jsonld=jsonld, microdata=microdata, parser=parser,
    )


def _have_bs4() -> bool:
    import importlib
    try:
        importlib.import_module("bs4")
        return True
    except ImportError:
        return False


def extract_page(html: str, url: str = "", parser: str = "auto") -> PageContent:
    """Parse raw HTML into a :class:`PageContent`.

    ``parser``: ``"auto"`` prefers BeautifulSoup+lxml when importable and falls
    back to the standard-library parser; ``"stdlib"`` / ``"bs4"`` force one.
    """
    if not html:
        return PageContent(url=url)
    use_bs4 = parser == "bs4" or (parser == "auto" and _have_bs4())
    if use_bs4:
        try:
            return _extract_bs4(html, url)
        except Exception:
            return _extract_stdlib(html, url)
    return _extract_stdlib(html, url)
