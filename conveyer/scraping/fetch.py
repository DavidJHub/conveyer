"""Fetching, with safety and politeness built in.

Two modes, one interface (:class:`FetchResult`):

* **offline** (the default) — pages are served from an in-memory corpus
  (``html_by_url``): the synthetic corpus, or a previously saved cache. Nothing
  touches the network, so the whole pipeline runs in CI / notebooks with no
  connectivity and no risk to third-party sites.
* **online** (``ScrapeConfig.offline=False``) — a ``requests`` session that is
  polite by construction: it obeys ``robots.txt``, sends a descriptive
  user-agent, rate-limits per domain, retries transient failures with
  exponential backoff, caps the response size, checks the content-type, and
  caches every fetch to disk so re-runs never re-hit a site.

The fetcher only *retrieves*; parsing is :mod:`conveyer.scraping.extract`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib import robotparser

from .classify import parse_url
from .config import ScrapeConfig


@dataclass
class FetchResult:
    url: str
    status: str = "error"          # ok | cached | error | skipped | offline_miss | robots_blocked
    http_status: Optional[int] = None
    final_url: str = ""
    content_type: str = ""
    html: str = ""
    error: str = ""
    from_cache: bool = False
    fetched_at: str = ""
    elapsed_ms: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "cached") and bool(self.html)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cache_path(cfg: ScrapeConfig, url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return os.path.join(cfg.cache_dir, f"{h}.json")


# --------------------------------------------------------------------------- #
# Fetcher
# --------------------------------------------------------------------------- #
class Fetcher:
    def __init__(self, cfg: ScrapeConfig, html_by_url: Optional[Dict[str, str]] = None):
        self.cfg = cfg
        self.html_by_url = html_by_url or {}
        self._session = None
        self._robots: Dict[str, Optional[robotparser.RobotFileParser]] = {}
        self._last_hit: Dict[str, float] = {}
        self._lock = threading.Lock()

    # -- offline ----------------------------------------------------------- #
    def _offline(self, url: str) -> FetchResult:
        html = self.html_by_url.get(url)
        if html is None:
            # tolerate trailing-slash / scheme differences
            for k, v in self.html_by_url.items():
                if k.rstrip("/") == url.rstrip("/"):
                    html = v
                    break
        if html is None:
            return FetchResult(url=url, status="offline_miss", fetched_at=_now())
        return FetchResult(url=url, status="ok", http_status=200, final_url=url,
                           content_type="text/html", html=html, fetched_at=_now())

    # -- cache ------------------------------------------------------------- #
    def _read_cache(self, url: str) -> Optional[FetchResult]:
        if not self.cfg.use_cache:
            return None
        path = _cache_path(self.cfg, url)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            return FetchResult(url=url, status="cached", http_status=d.get("http_status"),
                               final_url=d.get("final_url", url),
                               content_type=d.get("content_type", ""),
                               html=d.get("html", ""), from_cache=True,
                               fetched_at=d.get("fetched_at", ""))
        except Exception:
            return None

    def _write_cache(self, res: FetchResult) -> None:
        if not self.cfg.use_cache:
            return
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        try:
            with open(_cache_path(self.cfg, res.url), "w", encoding="utf-8") as fh:
                json.dump({"http_status": res.http_status, "final_url": res.final_url,
                           "content_type": res.content_type, "html": res.html,
                           "fetched_at": res.fetched_at}, fh)
        except Exception:
            pass

    # -- robots + politeness ---------------------------------------------- #
    def _robots_ok(self, url: str, u) -> bool:
        if not self.cfg.respect_robots:
            return True
        host = f"{u.scheme or 'https'}://{u.host}"
        rp = self._robots.get(host, "unset")
        if rp == "unset":
            rp = robotparser.RobotFileParser()
            rp.set_url(host + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # unreadable robots ⇒ be permissive but still polite
            self._robots[host] = rp
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.cfg.user_agent, url)
        except Exception:
            return True

    def _throttle(self, domain: str) -> None:
        with self._lock:
            last = self._last_hit.get(domain, 0.0)
            wait = self.cfg.rate_limit_per_domain - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_hit[domain] = time.monotonic()

    # -- online ------------------------------------------------------------ #
    def _online(self, url: str) -> FetchResult:
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        u = parse_url(url)
        if not u.host:
            return FetchResult(url=url, status="error", error="unparseable url", fetched_at=_now())
        if not self._robots_ok(url, u):
            return FetchResult(url=url, status="robots_blocked", final_url=url, fetched_at=_now())

        try:
            import requests
        except ImportError:
            return FetchResult(url=url, status="error", error="requests not installed",
                               fetched_at=_now())
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.cfg.user_agent,
                                          "Accept": "text/html,application/xhtml+xml"})

        backoff = self.cfg.retry_backoff
        last_err = ""
        for attempt in range(self.cfg.max_retries + 1):
            self._throttle(u.registrable or u.host)
            t0 = time.monotonic()
            try:
                resp = self._session.get(url, timeout=self.cfg.timeout, stream=True,
                                         allow_redirects=True)
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if ctype and not any(ctype == a for a in self.cfg.allowed_content_types):
                    resp.close()
                    return FetchResult(url=url, status="skipped", http_status=resp.status_code,
                                       final_url=str(resp.url), content_type=ctype,
                                       error="non-html content-type", fetched_at=_now())
                chunks, total = [], 0
                for chunk in resp.iter_content(chunk_size=16384, decode_unicode=False):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= self.cfg.max_bytes:
                        break
                raw = b"".join(chunks)
                encoding = resp.encoding or "utf-8"
                html = raw.decode(encoding, errors="replace")
                res = FetchResult(
                    url=url, status="ok" if resp.ok else "error",
                    http_status=resp.status_code, final_url=str(resp.url),
                    content_type=ctype or "text/html", html=html if resp.ok else "",
                    error="" if resp.ok else f"HTTP {resp.status_code}",
                    fetched_at=_now(), elapsed_ms=int((time.monotonic() - t0) * 1000))
                resp.close()
                if res.ok:
                    self._write_cache(res)
                    return res
                last_err = res.error
            except Exception as exc:  # network / timeout
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.max_retries:
                time.sleep(backoff)
                backoff *= 2
        return FetchResult(url=url, status="error", error=last_err, fetched_at=_now())

    # -- public ------------------------------------------------------------ #
    def fetch(self, url: str) -> FetchResult:
        if self.cfg.offline:
            return self._offline(url)
        return self._online(url)

    def fetch_many(self, urls: List[str]) -> List[FetchResult]:
        if self.cfg.offline:
            return [self._offline(u) for u in urls]
        from concurrent.futures import ThreadPoolExecutor
        workers = max(1, min(self.cfg.max_workers, len(urls)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._online, urls))
