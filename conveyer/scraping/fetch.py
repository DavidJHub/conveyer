"""Fetching, with safety and politeness built in.

Two modes, one interface (:class:`FetchResult`):

* **offline** (the default) — pages are served from an in-memory corpus
  (``html_by_url``): the synthetic corpus, or a previously saved cache. Nothing
  touches the network, so the whole pipeline runs in CI / notebooks with no
  connectivity and no risk to third-party sites.
* **online** (``ScrapeConfig.offline=False``) — a ``requests`` session that is
  polite by construction: it obeys ``robots.txt`` (fetched with a bounded
  timeout — the stdlib reader would block forever on a dead host), sends a
  descriptive user-agent, rate-limits per domain without blocking other
  domains, retries transient failures with exponential backoff, caps the
  response size AND the per-URL wall clock (``hard_timeout`` covers robots +
  retries + slow-dribbling bodies), checks the content-type, and caches every
  fetch to disk so re-runs never re-hit a site. :meth:`Fetcher.iter_fetch`
  yields results as they complete so callers can persist incrementally.

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
from typing import Dict, Iterator, List, Optional
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
    error: str = ""                # includes "truncated ..." when a cap fired mid-body
    from_cache: bool = False
    fetched_at: str = ""
    elapsed_ms: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "cached") and bool(self.html)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def base_url_of(url: str) -> str:
    """``scheme://host/`` for a URL — the fallback target when the deep link is
    unreachable (x.com/…/status/…/photo/1 → x.com/). Empty when unparseable or
    already a root."""
    u = parse_url(url)
    if not u.host:
        return ""
    base = f"{u.scheme or 'https'}://{u.host}/"
    return "" if base.rstrip("/") == str(url).rstrip("/") else base


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
            # lazily serve a previous online run's page from the disk cache —
            # one file per URL, read on demand. (The pipeline used to preload
            # EVERY cached page's HTML into one dict, which at 2MB/page killed
            # machines on big replays.)
            cached = self._read_cache(url)
            if cached is not None and cached.html:
                return cached
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
    def _load_robots(self, host: str):
        """Fetch robots.txt through the session with a *bounded* timeout.

        The stdlib ``RobotFileParser.read()`` uses urllib with no timeout — a
        single unresponsive host would hang the whole pipeline forever, which
        is exactly the "runs forever without any error" failure mode. Fetch it
        ourselves and only *parse* with robotparser.
        """
        rp = robotparser.RobotFileParser()
        try:
            resp = self._session.get(host + "/robots.txt",
                                     timeout=min(self.cfg.timeout, 10.0))
            if resp.status_code in (401, 403):     # stdlib semantics: forbidden ⇒ disallow
                rp.disallow_all = True
            elif resp.status_code >= 400:          # missing robots ⇒ allow
                rp.allow_all = True
            else:
                rp.parse(resp.text.splitlines())
            return rp
        except Exception:
            return None  # unreachable robots ⇒ be permissive but still polite

    def _robots_ok(self, url: str, u) -> bool:
        if not self.cfg.respect_robots:
            return True
        host = f"{u.scheme or 'https'}://{u.host}"
        with self._lock:
            cached = self._robots.get(host, "unset")
        if cached == "unset":
            cached = self._load_robots(host)
            with self._lock:
                self._robots[host] = cached
        if cached is None:
            return True
        try:
            return cached.can_fetch(self.cfg.user_agent, url)
        except Exception:
            return True

    def _throttle(self, domain: str) -> None:
        """Per-domain rate limit that never blocks other domains.

        Reserves the next available slot for this domain *inside* the lock,
        then sleeps *outside* it — sleeping while holding the lock would
        serialize every worker thread behind one domain's cool-down.
        """
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last_hit.get(domain, 0.0) + self.cfg.rate_limit_per_domain)
            self._last_hit[domain] = slot
        delay = slot - now
        if delay > 0:
            time.sleep(delay)

    # -- online ------------------------------------------------------------ #
    def _ensure_session(self):
        import requests
        with self._lock:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({"User-Agent": self.cfg.user_agent,
                                              "Accept": "text/html,application/xhtml+xml"})
        return self._session

    def _online(self, url: str) -> FetchResult:
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        u = parse_url(url)
        if not u.host:
            return FetchResult(url=url, status="error", error="unparseable url", fetched_at=_now())
        try:
            self._ensure_session()
        except ImportError:
            return FetchResult(url=url, status="error", error="requests not installed",
                               fetched_at=_now())

        # One wall-clock budget for the WHOLE url: robots + every retry + body
        # streaming. `timeout` alone cannot bound a slow-dribbling server (it
        # only limits the gap between bytes), so without this a handful of bad
        # hosts make the run look like it hangs forever.
        deadline = time.monotonic() + self.cfg.hard_timeout

        if not self._robots_ok(url, u):
            return FetchResult(url=url, status="robots_blocked", final_url=url, fetched_at=_now())

        backoff = self.cfg.retry_backoff
        last_err = ""
        for attempt in range(self.cfg.max_retries + 1):
            if time.monotonic() >= deadline:
                last_err = last_err or f"hard_timeout ({self.cfg.hard_timeout:.0f}s) exceeded"
                break
            self._throttle(u.registrable or u.host)
            t0 = time.monotonic()
            try:
                remaining = max(0.5, deadline - time.monotonic())
                resp = self._session.get(url, timeout=min(self.cfg.timeout, remaining),
                                         stream=True, allow_redirects=True)
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if ctype and not any(ctype == a for a in self.cfg.allowed_content_types):
                    resp.close()
                    return FetchResult(url=url, status="skipped", http_status=resp.status_code,
                                       final_url=str(resp.url), content_type=ctype,
                                       error="non-html content-type", fetched_at=_now())
                chunks, total, truncated = [], 0, ""
                for chunk in resp.iter_content(chunk_size=16384, decode_unicode=False):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= self.cfg.max_bytes:
                        truncated = f"truncated at {total} bytes (max_bytes)"
                        break
                    if time.monotonic() >= deadline:
                        truncated = f"truncated at {total} bytes (hard_timeout)"
                        break
                raw = b"".join(chunks)
                encoding = resp.encoding or "utf-8"
                html = raw.decode(encoding, errors="replace")
                res = FetchResult(
                    url=url, status="ok" if resp.ok else "error",
                    http_status=resp.status_code, final_url=str(resp.url),
                    content_type=ctype or "text/html", html=html if resp.ok else "",
                    error=truncated if resp.ok else f"HTTP {resp.status_code}",
                    fetched_at=_now(), elapsed_ms=int((time.monotonic() - t0) * 1000))
                resp.close()
                if res.ok:
                    self._write_cache(res)
                    return res
                last_err = res.error
            except Exception as exc:  # network / timeout
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.max_retries and time.monotonic() < deadline:
                time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
                backoff *= 2
        return FetchResult(url=url, status="error", error=last_err, fetched_at=_now())

    # -- public ------------------------------------------------------------ #
    def fetch(self, url: str) -> FetchResult:
        if self.cfg.offline:
            return self._offline(url)
        return self._online(url)

    def iter_fetch(self, urls: List[str]) -> Iterator[FetchResult]:
        """Yield each URL's result **as soon as it completes** (unordered when
        online). This is what lets the pipeline persist line-by-line instead of
        waiting for the whole batch."""
        if self.cfg.offline:
            for u in urls:
                yield self._offline(u)
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = max(1, min(self.cfg.max_workers, len(urls) or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._online, u) for u in urls]
            for fut in as_completed(futures):
                yield fut.result()

    def fetch_many(self, urls: List[str]) -> List[FetchResult]:
        """Batch convenience wrapper over :meth:`iter_fetch` (order not
        guaranteed online; match results by ``FetchResult.url``)."""
        return list(self.iter_fetch(urls))
