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
    status: str = "error"          # ok | cached | error | skipped | offline_miss |
                                   # robots_blocked | circuit_open | blocked
    http_status: Optional[int] = None
    final_url: str = ""
    content_type: str = ""
    html: str = ""
    error: str = ""                # includes "truncated ..." when a cap fired mid-body
    from_cache: bool = False
    fetched_at: str = ""
    elapsed_ms: Optional[int] = None
    # salvage: even a 403 answers with headers (platform fingerprints — a
    # Shopify store identifies itself on its block page) and an error body.
    # error_body is NEVER treated as page content — fingerprinting only.
    headers: Dict[str, str] = field(default_factory=dict)
    error_body: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "cached") and bool(self.html)


# HTTP statuses that mean "a wall, not a failure": the origin answered and
# will keep answering the same way — retrying only hammers it. 429 is the
# one retried, once, when Retry-After fits inside the wall-clock budget.
_BLOCK_STATUSES = {401, 403, 406, 451}

# A realistic desktop-browser header profile. Many CDNs 403 unknown
# User-Agent strings at the edge before the origin ever sees the request;
# presenting normal browser headers avoids that naive block. This changes
# *presentation only* — robots.txt compliance, rate limits, the per-domain
# circuit breaker and the hard wall-clock cap all still apply.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

_MAX_ERROR_BODY = 4096       # bytes of a non-ok body kept for fingerprinting
_MAX_HEADERS = 40            # response headers captured per fetch


def _parse_retry_after(value: str) -> Optional[float]:
    """RFC 9110 Retry-After: delta-seconds OR an HTTP-date. None when absent
    or unparseable — the caller then falls back to normal backoff retries."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _capture_headers(resp) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, (k, v) in enumerate(resp.headers.items()):
        if i >= _MAX_HEADERS:
            break
        out[str(k).lower()] = str(v)[:300]
    return out


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
        # circuit breaker: {domain: [consecutive_failures, successes]}. A domain
        # that only ever fails stops being fetched after the threshold — dead or
        # bot-walled hosts must not cost hard_timeout for every one of their URLs.
        self._domain_state: Dict[str, list] = {}
        self._lock = threading.Lock()

    # -- circuit breaker ---------------------------------------------------- #
    def _circuit_open(self, domain: str) -> bool:
        thr = getattr(self.cfg, "domain_failure_threshold", 0)
        if thr <= 0:
            return False
        with self._lock:
            fails, oks = self._domain_state.get(domain, [0, 0])
        return oks == 0 and fails >= thr

    def _record_outcome(self, domain: str, ok: bool) -> None:
        with self._lock:
            state = self._domain_state.setdefault(domain, [0, 0])
            if ok:
                state[0], state[1] = 0, state[1] + 1
            else:
                state[0] += 1

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
                               fetched_at=d.get("fetched_at", ""),
                               headers=dict(d.get("headers") or {}))
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
                           "fetched_at": res.fetched_at, "headers": res.headers}, fh)
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
        from urllib.parse import urlsplit
        netloc = urlsplit(url if "://" in url else "http://" + url).netloc or u.host
        host = f"{u.scheme or 'https'}://{netloc}"   # keep the port — robots
                                                     # lives on the same origin
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
                if getattr(self.cfg, "browser_headers", False):
                    self._session.headers.update(_BROWSER_HEADERS)
                else:
                    self._session.headers.update(
                        {"User-Agent": self.cfg.user_agent,
                         "Accept": "text/html,application/xhtml+xml"})
        return self._session

    def _online(self, url: str) -> FetchResult:
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        u = parse_url(url)
        if not u.host:
            return FetchResult(url=url, status="error", error="unparseable url", fetched_at=_now())
        domain = u.registrable or u.host
        if self._circuit_open(domain):
            return FetchResult(url=url, status="circuit_open",
                               error=f"domain circuit open "
                                     f"({self._domain_state[domain][0]} straight failures)",
                               fetched_at=_now())
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
            self._record_outcome(domain, ok=True)   # host answered; only blocked
            return FetchResult(url=url, status="robots_blocked", final_url=url, fetched_at=_now())

        backoff = self.cfg.retry_backoff
        last_err = ""
        # salvage from the most recent answered attempt: headers + a bounded
        # error-body excerpt survive into the final result even when the fetch
        # ultimately fails — a 403's headers still fingerprint the platform
        last_hdrs: Dict[str, str] = {}
        last_body = ""
        last_status: Optional[int] = None
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
                hdrs = _capture_headers(resp)
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if resp.ok and ctype and not any(ctype == a for a in self.cfg.allowed_content_types):
                    resp.close()
                    self._record_outcome(domain, ok=True)
                    return FetchResult(url=url, status="skipped", http_status=resp.status_code,
                                       final_url=str(resp.url), content_type=ctype,
                                       error="non-html content-type", fetched_at=_now(),
                                       headers=hdrs)
                chunks, total, truncated = [], 0, ""
                body_cap = self.cfg.max_bytes if resp.ok else _MAX_ERROR_BODY
                for chunk in resp.iter_content(chunk_size=16384, decode_unicode=False):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= body_cap:
                        truncated = f"truncated at {total} bytes (max_bytes)" if resp.ok else ""
                        break
                    if time.monotonic() >= deadline:
                        truncated = f"truncated at {total} bytes (hard_timeout)"
                        break
                raw = b"".join(chunks)
                encoding = resp.encoding or "utf-8"
                html = raw.decode(encoding, errors="replace")
                status_code = resp.status_code
                final_url = str(resp.url)
                resp.close()
                if resp.ok:
                    res = FetchResult(
                        url=url, status="ok", http_status=status_code,
                        final_url=final_url, content_type=ctype or "text/html",
                        html=html, error=truncated, fetched_at=_now(),
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                        headers=hdrs)
                    self._write_cache(res)
                    self._record_outcome(domain, ok=True)
                    return res
                last_hdrs, last_status = hdrs, status_code
                last_body = html[:_MAX_ERROR_BODY]
                last_err = f"HTTP {status_code}"
                if status_code in _BLOCK_STATUSES:
                    # a wall answers the same way every time — do not hammer it
                    self._record_outcome(domain, ok=False)
                    return FetchResult(url=url, status="blocked", http_status=status_code,
                                       final_url=final_url, content_type=ctype,
                                       error=last_err, fetched_at=_now(),
                                       headers=hdrs, error_body=last_body)
                if status_code == 429:
                    ra = _parse_retry_after(hdrs.get("retry-after", ""))
                    if ra is not None:
                        # honor the server's own pacing when it fits the budget
                        if attempt < self.cfg.max_retries \
                                and time.monotonic() + ra < deadline:
                            time.sleep(ra)
                            continue
                        self._record_outcome(domain, ok=False)
                        return FetchResult(url=url, status="blocked",
                                           http_status=status_code,
                                           final_url=final_url, content_type=ctype,
                                           error=last_err, fetched_at=_now(),
                                           headers=hdrs, error_body=last_body)
                    # no usable Retry-After: give it the normal backoff retries;
                    # the loop's final failure return keeps the salvage
            except Exception as exc:  # network / timeout
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.max_retries and time.monotonic() < deadline:
                time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
                backoff *= 2
        self._record_outcome(domain, ok=False)
        return FetchResult(url=url, status="error", http_status=last_status,
                           error=last_err, fetched_at=_now(),
                           headers=last_hdrs, error_body=last_body)

    # -- public ------------------------------------------------------------ #
    def fetch(self, url: str) -> FetchResult:
        if self.cfg.offline:
            return self._offline(url)
        return self._online(url)

    def iter_fetch(self, urls: List[str]) -> Iterator[FetchResult]:
        """Yield each URL's result **as soon as it completes** (unordered when
        online). This is what lets the pipeline persist line-by-line instead of
        waiting for the whole batch.

        Submission is a **sliding window** (~2× workers in flight) and every
        completed Future is dropped immediately. This is load-bearing: a
        completed ``Future`` retains its result forever, and each result's
        ``.html`` is a full page body (up to ``max_bytes`` = 2MB). The old
        implementation kept ALL futures in a list for the whole run, so the
        raw HTML of every page ever fetched stayed pinned in RAM — ~10-26GB
        by ~13k real pages, killing long runs at a consistent page count.
        With the window, at most ~2× workers results exist at once (~50MB
        worst case) no matter how many URLs the run covers."""
        if self.cfg.offline:
            for u in urls:
                yield self._offline(u)
            return
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        workers = max(1, min(self.cfg.max_workers, len(urls) or 1))
        window = max(2 * workers, 1)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            in_flight = set()
            it = iter(urls)
            try:
                while True:
                    while len(in_flight) < window:
                        u = next(it, None)
                        if u is None:
                            break
                        in_flight.add(ex.submit(self._online, u))
                    if not in_flight:
                        break
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    while done:
                        fut = done.pop()   # drop our reference before yielding:
                        res = fut.result()  # after the caller moves on, nothing
                        del fut             # holds this result — HTML is freed
                        yield res
                        del res
            finally:
                for fut in in_flight:   # generator closed early: drop the queue
                    fut.cancel()

    def fetch_many(self, urls: List[str]) -> List[FetchResult]:
        """Batch convenience wrapper over :meth:`iter_fetch` (order not
        guaranteed online; match results by ``FetchResult.url``). Holds every
        result at once — small batches only; long runs use ``iter_fetch``."""
        return list(self.iter_fetch(urls))
