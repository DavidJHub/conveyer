"""Make a re-run continue where the last one stopped — and refuse to lie about it.

:func:`~conveyer.scraping.pipeline.run_scrape` is already resumable: every
finished page is appended to a JSONL commit log (products first, page line last)
and ``resume=True`` skips the URLs that log vouches for. What that machinery
cannot know on its own is *whether the next run is the same run*. The commit log
is keyed by URL and nothing else, so pointing a different input at an output
directory that already holds a finished run silently **unions two URL
populations into one parquet** — every downstream count, rate and ranking is
then computed over a corpus that never existed.

This module supplies the missing identity. Each run writes ``run_manifest.json``
beside its outputs, recording

* **what it drew from** — the input file / directory (path, size, mtime) or the
  synthetic corpus (size, seed), plus the knobs that decide *which URLs are in
  scope*: ``max_urls``, ``dedupe_by``, ``only_recommended``, ``offline``;
* **how it labelled** — relevance vocabulary, learned model, vendor prior, fetch
  policy, matching thresholds.

:func:`prepare_run` compares that manifest against the config in hand:

==========  ================================================================
``new``     nothing accumulated in ``out_dir`` — start from scratch
``resume``  manifest matches — continue, skipping what is already done
``adopt``   state present but no manifest (a pre-manifest run) — continue
``fresh``   ``fresh=True`` — the stored run is discarded first
==========  ================================================================

A changed **input** raises :class:`ResumeMismatch` rather than quietly merging.
A changed **labelling** knob never blocks: those rows are stale, not
wrong-population, and the cure is the retroactive rescue that re-runs the
classifier over the cached HTML (``python -m conveyer.scraping.validate …
--reclassify --apply``) rather than a re-scrape. :func:`prepare_run` says so
when it sees one.

Typical use, in a notebook or a script::

    cfg = ScrapeConfig(clickstream_dir="data/conversations.parquet",
                       out_dir="outputs/scrape", resume=True)
    print(prepare_run(cfg).message)   # decides, explains, guards
    art = run_scrape(cfg)             # picks up where the last run stopped
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from .config import ScrapeConfig

MANIFEST_FILENAME = "run_manifest.json"

# Knobs that decide WHICH URLs a run works on. A change here makes the stored
# rows a different population, so resuming across one would merge two corpora —
# these are the ones prepare_run refuses to cross.
_INPUT_KNOBS = ("max_urls", "dedupe_by", "only_recommended", "offline")

# Knobs that decide HOW a page is labelled. A change here makes stored rows
# stale, not wrong-population — repairable in place from the cached HTML — so
# these only ever warn.
_LABEL_KNOBS = ("classifier", "extra_relevance_terms", "use_learned_model",
                "model_weight", "use_similarweb_prior", "prior_weight",
                "fetch_policy", "max_fetch_per_domain", "match_name_threshold",
                "coincide_threshold")

Change = Tuple[str, object, object]      # (key, stored, current)


class ResumeMismatch(RuntimeError):
    """The run stored in ``out_dir`` drew from a different input than this config."""


# --------------------------------------------------------------------------- #
# The commit log
# --------------------------------------------------------------------------- #
def iter_jsonl(path: str) -> Iterator[dict]:
    """Stream a JSONL file row by row, tolerating a torn final line from a
    crash. Never materializes the whole file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def pages_done(cfg: ScrapeConfig) -> int:
    """How many distinct URLs the commit log already vouches for."""
    return len({r.get("url") for r in iter_jsonl(cfg.pages_jsonl_path()) if r.get("url")})


# --------------------------------------------------------------------------- #
# Run identity
# --------------------------------------------------------------------------- #
def _plain(value):
    """JSON-comparable form of a config value (tuples/sets are order-agnostic)."""
    if isinstance(value, (tuple, list, set, frozenset)):
        return sorted(str(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _dir_digest(path: str) -> Tuple[int, str]:
    """(file count, short digest) over the data files in a star-schema dir.
    Cheap: stats only, never opens a file."""
    entries: List[Tuple[str, int, int]] = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if not name.lower().endswith((".parquet", ".csv", ".json", ".jsonl")):
                continue
            fp = os.path.join(root, name)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            entries.append((os.path.relpath(fp, path), int(st.st_size), int(st.st_mtime)))
    entries.sort()
    return len(entries), hashlib.sha1(repr(entries).encode()).hexdigest()[:16]


def source_identity(cfg: ScrapeConfig) -> dict:
    """Identity of the URL universe this config draws from."""
    path = os.path.abspath(cfg.clickstream_dir) if cfg.clickstream_dir else ""
    if not path or not os.path.exists(path):
        # _resolve_sources falls back to the synthetic corpus, whose identity is
        # entirely (size, seed)
        return {"kind": "synthetic", "n_pages": int(cfg.synthetic_n_pages),
                "seed": int(cfg.synthetic_seed)}
    if os.path.isdir(path):
        n_files, digest = _dir_digest(path)
        return {"kind": "dir", "path": path, "n_files": n_files, "digest": digest}
    st = os.stat(path)
    return {"kind": "file", "path": path, "size": int(st.st_size),
            "mtime": int(st.st_mtime)}


def input_fingerprint(cfg: ScrapeConfig) -> dict:
    fp = {"source": source_identity(cfg)}
    fp.update({k: _plain(getattr(cfg, k)) for k in _INPUT_KNOBS})
    return fp


def label_fingerprint(cfg: ScrapeConfig) -> dict:
    return {k: _plain(getattr(cfg, k)) for k in _LABEL_KNOBS}


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #
def manifest_path(cfg: ScrapeConfig) -> str:
    return os.path.join(cfg.out_dir, MANIFEST_FILENAME)


def read_manifest(cfg: ScrapeConfig) -> Optional[dict]:
    """The stored manifest, or ``None``. An unreadable/corrupt manifest reads as
    absent — it must never be the thing that stops a run."""
    try:
        with open(manifest_path(cfg), "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def write_manifest(cfg: ScrapeConfig, n_pages: int = 0,
                   merged: Optional[List[dict]] = None) -> dict:
    """Record this run's identity beside its outputs (atomic replace).

    ``merged`` carries the inputs an explicit ``on_mismatch="resume"`` folded
    into this table. Once a table spans two populations it must keep saying so —
    re-stamping it with only the newest input would launder exactly the mistake
    this module exists to prevent.
    """
    manifest = {
        "version": 1,
        "input": input_fingerprint(cfg),
        "labels": label_fingerprint(cfg),
        "pages": int(n_pages),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if merged:
        manifest["merged"] = list(merged)
    try:
        os.makedirs(cfg.out_dir, exist_ok=True)
        tmp = manifest_path(cfg) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        os.replace(tmp, manifest_path(cfg))
    except OSError:
        pass          # a manifest we cannot write must not break the run
    return manifest


def _diff(stored: Optional[dict], current: Optional[dict]) -> List[Change]:
    stored, current = stored or {}, current or {}
    return [(k, stored.get(k), current.get(k))
            for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)]


def manifest_input_drift(cfg: ScrapeConfig) -> List[Change]:
    """Input-fingerprint differences between the stored manifest and ``cfg``.
    Empty when there is no manifest — nothing to contradict."""
    manifest = read_manifest(cfg)
    if manifest is None:
        return []
    return _diff(manifest.get("input"), input_fingerprint(cfg))


# --------------------------------------------------------------------------- #
# Starting over, on purpose
# --------------------------------------------------------------------------- #
def reset_run_state(cfg: ScrapeConfig) -> int:
    """Discard the accumulated run so the next one starts from zero. Returns how
    many pages were dropped.

    Removes exactly what a run owns: the JSONL commit logs, the parquet part
    directories, the finalized parquets, the learned domain profiles and the
    manifest. Deliberately spared — the per-URL HTML cache (``cache_dir``) and
    the learned model (``model_path``): both are *derived* artifacts, so keeping
    them makes a fresh run fast and costs nothing in correctness.
    """
    n = pages_done(cfg)
    for f in (cfg.pages_jsonl_path(), cfg.products_jsonl_path(),
              cfg.pages_path(), cfg.products_path(),
              os.path.join(cfg.out_dir, "domain_profiles.json"),
              manifest_path(cfg)):
        try:
            os.remove(f)
        except OSError:
            pass
    for d in (cfg.pages_parts_dir(), cfg.products_parts_dir()):
        shutil.rmtree(d, ignore_errors=True)
    return n


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def _fmt(value: object, width: int = 72) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else repr(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _fmt_changes(changes: List[Change], indent: str = "    ") -> str:
    return "\n".join(f"{indent}{k}: {_fmt(old)}  ->  {_fmt(new)}" for k, old, new in changes)


@dataclass
class ResumeStatus:
    """What :func:`prepare_run` decided, and why."""

    action: str                                    # new | resume | adopt | fresh
    pages_done: int                                # pages already in the commit log
    out_dir: str
    input_changes: List[Change] = field(default_factory=list)
    label_changes: List[Change] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def resuming(self) -> bool:
        return self.action in ("resume", "adopt") and self.pages_done > 0

    @property
    def message(self) -> str:
        head = {
            "new": f"[resume] {self.out_dir}: no previous run here — starting from scratch.",
            "resume": (f"[resume] {self.out_dir}: {self.pages_done} pages already done — "
                       f"continuing from them (set fresh=True to start over)."),
            "adopt": (f"[resume] {self.out_dir}: {self.pages_done} pages already done — "
                      f"continuing from them."),
            "fresh": (f"[resume] {self.out_dir}: discarded {self.pages_done} pages of "
                      f"previous state — starting from scratch."),
        }[self.action]
        lines = [head]
        if self.label_changes:
            lines.append("[resume] classifier settings changed since those pages were "
                         "labelled — they keep the labels they were scraped with:")
            lines.append(_fmt_changes(self.label_changes))
            lines.append("[resume] repair them in place (no re-scrape): python -m "
                         "conveyer.scraping.validate <pages.parquet> --reclassify --apply")
        lines.extend(f"[resume] {n}" for n in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:      # so print(status) reads the same as print(status.message)
        return self.message


def prepare_run(cfg: ScrapeConfig, fresh: bool = False,
                on_mismatch: str = "error") -> ResumeStatus:
    """Decide how a run over ``cfg.out_dir`` should start, and make it so.

    ``fresh=True`` discards the stored run first (see :func:`reset_run_state`).
    Otherwise the stored manifest is compared with ``cfg`` and, when the *input*
    no longer matches, ``on_mismatch`` decides:

    * ``"error"`` (default) — raise :class:`ResumeMismatch`, changing nothing;
    * ``"fresh"`` — discard the stored run and start over;
    * ``"resume"`` — extend the stored table anyway (you mean it: the result
      spans both inputs).

    Returns a :class:`ResumeStatus` whose ``.message`` explains the decision.
    """
    if on_mismatch not in ("error", "fresh", "resume"):
        raise ValueError(f"on_mismatch must be error|fresh|resume, got {on_mismatch!r}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    if fresh:
        discarded = reset_run_state(cfg)
        write_manifest(cfg)
        return ResumeStatus("fresh", discarded, cfg.out_dir)

    done = pages_done(cfg)
    manifest = read_manifest(cfg)

    # Nothing accumulated: whatever a stale manifest claims, there is no
    # population to merge into, so just (re)stamp this run's identity.
    if done == 0:
        write_manifest(cfg)
        return ResumeStatus("new", 0, cfg.out_dir)

    if manifest is None:
        # Pages from a run that predates the manifest — nothing to compare
        # against, so they are taken on trust and stamped from here on.
        write_manifest(cfg, done)
        return ResumeStatus("adopt", done, cfg.out_dir, notes=[
            "those pages carry no run_manifest.json, so their input could not be "
            "verified — pass fresh=True if they came from a different one."])

    input_changes = _diff(manifest.get("input"), input_fingerprint(cfg))
    label_changes = _diff(manifest.get("labels"), label_fingerprint(cfg))
    merged: List[dict] = list(manifest.get("merged") or [])

    if input_changes:
        if on_mismatch == "error":
            raise ResumeMismatch(
                f"{cfg.out_dir} holds {done} pages from a run with a different input:\n"
                f"{_fmt_changes(input_changes)}\n"
                "Resuming would merge two URL populations into one parquet. Either:\n"
                "  * point out_dir at a new directory for this input, or\n"
                "  * start clean:  prepare_run(cfg, fresh=True), or\n"
                "  * extend it on purpose:  prepare_run(cfg, on_mismatch='resume').\n"
                "(A file merely re-copied — same content, new mtime — trips this too.)")
        if on_mismatch == "fresh":
            discarded = reset_run_state(cfg)
            write_manifest(cfg)
            return ResumeStatus("fresh", discarded, cfg.out_dir,
                                input_changes=input_changes,
                                notes=["the stored run drew from a different input "
                                       "(on_mismatch='fresh')."])
        # an accepted merge is remembered, so the table never reads as one
        # population again just because the newest input was stamped over it
        merged.append(manifest.get("input"))
        notes = ["extending a table built from a different input "
                 "(on_mismatch='resume') — its counts now span both."]
    else:
        notes = []

    if merged:
        notes.append(f"this table spans {len(merged) + 1} inputs merged on purpose — "
                     f"its counts are not a single population.")

    write_manifest(cfg, done, merged=merged)
    return ResumeStatus("resume", done, cfg.out_dir, input_changes=input_changes,
                        label_changes=label_changes, notes=notes)
