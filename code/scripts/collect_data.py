"""Crawl and download raw source documents for a run.

Reads seed URLs from 'raw_dataset/data_links.txt', crawls within configured depth
limits, and saves HTML/PDF/DOCX files under '/d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/raw/'.

A crawl manifest is written to 'raw/manifest.jsonl' so later parsing can attach
the original URL and download timestamp to each saved file.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from typing import TYPE_CHECKING
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import normalize_url, sha256_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_DEPTH_FRI = 2
_DEFAULT_DEPTH_UL = 1
_DEFAULT_DEPTH_V = 0

_ALLOWED_BINARY_SUFFIXES = {".pdf", ".docx"}
_ALLOWED_HTML_SUFFIXES = {"", ".html", ".htm", ".php", ".asp", ".aspx"}

# Image downloading turned off
_ALLOW_IMAGE_DOWNLOADS = False
_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

if TYPE_CHECKING:
    from urllib.robotparser import RobotFileParser


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_seed_urls(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Seed links file not found: {path}")

    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _is_ul_domain(netloc: str) -> bool:
    netloc = (netloc or "").lower()
    return netloc.endswith(".uni-lj.si") or netloc == "uni-lj.si"


def _is_fri_domain(netloc: str) -> bool:
    netloc = (netloc or "").lower()
    return netloc.endswith("fri.uni-lj.si")


def _is_ucilnica_domain(netloc: str) -> bool:
    netloc = (netloc or "").lower()
    host = netloc.split(":", 1)[0]
    return host == "ucilnica.fri.uni-lj.si"


def _classify_url(url: str) -> str:
    """Return one of: 'fri', 'ul', 'v'."""
    netloc = urlparse(url).netloc

    if _is_fri_domain(netloc):
        return "fri"
    if _is_ul_domain(netloc):
        return "ul"
    return "v"


def _safe_path_component(text: str, *, max_len: int = 120) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text)
    cleaned = cleaned.strip("_") or "_"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def _url_to_relative_path(url: str) -> Path:
    """Map a URL to a stable relative path under RAW_DIR.

    The goal is predictable, easiyl navigable file layouts while avoiding most
    filename collisions.
    """
    parsed = urlparse(url)
    netloc = _safe_path_component(parsed.netloc.lower() or "unknown")
    path = parsed.path or "/"

    # Normalize path and ensure we end in a file.
    if path.endswith("/"):
        path = path + "index.html"

    p = Path(path.lstrip("/"))
    suffix = p.suffix.lower()

    # Treat unknown extensions as HTML so parsing can still work.
    if suffix not in _ALLOWED_BINARY_SUFFIXES and suffix not in (".html", ".htm"):
        if suffix:
            p = p.with_suffix(p.suffix + ".html")
        else:
            p = p.with_suffix(".html")

    parts = [_safe_path_component(x) for x in p.parts if x and x not in (".", "..")]
    if not parts:
        parts = ["index.html"]

    # If there's a query string, include a short hash to reduce collisions.
    if parsed.query:
        qh = sha256_bytes(parsed.query.encode("utf-8"))[:10]
        name = Path(parts[-1])
        parts[-1] = f"{name.stem}__q_{qh}{name.suffix}"

    return Path(netloc, *parts)


def _url_suffix(url: str) -> str:
    try:
        return Path(urlparse(url).path or "").suffix.lower()
    except Exception:
        return ""


def _is_supported_link(url: str) -> bool:
    suffix = _url_suffix(url)
    if suffix in _ALLOWED_BINARY_SUFFIXES:
        return True
    if suffix in _ALLOWED_HTML_SUFFIXES:
        return True
    if _ALLOW_IMAGE_DOWNLOADS and suffix in _ALLOWED_IMAGE_SUFFIXES:
        return True
    return False


def _looks_like_attachment_url(url: str) -> bool:
    suffix = _url_suffix(url)
    if suffix in _ALLOWED_BINARY_SUFFIXES:
        return True

    u = (url or "").lower()
    return ".pdf" in u or ".docx" in u


def _is_html_content_type(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return "text/html" in ct or "application/xhtml+xml" in ct


def _looks_like_pdf(data: bytes) -> bool:
    # PDFs typically begin with %PDF-, but allow leading whitespace/newlines.
    head = (data or b"")[:2048].lstrip()
    return head.startswith(b"%PDF-")


def _looks_like_docx(data: bytes) -> bool:
    # DOCX is a ZIP container.
    head = (data or b"")[:8]
    return head.startswith(b"PK\x03\x04")


def _looks_like_html_bytes(data: bytes) -> bool:
    head = (data or b"")[:4096].lstrip()
    h = head.lower()
    return (b"<!doctype html" in h) or (b"<html" in h) or (b"<head" in h)


@dataclass
class ManifestWriter:
    manifest_path: Path
    known_relpaths: set[str] = field(default_factory=set)

    def load_existing(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            for line in self.manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rel = str(rec.get("relative_path", "") or "").strip()
                if rel:
                    self.known_relpaths.add(rel)
        except Exception as e:
            log.warning("Could not read existing manifest %s: %s", self.manifest_path, e)

    def has(self, relative_path: Path) -> bool:
        rel = str(relative_path).replace("\\", "/")
        return rel in self.known_relpaths

    def append(
        self,
        *,
        relative_path: Path,
        source_url: str,
        downloaded_from: str,
        sha256: str,
        saved_at: str,
        content_type: str | None,
        status_code: int | None,
        num_bytes: int | None,
    ) -> None:
        rel = str(relative_path).replace("\\", "/")
        if rel in self.known_relpaths:
            return

        record = {
            "relative_path": rel,
            "source_url": source_url,
            "downloaded_from": downloaded_from,
            "sha256": sha256,
            "saved_at": saved_at,
            "content_type": content_type or "",
            "status_code": status_code,
            "num_bytes": num_bytes,
        }

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.known_relpaths.add(rel)


@dataclass
class Crawler:
    raw_dir: Path
    manifest: ManifestWriter
    depth_fri: int
    depth_ul: int
    depth_v: int
    delay_seconds: float
    user_agent: str = "ul-fri-nlp-course-project/1.0"
    timeout_seconds: float = 20.0

    session: requests.Session = field(default_factory=requests.Session, init=False)
    _seen_urls: set[str] = field(default_factory=set, init=False)
    _robots_cache: dict[str, "RobotFileParser"] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.session.headers.update({"User-Agent": self.user_agent})

    def allowed_depth(self, url: str) -> int:
        group = _classify_url(url)
        if group == "fri":
            return self.depth_fri
        if group == "ul":
            return self.depth_ul
        return self.depth_v

    def can_fetch(self, url: str) -> bool:
        # Intentionally permissive: if robots.txt is unreachable, allow the request.
        try:
            from urllib.robotparser import RobotFileParser

            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = urljoin(base, "/robots.txt")

            cached = self._robots_cache.get(base)
            if cached is not None:
                return bool(cached.can_fetch(self.user_agent, url))

            rp = RobotFileParser()
            rp.set_url(robots_url)

            try:
                resp = self.session.get(robots_url, timeout=self.timeout_seconds)
                if resp.ok:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.parse([])
            except Exception:
                rp.parse([])

            self._robots_cache[base] = rp

            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _file_mtime_iso(self, path: Path) -> str:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        except Exception:
            return _utc_now_iso()

    def _sleep(self) -> None:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def _fetch(self, url: str) -> requests.Response | None:
        try:
            resp = self.session.get(url, timeout=self.timeout_seconds, allow_redirects=True)
            self._sleep()
            return resp
        except Exception as e:
            log.warning("Fetch failed: %s (%s)", url, e)
            return None

    def _save_bytes(self, rel_path: Path, data: bytes) -> Path:
        dest = self.raw_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def _download_and_save(self, url: str) -> tuple[Path | None, bool]:
        """Download a URL and save it.

        Returns: (relative_path, skipped)
        - relative_path is not None only when a file is saved (or already exists)
        - skipped=True means the URL was intentionally ignored (unsupported type)
        """

        if not _is_supported_link(url):
            return None, True

        # Policy: allow saving HTML pages from ucilnica, but do not download binary
        # resources (PDF/DOCX/...) from that host.
        parsed_url = urlparse(url)
        if _is_ucilnica_domain(parsed_url.netloc):
            # Avoid fetching obvious attachments from ucilnica.
            if _url_suffix(url) in _ALLOWED_BINARY_SUFFIXES or _looks_like_attachment_url(url):
                return None, True

        if not self.can_fetch(url):
            log.info("Blocked by robots.txt: %s", url)
            return None, False

        rel = _url_to_relative_path(url)
        # Some URLs (e.g., view.php?id=...) are mapped to an .html filename but may be
        # saved as a binary (.pdf/.docx) based on response headers. In update mode we
        # want to avoid refetching if the binary already exists.
        candidates = [rel]
        if rel.suffix.lower() in (".html", ".htm"):
            candidates.append(rel.with_suffix(".pdf"))
            candidates.append(rel.with_suffix(".docx"))

        dest = None
        for cand in candidates:
            d = self.raw_dir / cand
            if d.exists():
                rel = cand
                dest = d
                break
        if dest is None:
            dest = self.raw_dir / rel

        if dest.exists():
            # Backfill the manifest entry if needed.
            if not self.manifest.has(rel):
                try:
                    data = dest.read_bytes()
                    sha = sha256_bytes(data)
                    self.manifest.append(
                        relative_path=rel,
                        source_url=normalize_url(url),
                        downloaded_from=normalize_url(url),
                        sha256=sha,
                        saved_at=self._file_mtime_iso(dest),
                        content_type=None,
                        status_code=None,
                        num_bytes=len(data),
                    )
                except Exception as e:
                    log.warning("Could not backfill manifest for %s: %s", dest, e)
            return rel, False

        resp = self._fetch(url)
        if resp is None:
            return None, False
        if not resp.ok or resp.content is None:
            return None, False

        content = resp.content
        sha = sha256_bytes(content)
        saved_at = _utc_now_iso()

        content_type = (resp.headers.get("Content-Type") or "")
        ct_lower = content_type.lower()
        is_pdf = "application/pdf" in ct_lower
        is_docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in ct_lower
        is_html = _is_html_content_type(ct_lower)

        # Some servers mislabel content-types or redirect to login/HTML while keeping a .pdf URL.
        # Use lightweight magic-byte sniffing to keep the saved extension consistent with content.
        looks_pdf = _looks_like_pdf(content)
        looks_docx = _looks_like_docx(content)
        looks_html = _looks_like_html_bytes(content)

        if looks_pdf:
            is_pdf, is_docx, is_html = True, False, False
        elif looks_docx:
            is_pdf, is_docx, is_html = False, True, False
        elif looks_html and not (is_pdf or is_docx):
            is_html = True

        # Policy enforcement: after sniffing, skip binary content from ucilnica.
        if _is_ucilnica_domain(parsed_url.netloc) and (is_pdf or is_docx):
            return None, True

        # Hard stop: do not save other file types (zip, xlsx, images unless enabled, ...).
        if not (is_pdf or is_docx or is_html):
            # If images are enabled, accept common image types even if the server doesn't
            # set a perfect content-type.
            if _ALLOW_IMAGE_DOWNLOADS and _url_suffix(url) in _ALLOWED_IMAGE_SUFFIXES:
                pass
            else:
                return None, True

        # Enforce: download PDFs/DOCXs only from UL domains.
        if (is_pdf or is_docx) and not _is_ul_domain(urlparse(url).netloc):
            return None, True

        # Ensure the saved file suffix matches the detected content.
        if is_pdf and rel.suffix.lower() != ".pdf":
            rel = rel.with_suffix(".pdf")
        if is_docx and rel.suffix.lower() != ".docx":
            rel = rel.with_suffix(".docx")
        if is_html and rel.suffix.lower() in _ALLOWED_BINARY_SUFFIXES:
            # Preserve original suffix (useful for debugging) but make it parseable as HTML.
            rel = rel.with_suffix(rel.suffix + ".html")

        self._save_bytes(rel, content)
        self.manifest.append(
            relative_path=rel,
            source_url=normalize_url(url),
            downloaded_from=normalize_url(resp.url or url),
            sha256=sha,
            saved_at=saved_at,
            content_type=resp.headers.get("Content-Type"),
            status_code=resp.status_code,
            num_bytes=len(content),
        )
        return rel, False

    def _extract_links(self, base_url: str, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = str(a.get("href") or "").strip()
            if not href:
                continue
            href_lower = href.lower()
            if href_lower.startswith("mailto:") or href_lower.startswith("javascript:"):
                continue

            try:
                abs_url = urljoin(base_url, href)
                abs_url, _ = urldefrag(abs_url)
                parsed = urlparse(abs_url)
            except ValueError as e:
                # Some pages contain malformed hrefs (e.g., invalid bracketed IPv6 netlocs)
                # that cause urllib.parse to throw. Skip them so crawling can continue.
                log.debug("Skipping malformed link href=%r base=%r (%s)", href[:200], base_url, e)
                continue
            if parsed.scheme not in ("http", "https"):
                continue

            links.append(abs_url)
        return links

    def crawl(self, seeds: Iterable[str]) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest.load_existing()

        queue: deque[tuple[str, int]] = deque()
        for s in seeds:
            url = normalize_url(s)
            if url and _is_supported_link(url):
                queue.append((url, 0))

        pages_processed = 0
        failed_urls: list[str] = []

        while queue:
            url, depth = queue.popleft()
            url_norm = normalize_url(url)
            if not url_norm or url_norm in self._seen_urls:
                continue

            allowed = self.allowed_depth(url_norm)
            if depth > allowed:
                continue

            self._seen_urls.add(url_norm)

            # Download PDFs/DOCXs only from UL domains.
            rel, skipped = self._download_and_save(url_norm)
            if rel is None:
                if skipped:
                    continue
                failed_urls.append(url_norm)
                continue

            is_binary = rel.suffix.lower() in _ALLOWED_BINARY_SUFFIXES

            pages_processed += 1
            log.info("Saved: %s <- %s", rel, url_norm)

            if is_binary:
                continue

            # Only parse links from saved HTML files.
            if rel.suffix.lower() not in (".html", ".htm"):
                continue

            # If we have the saved HTML file, parse links from disk to avoid
            # relying on response decoding.
            try:
                html_text = (self.raw_dir / rel).read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            in_ucilnica = _is_ucilnica_domain(urlparse(url_norm).netloc)

            for link in self._extract_links(url_norm, html_text):
                # Containment rule: once we're inside ucilnica, do not follow links
                # that leave that host.
                if in_ucilnica and not _is_ucilnica_domain(urlparse(link).netloc):
                    continue
                if not _is_supported_link(link):
                    continue
                is_binary_link = _looks_like_attachment_url(link)

                # Do not download binary files from ucilnica, even if linked elsewhere.
                if is_binary_link and _is_ucilnica_domain(urlparse(link).netloc):
                    continue

                # Depth counts HTML hops. Attachments linked from a page do not
                # increase depth, so PDFs/DOCXs on a seed page are downloaded
                # even with depth 0.
                next_depth = depth if is_binary_link else depth + 1
                if next_depth > self.allowed_depth(link):
                    continue

                if is_binary_link and not _is_ul_domain(urlparse(link).netloc):
                    continue

                if normalize_url(link) not in self._seen_urls:
                    queue.append((link, next_depth))

        if failed_urls:
            failed_path = self.raw_dir / "failed_downloads.txt"
            with open(failed_path, "a", encoding="utf-8") as f:
                for u in failed_urls:
                    f.write(u + "\n")
            log.warning("Failed downloads: %d (see %s)", len(failed_urls), failed_path)

        log.info("Done. Saved %d file(s) under %s", pages_processed, self.raw_dir)


def _snapshot_inputs(*, seeds_file: Path, run_name: str | None, mode: str, depths: dict) -> list[str]:
    seeds = _read_seed_urls(seeds_file)

    config.INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(seeds_file, config.SEED_LINKS_SNAPSHOT_FILE)
    except Exception as e:
        log.warning("Could not snapshot seed links: %s", e)

    event = {
        "event": "collect_data",
        "timestamp": _utc_now_iso(),
        "run": (run_name or config.DEFAULT_RUN_NAME),
        "mode": mode,
        "depths": depths,
        "seed_file": str(seeds_file),
        "num_seeds": len(seeds),
    }
    try:
        with open(config.INPUTS_MANIFEST_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Could not write inputs manifest: %s", e)

    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run/dataset name (stored under /d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<name>/)")
    parser.add_argument("--mode", choices=["new", "update"], default="update", help="Create a fresh run folder or update an existing one")

    parser.add_argument("--depth-fri", type=int, default=_DEFAULT_DEPTH_FRI, help="Max crawl depth for fri.uni-lj.si")
    parser.add_argument("--depth-ul", type=int, default=_DEFAULT_DEPTH_UL, help="Max crawl depth for other *.uni-lj.si")
    parser.add_argument("--depth-v", "--depth-other", dest="depth_v", type=int, default=_DEFAULT_DEPTH_V, help="Max crawl depth for non-uni-lj.si domains")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config.apply_run(args.run, mode=args.mode)

    depths = {"fri": int(args.depth_fri), "ul": int(args.depth_ul), "v": int(args.depth_v)}

    seeds = _snapshot_inputs(seeds_file=config.DATA_LINKS_FILE, run_name=args.run, mode=args.mode, depths=depths)
    if not seeds:
        log.error("No seed URLs found in %s", config.DATA_LINKS_FILE)
        sys.exit(1)

    crawler = Crawler(
        raw_dir=config.RAW_DIR,
        manifest=ManifestWriter(config.RAW_MANIFEST_JSONL),
        depth_fri=int(args.depth_fri),
        depth_ul=int(args.depth_ul),
        depth_v=int(args.depth_v),
        delay_seconds=float(config.CRAWL_DELAY_SECONDS),
    )

    crawler.crawl(seeds)


if __name__ == "__main__":
    main()
