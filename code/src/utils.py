import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from langdetect import DetectorFactory, detect, LangDetectException

DetectorFactory.seed = 0

log = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_language(text: str, default: str = "sl") -> str:
    words = text.strip().split()
    if len(words) < 5:
        return default
    try:
        lang = detect(text)
        if lang in ("sl", "en"):
            return lang
        return default
    except LangDetectException:
        return default


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_noise(text: str, min_words: int = 10) -> bool:
    words = text.split()
    if len(words) < min_words:
        return True
    printable_ratio = sum(1 for c in text if c.isprintable()) / max(len(text), 1)
    if printable_ratio < 0.85:
        return True
    return False


def make_doc_id(source_path: str) -> str:
    return hashlib.md5(source_path.encode()).hexdigest()[:12]


def normalize_url(url: str) -> str:
    """Normalize URLs so IDs are stable across small variations."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()

    normalized = parsed._replace(scheme=scheme, netloc=netloc, fragment="")
    if normalized.path.endswith("/") and normalized.path != "/":
        normalized = normalized._replace(path=normalized.path.rstrip("/"))
    return urlunparse(normalized)


def make_stable_doc_id(*, url: str = "", sha256: str = "", fallback: str = "") -> str:
    """Prefer a URL-based ID, otherwise fall back to content hash / path.

    Trying to keep stable across machines (Windows Linux paths) and redownloads.
    """
    url_norm = normalize_url(url)
    if url_norm:
        return hashlib.md5(url_norm.encode("utf-8")).hexdigest()[:12]
    if sha256:
        return hashlib.md5(sha256.encode("utf-8")).hexdigest()[:12]
    if fallback:
        return make_doc_id(fallback)
    return hashlib.md5(b"unknown").hexdigest()[:12]


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}_chunk{index:04d}"


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
