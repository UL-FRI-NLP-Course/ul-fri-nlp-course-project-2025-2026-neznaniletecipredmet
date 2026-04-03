"""
Crawl URLs from raw_dataset/data_links.txt, save HTML pages and download
linked PDFs into code/data/raw/.
"""

import logging
import re
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HTML_DIR = config.RAW_DIR / "html"
PDF_DIR = config.RAW_DIR / "pdfs"

HEADERS = {
    "User-Agent": "FRI-NLP-RAG-Crawler/1.0 (university research project)"
}

_robot_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def can_fetch(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robot_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            log.warning("Could not read robots.txt for %s — proceeding anyway", base)
        _robot_cache[base] = rp
    return _robot_cache[base].can_fetch(HEADERS["User-Agent"], url)


def detect_language(url: str, soup: BeautifulSoup) -> str:
    if "/en/" in url or url.endswith("/en"):
        return "en"
    if "/sl/" in url or url.endswith("/sl"):
        return "sl"
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        lang = str(html_tag["lang"]).lower()
        if lang.startswith("en"):
            return "en"
        if lang.startswith("sl"):
            return "sl"
    return "sl"


def url_to_filename(url: str, suffix: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "_") or "index"
    slug = re.sub(r"[^\w\-]", "_", slug)
    return f"{parsed.netloc}__{slug}{suffix}"


def fetch(url: str) -> requests.Response | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response
    except Exception as e:
        log.error("Failed to fetch %s: %s", url, e)
        return None


def save_html(url: str, soup: BeautifulSoup, language: str) -> Path:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    filename = url_to_filename(url, ".html")
    path = HTML_DIR / filename
    content = str(soup)
    path.write_text(content, encoding="utf-8")
    log.info("Saved HTML [%s]: %s", language, path.name)
    return path


def download_pdf(url: str) -> Path | None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    filename = url_to_filename(url, ".pdf")
    path = PDF_DIR / filename
    if path.exists():
        log.info("PDF already exists, skipping: %s", path.name)
        return path
    if not can_fetch(url):
        log.warning("robots.txt disallows: %s", url)
        return None
    response = fetch(url)
    if response is None:
        return None
    path.write_bytes(response.content)
    log.info("Downloaded PDF: %s", path.name)
    return path


def collect_pdf_links(base_url: str, soup: BeautifulSoup) -> list[str]:
    pdf_urls = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        full_url = urljoin(base_url, href)
        if full_url.lower().endswith(".pdf"):
            pdf_urls.append(full_url)
    return pdf_urls


def process_url(url: str) -> None:
    url = url.strip()
    if not url or url.startswith("#"):
        return

    if not can_fetch(url):
        log.warning("robots.txt disallows: %s", url)
        return

    log.info("Processing: %s", url)
    response = fetch(url)
    if response is None:
        return

    soup = BeautifulSoup(response.text, "html.parser")
    language = detect_language(url, soup)
    save_html(url, soup, language)

    pdf_links = collect_pdf_links(url, soup)
    log.info("Found %d PDF link(s) on %s", len(pdf_links), url)
    for pdf_url in pdf_links:
        download_pdf(pdf_url)
        time.sleep(config.CRAWL_DELAY_SECONDS)

    time.sleep(config.CRAWL_DELAY_SECONDS)


def load_urls(path: Path) -> list[str]:
    if not path.exists():
        log.error("Data links file not found: %s", path)
        sys.exit(1)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main() -> None:
    urls = load_urls(config.DATA_LINKS_FILE)
    log.info("Loaded %d URLs from %s", len(urls), config.DATA_LINKS_FILE)

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    for url in urls:
        process_url(url)

    html_count = len(list(HTML_DIR.glob("*.html")))
    pdf_count = len(list(PDF_DIR.glob("*.pdf")))
    log.info("Done. Saved %d HTML files and %d PDF files.", html_count, pdf_count)


if __name__ == "__main__":
    main()
